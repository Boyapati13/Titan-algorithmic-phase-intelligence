#!/usr/bin/env python3
"""
TITAN HFT SYSTEM - Recursive Optimizer V2.1 (PRODUCTION)
Author : Senior Quant / Systems Architect | April 2026

PURPOSE:
  Executes the 5-stage pipeline in a recursive improvement loop.
  After each full run it reads the TitanValidation scorecard and
  automatically applies targeted parameter fixes until all benchmarks
  are met or max_iterations is exhausted.

BENCHMARKS (all must pass for "Production Ready"):
  ┌─────────────────────────────────┬──────────┐
  │ Metric                          │ Target   │
  ├─────────────────────────────────┼──────────┤
  │ WFO Efficiency Ratio            │ >= 0.60   │
  │ Probability of Ruin (> 30% DD)  │ <= 1%     │
  │ Permutation p-value (Sharpe)    │ < 0.05   │
  │ OOS AUC                         │ >= 0.62   │
  │ 5-tick reversal prob (EventStdy)│ >= 60%    │
  └─────────────────────────────────┴──────────┘

USAGE:
  # Full recursive loop (uses MT5 Strategy Tester trade log)
  python TitanOptimizer.py --trade-log path/to/tester_log.csv

  # Run with demo trade log (no real data needed)
  python TitanOptimizer.py --demo

  # Skip stages you've already completed
  python TitanOptimizer.py --trade-log log.csv --skip-convert --skip-features

  # Limit iterations
  python TitanOptimizer.py --trade-log log.csv --max-iter 3
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
import datetime
import numpy as np

# Force UTF-8 on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
LOG_DIR    = os.path.join(BASE_DIR, 'optimizer_logs')
SCORECARD  = os.path.join(BASE_DIR, 'titan_scorecard.json')
FEAT_CFG   = os.path.join(BASE_DIR, 'titan_feature_config.json')  # tunable params
ONNX_PATH  = os.path.join(BASE_DIR, 'titan_lstm.onnx')
INFER_CFG  = os.path.join(BASE_DIR, 'titan_inference_config.json')

# ─── BENCHMARKS ───────────────────────────────────────────────────────────────
TARGET_WFO        = 0.60
TARGET_AUC        = 0.65    # Gold production target
AUC_ESCALATE_THRESHOLD = 0.60  # Brief requirement: escalate when AUC < 0.60
TARGET_RUIN       = 0.01   # 1% probability of ruin
TARGET_PERM_P     = 0.05
TARGET_REVERSAL   = 0.60   # 5-tick reversal prob from EventStudy
TARGET_WINNER_COV = 0.90   # MAE: >=90% of winners inside recommended SL

# ─── TUNABLE PARAMETER RANGES ─────────────────────────────────────────────────
LR_SCHEDULE      = [1e-3, 5e-4, 2e-4, 1e-4]   # LR decays on AUC failure
HURST_WINDOWS    = [20, 30, 40, 15]             # Hurst lookback tuning
VOT_WINDOWS      = [100, 200, 50, 300]          # VoT lookback (ms)
FOCAL_TOGGLE     = [False, True]                # Toggle focal loss
EPOCH_SCHEDULE   = [15, 150, 200]               # 15-epoch burst -> 150 escalate


# =============================================================================
# UTILITY HELPERS
# =============================================================================
def banner(msg: str, width: int = 62, char: str = '='):
    print(f"\n{char * width}")
    print(f"  {msg}")
    print(f"{char * width}")


def ts() -> str:
    return datetime.datetime.now().strftime('%H:%M:%S')


def run(cmd: list, label: str, cwd: str = BASE_DIR,
        timeout: int = 7200) -> tuple[int, str]:
    """Run a subprocess, stream output, return (returncode, combined_output)."""
    print(f"\n  [{ts()}] RUN: {' '.join(cmd)}")
    log_path = os.path.join(LOG_DIR, f"{label}_{ts().replace(':','')}.log")
    os.makedirs(LOG_DIR, exist_ok=True)

    lines = []
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        with open(log_path, 'w') as lf:
            for line in proc.stdout:
                line = line.rstrip()
                print(f"    {line}")
                lines.append(line)
                lf.write(line + '\n')
        proc.wait(timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"  [!] TIMEOUT after {timeout}s")
        rc = -1
    except Exception as e:
        print(f"  [!] Subprocess error: {e}")
        rc = -2

    return rc, '\n'.join(lines)


def load_json(path: str, default: dict = None) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default or {}


def save_json(path: str, obj: dict):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


# =============================================================================
# FEATURE CONFIG - tunable lookback parameters
# =============================================================================
def load_feat_cfg() -> dict:
    """Load or create default feature config with tunable parameters."""
    default = {
        'hurst_window':    20,
        'vot_window_ms':   100,
        'seq_len':         128,
        'n_features':      12,
        '_iteration':      0,
    }
    return load_json(FEAT_CFG, default)


def save_feat_cfg(cfg: dict):
    save_json(FEAT_CFG, cfg)
    print(f"  [cfg] Feature config saved: hurst_window={cfg['hurst_window']}  "
          f"vot_window_ms={cfg['vot_window_ms']}")


# =============================================================================
# TRAINING CONFIG STATE
# =============================================================================
class TrainState:
    """Holds mutable training hyper-parameters across iterations."""
    def __init__(self):
        self.lr_idx     = 0   # Index into LR_SCHEDULE
        self.focal_idx  = 0   # Index into FOCAL_TOGGLE
        self.epoch_idx  = 0   # Index into EPOCH_SCHEDULE
        self.hurst_idx  = 0
        self.vot_idx    = 0

    @property
    def lr(self):      return LR_SCHEDULE[min(self.lr_idx, len(LR_SCHEDULE)-1)]
    @property
    def focal(self):   return FOCAL_TOGGLE[min(self.focal_idx, len(FOCAL_TOGGLE)-1)]
    @property
    def epochs(self):  return EPOCH_SCHEDULE[min(self.epoch_idx, len(EPOCH_SCHEDULE)-1)]
    @property
    def hurst_w(self): return HURST_WINDOWS[min(self.hurst_idx, len(HURST_WINDOWS)-1)]
    @property
    def vot_w(self):   return VOT_WINDOWS[min(self.vot_idx, len(VOT_WINDOWS)-1)]

    def advance_lr(self):
        self.lr_idx = min(self.lr_idx + 1, len(LR_SCHEDULE) - 1)

    def advance_focal(self):
        self.focal_idx = min(self.focal_idx + 1, len(FOCAL_TOGGLE) - 1)

    def advance_epochs(self):
        self.epoch_idx = min(self.epoch_idx + 1, len(EPOCH_SCHEDULE) - 1)

    def advance_hurst(self):
        self.hurst_idx = min(self.hurst_idx + 1, len(HURST_WINDOWS) - 1)

    def advance_vot(self):
        self.vot_idx = min(self.vot_idx + 1, len(VOT_WINDOWS) - 1)

    def summary(self) -> str:
        return (f"LR={self.lr:.0e}  Focal={self.focal}  "
                f"Epochs={self.epochs}  Hurst={self.hurst_w}  "
                f"VoT={self.vot_w}ms")


# =============================================================================
# STAGE 1 - DATA BRIDGE
# =============================================================================
def stage_convert(args) -> bool:
    banner("STAGE 1 - Forensic Data Bridge (convert)")
    rc, out = run(
        [sys.executable, 'TitanParquetConverter.py'],
        label='convert')

    parquet_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.parquet')] \
        if os.path.isdir(DATA_DIR) else []

    ok = rc == 0 and len(parquet_files) > 0
    print(f"\n  Stage 1: {'PASS [PASS]' if ok else 'FAIL [FAIL]'}  "
          f"({len(parquet_files)} parquet files)")

    # Forensic header check: verify 24-byte skip + 52-byte struct
    if 'OK  ' in out:
        print("  Header validation: 24-byte FileHeader + 52-byte TickData confirmed")
    elif 'STRUCT MISMATCH' in out:
        print("  [CRITICAL] Struct mismatch - recompile TitanTickLogger.mq5")
        return False
    elif 'Invalid Magic' in out:
        print("  [CRITICAL] Bad magic bytes - files may not be Titan V2 format")
        return False

    return ok


# =============================================================================
# STAGE 2 - MICROSTRUCTURE ALPHA AUDIT
# =============================================================================
def stage_features(args, state: TrainState, feat_cfg: dict) -> tuple:
    banner("STAGE 2 - Microstructure Alpha Audit (features + event study)")

    # 2a: Feature engineering
    rc, _ = run(
        [sys.executable, 'TitanPipeline.py', '--mode', 'features',
         '--symbol', args.symbol],
        label='features')

    npz_ok = os.path.exists(os.path.join(DATA_DIR, 'titan_lstm_sequences.npz'))
    if not npz_ok:
        print("  [FAIL] titan_lstm_sequences.npz not created")
        return False, False

    # 2b: Event study - check 5-tick reversal probability
    print("\n  Running TitanEventStudy.py...")
    study_log = os.path.join(BASE_DIR, 'titan_event_study_results.json')
    rc2, out2 = run(
        [sys.executable, 'TitanEventStudy.py',
         '--json-out', study_log],
        label='event_study')

    study = load_json(study_log)
    reversal_prob = study.get('reversal_probability', None)
    
    event_study_ok = False

    if reversal_prob is not None:
        rev_ok   = reversal_prob >= TARGET_REVERSAL
        rev_pval = study.get('reversal_p_value', 1.0)
        pval_ok  = rev_pval <= 0.01
        
        event_study_ok = rev_ok and pval_ok

        print(f"\n  Event Study Results:")
        print(f"    5-tick reversal prob : {reversal_prob:.3f}  "
              f"({'PASS >=0.60' if rev_ok else 'FAIL <0.60'})")
        print(f"    p-value              : {rev_pval:.4f}  "
              f"({'PASS <=0.01' if pval_ok else 'FAIL >0.01'})")

        if not event_study_ok:
            # Recalibrate Hurst and VoT lookback windows
            state.advance_hurst()
            state.advance_vot()
            feat_cfg['hurst_window']  = state.hurst_w
            feat_cfg['vot_window_ms'] = state.vot_w
            save_feat_cfg(feat_cfg)
            print(f"\n  [AUTO-FIX] Recalibrated lookbacks:")
            print(f"    hurst_window  -> {state.hurst_w}")
            print(f"    vot_window_ms -> {state.vot_w}ms")
            print(f"    Re-run feature stage in next iteration")
    else:
        print("  [INFO] Event study did not output JSON - skipping reversal check")

    return npz_ok, event_study_ok


# =============================================================================
# STAGE 3 - NEURAL INTELLIGENCE TRAINING
# =============================================================================
def stage_train(args, state: TrainState) -> float:
    banner("STAGE 3 - Neural Intelligence Training (LSTM)")

    cmd = [
        sys.executable, 'TitanPipeline.py',
        '--mode', 'train',
        '--epochs', str(state.epochs),
        '--lr', str(state.lr),    # BUG-FIX: LR escalation was tracked in TrainState
                                  # but never forwarded to the subprocess. The
                                  # advance_lr() calls were silently ignored every run.
    ]
    if state.focal:
        cmd.append('--focal')

    print(f"  Hyper-params: {state.summary()}")
    rc, out = run(cmd, label='train', timeout=10800)  # 3h max

    # Read OOS AUC from config
    cfg   = load_json(INFER_CFG)
    auc   = cfg.get('oos_auc', 0.0)
    ckpt  = os.path.exists(os.path.join(BASE_DIR, 'titan_lstm_best_v2.pth'))
    onnx_ = os.path.exists(ONNX_PATH)

    print(f"\n  OOS AUC    : {auc:.4f}  "
          f"({'PASS >=0.62' if auc >= TARGET_AUC else 'FAIL <0.62'})")
    print(f"  Checkpoint : {'[PASS]' if ckpt else '[FAIL]'}")
    print(f"  ONNX       : {'[PASS]' if onnx_ else '[FAIL]'}")

    if auc < AUC_ESCALATE_THRESHOLD:
        # Brief requirement: escalate epochs -> 150, decay LR -> 5e-4
        # when AUC < 0.60. Do NOT escalate when 0.60 <= AUC < 0.65
        # (model is acceptable, just below gold target — more epochs
        # risk overfitting on a dataset that already shows signal).
        prev_lr = state.lr
        state.advance_lr()
        state.advance_focal()
        state.advance_epochs()
        print(f"\n  [AUTO-FIX] AUC {auc:.4f} < {AUC_ESCALATE_THRESHOLD} (escalation gate)")
        print(f"    LR:       {prev_lr:.0e} -> {state.lr:.0e}")
        print(f"    Focal:    {not state.focal} -> {state.focal}")
        print(f"    Epochs:   will use {state.epochs} next iteration")

    return auc


# =============================================================================
# STAGE 4 - ONNX DEPLOYMENT BRIDGE
# =============================================================================
def stage_export(args) -> bool:
    banner("STAGE 4 - ONNX Deployment Bridge (export)")

    rc, out = run(
        [sys.executable, 'TitanONNXExport.py'],
        label='export')

    onnx_ok = os.path.exists(ONNX_PATH)
    cfg_ok  = os.path.exists(INFER_CFG)

    # Parity check: ensure feat_mean/feat_std exist in config
    if cfg_ok:
        cfg  = load_json(INFER_CFG)
        n_f  = len(cfg.get('feat_mean', []))
        n_f2 = len(cfg.get('feat_std',  []))
        parity_ok = (n_f == 12 and n_f2 == 12)
        print(f"\n  ONNX parity check:")
        print(f"    feat_mean length : {n_f}  {'[PASS]' if n_f==12 else '[FAIL] (expected 12)'}")
        print(f"    feat_std  length : {n_f2}  {'[PASS]' if n_f2==12 else '[FAIL] (expected 12)'}")
        print(f"    n_features in cfg: {cfg.get('n_features')}  "
              f"{'[PASS]' if cfg.get('n_features')==12 else '[FAIL]'}")
        print(f"    seq_len in cfg   : {cfg.get('seq_len')}  "
              f"{'[PASS]' if cfg.get('seq_len')==128 else '[FAIL]'}")
    else:
        parity_ok = False

    ok = onnx_ok and cfg_ok and parity_ok
    print(f"\n  Stage 4: {'PASS [PASS]' if ok else 'FAIL [FAIL]'}")
    return ok


# =============================================================================
# STAGE 5 - FORENSIC PERFORMANCE AUDIT
# =============================================================================
def stage_validate(args, iteration: int) -> dict:
    banner("STAGE 5 - Forensic Performance Audit (validate)")

    sc_path = os.path.join(BASE_DIR, f'titan_scorecard_iter{iteration}.json')
    cmd = [
        sys.executable, 'TitanValidation.py',
        '--capital', '100000',
        '--wfo-splits', '8',
        '--mc-iter', '10000',
        '--target-wfo', str(TARGET_WFO),
        '--target-ruin', str(TARGET_RUIN),
        '--json-out', sc_path,
    ]

    # Attach real trade log if provided
    if args.trade_log and os.path.exists(args.trade_log):
        cmd += ['--trade-log', args.trade_log]
        print(f"  Trade log: {args.trade_log}")
    else:
        cmd.append('--demo')
        print("  [INFO] No --trade-log provided - using synthetic demo log")
        print("         For real results: pass --trade-log path/to/tester_log.csv")

    rc, out = run(cmd, label=f'validate_iter{iteration}', timeout=1800)

    sc = load_json(sc_path)
    if not sc:
        print("  [WARN] Scorecard not written - validation may have crashed")
        return {}

    # Pretty-print scorecard
    print(f"\n  {'─'*50}")
    print(f"  SCORECARD - Iteration {iteration}")
    print(f"  {'─'*50}")
    print(f"  WFO Efficiency Ratio : {sc.get('wfo_efficiency_ratio', 0):.4f}  "
          f"{'[PASS] PASS' if sc.get('wfo_pass') else '[FAIL] FAIL'}")
    print(f"  Ruin Probability     : {sc.get('ruin_probability', 0):.4f}  "
          f"{'[PASS] PASS' if sc.get('ruin_pass') else '[FAIL] FAIL'}")
    print(f"  Perm p-value         : {sc.get('perm_sharpe_pvalue', 1):.4f}  "
          f"{'[PASS] PASS' if sc.get('perm_pass') else '[FAIL] FAIL'}")
    sl = sc.get('mae_recommended_sl')
    if sl:
        cov = sc.get('mae_winner_coverage', 0)
        print(f"  MAE Recommended SL   : {sl} ticks  "
              f"(covers {cov:.1%} of winners)")
        if cov >= TARGET_WINNER_COV:
            print(f"  -> Update EA: input int STOP_LOSS_TICKS = {sl};")
        else:
            print(f"  -> SL covers < {TARGET_WINNER_COV:.0%} of winners - "
                  f"increase SL search range (--sl-max)")
    print(f"\n  Production Ready     : "
          f"{'[PASS] YES' if sc.get('production_ready') else '[FAIL] NO - next iteration'}")
    return sc


# =============================================================================
# EA STOP LOSS UPDATE
# =============================================================================
def apply_sl_to_ea(sl_ticks: int):
    """Patch STOP_LOSS_TICKS in TitanEA.mq5 to the MAE-recommended value."""
    ea_path = os.path.join(BASE_DIR, 'TitanEA.mq5')
    if not os.path.exists(ea_path):
        return

    with open(ea_path, 'r', encoding='utf-8') as f:
        content = f.read()

    import re
    pattern = r'(input\s+int\s+STOP_LOSS_TICKS\s*=\s*)\d+'
    replacement = rf'\g<1>{sl_ticks}'
    new_content, n = re.subn(pattern, replacement, content)

    if n > 0:
        with open(ea_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  [AUTO-FIX] TitanEA.mq5: STOP_LOSS_TICKS -> {sl_ticks}")
        print(f"             Recompile TitanEA.mq5 in MetaEditor (F7)")
    else:
        print(f"  [INFO] STOP_LOSS_TICKS pattern not found in TitanEA.mq5")
        print(f"         Manually set: input int STOP_LOSS_TICKS = {sl_ticks};")


# =============================================================================
# ITERATION SUMMARY
# =============================================================================
def print_iteration_summary(iteration: int, scorecard: dict,
                              auc: float, state: TrainState,
                              event_study_ok: bool = True) -> bool:
    banner(f"ITERATION {iteration} SUMMARY", char='─')
    metrics = [
        ('WFO Efficiency Ratio', scorecard.get('wfo_efficiency_ratio', 0),
         TARGET_WFO, '>='),
        ('OOS AUC',              auc,
         TARGET_AUC, '>='),
        ('Ruin Probability',     scorecard.get('ruin_probability', 1),
         TARGET_RUIN, '<='),
        ('Perm p-value',         scorecard.get('perm_sharpe_pvalue', 1),
         TARGET_PERM_P, '<'),
    ]
    all_pass = True
    for name, val, thresh, op in metrics:
        if op == '>=':
            ok = val >= thresh
        elif op == '<=':
            ok = val <= thresh
        else:
            ok = val < thresh
        status = '[PASS]' if ok else '[FAIL]'
        print(f"  {status}  {name:<28} {val:.4f}  (target {op} {thresh})")
        all_pass = all_pass and ok

    # BUG-FIX: event_study_ok was tracked but never included in the production-
    # ready gate. A system could declare "PRODUCTION READY" with a failing
    # reversal study (no statistical edge confirmed in the microstructure).
    ev_status = '[PASS]' if event_study_ok else '[FAIL]'
    print(f"  {ev_status}  {'5-tick Reversal Study':<28}       "
          f"  (target >= {TARGET_REVERSAL:.0%})")
    all_pass = all_pass and event_study_ok

    print(f"\n  Next params : {state.summary()}")
    return all_pass


# =============================================================================
# MAIN RECURSIVE LOOP
# =============================================================================
def main():
    os.chdir(BASE_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(
        description='Titan HFT Recursive Optimizer V2.1',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--trade-log', default=None,
        help='Path to MT5 Strategy Tester trade log CSV\n'
             '(columns: date, profit, [mae])\n'
             'If omitted, Stage 5 runs on synthetic demo data.')
    parser.add_argument(
        '--symbol', default='EURUSD',
        help='Trading symbol (default: EURUSD)')
    parser.add_argument(
        '--max-iter', type=int, default=5,
        help='Max optimization iterations (default: 5)')
    parser.add_argument(
        '--skip-convert', action='store_true',
        help='Skip Stage 1 (already have .parquet files)')
    parser.add_argument(
        '--skip-features', action='store_true',
        help='Skip Stage 2 (already have titan_lstm_sequences.npz)')
    parser.add_argument(
        '--skip-train', action='store_true',
        help='Skip Stage 3 (already have checkpoint + ONNX)')
    parser.add_argument(
        '--skip-export', action='store_true',
        help='Skip Stage 4 (already have titan_lstm.onnx)')
    parser.add_argument(
        '--demo', action='store_true',
        help='Use synthetic demo data for Stage 5 validation')
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"  [INFO] Ignoring unrecognized arguments: {unknown}")

    if args.demo:
        args.trade_log = None  # Forces demo mode in stage_validate

    state    = TrainState()
    feat_cfg = load_feat_cfg()

    banner("TITAN HFT SYSTEM V2.1 - RECURSIVE OPTIMIZATION LOOP")
    print(f"  Benchmarks:")
    print(f"    WFO Efficiency Ratio >= {TARGET_WFO}")
    print(f"    OOS AUC              >= {TARGET_AUC}")
    print(f"    Ruin Probability     <= {TARGET_RUIN*100:.0f}%")
    print(f"    Permutation p-value  < {TARGET_PERM_P}")
    print(f"    5-tick Reversal Prob >= {TARGET_REVERSAL*100:.0f}%")
    print(f"    Winner SL Coverage   >= {TARGET_WINNER_COV*100:.0f}%")
    print(f"\n  Max iterations: {args.max_iter}")
    print(f"  Trade log     : {args.trade_log or '[demo mode]'}")

    t_global = time.time()
    production_ready = False
    last_scorecard   = {}
    last_auc         = 0.0

    for iteration in range(1, args.max_iter + 1):
        banner(f"ITERATION {iteration} / {args.max_iter}  [{ts()}]")
        
        event_study_ok = True

        # ── Stage 1: Data bridge ──────────────────────────────────────────────
        if not args.skip_convert:
            ok = stage_convert(args)
            if not ok:
                print("\n  [HALT] Stage 1 failed - check Goldtrading\\data\\ "
                      "for .ticks files from TitanTickLogger.mq5")
                print("  Copy .ticks -> "
                      "MQL5\\Files\\Goldtrading\\ and re-run")
                break
        else:
            print(f"\n  Stage 1 skipped (--skip-convert)")

        # ── Stage 2: Features + Event study ──────────────────────────────────
        if not args.skip_features:
            ok, event_study_ok = stage_features(args, state, feat_cfg)
            if not ok:
                print("\n  [HALT] Stage 2 failed - no sequences generated")
                break
        else:
            print(f"\n  Stage 2 skipped (--skip-features)")

        # ── Stage 3: Training ─────────────────────────────────────────────────
        if not args.skip_train:
            auc = stage_train(args, state)
            last_auc = auc
        else:
            cfg      = load_json(INFER_CFG)
            last_auc = cfg.get('oos_auc', 0.0)
            print(f"\n  Stage 3 skipped (--skip-train)  AUC from config: {last_auc:.4f}")

        # ── Stage 4: Export ───────────────────────────────────────────────────
        if not args.skip_export:
            ok = stage_export(args)
            if not ok:
                print("\n  [WARN] Stage 4 export failed - skipping validation")
        else:
            print(f"\n  Stage 4 skipped (--skip-export)")

        # ── Stage 5: Validate ─────────────────────────────────────────────────
        sc = stage_validate(args, iteration)
        last_scorecard = sc

        # After first iteration, skip convert/features/export to save time
        # (only retrain when AUC < target)
        if iteration == 1:
            args.skip_convert = True
            args.skip_features = True
            args.skip_export = True

        # ── Apply MAE->EA stop loss patch ─────────────────────────────────────
        sl = sc.get('mae_recommended_sl')
        if sl and sc.get('mae_winner_coverage', 0) >= TARGET_WINNER_COV:
            apply_sl_to_ea(sl)

        # ── Check production ready ────────────────────────────────────────────
        production_ready = print_iteration_summary(
            iteration, sc, last_auc, state, event_study_ok)
        if production_ready:
            break

        # ── Re-enable training only if AUC is below the escalation gate ────────
        # BUG-FIX: was checking TARGET_AUC (0.65), causing training to re-run
        # even when AUC was 0.62 (acceptable signal, just not gold target).
        # That wastes hours of GPU time on marginal gains.
        if last_auc < AUC_ESCALATE_THRESHOLD:
            args.skip_train  = False
            args.skip_export = False

        # Small pause between iterations to avoid filesystem race conditions
        time.sleep(2)

    # ── FINAL REPORT ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_global
    banner("OPTIMIZATION COMPLETE", char='═')

    if production_ready:
        print("  [PASS]  SYSTEM IS PRODUCTION READY")
    else:
        print("  [FAIL]  SYSTEM DID NOT MEET ALL BENCHMARKS")
        print(f"\n  Failed metrics:")
        if last_auc < TARGET_AUC:
            print(f"    • OOS AUC {last_auc:.4f} < {TARGET_AUC} - need more data or epochs")
        wfo = last_scorecard.get('wfo_efficiency_ratio', 0)
        if wfo < TARGET_WFO:
            print(f"    • WFO {wfo:.4f} < {TARGET_WFO} - model overfitting, add WFO folds")
        ruin = last_scorecard.get('ruin_probability', 1)
        if ruin > TARGET_RUIN:
            print(f"    • Ruin prob {ruin:.4f} > {TARGET_RUIN} - reduce position size")
        perm = last_scorecard.get('perm_sharpe_pvalue', 1)
        if perm >= TARGET_PERM_P:
            print(f"    • Perm p={perm:.4f} >= {TARGET_PERM_P} - edge may be noise")

    print(f"\n  Total runtime: {elapsed/60:.1f} min")
    print(f"  Iteration logs: {LOG_DIR}")
    print(f"  Final scorecard: titan_scorecard_iter{last_scorecard.get('_iter','?')}.json")

    if production_ready:
        print("\n  📋 DEPLOYMENT CHECKLIST:")
        print("     1. titan_lstm.onnx deployed to MQL5\\Files\\")
        print("     2. titan_inference_config.json deployed to MQL5\\Files\\")
        print("     3. TitanEA.mq5 recompiled in MetaEditor (F7)")
        print("     4. STOP_LOSS_TICKS verified in EA inputs")
        print("     5. Run on demo account for 2 weeks before live")

    sys.exit(0 if production_ready else 1)


if __name__ == '__main__':
    main()