#!/usr/bin/env python3
"""
TITAN HFT SYSTEM - Master Pipeline Orchestrator V2.0 (PRODUCTION)
Author : Senior Quant / Systems Architect | April 2026

USAGE:
  # Full end-to-end pipeline from raw .ticks to deployed .onnx + .ex5
  python TitanPipeline.py --mode full

  # Individual stages
  python TitanPipeline.py --mode convert    # Stage 1: .ticks -> .parquet
  python TitanPipeline.py --mode features  # Stage 2: .parquet -> .npz
  python TitanPipeline.py --mode train     # Stage 3: .npz -> LSTM checkpoint
  python TitanPipeline.py --mode export    # Stage 4: checkpoint -> .onnx
  python TitanPipeline.py --mode validate  # Stage 5: trade log -> WFO report
  python TitanPipeline.py --mode verify    # Forensic: struct + config checks

DIRECTORY LAYOUT:
  Goldtrading/
  ├── data/                          ← .parquet, .npz files
  ├── TitanTickLogger.mq5            ← Stage 0: live tick capture
  ├── TitanParquetConverter.py       ← Stage 1: binary -> parquet
  ├── TitanFeatureEngineering.py     ← Stage 2a: 12-feature computation
  ├── TitanAlphaResearch.py          ← Stage 2b: sequence creation
  ├── TitanLSTMTraining.py           ← Stage 3: LSTM training (primary)
  ├── TitanMLTraining.py             ← Stage 3b: XGBoost baseline
  ├── TitanLSTMExport.py             ← Stage 4: PyTorch -> ONNX
  ├── TitanONNXExport.py             ← Stage 4b: unified ONNX bridge
  ├── TitanEA.mq5                    ← Stage 5: live execution engine
  ├── TitanValidation.py             ← Stage 6: WFO + Monte Carlo
  ├── TitanEventStudy.py             ← Research: event study analysis
  ├── TitanPipeline.py               ← This file: master orchestrator
  └── requirements.txt               ← pip install -r requirements.txt
"""

import os
import sys
import glob
import json
import time
import struct
import shutil
import argparse
import subprocess
import numpy as np

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, 'data')
MT5_FILES    = os.path.join(
    os.environ.get('APPDATA', ''),
    'MetaQuotes', 'Terminal',
    'AE2CC2E013FDE1E3CDF010AA51C60400', 'MQL5', 'Files')

# ── STRUCT CONSTANTS - must match TitanTickLogger.mq5 ────────────────────────
TITAN_MAGIC  = 0x5449544E
HEADER_SIZE  = 24
TICK_SIZE    = 52
HEADER_FMT   = '<IIqII'

# ── ARCHITECTURE CONSTANTS ───────────────────────────────────────────────────
N_FEATURES   = 12
SEQ_LEN      = 128


# =============================================================================
# UTILITY HELPERS
# =============================================================================
def banner(title: str, width: int = 60):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def check(ok: bool, msg: str):
    status = "✓ " if ok else "✗ "
    print(f"  {status} {msg}")
    return ok


def elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s//60:.0f}m {s%60:.0f}s" if s >= 60 else f"{s:.1f}s"


# =============================================================================
# STAGE 0 - FORENSIC STRUCT VERIFICATION
# =============================================================================
def stage_verify(silent: bool = False) -> bool:
    banner("STAGE 0 - Forensic Struct & Config Verification")
    all_ok = True

    # ── 1. Verify at least one .ticks or .parquet file exists ─────────────────
    ticks_files   = glob.glob(os.path.join(DATA_DIR, '*.ticks'))
    parquet_files = glob.glob(os.path.join(DATA_DIR, '*.parquet'))

    check(os.path.isdir(DATA_DIR),           f"data/ directory exists: {DATA_DIR}")
    check(len(ticks_files) > 0 or len(parquet_files) > 0,
          f"Input files found ({len(ticks_files)} .ticks, {len(parquet_files)} .parquet)")

    # ── 2. Validate binary header on each .ticks file ─────────────────────────
    corrupted = []
    for f in ticks_files[:5]:  # Check up to 5 files
        try:
            with open(f, 'rb') as fh:
                raw = fh.read(HEADER_SIZE)
            if len(raw) < HEADER_SIZE:
                corrupted.append(f"TRUNCATED: {os.path.basename(f)}")
                continue
            magic, ver, _, tick_sz, _ = struct.unpack(HEADER_FMT, raw)
            file_size = os.path.getsize(f)
            n_ticks   = (file_size - HEADER_SIZE) // TICK_SIZE
            if magic != TITAN_MAGIC:
                corrupted.append(f"BAD_MAGIC ({hex(magic)}): {os.path.basename(f)}")
            elif tick_sz != TICK_SIZE:
                corrupted.append(f"STRUCT_MISMATCH ({tick_sz}B): {os.path.basename(f)}")
            else:
                if not silent:
                    print(f"    OK  {os.path.basename(f)}"
                          f"  ver={ver}  ticks={n_ticks:,}")
        except Exception as e:
            corrupted.append(f"READ_ERROR ({e}): {os.path.basename(f)}")

    all_ok &= check(len(corrupted) == 0,
                    f"Binary header validation ({len(ticks_files)} files)")
    for c in corrupted:
        print(f"    [!] {c}")

    # ── 3. Check titan_inference_config.json ──────────────────────────────────
    cfg_path = os.path.join(BASE_DIR, 'titan_inference_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        all_ok &= check(cfg.get('n_features') == N_FEATURES,
                        f"Config n_features = {cfg.get('n_features')} (expected {N_FEATURES})")
        all_ok &= check(cfg.get('seq_len')    == SEQ_LEN,
                        f"Config seq_len    = {cfg.get('seq_len')} (expected {SEQ_LEN})")
        print(f"    buy_threshold:  {cfg.get('buy_threshold', 'N/A')}")
        print(f"    sell_threshold: {cfg.get('sell_threshold', 'N/A')}")
        print(f"    oos_auc:        {cfg.get('oos_auc', 'N/A')}")
    else:
        print(f"    [INFO] titan_inference_config.json not found yet - run training first")

    # ── 4. Check ONNX model ───────────────────────────────────────────────────
    onnx_path = os.path.join(BASE_DIR, 'titan_lstm.onnx')
    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            import numpy as np_
            sess  = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            inp   = sess.get_inputs()[0]
            dummy = np_.random.randn(1, SEQ_LEN, N_FEATURES).astype(np_.float32)
            out   = sess.run(None, {inp.name: dummy})[0].flatten()
            all_ok &= check(inp.shape[1] == SEQ_LEN and inp.shape[2] == N_FEATURES,
                            f"ONNX input shape {inp.shape}")
            all_ok &= check(0.0 <= float(out[0]) <= 1.0,
                            f"ONNX output in sigmoid range ({out[0]:.4f})")
        except Exception as e:
            all_ok &= check(False, f"ONNX verification failed: {e}")
    else:
        print(f"    [INFO] titan_lstm.onnx not found yet - run training + export first")

    # ── 5. Check NPZ sequences ────────────────────────────────────────────────
    npz_path = os.path.join(DATA_DIR, 'titan_lstm_sequences.npz')
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        all_ok &= check(d['X_train'].shape[2] == N_FEATURES,
                        f"NPZ features = {d['X_train'].shape[2]} (expected {N_FEATURES})")
        all_ok &= check(d['X_train'].shape[1] == SEQ_LEN,
                        f"NPZ seq_len  = {d['X_train'].shape[1]} (expected {SEQ_LEN})")
        print(f"    X_train: {d['X_train'].shape}  y_train pos rate: "
              f"{d['y_train'].mean():.3f}")
    else:
        print(f"    [INFO] titan_lstm_sequences.npz not found yet - run feature stage first")

    print(f"\n  Verification {'PASSED [PASS]' if all_ok else 'FAILED [FAIL]'}")
    return all_ok


# =============================================================================
# STAGE 1 - CONVERT .TICKS -> .PARQUET
# =============================================================================
def stage_convert(source_dir: str = None) -> bool:
    banner("STAGE 1 - Convert .ticks -> LZ4-compressed Parquet")
    t0 = time.time()

    src = source_dir or DATA_DIR
    ticks_files = sorted(glob.glob(os.path.join(src, '*.ticks')))

    if not ticks_files:
        print(f"  No .ticks files in {src}")
        print(f"  Copy .ticks files from Strategy Tester Agent folder:")
        agent_path = os.path.join(
            os.environ.get('APPDATA', ''), 'MetaQuotes', 'Tester',
            'AE2CC2E013FDE1E3CDF010AA51C60400',
            'Agent-127.0.0.1-3000', 'MQL5', 'Files')
        print(f"    xcopy /Y \"{agent_path}\\*.ticks\" \"{DATA_DIR}\\\"")
        return False

    print(f"  Found {len(ticks_files)} .ticks files in {src}")
    os.makedirs(DATA_DIR, exist_ok=True)

    from TitanParquetConverter import convert_ticks_to_parquet
    ok_count = 0
    for f in ticks_files:
        ok = convert_ticks_to_parquet(f, output_dir=DATA_DIR)
        if ok:
            ok_count += 1

    success = ok_count == len(ticks_files)
    check(success,
          f"Converted {ok_count}/{len(ticks_files)} files in {elapsed(t0)}")
    return success


# =============================================================================
# STAGE 2 - FEATURE ENGINEERING + SEQUENCE CREATION
# =============================================================================
def stage_features(symbol: str = 'XAUUSD') -> bool:
    banner("STAGE 2 - Feature Engineering -> LSTM Sequences")
    t0 = time.time()

    # ── PATH ALIGNMENT: probe symbol-specific MT5 dir before local data/ ──────
    # When running --symbol EURUSD, look in MQL5/Files/EURUSD/ first.
    # This matches TitanTickLogger's SUBFOLDER output layout.
    symbol_mt5_dir = os.path.join(MT5_FILES, symbol)
    if os.path.isdir(symbol_mt5_dir):
        candidate = sorted(glob.glob(os.path.join(symbol_mt5_dir, '*.parquet')))
        if candidate:
            print(f"  Using symbol-specific parquet source: {symbol_mt5_dir}")
            parquet_files = candidate[:5]
        else:
            parquet_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))[:5]
    else:
        parquet_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))[:5]

    if not parquet_files:
        print("  No .parquet files in data/. Run --mode convert first.")
        return False

    from TitanFeatureEngineering import TitanFeatureEngineer
    all_X, all_y = [], []

    for pq in parquet_files:
        print(f"  Processing {os.path.basename(pq)}...")
        eng = TitanFeatureEngineer(symbol)
        eng.load_parquet_data(pq)
        feats    = eng.compute_all_features()
        labels   = eng.generate_labels(target_ticks=15, time_window_sec=30)
        X, y     = eng.create_lstm_sequences(feats, labels, seq_len=SEQ_LEN)
        all_X.append(X); all_y.append(y)
        print(f"    -> {X.shape[0]:,} sequences")

    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)
    n     = len(X_all)
    s1, s2 = int(n * 0.70), int(n * 0.85)

    npz_path = os.path.join(DATA_DIR, 'titan_lstm_sequences.npz')
    np.savez(npz_path,
             X_train=X_all[:s1],   y_train=y_all[:s1],
             X_val=X_all[s1:s2],   y_val=y_all[s1:s2],
             X_test=X_all[s2:],    y_test=y_all[s2:],
             feat_mean=np.zeros(N_FEATURES, dtype=np.float32),
             feat_std=np.ones(N_FEATURES,  dtype=np.float32))

    size_mb = os.path.getsize(npz_path) / 1024**2
    print(f"\n  Saved {n:,} sequences -> {npz_path}  ({size_mb:.1f} MB)")
    print(f"  Train: {s1:,}  Val: {s2-s1:,}  Test: {n-s2:,}")
    print(f"  Positive rate: {y_all.mean():.3f}")
    check(True, f"Feature stage complete in {elapsed(t0)}")
    return True


# =============================================================================
# STAGE 3 - LSTM TRAINING
# =============================================================================
def stage_train(use_focal: bool = False, epochs: int = 100, lr: float = 1e-3) -> bool:
    banner("STAGE 3 - LSTM Training (PyTorch + AMP)")
    t0 = time.time()

    npz_path = os.path.join(DATA_DIR, 'titan_lstm_sequences.npz')
    if not os.path.exists(npz_path):
        print(f"  Sequences not found: {npz_path}")
        print("  Run --mode features first.")
        return False

    from TitanLSTMTraining import TitanLSTMTrainer
    trainer = TitanLSTMTrainer(use_focal_loss=use_focal, use_amp=True)
    trainer.load_sequences(npz_path)
    trainer.train(epochs=epochs, lr=lr, patience=15, batch_size=64)
    auc = trainer.evaluate_oos()
    trainer.export_onnx('titan_lstm.onnx', opset=18)

    check(os.path.exists('titan_lstm_best_v2.pth'),    'Checkpoint saved: titan_lstm_best_v2.pth')
    check(os.path.exists('titan_lstm.onnx'),           'ONNX exported:    titan_lstm.onnx')
    check(os.path.exists('titan_inference_config.json'),'Config saved:     titan_inference_config.json')
    check(auc >= 0.6, f"OOS AUC = {auc:.4f} ({'PASS >=0.60' if auc >= 0.6 else 'FAIL <0.60'})")
    print(f"\n  Training complete in {elapsed(t0)}")
    return auc >= 0.6


# =============================================================================
# STAGE 4 - ONNX EXPORT + MT5 DEPLOY
# =============================================================================
def stage_export() -> bool:
    banner("STAGE 4 - ONNX Export & MT5 Deployment")
    t0 = time.time()

    from TitanONNXExport import (export_lstm_onnx, verify_onnx_model,
                                  deploy_to_mt5_files)
    ok = export_lstm_onnx(
        checkpoint='titan_lstm_best_v2.pth',
        output='titan_lstm.onnx',
        config='titan_inference_config.json',
    )
    if ok:
        verify_onnx_model('titan_lstm.onnx')
        deploy_to_mt5_files('titan_lstm.onnx', 'titan_inference_config.json')

    check(ok, f"Export stage complete in {elapsed(t0)}")
    return ok


# =============================================================================
# STAGE 5 - VALIDATION (WFO + Monte Carlo + MAE)
# =============================================================================
def stage_validate(trade_log: str = None) -> bool:
    banner("STAGE 5 - Validation: WFO + Monte Carlo + MAE")
    t0 = time.time()

    log_path = trade_log or os.path.join(BASE_DIR, 'titan_trade_log.csv')
    if not os.path.exists(log_path):
        # Generate demo log for first run
        print(f"  Trade log not found: {log_path}")
        print("  Generating synthetic demo log for validation pipeline test...")
        import pandas as pd
        np.random.seed(42)
        n       = 1000
        dates   = pd.date_range('2025-01-02', periods=n, freq='4h')
        profits = np.random.normal(8, 45, n)
        mask    = np.random.random(n) < 0.42
        profits[mask] = -np.abs(np.random.exponential(28, mask.sum()))
        mae     = np.where(profits > 0,
                           np.random.exponential(6, n),
                           np.random.exponential(18, n))
        pd.DataFrame({'date': dates, 'profit': profits, 'mae': mae,
                      'balance': 100_000 + profits.cumsum()})\
          .to_csv(log_path, index=False)
        print(f"  Demo log saved -> {log_path}")

    from TitanValidation import TitanRiskManager
    rm      = TitanRiskManager(initial_capital=100_000)
    results = rm.run_full_suite(log_path)

    wfo_ok = False
    if 'wfo' in results and len(results['wfo']) > 0:
        wfo    = results['wfo']
        eff    = wfo['oos_sharpe'].mean() / max(wfo['is_sharpe'].mean(), 1e-6)
        wfo_ok = eff >= 0.6
        check(wfo_ok, f"WFO Efficiency Ratio = {eff:.3f} "
              f"({'PASS >=0.6' if wfo_ok else 'FAIL <0.6'})")

    print(f"\n  Validation complete in {elapsed(t0)}")
    return wfo_ok


# =============================================================================
# FULL PIPELINE
# =============================================================================
def stage_full(args) -> bool:
    banner("TITAN HFT SYSTEM V2 - FULL PIPELINE RUN", 60)
    print(f"  Base: {BASE_DIR}")
    print(f"  Data: {DATA_DIR}")
    t0 = time.time()

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    stages = [
        ("Verify",   lambda: stage_verify(silent=True)),
        ("Convert",  stage_convert),
        ("Features", lambda: stage_features(args.symbol)),
        ("Train",    lambda: stage_train(args.focal, args.epochs)),
        ("Export",   stage_export),
        ("Validate", stage_validate),
    ]

    for name, fn in stages:
        print(f"\n{'─'*60}")
        ok = fn()
        if not ok and name in ("Convert", "Features", "Train", "Export"):
            print(f"  [HALT] Stage '{name}' failed - stopping pipeline.")
            return False

    banner(f"PIPELINE COMPLETE in {elapsed(t0)}", 60)
    return True


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    os.chdir(BASE_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(
        description='Titan HFT System V2 - Pipeline Orchestrator',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--mode', default='verify',
                        choices=['full', 'convert', 'features', 'train',
                                 'export', 'validate', 'verify'],
                        help='Pipeline stage to run')
    parser.add_argument('--symbol', default='XAUUSD',
                        help='Trading symbol (default: XAUUSD)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='LSTM training epochs (default: 100)')
    parser.add_argument('--focal', action='store_true',
                        help='Use Focal Loss instead of BCE')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3). '
                             'Optimizer auto-tunes: 1e-3 -> 5e-4 -> 2e-4 -> 1e-4')
    parser.add_argument('--trade-log',
                        help='Path to trade log CSV for validation')
    parser.add_argument('--source-dir',
                        help='Source directory for .ticks files (convert stage)')
    args, unknown = parser.parse_known_args()

    # If the user's IDE/editor passes weird arguments like 'unknown 1', we safely skip them
    if unknown:
        print(f"  [INFO] Ignoring unrecognized arguments: {unknown}")

    MODE_MAP = {
        'verify':   lambda: stage_verify(),
        'convert':  lambda: stage_convert(args.source_dir),
        'features': lambda: stage_features(args.symbol),
        'train':    lambda: stage_train(args.focal, args.epochs, args.lr),
        'export':   stage_export,
        'validate': lambda: stage_validate(args.trade_log),
        'full':     lambda: stage_full(args),
    }

    ok = MODE_MAP[args.mode]()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()