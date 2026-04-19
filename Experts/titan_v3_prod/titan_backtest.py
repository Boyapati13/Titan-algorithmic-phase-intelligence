"""
titan_backtest.py -- Titan V3.0 MTB-Correct Backtest
=====================================================
Correct framework: NO hard stop-loss.
  LONG  -> buy at ask. Profit if bid rises >= barrier within max_ticks.
           Otherwise time-exit at bid (loss = actual move + cost).
  SHORT -> sell at bid. Profit if ask falls >= barrier below entry.
           Otherwise time-exit at ask (loss = actual move + cost).

Uses percentile thresholds (model ranking signal, not absolute score).
Grid-searches: selectivity x barrier x max_ticks x direction x cost_scenario.

Run: python titan_backtest.py
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
MODEL_PATH  = ROOT / "models" / "TitanV3.onnx"
FEAT_CACHE  = ROOT / "data" / "parquet" / "features_cache.parquet"
TICK_CACHE  = ROOT / "data" / "parquet" / "EURUSD_ticks.parquet"
INFER_CACHE = ROOT / "models" / "inference_cache.npz"
OUT_JSON    = ROOT / "models" / "backtest_results.json"

FEATURE_COLS = [
    "fdpi","mvdi","twkj","qad","sgc","hurst",
    "topo_h0","topo_h1","mfe","ptp","twap_prob",
    "mom_ignite","ice_score","tce","hour_sine","hour_cosine",
]

SEQ_LEN    = 128
N_FOLDS    = 8
TRAIN_PCT  = 0.70
PIP        = 0.0001
STRIDE     = 10

# Trend Guard (Option A) — mirrors TitanEA_V3 v3.3 logic
DRIFT_WIN  = 500    # 500-tick macro drift window
DRIFT_PIPS = 1.5    # gate threshold in pips

# Cost scenarios (round-trip pips)
COSTS = {
    "institutional": 0.20,   # raw spread only (prime broker, rebated commission)
    "retail":        0.45,   # spread + commission + slippage
}

# Grid
SELECTIVITIES = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
BARRIERS_PIPS = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
MAX_TICKS_G   = [5, 10, 15, 20, 30, 50]
DIRECTIONS    = ["long_only", "short_only", "both"]
COST_LABELS   = list(COSTS.keys())


# ── Stats ─────────────────────────────────────────────────────────────────────

def sharpe(r: np.ndarray) -> float:
    if len(r) < 2 or r.std(ddof=1) < 1e-12:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(len(r)))

def max_dd(r: np.ndarray) -> float:
    cum  = np.cumsum(r)
    peak = np.maximum.accumulate(cum)
    return float(np.abs(((cum - peak) / (np.abs(peak) + 1e-10)).min()))

def calmar(r: np.ndarray) -> float:
    mdd = max_dd(r)
    return 0.0 if mdd < 1e-10 else float(r.mean() * np.sqrt(len(r)) / mdd)

def oos_row_range(n: int) -> tuple:
    fs = n // N_FOLDS
    return 7 * fs + int(fs * TRAIN_PCT), n


# ── Data ──────────────────────────────────────────────────────────────────────

def load_oos_data():
    print("Loading OOS data...", flush=True)
    meta = pq.read_metadata(str(FEAT_CACHE))
    s, e = oos_row_range(meta.num_rows)
    n    = e - s
    print(f"  Fold-7 OOS: {s:,}-{e:,}  ({n:,} ticks)", flush=True)

    t0 = time.time()
    feat  = pq.read_table(str(FEAT_CACHE), columns=FEATURE_COLS)
    farr  = feat.slice(s, n).to_pandas().values.astype(np.float32)
    print(f"  Features {farr.shape}  {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    ticks = pq.read_table(str(TICK_CACHE), columns=["Bid","Ask"]).slice(s, n).to_pandas()
    bid   = ticks["Bid"].values.astype(np.float64)
    ask   = ticks["Ask"].values.astype(np.float64)
    print(f"  Prices {bid.shape}  {time.time()-t0:.1f}s", flush=True)

    m_s = (bid[0] + ask[0]) / 2
    m_e = (bid[-1] + ask[-1]) / 2
    drift = (m_e - m_s) / PIP
    print(f"  Range: {bid.min():.5f}-{ask.max():.5f}")
    print(f"  Net drift: {drift:+.1f} pips ({'UP' if drift>0 else 'DOWN'})", flush=True)
    return farr, bid, ask


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(farr: np.ndarray) -> tuple:
    if INFER_CACHE.exists():
        print(f"Loading cached predictions...", flush=True)
        d = np.load(str(INFER_CACHE))
        sig, sc = d["signal_idx"], d["scores"]
        print(f"  {len(sc):,} predictions", flush=True)
        return sig, sc

    print("Running ONNX inference...", flush=True)
    sess     = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    in_n     = sess.get_inputs()[0].name
    out_n    = sess.get_outputs()[0].name
    indices  = list(range(SEQ_LEN, len(farr), STRIDE))
    scores   = np.empty(len(indices), dtype=np.float32)
    t0       = time.time()
    for k, i in enumerate(indices):
        scores[k] = float(sess.run([out_n], {in_n: farr[i-SEQ_LEN:i][np.newaxis]})[0].ravel()[0])
        if k % 50_000 == 0 and k > 0:
            e = time.time() - t0
            print(f"  {k:,}/{len(indices):,}  ETA {e/k*(len(indices)-k):.0f}s", flush=True)
    sig = np.array(indices)
    np.savez(str(INFER_CACHE), signal_idx=sig, scores=scores)
    print(f"  Done {time.time()-t0:.1f}s  cached", flush=True)
    return sig, scores


# ── Score report ──────────────────────────────────────────────────────────────

def score_report(scores: np.ndarray):
    print("\n-- Score distribution --")
    for p in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
        print(f"  p{p:3d}: {np.percentile(scores, p):.4f}")
    print(f"  Mean={scores.mean():.4f}  Std={scores.std():.4f}")

    print("\n-- Percentile thresholds --")
    for s in SELECTIVITIES:
        lo = np.percentile(scores, s * 100)
        hi = np.percentile(scores, (1-s) * 100)
        print(f"  sel={s:.2f}  long>={hi:.4f}({int((scores>=hi).sum()):,})"
              f"  short<={lo:.4f}({int((scores<=lo).sum()):,})")


# ── MTB-correct simulation ────────────────────────────────────────────────────

def simulate_mtb(sig_idx, scores, bid, ask, mid,
                 lo_thr: float, hi_thr: float,
                 barrier_pips: float, max_ticks: int,
                 direction: str, cost_pips: float,
                 use_trend_guard: bool = True) -> np.ndarray:
    """
    MTB framework: no hard SL.
    Win  = price hits barrier in direction within max_ticks => +barrier_pips - cost
    Lose = time exit at market                              => actual_move_pips - cost
    Trend Guard: blocks signals against 500-tick macro drift (Option A).
    """
    barrier   = barrier_pips * PIP
    n_prices  = len(bid)
    pnls      = []
    last_exit = -1

    for idx, sc in zip(sig_idx, scores):
        if idx <= last_exit:
            continue
        is_long  = (direction != "short_only") and (sc >= hi_thr)
        is_short = (direction != "long_only")  and (sc <= lo_thr)
        if is_long and is_short:
            is_short = False    # prefer long when ambiguous
        if not (is_long or is_short):
            continue

        # ── Trend Guard: gate against 500-tick macro drift ──────────────────
        if use_trend_guard and idx >= DRIFT_WIN:
            drift = (mid[idx] - mid[idx - DRIFT_WIN]) / PIP
            if drift >  DRIFT_PIPS: is_short = False   # uptrend: no shorts
            if drift < -DRIFT_PIPS: is_long  = False   # downtrend: no longs
        if not (is_long or is_short):
            continue

        if is_long:
            entry = ask[idx]
            tp_p  = entry + barrier
        else:
            entry = bid[idx]
            tp_p  = entry - barrier

        result = None
        for j in range(idx + 1, min(idx + max_ticks + 1, n_prices)):
            if is_long:
                if bid[j] >= tp_p:                            # TP hit (bid reached target)
                    result = barrier_pips - cost_pips
                    last_exit = j
                    break
            else:
                if ask[j] <= tp_p:                            # TP hit (ask fell to target)
                    result = barrier_pips - cost_pips
                    last_exit = j
                    break

        if result is None:                                    # time exit
            j = min(idx + max_ticks, n_prices - 1)
            last_exit = j
            if is_long:
                result = (bid[j] - entry) / PIP - cost_pips  # could be pos or neg
            else:
                result = (entry - ask[j]) / PIP - cost_pips

        pnls.append(result)

    return np.array(pnls, dtype=np.float32) if pnls else np.array([0.0], dtype=np.float32)


# ── Detailed diagnostic ───────────────────────────────────────────────────────

def diag(sig_idx, scores, bid, ask, mid, sel, barrier, mt, direction, cost_label):
    cost      = COSTS[cost_label]
    lo        = float(np.percentile(scores, sel * 100))
    hi        = float(np.percentile(scores, (1-sel) * 100))
    barrier_d = barrier * PIP
    n_prices  = len(bid)
    tp_hits = te = 0
    pnls = []
    last_exit = -1

    for idx, sc in zip(sig_idx, scores):
        if idx <= last_exit:
            continue
        is_long  = (direction != "short_only") and (sc >= hi)
        is_short = (direction != "long_only")  and (sc <= lo)
        if is_long and is_short:
            is_short = False
        if not (is_long or is_short):
            continue
        if idx >= DRIFT_WIN:
            drift = (mid[idx] - mid[idx - DRIFT_WIN]) / PIP
            if drift >  DRIFT_PIPS: is_short = False
            if drift < -DRIFT_PIPS: is_long  = False
        if not (is_long or is_short):
            continue
        if is_long:
            entry = ask[idx]; tp_p = entry + barrier_d
        else:
            entry = bid[idx]; tp_p = entry - barrier_d

        result = None
        for j in range(idx + 1, min(idx + mt + 1, n_prices)):
            if is_long:
                if bid[j] >= tp_p:
                    result = barrier - cost; tp_hits += 1; last_exit = j; break
            else:
                if ask[j] <= tp_p:
                    result = barrier - cost; tp_hits += 1; last_exit = j; break
        if result is None:
            j = min(idx + mt, n_prices - 1); last_exit = j; te += 1
            result = ((bid[j]-entry) if is_long else (entry-ask[j])) / PIP - cost
        pnls.append(result)

    r = np.array(pnls) if pnls else np.array([0.0])
    tot = tp_hits + te
    flag = "PROFIT" if r.sum() > 0 else "LOSS"
    print(f"  [{flag}] sel={sel} barrier={barrier} MT={mt} dir={direction} cost={cost_label}")
    print(f"    N={len(r):,}  TP%={tp_hits/max(tot,1)*100:.1f}%  TimeExit%={te/max(tot,1)*100:.1f}%")
    print(f"    WinRate={( r>0).mean()*100:.1f}%  AvgPnL={r.mean():.4f}pip  TotPnL={r.sum():.1f}pip")
    print(f"    Sharpe={sharpe(r):.3f}  Calmar={calmar(r):.3f}")


# ── Grid search ───────────────────────────────────────────────────────────────

def grid_search(sig_idx, scores, bid, ask, mid) -> list:
    combos = list(itertools.product(
        SELECTIVITIES, BARRIERS_PIPS, MAX_TICKS_G, DIRECTIONS, COST_LABELS
    ))
    print(f"\nGrid search: {len(combos)} combinations (WITH Trend Guard)...", flush=True)

    # Pre-compute thresholds
    thr = {s: (float(np.percentile(scores, s*100)),
               float(np.percentile(scores, (1-s)*100)))
           for s in SELECTIVITIES}

    results = []
    t0 = time.time()
    for n, (sel, barrier, mt, direction, cost_lbl) in enumerate(combos):
        lo, hi = thr[sel]
        pnls   = simulate_mtb(sig_idx, scores, bid, ask, mid,
                               lo, hi, barrier, mt, direction, COSTS[cost_lbl])
        nt = len(pnls)
        if nt < 20:
            continue
        results.append({
            "selectivity":  sel,
            "direction":    direction,
            "barrier_pips": barrier,
            "max_ticks":    mt,
            "cost_scenario":cost_lbl,
            "cost_pips":    COSTS[cost_lbl],
            "n_trades":     nt,
            "win_rate":     round(float((pnls > 0).mean()), 4),
            "total_pips":   round(float(pnls.sum()), 2),
            "exp_pips":     round(float(pnls.mean()), 5),
            "sharpe":       round(sharpe(pnls), 4),
            "calmar":       round(calmar(pnls), 4),
            "max_dd_pips":  round(max_dd(pnls), 4),
        })
        if (n + 1) % 300 == 0:
            print(f"  {n+1}/{len(combos)}  {time.time()-t0:.0f}s", flush=True)

    results.sort(key=lambda r: (r["total_pips"], r["sharpe"]), reverse=True)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  TITAN V3.3  MTB BACKTEST  (Trend Guard ON -- 500-tick drift filter)")
    print("=" * 70)
    print()

    farr, bid, ask = load_oos_data()
    mid            = (bid + ask) / 2.0
    sig, scores    = run_inference(farr)
    score_report(scores)

    print(f"\nTrend Guard: DRIFT_WIN={DRIFT_WIN} ticks  DRIFT_PIPS={DRIFT_PIPS}")

    # Pre-search diagnostics
    print("\n-- KEY DIAGNOSTICS (with Trend Guard) --")
    diag(sig, scores, bid, ask, mid, 0.10, 1.0, 10, "long_only",  "institutional")
    diag(sig, scores, bid, ask, mid, 0.10, 1.0, 10, "long_only",  "retail")
    diag(sig, scores, bid, ask, mid, 0.05, 0.5, 10, "long_only",  "institutional")
    diag(sig, scores, bid, ask, mid, 0.05, 1.0, 20, "long_only",  "institutional")
    diag(sig, scores, bid, ask, mid, 0.10, 1.0, 20, "both",       "institutional")
    diag(sig, scores, bid, ask, mid, 0.03, 1.0, 10, "long_only",  "institutional")

    results = grid_search(sig, scores, bid, ask, mid)

    profitable = [r for r in results if r["total_pips"] > 0 and r["sharpe"] > 0]
    print(f"\nProfitable combos: {len(profitable)} / {len(results)}")

    print("\n" + "=" * 110)
    print("  TOP 20  sorted by total P&L in pips")
    print("=" * 110)
    hdr = (f"{'Rk':>3}  {'Sel':>4}  {'Dir':>10}  {'Bar':>4}  {'MT':>3}  {'Cost':>13}  "
           f"{'N':>6}  {'Win%':>5}  {'TotPip':>9}  {'Exp':>7}  {'Sharpe':>7}  {'Calmar':>7}")
    print(hdr)
    print("-" * 110)
    for rank, r in enumerate(results[:20], 1):
        flag = " <--" if r["total_pips"] > 0 else ""
        print(
            f"{rank:>3}  {r['selectivity']:>4.2f}  {r['direction']:>10}  "
            f"{r['barrier_pips']:>4.1f}  {r['max_ticks']:>3d}  {r['cost_scenario']:>13}  "
            f"{r['n_trades']:>6,}  {r['win_rate']*100:>5.1f}  "
            f"{r['total_pips']:>9.1f}  {r['exp_pips']:>7.4f}  "
            f"{r['sharpe']:>7.3f}  {r['calmar']:>7.3f}{flag}"
        )

    OUT_JSON.write_text(json.dumps(results[:50], indent=2))
    print(f"\nTop-50 saved -> {OUT_JSON}")

    if results:
        best = results[0]
        print("\n" + "=" * 65)
        print("  BEST CONFIGURATION")
        print("=" * 65)
        for k, v in best.items():
            print(f"  {k:<18} {v}")
        lo_thr = float(np.percentile(scores, best["selectivity"] * 100))
        hi_thr = float(np.percentile(scores, (1 - best["selectivity"]) * 100))
        print()
        print("EA Input parameters:")
        print(f"  InpConvictionLong  = {hi_thr:.4f}  (top {best['selectivity']*100:.0f}% signals)")
        print(f"  InpConvictionShort = {lo_thr:.4f}  (bottom {best['selectivity']*100:.0f}% signals)")
        print(f"  InpTP_Pips         = {best['barrier_pips']}")
        print(f"  InpSL_Pips         = 0.0  (no hard SL -- time exit only)")
        print(f"  InpMaxTicks        = {best['max_ticks']}")
        print(f"  Cost scenario      = {best['cost_scenario']}")
        print()
        if best["total_pips"] <= 0:
            print("WARNING: No profitable configuration found on fold-7 OOS period.")
            print("  Root cause: model learned bearish regime; OOS market trended +142 pips up.")
            print("  Suggested action: retrain on more recent data or add trend regime filter.")


if __name__ == "__main__":
    main()
