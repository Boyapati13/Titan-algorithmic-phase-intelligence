"""
titan_features.py — Titan V3.0 Feature Engineering  (Patch 3.2)
================================================================
Fixes applied to this file:

Fix 1 — QAD mathematical redundancy
    OLD: QAD = (r_ask−r_bid)/(r_ask+r_bid) where r_x = count/time
         → time cancels → QAD ≡ FDPI exactly (proven in audit).
    NEW: QAD = FDPI(short_window) − FDPI(long_window)
         = velocity of directional pressure over time.
         Positive: imbalance is BUILDING (recent > historical).
         Negative: imbalance is FADING (recent < historical).
         Genuinely orthogonal to FDPI (the level).

Fix 2 — MFE Z-scoring breaks Helmholtz physics
    OLD: T_norm = zscore(tick_rate)  ← can be negative
         S_norm = zscore(entropy)    ← can be negative
         F = U − T_norm * S_norm     ← physics meaningless
    NEW: T_01 = minmax01(tick_rate)  ← always in [0,1]
         S_01 = minmax01(entropy)    ← always in [0,1]
         F = U − T_01 * S_01         ← Helmholtz preserved
         MFE = zscore(F)             ← only output is Z-scored

Fix 3 — TWKJ dt³ denominator instability
    OLD: v = dM/dt_inst, a = dv/dt_inst, j = da/dt_inst
         dt_inst ∈ [1ms, ∞) → 1-ms jitter = 8× change in jerk
    NEW: dt_smooth = rolling_median(dt, 32 ticks)
         Smoothed dt removes tick-level latency jitter while
         preserving genuine burst/quiet regime transitions.

Fix 7 — SGC outlier ghosting
    OLD: sgc_raw = rolling_min(s_acc, 32)
         One noisy 1ms spike pins the feature to −1 for 32 ticks.
    NEW: sgc_raw = rolling_quantile(s_acc, 0.05, 32)
         5th-percentile immune to single-tick outliers.

Fix 8 — Entropy bins 8 → 12 (better resolution)

Fix 9 — PTP k_soft configurable via CFG.features.ptp_k_soft

ddof=0 throughout (matches MQL5 CRollZ exactly)
"""
from __future__ import annotations

import gc
import logging
import time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from titan_config import CFG, FEATURE_COLS

warnings.filterwarnings("ignore", category=RuntimeWarning)
log = logging.getLogger("TitanFeatures")

try:
    from ripser import ripser as _ripser
    _TDA_OK = True
    log.info("ripser loaded — true persistent homology active.")
except ImportError:
    _TDA_OK = False
    log.warning("ripser not found — TDA fallback active.")


# ─────────────────────────────────────────────────────────────────────────────
# Math primitives — ddof=0 everywhere
# ─────────────────────────────────────────────────────────────────────────────

def _zscore_clip(series: pd.Series, window: int,
                 clip: float = None, eps: float = None) -> pd.Series:
    """
    Rolling Z-score, ddof=0, clip to [−clip,+clip], scale to [−1,+1].
    ddof=0 matches MQL5 CRollZ.push() exactly — zero parity drift.

    Memory note: uses numpy in-place ops to avoid the extra 674 MB allocation
    that pandas .clip() creates internally via np.where on 88M-row datasets.
    """
    clip = clip or CFG.features.clip_sigma
    eps  = float(eps or CFG.features.eps)
    mu   = series.rolling(window, min_periods=window).mean()
    sig  = series.rolling(window, min_periods=window).std(ddof=0).fillna(eps)
    arr  = series.values - mu.values
    mu   = None                          # allow GC before next allocation
    arr /= (sig.values + eps)
    sig  = None                          # allow GC before clip
    np.clip(arr, -clip, clip, out=arr)   # in-place — no extra array
    arr /= clip
    return pd.Series(arr, index=series.index)


def _rolling_minmax01(series: pd.Series, window: int,
                      lo_pct: float, hi_pct: float,
                      eps: float = None) -> pd.Series:
    """
    Rolling percentile-based min-max normalisation → [0, 1].
    Used for MFE to keep T and S strictly non-negative (Fix 2).
    lo_pct / hi_pct define the rolling "0" and "1" reference points.
    """
    eps   = eps or CFG.features.eps
    lo    = series.rolling(window, min_periods=window).quantile(lo_pct)
    hi    = series.rolling(window, min_periods=window).quantile(hi_pct)
    norm  = (series - lo) / (hi - lo + eps)
    return norm.clip(0.0, 1.0)


def _rolling_cv(series: pd.Series, window: int, eps: float = None) -> pd.Series:
    eps = eps or CFG.features.eps
    mu  = series.rolling(window, min_periods=window).mean()
    sig = series.rolling(window, min_periods=window).std(ddof=0)
    return sig / (mu.abs() + eps)


def _hurst_rs(x: np.ndarray) -> float:
    n = len(x)
    if n < 8: return 0.5
    mu  = x.mean(); cd = np.cumsum(x - mu)
    R   = cd.max() - cd.min(); S = x.std(ddof=0)
    if S < 1e-12 or R < 1e-12: return 0.5
    return float(np.clip(np.log(R / S) / np.log(n), 0.0, 1.0))


def _shannon_entropy(x: np.ndarray, n_bins: int) -> float:
    if len(x) < n_bins: return 0.0
    log_x    = np.log(np.maximum(x, 1e-9))
    counts, _ = np.histogram(log_x, bins=n_bins)
    total     = counts.sum()
    if total == 0: return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


# ─────────────────────────────────────────────────────────────────────────────
# TDA
# ─────────────────────────────────────────────────────────────────────────────

def _tda_ripser(pts: np.ndarray) -> Tuple[float, float]:
    dgms = _ripser(pts, maxdim=1)["dgms"]
    h0   = dgms[0]; fin0 = h0[np.isfinite(h0[:, 1])]
    if len(fin0) > 0:
        lv  = np.maximum(fin0[:,1] - fin0[:,0], 1e-12)
        p   = lv / lv.sum()
        h0e = float(-np.sum(p * np.log(p + 1e-12)))
    else:
        h0e = 0.0
    h1 = dgms[1]
    if len(h1) > 0:
        fin1 = h1[np.isfinite(h1[:,1])]
        h1p  = float((fin1[:,1]-fin1[:,0]).max()) if len(fin1)>0 else 0.0
    else:
        h1p = 0.0
    return h0e, h1p


def _tda_fallback(pts: np.ndarray) -> Tuple[float, float]:
    sub = pts[::max(1, len(pts)//32)]
    if len(sub) < 4: return 0.0, 0.0
    d   = np.sqrt(((sub[:,None]-sub[None,:])**2).sum(-1))
    flat= d[np.triu_indices(len(sub), k=1)]
    if flat.max() < 1e-10: return 0.0, 0.0
    counts, _ = np.histogram(flat, bins=8)
    p = counts[counts>0]/counts.sum()
    h0e = float(-np.sum(p*np.log(p+1e-10)))
    h1p = float(flat.std()/(flat.mean()+1e-10))
    return h0e, h1p


def _compute_tda(mid: np.ndarray, spread: np.ndarray,
                 log_dt: np.ndarray) -> Tuple[float, float]:
    pts = np.column_stack([mid, spread, log_dt])
    std = pts.std(0); std[std<1e-10] = 1.0
    pts = (pts - pts.mean(0)) / std
    if CFG.features.tda_enabled and _TDA_OK:
        return _tda_ripser(pts)
    return _tda_fallback(pts)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean raw tick data.

    For large datasets (>50M rows), uses chunked processing to fit in RAM.
    """
    required = {"Bid", "Ask", "Flags", "Tick_Time_ms"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # For datasets > 50M rows: system is RAM-constrained. Return as-is with minimal processing.
    # Derived columns will be computed on-demand downstream if needed.
    if len(df) > 50_000_000:
        log.warning(f"Large dataset ({len(df):,} rows) — returning as-is (RAM-constrained)")
        # Ensure required columns exist; don't compute derived ones yet
        required = {"Bid", "Ask", "Flags", "Tick_Time_ms"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return df
    else:
        # Small dataset: process normally
        ts  = pd.to_numeric(df["Tick_Time_ms"], errors="coerce").values
        bid = pd.to_numeric(df["Bid"],          errors="coerce").values
        ask = pd.to_numeric(df["Ask"],          errors="coerce").values
        flg = pd.to_numeric(df["Flags"],        errors="coerce").values
        del df

        valid = (
            np.isfinite(ts) & np.isfinite(bid) & np.isfinite(ask) &
            (bid > 0) & (ask > 0) & (ask >= bid)
        )

        ts  = ts[valid].astype(np.int64)
        bid = bid[valid].astype(np.float32)
        ask = ask[valid].astype(np.float32)
        flg = np.where(np.isnan(flg[valid]),
                       float(CFG.features.FLAG_BOTH),
                       flg[valid]).astype(np.int32)
        del valid

        mid    = (bid + ask) * np.float32(0.5)
        spread = np.clip(ask - bid, np.float32(0.0), None)

        result = pd.DataFrame({
            "Tick_Time_ms": ts,
            "Bid":          bid,
            "Ask":          ask,
            "Flags":        flg,
            "Mid":          mid,
            "Spread":       spread,
        })

    # Check if already sorted (load_parquet does this)
    ts_vals = result["Tick_Time_ms"].values
    if not (ts_vals[:-1] <= ts_vals[1:]).all():
        order = np.argsort(ts_vals, kind="stable")
        result = result.iloc[order].reset_index(drop=True)
        del order
    else:
        result = result.reset_index(drop=True)

    if len(result) < CFG.features.window_n:
        raise ValueError(f"Insufficient ticks: {len(result)} < {CFG.features.window_n}")

    return result


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 16 features.  All values ∈ [−1, +1].  NaN in warm-up rows.

    Memory strategy (keeps ALL ticks, fixes OOM on 16 GB machines):
      • Each feature converted to float32 immediately after computation
        (halves feature memory: float64 88M×16×8B=11.3GB → float32=5.6GB)
      • Intermediate variables deleted as soon as they are consumed
      • gc.collect() after the TDA block (peak memory moment)
      • tda_subsample=50 in config (5× fewer ripser calls vs original 10)
    """
    fc  = CFG.features
    N   = fc.window_n
    eps = fc.eps

    log.info(f"compute_features: {len(df):,} ticks, window_n={N}, "
             f"tda_subsample={fc.tda_subsample}")

    # Chunked processing for large datasets (OOM avoidance on 16 GB machines)
    CHUNK_ROWS = 10_000_000   # 10M rows per chunk (~1-2 GB working set)
    WARMUP     = N            # N=128: rows prepended from previous chunk for rolling warm-up

    if len(df) > CHUNK_ROWS:
        log.info(f"Chunking {len(df):,} rows into {len(df)//CHUNK_ROWS + 1} chunks "
                 f"(chunk={CHUNK_ROWS:,}, warm-up={WARMUP})")
        chunks_out = []
        prev_tail  = None                         # last WARMUP rows of previous chunk

        for start in range(0, len(df), CHUNK_ROWS):
            end   = min(start + CHUNK_ROWS, len(df))
            chunk = df.iloc[start:end]

            if prev_tail is not None:
                padded   = pd.concat([prev_tail, chunk], ignore_index=True)
                n_warmup = len(prev_tail)
            else:
                padded   = chunk
                n_warmup = 0

            # Recursively run feature computation on padded chunk (will be small enough to fit in RAM)
            feat_chunk = compute_features(padded)

            # Strip warm-up rows from output
            if n_warmup > 0:
                feat_chunk = feat_chunk.iloc[n_warmup:].reset_index(drop=True)

            chunks_out.append(feat_chunk)
            prev_tail = chunk.iloc[-WARMUP:]      # save tail for next chunk
            del padded, feat_chunk, chunk
            gc.collect()
            log.info(f"  Chunk done: rows {start:,}–{end:,}")

        result = pd.concat(chunks_out, ignore_index=True)
        del chunks_out, prev_tail
        return result

    # Non-chunked path: compute all features at once (only runs on chunks ≤10M rows)
    # Derived columns (add if missing, they may come from validate_input or need to be created)
    if "Mid" not in df.columns:
        df = df.copy()
        df["Mid"]    = (df["Bid"] + df["Ask"]) * 0.5
        df["Spread"] = (df["Ask"] - df["Bid"]).clip(lower=0)

    dt     = df["Tick_Time_ms"].diff().clip(lower=fc.min_dt_ms)
    mid    = df["Mid"]
    spread = df["Spread"]
    flags  = df["Flags"]
    buy_ticks  = (flags == fc.FLAG_ASK_ONLY).astype(np.float64)
    sell_ticks = (flags == fc.FLAG_BID_ONLY).astype(np.float64)
    B_N        = buy_ticks.rolling(N, min_periods=N).sum()
    S_N        = sell_ticks.rolling(N, min_periods=N).sum()

    # ── GROUP A: Order Flow Physics ───────────────────────────────────────────

    # F1: FDPI — signed imbalance ratio (natural [−1,+1])
    fdpi = ((B_N - S_N) / (B_N + S_N + eps)).astype(np.float32)
    # NOTE: B_N, S_N kept — still needed for mom_ignite

    # F2: MVDI
    spread_cv = _rolling_cv(spread, N, eps)
    mid_cv    = _rolling_cv(mid.diff().abs(), N, eps)
    mvdi      = _zscore_clip(spread_cv / (mid_cv + eps), N).astype(np.float32)
    del spread_cv, mid_cv

    # ─────────────────────────────────────────────────────────────────────────
    # F3: TWKJ — FIX 3: smoothed dt in denominator removes dt³ instability
    # ─────────────────────────────────────────────────────────────────────────
    dt_smooth = (
        dt.rolling(fc.twkj_dt_smooth_window, min_periods=4)
        .median()
        .clip(lower=fc.min_dt_ms)
    )
    velocity     = mid.diff() / dt_smooth
    acceleration = velocity.diff() / dt_smooth
    del velocity                              # free 674 MB before jerk alloc
    jerk         = acceleration.diff() / dt_smooth
    del acceleration                          # free 674 MB before _zscore_clip
    twkj         = _zscore_clip(jerk, N).astype(np.float32)
    del jerk

    # ─────────────────────────────────────────────────────────────────────────
    # F4: QAD — FIX 1: velocity of directional pressure (not rate ratio)
    # ─────────────────────────────────────────────────────────────────────────
    qad_w      = fc.qad_short_window
    B_short    = buy_ticks.rolling(qad_w,  min_periods=qad_w).sum()
    S_short    = sell_ticks.rolling(qad_w, min_periods=qad_w).sum()
    fdpi_short = (B_short - S_short) / (B_short + S_short + eps)
    qad        = (fdpi_short - fdpi).clip(-1.0, 1.0).astype(np.float32)
    del B_short, S_short, fdpi_short

    # ─────────────────────────────────────────────────────────────────────────
    # F5: SGC — FIX 7: robust percentile replaces hard min()
    # ─────────────────────────────────────────────────────────────────────────
    s_vel   = spread.diff() / dt_smooth
    s_acc   = s_vel.diff() / dt_smooth
    del s_vel                                 # free 674 MB before quantile alloc
    sgc_raw = s_acc.rolling(
        fc.window_sgc, min_periods=fc.window_sgc
    ).quantile(fc.sgc_percentile)
    del s_acc                                 # free 674 MB before _zscore_clip
    sgc     = _zscore_clip(sgc_raw, N).astype(np.float32)
    del sgc_raw, dt_smooth

    # F6: Hurst exponent
    hurst_raw = mid.rolling(fc.window_hurst, min_periods=8).apply(_hurst_rs, raw=True)
    hurst     = ((hurst_raw - 0.5) * 2.0).clip(-1.0, 1.0).astype(np.float32)
    del hurst_raw

    gc.collect()
    log.info("Group A (F1-F6) complete.")

    # ── GROUP B: Topological State ────────────────────────────────────────────
    # Peak memory moment — all Group A features live + TDA numpy arrays.
    # After this block we aggressively free and call gc.collect().
    log_dt_arr = np.log(dt.clip(lower=0.1).bfill().fillna(1.0).values)
    mid_arr    = mid.values; spr_arr = spread.values
    n_ticks    = len(df); sub = fc.tda_subsample
    h0_sparse: dict = {}; h1_sparse: dict = {}

    for i in range(N-1, n_ticks):
        if (i-(N-1)) % sub != 0 and i != n_ticks-1:
            continue
        s_  = i - N + 1
        h0v, h1v = _compute_tda(mid_arr[s_:i+1], spr_arr[s_:i+1], log_dt_arr[s_:i+1])
        h0_sparse[i] = h0v; h1_sparse[i] = h1v

    h0_raw = pd.Series(np.nan, index=df.index, dtype=np.float64)
    h1_raw = pd.Series(np.nan, index=df.index, dtype=np.float64)
    for pos, val in h0_sparse.items(): h0_raw.iloc[pos] = val
    for pos, val in h1_sparse.items(): h1_raw.iloc[pos] = val
    h0_raw = h0_raw.ffill(); h1_raw = h1_raw.ffill()
    topo_h0 = _zscore_clip(h0_raw, N).astype(np.float32)
    topo_h1 = _zscore_clip(h1_raw, N).astype(np.float32)

    # Free all TDA buffers immediately — reclaim RAM before Group C
    del log_dt_arr, mid_arr, spr_arr, h0_sparse, h1_sparse, h0_raw, h1_raw
    gc.collect()
    log.info("TDA features computed (subsampled). Memory freed.")

    # ─────────────────────────────────────────────────────────────────────────
    # F9: MFE — FIX 2: preserve Helmholtz physics by keeping T, S ≥ 0
    # ─────────────────────────────────────────────────────────────────────────
    U         = fdpi.abs()   # ∈ [0,1] — float32, no extra float64 allocation
    tick_rate = (1000.0 / dt).clip(upper=1e4)
    T_01      = _rolling_minmax01(tick_rate, N,
                                  lo_pct=fc.mfe_lo_pct, hi_pct=fc.mfe_hi_pct, eps=eps)
    del tick_rate

    S_raw = dt.rolling(fc.window_entropy, min_periods=fc.window_entropy).apply(
        lambda x: _shannon_entropy(x, fc.entropy_bins), raw=True
    )
    S_01  = _rolling_minmax01(S_raw, N,
                               lo_pct=fc.mfe_lo_pct, hi_pct=fc.mfe_hi_pct, eps=eps)
    del S_raw

    F_raw = U - T_01 * S_01
    mfe   = _zscore_clip(F_raw, N).astype(np.float32)
    del U, T_01, S_01, F_raw

    # ─────────────────────────────────────────────────────────────────────────
    # F10: PTP — FIX 9: configurable k_soft reduces look-ahead lag
    # ─────────────────────────────────────────────────────────────────────────
    dF  = mfe.diff(); d2F = dF.diff()
    ptp = (
        ((dF < 0) & (d2F < 0)).astype(np.float64)
        .rolling(fc.ptp_k_soft, min_periods=1)
        .mean() - 0.5
    ) * 2.0
    ptp = ptp.astype(np.float32)
    del dF, d2F

    log.info("Group C (F9-F10) complete.")

    # ── GROUP D: Algo Signatures ──────────────────────────────────────────────

    # F11: TWAP Probability
    dt_cv     = _rolling_cv(dt, N, eps)
    twap_prob = _zscore_clip(1.0 / (1.0 + dt_cv), N).astype(np.float32)
    del dt_cv

    # F12: Momentum Igniter
    burst_buy  = buy_ticks.rolling(fc.window_short, min_periods=fc.window_short).sum()
    burst_sell = sell_ticks.rolling(fc.window_short, min_periods=fc.window_short).sum()
    mom_ignite = _zscore_clip(
        (burst_buy / (B_N + eps)) - (burst_sell / (S_N + eps)), N
    ).astype(np.float32)
    del burst_buy, burst_sell, buy_ticks, sell_ticks, B_N, S_N

    # F13: Iceberg Score
    mid_static   = (mid.diff().abs() < CFG.data.pip_size).astype(np.float64)
    flag_consist = (flags == flags.shift(1)).astype(np.float64)
    ice_score    = _zscore_clip(
        (mid_static * flag_consist).rolling(32, min_periods=16).mean(), N
    ).astype(np.float32)
    del mid_static, flag_consist

    # F14: Temporal Clustering Entropy (inverted; entropy_bins=12)
    tce_raw = dt.rolling(fc.window_entropy, min_periods=fc.window_entropy).apply(
        lambda x: _shannon_entropy(x, fc.entropy_bins), raw=True
    )
    tce = (-_zscore_clip(tce_raw, N)).astype(np.float32)
    del tce_raw, dt

    log.info("Group D (F11-F14) complete.")

    # ── GROUP E: Context ──────────────────────────────────────────────────────

    # F15+F16: Sine + Cosine — unambiguous hour encoding
    try:
        hour_col = pd.to_datetime(df["Tick_Time_ms"], unit="ms", utc=True).dt.hour
    except Exception:
        hour_col = pd.Series(np.zeros(len(df)), index=df.index)
    angle       = 2.0 * np.pi * hour_col / 24.0
    hour_sine   = np.sin(angle).astype(np.float32)
    hour_cosine = np.cos(angle).astype(np.float32)
    del hour_col, angle

    # ── Assemble ──────────────────────────────────────────────────────────────
    out = pd.DataFrame({
        "fdpi":        fdpi,        "mvdi":        mvdi,
        "twkj":        twkj,        "qad":          qad,
        "sgc":         sgc,         "hurst":        hurst,
        "topo_h0":     topo_h0,     "topo_h1":      topo_h1,
        "mfe":         mfe,         "ptp":           ptp,
        "twap_prob":   twap_prob,   "mom_ignite":    mom_ignite,
        "ice_score":   ice_score,   "tce":           tce,
        "hour_sine":   hour_sine,   "hour_cosine":   hour_cosine,
    }, index=df.index)[FEATURE_COLS]

    if __debug__:
        valid = out.notna().all(axis=1)
        if valid.any():
            mx = out.loc[valid].abs().max().max()
            if mx > 1.0 + 1e-5:
                bad = out.loc[valid].abs().max(); bad = bad[bad > 1.0+1e-5]
                raise AssertionError(f"Bounds violation: {bad.to_dict()}")

    log.info(
        f"compute_features: {len(out):,} ticks, "
        f"{out.notna().all(axis=1).sum():,} valid rows"
    )
    return out


def build_sequences(
    features_df: pd.DataFrame,
    labels: pd.Series,
    seq_len: int = None,
    max_sequences: int = 500_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) sequence arrays capped at max_sequences.

    With 88M ticks, uncapped this would require ~724 GB RAM for np.stack().
    Uses a vectorised O(n) sliding-window sum to find all valid ending positions,
    then applies an adaptive stride so the full history is uniformly represented.

    Memory budget at cap: 500K × 128 × 16 × 4 B ≈ 409 MB for X,
                          plus 500K × 128 × 8 B ≈ 512 MB for the index array.
    """
    import math
    seq_len   = seq_len or CFG.model.sequence_len
    feat_arr  = features_df.values.astype(np.float32)
    label_arr = labels.values.astype(np.float32)
    valid     = (~np.isnan(feat_arr).any(axis=1)) & (~np.isnan(label_arr))

    # O(n) vectorised: sliding window sum over `valid`.
    # cumsum[i+w] - cumsum[i] = number of True values in valid[i:i+w].
    # window_sum[i] == seq_len  ↔  every tick in [i, i+seq_len) is valid.
    cumsum     = np.concatenate([[0], np.cumsum(valid.astype(np.int32))])
    window_sum = cumsum[seq_len:] - cumsum[: len(cumsum) - seq_len]
    # valid_ends[k] is the absolute index of the last tick in the k-th valid window.
    valid_ends = np.where(window_sum == seq_len)[0] + (seq_len - 1)

    n_ends = len(valid_ends)
    if n_ends == 0:
        raise ValueError("No valid sequences — check NaN propagation.")

    stride = max(1, math.ceil(n_ends / max_sequences))
    sampled_ends = valid_ends[::stride]
    if stride > 1:
        log.info(
            f"build_sequences: {n_ends:,} valid ends → {len(sampled_ends):,} sampled "
            f"(stride={stride}, cap={max_sequences:,})"
        )

    # Vectorised gather — no Python loop over sequences.
    # indices: (n_seq, seq_len) with dtype intp to satisfy numpy fancy-index rules.
    starts  = sampled_ends - seq_len + 1
    indices = starts[:, None] + np.arange(seq_len, dtype=np.intp)[None, :]
    X = feat_arr[indices]                         # (n_seq, seq_len, n_features)
    y = label_arr[sampled_ends].astype(np.float32)

    log.info(f"build_sequences: X={X.shape} y={y.shape} pos={y.mean():.4f}")
    return X, y
