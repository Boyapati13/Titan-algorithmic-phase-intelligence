"""
titan_labeler.py — Titan V3.0 Labeling Engine  (Patch 3.2)
===========================================================
Fixes applied to this file:

Fix 4  — Phase 3.5 Feature Independence test MUST halt
    OLD: if flagged: log.warning(...)   ← silently proceeds
    NEW: if flagged and cfg.p35_halt_on_corr: raise ValueError(...)
         Because QAD was formerly ≡ FDPI (correlation = 1.0), the old
         code would have logged a warning and then trained the LSTM on
         16 features with one perfectly redundant pair.  Now it halts.

Fix 10 — HMM Viterbi look-ahead bias (data leakage)
    OLD: self.hmm.predict(X)  uses Viterbi → bidirectional dynamic
         programming → state at t uses observations from t+1 to T_end.
         Labels built from these states embed future market knowledge
         → spectacular in-sample performance, guaranteed live failure.
    NEW: Causal EMA smoothing of GMM soft probabilities.
         At each tick t, only probabilities from t=0..t are used.
         No future observations enter the label generation process.

Fix 11 — Bisection trap in quiet/low-volatility regimes
    OLD: binary-search can never converge when the 5-pip floor prevents
         the hit rate from reaching 20% regardless of K.
         The bracket collapses to a degenerate interval and the labels
         are entirely broken for the epoch.
    NEW: stuck-detection: if |hi−lo| < bisect_floor_tol over 5 consecutive
         iterations, the labeler recognises the "floor-pinned" regime and
         returns all-zero labels for that epoch with a clear warning.
         Caller can filter or downsample these epochs before training.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from titan_config import CFG, FEATURE_COLS, LabelConfig, PhaseConfig

log = logging.getLogger("TitanLabeler")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3.5 Validator
# ─────────────────────────────────────────────────────────────────────────────

class Phase35Validator:
    """
    Mandatory pre-training hypothesis validation.
    Fix 4: Test 2 (feature independence) now HALTS with ValueError
    when any pair exceeds the correlation threshold.
    """

    def __init__(self):
        self.lc: LabelConfig = CFG.labels
        self.results: Dict   = {}

    def run(self, features_list: List[pd.DataFrame]) -> Dict:
        if not self.lc.run_phase_35:
            log.info("Phase 3.5 skipped (config.run_phase_35=False).")
            return {}
        log.info("=" * 60)
        log.info("PHASE 3.5  HYPOTHESIS VALIDATION SPRINT")
        log.info("=" * 60)
        if len(features_list) >= 2:
            self._test1_cluster_stability(features_list)
        else:
            log.warning("Test 1 skipped: fewer than 2 periods.")
        self._test2_feature_independence(features_list[0])
        log.info("Phase 3.5 PASSED.")
        return self.results

    def _fit_gmm_labels(self, feat: pd.DataFrame) -> np.ndarray:
        clean = feat.dropna()
        sc    = StandardScaler()
        X     = sc.fit_transform(clean.values)
        gmm   = GaussianMixture(
            n_components=self.lc.n_clusters, n_init=self.lc.gmm_n_init,
            covariance_type=self.lc.gmm_covariance, reg_covar=self.lc.gmm_reg_covar,
            random_state=self.lc.random_state, max_iter=500,
        )
        gmm.fit(X)
        return gmm.predict(X)

    def _test1_cluster_stability(self, fl: List[pd.DataFrame]):
        log.info("Test 1: Cluster stability (ARI)…")
        sets = [self._fit_gmm_labels(f) for f in fl]
        aris = []
        for i in range(len(sets)):
            for j in range(i+1, len(sets)):
                n   = min(len(sets[i]), len(sets[j]))
                ari = adjusted_rand_score(sets[i][:n], sets[j][:n])
                aris.append(ari)
                log.info(f"  ARI period {i} vs {j}: {ari:.4f}")
        mean_ari = float(np.mean(aris))
        self.results["mean_ari"] = mean_ari
        log.info(f"  Mean ARI: {mean_ari:.4f}  (threshold: {self.lc.ari_threshold})")
        if mean_ari < self.lc.ari_threshold:
            raise ValueError(
                f"Phase 3.5 Test 1 FAILED: Mean ARI {mean_ari:.4f} < "
                f"{self.lc.ari_threshold}. Cluster structure not stable."
            )
        log.info("  Test 1 PASSED.")

    def _test2_feature_independence(self, feat: pd.DataFrame):
        """
        Fix 4: HALT if any feature pair exceeds corr_threshold.
        Old code only logged a warning — silently trained on redundant data.

        Subsampling note: spearmanr on 88M × 16 rows requires ranking all
        columns twice → ~22 GB RAM.  50K random rows give the same correlation
        estimate to within ±0.01 and fit easily in 64 MB.
        """
        _MAX_CORR_ROWS = 50_000
        log.info("Test 2: Feature independence (Spearman |rho|)…")
        clean   = feat.dropna()
        if len(clean) > _MAX_CORR_ROWS:
            clean = clean.sample(_MAX_CORR_ROWS, random_state=42)
            log.info(f"  Subsampled to {_MAX_CORR_ROWS:,} rows for spearmanr.")
        rho, _  = spearmanr(clean.values)
        rho_arr = np.array(rho)
        thresh  = self.lc.corr_threshold
        flagged = [
            (FEATURE_COLS[i], FEATURE_COLS[j], float(rho_arr[i, j]))
            for i in range(len(FEATURE_COLS))
            for j in range(i+1, len(FEATURE_COLS))
            if abs(rho_arr[i, j]) > thresh
        ]
        self.results["redundant_pairs"] = flagged

        if flagged:
            msg = (
                f"Phase 3.5 Test 2 FAILED: {len(flagged)} feature pair(s) "
                f"exceed |rho| > {thresh}:\n"
                + "\n".join(f"  {a} ↔ {b}  rho={r:.4f}" for a,b,r in flagged)
                + "\n\nAction: remove or replace the lower-importance feature "
                "from each pair before training."
            )
            if self.lc.p35_halt_on_corr:
                raise ValueError(msg)
            else:
                log.warning(msg)   # soft mode (p35_halt_on_corr=False)
        else:
            log.info(f"  No redundant pairs (max |rho| < {thresh}).  Test 2 PASSED.")


# ─────────────────────────────────────────────────────────────────────────────
# GMM Fitter
# ─────────────────────────────────────────────────────────────────────────────

class GMMClusterFitter:
    """Fits GMM on 16-D feature space; assigns ticks to market phases."""

    def __init__(self):
        self.lc: LabelConfig    = CFG.labels
        self.pc: PhaseConfig    = CFG.phase
        self.gmm: Optional[GaussianMixture]  = None
        self.scaler: Optional[StandardScaler] = None
        self.phase_map: Dict[int, str]         = {}
        self.bic: float                        = 0.0

    def fit(self, features_df: pd.DataFrame) -> "GMMClusterFitter":
        """
        Fit GMM on a random subsample, then predict on the full dataset.

        Subsampling rationale: GMM.fit() on 88M rows × 16 features takes tens
        of hours (O(n × K × n_init × n_iter)) and needs ~11 GB RAM for the
        StandardScaler double copy.  50K rows are sufficient to find stable
        cluster centroids in 16D with K=5 — adding more rows gives diminishing
        returns while multiplying cost.  Predict/predict_proba on all rows is
        cheap (vectorised matrix math, no EM).
        """
        _MAX_GMM_ROWS = 50_000
        clean       = features_df.dropna()
        if len(clean) > _MAX_GMM_ROWS:
            fit_data = clean.sample(_MAX_GMM_ROWS, random_state=self.lc.random_state)
            log.info(f"GMM fit subsampled: {len(clean):,} → {_MAX_GMM_ROWS:,} rows.")
        else:
            fit_data = clean

        self.scaler = StandardScaler()
        X_fit       = self.scaler.fit_transform(fit_data.values)
        self.gmm    = GaussianMixture(
            n_components=self.lc.n_clusters, n_init=self.lc.gmm_n_init,
            covariance_type=self.lc.gmm_covariance, reg_covar=self.lc.gmm_reg_covar,
            random_state=self.lc.random_state, max_iter=500,
        )
        self.gmm.fit(X_fit)
        self.bic = self.gmm.bic(X_fit)
        self._assign_phases(fit_data, self.gmm.predict(X_fit))
        log.info(f"GMM: K={self.lc.n_clusters} BIC={self.bic:.1f} "
                 f"converged={self.gmm.converged_}")
        return self

    def _assign_phases(self, feat_df: pd.DataFrame, labels: np.ndarray):
        """Data-driven rank-based phase assignment — no hardcoded thresholds."""
        K         = self.lc.n_clusters
        centroids = {}
        for k in range(K):
            mask = labels == k
            m    = feat_df.iloc[mask].mean() if mask.sum() > 0 else pd.Series(
                0.0, index=FEATURE_COLS
            )
            centroids[k] = {
                "fdpi": float(m.get("fdpi",       0.0)),
                "mom":  float(m.get("mom_ignite", 0.0)),
                "mfe":  float(m.get("mfe",        0.0)),
            }
        ranked_mom  = sorted(range(K), key=lambda k: centroids[k]["mom"],  reverse=True)
        ranked_fdpi = sorted(range(K), key=lambda k: centroids[k]["fdpi"], reverse=True)
        ranked_mfe  = sorted(range(K), key=lambda k: centroids[k]["mfe"],  reverse=False)

        assigned: Dict[int, str] = {}
        assigned[ranked_mfe [self.pc.compression_mfe_rank]]   = "compression"
        assigned[ranked_fdpi[self.pc.distribution_fdpi_rank]] = "distribution"
        assigned[ranked_fdpi[self.pc.accumulation_fdpi_rank]] = "accumulation"
        assigned[ranked_mom [self.pc.ignition_mom_rank]]      = "ignition"

        for k in range(K):
            self.phase_map[k] = assigned.get(k, "refresh")
            log.info(
                f"  Cluster {k} → '{self.phase_map[k]}' "
                f"(fdpi={centroids[k]['fdpi']:.3f} "
                f"mom={centroids[k]['mom']:.3f} "
                f"mfe={centroids[k]['mfe']:.3f})"
            )

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        if self.gmm is None: raise RuntimeError("Call fit() first.")
        clean = features_df.dropna()
        X     = self.scaler.transform(clean.values)
        pred  = self.gmm.predict(X)
        out   = pd.Series(pd.NA, index=features_df.index, dtype="Int64", name="cluster")
        out.loc[clean.index] = pred
        return out

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Chunked predict_proba to avoid a full 88M × 16 float64 transform array.

        scaler.transform() upcasts float32 → float64 in sklearn, creating an
        88M × 16 × 8 B = 11.3 GB allocation.  Processing 500K rows at a time
        keeps the per-chunk transform at 64 MB while the output proba array
        is accumulated incrementally (88M × 5 × 8 B = 3.5 GB total).
        """
        _CHUNK = 500_000
        clean  = features_df.dropna()
        vals   = clean.values   # float32 view — no copy
        parts  = []
        for start in range(0, len(vals), _CHUNK):
            X_chunk = self.scaler.transform(vals[start : start + _CHUNK])
            parts.append(self.gmm.predict_proba(X_chunk))
        return np.concatenate(parts, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# HMM Smoother  —  FIX 10: causal EMA replaces Viterbi
# ─────────────────────────────────────────────────────────────────────────────

class HMMSmoother:
    """
    Fix 10: Replace Viterbi (bidirectional DP, future look-ahead bias)
    with causal EMA smoothing of GMM soft probabilities.

    Why Viterbi leaks future data:
        The Viterbi algorithm finds the globally optimal state sequence by
        using BOTH the forward and backward pass over the full sequence.
        State assignment at time t therefore depends on observations at
        times t+1, t+2, … T.  Labels derived from these states embed
        future market knowledge into the training targets.

    Fix: EMA of GMM predict_proba() — at each tick t, only p(t) and
    the running EMA of p(0..t−1) are used.  No future observations.
    """

    def __init__(self):
        self.lc: LabelConfig = CFG.labels

    def smooth(self, features_df: pd.DataFrame,
               gmm: GMMClusterFitter) -> pd.Series:
        """
        Causal phase smoothing via exponential moving average.

        Args:
            features_df : 16-feature DataFrame (may contain leading NaN)
            gmm         : fitted GMMClusterFitter with predict_proba()

        Returns:
            pd.Series of smoothed phase cluster assignments (Int64)
        """
        if not self.lc.hmm_enabled:
            return gmm.predict(features_df)

        # Get soft cluster probabilities (T_clean, K) via chunked predict_proba
        # (avoids the 11 GB scaler.transform peak — see GMMClusterFitter.predict_proba)
        clean = features_df.dropna()
        proba = gmm.predict_proba(clean)   # (T_clean, K) float64

        # Causal EMA in-place — no extra DataFrame copies (saves 2 × 3.5 GB).
        # Formula: ema[t] = alpha * x[t] + (1 - alpha) * ema[t-1]
        # Only past observations enter ema[t] → no look-ahead bias.
        alpha     = self.lc.hmm_ema_alpha
        one_minus = 1.0 - alpha
        for t in range(1, len(proba)):
            proba[t] = alpha * proba[t] + one_minus * proba[t - 1]
        states = proba.argmax(axis=1)
        del proba

        out = pd.Series(pd.NA, index=features_df.index, dtype="Int64",
                        name="causal_state")
        out.loc[clean.index] = states
        log.info(f"Causal EMA smoothing applied (alpha={alpha}).  "
                 f"No future look-ahead — label integrity preserved.")
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Phase Transition Labeler  (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

class PhaseTransitionLabeler:
    SIGNAL_TRANSITIONS = frozenset([
        ("compression",  "ignition"),
        ("accumulation", "distribution"),
        ("ignition",     "refresh"),
        ("distribution", "accumulation"),
        ("compression",  "accumulation"),
    ])

    def __init__(self, phase_map: Dict[int, str]):
        self.phase_map = phase_map

    def generate(self, state_series: pd.Series) -> pd.Series:
        lk     = CFG.phase.lookforward_ticks
        phases = state_series.map(
            lambda c: (self.phase_map.get(int(c), "refresh") if isinstance(c, (int, float, np.integer)) else c) if pd.notna(c) else "refresh"
        )
        n      = len(phases); labels = np.zeros(n, dtype=np.float32)
        phase_arr = phases.values
        for k in range(1, lk+1):
            if k >= n: break
            cur  = phase_arr[:n-k]; nxt = phase_arr[k:]
            is_s = np.array(
                [(c, x) in self.SIGNAL_TRANSITIONS for c, x in zip(cur, nxt)],
                dtype=bool
            )
            labels[:n-k] = np.maximum(labels[:n-k], (cur != nxt) & is_s)
        result = pd.Series(labels, index=phases.index, dtype=np.float32)
        log.info(f"Phase labels: pos_rate={result.mean():.4f}")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Microstructure Triple Barrier Labeler  —  v3.3 (replaces circular GMM labels)
# ─────────────────────────────────────────────────────────────────────────────

class MicrostructureTripleBarrier:
    """
    Forward-looking label that is causally clean — cannot be reconstructed
    from current features, eliminating the GMM-label circularity.

    label[i] = 1  if within the next `lookforward` ticks:
      (a) mid-price moves up >= barrier  (price confirmation)
      (b) GMM phase == 'ignition'        (only if require_ignition=True)

    With require_ignition=False (default): pure price MTB, zero GMM dependency.
    With require_ignition=True: MTB + microstructure gate (future ignition phase).
    """

    def generate(
        self,
        raw_df:           pd.DataFrame,
        features_df:      pd.DataFrame,
        phase_s:          pd.Series,
        lookforward:      int   = 10,
        barrier:          float = 5e-5,
        require_ignition: bool  = False,
    ) -> pd.Series:
        mid = ((raw_df["Bid"].values.astype(np.float64) +
                raw_df["Ask"].values.astype(np.float64)) / 2)
        n           = len(mid)
        is_ignition = (phase_s.values == "ignition")
        labels      = np.zeros(n, dtype=np.float32)

        for k in range(1, lookforward + 1):
            if k >= n:
                break
            price_up = (mid[k:] - mid[:n - k]) >= barrier
            signal   = (price_up & is_ignition[k:]) if require_ignition else price_up
            labels[:n - k] = np.maximum(labels[:n - k], signal.astype(np.float32))

        pos_rate = labels.mean()
        log.info(
            f"MTB labels: pos_rate={pos_rate:.4f}  barrier={barrier:.5f}  "
            f"lookforward={lookforward}  require_ignition={require_ignition}"
        )
        if pos_rate < 0.05:
            log.warning("MTB pos_rate < 5% — consider lowering mtb_barrier_pips.")
        if pos_rate > 0.50:
            log.warning("MTB pos_rate > 50% — consider raising mtb_barrier_pips.")

        return pd.Series(labels, index=features_df.index, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Volatility Parity Labeler  —  FIX 11: bisection stuck detection
# ─────────────────────────────────────────────────────────────────────────────

class VolatilityParityLabeler:
    """
    Fix 11: Bisection convergence guard for quiet/low-volatility regimes.

    The 5-pip floor (min_target_pips) prevents K from being lowered enough
    to achieve 20% positive rate in quiet markets.  Without this fix, the
    bisection bracket collapses to a degenerate interval and ALL labels become
    0 by accident (the model just barely never reaches the floor target).

    With the fix: if |hi−lo| < bisect_floor_tol over 5 consecutive iterations,
    we recognise a "floor-pinned" regime, return all-zero labels with a warning,
    and set a flag so the caller can exclude this epoch from training.
    """

    def __init__(self):
        self.lc:          LabelConfig = CFG.labels
        self.data                     = CFG.data
        self.floor_pinned: bool       = False

    def _rolling_vol(self, df: pd.DataFrame) -> pd.Series:
        dur = max(1.0, (df["Tick_Time_ms"].iloc[-1]-df["Tick_Time_ms"].iloc[0])/1000)
        tps = max(1, int(len(df)/dur))
        win = max(20, int(self.lc.vol_window_s * tps))
        return df["Mid"].rolling(win, min_periods=10).std().fillna(df["Mid"].std())

    def _check_hits(self, mid: np.ndarray, sigma: np.ndarray, K: float) -> np.ndarray:
        tgt_pip = self.lc.min_target_pips * self.data.pip_size
        mae_pip = self.lc.max_mae_pips    * self.data.pip_size
        n = len(mid); labels = np.zeros(n, dtype=np.float32)
        for i in range(n-1):
            end   = min(i+500, n)
            fwd   = mid[i+1:end]
            if len(fwd) == 0: continue
            dist  = max(sigma[i]*K, tgt_pip)
            entry = mid[i]
            s_hit = np.argmax(fwd <= entry-mae_pip) if (fwd<=entry-mae_pip).any() else len(fwd)
            t_hit = np.argmax(fwd >= entry+dist)    if (fwd>=entry+dist).any()    else len(fwd)
            if (fwd >= entry+dist).any() and t_hit < s_hit:
                labels[i] = 1.0
        return labels

    def _calibrate_K(self, df: pd.DataFrame, sigma: pd.Series) -> float:
        """
        Binary search for K such that hit_rate ≈ target_pos_rate.
        Fix 11: stuck detection — exits early and sets floor_pinned flag
        when the bracket collapses (floor-regime with no convergence possible).
        """
        mid_arr = df["Mid"].values; sig_arr = sigma.values
        lo, hi  = 0.5, 10.0; target = self.lc.target_pos_rate
        floor_tol  = self.lc.bisect_floor_tol
        stuck_cnt  = 0; prev_range = hi - lo
        K = (lo + hi) / 2.0

        for _ in range(self.lc.bisect_iters):
            K    = (lo + hi) / 2.0
            rate = self._check_hits(mid_arr, sig_arr, K).mean()

            if abs(rate - target) < 0.005:
                log.info(f"VP bisection converged: K={K:.4f} rate={rate:.4f}")
                self.floor_pinned = False
                return float(K)

            # Stuck detection (Fix 11)
            cur_range = hi - lo
            if abs(cur_range - prev_range) < floor_tol:
                stuck_cnt += 1
                if stuck_cnt >= 5:
                    log.warning(
                        f"VP bisection FLOOR-PINNED: bracket [{lo:.4f},{hi:.4f}] "
                        f"collapsed (range={cur_range:.6f} < tol={floor_tol}). "
                        "This epoch is in a low-volatility regime where the "
                        f"{self.lc.min_target_pips}-pip floor prevents convergence. "
                        "Returning all-zero labels for this epoch."
                    )
                    self.floor_pinned = True
                    return float(K)   # caller checks self.floor_pinned
            else:
                stuck_cnt = 0
            prev_range = cur_range
            lo, hi = (K, hi) if rate > target else (lo, K)

        log.warning(f"VP bisection: max iterations reached. K={K:.4f}")
        return float(K)

    def generate(self, df: pd.DataFrame,
                 hurst_series: Optional[pd.Series] = None) -> pd.Series:
        self.floor_pinned = False
        sigma  = self._rolling_vol(df)
        h_adj  = (1.0 + self.lc.hurst_k_scale * float(hurst_series.dropna().mean())
                  if hurst_series is not None else 1.0)
        K      = self._calibrate_K(df, sigma * h_adj)
        if self.floor_pinned:
            # Floor-pinned regime: return all-zero labels instead of broken labels
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        labels = self._check_hits(df["Mid"].values, sigma.values, K)
        return pd.Series(labels, index=df.index, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TitanLabeler — Master orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class TitanLabeler:
    """
    Full labeling pipeline — all three fixes applied:
      Phase 3.5 halts on redundant features  (Fix 4)
      HMM replaced with causal EMA           (Fix 10)
      VP bisection detects floor-pinned      (Fix 11)
    """

    def __init__(self):
        self.phase35 = Phase35Validator()
        self.gmm     = GMMClusterFitter()
        self.smoother= HMMSmoother()
        self.pt      = None
        self.vp      = VolatilityParityLabeler()
        self.fitted  = False

    def fit_and_label(
        self,
        df:            pd.DataFrame,
        features_df:   pd.DataFrame,
        periods:       Optional[List[pd.DataFrame]]      = None,
        features_list: Optional[List[pd.DataFrame]]      = None,
        blend_ratio:   float                             = None,
    ) -> pd.Series:
        blend     = blend_ratio if blend_ratio is not None else CFG.labels.blend_ratio
        all_feats = [features_df] + (features_list or [])

        # Fix 4: halts with ValueError if |rho| > threshold
        self.phase35.run(all_feats)

        # Fit GMM
        self.gmm.fit(features_df)

        # Fix 10: causal EMA smoothing (no Viterbi look-ahead)
        smoothed = self.smoother.smooth(features_df, self.gmm)
        phase_s  = smoothed.map(
            lambda c: self.gmm.phase_map.get(int(c), "refresh") if pd.notna(c) else "refresh"
        )

        lc = CFG.labels
        self.mtb  = MicrostructureTripleBarrier()
        pt_labels = self.mtb.generate(
            raw_df           = df,
            features_df      = features_df,
            phase_s          = phase_s,
            lookforward      = CFG.phase.lookforward_ticks,
            barrier          = lc.mtb_barrier_pips * CFG.data.pip_size,
            require_ignition = lc.mtb_require_ignition,
        )

        if blend > 0.0:
            vp_labels = self.vp.generate(df, features_df.get("hurst", None))
            if self.vp.floor_pinned:
                log.warning("VP labels are all-zero (floor-pinned regime). "
                            "Using pure phase-transition labels for this epoch.")
                final = pt_labels.astype(np.float32)
            else:
                idx   = pt_labels.index.intersection(vp_labels.index)
                final = (
                    (1.0 - blend) * pt_labels.loc[idx] + blend * vp_labels.loc[idx]
                ).clip(0.0, 1.0).round().astype(np.float32)
                log.info(
                    f"Label blend: PT_pos={pt_labels.sum():.0f} "
                    f"VP_pos={vp_labels.sum():.0f} "
                    f"Blended_pos={final.sum():.0f}"
                )
        else:
            final = pt_labels.astype(np.float32)

        self.fitted = True
        log.info(f"TitanLabeler complete: pos_rate={final.mean():.4f} n={len(final)}")
        return final
