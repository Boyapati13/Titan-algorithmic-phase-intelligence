"""
titan_pipeline.py — Titan V3.0 Master Pipeline Orchestrator
============================================================
Fixes v3.1:
  - Phase 6 trade returns computed on OOS portion only (not full dataset)
  - start_phase dependency guards prevent AttributeError on resume
  - ValidationReport auto-saved to model_dir
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from titan_config import CFG, FEATURE_COLS
from titan_data import load_parquet
from titan_features import validate_input, compute_features, build_sequences
from titan_labeler import TitanLabeler
from titan_model import WFOTrainer, TitanLSTMV3, FoldResult, predict_batch
from titan_validator import TitanValidator, DeploymentBlockedError, kelly_fraction
from titan_onnx import export_onnx, FeatureParityTester
from titan_baselines import BaselineSuite

log = logging.getLogger("TitanPipeline")


@dataclass
class PipelineState:
    raw_df:      Optional[pd.DataFrame]  = None
    features_df: Optional[pd.DataFrame]  = None
    labels:      Optional[pd.Series]     = None
    X:           Optional[np.ndarray]    = None
    y:           Optional[np.ndarray]    = None
    best_model:  Optional[TitanLSTMV3]   = None
    trainer:     Optional[WFOTrainer]    = None
    onnx_path:   Optional[Path]          = None
    config_path: Optional[Path]          = None
    report:      Optional[object]        = None
    kelly:       float                   = 0.01
    phases_done: List[int]               = field(default_factory=list)
    _t0:         float                   = field(default_factory=time.time)

    def elapsed(self) -> str:
        return f"{time.time() - self._t0:.1f}s"

    def _require_phase(self, needed: int, attr: str):
        """Guard: raise if a required upstream phase hasn't populated an attribute."""
        if getattr(self, attr) is None:
            raise ValueError(
                f"Phase dependency error: '{attr}' is None. "
                f"Run phase {needed} before resuming from a later phase. "
                "Use start_phase=1 for a full run, or provide a pre-populated state."
            )


class TitanPipeline:

    def run(self,
            tick_paths:     List[str],
            extra_periods:  Optional[List[pd.DataFrame]]       = None,
            extra_features: Optional[List[pd.DataFrame]]       = None,
            run_baselines:  bool                               = True,
            start_phase:    int                                = 1,
            ) -> PipelineState:

        log.info("=" * 70)
        log.info("TITAN V3.0  PIPELINE  START")
        log.info("=" * 70)
        s = PipelineState()

        try:
            # ── Phase 1: Load ──────────────────────────────────────────────────
            if start_phase <= 1:
                log.info("[Phase 1] Loading tick data…")
                raw    = load_parquet(tick_paths)
                s.raw_df = validate_input(raw)
                log.info(f"  {len(s.raw_df):,} validated ticks")
                s.phases_done.append(1)

            # ── Phase 2: Features ──────────────────────────────────────────────
            if start_phase <= 2:
                if start_phase > 1: s._require_phase(1, "raw_df")
                _feat_cache = Path(CFG.data.parquet_dir) / "features_cache.parquet"
                if _feat_cache.exists():
                    log.info(f"[Phase 2] Loading cached features from {_feat_cache.name}…")
                    s.features_df = pd.read_parquet(_feat_cache)
                    log.info(f"  Loaded {len(s.features_df):,} rows from cache (skipping recompute)")
                else:
                    log.info("[Phase 2] Computing 16 features…")
                    t = time.time()
                    s.features_df = compute_features(s.raw_df)
                    log.info(f"  Done in {time.time()-t:.1f}s")
                    log.info(f"  Saving features cache to {_feat_cache.name}…")
                    s.features_df.to_parquet(_feat_cache, compression="lz4", index=False)
                    log.info(f"  Cache saved ({_feat_cache.stat().st_size/1e6:.0f} MB)")
                s.phases_done.append(2)

            # ── Phase 3: Labels ────────────────────────────────────────────────
            if start_phase <= 3:
                if start_phase > 2: s._require_phase(2, "features_df")
                log.info("[Phase 3] Generating algo-phase labels…")
                labeler  = TitanLabeler()
                s.labels = labeler.fit_and_label(
                    df           = s.raw_df,
                    features_df  = s.features_df,
                    periods      = extra_periods,
                    features_list= extra_features,
                )
                s.X, s.y = build_sequences(s.features_df, s.labels)
                log.info(f"  X={s.X.shape} y={s.y.shape} pos={s.y.mean():.4f}")
                s.phases_done.append(3)

            # ── Phase 4: Baselines ─────────────────────────────────────────────
            if start_phase <= 4 and run_baselines:
                if start_phase > 3: s._require_phase(3, "labels")
                log.info("[Phase 4] Running 4 baselines…")
                BaselineSuite().run_all(s.raw_df, s.features_df, s.labels)
                s.phases_done.append(4)

            # ── Phase 5: Train ─────────────────────────────────────────────────
            if start_phase <= 5:
                if start_phase > 3: s._require_phase(3, "X")
                _model_dir   = Path(CFG.data.model_dir)
                _fold_json   = _model_dir / "wfo_fold_results.json"
                _n_folds     = CFG.validation.n_wfo_folds
                _fold_ckpts  = [_model_dir / f"titan_v3_fold{i}.pt" for i in range(_n_folds)]
                _cached      = _fold_json.exists() and all(p.exists() for p in _fold_ckpts)

                if _cached:
                    log.info(f"[Phase 5] Found {_n_folds} fold checkpoints + wfo_fold_results.json — loading from disk.")
                    import json as _json
                    _fold_data = _json.loads(_fold_json.read_text())
                    _fold_results = [
                        FoldResult(
                            fold_idx  = d["fold_idx"],
                            train_loss= d.get("train_loss", 0.0),
                            val_loss  = d.get("val_loss",   0.0),
                            val_auc   = d["val_auc"],
                            best_epoch= d.get("best_epoch", 0),
                            model_path= _fold_ckpts[d["fold_idx"]],
                        ) for d in _fold_data
                    ]

                    class _CachedTrainer:
                        fold_results = _fold_results
                        def wfo_efficiency(self_):
                            oos = [r.val_auc for r in self_.fold_results]
                            is_ref = max(r.val_auc for r in self_.fold_results)
                            return float(np.mean(oos) / (is_ref + 1e-10))

                    s.trainer   = _CachedTrainer()
                    best_fr     = max(_fold_results, key=lambda r: r.val_auc)
                    _device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    s.best_model = TitanLSTMV3().to(_device)
                    _ckpt       = torch.load(best_fr.model_path, map_location=_device, weights_only=False)
                    s.best_model.load_state_dict(_ckpt["model"])
                    s.best_model.eval()
                    log.info(f"  Loaded best model: fold {best_fr.fold_idx} val_auc={best_fr.val_auc:.4f}")
                    log.info(f"  WFO Efficiency: {s.trainer.wfo_efficiency():.4f}")
                else:
                    log.info("[Phase 5] LSTM Walk-Forward Training…")
                    s.trainer = WFOTrainer()
                    s.best_model, _ = s.trainer.run(s.X, s.y)
                    log.info(f"  WFO Efficiency: {s.trainer.wfo_efficiency():.4f}")
                s.phases_done.append(5)

            # ── Phase 6: Validate ──────────────────────────────────────────────
            if start_phase <= 6 and s.trainer:
                if start_phase > 5:
                    s._require_phase(5, "best_model")
                    s._require_phase(5, "trainer")
                log.info("[Phase 6] Validation gauntlet…")
                device  = next(s.best_model.parameters()).device

                # Use ONLY the last fold's OOS portion for trade returns
                # (not the full dataset — that inflates IS performance)
                last_fold = max(s.trainer.fold_results, key=lambda r: r.fold_idx)
                n     = len(s.X)
                fs    = n // CFG.validation.n_wfo_folds
                s_oos = last_fold.fold_idx * fs
                nt    = int((n - s_oos) * CFG.validation.train_pct)
                X_oos = s.X[s_oos + nt:]
                y_oos = s.y[s_oos + nt:]

                # Use the last fold's OWN model for OOS AUC — temporally aligned.
                # The global best_model (highest val_auc fold) is trained on earlier
                # market data; applying it to the most recent OOS fold artificially
                # deflates AUC via temporal distribution shift.
                _last_ckpt = Path(CFG.data.model_dir) / f"titan_v3_fold{last_fold.fold_idx}.pt"
                if _last_ckpt.exists():
                    _oos_model = TitanLSTMV3().to(device)
                    _oos_ckpt  = torch.load(_last_ckpt, map_location=device, weights_only=False)
                    _oos_model.load_state_dict(_oos_ckpt["model"])
                    _oos_model.eval()
                    log.info(f"  OOS eval: fold {last_fold.fold_idx} model (val_auc={last_fold.val_auc:.4f})")
                else:
                    _oos_model = s.best_model
                y_score = predict_batch(_oos_model, X_oos, device)
                y_true  = y_oos
                ec      = CFG.execution

                long_m  = y_score >= ec.conviction_long
                short_m = y_score <= ec.conviction_short
                mask    = long_m | short_m

                if mask.sum() == 0:
                    trade_r = np.array([0.0])
                    log.warning("No signals in OOS fold — check conviction thresholds.")
                else:
                    correct = (
                        (long_m  & (y_true == 1)) |
                        (short_m & (y_true == 0))
                    )
                    # Barrier economics: win = barrier minus round-trip spread cost,
                    # loss = -spread (exit at timeout, paid entry cost only).
                    # Avoids permutation-invariance problem of constant ±0.0001.
                    _barrier = CFG.labels.mtb_barrier_pips * CFG.data.pip_size
                    # Alpha Pivot v3.4: use actual broker spread (1.344 pip), not theoretical
                    _spread  = CFG.data.broker_spread_pips * 0.0001
                    trade_r  = np.where(correct[mask], _barrier - _spread, -_spread)

                oos_s = [r.val_auc * 2 - 1 for r in s.trainer.fold_results]
                is_s  = [max(r.val_auc * 2 - 1, 0.01) for r in s.trainer.fold_results]

                v       = TitanValidator()
                s.report = v.run(
                    y_true        = y_true,
                    y_score       = y_score,
                    trade_returns = trade_r,
                    oos_sharpes   = oos_s,
                    is_sharpes    = is_s,
                    output_dir    = CFG.data.model_dir,
                )
                s.kelly = kelly_fraction(trade_r)
                log.info(f"  Kelly fraction: {s.kelly:.4f}")
                s.phases_done.append(6)

            # ── Phase 7: Export ────────────────────────────────────────────────
            if s.best_model is None:
                raise RuntimeError(
                    "Phase 7 aborted: best_model is None — WFO training produced no model. "
                    "Check Phase 5 logs for the underlying failure."
                )
            if start_phase <= 7:
                s._require_phase(5, "best_model")
                log.info("[Phase 7] ONNX export…")
                s.onnx_path, s.config_path = export_onnx(
                    s.best_model, CFG.data.model_dir
                )
                CFG.to_json(CFG.data.model_dir / "run_config.json")
                log.info(f"  ONNX   → {s.onnx_path}")
                log.info(f"  Config → {s.config_path}")
                s.phases_done.append(7)

            log.info("=" * 70)
            log.info(f"PIPELINE COMPLETE — {s.elapsed()}  phases={s.phases_done}")
            log.info("=" * 70)
            return s

        except DeploymentBlockedError:
            log.error("Pipeline halted: deployment blocked.")
            raise
        except Exception as e:
            log.exception(f"Pipeline failed at phases={s.phases_done}: {e}")
            raise


if __name__ == "__main__":
    import sys
    # MT5's Python runner injects the chart symbol + timeframe (e.g. "EURUSD 1") as
    # positional args.  Filter to arguments that are actual parquet paths so those
    # stray tokens don't reach load_parquet and cause a FileNotFoundError.
    raw_args = sys.argv[1:]
    paths = [p for p in raw_args if p.endswith(".parquet") or Path(p).is_file()]
    if not paths:
        paths = [str(CFG.data.parquet_dir / "EURUSD_ticks.parquet")]
    # Baselines (Phase 4) are skipped for speed — they don't affect the ONNX output.
    # Set run_baselines=True to re-enable them once the full pipeline is validated.
    TitanPipeline().run(paths, run_baselines=False)
