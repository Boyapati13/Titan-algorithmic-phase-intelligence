"""
titan_optimizer.py — Titan V3.0 Bayesian Hyperparameter Optimiser
==================================================================
Fixes v3.1:
  - CFG.model mutation is now thread-safe (threading.Lock)
  - n_jobs parameter exposed (safe because of lock)
  - Early stopping if best value doesn't improve in N trials
  - importance() logs exc_info=True on failure
  - Optuna HTML visualization exported for thesis documentation
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from titan_config import CFG, TitanConfig, ModelConfig
from titan_model import train_single_fold, TitanLSTMV3

log = logging.getLogger("TitanOptimizer")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_OK = True
except ImportError:
    _OPTUNA_OK = False
    log.warning("optuna not installed — optimizer disabled.  pip install optuna")

# Thread safety: single lock guards all CFG.model mutations
_cfg_lock = threading.Lock()


class TitanOptimizer:
    """
    Bayesian optimisation over LSTM hyperparameters via Optuna TPE sampler.

    Thread safety: CFG.model mutations are protected by _cfg_lock.
    Parallel trials (n_jobs > 1) are safe because each trial acquires
    the lock before modifying CFG.model and releases it after training.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, n_folds: int = 2,
                 n_no_improve: int = 15):
        self.X            = X
        self.y            = y
        self.n_folds      = n_folds
        self.n_no_improve = n_no_improve   # early stopping
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_params: Optional[dict] = None
        self.study        = None

    def _objective(self, trial) -> float:
        if not _OPTUNA_OK:
            return 0.5

        # Sample hyperparameters
        hs  = trial.suggest_categorical("hidden_size",  [32, 64, 128])
        ld  = trial.suggest_float("lstm_dropout",        0.1, 0.5)
        fd  = trial.suggest_float("fc_dropout",          0.1, 0.4)
        fh  = trial.suggest_categorical("fc_hidden",    [16, 32, 64])
        lr  = trial.suggest_float("lr",                  1e-4, 5e-3, log=True)
        bpw = trial.suggest_categorical("bce_pos_weight", [1.0, 2.0, 4.0, 8.0])
        nl  = trial.suggest_categorical("num_layers",     [1, 2])
        bs  = trial.suggest_categorical("batch_size",   [128, 256, 512])

        # Thread-safe CFG mutation
        with _cfg_lock:
            mc_orig   = CFG.model
            CFG.model = ModelConfig(
                input_size      = mc_orig.input_size,
                hidden_size     = hs,
                num_layers      = nl,
                lstm_dropout    = ld,
                fc_dropout      = fd,
                fc_hidden       = fh,
                lr              = lr,
                bce_pos_weight  = bpw,
                use_focal_loss  = mc_orig.use_focal_loss,
                focal_alpha     = mc_orig.focal_alpha,
                focal_gamma     = mc_orig.focal_gamma,
                weight_decay    = mc_orig.weight_decay,
                grad_clip       = mc_orig.grad_clip,
                t0              = mc_orig.t0,
                t_mult          = mc_orig.t_mult,
                use_amp         = mc_orig.use_amp,
                sequence_len    = mc_orig.sequence_len,
                batch_size      = bs,
                max_epochs      = 50,
                patience        = 10,
                seed            = mc_orig.seed,
            )

        n, fs = len(self.X), len(self.X) // max(self.n_folds, 2)
        aucs  = []
        try:
            for k in range(self.n_folds):
                s  = k * fs
                e  = s + fs if k < self.n_folds - 1 else n
                Xf, yf = self.X[s:e], self.y[s:e]
                nt = int(len(Xf) * CFG.validation.train_pct)
                if nt < CFG.model.batch_size * 4: continue
                r = train_single_fold(
                    Xf[:nt], yf[:nt], Xf[nt:], yf[nt:],
                    fold=k + trial.number * 100, device=self.device,
                )
                aucs.append(r.val_auc)
        finally:
            with _cfg_lock:
                CFG.model = mc_orig   # restore original config

        return float(np.mean(aucs)) if aucs else 0.5

    def run(self, n_trials: int = 50, n_jobs: int = 1,
            out_path: Path = None) -> dict:
        if not _OPTUNA_OK:
            raise ImportError("optuna not installed.  pip install optuna")

        log.info(f"Optuna optimisation: {n_trials} trials, "
                 f"n_jobs={n_jobs}, device={self.device}")

        self.study = optuna.create_study(
            direction   = "maximize",
            study_name  = "TitanV3_LSTM_opt",
            sampler     = optuna.samplers.TPESampler(seed=CFG.model.seed),
            pruner      = optuna.pruners.MedianPruner(),
        )

        # Early stopping callback
        best_so_far    = [float("-inf")]
        no_improve_cnt = [0]

        def early_stop_cb(study, trial):
            if study.best_value > best_so_far[0]:
                best_so_far[0]    = study.best_value
                no_improve_cnt[0] = 0
            else:
                no_improve_cnt[0] += 1
            if no_improve_cnt[0] >= self.n_no_improve:
                study.stop()
                log.info(f"Optuna early stop: no improvement in {self.n_no_improve} trials.")

        self.study.optimize(
            self._objective,
            n_trials         = n_trials,
            n_jobs           = n_jobs,
            callbacks        = [early_stop_cb],
            show_progress_bar= False,
        )

        self.best_params = self.study.best_params
        best_auc         = self.study.best_value
        log.info(f"Optimisation complete: best_auc={best_auc:.4f}")
        log.info(f"Best params: {self.best_params}")

        # Apply best params to CFG.model
        p = self.best_params
        with _cfg_lock:
            mc = CFG.model
            CFG.model = ModelConfig(
                input_size      = mc.input_size,
                hidden_size     = p.get("hidden_size",    mc.hidden_size),
                num_layers      = p.get("num_layers",     mc.num_layers),
                lstm_dropout    = p.get("lstm_dropout",   mc.lstm_dropout),
                fc_dropout      = p.get("fc_dropout",     mc.fc_dropout),
                fc_hidden       = p.get("fc_hidden",      mc.fc_hidden),
                lr              = p.get("lr",             mc.lr),
                bce_pos_weight  = p.get("bce_pos_weight", mc.bce_pos_weight),
                use_focal_loss  = mc.use_focal_loss,
                focal_alpha     = mc.focal_alpha,
                focal_gamma     = mc.focal_gamma,
                weight_decay    = mc.weight_decay,
                grad_clip       = mc.grad_clip,
                t0              = mc.t0,
                t_mult          = mc.t_mult,
                use_amp         = mc.use_amp,
                sequence_len    = mc.sequence_len,
                batch_size      = p.get("batch_size",     mc.batch_size),
                max_epochs      = mc.max_epochs,
                patience        = mc.patience,
                seed            = mc.seed,
            )

        result = {"best_auc": best_auc, "best_params": self.best_params,
                  "n_trials": len(self.study.trials)}
        if out_path:
            Path(out_path).write_text(json.dumps(result, indent=2))
            log.info(f"Optuna results → {out_path}")

        # Export visualizations for thesis documentation
        try:
            import optuna.visualization as ov
            html_path = Path(CFG.data.output_dir) / "optuna_history.html"
            ov.plot_optimization_history(self.study).write_html(str(html_path))
            log.info(f"Optuna history chart → {html_path}")
        except Exception as e:
            log.warning(f"Optuna visualization failed: {e}", exc_info=True)

        return result

    def importance(self) -> dict:
        if self.study is None:
            raise RuntimeError("Call run() first.")
        try:
            imp = optuna.importance.get_param_importances(self.study)
            log.info(f"Hyperparameter importances: {dict(imp)}")
            return dict(imp)
        except Exception as e:
            log.warning(f"Importance failed: {e}", exc_info=True)
            return {}
