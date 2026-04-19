"""
titan_model.py — Titan V3.0 LSTM + Training  (Patch 3.2)
=========================================================
Fix 13 — Loss function: WeightedBCE as default (Focal Loss optional)

WHY Focal Loss is wrong for HFT:
  Focal Loss was designed for object detection (Lin et al., 2017) where
  "hard" examples are genuinely ambiguous objects with clean ground truth.
  In HFT microstructure data, the "hardest" examples to classify are:
    • Random noise ticks
    • Broker misprints / connectivity artifacts
    • Low-liquidity outlier quotes
  Focal Loss's (1-p_t)^γ weight exponentially amplifies attention to
  these noise events, forcing the LSTM to dedicate its weights to
  memorising random artefacts → guaranteed OOS overfitting.

  Weighted BCE with pos_weight is the correct tool:
    • Addresses class imbalance without amplifying noise
    • Stable training signal across all examples
    • Tested standard for imbalanced time-series classification

Configuration:
    use_focal_loss = False  (default) → WeightedBCELoss(pos_weight=4.0)
    use_focal_loss = True             → FocalLoss(alpha=0.75, gamma=2.0)
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from titan_config import CFG

log = logging.getLogger("TitanModel")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

class WeightedBCELoss(nn.Module):
    """
    Class-weighted Binary Cross-Entropy.

    Addresses class imbalance by up-weighting positive examples.
    Does NOT apply a focusing term — all examples contribute proportionally.
    More robust than Focal Loss on noisy financial microstructure data.

    pos_weight = N_neg / N_pos  (or set manually; default 4.0 for ~20% positive rate)
    """

    def __init__(self, pos_weight: float = None):
        super().__init__()
        self.pos_weight = pos_weight if pos_weight is not None else CFG.model.bce_pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = pred.clamp(1e-7, 1.0 - 1e-7)
        weight = torch.where(
            target.bool(),
            torch.full_like(target, self.pos_weight),
            torch.ones_like(target),
        )
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        return (weight * bce).mean()


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
    Use with caution on HFT data — amplifies noise examples.
    Kept for ablation studies; not the default.
    """

    def __init__(self, alpha: float = None, gamma: float = None):
        super().__init__()
        self.alpha = alpha or CFG.model.focal_alpha
        self.gamma = gamma or CFG.model.focal_gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred    = pred.clamp(1e-7, 1.0 - 1e-7)
        bce     = -(target * torch.log(pred) + (1-target)*torch.log(1-pred))
        p_t     = pred * target + (1-pred)*(1-target)
        alpha_t = self.alpha * target + (1-self.alpha)*(1-target)
        return (alpha_t * (1-p_t)**self.gamma * bce).mean()


def build_loss_fn() -> nn.Module:
    """
    Returns the configured loss function.
    Default: WeightedBCELoss (robust for noisy HFT data).
    Override: set CFG.model.use_focal_loss = True for ablation studies.
    """
    if CFG.model.use_focal_loss:
        log.info(
            f"Using FocalLoss(alpha={CFG.model.focal_alpha}, "
            f"gamma={CFG.model.focal_gamma}).  "
            "WARNING: Focal Loss amplifies hard (noisy) examples — "
            "use WeightedBCE for production."
        )
        return FocalLoss()
    else:
        log.info(
            f"Using WeightedBCELoss(pos_weight={CFG.model.bce_pos_weight}).  "
            "Robust to noisy HFT microstructure data."
        )
        return WeightedBCELoss()


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Architecture
# ─────────────────────────────────────────────────────────────────────────────

class TitanLSTMV3(nn.Module):
    """
    Sequence-to-scalar LSTM for algo-phase completion detection.
    Input  : (batch, 128, 16)  all features ∈ [−1, +1]
    Output : (batch,)  conviction score ∈ [0, 1]
    """

    def __init__(self, input_size=None, hidden_size=None, num_layers=None,
                 lstm_drop=None, fc_drop=None, fc_hidden=None):
        super().__init__()
        mc = CFG.model
        self.input_size  = input_size  or mc.input_size    # 16
        self.hidden_size = hidden_size or mc.hidden_size
        self.num_layers  = num_layers  or mc.num_layers
        _ld = lstm_drop if lstm_drop is not None else mc.lstm_dropout
        _fd = fc_drop   if fc_drop   is not None else mc.fc_dropout
        _fh = fc_hidden or mc.fc_hidden

        self.lstm = nn.LSTM(
            input_size  = self.input_size,
            hidden_size = self.hidden_size,
            num_layers  = self.num_layers,
            batch_first = True,
            dropout     = _ld if self.num_layers > 1 else 0.0,
        )
        self.bn    = nn.BatchNorm1d(self.hidden_size)
        self.drop1 = nn.Dropout(_fd)
        self.fc1   = nn.Linear(self.hidden_size, _fh)
        self.relu  = nn.ReLU()
        self.drop2 = nn.Dropout(_fd)
        self.fc2   = nn.Linear(_fh, 1)
        self.sig   = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p.data)
            elif "bias" in name:
                p.data.fill_(0.0)
                n = p.size(0)
                p.data[n//4:n//2].fill_(1.0)  # forget gate bias = 1
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _  = self.lstm(x)
        last    = self.bn(out[:, -1, :])
        last    = self.drop1(last)
        last    = self.relu(self.fc1(last))
        last    = self.drop2(last)
        return self.sig(self.fc2(last)).squeeze(-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Single fold training
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    fold_idx:   int
    train_loss: float
    val_loss:   float
    val_auc:    float
    best_epoch: int
    model_path: Path


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2: return 0.5
    return float(roc_auc_score(y_true, y_score))


def train_single_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    fold: int,
    device: torch.device,
) -> FoldResult:
    mc = CFG.model
    seed_everything(mc.seed + fold)

    model = TitanLSTMV3().to(device)
    # === FORENSIC: Architecture logging ===
    log.info(
        f"Fold {fold} TRAINING — Model architecture: "
        f"input_size={model.input_size} hidden_size={model.hidden_size} "
        f"num_layers={model.num_layers} (from CFG.model defaults)"
    )
    # Fix 13: configurable loss — WeightedBCE default
    loss_fn   = build_loss_fn()
    optimizer = optim.AdamW(model.parameters(), lr=mc.lr, weight_decay=mc.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=mc.t0, T_mult=mc.t_mult
    )
    log.info(f"Fold {fold}: params={model.count_params():,} device={device} "
             f"loss={'Focal' if mc.use_focal_loss else 'WeightedBCE'}")

    use_amp  = mc.use_amp and device.type == "cuda"
    scaler   = torch.cuda.amp.GradScaler(enabled=use_amp)
    dl_tr    = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=mc.batch_size, shuffle=True, drop_last=True, num_workers=0,
        pin_memory=(device.type=="cuda"),
    )
    dl_va    = DataLoader(
        TensorDataset(torch.tensor(X_va, dtype=torch.float32),
                      torch.tensor(y_va, dtype=torch.float32)),
        batch_size=mc.batch_size, shuffle=False, drop_last=False, num_workers=0,
    )

    best_val  = float("inf"); best_ep = 0; patience = 0
    ckpt_path = Path(CFG.data.model_dir) / f"titan_v3_fold{fold}.pt"

    for epoch in range(1, mc.max_epochs+1):
        model.train(); tr_loss = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), mc.grad_clip)
            scaler.step(optimizer); scaler.update()
            tr_loss += loss.item()
        scheduler.step()

        model.eval(); va_loss = 0.0; preds: list = []; gt: list = []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                p = model(xb)
                va_loss += loss_fn(p, yb).item()
                preds.extend(p.cpu().tolist()); gt.extend(yb.cpu().tolist())
        va_loss /= max(1, len(dl_va))
        va_auc   = _auc(np.array(gt), np.array(preds))

        if epoch % 10 == 0 or epoch == 1:
            log.info(f"  Fold {fold} Ep {epoch:3d} "
                     f"tr={tr_loss/len(dl_tr):.4f} "
                     f"va={va_loss:.4f} auc={va_auc:.4f}")

        if va_loss < best_val - 1e-5:
            best_val, best_ep, patience = va_loss, epoch, 0
            torch.save({"epoch":epoch,"model":model.state_dict(),
                        "val_loss":va_loss,"val_auc":va_auc}, ckpt_path)
            if epoch == 1 or epoch % 50 == 0:
                log.info(
                    f"    Checkpoint saved: {ckpt_path.name} "
                    f"(epoch={epoch} loss={va_loss:.4f} auc={va_auc:.4f})"
                )
        else:
            patience += 1
            if patience >= mc.patience:
                log.info(f"  Early stop ep={epoch} best={best_ep}")
                break

    # === FORENSIC: Verify checkpoint is valid ===
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        if "model" not in ckpt:
            raise KeyError(f"Checkpoint missing 'model' key. Keys: {list(ckpt.keys())}")
        # Quick validation: verify state_dict keys match model
        missing_keys, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing_keys:
            log.warning(
                f"Fold {fold} checkpoint missing keys: {missing_keys[:3]}... "
                "(May cause issues during final model load)"
            )
    except Exception as e:
        raise RuntimeError(
            f"Fold {fold}: Checkpoint validation failed at {ckpt_path}. "
            f"Training produced a corrupted or incomplete checkpoint.\n{e}"
        ) from e

    best_auc = ckpt["val_auc"]
    return FoldResult(fold, tr_loss/max(1,len(dl_tr)),
                      best_val, best_auc, best_ep, ckpt_path)


# ─────────────────────────────────────────────────────────────────────────────
# WFO Trainer
# ─────────────────────────────────────────────────────────────────────────────

class WFOTrainer:
    def __init__(self):
        self.vc           = CFG.validation
        self.mc           = CFG.model
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fold_results: List[FoldResult] = []

    def run(self, X: np.ndarray, y: np.ndarray) -> Tuple[TitanLSTMV3, List[FoldResult]]:
        n = len(X); fs = n // self.vc.n_wfo_folds; self.fold_results = []
        for k in range(self.vc.n_wfo_folds):
            s = k*fs; e = s+fs if k < self.vc.n_wfo_folds-1 else n
            Xf, yf = X[s:e], y[s:e]; nt = int(len(Xf)*self.vc.train_pct)
            if nt < self.mc.batch_size*4:
                log.warning(f"Fold {k}: insufficient data, skipping."); continue
            log.info(f"\nFold {k}/{self.vc.n_wfo_folds-1}: train={nt} val={len(Xf)-nt}")
            self.fold_results.append(
                train_single_fold(Xf[:nt], yf[:nt], Xf[nt:], yf[nt:], k, self.device)
            )
        if not self.fold_results: raise RuntimeError("All folds failed.")
        best = max(self.fold_results, key=lambda r: r.val_auc)
        log.info(f"\nBest fold {best.fold_idx}: val_auc={best.val_auc:.4f}")

        # === FORENSIC WEIGHT LOADING ===
        # Verify checkpoint file exists before attempting load
        best_path = best.model_path
        if not best_path.exists():
            raise FileNotFoundError(
                f"Best fold checkpoint missing: {best_path}\n"
                f"Training may have failed to save. Check Phase 5 logs."
            )

        # Load checkpoint and verify structure
        ckpt = torch.load(best_path, map_location=self.device)
        if "model" not in ckpt:
            raise KeyError(
                f"Checkpoint at {best_path} missing 'model' key. "
                f"Keys present: {list(ckpt.keys())}. Checkpoint corrupted?"
            )

        # Instantiate fresh model with EXACT same defaults as training
        model = TitanLSTMV3().to(self.device)  # Uses CFG.model defaults
        log.info(
            f"Model architecture: input_size={model.input_size} "
            f"hidden_size={model.hidden_size} num_layers={model.num_layers}"
        )

        # Load state dict with strict checking
        try:
            model.load_state_dict(ckpt["model"], strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"State dict mismatch in fold {best.fold_idx}. "
                f"Architecture mismatch between training and loading.\n{e}"
            ) from e

        model.eval()
        log.info(
            f"Loaded best fold {best.fold_idx}: "
            f"epoch={ckpt['epoch']} val_loss={ckpt['val_loss']:.4f} "
            f"val_auc={ckpt['val_auc']:.4f}"
        )
        return model, self.fold_results

    def wfo_efficiency(self) -> float:
        if not self.fold_results: return 0.0
        oos     = [r.val_auc for r in self.fold_results]
        best_is = max((r.val_auc for r in self.fold_results), default=0.0)
        if best_is < 1e-10:
            log.warning("WFO efficiency: IS near zero."); return 0.0
        return float(np.mean(oos) / best_is)


def predict_batch(model: TitanLSTMV3, X: np.ndarray,
                  device: torch.device = None, batch: int = 512) -> np.ndarray:
    if device is None: device = next(model.parameters()).device
    model.eval(); out = []
    with torch.inference_mode():
        for s in range(0, len(X), batch):
            xb = torch.tensor(X[s:s+batch], dtype=torch.float32).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)
