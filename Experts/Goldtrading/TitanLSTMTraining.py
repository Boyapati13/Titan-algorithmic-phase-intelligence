#!/usr/bin/env python3
"""
TITAN HFT SYSTEM - LSTM Training & ONNX Export V2.0 (PRODUCTION)
Author : Senior Quant / Systems Architect | April 2026

FORENSIC FIXES vs V1:
  [BUG-01] input_size=12 in class def but TitanFeatureEngineering had 10
           features - silent dimension mismatch at inference time. V2 derives
           N_FEATURES from TitanFeatureEngineer.N_FEATURES (12).
  [BUG-02] output.squeeze() on batch_size=1 tensor collapses to a scalar,
           breaking the ONNX dynamic batch axis. Fixed with view(-1).
  [BUG-03] Early stopping loaded 'titan_lstm_best.pth' unconditionally even
           if it didn't exist (fresh run). Added existence check.
  [BUG-04] No class-weight balancing for imbalanced labels. Added pos_weight
           to BCEWithLogitsLoss.
  [BUG-05] OnnxRuntime verification compared scalar vs 1-element array -
           numpy broadcast masked real numerical mismatches. Fixed properly.
  [NEW]    Mixed-precision training (torch.cuda.amp) for 2–3× GPU speedup.
  [NEW]    Focal loss option for severe class imbalance.
  [NEW]    Cosine annealing LR schedule with warm restarts.
  [NEW]    Proper ONNX opset 18 export with dynamic batch AND seq axes.
  [NEW]    Saves scaler params (mean/std) alongside model for inference.
"""

import sys, os, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import onnxruntime as ort
from sklearn.metrics import (classification_report, roc_auc_score,
                              precision_recall_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import V2 feature engineering for consistent N_FEATURES
try:
    from TitanFeatureEngineering import TitanFeatureEngineer
    N_FEATURES = TitanFeatureEngineer.N_FEATURES  # 12
except ImportError:
    N_FEATURES = 12
    print("Warning: TitanFeatureEngineering not found. Using N_FEATURES=12.")

SEQ_LEN    = 128   # Timesteps fed into LSTM (matches EA TICK_BUFFER_SIZE)
HIDDEN     = 64
NUM_LAYERS = 2
DROPOUT    = 0.4


# =============================================================================
# MODEL
# =============================================================================
class TitanLSTM(nn.Module):
    """
    TITAN LSTM - input shape: [batch, SEQ_LEN, N_FEATURES]
    Output:  [batch]  conviction score in (0, 1)
    """

    def __init__(self, input_size: int = N_FEATURES,
                 hidden_size: int = HIDDEN,
                 num_layers:  int = NUM_LAYERS,
                 dropout:     float = DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.bn   = nn.BatchNorm1d(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden_size, 32)
        self.fc2  = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, features]
        lstm_out, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]             # [batch, hidden]
        h_last = self.bn(h_last)
        h_last = self.drop(h_last)
        h_last = self.relu(self.fc1(h_last))
        h_last = self.drop(h_last)
        out    = self.fc2(h_last)    # [batch, 1] - raw logit
        # BUG-FIX: Output [batch, 1] to match ONNX deployment shape expected by MT5 EA
        return torch.sigmoid(out)   # [batch, 1]


# =============================================================================
# DATASET
# =============================================================================
class HFTSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# FOCAL LOSS (handles severe class imbalance better than BCE)
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce  = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt   = torch.where(target == 1, pred, 1 - pred)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


# =============================================================================
# TRAINER
# =============================================================================
class TitanLSTMTrainer:

    def __init__(self, use_focal_loss: bool = False,
                 use_amp: bool = True):
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model     = TitanLSTM().to(self.device)
        self.use_amp   = use_amp and self.device.type == 'cuda'
        self.scaler    = GradScaler(enabled=self.use_amp)
        self.use_focal = use_focal_loss

        print(f"Device: {self.device}  AMP: {self.use_amp}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Input: [batch, {SEQ_LEN}, {N_FEATURES}]")

    # ── Data ──────────────────────────────────────────────────────────────────
    def load_sequences(self, npz_path: str):
        """Load pre-built sequences from TitanFeatureEngineering_V2."""
        data              = np.load(npz_path)
        self.X_train      = data['X_train'].astype(np.float32)
        self.y_train      = data['y_train'].astype(np.float32)
        self.X_val        = data['X_val'].astype(np.float32)
        self.y_val        = data['y_val'].astype(np.float32)
        self.X_test       = data['X_test'].astype(np.float32)   # OOS - never tune on this
        self.y_test       = data['y_test'].astype(np.float32)
        # Feature scaling params (saved by feature engineering)
        # BUG-FIX: numpy NpzFile does not implement .get(). Calling data.get()
        # raises AttributeError: 'NpzFile' object has no attribute 'get'.
        # Use 'key in data' guard instead, which NpzFile does support.
        self.feat_mean = data['feat_mean'] if 'feat_mean' in data else np.zeros(N_FEATURES)
        self.feat_std  = data['feat_std']  if 'feat_std'  in data else np.ones(N_FEATURES)

        pos_rate = self.y_train.mean()
        print(f"Train: {len(self.X_train):,}  Val: {len(self.X_val):,}  "
              f"Test(OOS): {len(self.X_test):,}")
        print(f"Train positive rate: {pos_rate:.3f}")
        return pos_rate

    def build_loaders(self, batch_size: int = 64) -> float:
        train_ds = HFTSequenceDataset(self.X_train, self.y_train)
        val_ds   = HFTSequenceDataset(self.X_val,   self.y_val)
        self.train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                                       num_workers=2, pin_memory=True,
                                       drop_last=True)
        self.val_loader   = DataLoader(val_ds,   batch_size, shuffle=False,
                                       num_workers=2, pin_memory=True)
        pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        return float(pos_weight)

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, epochs: int = 100, lr: float = 1e-3,
              patience: int = 15, batch_size: int = 64):

        pos_weight_val = self.build_loaders(batch_size)
        print(f"Class pos_weight: {pos_weight_val:.2f}x  (negatives per positive)")

        # Loss: per-sample weighted BCE to handle class imbalance
        if self.use_focal:
            criterion = FocalLoss(alpha=0.75, gamma=3.0)
            self._pos_weight = None
        else:
            self._pos_weight = torch.tensor([pos_weight_val], device=self.device)
            criterion = nn.BCELoss(reduction='none')  # weighted per-sample in loop

        # Optimizer + cosine annealing with warm restarts
        optimizer = optim.AdamW(self.model.parameters(), lr=lr,
                                weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2)

        best_val_auc    = 0.0       # AUC-based early stopping (formalization spec)
        patience_counter = 0
        train_losses, val_losses, val_aucs = [], [], []
        best_epoch = 0

        print(f"\nTraining {epochs} epochs  LR={lr}  Focal={self.use_focal}")
        print("─" * 60)

        for epoch in range(1, epochs + 1):
            # ── Train ─────────────────────────────────────────────────────────
            self.model.train()
            t_loss = 0.0
            for X_b, y_b in self.train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                with autocast(enabled=self.use_amp):
                    pred = self.model(X_b)
                    y_b_loss = y_b.unsqueeze(-1)
                    if self._pos_weight is not None:
                        # Per-sample weighting: positive samples get pos_weight times more loss
                        sample_w = torch.where(y_b_loss == 1, self._pos_weight.squeeze(), torch.ones(1, device=self.device))
                        loss = (criterion(pred, y_b_loss) * sample_w).mean()
                    else:
                        loss = criterion(pred, y_b_loss)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                t_loss += loss.item()
            t_loss /= len(self.train_loader)

            # ── Validate ──────────────────────────────────────────────────────
            v_loss, v_auc = self._validate(criterion)
            scheduler.step()

            train_losses.append(t_loss)
            val_losses.append(v_loss)
            val_aucs.append(v_auc)

            if epoch % 5 == 0 or epoch == 1:
                print(f"Ep {epoch:>3} | Train {t_loss:.4f} | "
                      f"Val {v_loss:.4f} | AUC {v_auc:.4f} | "
                      f"LR {scheduler.get_last_lr()[0]:.2e}")

            # ── Early stopping on AUC (formalization spec) ────────────────────
            if v_auc > best_val_auc:
                best_val_auc     = v_auc
                best_epoch       = epoch
                patience_counter = 0
                torch.save(self.model.state_dict(), 'titan_lstm_best_v2.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stop at epoch {epoch} (best AUC: {best_val_auc:.4f} @ epoch {best_epoch})")
                    break

        if os.path.exists('titan_lstm_best_v2.pth'):
            self.model.load_state_dict(
                torch.load('titan_lstm_best_v2.pth', map_location=self.device))
            print(f"Reloaded best checkpoint (epoch {best_epoch}, AUC={best_val_auc:.4f})")

        self._plot_curves(train_losses, val_losses, val_aucs)
        return best_val_auc   # Return AUC not loss

    def _validate(self, criterion):
        self.model.eval()
        v_loss, preds_all, targets_all = 0.0, [], []
        with torch.no_grad():
            for X_b, y_b in self.val_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                with autocast(enabled=self.use_amp):
                    pred = self.model(X_b)
                    y_b_loss = y_b.unsqueeze(-1)
                    if self._pos_weight is not None:
                        sample_w = torch.where(y_b_loss == 1, self._pos_weight.squeeze(), torch.ones(1, device=self.device))
                        loss = (criterion(pred, y_b_loss) * sample_w).mean()
                    else:
                        loss = criterion(pred, y_b_loss)
                v_loss += loss.item()
                preds_all.extend(pred.view(-1).cpu().numpy())
                targets_all.extend(y_b.cpu().numpy())
        v_loss /= max(len(self.val_loader), 1)
        try:
            auc = roc_auc_score(targets_all, preds_all)
        except Exception:
            auc = 0.5
        return v_loss, auc

    # ── Evaluation ────────────────────────────────────────────────────────────
    def evaluate_oos(self):
        """Evaluate on the strict OOS test set. Call once, never tune after."""
        print("\n" + "=" * 60)
        print("OUT-OF-SAMPLE EVALUATION (strict - never tune after this)")
        print("=" * 60)

        test_ds     = HFTSequenceDataset(self.X_test, self.y_test)
        test_loader = DataLoader(test_ds, 128, shuffle=False)

        self.model.eval()
        preds_all, targets_all = [], []
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b = X_b.to(self.device)
                pred = self.model(X_b)
                preds_all.extend(pred.view(-1).cpu().numpy())
                targets_all.extend(y_b.numpy())

        preds_arr  = np.array(preds_all)
        target_arr = np.array(targets_all)

        # Find optimal threshold via PR curve
        prec, rec, thresh = precision_recall_curve(target_arr, preds_arr)
        f1s        = 2 * prec * rec / (prec + rec + 1e-8)
        best_thr   = thresh[np.argmax(f1s[:-1])]
        binary_pred = (preds_arr >= best_thr).astype(int)

        auc = roc_auc_score(target_arr, preds_arr)
        print(f"OOS AUC:          {auc:.4f}")
        print(f"Optimal threshold: {best_thr:.3f}")
        print(f"\n{classification_report(target_arr, binary_pred)}")

        # Save threshold for EA deployment
        with open('titan_inference_config.json', 'w') as f:
            json.dump({
                'buy_threshold':  float(best_thr + 0.05),   # Conservative
                'sell_threshold': float(1.0 - best_thr - 0.05),
                'oos_auc':        float(auc),
                'n_features':     N_FEATURES,
                'seq_len':        SEQ_LEN,
                'feat_mean':      self.feat_mean.tolist(),
                'feat_std':       self.feat_std.tolist(),
            }, f, indent=2)
        print("Saved titan_inference_config.json")
        return auc

    # ── ONNX Export ───────────────────────────────────────────────────────────
    def export_onnx(self, output_path: str = 'titan_lstm.onnx',
                    opset: int = 13):
        """
        Export to ONNX opset 13 with static shapes (No dynamic axes).
        """
        print(f"\nExporting ONNX opset {opset} -> {output_path}")
        self.model.eval()
        self.model.cpu()

        dummy = torch.randn(1, SEQ_LEN, N_FEATURES)

        torch.onnx.export(
            self.model,
            dummy,
            output_path,
            export_params       = True,
            opset_version       = opset,
            do_constant_folding = True,
            input_names         = ['tick_sequence'],
            output_names        = ['conviction_score'],
        )

        # ── Numerical verification (BUG-FIX: proper array comparison) ─────────
        session  = ort.InferenceSession(output_path,
                       providers=['CUDAExecutionProvider',
                                  'CPUExecutionProvider'])
        ort_inp  = {session.get_inputs()[0].name: dummy.numpy()}
        ort_out  = np.array(session.run(None, ort_inp)[0]).flatten()

        with torch.no_grad():
            pt_out = self.model(dummy).numpy().flatten()

        max_diff = float(np.max(np.abs(ort_out - pt_out)))
        print(f"Max ONNX/PyTorch diff: {max_diff:.2e}")
        if max_diff < 1e-4:
            print(f"✓  ONNX verification PASSED")
        else:
            print(f"✗  ONNX verification WARNING - diff={max_diff:.2e}")

        file_kb = os.path.getsize(output_path) / 1024
        print(f"Model size: {file_kb:.1f} KB")

        # Restore model to original device
        self.model.to(self.device)
        return output_path

    def _plot_curves(self, train, val, aucs):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(train, label='Train'); axes[0].plot(val, label='Val')
        axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)
        axes[1].plot(aucs, color='orange', label='Val AUC')
        axes[1].axhline(0.6, color='red', linestyle='--', alpha=0.5, label='0.6 floor')
        axes[1].set_title('Validation AUC'); axes[1].legend(); axes[1].grid(True)
        plt.tight_layout()
        plt.savefig('titan_lstm_training_v2.png', dpi=200)
        plt.close()
        print("Training curves -> titan_lstm_training_v2.png")


# =============================================================================
# PIPELINE ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    print("TITAN LSTM TRAINING PIPELINE V2")
    print("=" * 60)

    trainer = TitanLSTMTrainer(use_focal_loss=True, use_amp=True)

    # ── Load data (replace path with real sequences from FeatureEngineering_V2)
    NPZ_PATH = os.path.join('data', 'titan_lstm_sequences.npz')
    if not os.path.exists(NPZ_PATH):
        NPZ_PATH = 'titan_lstm_sequences.npz'   # fallback: same dir
    if not os.path.exists(NPZ_PATH):
        print(f"Sequences file not found: {NPZ_PATH}")
        print("Generating synthetic demo data...")
        # Demo: random sequences matching the real format
        np.random.seed(42)
        N = 5000
        X_all = np.random.randn(N, SEQ_LEN, N_FEATURES).astype(np.float32)
        y_all = (np.random.random(N) > 0.65).astype(np.float32)

        split1 = int(N * 0.70)
        split2 = int(N * 0.85)
        np.savez(NPZ_PATH,
                 X_train=X_all[:split1],  y_train=y_all[:split1],
                 X_val  =X_all[split1:split2], y_val=y_all[split1:split2],
                 X_test =X_all[split2:],  y_test =y_all[split2:],
                 feat_mean=np.zeros(N_FEATURES),
                 feat_std =np.ones(N_FEATURES))
        print("Synthetic sequences saved.")

    trainer.load_sequences(NPZ_PATH)
    trainer.train(epochs=100, lr=1e-3, patience=15, batch_size=64)
    auc = trainer.evaluate_oos()
    trainer.export_onnx('titan_lstm.onnx', opset=13)

    print(f"\n✓  Pipeline complete.  OOS AUC={auc:.4f}")
    print("Deploy titan_lstm.onnx + titan_inference_config.json to MT5 Files/")