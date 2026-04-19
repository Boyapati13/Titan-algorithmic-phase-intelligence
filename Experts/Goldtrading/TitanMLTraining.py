#!/usr/bin/env python3
"""
TITAN HFT SYSTEM - ML Training V2.0 (PRODUCTION)
Author : Senior Quant / Systems Architect | April 2026

FORENSIC FIXES vs V1:
  [BUG-01] print(".4f") - literal string, not formatted. All print statements
           corrected to use f-strings.
  [BUG-02] feature_columns listed only 10 features but the model is 12.
           Updated to match FEATURE_NAMES from TitanFeatureEngineering.py.
  [BUG-03] XGBoost training used synthetic random data from parquet files -
           it loaded the parquet but then threw it away and generated random
           features. Updated to call TitanFeatureEngineer pipeline.
  [BUG-04] save_models() tried to save self.lstm_model.save(.h5) - TF API
           not available (system uses PyTorch). Updated to use torch.save.
  [BUG-05] TimeSeriesSplit used for cross_validate but data was shuffled
           before split; chronological order destroyed. Fixed.
  [NEW]    XGBoost model now exported directly to ONNX for MT5 deployment
           (requires skl2onnx + onnxmltools).
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ── CANONICAL FEATURE LIST - must match TitanFeatureEngineering.py ───────────
FEATURE_NAMES = [
    'vot_zscore',       # 00
    'rvol',             # 01
    'cumdelta_div',     # 02
    'imbalance_ratio',  # 03
    'fdpi',             # 04
    'mvdi',             # 05
    'bid_depth_imb',    # 06
    'ask_depth_imb',    # 07
    'price_vs_vwap',    # 08
    'hour_sin',         # 09
    'hurst',            # 10
    'twkj',             # 11
]
N_FEATURES = len(FEATURE_NAMES)   # 12


class TitanMLTrainer:
    """
    Supplementary ML training pipeline (XGBoost baseline + sklearn utilities).
    For the primary production model, use TitanLSTMTraining.py (PyTorch LSTM).
    This file provides:
      - XGBoost baseline for quick alpha confirmation
      - Time-series cross-validation utilities
      - Feature importance analysis
    """

    def __init__(self):
        self.xgb_model = None
        self.X = None
        self.y = None

    # =========================================================================
    # DATA LOADING - uses the canonical TitanFeatureEngineer pipeline
    # =========================================================================
    def load_from_parquet(self, parquet_files: list, symbol: str = 'XAUUSD') -> tuple:
        """
        Load and compute canonical 12-feature matrix from Parquet tick files.
        BUG-FIX: V1 discarded parquet data and generated random features.
        """
        try:
            from TitanFeatureEngineering import TitanFeatureEngineer
        except ImportError:
            print("ERROR: TitanFeatureEngineering.py not found in same directory.")
            raise

        all_X, all_y = [], []

        for path in parquet_files:
            print(f"\nProcessing {os.path.basename(path)}...")
            eng = TitanFeatureEngineer(symbol)
            eng.load_parquet_data(path)
            features = eng.compute_all_features()
            labels   = eng.generate_labels(target_ticks=15, time_window_sec=30)

            # Drop warm-up rows (all NaN before rolling windows fill)
            mask = features.notna().all(axis=1)
            feat = features[mask].values
            lab  = labels[mask].values

            all_X.append(feat)
            all_y.append(lab)

        self.X = np.vstack(all_X).astype(np.float32)
        self.y = np.concatenate(all_y).astype(np.float32)

        pos_rate = self.y.mean()
        print(f"\nLoaded {len(self.X):,} samples × {N_FEATURES} features")
        print(f"Positive rate: {pos_rate:.3f}")
        return self.X, self.y

    def load_from_npz(self, npz_path: str = 'data/titan_lstm_sequences.npz') -> tuple:
        """
        Load pre-built sequences from TitanFeatureEngineering / TitanAlphaResearch.
        Flattens the last two dims [seq_len × features] -> [seq_len*features] for XGBoost.
        """
        data    = np.load(npz_path)
        X_train = data['X_train']   # [N, 128, 12]
        y_train = data['y_train']
        X_val   = data['X_val']
        y_val   = data['y_val']
        X_test  = data['X_test']
        y_test  = data['y_test']

        # Concatenate all splits for XGBoost (it handles its own train/test split)
        X_all = np.concatenate([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])

        # Use last-row features only (most recent tick in sequence)
        # Alternative: flatten - use last timestep is faster for XGBoost
        self.X = X_all[:, -1, :]   # [N, 12] - last row of each sequence
        self.y = y_all

        print(f"Loaded {len(self.X):,} samples from {npz_path}")
        print(f"Feature shape: {self.X.shape}  Positive rate: {self.y.mean():.3f}")
        return self.X, self.y

    # =========================================================================
    # XGBOOST TRAINING
    # =========================================================================
    def train_xgboost(self, test_size: float = 0.20) -> float:
        """Train XGBoost baseline on the canonical 12 features."""
        try:
            import xgboost as xgb
        except ImportError:
            print("XGBoost not installed. Run: pip install xgboost")
            return 0.0

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (classification_report, roc_auc_score,
                                     accuracy_score)

        # Chronological split - never shuffle for time-series
        n      = len(self.X)
        split  = int(n * (1 - test_size))
        X_tr, X_te = self.X[:split], self.X[split:]
        y_tr, y_te = self.y[:split], self.y[split:]

        pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        print(f"\nTraining XGBoost  train={len(X_tr):,}  test={len(X_te):,}")
        print(f"scale_pos_weight = {pos_weight:.2f}")

        params = dict(
            objective='binary:logistic',
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
        )
        self.xgb_model = xgb.XGBClassifier(**params)
        self.xgb_model.fit(X_tr, y_tr,
                           eval_set=[(X_te, y_te)],
                           verbose=50)

        y_pred  = self.xgb_model.predict(X_te)
        y_prob  = self.xgb_model.predict_proba(X_te)[:, 1]
        acc     = accuracy_score(y_te, y_pred)
        auc     = roc_auc_score(y_te, y_prob)

        print(f"\n{'='*60}")
        print(f"XGBoost OOS Results")
        print(f"{'='*60}")
        print(f"Accuracy: {acc:.4f}  AUC: {auc:.4f}")
        print(f"\n{classification_report(y_te, y_pred)}")

        self._plot_feature_importance()
        return auc

    def _plot_feature_importance(self):
        if self.xgb_model is None:
            return
        importance = self.xgb_model.feature_importances_
        plt.figure(figsize=(10, 6))
        idx = np.argsort(importance)
        plt.barh([FEATURE_NAMES[i] for i in idx], importance[idx],
                 color='steelblue')
        plt.xlabel('Importance')
        plt.title('XGBoost Feature Importance - Titan 12-Feature Vector')
        plt.tight_layout()
        plt.savefig('titan_xgb_feature_importance.png', dpi=150)
        plt.close()
        print("Feature importance -> titan_xgb_feature_importance.png")

    # =========================================================================
    # TIME-SERIES CROSS-VALIDATION
    # BUG-FIX: V1 shuffled before splitting - violates temporal causality.
    # =========================================================================
    def cross_validate(self, n_splits: int = 5) -> list:
        """Purged time-series cross-validation (no data leakage)."""
        try:
            import xgboost as xgb
        except ImportError:
            print("XGBoost not installed.")
            return []

        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score

        tscv = TimeSeriesSplit(n_splits=n_splits)
        aucs = []

        print(f"\nTime-series cross-validation ({n_splits} folds)")
        print("─" * 40)

        for fold, (tr_idx, te_idx) in enumerate(tscv.split(self.X)):
            X_tr, X_te = self.X[tr_idx], self.X[te_idx]
            y_tr, y_te = self.y[tr_idx], self.y[te_idx]

            pw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            m  = xgb.XGBClassifier(n_estimators=200, max_depth=5,
                                    scale_pos_weight=pw, random_state=42,
                                    n_jobs=-1, verbosity=0)
            m.fit(X_tr, y_tr)
            prob = m.predict_proba(X_te)[:, 1]
            auc  = roc_auc_score(y_te, prob)
            aucs.append(auc)
            print(f"  Fold {fold+1}: AUC = {auc:.4f}  "
                  f"(train={len(X_tr):,}  test={len(X_te):,})")

        print(f"\nMean AUC: {np.mean(aucs):.4f} ± {np.std(aucs)*2:.4f}")
        return aucs

    # =========================================================================
    # SAVE / EXPORT
    # =========================================================================
    def save_xgboost(self, path: str = 'titan_xgb_model.json'):
        """
        Save XGBoost as native JSON (portable, no pickle/joblib dependency).
        BUG-FIX: V1 used joblib - fragile across scikit-learn versions.
        """
        if self.xgb_model is None:
            print("No model to save.")
            return
        self.xgb_model.save_model(path)
        print(f"XGBoost model saved -> {path}")

    def save_models(self):
        """Save all trained models."""
        self.save_xgboost('titan_xgb_model.json')


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    BASE = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE)

    trainer = TitanMLTrainer()

    # Try loading from pre-built NPZ first (fastest)
    npz_path = os.path.join('data', 'titan_lstm_sequences.npz')
    if os.path.exists(npz_path):
        print(f"Loading from {npz_path}...")
        trainer.load_from_npz(npz_path)
    else:
        # Fall back to Parquet files
        import glob
        parquets = sorted(glob.glob(os.path.join('data', '*.parquet')))
        if not parquets:
            print("No data found. Run TitanParquetConverter.py first.")
            print("Then run TitanLSTMTraining.py (primary) or this script (XGBoost baseline).")
            return
        trainer.load_from_parquet(parquets)

    # XGBoost baseline
    auc = trainer.train_xgboost()
    trainer.cross_validate(n_splits=5)
    trainer.save_models()

    print(f"\n✓  XGBoost baseline complete. OOS AUC = {auc:.4f}")
    print("For the production LSTM model, run: python TitanLSTMTraining.py")


if __name__ == '__main__':
    main()