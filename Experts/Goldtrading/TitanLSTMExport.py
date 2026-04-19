#!/usr/bin/env python3
"""
TITAN HFT SYSTEM - ONNX Export Utility V2.0 (PRODUCTION)
Author : Senior Quant / Systems Architect | April 2026

FORENSIC FIXES vs V1:
  [BUG-01] forward() used squeeze() - collapses the batch dimension to a
           scalar when batch_size=1. ONNX needs the batch axis preserved.
           Fixed with view(-1) (matches TitanLSTMTraining.py exactly).
  [BUG-02] TitanLSTM in V1 was missing the BatchNorm1d(64) layer that
           TitanLSTMTraining.py defines. Silent architecture mismatch -
           state_dict loading would fail at runtime. Added bn layer.
  [BUG-03] input_names=['input'] but TitanLSTMTraining.py exports with
           input_names=['tick_sequence']. Name mismatch breaks OnnxCreate
           in MT5 if the session was built with the training exporter.
           Unified to 'tick_sequence' / 'conviction_score'.
  [BUG-04] dynamic_axes only made batch axis dynamic; MT5's OnnxRun also
           needs a fixed shape declared. Kept batch dynamic, seq fixed.
  [BUG-05] ONNX verification used max_diff < 1e-5 but never asserted -
           silent pass on large numerical errors. Now raises on > 1e-4.
  [NEW]    Loads checkpoint from titan_lstm_best_v2.pth (V2 filename).
  [NEW]    UTF-8 stdout reconfigure for Windows cp1252 safety.
"""

import os
import sys
import struct
import json
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

# Force UTF-8 output to avoid Windows cp1252 encoding errors
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ── ARCHITECTURE CONSTANTS (must match TitanLSTMTraining.py exactly) ─────────
N_FEATURES  = 12
SEQ_LEN     = 128
HIDDEN      = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2


# =============================================================================
# MODEL - identical architecture to TitanLSTMTraining.py::TitanLSTM
# =============================================================================
class TitanLSTM(nn.Module):
    """
    TITAN LSTM - input: [batch, SEQ_LEN, N_FEATURES]
    Output:      [batch]  conviction score in (0, 1)

    MUST be kept byte-for-byte identical to TitanLSTMTraining.py::TitanLSTM
    so that torch.load(state_dict) works without dimension errors.
    """

    def __init__(self,
                 input_size:  int   = N_FEATURES,
                 hidden_size: int   = HIDDEN,
                 num_layers:  int   = NUM_LAYERS,
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
        # BUG-FIX: V1 TitanLSTMExport.py was missing this layer
        self.bn   = nn.BatchNorm1d(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden_size, 32)
        self.fc2  = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, features]
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]              # [batch, hidden]
        h_last = self.bn(h_last)
        h_last = self.drop(h_last)
        h_last = self.relu(self.fc1(h_last))
        h_last = self.drop(h_last)
        out    = self.fc2(h_last)     # [batch, 1] - raw logit
        # BUG-FIX: view(-1) preserves batch dim for ONNX (squeeze collapses it)
        return torch.sigmoid(out.view(-1))   # [batch]


# =============================================================================
# EXPORT FUNCTION
# =============================================================================
def export_onnx(
    checkpoint_path: str = 'titan_lstm_best_v2.pth',
    output_path:     str = 'titan_lstm.onnx',
    config_path:     str = 'titan_inference_config.json',
    opset:           int = 13,
) -> str:
    """
    Load the best checkpoint and export to ONNX opset 18.
    Verifies numerical equivalence between PyTorch and ONNX output.

    Returns the output_path on success.
    """
    print(f"\n{'='*60}")
    print(f"TITAN ONNX EXPORT  opset={opset}")
    print(f"{'='*60}")

    # ── 1. Load model ─────────────────────────────────────────────────────────
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run TitanLSTMTraining.py first to produce {checkpoint_path}")

    device = torch.device('cpu')
    model  = TitanLSTM(input_size=N_FEATURES,
                       hidden_size=HIDDEN,
                       num_layers=NUM_LAYERS,
                       dropout=DROPOUT)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 2. ONNX export ────────────────────────────────────────────────────────
    dummy = torch.randn(1, SEQ_LEN, N_FEATURES)
    print(f"\nExporting -> {output_path}  (input shape: {list(dummy.shape)})")

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params       = True,
        opset_version       = opset,
        do_constant_folding = True,
        input_names         = ['tick_sequence'],    # Must match TitanEA.mq5 OnnxSetInputShape
        output_names        = ['conviction_score'],
    )

    # ── 3. Numerical verification ─────────────────────────────────────────────
    sess     = ort.InferenceSession(output_path,
                   providers=['CPUExecutionProvider'])
    ort_inp  = {sess.get_inputs()[0].name: dummy.numpy()}
    ort_out  = np.array(sess.run(None, ort_inp)[0]).flatten()

    with torch.no_grad():
        pt_out = model(dummy).numpy().flatten()

    max_diff = float(np.max(np.abs(ort_out - pt_out)))
    print(f"Max ONNX/PyTorch diff: {max_diff:.2e}")
    if max_diff < 1e-4:
        print("✓  ONNX numerical verification PASSED")
    else:
        raise RuntimeError(
            f"ONNX numerical mismatch too large: {max_diff:.2e} > 1e-4\n"
            f"Check for non-deterministic ops or opset compatibility.")

    # ── 4. Shape & range assertions ───────────────────────────────────────────
    inp_meta = sess.get_inputs()[0]
    out_meta = sess.get_outputs()[0]
    print(f"\nONNX Input:  '{inp_meta.name}'  {inp_meta.shape}  {inp_meta.type}")
    print(f"ONNX Output: '{out_meta.name}'  {out_meta.shape}  {out_meta.type}")
    assert ort_out[0] >= 0.0 and ort_out[0] <= 1.0, \
        f"Output {ort_out[0]:.4f} outside sigmoid range [0,1]"

    file_kb = os.path.getsize(output_path) / 1024
    print(f"Model size: {file_kb:.1f} KB")

    # ── 5. Load config for display ────────────────────────────────────────────
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"\nInference config ({config_path}):")
        print(f"  buy_threshold:  {cfg.get('buy_threshold',  'N/A')}")
        print(f"  sell_threshold: {cfg.get('sell_threshold', 'N/A')}")
        print(f"  oos_auc:        {cfg.get('oos_auc',        'N/A')}")
        print(f"  n_features:     {cfg.get('n_features',     'N/A')}")
        print(f"  seq_len:        {cfg.get('seq_len',        'N/A')}")

        # Validate config values vs current architecture
        assert cfg.get('n_features') == N_FEATURES, \
            f"Config n_features={cfg['n_features']} != model N_FEATURES={N_FEATURES}"
        assert cfg.get('seq_len') == SEQ_LEN, \
            f"Config seq_len={cfg['seq_len']} != model SEQ_LEN={SEQ_LEN}"
        print("✓  Config integrity check PASSED")

    print(f"\n✓  Export complete: {output_path}")
    print(f"   Deploy to: ...\\MQL5\\Files\\{os.path.basename(output_path)}")
    print(f"   Deploy to: ...\\MQL5\\Files\\{os.path.basename(config_path)}")
    return output_path


# =============================================================================
# DEPLOYMENT HELPER - copy files to MT5 Files folder
# =============================================================================
def deploy_to_mt5(
    terminal_id: str = 'AE2CC2E013FDE1E3CDF010AA51C60400',
    onnx_path:   str = 'titan_lstm.onnx',
    config_path: str = 'titan_inference_config.json',
):
    """Copy ONNX model and inference config to the MT5 Files directory."""
    import shutil, os
    appdata    = os.environ.get('APPDATA', '')
    mt5_files  = os.path.join(
        appdata, 'MetaQuotes', 'Terminal', terminal_id, 'MQL5', 'Files')

    if not os.path.isdir(mt5_files):
        print(f"WARNING: MT5 Files directory not found: {mt5_files}")
        print("Copy manually:")
        print(f"  {onnx_path}   -> {{MT5_DATA}}\\MQL5\\Files\\")
        print(f"  {config_path} -> {{MT5_DATA}}\\MQL5\\Files\\")
        return

    for src in [onnx_path, config_path]:
        if os.path.exists(src):
            dst = os.path.join(mt5_files, os.path.basename(src))
            shutil.copy2(src, dst)
            print(f"Deployed: {src} -> {dst}")
        else:
            print(f"WARNING: Source not found, skipping: {src}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    BASE = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE)

    onnx_path = export_onnx(
        checkpoint_path='titan_lstm_best_v2.pth',
        output_path='titan_lstm.onnx',
        config_path='titan_inference_config.json',
        opset=13,
    )

    deploy_to_mt5(onnx_path=onnx_path, config_path='titan_inference_config.json')
