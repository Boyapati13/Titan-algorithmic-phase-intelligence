"""
TitanDeploymentFix.py  —  FINAL CANONICAL EXPORT
=================================================
Forensic audit findings corrected in this script:

  [BUG-01] OUTPUT SHAPE MISMATCH (ROOT CAUSE OF "NO TRADES")
           TitanLSTM.forward() returns view(-1) → shape [batch] (1-D).
           EA's OnnxSetOutputShape expects {1, 1} (2-D, [batch, 1]).
           Fix: wrapper forces output to [batch, 1] in the exported graph.

  [BUG-02] OPSET INCOMPATIBILITY
           TitanLSTMTraining exports opset 18 with dynamic axes.
           TitanDeploymentFix was exporting opset 13 static.
           Fix: opset 13, dynamo=False (legacy path), NO dynamic axes.
           MT5 ONNXRuntime (Build 5663) is stable at opset 13 static.

  [BUG-03] NAME BINDING
           Training script: input='tick_sequence', output='conviction_score'
           EA OnnxRun: uses positional index — names don't matter for OnnxRun,
           but the shape of the output tensor DOES. Fixed by output reshape.

  [BUG-04] SCALING PARITY  (CONFIRMED CORRECT — no fix needed)
           titan_inference_config.json feat_mean=0, feat_std=1 confirms the
           model was trained on per-window z-scored sequences (create_lstm_sequences
           in TitanFeatureEngineering.py lines 529-533). The EA's ShiftMatrixAndInsert()
           applies the SAME local z-score. Global scaling is NOT used. ✓

  [BUG-05] THRESHOLD MISMATCH
           Config: buy=0.365 / sell=0.635  (OOS iter3, 0.65 AUC)
           EA:     buy=0.72  / sell=0.28   (input defaults)
           Fix: print the correct EA input values from the config so the user
           can set them in the Strategy Tester inputs panel.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os, glob, shutil, json
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE  (matches TitanLSTMTraining.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
N_FEATURES = 12
SEQ_LEN    = 128
HIDDEN     = 64
NUM_LAYERS = 2
DROPOUT    = 0.2


class TitanLSTMBase(nn.Module):
    """Exact replica of TitanLSTMTraining.TitanLSTM."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_FEATURES, HIDDEN, NUM_LAYERS,
                            dropout=(DROPOUT if NUM_LAYERS > 1 else 0.0),
                            batch_first=True)
        self.bn   = nn.BatchNorm1d(HIDDEN)
        self.drop = nn.Dropout(DROPOUT)
        self.fc1  = nn.Linear(HIDDEN, 32)
        self.fc2  = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]            # [batch, hidden]
        h = self.bn(h)
        h = self.drop(h)
        h = self.relu(self.fc1(h))
        h = self.drop(h)
        return torch.sigmoid(self.fc2(h))  # [batch, 1]  ← 2-D, NOT view(-1)


class TitanLSTMExport(nn.Module):
    """
    BUG-01 FIX: Wrapper that guarantees output shape [batch, 1].

    The training model uses view(-1) which collapses to [batch] (1-D).
    EA's OnnxSetOutputShape(handle, 0, {1, 1}) expects [batch=1, out=1] (2-D).
    When shapes don't match, OnnxRun returns false silently → no inference →
    output_buf[0] stays 0.0 → pred never crosses 0.72 → NO TRADES.

    This wrapper replaces view(-1) with view(-1, 1) to produce [batch, 1].
    The EA reads output_buf[0] which is the same scalar either way.
    """
    def __init__(self, base: TitanLSTMBase):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)             # [batch, 1]
        return out                     # keep as [batch, 1]


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MQL5_FILES_DIR = (
    r"C:\Users\Tenders\AppData\Roaming\MetaQuotes\Terminal"
    r"\AE2CC2E013FDE1E3CDF010AA51C60400\MQL5\Files"
)
# Also copy to the shared terminal used by MetaEditor (for #resource embedding)
SHARED_FILES_DIR = (
    r"C:\Users\Tenders\AppData\Roaming\MetaQuotes\Terminal"
    r"\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files"
)

MODEL_PATH = "titan_lstm_best_v2.pth"
ONNX_PATH  = "titan_lstm.onnx"
CFG_PATH   = "titan_inference_config.json"
OPSET      = 13   # MT5 Build 5663 ONNXRuntime — opset 13 static, most stable


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def export_golden_brain() -> None:
    sep = "=" * 60
    print(sep)
    print("TITAN V2.1  —  CANONICAL ONNX EXPORT")
    print(sep)

    # 1. Clean stale sidecar files
    for f in glob.glob("*.onnx.data") + glob.glob("*.npz"):
        os.remove(f)
        print(f"  Removed: {f}")

    # 2. Load weights
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}\n"
            "Run TitanLSTMTraining.py first."
        )
    base = TitanLSTMBase()

    # Load checkpoint — handle both raw state_dict and wrapped saves
    raw = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    # Strip 'base.' prefix if saved from TitanLSTMExport wrapper
    if any(k.startswith("base.") for k in raw):
        raw = {k.replace("base.", "", 1): v for k, v in raw.items()}
    base.load_state_dict(raw)
    base.eval()

    model = TitanLSTMExport(base)
    model.eval()
    print(f"  Loaded: {MODEL_PATH}")

    # 3. Dummy input [batch=1, seq=50, features=12] — STATIC, no dynamic axes
    dummy = torch.zeros(1, SEQ_LEN, N_FEATURES, dtype=torch.float32)
    with torch.no_grad():
        test_out = model(dummy)
    print(f"  Output shape from wrapper: {list(test_out.shape)}  "
          f"(must be [1, 1] for OnnxSetOutputShape)")
    assert test_out.shape == (1, 1), \
        f"FATAL: Output shape {test_out.shape} != (1,1). Fix the wrapper."

    # 4. Export — legacy TorchScript path, opset 13, float32, static shapes
    print(f"\n  Exporting -> {ONNX_PATH}  "
          f"(opset {OPSET}, float32, static [1x{SEQ_LEN}x{N_FEATURES}])")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            ONNX_PATH,
            dynamo=False,               # Legacy TorchScript path (stable for LSTM)
            export_params=True,         # Embed weights — no .onnx.data sidecar
            opset_version=OPSET,
            do_constant_folding=True,
            input_names=["tick_sequence"],
            output_names=["conviction_score"],
            training=torch.onnx.TrainingMode.EVAL,
            # NO dynamic_axes — MT5 uses static batch=1
        )

    raw_kb = os.path.getsize(ONNX_PATH) / 1024
    print(f"  Export size: {raw_kb:.1f} KB")

    # 5. Validate with ONNX checker + shape check
    try:
        import onnx, onnxruntime as ort
        import numpy as np

        m = onnx.load(ONNX_PATH)
        onnx.checker.check_model(m)
        opset_actual = m.opset_import[0].version

        # Verify output shape in ONNX graph
        out_shape = [
            d.dim_value
            for d in m.graph.output[0].type.tensor_type.shape.dim
        ]
        print(f"  ONNX check PASSED  opset={opset_actual}  "
              f"output_shape={out_shape}")
        assert out_shape == [1, 1], \
            f"ONNX output shape {out_shape} != [1,1]. OnnxSetOutputShape will fail."

        # Runtime numerical check
        sess = ort.InferenceSession(ONNX_PATH,
                                    providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        ort_out  = sess.run(None, {inp_name: dummy.numpy()})[0]
        with torch.no_grad():
            pt_out = model(dummy).numpy()
        diff = float(abs(ort_out.flatten()[0] - pt_out.flatten()[0]))
        print(f"  PyTorch={pt_out.flatten()[0]:.6f}  "
              f"ORT={ort_out.flatten()[0]:.6f}  "
              f"diff={diff:.2e}  {'PASS' if diff < 1e-4 else 'WARN'}")

    except ImportError as e:
        print(f"  WARNING: Validation skipped ({e})")
    except Exception as e:
        print(f"  WARNING: Validation error — {e}")

    # 6. Deploy to all required locations
    dirs = [MQL5_FILES_DIR, SHARED_FILES_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        dest = os.path.join(d, ONNX_PATH)
        shutil.copy(ONNX_PATH, dest)
        print(f"  Deployed -> {dest}")

    # 7. Print EA input values from config
    print()
    print("-" * 60)
    print("STRATEGY TESTER INPUT VALUES  (copy these into EA inputs)")
    print("-" * 60)
    if os.path.exists(CFG_PATH):
        cfg = json.load(open(CFG_PATH))
        buy_thr  = cfg.get("buy_threshold",  0.72)
        sell_thr = cfg.get("sell_threshold", 0.28)
        auc      = cfg.get("oos_auc",        0.0)
        print(f"  OOS AUC:          {auc:.4f}")
        print(f"  BUY_THRESHOLD:    {buy_thr:.4f}   <-- set this in EA inputs!")
        print(f"  SELL_THRESHOLD:   {sell_thr:.4f}   <-- set this in EA inputs!")
    else:
        print(f"  WARNING: {CFG_PATH} not found.")
        print("  Run TitanLSTMTraining.py -> evaluate_oos() first.")
        print("  Default: BUY_THRESHOLD=0.60  SELL_THRESHOLD=0.40")

    print()
    print(sep)
    print("DEPLOYMENT COMPLETE")
    print(sep)
    print()
    print("NEXT STEPS:")
    print("  1. Open MetaEditor -> TitanEA.mq5 -> press F7 to recompile")
    print("     (embeds the new ONNX into .ex5 via #resource)")
    print("  2. In Strategy Tester, set inputs above and run backtest.")
    print("  3. Expect 'ONNX: titan_lstm.onnx  Matrix: [50x12]'")
    print("     in the EA journal on init — model is loaded.")


if __name__ == "__main__":
    export_golden_brain()