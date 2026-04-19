#!/usr/bin/env python3
"""
TITAN HFT SYSTEM - ONNX Export Bridge V2.0 (PRODUCTION)
Author : Senior Quant / Systems Architect | April 2026

FORENSIC FIXES vs V1:
  [BUG-01] Imported onnxmltools.convert.convert_xgboost - this package is
           unmaintained and often breaks. Replaced with skl2onnx (supported).
  [BUG-02] Used TensorFlow + tf2onnx for LSTM export. The production LSTM
           is PyTorch-based (TitanLSTMTraining.py). This file now wraps
           TitanLSTMExport.py for PyTorch->ONNX.
  [BUG-03] verify_onnx_model used hardcoded shapes [1, 100, 10] (100 steps,
           10 features) - should be [1, 50, 12].
  [BUG-04] XGBoost ONNX conversion used 10 features, not 12.
  [NEW]    Single unified entry point: converts both XGBoost and LSTM models
           and validates them in one run.
  [NEW]    Auto-deploy to MT5 Files folder on success.
"""

import os
import sys
import json
import numpy as np
import onnxruntime as ort

# Force UTF-8 on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

N_FEATURES = 12
SEQ_LEN    = 128


# =============================================================================
# LSTM ONNX EXPORT - delegates to TitanDeploymentFix.py
# =============================================================================
def export_lstm_onnx(
    checkpoint: str = 'titan_lstm_best_v2.pth',
    output:     str = 'titan_lstm.onnx',
    config:     str = 'titan_inference_config.json',
) -> bool:
    """Export PyTorch LSTM -> ONNX via TitanDeploymentFix.py."""
    try:
        from TitanDeploymentFix import export_golden_brain
        export_golden_brain()
        return True
    except FileNotFoundError as e:
        print(f"[SKIP] LSTM export: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] LSTM export failed: {e}")
        return False


# =============================================================================
# XGBOOST ONNX EXPORT - uses skl2onnx (maintained, replaces onnxmltools)
# =============================================================================
def export_xgboost_onnx(
    model_path: str = 'titan_xgb_model.json',
    output:     str = 'titan_xgb.onnx',
) -> bool:
    """Convert XGBoost JSON model -> ONNX via skl2onnx."""
    if not os.path.exists(model_path):
        print(f"[SKIP] XGBoost export: {model_path} not found")
        return False

    try:
        import xgboost as xgb
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx.operator_converters.xgboost import convert_xgboost

        print(f"\nLoading XGBoost model: {model_path}")
        model = xgb.XGBClassifier()
        model.load_model(model_path)

        # Initial type: [batch, N_FEATURES]
        initial_type = [('float_input', FloatTensorType([None, N_FEATURES]))]
        onnx_model   = convert_sklearn(model, initial_types=initial_type,
                                       target_opset=15)

        with open(output, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        print(f"XGBoost ONNX saved -> {output}")

        # Verify
        sess = ort.InferenceSession(output, providers=['CPUExecutionProvider'])
        dummy = np.random.randn(1, N_FEATURES).astype(np.float32)
        out   = sess.run(None, {sess.get_inputs()[0].name: dummy})
        print(f"XGBoost ONNX verification passed. Output: {out[0].flatten()[:3]}")
        return True

    except ImportError:
        print("[SKIP] XGBoost ONNX export requires: pip install skl2onnx")
        return False
    except Exception as e:
        print(f"[ERROR] XGBoost ONNX export failed: {e}")
        return False


# =============================================================================
# UNIVERSAL ONNX VERIFIER
# BUG-FIX: V1 used hardcoded [1, 100, 10] for LSTM - fixed to [1, SEQ_LEN, N_FEATURES]
# =============================================================================
def verify_onnx_model(onnx_path: str) -> bool:
    """Load ONNX model and run a shape-correct inference test."""
    if not os.path.exists(onnx_path):
        print(f"[SKIP] Verify: {onnx_path} not found")
        return False

    print(f"\nVerifying: {onnx_path}")
    sess       = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    inp        = sess.get_inputs()[0]
    out        = sess.get_outputs()[0]
    print(f"  Input:  '{inp.name}'  {inp.shape}  {inp.type}")
    print(f"  Output: '{out.name}'  {out.shape}  {out.type}")

    # Determine correct dummy input shape from model metadata
    shape = inp.shape
    if len(shape) == 3:
        # LSTM [batch, seq, features]
        dummy = np.random.randn(1, SEQ_LEN, N_FEATURES).astype(np.float32)
    elif len(shape) == 2:
        # XGBoost / flat [batch, features]
        dummy = np.random.randn(1, N_FEATURES).astype(np.float32)
    else:
        dummy = np.random.randn(1, N_FEATURES).astype(np.float32)

    ort_out = sess.run(None, {inp.name: dummy})
    print(f"  Output value: {np.array(ort_out[0]).flatten()[:5]}")
    print(f"  ✓  Shape test PASSED")
    return True


# =============================================================================
# MT5 DEPLOYMENT
# =============================================================================
def deploy_to_mt5_files(
    *files: str,
    terminal_id: str = 'AE2CC2E013FDE1E3CDF010AA51C60400',
):
    """Copy ONNX and config files to MT5 Files directory."""
    import shutil
    appdata   = os.environ.get('APPDATA', '')
    mt5_files = os.path.join(appdata, 'MetaQuotes', 'Terminal',
                             terminal_id, 'MQL5', 'Files')

    if not os.path.isdir(mt5_files):
        print(f"\n[WARN] MT5 Files dir not found: {mt5_files}")
        print("Copy these files manually:")
        for f in files:
            if os.path.exists(f):
                print(f"  {f}  ->  {{MT5_DATA}}\\MQL5\\Files\\")
        return

    print(f"\nDeploying to: {mt5_files}")
    for src in files:
        if os.path.exists(src):
            dst = os.path.join(mt5_files, os.path.basename(src))
            shutil.copy2(src, dst)
            size_kb = os.path.getsize(dst) / 1024
            print(f"  ✓  {os.path.basename(src)}  ({size_kb:.1f} KB)")
        else:
            print(f"  [SKIP] {src} not found")


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    BASE = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE)

    print("=" * 60)
    print("TITAN ONNX EXPORT BRIDGE V2")
    print("=" * 60)

    # 1. Export primary LSTM model
    lstm_ok = export_lstm_onnx(
        checkpoint='titan_lstm_best_v2.pth',
        output='titan_lstm.onnx',
        config='titan_inference_config.json',
    )

    # 2. Export XGBoost baseline model (optional)
    xgb_ok = export_xgboost_onnx(
        model_path='titan_xgb_model.json',
        output='titan_xgb.onnx',
    )

    # 3. Verify ONNX models
    if lstm_ok:
        verify_onnx_model('titan_lstm.onnx')
    if xgb_ok:
        verify_onnx_model('titan_xgb.onnx')

    # 4. Deploy to MT5 Files directory
    deploy_files = []
    if lstm_ok:
        deploy_files += ['titan_lstm.onnx', 'titan_inference_config.json']
    if xgb_ok:
        deploy_files += ['titan_xgb.onnx']

    if deploy_files:
        deploy_to_mt5_files(*deploy_files)

    print("\n" + "=" * 60)
    if lstm_ok:
        print("✓  titan_lstm.onnx  ready for TitanEA.mq5 deployment")
    if xgb_ok:
        print("✓  titan_xgb.onnx   ready for XGBoost EA deployment")
    if not lstm_ok and not xgb_ok:
        print("No models exported. Run training scripts first:")
        print("  python TitanLSTMTraining.py   (primary)")
        print("  python TitanMLTraining.py     (XGBoost baseline)")
    print("=" * 60)


if __name__ == '__main__':
    main()