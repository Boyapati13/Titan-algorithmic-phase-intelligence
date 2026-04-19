"""
titan_onnx.py — ONNX Export, Verification, and Feature Parity CI
=================================================================
Fixes v3.1:
  - Export uses torch.randn dummy (not zeros — zeros are degenerate for LSTM)
  - _Wrapper output shape tested explicitly for [1,1]
  - FeatureParityTester raises ValueError (not silent fallback) if columns missing
  - Updated for 16-feature input
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from titan_config import CFG, FEATURE_COLS
from titan_model import TitanLSTMV3

log = logging.getLogger("TitanONNX")


class _Wrapper(torch.nn.Module):
    """Guarantees [batch, 1] 2-D output required by MQL5 ONNX runtime."""
    def __init__(self, inner: TitanLSTMV3):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)          # (batch,) or (batch, 1)
        if out.dim() == 1:
            out = out.unsqueeze(-1)  # → (batch, 1)
        # Explicit shape assertion — catches any regression
        assert out.dim() == 2 and out.shape[-1] == 1, \
            f"ONNX wrapper output must be [batch,1], got {tuple(out.shape)}"
        return out


def export_onnx(model: TitanLSTMV3, out_dir: Path = None) -> Tuple[Path, Path]:
    """
    Export to ONNX opset 13 with static shapes [1, 128, 16] → [1, 1].
    Runs onnx.checker + onnxruntime parity check automatically.
    Returns (onnx_path, config_json_path).
    """
    ec      = CFG.execution
    mc      = CFG.model
    out_dir = Path(out_dir or CFG.data.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_p  = out_dir / ec.onnx_filename
    cfg_p   = out_dir / ec.onnx_config_json

    # === FORENSIC: Verify loaded model architecture ===
    log.info(
        f"ONNX EXPORT — Model architecture: "
        f"input_size={model.input_size} hidden_size={model.hidden_size} "
        f"num_layers={model.num_layers}"
    )
    if model.input_size != mc.input_size or model.hidden_size != mc.hidden_size:
        raise ValueError(
            f"Model architecture mismatch before ONNX export:\n"
            f"  Expected: input_size={mc.input_size} hidden_size={mc.hidden_size}\n"
            f"  Got:      input_size={model.input_size} hidden_size={model.hidden_size}"
        )

    model.eval()
    # Use small random normal — avoids degenerate all-zero LSTM activations
    dummy   = torch.randn(1, mc.sequence_len, mc.input_size) * 0.1
    wrapped = _Wrapper(model)
    wrapped.eval()

    # Verify wrapper shape before export
    with torch.no_grad():
        test_out = wrapped(dummy)
    assert test_out.shape == (1, 1), \
        f"Pre-export shape check failed: {tuple(test_out.shape)} != (1,1)"

    torch.onnx.export(
        wrapped, dummy, str(onnx_p),
        opset_version       = ec.onnx_opset,
        input_names         = ["input"],
        output_names        = ["conviction"],
        do_constant_folding = True,
        export_params       = True,
        dynamic_axes        = None,   # static shapes — required for MQL5
        verbose             = False,
    )
    log.info(f"ONNX exported → {onnx_p}  (input: [1,{mc.sequence_len},{mc.input_size}])")

    # Graph validity check
    try:
        import onnx
        mdl = onnx.load(str(onnx_p))
        onnx.checker.check_model(mdl)
        n_nodes = len(mdl.graph.node)
        log.info(f"onnx.checker: valid.  Graph nodes: {n_nodes}")
    except ImportError:
        log.warning("onnx not installed — skipping graph check.")
    except Exception as e:
        raise RuntimeError(f"ONNX graph check failed: {e}") from e

    # Runtime parity check
    try:
        import onnxruntime as ort
        sess     = ort.InferenceSession(str(onnx_p), providers=["CPUExecutionProvider"])
        ort_out  = sess.run(["conviction"], {"input": dummy.numpy()})[0]
        with torch.inference_mode():
            pt_out = wrapped(dummy).numpy()
        delta = float(np.abs(ort_out - pt_out).max())
        if delta > 1e-5:
            raise ValueError(f"ONNX/PyTorch output mismatch: delta={delta:.2e}")
        log.info(f"ONNX/PyTorch parity: max_delta={delta:.2e}")
    except ImportError:
        log.warning("onnxruntime not installed — skipping parity check.")

    # Config JSON for MQL5 EA
    config = {
        "model_version":      "3.1",
        "onnx_filename":      ec.onnx_filename,
        "opset":              ec.onnx_opset,
        "input_shape":        [1, mc.sequence_len, mc.input_size],
        "output_shape":       [1, 1],
        "feature_order":      FEATURE_COLS,
        "feature_count":      len(FEATURE_COLS),
        "sequence_len":       mc.sequence_len,
        "conviction_long":    ec.conviction_long,
        "conviction_short":   ec.conviction_short,
        "window_n":           CFG.features.window_n,
        "eps":                CFG.features.eps,
        "clip_sigma":         CFG.features.clip_sigma,
        "min_dt_ms":          CFG.features.min_dt_ms,
        "flag_bid_only":      CFG.features.FLAG_BID_ONLY,
        "flag_ask_only":      CFG.features.FLAG_ASK_ONLY,
        "spread_guard_sigma": ec.spread_guard_sigma,
        "order_timeout_ms":   ec.order_timeout_ms,
        "daily_dd_limit":     ec.daily_dd_limit,
    }
    cfg_p.write_text(json.dumps(config, indent=2))
    log.info(f"Inference config → {cfg_p}")
    return onnx_p, cfg_p


class FeatureParityTester:
    """
    CI test: Python vs MQL5 feature outputs on the same tick sequence.
    Overall MAE must be < CFG.validation.max_drift_mae.
    Raises ValueError on missing columns (not silent fallback).
    """

    def __init__(self):
        self.threshold = CFG.validation.max_drift_mae
        self._py:   Optional[np.ndarray] = None
        self._mql5: Optional[np.ndarray] = None

    def load_python(self, path: Path) -> "FeatureParityTester":
        self._py = self._read(path)
        log.info(f"Python features: {self._py.shape}")
        return self

    def load_mql5(self, path: Path) -> "FeatureParityTester":
        self._mql5 = self._read(path)
        log.info(f"MQL5 features:   {self._mql5.shape}")
        return self

    def load_arrays(self, py: np.ndarray, mql5: np.ndarray) -> "FeatureParityTester":
        self._py   = py.astype(np.float64)
        self._mql5 = mql5.astype(np.float64)
        return self

    def _read(self, path: Path) -> np.ndarray:
        path = Path(path)
        if path.suffix == ".csv":
            df = pd.read_csv(path)
            missing = set(FEATURE_COLS) - set(df.columns)
            if missing:
                raise ValueError(
                    f"CSV missing required feature columns: {missing}.  "
                    "Ensure MQL5 export uses FEATURE_COLS order."
                )
            return df[FEATURE_COLS].values.astype(np.float64)
        return np.load(str(path)).astype(np.float64)

    def run(self) -> Dict:
        if self._py is None or self._mql5 is None:
            raise RuntimeError("Load both feature arrays before run().")
        if self._py.shape != self._mql5.shape:
            raise ValueError(
                f"Shape mismatch: py={self._py.shape} mql5={self._mql5.shape}"
            )
        log.info("\n" + "=" * 60)
        log.info("FEATURE PARITY  (Python vs MQL5)")
        log.info("=" * 60)
        all_pass = True
        results  = {}
        for i, name in enumerate(FEATURE_COLS):
            py_c   = self._py[:, i]
            mq_c   = self._mql5[:, i]
            valid  = ~(np.isnan(py_c) | np.isnan(mq_c))
            mae    = float(np.abs(py_c[valid] - mq_c[valid]).mean()) if valid.any() else 0.0
            mx_e   = float(np.abs(py_c[valid] - mq_c[valid]).max())  if valid.any() else 0.0
            passed = mae <= self.threshold
            if not passed:
                all_pass = False
                log.error(f"  FAIL {name:15s}: MAE={mae:.6f} max={mx_e:.6f}")
            else:
                log.info( f"  PASS {name:15s}: MAE={mae:.6f} max={mx_e:.6f}")
            results[name] = {"mae": mae, "max_err": mx_e, "passed": passed}

        overall = float(np.nanmean(np.abs(self._py - self._mql5)))
        results["overall_mae"] = overall
        results["all_passed"]  = all_pass
        log.info("=" * 60)
        log.info(f"Overall MAE={overall:.6f} | {'ALL PASS' if all_pass else 'FAILURES'}")
        log.info("=" * 60)

        if not all_pass:
            failed = [k for k, v in results.items()
                      if isinstance(v, dict) and not v.get("passed", True)]
            raise ValueError(
                f"Parity test FAILED for: {failed}.  "
                "Check window boundaries, ddof, and epsilon values."
            )
        return results
