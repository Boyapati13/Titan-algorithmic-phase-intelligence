"""
check_env.py — Titan V3.0 Environment Validation Script
==========================================================
Run before first use to verify all dependencies are installed
and meet minimum version requirements.

Usage: python check_env.py
Exit code 0 = all good.  Exit code 1 = missing/outdated packages.
"""
import sys
import importlib
import importlib.metadata

# ── Minimum required versions ─────────────────────────────────────────────────
MIN_VERSIONS = {
    "numpy":        "1.26.0",
    "pandas":       "2.0.0",
    "scipy":        "1.12.0",
    "sklearn":      "1.4.0",
    "pyarrow":      "15.0.0",
    "torch":        "2.2.0",
    "onnx":         "1.16.0",
    "onnxruntime":  "1.18.0",
    "ripser":       "0.6.0",
    "xgboost":      "2.0.0",
    "shap":         "0.44.0",
    "optuna":       "3.0.0",
}

# Packages that are optional (missing → warning, not failure)
OPTIONAL = {"ripser", "MetaTrader5"}


def _parse_ver(s: str) -> tuple:
    try:
        return tuple(int(x) for x in s.split(".")[:3])
    except ValueError:
        return (0,)


def check_package(pkg: str, min_ver: str) -> tuple[bool, str]:
    """Returns (ok, message)."""
    # importlib.metadata package name may differ from import name
    pkg_names = {
        "sklearn": "scikit-learn",
        "cv2":     "opencv-python",
    }
    meta_name = pkg_names.get(pkg, pkg)
    try:
        version = importlib.metadata.version(meta_name)
        ok      = _parse_ver(version) >= _parse_ver(min_ver)
        msg     = f"{'✓' if ok else '✗'}  {pkg:<18} {version:>12}  (req >= {min_ver})"
        return ok, msg
    except importlib.metadata.PackageNotFoundError:
        optional = pkg in OPTIONAL
        sym      = "⚠" if optional else "✗"
        msg      = f"{sym}  {pkg:<18} {'NOT INSTALLED':>12}  (req >= {min_ver})"
        return optional, msg   # optional missing = soft pass


def check_cuda() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"✓  CUDA available: {name} (device count: {torch.cuda.device_count()})"
        return "⚠  CUDA not available — CPU training only (slower)"
    except ImportError:
        return "✗  torch not installed — cannot check CUDA"


def check_lz4() -> tuple[bool, str]:
    try:
        import lz4.frame
        return True, "✓  lz4.frame available (Parquet compression)"
    except ImportError:
        return False, "✗  lz4 not installed — pip install lz4"


def main():
    print("=" * 60)
    print("  TITAN V3.0  ENVIRONMENT CHECK")
    print("=" * 60)

    all_ok    = True
    results   = []

    for pkg, min_v in MIN_VERSIONS.items():
        ok, msg = check_package(pkg, min_v)
        results.append((ok, msg))
        if not ok:
            all_ok = False

    for _, msg in results:
        print(msg)

    # LZ4 check (separate — not discoverable via importlib)
    lz4_ok, lz4_msg = check_lz4()
    print(lz4_msg)
    if not lz4_ok:
        all_ok = False

    print()
    print(check_cuda())
    print()

    # Python version
    pv = sys.version_info
    py_ok = pv >= (3, 9)
    print(f"{'✓' if py_ok else '✗'}  Python {pv.major}.{pv.minor}.{pv.micro}  (req >= 3.9)")
    if not py_ok:
        all_ok = False

    print()
    print("=" * 60)
    if all_ok:
        print("  RESULT: ALL CHECKS PASSED — environment is ready.")
    else:
        print("  RESULT: ISSUES FOUND — see ✗ entries above.")
        print("  Run: pip install -r requirements.txt")
    print("=" * 60)

    # Exit 1 for CI pipelines to detect failure
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
