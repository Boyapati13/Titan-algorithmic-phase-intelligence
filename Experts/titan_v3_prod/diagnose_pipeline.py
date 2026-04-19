"""
Quick diagnostics to identify pipeline failure points.
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from titan_config import CFG, FEATURE_COLS
from titan_data import load_parquet
from titan_features import validate_input, compute_features, build_sequences
from titan_labeler import TitanLabeler
from titan_model import WFOTrainer
from titan_validator import TitanValidator
from titan_onnx import export_onnx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("Diagnose")

def test_paths():
    """Verify all data paths exist and are readable."""
    log.info("=" * 60)
    log.info("PATH DIAGNOSTICS")
    log.info("=" * 60)

    checks = [
        ("parquet_dir", CFG.data.parquet_dir),
        ("model_dir", CFG.data.model_dir),
        ("output_dir", CFG.data.output_dir),
        ("log_dir", CFG.data.log_dir),
    ]

    for name, path in checks:
        exists = path.exists()
        is_dir = path.is_dir() if exists else False
        status = "✓ exists" if exists and is_dir else "✗ missing or not a directory"
        log.info(f"{name:15s}: {path} {status}")

    # Check for parquet file
    parquet_file = CFG.data.parquet_dir / "EURUSD_ticks.parquet"
    if parquet_file.exists():
        size_mb = parquet_file.stat().st_size / 1024 / 1024
        log.info(f"EURUSD_ticks.parquet: {size_mb:.1f} MB ✓")
    else:
        log.error(f"EURUSD_ticks.parquet: NOT FOUND ✗")

    log.info("=" * 60 + "\n")

def test_phase_1():
    """Test data loading."""
    log.info("=" * 60)
    log.info("PHASE 1: DATA LOADING")
    log.info("=" * 60)

    try:
        parquet_file = CFG.data.parquet_dir / "EURUSD_ticks.parquet"
        log.info(f"Loading {parquet_file}...")
        raw_df = load_parquet([str(parquet_file)])
        log.info(f"  ✓ Loaded {len(raw_df):,} rows")

        log.info("Validating input...")
        validated_df = validate_input(raw_df)
        log.info(f"  ✓ Validated {len(validated_df):,} rows")
        log.info(f"  Shape: {validated_df.shape}, dtypes: {validated_df.dtypes.to_dict()}")
        return validated_df
    except Exception as e:
        log.error(f"Phase 1 FAILED: {e}", exc_info=True)
        return None

def test_phase_2(raw_df):
    """Test feature computation."""
    log.info("=" * 60)
    log.info("PHASE 2: FEATURE COMPUTATION (SAMPLE — 10K ROWS)")
    log.info("=" * 60)

    try:
        # Sample 10K rows to test
        sample = raw_df.iloc[:10000].copy()
        log.info(f"Computing features on {len(sample):,} rows sample...")
        features_df = compute_features(sample)
        log.info(f"  ✓ Computed {len(features_df):,} features")
        log.info(f"  Columns: {list(features_df.columns)}")
        log.info(f"  Expected: {FEATURE_COLS}")

        missing = set(FEATURE_COLS) - set(features_df.columns)
        if missing:
            log.error(f"  ✗ Missing columns: {missing}")
        return features_df
    except Exception as e:
        log.error(f"Phase 2 FAILED: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    test_paths()
    raw_df = test_phase_1()
    if raw_df is not None:
        test_phase_2(raw_df)
