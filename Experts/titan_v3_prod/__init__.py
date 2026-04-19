"""Titan V3.0 patch 3.2 — all 8 audit errors corrected."""
from titan_config   import CFG, TitanConfig, FEATURE_COLS
from titan_features import compute_features, validate_input, build_sequences
from titan_labeler  import TitanLabeler
from titan_model    import TitanLSTMV3, WFOTrainer, build_loss_fn, predict_batch
from titan_validator import TitanValidator, DeploymentBlockedError, kelly_fraction
from titan_onnx     import export_onnx, FeatureParityTester
from titan_data     import load_parquet, ParquetConverter
from titan_logger   import TitanLogger, FeatureRecord, SignalRecord, TradeRecord
from titan_mt5_bridge import MT5Bridge
from titan_pipeline import TitanPipeline
from titan_baselines import BaselineSuite
from titan_optimizer import TitanOptimizer

__version__ = "3.2.0"
