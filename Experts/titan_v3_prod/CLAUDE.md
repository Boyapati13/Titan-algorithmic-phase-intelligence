# Titan V3.0 — Project Architecture

## Central Components (God Nodes)

| File | Role | Depends on |
|---|---|---|
| `titan_config.py` | Singleton CFG — all hyperparams, all data paths | nothing (root) |
| `titan_pipeline.py` | Orchestrator — runs phases 1-7, `__main__` entry point | all modules |
| `titan_features.py` | Feature engineering (16 features) + `validate_input` + `build_sequences` | titan_config |
| `titan_data.py` | Parquet loading, CSV conversion, FlagMapper | titan_config |
| `titan_model.py` | LSTM model, WFO trainer | titan_config |

## Full Dependency Graph

```
titan_pipeline.py
├── titan_config.py          (CFG singleton, all paths & thresholds)
├── titan_data.py            (load_parquet → raw DataFrame)
│   └── titan_config.py
├── titan_features.py        (validate_input, compute_features, build_sequences)
│   └── titan_config.py
├── titan_labeler.py         (GMM/HMM phase labels)
│   └── titan_config.py
├── titan_model.py           (TitanLSTMV3, WFOTrainer, predict_batch)
│   └── titan_config.py
├── titan_validator.py       (9-gate validation gauntlet, DeploymentBlockedError)
│   └── titan_config.py
├── titan_onnx.py            (ONNX export, FeatureParityTester)
│   └── titan_config.py
└── titan_baselines.py       (4 baseline models for comparison)
    └── titan_config.py
```

## Pipeline Phases

| Phase | Function | Key bottleneck |
|---|---|---|
| 1 | Load parquet → raw_df | I/O (626 MB file, 88M rows) |
| 2 | compute_features → 16 features | RAM (rolling ops) + TDA (ripser) |
| 3 | TitanLabeler → labels + build_sequences | RAM (max_sequences=500K cap) |
| 4 | BaselineSuite (4 models) | CPU |
| 5 | WFOTrainer (8-fold LSTM) | GPU/CPU |
| 6 | TitanValidator (9 gates) | — |
| 7 | export_onnx → TitanV3.onnx | — |

## Key Configuration (titan_config.py)

- `CFG.data.parquet_dir` — absolute path to `data/parquet/` (anchored to `__file__`)
- `CFG.data.model_dir` — absolute path to `models/`
- `CFG.features.tda_subsample = 1000` — 88K ripser calls on full dataset (~10 min)
- `CFG.features.window_n = 128` — sequence length (must equal model.sequence_len)
- `CFG.model.input_size = 16` — must equal len(FEATURE_COLS)
- `CFG.validation.min_auc = 0.65` — deployment gate threshold

## Live EA

`TitanEA_V3.mq5` runs **independently** of the Python pipeline.
- Loads ONNX from embedded resource `\\Files\\TitanV3.onnx`
- Retraining requires running the pipeline, then copying `models/TitanV3.onnx` → MT5 Files dir and recompiling the EA

## Known Issues Fixed (Session 1)

- MT5 passes `unknown 1` as argv — filtered in `titan_pipeline.py:__main__`
- `titan_config.py` paths now anchored to `Path(__file__).parent` (not CWD)
- `validate_input` uses numpy masking (no pandas block consolidation OOM)
- `_zscore_clip` uses `np.clip(out=arr)` (in-place, saves 674 MB per call)
- `build_sequences` caps at 500K sequences with adaptive stride
- `tda_subsample` raised 50 → 1000 for feasible runtime on 88M ticks

## OOM Fixes Applied (Session 2 — Current)

### Phase 1: Data Loading (titan_data.py, load_parquet)
**Problem:** 88M-row parquet file caused OOM during load due to intermediate allocations in row-group iteration and concatenation.

**Solution:** Switched to PyArrow's `pq.read_table()` (single efficient read) instead of row-group iteration:
```python
# OLD: For each row group, read → convert → concat → OOM on 76th row group
pf = pq.ParquetFile(p)
for i in range(n_row_groups):
    rg_table = pf.read_row_group(i, columns=required_cols)  # OOM during concat

# NEW: Single read, then downcast after (no intermediate allocations)
pf = pq.read_table(p, columns=required_cols)
file_df = pf.to_pandas()
file_df["Bid"] = file_df["Bid"].astype(np.float32)  # Downcast to save memory
file_df["Ask"] = file_df["Ask"].astype(np.float32)
```
Result: Successfully loads 88.3M rows in 20 seconds without OOM.

### Phase 2: Feature Computation (titan_features.py, compute_features)
**Problem:** Rolling window operations on 88M rows allocate ~674 MB temporary arrays per operation, causing OOM.

**Solution:** Implemented chunking dispatcher with warm-up overlap (lines 271–307):
- Splits 88M rows into 9 chunks of 10M rows each
- Prepends last 128 rows from previous chunk as rolling-window warm-up
- Recursively computes features on each padded chunk (fits in RAM)
- Strips warm-up rows before concatenating results
- Each chunk uses ~800 MB peak RAM (safe on 16 GB machine)

Result: Feature computation on 88M rows completes without OOM.

### Memory Strategy Summary
| Operation | Before | After | Savings |
|---|---|---|---|
| Parquet load | Row-group iter → OOM at 76/89 | Single PyArrow read | 100% success |
| Bid/Ask dtype | float64 in memory | float32 (downcast after load) | 50% |
| Feature compute | All 88M rows at once → OOM | 10M chunks with 128-row overlap | ~90% peak RAM reduction |

### Files Modified
- **`titan_data.py`** — `load_parquet()` function (lines 197–250)
  - Replaced row-group iteration with `pq.read_table()`
  - Added float32 downcast for Bid/Ask
  
- **`titan_features.py`** — `compute_features()` function (lines 271–307)
  - Added chunking dispatcher with CHUNK_ROWS=10M, WARMUP=128
  - Preserves rolling window correctness via warm-up overlap
