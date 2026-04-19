# Titan V3.0 — Algorithmic Phase Intelligence System

**HFT OTC Forex | Persistent Homology | Thermodynamic Phase Detection | LSTM | MQL5**

---

## Architecture

```
MT5 Tick Data  →  titan_features.py  →  titan_labeler.py  →  titan_model.py
                  (16 features)          (GMM/EMA labels)     (WFO LSTM)
                        ↓                                           ↓
               titan_baselines.py                        titan_validator.py
               (3 comparison models)                     (8 gates)
                                                               ↓
                                                        titan_onnx.py
                                                        (→ TitanEA_V3.mq5)
```

## Modules

| File | Purpose |
|------|---------|
| `titan_config.py` | Single source of truth — all hyperparameters |
| `titan_features.py` | 16 features: OFP, TDA, thermodynamics, algo-signatures |
| `titan_labeler.py` | Phase 3.5 validation + GMM + causal EMA + phase transition labels |
| `titan_model.py` | LSTM, WeightedBCE, WFO training |
| `titan_validator.py` | 9-gate statistical validation gauntlet |
| `titan_baselines.py` | Naive logit, OFP XGBoost, TWAP crossover, Random baselines |
| `titan_onnx.py` | ONNX export + Python/MQL5 feature parity CI |
| `titan_data.py` | MT5 binary parser, Parquet I/O |
| `titan_logger.py` | Non-blocking logger: features, signals, trades, drift |
| `titan_mt5_bridge.py` | Direct MT5 Python bridge (Windows only) |
| `titan_pipeline.py` | 7-phase orchestrator |
| `titan_optimizer.py` | Optuna Bayesian hyperparameter search |
| `TitanEA_V3.mq5` | MQL5 live execution engine (OnBookEvent) |

## Quick Start

```bash
pip install -r requirements.txt
python titan_pipeline.py data/parquet/EURUSD_2025.parquet
```

## Feature Groups

- **Group A — Order Flow Physics**: FDPI, MVDI, TWKJ, QAD, SGC, HURST
- **Group B — Topological State**: TOPO_H0, TOPO_H1 (persistent homology)
- **Group C — Thermodynamic Phase**: MFE (Helmholtz Free Energy), PTP
- **Group D — Algo Signatures**: TWAP_PROB, MOM_IGNITE, ICE_SCORE, TCE
- **Group E — Context**: HOUR_SINE, HOUR_COSINE

## Validation Gates

All 9 must pass before deployment:

1. OOS AUC ≥ 0.65
2. WFO Efficiency ≥ 0.60
3. Permutation p-value ≤ 0.05
4. MC Max Drawdown (95th) ≤ 25%
5. Probability of Ruin ≤ 1%
6. Feature Drift MAE ≤ 0.02
7. Annualised Sharpe ≥ 1.50
8. Calmar Ratio ≥ 0.80
9. Min Trade Count ≥ 100

## OTC FX Data Contract

**NO real volume is used anywhere.** All features derive exclusively from:
`Bid_Price | Ask_Price | Flags | Tick_Time_ms`

## Platform Note

`titan_mt5_bridge.py` requires Windows + MetaTrader5 terminal.
All other modules are platform-independent.
