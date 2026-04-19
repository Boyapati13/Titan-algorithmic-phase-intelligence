"""
titan_validator.py — Titan V3.0 Statistical Validation Gauntlet
================================================================
Fixes v3.1:
  - Gate 2: uses median(IS_Sharpe) not max(IS_Sharpe) — fairer overfitting measure
  - Gate 7: trade-count annualisation (not fixed 252) — correct for HFT
  - Gate 9: minimum trade count check (new gate)
  - ValidationReport.to_dict() / auto-save to JSON
  - kelly_fraction has upper cap = CFG.validation.max_kelly
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from titan_config import CFG

log = logging.getLogger("TitanValidator")


class DeploymentBlockedError(Exception):
    pass


@dataclass
class GateResult:
    name:      str
    value:     float
    threshold: float
    passed:    bool
    detail:    str = ""

    def __str__(self):
        s = "PASS" if self.passed else "FAIL"
        return f"[{s}] {self.name}: {self.value:.4f} (req {self.threshold}) — {self.detail}"


@dataclass
class ValidationReport:
    gates:      List[GateResult] = field(default_factory=list)
    all_passed: bool             = False
    oos_auc:    float            = 0.0
    wfo_eff:    float            = 0.0
    sharpe:     float            = 0.0
    calmar:     float            = 0.0
    max_dd:     float            = 0.0
    ruin_prob:  float            = 0.0
    perm_pval:  float            = 1.0
    drift_mae:  float            = 1.0
    kelly:      float            = 0.0
    n_trades:   int              = 0

    def summary(self) -> str:
        lines = ["", "=" * 62, "  TITAN V3.0  VALIDATION REPORT", "=" * 62]
        for g in self.gates:
            lines.append(f"  {g}")
        lines += ["=" * 62,
                  f"  VERDICT: {'CLEARED FOR DEPLOYMENT' if self.all_passed else 'DEPLOYMENT BLOCKED'}",
                  "=" * 62]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gates"] = [str(g) for g in self.gates]
        return d

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))
        log.info(f"ValidationReport saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────

def _max_drawdown(cum: np.ndarray) -> float:
    if len(cum) == 0:
        return 0.0
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / (np.abs(peak) + 1e-10)
    return float(np.abs(dd.min()))


def _sharpe_hft(r: np.ndarray) -> float:
    """
    HFT-correct Sharpe: annualise by trade frequency, not 252 days.
    Annualisation factor = sqrt(n_trades_per_year).
    Assumes each element of r is one trade return.
    For n total trades over ~252 active trading days:
      n_per_day = n / 252, ann = sqrt(n_per_day * 252) = sqrt(n)
    """
    if r.std(ddof=1) < 1e-10 or len(r) < 2:
        return 0.0
    ann = float(np.sqrt(len(r)))   # sqrt(n_trades) annualisation
    return float(r.mean() / r.std(ddof=1) * ann)


def _calmar(r: np.ndarray) -> float:
    mdd = _max_drawdown(np.cumsum(r))
    if mdd < 1e-10:
        return 0.0
    # Use the same HFT annualisation for consistency
    ann_ret = float(r.mean() * np.sqrt(len(r)))
    return ann_ret / mdd


def _ruin_prob(mc_paths: np.ndarray, threshold: float = None) -> float:
    thr = threshold or CFG.validation.ruin_threshold
    return float((mc_paths.min(axis=1) < thr).mean())


# ─────────────────────────────────────────────────────────────────────────────
class TitanValidator:
    """Run all 9 validation gates (8 original + Gate 9: minimum trade count)."""

    def __init__(self):
        self.vc     = CFG.validation
        self.report = ValidationReport()

    # Gate 1 — OOS AUC
    def _g1(self, y_true, y_score) -> GateResult:
        auc = float(roc_auc_score(y_true, y_score)) \
              if len(np.unique(y_true)) >= 2 else 0.5
        self.report.oos_auc = auc
        return GateResult("OOS AUC", auc, self.vc.min_auc, auc >= self.vc.min_auc,
                          "Model discrimination on unseen data.")

    # Gate 2 — WFO Efficiency (median IS Sharpe)
    def _g2(self, oos_sharpes: List[float], is_sharpes: List[float]) -> GateResult:
        is_ref  = float(np.median(is_sharpes)) if is_sharpes else 0.0
        eff     = float(np.mean(oos_sharpes) / (is_ref + 1e-10))
        self.report.wfo_eff = eff
        return GateResult("WFO Efficiency", eff, self.vc.min_wfo_efficiency,
                          eff >= self.vc.min_wfo_efficiency,
                          "OOS_Sharpe / median(IS_Sharpe) — overfitting measure.")

    # Gate 3 — Permutation test
    def _g3(self, r: np.ndarray) -> GateResult:
        actual = _sharpe_hft(r)
        rng    = np.random.default_rng(42)
        null   = [_sharpe_hft(rng.permutation(r)) for _ in range(self.vc.n_permutations)]
        pval   = float((np.array(null) >= actual).mean())
        self.report.perm_pval = pval
        return GateResult("Permutation p-value", pval, self.vc.max_perm_pval,
                          pval <= self.vc.max_perm_pval,
                          f"Actual Sharpe={actual:.3f} vs "
                          f"{self.vc.n_permutations} shuffles.")

    # Gate 4 — Monte Carlo max drawdown
    def _g4(self, r: np.ndarray) -> Tuple[GateResult, np.ndarray]:
        rng   = np.random.default_rng(42)
        n     = len(r)
        paths = np.array([
            np.cumsum(rng.choice(r, size=n, replace=True))
            for _ in range(self.vc.n_mc_paths)
        ])
        mdds  = np.array([_max_drawdown(p) for p in paths])
        mdd95 = float(np.percentile(mdds, self.vc.mc_confidence * 100))
        self.report.max_dd = mdd95
        return (GateResult("MC MaxDD 95th", mdd95, self.vc.max_mc_drawdown,
                           mdd95 <= self.vc.max_mc_drawdown,
                           f"{self.vc.n_mc_paths:,} resampled P&L paths."),
                paths)

    # Gate 5 — Probability of ruin
    def _g5(self, paths: np.ndarray) -> GateResult:
        ruin = _ruin_prob(paths)
        self.report.ruin_prob = ruin
        return GateResult("Prob of Ruin", ruin, self.vc.max_ruin_prob,
                          ruin <= self.vc.max_ruin_prob,
                          "Fraction of MC paths with cum return < -50%.")

    # Gate 6 — Feature drift MAE
    def _g6(self, py: Optional[np.ndarray], mql5: Optional[np.ndarray]) -> GateResult:
        if py is None or mql5 is None:
            self.report.drift_mae = 0.0
            return GateResult("Feature Drift MAE", 0.0, self.vc.max_drift_mae, True, "SKIPPED")
        if py.shape != mql5.shape:
            self.report.drift_mae = 1.0
            return GateResult("Feature Drift MAE", 1.0, self.vc.max_drift_mae, False,
                              f"Shape mismatch {py.shape} vs {mql5.shape}")
        mae = float(np.abs(py - mql5).mean())
        self.report.drift_mae = mae
        return GateResult("Feature Drift MAE", mae, self.vc.max_drift_mae,
                          mae <= self.vc.max_drift_mae,
                          "Python vs MQL5 per-feature MAE on shared ticks.")

    # Gate 7 — Sharpe (HFT annualisation)
    def _g7(self, r: np.ndarray) -> GateResult:
        s = _sharpe_hft(r)
        self.report.sharpe = s
        return GateResult("Ann. Sharpe (HFT)", s, self.vc.min_sharpe,
                          s >= self.vc.min_sharpe,
                          "Trade-count annualisation: Sharpe * sqrt(n_trades).")

    # Gate 8 — Calmar
    def _g8(self, r: np.ndarray) -> GateResult:
        c = _calmar(r)
        self.report.calmar = c
        return GateResult("Calmar Ratio", c, self.vc.min_calmar,
                          c >= self.vc.min_calmar, "Ann Return / Max Drawdown.")

    # Gate 9 — Minimum trade count (new)
    def _g9(self, n_trades: int) -> GateResult:
        passed = n_trades >= self.vc.min_trade_count
        self.report.n_trades = n_trades
        return GateResult("Min Trade Count", float(n_trades),
                          float(self.vc.min_trade_count), passed,
                          "Ensures statistical power for all other gates.")

    def run(self,
            y_true:         np.ndarray,
            y_score:        np.ndarray,
            trade_returns:  np.ndarray,
            oos_sharpes:    List[float],
            is_sharpes:     List[float],
            py_features:    Optional[np.ndarray] = None,
            mql5_features:  Optional[np.ndarray] = None,
            output_dir:     Optional[Path]        = None,
            ) -> ValidationReport:

        log.info("Running validation gauntlet…")
        if len(trade_returns) < 30:
            log.warning("Fewer than 30 trades — statistical power is low.")

        gates: List[GateResult] = []
        gates.append(self._g1(y_true, y_score))
        gates.append(self._g2(oos_sharpes, is_sharpes))
        gates.append(self._g3(trade_returns))
        g4, paths = self._g4(trade_returns)
        gates.append(g4)
        gates.append(self._g5(paths))
        gates.append(self._g6(py_features, mql5_features))
        gates.append(self._g7(trade_returns))
        gates.append(self._g8(trade_returns))
        gates.append(self._g9(len(trade_returns)))   # Gate 9 (new)

        self.report.gates      = gates
        self.report.all_passed = all(g.passed for g in gates)
        log.info(self.report.summary())

        # Auto-save report alongside ONNX export
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.report.save(Path(output_dir) / "validation_report.json")
        else:
            self.report.save(CFG.data.model_dir / "validation_report.json")

        if not self.report.all_passed:
            failed = [g.name for g in gates if not g.passed]
            raise DeploymentBlockedError(
                f"DEPLOYMENT BLOCKED — failed gates: {failed}"
            )
        return self.report


def kelly_fraction(trade_returns: np.ndarray, fractional: float = None) -> float:
    """
    Fractional Kelly with hard upper cap = CFG.validation.max_kelly.
    Prevents dangerous over-sizing from estimation error.
    """
    frac  = fractional or CFG.validation.kelly_fraction
    cap   = CFG.validation.max_kelly
    wins  = trade_returns[trade_returns > 0]
    losses= trade_returns[trade_returns < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.01
    p     = len(wins) / len(trade_returns)
    b     = wins.mean() / abs(losses.mean())
    f_raw = max(0.0, (p * b - (1 - p)) / (b + 1e-10))
    f_out = min(f_raw * frac, cap)   # hard upper cap
    log.info(f"Kelly: p={p:.3f} b={b:.3f} f*={f_raw:.4f} "
             f"deploy={f_out:.4f} (cap={cap})")
    return float(f_out)
