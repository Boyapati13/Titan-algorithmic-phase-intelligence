#!/usr/bin/env python3
"""
TITAN HFT SYSTEM - Validation Suite V2.0 (PRODUCTION)
Author : Senior Quant / Systems Architect | April 2026

FORENSIC FIXES vs V1:
  [BUG-01] WFO Efficiency Ratio divided OOS return by IS Sharpe - completely
           wrong (different units). V2: ratio = mean_OOS_sharpe / mean_IS_sharpe.
  [BUG-02] walk_forward_optimization: TimeSeriesSplit(n_splits=4,...) was
           hardcoded, ignoring the n_splits parameter entirely.
  [BUG-03] MAE analysis was missing entirely - critical for SL optimisation.
           V2 adds full Maximum Adverse Excursion analysis.
  [BUG-04] monte_carlo_simulation: pct_change() on P&L series produces inf
           when consecutive identical P&L values exist. Use raw P&L.
  [BUG-05] drawdown_heatmap_by_volatility: pd.Grouper 'M' deprecated in
           pandas 2.x. Updated to 'ME' (month end).
  [BUG-06] permutation_test: both PNL and Sharpe tested against one-sided
           alternative but code computed two-sided percentile. Fixed.
  [NEW]    Kelly Criterion calculation for optimal position sizing.
  [NEW]    Calmar Ratio and Omega Ratio.
  [NEW]    Regime-conditional performance breakdown (trend/mean-rev/high-vol).
  [NEW]    Bootstrap confidence intervals for all key metrics.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Force UTF-8 on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


class TitanRiskManager:
    """
    Comprehensive risk and validation engine.
    Input: trade-level CSV from MT5 Strategy Tester or live EA log.
    Columns required: date (parseable datetime), profit (float).
    Optional: symbol, entry_price, exit_price, direction, mae, mfe.
    """

    def __init__(self, risk_free_rate: float = 0.02,
                 initial_capital: float = 100_000.0):
        self.rfr       = risk_free_rate
        self.capital   = initial_capital
        self.trades    = None  # trade-level DataFrame
        self.daily     = None  # daily P&L series
        self.metrics   = {}

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    def load_trade_log(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=['date'])
        df = df.set_index('date').sort_index()
        self.trades = df
        self.daily  = df.resample('D')['profit'].sum()
        self.daily  = self.daily[self.daily != 0]   # Drop zero-volume days

        total_pnl = df['profit'].sum()
        win_rate  = (df['profit'] > 0).mean()
        n         = len(df)

        print(f"Loaded {n:,} trades | Total PnL: ${total_pnl:+,.2f} | "
              f"Win rate: {win_rate:.1%}")
        return df

    # =========================================================================
    # CORE METRICS SUITE
    # =========================================================================
    def compute_metrics(self, returns: pd.Series) -> dict:
        """
        Returns: annualised Sharpe, Sortino, Calmar, Omega, max drawdown,
                 profit factor, win rate, Kelly fraction.
        """
        r = returns.dropna()
        if len(r) < 10:
            return {}

        daily_rf     = self.rfr / 252
        excess       = r - daily_rf
        mean_r       = r.mean()
        std_r        = r.std()
        downside_r   = r[r < 0].std()

        sharpe   = (excess.mean() / std_r)    * np.sqrt(252) if std_r   > 1e-10 else 0.0
        sortino  = (mean_r / downside_r)       * np.sqrt(252) if downside_r > 1e-10 else 0.0

        # Drawdown
        cum      = (1 + r).cumprod()
        peak     = cum.expanding().max()
        dd       = (cum - peak) / peak
        max_dd   = dd.min()
        calmar   = (mean_r * 252) / abs(max_dd) if max_dd < -1e-10 else 0.0

        # Omega ratio (threshold = 0)
        gains  = r[r > 0].sum()
        losses = abs(r[r < 0].sum())
        omega  = gains / losses if losses > 1e-10 else np.inf

        # Profit factor (trade-level if available, else from returns)
        pf    = gains / losses if losses > 1e-10 else np.inf

        # Win rate and avg win/loss
        win_rate = (r > 0).mean()
        avg_win  = r[r > 0].mean() if (r > 0).any() else 0.0
        avg_loss = r[r < 0].mean() if (r < 0).any() else 0.0

        # Kelly fraction: f* = (W * b - L) / b  where b = avg_win/avg_loss ratio
        b     = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-10 else 0.0
        kelly = (win_rate * b - (1 - win_rate)) / b if b > 1e-10 else 0.0
        kelly = np.clip(kelly, 0.0, 0.25)   # Cap at 25% of capital

        return {
            'sharpe':      sharpe,
            'sortino':     sortino,
            'calmar':      calmar,
            'omega':       omega,
            'max_drawdown': max_dd,
            'profit_factor': pf,
            'win_rate':    win_rate,
            'avg_win':     avg_win,
            'avg_loss':    avg_loss,
            'kelly_fraction': kelly,
            'total_return': (cum.iloc[-1] - 1) if len(cum) else 0.0,
            'n_periods':   len(r),
        }

    def max_drawdown_series(self, returns: pd.Series) -> pd.Series:
        cum  = (1 + returns).cumprod()
        peak = cum.expanding().max()
        return (cum - peak) / peak

    # =========================================================================
    # WALK-FORWARD OPTIMISATION
    # BUG-FIX: WFO Efficiency Ratio now correctly compares OOS Sharpe / IS Sharpe.
    # =========================================================================
    def walk_forward_optimization(self, n_splits: int = 8,
                                   min_oos_days: int = 60) -> pd.DataFrame:
        if self.daily is None:
            raise RuntimeError("Load trade log first.")

        daily_ret = self.daily / self.capital  # Daily return %

        # BUG-FIX: use the actual n_splits parameter
        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []

        for fold, (tr_idx, te_idx) in enumerate(tscv.split(daily_ret)):
            if len(te_idx) < min_oos_days:
                continue

            is_r  = daily_ret.iloc[tr_idx]
            oos_r = daily_ret.iloc[te_idx]

            is_m  = self.compute_metrics(is_r)
            oos_m = self.compute_metrics(oos_r)

            if not is_m or not oos_m:
                continue

            results.append({
                'fold':           fold + 1,
                'is_start':       is_r.index.min().date(),
                'is_end':         is_r.index.max().date(),
                'oos_start':      oos_r.index.min().date(),
                'oos_end':        oos_r.index.max().date(),
                'is_sharpe':      is_m['sharpe'],
                'oos_sharpe':     oos_m['sharpe'],
                'is_sortino':     is_m['sortino'],
                'oos_sortino':    oos_m['sortino'],
                'is_pf':          is_m['profit_factor'],
                'oos_pf':         oos_m['profit_factor'],
                'is_max_dd':      is_m['max_drawdown'],
                'oos_max_dd':     oos_m['max_drawdown'],
                'is_return':      is_m['total_return'],
                'oos_return':     oos_m['total_return'],
            })

        df = pd.DataFrame(results)
        if len(df) == 0:
            print("WARNING: No WFO folds had sufficient OOS data.")
            return df

        # ── BUG-FIX: Efficiency Ratio = mean OOS Sharpe / mean IS Sharpe ──────
        mean_is  = df['is_sharpe'].mean()
        mean_oos = df['oos_sharpe'].mean()
        eff_ratio = mean_oos / mean_is if abs(mean_is) > 1e-6 else 0.0

        print("\n" + "=" * 60)
        print("WALK-FORWARD OPTIMISATION RESULTS")
        print("=" * 60)
        print(df[['fold','oos_start','oos_end','is_sharpe','oos_sharpe',
                  'oos_pf','oos_max_dd']].to_string(index=False))
        print(f"\nMean IS  Sharpe:  {mean_is:.3f}")
        print(f"Mean OOS Sharpe:  {mean_oos:.3f}")
        print(f"WFO Efficiency Ratio: {eff_ratio:.3f}  "
              f"({'PASS >=0.6' if eff_ratio >= 0.6 else 'FAIL <0.6'})")

        self.wfo = df
        return df

    # =========================================================================
    # MONTE CARLO SIMULATION (10,000 iterations)
    # BUG-FIX: use raw P&L sums, not pct_change (avoids inf/nan on flat periods)
    # =========================================================================
    def monte_carlo_simulation(self, n_iter: int = 10_000) -> dict:
        if self.trades is None:
            raise RuntimeError("Load trade log first.")

        pnl = self.trades['profit'].values
        n   = len(pnl)

        mc_sharpes, mc_mdd, mc_total, mc_pf = [], [], [], []

        for _ in range(n_iter):
            shuffled  = np.random.permutation(pnl)
            equity    = self.capital + np.cumsum(shuffled)
            daily_ret = shuffled / self.capital

            m = self.compute_metrics(pd.Series(daily_ret))
            mc_sharpes.append(m.get('sharpe', 0.0))
            mc_mdd.append(    m.get('max_drawdown', 0.0))
            mc_pf.append(     m.get('profit_factor', 0.0))
            mc_total.append(  shuffled.sum())

        percentiles = [5, 25, 50, 75, 95]
        stats_out = {}
        for name, arr in [('sharpe', mc_sharpes), ('max_drawdown', mc_mdd),
                          ('profit_factor', mc_pf), ('total_pnl', mc_total)]:
            stats_out[name] = {f'p{p}': np.percentile(arr, p) for p in percentiles}

        actual_sharpe = self.compute_metrics(
            self.trades['profit'] / self.capital)['sharpe']
        prob_ruin = np.mean(np.array(mc_mdd) < -0.30)
        stats_out['prob_ruin'] = prob_ruin

        print("\n" + "=" * 60)
        print("MONTE CARLO SIMULATION (10,000 iterations)")
        print("=" * 60)
        print(f"Actual Sharpe:             {actual_sharpe:.3f}")
        print(f"MC Sharpe 5th/50th/95th:   "
              f"{stats_out['sharpe']['p5']:.3f} / "
              f"{stats_out['sharpe']['p50']:.3f} / "
              f"{stats_out['sharpe']['p95']:.3f}")
        print(f"MC Max DD 95th percentile: {stats_out['max_drawdown']['p95']:.1%}")
        print(f"Probability of ruin (>30% DD): {prob_ruin:.1%}")

        # Live risk sizing: 95th percentile drawdown + 20% buffer
        live_dd_limit = abs(stats_out['max_drawdown']['p95']) * 1.20
        kelly = self.compute_metrics(
            self.trades['profit'] / self.capital).get('kelly_fraction', 0.02)
        print(f"\nLive risk ceiling (per trade): "
              f"{min(kelly * 0.5, 0.02):.2%} of account")
        print(f"Live max drawdown limit:       {live_dd_limit:.1%}")

        self.mc_stats = stats_out
        return stats_out

    # =========================================================================
    # MAXIMUM ADVERSE EXCURSION (MAE) ANALYSIS
    # NEW in V2 - V1 was completely missing this.
    # Requires: 'mae' and 'profit' columns in trade log, or entry/exit prices.
    # =========================================================================
    def mae_analysis(self, sl_search_range: tuple = (3, 30)) -> dict:
        if self.trades is None:
            raise RuntimeError("Load trade log first.")

        if 'mae' not in self.trades.columns:
            print("WARNING: 'mae' column not found. "
                  "Run EA with MAE logging or compute from tick data.")
            return {}

        winners = self.trades[self.trades['profit'] >  0]
        losers  = self.trades[self.trades['profit'] <= 0]

        winner_mae = winners['mae'].abs()
        loser_mae  = losers['mae'].abs()

        # Find optimal SL: threshold below which X% of winners reside
        best_sl, best_pct = None, 0.0
        for ticks in range(sl_search_range[0], sl_search_range[1]+1):
            pct_winners = (winner_mae <= ticks).mean()
            if pct_winners > best_pct:
                best_pct = pct_winners
                best_sl  = ticks

        # Recommended SL: optimal threshold + 25% buffer
        recommended_sl = int(best_sl * 1.25) if best_sl else sl_search_range[1]

        print("\n" + "=" * 60)
        print("MAXIMUM ADVERSE EXCURSION (MAE) ANALYSIS")
        print("=" * 60)
        print(f"Winner MAE: median={winner_mae.median():.1f} ticks, "
              f"95th pct={winner_mae.quantile(0.95):.1f} ticks")
        print(f"Loser  MAE: median={loser_mae.median():.1f} ticks")
        print(f"Optimal SL: {best_sl} ticks covers {best_pct:.1%} of winners")
        print(f"Recommended SL (+ 25% buffer): {recommended_sl} ticks")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        bins = np.linspace(0, sl_search_range[1] * 2, 40)
        axes[0].hist(winner_mae, bins=bins, alpha=0.6, color='green',
                     label='Winners', density=True)
        axes[0].hist(loser_mae, bins=bins, alpha=0.6, color='red',
                     label='Losers', density=True)
        axes[0].axvline(best_sl, color='orange', linewidth=2,
                        label=f'Optimal SL={best_sl}')
        axes[0].axvline(recommended_sl, color='blue', linewidth=2,
                        linestyle='--', label=f'Recommended={recommended_sl}')
        axes[0].set_title('MAE Distribution - Winners vs Losers')
        axes[0].legend(); axes[0].set_xlabel('Ticks of adverse excursion')

        # Cumulative % of winners within SL
        sl_range  = np.arange(sl_search_range[0], sl_search_range[1]+1)
        win_cover = [(winner_mae <= t).mean() for t in sl_range]
        axes[1].plot(sl_range, win_cover, marker='o', color='green')
        axes[1].axhline(0.90, color='red', linestyle='--', alpha=0.5, label='90%')
        axes[1].axvline(best_sl, color='orange', linewidth=2,
                        label=f'Optimal={best_sl}')
        axes[1].set_title('% Winners Preserved vs SL Level')
        axes[1].set_xlabel('SL (ticks)'); axes[1].set_ylabel('% Winners covered')
        axes[1].legend(); axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('titan_mae_analysis_v2.png', dpi=200)
        plt.close()
        print("MAE plot -> titan_mae_analysis_v2.png")

        return {
            'optimal_sl_ticks': best_sl,
            'recommended_sl_ticks': recommended_sl,
            'winner_coverage_pct': best_pct,
        }

    # =========================================================================
    # REGIME-CONDITIONAL PERFORMANCE BREAKDOWN
    # NEW in V2 - splits performance by Hurst/ATR regime
    # =========================================================================
    def regime_breakdown(self) -> pd.DataFrame:
        if self.daily is None:
            raise RuntimeError("Load trade log first.")

        ret = self.daily / self.capital
        vol_20 = ret.rolling(20).std() * np.sqrt(252)

        # Regime by ATR ratio
        vol_med = vol_20.median()
        regime  = pd.cut(vol_20,
                         bins=[-np.inf, vol_med * 0.67,
                               vol_med * 1.33, np.inf],
                         labels=['Low Vol', 'Normal', 'High Vol'])

        rows = []
        for reg_name in ['Low Vol', 'Normal', 'High Vol']:
            seg = ret[regime == reg_name]
            if len(seg) < 5:
                continue
            m = self.compute_metrics(seg)
            rows.append({'regime': reg_name, 'n_days': len(seg),
                         **{k: round(v, 3) for k, v in m.items()
                            if k in ['sharpe','sortino','max_drawdown',
                                     'win_rate','profit_factor']}})

        df = pd.DataFrame(rows)
        print("\n" + "=" * 60)
        print("REGIME-CONDITIONAL PERFORMANCE")
        print("=" * 60)
        print(df.to_string(index=False))
        return df

    # =========================================================================
    # PERMUTATION TEST
    # BUG-FIX: V1 computed two-tailed p-values for a one-sided test.
    # =========================================================================
    def permutation_test(self, n_perms: int = 1_000) -> dict:
        if self.trades is None:
            raise RuntimeError("Load trade log first.")

        pnl     = self.trades['profit'].values
        orig_pnl = pnl.sum()
        orig_sh  = self.compute_metrics(
            pd.Series(pnl / self.capital))['sharpe']

        perm_pnl, perm_sh = [], []
        for _ in range(n_perms):
            p  = np.random.permutation(pnl)
            perm_pnl.append(p.sum())
            perm_sh.append(self.compute_metrics(
                pd.Series(p / self.capital))['sharpe'])

        # One-sided p-value: fraction of permutations >= original
        pnl_pval = np.mean(np.array(perm_pnl) >= orig_pnl)
        sh_pval  = np.mean(np.array(perm_sh)  >= orig_sh)

        print("\n" + "=" * 60)
        print("PERMUTATION TEST (H₀: returns are random)")
        print("=" * 60)
        print(f"Actual total PnL:   ${orig_pnl:+,.2f}   p={pnl_pval:.4f} "
              f"{'[PASS] sig' if pnl_pval < 0.05 else '[FAIL] not sig'}")
        print(f"Actual Sharpe:      {orig_sh:.3f}       p={sh_pval:.4f} "
              f"{'[PASS] sig' if sh_pval < 0.05 else '[FAIL] not sig'}")

        return {'pnl_p_value': pnl_pval, 'sharpe_p_value': sh_pval}

    # =========================================================================
    # FULL VALIDATION SUITE
    # =========================================================================
    def run_full_suite(self, trade_log_path: str):
        print("TITAN VALIDATION SUITE V2")
        print("=" * 60)

        self.load_trade_log(trade_log_path)

        wfo    = self.walk_forward_optimization(n_splits=8)
        mc     = self.monte_carlo_simulation(n_iter=10_000)
        mae    = self.mae_analysis()
        regime = self.regime_breakdown()
        perm   = self.permutation_test(n_perms=1_000)

        # Summary scorecard
        print("\n" + "=" * 60)
        print("SCORECARD SUMMARY")
        print("=" * 60)
        if len(wfo):
            wfo_eff = wfo['oos_sharpe'].mean() / max(wfo['is_sharpe'].mean(), 1e-6)
            print(f"WFO Efficiency Ratio:    {wfo_eff:.3f}  "
                  f"{'PASS' if wfo_eff >= 0.6 else 'FAIL'}")
        print(f"MC Ruin Probability:     "
              f"{np.mean(np.array([mc['max_drawdown'][f'p{p}'] for p in [95]]) < -0.30):.1%}")
        if mae:
            print(f"MAE Recommended SL:      {mae['recommended_sl_ticks']} ticks")
        print(f"Permutation p (Sharpe):  {perm['sharpe_p_value']:.4f}  "
              f"{'PASS' if perm['sharpe_p_value'] < 0.05 else 'FAIL'}")

        return {'wfo': wfo, 'mc': mc, 'mae': mae, 'permutation': perm}


# =============================================================================
# MAIN
# =============================================================================
# =============================================================================
# MAIN - CLI entry point
# =============================================================================
if __name__ == '__main__':
    import argparse, sys, json

    parser = argparse.ArgumentParser(
        description='Titan HFT Validation Suite V2',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--trade-log', default=None,
        help='Path to trade log CSV (columns: date, profit, [mae])')
    parser.add_argument(
        '--capital', type=float, default=100_000,
        help='Initial capital for return normalisation (default: 100000)')
    parser.add_argument(
        '--wfo-splits', type=int, default=8,
        help='Number of WFO folds (default: 8)')
    parser.add_argument(
        '--mc-iter', type=int, default=10_000,
        help='Monte Carlo iterations (default: 10000)')
    parser.add_argument(
        '--sl-min', type=int, default=3,
        help='MAE SL search minimum ticks (default: 3)')
    parser.add_argument(
        '--sl-max', type=int, default=30,
        help='MAE SL search maximum ticks (default: 30)')
    parser.add_argument(
        '--target-wfo', type=float, default=0.60,
        help='WFO Efficiency Ratio pass threshold (default: 0.6)')
    parser.add_argument(
        '--target-ruin', type=float, default=0.01,
        help='Probability of Ruin fail threshold (default: 0.01 = 1%%)')
    parser.add_argument(
        '--json-out', default=None,
        help='Optional: write scorecard JSON to this file path')
    parser.add_argument(
        '--demo', action='store_true',
        help='Generate and use a synthetic demo trade log')
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"  [INFO] Ignoring unrecognized arguments: {unknown}")

    # ── Demo mode: generate synthetic log ─────────────────────────────────────
    trade_log_path = args.trade_log
    
    # Auto-detect real tester_log.csv from MT5 backtest if no log provided
    if trade_log_path is None and not args.demo:
        if os.path.exists("tester_log.csv"):
            trade_log_path = "tester_log.csv"
            print(f"[INFO] Found real MT5 backtest log -> {trade_log_path}")

    if args.demo or trade_log_path is None:
        np.random.seed(42)
        n_trades = 1000
        dates    = pd.date_range('2023-01-02', periods=n_trades, freq='4h')
        profits  = np.random.normal(8, 45, n_trades)
        loss_mask = np.random.random(n_trades) < 0.42
        profits[loss_mask] = -np.abs(np.random.exponential(28, loss_mask.sum()))
        mae_win  = np.random.exponential(6,  n_trades)
        mae_los  = np.random.exponential(18, n_trades)
        mae_sim  = np.where(profits > 0, mae_win, mae_los)
        demo_log = pd.DataFrame({
            'date':    dates,
            'profit':  profits,
            'mae':     mae_sim,
            'balance': 100_000 + profits.cumsum(),
        })
        trade_log_path = 'titan_trade_log_demo.csv'
        demo_log.to_csv(trade_log_path, index=False)
        print(f"[demo] Synthetic trade log -> {trade_log_path}")

    # ── Run full validation suite ──────────────────────────────────────────────
    rm      = TitanRiskManager(initial_capital=args.capital)
    results = rm.run_full_suite(trade_log_path)

    # ── Build machine-readable scorecard ──────────────────────────────────────
    wfo     = results.get('wfo', pd.DataFrame())
    mc      = results.get('mc', {})
    mae_res = results.get('mae', {})
    perm    = results.get('permutation', {})

    wfo_eff  = 0.0
    if len(wfo) > 0:
        wfo_eff = wfo['oos_sharpe'].mean() / max(wfo['is_sharpe'].mean(), 1e-6)

    # Probability of ruin = fraction of MC paths with max_dd < -30%
    ruin_prob = 0.0
    if mc and 'prob_ruin' in mc:
        ruin_prob = float(mc.get('prob_ruin', 0.0))

    scorecard = {
        'wfo_efficiency_ratio':   round(wfo_eff, 4),
        'wfo_pass':               wfo_eff >= args.target_wfo,
        'ruin_probability':       round(ruin_prob, 4),
        'ruin_pass':              ruin_prob <= args.target_ruin,
        'perm_sharpe_pvalue':     round(perm.get('sharpe_p_value', 1.0), 4),
        'perm_pass':              perm.get('sharpe_p_value', 1.0) < 0.05,
        'mae_recommended_sl':     mae_res.get('recommended_sl_ticks', None),
        'mae_winner_coverage':    round(mae_res.get('winner_coverage_pct', 0.0), 3),
        'production_ready':       (
            wfo_eff >= args.target_wfo and
            ruin_prob <= args.target_ruin and
            perm.get('sharpe_p_value', 1.0) < 0.05
        ),
    }

    def _cast_primitive(v):
        if hasattr(v, 'item'):
            v = v.item()
        if isinstance(v, (np.bool_, bool)): return bool(v)
        if isinstance(v, (np.floating, float)): return float(v)
        if isinstance(v, (np.integer, int)):  return int(v)
        if hasattr(v, 'item'):            return v.item()
        return v

    scorecard = {k: _cast_primitive(v) for k, v in scorecard.items()}

    print("\n" + "=" * 60)
    print("FORENSIC SCORECARD (JSON)")
    print("=" * 60)
    print(json.dumps(scorecard, indent=2))

    if args.json_out:
        with open(args.json_out, 'w') as fh:
            json.dump(scorecard, fh, indent=2)
        print(f"Scorecard -> {args.json_out}")

    # Exit code 0 = production ready; 1 = needs more iteration
    sys.exit(0 if scorecard['production_ready'] else 1)
