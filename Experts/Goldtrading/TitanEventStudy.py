#!/usr/bin/env python3
"""
TITAN HFT SYSTEM - Event Study V2.0 (PRODUCTION)
Author : Senior Quant / Systems Architect | April 2026

FORENSIC FIXES vs V1:
  [BUG-01] CRITICAL: Used TICK_FLAG constants 0x01 and 0x02 (bit 0 and bit 1).
           MQL5 correct values: BUY=32 (0x20), SELL=64 (0x40).
           All buy/sell volume aggregation was silently returning zeros.
  [BUG-02] print(".3f") - literal strings, not formatted f-strings.
           All print statements corrected.
  [BUG-03] detect_stacked_imbalances_in_zones: flags was the whole column
           series, not per-row values. int() on a pandas Series raises TypeError.
           Fixed with vectorised boolean masks.
  [BUG-04] measure_reversal_probability: data.loc[event_time:] may return an
           empty slice if the event_time index is not exact. Fixed with
           searchsorted for time-based lookup.
  [BUG-05] compute_vwap_and_sigmas: used static per-session VWAP (just std of
           prices, not volume-weighted sigma). Fixed to volumetric VWAP std.
  [NEW]    Full matplotlib Agg backend for headless server operation.
  [NEW]    Bootstrap confidence intervals on reversal probability.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats
warnings.filterwarnings('ignore')

# Force UTF-8 on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ── TICK FLAG CONSTANTS (match MQL5 documentation exactly) ───────────────────
TICK_FLAG_BID    = 2    # Bid price changed
TICK_FLAG_ASK    = 4    # Ask price changed
TICK_FLAG_LAST   = 8    # Last trade price changed (REAL TRADE TICK)
TICK_FLAG_VOLUME = 16   # Volume changed
TICK_FLAG_BUY    = 32   # Buy aggressor  - BUG-FIX: V1 had 0x01 (wrong!)
TICK_FLAG_SELL   = 64   # Sell aggressor - BUG-FIX: V1 had 0x02 (wrong!)


class TitanEventStudy:
    """
    Event study framework for validating scalping hypotheses.
    Measures 5-tick reversal probability in VWAP premium/discount zones.
    """

    def __init__(self):
        self.data    = None
        self.results = {}

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    def load_parquet(self, path: str) -> pd.DataFrame:
        """Load tick data from Parquet and add derived columns."""
        print(f"Loading {path}...")
        df = pd.read_parquet(path)

        # Normalise index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.set_index('timestamp')
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)

        df = df.sort_index()
        df['mid']      = (df['bid'] + df['ask']) * 0.5
        df['spread']   = df['ask'] - df['bid']

        # ─── TITAN V2.1 PRECISION BRIDGE: HYBRID QUOTE-TICK LOGIC ──────────────────
        df['is_trade']    = (df['flags'].astype(np.int32) & TICK_FLAG_LAST)  != 0
        df['is_buy']      = (df['flags'].astype(np.int32) & TICK_FLAG_BUY)   != 0
        df['is_sell']     = (df['flags'].astype(np.int32) & TICK_FLAG_SELL)  != 0

        # Tick-Test Fallback when broker flags are missing
        if not df['is_buy'].any() and not df['is_sell'].any():
            price = pd.Series(np.where(df['last'] > 0, df['last'], df['mid']), index=df.index)
            # Forward difference to know the direction of trade
            price_diff = price.diff()
            df['is_buy'] = price_diff > 0
            df['is_sell'] = price_diff < 0
            
            # Forward fill zero diffs to persist direction
            synth_dir = pd.Series(0, index=df.index)
            synth_dir.loc[df['is_buy']] = 1
            synth_dir.loc[df['is_sell']] = -1
            synth_dir = synth_dir.replace(0, np.nan).ffill().fillna(0)
            
            df['is_buy'] = synth_dir == 1
            df['is_sell'] = synth_dir == -1
            df['is_trade'] = True
        # ────────────────────────────────────────────────────────────────────────

        self.data = df
        trade_n = df['is_trade'].sum()
        print(f"Loaded {len(df):,} ticks  "
              f"Trades: {trade_n:,}  "
              f"Buys: {df['is_buy'].sum():,}  "
              f"Sells: {df['is_sell'].sum():,}")
        return df

    # =========================================================================
    # SESSION VWAP + SIGMA BANDS
    # BUG-FIX: V1 used price std, not volume-weighted sigma.
    # =========================================================================
    def compute_vwap_and_sigmas(self) -> pd.DataFrame:
        """
        Compute intraday session VWAP and σ bands (volume-weighted).
        Resets at midnight UTC each day.
        """
        df = self.data
        df['date'] = df.index.date

        pv_cumsum = {}; v_cumsum = {}; ss_cumsum = {}
        vwap_arr    = np.zeros(len(df))
        vwap_sd_arr = np.full(len(df), 1e-8)

        for i, (ts, row) in enumerate(df.iterrows()):
            d = row['date']
            if d not in pv_cumsum:
                pv_cumsum[d] = 0.0; v_cumsum[d] = 0.0; ss_cumsum[d] = 0.0

            if row['is_trade'] and row['volume'] > 0:
                p = row['last'] if row.get('last', 0) > 0 else row['mid']
                v = row['volume']
                pv_cumsum[d]  += p * v
                v_cumsum[d]   += v
                ss_cumsum[d]  += p * p * v

            if v_cumsum[d] > 0:
                vwap = pv_cumsum[d] / v_cumsum[d]
                var  = ss_cumsum[d] / v_cumsum[d] - vwap ** 2
                std  = max(np.sqrt(max(var, 0.0)), 1e-8)
                vwap_arr[i]    = vwap
                vwap_sd_arr[i] = std
            else:
                vwap_arr[i]    = row['mid']
                vwap_sd_arr[i] = 1e-8

        df['vwap']        = vwap_arr
        df['vwap_sd']     = vwap_sd_arr
        df['price_vs_vwap_z'] = (df['mid'] - df['vwap']) / df['vwap_sd']
        df.drop(columns=['date'], inplace=True, errors='ignore')

        print(f"VWAP computed  mean_z={df['price_vs_vwap_z'].mean():.3f}  "
              f"std_z={df['price_vs_vwap_z'].std():.3f}")
        return df

    # =========================================================================
    # ZONE IDENTIFICATION
    # =========================================================================
    def identify_zones(self, sigma_threshold: float = 1.5) -> tuple:
        """Premium > VWAP + σ·thresh    Discount < VWAP - σ·thresh"""
        df = self.data
        if 'price_vs_vwap_z' not in df.columns:
            self.compute_vwap_and_sigmas()

        premium  = df[df['price_vs_vwap_z'] >  sigma_threshold].copy()
        discount = df[df['price_vs_vwap_z'] < -sigma_threshold].copy()

        print(f"Premium zone:  {len(premium):,} ticks  "
              f"Discount zone: {len(discount):,} ticks")
        return premium, discount

    # =========================================================================
    # STACKED IMBALANCE DETECTION
    # BUG-FIX: V1 used 0x01/0x02 flags (wrong). V2 uses is_buy/is_sell.
    # BUG-FIX: V1 called int() on a full pandas Series - raised TypeError.
    # =========================================================================
    def detect_stacked_imbalances(self,
                                   zone_df: pd.DataFrame,
                                   window: str = '100ms',
                                   ratio_thresh: float = 3.0,
                                   stack_min: int = 3) -> pd.DataFrame:
        """Detect stacked imbalances using buy/sell volume in zone ticks."""
        if len(zone_df) < stack_min * 2:
            return pd.DataFrame()

        # Vectorised aggregation (no per-row int() conversion)
        buy_vol  = zone_df[zone_df['is_buy']].resample(window)['volume'].sum()
        sell_vol = zone_df[zone_df['is_sell']].resample(window)['volume'].sum()

        agg = pd.DataFrame({'buy': buy_vol, 'sell': sell_vol}).fillna(0.0)
        agg = agg[agg['buy'] + agg['sell'] > 0]

        eps = 1e-8
        agg['ratio'] = np.where(
            agg['buy'] >= agg['sell'],
             agg['buy']  / (agg['sell'] + eps),
            -agg['sell'] / (agg['buy']  + eps))

        # Count consecutive stacked windows (same sign, |ratio| >= threshold)
        events = []
        count, sign = 0, 0
        for ts, row in agg.iterrows():
            v = row['ratio']
            cur_sign = int(np.sign(v))
            if abs(v) >= ratio_thresh and cur_sign == sign:
                count += 1
            elif abs(v) >= ratio_thresh:
                count = 1; sign = cur_sign
            else:
                count = 0; sign = 0

            if count >= stack_min:
                events.append({
                    'timestamp': ts,
                    'direction': 'buy' if sign > 0 else 'sell',
                    'stack_count': count,
                    'ratio': abs(v),
                })

        return pd.DataFrame(events)

    # =========================================================================
    # REVERSAL PROBABILITY
    # BUG-FIX: data.loc[ts:] can be empty if ts not in index. Use searchsorted.
    # =========================================================================
    def measure_reversal_probability(self,
                                      events: pd.DataFrame,
                                      zone_type: str,
                                      ticks_ahead: int = 5,
                                      seconds_ahead: int = 30) -> pd.DataFrame:
        """Measure P(reversal) after each stacked-imbalance event."""
        df     = self.data
        ts_arr = df.index
        mid    = df['mid'].values
        results = []

        for _, ev in events.iterrows():
            t0  = ev['timestamp']
            direction = ev['direction']

            # BUG-FIX: searchsorted is O(log n), avoids empty-slice trap
            i0  = ts_arr.searchsorted(t0)
            i_end = min(i0 + 1000, len(df))  # Search window cap

            # Find end of time window
            t_limit = t0 + pd.Timedelta(seconds=seconds_ahead)
            i_lim   = ts_arr.searchsorted(t_limit)
            i_lim   = min(i_lim, len(df) - 1)

            if i0 + ticks_ahead >= i_lim:
                continue

            entry_price = mid[i0]
            future_mid  = mid[i0+1 : i0+ticks_ahead+1]
            if len(future_mid) == 0:
                continue

            price_change = future_mid[-1] - entry_price

            # Reversal: price goes against the imbalance direction
            if direction == 'buy':
                reversal = price_change < 0   # Buy imbalance -> expect down
            else:
                reversal = price_change > 0   # Sell imbalance -> expect up

            time_to_rev = (ts_arr[i0 + ticks_ahead] - t0).total_seconds()
            results.append({
                'event_time':       t0,
                'zone_type':        zone_type,
                'direction':        direction,
                'reversal':         reversal and (time_to_rev <= seconds_ahead),
                'price_change_pips': price_change * 10_000,
                'time_to_reversal': time_to_rev,
                'stack_count':      ev['stack_count'],
            })

        return pd.DataFrame(results)

    # =========================================================================
    # BOOTSTRAP CONFIDENCE INTERVAL
    # =========================================================================
    @staticmethod
    def bootstrap_ci(arr: np.ndarray, n_boot: int = 2000,
                     ci: float = 0.95) -> tuple:
        """Bootstrap CI for the mean of arr."""
        means = [np.random.choice(arr, len(arr), replace=True).mean()
                 for _ in range(n_boot)]
        lo = np.percentile(means, (1 - ci) / 2 * 100)
        hi = np.percentile(means, (1 + ci) / 2 * 100)
        return lo, hi

    # =========================================================================
    # FULL STUDY PIPELINE
    # =========================================================================
    def run_full_study(self, parquet_file: str,
                        sigma_thresh: float = 1.5,
                        ticks_ahead: int = 5,
                        seconds_ahead: int = 30,
                        output_plot: str = 'titan_event_study_v2.png') -> dict:
        """Run the complete event study pipeline and return results dict."""
        print("=" * 60)
        print("TITAN EVENT STUDY V2 - 5-Tick Reversal Probability")
        print("=" * 60)

        self.load_parquet(parquet_file)
        self.compute_vwap_and_sigmas()
        premium, discount = self.identify_zones(sigma_thresh)

        # Detect stacked imbalances in each zone
        prem_ev = self.detect_stacked_imbalances(premium,  ratio_thresh=3.0, stack_min=3)
        disc_ev = self.detect_stacked_imbalances(discount, ratio_thresh=3.0, stack_min=3)

        print(f"\nStacked imbalances - Premium: {len(prem_ev)}  Discount: {len(disc_ev)}")

        if len(prem_ev) + len(disc_ev) == 0:
            print("No stacked imbalances detected. Try reducing ratio_thresh or sigma_thresh.")
            return {}

        # Measure reversals
        prem_r = self.measure_reversal_probability(prem_ev, 'premium', ticks_ahead, seconds_ahead)
        disc_r = self.measure_reversal_probability(disc_ev, 'discount', ticks_ahead, seconds_ahead)
        all_r  = pd.concat([prem_r, disc_r], ignore_index=True)

        if len(all_r) == 0:
            print("No reversal measurements possible.")
            return {}

        # Compute rates + bootstrap CIs
        overall_rate = all_r['reversal'].mean()
        prem_rate    = prem_r['reversal'].mean() if len(prem_r) else 0.0
        disc_rate    = disc_r['reversal'].mean() if len(disc_r) else 0.0

        ov_lo, ov_hi = self.bootstrap_ci(all_r['reversal'].values)

        # Chi-squared test vs 50% null
        n     = len(all_r)
        obs   = all_r['reversal'].sum()
        chi2, p_val = stats.chisquare([obs, n - obs], f_exp=[n * 0.5, n * 0.5])

        print(f"\n{'='*60}")
        print(f"RESULTS - n={n} events, ticks_ahead={ticks_ahead}, "
              f"secs_ahead={seconds_ahead}")
        print(f"{'='*60}")
        print(f"Overall reversal probability:  {overall_rate:.3f}  "
              f"  95% CI [{ov_lo:.3f}, {ov_hi:.3f}]")
        print(f"Premium zone reversal rate:    {prem_rate:.3f}  (n={len(prem_r)})")
        print(f"Discount zone reversal rate:   {disc_rate:.3f}  (n={len(disc_r)})")
        print(f"χ² vs 50% null:  χ²={chi2:.3f}  p={p_val:.4f}  "
              f"{'SIGNIFICANT [PASS]' if p_val < 0.05 else 'NOT SIGNIFICANT [FAIL]'}")

        self.results = {
            'n_events':          n,
            'overall_rate':      overall_rate,
            'premium_rate':      prem_rate,
            'discount_rate':     disc_rate,
            'ci_95':             (ov_lo, ov_hi),
            'p_value':           p_val,
            'significant':       p_val < 0.05,
            'results_df':        all_r,
        }

        self._plot(all_r, prem_rate, disc_rate, output_plot)
        return self.results

    # =========================================================================
    # PLOT
    # =========================================================================
    def _plot(self, all_r: pd.DataFrame,
               prem_rate: float, disc_rate: float,
               save_path: str):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Titan Event Study V2 - Reversal Probability', fontweight='bold')

        # Reversal rate by zone
        axes[0].bar(['Premium', 'Discount'], [prem_rate, disc_rate],
                    color=['#e74c3c', '#2ecc71'])
        axes[0].axhline(0.5, color='grey', linestyle='--', label='50% baseline')
        axes[0].set_ylim(0, 1); axes[0].set_ylabel('Reversal Probability')
        axes[0].set_title('Rate by Zone'); axes[0].legend()

        # Price change distribution
        axes[1].hist(all_r[all_r['reversal']]['price_change_pips'],
                     bins=30, alpha=0.6, color='#2ecc71', label='Reversal', density=True)
        axes[1].hist(all_r[~all_r['reversal']]['price_change_pips'],
                     bins=30, alpha=0.6, color='#e74c3c', label='No reversal', density=True)
        axes[1].set_xlabel('Price change (pips)'); axes[1].set_title('P&L Distribution')
        axes[1].legend()

        # Time to reversal
        rev_df = all_r[all_r['reversal']]
        if len(rev_df):
            axes[2].hist(rev_df['time_to_reversal'], bins=30, color='#3498db', density=True)
            axes[2].set_xlabel('Seconds'); axes[2].set_title('Time to Reversal')
        else:
            axes[2].text(0.5, 0.5, 'No reversals', ha='center', va='center')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Plot saved -> {save_path}")


# =============================================================================
# ENTRY POINT - CLI with --json-out for TitanOptimizer.py integration
# =============================================================================
def main():
    import glob
    import argparse
    import json
    import sys

    BASE = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE)

    parser = argparse.ArgumentParser(
        description='Titan Event Study V2 - 5-tick reversal analysis',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--parquet', default=None,
        help='Specific parquet file to analyse (default: first in data/)')
    parser.add_argument(
        '--sigma-thresh', type=float, default=1.5,
        help='VWAP sigma threshold for event detection (default: 1.5)')
    parser.add_argument(
        '--ticks-ahead', type=int, default=5,
        help='Ticks ahead to measure reversal (default: 5)')
    parser.add_argument(
        '--seconds-ahead', type=int, default=30,
        help='Time window for reversal in seconds (default: 30)')
    parser.add_argument(
        '--json-out', default=None,
        help='Write machine-readable results JSON to this path\n'
             '(read by TitanOptimizer.py for auto-recalibration)')
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"  [INFO] Ignoring unrecognized arguments: {unknown}")

    # ── Locate parquet file ───────────────────────────────────────────────────
    if args.parquet and os.path.exists(args.parquet):
        target = args.parquet
    else:
        parquets = sorted(glob.glob(os.path.join('data', '*.parquet')))
        if not parquets:
            print("No parquet files in data/. Run TitanParquetConverter.py first.")
            if args.json_out:
                json.dump({'error': 'no_parquet_files',
                           'reversal_probability': None,
                           'reversal_p_value': 1.0,
                           'significant': False},
                          open(args.json_out, 'w'), indent=2)
            sys.exit(1)
        target = parquets[0]
    print(f"Analysing: {target}")

    # ── Run study ─────────────────────────────────────────────────────────────
    study   = TitanEventStudy()
    results = study.run_full_study(
        parquet_file   = target,
        sigma_thresh   = args.sigma_thresh,
        ticks_ahead    = args.ticks_ahead,
        seconds_ahead  = args.seconds_ahead,
        output_plot    = 'titan_event_study_v2.png',
    )

    # ── Print human-readable summary ──────────────────────────────────────────
    if results:
        rev_prob  = results.get('overall_rate', 0.0)
        rev_pval  = results.get('p_value',     1.0)
        sig       = results.get('significant', False)
        n_events  = results.get('n_events',    0)

        print(f"\n{'='*60}")
        print(f"EVENT STUDY RESULTS")
        print(f"{'='*60}")
        print(f"  Events detected          : {n_events:,}")
        print(f"  5-tick reversal prob     : {rev_prob:.4f}  "
              f"({'[PASS] PASS >=0.60' if rev_prob >= 0.60 else '[FAIL] FAIL <0.60'})")
        print(f"  p-value                  : {rev_pval:.4f}  "
              f"({'[PASS] PASS <=0.01' if rev_pval <= 0.01 else '[FAIL] FAIL >0.01'})")
        print(f"  Statistically significant: {'[PASS] Yes' if sig else '[FAIL] No'}")

        if rev_prob < 0.60 or rev_pval > 0.01:
            print(f"\n  [RECOMMENDATION] Recalibrate lookback windows:")
            print(f"    hurst_window  : increase (e.g. 20 -> 30)")
            print(f"    vot_window_ms : increase (e.g. 100 -> 200ms)")
            print(f"    sigma_thresh  : tighten (current: {args.sigma_thresh} -> try 2.0)")

        # ── Write JSON for TitanOptimizer.py ──────────────────────────────────
        out_obj = {
            'reversal_probability': round(rev_prob, 6),
            'reversal_p_value':     round(rev_pval, 6),
            'significant':          sig,
            'n_events':             n_events,
            'sigma_thresh_used':    args.sigma_thresh,
            'ticks_ahead':          args.ticks_ahead,
            'pass_prob':            rev_prob >= 0.60,
            'pass_pval':            rev_pval <= 0.01,
            'production_ready':     rev_prob >= 0.60 and rev_pval <= 0.01,
        }
        if args.json_out:
            with open(args.json_out, 'w') as fh:
                json.dump(out_obj, fh, indent=2)
            print(f"\n  Scorecard -> {args.json_out}")

        # Exit 0 = passes benchmarks; 1 = needs recalibration
        sys.exit(0 if out_obj['production_ready'] else 1)
    else:
        print("Event study returned no results - insufficient data in parquet file.")
        if args.json_out:
            json.dump({'error': 'insufficient_data',
                       'reversal_probability': None,
                       'reversal_p_value': 1.0,
                       'significant': False},
                      open(args.json_out, 'w'), indent=2)
        sys.exit(1)


if __name__ == '__main__':
    main()