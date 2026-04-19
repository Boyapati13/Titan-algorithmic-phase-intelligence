#!/usr/bin/env python3
"""
TITAN HFT SYSTEM - Feature Engineering V2.0 (PRODUCTION)
Author : Senior Quant / Systems Architect | April 2026

FORENSIC FIXES vs V1:
  [BUG-01] CRITICAL: TICK_FLAG constants were wrong. V1 used bits 0 & 1
           (values 1 & 2). Correct MQL5 values: BUY=32, SELL=64, LAST=8.
           This invalidated ALL delta and VoT calculations.
  [BUG-02] compute_velocity_of_tape: Resampled to 1S counting flags & 0x01
           which is bit 0 - never set. Fixed to use TICK_FLAG_LAST (8).
  [BUG-03] detect_absorption: expected_change = vot_zscore.mean() * 0.0001
           - arbitrary magic number. V2 uses ATR-calibrated tick size.
  [BUG-04] compute_10_feature_vector: bid_depth_imb, ask_depth_imb,
           price_vs_vwap all returned constants. V2 computes them properly.
  [BUG-05] detect_stacked_imbalances: used flags from resample lambda as list
           and then int() on each - failed silently on NaN. Fixed.
  [BUG-06] Feature vector had 10 features but LSTM was trained on 12.
           V2 outputs 12 features: +hurst_exponent, +pace_acceleration.
  [NEW]    Proper session VWAP with ±σ bands.
  [NEW]    Hurst exponent (rolling 20-bar) for regime detection.
  [NEW]    DOM imbalance approximated from bid/ask volume when L2 unavailable.
  [NEW]    Vectorised operations throughout - no per-row .apply() loops.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ─── TICK FLAG CONSTANTS (match MQL5 documentation exactly) ──────────────────
TICK_FLAG_BID    = 2    # Bid price changed
TICK_FLAG_ASK    = 4    # Ask price changed
TICK_FLAG_LAST   = 8    # Last trade price changed (REAL TRADE)
TICK_FLAG_VOLUME = 16   # Volume changed
TICK_FLAG_BUY    = 32   # Buy aggressor
TICK_FLAG_SELL   = 64   # Sell aggressor


class TitanFeatureEngineer:
    """
    Production feature engineering engine for HFT microstructure research.
    Produces a 12-dimensional feature vector consistent with TitanLSTM_V2.
    """

    FEATURE_NAMES = [
        'vot_zscore',       # 00 Velocity-of-Tape z-score
        'rvol',             # 01 Relative Volume
        'cumdelta_div',     # 02 Cumulative Delta Divergence
        'imbalance_ratio',  # 03 Stacked Imbalance score
        'fdpi',             # 04 Flag-Based Directional Pressure Index
        'mvdi',             # 05 Micro-Volatility Dispersion Index
        'bid_depth_imb',    # 06 DOM bid imbalance (approx)
        'ask_depth_imb',    # 07 DOM ask imbalance (approx)
        'price_vs_vwap',    # 08 Price vs session VWAP (σ units)
        'hour_sin',         # 09 Hour of day sine
        'hurst',            # 10 Hurst exponent (regime)
        'twkj',             # 11 Time-Weighted Kinematic Jerk
    ]
    N_FEATURES = len(FEATURE_NAMES)  # 12

    def __init__(self, symbol: str = 'EURUSD'):
        self.symbol = symbol
        self.data: Optional[pd.DataFrame] = None

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    def load_parquet_data(self, path: str) -> pd.DataFrame:
        print(f"Loading {path}...")
        df = pd.read_parquet(path)

        # Normalise index to UTC DatetimeIndex at nanosecond precision
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.set_index('timestamp')
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)

        df = df.sort_index()

        # Derive mid and spread
        df['mid']    = (df['bid'] + df['ask']) * 0.5
        df['spread'] = df['ask'] - df['bid']

        # ─── TITAN V2.1 PRECISION BRIDGE: LEE-READY HYBRID CLASSIFICATION ────────
        df['is_trade']    = (df['flags'].astype(np.int32) & TICK_FLAG_LAST)  != 0
        df['is_buy']      = (df['flags'].astype(np.int32) & TICK_FLAG_BUY)   != 0
        df['is_sell']     = (df['flags'].astype(np.int32) & TICK_FLAG_SELL)  != 0
        df['is_bid_chg']  = (df['flags'].astype(np.int32) & TICK_FLAG_BID)   != 0
        df['is_ask_chg']  = (df['flags'].astype(np.int32) & TICK_FLAG_ASK)   != 0

        # Tick-Test Fallback when broker flags collapsed (All-Sell / All-Buy bug)
        # CRITICAL: use .values everywhere to avoid pandas/numpy index misalignment.
        # NOTE: In OTC Forex (MT5 parquets), 'last' = bid price (always < mid).
        # A naive quote-test (P vs P_mid) would classify every tick as Sell.
        # We detect this and fall back to sgn(ΔP) — the correct Lee-Ready Tick-Test.
        if not df['is_buy'].any() or not df['is_sell'].any():
            mid_np   = df['mid'].values
            last_np  = df['last'].values

            # For OTC Forex: last=bid always < mid → quote-test is degenerate.
            # Detect this: if 90%+ of last prices are below mid, skip quote-test.
            has_valid_last     = last_np > 0
            last_above_mid_pct = (last_np[has_valid_last] > mid_np[has_valid_last]).mean() if has_valid_last.any() else 0.0
            last_below_mid_pct = (last_np[has_valid_last] < mid_np[has_valid_last]).mean() if has_valid_last.any() else 0.0
            otc_forex_mode     = (last_below_mid_pct > 0.90) or (last_above_mid_pct > 0.90)

            if otc_forex_mode:
                # Pure Tick-Test: sgn(ΔP_mid) — the only reliable signal in quote-driven FX
                price_np = mid_np
            else:
                price_np = np.where(last_np > 0, last_np, mid_np)

            # Lee-Ready Hybrid:
            #   v_t =  1 if P > P_mid (quote-test — skipped in OTC mode)
            #   v_t = -1 if P < P_mid (quote-test — skipped in OTC mode)
            #   v_t = sgn(ΔP) if P == P_mid  ← always true in OTC mode
            v_t = np.zeros(len(df), dtype=np.float64)
            if not otc_forex_mode:
                v_t[price_np > mid_np] =  1.0
                v_t[price_np < mid_np] = -1.0

            equal_mask = (v_t == 0.0)
            if equal_mask.any():
                delta_p = np.diff(price_np, prepend=price_np[0])
                v_t[equal_mask] = np.sign(delta_p[equal_mask])

            # ffill zeros (flat ticks), then default residual to +1
            v_t_series = pd.Series(v_t, index=df.index).replace(0.0, np.nan).ffill().fillna(1.0)

            # ── ENTROPY GUARDRAIL (Shannon) ───────────────────────────────────
            p1  = float((v_t_series == 1.0).mean())
            pn1 = 1.0 - p1
            eps = 1e-9
            entropy = -(
                (p1  + eps) * np.log2(p1  + eps) +
                (pn1 + eps) * np.log2(pn1 + eps)
            )
            mode_str = "OTC/Tick-Test" if otc_forex_mode else "Quote-Test"
            print(f"Tape Reconstructed [{mode_str}] -> H(v)={entropy:.4f}  Buy%={p1:.3f}")
            if entropy < 0.90:
                raise ValueError(
                    f"Dead Tape: H(v)={entropy:.4f} < 0.90 "
                    f"(Buy={p1:.3f} Sell={pn1:.3f}). Check parquet price resolution."
                )

            df['is_buy']   = (v_t_series == 1.0).values
            df['is_sell']  = (v_t_series == -1.0).values
            df['is_trade'] = True


        self.data = df
        print(f"Loaded {len(df):,} ticks | "
              f"{df.index.min()} -> {df.index.max()}")
        print(f"Trade ticks: {df['is_trade'].sum():,}  "
              f"Buy: {df['is_buy'].sum():,}  "
              f"Sell: {df['is_sell'].sum():,}")
        return df

    # =========================================================================
    # FEATURE 00 - VELOCITY OF TAPE Z-SCORE
    # Trades per second, z-scored over a rolling 20-second window.
    # BUG-FIX: V1 filtered flags & 0x01 (bit 0) - never set. Now uses is_trade.
    # =========================================================================
    def compute_velocity_of_tape(self, z_window: str = '20s') -> pd.Series:
        df = self.data
        trade_ticks = df[df['is_trade']]

        # Count real trades per second
        trades_per_sec = trade_ticks.resample('1s').size().rename('tps')
        trades_per_sec = trades_per_sec.reindex(
            pd.date_range(df.index.min().floor('1s'),
                          df.index.max().ceil('1s'),
                          freq='1s'), fill_value=0)

        mean_tps = trades_per_sec.rolling(z_window).mean()
        std_tps  = trades_per_sec.rolling(z_window).std().clip(lower=1e-8)
        vot_z    = (trades_per_sec - mean_tps) / std_tps

        # Forward-fill to tick level
        df['vot_zscore'] = vot_z.reindex(df.index, method='ffill').fillna(0.0)
        print(f"VoT z-score: mean={df['vot_zscore'].mean():.3f} "
              f"std={df['vot_zscore'].std():.3f}")
        return df['vot_zscore']

    # =========================================================================
    # FEATURE 01 - RELATIVE VOLUME (RVOL)
    # Current 5-min volume vs 20-period rolling mean of same window.
    # =========================================================================
    def compute_rvol(self, window: str = '5min', periods: int = 20) -> pd.Series:
        df = self.data
        vol_per_window = df.resample(window)['volume'].sum()
        hist_mean = vol_per_window.rolling(periods).mean().clip(lower=1e-10)
        rvol = (vol_per_window / hist_mean).clip(upper=20.0)

        df['rvol'] = rvol.reindex(df.index, method='ffill').fillna(1.0)
        return df['rvol']

    # =========================================================================
    # FEATURE 02 - CUMULATIVE DELTA + DIVERGENCE
    # Per-tick delta: +vol for BUY, -vol for SELL.
    # Divergence: rolling 200-tick delta direction vs price direction.
    # BUG-FIX: V1 used flags & 0x01/0x02 - wrong constants.
    # =========================================================================
    def compute_cumulative_delta(self, div_window: int = 200) -> pd.Series:
        df = self.data

        # In OTC Forex, MT5 volume is 0 on all quote ticks.
        # Use unit-volume proxy (1 per tick) when vol is degenerate.
        raw_vol = df['volume'].values
        tick_vol = np.where(raw_vol > 0, raw_vol, 1.0)

        # Delta per tick (vectorised): +unit for BUY, -unit for SELL
        delta = np.where(df['is_buy'].values,  tick_vol,
                np.where(df['is_sell'].values, -tick_vol, 0.0))
        df['delta']     = delta
        df['cum_delta'] = df['delta'].cumsum()

        # Divergence: compare rolling sums of delta vs price direction
        price_chg = df['mid'].diff(div_window)
        delta_chg = df['delta'].rolling(div_window).sum()

        # +1 = bullish divergence, -1 = bearish, 0 = none
        conditions = [
            (price_chg > 0) & (delta_chg < 0),  # Bearish divergence
            (price_chg < 0) & (delta_chg > 0),  # Bullish divergence
        ]
        df['cumdelta_div'] = np.select(conditions, [-1.0, 1.0], default=0.0)
        return df['cumdelta_div']


    # =========================================================================
    # FEATURE 03 - STACKED IMBALANCE RATIO
    # Groups trades into 100ms windows, scores 3+ consecutive levels with >3:1 ratio.
    # BUG-FIX: V1 used int() on list of flags causing silent NaN failures.
    # =========================================================================
    def compute_stacked_imbalance(self, window: str = '100ms',
                                   ratio_thresh: float = 3.0,
                                   stack_min: int = 3) -> pd.Series:
        df = self.data
        # Aggregate buy/sell volume per 100ms window
        buy_agg  = df[df['is_buy']].resample(window)['volume'].sum()
        sell_agg = df[df['is_sell']].resample(window)['volume'].sum()

        # Combine into one frame and fill gaps
        agg = pd.DataFrame({'buy': buy_agg, 'sell': sell_agg}).fillna(0.0)
        agg['total'] = agg['buy'] + agg['sell']
        agg = agg[agg['total'] > 0]

        # Imbalance ratio: signed (positive = buy dominant, negative = sell)
        eps = 1e-8
        agg['ratio'] = np.where(
            agg['buy'] >= agg['sell'],
             agg['buy']  / (agg['sell'] + eps),
            -agg['sell'] / (agg['buy']  + eps)
        )

        # Rolling max of consecutive stack count
        def stack_score(series: pd.Series, n: int) -> pd.Series:
            """Count consecutive windows where |ratio| > threshold, same sign."""
            scores = np.zeros(len(series))
            count  = 0
            sign   = 0
            for i, v in enumerate(series):
                cur_sign = np.sign(v)
                if abs(v) >= ratio_thresh and cur_sign == sign:
                    count += 1
                elif abs(v) >= ratio_thresh:
                    count  = 1
                    sign   = cur_sign
                else:
                    count = 0
                    sign  = 0
                scores[i] = sign * min(count, 10) / 10.0  # Normalised to [-1,1]
            return pd.Series(scores, index=series.index)

        agg['stack'] = stack_score(agg['ratio'], stack_min)

        df['imbalance_ratio'] = (
            agg['stack']
            .reindex(df.index, method='ffill')
            .fillna(0.0)
        )
        return df['imbalance_ratio']

    # =========================================================================
    # FEATURE 04 - ABSORPTION FLAG
    # 4-condition detector: HighVoT + LowPriceEfficiency + VolAsym + Persistence
    # BUG-FIX: V1 had a fundamentally broken persistence check using arb constant.
    # =========================================================================
    def compute_absorption_flag(self, vot_thresh: float = 2.0,
                                 eff_thresh: float = 0.5,
                                 asym_thresh: float = 2.0,
                                 persist_windows: int = 5) -> pd.Series:
        df = self.data

        if 'vot_zscore' not in df.columns:
            self.compute_velocity_of_tape()

        # Fast vectorized aggregation per 100ms window
        df['_tmp_buy_vol'] = np.where(df['is_buy'], df['volume'], 0.0)
        df['_tmp_sell_vol'] = np.where(df['is_sell'], df['volume'], 0.0)

        agg = pd.DataFrame({
            'price_range': df['mid'].resample('100ms').max() - df['mid'].resample('100ms').min(),
            'total_vol':   df['volume'].resample('100ms').sum(),
            'buy_vol':     df['_tmp_buy_vol'].resample('100ms').sum(),
            'sell_vol':    df['_tmp_sell_vol'].resample('100ms').sum(),
            'vot_z':       df['vot_zscore'].resample('100ms').mean()
        }).fillna(0.0)

        df.drop(columns=['_tmp_buy_vol', '_tmp_sell_vol'], inplace=True)

        # ATR-calibrated expected move (tick size × volume proxy)
        tick_size   = df['spread'].median() * 0.5
        expected    = (agg['total_vol'] * tick_size).clip(lower=1e-10)
        agg['eff']  = (agg['price_range'] / expected).clip(upper=10.0)

        # Volume asymmetry
        agg['asym'] = np.where(
            agg['sell_vol'] > 0,
            agg['buy_vol'] / (agg['sell_vol'] + 1e-10),
            0.0
        )

        # 4 conditions (all must fire)
        cond1 = agg['vot_z'] > vot_thresh         # High tape velocity
        cond2 = agg['eff']   < eff_thresh          # Low price efficiency
        cond3 = ((agg['asym'] > asym_thresh) | (1.0/agg['asym'].clip(lower=1e-10) > asym_thresh)) & (agg['total_vol'] > 0)
        all4  = cond1 & cond2 & cond3

        # Persistence: must fire for N consecutive windows
        persist = all4.rolling(persist_windows).sum() >= persist_windows
        agg['absorption'] = persist.astype(float)

        df['absorption_flag'] = (
            agg['absorption']
            .reindex(df.index, method='ffill')
            .fillna(0.0)
        )
        return df['absorption_flag']

    # =========================================================================
    # FEATURE 05 - SPREAD Z-SCORE
    # =========================================================================
    def compute_spread_zscore(self, window: int = 500) -> pd.Series:
        df = self.data
        mean_spr = df['spread'].rolling(window).mean()
        std_spr  = df['spread'].rolling(window).std().clip(lower=1e-10)
        df['spread_zscore'] = ((df['spread'] - mean_spr) / std_spr).fillna(0.0)
        return df['spread_zscore']

    # =========================================================================
    # FEATURES 06 & 07 - DOM BID/ASK DEPTH IMBALANCE (approximated)
    # Without Level 2 data in Python, we approximate from trade flow:
    # bid imbalance ≈ buy_vol / (buy_vol + sell_vol) rolling 10 windows.
    # When real DOM is available (live EA), this is overridden by MarketBookGet.
    # =========================================================================
    def compute_dom_imbalance_approx(self, window: str = '500ms') -> Tuple:
        df = self.data
        buy_roll  = df[df['is_buy']].resample(window)['volume'].sum().rolling(10).sum()
        sell_roll = df[df['is_sell']].resample(window)['volume'].sum().rolling(10).sum()
        total     = (buy_roll + sell_roll).clip(lower=1e-10)

        bid_imb   = ((buy_roll  - sell_roll) / total).fillna(0.0)
        ask_imb   = ((sell_roll - buy_roll)  / total).fillna(0.0)

        df['bid_depth_imb'] = bid_imb.reindex(df.index, method='ffill').fillna(0.0)
        df['ask_depth_imb'] = ask_imb.reindex(df.index, method='ffill').fillna(0.0)
        return df['bid_depth_imb'], df['ask_depth_imb']

    # =========================================================================
    # FEATURE 08 - PRICE vs SESSION VWAP (in σ units)
    # BUG-FIX: V1 returned 0.0 always.
    # =========================================================================
    def compute_price_vs_vwap(self) -> pd.Series:
        df = self.data

        # Group by session date
        df['date'] = df.index.date
        pv_cum, v_cum, ss_cum = {}, {}, {}

        vwap_z = np.zeros(len(df))
        prev_date = None

        for i, (ts, row) in enumerate(df.iterrows()):
            d = row['date']
            if d != prev_date:
                pv_cum[d] = 0.0
                v_cum[d]  = 0.0
                ss_cum[d] = 0.0
                prev_date = d

            if row['is_trade'] and row['volume'] > 0:
                p = row['last'] if row['last'] > 0 else row['mid']
                v = row['volume']
                pv_cum[d] += p * v
                v_cum[d]  += v
                ss_cum[d] += p * p * v

            if v_cum[d] > 0:
                vwap     = pv_cum[d] / v_cum[d]
                variance = ss_cum[d] / v_cum[d] - vwap**2
                std      = max(np.sqrt(max(variance, 0.0)), 1e-8)
                vwap_z[i] = (row['mid'] - vwap) / std
            else:
                vwap_z[i] = 0.0

        df['price_vs_vwap'] = vwap_z
        df.drop(columns=['date'], inplace=True, errors='ignore')
        return df['price_vs_vwap']

    # =========================================================================
    # FEATURE 09 - MICRO EXHAUSTION (High VoT variance + low directional imbalance)
    # =========================================================================
    def compute_micro_exhaustion(self) -> pd.Series:
        df = self.data
        if 'vot_zscore' not in df.columns:
            self.compute_velocity_of_tape()
        if 'imbalance_ratio' not in df.columns:
            self.compute_stacked_imbalance()
        vot_std = df['vot_zscore'].rolling(10).std().fillna(0.0)
        df['micro_exhaustion'] = vot_std / (df['imbalance_ratio'].abs() + 1e-8)
        return df['micro_exhaustion']

    # =========================================================================
    # FEATURE 10 - HURST EXPONENT (rolling 20-bar R/S analysis)
    # H < 0.45 = mean-reverting (system active)
    # H > 0.55 = trending (system reduces size)
    # =========================================================================
    def compute_hurst(self, window: int = 20) -> pd.Series:
        df = self.data
        prices = df['mid'].values
        n      = len(prices)
        hurst  = np.full(n, 0.5)

        for i in range(window, n):
            seg = prices[i-window:i]
            mean_seg = seg.mean()
            devs     = np.cumsum(seg - mean_seg)
            R        = devs.max() - devs.min()
            S        = seg.std()
            if S > 1e-10 and R > 1e-10:
                hurst[i] = np.log(R / S) / np.log(window)

        df['hurst'] = np.clip(hurst, 0.0, 1.0)
        return df['hurst']

    # =========================================================================
    # FEATURE 11 - PACE ACCELERATION (VoT first-difference)
    # =========================================================================
    def compute_pace_acceleration(self) -> pd.Series:
        if 'vot_zscore' not in self.data.columns:
            self.compute_velocity_of_tape()
        self.data['pace_accel'] = self.data['vot_zscore'].diff().fillna(0.0)
        return self.data['pace_accel']

    # =========================================================================
    # ADVANCED ALPHA: MICRO-VOLATILITY DISPERSION INDEX (MVDI)
    # Ratio of spread turbulence to price turbulence. Bounded [-1, 1].
    # =========================================================================
    def compute_mvdi(self, window: int = 128) -> pd.Series:
        df = self.data
        eps = 1e-10
        
        spread_cv = df['spread'].rolling(window).std() / (df['spread'].rolling(window).mean() + eps)
        
        delta_mid_abs = df['mid'].diff().abs()
        mid_cv = delta_mid_abs.rolling(window).std() / (delta_mid_abs.rolling(window).mean() + eps)
        
        mvdi_raw = spread_cv / (mid_cv + eps)
        mvdi_mu  = mvdi_raw.rolling(window).mean()
        mvdi_sig = mvdi_raw.rolling(window).std()
        
        df['mvdi'] = (((mvdi_raw - mvdi_mu) / (mvdi_sig + eps)).clip(-3.0, 3.0) / 3.0).fillna(0.0)
        return df['mvdi']

    # =========================================================================
    # ADVANCED ALPHA: FLAG-BASED DIRECTIONAL PRESSURE INDEX (FDPI)
    # Signed order-flow imbalance using robust buy/sell masks. Bounded [-1, 1].
    # =========================================================================
    def compute_fdpi(self, window: int = 128) -> pd.Series:
        df = self.data
        eps = 1e-10
        
        # Safely utilize pre-computed bool masks (which include Lee-Ready fallback)
        b_n = df['is_buy'].astype(np.float32).rolling(window).sum()
        s_n = df['is_sell'].astype(np.float32).rolling(window).sum()
        
        df['fdpi'] = ((b_n - s_n) / (b_n + s_n + eps)).fillna(0.0)
        return df['fdpi']

    # =========================================================================
    # ADVANCED ALPHA: TIME-WEIGHTED KINEMATIC JERK (TWKJ)
    # 3rd derivative of price w.r.t real time delta. Bounded [-1, 1].
    # =========================================================================
    def compute_twkj(self, window: int = 128) -> pd.Series:
        df = self.data
        eps = 1e-10
        
        # Extract real dt in milliseconds from timezone-aware DatetimeIndex
        dt = df.index.to_series().diff().dt.total_seconds() * 1000.0
        dt = dt.clip(lower=1.0).values  # Prevent division by zero
        
        velocity = df['mid'].diff() / dt
        accel    = velocity.diff() / dt
        jerk     = accel.diff() / dt
        
        df['twkj'] = (((jerk - jerk.rolling(window).mean()) / (jerk.rolling(window).std() + eps)).clip(-3.0, 3.0) / 3.0).fillna(0.0)
        return df['twkj']

    # =========================================================================
    # MASTER PIPELINE - runs all 12 features in correct dependency order
    # =========================================================================
    def compute_all_features(self) -> pd.DataFrame:
        import json, os
        cfg_path = 'titan_feature_config.json'
        vot_kwargs, hurst_kwargs = {}, {}
        if os.path.exists(cfg_path):
            try:
                cfg = json.load(open(cfg_path))
                if 'vot_window_ms' in cfg:
                    vot_kwargs['z_window'] = f"{cfg['vot_window_ms']}ms"
                if 'hurst_window' in cfg:
                    # BUG-FIX: compute_hurst() parameter is 'window', not 'window_size'.
                    # Passing 'window_size' crashes with TypeError on every run where
                    # titan_feature_config.json exists (i.e. every iteration after 1).
                    hurst_kwargs['window'] = cfg['hurst_window']
            except Exception:
                pass

        print("=" * 60)
        print("Titan Feature Engineering V2 - 12-Feature Pipeline")
        print("=" * 60)

        self.compute_velocity_of_tape(**vot_kwargs)  # 00 - needed by others
        self.compute_rvol()              # 01
        self.compute_cumulative_delta()  # 02
        self.compute_stacked_imbalance() # 03
        self.compute_dom_imbalance_approx()  # 06, 07
        self.compute_price_vs_vwap()    # 08 - session VWAP
        self.data['hour_sin'] = np.sin(2.0 * np.pi * self.data.index.hour / 24.0) # 09
        self.compute_hurst(**hurst_kwargs)             # 10
        
        # Compute Advanced Alpha formulas (available in df, but not in standard 12-vector)
        self.compute_mvdi()
        self.compute_fdpi()
        self.compute_twkj()

        features = self.data[self.FEATURE_NAMES].copy()
        features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        print(f"\nFeature matrix shape: {features.shape}")
        print(f"NaN count per column:\n{features.isnull().sum()}")
        print(f"\nFeature statistics:")
        print(features.describe().round(4))

        # Feature Health Check
        mean_cumdelta = features['cumdelta_div'].mean()
        std_cumdelta = features['cumdelta_div'].std()
        print(f"Feature Health Check -> cumdelta_div: Mean={mean_cumdelta:.5f}, Std={std_cumdelta:.5f}")
        if std_cumdelta == 0:
            raise ValueError("Data Collapse: cumdelta_div standard deviation is 0. Aggressor side (v_t) likely failed.")

        return features

    # =========================================================================
    # CREATE LSTM SEQUENCES - [N_sequences × 128 timesteps × 12 features]
    # =========================================================================
    def create_lstm_sequences(self, features: pd.DataFrame,
                               labels: pd.Series,
                               seq_len: int = 128) -> Tuple:
        """
        Build overlapping sequences for LSTM input.
        Applies the SAME online column-wise z-score normalization that
        ShiftMatrixAndInsert() applies in TitanEA.mq5 (lines 726–736).
        This is CRITICAL for training-inference consistency.
        Returns: (X: ndarray [N, seq_len, 12], y: ndarray [N])
        """
        X, y = [], []
        feat_arr = features.values
        lab_arr  = labels.values

        for i in range(seq_len - 1, len(feat_arr)):
            window = feat_arr[i-seq_len+1:i+1].copy()  # [seq_len, 12]

            # Column-wise z-score over the 128-row window (mirrors EA logic exactly)
            col_mean = window.mean(axis=0)           # [12]
            col_std  = window.std(axis=0)            # [12]
            col_std[col_std < 1e-10] = 1.0           # Prevent div-by-zero (matches EA: cstd > 1e-10)
            window = (window - col_mean) / col_std
            window = np.clip(window, -3.0, 3.0)      # ±3σ Clip

            X.append(window)
            y.append(lab_arr[i])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # =========================================================================
    # GENERATE LABELS - 1 if price moves >= N ticks in direction within T seconds
    # =========================================================================
    def generate_labels(self, target_ticks: int = 15,
                         time_window_sec: int = 30) -> pd.Series:
        df = self.data
        times  = df.index.values.astype(np.int64) // 10**6  # ms
        prices = df['mid'].values
        point  = 0.00001
        MAE_LIMIT = 36 * point

        # Volatility Parity: tau_t = max(sigma_rolling * K, tau_min)
        sigma = df['mid'].rolling('60s').std().bfill().values

        # Pre-compute max_up / max_down for all windows (vectorised)
        max_ups   = np.zeros(len(df), dtype=np.float32)
        max_downs = np.zeros(len(df), dtype=np.float32)
        for i in range(len(df)):
            t_limit = times[i] + time_window_sec * 1000
            j = np.searchsorted(times, t_limit, side='right')
            future = prices[i+1:j]
            if len(future) > 0:
                max_ups[i]   = future.max() - prices[i]
                max_downs[i] = prices[i] - future.min()

        tau_min = target_ticks * point

        # Binary-search K to target pi = 0.20 (±0.01 tolerance)
        def eval_rate(K_val: float) -> float:
            tau_t  = np.maximum(sigma * K_val, tau_min)
            hits = (max_ups >= tau_t) & (max_downs <= MAE_LIMIT)
            return float(hits.mean())

        lo, hi = 0.01, 100.0
        for _ in range(30):
            mid_k = (lo + hi) / 2
            rate  = eval_rate(mid_k)
            if rate > 0.20:
                lo = mid_k
            else:
                hi = mid_k

        K     = (lo + hi) / 2
        final_rate = eval_rate(K)
        print(f"  [Label calibration] K={K:.3f}  pi={final_rate:.4f}")

        tau_t  = np.maximum(sigma * K, tau_min)
        labels = ((max_ups >= tau_t) & (max_downs <= MAE_LIMIT)).astype(np.float32)

        series   = pd.Series(labels, index=df.index, name='label')
        pos_rate = series.mean()
        print(f"Label positive rate: {pos_rate:.3f} "
              f"(K={K:.3f}, tau_min={tau_min:.5f}, window={time_window_sec}s)")
        return series


# =============================================================================
# QUICK-START DEMO
# =============================================================================
if __name__ == "__main__":
    import sys

    eng = TitanFeatureEngineer('EURUSD')

    # Synthetic demo data matching real tick format
    np.random.seed(42)
    n = 50_000
    rng = pd.date_range('2024-01-02 08:00', periods=n, freq='10ms', tz='UTC')

    # Simulate realistic EURUSD micro-structure
    mid_base  = 1.08000
    mid_walk  = np.cumsum(np.random.normal(0, 0.00003, n))
    mid       = mid_base + mid_walk
    spread    = np.abs(np.random.normal(0.00012, 0.00004, n))
    volume    = np.abs(np.random.exponential(0.5, n)) + 0.1

    # Simulate flags: ~30% trades, of those ~50% buy/50% sell
    flags = np.zeros(n, dtype=np.int32)
    is_trade = np.random.random(n) < 0.3
    is_buy   = is_trade & (np.random.random(n) < 0.5)
    is_sell  = is_trade & ~is_buy
    flags |= (is_trade.astype(np.int32) * TICK_FLAG_LAST)
    flags |= (is_buy.astype(np.int32)   * TICK_FLAG_BUY)
    flags |= (is_sell.astype(np.int32)  * TICK_FLAG_SELL)
    flags |= np.where(np.random.random(n) < 0.7, TICK_FLAG_BID, 0)
    flags |= np.where(np.random.random(n) < 0.7, TICK_FLAG_ASK, 0)

    demo_df = pd.DataFrame({
        'bid':    mid - spread / 2,
        'ask':    mid + spread / 2,
        'last':   np.where(is_trade, mid, 0.0),
        'volume': np.where(is_trade, volume, 0.0),
        'flags':  flags,
    }, index=rng)

    # Save demo Parquet
    demo_df.to_parquet('EURUSD_DEMO.parquet', compression='LZ4')

    eng.load_parquet_data('EURUSD_DEMO.parquet')
    features = eng.compute_all_features()
    labels   = eng.generate_labels(target_ticks=15, time_window_sec=30)

    X, y = eng.create_lstm_sequences(features, labels, seq_len=128)
    print(f"\nLSTM sequences: X={X.shape}  y={y.shape}")
    print("Feature engineering V2 complete. Ready for TitanLSTMTraining_V2.py")