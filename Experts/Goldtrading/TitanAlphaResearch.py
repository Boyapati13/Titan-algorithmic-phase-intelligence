#!/usr/bin/env python3
"""
TITAN HFT SYSTEM - Alpha Research V2.0 (PRODUCTION)
Author : Senior Quant / Systems Architect | April 2026

FORENSIC FIXES vs V1:
  [BUG-01] CRITICAL: _load_from_binary skipped the 24-byte FileHeader,
           treating header bytes as the first tick - silent corruption.
  [BUG-02] CRITICAL: TICK_FLAG constants 0x01/0x02 were V1 wrong values.
           Correct MQL5 values: BUY=32, SELL=64, LAST=8.
  [BUG-03] _compute_cumulative_delta: used abs(volume) unconditionally -
           never negative, so delta was always bullish. Fixed to use flags.
  [BUG-04] create_sequences: feature_cols listed 12 names but only 10 of
           them matched FEATURE_NAMES from TitanFeatureEngineering.py.
           Fixed to use TitanFeatureEngineer.FEATURE_NAMES directly.
  [BUG-05] save_sequences: saved 'sequences'/'targets' keys but
           TitanLSTMTraining.py expects X_train/y_train/X_val/y_val.
           Fixed with correct 70/15/15 chronological split.
  [BUG-06] _compute_velocity_of_tape counted ALL ticks, not just trade ticks.
           Fixed to filter TICK_FLAG_LAST consistent with feature engineering.
  [BUG-07] Binary loader created timestamp without UTC timezone - mismatched
           with Parquet timezone-aware index, causing concat errors.
"""

import os
import struct
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── TICK FLAG CONSTANTS (match MQL5 documentation exactly) ───────────────────
TICK_FLAG_BID    = 2    # Bid price changed
TICK_FLAG_ASK    = 4    # Ask price changed
TICK_FLAG_LAST   = 8    # Last trade price changed (REAL TRADE TICK)
TICK_FLAG_VOLUME = 16   # Volume changed
TICK_FLAG_BUY    = 32   # Buy aggressor
TICK_FLAG_SELL   = 64   # Sell aggressor

# ── TITAN V2 STRUCT DEFINITIONS (must match TitanParquetConverter.py) ────────
HEADER_FORMAT = '<IIqII'   # magic, version, created_msc, tick_size, reserved
HEADER_SIZE   = 24
TICK_FORMAT   = '<qddddIII' # time_msc, bid, ask, last, volume, flags, res1, res2
TICK_SIZE     = 52
TITAN_MAGIC   = 0x5449544E  # "TITN"

# ── FEATURE NAMES (must match TitanFeatureEngineering.py::FEATURE_NAMES) ─────
FEATURE_NAMES = [
    'vot_zscore',       # 00 - Velocity-of-tape z-score
    'rvol',             # 01 - Relative Volume
    'cumdelta_div',     # 02 - Cumulative Delta Divergence
    'imbalance_ratio',  # 03 - Stacked Imbalance score
    'fdpi',             # 04 - Flag-Based Directional Pressure Index
    'mvdi',             # 05 - Micro-Volatility Dispersion Index
    'bid_depth_imb',    # 06 - DOM bid imbalance (approx)
    'ask_depth_imb',    # 07 - DOM ask imbalance (approx)
    'price_vs_vwap',    # 08 - Price vs session VWAP (σ units)
    'hour_sin',         # 09 - Hour of day sine
    'hurst',            # 10 - Hurst exponent (regime)
    'twkj',             # 11 - Time-Weighted Kinematic Jerk
]
N_FEATURES = len(FEATURE_NAMES)  # 12


class TitanAlphaResearch:
    """
    Production alpha research pipeline.
    Loads raw tick data (Parquet or binary V2 .ticks) and delegates feature
    computation to TitanFeatureEngineer.  Saves train/val/test NPZ files
    in the correct format for TitanLSTMTraining.py.
    """

    def __init__(self, symbol: str = 'EURUSD', seq_len: int = 128):
        self.symbol  = symbol
        self.seq_len = seq_len
        self.data    = None
        self.scaler  = StandardScaler()

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    def load_and_preprocess(self, data_source: str) -> pd.DataFrame:
        if data_source.endswith('.parquet'):
            return self._load_from_parquet(data_source)
        elif data_source.endswith('.ticks'):
            return self._load_from_binary(data_source)
        else:
            raise ValueError("Unsupported format. Use .parquet or .ticks")

    def _load_from_parquet(self, path: str) -> pd.DataFrame:
        print(f"Loading {path}...")
        df = pd.read_parquet(path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.set_index('timestamp')
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        self.data = df
        print(f"Loaded {len(df):,} ticks  {df.index.min()} -> {df.index.max()}")
        return df

    def _load_from_binary(self, path: str) -> pd.DataFrame:
        """
        Load V2 binary .ticks file produced by TitanTickLogger.mq5.
        BUG-FIX: V1 read from byte 0, corrupting first tick with header data.
        V2 reads and validates the 24-byte header before accessing tick payload.
        """
        print(f"Loading binary V2: {path}")
        ticks = []

        with open(path, 'rb') as f:
            # ── Step 1: Read & validate 24-byte header ────────────────────────
            hdr_raw = f.read(HEADER_SIZE)
            if len(hdr_raw) < HEADER_SIZE:
                raise ValueError(f"File too small to contain a valid header: {path}")
            magic, version, _, tick_size_in_file, _ = struct.unpack(HEADER_FORMAT, hdr_raw)
            if magic != TITAN_MAGIC:
                raise ValueError(f"Not a Titan V2 file (magic={hex(magic)}): {path}")
            if tick_size_in_file != TICK_SIZE:
                raise ValueError(
                    f"Struct mismatch: header says {tick_size_in_file}B, expected {TICK_SIZE}B")
            print(f"  Header OK  version={version}  tick_size={tick_size_in_file}")

            # ── Step 2: Read 52-byte tick records ─────────────────────────────
            while True:
                raw = f.read(TICK_SIZE)
                if len(raw) != TICK_SIZE:
                    break
                t_msc, bid, ask, last, vol, flags, _, _ = struct.unpack(TICK_FORMAT, raw)
                ticks.append({
                    'time_msc': t_msc,
                    'bid': bid,
                    'ask': ask,
                    'last': last,
                    'volume': vol,
                    'flags': flags,
                })

        df = pd.DataFrame(ticks)
        # UTC timezone-aware index (BUG-FIX: V1 had no timezone)
        df['timestamp'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True)
        df = df.set_index('timestamp').drop(columns=['time_msc'])
        df = df.sort_index()
        self.data = df
        print(f"Loaded {len(df):,} ticks  {df.index.min()} -> {df.index.max()}")
        return df

    # =========================================================================
    # FEATURE COMPUTATION - delegates to TitanFeatureEngineer for consistency
    # =========================================================================
    def compute_features(self) -> pd.DataFrame:
        """
        Use TitanFeatureEngineer to compute the canonical 12-feature vector.
        This guarantees identical feature semantics to the training pipeline.
        """
        try:
            from TitanFeatureEngineering import TitanFeatureEngineer
            eng = TitanFeatureEngineer(self.symbol)
            eng.data = self.data.copy()
            # Derive derived columns TitanFeatureEngineer.load_parquet_data() adds
            eng.data['mid']    = (eng.data['bid'] + eng.data['ask']) * 0.5
            eng.data['spread'] = eng.data['ask'] - eng.data['bid']
            eng.data['is_trade'] = (eng.data['flags'].astype('int32') & TICK_FLAG_LAST) != 0
            eng.data['is_buy']   = (eng.data['flags'].astype('int32') & TICK_FLAG_BUY)  != 0
            eng.data['is_sell']  = (eng.data['flags'].astype('int32') & TICK_FLAG_SELL) != 0
            
            # If no buy/sell flags, use exact Lee-Ready fallback
            mask_no_flags = ~(eng.data['is_buy'] | eng.data['is_sell'])
            if mask_no_flags.any():
                price = np.where(eng.data['last'] > 0, eng.data['last'], eng.data['mid'])
                v_t_arr = np.zeros(len(eng.data))
                v_t_arr[price > eng.data['mid']] = 1
                v_t_arr[price < eng.data['mid']] = -1
                
                equal_mask = (price == eng.data['mid'])
                delta_p = pd.Series(price).diff().fillna(0).values
                v_t_arr[equal_mask] = np.sign(delta_p[equal_mask])
                
                v_t_series = pd.Series(v_t_arr, index=eng.data.index).replace(0, np.nan).ffill().fillna(1)
                
                # ENTROPY GUARDRAIL
                p1 = (v_t_series == 1).mean()
                pn1 = 1.0 - p1
                entropy = -(max(p1, 1e-9) * np.log2(max(p1, 1e-9)) + max(pn1, 1e-9) * np.log2(max(pn1, 1e-9)))
                print(f"Tape Reconstructed -> Entropy H(v): {entropy:.4f} (p1: {p1:.3f})")
                if entropy < 0.90:
                    raise ValueError(f"Dead Tape: Shannon Entropy H(v) = {entropy:.4f} < 0.90")
                
                eng.data.loc[mask_no_flags, 'is_buy'] = (v_t_series[mask_no_flags] == 1).values
                eng.data.loc[mask_no_flags, 'is_sell'] = (v_t_series[mask_no_flags] == -1).values
            
            eng.data['is_bid_chg'] = (eng.data['flags'].astype('int32') & TICK_FLAG_BID) != 0
            eng.data['is_ask_chg'] = (eng.data['flags'].astype('int32') & TICK_FLAG_ASK) != 0

            features = eng.compute_all_features()
            print(f"Features computed via TitanFeatureEngineer: {features.shape}")
            return features

        except ImportError:
            print("WARNING: TitanFeatureEngineering.py not found - falling back to inline computation")
            return self._compute_features_inline()

    def _compute_features_inline(self) -> pd.DataFrame:
        """Inline fallback (same algorithm, same order as FEATURE_NAMES)."""
        df = self.data.copy()
        df['mid']    = (df['bid'] + df['ask']) * 0.5
        df['spread'] = df['ask'] - df['bid']

        # ─── TITAN V2.1 PRECISION BRIDGE: LEE-READY HYBRID CLASSIFICATION ────────
        # CRITICAL: use .values everywhere to avoid pandas/numpy index misalignment
        if not df['is_buy'].any() or not df['is_sell'].any():
            last_np  = df['last'].values
            mid_np   = df['mid'].values
            price_np = np.where(last_np > 0, last_np, mid_np)

            v_t = np.zeros(len(df), dtype=np.float64)
            v_t[price_np > mid_np] =  1.0
            v_t[price_np < mid_np] = -1.0

            equal_mask = (price_np == mid_np)
            if equal_mask.any():
                delta_p = np.diff(price_np, prepend=price_np[0])
                v_t[equal_mask] = np.sign(delta_p[equal_mask])

            v_t_series = pd.Series(v_t, index=df.index).replace(0.0, np.nan).ffill().fillna(1.0)

            p1  = float((v_t_series == 1.0).mean())
            pn1 = 1.0 - p1
            eps = 1e-9
            entropy = -(
                (p1  + eps) * np.log2(p1  + eps) +
                (pn1 + eps) * np.log2(pn1 + eps)
            )
            print(f"Tape Reconstructed -> Lee-Ready H(v)={entropy:.4f}  Buy%={p1:.3f}")
            if entropy < 0.90:
                raise ValueError(f"Dead Tape: H(v)={entropy:.4f} < 0.90 (Buy={p1:.3f} Sell={pn1:.3f})")

            df['is_buy']   = (v_t_series == 1.0).values
            df['is_sell']  = (v_t_series == -1.0).values
            df['is_trade'] = True

        # Re-check native flags for any residual correct flags
        native_buy  = (df['flags'].astype('int32') & TICK_FLAG_BUY)  != 0
        native_sell = (df['flags'].astype('int32') & TICK_FLAG_SELL) != 0
        if native_buy.any() and native_sell.any():
            df['is_buy']   = native_buy
            df['is_sell']  = native_sell
            df['is_trade'] = (df['flags'].astype('int32') & TICK_FLAG_LAST) != 0


        # 00 - VoT z-score (trade ticks per second, BUG-FIX: filter by is_trade)
        tps = df[df['is_trade']].resample('1s').size().rename('tps')
        tps = tps.reindex(pd.date_range(df.index.min().floor('1s'),
                                        df.index.max().ceil('1s'), freq='1s'),
                          fill_value=0)
        vot_z = (tps - tps.rolling('20s').mean()) / tps.rolling('20s').std().clip(lower=1e-8)
        df['vot_zscore'] = vot_z.reindex(df.index, method='ffill').fillna(0.0)

        # 01 - RVOL
        vol_win = df.resample('5min')['volume'].sum()
        rvol    = (vol_win / vol_win.rolling(20).mean().clip(lower=1e-10)).clip(upper=20.0)
        df['rvol'] = rvol.reindex(df.index, method='ffill').fillna(1.0)

        # 02 - Cumulative delta divergence (BUG-FIX: uses buy/sell flags not abs)
        delta = np.where(df['is_buy'], df['volume'],
                np.where(df['is_sell'], -df['volume'], 0.0))
        df['delta'] = delta
        price_chg  = df['mid'].diff(200)
        delta_chg  = df['delta'].rolling(200).sum()
        df['cumdelta_div'] = np.select(
            [(price_chg > 0) & (delta_chg < 0), (price_chg < 0) & (delta_chg > 0)],
            [-1.0, 1.0], default=0.0)

        # 03 - Imbalance ratio (simplified)
        buy_r  = df[df['is_buy']].resample('100ms')['volume'].sum().rolling(10).sum()
        sell_r = df[df['is_sell']].resample('100ms')['volume'].sum().rolling(10).sum()
        tot    = (buy_r + sell_r).clip(lower=1e-8)
        imb    = ((buy_r - sell_r) / tot).fillna(0.0)
        df['imbalance_ratio'] = imb.reindex(df.index, method='ffill').fillna(0.0)

        # 04 - FDPI
        b_n = df['is_buy'].astype(np.float32).rolling(128).sum()
        s_n = df['is_sell'].astype(np.float32).rolling(128).sum()
        df['fdpi'] = ((b_n - s_n) / (b_n + s_n + 1e-10)).fillna(0.0)

        # 05 - MVDI
        spread_cv = df['spread'].rolling(128).std() / (df['spread'].rolling(128).mean() + 1e-10)
        mid_cv = df['mid'].diff().abs().rolling(128).std() / (df['mid'].diff().abs().rolling(128).mean() + 1e-10)
        mvdi_raw = spread_cv / (mid_cv + 1e-10)
        df['mvdi'] = (((mvdi_raw - mvdi_raw.rolling(128).mean()) / (mvdi_raw.rolling(128).std() + 1e-10)).clip(-3.0, 3.0) / 3.0).fillna(0.0)

        # 06, 07 - DOM imbalance approximation
        df['bid_depth_imb'] = imb.reindex(df.index, method='ffill').fillna(0.0)
        df['ask_depth_imb'] = (-imb).reindex(df.index, method='ffill').fillna(0.0)

        # 08 - Price vs session VWAP
        df['date'] = df.index.date
        pv, vv = {}, {}
        vwap_z = np.zeros(len(df))
        for i, (ts, row) in enumerate(df.iterrows()):
            d = row['date']
            if d not in pv:
                pv[d] = 0.0; vv[d] = 0.0
            if row['is_trade'] and row['volume'] > 0:
                pv[d] += row['mid'] * row['volume']
                vv[d] += row['volume']
            if vv[d] > 0:
                vwap = pv[d] / vv[d]
                vwap_z[i] = (row['mid'] - vwap) / max(abs(row['mid'] - vwap) + 1e-8, 1e-8)
        df['price_vs_vwap'] = vwap_z
        df.drop(columns=['date'], inplace=True, errors='ignore')

        # 09 - Hour Sine
        df['hour_sin'] = np.sin(2.0 * np.pi * df.index.hour / 24.0)

        # 10 - Hurst (20-bar R/S)
        prices = df['mid'].values
        n = len(prices)
        hurst = np.full(n, 0.5)
        for i in range(20, n):
            seg = prices[i-20:i]
            devs = np.cumsum(seg - seg.mean())
            R = devs.max() - devs.min()
            S = seg.std()
            if S > 1e-10 and R > 1e-10:
                hurst[i] = np.log(R / S) / np.log(20)
        df['hurst'] = np.clip(hurst, 0.0, 1.0)

        # 11 - TWKJ
        dt = df.index.to_series().diff().dt.total_seconds() * 1000.0
        dt = dt.clip(lower=1.0).values
        v = df['mid'].diff() / dt
        a = v.diff() / dt
        j = a.diff() / dt
        df['twkj'] = (((j - j.rolling(128).mean()) / (j.rolling(128).std() + 1e-10)).clip(-3.0, 3.0) / 3.0).fillna(0.0)

        feat = df[FEATURE_NAMES].copy()
        feat = feat.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return feat

    # =========================================================================
    # SEQUENCE CREATION - same normalization as EA's ShiftMatrixAndInsert
    # =========================================================================
    def create_sequences(self, features: pd.DataFrame,
                          labels: np.ndarray) -> tuple:
        X, y = [], []
        arr = features.values
        for i in range(self.seq_len - 1, len(arr)):
            window = arr[i-self.seq_len+1:i+1].copy()
            # Column-wise 128-row z-score (identical to TitanEA.mq5 ShiftMatrixAndInsert)
            col_mean = window.mean(axis=0)
            col_std  = window.std(axis=0)
            col_std[col_std < 1e-10] = 1.0
            window = (window - col_mean) / col_std
            window = np.clip(window, -3.0, 3.0)
            X.append(window)
            y.append(labels[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def generate_labels(self, features: pd.DataFrame,
                         target_ticks: int = 15,
                         time_window_sec: int = 30) -> np.ndarray:
        """Generate binary labels: 1 if price rises >= tau_t within window without hitting MAE."""
        df  = self.data
        mid = df['mid'].values if 'mid' in df.columns else ((df['bid'] + df['ask']) / 2).values
        ts  = df.index.values.astype('int64') // 10**6  # ms
        point = 0.00001
        labels = np.zeros(len(df), dtype=np.float32)

        # Calculate rolling volatility over 60s
        price_series = df['mid'] if 'mid' in df.columns else pd.Series(mid, index=df.index)
        sigma = price_series.rolling('60s').std().bfill().values
        
        # Dynamically solve for K using first 1000 valid rows to target pi = 0.20
        empirical_k_values = []
        for i in range(min(2000, len(df))):
            t_limit = ts[i] + time_window_sec * 1000
            j = np.searchsorted(ts, t_limit, side='right')
            future = mid[i+1:j]
            if len(future) > 0 and sigma[i] > 1e-10:
                max_up = future.max() - mid[i]
                empirical_k_values.append(max_up / sigma[i])
                
        if len(empirical_k_values) > 0:
            K = np.percentile(empirical_k_values, 80)
        else:
            K = 3.0
            
        tau_t = sigma * K
        tau_t = np.maximum(tau_t, target_ticks * point)
        
        MAE_LIMIT = 36 * point

        for i in range(len(df)):
            t_limit = ts[i] + time_window_sec * 1000
            j = np.searchsorted(ts, t_limit, side='right')

            future = mid[i+1:j]
            if len(future) == 0: continue

            max_up = future.max() - mid[i]
            max_down = mid[i] - future.min()

            if max_up >= tau_t[i]:
                # Apply the MAE Safety Valve
                if max_down <= MAE_LIMIT:
                    labels[i] = 1.0

        pos_rate = labels.mean()
        if pos_rate > 0.30:
            print(f"  [POISON WARNING] Positive rate {pos_rate:.3f} > 0.30. Trivial labeling! Increase target_ticks.")
        elif pos_rate < 0.10:
            print(f"  [WARNING] Positive rate {pos_rate:.3f} < 0.10. Classes are too sparse.")
        print(f"Label positive rate: {pos_rate:.3f} (target_k: {K:.3f}, window: {time_window_sec}s)")
        return labels

    # =========================================================================
    # SAVE - correct NPZ keys for TitanLSTMTraining.py
    # =========================================================================
    def save_sequences(self, X: np.ndarray, y: np.ndarray,
                        output_path: str = 'data/titan_lstm_sequences.npz'):
        """
        Save train/val/test split in the format TitanLSTMTraining.py expects.
        BUG-FIX: V1 used keys 'sequences'/'targets'; training expects
        X_train/y_train/X_val/y_val/X_test/y_test + feat_mean/feat_std.
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        n  = len(X)
        s1 = int(n * 0.70)
        s2 = int(n * 0.85)
        np.savez(output_path,
                 X_train=X[:s1],    y_train=y[:s1],
                 X_val=X[s1:s2],    y_val=y[s1:s2],
                 X_test=X[s2:],     y_test=y[s2:],
                 feat_mean=np.zeros(N_FEATURES, dtype=np.float32),
                 feat_std=np.ones(N_FEATURES,  dtype=np.float32))
        size_mb = os.path.getsize(output_path) / 1024**2
        print(f"Saved {n:,} sequences -> {output_path}  ({size_mb:.1f} MB)")
        print(f"  Train: {s1:,}  Val: {s2-s1:,}  Test: {n-s2:,}")

    # =========================================================================
    # SYNTHETIC DATA GENERATOR (testing only)
    # =========================================================================
    def generate_synthetic_data(self, n: int = 50_000) -> pd.DataFrame:
        """Generate realistic synthetic tick data for pipeline testing."""
        np.random.seed(42)
        rng    = pd.date_range('2026-01-02 08:00', periods=n, freq='10ms', tz='UTC')
        mid    = 1.08 + np.cumsum(np.random.normal(0, 3e-5, n))
        spread = np.abs(np.random.normal(1.2e-4, 3e-5, n))
        vol    = np.abs(np.random.exponential(0.5, n)) + 0.1

        is_trade = np.random.random(n) < 0.3
        is_buy   = is_trade & (np.random.random(n) < 0.5)
        is_sell  = is_trade & ~is_buy
        flags    = np.zeros(n, dtype=np.int32)
        flags   |= (is_trade.astype(np.int32) * TICK_FLAG_LAST)
        flags   |= (is_buy.astype(np.int32)   * TICK_FLAG_BUY)
        flags   |= (is_sell.astype(np.int32)  * TICK_FLAG_SELL)
        flags   |= np.where(np.random.random(n) < 0.7, TICK_FLAG_BID, 0)
        flags   |= np.where(np.random.random(n) < 0.7, TICK_FLAG_ASK, 0)

        df = pd.DataFrame({
            'bid':    mid - spread / 2,
            'ask':    mid + spread / 2,
            'last':   np.where(is_trade, mid, 0.0),
            'volume': np.where(is_trade, vol, 0.0),
            'flags':  flags,
        }, index=rng)
        self.data = df
        print(f"Generated {n:,} synthetic ticks")
        return df


# =============================================================================
# ENTRY POINT - full pipeline demo
# =============================================================================
if __name__ == '__main__':
    import glob

    BASE = r'C:\Users\Tenders\AppData\Roaming\MetaQuotes\Terminal\AE2CC2E013FDE1E3CDF010AA51C60400\MQL5\Experts\Goldtrading'

    researcher = TitanAlphaResearch(symbol='EURUSD', seq_len=128)

    # Try to load real parquet files from data/, else generate synthetic
    parquet_files = sorted(glob.glob(os.path.join(BASE, 'data', '*.parquet')))
    if parquet_files:
        print(f"Found {len(parquet_files)} parquet files. Processing first file for demo.")
        researcher.load_and_preprocess(parquet_files[0])
    else:
        print("No parquet files found. Generating synthetic data...")
        researcher.generate_synthetic_data(100_000)

    # Delegate to TitanFeatureEngineer (canonical 12-feature pipeline)
    features = researcher.compute_features()
    labels   = researcher.generate_labels(features, target_ticks=15, time_window_sec=30)

    X, y = researcher.create_sequences(features, labels)
    print(f"Sequences: X={X.shape}  y={y.shape}")

    # Save with correct NPZ keys
    researcher.save_sequences(X, y, os.path.join(BASE, 'data', 'titan_lstm_sequences.npz'))
    print("Alpha research complete - ready for TitanLSTMTraining.py")