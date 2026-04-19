#!/usr/bin/env python3
"""
TITAN HFT SYSTEM - Parquet Converter V2.1 (PRODUCTION)
Author : Senior Quant / Systems Architect | April 2026
"""

import struct
import os
import glob
import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ─── V2 STRUCT DEFINITIONS ────────────────────────────────────────────────────
HEADER_FORMAT = '<IIqII'
HEADER_SIZE   = 24
TICK_FORMAT   = '<qddddIII'
TICK_SIZE     = 52
TITAN_MAGIC   = 0x5449544E

# ─── FIXED PATHS (Goldtrading folder layout) ──────────────────────────────────
TERMINAL_ID   = 'AE2CC2E013FDE1E3CDF010AA51C60400'
_APPDATA      = os.environ.get('APPDATA', '')

# 1. FIXED: MT5 Common Files directory (Where your .ticks are)
DEFAULT_TICKS_DIR = r"C:\Users\Tenders\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Goldtrading"

# 2. FIXED: Output directory (Where the AI training scripts look for data)
DEFAULT_OUTPUT_DIR = os.path.join(
    _APPDATA, 'MetaQuotes', 'Terminal', TERMINAL_ID, 
    'MQL5', 'Experts', 'Goldtrading', 'data'
)

# =============================================================================
# CORE CONVERSION FUNCTION
# =============================================================================
def convert_ticks_to_parquet(ticks_file_path: str,
                              output_dir: str = None) -> bool:
    """
    Convert a single Titan V2 binary .ticks file to LZ4-compressed Parquet.
    """
    if not os.path.exists(ticks_file_path):
        print(f"  [!] File not found: {ticks_file_path}")
        return False

    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    basename     = os.path.basename(ticks_file_path)
    parquet_file = os.path.join(out_dir, basename.replace('.ticks', '.parquet'))

    print(f"\n[CONVERT] {basename}")
    print(f"  Source : {ticks_file_path}")
    print(f"  Output : {parquet_file}")

    ticks_data = []
    try:
        with open(ticks_file_path, 'rb') as f:
            # Read & validate 24-byte FileHeader
            header_raw = f.read(HEADER_SIZE)
            if len(header_raw) < HEADER_SIZE:
                print(f"  [!] File too small to contain a valid header")
                return False

            magic, version, _, tick_size, _ = struct.unpack(HEADER_FORMAT, header_raw)

            if magic != TITAN_MAGIC:
                print(f"  [!] Invalid magic bytes: {hex(magic)}")
                return False

            if tick_size != TICK_SIZE:
                print(f"  [!] STRUCT MISMATCH: header tick_size={tick_size}B, expected {TICK_SIZE}B")
                return False

            file_size   = os.path.getsize(ticks_file_path)
            total_ticks = (file_size - HEADER_SIZE) // TICK_SIZE
            print(f"  Header OK  ver={version}  ticks={total_ticks:,}")

            # Read 52-byte tick records
            for _ in tqdm(range(total_ticks), desc="  Parsing", unit="ticks", leave=False):
                raw = f.read(TICK_SIZE)
                if not raw or len(raw) != TICK_SIZE:
                    break

                t_msc, bid, ask, last, vol, flags, _, _ = struct.unpack(TICK_FORMAT, raw)

                ticks_data.append({
                    'time_msc': t_msc,
                    'bid':      bid,
                    'ask':      ask,
                    'last':     last,
                    'volume':   vol,
                    'flags':    flags,
                })

    except Exception as e:
        print(f"  [!] Read error: {e}")
        return False

    if not ticks_data:
        return False

    # Build DataFrame
    df = pd.DataFrame(ticks_data)
    df['timestamp'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    # OTC NORMALIZATION
    df['bid'] = df['bid'].where(df['bid'] > 0, other=None).ffill()
    df['ask'] = df['ask'].where(df['ask'] > 0, other=None).ffill()
    df.dropna(subset=['bid', 'ask'], inplace=True)

    df['mid']    = (df['bid'] + df['ask']) * 0.5
    df['last']   = df['mid']
    df['volume'] = 0.0

    delta_mid = df['mid'].diff()
    df['flags'] = df['flags'].astype(int)
    
    TICK_FLAG_LAST = 8
    TICK_FLAG_BUY  = 32
    TICK_FLAG_SELL = 64
    
    df['flags'] |= TICK_FLAG_LAST
    df.loc[delta_mid > 0, 'flags'] |= TICK_FLAG_BUY
    df.loc[delta_mid < 0, 'flags'] |= TICK_FLAG_SELL

    # Write LZ4 Parquet
    cols = ['bid', 'ask', 'last', 'volume', 'flags']
    table = pa.Table.from_pandas(df[cols])
    pq.write_table(table, parquet_file, compression='LZ4')

    size_mb = os.path.getsize(parquet_file) / 1024**2
    print(f"  [OK] {len(df):,} ticks -> {parquet_file} ({size_mb:.1f} MB)")
    return True

def batch_convert(input_dir: str = None, output_dir: str = None) -> int:
    src = input_dir  or DEFAULT_TICKS_DIR
    dst = output_dir or DEFAULT_OUTPUT_DIR

    print(f"\n{'='*60}")
    print(f"TITAN PARQUET CONVERTER V2.1")
    print(f"{'='*60}")
    print(f"  Ticks source : {src}")
    print(f"  Parquet dest : {dst}")

    if not os.path.isdir(src):
        print(f"\n  [!] Ticks directory not found: {src}")
        return 0

    files = sorted(glob.glob(os.path.join(src, '*.ticks')))
    if not files:
        print(f"\n  [!] No .ticks files found.")
        return 0

    os.makedirs(dst, exist_ok=True)
    ok_count = 0
    for f in files:
        if convert_ticks_to_parquet(f, dst):
            ok_count += 1
    return ok_count

def main():
    parser = argparse.ArgumentParser(description='Titan V2 .ticks -> LZ4 Parquet converter')
    parser.add_argument('input', nargs='?', default=None)
    parser.add_argument('-o', '--output', default=None)

    args, unknown = parser.parse_known_args()
    
    batch_convert(args.input or DEFAULT_TICKS_DIR, args.output or DEFAULT_OUTPUT_DIR)

if __name__ == '__main__':
    main()