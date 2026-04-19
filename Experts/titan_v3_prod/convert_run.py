import time
from pathlib import Path
from titan_config import setup_logging
from titan_data import ParquetConverter

setup_logging("INFO")

# ── Source CSV (MT5 Strategy Tester output) ───────────────────────────────────
csv_path = Path(
    r"C:\Users\Tenders\AppData\Roaming\MetaQuotes\Tester"
    r"\AE2CC2E013FDE1E3CDF010AA51C60400"
    r"\Agent-127.0.0.1-3000\MQL5\Files\EURUSD_ticks.csv"
)

print("=" * 65)
print("  Titan V3.0 -- CSV -> Parquet Converter")
print("=" * 65)
print(f"  Source : {csv_path}")
print(f"  Size   : {csv_path.stat().st_size / 1e9:.2f} GB")
print("=" * 65)

t0 = time.time()
out = ParquetConverter().convert(src=csv_path)
elapsed = time.time() - t0

print("=" * 65)
print(f"  Output : {out}")
print(f"  Size   : {out.stat().st_size / 1e6:.1f} MB")
print(f"  Time   : {elapsed:.1f}s")
print("=" * 65)
print()
print("Next step:")
print('  python titan_pipeline.py "%s"' % out)
