# TitanDataLogger_EA — MT5 Tick Data Logger

## Purpose

The `TitanDataLogger_EA.mq5` is a standalone MetaTrader 5 Expert Advisor
that logs every raw tick to a CSV file. The output is directly importable
by `titan_data.py` with zero manual reformatting.

---

## Step-by-Step: Download Historical Data

### 1. Install the EA in MT5

```
Copy TitanDataLogger_EA.mq5 to:
  <MT5_install>/MQL5/Experts/Titan/TitanDataLogger_EA.mq5

In MetaEditor (F4 in MT5):
  Open the file → Press F7 to compile
  Confirm "0 errors" in the compiler output
```

### 2. Allow File Writing

```
MT5 → Tools → Options → Expert Advisors
☑ Allow automated trading
☑ Allow DLL imports (not required but recommended)

The EA writes to: <MT5_install>/MQL5/Files/
```

### 3. Attach to Chart

```
- Open EURUSD M1 chart
- Drag TitanDataLogger_EA from Navigator panel onto chart
- Set parameters:

  InpOutputFile    = "EURUSD_ticks.csv"   ← output filename
  InpWriteHeader   = true                 ← write CSV header (first run)
  InpAppendMode    = false                ← overwrite (set true to append)
  InpLogLive       = true                 ← log live ticks
  InpLogOnBook     = false                ← use OnTick (safer) vs OnBookEvent
  InpExportHistory = true                 ← export stored history at startup
  InpHistoricalDays = 30                  ← how many days to export
  InpHistStart     = 0                    ← 0 = auto (Days back from now)
  InpHistEnd       = 0                    ← 0 = now
  InpFilterZeroSpread = true              ← skip Ask==Bid ticks
  InpFlushEveryN   = 500                  ← flush to disk every 500 ticks
  InpStatusEveryN  = 10000               ← print status every 10k ticks

- Click OK → EA attaches
```

### 4. Wait for History Export

```
Watch the Experts tab (Ctrl+T) for log messages:
  [TitanLogger] Exporting history: 2026.03.15 → 2026.04.14 (30 days)
  [TitanLogger] Chunk 2026.03.15: 847,291 ticks
  [TitanLogger] Chunk 2026.03.22: 912,445 ticks
  ...
  [TitanLogger] History export COMPLETE: 5,832,104 ticks written.
  [TitanLogger] File location: C:\MT5\MQL5\Files\EURUSD_ticks.csv
```

### 5. Import into Python

```python
from titan_data import ParquetConverter

# Convert the CSV to LZ4 Parquet (fast I/O for training)
converter = ParquetConverter()
parquet_path = converter.convert("C:/MT5/MQL5/Files/EURUSD_ticks.csv")
print(f"Saved: {parquet_path}")

# Or load multiple parquet files directly
from titan_data import load_parquet
df = load_parquet(["data/parquet/EURUSD_ticks.parquet"])
print(df.head())
# Tick_Time_ms       Bid       Ask  Flags
# 1712530000123  1.07251   1.07279      6
# 1712530000188  1.07252   1.07280      4
```

---

## Output CSV Format

| Column | Type | Description |
|--------|------|-------------|
| `Tick_Time_ms` | int64 | Unix timestamp in milliseconds |
| `Bid` | float64 | Best bid price (5 decimal places) |
| `Ask` | float64 | Best ask price (5 decimal places) |
| `Flags` | int32 | Titan flag: 2=sell, 4=buy, 6=neutral, 1=trade print |

**Flag encoding matches `titan_config.py` exactly:**

| MT5 Raw Flag | Titan Flag | Meaning |
|---|---|---|
| `TICK_FLAG_BUY` or `TICK_FLAG_SELL` | 1 | Confirmed execution/trade print |
| `TICK_FLAG_ASK` only | 4 | Buy-side aggressor (ask updated) |
| `TICK_FLAG_BID` only | 2 | Sell-side aggressor (bid updated) |
| Both or neither | 6 | Market maker neutral refresh |

---

## Live Streaming Mode

Leave the EA attached for continuous live tick logging. The EA:
- Logs every tick via `OnTick()` (or `OnBookEvent()` if `InpLogOnBook=true`)
- Flushes to disk every `InpFlushEveryN` ticks (default 500)
- Prints a status line every `InpStatusEveryN` ticks (default 10,000)
- File can be read by Python while EA is still writing (safe — append mode)

---

## Troubleshooting

**"Cannot open file" error:**
- Check Tools → Options → Expert Advisors → Allow automated trading is ON
- Check file is not open in Excel or another process

**No ticks in history:**
- MT5 stores tick history for broker-dependent lookback (typically 30–90 days)
- Reduce `InpHistoricalDays` to match your broker's available history
- In MT5: Tools → History Center → select EURUSD → Download to extend history

**EA shows "No ticks for chunk":**
- This is normal for non-trading hours (weekends, holidays)
- The EA skips empty chunks automatically and continues

**CSV file is very large (>1GB):**
- 30 days of EURUSD ticks ≈ 500MB–2GB depending on broker tick density
- Convert to Parquet immediately: Parquet with LZ4 reduces size by ~70%
- Use `InpHistoricalDays = 7` to start with a smaller test dataset
