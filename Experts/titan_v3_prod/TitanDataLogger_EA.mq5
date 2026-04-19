//+------------------------------------------------------------------+
//| TitanDataLogger_EA.mq5                                          |
//| Titan V3.0 — MT5 Tick Data Logger Expert Advisor                |
//|                                                                  |
//| PURPOSE:                                                         |
//|   Logs every raw tick to a CSV file in the MT5 Files directory.  |
//|   The CSV is directly readable by titan_data.py (ParquetConverter|
//|   or load_parquet) without any manual export steps.              |
//|                                                                  |
//| OUTPUT FORMAT (CSV columns):                                     |
//|   Tick_Time_ms, Bid, Ask, Flags                                  |
//|                                                                  |
//| HOW TO USE:                                                      |
//|   1. Compile in MetaEditor → press F7                            |
//|   2. Attach to EURUSD M1 chart (timeframe doesn't matter)        |
//|   3. Set InpOutputFile = "EURUSD_ticks.csv"                      |
//|   4. Run for desired session (Asia / London / New York)          |
//|   5. Find the file in: MT5_install/MQL5/Files/                   |
//|   6. In Python:                                                  |
//|        from titan_data import ParquetConverter                   |
//|        ParquetConverter().convert("path/to/EURUSD_ticks.csv")    |
//|                                                                  |
//| DOWNLOAD HISTORICAL DATA (built-in MT5 export):                  |
//|   Use InpHistoricalDays > 0 to export the last N days of         |
//|   stored tick history at EA startup (before live streaming).     |
//|                                                                  |
//| FLAG ENCODING (matches titan_config.py exactly):                 |
//|   TICK_FLAG_BID  (0x02) → 2   sell-side aggressor                |
//|   TICK_FLAG_ASK  (0x04) → 4   buy-side aggressor                 |
//|   TICK_FLAG_BID+ASK → 6       MM neutral refresh                 |
//|   TICK_FLAG_BUY/SELL → 1      confirmed trade print              |
//+------------------------------------------------------------------+
#property copyright "Titan HFT Systems"
#property version   "1.00"
#property strict
#property description "Logs every tick to CSV. Use with titan_data.py."

//── Input parameters ─────────────────────────────────────────────────────────
input group "=== FILE ==="
input string   InpOutputFile      = "EURUSD_ticks.csv";  // Output CSV filename
input bool     InpWriteHeader     = true;                 // Write CSV header row
input bool     InpAppendMode      = false;                // Append to existing file

input group "=== LOGGING ==="
input bool     InpLogLive         = true;    // Log live ticks as they arrive
input bool     InpLogOnBook       = false;   // Use OnBookEvent (DOM-level, faster)

input group "=== HISTORICAL DOWNLOAD ==="
input bool     InpExportHistory   = true;    // Export stored history at startup
input int      InpHistoricalDays  = 30;      // Days of history to export (0 = skip)
input datetime InpHistStart       = 0;       // Custom start (0 = auto from Days)
input datetime InpHistEnd         = 0;       // Custom end   (0 = now)

input group "=== FILTERING ==="
input bool     InpFilterZeroSpread = true;   // Skip ticks where Ask == Bid
input bool     InpFilterNegative   = true;   // Skip ticks where Bid <= 0

input group "=== PERFORMANCE ==="
input int      InpFlushEveryN     = 500;     // Flush file buffer every N ticks
input int      InpStatusEveryN    = 10000;   // Print status every N ticks

//── Globals ───────────────────────────────────────────────────────────────────
int    g_file       = INVALID_HANDLE;
string g_filepath   = "";
long   g_tick_count = 0;
long   g_skip_count = 0;
bool   g_hist_done  = false;

//── Flag translation — matches titan_config.py FlagMapper exactly ────────────
int TitanFlag(uint mt5_flags)
  {
   // Trade prints (confirmed execution)
   if((mt5_flags & TICK_FLAG_BUY)  != 0) return 1;
   if((mt5_flags & TICK_FLAG_SELL) != 0) return 1;
   // Both sides updated simultaneously
   if((mt5_flags & TICK_FLAG_BID)  != 0 &&
      (mt5_flags & TICK_FLAG_ASK)  != 0)  return 6;
   // Ask-only: buy-side aggressor
   if((mt5_flags & TICK_FLAG_ASK)  != 0) return 4;
   // Bid-only: sell-side aggressor
   if((mt5_flags & TICK_FLAG_BID)  != 0) return 2;
   // Default: neutral refresh
   return 6;
  }

//+------------------------------------------------------------------+
//| Open output CSV file                                             |
//+------------------------------------------------------------------+
bool OpenFile()
  {
   int flags = FILE_WRITE | FILE_CSV | FILE_ANSI;
   if(InpAppendMode) flags |= FILE_READ;    // FILE_READ required for append
   
   g_file = FileOpen(InpOutputFile, flags, ',');
   if(g_file == INVALID_HANDLE)
     {
      Print("[TitanLogger] ERROR: Cannot open '", InpOutputFile, 
            "'. Error=", GetLastError(),
            ". Check Experts/AutoTrading permissions in MT5 Tools→Options.");
      return false;
     }
   
   g_filepath = TerminalInfoString(TERMINAL_DATA_PATH) +
                "\\MQL5\\Files\\" + InpOutputFile;
   
   // Move to end if appending
   if(InpAppendMode) FileSeek(g_file, 0, SEEK_END);
   
   // Write CSV header
   if(InpWriteHeader && !InpAppendMode)
     {
      FileWrite(g_file, "Tick_Time_ms", "Bid", "Ask", "Flags");
     }
   
   Print("[TitanLogger] File opened: ", g_filepath);
   return true;
  }

//+------------------------------------------------------------------+
//| Write one tick row                                               |
//+------------------------------------------------------------------+
void WriteTick(long time_msc, double bid, double ask, uint mt5_flags)
  {
   // Quality filters
   if(InpFilterZeroSpread && ask <= bid)
     { g_skip_count++; return; }
   if(InpFilterNegative && bid <= 0.0)
     { g_skip_count++; return; }
   
   int titan_flag = TitanFlag(mt5_flags);
   
   FileWrite(g_file,
             (string)time_msc,
             DoubleToString(bid, _Digits),
             DoubleToString(ask, _Digits),
             (string)titan_flag);
   
   g_tick_count++;
   
   // Periodic flush (prevents data loss on crash)
   if(g_tick_count % InpFlushEveryN == 0)
      FileFlush(g_file);
   
   // Status update
   if(g_tick_count % InpStatusEveryN == 0)
      Print("[TitanLogger] Logged: ", g_tick_count, " ticks | ",
            "Skipped: ", g_skip_count, " | ",
            "Last: Bid=", DoubleToString(bid,_Digits),
            " Ask=", DoubleToString(ask,_Digits),
            " Flag=", titan_flag);
  }

//+------------------------------------------------------------------+
//| Export historical tick data at startup                           |
//+------------------------------------------------------------------+
void ExportHistory()
  {
   if(!InpExportHistory || InpHistoricalDays <= 0) return;
   
   datetime end_dt   = (InpHistEnd   > 0) ? InpHistEnd   : TimeCurrent();
   datetime start_dt = (InpHistStart > 0) ? InpHistStart : 
                        end_dt - (datetime)(InpHistoricalDays * 86400);
   
   Print("[TitanLogger] Exporting history: ",
         TimeToString(start_dt), " → ", TimeToString(end_dt),
         " (", InpHistoricalDays, " days)");
   
   // MT5 stores history in chunks — loop through 7-day chunks to avoid OOM
   int   chunk_days = 7;
   long  hist_count = 0;
   datetime cur = start_dt;
   
   while(cur < end_dt)
     {
      datetime chunk_end = MathMin((datetime)(cur + chunk_days * 86400), end_dt);
      
      MqlTick ticks[];
      int n = CopyTicksRange(_Symbol, ticks, COPY_TICKS_ALL,
                             (long)cur * 1000,        // from: Unix ms
                             (long)chunk_end * 1000); // to:   Unix ms
      
      if(n <= 0)
        {
         Print("[TitanLogger] No ticks for chunk ", 
               TimeToString(cur), " → ", TimeToString(chunk_end));
         cur = chunk_end;
         continue;
        }
      
      Print("[TitanLogger] Chunk ", TimeToString(cur), ": ", n, " ticks");
      
      for(int i = 0; i < n; i++)
        {
         WriteTick(ticks[i].time_msc,
                   ticks[i].bid,
                   ticks[i].ask,
                   ticks[i].flags);
        }
      
      hist_count += n;
      cur = chunk_end;
      
      // Flush after each chunk
      FileFlush(g_file);
      Print("[TitanLogger] Chunk complete. Running total: ", hist_count, " ticks");
     }
   
   // Final flush
   FileFlush(g_file);
   g_hist_done = true;
   Print("[TitanLogger] History export COMPLETE: ", hist_count, " ticks written.");
   Print("[TitanLogger] File location: ", g_filepath);
  }

//+------------------------------------------------------------------+
//| OnInit                                                           |
//+------------------------------------------------------------------+
int OnInit()
  {
   Print("===========================================");
   Print("[TitanLogger] Titan V3.0 Data Logger EA");
   Print("[TitanLogger] Symbol=", _Symbol, " Digits=", _Digits);
   Print("===========================================");
   
   // Open the output file
   if(!OpenFile()) return INIT_FAILED;
   
   // Export historical data first
   ExportHistory();
   
   if(!InpLogLive)
     {
      Print("[TitanLogger] InpLogLive=false — history exported, EA will now idle.");
      Print("[TitanLogger] You can detach the EA.");
     }
   else
     {
      Print("[TitanLogger] Now logging live ticks. Keep EA attached.");
     }
   
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| OnDeinit                                                         |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(g_file != INVALID_HANDLE)
     {
      FileFlush(g_file);
      FileClose(g_file);
      g_file = INVALID_HANDLE;
     }
   Print("[TitanLogger] Stopped. Total ticks logged: ", g_tick_count,
         " | Skipped: ", g_skip_count);
   Print("[TitanLogger] File: ", g_filepath);
   Print("[TitanLogger] Python import command:");
   Print("[TitanLogger]   from titan_data import ParquetConverter");
   Print("[TitanLogger]   ParquetConverter().convert('", g_filepath, "')");
  }

//+------------------------------------------------------------------+
//| OnTick — fires on every new price quote                          |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(!InpLogLive || InpLogOnBook) return;
   if(g_file == INVALID_HANDLE)   return;
   
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;
   
   WriteTick(tick.time_msc, tick.bid, tick.ask, tick.flags);
  }

//+------------------------------------------------------------------+
//| OnBookEvent — fires on DOM changes (sub-millisecond)             |
//| Use InpLogOnBook=true for maximum tick resolution                |
//+------------------------------------------------------------------+
void OnBookEvent(const string &symbol)
  {
   if(!InpLogLive || !InpLogOnBook) return;
   if(symbol != _Symbol)           return;
   if(g_file == INVALID_HANDLE)    return;
   
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;
   
   WriteTick(tick.time_msc, tick.bid, tick.ask, tick.flags);
  }

//+------------------------------------------------------------------+
//| OnTimer — periodic status (optional)                             |
//+------------------------------------------------------------------+
void OnTimer()
  {
   if(g_file != INVALID_HANDLE)
      FileFlush(g_file);
  }
