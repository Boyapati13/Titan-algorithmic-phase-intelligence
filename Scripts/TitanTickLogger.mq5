#property copyright "Titan HFT System"
#property link      "https://titan-hft.com"
#property version   "1.00"
#property description "MQL5 Tick Logger for Phase 1 Data Engineering"
#property script_show_inputs

// Binary struct: 52 bytes (with padding for alignment)
struct TickData {
    long  time_msc;  // 8 bytes
    double bid;      // 8 bytes
    double ask;      // 8 bytes
    double last;     // 8 bytes
    double volume;   // 8 bytes
    uint   flags;    // 4 bytes
    uint   padding;  // 4 bytes
    uint   padding2; // 4 bytes, total 52 bytes
};

// Inputs
input int   BUFFER_SIZE         = 10000;     // Circular buffer size
input int   FLUSH_INTERVAL_MS   = 5000;      // Flush every 5 seconds
input double ZSCORE_THRESHOLD   = 4.0;       // Z-score threshold for bad ticks
input double SPREAD_MULTIPLIER  = 5.0;       // Spread multiplier for rejection
input int   ROLLING_WINDOW      = 500;       // Rolling window for stats

// Globals
TickData buffer[];
int      buffer_index = 0;
bool     buffer_full = false;
uint     last_flush_time = 0;
string   current_file = "";
int      file_handle = INVALID_HANDLE;
datetime current_day = 0;
string   rejection_log = "TickRejection.log";
int      rejection_handle = INVALID_HANDLE;

// Rolling vectors for filter
vector<double> midpoints(ROLLING_WINDOW);
vector<double> spreads(ROLLING_WINDOW);
int midpoint_index = 0;
int spread_index = 0;
bool midpoints_full = false;
bool spreads_full = false;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
    // Open rejection log
    rejection_handle = FileOpen(rejection_log, FILE_WRITE | FILE_TXT | FILE_ANSI);
    if (rejection_handle == INVALID_HANDLE) {
        Print("Failed to open rejection log: ", GetLastError());
        return INIT_FAILED;
    }

    // Resize buffer
    ArrayResize(buffer, BUFFER_SIZE);

    // Set timer for periodic flush
    EventSetMillisecondTimer(1000);  // Check every 1s

    Print("Titan Tick Logger initialized. Buffer size: ", BUFFER_SIZE);
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    // Final flush
    FlushBuffer();

    // Close files
    if (file_handle != INVALID_HANDLE) {
        FileClose(file_handle);
    }
    if (rejection_handle != INVALID_HANDLE) {
        FileClose(rejection_handle);
    }

    EventKillTimer();
    Print("Titan Tick Logger deinitialized.");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
    MqlTick tick;
    if (!SymbolInfoTick(_Symbol, tick)) {
        return;
    }

    double mid = (tick.bid + tick.ask) / 2.0;
    double spr = tick.ask - tick.bid;

    // Update rolling midpoints
    midpoints[midpoint_index] = mid;
    midpoint_index = (midpoint_index + 1) % ROLLING_WINDOW;
    if (midpoint_index == 0) midpoints_full = true;

    // Update rolling spreads
    spreads[spread_index] = spr;
    spread_index = (spread_index + 1) % ROLLING_WINDOW;
    if (spread_index == 0) spreads_full = true;

    // Bad tick filter
    bool bad_tick = false;

    // Z-score on midpoints
    if (midpoints_full) {
        double mean = midpoints.Mean();
        double std = midpoints.Std();
        if (std > 0) {
            double z = MathAbs((mid - mean) / std);
            if (z > ZSCORE_THRESHOLD) {
                bad_tick = true;
            }
        }
    }

    // Spread outlier check
    if (spreads_full) {
        vector<double> sorted_spreads = spreads;
        sorted_spreads.Sort(SORT_ASCENDING);
        double median_spread = sorted_spreads[ROLLING_WINDOW / 2];
        if (spr > median_spread * SPREAD_MULTIPLIER) {
            bad_tick = true;
        }
    }

    if (bad_tick) {
        // Log bad tick
        string msg = StringFormat("BAD_TICK %s %lld %.5f %.5f %.5f %u\n",
            TimeToString(tick.time, TIME_DATE | TIME_SECONDS),
            tick.time_msc, tick.bid, tick.ask, spr, tick.flags);
        FileWriteString(rejection_handle, msg);
        FileFlush(rejection_handle);
        return;
    }

    // Check for day change (file rotation)
    MqlDateTime dt;
    TimeToStruct(tick.time, dt);
    dt.hour = 0;
    dt.min = 0;
    dt.sec = 0;
    datetime tick_day = StructToTime(dt);
    if (tick_day != current_day) {
        // Close current file
        if (file_handle != INVALID_HANDLE) {
            FileClose(file_handle);
            file_handle = INVALID_HANDLE;
        }

        // Open new file
        string symbol = _Symbol;
        StringReplace(symbol, "/", "");  // Remove slashes if any
        string date_str = TimeToString(tick_day, TIME_DATE);
        StringReplace(date_str, ".", "");
        current_file = symbol + "_" + date_str + ".ticks";

        file_handle = FileOpen(current_file, FILE_WRITE | FILE_BIN);
        if (file_handle == INVALID_HANDLE) {
            Print("Failed to open tick file: ", current_file, " Error: ", GetLastError());
            return;
        }

        current_day = tick_day;
        Print("Opened new tick file: ", current_file);
    }

    // Add good tick to buffer
    buffer[buffer_index].time_msc = tick.time_msc;
    buffer[buffer_index].bid = tick.bid;
    buffer[buffer_index].ask = tick.ask;
    buffer[buffer_index].last = tick.last;
    buffer[buffer_index].volume = tick.volume;
    buffer[buffer_index].flags = tick.flags;
    buffer[buffer_index].padding = 0;  // Padding
    buffer[buffer_index].padding2 = 0; // Padding

    buffer_index++;
    if (buffer_index >= BUFFER_SIZE) {
        buffer_index = 0;
        buffer_full = true;
    }
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer() {
    uint now = GetTickCount();
    if (now - last_flush_time >= FLUSH_INTERVAL_MS) {
        FlushBuffer();
        last_flush_time = now;
    }
}

//+------------------------------------------------------------------+
//| Flush buffer to disk                                             |
//+------------------------------------------------------------------+
void FlushBuffer() {
    if (file_handle == INVALID_HANDLE) return;

    int count = buffer_full ? BUFFER_SIZE : buffer_index;
    if (count == 0) return;

    int start = buffer_full ? buffer_index : 0;

    // Write structs to file in correct order
    for (int i = 0; i < count; i++) {
        int idx = (start + i) % BUFFER_SIZE;
        FileWriteStruct(file_handle, buffer[idx]);
    }

    FileFlush(file_handle);

    // Reset buffer
    buffer_index = 0;
    buffer_full = false;

    Print("Flushed ", count, " ticks to ", current_file);
}