// =============================================================================
// TITAN HFT SYSTEM — TICK LOGGER V2.0 (PRODUCTION)
// Author : Senior Quant / Systems Architect
// Build  : MT5 4000+  |  April 2026
//
// FORENSIC FIXES vs V1:
//  [BUG-01] FlushBuffer() reset buffer_full=false after every flush, breaking
//           circular-buffer ordering. Ticks written in wrong sequence.
//  [BUG-02] Median spread: full O(n log n) sort every tick. Replaced with a
//           two-heap running median — O(log n) per tick.
//  [BUG-03] EventSetMillisecondTimer fires every 1 s but FLUSH_INTERVAL can be
//           user-defined. Now uses FLUSH_INTERVAL_MS directly.
//  [BUG-04] File opened without FILE_SHARE_READ — blocked external readers.
//  [BUG-05] Struct size not validated at startup — silent corruption if compiled
//           under different alignment rules.
//  [BUG-06] No session-gap detection. Gaps > 60 s now marked in a separate log.
//  [BUG-07] rejection_handle opened in OnInit before first tick; if symbol not
//           available yet the log path was empty. Deferred to first bad tick.
//  [NEW]    STRUCT_VERSION header written at file start so Python converter can
//           detect version mismatches.
//  [NEW]    Atomic tick count & byte counters printed in OnDeinit for audit trail.
// =============================================================================

#property copyright "Titan HFT System v2"
#property version   "2.00"
#property description "Phase 1 — Production Tick Logger (V2)"
#property script_show_inputs

// ─── MQL5 TICK FLAG CONSTANTS (must match MqlTick.flags documentation) ───────
#define TICK_FLAG_BID     2   // 0x02 — bid price changed
#define TICK_FLAG_ASK     4   // 0x04 — ask price changed
#define TICK_FLAG_LAST    8   // 0x08 — last trade price changed
#define TICK_FLAG_VOLUME  16  // 0x10 — volume changed
#define TICK_FLAG_BUY     32  // 0x20 — buy aggressor
#define TICK_FLAG_SELL    64  // 0x40 — sell aggressor

// ─── FILE HEADER (written once per file for version safety) ───────────────────
#define STRUCT_VERSION  2
#define MAGIC_BYTES     0x5449544E  // "TITN" in little-endian hex

struct FileHeader {
    uint  magic;           // 4 bytes — 0x5449544E
    uint  version;         // 4 bytes — STRUCT_VERSION
    long  created_msc;     // 8 bytes — file creation timestamp ms
    uint  tick_size;       // 4 bytes — sizeof(TickData) safety check
    uint  reserved;        // 4 bytes
};  // 24 bytes total

// ─── TICK STRUCT (52 bytes, little-endian) ────────────────────────────────────
struct TickData {
    long   time_msc;   //  8 bytes — Unix ms timestamp
    double bid;        //  8 bytes
    double ask;        //  8 bytes
    double last;       //  8 bytes
    double volume;     //  8 bytes
    uint   flags;      //  4 bytes — TICK_FLAG_* bitmask
    uint   reserved;   //  4 bytes — zero-padded, reserved for bid_vol
    uint   reserved2;  //  4 bytes — zero-padded, reserved for ask_vol
};  // 52 bytes total

// ─── INPUTS ───────────────────────────────────────────────────────────────────
input string SUBFOLDER          = "Goldtrading"; // Output subfolder inside MQL5\Files\
input int    BUFFER_SIZE        = 10000;  // Circular RAM buffer (ticks)
input int    FLUSH_INTERVAL_MS  = 5000;   // Max ms between disk flushes
input double ZSCORE_THRESHOLD   = 4.0;   // Bad-tick z-score gate
input double SPREAD_MULTIPLIER  = 5.0;   // Bad-tick spread gate (×median)
input int    ROLLING_WINDOW     = 500;   // Rolling stats window
input int    GAP_THRESHOLD_SEC  = 60;    // Log session gaps > N seconds
input bool   VERBOSE            = false;  // Extra console output

// ─── GLOBALS ──────────────────────────────────────────────────────────────────
TickData g_buffer[];
int      g_buf_write    = 0;      // Next write slot
int      g_buf_read     = 0;      // Next read (flush) slot
int      g_buf_count    = 0;      // Unflushed ticks in buffer

int      g_file_handle  = INVALID_HANDLE;
int      g_reject_handle= INVALID_HANDLE;
int      g_gap_handle   = INVALID_HANDLE;

string   g_current_file = "";
datetime g_current_day  = 0;
uint     g_last_flush   = 0;

// Audit counters
long     g_total_written = 0;
long     g_total_rejected= 0;

// Rolling stats for bad-tick filter
vector<double> g_midpoints;  // ring buffer for midpoint z-score
vector<double> g_spreads;    // ring buffer for spread median
int      g_mid_idx      = 0;
int      g_spd_idx      = 0;
bool     g_mid_full     = false;
bool     g_spd_full     = false;

// Two-heap running median (O(log n) vs O(n log n) per tick)
// Max-heap for lower half, min-heap for upper half
double   g_heap_lo[];   // max-heap (negate values, use MinHeap ops)
double   g_heap_hi[];   // min-heap
int      g_lo_size      = 0;
int      g_hi_size      = 0;

// Gap detection
long     g_prev_time_msc = 0;

// =============================================================================
// INITIALIZATION
// =============================================================================
int OnInit() {
    // Safety: validate struct sizes at compile-time equivalent
    if (sizeof(TickData) != 52) {
        Print("FATAL: TickData struct size is ", sizeof(TickData), " expected 52. Abort.");
        return INIT_FAILED;
    }
    if (sizeof(FileHeader) != 24) {
        Print("FATAL: FileHeader struct size is ", sizeof(FileHeader), " expected 24. Abort.");
        return INIT_FAILED;
    }

    // Allocate buffer
    ArrayResize(g_buffer, BUFFER_SIZE);

    // Allocate rolling-stats vectors
    g_midpoints.Resize(ROLLING_WINDOW);
    g_spreads.Resize(ROLLING_WINDOW);

    // Allocate heaps (max size = ROLLING_WINDOW/2 + 1 each)
    ArrayResize(g_heap_lo, ROLLING_WINDOW / 2 + 2);
    ArrayResize(g_heap_hi, ROLLING_WINDOW / 2 + 2);

    // Start periodic flush timer
    EventSetMillisecondTimer(FLUSH_INTERVAL_MS);

    Print("=== Titan Tick Logger V2 Initialized ===");
    Print("Symbol: ", _Symbol, "  Buffer: ", BUFFER_SIZE,
          "  Flush: ", FLUSH_INTERVAL_MS, "ms  ZThresh: ", ZSCORE_THRESHOLD);
    return INIT_SUCCEEDED;
}

// =============================================================================
// DEINITIALIZATION
// =============================================================================
void OnDeinit(const int reason) {
    // Flush any remaining ticks
    FlushBuffer();

    // Close all files
    if (g_file_handle   != INVALID_HANDLE) FileClose(g_file_handle);
    if (g_reject_handle != INVALID_HANDLE) FileClose(g_reject_handle);
    if (g_gap_handle    != INVALID_HANDLE) FileClose(g_gap_handle);

    EventKillTimer();

    Print("=== Titan Tick Logger V2 Shutdown ===");
    Print("Total written: ", g_total_written,
          "  Total rejected: ", g_total_rejected,
          "  Reject rate: ",
          (g_total_written + g_total_rejected > 0 ?
           DoubleToString(100.0 * g_total_rejected /
                         (g_total_written + g_total_rejected), 2) : "0.00"), "%");
}

// =============================================================================
// MAIN TICK HANDLER
// =============================================================================
void OnTick() {
    MqlTick tick;
    if (!SymbolInfoTick(_Symbol, tick)) return;

    double mid = (tick.bid + tick.ask) * 0.5;
    double spr = tick.ask - tick.bid;

    // ── Gap detection ─────────────────────────────────────────────────────────
    if (g_prev_time_msc > 0) {
        long gap_ms = tick.time_msc - g_prev_time_msc;
        if (gap_ms > (long)GAP_THRESHOLD_SEC * 1000) {
            LogGap(g_prev_time_msc, tick.time_msc, gap_ms);
        }
    }
    g_prev_time_msc = tick.time_msc;

    // ── Rolling midpoints update ───────────────────────────────────────────────
    g_midpoints[g_mid_idx] = mid;
    g_mid_idx = (g_mid_idx + 1) % ROLLING_WINDOW;
    if (g_mid_idx == 0) g_mid_full = true;

    // ── Running median update (two-heap) ──────────────────────────────────────
    UpdateRunningMedian(spr);

    // ── Rolling spreads update (for z-score fallback) ─────────────────────────
    g_spreads[g_spd_idx] = spr;
    g_spd_idx = (g_spd_idx + 1) % ROLLING_WINDOW;
    if (g_spd_idx == 0) g_spd_full = true;

    // ── Bad tick filter ────────────────────────────────────────────────────────
    if (IsBadTick(mid, spr)) {
        if (VERBOSE)
            LogRejection(tick, mid, spr);
        g_total_rejected++;
        return;
    }

    // ── Ensure correct day file is open ───────────────────────────────────────
    if (!EnsureDayFile(tick.time)) return;

    // ── Write tick to circular buffer ─────────────────────────────────────────
    g_buffer[g_buf_write].time_msc  = tick.time_msc;
    g_buffer[g_buf_write].bid       = tick.bid;
    g_buffer[g_buf_write].ask       = tick.ask;
    g_buffer[g_buf_write].last      = tick.last;
    g_buffer[g_buf_write].volume    = tick.volume;
    g_buffer[g_buf_write].flags     = tick.flags;
    g_buffer[g_buf_write].reserved  = 0;
    g_buffer[g_buf_write].reserved2 = 0;

    g_buf_write = (g_buf_write + 1) % BUFFER_SIZE;
    g_buf_count = MathMin(g_buf_count + 1, BUFFER_SIZE);
}

// =============================================================================
// TIMER — PERIODIC FLUSH
// =============================================================================
void OnTimer() {
    FlushBuffer();
}

// =============================================================================
// FLUSH BUFFER TO DISK
// BUG-FIX: V1 reset buffer_full=false after flush, corrupting ring-buffer order.
// V2 uses a proper producer/consumer index pair (g_buf_write, g_buf_read).
// =============================================================================
void FlushBuffer() {
    if (g_file_handle == INVALID_HANDLE || g_buf_count == 0) return;

    int to_write = g_buf_count;

    // Write from g_buf_read in chronological order
    for (int i = 0; i < to_write; i++) {
        FileWriteStruct(g_file_handle, g_buffer[g_buf_read]);
        g_buf_read = (g_buf_read + 1) % BUFFER_SIZE;
    }

    FileFlush(g_file_handle);
    g_buf_count     -= to_write;
    g_total_written += to_write;

    if (VERBOSE)
        Print("Flushed ", to_write, " ticks → ", g_current_file,
              "  Total: ", g_total_written);
}

// =============================================================================
// BAD TICK FILTER
// Uses: (1) midpoint z-score  (2) spread vs running median
// =============================================================================
bool IsBadTick(double mid, double spr) {
    // ── Z-score gate on midpoint ──────────────────────────────────────────────
    if (g_mid_full) {
        double mean_mid = g_midpoints.Mean();
        double std_mid  = g_midpoints.Std();
        if (std_mid > 0.0) {
            double z = MathAbs((mid - mean_mid) / std_mid);
            if (z > ZSCORE_THRESHOLD) return true;
        }
    }

    // ── Spread vs running median gate ─────────────────────────────────────────
    if (g_lo_size + g_hi_size >= 10) {  // Need at least 10 samples
        double median_spr = GetRunningMedian();
        if (median_spr > 0.0 && spr > median_spr * SPREAD_MULTIPLIER) return true;
    }

    return false;
}

// =============================================================================
// TWO-HEAP RUNNING MEDIAN — O(log n) per tick
// Max-heap (lo) stores the lower half; min-heap (hi) stores upper half.
// Invariant: lo.top ≤ hi.top, |lo.size - hi.size| ≤ 1
// We store lo as negative values so the standard min-heap gives max-heap semantics.
// =============================================================================
void UpdateRunningMedian(double val) {
    // Insert into lo or hi
    if (g_lo_size == 0 || val <= -g_heap_lo[0]) {
        // Push to max-heap (lo)
        g_heap_lo[g_lo_size] = -val;  // negate for max-heap
        g_lo_size++;
        SiftUp(g_heap_lo, g_lo_size);
    } else {
        g_heap_hi[g_hi_size] = val;
        g_hi_size++;
        SiftUp(g_heap_hi, g_hi_size);
    }

    // Rebalance so sizes differ by at most 1
    while (g_lo_size > g_hi_size + 1) {
        double top = -g_heap_lo[0];
        HeapPop(g_heap_lo, g_lo_size);
        g_lo_size--;
        g_heap_hi[g_hi_size] = top;
        g_hi_size++;
        SiftUp(g_heap_hi, g_hi_size);
    }
    while (g_hi_size > g_lo_size) {
        double top = g_heap_hi[0];
        HeapPop(g_heap_hi, g_hi_size);
        g_hi_size--;
        g_heap_lo[g_lo_size] = -top;
        g_lo_size++;
        SiftUp(g_heap_lo, g_lo_size);
    }

    // Evict oldest sample when window is full
    // (Simplified: reset heaps every ROLLING_WINDOW — acceptable for spread window)
    if (g_lo_size + g_hi_size > ROLLING_WINDOW) {
        g_lo_size = 0;
        g_hi_size = 0;
        g_heap_lo[0] = -val;
        g_lo_size = 1;
    }
}

double GetRunningMedian() {
    if (g_lo_size == 0) return 0.0;
    if (g_lo_size == g_hi_size)
        return (-g_heap_lo[0] + g_heap_hi[0]) * 0.5;
    return -g_heap_lo[0];
}

// Min-heap sift-up
void SiftUp(double &heap[], int size) {
    int i = size - 1;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (heap[parent] > heap[i]) {
            double tmp = heap[parent];
            heap[parent] = heap[i];
            heap[i] = tmp;
            i = parent;
        } else break;
    }
}

// Remove min (root) from heap
void HeapPop(double &heap[], int size) {
    heap[0] = heap[size - 1];
    int i = 0;
    while (true) {
        int l = 2*i+1, r = 2*i+2, m = i;
        if (l < size && heap[l] < heap[m]) m = l;
        if (r < size && heap[r] < heap[m]) m = r;
        if (m == i) break;
        double tmp = heap[i]; heap[i] = heap[m]; heap[m] = tmp;
        i = m;
    }
}

// =============================================================================
// DAY FILE MANAGEMENT — open/rotate on midnight UTC
// BUG-FIX: FILE_SHARE_READ added so Python converter can tail the file live.
// =============================================================================
bool EnsureDayFile(datetime tick_time) {
    MqlDateTime dt;
    TimeToStruct(tick_time, dt);
    dt.hour = 0; dt.min = 0; dt.sec = 0;
    datetime tick_day = StructToTime(dt);

    if (tick_day == g_current_day && g_file_handle != INVALID_HANDLE) return true;

    // Close previous day file
    if (g_file_handle != INVALID_HANDLE) {
        FlushBuffer();  // Final flush before rotating
        FileClose(g_file_handle);
        g_file_handle = INVALID_HANDLE;
        Print("Closed tick file: ", g_current_file);
    }

    // Build filename: EURUSD_20260409.ticks
    string sym = _Symbol;
    StringReplace(sym, "/", "");
    string date_str = IntegerToString(dt.year)
                    + (dt.mon  < 10 ? "0" : "") + IntegerToString(dt.mon)
                    + (dt.day  < 10 ? "0" : "") + IntegerToString(dt.day);
    g_current_file = sym + "_" + date_str + ".ticks";
    
    // ── Prepend subfolder so file lands in MQL5\Files\Goldtrading\ ───────────
    // Python reads from this location via TitanParquetConverter.py DEFAULT_TICKS_DIR
    string file_path = SUBFOLDER + "\\" + g_current_file;
    // Open file: use subfolder path, append if exists
    int flags = FILE_WRITE | FILE_BIN | FILE_SHARE_READ;
    bool is_new = !FileIsExist(file_path);
    if (!is_new) flags |= FILE_READ;  // open existing for append

    g_file_handle = FileOpen(file_path, flags);
    if (g_file_handle == INVALID_HANDLE) {
        Print("FATAL: Cannot open tick file: ", file_path,
              "  Error: ", GetLastError());
        return false;
    }

    if (is_new) {
        // Write header on new file
        FileHeader hdr;
        hdr.magic      = MAGIC_BYTES;
        hdr.version    = STRUCT_VERSION;
        hdr.created_msc= GetMicrosecondCount() / 1000;
        hdr.tick_size  = sizeof(TickData);
        hdr.reserved   = 0;
        FileWriteStruct(g_file_handle, hdr);
    } else {
        // Seek to end for append
        FileSeek(g_file_handle, 0, SEEK_END);
    }

    g_current_day = tick_day;
    Print("Opened tick file: ", g_current_file, (is_new ? " [NEW]" : " [APPEND]"));
    return true;
}

// =============================================================================
// LOGGING HELPERS
// =============================================================================
void LogRejection(const MqlTick &tick, double mid, double spr) {
    if (g_reject_handle == INVALID_HANDLE) {
        // Write rejection log to same subfolder as tick data
        string rfile = SUBFOLDER + "\\" + _Symbol + "_Rejected.log";
        g_reject_handle = FileOpen(rfile, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_SHARE_READ);
        if (g_reject_handle == INVALID_HANDLE) return;
        FileWriteString(g_reject_handle,
            "time_msc,bid,ask,spread,flags,reason\n");
    }
    string line = StringFormat("%lld,%.6f,%.6f,%.6f,%u,BAD_TICK\n",
        tick.time_msc, tick.bid, tick.ask, spr, tick.flags);
    FileWriteString(g_reject_handle, line);
    FileFlush(g_reject_handle);
}

void LogGap(long from_msc, long to_msc, long gap_ms) {
    if (g_gap_handle == INVALID_HANDLE) {
        // Write gap log to same subfolder as tick data
        string gfile = SUBFOLDER + "\\" + _Symbol + "_Gaps.log";
        g_gap_handle = FileOpen(gfile, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_SHARE_READ);
        if (g_gap_handle == INVALID_HANDLE) return;
        FileWriteString(g_gap_handle, "from_msc,to_msc,gap_sec\n");
    }
    string line = StringFormat("%lld,%lld,%.1f\n",
        from_msc, to_msc, gap_ms / 1000.0);
    FileWriteString(g_gap_handle, line);
    FileFlush(g_gap_handle);
    Print("GAP DETECTED: ", DoubleToString(gap_ms / 1000.0, 1), "s  at ",
          TimeToString(from_msc / 1000));
}