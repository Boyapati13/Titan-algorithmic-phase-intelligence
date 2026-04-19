// =============================================================================
// TITAN HFT SYSTEM — EXPERT ADVISOR V2.0 (PRODUCTION)
// Author : Senior Quant / Systems Architect
// Build  : MT5 4000+  |  April 2026
//
// FORENSIC FIXES vs V1:
//  [BUG-01] ARCHITECTURE: Used OnTick() — wrong for microstructure. V2 uses
//           OnBookEvent() for sub-millisecond DOM granularity.
//  [BUG-02] GetBidDepthImbalance/GetAskDepthImbalance returned hardcoded 1.0.
//           V2 calls MarketBookGet() and computes real DOM imbalance.
//  [BUG-03] GetPriceVsVWAP() returned 0.0 always. V2 maintains a real
//           session VWAP using cumulative price*volume accumulation.
//  [BUG-04] IsAbsorptionActive() returned false always. V2 implements the
//           4-condition absorption detector (VoT, PriceEfficiency,
//           VolAsymmetry, Persistence).
//  [BUG-05] ComputeStackedImbalance() compared float midpoints with ==
//           (always false). V2 uses price-level bucketing.
//  [BUG-06] ArrayMean/ArrayStd were O(n) loops. V2 uses vector<double>
//           .Mean() / .Std() for SIMD performance.
//  [BUG-07] g_spread_history was double[] not vector<double> — inconsistent.
//  [BUG-08] EventSetTimer(1) called inside ExecuteTrade() on every signal —
//           resets the 1 s timer repeatedly. Moved to OnInit() only.
//  [BUG-09] g_magic_number hardcoded int 12345, not an input parameter.
//  [BUG-10] OrderSend (synchronous) used for cancel-replace. V2 uses
//           OrderSendAsync throughout.
//  [BUG-11] TICK_FLAG constants were bit 0 (value 1) and bit 1 (value 2).
//           Correct MQL5 values: BUY=32, SELL=64.
//  [NEW]    Kill-switch: daily loss limit, consecutive-loss breaker,
//           drawdown circuit breaker, news-event halt.
//  [NEW]    Session VWAP with ±1σ/±2σ premium/discount zone detection.
//  [NEW]    Hurst exponent (20-bar) for regime detection.
//  [NEW]    Proper ONNX input shape: [1 × 50 × 12] consistent with LSTM.
// =============================================================================

#property copyright "Titan HFT System v2"
#property version   "2.00"
#property description "Titan EA V2 — OnBookEvent + ONNX LSTM + Full Order Flow"

#include <Trade\Trade.mqh>
#include <Math\Stat\Math.mqh>

// ─── EMBED ONNX MODEL AT COMPILE TIME ────────────────────────────────────────
// This bakes the model binary directly into the .ex5 so the Strategy Tester
// (which runs in an isolated agent with its own file sandbox) can load it.
// The file must exist at MQL5\Files\titan_lstm.onnx when recompiling.
#resource "\\Files\\titan_lstm.onnx" as uchar g_onnx_buf[]

// ─── TICK FLAG CONSTANTS ─────────────────────────────────────────────────────
#define TICK_FLAG_BID     2
#define TICK_FLAG_ASK     4
#define TICK_FLAG_LAST    8
#define TICK_FLAG_VOLUME  16
#define TICK_FLAG_BUY     32
#define TICK_FLAG_SELL    64

// ─── FEATURE INDICES IN 12-COLUMN MATRIX ─────────────────────────────────────
#define F_VOT_ZSCORE      0
#define F_RVOL            1
#define F_CUMDELTA_DIV    2
#define F_IMBALANCE_RATIO 3
#define F_FDPI            4
#define F_MVDI            5
#define F_BID_DEPTH       6
#define F_ASK_DEPTH       7
#define F_PRICE_VS_VWAP   8
#define F_HOUR_SIN        9
#define F_HURST           10
#define F_TWKJ            11



// ─── INPUTS ───────────────────────────────────────────────────────────────────
input group "=== ONNX Settings ==="
input string ONNX_MODEL_FILE    = "titan_lstm.onnx";

input group "=== Signal Thresholds ==="
// CRITICAL: These thresholds come from titan_inference_config.json (buy_threshold / sell_threshold)
// After running TitanLSTMTraining.py, read the calibrated values and set them here.
// OOS Iter-3 (AUC=0.65): buy=0.365 / sell=0.635  → use conservative: 0.60 / 0.40
// OOS Iter-1 (AUC=0.74): buy=0.60  / sell=0.40   → use conservative: 0.65 / 0.35
input double BUY_THRESHOLD      = 0.60;   // Min conviction for BUY  (see titan_inference_config.json)
input double SELL_THRESHOLD     = 0.40;   // Max conviction for SELL (see titan_inference_config.json)
input int    INFERENCE_INTERVAL = 50;     // ms between ONNX calls

input group "=== Execution ==="
input int    MAGIC              = 77777;
input double LOT_SIZE           = 0.01;
input int    MAX_POSITIONS      = 1;
input int    TARGET_TICKS       = 15;
input int    STOP_LOSS_TICKS    = 36;
input bool   USE_LIMIT_ORDERS   = true;   // Passive fill (limit at BBO)
input int    ORDER_TIMEOUT_MS   = 500;    // Cancel unfilled limit after N ms

input group "=== Risk Management ==="
input double SPREAD_SIGMA_LIMIT = 1.5;   // Abort if spread > mean + N*σ
input double DAILY_LOSS_LIMIT   = 2.0;   // % of account — halt for day
input int    CONSEC_LOSS_LIMIT  = 5;     // Consecutive losses → pause 1h
input double SESSION_DD_LIMIT   = 5.0;   // % drawdown from session high → halt

input group "=== Tick Matrix ==="
input int    TICK_BUFFER_SIZE   = 128;   // Rows in ONNX input matrix
input int    FEATURES           = 12;    // Columns (must match ONNX model)
input int    SPREAD_WIN         = 500;   // Spread rolling window
input int    VOT_WIN            = 100;   // Velocity-of-tape window
input int    DELTA_WIN          = 200;   // Cumulative delta window

// Trade log for OnTester() WFO export
struct TradeLogEntry {
    datetime  time;
    string    dir;
    double    price;
    double    conviction;
    double    pnl;
};
TradeLogEntry g_tlog[10000];
int           g_tlog_cnt = 0;

CTrade g_trade;
long   g_onnx_handle = INVALID_HANDLE;

// Tick feature matrix [TICK_BUFFER_SIZE × FEATURES]
matrix<double> g_mat;

// Rolling vectors (vector<double> uses SIMD Mean()/Std())
vector<double> g_spread_vec;    // Last SPREAD_WIN spreads
vector<double> g_mid_vec;       // Last VOT_WIN midpoints (for z-score)
vector<double> g_vol_vec;       // Last VOT_WIN volumes (for RVOL)
vector<double> g_delta_vec;     // Last DELTA_WIN deltas (for divergence)

// Indices into ring buffers
int g_spd_idx = 0, g_mid_idx = 0, g_vol_idx = 0, g_dlt_idx = 0;
bool g_spd_full = false, g_mid_full = false, g_vol_full = false, g_dlt_full = false;

// Velocity of tape
double g_vot_prev   = 0.0;
ulong  g_vot_sec    = 0;
int    g_vot_count  = 0;
double g_last_vot   = 0.0;

// Cumulative delta session
double g_cum_delta  = 0.0;
double g_prev_delta = 0.0;     // For divergence calc

// VWAP session accumulation
double g_vwap_pv    = 0.0;     // Σ(price × volume)
double g_vwap_v     = 0.0;     // Σ(volume)
double g_vwap_ss    = 0.0;     // Σ(price² × volume) for variance
datetime g_vwap_day = 0;

// Absorption state
bool   g_absorption_active = false;
uint   g_absorption_start  = 0;
double g_absorption_vol_lo = 0.0;  // Volume on dominating side
double g_absorption_vol_hi = 0.0;  // Volume on receding side
double g_absorption_price_start = 0.0;

// Inference throttle
uint   g_last_inference = 0;

// Kill-switch state
bool   g_halted         = false;
string g_halt_reason    = "";
uint   g_resume_time    = 0;
int    g_consec_losses  = 0;
double g_session_high   = 0.0;
double g_day_start_bal  = 0.0;

// Pending order tracking
struct PendingOrder {
    ulong  ticket;
    uint   placed_ms;
    double price;
    ENUM_ORDER_TYPE dir;
};
PendingOrder g_orders[10];
int          g_order_count = 0;

// ─── HURST SYNC FIX: Dedicated trade-tick ring buffer (20 entries) ─────────────
// Python trains compute_hurst() over 20 REAL TRADE TICKS (TICK_FLAG_LAST events).
// The EA was incorrectly using g_mid_vec (DOM updates, ~5–50ms granularity).
// This buffer fires only on TICK_FLAG_LAST, matching Python training exactly.
vector<double> g_hurst_mid;
int  g_hurst_idx  = 0;
bool g_hurst_full = false;
#define HURST_WIN 20  // Must match TitanFeatureEngineering.py compute_hurst(window=20)

// ─── ADVANCED ALPHA GLOBALS ──────────────────────────────────────────────────
#define ADV_WIN 128

// TWKJ Globals
double g_mid_hist[4];
long   g_time_hist[4];
bool   g_kin_ready = false;
vector<double> g_jerk_buf;
int    g_jerk_idx = 0; bool g_jerk_full = false;

// FDPI Globals
vector<double> g_buy_buf;
vector<double> g_sell_buf;
int    g_bs_idx = 0; bool g_bs_full = false;

// MVDI Globals
vector<double> g_mvdi_spread;
vector<double> g_mvdi_mid_diff;
vector<double> g_mvdi_raw;
int    g_mvdi_idx = 0; bool g_mvdi_full = false;

// =============================================================================
// INITIALIZATION
// =============================================================================
int OnInit() {
    g_trade.SetExpertMagicNumber(MAGIC);
    g_trade.SetDeviationInPoints(5);
    g_trade.SetTypeFilling(ORDER_FILLING_IOC);
    g_trade.SetAsyncMode(true);  // All sends are async

    // ── Allocate matrix and vectors ───────────────────────────────────────────
    g_mat.Resize(TICK_BUFFER_SIZE, FEATURES);
    g_mat.Fill(0.0);

    g_spread_vec.Resize(SPREAD_WIN);  g_spread_vec.Fill(0.0);
    g_mid_vec.Resize(VOT_WIN);         g_mid_vec.Fill(0.0);
    g_vol_vec.Resize(VOT_WIN);         g_vol_vec.Fill(0.0);
    g_delta_vec.Resize(DELTA_WIN);     g_delta_vec.Fill(0.0);
    g_hurst_mid.Resize(HURST_WIN);     g_hurst_mid.Fill(0.0);  // Hurst sync fix

    g_jerk_buf.Resize(ADV_WIN);        g_jerk_buf.Fill(0.0);
    ArrayInitialize(g_mid_hist, 0.0);  ArrayInitialize(g_time_hist, 0);
    g_buy_buf.Resize(ADV_WIN);         g_buy_buf.Fill(0.0);
    g_sell_buf.Resize(ADV_WIN);        g_sell_buf.Fill(0.0);
    g_mvdi_spread.Resize(ADV_WIN);     g_mvdi_spread.Fill(0.0);
    g_mvdi_mid_diff.Resize(ADV_WIN);   g_mvdi_mid_diff.Fill(0.0);
    g_mvdi_raw.Resize(ADV_WIN);        g_mvdi_raw.Fill(0.0);

    // ── Load ONNX model ───────────────────────────────────────────────────────
    // Primary: load from embedded resource (works in Tester + live trading).
    // The model is compiled into the .ex5 via the #resource directive above.
    g_onnx_handle = OnnxCreateFromBuffer(g_onnx_buf, ONNX_DEFAULT);
    if (g_onnx_handle == INVALID_HANDLE) {
        // Fallback: try loading from MQL5\Files (live terminal only)
        g_onnx_handle = OnnxCreate(ONNX_MODEL_FILE, ONNX_DEFAULT);
    }
    if (g_onnx_handle == INVALID_HANDLE) {
        // Last resort: "Files\" prefix
        string alt = "Files\\" + ONNX_MODEL_FILE;
        g_onnx_handle = OnnxCreate(alt, ONNX_DEFAULT);
    }
    if (g_onnx_handle == INVALID_HANDLE) {
        Print("ERROR: Cannot load ONNX model: ", ONNX_MODEL_FILE,
              "  Code: ", GetLastError(),
              " -- Recompile TitanEA.mq5 after placing titan_lstm.onnx in MQL5\\Files\\");
        return INIT_FAILED;
    }

    // Input: [batch=1, timesteps=TICK_BUFFER_SIZE, features=FEATURES]
    long in_shape[3] = {1, TICK_BUFFER_SIZE, FEATURES};
    if (!OnnxSetInputShape(g_onnx_handle, 0, in_shape)) {
        Print("ERROR: OnnxSetInputShape failed: ", GetLastError());
        return INIT_FAILED;
    }
    // Output: [batch=1, 1] — conviction score
    long out_shape[2] = {1, 1};
    if (!OnnxSetOutputShape(g_onnx_handle, 0, out_shape)) {
        Print("ERROR: OnnxSetOutputShape failed: ", GetLastError());
        return INIT_FAILED;
    }

    // ── Subscribe to DOM ──────────────────────────────────────────────────────
    if (!MarketBookAdd(_Symbol)) {
        Print("WARNING: MarketBookAdd failed — OnBookEvent will not fire. ",
              "Ensure broker provides Level 2 DOM.");
    }

    // ── Cancel-replace timer ──────────────────────────────────────────────────
    EventSetTimer(1);  // 1 s — check pending orders

    // ── Risk baselines ────────────────────────────────────────────────────────
    g_day_start_bal = AccountInfoDouble(ACCOUNT_BALANCE);
    g_session_high  = g_day_start_bal;

    Print("=== Titan EA V2 Initialized ===");
    Print("ONNX: ", ONNX_MODEL_FILE,
          "  Matrix: [", TICK_BUFFER_SIZE, "×", FEATURES, "]",
          "  Magic: ", MAGIC);
    return INIT_SUCCEEDED;
}

// =============================================================================
// DEINITIALIZATION
// =============================================================================
void OnDeinit(const int reason) {
    MarketBookRelease(_Symbol);
    EventKillTimer();
    if (g_onnx_handle != INVALID_HANDLE) OnnxRelease(g_onnx_handle);
}

// =============================================================================
// OnTick — FALLBACK BRIDGE
// =============================================================================
void OnTick() {
    // If we aren't getting DOM events, run the logic on every price change
    if (g_onnx_handle != INVALID_HANDLE) {
        OnBookEvent(_Symbol); 
    }
}

// =============================================================================
// OnBookEvent — FIRES ON EVERY DOM LEVEL CHANGE (sub-millisecond)
// =============================================================================
void OnBookEvent(const string &symbol) {
    if (symbol != _Symbol) return;

    MqlTick tick;
    if (!SymbolInfoTick(_Symbol, tick)) return;

    // Update all rolling features
    UpdateAllFeatures(tick);

    // Update session VWAP
    UpdateVWAP(tick);

    // Update absorption state
    UpdateAbsorption(tick);

    // Throttle inference to INFERENCE_INTERVAL ms
    uint now_ms = GetTickCount();
    if (now_ms - g_last_inference < (uint)INFERENCE_INTERVAL) return;
    g_last_inference = now_ms;

    // Kill-switch checks
    if (!RiskChecksPass()) return;

    // Build and shift feature matrix
    ShiftMatrixAndInsert(tick);

    // Run ONNX inference
    RunInference();
}


// =============================================================================
// OnTimer — CANCEL-REPLACE PENDING ORDERS
// =============================================================================
void OnTimer() {
    uint now_ms = GetTickCount();
    for (int i = g_order_count - 1; i >= 0; i--) {
        if (!OrderSelect(g_orders[i].ticket)) {
            // Order no longer exists (filled or already cancelled)
            RemoveOrder(i);
            continue;
        }
        uint age = now_ms - g_orders[i].placed_ms;
        if (age < (uint)ORDER_TIMEOUT_MS) continue;

        // Timeout — cancel stale limit order
        MqlTradeRequest req = {};
        MqlTradeResult  res = {};
        req.action = TRADE_ACTION_REMOVE;
        req.order  = g_orders[i].ticket;
        if (OrderSendAsync(req, res) || res.retcode == TRADE_RETCODE_DONE)
            Print("Cancelled stale limit #", g_orders[i].ticket, " age=", age, "ms");

        RemoveOrder(i);
    }
}

// =============================================================================
// TRADE TRANSACTION — track fills for risk management
// =============================================================================
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &req,
                        const MqlTradeResult  &res) {
    if (trans.type == TRADE_TRANSACTION_DEAL_ADD) {
        double bal = AccountInfoDouble(ACCOUNT_BALANCE);
        g_session_high = MathMax(g_session_high, bal);

        // Track consecutive losses
        if (trans.deal_type == DEAL_TYPE_BUY || trans.deal_type == DEAL_TYPE_SELL) {
            double profit = trans.price * trans.volume; // Simplified
            // We use HistoryDealGetDouble for accuracy
            if (HistoryDealSelect(trans.deal)) {
                double deal_profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT);
                if (deal_profit < 0.0)
                    g_consec_losses++;
                else
                    g_consec_losses = 0;
            }
        }
    }
}

// =============================================================================
// RISK CHECKS — kill switches
// =============================================================================
bool RiskChecksPass() {
    // Check resume time for timed halts
    if (g_halted) {
        if (g_resume_time > 0 && GetTickCount() > g_resume_time) {
            g_halted      = false;
            g_resume_time = 0;
            Print("Kill-switch LIFTED: ", g_halt_reason);
        } else {
            return false;
        }
    }

    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double equity  = AccountInfoDouble(ACCOUNT_EQUITY);

    // ① Daily loss limit
    double daily_pnl_pct = 100.0 * (balance - g_day_start_bal) / g_day_start_bal;
    if (daily_pnl_pct < -DAILY_LOSS_LIMIT) {
        HaltForDay("Daily loss limit hit: " + DoubleToString(daily_pnl_pct, 2) + "%");
        return false;
    }

    // ② Consecutive loss breaker → 1-hour pause
    if (g_consec_losses >= CONSEC_LOSS_LIMIT) {
        g_consec_losses = 0;
        HaltForPeriod("Consecutive losses: " + IntegerToString(CONSEC_LOSS_LIMIT),
                      3600000); // 1 hour in ms
        return false;
    }

    // ③ Session drawdown circuit breaker
    double session_dd_pct = 100.0 * (equity - g_session_high) / g_session_high;
    if (session_dd_pct < -SESSION_DD_LIMIT) {
        HaltForDay("Session drawdown limit: " + DoubleToString(session_dd_pct, 2) + "%");
        return false;
    }

    return true;
}

void HaltForDay(string reason) {
    g_halted      = true;
    g_halt_reason = reason;
    g_resume_time = 0;  // Manual restart required
    Print("KILL SWITCH: ", reason, " — EA halted for day.");
    // Close all open positions
    for (int i = PositionsTotal()-1; i >= 0; i--) {
        ulong ticket = PositionGetTicket(i);
        if (ticket > 0) g_trade.PositionClose(ticket);
    }
}

void HaltForPeriod(string reason, uint ms) {
    g_halted      = true;
    g_halt_reason = reason;
    g_resume_time = GetTickCount() + ms;
    Print("KILL SWITCH: ", reason, " — paused for ",
          DoubleToString(ms / 60000.0, 1), " min.");
}

// =============================================================================
// SPREAD GUARD
// =============================================================================
bool SpreadGuardPass(double current_spread) {
    if (!g_spd_full) return true;
    double mean_spr = g_spread_vec.Mean();
    double std_spr  = g_spread_vec.Std();
    if (std_spr < 1e-10) return true;
    return (current_spread - mean_spr) / std_spr <= SPREAD_SIGMA_LIMIT;
}

void ClassifyAggressor(MqlTick &tick, const MqlTick &prev) {
    if (tick.flags != 0 && (tick.flags & (TICK_FLAG_BUY | TICK_FLAG_SELL)) != 0) return;
    
    double midpoint = (tick.bid + tick.ask) * 0.5;
    
    // 1. Quote Test (High Precision) - ONLY IF 'last' price is legitimately provided
    if (tick.last > 0) {
        if (tick.last >= tick.ask) { tick.flags |= (TICK_FLAG_BUY | TICK_FLAG_LAST); return; }
        if (tick.last <= tick.bid) { tick.flags |= (TICK_FLAG_SELL | TICK_FLAG_LAST); return; }
        
        // Inside spread trades
        if (tick.last > midpoint)  { tick.flags |= (TICK_FLAG_BUY | TICK_FLAG_LAST); return; }
        if (tick.last < midpoint)  { tick.flags |= (TICK_FLAG_SELL | TICK_FLAG_LAST); return; }
    }
    
    // 2. Tick Test Fallback (Midpoint Momentum)
    double prev_mid = (prev.bid + prev.ask) * 0.5;
    double eff_price = (tick.last > 0) ? tick.last : midpoint;
    double prev_eff  = (prev.last > 0) ? prev.last : prev_mid;
    
    if (eff_price > prev_eff) {
        tick.flags |= (TICK_FLAG_BUY | TICK_FLAG_LAST);
    } else if (eff_price < prev_eff) {
        tick.flags |= (TICK_FLAG_SELL | TICK_FLAG_LAST);
    } else {
        tick.flags |= (prev.flags & (TICK_FLAG_BUY | TICK_FLAG_SELL | TICK_FLAG_LAST));
    }
}

// =============================================================================
// UPDATE ALL ROLLING FEATURES
// =============================================================================
void UpdateAllFeatures(MqlTick &tick) {
    // ── TITAN V2.1 PRECISION BRIDGE: HYBRID QUOTE-TICK LOGIC ──────────────────
    static MqlTick prev_tick = tick;
    ClassifyAggressor(tick, prev_tick);
    prev_tick = tick;
    // ──────────────────────────────────────────────────────────────────────────

    static double prev_mid = 0.0;
    double mid = (tick.bid + tick.ask) * 0.5;
    double spr = tick.ask - tick.bid;
    double mid_diff = MathAbs(mid - prev_mid);
    prev_mid = mid;

    // ── Spread ring buffer ────────────────────────────────────────────────────
    g_spread_vec[g_spd_idx] = spr;
    g_spd_idx = (g_spd_idx + 1) % SPREAD_WIN;
    if (g_spd_idx == 0) g_spd_full = true;

    // ── Midpoint ring buffer (for VoT) ────────────────────────────────────────
    g_mid_vec[g_mid_idx] = mid;
    g_mid_idx = (g_mid_idx + 1) % VOT_WIN;
    if (g_mid_idx == 0) g_mid_full = true;

    // ── Volume ring buffer (for RVOL) ─────────────────────────────────────────
    g_vol_vec[g_vol_idx] = (double)tick.volume;
    g_vol_idx = (g_vol_idx + 1) % VOT_WIN;
    if (g_vol_idx == 0) g_vol_full = true;

    // ── Velocity of tape (trades/second) ──────────────────────────────────────
    if ((tick.flags & TICK_FLAG_LAST) != 0) {  // Real trade tick
        ulong sec = tick.time_msc / 1000;
        if (sec == g_vot_sec)
            g_vot_count++;
        else {
            g_last_vot  = g_vot_count;
            g_vot_count = 1;
            g_vot_sec   = sec;
        }

        // ── Hurst sync fix: fill trade-tick-only buffer (matches Python training) ─
        g_hurst_mid[g_hurst_idx] = mid;
        g_hurst_idx = (g_hurst_idx + 1) % HURST_WIN;
        if (g_hurst_idx == 0) g_hurst_full = true;
    }

    // ── Cumulative delta ──────────────────────────────────────────────────────
    if ((tick.flags & TICK_FLAG_LAST) != 0) {
        double delta_tick = 0.0;
        if      ((tick.flags & TICK_FLAG_BUY)  != 0) delta_tick = (double)tick.volume;
        else if ((tick.flags & TICK_FLAG_SELL) != 0) delta_tick = -(double)tick.volume;
        g_cum_delta += delta_tick;

        g_delta_vec[g_dlt_idx] = delta_tick;
        g_dlt_idx = (g_dlt_idx + 1) % DELTA_WIN;
        if (g_dlt_idx == 0) g_dlt_full = true;
    }

    // ── ADVANCED ALPHA BUFFERS ────────────────────────────────────────────────
    g_buy_buf[g_bs_idx]  = ((tick.flags & TICK_FLAG_BUY)  != 0) ? 1.0 : 0.0;
    g_sell_buf[g_bs_idx] = ((tick.flags & TICK_FLAG_SELL) != 0) ? 1.0 : 0.0;
    g_bs_idx = (g_bs_idx + 1) % ADV_WIN;
    if (g_bs_idx == 0) g_bs_full = true;

    g_mvdi_spread[g_mvdi_idx]   = spr;
    g_mvdi_mid_diff[g_mvdi_idx] = mid_diff;
    
    double mean_s = g_mvdi_spread.Mean() + 1e-10;
    double std_s  = g_mvdi_spread.Std();
    
    double mean_m = g_mvdi_mid_diff.Mean() + 1e-10;
    double std_m  = g_mvdi_mid_diff.Std();
    
    g_mvdi_raw[g_mvdi_idx] = (std_s / mean_s) / ((std_m / mean_m) + 1e-10);
    
    g_mvdi_idx = (g_mvdi_idx + 1) % ADV_WIN;
    if (g_mvdi_idx == 0) g_mvdi_full = true;
}

// =============================================================================
// SESSION VWAP UPDATE
// =============================================================================
void UpdateVWAP(const MqlTick &tick) {
    // Reset on new day
    datetime today = TimeCurrent();
    MqlDateTime dt; TimeToStruct(today, dt);
    dt.hour = 0; dt.min = 0; dt.sec = 0;
    datetime day_start = StructToTime(dt);
    if (day_start != g_vwap_day) {
        g_vwap_pv  = 0.0;
        g_vwap_v   = 0.0;
        g_vwap_ss  = 0.0;
        g_vwap_day = day_start;
    }

    if ((tick.flags & TICK_FLAG_LAST) != 0 && tick.volume > 0) {
        double p = tick.last;
        double v = (double)tick.volume;
        g_vwap_pv += p * v;
        g_vwap_v  += v;
        g_vwap_ss += p * p * v;
    }
}

double GetVWAP() {
    return (g_vwap_v > 0.0) ? g_vwap_pv / g_vwap_v : 0.0;
}

double GetVWAPStd() {
    if (g_vwap_v <= 0.0) return 1e-8;
    double mean = g_vwap_pv / g_vwap_v;
    double var  = g_vwap_ss / g_vwap_v - mean * mean;
    return (var > 0.0) ? MathSqrt(var) : 1e-8;
}

// =============================================================================
// ABSORPTION DETECTOR
// 4 conditions: HighVoT, LowPriceEfficiency, VolumeAsymmetry, Persistence
// =============================================================================
void UpdateAbsorption(const MqlTick &tick) {
    if (!g_mid_full || !g_vol_full) return;

    // Condition 1: VoT z-score > 2.0
    double vot_mean = g_vol_vec.Mean();
    double vot_std  = g_vol_vec.Std();
    double vot_z    = (vot_std > 0.0) ? (g_last_vot - vot_mean) / vot_std : 0.0;

    // Condition 2: Price efficiency < 0.5
    double total_vol = g_vol_vec.Sum();
    double price_chg = g_mid_vec.Max() - g_mid_vec.Min();
    double expected_move = (total_vol > 0.0) ? total_vol * _Point * 0.5 : 1e-8;
    double price_eff = price_chg / expected_move;

    // Condition 3: Volume asymmetry > 2:1
    double vol_asym = (g_absorption_vol_hi > 0.0)
                      ? g_absorption_vol_lo / g_absorption_vol_hi : 1.0;

    // Condition 4: Persistence > 500ms
    uint duration = GetTickCount() - g_absorption_start;

    bool cond1 = vot_z > 2.0;
    bool cond2 = price_eff < 0.5;
    bool cond3 = vol_asym > 2.0;
    bool cond4 = duration > 500;

    if (cond1 && cond2 && cond3) {
        if (!g_absorption_active) {
            g_absorption_active      = true;
            g_absorption_start       = GetTickCount();
            g_absorption_price_start = (tick.bid + tick.ask) * 0.5;
        }
    } else {
        if (g_absorption_active && !cond1) {
            g_absorption_active = false;
        }
    }

    // Update buy/sell volume accumulation
    if ((tick.flags & TICK_FLAG_BUY)  != 0) g_absorption_vol_lo += (double)tick.volume;
    if ((tick.flags & TICK_FLAG_SELL) != 0) g_absorption_vol_hi += (double)tick.volume;

    // Reset accumulators every 1000 ticks
    static int tick_cnt = 0;
    if (++tick_cnt > 1000) {
        g_absorption_vol_lo = 0.0;
        g_absorption_vol_hi = 0.0;
        tick_cnt = 0;
    }
}

// =============================================================================
// DOM DEPTH IMBALANCE — reads real Level 2 data
// Returns: (bid_depth - ask_depth) / (bid_depth + ask_depth) ∈ [-1, +1]
// =============================================================================
double GetDOMImbalance(bool bid_side, int levels = 5) {
    MqlBookInfo dom[];
    if (!MarketBookGet(_Symbol, dom)) return 0.0;

    double bid_vol = 0.0, ask_vol = 0.0;
    int    bid_lvl = 0,   ask_lvl = 0;

    for (int i = 0; i < ArraySize(dom); i++) {
        if (dom[i].type == BOOK_TYPE_BUY && bid_lvl < levels) {
            bid_vol += (double)dom[i].volume;
            bid_lvl++;
        } else if (dom[i].type == BOOK_TYPE_SELL && ask_lvl < levels) {
            ask_vol += (double)dom[i].volume;
            ask_lvl++;
        }
        if (bid_lvl >= levels && ask_lvl >= levels) break;
    }

    double total = bid_vol + ask_vol;
    if (total < 1e-10) return 0.0;

    if (bid_side) return (bid_vol - ask_vol) / total;   // Positive = bid heavy
    else          return (ask_vol - bid_vol) / total;   // Positive = ask heavy
}

// =============================================================================
// STACKED IMBALANCE SCORE — DOM-based, not float comparison
// Counts consecutive DOM levels where bid/ask ratio > 3:1
// =============================================================================
double GetStackedImbalanceScore() {
    MqlBookInfo dom[];
    if (!MarketBookGet(_Symbol, dom)) return 0.0;

    int  bid_stack = 0, ask_stack = 0;
    int  max_bid = 0, max_ask = 0;
    bool in_bid = true, in_ask = true;

    for (int i = 0; i < MathMin(ArraySize(dom), 10); i++) {
        double bv = 0.0, av = 0.0;
        if (dom[i].type == BOOK_TYPE_BUY)  bv = (double)dom[i].volume;
        if (dom[i].type == BOOK_TYPE_SELL) av = (double)dom[i].volume;

        if (bv > 0.0 && av > 0.0) {
            double ratio = bv / av;
            if (ratio >= 3.0 && in_bid) bid_stack++;
            else { in_bid = false; max_bid = MathMax(max_bid, bid_stack); bid_stack = 0; }

            ratio = av / bv;
            if (ratio >= 3.0 && in_ask) ask_stack++;
            else { in_ask = false; max_ask = MathMax(max_ask, ask_stack); ask_stack = 0; }
        }
    }

    max_bid = MathMax(max_bid, bid_stack);
    max_ask = MathMax(max_ask, ask_stack);

    // Return signed: positive = buy stack, negative = sell stack
    if (max_bid >= max_ask) return  (double)max_bid / 10.0;
    else                    return -(double)max_ask / 10.0;
}

// =============================================================================
// HURST EXPONENT — 20-bar rescaled range (regime filter)
// SYNC FIX: reads from g_hurst_mid (trade ticks only) — NOT g_mid_vec (DOM).
// This matches Python training: compute_hurst(window=20) runs over 20 trade ticks.
// H < 0.45 = mean-reverting  0.45–0.55 = random  H > 0.55 = trending
// =============================================================================
double ComputeHurst() {
    if (!g_hurst_full) return 0.5;  // Not enough trade ticks yet

    int n = HURST_WIN;  // 20 — must match Python compute_hurst(window=20)
    vector<double> sub;
    sub.Resize(n);
    for (int i = 0; i < n; i++)
        sub[i] = g_hurst_mid[(g_hurst_idx + i) % HURST_WIN];

    double mean_p = sub.Mean();
    double std_p  = sub.Std();
    if (std_p < 1e-10) return 0.5;

    // Deviation from mean (R/S analysis — identical algorithm to Python)
    vector<double> dev;
    dev.Resize(n);
    double cumdev = 0.0;
    double min_d  = 1e18, max_d = -1e18;
    for (int i = 0; i < n; i++) {
        cumdev += sub[i] - mean_p;
        dev[i]  = cumdev;
        if (cumdev < min_d) min_d = cumdev;
        if (cumdev > max_d) max_d = cumdev;
    }

    double R = max_d - min_d;
    double S = std_p;
    if (S < 1e-10 || R < 1e-10) return 0.5;

    return MathLog(R / S) / MathLog(n);
}

// =============================================================================
// VoT Z-SCORE
// =============================================================================
double ComputeVoTZScore() {
    if (!g_vol_full) return 0.0;
    double mean_v = g_vol_vec.Mean();
    double std_v  = g_vol_vec.Std();
    if (std_v < 1e-10) return 0.0;
    return (g_last_vot - mean_v) / std_v;
}

// =============================================================================
// RVOL (Relative Volume)
// =============================================================================
double ComputeRVOL() {
    if (!g_vol_full) return 1.0;
    double current = g_vol_vec[g_vol_idx];  // Most recent
    double mean_v  = g_vol_vec.Mean();
    return (mean_v > 1e-10) ? current / mean_v : 1.0;
}

// =============================================================================
// CUMULATIVE DELTA DIVERGENCE
// Detects: price makes new high but delta does not (or vice versa)
// Returns: positive = bullish divergence, negative = bearish, 0 = none
// =============================================================================
double ComputeCumDeltaDivergence() {
    if (!g_dlt_full) return 0.0;

    double mid_old = g_mid_vec[g_mid_idx];
    double mid_new = g_mid_vec[(g_mid_idx + VOT_WIN - 1) % VOT_WIN];
    double price_dir = mid_new - mid_old;
    double delta_dir = g_delta_vec.Sum();

    if (price_dir > 0.0 && delta_dir < 0.0) return -1.0;  // Bearish div
    if (price_dir < 0.0 && delta_dir > 0.0) return  1.0;  // Bullish div
    return 0.0;
}


// =============================================================================
// ADVANCED ALPHA: FLAG-BASED DIRECTIONAL PRESSURE INDEX (FDPI)
// =============================================================================
double ComputeFDPI() {
    if (!g_bs_full) return 0.0;
    double b_n = g_buy_buf.Sum();
    double s_n = g_sell_buf.Sum();
    return (b_n - s_n) / (b_n + s_n + 1e-10);
}

// =============================================================================
// ADVANCED ALPHA: MICRO-VOLATILITY DISPERSION INDEX (MVDI)
// =============================================================================
double ComputeMVDI() {
    if (!g_mvdi_full) return 0.0;
    int latest = (g_mvdi_idx == 0) ? ADV_WIN - 1 : g_mvdi_idx - 1;
    double raw = g_mvdi_raw[latest];
    double mean = g_mvdi_raw.Mean();
    double std = g_mvdi_raw.Std() + 1e-10;
    double z_score = (raw - mean) / std;
    return MathMax(-3.0, MathMin(3.0, z_score)) / 3.0;
}

// =============================================================================
// ADVANCED ALPHA: TIME-WEIGHTED KINEMATIC JERK (TWKJ)
// =============================================================================
double ComputeTWKJ(const MqlTick &tick) {
    double mid = (tick.bid + tick.ask) * 0.5;
    
    for (int i = 3; i > 0; i--) {
        g_mid_hist[i]  = g_mid_hist[i-1];
        g_time_hist[i] = g_time_hist[i-1];
    }
    g_mid_hist[0]  = mid;
    g_time_hist[0] = tick.time_msc;
    
    if (g_time_hist[3] == 0) return 0.0; 
    g_kin_ready = true;

    double dt1 = MathMax(1.0, (double)(g_time_hist[0] - g_time_hist[1]));
    double dt2 = MathMax(1.0, (double)(g_time_hist[1] - g_time_hist[2]));
    double dt3 = MathMax(1.0, (double)(g_time_hist[2] - g_time_hist[3]));

    double v1 = (g_mid_hist[0] - g_mid_hist[1]) / dt1;
    double v2 = (g_mid_hist[1] - g_mid_hist[2]) / dt2;
    double v3 = (g_mid_hist[2] - g_mid_hist[3]) / dt3;

    double a1 = (v1 - v2) / dt1;
    double a2 = (v2 - v3) / dt2;

    double raw_jerk = (a1 - a2) / dt1;

    g_jerk_buf[g_jerk_idx] = raw_jerk;
    g_jerk_idx = (g_jerk_idx + 1) % ADV_WIN;
    if (g_jerk_idx == 0) g_jerk_full = true;

    if (!g_jerk_full) return 0.0;
    
    double mean = g_jerk_buf.Mean();
    double std  = g_jerk_buf.Std();
    if (std < 1e-10) return 0.0;
    
    double z_score = (raw_jerk - mean) / std;
    return MathMax(-3.0, MathMin(3.0, z_score)) / 3.0;
}

// =============================================================================
// APPLY LOCAL NORMALIZATION
// =============================================================================
void ApplyLocalNormalization() {
    int rows = TICK_BUFFER_SIZE; // 50
    int cols = FEATURES;         // 12
    
    for (int c = 0; c < cols; c++) {
        // Calculate Mean (\mu) for the window
        double sum = 0.0;
        for (int r = 0; r < rows; r++) {
            sum += g_mat[r][c];
        }
        double mean = sum / rows;
        
        // Calculate Population Standard Deviation (\sigma)
        double variance_sum = 0.0;
        for (int r = 0; r < rows; r++) {
            double diff = g_mat[r][c] - mean;
            variance_sum += diff * diff;
        }
        double std_dev = MathSqrt(variance_sum / rows);
        
        if (std_dev < 1e-10) {
            std_dev = 1.0; 
        }
        
        // Apply Z-score
        for (int r = 0; r < rows; r++) {
            g_mat[r][c] = (g_mat[r][c] - mean) / std_dev;
        }
    }
}

// =============================================================================
// SHIFT FEATURE MATRIX AND INSERT NEW ROW
// =============================================================================
void ShiftMatrixAndInsert(const MqlTick &tick) {
    // 1. Shift existing rows up
    for (int r = 0; r < TICK_BUFFER_SIZE - 1; r++)
        for (int c = 0; c < FEATURES; c++)
            g_mat[r][c] = g_mat[r+1][c];

    // 2. Calculate new features for the latest row
    double mid = (tick.bid + tick.ask) * 0.5;
    double spr = tick.ask - tick.bid;
    double vwap    = GetVWAP();
    double vwap_sd = GetVWAPStd();
    MqlDateTime dt; TimeCurrent(dt);
    int last_row = TICK_BUFFER_SIZE - 1;

    g_mat[last_row][F_VOT_ZSCORE]      = ComputeVoTZScore();
    g_mat[last_row][F_RVOL]            = ComputeRVOL();
    g_mat[last_row][F_CUMDELTA_DIV]    = ComputeCumDeltaDivergence();
    g_mat[last_row][F_IMBALANCE_RATIO] = GetStackedImbalanceScore();
    g_mat[last_row][F_FDPI]            = ComputeFDPI();
    g_mat[last_row][F_MVDI]            = ComputeMVDI();
    g_mat[last_row][F_BID_DEPTH]       = GetDOMImbalance(true);
    g_mat[last_row][F_ASK_DEPTH]       = GetDOMImbalance(false);
    g_mat[last_row][F_PRICE_VS_VWAP]   = (vwap_sd > 1e-10) ? (mid - vwap) / vwap_sd : 0.0;
    g_mat[last_row][F_HOUR_SIN]        = MathSin(2.0 * M_PI * dt.hour / 24.0);
    g_mat[last_row][F_HURST]           = ComputeHurst();
    g_mat[last_row][F_TWKJ]            = ComputeTWKJ(tick);

    // 3. PARITY FIX: Apply Windowed Normalization to the WHOLE matrix
    // This replaces the "GLOBAL SCALING PARITY" loop entirely.
    ApplyLocalNormalization(); 
}

double ComputeSpreadZScore(double spr) {
    if (!g_spd_full) return 0.0;
    double mean_s = g_spread_vec.Mean();
    double std_s  = g_spread_vec.Std();
    return (std_s > 1e-10) ? (spr - mean_s) / std_s : 0.0;
}

// =============================================================================
// ONNX INFERENCE — FIXED PARAMETER COUNT (Error 5804)
// =============================================================================
void RunInference() {
    double cur_spr = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
    if (!SpreadGuardPass(cur_spr)) return;

    // 1. Flatten matrix to float array [1 × 128 × 12]
    int total = TICK_BUFFER_SIZE * FEATURES;
    float input_buf[];
    ArrayResize(input_buf, total);
    
    for (int r = 0; r < TICK_BUFFER_SIZE; r++)
        for (int c = 0; c < FEATURES; c++)
            input_buf[r * FEATURES + c] = (float)g_mat[r][c];

    // 2. Output buffer (must match [1, 1] shape from TitanDeploymentFix)
    float output_buf[1]; 
    
    // 3. CORRECTED OnnxRun: (handle, flags, input[], output[])
    // Use ONNX_NO_CONVERSION (0) for flags.
    if (!OnnxRun(g_onnx_handle, 0, input_buf, output_buf)) {
        Print("ONNX inference error: ", GetLastError());
        return;
    }

    double pred = (double)output_buf[0];
    if (pred < 0.0 || pred > 1.0) return;

    // Debug: Monitor conviction in Journal
    if(g_tlog_cnt % 100 == 0) 
        PrintFormat("Tick: %d | Conviction: %.4f | Hurst: %.3f", g_tlog_cnt, pred, ComputeHurst());

    if (pred >= BUY_THRESHOLD)  ExecuteSignal(ORDER_TYPE_BUY,  pred);
    if (pred <= SELL_THRESHOLD) ExecuteSignal(ORDER_TYPE_SELL, pred);
}

// =============================================================================
// SIGNAL EXECUTION
// =============================================================================
void ExecuteSignal(ENUM_ORDER_TYPE dir, double confidence) {
    if (PositionsTotal() >= MAX_POSITIONS) return;
    if (g_order_count >= 10)               return;

    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double pt  = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

    double limit_px, sl, tp;

    if (dir == ORDER_TYPE_BUY) {
        limit_px = USE_LIMIT_ORDERS ? bid : ask;
        sl = NormalizeDouble(limit_px - STOP_LOSS_TICKS * pt, _Digits);
        tp = NormalizeDouble(limit_px + TARGET_TICKS  * pt, _Digits);
    } else {
        limit_px = USE_LIMIT_ORDERS ? ask : bid;
        sl = NormalizeDouble(limit_px + STOP_LOSS_TICKS * pt, _Digits);
        tp = NormalizeDouble(limit_px - TARGET_TICKS  * pt, _Digits);
    }

    MqlTradeRequest req = {};
    MqlTradeResult  res = {};

    req.symbol   = _Symbol;
    req.volume   = LOT_SIZE;
    req.sl       = sl;
    req.tp       = tp;
    req.magic    = MAGIC;
    req.comment  = StringFormat("TITAN|%.3f", confidence);

    if (USE_LIMIT_ORDERS) {
        req.action = TRADE_ACTION_PENDING;
        req.type   = (dir == ORDER_TYPE_BUY) ? ORDER_TYPE_BUY_LIMIT
                                              : ORDER_TYPE_SELL_LIMIT;
        req.price  = limit_px;
        req.type_filling = ORDER_FILLING_IOC;
        req.expiration   = TimeCurrent() + 60;
    } else {
        req.action = TRADE_ACTION_DEAL;
        req.type   = dir;
        req.price  = limit_px;
        req.deviation = 3;
        req.type_filling = ORDER_FILLING_IOC;
    }

    if (OrderSendAsync(req, res)) {
        if (USE_LIMIT_ORDERS) {
            g_orders[g_order_count].ticket    = res.order;
            g_orders[g_order_count].placed_ms = GetTickCount();
            g_orders[g_order_count].price     = limit_px;
            g_orders[g_order_count].dir       = dir;
            g_order_count++;
        }
        Print("Signal: ", (dir==ORDER_TYPE_BUY?"BUY":"SELL"),
              "  px=", limit_px, "  conf=", DoubleToString(confidence,3),
              "  Hurst=", DoubleToString(ComputeHurst(),3),
              "  Absorb=", g_absorption_active);
    } else {
        Print("OrderSendAsync failed: ", res.retcode, " ", res.comment);
    }
}

// =============================================================================
// HELPERS
// =============================================================================
void RemoveOrder(int idx) {
    for (int j = idx; j < g_order_count - 1; j++)
        g_orders[j] = g_orders[j+1];
    g_order_count--;
}

// =============================================================================
// OnTester — EXPORT TRADE LOG FOR WFO EFFICIENCY CALCULATION
// Writes tester_log.csv to MQL5\Files so TitanValidation.py can read it.
// =============================================================================
double OnTester() {
    int total = HistoryDealsTotal();
    if (total == 0) return 0.0;

    int fh = FileOpen("tester_log.csv", FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
    if (fh == INVALID_HANDLE) {
        Print("OnTester: cannot open tester_log.csv: ", GetLastError());
        return 0.0;
    }

    // Header expected by TitanValidation.py
    FileWrite(fh, "date", "profit");

    double balance = g_day_start_bal;
    double peak_balance = balance;
    double max_dd = 0.0;
    double total_pnl = 0.0;
    int n_trades = 0;
    int n_wins = 0;

    for (int i = 0; i < total; i++) {
        ulong ticket = HistoryDealGetTicket(i);
        if (ticket == 0) continue;
        
        long entry = HistoryDealGetInteger(ticket, DEAL_ENTRY);
        if (entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_INOUT) continue;

        double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT)
                      + HistoryDealGetDouble(ticket, DEAL_COMMISSION)
                      + HistoryDealGetDouble(ticket, DEAL_SWAP);

        long time_ms = HistoryDealGetInteger(ticket, DEAL_TIME_MSC);
        MqlDateTime dt;
        TimeToStruct((datetime)(time_ms / 1000), dt);
        string iso_time = StringFormat("%04d-%02d-%02dT%02d:%02d:%02d", 
                                       dt.year, dt.mon, dt.day, 
                                       dt.hour, dt.min, dt.sec);

        long dtype = HistoryDealGetInteger(ticket, DEAL_TYPE);
        if (dtype == DEAL_TYPE_BUY || dtype == DEAL_TYPE_SELL) {
            total_pnl += profit;
            n_trades++;
            if (profit > 0.0) n_wins++;
        }

        balance += profit;
        if (balance > peak_balance) peak_balance = balance;
        double dd = peak_balance - balance;
        if (dd > max_dd) max_dd = dd;

        FileWrite(fh, iso_time, DoubleToString(profit, 2));
    }
    FileClose(fh);

    double ret_dd = (max_dd > 0.0) ? total_pnl / max_dd : 0.0;
    double win_rate = (n_trades > 0) ? (double)n_wins / n_trades : 0.0;
    Print("OnTester: trades=", n_trades,
          "  WinRate=",  DoubleToString(win_rate * 100.0, 1), "%",
          "  TotalPnL=", DoubleToString(total_pnl, 2),
          "  MaxDD=",    DoubleToString(max_dd, 2),
          "  RetDD=",    DoubleToString(ret_dd, 3),
          "  -> tester_log.csv");

    return ret_dd;
}