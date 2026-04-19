#property copyright "Titan HFT System"
#property link      "https://titan-hft.com"
#property version   "1.00"
#property description "Titan Absorption Meter - Phase 4 Indicator Suite"
#property indicator_separate_window
#property indicator_buffers 2
#property indicator_plots   2

//--- input parameters
input int      Absorption_Lookback = 10;     // Bars to analyze for absorption
input double   Absorption_Threshold = 2.0;   // Standard deviation threshold
input int      Min_Order_Size = 1000000;     // Minimum order size to consider (in base units)
input color    Absorption_Color = clrOrange; // Color for absorption signals
input color    Normal_Color = clrGray;       // Color for normal activity

//--- indicator buffers
double AbsorptionBuffer[];
double AbsorptionSignal[];

//--- global variables
double price_levels[];
double volume_at_levels[];
int max_levels = 100;

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit() {
    // Set indicator properties
    IndicatorSetString(INDICATOR_SHORTNAME, "Titan Absorption Meter");
    IndicatorSetInteger(INDICATOR_DIGITS, 2);

    // Set plot properties
    PlotIndexSetString(0, PLOT_LABEL, "Absorption Level");
    PlotIndexSetInteger(0, PLOT_DRAW_TYPE, DRAW_LINE);
    PlotIndexSetInteger(0, PLOT_LINE_WIDTH, 2);
    PlotIndexSetInteger(0, PLOT_LINE_COLOR, Absorption_Color);

    PlotIndexSetString(1, PLOT_LABEL, "Absorption Signal");
    PlotIndexSetInteger(1, PLOT_DRAW_TYPE, DRAW_HISTOGRAM);
    PlotIndexSetInteger(1, PLOT_LINE_WIDTH, 1);
    PlotIndexSetInteger(1, PLOT_LINE_COLOR, Normal_Color);

    // Initialize buffers
    SetIndexBuffer(0, AbsorptionBuffer, INDICATOR_DATA);
    SetIndexBuffer(1, AbsorptionSignal, INDICATOR_DATA);
    ArraySetAsSeries(AbsorptionBuffer, true);
    ArraySetAsSeries(AbsorptionSignal, true);

    // Initialize price level tracking
    ArrayResize(price_levels, max_levels);
    ArrayResize(volume_at_levels, max_levels);
    ArrayInitialize(price_levels, 0);
    ArrayInitialize(volume_at_levels, 0);

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                       |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    // Cleanup arrays
    ArrayFree(price_levels);
    ArrayFree(volume_at_levels);
}

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[]) {

    int start = prev_calculated > 0 ? prev_calculated - 1 : Absorption_Lookback;

    for (int i = start; i < rates_total; i++) {
        // Calculate absorption metric
        double absorption_level = CalculateAbsorption(i, Absorption_Lookback, high, low, volume);
        AbsorptionBuffer[i] = absorption_level;

        // Generate signal based on threshold
        double signal = absorption_level > Absorption_Threshold ? absorption_level : 0;
        AbsorptionSignal[i] = signal;

        // Update price level tracking
        UpdatePriceLevels(i, high[i], low[i], volume[i]);
    }

    return rates_total;
}

//+------------------------------------------------------------------+
//| Calculate absorption level                                       |
//+------------------------------------------------------------------+
double CalculateAbsorption(int bar_index, int lookback, const double &high[], const double &low[], const long &volume[]) {
    if (bar_index < lookback) return 0;

    double total_volume = 0;
    double price_range_sum = 0;

    // Calculate average volume and price range over lookback
    for (int i = bar_index - lookback; i < bar_index; i++) {
        total_volume += (double)volume[i];
        price_range_sum += high[i] - low[i];
    }

    double avg_volume = total_volume / lookback;
    double avg_range = price_range_sum / lookback;

    // Current bar metrics
    double current_volume = (double)volume[bar_index];
    double current_range = high[bar_index] - low[bar_index];

    // Absorption occurs when high volume but small price movement
    if (current_range < avg_range * 0.5 && current_volume > avg_volume * Absorption_Threshold) {
        return (current_volume / avg_volume) / (avg_range / current_range + 0.001); // Avoid division by zero
    }

    return 0;
}

//+------------------------------------------------------------------+
//| Update price level tracking for absorption analysis              |
//+------------------------------------------------------------------+
void UpdatePriceLevels(int bar_index, double high, double low, long volume) {
    double price_step = SymbolInfoDouble(Symbol(), SYMBOL_POINT) * 10;
    if (price_step <= 0) return;

    int levels_count = (int)MathMax(1, (high - low) / price_step) + 1;
    if (levels_count > max_levels) levels_count = max_levels;
    if (levels_count <= 0) return;  // Guard: high == low (no range)

    for (int i = 0; i < levels_count; i++) {
        double level_price = low + i * price_step;
        int level_index    = i % max_levels;
        price_levels[level_index]      = level_price;
        volume_at_levels[level_index] += (double)volume / levels_count;
    }
}