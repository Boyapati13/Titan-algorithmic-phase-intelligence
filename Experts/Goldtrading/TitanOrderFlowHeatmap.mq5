#property copyright "Titan HFT System"
#property link      "https://titan-hft.com"
#property version   "1.00"
#property description "Titan Order Flow Heatmap - Phase 4 Indicator Suite"
#property indicator_separate_window
#property indicator_buffers 2  // Data buffer + colour-index buffer (required for DRAW_COLOR_HISTOGRAM)
#property indicator_plots   1

//--- input parameters
input int      Heatmap_Depth = 50;        // Price levels to display
input int      Time_Windows = 20;         // Time periods for heatmap
input color    Buy_Color = clrBlue;       // Color for buy volume
input color    Sell_Color = clrRed;       // Color for sell volume
input int      Update_Frequency = 100;    // Update every N ms

//--- indicator buffers
double HeatmapBuffer[];
double HeatmapColorIdx[];  // Colour index per bar (0 = Buy_Color, 1 = Sell_Color)

//--- global variables
int heatmap_handle;
uint last_update_ms;  // GetTickCount() timestamp (milliseconds)
int tick_count;

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit() {
    // Set indicator properties
    IndicatorSetString(INDICATOR_SHORTNAME, "Titan Order Flow Heatmap");
    IndicatorSetInteger(INDICATOR_DIGITS, 0);

    // Set plot properties
    PlotIndexSetString(0, PLOT_LABEL, "Volume Intensity");
    PlotIndexSetInteger(0, PLOT_TYPE, DRAW_COLOR_HISTOGRAM);
    PlotIndexSetInteger(0, PLOT_LINE_WIDTH, 1);

    // Bind data buffer
    SetIndexBuffer(0, HeatmapBuffer,  INDICATOR_DATA);
    SetIndexBuffer(1, HeatmapColorIdx, INDICATOR_COLOR_INDEX);  // Required by DRAW_COLOR_HISTOGRAM
    ArraySetAsSeries(HeatmapBuffer,   true);
    ArraySetAsSeries(HeatmapColorIdx, true);

    // Register both colours in slot 0 of plot 0
    PlotIndexSetInteger(0, PLOT_LINE_COLOR, 0, Buy_Color);
    PlotIndexSetInteger(0, PLOT_LINE_COLOR, 1, Sell_Color);

    heatmap_handle  = 0;
    last_update_ms  = 0;
    tick_count      = 0;

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                       |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    // Cleanup if needed
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

    // Throttle updates to Update_Frequency ms (fix: use GetTickCount, not TimeCurrent)
    uint now_ms = GetTickCount();
    if (now_ms - last_update_ms < (uint)Update_Frequency) {
        return rates_total;
    }
    last_update_ms = now_ms;

    int start = prev_calculated > 0 ? prev_calculated - 1 : 0;

    for (int i = start; i < rates_total; i++) {
        // Calculate volume intensity for this bar
        double volume_intensity = CalculateVolumeIntensity(i, time, tick_volume);
        HeatmapBuffer[i] = volume_intensity;

        // Set per-bar colour index (0 = bullish/Buy_Color, 1 = bearish/Sell_Color)
        HeatmapColorIdx[i] = (close[i] > open[i]) ? 0.0 : 1.0;
    }

    return rates_total;
}

//+------------------------------------------------------------------+
//| Calculate volume intensity for heatmap                           |
//+------------------------------------------------------------------+
double CalculateVolumeIntensity(int bar_index, const datetime &time[], const long &tick_volume[]) {
    // Simplified volume intensity calculation
    // In real implementation, this would analyze order book data

    static double prev_volume = 0;
    double current_volume = (double)tick_volume[bar_index];

    if (prev_volume == 0) {
        prev_volume = current_volume;
        return 0;
    }

    // Calculate volume change rate
    double volume_change = (current_volume - prev_volume) / prev_volume;
    prev_volume = current_volume;

    // Normalize to 0-100 scale
    double intensity = 50 + volume_change * 1000; // Arbitrary scaling
    intensity = MathMax(0, MathMin(100, intensity));

    return intensity;
}