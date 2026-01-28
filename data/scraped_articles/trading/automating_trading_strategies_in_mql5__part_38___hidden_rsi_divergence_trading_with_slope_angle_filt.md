---
title: Automating Trading Strategies in MQL5 (Part 38): Hidden RSI Divergence Trading with Slope Angle Filters
url: https://www.mql5.com/en/articles/20157
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T17:54:06.561217
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/20157&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049467280728894476)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 37)](https://www.mql5.com/en/articles/20031), we developed a **Regular RSI Divergence Convergence** system in MetaQuotes Language 5 (MQL5) that detected regular bullish and bearish divergences between price swings and Relative Strength Index (RSI) values, executed trades on signals with optional risk controls, and provided on-chart visualizations for enhanced analysis. In Part 38, we develop a [Hidden RSI Divergence Trading system](https://www.mql5.com/go?link=https://www.tradingsetupsreview.com/rsi-hidden-divergence-pullback-trading-guide/ "https://www.tradingsetupsreview.com/rsi-hidden-divergence-pullback-trading-guide/") with slope angle filters.

This system identifies hidden bullish and bearish divergences using swing points, applies clean checks with bar ranges and tolerance, filters signals via customizable slope angles on price and RSI lines, executes trades with risk management, and includes visual markers with angle displays on charts. We will cover the following topics:

1. [Understanding the Hidden RSI Divergence Strategy](https://www.mql5.com/en/articles/20157#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/20157#para2)
3. [Backtesting](https://www.mql5.com/en/articles/20157#para3)
4. [Conclusion](https://www.mql5.com/en/articles/20157#para4)

By the end, you’ll have a functional MQL5 strategy for trading hidden RSI divergences, ready for customization—let’s dive in!

### Understanding the Hidden RSI Divergence Strategy

The [hidden RSI divergence strategy](https://www.mql5.com/go?link=https://www.tradingsetupsreview.com/rsi-hidden-divergence-pullback-trading-guide/ "https://www.tradingsetupsreview.com/rsi-hidden-divergence-pullback-trading-guide/") focuses on identifying trend continuation opportunities by detecting specific mismatches between price swings and the Relative Strength Index (RSI) oscillator, which highlights underlying momentum strength in ongoing trends. For hidden bullish divergence, the price establishes a higher low while the RSI forms a lower low, suggesting that bearish pullbacks are weakening and the uptrend may resume. For hidden bearish divergence, the price creates a lower high, but the RSI shows a higher high, indicating that bullish corrections are fading and the downtrend could persist.

We intend to enhance reliability by filtering divergences with slope angles on both price and RSI lines to confirm sufficient steepness or flatness, apply tolerance thresholds for clean patterns without breaches, and enter trades accordingly—buying on hidden bullish signals or selling on hidden bearish ones—with defined risk parameters like stops, profits, and trailing mechanisms. By leveraging these elements, we can pursue high-probability continuation setups in established trends. Have a look below at the different setups we could have.

Hidden Bullish Divergence Setup:

![HIDDEN BULLISH RSI DIVERGENCE](https://c.mql5.com/2/178/Screenshot_2025-11-03_171924.png)

Hidden Bearish Divergence Setup:

![HIDDEN BEARISH RSI DIVERGENCE](https://c.mql5.com/2/178/Screenshot_2025-11-03_171950.png)

Our plan is to detect swing highs and lows with confirmation strength, validate hidden divergences through clean checks within specified bar ranges and tolerance, apply optional slope angle filters on price and RSI for signal quality, execute automated trades with customizable lot sizing and risk controls, and provide visual aids like colored lines and labels with angle displays on both charts, building an effective system for hidden divergence trading. In brief, here is a visual representation of our objectives.

![OVERALL PLAN](https://c.mql5.com/2/178/Screenshot_2025-11-03_145545.png)

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Experts folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some [input parameters](https://www.mql5.com/en/docs/basis/variables/inputvariables) and [global variables](https://www.mql5.com/en/docs/basis/variables/global) that we will use throughout the program.

```
//+------------------------------------------------------------------+
//|                         RSI Hidden Divergence Convergence EA.mq5 |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//|                                   https://t.me/Forex_Algo_Trader |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Allan Munene Mutiiria."
#property link      "https://t.me/Forex_Algo_Trader"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Input Parameters                                                 |
//+------------------------------------------------------------------+
input group "RSI Settings"
input int               RSI_Period   = 14;                // RSI Period
input ENUM_APPLIED_PRICE RSI_Applied = PRICE_CLOSE;       // RSI Applied Price
input group "Swing Settings"
input int   Swing_Strength    = 5;                        // Bars to confirm swing high/low
input int   Min_Bars_Between  = 5;                        // Min bars between swings for divergence
input int   Max_Bars_Between  = 50;                       // Max bars between swings for divergence
input double Tolerance        = 0.1;                      // Tolerance for clean divergence check
input group "Price Divergence Filter"
input bool   Use_Price_Slope_Filter   = false;            // Use Slope Angle Filter for Price
input double Price_Min_Slope_Degrees  = 10.0;             // Minimum Slope Angle in Degrees for Price (0 to disable min)
input double Price_Max_Slope_Degrees  = 80.0;             // Maximum Slope Angle in Degrees for Price (90 to disable max)
input group "RSI Divergence Filter"
input bool   Use_RSI_Slope_Filter    = true;              // Use Slope Angle Filter for RSI
input double RSI_Min_Slope_Degrees   = 1.0;               // Minimum Slope Angle in Degrees for RSI (0 to disable min)
input double RSI_Max_Slope_Degrees   = 89.0;              // Maximum Slope Angle in Degrees for RSI (90 to disable max)
input group "Trade Settings"
input double Lot_Size      = 0.01;                        // Fixed Lot Size
input int    Magic_Number  = 123456789;                   // Magic Number
input double SL_Pips       = 300.0;                       // Stop Loss in Pips (0 to disable)
input double TP_Pips       = 300.0;                       // Take Profit in Pips (0 to disable)
input group "Trailing Stop Settings"
input bool   Enable_Trailing_Stop     = true;             // Enable Trailing Stop
input double Trailing_Stop_Pips       = 30.0;             // Trailing Stop in Pips
input double Min_Profit_To_Trail_Pips = 50.0;             // Minimum Profit to Start Trailing in Pips
input group "Visualization"
input bool         Mark_Swings_On_Price = true;           // Mark Swing Points on Price Chart
input bool         Mark_Swings_On_RSI   = true;           // Mark Swing Points on RSI
input color        Bull_Color           = clrGreen;       // Bullish Divergence Color
input color        Bear_Color           = clrRed;         // Bearish Divergence Color
input color        Swing_High_Color     = clrRed;         // Color for Swing High Labels
input color        Swing_Low_Color      = clrGreen;       // Color for Swing Low Labels
input int          Line_Width           = 2;              // Divergence Line Width
input ENUM_LINE_STYLE Line_Style        = STYLE_SOLID;    // Divergence Line Style
input int          Font_Size            = 8;              // Swing Point Font Size

//+------------------------------------------------------------------+
//| Indicator Handles and Trade Object                               |
//+------------------------------------------------------------------+

int    RSI_Handle = INVALID_HANDLE;                               //--- RSI indicator handle
CTrade obj_Trade;                                                 //--- Trade object for position management
```

We start by including the "Trade" library with " [#include](https://www.mql5.com/en/docs/basis/preprosessor/include) <Trade\\Trade.mqh>" to enable built-in functions for managing positions and orders. Next, we define various [input parameters](https://www.mql5.com/en/docs/basis/variables/inputvariables) grouped by category for user customization. Under "RSI Settings", we set "RSI\_Period" to 14 for the RSI calculation length and "RSI\_Applied" to [PRICE\_CLOSE](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) to base it on closing prices. These are just the default settings. Feel free to customize them. In "Swing Settings", "Swing\_Strength" is set to 5 to determine the bars needed for confirming swing highs and lows, while "Min\_Bars\_Between" and "Max\_Bars\_Between" limit the divergence detection to between 5 and 50 bars, and "Tolerance" at 0.1 allows a small buffer for clean divergence checks, just like we did with the regular version.

For the "Price Divergence Filter" group, "Use\_Price\_Slope\_Filter" defaults to false to optionally enable angle-based filtering, with "Price\_Min\_Slope\_Degrees" at 10.0 and "Price\_Max\_Slope\_Degrees" at 80.0 to define acceptable slope ranges in degrees (min disabled at 0, max at 90). Similarly, the "RSI Divergence Filter" has "Use\_RSI\_Slope\_Filter" as true, with "RSI\_Min\_Slope\_Degrees" at 1.0 and "RSI\_Max\_Slope\_Degrees" at 89.0 for RSI line slopes. The rest of the parameters are identical to the previous regular version, except that we added the degrees filtering option.

Finally, we declare global variables: "RSI\_Handle" initialized to [INVALID\_HANDLE](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants) to store the RSI indicator reference, and "obj\_Trade" as a [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) instance for handling trade operations. We will now need to define the [global variables](https://www.mql5.com/en/docs/basis/variables/global) for swing points and initialize them.

```
//+------------------------------------------------------------------+
//| Swing Variables                                                  |
//+------------------------------------------------------------------+
double   Last_High_Price = 0.0;                                   //--- Last swing high price
datetime Last_High_Time  = 0;                                     //--- Last swing high time
double   Prev_High_Price = 0.0;                                   //--- Previous swing high price
datetime Prev_High_Time  = 0;                                     //--- Previous swing high time
double   Last_Low_Price  = 0.0;                                   //--- Last swing low price
datetime Last_Low_Time   = 0;                                     //--- Last swing low time
double   Prev_Low_Price  = 0.0;                                   //--- Previous swing low price
datetime Prev_Low_Time   = 0;                                     //--- Previous swing low time
double   Last_High_RSI   = 0.0;                                   //--- Last swing high RSI value
double   Prev_High_RSI   = 0.0;                                   //--- Previous swing high RSI value
double   Last_Low_RSI    = 0.0;                                   //--- Last swing low RSI value
double   Prev_Low_RSI    = 0.0;                                   //--- Previous swing low RSI value
```

We continue by declaring a set of global variables under the "Swing Variables" section to store details about the latest and prior swing points for both highs and lows. These include "Last\_High\_Price" and "Last\_High\_Time" for the most recent swing high's price and timestamp, along with "Prev\_High\_Price" and "Prev\_High\_Time" for the one before it. Similarly, for swing lows, we have "Last\_Low\_Price", "Last\_Low\_Time", "Prev\_Low\_Price", and "Prev\_Low\_Time". To link these with indicator data, we add "Last\_High\_RSI" and "Prev\_High\_RSI" for RSI values at those high points, as well as "Last\_Low\_RSI" and "Prev\_Low\_RSI" for the lows. All are initialized to zero to start fresh, enabling us to update and compare them dynamically during runtime for divergence detection. With these, we are all set. We just need to initialize the program, specifically the RSI indicator, and make sure we can reference its window so we can draw on it later, but this time with a degree visualization.

```
//+------------------------------------------------------------------+
//| Expert Initialization Function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   RSI_Handle = iRSI(_Symbol, _Period, RSI_Period, RSI_Applied);  //--- Create RSI indicator handle
   if (RSI_Handle == INVALID_HANDLE) {                            //--- Check if RSI creation failed
      Print("Failed to create RSI indicator");                    //--- Log error
      return(INIT_FAILED);                                        //--- Return initialization failure
   }
   long chart_id = ChartID();                                     //--- Get current chart ID
   string rsi_name = "RSI(" + IntegerToString(RSI_Period) + ")";  //--- Generate RSI indicator name
   int rsi_subwin = ChartWindowFind(chart_id, rsi_name);          //--- Find RSI subwindow
   if (rsi_subwin == -1) {                                        //--- Check if RSI subwindow not found
      if (!ChartIndicatorAdd(chart_id, 1, RSI_Handle)) {          //--- Add RSI to chart subwindow
         Print("Failed to add RSI indicator to chart");           //--- Log error
      }
   }
   obj_Trade.SetExpertMagicNumber(Magic_Number);                  //--- Set magic number for trade object
   Print("RSI Hidden Divergence EA initialized");                 //--- Log initialization success
   return(INIT_SUCCEEDED);                                        //--- Return initialization success
}
```

In the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, we begin by creating the RSI indicator handle with the [iRSI](https://www.mql5.com/en/docs/indicators/irsi) function, passing the current symbol, timeframe, RSI period, and applied price type to set up the oscillator. We then check if "RSI\_Handle" is [INVALID\_HANDLE](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants); if so, we log an error message using [Print](https://www.mql5.com/en/docs/common/print) and return "INIT\_FAILED" to halt initialization. Next, we retrieve the current chart ID with [ChartID](https://www.mql5.com/en/docs/chart_operations/chartid) and construct the RSI indicator name as a string combining "RSI(" with the period converted via the [IntegerToString](https://www.mql5.com/en/docs/convert/integertostring) function.

We attempt to locate the RSI subwindow using [ChartWindowFind](https://www.mql5.com/en/docs/chart_operations/chartwindowfind) with the chart ID and name; if it's not found (rsi\_subwin == -1), we add the indicator to subwindow 1 via [ChartIndicatorAdd](https://www.mql5.com/en/docs/chart_operations/chartindicatoradd), logging an error if that fails. After that, we configure the trade object by calling "obj\_Trade.SetExpertMagicNumber(Magic\_Number)" to associate our unique identifier with trades. Finally, we print a success message and return [INIT\_SUCCEEDED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode) to confirm proper setup. Upon initialization, we get the following outcome.

![INITIAL RUN](https://c.mql5.com/2/178/Screenshot_2025-11-03_151828.png)

Now that we can initialize the program and add the indicator to its subwindow that we can reference, we will need to check and draw the swing points on the chart so that we can use them to identify the divergences or convergences. Let us define some helper functions for that.

```
//+------------------------------------------------------------------+
//| Check for Swing High                                             |
//+------------------------------------------------------------------+
bool CheckSwingHigh(int bar, double& highs[]) {
   if (bar < Swing_Strength || bar + Swing_Strength >= ArraySize(highs)) return false; //--- Return false if bar index out of range for swing strength
   double current = highs[bar];                                   //--- Get current high price
   for (int i = 1; i <= Swing_Strength; i++) {                    //--- Iterate through adjacent bars
      if (highs[bar - i] >= current || highs[bar + i] >= current) return false; //--- Return false if not a swing high
   }
   return true;                                                   //--- Return true if swing high
}
//+------------------------------------------------------------------+
//| Check for Swing Low                                              |
//+------------------------------------------------------------------+
bool CheckSwingLow(int bar, double& lows[]) {
   if (bar < Swing_Strength || bar + Swing_Strength >= ArraySize(lows)) return false; //--- Return false if bar index out of range for swing strength
   double current = lows[bar];                                    //--- Get current low price
   for (int i = 1; i <= Swing_Strength; i++) {                    //--- Iterate through adjacent bars
      if (lows[bar - i] <= current || lows[bar + i] <= current) return false; //--- Return false if not a swing low
   }
   return true;                                                   //--- Return true if swing low
}
```

We define the "CheckSwingHigh" function, which takes an integer bar index and a reference to an array of high prices to determine if a swing high exists at that bar. It first checks if the bar is out of bounds based on "Swing\_Strength" using the [ArraySize](https://www.mql5.com/en/docs/array/arraysize) function to avoid index errors, returning false if so. We then retrieve the current high price and loop from 1 to "Swing\_Strength", verifying that no left or right adjacent bars have highs greater than or equal to the current one; if any do, it returns false, otherwise true to confirm a swing high.

Similarly, we create the "CheckSwingLow" function with the same structure but for low prices, ensuring the bar is in range, getting the current low, and checking in the loop that no adjacent bars have lows less than or equal to the current one, returning true only if it's a valid swing low.

```
//+------------------------------------------------------------------+
//| Check for Clean Divergence                                       |
//+------------------------------------------------------------------+
bool CleanDivergence(double rsi1, double rsi2, int shift1, int shift2, double& rsi_data[], bool bearish) {
   if (shift1 <= shift2) return false;                            //--- Return false if shifts invalid
   for (int b = shift2 + 1; b < shift1; b++) {                    //--- Iterate between shifts
      double interp_factor = (double)(b - shift2) / (shift1 - shift2); //--- Calculate interpolation factor
      double interp_rsi = rsi2 + interp_factor * (rsi1 - rsi2);   //--- Calculate interpolated RSI
      if (bearish) {                                              //--- Check for bearish divergence
         if (rsi_data[b] > interp_rsi + Tolerance) return false;  //--- Return false if RSI exceeds line plus tolerance
      } else {                                                    //--- Check for bullish divergence
         if (rsi_data[b] < interp_rsi - Tolerance) return false;  //--- Return false if RSI below line minus tolerance
      }
   }
   return true;                                                   //--- Return true if divergence is clean
}
//+------------------------------------------------------------------+
//| Calculate Visual Angle                                           |
//+------------------------------------------------------------------+
double CalculateVisualAngle(long chart_id, int sub_window, datetime time1, double val1, datetime time2, double val2) {
   int x1 = 0, y1 = 0, x2 = 0, y2 = 0;                            //--- Initialize pixel coordinates
   bool ok1 = ChartTimePriceToXY(chart_id, sub_window, time1, val1, x1, y1); //--- Convert first point to XY
   bool ok2 = ChartTimePriceToXY(chart_id, sub_window, time2, val2, x2, y2); //--- Convert second point to XY
   if (!ok1 || !ok2 || x1 == x2) return 0.0;                      //--- Return zero if conversion failed or same x
   double dx = (double)(x2 - x1);                                 //--- Calculate delta x
   double dy = (double)(y2 - y1);                                 //--- Calculate delta y
   if (dx == 0.0) return (dy > 0.0 ? -90.0 : 90.0);               //--- Handle vertical line case
   double angle = MathArctan(-dy / dx) * 180.0 / M_PI;            //--- Calculate angle in degrees
   return MathAbs(angle);                                         //--- Return absolute angle
}
```

We implement the "CleanDivergence" function to verify that the divergence line between two RSI points remains uncrossed by intermediate RSI values, ensuring a "clean" pattern without violations. It accepts parameters like "rsi1" and "rsi2" for the RSI values at the swings, "shift1" and "shift2" for their bar shifts (with "shift1" expected to be greater), a reference to the "rsi\_data" array, and a "bearish" boolean to distinguish divergence type. We first validate the shifts, returning false if invalid, then loop through bars between "shift2 + 1" and "shift1 - 1", calculating an "interp\_factor" as the normalized position and an "interp\_rsi" as the linearly interpolated value between "rsi1" and "rsi2". For bearish cases, we check if any "rsi\_data\[b\]" exceeds "interp\_rsi + Tolerance", returning false on breach; for bullish, we ensure no "rsi\_data\[b\]" falls below "interp\_rsi - Tolerance". If all checks pass, we return true to confirm the divergence is clean and reliable for signaling.

Next, we define the "CalculateVisualAngle" function to compute the visual slope angle in degrees between two points on the chart, aiding in divergence filtering. It takes "chart\_id" for the chart identifier, "sub\_window" to specify main or subwindow, along with "time1", "val1", "time2", and "val2" for the coordinates. We initialize pixel variables "x1", "y1", "x2", "y2" to zero, then convert chart time-price to XY pixels using [ChartTimePriceToXY](https://www.mql5.com/en/docs/chart_operations/charttimepricetoxy) for both points, storing success in "ok1" and "ok2". If conversion fails or x-coordinates match, return 0.0; otherwise, calculate "dx" as "x2-x1" and "dy" as "y2-y1". For vertical lines where "dx" is zero, return -90.0 or 90.0 based on "dy" sign; else, compute the "angle" with [MathArctan](https://www.mql5.com/en/docs/math/matharctan) on "-dy / dx", convert to degrees by multiplying by 180.0 over [M\_PI](https://www.mql5.com/en/docs/constants/namedconstants/mathsconstants), and return its absolute value via [MathAbs](https://www.mql5.com/en/docs/math/mathabs) for a positive slope measure. We all all the functions to help us identify the divergences, and thus we can implement that on the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler. We will start with hidden bearish divergence.

```
//+------------------------------------------------------------------+
//| Expert Tick Function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   static datetime last_time = 0;                                 //--- Store last processed time
   datetime current_time = iTime(_Symbol, _Period, 0);            //--- Get current bar time
   if (current_time == last_time) return;                         //--- Exit if bar not new
   last_time = current_time;                                      //--- Update last time
   int data_size = 200;                                           //--- Set data size for analysis
   double high_data[], low_data[], rsi_data[];                    //--- Declare arrays for high, low, RSI data
   datetime time_data[];                                          //--- Declare array for time data
   CopyHigh(_Symbol, _Period, 0, data_size, high_data);           //--- Copy high prices
   CopyLow(_Symbol, _Period, 0, data_size, low_data);             //--- Copy low prices
   CopyTime(_Symbol, _Period, 0, data_size, time_data);           //--- Copy time values
   CopyBuffer(RSI_Handle, 0, 0, data_size, rsi_data);             //--- Copy RSI values
   ArraySetAsSeries(high_data, true);                             //--- Set high data as series
   ArraySetAsSeries(low_data, true);                              //--- Set low data as series
   ArraySetAsSeries(time_data, true);                             //--- Set time data as series
   ArraySetAsSeries(rsi_data, true);                              //--- Set RSI data as series
   long chart_id = ChartID();                                     //--- Get current chart ID
   int rsi_window = ChartWindowFind(chart_id, "RSI(" + IntegerToString(RSI_Period) + ")"); //--- Find RSI subwindow
   // Find latest swing high
   int last_high_bar = -1, prev_high_bar = -1;                    //--- Initialize swing high bars
   for (int b = 1; b < data_size - Swing_Strength; b++) {         //--- Iterate through bars
      if (CheckSwingHigh(b, high_data)) {                         //--- Check for swing high
         if (last_high_bar == -1) {                               //--- Check if first swing high
            last_high_bar = b;                                    //--- Set last high bar
         } else {                                                 //--- Second swing high found
            prev_high_bar = b;                                    //--- Set previous high bar
            break;                                                //--- Exit loop
         }
      }
   }
   if (last_high_bar > 0 && time_data[last_high_bar] > Last_High_Time) { //--- Check new swing high
      Prev_High_Price = Last_High_Price;                          //--- Update previous high price
      Prev_High_Time = Last_High_Time;                            //--- Update previous high time
      Last_High_Price = high_data[last_high_bar];                 //--- Set last high price
      Last_High_Time = time_data[last_high_bar];                  //--- Set last high time
      Prev_High_RSI = Last_High_RSI;                              //--- Update previous high RSI
      Last_High_RSI = rsi_data[last_high_bar];                    //--- Set last high RSI
      string high_type = "H";                                     //--- Set default high type
      if (Prev_High_Price > 0.0) {                                //--- Check if previous high exists
         high_type = (Last_High_Price > Prev_High_Price) ? "HH" : "LH"; //--- Set high type
      }
      bool lower_high = Last_High_Price < Prev_High_Price;        //--- Check for lower high
      bool higher_rsi_high = Last_High_RSI > Prev_High_RSI;       //--- Check for higher RSI high
      int bars_diff = prev_high_bar - last_high_bar;              //--- Calculate bars between highs
      bool hidden_bear_div = false;                               //--- Initialize hidden bearish divergence flag
      double price_angle = 0.0;                                   //--- Initialize price angle
      double rsi_angle = 0.0;                                     //--- Initialize RSI angle
      if (Prev_High_Price > 0.0 && lower_high && higher_rsi_high && bars_diff >= Min_Bars_Between && bars_diff <= Max_Bars_Between) { //--- Check hidden bearish divergence conditions
         if (CleanDivergence(Prev_High_RSI, Last_High_RSI, prev_high_bar, last_high_bar, rsi_data, true)) { //--- Check clean divergence
            price_angle = CalculateVisualAngle(chart_id, 0, Prev_High_Time, Prev_High_Price, Last_High_Time, Last_High_Price); //--- Calculate price angle
            rsi_angle = CalculateVisualAngle(chart_id, rsi_window, Prev_High_Time, Prev_High_RSI, Last_High_Time, Last_High_RSI); //--- Calculate RSI angle
            if ((!Use_Price_Slope_Filter || (price_angle >= Price_Min_Slope_Degrees && price_angle <= Price_Max_Slope_Degrees)) &&
                (!Use_RSI_Slope_Filter || (rsi_angle >= RSI_Min_Slope_Degrees && rsi_angle <= RSI_Max_Slope_Degrees))) { //--- Check slope filters
               hidden_bear_div = true;                               //--- Set hidden bearish divergence flag
               // Draw divergence lines
               string line_name = "DivLine_HiddenBear_" + TimeToString(Last_High_Time); //--- Set divergence line name
               ObjectCreate(chart_id, line_name, OBJ_TREND, 0, Prev_High_Time, Prev_High_Price, Last_High_Time, Last_High_Price); //--- Create trend line for price divergence
               ObjectSetInteger(chart_id, line_name, OBJPROP_COLOR, Bear_Color); //--- Set line color
               ObjectSetInteger(chart_id, line_name, OBJPROP_WIDTH, Line_Width); //--- Set line width
               ObjectSetInteger(chart_id, line_name, OBJPROP_STYLE, Line_Style); //--- Set line style
               ObjectSetInteger(chart_id, line_name, OBJPROP_RAY, false); //--- Disable ray
               ObjectSetInteger(chart_id, line_name, OBJPROP_BACK, false); //--- Set to foreground
               if (rsi_window != -1) {                               //--- Check if RSI subwindow found
                  string rsi_line = "DivLine_RSI_HiddenBear_" + TimeToString(Last_High_Time); //--- Set RSI divergence line name
                  ObjectCreate(chart_id, rsi_line, OBJ_TREND, rsi_window, Prev_High_Time, Prev_High_RSI, Last_High_Time, Last_High_RSI); //--- Create trend line for RSI divergence
                  ObjectSetInteger(chart_id, rsi_line, OBJPROP_COLOR, Bear_Color); //--- Set line color
                  ObjectSetInteger(chart_id, rsi_line, OBJPROP_WIDTH, Line_Width); //--- Set line width
                  ObjectSetInteger(chart_id, rsi_line, OBJPROP_STYLE, Line_Style); //--- Set line style
                  ObjectSetInteger(chart_id, rsi_line, OBJPROP_RAY, false); //--- Disable ray
                  ObjectSetInteger(chart_id, rsi_line, OBJPROP_BACK, false); //--- Set to foreground
               }
            }
         }
      }
      // Draw swing label on price if enabled
      if (Mark_Swings_On_Price) {                                    //--- Check if marking swings on price enabled
         string swing_name = "SwingHigh_" + TimeToString(Last_High_Time); //--- Set swing high label name
         if (ObjectFind(chart_id, swing_name) < 0) {                 //--- Check if label exists
            string high_label_text = " " + high_type + (hidden_bear_div ? " Hidden Bear Div " + DoubleToString(price_angle, 1) + "°" : ""); //--- Set label text with angle if divergence
            ObjectCreate(chart_id, swing_name, OBJ_TEXT, 0, Last_High_Time, Last_High_Price); //--- Create swing high label
            ObjectSetString(chart_id, swing_name, OBJPROP_TEXT, high_label_text); //--- Set label text
            ObjectSetInteger(chart_id, swing_name, OBJPROP_COLOR, Swing_High_Color); //--- Set label color
            ObjectSetInteger(chart_id, swing_name, OBJPROP_ANCHOR, ANCHOR_LEFT_LOWER); //--- Set label anchor
            ObjectSetInteger(chart_id, swing_name, OBJPROP_FONTSIZE, Font_Size); //--- Set label font size
         }
      }
      // Draw corresponding swing label on RSI if enabled
      if (Mark_Swings_On_RSI && rsi_window != -1) {                  //--- Check if marking swings on RSI enabled and subwindow found
         string rsi_swing_name = "RSI_SwingHigh_" + TimeToString(Last_High_Time); //--- Set RSI swing high label name
         if (ObjectFind(chart_id, rsi_swing_name) < 0) {             //--- Check if label exists
            string high_label_text_rsi = " " + high_type + (hidden_bear_div ? " Hidden Bear Div " + DoubleToString(rsi_angle, 1) + "°" : ""); //--- Set label text with RSI angle if divergence
            ObjectCreate(chart_id, rsi_swing_name, OBJ_TEXT, rsi_window, Last_High_Time, Last_High_RSI); //--- Create RSI swing high label
            ObjectSetString(chart_id, rsi_swing_name, OBJPROP_TEXT, high_label_text_rsi); //--- Set label text
            ObjectSetInteger(chart_id, rsi_swing_name, OBJPROP_COLOR, Swing_High_Color); //--- Set label color
            ObjectSetInteger(chart_id, rsi_swing_name, OBJPROP_ANCHOR, ANCHOR_LEFT_LOWER); //--- Set label anchor
            ObjectSetInteger(chart_id, rsi_swing_name, OBJPROP_FONTSIZE, Font_Size); //--- Set label font size
         }
      }
      ChartRedraw(chart_id);                                         //--- Redraw chart
   }
}
```

In the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) function, which executes on every new tick, we first use a static "last\_time" variable to track the timestamp of the last processed bar, retrieving the current bar time with [iTime](https://www.mql5.com/en/docs/series/itime) passing [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol), [\_Period](https://www.mql5.com/en/docs/predefined/_period), and 0, exiting early if unchanged to prevent redundant calculations, then updating "last\_time". We define a "data\_size" of 200 bars for analysis and declare arrays for "high\_data", "low\_data", "rsi\_data", and "time\_data", filling them using [CopyHigh](https://www.mql5.com/en/docs/series/copyhigh), "CopyLow", "CopyTime", and [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer) to obtain recent high prices, low prices, timestamps, and RSI values from "RSI\_Handle".

We set these arrays as series with [ArraySetAsSeries](https://www.mql5.com/en/docs/array/arraysetasseries)" for reverse indexing (recent first) and get the "chart\_id" via "ChartID", along with locating the RSI subwindow index using [ChartWindowFind](https://www.mql5.com/en/docs/chart_operations/chartwindowfind) based on the period-formatted name. To identify the latest swing highs, we initialize "last\_high\_bar" and "prev\_high\_bar" to -1, looping from 1 to "data\_size - Swing\_Strength" and invoking "CheckSwingHigh" to find the two most recent valid swing highs, setting them and breaking after the second.

If a new swing high is detected by comparing "time\_data\[last\_high\_bar\]" to "Last\_High\_Time", we shift previous high variables to hold current last values, update "Last\_High\_Price", "Last\_High\_Time", and "Last\_High\_RSI" from the arrays, and assign "high\_type" as "H" by default, or "HH" for higher high or "LH" for lower high if a previous exists. We then assess hidden bearish divergence: if "Prev\_High\_Price" is set, price shows a lower high, RSI a higher high, and bar difference ("prev\_high\_bar - last\_high\_bar") is within "Min\_Bars\_Between" and "Max\_Bars\_Between", we call "CleanDivergence" with previous and last RSI highs, their shifts, "rsi\_data", and true for bearish.

If clean, we compute "price\_angle" using "CalculateVisualAngle" on the main chart (subwindow 0) with high times and prices, and "rsi\_angle" on the RSI subwindow with high times and RSI values. We verify slope filters: if not using price filter or "price\_angle" is between "Price\_Min\_Slope\_Degrees" and "Price\_Max\_Slope\_Degrees", and similarly for RSI with "Use\_RSI\_Slope\_Filter", "rsi\_angle", "RSI\_Min\_Slope\_Degrees", and "RSI\_Max\_Slope\_Degrees", we set "hidden\_bear\_div" to true, draw divergence lines, and create a trend line named "DivLine\_HiddenBear\_" plus timestamp on the main chart with [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) connecting highs, configuring properties like "OBJPROP\_COLOR" to "Bear\_Color", "OBJPROP\_WIDTH" to "Line\_Width", "OBJPROP\_STYLE" to "Line\_Style", disabling ray and setting foreground via the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) function. If the "rsi\_window" is valid, we similarly draw an RSI line named "DivLine\_RSI\_HiddenBear\_" in the subwindow connecting RSI highs with matching settings.

For price swing labels if "Mark\_Swings\_On\_Price" is true, we check for an existing text object named "SwingHigh\_" plus timestamp with "ObjectFind"; if absent, create it via "ObjectCreate" at the high point, set text with "ObjectSetString" including "high\_type" and " Hidden Bear Div " plus formatted "price\_angle" if divergent, apply "Swing\_High\_Color", anchor "ANCHOR\_LEFT\_LOWER", and "Font\_Size" using "ObjectSetInteger". If "Mark\_Swings\_On\_RSI" is true and a subwindow is found, we do the same for an RSI label named "RSI\_SwingHigh\_" at the RSI high, with text including the "rsi\_angle" if divergent, same color, anchor, and size. We conclude by redrawing the chart with the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/chartredraw) function. Upon compilation, we get the following outcome.

![HIDDEN BEARISH DIVERGENCE](https://c.mql5.com/2/178/Screenshot_2025-11-03_152900.png)

Now that we can identify and visualize the hidden bearish divergence, we need to trade it by opening a sell trade. However, before we open a sell trade, we want to close all the open positions to avoid overtrading. You can choose to keep them if you want. We will need some helper functions for that.

```
//+------------------------------------------------------------------+
//| Open Buy Position                                                |
//+------------------------------------------------------------------+
void OpenBuy() {
   double ask_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);      //--- Get current ask price
   double sl = (SL_Pips > 0) ? NormalizeDouble(ask_price - SL_Pips * _Point, _Digits) : 0; //--- Calculate stop loss if enabled
   double tp = (TP_Pips > 0) ? NormalizeDouble(ask_price + TP_Pips * _Point, _Digits) : 0; //--- Calculate take profit if enabled
   if (obj_Trade.PositionOpen(_Symbol, ORDER_TYPE_BUY, Lot_Size, 0, sl, tp)) { //--- Attempt to open buy position
      Print("Buy trade opened on hidden bullish divergence");     //--- Log buy trade open
   } else {                                                       //--- Handle open failure
      Print("Failed to open Buy: ", obj_Trade.ResultRetcodeDescription()); //--- Log error
   }
}
//+------------------------------------------------------------------+
//| Open Sell Position                                               |
//+------------------------------------------------------------------+
void OpenSell() {
   double bid_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);      //--- Get current bid price
   double sl = (SL_Pips > 0) ? NormalizeDouble(bid_price + SL_Pips * _Point, _Digits) : 0; //--- Calculate stop loss if enabled
   double tp = (TP_Pips > 0) ? NormalizeDouble(bid_price - TP_Pips * _Point, _Digits) : 0; //--- Calculate take profit if enabled
   if (obj_Trade.PositionOpen(_Symbol, ORDER_TYPE_SELL, Lot_Size, 0, sl, tp)) { //--- Attempt to open sell position
      Print("Sell trade opened on hidden bearish divergence");    //--- Log sell trade open
   } else {                                                       //--- Handle open failure
      Print("Failed to open Sell: ", obj_Trade.ResultRetcodeDescription()); //--- Log error
   }
}
//+------------------------------------------------------------------+
//| Close All Positions                                              |
//+------------------------------------------------------------------+
void CloseAll() {
   for (int p = PositionsTotal() - 1; p >= 0; p--) {              //--- Iterate through positions in reverse
      if (PositionGetTicket(p) > 0 && PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == Magic_Number) { //--- Check position details
         obj_Trade.PositionClose(PositionGetTicket(p));           //--- Close position
      }
   }
}
```

We define the "OpenBuy" function to handle opening long positions upon detecting a hidden bullish divergence. We first retrieve the current ask price using [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble), passing [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol) and "SYMBOL\_ASK". Then, if "SL\_Pips" is greater than zero, we calculate the stop loss as "ask\_price - SL\_Pips \* \_Point" normalized to "\_Digits" with [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble); otherwise, set it to zero. Similarly, for take profit, if "TP\_Pips" is positive, we compute "ask\_price + TP\_Pips \* [\_Point](https://www.mql5.com/en/docs/predefined/_point)" normalized, or zero if disabled. We attempt to open the buy position via "obj\_Trade.PositionOpen" passing the symbol, [ORDER\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type), "Lot\_Size", zero slippage, and the calculated sl and tp; if successful, log a message with "Print", otherwise print the error description from "obj\_Trade.ResultRetcodeDescription".

Next, we create the "OpenSell" function for short positions on hidden bearish signals, mirroring the structure: get the bid price with "SymbolInfoDouble" passing "\_Symbol" and "SYMBOL\_BID", set sl as "bid\_price + SL\_Pips \* \_Point" if enabled, tp as "bid\_price - TP\_Pips \* \_Point" if set, and use "obj\_Trade.PositionOpen" with "ORDER\_TYPE\_SELL". Log success or failure similarly.

Finally, we implement the "CloseAll" function to close existing trades before new ones, looping backward from the [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) function minus 1 to zero for safety. For each position, if [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket) with p returns a valid ticket greater than zero and matches our "\_Symbol" via "PositionGetString" with "POSITION\_SYMBOL" and "Magic\_Number" with "PositionGetInteger" using "POSITION\_MAGIC", we close it using "obj\_Trade.PositionClose" on the ticket. We now just need to add the functions to close all positions and open the respective ones. Here is how we call them.

```
// Hidden bearish divergence detected - Sell signal
CloseAll();                                           //--- Close all open positions
OpenSell();                                           //--- Open sell position
```

Upon compilation we get the following outcome.

![SELL TRADE CONFIRMATION](https://c.mql5.com/2/178/Screenshot_2025-11-03_154243.png)

We can see we can trade the bearish divergence signal. We now need to have a similar logic for the bullish divergence as well. Here is the logic we adapted to achieve that.

```
// Find latest swing low
int last_low_bar = -1, prev_low_bar = -1;                      //--- Initialize swing low bars
for (int b = 1; b < data_size - Swing_Strength; b++) {         //--- Iterate through bars
   if (CheckSwingLow(b, low_data)) {                           //--- Check for swing low
      if (last_low_bar == -1) {                                //--- Check if first swing low
         last_low_bar = b;                                     //--- Set last low bar
      } else {                                                 //--- Second swing low found
         prev_low_bar = b;                                     //--- Set previous low bar
         break;                                                //--- Exit loop
      }
   }
}
if (last_low_bar > 0 && time_data[last_low_bar] > Last_Low_Time) { //--- Check new swing low
   Prev_Low_Price = Last_Low_Price;                            //--- Update previous low price
   Prev_Low_Time = Last_Low_Time;                              //--- Update previous low time
   Last_Low_Price = low_data[last_low_bar];                    //--- Set last low price
   Last_Low_Time = time_data[last_low_bar];                    //--- Set last low time
   Prev_Low_RSI = Last_Low_RSI;                                //--- Update previous low RSI
   Last_Low_RSI = rsi_data[last_low_bar];                      //--- Set last low RSI
   string low_type = "L";                                      //--- Set default low type
   if (Prev_Low_Price > 0.0) {                                 //--- Check if previous low exists
      low_type = (Last_Low_Price < Prev_Low_Price) ? "LL" : "HL"; //--- Set low type
   }
   bool higher_low = Last_Low_Price > Prev_Low_Price;          //--- Check for higher low
   bool lower_rsi_low = Last_Low_RSI < Prev_Low_RSI;           //--- Check for lower RSI low
   int bars_diff = prev_low_bar - last_low_bar;                //--- Calculate bars between lows
   bool hidden_bull_div = false;                               //--- Initialize hidden bullish divergence flag
   double price_angle = 0.0;                                   //--- Initialize price angle
   double rsi_angle = 0.0;                                     //--- Initialize RSI angle
   if (Prev_Low_Price > 0.0 && higher_low && lower_rsi_low && bars_diff >= Min_Bars_Between && bars_diff <= Max_Bars_Between) { //--- Check hidden bullish divergence conditions
      if (CleanDivergence(Prev_Low_RSI, Last_Low_RSI, prev_low_bar, last_low_bar, rsi_data, false)) { //--- Check clean divergence
         price_angle = CalculateVisualAngle(chart_id, 0, Prev_Low_Time, Prev_Low_Price, Last_Low_Time, Last_Low_Price); //--- Calculate price angle
         rsi_angle = CalculateVisualAngle(chart_id, rsi_window, Prev_Low_Time, Prev_Low_RSI, Last_Low_Time, Last_Low_RSI); //--- Calculate RSI angle
         if ((!Use_Price_Slope_Filter || (price_angle >= Price_Min_Slope_Degrees && price_angle <= Price_Max_Slope_Degrees)) &&
             (!Use_RSI_Slope_Filter || (rsi_angle >= RSI_Min_Slope_Degrees && rsi_angle <= RSI_Max_Slope_Degrees))) { //--- Check slope filters
            hidden_bull_div = true;                               //--- Set hidden bullish divergence flag
            // Hidden bullish divergence detected - Buy signal
            CloseAll();                                           //--- Close all open positions
            OpenBuy();                                            //--- Open buy position
            // Draw divergence lines
            string line_name = "DivLine_HiddenBull_" + TimeToString(Last_Low_Time); //--- Set divergence line name
            ObjectCreate(chart_id, line_name, OBJ_TREND, 0, Prev_Low_Time, Prev_Low_Price, Last_Low_Time, Last_Low_Price); //--- Create trend line for price divergence
            ObjectSetInteger(chart_id, line_name, OBJPROP_COLOR, Bull_Color); //--- Set line color
            ObjectSetInteger(chart_id, line_name, OBJPROP_WIDTH, Line_Width); //--- Set line width
            ObjectSetInteger(chart_id, line_name, OBJPROP_STYLE, Line_Style); //--- Set line style
            ObjectSetInteger(chart_id, line_name, OBJPROP_RAY, false); //--- Disable ray
            ObjectSetInteger(chart_id, line_name, OBJPROP_BACK, false); //--- Set to foreground
            if (rsi_window != -1) {                               //--- Check if RSI subwindow found
               string rsi_line = "DivLine_RSI_HiddenBull_" + TimeToString(Last_Low_Time); //--- Set RSI divergence line name
               ObjectCreate(chart_id, rsi_line, OBJ_TREND, rsi_window, Prev_Low_Time, Prev_Low_RSI, Last_Low_Time, Last_Low_RSI); //--- Create trend line for RSI divergence
               ObjectSetInteger(chart_id, rsi_line, OBJPROP_COLOR, Bull_Color); //--- Set line color
               ObjectSetInteger(chart_id, rsi_line, OBJPROP_WIDTH, Line_Width); //--- Set line width
               ObjectSetInteger(chart_id, rsi_line, OBJPROP_STYLE, Line_Style); //--- Set line style
               ObjectSetInteger(chart_id, rsi_line, OBJPROP_RAY, false); //--- Disable ray
               ObjectSetInteger(chart_id, rsi_line, OBJPROP_BACK, false); //--- Set to foreground
            }
         }
      }
   }
   // Draw swing label on price if enabled
   if (Mark_Swings_On_Price) {                                    //--- Check if marking swings on price enabled
      string swing_name = "SwingLow_" + TimeToString(Last_Low_Time); //--- Set swing low label name
      if (ObjectFind(chart_id, swing_name) < 0) {                 //--- Check if label exists
         string low_label_text = " " + low_type + (hidden_bull_div ? " Hidden Bull Div " + DoubleToString(price_angle, 1) + "°" : ""); //--- Set label text with angle if divergence
         ObjectCreate(chart_id, swing_name, OBJ_TEXT, 0, Last_Low_Time, Last_Low_Price); //--- Create swing low label
         ObjectSetString(chart_id, swing_name, OBJPROP_TEXT, low_label_text); //--- Set label text
         ObjectSetInteger(chart_id, swing_name, OBJPROP_COLOR, Swing_Low_Color); //--- Set label color
         ObjectSetInteger(chart_id, swing_name, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER); //--- Set label anchor
         ObjectSetInteger(chart_id, swing_name, OBJPROP_FONTSIZE, Font_Size); //--- Set label font size
      }
   }
   // Draw corresponding swing label on RSI if enabled
   if (Mark_Swings_On_RSI && rsi_window != -1) {                  //--- Check if marking swings on RSI enabled and subwindow found
      string rsi_swing_name = "RSI_SwingLow_" + TimeToString(Last_Low_Time); //--- Set RSI swing low label name
      if (ObjectFind(chart_id, rsi_swing_name) < 0) {             //--- Check if label exists
         string low_label_text_rsi = " " + low_type + (hidden_bull_div ? " Hidden Bull Div " + DoubleToString(rsi_angle, 1) + "°" : ""); //--- Set label text with RSI angle if divergence
         ObjectCreate(chart_id, rsi_swing_name, OBJ_TEXT, rsi_window, Last_Low_Time, Last_Low_RSI); //--- Create RSI swing low label
         ObjectSetString(chart_id, rsi_swing_name, OBJPROP_TEXT, low_label_text_rsi); //--- Set label text
         ObjectSetInteger(chart_id, rsi_swing_name, OBJPROP_COLOR, Swing_Low_Color); //--- Set label color
         ObjectSetInteger(chart_id, rsi_swing_name, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER); //--- Set label anchor
         ObjectSetInteger(chart_id, rsi_swing_name, OBJPROP_FONTSIZE, Font_Size); //--- Set label font size
      }
   }
   ChartRedraw(chart_id);                                         //--- Redraw chart
}
```

Here, we just used the same logic as we did with the bearish divergence signal, only with inversed conditions. We added comments to make it self-explanatory. Upon compilation, we have the following outcome.

![BUY TRADE CONFIRMATION](https://c.mql5.com/2/178/Screenshot_2025-11-03_154755.png)

From the image, we can see that we are able to trade the bullish divergences as well. What we now need to do is optimize the profits by adding a trailing stop. We will define a function for that as well. It helps keep the code modular.

```
//+------------------------------------------------------------------+
//| Apply Trailing Stop to Positions                                 |
//+------------------------------------------------------------------+
void ApplyTrailingStop() {
   double point = _Point;                                         //--- Get symbol point value
   for (int i = PositionsTotal() - 1; i >= 0; i--) {              //--- Iterate through positions in reverse
      if (PositionGetTicket(i) > 0) {                             //--- Check valid ticket
         if (PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == Magic_Number) { //--- Check symbol and magic
            double sl = PositionGetDouble(POSITION_SL);           //--- Get current stop loss
            double tp = PositionGetDouble(POSITION_TP);           //--- Get current take profit
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN); //--- Get open price
            ulong ticket = PositionGetInteger(POSITION_TICKET);   //--- Get position ticket
            if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) { //--- Check buy position
               double newSL = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID) - Trailing_Stop_Pips * point, _Digits); //--- Calculate new stop loss
               if (newSL > sl && SymbolInfoDouble(_Symbol, SYMBOL_BID) - openPrice > Min_Profit_To_Trail_Pips * point) { //--- Check trailing conditions
                  obj_Trade.PositionModify(ticket, newSL, tp);    //--- Modify position with new stop loss
               }
            } else if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) { //--- Check sell position
               double newSL = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK) + Trailing_Stop_Pips * point, _Digits); //--- Calculate new stop loss
               if (newSL < sl && openPrice - SymbolInfoDouble(_Symbol, SYMBOL_ASK) > Min_Profit_To_Trail_Pips * point) { //--- Check trailing conditions
                  obj_Trade.PositionModify(ticket, newSL, tp);    //--- Modify position with new stop loss
               }
            }
         }
      }
   }
}
```

We implement the "ApplyTrailingStop" function to dynamically adjust stop losses on open positions once they reach a profitable threshold, helping lock in gains as the market moves favorably. We start by assigning the symbol's point value to "point" using [\_Point](https://www.mql5.com/en/docs/predefined/_point) for pip calculations. Then, we loop backward through all positions from the result of the [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) function minus 1 to zero to safely handle modifications without index shifts.

For each position, if [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket) passing i returns a valid ticket greater than zero, we verify it belongs to our symbol by calling [PositionGetString](https://www.mql5.com/en/docs/trading/positiongetstring) with "POSITION\_SYMBOL" and checking if it equals [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol), and matches our identifier via "PositionGetInteger" with "POSITION\_MAGIC" against "Magic\_Number". We retrieve the current "sl" with [PositionGetDouble](https://www.mql5.com/en/docs/trading/positiongetdouble) passing "POSITION\_SL", "tp" using "PositionGetDouble" with "POSITION\_TP", "openPrice" from "PositionGetDouble" with "POSITION\_PRICE\_OPEN", and the "ticket" via "PositionGetInteger" with "POSITION\_TICKET".

If it's a buy position checked by "PositionGetInteger" with "POSITION\_TYPE" equaling "POSITION\_TYPE\_BUY", we calculate a "newSL" as the current bid obtained from "SymbolInfoDouble" passing "\_Symbol" and "SYMBOL\_BID", minus "Trailing\_Stop\_Pips \* point", normalized to "\_Digits" with "NormalizeDouble". We then test if this "newSL" is greater than the existing "sl" and the profit, calculated as the bid from "SymbolInfoDouble" minus "openPrice", exceeds "Min\_Profit\_To\_Trail\_Pips \* point"; if so, we update the position using "obj\_Trade.PositionModify" with the ticket, new SL, and unchanged TP.

For sell positions where "POSITION\_TYPE" equals [POSITION\_TYPE\_SELL](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type) from "PositionGetInteger", we compute "newSL" as the ask from "SymbolInfoDouble" passing "\_Symbol" and "SYMBOL\_ASK", plus "Trailing\_Stop\_Pips \* point", normalized similarly. We check if "newSL" is less than the current "sl" and profit, derived from "openPrice" minus the ask via "SymbolInfoDouble", surpasses the minimum threshold, then modify accordingly with "obj\_Trade.PositionModify". This ensures trailing only activates on qualifying trades without affecting others. We can now call the function in our tick event handler to do the heavy lifting.

```
//+------------------------------------------------------------------+
//| Expert Tick Function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   if (Enable_Trailing_Stop && PositionsTotal() > 0) {            //--- Check if trailing stop enabled
      ApplyTrailingStop();                                        //--- Apply trailing stop to positions
   }
   static datetime last_time = 0;                                 //--- Store last processed time
   datetime current_time = iTime(_Symbol, _Period, 0);            //--- Get current bar time

   //--- THE REST OF THE LOGIC

}
```

We just call the trailing stop function above on every tick if we have positions open and we get the following outcome.

![TRAILING STOP ACTIVATION](https://c.mql5.com/2/178/Screenshot_2025-11-03_155458.png)

After applying the trailing stop, that is all. We are done. We now need to make sure we delete our chart objects when we remove the expert from the chart to avoid clutter.

```
//+------------------------------------------------------------------+
//| Expert Deinitialization Function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   if (RSI_Handle != INVALID_HANDLE) {                            //--- Check if RSI handle is valid
      long chart_id = ChartID();                                  //--- Get current chart ID
      string rsi_name = "RSI(" + IntegerToString(RSI_Period) + ")"; //--- Generate RSI indicator name
      int rsi_subwin = ChartWindowFind(chart_id, rsi_name);       //--- Find RSI subwindow
      if (rsi_subwin != -1) {                                     //--- Check if RSI subwindow found
         ChartIndicatorDelete(chart_id, rsi_subwin, rsi_name);    //--- Delete RSI indicator from chart
         Print("RSI indicator removed from chart");               //--- Log removal
      }
      IndicatorRelease(RSI_Handle);                               //--- Release RSI handle
   }
   ObjectsDeleteAll(0, "DivLine_");                               //--- Delete all divergence line objects
   ObjectsDeleteAll(0, "SwingHigh_");                             //--- Delete all swing high objects
   ObjectsDeleteAll(0, "SwingLow_");                              //--- Delete all swing low objects
   ObjectsDeleteAll(0, "RSI_SwingHigh_");                         //--- Delete all RSI swing high objects
   ObjectsDeleteAll(0, "RSI_SwingLow_");                          //--- Delete all RSI swing low objects
   Print("RSI Hidden Divergence EA deinitialized");               //--- Log deinitialization
}
```

We define the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) event handler, which accepts a constant integer "reason" parameter indicating the cause of deinitialization, to perform cleanup operations when the Expert Advisor is unloaded from the chart. If "RSI\_Handle" is not equal to [INVALID\_HANDLE](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants), we retrieve the current chart ID using "ChartID" and construct the RSI indicator name by combining "RSI(" with the period converted via the [IntegerToString](https://www.mql5.com/en/docs/convert/IntegerToString) function. We then locate the RSI subwindow with [ChartWindowFind](https://www.mql5.com/en/docs/chart_operations/chartwindowfind) passing the chart ID and name; if found (rsi\_subwin != -1), we remove the indicator from the chart using [ChartIndicatorDelete](https://www.mql5.com/en/docs/chart_operations/chartindicatordelete) with the chart ID, subwindow index, and name, logging the removal. Afterward, we release the indicator resources by calling [IndicatorRelease](https://www.mql5.com/en/docs/series/indicatorrelease) with the RSI handle.

Next, we clear all chart objects by invoking the [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/objectdeleteall) function multiple times: first with subwindow 0 and prefix "DivLine\_" to delete divergence lines, then "SwingHigh\_" for swing high labels, "SwingLow\_" for swing low labels, "RSI\_SwingHigh\_" for RSI high labels, and "RSI\_SwingLow\_" for RSI low labels. Finally, we log the deinitialization completion with "Print" to confirm the RSI Hidden Divergence EA has been properly shut down. Since we have achieved our objectives, the thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/178/Screenshot_2025-11-03_162540.png)

Backtest report:

![REPORT](https://c.mql5.com/2/178/Screenshot_2025-11-03_162553.png)

### Conclusion

In conclusion, we’ve developed a [hidden RSI divergence trading system](https://www.mql5.com/go?link=https://www.tradingsetupsreview.com/rsi-hidden-divergence-pullback-trading-guide/ "https://www.tradingsetupsreview.com/rsi-hidden-divergence-pullback-trading-guide/") in MQL5 that identifies hidden bullish and bearish divergences through swing high and low points with strength confirmation, validates them using clean checks within bar ranges and tolerance, and filters signals with customizable slope angles on price and RSI lines for improved accuracy. The system executes trades with fixed lots, optional stop loss, and take profit in pips, plus trailing stops for dynamic risk management, and provides visual feedback via colored trend lines and labeled swing points with angle displays on both price and RSI charts.

Disclaimer: This article is for educational purposes only. Trading carries significant financial risks, and market volatility may result in losses. Thorough backtesting and careful risk management are crucial before deploying this program in live markets.

With this hidden RSI divergence strategy, you’re equipped to trade continuation signals effectively, ready for further optimization in your trading journey. Happy trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20157.zip "Download all attachments in the single ZIP archive")

[RSI\_Hidden\_Divergence\_Convergence\_EA.mq5](https://www.mql5.com/en/articles/download/20157/RSI_Hidden_Divergence_Convergence_EA.mq5 "Download RSI_Hidden_Divergence_Convergence_EA.mq5")(69.59 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Trading Tools (Part 12): Enhancing the Correlation Matrix Dashboard with Interactivity](https://www.mql5.com/en/articles/20962)
- [Creating Custom Indicators in MQL5 (Part 5): WaveTrend Crossover Evolution Using Canvas for Fog Gradients, Signal Bubbles, and Risk Management](https://www.mql5.com/en/articles/20815)
- [MQL5 Trading Tools (Part 11): Correlation Matrix Dashboard (Pearson, Spearman, Kendall) with Heatmap and Standard Modes](https://www.mql5.com/en/articles/20945)
- [Creating Custom Indicators in MQL5 (Part 4): Smart WaveTrend Crossover with Dual Oscillators](https://www.mql5.com/en/articles/20811)
- [Building AI-Powered Trading Systems in MQL5 (Part 8): UI Polish with Animations, Timing Metrics, and Response Management Tools](https://www.mql5.com/en/articles/20722)
- [Creating Custom Indicators in MQL5 (Part 3): Multi-Gauge Enhancements with Sector and Round Styles](https://www.mql5.com/en/articles/20719)
- [Creating Custom Indicators in MQL5 (Part 2): Building a Gauge-Style RSI Display with Canvas and Needle Mechanics](https://www.mql5.com/en/articles/20632)

**[Go to discussion](https://www.mql5.com/en/forum/499455)**

![Optimizing Long-Term Trades: Engulfing Candles and Liquidity Strategies](https://c.mql5.com/2/179/19756-mastering-high-time-frame-trading-logo.png)[Optimizing Long-Term Trades: Engulfing Candles and Liquidity Strategies](https://www.mql5.com/en/articles/19756)

This is a high-timeframe-based EA that makes long-term analyses, trading decisions, and executions based on higher-timeframe analyses of W1, D1, and MN. This article will explore in detail an EA that is specifically designed for long-term traders who are patient enough to withstand and hold their positions during tumultuous lower time frame price action without changing their bias frequently until take-profit targets are hit.

![Circle Search Algorithm (CSA)](https://c.mql5.com/2/118/Circle_Search_Algorithm__LOGO.png)[Circle Search Algorithm (CSA)](https://www.mql5.com/en/articles/17143)

The article presents a new metaheuristic optimization Circle Search Algorithm (CSA) based on the geometric properties of a circle. The algorithm uses the principle of moving points along tangents to find the optimal solution, combining the phases of global exploration and local exploitation.

![Statistical Arbitrage Through Cointegrated Stocks (Part 7): Scoring System 2](https://c.mql5.com/2/179/20173-statistical-arbitrage-through-logo.png)[Statistical Arbitrage Through Cointegrated Stocks (Part 7): Scoring System 2](https://www.mql5.com/en/articles/20173)

This article describes two additional scoring criteria used for selection of baskets of stocks to be traded in mean-reversion strategies, more specifically, in cointegration based statistical arbitrage. It complements a previous article where liquidity and strength of the cointegration vectors were presented, along with the strategic criteria of timeframe and lookback period, by including the stability of the cointegration vectors and the time to mean reversion (half-time). The article includes the commented results of a backtest with the new filters applied and the files required for its reproduction are also provided.

![MetaTrader 5 Machine Learning Blueprint (Part 5): Sequential Bootstrapping—Debiasing Labels, Improving Returns](https://c.mql5.com/2/177/20059-metatrader-5-machine-learning-logo.png)[MetaTrader 5 Machine Learning Blueprint (Part 5): Sequential Bootstrapping—Debiasing Labels, Improving Returns](https://www.mql5.com/en/articles/20059)

Sequential bootstrapping reshapes bootstrap sampling for financial machine learning by actively avoiding temporally overlapping labels, producing more independent training samples, sharper uncertainty estimates, and more robust trading models. This practical guide explains the intuition, shows the algorithm step‑by‑step, provides optimized code patterns for large datasets, and demonstrates measurable performance gains through simulations and real backtests.

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/20157&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049467280728894476)

![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)

You are missing trading opportunities:

- Free trading apps
- Over 8,000 signals for copying
- Economic news for exploring financial markets

RegistrationLog in

latin characters without spaces

a password will be sent to this email

An error occurred


- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)

You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)

If you do not have an account, please [register](https://www.mql5.com/en/auth_register)

Allow the use of cookies to log in to the MQL5.com website.

Please enable the necessary setting in your browser, otherwise you will not be able to log in.

[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)

- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).