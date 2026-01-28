---
title: Automating Trading Strategies in MQL5 (Part 37): Regular RSI Divergence Convergence with Visual Indicators
url: https://www.mql5.com/en/articles/20031
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:25:18.403576
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/20031&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049120414875100492)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 36)](https://www.mql5.com/en/articles/19674), we developed a Supply and Demand Trading System in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) that identified supply and demand zones through consolidation ranges, validated them with impulsive moves, and traded retests with trend confirmation and customizable risk parameters. In Part 37, we develop a [Regular RSI Divergence Convergence](https://www.mql5.com/go?link=https://www.babypips.com/learn/forex/regular-divergence "https://www.babypips.com/learn/forex/regular-divergence") System with visual indicators. This system detects regular bullish and bearish divergences between price swings and [Relative Strength Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi") (RSI) values, executes trades on signals with optional risk controls, and provides on-chart visualizations for enhanced analysis. We will cover the following topics:

1. [Understanding the Regular RSI Divergence Convergence Strategy](https://www.mql5.com/en/articles/20031#para2)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/20031#para3)
3. [Backtesting](https://www.mql5.com/en/articles/20031#para4)
4. [Conclusion](https://www.mql5.com/en/articles/20031#para5)

By the end, you’ll have a functional MQL5 strategy for trading regular RSI divergences, ready for customization—let’s dive in!

### Understanding the Regular RSI Divergence Convergence Strategy

The [regular RSI divergence convergence strategy](https://www.mql5.com/go?link=https://www.babypips.com/learn/forex/regular-divergence "https://www.babypips.com/learn/forex/regular-divergence") focuses on spotting potential trend reversals by identifying mismatches between price action and the Relative Strength Index (RSI) oscillator, which measures momentum. For bullish divergence, the price forms a lower low while the RSI creates a higher low, signaling weakening bearish momentum and a likely bullish reversal. For bearish divergence, the price makes a higher high, but the RSI shows a lower high, indicating fading bullish momentum and a potential bearish shift.

We use confirmed swing points over a set number of bars to detect these patterns. To ensure a clean divergence, we apply a tolerance that avoids intermediate crossings. Entry is triggered when a bullish divergence is identified—take a long position at the close of the confirmation bar—or when a bearish divergence is spotted—enter a short position at the close of the confirmation bar. Positions are managed with risk controls, including predefined stop and profit targets, as well as dynamic trailing stops. By incorporating these elements, we capitalize on reversal setups. Have a look below at the different setups we could have.

Bullish Divergence Setup:

![REGULAR BULLISH DIVERGENCE](https://c.mql5.com/2/176/Screenshot_2025-10-22_153642.png)

Bearish Divergence Setup:

![REGULAR BEARISH DIVERGENCE](https://c.mql5.com/2/176/Screenshot_2025-10-22_153707.png)

Our plan is to scan for swing highs and lows using a strength confirmation, validate divergences within specified bar ranges and tolerance thresholds, execute automated trades with customizable lot sizing and risk parameters, and add visual aids like colored lines and labels for easy monitoring, building a system for divergence-based trading. In a nutshell, we want to achieve the following:

![STRATEGY BLUEPRINT](https://c.mql5.com/2/176/Screenshot_2025-10-22_144521.png)

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Experts folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some [input parameters](https://www.mql5.com/en/docs/basis/variables/inputvariables) and [global variables](https://www.mql5.com/en/docs/basis/variables/global) that we will use throughout the program.

```
//+------------------------------------------------------------------+
//|                        RSI Regular Divergence Convergence EA.mq5 |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//|                                   https://t.me/Forex_Algo_Trader |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Allan Munene Mutiiria."
#property link "https://t.me/Forex_Algo_Trader"
#property version "1.00"
#property strict
#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Input Parameters                                                 |
//+------------------------------------------------------------------+
input group "RSI Settings"
input int RSI_Period = 14;                          // RSI Period
input ENUM_APPLIED_PRICE RSI_Applied = PRICE_CLOSE; // RSI Applied Price

input group "Swing Settings"
input int Swing_Strength = 5;                       // Bars to confirm swing high/low
input int Min_Bars_Between = 5;                     // Min bars between swings for divergence
input int Max_Bars_Between = 50;                    // Max bars between swings for divergence
input double Tolerance = 0.1;                       // Tolerance for clean divergence check

input group "Trade Settings"
input double Lot_Size = 0.01;                       // Fixed Lot Size
input int Magic_Number = 123456789;                 // Magic Number
input double SL_Pips = 300.0;                       // Stop Loss in Pips (0 to disable)
input double TP_Pips = 300.0;                       // Take Profit in Pips (0 to disable)

input group "Trailing Stop Settings"
input bool Enable_Trailing_Stop = true;             // Enable Trailing Stop
input double Trailing_Stop_Pips = 30.0;             // Trailing Stop in Pips
input double Min_Profit_To_Trail_Pips = 50.0;       // Minimum Profit to Start Trailing in Pips

input group "Visualization"
input color Bull_Color = clrGreen;                  // Bullish Divergence Color
input color Bear_Color = clrRed;                    // Bearish Divergence Color
input color Swing_High_Color = clrRed;              // Color for Swing High Labels
input color Swing_Low_Color = clrGreen;             // Color for Swing Low Labels
input int Line_Width = 2;                           // Divergence Line Width
input ENUM_LINE_STYLE Line_Style = STYLE_SOLID;     // Divergence Line Style
input int Font_Size = 8;                            // Swing Point Font Size

//+------------------------------------------------------------------+
//| Indicator Handles and Trade Object                               |
//+------------------------------------------------------------------+
int RSI_Handle = INVALID_HANDLE;                   //--- RSI indicator handle
CTrade obj_Trade;                                  //--- Trade object for position management
```

We start by including the Trade library with " [#include](https://www.mql5.com/en/docs/basis/preprosessor/include) <Trade\\Trade.mqh>" to enable built-in functions for managing positions and orders. Next, we define various [input parameters](https://www.mql5.com/en/docs/basis/variables/inputvariables) grouped by category for user customization. Under "RSI Settings", we set "RSI\_Period" to 14 for the RSI calculation length and "RSI\_Applied" to [PRICE\_CLOSE](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) to base it on closing prices. In "Swing Settings", "Swing\_Strength" is set to 5 to determine the bars needed for confirming swing highs and lows, while "Min\_Bars\_Between" and "Max\_Bars\_Between" limit the divergence detection to between 5 and 50 bars, and "Tolerance" at 0.1 allows a small buffer for clean divergence checks.

For "Trade Settings", we include "Lot\_Size" at 0.01 for fixed position sizing, "Magic\_Number" as 123456789 to identify our trades, and "SL\_Pips" and "TP\_Pips" both at 300.0 to optionally set stop loss and take profit distances (disabled if zero). The "Trailing Stop Settings" group has "Enable\_Trailing\_Stop" as true to activate dynamic stops, with "Trailing\_Stop\_Pips" at 30.0 for the trailing distance and "Min\_Profit\_To\_Trail\_Pips" at 50.0 as the profit threshold to begin trailing. In "Visualization", we specify colors like "Bull\_Color" as clrGreen for bullish divergences, "Bear\_Color" as clrRed for bearish ones, "Swing\_High\_Color" as clrRed, and "Swing\_Low\_Color" as clrGreen for labels, along with "Line\_Width" at 2, "Line\_Style" as STYLE\_SOLID for divergence lines, and "Font\_Size" at 8 for text labels.

Finally, we declare global variables: "RSI\_Handle" initialized to "INVALID\_HANDLE" to store the RSI indicator reference, and "obj\_Trade" as a [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) instance for handling trade operations. We will now need to define the global variables for swing points and initialize them.

```
//+------------------------------------------------------------------+
//| Swing Variables                                                  |
//+------------------------------------------------------------------+
double Last_High_Price = 0.0;                      //--- Last swing high price
datetime Last_High_Time = 0;                       //--- Last swing high time
double Prev_High_Price = 0.0;                      //--- Previous swing high price
datetime Prev_High_Time = 0;                       //--- Previous swing high time
double Last_Low_Price = 0.0;                       //--- Last swing low price
datetime Last_Low_Time = 0;                        //--- Last swing low time
double Prev_Low_Price = 0.0;                       //--- Previous swing low price
datetime Prev_Low_Time = 0;                        //--- Previous swing low time
double Last_High_RSI = 0.0;                        //--- Last swing high RSI value
double Prev_High_RSI = 0.0;                        //--- Previous swing high RSI value
double Last_Low_RSI = 0.0;                         //--- Last swing low RSI value
double Prev_Low_RSI = 0.0;                         //--- Previous swing low RSI value
```

We continue by declaring a set of global variables under the "Swing Variables" section to store details about the latest and prior swing points for both highs and lows. These include "Last\_High\_Price" and "Last\_High\_Time" for the most recent swing high's price and timestamp, along with "Prev\_High\_Price" and "Prev\_High\_Time" for the one before it. Similarly, for swing lows, we have "Last\_Low\_Price", "Last\_Low\_Time", "Prev\_Low\_Price", and "Prev\_Low\_Time". To link these with indicator data, we add "Last\_High\_RSI" and "Prev\_High\_RSI" for RSI values at those high points, as well as "Last\_Low\_RSI" and "Prev\_Low\_RSI" for the lows. All are initialized to zero to start fresh, enabling us to update and compare them dynamically during runtime for divergence detection. With these, we are all set. We just need to initialize the program, specifically the RSI indicator, and make sure we can reference its window so we can draw on it later.

```
//+------------------------------------------------------------------+
//| Expert Initialization Function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   RSI_Handle = iRSI(_Symbol, _Period, RSI_Period, RSI_Applied); //--- Create RSI indicator handle
   if (RSI_Handle == INVALID_HANDLE) {             //--- Check if RSI creation failed
      Print("Failed to create RSI indicator");     //--- Log error
      return(INIT_FAILED);                         //--- Return initialization failure
   }
   long chart_id = ChartID();                      //--- Get current chart ID
   string rsi_name = "RSI(" + IntegerToString(RSI_Period) + ")"; //--- Generate RSI indicator name
   int rsi_subwin = ChartWindowFind(chart_id, rsi_name); //--- Find RSI subwindow
   if (rsi_subwin == -1) {                         //--- Check if RSI subwindow not found
      if (!ChartIndicatorAdd(chart_id, 1, RSI_Handle)) { //--- Add RSI to chart subwindow
         Print("Failed to add RSI indicator to chart"); //--- Log error
      }
   }
   obj_Trade.SetExpertMagicNumber(Magic_Number);   //--- Set magic number for trade object
   Print("RSI Divergence EA initialized");         //--- Log initialization success
   return(INIT_SUCCEEDED);                         //--- Return initialization success
}
```

In the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, we begin by creating the RSI indicator handle with [iRSI](https://www.mql5.com/en/docs/indicators/irsi), passing the current symbol, timeframe, RSI period, and applied price type to set up the oscillator. We then check if "RSI\_Handle" is "INVALID\_HANDLE"; if so, we log an error message using "Print" and return "INIT\_FAILED" to halt initialization. Next, we retrieve the current chart ID with [ChartID](https://www.mql5.com/en/docs/chart_operations/chartid) and construct the RSI indicator name as a string combining "RSI(" with the period converted via the [IntegerToString](https://www.mql5.com/en/docs/convert/integertostring) function.

We attempt to locate the RSI subwindow using [ChartWindowFind](https://www.mql5.com/en/docs/chart_operations/chartwindowfind) with the chart ID and name; if it's not found ("rsi\_subwin" == -1), we add the indicator to subwindow 1 via [ChartIndicatorAdd](https://www.mql5.com/en/docs/chart_operations/chartindicatoradd), logging an error if that fails. After that, we configure the trade object by calling "obj\_Trade.SetExpertMagicNumber(Magic\_Number)" to associate our unique identifier with trades. Finally, we print a success message and return "INIT\_SUCCEEDED" to confirm proper setup. When we initialize the program, we get the following outcome.

![INITIALIZATION](https://c.mql5.com/2/176/Screenshot_2025-10-22_140339.png)

Now that we can initialize the program and add the indicator to its subwindow that we can reference, we will need to check and draw the swing points on the chart so that we can use them to identify the divergences or convergences. Let us define some helper functions for that.

```
//+------------------------------------------------------------------+
//| Check for Swing High                                             |
//+------------------------------------------------------------------+
bool CheckSwingHigh(int bar, double& highs[]) {
   if (bar < Swing_Strength || bar + Swing_Strength >= ArraySize(highs)) return false; //--- Return false if bar index out of range for swing strength
   double current = highs[bar];                    //--- Get current high price
   for (int i = 1; i <= Swing_Strength; i++) {     //--- Iterate through adjacent bars
      if (highs[bar - i] >= current || highs[bar + i] >= current) return false; //--- Return false if not a swing high
   }
   return true;                                    //--- Return true if swing high
}

//+------------------------------------------------------------------+
//| Check for Swing Low                                              |
//+------------------------------------------------------------------+
bool CheckSwingLow(int bar, double& lows[]) {
   if (bar < Swing_Strength || bar + Swing_Strength >= ArraySize(lows)) return false; //--- Return false if bar index out of range for swing strength
   double current = lows[bar];                     //--- Get current low price
   for (int i = 1; i <= Swing_Strength; i++) {     //--- Iterate through adjacent bars
      if (lows[bar - i] <= current || lows[bar + i] <= current) return false; //--- Return false if not a swing low
   }
   return true;                                    //--- Return true if swing low
}
```

We define the "CheckSwingHigh" function, which takes an integer bar index and a reference to an array of high prices to determine if a swing high exists at that bar. It first checks if the bar is out of bounds based on "Swing\_Strength" using [ArraySize](https://www.mql5.com/en/docs/array/arraysize) to avoid index errors, returning false if so. We then retrieve the current high price and loop from 1 to "Swing\_Strength", verifying that no left or right adjacent bars have highs greater than or equal to the current one; if any do, it returns false, otherwise true to confirm a swing high.

Similarly, we create the "CheckSwingLow" function with the same structure but for low prices, ensuring the bar is in range, getting the current low, and checking in the loop that no adjacent bars have lows less than or equal to the current one, returning true only if it's a valid swing low. We can also define a function for detecting a clean RSI divergence.

```
//+------------------------------------------------------------------+
//| Check for Clean Divergence                                       |
//+------------------------------------------------------------------+
bool CleanDivergence(double rsi1, double rsi2, int shift1, int shift2, double& rsi_data[], bool bearish) {
   if (shift1 <= shift2) return false;             //--- Return false if shifts invalid
   for (int b = shift2 + 1; b < shift1; b++) {    //--- Iterate between shifts
      double interp_factor = (double)(b - shift2) / (shift1 - shift2); //--- Calculate interpolation factor
      double interp_rsi = rsi2 + interp_factor * (rsi1 - rsi2); //--- Calculate interpolated RSI
      if (bearish) {                               //--- Check for bearish divergence
         if (rsi_data[b] > interp_rsi + Tolerance) return false; //--- Return false if RSI exceeds line plus tolerance
      } else {                                     //--- Check for bullish divergence
         if (rsi_data[b] < interp_rsi - Tolerance) return false; //--- Return false if RSI below line minus tolerance
      }
   }
   return true;                                    //--- Return true if divergence is clean
}
```

Here, we implement the "CleanDivergence" function to verify that the divergence line between two RSI points remains uncrossed by intermediate RSI values, ensuring a "clean" pattern without violations. It accepts parameters like "rsi1" and "rsi2" for the RSI values at the swings, "shift1" and "shift2" for their bar shifts (with "shift1" expected to be greater), a reference to the "rsi\_data" array, and a "bearish" boolean to distinguish divergence type.

We first validate the shifts, returning false if invalid, then loop through bars between "shift2 + 1" and "shift1 - 1", calculating an "interp\_factor" as the normalized position and an "interp\_rsi" as the linearly interpolated value between "rsi1" and "rsi2". For bearish cases, we check if any "rsi\_data\[b\]" exceeds "interp\_rsi + Tolerance", returning false on breach; for bullish, we ensure no "rsi\_data\[b\]" falls below "interp\_rsi - Tolerance". If all checks pass, we return true to confirm the divergence is clean and reliable for signaling. We all all the functions to help us identify the divergences, and thus we can implement that on the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler. We will start with bearish divergence.

```
//+------------------------------------------------------------------+
//| Expert Tick Function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   static datetime last_time = 0;                //--- Store last processed time
   datetime current_time = iTime(_Symbol, _Period, 0); //--- Get current bar time
   if (current_time == last_time) return;        //--- Exit if bar not new
   last_time = current_time;                     //--- Update last time
   int data_size = 200;                          //--- Set data size for analysis
   double high_data[], low_data[], rsi_data[];   //--- Declare arrays for high, low, RSI data
   datetime time_data[];                         //--- Declare array for time data
   CopyHigh(_Symbol, _Period, 0, data_size, high_data); //--- Copy high prices
   CopyLow(_Symbol, _Period, 0, data_size, low_data); //--- Copy low prices
   CopyTime(_Symbol, _Period, 0, data_size, time_data); //--- Copy time values
   CopyBuffer(RSI_Handle, 0, 0, data_size, rsi_data); //--- Copy RSI values
   ArraySetAsSeries(high_data, true);            //--- Set high data as series
   ArraySetAsSeries(low_data, true);             //--- Set low data as series
   ArraySetAsSeries(time_data, true);            //--- Set time data as series
   ArraySetAsSeries(rsi_data, true);             //--- Set RSI data as series
   long chart_id = ChartID();                    //--- Get current chart ID
   // Find latest swing high
   int last_high_bar = -1, prev_high_bar = -1;   //--- Initialize swing high bars
   for (int b = 1; b < data_size - Swing_Strength; b++) { //--- Iterate through bars
      if (CheckSwingHigh(b, high_data)) {        //--- Check for swing high
         if (last_high_bar == -1) {              //--- Check if first swing high
            last_high_bar = b;                   //--- Set last high bar
         } else {                                //--- Second swing high found
            prev_high_bar = b;                   //--- Set previous high bar
            break;                               //--- Exit loop
         }
      }
   }
   if (last_high_bar > 0 && time_data[last_high_bar] > Last_High_Time) { //--- Check new swing high
      Prev_High_Price = Last_High_Price;          //--- Update previous high price
      Prev_High_Time = Last_High_Time;            //--- Update previous high time
      Last_High_Price = high_data[last_high_bar]; //--- Set last high price
      Last_High_Time = time_data[last_high_bar];  //--- Set last high time
      Prev_High_RSI = Last_High_RSI;              //--- Update previous high RSI
      Last_High_RSI = rsi_data[last_high_bar];    //--- Set last high RSI
      string high_type = "H";                     //--- Set default high type
      if (Prev_High_Price > 0.0) {                //--- Check if previous high exists
         high_type = (Last_High_Price > Prev_High_Price) ? "HH" : "LH"; //--- Set high type
      }
      bool higher_high = Last_High_Price > Prev_High_Price; //--- Check for higher high
      bool lower_rsi_high = Last_High_RSI < Prev_High_RSI; //--- Check for lower RSI high
      int bars_diff = prev_high_bar - last_high_bar; //--- Calculate bars between highs
      bool bear_div = false;                     //--- Initialize bearish divergence flag
      if (Prev_High_Price > 0.0 && higher_high && lower_rsi_high && bars_diff >= Min_Bars_Between && bars_diff <= Max_Bars_Between) { //--- Check bearish divergence conditions
         if (CleanDivergence(Prev_High_RSI, Last_High_RSI, prev_high_bar, last_high_bar, rsi_data, true)) { //--- Check clean divergence
            bear_div = true;                    //--- Set bearish divergence flag
            // Draw divergence lines
            string line_name = "DivLine_Bear_" + TimeToString(Last_High_Time); //--- Set divergence line name
            ObjectCreate(chart_id, line_name, OBJ_TREND, 0, Prev_High_Time, Prev_High_Price, Last_High_Time, Last_High_Price); //--- Create trend line for price divergence
            ObjectSetInteger(chart_id, line_name, OBJPROP_COLOR, Bear_Color); //--- Set line color
            ObjectSetInteger(chart_id, line_name, OBJPROP_WIDTH, Line_Width); //--- Set line width
            ObjectSetInteger(chart_id, line_name, OBJPROP_STYLE, Line_Style); //--- Set line style
            ObjectSetInteger(chart_id, line_name, OBJPROP_RAY, false); //--- Disable ray
            ObjectSetInteger(chart_id, line_name, OBJPROP_BACK, false); //--- Set to foreground
            int rsi_window = ChartWindowFind(chart_id, "RSI(" + IntegerToString(RSI_Period) + ")"); //--- Find RSI subwindow
            if (rsi_window != -1) {              //--- Check if RSI subwindow found
               string rsi_line = "DivLine_RSI_Bear_" + TimeToString(Last_High_Time); //--- Set RSI divergence line name
               ObjectCreate(chart_id, rsi_line, OBJ_TREND, rsi_window, Prev_High_Time, Prev_High_RSI, Last_High_Time, Last_High_RSI); //--- Create trend line for RSI divergence
               ObjectSetInteger(chart_id, rsi_line, OBJPROP_COLOR, Bear_Color); //--- Set line color
               ObjectSetInteger(chart_id, rsi_line, OBJPROP_WIDTH, Line_Width); //--- Set line width
               ObjectSetInteger(chart_id, rsi_line, OBJPROP_STYLE, Line_Style); //--- Set line style
               ObjectSetInteger(chart_id, rsi_line, OBJPROP_RAY, false); //--- Disable ray
               ObjectSetInteger(chart_id, rsi_line, OBJPROP_BACK, false); //--- Set to foreground
            }
         }
      }
      // Draw swing label
      string swing_name = "SwingHigh_" + TimeToString(Last_High_Time); //--- Set swing high label name
      if (ObjectFind(chart_id, swing_name) < 0) { //--- Check if label exists
         ObjectCreate(chart_id, swing_name, OBJ_TEXT, 0, Last_High_Time, Last_High_Price); //--- Create swing high label
         ObjectSetString(chart_id, swing_name, OBJPROP_TEXT, " " + high_type + (bear_div ? " Bear Div" : "")); //--- Set label text
         ObjectSetInteger(chart_id, swing_name, OBJPROP_COLOR, Swing_High_Color); //--- Set label color
         ObjectSetInteger(chart_id, swing_name, OBJPROP_ANCHOR, ANCHOR_LEFT_LOWER); //--- Set label anchor
         ObjectSetInteger(chart_id, swing_name, OBJPROP_FONTSIZE, Font_Size); //--- Set label font size
      }
      ChartRedraw(chart_id);                    //--- Redraw chart
   }
}
```

In the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler, which runs on every new tick, we use a static "last\_time" variable to track the timestamp of the last processed bar, retrieving the current bar time with [iTime](https://www.mql5.com/en/docs/series/itime) and exiting early if it's unchanged to avoid redundant processing on the same bar, then updating "last\_time". We set a "data\_size" of 200 bars for analysis, which is an arbitrary value we thought, you can increase or decrease as per your liking, and declare arrays for "high\_data", "low\_data", "rsi\_data", and "time\_data", populating them via [CopyHigh](https://www.mql5.com/en/docs/series/copyhigh), "CopyLow", "CopyTime", and [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer) to fetch recent high prices, low prices, timestamps, and RSI values from the handle. We configure these arrays as series with [ArraySetAsSeries](https://www.mql5.com/en/docs/array/arraysetasseries) for time-series indexing and obtain the "chart\_id" using "ChartID" for later object drawing.

To detect the latest swing highs, we initialize "last\_high\_bar" and "prev\_high\_bar" to -1, then loop backward from recent bars (1 to "data\_size - Swing\_Strength"), calling "CheckSwingHigh" to find the two most recent valid swing highs, assigning them accordingly, and breaking after the second. If a new swing high is confirmed by comparing "time\_data\[last\_high\_bar\]" against "Last\_High\_Time", we update the previous high variables with current last values, set the new "Last\_High\_Price", "Last\_High\_Time", "Last\_High\_RSI" from the arrays, and determine the "high\_type" as "H" default, or "HH" for higher high or "LH" for lower high if a previous exists.

We then evaluate bearish divergence conditions: if "Prev\_High\_Price" exists, price shows a higher high, RSI a lower high, and bar difference ("prev\_high\_bar - last\_high\_bar") falls within "Min\_Bars\_Between" and "Max\_Bars\_Between", we invoke "CleanDivergence" with previous and last RSI highs, their shifts, the "rsi\_data" array, and true for bearish mode. If clean, we set the "bear\_div" flag to true and draw visual elements: create a trend line object named "DivLine\_Bear\_" plus timestamp on the main chart with [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) connecting previous and last high points, setting properties like "OBJPROP\_COLOR" to "Bear\_Color", "OBJPROP\_WIDTH" to "Line\_Width", "OBJPROP\_STYLE" to "Line\_Style", disabling ray and setting foreground via the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) function.

We locate the RSI subwindow with "ChartWindowFind" using the period-based name; if found, create a similar RSI divergence line named "DivLine\_RSI\_Bear\_" in that subwindow, connecting RSI values with matching properties. Finally, for the swing label, we check if a text object named "SwingHigh\_" plus timestamp exists using "ObjectFind"; if not, create it with "ObjectCreate" at the high point, set text via "ObjectSetString" to include "high\_type" and " Bear Div" if applicable, apply "Swing\_High\_Color", anchor "ANCHOR\_LEFT\_LOWER", and "Font\_Size" with "ObjectSetInteger", then refresh the chart using the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/chartredraw) function. Upon compilation, we get the following outcome.

![BEARISH DIVERGENCE](https://c.mql5.com/2/176/Screenshot_2025-10-22_142556.png)

Now that we can identify and visualize the bearish divergence, we need to trade it by opening a sell trade. However, before we open a sell trade, we want to close all the open positions to avoid overtrading. You can choose to keep them if you want. We will need some helper functions for that.

```
//+------------------------------------------------------------------+
//| Open Buy Position                                                |
//+------------------------------------------------------------------+
void OpenBuy() {
   double ask_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK); //--- Get current ask price
   double sl = (SL_Pips > 0) ? NormalizeDouble(ask_price - SL_Pips * _Point, _Digits) : 0; //--- Calculate stop loss if enabled
   double tp = (TP_Pips > 0) ? NormalizeDouble(ask_price + TP_Pips * _Point, _Digits) : 0; //--- Calculate take profit if enabled
   if (obj_Trade.PositionOpen(_Symbol, ORDER_TYPE_BUY, Lot_Size, 0, sl, tp)) { //--- Attempt to open buy position
      Print("Buy trade opened on bullish divergence"); //--- Log buy trade open
   } else {                                        //--- Handle open failure
      Print("Failed to open Buy: ", obj_Trade.ResultRetcodeDescription()); //--- Log error
   }
}

//+------------------------------------------------------------------+
//| Open Sell Position                                               |
//+------------------------------------------------------------------+
void OpenSell() {
   double bid_price = SymbolInfoDouble(_Symbol, SYMBOL_BID); //--- Get current bid price
   double sl = (SL_Pips > 0) ? NormalizeDouble(bid_price + SL_Pips * _Point, _Digits) : 0; //--- Calculate stop loss if enabled
   double tp = (TP_Pips > 0) ? NormalizeDouble(bid_price - TP_Pips * _Point, _Digits) : 0; //--- Calculate take profit if enabled
   if (obj_Trade.PositionOpen(_Symbol, ORDER_TYPE_SELL, Lot_Size, 0, sl, tp)) { //--- Attempt to open sell position
      Print("Sell trade opened on bearish divergence"); //--- Log sell trade open
   } else {                                        //--- Handle open failure
      Print("Failed to open Sell: ", obj_Trade.ResultRetcodeDescription()); //--- Log error
   }
}

//+------------------------------------------------------------------+
//| Close All Positions                                              |
//+------------------------------------------------------------------+
void CloseAll() {
   for (int p = PositionsTotal() - 1; p >= 0; p--) { //--- Iterate through positions in reverse
      if (PositionGetTicket(p) > 0 && PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == Magic_Number) { //--- Check position details
         obj_Trade.PositionClose(PositionGetTicket(p)); //--- Close position
      }
   }
}
```

Since we will need to open both buy and sell positions, we create all the helper functions for that. First, we define the "OpenBuy" function to handle opening long positions upon detecting a bullish divergence. We have not yet defined a logic to identify buy signals; we are just leaping forward since we will need it either way. We first retrieve the current ask price using the [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) function. Then, if "SL\_Pips" is greater than zero, we calculate the stop loss as "ask\_price - SL\_Pips \* \_Point" normalized to the symbol's digits with [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble); otherwise, set it to zero. Similarly, for take profit, if "TP\_Pips" is positive, we compute "ask\_price + TP\_Pips \* \_Point" normalized, or zero if disabled. We attempt to open the buy position via "obj\_Trade.PositionOpen" with parameters like the symbol, [ORDER\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type), "Lot\_Size", no slippage, and the calculated sl and tp; if successful, log a message with "Print", otherwise print the error description from "obj\_Trade.ResultRetcodeDescription".

Next, we create the "OpenSell" function for short positions on bearish signals, mirroring the structure: get the bid price with "SymbolInfoDouble(\_Symbol, SYMBOL\_BID)", set sl as "bid\_price + SL\_Pips \* \_Point" if enabled, tp as "bid\_price - TP\_Pips \* \_Point" if set, and use "obj\_Trade.PositionOpen" with "ORDER\_TYPE\_SELL". Log success or failure similarly.

Finally, we implement the "CloseAll" function to shut down existing trades before new ones, looping backward from " [PositionsTotal()](https://www.mql5.com/en/docs/trading/positionstotal) \- 1" to zero for safety. For each position, if "PositionGetTicket(p)" is valid and matches our "\_Symbol" via " [PositionGetString(POSITION\_SYMBOL)](https://www.mql5.com/en/docs/trading/positiongetstring)" and "Magic\_Number" with "PositionGetInteger(POSITION\_MAGIC)", we close it using "obj\_Trade.PositionClose" on the ticket. We now need to add these functions when we have a signal by first closing the positions and then opening the respective signal positions. For the bearish divergence signal, we add the following functions.

```
// Bearish divergence detected - Sell signal
CloseAll();                         //--- Close all open positions
OpenSell();                         //--- Open sell position
```

Opon compilation, we get the following outcome.

![SHORT POSITION](https://c.mql5.com/2/176/Screenshot_2025-10-22_143954.png)

We can see we can trade the bearish divergence signal. We now need to have a similar logic for the bullish divergence as well. Here is the logic we adapted to achieve that.

```
// Find latest swing low
int last_low_bar = -1, prev_low_bar = -1; //--- Initialize swing low bars
for (int b = 1; b < data_size - Swing_Strength; b++) { //--- Iterate through bars
   if (CheckSwingLow(b, low_data)) {      //--- Check for swing low
      if (last_low_bar == -1) {           //--- Check if first swing low
         last_low_bar = b;                //--- Set last low bar
      } else {                            //--- Second swing low found
         prev_low_bar = b;                //--- Set previous low bar
         break;                           //--- Exit loop
      }
   }
}
if (last_low_bar > 0 && time_data[last_low_bar] > Last_Low_Time) { //--- Check new swing low
   Prev_Low_Price = Last_Low_Price;       //--- Update previous low price
   Prev_Low_Time = Last_Low_Time;         //--- Update previous low time
   Last_Low_Price = low_data[last_low_bar]; //--- Set last low price
   Last_Low_Time = time_data[last_low_bar]; //--- Set last low time
   Prev_Low_RSI = Last_Low_RSI;           //--- Update previous low RSI
   Last_Low_RSI = rsi_data[last_low_bar]; //--- Set last low RSI
   string low_type = "L";                 //--- Set default low type
   if (Prev_Low_Price > 0.0) {            //--- Check if previous low exists
      low_type = (Last_Low_Price < Prev_Low_Price) ? "LL" : "HL"; //--- Set low type
   }
   bool lower_low = Last_Low_Price < Prev_Low_Price; //--- Check for lower low
   bool higher_rsi_low = Last_Low_RSI > Prev_Low_RSI; //--- Check for higher RSI low
   int bars_diff = prev_low_bar - last_low_bar; //--- Calculate bars between lows
   bool bull_div = false;                 //--- Initialize bullish divergence flag
   if (Prev_Low_Price > 0.0 && lower_low && higher_rsi_low && bars_diff >= Min_Bars_Between && bars_diff <= Max_Bars_Between) { //--- Check bullish divergence conditions
      if (CleanDivergence(Prev_Low_RSI, Last_Low_RSI, prev_low_bar, last_low_bar, rsi_data, false)) { //--- Check clean divergence
         bull_div = true;                 //--- Set bullish divergence flag
         // Bullish divergence detected - Buy signal
         CloseAll();                      //--- Close all open positions
         OpenBuy();                       //--- Open buy position
         // Draw divergence lines
         string line_name = "DivLine_Bull_" + TimeToString(Last_Low_Time); //--- Set divergence line name
         ObjectCreate(chart_id, line_name, OBJ_TREND, 0, Prev_Low_Time, Prev_Low_Price, Last_Low_Time, Last_Low_Price); //--- Create trend line for price divergence
         ObjectSetInteger(chart_id, line_name, OBJPROP_COLOR, Bull_Color); //--- Set line color
         ObjectSetInteger(chart_id, line_name, OBJPROP_WIDTH, Line_Width); //--- Set line width
         ObjectSetInteger(chart_id, line_name, OBJPROP_STYLE, Line_Style); //--- Set line style
         ObjectSetInteger(chart_id, line_name, OBJPROP_RAY, false); //--- Disable ray
         ObjectSetInteger(chart_id, line_name, OBJPROP_BACK, false); //--- Set to foreground
         int rsi_window = ChartWindowFind(chart_id, "RSI(" + IntegerToString(RSI_Period) + ")"); //--- Find RSI subwindow
         if (rsi_window != -1) {          //--- Check if RSI subwindow found
            string rsi_line = "DivLine_RSI_Bull_" + TimeToString(Last_Low_Time); //--- Set RSI divergence line name
            ObjectCreate(chart_id, rsi_line, OBJ_TREND, rsi_window, Prev_Low_Time, Prev_Low_RSI, Last_Low_Time, Last_Low_RSI); //--- Create trend line for RSI divergence
            ObjectSetInteger(chart_id, rsi_line, OBJPROP_COLOR, Bull_Color); //--- Set line color
            ObjectSetInteger(chart_id, rsi_line, OBJPROP_WIDTH, Line_Width); //--- Set line width
            ObjectSetInteger(chart_id, rsi_line, OBJPROP_STYLE, Line_Style); //--- Set line style
            ObjectSetInteger(chart_id, rsi_line, OBJPROP_RAY, false); //--- Disable ray
            ObjectSetInteger(chart_id, rsi_line, OBJPROP_BACK, false); //--- Set to foreground
         }
      }
   }
   // Draw swing label
   string swing_name = "SwingLow_" + TimeToString(Last_Low_Time); //--- Set swing low label name
   if (ObjectFind(chart_id, swing_name) < 0) { //--- Check if label exists
      ObjectCreate(chart_id, swing_name, OBJ_TEXT, 0, Last_Low_Time, Last_Low_Price); //--- Create swing low label
      ObjectSetString(chart_id, swing_name, OBJPROP_TEXT, " " + low_type + (bull_div ? " Bull Div" : "")); //--- Set label text
      ObjectSetInteger(chart_id, swing_name, OBJPROP_COLOR, Swing_Low_Color); //--- Set label color
      ObjectSetInteger(chart_id, swing_name, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER); //--- Set label anchor
      ObjectSetInteger(chart_id, swing_name, OBJPROP_FONTSIZE, Font_Size); //--- Set label font size
   }
   ChartRedraw(chart_id);                 //--- Redraw chart
}
```

Here, we just used the same logic as we did with the bearish divergence signal, only with inversed conditions. We added comments to make it self-explanatory. Upon compilation, we have the following outcome.

![BULLISH DIVERGENCE POSITION](https://c.mql5.com/2/176/Screenshot_2025-10-22_144521__1.png)

From the image, we can see that we are able to trade the bullish divergences as well. What we now need to do is optimize the profits by adding a trailing stop. We will define a function for that as well. It helps keep the code modular.

```
//+------------------------------------------------------------------+
//| Apply Trailing Stop to Positions                                 |
//+------------------------------------------------------------------+
void ApplyTrailingStop() {
   double point = _Point;                         //--- Get symbol point value
   for (int i = PositionsTotal() - 1; i >= 0; i--) { //--- Iterate through positions in reverse
      if (PositionGetTicket(i) > 0) {             //--- Check valid ticket
         if (PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == Magic_Number) { //--- Check symbol and magic
            double sl = PositionGetDouble(POSITION_SL); //--- Get current stop loss
            double tp = PositionGetDouble(POSITION_TP); //--- Get current take profit
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN); //--- Get open price
            ulong ticket = PositionGetInteger(POSITION_TICKET); //--- Get position ticket
            if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) { //--- Check buy position
               double newSL = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID) - Trailing_Stop_Pips * point, _Digits); //--- Calculate new stop loss
               if (newSL > sl && SymbolInfoDouble(_Symbol, SYMBOL_BID) - openPrice > Min_Profit_To_Trail_Pips * point) { //--- Check trailing conditions
                  obj_Trade.PositionModify(ticket, newSL, tp); //--- Modify position with new stop loss
               }
            } else if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) { //--- Check sell position
               double newSL = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK) + Trailing_Stop_Pips * point, _Digits); //--- Calculate new stop loss
               if (newSL < sl && openPrice - SymbolInfoDouble(_Symbol, SYMBOL_ASK) > Min_Profit_To_Trail_Pips * point) { //--- Check trailing conditions
                  obj_Trade.PositionModify(ticket, newSL, tp); //--- Modify position with new stop loss
               }
            }
         }
      }
   }
}
```

Here, we implement the "ApplyTrailingStop" function to dynamically adjust stop losses on open positions once they reach a profitable threshold, helping lock in gains as the market moves favorably. We start by assigning the symbol's point value to "point" using "\_Point" for pip calculations. Then, we loop backward through all positions from the result of "PositionsTotal" minus 1 to zero to safely handle closures or modifications without index shifts.

For each position, if the [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket) function with parameter i returns a valid ticket greater than zero, we verify it belongs to our symbol by calling [PositionGetString](https://www.mql5.com/en/docs/trading/positiongetstring) with "POSITION\_SYMBOL" and checking if it equals [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol), and matches our identifier via "PositionGetInteger" with "POSITION\_MAGIC" against "Magic\_Number". We retrieve the current "sl" with "PositionGetDouble" passing "POSITION\_SL", "tp" using "PositionGetDouble" with "POSITION\_TP", "openPrice" from "PositionGetDouble" with "POSITION\_PRICE\_OPEN", and the "ticket" via "PositionGetInteger" with "POSITION\_TICKET".

If it's a buy position checked by [PositionGetInteger](https://www.mql5.com/en/docs/trading/positiongetinteger) with "POSITION\_TYPE" equaling [POSITION\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type), we calculate a "newSL" as the current bid obtained from "SymbolInfoDouble" passing "\_Symbol" and "SYMBOL\_BID", minus "Trailing\_Stop\_Pips \* point", normalized to "\_Digits" with "NormalizeDouble". We then test if this "newSL" is greater than the existing "sl" and the profit, calculated as the bid from "SymbolInfoDouble" minus "openPrice", exceeds "Min\_Profit\_To\_Trail\_Pips \* point"; if so, we update the position using "obj\_Trade.PositionModify" with the ticket, new SL, and unchanged TP.

For sell positions where "POSITION\_TYPE" equals [POSITION\_TYPE\_SELL](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type) from "PositionGetInteger", we compute "newSL" as the ask from "SymbolInfoDouble" passing "\_Symbol" and "SYMBOL\_ASK", plus "Trailing\_Stop\_Pips \* point", normalized similarly. We check if "newSL" is less than the current "sl" and profit, derived from "openPrice" minus the ask via "SymbolInfoDouble", surpasses the minimum threshold, then modify accordingly with "obj\_Trade.PositionModify". This ensures trailing only activates on qualifying trades without affecting others. We can now call the function in our tick event handler to do the heavy lifting.

```
if (Enable_Trailing_Stop && PositionsTotal() > 0) {               //--- Check if trailing stop enabled
   ApplyTrailingStop();                   //--- Apply trailing stop to positions
}
```

We just call the trailing stop function above on every tick if we have positions open and we get the following outcome.

![TRAILING STOP](https://c.mql5.com/2/176/Screenshot_2025-10-22_150046.png)

After applying the trailing stop, that is all. We are done. We now need to make sure we delete our chart objects when we remove the expert from the chart to avoid clutter.

```
//+------------------------------------------------------------------+
//| Expert Deinitialization Function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   if (RSI_Handle != INVALID_HANDLE) IndicatorRelease(RSI_Handle); //--- Release RSI handle if valid
   ObjectsDeleteAll(0, "DivLine_");                //--- Delete all divergence line objects
   ObjectsDeleteAll(0, "SwingHigh_");              //--- Delete all swing high objects
   ObjectsDeleteAll(0, "SwingLow_");               //--- Delete all swing low objects
   Print("RSI Divergence EA deinitialized");       //--- Log deinitialization
}
```

In the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) event handler, if "RSI\_Handle" does not equal [INVALID\_HANDLE](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants), we free the indicator's memory by calling [IndicatorRelease](https://www.mql5.com/en/docs/series/indicatorrelease) with "RSI\_Handle" as the argument. Next, we clear all relevant chart objects: using [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/objectdeleteall) with subwindow 0 and the prefix "DivLine\_" to remove divergence lines, then with "SwingHigh\_" for swing high labels, and "SwingLow\_" for swing low labels. Lastly, we output a confirmation message via [Print](https://www.mql5.com/en/docs/common/print) to log that the RSI Divergence EA has been successfully deinitialized. Since we have achieved our objectives, the thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/176/Screenshot_2025-10-22_155549.png)

Backtest report:

![REPORT](https://c.mql5.com/2/176/Screenshot_2025-10-22_155612.png)

### Conclusion

In conclusion, we’ve developed a [regular RSI divergence convergence](https://www.mql5.com/go?link=https://www.babypips.com/learn/forex/regular-divergence "https://www.babypips.com/learn/forex/regular-divergence") system in MQL5 that detects bullish and bearish divergences through swing high and low points with strength confirmation, validates them using clean checks within bar ranges and tolerance, and executes trades with fixed lots, optional stop loss, and take profit in pips, plus trailing stops for dynamic risk management.

Disclaimer: This article is for educational purposes only. Trading carries significant financial risks, and market volatility may result in losses. Thorough backtesting and careful risk management are crucial before deploying this program in live markets.

With this regular RSI divergence strategy, you’re equipped to trade divergence signals effectively, ready for further optimization in your trading journey. Happy trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20031.zip "Download all attachments in the single ZIP archive")

[RSI\_Regular\_Divergence\_Convergence\_EA.mq5](https://www.mql5.com/en/articles/download/20031/RSI_Regular_Divergence_Convergence_EA.mq5 "Download RSI_Regular_Divergence_Convergence_EA.mq5")(25.21 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/498959)**

![Machine Learning Blueprint (Part 4): The Hidden Flaw in Your Financial ML Pipeline — Label Concurrency](https://c.mql5.com/2/175/19850-machine-learning-blueprint-logo.png)[Machine Learning Blueprint (Part 4): The Hidden Flaw in Your Financial ML Pipeline — Label Concurrency](https://www.mql5.com/en/articles/19850)

Discover how to fix a critical flaw in financial machine learning that causes overfit models and poor live performance—label concurrency. When using the triple-barrier method, your training labels overlap in time, violating the core IID assumption of most ML algorithms. This article provides a hands-on solution through sample weighting. You will learn how to quantify temporal overlap between trading signals, calculate sample weights that reflect each observation's unique information, and implement these weights in scikit-learn to build more robust classifiers. Learning these essential techniques will make your trading models more robust, reliable and profitable.

![Building a Smart Trade Manager in MQL5: Automate Break-Even, Trailing Stop, and Partial Close](https://c.mql5.com/2/177/19911-building-a-smart-trade-manager-logo.png)[Building a Smart Trade Manager in MQL5: Automate Break-Even, Trailing Stop, and Partial Close](https://www.mql5.com/en/articles/19911)

Learn how to build a Smart Trade Manager Expert Advisor in MQL5 that automates trade management with break-even, trailing stop, and partial close features. A practical, step-by-step guide for traders who want to save time and improve consistency through automation.

![Black-Scholes Greeks: Gamma and Delta](https://c.mql5.com/2/178/20054-black-scholes-greeks-gamma-logo.png)[Black-Scholes Greeks: Gamma and Delta](https://www.mql5.com/en/articles/20054)

Gamma and Delta measure how an option’s value reacts to changes in the underlying asset’s price. Delta represents the rate of change of the option’s price relative to the underlying, while Gamma measures how Delta itself changes as price moves. Together, they describe an option’s directional sensitivity and convexity—critical for dynamic hedging and volatility-based trading strategies.

![Price Action Analysis Toolkit Development (Part 47): Tracking Forex Sessions and Breakouts in MetaTrader 5](https://c.mql5.com/2/177/19944-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 47): Tracking Forex Sessions and Breakouts in MetaTrader 5](https://www.mql5.com/en/articles/19944)

Global market sessions shape the rhythm of the trading day, and understanding their overlap is vital to timing entries and exits. In this article, we’ll build an interactive trading sessions  EA that brings those global hours to life directly on your chart. The EA automatically plots color‑coded rectangles for the Asia, Tokyo, London, and New York sessions, updating in real time as each market opens or closes. It features on‑chart toggle buttons, a dynamic information panel, and a scrolling ticker headline that streams live status and breakout messages. Tested on different brokers, this EA combines precision with style—helping traders see volatility transitions, identify cross‑session breakouts, and stay visually connected to the global market’s pulse.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=udglvctxhoetquaupwqgklvtycqkciyj&ssn=1769091916183039088&ssn_dr=0&ssn_sr=0&fv_date=1769091916&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F20031&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2037)%3A%20Regular%20RSI%20Divergence%20Convergence%20with%20Visual%20Indicators%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909191681629690&fz_uniq=5049120414875100492&sv=2552)

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).

![close](https://c.mql5.com/i/close.png)

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