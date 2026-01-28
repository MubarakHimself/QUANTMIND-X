---
title: Automating Trading Strategies in MQL5 (Part 39): Statistical Mean Reversion with Confidence Intervals and Dashboard
url: https://www.mql5.com/en/articles/20167
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:32:46.838536
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/20167&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062543330136466476)

MetaTrader 5 / Trading systems


### Introduction

In our [previous article (Part 38)](https://www.mql5.com/en/articles/20157), we developed a [Hidden RSI Divergence Trading](https://www.mql5.com/go?link=https://www.tradingsetupsreview.com/rsi-hidden-divergence-pullback-trading-guide/ "https://www.tradingsetupsreview.com/rsi-hidden-divergence-pullback-trading-guide/") system in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) that identified hidden bullish and bearish divergences using swing points, applied clean checks with bar ranges and tolerance, filtered signals via customizable slope angles on price and RSI lines, executed trades with risk management, and included visual markers with angle displays on charts. In Part 39, we develop a Statistical Mean Reversion system with confidence intervals and a dashboard.

This system analyzes price data over a defined period to compute statistical moments like mean, variance, [skewness](https://en.wikipedia.org/wiki/Skewness "https://en.wikipedia.org/wiki/Skewness"), kurtosis, and [Jarque-Bera](https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test "https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test") statistics, generates reversion signals based on confidence intervals with adaptive thresholds and higher timeframe confirmation, manages trades with equity-based sizing, trailing stops, partial closes, and time-based exits, while providing an on-chart dashboard for real-time monitoring. We will cover the following topics:

1. [Understanding the Statistical Mean Reversion Strategy](https://www.mql5.com/en/articles/20167#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/20167#para2)
3. [Backtesting](https://www.mql5.com/en/articles/20167#para3)
4. [Conclusion](https://www.mql5.com/en/articles/20167#para4)

By the end, you’ll have a functional MQL5 strategy for statistical mean reversion trading, ready for customization—let’s dive in!

### Understanding the Statistical Mean Reversion Strategy

The statistical mean reversion strategy leverages the tendency of prices to revert to their historical mean after significant deviations, enhanced by statistical analysis to identify [non-normal distributions](https://www.mql5.com/go?link=https://z-table.com/normal-vs-non-normal-distribution.html "https://z-table.com/normal-vs-non-normal-distribution.html") where such reversions are more probable due to asymmetry and tail risks.

For a buy setup, the price drops below the lower confidence interval with negative skewness signaling potential upside momentum from oversold conditions; for a sell setup, the price exceeds the upper interval with positive skewness indicating overbought conditions likely to correct downward, both filtered by Jarque-Bera statistics for non-normality confirmation and kurtosis thresholds to avoid excessively [leptokurtic](https://www.mql5.com/go?link=https://www.investopedia.com/terms/l/leptokurtic.asp "https://www.investopedia.com/terms/l/leptokurtic.asp") markets.

When using this strategy, we further refine entries by aligning them with higher timeframes for trend context. We implement dynamic risk controls, such as equity percentage sizing, base stop-loss and take-profit distances, trailing stops for profit locking, partial position closures at predefined profit levels, and time-based exits, to mitigate prolonged exposure. By integrating these statistical and risk elements, we aim to capture reliable reversion opportunities in volatile markets. Have a look below at the statistical distributions expressed diagrammatically that we will be using.

![STATISTICAL REVERSION DIAGRAMS](https://c.mql5.com/2/178/Screenshot_2025-11-04_113601.png)

The [Jarque-Bera](https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test "https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test") moment includes a combination of the skewness and kurtosis moments. We will represent it during implementation, do not worry. Our plan is to compute these statistical moments including mean, variance, skewness, kurtosis, and Jarque-Bera over a set period, generate buy or sell signals when price breaches confidence intervals with adaptive skewness thresholds and non-normality filters, confirm with optional higher timeframe data, execute trades using risk-based or fixed lots with base SL/TP, apply trailing stops, partial closes, and duration limits for management, and display real-time metrics via an on-chart dashboard, creating a comprehensive system for statistical reversion trading. In a nutshell, this is what we will be achieving at the end of the article.

![OBJECTIVE PLAN](https://c.mql5.com/2/178/Screenshot_2025-11-04_113926.png)

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Experts folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some [input parameters](https://www.mql5.com/en/docs/basis/variables/inputvariables) and [global variables](https://www.mql5.com/en/docs/basis/variables/global) that we will use throughout the program.

```
//+------------------------------------------------------------------+
//|                                 StatisticalReversionStrategy.mq5 |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//|                                   https://t.me/Forex_Algo_Trader |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Allan Munene Mutiiria."
#property link      "https://t.me/Forex_Algo_Trader"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Math\Stat\Math.mqh>
#include <ChartObjects\ChartObjectsTxtControls.mqh>

//+------------------------------------------------------------------+
//| Input Parameters                                                 |
//+------------------------------------------------------------------+
input group "=== Statistical Parameters ==="
input int InpPeriod = 50;                   // Period for statistical calculations
input double InpConfidenceLevel = 0.95;     // Confidence level for intervals (0.90-0.99)
input double InpJBThreshold = 2.0;          // Jarque-Bera threshold (lowered for more trades)
input double InpKurtosisThreshold = 5.0;    // Max excess kurtosis (relaxed)
input ENUM_TIMEFRAMES InpHigherTF = 0;      // Higher timeframe for confirmation (0 to disable)

input group "=== Trading Parameters ==="
input double InpRiskPercent = 1.0;          // Risk per trade (% of equity, 0 for fixed lots)
input double InpFixedLots = 0.01;           // Fixed lot size if InpRiskPercent = 0
input int InpBaseStopLossPips = 50;         // Base Stop Loss in pips
input int InpBaseTakeProfitPips = 100;      // Base Take Profit in pips
input int InpMagicNumber = 123456;          // Magic number for trades
input int InpMaxTradeHours = 48;            // Max trade duration in hours (0 to disable)

input group "=== Risk Management ==="
input bool InpUseTrailingStop = true;       // Enable trailing stop
input int InpTrailingStopPips = 30;         // Trailing stop distance in pips
input int InpTrailingStepPips = 10;         // Trailing step in pips
input bool InpUsePartialClose = true;       // Enable partial profit-taking
input double InpPartialClosePercent = 0.5;  // Percent of position to close at 50% TP

input group "=== Dashboard Parameters ==="
input bool InpShowDashboard = true;         // Show dashboard
input int InpDashboardX = 30;               // Dashboard X position
input int InpDashboardY = 30;               // Dashboard Y position
input int InpFontSize = 10;                 // Font size for dashboard text

//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+
CTrade trade;                                      //--- Trade object
datetime g_lastBarTime = 0;                        //--- Last processed bar time
double g_pointMultiplier = 1.0;                    //--- Point multiplier for broker digits
CChartObjectRectLabel* g_dashboardBg = NULL;       //--- Dashboard background object
CChartObjectRectLabel* g_headerBg = NULL;          //--- Header background object
CChartObjectLabel* g_titleLabel = NULL;            //--- Title label object
CChartObjectLabel* g_staticLabels[];               //--- Static labels array
CChartObjectLabel* g_valueLabels[];                //--- Value labels array
string g_staticNames[] = {
   "Symbol:", "Timeframe:", "Price:", "Skewness:", "Jarque-Bera:", "Kurtosis:",
   "Mean:", "Lower CI:", "Upper CI:", "Position:", "Lot Size:", "Profit:", "Duration:", "Signal:",
   "Equity:", "Balance:", "Free Margin:"
};                                                 //--- Static label names
int g_staticCount = ArraySize(g_staticNames);      //--- Number of static labels
```

We begin by including essential libraries: " [#include](https://www.mql5.com/en/docs/basis/preprosessor/include) <Trade\\Trade.mqh>" for trade operations, "#include <Math\\Stat\\Math.mqh>" for statistical calculations like moments and distributions, and "#include <ChartObjects\\ChartObjectsTxtControls.mqh>" to handle chart objects for the dashboard display. Next, we define grouped [input parameters](https://www.mql5.com/en/docs/basis/variables/inputvariables) for user configuration. In the "Statistical Parameters" group, "InpPeriod" defaults to 50 as the lookback for stats, "InpConfidenceLevel" at 0.95 sets the interval confidence (range 0.90-0.99), "InpJBThreshold" at 2.0 filters non-normality via Jarque-Bera (lowered for more signals), "InpKurtosisThreshold" at 5.0 caps excess kurtosis (relaxed threshold), and "InpHigherTF" as 0 disables or selects a higher timeframe for confirmation.

Under "Trading Parameters", "InpRiskPercent" at 1.0 enables equity-based risk (0 for fixed lots), "InpFixedLots" at 0.01 sets static sizing if risk is off, "InpBaseStopLossPips" at 50 and "InpBaseTakeProfitPips" at 100 define base risk/reward distances, "InpMagicNumber" as 123456 identifies trades, and "InpMaxTradeHours" at 48 limits duration (0 to disable). For "Risk Management", "InpUseTrailingStop" as true activates dynamic stops, with "InpTrailingStopPips" at 30 for distance and "InpTrailingStepPips" at 10 for adjustment increments; "InpUsePartialClose" as true enables partial exits, closing "InpPartialClosePercent" at 0.5 of the position at 50% profit.

In "Dashboard Parameters", "InpShowDashboard" as true toggles the visual panel, with "InpDashboardX" and "InpDashboardY" at 30 for positioning, and "InpFontSize" at 10 for text scaling.

We then declare [global variables](https://www.mql5.com/en/docs/basis/variables/global): "trade" as a [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) instance for order handling, "g\_lastBarTime" initialized to 0 to track processed bars, "g\_pointMultiplier" at 1.0 to adjust for broker digit differences, and chart objects like "g\_dashboardBg", "g\_headerBg", "g\_titleLabel" as NULL for dashboard elements, along with arrays "g\_staticLabels\[\]" and "g\_valueLabels\[\]" for labels. We define "g\_staticNames\[\]" as a string array holding fixed dashboard texts like "Symbol:", "Mean:", etc., and set "g\_staticCount" to its size via the [ArraySize](https://www.mql5.com/en/docs/array/arraysize) function for iteration. With that done, we can start with the easiest operation, which is creating the dashboard that we need. Let us have that logic in a function.

```
//+------------------------------------------------------------------+
//| Create Dashboard                                                 |
//+------------------------------------------------------------------+
bool CreateDashboard() {
   if (!InpShowDashboard) return true;             //--- Return if dashboard disabled

   Print("Creating dashboard...");                 //--- Log dashboard creation

// Create main background rectangle
   if (g_dashboardBg == NULL) {                    //--- Check if background exists
      g_dashboardBg = new CChartObjectRectLabel(); //--- Create background object
      if (!g_dashboardBg.Create(0, "StatReversion_DashboardBg", 0, InpDashboardX, InpDashboardY + 30, 300, g_staticCount * (InpFontSize + 6) + 30)) { //--- Create dashboard background
         Print("Error creating dashboard background: ", GetLastError()); //--- Log error
         return false;                             //--- Return failure
      }
      g_dashboardBg.Color(clrDodgerBlue);          //--- Set border color
      g_dashboardBg.BackColor(clrNavy);            //--- Set background color
      g_dashboardBg.BorderType(BORDER_FLAT);       //--- Set border type
      g_dashboardBg.Corner(CORNER_LEFT_UPPER);     //--- Set corner alignment
   }

// Create header background
   if (g_headerBg == NULL) {                       //--- Check if header background exists
      g_headerBg = new CChartObjectRectLabel();    //--- Create header background object
      if (!g_headerBg.Create(0, "StatReversion_HeaderBg", 0, InpDashboardX, InpDashboardY, 300, InpFontSize + 20)) { //--- Create header background
         Print("Error creating header background: ", GetLastError()); //--- Log error
         return false;                             //--- Return failure
      }
      g_headerBg.Color(clrDodgerBlue);             //--- Set border color
      g_headerBg.BackColor(clrDarkBlue);           //--- Set background color
      g_headerBg.BorderType(BORDER_FLAT);          //--- Set border type
      g_headerBg.Corner(CORNER_LEFT_UPPER);        //--- Set corner alignment
   }

// Create title label (centered)
   if (g_titleLabel == NULL) {                     //--- Check if title label exists
      g_titleLabel = new CChartObjectLabel();      //--- Create title label object
      if (!g_titleLabel.Create(0, "StatReversion_Title", 0, InpDashboardX + 75, InpDashboardY + 5)) { //--- Create title label
         Print("Error creating title label: ", GetLastError()); //--- Log error
         return false;                             //--- Return failure
      }
      if (!g_titleLabel.Font("Arial Bold") || !g_titleLabel.FontSize(InpFontSize + 2) || !g_titleLabel.Description("Statistical Reversion")) { //--- Set title properties
         Print("Error setting title properties: ", GetLastError()); //--- Log error
         return false;                             //--- Return failure
      }
      g_titleLabel.Color(clrWhite);                //--- Set title color
   }

// Initialize static labels (left-aligned)
   ArrayFree(g_staticLabels);                      //--- Free static labels array
   ArrayResize(g_staticLabels, g_staticCount);     //--- Resize static labels array
   int y_offset = InpDashboardY + 30 + 10;         //--- Set y offset for labels
   for (int i = 0; i < g_staticCount; i++) {       //--- Iterate through static labels
      g_staticLabels[i] = new CChartObjectLabel(); //--- Create static label object
      string label_name = "StatReversion_Static_" + IntegerToString(i); //--- Generate label name
      if (!g_staticLabels[i].Create(0, label_name, 0, InpDashboardX + 10, y_offset)) { //--- Create static label
         Print("Error creating static label: ", label_name, ", Error: ", GetLastError()); //--- Log error
         DeleteDashboard();                        //--- Delete dashboard
         return false;                             //--- Return failure
      }
      if (!g_staticLabels[i].Font("Arial") || !g_staticLabels[i].FontSize(InpFontSize) || !g_staticLabels[i].Description(g_staticNames[i])) { //--- Set static label properties
         Print("Error setting static label properties: ", label_name, ", Error: ", GetLastError()); //--- Log error
         DeleteDashboard();                        //--- Delete dashboard
         return false;                             //--- Return failure
      }
      g_staticLabels[i].Color(clrLightGray);       //--- Set static label color
      y_offset += InpFontSize + 6;                 //--- Update y offset
   }

// Initialize value labels (right-aligned, starting at center)
   ArrayFree(g_valueLabels);                       //--- Free value labels array
   ArrayResize(g_valueLabels, g_staticCount);      //--- Resize value labels array
   y_offset = InpDashboardY + 30 + 10;             //--- Reset y offset for values
   for (int i = 0; i < g_staticCount; i++) {       //--- Iterate through value labels
      g_valueLabels[i] = new CChartObjectLabel();  //--- Create value label object
      string label_name = "StatReversion_Value_" + IntegerToString(i); //--- Generate label name
      if (!g_valueLabels[i].Create(0, label_name, 0, InpDashboardX + 150, y_offset)) { //--- Create value label
         Print("Error creating value label: ", label_name, ", Error: ", GetLastError()); //--- Log error
         DeleteDashboard();                        //--- Delete dashboard
         return false;                             //--- Return failure
      }
      if (!g_valueLabels[i].Font("Arial") || !g_valueLabels[i].FontSize(InpFontSize) || !g_valueLabels[i].Description("")) { //--- Set value label properties
         Print("Error setting value label properties: ", label_name, ", Error: ", GetLastError()); //--- Log error
         DeleteDashboard();                        //--- Delete dashboard
         return false;                             //--- Return failure
      }
      g_valueLabels[i].Color(clrCyan);             //--- Set value label color
      y_offset += InpFontSize + 6;                 //--- Update y offset
   }

   ChartRedraw();                                  //--- Redraw chart
   Print("Dashboard created successfully");        //--- Log success
   return true;                                    //--- Return true on success
}
```

We define the "CreateDashboard" function to set up an on-chart visual panel for displaying strategy metrics if "InpShowDashboard" is true, otherwise returning true immediately. We log the creation start with [Print](https://www.mql5.com/en/docs/common/print), then check if "g\_dashboardBg" is [NULL](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants) and create a new "CChartObjectRectLabel" instance, using its "Create" method with parameters like subwindow 0, name "StatReversion\_DashboardBg", position based on "InpDashboardX" and "InpDashboardY + 30", width 300, and dynamic height calculated from "g\_staticCount" times ("InpFontSize + 6") plus 30; if creation fails, log the error from "GetLastError" and return false. We set properties such as "Color" to [clrDodgerBlue](https://www.mql5.com/en/docs/constants/objectconstants/webcolors) for the border, "BackColor" to "clrNavy" for the fill, "BorderType" to [BORDER\_FLAT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_border_type), and "Corner" to [CORNER\_LEFT\_UPPER](https://www.mql5.com/en/docs/constants/objectconstants/enum_basecorner).

Similarly, for the header, if "g\_headerBg" is NULL, we create another "CChartObjectRectLabel" at "InpDashboardX" and "InpDashboardY" with height "InpFontSize + 20", applying matching color and border settings, logging, and returning false on failure. For the title, if it is NULL, we instantiate a "CChartObjectLabel" and create it centered at "InpDashboardX + 75" and "InpDashboardY + 5" with name "StatReversion\_Title"; we then configure "Font" to "Arial Bold", "FontSize" to "InpFontSize + 2", "Description" to "Statistical Reversion", and "Color" to "clrWhite", handling errors by logging and returning false.

We free and resize "g\_staticLabels" to "g\_staticCount", setting an initial "y\_offset" as "InpDashboardY + 30 + 10", then loop to create each static label as a new "CChartObjectLabel" with name "StatReversion\_Static\_" plus index, positioned at "InpDashboardX + 10" and current "y\_offset"; set "Font" to "Arial", "FontSize" to "InpFontSize", "Description" from "g\_staticNames\[i\]", and "Color" to "clrLightGray", calling "DeleteDashboard" and returning false on any creation or property error, incrementing "y\_offset" by "InpFontSize + 6" each time.

Likewise, we free and resize "g\_valueLabels", reset "y\_offset", and loop to create value labels named "StatReversion\_Value\_" plus index at "InpDashboardX + 150" and "y\_offset", with empty initial "Description", "clrCyan" color, and same font settings, handling errors similarly. Finally, we invoke [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to refresh the display, log success, and return true. Using the same format now, we can create a function to update and delete the dashboard as below.

```
//+------------------------------------------------------------------+
//| Update Dashboard                                                 |
//+------------------------------------------------------------------+
void UpdateDashboard(double mean, double lower_ci, double upper_ci, double skewness, double jb_stat, double kurtosis,
                     double skew_buy, double skew_sell, string position, double lot_size, double profit, string duration, string signal) {
   if (!InpShowDashboard || ArraySize(g_valueLabels) != g_staticCount) { //--- Check if dashboard enabled and labels valid
      Print("Dashboard update skipped: Not initialized or invalid array size"); //--- Log skip
      return;                                      //--- Exit function
   }

   double balance = AccountInfoDouble(ACCOUNT_BALANCE); //--- Get account balance
   double equity = AccountInfoDouble(ACCOUNT_EQUITY); //--- Get account equity
   double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE); //--- Get free margin
   double price = iClose(_Symbol, _Period, 0);     //--- Get current close price

   string values[] = {
      _Symbol, EnumToString(_Period), DoubleToString(price, _Digits),
      DoubleToString(skewness, 4), DoubleToString(jb_stat, 2), DoubleToString(kurtosis, 2),
      DoubleToString(mean, _Digits), DoubleToString(lower_ci, _Digits), DoubleToString(upper_ci, _Digits),
      position, DoubleToString(lot_size, 2), DoubleToString(profit, 2), duration, signal,
      DoubleToString(equity, 2), DoubleToString(balance, 2), DoubleToString(free_margin, 2)
   };                                              //--- Set value strings

   color value_colors[] = {
      clrWhite, clrWhite, (price > 0 ? clrCyan : clrGray), (skewness != 0 ? clrCyan : clrGray), (jb_stat != 0 ? clrCyan : clrGray), (kurtosis != 0 ? clrCyan : clrGray),
      (mean != 0 ? clrCyan : clrGray), (lower_ci != 0 ? clrCyan : clrGray), (upper_ci != 0 ? clrCyan : clrGray),
      clrWhite, clrWhite, (profit > 0 ? clrLimeGreen : profit < 0 ? clrRed : clrGray), clrWhite,
      (signal == "Buy" ? clrLimeGreen : signal == "Sell" ? clrRed : clrGray),
      (equity > balance ? clrLimeGreen : equity < balance ? clrRed : clrGray), clrWhite, clrWhite
   };                                              //--- Set value colors

   for (int i = 0; i < g_staticCount; i++) {       //--- Iterate through values
      if (g_valueLabels[i] != NULL) {              //--- Check if label exists
         g_valueLabels[i].Description(values[i]);  //--- Set value description
         g_valueLabels[i].Color(value_colors[i]);  //--- Set value color
      } else {                                     //--- Handle null label
         Print("Warning: Value label ", i, " is NULL"); //--- Log warning
      }
   }
   ChartRedraw();                                  //--- Redraw chart
   Print("Dashboard updated: Signal=", signal, ", Position=", position, ", Profit=", profit); //--- Log update
}

//+------------------------------------------------------------------+
//| Delete Dashboard                                                 |
//+------------------------------------------------------------------+
void DeleteDashboard() {
   if (g_dashboardBg != NULL) {                    //--- Check if background exists
      g_dashboardBg.Delete();                      //--- Delete background
      delete g_dashboardBg;                        //--- Free background memory
      g_dashboardBg = NULL;                        //--- Set background to null
      Print("Dashboard background deleted");       //--- Log deletion
   }
   if (g_headerBg != NULL) {                       //--- Check if header background exists
      g_headerBg.Delete();                         //--- Delete header background
      delete g_headerBg;                           //--- Free header background memory
      g_headerBg = NULL;                           //--- Set header background to null
      Print("Header background deleted");          //--- Log deletion
   }
   if (g_titleLabel != NULL) {                     //--- Check if title label exists
      g_titleLabel.Delete();                       //--- Delete title label
      delete g_titleLabel;                         //--- Free title label memory
      g_titleLabel = NULL;                         //--- Set title label to null
      Print("Title label deleted");                //--- Log deletion
   }
   for (int i = 0; i < ArraySize(g_staticLabels); i++) { //--- Iterate through static labels
      if (g_staticLabels[i] != NULL) {             //--- Check if label exists
         g_staticLabels[i].Delete();               //--- Delete static label
         delete g_staticLabels[i];                 //--- Free static label memory
         g_staticLabels[i] = NULL;                 //--- Set static label to null
      }
   }
   for (int i = 0; i < ArraySize(g_valueLabels); i++) { //--- Iterate through value labels
      if (g_valueLabels[i] != NULL) {              //--- Check if label exists
         g_valueLabels[i].Delete();                //--- Delete value label
         delete g_valueLabels[i];                  //--- Free value label memory
         g_valueLabels[i] = NULL;                  //--- Set value label to null
      }
   }
   ArrayFree(g_staticLabels);                      //--- Free static labels array
   ArrayFree(g_valueLabels);                       //--- Free value labels array
   ChartRedraw();                                  //--- Redraw chart
   Print("Dashboard labels cleared");              //--- Log clearance
}
```

We define the "UpdateDashboard" function to refresh the on-chart panel with current strategy data, accepting parameters like "mean", "lower\_ci", "upper\_ci", "skewness", "jb\_stat", "kurtosis", "skew\_buy", "skew\_sell", "position", "lot\_size", "profit", "duration", and "signal" for display values. We first check if we are allowed to show it or label size mismatches the count, logging a skip and returning early if so. We retrieve account details: "balance" via [AccountInfoDouble](https://www.mql5.com/en/docs/account/accountinfodouble) with [ACCOUNT\_BALANCE](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_info_double), same for equity and free margin, and current "price" from the [iClose](https://www.mql5.com/en/docs/series/iclose) function on the symbol, timeframe, and shift 0, for the current.

We populate a "values" string array with formatted data like symbol, timeframe enum via the [EnumToString](https://www.mql5.com/en/docs/convert/enumtostring) function, price with [\_Digits](https://www.mql5.com/en/docs/predefined/_Digits) precision using [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring), statistical metrics rounded appropriately, position info, and account figures. We set a "value\_colors" array for conditional coloring, such as cyan for non-zero stats, lime green for positive profit or red for negative, and similar logic for signal and equity vs. balance. Looping through "g\_staticCount", if each "g\_valueLabels\[i\]" is not NULL, we update its "Description" to "values\[i\]" and "Color" to "value\_colors\[i\]", otherwise log a warning for null labels. We call the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function to refresh visuals and log the update with signal, position, and profit.

Next, we create the "DeleteDashboard" function for cleanup, checking if "g\_dashboardBg" is not NULL to invoke its "Delete" method, free memory with delete, set to NULL, and log deletion; repeat this for "g\_headerBg" and "g\_titleLabel". For arrays, we loop over "g\_staticLabels" size from [ArraySize](https://www.mql5.com/en/docs/array/arraysize), deleting, freeing, and nulling each non-NULL element; do the same for "g\_valueLabels". We then free both arrays with the [ArrayFree](https://www.mql5.com/en/docs/array/ArrayFree) function, redraw the chart, and log that the labels are cleared. We can now call these functions in the initialization event handlers.

```
//+------------------------------------------------------------------+
//| Expert Initialization Function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   trade.SetExpertMagicNumber(InpMagicNumber);     //--- Set magic number
   trade.SetDeviationInPoints(10);                 //--- Set deviation in points
   trade.SetTypeFilling(ORDER_FILLING_FOK);        //--- Set filling type

// Adjust point multiplier for broker digits
   if (_Digits == 5 || _Digits == 3)               //--- Check broker digits
      g_pointMultiplier = 10.0;                    //--- Set multiplier for 5/3 digits
   else                                            //--- Default digits
      g_pointMultiplier = 1.0;                     //--- Set multiplier for others

// Validate inputs (use local variables)
   double confidenceLevel = InpConfidenceLevel;    //--- Copy confidence level
   if (InpConfidenceLevel < 0.90 || InpConfidenceLevel > 0.99) { //--- Check confidence level range
      Print("Warning: InpConfidenceLevel out of range (0.90-0.99). Using 0.95."); //--- Log warning
      confidenceLevel = 0.95;                      //--- Set default confidence level
   }
   double riskPercent = InpRiskPercent;            //--- Copy risk percent
   if (InpRiskPercent < 0 || InpRiskPercent > 10) { //--- Check risk percent range
      Print("Warning: InpRiskPercent out of range (0-10). Using 1.0."); //--- Log warning
      riskPercent = 1.0;                           //--- Set default risk percent
   }

// Initialize dashboard (non-critical)
   if (InpShowDashboard && !CreateDashboard()) {   //--- Check if dashboard creation failed
      Print("Failed to initialize dashboard, continuing without it"); //--- Log failure
   }

   Print("Statistical Reversion Strategy Initialized. Period: ", InpPeriod, ", Confidence: ", confidenceLevel * 100, "% on ", _Symbol, "/", Period()); //--- Log initialization
   return(INIT_SUCCEEDED);                         //--- Return success
}

//+------------------------------------------------------------------+
//| Expert Deinitialization Function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   DeleteDashboard();                              //--- Delete dashboard
   Print("Statistical Reversion Strategy Deinitialized. Reason: ", reason); //--- Log deinitialization
}
```

In the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, we configure the trade object by setting its magic number with "trade.SetExpertMagicNumber" passing "InpMagicNumber", deviation to 10 points via "SetDeviationInPoints", and filling type to [ORDER\_FILLING\_FOK](https://www.mql5.com/en/book/automation/experts/experts_execution_filling) using "SetTypeFilling" for precise order execution. We adjust the "g\_pointMultiplier" based on broker digits: if "\_Digits" is 5 or 3, set it to 10.0 for proper pip scaling; otherwise, keep it at 1.0. To validate inputs, we copy "InpConfidenceLevel" to a local "confidenceLevel" and check if it's outside 0.90 to 0.99, logging a warning and defaulting to 0.95 if so; similarly, for "InpRiskPercent", validate against 0 to 10, warning and resetting to 1.0 if invalid. You could actually set your own. These are just arbitrary ranges that we choose.

If the dashboard is enabled and "CreateDashboard" fails, we log the issue and proceed without it. Finally, we print an initialization message with period, confidence percentage, symbol, and timeframe from "Period", returning [INIT\_SUCCEEDED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode). For the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) event handler, which takes a constant integer "reason", we call "DeleteDashboard" to clean up visuals and log deinitialization with the provided reason. On compilation, we get the following outcome.

![INITIAL RUN](https://c.mql5.com/2/178/Screenshot_2025-11-04_122613.png)

From the image, we can see that we are all good to go. Let us define some helper functions that we will need in the dashboard and statistical updates.

```
//+------------------------------------------------------------------+
//| Normal Inverse CDF Approximation                                 |
//+------------------------------------------------------------------+
double NormalInverse(double p) {
   double t = MathSqrt(-2.0 * MathLog(p < 0.5 ? p : 1.0 - p)); //--- Calculate t value
   double sign = (p < 0.5) ? -1.0 : 1.0;           //--- Determine sign
   return sign * (t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
                  (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)); //--- Return approximated inverse CDF
}

//+------------------------------------------------------------------+
//| Get Position Status                                              |
//+------------------------------------------------------------------+
string GetPositionStatus() {
   if (HasPosition(POSITION_TYPE_BUY)) return "Buy"; //--- Return "Buy" if buy position open
   if (HasPosition(POSITION_TYPE_SELL)) return "Sell"; //--- Return "Sell" if sell position open
   return "None";                                    //--- Return "None" if no position
}

//+------------------------------------------------------------------+
//| Get Current Lot Size                                             |
//+------------------------------------------------------------------+
double GetCurrentLotSize() {
   for (int i = PositionsTotal() - 1; i >= 0; i--) { //--- Iterate through positions
      if (PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == InpMagicNumber) //--- Check symbol and magic
         return PositionGetDouble(POSITION_VOLUME); //--- Return position volume
   }
   return 0.0;                                      //--- Return 0 if no position
}

//+------------------------------------------------------------------+
//| Get Current Profit                                               |
//+------------------------------------------------------------------+
double GetCurrentProfit() {
   for (int i = PositionsTotal() - 1; i >= 0; i--) { //--- Iterate through positions
      if (PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == InpMagicNumber) //--- Check symbol and magic
         return PositionGetDouble(POSITION_PROFIT); //--- Return position profit
   }
   return 0.0;                                      //--- Return 0 if no position
}

//+------------------------------------------------------------------+
//| Get Position Duration                                            |
//+------------------------------------------------------------------+
string GetPositionDuration() {
   for (int i = PositionsTotal() - 1; i >= 0; i--) { //--- Iterate through positions
      if (PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == InpMagicNumber) { //--- Check symbol and magic
         datetime open_time = (datetime)PositionGetInteger(POSITION_TIME); //--- Get open time
         datetime current_time = TimeCurrent(); //--- Get current time
         int hours = (int)((current_time - open_time) / 3600); //--- Calculate hours
         return IntegerToString(hours) + "h"; //--- Return duration string
      }
   }
   return "0h";                                   //--- Return "0h" if no position
}

//+------------------------------------------------------------------+
//| Get Signal Status                                                |
//+------------------------------------------------------------------+
string GetSignalStatus(bool buy_signal, bool sell_signal) {
   if (buy_signal) return "Buy";                  //--- Return "Buy" if buy signal
   if (sell_signal) return "Sell";                //--- Return "Sell" if sell signal
   return "None";                                 //--- Return "None" if no signal
}

//+------------------------------------------------------------------+
//| Check for Open Position of Type                                  |
//+------------------------------------------------------------------+
bool HasPosition(ENUM_POSITION_TYPE pos_type) {
   for (int i = PositionsTotal() - 1; i >= 0; i--) { //--- Iterate through positions
      if (PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == InpMagicNumber && PositionGetInteger(POSITION_TYPE) == pos_type) //--- Check details
         return true;                                //--- Return true if match
   }
   return false;                                     //--- Return false if no match
}

//+------------------------------------------------------------------+
//| Check for Any Open Position                                      |
//+------------------------------------------------------------------+
bool HasPosition() {
   return (HasPosition(POSITION_TYPE_BUY) || HasPosition(POSITION_TYPE_SELL)); //--- Check for buy or sell position
}
```

First, we implement the "NormalInverse" function to approximate the inverse [cumulative distribution function](https://en.wikipedia.org/wiki/Cumulative_distribution_function "https://en.wikipedia.org/wiki/Cumulative_distribution_function") (CDF) for a standard normal distribution, taking a probability "p" and computing a "t" value via [MathSqrt](https://www.mql5.com/en/docs/math/mathsqrt) on the negative log of the adjusted "p" (mirrored if above 0.5), determining a "sign" based on whether "p" is below 0.5, then returning the signed rational approximation using polynomial terms for accuracy in confidence interval calculations.

Next, we define helper functions for position and signal queries. The "GetPositionStatus" function checks for open buys or sells using "HasPosition" with [POSITION\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type) or "POSITION\_TYPE\_SELL", returning the type or "None" if absent. "GetCurrentLotSize" iterates backward through positions from [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) minus 1, verifying symbol with [PositionGetSymbol](https://www.mql5.com/en/docs/trading/positiongetsymbol) and magic via "PositionGetInteger" against "InpMagicNumber", returning the volume from "PositionGetDouble" with "POSITION\_VOLUME" if matched, or 0 otherwise. Similarly, "GetCurrentProfit" retrieves profit using "POSITION\_PROFIT".

For "GetPositionDuration", we loop to find a matching position, get its open time as a datetime from [PositionGetInteger](https://www.mql5.com/en/docs/trading/positiongetinteger) with "POSITION\_TIME", subtract from [TimeCurrent](https://www.mql5.com/en/docs/dateandtime/timecurrent) to compute hours, and format as a string like "Xh", defaulting to "0h" if none. "GetSignalStatus" simply returns "Buy" or "Sell" if the respective boolean is true, else "None". We create "HasPosition" to detect open positions of a specific [ENUM\_POSITION\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type), looping through positions and checking symbol, magic, and type with "PositionGetInteger" for "POSITION\_TYPE", returning true on match. An overload without type checks for any by calling the typed version for buy or sell. We can now use these functions, do the statistical computations, and generate signals. Here is the logic we use to achieve that.

```
//+------------------------------------------------------------------+
//| Expert Tick Function                                             |
//+------------------------------------------------------------------+
void OnTick() {
// Check for new bar to avoid over-calculation
   if (iTime(_Symbol, _Period, 0) == g_lastBarTime) { //--- Check if new bar
      UpdateDashboard(0, 0, 0, 0, 0, 0, 0, 0, GetPositionStatus(), GetCurrentLotSize(), GetCurrentProfit(), GetPositionDuration(), GetSignalStatus(false, false)); //--- Update dashboard with no signal
      return;                                        //--- Exit function
   }
   g_lastBarTime = iTime(_Symbol, _Period, 0);       //--- Update last bar time

// Check market availability
   if (!SymbolInfoDouble(_Symbol, SYMBOL_BID) || !SymbolInfoDouble(_Symbol, SYMBOL_ASK)) { //--- Check market data
      Print("Error: Market data unavailable for ", _Symbol); //--- Log error
      UpdateDashboard(0, 0, 0, 0, 0, 0, 0, 0, GetPositionStatus(), GetCurrentLotSize(), GetCurrentProfit(), GetPositionDuration(), "None"); //--- Update dashboard with no signal
      return;                                        //--- Exit function
   }

// Copy historical close prices
   double prices[];                                  //--- Declare prices array
   ArraySetAsSeries(prices, true);                   //--- Set as series
   int copied = CopyClose(_Symbol, _Period, 1, InpPeriod, prices); //--- Copy close prices
   if (copied != InpPeriod) {                        //--- Check copy success
      Print("Error copying prices: ", copied, ", Error: ", GetLastError()); //--- Log error
      UpdateDashboard(0, 0, 0, 0, 0, 0, 0, 0, GetPositionStatus(), GetCurrentLotSize(), GetCurrentProfit(), GetPositionDuration(), "None"); //--- Update dashboard with no signal
      return;                                        //--- Exit function
   }

// Calculate statistical moments
   double mean, variance, skewness, kurtosis;        //--- Declare statistical variables
   if (!MathMoments(prices, mean, variance, skewness, kurtosis, 0, InpPeriod)) { //--- Calculate moments
      Print("Error calculating moments: ", GetLastError()); //--- Log error
      UpdateDashboard(0, 0, 0, 0, 0, 0, 0, 0, GetPositionStatus(), GetCurrentLotSize(), GetCurrentProfit(), GetPositionDuration(), "None"); //--- Update dashboard with no signal
      return;                                        //--- Exit function
   }

// Jarque-Bera test
   double n = (double)InpPeriod;                     //--- Set sample size
   double jb_stat = n * (skewness * skewness / 6.0 + (kurtosis * kurtosis) / 24.0); //--- Calculate JB statistic

// Log statistical values
   Print("Stats: Skewness=", DoubleToString(skewness, 4), ", JB=", DoubleToString(jb_stat, 2), ", Kurtosis=", DoubleToString(kurtosis, 2)); //--- Log stats

// Adaptive skewness thresholds
   double skew_buy_threshold = -0.3 - 0.05 * kurtosis; //--- Calculate buy skew threshold
   double skew_sell_threshold = 0.3 + 0.05 * kurtosis; //--- Calculate sell skew threshold

// Kurtosis filter
   if (kurtosis > InpKurtosisThreshold) {          //--- Check kurtosis threshold
      Print("Trade skipped: High kurtosis (", kurtosis, ") > ", InpKurtosisThreshold); //--- Log skip
      UpdateDashboard(mean, 0, 0, skewness, jb_stat, kurtosis, skew_buy_threshold, skew_sell_threshold, GetPositionStatus(), GetCurrentLotSize(), GetCurrentProfit(), GetPositionDuration(), "None"); //--- Update dashboard with no signal
      return;                                      //--- Exit function
   }

   double std_dev = MathSqrt(variance);            //--- Calculate standard deviation

// Adaptive confidence interval
   double confidenceLevel = InpConfidenceLevel;    //--- Copy confidence level
   if (confidenceLevel < 0.90 || confidenceLevel > 0.99) //--- Validate confidence level
      confidenceLevel = 0.95;                      //--- Set default confidence level
   double z_score = NormalInverse(0.5 + confidenceLevel / 2.0); //--- Calculate z-score
   double ci_mult = z_score / MathSqrt(n);         //--- Calculate CI multiplier
   double upper_ci = mean + ci_mult * std_dev;     //--- Calculate upper CI
   double lower_ci = mean - ci_mult * std_dev;     //--- Calculate lower CI

// Current close price
   double current_price = iClose(_Symbol, _Period, 0); //--- Get current close price

// Higher timeframe confirmation (if enabled)
   bool htf_valid = true;                          //--- Initialize HTF validity
   if (InpHigherTF != 0) {                         //--- Check if HTF enabled
      double htf_prices[];                         //--- Declare HTF prices array
      ArraySetAsSeries(htf_prices, true);          //--- Set as series
      int htf_copied = CopyClose(_Symbol, InpHigherTF, 1, InpPeriod, htf_prices); //--- Copy HTF close prices
      if (htf_copied != InpPeriod) {               //--- Check HTF copy success
         Print("Error copying HTF prices: ", htf_copied, ", Error: ", GetLastError()); //--- Log error
         UpdateDashboard(mean, lower_ci, upper_ci, skewness, jb_stat, kurtosis, skew_buy_threshold, skew_sell_threshold, GetPositionStatus(), GetCurrentLotSize(), GetCurrentProfit(), GetPositionDuration(), "None"); //--- Update dashboard with no signal
         return;                                   //--- Exit function
      }
      double htf_mean, htf_variance, htf_skewness, htf_kurtosis; //--- Declare HTF stats
      if (!MathMoments(htf_prices, htf_mean, htf_variance, htf_skewness, htf_kurtosis, 0, InpPeriod)) { //--- Calculate HTF moments
         Print("Error calculating HTF moments: ", GetLastError()); //--- Log error
         UpdateDashboard(mean, lower_ci, upper_ci, skewness, jb_stat, kurtosis, skew_buy_threshold, skew_sell_threshold, GetPositionStatus(), GetCurrentLotSize(), GetCurrentProfit(), GetPositionDuration(), "None"); //--- Update dashboard with no signal
         return;                                   //--- Exit function
      }
      htf_valid = (current_price <= htf_mean && skewness <= 0) || (current_price >= htf_mean && skewness >= 0); //--- Check HTF validity
      Print("HTF Check: Price=", DoubleToString(current_price, _Digits), ", HTF Mean=", DoubleToString(htf_mean, _Digits), ", Valid=", htf_valid); //--- Log HTF check
   }

// Generate signals
   bool buy_signal = htf_valid && (current_price < lower_ci) && (skewness < skew_buy_threshold) && (jb_stat > InpJBThreshold); //--- Check buy signal conditions
   bool sell_signal = htf_valid && (current_price > upper_ci) && (skewness > skew_sell_threshold) && (jb_stat > InpJBThreshold); //--- Check sell signal conditions

// Fallback signal
   if (!buy_signal && !sell_signal) {              //--- Check no primary signal
      buy_signal = htf_valid && (current_price < mean - 0.3 * std_dev); //--- Check fallback buy
      sell_signal = htf_valid && (current_price > mean + 0.3 * std_dev); //--- Check fallback sell
      Print("Fallback Signal: Buy=", buy_signal, ", Sell=", sell_signal); //--- Log fallback signals
   }

// Log signal status
   Print("Signal Check: Buy=", buy_signal, ", Sell=", sell_signal, ", Price=", DoubleToString(current_price, _Digits),
         ", LowerCI=", DoubleToString(lower_ci, _Digits), ", UpperCI=", DoubleToString(upper_ci, _Digits),
         ", Skew=", DoubleToString(skewness, 4), ", BuyThresh=", DoubleToString(skew_buy_threshold, 4),
         ", SellThresh=", DoubleToString(skew_sell_threshold, 4), ", JB=", DoubleToString(jb_stat, 2)); //--- Log signal details

}
```

In the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler, we first verify if a new bar has formed by comparing the current bar time from [iTime](https://www.mql5.com/en/docs/series/itime) with [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol), [\_Period](https://www.mql5.com/en/docs/predefined/_period), and shift 0 against "g\_lastBarTime"; if not, we refresh the dashboard with placeholder zeros for stats, current position details from helpers like "GetPositionStatus", and "None" signal via "GetSignalStatus" passing false flags, then exit early. We update "g\_lastBarTime" to the current time and ensure market data availability by checking [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) for both [SYMBOL\_BID](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double) and "SYMBOL\_ASK"; if missing, log an error, update the dashboard similarly, and return. We declare a "prices" array, set it as a series with [ArraySetAsSeries](https://www.mql5.com/en/docs/array/arraysetasseries), and populate it using [CopyClose](https://www.mql5.com/en/docs/series/copyclose) from shift 1 for "InpPeriod" bars; if the copied count doesn't match, log the issue with [GetLastError](https://www.mql5.com/en/docs/check/getlasterror), update the dashboard, and exit.

To compute stats, we declare variables for "mean", "variance", "skewness", and "kurtosis", invoking [MathMoments](https://www.mql5.com/en/docs/standardlibrary/mathematics/stat/array_stat/mathmoments) on the prices array from index 0 to "InpPeriod"; on failure, log error, update dashboard, and return. We calculate the Jarque-Bera statistic as sample size "n" times (skewness squared over 6 plus kurtosis squared over 24), then log the values rounded for readability. Adaptive thresholds are set: "skew\_buy\_threshold" as -0.3 minus 0.05 times kurtosis for buys, and "skew\_sell\_threshold" as 0.3 plus 0.05 times kurtosis for sells. If kurtosis exceeds "InpKurtosisThreshold", we log a skip message, update the dashboard with current stats and "None", and exit to avoid high-tail-risk scenarios.

We derive "std\_dev" from [MathSqrt](https://www.mql5.com/en/docs/math/mathsqrt) of variance, validate "confidenceLevel" within bounds (defaulting to 0.95 if not), compute "z\_score" using "NormalInverse" on half plus half confidence, then "ci\_mult" as z over square root of n, yielding "upper\_ci" and "lower\_ci" as mean plus/minus ci\_mult times std\_dev. Current price is fetched via [iClose](https://www.mql5.com/en/docs/series/iclose) at shift 0. For higher timeframe confirmation if "InpHigherTF" is set, we copy HTF prices similarly, calculate its moments, and set "htf\_valid" true if price aligns with mean relative to skewness sign (below mean with negative skew or above with positive), logging the check; default to true if disabled.

Signals are generated: "buy\_signal" if HTF valid, price below lower CI, skewness under buy threshold, and JB above "InpJBThreshold"; "sell\_signal" mirrored for upper CI and sell threshold. If no primary signal, fallback to HTF-valid price breaches of mean minus/plus 0.3 std\_dev for buy/sell, logging these. We conclude the segment by logging detailed signal conditions, including price, CIs, skewness with thresholds, and JB. Upon compilation, we get the following outcome.

![SIGNAL CONFIRMATION](https://c.mql5.com/2/178/Screenshot_2025-11-04_125413.png)

From the image, we can see that we can do the computations and determine the signal threshold. Now that's done. We need to act on the signals. We want to close the existing positions, though, when we have new signals. Here is the logic we use to achieve that in a function.

```
//+------------------------------------------------------------------+
//| Close All Positions of Type                                      |
//+------------------------------------------------------------------+
void CloseAllPositions(ENUM_POSITION_TYPE pos_type) {
   for (int i = PositionsTotal() - 1; i >= 0; i--) { //--- Iterate through positions
      if (PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == InpMagicNumber && PositionGetInteger(POSITION_TYPE) == pos_type) { //--- Check details
         ulong ticket = PositionGetInteger(POSITION_TICKET); //--- Get ticket
         double profit = PositionGetDouble(POSITION_PROFIT); //--- Get profit
         trade.PositionClose(ticket);                //--- Close position
      }
   }
}
```

We define the "CloseAllPositions" function to shut down all open positions of a specified "ENUM\_POSITION\_TYPE", like buy or sell, ensuring we clear opposite trades before new signals. We loop backward from "PositionsTotal" minus 1 to zero for safe iteration without index issues during closures. For each, if "PositionGetSymbol" matches "\_Symbol", "PositionGetInteger" with "POSITION\_MAGIC" equals "InpMagicNumber", and type via "POSITION\_TYPE" aligns with "pos\_type", we retrieve the "ticket" using "PositionGetInteger" on "POSITION\_TICKET" and profit from "PositionGetDouble" with "POSITION\_PROFIT" (just in case you want to log and see the profit), then execute "trade.PositionClose" on the ticket to close it. We can now close the opposite trades and open new positions using the following logic.

```
// Position management: Close opposite positions
if (HasPosition(POSITION_TYPE_BUY) && sell_signal) //--- Check buy position with sell signal
   CloseAllPositions(POSITION_TYPE_BUY);        //--- Close all buys
if (HasPosition(POSITION_TYPE_SELL) && buy_signal) //--- Check sell position with buy signal
   CloseAllPositions(POSITION_TYPE_SELL);       //--- Close all sells

// Calculate lot size
double lot_size = InpFixedLots;                 //--- Set default lot size
double riskPercent = InpRiskPercent;            //--- Copy risk percent
if (riskPercent > 0) {                          //--- Check if risk percent enabled
   double account_equity = AccountInfoDouble(ACCOUNT_EQUITY); //--- Get account equity
   double sl_points = InpBaseStopLossPips * g_pointMultiplier; //--- Calculate SL points
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE); //--- Get tick value
   if (tick_value == 0) {                       //--- Check invalid tick value
      Print("Error: Invalid tick value for ", _Symbol); //--- Log error
      return;                                   //--- Exit function
   }
   lot_size = NormalizeDouble((account_equity * riskPercent / 100.0) / (sl_points * tick_value), 2); //--- Calculate risk-based lot size
   lot_size = MathMax(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN), MathMin(lot_size, SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX))); //--- Clamp lot size
   Print("Lot Size: Equity=", account_equity, ", SL Points=", sl_points, ", Tick Value=", tick_value, ", Lot=", lot_size); //--- Log lot size calculation
}

// Open new positions
if (!HasPosition() && buy_signal) {             //--- Check no position and buy signal
   double sl = current_price - InpBaseStopLossPips * _Point * g_pointMultiplier; //--- Calculate SL
   double tp = current_price + InpBaseTakeProfitPips * _Point * g_pointMultiplier; //--- Calculate TP
   if (trade.Buy(lot_size, _Symbol, 0, sl, tp, "StatReversion Buy: Skew=" + DoubleToString(skewness, 4) + ", JB=" + DoubleToString(jb_stat, 2))) { //--- Open buy order
      Print("Buy order opened. Mean: ", DoubleToString(mean, 5), ", Current: ", DoubleToString(current_price, 5)); //--- Log buy open
   } else {                                     //--- Handle buy open failure
      Print("Buy order failed: ", GetLastError()); //--- Log error
   }
} else if (!HasPosition() && sell_signal) {     //--- Check no position and sell signal
   double sl = current_price + InpBaseStopLossPips * _Point * g_pointMultiplier; //--- Calculate SL
   double tp = current_price - InpBaseTakeProfitPips * _Point * g_pointMultiplier; //--- Calculate TP
   if (trade.Sell(lot_size, _Symbol, 0, sl, tp, "StatReversion Sell: Skew=" + DoubleToString(skewness, 4) + ", JB=" + DoubleToString(jb_stat, 2))) { //--- Open sell order
      Print("Sell order opened. Mean: ", DoubleToString(mean, 5), ", Current: ", DoubleToString(current_price, 5)); //--- Log sell open
   } else {                                     //--- Handle sell open failure
      Print("Sell order failed: ", GetLastError()); //--- Log error
   }
}

// Update dashboard
UpdateDashboard(mean, lower_ci, upper_ci, skewness, jb_stat, kurtosis, skew_buy_threshold, skew_sell_threshold,
                GetPositionStatus(), lot_size, GetCurrentProfit(), GetPositionDuration(), GetSignalStatus(buy_signal, sell_signal)); //--- Update dashboard with signals
```

We manage positions by checking for opposites: if a buy is open and a sell signal emerges, we call "CloseAllPositions" with [POSITION\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type) to exit all longs; conversely for sells on a buy signal, ensuring no conflicting trades before new entries. For sizing, we default "lot\_size" to "InpFixedLots", but if "InpRiskPercent" is positive, we compute it dynamically from account equity via [AccountInfoDouble](https://www.mql5.com/en/docs/account/accountinfodouble) with [ACCOUNT\_EQUITY](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_info_double), SL distance adjusted by "g\_pointMultiplier", and tick value from [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) using [SYMBOL\_TRADE\_TICK\_VALUE](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double)—exiting with a log if invalid. The formula normalizes equity times risk percent over (SL points times tick value) to two decimals, then clamps between symbol's min and max volume with [MathMax](https://www.mql5.com/en/docs/math/mathmax) and [MathMin](https://www.mql5.com/en/docs/math/mathmin), logging the details.

If no position exists and a buy signal is active, we derive SL below current price by base pips times [\_Point](https://www.mql5.com/en/docs/predefined/_point) and multiplier, TP above similarly, then attempt "trade.Buy" with calculated lot, symbol, no price (market), SL, TP, and a comment including skewness and JB stats; log success with mean vs. current price or failure with the [GetLastError](https://www.mql5.com/en/docs/check/getlasterror) function. Mirror this for sells, adjusting SL above and TP below. Finally, we refresh the dashboard by passing computed stats like mean, CIs, skewness, JB, kurtosis, thresholds, position status, lot, profit, duration, and signal to the "UpdateDashboard" function, which now should be complete with all the information. Upon compilation, we get the following outcome.

![TRADE CONFIRMATION](https://c.mql5.com/2/178/Screenshot_2025-11-04_131643.png)

Now that we can open the positions, we need to manage them, specifically by trailing, taking partial closes, and closing them based on the time limit to prevent over-exposure. All these are just add-ons that you can skip through or condition whichever you don't want. Let us have them in functions.

```
//+------------------------------------------------------------------+
//| Manage Trailing Stop                                             |
//+------------------------------------------------------------------+
void ManageTrailingStop() {
   for (int i = PositionsTotal() - 1; i >= 0; i--) { //--- Iterate through positions
      if (PositionGetSymbol(i) != _Symbol || PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) //--- Check symbol and magic
         continue;                                   //--- Skip if not match

      ulong ticket = PositionGetInteger(POSITION_TICKET); //--- Get ticket
      double current_sl = PositionGetDouble(POSITION_SL); //--- Get current SL
      ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE); //--- Get position type
      double current_price = (pos_type == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK); //--- Get current price

      double trail_distance = InpTrailingStopPips * _Point * g_pointMultiplier; //--- Calculate trail distance
      double trail_step = InpTrailingStepPips * _Point * g_pointMultiplier; //--- Calculate trail step

      if (pos_type == POSITION_TYPE_BUY) {            //--- Check buy position
         double new_sl = current_price - trail_distance; //--- Calculate new SL
         if (new_sl > current_sl + trail_step || current_sl == 0) { //--- Check if update needed
            trade.PositionModify(ticket, new_sl, PositionGetDouble(POSITION_TP)); //--- Modify position
         }
      } else if (pos_type == POSITION_TYPE_SELL) {    //--- Check sell position
         double new_sl = current_price + trail_distance; //--- Calculate new SL
         if (new_sl < current_sl - trail_step || current_sl == 0) { //--- Check if update needed
            trade.PositionModify(ticket, new_sl, PositionGetDouble(POSITION_TP)); //--- Modify position
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Manage Partial Close                                             |
//+------------------------------------------------------------------+
void ManagePartialClose() {
   for (int i = PositionsTotal() - 1; i >= 0; i--) { //--- Iterate through positions
      if (PositionGetSymbol(i) != _Symbol || PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) //--- Check symbol and magic
         continue;                                   //--- Skip if not match

      ulong ticket = PositionGetInteger(POSITION_TICKET); //--- Get ticket
      double open_price = PositionGetDouble(POSITION_PRICE_OPEN); //--- Get open price
      double tp = PositionGetDouble(POSITION_TP); //--- Get TP
      double current_price = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK); //--- Get current price
      double volume = PositionGetDouble(POSITION_VOLUME); //--- Get position volume

      double half_tp_distance = MathAbs(tp - open_price) * 0.5; //--- Calculate half TP distance
      bool should_close = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY && current_price >= open_price + half_tp_distance) ||
                          (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL && current_price <= open_price - half_tp_distance); //--- Check if at half TP

      if (should_close) {                             //--- Check if partial close needed
         double close_volume = NormalizeDouble(volume * InpPartialClosePercent, 2); //--- Calculate close volume
         if (close_volume >= SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN)) { //--- Check minimum volume
            trade.PositionClosePartial(ticket, close_volume); //--- Close partial position
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Manage Time-Based Exit                                           |
//+------------------------------------------------------------------+
void ManageTimeBasedExit() {
   if (InpMaxTradeHours == 0) return;                //--- Exit if no max duration

   for (int i = PositionsTotal() - 1; i >= 0; i--) { //--- Iterate through positions
      if (PositionGetSymbol(i) != _Symbol || PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) //--- Check symbol and magic
         continue;                                   //--- Skip if not match

      ulong ticket = PositionGetInteger(POSITION_TICKET); //--- Get ticket
      datetime open_time = (datetime)PositionGetInteger(POSITION_TIME); //--- Get open time
      datetime current_time = TimeCurrent();         //--- Get current time
      if ((current_time - open_time) / 3600 >= InpMaxTradeHours) { //--- Check if duration exceeded
         double profit = PositionGetDouble(POSITION_PROFIT); //--- Get profit
         trade.PositionClose(ticket);                //--- Close position
      }
   }
}
```

Here, we implement the "ManageTrailingStop" function to dynamically adjust stop losses on open positions as prices move favorably, looping backward through positions from [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) minus 1 to zero. For matches on symbol and magic, we retrieve the ticket, current stop loss, type as [ENUM\_POSITION\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type) from [PositionGetInteger](https://www.mql5.com/en/docs/trading/positiongetinteger) on "POSITION\_TYPE", and relevant price (bid for buys or ask for sells). We calculate trail distance and step using input pips times "\_Point" and multiplier, then for buys, propose a new level below current price by distance, updating via "trade.PositionModify" if it's above current stop loss plus step or if unset; mirror for sells, moving stop loss above if below current minus step.

Next, the "ManagePartialClose" function handles profit-taking by closing a portion halfway to take profit (TP), iterating similarly to find qualifying positions. We get a ticket, open price with [PositionGetDouble](https://www.mql5.com/en/docs/trading/positiongetdouble) on [POSITION\_PRICE\_OPEN](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_double), TP, current price based on type, and volume via "POSITION\_VOLUME"; compute half TP distance as absolute TP minus open times 0.5, checking if price has reached it (above for buys, below for sells). If so, normalize a close volume as position size times "InpPartialClosePercent" to two decimals, and if at or above symbol min volume from [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) with [SYMBOL\_VOLUME\_MIN](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double), execute "trade.PositionClosePartial" on the ticket and volume.

For "ManageTimeBasedExit", we return early if "InpMaxTradeHours" is zero, otherwise loop to identify matching positions, fetching ticket, open time as datetime from "PositionGetInteger" with "POSITION\_TIME", and comparing against [TimeCurrent](https://www.mql5.com/en/docs/dateandtime/timecurrent) to see if hours elapsed meet or exceed the limit. If yes, note profit, and close with "trade.PositionClose" on the ticket, enforcing duration caps. We can now call these functions to do the position management.

```
if (InpUseTrailingStop)                      //--- Check trailing stop enabled
   ManageTrailingStop();                     //--- Manage trailing stop
if (InpUsePartialClose)                      //--- Check partial close enabled
   ManagePartialClose();                     //--- Manage partial close
ManageTimeBasedExit();                       //--- Manage time-based exit
```

When we compile, we get the following outcome.

![POSITIONS MANAGEMENT](https://c.mql5.com/2/178/Screenshot_2025-11-04_132954.png)

We can see we modify and trail the favourable positions actively. Since we have achieved our objectives, the thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/179/Screenshot_2025-11-04_223717.png)

Backtest report:

![REPORT](https://c.mql5.com/2/179/Screenshot_2025-11-04_223730.png)

### Conclusion

In conclusion, we’ve developed a statistical mean reversion system in MQL5 that analyzes price data for moments like mean, variance, [skewness](https://en.wikipedia.org/wiki/Skewness "https://en.wikipedia.org/wiki/Skewness"), kurtosis, and [Jarque-Bera](https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test "https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test") statistics, generates signals based on confidence interval breaches with adaptive thresholds and optional higher timeframe confirmation, executes trades with equity-based or fixed sizing, and incorporates trailing stops, partial closures, and time-based exits for comprehensive risk control, all enhanced by a real-time on-chart dashboard.

Disclaimer: This article is for educational purposes only. Trading carries significant financial risks, and market volatility may result in losses. Thorough backtesting and careful risk management are crucial before deploying this program in live markets.

With this statistical mean reversion strategy, you’re equipped to capitalize on non-normal distribution opportunities, ready for further optimization in your trading journey. Happy trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20167.zip "Download all attachments in the single ZIP archive")

[a\_\_Statistical\_Reversion\_Strategy\_EA.mq5](https://www.mql5.com/en/articles/download/20167/a__Statistical_Reversion_Strategy_EA.mq5 "Download a__Statistical_Reversion_Strategy_EA.mq5")(40.25 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/499847)**

![Developing a Trading Strategy: The Triple Sine Mean Reversion Method](https://c.mql5.com/2/180/20220-developing-a-trading-strategy-logo.png)[Developing a Trading Strategy: The Triple Sine Mean Reversion Method](https://www.mql5.com/en/articles/20220)

This article introduces the Triple Sine Mean Reversion Method, a trading strategy built upon a new mathematical indicator — the Triple Sine Oscillator (TSO). The TSO is derived from the sine cube function, which oscillates between –1 and +1, making it suitable for identifying overbought and oversold market conditions. Overall, the study demonstrates how mathematical functions can be transformed into practical trading tools.

![How can century-old functions update your trading strategies?](https://c.mql5.com/2/120/How_100-Year-Old_Features_Can_Update_Your_Trading_Strategies__LOGO.png)[How can century-old functions update your trading strategies?](https://www.mql5.com/en/articles/17252)

This article considers the Rademacher and Walsh functions. We will explore ways to apply these functions to financial time series analysis and also consider various applications for them in trading.

![Risk-Based Trade Placement EA with On-Chart UI (Part 2): Adding Interactivity and Logic](https://c.mql5.com/2/180/20159-risk-based-trade-placement-logo.png)[Risk-Based Trade Placement EA with On-Chart UI (Part 2): Adding Interactivity and Logic](https://www.mql5.com/en/articles/20159)

Learn how to build an interactive MQL5 Expert Advisor with an on-chart control panel. Know how to compute risk-based lot sizes and place trades directly from the chart.

![Reimagining Classic Strategies (Part 18): Searching For Candlestick Patterns](https://c.mql5.com/2/180/20223-reimagining-classic-strategies-logo.png)[Reimagining Classic Strategies (Part 18): Searching For Candlestick Patterns](https://www.mql5.com/en/articles/20223)

This article helps new community members search for and discover their own candlestick patterns. Describing these patterns can be daunting, as it requires manually searching and creatively identifying improvements. Here, we introduce the engulfing candlestick pattern and show how it can be enhanced for more profitable trading applications.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/20167&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062543330136466476)

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