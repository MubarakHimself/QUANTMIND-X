---
title: From Novice to Expert: Trading the RSI with Market Structure Awareness
url: https://www.mql5.com/en/articles/20554
categories: Trading, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T17:52:46.555472
---

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/20554&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049449155966905275)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/20554#para2)
- [Implementation](https://www.mql5.com/en/articles/20554#para3)
- [Testing](https://www.mql5.com/en/articles/20554#para4)
- [Conclusion](https://www.mql5.com/en/articles/20554#para5)
- [Key Lessons](https://www.mql5.com/en/articles/20554#para6)
- [Attachments](https://www.mql5.com/en/articles/20554#para7)

### Introduction

The journey toward consistent trading performance is often hindered by a reliance on fragmented, lagging signals. Conventional education emphasizes tools like channels for breakout trades or the [RSI oscillator](https://en.wikipedia.org/wiki/Relative_strength_index "https://en.wikipedia.org/wiki/Relative_strength_index") for overbought/oversold reversals. While foundational, these methods present well-documented pitfalls: breakouts often fail, retests may never occur, and the RSI can remain in extreme territories far longer than a trader’s capital can endure. This reactive approach leaves traders vulnerable to false signals and missed opportunities, perpetually entering after the optimal momentum has begun.

This article introduces a structured methodology designed to overcome these limitations by synergizing price action awareness with early momentum confirmation. We move beyond viewing market structure—specifically, trend channels—as mere boundaries for breakouts. Instead, we treat it as the foundational context that informs the strength and validity of momentum signals.

Our focus is on a precise technique: using the RSI not as a standalone reversal oracle, but as a confirmation engine within the framework of established market structure. By algorithmically detecting price channels and seeking RSI-based early confirmation at key structural levels, we aim to identify higher-probability entries that are both earlier and safer than traditional breakout-retest models.

Furthermore, we bridge the gap between conceptual strategy and executable edge by leveraging the MQL5 programming language. The manual identification of structures and signals is not only time-consuming but also subjective. Therefore, we will detail the development of an automated system that:

1. Algorithmically identifies valid trend channels.
2. Intelligently interprets RSI behavior in the context of these channels.
3. Executes and manages trades based on predefined, rule-based logic.

We will begin with a thorough analysis of common manual channel trading and its common patterns, establishing the core price action principles. We then progress to building a robust channel-detection algorithm. The cornerstone of our discussion is the innovative integration of this structural analysis with filtered RSI dynamics to generate high-confidence signals. Finally, we encapsulate this entire logic into a prototype automated trading system in MQL5, demonstrating a practical path from market theory to algorithmic execution.

**Understanding the concept and common structures**

Bullish and bearish flags represent a consolidation of momentum within a strong trend. While traditionally viewed as simple continuation patterns, their true value lies in the specific momentum behavior exhibited within their channel boundaries. The conventional breakout-and-retest approach, though logical, systematically surrenders a substantial portion of the ensuing move.

Our methodology refines this by targeting the precise moment within the consolidation where the trend's underlying strength reasserts itself, using a powerful confluence of structure and momentum.

The Anatomy of a Reliable Flag

Bullish Flag: Forms in a strong uptrend. It consists of a sharp, near-vertical flagpole (the initial impulsive move), followed by a downward-sloping or rectangular consolidation channel. The pattern is valid only if the consolidation does not retrace beyond the start of the flagpole.

![A bullish Flag](https://c.mql5.com/2/185/Bullish_flag.png)

A Bullish flag setup

Bearish Flag: The mirror image within a downtrend. A steep flagpole downward is followed by an upward-sloping or rectangular consolidation channel.

![A bearish Flag](https://c.mql5.com/2/185/A_bearish.png)

A bearish flag setup

Beyond the Breakout: The Divergence Edge

The standard breakout entry waits for price to exit the channel and then retest its boundary. Our approach seeks a higher-probability, earlier entry by identifying loss of momentum in the counter-trend move before the breakout even occurs. This is achieved through regular RSI divergence at the pattern's critical level.

For a bullish flag:

The Structural Context: Price is making a series of lower lows (LL) as it descends within the downward-sloping channel.

The Divergence Signal (Our Entry Confluence): As price makes its 3rd or 4th lower low (LL) near the channel's lower support boundary, the RSI forms a higher low (HL). This divergence indicates that the downward selling momentum within the consolidation is exhausting precisely at the logical support area. The larger uptrend is poised to resume.

For a bearish flag:

The structural context: Price is making a series of higher highs (HH) as it ascends within the upward-sloping channel.

The Divergence Signal (Our Entry Confluence): As price makes a new high (HH) near the channel's upper resistance boundary, the RSI forms a lower high (LH). This signals that the upward buying momentum within the pullback is failing exactly at the logical resistance. The larger downtrend is ready to continue.

![Showing RSI divergence and Structural Relationship](https://c.mql5.com/2/185/A.png)

Showing divergence channel confluence on a bearish flag setup

Why This Confluence Transforms the Trade:

This divergence-based logic directly addresses the core flaw of both simple breakout rules and basic overbought/oversold RSI signals:

1. Prevents "Catching the Falling Knife": An oversold RSI at channel support is common, but not a reliable buy signal. A bullish RSI divergence at that same support, however, provides evidence that the selling pressure is deteriorating, turning a mere "bounce level" into a high-probability "reversal point."
2. Offers Superior Risk Management: An entry triggered by divergence at a channel boundary allows for a logically tight stop-loss (placed just beyond the most recent swing low/high in the divergence). The reward-to-risk ratio is significantly improved compared to a breakout entry.
3. Seeks Earlier, High-Confirmation Entries: Instead of waiting for price action to complete the breakout (and often the best part of the move), we act on the momentum warning that precedes and predicts that breakout.

In essence, we are no longer just trading a geometric pattern. We are trading the proof of momentum exhaustion within that pattern, using divergence as our definitive evidence.

This precise, rule-based concept is perfectly suited for algorithmic translation. In the next section, we will define the exact computational rules to detect these flag structures and identify the critical RSI divergence at their boundaries, forming the core logic of our MQL5 expert advisor.

Join us as we deconstruct the market's blueprint and build a systematic process to trade it with greater clarity, timing, and discipline.

### Implementation

Successfully transforming a trading concept into a robust automated system requires a methodical, modular approach. While RSI divergence detection combined with channel-based structure confirmation sounds theoretically straightforward, the practical implementation presents significant technical complexity. To manage this complexity effectively, I've architected the solution as two independent yet complementary modules:

1. RSI Divergence Indicator: A standalone technical indicator that identifies and visually marks divergence patterns on price charts.
2. Equidistant Channel Expert Advisor: An automated trading assistant that detects and draws channel structures in real-time.

This modular architecture offers several strategic advantages. Each component can be developed, tested, and optimized independently, ensuring reliability before integration. Traders can use either tool separately based on their strategy, or combine them for enhanced conviction. Ultimately, these modules can be seamlessly integrated into a unified trading system that provides structure detection, signal confirmation, and automated execution.

The implementation follows a systematic progression. First, I'll detail the step-by-step construction of the RSI Divergence indicator, explaining the logic for pivot detection, divergence validation, and visual signal presentation. Next, I'll demonstrate the Channel Placer EA's algorithm for identifying and drawing equidistant channels on price structures. When these tools operate in concert, traders receive timely divergence alerts within confirmed channel contexts, transforming subjective technical analysis into a mechanical, rules-based trading methodology backed by algorithmic precision.

These next steps will guide you through the complete implementation journey, from individual component development to integrated system deployment, providing you with a professional-grade trading toolkit built on solid algorithmic foundations.

**RSI Divergence Detector Indicator**

Section 1: Indicator Framework Setup and Configuration

The foundation of our RSI Divergence Detector begins with establishing the proper MetaTrader 5 indicator framework. We start with the mandatory indicator properties that define how the tool interacts with the trading platform. The #property directives at lines 1-22 configure essential metadata: copyright information, version tracking, and, most importantly, the visual presentation settings.

The _indicator\_separate\_window_ declaration ensures our RSI calculations appear in their own dedicated sub-window below the main price chart, preventing visual clutter. We allocate 7 data buffers for storing various calculation results while defining only 3 actual plots for display—this separation between calculation buffers and visual outputs allows for efficient memory management while maintaining clean visualization.

The color and style configurations (lines 11-17) establish an intuitive visual language: DodgerBlue for the RSI line, orange for high pivots, and DeepSkyBlue for low pivots, creating immediate visual distinction between different types of price structure points.

```
//+------------------------------------------------------------------+
//|                                        RSIDivergenceDetector.mq5 |
//|                                    Copyright 2025, MetaQuotes Ltd|
//|                                            https://www.mql5.com  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property description "RSI Divergence Detector with Main Chart Arrows"
#property description "Shows divergence arrows on main chart, stores pivot values"

#property indicator_separate_window
#property indicator_buffers 7
#property indicator_plots   3
#property indicator_color1  clrDodgerBlue      // RSI line
#property indicator_color2  clrOrange          // RSI High Pivots
#property indicator_color3  clrDeepSkyBlue     // RSI Low Pivots
#property indicator_width1  2
#property indicator_width2  3
#property indicator_width3  3
#property indicator_label1  "RSI"
#property indicator_label2  "RSI High Pivot"
#property indicator_label3  "RSI Low Pivot"
```

Section 2: User Configuration and Input Parameters

Professional trading tools must balance automation with user control, which we achieve through the comprehensive input parameter system (lines 25-42). The input keyword creates user-modifiable variables that appear in the indicator's settings dialog, allowing traders to customize the detection algorithm to their specific trading style.

We provide control over the core RSI calculation (period and price source), pivot detection sensitivity through _InpPivotStrength_, and divergence filtering parameters. The boolean flags InpShowRegular and InpShowHidden give traders the ability to toggle between different divergence types based on their trading strategy.

Particularly important is the _InpRequireRSIBreak_ parameter (line 40), which implements the confirmation logic you requested—ensuring the RSI actually breaks through the previous pivot level before signaling a divergence. This adds a layer of validation that prevents premature signals and increases reliability.

```
//--- Input parameters
input int                InpRSIPeriod     = 14;            // RSI Period
input ENUM_APPLIED_PRICE InpRSIPrice      = PRICE_CLOSE;   // RSI Applied Price
input int                InpPivotStrength = 3;             // Pivot Strength (bars on each side)
input double             InpOverbought    = 70.0;          // Overbought Level
input double             InpOversold      = 30.0;          // Oversold Level
input bool               InpShowRegular   = true;          // Show Regular Divergences
input bool               InpShowHidden    = true;          // Show Hidden Divergences
input color              InpBullishColor  = clrLimeGreen;  // Bullish divergence arrow color
input color              InpBearishColor  = clrRed;        // Bearish divergence arrow color
input int                InpArrowSize     = 3;             // Arrow size on chart
input bool               InpAlertOnDivergence = true;      // Alert on divergence
input bool               InpSendNotification = false;      // Send notification
input bool               InpRequireRSIBreak = true;        // Require RSI to break pivot line
input double             InpMinDivergenceStrength = 2.0;   // Minimum RSI divergence strength
input int                InpMaxPivotDistance = 100;        // Max bars between pivots for divergence
input double             InpArrowOffsetPct = 0.3;          // Arrow offset percentage (0.3 = 30%)
```

Section 3: Data Structure Design and Memory Management

At lines 44-67, we implement the core data architecture using a custom _RSI\_PIVOT s_ tructure. This structure represents a critical design decision: rather than relying on simple arrays that mix different types of data, we create a dedicated data type that encapsulates all relevant information about each pivot point.

Each _RSI\_PIVOT_ instance stores the bar index, RSI value, corresponding price level, timestamp, pivot type (high/low), detection strength, and confirmation status.

The constructor (lines 59-62) ensures proper initialization, preventing common programming errors with uninitialized variables. We maintain a dynamic array rsiPivots\[\] to store these structures, with pivotCount tracking the current number of valid entries. This approach provides both data integrity and efficient memory usage, as we can easily add, remove, or search through pivot data without complex index management.

```
//--- Indicator buffers
double BufferRSI[];
double BufferRSIHighPivot[];
double BufferRSILowPivot[];
double BufferRSIHigh[];
double BufferRSILow[];
double BufferPivotHigh[];
double BufferPivotLow[];

//--- Global variables
int rsiHandle;
datetime lastAlertTime;
string indicatorPrefix = "RSI_DIV_";

//--- Structures for storing pivot data
struct RSI_PIVOT
{
   int barIndex;
   double value;
   double price;
   datetime time;
   bool isHigh;
   int strength;
   bool isConfirmed;

   // Constructor to initialize values
   RSI_PIVOT() : barIndex(-1), value(0.0), price(0.0), time(0),
                 isHigh(false), strength(0), isConfirmed(false) {}
};

RSI_PIVOT rsiPivots[];
int pivotCount = 0;
```

Section 4: Initialization and Resource Setup

The _OnInit()_ function (lines 70-125) handles the critical setup phase when the indicator loads. Lines 74-80 establish the connection between our calculation buffers and the indicator's display system using _SetIndexBuffer()._

Each buffer receives a specific role: BufferRSI stores the calculated RSI values, while _BufferRSIHighPivot_ and _BufferRSILowPivot_ hold the visual markers for pivot points.

Lines 83-94 configure the plotting behavior: the main RSI line uses DRAW\_LINE, while pivot points use _DRAW\_ARROW_ with character code 159 (a square symbol). We create a handle to the built-in RSI indicator at line 103 using iRSI(), which provides efficient access to pre-calculated RSI values without reinventing the wheel.

Lines 112-115 set up visual reference lines for overbought/oversold levels, and line 121 calls _CleanChartObjects()_ to remove any residual graphical elements from previous indicator runs, ensuring a clean starting state.

```
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
{
   SetIndexBuffer(0, BufferRSI, INDICATOR_DATA);
   SetIndexBuffer(1, BufferRSIHighPivot, INDICATOR_DATA);
   SetIndexBuffer(2, BufferRSILowPivot, INDICATOR_DATA);
   SetIndexBuffer(3, BufferRSIHigh, INDICATOR_DATA);
   SetIndexBuffer(4, BufferRSILow, INDICATOR_DATA);
   SetIndexBuffer(5, BufferPivotHigh, INDICATOR_DATA);
   SetIndexBuffer(6, BufferPivotLow, INDICATOR_DATA);

   //--- Set plot properties
   PlotIndexSetInteger(0, PLOT_DRAW_TYPE, DRAW_LINE);
   PlotIndexSetInteger(1, PLOT_DRAW_TYPE, DRAW_ARROW);
   PlotIndexSetInteger(2, PLOT_DRAW_TYPE, DRAW_ARROW);

   //--- Set arrow codes for RSI pivot points
   PlotIndexSetInteger(1, PLOT_ARROW, 159);  // Square dot for high pivots
   PlotIndexSetInteger(2, PLOT_ARROW, 159);  // Square dot for low pivots

   //--- Set empty values for arrow buffers
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(2, PLOT_EMPTY_VALUE, EMPTY_VALUE);

   //--- Set indicator labels
   IndicatorSetString(INDICATOR_SHORTNAME, "RSI Divergence (" + string(InpRSIPeriod) + ")");
   IndicatorSetInteger(INDICATOR_DIGITS, 2);

   //--- Create RSI handle
   rsiHandle = iRSI(_Symbol, _Period, InpRSIPeriod, InpRSIPrice);
   if(rsiHandle == INVALID_HANDLE)
   {
      Print("Failed to create RSI handle");
      return(INIT_FAILED);
   }

   //--- Set overbought/oversold levels
   IndicatorSetDouble(INDICATOR_LEVELVALUE, 0, InpOversold);
   IndicatorSetDouble(INDICATOR_LEVELVALUE, 1, InpOverbought);
   IndicatorSetInteger(INDICATOR_LEVELCOLOR, 0, clrSilver);
   IndicatorSetInteger(INDICATOR_LEVELCOLOR, 1, clrSilver);
   IndicatorSetInteger(INDICATOR_LEVELSTYLE, 0, STYLE_DOT);
   IndicatorSetInteger(INDICATOR_LEVELSTYLE, 1, STYLE_DOT);

   lastAlertTime = 0;

   //--- Clean any existing objects from previous runs
   CleanChartObjects();

   return(INIT_SUCCEEDED);
}
```

Section 5: Core Calculation Engine and Data Processing

The _OnCalculate()_ function (lines 128-169) serves as the main processing loop, called on every tick and bar update. We begin with validation at line 130, ensuring sufficient historical data exists for meaningful calculations.

Lines 133-142 implement intelligent recalculation logic: if this is the first run _(prev\_calculated == 0)_, we initialize our pivot array and reset counters; otherwise, we continue from where we left off, optimizing performance by avoiding redundant calculations. At line 145, we retrieve RSI values using CopyBuffer() from our RSI handle—this demonstrates proper use of [MetaTrader 5's](https://download.mql5.com/cdn/web/metaquotes.ltd/mt5/mt5setup.exe?utm_source=web.installer&utm_campaign=mql5.welcome.open "https://download.mql5.com/cdn/web/metaquotes.ltd/mt5/mt5setup.exe?utm_source=web.installer&utm_campaign=mql5.welcome.open") technical indicator API for data access.

The function then orchestrates the three main processing steps: finding RSI pivots, detecting divergences, and cleaning old data. This modular approach separates concerns while maintaining efficient data flow.

```
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
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
                const int &spread[])
{
   if(rates_total < InpRSIPeriod + 10) return(0);

   int start;
   if(prev_calculated == 0)
   {
      start = InpRSIPeriod + InpPivotStrength;
      ArrayResize(rsiPivots, rates_total);
      pivotCount = 0;
      CleanChartObjects();
   }
   else
   {
      start = prev_calculated - 1;
   }

   //--- Get RSI values
   if(CopyBuffer(rsiHandle, 0, 0, rates_total, BufferRSI) <= 0)
      return(0);

   //--- Find RSI pivots
   FindRSIPivots(rates_total, start, high, low, time);

   //--- Detect divergences and draw arrows on main chart
   DetectDivergences(rates_total, start, high, low, close, time);

   //--- Clean old pivot data
   CleanOldPivots(rates_total);

   return(rates_total);
}
```

Section 6: Pivot Detection Algorithm Implementation

The _FindRSIPivots()_ function (lines 172-218) implements the core swing point identification logic. Lines 175-176 clear previous pivot markers from the display buffers, ensuring we only show current, relevant pivots.

The loop at lines 178-216 iterates through bars, applying the pivot detection criteria. The _IsHighPivot()_ and _IsLowPivot()_ helper functions (lines 221-272) contain the actual detection logic: for a bar to qualify as a high pivot, its RSI value must be greater than a specified number of bars on both sides (controlled by InpPivotStrength).

When a valid pivot is found (lines 183-215), we populate a new _RSI\_PIVOT_ structure with all relevant data: the bar index for reference, RSI value for comparison, price level for divergence analysis, timestamp for tracking, and type classification. The i _sConfirmed_flag is initially set to false, allowing our divergence detection logic to mark pivots once they've contributed to a valid signal.

```
//+------------------------------------------------------------------+
//| Find RSI pivots function                                         |
//+------------------------------------------------------------------+
void FindRSIPivots(int rates_total, int start, const double &high[], const double &low[], const datetime &time[])
{
   // Clear pivot buffers
   ArrayFill(BufferRSIHighPivot, start, rates_total - start, EMPTY_VALUE);
   ArrayFill(BufferRSILowPivot, start, rates_total - start, EMPTY_VALUE);

   for(int i = start; i < rates_total - InpPivotStrength; i++)
   {
      //--- Check for RSI high pivot
      if(IsHighPivot(i, BufferRSI, InpPivotStrength))
      {
         if(pivotCount < ArraySize(rsiPivots))
         {
            rsiPivots[pivotCount].barIndex = i;
            rsiPivots[pivotCount].value = BufferRSI[i];
            rsiPivots[pivotCount].price = high[i];
            rsiPivots[pivotCount].time = time[i];
            rsiPivots[pivotCount].isHigh = true;
            rsiPivots[pivotCount].strength = InpPivotStrength;
            rsiPivots[pivotCount].isConfirmed = false;
            pivotCount++;

            BufferRSIHighPivot[i] = BufferRSI[i];
         }
      }

      //--- Check for RSI low pivot
      if(IsLowPivot(i, BufferRSI, InpPivotStrength))
      {
         if(pivotCount < ArraySize(rsiPivots))
         {
            rsiPivots[pivotCount].barIndex = i;
            rsiPivots[pivotCount].value = BufferRSI[i];
            rsiPivots[pivotCount].price = low[i];
            rsiPivots[pivotCount].time = time[i];
            rsiPivots[pivotCount].isHigh = false;
            rsiPivots[pivotCount].strength = InpPivotStrength;
            rsiPivots[pivotCount].isConfirmed = false;
            pivotCount++;

            BufferRSILowPivot[i] = BufferRSI[i];
         }
      }
   }
}
```

Section 7: Helper Functions for Pivot Detection

The _IsHighPivot()_ and _IsLowPivot()_ functions contain the precise logic for identifying swing points in RSI values. These functions implement what traders commonly refer to as "pivot strength" detection.

For a bar to qualify as a high pivot (lines 221-246), its RSI value must be greater than a specified number of preceding bars (looking left) AND greater than a specified number of following bars (looking right).

This bilateral verification ensures we're detecting true local maxima/minima rather than temporary fluctuations. The strength parameter controls sensitivity: higher values require more confirmation bars, resulting in fewer but more reliable pivots. These functions return simple boolean results that feed into our main pivot detection loop, demonstrating clean separation of detection logic from data management.

```
//+------------------------------------------------------------------+
//| Check if bar is a high pivot                                     |
//+------------------------------------------------------------------+
bool IsHighPivot(int index, double &buffer[], int strength)
{
   if(index < strength || index >= ArraySize(buffer) - strength)
      return false;

   double pivotValue = buffer[index];

   // Check left side
   for(int i = 1; i <= strength; i++)
   {
      if(buffer[index - i] > pivotValue)
         return false;
   }

   // Check right side
   for(int i = 1; i <= strength; i++)
   {
      if(buffer[index + i] > pivotValue)
         return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Check if bar is a low pivot                                      |
//+------------------------------------------------------------------+
bool IsLowPivot(int index, double &buffer[], int strength)
{
   if(index < strength || index >= ArraySize(buffer) - strength)
      return false;

   double pivotValue = buffer[index];

   // Check left side
   for(int i = 1; i <= strength; i++)
   {
      if(buffer[index - i] < pivotValue)
         return false;
   }

   // Check right side
   for(int i = 1; i <= strength; i++)
   {
      if(buffer[index + i] < pivotValue)
         return false;
   }

   return true;
}
```

Section 8: Divergence Detection Logic and Signal Generation

The _DetectDivergences()_function (lines 275-373) represents the intelligence core of our indicator. After validating we have sufficient pivot data at line 277, we implement a nested loop structure (lines 280-372) that compares all possible pivot pairs.

Line 283 applies a practical filter: pivots too far apart (exceeding _InpMaxPivotDistance)_ are ignored, as distant correlations often represent different market phases. The logic then separates into two main branches: high pivot comparisons for bearish divergences (lines 287-322) and low pivot comparisons for bullish divergences (lines 323-372).

For each divergence type, we check both regular and hidden varieties, applying the specific Check...Divergence() functions that encapsulate the exact price/RSI relationship criteria. When a valid divergence is found, lines 294-298 (for bearish) and 330-334 (for bullish) calculate appropriate arrow placement using a percentage-based offset from the current price range, ensuring arrows are clearly visible without obscuring price action.

```
//+------------------------------------------------------------------+
//| Detect divergences between price and RSI                         |
//+------------------------------------------------------------------+
void DetectDivergences(int rates_total, int start, const double &high[], const double &low[],
                      const double &close[], const datetime &time[])
{
   if(pivotCount < 4) return;

   // Check all pivot pairs for divergence
   for(int i = pivotCount - 1; i >= 0; i--)
   {
      for(int j = i - 1; j >= 0; j--)
      {
         // Skip if pivots are too far apart
         if(rsiPivots[i].barIndex - rsiPivots[j].barIndex > InpMaxPivotDistance)
            continue;

         // Check if both are high pivots
         if(rsiPivots[i].isHigh && rsiPivots[j].isHigh)
         {
            // Check for regular bearish divergence
            if(InpShowRegular && CheckBearishDivergence(rsiPivots[i], rsiPivots[j], rates_total))
            {
               // Check RSI break confirmation if required
               if(!InpRequireRSIBreak || CheckRSIBreak(rsiPivots[j], rsiPivots[i], rates_total, true))
               {
                  double arrowPrice = high[rsiPivots[i].barIndex];
                  double range = high[rsiPivots[i].barIndex] - low[rsiPivots[i].barIndex];
                  double offset = range * InpArrowOffsetPct;

                  DrawChartArrow(rsiPivots[i].barIndex, time[rsiPivots[i].barIndex],
                                arrowPrice + offset, false, "Bearish");
                  TriggerAlert("Regular Bearish Divergence detected!", time[rsiPivots[i].barIndex]);
                  // Mark as confirmed to avoid duplicate signals
                  rsiPivots[i].isConfirmed = true;
                  rsiPivots[j].isConfirmed = true;
               }
            }
            // Check for hidden bearish divergence
            else if(InpShowHidden && CheckHiddenBearishDivergence(rsiPivots[i], rsiPivots[j]))
            {
               if(!InpRequireRSIBreak || CheckRSIBreak(rsiPivots[j], rsiPivots[i], rates_total, true))
               {
                  double arrowPrice = high[rsiPivots[i].barIndex];
                  double range = high[rsiPivots[i].barIndex] - low[rsiPivots[i].barIndex];
                  double offset = range * InpArrowOffsetPct;

                  DrawChartArrow(rsiPivots[i].barIndex, time[rsiPivots[i].barIndex],
                                arrowPrice + offset, false, "Bearish(H)");
                  TriggerAlert("Hidden Bearish Divergence detected!", time[rsiPivots[i].barIndex]);
                  rsiPivots[i].isConfirmed = true;
                  rsiPivots[j].isConfirmed = true;
               }
            }
         }
         // Check if both are low pivots
         else if(!rsiPivots[i].isHigh && !rsiPivots[j].isHigh)
         {
            // Check for regular bullish divergence
            if(InpShowRegular && CheckBullishDivergence(rsiPivots[i], rsiPivots[j], rates_total))
            {
               // Check RSI break confirmation if required
               if(!InpRequireRSIBreak || CheckRSIBreak(rsiPivots[j], rsiPivots[i], rates_total, false))
               {
                  double arrowPrice = low[rsiPivots[i].barIndex];
                  double range = high[rsiPivots[i].barIndex] - low[rsiPivots[i].barIndex];
                  double offset = range * InpArrowOffsetPct;

                  DrawChartArrow(rsiPivots[i].barIndex, time[rsiPivots[i].barIndex],
                                arrowPrice - offset, true, "Bullish");
                  TriggerAlert("Regular Bullish Divergence detected!", time[rsiPivots[i].barIndex]);
                  rsiPivots[i].isConfirmed = true;
                  rsiPivots[j].isConfirmed = true;
               }
            }
            // Check for hidden bullish divergence
            else if(InpShowHidden && CheckHiddenBullishDivergence(rsiPivots[i], rsiPivots[j]))
            {
               if(!InpRequireRSIBreak || CheckRSIBreak(rsiPivots[j], rsiPivots[i], rates_total, false))
               {
                  double arrowPrice = low[rsiPivots[i].barIndex];
                  double range = high[rsiPivots[i].barIndex] - low[rsiPivots[i].barIndex];
                  double offset = range * InpArrowOffsetPct;

                  DrawChartArrow(rsiPivots[i].barIndex, time[rsiPivots[i].barIndex],
                                arrowPrice - offset, true, "Bullish(H)");
                  TriggerAlert("Hidden Bullish Divergence detected!", time[rsiPivots[i].barIndex]);
                  rsiPivots[i].isConfirmed = true;
                  rsiPivots[j].isConfirmed = true;
               }
            }
         }
      }
   }
}
```

Section 9: Divergence Validation and Confirmation Functions

The validation functions _CheckBearishDivergence()_, _CheckBullishDivergence()_, _CheckHiddenBearishDivergence(_), and _CheckHiddenBullishDivergence()_ (lines 376-445) implement the precise mathematical definitions of each divergence type.

Each function follows a consistent pattern: parameter validation, price relationship check, RSI relationship check, strength validation, and confirmation status verification.

For example, _CheckBearishDivergence()_ at lines 376-395 requires: 1) the newer pivot has a higher price than the older pivot, 2) the newer pivot has a lower RSI value (by at least InpMinDivergenceStrength), 3) the absolute difference meets minimum strength requirements, and 4) neither pivot has been previously confirmed in another divergence. These strict criteria prevent false signals and ensure only meaningful divergences trigger alerts.

```
//+------------------------------------------------------------------+
//| Check for regular bearish divergence                             |
//+------------------------------------------------------------------+
bool CheckBearishDivergence(RSI_PIVOT &pivot1, RSI_PIVOT &pivot2, int rates_total)
{
   // pivot1 is newer, pivot2 is older
   if(pivot1.barIndex <= pivot2.barIndex) return false;
   if(pivot1.barIndex >= rates_total - 1 || pivot2.barIndex >= rates_total - 1) return false;

   // Price makes higher high
   bool priceHigher = pivot1.price > pivot2.price;

   // RSI makes lower high
   bool rsiLower = pivot1.value < pivot2.value - InpMinDivergenceStrength;

   // Ensure there's enough divergence strength
   bool enoughStrength = MathAbs(pivot2.value - pivot1.value) >= InpMinDivergenceStrength;

   // Not already confirmed
   bool notConfirmed = !pivot1.isConfirmed && !pivot2.isConfirmed;

   return priceHigher && rsiLower && enoughStrength && notConfirmed;
}

//+------------------------------------------------------------------+
//| Check for regular bullish divergence                             |
//+------------------------------------------------------------------+
bool CheckBullishDivergence(RSI_PIVOT &pivot1, RSI_PIVOT &pivot2, int rates_total)
{
   // pivot1 is newer, pivot2 is older
   if(pivot1.barIndex <= pivot2.barIndex) return false;
   if(pivot1.barIndex >= rates_total - 1 || pivot2.barIndex >= rates_total - 1) return false;

   // Price makes lower low
   bool priceLower = pivot1.price < pivot2.price;

   // RSI makes higher low
   bool rsiHigher = pivot1.value > pivot2.value + InpMinDivergenceStrength;

   // Ensure there's enough divergence strength
   bool enoughStrength = MathAbs(pivot1.value - pivot2.value) >= InpMinDivergenceStrength;

   // Not already confirmed
   bool notConfirmed = !pivot1.isConfirmed && !pivot2.isConfirmed;

   return priceLower && rsiHigher && enoughStrength && notConfirmed;
}

//+------------------------------------------------------------------+
//| Check for hidden bearish divergence                              |
//+------------------------------------------------------------------+
bool CheckHiddenBearishDivergence(RSI_PIVOT &pivot1, RSI_PIVOT &pivot2)
{
   // Price makes lower high
   bool priceLower = pivot1.price < pivot2.price;

   // RSI makes higher high
   bool rsiHigher = pivot1.value > pivot2.value + InpMinDivergenceStrength;

   // Not already confirmed
   bool notConfirmed = !pivot1.isConfirmed && !pivot2.isConfirmed;

   return priceLower && rsiHigher && notConfirmed;
}

//+------------------------------------------------------------------+
//| Check for hidden bullish divergence                              |
//+------------------------------------------------------------------+
bool CheckHiddenBullishDivergence(RSI_PIVOT &pivot1, RSI_PIVOT &pivot2)
{
   // Price makes higher low
   bool priceHigher = pivot1.price > pivot2.price;

   // RSI makes lower low
   bool rsiLower = pivot1.value < pivot2.value - InpMinDivergenceStrength;

   // Not already confirmed
   bool notConfirmed = !pivot1.isConfirmed && !pivot2.isConfirmed;

   return priceHigher && rsiLower && notConfirmed;
}
```

Section 10: RSI Break Confirmation and Signal Validation

The CheckRSIBreak() function (lines 448-478) implements the additional validation layer you specified—ensuring the RSI actually breaks through the previous pivot level before considering a divergence confirmed.

This function addresses the common problem of premature divergence signals that occur before momentum actually shifts. Lines 456-458 establish a lookback window (10 bars) after the newer pivot, during which we monitor for the break condition.

For bearish divergences (lines 464-467), we check if RSI falls below the older pivot's value; for bullish divergences (lines 469-472), we check if RSI rises above the older pivot's value. This confirmation step transforms the indicator from a simple pattern detector into a momentum-shift validator, significantly increasing signal reliability.

```
//+------------------------------------------------------------------+
//| Check if RSI has broken the previous pivot level                 |
//+------------------------------------------------------------------+
bool CheckRSIBreak(RSI_PIVOT &olderPivot, RSI_PIVOT &newerPivot, int rates_total, bool isBearish)
{
   // For bearish: Check if RSI has broken below the older pivot's value
   // For bullish: Check if RSI has broken above the older pivot's value

   if(newerPivot.barIndex >= rates_total - 1) return false;

   // Look for break in recent bars after newer pivot
   int lookbackBars = 10;
   int startBar = newerPivot.barIndex + 1;
   int endBar = MathMin(rates_total - 1, newerPivot.barIndex + lookbackBars);

   for(int i = startBar; i <= endBar; i++)
   {
      if(isBearish)
      {
         // Bearish: RSI should break below older pivot value
         if(BufferRSI[i] < olderPivot.value)
            return true;
      }
      else
      {
         // Bullish: RSI should break above older pivot value
         if(BufferRSI[i] > olderPivot.value)
            return true;
      }
   }

   return false;
}
```

Section 11: Visual Signal Presentation and Chart Management

The _DrawChartArrow()_ function (lines 481-529) handles the visual presentation of divergence signals on the main price chart. Lines 483-484 generate unique object names using the timestamp, ensuring we can manage individual arrow/label pairs independently. Lines 487-488 clean up any existing objects with the same names, preventing duplication when the indicator recalculates.

The function then branches based on signal direction: bullish signals use OBJ\_ARROW\_BUY objects (lines 491-505) while bearish signals use OBJ\_ARROW\_SELL objects (lines 507-522). Each arrow receives configurable styling (color from InpBullishColor/InpBearishColor, size from InpArrowSize) and includes a text label showing the divergence type.

The CleanChartObjects() function (lines 532-542) provides systematic cleanup of all indicator-created objects, maintaining chart cleanliness.

```
//+------------------------------------------------------------------+
//| Draw arrow on main chart                                         |
//+------------------------------------------------------------------+
void DrawChartArrow(int barIndex, datetime time, double price, bool isBullish, string labelText)
{
   string arrowName = indicatorPrefix + "Arrow_" + IntegerToString(time);
   string labelName = indicatorPrefix + "Label_" + IntegerToString(time);

   // Remove existing objects with same name
   ObjectDelete(0, arrowName);
   ObjectDelete(0, labelName);

   if(isBullish)
   {
      // Draw bullish arrow (up arrow)
      if(!ObjectCreate(0, arrowName, OBJ_ARROW_BUY, 0, time, price))
      {
         Print("Failed to create arrow object: ", GetLastError());
         return;
      }

      ObjectSetInteger(0, arrowName, OBJPROP_COLOR, InpBullishColor);
      ObjectSetInteger(0, arrowName, OBJPROP_WIDTH, InpArrowSize);
      ObjectSetInteger(0, arrowName, OBJPROP_BACK, false);

      // Draw label
      if(!ObjectCreate(0, labelName, OBJ_TEXT, 0, time, price))
         return;

      ObjectSetString(0, labelName, OBJPROP_TEXT, labelText);
      ObjectSetInteger(0, labelName, OBJPROP_COLOR, InpBullishColor);
      ObjectSetInteger(0, labelName, OBJPROP_FONTSIZE, 8);
   }
   else
   {
      // Draw bearish arrow (down arrow)
      if(!ObjectCreate(0, arrowName, OBJ_ARROW_SELL, 0, time, price))
      {
         Print("Failed to create arrow object: ", GetLastError());
         return;
      }

      ObjectSetInteger(0, arrowName, OBJPROP_COLOR, InpBearishColor);
      ObjectSetInteger(0, arrowName, OBJPROP_WIDTH, InpArrowSize);
      ObjectSetInteger(0, arrowName, OBJPROP_BACK, false);

      // Draw label
      if(!ObjectCreate(0, labelName, OBJ_TEXT, 0, time, price))
         return;

      ObjectSetString(0, labelName, OBJPROP_TEXT, labelText);
      ObjectSetInteger(0, labelName, OBJPROP_COLOR, InpBearishColor);
      ObjectSetInteger(0, labelName, OBJPROP_FONTSIZE, 8);
   }

   ObjectSetInteger(0, labelName, OBJPROP_ANCHOR, ANCHOR_CENTER);
   ObjectSetInteger(0, labelName, OBJPROP_BACK, false);
}
```

Section 12: Cleanup Functions and Alert System

The _CleanChartObjects()_ function (lines 532-542) and _OnDeinit()_ function (lines 572-582) handle critical resource management. The pivot cleanup logic maintains a rolling window of the most recent 500 pivots, discarding older ones to prevent memory bloat during extended chart analysis. When the indicator is removed, OnDeinit() releases the RSI indicator handle using _IndicatorRelease()_ and cleans up all chart objects.

The _TriggerAlert()_ function implements a sophisticated notification system that balances signal awareness with notification fatigue management. Line 554 respects the user's preference setting for alerts.

Line 557 implements time-based filtering using _lastAlertTime_, preventing repeated alerts for the same bar—a common annoyance in trading indicators. When conditions are met, line 562 triggers a standard MetaTrader 5 alert with sound, and lines 565-568 optionally send platform notifications if enabled.

```
//+------------------------------------------------------------------+
//| Clean chart objects                                              |
//+------------------------------------------------------------------+
void CleanChartObjects()
{
   int total = ObjectsTotal(0, 0, -1);
   for(int i = total - 1; i >= 0; i--)
   {
      string name = ObjectName(0, i, 0, -1);
      if(StringFind(name, indicatorPrefix, 0) != -1)
      {
         ObjectDelete(0, name);
      }
   }
}

//+------------------------------------------------------------------+
//| Clean old pivot data                                             |
//+------------------------------------------------------------------+
void CleanOldPivots(int rates_total)
{
   if(pivotCount > 500)  // Keep last 500 pivots maximum
   {
      int newCount = 250;
      for(int i = 0; i < newCount; i++)
      {
         rsiPivots[i] = rsiPivots[pivotCount - newCount + i];
      }
      pivotCount = newCount;
   }
}

//+------------------------------------------------------------------+
//| Trigger alert function                                           |
//+------------------------------------------------------------------+
void TriggerAlert(string message, datetime time)
{
   if(!InpAlertOnDivergence) return;

   // Avoid repeated alerts for same bar
   if(time <= lastAlertTime) return;

   lastAlertTime = time;

   // Play sound
   Alert(message + " at ", TimeToString(time, TIME_DATE|TIME_MINUTES));

   // Send notification if enabled
   if(InpSendNotification)
      SendNotification("RSI Divergence: " + Symbol() + " " +
                       StringSubstr(EnumToString(_Period), 7) +
                       " - " + message + " at " + TimeToString(time));
}

//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                       |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Delete RSI handle
   if(rsiHandle != INVALID_HANDLE)
      IndicatorRelease(rsiHandle);

   // Clean up chart objects
   CleanChartObjects();
}
```

Having thoroughly examined the RSI Divergence Detector's architecture and implementation, the complete source code is available attached below the article. With our technical indicator foundation established, we now progress to the second component of our trading system: developing the Equidistant Channel Auto-Placement Expert Advisor. This automated tool will complement our divergence detection by identifying structural price patterns, ultimately creating an integrated algorithmic trading solution. Following implementation, we will share comprehensive test results for both projects and establish a strategic roadmap for continued development and system integration.

**Equidistant Channel Auto-Placement Expert Advisor**

The Equidistant Channel Auto-Placement EA establishes a robust foundation through a carefully designed enumeration system and comprehensive user-configurable parameters. The _ENUM\_CHANNEL\_TYPE_ enumeration provides clear semantic distinction between rising and falling channels, replacing traditional bullish/bearish terminology with more descriptive terms that align with trading setups: rising channels (higher lows) indicate sell opportunities, while falling channels (lower highs) suggest buy opportunities. This nomenclature shift represents a critical design decision that aligns channel terminology with actual trading behavior rather than abstract technical concepts.

Section 1: Architectural Foundation and User Configuration Framework

The input parameter system implements a sophisticated configuration interface that balances automation with user control. Each parameter serves a specific purpose in the channel detection algorithm: _LookbackBars_ defines the historical data scope, _SwingStrength_ controls sensitivity of swing point detection, while _MinTouchesPerLine_ introduces the novel validation requirement of multiple price touches per channel line—a feature that significantly reduces false signals. The parameter _TouchTolerancePips_ demonstrates attention to real-world trading conditions by accounting for minor price deviations that might otherwise invalidate genuine channel structures.

```
enum ENUM_CHANNEL_TYPE
{
   CHANNEL_NONE,
   CHANNEL_RISING,   // Higher lows - Sell setups (formerly bullish)
   CHANNEL_FALLING   // Lower highs - Buy setups (formerly bearish)
};

input bool   EnableRisingChannels   = true;
input bool   EnableFallingChannels  = true;
input int    LookbackBars           = 150;
input int    SwingStrength          = 2;
input bool   ShowChannel            = true;
input color  RisingChannelColor     = clrRed;
input color  FallingChannelColor    = clrLimeGreen;
input int    ChannelWidth           = 1;
input int    MinChannelLengthBars   = 15;
input double MinChannelHeightPct    = 0.5;
input bool   AlertOnNewChannel      = true;
input int    MinTouchesPerLine      = 2;
input double TouchTolerancePips     = 5.0;
input int    MaxExtensionBars       = 50;
input bool   ExtendLeft             = true;
input bool   ExtendRight            = false;
```

Section 2: State Management and Initialization Architecture

The EA implements a sophisticated state management system through global variables that track channel persistence and alert timing, addressing critical issues from previous implementations. The _lastAlertTime_ variable introduces intelligent alert throttling, preventing the continuous alert problem identified during testing. The _channelStartBar_ and _channelEndBar_ variables create a memory system that enables the EA to distinguish between new channel formations and existing channel persistence, a crucial feature for maintaining single-channel display behavior.

The initialization function demonstrates professional resource management practices by cleaning existing chart objects during startup. This proactive approach prevents object accumulation that could clutter the chart interface. The deinitialization function follows the principle of minimal intervention by leaving drawn channels intact for user analysis, while providing commented options for automatic cleanup—a design that balances automation with user control.

```
datetime lastBarTime = 0;
string currentChannelName = "";
string channelPrefix = "SmartCh_";
bool channelFound = false;
ENUM_CHANNEL_TYPE currentChannelType = CHANNEL_NONE;
datetime lastAlertTime = 0;
int channelStartBar = -1;
int channelEndBar = -1;

int OnInit()
{
   DeleteAllChannels();
   Print("Smart Single Channel EA initialized");
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   // Optional: Delete channel on exit
   // DeleteAllChannels();
}
```

Section 3: Intelligent Event-Driven Processing

The OnTick() function implements an optimized event-driven architecture that balances responsiveness with computational efficiency. The function establishes a new-bar detection mechanism using iTime() to ensure processing occurs only when meaningful price action updates are available, preventing wasteful computation during price consolidation. The innovative throttling logic introduces controlled processing frequency by checking channels only every third bar, a strategic design decision that dramatically reduces alert frequency while maintaining timely channel detection.

This throttling approach directly addresses the continuous alerting problem identified during testing, transforming the EA from a noisy, over-reactive system into a professional trading tool that provides signals only when meaningful channel developments occur. The modular architecture that delegates core processing to FindAndDrawSingleChannel() demonstrates adherence to the Single Responsibility Principle, separating event handling from business logic implementation.

```
void OnTick()
{
   datetime currentBarTime = iTime(_Symbol, _Period, 0);
   if(currentBarTime <= lastBarTime) return;
   lastBarTime = currentBarTime;

   int barShift = iBarShift(_Symbol, _Period, currentBarTime);
   if(barShift % 3 != 0) return;

   FindAndDrawSingleChannel();
}
```

Section 4: Core Channel Detection and Validation Logic

The FindAndDrawSingleChannel() function implements the EA's sophisticated decision-making engine, coordinating multiple detection algorithms and applying advanced validation criteria. The function demonstrates a parallel processing approach that simultaneously evaluates both rising and falling channel possibilities, with conditional execution based on user preferences. This architecture enables comprehensive market analysis while maintaining computational efficiency through early termination of disabled detection paths.

The channel selection logic implements a multi-criteria decision algorithm that considers both recency and validation strength. The scoring system prioritizes channels based on three factors: proximity to current price, validation robustness, and minimum touch requirements. This multi-dimensional evaluation ensures only high-quality, well-validated channels receive display priority.

The new channel detection logic introduces sophisticated state comparison that prevents redundant alerts through four validation criteria: channel existence, type changes, structural shifts, and temporal cooldown. This comprehensive approach eliminates the continuous alerting problem while maintaining sensitivity to genuine market structure changes.

```
void FindAndDrawSingleChannel()
{
   bool risingFound = false;
   int risePoint1 = -1, risePoint2 = -1;
   double riseSlope = 0;
   int riseTouchCount = 0;

   if(EnableRisingChannels)
      risingFound = FindRisingChannel(risePoint1, risePoint2, riseSlope, riseTouchCount);

   bool fallingFound = false;
   int fallPoint1 = -1, fallPoint2 = -1;
   double fallSlope = 0;
   int fallTouchCount = 0;

   if(EnableFallingChannels)
      fallingFound = FindFallingChannel(fallPoint1, fallPoint2, fallSlope, fallTouchCount);

   bool drawChannel = false;
   int point1 = -1, point2 = -1;
   double slope = 0;
   ENUM_CHANNEL_TYPE newType = CHANNEL_NONE;
   int touchCount = 0;

   if(risingFound && fallingFound)
   {
      if(risePoint1 > fallPoint1 || riseTouchCount > fallTouchCount + 1)
      {
         drawChannel = true;
         point1 = risePoint1;
         point2 = risePoint2;
         slope = riseSlope;
         newType = CHANNEL_RISING;
         touchCount = riseTouchCount;
      }
      else
      {
         drawChannel = true;
         point1 = fallPoint1;
         point2 = fallPoint2;
         slope = fallSlope;
         newType = CHANNEL_FALLING;
         touchCount = fallTouchCount;
      }
   }
   else if(risingFound && riseTouchCount >= MinTouchesPerLine * 2)
   {
      drawChannel = true;
      point1 = risePoint1;
      point2 = risePoint2;
      slope = riseSlope;
      newType = CHANNEL_RISING;
      touchCount = riseTouchCount;
   }
   else if(fallingFound && fallTouchCount >= MinTouchesPerLine * 2)
   {
      drawChannel = true;
      point1 = fallPoint1;
      point2 = fallPoint2;
      slope = fallSlope;
      newType = CHANNEL_FALLING;
      touchCount = fallTouchCount;
   }

   if(drawChannel && touchCount >= MinTouchesPerLine * 2)
   {
      bool isNewChannel = (!channelFound) ||
                         (currentChannelType != newType) ||
                         (MathAbs(point1 - channelStartBar) > 10) ||
                         (TimeCurrent() - lastAlertTime > 3600);

      if(isNewChannel)
      {
         DeleteAllChannels();

         if(DrawChannel(point1, point2, slope, newType))
         {
            channelFound = true;
            currentChannelType = newType;
            channelStartBar = point1;
            channelEndBar = point2;

            if(AlertOnNewChannel && (TimeCurrent() - lastAlertTime > 3600))
            {
               string typeStr = (newType == CHANNEL_RISING) ? "Rising (Sell Setup)" : "Falling (Buy Setup)";
               string message = StringFormat("%s channel detected on %s %s. %d touches confirmed.",
                           typeStr, Symbol(), PeriodToString(_Period), touchCount);
               Alert(message);
               lastAlertTime = TimeCurrent();
            }
         }
      }
   }
   else
   {
      if(channelFound && !IsChannelStillValid())
      {
         DeleteAllChannels();
         channelFound = false;
         currentChannelType = CHANNEL_NONE;
      }
   }
}
```

Section 5: Advanced Channel Detection with Touch Validation

The FindRisingChannel() and FindFallingChannel() functions implement sophisticated pattern recognition algorithms that go beyond simple swing point detection. These functions employ a dual-loop architecture that evaluates all possible swing point combinations within the lookback period, applying multiple validation filters to identify genuine channel structures.

The validation criteria demonstrate professional attention to market microstructure by ensuring proper channel slope direction and enforcing minimum structural integrity through channel length requirements. The slope validation prevents detection of near-horizontal structures that lack meaningful trend information.

The innovative touch validation system implemented through CountChannelTouches() represents the EA's most significant advancement over traditional channel detectors. By requiring multiple price interactions with both channel boundaries, this system ensures detected channels have been tested and validated by market action rather than being mathematical artifacts. The scoring algorithm combines recency, touch frequency, and channel height into a composite score that objectively identifies the most significant channel structure.

```
bool FindRisingChannel(int &point1, int &point2, double &slope, int &touchCount)
{
   int swingLows[];
   FindSwingLows(swingLows, SwingStrength, LookbackBars);

   if(ArraySize(swingLows) < 2) return false;

   int bestPoint1 = -1, bestPoint2 = -1;
   double bestScore = -1;
   int bestTouches = 0;

   for(int i = 0; i < ArraySize(swingLows) - 1; i++)
   {
      for(int j = i + 1; j < ArraySize(swingLows); j++)
      {
         int low1 = swingLows[i];
         int low2 = swingLows[j];

         if(iLow(NULL, 0, low1) <= iLow(NULL, 0, low2)) continue;
         if(MathAbs(low1 - low2) < MinChannelLengthBars) continue;

         double low1Price = iLow(NULL, 0, low1);
         double low2Price = iLow(NULL, 0, low2);

         double barDiff = MathAbs(low1 - low2);
         slope = (low1Price - low2Price) / barDiff;

         if(slope < 0.00005) continue;

         double channelHeight = CalculateChannelHeight(low1, low2, true);
         int touches = CountChannelTouches(low1, low2, slope, low2Price, channelHeight, true);

         double recencyScore = 100.0 - (low1 * 100.0 / LookbackBars);
         double touchScore = touches * 25.0;
         double heightScore = (channelHeight / SymbolInfoDouble(_Symbol, SYMBOL_POINT)) / 100.0;

         double totalScore = recencyScore + touchScore + heightScore;

         if(totalScore > bestScore && touches >= MinTouchesPerLine * 2)
         {
            bestScore = totalScore;
            bestPoint1 = low1;
            bestPoint2 = low2;
            bestTouches = touches;
         }
      }
   }

   if(bestScore > 0)
   {
      point1 = bestPoint1;
      point2 = bestPoint2;
      slope = (iLow(NULL, 0, bestPoint1) - iLow(NULL, 0, bestPoint2)) / MathAbs(bestPoint1 - bestPoint2);
      touchCount = bestTouches;
      return true;
   }

   return false;
}
```

Section 6: Mathematical Foundation for Channel Geometry

The _CalculateChannelHeight()_ and _CountChannelTouches()_ functions implement the mathematical core that transforms price data into validated channel structures. These functions demonstrate sophisticated algorithmic thinking by addressing the geometric challenges of channel detection in financial time series.

_CalculateChannelHeight()_ implements a dual-purpose algorithm that performs empirical height calculation by measuring maximum price deviation from the calculated baseline, while enforcing minimum structural requirements through percentage-based thresholds. This approach balances empirical observation with theoretical requirements, ensuring channels have meaningful trading dimensions.

_CountChannelTouches()_ introduces tolerance-based validation that accounts for real-world price behavior where exact line touches are rare. The tolerance calculation demonstrates professional attention to instrument-specific scaling, ensuring consistent behavior across different symbols and pip values. The dual-loop structure separately validates touches on both channel boundaries, providing detailed diagnostic information about channel integrity.

```
double CalculateChannelHeight(int point1, int point2, bool isRising)
{
   double maxHeight = 0;
   int startBar = MathMin(point1, point2);
   int endBar = MathMax(point1, point2);

   double price1 = (isRising) ? iLow(NULL, 0, point1) : iHigh(NULL, 0, point1);
   double price2 = (isRising) ? iLow(NULL, 0, point2) : iHigh(NULL, 0, point2);

   double slope = (price1 - price2) / (point1 - point2);
   double intercept = price1 - slope * point1;

   for(int bar = startBar; bar <= endBar; bar++)
   {
      double currentPrice = (isRising) ? iHigh(NULL, 0, bar) : iLow(NULL, 0, bar);
      double baseLinePrice = slope * bar + intercept;
      double deviation = MathAbs(currentPrice - baseLinePrice);

      if(deviation > maxHeight) maxHeight = deviation;
   }

   double minHeight = (isRising) ? iLow(NULL, 0, startBar) * MinChannelHeightPct / 100.0 :
                                   iHigh(NULL, 0, startBar) * MinChannelHeightPct / 100.0;

   return MathMax(maxHeight, minHeight);
}

int CountChannelTouches(int point1, int point2, double slope, double basePrice, double height, bool isRising)
{
   int touches = 0;
   int startBar = MathMin(point1, point2);
   int endBar = MathMax(point1, point2);

   double tolerance = TouchTolerancePips * SymbolInfoDouble(_Symbol, SYMBOL_POINT) * 10;

   for(int bar = startBar; bar <= endBar; bar++)
   {
      double currentPrice = (isRising) ? iLow(NULL, 0, bar) : iHigh(NULL, 0, bar);
      double baseLinePrice = slope * (bar - point2) + basePrice;

      if(MathAbs(currentPrice - baseLinePrice) <= tolerance) touches++;
   }

   for(int bar = startBar; bar <= endBar; bar++)
   {
      double currentPrice = (isRising) ? iHigh(NULL, 0, bar) : iLow(NULL, 0, bar);
      double parallelLinePrice = slope * (bar - point2) + basePrice + (isRising ? height : -height);

      if(MathAbs(currentPrice - parallelLinePrice) <= tolerance) touches++;
   }

   return touches;
}
```

Section 7: Swing Point Detection Engine

The _FindSwingLows()_ and _FindSwingHighs()_ functions implement robust swing point detection using a symmetrical validation approach that examines both left and right price action. These functions form the foundational layer upon which channel detection operates, requiring careful implementation to ensure reliable structural analysis.

Both functions employ identical algorithmic patterns with reversed comparison operators, demonstrating code reuse while maintaining logical clarity. The validation loops implement a rigorous "peak and trough" detection algorithm that requires a swing point to be higher (or lower) than a specified number of bars on both sides, controlled by the SwingStrength parameter. This bilateral verification ensures detected swing points represent genuine local extrema rather than temporary fluctuations.

The array management pattern demonstrates professional memory handling in MQL5, dynamically resizing arrays as valid swing points are identified. The final _ArraySort()_ operations ensure chronological ordering from most recent to oldest, facilitating efficient subsequent processing in channel detection algorithms.

```
void FindSwingLows(int &swingPoints[], int strength, int lookback)
{
   ArrayResize(swingPoints, 0);

   for(int i = strength; i < MathMin(lookback, Bars(NULL, 0) - strength); i++)
   {
      bool isSwingLow = true;
      double currentLow = iLow(NULL, 0, i);

      for(int left = 1; left <= strength && isSwingLow; left++)
         if(iLow(NULL, 0, i - left) < currentLow) isSwingLow = false;

      if(isSwingLow)
         for(int right = 1; right <= strength && isSwingLow; right++)
            if(iLow(NULL, 0, i + right) < currentLow) isSwingLow = false;

      if(isSwingLow)
      {
         int size = ArraySize(swingPoints);
         ArrayResize(swingPoints, size + 1);
         swingPoints[size] = i;
      }
   }

   ArraySort(swingPoints);
}
```

Section 8: Professional Channel Visualization with Controlled Extension

The _DrawChannel()_function implements sophisticated graphical representation with intelligent extension management, directly addressing the excessive channel drawing issues identified in testing. The function employs a structured approach to channel rendering that balances visual clarity with chart space management.

The function demonstrates professional coordinate calculation that translates mathematical channel parameters into visual elements. The extension logic implements controlled boundary management: ExtendLeft provides historical context while ExtendRight is deliberately disabled to prevent channels from projecting excessively into empty chart space—a direct response to the testing feedback about channels being drawn "too far from current price."

The channel drawing implementation employs separate trend line objects for base and parallel lines rather than using MT5's built-in channel object. This architectural decision provides finer control over visual properties and extension behavior. The label placement logic demonstrates attention to visual hierarchy by positioning descriptive text at appropriate price levels relative to channel boundaries.

```
bool DrawChannel(int point1, int point2, double slope, ENUM_CHANNEL_TYPE type)
{
   if(!ShowChannel) return false;

   DeleteAllChannels();

   datetime time1 = iTime(NULL, 0, point1);
   datetime time2 = iTime(NULL, 0, point2);

   double price1, price2;
   color channelColor;
   string channelLabel;
   bool isRising = (type == CHANNEL_RISING);

   if(isRising)
   {
      price1 = iLow(NULL, 0, point1);
      price2 = iLow(NULL, 0, point2);
      channelColor = RisingChannelColor;
      channelLabel = "Rising Channel (Sell)";
   }
   else
   {
      price1 = iHigh(NULL, 0, point1);
      price2 = iHigh(NULL, 0, point2);
      channelColor = FallingChannelColor;
      channelLabel = "Falling Channel (Buy)";
   }

   double channelHeight = CalculateChannelHeight(point1, point2, isRising);

   int extensionBars = MathMin(MaxExtensionBars, MathAbs(point1 - point2) / 2);
   datetime extendedTime1 = time1;
   datetime extendedTime2 = time2;

   if(ExtendLeft)
   {
      int extendBack = MathMin(extensionBars, point2);
      extendedTime2 = iTime(NULL, 0, point2 - extendBack);
   }

   if(ExtendRight)
   {
      int extendForward = MathMin(extensionBars, Bars(NULL, 0) - point1 - 1);
      extendedTime1 = iTime(NULL, 0, point1 + extendForward);
   }

   currentChannelName = channelPrefix + "Base";
   ObjectCreate(0, currentChannelName, OBJ_TREND, 0, extendedTime2, price2, time1, price1);
   ObjectSetInteger(0, currentChannelName, OBJPROP_COLOR, channelColor);
   ObjectSetInteger(0, currentChannelName, OBJPROP_WIDTH, ChannelWidth);
   ObjectSetInteger(0, currentChannelName, OBJPROP_RAY_RIGHT, false);
   ObjectSetInteger(0, currentChannelName, OBJPROP_RAY_LEFT, ExtendLeft);
   ObjectSetInteger(0, currentChannelName, OBJPROP_BACK, true);

   string parallelName = channelPrefix + "Parallel";
   double parallelPrice1 = price1 + (isRising ? channelHeight : -channelHeight);
   double parallelPrice2 = price2 + (isRising ? channelHeight : -channelHeight);

   ObjectCreate(0, parallelName, OBJ_TREND, 0, extendedTime2, parallelPrice2, time1, parallelPrice1);
   ObjectSetInteger(0, parallelName, OBJPROP_COLOR, channelColor);
   ObjectSetInteger(0, parallelName, OBJPROP_WIDTH, ChannelWidth);
   ObjectSetInteger(0, parallelName, OBJPROP_STYLE, STYLE_DASH);
   ObjectSetInteger(0, parallelName, OBJPROP_RAY_RIGHT, false);
   ObjectSetInteger(0, parallelName, OBJPROP_RAY_LEFT, ExtendLeft);
   ObjectSetInteger(0, parallelName, OBJPROP_BACK, true);

   string labelName = channelPrefix + "Label";
   ObjectCreate(0, labelName, OBJ_TEXT, 0, time1, isRising ? price1 + channelHeight * 1.1 : price1 - channelHeight * 1.1);
   ObjectSetString(0, labelName, OBJPROP_TEXT, channelLabel);
   ObjectSetInteger(0, labelName, OBJPROP_COLOR, channelColor);
   ObjectSetInteger(0, labelName, OBJPROP_FONTSIZE, 8);
   ObjectSetInteger(0, labelName, OBJPROP_BACK, true);

   return true;
}
```

Section 9: Channel Persistence Validation and Resource Management

The _IsChannelStillValid()_and _DeleteAllChannels()_ functions implement critical system maintenance and validation logic that ensures the EA operates reliably during extended market sessions. These functions address two key operational concerns: channel relevance over time and resource management.

IsChannelStillValid() implements a simplified but effective channel break detection algorithm that monitors recent price action for significant deviations from established channel boundaries. The function uses percentage-based thresholds to identify potential channel breaks, providing a pragmatic balance between sensitivity and robustness. The function's conservative approach reduces false invalidation during normal price fluctuation while maintaining responsiveness to genuine structural breaks.

DeleteAllChannels() demonstrates professional object management through systematic traversal and deletion of chart objects with the EA's naming prefix. The reverse iteration pattern ensures safe object deletion during iteration, a critical detail when modifying collections during traversal. The prefix-based filtering prevents interference with other chart objects, demonstrating consideration for multi-tool trading environments.

```
bool IsChannelStillValid()
{
   if(!channelFound) return false;

   int recentBars = 10;
   bool isRising = (currentChannelType == CHANNEL_RISING);

   for(int i = 0; i < recentBars; i++)
   {
      double high = iHigh(NULL, 0, i);
      double low = iLow(NULL, 0, i);

      if(isRising)
      {
         if(low > iLow(NULL, 0, channelStartBar) * 1.01) return false;
      }
      else
      {
         if(high < iHigh(NULL, 0, channelStartBar) * 0.99) return false;
      }
   }

   return true;
}

void DeleteAllChannels()
{
   int total = ObjectsTotal(0);
   for(int i = total - 1; i >= 0; i--)
   {
      string name = ObjectName(0, i);
      if(StringFind(name, channelPrefix) == 0)
         ObjectDelete(0, name);
   }
   currentChannelName = "";
}
```

### Testing

Our testing methodology involved direct deployment and observation of the indicator's performance on live charts. The system architecture successfully separates the RSI oscillator into its own dedicated window while maintaining clear signal visualization across both the main price chart and indicator interface.

Upon receiving divergence alerts, traders can perform immediate structural analysis to identify supporting channel formations, creating a multi-factor decision framework for manual trade execution. This approach combines automated pattern detection with discretionary structural validation.

The comprehensive input parameter system allows for extensive customization, enabling users to optimize detection algorithms, adjust alert thresholds, and modify visual displays to align with their individual trading methodologies and market conditions.

![](https://c.mql5.com/2/185/terminal64_JrkRIPl36c.gif)

Deploying the RSIDivergenceDetector

The screen capture below illustrates the Equidistant Channel Auto-Placement EA being tested in the Strategy Tester, demonstrating its effectiveness in accurately identifying and drawing channel structures. When used with the RSI Divergence Detector, trading signals can be validated against these structural formations, providing a multi-confirmation framework for higher-confidence execution decisions.

![Testing the ECAP](https://c.mql5.com/2/185/metatester64_8YRVFAiyOR.gif)

Testing the EquidistantChannel Auto-Placement EA in Strategy Tester

### Conclusion

The confluence of RSI divergence signals with established market structure, particularly channel boundaries, creates a powerful framework for identifying higher-probability trading setups. In this comprehensive discussion, we have successfully automated two complementary technical analysis components: RSI divergence pattern detection and intelligent equidistant channel placement.

These tools operate synergistically on the same chart without conflict—the RSI Divergence Detector functions as a custom indicator, while the Equidistant Channel Auto-Placement operates as an Expert Advisor. Our modular development approach enabled focused implementation and testing of each component while maintaining clear separation of concerns.

While the current independent modules provide immediate utility, the natural progression involves developing a unified trading system that integrates these capabilities with automated execution logic—an exciting prospect for future development.

The detailed code explanations and implementation walkthroughs have provided practical insights into professional MQL5 programming, trading system architecture, and algorithmic validation techniques. Complete source files for both projects are available in the attachments for further study and customization.

We welcome continued discussion, questions, and constructive feedback in the comments section below.

### Key Lessons

| Key Lessons | Description |
| --- | --- |
| 1\. Modular System Architecture | Separate complex systems into independent, testable modules (indicator + EA) that can be integrated later. |
| 2\. State Management for Alert Control | Implement time-based alert throttling (lastAlertTime) to prevent continuous alerts and notification fatigue. |
| 3\. Validation-First Design Pattern | Require multiple price touches and confirmation breaks before signaling patterns, prioritizing reliability. |
| 4\. Smart Object Lifecycle Management | Use unique naming prefixes and systematic cleanup (OnDeinit) to prevent chart object accumulation. |
| 5\. Performance-Optimized Event Handling | Implement new-bar detection and processing throttling to balance responsiveness with computational efficiency. |

### Attachments

| Source Filename | Version | Description |
| --- | --- | --- |
| [RSIDivergenceDetector.mq5](https://www.mql5.com/en/articles/download/20554/RSIDivergenceDetector.mq5) | 1.00 | Custom RSI divergence indicator that detects regular/hidden divergences, stores pivot values, and displays clear buy/sell arrows on the main chart with configurable alerts and RSI break confirmation. |
| [EquidistantChannelAuto-Placement.mq5](https://www.mql5.com/en/articles/download/20554/EquidistantChannelAuto-Placement.mq5) | 1.00 | Intelligent Expert Advisor that automatically detects and draws a single valid equidistant channel with touch validation, controlled extensions, and smart alert throttling for rising (sell) and falling (buy) setups. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20554.zip "Download all attachments in the single ZIP archive")

[RSIDivergenceDetector.mq5](https://www.mql5.com/en/articles/download/20554/RSIDivergenceDetector.mq5 "Download RSIDivergenceDetector.mq5")(23.56 KB)

[EquidistantChannelAuto-Placement.mq5](https://www.mql5.com/en/articles/download/20554/EquidistantChannelAuto-Placement.mq5 "Download EquidistantChannelAuto-Placement.mq5")(23.88 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [From Novice to Expert: Higher Probability Signals](https://www.mql5.com/en/articles/20658)
- [From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)
- [From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)
- [From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)
- [The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)
- [The MQL5 Standard Library Explorer (Part 4): Custom Signal Library](https://www.mql5.com/en/articles/20266)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/501804)**
(1)


![Sasha Voitko](https://c.mql5.com/avatar/avatar_na2.png)

**[Sasha Voitko](https://www.mql5.com/en/users/sasha180986-gmail)**
\|
21 Jan 2026 at 06:23

It's a great idea. Real time has not been able to utilise the full potential of the developments. RSI [signals appear](https://www.mql5.com/en/articles/2329 "Article: Signal Calculator ") only when pressing the "refresh" button or switching from one timeframe to another. The channel is drawn and after some time disappears, or even does not appear at all. I want to use it, but I can't do it yet.


![The View and Controller components for tables in the MQL5 MVC paradigm: Resizable elements](https://c.mql5.com/2/160/18941-komponenti-view-i-controller-logo__2.png)[The View and Controller components for tables in the MQL5 MVC paradigm: Resizable elements](https://www.mql5.com/en/articles/18941)

In the article, we will add the functionality of resizing controls by dragging edges and corners of the element with the mouse.

![Introduction to MQL5 (Part 31): Mastering API and WebRequest Function in MQL5 (V)](https://c.mql5.com/2/185/20546-introduction-to-mql5-part-31-logo__1.png)[Introduction to MQL5 (Part 31): Mastering API and WebRequest Function in MQL5 (V)](https://www.mql5.com/en/articles/20546)

Learn how to use WebRequest and external API calls to retrieve recent candle data, convert each value into a usable type, and save the information neatly in a table format. This step lays the groundwork for building an indicator that visualizes the data in candle format.

![Codex Pipelines, from Python to MQL5, for Indicator Selection: A Multi-Quarter Analysis of the XLF ETF with Machine Learning](https://c.mql5.com/2/186/20595-codex-pipelines-from-python-logo.png)[Codex Pipelines, from Python to MQL5, for Indicator Selection: A Multi-Quarter Analysis of the XLF ETF with Machine Learning](https://www.mql5.com/en/articles/20595)

We continue our look at how the selection of indicators can be pipelined when facing a ‘none-typical’ MetaTrader asset. MetaTrader 5 is primarily used to trade forex, and that is good given the liquidity on offer, however the case for trading outside of this ‘comfort-zone’, is growing bolder with not just the overnight rise of platforms like Robinhood, but also the relentless pursuit of an edge for most traders. We consider the XLF ETF for this article and also cap our revamped pipeline with a simple MLP.

![Automating Trading Strategies in MQL5 (Part 46): Liquidity Sweep on Break of Structure (BoS)](https://c.mql5.com/2/185/20569-automating-trading-strategies-logo__1.png)[Automating Trading Strategies in MQL5 (Part 46): Liquidity Sweep on Break of Structure (BoS)](https://www.mql5.com/en/articles/20569)

In this article, we build a Liquidity Sweep on Break of Structure (BoS) system in MQL5 that detects swing highs/lows over a user-defined length, labels them as HH/HL/LH/LL to identify BOS (HH in uptrend or LL in downtrend), and spots liquidity sweeps when price wicks beyond the swing but closes back inside on a bullish/bearish candle.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/20554&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049449155966905275)

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