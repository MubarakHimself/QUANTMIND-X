---
title: Price Action Analysis Toolkit Development (Part 13): RSI Sentinel Tool
url: https://www.mql5.com/en/articles/17198
categories: Trading Systems, Indicators
relevance_score: 6
scraped_at: 2026-01-23T11:36:50.021557
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/17198&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062591227611751711)

MetaTrader 5 / Trading systems


### Contents

- [Introduction](https://www.mql5.com/en/articles/17198#para1)
- [Strategy Overview](https://www.mql5.com/en/articles/17198#para2)
- [MQL5 Code](https://www.mql5.com/en/articles/17198#para3)
- [Code Breakdown](https://www.mql5.com/en/articles/17198#para4)
- [Testing and Results](https://www.mql5.com/en/articles/17198#para5)
- [Conclusion](https://www.mql5.com/en/articles/17198#para6)

### Introduction

Divergence is a concept in technical analysis where the movement of an indicator, such as momentum or oscillators, deviates from the movement of price. Essentially, when the price forms new highs or lows that aren’t reflected by the indicator, it can signal a weakening trend and potentially foreshadow a reversal or a change in momentum. RSI divergence is a simple way to spot potential market reversals. When price moves in one direction while the RSI goes in another, it could signal a change in trend. However, scanning your charts by hand for these signals can be slow and prone to errors. That's where automation comes in.

In this article, we'll build an MQL5 Expert Advisor that automatically detects RSI divergence signals. The EA will mark these signals with clear arrows on your chart and provide a brief summary so you can quickly see what’s happening. Whether you're a beginner or a seasoned trader, this tool helps you spot trading opportunities that you can validate before executing trades, all without spending hours on manual analysis. Let’s dive in and see how this RSI Divergence EA can simplify your trading process.

### Strategy Overview

Understanding RSI Divergence

RSI divergence occurs when the Relative Strength Index (RSI) moves in a different direction than the asset’s price, signaling a potential shift in price momentum. This contrast between RSI and price action is a key indicator that traders use to anticipate market reversals or trend continuations. Typically, the RSI follows the price’s momentum, thereby confirming prevailing trends. However, when divergence appears, it reveals a discrepancy that often precedes a significant price movement. Recognizing these signals early can be crucial for timing market entries and exits.

In the context of RSI divergence, there are two main types

**1\.** Regular RSI Divergence

Regular RSI divergence is generally seen as a reversal signal. It indicates that the current trend is losing strength and may be about to reverse.

- Regular Bullish RSI Divergence

Occurs when the price forms a lower low while the RSI forms a higher low. This suggests that, although the price is declining, the momentum is beginning to shift upward, hinting at a potential reversal to an uptrend.

![Bullish Divergence](https://c.mql5.com/2/119/image1.png)

Fig 1. Bullish Divergence

- Regular Bearish RSI Divergence

Happens when the price forms a higher high while the RSI forms a lower high. Despite the rising price, the weakening momentum (as shown by the RSI) signals that a downturn could be on the horizon.

![Sell Signal](https://c.mql5.com/2/119/DIVERGENCE__1.png)

Fig 2. Bearish Divergence

**2.** Hidden RSI Divergence

Hidden RSI divergence is interpreted as a signal for trend continuation, rather than an impending reversal. It confirms that the current trend still has strength, even when the RSI and price temporarily diverge.

- Hidden Bullish RSI Divergence: In an uptrend, if the price forms a higher low while the RSI forms a lower low, it indicates that the correction is only temporary and that the uptrend is likely to continue.

![hidden bullish](https://c.mql5.com/2/119/HIDDEN_BULLISH_DIVERGENCE__1.png)

Fig 3. Hidden bullish divergence

- Hidden Bearish RSI Divergence: In a downtrend, when the price forms a lower high while the RSI forms a higher high, it confirms the strength of the downtrend and suggests that the downward movement is likely to persist.

![bearish divergence](https://c.mql5.com/2/119/Bearish_Divergence.png)

Fig 4. Hidden bearish divergence

Below is a summary table that encapsulates the key differences between the types of RSI divergence:

| RSI Divergence Type | Price Action | RSI Action | Signal Type | Expectation |
| --- | --- | --- | --- | --- |
| Regular Bullish | Low Low(LL) | Higher Low(HL) | Reversal Up | Downtrend to Uptrend |
| Regular Bearish | Higher High(HH) | Lower Higher(LH) | Reversal Down | Uptrend to Downtrend |
| Hidden Bullish | High Low(HL) | Low Low(LL) | Continuation Up | Uptrend Continues |
| Hidden Bearish | Lower High (LH) | High High(HH) | Continuation Down | Downtrend Continues |

In summary, this EA continuously scans both price and RSI data over a defined lookback period to detect discrepancies between their movements, what we call RSI divergence.

Below is what it does:

1\. Data Collection and Preparation

The EA gathers RSI values along with corresponding price data (lows, highs, closes, and time) from recent bars. This ensures that the analysis is always based on the latest, complete information.

2\. Identifying Swing Points

It then determines local swing highs and lows in both the price and RSI data. These swing points serve as the reference markers for our divergence analysis.

3\. Detecting Regular Divergence

- Regular Bullish Divergence: The EA looks for instances where the price makes a lower low while the RSI forms a higher low, signaling that a downtrend may be losing momentum and could reverse upward.
- Regular Bearish Divergence: It also checks for situations where the price makes a higher high while the RSI forms a lower high, indicating that an uptrend might be nearing its end as momentum wanes.

4\. Detecting Hidden Divergence

- Hidden Bullish Divergence: In an uptrend, if the price forms a higher low but the RSI records a lower low, the EA identifies this as a sign that the overall upward trend is still strong despite a temporary pullback.
- Hidden Bearish Divergence: Conversely, during a downtrend, if the price makes a lower high while the RSI shows a higher high, it confirms that the downtrend is likely to continue.

5\. Visual and Log Signal Generation

Once a divergence is detected, be it regular or hidden, the EA visually marks the event on the chart (using arrows and labels) and logs the signal's details for further analysis or backtesting. Check out for more about how it does the processes above in the [Code Breakdown](https://www.mql5.com/en/articles/17198#para4) section below.

### MQL5 Code

```
//+--------------------------------------------------------------------+
//|                                                RSI Divergence.mql5 |
//|                                 Copyright 2025, Christian Benjamin |
//|                                               https://www.mql5.com |
//+--------------------------------------------------------------------+
#property copyright "2025, MetaQuotes Software Corp."
#property link      "https://www.mql5.com/en/users/lynnchris"
#property version   "1.0"
#property strict

//---- Input parameters
input int    InpRSIPeriod             = 14;    // RSI period
input int    InpSwingLeft             = 1;     // Bars to the left for swing detection (relaxed)
input int    InpSwingRight            = 1;     // Bars to the right for swing detection (relaxed)
input int    InpLookback              = 100;   // Number of bars to scan for divergence
input int    InpEvalBars              = 5;     // Bars after which to evaluate a signal
input int    InpMinBarsBetweenSignals = 1;     // Minimum bars between same-type signals (allows frequent re-entry)
input double InpArrowOffset           = 3.0;   // Arrow offset (in points) for display
input double InpMinSwingDiffPct       = 0.05;  // Lower minimum % difference to qualify as a swing
input double InpMinRSIDiff            = 1.0;   // Lower minimum difference in RSI between swing points

// Optional RSI threshold filter for bullish divergence (disabled by default)
input bool   InpUseRSIThreshold       = false; // If true, require earlier RSI swing to be oversold for bullish divergence
input double InpRSIOversold           = 30;    // RSI oversold level
input double InpRSIOverbought         = 70;    // RSI overbought level (if needed for bearish)

//---- Global variables
int      rsiHandle;         // Handle for the RSI indicator
double   rsiBuffer[];       // Buffer for RSI values
double   lowBuffer[];       // Buffer for low prices
double   highBuffer[];      // Buffer for high prices
double   closeBuffer[];     // Buffer for close prices
datetime timeBuffer[];       // Buffer for bar times
int      g_totalBars = 0;   // Number of bars in our copied arrays
datetime lastBarTime = 0;   // Time of last closed bar

//---- Structure to hold signal information
struct SignalInfo
  {
   string            type;         // e.g. "RegBearish Divergence", "HiddenBullish Divergence"
   int               barIndex;     // Bar index where the signal was generated
   datetime          signalTime;   // Time of the signal bar
   double            signalPrice;  // Price used for the signal (swing high for bearish, swing low for bullish)
  };

SignalInfo signals[];  // Global array to store signals

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   rsiHandle = iRSI(_Symbol, _Period, InpRSIPeriod, PRICE_CLOSE);
   if(rsiHandle == INVALID_HANDLE)
     {
      Print("Error creating RSI handle");
      return(INIT_FAILED);
     }
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(rsiHandle != INVALID_HANDLE)
      IndicatorRelease(rsiHandle);
   EvaluateSignalsAndPrint();
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
// Process once per new closed candle (using bar1's time)
   datetime currentBarTime = iTime(_Symbol, _Period, 1);
   if(currentBarTime == lastBarTime)
      return;
   lastBarTime = currentBarTime;

//--- Copy RSI data
   ArrayResize(rsiBuffer, InpLookback);
   ArraySetAsSeries(rsiBuffer, true);
   if(CopyBuffer(rsiHandle, 0, 0, InpLookback, rsiBuffer) <= 0)
     {
      Print("Error copying RSI data");
      return;
     }

//--- Copy price and time data
   ArrayResize(lowBuffer, InpLookback);
   ArrayResize(highBuffer, InpLookback);
   ArrayResize(closeBuffer, InpLookback);
   ArraySetAsSeries(lowBuffer, true);
   ArraySetAsSeries(highBuffer, true);
   ArraySetAsSeries(closeBuffer, true);
   if(CopyLow(_Symbol, _Period, 0, InpLookback, lowBuffer) <= 0 ||
      CopyHigh(_Symbol, _Period, 0, InpLookback, highBuffer) <= 0 ||
      CopyClose(_Symbol, _Period, 0, InpLookback, closeBuffer) <= 0)
     {
      Print("Error copying price data");
      return;
     }

   ArrayResize(timeBuffer, InpLookback);
   ArraySetAsSeries(timeBuffer, true);
   if(CopyTime(_Symbol, _Period, 0, InpLookback, timeBuffer) <= 0)
     {
      Print("Error copying time data");
      return;
     }

   g_totalBars = InpLookback;

//--- Identify swing lows and swing highs
   int swingLows[];
   int swingHighs[];
   int startIndex = InpSwingLeft;
   int endIndex   = g_totalBars - InpSwingRight;
   for(int i = startIndex; i < endIndex; i++)
     {
      if(IsSignificantSwingLow(i, InpSwingLeft, InpSwingRight))
        {
         ArrayResize(swingLows, ArraySize(swingLows) + 1);
         swingLows[ArraySize(swingLows) - 1] = i;
        }
      if(IsSignificantSwingHigh(i, InpSwingLeft, InpSwingRight))
        {
         ArrayResize(swingHighs, ArraySize(swingHighs) + 1);
         swingHighs[ArraySize(swingHighs) - 1] = i;
        }
     }

//--- Bearish Divergence (using swing highs)
   if(ArraySize(swingHighs) >= 2)
     {
      ArraySort(swingHighs); // ascending order: index 0 is most recent
      int recent   = swingHighs[0];
      int previous = swingHighs[1];

      // Regular Bearish Divergence: Price makes a higher high while RSI makes a lower high
      if(highBuffer[recent] > highBuffer[previous] &&
         rsiBuffer[recent] < rsiBuffer[previous] &&
         (rsiBuffer[previous] - rsiBuffer[recent]) >= InpMinRSIDiff)
        {
         Print("Regular Bearish Divergence detected at bar ", recent);
         DisplaySignal("RegBearish Divergence", recent);
        }
      // Hidden Bearish Divergence: Price makes a lower high while RSI makes a higher high
      else
         if(highBuffer[recent] < highBuffer[previous] &&
            rsiBuffer[recent] > rsiBuffer[previous] &&
            (rsiBuffer[recent] - rsiBuffer[previous]) >= InpMinRSIDiff)
           {
            Print("Hidden Bearish Divergence detected at bar ", recent);
            DisplaySignal("HiddenBearish Divergence", recent);
           }
     }

//--- Bullish Divergence (using swing lows)
   if(ArraySize(swingLows) >= 2)
     {
      ArraySort(swingLows); // ascending order: index 0 is most recent
      int recent   = swingLows[0];
      int previous = swingLows[1];

      // Regular Bullish Divergence: Price makes a lower low while RSI makes a higher low
      if(lowBuffer[recent] < lowBuffer[previous] &&
         rsiBuffer[recent] > rsiBuffer[previous] &&
         (rsiBuffer[recent] - rsiBuffer[previous]) >= InpMinRSIDiff)
        {
         // Optionally require the earlier swing's RSI be oversold
         if(!InpUseRSIThreshold || rsiBuffer[previous] <= InpRSIOversold)
           {
            Print("Regular Bullish Divergence detected at bar ", recent);
            DisplaySignal("RegBullish Divergence", recent);
           }
        }
      // Hidden Bullish Divergence: Price makes a higher low while RSI makes a lower low
      else
         if(lowBuffer[recent] > lowBuffer[previous] &&
            rsiBuffer[recent] < rsiBuffer[previous] &&
            (rsiBuffer[previous] - rsiBuffer[recent]) >= InpMinRSIDiff)
           {
            Print("Hidden Bullish Divergence detected at bar ", recent);
            DisplaySignal("HiddenBullish Divergence", recent);
           }
     }
  }

//+------------------------------------------------------------------------+
//| IsSignificantSwingLow: Determines if the bar at 'index' is a swing low |
//+------------------------------------------------------------------------+
bool IsSignificantSwingLow(int index, int left, int right)
  {
   double currentLow = lowBuffer[index];
// Check left side for a local minimum condition
   for(int i = index - left; i < index; i++)
     {
      if(i < 0)
         continue;
      double pctDiff = MathAbs((lowBuffer[i] - currentLow) / currentLow) * 100.0;
      if(lowBuffer[i] < currentLow && pctDiff > InpMinSwingDiffPct)
         return false;
     }
// Check right side for a local minimum condition
   for(int i = index + 1; i <= index + right; i++)
     {
      if(i >= g_totalBars)
         break;
      double pctDiff = MathAbs((lowBuffer[i] - currentLow) / currentLow) * 100.0;
      if(lowBuffer[i] < currentLow && pctDiff > InpMinSwingDiffPct)
         return false;
     }
   return true;
  }

//+--------------------------------------------------------------------------+
//| IsSignificantSwingHigh: Determines if the bar at 'index' is a swing high |
//+--------------------------------------------------------------------------+
bool IsSignificantSwingHigh(int index, int left, int right)
  {
   double currentHigh = highBuffer[index];
// Check left side for a local maximum condition
   for(int i = index - left; i < index; i++)
     {
      if(i < 0)
         continue;
      double pctDiff = MathAbs((currentHigh - highBuffer[i]) / currentHigh) * 100.0;
      if(highBuffer[i] > currentHigh && pctDiff > InpMinSwingDiffPct)
         return false;
     }
// Check right side for a local maximum condition
   for(int i = index + 1; i <= index + right; i++)
     {
      if(i >= g_totalBars)
         break;
      double pctDiff = MathAbs((currentHigh - highBuffer[i]) / currentHigh) * 100.0;
      if(highBuffer[i] > currentHigh && pctDiff > InpMinSwingDiffPct)
         return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
//| DisplaySignal: Draws an arrow on the chart and records the signal|
//+------------------------------------------------------------------+
void DisplaySignal(string signalText, int barIndex)
  {
// Prevent duplicate signals on the same bar (or too close)
   for(int i = 0; i < ArraySize(signals); i++)
     {
      if(StringFind(signals[i].type, signalText) != -1)
         if(MathAbs(signals[i].barIndex - barIndex) < InpMinBarsBetweenSignals)
            return;
     }

// Update a "LatestSignal" label for regular signals.
   if(StringFind(signalText, "Reg") != -1)
     {
      string labelName = "LatestSignal";
      if(ObjectFind(0, labelName) == -1)
        {
         if(!ObjectCreate(0, labelName, OBJ_LABEL, 0, 0, 0))
           {
            Print("Failed to create LatestSignal label");
            return;
           }
         ObjectSetInteger(0, labelName, OBJPROP_CORNER, 0);
         ObjectSetInteger(0, labelName, OBJPROP_XDISTANCE, 10);
         ObjectSetInteger(0, labelName, OBJPROP_YDISTANCE, 20);
         ObjectSetInteger(0, labelName, OBJPROP_COLOR, clrWhite);
        }
      ObjectSetString(0, labelName, OBJPROP_TEXT, signalText);
     }

// Create an arrow object for the signal.
   string arrowName = "Arrow_" + signalText + "_" + IntegerToString(barIndex);
   if(ObjectFind(0, arrowName) < 0)
     {
      int arrowCode = 0;
      double arrowPrice = 0.0;
      color arrowColor = clrWhite;
      double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

      if(StringFind(signalText, "Bullish") != -1)
        {
         arrowCode  = 233; // Wingdings up arrow
         arrowColor = clrLime;
         arrowPrice = lowBuffer[barIndex] - (InpArrowOffset * point);
        }
      else
         if(StringFind(signalText, "Bearish") != -1)
           {
            arrowCode  = 234; // Wingdings down arrow
            arrowColor = clrRed;
            arrowPrice = highBuffer[barIndex] + (InpArrowOffset * point);
           }

      if(!ObjectCreate(0, arrowName, OBJ_ARROW, 0, timeBuffer[barIndex], arrowPrice))
        {
         Print("Failed to create arrow object ", arrowName);
         return;
        }
      ObjectSetInteger(0, arrowName, OBJPROP_COLOR, arrowColor);
      ObjectSetInteger(0, arrowName, OBJPROP_ARROWCODE, arrowCode);
     }

// Record the signal for evaluation.
   SignalInfo sig;
   sig.type = signalText;
   sig.barIndex = barIndex;
   sig.signalTime = timeBuffer[barIndex];
   if(StringFind(signalText, "Bullish") != -1)
      sig.signalPrice = lowBuffer[barIndex];
   else
      sig.signalPrice = highBuffer[barIndex];
   ArrayResize(signals, ArraySize(signals) + 1);
   signals[ArraySize(signals) - 1] = sig;

   UpdateSignalCountLabel();
  }

//+------------------------------------------------------------------+
//| UpdateSignalCountLabel: Updates a label showing signal counts    |
//+------------------------------------------------------------------+
void UpdateSignalCountLabel()
  {
   int regCount = 0, hidCount = 0;
   for(int i = 0; i < ArraySize(signals); i++)
     {
      if(StringFind(signals[i].type, "Reg") != -1)
         regCount++;
      else
         if(StringFind(signals[i].type, "Hidden") != -1)
            hidCount++;
     }
   string countText = "Regular Signals: " + IntegerToString(regCount) +
                      "\nHidden Signals: " + IntegerToString(hidCount);
   string countLabel = "SignalCount";
   if(ObjectFind(0, countLabel) == -1)
     {
      if(!ObjectCreate(0, countLabel, OBJ_LABEL, 0, 0, 0))
        {
         Print("Failed to create SignalCount label");
         return;
        }
      ObjectSetInteger(0, countLabel, OBJPROP_CORNER, 0);
      ObjectSetInteger(0, countLabel, OBJPROP_XDISTANCE, 10);
      ObjectSetInteger(0, countLabel, OBJPROP_YDISTANCE, 40);
      ObjectSetInteger(0, countLabel, OBJPROP_COLOR, clrYellow);
     }
   ObjectSetString(0, countLabel, OBJPROP_TEXT, countText);
  }

//+--------------------------------------------------------------------+
//| EvaluateSignalsAndPrint: After backtesting, prints signal accuracy |
//+--------------------------------------------------------------------+
void EvaluateSignalsAndPrint()
  {
   double closeAll[];
   int totalBars = CopyClose(_Symbol, _Period, 0, WHOLE_ARRAY, closeAll);
   if(totalBars <= 0)
     {
      Print("Error copying complete close data for evaluation");
      return;
     }
   ArraySetAsSeries(closeAll, true);

   int totalEvaluated = 0, regTotal = 0, hidTotal = 0;
   int regEval = 0, hidEval = 0;
   int regCorrect = 0, hidCorrect = 0;

   for(int i = 0; i < ArraySize(signals); i++)
     {
      int evalIndex = signals[i].barIndex - InpEvalBars;
      if(evalIndex < 0)
         continue;
      double evalClose = closeAll[evalIndex];

      if(StringFind(signals[i].type, "Bullish") != -1)
        {
         if(StringFind(signals[i].type, "Reg") != -1)
           {
            regTotal++;
            regEval++;
            if(evalClose > signals[i].signalPrice)
               regCorrect++;
           }
         else
            if(StringFind(signals[i].type, "Hidden") != -1)
              {
               hidTotal++;
               hidEval++;
               if(evalClose > signals[i].signalPrice)
                  hidCorrect++;
              }
         totalEvaluated++;
        }
      else
         if(StringFind(signals[i].type, "Bearish") != -1)
           {
            if(StringFind(signals[i].type, "Reg") != -1)
              {
               regTotal++;
               regEval++;
               if(evalClose < signals[i].signalPrice)
                  regCorrect++;
              }
            else
               if(StringFind(signals[i].type, "Hidden") != -1)
                 {
                  hidTotal++;
                  hidEval++;
                  if(evalClose < signals[i].signalPrice)
                     hidCorrect++;
                 }
            totalEvaluated++;
           }
     }

   double overallAccuracy = (totalEvaluated > 0) ? (double)(regCorrect + hidCorrect) / totalEvaluated * 100.0 : 0.0;
   double regAccuracy = (regEval > 0) ? (double)regCorrect / regEval * 100.0 : 0.0;
   double hidAccuracy = (hidEval > 0) ? (double)hidCorrect / hidEval * 100.0 : 0.0;

   Print("----- Backtest Signal Evaluation -----");
   Print("Total Signals Generated: ", ArraySize(signals));
   Print("Signals Evaluated: ", totalEvaluated);
   Print("Overall Accuracy: ", DoubleToString(overallAccuracy, 2), "%");
   Print("Regular Signals: ", regTotal, " | Evaluated: ", regEval, " | Accuracy: ", DoubleToString(regAccuracy, 2), "%");
   Print("Hidden Signals:  ", hidTotal, " | Evaluated: ", hidEval, " | Accuracy: ", DoubleToString(hidAccuracy, 2), "%");
  }
//+------------------------------------------------------------------+
```

### Code Breakdown

**1.** Header Information and Input Parameters

At the very top of our script, we have a well-defined header that provides key information about the code.

File and Author Information

The header specifies the file name ( _RSI Divergence.mql5_), the copyright notice, and a link to the author’s profile. This ensures proper attribution and gives users a reference point if they need to check for updates or additional documentation

Versioning and Compilation Directives

The _#property_ directives set important properties such as the version number and the use of strict compilation rules _(#property strict)_. This helps in maintaining consistency and reducing potential errors during development and deployment. Moving on, the input parameters section is essential for customization. These parameters allow you, or any user, to fine-tune the behavior of the divergence detection logic without modifying the core code. Below are a few highlights:

RSI and Swing Detection Parameters

- _InpRSIPeriod:_ Sets the period for the RSI indicator.
- _InpSwingLeft_ and _InpSwingRight_: Define how many bars on each side are considered when detecting swing points. Adjusting these values makes the swing detection either more relaxed or more strict.

Divergence and Signal Evaluation Settings

- _InpLookback:_ Determines how many bars in the past the script will scan for divergences.
- _InpEvalBars:_ Specifies the number of bars to wait before evaluating if a signal was successful.
- _InpMinBarsBetweenSignals:_ Helps in avoiding duplicate signals by enforcing a minimum bar separation between similar signals.

Display Customizations

- InpArrowOffset: Sets the distance (in points) that arrows are offset from the swing point, enhancing visual clarity on the chart.

Optional RSI Threshold Filter

- _InpUseRSIThreshold_, along with _InpRSIOversold_ and _InpRSIOverbought_, provides an extra layer of filtering. This ensures that, for bullish divergence, the earlier RSI swing is in the oversold region—if the user chooses to enable this filter.

```
//+------------------------------------------------------------------+
//|                                           RSI Divergence.mql5    |
//|                               Copyright 2025, Christian Benjamin |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "2025, MetaQuotes Software Corp."
#property link      "https://www.mql5.com/en/users/lynnchris"
#property version   "1.0"
#property strict

//---- Input parameters
input int    InpRSIPeriod             = 14;    // RSI period
input int    InpSwingLeft             = 1;     // Bars to the left for swing detection (relaxed)
input int    InpSwingRight            = 1;     // Bars to the right for swing detection (relaxed)
input int    InpLookback              = 100;   // Number of bars to scan for divergence
input int    InpEvalBars              = 5;     // Bars after which to evaluate a signal
input int    InpMinBarsBetweenSignals = 1;     // Minimum bars between same-type signals (allows frequent re-entry)
input double InpArrowOffset           = 3.0;   // Arrow offset (in points) for display
input double InpMinSwingDiffPct       = 0.05;  // Lower minimum % difference to qualify as a swing
input double InpMinRSIDiff            = 1.0;   // Lower minimum difference in RSI between swing points

// Optional RSI threshold filter for bullish divergence (disabled by default)
input bool   InpUseRSIThreshold       = false; // If true, require earlier RSI swing to be oversold for bullish divergence
input double InpRSIOversold           = 30;    // RSI oversold level
input double InpRSIOverbought         = 70;    // RSI overbought level (if needed for bearish)
```

**2.** Indicator Initialization

In this part, we initialize our RSI indicator. The _OnInit()_ function creates a handle for the RSI indicator using parameters such as symbol, timeframe, and the RSI period specified by the user. This step is crucial because every subsequent operation depends on having a valid RSI handle to retrieve indicator data.

- The _iRSI_ function is called with the necessary parameters.
- Error handling is implemented to catch any failure in creating the handle.
- The initialization ensures that our indicator is ready for data acquisition and analysis.

```
int OnInit()
{
   // Create the RSI indicator handle with the specified period
   rsiHandle = iRSI(_Symbol, _Period, InpRSIPeriod, PRICE_CLOSE);
   if(rsiHandle == INVALID_HANDLE)
   {
      Print("Error creating RSI handle");
      return(INIT_FAILED);
   }
   return(INIT_SUCCEEDED);
}
```

**3.** Data Acquisition on Each New Candle

In the _OnTick()_ function, we check if a new candle has closed before processing. This ensures that our analysis is always working with completed data. We then copy arrays of RSI values, lows, highs, closes, and time data over a configurable lookback period. Setting the arrays as series makes sure that the data is ordered with the most recent bar at index 0.

- The code waits for the next closed candle to avoid processing incomplete data.
- RSI and price data are retrieved using functions like _CopyBuffer_ and _CopyLow/High/Close/Time_.
- Using _ArraySetAsSeries_ preserves the correct order for time series analysis.

```
void OnTick()
{
   // Process only once per new closed candle by comparing bar times
   datetime currentBarTime = iTime(_Symbol, _Period, 1);
   if(currentBarTime == lastBarTime)
      return;
   lastBarTime = currentBarTime;

   // Copy RSI data for a given lookback period
   ArrayResize(rsiBuffer, InpLookback);
   ArraySetAsSeries(rsiBuffer, true);
   if(CopyBuffer(rsiHandle, 0, 0, InpLookback, rsiBuffer) <= 0)
   {
      Print("Error copying RSI data");
      return;
   }

   // Copy price data (lows, highs, closes) and time data for analysis
   ArrayResize(lowBuffer, InpLookback);
   ArrayResize(highBuffer, InpLookback);
   ArrayResize(closeBuffer, InpLookback);
   ArraySetAsSeries(lowBuffer, true);
   ArraySetAsSeries(highBuffer, true);
   ArraySetAsSeries(closeBuffer, true);
   if(CopyLow(_Symbol, _Period, 0, InpLookback, lowBuffer) <= 0 ||
      CopyHigh(_Symbol, _Period, 0, InpLookback, highBuffer) <= 0 ||
      CopyClose(_Symbol, _Period, 0, InpLookback, closeBuffer) <= 0)
   {
      Print("Error copying price data");
      return;
   }

   ArrayResize(timeBuffer, InpLookback);
   ArraySetAsSeries(timeBuffer, true);
   if(CopyTime(_Symbol, _Period, 0, InpLookback, timeBuffer) <= 0)
   {
      Print("Error copying time data");
      return;
   }

   g_totalBars = InpLookback;

   // (Further processing follows here...)
}
```

**4.** Swing Detection (Identifying Swing Lows and Swing Highs)

Before we can detect divergences, we must first pinpoint significant swing points. Two helper functions, _IsSignificantSwingLow_ and _IsSignificantSwingHigh,_ are used to identify local minimums and maximums. They do this by comparing a bar’s low or high with its neighboring bars within a given window and checking that the percentage difference meets a set threshold.

- The functions check both left and right of the current bar.
- They calculate the percentage difference to ensure that only significant swings are marked.
- This filtering reduces noise, ensuring that our divergence analysis focuses on meaningful market moves.

```
bool IsSignificantSwingLow(int index, int left, int right)
{
   double currentLow = lowBuffer[index];
   // Check left side for local minimum condition
   for(int i = index - left; i < index; i++)
   {
      if(i < 0) continue;
      double pctDiff = MathAbs((lowBuffer[i] - currentLow) / currentLow) * 100.0;
      if(lowBuffer[i] < currentLow && pctDiff > InpMinSwingDiffPct)
         return false;
   }
   // Check right side for local minimum condition
   for(int i = index + 1; i <= index + right; i++)
   {
      if(i >= g_totalBars) break;
      double pctDiff = MathAbs((lowBuffer[i] - currentLow) / currentLow) * 100.0;
      if(lowBuffer[i] < currentLow && pctDiff > InpMinSwingDiffPct)
         return false;
   }
   return true;
}

bool IsSignificantSwingHigh(int index, int left, int right)
{
   double currentHigh = highBuffer[index];
   // Check left side for local maximum condition
   for(int i = index - left; i < index; i++)
   {
      if(i < 0) continue;
      double pctDiff = MathAbs((currentHigh - highBuffer[i]) / currentHigh) * 100.0;
      if(highBuffer[i] > currentHigh && pctDiff > InpMinSwingDiffPct)
         return false;
   }
   // Check right side for local maximum condition
   for(int i = index + 1; i <= index + right; i++)
   {
      if(i >= g_totalBars) break;
      double pctDiff = MathAbs((currentHigh - highBuffer[i]) / currentHigh) * 100.0;
      if(highBuffer[i] > currentHigh && pctDiff > InpMinSwingDiffPct)
         return false;
   }
   return true;
}
```

**5.** Divergence Detection: Bearish and Bullish Divergences

Once swing points are identified, the algorithm compares recent swings to detect divergences. For bearish divergence, the code looks at two swing highs and checks if the price is making a higher high while the RSI shows a lower high (or vice versa for hidden bearish divergence). For bullish divergence, it similarly compares two swing lows. An optional RSI threshold can further validate bullish signals by ensuring the earlier RSI reading is in the oversold territory.

- Two recent swing points (either highs or lows) are used for divergence analysis.
- The conditions for regular and hidden divergences are clearly separated.
- Optional parameters (like the RSI oversold condition) provide additional filtering for signal strength.

```
// --- Bearish Divergence (using swing highs)
if(ArraySize(swingHighs) >= 2)
{
   ArraySort(swingHighs); // Ensure ascending order: index 0 is most recent
   int recent   = swingHighs[0];
   int previous = swingHighs[1];

   // Regular Bearish Divergence: Price makes a higher high while RSI makes a lower high
   if(highBuffer[recent] > highBuffer[previous] &&
      rsiBuffer[recent] < rsiBuffer[previous] &&
      (rsiBuffer[previous] - rsiBuffer[recent]) >= InpMinRSIDiff)
   {
      Print("Regular Bearish Divergence detected at bar ", recent);
      DisplaySignal("RegBearish Divergence", recent);
   }
   // Hidden Bearish Divergence: Price makes a lower high while RSI makes a higher high
   else if(highBuffer[recent] < highBuffer[previous] &&
           rsiBuffer[recent] > rsiBuffer[previous] &&
           (rsiBuffer[recent] - rsiBuffer[previous]) >= InpMinRSIDiff)
   {
      Print("Hidden Bearish Divergence detected at bar ", recent);
      DisplaySignal("HiddenBearish Divergence", recent);
   }
}

// --- Bullish Divergence (using swing lows)
if(ArraySize(swingLows) >= 2)
{
   ArraySort(swingLows); // Ensure ascending order: index 0 is most recent
   int recent   = swingLows[0];
   int previous = swingLows[1];

   // Regular Bullish Divergence: Price makes a lower low while RSI makes a higher low
   if(lowBuffer[recent] < lowBuffer[previous] &&
      rsiBuffer[recent] > rsiBuffer[previous] &&
      (rsiBuffer[recent] - rsiBuffer[previous]) >= InpMinRSIDiff)
   {
      // Optionally require the earlier RSI swing to be oversold
      if(!InpUseRSIThreshold || rsiBuffer[previous] <= InpRSIOversold)
      {
         Print("Regular Bullish Divergence detected at bar ", recent);
         DisplaySignal("RegBullish Divergence", recent);
      }
   }
   // Hidden Bullish Divergence: Price makes a higher low while RSI makes a lower low
   else if(lowBuffer[recent] > lowBuffer[previous] &&
           rsiBuffer[recent] < rsiBuffer[previous] &&
           (rsiBuffer[previous] - rsiBuffer[recent]) >= InpMinRSIDiff)
   {
      Print("Hidden Bullish Divergence detected at bar ", recent);
      DisplaySignal("HiddenBullish Divergence", recent);
   }
}
```

**6.** Signal Display and Recording

When a divergence is detected, it’s important to mark the signal visually and record its details for later evaluation. The _DisplaySignal()_ function not only creates an arrow on the chart (using different arrow codes and colors for bullish and bearish signals) but also updates a label for the latest signal and stores the signal’s metadata in a global array. This systematic recording enables later _backtesting_ of the strategy.

- Duplicate signals are prevented by checking if a signal for a similar bar already exists.
- Visual cues like arrows and labels enhance the readability of the chart.
- Every signal is stored with details such as type, bar index, time, and price, facilitating later performance evaluation.

```
void DisplaySignal(string signalText, int barIndex)
{
   // Prevent duplicate signals on the same or nearby bars
   for(int i = 0; i < ArraySize(signals); i++)
   {
      if(StringFind(signals[i].type, signalText) != -1)
         if(MathAbs(signals[i].barIndex - barIndex) < InpMinBarsBetweenSignals)
            return;
   }

   // Update a label for the latest regular signal
   if(StringFind(signalText, "Reg") != -1)
   {
      string labelName = "LatestSignal";
      if(ObjectFind(0, labelName) == -1)
      {
         if(!ObjectCreate(0, labelName, OBJ_LABEL, 0, 0, 0))
         {
            Print("Failed to create LatestSignal label");
            return;
         }
         ObjectSetInteger(0, labelName, OBJPROP_CORNER, 0);
         ObjectSetInteger(0, labelName, OBJPROP_XDISTANCE, 10);
         ObjectSetInteger(0, labelName, OBJPROP_YDISTANCE, 20);
         ObjectSetInteger(0, labelName, OBJPROP_COLOR, clrWhite);
      }
      ObjectSetString(0, labelName, OBJPROP_TEXT, signalText);
   }

   // Create an arrow object to mark the signal on the chart
   string arrowName = "Arrow_" + signalText + "_" + IntegerToString(barIndex);
   if(ObjectFind(0, arrowName) < 0)
   {
      int arrowCode = 0;
      double arrowPrice = 0.0;
      color arrowColor = clrWhite;
      double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

      if(StringFind(signalText, "Bullish") != -1)
      {
         arrowCode  = 233; // Wingdings up arrow
         arrowColor = clrLime;
         arrowPrice = lowBuffer[barIndex] - (InpArrowOffset * point);
      }
      else if(StringFind(signalText, "Bearish") != -1)
      {
         arrowCode  = 234; // Wingdings down arrow
         arrowColor = clrRed;
         arrowPrice = highBuffer[barIndex] + (InpArrowOffset * point);
      }

      if(!ObjectCreate(0, arrowName, OBJ_ARROW, 0, timeBuffer[barIndex], arrowPrice))
      {
         Print("Failed to create arrow object ", arrowName);
         return;
      }
      ObjectSetInteger(0, arrowName, OBJPROP_COLOR, arrowColor);
      ObjectSetInteger(0, arrowName, OBJPROP_ARROWCODE, arrowCode);
   }

   // Record the signal details for later backtesting evaluation
   SignalInfo sig;
   sig.type = signalText;
   sig.barIndex = barIndex;
   sig.signalTime = timeBuffer[barIndex];
   if(StringFind(signalText, "Bullish") != -1)
      sig.signalPrice = lowBuffer[barIndex];
   else
      sig.signalPrice = highBuffer[barIndex];
   ArrayResize(signals, ArraySize(signals) + 1);
   signals[ArraySize(signals) - 1] = sig;

   UpdateSignalCountLabel();
}
```

Backtest Evaluation on Deinitialization

Finally, the _EvaluateSignalsAndPrint()_ function is called when the expert is _deinitialized_. This function retrospectively evaluates all recorded signals by comparing the price movement a few bars after the signal against the signal’s recorded price. It calculates the accuracy for both regular and hidden signals, providing valuable feedback on the performance of our divergence strategy.

- The function retrieves complete historical close data.
- Each signal is evaluated after a fixed number of bars (as set by _InpEvalBars_).
- Accuracy metrics are computed for overall signals as well as separately for regular and hidden signals, aiding in performance validation.

```
void EvaluateSignalsAndPrint()
{
   double closeAll[];
   int totalBars = CopyClose(_Symbol, _Period, 0, WHOLE_ARRAY, closeAll);
   if(totalBars <= 0)
   {
      Print("Error copying complete close data for evaluation");
      return;
   }
   ArraySetAsSeries(closeAll, true);

   int totalEvaluated = 0, regTotal = 0, hidTotal = 0;
   int regEval = 0, hidEval = 0;
   int regCorrect = 0, hidCorrect = 0;

   for(int i = 0; i < ArraySize(signals); i++)
   {
      int evalIndex = signals[i].barIndex - InpEvalBars;
      if(evalIndex < 0)
         continue;
      double evalClose = closeAll[evalIndex];

      if(StringFind(signals[i].type, "Bullish") != -1)
      {
         if(StringFind(signals[i].type, "Reg") != -1)
         {
            regTotal++;
            regEval++;
            if(evalClose > signals[i].signalPrice)
               regCorrect++;
         }
         else if(StringFind(signals[i].type, "Hidden") != -1)
         {
            hidTotal++;
            hidEval++;
            if(evalClose > signals[i].signalPrice)
               hidCorrect++;
         }
         totalEvaluated++;
      }
      else if(StringFind(signals[i].type, "Bearish") != -1)
      {
         if(StringFind(signals[i].type, "Reg") != -1)
         {
            regTotal++;
            regEval++;
            if(evalClose < signals[i].signalPrice)
               regCorrect++;
         }
         else if(StringFind(signals[i].type, "Hidden") != -1)
         {
            hidTotal++;
            hidEval++;
            if(evalClose < signals[i].signalPrice)
               hidCorrect++;
         }
         totalEvaluated++;
      }
   }

   double overallAccuracy = (totalEvaluated > 0) ? (double)(regCorrect + hidCorrect) / totalEvaluated * 100.0 : 0.0;
   double regAccuracy = (regEval > 0) ? (double)regCorrect / regEval * 100.0 : 0.0;
   double hidAccuracy = (hidEval > 0) ? (double)hidCorrect / hidEval * 100.0 : 0.0;

   Print("----- Backtest Signal Evaluation -----");
   Print("Total Signals Generated: ", ArraySize(signals));
   Print("Signals Evaluated: ", totalEvaluated);
   Print("Overall Accuracy: ", DoubleToString(overallAccuracy, 2), "%");
   Print("Regular Signals: ", regTotal, " | Evaluated: ", regEval, " | Accuracy: ", DoubleToString(regAccuracy, 2), "%");
   Print("Hidden Signals:  ", hidTotal, " | Evaluated: ", hidEval, " | Accuracy: ", DoubleToString(hidAccuracy, 2), "%");
}
```

### Testing and Results

After successfully compiling your EA using _MetaEditor_, drag your EA onto the chart for testing. Make sure you're using a demo account to avoid risking real money. You can also add the RSI indicator to your chart for easy signal confirmation while testing your EA. To do this, navigate to the Indicators tab, select the RSI indicator under the Panels folder, and set your preferred parameters, making sure they match those in your EA. Check out the GIF below, which illustrates how to add the RSI indicator window on an MetaTrader 5 chart. You can also see the confirmed signal, a regular bullish divergence on a one-minute timeframe.

![Bullish Convergence](https://c.mql5.com/2/119/Bullish_convergence.gif)

Fig 5. Setting Indicator and Test Result 1

Below is another test we conducted on Boom 500, confirmed by both the price action and the RSI indicator, showing a sell signal.

![testing](https://c.mql5.com/2/119/TEST_RESULT.png)

Fig 6. Test Result 2

Another test was conducted using backtesting on the GIF below, which shows several positive shifts. If you look closely, you'll notice both hidden continuation signals and regular signals. However, a few signals need to be filtered out due to a lack of confirmation, despite their positive impact.

![Backtesting](https://c.mql5.com/2/119/Backtesting.gif)

Fig 7. Test Result 3

### Conclusion

This tool has proven to be extremely aligned with price action, which is the core aim of our series to create as many price action analysis tools as possible. I have truly appreciated how effectively the RSI indicator interacts with price action by extracting positive signals from divergences. The tests we conducted have shown promising results and a positive trend.

However, I believe it is time to introduce another enhancement that uses external libraries for precise and accurate swing identification, thereby improving signal accuracy. My advice is to test the tool thoroughly and adjust its parameters to suit your trading style. Remember that each generated signal should be cross-checked before entry, as the tool is designed to help you monitor the market and confirm your overall strategy.

| Date | Tool Name | Description | Version | Updates | Notes |
| --- | --- | --- | --- | --- | --- |
| 01/10/24 | [Chart Projector](https://www.mql5.com/en/articles/16014) | Script to overlay the previous day's price action with ghost effect. | 1.0 | Initial Release | First tool in Lynnchris Tool Chest |
| 18/11/24 | [Analytical Comment](https://www.mql5.com/en/articles/15927) | It provides previous day's information in a tabular format, as well as anticipates the future direction of the market. | 1.0 | Initial Release | Second tool in the Lynnchris Tool Chest |
| 27/11/24 | [Analytics Master](https://www.mql5.com/en/articles/16434) | Regular Update of market metrics after every two hours | 1.01 | Second Release | Third tool in the Lynnchris Tool Chest |
| 02/12/24 | [Analytics Forecaster](https://www.mql5.com/en/articles/16559) | Regular Update of market metrics after every two hours with telegram integration | 1.1 | Third Edition | Tool number 4 |
| 09/12/24 | [Volatility Navigator](https://www.mql5.com/en/articles/16560) | The EA analyzes market conditions using the Bollinger Bands, RSI and ATR indicators | 1.0 | Initial Release | Tool Number 5 |
| 19/12/24 | [Mean Reversion Signal Reaper](https://www.mql5.com/en/articles/16700) | Analyzes market using mean reversion strategy and provides signal | 1.0 | Initial Release | Tool number 6 |
| 9/01/25 | [Signal Pulse](https://www.mql5.com/en/articles/16861) | Multiple timeframe analyzer | 1.0 | Initial Release | Tool number 7 |
| 17/01/25 | [Metrics Board](https://www.mql5.com/en/articles/16584) | Panel with button for analysis | 1.0 | Initial Release | Tool number 8 |
| 21/01/25 | [External Flow](https://www.mql5.com/en/articles/16967) | Analytics through external libraries | 1.0 | Initial Release | Tool number 9 |
| 27/01/25 | [VWAP](https://www.mql5.com/en/articles/16984) | Volume Weighted Average Price | 1.3 | Initial Release | Tool number 10 |
| 02/02/25 | [Heikin Ashi](https://www.mql5.com/en/articles/17021) | Trend Smoothening and reversal signal identification | 1.0 | Initial Release | Tool number 11 |
| 04/02/25 | [FibVWAP](https://www.mql5.com/en/articles/17121) | Signal generation through python analysis | 1.0 | Initial Release | Tool number  12 |
| 14/02/25 | RSI DIVERGENCE | Price action versus RSI divergences | 1.0 | Initial Release | Tool number 13 |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17198.zip "Download all attachments in the single ZIP archive")

[RSI\_DIVERGENCE.mq5](https://www.mql5.com/en/articles/download/17198/rsi_divergence.mq5 "Download RSI_DIVERGENCE.mq5")(33.61 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Price Action Analysis Toolkit (Part 55): Designing a CPI Mini-Candle Overlay for Intra-bar Pressure](https://www.mql5.com/en/articles/20949)
- [Price Action Analysis Toolkit Development (Part 54): Filtering Trends with EMA and Smoothed Price Action](https://www.mql5.com/en/articles/20851)
- [Price Action Analysis Toolkit Development (Part 53): Pattern Density Heatmap for Support and Resistance Zone Discovery](https://www.mql5.com/en/articles/20390)
- [Price Action Analysis Toolkit Development (Part 52): Master Market Structure with Multi-Timeframe Visual Analysis](https://www.mql5.com/en/articles/20387)
- [Price Action Analysis Toolkit Development (Part 51): Revolutionary Chart Search Technology for Candlestick Pattern Discovery](https://www.mql5.com/en/articles/20313)
- [Price Action Analysis Toolkit Development (Part 50): Developing the RVGI, CCI and SMA Confluence Engine in MQL5](https://www.mql5.com/en/articles/20262)
- [Price Action Analysis Toolkit Development (Part 49): Integrating Trend, Momentum, and Volatility Indicators into One MQL5 System](https://www.mql5.com/en/articles/20168)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/481589)**
(1)


![linfo2](https://c.mql5.com/avatar/2023/4/6438c14d-e2f0.png)

**[linfo2](https://www.mql5.com/en/users/neilhazelwood)**
\|
18 Feb 2025 at 20:05

Thanks you Chris once again for your well thought out and helpful articles . Much appreciated as a template for me to modify . I also appreciate your erudite thinking and practical implementations. For me I had issues with EvaluateSignalsAndPrint , it is written the function returns Error Copying Compete close data for evaluation . The last error is '4003' . I am not smart enough to work out why when  using the [WHOLE\_ARRAY](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants "MQL5 documentation: Other Constants") variable the function fails  . For me if I replace the 'WHOLE\_ARRAY in the copyclose  with a count of the closed bars I can get a returned 'Backtest Signal Evaluation' . BTW  I am in a strange time zone GMT +13 and sometime have issues with local and server dates and times but maybe this may help someone


![Automating Trading Strategies in MQL5 (Part 7): Building a Grid Trading EA with Dynamic Lot Scaling](https://c.mql5.com/2/119/Automating_Trading_Strategies_in_MQL5_Part_7__LOGO.png)[Automating Trading Strategies in MQL5 (Part 7): Building a Grid Trading EA with Dynamic Lot Scaling](https://www.mql5.com/en/articles/17190)

In this article, we build a grid trading expert advisor in MQL5 that uses dynamic lot scaling. We cover the strategy design, code implementation, and backtesting process. Finally, we share key insights and best practices for optimizing the automated trading system.

![Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (II): Modularization](https://c.mql5.com/2/119/Creating_a_Trading_Administrator_Panel_in_MQL5_Part_IX___LOGO.png)[Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (II): Modularization](https://www.mql5.com/en/articles/16562)

In this discussion, we take a step further in breaking down our MQL5 program into smaller, more manageable modules. These modular components will then be integrated into the main program, enhancing its organization and maintainability. This approach simplifies the structure of our main program and makes the individual components reusable in other Expert Advisors (EAs) and indicator developments. By adopting this modular design, we create a solid foundation for future enhancements, benefiting both our project and the broader developer community.

![Animal Migration Optimization (AMO) algorithm](https://c.mql5.com/2/90/logo-amo_15543.png)[Animal Migration Optimization (AMO) algorithm](https://www.mql5.com/en/articles/15543)

The article is devoted to the AMO algorithm, which models the seasonal migration of animals in search of optimal conditions for life and reproduction. The main features of AMO include the use of topological neighborhood and a probabilistic update mechanism, which makes it easy to implement and flexible for various optimization tasks.

![Developing a Replay System (Part 59): A New Future](https://c.mql5.com/2/87/Desenvolvendo_um_sistema_de_Replay_Parte_59__LOGO__3.png)[Developing a Replay System (Part 59): A New Future](https://www.mql5.com/en/articles/12075)

Having a proper understanding of different ideas allows us to do more with less effort. In this article, we'll look at why it's necessary to configure a template before the service can interact with the chart. Also, what if we improve the mouse pointer so we can do more things with it?

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/17198&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062591227611751711)

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