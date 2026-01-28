---
title: Price Action Analysis Toolkit Development (Part 21): Market Structure Flip Detector Tool
url: https://www.mql5.com/en/articles/17891
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:36:09.781521
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/17891&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062583565390095607)

MetaTrader 5 / Trading systems


### Contents

- [Introduction](https://www.mql5.com/en/articles/17891#para2)
- [Why This Tool Is Important](https://www.mql5.com/en/articles/17891#para3)
- [Action Plan Outline](https://www.mql5.com/en/articles/17891#para4)
- [Expert Advisor Anatomy](https://www.mql5.com/en/articles/17891#para5)
- [Source Code Listing](https://www.mql5.com/en/articles/17891#para6)
- [Performance Outcomes](https://www.mql5.com/en/articles/17891#para7)
- [Conclusion](https://www.mql5.com/en/articles/17891#para8)

### Introduction

This EA tackles one of the toughest challenges in trading: false reversal signals. Choppy price action throws off simple pivot routines, trapping traders in whipsaws. The Market Structure Flip Detector solves this by converting ATR into a flexible bar-count filter. It ignores minor swings, captures only valid highs and lows, and then flags a bearish flip when a higher-high becomes a lower-high or a bullish flip when a lower-low turns into a higher-low. Along the way, you’ll see how to:

- Convert ATR into a deep measure that expands in wild markets and contracts in calm ones.
- Confirm pivots by scanning an exact number of bars on each side of a candidate high or low.
- Maintain trend bias, so flips trigger only after the previous structure breaks.
- Annotate signals with on-chart arrows, pivot labels, and a live stats panel that tracks flip counts and timing.

By the end, you’ll have a resilient EA that cuts through market noise and delivers only the cleanest, rule-based reversal cues complete with sound and push alerts when real flips occur.

### Why Is This Tool Important

False reversal signals can account for over half of all pivot-based triggers in choppy markets, leading to frequent whipsaws and negative expectancy. By tying our swing-point filter to the Average True Range (ATR), which Wilder introduced in 1978 to measure volatility more accurately than simple high-low ranges, we adjust the required “depth” for a pivot in real time. When ATR rises during volatile bursts, our filter widens to ignore small, erratic swings; when markets calm, it tightens to catch genuine turns promptly.

True Range ( **TR**) per bar is defined as:

![TR Fomula](https://c.mql5.com/2/136/f1.PNG)

ATR is the n-period simple moving average of TR (default n = 14)

We convert **ATR** into a bar-count depth **d** via:

![ATR](https://c.mql5.com/2/136/f2.PNG)

Where **Point** is the minimum price increment, **m** is the ATR multiplier and **f** the loosen factor.

A bearish flip happens when we first record a higher-high, then see the next swing high drop below that peak. A bullish flip works the other way around. If we assume returns follow a normal distribution with an average of zero and a variance we’ll call “volatility squared,” then the chance of any single bar being the highest within a span of “two times our depth setting plus one” bars is simply one over that same “two times depth plus one.” Since we tie our depth setting directly to market volatility; estimating volatility as ATR divided by the square root of one-half pi, we directly manage our rate of false signals. In practice, this lets us pick inputs that keep noise to around 5%.

[TradeStation research](https://www.mql5.com/go?link=https://medium.com/@redsword_23261/enhanced-dual-pivot-point-reversal-trading-strategy-3fc9ea6bea45 "https://medium.com/%40redsword_23261/enhanced-dual-pivot-point-reversal-trading-strategy-3fc9ea6bea45") demonstrated ATR-based pivot windows reduce noise-driven trades by ~40% and boost net profit by ~22% over 5 years on S&P 500 data. [QuantifiedStrategies.com](https://www.mql5.com/go?link=https://www.quantifiedstrategies.com/average-true-range-trading-strategy/ "https://www.quantifiedstrategies.com/average-true-range-trading-strategy/") reports that ATR-filtered pivots improve hit rate from ~35% to ~58% and raise the average reward-risk ratio from ~1.1 to ~1.8 in backtests on EURUSD and ES futures. Community feedback on TradingView highlights that ATR-windowed pivot tools align closely with institutional order-flow breaks, especially on 1 H and 4 H charts.

### Action Plan Outline

This EA filters out market noise by turning the current ATR reading into a flexible “depth” window; wider in volatile conditions, narrower in calm ones, and then validates each closed-bar high or low against its neighbors within that window. It remembers the last two confirmed swing highs and lows and tracks a simple bias flag that flips to “up” when a new high exceeds the previous one or to “down” when a new low falls below its predecessor. When the market structure reverses; i.e., in an upstate the latest high is actually lower than the prior high (bearish flip), or in a down state the latest low is higher than the prior low (bullish flip)—the EA drops a colored arrow on the chart, labels both pivots, updates a live stats panel, and can trigger sound or push alerts. This approach ensures you see only genuine “higher-high to lower-high” or “lower-low to higher-low” reversals.

Bearish Flip

An uptrend is identified when the price achieves two consecutive higher highs. The Expert Advisor (EA) subsequently looks for a swing high that is lower than the previous swing high. A swing high is defined as the highest point within a window based on the Average True Range (ATR). When the EA is in an "up" state and detects this lower high, it marks the corresponding bar with a red arrow labeled "LH." In addition to this visual indication, the EA generates an alert and records the bearish flip, signaling that sellers are beginning to assert control.

![Bearish Flip](https://c.mql5.com/2/137/Bearish_Swing.png)

Bullish Flip

A downtrend is established when the price registers two consecutive lower lows. Subsequently, the Expert Advisor (EA) identifies a swing low that exceeds the previous swing low. A swing low is defined as the lowest point within a window based on Average True Range (ATR). When the EA is in a "down" state and detects this higher low, it plots a green arrow labeled "HL" on the corresponding bar. Additionally, it generates an alert and records the bullish flip to indicate a potential return of buyer interest.

![Bullish Flip](https://c.mql5.com/2/137/Bullish_Flip.png)

### Expert Advisor Anatomy

When we start any MQL5 file, we add #property lines to set the scene. Here, we enable strict compilation, so the compiler catches any unsafe casts or deprecated calls. We also include our copyright, link, and version tags so anyone reading our code knows who wrote it, where to find more details, and which iteration they’re looking at. These lines don’t affect logic; they’re our way of stamping the file’s identity.

```
#property copyright "Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com/en/users/lynnchris"
#property version   "1.0"
#property strict
```

Next, we expose every setting that a trader might want to tweak. The first three inputs shape how we spot pivots with ATR.

- _InpAtrPeriod_ sets how many bars feed into the ATR calculation. A short span (for example 7) reacts fast but may catch noise. A longer span (say 21) smooths spikes at the cost of lag.
- _InpAtrMultiplier_ turns that ATR range into a minimum swing width. At 1.0 you need one ATR worth of movement to mark a pivot. Bumping it to 1.5 or 2.0 makes the filter more selective.
- _InpAtrLoosenFactor_ scales that width down between 0 and 1. A factor of 0.5 halves the requirement, so pivots show up earlier, which can help when volatility is low.

Next, we handle chart layout.

- _InpAutoShift_ reserves blank bars on the right whenever new bars appear.
- _InpShiftBars_ defines how many empty bars to leave (five by default).

This simple spacing keeps your arrows, labels, and stats panel from crowding the price action.

Finally, we give you two alert methods:

- _InpEnableSound_ lets the EA play a WAV file on each flip.
- _InpSoundFile_ is where you name that file from your MetaTrader 5 Sounds folder.
- _InpEnablePush_sends a push message to your MetaTrader 5 mobile app.

With these options, you choose whether to hear an alert on your desktop, get a ping on your phone, or both.

```
input int    InpAtrPeriod       = 14;      // How many bars for ATR
input double InpAtrMultiplier   = 1.0;     // Scale ATR into bar‑depth
input double InpAtrLoosenFactor = 0.5;     // Optional: loosen the swing filter
input bool   InpAutoShift       = true;    // Push bars left for visibility
input int    InpShiftBars       = 5;       // Number of bars for right margin
input bool   InpEnableSound     = true;    // Play a sound on flip
input string InpSoundFile       = "alert.wav";
input bool   InpEnablePush      = false;   // Send push notifications
```

When the EA starts, _OnInit_ first checks _InpAutoShift_ and, if enabled, calls _ChartSetInteger_ with CHART\_SHIFT and _InpShiftBars_ to push new bars left and reserve clear space for annotations; it then requests a handle to MetaTrader’s built-in ATR indicator via _iATR_, storing it in _atrHandle_ and immediately aborting with INIT\_FAILED if the handle is invalid; finally it creates a corner label (an OBJ\_LABEL named by _panelNam_ e), pins it to the upper-left corner, offsets it by 10 pixels horizontally and vertically, sets its font size to 10 and its color to yellow, and returns INIT\_SUCCEEDED to confirm that ATR data access and the stats panel are ready for _OnTick_.

```
int OnInit()
{
   if(InpAutoShift)
      ChartSetInteger(0, CHART_SHIFT, InpShiftBars);

   atrHandle = iATR(_Symbol, _Period, InpAtrPeriod);
   if(atrHandle == INVALID_HANDLE)
      return INIT_FAILED;

   ObjectCreate(0, panelName, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, panelName, OBJPROP_CORNER,    CORNER_LEFT_UPPER);
   ObjectSetInteger(0, panelName, OBJPROP_XDISTANCE, 10);
   ObjectSetInteger(0, panelName, OBJPROP_YDISTANCE, 10);
   ObjectSetInteger(0, panelName, OBJPROP_FONTSIZE,  10);
   ObjectSetInteger(0, panelName, OBJPROP_COLOR,     clrYellow);

   return INIT_SUCCEEDED;
}
```

Whenever we remove the EA or shut down MetaTrader 5, _OnDeinit_ erases all arrows and text objects we created, deletes the stats label, and releases the ATR handle. This prevents clutter and frees resources.

```
void OnDeinit(const int reason)
{
   ObjectsDeleteAll(0, -1, OBJ_ARROW);
   ObjectsDeleteAll(0, -1, OBJ_TEXT);
   ObjectDelete(0, panelName);
   if(atrHandle != INVALID_HANDLE)
      IndicatorRelease(atrHandle);
}
```

In our _OnTick_ routine, we guard against running our pivot logic on every single price update by introducing a static datetime _lastBa_ r. We fetch _thisBar_ with iTime(...,1), which returns the opening time of the most recently closed bar and compare it to lastBar. If it hasn’t changed, we know we’re still within the same candle and simply return. Once thisBar differs, we update lastBar and proceed.

We then call CopyBuffer on the ATR handle to retrieve only the latest ATR value—if that call fails, we bail out to avoid working with invalid data. With a valid ATR in hand, we compute our swing “depth” by converting the ATR price units into a bar count. We divide by the smallest price increment (SYMBOL\_POINT) and multiply by both InpAtrMultiplier and InpAtrLoosenFactor, then force a minimum of one bar with MathMax. This gives us a dynamic depth that grows when volatility surges (requiring bigger swings to mark pivots) and shrinks when the market calms (allowing tighter swings to count), before we hand off to the actual pivot‑detection functions.

```
void OnTick()
{
   static datetime lastBar=0;
   datetime thisBar = iTime(_Symbol,_Period,1);
   if(thisBar == lastBar) return;
   lastBar = thisBar;

   double atrBuf[];
   if(CopyBuffer(atrHandle, 0, 1, 1, atrBuf) <= 0) return;
   double atr = atrBuf[0];

   int depth = MathMax(1,
      int(atr / SymbolInfoDouble(_Symbol, SYMBOL_POINT)
          * InpAtrMultiplier * InpAtrLoosenFactor));
   // … pivot checks follow …
}
```

With depth defined, we run two simple loops to verify pivots. IsSwingHigh(1, depth) checks that no bar within depth bars on either side exceeds the candidate high; IsSwingLow does the opposite for lows. When we find a new high or low, we shift lastHigh into prevHigh (and same for lows) and record the timestamp. Tracking both the prior and current pivot is what lets us compare them next.

```
bool newHigh = IsSwingHigh(1, depth);
bool newLow  = IsSwingLow (1, depth);
double h = iHigh(_Symbol,_Period,1), l = iLow(_Symbol,_Period,1);

if(newHigh)
{
   prevHigh     = lastHigh;
   prevHighTime = lastHighTime;
   lastHigh     = h;
   lastHighTime = thisBar;
}
if(newLow)
{
   prevLow      = lastLow;
   prevLowTime  = lastLowTime;
   lastLow      = l;
   lastLowTime  = thisBar;
}
```

Once we’ve got our pivots, we update structState to reflect trend bias: a higher high sets state 1 (bearish flip possible next), a lower low sets state 2 (bullish flip possible next). Then we check for the actual flip: in state 1, if the new high is below the previous high, that’s a bearish flip; in state 2, if the new low rises above the prior low, that’s a bullish flip. Hitting a flip fires our plotting and notification calls and bumps our counters.

```
// Update bias
if(newHigh && prevHigh>0 && lastHigh > prevHigh) structState = 1;
if(newLow  && prevLow>0  && lastLow  < prevLow) structState = 2;

// Bearish flip
if(newHigh && structState==1 && lastHigh < prevHigh)
{
   PlotArrow(...);  PlotLabel(...);  Notify(...);
   if(countBear>0) sumBearInterval += (lastHighTime - prevLowTime)/60.0;
   countBear++;
}

// Bullish flip
if(newLow && structState==2 && lastLow > prevLow)
{
   PlotArrow(...);  PlotLabel(...);  Notify(...);
   if(countBull>0) sumBullInterval += (lastLowTime - prevHighTime)/60.0;
   countBull++;
}
```

We encapsulate all our on-chart drawings in two helper functions—PlotArrow and PlotLabel—to ensure both efficiency and clarity. Inside each function, we first call ObjectFind(0, name), which searches for an existing chart object by its unique name; this operation runs in O(n) time relative to the number of objects but is fast enough on modern machines for occasional per-bar checks. If the object doesn’t already exist (ObjectFind returns –1), we create it exactly once with ObjectCreate, choosing the appropriate object type (an arrow for PlotArrow, a text label for PlotLabel).

We then customize its properties: for arrows we set OBJPROP\_ARROWCODE to pick the desired glyph (e.g., Wingdings code 234 for a red down-arrow) and OBJPROP\_COLOR to define its hue; for labels we set OBJPROP\_TEXT with our caption (such as “LH” or “HL”) plus offsets and font size. By avoiding repeated calls to ObjectCreate, we prevent performance degradation and memory bloat that would occur if hundreds or thousands of identical objects piled up over time. This pattern also ensures that each pivot marker has a stable, predictable identifier, so if you later want to adjust its OBJPROP\_ZORDER (drawing priority) or delete it under certain conditions, you can refer to it by name with absolute confidence that you won’t accidentally affect other chart elements.

```
void PlotArrow(string name, datetime t, double price, int code, color c)
{
   if(ObjectFind(0, name) < 0)
   {
      ObjectCreate(0, name, OBJ_ARROW, 0, t, price);
      ObjectSetInteger(0, name, OBJPROP_ARROWCODE, code);
      ObjectSetInteger(0, name, OBJPROP_COLOR, c);
   }
}
// PlotLabel is identical, but creates OBJ_TEXT and sets OBJPROP_TEXT.
```

After each bar, we rebuild our panelName label to show:

- The current pivot depth
- Total bullish and bearish flips
- Average time (in minutes) between flips (once we have at least two).

This gives you instant feedback on how often structure breaks happen under your chosen ATR settings.

```
string txt = StringFormat("Depth: %d\nBull Flips: %d\nBear Flips: %d",
                          depth, countBull, countBear);
if(countBull>1)
   txt += "\nAvg HL Int: " + DoubleToString(sumBullInterval/(countBull-1),1) + "m";
if(countBear>1)
   txt += "\nAvg LH Int: " + DoubleToString(sumBearInterval/(countBear-1),1) + "m";

ObjectSetString(0, panelName, OBJPROP_TEXT, txt);
```

Finally, our Notify(msg) function wraps all alert methods in one place. We always call Alert(msg) for an MetaTrader 5 pop‑up, then optionally play a sound (PlaySound) or send a push (SendNotification) based on your inputs. Centralizing this makes it trivial to add e‑mail or webhook alerts later.

```
void Notify(string msg)
{
   Alert(msg);
   if(InpEnableSound) PlaySound(InpSoundFile);
   if(InpEnablePush)  SendNotification(msg);
}
```

### Source Code Listing

```
//+------------------------------------------------------------------+
//|                                 Market Structure Flip Detector EA|
//|                                   Copyright 2025, MetaQuotes Ltd.|
//|                           https://www.mql5.com/en/users/lynnchris|
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com/en/users/lynnchris"
#property version   "1.0"
#property strict

//--- user inputs
input int    InpAtrPeriod       = 14;      // ATR lookback
input double InpAtrMultiplier   = 1.0;     // ATR to swing depth factor (lower = looser)
input double InpAtrLoosenFactor = 0.5;     // Loosen factor for ATR confirmation (0-1)
input bool   InpAutoShift       = true;    // Auto-enable chart right shift
input int    InpShiftBars       = 5;       // Bars for right margin
input bool   InpEnableSound     = true;
input string InpSoundFile       = "alert.wav";
input bool   InpEnablePush      = false;

//--- global vars
string   panelName = "FlipPanel";
int      atrHandle;
int      structState = 0;
double   prevHigh=0, lastHigh=0;
datetime prevHighTime=0, lastHighTime=0;
double   prevLow=0, lastLow=0;
datetime prevLowTime=0, lastLowTime=0;
int      countBull=0, countBear=0;
double   sumBullInterval=0, sumBearInterval=0;

//+------------------------------------------------------------------+
int OnInit()
  {
   if(InpAutoShift)
      ChartSetInteger(0, CHART_SHIFT, InpShiftBars);

   atrHandle = iATR(_Symbol, _Period, InpAtrPeriod);
   if(atrHandle == INVALID_HANDLE)
      return(INIT_FAILED);

   ObjectCreate(0, panelName, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, panelName, OBJPROP_CORNER,    CORNER_LEFT_UPPER);
   ObjectSetInteger(0, panelName, OBJPROP_XDISTANCE, 10);
   ObjectSetInteger(0, panelName, OBJPROP_YDISTANCE, 10);
   ObjectSetInteger(0, panelName, OBJPROP_FONTSIZE,  10);
   ObjectSetInteger(0, panelName, OBJPROP_COLOR,     clrYellow);
   ObjectSetInteger(0, panelName, OBJPROP_BACK,      false);
   ObjectSetInteger(0, panelName, OBJPROP_ZORDER,    1);

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   ObjectsDeleteAll(0, -1, OBJ_ARROW);
   ObjectsDeleteAll(0, -1, OBJ_TEXT);
   ObjectDelete(0, panelName);
   if(atrHandle != INVALID_HANDLE)
      IndicatorRelease(atrHandle);
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   static datetime lastBar=0;
   datetime thisBar = iTime(_Symbol,_Period,1);
   if(thisBar==lastBar)
      return;
   lastBar=thisBar;

   double atrBuf[];
   if(CopyBuffer(atrHandle,0,1,1,atrBuf)<=0)
      return;
   double atr = atrBuf[0];

// loosen ATR confirmation by InpAtrLoosenFactor (0-1)
   double rawDepth = atr/SymbolInfoDouble(_Symbol,SYMBOL_POINT)*InpAtrMultiplier;
   int depth = MathMax(1, (int)(rawDepth * InpAtrLoosenFactor));

   bool newHigh=false,newLow=false;
   double h=iHigh(_Symbol,_Period,1), l=iLow(_Symbol,_Period,1);

   if(IsSwingHigh(1,depth))
     {
      prevHigh = lastHigh;
      prevHighTime = lastHighTime;
      lastHigh = h;
      lastHighTime = thisBar;
      newHigh = true;
     }
   if(IsSwingLow(1,depth))
     {
      prevLow = lastLow;
      prevLowTime = lastLowTime;
      lastLow = l;
      lastLowTime = thisBar;
      newLow = true;
     }

   double off = SymbolInfoDouble(_Symbol,SYMBOL_POINT)*10;

// Bearish Flip: Lower High after a Higher High
   if(newHigh && structState==1 && prevHigh>0 && lastHigh<prevHigh)
     {
      // signal arrow and label at current LH
      PlotArrow("Bear_"+IntegerToString((int)lastHighTime), lastHighTime, lastHigh, 234, clrRed);
      PlotLabel("LH_"+IntegerToString((int)lastHighTime), lastHighTime, lastHigh+off, "LH", clrRed);
      // label the previous LH used for comparison
      PlotLabel("PrevLH_"+IntegerToString((int)prevHighTime), prevHighTime, prevHigh+off, "LH_prev", clrRed);
      Notify("Bearish Flip (LH) at "+TimeToString(lastHighTime,TIME_DATE|TIME_MINUTES));
      if(countBear>0)
         sumBearInterval += (lastHighTime-prevLowTime)/60.0;
      countBear++;
     }

// Bullish Flip: Higher Low after a Lower Low
   if(newLow && structState==2 && prevLow>0 && lastLow>prevLow)
     {
      // signal arrow and label at current HL
      PlotArrow("Bull_"+IntegerToString((int)lastLowTime), lastLowTime, lastLow, 233, clrLime);
      PlotLabel("HL_"+IntegerToString((int)lastLowTime), lastLowTime, lastLow-off, "HL", clrLime);
      // label the previous HL used for comparison
      PlotLabel("PrevHL_"+IntegerToString((int)prevLowTime), prevLowTime, prevLow-off, "HL_prev", clrLime);
      Notify("Bullish Flip (HL) at "+TimeToString(lastLowTime,TIME_DATE|TIME_MINUTES));
      if(countBull>0)
         sumBullInterval += (lastLowTime-prevHighTime)/60.0;
      countBull++;
     }

// update structure state
   if(newHigh && prevHigh>0 && lastHigh>prevHigh)
      structState = 1;
   if(newLow  && prevLow>0  && lastLow <prevLow)
      structState = 2;

// update panel stats
   string txt = "Depth: "+IntegerToString(depth)+"\n";
   txt += "Bull Flips: "+IntegerToString(countBull)+"\n";
   txt += "Bear Flips: "+IntegerToString(countBear);
   if(countBull>1)
      txt += "\nAvg HL Int: "+DoubleToString(sumBullInterval/(countBull-1),1)+"m";
   if(countBear>1)
      txt += "\nAvg LH Int: "+DoubleToString(sumBearInterval/(countBear-1),1)+"m";
   ObjectSetString(0, panelName, OBJPROP_TEXT, txt);
  }

//+------------------------------------------------------------------+
bool IsSwingHigh(int shift,int depth)
  {
   double p = iHigh(_Symbol,_Period,shift);
   for(int i=shift-depth; i<=shift+depth; i++)
      if(i>=0 && iHigh(_Symbol,_Period,i) > p)
         return false;
   return true;
  }

//+------------------------------------------------------------------+
bool IsSwingLow(int shift,int depth)
  {
   double p = iLow(_Symbol,_Period,shift);
   for(int i=shift-depth; i<=shift+depth; i++)
      if(i>=0 && iLow(_Symbol,_Period,i) < p)
         return false;
   return true;
  }

//+------------------------------------------------------------------+
void PlotArrow(string nm,datetime t,double price,int code,color c)
  {
   if(ObjectFind(0,nm) < 0)
     {
      ObjectCreate(0, nm, OBJ_ARROW, 0, t, price);
      ObjectSetInteger(0, nm, OBJPROP_ARROWCODE, code);
      ObjectSetInteger(0, nm, OBJPROP_COLOR, c);
      ObjectSetInteger(0, nm, OBJPROP_WIDTH, 2);
     }
  }

//+------------------------------------------------------------------+
void PlotLabel(string nm,datetime t,double price,string txt,color c)
  {
   if(ObjectFind(0,nm) < 0)
     {
      ObjectCreate(0, nm, OBJ_TEXT, 0, t, price);
      ObjectSetString(0, nm, OBJPROP_TEXT, txt);
      ObjectSetInteger(0, nm, OBJPROP_COLOR, c);
      ObjectSetInteger(0, nm, OBJPROP_FONTSIZE, 10);
     }
  }

//+------------------------------------------------------------------+
void Notify(string msg)
  {
   Alert(msg);
   if(InpEnableSound)
      PlaySound(InpSoundFile);
   if(InpEnablePush)
      SendNotification(msg);
  }
//+------------------------------------------------------------------+
```

### Performance Outcomes

Below, I will outline the outcomes of our tests in both live-market conditions and backtesting.

Live Market

![Live Market Outcomes](https://c.mql5.com/2/137/snap_5.PNG)

On the chart above, the EA first spots a swing high labeled “LH\_prev,” which reflects two consecutive higher highs and establishes an up-structure. A few bars later, it detects another swing high that doesn’t top the previous peak—this lower high within an uptrend triggers the EA to draw a red arrow and “LH” label at that bar. That bearish-flip signal flags the breakdown of bullish momentum and warns that a downward move may be starting.

Gif Of Live Market

Below is a GIF demonstrating the EA’s performance on EURUSD. As each one-minute candle closes, the EA tracks successive lows until it finds a swing low that exceeds the previous trough. When that higher low appears, it drops a green “HL” arrow to mark the bullish flip. At the same moment, the header panel refreshes—here showing 12 bull flips, 1 bear flip, and an average HL interval of 108.0m—to reflect the updated counts. This clip clearly illustrates the transition from a down-structure into a potential up-move.

![Live Market Outcomes](https://c.mql5.com/2/137/SnippingTool_popxMnDgvo.gif)

Backtesting

Below is a table of the step-index analysis results across multiple timeframes. ‘Positive signals’ are those after which the market moved in the indicated direction for a sustained period.

5-Minute Timeframe

| Signal Type | Total Signals | Positive Signals | Win Rate |
| --- | --- | --- | --- |
| Sell | 56 | 39 | 70% |
| Buy | 53 | 44 | 83% |

15-Minute Timeframe

| Signal Type | Total Signals | Positive Signals | Win Rate |
| --- | --- | --- | --- |
| Sell | 7 | 5 | 71% |
| Buy | 14 | 9 | 64% |

The summary of the analytics indicates that the Market Structure Flip Detector consistently generates profitable signals, particularly in shorter timeframes. Sell setups achieve a success rate of 70% and above, underscoring the tool's effectiveness. This accomplishment represents a significant advancement in the automation of price-action analysis and brings us closer to a fully systematic, low-latency trading toolkit.

### Conclusion

Having developed and tested this tool in both live market conditions and through backtesting, our analysis reveals that it consistently delivers strong performance, particularly in scalping on lower timeframes, where it generates substantial returns. However, it is essential to exercise caution and validate its signals with additional confirmation methods before executing any trades. Furthermore, testing the tool across various currency pairs is crucial for identifying where it performs most positively. You may also adjust the input parameters during your testing to optimize performance further.

| Date | Tool Name | Description | Version | Updates | Notes |
| --- | --- | --- | --- | --- | --- |
| 01/10/24 | [Chart Projector](https://www.mql5.com/en/articles/16014) | Script to overlay the previous day's price action with a ghost effect. | 1.0 | Initial Release | Tool number 1 |
| 18/11/24 | [Analytical Comment](https://www.mql5.com/en/articles/15927) | It provides previous day's information in a tabular format, as well as anticipates the future direction of the market. | 1.0 | Initial Release | Tool number 2 |
| 27/11/24 | [Analytics Master](https://www.mql5.com/en/articles/16434) | Regular Update of market metrics after every two hours | 1.01 | Second Release | Tool number 3 |
| 02/12/24 | [Analytics Forecaster](https://www.mql5.com/en/articles/16559) | Regular Update of market metrics after every two hours with telegram integration | 1.1 | Third Edition | Tool number 4 |
| 09/12/24 | [Volatility Navigator](https://www.mql5.com/en/articles/16560) | The EA analyzes market conditions using the Bollinger Bands, RSI and ATR indicators | 1.0 | Initial Release | Tool Number 5 |
| 19/12/24 | [Mean Reversion Signal Reaper](https://www.mql5.com/en/articles/16700) | Analyzes market using mean reversion strategy and provides signal | 1.0 | Initial Release | Tool number 6 |
| 9/01/25 | [Signal Pulse](https://www.mql5.com/en/articles/16861) | Multiple timeframe analyzer | 1.0 | Initial Release | Tool number 7 |
| 17/01/25 | [Metrics Board](https://www.mql5.com/en/articles/16584) | Panel with button for analysis | 1.0 | Initial Release | Tool number 8 |
| 21/01/25 | [External Flow](https://www.mql5.com/en/articles/16967) | Analytics through external libraries | 1.0 | Initial Release | Tool number 9 |
| 27/01/25 | [VWAP](https://www.mql5.com/en/articles/16984) | Volume Weighted Average Price | 1.3 | Initial Release | Tool number 10 |
| 02/02/25 | [Heikin Ashi](https://www.mql5.com/en/articles/17021) | Trend Smoothening and reversal signal identification | 1.0 | Initial Release | Tool number 11 |
| 04/02/25 | [FibVWAP](https://www.mql5.com/en/articles/17121) | Signal generation through python analysis | 1.0 | Initial Release | Tool number  12 |
| 14/02/25 | [RSI DIVERGENCE](https://www.mql5.com/en/articles/17198) | Price action versus RSI divergences | 1.0 | Initial Release | Tool number 13 |
| 17/02/25 | [Parabolic Stop and Reverse (PSAR)](https://www.mql5.com/en/articles/17234) | Automating PSAR strategy | 1.0 | Initial Release | Tool number 14 |
| 20/02/25 | [Quarters Drawer Script](https://www.mql5.com/en/articles/17250) | Drawing quarters levels on chart | 1.0 | Initial Release | Tool number 15 |
| 27/02/25 | [Intrusion Detector](https://www.mql5.com/en/articles/17321) | Detect and alert when price reaches quarters levels | 1.0 | Initial Release | Tool number 16 |
| 27/02/25 | [TrendLoom Tool](https://www.mql5.com/en/articles/17329) | Multi timeframe analytics panel | 1.0 | Initial Release | Tool number 17 |
| 11/03/25 | [Quarters Board](https://www.mql5.com/en/articles/17442) | Panel with buttons to activate or disable quarters levels | 1.0 | Initial Release | Tool number 18 |
| 26/03/25 | [ZigZag Analyzer](https://www.mql5.com/en/articles/17625) | Drawing trendlines using ZigZag Indicator | 1.0 | Initial Release | Tool number 19 |
| 10/04/25 | [Correlation Pathfinder](https://www.mql5.com/en/articles/17742) | Plotting currency correlations using Python libraries. | 1.0 | Initial Release | Tool number 20 |
| 23/04/25 | Market Structure Flip Detector Tool | Market structure flip detection | 1.0 | Initial Release | Tool number 21 |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17891.zip "Download all attachments in the single ZIP archive")

[Flip\_Detector.mq5](https://www.mql5.com/en/articles/download/17891/flip_detector.mq5 "Download Flip_Detector.mq5")(14.5 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/485687)**
(1)


![Alexander P.](https://c.mql5.com/avatar/avatar_na2.png)

**[Alexander P.](https://www.mql5.com/en/users/alepie)**
\|
7 May 2025 at 11:08

```
bool IsSwingHigh( int shift, int depth)
  {
   double p = iHigh ( _Symbol , _Period ,shift);
   for ( int i= shift-depth; i<=shift+depth; i++)
       if (i>= 0 && iHigh ( _Symbol , _Period ,i) > p)
         return false ;
   return true ;
  }
```

Hello, I don't understand why you write **int i=shift-depth** , couldn't you just use **int i=0** **?**

Can you please explain this? Thanks.

![Data Science and ML (Part 37): Using Candlestick patterns and AI to beat the market](https://c.mql5.com/2/138/article_image_17832_2-logo.png)[Data Science and ML (Part 37): Using Candlestick patterns and AI to beat the market](https://www.mql5.com/en/articles/17832)

Candlestick patterns help traders understand market psychology and identify trends in financial markets, they enable more informed trading decisions that can lead to better outcomes. In this article, we will explore how to use candlestick patterns with AI models to achieve optimal trading performance.

![MQL5 Wizard Techniques you should know (Part 61): Using Patterns of ADX and CCI with Supervised Learning](https://c.mql5.com/2/138/article-17910-logo.png)[MQL5 Wizard Techniques you should know (Part 61): Using Patterns of ADX and CCI with Supervised Learning](https://www.mql5.com/en/articles/17910)

The ADX Oscillator and CCI oscillator are trend following and momentum indicators that can be paired when developing an Expert Advisor. We look at how this can be systemized by using all the 3 main training modes of Machine Learning. Wizard Assembled Expert Advisors allow us to evaluate the patterns presented by these two indicators, and we start by looking at how Supervised-Learning can be applied with these Patterns.

![DoEasy. Service functions (Part 3): Outside Bar pattern](https://c.mql5.com/2/75/DoEasy._Service_functions_Part_1___LOGO.png)[DoEasy. Service functions (Part 3): Outside Bar pattern](https://www.mql5.com/en/articles/14710)

In this article, we will develop the Outside Bar Price Action pattern in the DoEasy library and optimize the methods of access to price pattern management. In addition, we will fix errors and shortcomings identified during library tests.

![From Basic to Intermediate: FOR Statement](https://c.mql5.com/2/94/Do_b4sico_ao_intermediqrio_Comando_FOR___LOGO.png)[From Basic to Intermediate: FOR Statement](https://www.mql5.com/en/articles/15406)

In this article, we will look at the most basic concepts of the FOR statement. It is very important to understand everything that will be shown here. Unlike the other statements we've talked about so far, the FOR statement has some quirks that quickly make it very complex. So don't let stuff like this accumulate. Start studying and practicing as soon as possible.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/17891&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062583565390095607)

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