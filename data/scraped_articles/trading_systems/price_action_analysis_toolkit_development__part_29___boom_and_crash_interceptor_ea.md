---
title: Price Action Analysis Toolkit Development (Part 29): Boom and Crash Interceptor EA
url: https://www.mql5.com/en/articles/18616
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:36:00.236643
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/18616&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069583335585417054)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/18616#para2)
- [Understanding The Strategy](https://www.mql5.com/en/articles/18616#para3)
- [Code Breakdown](https://www.mql5.com/en/articles/18616#para4)
- [Backtesting and Results](https://www.mql5.com/en/articles/18616#para5)
- [Conclusion](https://www.mql5.com/en/articles/18616#para6)

### Introduction

In [modern air-defence networks](https://www.mql5.com/go?link=https://www.armscontrol.org/factsheets/missile-defense-systems-glance "https://www.armscontrol.org/factsheets/missile-defense-systems-glance"), a layered array of sensors evaluates every radar echo and infrared trace with forensic precision.  Each contact is measured for speed, trajectory, and signature, then cross-checked against an extensive threat library.  Only when several independent filters concur does the system authorise the launch of an interceptor, preserving resources while guaranteeing that genuine threats are neutralised without delay.

The Boom & Crash Interceptor EA adopts the same disciplined approach to market data.  A rolling velocity window determines whether the current price impulse eclipses recent behaviour; an ATR-based surge multiplier confirms that volatility has expanded meaningfully; and moving-average trend filters validate directional bias.  Optional pivot-zone and session-hour gates further suppress signals that would otherwise emerge during thin-liquidity periods.

When every layer confirms, the EA plots a definitive “BOOM” or “CRASH” arrow on the chart, complete with user-defined colors, offsets, and CSV logging. This ensures that attention is reserved for high-probability opportunities.  The following pages explain how each detection layer can be calibrated and how to deploy this MQL5 tool as a robust, signal-driven component of a broader trading framework.

### Understanding The Strategy

Drawing inspiration from missile-interceptor defense, the Boom & Crash Interceptor EA approaches market spikes as if they were incoming threats. A missile, in military terms, is any projectile-guided or unguided, launched with the intent to strike a target. Modern variants are self-propelled rockets equipped with warheads and guided by inertial, radar, satellite, or optical systems; examples range from intercontinental ballistic missiles (ICBMs) to cruise missiles and surface-to-air defensive rounds. Historically, even a hurled spear or arrow qualifies as a missile because its trajectory is deliberately directed at a target.

In the trading arena, a sudden, sharp surge in price can be just as dangerous: it appears quickly, often unexpectedly, and can inflict heavy damage on an unprepared account. Here, the missile-interceptor analogy proves valuable. A missile interceptor is a defensive system that detects, tracks, and neutralizes an incoming warhead before impact. Its operation unfolds in four tightly integrated phases:

Detection and Tracking

- Ground, ship-borne, or space-based sensors register the launch of a hostile projectile.
- Fire-control software computes trajectory, velocity, and projected impact point.

Launch and Guidance

- Once verified, an interceptor lifts off from a silo, vehicle, vessel, or aircraft.
- Course corrections are provided via inertial guidance, mid-course datalinks, and terminal-phase radar or infrared homing.

Engagement (Kill) Methods

- Hit-to-kill: the interceptor collides directly with the warhead at closing speeds exceeding Mach 10, relying on kinetic energy for destruction.
- Proximity blast: a controlled detonation near the target shatters or deflects it with shrapnel or a focused shock wave.

Layered Defense

- Short-range systems (e.g., Patriot PAC-3) engage threats in the terminal phase.
- Mid-course systems (e.g., SM-3, Ground-based Midcourse Defense) attack in space.
- Boost-phase concepts, still experimental, aim at missiles moments after launch.

These layers exist to maximize protection while minimizing false launches, a balance achieved through rapid data fusion and strict confirmation logic.

The Boom & Crash Interceptor EA mirrors this architecture. Continuous market surveillance functions as the radar net; velocity thresholds, ATR surge filters, and moving-average alignment act as independent confirmation layers; optional pivot-zone and session filters imitate decoy discrimination; and, only when all criteria concur, the EA “fires” by plotting a BOOM or CRASH arrow and logging the event. In doing so, it reserves trader attention, and trading capital-for threats that genuinely warrant a response, just as a missile-defense battery conserves its interceptors for warheads that pose real danger.

![](https://c.mql5.com/2/152/BOOM_and_CRASH_INTERCEPTOR.png)

Fig 1. Strategy

Velocity Radar

Measures the price change over VelocityHistoryBars and compares it to a historical percentile threshold-only “unusually large” moves pass.

```
// Record current vs. old price
double priceNow = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
double priceOld = velHistory[VelocityHistoryBars - 1];
double delta    = priceNow - priceOld;

// Build and sort past deltas
double d[];
ArrayResize(d, VelocityHistoryBars - 1);
for(int i = 1; i < VelocityHistoryBars; i++)
    d[i - 1] = velHistory[0] - velHistory[i];
ArraySort(d);

// Pick the (100–VelocityPctile)% threshold
int idx      = (int)MathRound((VelocityPctile / 100.0) * (ArraySize(d) - 1));
double velTh = d[ArraySize(d) - 1 - idx];

// Pass if current move exceeds threshold
bool okVel = (delta > velTh || delta < -velTh);
```

ATR Surge Detector

Checks for a sudden jump in volatility by comparing the latest ATR to the previous bar’s ATR multiplied by ATRMultiplier.

```
// Fetch two most recent ATR values
double atrArr[2];
CopyBuffer(atrHandle, 0, 0, 2, atrArr);

// Pass if ATR_now > ATR_prev × ATRMultiplier
bool okATR = (atrArr[0] > atrArr[1] * ATRMultiplier);
```

Trend Alignment Check

Ensures the price move is in line with the trend by verifying that the SMA is also rising on up-moves (or falling on down-moves).

```
// Fetch two most recent SMA values
double maArr[2];
CopyBuffer(maHandle, 0, 0, 2, maArr);

// If delta>0 require SMA_now>SMA_prev; if delta<0 require SMA_now<SMA_prev
bool okTrend = (delta > 0
                ? maArr[0] > maArr[1]
                : maArr[0] < maArr[1]);
```

Pivot-Zone Filter (optional)

Blocks signals too close to the prior bar’s pivot, enforcing a buffer of ZoneBufferPoints.

```
// Compute prior bar’s pivot
double h1    = iHigh(_Symbol, MainTF, 1),
       l1    = iLow (_Symbol, MainTF, 1),
       c1    = iClose(_Symbol, MainTF, 1);
double pivot = (h1 + l1 + c1) / 3.0;

// Pass if priceNow is beyond pivot ± buffer
bool okZone = (delta > 0
               ? priceNow < pivot - ZoneBufferPoints * _Point
               : priceNow > pivot + ZoneBufferPoints * _Point);
```

Final “Missile Launch” Logic

Fires a Boom (up arrow) or Crash (down arrow) only when all filters pass.

```
// Determine direction
bool isBoom  = (delta  >  velTh);
bool isCrash = (delta  < -velTh);

// Fire only if velocity, ATR, trend, and zone all passed
bool fire    = ((isBoom || isCrash) && okVel && okATR && okTrend && okZone);

if(fire)
    GenerateSignal(isBoom, priceNow, delta, atrArr[0], MainTF);
```

### Code Breakdown

Below is a systematic walk-through of every major section of the “Boom & Crash Interceptor” Expert Advisor.

Header and compilation directives

The opening comment block and the subsequent #property lines act as the EA’s business card.  They assign copyright ownership, a support link, the current version, and the strict compilation flag.  Compiling in strict mode forces MetaTrader 5 to apply the most rigorous type and range checks, helping us catch latent errors early in the build process.  These details also allow the MetaQuotes Market to track product lineage and enable smooth, versioned updates for end-users.

```
//+------------------------------------------------------------------+
//|                                     Boom and Crash Interceptor EA|
//|                                   Copyright 2025, MetaQuotes Ltd.|
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com/en/users/lynnchris"
#property version   "1.0"
#property strict
```

Library include and trade object

By including <Trade\\Trade.mqh> and instantiating a global CTrade object, the code gains access to an object-oriented wrapper for MetaTrader’s trading API.  Even though the current version only issues alerts and graphics, the presence of this object positions the code base for seamless migration to live order execution when required.

```
#include <Trade\Trade.mqh>
CTrade trade;            // object-oriented trading wrapper
```

External inputs-the EA’s control panel

All run-time parameters are declared with the input keyword, enabling optimisation in the Strategy Tester or quick edits in the chart’s “Inputs” tab. They fall into four logical groups:

• Signal-generation settings: timeframes, velocity look-back, ATR parameters, moving-average parameters.

• Optional filters: pivot-zone logic and session hours.

• Presentation settings: dashboard placement, colours, line widths, arrow offsets.

• Data logging: CSV filename.

```
input ENUM_TIMEFRAMES MainTF            = PERIOD_CURRENT; // Signal TF
input int            VelocityHistoryBars = 96;
input double         VelocityPctile      = 120.0;          // >100 ⇒ extreme
input int            ATRPeriod           = 14;
input double         ATRMultiplier       = 1.5;
input bool           UseZoneFilter       = true;
input int            SessionStartHour    = 7;
input int            SessionEndHour      = 17;
input string         LogFilename         = "BoCrashLog.csv";
```

An interesting nuance is the default VelocityPctile value of 120%.  Because the percentile algorithm caps at the extreme end of the distribution, setting a value beyond 100% effectively demands that price surpass the most violent move seen within the look-back window, making the trigger deliberately selective.

Global variables and naming conventions

The array velHistory\[\] stores the rolling window required for velocity analysis, while multiple string constants centralize GUI object names.  Consolidating names in arrays (dashNames, hVelUp, etc.) guarantees deterministic creation and deletion, preventing orphaned objects and simplifying future refactoring.

```
double  velHistory[];
string  dashBG      = "DashBG";
string  dashNames[] = {"Delta","VelThr","ATR","ATRm","Trend",
                       "Pivot","Zone","Signal"};
string  hVelUp      = "VelUp";
string  hVelDown    = "VelDown";
```

OnInit(): resource acquisition and GUI construction

During initialization, we resize the velocity array, acquire handles for ATR and moving-average indicators, and terminate with INIT\_FAILED if any handle is unavailable-failing early is preferable to runtime instability.  The EA then opens or creates the CSV log file, writing a header if the file is new and seeking to the end to append subsequent records.  Finally, the routine erects a semi-transparent dashboard rectangle, populates it with textual labels, and pre-creates horizontal and vertical reference lines.  All graphical elements are live on the chart before the first market tick arrives.

```
int OnInit()
  {
   ArrayResize(velHistory,VelocityHistoryBars);

   atrHandle = iATR(_Symbol, ATRTF, ATRPeriod);
   maHandle  = iMA (_Symbol, TrendTF, TrendMAPeriod, 0, MODE_SMA, PRICE_CLOSE);
   if(atrHandle==INVALID_HANDLE || maHandle==INVALID_HANDLE)
      return INIT_FAILED;

   logHandle = FileOpen(LogFilename, FILE_READ|FILE_WRITE|FILE_CSV|FILE_ANSI);
   if(logHandle>=0 && FileSize(logHandle)==0)
         FileWrite(logHandle,"DateTime,Type,Velocity,ATR,Price");

   // dashboard background
   ObjectCreate(0,dashBG,OBJ_RECTANGLE_LABEL,0,0,0);
   ObjectSetInteger(0,dashBG,OBJPROP_XSIZE,200);
   ObjectSetInteger(0,dashBG,OBJPROP_YSIZE,140);

   // dashboard labels
   for(int i=0;i<ArraySize(dashNames);i++)
     {
      string name="Dash_"+dashNames[i];
      ObjectCreate(0,name,OBJ_LABEL,0,0,0);
      ObjectSetString(0,name,OBJPROP_TEXT,dashNames[i]+": ?");
     }

   // reference lines
   ObjectCreate(0,hVelUp ,OBJ_HLINE,0,0,0);
   ObjectCreate(0,hVelDown,OBJ_HLINE,0,0,0);
   return INIT_SUCCEEDED;
  }
```

OnDeinit(): orderly shutdown

At removal, the EA releases indicator handles, closes the file handle, and deletes every graphical object referenced in the naming arrays.  Such symmetry between creation and destruction is vital for keeping the terminal’s object list clean and conserving operating-system resources.

```
void OnDeinit(const int reason)
  {
   if(atrHandle!=INVALID_HANDLE) IndicatorRelease(atrHandle);
   if(maHandle !=INVALID_HANDLE) IndicatorRelease(maHandle);
   if(logHandle>=0)              FileClose(logHandle);

   string objs[]={dashBG,hVelUp,hVelDown};
   for(int i=0;i<ArraySize(objs);i++) ObjectDelete(0,objs[i]);
   for(int i=0;i<ArraySize(dashNames);i++)
      ObjectDelete(0,"Dash_"+dashNames[i]);
  }
```

OnTick(): real-time decision engine

The routine first confirms that a new bar has closed on MainTF.  If not, the function returns immediately, sparing unnecessary computation.  Should a bar transition be detected, the code observes the optional session filter, ensuring operations occur only within the trader-defined window.

```
void OnTick()
  {
   // process only on completed bar
   int cur=iBars(_Symbol,MainTF)-1;
   if(cur==lastBar) return;
   lastBar=cur;

   // optional session filter
   MqlDateTime now; TimeToStruct(TimeCurrent(),now);
   if(UseSessionFilter && (now.hour<SessionStartHour || now.hour>SessionEndHour))
      return;

   // velocity calculation
   double priceNow = SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   ArrayMove(velHistory,1,0,VelocityHistoryBars-1); // shift right
   velHistory[0]=priceNow;
   double delta  = priceNow-velHistory[VelocityHistoryBars-1];
   double velTh  = ComputeVelocityPercentile(velHistory,VelocityPctile);
   bool   okVel  = (delta>velTh || delta<-velTh);

   // ATR, MA, pivot-zone checks ...
   /* …remaining signal logic… */

   if(fire) GenerateSignal(isBoom,priceNow,delta,atrArr[0],MainTF);
  }
```

The latest ask price is then inserted at the front of velHistory, pushing older entries one index down.  We calculate delta, the distance between the newest and the oldest price in the window, and retrieve a percentile threshold using ComputeVelocityPercentile().  A velocity condition passes when delta exceeds the upper or lower threshold.

ATR and moving-average filters are assessed on potentially different, lower time-frames, capturing short-term volatility expansion and confirming directional bias.  The pivot-zone filter, when enabled, computes a classical H+L+C pivot on the previous candle and restricts signals to moves that still reside on the “wrong” side of that pivot plus a configurable buffer, therefore seeking mean-reversion potential.

The dashboard is updated with raw diagnostic values; colors immediately flag whether each criterion is met.  Meanwhile, the horizontal velocity and pivot lines plus the vertical timing line are repositioned to keep the chart’s visual context in sync with the newest bar.  A signal is considered valid only if all four gates-velocity spike, ATR surge, trend agreement, and zone clearance-return true.

Helper function: UpdateLabel()

This small routine centralizes dashboard text and color updates.  By encapsulating formatting in one location, we ensure consistent styling and make global aesthetic adjustments straightforward.

```
void UpdateLabel(int idx,string txt,bool pass)
  {
   string name="Dash_"+dashNames[idx];
   ObjectSetString (0,name,OBJPROP_TEXT ,txt);
   ObjectSetInteger(0,name,OBJPROP_COLOR, pass ? clrLime : clrRed);
  }
```

Helper function: ComputeVelocityPercentile()

The algorithm converts the stored price series into a list of signed deltas, sorts the list, and indexes it according to the user-requested percentile.  Selecting from the extreme end of the ordered array allows the method to respect the sign of the movement (boom or crash) without additional logic. Percentile values above 100% collapse to the maximum element, enforcing an “all-time extreme within the window” threshold.

```
double ComputeVelocityPercentile(double &hist[],double pct)
  {
   int n=ArraySize(hist);
   if(n<2) return 0;
   double d[]; ArrayResize(d,n-1);
   for(int i=1;i<n;i++) d[i-1]=hist[0]-hist[i];
   ArraySort(d);                        // ascending
   int idx=(int)MathRound((pct/100.0)*(ArraySize(d)-1));
   return d[ArraySize(d)-1-idx];        // mirror to get extreme with sign
  }
```

Helper function: SetHLine()

By manipulating existing object properties instead of recreating lines, this method avoids flicker and graphical lag, maintaining a smooth user experience even on lower-spec hardware.

```
void SetHLine(string name,double price,color clr)
  {
   ObjectSetDouble (0,name,OBJPROP_PRICE, price);
   ObjectSetInteger(0,name,OBJPROP_COLOR, clr);
   ObjectSetInteger(0,name,OBJPROP_WIDTH, 1);
  }
```

Helper function: GenerateSignal()

When all filters concur, this routine draws a color-coded Wingdings arrow at an offset from the bar’s low (boom) or high (crash), writes a comprehensive entry to the CSV log (timestamp, type, delta, ATR, price), and triggers an audible alert. Retaining CTrade in scope means converting these visual markers into live orders merely involves invoking trade.Buy() or trade.Sell() here.

```
void GenerateSignal(bool isBoom,double price,double delta,
                    double atr,ENUM_TIMEFRAMES tf)
  {
   int    code = isBoom ? 233 : 234;
   double y    = isBoom ? iLow(_Symbol,tf,0)-ArrowOffsetPips*_Point
                        : iHigh(_Symbol,tf,0)+ArrowOffsetPips*_Point;
   color  clr  = isBoom ? BoomArrowColor : CrashArrowColor;
   string tag  = (isBoom?"BOOM":"CRASH")+"_"+
                 TimeToString(TimeCurrent(),TIME_SECONDS);

   datetime t0 = iTime(_Symbol,tf,0);
   ObjectCreate(0,tag,OBJ_ARROW,0,t0,y);
   ObjectSetInteger(0,tag,OBJPROP_ARROWCODE,code);
   ObjectSetInteger(0,tag,OBJPROP_COLOR,clr);

   if(logHandle>=0)
      FileWrite(logHandle,TimeToString(TimeCurrent(),TIME_DATE|TIME_SECONDS),
                isBoom?"BOOM":"CRASH",DoubleToString(delta,2),
                DoubleToString(atr,_Digits),DoubleToString(price,_Digits));

   Alert((isBoom?"BOOM":"CRASH")+" signal @"+
         DoubleToString(price,_Digits));
  }
```

Strategic intent and extension opportunities.

```
// Inside GenerateSignal(), after drawing the arrow:
double lot   = 0.02;                        // example risk sizing
double sl    = isBoom ? price - atr*2 : price + atr*2;
double tp    = pivot;                       // mean-reversion target

if(AutoTrade)                               // user-controlled switch
  {
   if(isBoom) trade.Buy(lot,_Symbol,price,sl,tp,"Boom-auto");
   else       trade.Sell(lot,_Symbol,price,sl,tp,"Crash-auto");
  }
```

In conceptual terms, the EA searches for explosive, single-bar accelerations occurring amid a fresh volatility expansion yet still positioned for mean reversion toward the prior pivot.  Such behaviour is characteristic of Boom/Crash indices, where abrupt spikes frequently retrace.  Presently the code generates informative alerts; integrating automated execution would require adding position sizing, stop-loss (for example, the opposite velocity line), and take-profit logic (perhaps the pivot or half the buffer). Finally, the linear array shift inside velHistory is acceptable for the default 96-bar window but could be replaced by a circular buffer should the user elect much larger histories.

### Backtesting and Results

Before sending our interceptor live, let’s kick off a comprehensive backtest on a demo account to fine-tune its settings:

1\. Compile and Load

After your EA compiles cleanly in MetaEditor, switch over to MetaTrader.

2\. Open the Strategy Tester

In MetaTrader’s toolbar, click the “Strategy Tester” icon. Pick your Boom & Crash Interceptor EA from the Expert list.

3\. Configure Your Test

- Symbol & Timeframe: Match the exact instrument and MainTF you’ll trade live.
- Period: Choose a span that covers different market conditions (quiet, trending, volatile).
- Inputs: Adjust VelocityHistoryBars, VelocityPctile, ATRMultiplier, TrendMAPeriod, ZoneBufferPoints, etc., to see how they affect signal frequency and accuracy.
- Execution: Use “Every Tick” mode for the most precise results.

4\. Run and Review

Click Start. Once the backtest finishes, examine the equity curve, list of signals, and any failed or late intercepts. Note where signals clustered or missed real spikes.

5\. Refine & Repeat

Tweak one or two inputs at a time, maybe tighten your velocity percentile or widen the pivot buffer, then retest. Iterate until your demo results show clean, repeatable Boom/Crash signals with acceptable drawdown.

Once you’re confident in the EA’s performance across multiple demo runs, you’ll be ready to deploy it in a live account under controlled risk.

Below, I’m sharing the backtest results for the Boom & Crash Interceptor EA, and I found the outcome especially insightful.

![](https://c.mql5.com/2/152/missille_interceptor_ea.gif)

Fig 2. Boom 900 Backtesting

In the Boom 900 backtest, our EA recorded every Boom and Crash signal in the journal, just as expected. The pivot line appears in blue and the velocity threshold in orange, giving a clear visual guide. You’ll also spot green upward arrows marking each BOOM and red downward arrows marking each CRASH. Below, I’ve included a still image for a clear view of what you saw in the GIF is captured here.

![](https://c.mql5.com/2/152/boom_900.png)

Fig 3. Backtesting Results

### Conclusion

We’ve built and rigorously backtested the Boom & Crash Interceptor EA-and the results speak for themselves. It nails extreme price moves by tracking velocity, ATR surges, trend alignment, and pivot-zone breaks, then plots only the cleanest Boom (green) or Crash (red) arrows. Our Boom 900 demo run logged every signal, rendered pivot and velocity lines in blue and orange, and matched the GIF perfectly with the still image.

That said, a few signals might still benefit from extra filtering. Think of this EA as your tactical radar, not a standalone order-executor, always layer in your own time-of-day, pattern, or fundamental checks before you pull the trigger. And if you want even fewer false positives, shift up to larger timeframes (H1, H4 or Daily) to smooth out the noise.

With disciplined backtesting, smart filtering and a controlled live rollout, the Boom & Crash Interceptor EA becomes a genuine market-spike interceptor, putting you ahead of the pack, not scrambling behind each move.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| [Chart Projector](https://www.mql5.com/en/articles/16014) | [Analytical Comment](https://www.mql5.com/en/articles/15927) | [Analytics Master](https://www.mql5.com/en/articles/16434) | [Analytics Forecaster](https://www.mql5.com/en/articles/16559) | [Volatility Navigator](https://www.mql5.com/en/articles/16560) | [Mean Reversion Signal Reaper](https://www.mql5.com/en/articles/16700) |
| [Signal Pulse](https://www.mql5.com/en/articles/16861) | [Metrics Board](https://www.mql5.com/en/articles/16584) | [External Flow](https://www.mql5.com/en/articles/16967) | [VWAP](https://www.mql5.com/en/articles/16984) | [Heikin Ashi](https://www.mql5.com/en/articles/17021) | [FibVWAP](https://www.mql5.com/en/articles/17121) |
| [RSI DIVERGENCE](https://www.mql5.com/en/articles/17198) | [Parabolic Stop and Reverse (PSAR)](https://www.mql5.com/en/articles/17234) | [Quarters Drawer Script](https://www.mql5.com/en/articles/17250) | [Intrusion Detector](https://www.mql5.com/en/articles/17321) | [TrendLoom Tool](https://www.mql5.com/en/articles/17329) | [Quarters Board](https://www.mql5.com/en/articles/17442) |
| [ZigZag Analyzer](https://www.mql5.com/en/articles/17625) | [Correlation Pathfinder](https://www.mql5.com/en/articles/17742) | [Market Structure Flip Detector Tool](https://www.mql5.com/en/articles/17891) | [Correlation Dashboard](https://www.mql5.com/en/articles/18052) | [Currency Strength Meter](https://www.mql5.com/en/articles/18108) | [PAQ Analysis Tool](https://www.mql5.com/en/articles/18207) |
| [Dual EMA Fractal Breaker](https://www.mql5.com/en/articles/18297) | [Pin bar, Engulfing and RSI divergence](https://www.mql5.com/en/articles/17962) | [Liquidity Sweep](https://www.mql5.com/en/articles/18379) | [Opening Range Breakout Tool](https://www.mql5.com/en/articles/18486) | Boom and Crash Interceptor |  |

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
**[Go to discussion](https://www.mql5.com/en/forum/490057)**
(9)


![Harry Davis](https://c.mql5.com/avatar/avatar_na2.png)

**[Harry Davis](https://www.mql5.com/en/users/foxmor)**
\|
2 Jul 2025 at 03:57

Hi,

Can I download this please, cant find any links?

Regards

Harry

![michal_](https://c.mql5.com/avatar/avatar_na2.png)

**[michal\_](https://www.mql5.com/en/users/michal_)**
\|
4 Aug 2025 at 09:09

There is algo problem with velocity detector.

```
ArraySort(d);
```

Working in the buy direction good,

but for negative move is it the sorting bad.

You need do it separatelly or change the index formula ( in sell negative values sorting!!! )

![michal_](https://c.mql5.com/avatar/avatar_na2.png)

**[michal\_](https://www.mql5.com/en/users/michal_)**
\|
4 Aug 2025 at 09:29

```

if(delta>0)
{
for(int i = 1; i < VelocityHistoryBars; i++)
    d[i - 1] = velHistory[0] - velHistory[i];

ArraySort(d);
}

if(delta<0)
{
for(int i = 1; i < VelocityHistoryBars; i++)
    d[i - 1] = velHistory[i] - velHistory[0];

ArraySort(d);
}

int idx= (int)MathRound((VelocityPctile / 100.0) * (ArraySize(d) - 1));
double velTh = d[ArraySize(d) - 1 - idx];

bool okVel = MathAbs(delta) > velTh;
```

![michal_](https://c.mql5.com/avatar/avatar_na2.png)

**[michal\_](https://www.mql5.com/en/users/michal_)**
\|
6 Aug 2025 at 08:56

I have some variants to pivot alignment.

I do not know if this is better.

( from price which is pivot up we await move up .... )

```
bool okZone = false;
if((delta > 0 ) && (priceNow > pivot)) okZone = true;
if((delta < 0 ) && (priceNow < pivot)) okZone = true;
```

```
bool okZone = false;

if((delta > 0 ) && (priceNow > pivot) && (priceNow < ( pivot + ZONE_Points * _Point)))  okZone = true;
if((delta < 0 ) && (priceNow < pivot) && (priceNow > ( pivot - ZONE_Points * _Point)))  okZone = true;
```

![michal_](https://c.mql5.com/avatar/avatar_na2.png)

**[michal\_](https://www.mql5.com/en/users/michal_)**
\|
7 Aug 2025 at 10:54

Good in-direction bars rating for velocity detector:

We can filter parcentually green/red bars.

(Negative direction was flipped to positive values)

0.8 = 80% bars was on the right way.

```
   double goodBars = 1;

   for(int i=ArraySize(d) - 1; i>=0; i--)
         if(d[i] < 0) goodBars=1 - (i+1.0)/ArraySize(d);
```

![Moving Average in MQL5 from scratch: Plain and simple](https://c.mql5.com/2/102/Moving_average_in_MQL5_from_scratch__LOGO.png)[Moving Average in MQL5 from scratch: Plain and simple](https://www.mql5.com/en/articles/16308)

Using simple examples, we will examine the principles of calculating moving averages, as well as learn about the ways to optimize indicator calculations, including moving averages.

![Volumetric neural network analysis as a key to future trends](https://c.mql5.com/2/101/Trading_Analysis_with_LSTM_Volume_as_a_Key_to_Future_Trends__LOGO__2.png)[Volumetric neural network analysis as a key to future trends](https://www.mql5.com/en/articles/16062)

The article explores the possibility of improving price forecasting based on trading volume analysis by integrating technical analysis principles with LSTM neural network architecture. Particular attention is paid to the detection and interpretation of anomalous volumes, the use of clustering and the creation of features based on volumes and their definition in the context of machine learning.

![From Novice to Expert: Animated News Headline Using MQL5 (III) — Indicator Insights](https://c.mql5.com/2/153/18528-from-novice-to-expert-animated-logo.png)[From Novice to Expert: Animated News Headline Using MQL5 (III) — Indicator Insights](https://www.mql5.com/en/articles/18528)

In this article, we’ll advance the News Headline EA by introducing a dedicated indicator insights lane—a compact, on-chart display of key technical signals generated from popular indicators such as RSI, MACD, Stochastic, and CCI. This approach eliminates the need for multiple indicator subwindows on the MetaTrader 5 terminal, keeping your workspace clean and efficient. By leveraging the MQL5 API to access indicator data in the background, we can process and visualize market insights in real-time using custom logic. Join us as we explore how to manipulate indicator data in MQL5 to create an intelligent and space-saving scrolling insights system, all within a single horizontal lane on your trading chart.

![Mastering Log Records (Part 9): Implementing the builder pattern and adding default configurations](https://c.mql5.com/2/151/18602-mastering-log-records-part-logo.png)[Mastering Log Records (Part 9): Implementing the builder pattern and adding default configurations](https://www.mql5.com/en/articles/18602)

This article shows how to drastically simplify the use of the Logify library with the Builder pattern and automatic default configurations. It explains the structure of the specialized builders, how to use them with smart auto-completion, and how to ensure a functional log even without manual configuration. It also covers tweaks for MetaTrader 5 build 5100.

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/18616&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069583335585417054)

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