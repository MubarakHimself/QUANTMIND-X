---
title: The MQL5 Standard Library Explorer (Part 1): Introduction with CTrade, CiMA, and CiATR
url: https://www.mql5.com/en/articles/19341
categories: Trading Systems, Integration
relevance_score: 3
scraped_at: 2026-01-23T18:31:04.546172
---

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/19341&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069507112800814646)

MetaTrader 5 / Examples


### Discussion Structure:

1. [Introduction](https://www.mql5.com/en/articles/19341#para1)
2. [Implementation](https://www.mql5.com/en/articles/19341#para2)
3. [Tests and results](https://www.mql5.com/en/articles/19341#para3)
4. [Conclusion](https://www.mql5.com/en/articles/19341#para4)
5. [Key Takeaways](https://www.mql5.com/en/articles/19341#para5)
6. [Attachments](https://www.mql5.com/en/articles/19341#para6)

### Introduction

Many developers, especially those new to MQL5, often shy away from using classes and libraries. This hesitation is understandable; simple, [procedural programming](https://en.wikipedia.org/wiki/Procedural_programming "https://en.wikipedia.org/wiki/Procedural_programming") feels more immediate and easier to conceptualize. While this is a valid starting point, progressing to more sophisticated systems requires a shift in mindset. Embracing [object-oriented programming (OOP](https://en.wikipedia.org/wiki/Object-oriented_programming "https://en.wikipedia.org/wiki/Object-oriented_programming")) isn't just about added complexity—it's about achieving a higher level of professionalism, efficiency, and simplicity eventually.

The [MQL5 Standard Library](https://www.mql5.com/en/docs/standardlibrary) is the perfect embodiment of this principle. It is not a cryptic set of rules to be mastered, but rather a collection of powerful, pre-built tools designed to eliminate repetitive tasks. Think of it not as inventing the wheel from scratch, but as assembling a high-performance vehicle from expertly crafted parts. The library provides the puzzle pieces; your job is to join them to create a complete picture.

This article series is designed to be that guide. Whether you're a newcomer eager to build your first expert advisor or a seasoned pro looking to streamline your code, we will demystify the [Standard Library](https://www.mql5.com/en/docs/standardlibrary). Our goal is to show you how to construct complex, robust trading systems with minimal boilerplate code, leveraging the work already done by MetaQuotes' developers.

I have consistently observed that many developers overlook the profound utility of the standard library, often struggling to implement features that are already available within it. This library is a gateway to simplicity, offering elegant solutions to common challenges in algorithmic trading.

Furthermore, the library's potential is largely untapped. Many of its components remain in the background, hidden gems waiting to be discovered and applied. With every update of MetaTrader 5, the Standard Library grows more powerful and expansive. By learning to work with it, we don't just use a tool—we learn a language of efficiency that allows us to extend its capabilities and invent new features ourselves.

This discussion pulls back the curtain and reviews these hidden gems. Let's embark on a journey to unlock the full potential of the MQL5 Standard Library, transforming the way we code and trade.

In this first part of our series, we develop a complete Expert Advisor that demonstrates how three popular classes from the MQL5 Standard Library—CTrade, CiMA, and CiATR — can be combined into a practical trading system. The focus here is to deliver a fully functional idea that readers can compile, test, and adjust without needing to wait for future installments.

**Background**

While the MQL5 documentation provides the formal definition of classes like [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade), [CiMA](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/trendindicators/cima), and [CiATR](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/oscillatorindicators/ciatr)—listing their inheritance, available methods, and brief descriptions—it stops short of showing developers how to combine them into practical trading systems. For a newcomer, this gap can be intimidating; the docs tell you what exists, but not how to use it effectively. That is why our discussion and many other articles are crucial.

Here we go beyond the API listings and step into application. We explain not only what each class represents, but also how to integrate it within the Expert Advisor lifecycle, how its outputs should be interpreted, and how it interacts with other classes to form a complete strategy. In short, this series bridges the gap between reference documentation and real-world practice, giving readers a guided path from raw class definitions to a working trading robot.

**Why these classes?**

The MQL5 Standard Library is bundled with every MetaTrader 5 installation, and its source files are readily accessible in MetaEditor 5. By default, these files are located under the Include directory of your MQL5 installation (for example, _MQL5\\Include\\Trade\ or MQL5\\Include\\Indicators\_). Each module is organized into header files (.mqh) that define reusable classes such as CTrade, CiMA, and CiATR. In MetaEditor, you can easily browse these files from the Navigator panel under Include → Standard Library, or open them directly via File → Open. This accessibility means you are free to use the classes in your projects by adding #include statements, and also to inspect the source code itself, learn how the library is structured, and even extend or customize it for your own development needs

![ShareX_fVhTmSxwMI](https://c.mql5.com/2/170/ShareX_fVhTmSxwMI.gif)

Figure 1. Accessing the MQL5 Standard Library in MetaEditor 5

CTrade—the execution engine

[CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) is the [Standard-Library](https://www.mql5.com/en/docs/standardlibrary) wrapper that handles trade submission and basic order management for you. Instead of constructing low-level [MqlTradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderesult) structures every time, CTrade exposes simple methods (Buy, Sell, PositionClose, etc.) and then reports standardized results you can inspect. In the EA, we’ll rely on CTrade for all execution tasks: placing entries with the computed size, checking the returned status, logging the failure reason when a request is rejected, and performing safe retries or disabling trading when systemic errors appear. Using CTrade makes the execution code shorter, easier to audit, and more consistent across brokers.

```
#include <Trade\Trade.mqh>

// Global declarations
CTrade trade;
```

CiATR — volatility and sizing

[CiATR](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/oscillatorindicators/ciatr) ( _from Indicators\\Oscillators.mqh_) gives us Average True Range in price units, which we use as the volatility engine. In practice we read the ATR value to (a) decide whether the market is too choppy or too quiet to trade, (b) compute a dynamic stop distance (e.g., stop = ATR \* multiplier), and (c) convert that distance into money-per-lot so we can compute a risk-based lot size. The EA will create a CiATR instance once, refresh it each tick, check that its output is valid, and use a small helper routine to convert ATR → loss-per-lot → lots normalized to the broker’s volume step.

```
#include <Indicators\Oscilators.mqh>
CiATR atr;
```

CiMA — the trend detector

[CiMA](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/trendindicators/cima) ( _from Indicators\\Trend.mqh_) wraps moving-average logic and buffer handling so you have readable, stable MA values available in your EA. We use two CiMA instances (fast and slow) to produce a simple directional filter: when the fast MA is above the slow MA the market is considered “uptrend,” and vice versa. CiMA also provides convenient access to the recent MA values (current or last closed bar) so you can implement pullback logic (e.g., enter when price pulls back toward the fast MA by a fraction of ATR). Like the others, create CiMA once, refresh on ticks, guard against EMPTY\_VALUE, and prefer Main(1) for bar-close decisions if you want non-repainting signals.

```
#include <Indicators\Trend.mqh>
CiMA fastMA, slowMA;
```

Essential lifecycle (brief)

- OnInit()—Create() each class instance once.
- OnTick()—call Refresh() each tick, read Main() (check for EMPTY\_VALUE), run decision logic, compute size, then call CTrade methods to execute.
- OnDeinit()—call FullRelease() on indicators and clean up chart objects

With these principles in place—clear responsibilities for each class, a disciplined lifecycle, unit-aware sizing, and robust execution checks—we can now move into the Implementation of a concise, readable EA that wires CiMA, CiATR and CTrade into the full decision pipeline (Init → Refresh → Detect → Size → Execute → Manage) and includes the safety and logging hooks we’ll need for reliable testing.

### Implementation

With the foundation set, we now turn to the practical side of this discussion—how to actually bring these classes together into a working trading solution. The aim here is not just to present abstract descriptions, but to walk through clear, structured steps that show how each class—whether for trade execution, indicator analysis, or account monitoring—fits into the broader trading logic. This section will guide you from concept to code, ensuring that the transition from theory to practice is both logical and approachable.

File header, includes and user inputs

This first block declares what the program is and brings in the building blocks the EA uses. It also exposes a set of user inputs so the reader can tune periods, risk, and safety limits without changing the code. Keeping parameters at the top makes experiments and classroom testing easy.

```
//+------------------------------------------------------------------+
//| CiMA_Trend_Trader.mq5                                            |
//| Compact, working EA that uses CiMA (trend), CiATR (volatility)   |
//| and CTrade (execution). Adjustable inputs.                       |
//+------------------------------------------------------------------+
#property copyright "Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
#property version   "1.00"

#include <Trade\Trade.mqh>
#include <Indicators\Trend.mqh>          // CiMA
#include <Indicators\Oscilators.mqh>     // CiATR

input int    InpFastMAPeriod   = 8;      // Fast MA period
input int    InpSlowMAPeriod   = 34;     // Slow MA period
input int    InpATRPeriod      = 14;     // ATR period
input double InpRiskPercent    = 0.5;    // % of balance risk per trade
input double InpATRstopMult    = 1.5;    // SL = ATR * this
input double InpPullbackFactor = 0.5;    // price within this * ATR of fastMA to consider entry
input double InpTakeProfitRR   = 2.0;    // TP = R * RR
input double InpMaxChopPct     = 0.006;  // skip if ATR/price > this (too choppy)
input double InpMinVolPct      = 0.0005; // skip if ATR/price < this (too quiet)
input double InpMaxDailyLossPct= 2.0;    // disable trading for the day if loss exceeds this %
input int    InpMaxPosPerSym   = 1;      // max open positions per symbol (basic cap)
input int    InpMagicNumber    = 202507; // EA magic number for labeling
input int    InpSlippage       = 5;      // slippage in points (not required for CTrade but useful to log)
input bool   InpShowTradeLabels = true;  // draw arrows/labels on chart
```

Global objects and simple state

Here we declare the shared objects the EA will use continuously: the trade handler and the indicator instances. A small state variable records recent trade time. These global variables persist across initialization, ticks, and deinitialization, so the logic can reuse handles and avoid expensive re-creation.

```
// Global objects
CTrade  trade;
CiMA    fastMA, slowMA;
CiATR   atr;

// Internal tracking
datetime g_lastTradeTime = 0;
```

Initialization: creating indicators and preparing the EA

On startup we create each indicator once and validate creation. If any creation step fails, the EA stops with a clear message. A short timer is set for optional housekeeping. This pattern avoids resource leaks and keeps per-tick work minimal.

```
int OnInit()
{
  // Create indicator objects
  if(!fastMA.Create(_Symbol, _Period, InpFastMAPeriod, 0, MODE_EMA, PRICE_CLOSE))
  {
    Print("[OnInit] fastMA.Create failed for ", _Symbol);
    return(INIT_FAILED);
  }
  if(!slowMA.Create(_Symbol, _Period, InpSlowMAPeriod, 0, MODE_EMA, PRICE_CLOSE))
  {
    Print("[OnInit] slowMA.Create failed for ", _Symbol);
    return(INIT_FAILED);
  }
  if(!atr.Create(_Symbol, _Period, InpATRPeriod))
  {
    Print("[OnInit] atr.Create failed for ", _Symbol);
    return(INIT_FAILED);
  }

  EventSetTimer(1); // optional: ensure OnTimer for housekeeping if needed
  Print("CiMA_Trend_Trader initialized: ", _Symbol, " ", EnumToString(_Period));
  return(INIT_SUCCEEDED);
}
```

Deinitialization and optional timer

When the EA is removed, we release all indicator resources and stop the timer. The timer handler itself is left as a small hook for diagnostics or resets if you want to expand the EA later.

```
void OnDeinit(const int reason)
{
  // release indicator resources
  fastMA.FullRelease(true);
  slowMA.FullRelease(true);
  atr.FullRelease(true);
  EventKillTimer();
}

void OnTimer()
{
  // Simple daily reset or housekeeping can be added here if required
}
```

Position counting (safety control)

Before opening new trades, the EA checks how many positions are already open on this symbol. This function scans current positions and returns the count. It’s used to respect a simple maximum-concurrency rule.

```
int CountOpenPositionsForSymbol()
{
  int total = PositionsTotal();
  int count = 0;
  for(int i=0;i<total;i++)
  {
    ulong ticket = PositionGetTicket(i);
    if(ticket==0) continue;
    if(!PositionSelectByTicket(ticket)) continue;
    string sym = PositionGetString(POSITION_SYMBOL);
    if(sym == _Symbol) count++;
  }
  return(count);
}
```

Daily-loss guard (risk protection)

To avoid runaway losses, this routine scans the closed deals for today and sums profit/loss. If today's losses exceed the configured percentage of account balance, the EA refuses to open further trades. It’s a simple but effective emergency brake.

```
bool IsDailyLossExceeded()
{
  // compute today's closed profit/loss for the account in account currency
  double closedPL = 0.0;
  datetime dayStart = iTime(_Symbol, PERIOD_D1, 0); // start of current day on chart's server timezone
  ulong from = (ulong)dayStart;

  int deals = HistoryDealsTotal();
  for(int i=deals-1;i>=0;i--)
  {
    ulong ticket = HistoryDealGetTicket(i);
    if(ticket==0) continue;
    datetime t = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
    if(t < dayStart) continue;
    double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT) + HistoryDealGetDouble(ticket, DEAL_COMMISSION) + HistoryDealGetDouble(ticket, DEAL_SWAP);
    closedPL += profit;
  }

  // If closedPL is negative and its absolute exceeds percent of balance, return true
  double threshold = AccountInfoDouble(ACCOUNT_BALANCE) * (InpMaxDailyLossPct/100.0);
  if(closedPL < -threshold) return(true);
  return(false);
}
```

Lot normalization (broker rules)

Different brokers restrict minimum, maximum, and step sizes for volume. This helper converts a raw lot number into a valid lot that complies with the symbol settings, returning zero when the computed volume is below the broker minimum. Normalization prevents rejected orders due to invalid volumes.

```
double NormalizeLotToStep(double lots)
{
  double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
  double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
  double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
  if(lotStep <= 0) lotStep = 0.01; // fallback
  if(minLot <= 0) minLot = 0.01;

  // floor to step
  double steps = MathFloor(lots / lotStep);
  double result = steps * lotStep;
  if(result < minLot) result = 0.0;
  if(result > maxLot) result = maxLot;

  // round to reasonable decimals (broker may require 2-3 decimals)
  int digits = 2;
  // attempt to infer decimals from lotStep
  if(lotStep < 0.01) digits = MathMax(0, (int)MathCeil(-MathLog10(lotStep)));
  result = NormalizeDouble(result, digits);
  return(result);
}
```

Volume calculation from ATR (risk sizing)

This routine shows the practical conversion from a volatility distance into a trade volume. It first computes the monetary loss per lot for the proposed stop distance, then divides the allowed risk amount by that loss to get lots. A safe fallback is included for symbols lacking tick-value metadata.

```
double CalculateVolume(double stop_distance_price)
{
  if(stop_distance_price <= 0) return(0.0);

  double risk_amount = AccountInfoDouble(ACCOUNT_BALANCE) * (InpRiskPercent/100.0);

  double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
  double tick_size  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

  if(tick_value <= 0 || tick_size <= 0)
  {
    // fallback approximate using point and lot value: (this may be inaccurate for some symbols)
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    double contract_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
    if(point<=0 || contract_size<=0) return(0.0);
    double value_per_price = contract_size; // rough
    double loss_per_lot = stop_distance_price * value_per_price;
    if(loss_per_lot <= 0) return(0.0);
    double lots = risk_amount / loss_per_lot;
    return NormalizeLotToStep(lots);
  }

  double value_per_price = tick_value / tick_size; // money per price unit for 1 lot
  double loss_per_lot = stop_distance_price * value_per_price;
  if(loss_per_lot <= 0) return(0.0);

  double lots = risk_amount / loss_per_lot;
  double normalized = NormalizeLotToStep(lots);
  return(normalized);
}
```

Visual marking of entries

For clarity during testing, this helper draws an arrow and a short label at the time of an entry. Visual cues are invaluable when matching numeric logs to chart events and help students validate the EA’s decisions at a glance.

```
void DrawEntryLabel(bool isLong, datetime when, double price, string tag)
{
  if(!InpShowTradeLabels) return;
  string objName = tag + "_" + IntegerToString((int)when);
  int arrow = isLong ? 233 : 234; // Wingdings arrows
  // create arrow
  if(!ObjectCreate(0, objName, OBJ_ARROW, 0, when, price))
  {
    // try to remove old and recreate
    ObjectDelete(0, objName);
    ObjectCreate(0, objName, OBJ_ARROW, 0, when, price);
  }
  ObjectSetInteger(0, objName, OBJPROP_COLOR, isLong ? clrBlue : clrRed);
  ObjectSetInteger(0, objName, OBJPROP_WIDTH, 2);
  ObjectSetInteger(0, objName, OBJPROP_ARROWCODE, arrow);
  ObjectSetString(0, objName, OBJPROP_TEXT, tag);
}
```

Order entry attempts (long and short)

These two procedures encapsulate the full pre-check → sizing → execution flow for each direction. They perform safety checks, compute stop and target, obtain a normalized volume and then place an order. On failure they log the reason; on success they record the time and optionally mark the chart.

```
void TryOpenLong(double atr_val)
{
  // safety checks
  if(CountOpenPositionsForSymbol() >= InpMaxPosPerSym) return;
  if(IsDailyLossExceeded()) return;

  double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
  double stop_distance_price = atr_val * InpATRstopMult;
  if(stop_distance_price<=0) return;

  double sl = price - stop_distance_price;
  double tp = price + stop_distance_price * InpTakeProfitRR;

  double vol = CalculateVolume(stop_distance_price);
  if(vol <= 0) return;

  // attempt to place market buy
  bool ok = trade.Buy(vol, NULL, 0.0, sl, tp, "CiMA_Long_");
  if(!ok)
  {
    PrintFormat("[TryOpenLong] Buy failed. retcode=%d comment=%s", trade.ResultRetcode(), trade.ResultComment());
    return;
  }

  // success
  PrintFormat("[TryOpenLong] Bought %.2f lots at %.5f SL=%.5f TP=%.5f", vol, price, sl, tp);
  g_lastTradeTime = TimeCurrent();
  DrawEntryLabel(true, TimeCurrent(), price, "CiMA_LONG");
}

void TryOpenShort(double atr_val)
{
  // safety checks
  if(CountOpenPositionsForSymbol() >= InpMaxPosPerSym) return;
  if(IsDailyLossExceeded()) return;

  double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
  double stop_distance_price = atr_val * InpATRstopMult;
  if(stop_distance_price<=0) return;

  double sl = price + stop_distance_price;
  double tp = price - stop_distance_price * InpTakeProfitRR;

  double vol = CalculateVolume(stop_distance_price);
  if(vol <= 0) return;

  bool ok = trade.Sell(vol, NULL, 0.0, sl, tp, "CiMA_Short_");
  if(!ok)
  {
    PrintFormat("[TryOpenShort] Sell failed. retcode=%d comment=%s", trade.ResultRetcode(), trade.ResultComment());
    return;
  }

  PrintFormat("[TryOpenShort] Sold %.2f lots at %.5f SL=%.5f TP=%.5f", vol, price, sl, tp);
  g_lastTradeTime = TimeCurrent();
  DrawEntryLabel(false, TimeCurrent(), price, "CiMA_SHORT");
}
```

Per-tick decision pipeline

On each market tick, the EA refreshes its indicators, reads the latest values, checks basic volatility filters, and then applies the pullback entry rule. This central loop ties all helpers together: it is the heart of the strategy and the point where indicators, sizing, and execution meet.

```
void OnTick()
{
  // refresh indicators (cheap relative to trading operations)
  // Refresh indicators (the Refresh method returns void in the Standard Library; use flag -1 to update recent data)
  fastMA.Refresh(-1);
  slowMA.Refresh(-1);
  atr.Refresh(-1);
  // Note: Refresh does not return a status — check indicator values for EMPTY_VALUE below and skip until ready


  double fast = fastMA.Main(0);
  double slow = slowMA.Main(0);
  double atr_val = atr.Main(0); // price units

  if(fast==EMPTY_VALUE || slow==EMPTY_VALUE || atr_val==EMPTY_VALUE) return;

  double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
  bool uptrend = (fast > slow);
  bool downtrend = (fast < slow);

  // normalized volatility check
  double norm_vol = atr_val / price;
  if(norm_vol > InpMaxChopPct || norm_vol < InpMinVolPct)
  {
    // skip trading in these conditions
    return;
  }

  // Pullback distance between price and fast MA
  double dist = MathAbs(price - fast);

  // Only consider entries on bar close for non-repainting behaviour
  // ensure we use the current completed bar (shift=1) for MA reference if you prefer
  // for simplicity here we use Main(0) but teachers may show Main(1) alternative

  if(uptrend && dist <= InpPullbackFactor * atr_val)
  {
    TryOpenLong(atr_val);
  }
  else if(downtrend && dist <= InpPullbackFactor * atr_val)
  {
    TryOpenShort(atr_val);
  }
}
```

Small utility for readable logs

A tiny helper converts the timeframe enum into a short text label. It improves logged messages and makes debugging clearer when running multiple timeframes.

```
string EnumToString(const ENUM_TIMEFRAMES tf)
{
  switch(tf)
  {
    case PERIOD_M1: return("M1");
    case PERIOD_M5: return("M5");
    case PERIOD_M15: return("M15");
    case PERIOD_H1: return("H1");
    case PERIOD_D1: return("D1");
    default: return(IntegerToString((int)tf));
  }
}
```

### Tests and Results

The CiMA-Trend-Trader EA, once compiled, can be evaluated in the Strategy Tester. During this stage, my focus was not on immediate profitability but rather on confirming that the EA functions correctly and adheres to its primary design objective—following trends. In my test runs, I observed that the EA performs more consistently on higher timeframes such as H4 and above, where market trends are clearer and more respected by trading systems.

On the other hand, lower timeframes tend to exhibit choppy and erratic structures, which introduce noise and can negatively affect the results of a trend-following approach like ours. The following images illustrate my experiences in the Strategy Tester, highlighting how the EA responds across different market conditions.

Initial Test on a 30-Minute Time Frame Chart

![metatester64_mZwsgVXny3.gif](https://c.mql5.com/2/170/metatester64_mZwsgVXny3.gif)

Figure 2. Strategy Tester visualization of GBPUSD, M30 timeframe with CiMA\_Trend\_Trader EA from June 2024

![terminal64_co9AY2VC6g.png](https://c.mql5.com/2/170/terminal64_co9AY2VC6g.png)

Figure 3. Steady equity decline during initial test on M30 timeframe

![terminal64_lZ68zax1Hi.png](https://c.mql5.com/2/170/terminal64_lZ68zax1Hi.png)

Figure 4. GBPUSD, M30, Backtest Results

The initial M30 test on GBPUSD was disappointing from a profitability standpoint. Over a one-year simulation, the EA produced a steady series of losing trades, and the equity curve declined by roughly 50%. Visual inspection in the Strategy Tester revealed the cause: the M30 market structure on GBPUSD contains many short, false moves and choppy consolidation phases where the system repeatedly entered into “fake” trends and was stopped out.

Rather than continuing to tune aggressively around those noisy signals, I changed the Strategy Tester timeframe to a higher resolution—H4—and reran the test. The H4 results in the next section show markedly improved behavior; trends are clearer, signals are less frequent, and individual trades are more respectful of larger market structure.

Test on a 4-Hour Timeframe Chart

![terminal64_706hvT4Eyn.png](https://c.mql5.com/2/170/terminal64_706hvT4Eyn.png)

Figure 5. Slight increase in equity during a test on the H4 timeframe

![terminal64_zrDt2ZmaZM.png](https://c.mql5.com/2/170/terminal64_zrDt2ZmaZM.png)

Figure 6. GBPUSD, H4, Backtest Results

Higher timeframes demonstrated a stronger and more consistent respect for trends, as confirmed by the results we presented above. This indicates that our trend-following system can achieve more reliable outcomes when filtering signals through higher timeframe data. My decision to shift to H4 was an extra step aimed at improving the system’s performance, and indeed it highlighted the potential benefits. That said, there are still many parameters within the EA that can be fine-tuned to further enhance the results.

However, trading on higher timeframes comes with its own trade-offs. It requires the trader to exercise patience, as signals occur less frequently and positions may take days or even weeks to unfold. This approach is therefore best suited for disciplined traders with a long-term perspective, rather than for scalpers or those seeking quick, frequent profits. Larger equity becomes necessary for this trading strategy, as it allows for higher lot sizes per trade. This, in turn, enhances the overall gains that can accumulate over the long term.

Below is an image (Figure 7) of the H4 traded chart, displaying both trade history arrows and signal arrows. The trends are clearly visible, with the moving averages generated by the CiMA class providing a smooth depiction of market direction. This clarity is largely due to the choice of a higher timeframe, which naturally filters out much of the noise seen on lower timeframes.

![metatester64_Zz8Xtlaubg.png](https://c.mql5.com/2/170/metatester64_Zz8Xtlaubg.png)

Figure 7. GBPUSD, H4—Trend Visualization in Strategy Tester

### Conclusion

It is entirely practical to build a functional Expert Advisor with few errors by leveraging the MQL5 Standard Library. Using CTrade, CiMA, and CiATR, we brought our trend-following concept to life with compact, readable code that minimized development friction and reduced common mistakes such as order mismanagement or redundant handle creation. By relying on tested library classes, we could focus more on strategy logic than on low-level implementation details.

In the Strategy Tester, the EA ran smoothly and allowed us to iterate quickly across multiple timeframes. The results reflected a familiar pattern: higher timeframes such as H4 produced clearer, stronger trend captures, while smaller intervals like M30 suffered from false breakouts and noisy consolidations. This highlighted an important design principle—trend-following systems benefit from patience and longer horizons, while short-term approaches require additional filters and safeguards to remain profitable.

Perhaps the most valuable lesson was the process itself, moving from concept to code to results. We began with a trading idea, expressed it through modular standard-library classes, and validated its behavior with strategy tester visualization and backtest results. This systematic workflow produced a working EA and reinforced good programming practices such as separating concerns, using visual diagnostics, and testing across regimes. These practices form the backbone of professional MQL5 development.

To capture the key insights, we have prepared a table with lessons summarizing the programming principles, strategy considerations, and testing techniques that emerged from this project. Each lesson is directly tied to real implementation challenges and solutions, making it a practical reference for readers pursuing their own EA development journey.

Find, the table of attachments below with the project source file. The CiMA\_Trend\_Trader.mq5 EA serves as both a practical demonstration and a reusable starting point for readers who want to experiment further. You can test, modify, and extend the code—perhaps by integrating other standard-library classes such as CiCCI, CiMACD, or volatility filters—to explore new approaches and refine your own trading systems.

Your feedback and insights are welcome. We invite you to share your experiences, suggest improvements, and explore new directions. Stay tuned for our next publication!

### Key Takeaways

| Key lesson | Description: |
| --- | --- |
| Concept → Code | Translate rules into a clear step-by-step algorithm and pseudo-code before implementing; this makes behavior reproducible and easier to test. |
| OOP structure | Use small, focused classes or modules (indicators, sizing, execution) so each part has a single responsibility and can be reused or swapped. |
| Lifecycle management | Create indicator instances once, refresh per tick, and release on deinitialization to avoid resource leaks and test instability. |
| Unit conversions | Be explicit about units: convert ATR (price units) to points and monetary loss using tick size/value to ensure correct stops and sizing. |
| Position sizing | Make sizing a pure calculation: risk amount ÷ loss-per-lot, then normalize to broker volume rules for repeatable exposure. |
| Execution & error handling | Centralize trade submission, always check result codes/messages, and implement clear retry or fail-safe policies on common errors. |
| State control | Maintain explicit state (open counts, last trade time, enabled flag) and update it centrally to avoid race conditions and logic drift. |
| Defensive coding | Anticipate missing data, zero tick values, and short histories; fail gracefully with logs and fallback behaviour rather than crashing. |
| Modularity | Separate concerns (signals, sizing, safety, drawing, execution) into functions or files to make testing and maintenance simpler. |
| Logging & visualization | Log decision-level values and use chart overlays (MAs, ATR bands, entry markers) to correlate numeric logic with visual behavior. |
| High-quality testing | Run 99% tick-mode backtests with realistic spreads/swaps; poor data leads to misleading conclusions. |
| Optimization discipline | Optimize conservatively, use out-of-sample or walk-forward validation, and focus on stable parameter regions rather than peak backtest gains. |
| Performance awareness | Minimize per-tick allocations, reuse handles, and measure CPU/memory when scaling to many symbols or timeframes. |
| Runtime protections | Include operational safeguards: max daily loss, max positions, emergency stop and health checks to limit live risk. |
| Experiment workflow | Track experiments (code versions, .set files, metrics), run A/B comparisons, and iterate on small, measurable changes. |

### Attachments

| File name | Description |
| --- | --- |
| CiMA\_Trend\_Trader.mq5 | Standalone trend-following Expert Advisor that demonstrates practical use of three Standard Library classes: CiMA (fast & slow-moving averages for direction), CiATR (ATR-based volatility for dynamic stops and position sizing), and CTrade (order execution). Features tunable inputs (MA periods, ATR period & multipliers, risk %, TP/SL RR, volatility filters, max daily loss, max positions), broker-aware lot normalization, safety checks, simple visual entry markers, and clear logging for Strategy Tester validation. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19341.zip "Download all attachments in the single ZIP archive")

[CiMA\_Trend\_Trader.mq5](https://www.mql5.com/en/articles/download/19341/CiMA_Trend_Trader.mq5 "Download CiMA_Trend_Trader.mq5")(25.85 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [From Novice to Expert: Higher Probability Signals](https://www.mql5.com/en/articles/20658)
- [From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)
- [From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)
- [From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)
- [From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)
- [The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)
- [The MQL5 Standard Library Explorer (Part 4): Custom Signal Library](https://www.mql5.com/en/articles/20266)

**[Go to discussion](https://www.mql5.com/en/forum/496091)**

![Automating Trading Strategies in MQL5 (Part 33): Creating a Price Action Shark Harmonic Pattern System](https://c.mql5.com/2/171/19479-automating-trading-strategies-logo.png)[Automating Trading Strategies in MQL5 (Part 33): Creating a Price Action Shark Harmonic Pattern System](https://www.mql5.com/en/articles/19479)

In this article, we develop a Shark pattern system in MQL5 that identifies bullish and bearish Shark harmonic patterns using pivot points and Fibonacci ratios, executing trades with customizable entry, stop-loss, and take-profit levels based on user-selected options. We enhance trader insight with visual feedback through chart objects like triangles, trendlines, and labels to clearly display the X-A-B-C-D pattern structure

![Moving to MQL5 Algo Forge (Part 3): Using External Repositories in Your Own Projects](https://c.mql5.com/2/170/19436-perehodim-na-mql5-algo-forge-logo.png)[Moving to MQL5 Algo Forge (Part 3): Using External Repositories in Your Own Projects](https://www.mql5.com/en/articles/19436)

Let's explore how you can start integrating external code from any repository in the MQL5 Algo Forge storage into your own project. In this article, we finally turn to this promising, yet more complex, task: how to practically connect and use libraries from third-party repositories within MQL5 Algo Forge.

![How to build and optimize a cycle-based trading system (Detrended Price Oscillator - DPO)](https://c.mql5.com/2/171/19547-how-to-build-and-optimize-a-logo.png)[How to build and optimize a cycle-based trading system (Detrended Price Oscillator - DPO)](https://www.mql5.com/en/articles/19547)

This article explains how to design and optimise a trading system using the Detrended Price Oscillator (DPO) in MQL5. It outlines the indicator's core logic, demonstrating how it identifies short-term cycles by filtering out long-term trends. Through a series of step-by-step examples and simple strategies, readers will learn how to code it, define entry and exit signals, and conduct backtesting. Finally, the article presents practical optimization methods to enhance performance and adapt the system to changing market conditions.

![Overcoming The Limitation of Machine Learning (Part 4): Overcoming Irreducible Error Using Multiple Forecast Horizons](https://c.mql5.com/2/171/19383-overcoming-the-limitation-of-logo__1.png)[Overcoming The Limitation of Machine Learning (Part 4): Overcoming Irreducible Error Using Multiple Forecast Horizons](https://www.mql5.com/en/articles/19383)

Machine learning is often viewed through statistical or linear algebraic lenses, but this article emphasizes a geometric perspective of model predictions. It demonstrates that models do not truly approximate the target but rather map it onto a new coordinate system, creating an inherent misalignment that results in irreducible error. The article proposes that multi-step predictions, comparing the model’s forecasts across different horizons, offer a more effective approach than direct comparisons with the target. By applying this method to a trading model, the article demonstrates significant improvements in profitability and accuracy without changing the underlying model.

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/19341&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069507112800814646)

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