---
title: Using optimization algorithms to configure EA parameters on the fly
url: https://www.mql5.com/en/articles/14183
categories: Trading, Integration, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T18:01:46.944516
---

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/14183&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049572430118235609)

MetaTrader 5 / Tester


### Contents

1\. [Introduction](https://www.mql5.com/en/articles/14183#tag1)

2\. [Architecture of an EA with self-optimization](https://www.mql5.com/en/articles/14183#tag2)

3\. [Virtualization of indicators](https://www.mql5.com/en/articles/14183#tag3)

4\. [Virtualization of strategy](https://www.mql5.com/en/articles/14183#tag4)

5\. [Testing functionality](https://www.mql5.com/en/articles/14183#tag5)

### 1\. Introduction

I often get asked questions about how to apply optimization algorithms when working with EAs and strategies in general. In this article I would like to look at the practical aspects of using optimization algorithms.

In today's financial world, where every millisecond can make a huge difference, algorithmic trading is becoming increasingly necessary. Optimization algorithms play a key role in creating efficient trading strategies. Perhaps some skeptics believe that optimization algorithms and trading have no common ground. However, in this article I will show how these two areas can interact and what benefits can be obtained from this interaction.

For novice traders, understanding the basic principles of optimization algorithms can be a powerful tool in finding profitable trades and minimizing risks. For seasoned professionals, deep knowledge in this area can open up new horizons and help create sophisticated trading strategies that exceed expectations.

**Self-optimization** in an EA is a process involving the EA adapting its trading strategy parameters to achieve better performance based on historical data and current market conditions. The process may include the following aspects:

- **Data collection and analysis.** The EA should collect and analyze historical market data. This may involve the use of various data analysis techniques such as statistical analysis, machine learning and artificial intelligence.

- **Setting targets**. The EA should have clearly defined targets it strives to achieve. This could be maximizing profit, minimizing risk, or achieving a certain level of profitability.

- **Applying optimization algorithms**. To achieve the best results, an EA can use various optimization algorithms. These algorithms help the EA find the optimal values of the strategy parameters.

- **Testing and validation.** After optimization, the EA should be backtested and validated against current market conditions to ensure its efficiency. Testing helps evaluate the EA performance and its ability to adapt to changing market conditions.

- **Monitoring and updating.** The EA should constantly monitor its performance and update its strategy parameters if necessary. Markets are constantly changing, and the EA should be willing to adapt to new conditions and changes in trends and volatility.

There are several main scenarios for using optimization algorithms in trading:

- **Optimization of trading strategy parameters.** Optimization algorithms can be used to adjust the parameters of trading strategies. Using these methods, we can determine the best values for parameters such as moving average periods, stop loss and take profit levels, or other parameters associated with trading signals and rules.

- **Optimizing the time of market entry/exit**. Optimization algorithms can help determine optimal times to enter and exit positions based on historical data and current market conditions. For example, optimization algorithms can be used to determine optimal time intervals for trading signals.

- **Portfolio management.** Optimization algorithms can help determine the optimal asset allocation in a portfolio to achieve given goals. For example, we can use optimization techniques, such as Mean-Variance Optimization, to find the most efficient mix of assets given expected return and risk. This may include determining the optimal mix between stocks, bonds and other assets, as well as optimizing position sizes and portfolio diversification.

- **Development of trading strategies.** Optimization algorithms can be used to develop new trading strategies. For example, we can use genetic programming to evolutionarily search for optimal rules for entering and exiting positions based on historical data.

- **Risk management.** Optimization algorithms can help manage risk in trading. For example, we can use optimization algorithms to calculate the optimal position size or determine a dynamic stop loss level that minimizes potential losses.

- **Selecting the best trading instruments.** Optimization algorithms can help in choosing the best trading instruments or assets to trade. For example, optimization algorithms can be used to rank assets based on various criteria such as profitability, volatility or liquidity.

- **Forecasting financial markets.** Optimization algorithms can be used to predict financial markets. Optimization algorithms can be used to tune the parameters of predictive models or to select optimal combinations of predictive models.

These are just some examples of scenarios for using optimization algorithms in trading. Overall, optimization algorithms can help automate and improve various aspects of trading, from finding optimal strategies to risk and portfolio management.

### 2\. Architecture of an EA with self-optimization

To ensure EA self-optimization, several schemes are possible, but one of the simplest and minimally necessary to implement any required capabilities and functionality is the one presented in Figure 1.

On the "History" time line, the EA is at the "time now" point, where the optimization decision is made. The "EA" calls the "Manager function", which manages the optimization process. The EA passes the "optimization parameters" to this function.

In turn, the manager requests a set of parameters from the "optimization ALGO" or "AO", which will now and henceforth be referred to as "set". After that, the manager transfers the set to the virtual trading strategy "EA Virt", which is a complete analogue of the real strategy that works and carries out trading operations, "EA".

"EA Virt" carries out virtual trading from the "past" point in history to the "time now" point. The manager runs "EA Virt" as many times as specified in the population size of the "optimization parameters". "EA Virt" in turn returns the result of the run on history in the form of "ff result".

"ff result" is the result of a fitness function, or fitness, or an optimization criterion, which can be anything at the user discretion. This could be, for example, a balance, profit factor, mathematical expectation, or a complex criterion, as well as integral, or cumulative differential one, measured at many points in "History" time. Thus, the fitness function result, or "ff result", is what the user considers to be an important indicator of the trading strategy quality.

Next, "ff result", which is an assessment of a specific set, is passed by the manager to the optimization algorithm.

When the stop condition is reached, the manager transfers the best set to the "EA", after which the EA continues working (trading) with new updated parameters from the "time now" point to the "reoptimiz" re-optimization point, where it is optimized again to a given history depth.

The re-optimization point can be chosen for various reasons. It can be a strictly defined number of history bars, as in the example below, or some specific condition, for example, a decrease in trading indicators to a certain critical level.

![Scheme](https://c.mql5.com/2/69/Scheme.png)

Figure 1. EA self-optimization structure

According to the "optimization ALGO" optimization algorithm scheme, it can be considered as a "black box" that performs its work autonomously (however, everything outside is also a "black box" for it), regardless of the specific trading strategy, manager and virtual strategy. The manager requests a set from the optimization algorithm, and sends back an assessment of this set. This assessment is used by the optimization algorithm to determine the next set. This cycle continues until the best set of parameters is found to meet the user's requirements. Thus, the optimization algorithm searches for the optimal parameters - precisely the ones that satisfy the user's needs specified via the fitness function in "EA Virt".

### 3\. Virtualization of indicators

To run the EA on historical data, we need to create a virtual copy of a trading strategy that will perform the same trading operations as when working on a trading account. When indicators are not used, virtualization of logical conditions within the EA becomes relatively simple. We only need to describe the logical actions according to a time point in the price series. At the same time, the use of indicators is a more complex task and most of the time, trading strategies rely on various indicators.

The problem is that when searching for optimal indicator parameters, it is necessary to create indicator handles with the current set at a given iteration. Upon completion of the run on historical data, these handles should be deleted, otherwise the RAM may quickly fill up, especially if there is a large number of possible options for a set of parameters (sets). This is not a problem if this procedure is carried out on a symbol chart, but deleting handles is not allowed in the tester.

To solve the problem, we will need to "virtualize" the calculation of the indicator within the executable EA to avoid the use of handles. Let's take the Stochastic indicator as an example.

The calculation part of each indicator contains the standard OnCalculate function. This function should be renamed, say, to "Calculate", and left virtually unchanged.

We need to design the indicator as a class (a structure is also suitable). Let’s call it "C\_Stochastic". In the class declaration, we need to register the main indicator buffers as public fields (additional calculation buffers can be private) and declare the Init initialization function you need to pass the indicator parameters to.

```
//——————————————————————————————————————————————————————————————————————————————
class C_iStochastic
{
  public: void Init (const int InpKPeriod,       // K period
                     const int InpDPeriod,       // D period
                     const int InpSlowing)       // Slowing
  {
    inpKPeriod = InpKPeriod;
    inpDPeriod = InpDPeriod;
    inpSlowing = InpSlowing;
  }

  public: int Calculate (const int rates_total,
                         const int prev_calculated,
                         const double &high  [],
                         const double &low   [],
                         const double &close []);

  //--- indicator buffers
  public:  double ExtMainBuffer   [];
  public:  double ExtSignalBuffer [];
  private: double ExtHighesBuffer [];
  private: double ExtLowesBuffer  [];

  private: int inpKPeriod; // K period
  private: int inpDPeriod; // D period
  private: int inpSlowing; // Slowing
};
//——————————————————————————————————————————————————————————————————————————————
```

As well as the calculation of the indicator itself in the "Calculate" method. The calculation of the indicator is absolutely no different from the indicator included in the terminal standard delivery. The only difference is the size distribution for indicator buffers and their initialization.

This is a very simple example to understand the principle of indicator virtualization. The calculation is performed for the entire depth of periods specified in the indicator parameters. It is possible to organize the ability to additionally calculate only the last bar and implement ring buffers, but the purpose of the article is to show a simple example that requires minimal intervention in converting the indicator into a virtual form and is accessible to users with minimal programming skills.

```
//——————————————————————————————————————————————————————————————————————————————
int C_iStochastic::Calculate (const int rates_total,
                              const int prev_calculated,
                              const double &high  [],
                              const double &low   [],
                              const double &close [])
{
  if (rates_total <= inpKPeriod + inpDPeriod + inpSlowing) return (0);

  ArrayResize (ExtHighesBuffer, rates_total);
  ArrayResize (ExtLowesBuffer,  rates_total);
  ArrayResize (ExtMainBuffer,   rates_total);
  ArrayResize (ExtSignalBuffer, rates_total);

  ArrayInitialize (ExtHighesBuffer, 0.0);
  ArrayInitialize (ExtLowesBuffer,  0.0);
  ArrayInitialize (ExtMainBuffer,   0.0);
  ArrayInitialize (ExtSignalBuffer, 0.0);

  int i, k, start;

  start = inpKPeriod - 1;

  if (start + 1 < prev_calculated)
  {
    start = prev_calculated - 2;
    Print ("start ", start);
  }
  else
  {
    for (i = 0; i < start; i++)
    {
      ExtLowesBuffer  [i] = 0.0;
      ExtHighesBuffer [i] = 0.0;
    }
  }

  //--- calculate HighesBuffer[] and ExtHighesBuffer[]
  for (i = start; i < rates_total && !IsStopped (); i++)
  {
    double dmin =  1000000.0;
    double dmax = -1000000.0;

    for (k = i - inpKPeriod + 1; k <= i; k++)
    {
      if (dmin > low  [k]) dmin = low  [k];
      if (dmax < high [k]) dmax = high [k];
    }

    ExtLowesBuffer  [i] = dmin;
    ExtHighesBuffer [i] = dmax;
  }

  //--- %K
  start = inpKPeriod - 1 + inpSlowing - 1;

  if (start + 1 < prev_calculated) start = prev_calculated - 2;
  else
  {
    for (i = 0; i < start; i++) ExtMainBuffer [i] = 0.0;
  }

  //--- main cycle
  for (i = start; i < rates_total && !IsStopped (); i++)
  {
    double sum_low  = 0.0;
    double sum_high = 0.0;

    for (k = (i - inpSlowing + 1); k <= i; k++)
    {
      sum_low  += (close [k] - ExtLowesBuffer [k]);
      sum_high += (ExtHighesBuffer [k] - ExtLowesBuffer [k]);
    }

    if (sum_high == 0.0) ExtMainBuffer [i] = 100.0;
    else                 ExtMainBuffer [i] = sum_low / sum_high * 100;
  }

  //--- signal
  start = inpDPeriod - 1;

  if (start + 1 < prev_calculated) start = prev_calculated - 2;
  else
  {
    for (i = 0; i < start; i++) ExtSignalBuffer [i] = 0.0;
  }

  for (i = start; i < rates_total && !IsStopped (); i++)
  {
    double sum = 0.0;
    for (k = 0; k < inpDPeriod; k++) sum += ExtMainBuffer [i - k];
    ExtSignalBuffer [i] = sum / inpDPeriod;
  }

  //--- OnCalculate done. Return new prev_calculated.
  return (rates_total);
}
//——————————————————————————————————————————————————————————————————————————————
```

Additionally, I will give an example of virtualization of the MACD indicator:

```
//——————————————————————————————————————————————————————————————————————————————
class C_iMACD
{
  public: void Init (const int InpFastEMA,       // Fast   EMA period
                     const int InpSlowEMA,       // Slow   EMA period
                     const int InpSignalSMA)     // Signal SMA period
  {
    inpFastEMA   = InpFastEMA;
    inpSlowEMA   = InpSlowEMA;
    inpSignalSMA = InpSignalSMA;

    maxPeriod = InpFastEMA;
    if (maxPeriod < InpSlowEMA)   maxPeriod = InpSlowEMA;
    if (maxPeriod < InpSignalSMA) maxPeriod = InpSignalSMA;
  }

  public: int Calculate (const int rates_total,
                         const int prev_calculated,
                         const double &close []);

  //--- indicator buffers
  public:  double ExtMacdBuffer   [];
  public:  double ExtSignalBuffer [];
  private: double ExtFastMaBuffer [];
  private: double ExtSlowMaBuffer [];

  private: int ExponentialMAOnBuffer (const int rates_total, const int prev_calculated, const int begin, const int period, const double& price [],double& buffer []);
  private: int SimpleMAOnBuffer      (const int rates_total, const int prev_calculated, const int begin, const int period, const double& price [],double& buffer []);

  private: int inpFastEMA;   // Fast EMA period
  private: int inpSlowEMA;   // Slow EMA period
  private: int inpSignalSMA; // Signal SMA period
  private: int maxPeriod;
};
//——————————————————————————————————————————————————————————————————————————————
```

Indicator specified part:

```
//——————————————————————————————————————————————————————————————————————————————
int C_iMACD::Calculate (const int rates_total,
                        const int prev_calculated,
                        const double &close [])
{
  if (rates_total < maxPeriod) return (0);

  ArrayResize (ExtMacdBuffer,   rates_total);
  ArrayResize (ExtSignalBuffer, rates_total);
  ArrayResize (ExtFastMaBuffer, rates_total);
  ArrayResize (ExtSlowMaBuffer, rates_total);

  ArrayInitialize (ExtMacdBuffer,   0.0);
  ArrayInitialize (ExtSignalBuffer, 0.0);
  ArrayInitialize (ExtFastMaBuffer, 0.0);
  ArrayInitialize (ExtSlowMaBuffer, 0.0);

  ExponentialMAOnBuffer (rates_total, prev_calculated, 0, inpFastEMA, close, ExtFastMaBuffer);
  ExponentialMAOnBuffer (rates_total, prev_calculated, 0, inpSlowEMA, close, ExtSlowMaBuffer);

  int start;
  if (prev_calculated == 0) start = 0;

  else start = prev_calculated - 1;

  //--- calculate MACD
  for (int i = start; i < rates_total && !IsStopped (); i++) ExtMacdBuffer [i] = ExtFastMaBuffer [i] - ExtSlowMaBuffer [i];

  //--- calculate Signal
  SimpleMAOnBuffer (rates_total, prev_calculated, 0, inpSignalSMA, ExtMacdBuffer, ExtSignalBuffer);

  return (rates_total);
}
//——————————————————————————————————————————————————————————————————————————————
```

The exponential smoothing calculation does not need to be changed at all:

```
//——————————————————————————————————————————————————————————————————————————————
int C_iMACD::ExponentialMAOnBuffer (const int rates_total, const int prev_calculated, const int begin, const int period, const double& price [],double& buffer [])
{
  //--- check period
  if (period <= 1 || period > (rates_total - begin)) return (0);

  //--- save and clear 'as_series' flags
  bool as_series_price  = ArrayGetAsSeries (price);
  bool as_series_buffer = ArrayGetAsSeries (buffer);

  ArraySetAsSeries (price, false);
  ArraySetAsSeries (buffer, false);

  //--- calculate start position
  int    start_position;
  double smooth_factor = 2.0 / (1.0 + period);

  if (prev_calculated == 0) // first calculation or number of bars was changed
  {
    //--- set empty value for first bars
    for (int i = 0; i < begin; i++) buffer [i] = 0.0;

    //--- calculate first visible value
    start_position = period + begin;
    buffer [begin] = price [begin];

    for (int i = begin + 1; i < start_position; i++) buffer [i] = price [i] * smooth_factor + buffer [i - 1] * (1.0 - smooth_factor);
  }
  else start_position = prev_calculated - 1;

  //--- main loop
  for (int i = start_position; i < rates_total; i++) buffer [i] = price [i] * smooth_factor + buffer [i - 1] * (1.0 - smooth_factor);

  //--- restore as_series flags
  ArraySetAsSeries (price,  as_series_price);
  ArraySetAsSeries (buffer, as_series_buffer);
  //---
  return (rates_total);
}
//——————————————————————————————————————————————————————————————————————————————
```

The calculation of simple smoothing also does not require changes:

```
//——————————————————————————————————————————————————————————————————————————————
int C_iMACD::SimpleMAOnBuffer (const int rates_total, const int prev_calculated, const int begin, const int period, const double& price [],double& buffer [])
{
  //--- check period
  if (period <= 1 || period > (rates_total - begin)) return (0);

  //--- save as_series flags
  bool as_series_price = ArrayGetAsSeries (price);
  bool as_series_buffer = ArrayGetAsSeries (buffer);

  ArraySetAsSeries (price, false);
  ArraySetAsSeries (buffer, false);

  //--- calculate start position
  int start_position;

  if (prev_calculated == 0) // first calculation or number of bars was changed
  {
    //--- set empty value for first bars
    start_position = period + begin;

    for (int i = 0; i < start_position - 1; i++) buffer [i] = 0.0;

    //--- calculate first visible value
    double first_value = 0;

    for (int i = begin; i < start_position; i++) first_value += price [i];

    buffer [start_position - 1] = first_value / period;
  }
  else start_position = prev_calculated - 1;

  //--- main loop
  for (int i = start_position; i < rates_total; i++) buffer [i] = buffer [i - 1] + (price [i] - price [i - period]) / period;

  //--- restore as_series flags
  ArraySetAsSeries (price, as_series_price);
  ArraySetAsSeries (buffer, as_series_buffer);

  //---
  return (rates_total);
}
//——————————————————————————————————————————————————————————————————————————————
```

### 4\. Virtualization of strategy

One of the readers of my articles on optimization algorithms, [LUIS ALBERTO BIANUCCI](https://www.mql5.com/en/users/farmabio), kindly provided the code for an EA based on the Stochastic indicator. He asked me to create an example based on this code in order to demonstrate a way to arrange self-learning in EA using the [AO Core](https://www.mql5.com/en/market/product/92455) library and consider this example in the article. Thus, other users will be able to use this method when connecting optimization algorithms in their own developments. I would like to emphasize that this method is suitable for connecting any optimization algorithms discussed in my "Population optimization algorithms" series, due to the fact that the algorithms are designed in a universal form and can be successfully applied in any user projects.

We have already looked at the virtualization of an indicator as part of an EA. Now we move on to consider the virtualization of a strategy. At the beginning of the EA code, we will declare the import of the library, the include files of the standard trading library and the include file of the virtual stochastic.

Next comes the "input" - the EA parameters, among which InpKPeriod\_P and InpUpperLevel\_P are the most notable. They represent the period and levels of the Stochastic indicator and need to be optimized.

input string   InpKPeriod\_P        = "18\|9\|3\|24";  //STO K period:      it is necessary to optimize

input string   InpUpperLevel\_P  = "96\|88\|2\|98"; //STO upper level: it is necessary to optimize

The parameters are declared with a string type, the parameters are compound and include default values, optimization initial value, step and final value.

```
//——————————————————————————————————————————————————————————————————————————————
#import "\\Market\\AO Core.ex5"
bool   Init (int colonySize, double &range_min [], double &range_max [], double &range_step []);
//------------------------------------------------------------------------------
void   Preparation    ();
void   GetVariantCalc (double &variant [], int pos);
void   SetFitness     (double value,       int pos);
void   Revision       ();
//------------------------------------------------------------------------------
void   GetVariant     (double &variant [], int pos);
double GetFitness     (int pos);
#import
//——————————————————————————————————————————————————————————————————————————————

#include <Trade\Trade.mqh>;
#include "cStochastic.mqh"

input group         "==== GENERAL ====";
sinput long         InpMagicNumber      = 132516;       //Magic Number
sinput double       InpLotSize          = 0.01;         //Lots

input group         "==== Trading ====";
input int           InpStopLoss         = 1450;         //Stoploss
input int           InpTakeProfit       = 1200;         //Takeprofit

input group         "==== Stochastic ==|value|start|step|end|==";
input string        InpKPeriod_P        = "18|9|3|24";  //STO K period   : it is necessary to optimize
input string        InpUpperLevel_P     = "96|88|2|98"; //STO upper level: it is necessary to optimize

input group         "====Self-optimization====";
sinput bool         SelfOptimization    = true;
sinput int          InpBarsOptimize     = 18000;        //Number of bars in the history for optimization
sinput int          InpBarsReOptimize   = 1440;         //After how many bars, EA will reoptimize
sinput int          InpPopSize          = 50;           //Population size
sinput int          NumberFFlaunches    = 10000;        //Number of runs in the history during optimization
sinput int          Spread              = 10;           //Spread

MqlTick Tick;
CTrade  Trade;

C_iStochastic IStoch;

double Set        [];
double Range_Min  [];
double Range_Step [];
double Range_Max  [];

double TickSize = 0.0;
```

When initializing the EA in the OnInit function, we will set the size of the parameter arrays in accordance with the number of parameters being optimized: Set - a set of parameters, Range\_Min - minimum parameter values (starting values), Range\_Step - parameter step and Range\_Max - maximum parameter values. Extract the corresponding values from the string parameters and assign them to arrays.

```
//——————————————————————————————————————————————————————————————————————————————
int OnInit ()
{
  TickSize = SymbolInfoDouble (_Symbol, SYMBOL_TRADE_TICK_SIZE);

  ArrayResize (Set,        2);
  ArrayResize (Range_Min,  2);
  ArrayResize (Range_Step, 2);
  ArrayResize (Range_Max,  2);

  string result [];
  if (StringSplit (InpKPeriod_P, StringGetCharacter ("|", 0), result) != 4) return INIT_FAILED;

  Set        [0] = (double)StringToInteger (result [0]);
  Range_Min  [0] = (double)StringToInteger (result [1]);
  Range_Step [0] = (double)StringToInteger (result [2]);
  Range_Max  [0] = (double)StringToInteger (result [3]);

  if (StringSplit (InpUpperLevel_P, StringGetCharacter ("|", 0), result) != 4) return INIT_FAILED;

  Set        [1] = (double)StringToInteger (result [0]);
  Range_Min  [1] = (double)StringToInteger (result [1]);
  Range_Step [1] = (double)StringToInteger (result [2]);
  Range_Max  [1] = (double)StringToInteger (result [3]);

  IStoch.Init ((int)Set [0], 1, 3);

  //  set magicnumber to trade object
  Trade.SetExpertMagicNumber (InpMagicNumber);

  //---
  return (INIT_SUCCEEDED);
}
//——————————————————————————————————————————————————————————————————————————————
```

In the OnTick function of the EA code, we insert self-optimization call block - the "Optimize" function, which is the "manager" in the Figure 1 diagram and starts optimization. Use values from the Set array where external variables that need to be optimized were to be used.

```
//——————————————————————————————————————————————————————————————————————————————
void OnTick ()
{
  //----------------------------------------------------------------------------
  if (!IsNewBar ())
  {
    return;
  }

  //----------------------------------------------------------------------------
  if (SelfOptimization)
  {
    //--------------------------------------------------------------------------
    static datetime LastOptimizeTime = 0;

    datetime timeNow  = iTime (_Symbol, PERIOD_CURRENT, 0);
    datetime timeReop = iTime (_Symbol, PERIOD_CURRENT, InpBarsReOptimize);

    if (LastOptimizeTime <= timeReop)
    {
      LastOptimizeTime = timeNow;
      Print ("-------------------Start of optimization----------------------");

      Print ("Old set:");
      ArrayPrint (Set);

      Optimize (Set,
                Range_Min,
                Range_Step,
                Range_Max,
                InpBarsOptimize,
                InpPopSize,
                NumberFFlaunches,
                Spread * SymbolInfoDouble (_Symbol, SYMBOL_TRADE_TICK_SIZE));

      Print ("New set:");
      ArrayPrint (Set);

      IStoch.Init ((int)Set [0], 1, 3);
    }
  }

  //----------------------------------------------------------------------------
  if (!SymbolInfoTick (_Symbol, Tick))
  {
    Print ("Failed to get current symbol tick"); return;
  }

  //data preparation------------------------------------------------------------
  MqlRates rates [];
  int dataCount = CopyRates (_Symbol, PERIOD_CURRENT, 0, (int)Set [0] + 1 + 3 + 1, rates);

  if (dataCount == -1)
  {
    Print ("Data get error");
    return;
  }

  double hi [];
  double lo [];
  double cl [];

  ArrayResize (hi, dataCount);
  ArrayResize (lo, dataCount);
  ArrayResize (cl, dataCount);

  for (int i = 0; i < dataCount; i++)
  {
    hi [i] = rates [i].high;
    lo [i] = rates [i].low;
    cl [i] = rates [i].close;
  }

  int calc = IStoch.Calculate (dataCount, 0, hi, lo, cl);
  if (calc <= 0) return;

  double buff0 = IStoch.ExtMainBuffer [ArraySize (IStoch.ExtMainBuffer) - 2];
  double buff1 = IStoch.ExtMainBuffer [ArraySize (IStoch.ExtMainBuffer) - 3];

  //----------------------------------------------------------------------------
  // count open positions
  int cntBuy, cntSell;
  if (!CountOpenPositions (cntBuy, cntSell))
  {
    Print ("Failed to count open positions");
    return;
  }

  //----------------------------------------------------------------------------
  // check for buy
  if (cntBuy == 0 && buff1 <= (100 - (int)Set [1]) && buff0 > (100 - (int)Set [1]))
  {
    ClosePositions (2);

    double sl = NP (Tick.bid - InpStopLoss   * TickSize);
    double tp = NP (Tick.bid + InpTakeProfit * TickSize);

    Trade.PositionOpen (_Symbol, ORDER_TYPE_BUY, InpLotSize, Tick.ask, sl, tp, "Stochastic EA");
  }

  //----------------------------------------------------------------------------
  // check for sell
  if (cntSell == 0 && buff1 >= (int)Set [1] && buff0 < (int)Set [1])
  {
    ClosePositions (1);

    double sl = NP (Tick.ask + InpStopLoss   * TickSize);
    double tp = NP (Tick.ask - InpTakeProfit * TickSize);

    Trade.PositionOpen (_Symbol, ORDER_TYPE_SELL, InpLotSize, Tick.bid, sl, tp, "Stochastic EA");
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The Optimize function performs the same actions that are usually performed in optimization algorithm testing scripts in the "Population optimization algorithms" article series:

1\. Initialization of the optimization algorithm.

2.1. Population preparation.

2.2. Obtaining a set of parameters from the optimization algorithm.

2.3. Calculation of the fitness function with parameters passed to it.

2.4. Updating the best solution.

2.5. Obtaining the best solution from the algorithm.

```
//——————————————————————————————————————————————————————————————————————————————
void Optimize (double      &set        [],
               double      &range_min  [],
               double      &range_step [],
               double      &range_max  [],
               const int    inpBarsOptimize,
               const int    inpPopSize,
               const int    numberFFlaunches,
               const double spread)
{
  //----------------------------------------------------------------------------
  double parametersSet [];
  ArrayResize(parametersSet, ArraySize(set));

  //----------------------------------------------------------------------------
  int epochCount = numberFFlaunches / inpPopSize;

  Init(inpPopSize, range_min, range_max, range_step);

  // Optimization-------------------------------------------------------------
  for (int epochCNT = 1; epochCNT <= epochCount && !IsStopped (); epochCNT++)
  {
    Preparation ();

    for (int set = 0; set < inpPopSize; set++)
    {
      GetVariantCalc (parametersSet, set);
      SetFitness     (VirtualStrategy (parametersSet, inpBarsOptimize, spread), set);
    }

    Revision ();
  }

  Print ("Fitness: ", GetFitness (0));
  GetVariant (parametersSet, 0);
  ArrayCopy (set, parametersSet, 0, 0, WHOLE_ARRAY);
}
//——————————————————————————————————————————————————————————————————————————————
```

The VirtualStrategy function performs strategy testing on historical data (in the diagram in Figure 1, this is "EA Virt"). It takes the "set" array of parameters, "barsOptimize" - the number of bars to optimize and the "spread" value.

First, the data is prepared. Historical data is loaded into the "rates" array. Then the arrays "hi", "lo" and "cl", required for calculating Stochastic, are created.

Next, the Stochastic indicator is initialized and calculated based on historical data. If the calculation fails, the function returns the "-DBL\_MAX" value (the worst possible value of the fitness function).

This is followed by testing the strategy on historical data with its logic fully corresponding to the EA main code. The "deals" object is created to store deals. Then there is a run through the historical data, where the conditions for opening and closing positions are checked for each bar based on the indicator value and the "upLevel" and "dnLevel" levels. If the conditions are met, the position is opened or closed.

Upon completing the run through historical data, the function checks the number of completed trades. If there were no trades, the function returns the "-DBL\_MAX" value. Otherwise, the function returns the final balance.

The VirtualStrategy return value is the fitness function value. In this case, this is the value of the final balance in points (as was said earlier, the fitness function can be a balance, a profit factor, or any other indicator of the result quality of trading operations based on historical data).

It is important to note that the virtual strategy should match the EA's strategy as closely as possible. In this example, trading is carried out at opening prices, which corresponds to the control of the bar opening in the main EA. If the trading strategy logic is executed on every tick, then the user needs to take care of downloading the tick history during virtual testing and adjust the VirtualStrategy function accordingly.

```
//——————————————————————————————————————————————————————————————————————————————
double VirtualStrategy (double &set [], int barsOptimize, double spread)
{
  //data preparation------------------------------------------------------------
  MqlRates rates [];
  int dataCount = CopyRates(_Symbol, PERIOD_CURRENT, 0, barsOptimize + 1, rates);

  if (dataCount == -1)
  {
    Print ("Data get error");
    return -DBL_MAX;
  }

  double hi [];
  double lo [];
  double cl [];

  ArrayResize (hi, dataCount);
  ArrayResize (lo, dataCount);
  ArrayResize (cl, dataCount);

  for (int i = 0; i < dataCount; i++)
  {
    hi [i] = rates [i].high;
    lo [i] = rates [i].low;
    cl [i] = rates [i].close;
  }

  C_iStochastic iStoch;
  iStoch.Init ((int)set [0], 1, 3);

  int calc = iStoch.Calculate (dataCount, 0, hi, lo, cl);
  if (calc <= 0) return -DBL_MAX;

  //============================================================================
  //test of strategy on history-------------------------------------------------
  S_Deals deals;

  double iStMain0 = 0.0;
  double iStMain1 = 0.0;
  double upLevel  = set [1];
  double dnLevel  = 100.0 - set [1];
  double balance  = 0.0;

  //running through history-----------------------------------------------------
  for (int i = 2; i < dataCount; i++)
  {
    if (i >= dataCount)
    {
      deals.ClosPos (-1, rates [i].open, spread);
      deals.ClosPos (1, rates [i].open, spread);
      break;
    }

    iStMain0 = iStoch.ExtMainBuffer [i - 1];
    iStMain1 = iStoch.ExtMainBuffer [i - 2];

    if (iStMain0 == 0.0 || iStMain1 == 0.0) continue;

    //buy-------------------------------
    if (iStMain1 <= dnLevel && dnLevel < iStMain0)
    {
      deals.ClosPos (-1, rates [i].open, spread);

      if (deals.GetBuys () == 0) deals.OpenPos (1, rates [i].open, spread);
    }

    //sell------------------------------
    if (iStMain1 >= upLevel && upLevel > iStMain0)
    {
      deals.ClosPos (1, rates [i].open, spread);

      if (deals.GetSels () == 0) deals.OpenPos (-1, rates [i].open, spread);
    }
  }
  //----------------------------------------------------------------------------

  if (deals.histSelsCNT + deals.histBuysCNT <= 0) return -DBL_MAX;
  return deals.balance;
}
//——————————————————————————————————————————————————————————————————————————————
```

If we want to use the optimization algorithm from the "Population optimization algorithms" series (the "Evolution of Social Groups", or ESG, algorithm is provided as an example in the article archive), then we need to specify the path to the algorithm in the EA:

```
#include "AO_ESG.mqh"
```

In the Optimize function, declare the ESG algorithm object and configure the boundary values of the optimized parameters. Then, the Optimize function would look like this for using ESG:

```
//——————————————————————————————————————————————————————————————————————————————
void Optimize (double      &set        [],
               double      &range_min  [],
               double      &range_step [],
               double      &range_max  [],
               const int    inpBarsOptimize,
               const int    inpPopSize,
               const int    numberFFlaunches,
               const double spread)
{
  //----------------------------------------------------------------------------
  int epochCount = numberFFlaunches / inpPopSize;

  C_AO_ESG AO;

  int    Population_P     = 200;   //Population size
  int    Groups_P         = 100;   //Number of groups
  double GroupRadius_P    = 0.1;   //Group radius
  double ExpansionRatio_P = 2.0;   //Expansion ratio
  double Power_P          = 10.0;  //Power

  AO.Init (ArraySize (set), Population_P, Groups_P, GroupRadius_P, ExpansionRatio_P, Power_P);

  for (int i = 0; i < ArraySize (set); i++)
  {
    AO.rangeMin  [i] = range_min  [i];
    AO.rangeStep [i] = range_step [i];
    AO.rangeMax  [i] = range_max  [i];
  }

  // Optimization-------------------------------------------------------------
  for (int epochCNT = 1; epochCNT <= epochCount && !IsStopped (); epochCNT++)
  {
    AO.Moving ();

    for (int set = 0; set < ArraySize (AO.a); set++)
    {
      AO.a [set].f = VirtualStrategy (AO.a [set].c, inpBarsOptimize, spread);
    }

    AO.Revision ();
  }

  Print ("Fitness: ", AO.fB);
  ArrayCopy (set, AO.cB, 0, 0, WHOLE_ARRAY);
}
//——————————————————————————————————————————————————————————————————————————————
```

The search algorithm strategies presented in my series of optimization articles provide a simple and clear approach for analyzing and comparing them with each other, in their purest form - "as is". They do not include methods speeding up the search, such as eliminating duplicates and some other techniques, and require more iterations and time.

### 5\. Testing functionality

Let's test our self-optimizing EA based on the Stochastic indicator for a period of one year with the parameters shown below in the screenshot, first in the "false" mode of the SelfOptimization parameter, i.e. without self-optimization.

![photo_](https://c.mql5.com/2/69/photo_2024-02-05_14-22-07.jpg)

Figure 2. EA settings

![OriginalTest](https://c.mql5.com/2/69/OriginalTest.png)

Figure 3. Results with self-optimization disabled

![SelfOpt](https://c.mql5.com/2/69/SelfOpt__2.png)

Figure 4. Results with self-optimization enabled

### Summary

In this article, we looked at arranging self-optimization in an EA. This method is very simple and requires minimal intervention in the EA source code. For each specific strategy, it is recommended to conduct a series of experiments to determine the optimal lengths of historical segments, on which optimization and trading take place. These values are individual and strategy dependent.

It is important to understand that optimization cannot lead to positive results if the strategy itself does not have profitable sets. It is impossible to extract gold from sand if there is no gold there in the first place. Optimization is a useful tool for improving strategy performance, but it cannot create profitable sets where there are none. Therefore, you should first develop a strategy that has the potential for profit, and then use optimization to improve it.

The advantages of this approach are the ability to test a strategy on historical data using walk-forward testing and find suitable optimization criteria that correspond to a specific strategy. Walk-forward testing allows us to evaluate the efficiency of a strategy on historical data, taking into account changes in market conditions over time. This helps avoid over-optimization, when a strategy works well only over a certain period of history, but cannot be successfully applied in real time. Thus, walk-forward testing provides a more reliable assessment of strategy performance.

Walk-forward testing (WFT) is a technique for assessing and testing trading strategies in financial markets. It is used to determine the efficiency and sustainability of trading strategies based on historical data and their ability to provide profitability in the future.

The basic idea of WFT is to divide the available data into several periods: a historical period used to develop and tune a strategy (training period), and subsequent periods used to evaluate and test the strategy (test periods). The process is repeated several times. Each time the training period is shifted forward by one step, and the test period is also shifted forward. Thus, the strategy is tested over various timeframes to ensure that it is able to adapt to changing market conditions.

Walk-forward testing is a more realistic way to evaluate strategies because it takes into account changes in market conditions over time. It also helps avoid overtraining a strategy on historical data and provides a more accurate understanding of its performance in the real world.

In the archive attached to this article, you will find examples demonstrating how to connect optimization algorithms to the EA. You will be able to study the code and apply it to your specific strategy to achieve optimal results.

The example given is not intended for trading on real accounts, since there are no necessary checks, and is intended only to demonstrate the possibility of self-optimization.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14183](https://www.mql5.com/ru/articles/14183)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14183.zip "Download all attachments in the single ZIP archive")

[Self-Optimization.zip](https://www.mql5.com/en/articles/download/14183/self-optimization.zip "Download Self-Optimization.zip")(14.18 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Neuroboids Optimization Algorithm (NOA)](https://www.mql5.com/en/articles/16992)
- [Successful Restaurateur Algorithm (SRA)](https://www.mql5.com/en/articles/17380)
- [Billiards Optimization Algorithm (BOA)](https://www.mql5.com/en/articles/17325)
- [Chaos Game Optimization (CGO)](https://www.mql5.com/en/articles/17047)
- [Blood inheritance optimization (BIO)](https://www.mql5.com/en/articles/17246)
- [Circle Search Algorithm (CSA)](https://www.mql5.com/en/articles/17143)
- [Royal Flush Optimization (RFO)](https://www.mql5.com/en/articles/17063)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/468266)**
(24)


![Andrey Dik](https://c.mql5.com/avatar/2024/8/66be0662-3c24.png)

**[Andrey Dik](https://www.mql5.com/en/users/joo)**
\|
2 Mar 2024 at 18:30

**Rorschach [#](https://www.mql5.com/ru/forum/462534/page2#comment_52589771):**

It could be this.

The question was on Saber's cart, it's not there now.

Well, if there is an interest in the topic of the influence of the quality of HCS on search engine optimisation algorithms, then an article on this topic will be useful. I myself am interested in this question, the answer to which is not obvious.


![Rorschach](https://c.mql5.com/avatar/2022/3/6244E941-EF6F.jpg)

**[Rorschach](https://www.mql5.com/en/users/rorschach)**
\|
2 Mar 2024 at 18:51

**Andrey Dik [#](https://www.mql5.com/ru/forum/462534/page2#comment_52589828):**

Well, if there is an interest in the topic of the influence of the quality of HCS on search engine optimisation algorithms, then an article on this topic will be useful. I myself am interested in this question, the answer to which is not obvious.

[According to the article on hubra, stratification improves results.](https://www.mql5.com/go?link=https://habr.com/ru/articles/496750/ "https://habr.com/ru/articles/496750/")

[It is definitely not worth using the regular algorithm.](https://www.mql5.com/ru/forum/86386/page1720#comment_15971654)

There was also a case in history where a bad GSC caused a satellite to fly off in the wrong place.

So it should be an interesting topic.

![Andrey Dik](https://c.mql5.com/avatar/2024/8/66be0662-3c24.png)

**[Andrey Dik](https://www.mql5.com/en/users/joo)**
\|
2 Mar 2024 at 19:15

**Rorschach [#](https://www.mql5.com/ru/forum/462534/page3#comment_52590010):**

[According to an article on hubra, stratification improves the outcome.](https://www.mql5.com/go?link=https://habr.com/ru/articles/496750/ "https://habr.com/ru/articles/496750/")

[Definitely not worth using the regular algorithm.](https://www.mql5.com/ru/forum/86386/page1720#comment_15971654)

There was also a case in history when a satellite flew off to the wrong place because of a bad GSC.

So it should be an interesting topic.

On the given links did not find discussion of the impact of the quality of the GSF on the AO. But, the topic does not become less interesting.

![Rorschach](https://c.mql5.com/avatar/2022/3/6244E941-EF6F.jpg)

**[Rorschach](https://www.mql5.com/en/users/rorschach)**
\|
2 Mar 2024 at 21:06

**Andrey Dik [#](https://www.mql5.com/ru/forum/462534/page3#comment_52590150):**

I did not find any discussion of the impact of the quality of the HSC on the AO in the links provided. But, the topic does not become less interesting.

Yes, the influence on AO is not investigated, the links only hint that the quality of the BOP is important.

![Andrey Dik](https://c.mql5.com/avatar/2024/8/66be0662-3c24.png)

**[Andrey Dik](https://www.mql5.com/en/users/joo)**
\|
22 Mar 2024 at 17:42

**Rorschach [#](https://www.mql5.com/ru/forum/462534/page3#comment_52590617):**

Yes, the effect on AO has not been researched, the references only hint that the quality of the HGF is important.

[There is an article about the influence of the quality of the DSP on the results of optimisation algorithms.](https://www.mql5.com/ru/articles/14413)

![Balancing risk when trading multiple instruments simultaneously](https://c.mql5.com/2/69/Balancing_risk_when_trading_several_trading_instruments_simultaneously______LOGO.png)[Balancing risk when trading multiple instruments simultaneously](https://www.mql5.com/en/articles/14163)

This article will allow a beginner to write an implementation of a script from scratch for balancing risks when trading multiple instruments simultaneously. Besides, it may give experienced users new ideas for implementing their solutions in relation to the options proposed in this article.

![MQL5 Wizard Techniques you should know (Part 22): Conditional GANs](https://c.mql5.com/2/80/MQL5_Wizard_Techniques_you_should_know_Part_22____LOGO.png)[MQL5 Wizard Techniques you should know (Part 22): Conditional GANs](https://www.mql5.com/en/articles/15029)

Generative Adversarial Networks are a pairing of Neural Networks that train off of each other for more accurate results. We adopt the conditional type of these networks as we look to possible application in forecasting Financial time series within an Expert Signal Class.

![Gain An Edge Over Any Market (Part II): Forecasting Technical Indicators](https://c.mql5.com/2/80/Gain_An_Edge_Over_Any_Market_Part_II___LOGO.png)[Gain An Edge Over Any Market (Part II): Forecasting Technical Indicators](https://www.mql5.com/en/articles/14936)

Did you know that we can gain more accuracy forecasting certain technical indicators than predicting the underlying price of a traded symbol? Join us to explore how to leverage this insight for better trading strategies.

![Neural networks made easy (Part 73): AutoBots for predicting price movements](https://c.mql5.com/2/64/Neural_networks_are_easy_jPart_73u__AutoBots_for_predicting_price_movement_LOGO.png)[Neural networks made easy (Part 73): AutoBots for predicting price movements](https://www.mql5.com/en/articles/14095)

We continue to discuss algorithms for training trajectory prediction models. In this article, we will get acquainted with a method called "AutoBots".

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/14183&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049572430118235609)

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