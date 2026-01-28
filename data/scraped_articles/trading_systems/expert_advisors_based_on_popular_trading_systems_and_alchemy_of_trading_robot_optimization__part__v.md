---
title: Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part  V)
url: https://www.mql5.com/en/articles/1525
categories: Trading Systems
relevance_score: 12
scraped_at: 2026-01-22T17:15:46.018528
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/1525&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049003514455237391)

MetaTrader 4 / Trading systems


### Introduction

In my previous articles from this cycle ( [1](https://www.mql5.com/en/articles/1516), [2](https://www.mql5.com/en/articles/1517), [3](https://www.mql5.com/en/articles/1521), [4](https://www.mql5.com/en/articles/1523)) I described simplest trading systems, the distinctive of which was working on only one frame. As a result, such a trading system has absolutely no reaction to the change of market trends in a more global time scale. This may result in losses in conditions of a changed market, such changes are not detected in a system of that kind. Actually, in live trading systems based on data obtained from a chart of only one timeframe can hardly be used. Usually at least two timeframes are used for a normal operation. A current trend is usually identified on a chart of a higher timeframe, while the point of market entering in the direction of this trend is calculated on a chart of a smaller timeframe. In my opinion, examples of simplest trading strategies described in previous articles are enough for a reader to learn designing such systems. So now let's discuss methods to improve such trading systems on the basis of the above described reasoning.

### Trading System using Two Timeframes

From the point of view of logics, there is no difference, on the basis of what trading system described in previous articles we will build a more complicated system. As for their initial essence, each simplest trading system can be presented in the following form:

For long positions:

![](https://c.mql5.com/2/16/1_3.png)

For short positions:

![](https://c.mql5.com/2/16/2_3.png)

In our trading system using two timeframes, these conditions for market entering will be defined on the basis of indicators calculated on a smaller timeframe. Trend direction will be identified on a higher timeframe. So, the algorithm containing these conditions will look like this:

For long positions:

![](https://c.mql5.com/2/16/3_2.png)

For short positions:

![](https://c.mql5.com/2/16/4_1.png)

In this case the Trend variable defines only the direction of a current trend on a higher timeframe and the additional condition for market entering limits trading actions of an Expert Advisor only to the direction of this global trend. From the point of view of program code, it makes no difference using what algorithm the current trend will be detected on a higher timeframe. So it is up to an EA writer to decide what algorithms to use both for calculating a market entering point on a smaller timeframe and detecting the current trend on a higher timeframe. Let's analyze the earlier described algorithm with the OsMA oscillator represented by the EA Exp\_5.mq4, to define the current trend let's use the moving J2JMA.mq4. In such a case the condition of trade defining will be very simple:

![](https://c.mql5.com/2/16/trend.png)

![](https://c.mql5.com/2/16/j2jma_2.gif)

So now let's add some code to the existing Exp\_5.mq4 including into it the above described logics. the ready code will look like this:

```
//For the EA operation Metatrader\EXPERTS\indicators folder must
//contain indicators 5c_OsMA.mq4 and J2JMA.mq4
//+==================================================================+
//|                                                       Exp_11.mq4 |
//|                             Copyright © 2008,   Nikolay Kositsin |
//|                              Khabarovsk,   farria@mail.redcom.ru |
//+==================================================================+
#property copyright "Copyright © 2008, Nikolay Kositsin"
#property link "farria@mail.redcom.ru"
//----+ +--------------------------------------------------------------------------+
//---- EA INPUT PARAMETERS FOR BUY TRADES
extern bool   Test_Up = true;//filter of trade calculations direction
extern double Money_Management_Up = 0.1;
//---- input parameters of the custom indicator J2JMA.mq4
extern int    TimeframeX_Up = 240;
extern int    Length1X_Up = 4;  // depth of the first smoothing
extern int    Phase1X_Up = 100; // parameter of the first smoothing
       //changing in the range -100 ... +100, influences the quality
       //of the transient process of averaging;
extern int    Length2X_Up = 4;  // depth of the second smoothing
extern int    Phase2X_Up = 100; // parameter of the second smoothing,
       //changing in the range -100 ... +100, influences the quality
       //of the transient process of averaging;
extern int    IPCX_Up = 0;/* Selecting prices on which the indicator will
be calculated (0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL,
6-WEIGHTED, 7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close.) */
//---- input parameters of the custom indicator 5c_OsMA.mq4
extern int    Timeframe_Up = 60;
extern double IndLevel_Up = 0; // breakout level of the indicator
extern int    FastEMA_Up = 12;  // quick EMA period
extern int    SlowEMA_Up = 26;  // slow EMA period
extern int    SignalSMA_Up = 9;  // signal SMA period
extern int    STOPLOSS_Up = 50;  // stop loss
extern int    TAKEPROFIT_Up = 100; // take profit
extern int    TRAILINGSTOP_Up = 0; // trailing stop
extern int    PriceLevel_Up =40; // difference between the current price and
                                         // the price of a pending order triggering
extern bool   ClosePos_Up = true; // forced position closing allowed
//----+ +--------------------------------------------------------------------------+
//---- EA INPUT PARAMETERS FOR SELL TRADES
extern bool   Test_Dn = true;//filter of trade calculations direction
extern double Money_Management_Dn = 0.1;
//---- input parameters of the custom indicator J2JMA.mq4
extern int    TimeframeX_Dn = 240;
extern int    Length1X_Dn = 4;  // smoothing depth
extern int    Phase1X_Dn = 100;  // parameter of the first smoothing
       //changing in the range -100 ... +100, influences the quality
       //of the transient process of averaging;
extern int    Length2X_Dn = 4;  // smoothing depth
extern int    Phase2X_Dn = 100; // parameter of the second smoothing
       //changing in the range -100 ... +100, influences the quality
       //of the transient process of averaging;
extern int    IPCX_Dn = 0;/* Selecting prices on which the indicator will
be calculated (0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL,
6-WEIGHTED, 7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close.) */
//---- input parameters of the custom indicator 5c_OsMA.mq4
extern int    Timeframe_Dn = 60;
extern double IndLevel_Dn = 0; // breakout level of the indicator
extern int    FastEMA_Dn = 12;  // quick EMA period
extern int    SlowEMA_Dn = 26;  // slow EMA period
extern int    SignalSMA_Dn = 9;  // signal SMA period
extern int    STOPLOSS_Dn = 50;  // stop loss
extern int    TAKEPROFIT_Dn = 100; // take profit
extern int    TRAILINGSTOP_Dn = 0; // trailing stop
extern int    PriceLevel_Dn = 40; // difference between the current price and
                                         // the price of a pending order triggering
extern bool   ClosePos_Dn = true; // forced position closing allowed
//----+ +--------------------------------------------------------------------------+
//---- Integer variables for the minimum of calculation bars
int MinBarX_Up, MinBar_Up, MinBarX_Dn, MinBar_Dn;
//+==================================================================+
//| TimeframeCheck() functions                                       |
//+==================================================================+
void TimeframeCheck(string Name, int Timeframe)
  {
//----+
   //---- Checking the correctness of Timeframe variable value
   if (Timeframe != 1)
    if (Timeframe != 5)
     if (Timeframe != 15)
      if (Timeframe != 30)
       if (Timeframe != 60)
        if (Timeframe != 240)
         if (Timeframe != 1440)
           Print(StringConcatenate("TimeframeCheck: Parameter ",Name,
                     " cannot ", "be equal to ", Timeframe, "!!!"));
//----+
  }
//+==================================================================+
//| Custom Expert functions                                          |
//+==================================================================+
#include <Lite_EXPERT1.mqh>
//+==================================================================+
//| Custom Expert initialization function                            |
//+==================================================================+
int init()
  {
//---- Checking the correctness of Timeframe_Up variable value
   TimeframeCheck("Timeframe_Up", Timeframe_Up);
   //---- Checking the correctness of TimeframeX_Up variable value
   TimeframeCheck("TimeframeX_Up", TimeframeX_Up);
//---- Checking the correctness of Timeframe_Dn variable value
   TimeframeCheck("Timeframe_Dn", Timeframe_Dn);
   //---- Checking the correctness of Timeframe_Dn variable value
   TimeframeCheck("TimeframeX_Dn", TimeframeX_Dn);
//---- Initialization of variables
   MinBar_Up  = 3 + MathMax(FastEMA_Up, SlowEMA_Up) + SignalSMA_Up;
   MinBarX_Up  = 3 + 30 + 30;
   MinBar_Dn  = 3 + MathMax(FastEMA_Dn, SlowEMA_Dn) + SignalSMA_Dn;
   MinBarX_Dn  = 3 + 30 + 30;
//---- end of initialization
   return(0);
  }
//+==================================================================+
//| expert deinitialization function                                 |
//+==================================================================+
int deinit()
  {
//----+

    //---- End of the EA deinitialization
    return(0);
//----+
  }
//+==================================================================+
//| Custom Expert iteration function                                 |
//+==================================================================+
int start()
  {
   //----+ Declaring local variables
   double J2JMA1, J2JMA2, Osc1, Osc2;
   //----+ Declaring static variables
   //----+ +---------------------------------------------------------------+
   static double TrendX_Up, TrendX_Dn;
   static datetime StopTime_Up, StopTime_Dn;
   static int LastBars_Up, LastBarsX_Up, LastBarsX_Dn, LastBars_Dn;
   static bool BUY_Sign, BUY_Stop, SELL_Sign, SELL_Stop;
   //----+ +---------------------------------------------------------------+
   //----++ CODE FOR LONG POSITIONS                                        |
   //----+ +---------------------------------------------------------------+
   if (Test_Up)
    {
      int IBARS_Up = iBars(NULL, Timeframe_Up);
      int IBARSX_Up = iBars(NULL, TimeframeX_Up);

      if (IBARS_Up >= MinBar_Up && IBARSX_Up >= MinBarX_Up)
       {
         //----+ +----------------------+
         //----+ DEFINING TREND         |
         //----+ +----------------------+
         if (LastBarsX_Up != IBARSX_Up)
          {
           //----+ Initialization of variables
           LastBarsX_Up = IBARSX_Up;
           BUY_Stop = false;

           //----+ calculating indicator values for J2JMA
           J2JMA1 = iCustom(NULL, TimeframeX_Up,
                                "J2JMA", Length1X_Up, Length2X_Up,
                                             Phase1X_Up, Phase2X_Up,
                                                     0, IPCX_Up, 0, 1);
           //---
           J2JMA2 = iCustom(NULL, TimeframeX_Up,
                                "J2JMA", Length1X_Up, Length2X_Up,
                                             Phase1X_Up, Phase2X_Up,
                                                     0, IPCX_Up, 0, 2);

           //----+ defining trend
           TrendX_Up = J2JMA1 - J2JMA2;
           //----+ defining a signal for closing trades
           if (TrendX_Up < 0)
                      BUY_Stop = true;
          }

         //----+ +----------------------------------------+
         //----+ DEFINING SIGNAL FOR MARKET ENTERING      |
         //----+ +----------------------------------------+
         if (LastBars_Up != IBARS_Up)
          {
           //----+ Initialization of variables
           BUY_Sign = false;
           LastBars_Up = IBARS_Up;
           StopTime_Up = iTime(NULL, Timeframe_Up, 0)
                                          + 60 * Timeframe_Up;
           //----+ calculating indicator values
           Osc1 = iCustom(NULL, Timeframe_Up,
                         "5c_OsMA", FastEMA_Up, SlowEMA_Up,
                                               SignalSMA_Up, 5, 1);
           //---
           Osc2 = iCustom(NULL, Timeframe_Up,
                         "5c_OsMA", FastEMA_Up, SlowEMA_Up,
                                               SignalSMA_Up, 5, 2);

           //----+ defining signals for trades
           if (TrendX_Up > 0)
            if (Osc2 < IndLevel_Up)
             if (Osc1 > IndLevel_Up)
                        BUY_Sign = true;
          }

         //----+ +-------------------+
         //----+ EXECUTION OF TRADES |
         //----+ +-------------------+
         if (!OpenBuyLimitOrder1(BUY_Sign, 1,
              Money_Management_Up, STOPLOSS_Up, TAKEPROFIT_Up,
                                            PriceLevel_Up, StopTime_Up))
                                                                 return(-1);
         if (ClosePos_Up)
                if (!CloseOrder1(BUY_Stop, 1))
                                        return(-1);

         if (!Make_TreilingStop(1, TRAILINGSTOP_Up))
                                              return(-1);
        }
     }
   //----+ +---------------------------------------------------------------+
   //----++ CODE FOR SHORT POSITIONS                                       |
   //----+ +---------------------------------------------------------------+
   if (Test_Dn)
    {
      int IBARS_Dn = iBars(NULL, Timeframe_Dn);
      int IBARSX_Dn = iBars(NULL, TimeframeX_Dn);

      if (IBARS_Dn >= MinBar_Dn && IBARSX_Dn >= MinBarX_Dn)
       {
         //----+ +----------------------+
         //----+ DEFINING TREND         |
         //----+ +----------------------+
         if (LastBarsX_Dn != IBARSX_Dn)
          {
           //--- Initialization of variables
           LastBarsX_Dn = IBARSX_Dn;
           SELL_Stop = false;

           //----+ calculating indicator values for J2JMA
           J2JMA1 = iCustom(NULL, TimeframeX_Dn,
                                "J2JMA", Length1X_Dn, Length2X_Dn,
                                             Phase1X_Dn, Phase2X_Dn,
                                                     0, IPCX_Dn, 0, 1);
           //---
           J2JMA2 = iCustom(NULL, TimeframeX_Dn,
                                "J2JMA", Length1X_Dn, Length2X_Dn,
                                             Phase1X_Dn, Phase2X_Dn,
                                                     0, IPCX_Dn, 0, 2);

           //----+ defining trend
           TrendX_Dn = J2JMA1 - J2JMA2;
           //----+ defining a signal for closing trades
           if (TrendX_Dn > 0)
                      SELL_Stop = true;
          }

         //----+ +----------------------------------------+
         //----+ DEFINING SIGNAL FOR MARKET ENTERING      |
         //----+ +----------------------------------------+
         if (LastBars_Dn != IBARS_Dn)
          {
           //----+ Initialization of variables
           SELL_Sign = false;
           LastBars_Dn = IBARS_Dn;
           StopTime_Dn = iTime(NULL, Timeframe_Dn, 0)
                                          + 60 * Timeframe_Dn;
           //----+ calculating indicator values
           Osc1 = iCustom(NULL, Timeframe_Dn,
                         "5c_OsMA", FastEMA_Dn, SlowEMA_Dn,
                                               SignalSMA_Dn, 5, 1);
           //---
           Osc2 = iCustom(NULL, Timeframe_Dn,
                         "5c_OsMA", FastEMA_Dn, SlowEMA_Dn,
                                               SignalSMA_Dn, 5, 2);

           //----+ defining signals for trades
           if (TrendX_Dn < 0)
            if (Osc2 > IndLevel_Dn)
             if (Osc1 < IndLevel_Dn)
                        SELL_Sign = true;
          }

         //----+ +-------------------+
         //----+ EXECUTION OF TRADES |
         //----+ +-------------------+
          if (!OpenSellLimitOrder1(SELL_Sign, 2,
              Money_Management_Dn, STOPLOSS_Dn, TAKEPROFIT_Dn,
                                            PriceLevel_Dn, StopTime_Dn))
                                                                 return(-1);
          if (ClosePos_Dn)
                if (!CloseOrder1(SELL_Stop, 2))
                                        return(-1);

          if (!Make_TreilingStop(2, TRAILINGSTOP_Dn))
                                                  return(-1);
        }
     }
   //----+ +---------------------------------------------------------------+
//----+

    return(0);
  }
//+------------------------------------------------------------------+
```

Visually this code is twice larger than the initial Exp\_5mq4 code, though the idea was seemingly not so large! Now let's discuss the result. Again I will analyze only the EA part for long positions, for short ones it is analogous. The additional source code for getting necessary values of the J2JMA indicator looks like this:

```
//----+ calculating indicator values for J2JMA
           J2JMA1 = iCustom(NULL, TimeframeX_Up, "J2JMA", Length1X_Up, Phase1X_Up, Length2X_Up, Phase2X_Up, 0, IPCX_Up, 0, 1);
           //---
           J2JMA2 = iCustom(NULL, TimeframeX_Up, "J2JMA", Length1X_Up, Phase1X_Up, Length2X_Up, Phase2X_Up, 0, IPCX_Up, 0, 2);
//----+ defining trend
           Trend_Up = J2JMA1 - J2JMA2;
```

On this ground the EA head part now contains the declaration of six new external variables corresponding to J2JMA indicator call:

```
extern int    TimeframeX_Up = 240;
extern int    Length1X_Up = 4;  // depth of the first smoothing
extern int    Phase1X_Up = 100; // parameter of the first smoothing
       //changing in the range -100 ... +100, influences the quality
       //of the transient process of averaging;
extern int    Length2X_Up = 4;  // depth of the second smoothing
extern int    Phase2X_Up = 100; // parameter of the second smoothing,
       //changing in the range -100 ... +100, influences the quality
       //of the transient process of averaging;
extern int    IPCX_Up = 0;/* Selecting prices on which the indicator will
be calculated (0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL,
6-WEIGHTED, 7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close.) */
```

A new analogous variable MinBarX\_Up is added to the line of declaration of global variables for the minimum of calculation bars, which follows the EA external variables:

```
//---- Integer variables for the minimum of calculation bars
int MinBarX_Up, MinBar_Up, MinBarX_Dn, MinBar_Dn;
```

In the EA initialization block additional check of the new external variable TimeframeХ\_Up correctness is made:

```
//---- Checking the correctness of TimeframeX_Up variable value
   TimeframeCheck("TimeframeХ_Up", TimeframeX_Up);
```

В этом же блоке делают инициализацию переменной MinBarX\_Up:

```
MinBarX_Up  = 3 + 30 + 30;
```

Further code modifications are performed in the start() function block of the EA. Two new variables are added in the line of local variables declaration: J2JMA1 and J2JMA2:

```
//----+ Declaring local variables
   double J2JMA1, J2JMA2, Osc1, Osc2;
```

The Trend\_Up variable is declared as a static variable because it is initialized only once at bar changing, its value is used in further tick of the start() function:

```
static double TrendX_Up, TrendX_Dn;
```

By analogy the variable LastBarsX\_Up is declared as static:

```
static int LastBars_Up, LastBarsX_Up, LastBarsX_Dn, LastBars_Dn;
```

In the code for long positions the check of sufficiency for calculations becomes more complicated:

```
if (Test_Up)
    {
      int IBARS_Up = iBars(NULL, Timeframe_Up);
      int IBARSX_Up = iBars(NULL, TimeframeX_Up);

      if (IBARS_Up >= MinBar_Up && IBARSX_Up >= MinBarX_Up)
       {
         // CODE FOR LONG POSITIONS
       }
     }
```

and a new block is added:

```
         //----+ +----------------------+
         //----+ DEFINING TREND         |
         //----+ +----------------------+
         if (LastBarsX_Up != IBARSX_Up)
          {
           //----+ Initialization of variables
           LastBarsX_Up = IBARSX_Up;
           BUY_Stop = false;

           //----+ calculating indicator values for J2JMA
           J2JMA1 = iCustom(NULL, TimeframeX_Up,
                                "J2JMA", Length1X_Up, Phase1X_Up,
                                            Length2X_Up, Phase2X_Up,
                                                     0, IPCX_Up, 0, 1);
           //---
           J2JMA2 = iCustom(NULL, TimeframeX_Up,
                                "J2JMA", Length1X_Up, Phase1X_Up,
                                            Length2X_Up, Phase2X_Up,
                                                     0, IPCX_Up, 0, 2);

           //----+ defining trend
           TrendX_Up = J2JMA1 - J2JMA2;
           //----+ defining a signal for closing trades
           if (Trend_Up < 0)
                      BUY_Stop = true;
          }
```

In this block the necessary to us variable Trend\_Up is initialized, besides signals for the forced closing of open positions are defined here (initialization of the BUY\_Stop variable). Generally, in the initial Exp\_5.mq4 the last variable was initialized in the block "DEFINING SIGNALS FOR MARKET ENTERING", but it is more logical in the new EA to place this initialization in the block "DEFINING TREND" and to change the algorithm of its initialization.

And the most important thing is a small change of signal defining algorithm in the block "DEFINING SIGNALS FOR MARKET ENTERING":

```
        //----+ defining signals for trades
           if (TrendX_Up > 0)
            if (Osc2 < IndLevel_Up)
             if (Osc1 > IndLevel_Up)
                        BUY_Sign = true;
```

After all modifications this algorithm takes into account the direction of a current trend with the help of the variable Trend\_Up.

Now about some details of the EA optimization. Naturally, the EA should be optimized separately either for only long or short positions, and even in this case there are too many external variables for optimization. Probably, it is not reasonable to optimize all these variable at the same time. The more so - the genetic algorithm of optimization will not optimize more than eight variables! The most suitable solution in this case is fixing values of some variables and optimizing only the part remaining unfixed - the most urgent variables. And after the optimization select the most suiting variant and try to optimize remaining parameters.

For example, for long positions this may look like this:

![](https://c.mql5.com/2/16/5.png)

A file with these settings for the tester Exp\_11.ini is in TESTER.zip archive. Here we don't need to optimize Money\_Management\_Up, as well as TimeframeX\_Up. AS for the TimeframeX\_Up variable, it should be noted that initially its value must be larger than that of the variable Timeframe\_Up. Values of Length1X\_Up can be changed in quite a large range, values of Phase1X\_Up in the range from -100 to 100. Parameters Length2X\_Up, Phase2X\_Up and IPCX\_Up should be better fixed at the first optimization, the same is about the IndLevel\_Up parameter described in my [previous article](https://www.mql5.com/en/articles/1523 "Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Cont.)") describing Exp\_5.mq4. For FastEMA\_Up and SlowEMA\_Up parameters lower values of parameter changing shouldn't be too small. Of course, they can show amazing results, but will their be any sense in such results? The reasonability of using a trailing stop should be also checked after optimization. But the forced position closing by the logical variable ClosePos\_Up should be applied always at trend changing. To its value should be better fixed equal to 'true'.

During optimization chart period in the Strategy tester should be equal to the value of the variable Timeframe\_Up or Timeframe\_Dn (depending on trading direction during optimization) and in the final testing or in operation on an account the chart period should be set equal to the smallest of these values. There is one more important detail. This Expert Advisor uses at least two timeframes, so be attentive when downloading history data for optimizations, testing and operation on an account, especially if you use several accounts opened at different dealers.

In the [fourth article](https://www.mql5.com/en/articles/1523 "Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Cont.)") I described exporting optimization results for further statistical analysis in Microsoft Excel. In my opinion, the EA offered in this article suits best for such procedures. If anyone wants to try this, I modified the EA code with account for recommendations of this article (Exp\_11\_2.mq4). The code is attached to the article.

### One More Example of an EA Using for Calculations Data of Two Charts of Different Timeframes

I suppose one example of an EA based on this idea is not enough for this article, so I will include one more Expert Advisor constructed according to this principle. As the basis I will use my first EA Exp\_1.mq4 from my [first article](https://www.mql5.com/en/articles/1516 "Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization"). The code part responsible for defining conditions for market entering and manages positions is ready. Now we need to define the active market trend for a larger timeframe. In this Expert Advisor I use the indicator MAMA\_NK.mq4:

![](https://c.mql5.com/2/16/mama.gif)

The condition for defining a trend direction in this case is the difference of values of two movings on the first bar:

![](https://c.mql5.com/2/16/trend1.png)

![](https://c.mql5.com/2/16/mama_trend_2.gif)

Let's wright a cody by analogy, code of Exp\_11.mq4 is used as a template:

```
//+==================================================================+
//|                                                       Exp_12.mq4 |
//|                             Copyright © 2008,   Nikolay Kositsin |
//|                              Khabarovsk,   farria@mail.redcom.ru |
//+==================================================================+
#property copyright "Copyright © 2008, Nikolay Kositsin"
#property link "farria@mail.redcom.ru"
//----+ +---------------------------------------------------------------------------+
//---- EA INPUT PARAMETERS FOR BUY TRADES
extern bool   Test_Up = true;//filter of trade calculations direction
extern double Money_Management_Up = 0.1;
//----
extern int    TimeframeX_Up = 240;
extern double FastLimitX_Up = 0.5;
extern double SlowLimitX_Up = 0.05;
extern int    IPCX_Up = 9;/* Selecting prices, upon which the indicator will be calculated
(0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL, 6-WEIGHTED,
7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close, 15-Heiken Ashi Open0.) */
//---- + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
extern int    Timeframe_Up = 60;
extern int    Length_Up = 4;  // smoothing depth
extern int    Phase_Up = 100; // parameter changing in the range
          //-100 ... +100, influences the quality of a transient process;
extern int    IPC_Up = 0;/* Selecting prices, upon which the indicator will
be calculated (0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL,
6-WEIGHTED, 7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close.) */
extern int    STOPLOSS_Up = 50;  // stop loss
extern int    TAKEPROFIT_Up = 100; // take profit
extern bool   ClosePos_Up = true; // forced position closing is allowed
//----+ +---------------------------------------------------------------------------+
//---- EA INPUT PARAMETERS FOR SELL TRADES
extern bool   Test_Dn = true;//filter of trade calculations direction
extern double Money_Management_Dn = 0.1;
//----
extern int    TimeframeX_Dn = 60;
extern double FastLimitX_Dn = 0.5;
extern double SlowLimitX_Dn = 0.05;
extern int    IPCX_Dn = 9;/* Selecting prices, upon which the indicator will be calculated
(0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL, 6-WEIGHTED,
7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close, 15-Heiken Ashi Open0.) */
//---- + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
extern int    Timeframe_Dn = 60;
extern int    Length_Dn = 4;  // smoothing depth
extern int    Phase_Dn = 100; // parameter changing in the range
         // -100 ... +100, influences the quality of a transient process;
extern int    IPC_Dn = 0;/* Selecting prices, upon which the indicator will
be calculated (0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL,
6-WEIGHTED, 7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close.) */
extern int   STOPLOSS_Dn = 50;  // stop loss
extern int   TAKEPROFIT_Dn = 100; // take profit
extern bool   ClosePos_Dn = true; // forced position closing is allowed
//----+ +---------------------------------------------------------------------------+
//---- Integer variables for the minimum of counted bars
int MinBar_Up, MinBar_Dn, MinBarX_Up, MinBarX_Dn;
//+==================================================================+
//| Custom Expert functions                                          |
//+==================================================================+
#include <Lite_EXPERT1.mqh>
//+==================================================================+
//| TimeframeCheck() functions                                       |
//+==================================================================+
void TimeframeCheck(string Name, int Timeframe)
  {
//----+
   //---- Checking the correctness of Timeframe variable value
   if (Timeframe != 1)
    if (Timeframe != 5)
     if (Timeframe != 15)
      if (Timeframe != 30)
       if (Timeframe != 60)
        if (Timeframe != 240)
         if (Timeframe != 1440)
           Print(StringConcatenate("Parameter ",Name,
                     " cannot ", "be equal to ", Timeframe, "!!!"));
//----+
  }
//+==================================================================+
//| Custom Expert initialization function                            |
//+==================================================================+
int init()
  {
//---- Checking the correctness of Timeframe_Up variable value
   TimeframeCheck("TimeframeX_Up", TimeframeX_Up);
//---- Checking the correctness of Timeframe_Up variable value
   TimeframeCheck("Timeframe_Up", Timeframe_Up);
//---- Checking the correctness of Timeframe_Dn variable value
   TimeframeCheck("TimeframeX_Dn", TimeframeX_Dn);
//---- Checking the correctness of Timeframe_Dn variable value
   TimeframeCheck("Timeframe_Dn", Timeframe_Dn);
//---- Initialization of variables
   MinBarX_Up = 1 + 7;
   MinBar_Up = 4 + 39 + 30;
   MinBarX_Dn = 1 + 7;
   MinBar_Dn = 4 + 39 + 30;
//---- end of initialization
   return(0);
  }
//+==================================================================+
//| expert deinitialization function                                 |
//+==================================================================+
int deinit()
  {
//----+

    //---- End of the EA deinitialization
    return(0);
//----+
  }
//+==================================================================+
//| Custom Expert iteration function                                 |
//+==================================================================+
int start()
  {
   //----+ Declaration of local variables
   int    bar;
   double Mov[3], dMov12, dMov23, Mama1, Fama1;
   //----+ declaration of static variables
   static double TrendX_Up, TrendX_Dn;
   static int LastBars_Up, LastBars_Dn, LastBarsX_Up, LastBarsX_Dn;
   static bool BUY_Sign, BUY_Stop, SELL_Sign, SELL_Stop;

   //----+ +---------------------------------------------------------------+
   //----++ CODE FOR LONG POSITIONS                                        |
   //----+ +---------------------------------------------------------------+
   if (Test_Up)
    {
      int IBARS_Up = iBars(NULL, Timeframe_Up);
      int IBARSX_Up = iBars(NULL, TimeframeX_Up);

      if (IBARS_Up >= MinBar_Up && IBARSX_Up >= MinBarX_Up)
       {
         //----+ +----------------------+
         //----+ DEFINING TREND         |
         //----+ +----------------------+
         if (LastBarsX_Up != IBARSX_Up)
          {
           //----+ Initialization of variables
           LastBarsX_Up = IBARSX_Up;
           BUY_Stop = false;

           //----+ calculating indicator values
           Fama1 = iCustom(NULL, TimeframeX_Up,
                    "MAMA_NK", FastLimitX_Up, SlowLimitX_Up, IPCX_Up, 0, 1);
           //---
           Mama1 = iCustom(NULL, TimeframeX_Up,
                    "MAMA_NK", FastLimitX_Up, SlowLimitX_Up, IPCX_Up, 1, 1);
           //----+ defining trend
           TrendX_Up = Mama1 - Fama1;
           //----+ defining signals for trade closing
           if (TrendX_Up < 0)
                      BUY_Stop = true;
          }

         //----+ +----------------------------------------+
         //----+ DEFINING SIGNAL FOR MARKET ENTERING      |
         //----+ +----------------------------------------+
         if (LastBars_Up != IBARS_Up)
          {
           //----+ Initialization of variables
           BUY_Sign = false;
           LastBars_Up = IBARS_Up;

           //----+ calculating indicator values and uploading them into buffer
           for(bar = 1; bar <= 3; bar++)
                     Mov[bar - 1]=
                         iCustom(NULL, Timeframe_Up,
                                "JFatl", Length_Up, Phase_Up,
                                                   0, IPC_Up, 0, bar);

           //----+ defining signals for trades
           dMov12 = Mov[0] - Mov[1];
           dMov23 = Mov[1] - Mov[2];

           if (TrendX_Up > 0)
              if (dMov23 < 0)
                if (dMov12 > 0)
                        BUY_Sign = true;
          }

         //----+ +-------------------+
         //----+ EXECUTION OF TRADES |
         //----+ +-------------------+
          if (!OpenBuyOrder1(BUY_Sign, 1, Money_Management_Up,
                                          STOPLOSS_Up, TAKEPROFIT_Up))
                                                                 return(-1);
          if (ClosePos_Up)
                if (!CloseOrder1(BUY_Stop, 1))
                                        return(-1);
        }
     }

   //----+ +---------------------------------------------------------------+
   //----++ CODE FOR SHORT POSITIONS                                       |
   //----+ +---------------------------------------------------------------+
   if (Test_Dn)
    {
      int IBARS_Dn = iBars(NULL, Timeframe_Dn);
      int IBARSX_Dn = iBars(NULL, TimeframeX_Dn);

      if (IBARS_Dn >= MinBar_Dn && IBARSX_Dn >= MinBarX_Dn)
       {
         //----+ +----------------------+
         //----+ DEFINING TREND         |
         //----+ +----------------------+
         if (LastBarsX_Dn != IBARSX_Dn)
          {
           //----+ Initialization of variables
           LastBarsX_Dn = IBARSX_Dn;
           SELL_Stop = false;

           //----+ calculating indicator values
           Fama1 = iCustom(NULL, TimeframeX_Dn,
                    "MAMA_NK", FastLimitX_Dn, SlowLimitX_Dn, IPCX_Dn, 0, 1);
           //---
           Mama1 = iCustom(NULL, TimeframeX_Dn,
                    "MAMA_NK", FastLimitX_Dn, SlowLimitX_Dn, IPCX_Dn, 1, 1);
           //----+ defining trend
           TrendX_Dn = Mama1 - Fama1;
           //----+ defining signals for trade closing
           if (TrendX_Dn > 0)
                      SELL_Stop = true;
          }

         //----+ +----------------------------------------+
         //----+ DEFINING SIGNAL FOR MARKET ENTERING      |
         //----+ +----------------------------------------+
         if (LastBars_Dn != IBARS_Dn)
          {
           //----+ Initialization of variables
           SELL_Sign = false;
                      LastBars_Dn = IBARS_Dn;

           //----+ calculating indicator values and uploading them into buffer
           for(bar = 1; bar <= 3; bar++)
                     Mov[bar - 1]=
                         iCustom(NULL, Timeframe_Dn,
                                "JFatl", Length_Dn, Phase_Dn,
                                                   0, IPC_Dn, 0, bar);

           //----+ defining signals for trades
           dMov12 = Mov[0] - Mov[1];
           dMov23 = Mov[1] - Mov[2];

           if (TrendX_Dn < 0)
              if (dMov23 > 0)
                if (dMov12 < 0)
                       SELL_Sign = true;
          }

         //----+ +-------------------+
         //----+ EXECUTION OF TRADES |
         //----+ +-------------------+
          if (!OpenSellOrder1(SELL_Sign, 2, Money_Management_Dn,
                                            STOPLOSS_Dn, TAKEPROFIT_Dn))
                                                                   return(-1);
          if (ClosePos_Dn)
                if (!CloseOrder1(SELL_Stop, 2))
                                        return(-1);
        }
     }
//----+

    return(0);
  }
//+------------------------------------------------------------------+
```

Though the basic algorithm of this EA differs from the one underlying the previous EA, the general idea of using two graphs appears to be absolutely operating in this case too.

### Conclusion

I think the approach of constructing automated trading systems described in this article will help readers who already have some experience in EA writing to construct analogous Expert Advisors with minimal waste of effort. It should be also added here that the practical usefulness of such expert Advisors depends greatly on their proper optimization.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1525](https://www.mql5.com/ru/articles/1525)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1525.zip "Download all attachments in the single ZIP archive")

[EXPERTS.zip](https://www.mql5.com/en/articles/download/1525/EXPERTS.zip "Download EXPERTS.zip")(19.56 KB)

[INCLUDE.zip](https://www.mql5.com/en/articles/download/1525/INCLUDE.zip "Download INCLUDE.zip")(25.55 KB)

[indicators.zip](https://www.mql5.com/en/articles/download/1525/indicators.zip "Download indicators.zip")(14.02 KB)

[TESTER.zip](https://www.mql5.com/en/articles/download/1525/TESTER.zip "Download TESTER.zip")(6.72 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Simple Trading Systems Using Semaphore Indicators](https://www.mql5.com/en/articles/358)
- [Averaging Price Series for Intermediate Calculations Without Using Additional Buffers](https://www.mql5.com/en/articles/180)
- [Creating an Indicator with Multiple Indicator Buffers for Newbies](https://www.mql5.com/en/articles/48)
- [Creating an Expert Advisor, which Trades on a Number of Instruments](https://www.mql5.com/en/articles/105)
- [The Principles of Economic Calculation of Indicators](https://www.mql5.com/en/articles/109)
- [Practical Implementation of Digital Filters in MQL5 for Beginners](https://www.mql5.com/en/articles/32)
- [Custom Indicators in MQL5 for Newbies](https://www.mql5.com/en/articles/37)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39449)**
(2)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
18 Jun 2008 at 11:34

dear sirs, i would like to set up a


![hopokuk](https://c.mql5.com/avatar/avatar_na2.png)

**[hopokuk](https://www.mql5.com/en/users/hopokuk)**
\|
20 Jun 2008 at 03:43

**neilmonk1234:**

dear sirs, i would like to set up a

The optimization status is 172/10496 (1030301)

[genetic algorithm](https://www.mql5.com/en/articles/55 "Article: Genetic Algorithms Made Easy!") is ON

It has been 30 hours 32 minutes so far (after this it says 1822.45.07)

How much longer is this likely to take?

I haven't used genetic algorithm before, I thought it was meant to speed up as it progresses?


![Show Must Go On, or Once Again about ZigZag](https://c.mql5.com/2/16/620_30.gif)[Show Must Go On, or Once Again about ZigZag](https://www.mql5.com/en/articles/1531)

About an obvious but still substandard method of ZigZag composition, and what it results in: the Multiframe Fractal ZigZag indicator that represents ZigZags built on three larger ons, on a single working timeframe (TF). In their turn, those larger TFs may be non-standard, too, and range from M5 to MN1.

![Comfortable Scalping](https://c.mql5.com/2/15/553_7.gif)[Comfortable Scalping](https://www.mql5.com/en/articles/1509)

The article describes the method of creating a tool for comfortable scalping. However, such an approach to trade opening can be applied in any trading.

![Integrating MetaTrader 4  Client Terminal with MS SQL Server](https://c.mql5.com/2/16/625_23.gif)[Integrating MetaTrader 4 Client Terminal with MS SQL Server](https://www.mql5.com/en/articles/1533)

The article gives an example of integrating MetaTrader 4 Client Terminal with MS SQL Server using a dll. Attached are both source codes in С++ and in MQL4, and a ready-made and compiled Visual C++ 6.0 SP5 project.

![Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part IV)](https://c.mql5.com/2/15/595_130.gif)[Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part IV)](https://www.mql5.com/en/articles/1523)

In this article the author continues to analyze implementation algorithms of simplest trading systems and introduces recording of optimization results in backtesting into one html file in the form of a table. The article will be useful for beginning traders and EA writers.

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/1525&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049003514455237391)

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