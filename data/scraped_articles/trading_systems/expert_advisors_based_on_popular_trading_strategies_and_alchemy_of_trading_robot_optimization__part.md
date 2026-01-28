---
title: Expert Advisors Based on Popular Trading Strategies and Alchemy of Trading Robot Optimization (Part VI)
url: https://www.mql5.com/en/articles/1535
categories: Trading Systems
relevance_score: 9
scraped_at: 2026-01-22T17:39:52.810455
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/1535&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049287759685855412)

MetaTrader 4 / Trading systems


### Introduction

In my [previous article](https://www.mql5.com/en/articles/1525), I gave a detailed description of writing Expert Advisors that process the information arriving from two different timeframes. However, the matter is that this information is often insufficient to accurately enter the market. For example, if a smaller timeframe is equal to H1, then entering the market immediately at the change of a one-hour bar is not often the best solution, since the trend on the timeframe smaller than H1 and usually existing in the price noise may work against the position to be opened. In many cases, this short-term trend can be detected rather easily. In such a case, if it moves against the position to be opened, you should postpone your entering the market until this trend acting on the smallest timeframe changes its direction for an opposite one. Or, in the worst case, you can enter the market before the next one-hour bar changes. It is this task that I will try to solve in my article.

### Elder's Triple Screen System

Alexander Elder is known to be the author of rather popular books on the psychology of trading and crowd behavior. It was him who invented the idea to use the charts of three timeframes in analyzing financial markets. These charts were named Elder's Triple Screen. We have already learned in my [previous article](https://www.mql5.com/en/articles/1525) how to construct a double screen. Now we have to add the third screen to it. As the examples of further code complication, we could use some ready EAs from my [previous article](https://www.mql5.com/en/articles/1525). However, in this present article, I decided to build another EA (Exp\_14.mq4) based on the same procedures, just for a change.

As the initial basis to write the code, I take Exp\_12.mq4 in which I replace the alerting moving average, JFatl.nq4, with oscillator JCCIX.mq4 and the trend-following indicator MAMA\_NK.mq4 consisting of two MAs with indicator StepMA\_Stoch\_NK.mq4 consisting of a couple of stochastic oscillators. Eventually, the initial algorithm femains the same, I just changed the calls to custom indicators, the external variables of the EA and the initialization of constants in the block of the init() function, and I also complicated the code of blocks aimed at detecting the signals to enter the market. I present the working algorithm of this EA once again using two timeframes in a very general form, as I did in my previous article. However, I do it in a bit more details this time.

For long positions, we have:

![](https://c.mql5.com/2/16/long1.png)

And for short positions:

![](https://c.mql5.com/2/16/short1.png)

The resulting algorithm we have to realize in the program code for using the charts of three different timeframes as rationally as possible appears as follows.

For long positions:

![](https://c.mql5.com/2/16/long2.png)

For short positions:

[![](https://c.mql5.com/2/16/short2_small.png)](https://c.mql5.com/2/16/short2.png "https://c.mql5.com/2/16/short2.png")

The realization of this algorithm in a program code based on Exp\_15.mq4 can be presented as follows:

```
//+==================================================================+
//|                                                       Exp_15.mq4 |
//|                             Copyright © 2008,   Nikolay Kositsin |
//|                              Khabarovsk,   farria@mail.redcom.ru |
//+==================================================================+
#property copyright "Copyright © 2008, Nikolay Kositsin"
#property link "farria@mail.redcom.ru"
//----+ +---------------------------------------------------------------------------+
//---- EXPERT ADVISORS INPUTS FOR BUY TRADES
extern bool   Test_Up = true;//filter of trade calculations direction
extern double Money_Management_Up = 0.1;
//---- + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
extern int    TimeframeX_Up = 1440;
extern int    PeriodWATR_Up = 10;
extern double Kwatr_Up = 1.0000;
extern int    HighLow_Up = 0;
//---- + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
extern int    Timeframe_Up = 240;
extern int    JJLength_Up = 8;  // depth of JJMA smoothing for the entry price
extern int    JXLength_Up = 8;  // depth of JurX smoothing for the obtained indicator
extern int    Phase_Up = 100;// the parameter ranging
                        // from -100 to +100 influences the process quality;
extern int    IPC_Up = 0; /* Selecting price to calculate
the indicator on (0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL,
6-WEIGHTED, 7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close.) */
//---- + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
extern int    TimeframeN_Up = 15;
extern int    Noise_period_Up = 8;
//extern int    SmoothN_Up = 7;
//extern int    MaMethodN_Up = 1;
//---- + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
extern int    STOPLOSS_Up = 50;  // StopLoss
extern int    TAKEPROFIT_Up = 100; // TakeProfit
extern bool   ClosePos_Up = true; // enable forcible closing the position
//----+ +---------------------------------------------------------------------------+
//---- EXPERT ADVISORS INPUTS FOR SELL TRADES
extern bool   Test_Dn = true;//filter of trade calculations direction
extern double Money_Management_Dn = 0.1;
//---- + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
extern int    TimeframeX_Dn = 1440;
extern int    PeriodWATR_Dn = 10;
extern double Kwatr_Dn = 1.0000;
extern int    HighLow_Dn = 0;
//---- + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
extern int    Timeframe_Dn = 240;
extern int    JJLength_Dn = 8;  // depth of JJMA smoothing for the entry price
extern int    JXLength_Dn = 8;  // depth of JurX smoothing for the obtained indicator
extern int    Phase_Dn = 100;// the parameter ranging
                        // from -100 to +100 influences the process quality;
extern int    IPC_Dn = 0; /* Selecting price to calculate
the indicator on (0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL,
6-WEIGHTED, 7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close.) */
//---- + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
extern int    TimeframeN_Dn = 15;
extern int    Noise_period_Dn = 8;
//extern int    SmoothN_Dn = 7;
//extern int    MaMethodN_Dn = 1;
//---- + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
extern int    STOPLOSS_Dn = 50;  // StopLoss
extern int    TAKEPROFIT_Dn = 100; // TakeProfit
extern bool   ClosePos_Dn = true; // enable forcible closing the position
//----+ +---------------------------------------------------------------------------+
//---- Integer variables for calling to custom indicators
int SmoothN_Up = 7, SmoothN_Dn = 7, MaMethodN_Up = 1, MaMethodN_Dn = 1;
//---- Integer variables for the minimum of reference bars
int MinBar_Up, MinBar_Dn, MinBarX_Up, MinBarX_Dn, MinBarN_Up, MinBarN_Dn;
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
   //---- Checking the value of variable Timeframe for correctness
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
//---- Checking the values of timeframe variables for correctness
   TimeframeCheck("TimeframeX_Up", TimeframeX_Up);
   TimeframeCheck("Timeframe_Up", Timeframe_Up);
   TimeframeCheck("TimeframeN_Up", TimeframeN_Up);

//---- Checking the values of timeframe variables for correctness
   TimeframeCheck("TimeframeX_Dn", TimeframeX_Dn);
   TimeframeCheck("Timeframe_Dn", Timeframe_Dn);
   TimeframeCheck("TimeframeN_Dn", TimeframeN_Dn);

//---- Initialization of variables
   MinBarX_Up = 2 + PeriodWATR_Up;
   MinBar_Up = 4 + 3 * JXLength_Up + 30;
   MinBarN_Up = 4 + Noise_period_Up + SmoothN_Up;

//---- Initialization of variables
   MinBarX_Dn = 2 + PeriodWATR_Dn;
   MinBar_Dn = 4 + 3 * JXLength_Dn + 30;
   MinBarN_Dn = 4 + Noise_period_Dn + SmoothN_Dn;

//---- initialization complete
   return(0);
  }
//+==================================================================+
//| expert deinitialization function                                 |
//+==================================================================+
int deinit()
  {
//----+

    //---- EA deinitialization complete
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
   double JCCIX[2], Trend, Fast_StepMA, Slow_StepMA, MA1, MA2;
   //----+ Declaration of static variables
   static datetime StopTime_Up, StopTime_Dn;
   static double   TrendX_Up, TrendX_Dn, OldTrend_Up, OldTrend_Dn;
   //---
   static int  LastBars_Up, LastBars_Dn;
   static int  LastBarsX_Up, LastBarsX_Dn, LastBarsN_Up, LastBarsN_Dn;
   //---
   static bool BUY_Sign, BUY_Stop, SELL_Sign, SELL_Stop;
   static bool SecondStart_Up, SecondStart_Dn, NoiseBUY_Sign, NoiseSELL_Sign;

   //----+ +---------------------------------------------------------------+
   //----++ CODE FOR LONG POSITIONS                                        |
   //----+ +---------------------------------------------------------------+
   if (Test_Up)
    {
      int IBARS_Up = iBars(NULL, Timeframe_Up);
      int IBARSX_Up = iBars(NULL, TimeframeX_Up);
      int IBARSN_Up = iBars(NULL, TimeframeN_Up);
      //---
      if (IBARS_Up >= MinBar_Up && IBARSX_Up >= MinBarX_Up && IBARSN_Up >= MinBarN_Up)
       {
         //----+ +----------------------+
         //----+ DETECTING A TREND      |
         //----+ +----------------------+
         if (LastBarsX_Up != IBARSX_Up)
          {
           //----+ Initialization of variables
           LastBarsX_Up = IBARSX_Up;
           BUY_Sign = false;
           BUY_Stop = false;

           //----+ calculating the values of indicators
           Fast_StepMA = iCustom(NULL, TimeframeX_Up, "StepMA_Stoch_NK",
                                 PeriodWATR_Up, Kwatr_Up, HighLow_Up, 0, 1);
           //---
           Slow_StepMA = iCustom(NULL, TimeframeX_Up, "StepMA_Stoch_NK",
                                 PeriodWATR_Up, Kwatr_Up, HighLow_Up, 1, 1);
           //----+ detecting a trend
           TrendX_Up = Fast_StepMA - Slow_StepMA;
           //----+ defining a signal to close trades
           if (TrendX_Up < 0)
                      BUY_Stop = true;
          }

         //----+ +----------------------------------------+
         //----+ DETECTING SIGNALS TO ENTER THE MARKET    |
         //----+ +----------------------------------------+
         if (LastBars_Up != IBARS_Up && TrendX_Up > 0)
          {
           //----+ Initialization of variables
           BUY_Sign = false;
           LastBars_Up = IBARS_Up;
           //----+ Initialization of noise variables
           NoiseBUY_Sign = false;
           StopTime_Up = iTime(NULL, Timeframe_Up, 0)
                                          + 50 * Timeframe_Up;

           //----+ Initialization of zero
           if (!SecondStart_Up)
            {
              //--- Search for trend direction at the first start
              for(bar = 2; bar < IBARS_Up - 1; bar++)
               {
                 JCCIX[0] = iCustom(NULL, Timeframe_Up,
                    "JCCIX", JJLength_Up, JXLength_Up, Phase_Up, IPC_Up, 0, bar);
                 //---
                 JCCIX[1] =  iCustom(NULL, Timeframe_Up,
                    "JCCIX", JJLength_Up, JXLength_Up, Phase_Up, IPC_Up, 0, bar + 1);
                 //---
                 OldTrend_Up = JCCIX[0] - JCCIX[1];
                 //---
                 if (OldTrend_Up != 0)
                   {
                    SecondStart_Up = true;
                    break;
                   }
               }
            }

           //----+ calculating the values of indicators and loading them to a buffer
           for(bar = 1; bar < 3; bar++)
                     JCCIX[bar - 1] =
                         iCustom(NULL, Timeframe_Up,
                                "JCCIX", JJLength_Up, JXLength_Up,
                                                   Phase_Up, IPC_Up, 0, bar);

           //----+ detecting signals for trades
           Trend = JCCIX[0] - JCCIX[1];

           if (TrendX_Up > 0)
              if (OldTrend_Up < 0)
                         if (Trend > 0)
                                 BUY_Sign = true;
           if (Trend != 0)
                   OldTrend_Up = Trend;
          }

         //----+ +------------------------------------------------+
         //----+ DETECTING NOISE SIGNALS TO ENTER THE MARKET      |
         //----+ +------------------------------------------------+
         if (BUY_Sign)
          if (LastBarsN_Up != IBARSN_Up)
           {
             NoiseBUY_Sign = false;
             LastBarsN_Up = IBARSN_Up;
             //---
             MA1 = iCustom(NULL, TimeframeN_Up,
                      "2Moving Avereges", Noise_period_Up, SmoothN_Up,
                         MaMethodN_Up, MaMethodN_Up, PRICE_LOW, 0, 0, 1);
             //---
             MA2 = iCustom(NULL, TimeframeN_Up,
                       "2Moving Avereges", Noise_period_Up, SmoothN_Up,
                         MaMethodN_Up, MaMethodN_Up, PRICE_LOW, 0, 0, 2);
             //---
             if (MA1 > MA2 || TimeCurrent() > StopTime_Up)
                                           NoiseBUY_Sign = true;

           }

         //----+ +-------------------+
         //----+ MAKING TRADES       |
         //----+ +-------------------+
         if (NoiseBUY_Sign)
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
      int IBARSN_Dn = iBars(NULL, TimeframeN_Dn);
      //---
      if (IBARS_Dn >= MinBar_Dn && IBARSX_Dn >= MinBarX_Dn && IBARSN_Dn >= MinBarN_Dn)
       {
         //----+ +----------------------+
         //----+ DETECTING A TREND      |
         //----+ +----------------------+
         if (LastBarsX_Dn != IBARSX_Dn)
          {
           //----+ Initialization of variables
           LastBarsX_Dn = IBARSX_Dn;
           SELL_Sign = false;
           SELL_Stop = false;

           //----+ calculating the values of indicators
           Fast_StepMA = iCustom(NULL, TimeframeX_Dn, "StepMA_Stoch_NK",
                                 PeriodWATR_Dn, Kwatr_Dn, HighLow_Dn, 0, 1);
           //---
           Slow_StepMA = iCustom(NULL, TimeframeX_Dn, "StepMA_Stoch_NK",
                                 PeriodWATR_Dn, Kwatr_Dn, HighLow_Dn, 1, 1);
           //----+ detecting a trend
           TrendX_Dn = Fast_StepMA - Slow_StepMA;
           //----+ defining a signal to close trades
           if (TrendX_Dn > 0)
                      SELL_Stop = true;
          }

         //----+ +----------------------------------------+
         //----+ DETECTING SIGNALS TO ENTER THE MARKET    |
         //----+ +----------------------------------------+
         if (LastBars_Dn != IBARS_Dn && TrendX_Dn < 0)
          {
           //----+ Initialization of variables
           SELL_Sign = false;
           LastBars_Dn = IBARS_Dn;
           //----+ Initialization of noise variables
           NoiseSELL_Sign = false;
           StopTime_Dn = iTime(NULL, Timeframe_Dn, 0)
                                          + 50 * Timeframe_Dn;

           //----+ Initialization of zero
           if (!SecondStart_Dn)
            {
              //--- Search for trend direction at the first start
              for(bar = 2; bar < IBARS_Dn - 1; bar++)
               {
                 JCCIX[0] = iCustom(NULL, Timeframe_Dn,
                    "JCCIX", JJLength_Dn, JXLength_Dn, Phase_Dn, IPC_Dn, 0, bar);
                 //---
                 JCCIX[1] =  iCustom(NULL, Timeframe_Dn,
                    "JCCIX", JJLength_Dn, JXLength_Dn, Phase_Dn, IPC_Dn, 0, bar + 1);
                 //---
                 OldTrend_Dn = JCCIX[0] - JCCIX[1];
                 //---
                 if (OldTrend_Dn != 0)
                   {
                    SecondStart_Dn = true;
                    break;
                   }
               }
            }

           //----+ calculating the values of indicators and loading them to a buffer
           for(bar = 1; bar < 3; bar++)
                     JCCIX[bar - 1]=
                         iCustom(NULL, Timeframe_Dn,
                                "JCCIX", JJLength_Dn, JXLength_Dn,
                                                   Phase_Dn, IPC_Dn, 0, bar);
           //----+ detecting signals for trades
           Trend = JCCIX[0] - JCCIX[1];
           //---
           if (TrendX_Dn < 0)
              if (OldTrend_Dn > 0)
                         if (Trend < 0)
                                 SELL_Sign = true;
           if (Trend != 0)
                   OldTrend_Dn = Trend;
          }

         //----+ +------------------------------------------------+
         //----+ DETECTING NOISE SIGNALS TO ENTER THE MARKET      |
         //----+ +------------------------------------------------+
         if (SELL_Sign)
          if (LastBarsN_Dn != IBARSN_Dn)
           {
             NoiseSELL_Sign = false;
             LastBarsN_Dn = IBARSN_Dn;
             //---
             MA1 = iCustom(NULL, TimeframeN_Dn,
                      "2Moving Avereges", Noise_period_Dn, SmoothN_Dn,
                         MaMethodN_Dn, MaMethodN_Dn, PRICE_HIGH, 0, 0, 1);
             //---
             MA2 = iCustom(NULL, TimeframeN_Dn,
                      "2Moving Avereges", Noise_period_Dn, SmoothN_Dn,
                         MaMethodN_Dn, MaMethodN_Dn, PRICE_HIGH, 0, 0, 2);
             //---
             if (MA1 < MA2 || TimeCurrent() > StopTime_Dn)
                                           NoiseSELL_Sign = true;
           }

         //----+ +-------------------+
         //----+ MAKING TRADES       |
         //----+ +-------------------+
         if (NoiseSELL_Sign)
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

Now we should to get in more details about turning Exp\_14.mq4 into Exp\_15.mq4. We have a new module, "DETECTING NOISE SIGNALS TO ENTER THE MARKET", in our program code. The point of this module operation can be expressed as follows (I'm considering the algorithm for long positions only):

Signal NoiseBUY\_Sign occurs if the trend direction on the smallest timeframe coincides with the direction of the market entering signal, BUY\_Sig. Or, in case of this trend mismatching, signal NoiseBUY\_Sign occurs before the regular change of the bar.

As a trend-following MA, I used an indicator obtained by double smoothing the price sequence by standard averaging algorithms. As an external parameter for the EA from this module, I used only two variables:

```
extern int    TimeframeN_Up = 15;
extern int    Noise_period_Up = 8;
```

I made the most external variables of custom indicator 2Moving Avereges.mq4 fixed (initialization of global variables):

```
int SmoothN_Up = 7, SmoothN_Dn = 7, MaMethodN_Up = 1, MaMethodN_Dn = 1;
```

The resting logic of adding the code is absolutely the same as what I did in my [previous article](https://www.mql5.com/en/articles/1525).

### General Idea of Constructing Expert Advisors Using Three Timeframes

In general, the code of the EA is ready, and I could stop at this stage. However, in my opinion, the smallest part of the work has been really done. The first trading system written on the basis of this idea would hardly produce good results in real trading. So we should be geared to the fact that we have to write the code of more than one or two similar EAs to choose a more proper version. So now we are facing a task to abstract from the specific trading signals calculating algorithms and use only the triple screen algorithm itself. Which, in general, is not a problem. The resulting code without algorithms will appear as follows:

```
//+==================================================================+
//|                                                 ThreeScreens.mqh |
//|                             Copyright © 2008,   Nikolay Kositsin |
//|                              Khabarovsk,   farria@mail.redcom.ru |
//+==================================================================+
#property copyright "Copyright © 2008, Nikolay Kositsin"
#property link "farria@mail.redcom.ru"
//----+ +---------------------------------------------------------------------------+
//---- EXPERT ADVISORS INPUTS FOR BUY TRADES
extern bool   Test_Up = true;//filter of trade calculations direction
extern double Money_Management_Up = 0.1;
//---- + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
extern int    TimeframeX_Up = 1440;
// Declarations and initializations of the EA external parameters for long positions for the largest timeframe
//---- + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
extern int    Timeframe_Up = 240;
// Declarations and initializations of the EA external parameters for long positions for the middle timeframe
//---- + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
extern int    TimeframeN_Up = 15;
// Declarations and initializations of the EA external parameters for long positions for the smallest timeframe
//---- + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
extern int    STOPLOSS_Up = 50;  // StopLoss
extern int    TAKEPROFIT_Up = 100; // TakeProfit
extern bool   ClosePos_Up = true; // enable forcible closing the position
//----+ +---------------------------------------------------------------------------+
//---- EXPERT ADVISORS INPUTS FOR SELL TRADES
extern bool   Test_Dn = true;//filter of trade calculations direction
extern double Money_Management_Dn = 0.1;
//---- + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
extern int    TimeframeX_Dn = 1440;
// Declarations and initializations of the EA external parameters for short positions for largest timeframe
//---- + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
extern int    Timeframe_Dn = 240;
// Declarations and initializations of the EA external parameters for short positions for middle timeframe
//---- + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
extern int    TimeframeN_Dn = 15;
// Declarations and initializations of the EA external parameters for short positions for the smallest timeframe
//---- + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
extern int    STOPLOSS_Dn = 50;  // StopLoss
extern int    TAKEPROFIT_Dn = 100; // TakeProfit
extern bool   ClosePos_Dn = true; // enable forcible closing the position
//----+ +---------------------------------------------------------------------------+
//---- Integer variables for the minimum of reference bars
int MinBar_Up, MinBar_Dn, MinBarX_Up, MinBarX_Dn, MinBarN_Up, MinBarN_Dn;
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
   //---- Checking the value of variable Timeframe for correctness
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
//---- Checking the values of timeframe variables for correctness
   TimeframeCheck("TimeframeX_Up", TimeframeX_Up);
   TimeframeCheck("Timeframe_Up", Timeframe_Up);
   TimeframeCheck("TimeframeN_Up", TimeframeN_Up);

//---- Checking the values of timeframe variables for correctness
   TimeframeCheck("TimeframeX_Dn", TimeframeX_Dn);
   TimeframeCheck("Timeframe_Dn", Timeframe_Dn);
   TimeframeCheck("TimeframeN_Dn", TimeframeN_Dn);

//---- Initialization of variables for long positions
   MinBarX_Up = // initialization of the variable for the minimum reference bars for largest timeframe
   MinBar_Up = // initialization of the variable for the minimum reference bars for middle timeframe
   MinBarN_Up = // initialization of the variable for the minimum reference bars for the smallest timeframe

//---- Initialization of variables for short positions
   MinBarX_Dn =  // initialization of the variable for the minimum reference bars for largest timeframe
   MinBar_Dn =  // initialization of the variable for the minimum reference bars for middle timeframe
   MinBarN_Dn = // initialization of the variable for the minimum reference bars for the smallest timeframe

//---- initialization complete
   return(0);
  }
//+==================================================================+
//| expert deinitialization function                                 |
//+==================================================================+
int deinit()
  {
//----+

    //---- EA deinitialization complete
    return(0);
//----+
  }
//+==================================================================+
//| Custom Expert iteration function                                 |
//+==================================================================+
int start()
  {
   //----+ Declaration of local variables of trading algorithms
   //----+ Declaration of static variables of trading algortihms

   //----+ Declaration of static variables
   static datetime StopTime_Up, StopTime_Dn;
   //---
   static int  LastBars_Up, LastBars_Dn;
   static int  LastBarsX_Up, LastBarsX_Dn;
   static int  LastBarsN_Up, LastBarsN_Dn;
   //---
   static bool BUY_Sign, BUY_Stop;
   static bool SELL_Sign, SELL_Stop;
   static bool NoiseBUY_Sign, NoiseSELL_Sign;
   static double TrendX_Up, TrendX_Dn;

   //----+ +---------------------------------------------------------------+
   //----++ CODE FOR LONG POSITIONS                                        |
   //----+ +---------------------------------------------------------------+
   if (Test_Up)
    {
      int IBARS_Up = iBars(NULL, Timeframe_Up);
      int IBARSX_Up = iBars(NULL, TimeframeX_Up);
      int IBARSN_Up = iBars(NULL, TimeframeN_Up);
      //---
      if (IBARS_Up >= MinBar_Up && IBARSX_Up >= MinBarX_Up && IBARSN_Up >= MinBarN_Up)
       {
         //----+ +----------------------+
         //----+ DETECTING A TREND      |
         //----+ +----------------------+
         if (LastBarsX_Up != IBARSX_Up)
          {
           //----+ Initialization of variables
           LastBarsX_Up = IBARSX_Up;
           BUY_Sign = false;
           BUY_Stop = false;

           // Trend direction detecting algorithm on the largest timeframe
                                              //(initializing variable TrendX_Up)

           //----+ defining a signal to close trades
           if (TrendX_Up < 0)
                      BUY_Stop = true;
          }

         //----+ +----------------------------------------+
         //----+ DETECTING SIGNALS TO ENTER THE MARKET    |
         //----+ +----------------------------------------+
         if (LastBars_Up != IBARS_Up && TrendX_Up > 0)
          {
           //----+ Initialization of variables
           BUY_Sign = false;
           LastBars_Up = IBARS_Up;
           //----+ Initialization of noise variables
           NoiseBUY_Sign = false;
           StopTime_Up = iTime(NULL, Timeframe_Up, 0)
                                          + 50 * Timeframe_Up;

           // Entering point determining algorithm on the middle timeframe
                                                //(Initializing variable BUY_Sign)

          }

         //----+ +------------------------------------------------+
         //----+ DETECTING NOISE SIGNALS TO ENTER THE MARKET      |
         //----+ +------------------------------------------------+
         if (BUY_Sign)
          if (LastBarsN_Up != IBARSN_Up)
           {
             NoiseBUY_Sign = false;
             LastBarsN_Up = IBARSN_Up;
             //---

             // Entering point precising algorithm on the smallest timeframe
                                            //(Initializing variable NoiseBUY_Sign)

           }

         //----+ +-------------------+
         //----+ MAKING TRADES       |
         //----+ +-------------------+
         if (NoiseBUY_Sign)
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
      int IBARSN_Dn = iBars(NULL, TimeframeN_Dn);
      //---
      if (IBARS_Dn >= MinBar_Dn && IBARSX_Dn >= MinBarX_Dn && IBARSN_Dn >= MinBarN_Dn)
       {
         //----+ +----------------------+
         //----+ DETECTING A TREND      |
         //----+ +----------------------+
         if (LastBarsX_Dn != IBARSX_Dn)
          {
           //----+ Initialization of variables
           LastBarsX_Dn = IBARSX_Dn;
           SELL_Sign = false;
           SELL_Stop = false;

           // Trend direction detecting algorithm on the largest timeframe
                                               //(initializing variable TrendX_Dn)

           //----+ defining a signal to close trades
           if (TrendX_Dn > 0)
                      SELL_Stop = true;
          }

         //----+ +----------------------------------------+
         //----+ DETECTING SIGNALS TO ENTER THE MARKET    |
         //----+ +----------------------------------------+
         if (LastBars_Dn != IBARS_Dn && TrendX_Dn < 0)
          {
           //----+ Initialization of variables
           SELL_Sign = false;
           LastBars_Dn = IBARS_Dn;
           //----+ Initialization of noise variables
           NoiseSELL_Sign = false;
           StopTime_Dn = iTime(NULL, Timeframe_Dn, 0)
                                          + 50 * Timeframe_Dn;

           // Entering point determining algorithm on the middle timeframe
                                                //(Initializing variable SELL_Sign)

          }

         //----+ +------------------------------------------------+
         //----+ DETECTING NOISE SIGNALS TO ENTER THE MARKET      |
         //----+ +------------------------------------------------+
         if (SELL_Sign)
          if (LastBarsN_Dn != IBARSN_Dn)
           {
             NoiseSELL_Sign = false;
             LastBarsN_Dn = IBARSN_Dn;
             //---

             // Entering point precising algorithm on the smallest timeframe
                                           //(Initializing variable NoiseSELL_Sign)

           }

         //----+ +-------------------+
         //----+ MAKING TRADES       |
         //----+ +-------------------+
         if (NoiseSELL_Sign)
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

If we use this code as a template for writing EAs, we should, first of all, initializa variables in the corresponding blocks: "DETECTING A TREND" and "DETECTING NOISE SIGNAL TO ENTER THE MARKET":

```
TrendX_Up = 1;
TrendX_Dn =-1;
Noise8uy_Sign = true;
NoiseSELL_Sign = true;
```

After that, you can add your own codes into the blocks of "DETECTING SIGNALS TO ENTER THE MARKET" and ajust the EA to work with this code. You can learn how to do this in the code of EA Exp\_15\_A.mq4, in which there are only algorithms for detecting signals for entering the market for the middle timeframe, while there are no algorithms aimed at detecting the trend on the largest timeframe or those aimed at detecting the noise trend for the smallest timeframe. You should pay attention to the initializations of variables for the smallest amount of bars in block int init(), in this case:

```
//---- Initialization of variables
   MinBarX_Up = 0;
   MinBar_Up = 4 + 3 * JXLength_Up + 30;
   MinBarN_Up = 0;

//---- Initialization of variables
   MinBarX_Dn = 0;
   MinBar_Dn = 4 + 3 * JXLength_Dn + 30;
   MinBarN_Dn = 0;
```

At the second step, remove initializations from the blocks of "DETECTING A TREND":

```
TrendX_Up = 1;
TrendX_Dn =-1;
```

Add you code to detect the trend direction into these blocks and adjust the EA once again. This stage of code writing is displayed in Exp\_15\_B.mq4. Please don't forget to initialize variables MinBarX\_Up and MinBarX\_Dn in block init():

```
//---- Initialization of variables
   MinBarX_Up = 2 + PeriodWATR_Up;
   MinBar_Up = 4 + 3 * JXLength_Up + 30;
   MinBarN_Up = 0;

//---- Initialization of variables
   MinBarX_Dn = 2 + PeriodWATR_Dn;
   MinBar_Dn = 4 + 3 * JXLength_Dn + 30;
   MinBarN_Dn = 0;
```

As a result, we have an EA working on two timeframes. At the third step, in absolutely the same way, as the EA code in the blocks of "DETECTING NOISE SIGNALS TO ENTER THE MARKET", having preliminarily removed initializations

```
Noise8uy_Sign = true;
NoiseSELL_Sign = true;
```

from those blocks and added arithmetic operations for the initialization of variables for the smallest amount of bars in block int init(), in this case:

```
//---- Initialization of variables
   MinBarX_Up = 2 + PeriodWATR_Up;
   MinBar_Up = 4 + 3 * JXLength_Up + 30;
   MinBarN_Up = 4 + Noise_period_Up + SmoothN_Up;

//---- Initialization of variables
   MinBarX_Dn = 2 + PeriodWATR_Dn;
   MinBar_Dn = 4 + 3 * JXLength_Dn + 30;
   MinBarN_Dn = 4 + Noise_period_Dn + SmoothN_Dn;
```

Thus, the code of the EA turns out to be made within three stages. However, if you try to build this code within only one stage, you may make some errors in it, which cannot be easily detected in future!

### Conclusion

In this present article, I have presented my version of a general approach to writing Expert Advisors using three timeframes. Technically, this idea can be rather easily realized in MQL4. There is another questions, though: "What solutions would help such an idea reveal its certain practical sense?"

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1535](https://www.mql5.com/ru/articles/1535)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1535.zip "Download all attachments in the single ZIP archive")

[EXPERTS.zip](https://www.mql5.com/en/articles/download/1535/EXPERTS.zip "Download EXPERTS.zip")(20.37 KB)

[INCLUDE.zip](https://www.mql5.com/en/articles/download/1535/INCLUDE.zip "Download INCLUDE.zip")(34.56 KB)

[indicators.zip](https://www.mql5.com/en/articles/download/1535/indicators.zip "Download indicators.zip")(20.7 KB)

[TESTER.zip](https://www.mql5.com/en/articles/download/1535/TESTER.zip "Download TESTER.zip")(6.29 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/39516)**
(2)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
30 Jun 2009 at 22:02

I like your idea of blocking noise. It seems to me that noise is the bane of trading. Finding a pure trend is key, but often finding a trend is like seeing faces in the clouds even in live trading. How do we stay with the substantial trend, and reject the noise?

I don't want to sound flip, but isn't that what we're all trying to do?

Still, given that, there must be a scientific conclusion. After all, if the market is truly random, even that fact can be exploited for profit.

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
5 Aug 2009 at 01:04

Pls the explainations within the codes are not in english. What can i do to change them so that I can understand them?


![HTML Walkthrough Using MQL4](https://c.mql5.com/2/16/680_46.gif)[HTML Walkthrough Using MQL4](https://www.mql5.com/en/articles/1544)

HTML is nowadays one of the wide-spread types of documents. MetaTrader 4 Client Terminal allows you to save statements, test and optimization reports as .htm files. It is sometimes necessary to get information from such files in an MQL4 program. The article describes one of variations of how to get the tag structure and contents from HTML.

![How to Write Fast Non-Redrawing ZigZags](https://c.mql5.com/2/16/687_36.jpg)[How to Write Fast Non-Redrawing ZigZags](https://www.mql5.com/en/articles/1545)

A rather universal approach to writing indicators of the ZigZag type is proposed. The method includes a significant part of ZigZags already described and allows you to create new ones relatively easily.

![Idleness is the Stimulus to Progress. Semiautomatic Marking a Template](https://c.mql5.com/2/16/721_16.gif)[Idleness is the Stimulus to Progress. Semiautomatic Marking a Template](https://www.mql5.com/en/articles/1556)

Among the dozens of examples of how to work with charts, there is a method of manual marking a template. Trend lines, channels, support/resistance levels, etc. are imposed in a chart. Surely, there are some special programs for this kind of work. Everyone decides on his/her own which method to use. In this article, I offer you for your consideration the methods of manual marking with subsequent automating some elements of the repeated routine actions.

![Idleness is the Stimulus to Progress, or How to Work with Graphics Interacively](https://c.mql5.com/2/16/693_14.gif)[Idleness is the Stimulus to Progress, or How to Work with Graphics Interacively](https://www.mql5.com/en/articles/1546)

An indicator for interactive working with trend lines, Fibo levels, icons manually imposed on a chart. It allows you to draw the colored zones of Fibo levels, shows the moments of the price crossing the trend line, manages the "Price label" object.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/1535&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049287759685855412)

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