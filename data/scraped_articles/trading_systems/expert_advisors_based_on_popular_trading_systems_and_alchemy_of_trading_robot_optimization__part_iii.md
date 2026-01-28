---
title: Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part III)
url: https://www.mql5.com/en/articles/1521
categories: Trading Systems
relevance_score: 12
scraped_at: 2026-01-22T17:16:05.947171
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/1521&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049007590379201316)

MetaTrader 4 / Trading systems


### Introduction

I have received an offer from a reader of the [previous article](https://www.mql5.com/en/articles/1517) to automatize a little the process of backtesting to get the possibility of receiving results of all optimizations at the same time. Moreover, it is not very convenient to shift the testing period manually, this process should be also automated. The idea is great. The more so, MQL4 means offer all possibilities for its implementation. So I will start the article from the solution of this problem.

### Backtesting Automation

What we need for the solution of this task:

1\. Write in the necessary Expert Advisor below its heading a line of the following content:

```
//+==================================================================+
//| Custom BackTesting function                                      |
//+==================================================================+
#include <IsBackTestingTime.mqh>
```

Using this directive we include the function IsBackTestingTime() into the EA code. And don't forget to place the IsBackTestingTime.mqh file into the folder INCLUDE. This function:

```
bool IsBackTestingTime()
 {
 }
```

is used to define period of time, within which the backtesting optimization or backtesting will take place. In this time periods the function always returns 'true', in other times it returns 'false'. Besides this function external EA variables are added to the EA code by this directive:

```
//---- Declaration of external variables for backtesting
extern datetime Start_Time = D'2007.01.01'; // start time of zero optimization
extern int Opt_Period = 3; // optimization period in months, if less than zero, parameters are in days
extern int Test_Period = 2; // testing period in months
extern int Period_Shift = 1; // step of optimization period shift in months
extern int Opt_Number = 0; // optimization number
```

I hope, the meaning of these variables is clear for those who have read [my previous article](https://www.mql5.com/en/articles/1517), so there is no need to explain them once again.

2\. In the block of the starting function before the EA code place the simplest universal code of IsBackTestingTime() call, it limits EA operation to certain time frames depending on the number of the backtesting optimization.

```
 //----+ Execution of backtesting conditions
   if (!IsBackTestingTime())
                       return(0);
```

Schematically it will look like this:

```
//+==================================================================+
//|                                                 Exp_BackTest.mq4 |
//|                             Copyright © 2008,   Nikolay Kositsin |
//|                              Khabarovsk,   farria@mail.redcom.ru |
//+==================================================================+
#property copyright "Copyright © 2008, Nikolay Kositsin"
#property link "farria@mail.redcom.ru"
//+==================================================================+
//| Custom BackTesting function                                      |
//+==================================================================+
#include <IsBackTestingTime.mqh>
//---- INPUT PARAMETERS OF THE EA
//---- GLOBAL VARIABLES OF THE EA
//+==================================================================+
//| USER-DEFINED FUNCTIONS OF THE EA                                 |
//+==================================================================+

//+==================================================================+
//| Custom Expert initialization function                            |
//+==================================================================+
int init()
  {
//----+ +------------------------------------------------------------+
//---- CODE FOR THE EA INITIALIZATION
//----+ +------------------------------------------------------------+
//---- end of initialization
   return(0);
  }
//+==================================================================+
//| Custom Expert iteration function                                 |
//+==================================================================+
int start()
  {
   //----+ Execution of backtesting conditions
   if (!IsBackTestingTime())
                       return(0);
   //----+ +---------------------------------------------------------+
   //----+ CODE OF THE EA ALGORITHM
   //----+ +---------------------------------------------------------+
//----+

    return(0);
  }
//+------------------------------------------------------------------+
```

If you are interested in the detailed problem solution on the example of a ready EA, see the EA code Exp\_5\_1.mq4, which is the EA form the [previous article](https://www.mql5.com/en/articles/1517) Exp\_5.mq4 modified for backtesting. Actually there is no large difference in the optimization of the like EA as compared to a simple Expert Advisor. However, I think backtesting variables shouldn't be optimized, except for Opt\_Number, though you may have another opinion.

It is important to remember that after optimization during testing we get results not inside the optimization period, but after it, beyond its right border! It is not the best decision to make all backtesting optimization within one run using a genetic algorithm, it is much more interesting to analyze deeper each backtesting optimization separately without the optimization of the input variable Opt\_Number.

But even in this case such an approach facilitates the understanding of the EA behavior. And it should be remembered that the value of the external variable Opt\_Number can change from zero to a certain maximal value that can be defined the following way: from the total period in which all backtesting optimizations are conducted (in months) subtract the period of backtesting optimization in months (Opt\_Period) and subtract backtesting period (Test\_Period). Add one to the obtained value. The obtained result will be the maximum of the Opt\_Number variable if Period\_Shift is equal to one.

### Trading Systems Based on the Crossing of Two Movings

This variant of trading systems is quite popular. Now let's analyze the algorithm underlying such strategies. For long positions the entering algorithm looks like this:

![](https://c.mql5.com/2/16/2mov1.gif)

For short positions it is the following:

![](https://c.mql5.com/2/16/2mov2_3.gif)

As indicators two identical movings with different parameters defining averaging in indicators can be used. It is assumed that the parameter defining the averaging of MovA moving is always smaller than the same parameter of MovB moving. Thus in this trading system MovA is a quick moving, MovB is a slow one. Here is the trading system implementation variant based on two JMA movings:

```
//+==================================================================+
//|                                                        Exp_6.mq4 |
//|                             Copyright © 2007,   Nikolay Kositsin |
//|                              Khabarovsk,   farria@mail.redcom.ru |
//+==================================================================+
#property copyright "Copyright © 2007, Nikolay Kositsin"
#property link "farria@mail.redcom.ru"
//----+ +-------------------------------------------------------------------------------+
//---- EA INPUT PARAMETERS FOR BUY TRADES
extern bool   Test_Up = true;//filter of trade calculations direction
extern int    Timeframe_Up = 240;
extern double Money_Management_Up = 0.1;
extern int    LengthA_Up = 4;  // smoothing depth of the quick moving
extern int    PhaseA_Up = 100; // parameter changing in the range
          //-100 ... +100, influences the quality of transient process of quick moving
extern int    IPCA_Up = 0;/* Selecting prices, on which the indicator will be
calculated by the quick moving (0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL,
6-WEIGHTED, 7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close.) */
extern int    LengthB_Up = 4; // smoothing depth increment of slow moving to quick one
extern int    PhaseB_Up = 100; // parameter changing in the range
          //-100 ... +100, influences the quality of transient process of slow moving;
extern int    IPCB_Up = 0;/* Selecting prices, on which the indicator will be
calculated by the slow moving (0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL,
6-WEIGHTED, 7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close.) */
extern int    STOPLOSS_Up = 50;  // stop loss
extern int    TAKEPROFIT_Up = 100; // take profit
extern bool   ClosePos_Up = true; // forced position closing allowed
//----+ +-------------------------------------------------------------------------------+
//---- EA INPUT PARAMETERS FOR SELL TRADES
extern bool   Test_Dn = true;//filter of trade calculations direction
extern int    Timeframe_Dn = 240;
extern double Money_Management_Dn = 0.1;
extern int    LengthA_Dn = 4;  // smoothing depth of the quick moving
extern int    PhaseA_Dn = 100; // parameter changing in the range
         // -100 ... +100, influences the quality of transient process of quick moving;
extern int    IPCA_Dn = 0;/* Selecting prices, on which the indicator will be
calculated by the quick moving (0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL,
6-WEIGHTED, 7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close.) */
extern int    LengthB_Dn = 4; // smoothing depth increment of slow moving to quick one
extern int    PhaseB_Dn = 100; // parameter changing in the range
         // -100 ... +100, influences the quality of transient process of slow moving;
extern int    IPCB_Dn = 0;/* Selecting prices, on which the indicator will be
calculated by the slow moving(0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL,
6-WEIGHTED, 7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close.) */
extern int   STOPLOSS_Dn = 50;  // stop loss
extern int   TAKEPROFIT_Dn = 100; // take profit
extern bool   ClosePos_Dn = true; // forced position closing allowed
//----+ +-------------------------------------------------------------------------------+
//---- Integer variables for the minimum of calculation bars
int MinBar_Up, MinBar_Dn;
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
   if (Timeframe_Up != 1)
    if (Timeframe_Up != 5)
     if (Timeframe_Up != 15)
      if (Timeframe_Up != 30)
       if (Timeframe_Up != 60)
        if (Timeframe_Up != 240)
         if (Timeframe_Up != 1440)
           Print(StringConcatenate("Parameter Timeframe_Up cannot ",
                                  "be equal to ", Timeframe_Up, "!!!"));
//---- Checking the correctness of Timeframe_Dn variable value
   if (Timeframe_Dn != 1)
    if (Timeframe_Dn != 5)
     if (Timeframe_Dn != 15)
      if (Timeframe_Dn != 30)
       if (Timeframe_Dn != 60)
        if (Timeframe_Dn != 240)
         if (Timeframe_Dn != 1440)
           Print(StringConcatenate("Parameter Timeframe_Dn cannot ",
                                 "be equal to ", Timeframe_Dn, "!!!"));
//---- Initialization of variables
   MinBar_Up = 4 + 30;
   MinBar_Dn = 4 + 30;
//---- end of initialization
   return(0);
  }
//+==================================================================+
//| expert deinitialization function                                 |
//+==================================================================+
int deinit()
  {
//----+

    //---- End of EA deinitialization
    return(0);
//----+
  }
//+==================================================================+
//| Custom Expert iteration function                                 |
//+==================================================================+
int start()
  {
   //----+ Declaring local variables
   int    bar;
   double MovA[2], MovB[2];
   //----+ Declaring static variables
   static int LastBars_Up, LastBars_Dn;
   static bool BUY_Sign, BUY_Stop, SELL_Sign, SELL_Stop;

   //----++ CODE FOR LONG POSITIONS
   if (Test_Up)
    {
      int IBARS_Up = iBars(NULL, Timeframe_Up);

      if (IBARS_Up >= MinBar_Up)
       {
         if (LastBars_Up != IBARS_Up)
          {
           //----+ Initialization of variables
           BUY_Sign = false;
           BUY_Stop = false;
           LastBars_Up = IBARS_Up;

           //----+ CALCULATING INDICATOR VALUES AND UPLOADING THEM TO BUFFERS
           for(bar = 1; bar < 3; bar++)
                     MovA[bar - 1] =
                         iCustom(NULL, Timeframe_Up,
                                "JJMA", LengthA_Up, PhaseA_Up,
                                                   0, IPCA_Up, 0, bar);
           for(bar = 1; bar < 3; bar++)
                MovB[bar - 1] =
                   iCustom(NULL, Timeframe_Up,
                     "JJMA", LengthA_Up + LengthB_Up, PhaseB_Up,
                                                   0, IPCB_Up, 0, bar);

           //----+ DEFINING SIGNALS FOR TRADES
           if ( MovA[1] < MovB[1])
               if ( MovA[0] > MovB[0])
                        BUY_Sign = true;

            if ( MovA[0] > MovB[0])
                        BUY_Stop = true;
          }

          //----+ EXECUTION OF TRADES
          if (!OpenBuyOrder1(BUY_Sign, 1, Money_Management_Up,
                                          STOPLOSS_Up, TAKEPROFIT_Up))
                                                                 return(-1);
          if (ClosePos_Up)
                if (!CloseOrder1(BUY_Stop, 1))
                                        return(-1);
        }
     }

   //----++ CODE FOR SHORT POSITIONS
   if (Test_Dn)
    {
      int IBARS_Dn = iBars(NULL, Timeframe_Dn);

      if (IBARS_Dn >= MinBar_Dn)
       {
         if (LastBars_Dn != IBARS_Dn)
          {
           //----+ Initialization of variables
           SELL_Sign = false;
           SELL_Stop = false;
           LastBars_Dn = IBARS_Dn;

           //----+ CALCULATING INDICATOR VALUES AND UPLOADING THEM TO BUFFERS
           for(bar = 1; bar < 3; bar++)
                     MovA[bar - 1] =
                         iCustom(NULL, Timeframe_Dn,
                                "JJMA", LengthA_Dn, PhaseA_Dn,
                                                   0, IPCA_Dn, 0, bar);
           for(bar = 1; bar < 3; bar++)
                MovB[bar - 1] =
                   iCustom(NULL, Timeframe_Dn,
                     "JJMA", LengthA_Dn + LengthB_Dn, PhaseB_Dn,
                                                   0, IPCB_Dn, 0, bar);

           //----+ DEFINING SIGNALS FOR TRADES
           if ( MovA[1] > MovB[1])
               if ( MovA[0] < MovB[0])
                        SELL_Sign = true;

            if ( MovA[0] < MovB[0])
                        SELL_Stop = true;
          }
          //----+ EXECUTION OF TRADES
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

### Trading Systems Based on Crossing of Two Oscillators

The like trading strategy can be used not only with movings, but also with oscillators, but in this case, as written in the [previous article](https://www.mql5.com/en/articles/1517), it is better to place pending orders instead of entering the market immediately after receiving signals. As a vivid example MACD diagram can be used.

For placing pending BuyLimit orders the algorithm will be like this:

![](https://c.mql5.com/2/16/macd1.gif)

Here is the algorithm for orders of SellLimit type:

![](https://c.mql5.com/2/16/macd2.gif)

The code of this EA is similar to the code of the previous EA:

```
//+==================================================================+
//|                                                        Exp_7.mq4 |
//|                             Copyright © 2007,   Nikolay Kositsin |
//|                              Khabarovsk,   farria@mail.redcom.ru |
//+==================================================================+
#property copyright "Copyright © 2007, Nikolay Kositsin"
#property link "farria@mail.redcom.ru"
//----+ +---------------------------------------------------------------------------+
//---- EA INPUT PARAMETERS FOR BUY TRADES
extern bool   Test_Up = true; //filter of trade calculations direction
extern int    Timeframe_Up = 240;
extern double Money_Management_Up = 0.1;
extern int    FST_period_Up = 12;  // period of the quick moving
extern int    SLO_period_Up = 22; // period increment of slow moving to quick one
extern int    SIGN_period_Up = 8; // period of the signal line
extern int    Price_Up = 0;  // selecting prices, upon which MACD is calculated
extern int    STOPLOSS_Up = 50;  // stop loss
extern int    TAKEPROFIT_Up = 100; // take profit
extern int    PriceLevel_Up =40; // difference between the current price and
                                         // price of pending order triggering
extern bool   ClosePos_Up = true; // forced position closing allowed
//----+ +---------------------------------------------------------------------------+
//---- EA INPUT PARAMETERS FOR SELL TRADES
extern bool   Test_Dn = true; //filter of trade calculations direction
extern int    Timeframe_Dn = 240;
extern double Money_Management_Dn = 0.1;
extern int    FST_period_Dn = 12;  // period of the quick moving
extern int    SLO_period_Dn = 22; // period increment of slow moving to quick one
extern int    SIGN_period_Dn = 8; // period of the signal line
extern int    Price_Dn = 0;  // selecting prices, upon which MACD is calculated
extern int    STOPLOSS_Dn = 50;  // stop loss
extern int    TAKEPROFIT_Dn = 100; // take profit
extern int    PriceLevel_Dn =40;  // difference between the current price and
                                         // price of pending order triggering
extern bool   ClosePos_Dn = true; // forced position closing allowed
//----+ +---------------------------------------------------------------------------+
//---- Integer variables for the minimum of calculation bars
int MinBar_Up, MinBar_Dn;
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
   if (Timeframe_Up != 1)
    if (Timeframe_Up != 5)
     if (Timeframe_Up != 15)
      if (Timeframe_Up != 30)
       if (Timeframe_Up != 60)
        if (Timeframe_Up != 240)
         if (Timeframe_Up != 1440)
           Print(StringConcatenate("Parameter Timeframe_Up cannot ",
                                  "be equal to ", Timeframe_Up, "!!!"));
//---- Checking the correctness of Timeframe_Dn variable value
   if (Timeframe_Dn != 1)
    if (Timeframe_Dn != 5)
     if (Timeframe_Dn != 15)
      if (Timeframe_Dn != 30)
       if (Timeframe_Dn != 60)
        if (Timeframe_Dn != 240)
         if (Timeframe_Dn != 1440)
           Print(StringConcatenate("Parameter Timeframe_Dn cannot ",
                                 "be equal to ", Timeframe_Dn, "!!!"));
//---- Initialization of variables
   MinBar_Up = 4 + FST_period_Up + SLO_period_Up + SIGN_period_Up;
   MinBar_Dn = 4 + FST_period_Dn + SLO_period_Dn + SIGN_period_Dn;
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
   int    bar;
   double MovA[2], MovB[2];
   //----+ Declaring static variables
   static int LastBars_Up, LastBars_Dn;
   static datetime StopTime_Up, StopTime_Dn;
   static bool BUY_Sign, BUY_Stop, SELL_Sign, SELL_Stop;

   //----++ CODE FOR LONG POSITIONS
   if (Test_Up)
    {
      int IBARS_Up = iBars(NULL, Timeframe_Up);

      if (IBARS_Up >= MinBar_Up)
       {
         if (LastBars_Up != IBARS_Up)
          {
           //----+ Initialization of variables
           BUY_Sign = false;
           BUY_Stop = false;
           LastBars_Up = IBARS_Up;
           StopTime_Up = iTime(NULL, Timeframe_Up, 0)
                                            + 60 * Timeframe_Up;

           //----+ CALCULATING INDICATOR VALUES AND UPLOADING THEM TO BUFFERS
           for(bar = 1; bar < 3; bar++)
             MovA[bar - 1] = iMACD(NULL, Timeframe_Up,
                      FST_period_Up, FST_period_Up + SLO_period_Up,
                                         SIGN_period_Up, Price_Up, 0, bar);

           for(bar = 1; bar < 3; bar++)
             MovB[bar - 1] = iMACD(NULL, Timeframe_Up,
                      FST_period_Up, FST_period_Up + SLO_period_Up,
                                         SIGN_period_Up, Price_Up, 1, bar);

           //----+ DEFINING SIGNALS FOR TRADES
           if ( MovA[1] < MovB[1])
               if ( MovA[0] > MovB[0])
                        BUY_Sign = true;

            if ( MovA[0] > MovB[0])
                        BUY_Stop = true;
          }

          //----+ EXECUTION OF TRADES
          if (!OpenBuyLimitOrder1(BUY_Sign, 1,
              Money_Management_Up, STOPLOSS_Up, TAKEPROFIT_Up,
                                            PriceLevel_Up, StopTime_Up))
                                                                 return(-1);
          if (ClosePos_Up)
                if (!CloseOrder1(BUY_Stop, 1))
                                        return(-1);
        }
     }

   //----++ CODE FOR SHORT POSITIONS
   if (Test_Dn)
    {
      int IBARS_Dn = iBars(NULL, Timeframe_Dn);

      if (IBARS_Dn >= MinBar_Dn)
       {
         if (LastBars_Dn != IBARS_Dn)
          {
           //----+ Initialization of variables
           SELL_Sign = false;
           SELL_Stop = false;
           LastBars_Dn = IBARS_Dn;
           StopTime_Dn = iTime(NULL, Timeframe_Dn, 0)
                                            + 60 * Timeframe_Dn;

           //----+ CALCULATING INDICATOR VALUES AND UPLOADING THEM TO BUFFERS
           for(bar = 1; bar < 3; bar++)
             MovA[bar - 1] = iMACD(NULL, Timeframe_Dn,
                      FST_period_Dn, FST_period_Dn + SLO_period_Dn,
                                         SIGN_period_Dn, Price_Dn, 0, bar);

           for(bar = 1; bar < 3; bar++)
             MovB[bar - 1] = iMACD(NULL, Timeframe_Dn,
                      FST_period_Dn, FST_period_Dn + SLO_period_Dn,
                                         SIGN_period_Dn, Price_Dn, 1, bar);

           //----+ DEFINING SIGNALS FOR TRADES
           if ( MovA[1] > MovB[1])
               if ( MovA[0] < MovB[0])
                        SELL_Sign = true;

            if ( MovA[0] < MovB[0])
                        SELL_Stop = true;
          }
          //----+ EXECUTION OF TRADES
          if (!OpenSellLimitOrder1(SELL_Sign, 2,
              Money_Management_Dn, STOPLOSS_Dn, TAKEPROFIT_Dn,
                                            PriceLevel_Dn, StopTime_Dn))
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

### Conclusion

One more article is over. One more trading system was implemented in Expert Advisors based on absolutely different variants of indicators. I hope this article will be useful for beginning EA writers for further development of skills of converting a correctly formalized algorithm into a ready and absolutely operating code of your Expert Advisors.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1521](https://www.mql5.com/ru/articles/1521)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1521.zip "Download all attachments in the single ZIP archive")

[EXPERTS.zip](https://www.mql5.com/en/articles/download/1521/EXPERTS.zip "Download EXPERTS.zip")(8.58 KB)

[INCLUDE.zip](https://www.mql5.com/en/articles/download/1521/INCLUDE.zip "Download INCLUDE.zip")(17.88 KB)

[indicators.zip](https://www.mql5.com/en/articles/download/1521/indicators.zip "Download indicators.zip")(5.19 KB)

[TESTER.zip](https://www.mql5.com/en/articles/download/1521/TESTER.zip "Download TESTER.zip")(5.27 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/39446)**
(1)


![Leith Welsford](https://c.mql5.com/avatar/avatar_na2.png)

**[Leith Welsford](https://www.mql5.com/en/users/leithwelsford)**
\|
11 Feb 2017 at 12:52

Hi Nikolay,

I find you articles a great source of ideas and have been going through you mql4 contributions in detail - thank you...

Regarding the backtesting (IsBackTestingTime) automation though - you have referenced some source code I'd like to test, also writing the results to [test reports](https://www.mql5.com/en/articles/403 "Article ") (TestReport). I cannot find the source code for these functions and header files anywhere. Would it be possible to get access to this for testing please?

Also I see you're active with the mql5 trading system now - what is your opinion on the best tool to start with, mql4 or mql5???

Thank you for your contributions to this material...

Regards,

Leith

![A Non-Trading EA Testing Indicators](https://c.mql5.com/2/16/627_11.gif)[A Non-Trading EA Testing Indicators](https://www.mql5.com/en/articles/1534)

All indicators can be divided into two groups: static indicators, the displaying of which, once shown, always remains the same in history and does not change with new incoming quotes, and dynamic indicators that display their status for the current moment only and are fully redrawn when a new price comes. The efficiency of a static indicator is directly visible on the chart. But how can we check whether a dynamic indicator works ok? This is the question the article is devoted to.

![MetaEditor:Templates as a Spot to Stand On](https://c.mql5.com/2/15/573_135.jpg)[MetaEditor:Templates as a Spot to Stand On](https://www.mql5.com/en/articles/1514)

It may be news to many our readers that all preparations for writing an EA can be performed once and then used continuously.

![Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part IV)](https://c.mql5.com/2/15/595_130.gif)[Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part IV)](https://www.mql5.com/en/articles/1523)

In this article the author continues to analyze implementation algorithms of simplest trading systems and introduces recording of optimization results in backtesting into one html file in the form of a table. The article will be useful for beginning traders and EA writers.

![Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part II)](https://c.mql5.com/2/15/576_89.gif)[Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part II)](https://www.mql5.com/en/articles/1517)

In this article the author continues to analyze implementation algorithms of simplest trading systems and describes some relevant details of using optimization results. The article will be useful for beginning traders and EA writers.

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/1521&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049007590379201316)

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