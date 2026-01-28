---
title: Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part II)
url: https://www.mql5.com/en/articles/1517
categories: Trading Systems
relevance_score: 12
scraped_at: 2026-01-22T17:16:17.928690
---

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/1517&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049009398560432942)

MetaTrader 4 / Trading systems


### Introduction

Optimization of trading systems is widely discussed in various literature and it would be illogical to write here in details what can be easily found in the Internet. So here we will discuss in details only fundamental simple ideas underlying the logics of understanding the point of using the results of automated systems optimization generally and the practical usefulness of optimization particularly.

I suppose you know that it is easy to upload into a strategy tester an Expert Advisor constructed on the basis of even a relatively simple trading system, after its optimization you can get amazing testing results almost on any historic data that coincide with historic data of optimization. But here a natural question occurs: "What does the result have to do with the forecasting of an automated system behavior, even if the result is tremendous but has been achieved after adjusting the system according to history?"

At the moment this question is rather acute. One of the reasons for this is that after first successful optimizations, a beginning EA writer can get the wrong idea about a thoughtless use of optimization results in live trading and the consequences of it can be damaging. And the beginning EA writers can loose their own money or the money of those who will use the ready Expert Advisor. So I will start my article with this topic.

### Backtesting or Testing Optimization Results without Fanaticism

Usually after first experiences of EA optimization one can decide to construct a strategy of using in trading maximally profitable optimization results with minimal drawdown hoping that the system with the like set of parameters will be profitable not only in the optimization period, but also in the nearest future. Schematically it will look like this:

![](https://c.mql5.com/2/16/1_3.png)

This is the logics used by many beginning EA writers after the first acquaintance with a tester of trading strategies. But unfortunately we cannot dip into the future and define the efficiency of a trading system using this logic, and testing of a strategy on a demo account in the real-time mode in terms of a live market for a regular recording of its operation results is a very tiresome occupation!

If anyone is patient enough to wait for the result for years, then there are no problems! However, after that a system may become absolutely useless and it will be very disappointing to have so much wasted time and Internet traffic. Moreover, such an approach does not allow testing many trading strategies. One or, maximum, two! And the last thing - so many efforts invested into the testing of such a trading strategy deprives its creator of the ability to estimate critically its operation results!

Psychologically in such a situation it is quite difficult to face the truth and admit that the resulting strategy was just a mere waste of time and effort. Naturally, such waste of one's effort and time can hardly lead to the creation of a profitable Expert Advisor.

So, in this situation the only possible way out is modeling future on the basis of historic data. It is quite easy. All we need is shift the time borders from the previous scheme to the left along the time axis. In such a case after optimization you will have enough time for estimating the optimization results on the recent data that is closer to the present than the data of optimization.

![](https://c.mql5.com/2/16/2_3.png)

In many cases, including the one described above, such an approach allows estimating the actual efficiency of the like strategies of using optimization results for any certain Expert Advisor that has optimized parameters. Schematically the meaning of such analysis of optimization strategies can be presented in the following form:

![](https://c.mql5.com/2/16/3_7.png)

I suppose this diagram is quite clear. For example, we take the year 2007 as the analyzing period. For the first optimization the period of optimization will be from 01.01.2007 to 31.03.2007, and the testing period - from 01.04.2007 to 31.05.2007. After testing and recording the results, we shift optimization and testing periods by one month forward: optimization period will be from 01.02.2007 to 30.04.2007, and testing period - from 01.05.2007 to 30.06.2007. And so on. As a result we have a table for recording results of each optimization, for example this one:

| Final testing parameter/Run | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Net profit |  |  |  |  |  |  |  |  |  |  |  |  |
| Total profit |  |  |  |  |  |  |  |  |  |  |  |  |
| Total loss |  |  |  |  |  |  |  |  |  |  |  |  |
| Profitability |  |  |  |  |  |  |  |  |  |  |  |  |
| Absolute drawdown |  |  |  |  |  |  |  |  |  |  |  |  |
| Relative drawdown |  |  |  |  |  |  |  |  |  |  |  |  |
| Total trades |  |  |  |  |  |  |  |  |  |  |  |  |

Of course, you will have to fill in the table cells after each optimization run. It will not cause any large problems analyzing such a table and processing information contained in it, so you can easily draw conclusions of this analysis. I suppose, this kind of analysis of EA behavior on the available history data helps to estimate correctly the EA optimization results and avoid delusions regarding this process. And time saving as compared to testing on a demo account is fabulous!

The technique of back testing is quite easy-to-understand by beginning EA writers, though this investigation also requires time and effort. Even if you obtain poor results during you EA backtesting, don't get upset: virtual losses are much better that live using of lossmaking strategies that seemed to be strongly profitable.

This is all I wanted to tell you about backtesting. Note, selecting maximal surplus with the minimal drawdown is not the only possible system optimization strategy. It is offered to you only for the introduction of backtesting procedure. Generally, the most of time during backtesting is invested into optimization, while the testing of optimization strategies requires very little time, so it is more reasonable to test several strategies at a time to have larger statistical material for further conclusions. Consequently the table of testing results will be much larger.

### Oscillator Trading Strategy

On the basis of oscillators one can construct many different trading strategies. In this article I will describe the most popular system based on entrances and exits performed in overbought and oversold areas:

![](https://c.mql5.com/2/16/4_3.png)

The majority of oscillators change their values from a certain minimum to some maximum, each oscillator has its own values. At some distance from its extremums, levels UpLevel and DownLevel are placed.

In such a system a signal to buy occurs if the oscillator leaves the non-trend area and enters the overbought area:

![](https://c.mql5.com/2/16/osc1.gif)

A signal to sell occurs when the oscillator leaves the non-trend area and enters the oversold area:

![](https://c.mql5.com/2/16/osc3.gif)

Besides these main signals, the system also has additional signals that appear when the oscillator exits oversold and overbought areas and enters the non-trend area, the signals of the so called correction.

A signal to buy occurs when the oscillator exits the oversold area and enters the non-trend area (correction of a falling trend):

![](https://c.mql5.com/2/16/osc2.gif)

A signal to sell occurs when it exits the overbought area and enters the non-trend area (correction of a rising trend):

![](https://c.mql5.com/2/16/osc4.gif)

### Code of an Expert Advisor for the Oscillator Trading System

Thus we have four algorithms for entering the market. Looking attentively at the two variants of algorithms for long positions we can conclude that they are absolutely identical and differ only in the position of a breakout level, which is absolutely irrelevant from the point of view of program code writing.The situation is analogous for short positions. Here is my variant implementing the oscillator trading system:

```
//+==================================================================+
//|                                                        Exp_3.mq4 |
//|                             Copyright © 2007,   Nikolay Kositsin |
//|                              Khabarovsk,   farria@mail.redcom.ru |
//+==================================================================+
#property copyright "Copyright © 2007, Nikolay Kositsin"
#property link "farria@mail.redcom.ru"
//----+ +--------------------------------------------------------------------------+
//---- EA INPUT PARAMETERS FOR BUY TRADES
extern bool   Test_Up1 = true;//filter of trade calculations direction
extern int    Timeframe_Up1 = 240;
extern double Money_Management_Up1 = 0.1;
extern double IndLevel_Up1 = 0.8; // breakout level of the indicator
extern int    JLength_Up1 = 8;  // depth of JJMA smoothing of entering price
extern int    XLength_Up1 = 8;  // depth of JurX smoothing of obtained indicator
extern int    Phase_Up1 = 100; // parameter changing in the range -100 ... +100,
                              //influences the quality of transient processes of smoothing
extern int    IPC_Up1 = 0;/* Selecting prices on which the indicator will be calculated
(0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL, 6-WEIGHTED, 7-Heiken Ashi Close,
8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW, 11-Heiken Ashi Low, 12-Heiken Ashi High,
13-Heiken Ashi Open, 14-Heiken Ashi Close.) */
extern int    STOPLOSS_Up1 = 50;  // stoploss
extern int    TAKEPROFIT_Up1 = 100; // takeprofit
extern int    TRAILINGSTOP_Up1 = 0; // trailing stop
extern bool   ClosePos_Up1 = true; // forced position closing allowed
//----+ +--------------------------------------------------------------------------+
///---- EA INPUT PARAMETERS FOR BUY TRADES
extern bool   Test_Up2 = true;//filter of trade calculations direction
extern int    Timeframe_Up2 = 240;
extern double Money_Management_Up2 = 0.1;
extern double IndLevel_Up2 = -0.8; // breakout level of the indicator
extern int    JLength_Up2 = 8;  // depth of JJMA smoothing of entering price
extern int    XLength_Up2 = 8;  // depth of JurX smoothing of obtained indicator
extern int    Phase_Up2 = 100; // parameter changing in the range -100 ... +100,
                              //influences the quality of transient processes of smoothing
extern int    IPC_Up2 = 0;/* Selecting prices on which the indicator will be calculated
(0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL, 6-WEIGHTED, 7-Heiken Ashi Close,
8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW, 11-Heiken Ashi Low, 12-Heiken Ashi High,
13-Heiken Ashi Open, 14-Heiken Ashi Close.) */
extern int    STOPLOSS_Up2 = 50;  // stoploss
extern int    TAKEPROFIT_Up2 = 100; // takeprofit
extern int    TRAILINGSTOP_Up2 = 0; // trailing stop
extern bool   ClosePos_Up2 = true; // forced position closing allowed
//----+ +--------------------------------------------------------------------------+
//---- EA INPUT PARAMETERS FOR SELL TRADES
extern bool   Test_Dn1 = true;//filter of trade calculations direction
extern int    Timeframe_Dn1 = 240;
extern double Money_Management_Dn1 = 0.1;
extern double IndLevel_Dn1 = 0.8; // breakout level of the indicator
extern int    JLength_Dn1 = 8;  // depth of JJMA smoothing of entering price
extern int    XLength_Dn1 = 8;  // depth of JurX smoothing of obtained indicator
extern int    Phase_Dn1 = 100; // parameter changing in the range -100 ... +100,
                              //influences the quality of transient processes of smoothing
extern int    IPC_Dn1 = 0;/* Selecting prices on which the indicator will be calculated
(0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL, 6-WEIGHTED, 7-Heiken Ashi Close,
8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW, 11-Heiken Ashi Low, 12-Heiken Ashi High,
13-Heiken Ashi Open, 14-Heiken Ashi Close.) */
extern int    STOPLOSS_Dn1 = 50;  // stoploss
extern int    TAKEPROFIT_Dn1 = 100; // takeprofit
extern int    TRAILINGSTOP_Dn1 = 0; // trailing stop
extern bool   ClosePos_Dn1 = true; // forced position closing allowed
//----+ +--------------------------------------------------------------------------+
//---- EA INPUT PARAMETERS FOR SELL TRADES
extern bool   Test_Dn2 = true;//filter of trade calculations direction
extern int    Timeframe_Dn2 = 240;
extern double Money_Management_Dn2 = 0.1;
extern double IndLevel_Dn2 = -0.8; // breakout level of the indicator
extern int    JLength_Dn2 = 8;  // depth of JJMA smoothing of entering price
extern int    XLength_Dn2 = 8;  // depth of JurX smoothing of obtained indicator
extern int    Phase_Dn2 = 100; // parameter changing in the range -100 ... +100,
                              //influences the quality of transient processes of smoothing
extern int    IPC_Dn2 = 0;/* Selecting prices on which the indicator will be calculated
(0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL, 6-WEIGHTED, 7-Heiken Ashi Close,
8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW, 11-Heiken Ashi Low, 12-Heiken Ashi High,
13-Heiken Ashi Open, 14-Heiken Ashi Close.) */
extern int    STOPLOSS_Dn2 = 50;  // stoploss
extern int    TAKEPROFIT_Dn2 = 100; // takeprofit
extern int    TRAILINGSTOP_Dn2 = 0; // trailing stop
extern bool   ClosePos_Dn2 = true; // forced position closing allowed
//----+ +--------------------------------------------------------------------------+
//---- Integer variables for the minimum of calculation bars
int MinBar_Up1, MinBar_Dn1;
int MinBar_Up2, MinBar_Dn2;
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
//| Custom Expert functions                                          |
//+==================================================================+
#include <Lite_EXPERT1.mqh>
//+==================================================================+
//| Custom Expert initialization function                            |
//+==================================================================+
int init()
  {
//---- Checking the correctness of Timeframe_Up1 variable value
   TimeframeCheck("Timeframe_Up1", Timeframe_Up1);
//---- Checking the correctness of Timeframe_Up2 variable value
   TimeframeCheck("Timeframe_Up2", Timeframe_Up2);
//---- Checking the correctness of Timeframe_Dn1 variable value
   TimeframeCheck("Timeframe_Dn1", Timeframe_Dn1);
//---- Checking the correctness of Timeframe_Dn2 variable value
   TimeframeCheck("Timeframe_Dn2", Timeframe_Dn2);
//---- Initialization of variables
   MinBar_Up1 = 3 + 3 * XLength_Up1 + 30;
   MinBar_Up2 = 3 + 3 * XLength_Up2 + 30;
   MinBar_Dn1 = 3 + 3 * XLength_Dn1 + 30;
   MinBar_Dn2 = 3 + 3 * XLength_Dn2 + 30;
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
   double Osc1, Osc2;
   //----+ Declaring static variables
   //----+ +---------------------------------------------------------------+
   static int LastBars_Up1, LastBars_Dn1;
   static bool BUY_Sign1, BUY_Stop1, SELL_Sign1, SELL_Stop1;
   //----+ +---------------------------------------------------------------+
   static int LastBars_Up2, LastBars_Dn2;
   static bool BUY_Sign2, BUY_Stop2, SELL_Sign2, SELL_Stop2;
   //----+ +---------------------------------------------------------------+
   //----++ CODE FOR LONG POSITIONS 1
   if (Test_Up1)
    {
      int IBARS_Up1 = iBars(NULL, Timeframe_Up1);

      if (IBARS_Up1 >= MinBar_Up1)
       {
         if (LastBars_Up1 != IBARS_Up1)
          {
           //----+ Initialization of variables
           BUY_Sign1 = false;
           BUY_Stop1 = false;
           LastBars_Up1 = IBARS_Up1;

           //----+ CALCULATING INDICATOR VALUES AND UPLOADING THEM TO BUFFERS
           Osc1 = iCustom(NULL, Timeframe_Up1,
                 "JCCIX", JLength_Up1, XLength_Up1,
                                Phase_Up1, IPC_Up1, 0, 1);

           Osc2 = iCustom(NULL, Timeframe_Up1,
                 "JCCIX", JLength_Up1, XLength_Up1,
                                Phase_Up1, IPC_Up1, 0, 2);

           //----+ DEFINING SIGNALS FOR TRADES
           if (Osc2 < IndLevel_Up1)
             if (Osc1 > IndLevel_Up1)
                          BUY_Sign1 = true;

           if (Osc1 < IndLevel_Up1)
                          BUY_Stop1 = true;
          }
          //----+ EXECUTION OF TRADES
          if (!OpenBuyOrder1(BUY_Sign1, 1, Money_Management_Up1,
                                          STOPLOSS_Up1, TAKEPROFIT_Up1))
                                                                 return(-1);
          if (ClosePos_Up1)
                if (!CloseOrder1(BUY_Stop1, 1))
                                        return(-1);

          if (!Make_TreilingStop(1, TRAILINGSTOP_Up1))
                                                  return(-1);
        }
     }
   //----+ +---------------------------------------------------------------+
   //----++ CODE FOR LONG POSITIONS 2
   if (Test_Up2)
    {
      int IBARS_Up2 = iBars(NULL, Timeframe_Up2);

      if (IBARS_Up2 >= MinBar_Up2)
       {
         if (LastBars_Up2 != IBARS_Up2)
          {
           //----+ Initialization of variables
           BUY_Sign2 = false;
           BUY_Stop2 = false;
           LastBars_Up2 = IBARS_Up2;

           //----+ CALCULATING INDICATOR VALUES AND UPLOADING THEM TO BUFFERS
           Osc1 = iCustom(NULL, Timeframe_Up2,
                 "JCCIX", JLength_Up2, XLength_Up2,
                                Phase_Up2, IPC_Up2, 0, 1);

           Osc2 = iCustom(NULL, Timeframe_Up2,
                 "JCCIX", JLength_Up2, XLength_Up2,
                                Phase_Up2, IPC_Up2, 0, 2);

           //----+ DEFINING SIGNALS FOR TRADES
           if (Osc2 < IndLevel_Up2)
             if (Osc1 > IndLevel_Up2)
                          BUY_Sign2 = true;

           if (Osc1 < IndLevel_Up2)
                          BUY_Stop2 = true;
          }
          //----+ EXECUTION OF TRADES
          if (!OpenBuyOrder1(BUY_Sign2, 2, Money_Management_Up2,
                                          STOPLOSS_Up2, TAKEPROFIT_Up2))
                                                                 return(-1);
          if (ClosePos_Up2)
                if (!CloseOrder1(BUY_Stop2, 2))
                                        return(-1);

          if (!Make_TreilingStop(2, TRAILINGSTOP_Up2))
                                                  return(-1);
        }
     }
   //----+ +---------------------------------------------------------------+
   //----++ CODE FOR SHORT POSITIONS 1
   if (Test_Dn1)
    {
      int IBARS_Dn1 = iBars(NULL, Timeframe_Dn1);

      if (IBARS_Dn1 >= MinBar_Dn1)
       {
         if (LastBars_Dn1 != IBARS_Dn1)
          {
           //----+ Initialization of variables
           SELL_Sign1 = false;
           SELL_Stop1 = false;
           LastBars_Dn1 = IBARS_Dn1;

           //----+ CALCULATING INDICATOR VALUES AND UPLOADING THEM TO BUFFERS
           Osc1 = iCustom(NULL, Timeframe_Dn1,
                 "JCCIX", JLength_Dn1, XLength_Dn1,
                                Phase_Dn1, IPC_Dn1, 0, 1);

           Osc2 = iCustom(NULL, Timeframe_Dn1,
                 "JCCIX", JLength_Dn1, XLength_Dn1,
                                Phase_Dn1, IPC_Dn1, 0, 2);

           //----+ DEFINING SIGNALS FOR TRADES
           if (Osc2 > IndLevel_Dn1)
             if (Osc1 < IndLevel_Dn1)
                          SELL_Sign1 = true;

           if (Osc1 > IndLevel_Dn1)
                          SELL_Stop1 = true;
          }
          //----+ EXECUTION OF TRADES
          if (!OpenSellOrder1(SELL_Sign1, 3, Money_Management_Dn1,
                                          STOPLOSS_Dn1, TAKEPROFIT_Dn1))
                                                                 return(-1);
          if (ClosePos_Dn1)
                if (!CloseOrder1(SELL_Stop1, 3))
                                        return(-1);

          if (!Make_TreilingStop(3, TRAILINGSTOP_Dn1))
                                                  return(-1);
        }
     }
   //----+ +---------------------------------------------------------------+
   //----++ CODE FOR SHORT POSITIONS 2
   if (Test_Dn2)
    {
      int IBARS_Dn2 = iBars(NULL, Timeframe_Dn2);

      if (IBARS_Dn2 >= MinBar_Dn2)
       {
         if (LastBars_Dn2 != IBARS_Dn2)
          {
           //----+ Initialization of variables
           SELL_Sign2 = false;
           SELL_Stop2 = false;
           LastBars_Dn2 = IBARS_Dn2;

           //----+ CALCULATING INDICATOR VALUES AND UPLOADING THEM TO BUFFERS
           Osc1 = iCustom(NULL, Timeframe_Dn2,
                 "JCCIX", JLength_Dn2, XLength_Dn2,
                                Phase_Dn2, IPC_Dn2, 0, 1);

           Osc2 = iCustom(NULL, Timeframe_Dn2,
                 "JCCIX", JLength_Dn2, XLength_Dn2,
                                Phase_Dn2, IPC_Dn2, 0, 2);

           //----+ DEFINING SIGNALS FOR TRADES
           if (Osc2 > IndLevel_Dn2)
             if (Osc1 < IndLevel_Dn2)
                          SELL_Sign2 = true;

           if (Osc1 > IndLevel_Dn2)
                          SELL_Stop2 = true;
          }
          //----+ EXECUTION OF TRADES
          if (!OpenSellOrder1(SELL_Sign2, 4, Money_Management_Dn2,
                                          STOPLOSS_Dn2, TAKEPROFIT_Dn2))
                                                                 return(-1);
          if (ClosePos_Dn2)
                if (!CloseOrder1(SELL_Stop2, 4))
                                        return(-1);

           if (!Make_TreilingStop(4, TRAILINGSTOP_Dn2))
                                                  return(-1);
        }
     }
   //----+ +---------------------------------------------------------------+
//----+

    return(0);
  }
//+------------------------------------------------------------------+
```

This code is twice longer than the code presented in the previous article. By the way, imagine how long this EA code could be if it contained the full program code instead of calls of order management functions! It should be noted here that this EA contains new input parameters: IndLevel\_Up1, IndLevel\_Up2, IndLevel\_Dn1, IndLevel\_Dn2. IndLevel\_Up1 and IndLevel\_Up2 define the values of Uplevel and DownLevel for two algorithms of long positions, while IndLevel\_Dn1 and IndLevel\_Dn2 define values of DownLevel and Uplevel for two algorithms of short positions.

During the optimization of this Expert Advisor it should be taken into account that the value of these levels can vary from -1.0 to +1.0. If you want to substitute the JCCIX oscillator in this EA by any other oscillator, take into account that the maximal and minimal values of these levels can be different. The source indicator JCCIX is the analogue of the CCI indicator, in which the smoothing algorithm by common Moving Averages is substituted by JMA and ultralinear smoothing. The EA uses Trailing Stops, values of which are defined by the EA input parameters like TRAILINGSTOP\_Up1, TRAILINGSTOP\_Up2, TRAILINGSTOP\_Dn1, TRAILINGSTOP\_Dn2. As for all other terms, this EA is fully analogous to the EA described in the previous article.

### Changing Immediate Market Entering into Pending Orders

In many cases in EAs analogous to the one described above, changing the immediate market entering into pending orders allows making market entrances more precise and acquiring larger profit with less possibility of reaching Stop Loss. A set of functions from the file Lite\_EXPERT1.mqh allows making easily such a substitution.

All we need is substitute functions OpenBuyOrder1() and OpenSellOrder1() by OpenBuyLimitOrder1() and OpenSellLimitOrder1() respectively. During such function substitution new input variables of these functions must be initialized: LEVEL and Expiration. For example, we can build a trading strategy, in which the LEVEL variable will be defined by the EA input parameters. The date of pending order canceling can be set as the time of the next change of the current bar.

I will not repeat the same code here. The changed code of the above EA is attached to the article (Exp\_4.mq4). As an example of using pending orders I have described an oscillator system using the OSMA oscillator:

```
//For the EA operation the Metatrader\EXPERTS\indicators folder
//must contain the 5c_OsMA.mq4 indicator
//+==================================================================+
//|                                                        Exp_5.mq4 |
//|                             Copyright © 2008,   Nikolay Kositsin |
//|                              Khabarovsk,   farria@mail.redcom.ru |
//+==================================================================+
#property copyright "Copyright © 2008, Nikolay Kositsin"
#property link "farria@mail.redcom.ru"
//----+ +--------------------------------------------------------------------------+
//---- EA INPUT PARAMETERS FOR BUY TRADES
extern bool   Test_Up = true;//filter of trade calculations direction
extern int    Timeframe_Up = 240;
extern double Money_Management_Up = 0.1;
extern double IndLevel_Up = 0; // breakout level of the indicator
extern int    FastEMA_Up = 12;  // period of quick EMA
extern int    SlowEMA_Up = 26;  // period of slow EMA
extern int    SignalSMA_Up = 9;  // period of signal SMA
extern int    STOPLOSS_Up = 50;  // stoploss
extern int    TAKEPROFIT_Up = 100; // takeprofit
extern int    TRAILINGSTOP_Up = 0; // trailing stop
extern int    PriceLevel_Up =40; // difference between the current price and
                                         // the price of pending order triggering
extern bool   ClosePos_Up = true; // forced position closing allowed
//----+ +--------------------------------------------------------------------------+
//---- EA INPUT PARAMETERS FOR SELL TRADES
extern bool   Test_Dn = true;//filter of trade calculations direction
extern int    Timeframe_Dn = 240;
extern double Money_Management_Dn = 0.1;
extern double IndLevel_Dn = 0; // breakout level of the indicator
extern int    FastEMA_Dn = 12;  // period of quick EMA
extern int    SlowEMA_Dn = 26;  // period of slow EMA
extern int    SignalSMA_Dn = 9;  // period of signal SMA
extern int    STOPLOSS_Dn = 50;  // stoploss
extern int    TAKEPROFIT_Dn = 100; // takeprofit
extern int    TRAILINGSTOP_Dn = 0; // trailing stop
extern int    PriceLevel_Dn = 40; // difference between the current price and
                                         // the price of pending order triggering
extern bool   ClosePos_Dn = true; // forced position closing allowed
//----+ +--------------------------------------------------------------------------+
//---- Integer variables for the minimum of calculation bars
int MinBar_Up, MinBar_Dn;
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
//---- Checking the correctness of Timeframe_Dn variable value
   TimeframeCheck("Timeframe_Dn", Timeframe_Dn);
//---- Initialization of variables
   MinBar_Up  = 3 + MathMax(FastEMA_Up, SlowEMA_Up) + SignalSMA_Up;
   MinBar_Dn  = 3 + MathMax(FastEMA_Dn, SlowEMA_Dn) + SignalSMA_Dn;
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
   double Osc1, Osc2;
   //----+ Declaring static variables
   //----+ +---------------------------------------------------------------+
   static datetime StopTime_Up, StopTime_Dn;
   static int LastBars_Up, LastBars_Dn;
   static bool BUY_Sign1, BUY_Stop1, SELL_Sign1, SELL_Stop1;
   //----+ +---------------------------------------------------------------+
   //----++ CODE FOR LONG POSITIONS 1
   if (Test_Up)
    {
      int IBARS_Up = iBars(NULL, Timeframe_Up);

      if (IBARS_Up >= MinBar_Up)
       {
         if (LastBars_Up != IBARS_Up)
          {
           //----+ Initialization of variables
           BUY_Sign1 = false;
           BUY_Stop1 = false;
           LastBars_Up = IBARS_Up;
           StopTime_Up = iTime(NULL, Timeframe_Up, 0)
                                            + 60 * Timeframe_Up;

           //----+ CALCULATING INDICATOR VALUES AND UPLOADING THEM TO BUFFERS
           Osc1 = iCustom(NULL, Timeframe_Up,
                         "5c_OsMA", FastEMA_Up, SlowEMA_Up,
                                               SignalSMA_Up, 5, 1);

           Osc2 = iCustom(NULL, Timeframe_Up,
                         "5c_OsMA", FastEMA_Up, SlowEMA_Up,
                                               SignalSMA_Up, 5, 2);

           //----+ DEFINING SIGNALS FOR TRADES
           if (Osc2 < IndLevel_Up)
             if (Osc1 > IndLevel_Up)
                          BUY_Sign1 = true;

           if (Osc1 < IndLevel_Up)
                          BUY_Stop1 = true;
          }
          //----+ EXECUTION OF TRADES
          if (!OpenBuyLimitOrder1(BUY_Sign1, 1,
              Money_Management_Up, STOPLOSS_Up, TAKEPROFIT_Up,
                                            PriceLevel_Up, StopTime_Up))
                                                                 return(-1);
          if (ClosePos_Up)
                if (!CloseOrder1(BUY_Stop1, 1))
                                        return(-1);

          if (!Make_TreilingStop(1, TRAILINGSTOP_Up))
                                                  return(-1);
        }
     }
   //----+ +---------------------------------------------------------------+
   //----++ CODE FOR SHORT POSITIONS 1
   if (Test_Dn)
    {
      int IBARS_Dn = iBars(NULL, Timeframe_Dn);

      if (IBARS_Dn >= MinBar_Dn)
       {
         if (LastBars_Dn != IBARS_Dn)
          {
           //----+ Initialization of variables
           SELL_Sign1 = false;
           SELL_Stop1 = false;
           LastBars_Dn = IBARS_Dn;
           StopTime_Dn = iTime(NULL, Timeframe_Dn, 0)
                                            + 60 * Timeframe_Dn;

           //----+ CALCULATING INDICATOR VALUES AND UPLOADING THEM TO BUFFERS
           Osc1 = iCustom(NULL, Timeframe_Dn,
                         "5c_OsMA", FastEMA_Dn, SlowEMA_Dn,
                                               SignalSMA_Dn, 5, 1);

           Osc2 = iCustom(NULL, Timeframe_Dn,
                         "5c_OsMA", FastEMA_Dn, SlowEMA_Dn,
                                               SignalSMA_Dn, 5, 2);

           //----+ DEFINING SIGNALS FOR TRADES
           if (Osc2 > IndLevel_Dn)
             if (Osc1 < IndLevel_Dn)
                          SELL_Sign1 = true;

           if (Osc1 > IndLevel_Dn)
                          SELL_Stop1 = true;
          }
          //----+ EXECUTION OF TRADES
          if (!OpenSellLimitOrder1(SELL_Sign1, 2,
              Money_Management_Dn, STOPLOSS_Dn, TAKEPROFIT_Dn,
                                            PriceLevel_Dn, StopTime_Dn))
                                                                 return(-1);
          if (ClosePos_Dn)
                if (!CloseOrder1(SELL_Stop1, 2))
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

Instead of two breakout levels UpLevel and DownLevel, one level is used, that is why the EA contains only two position managing algorithms. Usually in a trading system connected with OsMA this level is selected equal to zero, but I decided to leave it in external variables of the EA, so it can be changed. I.e. it should be taken into account that OSMA indicator does not have maximum and minimum within which the indicator changes, consequently the breakout level does not have limitations of its values. Though, as I've said, usually it is equal to zero. To define the time of pending order canceling, static variables StopTime\_Up and StopTime\_Dn are used, once at bar changing they are initialized by the time of the next bar changing.

### Conclusion

In the conclusion I'd like to add that oscillator trading systems give many false signals against the current market trend. So such EAs are better included into operation in periods of market flat or used for opening positions only by trend.

As for backtesting, I can say once again that this is probably the best way for a beginning EA writer to estimate correctly optimization results. And there are no problems with EAs that show tremendous results at the adjustment to history. However, it is more difficult to understand, how ready EAs should be used to avoid situations when market goes far from optimized EA parameters.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1517](https://www.mql5.com/ru/articles/1517)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1517.zip "Download all attachments in the single ZIP archive")

[EXPERTS.zip](https://www.mql5.com/en/articles/download/1517/EXPERTS.zip "Download EXPERTS.zip")(7.48 KB)

[INCLUDE.zip](https://www.mql5.com/en/articles/download/1517/INCLUDE.zip "Download INCLUDE.zip")(24.26 KB)

[indicators.zip](https://www.mql5.com/en/articles/download/1517/indicators.zip "Download indicators.zip")(5.59 KB)

[TESTER.zip](https://www.mql5.com/en/articles/download/1517/TESTER.zip "Download TESTER.zip")(5.28 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/39445)**

![MetaEditor:Templates as a Spot to Stand On](https://c.mql5.com/2/15/573_135.jpg)[MetaEditor:Templates as a Spot to Stand On](https://www.mql5.com/en/articles/1514)

It may be news to many our readers that all preparations for writing an EA can be performed once and then used continuously.

![Fallacies, Part 1: Money Management is Secondary and Not Very Important](https://c.mql5.com/2/15/601_23.gif)[Fallacies, Part 1: Money Management is Secondary and Not Very Important](https://www.mql5.com/en/articles/1526)

The first demonstration of testing results of a strategy based on 0.1 lot is becoming a standard de facto in the Forum. Having received "not so bad" from professionals, a beginner sees that "0.1" testing brings rather modest results and decides to introduce an aggressive money management thinking that positive mathematic expectation automatically provides positive results. Let's see what results can be achieved. Together with that we will try to construct several artificial balance graphs that are very instructive.

![Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part III)](https://c.mql5.com/2/15/584_49.gif)[Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part III)](https://www.mql5.com/en/articles/1521)

In this article the author continues to analyze implementation algorithms of simplest trading systems and introduces backtesting automation. The article will be useful for beginning traders and EA writers.

![Comparative Analysis of 30 Indicators and Oscillators](https://c.mql5.com/2/15/577_13.gif)[Comparative Analysis of 30 Indicators and Oscillators](https://www.mql5.com/en/articles/1518)

The article describes an Expert Advisor that allows conducting the comparative analysis of 30 indicators and oscillators aiming at the formation of an effective package of indexes for trading.

[![](https://www.mql5.com/ff/si/0nfwvn6yhmgzf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F117%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dorder.expert%26utm_content%3Dorder.freelance%26utm_campaign%3D0622.MQL5.com.Internal&a=tunpwtbhegzufrqocbwiszessdutnobs&s=d9e7484e15300021b4066b1df77a94a1352f9e7c326d5113006bb4f6476bafeb&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=fhsctsdtltfeybejhpijyctxqroyovpa&ssn=1769091376246420825&ssn_dr=0&ssn_sr=0&fv_date=1769091376&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1517&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Expert%20Advisors%20Based%20on%20Popular%20Trading%20Systems%20and%20Alchemy%20of%20Trading%20Robot%20Optimization%20(Part%20II)%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176909137661615598&fz_uniq=5049009398560432942&sv=2552)

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