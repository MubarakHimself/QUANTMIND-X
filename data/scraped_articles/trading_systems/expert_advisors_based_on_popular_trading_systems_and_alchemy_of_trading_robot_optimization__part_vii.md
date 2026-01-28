---
title: Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part VII)
url: https://www.mql5.com/en/articles/1542
categories: Trading Systems
relevance_score: 12
scraped_at: 2026-01-22T17:15:35.433182
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/1542&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049001328316883716)

MetaTrader 4 / Trading systems


### Introduction

Thus, MetaQuotes Software Corp. opened registration for participating in the Automated Trading Championship 2008 on the 1st of July 2008! It would be a bit inconsequential of me to miss such an opportunity and not to continue the series of my articles with the representation of the logic of building an EA that would formally meet all Rules of the Automated Trading Championship 2008 and that wouldn't admit any critical error during the contest, for which it could be disqualified!

It is quite natural that for realization of this event I am using the simplest algorithms of opening positions, the trade factors of which will not be really interesting to us in the context of this article, while the most elementary little things may become most important for searching examination, because they can put one's participation in the Championship forward to several years later!

### General Idea of Writing an EA

In my opinion, it would be most demonstrative, in our case, to sketch such an Expert Advisor with a detailed description of its construction that actually provides its correct behavior in its interaction with the trade server. The Rules of the Championship determine the amount of open positions and pending orders placed simultaneously to be equal to three. So it would be reasonable to use three strategies in one EA, one strategy per position.

We will use the same algorithms with different parameters for opening long and short positions, while making these algorithms open only one position at a time requires from us assigning them the same magic numbers. Thus, we will have six algorithms to enter the market and only three magic numbers! As an entering algorithm, I use a trading system based on change the moving average direction from my [very first article](https://www.mql5.com/en/articles/1516) in this series. To make the algorithm differ from each other, I use different moving averages in them!

### Expert Advisor Code

Below is a version of the EA code:

```
//+==================================================================+
//|                                                 Exp_16_Champ.mq4 |
//|                             Copyright © 2008,   Nikolay Kositsin |
//|                              Khabarovsk,   farria@mail.redcom.ru |
//+==================================================================+
#property copyright "Copyright © 2008, Nikolay Kositsin"
#property link "farria@mail.redcom.ru"
//----+ +-----------------------------------------------------------------------+
//---- EXPERT ADVISOR'S INPUTS FOR BUY TRADES
extern bool   Test_Up1 = true;//a filter for trades calculation direction
extern int    Timeframe_Up1 = 60;
extern double Money_Management_Up1 = 0.1;
extern int    Length_Up1 = 4;  // smoothing depth
extern int    Phase_Up1 = 100; // parameter ranging within
          //-100 ... +100, it affects the quality of the transient process;
extern int    IPC_Up1 = 0;/* Choosing prices to calculate
the indicator on (0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL,
6-WEIGHTED, 7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close.) */
extern int    STOPLOSS_Up1 = 50;  // StopLoss
extern int    TAKEPROFIT_Up1 = 100; // TakeProfit
extern bool   ClosePos_Up1 = true; // enable forcible closing the position
//----+ +-----------------------------------------------------------------------+
//---- EXPERT ADVISOR'S INPUTS FOR SELL TRADES
extern bool   Test_Dn1 = true;//a filter for trades calculation direction
extern int    Timeframe_Dn1 = 60;
extern double Money_Management_Dn1 = 0.1;
extern int    Length_Dn1 = 4;  // smoothing depth
extern int    Phase_Dn1 = 100; // parameter ranging within
         // -100 ... +100, it affects the quality of the transient process;
extern int    IPC_Dn1 = 0;/* Choosing prices to calculate
the indicator on (0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL,
6-WEIGHTED, 7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close.) */
extern int   STOPLOSS_Dn1 = 50;  // StopLoss
extern int   TAKEPROFIT_Dn1 = 100; // TakeProfit
extern bool   ClosePos_Dn1 = true; // enable forcible closing the position
//----+ +-----------------------------------------------------------------------+
//---- EXPERT ADVISOR'S INPUTS FOR BUY TRADES
extern bool   Test_Up2 = true;//a filter for trades calculation direction
extern int    Timeframe_Up2 = 60;
extern double Money_Management_Up2 = 0.1;
extern int    Length1_Up2 = 4;  // first smoothing depth
extern int    Phase1_Up2 = 100; // parameter of the first smoothing,
       //ranging within -100 ... +100, it affects the quality
       //of the averaging transient;
extern int    Length2_Up2 = 4;  // second smoothing depth
extern int    Phase2_Up2 = 100; // parameter of the second smoothing,
       //ranging within -100 ... +100, it affects the quality
       //of the averaging transient;
extern int    IPC_Up2 = 0;/* Choosing prices to calculate
the indicator on (0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL,
6-WEIGHTED, 7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close.) */
extern int    STOPLOSS_Up2 = 50;  // StopLoss
extern int    TAKEPROFIT_Up2 = 100; // TakeProfit
extern bool   ClosePos_Up2 = true; // enable forcible closing the position
//----+ +-----------------------------------------------------------------------+
//---- EXPERT ADVISOR'S INPUTS FOR SELL TRADES
extern bool   Test_Dn2 = true;//a filter for trades calculation direction
extern int    Timeframe_Dn2 = 60;
extern double Money_Management_Dn2 = 0.1;
extern int    Length1_Dn2 = 4;  // smoothing depth
extern int    Phase1_Dn2 = 100;  // parameter of the first smoothing,
       //ranging within -100 ... +100, it affects the quality
       //of the averaging transient;
extern int    Length2_Dn2 = 4;  // smoothing depth
extern int    Phase2_Dn2 = 100; // parameter of the second smoothing,
       //ranging within -100 ... +100, it affects the quality
       //of the averaging transient;
extern int    IPC_Dn2 = 0;/* Choosing prices to calculate
the indicator on (0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL,
6-WEIGHTED, 7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close.) */
extern int   STOPLOSS_Dn2 = 50;  // StopLoss
extern int   TAKEPROFIT_Dn2 = 100; // TakeProfit
extern bool   ClosePos_Dn2 = true; // enable forcible closing the position
//----+ +-----------------------------------------------------------------------+
//---- EXPERT ADVISOR'S INPUTS FOR BUY TRADES
extern bool   Test_Up3 = true;//a filter for trades calculation direction
extern int    Timeframe_Up3 = 60;
extern double Money_Management_Up3 = 0.1;
extern int    Period_Up3 = 10;  // LSMA period
extern int    Length_Up3 = 4;  // smoothing depth
extern int    Phase_Up3 = 100; // parameter of smoothing,
       //ranging within -100 ... +100, it affects the quality
       //of the averaging transient;
extern int    IPC_Up3 = 0;/* Choosing prices to calculate
the indicator on (0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL,
6-WEIGHTED, 7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close.) */
extern int    STOPLOSS_Up3 = 50;  // StopLoss
extern int    TAKEPROFIT_Up3 = 100; // TakeProfit
extern bool   ClosePos_Up3 = true; // enable forcible closing the position
//----+ +-----------------------------------------------------------------------+
//---- EXPERT ADVISOR'S INPUTS FOR SELL TRADES
extern bool   Test_Dn3 = true;//a filter for trades calculation direction
extern int    Timeframe_Dn3 = 60;
extern double Money_Management_Dn3 = 0.1;
extern int    Period_Dn3 = 10;  // LSMA period
extern int    Length_Dn3 = 4;  // smoothing depth
extern int    Phase_Dn3 = 100;  // parameter smoothing,
       //ranging within -100 ... +100, it affects the quality
       //of the averaging transient;
extern int    IPC_Dn3 = 0;/* Choosing prices to calculate
the indicator on (0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL,
6-WEIGHTED, 7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close.) */
extern int   STOPLOSS_Dn3 = 50;  // StopLoss
extern int   TAKEPROFIT_Dn3 = 100; // TakeProfit
extern bool   ClosePos_Dn3 = true; // enable forcible closing the position
//----+ +-----------------------------------------------------------------------+

//---- Integer variables for the minimum of estimated bars
int MinBar_Up1, MinBar_Up2, MinBar_Up3;
int MinBar_Dn1, MinBar_Dn2, MinBar_Dn3;
//+==================================================================+
//| Custom Expert functions                                          |
//+==================================================================+
#include <Lite_EXPERT_Champ.mqh>
//+==================================================================+
//| TimeframeCheck() functions                                       |
//+==================================================================+
void TimeframeCheck(string Name, int Timeframe)
  {
//----+
   //---- Checking the value of the 'Timeframe' variable for correctness
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
//---- Checking the values of short-position timeframe variables for correctness
   TimeframeCheck("Timeframe_Up1", Timeframe_Up1);
   TimeframeCheck("Timeframe_Up2", Timeframe_Up2);
   TimeframeCheck("Timeframe_Up3", Timeframe_Up3);

//---- Checking the values of long-position timeframe variables for correctness
   TimeframeCheck("Timeframe_Dn1", Timeframe_Dn1);
   TimeframeCheck("Timeframe_Dn2", Timeframe_Dn2);
   TimeframeCheck("Timeframe_Dn3", Timeframe_Dn3);

//---- Initialization of variables
   MinBar_Up1 = 4 + 39 + 30;
   MinBar_Up2 = 4 + 30;
   MinBar_Up3 = 4 + Period_Up3 + 30;

   MinBar_Dn1 = 4 + 39 + 30;
   MinBar_Dn2 = 4 + 30;
   MinBar_Dn3 = 4 + Period_Dn3 + 30;
//---- initialization complete
   return(0);
  }
//+==================================================================+
//| expert deinitialization function                                 |
//+==================================================================+
int deinit()
  {
//----+

    //---- Expert Advisor initialization complete
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
   double Mov[3], dMov12, dMov23;
   //----+ Declaration of static variables
   static int LastBars_Up1, LastBars_Dn1;
   static int LastBars_Up2, LastBars_Dn2;
   static int LastBars_Up3, LastBars_Dn3;

   static bool BUY_Sign1, BUY_Stop1, SELL_Sign1, SELL_Stop1;
   static bool BUY_Sign2, BUY_Stop2, SELL_Sign2, SELL_Stop2;
   static bool BUY_Sign3, BUY_Stop3, SELL_Sign3, SELL_Stop3;

   //+------------------------------------------------------------------------+

   //----++ CODE FOR LONG POSITIONS
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

           //----+ CALCULATING THE INDICATORS' VALUES AND LOADING THEM TO BUFFERS
           for(bar = 1; bar <= 3; bar++)
                     Mov[bar - 1]=
                         iCustom(NULL, Timeframe_Up1,
                                "JFatl", Length_Up1, Phase_Up1,
                                                   0, IPC_Up1, 0, bar);

           //----+ DETERMINING SIGNALS FOR TRADES
           dMov12 = Mov[0] - Mov[1];
           dMov23 = Mov[1] - Mov[2];

           if (dMov23 < 0)
              if (dMov12 > 0)
                        BUY_Sign1 = true;

           if (dMov12 < 0)
                        BUY_Stop1 = true;
          }
          //----+ MAKING TRADES
          if (!OpenBuyOrder_Ch(BUY_Sign1, 1, Money_Management_Up1,
                                          STOPLOSS_Up1, TAKEPROFIT_Up1))
                                                                 return(-1);
          if (ClosePos_Up1)
                if (!CloseOrder_Ch(BUY_Stop1, 1))
                                        return(-1);
        }
     }

   //----++ CODE FOR SHORT POSITIONS
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

           //----+ CALCULATING THE INDICATORS' VALUES AND LOADING THEM TO BUFFERS
           for(bar = 1; bar <= 3; bar++)
                     Mov[bar - 1]=
                         iCustom(NULL, Timeframe_Dn1,
                                "JFatl", Length_Dn1, Phase_Dn1,
                                                   0, IPC_Dn1, 0, bar);

           //----+ DETERMINING SIGNALS FOR TRADES
           dMov12 = Mov[0] - Mov[1];
           dMov23 = Mov[1] - Mov[2];

           if (dMov23 > 0)
              if (dMov12 < 0)
                       SELL_Sign1 = true;

           if (dMov12 > 0)
                       SELL_Stop1 = true;
          }
          //----+ MAKING TRADES
          if (!OpenSellOrder_Ch(SELL_Sign1, 1, Money_Management_Dn1,
                                            STOPLOSS_Dn1, TAKEPROFIT_Dn1))
                                                                   return(-1);
          if (ClosePos_Dn1)
                if (!CloseOrder_Ch(SELL_Stop1, 1))
                                          return(-1);
        }
     }
    //+------------------------------------------------------------------------+
    //----++ CODE FOR LONG POSITIONS
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

           //----+ CALCULATING THE INDICATORS' VALUES AND LOADING THEM TO BUFFERS
           for(bar = 1; bar <= 3; bar++)
                     Mov[bar - 1]=
                         iCustom(NULL, Timeframe_Up2,
                                "J2JMA", Length1_Up2, Length2_Up2,
                                             Phase1_Up2, Phase2_Up2,
                                                  0, IPC_Up2, 0, bar);

           //----+ DETERMINING SIGNALS FOR TRADES
           dMov12 = Mov[0] - Mov[1];
           dMov23 = Mov[1] - Mov[2];

           if (dMov23 < 0)
              if (dMov12 > 0)
                        BUY_Sign2 = true;

           if (dMov12 < 0)
                        BUY_Stop2 = true;
          }
          //----+ MAKING TRADES
          if (!OpenBuyOrder_Ch(BUY_Sign2, 2, Money_Management_Up2,
                                          STOPLOSS_Up2, TAKEPROFIT_Up2))
                                                                 return(-1);
          if (ClosePos_Up2)
                if (!CloseOrder_Ch(BUY_Stop2, 2))
                                          return(-1);
        }
     }

   //----++ CODE FOR SHORT POSITIONS
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

           //----+ CALCULATING THE INDICATORS' VALUES AND LOADING THEM TO BUFFERS
           for(bar = 1; bar <= 3; bar++)
                     Mov[bar - 1]=
                         iCustom(NULL, Timeframe_Dn2,
                                "J2JMA", Length1_Dn2, Length2_Dn2,
                                               Phase1_Dn2, Phase2_Dn2,
                                                   0, IPC_Dn2, 0, bar);

           //----+ DETERMINING SIGNALS FOR TRADES
           dMov12 = Mov[0] - Mov[1];
           dMov23 = Mov[1] - Mov[2];

           if (dMov23 > 0)
              if (dMov12 < 0)
                       SELL_Sign2 = true;

           if (dMov12 > 0)
                       SELL_Stop2 = true;
          }
          //----+ MAKING TRADES
          if (!OpenSellOrder_Ch(SELL_Sign2, 2, Money_Management_Dn2,
                                            STOPLOSS_Dn2, TAKEPROFIT_Dn2))
                                                                   return(-1);
          if (ClosePos_Dn2)
                if (!CloseOrder_Ch(SELL_Stop2, 2))
                                           return(-1);
        }
     }
    //+------------------------------------------------------------------------+
    //----++ CODE FOR LONG POSITIONS
   if (Test_Up3)
    {
      int IBARS_Up3 = iBars(NULL, Timeframe_Up3);

      if (IBARS_Up3 >= MinBar_Up3)
       {
         if (LastBars_Up3 != IBARS_Up3)
          {
           //----+ Initialization of variables
           BUY_Sign3 = false;
           BUY_Stop3 = false;
           LastBars_Up3 = IBARS_Up3;

           //----+ CALCULATING THE INDICATORS' VALUES AND LOADING THEM TO BUFFERS
           for(bar = 1; bar <= 3; bar++)
                     Mov[bar - 1]=
                         iCustom(NULL, Timeframe_Up3,
                              "JLSMA", Period_Up3, Length_Up3, Phase_Up3,
                                                         0, IPC_Up3, 0, bar);

           //----+ DETERMINING SIGNALS FOR TRADES
           dMov12 = Mov[0] - Mov[1];
           dMov23 = Mov[1] - Mov[2];

           if (dMov23 < 0)
              if (dMov12 > 0)
                        BUY_Sign3 = true;

           if (dMov12 < 0)
                        BUY_Stop3 = true;
          }
          //----+ MAKING TRADES
          if (!OpenBuyOrder_Ch(BUY_Sign3, 3, Money_Management_Up3,
                                          STOPLOSS_Up3, TAKEPROFIT_Up3))
                                                                 return(-1);
          if (ClosePos_Up3)
                if (!CloseOrder_Ch(BUY_Stop3, 3))
                                          return(-1);
        }
     }

   //----++ CODE FOR SHORT POSITIONS
   if (Test_Dn3)
    {
      int IBARS_Dn3 = iBars(NULL, Timeframe_Dn3);

      if (IBARS_Dn3 >= MinBar_Dn3)
       {
         if (LastBars_Dn3 != IBARS_Dn3)
          {
           //----+ Initialization of variables
           SELL_Sign3 = false;
           SELL_Stop3 = false;
           LastBars_Dn3 = IBARS_Dn3;

           //----+ CALCULATING THE INDICATORS' VALUES AND LOADING THEM TO BUFFERS
           for(bar = 1; bar <= 3; bar++)
                     Mov[bar - 1]=
                         iCustom(NULL, Timeframe_Dn3,
                              "JLSMA", Period_Dn3, Length_Dn3, Phase_Dn3,
                                                         0, IPC_Dn3, 0, bar);

           //----+ DETERMINING SIGNALS FOR TRADES
           dMov12 = Mov[0] - Mov[1];
           dMov23 = Mov[1] - Mov[2];

           if (dMov23 > 0)
              if (dMov12 < 0)
                       SELL_Sign3 = true;

           if (dMov12 > 0)
                       SELL_Stop3 = true;
          }
          //----+ MAKING TRADES
          if (!OpenSellOrder_Ch(SELL_Sign3, 3, Money_Management_Dn3,
                                            STOPLOSS_Dn3, TAKEPROFIT_Dn3))
                                                                   return(-1);
          if (ClosePos_Dn3)
                if (!CloseOrder_Ch(SELL_Stop3, 3))
                                          return(-1);
        }
     }
    //+------------------------------------------------------------------------+
//----+

    return(0);
  }
//+------------------------------------------------------------------+
```

Those who want to have an EA without the limitations stated by the Championship can go in for Exp\_16.mq4. In fact, we can see the diversification of trading strategies among three algorithms differing from each other in a certain way. Generally, we have three absolutely independent automated trading strategies in one Expert Advisor. When writing the code, I modified the names of variables in each ATS to avoid coincidences.

To make trades, I used functions similar to those in file Lite\_EXPERT1.mqh, which are represented by file Lite\_EXPERT\_Champ.mqh. These functions required minimal changes and corrections in their codes to meet the requirements of the Championship. Alone using the code of such functions in the codes of other Participants' EAs is quite legal, since they are just executive elements of the EA, which have no relation to its intellectual filling that actually controls these elements!

So there is no special need to get into all details of these functions or creating something similar and still your own. It would be quite sufficient to read my previous articles in this series to be able to use them. Generally speaking, the use of such functions in writing EAs is as effective as the use of chips in developing and manufacturing electronic devices.

Below is a brief description of what has been considered in the construction of these functions.

1\. All functions for opening positions and placing pending orders detect the amount of already opened positions and placed pending orders and, if their amount exceeds two, don't take any actions.

2\. The minimal distance from the order open price, at which functions OpenBuyOrder\_Ch(), OpenSellOrder\_Ch(), OpenBuyLimitOrder\_Ch(), OpenBuyStopOrder\_Ch(), OpenSellLimitOrder\_Ch() and OpenSellStopOrder\_Ch() place TakeProfit, is always larger than that defined by the Championship Rules as a scalping one. The minimal distance fo StopLoss is determined by the properties of the traded symbol and is requested from the server. This also relates to pending orders. However, you should consider the fact that pending orders are sometimes processed at prices that turn out to be worse than the requested ones. Your TakeProfits may get into the 'range' of a scalping strategy, in this case! So it would be better to stay away from scalping strategies. Otherwise, at a fair number of gaps (which is quite possible at the end of year), you can find out that your EA has directly come under the definition of a scalping strategy!

However, it would be instructive to recall that, if TakeProfits are too large, the EA may make too few trades, which may cause its disqualification, too!

3\. The smallest and the largest size of a position to be opened, as well as the minimal step of changes, determined by the Championship Rules are written in all these functions with the values of the variables to be initialized. So such errors have already been avoided!

4\. Prior to opening a position, functions OpenBuyOrder\_Ch() and OpenSellOrder\_Ch() check the size of this position and check the position itself for the presence of enough money on the deposit for such a size, and reduce the amount of lots to acceptable values. So, when working with this functions, your EA is safe from errors like "invalid trade volume", in any case. Unfortunately, it is impossible to correct the lot size in pending orders in this manner, since it is impossible to forecast the amount of free assets on the deposit as of the moment the pending order triggers. So an EA writer must be very attentive when initializing external variables 'Money\_Management' of functions OpenBuyLimitOrder\_Ch(), OpenBuyStopOrder\_Ch(), OpenSellLimitOrder\_Ch() and OpenSellStopOrder\_Ch(), especially for the large values of StopLosses.

5\. All functions for positions management hold correct pauses between trades according to the codes of errors occurring.

6\. Prior to closing a position or deleting a pending order, or moving the StopLosses, functions CloseOrder\_Ch(), CloseOrder\_Ch() and Make\_TreilingStop\_Ch() check the position for being freezed and, if it is freezed, they don't take any actions.

7\. Prior to closing a position, function CloseOrder\_Ch() checks its net surplus for not being scalping-like. If the position turns out to be within the scalping range, it does not take any actions.

8\. Function Make\_TreilingStop\_Ch() does not move the StopLoss into the price range within which the profit of the position closed by this StopLoss may get into the 'scalping range'.

### Conclusion

Well, there is all there is to what I wanted to tell about the exemplary behavior of an EA on the Automated Trading Championship. Of course, there is one more, rather actual problem that relates to the EA's luxury consumption of CPU resources. However, in most cases, this problem often depends on ineffectively written indicators the EA calls during its operations. And this is an absolutely different pair of shoes.

Translated from Russian by MetaQuotes Software Corp.

Original article: [http://articles.mql4.com/ru/articles/1542](https://www.mql5.com/ru/articles/1542)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1542](https://www.mql5.com/ru/articles/1542)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1542.zip "Download all attachments in the single ZIP archive")

[EXPERTS.zip](https://www.mql5.com/en/articles/download/1542/EXPERTS.zip "Download EXPERTS.zip")(7.42 KB)

[INCLUDE.zip](https://www.mql5.com/en/articles/download/1542/INCLUDE.zip "Download INCLUDE.zip")(20.82 KB)

[indicators.zip](https://www.mql5.com/en/articles/download/1542/indicators.zip "Download indicators.zip")(7.85 KB)

[TESTER.zip](https://www.mql5.com/en/articles/download/1542/TESTER.zip "Download TESTER.zip")(4.49 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/39515)**
(12)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
3 Nov 2008 at 08:28

<a href=http://www.powerleveling2000.com/WowGold.aspx>buy cheap

wow goldBuy Cheaper WOW gold</a>,

\[url=http://www.powerleveling2000.com/PowerLeveling.aspx\]wow

powerlevelingwow power leveling \[/url\] at

[http://www.powerleveling2000.com/PowerLeveling.aspx](https://www.mql5.com/go?link=http://www.powerleveling2000.com/PowerLeveling.aspx "http://www.powerleveling2000.com/PowerLeveling.aspx")

![Maria Sountsova](https://c.mql5.com/avatar/avatar_na2.png)

**[Maria Sountsova](https://www.mql5.com/en/users/maria)**
\|
16 Dec 2008 at 09:41

Dear all! Now you can have all attached files with English comments on them.

Sorry for this delay.

![payne](https://c.mql5.com/avatar/2010/3/4B9FFF3E-20E4.jpg)

**[payne](https://www.mql5.com/en/users/payne)**
\|
13 Jan 2010 at 00:21

**maria:**

Dear all! Now you can have all attached files with English comments on them.

Sorry for this delay.

Are the "English" commented versions of the attachements found in this article applicable to all articles in this series or does each article have their own and different version of the attachements. That is, do I need to download the attachements for each article--(because they are all unique)--or does this last "English" commented version cover all articles attachments?

Thanks,

Payne

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
22 Feb 2010 at 11:53

Hi Nikolay Kositsin,

I've got this error during compile...

[Function](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") "DeleteOrder1" is not referenced and will be removed from exp-file

Function "OpenBuyLimitOrder1" is not referenced and will be removed from exp-file

Can you pls include installation instructions??? Thx...

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
18 Jul 2010 at 15:13

Now there
are many training video courses about trading at the Forex. But none video can
help to install robots in Meta Trader. I was looking for such video for a long
time. Now I have it. It is full of all information, which is needed for
beginners. I must say, that this videoclip can simplify our life. Here is
link:http://clicks.aweber.com/y/ct/?l=NDy8Z&m=I\_GkiJXSxTWeOp&b=FjM5DtBrLOn1KWduGTZB0g.I
advice to see it for all beginners, and you will not have any problems. Also I
use information from this forum. Thanks a lot.

![Individual Psychology of a Trader](https://c.mql5.com/2/14/225_1.png)[Individual Psychology of a Trader](https://www.mql5.com/en/articles/1437)

A portrait of a trader's behavior on the financial market. Author's own menu from the book "Как играть и выигрывать на бирже" ("How to speculate on stock exchange and win") by A. Elder.

![A Trader's Assistant Based on Extended MACD Analysis](https://c.mql5.com/2/15/581_4.jpg)[A Trader's Assistant Based on Extended MACD Analysis](https://www.mql5.com/en/articles/1519)

Script 'Trader's Assistant' helps you to make a decision on opening positions, based on the extended analysis of the MACD status for the last three bars in the real-time trading on any timeframe. It can also be used for back testing.

![Drawing Horizontal Break-Through Levels Using Fractals](https://c.mql5.com/2/14/220_15.jpg)[Drawing Horizontal Break-Through Levels Using Fractals](https://www.mql5.com/en/articles/1435)

The article describes creation of an indicator that would display the support/resistance levels using up/down fractals.

![Grouped File Operations](https://c.mql5.com/2/16/677_43.gif)[Grouped File Operations](https://www.mql5.com/en/articles/1543)

It is sometimes necessary to perform identical operations with a group of files. If you have a list of files included into a group, then it is no problem. However, if you need to make this list yourself, then a question arises: "How can I do this?" The article proposes doing this using functions FindFirstFile() and FindNextFile() included in kernel32.dll.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=fjkmkwbevwgyylrhdccnoibbxjlmwvgt&ssn=1769091333907192423&ssn_dr=0&ssn_sr=0&fv_date=1769091333&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1542&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Expert%20Advisors%20Based%20on%20Popular%20Trading%20Systems%20and%20Alchemy%20of%20Trading%20Robot%20Optimization%20(Part%20VII)%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176909133329975797&fz_uniq=5049001328316883716&sv=2552)

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