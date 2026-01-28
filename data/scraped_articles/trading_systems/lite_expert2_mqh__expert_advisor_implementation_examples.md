---
title: Lite_EXPERT2.mqh: Expert Advisor Implementation Examples
url: https://www.mql5.com/en/articles/1384
categories: Trading Systems
relevance_score: 6
scraped_at: 2026-01-23T11:51:13.160151
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/1384&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062766990558406783)

MetaTrader 4 / Trading systems


### Introduction

In my previous article entitled ["Lite\_EXPERT2.mqh: Functional Kit for Developers of Expert Advisors"](https://www.mql5.com/en/articles/1380), I familiarized the readers with the Lite\_EXPERT2.mqh functions. In this article, I will give real Expert Advisor implementation examples that use these functions. I believe, the operation of the trading functions

```
OpenBuyOrder1_()
OpenSellOrder1_()
OpenBuyOrder2_()
OpenSellOrder2_()
OpenBuyLimitOrder1_()
OpenBuyStopOrder1_()
OpenSellLimitOrder1_()
OpenSellStopOrder1_()
Make_BuyTrailingStop_()
Make_SellTrailingStop_()
```

is not materially different from the similar functions provided in the Lite\_EXPERT1.mqh file.

The initialization of a few more external variables that they contain, is unlikely to cause any confusion (see Exp\_0\_1.mq4 and Exp\_0.mq4, Exp\_1.mq4 and EXP\_1\_1.mq4). So there is absolutely no need to go back to them once again. I will proceed directly to the examples built using the trading functions that use absolute values of price chart levels as external variables for pending orders.

```
ddOpenBuyOrder1_()
dOpenSellOrder1_()
dOpenBuyOrder2_()
dOpenSellOrder2_()
dOpenBuyLimitOrder1_()
dOpenBuyStopOrder1_()
dOpenSellLimitOrder1_()
dOpenSellStopOrder1_()
dModifyOpenBuyOrder_()
dModifyOpenSellOrder_()
dModifyOpenBuyOrderS()
dModifyOpenSellOrderS()
dMake_BuyTrailingStop_()
dMake_SellTrailingStop_()
```

The trading strategies that will be discussed further in this article are based on the Average True Range indicator, so this is what I am going to start this article with.

### Using the Average True Range Indicator in Mechanical Trading Systems

The [Average True Range](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/atr) indicator (hereinafter - ATR or Average True Range) was developed by Welles Wilder and first introduced in his book "New Concepts in Technical Trading Systems" in 1978. This indicator subsequently became quite popular and is still included in many technical analysis software suites. The ATR indicator itself does not indicate current trend directions but provides a graphic image of volatility or activity of the market under consideration.

![](https://c.mql5.com/2/13/atr.gif)

Essentially, this indicator can be used in mechanical trading systems in two variants:

1\. Filtering signals of the trading system for identification of trend and non-trend market conditions.

In this case the trend direction and entry signals are received from other indicators, while the ATR indicator only provides an additional entry condition. Such additional condition can, for example, be the breakout of the average indicator value by the indicator itself. To get the average ATR value, it is convenient to use the signal average line based on ATR.

2\. Adaptive pending orders.

The absolute value of this indicator determines the distances from the bar opening price, outside of which strong price fluctuations are very likely to begin. It is therefore very convenient to use these levels for setting pending orders for opening positions and Stop Loss levels. In this case, we have the opportunity to use the ATR indicator to set orders at a certain distance from the price that would adapt to the current market volatility at every deal. A fixed distance set in the real market conditions is seen much like an old army general, always in a state of combat readiness to the war that has long become history. This often results in a much more interesting trading system performance in the real, ever-changing market conditions. In a very much similar way, you can use the ATR distance to move Trailing Stop levels to the price at every bar change.

Now, we can proceed to developing Expert Advisors using the Lite\_EXPERT2.mqh file. The best way to do it is to start with the modernization of the Expert Advisors built on the basis of Lite\_EXPERT1.mqh in order to give them better flexibility in trading.

### Trading System that Uses Changes in MA Direction as Signals for Market Entry and Exit

I have already provided a detailed description of such a system in my article entitled ["Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization"](https://www.mql5.com/en/articles/1516) devoted to very basic trading systems.

It is time to make it more complex. An almost similar trading system based on the Lite\_EXPERT2.mqh functions is developed in the Exp\_1\_1.mq4 Expert Advisor (original Exp\_1.mq4). We just need to replace fixed Stop Loss and Take Profit with the ones recalculated in ATR units and add the similar Trailing Stop levels that will shift once upon each bar change. This is best implemented in two stages. We first replace the Stop Loss and Take Profit (the Exp\_17\_A.mq4 Expert Advisor) and after the code is checked for errors and for being in conformity with the selected trading strategy, we add the Trailing Stop levels (the Exp\_17.mq4 Expert Advisor). In the article, I will only provide the final version with more detailed descriptions of changes made to the code.

```
//+X================================================================X+
//|                                                       Exp_17.mq4 |
//|                             Copyright © 2009,   Nikolay Kositsin |
//|                              Khabarovsk,   farria@mail.redcom.ru |
//+X================================================================X+
#property copyright "Copyright © 2009, Nikolay Kositsin"
#property link "farria@mail.redcom.ru"
//+--------------------------------------------------------+
//| INPUT PARAMETERS OF THE EXPERT ADVISOR FOR BUY DEALS   |
//+--------------------------------------------------------+
extern bool   Test_Up = true;         // filter of direction of deal calculations
extern int    Timeframe_Up = 240;
extern double Money_Management_Up = 0.1;
extern int    Length_Up = 4;          // depth of smoothing
extern int    Phase_Up = 100;         // parameter varying within the range (-100..+100), impacts the transitional process quality;
extern int    IPC_Up = 0;
extern int    ATR_Period_Up = 14;     // True Range averaging period
extern int    LevMinimum_Up = 40;     // Minimum value in points below
                                      // which pending order values cannot fall
extern int    STOPLOSS_Up = 100;      // Stop Loss expressed as percentage of ATR
extern int    TAKEPROFIT_Up = 200;    // Take Profit expressed as percentage of ATR
extern int    TRAILINGSTOP_Up = 100;  // Trailing Stop expressed as percentage of ATR
extern bool   ClosePos_Up = true;     // permission to forcibly close a position
//+--------------------------------------------------------+
//| INPUT PARAMETERS OF THE EXPERT ADVISOR FOR SELL DEALS  |
//+--------------------------------------------------------+
extern bool   Test_Dn = true;         // filter of direction of deal calculations
extern int    Timeframe_Dn = 240;
extern double Money_Management_Dn = 0.1;
extern int    Length_Dn = 4;          // depth of smoothing
extern int    Phase_Dn = 100;         // parameter varying within the range (-100..+100), impacts the transitional process quality;
extern int    IPC_Dn = 0;
extern int    ATR_Period_Dn = 14;     // True Range averaging period
extern int    LevMinimum_Dn = 40;     // Minimum value in points below
                                      // which pending order values cannot fall
extern int    STOPLOSS_Dn = 100;      // Stop Loss expressed as percentage of ATR
extern int    TAKEPROFIT_Dn = 200;    // Take Profit expressed as percentage of ATR
extern int    TRAILINGSTOP_Dn = 100;  // Trailing Stop expressed as percentage of ATR
extern bool   ClosePos_Dn = true;     // permission to forcibly close a position
//+---------------------------------------------+
//---- declaration of integer variables for the minimum calculation bars
int MinBar_Up, MinBar_Dn;
//---- declaration of integer variables for chart time frames in seconds
int Period_Up, Period_Dn;
//---- declaration of floating point variables for pending orders
double _STOPLOSS_Up, _TAKEPROFIT_Up, _LevMinimum_Up, _TRAILINGSTOP_Up;
double _STOPLOSS_Dn, _TAKEPROFIT_Dn, _LevMinimum_Dn, _TRAILINGSTOP_Dn;
//+X================================================================X+
//| Custom Expert functions                                          |
//+X================================================================X+
#include <Lite_EXPERT2.mqh>
//+X================================================================X+
//| Custom Expert initialization function                            |
//+X================================================================X+
int init()
  {
//---- Checking correctness of the Timeframe_Up variable value
   TimeframeCheck("Timeframe_Up", Timeframe_Up);
//---- Checking correctness of the Timeframe_Dn variable value
   TimeframeCheck("Timeframe_Dn", Timeframe_Dn);
//---- Initialization of variables
   MinBar_Up = 4 + 39 + 30;// four bars for entry signals + FATL filter length + JMA filter length
   MinBar_Up = MathMax(MinBar_Up, ATR_Period_Up + 1);
   MinBar_Dn = 4 + 39 + 30;// four bars for entry signals + FATL filter length + JMA filter length
   MinBar_Dn = MathMax(MinBar_Dn, ATR_Period_Dn + 1);
//----
   Period_Up = Timeframe_Up * 60; // chart time frame for long positions in seconds
   Period_Dn = Timeframe_Dn * 60; // chart time frame for short positions in seconds

//---- Conversion of percent to fraction
   _STOPLOSS_Up = STOPLOSS_Up / 100.0;
   _TAKEPROFIT_Up = TAKEPROFIT_Up / 100.0;
   _TRAILINGSTOP_Up = TRAILINGSTOP_Up / 100.0;
//---- Conversion of percent to fraction
   _STOPLOSS_Dn = STOPLOSS_Dn / 100.0;
   _TAKEPROFIT_Dn = TAKEPROFIT_Dn / 100.0;
   _TRAILINGSTOP_Dn = TRAILINGSTOP_Dn / 100.0;
//---- Conversion of the minimum points to the price distance minimum
   _LevMinimum_Up = LevMinimum_Up * Point;
   _LevMinimum_Dn = LevMinimum_Dn * Point;
//---- initialization complete
   return(0);
  }
//+X================================================================X+
//| expert deinitialization function                                 |
//+X================================================================X+
int deinit()
  {
//----+ Deleting global variables after testing and optimizations
   TimeLevelGlobalVariableDel(Symbol(), 1);
   TimeLevelGlobalVariableDel(Symbol(), 2);
   //---- Expert Advisor deinitialization complete
   return(0);
//----+
  }
//+X================================================================X+
//| Custom Expert iteration function                                 |
//+X================================================================X+
int start()
  {
//----+
   //----+ Declaration of local variables
   int bar;
   double Mov[3], dMov12, dMov23, ATR, Level, open;
   //----+ Declaration of static variables
   static datetime TradeTimeLevel_Up, TradeTimeLevel_Dn;
   //----
   static bool BUY_Sign, BUY_Stop, SELL_Sign, SELL_Stop;
   static bool TrailSignal_Up, TrailSignal_Dn;
   //----
   static double dStopLoss_Up, dTakeProfit_Up, dTrailingStop_Up;
   static double dStopLoss_Dn, dTakeProfit_Dn, dTrailingStop_Dn;

   //+---------------------------+
   //| CODE FOR LONG POSITIONS   |
   //+---------------------------+
   if (Test_Up)
      if (MinBarCheck(Symbol(), Timeframe_Up, MinBar_Up))
       {
         if (IsNewBar(0, Symbol(), Timeframe_Up))
          {
           //----+ Zeroing out trading signals
           BUY_Sign = false;
           BUY_Stop = false;
           //---- Getting the time limit for disabling
                                         // the next trading operation
           TradeTimeLevel_Up = iTime(NULL, Timeframe_Up, 0);
           if (TradeTimeLevel_Up == 0)
            return(-1);
           TradeTimeLevel_Up += Period_Up;

           //----+ CALCULATING AND LOADING INDICATOR VALUES TO BUFFERS
           for(bar = 1; bar <= 3; bar++)
                     Mov[bar - 1]=
                         iCustom(NULL, Timeframe_Up, "JFatl",
                                 Length_Up, Phase_Up, 0, IPC_Up, 0, bar);

           //----+ DETERMINING SIGNALS FOR DEALS
           dMov12 = Mov[0] - Mov[1];
           dMov23 = Mov[1] - Mov[2];
           //---- Getting a signal for opening a position
           if (dMov23 < 0)
              if (dMov12 > 0)
                        BUY_Sign = true;
           //---- Getting a signal for closing a position
           if (dMov12 < 0)
              if (ClosePos_Up)
                        BUY_Stop = true;

           //----+ CALCULATION OF PENDING ORDERS FOR LONG POSITIONS
           // Make order calculation only if the trading signal is available
           if (BUY_Sign)
            {
             //---- Getting the initial ATR value
             ATR = iATR(NULL, Timeframe_Up, ATR_Period_Up, 1);
             //---- Getting the current price
             open = iOpen(Symbol(), Timeframe_Up, 0);

             //---- Calculating the distance to the Stop Loss
             Level = ATR * _STOPLOSS_Up;
             //---- Checking the distance to the Stop Loss against the minimum value
             if (Level < _LevMinimum_Up)
                     Level = _LevMinimum_Up;
             //---- Determining the absolute Stop Loss value
             dStopLoss_Up = open - Level;

             //---- Calculating the distance to the Take Profit
             Level = ATR * _TAKEPROFIT_Up;
             //---- Checking the distance to the Take Profit against the minimum value
             if (Level < _LevMinimum_Up)
                     Level = _LevMinimum_Up;
             //---- Determining the absolute Take Profit value
             dTakeProfit_Up = open + Level;

             //---- Correcting values of pending
                           // orders, given the direction of the trade
             dGhartVelueCorrect(OP_BUY, dStopLoss_Up);
             dGhartVelueCorrect(OP_BUY, dTakeProfit_Up);
            }

           //----+ CALCULATION OF TRAILING STOPS FOR LONG POSITIONS
           dTrailingStop_Up = 0;
           TrailSignal_Up = false;
           //----
           if (TRAILINGSTOP_Up > 0)
            // Calculate the Trailing Stop only if the necessary position exists
            if (OrderSelect_(Symbol(), OP_BUY, 1, MODE_TRADES))
             // Move Trailing Stop if the position is opened on a non-zero bar
             if (iBarShift(NULL, Timeframe_Up, OrderOpenTime(), false) > 0)
              {
               TrailSignal_Up = true;
               //---- Getting the initial ATR value
               ATR = iATR(NULL, Timeframe_Up, ATR_Period_Up, 1);
               //---- Getting the current price
               open = iOpen(Symbol(), Timeframe_Up, 0);
               //---- Calculating the distance to the Stop Loss
               Level = ATR * _TRAILINGSTOP_Up;
               //---- Checking the distance to the Stop Loss against the minimum value
               if (Level < _LevMinimum_Up)
                        Level = _LevMinimum_Up;
               //---- Getting the absolute Trailing Stop value
               dTrailingStop_Up = open - Level;
               //---- Correcting the absolute Trailing Stop value,
                        // given the direction of the trade (for position
                           // modification functions the value of the cmd variable is inverse!)
               dGhartVelueCorrect(OP_SELL, dTrailingStop_Up);
              }
          }

         //----+ DEAL EXECUTION
         if (!dOpenBuyOrder1_(BUY_Sign, 1, TradeTimeLevel_Up,
               Money_Management_Up, 5, dStopLoss_Up, dTakeProfit_Up))
                                                             return(-1);
         //----
         if (!CloseBuyOrder1_(BUY_Stop, 1))
                                      return(-1);
         //----
         if (!dMake_BuyTrailingStop_(TrailSignal_Up, 1,
                          TradeTimeLevel_Up, dTrailingStop_Up))
                                                       return(-1);
       }
   //+---------------------------+
   //| CODE FOR SHORT POSITIONS  |
   //+---------------------------+
   if (Test_Dn)
      if (MinBarCheck(Symbol(), Timeframe_Dn, MinBar_Dn))
       {
         if (IsNewBar(1, Symbol(), Timeframe_Dn))
          {
           //----+ Zeroing out trading signals
           SELL_Sign = false;
           SELL_Stop = false;
           //---- Getting the time limit for disabling
                                         // the next trading operation
           TradeTimeLevel_Dn = iTime(NULL, Timeframe_Dn, 0);
           if (TradeTimeLevel_Dn == 0)
            return(-1);
           TradeTimeLevel_Dn += Period_Dn;

           //----+ CALCULATING AND LOADING INDICATOR VALUES TO BUFFERS
           for(bar = 1; bar <= 3; bar++)
                     Mov[bar - 1]=
                         iCustom(NULL, Timeframe_Dn, "JFatl",
                                 Length_Dn, Phase_Dn, 0, IPC_Dn, 0, bar);

           //----+ DETERMINING SIGNALS FOR DEALS
           dMov12 = Mov[0] - Mov[1];
           dMov23 = Mov[1] - Mov[2];
           //---- Getting a signal for opening a position
           if (dMov23 > 0)
              if (dMov12 < 0)
                       SELL_Sign = true;
           //---- Getting a signal for closing a position
           if (dMov12 > 0)
               if (ClosePos_Dn)
                       SELL_Stop = true;

           //----+ CALCULATION OF PENDING ORDERS FOR SHORT POSITIONS
           // Make order calculation only if the trading signal is available
           if (SELL_Sign)
            {
             //---- Getting the initial ATR value
             ATR = iATR(NULL, Timeframe_Dn, ATR_Period_Dn, 1);
             //---- Getting the current price
             open = iOpen(Symbol(), Timeframe_Dn, 0);

             //---- Calculating the distance to the Stop Loss
             Level = ATR * _STOPLOSS_Dn;
             //---- Checking the distance to the Stop Loss against the minimum value
             if (Level < _LevMinimum_Dn)
                     Level = _LevMinimum_Dn;
             //---- Determining the absolute Stop Loss value
             dStopLoss_Dn = open + Level;

             //---- Calculating the distance to the Take Profit
             Level = ATR * _TAKEPROFIT_Dn;
             //---- Checking the distance to the Take Profit against the minimum value
             if (Level < _LevMinimum_Dn)
                     Level = _LevMinimum_Dn;
             //---- Determining the absolute Take Profit value
             dTakeProfit_Dn = open - Level;

             //---- Correcting values of pending orders, given the direction of the trade
             dGhartVelueCorrect(OP_SELL, dStopLoss_Dn);
             dGhartVelueCorrect(OP_SELL, dTakeProfit_Dn);
            }

           //----+ CALCULATION OF TRAILING STOPS FOR SHORT POSITIONS
           dTrailingStop_Dn = 0;
           TrailSignal_Dn = false;
           //----
           if (TRAILINGSTOP_Dn > 0)
            // Calculate the Trailing Stop only if the necessary position exists
            if (OrderSelect_(Symbol(), OP_SELL, 2, MODE_TRADES))
             // Move Trailing Stop if the position is opened on a non-zero bar
             if (iBarShift(NULL, Timeframe_Dn, OrderOpenTime(), false) > 0)
              {
               TrailSignal_Dn = true;
               //---- Getting the initial ATR value
               ATR = iATR(NULL, Timeframe_Dn, ATR_Period_Dn, 1);
               //---- Getting the current price
               open = iOpen(Symbol(), Timeframe_Dn, 0);
               //---- Calculating the distance to the Stop Loss
               Level = ATR * _TRAILINGSTOP_Dn;
               //---- Checking the distance to the Stop Loss against the minimum value
               if (Level < _LevMinimum_Dn)
                        Level = _LevMinimum_Dn;
               //---- Getting the absolute Trailing Stop value
               dTrailingStop_Dn = open + Level;
               //---- Correcting the absolute Trailing Stop value,
                        // given the direction of the trade (for position
                           // modification functions the value of the cmd variable is inverse!)
               dGhartVelueCorrect(OP_BUY, dTrailingStop_Dn);
              }
          }

         //----+ DEAL EXECUTION
         if (!dOpenSellOrder1_(SELL_Sign, 2, TradeTimeLevel_Dn,
               Money_Management_Dn, 5, dStopLoss_Dn, dTakeProfit_Dn))
                                                             return(-1);
         //----
         if (!CloseSellOrder1_(SELL_Stop, 2))
                                      return(-1);
         //----
         if (!dMake_SellTrailingStop_(TrailSignal_Dn, 2,
                          TradeTimeLevel_Dn, dTrailingStop_Dn))
                                                       return(-1);
       }

    return(0);
//----+
  }
//+X----------------------+ <<< The End >>> +-----------------------X+
```

So, we have a new pair of variables in the block of external variables of the Expert Advisor - ATR\_Period\_Up and ATR\_Period\_Dn that can be used to change the ATR indicator values involved in the calculation of pending orders. Now, the logical meaning of values of external variables for Stop Loss, Take Profit and Trailing Stop is somewhat different. These values used to represent relative distance in points from the order to the current price. They now represent percent of the ATR indicator value on the first bar. In other words, to calculate the order, we take percent of the ATR indicator and add it to the value of the zero bar opening price. So the best way to convert percent to the floating point value is to use the init() block of the Expert Advisor where these calculations will be done only once and the calculated values will be saved to the variables declared with a global scope.

Due to the new ATR indicator available, formulas for initialization of the LevMinimum\_Up and LevMinimum\_Dn variables in the init() block have changed. The start() block of the Expert Advisor features new variables declared as static to store values between the ticks of the terminal. The code for the calculation of pending orders and Trailing Stop levels is arranged into small modules inside the blocks for getting deal signals. Now in order to execute a deal, we use different functions where the value of five is used upon initialization of the Margin\_Mode variable as a more logical value in the floating Stop Loss conditions.

To demonstrate the opportunities offered by the use of the IndicatorCounted\_() and ReSetAsIndexBuffer() functions, in this Expert Advisor we replaced the custom indicator JFatl.mq4 with the JFATL() function inserted in the code of the Expert Advisor.

```
bool JFATL(int Number, string symbol, int timeframe,
               int Length, int Phase, int IPC, double& Buffer[])
```

The function gets input parameters of the indicator and the Buffer\[\] array. In case of successful calculation, the function returns true, otherwise false. The array is by reference converted to the indicator buffer analog filled with the JFATL indicator values. In the function, the JJMASeries() function is replaced with JJMASeries1() that does not perform calculations on the zero bar. We have also replaced the PriceSeries() function with the iPriceSeries() function. The indicator initialization block has been moved to the "Zero initialization" block. Please note that the JJMASeries1() function is only used within this function in the Expert Advisor so the Number variable value is not recalculated and is passed to the JJMASeries1() directly. The same applies to the IndicatorCounted\_() function.

I have already talked about such replacements of indicators with function in my other articles devoted to this subject: [1](https://www.mql5.com/en/articles/1456), [2](https://www.mql5.com/en/articles/1457), [3](https://www.mql5.com/en/articles/1463). The Expert Advisor that features this replacement is represented by the Exp\_17\_.mq4 file. We should note the fact that the JMA smoothing algorithm employed in the JFatl.mq4 indicator is quite resource consuming and such replacement of the indicator with the indicator function results in a quite significant optimization speed gain in this Expert Advisor when compared to the previous version. And finally, for the most lazy of you, the same Expert Advisor (Exp\_17R.mq4) was developed in such a way that it can contain all the necessary functions within its code, without requiring any additional include files or indicators for its compilation and operation. The operation of all three analogs of this Expert Advisor is identical! Except perhaps the fact that IPC\_Up and IPC\_Dn variable values in the last Expert Advisor vary within a somewhat smaller range (0-10) due to the lack of calling the Heiken Ashi#.mq4 indicator.

In the [forum](https://www.mql5.com/en/forum/mql4), you can occasionally see some MQL4 programming gurus frown on the idea of writing such indicator functions, which in their opinion is like putting on your pants by the head. I personally only spend fifteen minutes of my time writing such a function based on a quite easy-to-understand code. So if one can be six times faster when running an optimization marathon with the pants put on that way, I would stick with this option!

**Breakout System for Trading the News**

This version of the trading system has already been provided for your consideration in [my article](https://www.mql5.com/en/articles/1523) in the form of the Exp\_10.mq4 Expert Advisor. The Exp\_10\_1.mq4 Expert Advisor based on the Lite\_EXPERT2.mqh functions is completely analogous. It is slightly more complicated than the original version but much more reliable as it does not get affected by different cases of Expert Advisor, Terminal or Operating System restart. To determine the time after which an open position should be closed, this Expert Advisor uses the TradeTimeLevelCheck() function:

This function returns true after the point of time the value of which was passed as an input parameter to the function for placing pending orders or opening positions. The value as such is obtained by the function from the global variable.

Now we need to change the algorithm of pending order calculation. But in this case, Stop orders should also be dynamically calculated in addition to Stop Loss and Take Profit. Essentially, this does not change anything for us and everything is implemented in the same way. Furthermore, Trailing Stop in the original Expert Advisor works on every tick, while we need to move it at every bar change. The final code of the Expert Advisor (Exp\_18.mq4) is certainly not as simple as the original one but the program logic is quite concise and straightforward. Exp\_18R.mq4 is a complete analog of the last Expert Advisor implemented in the form of a finished, self-contained file.

### Conclusion

I believe, the Lite\_EXPERT2.mqh custom functions are not anything new when compared to the Lite\_EXPERT1.mqh functions in terms of programming approach.

They simply enhance programming functionality, while remaining essentially the same in terms of application. So after a careful study of the Lite\_EXPERT1.mqh functions, you should not find any difficulty in learning the functionality of Lite\_EXPERT2.mqh quickly and easily.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1384](https://www.mql5.com/ru/articles/1384)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1384.zip "Download all attachments in the single ZIP archive")

[Experts.zip](https://www.mql5.com/en/articles/download/1384/Experts.zip "Download Experts.zip")(72.52 KB)

[Exp\_10R.mq4](https://www.mql5.com/en/articles/download/1384/Exp_10R.mq4 "Download Exp_10R.mq4")(58.81 KB)

[Exp\_17R.mq4](https://www.mql5.com/en/articles/download/1384/Exp_17R.mq4 "Download Exp_17R.mq4")(142.05 KB)

[Exp\_17R\_.mq4](https://www.mql5.com/en/articles/download/1384/Exp_17R_.mq4 "Download Exp_17R_.mq4")(143.73 KB)

[Exp\_18R.mq4](https://www.mql5.com/en/articles/download/1384/Exp_18R.mq4 "Download Exp_18R.mq4")(66.33 KB)

[Include.zip](https://www.mql5.com/en/articles/download/1384/Include.zip "Download Include.zip")(53.88 KB)

[Indicators.zip](https://www.mql5.com/en/articles/download/1384/Indicators.zip "Download Indicators.zip")(4.99 KB)

[TESTER.zip](https://www.mql5.com/en/articles/download/1384/TESTER.zip "Download TESTER.zip")(14.32 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/39150)**

![Testing and Optimization of Expert Advisors](https://c.mql5.com/2/17/824_37.gif)[Testing and Optimization of Expert Advisors](https://www.mql5.com/en/articles/1385)

The article provides a detailed description of the process of testing and optimizing Expert Advisors in the MetaTrader 4 Strategy Tester. The importance of such information and the need for this publication cannot be underestimated. A lot of users who only get started with the MetaTrader 4 trading platform have a very vague idea of what and how needs to be done when working with Expert Advisors. The proposed article gives simple and clear answers to all of these questions and provides a slightly more professional approach to handling these issues using a specific example.

![Advanced Analysis of a Trading Account](https://c.mql5.com/2/17/830_31.png)[Advanced Analysis of a Trading Account](https://www.mql5.com/en/articles/1383)

The article deals with the automatic system for analyzing any trading account in MetaTrader 4 terminal. Technical aspects of a generated report and interpretation of the obtained results are considered. Conclusions on improving trading factors are drawn after the detailed review of the report. MQLab™ Graphic Report script is used for analysis.

![Video tutorial: MetaTrader Signals Service](https://c.mql5.com/2/0/signal-video.png)[Video tutorial: MetaTrader Signals Service](https://www.mql5.com/en/articles/821)

In just 15 minutes, this video tutorial explains what MetaTrader Signals Service is, and demonstrates in great detail how to subscribe to trade signals and how to become a signal provider in our service. By watching this tutorial, you will be able to subscribe to any trading signal, or publish and promote your own signals in our service.

![Visual Optimization of Indicator and Signal Profitability](https://c.mql5.com/2/17/820_9.gif)[Visual Optimization of Indicator and Signal Profitability](https://www.mql5.com/en/articles/1381)

This article is a continuation and development of my previous article "Visual Testing of Profitability of Indicators and Alerts". Having added some interactivity to the parameter changing process and having reworked the study objectives, I have managed to get a new tool that does not only show the prospective trade results based on the signals used but also allows you to immediately get a layout of deals, balance chart and the end result of trading by moving virtual sliders that act as controls for signal parameter values in the main chart.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/1384&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062766990558406783)

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