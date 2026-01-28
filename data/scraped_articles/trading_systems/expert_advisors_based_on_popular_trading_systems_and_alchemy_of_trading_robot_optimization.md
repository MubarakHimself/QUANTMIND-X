---
title: Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization
url: https://www.mql5.com/en/articles/1516
categories: Trading Systems
relevance_score: 12
scraped_at: 2026-01-22T17:16:29.011259
---

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/1516&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049011911116301114)

MetaTrader 4 / Trading systems


### Introduction

The majority of Forex trading books usually offer simplest trading systems as the educational material. But as of today such systems exist only as general instructions without a proper implementation of such trading strategies in the form of ready-to-use Expert Advisors. So now it is impossible to estimate whether such examples contain any practical value. If we look through numerous forums dedicated to EA writing, we can conclude that almost every beginning EA writer has to reinvent the wheel and develop their first Expert Advisors based on simplest trading systems from the very beginning.

I consider the value of such work rather doubtful and think it is necessary to make available to any beginning trader correctly developed EAs based on these trading systems thus delivering from the necessity of starting everything form scratch. In this article I would like to offer my own variant of the problem solution. For EA building I will use indicators from my own library that I have already used in the article [**Effective Averaging Algorithms with Minimal Lag: Use in Indicators**](https://www.mql5.com/en/articles/1450).

### General EA Structure Scheme

Before starting the main description, I would like to draw your attention to the fact that all EAs contained in the article are built in accordance to one and the same scheme:

![](https://c.mql5.com/2/16/1_1_png.jpg)

### Trading System Based on Moving Direction Change

Now I will try to explain in details the idea of the offered EA structure scheme on the example of a ready Expert Advisor explaining all the minor details of working with it. In this system the signal to buy is the Moving direction change from falling into rising:

![](https://c.mql5.com/2/16/jfatlbuy_1.gif)

The signal to sell here is the Moving direction change from rising into falling:

![](https://c.mql5.com/2/16/jfatlsell_1.gif)

For the calculation of trend signals we use the moving value on the third, second and first bars. Moving values on the zero bar are not taken into account, i.e. the system works only on closed bars. As a moving the custom indicator JFATL is used - it is a simple digital FATL filter with additional JMA smoothing. In the Internet you can often see statements that entering a trade at FATL direction change promises several points of profit, so each reader can be easily convinced how efficient this strategy is in reality. Here is the version of system implementation in the form of an Expert Advisor:

### Expert Advisor Code

```
//+==================================================================+
//|                                                        Exp_1.mq4 |
//|                             Copyright © 2007,   Nikolay Kositsin |
//|                              Khabarovsk,   farria@mail.redcom.ru |
//+==================================================================+
#property copyright "Copyright © 2007, Nikolay Kositsin"
#property link "farria@mail.redcom.ru"
//---- EA INPUT PARAMETERS FOR BUY TRADES
extern bool   Test_Up = true;//filter of trade calculations direction
extern int    Timeframe_Up = 240;
extern double Money_Management_Up = 0.1;
extern int    Length_Up = 4;  // smoothing depth
extern int    Phase_Up = 100; // parameter changing in the range
          //-100 ... +100, influences the quality of the transient process;
extern int    IPC_Up = 0;/* Selecting prices, on which the indicator will
be calculated (0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL,
6-WEIGHTED, 7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close.) */
extern int    STOPLOSS_Up = 50;  // stoploss
extern int    TAKEPROFIT_Up = 100; // takeprofit
extern bool   ClosePos_Up = true; // forced position closing allowed
//---- EA INOUT PARAMETERS FOR SELL TRADES
extern bool   Test_Dn = true;//filter of trade calculations direction
extern int    Timeframe_Dn = 240;
extern double Money_Management_Dn = 0.1;
extern int    Length_Dn = 4;  // smoothing depth
extern int    Phase_Dn = 100; // parameter changing in the range
         // -100 ... +100, influences the quality of the transient process;
extern int    IPC_Dn = 0;/* Selecting prices, on which the indicator will
be calculated (0-CLOSE, 1-OPEN, 2-HIGH, 3-LOW, 4-MEDIAN, 5-TYPICAL,
6-WEIGHTED, 7-Heiken Ashi Close, 8-SIMPL, 9-TRENDFOLLOW, 10-0.5*TRENDFOLLOW,
11-Heiken Ashi Low, 12-Heiken Ashi High, 13-Heiken Ashi Open,
14-Heiken Ashi Close.) */
extern int   STOPLOSS_Dn = 50;  // stoploss
extern int   TAKEPROFIT_Dn = 100; // takeprofit
extern bool   ClosePos_Dn = true; // forced position closing allowed
//---- Integer variables for the minimum of counted bars
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
           Print(StringConcatenate("Timeframe_Up parameter cannot ",
                                  "be equal to ", Timeframe_Up, "!!!"));
//---- Checking the correctness of Timeframe_Dn variable value
   if (Timeframe_Dn != 1)
    if (Timeframe_Dn != 5)
     if (Timeframe_Dn != 15)
      if (Timeframe_Dn != 30)
       if (Timeframe_Dn != 60)
        if (Timeframe_Dn != 240)
         if (Timeframe_Dn != 1440)
           Print(StringConcatenate("Timeframe_Dn parameter cannot ",
                                 "be equal to ", Timeframe_Dn, "!!!"));
//---- Initialization of variables
   MinBar_Up = 4 + 39 + 30;
   MinBar_Dn = 4 + 39 + 30;
//---- end of initialization
   return(0);
  }
//+==================================================================+
//| Expert Advisor deinitialization function                         |
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
   //----+ Declaration of local variables
   int    bar;
   double Mov[3], dMov12, dMov23;
   //----+ Declaration of static variables
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
           for(bar = 1; bar <= 3; bar++)
                     Mov[bar - 1]=
                         iCustom(NULL, Timeframe_Up,
                                "JFatl", Length_Up, Phase_Up,
                                                   0, IPC_Up, 0, bar);

           //----+ DEFINING SIGNALS FOR TRADES
           dMov12 = Mov[0] - Mov[1];
           dMov23 = Mov[1] - Mov[2];

           if (dMov23 < 0)
              if (dMov12 > 0)
                        BUY_Sign = true;

           if (dMov12 < 0)
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
           for(bar = 1; bar <= 3; bar++)
                     Mov[bar - 1]=
                         iCustom(NULL, Timeframe_Dn,
                                "JFatl", Length_Dn, Phase_Dn,
                                                   0, IPC_Dn, 0, bar);

           //----+ DEFINING SIGNALS FOR TRADES
           dMov12 = Mov[0] - Mov[1];
           dMov23 = Mov[1] - Mov[2];

           if (dMov23 > 0)
              if (dMov12 < 0)
                       SELL_Sign = true;

           if (dMov12 > 0)
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

To get Buy and Sell signals two absolutely similar independent algorithms are used; each of the algorithms has its own external parameters for optimization. My own experience proves that using this kind of approach to this EA writing is much more profitable than the variant with only one algorithm for detecting Buy and Sell signals. Those who are interested in the variant with a single moving both for short and long positions can study this algorithm in the EXP\_0.mq4 EA; we will continue discussing the EA with two movings. The Expert Advisor can open one position in Buy direction and one position in Sell direction simultaneously for one traded pair. The EA executes trades from the market. Trades are closed by StopLoss and TakeProfit orders. If trend signals appear opposite to positions opened in the EA, the EA allows the forced trade closing. The method of receiving signals to exit trades is analogous with that of getting signals to enter, but is of the contrary character.

### Contents of Lite\_EXPERT1.mqh File

For the sake of maximal economy of your own labor try to use the maximal number of universal user-defined functions when writing Expert Advisors. And later assemble an EA code from different parts, like it is done in different equipment producing factories, where each new product contains the maximal number of unified and standard details, blocks, modules. This is the reason why in all EAs in "Execution of Trades" block several universal user-defined functions are used; they are included into an EA code by the #include <Lite\_EXPERT1.mqh> directive:

```
bool OpenBuyOrder1
(
  bool BUY_Signal, int MagicNumber,
             double Money_Management, int STOPLOSS, int TAKEPROFIT
)

bool OpenSellOrder1
(
  bool SELL_Signal, int MagicNumber,
             double Money_Management, int STOPLOSS, int TAKEPROFIT
)

CountLastTime
(
  int lastError
)

bool CloseOrder1
(
  bool Stop_Signal, int MagicNumber
)

int StopCorrect
(
  string symbol, int Stop
)

bool MarginCheck
(
  string symbol, int Cmd, double& Lot
)

double GetFreeMargin()

bool DeleteOrder1
(
  bool& CloseStop, int MagicNumber
)

bool OpenBuyLimitOrder1
(
  bool& Order_Signal, int MagicNumber,
       double Money_Management, int STOPLOSS, int TAKEPROFIT,
                                      int LEVEL, datetime Expiration
)

bool OpenBuyStopOrder1
(
  bool& Order_Signal, int MagicNumber,
       double Money_Management, int STOPLOSS, int TAKEPROFIT,
                                      int LEVEL, datetime Expiration
)

bool OpenSellLimitOrder1
(
  bool& Order_Signal, int MagicNumber,
       double Money_Management, int STOPLOSS, int TAKEPROFIT,
                                      int LEVEL, datetime Expiration
)

bool OpenSellStopOrder1
(
  bool& Order_Signal, int MagicNumber,
       double Money_Management, int STOPLOSS, int TAKEPROFIT,
                                      int LEVEL, datetime Expiration
)

bool OpenBuyOrder2
(
  bool BUY_Signal, int MagicNumber,
             double Money_Management, int STOPLOSS, int TAKEPROFIT
)

bool OpenSellOrder2
(
  bool SELL_Signal, int MagicNumber,
             double Money_Management, int STOPLOSS, int TAKEPROFIT
)

bool Make_TreilingStop
(
  int MagicNumber, int TRAILINGSTOP
)
```

The OpenBuyOrder1() function opens long positions when it is called, if the value of the external variable BUY\_Signal is equal to true and there are no open positions, the identification (magic) number of which is equal to the value of the MagicNumber variable. Values of external variables STOPLOSS and TAKEPROFIT define correspondingly the value of StopLoss and TakeProfit in points. The value of Money\_Management variable can vary from zero to one. This variable denotes what part of available deposit is used for trade execution. If the value of this variable is less than zero, the OpenBuyOrder1() function will use its value as lot size! Short positions are opened in the analogous way when the OpenSellOrder1() function is called. Both functions open positions irrespective of each other, but only one command for trade execution can be sent to a server within 11 seconds. Besides the execution of trades, OpenBuyOrder1() and OpenSellOrder1() functions record into a log file the information about opened trades.

If the Stop\_Signal variable gets the true value, the CloseOrder1() function, when referred to, closes a position with the magic number equal to the value of MagicNumber variable.

StopCorrect() function accepts as the Stop parameter the StopLoss or TakeProfit value, checks its correspondence with the minimally accepted value, if necessary changes it to the minimally allowed value and returns it taking into account possible corrections.

Assignment of the MarginCheck() function denotes the lessening of lot size that is used in an opened trade up to the maximal size, at which free margin is enough to open the trade, for the case when the free margin is not enough with the current lot size.

OpenBuyOrder1(), OpenSellOrder1() and CloseOrder1() are used inside the start() function, while StopCorrect() and MarginCheck() functions are used inside the code of OpenBuyOrder1() and OpenSellOrder1().

At a correct finish of OpenBuyOrder1(), OpenSellOrder1() and CloseOrder1() any of the functions returns 'true', if an error occurs during the execution of the functions, the returned value is 'false'. All the three functions: OpenBuyOrder1(), OpenSellOrder1() and CloseOrder1() at trade execution send the values of their external variables BUY\_Signal, SELL\_Signal and Stop\_Signal into 'false'!

The GetFreeMargin() function returns free margin size of the current account allowing for profit and loss that can be used for position opening. This function is used to calculate the lot size.

The CountLastTime() function makes the initialization of the LastTime variable with account for an error that has occurred during trade execution. The function should be called immediately after trade execution, for example:

```
  //----+ Open Buy position
  ticket = OrderSend(Symb, OP_BUY, Lot, ask, 3,
            Stoploss, TakeProfit, NULL, MagicNumber, 0, Lime);

  //---- Calculating a pause between trade operations
  CountLastTime(GetLastError());
```

Besides the above enumerated functions, the file Lite\_EXPERT.mqh contains four more functions for placing pending orders and one function for deleting pending orders: OpenBuyLimitOrder(), OpenBuyStopOrder1(), OpenSellLimitOrder1(), OpenSellStopOrder1(), DeleteOrder1(). Assignment of external variables of these functions is absolutely analogous and understandable from their names. New variables LEVEL and Expiration are necessary to send to a function the distance in points from the current price, on which pending orders are placed, and the pending order expiration date accordingly.

Besides all discussed functions the file contains two more functions: OpenBuylOrder2() and OpenSellOrder2(), which are full analogues of OpenBuyOrder1() and OpenSellOrder1(), except for one detail. These functions first open positions without StopLoss and TakeProfit orders and after that modify already opened positions setting StopLoss and TakeProfit. These functions are necessary for operation in EAs developed to be used with brokers that do not allow a Client to place Stop Loss and Take Profit when opening a position upon market because of the "Market Watch" execution type. In such brokerage companies Stop Loss and Take Profit orders are placed by way of modifying an opened position.

And the last function included into the file is Make\_TreilingStop(). The function performs a standard trailing stop.

Besides functions, Lite\_EXPERT.mqh contains a full variable LastTime, which is declared on a global level because it is used in all functions for opening orders.

In my opinion, this set of functions is quite convenient to use in practice for EA writing, it will save much time for beginning EA writers eliminating the necessity to write this code. As an example you can take any function from the offered set of functions:

```
//+==================================================================+
//| OpenBuyOrder1()                                                  |
//+==================================================================+
bool OpenBuyOrder1
        (bool& BUY_Signal, int MagicNumber,
                double Money_Management, int STOPLOSS, int TAKEPROFIT)
{
//----+
  if (!BUY_Signal)
           return(true);
  //---- Checking the expiration of minimal time interval
                                    //between two trade operations
  if (TimeCurrent() < LastTime)
                          return(true);
  int total = OrdersTotal();
  //---- Checking the presence of an opened position
          //with the magic number equal to Value of MagicNumber variable
  for(int ttt = total - 1; ttt >= 0; ttt--)
      if (OrderSelect(ttt, SELECT_BY_POS, MODE_TRADES))
                      if (OrderMagicNumber() == MagicNumber)
                                                      return(true);
  string OrderPrice, Symb = Symbol();
  int    ticket, StLOSS, TkPROFIT;
  double LOTSTEP, MINLOT, MAXLOT, MARGINREQUIRED;
  double FreeMargin, LotVel, Lot, ask, Stoploss, TakeProfit;

  //----+ calculating lot size for position opening
  if (Money_Management > 0)
    {
      MARGINREQUIRED = MarketInfo(Symb, MODE_MARGINREQUIRED);
      if (MARGINREQUIRED == 0.0)
                    return(false);

      LotVel = GetFreeMargin()
               * Money_Management / MARGINREQUIRED;
    }
  else
    LotVel = MathAbs(Money_Management);
  //----
  LOTSTEP = MarketInfo(Symb, MODE_LOTSTEP);
  if (LOTSTEP <= 0)
              return(false);
  //---- fixing lot size for the nearest standard value
  Lot = LOTSTEP * MathFloor(LotVel / LOTSTEP);

  //----+ checking lot for minimally accepted value
  MINLOT = MarketInfo(Symb, MODE_MINLOT);
  if (MINLOT < 0)
         return(false);
  if (Lot < MINLOT)
          return(true);

  //----+ checking lot for maximally accepted value
  MAXLOT = MarketInfo(Symb, MODE_MAXLOT);
  if (MAXLOT < 0)
         return(false);
  if (Lot > MAXLOT)
          Lot = MAXLOT;

  //----+ checking if free margin is enough for lot size
  if (!MarginCheck(Symb, OP_BUY, Lot))
                               return(false);
  if (Lot < MINLOT)
          return(true);
  //----
  ask = NormalizeDouble(Ask, Digits);
  if (ask == 0.0)
          return(false);
  //----
  StLOSS = StopCorrect(Symb, STOPLOSS);
  if (StLOSS < 0)
          return(false);
  //----
  Stoploss = NormalizeDouble(ask - StLOSS * Point, Digits);
  if (Stoploss < 0)
         return(false);
  //----
  TkPROFIT = StopCorrect(Symb, TAKEPROFIT);
  if (TkPROFIT < 0)
          return(false);
  //----
  TakeProfit = NormalizeDouble(ask + TkPROFIT * Point, Digits);
  if (TakeProfit < 0)
         return(false);

  Print(StringConcatenate
         ("Open for ", Symb,
            " a Buy position with the magic number ", MagicNumber));

  //----+ Open Buy position
  ticket = OrderSend(Symb, OP_BUY, Lot, ask, 3,
            Stoploss, TakeProfit, NULL, MagicNumber, 0, Lime);

  //---- Calculating pause between trade operations
  CountLastTime(GetLastError());
  //----
  if(ticket > 0)
   {
     if (OrderSelect(ticket, SELECT_BY_TICKET))
       {
         BUY_Signal = false;
         OpderPrice = DoubleToStr(OrderOpenPrice(), Digits);
         Print(StringConcatenate(Symb, " BUY order with the ticket No",
                ticket, " and magic number ", OrderMagicNumber(),
                                         " opened with the price ",OpderPrice));
         return(true);
       }
     else
       {
         Print(StringConcatenate("Failed to open ", Symb,
            " BUY order with the magic number ", MagicNumber, "!!!"));
         return(true);
       }
    }
  else
    {
      Print(StringConcatenate("Failed to open ", Symb,
           " BUY order with the magic number ", MagicNumber, "!!!"));
      return(true);
    }
  //----
  return(true);
//----+
}
```

Writing such a code without evident and non-evident errors is quite a difficult and long job for a beginning MQL4 user. And using such a ready-to-use universal code (written by a specialist) in one's own Expert Advisors is very easy:

```
          //----+ EXECUTION OF TRADES//---- Initialization of variables
   MinBar_Up = 4 + 39 + 30;

          if (!OpenBuyOrder1(BUY_Sign, 1, Money_Management_Up,
                                          STOPLOSS_Up, TAKEPROFIT_Up))
                                                                 return(-1);
          if (ClosePos_Up)
                if (!CloseOrder1(BUY_Stop, 1))
                                        return(-1);
```

Just a few lines of universal, user-defined function calls - and the code is ready! All you need is once understand how the function calls are written. What is more important - the EA code becomes quite simple and easy-to-understand, due to this you can implement any trading strategy in your own Expert Advisors. In the EA Exp\_1.mq4 functions for opening pending orders and trailing stop are not used, so during the EA compilation MetaEditor will show the warning about the deletion of these functions from the EA:

![](https://c.mql5.com/2/16/8.png)

### Additional Explanations about the EA Code

Now we can start discussing the remaining EA code. The EA consists of two almost identical algorithms so to understand the details properly it is enough to analyze for example the EA part that opens long positions. The EA code already contains comments explaining the meaning of separate code fragments. So let's analyze details that are not commented. In the initialization block the MinBar\_Up variable is initialized:

The purpose of this variable is storing in the EA memory the minimal amount of bars; at less bars the EA operation in long direction is impossible. This value is defined from the algorithm of the custom indicator JFATL.mq4. For the calculation of only one value of the FATL digital filter, 39 chart bars are needed. To obtain JFATL smoothed by JMA algorithm at least 30 FATL values are needed and the EA algorithm of signal detection for trades uses three last but one chart bars plus the fourth one - the zero bar. Checking whether the number of bars is enough for further calculations looks like this:

```
if (IBARS_Up >= MinBar_Up)
```

The check

```
if (LastBars_Up != IBARS_Up)
```

is necessary to eliminate the EA's signals recalculation for entering the market at each tick; the EA should do this only at bar changing, which will sufficiently save computer resources and EA optimization time. This is the reason why the LastBars\_Up variable is declared as a static variable - to remember the number of bars on the previous tick of the int start() function. Initialization of BUY\_Sign and BUY\_Stop for entering and exiting the market is performed only once at bar changing; they should hold the value until the trade is executed or closed or until one more change of bars takes place. That is why these variables are declared as static. I suppose, other EA details are quite clear and can be understood from the code.

### Replacing Moving in the Expert Advisor

Now I would like to dwell on the possibility of the EA modification to use another moving. As an example let's use the custom indicator J2JMA.mq4 from my library, its call will look like this:

```
           //----+ CALCULATING INDICATOR VALUES AND UPLOADING THEM INTO BUFFERS
           for(bar = 1; bar <= 3; bar++)
                     Mov[bar - 1]=
                         iCustom(NULL, Timeframe_Up,
                                "J2JMA", Length1_Up, Length2_Up,
                                             Phase1_Up, Phase2_Up,
                                                  0, IPC_Up, 0, bar);
```

The task consists in changing a little the block of EA external parameters (again only half of the algorithm is described in the example):

```
//---- EA INPUT PARAMETERS FOR BUY TRADES
extern bool   Test_Up = true;//filter of trade calculations direction
extern int    Timeframe_Up=240;
extern double Money_Management_Up = 0.1;
extern int    Length1_Up = 4;  // depth of the first smoothing
extern int    Phase1_Up = 100; // parameter of the first smoothing,
       //changing in the range -100 ... +100, influences the quality
       //of the transient process of averaging;
extern int    Length2_Up = 4;  // depth of the second smoothing
extern int    Phase2_Up = 100; // parameter of the second smoothing,
       //changing in the range -100 ... +100, influences the quality
       //of the transient process of averaging;
```

In initialization block the variable value should be changed:

```
//---- Initialization of variables
   MinBar_Up = 4 + 30 + 30;
```

Now we have two consecutive JMA smoothing, each one needs at least 30 bars and 4 bars for the calculation algorithm of the EA. After that in the block of gaining source values the custom indicator reference should be changed into the one placed at the beginning of the paragraph. For the second part of the algorithm everything is done in the same way. Thus, we may use not only movings, but also oscillators which can be sometimes very useful. The ready EA code based on J2JMA is included into EXP\_2.mqh file.

### EA Optimization

Now on the example of the given EA let us discuss some optimization details of all EAs that will be included in these articles. As mentioned earlier, there are two independent algorithms in the EA - for working with long and short positions. Naturally, it is more convenient and quicker to optimize the EA only for Buy or Sell direction at a time. For this purpose there are two external variables of the EA - Test\_Up and Test\_Dn accordingly.

Assigning the 'false' value to one of these logical variables we exclude all calculations in this direction and as a result optimization takes less time. After optimization of the EA parameters in one direction we may change the values of Test\_Up and Test\_Dn variables into opposite ones and optimize the EA in the opposite direction. And only after that assign 'true' to both variables and test the result. The process of optimization and testing is explained quite well in the article [Testing of Expert Advisors in MetaTrader 4 Client Terminal: an Outward Glance](https://www.mql5.com/en/articles/1417 "Testing of Expert Advisors in MetaTrader 4 Client Terminal: an Outward Glance"), optimization for this EA is performed as described in it. Start the Strategy Tester, upload the EA, select a trade pair, pair period, optimization method and period of time for the optimization:

![](https://c.mql5.com/2/16/2_1.png)

after that select "Expert properties" in the tester and move to "Testing":

![](https://c.mql5.com/2/16/3_4.png)

Here we define the deposit size, select one direction (Long or Short) for optimization and select the genetic optimization algorithm. After that move to "Inputs" tab:

![](https://c.mql5.com/2/16/4_1.png)

Here we assign 'true' to one of the external variables Test\_Up and Test\_Dn and 'false' to the second one. Then assign to Timeframe\_Up and Timeframe\_Dn variables values of chart periods, on which the optimization will be conducted, in Money\_Management\_Up and Money\_Management\_Dn define the part of deposit used to execute Buy and Sell trades correspondingly.

For remaining external variables set changing limits during optimization. After that flag optimized external variables. Close the tab clicking "Ok" and start optimization clicking "Start" in the Tester. After the optimization is over

![](https://c.mql5.com/2/16/5.png)

move to "Results" tab in the Tester:

![](https://c.mql5.com/2/16/10.png)

and upload the satisfying optimization result into Strategy Tester. After that conduct the same procedures in the opposite direction. As a result we get an EA with uploaded parameters, which is profitable in the time period of the optimization. Note, the EA properties tab in MetaTrader 4 differs from the one shown above:

![](https://c.mql5.com/2/16/6.png)

It is not very convenient to use the window in this form and the maximized form is more preferable in such cases. Client terminal does not allow maximizing the window, so an additional file for maximizing was created (OpenExp.exe). This file maximized the properties window of EAs the name of whose fies start from Exp\_ c (case sensitive). For using the file it should be started from some directory, after that the program module will wait for the appearance of EA properties window and change its sizes; in such a moment it is not recommended to move a mouse.

But any EA writer beginning to use testing may have many questions about various testing details. First of all, let's decide what chart period should be used for testing and optimization in our case. In my opinion, you can hardly hear a unique answer. It is considered that the larger the chart period of EA operation is, the more stable are the visible regularities used in the trading system. It is like in live trading. But in the long run the most optimal is the result that can be obtained only during serious analysis and comparison of different testing and optimization variants. It is the same with the security chosen for optimization. Each security has its peculiar character. Now about modeling. Experience shows, that EAs receiving signals for trade execution at the moment of the first bar changing are quite well optimized when modeled upon control points without any sufficient quality and quantity losses during optimizations. Naturally, such statements should be checked by one's own experience. I am convinced, it is useless to optimize such an EA with the modeling of all ticks. Now let's talk about one more important parameter - time period for which optimization is conducted:

![](https://c.mql5.com/2/16/11.png)

Different values are possible here, depending on the purpose of the EA optimization. What is important here, is that the period cannot exceed the available history data, otherwise you may have the following:

![](https://c.mql5.com/2/16/12.png)

The lower bound of all history data available in quotes archive is the following:

![](https://c.mql5.com/2/16/9.png)

Naturally errors of the following kind can occur in the operation of the Expert Advisor and indicators included into it during optimization and testing:

![](https://c.mql5.com/2/16/error.png)

These errors are not connected with the EA itself! It is just that optimization period should be chosen from what is available, and not what is desired!

Now I will give some explanations about the "Inputs" tab of an Expert Advisor. Again, let's analyze only part of them - for long positions. I have already mentioned Test\_Up. The meaning of the Timeframe\_Up parameter is clear. Money\_Management\_Up has also been described earlier. Let's now analyze the Length\_Up parameter. The meaning of this parameter is analogous to the Period parameter of simple moving averages. This parameter can have values from one to infinity. There is no sense in setting the upper parameter more than 150, in most cases it is easier to move to a higher timeframe. During the optimization of this parameter note that the larger the parameter is, the more stable and long-term trends are detected by the JFATL indicator. Thus in this trading system we have a non-evident dependence between the Length\_Up parameter and STOPLOSS\_Up and TAKEPROFIT\_Up. Logically, stop loss and take profit directly depend on Length\_Up. Of course, we can place orders irrespective of Length\_Up, but in such cases the profitability of a trading system will be defined not by the system properties, but by the current market situation, which is not defined in this trading system! The meaning of this statement can be understood on the following example. Suppose we managed to obtain quite good results of EA testing during the optimization of this EA on EUR/USD with the following parameters:

```
extern bool   Test_Up = true;
extern int    Timeframe_Up = 1;
extern double Money_Management_Up = 0.1;
extern int    Length_Up = 4;
extern int    Phase_Up = 100;
extern int    IPC_Up = 0;
extern int    STOPLOSS_Up = 100;
extern int    TAKEPROFIT_Up = 200;
extern bool   ClosePos_Up = false;
```

The fact is, JFATL with Length\_Up equal to four is an indicator of very quick trends; this combined with the minute chart, on which the EA operates, gives such a system the possibility to fix the scale of price changing even in ten-fifteen points, that is why the outstanding testing result with such large stop loss and take profit values denotes only the fact that during the optimization the market experienced a strong trend, which is not detected in the system itself. So it should be understood that after uploading these parameters the EA will hardly show such good results. However, if you can detect the presence of a strong trend in the market using some other tools, then using of such parameters can be justified.

The range of the variable Phase\_Up values change is from -100 to +100. When the values are equal to -100 transient processes of JFATL moving are of minimal character, but my experience shows that the best optimization results are obtained when the value is +100. The IPC\_Up variable defines, what prices will be used for further processing by JFATL algorithm. Using the ClosePos\_Up variable enables the forced position closing if a trend against an opened position starts. It should be taken into account that if take profit and stop loss are placed too far from the market, all positions will be closed upon Moving signals and TP and SL will not influence trading if the value of ClosePos\_Up is equal to 'true'!

### Conclusion

Here I have to end the explanation, the optimization topic discussed at the end of the article is too large. So what is remained will be described in the next article. The main purpose of the first article is showing to a reader my own method of writing Expert Advisors and offering an easier and rather universal way of building an EA code having very little experience of EA writing. This task is solved in the article, I suppose. In the next article I will explain some features of analyzing optimization results and will offer you one for trading system. As for the EAs that are included into the article as examples, please note that the are too simple and can be hardly used for a full-valued automated trading. However, they can be quite useful for the automation of separate trading operations as tools of working when a trader leaves his client terminal at some moments of time.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1516](https://www.mql5.com/ru/articles/1516)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1516.zip "Download all attachments in the single ZIP archive")

[EXPERTS.zip](https://www.mql5.com/en/articles/download/1516/EXPERTS.zip "Download EXPERTS.zip")(5.33 KB)

[INCLUDE.zip](https://www.mql5.com/en/articles/download/1516/INCLUDE.zip "Download INCLUDE.zip")(18.44 KB)

[indicators.zip](https://www.mql5.com/en/articles/download/1516/indicators.zip "Download indicators.zip")(8.95 KB)

[TESTER.zip](https://www.mql5.com/en/articles/download/1516/TESTER.zip "Download TESTER.zip")(3.11 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/39442)**
(6)


![Alain Verleyen](https://c.mql5.com/avatar/2024/5/663a6cdf-e866.jpg)

**[Alain Verleyen](https://www.mql5.com/en/users/angevoyageur)**
\|
22 Feb 2015 at 13:45

**ubqitous:**

The indicators zip file is corrupt - will not decompress. Tried winzip and 7-zip.

You certainly have a problem on your side. I just tried and the zip is ok. (I also tried for the other posts you made).


![jaddin1953](https://c.mql5.com/avatar/avatar_na2.png)

**[jaddin1953](https://www.mql5.com/en/users/jaddin1953)**
\|
5 Jul 2015 at 12:15

indicators, when copied to proper MQL subdir, gives "unspecified" error. but only for those with numbers in the name..


![cyshanhu](https://c.mql5.com/avatar/avatar_na2.png)

**[cyshanhu](https://www.mql5.com/en/users/cyshanhu)**
\|
18 Sep 2015 at 04:16

**Alain Verleyen:**

You certainly have a problem on your side. I just tried and the zip is ok. (I also tried for the other posts you made).

i also see the same problem that the indicator and tester zip files cant be unziped ...why?

my email:cyshanhu@live.cn

![Willhelm Friedrich](https://c.mql5.com/avatar/avatar_na2.png)

**[Willhelm Friedrich](https://www.mql5.com/en/users/contrahour)**
\|
29 May 2018 at 03:49

I am new to MQL4 and these articles have been extremely helpful.  Thank you.


![sam76](https://c.mql5.com/avatar/avatar_na2.png)

**[sam76](https://www.mql5.com/en/users/sam76)**
\|
2 Jun 2018 at 08:52

Hey Do you have any Ready Made profitable non repaint EA's for sale....pls let me know....i Have lost only not yet Earned a 1 $  After all these 6 -7 Year's?


![Comparative Analysis of 30 Indicators and Oscillators](https://c.mql5.com/2/15/577_13.gif)[Comparative Analysis of 30 Indicators and Oscillators](https://www.mql5.com/en/articles/1518)

The article describes an Expert Advisor that allows conducting the comparative analysis of 30 indicators and oscillators aiming at the formation of an effective package of indexes for trading.

![Metalanguage of Graphical Lines-Requests. Trading and Qualified Trading Learning](https://c.mql5.com/2/15/597_26.gif)[Metalanguage of Graphical Lines-Requests. Trading and Qualified Trading Learning](https://www.mql5.com/en/articles/1524)

The article describes a simple, accessible language of graphical trading requests compatible with traditional technical analysis. The attached Gterminal is a half-automated Expert Advisor using in trading results of graphical analysis. Better used for self-education and training of beginning traders.

![Fallacies, Part 1: Money Management is Secondary and Not Very Important](https://c.mql5.com/2/15/601_23.gif)[Fallacies, Part 1: Money Management is Secondary and Not Very Important](https://www.mql5.com/en/articles/1526)

The first demonstration of testing results of a strategy based on 0.1 lot is becoming a standard de facto in the Forum. Having received "not so bad" from professionals, a beginner sees that "0.1" testing brings rather modest results and decides to introduce an aggressive money management thinking that positive mathematic expectation automatically provides positive results. Let's see what results can be achieved. Together with that we will try to construct several artificial balance graphs that are very instructive.

![Easy Way to Publish a Video at MQL4.Community](https://c.mql5.com/2/15/582_26.jpg)[Easy Way to Publish a Video at MQL4.Community](https://www.mql5.com/en/articles/1520)

It is usually easier to show, than to explain. We offer a simple and free way to create a video clip using CamStudio for publishing it in MQL.community forums.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/1516&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049011911116301114)

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