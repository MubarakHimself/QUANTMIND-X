---
title: Price Action. Automating the Inside Bar Trading Strategy
url: https://www.mql5.com/en/articles/1771
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:50:52.069435
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/1771&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062762304749086825)

MetaTrader 4 / Examples


### Introduction

All Forex traders come across the Price Action at some point. This is not a mere chart analysis technique but the entire system for defining the possible future price movement direction. In this article, we will look at the Inside Bar pattern in details and develop an EA for tracking the Inside Bar and performing trades based on the pattern.

### About Price Action

The Price Action is a non-indicator price movement detection method using simple and complex patterns, as well as auxiliary chart elements (horizontal, vertical and trend lines, Fibo levels, support/resistance levels, etc.).

At first glance, the method may seem rather complicated but actually this is not the case. The method is gaining popularity from year to year, since its advantages are evident, for example, when compared to the methods involving technical indicators.

### Inside Bar

Inside Bar is a bar having its body and wicks contained entirely within the range of the previous (mother) bar. The inside bar's High lies lower and Low is located higher than the mother bar's ones. Mother and inside bars form a pattern considered to be a potential entry signal.

This is a two-sided pattern, since it may indicate either a reversal, or a trend continuation.

![Fig. 1. Inside bar](https://c.mql5.com/2/19/Fig1_price_action_inside_bar1__1.png)

Fig. 1. Inside bar

![Fig. 2. Inside Bar pattern layout ](https://c.mql5.com/2/19/Fig2_inbar_shema__1.png)

Fig. 2. Inside Bar pattern layout

**Inside bar rules:**

- The Inside Bar pattern is significant on higher timeframes, like H4 or D1.
- The pattern can indicate either a trend reversal or a continuation.
- Apply additional graphical analysis tools for more precise entry, including trend lines, support/resistance levels, Fibo levels, other Price Action patterns, etc.
- Use pending orders to avoid premature or false market entries.
- Do not use inside bars repeatedly occurring in the flat market as market entry signals.

![Fig. 3. Defining the genuine inside bar on GBPUSD D1](https://c.mql5.com/2/19/Fig3_price_action_inside_bar1__2.png)

Fig. 3. Defining the genuine inside bar on GBPUSD D1

Keeping all this in mind, let's try to define a genuine inside bar. On the above chart, we can see that a bullish bar was formed after the sharp downward movement. However, the bar lies completely within the boundaries of the previous one. The pattern is confirmed by the fact that it is formed at the support level. The third confirmation is the absence of flat. Since the pattern satisfies the rules, it can be considered **genuine**.

### Defining Entry Points and Setting Stop Orders

So, we have found a genuine inside bar on the chart (Fig. 3). How should we enter the market and where should we set our stop orders? Let's examine the Figure 4.

![Fig. 4. Setting Buy Stop and stop orders ](https://c.mql5.com/2/19/Fig4_price_action_inside_bar_buystop__1.png)

Fig. 4. Setting **Buy Stop** and stop orders

First, we should consider the stop level setting rules using the example above:

1. Set a Buy Stop pending order slightly higher than a mother bar's High price (only several points higher, for confirmation).
2. Set a Stop Loss level below a support level, as well as a mother bar's Low price. This is an additional protection in case a pending order is triggered and the price reaches the support level just to bounce back and start moving in the right direction later on.
3. Set a Take Profit level slightly lower than the nearest resistance level.

Do not forget that an inside bar may be followed either by a trend reversal or continuation meaning that we need a Sell Stop order as well.

![Fig. 5. Setting Sell Stop and stop orders](https://c.mql5.com/2/19/Fig5_price_action_inside_bar_sellstop__1.png)

Fig. 5. Setting **Sell Stop** and stop orders

First, we should consider the stop level setting rules using the example above:

1. Set a Sell Stop pending order slightly lower than a mother bar's Low price (only several points lower, for confirmation).
2. Set a Stop Loss level above a mother bar's High price.
3. Set a Take Profit level slightly higher than the nearest support level.

### Developing an Expert Advisor Based on Inside Bar Trading

Now that we know all the necessary rules of defining a genuine inside bar, entering the market and setting stop orders, we can finally implement the appropriate Expert Advisor that will trade using the Inside Bar pattern.

Open MetaEditor from the MetaTrader 4 terminal and create a new Expert Advisor (I believe, I do not have to dwell upon this, since the website provides plenty of information on how to create an Expert Advisor). All parameters are left blank at this stage. You can name them whatever you like. The resulting code will look as follows:

```
//+------------------------------------------------------------------+
//|                                                    InsideBar.mq4 |
//|                                  Copyright 2015, Iglakov Dmitry. |
//|                                               cjdmitri@gmail.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Iglakov Dmitry."
#property link      "cjdmitri@gmail.com"
#property version   "1.00"
#property strict
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

  }
//+------------------------------------------------------------------+
```

### Converting the Pattern into MQL4 Algorithm

After we have created the EA, we need to define an inside bar after a candle is closed. To do this, we introduce new variables and assign values to them. See the code below:

```
//+------------------------------------------------------------------+
//|                                                    InsideBar.mq4 |
//|                                  Copyright 2015, Iglakov Dmitry. |
//|                                               cjdmitri@gmail.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Iglakov Dmitry."
#property link      "cjdmitri@gmail.com"
#property version   "1.00"
#property strict

double   open1,//first candle Open price
open2,    //second candle Open price
close1,   //first candle Close price
close2,   //second candle Close price
low1,     //first candle Low price
low2,     //second candle Low price
high1,    //first candle High price
high2;    //second candle High price
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- define prices of the necessary bars
   open1        = NormalizeDouble(iOpen(Symbol(), Period(), 1), Digits);
   open2        = NormalizeDouble(iOpen(Symbol(), Period(), 2), Digits);
   close1       = NormalizeDouble(iClose(Symbol(), Period(), 1), Digits);
   close2       = NormalizeDouble(iClose(Symbol(), Period(), 2), Digits);
   low1         = NormalizeDouble(iLow(Symbol(), Period(), 1), Digits);
   low2         = NormalizeDouble(iLow(Symbol(), Period(), 2), Digits);
   high1        = NormalizeDouble(iHigh(Symbol(), Period(), 1), Digits);
   high2        = NormalizeDouble(iHigh(Symbol(), Period(), 2), Digits);
  }
//+------------------------------------------------------------------+
```

As an example, let's consider that a mother bar is bearish(bar 2), while an inside one is bullish(bar 1). Let's add a number of conditions to the OnTick() function body:

```
void OnTick()
  {
//--- define prices of the necessary bars
   open1        = NormalizeDouble(iOpen(Symbol(), Period(), 1), Digits);
   open2        = NormalizeDouble(iOpen(Symbol(), Period(), 2), Digits);
   close1       = NormalizeDouble(iClose(Symbol(), Period(), 1), Digits);
   close2       = NormalizeDouble(iClose(Symbol(), Period(), 2), Digits);
   low1         = NormalizeDouble(iLow(Symbol(), Period(), 1), Digits);
   low2         = NormalizeDouble(iLow(Symbol(), Period(), 2), Digits);
   high1        = NormalizeDouble(iHigh(Symbol(), Period(), 1), Digits);
   high2        = NormalizeDouble(iHigh(Symbol(), Period(), 2), Digits);
//--- if the second bar is bearish, while the first one is bullish
   if(open2>close2 && //the second bar is bullish
      close1>open1 && //the first bar is bearish
      high2>high1 &&  //the bar 2 High exceeds the first one's High
      open2>close1 && //the second bar's Open exceeds the first bar's Close
      low2<low1)      //the second bar's Low is lower than the first bar's Low
     {
      //--- we have listed all the conditions defining that the first bar is completely within the second one
     }
  }
```

- Create customizable variables: stop orders, slippage, order expiration time, EA magic number, trading lot. _Stop loss may be omitted, since it is to be defined according to the inside bar rules._
- Enter local variables to normalize the look of the variables.
- Stop orders are set at a certain distance from the bar price values. In order to implement that, add the **Interval** variable responsible for the interval between High/Low prices of bars and stop order levels, as well as pending order levels.
- Add the **timeBarInside** variable to avoid order re-opening on this pattern.
- Add the **bar2size** variable to ensure that a mother bar is big enough, which is a good sign that the current market is not flat.

As a result, we obtain the following code:

```
//+------------------------------------------------------------------+
//|                                                    InsideBar.mq4 |
//|                                  Copyright 2015, Iglakov Dmitry. |
//|                                               cjdmitri@gmail.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Iglakov Dmitry."
#property link      "cjdmitri@gmail.com"
#property version   "1.00"
#property strict

extern int     interval          = 20;                               //Interval
extern double  lot               = 0.1;                              //Lot Size
extern int     TP                = 300;                              //Take Profit
extern int     magic             = 555124;                           //Magic number
extern int     slippage          = 2;                                //Slippage
extern int     ExpDate           = 48;                               //Expiration Hour Order
extern int     bar2size          = 800;                              //Bar 2 Size

double   buyPrice,//define BuyStop price
buyTP,      //Take Profit BuyStop
buySL,      //Stop Loss BuyStop
sellPrice,  //define SellStop price
sellTP,     //Take Profit SellStop
sellSL;     //Stop Loss SellStop

double   open1,//first candle Open price
open2,    //second candle Open price
close1,   //first candle Close price
close2,   //second candle Close price
low1,     //first candle Low price
low2,     //second candle Low price
high1,    //first candle High price
high2;    //second candle High price

datetime _ExpDate=0;          //local variable to define a pending order expiration time
double     _bar2size;
datetime timeBarInside;         //time of the bar, at which inside bar orders were opened, to avoid re-opening
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   double   _bid     = NormalizeDouble(MarketInfo(Symbol(), MODE_BID), Digits); //define a lower price
   double   _ask     = NormalizeDouble(MarketInfo(Symbol(), MODE_ASK), Digits); //define an upper price
   double   _point   = MarketInfo(Symbol(), MODE_POINT);
//--- define prices of the necessary bars
   open1        = NormalizeDouble(iOpen(Symbol(), Period(), 1), Digits);
   open2        = NormalizeDouble(iOpen(Symbol(), Period(), 2), Digits);
   close1       = NormalizeDouble(iClose(Symbol(), Period(), 1), Digits);
   close2       = NormalizeDouble(iClose(Symbol(), Period(), 2), Digits);
   low1         = NormalizeDouble(iLow(Symbol(), Period(), 1), Digits);
   low2         = NormalizeDouble(iLow(Symbol(), Period(), 2), Digits);
   high1        = NormalizeDouble(iHigh(Symbol(), Period(), 1), Digits);
   high2        = NormalizeDouble(iHigh(Symbol(), Period(), 2), Digits);
//---
   _bar2size=NormalizeDouble(((high2-low2)/_point),0);
//--- if the second bar is bearish, while the first one is bullish
   if(timeBarInside!=iTime(Symbol(),Period(),1) && //no orders have been opened at this pattern yet
      _bar2size>bar2size && //the second bar is big enough, so the market is not flat
      open2>close2 && //the second bar is bullish
      close1>open1 && //the first bar is bearish
      high2>high1 &&  //the bar 2 High exceeds the first one's High
      open2>close1 && //the second bar's Open exceeds the first one's Close
      low2<low1)      //the second bar's Low is lower than the first one's Low
     {
      //--- we have listed all the conditions defining that the first bar is completely within the second one
      timeBarInside=iTime(Symbol(),Period(),1); //indicate that orders are already placed on this pattern
     }
  }
//+------------------------------------------------------------------+
```

### Defining Stop Order Levels

Now that all preparations are complete, we only have to define stop order levels and order prices. Also, do not forget about an order expiration time calculation.

Let's add the following code to the **OnTick****()** function body:

```
buyPrice=NormalizeDouble(high2+interval*_point,Digits);       //define an order price considering the interval
      buySL=NormalizeDouble(low2-interval*_point,Digits);     //define a stop loss considering the interval
      buyTP=NormalizeDouble(buyPrice+TP*_point,Digits);       //define a take profit
      _ExpDate=TimeCurrent()+ExpDate*60*60;                   //a pending order expiration time calculation
      sellPrice=NormalizeDouble(low2-interval*_point,Digits);
      sellSL=NormalizeDouble(high2+interval*_point,Digits);
      sellTP=NormalizeDouble(sellPrice-TP*_point,Digits);
```

### Correction of Execution Errors

If you have ever engaged in the development of Expert Advisors, you probably know that errors often happen when closing and setting orders, including waiting time, incorrect stops, etc. To eliminate such errors, we should write a separate function with a small built-in handler of basic errors.

```
//+----------------------------------------------------------------------------------------------------------------------+
//| The function opens or sets an order                                                                                  |
//| symbol      - symbol, at which a deal is performed.                                                                  |
//| cmd         - a deal (may be equal to any of the deal values).                                                       |
//| volume      - amount of lots.                                                                                        |
//| price       - Open price.                                                                                            |
//| slippage    - maximum price deviation for market buy or sell orders.                                                 |
//| stoploss    - position close price when an unprofitability level is reached (0 if there is no unprofitability level).|
//| takeprofit  - position close price when a profitability level is reached (0 if there is no profitability level).     |
//| comment     - order comment. The last part of comment can be changed by the trade server.                            |
//| magic       - order magic number. It can be used as a user-defined ID.                                               |
//| expiration  - pending order expiration time.                                                                         |
//| arrow_color - open arrow color on a chart. If the parameter is absent or equal to CLR_NONE,                          |
//|               the open arrow is not displayed on a chart.                                                            |
//+----------------------------------------------------------------------------------------------------------------------+
int OrderOpenF(string     OO_symbol,
               int        OO_cmd,
               double     OO_volume,
               double     OO_price,
               int        OO_slippage,
               double     OO_stoploss,
               double     OO_takeprofit,
               string     OO_comment,
               int        OO_magic,
               datetime   OO_expiration,
               color      OO_arrow_color)
  {
   int      result      = -1;    //result of opening an order
   int      Error       = 0;     //error when opening an order
   int      attempt     = 0;     //amount of performed attempts
   int      attemptMax  = 3;     //maximum amount of attempts
   bool     exit_loop   = false; //exit the loop
   string   lang=TerminalInfoString(TERMINAL_LANGUAGE);  //trading terminal language, for defining the language of the messages
   double   stopllvl=NormalizeDouble(MarketInfo(OO_symbol,MODE_STOPLEVEL)*MarketInfo(OO_symbol,MODE_POINT),Digits);  //minimum stop loss/ take profit level, in points
                                                                                                                     //the module provides safe order opening.
//--- check stop orders for buying
   if(OO_cmd==OP_BUY || OO_cmd==OP_BUYLIMIT || OO_cmd==OP_BUYSTOP)
     {
      double tp = (OO_takeprofit - OO_price)/MarketInfo(OO_symbol, MODE_POINT);
      double sl = (OO_price - OO_stoploss)/MarketInfo(OO_symbol, MODE_POINT);
      if(tp>0 && tp<=stopllvl)
        {
         OO_takeprofit=OO_price+stopllvl+2*MarketInfo(OO_symbol,MODE_POINT);
        }
      if(sl>0 && sl<=stopllvl)
        {
         OO_stoploss=OO_price -(stopllvl+2*MarketInfo(OO_symbol,MODE_POINT));
        }
     }
//--- check stop orders for selling
   if(OO_cmd==OP_SELL || OO_cmd==OP_SELLLIMIT || OO_cmd==OP_SELLSTOP)
     {
      double tp = (OO_price - OO_takeprofit)/MarketInfo(OO_symbol, MODE_POINT);
      double sl = (OO_stoploss - OO_price)/MarketInfo(OO_symbol, MODE_POINT);
      if(tp>0 && tp<=stopllvl)
        {
         OO_takeprofit=OO_price -(stopllvl+2*MarketInfo(OO_symbol,MODE_POINT));
        }
      if(sl>0 && sl<=stopllvl)
        {
         OO_stoploss=OO_price+stopllvl+2*MarketInfo(OO_symbol,MODE_POINT);
        }
     }
//--- while loop
   while(!exit_loop)
     {
      result=OrderSend(OO_symbol,OO_cmd,OO_volume,OO_price,OO_slippage,OO_stoploss,OO_takeprofit,OO_comment,OO_magic,OO_expiration,OO_arrow_color); //attempt to open an order using the specified parameters
      //--- if there is an error when opening an order
      if(result<0)
        {
         Error = GetLastError();                                     //assign a code to an error
         switch(Error)                                               //error enumeration
           {                                                         //order closing error enumeration and an attempt to fix them
            case  2:
               if(attempt<attemptMax)
                 {
                  attempt=attempt+1;                                 //define one more attempt
                  Sleep(3000);                                       //3 seconds of delay
                  RefreshRates();
                  break;                                             //exit switch
                 }
               if(attempt==attemptMax)
                 {
                  attempt=0;                                         //reset the amount of attempts to zero
                  exit_loop = true;                                  //exit while
                  break;                                             //exit switch
                 }
            case  3:
               RefreshRates();
               exit_loop = true;                                     //exit while
               break;                                                //exit switch
            case  4:
               if(attempt<attemptMax)
                 {
                  attempt=attempt+1;                                 //define one more attempt
                  Sleep(3000);                                       //3 seconds of delay
                  RefreshRates();
                  break;                                             //exit switch
                 }
               if(attempt==attemptMax)
                 {
                  attempt = 0;                                       //reset the amount of attempts to zero
                  exit_loop = true;                                  //exit while
                  break;                                             //exit switch
                 }
            case  5:
               exit_loop = true;                                     //exit while
               break;                                                //exit switch
            case  6:
               if(attempt<attemptMax)
                 {
                  attempt=attempt+1;                                 //define one more attempt
                  Sleep(5000);                                       //3 seconds of delay
                  break;                                             //exit switch
                 }
               if(attempt==attemptMax)
                 {
                  attempt = 0;                                       //reset the amount of attempts to zero
                  exit_loop = true;                                  //exit while
                  break;                                             //exit switch
                 }
            case  8:
               if(attempt<attemptMax)
                 {
                  attempt=attempt+1;                                 //define one more attempt
                  Sleep(7000);                                       //3 seconds of delay
                  break;                                             //exit switch
                 }
               if(attempt==attemptMax)
                 {
                  attempt = 0;                                       //reset the amount of attempts to zero
                  exit_loop = true;                                  //exit while
                  break;                                             //exit switch
                 }
            case 64:
               exit_loop = true;                                     //exit while
               break;                                                //exit switch
            case 65:
               exit_loop = true;                                     //exit while
               break;                                                //exit switch
            case 128:
               Sleep(3000);
               RefreshRates();
               continue;                                             //exit switch
            case 129:
               if(attempt<attemptMax)
                 {
                  attempt=attempt+1;                                 //define one more attempt
                  Sleep(3000);                                       //3 seconds of delay
                  RefreshRates();
                  break;                                             //exit switch
                 }
               if(attempt==attemptMax)
                 {
                  attempt = 0;                                       //reset the amount of attempts to zero
                  exit_loop = true;                                  //exit while
                  break;                                             //exit switch
                 }
            case 130:
               exit_loop=true;                                       //exit while
               break;
            case 131:
               exit_loop = true;                                     //exit while
               break;                                                //exit switch
            case 132:
               Sleep(10000);                                         //sleep for 10 seconds
               RefreshRates();                                       //update data
               //exit_loop = true;                                   //exit while
               break;                                                //exit switch
            case 133:
               exit_loop=true;                                       //exit while
               break;                                                //exit switch
            case 134:
               exit_loop=true;                                       //exit while
               break;                                                //exit switch
            case 135:
               if(attempt<attemptMax)
                 {
                  attempt=attempt+1;                                 //define one more attempt
                  RefreshRates();
                  break;                                             //exit switch
                 }
               if(attempt==attemptMax)
                 {
                  attempt = 0;                                       //set the number of attempts to zero
                  exit_loop = true;                                  //exit while
                  break;                                             //exit switch
                 }
            case 136:
               if(attempt<attemptMax)
                 {
                  attempt=attempt+1;                                 //define one more attempt
                  RefreshRates();
                  break;                                             //exit switch
                 }
               if(attempt==attemptMax)
                 {
                  attempt = 0;                                       //set the amount of attempts to zero
                  exit_loop = true;                                  //exit while
                  break;                                             //exit switch
                 }
            case 137:
               if(attempt<attemptMax)
                 {
                  attempt=attempt+1;
                  Sleep(2000);
                  RefreshRates();
                  break;
                 }
               if(attempt==attemptMax)
                 {
                  attempt=0;
                  exit_loop=true;
                  break;
                 }
            case 138:
               if(attempt<attemptMax)
                 {
                  attempt=attempt+1;
                  Sleep(1000);
                  RefreshRates();
                  break;
                 }
               if(attempt==attemptMax)
                 {
                  attempt=0;
                  exit_loop=true;
                  break;
                 }
            case 139:
               exit_loop=true;
               break;
            case 141:
               Sleep(5000);
               exit_loop=true;
               break;
            case 145:
               exit_loop=true;
               break;
            case 146:
               if(attempt<attemptMax)
                 {
                  attempt=attempt+1;
                  Sleep(2000);
                  RefreshRates();
                  break;
                 }
               if(attempt==attemptMax)
                 {
                  attempt=0;
                  exit_loop=true;
                  break;
                 }
            case 147:
               if(attempt<attemptMax)
                 {
                  attempt=attempt+1;
                  OO_expiration=0;
                  break;
                 }
               if(attempt==attemptMax)
                 {
                  attempt=0;
                  exit_loop=true;
                  break;
                 }
            case 148:
               exit_loop=true;
               break;
            default:
               Print("Error: ",Error);
               exit_loop=true; //exit while
               break;          //other options
           }
        }
      //--- if no errors detected
      else
        {
         if(lang == "Russian") {Print("Ордер успешно открыт. ", result);}
         if(lang == "English") {Print("The order is successfully opened.", result);}
         Error = 0;                                //reset the error code to zero
         break;                                    //exit while
         //errorCount =0;                          //reset the amount of attempts to zero
        }
     }
   return(result);
  }
//+------------------------------------------------------------------+
```

As a result, we obtain the following code:

```
//+------------------------------------------------------------------+
//|                                                    InsideBar.mq4 |
//|                                  Copyright 2015, Iglakov Dmitry. |
//|                                               cjdmitri@gmail.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Iglakov Dmitry."
#property link      "cjdmitri@gmail.com"
#property version   "1.00"
#property strict

extern int     interval          = 20;                               //Interval
extern double  lot               = 0.1;                              //Lot Size
extern int     TP                = 300;                              //Take Profit
extern int     magic             = 555124;                           //Magic number
extern int     slippage          = 2;                                //Slippage
extern int     ExpDate           = 48;                               //Expiration Hour Order
extern int     bar2size          = 800;                              //Bar 2 Size

double   buyPrice,//define BuyStop price
buyTP,      //Take Profit BuyStop
buySL,      //Stop Loss BuyStop
sellPrice,  //define SellStop price
sellTP,     //Take Profit SellStop
sellSL;     //Stop Loss SellStop

double   open1,//first candle Open price
open2,    //second candle Open price
close1,   //first candle Close price
close2,   //second candle Close price
low1,     //first candle Low price
low2,     //second candle Low price
high1,    //first candle High price
high2;    //second candle High price

datetime _ExpDate=0;          //local variable to define a pending order expiration time
double     _bar2size;
datetime timeBarInside;       //time of the bar, at which inside bar orders were opened, to avoid re-opening
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   double   _bid     = NormalizeDouble(MarketInfo(Symbol(), MODE_BID), Digits); //define a lower price
   double   _ask     = NormalizeDouble(MarketInfo(Symbol(), MODE_ASK), Digits); //define an upper price
   double   _point   = MarketInfo(Symbol(), MODE_POINT);
//--- define prices of the necessary bars
   open1        = NormalizeDouble(iOpen(Symbol(), Period(), 1), Digits);
   open2        = NormalizeDouble(iOpen(Symbol(), Period(), 2), Digits);
   close1       = NormalizeDouble(iClose(Symbol(), Period(), 1), Digits);
   close2       = NormalizeDouble(iClose(Symbol(), Period(), 2), Digits);
   low1         = NormalizeDouble(iLow(Symbol(), Period(), 1), Digits);
   low2         = NormalizeDouble(iLow(Symbol(), Period(), 2), Digits);
   high1        = NormalizeDouble(iHigh(Symbol(), Period(), 1), Digits);
   high2        = NormalizeDouble(iHigh(Symbol(), Period(), 2), Digits);
//---
   _bar2size=NormalizeDouble(((high2-low2)/_point),0);
//--- if the second bar is bearish, while the first one is bullish
   if(timeBarInside!=iTime(Symbol(),Period(),1) && //no orders have been opened at this pattern yet
      _bar2size>bar2size && //the second bar is big enough, so the market is not flat
      open2>close2 && //the second bar is bullish
      close1>open1 && //the first bar is bearish
      high2>high1 &&  //the bar 2 High exceeds the first one's High
      open2>close1 && //the second bar's Open exceeds the first one's Close
      low2<low1)      //the second bar's Low is lower than the first one's Low
     {
      buyPrice=NormalizeDouble(high2+interval*_point,Digits); //define an order price considering the interval
      buySL=NormalizeDouble(low2-interval*_point,Digits);     //define a stop loss considering the interval
      buyTP=NormalizeDouble(buyPrice+TP*_point,Digits);       //define a take profit
      _ExpDate=TimeCurrent()+ExpDate*60*60;                   //pending order expiration time calculation
      sellPrice=NormalizeDouble(low2-interval*_point,Digits);
      sellSL=NormalizeDouble(high2+interval*_point,Digits);
      sellTP=NormalizeDouble(sellPrice-TP*_point,Digits);
      OrderOpenF(Symbol(),OP_BUYSTOP,lot,buyPrice,slippage,buySL,buyTP,NULL,magic,_ExpDate,Blue);
      OrderOpenF(Symbol(),OP_SELLSTOP,lot,sellPrice,slippage,sellSL,sellTP,NULL,magic,_ExpDate,Blue);
      //--- we have listed all the conditions defining that the first bar is completely within the second one
      timeBarInside=iTime(Symbol(),Period(),1); //indicate that orders are already placed on this pattern
     }
  }
//+------------------------------------------------------------------+
```

Now, let's perform the compilation and check for error messages in the log.

### Testing the Expert Advisor

It is time to test our Expert Advisor. Let's launch the strategy tester and set the input parameters. I have specified the parameters as follows:

![Fig. 6. Input parameters for testing](https://c.mql5.com/2/19/Fig6_input_parameters.png)

Fig. 6. Input parameters for testing

1. Select a symbol (it is **CADJPY** in my case).
2. Be sure to set "Every tick" mode and define that testing is to be performed on history data. I have selected the entire year of 2014.
3. Set **D1** timeframe.
4. Launch the test.
5. After the test is complete, check the log. As we can see, no execution errors have occurred in the process.

Below is the EA testing journal:

![Fig. 7. Expert Advisor testing journal](https://c.mql5.com/2/19/tester_log__1.png)

Fig. 7. Expert Advisor testing journal

Make sure there are no mistakes and optimize the EA.

### Optimization

I have selected the following parameters for optimization:

![Fig. 8. Optimization parameters](https://c.mql5.com/2/19/Fig8_optimization_parameters.png)

Fig. 8. Optimization parameters

![Fig. 9. Optimization settings](https://c.mql5.com/2/19/optimis2__1.png)

Fig. 9. Optimization settings

Thus, we now have the ready-to-use robot.

### Optimization and Test Results

![Fig. 10. Test results](https://c.mql5.com/2/19/result__1.png)

Fig. 10. Test results

![Fig. 11. Test results graph](https://c.mql5.com/2/19/result3__1.png)

Fig. 11. Test results graph

### Conclusion

1. We have developed the ready-to-use Expert Advisor trading the Inside Bar pattern.
2. We have made sure that **Price Action patterns can work** even with no additional market entry filters.
3. No tricks (like Martingale or averaging) have been used.
4. The drawdown has been minimized through the correct setting of the stop orders.
5. No technical indicators have been used. The trading robot is based solely on reading a "bare" chart.

Thank you for reading! I hope this article has been helpful.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1771](https://www.mql5.com/ru/articles/1771)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1771.zip "Download all attachments in the single ZIP archive")

[insidebar.mq4](https://www.mql5.com/en/articles/download/1771/insidebar.mq4 "Download insidebar.mq4")(38.7 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Price Action. Automating the Engulfing Pattern Trading Strategy](https://www.mql5.com/en/articles/1946)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/62294)**
(24)


![EA-trader](https://c.mql5.com/avatar/avatar_na2.png)

**[EA-trader](https://www.mql5.com/en/users/ea-trader)**
\|
8 Jul 2018 at 16:04

Use 4 hour, main trend direction price action confirmation for a solid EA


![EA-trader](https://c.mql5.com/avatar/avatar_na2.png)

**[EA-trader](https://www.mql5.com/en/users/ea-trader)**
\|
8 Jul 2018 at 16:05

**Roberto Jacobs:**

If youwrongdeterminingsupport andresistance,willburnyour money.

60 TRADES WIN -  40 LOSE LITTLE= PROFITABLE STRATEGY.

![manivishnoi26](https://c.mql5.com/avatar/avatar_na2.png)

**[manivishnoi26](https://www.mql5.com/en/users/manivishnoi26)**
\|
30 Jul 2018 at 20:41

**EA-trader:**

60 TRADES WIN -  40 LOSE LITTLE= PROFITABLE STRATEGY.

Where can i download this ea


![tjanos2000](https://c.mql5.com/avatar/avatar_na2.png)

**[tjanos2000](https://www.mql5.com/en/users/tjanos2000)**
\|
13 Jan 2022 at 07:12

Thanks for the article and the [attached code](https://www.mql5.com/en/articles/24#insert-code "Article: MQL5.community - User Memo ").  I find it very informative.

It seems to me that the comments for bullish/bearish candles are mixed up:

```
    if(timeBarInside!=iTime(Symbol(),Period(),1) && //no orders have been opened at this pattern yet
      _bar2size>bar2size && //the second bar is big enough, so the market is not flat
      open2>close2 && //the second bar is bullish
      close1>open1 && //the first bar is bearish
      high2>high1 &&  //the bar 2 High exceeds the first one's High
      open2>close1 && //the second bar's Open exceeds the first one's Close
      low2<low1)      //the second bar's Low is lower than the first one's Low
     { ...
```

In case of candle price open is greater than close that is bearish while close is grater than open is bullish.

I know it might be a bit late for the correction if I am right but I have just come across with your code and had a look.

![Michael Anthony Latham](https://c.mql5.com/avatar/2021/1/600A1041-4094.jpg)

**[Michael Anthony Latham](https://www.mql5.com/en/users/muraridas)**
\|
1 Feb 2022 at 01:24

Please correct me if I'm wrong, but the EA does not appear to include code for entry around Support and [Resistance levels](https://www.mql5.com/en/articles/1742 "Article: Method of Building Resistance and Support Levels Using MQL5 "), which could be critical to it's success.


![Using Layouts and Containers for GUI Controls: The CGrid Class](https://c.mql5.com/2/20/avatar.png)[Using Layouts and Containers for GUI Controls: The CGrid Class](https://www.mql5.com/en/articles/1998)

This article presents an alternative method of GUI creation based on layouts and containers, using one layout manager — the CGrid class. The CGrid class is an auxiliary control that acts as a container for other containers and controls using a grid layout.

![Managing the MetaTrader Terminal via DLL](https://c.mql5.com/2/19/MetaTrader-dll.png)[Managing the MetaTrader Terminal via DLL](https://www.mql5.com/en/articles/1903)

The article deals with managing MetaTrader user interface elements via an auxiliary DLL library using the example of changing push notification delivery settings. The library source code and the sample script are attached to the article.

![An Introduction to Fuzzy Logic](https://c.mql5.com/2/19/avatar__4.png)[An Introduction to Fuzzy Logic](https://www.mql5.com/en/articles/1991)

Fuzzy logic expands our boundaries of mathematical logic and set theory. This article reveals the basic principles of fuzzy logic as well as describes two fuzzy inference systems using Mamdani-type and Sugeno-type models. The examples provided will describe implementation of fuzzy models based on these two systems using the FuzzyNet library for MQL5.

![Drawing Dial Gauges Using the CCanvas Class](https://c.mql5.com/2/19/gg_cases.png)[Drawing Dial Gauges Using the CCanvas Class](https://www.mql5.com/en/articles/1699)

We can find dial gauges in cars and airplanes, in industrial production and everyday life. They are used in all spheres which require quick response to behavior of a controlled value. This article describes the library of dial gauges for MetaTrader 5.

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/1771&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062762304749086825)

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