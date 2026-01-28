---
title: Cascade Order Trading Strategy Based on EMA Crossovers for MetaTrader 5
url: https://www.mql5.com/en/articles/15250
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T18:00:28.589368
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=lmiicfeznvlxyxumzewjfzyujyatmqdi&ssn=1769094027842134236&ssn_dr=0&ssn_sr=0&fv_date=1769094027&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15250&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Cascade%20Order%20Trading%20Strategy%20Based%20on%20EMA%20Crossovers%20for%20MetaTrader%205%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909402725861198&fz_uniq=5049556603163749770&sv=2552)

MetaTrader 5 / Trading


### Introduction

In this article, we are demonstrating the Cascade Order Trading Strategy of Forex Trading Expert Advisor (EA) in [MetaQuotes Language 5](https://www.metatrader5.com/en/automated-trading/mql5 "https://www.metatrader5.com/enMetaQoutesLanguage5") (MQL5)  for [MetaTrader 5](https://www.metatrader5.com/es/trading-platform "https://www.metatrader5.com/enMetaTrader5"). In this article of the MQL5 Expert Advisor using moving average crossovers as the basis for trading strategy, this MQL5 article automates trading choices on the MetaTrader 5 platform. This article incorporates essential features for position initialization, adjustment, and monitoring and makes use of the Trade.mqh library for effective order administration.

First, two exponential moving averages (EMA) with predetermined periods are initially used as part of the technique. Depending on the direction of the cross, a buy or sell signal is produced when these moving averages cross. Orders are set with specified take profit and stop loss, which is adjusted dynamically as the market forward.

Additionally, the script has a routine for identifying new bars, which are essential for guaranteeing that trading decisions are made based on completed candle formations. Furthermore, a feature to adjust current holding upon reaching profit targets is offered.

This expert recommendation, when taken as a whole demonstrates how MQL5 may be utilized to put into practice a systematic trading strategy by using automated execution and technical indicators to carry out trades by predetermined rules.

In this article, we are discussing the following topics:

1. [Explanation of Cascade Order Trading Strategy](https://www.mql5.com/en/articles/15250#para1)
2. [Implementation of EA in MQL5](https://www.mql5.com/en/articles/15250#para2)
3. [Conclusion](https://www.mql5.com/en/articles/15250#para3)

### Explanation of Cascade Order Trading Strategy

The Cascade Order Trading Strategy describes a technique in which the results or terms of earlier orders are used to determine the placement of later order. To manage and optimize entrances, exits, and position sizes based on market movements and predetermined rules, this strategy is frequently employed in trading. This is a detailed description of how a trading strategy using cascade orders could operate.

Essential components of the Cascade Order Trading Strategy:

1. Consecutive Order placement: - A cascade order strategy involves the consecutive initiation of transactions in response to the predetermined events or conditions. An initial order could be placed by a trader, for instance in response to a specific technical indicator's signal.
2. Contingent Order: - The future orders are placed based on the results of earlier transactions or the state of the market. If the market advances in your favor, this can entail placing more orders to scale into a position.
3. Scaling In and Out: - To control risk and optimize profit potential, the cascade strategy frequently entails progressively stepping into a position. On the other hand, scaling out entails decreasing position size when profit targets are met or when the market moves negatively
4. Risk Management: - To minimize losses, effective risk management is essential in cascade techniques. This entails establishing stop loss thresholds for every order or dynamically modifying them as the position changes.

Profit-Taking: when certain requirements are met, actions are taken to secure the gains. Profit targets are set for each order at all stages of the transaction. This guarantees traders profit while permitting additional upside if market circumstances demand it.

Here is the graph summarizing the cascade order trading strategy:

![Cascade Order Trading Strategy](https://c.mql5.com/2/118/IMG_20240706_190255_834__2.jpg)

### Implementation of the EA in MQL5

Firstly, the MQL5's Trade.mqh library is a strong and practical library that makes trading activities easier. It offers a high-level interface for opening, changing, and deleting positions and orders. When we include Trade.mqh it gives us access to the CTrade class, which simplifies and encapsulates many of the intricate details of trading activities, improving the read ability and maintain ability of our code.

```
#include <Trade/Trade.mqh>
CTrade obj_Trade;
```

After including the Trade.mqh, we can access the CTrade class which is instant instantiated as obj\_Trade. We have noticed the many functionalities we have for trading are encapsulated in the CTrade class. The following are some of the main techniques that the CTrade class offers:

1.  Placing an Order
2.  Modification of Order
3.  Order Termination

Next, we graduate to global variables which are very crucial to the trading strategy's operation and serve a variety of functions in the Expert Advisor (EA). Let's talk about each global variable and what it means :

-  Variable in integers

```
int handleMAFast;
int handleMASlow;
```

The handles, or IDs, for the slow and fast-moving average indicators computed by the OnInit function are stored in these variables. To retrieve the current values of these indicators, the handlers are required to access their buffers.

- Double Arrays for Moving Averages

```
double maSlow[],maFast[];
```

The values of the fast and slow-moving averages derived from the indicator buffers are kept in these arrays. For analysis and trading decisions, they are utilized to store the present and past values of the indicators.

- Double Take Profit and Stop Loss Variables

```
double takeProfit = 0;
double stopLoss = 0;
```

The take profit (TP) and stop loss (SL) levels for the trading operations are currently stored in this variable. They are utilized for putting in or changing trading orders, and they are updated according to market conditions.

-  System State Booked Variables

```
bool isBuySystemInitiated = false;
bool isSellSystemInitiated = false;
```

These booked flags monitor the buy and sell trading systems' starting state. They assist in preventing unnecessary or duplicate orders and ensuring that orders are only placed when certain criteria (like moving average crossings) are satisfied.

- Parameters for input

```
input int slPts = 300;
input int tpPts = 300;
input double lot = 0.01;
input int slPts_Min = 100;
input int fastPeriods = 10;
input int slowPeriods = 20;
```

These variables are input parameters that, without changing the code itself, enable external configuration of the EA's behavior. Traders can modify these parameters via the MetaTrader interface's EA settings. They manage variables including take-profit, stop-loss, lot size, and fast-moving periods.

After explaining the meaning of variables we have realized data that needs to be accessed across ticks and functions is stored in global variables in MQL5. This is because they serve to store vital data, including order parameters, trade status, indicator values, and user-defined settings. They are used to carry out the trading strategy, maintain positions, and respond to market conditions, different EA functions access and alter these variables.

Certainly! Let's break down the initialization part of the code. The OnInit function is responsible for setting up the initial state of the expert advisor. What happens first is creating handles for the fast and slow-moving averages using the iMA function. Check if these handles are valid. If any handle is invalid, the initialization fails. Then set the arrays maFast and maSlow to be time series arrays using ArraySetAsSeries. Finally, return success.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){

   handleMAFast = iMA(_Symbol,_Period,fastPeriods,0,MODE_EMA,PRICE_CLOSE);
   if (handleMAFast == INVALID_HANDLE){
      Print("UNABLE TO LOAD FAST MA, REVERTING NOW");
      return (INIT_FAILED);
   }

   handleMASlow = iMA(_Symbol,_Period,slowPeriods,0,MODE_EMA,PRICE_CLOSE);
   if (handleMASlow == INVALID_HANDLE){
      Print("UNABLE TO LOAD SLOW MA, REVERTING NOW");
      return (INIT_FAILED);
   }

   ArraySetAsSeries(maFast,true);
   ArraySetAsSeries(maSlow,true);

   return(INIT_SUCCEEDED);
}
```

Afterward, we look at the deinitialization function. Within the MQL5 expert advisor system, the function is invoked when the expert advisor is closed or removed from the chart. This is where cleaning of tasks takes place.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
```

In this instance, the function is empty, indicating that no particular actions are taken upon deinitializing the EA.

An integer parameter reason which specifies the cause of the deinitialization, is taken by the function. This could occur because the terminal is closed, the EA was taken off the chart, or for other reasons.

We now move to the OnTick function, which is triggered each time a new tick (price change) is received. The trading logic is applied in this function. The OnTick function in the provided code carries out the moving average crossover-based trading strategy.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){

   double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);

   if (CopyBuffer(handleMAFast,0,1,3,maFast) < 3){
      Print("NO ENOUGH DATA FROM FAST MA FOR ANALYSIS, REVERTING NOW");
      return;
   }
   if (CopyBuffer(handleMASlow,0,1,3,maSlow) < 3){
      Print("NO ENOUGH DATA FROM SLOW MA FOR ANALYSIS, REVERTING NOW");
      return;
   }

   //if (IsNewBar()){Print("FAST MA DATA:");ArrayPrint(maFast,6);}

   if (PositionsTotal()==0){
      isBuySystemInitiated=false;isSellSystemInitiated=false;
   }

   if (PositionsTotal()==0 && IsNewBar()){
      if (maFast[0] > maSlow[0] && maFast[1] < maSlow[1]){
         Print("BUY SIGNAL");
         takeProfit = Ask+tpPts*_Point;
         stopLoss = Ask-slPts*_Point;
         obj_Trade.Buy(lot,_Symbol,Ask,stopLoss,0);
         isBuySystemInitiated = true;
      }
      else if (maFast[0] < maSlow[0] && maFast[1] > maSlow[1]){
         Print("SELL SIGNAL");
         takeProfit = Bid-tpPts*_Point;
         stopLoss = Bid+slPts*_Point;
         obj_Trade.Sell(lot,_Symbol,Bid,stopLoss,0);
         isSellSystemInitiated = true;
      }
   }

   else {
      if (isBuySystemInitiated && Ask >= takeProfit){
         takeProfit = takeProfit+tpPts*_Point;
         stopLoss = Ask-slPts_Min*_Point;
         obj_Trade.Buy(lot,_Symbol,Ask,0);
         ModifyTrades(POSITION_TYPE_BUY,stopLoss);
      }
      else if (isSellSystemInitiated && Bid <= takeProfit){
         takeProfit = takeProfit-tpPts*_Point;
         stopLoss = Bid+slPts_Min*_Point;
         obj_Trade.Sell(lot,_Symbol,Bid,0);
         ModifyTrades(POSITION_TYPE_SELL,stopLoss);
      }
   }
}
```

Here is the detailed breakdown of the OnTick function for easy understanding:

Obtain Ask and Bid prices: The function obtains the Ask and Bid prices at the moment and rounds them to the appropriate number of decimal places. Retrieving and normalizing current prices is essential for a cascade order trading strategy to make well-informed decisions about placing orders at different price points. Maintaining a close eye on current market pricing is necessary when putting cascade order trading strategy into practice. This entails getting the Ask and Bid prices for the symbol you are trading in MetaTrader. The current Ask price for the designated symbol (\_Symbol) is obtained by calling the function. The price at which you can purchase this asset is known as the ask price. The current bid price for the symbol is retrieved by this function call. The price at which you can purchase this asset is known as the bid price.

```
double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);
double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);
```

Normalization ensures that prices retrieved meet trading symbols requirements for accuracy. To avoid problems with floating point math also depends on this. The value is rounded to the designated number of decimal places (\_Digits) by this function call. \_Digits which denotes the number of decimal places the symbol supports, is usually predefined for the symbol.

Duplicate the Moving Average Buffers: The function gets the most recent values for the moving averages both slow and fast. The function terminates and an error message is printed if there is insufficient data. Moving averages are used in a cascade order trading strategy to spot trades and pinpoint the based times to place orders. For the strategy to be reliable it is imperative that before beginning the study there is enough data from the moving averages.

```
if (CopyBuffer(handleMAFast,0,1,3,maFast) < 3){
      Print("NO ENOUGH DATA FROM FAST MA FOR ANALYSIS, REVERTING NOW");
      return;
   }
   if (CopyBuffer(handleMASlow,0,1,3,maSlow) < 3){
      Print("NO ENOUGH DATA FROM SLOW MA FOR ANALYSIS, REVERTING NOW");
      return;
   }
```

This function transfers information into the maFast array from the fast-moving average indicator buffer. These include the fast MA indicator's handle, the buffer number, which is usually 0 for the indicator's main line, the beginning point from which the data should be copied, the amount of data to be copied, and the array in which the data copy will be kept. The number of elements successfully copied is returned by the function. While we are verifying the data viability it determines whether the maFast had had fewer than three elements copied. Less than three components transferred indicates insufficient data from the rapidly moving average to support a solid analysis. The function ends execution by printing a message and returning early. The slow-moving average with handle MASlow and maSlow follows the same reasoning.

Here is a graph of the moving averages plotted against price data. The plot includes the price series, the fast-moving average (10-day EMA) and the slow-moving averages (20-day EMA).

![MOVING AVERAGES CROSSOVER](https://c.mql5.com/2/118/IMG_20240709_202725_584__2.jpg)

Before we make any trading decisions, the strategy makes sure that there is a minimum amount of historical data from the moving averages. Making more trustworthy and knowledgeable selections is aided by this. The strategy prevents inadequate information, which could result in false signals and possibly losing trades, by verifying that there is enough data. This snippet serves as a validation step before trading. If the condition is not satisfied, the strategy will revert and execute no trades or analyses for that tick.

Moving averages are used to determine the direction and strength of a trend in a cascade-order trading strategy. If the fast-moving average crosses above or below the slow-moving average for example you may place orders at different levels.

Check for Open Positions: The buy and sell system initiation flags are reset if there are no open positions. To prevent transaction overlap and efficiently manage the status of the trading system, it is imperative when using a cascade order trading strategy to check for open positions. One of the mechanisms we have realized in this function is to reset specific flags if there are no open spots.

```
//if (IsNewBar()){Print("FAST MA DATA:");ArrayPrint(maFast,6);}

   if (PositionsTotal()==0){
      isBuySystemInitiated=false;isSellSystemInitiated=false;
   }
```

The total number of open positions for the current trading account is returned by this function. Finding out if there are any active trades is helpful. When we assume that PositionasTotal() == 0, this condition determines if there are 0 open positions overall, which indicates that no trades are open at the moment. The flags isBuySytemInitiated and isSellSystemInitiated are set to false when there are no open positions. Resets the flag indicating whether or not the buying system has been initiated (\_isBuySystemInitiated = false;). Resets the flag indicating whether or not the sell system has been initiated (\_isSellSystemInitiated = false;).

The approach makes sure that new trades are only begun when there are no open positions by periodically checking for open positions and resetting initiation flags. By doing this overlapping trades that can result in over-exposure to risk or contradictory trading actions are avoided. We manage the trading system by initiating flags, isBuySystemInitiated and isSellSystemInitiated. These flags make sure that, until the present positions are closed, the system does not start identical buy or sell orders inside a particular trend more than once. We also get aid in keeping the strategy's logical flow intact in this check. The approach can accurately react to market conditions and trends by making sure that new orders are strongly placed when appropriate, that is when no open positions are open. We achieve better risk management by preventing numerous positions from being open at the same time. In this manner, the strategy can limit market exposure and prevent possible over-leveraging.

New Bar Detection: Based on moving average crossings, it looks for buy or sell signals if a new bar (candle) has formed and there are no open positions. To prevent multiple executions of the trading strategy, it is important to detect the emergence of a new bar. Using the finished bar's data as a basis for trading choices as opposed to possibly unstable intra-bar price changes, improves trading accuracy.

```
if (PositionsTotal()==0 && IsNewBar()){
      if (maFast[0] > maSlow[0] && maFast[1] < maSlow[1]){
         Print("BUY SIGNAL");
         takeProfit = Ask+tpPts*_Point;
         stopLoss = Ask-slPts*_Point;
         obj_Trade.Buy(lot,_Symbol,Ask,stopLoss,0);
         isBuySystemInitiated = true;
      }
      else if (maFast[0] < maSlow[0] && maFast[1] > maSlow[1]){
         Print("SELL SIGNAL");
         takeProfit = Bid-tpPts*_Point;
         stopLoss = Bid+slPts*_Point;
         obj_Trade.Sell(lot,_Symbol,Bid,stopLoss,0);
         isSellSystemInitiated = true;
      }
   }
```

When PositionsTotal == 0, this condition makes sure that the if block's function only runs if there are no open positions. By doing this, it is prevented that trades would overlap and new trades will only be opened after the closing of older ones. The function IsNewBar() determines whether a new bar has formed since the last run. Essentially, when the opening time was previously recorded, this method should return true.

In the Buy Signal, here we determine if the recent bar has seen a bullish crossover or the fast-moving average (MA) crossing over the slow-moving average. If this is the case, it computes the take profit and stop loss levels and publishes a "BUY SIGNAL" message. After that, it sets the isBuySystemInitiated flag to true and submits a buy order using obj\_Trade.Buy(). Below is an EMA Crossover Buy Signal of Cascade Order Trading Strategy:

![Buy signal](https://c.mql5.com/2/118/IMG_20240711_101023_484.jpg)

In the Sell Signal, here we determine if the recent bar represents a bearish crossing, meaning that the fast-moving average (MA) has crossed below the slow-moving average. If this is the case, it computes the take profit and stop loss levels and outputs a "SELL SIGNAL" message. After that, it uses obj\_Trade to send a sell order. The isSellSystemInitiated flag is set to true by Sell(). Below is an EMA Crossover Sell Signal of Cascade Order Trading Strategy:

![Sell Signal](https://c.mql5.com/2/118/IMG_20240709_211253_504.jpg)

The strategy makes sure that the trade logic is only run once per bar by identifying when a new bar is detected. By doing this, the issue of several orders being placed within the same bar is avoided, which can result in overtrading and higher transaction fees. When we use finished bar data to inform trading decisions guarantees that the choices are founded on trustworthy information. Price changes inside bars might be noisy and produce erroneous indications. The trading system's state is effectively managed by combining the checks for no open positions with the detection of a new bar it guarantees that fresh trades are made only when necessary, preserving the strategy's logical progression

Execute Buy or Sell Orders: - A buy order is placed if the fast-moving average crosses over the slow-moving average. It initiates a sell order if the fast MA  crosses below the slow MA. It also modifies the initiation flags and sets the take profit and stop loss (SL) settings. This "cascading" of the orders helps to gradually establish a position as the price moves in favor of the first deal.

```
else {
      if (isBuySystemInitiated && Ask >= takeProfit){
         takeProfit = takeProfit+tpPts*_Point;
         stopLoss = Ask-slPts_Min*_Point;
         obj_Trade.Buy(lot,_Symbol,Ask,0);
         ModifyTrades(POSITION_TYPE_BUY,stopLoss);
      }
```

In the Buy Order Cascades, the strategy determines whether the take profit level has been reached and whether the ask price is currently more than or equal to it. This circumstance means that an extra buy order has been triggered since the price has moved in favor of the current buy position. The take profit level is raised by a predetermined amount of points, establishing a new global for the subsequent increment in the cascade. Modification of Stop Loss ensures that the risk is controlled for the new purchase order and the stop loss level is changed to a new level below the asking price. Additional of Buy Order places a second buy order with the given lot size, effective at the current Ask price. Modification of current Trades manages risk for the overall position, the stop loss for current purchase positions is changed to the new stop loss level.

The sequential Sales Order, strategy determines whether the current Bid price is less than or equal to the takeProfit level and whether the sale system has begun. This situation means that an extra sell order has been triggered since the price has moved in favor of the current sell position. The take profit level is lowered by a predetermined amount of points, establishing a new goal for the subsequent increment in the cascade. Updating the stop loss ensures that the risk is controlled for the new sell order, the stop loss level is changed to a new level above the existing Bid price. An additional sell order with the specified lot size is executed at the current Bid price. Modifying current Trades controls risk across the board, the stop loss for current sell positions is changed to the new stop loss level.

Adjust current positions: If there are open positions and the price hits the take-profit threshold, more orders are placed and the stop loss amounts are updated appropriately. This aids in incremental profit maximization and risk management.

```
else if (isSellSystemInitiated && Bid <= takeProfit){
         takeProfit = takeProfit-tpPts*_Point;
         stopLoss = Bid+slPts_Min*_Point;
         obj_Trade.Sell(lot,_Symbol,Bid,0);
         ModifyTrades(POSITION_TYPE_SELL,stopLoss);
      }
   }
}
```

A boolean flag called isBuySystemInitiated indicates whether or not the buying system has been started. A boolean flag called isSellSystemInitiated indicates whether or not the selling system has started. The going rate for the ask, the current bid price, the amount of take profit, the number of points needed ed to modify the take-profit threshold, the level of the stop loss, the stop loss adjustment's minimal point, the trade's lot size, the ability to place a purchase order, the ability to submit a sale order, and the feature that allows you to change an existing position's stop loss.

We now verify whether the UT system has been started and whether the asking price as of right now has hit or surpassed the take-profit threshold. Rising the Take profit level by a predetermined amount of points. This enables the strategy to capture additional possible gains by setting a newer take profit goal higher than the existing one. Setting the stop loss at a new, lower price than the asking one. Locking in part of the previous gains aids in risk management. Completing additional Buy Orders at the Asking price executed the second buy order with the designated lot size at the asking price. As the price moves in the initial trades favour, this gradually increases the position. Updates all the current buy positions' stop losses to reflect the new stop loss amount. This aids in controlling the total risk associated with the buy position.

Next, we verify whether the sale mechanism has been started and whether the bid price has risen to or dropped below the take-profit threshold. Reduce Take profit level by a fixed Number of points. As a result, the new take-profit goal is less than the existing level. After updating the stop loss this will move the stop loss to a new level above the bid price. Locking in part of the previous gains aids in risk management. Complete an extra sell order with a designated lot size at the current bid price. As the price moves in the initial trade's favor, this gradually increases the position. Updates all current sell positions' stop losses to reflect the new stop loss amount. This aids in controlling the total risk associated with the sell position.

As the market shifts in favor of the first transaction, the strategy gradually increases the trading position. As a result, the trader can increase their position without taking on excessive risk at first and profit from significant trends. The strategy effectively manages risk by varying the stop loss levels for both new orders and open positions. If the market turns south this helps preserve gains and reduce losses. Whenever the market advances in the direction of the cascade strategy's preference, new take-profit levels are progressively set ito maximize profits. This permits future growth while guarenteeing a progressive lock-in of earnings.

Let's now look at the two utility functions in our MQL5 Expert Advisor (EA) code. IsNewBar and ModifyTrades. These routines provide necessary auxiliary operations to support the core trading logic that is provided in the OnTick function.

We are starting with the IsNewBar function. This function examines the chart to see if a new bar, or candle, has formed. This is necessary to guarantee that specific actions, such as opening new trades, are carried out just once per bar.

```
//+------------------------------------------------------------------+

bool IsNewBar(){
   static int prevBars = 0;
   int currBars = iBars(_Symbol,_Period);
   if (prevBars==currBars) return (false);
   prevBars = currBars;
   return (true);
}
```

The number of bars from the preceding tick is stored in the Static Variable prevBars. The variable's value is preserved in between function calls thanks to the static keyword. Present from Bar count\*\*: currBars retrieves the current bar count the graph. Comparison\*\*: The function returns false if there hasn't been a new bar generated and prevBars and currBars are equal. A new had emerged if they disagreed. The function returns true after updating prevBars to currBars.

Let's now move to the ModifiyTrades function. Based on the specified position type, the ModifyTrades function adjusts the stop loss (SL) levels of active positions.

```
void ModifyTrades(ENUM_POSITION_TYPE posType, double sl){
   for (int i=0; i<=PositionsTotal(); i++){
      ulong ticket = PositionGetTicket(i);
      if (ticket > 0){
         if (PositionSelectByTicket(ticket)){
            ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            if (type==posType){
               obj_Trade.PositionModify(ticket,sl,0);
            }
         }
      }
   }
}
```

In this function:

1. Loop over positions: uses positionsTotal() to go over all open positions.
2. Get Ticket: Returns the place at index{i}'s ticket number.
3. Verify Ticket: verify that the ticket number is real (more than 0).
4. Select Position: Uses PositionSelectByTicket} to choose the position.
5.  \*\*Check Position Type\*\*: Verifies that the specified posType matches the position type.
6. \\*\\* Modify Position\*\*: Use PositionModify to change the position type matches.

The full code of Cascade Order Trading Strategy is as below:

```
//|                                                     CASCADE ORDERING.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <Trade/Trade.mqh>
CTrade obj_Trade;

int handleMAFast;
int handleMASlow;
double maSlow[], maFast[];

double takeProfit = 0;
double stopLoss = 0;
bool isBuySystemInitiated = false;
bool isSellSystemInitiated = false;

input int slPts = 300;
input int tpPts = 300;
input double lot = 0.01;
input int slPts_Min = 100;
input int fastPeriods = 10;
input int slowPeriods = 20;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
    handleMAFast = iMA(_Symbol, _Period, fastPeriods, 0, MODE_EMA, PRICE_CLOSE);
    if (handleMAFast == INVALID_HANDLE) {
        Print("UNABLE TO LOAD FAST MA, REVERTING NOW");
        return (INIT_FAILED);
    }

    handleMASlow = iMA(_Symbol, _Period, slowPeriods, 0, MODE_EMA, PRICE_CLOSE);
    if (handleMASlow == INVALID_HANDLE) {
        Print("UNABLE TO LOAD SLOW MA, REVERTING NOW");
        return (INIT_FAILED);
    }

    ArraySetAsSeries(maFast, true);
    ArraySetAsSeries(maSlow, true);

    return (INIT_SUCCEEDED);
}
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    // Cleanup code if necessary
}
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
    double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK), _Digits);
    double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID), _Digits);

    if (CopyBuffer(handleMAFast, 0, 1, 3, maFast) < 3) {
        Print("NO ENOUGH DATA FROM FAST MA FOR ANALYSIS, REVERTING NOW");
        return;
    }
    if (CopyBuffer(handleMASlow, 0, 1, 3, maSlow) < 3) {
        Print("NO ENOUGH DATA FROM SLOW MA FOR ANALYSIS, REVERTING NOW");
        return;
    }

    if (PositionsTotal() == 0) {
        isBuySystemInitiated = false;
        isSellSystemInitiated = false;
    }

    if (PositionsTotal() == 0 && IsNewBar()) {
        if (maFast[0] > maSlow[0] && maFast[1] < maSlow[1]) {
            Print("BUY SIGNAL");
            takeProfit = Ask + tpPts * _Point;
            stopLoss = Ask - slPts * _Point;
            obj_Trade.Buy(lot, _Symbol, Ask, stopLoss, 0);
            isBuySystemInitiated = true;
        } else if (maFast[0] < maSlow[0] && maFast[1] > maSlow[1]) {
            Print("SELL SIGNAL");
            takeProfit = Bid - tpPts * _Point;
            stopLoss = Bid + slPts * _Point;
            obj_Trade.Sell(lot, _Symbol, Bid, stopLoss, 0);
            isSellSystemInitiated = true;
        }
    } else {
        if (isBuySystemInitiated && Ask >= takeProfit) {
            takeProfit = takeProfit + tpPts * _Point;
            stopLoss = Ask - slPts_Min * _Point;
            obj_Trade.Buy(lot, _Symbol, Ask, 0);
            ModifyTrades(POSITION_TYPE_BUY, stopLoss);
        } else if (isSellSystemInitiated && Bid <= takeProfit) {
            takeProfit = takeProfit - tpPts * _Point;
            stopLoss = Bid + slPts_Min * _Point;
            obj_Trade.Sell(lot, _Symbol, Bid, 0);
            ModifyTrades(POSITION_TYPE_SELL, stopLoss);
        }
    }
}

    static int prevBars = 0;
    int currBars = iBars(_Symbol, _Period);
    if (prevBars == currBars) return (false);
    prevBars = currBars;
    return (true);
}

//+------------------------------------------------------------------+
//| ModifyTrades Function                                            |
//+------------------------------------------------------------------+

void ModifyTrades(ENUM_POSITION_TYPE posType, double sl) {
    for (int i = 0; i <= PositionsTotal(); i++) {
        ulong ticket = PositionGetTicket(i);
        if (ticket > 0) {
            if (PositionSelectByTicket(ticket)) {
                ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE) PositionGetInteger(POSITION_TYPE);
                if (type == posType) {
                    obj_Trade.PositionModify(ticket, sl, 0);
                }
            }
        }
    }
}
```

### Conclusion

Finally, this MQL5 expert advisor is a great example of a well-executed moving average crossover trading strategy. The article exhibits efficient automation of trading choices by utilizing the Trade.mqh library to streamline order management and include dynamic levels for taking profit and stopping loss.

Among this expert advisor's salient attributes are:

- Initialization and Management: Buy and sell signals based on crossovers are systematically managed, and moving averages are initialized efficiently.
- Risk management: Using take-profit and stop-loss procedures to reduce risks and guarantee rewards within user-configurable bounds.
- Modularity and Flexibility: Using input parameters and global variables to customize and adapt to different market situations and trading preferences.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15250.zip "Download all attachments in the single ZIP archive")

[Cascade\_Order\_Trading\_Strategy\_Based\_on\_EMA\_Crossovers\_for\_MetaTrader\_5.mq5](https://www.mql5.com/en/articles/download/15250/cascade_order_trading_strategy_based_on_ema_crossovers_for_metatrader_5.mq5 "Download Cascade_Order_Trading_Strategy_Based_on_EMA_Crossovers_for_MetaTrader_5.mq5")(4.51 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Implementing a Bollinger Bands Trading Strategy with MQL5: A Step-by-Step Guide](https://www.mql5.com/en/articles/15394)
- [Creating a Daily Drawdown Limiter EA in MQL5](https://www.mql5.com/en/articles/15199)
- [Developing Zone Recovery Martingale strategy in MQL5](https://www.mql5.com/en/articles/15067)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/469874)**
(2)


![ceejay1962](https://c.mql5.com/avatar/avatar_na2.png)

**[ceejay1962](https://www.mql5.com/en/users/ceejay1962)**
\|
22 Aug 2024 at 02:44

**MetaQuotes:**

Check out the new article: [Cascade Order Trading Strategy Based on EMA Crossovers for MetaTrader 5](https://www.mql5.com/en/articles/15250).

Author: [Kikkih25](https://www.mql5.com/en/users/Kikkih25 "Kikkih25")

Thanks for posting this article, it has been very useful. I integrated the cascade ordering part into my own EA, and its made a big impact on profitability!

![Juan Luis De Frutos Blanco](https://c.mql5.com/avatar/2023/2/63df76f5-9ce7.jpg)

**[Juan Luis De Frutos Blanco](https://www.mql5.com/en/users/febrero59)**
\|
10 Oct 2024 at 07:43

Muchas gracias por la buena explicación y la simplicidad: Un gran trabajo formativo.

Ahora, el paso definitivo sería conseguir que fuera medianamente rentable. ¿Alguna idea?

![Portfolio Optimization in Python and MQL5](https://c.mql5.com/2/84/Portfolio_Optimization_in_Python_and_MQL5__LOGO.png)[Portfolio Optimization in Python and MQL5](https://www.mql5.com/en/articles/15288)

This article explores advanced portfolio optimization techniques using Python and MQL5 with MetaTrader 5. It demonstrates how to develop algorithms for data analysis, asset allocation, and trading signal generation, emphasizing the importance of data-driven decision-making in modern financial management and risk mitigation.

![Price Driven CGI Model: Theoretical Foundation](https://c.mql5.com/2/84/Price_Driven_CGI_Model___Theoretical_Foundation___LOGO.png)[Price Driven CGI Model: Theoretical Foundation](https://www.mql5.com/en/articles/14964)

Let's discuss the data manipulation algorithm, as we dive deeper into conceptualizing the idea of using price data to drive CGI objects. Think about transferring the effects of events, human emotions and actions on financial asset prices to a real-life model. This study delves into leveraging price data to influence the scale of a CGI object, controlling growth and emotions. These visible effects can establish a fresh analytical foundation for traders. Further insights are shared in the article.

![Developing an Expert Advisor (EA) based on the Consolidation Range Breakout strategy in MQL5](https://c.mql5.com/2/84/Developing_an_Expert_Advisor_based_on_the_Consolidation_Range_Breakout_strategy_in_MQL5___LOGO.png)[Developing an Expert Advisor (EA) based on the Consolidation Range Breakout strategy in MQL5](https://www.mql5.com/en/articles/15311)

This article outlines the steps to create an Expert Advisor (EA) that capitalizes on price breakouts after consolidation periods. By identifying consolidation ranges and setting breakout levels, traders can automate their trading decisions based on this strategy. The Expert Advisor aims to provide clear entry and exit points while avoiding false breakouts

![Creating an Interactive Graphical User Interface in MQL5 (Part 2): Adding Controls and Responsiveness](https://c.mql5.com/2/84/Creating_an_Interactive_Graphical_User_Interface_in_MQL5_0Part_2v___LOGO.png)[Creating an Interactive Graphical User Interface in MQL5 (Part 2): Adding Controls and Responsiveness](https://www.mql5.com/en/articles/15263)

Enhancing the MQL5 GUI panel with dynamic features can significantly improve the trading experience for users. By incorporating interactive elements, hover effects, and real-time data updates, the panel becomes a powerful tool for modern traders.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/15250&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049556603163749770)

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