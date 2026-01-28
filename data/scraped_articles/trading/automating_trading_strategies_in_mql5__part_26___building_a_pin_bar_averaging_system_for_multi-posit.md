---
title: Automating Trading Strategies in MQL5 (Part 26): Building a Pin Bar Averaging System for Multi-Position Trading
url: https://www.mql5.com/en/articles/19087
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T17:55:26.934395
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/19087&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049485237987159132)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 25)](https://www.mql5.com/en/articles/19077), we developed a [trendline trading](https://www.mql5.com/go?link=https://howtotrade.com/trading-strategies/trendline-strategy/ "https://howtotrade.com/trading-strategies/trendline-strategy/") system in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) that used [least squares fit](https://en.wikipedia.org/wiki/Least_squares "https://en.wikipedia.org/wiki/Least_squares") to detect support and resistance trendlines, generating automated trades based on price touches with visual feedback. In Part 26, we create a [Pin Bar](https://www.mql5.com/go?link=https://priceaction.com/price-action-university/strategies/pin-bar/ "https://priceaction.com/price-action-university/strategies/pin-bar/") Averaging program that identifies pin bar candlestick patterns to initiate trades and manages multiple positions through an averaging strategy, incorporating trailing stops, breakeven adjustments, and a dashboard for real-time monitoring. We will cover the following topics:

1. [Understanding the Pin Bar Averaging Framework](https://www.mql5.com/en/articles/19087#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/19087#para2)
3. [Backtesting](https://www.mql5.com/en/articles/19087#para3)
4. [Conclusion](https://www.mql5.com/en/articles/19087#para4)

By the end, you’ll have a powerful MQL5 strategy for pin bar-based trading, ready for customization—let’s get started!

### Understanding the Pin Bar Averaging Framework

We’re building an automated trading system that capitalizes on [pin bar candlestick patterns](https://www.mql5.com/go?link=https://priceaction.com/price-action-university/strategies/pin-bar/ "https://priceaction.com/price-action-university/strategies/pin-bar/"), which are single-candle formations characterized by a long wick and a small body, often signaling strong price reversals at key market levels. Pin bar strategies are popular in trading because they identify moments where price rejection occurs, offering high-probability entry points for trades, especially when combined with [support and resistance](https://www.mql5.com/go?link=https://priceaction.com/price-action-university/strategies/support-resistance-levels/ "https://priceaction.com/price-action-university/strategies/support-resistance-levels/") levels. Here is a visualization of some common formations.

![PIN BAR FRAMEWORK](https://c.mql5.com/2/162/Screenshot_2025-08-07_010410.png)

Our approach will focus on detecting these [pin bars](https://www.mql5.com/go?link=https://priceaction.com/price-action-university/strategies/pin-bar/ "https://priceaction.com/price-action-university/strategies/pin-bar/") within the current timeframe and utilizing an averaging strategy to open additional positions if the market moves against the initial trade. This aims to improve the overall trade outcome while managing risk through the use of trailing stops and breakeven adjustments. To achieve this, we first will identify pin bars relative to a [support or resistance level](https://www.mql5.com/go?link=https://priceaction.com/price-action-university/strategies/support-resistance-levels/ "https://priceaction.com/price-action-university/strategies/support-resistance-levels/") derived from the previous H4 candle’s close, ensuring trades align with significant market zones.

Then, we will implement an averaging mechanism to add positions at predefined price intervals, enhancing flexibility in volatile conditions. Finally, we will incorporate a dashboard to display real-time trade metrics and use visual indicators like lines to mark key levels, ensuring we can monitor and adjust our strategy effectively. Have a look at what we will be aiming to achieve, and then we can proceed to the implementation.

![STRATEGY FRAMEWORK](https://c.mql5.com/2/162/Screenshot_2025-08-07_005540.png)

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Indicators folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will start by declaring some [inputs](https://www.mql5.com/en/docs/basis/variables/inputvariables) and [global variables](https://www.mql5.com/en/docs/basis/variables/global) that will make the program more dynamic.

```
//+------------------------------------------------------------------+
//|                                      a. Pin Bar Averaging EA.mq5 |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//|                                   https://t.me/Forex_Algo_Trader |
//+------------------------------------------------------------------+
#property copyright   "Copyright 2025, Allan Munene Mutiiria."
#property link        "https://t.me/Forex_Algo_Trader"
#property version     "1.00"
#property strict

#include <Trade\Trade.mqh>                         //--- Include Trade library for trading operations
CTrade obj_Trade;                                  //--- Instantiate trade object

//+------------------------------------------------------------------+
//| Trading signal enumeration                                       |
//+------------------------------------------------------------------+
enum EnableTradingBySignal {                       //--- Define trading signal enum
   ENABLED  = 1,                                   // Enable trading signals
   DISABLED = 0                                    // Disable trading signals
};

//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input bool   useSignalMode = DISABLED;             // Set signal mode (ENABLED/DISABLED)
input int    orderDistancePips = 50;               // Set order distance (pips)
input double lotMultiplier = 1;                    // Set lot size multiplier
input bool   useRSIFilter = false;                 // Enable RSI filter
input int    magicNumber = 123456789;              // Set magic number
input double initialLotSize = 0.01;                // Set initial lot size
input int    compoundPercent = 2;                  // Set compounding percent (0 for fixed lots)
input int    maxOrders = 5;                        // Set maximum orders
input double stopLossPips = 400;                   // Set stop loss (pips)
input double takeProfitPips = 200;                 // Set take profit (pips)
input bool   useAutoTakeProfit = true;             // Enable auto take profit
input bool   useTrailingStop = true;               // Enable trailing stop
input double trailingStartPips = 15;               // Set trailing start (pips)
input double breakevenPips = 10;                   // Set breakeven (pips)
input string orderComment = "Forex_Algo_Trader";   // Set order comment
input color  lineColor = clrBlue;                  // Set line color
input int    lineWidth = 2;                        // Set line width

//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
bool   isTradingAllowed();                         //--- Declare trading allowed check
double slBreakevenMinus = 0;                       //--- Initialize breakeven minus
double normalizedPoint;                            //--- Declare normalized point
ulong  currentTicket = 0;                          //--- Initialize current ticket
double buyCount, currentBuyLot, totalBuyLots;      //--- Declare buy metrics
double sellCount, currentSellLot, totalSellLots;   //--- Declare sell metrics
double totalSum, totalSwap;                        //--- Declare total sum and swap
double buyProfit, sellProfit, totalOperations;     //--- Declare profit and operations
double buyWeightedSum, sellWeightedSum;            //--- Declare weighted sums
double buyBreakEvenPrice, sellBreakEvenPrice;      //--- Declare breakeven prices
double minBuyLot, minSellLot;                      //--- Declare minimum lot sizes
double maxSellPrice, minBuyPrice;                  //--- Declare price extremes
```

To lay the foundation for the [Pin Bar Averaging system](https://www.mql5.com/go?link=https://priceaction.com/price-action-university/strategies/pin-bar/ "https://priceaction.com/price-action-university/strategies/pin-bar/") in MQL5 to automate trading based on pin bar patterns with a robust position management system, we first include the "<Trade\\Trade.mqh>" library and instantiate "obj\_Trade" as a [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) object to handle trade operations like opening and closing positions. Then, we proceed to define the "EnableTradingBySignal" [enumeration](https://www.mql5.com/en/docs/basis/types/integer/enumeration) with "ENABLED" (1) and "DISABLED" (0) to control whether trading signals are used for position management. Next, we set up [input parameters](https://www.mql5.com/en/docs/basis/variables/inputvariables) to customize the EA: a boolean to toggle signal mode, order distance in pips, lot size multiplier, RSI filter toggle, magic number for trade identification, initial lot size, compounding percentage (0 for fixed lots), maximum orders, stop loss and take profit in pips, toggles for auto take profit and trailing stop, trailing start and breakeven in pips, order comment, and line color and width for visual indicators.

Last, we declare [global variables](https://www.mql5.com/en/docs/basis/variables/global): a function "isTradingAllowed" to check trading conditions, "slBreakevenMinus" initialized to 0 for stop loss adjustments, "normalizedPoint" for price scaling, "currentTicket" for tracking trades, counters and sums like "buyCount", "currentBuyLot", "totalBuyLots", "sellCount", "currentSellLot", "totalSellLots", "totalSum", "totalSwap", "buyProfit", "sellProfit", "totalOperations", "buyWeightedSum", "sellWeightedSum", "buyBreakEvenPrice", "sellBreakEvenPrice", "minBuyLot", "minSellLot", "maxSellPrice", and "minBuyPrice" for position metrics, establishing the EA’s core framework for pin bar detection and averaging. We can now go on to initializing the program since most of the heavy lifting will be done on tick-based production.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   normalizedPoint = _Point;                       //--- Initialize point value
   if (_Digits == 5 || _Digits == 3) {             //--- Check for 5 or 3 digit symbols
      normalizedPoint *= 10;                       //--- Adjust point value
   }
   ChartSetInteger(0, CHART_SHOW_GRID, false);     //--- Disable chart grid
   obj_Trade.SetExpertMagicNumber(magicNumber);    //--- Set magic number for trade object
   obj_Trade.SetTypeFilling(ORDER_FILLING_IOC);    //--- Set order filling type
   return(INIT_SUCCEEDED);                         //--- Return success
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   ObjectsDeleteAll(0);                            //--- Delete all chart objects
   ChartRedraw(0);                                 //--- Redraw chart
}
```

First, in the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, we initialize "normalizedPoint" to [\_Point](https://www.mql5.com/en/docs/predefined/_point) and adjust it by multiplying by 10 for 5 or 3-digit symbols using [\_Digits](https://www.mql5.com/en/docs/predefined/_digits) to ensure accurate price calculations, disable the chart grid with [ChartSetInteger](https://www.mql5.com/en/docs/chart_operations/chartsetinteger) setting [CHART\_SHOW\_GRID](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer) to false for a cleaner display, configure "obj\_Trade" with "SetExpertMagicNumber" using "magicNumber" for trade identification, set the order filling type to "ORDER\_FILLING\_IOC" with "SetTypeFilling", and return "INIT\_SUCCEEDED" to confirm successful setup. Then, we proceed to the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) event handler, where we remove all chart objects with [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/ObjectDeleteAll) to clear visual elements like the dashboard and lines that we will define later, we only did this so we can be sure to clean the chart already, and call [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/chartredraw) to refresh the chart, ensuring a clean exit. Before we go deep into the complex trading logic, let us define some helper functions that we will need to make the program dynamic and easy to maintain.

```
//+------------------------------------------------------------------+
//| Count total trades                                               |
//+------------------------------------------------------------------+
int CountTrades() {
   int positionCount = 0;                         //--- Initialize position count
   for (int trade = PositionsTotal() - 1; trade >= 0; trade--) { //--- Iterate through positions
      ulong ticket = PositionGetTicket(trade);    //--- Get position ticket
      if (ticket == 0) continue;                  //--- Skip invalid tickets
      if (PositionGetString(POSITION_SYMBOL) != Symbol() || PositionGetInteger(POSITION_MAGIC) != magicNumber) continue; //--- Skip non-matching positions
      if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL || PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) { //--- Check trade type
         positionCount++;                         //--- Increment position count
      }
   }
   return(positionCount);                         //--- Return total count
}

//+------------------------------------------------------------------+
//| Count buy trades                                                 |
//+------------------------------------------------------------------+
int CountTradesBuy() {
   int buyPositionCount = 0;                      //--- Initialize buy position count
   for (int trade = PositionsTotal() - 1; trade >= 0; trade--) { //--- Iterate through positions
      ulong ticket = PositionGetTicket(trade);    //--- Get position ticket
      if (ticket == 0) continue;                  //--- Skip invalid tickets
      if (PositionGetString(POSITION_SYMBOL) != Symbol() || PositionGetInteger(POSITION_MAGIC) != magicNumber) continue; //--- Skip non-matching positions
      if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) { //--- Check buy position
         buyPositionCount++;                      //--- Increment buy count
      }
   }
   return(buyPositionCount);                      //--- Return buy count
}

//+------------------------------------------------------------------+
//| Count sell trades                                                |
//+------------------------------------------------------------------+
int CountTradesSell() {
   int sellPositionCount = 0;                     //--- Initialize sell position count
   for (int trade = PositionsTotal() - 1; trade >= 0; trade--) { //--- Iterate through positions
      ulong ticket = PositionGetTicket(trade);    //--- Get position ticket
      if (ticket == 0) continue;                  //--- Skip invalid tickets
      if (PositionGetString(POSITION_SYMBOL) != Symbol() || PositionGetInteger(POSITION_MAGIC) != magicNumber) continue; //--- Skip non-matching positions
      if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) { //--- Check sell position
         sellPositionCount++;                     //--- Increment sell count
      }
   }
   return(sellPositionCount);                     //--- Return sell count
}

//+------------------------------------------------------------------+
//| Normalize price                                                  |
//+------------------------------------------------------------------+
double NormalizePrice(double price) {
   return(NormalizeDouble(price, _Digits));       //--- Normalize price to symbol digits
}

//+------------------------------------------------------------------+
//| Get lot digit for normalization                                  |
//+------------------------------------------------------------------+
int fnGetLotDigit() {
   double lotStepValue = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP); //--- Get lot step value
   if (lotStepValue == 1) return(0);              //--- Return 0 for step 1
   if (lotStepValue == 0.1) return(1);            //--- Return 1 for step 0.1
   if (lotStepValue == 0.01) return(2);           //--- Return 2 for step 0.01
   if (lotStepValue == 0.001) return(3);          //--- Return 3 for step 0.001
   if (lotStepValue == 0.0001) return(4);         //--- Return 4 for step 0.0001
   return(1);                                     //--- Default to 1
}

//+------------------------------------------------------------------+
//| Check buy orders for specific magic number                       |
//+------------------------------------------------------------------+
int CheckBuyOrders(int magic) {
   int buyOrderCount = 0;                         //--- Initialize buy order count
   for (int i = PositionsTotal() - 1; i >= 0; i--) { //--- Iterate through positions
      ulong ticket = PositionGetTicket(i);         //--- Get position ticket
      if (ticket == 0) continue;                  //--- Skip invalid tickets
      if (PositionGetInteger(POSITION_MAGIC) != magic) continue; //--- Skip non-matching magic
      if (PositionGetString(POSITION_SYMBOL) == Symbol()) { //--- Check symbol
         if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) { //--- Check buy position
            buyOrderCount++;                      //--- Increment buy count
            break;                                //--- Exit loop
         }
      }
   }
   return(buyOrderCount);                         //--- Return buy order count
}

//+------------------------------------------------------------------+
//| Check sell orders for specific magic number                      |
//+------------------------------------------------------------------+
int CheckSellOrders(int magic) {
   int sellOrderCount = 0;                         //--- Initialize sell order count
   for (int i = PositionsTotal() - 1; i >= 0; i--) { //--- Iterate through positions
      ulong ticket = PositionGetTicket(i);         //--- Get position ticket
      if (ticket == 0) continue;                   //--- Skip invalid tickets
      if (PositionGetInteger(POSITION_MAGIC) != magic) continue; //--- Skip non-matching magic
      if (PositionGetString(POSITION_SYMBOL) == Symbol()) { //--- Check symbol
         if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) { //--- Check sell position
            sellOrderCount++;                      //--- Increment sell count
            break;                                 //--- Exit loop
         }
      }
   }
   return(sellOrderCount);                         //--- Return sell order count
}

//+------------------------------------------------------------------+
//| Check total buy orders                                           |
//+------------------------------------------------------------------+
int CheckTotalBuyOrders(int magic) {
   int totalBuyOrderCount = 0;                      //--- Initialize total buy order count
   for (int i = PositionsTotal() - 1; i >= 0; i--) { //--- Iterate through positions
      ulong ticket = PositionGetTicket(i);          //--- Get position ticket
      if (ticket == 0) continue;                    //--- Skip invalid tickets
      if (PositionGetInteger(POSITION_MAGIC) != magic) continue; //--- Skip non-matching magic
      if (PositionGetString(POSITION_SYMBOL) == Symbol()) { //--- Check symbol
         if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) { //--- Check buy position
            totalBuyOrderCount++;                   //--- Increment buy count
         }
      }
   }
   return(totalBuyOrderCount);                      //--- Return total buy count
}

//+------------------------------------------------------------------+
//| Check total sell orders                                          |
//+------------------------------------------------------------------+
int CheckTotalSellOrders(int magic) {
   int totalSellOrderCount = 0;                      //--- Initialize total sell order count
   for (int i = PositionsTotal() - 1; i >= 0; i--) { //--- Iterate through positions
      ulong ticket = PositionGetTicket(i);           //--- Get position ticket
      if (ticket == 0) continue;                     //--- Skip invalid tickets
      if (PositionGetInteger(POSITION_MAGIC) != magic) continue; //--- Skip non-matching magic
      if (PositionGetString(POSITION_SYMBOL) == Symbol()) { //--- Check symbol
         if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) { //--- Check sell position
            totalSellOrderCount++;                   //--- Increment sell count
         }
      }
   }
   return(totalSellOrderCount);                      //--- Return total sell count
}

//+------------------------------------------------------------------+
//| Check market buy orders                                          |
//+------------------------------------------------------------------+
int CheckMarketBuyOrders() {
   int marketBuyCount = 0;                        //--- Initialize market buy count
   for (int i = PositionsTotal() - 1; i >= 0; i--) { //--- Iterate through positions
      ulong ticket = PositionGetTicket(i);         //--- Get position ticket
      if (ticket == 0) continue;                  //--- Skip invalid tickets
      if (PositionGetInteger(POSITION_MAGIC) != magicNumber) continue; //--- Skip non-matching magic
      if (PositionGetString(POSITION_SYMBOL) == Symbol()) { //--- Check symbol
         if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) { //--- Check buy position
            marketBuyCount++;                     //--- Increment buy count
         }
      }
   }
   return(marketBuyCount);                        //--- Return market buy count
}

//+------------------------------------------------------------------+
//| Check market sell orders                                         |
//+------------------------------------------------------------------+
int CheckMarketSellOrders() {
   int marketSellCount = 0;                       //--- Initialize market sell count
   for (int i = PositionsTotal() - 1; i >= 0; i--) { //--- Iterate through positions
      ulong ticket = PositionGetTicket(i);         //--- Get position ticket
      if (ticket == 0) continue;                  //--- Skip invalid tickets
      if (PositionGetInteger(POSITION_MAGIC) != magicNumber) continue; //--- Skip non-matching magic
      if (PositionGetString(POSITION_SYMBOL) == Symbol()) { //--- Check symbol
         if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) { //--- Check sell position
            marketSellCount++;                    //--- Increment sell count
         }
      }
   }
   return(marketSellCount);                       //--- Return market sell count
}

//+------------------------------------------------------------------+
//| Close all buy positions                                          |
//+------------------------------------------------------------------+
void CloseBuy() {
   while (CheckMarketBuyOrders() > 0) {           //--- Check buy orders exist
      for (int i = PositionsTotal() - 1; i >= 0; i--) { //--- Iterate through positions
         ulong ticket = PositionGetTicket(i);      //--- Get position ticket
         if (ticket == 0) continue;               //--- Skip invalid tickets
         if (PositionGetString(POSITION_SYMBOL) == Symbol() && PositionGetInteger(POSITION_MAGIC) == magicNumber) { //--- Check symbol and magic
            if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) { //--- Check buy position
               obj_Trade.PositionClose(ticket);   //--- Close position
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Close all sell positions                                         |
//+------------------------------------------------------------------+
void CloseSell() {
   while (CheckMarketSellOrders() > 0) {          //--- Check sell orders exist
      for (int i = PositionsTotal() - 1; i >= 0; i--) { //--- Iterate through positions
         ulong ticket = PositionGetTicket(i);      //--- Get position ticket
         if (ticket == 0) continue;               //--- Skip invalid tickets
         if (PositionGetString(POSITION_SYMBOL) == Symbol() && PositionGetInteger(POSITION_MAGIC) == magicNumber) { //--- Check symbol and magic
            if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) { //--- Check sell position
               obj_Trade.PositionClose(ticket);   //--- Close position
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Calculate lot size                                               |
//+------------------------------------------------------------------+
double GetLots() {
   double calculatedLot;                          //--- Initialize calculated lot
   double minLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN); //--- Get minimum lot
   double maxLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX); //--- Get maximum lot
   if (compoundPercent != 0) {                    //--- Check compounding
      calculatedLot = NormalizeDouble(AccountInfoDouble(ACCOUNT_BALANCE) * compoundPercent / 100 / 10000, fnGetLotDigit()); //--- Calculate compounded lot
      if (calculatedLot < minLot) calculatedLot = minLot; //--- Enforce minimum lot
      if (calculatedLot > maxLot) calculatedLot = maxLot; //--- Enforce maximum lot
   } else {
      calculatedLot = initialLotSize;             //--- Use fixed lot size
   }
   return(calculatedLot);                         //--- Return calculated lot
}

//+------------------------------------------------------------------+
//| Check account free margin                                        |
//+------------------------------------------------------------------+
double AccountFreeMarginCheck(string symbol, int orderType, double volume) {
   double marginRequired = 0.0;                   //--- Initialize margin required
   double price = orderType == ORDER_TYPE_BUY ? SymbolInfoDouble(symbol, SYMBOL_ASK) : SymbolInfoDouble(symbol, SYMBOL_BID); //--- Get price
   double calculatedMargin;                       //--- Declare calculated margin
   bool success = OrderCalcMargin(orderType == ORDER_TYPE_BUY ? ORDER_TYPE_BUY : ORDER_TYPE_SELL, symbol, volume, price, calculatedMargin); //--- Calculate margin
   if (success) marginRequired = calculatedMargin; //--- Set margin if successful
   return AccountInfoDouble(ACCOUNT_MARGIN_FREE) - marginRequired; //--- Return free margin
}

//+------------------------------------------------------------------+
//| Check if trading is allowed                                      |
//+------------------------------------------------------------------+
bool isTradingAllowed() {
   bool isAllowed = false;                        //--- Initialize allowed flag
   return(true);                                  //--- Return true
}
```

Here, we implement utility functions for the program to manage trade counting, position closing, lot size calculation, margin checking, and trading permissions, ensuring robust trade handling. First, we create functions to count trades: "CountTrades" tallies total positions by iterating through [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal), checking for valid tickets with [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket), matching "Symbol" and "magicNumber", and incrementing "positionCount" for buy or sell positions; "CountTradesBuy" and "CountTradesSell" count buy and sell positions respectively, filtering by [POSITION\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type) or "POSITION\_TYPE\_SELL"; "CheckBuyOrders" and "CheckSellOrders" detect at least one buy or sell position with a specific magic number, breaking after the first match; "CheckTotalBuyOrders" and "CheckTotalSellOrders" count all buy or sell positions with a magic number; and "CheckMarketBuyOrders" and "CheckMarketSellOrders" count buy or sell positions with the magic number.

Then, we proceed to implement "NormalizePrice" to normalize prices to [\_Digits](https://www.mql5.com/en/docs/predefined/_digits) using [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble), and "fnGetLotDigit" to return the appropriate decimal precision for lot sizes based on [SYMBOL\_VOLUME\_STEP](https://www.mql5.com/en/book/automation/symbols/symbols_volume) (e.g., 0 for 1, 1 for 0.1). Next, we develop "CloseBuy" and "CloseSell" to close all buy or sell positions by looping through positions, checking "Symbol" and "magicNumber", and using "obj\_Trade.PositionClose" until "CheckMarketBuyOrders" or "CheckMarketSellOrders" returns 0. Last, we implement "GetLots" to calculate lot size based on "compoundPercent" (normalizing " [AccountInfoDouble (ACCOUNT\_BALANCE)](https://www.mql5.com/en/docs/account/accountinfodouble) \\* compoundPercent / 100 / 10000" with "fnGetLotDigit", constrained by [SYMBOL\_VOLUME\_MIN](https://www.mql5.com/en/book/automation/symbols/symbols_volume) and "SYMBOL\_VOLUME\_MAX") or "initialLotSize", and "AccountFreeMarginCheck" to compute available margin by calculating required margin with [OrderCalcMargin](https://www.mql5.com/en/docs/trading/OrderCalcMargin) for the given order type and volume, and "isTradingAllowed" as a placeholder returning true. For visualization, we will need functions to draw lines and labels on the chart.

```
//+------------------------------------------------------------------+
//| Draw support/resistance line                                     |
//+------------------------------------------------------------------+
void MakeLine(double price) {
   string name = "level";                         //--- Set line name
   if (ObjectFind(0, name) != -1) {               //--- Check if line exists
      ObjectMove(0, name, 0, iTime(Symbol(), PERIOD_CURRENT, 0), price); //--- Move line
      return;                                     //--- Exit function
   }
   ObjectCreate(0, name, OBJ_HLINE, 0, 0, price); //--- Create horizontal line
   ObjectSetInteger(0, name, OBJPROP_COLOR, lineColor); //--- Set color
   ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_SOLID); //--- Set style
   ObjectSetInteger(0, name, OBJPROP_WIDTH, lineWidth); //--- Set width
   ObjectSetInteger(0, name, OBJPROP_BACK, true); //--- Set to background
}

//+------------------------------------------------------------------+
//| Create dashboard label                                           |
//+------------------------------------------------------------------+
void LABEL(string labelName, string fontName, int fontSize, int xPosition, int yPosition, color textColor, int corner, string labelText) {
   if (ObjectFind(0, labelName) < 0) {            //--- Check if label exists
      ObjectCreate(0, labelName, OBJ_LABEL, 0, 0, 0); //--- Create label
   }
   ObjectSetString(0, labelName, OBJPROP_TEXT, labelText); //--- Set label text
   ObjectSetString(0, labelName, OBJPROP_FONT, fontName); //--- Set font
   ObjectSetInteger(0, labelName, OBJPROP_FONTSIZE, fontSize); //--- Set font size
   ObjectSetInteger(0, labelName, OBJPROP_COLOR, textColor); //--- Set text color
   ObjectSetInteger(0, labelName, OBJPROP_CORNER, corner); //--- Set corner
   ObjectSetInteger(0, labelName, OBJPROP_XDISTANCE, xPosition); //--- Set x position
   ObjectSetInteger(0, labelName, OBJPROP_YDISTANCE, yPosition); //--- Set y position
}
```

To create visual elements for the program, we develop the "MakeLine" function, which will draw a horizontal line at a specified "price" to mark a support or resistance level, setting its name to "level", checking if it exists with [ObjectFind](https://www.mql5.com/en/docs/objects/ObjectFind), moving it with [ObjectMove](https://www.mql5.com/en/docs/objects/objectmove) to the current bar time from [iTime](https://www.mql5.com/en/docs/series/itime) if found, or creating it with [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) as [OBJ\_HLINE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object), and setting "OBJPROP\_COLOR" to "lineColor", [OBJPROP\_STYLE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to "STYLE\_SOLID", "OBJPROP\_WIDTH" to "lineWidth", and "OBJPROP\_BACK" to true using [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) for background placement.

Then, we proceed to implement the "LABEL" function, which will create or update dashboard labels by checking if "labelName" exists, creating an [OBJ\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) with "ObjectCreate" if not, and setting properties with "ObjectSetString" for "OBJPROP\_TEXT" to "labelText" and "OBJPROP\_FONT" to "fontName", and "ObjectSetInteger" for "OBJPROP\_FONTSIZE" to "fontSize", "OBJPROP\_COLOR" to "textColor", [OBJPROP\_CORNER](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to "corner", "OBJPROP\_XDISTANCE" to "xPosition", and "OBJPROP\_YDISTANCE" to the y-position. We can then define indicator utility functions that we will use.

```
//+------------------------------------------------------------------+
//| Calculate ATR indicator                                          |
//+------------------------------------------------------------------+
double MyiATR(string symbol, ENUM_TIMEFRAMES timeframe, int period, int shift) {
   int handle = iATR(symbol, timeframe, period);  //--- Create ATR handle
   if (handle == INVALID_HANDLE) return 0;        //--- Check invalid handle
   double buffer[1];                              //--- Declare buffer
   if (CopyBuffer(handle, 0, shift, 1, buffer) != 1) buffer[0] = 0; //--- Copy ATR value
   IndicatorRelease(handle);                      //--- Release handle
   return buffer[0];                              //--- Return ATR value
}

//+------------------------------------------------------------------+
//| Check bullish engulfing pattern                                  |
//+------------------------------------------------------------------+
bool BullishEngulfingExists() {
   if (iOpen(Symbol(), PERIOD_CURRENT, 1) <= iClose(Symbol(), PERIOD_CURRENT, 2) && iClose(Symbol(), PERIOD_CURRENT, 1) >= iOpen(Symbol(), PERIOD_CURRENT, 2) && iOpen(Symbol(), PERIOD_CURRENT, 2) - iClose(Symbol(), PERIOD_CURRENT, 2) >= 10 * _Point && iClose(Symbol(), PERIOD_CURRENT, 1) - iOpen(Symbol(), PERIOD_CURRENT, 1) >= 10 * _Point) { //--- Check bullish engulfing conditions
      return(true);                               //--- Return true
   }
   return(false);                                 //--- Return false
}

//+------------------------------------------------------------------+
//| Check bullish harami pattern                                     |
//+------------------------------------------------------------------+
bool BullishHaramiExists() {
   if (iClose(Symbol(), PERIOD_CURRENT, 2) < iOpen(Symbol(), PERIOD_CURRENT, 2) && iOpen(Symbol(), PERIOD_CURRENT, 1) < iClose(Symbol(), PERIOD_CURRENT, 1) && iOpen(Symbol(), PERIOD_CURRENT, 2) - iClose(Symbol(), PERIOD_CURRENT, 2) > MyiATR(Symbol(), PERIOD_CURRENT, 14, 2) && iOpen(Symbol(), PERIOD_CURRENT, 2) - iClose(Symbol(), PERIOD_CURRENT, 2) > 4 * (iClose(Symbol(), PERIOD_CURRENT, 1) - iOpen(Symbol(), PERIOD_CURRENT, 1))) { //--- Check bullish harami conditions
      return(true);                               //--- Return true
   }
   return(false);                                 //--- Return false
}

//+------------------------------------------------------------------+
//| Check doji at bottom pattern                                     |
//+------------------------------------------------------------------+
bool DojiAtBottomExists() {
   if (iOpen(Symbol(), PERIOD_CURRENT, 3) - iClose(Symbol(), PERIOD_CURRENT, 3) >= 8 * _Point && MathAbs(iClose(Symbol(), PERIOD_CURRENT, 2) - iOpen(Symbol(), PERIOD_CURRENT, 2)) <= 1 * _Point && iClose(Symbol(), PERIOD_CURRENT, 1) - iOpen(Symbol(), PERIOD_CURRENT, 1) >= 8 * _Point) { //--- Check doji at bottom conditions
      return(true);                               //--- Return true
   }
   return(false);                                 //--- Return false
}

//+------------------------------------------------------------------+
//| Check doji at top pattern                                        |
//+------------------------------------------------------------------+
bool DojiAtTopExists() {
   if (iClose(Symbol(), PERIOD_CURRENT, 3) - iOpen(Symbol(), PERIOD_CURRENT, 3) >= 8 * _Point && MathAbs(iClose(Symbol(), PERIOD_CURRENT, 2) - iOpen(Symbol(), PERIOD_CURRENT, 2)) <= 1 * _Point && iOpen(Symbol(), PERIOD_CURRENT, 1) - iClose(Symbol(), PERIOD_CURRENT, 1) >= 8 * _Point) { //--- Check doji at top conditions
      return(true);                               //--- Return true
   }
   return(false);                                 //--- Return false
}

//+------------------------------------------------------------------+
//| Check bearish harami pattern                                     |
//+------------------------------------------------------------------+
bool BearishHaramiExists() {
   if (iClose(Symbol(), PERIOD_CURRENT, 2) > iClose(Symbol(), PERIOD_CURRENT, 1) && iOpen(Symbol(), PERIOD_CURRENT, 2) < iOpen(Symbol(), PERIOD_CURRENT, 1) && iClose(Symbol(), PERIOD_CURRENT, 2) > iOpen(Symbol(), PERIOD_CURRENT, 2) && iOpen(Symbol(), PERIOD_CURRENT, 1) > iClose(Symbol(), PERIOD_CURRENT, 1) && iClose(Symbol(), PERIOD_CURRENT, 2) - iOpen(Symbol(), PERIOD_CURRENT, 2) > MyiATR(Symbol(), PERIOD_CURRENT, 14, 2) && iClose(Symbol(), PERIOD_CURRENT, 2) - iOpen(Symbol(), PERIOD_CURRENT, 2) > 4 * (iOpen(Symbol(), PERIOD_CURRENT, 1) - iClose(Symbol(), PERIOD_CURRENT, 1))) { //--- Check bearish harami conditions
      return(true);                               //--- Return true
   }
   return(false);                                 //--- Return false
}

//+------------------------------------------------------------------+
//| Check long up candle pattern                                     |
//+------------------------------------------------------------------+
bool LongUpCandleExists() {
   if (iOpen(Symbol(), PERIOD_CURRENT, 2) < iClose(Symbol(), PERIOD_CURRENT, 2) && iHigh(Symbol(), PERIOD_CURRENT, 2) - iLow(Symbol(), PERIOD_CURRENT, 2) >= 40 * _Point && iHigh(Symbol(), PERIOD_CURRENT, 2) - iLow(Symbol(), PERIOD_CURRENT, 2) > 2.5 * MyiATR(Symbol(), PERIOD_CURRENT, 14, 2) && iClose(Symbol(), PERIOD_CURRENT, 1) < iOpen(Symbol(), PERIOD_CURRENT, 1) && iOpen(Symbol(), PERIOD_CURRENT, 1) - iClose(Symbol(), PERIOD_CURRENT, 1) > 10 * _Point) { //--- Check long up candle conditions
      return(true);                               //--- Return true
   }
   return(false);                                 //--- Return false
}

//+------------------------------------------------------------------+
//| Check long down candle pattern                                   |
//+------------------------------------------------------------------+
bool LongDownCandleExists() {
   if (iOpen(Symbol(), PERIOD_CURRENT, 1) > iClose(Symbol(), PERIOD_CURRENT, 1) && iHigh(Symbol(), PERIOD_CURRENT, 1) - iLow(Symbol(), PERIOD_CURRENT, 1) >= 40 * _Point && iHigh(Symbol(), PERIOD_CURRENT, 1) - iLow(Symbol(), PERIOD_CURRENT, 1) > 2.5 * MyiATR(Symbol(), PERIOD_CURRENT, 14, 1)) { //--- Check long down candle conditions
      return(true);                               //--- Return true
   }
   return(false);                                 //--- Return false
}

//+------------------------------------------------------------------+
//| Check bearish engulfing pattern                                  |
//+------------------------------------------------------------------+
bool BearishEngulfingExists() {
   if (iOpen(Symbol(), PERIOD_CURRENT, 1) >= iClose(Symbol(), PERIOD_CURRENT, 2) && iClose(Symbol(), PERIOD_CURRENT, 1) <= iOpen(Symbol(), PERIOD_CURRENT, 2) && iOpen(Symbol(), PERIOD_CURRENT, 2) - iClose(Symbol(), PERIOD_CURRENT, 2) >= 10 * _Point && iClose(Symbol(), PERIOD_CURRENT, 1) - iOpen(Symbol(), PERIOD_CURRENT, 1) >= 10 * _Point) { //--- Check bearish engulfing conditions
      return(true);                               //--- Return true
   }
   return(false);                                 //--- Return false
}

//+------------------------------------------------------------------+
//| Calculate average range over 4 days                              |
//+------------------------------------------------------------------+
double AveRange4() {
   double rangeSum = 0;                           //--- Initialize range sum
   int count = 0;                                 //--- Initialize count
   int index = 1;                                 //--- Initialize index
   while (count < 4) {                            //--- Loop until 4 days
      MqlDateTime dateTime;                       //--- Declare datetime structure
      TimeToStruct(iTime(Symbol(), PERIOD_CURRENT, index), dateTime); //--- Convert time
      if (dateTime.day_of_week != 0) {            //--- Check non-Sunday
         rangeSum += iHigh(Symbol(), PERIOD_CURRENT, index) - iLow(Symbol(), PERIOD_CURRENT, index); //--- Add range
         count++;                                 //--- Increment count
      }
      index++;                                    //--- Increment index
   }
   return(rangeSum / 4.0);                        //--- Return average range
}

//+------------------------------------------------------------------+
//| Check buy pinbar                                                 |
//+------------------------------------------------------------------+
bool IsBuyPinbar() {
   double currentOpen, currentClose, currentHigh, currentLow; //--- Declare current candle variables
   double previousHigh, previousLow, previousClose, previousOpen; //--- Declare previous candle variables
   double currentRange, previousRange, currentHigherPart, currentHigherPart1; //--- Declare range variables
   currentOpen = iOpen(Symbol(), PERIOD_CURRENT, 1); //--- Get current open
   currentClose = iClose(Symbol(), PERIOD_CURRENT, 1); //--- Get current close
   currentHigh = iHigh(Symbol(), PERIOD_CURRENT, 0); //--- Get current high
   currentLow = iLow(Symbol(), PERIOD_CURRENT, 1); //--- Get current low
   previousOpen = iOpen(Symbol(), PERIOD_CURRENT, 2); //--- Get previous open
   previousClose = iClose(Symbol(), PERIOD_CURRENT, 2); //--- Get previous close
   previousHigh = iHigh(Symbol(), PERIOD_CURRENT, 2); //--- Get previous high
   previousLow = iLow(Symbol(), PERIOD_CURRENT, 2); //--- Get previous low
   currentRange = currentHigh - currentLow;       //--- Calculate current range
   previousRange = previousHigh - previousLow;    //--- Calculate previous range
   currentHigherPart = currentHigh - currentRange * 0.4; //--- Calculate higher part
   currentHigherPart1 = currentHigh - currentRange * 0.4; //--- Calculate higher part
   double averageDailyRange = AveRange4();        //--- Get average daily range
   if ((currentClose > currentHigherPart1 && currentOpen > currentHigherPart) && //--- Check close/open in higher third
       (currentRange > averageDailyRange * 0.5) && //--- Check pinbar size
       (currentLow + currentRange * 0.25 < previousLow)) { //--- Check nose length
      double lowArray[3];                         //--- Declare low array
      CopyLow(Symbol(), PERIOD_CURRENT, 3, 3, lowArray); //--- Copy low prices
      int minIndex = ArrayMinimum(lowArray);      //--- Find minimum low index
      if (lowArray[minIndex] > currentLow) return(true); //--- Confirm buy pinbar
   }
   return(false);                                 //--- Return false
}

//+------------------------------------------------------------------+
//| Check sell pinbar                                                |
//+------------------------------------------------------------------+
bool IsSellPinbar() {
   double currentOpen, currentClose, currentHigh, currentLow; //--- Declare current candle variables
   double previousHigh, previousLow, previousClose, previousOpen; //--- Declare previous candle variables
   double currentRange, previousRange, currentLowerPart, currentLowerPart1; //--- Declare range variables
   currentOpen = iOpen(Symbol(), PERIOD_CURRENT, 1); //--- Get current open
   currentClose = iClose(Symbol(), PERIOD_CURRENT, 1); //--- Get current close
   currentHigh = iHigh(Symbol(), PERIOD_CURRENT, 1); //--- Get current high
   currentLow = iLow(Symbol(), PERIOD_CURRENT, 1); //--- Get current low
   previousOpen = iOpen(Symbol(), PERIOD_CURRENT, 2); //--- Get previous open
   previousClose = iClose(Symbol(), PERIOD_CURRENT, 2); //--- Get previous close
   previousHigh = iHigh(Symbol(), PERIOD_CURRENT, 2); //--- Get previous high
   previousLow = iLow(Symbol(), PERIOD_CURRENT, 2); //--- Get previous low
   currentRange = currentHigh - currentLow;       //--- Calculate current range
   previousRange = previousHigh - previousLow;    //--- Calculate previous range
   currentLowerPart = currentLow + currentRange * 0.4; //--- Calculate lower part
   currentLowerPart1 = currentLow + currentRange * 0.4; //--- Calculate lower part
   double averageDailyRange = AveRange4();        //--- Get average daily range
   if ((currentClose < currentLowerPart1 && currentOpen < currentLowerPart) && //--- Check close/open in lower third
       (currentRange > averageDailyRange * 0.5) && //--- Check pinbar size
       (currentHigh - currentRange * 0.25 > previousHigh)) { //--- Check nose length
      double highArray[3];                        //--- Declare high array
      CopyHigh(Symbol(), PERIOD_CURRENT, 3, 3, highArray); //--- Copy high prices
      int maxIndex = ArrayMaximum(highArray);     //--- Find maximum high index
      if (highArray[maxIndex] < currentHigh) return(true); //--- Confirm sell pinbar
   }
   return(false);                                 //--- Return false
}
```

Here, we implement functions to detect candlestick patterns and calculate the Average True Range (ATR) for our system. First, we create the "MyiATR" function, which computes the ATR by creating a handle with the [iATR](https://www.mql5.com/en/docs/indicators/IATR) function for the given symbol, timeframe, and period, returning 0 if the handle is invalid, copying the ATR value into a buffer with [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer), releasing the handle with [IndicatorRelease](https://www.mql5.com/en/docs/series/indicatorrelease), and returning the ATR value.

Then, we proceed to implement candlestick pattern detection functions: "BullishEngulfingExists" checks if the current candle engulfs the previous bearish candle with significant body sizes; "BullishHaramiExists" identifies a small bullish candle within a larger bearish candle using "MyiATR" for size comparison; "DojiAtBottomExists" detects a doji between a bearish and bullish candle for a morning star pattern; "DojiAtTopExists" identifies a doji between a bullish and bearish candle for an evening star pattern; "BearishHaramiExists" checks for a small bearish candle within a larger bullish candle; "LongUpCandleExists" confirms a strong bullish candle followed by a bearish one using ATR; "LongDownCandleExists" detects a strong bearish candle; and "BearishEngulfingExists" verifies a bearish candle engulfing a bullish one.

Last, we implement "IsBuyPinbar" and "IsSellPinbar", which identify pin bars by checking if the current candle’s close and open are in the upper or lower third of its range, the range exceeds half the average daily range from "AveRange4" (which averages the high-low range over four non-Sunday days), and the pin bar’s nose extends beyond the previous candle’s low or high, confirmed by comparing recent lows or highs with [CopyLow](https://www.mql5.com/en/docs/series/copylow) or [CopyHigh](https://www.mql5.com/en/docs/series/copyhigh) and "ArrayMinimum" or [ArrayMaximum](https://www.mql5.com/en/docs/array/arraymaximum) functions. Then, we can define some functions to get the signal type for display purposes and the weighted price for position management.

```
//+------------------------------------------------------------------+
//| Analyze candlestick patterns                                     |
//+------------------------------------------------------------------+
string CandleStick_Analyzer() {
   string candlePattern, comment1 = "", comment2 = "", comment3 = ""; //--- Initialize pattern strings
   string comment4 = "", comment5 = "", comment6 = "", comment7 = ""; //--- Initialize pattern strings
   string comment8 = "", comment9 = "";                               //--- Initialize pattern strings
   if (BullishEngulfingExists()) comment1 = " Bullish Engulfing ";    //--- Check bullish engulfing
   if (BullishHaramiExists()) comment2 = " Bullish Harami ";          //--- Check bullish harami
   if (LongUpCandleExists()) comment3 = " Bullish LongUp ";           //--- Check long up candle
   if (DojiAtBottomExists()) comment4 = " MorningStar Doji ";         //--- Check morning star doji
   if (DojiAtTopExists()) comment5 = " EveningStar Doji ";            //--- Check evening star doji
   if (BearishHaramiExists()) comment6 = " Bearish Harami ";          //--- Check bearish harami
   if (BearishEngulfingExists()) comment7 = " Bearish Engulfing ";    //--- Check bearish engulfing
   if (LongDownCandleExists()) comment8 = " Bearish LongDown ";       //--- Check long down candle
   candlePattern = comment1 + comment2 + comment3 + comment4 + comment5 + comment6 + comment7 + comment8 + comment9; //--- Combine patterns
   return(candlePattern);                                             //--- Return combined pattern
}

//+------------------------------------------------------------------+
//| Calculate average price for order type                           |
//+------------------------------------------------------------------+
double rata_price(int orderType) {
   double totalVolume = 0;                        //--- Initialize total volume
   double weightedOpenSum = 0;                    //--- Initialize weighted open sum
   double averagePrice = 0;                       //--- Initialize average price
   for (int positionIndex = 0; positionIndex < PositionsTotal(); positionIndex++) { //--- Iterate through positions
      ulong ticket = PositionGetTicket(positionIndex); //--- Get position ticket
      if (ticket == 0) continue;                  //--- Skip invalid tickets
      if (PositionGetString(POSITION_SYMBOL) == Symbol() && PositionGetInteger(POSITION_MAGIC) == magicNumber && (PositionGetInteger(POSITION_TYPE) == orderType)) { //--- Check position match
         totalVolume += PositionGetDouble(POSITION_VOLUME); //--- Add volume
         weightedOpenSum += (PositionGetDouble(POSITION_VOLUME) * PositionGetDouble(POSITION_PRICE_OPEN)); //--- Add weighted open
      }
   }
   if (totalVolume != 0) {                        //--- Check non-zero volume
      averagePrice = weightedOpenSum / totalVolume; //--- Calculate average price
   }
   return(averagePrice);                          //--- Return average price
}
```

For enhanced position management, we create the "CandleStick\_Analyzer" function, which initializes string variables like "comment1" to "comment9" as empty, checks for candlestick patterns using functions like "BullishEngulfingExists" that we did alreday define, assigns descriptive strings (e.g., " Bullish Engulfing ") to corresponding variables if patterns are detected, and concatenates them into "candlePattern" to return a combined string of detected patterns for dashboard display.

Then, we proceed to implement the "rata\_price" function, which calculates the weighted average price for a specified "orderType" (buy or sell) by initializing "totalVolume" and "weightedOpenSum" to 0, iterating through "PositionsTotal" to sum [POSITION\_VOLUME](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_double) and the product of "POSITION\_VOLUME" and [POSITION\_PRICE\_OPEN](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_double) for positions matching "Symbol", "magicNumber", and "orderType" using "PositionGetTicket", [PositionGetString](https://www.mql5.com/en/docs/trading/positiongetstring), and "PositionGetInteger", and computing "averagePrice" as "weightedOpenSum / totalVolume" if "totalVolume" is non-zero, returning the result, providing critical pattern analysis for trade signals and accurate average price calculations for averaging and take-profit adjustments. For positions, we will need to get their metrics first. Let us define a logic for that.

```
//+------------------------------------------------------------------+
//| Calculate position metrics                                       |
//+------------------------------------------------------------------+
void calculatePositionMetrics() {
   buyCount = 0;                                  //--- Reset buy count
   currentBuyLot = 0;                             //--- Reset current buy lot
   totalBuyLots = 0;                              //--- Reset total buy lots
   sellCount = 0;                                 //--- Reset sell count
   currentSellLot = 0;                            //--- Reset current sell lot
   totalSellLots = 0;                             //--- Reset total sell lots
   totalSum = 0;                                  //--- Reset total sum
   totalSwap = 0;                                 //--- Reset total swap
   buyProfit = 0;                                 //--- Reset buy profit
   sellProfit = 0;                                //--- Reset sell profit
   buyWeightedSum = 0;                            //--- Reset buy weighted sum
   sellWeightedSum = 0;                           //--- Reset sell weighted sum
   buyBreakEvenPrice = 0;                         //--- Reset buy breakeven price
   sellBreakEvenPrice = 0;                        //--- Reset sell breakeven price
   minBuyLot = 9999;                              //--- Initialize min buy lot
   minSellLot = 9999;                             //--- Initialize min sell lot
   maxSellPrice = 0;                              //--- Initialize max sell price
   minBuyPrice = 999999999;                       //--- Initialize min buy price
   for (int i = 0; i < PositionsTotal(); i++) {   //--- Iterate through positions
      ulong ticket = PositionGetTicket(i);        //--- Get position ticket
      if (ticket == 0) continue;                  //--- Skip invalid tickets
      if (PositionGetString(POSITION_SYMBOL) != Symbol()) continue; //--- Skip non-matching symbols
      if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) { //--- Check buy position
         buyCount++;                              //--- Increment buy count
         totalOperations++;                       //--- Increment total operations
         currentBuyLot = PositionGetDouble(POSITION_VOLUME); //--- Set current buy lot
         buyProfit += PositionGetDouble(POSITION_PROFIT); //--- Add buy profit
         totalBuyLots += PositionGetDouble(POSITION_VOLUME); //--- Add to total buy lots
         minBuyLot = MathMin(minBuyLot, PositionGetDouble(POSITION_VOLUME)); //--- Update min buy lot
         buyWeightedSum += PositionGetDouble(POSITION_VOLUME) * PositionGetDouble(POSITION_PRICE_OPEN); //--- Add weighted open price
         minBuyPrice = MathMin(minBuyPrice, PositionGetDouble(POSITION_PRICE_OPEN)); //--- Update min buy price
      }
      if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) { //--- Check sell position
         sellCount++;                             //--- Increment sell count
         totalOperations++;                       //--- Increment total operations
         currentSellLot = PositionGetDouble(POSITION_VOLUME); //--- Set current sell lot
         sellProfit += PositionGetDouble(POSITION_PROFIT); //--- Add sell profit
         totalSellLots += PositionGetDouble(POSITION_VOLUME); //--- Add to total sell lots
         minSellLot = MathMin(minSellLot, PositionGetDouble(POSITION_VOLUME)); //--- Update min sell lot
         sellWeightedSum += PositionGetDouble(POSITION_VOLUME) * PositionGetDouble(POSITION_PRICE_OPEN); //--- Add weighted open price
         maxSellPrice = MathMax(maxSellPrice, PositionGetDouble(POSITION_PRICE_OPEN)); //--- Update max sell price
      }
   }
   if (totalBuyLots > 0) {                        //--- Check buy lots
      buyBreakEvenPrice = buyWeightedSum / totalBuyLots; //--- Calculate buy breakeven
   }
   if (totalSellLots > 0) {                       //--- Check sell lots
      sellBreakEvenPrice = sellWeightedSum / totalSellLots; //--- Calculate sell breakeven
   }
}
```

To compute essential metrics for managing multiple positions effectively, we implement the "calculatePositionMetrics" function. First, we reset key variables to zero or their respective initial value for accurate tracking. Then, we proceed to iterate through all positions using [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal), retrieving each position’s ticket with [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket), skipping invalid tickets or non-matching symbols with [PositionGetString](https://www.mql5.com/en/docs/trading/positiongetstring), and for buy positions ( [POSITION\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type)), incrementing "buyCount" and "totalOperations", setting "currentBuyLot", adding "POSITION\_PROFIT" to "buyProfit" and "POSITION\_VOLUME" to "totalBuyLots", updating "minBuyLot" with [MathMin](https://www.mql5.com/en/docs/math/mathmin), adding weighted open price to "buyWeightedSum", and updating "minBuyPrice"; for sell positions ( [POSITION\_TYPE\_SELL](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type)), we perform similar updates for sell metrics. Last, we calculate "buyBreakEvenPrice" as "buyWeightedSum / totalBuyLots" if "totalBuyLots" is positive, and "sellBreakEvenPrice" as "sellWeightedSum / totalSellLots" if "totalSellLots" is positive, providing weighted average prices for breakeven management, ensuring precise tracking of position metrics for averaging and risk control. With these functions, we are all set to begin the position opening logic. We will do this in the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   static datetime previousBarTime = 0;           //--- Store previous bar time
   if (previousBarTime != iTime(Symbol(), PERIOD_CURRENT, 0)) { //--- Check new bar
      previousBarTime = iTime(Symbol(), PERIOD_CURRENT, 0); //--- Update previous bar time
      ChartRedraw(0);                             //--- Redraw chart
   } else {
      return;                                     //--- Exit if not new bar
   }
   if (iVolume(Symbol(), PERIOD_H4, 0) > iVolume(Symbol(), PERIOD_H4, 1)) return; //--- Exit if volume increased
   double supportResistanceLevel = NormalizeDouble(iClose(Symbol(), PERIOD_H4, 1), _Digits); //--- Get support/resistance level
   ObjectDelete(0, "level");                      //--- Delete existing level line
   MakeLine(supportResistanceLevel);              //--- Draw support/resistance line
   if (SymbolInfoInteger(Symbol(), SYMBOL_SPREAD) > 150) return; //--- Exit if spread too high
   int totalBuyPositions = 0;                     //--- Initialize buy positions count
   int totalSellPositions = 0;                    //--- Initialize sell positions count
   for (int i = 0; i < PositionsTotal(); i++) {   //--- Iterate through positions
      ulong ticket = PositionGetTicket(i);        //--- Get position ticket
      if (ticket == 0) continue;                  //--- Skip invalid tickets
      if (PositionGetString(POSITION_SYMBOL) != Symbol() || PositionGetInteger(POSITION_MAGIC) != magicNumber) continue; //--- Skip non-matching positions
      if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) { //--- Check buy position
         totalBuyPositions++;                     //--- Increment buy count
      }
      if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) { //--- Check sell position
         totalSellPositions++;                    //--- Increment sell count
      }
   }
}
```

In the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler, we implement the initial logic for the pin bar averaging system to manage trading decisions and visual updates on each new bar. First, we check for a new bar by comparing "previousBarTime" (static, initialized to 0) with the current bar time from [iTime](https://www.mql5.com/en/docs/series/itime) for the current symbol and period at shift 0, updating "previousBarTime" and calling [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) if a new bar is detected, or exiting if not.

Then, we proceed to exit if the current H4 bar’s volume from [iVolume](https://www.mql5.com/en/docs/series/ivolume) exceeds the previous bar’s, avoiding high-volatility periods. Next, we calculate the support/resistance level as the normalized close price of the previous H4 bar using [iClose](https://www.mql5.com/en/docs/series/iclose) and [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble), delete any existing "level" line with [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete), and draw a new horizontal line with "MakeLine" at this level. Last, we check if the spread from [SymbolInfoInteger](https://www.mql5.com/en/docs/marketinformation/symbolinfointeger) exceeds 150 points, exiting if too high, and count open positions by iterating through [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal), using "PositionGetTicket" to get tickets, skipping invalid or non-matching symbol and "magicNumber" positions, and incrementing "totalBuyPositions" or "totalSellPositions" for buy or sell positions identified with the [PositionGetInteger](https://www.mql5.com/en/docs/trading/positiongetinteger) function. This initial setup ensures the EA processes trades only on new bars with favorable conditions and maintains an updated visual reference. Upon compilation, we get the following outcome.

![SUPPORT RESISTANCE LEVEL MARKER](https://c.mql5.com/2/162/Screenshot_2025-08-06_234418.png)

From the image, we can see that we mark the support and resistance levels on the chart dynamically. We now need to move on to adding the positions dynamically.

```
if (CheckMarketBuyOrders() < 70 && CheckMarketSellOrders() < 70) { //--- Check order limits
   if (supportResistanceLevel > iOpen(Symbol(), PERIOD_CURRENT, 0) && useSignalMode == DISABLED) { //--- Check buy condition
      if (IsBuyPinbar() && totalBuyPositions < maxOrders && (isTradingAllowed() || totalBuyPositions > 0)) { //--- Check buy pinbar and limits
         double buyStopLoss = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK) - stopLossPips * normalizedPoint, _Digits); //--- Calculate buy stop loss
         double buyTakeProfit = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK) + takeProfitPips * normalizedPoint, _Digits); //--- Calculate buy take profit
         if (AccountFreeMarginCheck(Symbol(), ORDER_TYPE_BUY, GetLots()) > 0) { //--- Check margin
            obj_Trade.PositionOpen(Symbol(), ORDER_TYPE_BUY, GetLots(), SymbolInfoDouble(_Symbol, SYMBOL_ASK), buyStopLoss, buyTakeProfit, orderComment); //--- Open buy position
            if (useAutoTakeProfit) {             //--- Check auto take profit
               ModifyTP(ORDER_TYPE_BUY, rata_price(ORDER_TYPE_BUY) + takeProfitPips * normalizedPoint); //--- Modify take profit
            }
            CloseSell();                         //--- Close sell positions
         }
      }
   }
   if (supportResistanceLevel < iOpen(Symbol(), PERIOD_CURRENT, 0) && useSignalMode == DISABLED) { //--- Check sell condition
      if (IsSellPinbar() && totalSellPositions < maxOrders && (isTradingAllowed() || totalSellPositions > 0)) { //--- Check sell pinbar and limits
         double sellStopLoss = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID) + stopLossPips * normalizedPoint, _Digits); //--- Calculate sell stop loss
         double sellTakeProfit = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID) - takeProfitPips * normalizedPoint, _Digits); //--- Calculate sell take profit
         if (AccountFreeMarginCheck(Symbol(), ORDER_TYPE_SELL, GetLots()) > 0) { //--- Check margin
            obj_Trade.PositionOpen(Symbol(), ORDER_TYPE_SELL, GetLots(), SymbolInfoDouble(_Symbol, SYMBOL_BID), sellStopLoss, sellTakeProfit, orderComment); //--- Open sell position
            if (useAutoTakeProfit) {             //--- Check auto take profit
               ModifyTP(ORDER_TYPE_SELL, rata_price(ORDER_TYPE_SELL) - takeProfitPips * normalizedPoint); //--- Modify take profit
            }
            CloseBuy();                          //--- Close buy positions
         }
      }
   }
}
if (CountTrades() == 0) {                       //--- Check no trades
   if (supportResistanceLevel > iOpen(Symbol(), PERIOD_CURRENT, 0) && useSignalMode == ENABLED) { //--- Check buy signal mode
      if (IsBuyPinbar() && CountTrades() < maxOrders) { //--- Check buy pinbar and limit
         obj_Trade.PositionOpen(Symbol(), ORDER_TYPE_BUY, GetLots(), SymbolInfoDouble(_Symbol, SYMBOL_ASK), SymbolInfoDouble(_Symbol, SYMBOL_ASK) - stopLossPips * normalizedPoint, SymbolInfoDouble(_Symbol, SYMBOL_ASK) + (takeProfitPips * normalizedPoint), orderComment); //--- Open buy position
      }
   }
}
if (CountTrades() == 0) {                       //--- Check no trades
   if (supportResistanceLevel < iOpen(Symbol(), PERIOD_CURRENT, 0) && useSignalMode == ENABLED) { //--- Check sell signal mode
      if (IsSellPinbar() && CountTrades() < maxOrders) { //--- Check sell pinbar and limit
         obj_Trade.PositionOpen(Symbol(), ORDER_TYPE_SELL, GetLots(), SymbolInfoDouble(_Symbol, SYMBOL_BID), SymbolInfoDouble(_Symbol, SYMBOL_BID) + stopLossPips * normalizedPoint, SymbolInfoDouble(_Symbol, SYMBOL_BID) - (takeProfitPips * normalizedPoint), orderComment); //--- Open sell position
      }
   }
}
```

We continue the tick function implementation, adding logic to open new positions based on pin bar signals and market conditions. First, we check if open buy and sell orders are below 70 using "CheckMarketBuyOrders" and "CheckMarketSellOrders", ensuring the EA doesn’t exceed practical limits. Then, if "useSignalMode" is "DISABLED", we evaluate buy conditions: when "supportResistanceLevel" exceeds the current open price from [iOpen](https://www.mql5.com/en/docs/series/iopen), a buy pin bar is detected with "IsBuyPinbar", "totalBuyPositions" is below "maxOrders", and trading is allowed via "isTradingAllowed" or existing buys exist, we calculate "buyStopLoss" and "buyTakeProfit" using [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) with "stopLossPips" and "takeProfitPips" adjusted by "normalizedPoint", verify margin with "AccountFreeMarginCheck", open a buy position with "obj\_Trade.

PositionOpen" using "GetLots", modify take profit with "ModifyTP" if "useAutoTakeProfit" is true, and close sell positions with "CloseSell"; similar logic applies for sell conditions when "supportResistanceLevel" is below the open price, using "IsSellPinbar". Next, if no trades exist ("CountTrades" is 0) and "useSignalMode" is "ENABLED", we open a buy position on a buy pin bar with "IsBuyPinbar" and "CountTrades" below "maxOrders", using "obj\_Trade.PositionOpen" with calculated stop loss and take profit, and similarly for sell positions with "IsSellPinbar", ensuring the EA opens positions based on pin bar signals at key levels with proper risk management. Upon compilation, we get the following outcome.

![CONFIRMED SIGNAL](https://c.mql5.com/2/162/Screenshot_2025-08-07_000006.png)

Since we now confirm signals and open positions, we need to manage the signals. What we will do is define some functions for that.

```
//+------------------------------------------------------------------+
//| Update stop loss and take profit                                 |
//+------------------------------------------------------------------+
void updateStopLossTakeProfit() {
   for (int i = PositionsTotal() - 1; i >= 0; i--) { //--- Iterate through positions
      ulong ticket = PositionGetTicket(i);           //--- Get position ticket
      if (ticket == 0) continue;                     //--- Skip invalid tickets
      if (PositionGetString(POSITION_SYMBOL) != Symbol()) continue; //--- Skip non-matching symbols
      if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) { //--- Check buy position
         double buyTakeProfitLevel = (buyBreakEvenPrice + takeProfitPips * _Point) * (takeProfitPips > 0); //--- Calculate buy take profit
         double buyStopLossLevel = PositionGetDouble(POSITION_SL); //--- Get current stop loss
         if (slBreakevenMinus > 0) {                 //--- Check breakeven adjustment
            buyStopLossLevel = (buyBreakEvenPrice - slBreakevenMinus * _Point); //--- Set breakeven stop loss
         }
         if (buyCount == 1) {                        //--- Check single buy position
            buyTakeProfitLevel = NormalizePrice(PositionGetDouble(POSITION_PRICE_OPEN) + takeProfitPips * _Point) * (takeProfitPips > 0); //--- Set take profit
            if (laterUseSL > 0) {                    //--- Check unused stop loss
               buyStopLossLevel = (PositionGetDouble(POSITION_PRICE_OPEN) - laterUseSL * _Point); //--- Set stop loss
            }
         }
         buyTakeProfitLevel = NormalizePrice(buyTakeProfitLevel); //--- Normalize take profit
         buyStopLossLevel = NormalizePrice(buyStopLossLevel); //--- Normalize stop loss
         if (SymbolInfoDouble(_Symbol, SYMBOL_BID) >= buyTakeProfitLevel && buyTakeProfitLevel > 0) { //--- Check take profit hit
            obj_Trade.PositionClose(ticket);         //--- Close position
         }
         if (SymbolInfoDouble(_Symbol, SYMBOL_BID) <= buyStopLossLevel) { //--- Check stop loss hit
            obj_Trade.PositionClose(ticket);         //--- Close position
         }
         if (NormalizePrice(PositionGetDouble(POSITION_TP)) != buyTakeProfitLevel || NormalizePrice(PositionGetDouble(POSITION_SL)) != buyStopLossLevel) { //--- Check modification needed
            obj_Trade.PositionModify(ticket, buyStopLossLevel, buyTakeProfitLevel); //--- Modify position
         }
      }
      if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) { //--- Check sell position
         double sellTakeProfitLevel = (sellBreakEvenPrice - takeProfitPips * _Point) * (takeProfitPips > 0); //--- Calculate sell take profit
         double sellStopLossLevel = PositionGetDouble(POSITION_SL); //--- Get current stop loss
         if (slBreakevenMinus > 0) {                //--- Check breakeven adjustment
            sellStopLossLevel = (sellBreakEvenPrice + slBreakevenMinus * _Point); //--- Set breakeven stop loss
         }
         if (sellCount == 1) {                      //--- Check single sell position
            sellTakeProfitLevel = (PositionGetDouble(POSITION_PRICE_OPEN) - takeProfitPips * _Point) * (takeProfitPips > 0); //--- Set take profit
            if (laterUseSL > 0) {                   //--- Check unused stop loss
               sellStopLossLevel = (PositionGetDouble(POSITION_PRICE_OPEN) + laterUseSL * _Point); //--- Set stop loss
            }
         }
         sellTakeProfitLevel = NormalizePrice(sellTakeProfitLevel); //--- Normalize take profit
         sellStopLossLevel = NormalizePrice(sellStopLossLevel); //--- Normalize stop loss
         if (SymbolInfoDouble(_Symbol, SYMBOL_ASK) <= sellTakeProfitLevel) { //--- Check take profit hit
            obj_Trade.PositionClose(ticket);        //--- Close position
         }
         if (SymbolInfoDouble(_Symbol, SYMBOL_ASK) >= sellStopLossLevel && sellStopLossLevel > 0) { //--- Check stop loss hit
            obj_Trade.PositionClose(ticket);        //--- Close position
         }
         if (NormalizePrice(PositionGetDouble(POSITION_TP)) != sellTakeProfitLevel || NormalizePrice(PositionGetDouble(POSITION_SL)) != sellStopLossLevel) { //--- Check modification needed
            obj_Trade.PositionModify(ticket, sellStopLossLevel, sellTakeProfitLevel); //--- Modify position
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Add averaging order                                              |
//+------------------------------------------------------------------+
void addAveragingOrder() {
   int positionIndex = 0;                         //--- Initialize position index
   double lastOpenPrice = 0;                      //--- Initialize last open price
   double lastLotSize = 0;                        //--- Initialize last lot size
   bool isLastBuy = false;                        //--- Initialize buy flag
   int totalBuyPositions = 0;                     //--- Initialize buy positions count
   int totalSellPositions = 0;                    //--- Initialize sell positions count
   long currentSpread = SymbolInfoInteger(Symbol(), SYMBOL_SPREAD); //--- Get current spread
   double supportResistanceLevel = iClose(Symbol(), PERIOD_H4, 1); //--- Get support/resistance level
   for (positionIndex = 0; positionIndex < PositionsTotal(); positionIndex++) { //--- Iterate through positions
      ulong ticket = PositionGetTicket(positionIndex); //--- Get position ticket
      if (ticket == 0) continue;                  //--- Skip invalid tickets
      if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY && PositionGetString(POSITION_SYMBOL) == Symbol() && PositionGetInteger(POSITION_MAGIC) == magicNumber) { //--- Check buy position
         if (lastOpenPrice == 0) {                //--- Check initial price
            lastOpenPrice = PositionGetDouble(POSITION_PRICE_OPEN); //--- Set initial price
         }
         if (lastOpenPrice > PositionGetDouble(POSITION_PRICE_OPEN)) { //--- Check lower price
            lastOpenPrice = PositionGetDouble(POSITION_PRICE_OPEN); //--- Update last price
         }
         if (lastLotSize < PositionGetDouble(POSITION_VOLUME)) { //--- Check larger lot
            lastLotSize = PositionGetDouble(POSITION_VOLUME); //--- Update lot size
         }
         isLastBuy = true;                        //--- Set buy flag
         totalBuyPositions++;                     //--- Increment buy count
      }
      if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL && PositionGetString(POSITION_SYMBOL) == Symbol() && PositionGetInteger(POSITION_MAGIC) == magicNumber) { //--- Check sell position
         if (lastOpenPrice == 0) {                //--- Check initial price
            lastOpenPrice = PositionGetDouble(POSITION_PRICE_OPEN); //--- Set initial price
         }
         if (lastOpenPrice < PositionGetDouble(POSITION_PRICE_OPEN)) { //--- Check higher price
            lastOpenPrice = PositionGetDouble(POSITION_PRICE_OPEN); //--- Update last price
         }
         if (lastLotSize < PositionGetDouble(POSITION_VOLUME)) { //--- Check larger lot
            lastLotSize = PositionGetDouble(POSITION_VOLUME); //--- Update lot size
         }
         isLastBuy = false;                       //--- Clear buy flag
         totalSellPositions++;                    //--- Increment sell count
      }
   }
   if (isLastBuy) {                               //--- Check buy position
      if (supportResistanceLevel > iOpen(Symbol(), PERIOD_CURRENT, 0)) { //--- Check buy condition
         if (IsBuyPinbar() && SymbolInfoDouble(_Symbol, SYMBOL_BID) <= lastOpenPrice - (orderDistancePips * _Point)) { //--- Check buy pinbar and distance
            obj_Trade.PositionOpen(Symbol(), ORDER_TYPE_BUY, NormalizeDouble((lastLotSize * lotMultiplier), fnGetLotDigit()), SymbolInfoDouble(_Symbol, SYMBOL_ASK), SymbolInfoDouble(_Symbol, SYMBOL_ASK) - stopLossPips * normalizedPoint, SymbolInfoDouble(_Symbol, SYMBOL_ASK) + (takeProfitPips * normalizedPoint), orderComment); //--- Open buy position
            isLastBuy = false;                    //--- Clear buy flag
            return;                               //--- Exit function
         }
      }
   } else if (!isLastBuy) {                       //--- Check sell position
      if (supportResistanceLevel < iOpen(Symbol(), PERIOD_CURRENT, 0)) { //--- Check sell condition
         if (IsSellPinbar() && SymbolInfoDouble(_Symbol, SYMBOL_ASK) >= lastOpenPrice + (orderDistancePips * _Point)) { //--- Check sell pinbar and distance
            obj_Trade.PositionOpen(Symbol(), ORDER_TYPE_SELL, NormalizeDouble((lastLotSize * lotMultiplier), fnGetLotDigit()), SymbolInfoDouble(_Symbol, SYMBOL_BID), SymbolInfoDouble(_Symbol, SYMBOL_BID) + stopLossPips * normalizedPoint, SymbolInfoDouble(_Symbol, SYMBOL_BID) - (takeProfitPips * normalizedPoint), orderComment); //--- Open sell position
            return;                               //--- Exit function
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Add averaging order with auto take profit                        |
//+------------------------------------------------------------------+
void addAveragingOrderWithAutoTP() {
   int positionIndex = 0;                         //--- Initialize position index
   double lastOpenPrice = 0;                      //--- Initialize last open price
   double lastLotSize = 0;                        //--- Initialize last lot size
   bool isLastBuy = false;                        //--- Initialize buy flag
   int totalBuyPositions = 0;                     //--- Initialize buy positions count
   int totalSellPositions = 0;                    //--- Initialize sell positions count
   long currentSpread = SymbolInfoInteger(Symbol(), SYMBOL_SPREAD); //--- Get current spread
   double supportResistanceLevel = iClose(Symbol(), PERIOD_H4, 1); //--- Get support/resistance level
   for (positionIndex = 0; positionIndex < PositionsTotal(); positionIndex++) { //--- Iterate through positions
      ulong ticket = PositionGetTicket(positionIndex); //--- Get position ticket
      if (ticket == 0) continue;                  //--- Skip invalid tickets
      if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY && PositionGetString(POSITION_SYMBOL) == Symbol() && PositionGetInteger(POSITION_MAGIC) == magicNumber) { //--- Check buy position
         if (lastOpenPrice == 0) {                //--- Check initial price
            lastOpenPrice = PositionGetDouble(POSITION_PRICE_OPEN); //--- Set initial price
         }
         if (lastOpenPrice > PositionGetDouble(POSITION_PRICE_OPEN)) { //--- Check lower price
            lastOpenPrice = PositionGetDouble(POSITION_PRICE_OPEN); //--- Update last price
         }
         if (lastLotSize < PositionGetDouble(POSITION_VOLUME)) { //--- Check larger lot
            lastLotSize = PositionGetDouble(POSITION_VOLUME); //--- Update lot size
         }
         isLastBuy = true;                        //--- Set buy flag
         totalBuyPositions++;                     //--- Increment buy count
      }
      if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL && PositionGetString(POSITION_SYMBOL) == Symbol() && PositionGetInteger(POSITION_MAGIC) == magicNumber) { //--- Check sell position
         if (lastOpenPrice == 0) {                //--- Check initial price
            lastOpenPrice = PositionGetDouble(POSITION_PRICE_OPEN); //--- Set initial price
         }
         if (lastOpenPrice < PositionGetDouble(POSITION_PRICE_OPEN)) { //--- Check higher price
            lastOpenPrice = PositionGetDouble(POSITION_PRICE_OPEN); //--- Update last price
         }
         if (lastLotSize < PositionGetDouble(POSITION_VOLUME)) { //--- Check larger lot
            lastLotSize = PositionGetDouble(POSITION_VOLUME); //--- Update lot size
         }
         isLastBuy = false;                       //--- Clear buy flag
         totalSellPositions++;                    //--- Increment sell count
      }
   }
   if (isLastBuy) {                               //--- Check buy position
      if (supportResistanceLevel > iOpen(Symbol(), PERIOD_CURRENT, 0)) { //--- Check buy condition
         if (IsBuyPinbar() && SymbolInfoDouble(_Symbol, SYMBOL_BID) <= lastOpenPrice - (orderDistancePips * _Point)) { //--- Check buy pinbar and distance
            obj_Trade.PositionOpen(Symbol(), ORDER_TYPE_BUY, NormalizeDouble((lastLotSize * lotMultiplier), fnGetLotDigit()), SymbolInfoDouble(_Symbol, SYMBOL_ASK), 0, 0, orderComment); //--- Open buy position
            calculatePositionMetrics();           //--- Calculate position metrics
            updateStopLossTakeProfit();           //--- Update stop loss and take profit
            isLastBuy = false;                    //--- Clear buy flag
            return;                               //--- Exit function
         }
      }
   } else if (!isLastBuy) {                       //--- Check sell position
      if (supportResistanceLevel < iOpen(Symbol(), PERIOD_CURRENT, 0)) { //--- Check sell condition
         if (IsSellPinbar() && SymbolInfoDouble(_Symbol, SYMBOL_ASK) >= lastOpenPrice + (orderDistancePips * _Point)) { //--- Check sell pinbar and distance
            obj_Trade.PositionOpen(Symbol(), ORDER_TYPE_SELL, NormalizeDouble((lastLotSize * lotMultiplier), fnGetLotDigit()), SymbolInfoDouble(_Symbol, SYMBOL_BID), 0, 0, orderComment); //--- Open sell position
            calculatePositionMetrics();           //--- Calculate position metrics
            updateStopLossTakeProfit();           //--- Update stop loss and take profit
            return;                               //--- Exit function
         }
      }
   }
}
```

Here, we implement the "updateStopLossTakeProfit" and "addAveragingOrder" functions, along with "addAveragingOrderWithAutoTP", to manage stop loss, take profit, and averaging trades, ensuring dynamic position adjustments. First, we develop the "updateStopLossTakeProfit" function, which iterates through all positions. For buy positions ( [POSITION\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type)), we calculate "buyTakeProfitLevel" based on "buyBreakEvenPrice" plus "takeProfitPips \* \_Point" if "takeProfitPips" is positive, get the current stop loss with [PositionGetDouble](https://www.mql5.com/en/docs/trading/positiongetdouble), adjust it to "buyBreakEvenPrice - slBreakevenMinus \* \_Point" if "slBreakevenMinus" is positive, or for single positions ("buyCount == 1"), set take profit and stop loss based on [POSITION\_PRICE\_OPEN](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_double) adjusted by take profit and stop loss, normalize both levels with "NormalizePrice", close positions with "obj\_Trade.PositionClose" if the bid price hits take profit or stop loss, and modify positions with "obj\_Trade.PositionModify" if levels differ; similar logic applies for sell positions using "sellBreakEvenPrice" and ask price.

Then, we proceed to implement the "addAveragingOrder" function, which tracks the latest position by iterating through [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal), updating "lastOpenPrice" to the lowest buy or highest sell price and "lastLotSize" to the largest volume, setting "isLastBuy" accordingly. For buys, if "supportResistanceLevel" exceeds the current open price and a buy pin bar is detected with "IsBuyPinbar" and the bid price is below "lastOpenPrice" by "orderDistancePips \* \_Point", we open a buy position with "obj\_Trade.PositionOpen" using a lot size of "lastLotSize \* lotMultiplier" normalized by "fnGetLotDigit", with calculated stop loss and take profit, and clear "isLastBuy"; for sells, we check if the ask price is above "lastOpenPrice" by "orderDistancePips \* \_Point" and open a sell position similarly.

Last, we implement "addAveragingOrderWithAutoTP", which follows the same logic as "addAveragingOrder" but opens positions without initial stop loss or take profit (set to 0), calls "calculatePositionMetrics" to update metrics like "buyBreakEvenPrice", and invokes "updateStopLossTakeProfit" to set breakeven-based levels, ensuring dynamic adjustments for averaging trades. We can now call these functions in the tick logic for the logic to take effect.

```
if (useSignalMode == ENABLED && CountTradesBuy() >= 1 && CountTradesBuy() < maxOrders && useAutoTakeProfit == false) { //--- Check buy averaging
   addAveragingOrder();                        //--- Add buy averaging order
}
if (useSignalMode == ENABLED && CountTradesSell() >= 1 && CountTradesSell() < maxOrders && useAutoTakeProfit == false) { //--- Check sell averaging
   addAveragingOrder();                        //--- Add sell averaging order
}
if (useSignalMode == ENABLED && CountTradesBuy() >= 1 && CountTradesBuy() < maxOrders && useAutoTakeProfit == true) { //--- Check buy averaging with auto TP
   addAveragingOrderWithAutoTP();              //--- Add buy averaging order with auto TP
}
if (useSignalMode == ENABLED && CountTradesSell() >= 1 && CountTradesSell() < maxOrders && useAutoTakeProfit == true) { //--- Check sell averaging with auto TP
   addAveragingOrderWithAutoTP();              //--- Add sell averaging order with auto TP
}
```

We complete the tick logic implementation by adding logic to handle averaging trades under specific conditions, enhancing the EA’s ability to scale into positions dynamically. First, when "useSignalMode" is "ENABLED", we check if there is at least one buy position with "CountTradesBuy" and the number of buy positions is below "maxOrders"; if "useAutoTakeProfit" is false, we call "addAveragingOrder" to open an additional buy position based on pin bar detection and price distance criteria, using a multiplied lot size.

Then, we proceed to apply the same logic for sell positions, checking "CountTradesSell" and calling "addAveragingOrder" if "useAutoTakeProfit" is false to add a sell position under similar conditions. Next, for buy positions when "useAutoTakeProfit" is true, we call "addAveragingOrderWithAutoTP" to open a buy position without an initial stop loss or take profit, followed by updating metrics and adjusting breakeven-based levels. Last, we repeat this for sell positions when "useAutoTakeProfit" is true, invoking "addAveragingOrderWithAutoTP" to add a sell position with dynamic stop loss and take profit adjustments. This logic will ensure the EA effectively manages averaging trades in signal mode, adapting to market movements. Upon compilation, we have the following outcome.

![AVERAGING SAMPLE](https://c.mql5.com/2/162/Screenshot_2025-08-07_001815.png)

Now that we have added the averaging option, what remains is adding a trailing stop logic for risk management. The logic will need to be done on every tick for precise risk control, so we will add the logic outside the bar restriction logic.

```
double setPointValue = normalizedPoint;         //--- Set point value for calculations
if (useTrailingStop && trailingStartPips > 0 && breakevenPips < trailingStartPips) { //--- Check trailing stop conditions
   double averageBuyPrice = rata_price(ORDER_TYPE_BUY); //--- Calculate average buy price
   double trailingReference = 0;                //--- Initialize trailing reference
   for (int iTrade = 0; iTrade < PositionsTotal(); iTrade++) { //--- Iterate through positions
      ulong ticket = PositionGetTicket(iTrade); //--- Get position ticket
      if (ticket == 0) continue;                //--- Skip invalid tickets
      if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY && PositionGetString(POSITION_SYMBOL) == Symbol() && PositionGetInteger(POSITION_MAGIC) == magicNumber) { //--- Check buy position
         if (useAutoTakeProfit) {               //--- Check auto take profit
            trailingReference = averageBuyPrice; //--- Use average buy price
         } else {                               //--- Use open price
            trailingReference = PositionGetDouble(POSITION_PRICE_OPEN); //--- Set open price
         }
         if (SymbolInfoDouble(_Symbol, SYMBOL_BID) - trailingReference > trailingStartPips * setPointValue) { //--- Check trailing condition
            if (SymbolInfoDouble(_Symbol, SYMBOL_BID) - ((trailingStartPips - breakevenPips) * setPointValue) > PositionGetDouble(POSITION_SL)) { //--- Check stop loss adjustment
               obj_Trade.PositionModify(ticket, SymbolInfoDouble(_Symbol, SYMBOL_BID) - ((trailingStartPips - breakevenPips) * setPointValue), PositionGetDouble(POSITION_TP)); //--- Modify position
            }
         }
      }
   }
   double averageSellPrice = rata_price(ORDER_TYPE_SELL); //--- Calculate average sell price
   for (int iTrade2 = 0; iTrade2 < PositionsTotal(); iTrade2++) { //--- Iterate through positions
      ulong ticket2 = PositionGetTicket(iTrade2); //--- Get position ticket
      if (ticket2 == 0) continue;               //--- Skip invalid tickets
      if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL && PositionGetString(POSITION_SYMBOL) == Symbol() && PositionGetInteger(POSITION_MAGIC) == magicNumber) { //--- Check sell position
         if (useAutoTakeProfit) {               //--- Check auto take profit
            trailingReference = averageSellPrice; //--- Use average sell price
         } else {                               //--- Use open price
            trailingReference = PositionGetDouble(POSITION_PRICE_OPEN); //--- Set open price
         }
         if (trailingReference - SymbolInfoDouble(_Symbol, SYMBOL_ASK) > trailingStartPips * setPointValue) { //--- Check trailing condition
            if (SymbolInfoDouble(_Symbol, SYMBOL_ASK) + ((trailingStartPips - breakevenPips) * setPointValue) < PositionGetDouble(POSITION_SL) || PositionGetDouble(POSITION_SL) == 0) { //--- Check stop loss adjustment
               obj_Trade.PositionModify(ticket2, SymbolInfoDouble(_Symbol, SYMBOL_ASK) + ((trailingStartPips - breakevenPips) * setPointValue), PositionGetDouble(POSITION_TP)); //--- Modify position
            }
         }
      }
   }
}
```

We implement the trailing stop logic by first setting "setPointValue" to "normalizedPoint" for consistent price calculations and checking if "useTrailingStop" is true, "trailingStartPips" is positive, and "breakevenPips" is less than "trailingStartPips" to ensure valid trailing conditions. Then, we proceed to handle buy positions by calculating "averageBuyPrice" using "rata\_price" for [ORDER\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type), iterating through all positions to get valid buy position tickets that match "Symbol" and "magicNumber", setting "trailingReference" to "averageBuyPrice" if "useAutoTakeProfit" is true or to "POSITION\_PRICE\_OPEN" otherwise, and modifying the stop loss with "obj\_Trade.PositionModify" to [SYMBOL\_BID](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double) \- (trailingStartPips - breakevenPips) \* setPointValue" if the bid price exceeds "trailingReference" by "trailingStartPips \* setPointValue" and the new stop loss is higher than the current one.

Next, we apply similar logic for sell positions, calculating "averageSellPrice" with "rata\_price" for "ORDER\_TYPE\_SELL", iterating through positions, setting "trailingReference" to "averageSellPrice" or [POSITION\_PRICE\_OPEN](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_double), and modifying the stop loss to "SYMBOL\_ASK + (trailingStartPips - breakevenPips) \* setPointValue" if the ask price is below "trailingReference" by "trailingStartPips \* setPointValue" and the new stop loss is lower or unset. Last, we ensure modifications maintain the existing take profit via " [PositionGetDouble(POSITION\_TP)](https://www.mql5.com/en/docs/trading/positiongetdouble)" and call [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) in the parent function to update the chart. Upon compilation, we get the following outcome.

Before trailing stop:

![BEFORE TRAILING STOP](https://c.mql5.com/2/162/Screenshot_2025-08-07_002741.png)

After trailing stop:

![AFTER TRAILING STOP](https://c.mql5.com/2/162/Screenshot_2025-08-07_002859.png)

Now that we have completed the position management logic, we can create a dashboard to visualize the account metrics. We use a function for that as well for easier management.

```
//+------------------------------------------------------------------+
//| Display dashboard information                                    |
//+------------------------------------------------------------------+
void Display_Info() {
   buyCount = 0;                                  //--- Reset buy count
   currentBuyLot = 0;                             //--- Reset current buy lot
   totalBuyLots = 0;                              //--- Reset total buy lots
   sellCount = 0;                                 //--- Reset sell count
   currentSellLot = 0;                            //--- Reset current sell lot
   totalSellLots = 0;                             //--- Reset total sell lots
   totalSum = 0;                                  //--- Reset total sum
   totalSwap = 0;                                 //--- Reset total swap
   buyProfit = 0;                                 //--- Reset buy profit
   sellProfit = 0;                                //--- Reset sell profit
   buyWeightedSum = 0;                            //--- Reset buy weighted sum
   sellWeightedSum = 0;                           //--- Reset sell weighted sum
   buyBreakEvenPrice = 0;                         //--- Reset buy breakeven price
   sellBreakEvenPrice = 0;                        //--- Reset sell breakeven price
   minBuyLot = 9999;                              //--- Initialize min buy lot
   minSellLot = 9999;                             //--- Initialize min sell lot
   maxSellPrice = 0;                              //--- Initialize max sell price
   minBuyPrice = 999999999;                       //--- Initialize min buy price
   for (int i = 0; i < PositionsTotal(); i++) {   //--- Iterate through positions
      ulong ticket = PositionGetTicket(i);        //--- Get position ticket
      if (ticket == 0) continue;                  //--- Skip invalid tickets
      if (PositionGetString(POSITION_SYMBOL) != Symbol()) continue; //--- Skip non-matching symbols
      if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) { //--- Check buy position
         buyCount++;                              //--- Increment buy count
         totalOperations++;                       //--- Increment total operations
         currentBuyLot = PositionGetDouble(POSITION_VOLUME); //--- Set current buy lot
         buyProfit += PositionGetDouble(POSITION_PROFIT); //--- Add buy profit
         totalBuyLots += PositionGetDouble(POSITION_VOLUME); //--- Add to total buy lots
         minBuyLot = MathMin(minBuyLot, PositionGetDouble(POSITION_VOLUME)); //--- Update min buy lot
         buyWeightedSum += PositionGetDouble(POSITION_VOLUME) * PositionGetDouble(POSITION_PRICE_OPEN); //--- Add weighted open price
         minBuyPrice = MathMin(minBuyPrice, PositionGetDouble(POSITION_PRICE_OPEN)); //--- Update min buy price
      }
      if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) { //--- Check sell position
         sellCount++;                             //--- Increment sell count
         totalOperations++;                       //--- Increment total operations
         currentSellLot = PositionGetDouble(POSITION_VOLUME); //--- Set current sell lot
         sellProfit += PositionGetDouble(POSITION_PROFIT); //--- Add sell profit
         totalSellLots += PositionGetDouble(POSITION_VOLUME); //--- Add to total sell lots
         minSellLot = MathMin(minSellLot, PositionGetDouble(POSITION_VOLUME)); //--- Update min sell lot
         sellWeightedSum += PositionGetDouble(POSITION_VOLUME) * PositionGetDouble(POSITION_PRICE_OPEN); //--- Add weighted open price
         maxSellPrice = MathMax(maxSellPrice, PositionGetDouble(POSITION_PRICE_OPEN)); //--- Update max sell price
      }
   }
   if (totalBuyLots > 0) {                        //--- Check buy lots
      buyBreakEvenPrice = buyWeightedSum / totalBuyLots; //--- Calculate buy breakeven
   }
   if (totalSellLots > 0) {                       //--- Check sell lots
      sellBreakEvenPrice = sellWeightedSum / totalSellLots; //--- Calculate sell breakeven
   }
   int minutesRemaining, secondsRemaining;        //--- Declare time variables
   minutesRemaining = (int)(PeriodSeconds() - (TimeCurrent() - iTime(Symbol(), PERIOD_CURRENT, 0))); //--- Calculate remaining time
   secondsRemaining = minutesRemaining % 60;      //--- Calculate seconds
   minutesRemaining = minutesRemaining / 60;      //--- Calculate minutes
   long currentSpread = SymbolInfoInteger(Symbol(), SYMBOL_SPREAD); //--- Get current spread
   string spreadPrefix = "", minutesPrefix = "", secondsPrefix = ""; //--- Initialize prefixes
   if (currentSpread < 10) spreadPrefix = "..";   //--- Set spread prefix for single digit
   else if (currentSpread < 100) spreadPrefix = "."; //--- Set spread prefix for double digit
   if (minutesRemaining < 10) minutesPrefix = "0"; //--- Set minutes prefix
   if (secondsRemaining < 10) secondsPrefix = "0"; //--- Set seconds prefix
   int blinkingColorIndex;                        //--- Declare blinking color index
   color equityColor = clrGreen;                  //--- Initialize equity color
   if (AccountInfoDouble(ACCOUNT_EQUITY) - AccountInfoDouble(ACCOUNT_BALANCE) < 0.0) { //--- Check negative equity
      equityColor = clrRed;                       //--- Set equity color to red
   }
   color profitColor = (buyProfit + sellProfit >= 0) ? clrGreen : clrRed; //--- Set profit color
   MqlDateTime currentDateTime;                   //--- Declare datetime structure
   TimeToStruct(TimeCurrent(), currentDateTime);  //--- Convert current time
   if (currentDateTime.sec >= 0 && currentDateTime.sec < 10) { //--- Check first 10 seconds
      blinkingColorIndex = clrRed;                //--- Set red color
   }
   if (currentDateTime.sec >= 10 && currentDateTime.sec < 20) { //--- Check next 10 seconds
      blinkingColorIndex = clrOrange;             //--- Set orange color
   }
   if (currentDateTime.sec >= 20 && currentDateTime.sec < 30) { //--- Check next 10 seconds
      blinkingColorIndex = clrBlue;               //--- Set blue color
   }
   if (currentDateTime.sec >= 30 && currentDateTime.sec < 40) { //--- Check next 10 seconds
      blinkingColorIndex = clrDodgerBlue;         //--- Set dodger blue color
   }
   if (currentDateTime.sec >= 40 && currentDateTime.sec < 50) { //--- Check next 10 seconds
      blinkingColorIndex = clrYellow;             //--- Set yellow color
   }
   if (currentDateTime.sec >= 50 && currentDateTime.sec <= 59) { //--- Check last 10 seconds
      blinkingColorIndex = clrYellow;             //--- Set yellow color
   }
   if (ObjectFind(0, "DashboardBG") < 0) {        //--- Check dashboard background
      ObjectCreate(0, "DashboardBG", OBJ_RECTANGLE_LABEL, 0, 0, 0); //--- Create dashboard background
      ObjectSetInteger(0, "DashboardBG", OBJPROP_CORNER, 0); //--- Set corner
      ObjectSetInteger(0, "DashboardBG", OBJPROP_XDISTANCE, 100); //--- Set x distance
      ObjectSetInteger(0, "DashboardBG", OBJPROP_YDISTANCE, 20); //--- Set y distance
      ObjectSetInteger(0, "DashboardBG", OBJPROP_XSIZE, 260); //--- Set width
      ObjectSetInteger(0, "DashboardBG", OBJPROP_YSIZE, 300); //--- Set height
      ObjectSetInteger(0, "DashboardBG", OBJPROP_BGCOLOR, clrLightGray); //--- Set background color
      ObjectSetInteger(0, "DashboardBG", OBJPROP_BORDER_TYPE, BORDER_FLAT); //--- Set border type
      ObjectSetInteger(0, "DashboardBG", OBJPROP_COLOR, clrBlack); //--- Set border color
      ObjectSetInteger(0, "DashboardBG", OBJPROP_BACK, false); //--- Set to foreground
   }
   if (ObjectFind(0, "CLOSE ALL") < 0) {          //--- Check close all button
      ObjectCreate(0, "CLOSE ALL", OBJ_BUTTON, 0, 0, 0); //--- Create close all button
      ObjectSetInteger(0, "CLOSE ALL", OBJPROP_CORNER, 0); //--- Set corner
      ObjectSetInteger(0, "CLOSE ALL", OBJPROP_XDISTANCE, 110); //--- Set x distance
      ObjectSetInteger(0, "CLOSE ALL", OBJPROP_YDISTANCE, 280); //--- Set y distance
      ObjectSetInteger(0, "CLOSE ALL", OBJPROP_XSIZE, 240); //--- Set width
      ObjectSetInteger(0, "CLOSE ALL", OBJPROP_YSIZE, 25); //--- Set height
      ObjectSetString(0, "CLOSE ALL", OBJPROP_TEXT, "Close All Positions"); //--- Set button text
      ObjectSetInteger(0, "CLOSE ALL", OBJPROP_COLOR, clrWhite); //--- Set text color
      ObjectSetInteger(0, "CLOSE ALL", OBJPROP_BGCOLOR, clrRed); //--- Set background color
      ObjectSetInteger(0, "CLOSE ALL", OBJPROP_BORDER_COLOR, clrBlack); //--- Set border color
   }
   string headerText = "Pin Bar Averaging EA";    //--- Set header text
   LABEL("Header", "Impact", 20, 110, 20, clrNavy, 0, headerText); //--- Create header label
   string copyrightText = "Copyright 2025, Allan Munene Mutiiria"; //--- Set copyright text
   LABEL("Copyright", "Arial", 9, 110, 55, clrBlack, 0, copyrightText); //--- Create copyright label
   string linkText = "https://t.me/Forex_Algo_Trader"; //--- Set link text
   LABEL("Link", "Arial", 9, 110, 70, clrBlue, 0, linkText); //--- Create link label
   string accountHeader = "Account Information";  //--- Set account header
   LABEL("AccountHeader", "Arial Bold", 10, 110, 90, clrBlack, 0, accountHeader); //--- Create account header label
   string balanceText = "Balance: " + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2); //--- Set balance text
   LABEL("Balance", "Arial", 9, 120, 105, clrBlack, 0, balanceText); //--- Create balance label
   string equityText = "Equity: " + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2); //--- Set equity text
   LABEL("Equity", "Arial", 9, 120, 120, equityColor, 0, equityText); //--- Create equity label
   string marginText = "Free Margin: " + DoubleToString(AccountInfoDouble(ACCOUNT_MARGIN_FREE), 2); //--- Set margin text
   LABEL("Margin", "Arial", 9, 120, 135, clrBlack, 0, marginText); //--- Create margin label
   string profitText = "Open Profit: " + DoubleToString(buyProfit + sellProfit, 2); //--- Set profit text
   LABEL("Profit", "Arial", 9, 120, 150, profitColor, 0, profitText); //--- Create profit label
   string positionsText = "Buy Positions: " + IntegerToString((int)buyCount) + " Sell Positions: " + IntegerToString((int)sellCount); //--- Set positions text
   LABEL("Positions", "Arial", 9, 120, 165, clrBlack, 0, positionsText); //--- Create positions label
   string buyBEText = "Buy Break Even: " + (buyCount > 0 ? DoubleToString(buyBreakEvenPrice, _Digits) : "-"); //--- Set buy breakeven text
   LABEL("BuyBE", "Arial", 9, 120, 180, clrBlack, 0, buyBEText); //--- Create buy breakeven label
   string sellBEText = "Sell Break Even: " + (sellCount > 0 ? DoubleToString(sellBreakEvenPrice, _Digits) : "-"); //--- Set sell breakeven text
   LABEL("SellBE", "Arial", 9, 120, 195, clrBlack, 0, sellBEText); //--- Create sell breakeven label
   string spreadText = "Spread: " + spreadPrefix + IntegerToString((int)currentSpread) + " points"; //--- Set spread text
   LABEL("Spread", "Arial", 9, 120, 210, clrBlack, 0, spreadText); //--- Create spread label
   string timeText = "Time to next bar: " + minutesPrefix + IntegerToString(minutesRemaining) + ":" + secondsPrefix + IntegerToString(secondsRemaining); //--- Set time text
   LABEL("Time", "Arial", 9, 120, 225, clrBlack, 0, timeText); //--- Create time label
   string pinbarText;                             //--- Declare pinbar text
   if (IsBuyPinbar()) pinbarText = "Buy Pinbar";  //--- Check buy pinbar
   else if (IsSellPinbar()) pinbarText = "Sell Pinbar"; //--- Check sell pinbar
   else pinbarText = "None";                      //--- Set no pinbar
   LABEL("Pinbar", "Arial", 9, 120, 240, clrBlack, 0, "Pinbar Signal: " + pinbarText); //--- Create pinbar label
   string patternText = "Candle Pattern: " + CandleStick_Analyzer(); //--- Set candlestick pattern text
   LABEL("Pattern", "Arial", 9, 120, 255, clrBlack, 0, patternText); //--- Create pattern label
}
```

We implement the "Display\_Info" function to create a comprehensive dashboard for real-time trade monitoring. First, we reset key metrics like "buyCount" and others to their respective initialization values, then iterate through all positions to update these metrics for buy and sell positions matching "Symbol", incrementing counts, summing profits, volumes, weighted open prices, and tracking min/max prices, and calculating breakeven prices if applicable.

Then, we proceed to calculate the time remaining to the next bar using [PeriodSeconds](https://www.mql5.com/en/docs/common/periodseconds) minus the difference between current time and [iTime](https://www.mql5.com/en/docs/series/itime), converting to "minutesRemaining" and "secondsRemaining", and set formatting prefixes for spread and time display. Next, we determine "equityColor" (green or red based on equity vs. balance) and "profitColor" (green or red based on total profit), and set a blinking color index based on the current second for visual effect. Last, we create a dashboard background with [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) as [OBJ\_RECTANGLE\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) if "DashboardBG" doesn’t exist, a "CLOSE ALL" button as "OBJ\_BUTTON", and multiple labels using the "LABEL" function to display the EA’s title, copyright, link, account information (balance, equity, free margin, profit), position counts, breakeven prices, spread, time to next bar, pin bar signal, and candlestick patterns from "CandleStick\_Analyzer", ensuring clear and dynamic trade information visualization. For the button, we implement its logic in the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler.

```
//+------------------------------------------------------------------+
//| Handle chart events                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam) {
   if (id == CHARTEVENT_OBJECT_CLICK) {           //--- Check object click event
      if (sparam == "CLOSE ALL") {                //--- Check close all button
         ObjectSetInteger(0, "CLOSE ALL", OBJPROP_STATE, false); //--- Reset button state
         for (int positionIndex = PositionsTotal() - 1; positionIndex >= 0; positionIndex--) { //--- Iterate through positions
            ulong ticket = PositionGetTicket(positionIndex); //--- Get position ticket
            if (ticket == 0) continue;            //--- Skip invalid tickets
            if (PositionGetString(POSITION_SYMBOL) == Symbol()) { //--- Check symbol
               obj_Trade.PositionClose(ticket);   //--- Close position
            }
         }
      }
   }
}
```

In the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) function, we check if the event "id" is [CHARTEVENT\_OBJECT\_CLICK](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents) to detect object clicks on the chart. Then, we proceed to verify if the clicked object "sparam" is the "CLOSE ALL" button, and if so, we reset the button’s state to false using [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) with "OBJPROP\_STATE". Next, we iterate through all positions, retrieving each position’s ticket with [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket), skipping invalid tickets, and checking if the position’s symbol matches the current "Symbol" with the [PositionGetString](https://www.mql5.com/en/docs/trading/positiongetstring) function. Last, for matching positions, we close them using "obj\_Trade.PositionClose" to execute the user’s command to close all positions, ensuring the dashboard’s "Close All Positions" button provides a responsive way to manage open trades manually. Upon calling the function in [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) and compilation, we get the following outcome.

![COMPLETE PIN BAR AVERAGING SYSTEM](https://c.mql5.com/2/162/Screenshot_2025-08-07_004402.png)

From the image, we can see that it can detect and visualize the support or resistance level, open and average positions, trail the positions, and visualize the account metadata in a panel, hence achieving our objectives. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/162/Screenshot_2025-08-07_143745.png)

Backtest report:

![REPORT](https://c.mql5.com/2/162/Screenshot_2025-08-07_143828.png)

### Conclusion

In conclusion, we’ve developed a [Pin Bar](https://www.mql5.com/go?link=https://priceaction.com/price-action-university/strategies/pin-bar/ "https://priceaction.com/price-action-university/strategies/pin-bar/") Averaging system in MQL5, leveraging pin bar candlestick patterns to initiate trades and manage multiple positions through an averaging strategy, enhanced with trailing stops, breakeven adjustments, and a dynamic dashboard for real-time monitoring. Through modular components like the "CandleStick\_Analyzer" and "addAveragingOrder" [functions](https://www.mql5.com/en/docs/basis/function), this program offers a disciplined approach to reversal trading with customizable risk controls.

Disclaimer: This article is for educational purposes only. Trading carries significant financial risks, and market volatility may result in losses. Thorough backtesting and careful risk management are crucial before deploying this program in live markets.

By leveraging the concepts and implementation presented, you can adapt this pin bar system to your trading style, enhancing your algorithmic strategies. Happy trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19087.zip "Download all attachments in the single ZIP archive")

[a.\_Pin\_Bar\_Averaging\_EA.mq5](https://www.mql5.com/en/articles/download/19087/a._pin_bar_averaging_ea.mq5 "Download a._Pin_Bar_Averaging_EA.mq5")(79.58 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Trading Tools (Part 12): Enhancing the Correlation Matrix Dashboard with Interactivity](https://www.mql5.com/en/articles/20962)
- [Creating Custom Indicators in MQL5 (Part 5): WaveTrend Crossover Evolution Using Canvas for Fog Gradients, Signal Bubbles, and Risk Management](https://www.mql5.com/en/articles/20815)
- [MQL5 Trading Tools (Part 11): Correlation Matrix Dashboard (Pearson, Spearman, Kendall) with Heatmap and Standard Modes](https://www.mql5.com/en/articles/20945)
- [Creating Custom Indicators in MQL5 (Part 4): Smart WaveTrend Crossover with Dual Oscillators](https://www.mql5.com/en/articles/20811)
- [Building AI-Powered Trading Systems in MQL5 (Part 8): UI Polish with Animations, Timing Metrics, and Response Management Tools](https://www.mql5.com/en/articles/20722)
- [Creating Custom Indicators in MQL5 (Part 3): Multi-Gauge Enhancements with Sector and Round Styles](https://www.mql5.com/en/articles/20719)
- [Creating Custom Indicators in MQL5 (Part 2): Building a Gauge-Style RSI Display with Canvas and Needle Mechanics](https://www.mql5.com/en/articles/20632)

**[Go to discussion](https://www.mql5.com/en/forum/493606)**

![Self Optimizing Expert Advisors in MQL5 (Part 12): Building Linear Classifiers Using Matrix Factorization](https://c.mql5.com/2/163/18987-self-optimizing-expert-advisors-logo.png)[Self Optimizing Expert Advisors in MQL5 (Part 12): Building Linear Classifiers Using Matrix Factorization](https://www.mql5.com/en/articles/18987)

This article explores the powerful role of matrix factorization in algorithmic trading, specifically within MQL5 applications. From regression models to multi-target classifiers, we walk through practical examples that demonstrate how easily these techniques can be integrated using built-in MQL5 functions. Whether you're predicting price direction or modeling indicator behavior, this guide lays a strong foundation for building intelligent trading systems using matrix methods.

![From Basic to Intermediate: Template and Typename (I)](https://c.mql5.com/2/109/Do_basico_ao_intermedivrio__Template_e_Typename_I___LOGO.png)[From Basic to Intermediate: Template and Typename (I)](https://www.mql5.com/en/articles/15658)

In this article, we start considering one of the concepts that many beginners avoid. This is related to the fact that templates are not an easy topic, as many do not understand the basic principle underlying the template: overload of functions and procedures.

![From Basic to Intermediate: Template and Typename (II)](https://c.mql5.com/2/110/Do_btsico_ao_intermedi0rio__Template_e_Typename_I___LOGO.png)[From Basic to Intermediate: Template and Typename (II)](https://www.mql5.com/en/articles/15668)

This article explains how to deal with one of the most difficult programming situations you can encounter: using different types in the same function or procedure template. Although we have spent most of our time focusing only on functions, everything covered here is useful and can be applied to procedures.

![From Basic to Intermediate: Floating point](https://c.mql5.com/2/105/logo-Do_bdsico_ao_intermediprio-Ponto_Flutuante.png)[From Basic to Intermediate: Floating point](https://www.mql5.com/en/articles/15611)

This article is a brief introduction to the concept of floating-point numbers. Since this text is very complex please, read it attentively and carefully. Do not expect to quickly master the floating-point system. It only becomes clear over time, as you gain experience using it. But this article will help you understand why your application sometimes produces results different from what you expect.

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/19087&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049485237987159132)

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