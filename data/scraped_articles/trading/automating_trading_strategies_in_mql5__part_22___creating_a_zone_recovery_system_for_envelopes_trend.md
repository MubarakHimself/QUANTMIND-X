---
title: Automating Trading Strategies in MQL5 (Part 22): Creating a Zone Recovery System for Envelopes Trend Trading
url: https://www.mql5.com/en/articles/18720
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 7
scraped_at: 2026-01-22T17:47:14.674951
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/18720&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049378855942204015)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 21)](https://www.mql5.com/en/articles/18660), we explored a [neural network](https://en.wikipedia.org/wiki/Neural_network "https://en.wikipedia.org/wiki/Neural_network")-based trading strategy enhanced with adaptive learning rates to improve prediction accuracy for market movements in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5). In Part 22, we shift focus to creating a [Zone Recovery System](https://www.mql5.com/go?link=https://tradingliteracy.com/zone-recovery-in-trading/ "https://tradingliteracy.com/zone-recovery-in-trading/") integrated with an Envelopes trend-trading strategy, combining Relative Strength Index (RSI) and Envelopes indicators to automate trades and manage losses effectively. We will cover the following topics:

1. [Understanding the Zone Recovery Envelopes Trend Architecture](https://www.mql5.com/en/articles/18720#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/18720#para2)
3. [Backtesting](https://www.mql5.com/en/articles/18720#para3)
4. [Conclusion](https://www.mql5.com/en/articles/18720#para4)

By the end, you’ll have a robust MQL5 trading system designed for dynamic market conditions, ready for implementation and testing—let’s get started!

### Understanding the Zone Recovery Envelopes Trend Architecture

Zone recovery is a smart trading strategy that helps us turn potential losses into wins by placing extra trades when the market moves against us, aiming to come out ahead or break even. Imagine you buy a currency pair expecting it to rise, but it drops— [zone recovery](https://www.mql5.com/go?link=https://tradingliteracy.com/zone-recovery-in-trading/ "https://tradingliteracy.com/zone-recovery-in-trading/") steps in by setting a price range, or “zone,” where we place opposite trades to recover losses if the price bounces back. We plan to develop an automated system in MetaQuotes Language 5 (MQL5) that leverages this concept to trade forex markets while maintaining low risks and maximizing profits.

To make this work, we will utilize two technical indicators to identify the optimal times to enter trades. One indicator will check the market’s energy, ensuring we only trade when there’s a strong push in one direction, avoiding weak or messy signals. The other, called [Envelopes](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/envelopes "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/envelopes"), will draw a channel around the market’s average price, showing us when prices stretch too far up or down, signaling a likely snap-back moment to jump in. These indicators will work together to find high-chance trades where the price is ready to reverse within a trend.

Here’s how we intend to pull it all together: we will start by placing a trade when our indicators signal a reversal, like when the price hits the edge of the Envelopes channel with strong momentum. If the market moves the wrong way, we’ll activate the zone recovery by opening counter-trades within our set price zone, carefully sized to balance risk and recovery. We’ll limit the number of trades to avoid getting carried away, ensuring the system stays disciplined. This setup will let us chase trend opportunities while having a safety net for when things don’t go as planned, adaptable to both wild and calm markets. Stick with us as we turn this plan into reality and test it out! See the implementation plan below.

![STRATEGY PLAN](https://c.mql5.com/2/154/Screenshot_2025-07-02_141746.png)

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Indicators folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will start by declaring some [input variables](https://www.mql5.com/en/docs/basis/variables/inputvariables) that will help us control the key values of the program easily.

```
//+------------------------------------------------------------------+
//|                 Envelopes Trend Bounce with Zone Recovery EA.mq5 |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//|                                   https://t.me/Forex_Algo_Trader |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Allan Munene Mutiiria."
#property link      "https://t.me/Forex_Algo_Trader"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>                                             //--- Include trade library

enum TradingLotSizeOptions { FIXED_LOTSIZE, UNFIXED_LOTSIZE };         //--- Define lot size options

input group "======= EA GENERAL SETTINGS ======="
input TradingLotSizeOptions lotOption = UNFIXED_LOTSIZE;               // Lot Size Option
input double initialLotSize = 0.01;                                    // Initial Lot Size
input double riskPercentage = 1.0;                                     // Risk Percentage (%)
input int    riskPoints = 300;                                         // Risk Points
input int    magicNumber = 123456789;                                  // Magic Number
input int    maxOrders = 1;                                            // Maximum Initial Positions
input double zoneTargetPoints = 600;                                   // Zone Target Points
input double zoneSizePoints = 300;                                     // Zone Size Points
input bool   restrictMaxOrders = true;                                 // Apply Maximum Orders Restriction
```

Here, we lay the foundation for our Zone Recovery System for Envelopes Trend Trading in MQL5 by setting up essential components and user-configurable settings. We begin by including the "<Trade/Trade.mqh>" library, which provides the " [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade)" class for executing trading operations like opening and closing positions. This inclusion is vital as it equips our Expert Advisor (EA) with the tools needed to interact with the market seamlessly, especially order initiations. See below how to open the file.

![MQL5 TRADE OPERATIONS FILE](https://c.mql5.com/2/154/C_CTRADE.png)

We then define the "TradingLotSizeOptions" [enumeration](https://www.mql5.com/en/docs/basis/types/integer/enumeration) with two values: "FIXED\_LOTSIZE" and "UNFIXED\_LOTSIZE". This allows us to offer users a choice between a constant lot size or one that adjusts dynamically based on risk parameters, providing flexibility in trade sizing to suit different trading styles. Next, we configure the input parameters under the "EA GENERAL SETTINGS" group, which users can adjust in the MetaTrader 5 platform.

The "lotOption" input, set to "UNFIXED\_LOTSIZE" by default, determines whether trades use a fixed or risk-based lot size. The "initialLotSize" (0.01) sets the lot size for fixed trades, while "riskPercentage" (1.0%) and "riskPoints" (300) define the account balance percentage and stop-loss distance for dynamic lot sizing. These settings control how much risk we take per trade, ensuring the EA aligns with the user’s risk tolerance.

We assign a unique "magicNumber" (123456789) to identify our EA’s trades, allowing us to distinguish them from other trades on the same account. The "maxOrders" (1) and "restrictMaxOrders" (true) [inputs](https://www.mql5.com/en/docs/basis/variables/inputvariables) limit the number of initial positions, preventing the EA from opening too many trades at once. Finally, "zoneTargetPoints" (600) and "zoneSizePoints" (300) establish the profit target and recovery zone size in points, defining the boundaries for our zone recovery strategy. Upon compilation, we get the following output.

![LOADED INPUTS](https://c.mql5.com/2/154/Screenshot_2025-07-02_145112.png)

With the inputs loaded, we can now begin the core logic declaration for the entire system. We will start by declaring some structures and classes that we will use since we want to apply an [Object Oriented Programming](https://www.mql5.com/en/docs/basis/oop) (OOP) approach.

```
class MarketZoneTrader {
private:
   //--- Trade State Definition
   enum TradeState { INACTIVE, RUNNING, TERMINATING };                 //--- Define trade lifecycle states

   //--- Data Structures
   struct TradeMetrics {
      bool   operationSuccess;                                         //--- Track operation success
      double totalVolume;                                              //--- Sum closed trade volumes
      double netProfitLoss;                                            //--- Accumulate profit/loss
   };

   struct ZoneBoundaries {
      double zoneHigh;                                                 //--- Upper recovery zone boundary
      double zoneLow;                                                  //--- Lower recovery zone boundary
      double zoneTargetHigh;                                           //--- Upper profit target
      double zoneTargetLow;                                            //--- Lower profit target
   };

   struct TradeConfig {
      string         marketSymbol;                                     //--- Trading symbol
      double         openPrice;                                        //--- Position entry price
      double         initialVolume;                                    //--- Initial trade volume
      long           tradeIdentifier;                                  //--- Magic number
      string         tradeLabel;                                       //--- Trade comment
      ulong          activeTickets[];                                  //--- Active position tickets
      ENUM_ORDER_TYPE direction;                                       //--- Trade direction
      double         zoneProfitSpan;                                   //--- Profit target range
      double         zoneRecoverySpan;                                 //--- Recovery zone range
      double         accumulatedBuyVolume;                             //--- Total buy volume
      double         accumulatedSellVolume;                            //--- Total sell volume
      TradeState     currentState;                                     //--- Current trade state
   };

   struct LossTracker {
      double tradeLossTracker;                                         //--- Track cumulative profit/loss
   };
};
```

Here, we define the core structure of our system for Envelopes Trend Trading in MQL5 by implementing the "MarketZoneTrader" [class](https://www.mql5.com/en/docs/basis/types/classes), focusing on its private section with trade state definitions and data structures. This logic will help organize the critical components needed to manage trades, track recovery zones, and monitor performance. We begin by defining the "MarketZoneTrader" class, which serves as the backbone of our Expert Advisor (EA), [encapsulating](https://www.mql5.com/en/docs/basis/oop/incapsulation) the logic for our trading strategy.

Within its private section, we introduce the "TradeState" [enumeration](https://www.mql5.com/en/docs/basis/types/integer/enumeration) with three states: "INACTIVE", "RUNNING", and "TERMINATING". These states allow us to track the lifecycle of our trading operations, ensuring we know whether the EA is idle, actively managing trades, or closing positions. This is crucial for maintaining control over the trading process, as it helps us coordinate actions like opening recovery trades or finalizing positions.

Next, we create the "TradeMetrics" [structure](https://www.mql5.com/en/docs/basis/types/classes) to store key performance data for our trades. It includes "operationSuccess" to track whether trade actions (like closing positions) succeed, "totalVolume" to sum the volumes of closed trades, and "netProfitLoss" to accumulate the profit or loss from those trades. This structure helps us evaluate the outcome of our trading actions, providing a clear picture of performance during recovery or closure.

We then define the "ZoneBoundaries" structure, which holds the price levels for our zone recovery strategy. The "zoneHigh" and "zoneLow" variables mark the upper and lower boundaries of the recovery zone, where we place counter-trades to mitigate losses. The "zoneTargetHigh" and "zoneTargetLow" set the profit targets above and below the zone, defining when we exit trades profitably. These boundaries are essential for our strategy, as they guide when to trigger recovery actions or close positions. Here is what they would look like in visualization, just you have a clear picture of why we need the structure.

![ZONES SAMPLE](https://c.mql5.com/2/154/Screenshot_2025-07-02_151203.png)

Next, the "TradeConfig" [structure](https://www.mql5.com/en/docs/basis/types/classes) is where we store the trading setup. It includes "marketSymbol" for the currency pair, "openPrice" for the entry price, and "initialVolume" for the trade size. The "tradeIdentifier" holds our unique magic number, and "tradeLabel" adds a comment for trade identification. The "activeTickets" array tracks open position tickets, while "direction" specifies whether the trade is a buy or sell. We also include "zoneProfitSpan" and "zoneRecoverySpan" to define the profit target and recovery zone sizes in price units, and "accumulatedBuyVolume" and "accumulatedSellVolume" to monitor total volumes for each trade type. The "currentState" variable, using the "TradeState" enumeration, tracks the trading state, tying everything together.

Finally, we add the "LossTracker" structure with a single "tradeLossTracker" variable to monitor cumulative profit or loss across trades. This helps us assess the financial impact of our recovery actions, ensuring we can adjust our strategy if losses grow too large. We can then define some member variables to help store the other, less impactful but necessary trade information.

```
//--- Member Variables
TradeConfig           m_tradeConfig;                                //--- Store trade configuration
ZoneBoundaries        m_zoneBounds;                                 //--- Store zone boundaries
LossTracker           m_lossTracker;                                //--- Track profit/loss
string                m_lastError;                                  //--- Store error message
int                   m_errorStatus;                                //--- Store error code
CTrade                m_tradeExecutor;                              //--- Manage trade execution
int                   m_handleRsi;                                  //--- RSI indicator handle
int                   m_handleEnvUpper;                             //--- Upper Envelopes handle
int                   m_handleEnvLower;                             //--- Lower Envelopes handle
double                m_rsiBuffer[];                                //--- RSI data buffer
double                m_envUpperBandBuffer[];                       //--- Upper Envelopes buffer
double                m_envLowerBandBuffer[];                       //--- Lower Envelopes buffer
TradingLotSizeOptions m_lotOption;                                  //--- Lot size option
double                m_initialLotSize;                             //--- Fixed lot size
double                m_riskPercentage;                             //--- Risk percentage
int                   m_riskPoints;                                 //--- Risk points
int                   m_maxOrders;                                  //--- Maximum positions
bool                  m_restrictMaxOrders;                          //--- Position restriction flag
double                m_zoneTargetPoints;                           //--- Profit target points
double                m_zoneSizePoints;                             //--- Recovery zone points
```

We define key member variables in the "MarketZoneTrader" class’s [private](https://www.mql5.com/en/book/oop/structs_and_unions/structs_access) section to manage trade settings, recovery zones, and indicator data. We use "m\_tradeConfig" ("TradeConfig" structure) to store trade details like symbol and direction, "m\_zoneBounds" ("ZoneBoundaries" structure) for recovery zone and profit target prices, and "m\_lossTracker" ("LossTracker" structure) to track profits or losses. For error handling, "m\_lastError" (string) and "m\_errorStatus" (integer) log issues, while "m\_tradeExecutor" ("CTrade" class) handles trade operations.

Indicator handles—"m\_handleRsi", "m\_handleEnvUpper", "m\_handleEnvLower"—access RSI and Envelopes data, with "m\_rsiBuffer", "m\_envUpperBandBuffer", and "m\_envLowerBandBuffer" arrays storing their values. We store input settings in "m\_lotOption" ("TradingLotSizeOptions"), "m\_initialLotSize", "m\_riskPercentage", "m\_riskPoints", "m\_maxOrders", "m\_restrictMaxOrders", "m\_zoneTargetPoints", and "m\_zoneSizePoints" to control lot sizing, position limits, and zone sizes. These variables form the backbone for managing trades and indicators, preparing us for the trading logic ahead. We then need to define some helper functions that we will use frequently within the program.

```
//--- Error Handling
void logError(string message, int code) {
   //--- Error Logging Start
   m_lastError = message;                                           //--- Store error message
   m_errorStatus = code;                                            //--- Store error code
   Print("Error: ", message);                                       //--- Log error to Experts tab
   //--- Error Logging End
}

//--- Market Data Access
double getMarketVolumeStep() {
   //--- Volume Step Retrieval Start
   return SymbolInfoDouble(m_tradeConfig.marketSymbol, SYMBOL_VOLUME_STEP); //--- Retrieve broker's volume step
   //--- Volume Step Retrieval End
}

double getMarketAsk() {
   //--- Ask Price Retrieval Start
   return SymbolInfoDouble(m_tradeConfig.marketSymbol, SYMBOL_ASK); //--- Retrieve ask price
   //--- Ask Price Retrieval End
}

double getMarketBid() {
   //--- Bid Price Retrieval Start
   return SymbolInfoDouble(m_tradeConfig.marketSymbol, SYMBOL_BID); //--- Retrieve bid price
   //--- Bid Price Retrieval End
}
```

Here, we add critical utility functions for error handling and market data access. The "logError" function stores "message" in "m\_lastError", "code" in "m\_errorStatus", and logs the message via [Print](https://www.mql5.com/en/docs/common/print) to the Experts tab for debugging. The "getMarketVolumeStep" function uses [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) with [SYMBOL\_VOLUME\_STEP](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double) to fetch the broker’s volume increment for "m\_tradeConfig.marketSymbol", ensuring valid trade sizes. The "getMarketAsk" and "getMarketBid" functions retrieve ask and bid prices using "SymbolInfoDouble" with [SYMBOL\_ASK](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double) and "SYMBOL\_BID", respectively, for accurate trade pricing.

We can now define the major functions for executing trade operations. Let's start with ones that will help us initialize, store trade tickets for tracking and monitoring operations, and closing of the trades, as this is the less complex logic.

```
//--- Trade Initialization
bool configureTrade(ulong ticket) {
   //--- Trade Configuration Start
   if (!PositionSelectByTicket(ticket)) {                               //--- Select position by ticket
      logError("Failed to select ticket " + IntegerToString(ticket), INIT_FAILED); //--- Log selection failure
      return false;                                                     //--- Return failure
   }
   m_tradeConfig.marketSymbol = PositionGetString(POSITION_SYMBOL);     //--- Set symbol
   m_tradeConfig.tradeLabel = __FILE__;                                 //--- Set trade comment
   m_tradeConfig.tradeIdentifier = PositionGetInteger(POSITION_MAGIC);  //--- Set magic number
   m_tradeConfig.direction = (ENUM_ORDER_TYPE)PositionGetInteger(POSITION_TYPE);   //--- Set direction
   m_tradeConfig.openPrice = PositionGetDouble(POSITION_PRICE_OPEN);    //--- Set entry price
   m_tradeConfig.initialVolume = PositionGetDouble(POSITION_VOLUME);    //--- Set initial volume
   m_tradeExecutor.SetExpertMagicNumber(m_tradeConfig.tradeIdentifier); //--- Set magic number for executor
   return true;                                                         //--- Return success
   //--- Trade Configuration End
}

//--- Trade Ticket Management
void storeTradeTicket(ulong ticket) {
   //--- Ticket Storage Start
   int ticketCount = ArraySize(m_tradeConfig.activeTickets);        //--- Get ticket count
   ArrayResize(m_tradeConfig.activeTickets, ticketCount + 1);       //--- Resize ticket array
   m_tradeConfig.activeTickets[ticketCount] = ticket;               //--- Store ticket
   //--- Ticket Storage End
}

//--- Trade Execution
ulong openMarketTrade(ENUM_ORDER_TYPE tradeDirection, double tradeVolume, double price) {
   //--- Trade Opening Start
   ulong ticket = 0;                                                //--- Initialize ticket
   if (m_tradeExecutor.PositionOpen(m_tradeConfig.marketSymbol, tradeDirection, tradeVolume, price, 0, 0, m_tradeConfig.tradeLabel)) { //--- Open position
      ticket = m_tradeExecutor.ResultOrder();                       //--- Get ticket
   } else {
      Print("Failed to open trade: Direction=", EnumToString(tradeDirection), ", Volume=", tradeVolume); //--- Log failure
   }
   return ticket;                                                   //--- Return ticket
   //--- Trade Opening End
}

//--- Trade Closure
void closeActiveTrades(TradeMetrics &metrics) {
   //--- Trade Closure Start
   for (int i = ArraySize(m_tradeConfig.activeTickets) - 1; i >= 0; i--) {    //--- Iterate tickets in reverse
      if (m_tradeConfig.activeTickets[i] > 0) {                               //--- Check valid ticket
         if (m_tradeExecutor.PositionClose(m_tradeConfig.activeTickets[i])) { //--- Close position
            m_tradeConfig.activeTickets[i] = 0;                               //--- Clear ticket
            metrics.totalVolume += m_tradeExecutor.ResultVolume();            //--- Accumulate volume
            if ((ENUM_ORDER_TYPE)PositionGetInteger(POSITION_TYPE) == ORDER_TYPE_BUY) { //--- Check buy position
               metrics.netProfitLoss += m_tradeExecutor.ResultVolume() * (m_tradeExecutor.ResultPrice() - PositionGetDouble(POSITION_PRICE_OPEN)); //--- Calculate buy profit
            } else {                                                          //--- Handle sell position
               metrics.netProfitLoss += m_tradeExecutor.ResultVolume() * (PositionGetDouble(POSITION_PRICE_OPEN) - m_tradeExecutor.ResultPrice()); //--- Calculate sell profit
            }
         } else {
            metrics.operationSuccess = false;                                  //--- Mark failure
            Print("Failed to close ticket: ", m_tradeConfig.activeTickets[i]); //--- Log failure
         }
      }
   }
   //--- Trade Closure End
}

//--- Bar Detection
bool isNewBar() {
   //--- New Bar Detection Start
   static datetime previousTime = 0;                                      //--- Store previous bar time
   datetime currentTime = iTime(m_tradeConfig.marketSymbol, Period(), 0); //--- Get current bar time
   bool result = (currentTime != previousTime);                           //--- Check for new bar
   previousTime = currentTime;                                            //--- Update previous time
   return result;                                                         //--- Return new bar status
   //--- New Bar Detection End
}
```

Here, we dive into the core logic of our program, crafting functions to set up trades, track positions, execute orders, close trades, and time our actions. We start by creating the "configureTrade" function to prepare a trade for a given "ticket". First, we try selecting the position with the [PositionSelectByTicket](https://www.mql5.com/en/docs/trading/positionselectbyticket) function. If it doesn’t work, we log the issue using "logError" and exit with false. When it succeeds, we fill "m\_tradeConfig" with details: we grab "marketSymbol" using the [PositionGetString](https://www.mql5.com/en/docs/trading/positiongetstring) function, set "tradeLabel" to [\_\_FILE\_\_](https://www.mql5.com/en/docs/constant_indices), and pull "tradeIdentifier" and "direction" from [PositionGetInteger](https://www.mql5.com/en/docs/trading/positiongetinteger), casting the latter to [ENUM\_ORDER\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type). Then, we set "openPrice" and "initialVolume" with [PositionGetDouble](https://www.mql5.com/en/docs/trading/positiongetdouble) and tag "m\_tradeExecutor" with "SetExpertMagicNumber", ensuring our trade is ready to roll.

Next, we create the "storeTradeTicket" function to keep our open positions organized. We check the size of "m\_tradeConfig.activeTickets" with the [ArraySize](https://www.mql5.com/en/docs/array/arraysize) function, stretch the array by one slot using the [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) function, and slip the new "ticket" into place, so we always know which trades are active. Moving on, we create the "openMarketTrade" function to place trades in the market. We call "m\_tradeExecutor.PositionOpen" with "tradeDirection", "tradeVolume", "price", and "m\_tradeConfig" details. If it goes through, we assign the "ticket" with "ResultOrder"; if not, we log the error with "Print", keeping our trade execution tight.

Then, we tackle closing positions with the "closeActiveTrades" function. We loop backward through "m\_tradeConfig.activeTickets", closing each valid ticket with "m\_tradeExecutor.PositionClose". When a closure works, we clear the ticket, add "ResultVolume" to "metrics.totalVolume", and calculate "metrics.netProfitLoss" using the "PositionGetInteger" and "PositionGetDouble" functions to check trade direction. If something fails, we flag "metrics.operationSuccess" as false and log it with [Print](https://www.mql5.com/en/docs/common/print), ensuring we track every outcome.

Finally, we add the "isNewBar" function to help trade once per bar, reducing resource usage. We fetch the current bar time for "m\_tradeConfig.marketSymbol" with the [iTime](https://www.mql5.com/en/docs/series/itime) function, compare it to "previousTime", and update "previousTime" if it’s different, letting us know when a new bar arrives to check for trade signals. Finally, we will need a function to calculate the trading volume and a function to open the trades.

```
//--- Lot Size Calculation
double calculateLotSize(double riskPercent, int riskPips) {
   //--- Lot Size Calculation Start
   double riskMoney = AccountInfoDouble(ACCOUNT_BALANCE) * riskPercent / 100;                //--- Calculate risk amount
   double tickSize = SymbolInfoDouble(m_tradeConfig.marketSymbol, SYMBOL_TRADE_TICK_SIZE);   //--- Get tick size
   double tickValue = SymbolInfoDouble(m_tradeConfig.marketSymbol, SYMBOL_TRADE_TICK_VALUE); //--- Get tick value
   if (tickSize == 0 || tickValue == 0) {                           //--- Validate tick data
      Print("Invalid tick size or value");                          //--- Log invalid data
      return -1;                                                    //--- Return invalid lot
   }
   double lotValue = (riskPips * _Point) / tickSize * tickValue;    //--- Calculate lot value
   if (lotValue == 0) {                                             //--- Validate lot value
      Print("Invalid lot value");                                   //--- Log invalid lot
      return -1;                                                    //--- Return invalid lot
   }
   return NormalizeDouble(riskMoney / lotValue, 2);                 //--- Return normalized lot size
   //--- Lot Size Calculation End
}

//--- Order Execution
int openOrder(ENUM_ORDER_TYPE orderType, double stopLoss, double takeProfit) {
   //--- Order Opening Start
   int ticket;                                                      //--- Initialize ticket
   double openPrice;                                                //--- Initialize open price

   if (orderType == ORDER_TYPE_BUY) {                               //--- Check buy order
      openPrice = NormalizeDouble(getMarketAsk(), Digits());        //--- Set buy price
   } else if (orderType == ORDER_TYPE_SELL) {                       //--- Check sell order
      openPrice = NormalizeDouble(getMarketBid(), Digits());        //--- Set sell price
   } else {
      Print("Invalid order type");                                  //--- Log invalid type
      return -1;                                                    //--- Return invalid ticket
   }

   double lotSize = 0;                                              //--- Initialize lot size

   if (m_lotOption == FIXED_LOTSIZE) {                              //--- Check fixed lot
      lotSize = m_initialLotSize;                                   //--- Use fixed lot size
   } else if (m_lotOption == UNFIXED_LOTSIZE) {                     //--- Check dynamic lot
      lotSize = calculateLotSize(m_riskPercentage, m_riskPoints);   //--- Calculate risk-based lot
   }

   if (lotSize <= 0) {                                              //--- Validate lot size
      Print("Invalid lot size: ", lotSize);                         //--- Log invalid lot
      return -1;                                                    //--- Return invalid ticket
   }

   if (m_tradeExecutor.PositionOpen(m_tradeConfig.marketSymbol, orderType, lotSize, openPrice, 0, 0, __FILE__)) { //--- Open position
      ticket = (int)m_tradeExecutor.ResultOrder();                  //--- Get ticket
      Print("New trade opened: Ticket=", ticket, ", Type=", EnumToString(orderType), ", Volume=", lotSize); //--- Log success
   } else {
      ticket = -1;                                                  //--- Set invalid ticket
      Print("Failed to open order: Type=", EnumToString(orderType), ", Volume=", lotSize); //--- Log failure
   }

   return ticket;                                                   //--- Return ticket
   //--- Order Opening End
}
```

We start with the "calculateLotSize" function to determine the trade size based on risk parameters. First, we calculate the "riskMoney" by taking a percentage of the account balance using [AccountInfoDouble](https://www.mql5.com/en/docs/account/accountinfodouble) with [ACCOUNT\_BALANCE](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_info_double) and "riskPercent". Then, we fetch "tickSize" and "tickValue" for "m\_tradeConfig.marketSymbol" using [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) with "SYMBOL\_TRADE\_TICK\_SIZE" and "SYMBOL\_TRADE\_TICK\_VALUE". If either is zero, we log an error with "Print" and return -1 to avoid invalid calculations. We compute the "lotValue" using "riskPips", [\_Point](https://www.mql5.com/en/docs/predefined/_point), "tickSize", and "tickValue", and if it’s zero, we log another error and return -1. Finally, we return the lot size with [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble) to two decimal places, ensuring it matches broker requirements.

Next, we create the "openOrder" function to place trades. We initialize "ticket" and "openPrice", then check "orderType". For [ORDER\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type), we set "openPrice" using "getMarketAsk" and "NormalizeDouble" with [Digits](https://www.mql5.com/en/docs/check/digits); for "ORDER\_TYPE\_SELL", we use "getMarketBid". If "orderType" is invalid, we log it with "Print" and return -1. We determine "lotSize" based on "m\_lotOption": for "FIXED\_LOTSIZE", we use "m\_initialLotSize"; for "UNFIXED\_LOTSIZE", we call "calculateLotSize" with "m\_riskPercentage" and "m\_riskPoints". If "lotSize" is invalid, we log the error with "Print" and return -1. We then open the position with "m\_tradeExecutor.PositionOpen" using "m\_tradeConfig.marketSymbol", "orderType", "lotSize", "openPrice", and "FILE" as the comment. On success, we set "ticket" with "ResultOrder" and log it with "Print"; on failure, we set "ticket" to -1 and log the error. Finally, we return the ticket value.

After doing that, we need to initialize the system values. We can achieve that via a dedicated function, but to keep everything simple, we will use the [constructor](https://www.mql5.com/en/book/oop/structs_and_unions/structs_ctor_dtor). It is advisable to define the constructor in a public access modifier so it is available everywhere in the program. Let us define the [destructor](https://www.mql5.com/en/book/oop/structs_and_unions/structs_ctor_dtor) here, too.

```
public:
   //--- Constructor
   MarketZoneTrader(TradingLotSizeOptions lotOpt, double initLot, double riskPct, int riskPts, int maxOrds, bool restrictOrds, double targetPts, double sizePts) {
      //--- Constructor Start
      m_tradeConfig.currentState = INACTIVE;                           //--- Set initial state
      ArrayResize(m_tradeConfig.activeTickets, 0);                     //--- Initialize ticket array
      m_tradeConfig.zoneProfitSpan = targetPts * _Point;               //--- Set profit target
      m_tradeConfig.zoneRecoverySpan = sizePts * _Point;               //--- Set recovery zone
      m_lossTracker.tradeLossTracker = 0.0;                            //--- Initialize loss tracker
      m_lotOption = lotOpt;                                            //--- Set lot size option
      m_initialLotSize = initLot;                                      //--- Set initial lot
      m_riskPercentage = riskPct;                                      //--- Set risk percentage
      m_riskPoints = riskPts;                                          //--- Set risk points
      m_maxOrders = maxOrds;                                           //--- Set max positions
      m_restrictMaxOrders = restrictOrds;                              //--- Set restriction flag
      m_zoneTargetPoints = targetPts;                                  //--- Set target points
      m_zoneSizePoints = sizePts;                                      //--- Set zone points
      m_tradeConfig.marketSymbol = _Symbol;                            //--- Set symbol
      m_tradeConfig.tradeIdentifier = magicNumber;                     //--- Set magic number
      //--- Constructor End
   }

   //--- Destructor
   ~MarketZoneTrader() {
      //--- Destructor Start
      cleanup();                                                       //--- Release resources
      //--- Destructor End
   }
```

We continue by defining the [constructor and destructor](https://www.mql5.com/en/book/oop/structs_and_unions/structs_ctor_dtor) for the "MarketZoneTrader" class in its public section. We begin with the "MarketZoneTrader" constructor, which takes parameters "lotOpt", "initLot", "riskPct", "riskPts", "maxOrds", "restrictOrds", "targetPts", and "sizePts". We initialize the trading environment by setting "m\_tradeConfig.currentState" to "INACTIVE" to indicate no active trades. Next, we clear the "m\_tradeConfig.activeTickets" array using [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) to zero, preparing it for new tickets. We calculate "m\_tradeConfig.zoneProfitSpan" and "m\_tradeConfig.zoneRecoverySpan" by multiplying "targetPts" and "sizePts" with "\_Point", setting the profit target and recovery zone sizes in price units. We reset "m\_lossTracker.tradeLossTracker" to 0.0 to start tracking profits or losses from scratch.

Then, we assign the input parameters to member variables: "m\_lotOption" to "lotOpt", "m\_initialLotSize" to "initLot", "m\_riskPercentage" to "riskPct", "m\_riskPoints" to "riskPts", "m\_maxOrders" to "maxOrds", "m\_restrictMaxOrders" to "restrictOrds", "m\_zoneTargetPoints" to "targetPts", and "m\_zoneSizePoints" to "sizePts". We set "m\_tradeConfig.marketSymbol" to [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol) to trade the current chart’s symbol and assign "m\_tradeConfig.tradeIdentifier" to "magicNumber" for unique trade identification. This setup ensures our EA reflects user settings and is ready to trade.

Next, we define the "~MarketZoneTrader" destructor to clean up resources. We call the "cleanup" function to release any allocated resources, such as indicator handles, ensuring the EA shuts down cleanly without memory leaks. It is good to note that the constructor and destructor have the same class name wording, only that the destructor has a [tilde](https://en.wikipedia.org/wiki/Tilde "https://en.wikipedia.org/wiki/Tilde") (~) before it. Just that. Here is the function for destroying the class when not needed.

```
//--- Cleanup
void cleanup() {
   //--- Cleanup Start
   IndicatorRelease(m_handleRsi);                                   //--- Release RSI handle
   ArrayFree(m_rsiBuffer);                                          //--- Free RSI buffer
   IndicatorRelease(m_handleEnvUpper);                              //--- Release upper Envelopes handle
   ArrayFree(m_envUpperBandBuffer);                                 //--- Free upper Envelopes buffer
   IndicatorRelease(m_handleEnvLower);                              //--- Release lower Envelopes handle
   ArrayFree(m_envLowerBandBuffer);                                 //--- Free lower Envelopes buffer
   //--- Cleanup End
}
```

We simply use the [IndicatorRelease](https://www.mql5.com/en/docs/series/indicatorrelease) function to release the indicator handles and the [ArrayFree](https://www.mql5.com/en/docs/array/ArrayFree) function to release the storage arrays. Since we have touched the indicators, let us define an initialization function that we will call when starting the program.

```
//--- Getters
TradeState getCurrentState() {
   //--- Get Current State Start
   return m_tradeConfig.currentState;                               //--- Return trade state
   //--- Get Current State End
}

double getZoneTargetHigh() {
   //--- Get Target High Start
   return m_zoneBounds.zoneTargetHigh;                              //--- Return profit target high
   //--- Get Target High End
}

double getZoneTargetLow() {
   //--- Get Target Low Start
   return m_zoneBounds.zoneTargetLow;                               //--- Return profit target low
   //--- Get Target Low End
}

double getZoneHigh() {
   //--- Get Zone High Start
   return m_zoneBounds.zoneHigh;                                    //--- Return recovery zone high
   //--- Get Zone High End
}

double getZoneLow() {
   //--- Get Zone Low Start
   return m_zoneBounds.zoneLow;                                     //--- Return recovery zone low
   //--- Get Zone Low End
}

//--- Initialization
int initialize() {
   //--- Initialization Start
   m_tradeExecutor.SetExpertMagicNumber(m_tradeConfig.tradeIdentifier); //--- Set magic number
   int totalPositions = PositionsTotal();                               //--- Get total positions

   for (int i = 0; i < totalPositions; i++) {                           //--- Iterate positions
      ulong ticket = PositionGetTicket(i);                              //--- Get ticket
      if (PositionSelectByTicket(ticket)) {                             //--- Select position
         if (PositionGetString(POSITION_SYMBOL) == m_tradeConfig.marketSymbol && PositionGetInteger(POSITION_MAGIC) == m_tradeConfig.tradeIdentifier) { //--- Check symbol and magic
            if (activateTrade(ticket)) {                                //--- Activate position
               Print("Existing position activated: Ticket=", ticket);   //--- Log activation
            } else {
               Print("Failed to activate existing position: Ticket=", ticket); //--- Log failure
            }
         }
      }
   }

   m_handleRsi = iRSI(m_tradeConfig.marketSymbol, PERIOD_CURRENT, 8, PRICE_CLOSE); //--- Initialize RSI
   if (m_handleRsi == INVALID_HANDLE) {                             //--- Check RSI
      Print("Failed to initialize RSI indicator");                  //--- Log failure
      return INIT_FAILED;                                           //--- Return failure
   }

   m_handleEnvUpper = iEnvelopes(m_tradeConfig.marketSymbol, PERIOD_CURRENT, 150, 0, MODE_SMA, PRICE_CLOSE, 0.1); //--- Initialize upper Envelopes
   if (m_handleEnvUpper == INVALID_HANDLE) {                        //--- Check upper Envelopes
      Print("Failed to initialize upper Envelopes indicator");      //--- Log failure
      return INIT_FAILED;                                           //--- Return failure
   }

   m_handleEnvLower = iEnvelopes(m_tradeConfig.marketSymbol, PERIOD_CURRENT, 95, 0, MODE_SMA, PRICE_CLOSE, 1.4); //--- Initialize lower Envelopes
   if (m_handleEnvLower == INVALID_HANDLE) {                        //--- Check lower Envelopes
      Print("Failed to initialize lower Envelopes indicator");      //--- Log failure
      return INIT_FAILED;                                           //--- Return failure
   }

   ArraySetAsSeries(m_rsiBuffer, true);                             //--- Set RSI buffer
   ArraySetAsSeries(m_envUpperBandBuffer, true);                    //--- Set upper Envelopes buffer
   ArraySetAsSeries(m_envLowerBandBuffer, true);                    //--- Set lower Envelopes buffer

   Print("EA initialized successfully");                            //--- Log success
   return INIT_SUCCEEDED;                                           //--- Return success
   //--- Initialization End
}
```

Here, we start by creating simple getter functions to access key trading data. The "getCurrentState" function returns "m\_tradeConfig.currentState", letting us check if the system is in the "INACTIVE", "RUNNING", or "TERMINATING" state. Next, we build "getZoneTargetHigh" and "getZoneTargetLow" to retrieve "m\_zoneBounds.zoneTargetHigh" and "m\_zoneBounds.zoneTargetLow", providing the profit target prices for our trades. Then, we add "getZoneHigh" and "getZoneLow" to fetch "m\_zoneBounds.zoneHigh" and "m\_zoneBounds.zoneLow", giving us the recovery zone boundaries.

Moving on, we craft the "initialize" function to set up our Expert Advisor (EA). We begin by assigning "m\_tradeConfig.tradeIdentifier" to "m\_tradeExecutor" using "SetExpertMagicNumber" to tag our trades. We then check for existing positions with "PositionsTotal" and loop through them, grabbing each "ticket" with "PositionGetTicket". If [PositionSelectByTicket](https://www.mql5.com/en/docs/trading/positionselectbyticket) succeeds and the position matches "m\_tradeConfig.marketSymbol" and "m\_tradeConfig.tradeIdentifier" (via [PositionGetString](https://www.mql5.com/en/docs/trading/positiongetstring) and "PositionGetInteger"), we call "activateTrade" to manage it, logging success or failure with "Print".

Next, we set up our indicators. We create the RSI handle with the [iRSI](https://www.mql5.com/en/docs/indicators/irsi) function for "m\_tradeConfig.marketSymbol" using an 8-period setting on the current timeframe and "PRICE\_CLOSE". If "m\_handleRsi" is [INVALID\_HANDLE](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants), we log the error with "Print" and return "INIT\_FAILED". We then initialize the Envelopes indicators: "m\_handleEnvUpper" with the "iEnvelopes" function using a 150-period, simple moving average, 0.1 deviation, and "PRICE\_CLOSE", and "m\_handleEnvLower" with a 95-period, 1.4 deviation. If either handle is "INVALID\_HANDLE", we log the failure and return "INIT\_FAILED". Finally, we configure "m\_rsiBuffer", "m\_envUpperBandBuffer", and "m\_envLowerBandBuffer" as time-series arrays with [ArraySetAsSeries](https://www.mql5.com/en/docs/array/arraysetasseries), log success with "Print", and return [INIT\_SUCCEEDED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode). We can now call this function on the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, but first, we will need an instance of the class.

```
//--- Global Instance
MarketZoneTrader *trader = NULL;                                        //--- Declare trader instance
```

Here, we set up the global instance of our system by declaring a pointer to the "MarketZoneTrader" class. We create the "trader" variable as a [pointer](https://www.mql5.com/en/docs/basis/types/object_pointers) to "MarketZoneTrader" and initialize it to "NULL". This step ensures we have a single, globally accessible instance of our trading system that we can use throughout the Expert Advisor (EA) to manage all trading operations, such as initializing trades, executing orders, and handling recovery zones. By starting with "NULL", we prepare the "trader" to be properly instantiated later, preventing any premature access before the EA is fully set up. We can now proceed to call the function.

```
int OnInit() {
   //--- EA Initialization Start
   trader = new MarketZoneTrader(lotOption, initialLotSize, riskPercentage, riskPoints, maxOrders, restrictMaxOrders, zoneTargetPoints, zoneSizePoints); //--- Create trader instance
   return trader.initialize();                                           //--- Initialize EA
   //--- EA Initialization End
}
```

In the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, we start by creating a new instance of the "MarketZoneTrader" class, assigning it to the global "trader" pointer. We pass the user-defined input parameters—"lotOption", "initialLotSize", "riskPercentage", "riskPoints", "maxOrders", "restrictMaxOrders", "zoneTargetPoints", and "zoneSizePoints"—to the [constructor](https://www.mql5.com/en/book/oop/structs_and_unions/structs_ctor_dtor) to configure the trading system with the desired settings. Then, we call the "initialize" function on "trader" to set up the EA, including trade tagging, existing position checks, and indicator initialization, and return its result to signal whether the setup was successful. This function ensures our EA is fully prepared to start trading with the specified configurations. Upon compilation, we have the following output.

![INITIALIZATION IMAGE](https://c.mql5.com/2/154/Screenshot_2025-07-02_165834.png)

From the image, we can see that the program initialized successfully. However, there is an issue when we try to remove the program. See below.

![OBJECTS MEMORY LEAK](https://c.mql5.com/2/154/Screenshot_2025-07-02_183952.png)

From the image, we can see there are undeleted objects that lead to a memory leak. To solve this, we need to do the object cleanup. To achieve that, we use the following logic.

```
void OnDeinit(const int reason) {
   //--- EA Deinitialization Start
   if (trader != NULL) {                                                 //--- Check trader existence
      delete trader;                                                     //--- Delete trader
      trader = NULL;                                                     //--- Clear pointer
      Print("EA deinitialized");                                         //--- Log deinitialization
   }
   //--- EA Deinitialization End
}
```

To handle the cleanup, in the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) event handler, we begin by checking if the "trader" pointer is not "NULL" to ensure the "MarketZoneTrader" instance exists. If it does, we use the [delete](https://www.mql5.com/en/docs/basis/operators/deleteoperator) operator to free the memory allocated for "trader", preventing memory leaks. Then, we set "trader" to "NULL" to avoid accidental access to the deallocated memory. Finally, we log a message with the "Print" function to confirm the EA has been deinitialized. This function ensures our EA exits cleanly, releasing resources properly. We can now continue defining the main logic to handle signal evaluations and the management of opened trades. We will need utility functions for that.

```
//--- Position Management
bool activateTrade(ulong ticket) {
   //--- Position Activation Start
   m_tradeConfig.currentState = INACTIVE;                           //--- Set state to inactive
   ArrayResize(m_tradeConfig.activeTickets, 0);                     //--- Clear tickets
   m_lossTracker.tradeLossTracker = 0.0;                            //--- Reset loss tracker
   if (!configureTrade(ticket)) {                                    //--- Configure trade
      return false;                                                 //--- Return failure
   }
   storeTradeTicket(ticket);                                        //--- Store ticket
   if (m_tradeConfig.direction == ORDER_TYPE_BUY) {                 //--- Handle buy position
      m_zoneBounds.zoneHigh = m_tradeConfig.openPrice;              //--- Set zone high
      m_zoneBounds.zoneLow = m_zoneBounds.zoneHigh - m_tradeConfig.zoneRecoverySpan; //--- Set zone low
      m_tradeConfig.accumulatedBuyVolume = m_tradeConfig.initialVolume; //--- Set buy volume
      m_tradeConfig.accumulatedSellVolume = 0.0;                    //--- Reset sell volume
   } else {                                                         //--- Handle sell position
      m_zoneBounds.zoneLow = m_tradeConfig.openPrice;               //--- Set zone low
      m_zoneBounds.zoneHigh = m_zoneBounds.zoneLow + m_tradeConfig.zoneRecoverySpan; //--- Set zone high
      m_tradeConfig.accumulatedSellVolume = m_tradeConfig.initialVolume; //--- Set sell volume
      m_tradeConfig.accumulatedBuyVolume = 0.0;                     //--- Reset buy volume
   }
   m_zoneBounds.zoneTargetHigh = m_zoneBounds.zoneHigh + m_tradeConfig.zoneProfitSpan; //--- Set target high
   m_zoneBounds.zoneTargetLow = m_zoneBounds.zoneLow - m_tradeConfig.zoneProfitSpan; //--- Set target low
   m_tradeConfig.currentState = RUNNING;                            //--- Set state to running
   return true;                                                     //--- Return success
   //--- Position Activation End
}

//--- Tick Processing
void processTick() {
   //--- Tick Processing Start
   double askPrice = NormalizeDouble(getMarketAsk(), Digits());     //--- Get ask price
   double bidPrice = NormalizeDouble(getMarketBid(), Digits());     //--- Get bid price

   if (!isNewBar()) return;                                         //--- Exit if not new bar

   if (!CopyBuffer(m_handleRsi, 0, 0, 3, m_rsiBuffer)) {            //--- Load RSI data
      Print("Error loading RSI data. Reverting.");                  //--- Log RSI failure
      return;                                                       //--- Exit
   }

   if (!CopyBuffer(m_handleEnvUpper, 0, 0, 3, m_envUpperBandBuffer)) { //--- Load upper Envelopes
      Print("Error loading upper envelopes data. Reverting.");         //--- Log failure
      return;                                                          //--- Exit
   }

   if (!CopyBuffer(m_handleEnvLower, 1, 0, 3, m_envLowerBandBuffer)) { //--- Load lower Envelopes
      Print("Error loading lower envelopes data. Reverting.");         //--- Log failure
      return;                                                          //--- Exit
   }

   int ticket = 0;                                                     //--- Initialize ticket

   const int rsiOverbought = 70;                                       //--- Set RSI overbought level
   const int rsiOversold = 30;                                         //--- Set RSI oversold level

   if (m_rsiBuffer[1] < rsiOversold && m_rsiBuffer[2] > rsiOversold && m_rsiBuffer[0] < rsiOversold) { //--- Check buy signal
      if (askPrice > m_envUpperBandBuffer[0]) {                        //--- Confirm price above upper Envelopes
         if (!m_restrictMaxOrders || PositionsTotal() < m_maxOrders) { //--- Check position limit
            ticket = openOrder(ORDER_TYPE_BUY, 0, 0);                  //--- Open buy order
         }
      }
   } else if (m_rsiBuffer[1] > rsiOverbought && m_rsiBuffer[2] < rsiOverbought && m_rsiBuffer[0] > rsiOverbought) { //--- Check sell signal
      if (bidPrice < m_envLowerBandBuffer[0]) {                        //--- Confirm price below lower Envelopes
         if (!m_restrictMaxOrders || PositionsTotal() < m_maxOrders) { //--- Check position limit
            ticket = openOrder(ORDER_TYPE_SELL, 0, 0);                 //--- Open sell order
         }
      }
   }

   if (ticket > 0) {                                                //--- Check if trade opened
      if (activateTrade(ticket)) {                                  //--- Activate position
         Print("New position activated: Ticket=", ticket);          //--- Log activation
      } else {
         Print("Failed to activate new position: Ticket=", ticket); //--- Log failure
      }
   }
   //--- Tick Processing End
}
```

Here, we continue developing our program by implementing the "activateTrade" and "processTick" functions within the "MarketZoneTrader" class to manage positions and handle market ticks. We start with the "activateTrade" function to activate a trade for a given "ticket". First, we set "m\_tradeConfig.currentState" to "INACTIVE" and clear "m\_tradeConfig.activeTickets" using the [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) function to reset the ticket list. We reset "m\_lossTracker.tradeLossTracker" to 0.0, then call "configureTrade" with "ticket". If it fails, we return false. Next, we save the "ticket" with "storeTradeTicket". For a buy trade ("m\_tradeConfig.direction" as [ORDER\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type)), we set "m\_zoneBounds.zoneHigh" to "m\_tradeConfig.openPrice", calculate "m\_zoneBounds.zoneLow" by subtracting "m\_tradeConfig.zoneRecoverySpan", and update "m\_tradeConfig.accumulatedBuyVolume" to "m\_tradeConfig.initialVolume" while resetting "m\_tradeConfig.accumulatedSellVolume".

For a sell trade, we set "m\_zoneBounds.zoneLow" to "m\_tradeConfig.openPrice", add "m\_tradeConfig.zoneRecoverySpan" for "m\_zoneBounds.zoneHigh", and adjust volumes accordingly. We then set "m\_zoneBounds.zoneTargetHigh" and "m\_zoneBounds.zoneTargetLow" using "m\_tradeConfig.zoneProfitSpan", change "m\_tradeConfig.currentState" to "RUNNING", and return true.

Next, we create the "processTick" function to handle market ticks. We fetch "askPrice" and "bidPrice" using "getMarketAsk" and "getMarketBid", normalized with [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble) and "Digits". If "isNewBar" returns false, we exit to save resources. We load indicator data with [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer) for "m\_handleRsi" into "m\_rsiBuffer", "m\_handleEnvUpper" into "m\_envUpperBandBuffer", and "m\_handleEnvLower" into "m\_envLowerBandBuffer", logging errors with "Print" and exiting if any fail. For trade signals, we set "rsiOverbought" to 70 and "rsiOversold" to 30.

If "m\_rsiBuffer" indicates an oversold condition and "askPrice" exceeds "m\_envUpperBandBuffer", we open a buy order with "openOrder" if "m\_restrictMaxOrders" is false or [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) is below "m\_maxOrders". For an overbought condition with "bidPrice" below "m\_envLowerBandBuffer", we open a sell order. If a valid "ticket" is returned, we call "activateTrade" and log the outcome to the journal. We can now run the function in the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler to process the signal evaluation and position initiation.

```
void OnTick() {
   //--- Tick Handling Start
   if (trader != NULL) {                                                 //--- Check trader existence
      trader.processTick();                                              //--- Process tick
   }
   //--- Tick Handling End
}
```

In the "OnTick" event handler, we start by checking if the "trader" pointer, our instance of the "MarketZoneTrader" class, is not "NULL" to ensure the trading system is initialized. If it exists, we call the "processTick" function on "trader" to handle each market tick, evaluating positions, checking indicator signals, and executing trades as needed. Upon compilation, we have the following outcome.

![INITIAL POSITION](https://c.mql5.com/2/154/Screenshot_2025-07-02_190550.png)

From the image, we can see that we have identified a signal, evaluated it, and initiated a buy position. What we now need to do is manage the open positions. We will handle that in functions for modularity.

```
//--- Market Tick Evaluation
void evaluateMarketTick() {
   //--- Tick Evaluation Start
   if (m_tradeConfig.currentState == INACTIVE) return;              //--- Exit if inactive
   if (m_tradeConfig.currentState == TERMINATING) {                 //--- Check terminating state
      finalizePosition();                                           //--- Finalize position
      return;                                                       //--- Exit
   }
}
```

Here, we implement the "evaluateMarketTick" function within the "MarketZoneTrader" class to assess market conditions for active trades. We begin by checking the "m\_tradeConfig.currentState" to see if it’s "INACTIVE". If it is, we exit immediately to avoid unnecessary processing when no trades are active. Next, we check if "m\_tradeConfig.currentState" is "TERMINATING". If so, we call the "finalizePosition" function to close all open positions and complete the trade cycle, then exit. Here is the function to close the trades.

```
//--- Position Finalization
bool finalizePosition() {
   //--- Position Finalization Start
   m_tradeConfig.currentState = TERMINATING;                        //--- Set terminating state
   TradeMetrics metrics = {true, 0.0, 0.0};                         //--- Initialize metrics
   closeActiveTrades(metrics);                                       //--- Close all trades
   if (metrics.operationSuccess) {                                  //--- Check success
      ArrayResize(m_tradeConfig.activeTickets, 0);                  //--- Clear tickets
      m_tradeConfig.currentState = INACTIVE;                        //--- Set inactive state
      Print("Position closed successfully");                        //--- Log success
   } else {
      Print("Failed to close position");                            //--- Log failure
   }
   return metrics.operationSuccess;                                 //--- Return status
   //--- Position Finalization End
}
```

We start by setting "m\_tradeConfig.currentState" to "TERMINATING" to indicate the trade cycle is ending. This helps to prevent the management cycle when we are in the process of closing the trades. Then, we initialize a "TradeMetrics" [structure](https://www.mql5.com/en/docs/basis/types/classes) named "metrics" with "operationSuccess" set to true, "totalVolume" to 0.0, and "netProfitLoss" to 0.0 to track closure outcomes. We call "closeActiveTrades" with "metrics" to close all open positions listed in "m\_tradeConfig.activeTickets". If "metrics.operationSuccess" remains true, we clear "m\_tradeConfig.activeTickets" using [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) to reset the ticket list, set "m\_tradeConfig.currentState" to "INACTIVE" to mark the system as idle, and log success with "Print".

If closure fails, we log the failure with "Print". Finally, we return "metrics.operationSuccess" to indicate whether the process completed successfully. If we did not close the trades at this point, it means that we are not in the position closure process, so we can proceed with the evaluation to see if the price hit the recovery zones or target levels. We will start with the buy instance.

```
double currentPrice;                                             //--- Initialize price
if (m_tradeConfig.direction == ORDER_TYPE_BUY) {                 //--- Handle buy position
   currentPrice = getMarketBid();                                //--- Get bid price
   if (currentPrice > m_zoneBounds.zoneTargetHigh) {             //--- Check profit target
      Print("Closing position: Bid=", currentPrice, " > TargetHigh=", m_zoneBounds.zoneTargetHigh); //--- Log closure
      finalizePosition();                                        //--- Close position
      return;                                                    //--- Exit
   } else if (currentPrice < m_zoneBounds.zoneLow) {             //--- Check recovery trigger
      Print("Triggering recovery trade: Bid=", currentPrice, " < ZoneLow=", m_zoneBounds.zoneLow); //--- Log recovery
      triggerRecoveryTrade(ORDER_TYPE_SELL, currentPrice);       //--- Open sell recovery
   }
}
```

We continue by implementing logic within the "evaluateMarketTick" function of the "MarketZoneTrader" class to handle buy positions. We start by declaring "currentPrice" to store the market price. If "m\_tradeConfig.direction" is [ORDER\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type), we set "currentPrice" using the "getMarketBid" function to fetch the bid price, as this is the price at which we can close a buy position. Next, we check if "currentPrice" exceeds "m\_zoneBounds.zoneTargetHigh". If it does, we log the closure with "Print", showing the bid price and target, then call "finalizePosition" to close the trade and exit with "return".

If "currentPrice" falls below "m\_zoneBounds.zoneLow", we log a recovery trigger with "Print" and call "triggerRecoveryTrade" with [ORDER\_TYPE\_SELL](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type) and "currentPrice" to open a sell trade to mitigate losses. This logic ensures we close profitable buy trades or initiate recovery for losing ones, keeping our strategy responsive. Here is the logic for the function responsible for opening recovery trades.

```
//--- Recovery Trade Handling
void triggerRecoveryTrade(ENUM_ORDER_TYPE tradeDirection, double price) {
   //--- Recovery Trade Start
   TradeMetrics metrics = {true, 0.0, 0.0};                         //--- Initialize metrics
   closeActiveTrades(metrics);                                      //--- Close existing trades
   for (int i = 0; i < 10 && !metrics.operationSuccess; i++) {      //--- Retry closure
      Sleep(1000);                                                  //--- Wait 1 second
      metrics.operationSuccess = true;                              //--- Reset success flag
      closeActiveTrades(metrics);                                   //--- Retry closure
   }
   m_lossTracker.tradeLossTracker += metrics.netProfitLoss;         //--- Update loss tracker
   if (m_lossTracker.tradeLossTracker > 0 && metrics.operationSuccess) { //--- Check positive profit
      Print("Closing position due to positive profit: ", m_lossTracker.tradeLossTracker); //--- Log closure
      finalizePosition();                                           //--- Close position
      m_lossTracker.tradeLossTracker = 0.0;                         //--- Reset loss tracker
      return;                                                       //--- Exit
   }
   double tradeSize = determineRecoverySize(tradeDirection);        //--- Calculate trade size
   ulong ticket = openMarketTrade(tradeDirection, tradeSize, price); //--- Open recovery trade
   if (ticket > 0) {                                                //--- Check if trade opened
      storeTradeTicket(ticket);                                     //--- Store ticket
      m_tradeConfig.direction = tradeDirection;                     //--- Update direction
      if (tradeDirection == ORDER_TYPE_BUY) m_tradeConfig.accumulatedBuyVolume += tradeSize; //--- Update buy volume
      else m_tradeConfig.accumulatedSellVolume += tradeSize;        //--- Update sell volume
      Print("Recovery trade opened: Ticket=", ticket, ", Direction=", EnumToString(tradeDirection), ", Volume=", tradeSize); //--- Log recovery trade
   }
   //--- Recovery Trade End
}

//--- Recovery Size Calculation
double determineRecoverySize(ENUM_ORDER_TYPE tradeDirection) {
   //--- Recovery Size Calculation Start
   double tradeSize = -m_lossTracker.tradeLossTracker / m_tradeConfig.zoneProfitSpan; //--- Calculate lot size
   tradeSize = MathCeil(tradeSize / getMarketVolumeStep()) * getMarketVolumeStep(); //--- Round to volume step
   return tradeSize;                                                //--- Return trade size
   //--- Recovery Size Calculation End
}
```

To handle cases where the market needs to trigger recovery instances, we start with the "triggerRecoveryTrade" function to handle recovery trades when a position moves against us. First, we initialize a "TradeMetrics" [structure](https://www.mql5.com/en/docs/basis/types/classes) named "metrics" with "operationSuccess" set to true, "totalVolume" to 0.0, and "netProfitLoss" to 0.0. We call "closeActiveTrades" with "metrics" to close existing positions. If "metrics.operationSuccess" is false, we retry up to 10 times, waiting one second with [Sleep](https://www.mql5.com/en/docs/common/sleep) and resetting "operationSuccess" before each attempt.

We update "m\_lossTracker.tradeLossTracker" by adding "metrics.netProfitLoss". If "m\_lossTracker.tradeLossTracker" is positive and "metrics.operationSuccess" is true, we log the closure with "Print", call "finalizePosition", reset "m\_lossTracker.tradeLossTracker" to 0.0, and exit with "return". Otherwise, we calculate the recovery "tradeSize" using "determineRecoverySize" with "tradeDirection", then open a new trade with "openMarketTrade" using "tradeDirection", "tradeSize", and "price".

If the returned "ticket" is valid, we save it with "storeTradeTicket", update "m\_tradeConfig.direction", adjust "m\_tradeConfig.accumulatedBuyVolume" or "m\_tradeConfig.accumulatedSellVolume" based on "tradeDirection", and log the trade with "Print" using [EnumToString](https://www.mql5.com/en/docs/convert/enumtostring). Next, we create the "determineRecoverySize" function to calculate the lot size for recovery trades. We compute "tradeSize" by dividing the negative "m\_lossTracker.tradeLossTracker" by "m\_tradeConfig.zoneProfitSpan" to size the trade to cover losses. We then round "tradeSize" to the broker’s volume step using [MathCeil](https://www.mql5.com/en/docs/math/mathceil) and "getMarketVolumeStep" to ensure compliance, and return the result. This now handles the recovery instances, and we can continue with the logic for handling the sell zones. The logic is just the opposite of buy, so we will not invest much time in that. The final full function will be as follows.

```
//--- Market Tick Evaluation
void evaluateMarketTick() {
   //--- Tick Evaluation Start
   if (m_tradeConfig.currentState == INACTIVE) return;              //--- Exit if inactive
   if (m_tradeConfig.currentState == TERMINATING) {                 //--- Check terminating state
      finalizePosition();                                           //--- Finalize position
      return;                                                       //--- Exit
   }
   double currentPrice;                                             //--- Initialize price
   if (m_tradeConfig.direction == ORDER_TYPE_BUY) {                 //--- Handle buy position
      currentPrice = getMarketBid();                                //--- Get bid price
      if (currentPrice > m_zoneBounds.zoneTargetHigh) {             //--- Check profit target
         Print("Closing position: Bid=", currentPrice, " > TargetHigh=", m_zoneBounds.zoneTargetHigh); //--- Log closure
         finalizePosition();                                        //--- Close position
         return;                                                    //--- Exit
      } else if (currentPrice < m_zoneBounds.zoneLow) {             //--- Check recovery trigger
         Print("Triggering recovery trade: Bid=", currentPrice, " < ZoneLow=", m_zoneBounds.zoneLow); //--- Log recovery
         triggerRecoveryTrade(ORDER_TYPE_SELL, currentPrice);       //--- Open sell recovery
      }
   } else if (m_tradeConfig.direction == ORDER_TYPE_SELL) {         //--- Handle sell position
      currentPrice = getMarketAsk();                                //--- Get ask price
      if (currentPrice < m_zoneBounds.zoneTargetLow) {              //--- Check profit target
         Print("Closing position: Ask=", currentPrice, " < TargetLow=", m_zoneBounds.zoneTargetLow); //--- Log closure
         finalizePosition();                                        //--- Close position
         return;                                                    //--- Exit
      } else if (currentPrice > m_zoneBounds.zoneHigh) {            //--- Check recovery trigger
         Print("Triggering recovery trade: Ask=", currentPrice, " > ZoneHigh=", m_zoneBounds.zoneHigh); //--- Log recovery
         triggerRecoveryTrade(ORDER_TYPE_BUY, currentPrice);        //--- Open buy recovery
      }
   }
   //--- Tick Evaluation End
}
```

The function now handles all the directions for recovery. Upon compilation, we have the following outcome.

![FINAL OUTCOME](https://c.mql5.com/2/154/Screenshot_2025-07-02_194850.png)

From the image, we can see that we successfully handle positions that are triggered due to trend bounce signals. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/154/Screenshot_2025-07-02_201138.png)

Backtest report:

![REPORT](https://c.mql5.com/2/154/Screenshot_2025-07-02_201105.png)

### Conclusion

In conclusion, we have built a robust MQL5 program that implements a [Zone Recovery System](https://www.mql5.com/go?link=https://tradingliteracy.com/zone-recovery-in-trading/ "https://tradingliteracy.com/zone-recovery-in-trading/") for Envelopes Trend Trading, combining Relative Strength Index (RSI) and Envelopes indicators to identify trade opportunities and manage losses through structured recovery zones, using an [Object Oriented Programming](https://www.mql5.com/en/docs/basis/oop) (OOP) approach. Using components like the "MarketZoneTrader" class, structures such as "TradeConfig" and "ZoneBoundaries", and functions like "processTick" and "triggerRecoveryTrade", we created a flexible system that you can adjust by tweaking parameters like "zoneTargetPoints" or "riskPercentage" to fit various market conditions.

Disclaimer: This article is for educational purposes only. Trading involves significant financial risks, and market volatility can lead to losses. Thorough backtesting and careful risk management are essential before using this program in live markets.

With the foundation laid in this article, you can refine this zone recovery system or adapt its logic to develop new trading strategies, fueling your progress in algorithmic trading. Happy trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18720.zip "Download all attachments in the single ZIP archive")

[Envelopes\_Trend\_Bounce\_with\_Zone\_Recovery\_EA.mq5](https://www.mql5.com/en/articles/download/18720/envelopes_trend_bounce_with_zone_recovery_ea.mq5 "Download Envelopes_Trend_Bounce_with_Zone_Recovery_EA.mq5")(34.09 KB)

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

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/490563)**
(2)


![Sabrina Hellal](https://c.mql5.com/avatar/2025/7/6867ab4a-8eff.png)

**[Sabrina Hellal](https://www.mql5.com/en/users/sabrinamql4)**
\|
9 Jul 2025 at 13:44

Thank so much 🙏


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
9 Jul 2025 at 16:46

**Sabrina Hellal [#](https://www.mql5.com/en/forum/490563#comment_57454719):**

Thank so much 🙏

Very much welcomed. Thanks

![From Novice to Expert: Animated News Headline Using MQL5 (IV) — Locally hosted AI model market insights](https://c.mql5.com/2/154/18685-from-novice-to-expert-animated-logo__1.png)[From Novice to Expert: Animated News Headline Using MQL5 (IV) — Locally hosted AI model market insights](https://www.mql5.com/en/articles/18685)

In today's discussion, we explore how to self-host open-source AI models and use them to generate market insights. This forms part of our ongoing effort to expand the News Headline EA, introducing an AI Insights Lane that transforms it into a multi-integration assistive tool. The upgraded EA aims to keep traders informed through calendar events, financial breaking news, technical indicators, and now AI-generated market perspectives—offering timely, diverse, and intelligent support to trading decisions. Join the conversation as we explore practical integration strategies and how MQL5 can collaborate with external resources to build a powerful and intelligent trading work terminal.

![MQL5 Wizard Techniques you should know (Part 73): Using Patterns of Ichimoku and the ADX-Wilder](https://c.mql5.com/2/154/18723-mql5-wizard-techniques-you-logo__1.png)[MQL5 Wizard Techniques you should know (Part 73): Using Patterns of Ichimoku and the ADX-Wilder](https://www.mql5.com/en/articles/18723)

The Ichimoku-Kinko-Hyo Indicator and the ADX-Wilder oscillator are a pairing that could be used in complimentarily within an MQL5 Expert Advisor. The Ichimoku is multi-faceted, however for this article, we are relying on it primarily for its ability to define support and resistance levels. Meanwhile, we also use the ADX to define our trend. As usual, we use the MQL5 wizard to build and test any potential these two may possess.

![Neural Networks in Trading: Hyperbolic Latent Diffusion Model (HypDiff)](https://c.mql5.com/2/100/Neural_Networks_in_Trading__Hyperbolic_Latent_Diffusion_Model___LOGO2.png)[Neural Networks in Trading: Hyperbolic Latent Diffusion Model (HypDiff)](https://www.mql5.com/en/articles/16306)

The article considers methods of encoding initial data in hyperbolic latent space through anisotropic diffusion processes. This helps to more accurately preserve the topological characteristics of the current market situation and improves the quality of its analysis.

![Statistical Arbitrage Through Cointegrated Stocks (Part 1): Engle-Granger and Johansen Cointegration Tests](https://c.mql5.com/2/154/18702-statistical-arbitrage-through-logo.png)[Statistical Arbitrage Through Cointegrated Stocks (Part 1): Engle-Granger and Johansen Cointegration Tests](https://www.mql5.com/en/articles/18702)

This article aims to provide a trader-friendly, gentle introduction to the most common cointegration tests, along with a simple guide to understanding their results. The Engle-Granger and Johansen cointegration tests can reveal statistically significant pairs or groups of assets that share long-term dynamics. The Johansen test is especially useful for portfolios with three or more assets, as it calculates the strength of cointegrating vectors all at once.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=cqyuqizfnoimdfyopsxnuotzxccotqsz&ssn=1769093232324563622&ssn_dr=0&ssn_sr=0&fv_date=1769093232&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18720&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2022)%3A%20Creating%20a%20Zone%20Recovery%20System%20for%20Envelopes%20Trend%20Trading%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909323253137362&fz_uniq=5049378855942204015&sv=2552)

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