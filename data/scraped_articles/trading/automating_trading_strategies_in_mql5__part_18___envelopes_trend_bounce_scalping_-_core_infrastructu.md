---
title: Automating Trading Strategies in MQL5 (Part 18): Envelopes Trend Bounce Scalping - Core Infrastructure and Signal Generation (Part I)
url: https://www.mql5.com/en/articles/18269
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T17:56:27.037379
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/18269&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049499557408124056)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 17)](https://www.mql5.com/en/articles/18038), we automated the Grid-Mart Scalping Strategy with a dynamic dashboard for real-time trade monitoring. In Part 18, we begin automating the Envelopes Trend Bounce Scalping Strategy in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5), developing the Expert Advisor’s core infrastructure and signal generation logic. We will cover the following topics:

1. [Understanding the Strategy](https://www.mql5.com/en/articles/18269#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/18269#para2)
3. [Backtesting](https://www.mql5.com/en/articles/18269#para3)
4. [Conclusion](https://www.mql5.com/en/articles/18269#para4)

By the end, you’ll have a solid foundation for scalping trend bounces, ready for trade execution in the next part—let’s dive in!

### Understanding the Strategy

The Envelopes Trend Bounce Scalping Strategy uses the [Envelopes indicator](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/envelopes "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/envelopes"), which creates upper and lower bands around a moving average with a set deviation (e.g., 0.1% to 1.4%), to identify price reversals for scalping small profits. It generates buy signals when the price touches the lower band in an uptrend and sell signals when it hits the upper band in a downtrend, confirmed by trend filters like a 200-period [Exponential Moving Average](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma") (EMA) or 8-period [Relative Strength Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi") (RSI). This strategy works well in trending markets but requires strict risk management to avoid false signals in ranging conditions, and we will take care of that.

Our implementation plan involves building a program to automate this strategy by initializing Envelopes and trend indicators, detecting bounce signals, and setting up robust signal validation. We will use modular functions to calculate band interactions and filter trades, ensuring precision in high-frequency scalping. Risk controls, such as maximum trade frequency and signal confirmation, will maintain reliability across market conditions. In a nutshell, here is a visualization of what we aim to achieve.

![BLUEPRINT](https://c.mql5.com/2/145/Screenshot_2025-05-25_213759.png)

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Experts folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some [global variables](https://www.mql5.com/en/docs/basis/variables/global) and [inputs](https://www.mql5.com/en/docs/basis/variables/inputvariables) that we will use throughout the program.

```
//+------------------------------------------------------------------+
//|                      Envelopes Trend Bounce Scalping Strategy EA |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//|                                   https://t.me/Forex_Algo_Trader |
//+------------------------------------------------------------------+

#property copyright "Copyright 2025, Allan Munene Mutiiria."
#property link      "https://t.me/Forex_Algo_Trader"
#property version   "1.00"
#property strict                                           //--- Enable strict compilation for MQL5 compatibility

//--- Include trade operations library
#include <Trade\Trade.mqh> //--- Import MQL5 trade functions for order execution

//--- Input parameters for user configuration
input string OrderComment = __FILE__;                     // Comment to orders
input int MagicNumber = 123456789;                        // Unique identifier for EA orders
double PipPointOverride = 0;                              //--- Override pip point value manually (0 for auto-detection)
input int MaxDeviationSlippage = 10;                      //--- Set maximum slippage in points for trades
bool AllowManualTPSLChanges = true;                       //--- Permit manual adjustment of TP and SL lines on chart
bool OneQuotePerBar = false;                              //--- Process only first tick per bar if true to limit trades
bool AlertOnError = false;                                //--- Trigger MetaTrader alerts for errors
bool NotificationOnError = false;                         //--- Send push notifications for errors
bool EmailOnError = true;                                 //--- Send email notifications for errors
bool DisplayOnChartError = true;                          //--- Show error messages on chart
bool DisplayOrderInfo = false;                            //--- Show order details on chart if enabled
ENUM_TIMEFRAMES DisplayOrderDuringTimeframe = PERIOD_M1;   //--- Set timeframe for order info display (default: M1)
input string CComment = __FILE__;                         //--- Add secondary comment (default: file name)

//--- Global variables for EA functionality
double PipPoint = 0.0001;                                 //--- Initialize pip point (default for 4-digit symbols)
uint OrderFillingType = -1;                               //--- Store order filling type (FOK, IOC, or Return)
uint AccountMarginMode = -1;                              //--- Store account margin mode (Netting or Hedging)
bool StopEA = false;                                      //--- Pause EA operations if true
double UnitsOneLot = 100000;                              //--- Define standard lot size (100,000 units for forex)
int IsDemoLiveOrVisualMode = false;                       //--- Flag demo, live, or visual backtest mode
string Error;                                             //--- Hold current error message
string ErrorPreviousQuote;                                //--- Hold previous quote's error message
string OrderInfoComment;                                  //--- Store order information comments
```

We start by setting up the core infrastructure for our program, focusing on libraries, user inputs, and global variables for signal generation. We include the "Trade.mqh" library using the [#include](https://www.mql5.com/en/docs/basis/preprosessor/include) directive for trade operations. We define inputs like "OrderComment", "MagicNumber" (123456789), "PipPointOverride", and "MaxDeviationSlippage" (10), plus booleans "AllowManualTPSLChanges" (true), "EmailOnError" (true), and "DisplayOrderDuringTimeframe" ("PERIOD\_M1").

We initialize globals like "PipPoint" (0.0001), "OrderFillingType", "AccountMarginMode", "StopEA" (false), and "UnitsOneLot" (100,000), along with "Error" and "OrderInfoComment" for error and order tracking, laying the groundwork for indicator setup. We can now define some constants and enumerations that we will use too.

```
//--- Define constants for order types
#define OP_BUY 0                                          //--- Represent Buy order type
#define OP_SELL 1                                         //--- Represent Sell order type

//--- Define constants for market data retrieval
#define MODE_TIME 5                                       //--- Retrieve symbol time
#define MODE_BID 9                                        //--- Retrieve Bid price
#define MODE_ASK 10                                       //--- Retrieve Ask price
#define MODE_POINT 11                                     //--- Retrieve point size
#define MODE_DIGITS 12                                    //--- Retrieve digit count
#define MODE_SPREAD 13                                    //--- Retrieve spread
#define MODE_STOPLEVEL 14                                 //--- Retrieve stop level
#define MODE_LOTSIZE 15                                   //--- Retrieve lot size
#define MODE_TICKVALUE 16                                 //--- Retrieve tick value
#define MODE_TICKSIZE 17                                  //--- Retrieve tick size
#define MODE_SWAPLONG 18                                  //--- Retrieve swap long
#define MODE_SWAPSHORT 19                                 //--- Retrieve swap short
#define MODE_STARTING 20                                  //--- Unused, return 0
#define MODE_EXPIRATION 21                                //--- Unused, return 0
#define MODE_TRADEALLOWED 22                              //--- Unused, return 0
#define MODE_MINLOT 23                                    //--- Retrieve minimum lot
#define MODE_LOTSTEP 24                                   //--- Retrieve lot step
#define MODE_MAXLOT 25                                    //--- Retrieve maximum lot
#define MODE_SWAPTYPE 26                                  //--- Retrieve swap mode
#define MODE_PROFITCALCMODE 27                            //--- Retrieve profit calculation mode
#define MODE_MARGINCALCMODE 28                            //--- Unused, return 0
#define MODE_MARGININIT 29                                //--- Unused, return 0
#define MODE_MARGINMAINTENANCE 30                         //--- Unused, return 0
#define MODE_MARGINHEDGED 31                              //--- Unused, return 0
#define MODE_MARGINREQUIRED 32                            //--- Unused, return 0
#define MODE_FREEZELEVEL 33                               //--- Retrieve freeze level

//--- Define string conversion macros
#define CharToStr CharToString                            //--- Convert char to string
#define DoubleToStr DoubleToString                        //--- Convert double to string
#define StrToDouble StringToDouble                        //--- Convert string to double
#define StrToInteger (int)StringToInteger                 //--- Convert string to integer
#define StrToTime StringToTime                            //--- Convert string to datetime
#define TimeToStr TimeToString                            //--- Convert datetime to string
#define StringGetChar StringGetCharacter                  //--- Get character from string
#define StringSetChar StringSetCharacter                  //--- Set character in string

//--- Define enumerations for order grouping
enum ORDER_GROUP_TYPE {
   Single=1,                                              //--- Group as single order
   SymbolOrderType=2,                                     //--- Group by symbol and order type
   Basket=3,                                              //--- Group all orders as a basket
   SymbolCode=4                                           //--- Group by symbol
};

//--- Define enumerations for profit calculation
enum ORDER_PROFIT_CALCULATION_TYPE {
   Pips=1,                                                //--- Calculate profit in pips
   Money=2,                                               //--- Calculate profit in currency
   EquityPercentage=3                                     //--- Calculate profit as equity percentage
};

//--- Define enumerations for CRUD operations
enum CRUD {
   NoAction=0,                                            //--- Perform no action
   Created=1,                                             //--- Create item
   Updated=2,                                             //--- Update item
   Deleted=3                                              //--- Delete item
};
```

Here, we enhance the program by defining constants, [macros](https://www.mql5.com/en/docs/basis/preprosessor/constant), and enumerations to streamline order handling and data retrieval. We start with constants like "OP\_BUY" (0) and "OP\_SELL" (1) to represent order types, and a series of "MODE\_" constants (e.g., "MODE\_BID" = 9, "MODE\_ASK" = 10) to fetch market data such as bid/ask prices, spreads, and lot sizes. We also define string conversion macros, such as [CharToString](https://www.mql5.com/en/docs/convert/chartostring) and [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring), to simplify data type conversions.

Next, we create the "ORDER\_GROUP\_TYPE" [enumeration](https://www.mql5.com/en/book/basis/builtin_types/enums) to categorize orders (e.g., "Single" = 1, "SymbolOrderType" = 2), the "ORDER\_PROFIT\_CALCULATION\_TYPE" enumeration for profit metrics ("Pips" = 1, "Money" = 2), and the "CRUD" enumeration to manage operations ("Created" = 1, "Updated" = 2). These definitions will ensure consistent data handling and support modular signal generation for the program. We can then define some helper functions as follows.

```
//--- Copy single indicator buffer value
double CopyBufferOneValue(int handle, int index, int shift) {
   double buf[];                                          //--- Declare array for buffer data
   //--- Copy one value from indicator buffer
   if(CopyBuffer(handle, index, shift, 1, buf) > 0)
      return(buf[0]);                                     //--- Return buffer value
   return EMPTY_VALUE;                                    //--- Return EMPTY_VALUE on failure
}

//--- Retrieve current Ask price
double Ask_LibFunc() {
   MqlTick last_tick;                                     //--- Declare tick data structure
   SymbolInfoTick(_Symbol, last_tick);                    //--- Fetch latest tick for symbol
   return last_tick.ask;                                  //--- Return Ask price
}

//--- Retrieve current Bid price
double Bid_LibFunc() {
   MqlTick last_tick;                                     //--- Declare tick data structure
   SymbolInfoTick(_Symbol, last_tick);                    //--- Fetch latest tick for symbol
   return last_tick.bid;                                  //--- Return Bid price
}

//--- Retrieve account equity
double AccountEquity_LibFunc() {
   return AccountInfoDouble(ACCOUNT_EQUITY);              //--- Return current equity
}

//--- Retrieve account free margin
double AccountFreeMargin_LibFunc() {
   return AccountInfoDouble(ACCOUNT_MARGIN_FREE);         //--- Return free margin
}

//--- Retrieve market information for a symbol
double MarketInfo_LibFunc(string symbol, int type) {
   switch(type) {                                                           //--- Handle requested info type
   case MODE_LOW:
      return(SymbolInfoDouble(symbol, SYMBOL_LASTLOW));                     //--- Return last low price
   case MODE_HIGH:
      return(SymbolInfoDouble(symbol, SYMBOL_LASTHIGH));                    //--- Return last high price
   case MODE_TIME:
      return((double)SymbolInfoInteger(symbol, SYMBOL_TIME));               //--- Return symbol time
   case MODE_BID:
      return(Bid_LibFunc());                                                //--- Return Bid price
   case MODE_ASK:
      return(Ask_LibFunc());                                                //--- Return Ask price
   case MODE_POINT:
      return(SymbolInfoDouble(symbol, SYMBOL_POINT));                       //--- Return point size
   case MODE_DIGITS:
      return((double)SymbolInfoInteger(symbol, SYMBOL_DIGITS));             //--- Return digit count
   case MODE_SPREAD:
      return((double)SymbolInfoInteger(symbol, SYMBOL_SPREAD));             //--- Return spread
   case MODE_STOPLEVEL:
      return((double)SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL));  //--- Return stop level
   case MODE_LOTSIZE:
      return(SymbolInfoDouble(symbol, SYMBOL_TRADE_CONTRACT_SIZE));         //--- Return contract size
   case MODE_TICKVALUE:
      return(SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE));            //--- Return tick value
   case MODE_TICKSIZE:
      return(SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE));             //--- Return tick size
   case MODE_SWAPLONG:
      return(SymbolInfoDouble(symbol, SYMBOL_SWAP_LONG));                   //--- Return swap long
   case MODE_SWAPSHORT:
      return(SymbolInfoDouble(symbol, SYMBOL_SWAP_SHORT));                  //--- Return swap short
   case MODE_MINLOT:
      return(SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN));                  //--- Return minimum lot
   case MODE_LOTSTEP:
      return(SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP));                 //--- Return lot step
   case MODE_MAXLOT:
      return(SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX));                  //--- Return maximum lot
   case MODE_SWAPTYPE:
      return((double)SymbolInfoInteger(symbol, SYMBOL_SWAP_MODE));          //--- Return swap mode
   case MODE_PROFITCALCMODE:
      return((double)SymbolInfoInteger(symbol, SYMBOL_TRADE_CALC_MODE));    //--- Return profit calc mode
   case MODE_FREEZELEVEL:
      return((double)SymbolInfoInteger(symbol, SYMBOL_TRADE_FREEZE_LEVEL)); //--- Return freeze level
   default:
      return(0);                                         //--- Return 0 for unknown type
   }
   return(0);                                            //--- Ensure fallback return
}
```

We create utility functions to enable efficient data retrieval for signal generation. We make the "CopyBufferOneValue" function, which takes an indicator handle, buffer index, and shift as inputs, copies a single value from an indicator buffer into the "buf" array, and returns the value or [EMPTY\_VALUE](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants) on failure. This function is crucial for fetching precise indicator data, such as Envelope band values.

Next, we define the "Ask\_LibFunc" and "Bid\_LibFunc" functions to retrieve current Ask and Bid prices, respectively, using the [MqlTick](https://www.mql5.com/en/docs/constants/structures/mqltick) structure and [SymbolInfoTick](https://www.mql5.com/en/docs/marketinformation/symbolinfotick) to capture the latest tick data for the current symbol. We also implement the "AccountEquity\_LibFunc" and "AccountFreeMargin\_LibFunc" functions to return the account’s equity and free margin via [AccountInfoDouble](https://www.mql5.com/en/docs/account/accountinfodouble), supporting risk management calculations.

Finally, we create the "MarketInfo\_LibFunc" function, which uses a switch statement with "MODE\_" constants (e.g., "MODE\_BID", "MODE\_ASK") to fetch various symbol properties like spread, lot size, or swap rates, returning 0 for unsupported types. These functions will provide the data foundation for generating accurate trading signals. We can now graduate to defining classes and functions that will house the main logic.

```
//--- Define function interface
interface IFunction {
   double GetValue(int index);                            //--- Retrieve value at index
   void Evaluate();                                       //--- Execute function evaluation
   void Init();                                           //--- Initialize function
};

//--- Define base class for handling double values in a circular buffer
class DoubleFunction : public IFunction {
private:
   double _values[];                                      //--- Store array of historical values
   int _zeroIndex;                                        //--- Track current index in circular buffer

protected:
   int ValueCount;                                        //--- Define number of values to store

public:
   //--- Initialize the circular buffer
   void Init() {
      _zeroIndex = -1;                                    //--- Set initial index to -1
      ArrayResize(_values, ValueCount);                   //--- Resize array to hold ValueCount elements
      ArrayInitialize(_values, GetCurrentValue());        //--- Fill array with current value
   }

   //--- Update buffer with new value
   void Evaluate() {
      double currentValue = GetCurrentValue();            //--- Retrieve current value
      _zeroIndex = (_zeroIndex + 1) % ValueCount;         //--- Increment index, wrap around if needed
      _values[_zeroIndex] = currentValue;                 //--- Store new value at current index
   }

   //--- Retrieve value at specified index
   double GetValue(int requestIndex = 0) {
      int requiredIndex = (_zeroIndex + ValueCount - requestIndex) % ValueCount; //--- Calculate index for requested value
      return _values[requiredIndex];                      //--- Return value at calculated index
   }

   //--- Declare pure virtual method for getting current value
   virtual double GetCurrentValue() = 0;                  //--- Require derived classes to implement
};

//--- Define base class for Ask and Bid price functions
class AskBidFunction : public DoubleFunction {
public:
   //--- Initialize AskBidFunction
   void AskBidFunction() {
      ValueCount = 2;                                     //--- Set buffer to store 2 values
   }
};

//--- Define class for retrieving Ask price
class AskFunction : public AskBidFunction {
public:
   //--- Retrieve current Ask price
   double GetCurrentValue() {
      return Ask_LibFunc();                               //--- Return Ask price using utility function
   }
};

//--- Define class for retrieving Bid price
class BidFunction : public AskBidFunction {
public:
   //--- Retrieve current Bid price
   double GetCurrentValue() {
      return Bid_LibFunc();                               //--- Return Bid price using utility function
   }
};

//--- Declare global function pointers for Ask and Bid
IFunction *AskFunc;                                       //--- Point to Ask price function
IFunction *BidFunc;                                       //--- Point to Bid price function

//--- Retrieve order filling type for current symbol
uint GetFillingType() {
   uint fillingType = -1;                                 //--- Initialize filling type as invalid
   uint filling = (uint)SymbolInfoInteger(Symbol(), SYMBOL_FILLING_MODE); //--- Get symbol filling mode
   if ((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK) {
      fillingType = ORDER_FILLING_FOK;                    //--- Set Fill or Kill type
      Print("Filling type: FOK");                         //--- Log FOK filling type
   } else if ((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC) {
      fillingType = ORDER_FILLING_IOC;                    //--- Set Immediate or Cancel type
      Print("Filling type: IOC");                         //--- Log IOC filling type
   } else {
      fillingType = ORDER_FILLING_RETURN;                 //--- Set Return type as default
      Print("Filling type: RETURN");                      //--- Log Return filling type
   }
   return fillingType;                                    //--- Return determined filling type
}

//--- Retrieve trade execution mode for current symbol
uint GetExecutionType() {
   uint executionType = -1;                               //--- Initialize execution type as invalid
   uint execution = (uint)SymbolInfoInteger(Symbol(), SYMBOL_TRADE_EXEMODE); //--- Get symbol execution mode
   if ((execution & SYMBOL_TRADE_EXECUTION_MARKET) == SYMBOL_TRADE_EXECUTION_MARKET) {
      executionType = SYMBOL_TRADE_EXECUTION_MARKET;      //--- Set Market execution mode
      Print("Deal execution mode: Market execution, deviation setting will be ignored."); //--- Log Market mode
   } else if ((execution & SYMBOL_TRADE_EXECUTION_INSTANT) == SYMBOL_TRADE_EXECUTION_INSTANT) {
      executionType = SYMBOL_TRADE_EXECUTION_INSTANT;     //--- Set Instant execution mode
      Print("Deal execution mode: Instant execution, deviation setting might be taken into account, depending on your broker."); //--- Log Instant mode
   } else if ((execution & SYMBOL_TRADE_EXECUTION_REQUEST) == SYMBOL_TRADE_EXECUTION_REQUEST) {
      executionType = SYMBOL_TRADE_EXECUTION_REQUEST;     //--- Set Request execution mode
      Print("Deal execution mode: Request execution, deviation setting might be taken into account, depending on your broker."); //--- Log Request mode
   } else if ((execution & SYMBOL_TRADE_EXECUTION_EXCHANGE) == SYMBOL_TRADE_EXECUTION_EXCHANGE) {
      executionType = SYMBOL_TRADE_EXECUTION_EXCHANGE;    //--- Set Exchange execution mode
      Print("Deal execution mode: Exchange execution, deviation setting will be ignored."); //--- Log Exchange mode
   }
   return executionType;                                  //--- Return determined execution type
}

//--- Retrieve account margin mode
uint GetAccountMarginMode() {
   uint marginMode = -1;                                  //--- Initialize margin mode as invalid
   marginMode = (ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE); //--- Get account margin mode
   if (marginMode == ACCOUNT_MARGIN_MODE_RETAIL_NETTING) {
      Print("Account margin mode: Netting");              //--- Log Netting mode
   } else if (marginMode == ACCOUNT_MARGIN_MODE_RETAIL_HEDGING) {
      Print("Account margin mode: Hedging");              //--- Log Hedging mode
   } else if (marginMode == ACCOUNT_MARGIN_MODE_EXCHANGE) {
      Print("Account margin mode: Exchange");             //--- Log Exchange mode
   } else {
      Print("Unknown margin type");                       //--- Log unknown margin mode
   }
   return marginMode;                                     //--- Return determined margin mode
}

//--- Retrieve description for trade error code
string GetErrorDescription(int error_code) {
   string description = "";                               //--- Initialize empty description
   switch (error_code) {                                  //--- Match error code to description
   case 10004: description = "Requote"; break;            //--- Set Requote error
   case 10006: description = "Request rejected"; break;   //--- Set Request rejected error
   case 10007: description = "Request canceled by trader"; break; //--- Set Trader cancel error
   case 10008: description = "Order placed"; break;       //--- Set Order placed status
   case 10009: description = "Request completed"; break;  //--- Set Request completed status
   case 10010: description = "Only part of the request was completed"; break; //--- Set Partial completion error
   case 10011: description = "Request processing error"; break; //--- Set Processing error
   case 10012: description = "Request canceled by timeout"; break; //--- Set Timeout cancel error
   case 10013: description = "Invalid request"; break;    //--- Set Invalid request error
   case 10014: description = "Invalid volume in the request"; break; //--- Set Invalid volume error
   case 10015: description = "Invalid price in the request"; break; //--- Set Invalid price error
   case 10016: description = "Invalid stops in the request"; break; //--- Set Invalid stops error
   case 10017: description = "Trade is disabled"; break;  //--- Set Trade disabled error
   case 10018: description = "Market is closed"; break;   //--- Set Market closed error
   case 10019: description = "There is not enough money to complete the request"; break; //--- Set Insufficient funds error
   case 10020: description = "Prices changed"; break;     //--- Set Price change error
   case 10021: description = "There are no quotes to process the request"; break; //--- Set No quotes error
   case 10022: description = "Invalid order expiration date in the request"; break; //--- Set Invalid expiration error
   case 10023: description = "Order state changed"; break; //--- Set Order state change error
   case 10024: description = "Too frequent requests"; break; //--- Set Too frequent requests error
   case 10025: description = "No changes in request"; break; //--- Set No changes error
   case 10026: description = "Autotrading disabled by server"; break; //--- Set Server autotrading disabled error
   case 10027: description = "Autotrading disabled by client terminal"; break; //--- Set Client autotrading disabled error
   case 10028: description = "Request locked for processing"; break; //--- Set Request locked error
   case 10029: description = "Order or position frozen"; break; //--- Set Frozen order error
   case 10030: description = "Invalid order filling type"; break; //--- Set Invalid filling type error
   case 10031: description = "No connection with the trade server"; break; //--- Set No server connection error
   case 10032: description = "Operation is allowed only for live accounts"; break; //--- Set Live account only error
   case 10033: description = "The number of pending orders has reached the limit"; break; //--- Set Pending order limit error
   case 10034: description = "The volume of orders and positions for the symbol has reached the limit"; break; //--- Set Symbol volume limit error
   case 10035: description = "Incorrect or prohibited order type"; break; //--- Set Incorrect order type error
   case 10036: description = "Position with the specified POSITION_IDENTIFIER has already been closed"; break; //--- Set Position closed error
   case 10038: description = "A close volume exceeds the current position volume"; break; //--- Set Excessive close volume error
   case 10039: description = "A close order already exists for a specified position"; break; //--- Set Existing close order error
   case 10040: description = "The number of open positions simultaneously present on an account has reached the limit"; break; //--- Set Position limit error
   case 10041: description = "The pending order activation request is rejected, the order is canceled"; break; //--- Set Order activation rejected error
   case 10042: description = "The request is rejected, because the 'Only long positions are allowed' rule is set for the symbol"; break; //--- Set Long-only rule error
   case 10043: description = "The request is rejected, because the 'Only short positions are allowed' rule is set for the symbol"; break; //--- Set Short-only rule error
   case 10044: description = "The request is rejected, because the 'Only position closing is allowed' rule is set for the symbol"; break; //--- Set Close-only rule error
   case 10045: description = "The request is rejected, because 'Position closing is allowed only by FIFO rule' flag is set for the trading account"; break; //--- Set FIFO closing rule error
   case 10046: description = "The request is rejected, because the 'Opposite positions on a single symbol are disabled' rule is set for the trading account"; break; //--- Set Opposite positions disabled error
   default: description = "Unknown error code " + IntegerToString(error_code); break; //--- Set unknown error with code
   }
   return description;                                    //--- Return error description
}

//--- Set pip point value for current symbol
void SetPipPoint() {
   if (PipPointOverride != 0) {
      PipPoint = PipPointOverride;                        //--- Use manual override if specified
   } else {
      PipPoint = GetRealPipPoint(Symbol());               //--- Calculate pip point automatically
   }
   Print("Pip (forex)/ Point (indices): " + DoubleToStr(PipPoint, 5)); //--- Log calculated pip point
}

//--- Calculate real pip point based on symbol digits
double GetRealPipPoint(string Currency) {
   double calcPoint = 0;                                  //--- Initialize pip point value
   double calcDigits = Digits();                          //--- Get symbol's decimal digits
   Print("Number of digits after decimal point: " + DoubleToString(calcDigits)); //--- Log digit count
   if (calcDigits == 0) {
      calcPoint = 1;                                      //--- Set pip point to 1 for 0 digits
   } else if (calcDigits == 1) {
      calcPoint = 1;                                      //--- Set pip point to 1 for 1 digit
   } else if (calcDigits == 2) {
      calcPoint = 0.1;                                    //--- Set pip point to 0.1 for 2 digits
   } else if (calcDigits == 3) {
      calcPoint = 0.01;                                   //--- Set pip point to 0.01 for 3 digits
   } else if (calcDigits == 4 || calcDigits == 5) {
      calcPoint = 0.0001;                                 //--- Set pip point to 0.0001 for 4 or 5 digits
   }
   return calcPoint;                                      //--- Return calculated pip point
}

//--- Calculate required margin for an order
bool MarginRequired(ENUM_ORDER_TYPE type, double volume, double &marginRequired) {
   double price;                                          //--- Declare price variable
   if (type == ORDER_TYPE_BUY) {
      price = Ask_LibFunc();                              //--- Set price to Ask for Buy orders
   } else if (type == ORDER_TYPE_SELL) {
      price = Bid_LibFunc();                              //--- Set price to Bid for Sell orders
   } else {
      string message = "MarginRequired: Unsupported ENUM_ORDER_TYPE"; //--- Prepare error message
      HandleErrors(message);                              //--- Log unsupported order type error
      price = Ask_LibFunc();                              //--- Default to Ask price
   }
   if (!OrderCalcMargin(type, _Symbol, volume, price, marginRequired)) {
      HandleErrors(StringFormat("Couldn't calculate required margin, error: %d", GetLastError())); //--- Log margin calculation error
      return false;                                       //--- Return false on failure
   }
   return true;                                           //--- Return true on success
}

//--- Create horizontal line on chart for TP/SL visualization
bool HLineCreate(const long chart_ID = 0, const string name = "HLine", const int sub_window = 0,
                 double price = 0, const color clr = clrRed, const ENUM_LINE_STYLE style = STYLE_SOLID,
                 const int width = 1, const bool back = false, const bool selection = true,
                 const bool hidden = true, const long z_order = 0) {
   uint lineFindResult = ObjectFind(chart_ID, name);      //--- Check if line already exists
   if (lineFindResult != UINT_MAX) {
      Print("HLineCreate object already exists: " + name); //--- Log existing line error
      return false;                                       //--- Return false if line exists
   }
   if (!price) {
      price = Bid_LibFunc();                              //--- Default to Bid price if not specified
   }
   ResetLastError();                                      //--- Clear last error
   if (!ObjectCreate(chart_ID, name, OBJ_HLINE, sub_window, 0, price)) {
      Print(__FUNCTION__, ": failed to create a horizontal line! Error code = ", GetLastError()); //--- Log line creation error
      return false;                                       //--- Return false on failure
   }
   ObjectSetInteger(chart_ID, name, OBJPROP_COLOR, clr);  //--- Set line color
   ObjectSetInteger(chart_ID, name, OBJPROP_STYLE, style); //--- Set line style
   ObjectSetInteger(chart_ID, name, OBJPROP_WIDTH, width); //--- Set line width
   ObjectSetInteger(chart_ID, name, OBJPROP_BACK, back);   //--- Set background rendering
   if (AllowManualTPSLChanges) {
      ObjectSetInteger(chart_ID, name, OBJPROP_SELECTABLE, selection); //--- Enable line selection
      ObjectSetInteger(chart_ID, name, OBJPROP_SELECTED, selection);   //--- Set line as selected
   }
   ObjectSetInteger(chart_ID, name, OBJPROP_HIDDEN, hidden); //--- Hide line in object list
   ObjectSetInteger(chart_ID, name, OBJPROP_ZORDER, z_order); //--- Set mouse click priority
   return true;                                           //--- Return true on success
}

//--- Move existing horizontal line on chart
bool HLineMove(const long chart_ID = 0, const string name = "HLine", double price = 0) {
   uint lineFindResult = ObjectFind(ChartID(), name);     //--- Check if line exists
   if (lineFindResult == UINT_MAX) {
      Print("HLineMove didn't find object: " + name);      //--- Log missing line error
      return false;                                       //--- Return false if line not found
   }
   if (!price) {
      price = SymbolInfoDouble(Symbol(), SYMBOL_BID);      //--- Default to Bid price if not specified
   }
   ResetLastError();                                      //--- Clear last error
   if (!ObjectMove(chart_ID, name, 0, 0, price)) {
      Print(__FUNCTION__, ": failed to move the horizontal line! Error code = ", GetLastError()); //--- Log line move error
      return false;                                       //--- Return false on failure
   }
   return true;                                           //--- Return true on success
}

//--- Delete chart object by name
bool AnyChartObjectDelete(const long chart_ID = 0, const string name = "") {
   uint lineFindResult = ObjectFind(ChartID(), name);     //--- Check if object exists
   if (lineFindResult == UINT_MAX) {
      return false;                                       //--- Return false if object not found
   }
   ResetLastError();                                      //--- Clear last error
   if (!ObjectDelete(chart_ID, name)) {
      Print(__FUNCTION__, ": failed to delete a horizontal line! Error code = ", GetLastError()); //--- Log deletion error
      return false;                                       //--- Return false on failure
   }
   return true;                                           //--- Return true on success
}

//--- Placeholder function for future use
int Dummy(string message) {
   return 0;                                              //--- Return 0 (no operation)
}
```

Here, we advance the program by defining an interface, classes, and utility functions to manage price data and chart visuals. We create the "IFunction" [interface](https://www.mql5.com/en/docs/basis/types/classes), specifying methods "GetValue", "Evaluate", and "Init" to standardize data retrieval for price functions. We then define the "DoubleFunction" class, inheriting from "IFunction", to manage a circular buffer with the "\_values" array and "\_zeroIndex" variable, implementing "Init" to set up the buffer, "Evaluate" to update values, and "GetValue" to fetch historical data, with a pure virtual "GetCurrentValue" method for subclasses.

We derive the "AskBidFunction" class from "DoubleFunction", setting "ValueCount" to 2 for buffering Ask/Bid prices, and create the "AskFunction" and "BidFunction" classes to return current prices via "Ask\_LibFunc" and "Bid\_LibFunc" in their "GetCurrentValue" methods, respectively. Global pointers "AskFunc" and "BidFunc" are declared to access these functions. We also implement the "GetFillingType", "GetExecutionType", and "GetAccountMarginMode" functions to determine broker-specific settings, logging modes like [ORDER\_FILLING\_FOK](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type_filling) or "ACCOUNT\_MARGIN\_MODE\_RETAIL\_HEDGING". The "GetErrorDescription" function maps error codes to readable strings, aiding debugging.

Additionally, we define "SetPipPoint" and "GetRealPipPoint" to calculate the "PipPoint" variable based on symbol digits, "MarginRequired" to compute margin needs using "Ask\_LibFunc" or "Bid\_LibFunc", and "HLineCreate", "HLineMove", and "AnyChartObjectDelete" to manage chart lines for TP/SL visualization, with "Dummy" as a placeholder for future use. We can now declare the indicators to be used, and so, we will need to define some extra inputs for dynamic control as below.

```
//--- Input parameters for indicators
input int iMA_SMA8_ma_period = 14;                        //--- Set SMA period for trend filter (M30 timeframe)
input int iMA_SMA8_ma_shift = 2;                          //--- Set SMA shift for trend filter
input int iMA_SMA_4_ma_period = 9;                        //--- Set SMA period for reverse trend filter (M30 timeframe)
input int iMA_SMA_4_ma_shift = 0;                         //--- Set SMA shift for reverse trend filter
input int iMA_EMA200_ma_period = 200;                     //--- Set EMA period for long-term trend (M1 timeframe)
input int iMA_EMA200_ma_shift = 0;                        //--- Set EMA shift for long-term trend
input int iRSI_RSI_ma_period = 8;                         //--- Set RSI period for overbought/oversold signals (M1 timeframe)
input int iEnvelopes_ENV_LOW_ma_period = 95;              //--- Set Envelopes period for lower band (M1 timeframe)
input int iEnvelopes_ENV_LOW_ma_shift = 0;                //--- Set Envelopes shift for lower band
input double iEnvelopes_ENV_LOW_deviation = 1.4;          //--- Set Envelopes deviation for lower band (1.4%)
input int iEnvelopes_ENV_UPPER_ma_period = 150;           //--- Set Envelopes period for upper band (M1 timeframe)
input int iEnvelopes_ENV_UPPER_ma_shift = 0;              //--- Set Envelopes shift for upper band
input double iEnvelopes_ENV_UPPER_deviation = 0.1;        //--- Set Envelopes deviation for upper band (0.1%)

//--- Indicator handle declarations
int hd_iMA_SMA8;                                          //--- Store handle for 8-period SMA
//--- Retrieve 8-period SMA value
double fn_iMA_SMA8(string symbol, int shift) {
   int index = 0;                                         //--- Set buffer index to 0
   return CopyBufferOneValue(hd_iMA_SMA8, index, shift);  //--- Return SMA value at specified shift
}

int hd_iMA_EMA200;                                        //--- Store handle for 200-period EMA
//--- Retrieve 200-period EMA value
double fn_iMA_EMA200(string symbol, int shift) {
   int index = 0;                                         //--- Set buffer index to 0
   return CopyBufferOneValue(hd_iMA_EMA200, index, shift); //--- Return EMA value at specified shift
}

int hd_iRSI_RSI;                                          //--- Store handle for 8-period RSI
//--- Retrieve RSI value
double fn_iRSI_RSI(string symbol, int shift) {
   int index = 0;                                         //--- Set buffer index to 0
   return CopyBufferOneValue(hd_iRSI_RSI, index, shift);  //--- Return RSI value at specified shift
}

int hd_iEnvelopes_ENV_LOW;                                //--- Store handle for lower Envelopes band
//--- Retrieve lower Envelopes band value
double fn_iEnvelopes_ENV_LOW(string symbol, int mode, int shift) {
   int index = mode;                                      //--- Set buffer index to specified mode
   return CopyBufferOneValue(hd_iEnvelopes_ENV_LOW, index, shift); //--- Return lower Envelopes value
}

int hd_iEnvelopes_ENV_UPPER;                              //--- Store handle for upper Envelopes band
//--- Retrieve upper Envelopes band value
double fn_iEnvelopes_ENV_UPPER(string symbol, int mode, int shift) {
   int index = mode;                                      //--- Set buffer index to specified mode
   return CopyBufferOneValue(hd_iEnvelopes_ENV_UPPER, index, shift); //--- Return upper Envelopes value
}

int hd_iMA_SMA_4;                                         //--- Store handle for 4-period SMA
//--- Retrieve 4-period SMA value
double fn_iMA_SMA_4(string symbol, int shift) {
   int index = 0;                                         //--- Set buffer index to 0
   return CopyBufferOneValue(hd_iMA_SMA_4, index, shift); //--- Return SMA value at specified shift
}
```

To configure the indicator settings, we use the "input" directive to declare the external user inputs and then call the "CopyBufferOneValue" function to get the specified values of the indicators. We can then create a class for order management to lay the core infrastructure backbone.

```
//--- Commission variables
double CommissionAmountPerTrade = 0.0;                    //--- Set fixed commission per trade (default: 0)
double CommissionPercentagePerLot = 0.0;                  //--- Set commission percentage per lot (default: 0)
double CommissionAmountPerLot = 0.0;                      //--- Set fixed commission per lot (default: 0)
double TotalCommission = 0.0;                             //--- Track total commission for all trades
bool UseCommissionInProfitInPips = false;                 //--- Exclude commission from pip profit if false

//--- Define class for order close information
class OrderCloseInfo {
public:
   string ModuleCode;                                     //--- Store module identifier for close condition
   double Price;                                          //--- Store price for TP or SL
   int Percentage;                                        //--- Store percentage of order to close
   bool IsOld;                                            //--- Flag outdated close info

   //--- Default constructor
   void OrderCloseInfo() {}                               //--- Initialize empty close info

   //--- Copy constructor
   void OrderCloseInfo(OrderCloseInfo* ordercloseinfo) {
      ModuleCode = ordercloseinfo.ModuleCode;             //--- Copy module code
      Price = ordercloseinfo.Price;                       //--- Copy price
      Percentage = ordercloseinfo.Percentage;             //--- Copy percentage
      IsOld = ordercloseinfo.IsOld;                       //--- Copy old flag
   }

   //--- Check if Stop Loss is hit
   bool IsClosePriceSLHit(ENUM_ORDER_TYPE type, double ask, double bid) {
      switch (type) {
      case ORDER_TYPE_BUY:
         return bid <= Price;                             //--- Return true if Bid falls below SL for Buy
      case ORDER_TYPE_SELL:
         return ask >= Price;                             //--- Return true if Ask rises above SL for Sell
      }
      return false;                                       //--- Return false for invalid type
   }

   //--- Check if Take Profit is hit
   bool IsClosePriceTPHit(ENUM_ORDER_TYPE type, double ask, double bid) {
      switch (type) {
      case ORDER_TYPE_BUY:
         return bid >= Price;                             //--- Return true if Bid reaches TP for Buy
      case ORDER_TYPE_SELL:
         return ask <= Price;                             //--- Return true if Ask reaches TP for Sell
      }
      return false;                                       //--- Return false for invalid type
   }

   //--- Destructor
   void ~OrderCloseInfo() {}                              //--- Clean up close info
};

//--- Define class for managing order details
class Order {
public:
   ulong Ticket;                                          //--- Store unique order ticket
   ENUM_ORDER_TYPE Type;                                  //--- Store order type (Buy/Sell)
   ENUM_ORDER_STATE State;                                //--- Store order state (e.g., Filled)
   long MagicNumber;                                      //--- Store EA’s magic number
   double Lots;                                           //--- Store order volume in lots
   double OrderFilledLots;                                //--- Store filled volume
   datetime OpenTime;                                     //--- Store order open time
   double OpenPrice;                                      //--- Store order open price
   datetime CloseTime;                                    //--- Store order close time
   double ClosePrice;                                     //--- Store order close price
   double StopLoss;                                       //--- Store Stop Loss price
   double StopLossManual;                                 //--- Store manually set Stop Loss
   double TakeProfit;                                     //--- Store Take Profit price
   double TakeProfitManual;                               //--- Store manually set Take Profit
   datetime Expiration;                                   //--- Store order expiration time
   double CurrentProfitPips;                              //--- Store current profit in pips
   double HighestProfitPips;                              //--- Store highest profit in pips
   double LowestProfitPips;                               //--- Store lowest profit in pips
   string Comment;                                        //--- Store order comment
   uint TradeRetCode;                                     //--- Store trade result code
   ulong TradeDealTicket;                                 //--- Store deal ticket
   double TradePrice;                                     //--- Store trade price
   double TradeVolume;                                    //--- Store trade volume
   double Commission;                                     //--- Store commission cost
   double CommissionInPips;                               //--- Store commission in pips
   string SymbolCode;                                     //--- Store symbol code
   bool IsAwaitingDealExecution;                          //--- Flag pending deal execution
   OrderCloseInfo* CloseInfosTP[];                        //--- Store Take Profit close info
   OrderCloseInfo* CloseInfosSL[];                        //--- Store Stop Loss close info
   Order* ParentOrder;                                    //--- Store parent order for splits
   bool MustBeVisibleOnChart;                             //--- Flag chart visibility

   //--- Initialize order with visibility flag
   void Order(bool mustBeVisibleOnChart) {
      OrderFilledLots = 0.0;                              //--- Set filled lots to 0
      OpenPrice = 0.0;                                    //--- Set open price to 0
      ClosePrice = 0.0;                                   //--- Set close price to 0
      Commission = 0.0;                                   //--- Set commission to 0
      CommissionInPips = 0.0;                             //--- Set commission in pips to 0
      MustBeVisibleOnChart = mustBeVisibleOnChart;        //--- Set chart visibility flag
   }

   //--- Copy order details with visibility flag
   void Order(Order* order, bool mustBeVisibleOnChart) {
      Ticket = order.Ticket;                              //--- Copy ticket
      Type = order.Type;                                  //--- Copy order type
      State = order.State;                                //--- Copy order state
      MagicNumber = order.MagicNumber;                    //--- Copy magic number
      Lots = order.Lots;                                  //--- Copy lots
      OpenTime = order.OpenTime;                          //--- Copy open time
      OpenPrice = order.OpenPrice;                        //--- Copy open price
      CloseTime = order.CloseTime;                        //--- Copy close time
      ClosePrice = order.ClosePrice;                      //--- Copy close price
      StopLoss = order.StopLoss;                          //--- Copy Stop Loss
      StopLossManual = order.StopLossManual;              //--- Copy manual Stop Loss
      TakeProfit = order.TakeProfit;                      //--- Copy Take Profit
      TakeProfitManual = order.TakeProfitManual;          //--- Copy manual Take Profit
      Expiration = order.Expiration;                      //--- Copy expiration
      CurrentProfitPips = order.CurrentProfitPips;        //--- Copy current profit
      HighestProfitPips = order.HighestProfitPips;        //--- Copy highest profit
      LowestProfitPips = order.LowestProfitPips;          //--- Copy lowest profit
      Comment = order.Comment;                            //--- Copy comment
      TradeRetCode = order.TradeRetCode;                  //--- Copy trade result code
      TradeDealTicket = order.TradeDealTicket;            //--- Copy deal ticket
      TradePrice = order.TradePrice;                      //--- Copy trade price
      TradeVolume = order.TradeVolume;                    //--- Copy trade volume
      Commission = order.Commission;                      //--- Copy commission
      CommissionInPips = order.CommissionInPips;          //--- Copy commission in pips
      SymbolCode = order.SymbolCode;                      //--- Copy symbol code
      IsAwaitingDealExecution = order.IsAwaitingDealExecution; //--- Copy execution flag
      ParentOrder = order.ParentOrder;                    //--- Copy parent order
      MustBeVisibleOnChart = mustBeVisibleOnChart;        //--- Set visibility flag
   }

   //--- Split order into partial close
   Order* SplitOrder(int percentageToSplitOff) {
      Order* splittedOffPieceOfOrder = new Order(&this, true); //--- Create new order for split
      splittedOffPieceOfOrder.Lots = CalcVolumePartialClose(this.Lots, percentageToSplitOff); //--- Calculate split volume
      if (this.Lots - splittedOffPieceOfOrder.Lots < 1e-13) {
         splittedOffPieceOfOrder.MustBeVisibleOnChart = false; //--- Hide split if no volume remains
         splittedOffPieceOfOrder.Lots = 0;                   //--- Set split volume to 0
      } else {
         this.Lots = this.Lots - splittedOffPieceOfOrder.Lots; //--- Reduce original order volume
      }
      return splittedOffPieceOfOrder;                        //--- Return split order
   }

   //--- Calculate profit in pipettes
   double CalculateProfitPipettes() {
      double closePrice = GetClosePrice();                   //--- Get current close price
      switch (Type) {
      case ORDER_TYPE_BUY:
         return (closePrice - OpenPrice);                    //--- Return Buy profit in pipettes
      case ORDER_TYPE_SELL:
         return (OpenPrice - closePrice);                    //--- Return Sell profit in pipettes
      }
      return 0;                                              //--- Return 0 for invalid type
   }

   //--- Calculate profit in pips
   double CalculateProfitPips() {
      double pipettes = CalculateProfitPipettes();           //--- Get profit in pipettes
      double pips = pipettes / PipPoint;                     //--- Convert to pips
      if (UseCommissionInProfitInPips) {
         return pips - CommissionInPips;                     //--- Subtract commission if enabled
      }
      return pips;                                           //--- Return profit in pips
   }

   //--- Calculate profit in account currency
   double CalculateProfitCurrency() {
      double closePrice = GetClosePrice();                   //--- Get current close price
      switch (Type) {
      case OP_BUY:
         return (closePrice - OpenPrice) * (UnitsOneLot * TradeVolume) - Commission; //--- Return Buy profit
      case OP_SELL:
         return (OpenPrice - closePrice) * (UnitsOneLot * TradeVolume) - Commission; //--- Return Sell profit
      }
      return 0;                                              //--- Return 0 for invalid type
   }

   //--- Calculate profit as equity percentage
   double CalculateProfitEquityPercentage() {
      double closePrice = GetClosePrice();                   //--- Get current close price
      switch (Type) {
      case OP_BUY:
         return 100 * ((closePrice - OpenPrice) * (UnitsOneLot * TradeVolume) - Commission) / AccountEquity_LibFunc(); //--- Return Buy equity percentage
      case OP_SELL:
         return 100 * ((OpenPrice - closePrice) * (UnitsOneLot * TradeVolume) - Commission) / AccountEquity_LibFunc(); //--- Return Sell equity percentage
      }
      return 0;                                              //--- Return 0 for invalid type
   }

   //--- Calculate price difference in pips
   double CalculateValueDifferencePips(double value) {
      double divOpenPrice = 0.0;                             //--- Initialize price difference
      switch (Type) {
      case OP_BUY:
         divOpenPrice = (value - OpenPrice);                 //--- Calculate Buy difference
         break;
      case OP_SELL:
         divOpenPrice = (OpenPrice - value);                 //--- Calculate Sell difference
         break;
      }
      double pipsDivOpenPrice = divOpenPrice / PipPoint;     //--- Convert to pips
      return pipsDivOpenPrice;                               //--- Return difference in pips
   }

   //--- Retrieve realized profit in pips
   double GetProfitPips() {
      if (CloseTime > 0) {                                   //--- Check if order is closed
         switch (Type) {
         case ORDER_TYPE_BUY: {
            double pipettes = ClosePrice - OpenPrice;        //--- Calculate Buy pipettes
            return pipettes / PipPoint;                      //--- Return Buy profit in pips
         }
         case ORDER_TYPE_SELL: {
            double pipettes = OpenPrice - ClosePrice;        //--- Calculate Sell pipettes
            return pipettes / PipPoint;                      //--- Return Sell profit in pips
         }
         }
      }
      return 0;                                              //--- Return 0 if not closed
   }

   //--- Check if module has processed close info
   bool IsAlreadyProcessedByModule(string moduleCode, OrderCloseInfo* &closeInfos[]) {
      for (int i = 0; i < ArraySize(closeInfos); i++) {
         if (closeInfos[i].ModuleCode == moduleCode && closeInfos[i].IsOld) {
            return true;                                     //--- Return true if processed and old
         }
      }
      return false;                                          //--- Return false if not processed
   }

   //--- Check if module has active close info
   bool HasAValueAlreadyByModule(string moduleCode, OrderCloseInfo* &closeInfos[]) {
      for (int i = 0; i < ArraySize(closeInfos); i++) {
         if (closeInfos[i].ModuleCode == moduleCode && !closeInfos[i].IsOld) {
            return true;                                     //--- Return true if active
         }
      }
      return false;                                          //--- Return false if no active info
   }

   //--- Draw TP and SL lines on chart
   void Paint() {
      if (IsDemoLiveOrVisualMode) {                          //--- Check for demo/live/visual mode
         for (int i = 0; i < ArraySize(CloseInfosTP); i++) {
            if (CloseInfosTP[i].IsOld) continue;             //--- Skip outdated TP info
            PaintTPInfo(CloseInfosTP[i].Price);              //--- Draw TP line
         }
         for (int i = 0; i < ArraySize(CloseInfosSL); i++) {
            if (CloseInfosSL[i].IsOld) continue;             //--- Skip outdated SL info
            PaintSLInfo(CloseInfosSL[i].Price);              //--- Draw SL line
         }
      }
   }

   //--- Set Take Profit information
   bool SetTPInfo(string moduleCode, double price, int percentage) {
      uint result = SetCloseInfo(CloseInfosTP, moduleCode, price, percentage); //--- Update TP info
      if (result != NoAction) {
         if (IsDemoLiveOrVisualMode) {
            PaintTPInfo(price);                              //--- Draw TP line
         }
         return true;                                        //--- Return true on success
      }
      return false;                                          //--- Return false on no action
   }

   //--- Set Stop Loss information
   bool SetSLInfo(string moduleCode, double price, int percentage) {
      uint result = SetCloseInfo(CloseInfosSL, moduleCode, price, percentage); //--- Update SL info
      if (result != NoAction) {
         if (IsDemoLiveOrVisualMode) {
            PaintSLInfo(price);                              //--- Draw SL line
         }
         return true;                                        //--- Return true on success
      }
      return false;                                          //--- Return false on no action
   }

   //--- Retrieve closest Stop Loss price
   double GetClosestSL() {
      double closestSL = 0;                                  //--- Initialize closest SL
      for (int cli = 0; cli < ArraySize(CloseInfosSL); cli++) {
         if (CloseInfosSL[cli].IsOld) continue;              //--- Skip outdated SL
         if ((Type == ORDER_TYPE_BUY && (closestSL == 0 || CloseInfosSL[cli].Price > closestSL)) ||
             (Type == ORDER_TYPE_SELL && (closestSL == 0 || CloseInfosSL[cli].Price < closestSL))) {
            closestSL = CloseInfosSL[cli].Price;             //--- Update closest SL
         }
      }
      return closestSL;                                      //--- Return closest SL price
   }

   //--- Retrieve closest Take Profit price
   double GetClosestTP() {
      double closestTP = 0;                                  //--- Initialize closest TP
      for (int cli = 0; cli < ArraySize(CloseInfosTP); cli++) {
         if (CloseInfosTP[cli].IsOld) continue;              //--- Skip outdated TP
         if ((Type == ORDER_TYPE_BUY && (closestTP == 0 || CloseInfosTP[cli].Price < closestTP)) ||
             (Type == ORDER_TYPE_SELL && (closestTP == 0 || CloseInfosTP[cli].Price > closestTP))) {
            closestTP = CloseInfosTP[cli].Price;             //--- Update closest TP
         }
      }
      return closestTP;                                      //--- Return closest TP price
   }

   //--- Remove Stop Loss information
   bool RemoveSLInfo(string moduleCode) {
      RemoveCloseInfo(CloseInfosSL, moduleCode);             //--- Remove SL info
      if (IsDemoLiveOrVisualMode) {
         double newValue = NULL;                             //--- Initialize new SL value
         for (int i = 0; i < ArraySize(CloseInfosSL); i++) {
            if ((Type == OP_BUY && (newValue == NULL || CloseInfosSL[i].Price > newValue)) ||
                (Type == OP_SELL && (newValue == NULL || CloseInfosSL[i].Price < newValue))) {
               newValue = CloseInfosSL[i].Price;             //--- Update new SL value
            }
         }
         if (newValue == NULL) {
            AnyChartObjectDelete(ChartID(), IntegerToString(Ticket) + "_SL"); //--- Delete SL line
         } else {
            HLineMove(ChartID(), IntegerToString(Ticket) + "_SL", newValue); //--- Move SL line
         }
      }
      return true;                                           //--- Return true on success
   }

   //--- Remove Take Profit information
   bool RemoveTPInfo(string moduleCode) {
      RemoveCloseInfo(CloseInfosTP, moduleCode);             //--- Remove TP info
      if (IsDemoLiveOrVisualMode) {
         double newValue = NULL;                             //--- Initialize new TP value
         for (int i = 0; i < ArraySize(CloseInfosTP); i++) {
            if ((Type == OP_BUY && (newValue == NULL || CloseInfosTP[i].Price < newValue)) ||
                (Type == OP_SELL && (newValue == NULL || CloseInfosTP[i].Price > newValue))) {
               newValue = CloseInfosTP[i].Price;             //--- Update new TP value
            }
         }
         if (newValue == NULL) {
            AnyChartObjectDelete(ChartID(), IntegerToString(Ticket) + "_TP"); //--- Delete TP line
         } else {
            HLineMove(ChartID(), IntegerToString(Ticket) + "_TP", newValue); //--- Move TP line
         }
      }
      return true;                                           //--- Return true on success
   }

   //--- Set or update close info (TP or SL)
   CRUD SetCloseInfo(OrderCloseInfo* &closeInfos[], string moduleCode, double price, int percentage) {
      for (int i = 0; i < ArraySize(closeInfos); i++) {
         if (closeInfos[i].ModuleCode == moduleCode) {
            closeInfos[i].Price = price;                     //--- Update existing price
            return Updated;                                  //--- Return Updated status
         }
      }
      int newSize = ArraySize(closeInfos) + 1;              //--- Calculate new array size
      ArrayResize(closeInfos, newSize);                     //--- Resize close info array
      closeInfos[newSize-1] = new OrderCloseInfo();         //--- Create new close info
      closeInfos[newSize-1].Price = price;                  //--- Set price
      closeInfos[newSize-1].Percentage = percentage;        //--- Set percentage
      closeInfos[newSize-1].ModuleCode = moduleCode;        //--- Set module code
      return Created;                                       //--- Return Created status
   }

   //--- Remove close info (TP or SL)
   CRUD RemoveCloseInfo(OrderCloseInfo* &closeInfos[], string moduleCode) {
      int removedCount = 0;                                  //--- Track removed items
      int arraySize = ArraySize(closeInfos);                 //--- Get current array size
      for (int i = 0; i < arraySize; i++) {
         if (closeInfos[i].ModuleCode == moduleCode) {
            removedCount++;                                  //--- Increment removed count
            if (closeInfos[i] != NULL && CheckPointer(closeInfos[i]) == POINTER_DYNAMIC) {
               delete(closeInfos[i]);                        //--- Delete dynamic close info
            }
            continue;                                        //--- Skip to next item
         }
         closeInfos[i - removedCount] = closeInfos[i];       //--- Shift remaining items
      }
      ArrayResize(closeInfos, arraySize - removedCount);     //--- Resize array
      return Deleted;                                       //--- Return Deleted status
   }

   //--- Destructor for order cleanup
   void ~Order() {
      for (int i = 0; i < ArraySize(CloseInfosTP); i++) {
         if (CloseInfosTP[i] != NULL && CheckPointer(CloseInfosTP[i]) == POINTER_DYNAMIC) {
            delete(CloseInfosTP[i]);                         //--- Delete dynamic TP info
         }
      }
      for (int i = 0; i < ArraySize(CloseInfosSL); i++) {
         if (CloseInfosSL[i] != NULL && CheckPointer(CloseInfosSL[i]) == POINTER_DYNAMIC) {
            delete(CloseInfosSL[i]);                         //--- Delete dynamic SL info
         }
      }
      if (IsDemoLiveOrVisualMode && MustBeVisibleOnChart) {
         AnyChartObjectDelete(ChartID(), IntegerToString(Ticket) + "_TP"); //--- Delete TP line
         AnyChartObjectDelete(ChartID(), IntegerToString(Ticket) + "_SL"); //--- Delete SL line
      }
   }

private:
   //--- Retrieve close price for profit calculation
   double GetClosePrice() {
      if (ClosePrice > 1e-5) {
         return ClosePrice;                                  //--- Return stored close price if set
      } else if (Type == OP_BUY) {
         return SymbolInfoDouble(SymbolCode, SYMBOL_BID);    //--- Return Bid for Buy orders
      }
      return SymbolInfoDouble(SymbolCode, SYMBOL_ASK);       //--- Return Ask for Sell orders
   }

   //--- Calculate volume for partial close
   double CalcVolumePartialClose(double orderVolume, int percentage) {
      return RoundVolume(orderVolume * ((double)percentage / 100)); //--- Return rounded volume
   }

   //--- Round volume to broker specifications
   double RoundVolume(double volume) {
      string pair = Symbol();                                //--- Get current symbol
      double lotStep = MarketInfo_LibFunc(pair, MODE_LOTSTEP); //--- Get lot step
      double minLot = MarketInfo_LibFunc(pair, MODE_MINLOT); //--- Get minimum lot
      volume = MathRound(volume / lotStep) * lotStep;        //--- Round volume to lot step
      if (volume < minLot) volume = minLot;                  //--- Enforce minimum lot
      return volume;                                         //--- Return rounded volume
   }

   //--- Draw Stop Loss line on chart
   void PaintSLInfo(double value) {
      double currentValue;                                   //--- Declare current value
      if (ObjectGetDouble(ChartID(), IntegerToString(Ticket) + "_SL", OBJPROP_PRICE, 0, currentValue)) {
         if (Type == OP_BUY && value > currentValue) {
            HLineMove(ChartID(), IntegerToString(Ticket) + "_SL", value); //--- Move SL line for Buy
         } else if (Type == OP_SELL && value < currentValue) {
            HLineMove(ChartID(), IntegerToString(Ticket) + "_SL", value); //--- Move SL line for Sell
         }
      } else {
         HLineCreate(ChartID(), IntegerToString(Ticket) + "_SL", 0, value, clrRed); //--- Create red SL line
      }
   }

   //--- Draw Take Profit line on chart
   void PaintTPInfo(double value) {
      double currentValue;                                   //--- Declare current value
      if (ObjectGetDouble(ChartID(), IntegerToString(Ticket) + "_TP", OBJPROP_PRICE, 0, currentValue)) {
         if (Type == OP_BUY && value < currentValue) {
            HLineMove(ChartID(), IntegerToString(Ticket) + "_TP", value); //--- Move TP line for Buy
         } else if (Type == OP_SELL && value > currentValue) {
            HLineMove(ChartID(), IntegerToString(Ticket) + "_TP", value); //--- Move TP line for Sell
         }
      } else {
         HLineCreate(ChartID(), IntegerToString(Ticket) + "_TP", 0, value, clrGreen); //--- Create green TP line
      }
   }
};
```

To establish commission handling and order management core logic, we define commission variables like "CommissionAmountPerTrade" (0.0), "CommissionPercentagePerLot" (0.0), "CommissionAmountPerLot" (0.0), "TotalCommission" (0.0) to track trading costs, and "UseCommissionInProfitInPips" (false) to exclude commissions from pip calculations, ensuring accurate profit tracking.

We create the "OrderCloseInfo" class to manage trade closure details, with variables "ModuleCode" for module identification, "Price" for TP/SL levels, "Percentage" for partial closes, and "IsOld" to flag outdated data. Its methods include "IsClosePriceSLHit" and "IsClosePriceTPHit" to check if stop-loss or take-profit levels are reached for [ORDER\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type) or "ORDER\_TYPE\_SELL" orders, using "ask" and "bid" prices.

We then define the "Order" class to encapsulate order details, including variables like "Ticket", "Type" ( [ENUM\_ORDER\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type)), "State" ( [ENUM\_ORDER\_STATE](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_state)), "MagicNumber", "Lots", "OpenPrice", "StopLoss", "TakeProfit", and "Commission". Key methods include "Order" constructors for initialization, "SplitOrder" to handle partial closures, "CalculateProfitPips" and "CalculateProfitCurrency" for profit calculations using "PipPoint" and "UnitsOneLot", and "SetTPInfo" and "SetSLInfo" to update "CloseInfosTP" and "CloseInfosSL" arrays.

The "Paint" method draws TP/SL lines using "PaintTPInfo" and "PaintSLInfo", while "GetClosestSL" and "GetClosestTP" retrieve the nearest stop-loss and take-profit prices. The "SetCloseInfo" and "RemoveCloseInfo" methods, returning "CRUD" statuses, manage TP/SL updates, and private methods like "GetClosePrice" and "RoundVolume" ensure accurate price and volume handling. These structures will support robust order management for scalping signals. Let us then define a function to collect and group the collected orders as below.

```
//--- Define class for managing a collection of orders
class OrderCollection {
private:
   Order* _orders[];                                      //--- Store array of order pointers
   int _pointer;                                          //--- Track current iteration index
   int _size;                                             //--- Track number of orders

public:
   //--- Initialize empty order collection
   void OrderCollection() {
      _pointer = -1;                                      //--- Set initial pointer to -1
      _size = 0;                                          //--- Set initial size to 0
   }

   //--- Destructor to clean up orders
   void ~OrderCollection() {
      for (int i = 0; i < ArraySize(_orders); i++) {
         delete(_orders[i]);                              //--- Delete each order object
      }
   }

   //--- Add order to collection
   void Add(Order* item) {
      _size = _size + 1;                                  //--- Increment size
      ArrayResize(_orders, _size, 8);                     //--- Resize array with reserve capacity
      _orders[(_size - 1)] = item;                        //--- Store order at last index
   }

   //--- Remove order at specified index
   Order* Remove(int index) {
      Order* removed = NULL;                              //--- Initialize removed order as null
      if (index >= 0 && index < _size) {                  //--- Check valid index
         removed = _orders[index];                        //--- Store order to be removed
         for (int i = index; i < (_size - 1); i++) {
            _orders[i] = _orders[i + 1];                  //--- Shift orders left
         }
         ArrayResize(_orders, ArraySize(_orders) - 1, 8); //--- Reduce array size
         _size = _size - 1;                               //--- Decrement size
      }
      return removed;                                     //--- Return removed order or null
   }

   //--- Retrieve order at specified index
   Order* Get(int index) {
      if (index >= 0 && index < _size) {                  //--- Check valid index
         return _orders[index];                           //--- Return order at index
      }
      return NULL;                                        //--- Return null for invalid index
   }

   //--- Retrieve number of orders
   int Count() {
      return _size;                                       //--- Return current size
   }

   //--- Reset iterator to start
   void Rewind() {
      _pointer = -1;                                      //--- Set pointer to -1
   }

   //--- Move to next order
   Order* Next() {
      _pointer++;                                         //--- Increment pointer
      if (_pointer == _size) {                            //--- Check if at end
         Rewind();                                        //--- Reset pointer
         return NULL;                                     //--- Return null
      }
      return Current();                                   //--- Return current order
   }

   //--- Move to previous order
   Order* Prev() {
      _pointer--;                                         //--- Decrement pointer
      if (_pointer == -1) {                               //--- Check if before start
         return NULL;                                     //--- Return null
      }
      return Current();                                   //--- Return current order
   }

   //--- Check if more orders exist
   bool HasNext() {
      return (_pointer < (_size - 1));                    //--- Return true if pointer is before end
   }

   //--- Retrieve current order
   Order* Current() {
      return _orders[_pointer];                           //--- Return order at current pointer
   }

   //--- Retrieve current iterator index
   int Key() {
      return _pointer;                                    //--- Return current pointer
   }

   //--- Find index by order ticket
   int GetKeyByTicket(ulong ticket) {
      int keyFound = -1;                                  //--- Initialize found index as -1
      for (int i = 0; i < ArraySize(_orders); i++) {
         if (_orders[i].Ticket == ticket) {               //--- Check ticket match
            keyFound = i;                                 //--- Set found index
         }
      }
      return keyFound;                                    //--- Return found index or -1
   }
};

//--- Define class for managing order operations with broker
class OrderRepository {
private:
   //--- Retrieve order by ticket
   static Order* getByTicket(ulong ticket) {
      bool orderSelected = OrderSelect(ticket);           //--- Select order by ticket
      if (orderSelected) {                                //--- Check if selection succeeded
         Order* order = new Order(false);                 //--- Create new order object
         OrderRepository::fetchSelected(order);           //--- Populate order details
         return order;                                    //--- Return order object
      } else {
         return NULL;                                     //--- Return null if selection failed
      }
   }

   //--- Retrieve close time for historical order
   static datetime OrderCloseTime(ulong ticket) {
      return (datetime)(HistoryOrderGetInteger(ticket, ORDER_TIME_DONE_MSC) / 1000); //--- Return close time in seconds
   }

   //--- Retrieve close price for historical order
   static double OrderClosePrice(ulong ticket) {
      return HistoryOrderGetDouble(ticket, ORDER_PRICE_CURRENT); //--- Return close price
   }

   //--- Populate order details from selected order
   static void fetchSelected(Order& order) {
      COrderInfo orderInfo;                               //--- Declare order info object
      order.Ticket = orderInfo.Ticket();                  //--- Set order ticket
      order.Type = orderInfo.OrderType();                 //--- Set order type
      order.State = orderInfo.State();                    //--- Set order state
      order.MagicNumber = orderInfo.Magic();              //--- Set magic number
      order.Lots = orderInfo.VolumeInitial();             //--- Set initial volume
      order.OpenPrice = orderInfo.PriceOpen();            //--- Set open price
      order.StopLoss = orderInfo.StopLoss();              //--- Set Stop Loss
      order.TakeProfit = orderInfo.TakeProfit();          //--- Set Take Profit
      order.Expiration = orderInfo.TimeExpiration();      //--- Set expiration time
      order.Comment = orderInfo.Comment();                //--- Set comment
      order.OpenTime = orderInfo.TimeSetup();             //--- Set open time
      order.CloseTime = OrderCloseTime(order.Ticket); //--- Set close time
      order.SymbolCode = orderInfo.Symbol();              //--- Set symbol code
      order.TradeVolume = orderInfo.VolumeInitial();      //--- Set trade volume
      CalculateAndSetCommision(order);                    //--- Calculate and set commission
   }

   //--- Modify order’s Stop Loss or Take Profit
   static bool modify(ulong ticket, double stopLoss = NULL, double takeProfit = NULL) {
      CTrade trade;                                       //--- Declare trade object
      Order* order = OrderRepository::getByTicket(ticket); //--- Retrieve order by ticket
      double price = order.OpenPrice;                     //--- Set price to open price
      stopLoss = (stopLoss == NULL) ? order.StopLoss : stopLoss; //--- Use existing SL if null
      takeProfit = (takeProfit == NULL) ? order.TakeProfit : takeProfit; //--- Use existing TP if null
      datetime expiration = order.Expiration;             //--- Set expiration
      bool result = false;                                //--- Initialize result as false
      if (order.State == ORDER_STATE_PLACED) {            //--- Check if order is pending
         result = trade.OrderModify(ticket, price, stopLoss, takeProfit, ORDER_TIME_SPECIFIED, expiration, 0); //--- Modify pending order
      } else if (order.State == ORDER_STATE_FILLED) {     //--- Check if order is filled
         result = trade.PositionModify(ticket, stopLoss, takeProfit); //--- Modify position
      }
      if (CheckPointer(order) == POINTER_DYNAMIC) {        //--- Check if order is dynamic
         delete(order);                                   //--- Delete order object
      }
      return result;                                      //--- Return modification result
   }

public:
   //--- Retrieve open and pending orders
   static OrderCollection* GetOpenOrders(int magic = NULL, int type = NULL, string symbolCode = NULL) {
      OrderCollection* orders = new OrderCollection();    //--- Create new order collection
      //--- Process pending orders
      for (int orderIndex = 0; orderIndex < OrdersTotal(); orderIndex++) {
         bool orderSelected = OrderSelect(OrderGetTicket(orderIndex)); //--- Select order by index
         if (orderSelected) {                             //--- Check if selection succeeded
            Order* order = new Order(false);              //--- Create new order object
            OrderRepository::fetchSelected(order);        //--- Populate order details
            if ((magic == NULL || magic == order.MagicNumber) &&
                (type == NULL || type == order.Type) &&
                (symbolCode == NULL || symbolCode == order.SymbolCode)) { //--- Filter by magic, type, symbol
               orders.Add(order);                         //--- Add order to collection
            } else {
               if (CheckPointer(order) == POINTER_DYNAMIC) {
                  delete(order);                          //--- Delete unused order object
               }
            }
         }
      }
      //--- Process open positions (netting system)
      int total = PositionsTotal();                       //--- Get total positions
      for (int i = total - 1; i >= 0; i--) {
         ulong position_ticket = PositionGetTicket(i);    //--- Get position ticket
         string position_symbol = PositionGetString(POSITION_SYMBOL); //--- Get position symbol
         long position_magicNumber = PositionGetInteger(POSITION_MAGIC); //--- Get position magic number
         double volume = PositionGetDouble(POSITION_VOLUME); //--- Get position volume
         double open_price = PositionGetDouble(POSITION_PRICE_OPEN); //--- Get open price
         datetime open_time = (datetime)PositionGetInteger(POSITION_TIME); //--- Get open time
         ENUM_POSITION_TYPE positionType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE); //--- Get position type
         if (position_magicNumber == MagicNumber) {       //--- Check matching magic number
            Order* order = new Order(false);              //--- Create new order object
            order.Ticket = position_ticket;               //--- Set order ticket
            if (positionType == POSITION_TYPE_BUY) {      //--- Check if Buy position
               order.Type = ORDER_TYPE_BUY;               //--- Set Buy type
            } else if (positionType == POSITION_TYPE_SELL) { //--- Check if Sell position
               order.Type = ORDER_TYPE_SELL;              //--- Set Sell type
            }
            order.Lots = volume;                          //--- Set order volume
            order.TradeVolume = volume;                   //--- Set trade volume
            order.OpenPrice = open_price;                 //--- Set open price
            order.OpenTime = open_time;                   //--- Set open time
            order.MagicNumber = position_magicNumber;     //--- Set magic number
            order.SymbolCode = position_symbol;           //--- Set symbol code
            if ((magic == NULL || magic == order.MagicNumber) &&
                (type == NULL || type == order.Type) &&
                (symbolCode == NULL || symbolCode == order.SymbolCode)) { //--- Filter by magic, type, symbol
               orders.Add(order);                         //--- Add order to collection
            } else {
               if (CheckPointer(order) == POINTER_DYNAMIC) {
                  order.Ticket = -1;                      //--- Invalidate ticket
                  delete(order);                          //--- Delete unused order object
               }
            }
         }
      }
      return orders;                                      //--- Return order collection
   }

   //--- Execute Buy order
   static ulong ExecuteOpenBuy(Order* order) {
      ulong orderTicket = ULONG_MAX;                      //--- Initialize ticket as invalid
      MqlTradeRequest request = {};                       //--- Declare trade request
      MqlTradeResult result = {};                         //--- Declare trade result
      request.action = TRADE_ACTION_DEAL;                 //--- Set action to deal
      request.symbol = Symbol();                          //--- Set symbol to current
      request.volume = order.Lots;                        //--- Set volume
      request.type = ORDER_TYPE_BUY;                      //--- Set Buy type
      request.price = Ask_LibFunc();                      //--- Set price to Ask
      request.deviation = MaxDeviationSlippage;           //--- Set maximum slippage
      request.magic = MagicNumber;                        //--- Set magic number
      request.comment = order.Comment;                    //--- Set order comment
      request.type_filling = (ENUM_ORDER_TYPE_FILLING)OrderFillingType; //--- Set filling type
      ResetLastError();                                   //--- Clear last error
      if (OrderSend(request, result)) {                   //--- Send trade request
         if (result.retcode == TRADE_RETCODE_DONE || result.retcode == TRADE_RETCODE_PLACED) { //--- Check success
            orderTicket = result.order;                   //--- Store order ticket
            order.Ticket = orderTicket;                   //--- Update order ticket
            order.IsAwaitingDealExecution = true;         //--- Flag awaiting execution
         } else {
            Print(StringFormat("OrderSend: retcode=%u", result.retcode)); //--- Log return code
         }
      } else {
         Print(StringFormat("OrderSend: error %d: %s", GetLastError(), GetErrorDescription(result.retcode))); //--- Log error
      }
      return orderTicket;                                 //--- Return order ticket
   }

   //--- Execute Sell order
   static ulong ExecuteOpenSell(Order* order) {
      ulong orderTicket = ULONG_MAX;                      //--- Initialize ticket as invalid
      MqlTradeRequest request = {};                       //--- Declare trade request
      MqlTradeResult result = {};                         //--- Declare trade result
      request.action = TRADE_ACTION_DEAL;                 //--- Set action to deal
      request.symbol = Symbol();                          //--- Set symbol to current
      request.volume = order.Lots;                        //--- Set volume
      request.type = ORDER_TYPE_SELL;                     //--- Set Sell type
      request.price = Bid_LibFunc();                      //--- Set price to Bid
      request.deviation = MaxDeviationSlippage;           //--- Set maximum slippage
      request.magic = MagicNumber;                        //--- Set magic number
      request.comment = order.Comment;                    //--- Set order comment
      request.type_filling = (ENUM_ORDER_TYPE_FILLING)OrderFillingType; //--- Set filling type
      ResetLastError();                                   //--- Clear last error
      if (OrderSend(request, result)) {                   //--- Send trade request
         if (result.retcode == TRADE_RETCODE_DONE || result.retcode == TRADE_RETCODE_PLACED) { //--- Check success
            orderTicket = result.order;                   //--- Store order ticket
            order.Ticket = orderTicket;                   //--- Update order ticket
            order.IsAwaitingDealExecution = true;         //--- Flag awaiting execution
         } else {
            Print(StringFormat("OrderSend: retcode=%u", result.retcode)); //--- Log return code
         }
      } else {
         Print(StringFormat("OrderSend: error %d: %s", GetLastError(), GetErrorDescription(result.retcode))); //--- Log error
      }
      return orderTicket;                                 //--- Return order ticket
   }

   //--- Close position (hedging accounts only)
   static bool ClosePosition(Order* order) {
      CPositionInfo m_position;                           //--- Declare position info object
      CTrade m_trade;                                     //--- Declare trade object
      bool foundPosition = false;                         //--- Initialize position found flag
      for (int i = PositionsTotal() - 1; i >= 0; i--) {   //--- Iterate positions
         if (m_position.SelectByIndex(i)) {               //--- Select position by index
            if (m_position.Ticket() == order.Ticket) {    //--- Check matching ticket
               foundPosition = true;                      //--- Set position found
               uint returnCode = 0;                       //--- Initialize return code
               if (m_trade.PositionClosePartial(order.Ticket, NormalizeDouble(order.Lots, 2), MaxDeviationSlippage)) { //--- Attempt partial close
                  returnCode = m_trade.ResultRetcode();   //--- Get return code
                  if (returnCode == TRADE_RETCODE_DONE || returnCode == TRADE_RETCODE_PLACED) { //--- Check success
                     ulong orderTicket = m_trade.ResultOrder(); //--- Get new order ticket
                     order.Ticket = orderTicket;          //--- Update order ticket
                     order.IsAwaitingDealExecution = true; //--- Flag awaiting execution
                     Print(StringFormat("Successfully created a close order (%d) by EA (%d). Awaiting execution.", orderTicket, MagicNumber)); //--- Log success
                     return true;                         //--- Return true
                  }
                  Print(StringFormat("Placing close order failed, Return code: %d", returnCode)); //--- Log failure
               }
            }
         }
      }
      return false;                                       //--- Return false if position not found
   }

   //--- Retrieve recently closed orders
   static OrderCollection* GetLastClosedOrders(datetime startDatetime = NULL) {
      OrderCollection* lastClosedOrders = new OrderCollection(); //--- Create new order collection
      long positionIds[];                                 //--- Store position IDs
      if (HistorySelect(0, TimeCurrent())) {              //--- Select trade history
         for (int i = HistoryDealsTotal() - 1; i >= 0; i--) { //--- Iterate deals
            ulong dealId = HistoryDealGetTicket(i);       //--- Get deal ticket
            long magicNumber = HistoryDealGetInteger(dealId, DEAL_MAGIC); //--- Get deal magic number
            string symbol = HistoryDealGetString(dealId, DEAL_SYMBOL); //--- Get deal symbol
            if ((magicNumber != MagicNumber && magicNumber != 0) || symbol != Symbol()) { //--- Filter by magic and symbol
               continue;                                  //--- Skip non-matching deals
            }
            if (HistoryDealGetInteger(dealId, DEAL_ENTRY) == DEAL_ENTRY_OUT) { //--- Check if deal is close
               datetime closetime = (datetime)HistoryDealGetInteger(dealId, DEAL_TIME); //--- Get close time
               if (startDatetime > closetime) {           //--- Check if before start time
                  break;                                  //--- Exit loop
               }
               long positionId = HistoryDealGetInteger(dealId, DEAL_POSITION_ID); //--- Get position ID
               for (int pi = 0; pi < ArraySize(positionIds); pi++) { //--- Check existing IDs
                  if (positionIds[pi] == positionId) {    //--- Skip duplicates
                     continue;
                  }
               }
               int size = ArraySize(positionIds);         //--- Get current ID array size
               ArrayResize(positionIds, size + 1);        //--- Add new ID
               positionIds[size] = positionId;            //--- Store position ID
            }
         }
      }
      for (int i = 0; i < ArraySize(positionIds); i++) {  //--- Process each position
         if (HistorySelectByPosition(positionIds[i])) {    //--- Select position history
            Order* order = new Order(false);              //--- Create new order object
            double currentOutVolume = 0;                  //--- Track closed volume
            for (int j = 0; j < HistoryDealsTotal(); j++) { //--- Iterate deals
               ulong ticket = HistoryDealGetTicket(j);    //--- Get deal ticket
               if (HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_IN) { //--- Check if open deal
                  datetime openTime = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME); //--- Get open time
                  double openPrice = HistoryDealGetDouble(ticket, DEAL_PRICE); //--- Get open price
                  double lots = HistoryDealGetDouble(ticket, DEAL_VOLUME); //--- Get volume
                  if (order.Ticket == 0) {                //--- Check if first deal
                     order.Ticket = HistoryDealGetInteger(ticket, DEAL_ORDER); //--- Set order ticket
                     long dealType = HistoryDealGetInteger(ticket, DEAL_TYPE); //--- Get deal type
                     if (dealType == ORDER_TYPE_BUY) {    //--- Check if Buy
                        order.Type = ORDER_TYPE_SELL;     //--- Set Sell type (reversed for close)
                     } else if (dealType == ORDER_TYPE_SELL) { //--- Check if Sell
                        order.Type = ORDER_TYPE_BUY;      //--- Set Buy type (reversed for close)
                     } else {
                        Alert("Unknown order.Type in GetLastClosedOrder"); //--- Log unknown type
                     }
                     order.OpenTime = openTime;           //--- Set open time
                     order.OpenPrice = openPrice;         //--- Set open price
                     order.Lots = lots;                   //--- Set volume
                  } else {
                     double averagePrice = ((order.OpenPrice * order.Lots) + (openPrice * lots)) / (order.Lots + lots); //--- Calculate average price
                     order.Lots = order.Lots + lots;      //--- Add volume
                     order.OpenPrice = averagePrice;      //--- Update open price
                  }
               } else if (HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT) { //--- Check if close deal
                  double dealLots = HistoryDealGetDouble(ticket, DEAL_VOLUME); //--- Get close volume
                  double dealClosePrice = HistoryDealGetDouble(ticket, DEAL_PRICE); //--- Get close price
                  if (order.CloseTime == 0) {          //--- Check if first close
                     order.CloseTime = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME); //--- Set close time
                     order.ClosePrice = dealClosePrice; //--- Set close price
                     currentOutVolume = dealLots;       //--- Set initial close volume
                  } else {
                     double averagePrice = ((order.ClosePrice * currentOutVolume) + (dealClosePrice * dealLots)) / (currentOutVolume + dealLots); //--- Calculate average close price
                     order.CloseTime = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME); //--- Update close time
                     order.ClosePrice = averagePrice;   //--- Update close price
                     currentOutVolume += dealLots;      //--- Add close volume
                  }
               }
            }
            lastClosedOrders.Add(order);                 //--- Add order to collection
         }
      }
      return lastClosedOrders;                         //--- Return closed orders collection
   }

   //--- Open order (Buy or Sell)
   static bool OpenOrder(Order* order) {
      double price = NULL;                                //--- Initialize price
      ulong ticketId = -1;                                //--- Initialize ticket
      switch (order.Type) {
      case ORDER_TYPE_BUY:
         ticketId = ExecuteOpenBuy(order);                //--- Execute Buy order
         break;
      case ORDER_TYPE_SELL:
         ticketId = ExecuteOpenSell(order);               //--- Execute Sell order
         break;
      }
      bool success = ticketId != ULONG_MAX;               //--- Check if order was opened
      if (success) {
         Print(StringFormat("Successfully opened an order (%d) by EA (%d)", ticketId, MagicNumber)); //--- Log success
      }
      return success;                                     //--- Return true if successful
   }

   //--- Calculate and set commission for order
   static void CalculateAndSetCommision(Order& order) {
      order.Commission = 0.0;                             //--- Initialize commission
      order.CommissionInPips = 0.0;                       //--- Initialize commission in pips
      order.Commission = 2.0 * CommissionAmountPerTrade + //--- Add roundtrip fixed commission
                         CommissionPercentagePerLot * order.Lots * UnitsOneLot + //--- Add percentage commission
                         CommissionAmountPerLot * order.Lots; //--- Add per-lot commission
      if (order.Lots > 1.0e-5 && order.Commission > 1.0e-5) { //--- Check valid volume and commission
         order.CommissionInPips = order.Commission / (order.Lots * UnitsOneLot * PipPoint); //--- Calculate commission in pips
      }
   }
};
```

We define the "OrderCollection" class to manage a collection of "Order" objects, using the "\_orders" array, "\_pointer" for iteration, and "\_size" to track order count. Its methods include "OrderCollection" for initialization, "Add" to append orders, "Remove" to delete orders at an index, "Get" to retrieve orders, "Count" for size, and iterator methods like "Rewind", "Next", "Prev", "HasNext", "Current", "Key", and "GetKeyByTicket" to navigate and locate orders by ticket.

We also create the "OrderRepository" class to handle broker interactions, with private methods like "getByTicket" to fetch orders using [OrderSelect](https://www.mql5.com/en/docs/trading/orderselect), "OrderCloseTime" and "OrderClosePrice" to retrieve historical order data, "fetchSelected" to populate "Order" details via "COrderInfo", and "modify" to update stop-loss/take-profit using "CTrade". Public methods include "GetOpenOrders" to collect open and pending orders filtered by "MagicNumber", "Type", or "SymbolCode", handling both pending orders and netting positions.

The "ExecuteOpenBuy" and "ExecuteOpenSell" functions send trade requests with [MqlTradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderequest) and [MqlTradeResult](https://www.mql5.com/en/docs/constants/structures/mqltraderesult), setting "ORDER\_TYPE\_BUY" or [ORDER\_TYPE\_SELL](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type), "Ask\_LibFunc" or "Bid\_LibFunc" prices, and "OrderFillingType", while "ClosePosition" closes hedging positions using "CPositionInfo" and "CTrade". The "GetLastClosedOrders" function retrieves recently closed orders by analyzing deal history with [HistorySelect](https://www.mql5.com/en/docs/trading/historyselect), and "CalculateAndSetCommision" computes commissions using "CommissionAmountPerTrade", "CommissionPercentagePerLot", and "CommissionAmountPerLot". Next, we can group the orders and hash them efficiently.

```
//--- Define class for grouping order tickets
class OrderGroupData {
public:
   ulong OrderTicketIds[];                                //--- Store array of order ticket IDs

   //--- Default constructor
   void OrderGroupData() {}                               //--- Initialize empty group

   //--- Copy constructor
   void OrderGroupData(OrderGroupData* ordergroupdata) {} //--- Initialize from existing group (empty)

   //--- Add ticket to group
   void Add(ulong ticketId) {
      int size = ArraySize(OrderTicketIds);               //--- Get current array size
      ArrayResize(OrderTicketIds, size + 1);              //--- Resize array
      OrderTicketIds[size] = ticketId;                    //--- Store ticket at last index
   }

   //--- Remove ticket from group
   void Remove(ulong ticketId) {
      int size = ArraySize(OrderTicketIds);               //--- Get current array size
      int counter = 0;                                    //--- Track new array position
      int counterFound = 0;                               //--- Track removed tickets
      for (int i = 0; i < size; i++) {
         if (OrderTicketIds[i] == ticketId) {             //--- Check matching ticket
            counterFound++;                               //--- Increment found count
            continue;                                     //--- Skip to next
         } else {
            OrderTicketIds[counter] = OrderTicketIds[i];  //--- Shift ticket
            counter++;                                    //--- Increment new position
         }
      }
      if (counterFound > 0) {                             //--- Check if tickets were removed
         ArrayResize(OrderTicketIds, counter);            //--- Resize array to new size
      }
   }

   //--- Destructor
   void ~OrderGroupData() {}                              //--- Clean up group
};

//--- Define class for hash map entry
class OrderGroupHashEntry {
public:
   string _key;                                           //--- Store entry key
   OrderGroupData* _val;                                  //--- Store order group data
   OrderGroupHashEntry* _next;                            //--- Point to next entry

   //--- Default constructor
   OrderGroupHashEntry() {
      _key = NULL;                                        //--- Set key to null
      _val = NULL;                                        //--- Set value to null
      _next = NULL;                                       //--- Set next to null
   }

   //--- Constructor with key and value
   OrderGroupHashEntry(string key, OrderGroupData* val) {
      _key = key;                                         //--- Set key
      _val = val;                                         //--- Set value
      _next = NULL;                                       //--- Set next to null
   }

   //--- Destructor
   ~OrderGroupHashEntry() {
      if (_val != NULL && CheckPointer(_val) == POINTER_DYNAMIC) { //--- Check if value is dynamic
         delete(_val);                                    //--- Delete value
      }
   }
};

//--- Define class for hash map of order groups
class OrderGroupHashMap {
private:
   uint _hashSlots;                                       //--- Store number of hash slots
   int _resizeThreshold;                                  //--- Store resize threshold
   int _hashEntryCount;                                   //--- Track number of entries
   OrderGroupHashEntry* _buckets[];                       //--- Store hash buckets
   bool _adoptValues;                                     //--- Flag value adoption
   uint _foundIndex;                                      //--- Store found index
   OrderGroupHashEntry* _foundEntry;                      //--- Store found entry
   OrderGroupHashEntry* _foundPrev;                       //--- Store previous entry

   //--- Initialize hash map
   void init(uint size, bool adoptValues) {
      _hashSlots = 0;                                     //--- Set initial slots to 0
      _hashEntryCount = 0;                                //--- Set initial entry count to 0
      _adoptValues = adoptValues;                         //--- Set value adoption flag
      rehash(size);                                       //--- Resize hash map
   }

   //--- Calculate hash for key
   uint hash(string s) {
      uchar c[];                                          //--- Declare character array
      uint h = 0;                                         //--- Initialize hash
      if (s != NULL) {                                    //--- Check if key is valid
         h = 5381;                                        //--- Set initial hash value
         int n = StringToCharArray(s, c);                 //--- Convert string to chars
         for (int i = 0; i < n; i++) {
            h = ((h << 5) + h) + c[i];                    //--- Update hash
         }
      }
      return h % _hashSlots;                              //--- Return hash modulo slots
   }

   //--- Find entry by key
   bool find(string keyName) {
      bool found = false;                                 //--- Initialize found flag
      _foundPrev = NULL;                                  //--- Set previous to null
      _foundIndex = hash(keyName);                        //--- Calculate hash index
      if (_foundIndex <= _hashSlots) {                    //--- Check valid index
         for (OrderGroupHashEntry* e = _buckets[_foundIndex]; e != NULL; e = e._next) { //--- Iterate bucket
            if (e._key == keyName) {                      //--- Check key match
               _foundEntry = e;                           //--- Store found entry
               found = true;                              //--- Set found flag
               break;                                     //--- Exit loop
            }
            _foundPrev = e;                               //--- Update previous
         }
      }
      return found;                                       //--- Return found status
   }

   //--- Retrieve number of slots
   uint getSlots() {
      return _hashSlots;                                  //--- Return slot count
   }

   //--- Resize hash map
   bool rehash(uint newSize) {
      bool ret = false;                                   //--- Initialize return flag
      OrderGroupHashEntry* oldTable[];                    //--- Declare old table
      uint oldSize = _hashSlots;                          //--- Store current size
      if (newSize <= getSlots()) {                        //--- Check if resize is needed
         ret = false;                                     //--- Set failure
      } else if (ArrayResize(_buckets, newSize) != newSize) { //--- Resize buckets
         ret = false;                                     //--- Set failure
      } else if (ArrayResize(oldTable, oldSize) != oldSize) { //--- Resize old table
         ret = false;                                     //--- Set failure
      } else {
         uint i = 0;                                      //--- Initialize index
         for (i = 0; i < oldSize; i++) {                  //--- Copy buckets
            oldTable[i] = _buckets[i];                    //--- Store old bucket
         }
         for (i = 0; i < newSize; i++) {                  //--- Clear new buckets
            _buckets[i] = NULL;                           //--- Set to null
         }
         _hashSlots = newSize;                            //--- Update slot count
         _resizeThreshold = (int)_hashSlots / 4 * 3;      //--- Set resize threshold
         for (uint oldHashCode = 0; oldHashCode < oldSize; oldHashCode++) { //--- Rehash entries
            OrderGroupHashEntry* next = NULL;             //--- Initialize next
            for (OrderGroupHashEntry* e = oldTable[oldHashCode]; e != NULL; e = next) { //--- Iterate old bucket
               next = e._next;                            //--- Store next entry
               uint newHashCode = hash(e._key);           //--- Calculate new hash
               e._next = _buckets[newHashCode];           //--- Link to new bucket
               _buckets[newHashCode] = e;                 //--- Store in new bucket
            }
            oldTable[oldHashCode] = NULL;                 //--- Clear old bucket
         }
         ret = true;                                      //--- Set success
      }
      return ret;                                         //--- Return resize result
   }

public:
   //--- Default constructor
   OrderGroupHashMap() {
      init(13, false);                                    //--- Initialize with 13 slots, no adoption
   }

   //--- Constructor with adoption flag
   OrderGroupHashMap(bool adoptValues) {
      init(13, adoptValues);                              //--- Initialize with 13 slots
   }

   //--- Constructor with size
   OrderGroupHashMap(int size) {
      init(size, false);                                  //--- Initialize with specified size, no adoption
   }

   //--- Constructor with size and adoption
   OrderGroupHashMap(int size, bool adoptValues) {
      init(size, adoptValues);                            //--- Initialize with size and adoption
   }

   //--- Destructor
   ~OrderGroupHashMap() {
      for (uint i = 0; i < _hashSlots; i++) {            //--- Iterate buckets
         OrderGroupHashEntry* nextEntry = NULL;           //--- Initialize next
         for (OrderGroupHashEntry* entry = _buckets[i]; entry != NULL; entry = nextEntry) { //--- Iterate entries
            nextEntry = entry._next;                      //--- Store next entry
            if (_adoptValues && entry._val != NULL && CheckPointer(entry._val) == POINTER_DYNAMIC) { //--- Check if value is dynamic
               delete entry._val;                         //--- Delete value
            }
            delete entry;                                 //--- Delete entry
         }
         _buckets[i] = NULL;                              //--- Clear bucket
      }
   }

   //--- Check if key exists
   bool ContainsKey(string keyName) {
      return find(keyName);                               //--- Return true if key found
   }

   //--- Retrieve group data by key
   OrderGroupData* Get(string keyName) {
      OrderGroupData* obj = NULL;                         //--- Initialize return object
      if (find(keyName)) {                                //--- Check if key exists
         obj = _foundEntry._val;                          //--- Set return object
      }
      return obj;                                         //--- Return group data or null
   }

   //--- Retrieve all group data
   void GetAllData(OrderGroupData* &data[]) {
      for (uint i = 0; i < _hashSlots; i++) {            //--- Iterate buckets
         OrderGroupHashEntry* nextEntry = NULL;           //--- Initialize next
         for (OrderGroupHashEntry* entry = _buckets[i]; entry != NULL; entry = nextEntry) { //--- Iterate entries
            if (entry._val != NULL) {                     //--- Check valid value
               int size = ArraySize(data);                //--- Get current array size
               ArrayResize(data, size + 1);               //--- Resize array
               data[size] = entry._val;                   //--- Store value
               nextEntry = entry._next;                   //--- Move to next
            }
         }
      }
   }

   //--- Store or update group data
   OrderGroupData* Put(string keyName, OrderGroupData* obj) {
      OrderGroupData* ret = NULL;                         //--- Initialize return value
      if (find(keyName)) {                                //--- Check if key exists
         ret = _foundEntry._val;                          //--- Store existing value
         if (_adoptValues && _foundEntry._val != NULL && CheckPointer(_foundEntry._val) == POINTER_DYNAMIC) { //--- Check if value is dynamic
            delete _foundEntry._val;                      //--- Delete existing value
         }
         _foundEntry._val = obj;                          //--- Update value
      } else {
         OrderGroupHashEntry* e = new OrderGroupHashEntry(keyName, obj); //--- Create new entry
         OrderGroupHashEntry* first = _buckets[_foundIndex]; //--- Get current bucket head
         e._next = first;                                 //--- Link new entry
         _buckets[_foundIndex] = e;                       //--- Store new entry
         _hashEntryCount++;                               //--- Increment entry count
         if (_hashEntryCount > _resizeThreshold) {        //--- Check if resize needed
            rehash(_hashSlots / 2 * 3);                   //--- Resize hash map
         }
      }
      return ret;                                         //--- Return previous value or null
   }

   //--- Delete entry by key
   bool Delete(string keyName) {
      bool found = false;                                 //--- Initialize found flag
      if (find(keyName)) {                                //--- Check if key exists
         OrderGroupHashEntry* next = _foundEntry._next;   //--- Store next entry
         if (_foundPrev != NULL) {                        //--- Check if previous exists
            _foundPrev._next = next;                      //--- Update previous link
         } else {
            _buckets[_foundIndex] = next;                 //--- Update bucket head
         }
         if (_adoptValues && _foundEntry._val != NULL && CheckPointer(_foundEntry._val) == POINTER_DYNAMIC) { //--- Check if value is dynamic
            delete _foundEntry._val;                      //--- Delete value
         }
         delete _foundEntry;                              //--- Delete entry
         _hashEntryCount--;                               //--- Decrement entry count
         found = true;                                    //--- Set found flag
      }
      return found;                                       //--- Return true if deleted
   }

   //--- Delete multiple keys
   int DeleteKeys(const string& keys[]) {
      int count = 0;                                      //--- Initialize delete count
      for (int i = 0; i < ArraySize(keys); i++) {         //--- Iterate keys
         if (Delete(keys[i])) {                           //--- Attempt to delete key
            count++;                                      //--- Increment count
         }
      }
      return count;                                       //--- Return number of deleted keys
   }

   //--- Delete all keys except specified
   int DeleteKeysExcept(const string& keys[]) {
      int index = 0, count = 0;                           //--- Initialize index and count
      string hashedKeys[];                                //--- Declare hashed keys array
      ArrayResize(hashedKeys, _hashEntryCount);           //--- Resize to entry count
      for (uint i = 0; i < _hashSlots; i++) {            //--- Iterate buckets
         OrderGroupHashEntry* nextEntry = NULL;           //--- Initialize next
         for (OrderGroupHashEntry* entry = _buckets[i]; entry != NULL; entry = nextEntry) { //--- Iterate entries
            nextEntry = entry._next;                      //--- Store next
            if (entry._key != NULL) {                     //--- Check valid key
               hashedKeys[index] = entry._key;            //--- Store key
               index++;                                   //--- Increment index
            }
         }
      }
      for (int i = 0; i < ArraySize(hashedKeys); i++) {   //--- Iterate hashed keys
         bool keep = false;                               //--- Initialize keep flag
         for (int j = 0; j < ArraySize(keys); j++) {      //--- Check against keep keys
            if (hashedKeys[i] == keys[j]) {               //--- Check match
               keep = true;                               //--- Set keep flag
               break;                                     //--- Exit loop
            }
         }
         if (!keep) {                                     //--- Check if key should be deleted
            if (Delete(hashedKeys[i])) {                  //--- Attempt to delete
               count++;                                   //--- Increment count
            }
         }
      }
      return count;                                       //--- Return number of deleted keys
   }
};
```

To manage and group order tickets efficiently, supporting organized trade handling, we define the "OrderGroupData" class to store order ticket IDs in the "OrderTicketIds" array, with methods like "OrderGroupData" for initialization, "Add" to append ticket IDs, "Remove" to delete specific tickets by shifting remaining entries, and a destructor "~OrderGroupData" for cleanup, ensuring dynamic ticket management.

Next, we create the "OrderGroupHashEntry" class to represent entries in a hash map, containing "\_key" for the entry identifier, "\_val" to hold an "OrderGroupData" object, and "\_next" for linking entries in case of collisions. Its constructors "OrderGroupHashEntry" initialize entries, and the destructor "~OrderGroupHashEntry" frees dynamic "OrderGroupData" objects if needed.

We then implement the "OrderGroupHashMap" class to manage order groups using a hash table, with private variables "\_hashSlots" for bucket count, "\_resizeThreshold" for resizing triggers, "\_hashEntryCount" for tracking entries, and "\_buckets" to store "OrderGroupHashEntry" arrays. The private "init" method sets up the hash map, "hash" computes a hash value for keys, "find" locates entries by key, and "rehash" resizes the table.

Public methods include constructors "OrderGroupHashMap" for various initialization options, "ContainsKey" to check key existence, "Get" to retrieve "OrderGroupData", "GetAllData" to collect all groups, "Put" to add or update entries, "Delete" to remove a key, "DeleteKeys" for multiple deletions, and "DeleteKeysExcept" to remove all but specified keys. The destructor "~OrderGroupHashMap" ensures proper cleanup. We now need to manage trading states and we need an extra class for that operation.

```
//--- Declare global array to track recent order results
int LastOrderResults[];                                   //--- Store outcomes of recent trades (1 for profit, 0 for loss)

//--- Define class for managing trading state and orders
class Wallet {
private:
   int _openedBuyOrderCount;                              //--- Track number of open Buy orders
   int _openedSellOrderCount;                             //--- Track number of open Sell orders
   ulong _closedOrderCount;                               //--- Track total closed orders
   int _lastOrderResultSize;                              //--- Store size of LastOrderResults array
   ENUM_TIMEFRAMES _lastOrderResultByTimeframe;           //--- Store timeframe for tracking closed orders
   datetime _lastBarStartTime;                            //--- Store start time of last bar
   OrderCollection* _openOrders;                          //--- Store currently open orders
   OrderGroupHashMap* _openOrdersSymbolType;              //--- Store open orders grouped by symbol and type
   OrderGroupHashMap* _openOrdersSymbol;                  //--- Store open orders grouped by symbol
   OrderCollection* _pendingOpenOrders;                   //--- Store pending open orders
   OrderCollection* _pendingCloseOrders;                  //--- Store pending close orders
   Order* _mostRecentOpenOrder;                           //--- Store most recently opened order
   Order* _mostRecentClosedOrder;                         //--- Store most recently closed order
   OrderCollection* _recentClosedOrders;                  //--- Store recently closed orders

public:
   //--- Initialize wallet
   void Wallet() {
      _openedBuyOrderCount = 0;                           //--- Set Buy order count to 0
      _openedSellOrderCount = 0;                          //--- Set Sell order count to 0
      _closedOrderCount = 0;                              //--- Set closed order count to 0
      _lastOrderResultSize = 0;                           //--- Set result size to 0
      _lastOrderResultByTimeframe = NULL;                 //--- Set timeframe to null
      _lastBarStartTime = NULL;                           //--- Set bar start time to null
      _pendingOpenOrders = new OrderCollection();         //--- Create pending open orders collection
      _pendingCloseOrders = new OrderCollection();        //--- Create pending close orders collection
      _recentClosedOrders = new OrderCollection();        //--- Create recent closed orders collection
      _openOrdersSymbolType = NULL;                       //--- Set symbol-type group to null
      _openOrdersSymbol = NULL;                           //--- Set symbol group to null
      _openOrders = new OrderCollection();                //--- Create open orders collection
      _mostRecentOpenOrder = NULL;                        //--- Set recent open order to null
      _mostRecentClosedOrder = NULL;                      //--- Set recent closed order to null
   }

   //--- Destructor to clean up wallet
   void ~Wallet() {
      delete(_pendingOpenOrders);                         //--- Delete pending open orders
      delete(_pendingCloseOrders);                        //--- Delete pending close orders
      delete(_recentClosedOrders);                        //--- Delete recent closed orders
      if (_openOrders != NULL) {                          //--- Check if open orders exist
         delete(_openOrders);                             //--- Delete open orders
      }
      if (_mostRecentOpenOrder != NULL) {                 //--- Check if recent open order exists
         delete(_mostRecentOpenOrder);                    //--- Delete recent open order
      }
      if (_mostRecentClosedOrder != NULL) {               //--- Check if recent closed order exists
         delete(_mostRecentClosedOrder);                  //--- Delete recent closed order
      }
      if (_openOrdersSymbolType != NULL) {                //--- Check if symbol-type group exists
         delete(_openOrdersSymbolType);                   //--- Delete symbol-type group
      }
      if (_openOrdersSymbol != NULL) {                    //--- Check if symbol group exists
         delete(_openOrdersSymbol);                       //--- Delete symbol group
      }
   }

   //--- Handle new tick event
   void HandleTick() {
      if (_lastOrderResultByTimeframe != NULL) {          //--- Check if timeframe is set
         datetime newBarStartTime = iTime(_Symbol, _lastOrderResultByTimeframe, 0); //--- Get current bar start
         if (_lastBarStartTime == newBarStartTime) {      //--- Check if same bar
            return;                                       //--- Exit if no new bar
         } else {
            _lastBarStartTime = newBarStartTime;          //--- Update bar start time
            for (int i = 0; i < _recentClosedOrders.Count(); i++) { //--- Iterate closed orders
               Order* order = _recentClosedOrders.Get(i); //--- Get closed order
               if (CheckPointer(order) != POINTER_INVALID && CheckPointer(order) == POINTER_DYNAMIC) { //--- Check dynamic pointer
                  delete(order);                             //--- Delete order
               }
               _recentClosedOrders.Remove(i);             //--- Remove order
            }
            PrintOrderChanges();                          //--- Log order changes
         }
      }
   }

   //--- Set size of order results array
   void SetLastOrderResultsSize(int size) {
      if (size > _lastOrderResultSize) {                  //--- Check if size increased
         ArrayResize(LastOrderResults, size);             //--- Resize results array
         ArrayInitialize(LastOrderResults, 1);            //--- Initialize with 1 (assume profit)
         _lastOrderResultSize = size;                     //--- Update result size
      }
   }

   //--- Set timeframe for tracking closed orders
   void SetLastClosedOrdersByTimeframe(ENUM_TIMEFRAMES timeframe) {
      if (_lastOrderResultByTimeframe != NULL && timeframe <= _lastOrderResultByTimeframe) { //--- Check if timeframe is valid
         return;                                          //--- Exit if no change needed
      }
      _lastOrderResultByTimeframe = timeframe;            //--- Set new timeframe
      _lastBarStartTime = iTime(_Symbol, _lastOrderResultByTimeframe, 0); //--- Set bar start time
   }

   //--- Retrieve recent closed orders
   OrderCollection* GetRecentClosedOrders() {
      return _recentClosedOrders;                         //--- Return closed orders collection
   }

   //--- Activate order grouping types
   void ActivateOrderGroups(ORDER_GROUP_TYPE &groupTypes[]) {
      for (int i = 0; i < ArrayRange(groupTypes, 0); i++) { //--- Iterate group types
         if (groupTypes[i] == SymbolOrderType && _openOrdersSymbolType == NULL) { //--- Check symbol-type grouping
            _openOrdersSymbolType = new OrderGroupHashMap(); //--- Create symbol-type hash map
         } else if (groupTypes[i] == SymbolCode && _openOrdersSymbol == NULL) { //--- Check symbol grouping
            _openOrdersSymbol = new OrderGroupHashMap();  //--- Create symbol hash map
         }
      }
   }

   //--- Retrieve open orders
   OrderCollection* GetOpenOrders() {
      if (_openOrders == NULL) {                          //--- Check if orders are loaded
         LoadOrdersFromBroker();                          //--- Load orders from broker
      }
      return _openOrders;                                 //--- Return open orders collection
   }

   //--- Retrieve open order by ticket
   Order* GetOpenOrder(ulong ticketId) {
      int index = _openOrders.GetKeyByTicket(ticketId);   //--- Find order index by ticket
      if (index == -1) {                                  //--- Check if not found
         return NULL;                                     //--- Return null
      }
      return _openOrders.Get(index);                      //--- Return order at index
   }

   //--- Retrieve grouped orders by symbol and type
   void GetOpenOrdersSymbolOrderType(OrderGroupData* &data[]) {
      _openOrdersSymbolType.GetAllData(data);             //--- Populate data with grouped orders
   }

   //--- Retrieve grouped orders by symbol
   void GetOpenOrdersSymbol(OrderGroupData* &data[]) {
      _openOrdersSymbol.GetAllData(data);                 //--- Populate data with grouped orders
   }

   //--- Retrieve pending open orders
   OrderCollection* GetPendingOpenOrders() {
      return _pendingOpenOrders;                          //--- Return pending open orders
   }

   //--- Retrieve pending close orders
   OrderCollection* GetPendingCloseOrders() {
      return _pendingCloseOrders;                         //--- Return pending close orders
   }

   //--- Reset pending orders
   void ResetPendingOrders() {
      delete(_pendingOpenOrders);                         //--- Delete existing pending open orders
      delete(_pendingCloseOrders);                        //--- Delete existing pending close orders
      _pendingOpenOrders = new OrderCollection();         //--- Create new pending open orders
      _pendingCloseOrders = new OrderCollection();        //--- Create new pending close orders
      Print("Wallet has " + IntegerToString(_pendingOpenOrders.Count()) + " pending open orders now."); //--- Log open orders count
      Print("Wallet has " + IntegerToString(_pendingCloseOrders.Count()) + " pending close orders now."); //--- Log close orders count
   }

   //--- Check if orders are being opened
   bool AreOrdersBeingOpened() {
      for (int i = _pendingOpenOrders.Count() - 1; i >= 0; i--) { //--- Iterate pending open orders
         if (_pendingOpenOrders.Get(i).IsAwaitingDealExecution) { //--- Check execution status
            return true;                                     //--- Return true if awaiting execution
         }
      }
      return false;                                       //--- Return false if no orders pending
   }

   //--- Check if orders are being closed
   bool AreOrdersBeingClosed() {
      for (int i = _pendingCloseOrders.Count() - 1; i >= 0; i--) { //--- Iterate pending close orders
         if (_pendingCloseOrders.Get(i).IsAwaitingDealExecution) { //--- Check execution status
            return true;                                     //--- Return true if awaiting execution
         }
      }
      return false;                                       //--- Return false if no orders pending
   }

   //--- Reset open orders
   void ResetOpenOrders() {
      _openedBuyOrderCount = 0;                           //--- Reset Buy order count
      _openedSellOrderCount = 0;                          //--- Reset Sell order count
      if (_openOrders != NULL) {                          //--- Check if open orders exist
         delete(_openOrders);                             //--- Delete open orders
         _openOrders = new OrderCollection();             //--- Create new open orders
      }
      if (_openOrdersSymbol != NULL) {                    //--- Check if symbol group exists
         delete(_openOrdersSymbol);                       //--- Delete symbol group
         _openOrdersSymbol = new OrderGroupHashMap();     //--- Create new symbol group
      }
      if (_openOrdersSymbolType != NULL) {                //--- Check if symbol-type group exists
         delete(_openOrdersSymbolType);                   //--- Delete symbol-type group
         _openOrdersSymbolType = new OrderGroupHashMap(); //--- Create new symbol-type group
      }
   }

   //--- Retrieve most recent open order
   Order* GetMostRecentOpenOrder() {
      return _mostRecentOpenOrder;                        //--- Return recent open order
   }

   //--- Retrieve most recent closed order
   Order* GetMostRecentClosedOrder() {
      return _mostRecentClosedOrder;                      //--- Return recent closed order
   }

   //--- Load orders from broker
   void LoadOrdersFromBroker() {
      OrderCollection* brokerOrders = OrderRepository::GetOpenOrders(MagicNumber, NULL, Symbol()); //--- Retrieve open orders
      for (int i = 0; i < brokerOrders.Count(); i++) {    //--- Iterate broker orders
         Order* openOrder = brokerOrders.Get(i);          //--- Get open order
         AddOrderToOpenOrderCollections(openOrder);       //--- Add to collections
         SetMostRecentOpenOrClosedOrder(openOrder);       //--- Update recent order
         CountAddedOrder(openOrder);                      //--- Update order counts
      }
      OrderCollection* lastClosedOrders = OrderRepository::GetLastClosedOrders(_lastBarStartTime); //--- Retrieve closed orders
      for (int i = 0; i < lastClosedOrders.Count(); i++) { //--- Iterate closed orders
         Order* closedOrder = lastClosedOrders.Get(i);    //--- Get closed order
         _recentClosedOrders.Add(new Order(closedOrder, true)); //--- Add to recent closed orders
         SetMostRecentOpenOrClosedOrder(closedOrder);     //--- Update recent order
      }
      delete(lastClosedOrders);                           //--- Delete closed orders collection
      delete(brokerOrders);                               //--- Delete broker orders collection
      PrintOrderChanges();                                //--- Log order changes
      Print("Wallet has " + IntegerToString(GetOpenedOrderCount()) + " orders now."); //--- Log total open orders
   }

   //--- Move pending open order to open status
   void SetPendingOpenOrderToOpen(Order* justOpenedOrder) {
      bool success = false;                               //--- Initialize success flag
      int key = _pendingOpenOrders.GetKeyByTicket(justOpenedOrder.Ticket); //--- Find order by ticket
      if (key != -1) {                                    //--- Check if order found
         if (_mostRecentOpenOrder != NULL && CheckPointer(_mostRecentOpenOrder) != POINTER_INVALID && CheckPointer(_mostRecentOpenOrder) == POINTER_DYNAMIC) { //--- Check existing recent order
            delete(_mostRecentOpenOrder);                 //--- Delete recent open order
         }
         _mostRecentOpenOrder = new Order(justOpenedOrder, false); //--- Set new recent open order
         AddOrderToOpenOrderCollections(justOpenedOrder); //--- Add to collections
         CountAddedOrder(justOpenedOrder);                //--- Update order counts
         delete(justOpenedOrder);                         //--- Delete input order
         _pendingOpenOrders.Remove(key);                  //--- Remove from pending
         success = true;                                  //--- Set success flag
      }
      if (success) {                                      //--- Check if successful
         PrintOrderChanges();                             //--- Log order changes
      } else {
         Alert("Couldn't move pending open order to opened orders for ticketid: " + IntegerToString(justOpenedOrder.Ticket)); //--- Log failure
      }
   }

   //--- Cancel pending open order
   bool CancelPendingOpenOrder(Order* justOpenedOrder) {
      int key = _pendingOpenOrders.GetKeyByTicket(justOpenedOrder.Ticket); //--- Find order by ticket
      if (key != -1) {                                    //--- Check if order found
         delete(justOpenedOrder);                         //--- Delete order
         _pendingOpenOrders.Remove(key);                  //--- Remove from pending
      } else {
         Alert("Couldn't cancel pending open order for ticketid: " + IntegerToString(justOpenedOrder.Ticket)); //--- Log failure
      }
      PrintOrderChanges();                                //--- Log order changes
      return key != -1;                                   //--- Return true if canceled
   }

   //--- Move all open orders to pending close
   void SetAllOpenOrdersToPendingClose() {
      bool success = false;                               //--- Initialize success flag
      for (int i = _openOrders.Count() - 1; i >= 0; i--) { //--- Iterate open orders
         Order* order = _openOrders.Get(i);               //--- Get open order
         if (MoveOpenOrderToPendingCloseOrders(order)) {  //--- Move to pending close
            success = true;                               //--- Set success flag
         }
      }
      if (success) {                                      //--- Check if changes made
         PrintOrderChanges();                             //--- Log order changes
      }
   }

   //--- Move single open order to pending close
   bool SetOpenOrderToPendingClose(Order* orderToClose) {
      bool success = MoveOpenOrderToPendingCloseOrders(orderToClose); //--- Move to pending close
      if (success) {                                      //--- Check if successful
         PrintOrderChanges();                             //--- Log order changes
         return true;                                     //--- Return true
      }
      Alert("Couldn't move open order to pendingclose orders for ticketid: " + IntegerToString(orderToClose.Ticket)); //--- Log failure
      return false;                                       //--- Return false
   }

   //--- Add order to pending close
   bool AddPendingCloseOrder(Order* orderToClose) {
      _pendingCloseOrders.Add(new Order(orderToClose, false)); //--- Add new order to pending close
      if (CheckPointer(orderToClose) != POINTER_INVALID && CheckPointer(orderToClose) == POINTER_DYNAMIC) { //--- Check dynamic pointer
         delete(orderToClose);                            //--- Delete input order
      }
      PrintOrderChanges();                                //--- Log order changes
      return true;                                        //--- Return true
   }

   //--- Move pending close order to closed status
   bool SetPendingCloseOrderToClosed(Order* justClosedOrder) {
      int key = _pendingCloseOrders.GetKeyByTicket(justClosedOrder.Ticket); //--- Find order by ticket
      if (key != -1) {                                    //--- Check if order found
         if (_lastOrderResultSize > 0) {                  //--- Check if results tracking enabled
            for (int i = ArraySize(LastOrderResults) - 1; i > 0; i--) { //--- Shift results
               LastOrderResults[i] = LastOrderResults[i - 1]; //--- Move previous result
            }
            LastOrderResults[0] = justClosedOrder.CalculateProfitPips() > 0 ? 1 : 0; //--- Set result (1 for profit, 0 for loss)
         }
         if (_mostRecentClosedOrder != NULL && CheckPointer(_mostRecentClosedOrder) != POINTER_INVALID && CheckPointer(_mostRecentClosedOrder) == POINTER_DYNAMIC) { //--- Check existing recent closed order
            delete(_mostRecentClosedOrder);               //--- Delete recent closed order
         }
         _mostRecentClosedOrder = new Order(justClosedOrder, false); //--- Set new recent closed order
         _recentClosedOrders.Add(new Order(justClosedOrder, true)); //--- Add to recent closed orders
         _pendingCloseOrders.Remove(key);                 //--- Remove from pending close
         delete(justClosedOrder);                         //--- Delete input order
         _closedOrderCount++;                             //--- Increment closed count
         PrintOrderChanges();                             //--- Log order changes
         return true;                                     //--- Return true
      }
      Alert("Couldn't move open order to removed order for ticketid: " + IntegerToString(justClosedOrder.Ticket)); //--- Log failure
      return false;                                       //--- Return false
   }

   //--- Retrieve total open order count
   int GetOpenedOrderCount() {
      return _openedBuyOrderCount + _openedSellOrderCount; //--- Return sum of Buy and Sell orders
   }

   //--- Retrieve closed order count
   ulong GetClosedOrderCount() {
      return _closedOrderCount;                           //--- Return closed order count
   }

private:
   //--- Add order to open order collections
   void AddOrderToOpenOrderCollections(Order* order) {
      Order* newOpenOrder = new Order(order, true);       //--- Create new order with visibility
      _openOrders.Add(newOpenOrder);                      //--- Add to open orders
      if (IsSymbolOrderTypeOrderGroupActivated()) {       //--- Check symbol-type grouping
         string key = GetOrderGroupSymbolOrderTypeKey(order); //--- Get symbol-type key
         OrderGroupData* orderGroupData = _openOrdersSymbolType.Get(key); //--- Retrieve group data
         if (orderGroupData == NULL) {                    //--- Check if group exists
            orderGroupData = new OrderGroupData();        //--- Create new group
         }
         orderGroupData.Add(newOpenOrder.Ticket);         //--- Add order ticket
         _openOrdersSymbolType.Put(key, orderGroupData);  //--- Store group data
      }
      if (IsSymbolOrderGroupActivated()) {                //--- Check symbol grouping
         string key = GetOrderGroupSymbolKey(order);      //--- Get symbol key
         OrderGroupData* orderGroupData = _openOrdersSymbol.Get(key); //--- Retrieve group data
         if (orderGroupData == NULL) {                    //--- Check if group exists
            orderGroupData = new OrderGroupData();        //--- Create new group
         }
         orderGroupData.Add(newOpenOrder.Ticket);         //--- Add order ticket
         _openOrdersSymbol.Put(key, orderGroupData);      //--- Store group data
      }
      PrintOrderChanges();                                //--- Log order changes
   }

   //--- Remove order from open order collections
   bool RemoveOrderFromOpenOrderCollections(Order* order) {
      int key = GetOpenOrders().GetKeyByTicket(order.Ticket); //--- Find order by ticket
      if (key != -1) {                                    //--- Check if order found
         GetOpenOrders().Remove(key);                     //--- Remove from open orders
         if (_openOrdersSymbolType != NULL) {             //--- Check symbol-type grouping
            string symbolOrderTypeKey = GetOrderGroupSymbolOrderTypeKey(order); //--- Get symbol-type key
            OrderGroupData* symbolOrderTypeGroupData = _openOrdersSymbolType.Get(symbolOrderTypeKey); //--- Retrieve group
            symbolOrderTypeGroupData.Remove(order.Ticket); //--- Remove ticket
         }
         if (_openOrdersSymbol != NULL) {                 //--- Check symbol grouping
            string symbolKey = GetOrderGroupSymbolKey(order); //--- Get symbol key
            OrderGroupData* symbolGroupData = _openOrdersSymbol.Get(symbolKey); //--- Retrieve group
            symbolGroupData.Remove(order.Ticket);         //--- Remove ticket
         }
      }
      return key != -1;                                   //--- Return true if removed
   }

   //--- Update most recent open or closed order
   void SetMostRecentOpenOrClosedOrder(Order* order) {
      if (order.CloseTime == 0) {                         //--- Check if open order
         if (_mostRecentOpenOrder == NULL) {              //--- Check if no recent open order
            _mostRecentOpenOrder = new Order(order, false); //--- Set new recent open order
         } else if (_mostRecentOpenOrder.OpenTime < order.OpenTime) { //--- Check if newer
            delete(_mostRecentOpenOrder);                 //--- Delete existing
            _mostRecentOpenOrder = new Order(order, false); //--- Set new recent open order
         }
      } else {                                            //--- Handle closed order
         if (_mostRecentClosedOrder == NULL) {            //--- Check if no recent closed order
            _mostRecentClosedOrder = new Order(order, false); //--- Set new recent closed order
         } else if (_mostRecentClosedOrder.CloseTime < order.CloseTime) { //--- Check if newer
            delete(_mostRecentClosedOrder);               //--- Delete existing
            _mostRecentClosedOrder = new Order(order, false); //--- Set new recent closed order
         }
      }
   }

   //--- Move open order to pending close
   bool MoveOpenOrderToPendingCloseOrders(Order* orderToClose) {
      if (RemoveOrderFromOpenOrderCollections(orderToClose)) { //--- Remove from open collections
         CountRemovedOrder(orderToClose);                 //--- Update order counts
         _pendingCloseOrders.Add(new Order(orderToClose, false)); //--- Add to pending close
         if (orderToClose.OpenTime == _mostRecentOpenOrder.OpenTime) { //--- Check if recent open order
            if (CheckPointer(_mostRecentOpenOrder) != POINTER_INVALID && CheckPointer(_mostRecentOpenOrder) == POINTER_DYNAMIC) { //--- Check dynamic pointer
               delete(_mostRecentOpenOrder);              //--- Delete recent open order
            }
            _mostRecentOpenOrder = NULL;                  //--- Clear recent open order
            Order* newMostRecentOpenOrder = GetLastOpenOrder(); //--- Get new recent open order
            if (newMostRecentOpenOrder != NULL) {         //--- Check if new order exists
               SetMostRecentOpenOrClosedOrder(newMostRecentOpenOrder); //--- Update recent order
            }
         }
         if (CheckPointer(orderToClose) != POINTER_INVALID && CheckPointer(orderToClose) == POINTER_DYNAMIC) { //--- Check dynamic pointer
            delete(orderToClose);                         //--- Delete input order
         }
         return true;                                     //--- Return true
      }
      return false;                                       //--- Return false if failed
   }

   //--- Retrieve last open order
   Order* GetLastOpenOrder() {
      Order* order = NULL;                                //--- Initialize order
      for (int i = _openOrders.Count() - 1; i >= 0; i--) { //--- Iterate open orders
         return _openOrders.Get(i);                       //--- Return last order
      }
      return NULL;                                        //--- Return null if none
   }

   //--- Generate key for symbol and order type
   string GetOrderGroupSymbolOrderTypeKey(Order* order) {
      return order.SymbolCode + IntegerToString(order.Type); //--- Combine symbol and type
   }

   //--- Generate key for symbol
   string GetOrderGroupSymbolKey(Order* order) {
      return order.SymbolCode;                            //--- Return symbol code
   }

   //--- Check if symbol-type grouping is active
   bool IsSymbolOrderTypeOrderGroupActivated() {
      return _openOrdersSymbolType != NULL;               //--- Return true if active
   }

   //--- Check if symbol grouping is active
   bool IsSymbolOrderGroupActivated() {
      return _openOrdersSymbol != NULL;                   //--- Return true if active
   }

   //--- Increment order count for added order
   void CountAddedOrder(Order* order) {
      if (order.Type == ORDER_TYPE_BUY) {                 //--- Check if Buy order
         _openedBuyOrderCount++;                          //--- Increment Buy count
      } else if (order.Type == ORDER_TYPE_SELL) {         //--- Check if Sell order
         _openedSellOrderCount++;                         //--- Increment Sell count
      }
   }

   //--- Decrement order count for removed order
   void CountRemovedOrder(Order* order) {
      if (order.Type == ORDER_TYPE_BUY) {                 //--- Check if Buy order
         _openedBuyOrderCount--;                          //--- Decrement Buy count
      } else if (order.Type == ORDER_TYPE_SELL) {         //--- Check if Sell order
         _openedSellOrderCount--;                         //--- Decrement Sell count
      }
   }

   //--- Log order state changes
   void PrintOrderChanges() {
      if (DisplayOrderInfo && IsDemoLiveOrVisualMode) {   //--- Check if display enabled
         string comment = "\n     ------------------------------------------------------------"; //--- Start comment
         comment += "\n      :: Pending open orders: " + IntegerToString(_pendingOpenOrders.Count()); //--- Add pending open count
         comment += "\n      :: Open orders: " + IntegerToString(_openedBuyOrderCount) + " (Buy), " + IntegerToString(_openedSellOrderCount) + " (Sell)"; //--- Add open counts
         comment += "\n      :: Pending close orders: " + IntegerToString(_pendingCloseOrders.Count()); //--- Add pending close count
         comment += "\n      :: Recently closed orders: " + IntegerToString(_recentClosedOrders.Count()); //--- Add closed count
         OrderInfoComment = comment;                      //--- Store comment
      }
   }
};
```

Here, we implement the "Wallet" class to manage the trading state and orders. We declare the global "LastOrderResults" array to store trade outcomes (1 for profit, 0 for loss). The "Wallet" [class](https://www.mql5.com/en/docs/basis/types/classes) uses private variables like "\_openedBuyOrderCount", "\_openedSellOrderCount", "\_closedOrderCount", "\_lastOrderResultSize", "\_lastOrderResultByTimeframe" ( [ENUM\_TIMEFRAMES](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes)), and "\_lastBarStartTime" to track order counts and timing, alongside pointers to "OrderCollection" objects ("\_openOrders", "\_pendingOpenOrders", "\_pendingCloseOrders", "\_recentClosedOrders") and "OrderGroupHashMap" objects ("\_openOrdersSymbolType", "\_openOrdersSymbol") for grouping, plus "\_mostRecentOpenOrder" and "\_mostRecentClosedOrder" for recent orders.

We define the "Wallet" [constructor](https://www.mql5.com/en/book/oop/classes_and_interfaces/classes_ctors) to initialize counts to 0, create new "OrderCollection" instances, and set pointers to null, with the "~Wallet" [destructor](https://www.mql5.com/en/book/oop/structs_and_unions/structs_ctor_dtor) cleaning up all dynamic objects. The "HandleTick" method updates "\_recentClosedOrders" on new bars using [iTime](https://www.mql5.com/en/docs/series/itime), while "SetLastOrderResultsSize" resizes "LastOrderResults", and "SetLastClosedOrdersByTimeframe" sets "\_lastOrderResultByTimeframe". Public methods like "GetRecentClosedOrders", "GetOpenOrders", "GetOpenOrder", "GetPendingOpenOrders", and "GetPendingCloseOrders" retrieve order collections, with "ActivateOrderGroups" enabling grouping by "ORDER\_GROUP\_TYPE" values ("SymbolOrderType", "SymbolCode").

The "ResetPendingOrders" and "ResetOpenOrders" methods reinitialize collections, and "AreOrdersBeingOpened" and "AreOrdersBeingClosed" check pending execution status. We implement "LoadOrdersFromBroker" to populate orders via "OrderRepository::GetOpenOrders" and "GetLastClosedOrders", "SetPendingOpenOrderToOpen" and "CancelPendingOpenOrder" to manage pending opens, "SetAllOpenOrdersToPendingClose" and "SetOpenOrderToPendingClose" for closures, and "SetPendingCloseOrderToClosed" to update "LastOrderResults" and "\_closedOrderCount".

Private methods like "AddOrderToOpenOrderCollections", "RemoveOrderFromOpenOrderCollections", "SetMostRecentOpenOrClosedOrder", "GetOrderGroupSymbolOrderTypeKey", and "PrintOrderChanges" support order grouping and logging to "OrderInfoComment". This class will centralize order management for scalping. To see the current progress, let us define a base class for managing the startup and we can call it on initialization to visualize the milestone.

```
//--- Define enumeration for trade actions
enum TradeAction {
   UnknownAction = 0,                                     //--- Represent unknown action
   OpenBuyAction = 1,                                     //--- Represent open Buy action
   OpenSellAction = 2,                                    //--- Represent open Sell action
   CloseBuyAction = 3,                                    //--- Represent close Buy action
   CloseSellAction = 4                                    //--- Represent close Sell action
};

//--- Define interface for trader
interface ITrader {
   void HandleTick();                                     //--- Handle tick event
   void Init();                                           //--- Initialize trader
   Wallet* GetWallet();                                   //--- Retrieve wallet
};

//--- Declare global trader pointer
ITrader *_ea;                                             //--- Store EA instance

//--- Define main Expert Advisor class
class EA : public ITrader {
private:
   bool _firstTick;                                       //--- Track first tick
   Wallet* _wallet;                                       //--- Store wallet

public:
   //--- Initialize EA
   void EA() {
      _firstTick = true;                                  //--- Set first tick flag
      _wallet = new Wallet();                             //--- Create wallet
      _wallet.SetLastClosedOrdersByTimeframe(DisplayOrderDuringTimeframe); //--- Set closed orders timeframe
   }

   //--- Destructor to clean up EA
   void ~EA() {
      delete(_wallet);                                    //--- Delete wallet
   }

   //--- Initialize EA components
   void Init() {
      IsDemoLiveOrVisualMode = !MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_VISUAL_MODE); //--- Set mode flag
      UnitsOneLot = MarketInfo_LibFunc(Symbol(), MODE_LOTSIZE); //--- Set lot size
      _wallet.LoadOrdersFromBroker();                     //--- Load orders from broker
   }

   //--- Handle tick event
   void HandleTick() {
      if (MQLInfoInteger(MQL_TESTER) == 0) {              //--- Check if not in tester
         SyncOrders();                                    //--- Synchronize orders
      }
      if (AllowManualTPSLChanges) {                       //--- Check if manual TP/SL allowed
         SyncManualTPSLChanges();                         //--- Synchronize manual TP/SL
      }
      AskFunc.Evaluate();                                 //--- Update Ask price
      BidFunc.Evaluate();                                 //--- Update Bid price
      UpdateOrders();                                     //--- Update order profits
      if (!StopEA) {                                      //--- Check if EA not stopped
         _wallet.HandleTick();                            //--- Handle wallet tick
         if (ExecutePendingCloseOrders()) {               //--- Execute close orders
            if (!ExecutePendingOpenOrders()) {            //--- Execute open orders
               HandleErrors(StringFormat("Open (all) order(s) failed. Please check EA %d and look at the Journal and Expert tab.", MagicNumber)); //--- Log error
            }
         } else {
            HandleErrors(StringFormat("Close (all) order(s) failed! Please check EA %d and look at the Journal and Expert tab.", MagicNumber)); //--- Log error
         }
      } else {
         if (ExecutePendingCloseOrders()) {               //--- Execute close orders
            _wallet.SetAllOpenOrdersToPendingClose();     //--- Move open orders to pending close
         } else {
            HandleErrors(StringFormat("Close (all) order(s) failed! Please check EA %d and look at the Journal and Expert tab.", MagicNumber)); //--- Log error
         }
      }
      if (_firstTick) {                                   //--- Check if first tick
         _firstTick = false;                              //--- Clear first tick flag
      }
   }

   //--- Retrieve wallet
   Wallet* GetWallet() {
      return _wallet;                                     //--- Return wallet
   }

private:
   //--- Synchronize orders with broker
   void SyncOrders() {
      OrderCollection* currentOpenOrders = OrderRepository::GetOpenOrders(MagicNumber, NULL, Symbol()); //--- Retrieve open orders
      if (currentOpenOrders.Count() != (_wallet.GetOpenOrders().Count() + _wallet.GetPendingCloseOrders().Count())) { //--- Check order mismatch
         Print("(Manual) orderchanges detected" + " (found in MT: " + IntegerToString(currentOpenOrders.Count()) + " and in wallet: " + IntegerToString(_wallet.GetOpenOrders().Count()) + "), resetting EA, loading open orders."); //--- Log mismatch
         _wallet.ResetOpenOrders();                       //--- Reset open orders
         _wallet.ResetPendingOrders();                    //--- Reset pending orders
         _wallet.LoadOrdersFromBroker();                  //--- Reload orders
      }
      delete(currentOpenOrders);                          //--- Delete orders collection
   }

   //--- Synchronize manual TP/SL changes
   void SyncManualTPSLChanges() {
      _wallet.GetOpenOrders().Rewind();                   //--- Reset orders iterator
      while (_wallet.GetOpenOrders().HasNext()) {         //--- Iterate orders
         Order* order = _wallet.GetOpenOrders().Next();   //--- Get order
         uint lineFindResult = ObjectFind(ChartID(), IntegerToString(order.Ticket) + "_SL"); //--- Find SL line
         if (lineFindResult != UINT_MAX) {                //--- Check if SL line exists
            double currentPosition = ObjectGetDouble(ChartID(), IntegerToString(order.Ticket) + "_SL", OBJPROP_PRICE); //--- Get SL position
            if ((order.StopLossManual == 0 && currentPosition != order.GetClosestSL()) || //--- Check manual SL change
                (order.StopLossManual != 0 && currentPosition != order.StopLossManual)) { //--- Check manual SL mismatch
               order.StopLossManual = currentPosition;       //--- Update manual SL
            }
         }
         lineFindResult = ObjectFind(ChartID(), IntegerToString(order.Ticket) + "_TP"); //--- Find TP line
         if (lineFindResult != UINT_MAX) {                //--- Check if TP line exists
            double currentPosition = ObjectGetDouble(ChartID(), IntegerToString(order.Ticket) + "_TP", OBJPROP_PRICE); //--- Get TP position
            if ((order.TakeProfitManual == 0 && currentPosition != order.GetClosestTP()) || //--- Check manual TP change
                (order.TakeProfitManual != 0 && currentPosition != order.TakeProfitManual)) { //--- Check manual TP mismatch
               order.TakeProfitManual = currentPosition;     //--- Update manual TP
            }
         }
      }
   }

   //--- Update order profits
   void UpdateOrders() {
      _wallet.GetOpenOrders().Rewind();                   //--- Reset orders iterator
      while (_wallet.GetOpenOrders().HasNext()) {         //--- Iterate orders
         Order* order = _wallet.GetOpenOrders().Next();   //--- Get order
         double pipsProfit = order.CalculateProfitPips(); //--- Calculate profit
         order.CurrentProfitPips = pipsProfit;            //--- Update current profit
         if (pipsProfit < order.LowestProfitPips) {       //--- Check if lowest profit
            order.LowestProfitPips = pipsProfit;          //--- Update lowest profit
         } else if (pipsProfit > order.HighestProfitPips) { //--- Check if highest profit
            order.HighestProfitPips = pipsProfit;         //--- Update highest profit
         }
      }
   }

   //--- Execute pending close orders
   bool ExecutePendingCloseOrders() {
      OrderCollection* pendingCloseOrders = _wallet.GetPendingCloseOrders(); //--- Retrieve pending close orders
      int ordersToCloseCount = pendingCloseOrders.Count(); //--- Get count
      if (ordersToCloseCount == 0) {                      //--- Check if no orders
         return true;                                     //--- Return true
      }
      if (_wallet.AreOrdersBeingOpened()) {               //--- Check if orders being opened
         return true;                                     //--- Return true
      }
      int ordersCloseSuccessCount = 0;                    //--- Initialize success count
      for (int i = ordersToCloseCount - 1; i >= 0; i--) { //--- Iterate orders
         Order* pendingCloseOrder = pendingCloseOrders.Get(i); //--- Get order
         if (pendingCloseOrder.IsAwaitingDealExecution) { //--- Check if awaiting execution
            ordersCloseSuccessCount++;                    //--- Increment success count
            continue;                                     //--- Move to next
         }
         bool success;                                    //--- Declare success flag
         if (AccountMarginMode == ACCOUNT_MARGIN_MODE_RETAIL_NETTING) { //--- Check netting mode
            Order* reversedOrder = new Order(pendingCloseOrder, false); //--- Create reversed order
            reversedOrder.Type = pendingCloseOrder.Type == ORDER_TYPE_BUY ? ORDER_TYPE_SELL : ORDER_TYPE_BUY; //--- Set opposite type
            success = OrderRepository::OpenOrder(reversedOrder); //--- Open reversed order
            if (success) {                                //--- Check if successful
               pendingCloseOrder.Ticket = reversedOrder.Ticket; //--- Update ticket
            }
            delete(reversedOrder);                        //--- Delete reversed order
         } else {
            success = OrderRepository::ClosePosition(pendingCloseOrder); //--- Close position
         }
         if (success) {                                   //--- Check if successful
            ordersCloseSuccessCount++;                    //--- Increment success count
         }
      }
      return ordersCloseSuccessCount == ordersToCloseCount; //--- Return true if all successful
   }

   //--- Execute pending open orders
   bool ExecutePendingOpenOrders() {
      OrderCollection* pendingOpenOrders = _wallet.GetPendingOpenOrders(); //--- Retrieve pending open orders
      int ordersToOpenCount = pendingOpenOrders.Count(); //--- Get count
      if (ordersToOpenCount == 0) {                       //--- Check if no orders
         return true;                                     //--- Return true
      }
      int ordersOpenSuccessCount = 0;                     //--- Initialize success count
      for (int i = ordersToOpenCount - 1; i >= 0; i--) { //--- Iterate orders
         Order* order = pendingOpenOrders.Get(i);         //--- Get order
         if (order.IsAwaitingDealExecution) {             //--- Check if awaiting execution
            ordersOpenSuccessCount++;                     //--- Increment success count
            continue;                                     //--- Move to next
         }
         bool isTradeContextFree = false;                 //--- Initialize trade context flag
         double StartWaitingTime = GetTickCount();        //--- Start timer
         while (true) {                                   //--- Wait for trade context
            if (MQL5InfoInteger(MQL5_TRADE_ALLOWED)) {    //--- Check if trade allowed
               isTradeContextFree = true;                 //--- Set trade context free
               break;                                     //--- Exit loop
            }
            int MaxWaiting_sec = 10;                      //--- Set max wait time
            if (IsStopped()) {                            //--- Check if EA stopped
               HandleErrors("The expert was stopped by a user action."); //--- Log error
               break;                                     //--- Exit loop
            }
            if (GetTickCount() - StartWaitingTime > MaxWaiting_sec * 1000) { //--- Check if timeout
               HandleErrors(StringFormat("The (%d seconds) waiting time exceeded. Trade not allowed: EA disabled, market closed or trade context still not free.", MaxWaiting_sec)); //--- Log error
               break;                                     //--- Exit loop
            }
            Sleep(100);                                   //--- Wait briefly
         }
         if (!isTradeContextFree) {                       //--- Check if trade context not free
            if (!_wallet.CancelPendingOpenOrder(order)) { //--- Attempt to cancel order
               HandleErrors("Failed to cancel an order (because it couldn't open). Please see the Journal and Expert tab in Metatrader for more information."); //--- Log error
            }
            continue;                                     //--- Move to next
         }
         bool success = OrderRepository::OpenOrder(order); //--- Open order
         if (success) {                                   //--- Check if successful
            ordersOpenSuccessCount++;                     //--- Increment success count
         } else {
            if (!_wallet.CancelPendingOpenOrder(order)) { //--- Attempt to cancel order
               HandleErrors("Failed to cancel an order (because it couldn't open). Please see the Journal and Expert tab in Metatrader for more information."); //--- Log error
            }
         }
      }
      return ordersOpenSuccessCount == ordersToOpenCount; //--- Return true if all successful
   }
};
```

Here, we finalize the core infrastructure by defining trade actions and the main Expert Advisor logic. We create the "TradeAction" enumeration to categorize trade operations, including "UnknownAction" (0), "OpenBuyAction" (1), "OpenSellAction" (2), "CloseBuyAction" (3), and "CloseSellAction" (4), providing a clear framework for trade management. We will use this later. We then define the "ITrader" [interface](https://www.mql5.com/en/docs/basis/types/classes) with methods "HandleTick", "Init", and "GetWallet" to standardize EA functionality, and declare a global "\_ea" pointer of type "ITrader" to store the EA instance.

We implement the "EA" class, inheriting from "ITrader", with private variables "\_firstTick" to track initial ticks and "\_wallet" to manage the "Wallet" instance. The "EA" constructor initializes "\_firstTick" to true and creates a new "Wallet", setting its timeframe via "SetLastClosedOrdersByTimeframe" with "DisplayOrderDuringTimeframe". The "~EA" destructor cleans up "\_wallet". The "Init" method sets "IsDemoLiveOrVisualMode" using [MQLInfoInteger](https://www.mql5.com/en/docs/check/mqlinfointeger), assigns "UnitsOneLot" via "MarketInfo\_LibFunc", and calls "\_wallet.LoadOrdersFromBroker".

The "HandleTick" method manages ticks by calling "SyncOrders" (except in tester mode), "SyncManualTPSLChanges" if "AllowManualTPSLChanges" is true, updating "AskFunc" and "BidFunc", and invoking "UpdateOrders" and "\_wallet.HandleTick". It executes "ExecutePendingCloseOrders" and "ExecutePendingOpenOrders" unless "StopEA" is true, logging errors via "HandleErrors" if needed, and clears "\_firstTick".

Private methods include "SyncOrders" to synchronize orders using "OrderRepository::GetOpenOrders", "SyncManualTPSLChanges" to update manual TP/SL via [ObjectFind](https://www.mql5.com/en/docs/objects/ObjectFind) and [ObjectGetDouble](https://www.mql5.com/en/docs/objects/objectgetdouble), "UpdateOrders" to refresh profit metrics with "CalculateProfitPips", "ExecutePendingCloseOrders" to close positions using "OrderRepository::ClosePosition" or "OpenOrder" for netting, and "ExecutePendingOpenOrders" to open orders with "OrderRepository::OpenOrder", ensuring trade context via "MQL5InfoInteger" and handling cancellations. We can now call this on the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler.

```
//--- Set up chart appearance
void SetupChart() {
   ChartSetInteger(ChartID(), CHART_FOREGROUND, 0, false); //--- Set chart foreground to background
}

//--- Initialize Expert Advisor
int OnInit() {
   ChartSetInteger(0, CHART_COLOR_BACKGROUND, clrWhite); //--- Set chart background to white
   ChartSetInteger(0, CHART_COLOR_CANDLE_BEAR, clrRed); //--- Set bearish candles to red
   ChartSetInteger(0, CHART_COLOR_CANDLE_BULL, clrGreen); //--- Set bullish candles to green
   ChartSetInteger(0, CHART_COLOR_ASK, clrDarkRed); //--- Set Ask line to dark red
   ChartSetInteger(0, CHART_COLOR_BID, clrDarkGreen); //--- Set Bid line to dark green
   ChartSetInteger(0, CHART_COLOR_CHART_DOWN, clrRed); //--- Set downward movement to red
   ChartSetInteger(0, CHART_COLOR_CHART_UP, clrGreen); //--- Set upward movement to green
   ChartSetInteger(0, CHART_COLOR_GRID, clrLightGray); //--- Set grid to light gray
   ChartSetInteger(0, CHART_COLOR_FOREGROUND, clrBlack); //--- Set axis and text to black
   ChartSetInteger(0, CHART_COLOR_LAST, clrBlack); //--- Set last price line to black
   OrderFillingType = GetFillingType();                //--- Retrieve order filling type
   if ((int)OrderFillingType == -1) {                  //--- Check if invalid
      HandleErrors("Unsupported filling type " + IntegerToString((int)OrderFillingType)); //--- Log error
      return (INIT_FAILED);                            //--- Return failure
   }
   GetExecutionType();                                 //--- Retrieve execution type
   AccountMarginMode = GetAccountMarginMode();         //--- Retrieve margin mode
   SetPipPoint();                                      //--- Set pip point
   if (PipPoint == 0) {                                //--- Check if invalid
      HandleErrors("Couldn't find correct pip/point for symbol."); //--- Log error
      return (INIT_FAILED);                            //--- Return failure
   }
   AskFunc = new AskFunction();                        //--- Create Ask function
   AskFunc.Init();                                     //--- Initialize Ask function
   BidFunc = new BidFunction();                        //--- Create Bid function
   BidFunc.Init();                                     //--- Initialize Bid function
   OrderInfoComment = "";                              //--- Initialize order comment
   _ea = new EA();                                     //--- Create EA instance
   _ea.Init();                                         //--- Initialize EA
   SetupChart();                                       //--- Set up chart
   hd_iMA_SMA8 = iMA(NULL, PERIOD_M30, iMA_SMA8_ma_period, iMA_SMA8_ma_shift, MODE_SMA, PRICE_CLOSE); //--- Initialize 8-period SMA
   if (hd_iMA_SMA8 < 0) {                              //--- Check if failed
      HandleErrors(StringFormat("Could not find indicator 'iMA'. Error: %d", GetLastError())); //--- Log error
      return -1;                                       //--- Return failure
   }
   hd_iMA_EMA200 = iMA(NULL, PERIOD_M1, iMA_EMA200_ma_period, iMA_EMA200_ma_shift, MODE_EMA, PRICE_CLOSE); //--- Initialize 200-period EMA
   if (hd_iMA_EMA200 < 0) {                            //--- Check if failed
      HandleErrors(StringFormat("Could not find indicator 'iMA'. Error: %d", GetLastError())); //--- Log error
      return -1;                                       //--- Return failure
   }
   hd_iRSI_RSI = iRSI(NULL, PERIOD_M1, iRSI_RSI_ma_period, PRICE_CLOSE); //--- Initialize RSI
   if (hd_iRSI_RSI < 0) {                              //--- Check if failed
      HandleErrors(StringFormat("Could not find indicator 'iRSI'. Error: %d", GetLastError())); //--- Log error
      return -1;                                       //--- Return failure
   }
   hd_iEnvelopes_ENV_LOW = iEnvelopes(NULL, PERIOD_M1, iEnvelopes_ENV_LOW_ma_period, iEnvelopes_ENV_LOW_ma_shift, MODE_SMA, PRICE_CLOSE, iEnvelopes_ENV_LOW_deviation); //--- Initialize lower Envelopes
   if (hd_iEnvelopes_ENV_LOW < 0) {                    //--- Check if failed
      HandleErrors(StringFormat("Could not find indicator 'iEnvelopes'. Error: %d", GetLastError())); //--- Log error
      return -1;                                       //--- Return failure
   }
   hd_iEnvelopes_ENV_UPPER = iEnvelopes(NULL, PERIOD_M1, iEnvelopes_ENV_UPPER_ma_period, iEnvelopes_ENV_UPPER_ma_shift, MODE_SMA, PRICE_CLOSE, iEnvelopes_ENV_UPPER_deviation); //--- Initialize upper Envelopes
   if (hd_iEnvelopes_ENV_UPPER < 0) {                  //--- Check if failed
      HandleErrors(StringFormat("Could not find indicator 'iEnvelopes'. Error: %d", GetLastError())); //--- Log error
      return -1;                                       //--- Return failure
   }
   hd_iMA_SMA_4 = iMA(NULL, PERIOD_M30, iMA_SMA_4_ma_period, iMA_SMA_4_ma_shift, MODE_SMA, PRICE_CLOSE); //--- Initialize 4-period SMA
   if (hd_iMA_SMA_4 < 0) {                             //--- Check if failed
      HandleErrors(StringFormat("Could not find indicator 'iMA'. Error: %d", GetLastError())); //--- Log error
      return -1;                                       //--- Return failure
   }
   return (INIT_SUCCEEDED);                            //--- Return success
}

//--- Handle errors
void HandleErrors(string errorMessage) {
   Print(errorMessage);                                //--- Log error
   if (Error != NULL || errorMessage == ErrorPreviousQuote) { //--- Check existing or repeated error
      return;                                          //--- Exit
   }
   if (AlertOnError) Alert(errorMessage);              //--- Trigger alert if enabled
   if (NotificationOnError) SendNotification(StringFormat("Error by EA (%d) %s", MagicNumber, errorMessage)); //--- Send notification if enabled
   if (EmailOnError) SendMail(StringFormat("Error by EA (%d)", MagicNumber), errorMessage); //--- Send email if enabled
   Error = errorMessage;                               //--- Set current error
   ErrorPreviousQuote = Error;                         //--- Set previous error
}
```

To initialize the program, we define the "SetupChart" function to configure the chart appearance by setting [CHART\_FOREGROUND](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer) to false using [ChartSetInteger](https://www.mql5.com/en/docs/chart_operations/chartsetinteger), ensuring the chart background is prioritized for clarity. We implement the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) function to initialize the EA, starting with chart customization via "ChartSetInteger" to set colors: "CHART\_COLOR\_BACKGROUND" to white, [CHART\_COLOR\_CANDLE\_BEAR](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer) to red, "CHART\_COLOR\_CANDLE\_BULL" to green, "CHART\_COLOR\_ASK" to dark red, "CHART\_COLOR\_BID" to dark green, and others for visual distinction.

We call "GetFillingType" to set "OrderFillingType", returning [INIT\_FAILED](https://www.mql5.com/en/docs/basis/function/events) if invalid, and invoke "GetExecutionType" and "GetAccountMarginMode" to configure trading modes. The "SetPipPoint" function sets "PipPoint", with a failure check, and we instantiate "AskFunc" and "BidFunc" as "AskFunction" and "BidFunction" objects, calling their "Init" methods.

We create the "\_ea" instance of the "EA" class, initialize it with "Init", and call "SetupChart". We initialize indicator handles using "iMA" for "hd\_iMA\_SMA8" (M30, 14-period SMA), "hd\_iMA\_EMA200" (M1, 200-period EMA), "hd\_iMA\_SMA\_4" (M30, 9-period SMA), "iRSI" for "hd\_iRSI\_RSI" (M1, 8-period), and "iEnvelopes" for "hd\_iEnvelopes\_ENV\_LOW" (M1, 95-period, 1.4% deviation) and "hd\_iEnvelopes\_ENV\_UPPER" (M1, 150-period, 0.1% deviation), returning -1 on failure with "HandleErrors".

We define the "HandleErrors" function to log errors via [Print](https://www.mql5.com/en/docs/common/print), skipping duplicates using "Error" and "ErrorPreviousQuote", and supporting notifications via "Alert", "SendNotification", or "SendMail" based on "AlertOnError", "NotificationOnError", and "EmailOnError", updating "Error" and "ErrorPreviousQuote". This setup now ensures the program is ready for core logic handling, and when we run it, we have the following input outcome.

![INPUTS SECTION](https://c.mql5.com/2/145/Screenshot_2025-05-25_185123.png)

From the visualization, we can see that the user can input inputs for the program control. When we run the program, we have the following outcome.

![INITIALIZATION CONFIRMATION](https://c.mql5.com/2/145/Screenshot_2025-05-25_185931.png)

From the image, we can see that we have successfully initialized the program and it is ready to take orders. With that, we have defined the core infrastructure that initializes the program. The thing that remains is backtesting the program to ensure it starts correctly, and that is handled in the next section.

### Backtesting

We compiled the backtest in one [Graphics Interchange Format](https://en.wikipedia.org/wiki/GIF "https://en.wikipedia.org/wiki/GIF") (GIF) file as below to showcase the initialization logic of the program.

![BACKTEST](https://c.mql5.com/2/145/SCALPER_1.gif)

### Conclusion

In conclusion, we have successfully laid the groundwork for automating the [Envelopes](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/envelopes "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/envelopes") Trend Bounce Scalping Strategy in MQL5, establishing a robust Expert Advisor infrastructure and signal generation framework. We configured essential components, including indicator initialization, order management classes, and error handling, to support precise scalping operations. This foundation sets the stage for implementing trade execution and dynamic management in the next part, bringing us closer to a fully automated trading program. Keep tuned.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18269.zip "Download all attachments in the single ZIP archive")

[Envelopes\_Trend\_Bounce\_Scalping\_Part\_1.mq5](https://www.mql5.com/en/articles/download/18269/envelopes_trend_bounce_scalping_part_1.mq5 "Download Envelopes_Trend_Bounce_Scalping_Part_1.mq5")(295.02 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/488024)**

![ALGLIB library optimization methods (Part I)](https://c.mql5.com/2/99/Alglib_Library_Optimization_Techniques_PartI___LOGO__1.png)[ALGLIB library optimization methods (Part I)](https://www.mql5.com/en/articles/16133)

In this article, we will get acquainted with the ALGLIB library optimization methods for MQL5. The article includes simple and clear examples of using ALGLIB to solve optimization problems, which will make mastering the methods as accessible as possible. We will take a detailed look at the connection of such algorithms as BLEIC, L-BFGS and NS, and use them to solve a simple test problem.

![Neural Networks in Trading: Market Analysis Using a Pattern Transformer](https://c.mql5.com/2/97/Market_Situation_Analysis_Using_Pattern_Transformer___LOGO.png)[Neural Networks in Trading: Market Analysis Using a Pattern Transformer](https://www.mql5.com/en/articles/16130)

When we use models to analyze the market situation, we mainly focus on the candlestick. However, it has long been known that candlestick patterns can help in predicting future price movements. In this article, we will get acquainted with a method that allows us to integrate both of these approaches.

![From Basic to Intermediate: Array (III)](https://c.mql5.com/2/99/Do_b4sico_ao_intermedierio__Array_III__LOGO.png)[From Basic to Intermediate: Array (III)](https://www.mql5.com/en/articles/15473)

In this article, we will look at how to work with arrays in MQL5, including how to pass information between functions and procedures using arrays. The purpose is to prepare you for what will be demonstrated and explained in future materials in the series. Therefore, I strongly recommend that you carefully study what will be shown in this article.

![Developing a multi-currency Expert Advisor (Part 19): Creating stages implemented in Python](https://c.mql5.com/2/99/Developing_a_Multicurrency_Advisor_Part_19__LOGO.png)[Developing a multi-currency Expert Advisor (Part 19): Creating stages implemented in Python](https://www.mql5.com/en/articles/15911)

So far we have considered the automation of launching sequential procedures for optimizing EAs exclusively in the standard strategy tester. But what if we would like to perform some handling of the obtained data using other means between such launches? We will attempt to add the ability to create new optimization stages performed by programs written in Python.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=kyvjzzqdposvqstmkeztjurrozcoxbos&ssn=1769093784497825632&ssn_dr=0&ssn_sr=0&fv_date=1769093784&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18269&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2018)%3A%20Envelopes%20Trend%20Bounce%20Scalping%20-%20Core%20Infrastructure%20and%20Signal%20Generation%20(Part%20I)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909378478930168&fz_uniq=5049499557408124056&sv=2552)

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