---
title: MQL5 Trading Toolkit (Part 3): Developing a Pending Orders Management EX5 Library
url: https://www.mql5.com/en/articles/15888
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:38:51.779818
---

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/15888&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062615760464946583)

MetaTrader 5 / Examples


### Introduction

In the [first](https://www.mql5.com/en/articles/14822) and [second](https://www.mql5.com/en/articles/15224) articles, you were introduced to the development and programming of EX5 libraries from scratch, how to create detailed documentation to help end users implement them in their MQL5 projects, and received a practical, step-by-step demonstration on how to import and implement them in various MQL5 applications.

In this article, we will develop a comprehensive Pending Orders Management EX5 library and create a graphical user interface (GUI) panel to demonstrate how to import and implement the library as a hands-on practical example. This Pending Orders Management library will use only MQL5 standard functions to open, modify, and delete different types of pending orders. It will serve as a valuable learning resource for new MQL5 programmers looking to learn how to code simple to advanced pending order management modules. You will also learn how to filter and sort pending orders based on different categories.

The functions we create below will be exported and compiled into the EX5 library, making it a valuable asset that will dramatically reduce the development time of future MQL5 projects. By simply importing the Pending Orders Management EX5 library, you can implement these functions, thereby significantly shortening both the codebase and the overall development and coding process.

### Create A New EX5 Library Source Code File (.mq5)

To get started, open your _MetaEditor IDE_, and launch the MQL Wizard using the ' _New_' menu item button. You'll need to create a new _Library_ source code file, which we will name _PendingOrdersManager.mq5_. This file will contain the core functions for managing pending orders. Save it in the _Libraries\\Toolkit_ folder we set up in the [first article](https://www.mql5.com/en/articles/14822). This is the same directory where we previously saved the _Positions Manager EX5 library_, keeping our project organized and consistent.

![PendingOrdersManager.mql5 Saved_Directory](https://c.mql5.com/2/97/PendingOrdersManager_Saved_Directory.png)

If you need a refresher on how to create an _EX5 library_ in MQL5, I recommend going back to the first article. There, we covered the step-by-step process of setting up your folder structure and creating libraries in detail. Following that approach will not only help streamline development but also make it easier to maintain and reuse your libraries in future MQL5 projects.

### Preprocessor Directives, Global Variables, and Positions Manager EX5 Library Imports

In our newly created _PendingOrdersManager.mq5_ library source code file, we will begin by defining all the preprocessor directives, global variables, library imports, and trade structures. This ensures they are available throughout our pending orders management library and can be reused in various functions as needed.

We will start by defining a preprocessor macro using _#define_ to create a constant that represents a blank string. This constant, named _ALL\_SYMBOLS_, will act as a placeholder in the pending order management functions that require a symbol string as an input parameter. By passing this empty string, the functions will interpret it as a command to apply the specified action to all available symbols, rather than just a single one.

This approach is simple yet highly effective, especially when managing multiple instruments, as it streamlines the process of handling pending orders across different symbols. To implement this, define the _ALL\_SYMBOLS_ constant and place it directly below the #property directives in your code.

```
#define ALL_SYMBOLS "" //-- Used as a function parameter to select all symbols
```

We also need to create two additional constants: one to define the maximum number of times we can retry sending an order to the trade server if it fails, and another to set the delay duration between retry attempts. This helps prevent overwhelming the trade server with rapid, consecutive requests. We will name the first constant _MAX\_ORDER\_RETRIES_ and set its value to _600_. The second constant will be _ORDER\_RETRY\_DELAY_, with a value of _500_ (milliseconds).

This means that if no critical error occurs, we can attempt to open, modify, or delete an order up to 600 times. After each attempt, the function will pause for half a second ( _500 milliseconds_) before trying again. This delay ensures that we don’t overload the server and that resources are used efficiently. It also allows for any delays, such as market inactivity or the unavailability of tick data, to be resolved before all retry attempts are exhausted, increasing the likelihood of successful order execution.

```
#define MAX_ORDER_RETRIES 600 //-- Sets the order sending retry limit
#define ORDER_RETRY_DELAYS 500//-- Sets the duration to pause before re-sending a failed order request in milliseconds
```

Next, add the MQL5 _MqlTradeRequest_ and _MqlTradeResult_ data structures. These structures are important, as they will handle all communication with the trade server when we open, modify, and delete different pending orders. The _tradeRequest_ structure will be responsible for storing the details of the trade action we will execute, including order parameters like price, stop loss, and take profit. Meanwhile, the _tradeResult_ structure will capture and save the outcome of the trade operation, providing us with feedback from the server about whether our request was successful or encountered an error.

```
//-- Trade operations request and result data structures global variables
MqlTradeRequest tradeRequest;
MqlTradeResult  tradeResult;
```

To track and store the status of various pending orders, we need to create several types of variables that will hold this status information. These order status variables will be declared as global variables to ensure they are accessible from every scope within our library, allowing any function to reference and update them as needed. We will initialize these variables with default values when we declare them and then later update them with real-time accurate data within the _GetPendingOrdersData(...)_ function, which we will develop later in this article. This will ensure the status of all pending orders will remain consistent and accurately updated throughout the execution of the calling functions.

```
//-- Pending orders status global variables
//-------------------------------------------------------------------------------------------------------------------
int accountBuyStopOrdersTotal = 0, accountSellStopOrdersTotal = 0,
    accountBuyLimitOrdersTotal = 0, accountSellLimitOrdersTotal = 0,

    symbolPendingOrdersTotal = 0,
    symbolBuyStopOrdersTotal = 0, symbolSellStopOrdersTotal = 0,
    symbolBuyLimitOrdersTotal = 0, symbolSellLimitOrdersTotal = 0,

    magicPendingOrdersTotal = 0,
    magicBuyStopOrdersTotal = 0, magicSellStopOrdersTotal = 0,
    magicBuyLimitOrdersTotal = 0, magicSellLimitOrdersTotal = 0;

double accountPendingOrdersVolumeTotal = 0.0,
       accountBuyStopOrdersVolumeTotal = 0.0, accountSellStopOrdersVolumeTotal = 0.0,
       accountBuyLimitOrdersVolumeTotal = 0.0, accountSellLimitOrdersVolumeTotal = 0.0,

       symbolPendingOrdersVolumeTotal = 0.0,
       symbolBuyStopOrdersVolumeTotal = 0.0, symbolSellStopOrdersVolumeTotal = 0.0,
       symbolBuyLimitOrdersVolumeTotal = 0.0, symbolSellLimitOrdersVolumeTotal = 0.0,

       magicPendingOrdersVolumeTotal = 0.0,
       magicBuyStopOrdersVolumeTotal = 0.0, magicSellStopOrdersVolumeTotal = 0.0,
       magicBuyLimitOrdersVolumeTotal = 0.0, magicSellLimitOrdersVolumeTotal = 0.0;
```

It is common to encounter various types of errors when opening, modifying, or deleting orders. To efficiently handle and resolve these errors, we will need a dedicated function designed to manage them. Additionally, we require a function to check critical trading permissions, such as whether our Expert Advisor is authorized to trade, if algorithmic trading is enabled in the terminal, or if our broker has allowed algorithmic trading for the account.

Fortunately, we already developed similar functions in the _PositionsManager.ex5_ library from the previous articles. Instead of recreating these functions, we will simply import this library, making these error-handling and permission-checking functions readily available for use in our _Pending Orders Manager library_. Import the _ErrorAdvisor(...)_ and _TradingIsAllowed()_ functions.

```
//+----------------------------------------------------------------------------+
//| PositionsManager.ex5 imports                                               |
//+----------------------------------------------------------------------------+
#import "Toolkit/PositionsManager.ex5" //-- Opening import directive
//-- Function descriptions for the imported function prototypes

//-- Error Handling and Permission Status Functions
bool   ErrorAdvisor(string callingFunc, string symbol, int tradeServerErrorCode);
bool   TradingIsAllowed();

#import //--- Closing import directive
```

### Print Order Details Function

The _PrintOrderDetails(...)_ function logs detailed information about a trade request and the server's response in the Expert Advisor log within _MetaTrader 5_. It takes two string parameters: _header_, which holds a custom message, and _symbol_, which represents the trading symbol or instrument being processed. The function outputs a comprehensive report that includes essential order details such as the _symbol, order type, volume, price, stop loss, take profit, comment, magic number,_ and any _deviations_. Additionally, it logs important information about the trade result, including the server response and any _runtime errors_. The main purpose of this function is to assist in debugging and monitoring trade operations by providing a clear, formatted overview of each order's parameters and status.

```
void PrintOrderDetails(string header, string symbol)
  {
   string orderDescription;
   int symbolDigits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
//-- Print the order details
   orderDescription += "_______________________________________________________________________________________\r\n";
   orderDescription += "--> "  + tradeRequest.symbol + " " + EnumToString(tradeRequest.type) + " " + header +
                       " <--\r\n";
   orderDescription += "Volume: " + StringFormat("%d", tradeRequest.volume) + "\r\n";
   orderDescription += "Price: " + DoubleToString(tradeRequest.price, symbolDigits) + "\r\n";
   orderDescription += "Stop Loss: " + DoubleToString(tradeRequest.sl, symbolDigits) + "\r\n";
   orderDescription += "Take Profit: " + DoubleToString(tradeRequest.tp, symbolDigits) + "\r\n";
   orderDescription += "Comment: " + tradeRequest.comment + "\r\n";
   orderDescription += "Magic Number: " + StringFormat("%d", tradeRequest.magic) + "\r\n";
   orderDescription += "Order filling: " + EnumToString(tradeRequest.type_filling)+ "\r\n";
   orderDescription += "Deviation points: " + StringFormat("%G", tradeRequest.deviation) + "\r\n";
   orderDescription += "RETCODE: " + (string)(tradeResult.retcode) + "\r\n";
   orderDescription += "Runtime Code: " + (string)(GetLastError()) + "\r\n";
   orderDescription += "---";
   Print(orderDescription);
  }
```

### Open Buy Limit Function

The _OpenBuyLimit(...)_ function will be tasked with the responsibility of placing new _buy limit_ pending orders in the _MetaTrader 5_ trading terminal. This function ensures that all necessary conditions are met before submitting the order request, with built-in error handling to manage potential issues. It verifies that all parameters are valid and conform to the broker’s requirements. Additionally, it implements a retry loop, which increases the likelihood of successful order placement, making it a reliable and efficient tool for managing pending orders. The function's comprehensive error handling and logging capabilities further aid in identifying and resolving issues that may occur during the order placement process. The function will return a _boolean_ value of _true_ if it successfully opens a buy limit order, and _false_ if the order request is unsuccessful.

#### What Is A Buy Limit Order?

A buy limit order is a request to buy at an _Ask_ price that is equal to or lower than the specified order entry price. The buy limit entry price is valid only when the current market price is higher than the order's entry price. This type of order is useful when you anticipate that the symbol's price will drop to your desired entry level and then rise, allowing you to profit from the upward movement.

![Buy Limit Order](https://c.mql5.com/2/95/Buy_Limit_Order_2png.png)

Let us start by defining the function, which will accept seven parameters, each with a specific role in the process of placing a buy limit order. The function will be exportable, allowing it to be accessed externally in the EX5 library. These parameters are:

1. **magicNumber (unsigned long)**: A unique identifier for the trade, used to distinguish orders placed by our Expert Advisor from others.
2. **symbol (string)**: The trading instrument or symbol (such as _EURUSD_) for which the order will be placed.
3. **entryPrice (double)**: The price level at which the buy limit order will be triggered. This is the target price at which you want to buy the asset.
4. **lotSize (double)**: The volume or size of the order to be placed, representing how many lots of the asset will be traded.
5. **sl (int)**: The stop loss value, measured in pips. This defines the maximum loss the trade can incur before being automatically closed.
6. **tp (int)**: The take profit value, also measured in pips. This defines the price level where profits will be automatically secured by closing the order.
7. **orderComment (string)**: A comment or note attached to the order for identification or tracking purposes.

```
bool OpenBuyLimit(ulong magicNumber, string symbol, double entryPrice, double lotSize, int sl, int tp, string orderComment) export
  {
```

After defining the function and its parameters, the first step in the function is to check whether algorithmic or Expert Advisor trading is enabled in _MetaTrader 5_. If it is disabled, we will exit the function immediately, preventing the order from being placed.

```
if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }
```

Next, we create the _tpPrice_ and _slPrice_ variables to save the take profit and stop loss price values and then retrieve key information about the trading symbol, such as the number of decimal places ( _digits_), minimum stop level, and the spread. These details will help us ensure the entry price and order parameters are correctly set for the chosen symbol.

```
double tpPrice = 0.0, slPrice = 0.0;

//-- Get some information about the orders symbol
   int symbolDigits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   int symbolStopLevel = (int)SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double symbolPoint = SymbolInfoDouble(symbol, SYMBOL_POINT);
   int spread = (int)SymbolInfoInteger(symbol, SYMBOL_SPREAD);

//-- Save the order type enumeration
   ENUM_ORDER_TYPE orderType = ORDER_TYPE_BUY_LIMIT;
```

Before we send the order, we need to check whether the entry price is valid, ensuring that the price is not too close to the market price or below the broker’s stop level. If the price is invalid, we will log an error and exit the function.

```
//-- check if the entry price is valid
   if(
      SymbolInfoDouble(symbol, SYMBOL_ASK) - (symbolStopLevel * symbolPoint) <
      entryPrice + (spread * symbolPoint)
   )
     {
      Print(
         "\r\n", __FUNCTION__, ": Can't open a new ", EnumToString(orderType),
         ". (Reason --> INVALID ENTRY PRICE: ", DoubleToString(entryPrice, symbolDigits), ")\r\n"
      );
      return(false); //-- Invalid entry price, log the error, exit the function and return false
     }
```

Let us ensure the stop loss ( _SL_) and take profit ( _TP_) values meet the broker’s minimum requirements. If necessary, we will adjust the _SL_ and _TP_ to comply with these rules before proceeding.

```
//-- Check the validity of the sl and tp
   if(sl > 0 && sl < symbolStopLevel)
     {
      sl = symbolStopLevel;
     }
   if(tp > 0 && tp < symbolStopLevel)
     {
      tp = symbolStopLevel;
     }

   slPrice = (sl > 0) ? NormalizeDouble(entryPrice - sl * symbolPoint, symbolDigits) : 0;
   tpPrice = (tp > 0) ? NormalizeDouble(entryPrice + tp * symbolPoint, symbolDigits) : 0;
```

Once we have validated the price and other parameters, we will prepare the _tradeRequest_ structure with details like the order type, symbol, price, volume, stop loss, take profit, and a comment. We will also ensure the lot size is within the broker’s allowed limits and reset any previous trade results or errors.

```
//-- reset the the tradeRequest and tradeResult values by zeroing them
   ZeroMemory(tradeRequest);
   ZeroMemory(tradeResult);

//-- initialize the parameters to open a buy limit order
   tradeRequest.type = orderType;
   tradeRequest.action = TRADE_ACTION_PENDING;
   tradeRequest.magic = magicNumber;
   tradeRequest.symbol = symbol;
   tradeRequest.price = NormalizeDouble(entryPrice, symbolDigits);
   tradeRequest.tp = tpPrice;
   tradeRequest.sl = slPrice;
   tradeRequest.comment = orderComment;
   tradeRequest.deviation = SymbolInfoInteger(symbol, SYMBOL_SPREAD) * 2;

//-- Set and moderate the lot size or volume
//-- Verify that volume is not less than allowed minimum
   lotSize = MathMax(lotSize, SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN));

//-- Verify that volume is not more than allowed maximum
   lotSize = MathMin(lotSize, SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX));

//-- Round down to nearest volume step
   lotSize = MathFloor(lotSize / SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP)) * SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   tradeRequest.volume = lotSize;

//--- Reset error cache so that we get an accurate runtime error code in the ErrorAdvisor function
   ResetLastError();
```

If the order request is valid, we will attempt to send it to the broker’s trade server. If the order placement fails, we will retry re-sending the order request up to 600 times while pausing for half a second (5 _00 milliseconds_) before retrying, making multiple attempts to ensure the order is placed successfully. This retry mechanism greatly increases the chance of successfully opening the order, especially in cases of temporary network or server issues.

If the order is successfully placed, we will log the details of the order and confirm that the trade server has processed it. If the order fails even after multiple attempts, we will use the _ErrorAdvisor(..)_ function we imported from the _PositionsManager.ex5_ library to diagnose the issue and return _false_ to indicate that the order was not placed.

```
for(int loop = 0; loop <= MAX_ORDER_RETRIES; loop++) //-- try opening the order until it is successful
     {
      //--- send order to the trade server
      if(OrderSend(tradeRequest, tradeResult))
        {
         //-- Print the order details
         PrintOrderDetails("Sent OK", symbol);

         //-- Confirm order execution
         if(tradeResult.retcode == 10008 || tradeResult.retcode == 10009)
           {
            Print(
               __FUNCTION__, ": CONFIRMED: Successfully openend a ", symbol,
               " ", EnumToString(orderType), " #", tradeResult.order, ", Price: ", tradeResult.price
            );
            PrintFormat("retcode=%u  deal=%I64u  order=%I64u", tradeResult.retcode, tradeResult.deal, tradeResult.order);
            Print("_______________________________________________________________________________________");
            return(true); //-- exit the function
            //break; //--- success - order placed ok. exit the for loop
           }
        }
      else //-- Order request failed
        {
         //-- Print the order details
         PrintOrderDetails("Sending Failed", symbol);

         //-- order not sent or critical error found
         if(!ErrorAdvisor(__FUNCTION__, symbol, tradeResult.retcode) || IsStopped())
           {
            Print(
               __FUNCTION__, ": ", symbol, " ERROR opening a ", EnumToString(orderType),
               " at: ", tradeRequest.price, ", Lot\\Vol: ", tradeRequest.volume
            );
            Print("_______________________________________________________________________________________");
            return(false); //-- exit the function
            //break; //-- exit the for loop

            Sleep(ORDER_RETRY_DELAYS);//-- Small pause before retrying to avoid overwhelming the trade server
           }
        }
     }
```

Here is the complete function with all the code segments in the correct sequence. Ensure that your _OpenBuyLimit(...)_ function contains the following code in full.

```
bool OpenBuyLimit(ulong magicNumber, string symbol, double entryPrice, double lotSize, int sl, int tp, string orderComment) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

   double tpPrice = 0.0, slPrice = 0.0;

//-- Get some information about the orders symbol
   int symbolDigits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   int symbolStopLevel = (int)SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double symbolPoint = SymbolInfoDouble(symbol, SYMBOL_POINT);
   int spread = (int)SymbolInfoInteger(symbol, SYMBOL_SPREAD);

//-- Save the order type enumeration
   ENUM_ORDER_TYPE orderType = ORDER_TYPE_BUY_LIMIT;

//-- check if the entry price is valid
   if(
      SymbolInfoDouble(symbol, SYMBOL_ASK) - (symbolStopLevel * symbolPoint) <
      entryPrice + (spread * symbolPoint)
   )
     {
      Print(
         "\r\n", __FUNCTION__, ": Can't open a new ", EnumToString(orderType),
         ". (Reason --> INVALID ENTRY PRICE: ", DoubleToString(entryPrice, symbolDigits), ")\r\n"
      );
      return(false); //-- Invalid entry price, log the error, exit the function and return false
     }

//-- Check the validity of the sl and tp
   if(sl > 0 && sl < symbolStopLevel)
     {
      sl = symbolStopLevel;
     }
   if(tp > 0 && tp < symbolStopLevel)
     {
      tp = symbolStopLevel;
     }

   slPrice = (sl > 0) ? NormalizeDouble(entryPrice - sl * symbolPoint, symbolDigits) : 0;
   tpPrice = (tp > 0) ? NormalizeDouble(entryPrice + tp * symbolPoint, symbolDigits) : 0;

//-- reset the the tradeRequest and tradeResult values by zeroing them
   ZeroMemory(tradeRequest);
   ZeroMemory(tradeResult);

//-- initialize the parameters to open a buy limit order
   tradeRequest.type = orderType;
   tradeRequest.action = TRADE_ACTION_PENDING;
   tradeRequest.magic = magicNumber;
   tradeRequest.symbol = symbol;
   tradeRequest.price = NormalizeDouble(entryPrice, symbolDigits);
   tradeRequest.tp = tpPrice;
   tradeRequest.sl = slPrice;
   tradeRequest.comment = orderComment;
   tradeRequest.deviation = SymbolInfoInteger(symbol, SYMBOL_SPREAD) * 2;

//-- Set and moderate the lot size or volume
//-- Verify that volume is not less than allowed minimum
   lotSize = MathMax(lotSize, SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN));

//-- Verify that volume is not more than allowed maximum
   lotSize = MathMin(lotSize, SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX));

//-- Round down to nearest volume step
   lotSize = MathFloor(lotSize / SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP)) * SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   tradeRequest.volume = lotSize;

//--- Reset error cache so that we get an accurate runtime error code in the ErrorAdvisor function
   ResetLastError();

   for(int loop = 0; loop <= MAX_ORDER_RETRIES; loop++) //-- try opening the order until it is successful
     {
      //--- send order to the trade server
      if(OrderSend(tradeRequest, tradeResult))
        {
         //-- Print the order details
         PrintOrderDetails("Sent OK", symbol);

         //-- Confirm order execution
         if(tradeResult.retcode == 10008 || tradeResult.retcode == 10009)
           {
            Print(
               __FUNCTION__, ": CONFIRMED: Successfully openend a ", symbol,
               " ", EnumToString(orderType), " #", tradeResult.order, ", Price: ", tradeResult.price
            );
            PrintFormat("retcode=%u  deal=%I64u  order=%I64u", tradeResult.retcode, tradeResult.deal, tradeResult.order);
            Print("_______________________________________________________________________________________");
            return(true); //-- exit the function
            //break; //--- success - order placed ok. exit the for loop
           }
        }
      else //-- Order request failed
        {
         //-- Print the order details
         PrintOrderDetails("Sending Failed", symbol);

         //-- order not sent or critical error found
         if(!ErrorAdvisor(__FUNCTION__, symbol, tradeResult.retcode) || IsStopped())
           {
            Print(
               __FUNCTION__, ": ", symbol, " ERROR opening a ", EnumToString(orderType),
               " at: ", tradeRequest.price, ", Lot\\Vol: ", tradeRequest.volume
            );
            Print("_______________________________________________________________________________________");
            return(false); //-- exit the function
            //break; //-- exit the for loop

            Sleep(ORDER_RETRY_DELAYS);//-- Small pause before retrying to avoid overwhelming the trade server
           }
        }
     }
   return(false);
  }
```

### Open Buy Stop Function

The _OpenBuyStop(...)_ function is responsible for placing new _buy stop_ pending orders. It follows a similar approach to the _OpenBuyLimit(...)_ function described earlier. The function will return _true_ if it successfully places a buy stop order, and _false_ if the order request fails.

#### What Is A Buy Stop Order?

A buy stop order is a request to buy at an _Ask_ price that is equal to or higher than the specified order entry price. The buy stop entry price is valid only when the current market price is lower than the order's entry price. This type of order is useful when you anticipate that the symbol's price will rise to the specified entry level and then continue to move upward, allowing you to profit from the bullish trend.

![Buy Stop Order](https://c.mql5.com/2/96/Buy_Stop_Order_2_70c.png)

Below is the _OpenBuyStop(...)_ function, with explanatory comments added to help you quickly understand how each part of the code works.

```
bool OpenBuyStop(ulong magicNumber, string symbol, double entryPrice, double lotSize, int sl, int tp, string orderComment) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

   double tpPrice = 0.0, slPrice = 0.0;

//-- Get some information about the orders symbol
   int symbolDigits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   int symbolStopLevel = (int)SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double symbolPoint = SymbolInfoDouble(symbol, SYMBOL_POINT);
   int spread = (int)SymbolInfoInteger(symbol, SYMBOL_SPREAD);

//-- Save the order type enumeration
   ENUM_ORDER_TYPE orderType = ORDER_TYPE_BUY_STOP;

//-- check if the entry price is valid
   if(
      SymbolInfoDouble(symbol, SYMBOL_ASK) + (symbolStopLevel * symbolPoint) >
      entryPrice + (spread * symbolPoint)
   )
     {
      Print(
         "\r\n", __FUNCTION__, ": Can't open a new ", EnumToString(orderType),
         ". (Reason --> INVALID ENTRY PRICE: ", DoubleToString(entryPrice, symbolDigits), ")\r\n"
      );
      return(false); //-- Invalid entry price, log the error, exit the function and return false
     }

//--- Validate Stop Loss (sl) and Take Profit (tp)
   if(sl > 0 && sl < symbolStopLevel)
     {
      sl = symbolStopLevel;
     }
   if(tp > 0 && tp < symbolStopLevel)
     {
      tp = symbolStopLevel;
     }

   slPrice = (sl > 0) ? NormalizeDouble(entryPrice - sl * symbolPoint, symbolDigits) : 0;
   tpPrice = (tp > 0) ? NormalizeDouble(entryPrice + tp * symbolPoint, symbolDigits) : 0;

//-- reset the the tradeRequest and tradeResult values by zeroing them
   ZeroMemory(tradeRequest);
   ZeroMemory(tradeResult);

//-- initialize the parameters to open a buy stop order
   tradeRequest.type = orderType;
   tradeRequest.action = TRADE_ACTION_PENDING;
   tradeRequest.magic = magicNumber;
   tradeRequest.symbol = symbol;
   tradeRequest.price = NormalizeDouble(entryPrice, symbolDigits);
   tradeRequest.tp = tpPrice;
   tradeRequest.sl = slPrice;
   tradeRequest.comment = orderComment;
   tradeRequest.deviation = SymbolInfoInteger(symbol, SYMBOL_SPREAD) * 2;

//-- Set and moderate the lot size or volume
//-- Verify that volume is not less than allowed minimum
   lotSize = MathMax(lotSize, SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN));

//-- Verify that volume is not more than allowed maximum
   lotSize = MathMin(lotSize, SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX));

//-- Round down to nearest volume step
   lotSize = MathFloor(lotSize / SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP)) * SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   tradeRequest.volume = lotSize;

//--- Reset error cache so that we get an accurate runtime error code in the ErrorAdvisor function
   ResetLastError();

   for(int loop = 0; loop <= MAX_ORDER_RETRIES; loop++) //-- try opening the order until it is successful
     {
      //--- send order to the trade server
      if(OrderSend(tradeRequest, tradeResult))
        {
         //-- Print the order details
         PrintOrderDetails("Sent OK", symbol);

         //-- Confirm order execution
         if(tradeResult.retcode == 10008 || tradeResult.retcode == 10009)
           {
            Print(
               __FUNCTION__, ": CONFIRMED: Successfully openend a ", symbol,
               " ", EnumToString(orderType), " #", tradeResult.order, ", Price: ", tradeResult.price
            );
            PrintFormat("retcode=%u  deal=%I64u  order=%I64u", tradeResult.retcode, tradeResult.deal, tradeResult.order);
            Print("_______________________________________________________________________________________");
            return(true); //-- exit the function
            //break; //--- success - order placed ok. exit the for loop
           }
        }
      else //-- Order request failed
        {
         //-- Print the order details
         PrintOrderDetails("Sending Failed", symbol);

         //-- order not sent or critical error found
         if(!ErrorAdvisor(__FUNCTION__, symbol, tradeResult.retcode) || IsStopped())
           {
            Print(
               __FUNCTION__, ": ", symbol, " ERROR opening a ", EnumToString(orderType),
               " at: ", tradeRequest.price, ", Lot\\Vol: ", tradeRequest.volume
            );
            Print("_______________________________________________________________________________________");
            return(false); //-- exit the function
            //break; //-- exit the for loop

            Sleep(ORDER_RETRY_DELAYS);//-- Small pause before retrying to avoid overwhelming the trade server
           }
        }
     }
   return(false);
  }
```

### Open Sell Limit Function

The _OpenSellLimit(...)_ function is responsible for placing new _sell limit_ pending orders. The function returns _true_ when a sell limit order is placed successfully, and _false_ if the order request is unsuccessful.

#### What Is A Sell Limit Order?

A sell limit order is a request to sell at a _Bid_ price that is equal to or higher than the specified order entry price. The sell limit entry price is valid only when the current market price is lower than the order's entry price. This type of order is useful when you expect the symbol's price to rise to the specified entry price level and then reverse direction, allowing you to profit from the anticipated downward movement or bearish trend.

![Sell Limit Order](https://c.mql5.com/2/96/Sell_Limit_Order_705.png)

Below is the _OpenSellLimit(...)_ function, complete with explanatory comments to help you better understand how each part of the code operates.

```
bool OpenSellLimit(ulong magicNumber, string symbol, double entryPrice, double lotSize, int sl, int tp, string orderComment) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

   double tpPrice = 0.0, slPrice = 0.0;

//-- Get some information about the orders symbol
   int symbolDigits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   int symbolStopLevel = (int)SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double symbolPoint = SymbolInfoDouble(symbol, SYMBOL_POINT);
   int spread = (int)SymbolInfoInteger(symbol, SYMBOL_SPREAD);

//-- Save the order type enumeration
   ENUM_ORDER_TYPE orderType = ORDER_TYPE_SELL_LIMIT;

//-- check if the entry price is valid
   if(
      SymbolInfoDouble(symbol, SYMBOL_BID) + (symbolStopLevel * symbolPoint) >
      entryPrice - (spread * symbolPoint)
   )
     {
      Print(
         "\r\n", __FUNCTION__, ": Can't open a new ", EnumToString(orderType),
         ". (Reason --> INVALID ENTRY PRICE: ", DoubleToString(entryPrice, symbolDigits), ")\r\n"
      );
      return(false); //-- Invalid entry price, log the error, exit the function and return false
     }

//-- Check the validity of the sl and tp
   if(sl > 0 && sl < symbolStopLevel)
     {
      sl = symbolStopLevel;
     }
   if(tp > 0 && tp < symbolStopLevel)
     {
      tp = symbolStopLevel;
     }

   slPrice = (sl > 0) ? NormalizeDouble(entryPrice + sl * symbolPoint, symbolDigits) : 0;
   tpPrice = (tp > 0) ? NormalizeDouble(entryPrice - tp * symbolPoint, symbolDigits) : 0;

//-- reset the the tradeRequest and tradeResult values by zeroing them
   ZeroMemory(tradeRequest);
   ZeroMemory(tradeResult);

//-- initialize the parameters to open a sell limit order
   tradeRequest.type = orderType;
   tradeRequest.action = TRADE_ACTION_PENDING;
   tradeRequest.magic = magicNumber;
   tradeRequest.symbol = symbol;
   tradeRequest.price = NormalizeDouble(entryPrice, symbolDigits);
   tradeRequest.tp = tpPrice;
   tradeRequest.sl = slPrice;
   tradeRequest.comment = orderComment;
   tradeRequest.deviation = SymbolInfoInteger(symbol, SYMBOL_SPREAD) * 2;

//-- Set and moderate the lot size or volume
//-- Verify that volume is not less than allowed minimum
   lotSize = MathMax(lotSize, SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN));

//-- Verify that volume is not more than allowed maximum
   lotSize = MathMin(lotSize, SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX));

//-- Round down to nearest volume step
   lotSize = MathFloor(lotSize / SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP)) * SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   tradeRequest.volume = lotSize;

//--- Reset error cache so that we get an accurate runtime error code in the ErrorAdvisor function
   ResetLastError();

   for(int loop = 0; loop <= MAX_ORDER_RETRIES; loop++) //-- try opening the order until it is successful
     {
      //--- send order to the trade server
      if(OrderSend(tradeRequest, tradeResult))
        {
         //-- Print the order details
         PrintOrderDetails("Sent OK", symbol);

         //-- Confirm order execution
         if(tradeResult.retcode == 10008 || tradeResult.retcode == 10009)
           {
            Print(
               __FUNCTION__, ": CONFIRMED: Successfully openend a ", symbol,
               " ", EnumToString(orderType), " #", tradeResult.order, ", Price: ", tradeResult.price
            );
            PrintFormat("retcode=%u  deal=%I64u  order=%I64u", tradeResult.retcode, tradeResult.deal, tradeResult.order);
            Print("_______________________________________________________________________________________");
            return(true); //-- exit the function
            //break; //--- success - order placed ok. exit the for loop
           }
        }
      else //-- Order request failed
        {
         //-- Print the order details
         PrintOrderDetails("Sending Failed", symbol);

         //-- order not sent or critical error found
         if(!ErrorAdvisor(__FUNCTION__, symbol, tradeResult.retcode) || IsStopped())
           {
            Print(
               __FUNCTION__, ": ", symbol, " ERROR opening a ", EnumToString(orderType),
               " at: ", tradeRequest.price, ", Lot\\Vol: ", tradeRequest.volume
            );
            Print("_______________________________________________________________________________________");
            return(false); //-- exit the function
            //break; //-- exit the for loop

            Sleep(ORDER_RETRY_DELAYS);//-- Small pause before retrying to avoid overwhelming the trade server
           }
        }
     }
   return(false);
  }
```

### Open Sell Stop Function

The _OpenSellStop(...)_ function is designed to place new _sell stop_ pending orders. The function returns _true_ when a sell stop order is successfully placed, indicating the request was processed without issues. If the order placement fails for any reason, it returns _false_, signaling an unsuccessful attempt.

#### What Is A Sell Stop Order?

A sell stop order is a request to sell at a _Bid_ price that is equal to or lower than the specified order entry price. The sell stop entry price is only valid when the current market price is higher than the order’s entry price. This type of order is useful when you anticipate the price of a symbol will drop to the specified entry level and then continue to decline, allowing you to profit from the expected downward movement or bearish trend.

![Sell Stop Order](https://c.mql5.com/2/96/Sell_Stop_Order_70u.png)

The _OpenSellStop(...)_ function is provided below, along with detailed comments to guide you through the functionality of each part of the code.

```
bool OpenSellStop(ulong magicNumber, string symbol, double entryPrice, double lotSize, int sl, int tp, string orderComment) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

   double slPrice = 0.0, tpPrice = 0.0;

//-- Get some information about the orders symbol
   int symbolDigits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   int symbolStopLevel = (int)SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double symbolPoint = SymbolInfoDouble(symbol, SYMBOL_POINT);
   int spread = (int)SymbolInfoInteger(symbol, SYMBOL_SPREAD);

//-- Save the order type enumeration
   ENUM_ORDER_TYPE orderType = ORDER_TYPE_SELL_STOP;

//-- check if the entry price is valid
   if(
      SymbolInfoDouble(symbol, SYMBOL_BID) - (symbolStopLevel * symbolPoint) <
      entryPrice - (spread * symbolPoint)
   )
     {
      Print(
         "\r\n", __FUNCTION__, ": Can't open a new ", EnumToString(orderType),
         ". (Reason --> INVALID ENTRY PRICE: ", DoubleToString(entryPrice, symbolDigits), ")\r\n"
      );
      return(false); //-- Invalid entry price, log the error, exit the function and return false
     }

//-- Check the validity of the sl and tp
   if(sl > 0 && sl < symbolStopLevel)
     {
      sl = symbolStopLevel;
     }
   if(tp > 0 && tp < symbolStopLevel)
     {
      tp = symbolStopLevel;
     }

   slPrice = (sl > 0) ? NormalizeDouble(entryPrice + sl * symbolPoint, symbolDigits) : 0;
   tpPrice = (tp > 0) ? NormalizeDouble(entryPrice - tp * symbolPoint, symbolDigits) : 0;

//-- reset the the tradeRequest and tradeResult values by zeroing them
   ZeroMemory(tradeRequest);
   ZeroMemory(tradeResult);

//-- initialize the parameters to open a sell stop order
   tradeRequest.type = orderType;
   tradeRequest.action = TRADE_ACTION_PENDING;
   tradeRequest.magic = magicNumber;
   tradeRequest.symbol = symbol;
   tradeRequest.price = NormalizeDouble(entryPrice, symbolDigits);
   tradeRequest.tp = tpPrice;
   tradeRequest.sl = slPrice;
   tradeRequest.comment = orderComment;
   tradeRequest.deviation = SymbolInfoInteger(symbol, SYMBOL_SPREAD) * 2;

//-- Set and moderate the lot size or volume
//-- Verify that volume is not less than allowed minimum
   lotSize = MathMax(lotSize, SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN));

//-- Verify that volume is not more than allowed maximum
   lotSize = MathMin(lotSize, SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX));

//-- Round down to nearest volume step
   lotSize = MathFloor(lotSize / SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP)) * SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   tradeRequest.volume = lotSize;

//--- Reset error cache so that we get an accurate runtime error code in the ErrorAdvisor function
   ResetLastError();

   for(int loop = 0; loop <= MAX_ORDER_RETRIES; loop++) //-- try opening the order until it is successful
     {
      //--- send order to the trade server
      if(OrderSend(tradeRequest, tradeResult))
        {
         //-- Print the order details
         PrintOrderDetails("Sent OK", symbol);

         //-- Confirm order execution
         if(tradeResult.retcode == 10008 || tradeResult.retcode == 10009)
           {
            Print(
               __FUNCTION__, ": CONFIRMED: Successfully openend a ", symbol,
               " ", EnumToString(orderType), " #", tradeResult.order, ", Price: ", tradeResult.price
            );
            PrintFormat("retcode=%u  deal=%I64u  order=%I64u", tradeResult.retcode, tradeResult.deal, tradeResult.order);
            Print("_______________________________________________________________________________________");
            return(true); //-- exit the function
            //break; //--- success - order placed ok. exit the for loop
           }
        }
      else //-- Order request failed
        {
         //-- Print the order details
         PrintOrderDetails("Sending Failed", symbol);

         //-- order not sent or critical error found
         if(!ErrorAdvisor(__FUNCTION__, symbol, tradeResult.retcode) || IsStopped())
           {
            Print(
               __FUNCTION__, ": ", symbol, " ERROR opening a ", EnumToString(orderType),
               " at: ", tradeRequest.price, ", Lot\\Vol: ", tradeRequest.volume
            );
            Print("_______________________________________________________________________________________");
            return(false); //-- exit the function
            //break; //-- exit the for loop

            Sleep(ORDER_RETRY_DELAYS);//-- Small pause before retrying to avoid overwhelming the trade server
           }
        }
     }
   return(false);
  }
```

### Modify Pending Order by Ticket Function

The _ModifyPendingOrderByTicket(...)_ function is designed to modify the stop-loss ( _SL_), take-profit ( _TP_), or _entry price_ of an active pending order. It performs validations to ensure the modifications are within the broker’s rules, logs the progress and any errors that may occur, and uses a retry mechanism to attempt the modification multiple times in case of failure or unfavorable order entry conditions.

The function accepts four parameters:

1. **orderTicket (ulong)**: The unique identifier for the order. This ticket number is used to reference the specific order you want to modify.
2. **newEntryPrice (double)**: The new entry price for the pending order. If the value is set to 0, the function will retain the current entry price.
3. **newSl (int)**: The new stop-loss (SL) value, expressed in points. If the value is set to 0, the SL will either be removed or remain unchanged.
4. **newTp (int)**: The new take-profit (TP) value, expressed in points. If the value is set to 0, the TP will either be removed or remain unchanged.

This function will be of type _bool_ and is expected to return a boolean value of _true_ if the order is successfully modified or a boolean value of _false_ if the modification fails, either due to invalid parameters, errors in selecting the order, or server issues. Let us begin by coding the function definition below.

```
bool ModifyPendingOrderByTicket(ulong orderTicket, double newEntryPrice, int newSl, int newTp) export
  {
```

Before modifying the order, we must ensure that the trading environment allows for algorithmic trading. We will use the _TradingIsAllowed()_ function to confirm this.

```
if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }
```

To make sure we are modifying the correct order, we need to select the order using the standard MQL5 _OrderSelect()_ function and the provided _orderTicket_ function parameter. If the order ticket is not valid or is not successfully selected, we will log this error and exit the function, returning a boolean value of _false_. If order ticket selection is successful, we will print a short log message and proceed with processing the order.

```
//--- Confirm and select the order using the provided orderTicket
   ResetLastError(); //--- Reset error cache incase of ticket selection errors
   if(OrderSelect(orderTicket))
     {
      //---Order selected
      Print("\r\n_______________________________________________________________________________________");
      Print(__FUNCTION__, ": Order with ticket:", orderTicket, " selected and ready to set SLTP.");
     }
   else
     {
      Print("\r\n_______________________________________________________________________________________");
      Print(__FUNCTION__, ": Selecting order with ticket:", orderTicket, " failed. ERROR: ", GetLastError());
      return(false); //-- Exit the function
     }
```

Once the order is selected, we will create some variables of type _double_ to store our _take profit price_ and _stop loss price_ and initialize them with default values of _zero_( _0.0_). Then we will fetch details about the order (like _symbol, volume, current SL/TP,_ etc.) that will be needed for validation and modification.

```
double newTpPrice = 0.0, newSlPrice = 0.0;

//--- Order ticket selected, save the order properties
   string orderSymbol = OrderGetString(ORDER_SYMBOL);
   double currentEntryPrice = OrderGetDouble(ORDER_PRICE_OPEN);
   double volume = OrderGetDouble(ORDER_VOLUME_INITIAL);
   double currentOrderSlPrice = OrderGetDouble(ORDER_SL);
   double currentOrderTpPrice = OrderGetDouble(ORDER_TP);
   ENUM_ORDER_TYPE orderType = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
   double orderPriceCurrent = OrderGetDouble(ORDER_PRICE_CURRENT);

//-- Get some information about the orders symbol
   int symbolDigits = (int)SymbolInfoInteger(orderSymbol, SYMBOL_DIGITS); //-- Number of symbol decimal places
   int symbolStopLevel = (int)SymbolInfoInteger(orderSymbol, SYMBOL_TRADE_STOPS_LEVEL);
   double symbolPoint = SymbolInfoDouble(orderSymbol, SYMBOL_POINT);
   int spread = (int)SymbolInfoInteger(orderSymbol, SYMBOL_SPREAD);
```

Next, we validate the new _entry price_ to ensure it is correct based on the order type (e.g., _buy limit, buy stop, sell stop, and sell limit_). This will ensure that we detect and reject invalid prices early.

```
if(newEntryPrice > 0.0)
     {
      if(orderType == ORDER_TYPE_BUY_STOP || orderType == ORDER_TYPE_SELL_LIMIT)
        {
         if(
            SymbolInfoDouble(orderSymbol, SYMBOL_ASK) + (symbolStopLevel * symbolPoint) >
            newEntryPrice - (spread * symbolPoint)
         )
           {
            Print(
               "\r\n", __FUNCTION__, ": Can't MODIFY ", EnumToString(orderType),
               ". (Reason --> INVALID NEW ENTRY PRICE: ", DoubleToString(newEntryPrice, symbolDigits), ")\r\n"
            );
            return(false); //-- Invalid new entry price, log the error, exit the function and return false
           }
        }

      if(orderType == ORDER_TYPE_BUY_LIMIT || orderType == ORDER_TYPE_SELL_STOP)
        {
         if(
            SymbolInfoDouble(orderSymbol, SYMBOL_BID) - (symbolStopLevel * symbolPoint) <
            newEntryPrice + (spread * symbolPoint)
         )
           {
            Print(
               "\r\n", __FUNCTION__, ": Can't MODIFY ", EnumToString(orderType),
               ". (Reason --> INVALID NEW ENTRY PRICE: ", DoubleToString(newEntryPrice, symbolDigits), ")\r\n"
            );
            return(false); //-- Invalid new entry price, log the error, exit the function and return false
           }
        }
     }
   else
     {
      newEntryPrice = currentEntryPrice; //-- Do not modify the entry price
     }
```

We will also need to calculate the new _stop-loss_ and _take-profit_ prices based on the input parameters and validate these values against the broker's stop level.

```
if(orderType == ORDER_TYPE_BUY_STOP || orderType == ORDER_TYPE_BUY_LIMIT)
     {
      if(newSl == 0)
        {
         newSlPrice = 0.0; //-- Remove the sl
        }
      else
        {
         newSlPrice = newEntryPrice - (newSl * symbolPoint);
        }
      if(newTp == 0)
        {
         newTpPrice = 0.0; //-- Remove the tp
        }
      else
        {
         newTpPrice = newEntryPrice + (newTp * symbolPoint);
        }

      //-- Check the validity of the newSlPrice
      if(newSlPrice > 0 && newEntryPrice - newSlPrice < symbolStopLevel * symbolPoint)
        {
         Print(
            "\r\n", __FUNCTION__, ": Can't modify ", EnumToString(orderType),
            ". (Reason --> INVALID NEW SL PRICE: ", DoubleToString(newSlPrice, symbolDigits), ")\r\n"
         );
         return(false); //-- Invalid sl price, log the error, exit the function and return false
        }

      //-- Check the validity of the newTpPrice
      if(newTpPrice > 0 && newTpPrice - newEntryPrice < symbolStopLevel * symbolPoint)
        {
         Print(
            "\r\n", __FUNCTION__, ": Can't modify ", EnumToString(orderType),
            ". (Reason --> INVALID NEW TP PRICE: ", DoubleToString(newTpPrice, symbolDigits), ")\r\n"
         );
         return(false); //-- Invalid tp price, log the error, exit the function and return false
        }
     }

   if(orderType == ORDER_TYPE_SELL_STOP || orderType == ORDER_TYPE_SELL_LIMIT)
     {
      if(newSl == 0)
        {
         newSlPrice = 0.0; //-- Remove the sl
        }
      else
        {
         newSlPrice = newEntryPrice + (newSl * symbolPoint);
        }
      if(newTp == 0)
        {
         newTpPrice = 0.0; //-- Remove the tp
        }
      else
        {
         newTpPrice = newEntryPrice - (newTp * symbolPoint);
        }

      //-- Check the validity of the newSlPrice
      if(newSlPrice > 0 && newSlPrice - newEntryPrice < symbolStopLevel * symbolPoint)
        {
         Print(
            "\r\n", __FUNCTION__, ": Can't modify ", EnumToString(orderType),
            ". (Reason --> INVALID NEW SL PRICE: ", DoubleToString(newSlPrice, symbolDigits), ")\r\n"
         );
         return(false); //-- Invalid sl price, log the error, exit the function and return false
        }

      //-- Check the validity of the newTpPrice
      if(newTpPrice > 0 && newEntryPrice - newTpPrice < symbolStopLevel * symbolPoint)
        {
         Print(
            "\r\n", __FUNCTION__, ": Can't modify ", EnumToString(orderType),
            ". (Reason --> INVALID NEW TP PRICE: ", DoubleToString(newTpPrice, symbolDigits), ")\r\n"
         );
         return(false); //-- Invalid tp price, log the error, exit the function and return false
        }
     }
```

Since we have now validated all the values, we will proceed by preparing the trade request to modify the pending order by setting the appropriate parameters. We will first reset the _tradeRequest_ data structure and then initialize it with our verified and valid order data.

```
   ZeroMemory(tradeRequest);
   ZeroMemory(tradeResult);

//-- initialize the parameters to set the sltp
   tradeRequest.action = TRADE_ACTION_MODIFY; //-- Trade operation type for modifying the pending order
   tradeRequest.order = orderTicket;
   tradeRequest.symbol = orderSymbol;
   tradeRequest.price = newEntryPrice;
   tradeRequest.sl = newSlPrice;
   tradeRequest.tp = newTpPrice;
   tradeRequest.deviation = SymbolInfoInteger(orderSymbol, SYMBOL_SPREAD) * 2;
```

Next, we implement a retry mechanism to resend the modification request multiple times if the initial attempt fails, increasing the likelihood of successfully modifying the order. In case of a critical error or if all retry attempts are exhausted, the function will return _false_ and exit. If the order is successfully modified, the function will return _true_ and exit. To avoid overwhelming the trade server with a rapid series of consecutive requests, we will also introduce a delay between iterations using the _Sleep(...)_ function.

```
ResetLastError(); //--- reset error cache so that we get an accurate runtime error code in the ErrorAdvisor function

   for(int loop = 0; loop <= MAX_ORDER_RETRIES; loop++) //-- try modifying the price open, sl, and tp until the request is successful
     {
      //--- send order to the trade server
      if(OrderSend(tradeRequest, tradeResult))
        {
         //-- Confirm order execution
         if(tradeResult.retcode == 10008 || tradeResult.retcode == 10009)
           {
            PrintFormat("Successfully modified Pending Order: #%I64d %s %s", orderTicket, orderSymbol, EnumToString(orderType));
            PrintFormat("retcode=%u  runtime_code=%u", tradeResult.retcode, GetLastError());
            Print("_______________________________________________________________________________________\r\n\r\n");
            return(true); //-- exit function
            //break; //--- success - order placed ok. exit for loop
           }
        }
      else  //-- Order request failed
        {
         //-- order not sent or critical error found
         if(!ErrorAdvisor(__FUNCTION__, orderSymbol, tradeResult.retcode) || IsStopped())
           {
            PrintFormat("ERROR modifying Pending Order: #%I64d %s %s", orderTicket, orderSymbol, EnumToString(orderType));
            Print("_______________________________________________________________________________________\r\n\r\n");
            return(false); //-- exit function
            //break; //-- exit for loop

            Sleep(ORDER_RETRY_DELAYS);//-- Small pause before retrying to avoid overwhelming the trade server
           }
        }
     }
```

Here is the full function code including some of the missing parts with all the code segments in their correct order.

```
bool ModifyPendingOrderByTicket(ulong orderTicket, double newEntryPrice, int newSl, int newTp) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

//--- Confirm and select the order using the provided orderTicket
   ResetLastError(); //--- Reset error cache incase of ticket selection errors
   if(OrderSelect(orderTicket))
     {
      //---Order selected
      Print("\r\n_______________________________________________________________________________________");
      Print(__FUNCTION__, ": Order with ticket:", orderTicket, " selected and ready to set SLTP.");
     }
   else
     {
      Print("\r\n_______________________________________________________________________________________");
      Print(__FUNCTION__, ": Selecting order with ticket:", orderTicket, " failed. ERROR: ", GetLastError());
      return(false); //-- Exit the function
     }

//-- create variables to store the calculated tp and sl prices to send to the trade server
   double newTpPrice = 0.0, newSlPrice = 0.0;

//--- Order ticket selected, save the order properties
   string orderSymbol = OrderGetString(ORDER_SYMBOL);
   double currentEntryPrice = OrderGetDouble(ORDER_PRICE_OPEN);
   double volume = OrderGetDouble(ORDER_VOLUME_INITIAL);
   double currentOrderSlPrice = OrderGetDouble(ORDER_SL);
   double currentOrderTpPrice = OrderGetDouble(ORDER_TP);
   ENUM_ORDER_TYPE orderType = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
   double orderPriceCurrent = OrderGetDouble(ORDER_PRICE_CURRENT);

//-- Get some information about the orders symbol
   int symbolDigits = (int)SymbolInfoInteger(orderSymbol, SYMBOL_DIGITS); //-- Number of symbol decimal places
   int symbolStopLevel = (int)SymbolInfoInteger(orderSymbol, SYMBOL_TRADE_STOPS_LEVEL);
   double symbolPoint = SymbolInfoDouble(orderSymbol, SYMBOL_POINT);
   int spread = (int)SymbolInfoInteger(orderSymbol, SYMBOL_SPREAD);

//-- Check the validity of the newEntryPrice
   if(newEntryPrice > 0.0)
     {
      if(orderType == ORDER_TYPE_BUY_STOP || orderType == ORDER_TYPE_SELL_LIMIT)
        {
         if(
            SymbolInfoDouble(orderSymbol, SYMBOL_ASK) + (symbolStopLevel * symbolPoint) >
            newEntryPrice - (spread * symbolPoint)
         )
           {
            Print(
               "\r\n", __FUNCTION__, ": Can't MODIFY ", EnumToString(orderType),
               ". (Reason --> INVALID NEW ENTRY PRICE: ", DoubleToString(newEntryPrice, symbolDigits), ")\r\n"
            );
            return(false); //-- Invalid new entry price, log the error, exit the function and return false
           }
        }

      if(orderType == ORDER_TYPE_BUY_LIMIT || orderType == ORDER_TYPE_SELL_STOP)
        {
         if(
            SymbolInfoDouble(orderSymbol, SYMBOL_BID) - (symbolStopLevel * symbolPoint) <
            newEntryPrice + (spread * symbolPoint)
         )
           {
            Print(
               "\r\n", __FUNCTION__, ": Can't MODIFY ", EnumToString(orderType),
               ". (Reason --> INVALID NEW ENTRY PRICE: ", DoubleToString(newEntryPrice, symbolDigits), ")\r\n"
            );
            return(false); //-- Invalid new entry price, log the error, exit the function and return false
           }
        }
     }
   else
     {
      newEntryPrice = currentEntryPrice; //-- Do not modify the entry price
     }

//-- Calculate and store the non-validated sl and tp prices
   if(orderType == ORDER_TYPE_BUY_STOP || orderType == ORDER_TYPE_BUY_LIMIT)
     {
      if(newSl == 0)
        {
         newSlPrice = 0.0; //-- Remove the sl
        }
      else
        {
         newSlPrice = newEntryPrice - (newSl * symbolPoint);
        }
      if(newTp == 0)
        {
         newTpPrice = 0.0; //-- Remove the tp
        }
      else
        {
         newTpPrice = newEntryPrice + (newTp * symbolPoint);
        }

      //-- Check the validity of the newSlPrice
      if(newSlPrice > 0 && newEntryPrice - newSlPrice < symbolStopLevel * symbolPoint)
        {
         Print(
            "\r\n", __FUNCTION__, ": Can't modify ", EnumToString(orderType),
            ". (Reason --> INVALID NEW SL PRICE: ", DoubleToString(newSlPrice, symbolDigits), ")\r\n"
         );
         return(false); //-- Invalid sl price, log the error, exit the function and return false
        }

      //-- Check the validity of the newTpPrice
      if(newTpPrice > 0 && newTpPrice - newEntryPrice < symbolStopLevel * symbolPoint)
        {
         Print(
            "\r\n", __FUNCTION__, ": Can't modify ", EnumToString(orderType),
            ". (Reason --> INVALID NEW TP PRICE: ", DoubleToString(newTpPrice, symbolDigits), ")\r\n"
         );
         return(false); //-- Invalid tp price, log the error, exit the function and return false
        }
     }

   if(orderType == ORDER_TYPE_SELL_STOP || orderType == ORDER_TYPE_SELL_LIMIT)
     {
      if(newSl == 0)
        {
         newSlPrice = 0.0; //-- Remove the sl
        }
      else
        {
         newSlPrice = newEntryPrice + (newSl * symbolPoint);
        }
      if(newTp == 0)
        {
         newTpPrice = 0.0; //-- Remove the tp
        }
      else
        {
         newTpPrice = newEntryPrice - (newTp * symbolPoint);
        }

      //-- Check the validity of the newSlPrice
      if(newSlPrice > 0 && newSlPrice - newEntryPrice < symbolStopLevel * symbolPoint)
        {
         Print(
            "\r\n", __FUNCTION__, ": Can't modify ", EnumToString(orderType),
            ". (Reason --> INVALID NEW SL PRICE: ", DoubleToString(newSlPrice, symbolDigits), ")\r\n"
         );
         return(false); //-- Invalid sl price, log the error, exit the function and return false
        }

      //-- Check the validity of the newTpPrice
      if(newTpPrice > 0 && newEntryPrice - newTpPrice < symbolStopLevel * symbolPoint)
        {
         Print(
            "\r\n", __FUNCTION__, ": Can't modify ", EnumToString(orderType),
            ". (Reason --> INVALID NEW TP PRICE: ", DoubleToString(newTpPrice, symbolDigits), ")\r\n"
         );
         return(false); //-- Invalid tp price, log the error, exit the function and return false
        }
     }

//-- Print order properties before modification
   string orderProperties = "--> "  + orderSymbol + " " + EnumToString(orderType) + " SLTP Modification Details" +
                            " <--\r\n";
   orderProperties += "------------------------------------------------------------\r\n";
   orderProperties += "Ticket: " + (string)orderTicket + "\r\n";
   orderProperties += "Volume: " + DoubleToString(volume, symbolDigits) + "\r\n";
   orderProperties += "Price Open: " + DoubleToString(currentEntryPrice, symbolDigits) +
                      "   -> New Proposed Price Open: " + DoubleToString(newEntryPrice, symbolDigits) + "\r\n";
   orderProperties += "Current SL: " + DoubleToString(currentOrderSlPrice, symbolDigits) +
                      "   -> New Proposed SL: " + DoubleToString(newSlPrice, symbolDigits) + "\r\n";
   orderProperties += "Current TP: " + DoubleToString(currentOrderTpPrice, symbolDigits) +
                      "   -> New Proposed TP: " + DoubleToString(newTpPrice, symbolDigits) + "\r\n";
   orderProperties += "Comment: " + OrderGetString(ORDER_COMMENT) + "\r\n";
   orderProperties += "Magic Number: " + (string)OrderGetInteger(ORDER_MAGIC) + "\r\n";
   orderProperties += "---";

//-- Print verified order properties before modification
   orderProperties += "--> Validated and Confirmed NewEntry, SL, and TP Prices: <--\r\n";
   orderProperties += "Order Price Current: " + DoubleToString(orderPriceCurrent, symbolDigits) + "\r\n";
   orderProperties += "Current Entry Price: " + DoubleToString(currentEntryPrice, symbolDigits) +
                      ", New Entry Price: " + DoubleToString(newEntryPrice, symbolDigits) + "\r\n";
   orderProperties += "Current SL: " + DoubleToString(currentOrderSlPrice, symbolDigits) +
                      "   -> New SL: " + DoubleToString(newSlPrice, symbolDigits) + "\r\n";
   orderProperties += "Current TP: " + DoubleToString(currentOrderTpPrice, symbolDigits) +
                      "   -> New TP: " + DoubleToString(newTpPrice, symbolDigits) + "\r\n";
   Print(orderProperties);

//-- reset the the tradeRequest and tradeResult values by zeroing them
   ZeroMemory(tradeRequest);
   ZeroMemory(tradeResult);

//-- initialize the parameters to set the sltp
   tradeRequest.action = TRADE_ACTION_MODIFY; //-- Trade operation type for modifying the pending order
   tradeRequest.order = orderTicket;
   tradeRequest.symbol = orderSymbol;
   tradeRequest.price = newEntryPrice;
   tradeRequest.sl = newSlPrice;
   tradeRequest.tp = newTpPrice;
   tradeRequest.deviation = SymbolInfoInteger(orderSymbol, SYMBOL_SPREAD) * 2;

   ResetLastError(); //--- reset error cache so that we get an accurate runtime error code in the ErrorAdvisor function

   for(int loop = 0; loop <= MAX_ORDER_RETRIES; loop++) //-- try modifying the price open, sl, and tp until the request is successful
     {
      //--- send order to the trade server
      if(OrderSend(tradeRequest, tradeResult))
        {
         //-- Confirm order execution
         if(tradeResult.retcode == 10008 || tradeResult.retcode == 10009)
           {
            PrintFormat("Successfully modified Pending Order: #%I64d %s %s", orderTicket, orderSymbol, EnumToString(orderType));
            PrintFormat("retcode=%u  runtime_code=%u", tradeResult.retcode, GetLastError());
            Print("_______________________________________________________________________________________\r\n\r\n");
            return(true); //-- exit function
            //break; //--- success - order placed ok. exit for loop
           }
        }
      else  //-- Order request failed
        {
         //-- order not sent or critical error found
         if(!ErrorAdvisor(__FUNCTION__, orderSymbol, tradeResult.retcode) || IsStopped())
           {
            PrintFormat("ERROR modifying Pending Order: #%I64d %s %s", orderTicket, orderSymbol, EnumToString(orderType));
            Print("_______________________________________________________________________________________\r\n\r\n");
            return(false); //-- exit function
            //break; //-- exit for loop

            Sleep(ORDER_RETRY_DELAYS);//-- Small pause before retrying to avoid overwhelming the trade server
           }
        }
     }
   return(false);
  }
```

### Delete Pending Order by Ticket Function

The _DeletePendingOrderByTicket(...)_ function is responsible for deleting a pending order using its _unique ticket number_. It first checks whether algorithmic trading is enabled, selects the order, and then attempts to delete it from the trade server. Throughout the process, the function logs important details about the order at different stages and employs a retry mechanism to maximize the chances of successful deletion, retrying multiple times if necessary.

The function accepts one parameter of type _ulong_, which is the _orderTicket_ representing the unique ticket number of the order to be deleted. It returns a _bool_, where _true_ indicates the pending order was successfully deleted, and _false_ indicates the deletion failed, or a critical error was encountered.

Below is the full function code, with detailed comments to ensure easier understanding.

```
bool DeletePendingOrderByTicket(ulong orderTicket) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

//--- Confirm and select the order using the provided orderTicket
   ResetLastError(); //--- Reset error cache incase of ticket selection errors
   if(OrderSelect(orderTicket))
     {
      //---Order selected
      Print("...........................................................................................");
      Print(__FUNCTION__, ": Order with ticket:", orderTicket, " selected and ready to be deleted.");
     }
   else
     {
      Print("...........................................................................................");
      Print(__FUNCTION__, ": Selecting order with ticket:", orderTicket, " failed. ERROR: ", GetLastError());
      return(false); //-- Exit the function
     }

//--- Order ticket selected, save the order properties
   string orderSymbol = OrderGetString(ORDER_SYMBOL);
   double orderVolume = OrderGetDouble(ORDER_VOLUME_CURRENT);
   int symbolDigits = (int)SymbolInfoInteger(orderSymbol, SYMBOL_DIGITS);
   ENUM_ORDER_TYPE orderType = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);

//-- Print order properties before deleting it
   string orderProperties;
   orderProperties += "-- "  + orderSymbol + " " + EnumToString(orderType) + " Details" +
   " -------------------------------------------------------------\r\n";
   orderProperties += "Ticket: " + (string)orderTicket + "\r\n";
   orderProperties += "Volume: " + DoubleToString(orderVolume) + "\r\n";
   orderProperties += "Price Open: " + DoubleToString(OrderGetDouble(ORDER_PRICE_OPEN), symbolDigits) + "\r\n";
   orderProperties += "SL: " + DoubleToString(OrderGetDouble(ORDER_SL), symbolDigits) + "\r\n";
   orderProperties += "TP: " + DoubleToString(OrderGetDouble(ORDER_TP), symbolDigits) + "\r\n";
   orderProperties += "Comment: " + OrderGetString(ORDER_COMMENT) + "\r\n";
   orderProperties += "Magic Number: " + (string)OrderGetInteger(ORDER_MAGIC) + "\r\n";
   orderProperties += "_______________________________________________________________________________________";
   Print(orderProperties);

//-- reset the the tradeRequest and tradeResult values by zeroing them
   ZeroMemory(tradeRequest);
   ZeroMemory(tradeResult);

//-- initialize the trade reqiest parameters to delete the order
   tradeRequest.action = TRADE_ACTION_REMOVE; //-- Trade operation type for deleting an order
   tradeRequest.order = orderTicket;

   ResetLastError(); //--- reset error cache so that we get an accurate runtime error code in the ErrorAdvisor function

   for(int loop = 0; loop <= MAX_ORDER_RETRIES; loop++) //-- try deleting the order until the request is successful
     {
      //--- send order to the trade server
      if(OrderSend(tradeRequest, tradeResult))
        {
         //-- Confirm order execution
         if(tradeResult.retcode == 10008 || tradeResult.retcode == 10009)
           {
            Print(__FUNCTION__, "_________________________________________________________________________");
            PrintFormat("Successfully deleted order #%I64d %s %s", orderTicket, orderSymbol, EnumToString(orderType));
            PrintFormat("retcode=%u  runtime_code=%u", tradeResult.retcode, GetLastError());
            Print("_______________________________________________________________________________________");
            return(true); //-- exit function
            //break; //--- success - order placed ok. exit for loop
           }
        }
      else  //-- order deleting request failed
        {
         //-- order not sent or critical error found
         if(!ErrorAdvisor(__FUNCTION__, orderSymbol, tradeResult.retcode) || IsStopped())
           {
            Print(__FUNCTION__, "_________________________________________________________________________");
            PrintFormat("ERROR deleting order #%I64d %s %s", orderTicket, orderSymbol, EnumToString(orderType));
            Print("_______________________________________________________________________________________");
            return(false); //-- exit function
            //break; //-- exit for loop

            Sleep(ORDER_RETRY_DELAYS);//-- Small pause before retrying to avoid overwhelming the trade server
           }
        }
     }
   return(false);
  }
```

### Delete All Pending Orders Function

The _DeleteAllPendingOrders(...)_ function is designed to delete all pending orders for a given _symbol_ and _magic number_. It first ensures that algorithmic trading is enabled before proceeding. The function will iterate through all open orders, check if they match the specified _symbol_ and _magic number_, and attempt to delete them one by one. In cases where some orders cannot be immediately deleted, the function uses a callback and retry mechanism to continuously attempt deletion until all targeted orders are removed. Throughout the process, it logs errors and checks for critical issues to avoid potential lockups in an infinite loop.

The function accepts two optional parameters:

1. **symbol (string)**: A _string_ parameter that represents the trading _symbol_. It defaults to _ALL\_SYMBOLS_, meaning it will target orders for _all symbols_ unless a specific _symbol_ is provided.
2. **magicNumber (ulong)**: An _unsigned long integer_ parameter representing the _unique magic number_ of the orders. It defaults to _0_, meaning the function will target _all orders_ regardless of their _magic number_ unless a specific value is given.

The function returns a _boolean_ value ( _bool_), where _true_ indicates that all the targeted pending orders were successfully deleted, and _false_ indicates a failure to delete some or all orders or when a critical error was encountered.

Below is the full function code, with comments added for easier understanding:

```
bool DeleteAllPendingOrders(string symbol = ALL_SYMBOLS, ulong magicNumber = 0) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

   bool returnThis = false;

//-- Scan for symbol and magic number for the specified orders and close them
   int totalOpenOrders = OrdersTotal();
   for(int x = 0; x < totalOpenOrders; x++)
     {
      //--- Get order properties
      ulong orderTicket = OrderGetTicket(x); //-- Get ticket to select the order
      string selectedSymbol = OrderGetString(ORDER_SYMBOL);
      ulong orderMagicNo = OrderGetInteger(ORDER_MAGIC);

      //-- Filter orders by symbol and magic number
      if(
         (symbol != ALL_SYMBOLS && symbol != selectedSymbol) ||
         (magicNumber != 0 && orderMagicNo != magicNumber)
      )
        {
         continue;
        }

      //-- Delete the order
      DeletePendingOrderByTicket(orderTicket);
     }

//-- Confirm that we have closed all the orders being targeted
   int breakerBreaker = 0; //-- Variable that safeguards and makes sure we are not locked in an infinite loop
   while(SymbolOrdersTotal(symbol, magicNumber) > 0)
     {
      breakerBreaker++;
      DeleteAllPendingOrders(symbol, magicNumber); //-- We still have some open orders, do a function callback
      Sleep(ORDER_RETRY_DELAYS); //-- Micro sleep to pace the execution and give some time to the trade server

      //-- Check for critical errors so that we exit the loop if we run into trouble
      if(!ErrorAdvisor(__FUNCTION__, symbol, GetLastError()) || IsStopped() || breakerBreaker > MAX_ORDER_RETRIES)
        {
         break;
        }
     }

//-- Final confirmations that all targeted orders have been closed
   if(SymbolOrdersTotal(symbol, magicNumber) == 0)
     {
      returnThis = true; //-- Save this status for the function return value
     }

   return(returnThis);
  }
```

To make deleting all pending orders quicker and less tedious, we will overload the _DeleteAllPendingOrders(...)_ function with a second version that takes no arguments. When called, it will delete all pending orders in the account. Below is the overloaded DeleteAllPendingOrders() function:

```
bool DeleteAllPendingOrders() export
  {
   return(DeleteAllPendingOrders(ALL_SYMBOLS, 0));
  }
```

### Delete All Buy Stops Function

The _DeleteAllBuyStops(...)_ function is responsible for deleting all pending buy stop orders for a given _symbol_ and _magic number_. Similar to other deletion functions, it first checks if algorithmic trading is enabled. The function then scans through all open orders, identifies those that match the _symbol, magic number_, and the _order type_ ( _buy stop_), and attempts to delete them. If any orders are not deleted during the initial pass, the function uses a callback and retry mechanism to continuously attempt deletion. It ensures that no infinite loops occur by setting safeguards and checks for errors. Throughout the process, the function logs relevant information and manages critical errors.

The function accepts two optional parameters:

1. **symbol (string)**: A _string_ representing the trading symbol. It defaults to _ALL\_SYMBOLS_, meaning it will target _buy stop orders_ across _all symbols_ unless a specific symbol is provided.
2. **magicNumber (ulong)**: An _unsigned long integer_ representing the _unique magic number_ assigned to the orders. It defaults to _0_, meaning the function will _target all buy stop orders_ regardless of their _magic number_ unless a specific _magic number_ is specified.

The function returns a _boolean_ value ( _bool_), where _true_ indicates that all targeted _buy stop orders_ were successfully deleted, and _false_ indicates that the function failed to delete some or all of the orders or encountered a critical issue.

Below is the full function code with helpful comments to make understanding easier:

```
bool DeleteAllBuyStops(string symbol = ALL_SYMBOLS, ulong magicNumber = 0) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

   bool returnThis = false;

//-- Scan for symbol and magic number specific buy stop orders and close them
   int totalOpenOrders = OrdersTotal();
   for(int x = 0; x < totalOpenOrders; x++)
     {
      //--- Get order properties
      ulong orderTicket = OrderGetTicket(x); //-- Get ticket to select the order
      string selectedSymbol = OrderGetString(ORDER_SYMBOL);
      ulong orderMagicNo = OrderGetInteger(ORDER_MAGIC);
      ulong orderType = OrderGetInteger(ORDER_TYPE);

      //-- Filter order by symbol, type and magic number
      if(
         (symbol != ALL_SYMBOLS && symbol != selectedSymbol) || (orderType != ORDER_TYPE_BUY_STOP) ||
         (magicNumber != 0 && orderMagicNo != magicNumber)
      )
        {
         continue;
        }

      //-- Close the order
      DeletePendingOrderByTicket(orderTicket);
     }

//-- Confirm that we have closed all the buy stop orders being targeted
   int breakerBreaker = 0; //-- Variable that safeguards and makes sure we are not locked in an infinite loop
   while(SymbolBuyStopOrdersTotal(symbol, magicNumber) > 0)
     {
      breakerBreaker++;
      DeleteAllBuyStops(symbol, magicNumber); //-- We still have some open buy stop orders, do a function callback
      Sleep(ORDER_RETRY_DELAYS); //-- Micro sleep to pace the execution and give some time to the trade server

      //-- Check for critical errors so that we exit the loop if we run into trouble
      if(!ErrorAdvisor(__FUNCTION__, symbol, GetLastError()) || IsStopped() || breakerBreaker > MAX_ORDER_RETRIES)
        {
         break;
        }
     }

   if(SymbolBuyStopOrdersTotal(symbol, magicNumber) == 0)
     {
      returnThis = true;
     }
   return(returnThis);
  }
```

### Delete All Buy Limits Function

The _DeleteAllBuyLimits(...)_ function is tasked with deleting all pending _buy limit orders_ for a given _symbol_ and _magic number_. The function returns a _boolean_ value ( _bool_), where _true_ indicates that all targeted _buy limit orders_ were successfully deleted, and _false_ indicates that the function failed to delete some or all of the orders or encountered a critical issue.

```
bool DeleteAllBuyLimits(string symbol = ALL_SYMBOLS, ulong magicNumber = 0) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

   bool returnThis = false;

//-- Scan for symbol and magic number specific buy limit orders and close them
   int totalOpenOrders = OrdersTotal();
   for(int x = 0; x < totalOpenOrders; x++)
     {
      //--- Get order properties
      ulong orderTicket = OrderGetTicket(x); //-- Get ticket to select the order
      string selectedSymbol = OrderGetString(ORDER_SYMBOL);
      ulong orderMagicNo = OrderGetInteger(ORDER_MAGIC);
      ulong orderType = OrderGetInteger(ORDER_TYPE);

      //-- Filter order by symbol, type and magic number
      if(
         (symbol != ALL_SYMBOLS && symbol != selectedSymbol) || (orderType != ORDER_TYPE_BUY_LIMIT) ||
         (magicNumber != 0 && orderMagicNo != magicNumber)
      )
        {
         continue;
        }

      //-- Close the order
      DeletePendingOrderByTicket(orderTicket);
     }

//-- Confirm that we have closed all the buy limit orders being targeted
   int breakerBreaker = 0; //-- Variable that safeguards and makes sure we are not locked in an infinite loop
   while(SymbolBuyLimitOrdersTotal(symbol, magicNumber) > 0)
     {
      breakerBreaker++;
      DeleteAllBuyLimits(symbol, magicNumber); //-- We still have some open buy limit orders, do a function callback
      Sleep(ORDER_RETRY_DELAYS); //-- Micro sleep to pace the execution and give some time to the trade server

      //-- Check for critical errors so that we exit the loop if we run into trouble
      if(!ErrorAdvisor(__FUNCTION__, symbol, GetLastError()) || IsStopped() || breakerBreaker > MAX_ORDER_RETRIES)
        {
         break;
        }
     }

   if(SymbolBuyLimitOrdersTotal(symbol, magicNumber) == 0)
     {
      returnThis = true;
     }
   return(returnThis);
  }
```

### Delete All Sell Stops Function

The _DeleteAllSellStops(...)_ function is responsible for deleting all pending _sell stop orders_ associated with a specific _symbol_ and _magic number_. It returns a _boolean_ value ( _bool_), where _true_ signifies that all targeted _sell stop orders_ were deleted successfully, and _false_ indicates that the function either failed to delete some or all of the orders or encountered a critical error.

```
bool DeleteAllSellStops(string symbol = ALL_SYMBOLS, ulong magicNumber = 0) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

   bool returnThis = false;

//-- Scan for symbol and magic number specific sell stop orders and close them
   int totalOpenOrders = OrdersTotal();
   for(int x = 0; x < totalOpenOrders; x++)
     {
      //--- Get order properties
      ulong orderTicket = OrderGetTicket(x); //-- Get ticket to select the order
      string selectedSymbol = OrderGetString(ORDER_SYMBOL);
      ulong orderMagicNo = OrderGetInteger(ORDER_MAGIC);
      ulong orderType = OrderGetInteger(ORDER_TYPE);

      //-- Filter order by symbol, type and magic number
      if(
         (symbol != ALL_SYMBOLS && symbol != selectedSymbol) || (orderType != ORDER_TYPE_SELL_STOP) ||
         (magicNumber != 0 && orderMagicNo != magicNumber)
      )
        {
         continue;
        }

      //-- Close the order
      DeletePendingOrderByTicket(orderTicket);
     }

//-- Confirm that we have closed all the sell stop orders being targeted
   int breakerBreaker = 0; //-- Variable that safeguards and makes sure we are not locked in an infinite loop
   while(SymbolSellStopOrdersTotal(symbol, magicNumber) > 0)
     {
      breakerBreaker++;
      DeleteAllSellStops(symbol, magicNumber); //-- We still have some open sell stop orders, do a function callback
      Sleep(ORDER_RETRY_DELAYS); //-- Micro sleep to pace the execution and give some time to the trade server

      //-- Check for critical errors so that we exit the loop if we run into trouble
      if(!ErrorAdvisor(__FUNCTION__, symbol, GetLastError()) || IsStopped() || breakerBreaker > MAX_ORDER_RETRIES)
        {
         break;
        }
     }

   if(SymbolSellStopOrdersTotal(symbol, magicNumber) == 0)
     {
      returnThis = true;
     }
   return(returnThis);
  }
```

### Delete All Sell Limits Function

The _DeleteAllSellLimits(...)_ function handles the deletion of all pending _sell limit orders_ linked to a particular _symbol_ and _magic number_. It returns a boolean ( _bool_) value of _true_ if all specified _sell limit orders_ were successfully removed, and _false_ if the function was unable to delete some or all of the orders or ran into a critical error.

```
bool DeleteAllSellLimits(string symbol = ALL_SYMBOLS, ulong magicNumber = 0) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

   bool returnThis = false;

//-- Scan for symbol and magic number specific sell limit orders and close them
   int totalOpenOrders = OrdersTotal();
   for(int x = 0; x < totalOpenOrders; x++)
     {
      //--- Get order properties
      ulong orderTicket = OrderGetTicket(x); //-- Get ticket to select the order
      string selectedSymbol = OrderGetString(ORDER_SYMBOL);
      ulong orderMagicNo = OrderGetInteger(ORDER_MAGIC);
      ulong orderType = OrderGetInteger(ORDER_TYPE);

      //-- Filter order by symbol, type and magic number
      if(
         (symbol != ALL_SYMBOLS && symbol != selectedSymbol) || (orderType != ORDER_TYPE_SELL_LIMIT) ||
         (magicNumber != 0 && orderMagicNo != magicNumber)
      )
        {
         continue;
        }

      //-- Close the order
      DeletePendingOrderByTicket(orderTicket);
     }

//-- Confirm that we have closed all the sell limit orders being targeted
   int breakerBreaker = 0; //-- Variable that safeguards and makes sure we are not locked in an infinite loop
   while(SymbolSellLimitOrdersTotal(symbol, magicNumber) > 0)
     {
      breakerBreaker++;
      DeleteAllSellLimits(symbol, magicNumber); //-- We still have some open sell limit orders, do a function callback
      Sleep(ORDER_RETRY_DELAYS); //-- Micro sleep to pace the execution and give some time to the trade server

      //-- Check for critical errors so that we exit the loop if we run into trouble
      if(!ErrorAdvisor(__FUNCTION__, symbol, GetLastError()) || IsStopped() || breakerBreaker > MAX_ORDER_RETRIES)
        {
         break;
        }
     }

   if(SymbolSellLimitOrdersTotal(symbol, magicNumber) == 0)
     {
      returnThis = true;
     }
   return(returnThis);
  }
```

### Delete All Magic Orders Function

The _DeleteAllMagicOrders(...)_ function is responsible for deleting all pending orders associated with a specific _magic number_. It returns a boolean ( _bool_), with _true_ indicating that all relevant magic orders were deleted successfully, and _false_ indicating that the function either failed to delete some or all of the orders or encountered a critical error.

```
bool DeleteAllMagicOrders(ulong magicNumber) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

   bool returnThis = false;

//-- Variables to store the selected orders data
   ulong orderTicket, orderMagicNo;
   string orderSymbol;

//-- Scan for magic number specific orders and delete them
   int totalOpenOrders = OrdersTotal();
   for(int x = 0; x < totalOpenOrders; x++)
     {
      //--- Get order properties
      orderTicket = OrderGetTicket(x); //-- Get ticket to select the order
      orderMagicNo = OrderGetInteger(ORDER_MAGIC);
      orderSymbol = OrderGetString(ORDER_SYMBOL);

      //-- Filter orders by magic number
      if(magicNumber == orderMagicNo)
        {
         //-- Delete the order
         DeletePendingOrderByTicket(orderTicket);
        }
     }

//-- Confirm that we have deleted all the orders being targeted
   int breakerBreaker = 0; //-- Variable that safeguards and makes sure we are not locked in an infinite loop
   while(MagicOrdersTotal(magicNumber) > 0)
     {
      breakerBreaker++;
      DeleteAllMagicOrders(magicNumber); //-- We still have some open orders, do a function callback
      Sleep(ORDER_RETRY_DELAYS); //-- Micro sleep to pace the execution and give some time to the trade server

      //-- Check for critical errors so that we exit the loop if we run into trouble
      if(!ErrorAdvisor(__FUNCTION__, orderSymbol, GetLastError()) || IsStopped() || breakerBreaker > MAX_ORDER_RETRIES)
        {
         break;
        }
     }

   if(MagicOrdersTotal(magicNumber) == 0)
     {
      returnThis = true;
     }
   return(returnThis);
  }
```

### Get Pending Orders Data Function

For any autonomous algorithmic trading system or Expert Advisor to be reliable and consistently profitable eventually, it must be aware of all position and order statuses in the account. In the [second article](https://www.mql5.com/en/articles/15224), we developed a function to monitor positions that collect data on all open positions. Since this article focuses on pending orders, we will create a function to scan and store data for all pending orders, making it available throughout the library for use by other functions. We will achieve this using the global variables established at the beginning of this article.

Because these global variables cannot be exported or accessed outside the library source code file or the .EX5 binary file, we will pass them to various exportable pending orders status functions, allowing them to be accessible in the final .EX5 library. We will begin by creating a function to retrieve all pending orders' data, which we will name _GetPendingOrdersData(...)_.

The _GetPendingOrdersData(...)_ function will retrieve and save the status details of open pending orders in relation to the _account, specific trading symbols_, and the Expert Advisor's _magic number_. It will begin by resetting various global variables that store totals and volumes for different types of pending orders, including buy stops, buy limits, sell stops, and sell limits. After this reset, the function will check for any open pending orders and iterate through them to collect relevant data. It will then filter the orders based on the provided _symbol_ and _magic number_, accumulating totals and volumes for the matching pending orders. This comprehensive approach will ensure that the function captures real-time data on the status of pending orders, which can be utilized by other functions in the Expert Advisor.

The function will accept two parameters:

1. **symbol (string)**: A _string_ representing the trading _symbol_ for which the pending order data will be retrieved. It will default to _ALL\_SYMBOLS_, indicating that it will collect data for all symbols unless a specific symbol is provided.
2. **magicNumber (ulong)**: An _unsigned long integer_ that represents the unique _magic number_ assigned to the Expert Advisor's pending orders. This parameter will default to 0, allowing the function to retrieve data for all pending orders, regardless of their magic number, unless a specific magic number is specified.

The function will not return any value ( _void_), as its main purpose is to populate and update global status variables with the details of the pending orders. These variables will then be accessed by other parts of the library to inform decision-making and trading strategies.

Below is the full function code with detailed comments to enhance understanding:

```
void GetPendingOrdersData(string symbol, ulong magicNumber)
  {
//-- Reset the account open pending orders status
   accountBuyStopOrdersTotal = 0;
   accountBuyLimitOrdersTotal = 0;

   accountSellStopOrdersTotal = 0;
   accountSellLimitOrdersTotal = 0;

   accountPendingOrdersVolumeTotal = 0.0;

   accountBuyStopOrdersVolumeTotal = 0.0;
   accountBuyLimitOrdersVolumeTotal = 0.0;

   accountSellStopOrdersVolumeTotal = 0.0;
   accountSellLimitOrdersVolumeTotal = 0.0;

//-- Reset the EA's magic open pending orders status
   magicPendingOrdersTotal = 0;

   magicBuyStopOrdersTotal = 0;
   magicBuyLimitOrdersTotal = 0;

   magicSellStopOrdersTotal = 0;
   magicSellLimitOrdersTotal = 0;

   magicPendingOrdersVolumeTotal = 0.0;

   magicBuyStopOrdersVolumeTotal = 0.0;
   magicBuyLimitOrdersVolumeTotal = 0.0;

   magicSellStopOrdersVolumeTotal = 0.0;
   magicSellLimitOrdersVolumeTotal = 0.0;

//-- Reset the symbol open pending orders status
   symbolPendingOrdersTotal = 0;

   symbolBuyStopOrdersTotal = 0;
   symbolBuyLimitOrdersTotal = 0;

   symbolSellStopOrdersTotal = 0;
   symbolSellLimitOrdersTotal = 0;

   symbolPendingOrdersVolumeTotal = 0.0;

   symbolBuyStopOrdersVolumeTotal = 0.0;
   symbolBuyLimitOrdersVolumeTotal = 0.0;

   symbolSellStopOrdersVolumeTotal = 0.0;
   symbolSellLimitOrdersVolumeTotal = 0.0;

//-- Update and save the open pending orders status with realtime data
   int totalOpenPendingOrders = OrdersTotal();
   if(totalOpenPendingOrders > 0)
     {
      //-- Scan for symbol and magic number specific pending orders and save their status
      for(int x = 0; x < totalOpenPendingOrders; x++)
        {
         //--- Get the pending orders properties
         ulong  orderTicket = OrderGetTicket(x); //-- Get ticket to select the pending order
         string selectedSymbol = OrderGetString(ORDER_SYMBOL);
         ulong orderMagicNo = OrderGetInteger(ORDER_MAGIC);

         //-- Filter pending orders by magic number
         if(magicNumber != 0 && orderMagicNo != magicNumber)
           {
            continue;
           }

         //-- Save the account pending orders status first
         accountPendingOrdersVolumeTotal += OrderGetDouble(ORDER_VOLUME_CURRENT);

         if(OrderGetInteger(ORDER_TYPE) == ORDER_TYPE_BUY_STOP)
           {
            //-- Account properties
            ++accountBuyStopOrdersTotal;
            accountBuyStopOrdersVolumeTotal += OrderGetDouble(ORDER_VOLUME_CURRENT);
           }
         if(OrderGetInteger(ORDER_TYPE) == ORDER_TYPE_BUY_LIMIT)
           {
            //-- Account properties
            ++accountBuyLimitOrdersTotal;
            accountBuyLimitOrdersVolumeTotal += OrderGetDouble(ORDER_VOLUME_CURRENT);
           }
         if(OrderGetInteger(ORDER_TYPE) == ORDER_TYPE_SELL_STOP)
           {
            //-- Account properties
            ++accountSellStopOrdersTotal;
            accountSellStopOrdersVolumeTotal += OrderGetDouble(ORDER_VOLUME_CURRENT);
           }
         if(OrderGetInteger(ORDER_TYPE) == ORDER_TYPE_SELL_LIMIT)
           {
            //-- Account properties
            ++accountSellLimitOrdersTotal;
            accountSellLimitOrdersVolumeTotal += OrderGetDouble(ORDER_VOLUME_CURRENT);
           }

         //-- Filter pending orders openend by EA and save their status
         if(
            OrderGetInteger(ORDER_REASON) == ORDER_REASON_EXPERT &&
            orderMagicNo == magicNumber
         )
           {
            ++magicPendingOrdersTotal;
            magicPendingOrdersVolumeTotal += OrderGetDouble(ORDER_VOLUME_CURRENT);
            if(OrderGetInteger(ORDER_TYPE) == ORDER_TYPE_BUY_STOP)
              {
               //-- Magic properties
               ++magicBuyStopOrdersTotal;
               magicBuyStopOrdersVolumeTotal += OrderGetDouble(ORDER_VOLUME_CURRENT);
              }
            if(OrderGetInteger(ORDER_TYPE) == ORDER_TYPE_BUY_LIMIT)
              {
               //-- Magic properties
               ++magicBuyLimitOrdersTotal;
               magicBuyLimitOrdersVolumeTotal += OrderGetDouble(ORDER_VOLUME_CURRENT);
              }
            if(OrderGetInteger(ORDER_TYPE) == ORDER_TYPE_SELL_STOP)
              {
               //-- Magic properties
               ++magicSellStopOrdersTotal;
               magicSellStopOrdersVolumeTotal += OrderGetDouble(ORDER_VOLUME_CURRENT);
              }
            if(OrderGetInteger(ORDER_TYPE) == ORDER_TYPE_SELL_LIMIT)
              {
               //-- Magic properties
               ++magicSellLimitOrdersTotal;
               magicSellLimitOrdersVolumeTotal += OrderGetDouble(ORDER_VOLUME_CURRENT);
              }
           }

         //-- Filter positions by symbol
         if(symbol == ALL_SYMBOLS || selectedSymbol == symbol)
           {
            ++symbolPendingOrdersTotal;
            symbolPendingOrdersVolumeTotal += OrderGetDouble(ORDER_VOLUME_CURRENT);
            if(OrderGetInteger(ORDER_TYPE) == ORDER_TYPE_BUY_STOP)
              {
               ++symbolBuyStopOrdersTotal;
               symbolBuyStopOrdersVolumeTotal += OrderGetDouble(ORDER_VOLUME_CURRENT);
              }
            if(OrderGetInteger(ORDER_TYPE) == ORDER_TYPE_BUY_LIMIT)
              {
               ++symbolBuyLimitOrdersTotal;
               symbolBuyLimitOrdersVolumeTotal += OrderGetDouble(ORDER_VOLUME_CURRENT);
              }
            if(OrderGetInteger(ORDER_TYPE) == ORDER_TYPE_SELL_STOP)
              {
               ++symbolSellStopOrdersTotal;
               symbolSellStopOrdersVolumeTotal += OrderGetDouble(ORDER_VOLUME_CURRENT);
              }
            if(OrderGetInteger(ORDER_TYPE) == ORDER_TYPE_SELL_LIMIT)
              {
               ++symbolSellLimitOrdersTotal;
               symbolSellLimitOrdersVolumeTotal += OrderGetDouble(ORDER_VOLUME_CURRENT);
              }
           }
        }
     }
  }
```

### Buy Stop Orders Total Function

Returns an integer representing the total number of open buy stop orders in the account.

```
int BuyStopOrdersTotal() export
  {
   GetPendingOrdersData(ALL_SYMBOLS, 0);
   return(accountBuyStopOrdersTotal);
  }
```

### Buy Limit Orders Total Function

Returns an integer representing the total number of open buy limit orders in the account.

```
int BuyLimitOrdersTotal() export
  {
   GetPendingOrdersData(ALL_SYMBOLS, 0);
   return(accountBuyLimitOrdersTotal);
  }
```

### Sell Stop Orders Total Function

Returns an integer representing the total number of open sell stop orders in the account.

```
int SellStopOrdersTotal() export
  {
   GetPendingOrdersData(ALL_SYMBOLS, 0);
   return(accountSellStopOrdersTotal);
  }
```

### Sell Limit Orders Total Function

Returns an integer representing the total number of open sell limit orders in the account.

```
int SellLimitOrdersTotal() export
  {
   GetPendingOrdersData(ALL_SYMBOLS, 0);
   return(accountSellLimitOrdersTotal);
  }
```

### Orders Total Volume Function

Returns a double value representing the total volume/lot/quantity of all the open orders in the account.

```
double OrdersTotalVolume() export
  {
   GetPendingOrdersData(ALL_SYMBOLS, 0);
   return(accountPendingOrdersVolumeTotal);
  }
```

### Buy Stop Orders Total Volume Function

Returns a double value representing the total volume/lot/quantity of all the open buy stop orders in the account.

```
double BuyStopOrdersTotalVolume() export
  {
   GetPendingOrdersData(ALL_SYMBOLS, 0);
   return(accountBuyStopOrdersVolumeTotal);
  }
```

### Buy Limit Orders Total Volume Function

Returns a double value representing the total volume/lot/quantity of all the open buy limit orders in the account.

```
double BuyLimitOrdersTotalVolume() export
  {
   GetPendingOrdersData(ALL_SYMBOLS, 0);
   return(accountBuyLimitOrdersVolumeTotal);
  }
```

### Sell Stop Orders Total Volume Function

Returns a double value representing the total volume/lot/quantity of all the open sell stop orders in the account.

```
double SellStopOrdersTotalVolume() export
  {
   GetPendingOrdersData(ALL_SYMBOLS, 0);
   return(accountSellStopOrdersVolumeTotal);
  }
```

### Sell Limit Orders Total Volume Function

Returns a double value representing the total volume/lot/quantity of all the open sell limit orders in the account.

```
double SellLimitOrdersTotalVolume() export
  {
   GetPendingOrdersData(ALL_SYMBOLS, 0);
   return(accountSellLimitOrdersVolumeTotal);
  }
```

### Magic Orders Total Function

Returns an integer value of the total number of all the open pending orders for the specified magic number in the account.

```
int MagicOrdersTotal(ulong magicNumber) export
  {
   GetPendingOrdersData(ALL_SYMBOLS, magicNumber);
   return(magicPendingOrdersTotal);
  }
```

### Magic Buy Stop Orders Total Function

Returns an integer value of the total number of all the open buy stop orders for the specified magic number in the account.

```
int MagicBuyStopOrdersTotal(ulong magicNumber) export
  {
   GetPendingOrdersData(ALL_SYMBOLS, magicNumber);
   return(magicBuyStopOrdersTotal);
  }
```

### Magic Buy Limit Orders Total Function

Returns an integer value of the total number of all the open buy limit orders for the specified magic number in the account.

```
int MagicBuyLimitOrdersTotal(ulong magicNumber) export
  {
   GetPendingOrdersData(ALL_SYMBOLS, magicNumber);
   return(magicBuyLimitOrdersTotal);
  }
```

### Magic Sell Stop Orders Total Function

Returns an integer value of the total number of all the open sell stop orders for the specified magic number in the account.

```
int MagicSellStopOrdersTotal(ulong magicNumber) export
  {
   GetPendingOrdersData(ALL_SYMBOLS, magicNumber);
   return(magicSellStopOrdersTotal);
  }
```

### Magic Sell Limit Orders Total Function

Returns an integer value of the total number of all the open sell limit orders for the specified magic number in the account.

```
int MagicSellLimitOrdersTotal(ulong magicNumber) export
  {
   GetPendingOrdersData(ALL_SYMBOLS, magicNumber);
   return(magicSellLimitOrdersTotal);
  }
```

### Magic Orders Total Volume Function

Returns a double value representing the total volume/lot/quantity of all the open pending orders for the specified magic number in the account.

```
double MagicOrdersTotalVolume(ulong magicNumber) export
  {
   GetPendingOrdersData(ALL_SYMBOLS, magicNumber);
   return(magicPendingOrdersVolumeTotal);
  }
```

### Magic Buy Stop Orders Total Volume Function

Returns a double value representing the total volume/lot/quantity of all the open buy stop orders for the specified magic number in the account.

```
double MagicBuyStopOrdersTotalVolume(ulong magicNumber) export
  {
   GetPendingOrdersData(ALL_SYMBOLS, magicNumber);
   return(magicBuyStopOrdersVolumeTotal);
  }
```

### Magic Buy Limit Orders Total Volume Function

Returns a double value representing the total volume/lot/quantity of all the open buy limit orders for the specified magic number in the account.

```
double MagicBuyLimitOrdersTotalVolume(ulong magicNumber) export
  {
   GetPendingOrdersData(ALL_SYMBOLS, magicNumber);
   return(magicBuyLimitOrdersVolumeTotal);
  }
```

### Magic Sell Stop Orders Total Volume Function

Returns a double value representing the total volume/lot/quantity of all the open sell stop orders for the specified magic number in the account.

```
double MagicSellStopOrdersTotalVolume(ulong magicNumber) export
  {
   GetPendingOrdersData(ALL_SYMBOLS, magicNumber);
   return(magicSellStopOrdersVolumeTotal);
  }
```

### Magic Sell Limit Orders Total Volume Function

Returns a double value representing the total volume/lot/quantity of all the open sell limit orders for the specified magic number in the account.

```
double MagicSellLimitOrdersTotalVolume(ulong magicNumber) export
  {
   GetPendingOrdersData(ALL_SYMBOLS, magicNumber);
   return(magicSellLimitOrdersVolumeTotal);
  }
```

### Symbol Orders Total Function

Returns an integer value representing the total number of all the open pending orders for a specified symbol and magic number in the account.

```
int SymbolOrdersTotal(string symbol, ulong magicNumber) export
  {
   GetPendingOrdersData(symbol, magicNumber);
   return(symbolPendingOrdersTotal);
  }
```

### Symbol Buy Stop Orders Total Function

Returns an integer value representing the total number of all the open buy stop orders for a specified symbol and magic number in the account.

```
int SymbolBuyStopOrdersTotal(string symbol, ulong magicNumber) export
  {
   GetPendingOrdersData(symbol, magicNumber);
   return(symbolBuyStopOrdersTotal);
  }
```

### Symbol Buy Limit Orders Total Function

Returns an integer value representing the total number of all the open buy limit orders for a specified symbol and magic number in the account.

```
int SymbolBuyLimitOrdersTotal(string symbol, ulong magicNumber) export
  {
   GetPendingOrdersData(symbol, magicNumber);
   return(symbolBuyLimitOrdersTotal);
  }
```

### Symbol Sell Stop Orders Total Function

Returns an integer value representing the total number of all the open sell stop orders for a specified symbol and magic number in the account.

```
int SymbolSellStopOrdersTotal(string symbol, ulong magicNumber) export
  {
   GetPendingOrdersData(symbol, magicNumber);
   return(symbolSellStopOrdersTotal);
  }
```

### Symbol Sell Limit Orders Total Function

Returns an integer value representing the total number of all the open sell limit orders for a specified symbol and magic number in the account.

```
int SymbolSellLimitOrdersTotal(string symbol, ulong magicNumber) export
  {
   GetPendingOrdersData(symbol, magicNumber);
   return(symbolSellLimitOrdersTotal);
  }
```

### Symbol Orders Total Volume Function

Returns a double value representing the total volume/lot/quantity of all the open pending orders for the specified symbol and magic number in the account.

```
double SymbolOrdersTotalVolume(string symbol, ulong magicNumber) export
  {
   GetPendingOrdersData(symbol, magicNumber);
   return(symbolPendingOrdersVolumeTotal);
  }
```

### Symbol Buy Stop Orders Total Volume Function

Returns a double value representing the total volume/lot/quantity of all the open buy stop orders for the specified symbol and magic number in the account.

```
double SymbolBuyStopOrdersTotalVolume(string symbol, ulong magicNumber) export
  {
   GetPendingOrdersData(symbol, magicNumber);
   return(symbolBuyStopOrdersVolumeTotal);
  }
```

### Symbol Buy Limit Orders Total Volume Function

Returns a double value representing the total volume/lot/quantity of all the open buy limit orders for the specified symbol and magic number in the account.

```
double SymbolBuyLimitOrdersTotalVolume(string symbol, ulong magicNumber) export
  {
   GetPendingOrdersData(symbol, magicNumber);
   return(symbolBuyLimitOrdersVolumeTotal);
  }
```

### Symbol Sell Stop Orders Total Volume Function

Returns a double value representing the total volume/lot/quantity of all the open sell stop orders for the specified symbol and magic number in the account.

```
double SymbolSellStopOrdersTotalVolume(string symbol, ulong magicNumber) export
  {
   GetPendingOrdersData(symbol, magicNumber);
   return(symbolSellStopOrdersVolumeTotal);
  }
```

### Symbol Sell Limit Orders Total Volume Function

Returns a double value representing the total volume/lot/quantity of all the open sell limit orders for the specified symbol and magic number in the account.

```
double SymbolSellLimitOrdersTotalVolume(string symbol, ulong magicNumber) export
  {
   GetPendingOrdersData(symbol, magicNumber);
   return(symbolSellLimitOrdersVolumeTotal);
  }
```

### Account Orders Status Function

Returns a pre-formatted string that contains the status of the account orders, which can be printed to the log or shown in the chart comments. The function takes a single _boolean_ parameter named _formatForComment_. If _formatForComment_ is set to _true_, the function formats the data for display in the chart window; if _false_, it formats the data for the Expert Advisor's log tab.

```
string AccountOrdersStatus(bool formatForComment) export
  {
   GetPendingOrdersData(ALL_SYMBOLS, 0); //-- Update the orders status variables before we display their data
   string spacer = "";
   if(formatForComment) //-- Add some formating space for the chart comment string
     {
      spacer = "                                        ";
     }
   string accountOrdersStatus = "\r\n" + spacer + "|---------------------------------------------------------------------------\r\n";
   accountOrdersStatus += spacer + "| " + (string)AccountInfoInteger(ACCOUNT_LOGIN) + " - ACCOUNT ORDERS STATUS \r\n";
   accountOrdersStatus += spacer + "|---------------------------------------------------------------------------\r\n";
   accountOrdersStatus += spacer + "|     Total Open:   " + (string)OrdersTotal() + "\r\n";
   accountOrdersStatus += spacer + "|     Total Volume: " + (string)accountPendingOrdersVolumeTotal + "\r\n";
   accountOrdersStatus += spacer + "|------------------------------------------------------------------\r\n";
   accountOrdersStatus += spacer + "| BUY STOP ORDERS: \r\n";
   accountOrdersStatus += spacer + "|     Total Open:   " + (string)accountBuyStopOrdersTotal + "\r\n";
   accountOrdersStatus += spacer + "|     Total Volume: " + (string)accountBuyStopOrdersVolumeTotal + "\r\n";
   accountOrdersStatus += spacer + "|------------------------------------------------------------------\r\n";
   accountOrdersStatus += spacer + "| BUY LIMIT ORDERS: \r\n";
   accountOrdersStatus += spacer + "|     Total Open:   " + (string)accountBuyLimitOrdersTotal + "\r\n";
   accountOrdersStatus += spacer + "|     Total Volume: " + (string)accountBuyLimitOrdersVolumeTotal + "\r\n";
   accountOrdersStatus += spacer + "|---------------------------------------------------------------------------\r\n";
   accountOrdersStatus += spacer + "| SELL STOP ORDERS: \r\n";
   accountOrdersStatus += spacer + "|     Total Open:   " + (string)accountSellStopOrdersTotal + "\r\n";
   accountOrdersStatus += spacer + "|     Total Volume: " + (string)accountSellStopOrdersVolumeTotal + "\r\n";
   accountOrdersStatus += spacer + "|---------------------------------------------------------------------------\r\n";
   accountOrdersStatus += spacer + "| SELL LIMIT ORDERS: \r\n";
   accountOrdersStatus += spacer + "|     Total Open:   " + (string)accountSellLimitOrdersTotal + "\r\n";
   accountOrdersStatus += spacer + "|     Total Volume: " + (string)accountSellLimitOrdersVolumeTotal + "\r\n";
   accountOrdersStatus += spacer + "|---------------------------------------------------------------------------\r\n";
   accountOrdersStatus += spacer + "\r\n";
   return(accountOrdersStatus);
  }
```

### Magic Orders Status Function

Returns a pre-formatted string that indicates the status of the magic orders, which can be printed to the log or displayed in the chart comments. The function accepts two parameters: an _unsigned long_ called _magicNumber_ to specify the targeted magic number, and a _boolean_ named _formatForComment_ to determine the formatting type. If _formatForComment_ is true, the function formats the data for display in the chart window; if _false_, it formats the data for the Expert Advisor's log tab.

```
string MagicOrdersStatus(ulong magicNumber, bool formatForComment) export
  {
   GetPendingOrdersData(ALL_SYMBOLS, magicNumber); //-- Update the order status variables before we display their data
   string spacer = "";
   if(formatForComment) //-- Add some formating space for the chart comment string
     {
      spacer = "                                        ";
     }
   string magicOrdersStatus = "\r\n" + spacer + "|---------------------------------------------------------------------------\r\n";
   magicOrdersStatus += spacer + "| " + (string)magicNumber + " - MAGIC ORDERS STATUS \r\n";
   magicOrdersStatus += spacer + "|---------------------------------------------------------------------------\r\n";
   magicOrdersStatus += spacer + "|     Total Open:   " + (string)magicPendingOrdersTotal + "\r\n";
   magicOrdersStatus += spacer + "|     Total Volume: " + (string)magicPendingOrdersVolumeTotal + "\r\n";
   magicOrdersStatus += spacer + "|------------------------------------------------------------------\r\n";
   magicOrdersStatus += spacer + "| BUY STOP ORDERS: \r\n";
   magicOrdersStatus += spacer + "|     Total Open:   " + (string)magicBuyStopOrdersTotal + "\r\n";
   magicOrdersStatus += spacer + "|     Total Volume: " + (string)magicBuyStopOrdersVolumeTotal + "\r\n";
   magicOrdersStatus += spacer + "|------------------------------------------------------------------\r\n";
   magicOrdersStatus += spacer + "| BUY LIMIT ORDERS: \r\n";
   magicOrdersStatus += spacer + "|     Total Open:   " + (string)magicBuyLimitOrdersTotal + "\r\n";
   magicOrdersStatus += spacer + "|     Total Volume: " + (string)magicBuyLimitOrdersVolumeTotal + "\r\n";
   magicOrdersStatus += spacer + "|---------------------------------------------------------------------------\r\n";
   magicOrdersStatus += spacer + "| SELL STOP ORDERS: \r\n";
   magicOrdersStatus += spacer + "|     Total Open:   " + (string)magicSellStopOrdersTotal + "\r\n";
   magicOrdersStatus += spacer + "|     Total Volume: " + (string)magicSellStopOrdersVolumeTotal + "\r\n";
   magicOrdersStatus += spacer + "|------------------------------------------------------------------\r\n";
   magicOrdersStatus += spacer + "| SELL LIMIT ORDERS: \r\n";
   magicOrdersStatus += spacer + "|     Total Open:   " + (string)magicSellLimitOrdersTotal + "\r\n";
   magicOrdersStatus += spacer + "|     Total Volume: " + (string)magicSellLimitOrdersVolumeTotal + "\r\n";
   magicOrdersStatus += spacer + "|---------------------------------------------------------------------------\r\n";
   magicOrdersStatus += spacer + "\r\n";
   return(magicOrdersStatus);
  }
```

### Symbol Orders Status Function

Returns a pre-formatted string that indicates the status of the symbol orders, which can be printed to the log or displayed in the chart comments. The function accepts three parameters: a _string_ named _symbol_, an _unsigned long_ called _magicNumber_ to specify the targeted magic number ( _a value of zero (0) will disable the magic number filter_), and a _boolean_ named _formatForComment_ to determine the formatting type. If _formatForComment_ is _true_, the function formats the data for display in the chart window; if _false_, it formats the data for the Expert Advisor's log tab.

```
string SymbolOrdersStatus(string symbol, ulong magicNumber, bool formatForComment) export
  {
   GetPendingOrdersData(symbol, magicNumber); //-- Update the order status variables before we display their data
   string spacer = "";
   if(formatForComment) //-- Add some formating space for the chart comment string
     {
      spacer = "                                        ";
     }
   string symbolOrdersStatus = "\r\n" + spacer + "|---------------------------------------------------------------------------\r\n";
   symbolOrdersStatus += spacer + "| " + symbol + " - SYMBOL ORDERS STATUS \r\n";
   symbolOrdersStatus += spacer + "|---------------------------------------------------------------------------\r\n";
   symbolOrdersStatus += spacer + "|     Total Open:   " + (string)symbolPendingOrdersTotal + "\r\n";
   symbolOrdersStatus += spacer + "|     Total Volume: " + (string)symbolPendingOrdersVolumeTotal + "\r\n";
   symbolOrdersStatus += spacer + "|------------------------------------------------------------------\r\n";
   symbolOrdersStatus += spacer + "| BUY STOP ORDERS: \r\n";
   symbolOrdersStatus += spacer + "|     Total Open:   " + (string)symbolBuyStopOrdersTotal + "\r\n";
   symbolOrdersStatus += spacer + "|     Total Volume: " + (string)symbolBuyStopOrdersVolumeTotal + "\r\n";
   symbolOrdersStatus += spacer + "|------------------------------------------------------------------\r\n";
   symbolOrdersStatus += spacer + "| BUY LIMIT ORDERS: \r\n";
   symbolOrdersStatus += spacer + "|     Total Open:   " + (string)symbolBuyLimitOrdersTotal + "\r\n";
   symbolOrdersStatus += spacer + "|     Total Volume: " + (string)symbolBuyLimitOrdersVolumeTotal + "\r\n";
   symbolOrdersStatus += spacer + "|---------------------------------------------------------------------------\r\n";
   symbolOrdersStatus += spacer + "| SELL STOP ORDERS: \r\n";
   symbolOrdersStatus += spacer + "|     Total Open:   " + (string)symbolSellStopOrdersTotal + "\r\n";
   symbolOrdersStatus += spacer + "|     Total Volume: " + (string)symbolSellStopOrdersVolumeTotal + "\r\n";
   symbolOrdersStatus += spacer + "|------------------------------------------------------------------\r\n";
   symbolOrdersStatus += spacer + "| SELL LIMIT ORDERS: \r\n";
   symbolOrdersStatus += spacer + "|     Total Open:   " + (string)symbolSellLimitOrdersTotal + "\r\n";
   symbolOrdersStatus += spacer + "|     Total Volume: " + (string)symbolSellLimitOrdersVolumeTotal + "\r\n";
   symbolOrdersStatus += spacer + "|---------------------------------------------------------------------------\r\n";
   symbolOrdersStatus += spacer + "\r\n";
   return(symbolOrdersStatus);
  }
```

The extensive library functions we have created above make up our _Pending Orders Management EX5_ Library. Attached at the bottom of this article, you will find the source code file, _PendingOrdersManager.mq5_, as well as the compiled binary executable file, _PendingOrdersManager.ex5_, which you can easily import and use in your MQL5 projects.

### How to Import and Implement our Pending Orders Management EX5 Library

We have developed a robust _Pending Orders Management EX5 library_ that includes essential functions for handling pending orders, retrieving their status, and displaying relevant information. Now, it’s time to document and demonstrate how to correctly import and use this library in any MQL5 project.

To make the implementation process easier, we will begin by outlining all the functions and modules within the Pending Orders Management library, along with practical code examples in the documentation section below. This will give users a clear understanding of the components included in the _PendingOrdersManager.ex5_ binary file.

### Pending Orders Manager EX5 Library Documentation

**Step 1: Copy the library executable files ( _PositionsManager.ex5_ and _PendingOrdersManager.ex5_)**

Place the _PositionsManager.ex5_ and _PendingOrdersManager.ex5_ files in the _MQL5/Libraries/Toolkit_ folder or the same folder, as the source code file importing the library. Make sure these files are downloaded and copied to the specified location if they are not already there. Copies of both files are attached at the end of this article for your convenience.

**Step 2: Import the function prototype descriptions**

In the header section of your source code, add the _import_ directives for the _Pending Orders Manager library_ and its function _prototype descriptions_. Use the following code segment to efficiently import all functions or modules from the _PendingOrdersManager.ex5_ library. I’ve also created a blank Expert Advisor template ( _PendingOrdersManager\_Imports\_Template.mq5_) that includes this code segment. You can comment out or remove any function descriptions you don’t need for your project. The _PendingOrdersManager\_Imports\_Template.mq5_ file is also attached at the end of this article.

```
//+------------------------------------------------------------------------------------------+
//-- Copy and paste the import derictives below to use the Pending Orders Manager EX5 Library
//---
//+-------------------------------------------------------------------------------------+
//| PendingOrdersManager.ex5 imports template                                           |
//+-------------------------------------------------------------------------------------+
#import "Toolkit/PendingOrdersManager.ex5" //-- Opening import directive
//-- Function descriptions for the imported function prototypes

//-- Pending Orders Execution and Modification Functions
bool OpenBuyLimit(ulong magicNumber, string symbol, double entryPrice, double lotSize, int sl, int tp, string orderComment);
bool OpenBuyStop(ulong magicNumber, string symbol, double entryPrice, double lotSize, int sl, int tp, string orderComment);
bool OpenSellLimit(ulong magicNumber, string symbol, double entryPrice, double lotSize, int sl, int tp, string orderComment);
bool OpenSellStop(ulong magicNumber, string symbol, double entryPrice, double lotSize, int sl, int tp, string orderComment);
bool ModifyPendingOrderByTicket(ulong orderTicket, double newEntryPrice, int newSl, int newTp);
bool DeletePendingOrderByTicket(ulong orderTicket);
bool DeleteAllPendingOrders(string symbol, ulong magicNumber);
bool DeleteAllPendingOrders();
bool DeleteAllBuyStops(string symbol, ulong magicNumber);
bool DeleteAllBuyLimits(string symbol, ulong magicNumber);
bool DeleteAllSellStops(string symbol, ulong magicNumber);
bool DeleteAllSellLimits(string symbol, ulong magicNumber);
bool DeleteAllMagicOrders(ulong magicNumber);

//-- Pending Orders Status Monitoring Functions
int BuyStopOrdersTotal();
int BuyLimitOrdersTotal();
int SellStopOrdersTotal();
int SellLimitOrdersTotal();
double OrdersTotalVolume();
double BuyStopOrdersTotalVolume();
double BuyLimitOrdersTotalVolume();
double SellStopOrdersTotalVolume();
double SellLimitOrdersTotalVolume();

//-- Pending Orders Filtered By Magic Number Status Monitoring Functions
int MagicOrdersTotal(ulong magicNumber);
int MagicBuyStopOrdersTotal(ulong magicNumber);
int MagicBuyLimitOrdersTotal(ulong magicNumber);
int MagicSellStopOrdersTotal(ulong magicNumber);
int MagicSellLimitOrdersTotal(ulong magicNumber);
double MagicOrdersTotalVolume(ulong magicNumber);
double MagicBuyStopOrdersTotalVolume(ulong magicNumber);
double MagicBuyLimitOrdersTotalVolume(ulong magicNumber);
double MagicSellStopOrdersTotalVolume(ulong magicNumber);
double MagicSellLimitOrdersTotalVolume(ulong magicNumber);

//-- Pending Orders Filtered By Symbol and/or Magic Number Status Monitoring Functions
int SymbolOrdersTotal(string symbol, ulong magicNumber);
int SymbolBuyStopOrdersTotal(string symbol, ulong magicNumber);
int SymbolBuyLimitOrdersTotal(string symbol, ulong magicNumber);
int SymbolSellStopOrdersTotal(string symbol, ulong magicNumber);
int SymbolSellLimitOrdersTotal(string symbol, ulong magicNumber);
double SymbolOrdersTotalVolume(string symbol, ulong magicNumber);
double SymbolBuyStopOrdersTotalVolume(string symbol, ulong magicNumber);
double SymbolBuyLimitOrdersTotalVolume(string symbol, ulong magicNumber);
double SymbolSellStopOrdersTotalVolume(string symbol, ulong magicNumber);
double SymbolSellLimitOrdersTotalVolume(string symbol, ulong magicNumber);

//-- Log and Data Display Functions
string AccountOrdersStatus(bool formatForComment);
string MagicOrdersStatus(ulong magicNumber, bool formatForComment);
string SymbolOrdersStatus(string symbol, ulong magicNumber, bool formatForComment);

#import //--- Closing import directive
//+-------------------------------------------------------------------------------------+
```

With the .EX5 library prototype functions now imported into your MQL5 project or source code, the table below provides detailed documentation and practical examples of how to implement and use these functions in your code.

| Function Prototype Description | Description | Example Use Case |
| --- | --- | --- |
| ```<br>bool OpenBuyLimit(<br>   ulong magicNumber,<br>   string symbol,<br>   double entryPrice,<br>   double lotSize,<br>   int sl,<br>   int tp,<br>   string orderComment<br>);<br>``` | Opens a new buy limit order matching the specified parameters. | ```<br>double symbolPoint = SymbolInfoDouble(_Symbol, SYMBOL_POINT);<br>int spread = (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);<br>//---<br>ulong magicNo = 123;<br>string symbol = _Symbol;<br>double entryPrice = SymbolInfoDouble(<br>                       _Symbol, SYMBOL_ASK) - ((spread * 20) <br>                       * symbolPoint<br>                    );<br>double lotSize = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);<br>int sl = spread * 50;  //-- pips<br>int tp = spread * 100; //-- pips<br>string orderComment = "Pending Orders Manager Buy Limit Order";<br>OpenBuyLimit(<br>   magicNumber, symbol, entryPrice, <br>   lotSize, sl, tp, orderComment<br>);<br>``` |
| ```<br>bool OpenBuyStop(<br>   ulong magicNumber,<br>   string symbol,<br>   double entryPrice,<br>   double lotSize,<br>   int sl,<br>   int tp,<br>   string orderComment<br>);<br>``` | Opens a new buy stop order matching the specified parameters. | ```<br>double symbolPoint = SymbolInfoDouble(_Symbol, SYMBOL_POINT);<br>int spread = (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);<br>//---<br>ulong magicNo = 123;<br>string symbol = _Symbol;<br>double entryPrice = SymbolInfoDouble(<br>                       _Symbol, SYMBOL_ASK) + ((spread * 20) <br>                       * symbolPoint<br>                    );<br>double lotSize = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);<br>int sl = spread * 50;  //-- pips<br>int tp = spread * 100; //-- pips<br>string orderComment = "Pending Orders Manager Buy Stop Order";<br>OpenBuyStop(<br>   magicNumber, symbol, entryPrice, <br>   lotSize, sl, tp, orderComment<br>);<br>``` |
| ```<br>bool OpenSellLimit(<br>   ulong magicNumber,<br>   string symbol,<br>   double entryPrice,<br>   double lotSize,<br>   int sl,<br>   int tp,<br>   string orderComment<br>);<br>``` | Opens a new sell limit order matching the specified parameters. | ```<br>double symbolPoint = SymbolInfoDouble(_Symbol, SYMBOL_POINT);<br>int spread = (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);<br>//---<br>ulong magicNo = 123;<br>string symbol = _Symbol;<br>double entryPrice = SymbolInfoDouble(<br>                       _Symbol, SYMBOL_ASK) + ((spread * 20) <br>                       * symbolPoint<br>                    );<br>double lotSize = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);<br>int sl = spread * 50;  //-- pips<br>int tp = spread * 100; //-- pips<br>string orderComment = "Pending Orders Manager Sell Limit Order";<br>OpenSellLimit(<br>   magicNumber, symbol, entryPrice, <br>   lotSize, sl, tp, orderComment<br>);<br>``` |
| ```<br>bool OpenSellStop(<br>   ulong magicNumber,<br>   string symbol,<br>   double entryPrice,<br>   double lotSize,<br>   int sl,<br>   int tp,<br>   string orderComment<br>);<br>``` | Opens a new sell stop order matching the specified parameters. | ```<br>double symbolPoint = SymbolInfoDouble(_Symbol, SYMBOL_POINT);<br>int spread = (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);<br>//---<br>ulong magicNo = 123;<br>string symbol = _Symbol;<br>double entryPrice = SymbolInfoDouble(<br>                       _Symbol, SYMBOL_ASK) - ((spread * 20) <br>                       * symbolPoint<br>                    );<br>double lotSize = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);<br>int sl = spread * 50;  //-- pips<br>int tp = spread * 100; //-- pips<br>string orderComment = "Pending Orders Manager Sell Stop Order";<br>OpenSellStop(<br>   magicNumber, symbol, entryPrice, <br>   lotSize, sl, tp, orderComment<br>);<br>``` |
| ```<br>bool ModifyPendingOrderByTicket(<br>   ulong orderTicket,<br>   double newEntryPrice,<br>   int newSl,<br>   int newTp<br>);<br>``` | Modifies a pending order using the specified parameters. | ```<br>double symbolPoint = SymbolInfoDouble(_Symbol, SYMBOL_POINT);<br>int spread = (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);<br>int totalOpenOders = OrdersTotal();<br>for(int x = 0; x < totalOpenOders; x++)<br>  {<br>   ulong orderTicket = OrderGetTicket(x);<br>   if(orderTicket > 0)<br>     {<br>      //-- Modify a buy stop order<br>      if(OrderGetInteger(ORDER_TYPE) == ORDER_TYPE_BUY_STOP)<br>        {<br>         double newEntryPrice = OrderGetDouble(<br>                                   ORDER_PRICE_OPEN) + <br>                                   ((spread * 40) * symbolPoint<br>                                );<br>         int newSl = 0; //-- Do not modify the stop loss level<br>         int newTp = 0; //-- Don not modify the take profit level<br>         ModifyPendingOrderByTicket(<br>            orderTicket, newEntryPrice, newSl, newTp<br>         );<br>         break;<br>        }<br>     }<br>  }<br>``` |
| ```<br>bool DeletePendingOrderByTicket(<br>   ulong orderTicket<br>);<br>``` | Deletes a pending order based on the provided ticket number parameter. | ```<br>for(int x = 0; x < totalOpenOders; x++)<br>  {<br>   ulong orderTicket = OrderGetTicket(x);<br>   if(orderTicket > 0)<br>     {<br>      DeletePendingOrderByTicket(orderTicket);<br>      break;<br>     }<br>  }<br>``` |
| ```<br>bool DeleteAllPendingOrders(<br>   string symbol = ALL_SYMBOLS,<br>   ulong magicNumber = 0<br>);<br>``` | Deletes all pending orders based on the provided symbol name and magic number parameters. | ```<br>//Deletes all orders in the account<br>DeleteAllPendingOrders("", 0);<br>//Deletes all orders belonging to the symbol<br>DeleteAllPendingOrders(_Symbol, 0);<br>//Deletes all orders that have a magic number 101<br>DeleteAllPendingOrders("", 101);<br>//Deletes all EURUSD orders that have a magic number 101<br>DeleteAllPendingOrders("EURUSD", 101);<br>``` |
| ```<br>bool DeleteAllPendingOrders()<br>``` | Simply deletes all open pending orders in the account. | ```<br>//Deletes all orders in the account<br>DeleteAllPendingOrders();<br>``` |
| ```<br>bool DeleteAllBuyStops(<br>   string symbol = ALL_SYMBOLS,<br>   ulong magicNumber = 0<br>);<br>``` | Deletes all buy stop orders based on the provided symbol name and magic number parameters. | ```<br>//Deletes all buy stops in the account<br>DeleteAllBuyStops("", 0);<br>//Deletes all buy stops belonging to the symbol<br>DeleteAllBuyStops(_Symbol, 0);<br>//Deletes all buy stops that have a magic number 101<br>DeleteAllBuyStops("", 101);<br>//Deletes all EURUSD buy stops that have a magic number 101<br>DeleteAllBuyStops("EURUSD", 101);<br>``` |
| ```<br>bool DeleteAllBuyLimits(<br>   string symbol = ALL_SYMBOLS,<br>   ulong magicNumber = 0<br>);<br>``` | Deletes all buy limit orders based on the provided symbol name and magic number parameters. | ```<br>//Deletes all buy limits in the account<br>DeleteAllBuyLimits("", 0);<br>//Deletes all buy limits belonging to the symbol<br>DeleteAllBuyLimits(_Symbol, 0);<br>//Deletes all buy limits that have a magic number 101<br>DeleteAllBuyLimits("", 101);<br>//Deletes all GBPUSD buy limits that have a magic number 101<br>DeleteAllBuyLimits("GBPUSD", 101);<br>``` |
| ```<br>bool DeleteAllSellStops(<br>   string symbol = ALL_SYMBOLS,<br>   ulong magicNumber = 0<br>);<br>``` | Deletes all sell stop orders based on the provided symbol name and magic number parameters. | ```<br>//Deletes all sell stops in the account<br>DeleteAllSellStops("", 0);<br>//Deletes all sell stops belonging to the symbol<br>DeleteAllSellStops(_Symbol, 0);<br>//Deletes all sell stops that have a magic number 101<br>DeleteAllSellStops("", 101);<br>//Deletes all JPYUSD sell stops that have a magic number 101<br>DeleteAllSellStops("JPYUSD", 101);<br>``` |
| ```<br>bool DeleteAllSellLimits(<br>   string symbol = ALL_SYMBOLS,<br>   ulong magicNumber = 0<br>);<br>``` | Deletes all sell limit orders based on the provided symbol name and magic number parameters. | ```<br>//Deletes all sell limits in the account<br>DeleteAllSellLimits("", 0);<br>//Deletes all sell limits belonging to the symbol<br>DeleteAllSellLimits(_Symbol, 0);<br>//Deletes all sell limits that have a magic number 101<br>DeleteAllSellLimits("", 101);<br>//Deletes all AUDJPY sell limits that have a magic number 101<br>DeleteAllSellLimits("AUDJPY", 101);<br>``` |
| ```<br>bool DeleteAllMagicOrders(<br>   ulong magicNumber<br>);<br>``` | Deletes all pending orders based on the provided magic number parameter. | ```<br>//-- Deletes all orders open in the account<br>DeleteAllMagicOrders("", 0);<br>//-- Deletes all orders that have a magic number 101<br>DeleteAllMagicOrders(101);<br>``` |
| ```<br>int BuyStopOrdersTotal();<br>``` | Returns the total number of open buy stop orders. | ```<br>//Get the total number of open buy stops in the account<br>BuyStopOrdersTotal();<br>``` |
| ```<br>int BuyLimitOrdersTotal();<br>``` | Returns the total number of open buy limit orders. | ```<br>//Get the total number of open buy limits in the account<br>BuyLimitOrdersTotal();<br>``` |
| ```<br>int SellStopOrdersTotal();<br>``` | Returns the total number of open sell stop orders. | ```<br>//Get the total number of open sell stops in the account<br>SellStopOrdersTotal();<br>``` |
| ```<br>int SellLimitOrdersTotal();<br>``` | Returns the total number of open sell limit orders. | ```<br>//Get the total number of open sell limits in the account<br>SellLimitOrdersTotal();<br>``` |
| ```<br>double OrdersTotalVolume();<br>``` | Returns the total volume of all the open orders. | ```<br>//Get the total volume/lot of open orders in the account<br>OrdersTotalVolume();<br>``` |
| ```<br>double BuyStopOrdersTotalVolume();<br>``` | Returns the total volume of all the buy stop orders. | ```<br>//Get the total volume/lot of open buy stops in the account<br>BuyStopOrdersTotalVolume();<br>``` |
| ```<br>double BuyLimitOrdersTotalVolume();<br>``` | Returns the total volume of all the buy limit orders. | ```<br>//Get the total volume/lot of open buy limits in the account<br>BuyLimitOrdersTotalVolume();<br>``` |
| ```<br>double SellStopOrdersTotalVolume();<br>``` | Returns the total volume of all the sell stop orders. | ```<br>//Get the total volume/lot of open sell stops in the account<br>SellStopOrdersTotalVolume();<br>``` |
| ```<br>double SellLimitOrdersTotalVolume();<br>``` | Returns the total volume of all the sell limit orders. | ```<br>//Get the total volume/lot of open sell limits in the account<br>SellLimitOrdersTotalVolume();<br>``` |
| ```<br>int MagicOrdersTotal(<br>   ulong magicNumber<br>);<br>``` | Returns the number of open orders matching the specified magic number. | ```<br>//Get the total open pending orders for magic number 101<br>MagicOrdersTotal(101);<br>``` |
| ```<br>int MagicBuyStopOrdersTotal(<br>   ulong magicNumber<br>);<br>``` | Returns the number of open buy stop orders matching the specified magic number. | ```<br>//Get the total open buy stop orders for magic number 101<br>MagicBuyStopOrdersTotal(101);<br>``` |
| ```<br>int MagicBuyLimitOrdersTotal(<br>   ulong magicNumber<br>);<br>``` | Returns the number of open buy limit orders matching the specified magic number. | ```<br>//Get the total open buy limit orders for magic number 101<br>MagicBuyLimitOrdersTotal(101);<br>``` |
| ```<br>int MagicSellStopOrdersTotal(<br>   ulong magicNumber<br>);<br>``` | Returns the number of open sell stop orders matching the specified magic number. | ```<br>//Get the total open sell stop orders for magic number 101<br>MagicSellStopOrdersTotal(101);<br>``` |
| ```<br>int MagicSellLimitOrdersTotal(<br>   ulong magicNumber<br>);<br>``` | Returns the number of open sell limit orders matching the specified magic number. | ```<br>//Get the total open sell limit orders for magic number 101<br>MagicSellLimitOrdersTotal(101);<br>``` |
| ```<br>double MagicOrdersTotalVolume(<br>   ulong magicNumber<br>);<br>``` | Returns the total volume of all the open orders matching the specified magic number. | ```<br>//Get the total volume/lot of all open orders for magic 101<br>MagicOrdersTotalVolume(101);<br>``` |
| ```<br>double MagicBuyStopOrdersTotalVolume(<br>   ulong magicNumber<br>);<br>``` | Returns the total volume of all the open buy stop orders matching the specified magic number. | ```<br>//Get the total volume/lot of all buy stop orders for magic 101<br>MagicBuyStopOrdersTotalVolume(101);<br>``` |
| ```<br>double MagicBuyLimitOrdersTotalVolume(<br>   ulong magicNumber<br>);<br>``` | Returns the total volume of all the open buy limit orders matching the specified magic number. | ```<br>//Get the total volume/lot of all buy limit orders for magic 101<br>MagicBuyLimitOrdersTotalVolume(101);<br>``` |
| ```<br>double MagicSellStopOrdersTotalVolume(<br>   ulong magicNumber<br>);<br>``` | Returns the total volume of all the open sell stop orders matching the specified magic number. | ```<br>//Get the total volume/lot of all sell stop orders for magic 101<br>MagicSellStopOrdersTotalVolume(101);<br>``` |
| ```<br>double MagicSellLimitOrdersTotalVolume(<br>   ulong magicNumber<br>);<br>``` | Returns the total volume of all the open sell limit orders matching the specified magic number. | ```<br>//Get the total volume/lot of all sell limit orders for magic 101<br>MagicSellLimitOrdersTotalVolume(101);<br>``` |
| ```<br>int SymbolOrdersTotal(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Returns the number of open orders matching the specified symbol and magic number. | ```<br>//Get the total open orders for the symbol<br>SymbolOrdersTotal(_Symbol, 0);<br>//Get the total open orders for the symbol and magic 101<br>SymbolOrdersTotal(_Symbol, 101);<br>``` |
| ```<br>int SymbolBuyStopOrdersTotal(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Returns the number of open buy stop orders matching the specified symbol and magic number. | ```<br>//Get the total buy stop orders for the symbol<br>SymbolBuyStopOrdersTotal(_Symbol, 0);<br>//Get the total buy stop orders for the symbol and magic number 101<br>SymbolBuyStopOrdersTotal(_Symbol, 101);<br>``` |
| ```<br>int SymbolBuyLimitOrdersTotal(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Returns the number of open buy limit orders matching the specified symbol and magic number. | ```<br>//Get the total buy limit orders for the symbol<br>SymbolBuyLimitOrdersTotal(_Symbol, 0);<br>//Get the total buy limit orders for the symbol and magic number 101<br>SymbolBuyLimitOrdersTotal(_Symbol, 101);<br>``` |
| ```<br>int SymbolSellStopOrdersTotal(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Returns the number of open sell stop orders matching the specified symbol and magic number. | ```<br>//Get the total sell stop orders for the symbol<br>SymbolSellStopOrdersTotal(_Symbol, 0);<br>//Get the total sell stop orders for the symbol and magic 101<br>SymbolSellStopOrdersTotal(_Symbol, 101);<br>``` |
| ```<br>int SymbolSellLimitOrdersTotal(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Returns the number of open sell limit orders matching the specified symbol and magic number. | ```<br>//Get the total sell limit orders for the symbol<br>SymbolSellLimitOrdersTotal(_Symbol, 0);<br>//Get the total sell limit orders for the symbol and magic 101<br>SymbolSellLimitOrdersTotal(_Symbol, 101);<br>``` |
| ```<br>double SymbolOrdersTotalVolume(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Returns the total volume of all the open orders matching the specified symbol and magic number. | ```<br>//Get the total orders volume/lot for the symbol<br>SymbolOrdersTotalVolume(_Symbol, 0);<br>//Get the total orders volume/lot for the symbol and magic 101<br>SymbolOrdersTotalVolume(_Symbol, 101);<br>``` |
| ```<br>double SymbolBuyStopOrdersTotalVolume(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Returns the total volume of all the open buy stop orders matching the specified symbol and magic number. | ```<br>//Get the total buy stops volume/lot for the symbol<br>SymbolBuyStopOrdersTotalVolume(_Symbol, 0);<br>//Get the total buy stops volume/lot for the symbol and magic 101<br>SymbolBuyStopOrdersTotalVolume(_Symbol, 101);<br>``` |
| ```<br>double SymbolBuyLimitOrdersTotalVolume(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Returns the total volume of all the open buy limit orders matching the specified symbol and magic number. | ```<br>//Get the total buy limits volume/lot for the symbol<br>SymbolBuyLimitOrdersTotalVolume(_Symbol, 0);<br>//Get the total buy limits volume/lot for symbol and magic 101<br>SymbolBuyLimitOrdersTotalVolume(_Symbol, 101);<br>``` |
| ```<br>double SymbolSellStopOrdersTotalVolume(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Returns the total volume of all the open sell stop orders matching the specified symbol and magic number. | ```<br>//Get the total sell stops volume/lot for symbol<br>SymbolSellStopOrdersTotalVolume(_Symbol, 0);<br>//Get the total sell stops volume/lot for symbol and magic 101<br>SymbolSellStopOrdersTotalVolume(_Symbol, 101);<br>``` |
| ```<br>double SymbolSellLimitOrdersTotalVolume(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Returns the total volume of all the open sell limit orders matching the specified symbol and magic number. | ```<br>//Get the total sell limits volume/lot for symbol<br>SymbolSellLimitOrdersTotalVolume(_Symbol, 0);<br>//Get the total sell limits volume/lot for symbol and magic 101<br>SymbolSellLimitOrdersTotalVolume(_Symbol, 101);<br>``` |
| ```<br>string AccountOrdersStatus(<br>   bool formatForComment<br>);<br>``` | Prints a string formatted status of all open orders on the symbol chart or Experts tab in MetaTrader 5. | ```<br>//Print the status of all open orders <br>//formatted for the chart comments<br>Comment(AccountOrdersStatus(true));<br>//Print the status of all open orders <br>//formatted for the Experts tab<br>Print(AccountOrdersStatus(false));<br>//Activate an alert with the status of all <br>//open orders formatted for printing<br>Print(AccountOrdersStatus(false));<br>``` |
| ```<br>string MagicOrdersStatus(<br>   ulong magicNumber,<br>   bool formatForComment<br>);<br>``` | Prints a string formatted status of all open orders matching the specified magic number on the symbol chart or Experts tab in MetaTrader 5. | ```<br>//Print the status of all open orders matching <br>//magic number 101 formatted for the chart comments<br>Comment(MagicOrdersStatus(101, true));<br>//Print the status of all open orders matching<br>//magic number 101 formatted for the Experts tab<br>Print(MagicOrdersStatus(101, false));<br>//Activate an alert with the status of all open orders<br>//matching magic number 101 formatted for printing<br>Print(MagicOrdersStatus(101, false));<br>``` |
| ```<br>string SymbolOrdersStatus(<br>   string symbol,<br>   ulong magicNumber,<br>   bool formatForComment<br>);<br>``` | Prints a string formatted status of all open orders matching the specified symbol and magic number on the symbol chart or Experts tab in MetaTrader 5. | ```<br>//Print the status of all open orders matching<br>//the symbol and magic number 101 formatted for the chart comments<br>Comment(SymbolOrdersStatus(_Symbol, 101, true));<br>//Print the status of all open orders matching<br>//the symbol and magic number 101 formatted for the Experts tab<br>Print(SymbolOrdersStatus(_Symbol, 101, false));<br>//Activate an alert with the status of all open orders<br>//matching the symbol and magic number 101 formatted for printing<br>Print(SymbolOrdersStatus(_Symbol, 101, false));<br>``` |

With the library imported, you can now easily _open, modify, delete,_ or _retrieve pending order status data_ through simple function calls. To demonstrate this, let's create a practical graphical user interface (GUI) trading panel for managing pending orders in the next section.

### How to Develop a Pending Orders Panel (GUI) Powered by The Pending Orders Manager EX5 Library

In this section, we will develop a graphical user interface (GUI) for a _Pending Orders Panel_ Expert Advisor, which utilizes the Pending Orders Manager EX5 library to open, delete, and monitor all orders associated with a specified magic number tied to the panel. This example provides a practical demonstration of how to import and implement the EX5 library we just created in a real-world MQL5 application.

The Pending Orders Panel will leverage the MQL5 standard library for panels and dialogs, allowing us to keep the codebase minimal and efficient. Below is a picture of the final Pending Orders Panel GUI.

![MQL5 Pending Orders Panel (GUI)](https://c.mql5.com/2/96/PendingOrdersPanel_nGUIb__750px.png)

To start building the GUI, create a new Expert Advisor using the _MetaEditor IDE's MQL Wizard_, and name it _PendingOrdersPanel.mq5_. Since our Pending Orders Panel Expert Advisor will integrate the _PendingOrdersManager.ex5_ library, the first step is to import and include the library’s function prototype descriptions, as outlined earlier. Place the library import code just below the _#property_ directives. Given that the library contains numerous functions, we will only import the specific function prototypes required, which are listed in the code below.

```
#import "Toolkit/PendingOrdersManager.ex5" //-- Opening import directive
//-- Function descriptions for the imported function prototypes

//-- Pending Orders Execution and Modification Functions
bool OpenBuyLimit(ulong magicNumber, string symbol, double entryPrice, double lotSize, int sl, int tp, string orderComment);
bool OpenBuyStop(ulong magicNumber, string symbol, double entryPrice, double lotSize, int sl, int tp, string orderComment);
bool OpenSellLimit(ulong magicNumber, string symbol, double entryPrice, double lotSize, int sl, int tp, string orderComment);
bool OpenSellStop(ulong magicNumber, string symbol, double entryPrice, double lotSize, int sl, int tp, string orderComment);
//--
int MagicOrdersTotal(ulong magicNumber);
int MagicBuyStopOrdersTotal(ulong magicNumber);
int MagicBuyLimitOrdersTotal(ulong magicNumber);
int MagicSellStopOrdersTotal(ulong magicNumber);
int MagicSellLimitOrdersTotal(ulong magicNumber);
//--
bool DeleteAllBuyStops(string symbol, ulong magicNumber);
bool DeleteAllBuyLimits(string symbol, ulong magicNumber);
bool DeleteAllSellStops(string symbol, ulong magicNumber);
bool DeleteAllSellLimits(string symbol, ulong magicNumber);
bool DeleteAllMagicOrders(ulong magicNumber);

#import //--- Closing import directive
```

Create the global variables that we will use to store the order properties.

```
//-- Global variables
//-----------------------
ulong magicNo = 10101010;
double symbolPoint = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
long symbolDigits = SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
int spread = (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
double volumeLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
int sl = (int)spread * 50;
int tp = (int)spread * 100;
```

Next, we need to define some constants to store all the graphical object properties. This will make it easier for us to update or modify their values in one central place should we want to make any changes to the order panels' appearance.

```
//-- Define some values for the main panel
#define MAIN_PANEL_NAME string("Orders Panel - Trading: " + _Symbol + " - Magic No: " + IntegerToString(magicNo))
#define MAIN_PANEL_SUBWINDOW 0
#define MAIN_PANEL_X1 350
#define MAIN_PANEL_Y1 10
#define MAIN_PANEL_WIDTH int(800 + MAIN_PANEL_X1)
#define MAIN_PANEL_X2 MAIN_PANEL_WIDTH

//-- Define the GUI objects general properties
#define GUI_OBJECTS_MARGIN 5
#define GUI_OBJECTS_HEIGHT 40//40
#define GUI_OBJECTS_WIDTH int((MAIN_PANEL_WIDTH) / 7)
#define GUI_OBJECTS_FONT_SIZE 9//10
#define GUI_OBJECTS_HEADER_FONT_SIZE GUI_OBJECTS_FONT_SIZE
//-----
#define MAIN_PANEL_HEIGHT int(((GUI_OBJECTS_HEIGHT + (GUI_OBJECTS_MARGIN * 2)) * 10) + MAIN_PANEL_Y1)
#define MAIN_PANEL_Y2 MAIN_PANEL_HEIGHT

//-- Define the GUI objects colors
#define GUI_OBJECTS_HEADING_COLOR clrNavy

#define GUI_OBJECTS_BUY_BTN_COLOR clrWhite
#define GUI_OBJECTS_BUY_BTN_BG_COLOR clrBlue
#define GUI_OBJECTS_BUY_EDIT_COLOR clrBlue
#define GUI_OBJECTS_BUY_EDIT_BG_COLOR clrAliceBlue

#define GUI_OBJECTS_SELL_BTN_COLOR clrWhite
#define GUI_OBJECTS_SELL_BTN_BG_COLOR clrCrimson
#define GUI_OBJECTS_SELL_EDIT_COLOR clrMaroon
#define GUI_OBJECTS_SELL_EDIT_BG_COLOR clrMistyRose

/*------------------------------------------------------
* Define GUI components for the heading labels ****
*-----------------------------------------------------*/
//-- Define values for the lotVolHeaderLabel
#define VOLUME_LOT_LABEL_NAME "Volume Lot Header Label"
#define VOLUME_LOT_LABEL_SUBWINDOW 0
#define VOLUME_LOT_LABEL_X1 int(GUI_OBJECTS_MARGIN + GUI_OBJECTS_WIDTH)
#define VOLUME_LOT_LABEL_Y1 int(GUI_OBJECTS_MARGIN * 2)
#define VOLUME_LOT_LABEL_TEXT "VOLUME/LOT"

//-- Define values for the openPriceHeaderLabel
#define OPEN_PRICE_LABEL_NAME "Open Price Header Label"
#define OPEN_PRICE_LABEL_SUBWINDOW 0
#define OPEN_PRICE_LABEL_X1 (int(GUI_OBJECTS_WIDTH) * 2)
#define OPEN_PRICE_LABEL_Y1 int(GUI_OBJECTS_MARGIN * 2)
#define OPEN_PRICE_LABEL_TEXT "OPENING PRICE"

//-- Define values for the slHeaderLabel
#define SL_LABEL_NAME "Sl Header Label"
#define SL_LABEL_SUBWINDOW 0
#define SL_LABEL_X1 (int(GUI_OBJECTS_WIDTH) * 3)
#define SL_LABEL_Y1 int(GUI_OBJECTS_MARGIN * 2)
#define SL_LABEL_TEXT "SL (Pips)"

//-- Define values for the tpHeaderLabel
#define TP_LABEL_NAME "Tp Header Label"
#define TP_LABEL_SUBWINDOW 0
#define TP_LABEL_X1 (int(GUI_OBJECTS_WIDTH) * 3.75)
#define TP_LABEL_Y1 int(GUI_OBJECTS_MARGIN * 2)
#define TP_LABEL_TEXT "TP (Pips)"

/*------------------------------------------------------
* Define Buy Stop Order GUI components ****
*-----------------------------------------------------*/
//-- Define values for the buyStopBtn
#define BUY_STOP_BTN_NAME "Buy Stop Button"
#define BUY_STOP_BTN_SUBWINDOW 0
#define BUY_STOP_BTN_X1 GUI_OBJECTS_MARGIN
#define BUY_STOP_BTN_Y1 int(GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT)
#define BUY_STOP_BTN_TEXT "BUY STOP"

//-- Define values for the buyStopVolumeLotEdit
#define BUY_STOP_VOLUME_LOT_EDIT_NAME "Buy Stop Volume Lot Edit"
#define BUY_STOP_VOLUME_LOT_EDIT_SUBWINDOW 0
#define BUY_STOP_VOLUME_LOT_EDIT_X1 int(GUI_OBJECTS_MARGIN + GUI_OBJECTS_WIDTH)
#define BUY_STOP_VOLUME_LOT_EDIT_Y1 int(GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT)

//-- Define values for the buyStopOpenPriceEdit
#define BUY_STOP_OPEN_PRICE_EDIT_NAME "Buy Stop Open Price Edit"
#define BUY_STOP_OPEN_PRICE_EDIT_SUBWINDOW 0
#define BUY_STOP_OPEN_PRICE_EDIT_X1 int((GUI_OBJECTS_WIDTH) * 2)
#define BUY_STOP_OPEN_PRICE_EDIT_Y1 int(GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT)

//-- Define values for the buyStopSlEdit
#define BUY_STOP_SL_EDIT_NAME "Buy Stop SL Edit"
#define BUY_STOP_SL_EDIT_SUBWINDOW 0
#define BUY_STOP_SL_EDIT_X1 int(GUI_OBJECTS_WIDTH * 3)
#define BUY_STOP_SL_EDIT_Y1 int(GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT)

//-- Define values for the buyStopTpEdit
#define BUY_STOP_TP_EDIT_NAME "Buy Stop TP Edit"
#define BUY_STOP_TP_EDIT_SUBWINDOW 0
#define BUY_STOP_TP_EDIT_X1 int(GUI_OBJECTS_WIDTH * 3.7)
#define BUY_STOP_TP_EDIT_Y1 int(GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT)

/*------------------------------------------------------
* Define Sell Stop Order GUI components ****
*-----------------------------------------------------*/
//-- Define values for the sellStopBtn
#define SELL_STOP_BTN_NAME "Sell Stop Button"
#define SELL_STOP_BTN_SUBWINDOW 0
#define SELL_STOP_BTN_X1 GUI_OBJECTS_MARGIN
#define SELL_STOP_BTN_Y1 int((GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT) * 2)
#define SELL_STOP_BTN_TEXT "SELL STOP"

//-- Define values for the sellStopVolumeLotEdit
#define SELL_STOP_VOLUME_LOT_EDIT_NAME "Sell Stop Volume Lot Edit"
#define SELL_STOP_VOLUME_LOT_EDIT_SUBWINDOW 0
#define SELL_STOP_VOLUME_LOT_EDIT_X1 int(GUI_OBJECTS_MARGIN + GUI_OBJECTS_WIDTH)
#define SELL_STOP_VOLUME_LOT_EDIT_Y1 int((GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT) * 2)

//-- Define values for the sellStopOpenPriceEdit
#define SELL_STOP_OPEN_PRICE_EDIT_NAME "Sell Stop Open Price Edit"
#define SELL_STOP_OPEN_PRICE_EDIT_SUBWINDOW 0
#define SELL_STOP_OPEN_PRICE_EDIT_X1 int((GUI_OBJECTS_WIDTH) * 2)
#define SELL_STOP_OPEN_PRICE_EDIT_Y1 int((GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT) * 2)

//-- Define values for the sellStopSlEdit
#define SELL_STOP_SL_EDIT_NAME "Sell Stop SL Edit"
#define SELL_STOP_SL_EDIT_SUBWINDOW 0
#define SELL_STOP_SL_EDIT_X1 int(GUI_OBJECTS_WIDTH * 3)
#define SELL_STOP_SL_EDIT_Y1 int((GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT) * 2)

//-- Define values for the sellStopTpEdit
#define SELL_STOP_TP_EDIT_NAME "Sell Stop TP Edit"
#define SELL_STOP_TP_EDIT_SUBWINDOW 0
#define SELL_STOP_TP_EDIT_X1 int(GUI_OBJECTS_WIDTH * 3.7)
#define SELL_STOP_TP_EDIT_Y1 int((GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT) * 2)

/*------------------------------------------------------
* Define Buy Limit Order GUI components ****
*-----------------------------------------------------*/
//-- Define values for the buyLimitBtn
#define BUY_LIMIT_BTN_NAME "Buy Limit Button"
#define BUY_LIMIT_BTN_SUBWINDOW 0
#define BUY_LIMIT_BTN_X1 GUI_OBJECTS_MARGIN
#define BUY_LIMIT_BTN_Y1 int((GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT) * 3)
#define BUY_LIMIT_BTN_TEXT "BUY LIMIT"

//-- Define values for the buyLimitVolumeLotEdit
#define BUY_LIMIT_VOLUME_LOT_EDIT_NAME "Buy Limit Volume Lot Edit"
#define BUY_LIMIT_VOLUME_LOT_EDIT_SUBWINDOW 0
#define BUY_LIMIT_VOLUME_LOT_EDIT_X1 int(GUI_OBJECTS_MARGIN + GUI_OBJECTS_WIDTH)
#define BUY_LIMIT_VOLUME_LOT_EDIT_Y1 int((GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT) * 3)

//-- Define values for the buySLimitOpenPriceEdit
#define BUY_LIMIT_OPEN_PRICE_EDIT_NAME "Buy Limit Open Price Edit"
#define BUY_LIMIT_OPEN_PRICE_EDIT_SUBWINDOW 0
#define BUY_LIMIT_OPEN_PRICE_EDIT_X1 int((GUI_OBJECTS_WIDTH) * 2)
#define BUY_LIMIT_OPEN_PRICE_EDIT_Y1 int((GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT) * 3)

//-- Define values for the buyLimitSlEdit
#define BUY_LIMIT_SL_EDIT_NAME "Buy Limit SL Edit"
#define BUY_LIMIT_SL_EDIT_SUBWINDOW 0
#define BUY_LIMIT_SL_EDIT_X1 int(GUI_OBJECTS_WIDTH * 3)
#define BUY_LIMIT_SL_EDIT_Y1 int((GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT) * 3)

//-- Define values for the buyLimitTpEdit
#define BUY_LIMIT_TP_EDIT_NAME "Buy Limit TP Edit"
#define BUY_LIMIT_TP_EDIT_SUBWINDOW 0
#define BUY_LIMIT_TP_EDIT_X1 int(GUI_OBJECTS_WIDTH * 3.7)
#define BUY_LIMIT_TP_EDIT_Y1 int((GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT) * 3)

/*------------------------------------------------------
* Define Sell Limit Order GUI components ****
*-----------------------------------------------------*/
//-- Define values for the sellLimitBtn
#define SELL_LIMIT_BTN_NAME "Sell Limit Button"
#define SELL_LIMIT_BTN_SUBWINDOW 0
#define SELL_LIMIT_BTN_X1 GUI_OBJECTS_MARGIN
#define SELL_LIMIT_BTN_Y1 int((GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT) * 4)
#define SELL_LIMIT_BTN_TEXT "SELL LIMIT"

//-- Define values for the sellLimitVolumeLotEdit
#define SELL_LIMIT_VOLUME_LOT_EDIT_NAME "Sell Limit Volume Lot Edit"
#define SELL_LIMIT_VOLUME_LOT_EDIT_SUBWINDOW 0
#define SELL_LIMIT_VOLUME_LOT_EDIT_X1 int(GUI_OBJECTS_MARGIN + GUI_OBJECTS_WIDTH)
#define SELL_LIMIT_VOLUME_LOT_EDIT_Y1 int((GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT) * 4)

//-- Define values for the sellLimitOpenPriceEdit
#define SELL_LIMIT_OPEN_PRICE_EDIT_NAME "Sell Limit Open Price Edit"
#define SELL_LIMIT_OPEN_PRICE_EDIT_SUBWINDOW 0
#define SELL_LIMIT_OPEN_PRICE_EDIT_X1 int((GUI_OBJECTS_WIDTH) * 2)
#define SELL_LIMIT_OPEN_PRICE_EDIT_Y1 int((GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT) * 4)

//-- Define values for the sellLimitSlEdit
#define SELL_LIMIT_SL_EDIT_NAME "Sell Limit SL Edit"
#define SELL_LIMIT_SL_EDIT_SUBWINDOW 0
#define SELL_LIMIT_SL_EDIT_X1 int(GUI_OBJECTS_WIDTH * 3)
#define SELL_LIMIT_SL_EDIT_Y1 int((GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT) * 4)

//-- Define values for the sellLimitTpEdit
#define SELL_LIMIT_TP_EDIT_NAME "Sell Limit TP Edit"
#define SELL_LIMIT_TP_EDIT_SUBWINDOW 0
#define SELL_LIMIT_TP_EDIT_X1 int(GUI_OBJECTS_WIDTH * 3.7)
#define SELL_LIMIT_TP_EDIT_Y1 int((GUI_OBJECTS_MARGIN + GUI_OBJECTS_HEIGHT) * 4)

/*------------------------------------------------------
* Define Order Status GUI components ****
*-----------------------------------------------------*/
//-- Define values for the orders status
#define STATUS_HEADER_FONT_SIZE int(GUI_OBJECTS_FONT_SIZE)// / 1.1)
#define STATUS_EDIT_FONT_SIZE int(GUI_OBJECTS_FONT_SIZE)// / 1.1)
#define STATUS_EDIT_WIDTH int((MAIN_PANEL_WIDTH / 1.485) - (GUI_OBJECTS_MARGIN * 2))
#define STATUS_EDIT_COLOR clrBlack
#define STATUS_EDIT_BG_COLOR clrLemonChiffon
#define STATUS_EDIT_BORDER_COLOR clrMidnightBlue
#define DELETE_ORDERS_BTN_COLOR clrLightYellow
#define DELETE_BUY_ORDERS_BTN_BG_COLOR clrRoyalBlue
#define DELETE_SELL_ORDERS_BTN_BG_COLOR clrCrimson
#define DELETE_ALL_ORDERS_BTN_BG_COLOR clrMediumVioletRed
#define DELETE_ORDERS_BTN_BORDER_COLOR clrBlack
#define DELETE_ORDERS_BTN_WIDTH int((STATUS_EDIT_WIDTH / 1.93) - (GUI_OBJECTS_MARGIN * 3))
#define DELETE_ORDERS_BTN_FONT_SIZE int((GUI_OBJECTS_FONT_SIZE))// / 1.05)

//-- Define values for the magicOrderStatusLabel
#define MAGIC_ORDER_STATUS_LABEL_NAME "Magic Order Status Label"
#define MAGIC_ORDER_STATUS_LABEL_SUBWINDOW 0
#define MAGIC_ORDER_STATUS_LABEL_X1 int(GUI_OBJECTS_MARGIN * 3)
#define MAGIC_ORDER_STATUS_LABEL_Y1 int((GUI_OBJECTS_HEIGHT * 6) + (GUI_OBJECTS_MARGIN * 2))
#define MAGIC_ORDER_STATUS_LABEL_TEXT string("MAGIC No: " + IntegerToString(magicNo) + " - TOTAL OPEN ORDERS: ")

//-- Define values for the magicOrdersStatusEdit
#define MAGIC_ORDER_STATUS_EDIT_NAME "Magic Order Status Edit"
#define MAGIC_ORDER_STATUS_EDIT_SUBWINDOW 0
#define MAGIC_ORDER_STATUS_EDIT_X1 int(GUI_OBJECTS_MARGIN * 2)
#define MAGIC_ORDER_STATUS_EDIT_Y1 int((MAGIC_ORDER_STATUS_LABEL_Y1) + (GUI_OBJECTS_HEIGHT / 1.7))

//-- Define values for the deleteAllMagicBuyStopsBtn
#define DELETE_ALL_MAGIC_BUY_STOPS_BTN_NAME "Delete All Magic Buy Stops Btn"
#define DELETE_ALL_MAGIC_BUY_STOPS_BTN_SUBWINDOW 0
#define DELETE_ALL_MAGIC_BUY_STOPS_BTN_X1 int(GUI_OBJECTS_MARGIN * 2)
#define DELETE_ALL_MAGIC_BUY_STOPS_BTN_Y1 int((MAGIC_ORDER_STATUS_EDIT_Y1) + (GUI_OBJECTS_HEIGHT + GUI_OBJECTS_MARGIN))
#define DELETE_ALL_MAGIC_BUY_STOPS_BTN_TEXT "DELETE ALL MAGIC BUY STOPS"

//-- Define values for the deleteAllMagicSellStopsBtn
#define DELETE_ALL_MAGIC_SELL_STOPS_BTN_NAME "Delete All Magic Sell Stops Btn"
#define DELETE_ALL_MAGIC_SELL_STOPS_BTN_SUBWINDOW 0
#define DELETE_ALL_MAGIC_SELL_STOPS_BTN_X1 int((GUI_OBJECTS_MARGIN * 3) + DELETE_ORDERS_BTN_WIDTH)
#define DELETE_ALL_MAGIC_SELL_STOPS_BTN_Y1 int((MAGIC_ORDER_STATUS_EDIT_Y1) + (GUI_OBJECTS_HEIGHT + GUI_OBJECTS_MARGIN))
#define DELETE_ALL_MAGIC_SELL_STOPS_BTN_TEXT "DELETE ALL MAGIC SELL STOPS"

//-- Define values for the deleteAllMagicBuyLimitsBtn
#define DELETE_ALL_MAGIC_BUY_LIMITS_BTN_NAME "Delete All Magic Buy Limits Btn"
#define DELETE_ALL_MAGIC_BUY_LIMITS_BTN_SUBWINDOW 0
#define DELETE_ALL_MAGIC_BUY_LIMITS_BTN_X1 int(GUI_OBJECTS_MARGIN * 2)
#define DELETE_ALL_MAGIC_BUY_LIMITS_BTN_Y1 int((DELETE_ALL_MAGIC_BUY_STOPS_BTN_Y1) + (GUI_OBJECTS_HEIGHT + GUI_OBJECTS_MARGIN))
#define DELETE_ALL_MAGIC_BUY_LIMITS_BTN_TEXT "DELETE ALL MAGIC BUY LIMITS"

//-- Define values for the deleteAllMagicSellLimitsBtn
#define DELETE_ALL_MAGIC_SELL_LIMITS_BTN_NAME "Delete All Magic Sell Limits Btn"
#define DELETE_ALL_MAGIC_SELL_LIMITS_BTN_SUBWINDOW 0
#define DELETE_ALL_MAGIC_SELL_LIMITS_BTN_X1 int((GUI_OBJECTS_MARGIN * 3) + DELETE_ORDERS_BTN_WIDTH)
#define DELETE_ALL_MAGIC_SELL_LIMITS_BTN_Y1 DELETE_ALL_MAGIC_BUY_LIMITS_BTN_Y1//int((MAGIC_ORDER_STATUS_EDIT_Y1) + (GUI_OBJECTS_HEIGHT + GUI_OBJECTS_MARGIN))
#define DELETE_ALL_MAGIC_SELL_LIMITS_BTN_TEXT "DELETE ALL MAGIC SELL LIMITS"

//-- Define values for the deleteAllMagicOrdersBtn
#define DELETE_ALL_MAGIC_ORDERS_BTN_NAME "Delete All Magic Orders Btn"
#define DELETE_ALL_MAGIC_ORDERS_BTN_SUBWINDOW 0
#define DELETE_ALL_MAGIC_ORDERS_BTN_X1 int(GUI_OBJECTS_MARGIN * 2)
#define DELETE_ALL_MAGIC_ORDERS_BTN_Y1 int((DELETE_ALL_MAGIC_BUY_LIMITS_BTN_Y1) + (GUI_OBJECTS_HEIGHT + GUI_OBJECTS_MARGIN))
#define DELETE_ALL_MAGIC_ORDERS_BTN_TEXT "DELETE ALL MAGIC PENDING ORDERS"
```

Add or include the MQL5 standard classes for panels and dialogs to our code.

```
//-- Include the MQL5 standard library for panels and dialogs
#include <Controls\Dialog.mqh>
#include <Controls\Button.mqh>
#include <Controls\Label.mqh>
#include <Controls\Edit.mqh>
```

With the panel and dialog classes now included and available in our file, we can proceed to create their objects to extend their functionality within our code.

```
//-- Create objects for the included standard classes
CAppDialog mainPanelWindow;

//-- Create the header label components
CLabel lotVolHeaderLabel;
CLabel openPriceHeaderLabel;
CLabel slHeaderLabel;
CLabel tpHeaderLabel;

//-- Create the buy stop GUI components
//--BuyStopBtn
CButton buyStopBtn;
//--BuyStopEdits
CEdit buyStopVolumeLotEdit;
CEdit buyStopOpenPriceEdit;
CEdit buyStopSlEdit;
CEdit buyStopTpEdit;

//-- Create the sell stop GUI components
//--SellStopBtn
CButton sellStopBtn;
//--sellStopEdits
CEdit sellStopVolumeLotEdit;
CEdit sellStopOpenPriceEdit;
CEdit sellStopSlEdit;
CEdit sellStopTpEdit;

//-- Create the buy limit GUI components
//--BuyLimitBtn
CButton buyLimitBtn;
//--BuyLimitEdits
CEdit buyLimitVolumeLotEdit;
CEdit buyLimitOpenPriceEdit;
CEdit buyLimitSlEdit;
CEdit buyLimitTpEdit;

//-- Create the sell limit GUI components
//--sellLimitBtn
CButton sellLimitBtn;
//--sellLimitEdits
CEdit sellLimitVolumeLotEdit;
CEdit sellLimitOpenPriceEdit;
CEdit sellLimitSlEdit;
CEdit sellLimitTpEdit;

//-- Create the order status GUI components
//--magic order status
CLabel magicOrderStatusLabel;
CEdit magicOrdersStatusEdit;
//--Magic orders delete buttons
CButton deleteAllMagicBuyStopsBtn;
CButton deleteAllMagicSellStopsBtn;
CButton deleteAllMagicBuyLimitsBtn;
CButton deleteAllMagicSellLimitsBtn;
CButton deleteAllMagicOrdersBtn;
```

Create the variables that will be responsible for storing the order entry prices and a string to store the order status of all orders opened by our Expert Advisor.

```
//-- Default starting entry prices for different pending orders
double buyStopEntryPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK) + ((spread * 20) * symbolPoint);
double buyLimitEntryPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - ((spread * 20) * symbolPoint);
double sellStopEntryPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - ((spread * 20) * symbolPoint);
double sellLimitEntryPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK) + ((spread * 20) * symbolPoint);

//-- String values for the orders status
string magicOrderStatus;
```

With the head section of our code complete, the next step is to create the function responsible for generating and loading the graphical user interface during the Expert Advisor’s initialization. We will name this function _CreateGui()_.

```
void CreateGui()
  {
//-- Create the orders panel
   mainPanelWindow.Create(
      0, MAIN_PANEL_NAME, MAIN_PANEL_SUBWINDOW,
      MAIN_PANEL_X1, MAIN_PANEL_Y1, MAIN_PANEL_X2, MAIN_PANEL_Y2
   );
   /*------------------------------------------------------
   * Header Labels GUI components creation ****
   *-----------------------------------------------------*/
//--Create the lot volume header label
   lotVolHeaderLabel.Create(
      0, VOLUME_LOT_LABEL_NAME, VOLUME_LOT_LABEL_SUBWINDOW,
      VOLUME_LOT_LABEL_X1,
      VOLUME_LOT_LABEL_Y1,
      GUI_OBJECTS_WIDTH,
      GUI_OBJECTS_HEIGHT
   );
   lotVolHeaderLabel.Text(VOLUME_LOT_LABEL_TEXT);
   lotVolHeaderLabel.Color(GUI_OBJECTS_HEADING_COLOR);
   lotVolHeaderLabel.FontSize(GUI_OBJECTS_HEADER_FONT_SIZE);
   mainPanelWindow.Add(lotVolHeaderLabel);

//--Create the open price header label
   openPriceHeaderLabel.Create(
      0, OPEN_PRICE_LABEL_NAME, OPEN_PRICE_LABEL_SUBWINDOW,
      OPEN_PRICE_LABEL_X1, OPEN_PRICE_LABEL_Y1,
      GUI_OBJECTS_WIDTH, GUI_OBJECTS_HEIGHT
   );
   openPriceHeaderLabel.Text(OPEN_PRICE_LABEL_TEXT);
   openPriceHeaderLabel.Color(GUI_OBJECTS_HEADING_COLOR);
   openPriceHeaderLabel.FontSize(GUI_OBJECTS_HEADER_FONT_SIZE);
   mainPanelWindow.Add(openPriceHeaderLabel);

//--Create the sl header label
   slHeaderLabel.Create(
      0, SL_LABEL_NAME, SL_LABEL_SUBWINDOW,
      SL_LABEL_X1, SL_LABEL_Y1,
      int(GUI_OBJECTS_WIDTH / 1.4), GUI_OBJECTS_HEIGHT
   );
   slHeaderLabel.Text(SL_LABEL_TEXT);
   slHeaderLabel.Color(GUI_OBJECTS_HEADING_COLOR);
   slHeaderLabel.FontSize(GUI_OBJECTS_HEADER_FONT_SIZE);
   mainPanelWindow.Add(slHeaderLabel);

//--Create the tp header label
   tpHeaderLabel.Create(
      0, TP_LABEL_NAME, TP_LABEL_SUBWINDOW,
      TP_LABEL_X1, TP_LABEL_Y1,
      int(GUI_OBJECTS_WIDTH / 1.4), GUI_OBJECTS_HEIGHT
   );
   tpHeaderLabel.Text(TP_LABEL_TEXT);
   tpHeaderLabel.Color(GUI_OBJECTS_HEADING_COLOR);
   tpHeaderLabel.FontSize(GUI_OBJECTS_HEADER_FONT_SIZE);
   mainPanelWindow.Add(tpHeaderLabel);

   /*------------------------------------------------------
   * Buy Stop Order GUI components creation ****
   *-----------------------------------------------------*/
//--Create the open buy stop button
   buyStopBtn.Create(
      0, BUY_STOP_BTN_NAME, BUY_STOP_BTN_SUBWINDOW,
      BUY_STOP_BTN_X1, BUY_STOP_BTN_Y1,
      0, 0
   );
   buyStopBtn.Text(BUY_STOP_BTN_TEXT);
   buyStopBtn.Width(GUI_OBJECTS_WIDTH);
   buyStopBtn.Height(GUI_OBJECTS_HEIGHT);
   buyStopBtn.Color(GUI_OBJECTS_BUY_BTN_COLOR);
   buyStopBtn.ColorBackground(GUI_OBJECTS_BUY_BTN_BG_COLOR);
   buyStopBtn.FontSize(GUI_OBJECTS_FONT_SIZE);
   mainPanelWindow.Add(buyStopBtn);

//--Create the buy stop volume lot edit to get the buy stop volume/lot user input
   buyStopVolumeLotEdit.Create(
      0, BUY_STOP_VOLUME_LOT_EDIT_NAME, BUY_STOP_VOLUME_LOT_EDIT_SUBWINDOW,
      BUY_STOP_VOLUME_LOT_EDIT_X1, BUY_STOP_VOLUME_LOT_EDIT_Y1,
      0, 0
   );
   buyStopVolumeLotEdit.Text(DoubleToString(volumeLot));
   buyStopVolumeLotEdit.Width(GUI_OBJECTS_WIDTH);
   buyStopVolumeLotEdit.Height(GUI_OBJECTS_HEIGHT);
   buyStopVolumeLotEdit.Color(GUI_OBJECTS_BUY_EDIT_COLOR);
   buyStopVolumeLotEdit.ColorBackground(GUI_OBJECTS_BUY_EDIT_BG_COLOR);
   buyStopVolumeLotEdit.FontSize(GUI_OBJECTS_FONT_SIZE);
   buyStopVolumeLotEdit.ColorBorder(GUI_OBJECTS_BUY_BTN_BG_COLOR);
   mainPanelWindow.Add(buyStopVolumeLotEdit);

//--Create the buy stop price edit to get the buy stop opening price user input
   buyStopOpenPriceEdit.Create(
      0, BUY_STOP_OPEN_PRICE_EDIT_NAME, BUY_STOP_OPEN_PRICE_EDIT_SUBWINDOW,
      BUY_STOP_OPEN_PRICE_EDIT_X1, BUY_STOP_OPEN_PRICE_EDIT_Y1,
      0, 0
   );
   buyStopOpenPriceEdit.Text(DoubleToString(buyStopEntryPrice, int(symbolDigits)));
   buyStopOpenPriceEdit.Width(GUI_OBJECTS_WIDTH);
   buyStopOpenPriceEdit.Height(GUI_OBJECTS_HEIGHT);
   buyStopOpenPriceEdit.Color(GUI_OBJECTS_BUY_EDIT_COLOR);
   buyStopOpenPriceEdit.ColorBackground(GUI_OBJECTS_BUY_EDIT_BG_COLOR);
   buyStopOpenPriceEdit.FontSize(GUI_OBJECTS_FONT_SIZE);
   buyStopOpenPriceEdit.ColorBorder(GUI_OBJECTS_BUY_BTN_BG_COLOR);
   mainPanelWindow.Add(buyStopOpenPriceEdit);

//--Create the buy stop sl edit to get the buy stop sl user input
   buyStopSlEdit.Create(
      0, BUY_STOP_SL_EDIT_NAME, BUY_STOP_SL_EDIT_SUBWINDOW,
      BUY_STOP_SL_EDIT_X1, BUY_STOP_SL_EDIT_Y1,
      0, 0
   );
   buyStopSlEdit.Text(IntegerToString(sl));
   buyStopSlEdit.Width(int(GUI_OBJECTS_WIDTH / 1.4));
   buyStopSlEdit.Height(GUI_OBJECTS_HEIGHT);
   buyStopSlEdit.Color(GUI_OBJECTS_BUY_EDIT_COLOR);
   buyStopSlEdit.ColorBackground(GUI_OBJECTS_BUY_EDIT_BG_COLOR);
   buyStopSlEdit.FontSize(GUI_OBJECTS_FONT_SIZE);
   buyStopSlEdit.ColorBorder(GUI_OBJECTS_BUY_BTN_BG_COLOR);
   mainPanelWindow.Add(buyStopSlEdit);

//--Create the buy stop tp edit to get the buy stop tp user input
   buyStopTpEdit.Create(
      0, BUY_STOP_TP_EDIT_NAME, BUY_STOP_TP_EDIT_SUBWINDOW,
      BUY_STOP_TP_EDIT_X1, BUY_STOP_TP_EDIT_Y1,
      0, 0
   );
   buyStopTpEdit.Text(IntegerToString(tp));
   buyStopTpEdit.Width(GUI_OBJECTS_WIDTH);
   buyStopTpEdit.Height(GUI_OBJECTS_HEIGHT);
   buyStopTpEdit.Color(GUI_OBJECTS_BUY_EDIT_COLOR);
   buyStopTpEdit.ColorBackground(GUI_OBJECTS_BUY_EDIT_BG_COLOR);
   buyStopTpEdit.FontSize(GUI_OBJECTS_FONT_SIZE);
   buyStopTpEdit.ColorBorder(GUI_OBJECTS_BUY_BTN_BG_COLOR);
   mainPanelWindow.Add(buyStopTpEdit);

   /*------------------------------------------------------
   * Sell Stop Order GUI components creation ****
   *-----------------------------------------------------*/
//--Create the open sell stop button
   sellStopBtn.Create(
      0, SELL_STOP_BTN_NAME, SELL_STOP_BTN_SUBWINDOW,
      SELL_STOP_BTN_X1, SELL_STOP_BTN_Y1,
      0, 0
   );
   sellStopBtn.Text(SELL_STOP_BTN_TEXT);
   sellStopBtn.Width(GUI_OBJECTS_WIDTH);
   sellStopBtn.Height(GUI_OBJECTS_HEIGHT);
   sellStopBtn.Color(GUI_OBJECTS_SELL_BTN_COLOR);
   sellStopBtn.ColorBackground(GUI_OBJECTS_SELL_BTN_BG_COLOR);
   sellStopBtn.FontSize(GUI_OBJECTS_FONT_SIZE);
   mainPanelWindow.Add(sellStopBtn);

//--Create the sell stop volume lot edit to get the sell stop volume/lot user input
   sellStopVolumeLotEdit.Create(
      0, SELL_STOP_VOLUME_LOT_EDIT_NAME, SELL_STOP_VOLUME_LOT_EDIT_SUBWINDOW,
      SELL_STOP_VOLUME_LOT_EDIT_X1, SELL_STOP_VOLUME_LOT_EDIT_Y1,
      0, 0
   );
   sellStopVolumeLotEdit.Text(DoubleToString(volumeLot));
   sellStopVolumeLotEdit.Width(GUI_OBJECTS_WIDTH);
   sellStopVolumeLotEdit.Height(GUI_OBJECTS_HEIGHT);
   sellStopVolumeLotEdit.Color(GUI_OBJECTS_SELL_EDIT_COLOR);
   sellStopVolumeLotEdit.ColorBackground(GUI_OBJECTS_SELL_EDIT_BG_COLOR);
   sellStopVolumeLotEdit.FontSize(GUI_OBJECTS_FONT_SIZE);
   sellStopVolumeLotEdit.ColorBorder(GUI_OBJECTS_SELL_BTN_BG_COLOR);
   mainPanelWindow.Add(sellStopVolumeLotEdit);

//--Create the sell stop price edit to get the sell stop opening price user input
   sellStopOpenPriceEdit.Create(
      0, SELL_STOP_OPEN_PRICE_EDIT_NAME, SELL_STOP_OPEN_PRICE_EDIT_SUBWINDOW,
      SELL_STOP_OPEN_PRICE_EDIT_X1, SELL_STOP_OPEN_PRICE_EDIT_Y1,
      0, 0
   );
   sellStopOpenPriceEdit.Text(DoubleToString(sellStopEntryPrice, int(symbolDigits)));
   sellStopOpenPriceEdit.Width(GUI_OBJECTS_WIDTH);
   sellStopOpenPriceEdit.Height(GUI_OBJECTS_HEIGHT);
   sellStopOpenPriceEdit.Color(GUI_OBJECTS_SELL_EDIT_COLOR);
   sellStopOpenPriceEdit.ColorBackground(GUI_OBJECTS_SELL_EDIT_BG_COLOR);
   sellStopOpenPriceEdit.FontSize(GUI_OBJECTS_FONT_SIZE);
   sellStopOpenPriceEdit.ColorBorder(GUI_OBJECTS_SELL_BTN_BG_COLOR);
   mainPanelWindow.Add(sellStopOpenPriceEdit);

//--Create the sell stop sl edit to get the sell stop sl user input
   sellStopSlEdit.Create(
      0, SELL_STOP_SL_EDIT_NAME, SELL_STOP_SL_EDIT_SUBWINDOW,
      SELL_STOP_SL_EDIT_X1, SELL_STOP_SL_EDIT_Y1,
      0, 0
   );
   sellStopSlEdit.Text(IntegerToString(sl));
   sellStopSlEdit.Width(int(GUI_OBJECTS_WIDTH / 1.4));
   sellStopSlEdit.Height(GUI_OBJECTS_HEIGHT);
   sellStopSlEdit.Color(GUI_OBJECTS_SELL_EDIT_COLOR);
   sellStopSlEdit.ColorBackground(GUI_OBJECTS_SELL_EDIT_BG_COLOR);
   sellStopSlEdit.FontSize(GUI_OBJECTS_FONT_SIZE);
   sellStopSlEdit.ColorBorder(GUI_OBJECTS_SELL_BTN_BG_COLOR);
   mainPanelWindow.Add(sellStopSlEdit);

//--Create the sell stop tp edit to get the sell stop tp user input
   sellStopTpEdit.Create(
      0, SELL_STOP_TP_EDIT_NAME, SELL_STOP_TP_EDIT_SUBWINDOW,
      SELL_STOP_TP_EDIT_X1, SELL_STOP_TP_EDIT_Y1,
      0, 0
   );
   sellStopTpEdit.Text(IntegerToString(tp));
   sellStopTpEdit.Width(GUI_OBJECTS_WIDTH);
   sellStopTpEdit.Height(GUI_OBJECTS_HEIGHT);
   sellStopTpEdit.Color(GUI_OBJECTS_SELL_EDIT_COLOR);
   sellStopTpEdit.ColorBackground(GUI_OBJECTS_SELL_EDIT_BG_COLOR);
   sellStopTpEdit.FontSize(GUI_OBJECTS_FONT_SIZE);
   sellStopTpEdit.ColorBorder(GUI_OBJECTS_SELL_BTN_BG_COLOR);
   mainPanelWindow.Add(sellStopTpEdit);

   /*------------------------------------------------------
   * Buy Limit Order GUI components creation ****
   *-----------------------------------------------------*/
//--Create the open buy limit button
   buyLimitBtn.Create(
      0, BUY_LIMIT_BTN_NAME, BUY_LIMIT_BTN_SUBWINDOW,
      BUY_LIMIT_BTN_X1, BUY_LIMIT_BTN_Y1,
      0, 0
   );
   buyLimitBtn.Text(BUY_LIMIT_BTN_TEXT);
   buyLimitBtn.Width(GUI_OBJECTS_WIDTH);
   buyLimitBtn.Height(GUI_OBJECTS_HEIGHT);
   buyLimitBtn.Color(GUI_OBJECTS_BUY_BTN_COLOR);
   buyLimitBtn.ColorBackground(GUI_OBJECTS_BUY_BTN_BG_COLOR);
   buyLimitBtn.FontSize(GUI_OBJECTS_FONT_SIZE);
   mainPanelWindow.Add(buyLimitBtn);

//--Create the buy limit volume lot edit to get the buy limit volume/lot user input
   buyLimitVolumeLotEdit.Create(
      0, BUY_LIMIT_VOLUME_LOT_EDIT_NAME, BUY_LIMIT_VOLUME_LOT_EDIT_SUBWINDOW,
      BUY_LIMIT_VOLUME_LOT_EDIT_X1, BUY_LIMIT_VOLUME_LOT_EDIT_Y1,
      0, 0
   );
   buyLimitVolumeLotEdit.Text(DoubleToString(volumeLot));
   buyLimitVolumeLotEdit.Width(GUI_OBJECTS_WIDTH);
   buyLimitVolumeLotEdit.Height(GUI_OBJECTS_HEIGHT);
   buyLimitVolumeLotEdit.Color(GUI_OBJECTS_BUY_EDIT_COLOR);
   buyLimitVolumeLotEdit.ColorBackground(GUI_OBJECTS_BUY_EDIT_BG_COLOR);
   buyLimitVolumeLotEdit.FontSize(GUI_OBJECTS_FONT_SIZE);
   buyLimitVolumeLotEdit.ColorBorder(GUI_OBJECTS_BUY_BTN_BG_COLOR);
   mainPanelWindow.Add(buyLimitVolumeLotEdit);

//--Create the buy limit price edit to get the buy limit opening price user input
   buyLimitOpenPriceEdit.Create(
      0, BUY_LIMIT_OPEN_PRICE_EDIT_NAME, BUY_LIMIT_OPEN_PRICE_EDIT_SUBWINDOW,
      BUY_LIMIT_OPEN_PRICE_EDIT_X1, BUY_LIMIT_OPEN_PRICE_EDIT_Y1,
      0, 0
   );
   buyLimitOpenPriceEdit.Text(DoubleToString(buyLimitEntryPrice, int(symbolDigits)));
   buyLimitOpenPriceEdit.Width(GUI_OBJECTS_WIDTH);
   buyLimitOpenPriceEdit.Height(GUI_OBJECTS_HEIGHT);
   buyLimitOpenPriceEdit.Color(GUI_OBJECTS_BUY_EDIT_COLOR);
   buyLimitOpenPriceEdit.ColorBackground(GUI_OBJECTS_BUY_EDIT_BG_COLOR);
   buyLimitOpenPriceEdit.FontSize(GUI_OBJECTS_FONT_SIZE);
   buyLimitOpenPriceEdit.ColorBorder(GUI_OBJECTS_BUY_BTN_BG_COLOR);
   mainPanelWindow.Add(buyLimitOpenPriceEdit);

//--Create the buy limit sl edit to get the buy limit sl user input
   buyLimitSlEdit.Create(
      0, BUY_LIMIT_SL_EDIT_NAME, BUY_LIMIT_SL_EDIT_SUBWINDOW,
      BUY_LIMIT_SL_EDIT_X1, BUY_LIMIT_SL_EDIT_Y1,
      0, 0
   );
   buyLimitSlEdit.Text(IntegerToString(sl));
   buyLimitSlEdit.Width(int(GUI_OBJECTS_WIDTH / 1.4));
   buyLimitSlEdit.Height(GUI_OBJECTS_HEIGHT);
   buyLimitSlEdit.Color(GUI_OBJECTS_BUY_EDIT_COLOR);
   buyLimitSlEdit.ColorBackground(GUI_OBJECTS_BUY_EDIT_BG_COLOR);
   buyLimitSlEdit.FontSize(GUI_OBJECTS_FONT_SIZE);
   buyLimitSlEdit.ColorBorder(GUI_OBJECTS_BUY_BTN_BG_COLOR);
   mainPanelWindow.Add(buyLimitSlEdit);

//--Create the buy limit tp edit to get the buy limit tp user input
   buyLimitTpEdit.Create(
      0, BUY_LIMIT_TP_EDIT_NAME, BUY_LIMIT_TP_EDIT_SUBWINDOW,
      BUY_LIMIT_TP_EDIT_X1, BUY_LIMIT_TP_EDIT_Y1,
      0, 0
   );
   buyLimitTpEdit.Text(IntegerToString(tp));
   buyLimitTpEdit.Width(GUI_OBJECTS_WIDTH);
   buyLimitTpEdit.Height(GUI_OBJECTS_HEIGHT);
   buyLimitTpEdit.Color(GUI_OBJECTS_BUY_EDIT_COLOR);
   buyLimitTpEdit.ColorBackground(GUI_OBJECTS_BUY_EDIT_BG_COLOR);
   buyLimitTpEdit.FontSize(GUI_OBJECTS_FONT_SIZE);
   buyLimitTpEdit.ColorBorder(GUI_OBJECTS_BUY_BTN_BG_COLOR);
   mainPanelWindow.Add(buyLimitTpEdit);

   /*------------------------------------------------------
   * Sell Limit Order GUI components creation ****
   *-----------------------------------------------------*/
//--Create the open sell limit button
   sellLimitBtn.Create(
      0, SELL_LIMIT_BTN_NAME, SELL_LIMIT_BTN_SUBWINDOW,
      SELL_LIMIT_BTN_X1, SELL_LIMIT_BTN_Y1,
      0, 0
   );
   sellLimitBtn.Text(SELL_LIMIT_BTN_TEXT);
   sellLimitBtn.Width(GUI_OBJECTS_WIDTH);
   sellLimitBtn.Height(GUI_OBJECTS_HEIGHT);
   sellLimitBtn.Color(GUI_OBJECTS_SELL_BTN_COLOR);
   sellLimitBtn.ColorBackground(GUI_OBJECTS_SELL_BTN_BG_COLOR);
   sellLimitBtn.FontSize(GUI_OBJECTS_FONT_SIZE);
   mainPanelWindow.Add(sellLimitBtn);

//--Create the sell limit volume lot edit to get the sell limit volume/lot user input
   sellLimitVolumeLotEdit.Create(
      0, SELL_LIMIT_VOLUME_LOT_EDIT_NAME, SELL_LIMIT_VOLUME_LOT_EDIT_SUBWINDOW,
      SELL_LIMIT_VOLUME_LOT_EDIT_X1, SELL_LIMIT_VOLUME_LOT_EDIT_Y1,
      0, 0
   );
   sellLimitVolumeLotEdit.Text(DoubleToString(volumeLot));
   sellLimitVolumeLotEdit.Width(GUI_OBJECTS_WIDTH);
   sellLimitVolumeLotEdit.Height(GUI_OBJECTS_HEIGHT);
   sellLimitVolumeLotEdit.Color(GUI_OBJECTS_SELL_EDIT_COLOR);
   sellLimitVolumeLotEdit.ColorBackground(GUI_OBJECTS_SELL_EDIT_BG_COLOR);
   sellLimitVolumeLotEdit.FontSize(GUI_OBJECTS_FONT_SIZE);
   sellLimitVolumeLotEdit.ColorBorder(GUI_OBJECTS_SELL_BTN_BG_COLOR);
   mainPanelWindow.Add(sellLimitVolumeLotEdit);

//--Create the sell limit price edit to get the sell limit opening price user input
   sellLimitOpenPriceEdit.Create(
      0, SELL_LIMIT_OPEN_PRICE_EDIT_NAME, SELL_LIMIT_OPEN_PRICE_EDIT_SUBWINDOW,
      SELL_LIMIT_OPEN_PRICE_EDIT_X1, SELL_LIMIT_OPEN_PRICE_EDIT_Y1,
      0, 0
   );
   sellLimitOpenPriceEdit.Text(DoubleToString(sellLimitEntryPrice, int(symbolDigits)));
   sellLimitOpenPriceEdit.Width(GUI_OBJECTS_WIDTH);
   sellLimitOpenPriceEdit.Height(GUI_OBJECTS_HEIGHT);
   sellLimitOpenPriceEdit.Color(GUI_OBJECTS_SELL_EDIT_COLOR);
   sellLimitOpenPriceEdit.ColorBackground(GUI_OBJECTS_SELL_EDIT_BG_COLOR);
   sellLimitOpenPriceEdit.FontSize(GUI_OBJECTS_FONT_SIZE);
   sellLimitOpenPriceEdit.ColorBorder(GUI_OBJECTS_SELL_BTN_BG_COLOR);
   mainPanelWindow.Add(sellLimitOpenPriceEdit);

//--Create the sell limit sl edit to get the sell limit sl user input
   sellLimitSlEdit.Create(
      0, SELL_LIMIT_SL_EDIT_NAME, SELL_LIMIT_SL_EDIT_SUBWINDOW,
      SELL_LIMIT_SL_EDIT_X1, SELL_LIMIT_SL_EDIT_Y1,
      0, 0
   );
   sellLimitSlEdit.Text(IntegerToString(sl));
   sellLimitSlEdit.Width(int(GUI_OBJECTS_WIDTH / 1.4));
   sellLimitSlEdit.Height(GUI_OBJECTS_HEIGHT);
   sellLimitSlEdit.Color(GUI_OBJECTS_SELL_EDIT_COLOR);
   sellLimitSlEdit.ColorBackground(GUI_OBJECTS_SELL_EDIT_BG_COLOR);
   sellLimitSlEdit.FontSize(GUI_OBJECTS_FONT_SIZE);
   sellLimitSlEdit.ColorBorder(GUI_OBJECTS_SELL_BTN_BG_COLOR);
   mainPanelWindow.Add(sellLimitSlEdit);

//--Create the sell limit tp edit to get the sell limit tp user input
   sellLimitTpEdit.Create(
      0, SELL_LIMIT_TP_EDIT_NAME, SELL_LIMIT_TP_EDIT_SUBWINDOW,
      SELL_LIMIT_TP_EDIT_X1, SELL_LIMIT_TP_EDIT_Y1,
      0, 0
   );
   sellLimitTpEdit.Text(IntegerToString(tp));
   sellLimitTpEdit.Width(GUI_OBJECTS_WIDTH);
   sellLimitTpEdit.Height(GUI_OBJECTS_HEIGHT);
   sellLimitTpEdit.Color(GUI_OBJECTS_SELL_EDIT_COLOR);
   sellLimitTpEdit.ColorBackground(GUI_OBJECTS_SELL_EDIT_BG_COLOR);
   sellLimitTpEdit.FontSize(GUI_OBJECTS_FONT_SIZE);
   sellLimitTpEdit.ColorBorder(GUI_OBJECTS_SELL_BTN_BG_COLOR);
   mainPanelWindow.Add(sellLimitTpEdit);

   /*-------------------------------------------------------------
   * Status Labels and readonly edits GUI components creation ****
   *------------------------------------------------------------*/
//--Create the order magic status label
   magicOrderStatusLabel.Create(
      0, MAGIC_ORDER_STATUS_LABEL_NAME, MAGIC_ORDER_STATUS_LABEL_SUBWINDOW,
      MAGIC_ORDER_STATUS_LABEL_X1,
      MAGIC_ORDER_STATUS_LABEL_Y1,
      GUI_OBJECTS_WIDTH,
      GUI_OBJECTS_HEIGHT
   );
   magicOrderStatusLabel.Text(MAGIC_ORDER_STATUS_LABEL_TEXT + " - (Total Open Orders: " + (string(MagicOrdersTotal(magicNo))) + ")");
   magicOrderStatusLabel.Color(STATUS_EDIT_COLOR);
   magicOrderStatusLabel.FontSize(STATUS_HEADER_FONT_SIZE);
   mainPanelWindow.Add(magicOrderStatusLabel);

//--Create the magic order status edit to display the magic orders status
   magicOrdersStatusEdit.Create(
      0, MAGIC_ORDER_STATUS_EDIT_NAME, MAGIC_ORDER_STATUS_EDIT_SUBWINDOW,
      MAGIC_ORDER_STATUS_EDIT_X1, MAGIC_ORDER_STATUS_EDIT_Y1,
      0, 0
   );
   magicOrdersStatusEdit.ReadOnly(true);
   magicOrdersStatusEdit.Text(magicOrderStatus);
   magicOrdersStatusEdit.Width(STATUS_EDIT_WIDTH);
   magicOrdersStatusEdit.Height(GUI_OBJECTS_HEIGHT);
   magicOrdersStatusEdit.Color(STATUS_EDIT_COLOR);
   magicOrdersStatusEdit.ColorBackground(STATUS_EDIT_BG_COLOR);
   magicOrdersStatusEdit.FontSize(STATUS_EDIT_FONT_SIZE);
   magicOrdersStatusEdit.ColorBorder(STATUS_EDIT_BORDER_COLOR);
   mainPanelWindow.Add(magicOrdersStatusEdit);

//--Create the delete all magic buy stops button
   deleteAllMagicBuyStopsBtn.Create(
      0, DELETE_ALL_MAGIC_BUY_STOPS_BTN_NAME, DELETE_ALL_MAGIC_BUY_STOPS_BTN_SUBWINDOW,
      DELETE_ALL_MAGIC_BUY_STOPS_BTN_X1, DELETE_ALL_MAGIC_BUY_STOPS_BTN_Y1,
      0, 0
   );
   deleteAllMagicBuyStopsBtn.Text(DELETE_ALL_MAGIC_BUY_STOPS_BTN_TEXT);
   deleteAllMagicBuyStopsBtn.Width(DELETE_ORDERS_BTN_WIDTH);
   deleteAllMagicBuyStopsBtn.Height(GUI_OBJECTS_HEIGHT);
   deleteAllMagicBuyStopsBtn.Color(DELETE_ORDERS_BTN_COLOR);
   deleteAllMagicBuyStopsBtn.ColorBackground(DELETE_BUY_ORDERS_BTN_BG_COLOR);
   deleteAllMagicBuyStopsBtn.ColorBorder(DELETE_ORDERS_BTN_BORDER_COLOR);
   deleteAllMagicBuyStopsBtn.FontSize(DELETE_ORDERS_BTN_FONT_SIZE);
   mainPanelWindow.Add(deleteAllMagicBuyStopsBtn);

//--Create the delete all magic sell stops button
   deleteAllMagicSellStopsBtn.Create(
      0, DELETE_ALL_MAGIC_SELL_STOPS_BTN_NAME, DELETE_ALL_MAGIC_SELL_STOPS_BTN_SUBWINDOW,
      DELETE_ALL_MAGIC_SELL_STOPS_BTN_X1, DELETE_ALL_MAGIC_SELL_STOPS_BTN_Y1,
      0, 0
   );
   deleteAllMagicSellStopsBtn.Text(DELETE_ALL_MAGIC_SELL_STOPS_BTN_TEXT);
   deleteAllMagicSellStopsBtn.Width(DELETE_ORDERS_BTN_WIDTH);
   deleteAllMagicSellStopsBtn.Height(GUI_OBJECTS_HEIGHT);
   deleteAllMagicSellStopsBtn.Color(DELETE_ORDERS_BTN_COLOR);
   deleteAllMagicSellStopsBtn.ColorBackground(DELETE_SELL_ORDERS_BTN_BG_COLOR);
   deleteAllMagicSellStopsBtn.ColorBorder(DELETE_ORDERS_BTN_BORDER_COLOR);
   deleteAllMagicSellStopsBtn.FontSize(DELETE_ORDERS_BTN_FONT_SIZE);
   mainPanelWindow.Add(deleteAllMagicSellStopsBtn);

//--Create the delete all magic buy limits button
   deleteAllMagicBuyLimitsBtn.Create(
      0, DELETE_ALL_MAGIC_BUY_LIMITS_BTN_NAME, DELETE_ALL_MAGIC_BUY_LIMITS_BTN_SUBWINDOW,
      DELETE_ALL_MAGIC_BUY_LIMITS_BTN_X1, DELETE_ALL_MAGIC_BUY_LIMITS_BTN_Y1,
      0, 0
   );
   deleteAllMagicBuyLimitsBtn.Text(DELETE_ALL_MAGIC_BUY_LIMITS_BTN_TEXT);
   deleteAllMagicBuyLimitsBtn.Width(DELETE_ORDERS_BTN_WIDTH);
   deleteAllMagicBuyLimitsBtn.Height(GUI_OBJECTS_HEIGHT);
   deleteAllMagicBuyLimitsBtn.Color(DELETE_ORDERS_BTN_COLOR);
   deleteAllMagicBuyLimitsBtn.ColorBackground(DELETE_BUY_ORDERS_BTN_BG_COLOR);
   deleteAllMagicBuyLimitsBtn.ColorBorder(DELETE_ORDERS_BTN_BORDER_COLOR);
   deleteAllMagicBuyLimitsBtn.FontSize(DELETE_ORDERS_BTN_FONT_SIZE);
   mainPanelWindow.Add(deleteAllMagicBuyLimitsBtn);

//--Create the delete all magic sell limits button
   deleteAllMagicSellLimitsBtn.Create(
      0, DELETE_ALL_MAGIC_SELL_LIMITS_BTN_NAME, DELETE_ALL_MAGIC_SELL_LIMITS_BTN_SUBWINDOW,
      DELETE_ALL_MAGIC_SELL_LIMITS_BTN_X1, DELETE_ALL_MAGIC_SELL_LIMITS_BTN_Y1,
      0, 0
   );
   deleteAllMagicSellLimitsBtn.Text(DELETE_ALL_MAGIC_SELL_LIMITS_BTN_TEXT);
   deleteAllMagicSellLimitsBtn.Width(DELETE_ORDERS_BTN_WIDTH);
   deleteAllMagicSellLimitsBtn.Height(GUI_OBJECTS_HEIGHT);
   deleteAllMagicSellLimitsBtn.Color(DELETE_ORDERS_BTN_COLOR);
   deleteAllMagicSellLimitsBtn.ColorBackground(DELETE_SELL_ORDERS_BTN_BG_COLOR);
   deleteAllMagicSellLimitsBtn.ColorBorder(DELETE_ORDERS_BTN_BORDER_COLOR);
   deleteAllMagicSellLimitsBtn.FontSize(DELETE_ORDERS_BTN_FONT_SIZE);
   mainPanelWindow.Add(deleteAllMagicSellLimitsBtn);

//--Create the delete all magic orders button
   deleteAllMagicOrdersBtn.Create(
      0, DELETE_ALL_MAGIC_ORDERS_BTN_NAME, DELETE_ALL_MAGIC_ORDERS_BTN_SUBWINDOW,
      DELETE_ALL_MAGIC_ORDERS_BTN_X1, DELETE_ALL_MAGIC_ORDERS_BTN_Y1,
      0, 0
   );
   deleteAllMagicOrdersBtn.Text(DELETE_ALL_MAGIC_ORDERS_BTN_TEXT);
   deleteAllMagicOrdersBtn.Width(STATUS_EDIT_WIDTH);
   deleteAllMagicOrdersBtn.Height(GUI_OBJECTS_HEIGHT);
   deleteAllMagicOrdersBtn.Color(DELETE_ORDERS_BTN_COLOR);
   deleteAllMagicOrdersBtn.ColorBackground(DELETE_ALL_ORDERS_BTN_BG_COLOR);
   deleteAllMagicOrdersBtn.ColorBorder(DELETE_ORDERS_BTN_BORDER_COLOR);
   deleteAllMagicOrdersBtn.FontSize(DELETE_ORDERS_BTN_FONT_SIZE);
   mainPanelWindow.Add(deleteAllMagicOrdersBtn);

//--Call the Run() method to load the main panel window
   mainPanelWindow.Run();
  }
```

We now need to detect any button presses or activations by populating the _OnChartEvent(...)_ function. This function will serve as the event handler for all user interactions, making it the core component of our panel's responsiveness. In this section, we will call the imported prototype functions from the _PendingOrdersManager.ex5_ library, allowing us to manage and manipulate pending orders based on user input. For example, if a user clicks a button to place a new pending order, the corresponding library function will be triggered to execute the operation.

To further enhance the user experience with the Pending Orders Panel, we will also integrate auditory feedback. Different sounds will be played to signal the outcome of an action, such as whether an order was successfully placed or if it failed due to some error. This provides real-time feedback and adds an additional layer of interactivity, making the panel more intuitive and user-friendly. Combining visual and auditory cues will ensure that users can easily follow the status of their actions and respond accordingly.

```
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//--Detect any clicks or events performed to the orders panel window and make it moveable
   mainPanelWindow.ChartEvent(id, lparam, dparam, sparam);

//--Detect any click events on the chart
   if(id == CHARTEVENT_OBJECT_CLICK)
     {
      //--Detect when the buyStopBtn is clicked and open a new buy stop order
      if(sparam == buyStopBtn.Name())
        {
         Print(__FUNCTION__, " CHARTEVEN_OBJECT_CLICK: ", sparam);
         Print("Opening a new Buy Stop Order with the details below: ");
         Print("Volume: ", buyStopVolumeLotEdit.Text());
         Print("Open Price: ", buyStopOpenPriceEdit.Text());
         Print("Sl (Pips): ", buyStopSlEdit.Text());
         Print("Tp (Pips): ", buyStopTpEdit.Text());
         if(
            OpenBuyStop(
               magicNo, _Symbol, StringToDouble(buyStopOpenPriceEdit.Text()),
               StringToDouble(buyStopVolumeLotEdit.Text()), (uint)StringToInteger(buyStopSlEdit.Text()),
               (uint)StringToInteger(buyStopTpEdit.Text()), "EX5 PendingOrdersManager Panel"
            )
         )
           {
            PlaySound("ok.wav");//-- Order placed ok
           }
         else
           {
            PlaySound("alert2.wav");//-- Order failed
           }
        }

      //--Detect when the sellStopBtn is clicked and open a new sell stop order
      if(sparam == sellStopBtn.Name())
        {
         Print(__FUNCTION__, " CHARTEVEN_OBJECT_CLICK: ", sparam);
         Print("Opening a new Sell Stop Order with the details below: ");
         Print("Volume: ", sellStopVolumeLotEdit.Text());
         Print("Open Price: ", sellStopOpenPriceEdit.Text());
         Print("Sl (Pips): ", sellStopSlEdit.Text());
         Print("Tp (Pips): ", sellStopTpEdit.Text());
         if(
            OpenSellStop(
               magicNo, _Symbol, StringToDouble(sellStopOpenPriceEdit.Text()),
               StringToDouble(sellStopVolumeLotEdit.Text()), (uint)StringToInteger(sellStopSlEdit.Text()),
               (uint)StringToInteger(sellStopTpEdit.Text()), "EX5 PendingOrdersManager Panel"
            )
         )
           {
            PlaySound("ok.wav");//-- Order placed ok
           }
         else
           {
            PlaySound("alert2.wav");//-- Order failed
           }
        }

      //--Detect when the buyLimitBtn is clicked and open a new buy limit order
      if(sparam == buyLimitBtn.Name())
        {
         Print(__FUNCTION__, " CHARTEVEN_OBJECT_CLICK: ", sparam);
         Print("Opening a new Buy Limit Order with the details below: ");
         Print("Volume: ", buyLimitVolumeLotEdit.Text());
         Print("Open Price: ", buyLimitOpenPriceEdit.Text());
         Print("Sl (Pips): ", buyLimitSlEdit.Text());
         Print("Tp (Pips): ", buyLimitTpEdit.Text());
         if(
            OpenBuyLimit(
               magicNo, _Symbol, StringToDouble(buyLimitOpenPriceEdit.Text()),
               StringToDouble(buyLimitVolumeLotEdit.Text()), (uint)StringToInteger(buyLimitSlEdit.Text()),
               (uint)StringToInteger(buyLimitTpEdit.Text()), "EX5 PendingOrdersManager Panel"
            )
         )
           {
            PlaySound("ok.wav");//-- Order placed ok
           }
         else
           {
            PlaySound("alert2.wav");//-- Order failed
           }
        }

      //--Detect when the sellLimitBtn is clicked and open a new sell limit order
      if(sparam == sellLimitBtn.Name())
        {
         Print(__FUNCTION__, " CHARTEVEN_OBJECT_CLICK: ", sparam);
         Print("Opening a new Sell Limit Order with the details below: ");
         Print("Volume: ", sellLimitVolumeLotEdit.Text());
         Print("Open Price: ", sellLimitOpenPriceEdit.Text());
         Print("Sl (Pips): ", sellLimitSlEdit.Text());
         Print("Tp (Pips): ", sellLimitTpEdit.Text());

         if(
            OpenSellLimit(
               magicNo, _Symbol, StringToDouble(sellLimitOpenPriceEdit.Text()),
               StringToDouble(sellLimitVolumeLotEdit.Text()), (uint)StringToInteger(sellLimitSlEdit.Text()),
               (uint)StringToInteger(sellLimitTpEdit.Text()), "EX5 PendingOrdersManager Panel"
            )
         )
           {
            PlaySound("ok.wav");//-- Order placed ok
           }
         else
           {
            PlaySound("alert2.wav");//-- Order failed
           }
        }

      //--Detect when the deleteAllMagicBuyStopsBtn is clicked and delete all the specified orders
      if(sparam == deleteAllMagicBuyStopsBtn.Name() && MagicBuyStopOrdersTotal(magicNo) > 0)
        {
         Print(__FUNCTION__, " CHARTEVEN_OBJECT_CLICK: ", sparam);
         Print("Deleting all the buy stop orders with magic number: ", magicNo);
         if(DeleteAllBuyStops("", magicNo))
           {
            PlaySound("ok.wav");//-- Orders deleted ok
           }
         else
           {
            PlaySound("alert2.wav");//-- Order deleting failed
           }
        }

      //--Detect when the deleteAllMagicSellStopsBtn is clicked and delete all the specified orders
      if(sparam == deleteAllMagicSellStopsBtn.Name() && MagicSellStopOrdersTotal(magicNo) > 0)
        {
         Print(__FUNCTION__, " CHARTEVEN_OBJECT_CLICK: ", sparam);
         Print("Deleting all the sell stop orders with magic number: ", magicNo);
         if(DeleteAllSellStops("", magicNo))
           {
            PlaySound("ok.wav");//-- Orders deleted ok
           }
         else
           {
            PlaySound("alert2.wav");//-- Order deleting failed
           }
        }

      //--Detect when the deleteAllMagicBuyLimitsBtn is clicked and delete all the specified orders
      if(sparam == deleteAllMagicBuyLimitsBtn.Name() && MagicBuyLimitOrdersTotal(magicNo) > 0)
        {
         Print(__FUNCTION__, " CHARTEVEN_OBJECT_CLICK: ", sparam);
         Print("Deleting all the buy limit orders with magic number: ", magicNo);
         if(DeleteAllBuyLimits("", magicNo))
           {
            PlaySound("ok.wav");//-- Orders deleted ok
           }
         else
           {
            PlaySound("alert2.wav");//-- Order deleting failed
           }
        }

      //--Detect when the deleteAllMagicSellLimitsBtn is clicked and delete all the specified orders
      if(sparam == deleteAllMagicSellLimitsBtn.Name() && MagicSellLimitOrdersTotal(magicNo) > 0)
        {
         Print(__FUNCTION__, " CHARTEVEN_OBJECT_CLICK: ", sparam);
         Print("Deleting all the sell limit orders with magic number: ", magicNo);
         if(DeleteAllSellLimits("", magicNo))
           {
            PlaySound("ok.wav");//-- Orders deleted ok
           }
         else
           {
            PlaySound("alert2.wav");//-- Order deleting failed
           }
        }

      //--Detect when the deleteAllMagicOrdersBtn is clicked and delete all the specified orders
      if(sparam == deleteAllMagicOrdersBtn.Name() && MagicOrdersTotal(magicNo) > 0)
        {
         Print(__FUNCTION__, " CHARTEVEN_OBJECT_CLICK: ", sparam);
         Print("Deleting all the open peding orders with magic number: ", magicNo);
         if(DeleteAllMagicOrders(magicNo))
           {
            PlaySound("ok.wav");//-- Orders deleted ok
           }
         else
           {
            PlaySound("alert2.wav");//-- Order deleting failed
           }
        }
     }
  }
```

You'll notice that as soon as the Expert Advisor is loaded, it automatically populates the order _volume/lot, entry prices, stop loss (SL),_ and _take profit (TP)_ fields with predefined values based on the current spread. These values give the user a starting point for order placement, streamlining the trading process and minimizing manual input. Additionally, the order delete buttons, located below the magic order statuses, are initially grayed out and disabled if no pending or active orders matching the Expert Advisor’s assigned magic number are detected. Once the Expert Advisor identifies matching orders, the buttons will activate and change color, signaling that they are now operational and ready to execute delete commands.

![Orders Panel Disabled Order Delete Buttons](https://c.mql5.com/2/96/Orders_Panel_Disabled_order_delete_buttons.png)

When the Expert Advisor detects that orders matching the assigned magic number are opened, the order delete buttons are activated, changing color to indicate that they are enabled and can be used.

![Orders Panel Two Enabled Order Delete Buttons](https://c.mql5.com/2/96/Orders_Panel_Two_Enabled_order_delete_buttons.png)

![Orders Panel All Enabled Order Delete Buttons](https://c.mql5.com/2/96/Orders_Panel_All_Enabled_order_delete_buttons.png)

To implement this feature, we need to add the following code in the _OnTick()_ function. This will allow us to use the _imported EX5 library functions_ to continuously monitor and update the graphical user interface in real time, ensuring that it stays in sync with the status of the orders on every incoming tick.

```
void OnTick()
  {
//---
   magicOrderStatus = " Buy Stops: " + (string(MagicBuyStopOrdersTotal(magicNo))) +
                      ", Sell Stops: " + (string(MagicSellStopOrdersTotal(magicNo))) +
                      ", Buy Limits: " + (string(MagicBuyLimitOrdersTotal(magicNo))) +
                      ", Sell Limits: " + (string(MagicSellLimitOrdersTotal(magicNo))) +
                      " ";
   magicOrderStatusLabel.Text(MAGIC_ORDER_STATUS_LABEL_TEXT + (string(MagicOrdersTotal(magicNo))));
   magicOrdersStatusEdit.Text(magicOrderStatus);

//-- Disable and change the background color of the deleteAllMagicBuyStopsBtn depending on the open orders status
   if(MagicBuyStopOrdersTotal(magicNo) == 0)
     {
      deleteAllMagicBuyStopsBtn.Disable();
      deleteAllMagicBuyStopsBtn.ColorBackground(clrLightSlateGray);
     }
   else
     {
      deleteAllMagicBuyStopsBtn.Enable();
      deleteAllMagicBuyStopsBtn.ColorBackground(DELETE_BUY_ORDERS_BTN_BG_COLOR);
     }

//-- Disable and change the background color of the deleteAllMagicSellStopsBtn depending on the open orders status
   if(MagicSellStopOrdersTotal(magicNo) == 0)
     {
      deleteAllMagicSellStopsBtn.Disable();
      deleteAllMagicSellStopsBtn.ColorBackground(clrLightSlateGray);
     }
   else
     {
      deleteAllMagicSellStopsBtn.Enable();
      deleteAllMagicSellStopsBtn.ColorBackground(DELETE_SELL_ORDERS_BTN_BG_COLOR);
     }

//-- Disable and change the background color of the deleteAllMagicBuyLimitsBtn depending on the open orders status
   if(MagicBuyLimitOrdersTotal(magicNo) == 0)
     {
      deleteAllMagicBuyLimitsBtn.Disable();
      deleteAllMagicBuyLimitsBtn.ColorBackground(clrLightSlateGray);
     }
   else
     {
      deleteAllMagicBuyLimitsBtn.Enable();
      deleteAllMagicBuyLimitsBtn.ColorBackground(DELETE_BUY_ORDERS_BTN_BG_COLOR);
     }

//-- Disable and change the background color of the deleteAllMagicSellLimitsBtn depending on the open orders status
   if(MagicSellLimitOrdersTotal(magicNo) == 0)
     {
      deleteAllMagicSellLimitsBtn.Disable();
      deleteAllMagicSellLimitsBtn.ColorBackground(clrLightSlateGray);
     }
   else
     {
      deleteAllMagicSellLimitsBtn.Enable();
      deleteAllMagicSellLimitsBtn.ColorBackground(DELETE_SELL_ORDERS_BTN_BG_COLOR);
     }

//-- Disable and change the background color of the deleteAllMagicOrdersBtn depending on the open orders status
   if(MagicOrdersTotal(magicNo) == 0)
     {
      deleteAllMagicOrdersBtn.Disable();
      deleteAllMagicOrdersBtn.ColorBackground(clrLightSlateGray);
     }
   else
     {
      deleteAllMagicOrdersBtn.Enable();
      deleteAllMagicOrdersBtn.ColorBackground(DELETE_ALL_ORDERS_BTN_BG_COLOR);
     }
  }
```

Finally, we must ensure that all resources are properly released and cleaned up when the Expert Advisor terminates. To do this, add the following code to the _OnDeinit(...)_ function.

```
void OnDeinit(const int reason)
  {
//---
   //-- Delete and garbage collect the graphical user interface and other graphical objects
   mainPanelWindow.Destroy();

//-- Clear any chart comments
   Comment("");
  }
```

The _PendingOrdersPanel.mq5_ file is attached at the end of this article.

### Conclusion

We've built an all-round _Pending Orders Management EX5_ library that shows how you can _open, modify, delete, sort,_ and _filter_ different types of pending orders. This library is a useful resource for any MQL5 developer who needs a flexible and easy-to-use tool for managing pending orders, allowing them to quickly get the status or take action on orders with just a simple function call.

The library is feature-packed and comes with clear documentation and real-world examples. In the next article, we'll apply a similar method to create a History Management EX5 library, which will make handling deals and order history in MQL5 much easier.

Thanks for following along, and all the best in your trading and MQL5 programming!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15888.zip "Download all attachments in the single ZIP archive")

[PositionsManager.ex5](https://www.mql5.com/en/articles/download/15888/positionsmanager.ex5 "Download PositionsManager.ex5")(45.72 KB)

[PendingOrdersManager.ex5](https://www.mql5.com/en/articles/download/15888/pendingordersmanager.ex5 "Download PendingOrdersManager.ex5")(42.52 KB)

[PendingOrdersManager.mq5](https://www.mql5.com/en/articles/download/15888/pendingordersmanager.mq5 "Download PendingOrdersManager.mq5")(84.49 KB)

[PendingOrdersManager\_Imports\_Template.mq5](https://www.mql5.com/en/articles/download/15888/pendingordersmanager_imports_template.mq5 "Download PendingOrdersManager_Imports_Template.mq5")(4.79 KB)

[PendingOrdersPanel.mq5](https://www.mql5.com/en/articles/download/15888/pendingorderspanel.mq5 "Download PendingOrdersPanel.mq5")(50.92 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Trading Toolkit (Part 8): How to Implement and Use the History Manager EX5 Library in Your Codebase](https://www.mql5.com/en/articles/17015)
- [MQL5 Trading Toolkit (Part 7): Expanding the History Management EX5 Library with the Last Canceled Pending Order Functions](https://www.mql5.com/en/articles/16906)
- [MQL5 Trading Toolkit (Part 6): Expanding the History Management EX5 Library with the Last Filled Pending Order Functions](https://www.mql5.com/en/articles/16742)
- [MQL5 Trading Toolkit (Part 5): Expanding the History Management EX5 Library with Position Functions](https://www.mql5.com/en/articles/16681)
- [MQL5 Trading Toolkit (Part 4): Developing a History Management EX5 Library](https://www.mql5.com/en/articles/16528)
- [MQL5 Trading Toolkit (Part 2): Expanding and Implementing the Positions Management EX5 Library](https://www.mql5.com/en/articles/15224)

**[Go to discussion](https://www.mql5.com/en/forum/474866)**

![Developing a Replay System (Part 48): Understanding the concept of a service](https://c.mql5.com/2/76/Desenvolvendo_um_sistema_de_Replay_9Parte_480___LOGO.png)[Developing a Replay System (Part 48): Understanding the concept of a service](https://www.mql5.com/en/articles/11781)

How about learning something new? In this article, you will learn how to convert scripts into services and why it is useful to do so.

![Visualizing deals on a chart (Part 1): Selecting a period for analysis](https://c.mql5.com/2/79/Visualization_of_trades_on_a_chart_Part_1_____LOGO.png)[Visualizing deals on a chart (Part 1): Selecting a period for analysis](https://www.mql5.com/en/articles/14903)

Here we are going to develop a script from scratch that simplifies unloading print screens of deals for analyzing trading entries. All the necessary information on a single deal is to be conveniently displayed on one chart with the ability to draw different timeframes.

![Creating a Trading Administrator Panel in MQL5 (Part IV): Login Security Layer](https://c.mql5.com/2/98/Creating_a_Trading_Administrator_Panel_in_MQL5_Part_IV__Logo.png)[Creating a Trading Administrator Panel in MQL5 (Part IV): Login Security Layer](https://www.mql5.com/en/articles/16079)

Imagine a malicious actor infiltrating the Trading Administrator room, gaining access to the computers and the Admin Panel used to communicate valuable insights to millions of traders worldwide. Such an intrusion could lead to disastrous consequences, such as the unauthorized sending of misleading messages or random clicks on buttons that trigger unintended actions. In this discussion, we will explore the security measures in MQL5 and the new security features we have implemented in our Admin Panel to safeguard against these threats. By enhancing our security protocols, we aim to protect our communication channels and maintain the trust of our global trading community. Find more insights in this article discussion.

![Neural Network in Practice: Least Squares](https://c.mql5.com/2/76/Rede_neural_na_protica_Manimos_Quadrados___LOGO.png)[Neural Network in Practice: Least Squares](https://www.mql5.com/en/articles/13670)

In this article, we'll look at a few ideas, including how mathematical formulas are more complex in appearance than when implemented in code. In addition, we will consider how to set up a chart quadrant, as well as one interesting problem that may arise in your MQL5 code. Although, to be honest, I still don't quite understand how to explain it. Anyway, I'll show you how to fix it in code.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/15888&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062615760464946583)

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