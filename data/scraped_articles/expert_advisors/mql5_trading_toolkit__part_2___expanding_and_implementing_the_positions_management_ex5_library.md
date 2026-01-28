---
title: MQL5 Trading Toolkit (Part 2): Expanding and Implementing the Positions Management EX5 Library
url: https://www.mql5.com/en/articles/15224
categories: Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:28:43.554305
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=wtwgqpcxcurigyxbsljhkpubmjnribua&ssn=1769178521550047116&ssn_dr=0&ssn_sr=0&fv_date=1769178521&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15224&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Trading%20Toolkit%20(Part%202)%3A%20Expanding%20and%20Implementing%20the%20Positions%20Management%20EX5%20Library%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917852124316282&fz_uniq=5068284000604190616&sv=2552)

MetaTrader 5 / Examples


### Introduction

In the [first article](https://www.mql5.com/en/articles/14822), we analyzed MQL5 code libraries in detail. We covered different library types, their benefits, creating EX5 libraries, and the components of an EX5 library source code file (. _mq5_). This gave you a strong foundation in EX5 libraries and their creation process. We then created a practical example of an EX5 Positions Management library, demonstrating how to code exportable functions with MQL5.

In this article, we will continue building on that foundation. We will expand the Positions Management EX5 library and create two basic Expert Advisors. One of these example Expert Advisors will utilize a graphical trade and information panel, demonstrating how to import and implement the Positions Management EX5 library in practice. This will serve as a real-world example of how to create and integrate an EX5 library into your MQL5 code. To begin, let us first break down the process of importing and using an already developed and compiled .ex5 binary library.

### How to Import and Implement an EX5 Library

To import and use an .EX5 library in your MQL5 code _(Expert Advisors, Custom Indicators, Scripts, or Services)_, you need to insert the _#import_ directives just below the _#property_ directives in the head or top section of your source code file. To include the binary compiled library, start by specifying the _#import_ directive followed by the file path where the library is stored. By default, MQL5 searches for libraries in two locations to save you time referencing them directly in your code. The first location is the " _MQL5/Libraries_" folder, which is the default predefined location for storing libraries. If the library isn't found there, MQL5 will then search the folder where your MQL program itself is located. If your EX5 library is stored directly in the "_MQL5/Libraries/_" folder or in the same folder as your source code, just specify the library name enclosed in double quotes after the #import directive without specifying the folder path.

After specifying the library folder path, provide the library name followed by the _.ex5_ extension. On the next new line, add the definitions or descriptions of all the exported function prototypes to be imported into your code. Finally, end the import code segment with another _#import_ directive to close it.

```
#import "FilePath/LibraryName.ex5" //-- Opening .EX5 Library import directive

   //-- Function definitions/descriptions prototypes
   int  FunctionPrototype1();
   void FunctionPrototype2(bool y);
   bool FunctionPrototype3(double x);

#import //--- Closing .EX5 Library import directive
```

It is required to specify and provide the _.ex5_ extension after the library name when declaring the import directive. Omitting the extension will indicate that you are importing a _.DLL_ library by default.

You can also import and implement multiple _.ex5 libraries_ in a single MQL5 file. The code structure for importing multiple _.ex5 libraries_ is similar to importing a single library, with the only difference being the placement of the closing _#import_ directives. For multiple library imports, the closing _#import_ directive of the first library should be followed by the name of the next _.ex5 library_ being imported. This will close the first import directive while initiating the next import directive, and so on. When closing the final import directive of the last library, make sure that it ends without a library name.

```
#import "FilePath/LibraryName.ex5"  //-- Opening .EX5 Library import directive

   //-- Function definitions/descriptions prototypes for the first library here
   int  FunctionPrototype1();
   void FunctionPrototype2(bool y);

#import "FilePath/SecondLibraryName.ex5"
   //-- Function definitions/descriptions prototypes for the second library here
   bool  FunctionPrototype();
   string FunctionPrototype2(bool z);

#import //--- Closing .EX5 Library import directive
```

When working with multiple libraries in MQL5, it is required to give each a unique name. It does not matter if all of these libraries are stored in different folders, having distinct names is a requirement so as not to encounter any errors.

Each library creates its own isolated environment or "namespace". This means that functions within a library are associated with that library's name. You can freely name functions within a library without worrying about conflicts, even if they match built-in function names. However, it's generally recommended to avoid such naming for clarity.

If you happen to have functions with the same name in different libraries, the system will prioritize the function based on specific rules. This prevents confusion when calling functions with identical names. Once you have successfully imported the library's function prototypes, you can seamlessly integrate them into your code and treat them just like any other local function you've defined yourself.

Further down the article, I provide a detailed explanation of how to incorporate and utilize EX5 libraries in a practical setting. You will find two in-depth demonstrations: one where we code a VIDyA trading strategy-based Expert Advisor, and another utilizing a graphical user interface (GUI). These Expert Advisors will integrate and leverage our custom-built Positions Management EX5 library. These hands-on examples will offer valuable insights into implementing EX5 libraries in real-world Expert Advisors.

### Common EX5 Library Implementation Runtime Errors

Debugging EX5 libraries can be challenging, as most common errors related to the imported prototype functions occur during runtime when loading the final compiled MQL5 app in the trading terminal. These errors typically arise from coding the header import library directives section with incorrect values like the library file paths or names, function prototype descriptions, types, names, full list of parameters, and return values during the import declarations. The compiler is not expected or tasked with detecting these import declaration errors during compile time because it cannot access the source code of the imported library, as it is encapsulated and already compiled into a binary executable format (. _ex5_) module.

Any source code file that contains these errors will successfully compile, but when you try to load the compiled MQL5 app in the trading terminal, it will fail and generate runtime errors. These errors are displayed in either the _Experts tab_ or the _Journal tab_ of the MetaTrader5 Terminal. Here are the most common errors you might encounter:

> Unresolved Import Function Call: _(cannot find 'Function\_Name' in 'Library\_Name.ex5')_

- **Description**: This runtime error occurs when trying to load the MQL5 app in a MetaTrader5 chart and is displayed in the _Experts_ tab. It is caused by incorrect function prototype definitions or descriptions, such as type, name, or parameters, provided in the import directive section of the library.
- **Resolution**: Ensure the import code segment of the library is properly coded with the correct function prototype definitions as required and recompile your code.

> > ![Unresolved import function call EX5 error](https://c.mql5.com/2/83/ex5_library_unresolved_import_function_call_error.png)
>
> Cannot Open File 'Library\_Name.ex5': _(loading of ExpertAdvisor\_Name (GBPJPY,M15) failed \[0\])_

- **Description**: This runtime error occurs when trying to load the MQL5 app in a MetaTrader5 chart and is displayed in the _Journal_ tab. It is caused when the imported EX5 library file cannot be located and loaded.
- **Resolution**: Ensure the correct file path to the library is specified in the import code segment of the library, and recompile your code.

> > ![can not open EX5 library file error](https://c.mql5.com/2/83/can_not_open_ex5_library_file_error.png)

While other errors may arise when working with imported libraries in MQL5, the above runtime errors are the most common and troublesome for beginner developers. These errors are particularly challenging because they are easily overlooked and not designed to be detected by the compiler during compilation.

### How to Update and Redeploy EX5 Libraries

It is important to follow the correct sequence when redeploying your EX5 libraries after every update to ensure that the new changes are properly integrated into the MQL5 projects using the library. The compilation sequence is the most crucial step in updating and redeploying libraries in MQL5. To ensure all the new changes and updates are utilized in all projects importing the library, follow these steps:

1. **Compile the New EX5 File**: Begin by compiling the updated .mq5 library source code file to create the new .ex5 executable binary file.
2. **Update Imported Function Prototypes**: In all MQL5 projects that use the EX5 library, update any function prototype import definitions if they have changed in the new .ex5 library update.
3. **Compile the Projects**: Recompile all the MQL5 projects that implement the EX5 library.

By following this sequence, you will ensure that all the updates in the EX5 library are reflected and integrated into all the projects that import the library.

### Trailing Stop Loss Function

Before we can implement our positions' management library, let us expand it further by adding some vital functions. We will begin by adding a trailing stop loss management module or function, as our library will not be complete without this essential feature. A trailing stop loss is an important part of any trading strategy, as when properly implemented, it has the potential to increase a system's profit margins and overall success rate.

The trailing stop loss function will be called _SetTrailingStopLoss()_ and will be responsible for setting the _trailing SL_ of an existing open position using the position's ticket as a filtering mechanism. It will take a position's ticket number and the desired trailing stop loss in pips (points) as arguments or parameters and attempt to update the position's stop loss on the trading server when certain conditions are met. This function should be called continuously on every tick to modify the trailing stop loss in real-time, as the status of the targeted positions is constantly changing.

It will first check if trading is allowed and if the provided trailing stop loss in pips (points) is valid. It will then select the position, retrieve and save all the necessary symbol information, and calculate the trailing stop loss price. If the calculated price is valid, it will send an order to the trade server to set the stop loss. If the order is successfully executed, the function will return _true_; otherwise, it will return _false_.

First, we will begin by creating the function definition. Our trailing stop loss function will be of type _bool_ and will take two parameters:

1. **ulong positionTicket**: This is a unique identifier for the position we will modify.
2. **int trailingStopLoss**: This is the desired Stop Loss level in pips (points) from the position's current price.

The _export_ keyword indicates that this library function can be called from any MQL5 source code file or project that imports it.

```
bool SetTrailingStopLoss(ulong positionTicket, int trailingStopLoss) export
  {
    //-- place the function body here
  }
```

We need to check if trading is allowed and if the _trailingStopLoss_ parameter is greater than _zero_. If either condition is not met, we exit the function and return _false_ to terminate the operation.

```
if(!TradingIsAllowed() || trailingStopLoss == 0)
     {
      return(false); //--- algo trading is disabled or trailing stop loss is invalid, exit function
     }
```

Next, we confirm and select the position using the provided _positionTicket_. If the position selection fails, we print an error message and exit the function.

```
//--- Confirm and select the position using the provided positionTicket
   ResetLastError(); //--- Reset error cache incase of ticket selection errors
   if(!PositionSelectByTicket(positionTicket))
     {
      //---Position selection failed
      Print("\r\n_______________________________________________________________________________________");
      Print(__FUNCTION__, ": Selecting position with ticket:", positionTicket, " failed. ERROR: ", GetLastError());
      return(false); //-- Exit the function
     }
```

We then create some variables to help us store and validate the trailing stop loss. We begin by creating the _slPrice_ varialbe to store the calculated trailing stop loss price and then save the position properties such as symbol, entry price, volume, current stop loss price, current take profit price, and position type.

```
//-- create variable to store the calculated trailing sl prices to send to the trade server
   double slPrice = 0.0;

//--- Position ticket selected, save the position properties
   string positionSymbol = PositionGetString(POSITION_SYMBOL);
   double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   double volume = PositionGetDouble(POSITION_VOLUME);
   double currentPositionSlPrice = PositionGetDouble(POSITION_SL);
   double currentPositionTpPrice = PositionGetDouble(POSITION_TP);
   ENUM_POSITION_TYPE positionType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
```

Likewise, we continue to save the different symbol properties associated with the selected position. These properties will also be used later to validate and calculate the trailing stop loss.

```
//-- Get some information about the positions symbol
   int symbolDigits = (int)SymbolInfoInteger(positionSymbol, SYMBOL_DIGITS); //-- Number of symbol decimal places
   int symbolStopLevel = (int)SymbolInfoInteger(positionSymbol, SYMBOL_TRADE_STOPS_LEVEL);
   double symbolPoint = SymbolInfoDouble(positionSymbol, SYMBOL_POINT);
   double positionPriceCurrent = PositionGetDouble(POSITION_PRICE_CURRENT);
   int spread = (int)SymbolInfoInteger(positionSymbol, SYMBOL_SPREAD);
```

We check if the trailing stop loss value provided is less than the symbol's trade _stop level_. If it is, we adjust the trailing stop loss to be equal to the symbol's stop level.

```
//-- Check if the trailing stop loss is less than the symbol trade stop levels
   if(trailingStopLoss < symbolStopLevel)
     {
      //-- Trailing stop loss is less than the allowed level for the current symbol
      trailingStopLoss = symbolStopLevel; //-- Set it to the symbol stop level by default
     }
```

The next step involves calculating the trailing stop loss price based on whether the position is a buy or sell. For buy positions, the stop loss price is set below the current price, while for sell positions, it is set above the current price. We also validate that the calculated stop loss price is within valid bounds.

```
//-- Calculate and store the trailing stop loss price
   if(positionType == POSITION_TYPE_BUY)
     {
      slPrice = positionPriceCurrent - (trailingStopLoss * symbolPoint);

      //-- Check if the proposed slPrice for the trailing stop loss is valid
      if(slPrice < entryPrice || slPrice < currentPositionSlPrice)
        {
         return(false); //-- Exit the function, proposed trailing stop loss price is invalid
        }
     }
   else  //-- SELL POSITION
     {
      slPrice = positionPriceCurrent + (trailingStopLoss * symbolPoint);

      //-- Check if the proposed slPrice for the trailing stop loss is valid
      if(slPrice > entryPrice || slPrice > currentPositionSlPrice)
        {
         return(false); //-- Exit the function, proposed trailing stop loss price is invalid
        }
     }
```

Before setting the trailing stop loss, let us print the details of the position to the MetaTrader5's log. This includes the symbol, position type, volume, entry price, current stop loss and take profit prices, and other relevant information.

```
//-- Print position properties before setting the trailing stop loss
   string positionProperties = "--> "  + positionSymbol + " " + EnumToString(positionType) + " Trailing Stop Loss Modification Details" +
                               " <--\r\n";
   positionProperties += "------------------------------------------------------------\r\n";
   positionProperties += "Ticket: " + (string)positionTicket + "\r\n";
   positionProperties += "Volume: " + StringFormat("%G", volume) + "\r\n";
   positionProperties += "Price Open: " + StringFormat("%G", entryPrice) + "\r\n";
   positionProperties += "Current SL: " + StringFormat("%G", currentPositionSlPrice) + "   -> New Trailing SL: " + (string)slPrice + "\r\n";
   positionProperties += "Current TP: " + StringFormat("%G", currentPositionTpPrice) + "\r\n";
   positionProperties += "Comment: " + PositionGetString(POSITION_COMMENT) + "\r\n";
   positionProperties += "Magic Number: " + (string)PositionGetInteger(POSITION_MAGIC) + "\r\n";
   positionProperties += "---";
   Print(positionProperties);
```

We reset the _tradeRequest_ and _tradeResult_ structures to _zero_. Then, we initialize the parameters required to set the stop loss and take profit.

```
//-- reset the the tradeRequest and tradeResult values by zeroing them
   ZeroMemory(tradeRequest);
   ZeroMemory(tradeResult);

//-- initialize the parameters to set the sltp
   tradeRequest.action = TRADE_ACTION_SLTP; //-- Trade operation type for setting sl and tp
   tradeRequest.position = positionTicket;
   tradeRequest.symbol = positionSymbol;
   tradeRequest.sl = slPrice;
   tradeRequest.tp = currentPositionTpPrice;
```

Finally, we reset the error cache and send the order to the trade server until it is successful or up to 101 retries. If the order is successfully executed, we print a success message, return _true_, and exit the function. If the order request fails, we handle the error, return _false_, and exit the function.

```
ResetLastError(); //--- reset error cache so that we get an accurate runtime error code in the ErrorAdvisor function

   for(int loop = 0; loop <= 100; loop++) //-- try modifying the sl and tp 101 times untill the request is successful
     {
      //--- send order to the trade server
      if(OrderSend(tradeRequest, tradeResult))
        {
         //-- Confirm order execution
         if(tradeResult.retcode == 10008 || tradeResult.retcode == 10009)
           {
            PrintFormat("Successfully set the Trailing SL for #%I64d %s %s", positionTicket, positionSymbol, EnumToString(positionType));
            PrintFormat("retcode=%u  runtime_code=%u", tradeResult.retcode, GetLastError());
            Print("_______________________________________________________________________________________\r\n\r\n");
            return(true); //-- exit function
            break; //--- success - order placed ok. exit for loop
           }
        }
      else  //-- Order request failed
        {
         //-- order not sent or critical error found
         if(!ErrorAdvisor(__FUNCTION__, positionSymbol, tradeResult.retcode) || IsStopped())
           {
            PrintFormat("ERROR setting the Trailing SL for #%I64d %s %s", positionTicket, positionSymbol, EnumToString(positionType));
            Print("_______________________________________________________________________________________\r\n\r\n");
            return(false); //-- exit function
            break; //-- exit for loop
           }
        }
     }
   return(false);
  }
```

Make sure that all the _SetTrailingStopLoss()_ function code segments are complete in the sequence below:

```
bool SetTrailingStopLoss(ulong positionTicket, int trailingStopLoss) export
  {
//-- first check if the EA is allowed to trade and the trailing stop loss parameter is more than zero
   if(!TradingIsAllowed() || trailingStopLoss == 0)
     {
      return(false); //--- algo trading is disabled or trailing stop loss is invalid, exit function
     }

//--- Confirm and select the position using the provided positionTicket
   ResetLastError(); //--- Reset error cache incase of ticket selection errors
   if(!PositionSelectByTicket(positionTicket))
     {
      //---Position selection failed
      Print("\r\n_______________________________________________________________________________________");
      Print(__FUNCTION__, ": Selecting position with ticket:", positionTicket, " failed. ERROR: ", GetLastError());
      return(false); //-- Exit the function
     }

//-- create variable to store the calculated trailing sl prices to send to the trade server
   double slPrice = 0.0;

//--- Position ticket selected, save the position properties
   string positionSymbol = PositionGetString(POSITION_SYMBOL);
   double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   double volume = PositionGetDouble(POSITION_VOLUME);
   double currentPositionSlPrice = PositionGetDouble(POSITION_SL);
   double currentPositionTpPrice = PositionGetDouble(POSITION_TP);
   ENUM_POSITION_TYPE positionType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

//-- Get some information about the positions symbol
   int symbolDigits = (int)SymbolInfoInteger(positionSymbol, SYMBOL_DIGITS); //-- Number of symbol decimal places
   int symbolStopLevel = (int)SymbolInfoInteger(positionSymbol, SYMBOL_TRADE_STOPS_LEVEL);
   double symbolPoint = SymbolInfoDouble(positionSymbol, SYMBOL_POINT);
   double positionPriceCurrent = PositionGetDouble(POSITION_PRICE_CURRENT);
   int spread = (int)SymbolInfoInteger(positionSymbol, SYMBOL_SPREAD);

//-- Check if the trailing stop loss is less than the symbol trade stop levels
   if(trailingStopLoss < symbolStopLevel)
     {
      //-- Trailing stop loss is less than the allowed level for the current symbol
      trailingStopLoss = symbolStopLevel; //-- Set it to the symbol stop level by default
     }

//-- Calculate and store the trailing stop loss price
   if(positionType == POSITION_TYPE_BUY)
     {
      slPrice = positionPriceCurrent - (trailingStopLoss * symbolPoint);

      //-- Check if the proposed slPrice for the trailing stop loss is valid
      if(slPrice < entryPrice || slPrice < currentPositionSlPrice)
        {
         return(false); //-- Exit the function, proposed trailing stop loss price is invalid
        }
     }
   else  //-- SELL POSITION
     {
      slPrice = positionPriceCurrent + (trailingStopLoss * symbolPoint);

      //-- Check if the proposed slPrice for the trailing stop loss is valid
      if(slPrice > entryPrice || slPrice > currentPositionSlPrice)
        {
         return(false); //-- Exit the function, proposed trailing stop loss price is invalid
        }
     }

//-- Print position properties before setting the trailing stop loss
   string positionProperties = "--> "  + positionSymbol + " " + EnumToString(positionType) + " Trailing Stop Loss Modification Details" +
                               " <--\r\n";
   positionProperties += "------------------------------------------------------------\r\n";
   positionProperties += "Ticket: " + (string)positionTicket + "\r\n";
   positionProperties += "Volume: " + StringFormat("%G", volume) + "\r\n";
   positionProperties += "Price Open: " + StringFormat("%G", entryPrice) + "\r\n";
   positionProperties += "Current SL: " + StringFormat("%G", currentPositionSlPrice) + "   -> New Trailing SL: " + (string)slPrice + "\r\n";
   positionProperties += "Current TP: " + StringFormat("%G", currentPositionTpPrice) + "\r\n";
   positionProperties += "Comment: " + PositionGetString(POSITION_COMMENT) + "\r\n";
   positionProperties += "Magic Number: " + (string)PositionGetInteger(POSITION_MAGIC) + "\r\n";
   positionProperties += "---";
   Print(positionProperties);

//-- reset the the tradeRequest and tradeResult values by zeroing them
   ZeroMemory(tradeRequest);
   ZeroMemory(tradeResult);

//-- initialize the parameters to set the sltp
   tradeRequest.action = TRADE_ACTION_SLTP; //-- Trade operation type for setting sl and tp
   tradeRequest.position = positionTicket;
   tradeRequest.symbol = positionSymbol;
   tradeRequest.sl = slPrice;
   tradeRequest.tp = currentPositionTpPrice;

   ResetLastError(); //--- reset error cache so that we get an accurate runtime error code in the ErrorAdvisor function

   for(int loop = 0; loop <= 100; loop++) //-- try modifying the sl and tp 101 times untill the request is successful
     {
      //--- send order to the trade server
      if(OrderSend(tradeRequest, tradeResult))
        {
         //-- Confirm order execution
         if(tradeResult.retcode == 10008 || tradeResult.retcode == 10009)
           {
            PrintFormat("Successfully set the Trailing SL for #%I64d %s %s", positionTicket, positionSymbol, EnumToString(positionType));
            PrintFormat("retcode=%u  runtime_code=%u", tradeResult.retcode, GetLastError());
            Print("_______________________________________________________________________________________\r\n\r\n");
            return(true); //-- exit function
            break; //--- success - order placed ok. exit for loop
           }
        }
      else  //-- Order request failed
        {
         //-- order not sent or critical error found
         if(!ErrorAdvisor(__FUNCTION__, positionSymbol, tradeResult.retcode) || IsStopped())
           {
            PrintFormat("ERROR setting the Trailing SL for #%I64d %s %s", positionTicket, positionSymbol, EnumToString(positionType));
            Print("_______________________________________________________________________________________\r\n\r\n");
            return(false); //-- exit function
            break; //-- exit for loop
           }
        }
     }
   return(false);
  }
```

### Close All Positions Function

This function is designed to be very flexible and will be responsible for closing all open positions based on specified parameters. We will name the function _CloseAllPositions()_. It will scan for open positions matching the provided arguments or parameters of _symbol_ and _magic number_ and attempt to close them all. If trading is not allowed, the function will stop executing and exit immediately. The function will loop through all the open positions, filter them based on the specified criteria, and close all the matching positions.

After it finishes attempting to close all positions, it will then enter a loop to confirm that all targeted positions have been closed, handle any errors, and ensure we don't run into an infinite loop. When our function successfully closes all positions, it will exit, returning _true_; otherwise, it will return _false_.

Let's begin by defining the _CloseAllPositions()_ function. It will return a bool and accept two parameters with default values:

1. **string symbol**: This defaults to _ALL\_SYMBOLS_.
2. **ulong magicNumber**: This defaults to _0_.

```
bool CloseAllPositions(string symbol = ALL_SYMBOLS, ulong magicNumber = 0) export
  {
//-- Functions body goes here
  }
```

We need to check if trading is allowed. If trading is not allowed, we exit the function and return false.

```
if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }
```

Create a new bool type variable _returnThis_ to store the return value of the function and give it a default value of false.

```
bool returnThis = false;
```

Retrieve and save the total number of open positions and use this value in a for loop that will allow us to access and process all open positions. On each iteration, we will save the selected position properties and use this data to filter positions based on the provided _symbol_ and _magic number_. If the position does not match the criteria, we continue to the next position. If the position matches the criteria, we close the position using the _ClosePositionByTicket()_ function.

```
int totalOpenPositions = PositionsTotal();
   for(int x = 0; x < totalOpenPositions; x++)
     {
      //--- Get position properties
      ulong positionTicket = PositionGetTicket(x); //-- Get ticket to select the position
      string selectedSymbol = PositionGetString(POSITION_SYMBOL);
      ulong positionMagicNo = PositionGetInteger(POSITION_MAGIC);

      //-- Filter positions by symbol and magic number
      if(
         (symbol != ALL_SYMBOLS && symbol != selectedSymbol) ||
         (magicNumber != 0 && positionMagicNo != magicNumber)
      )
        {
         continue;
        }

      //-- Close the position
      ClosePositionByTicket(positionTicket);
     }
```

We have now iterated over all the specified open positions and sent termination requests to the trade server to close them. Before concluding the function, we need to confirm that all the targeted positions have actually been closed. To do this, we will use a loop that repeatedly calls the _CloseAllPositions()_ function recursively until all positions matching our criteria are closed. In each iteration, we will attempt to close any remaining positions, pause sending the order for a short period to pace the execution to avoid overwhelming the trade server and increment a breaker counter to avoid infinite loops. We will also check for critical errors and other exit conditions (such as the script being stopped or exceeding the maximum loop count). If any of these conditions are met, we will break out of the loop.

```
int breakerBreaker = 0; //-- Variable that safeguards and makes sure we are not locked in an infinite loop
   while(SymbolPositionsTotal(symbol, magicNumber) > 0)
     {
      breakerBreaker++;
      CloseAllPositions(symbol, magicNumber); //-- We still have some open positions, do a function callback
      Sleep(100); //-- Micro sleep to pace the execution and give some time to the trade server

      //-- Check for critical errors so that we exit the loop if we run into trouble
      if(!ErrorAdvisor(__FUNCTION__, symbol, GetLastError()) || IsStopped() || breakerBreaker > 101)
        {
         break;
        }
     }
```

Re-check to confirm that all the targeted positions have been closed and save this status in the return variable before finally concluding and exiting the function.

```
if(SymbolPositionsTotal(symbol, magicNumber) == 0)
     {
      returnThis = true; //-- Save this status for the function return value
     }

   return(returnThis);
```

Confirm that all the _CloseAllPositions()_ function code segments are complete in the sequence below:

```
bool CloseAllPositions(string symbol = ALL_SYMBOLS, ulong magicNumber = 0) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

   bool returnThis = false;

//-- Scan for symbol and magic number specific positions and close them
   int totalOpenPositions = PositionsTotal();
   for(int x = 0; x < totalOpenPositions; x++)
     {
      //--- Get position properties
      ulong positionTicket = PositionGetTicket(x); //-- Get ticket to select the position
      string selectedSymbol = PositionGetString(POSITION_SYMBOL);
      ulong positionMagicNo = PositionGetInteger(POSITION_MAGIC);

      //-- Filter positions by symbol and magic number
      if(
         (symbol != ALL_SYMBOLS && symbol != selectedSymbol) ||
         (magicNumber != 0 && positionMagicNo != magicNumber)
      )
        {
         continue;
        }

      //-- Close the position
      ClosePositionByTicket(positionTicket);
     }

//-- Confirm that we have closed all the positions being targeted
   int breakerBreaker = 0; //-- Variable that safeguards and makes sure we are not locked in an infinite loop
   while(SymbolPositionsTotal(symbol, magicNumber) > 0)
     {
      breakerBreaker++;
      CloseAllPositions(symbol, magicNumber); //-- We still have some open positions, do a function callback
      Sleep(100); //-- Micro sleep to pace the execution and give some time to the trade server

      //-- Check for critical errors so that we exit the loop if we run into trouble
      if(!ErrorAdvisor(__FUNCTION__, symbol, GetLastError()) || IsStopped() || breakerBreaker > 101)
        {
         break;
        }
     }

//-- Final confirmations that all targeted positions have been closed
   if(SymbolPositionsTotal(symbol, magicNumber) == 0)
     {
      returnThis = true; //-- Save this status for the function return value
     }

   return(returnThis);
  }
```

### Close All Positions Function Overloading

For convenience, we will overload the _CloseAllPositions()_ function with a second version that accepts no parameters and closes all open positions in the account when called. This function will be exportable and ready for use in the Positions Manager EX5 library as well.

```
//+------------------------------------------------------------------+
//| CloseAllPositions(): Closes all positions in the account         |
//+------------------------------------------------------------------+
bool CloseAllPositions() export
  {
   return(CloseAllPositions(ALL_SYMBOLS, 0));
  }
```

### Sorting and Filtering Position Closing Functions

By browsing the MQL5 developer forums, you will often encounter questions from beginner MQL5 developers seeking assistance with coding algorithms to filter, sort, and manage various position operations, such as closing or modifying specific positions based on criteria like magic number, profit, or loss status, e.t.c. The following library of functions aims to address this need, making it easier and quicker to implement these operations effectively.

The sorting and filtering position-closing functions below implement a similar approach to the _CloseAllPositions()_ function, but have notable differences unique to each. They use a recursive programming strategy to ensure all specified positions are closed and include trace logs to print and record any encountered errors in the Expert Advisors log for end-user diagnostics. An added advantage of these functions is their high success rate in accomplishing their specified goals, as they recursively scan for recoverable errors and send the specified trade requests multiple times to ensure the orders are successful. For a more in-depth understanding of how each function works, I have included detailed code comments to explain the structure and organization of each code component within the functions.

### Close All Buy Positions

The _CloseAllBuyPositions()_ function is responsible for closing all open buy positions that match the provided symbol name and magic number function parameters or arguments. It returns a bool, indicating _true_ if it successfully closes all specified positions and _false_ if it does not.

```
bool CloseAllBuyPositions(string symbol = ALL_SYMBOLS, ulong magicNumber = 0) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

   bool returnThis = false;

//-- Scan for symbol and magic number specific buy positions and close them
   int totalOpenPositions = PositionsTotal();
   for(int x = 0; x < totalOpenPositions; x++)
     {
      //--- Get position properties
      ulong positionTicket = PositionGetTicket(x); //-- Get ticket to select the position
      string selectedSymbol = PositionGetString(POSITION_SYMBOL);
      ulong positionMagicNo = PositionGetInteger(POSITION_MAGIC);
      ulong positionType = PositionGetInteger(POSITION_TYPE);

      //-- Filter positions by symbol, type and magic number
      if(
         (symbol != ALL_SYMBOLS && symbol != selectedSymbol) || (positionType != POSITION_TYPE_BUY) ||
         (magicNumber != 0 && positionMagicNo != magicNumber)
      )
        {
         continue;
        }

      //-- Close the position
      ClosePositionByTicket(positionTicket);
     }

//-- Confirm that we have closed all the buy positions being targeted
   int breakerBreaker = 0; //-- Variable that safeguards and makes sure we are not locked in an infinite loop
   while(SymbolBuyPositionsTotal(symbol, magicNumber) > 0)
     {
      breakerBreaker++;
      CloseAllBuyPositions(symbol, magicNumber); //-- We still have some open buy positions, do a function callback
      Sleep(100); //-- Micro sleep to pace the execution and give some time to the trade server

      //-- Check for critical errors so that we exit the loop if we run into trouble
      if(!ErrorAdvisor(__FUNCTION__, symbol, GetLastError()) || IsStopped() || breakerBreaker > 101)
        {
         break;
        }
     }

   if(SymbolBuyPositionsTotal(symbol, magicNumber) == 0)
     {
      returnThis = true;
     }
   return(returnThis);
  }
```

### Close All Sell Positions

The _CloseAllSellPositions()_ function is tasked with closing all open sell positions that match the provided symbol name and magic number function parameters or arguments. It returns a bool, indicating _true_ if it successfully closes all specified positions and _false_ if it does not.

```
bool CloseAllSellPositions(string symbol = ALL_SYMBOLS, ulong magicNumber = 0) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

   bool returnThis = false;

//-- Scan for symbol and magic number specific sell positions and close them
   int totalOpenPositions = PositionsTotal();
   for(int x = 0; x < totalOpenPositions; x++)
     {
      //--- Get position properties
      ulong positionTicket = PositionGetTicket(x); //-- Get ticket to select the position
      string selectedSymbol = PositionGetString(POSITION_SYMBOL);
      ulong positionMagicNo = PositionGetInteger(POSITION_MAGIC);
      ulong positionType = PositionGetInteger(POSITION_TYPE);

      //-- Filter positions by symbol, type and magic number
      if(
         (symbol != ALL_SYMBOLS && symbol != selectedSymbol) || (positionType != POSITION_TYPE_SELL) ||
         (magicNumber != 0 && positionMagicNo != magicNumber)
      )
        {
         continue;
        }

      //-- Close the position
      ClosePositionByTicket(positionTicket);
     }

//-- Confirm that we have closed all the sell positions being targeted
   int breakerBreaker = 0; //-- Variable that safeguards and makes sure we are not locked in an infinite loop
   while(SymbolSellPositionsTotal(symbol, magicNumber) > 0)
     {
      breakerBreaker++;
      CloseAllSellPositions(symbol, magicNumber); //-- We still have some open sell positions, do a function callback
      Sleep(100); //-- Micro sleep to pace the execution and give some time to the trade server

      //-- Check for critical errors so that we exit the loop if we run into trouble
      if(!ErrorAdvisor(__FUNCTION__, symbol, GetLastError()) || IsStopped() || breakerBreaker > 101)
        {
         break;
        }
     }

   if(SymbolSellPositionsTotal(symbol, magicNumber) == 0)
     {
      returnThis = true;
     }
   return(returnThis);
  }
```

### Close All Magic Positions

The _CloseAllMagicPositions()_ function is responsible for closing all open positions that match the provided magic number function parameter or argument. It returns a bool, indicating _true_ if it successfully closes all specified positions and _false_ if it does not.

```
bool CloseAllMagicPositions(ulong magicNumber) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

   bool returnThis = false;

//-- Variables to store the selected positions data
   ulong positionTicket, positionMagicNo;
   string positionSymbol;

//-- Scan for magic number specific positions and close them
   int totalOpenPositions = PositionsTotal();
   for(int x = 0; x < totalOpenPositions; x++)
     {
      //--- Get position properties
      positionTicket = PositionGetTicket(x); //-- Get ticket to select the position
      positionMagicNo = PositionGetInteger(POSITION_MAGIC);
      positionSymbol = PositionGetString(POSITION_SYMBOL);

      //-- Filter positions by magic number
      if(magicNumber == positionMagicNo)
        {
         //-- Close the position
         ClosePositionByTicket(positionTicket);
        }
     }

//-- Confirm that we have closed all the positions being targeted
   int breakerBreaker = 0; //-- Variable that safeguards and makes sure we are not locked in an infinite loop
   while(MagicPositionsTotal(magicNumber) > 0)
     {
      breakerBreaker++;
      CloseAllMagicPositions(magicNumber); //-- We still have some open positions, do a function callback
      Sleep(100); //-- Micro sleep to pace the execution and give some time to the trade server

      //-- Check for critical errors so that we exit the loop if we run into trouble
      if(!ErrorAdvisor(__FUNCTION__, positionSymbol, GetLastError()) || IsStopped() || breakerBreaker > 101)
        {
         break;
        }
     }

   if(MagicPositionsTotal(magicNumber) == 0)
     {
      returnThis = true;
     }
   return(returnThis);
  }
```

### Close All Profitable Positions

The _CloseAllProfitablePositions()_ function closes all profitable open positions that match the provided symbol name and magic number function parameters or arguments. It returns a bool, indicating _true_ if it successfully closes all specified positions and _false_ if it does not.

```
bool CloseAllProfitablePositions(string symbol = ALL_SYMBOLS, ulong magicNumber = 0) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

//-- Scan for profitable positions that match the specified symbol and magic number to close them
   int totalOpenPositions = PositionsTotal();
   for(int x = 0; x < totalOpenPositions; x++)
     {
      //--- Get position properties
      ulong positionTicket = PositionGetTicket(x); //-- Get ticket to select the position
      string selectedSymbol = PositionGetString(POSITION_SYMBOL);
      ulong positionMagicNo = PositionGetInteger(POSITION_MAGIC);
      double positionProfit = PositionGetDouble(POSITION_PROFIT);

      //-- Filter positions by symbol, magic number and profit
      if(
         ((symbol != ALL_SYMBOLS && symbol != selectedSymbol) || (magicNumber != 0 && positionMagicNo != magicNumber)) ||
         positionProfit <= 0
      )
        {
         continue;
        }

      //-- Close the position
      ClosePositionByTicket(positionTicket);
     }
   return(true);
  }
```

\\_\\_\\_

### Close All Profitable Buy Positions

The _CloseAllProfitableBuyPositions()_ function closes all profitable open buy positions that match the provided symbol name and magic number function parameters or arguments. It returns a bool, indicating _true_ if it successfully closes all specified positions and _false_ if it does not.

```
bool CloseAllProfitableBuyPositions(string symbol = ALL_SYMBOLS, ulong magicNumber = 0) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

//-- Scan for profitable positions that match the specified symbol and magic number to close them
   int totalOpenPositions = PositionsTotal();
   for(int x = 0; x < totalOpenPositions; x++)
     {
      //--- Get position properties
      ulong positionTicket = PositionGetTicket(x); //-- Get ticket to select the position
      string selectedSymbol = PositionGetString(POSITION_SYMBOL);
      ulong positionMagicNo = PositionGetInteger(POSITION_MAGIC);
      double positionProfit = PositionGetDouble(POSITION_PROFIT);

      //-- Filter positions by symbol, magic number, profit and type
      if(
         ((symbol != ALL_SYMBOLS && symbol != selectedSymbol) || (magicNumber != 0 && positionMagicNo != magicNumber)) ||
         positionProfit <= 0 || PositionGetInteger(POSITION_TYPE) != POSITION_TYPE_BUY
      )
        {
         continue;
        }

      //-- Close the position
      ClosePositionByTicket(positionTicket);
     }
   return(true);
  }
```

### Close All Profitable Sell Positions

The _CloseAllProfitableSellPositions()_ function closes all profitable open sell positions that match the provided symbol name and magic number function parameters or arguments. It returns a bool, indicating _true_ if it successfully closes all specified positions and _false_ if it does not.

```
bool CloseAllProfitableSellPositions(string symbol = ALL_SYMBOLS, ulong magicNumber = 0) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

//-- Scan for profitable positions that match the specified symbol and magic number to close them
   int totalOpenPositions = PositionsTotal();
   for(int x = 0; x < totalOpenPositions; x++)
     {
      //--- Get position properties
      ulong positionTicket = PositionGetTicket(x); //-- Get ticket to select the position
      string selectedSymbol = PositionGetString(POSITION_SYMBOL);
      ulong positionMagicNo = PositionGetInteger(POSITION_MAGIC);
      double positionProfit = PositionGetDouble(POSITION_PROFIT);

      //-- Filter positions by symbol, magic number, profit and type
      if(
         ((symbol != ALL_SYMBOLS && symbol != selectedSymbol) || (magicNumber != 0 && positionMagicNo != magicNumber)) ||
         positionProfit <= 0 || PositionGetInteger(POSITION_TYPE) != POSITION_TYPE_SELL
      )
        {
         continue;
        }

      //-- Close the position
      ClosePositionByTicket(positionTicket);
     }
   return(true);
  }
```

### Close All Loss Positions

The _CloseAllLossPositions()_ function closes all losing open positions that match the provided symbol name and magic number function parameters or arguments. It returns a bool, indicating _true_ if it successfully closes all specified positions and _false_ if it does not.

```
bool CloseAllLossPositions(string symbol = ALL_SYMBOLS, ulong magicNumber = 0) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

//-- Scan for loss positions that match the specified symbol and magic number and close them
   int totalOpenPositions = PositionsTotal();
   for(int x = 0; x < totalOpenPositions; x++)
     {
      //--- Get position properties
      ulong positionTicket = PositionGetTicket(x); //-- Get ticket to select the position
      string selectedSymbol = PositionGetString(POSITION_SYMBOL);
      ulong positionMagicNo = PositionGetInteger(POSITION_MAGIC);
      double positionProfit = PositionGetDouble(POSITION_PROFIT);

      //-- Filter positions by symbol, magic number and profit
      if(
         ((symbol != ALL_SYMBOLS && symbol != selectedSymbol) || (magicNumber != 0 && positionMagicNo != magicNumber)) ||
         positionProfit > 0
      )
        {
         continue;
        }

      //-- Close the position
      ClosePositionByTicket(positionTicket);
     }

   return(true);
  }
```

### Close All Loss Buy Positions

The _CloseAllLossBuyPositions()_ function closes all losing open buy positions that match the provided symbol name and magic number function parameters or arguments. It returns a bool, indicating _true_ if it successfully closes all specified positions and _false_ if it does not.

```
bool CloseAllLossBuyPositions(string symbol = ALL_SYMBOLS, ulong magicNumber = 0) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

//-- Scan for loss positions that match the specified symbol and magic number and close them
   int totalOpenPositions = PositionsTotal();
   for(int x = 0; x < totalOpenPositions; x++)
     {
      //--- Get position properties
      ulong positionTicket = PositionGetTicket(x); //-- Get ticket to select the position
      string selectedSymbol = PositionGetString(POSITION_SYMBOL);
      ulong positionMagicNo = PositionGetInteger(POSITION_MAGIC);
      double positionProfit = PositionGetDouble(POSITION_PROFIT);

      //-- Filter positions by symbol, magic number, profit and type
      if(
         ((symbol != ALL_SYMBOLS && symbol != selectedSymbol) || (magicNumber != 0 && positionMagicNo != magicNumber)) ||
         positionProfit > 0 || PositionGetInteger(POSITION_TYPE) != POSITION_TYPE_BUY
      )
        {
         continue;
        }

      //-- Close the position
      ClosePositionByTicket(positionTicket);
     }

   return(true);
  }
```

### Close All Loss Sell Positions

The _CloseAllLossSellPositions()_ function closes all losing open sell positions that match the provided symbol name and magic number function parameters or arguments. It returns a bool, indicating _true_ if it successfully closes all specified positions and _false_ if it does not.

```
bool CloseAllLossSellPositions(string symbol = ALL_SYMBOLS, ulong magicNumber = 0) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

//-- Scan for loss positions that match the specified symbol and magic number and close them
   int totalOpenPositions = PositionsTotal();
   for(int x = 0; x < totalOpenPositions; x++)
     {
      //--- Get position properties
      ulong positionTicket = PositionGetTicket(x); //-- Get ticket to select the position
      string selectedSymbol = PositionGetString(POSITION_SYMBOL);
      ulong positionMagicNo = PositionGetInteger(POSITION_MAGIC);
      double positionProfit = PositionGetDouble(POSITION_PROFIT);

      //-- Filter positions by symbol, magic number, profit and type
      if(
         ((symbol != ALL_SYMBOLS && symbol != selectedSymbol) || (magicNumber != 0 && positionMagicNo != magicNumber)) ||
         positionProfit > 0 || PositionGetInteger(POSITION_TYPE) != POSITION_TYPE_SELL
      )
        {
         continue;
        }

      //-- Close the position
      ClosePositionByTicket(positionTicket);
     }

   return(true);
  }
```

### Position Status Functions

When developing a trading system, it is important to keep track of the account status using real-time data on the various positions. Whether you are developing a grid-based strategy or a conservative price action strategy, having a clear and straightforward overview of all open positions is critical for the success of your trading system. However, there are no straight out-of-the-box standard language functions that provide this information. This EX5 library aims to simplify position information gathering with one-line function calls. The exportable functions below will give you the edge you need to monitor positions and decide whether to close or add to them, forming one of the foundational pillars of your trading system.

### Get Positions Data

The _GetPositionsData()_ function plays a crucial role in gathering and storing all the required position status information. It saves this data in global variables that are readily accessible within the entire library. These variables are meant to be continuously updated on every tick, ensuring they remain accurate and reliable.

Place the following global variable declarations at the top of our library below the trade operations request and result data structures global variables declarations.

```
string accountCurrency = AccountInfoString(ACCOUNT_CURRENCY);

//-- Position status global variables
//-------------------------------------------------------------------------------------------------------------------
int accountBuyPositionsTotal = 0, accountSellPositionsTotal = 0,
    symbolPositionsTotal = 0, symbolBuyPositionsTotal = 0, symbolSellPositionsTotal = 0,
    magicPositionsTotal = 0, magicBuyPositionsTotal = 0, magicSellPositionsTotal = 0;
double accountPositionsVolumeTotal = 0.0, accountBuyPositionsVolumeTotal = 0.0, accountSellPositionsVolumeTotal = 0.0,
       accountBuyPositionsProfit = 0.0, accountSellPositionsProfit = 0.0,
       symbolPositionsVolumeTotal = 0.0, symbolBuyPositionsVolumeTotal = 0.0,
       symbolSellPositionsVolumeTotal = 0.0, symbolPositionsProfit = 0.0,
       symbolBuyPositionsProfit = 0.0, symbolSellPositionsProfit = 0.0,
       magicPositionsVolumeTotal = 0.0, magicBuyPositionsVolumeTotal = 0.0,
       magicSellPositionsVolumeTotal = 0.0, magicPositionsProfit = 0.0,
       magicBuyPositionsProfit = 0.0, magicSellPositionsProfit = 0.0;
```

Update and save the position status data in the _GetPositionsData()_ function below.

```
void GetPositionsData(string symbol, ulong magicNumber)
  {
//-- Reset the acount open positions status
   accountBuyPositionsTotal = 0;
   accountSellPositionsTotal = 0;
   accountPositionsVolumeTotal = 0.0;
   accountBuyPositionsVolumeTotal = 0.0;
   accountSellPositionsVolumeTotal = 0.0;
   accountBuyPositionsProfit = 0.0;
   accountSellPositionsProfit = 0.0;

//-- Reset the EA's magic open positions status
   magicPositionsTotal = 0;
   magicBuyPositionsTotal = 0;
   magicSellPositionsTotal = 0;
   magicPositionsVolumeTotal = 0.0;
   magicBuyPositionsVolumeTotal = 0.0;
   magicSellPositionsVolumeTotal = 0.0;
   magicPositionsProfit = 0.0;
   magicBuyPositionsProfit = 0.0;
   magicSellPositionsProfit = 0.0;

//-- Reset the symbol open positions status
   symbolPositionsTotal = 0;
   symbolBuyPositionsTotal = 0;
   symbolSellPositionsTotal = 0;
   symbolPositionsVolumeTotal = 0.0;
   symbolBuyPositionsVolumeTotal = 0.0;
   symbolSellPositionsVolumeTotal = 0.0;
   symbolPositionsProfit = 0.0;
   symbolBuyPositionsProfit = 0.0;
   symbolSellPositionsProfit = 0.0;

//-- Update and save the open positions status with realtime data
   int totalOpenPositions = PositionsTotal();
   if(totalOpenPositions > 0)
     {
      //-- Scan for symbol and magic number specific positions and save their status
      for(int x = 0; x < totalOpenPositions; x++)
        {
         //--- Get position properties
         ulong  positionTicket = PositionGetTicket(x); //-- Get ticket to select the position
         string selectedSymbol = PositionGetString(POSITION_SYMBOL);
         ulong positionMagicNo = PositionGetInteger(POSITION_MAGIC);

         //-- Filter positions by magic number
         if(magicNumber != 0 && positionMagicNo != magicNumber)
           {
            continue;
           }

         //-- Save the account positions status first
         accountPositionsVolumeTotal += PositionGetDouble(POSITION_VOLUME);

         if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
           {
            //-- Account properties
            ++accountBuyPositionsTotal;
            accountBuyPositionsVolumeTotal += PositionGetDouble(POSITION_VOLUME);
            accountBuyPositionsProfit += PositionGetDouble(POSITION_PROFIT);
           }
         else //-- POSITION_TYPE_SELL
           {
            //-- Account properties
            ++accountSellPositionsTotal;
            accountSellPositionsVolumeTotal += PositionGetDouble(POSITION_VOLUME);
            accountSellPositionsProfit += PositionGetDouble(POSITION_PROFIT);
           }

         //-- Filter positions openend by EA and save their status
         if(
            PositionGetInteger(POSITION_REASON) == POSITION_REASON_EXPERT &&
            positionMagicNo == magicNumber
         )
           {
            ++magicPositionsTotal;
            magicPositionsProfit += PositionGetDouble(POSITION_PROFIT);
            magicPositionsVolumeTotal += PositionGetDouble(POSITION_VOLUME);
            if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
              {
               //-- Magic properties
               ++magicBuyPositionsTotal;
               magicBuyPositionsProfit += PositionGetDouble(POSITION_PROFIT);
               magicBuyPositionsVolumeTotal += PositionGetDouble(POSITION_VOLUME);
              }
            else //-- POSITION_TYPE_SELL
              {
               //-- Magic properties
               ++magicSellPositionsTotal;
               magicSellPositionsProfit += PositionGetDouble(POSITION_PROFIT);
               magicSellPositionsVolumeTotal += PositionGetDouble(POSITION_VOLUME);
              }
           }

         //-- Filter positions by symbol
         if(symbol == ALL_SYMBOLS || selectedSymbol == symbol)
           {
            ++symbolPositionsTotal;
            symbolPositionsVolumeTotal += PositionGetDouble(POSITION_VOLUME);
            symbolPositionsProfit += PositionGetDouble(POSITION_PROFIT);
            if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
              {
               ++symbolBuyPositionsTotal;
               symbolBuyPositionsVolumeTotal += PositionGetDouble(POSITION_VOLUME);
               symbolBuyPositionsProfit += PositionGetDouble(POSITION_PROFIT);
              }
            else //-- POSITION_TYPE_SELL
              {
               ++symbolSellPositionsTotal;
               symbolSellPositionsVolumeTotal += PositionGetDouble(POSITION_VOLUME);
               symbolSellPositionsProfit += PositionGetDouble(POSITION_PROFIT);
              }
           }
        }
     }
  }
```

To gain access to the different position status properties we have captured and saved in the global variables above, we need to create simple exportable functions that can be externally accessible from an external code base. We will accomplish this by coding the functions below.

### Buy Positions Total

Returns an integer value of the total number of all the open buy positions in the account.

```
int BuyPositionsTotal() export
  {
   GetPositionsData(ALL_SYMBOLS, 0);
   return(accountBuyPositionsTotal);
  }
```

### Sell Positions Total

Returns an integer value of the total number of all the open sell positions in the account.

```
int SellPositionsTotal() export
  {
   GetPositionsData(ALL_SYMBOLS, 0);
   return(accountSellPositionsTotal);
  }
```

### Positions Total Volume

Returns a double value of the total volume/lot/quantity of all the open positions in the account.

```
double PositionsTotalVolume() export
  {
   GetPositionsData(ALL_SYMBOLS, 0);
   return(accountPositionsVolumeTotal);
  }
```

### Buy Positions Total Volume

Returns a double value of the total volume/lot/quantity of all the open buy positions in the account.

```
double BuyPositionsTotalVolume() export
  {
   GetPositionsData(ALL_SYMBOLS, 0);
   return(accountBuyPositionsVolumeTotal);
  }
```

### Sell Positions Total Volume

Returns a double value of the total volume/lot/quantity of all the open sell positions in the account.

```
double SellPositionsTotalVolume() export
  {
   GetPositionsData(ALL_SYMBOLS, 0);
   return(accountSellPositionsVolumeTotal);
  }
```

### Buy Positions Profit

Returns a double value of the total profit of all the open buy positions in the account.

```
double BuyPositionsProfit() export
  {
   GetPositionsData(ALL_SYMBOLS, 0);
   return(accountBuyPositionsProfit);
  }
```

### Sell Positions Profit

Returns a double value of the total profit of all the open sell positions in the account.

```
double SellPositionsProfit() export
  {
   GetPositionsData(ALL_SYMBOLS, 0);
   return(accountSellPositionsProfit);
  }
```

### Magic Positions Total

Returns an integer value of the total number of all the open positions for the specified magic number in the account.

```
int MagicPositionsTotal(ulong magicNumber) export
  {
   GetPositionsData(ALL_SYMBOLS, magicNumber);
   return(magicPositionsTotal);
  }
```

### Magic Buy Positions Total

Returns an integer value of the total number of all the open buy positions for the specified magic number in the account.

```
int MagicBuyPositionsTotal(ulong magicNumber) export
  {
   GetPositionsData(ALL_SYMBOLS, magicNumber);
   return(magicBuyPositionsTotal);
  }
```

### Magic Sell Positions Total

Returns an integer value of the total number of all the open sell positions for the specified magic number in the account.

```
int MagicSellPositionsTotal(ulong magicNumber) export
  {
   GetPositionsData(ALL_SYMBOLS, magicNumber);
   return(magicSellPositionsTotal);
  }
```

### Magic Positions Total Volume

Returns a double value of the total volume/lot/quantity of all the open positions for the specified magic number in the account.

```
double MagicPositionsTotalVolume(ulong magicNumber) export
  {
   GetPositionsData(ALL_SYMBOLS, magicNumber);
   return(magicPositionsVolumeTotal);
  }
```

### Magic Buy Positions Total Volume

Returns a double value of the total volume/lot/quantity of all the open buy positions for the specified magic number in the account.

```
double MagicBuyPositionsTotalVolume(ulong magicNumber) export
  {
   GetPositionsData(ALL_SYMBOLS, magicNumber);
   return(magicBuyPositionsVolumeTotal);
  }
```

### Magic Sell Positions Total Volume

Returns a double value of the total volume/lot/quantity of all the open sell positions for the specified magic number in the account.

```
double MagicSellPositionsTotalVolume(ulong magicNumber) export
  {
   GetPositionsData(ALL_SYMBOLS, magicNumber);
   return(magicSellPositionsVolumeTotal);
  }
```

### Magic Positions Profit

Returns a double value of the total profit of all the open positions for the specified magic number in the account.

```
double MagicPositionsProfit(ulong magicNumber) export
  {
   GetPositionsData(ALL_SYMBOLS, magicNumber);
   return(magicPositionsProfit);
  }
```

### Magic Buy Positions Profit

Returns a double value of the total profit of all the open buy positions for the specified magic number in the account.

```
double MagicBuyPositionsProfit(ulong magicNumber) export
  {
   GetPositionsData(ALL_SYMBOLS, magicNumber);
   return(magicBuyPositionsProfit);
  }
```

### Magic Sell Positions Profit

Returns a double value of the total profit of all the open sell positions for the specified magic number in the account.

```
double MagicSellPositionsProfit(ulong magicNumber) export
  {
   GetPositionsData(ALL_SYMBOLS, magicNumber);
   return(magicSellPositionsProfit);
  }
```

### Symbol Positions Total

Returns an integer value of the total number of all the open positions for a specified symbol in the account.

```
int SymbolPositionsTotal(string symbol, ulong magicNumber) export
  {
   GetPositionsData(symbol, magicNumber);
   return(symbolPositionsTotal);
  }
```

### Symbol Buy Positions Total

Returns an integer value of the total number of all the open buy positions for a specified symbol in the account.

```
int SymbolBuyPositionsTotal(string symbol, ulong magicNumber) export
  {
   GetPositionsData(symbol, magicNumber);
   return(symbolBuyPositionsTotal);
  }
```

### Symbol Sell Positions Total

Returns an integer value of the total number of all the open sell positions for a specified symbol in the account.

```
int SymbolSellPositionsTotal(string symbol, ulong magicNumber) export
  {
   GetPositionsData(symbol, magicNumber);
   return(symbolSellPositionsTotal);
  }
```

### Symbol Positions Total Volume

Returns a double value of the total volume/lot/quantity of all the open positions for a specified symbol in the account.

```
double SymbolPositionsTotalVolume(string symbol, ulong magicNumber) export
  {
   GetPositionsData(symbol, magicNumber);
   return(symbolPositionsVolumeTotal);
  }
```

### Symbol Buy Positions Total Volume

Returns a double value of the total volume/lot/quantity of all the open buy positions for a specified symbol in the account.

```
double SymbolBuyPositionsTotalVolume(string symbol, ulong magicNumber) export
  {
   GetPositionsData(symbol, magicNumber);
   return(symbolBuyPositionsVolumeTotal);
  }
```

### Symbol Sell Positions Total Volume

Returns a double value of the total volume/lot/quantity of all the open sell positions for a specified symbol in the account.

```
double SymbolSellPositionsTotalVolume(string symbol, ulong magicNumber) export
  {
   GetPositionsData(symbol, magicNumber);
   return(symbolSellPositionsVolumeTotal);
  }
```

### Symbol Positions Profit

Returns a double value of the total profit of all the open positions for a specified symbol in the account.

```
double SymbolPositionsProfit(string symbol, ulong magicNumber) export
  {
   GetPositionsData(symbol, magicNumber);
   return(NormalizeDouble(symbolPositionsProfit, 2));
  }
```

### Symbol Buy Positions Profit

Returns a double value of the total profit of all the open buy positions for a specified symbol in the account.

```
double SymbolBuyPositionsProfit(string symbol, ulong magicNumber) export
  {
   GetPositionsData(symbol, magicNumber);
   return(NormalizeDouble(symbolBuyPositionsProfit, 2));
  }
```

### Symbol Sell Positions Profit

Returns a double value of the total profit of all the open sell positions for a specified symbol in the account.

```
double SymbolSellPositionsProfit(string symbol, ulong magicNumber) export
  {
   GetPositionsData(symbol, magicNumber);
   return(NormalizeDouble(symbolSellPositionsProfit, 2));
  }
```

### Account Positions Status

Returns a pre-formatted string containing the account positions' status, which can be printed to the log or displayed in the chart comments. The function accepts a single boolean parameter called _formatForComment_. If _formatForComment_ is _true_, the function formats the data for display in the chart window. If it is _false_, the data is formatted for display in the Expert Advisor's logs tab.

```
string AccountPositionsStatus(bool formatForComment) export
  {
   GetPositionsData(ALL_SYMBOLS, 0); //-- Update the position status variables before we display their data
   string spacer = "";
   if(formatForComment) //-- Add some formating space for the chart comment string
     {
      spacer = "                                        ";
     }
   string accountPositionsStatus = "\r\n" + spacer + "|---------------------------------------------------------------------------\r\n";
   accountPositionsStatus += spacer + "| " + (string)AccountInfoInteger(ACCOUNT_LOGIN) + " - ACCOUNT POSTIONS STATUS \r\n";
   accountPositionsStatus += spacer + "|---------------------------------------------------------------------------\r\n";
   accountPositionsStatus += spacer + "|     Total Open:   " + (string)PositionsTotal() + "\r\n";
   accountPositionsStatus += spacer + "|     Total Volume: " + (string)accountPositionsVolumeTotal + "\r\n";
   accountPositionsStatus += spacer + "|     Total Profit: " +
   (string)(NormalizeDouble(AccountInfoDouble(ACCOUNT_PROFIT), 2)) + accountCurrency + "\r\n";
   accountPositionsStatus += spacer + "|------------------------------------------------------------------\r\n";
   accountPositionsStatus += spacer + "| BUY POSITIONS: \r\n";
   accountPositionsStatus += spacer + "|     Total Open:   " + (string)accountBuyPositionsTotal + "\r\n";
   accountPositionsStatus += spacer + "|     Total Volume: " + (string)accountBuyPositionsVolumeTotal + "\r\n";
   accountPositionsStatus += spacer + "|     Total Profit: " + (string)(NormalizeDouble(accountBuyPositionsProfit, 2)) +
   accountCurrency + "\r\n";
   accountPositionsStatus += spacer + "|------------------------------------------------------------------\r\n";
   accountPositionsStatus += spacer + "| SELL POSITIONS: \r\n";
   accountPositionsStatus += spacer + "|     Total Open:   " + (string)accountSellPositionsTotal + "\r\n";
   accountPositionsStatus += spacer + "|     Total Volume: " + (string)accountSellPositionsVolumeTotal + "\r\n";
   accountPositionsStatus += spacer + "|     Total Profit: " + (string)(NormalizeDouble(accountSellPositionsProfit, 2)) +
   accountCurrency + "\r\n";
   accountPositionsStatus += spacer + "|---------------------------------------------------------------------------\r\n";
   accountPositionsStatus += spacer + "\r\n";
   return(accountPositionsStatus);
  }
```

### Magic Positions Status

Returns a pre-formatted string containing the account magic positions' status, which can be printed to the log or displayed in the chart comments. The function accepts two arguments or parameters. An unsigned long _magicNumber_ and a boolean _formatForComment_. If _formatForComment_ is _true_, the function formats the data for display in the chart window. If it is _false_, the data is formatted for display in the Expert Advisor's logs tab.

```
string MagicPositionsStatus(ulong magicNumber, bool formatForComment) export
  {
   GetPositionsData(ALL_SYMBOLS, magicNumber); //-- Update the position status variables before we display their data
   string spacer = "";
   if(formatForComment) //-- Add some formating space for the chart comment string
     {
      spacer = "                                        ";
     }
   string magicPositionsStatus = "\r\n" + spacer + "|---------------------------------------------------------------------------\r\n";
   magicPositionsStatus += spacer + "| " + (string)magicNumber + " - MAGIC POSTIONS STATUS \r\n";
   magicPositionsStatus += spacer + "|---------------------------------------------------------------------------\r\n";
   magicPositionsStatus += spacer + "|     Total Open:   " + (string)magicPositionsTotal + "\r\n";
   magicPositionsStatus += spacer + "|     Total Volume: " + (string)magicPositionsVolumeTotal + "\r\n";
   magicPositionsStatus += spacer + "|     Total Profit: " +
   (string)(NormalizeDouble(magicPositionsProfit, 2)) + accountCurrency + "\r\n";
   magicPositionsStatus += spacer + "|------------------------------------------------------------------\r\n";
   magicPositionsStatus += spacer + "| BUY POSITIONS: \r\n";
   magicPositionsStatus += spacer + "|     Total Open:   " + (string)magicBuyPositionsTotal + "\r\n";
   magicPositionsStatus += spacer + "|     Total Volume: " + (string)magicBuyPositionsVolumeTotal + "\r\n";
   magicPositionsStatus += spacer + "|     Total Profit: " + (string)(NormalizeDouble(magicBuyPositionsProfit, 2)) +
   accountCurrency + "\r\n";
   magicPositionsStatus += spacer + "|------------------------------------------------------------------\r\n";
   magicPositionsStatus += spacer + "| SELL POSITIONS: \r\n";
   magicPositionsStatus += spacer + "|     Total Open:   " + (string)magicSellPositionsTotal + "\r\n";
   magicPositionsStatus += spacer + "|     Total Volume: " + (string)magicSellPositionsVolumeTotal + "\r\n";
   magicPositionsStatus += spacer + "|     Total Profit: " + (string)(NormalizeDouble(magicSellPositionsProfit, 2)) +
   accountCurrency + "\r\n";
   magicPositionsStatus += spacer + "|---------------------------------------------------------------------------\r\n";
   magicPositionsStatus += spacer + "\r\n";
   return(magicPositionsStatus);
  }
```

### Symbol Positions Status

Returns a pre-formatted string containing the account symbol positions' status, which can be printed to the log or displayed in the chart comments. The function accepts three arguments or parameters. A string _symbol_, unsigned long _magicNumber_, and a bool _formatForComment_. If the _formatForComment_ boolean value is _true_, the function formats the data for display in the chart window. If it is _false_, the data is formatted for display in the Expert Advisor's logs tab.

```
string SymbolPositionsStatus(string symbol, ulong magicNumber, bool formatForComment) export
  {
   GetPositionsData(symbol, magicNumber); //-- Update the position status variables before we display their data
   string spacer = "";
   if(formatForComment) //-- Add some formating space for the chart comment string
     {
      spacer = "                                        ";
     }
   string symbolPositionsStatus = "\r\n" + spacer + "|---------------------------------------------------------------------------\r\n";
   symbolPositionsStatus += spacer + "| " + symbol + " - SYMBOL POSTIONS STATUS \r\n";
   symbolPositionsStatus += spacer + "|---------------------------------------------------------------------------\r\n";
   symbolPositionsStatus += spacer + "|     Total Open:   " + (string)symbolPositionsTotal + "\r\n";
   symbolPositionsStatus += spacer + "|     Total Volume: " + (string)symbolPositionsVolumeTotal + "\r\n";
   symbolPositionsStatus += spacer + "|     Total Profit: " +
   (string)(NormalizeDouble(symbolPositionsProfit, 2)) + accountCurrency + "\r\n";
   symbolPositionsStatus += spacer + "|------------------------------------------------------------------\r\n";
   symbolPositionsStatus += spacer + "| BUY POSITIONS: \r\n";
   symbolPositionsStatus += spacer + "|     Total Open:   " + (string)symbolBuyPositionsTotal + "\r\n";
   symbolPositionsStatus += spacer + "|     Total Volume: " + (string)symbolBuyPositionsVolumeTotal + "\r\n";
   symbolPositionsStatus += spacer + "|     Total Profit: " + (string)(NormalizeDouble(symbolBuyPositionsProfit, 2)) +
   accountCurrency + "\r\n";
   symbolPositionsStatus += spacer + "|------------------------------------------------------------------\r\n";
   symbolPositionsStatus += spacer + "| SELL POSITIONS: \r\n";
   symbolPositionsStatus += spacer + "|     Total Open:   " + (string)symbolSellPositionsTotal + "\r\n";
   symbolPositionsStatus += spacer + "|     Total Volume: " + (string)symbolSellPositionsVolumeTotal + "\r\n";
   symbolPositionsStatus += spacer + "|     Total Profit: " + (string)(NormalizeDouble(symbolSellPositionsProfit, 2)) +
   accountCurrency + "\r\n";
   symbolPositionsStatus += spacer + "|---------------------------------------------------------------------------\r\n";
   symbolPositionsStatus += spacer + "\r\n";
   return(symbolPositionsStatus);
  }
```

### How to Import and Implement our Positions Management EX5 Library

We have developed a comprehensive Positions Management EX5 library, containing all essential functions for position operations, status extraction, and display modules. It is now time to document and explain how to effectively import and utilize this library in any MQL5 project.

To streamline the implementation process, let us begin by outlining all the functions or modules within the Positions Management library, along with some real-world code example use cases. This will provide the library users with a quick overview of the components included in the _PositionsManager.ex5_ binary file.

### Positions Manager EX5 Library Documentation

| Function Prototype Description | Description | Example Use Case |
| --- | --- | --- |
| ```<br>bool ErrorAdvisor(<br>   string callingFunc, <br>   string symbol, <br>   int tradeServerErrorCode<br>);<br>``` | Manages trade server and runtime errors when handling positions and orders. Returns _true_ if the error is recoverable and the trade request can be resent, _false_ if the error is critical and the request sending should stop. | ```<br>ResetLastError(); //-- Reset and clear the last error<br>//--------------------------------------------------------------<br>//-- Insert code to send the order request to the trade server<br>//--------------------------------------------------------------<br>string symbol = _Symbol; //Symbol being traded<br>int retcode = tradeResult.retcode;//Trade Request Structure (MqlTradeRequest)<br>if(!ErrorAdvisor(__FUNCTION__, symbol, retcode)<br>  {<br>//Critical error found<br>//Order can not be executed. Exit function or log this error<br>  }<br>``` |
| ```<br>bool TradingIsAllowed();<br>``` | Verifies if the expert advisor has been given permission to execute trades by the user, trade server, and broker. | ```<br>if(!TradingIsAllowed())<br>  {<br>   //--- algo trading is disabled, exit function<br>   return(false);<br>  }<br>``` |
| ```<br>bool OpenBuyPosition(<br>   ulong magicNumber,<br>   string symbol,<br>   double lotSize,<br>   int sl, int tp,<br>   string positionComment<br>);<br>``` | Opens a new buy position matching the specified parameters. | ```<br>ulong magicNo = 123;<br>string symbol = _Symbol;<br>double lotSize = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);<br>int sl = 500; //-- pips<br>int tp = 1000; //-- pips<br>string comment = "Buy position";<br>OpenBuyPosition(magicNo, symbol, lotSize, sl, tp, comment);<br>``` |
| ```<br>bool OpenSellPosition(<br>   ulong magicNumber,<br>   string symbol,<br>   double lotSize,<br>   int sl,<br>   int tp,<br>   string positionComment<br>);<br>``` | Opens a new sell position matching the specified parameters. | ```<br>ulong magicNo = 123;<br>string symbol = _Symbol;<br>double lotSize = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);<br>int sl = 500; //-- pips<br>int tp = 1000; //-- pips<br>string comment = "Sell position";<br>OpenSellPosition(magicNo, symbol, lotSize, sl, tp, comment);<br>``` |
| ```<br>bool SetSlTpByTicket(<br>   ulong positionTicket,<br>   int sl,<br>   int tp<br>);<br>``` | Sets the stop loss for a position matching the specified ticket. | ```<br>int sl = 500, int tp = 1000; //-- pips<br>int totalOpenPostions = PositionsTotal();<br>for(int x = 0; x < totalOpenPostions; x++)<br>{<br>   ulong positionTicket = PositionGetTicket(x);<br>   if(positionTicket > 0)<br>   {<br>      SetSlTpByTicket(positionTicket, sl, tp);<br>   }   <br>}<br>``` |
| ```<br>bool ClosePositionByTicket(<br>   ulong positionTicket<br>);<br>``` | Closes a position matching the specified ticket. | ```<br>//-- Example to close all open positions<br>for(int x = 0; x < totalOpenPostions; x++)<br>{<br>   ulong positionTicket = PositionGetTicket(x);<br>   if(positionTicket > 0)<br>   {<br>      ClosePositionByTicket(positionTicket);<br>   }   <br>}<br>``` |
| ```<br>bool SetTrailingStopLoss(<br>   ulong positionTicket,<br>   int trailingStopLoss<br>);<br>``` | Sets a trailing stop loss for a position matching the specified ticket. This function should be executed on every tick in the _OnTick()_ function to update the trailing stop loss in real-time. | ```<br>//-- Execute this on every tick<br>//-- Example to set 500 pips trailing stop loss for all positions<br>int trailingStopLoss = 500; //-- 500 pips trailing stop loss<br>for(int x = 0; x < totalOpenPostions; x++)<br>{<br>   ulong positionTicket = PositionGetTicket(x);<br>   if(positionTicket > 0)<br>   {<br>      SetTrailingStopLoss(positionTicket, trailingStopLoss);<br>   }   <br>}<br>``` |
| ```<br>bool CloseAllPositions(<br>   string symbol, <br>   ulong magicNumber<br>);<br>``` | Versatile function that closes all open positions matching the specified parameters. | ```<br>//Close all positions<br>CloseAllPositions("", 0);<br>//Only close all positions matching a magic number value of 1<br>CloseAllPositions("", 1);<br>//Only close all current symbol positions<br>CloseAllPositions(_Symbol, 0);<br>``` |
| ```<br>bool CloseAllPositions();<br>``` | Closes all open positions. | ```<br>//Close all open positions in the account<br>CloseAllPositions();<br>``` |
| ```<br>bool CloseAllBuyPositions(<br>   string symbol, <br>   ulong magicNumber<br>);<br>``` | Closes all buy positions matching the specified parameters. | ```<br>//Close all buy positions for the current symbol<br>CloseAllBuyPositions(_Symbol, 0);<br>//Close all buy positions matching magic number 1 for all symbols<br>CloseAllBuyPositions("", 1);<br>//Close all buy positions in the account<br>CloseAllBuyPositions("", 0);<br>``` |
| ```<br>bool CloseAllSellPositions(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Closes all sell positions matching the specified parameters. | ```<br>//Close all sell positions for the current symbol<br>CloseAllSellPositions(_Symbol, 0);<br>//Close all sell positions matching magic number 1 for all symbols<br>CloseAllSellPositions("", 1);<br>//Close all sell positions in the account<br>CloseAllSellPositions("", 0);<br>``` |
| ```<br>bool CloseAllMagicPositions(<br>   ulong magicNumber<br>);<br>``` | Closes all positions matching the specified magic number. | ```<br>//Close all positions matching magic number 1<br>CloseAllMagicPositions(1);<br>//Close all positions in the account<br>CloseAllMagicPositions(0);<br>``` |
| ```<br>bool CloseAllProfitablePositions(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Closes all profitable positions matching the specified parameters. | ```<br>//Close all profitable positions for the current symbol<br>CloseAllProfitablePositions(_Symbol, 0);<br>//Close all profitable positions matching magic number 1 for all symbols<br>CloseAllProfitablePositions("", 1);<br>//Close all profitable positions in the account<br>CloseAllProfitablePositions("", 0);<br>``` |
| ```<br>bool CloseAllProfitableBuyPositions(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Closes all profitable buy positions matching the specified parameters. | ```<br>//Close all profitable buy positions for the current symbol<br>CloseAllProfitableBuyPositions(_Symbol, 0);<br>//Close all profitable buy positions matching magic number 1 for all symbols<br>CloseAllProfitableBuyPositions("", 1);<br>//Close all profitable buy positions in the account<br>CloseAllProfitableBuyPositions("", 0);<br>``` |
| ```<br>bool CloseAllProfitableSellPositions(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Closes all profitable sell positions matching the specified parameters. | ```<br>//Close all profitable sell positions for the current symbol<br>CloseAllProfitableSellPositions(_Symbol, 0);<br>//Close all profitable sell positions matching magic number 1 for all symbols<br>CloseAllProfitableSellPositions("", 1);<br>//Close all profitable sell positions in the account<br>CloseAllProfitableSellPositions("", 0);<br>``` |
| ```<br>bool CloseAllLossPositions(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Closes all loss positions matching the specified parameters. | ```<br>//Close all loss positions for the current symbol<br>CloseAllLossPositions(_Symbol, 0);<br>//Close all loss positions matching magic number 1 for all symbols<br>CloseAllLossPositions("", 1);<br>//Close all loss positions in the account<br>CloseAllLossPositions("", 0);<br>``` |
| ```<br>bool CloseAllLossBuyPositions(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Closes all loss buy positions matching the specified parameters. | ```<br>//Close all loss buy positions for the current symbol<br>CloseAllLossBuyPositions(_Symbol, 0);<br>//Close all loss buy positions matching magic number 1 for all symbols<br>CloseAllLossBuyPositions("", 1);<br>//Close all loss buy positions in the account<br>CloseAllLossBuyPositions("", 0);<br>``` |
| ```<br>bool CloseAllLossSellPositions(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Closes all loss sell positions matching the specified parameters. | ```<br>//Close all loss sell positions for the current symbol<br>CloseAllLossSellPositions(_Symbol, 0);<br>//Close all loss sell positions matching magic number 1 for all symbols<br>CloseAllLossSellPositions("", 1);<br>//Close all loss sell positions in the account<br>CloseAllLossSellPositions("", 0);<br>``` |
| ```<br>int BuyPositionsTotal();<br>``` | Returns the number of open buy positions. | ```<br>//Get the total number of open buy positions in the account<br>BuyPositionsTotal();<br>``` |
| ```<br>int SellPositionsTotal();<br>``` | Returns the number of open sell positions. | ```<br>//Get the total number of open sell positions in the account<br>SellPositionsTotal();<br>``` |
| ```<br>double PositionsTotalVolume();<br>``` | Returns the total volume of all the open positions. | ```<br>//Get the total volume of all open positions in the account<br>PositionsTotalVolume();<br>``` |
| ```<br>double BuyPositionsTotalVolume();<br>``` | Returns the total volume of all the buy open positions. | ```<br>//Get the total volume of all open buy positions in the account<br>BuyPositionsTotalVolume();<br>``` |
| ```<br>double SellPositionsTotalVolume();<br>``` | Returns the total volume of all the sell open positions. | ```<br>//Get the total volume of all open sell positions in the account<br>SellPositionsTotalVolume();<br>``` |
| ```<br>double BuyPositionsProfit();<br>``` | Returns the total profit of all the buy open positions. | ```<br>//Get the total profit of all open buy positions in the account<br>BuyPositionsProfit();<br>``` |
| ```<br>double SellPositionsProfit();<br>``` | Returns the total profit of all the sell open positions. | ```<br>//Get the total profit of all open sell positions in the account<br>SellPositionsProfit();<br>``` |
| ```<br>int MagicPositionsTotal(<br>   ulong magicNumber<br>);<br>``` | Returns the number of open positions matching the specified magic number. | ```<br>//Get the total number of open positions matching magic number 1<br>MagicPositionsTotal(1);<br>``` |
| ```<br>int MagicBuyPositionsTotal(<br>   ulong magicNumber<br>);<br>``` | Returns the number of open buy positions matching the specified magic number. | ```<br>//Get the total number of open buy positions matching magic number 1<br>MagicBuyPositionsTotal(1);<br>``` |
| ```<br>int MagicSellPositionsTotal(<br>   ulong magicNumber<br>);<br>``` | Returns the number of open sell positions matching the specified magic number. | ```<br>//Get the total number of open sell positions matching magic number 1<br>MagicSellPositionsTotal(1);<br>``` |
| ```<br>double MagicPositionsTotalVolume(<br>   ulong magicNumber<br>);<br>``` | Returns the total volume of all the open positions matching the specified magic number. | ```<br>//Get the total volume of open positions matching magic number 1<br>MagicPositionsTotalVolume(1);<br>``` |
| ```<br>double MagicBuyPositionsTotalVolume(<br>   ulong magicNumber<br>);<br>``` | Returns the total volume of all the open buy positions matching the specified magic number. | ```<br>//Get the total volume of open buy positions matching magic number 1<br>MagicBuyPositionsTotalVolume(1);<br>``` |
| ```<br>double MagicSellPositionsTotalVolume(<br>   ulong magicNumber<br>);<br>``` | Returns the total volume of all the open sell positions matching the specified magic number. | ```<br>//Get the total volume of open sell positions matching magic number 1<br>MagicSellPositionsTotalVolume(1);<br>``` |
| ```<br>double MagicPositionsProfit(<br>   ulong magicNumber<br>);<br>``` | Returns the total profit of all the open positions matching the specified magic number. | ```<br>//Get the total profit of open positions matching magic number 1<br>MagicPositionsProfit(1);<br>``` |
| ```<br>double MagicBuyPositionsProfit(<br>   ulong magicNumber<br>);<br>``` | Returns the total profit of all the open buy positions matching the specified magic number. | ```<br>//Get the total profit of open buy positions matching magic number 1<br>MagicBuyPositionsProfit(1);<br>``` |
| ```<br>double MagicSellPositionsProfit(<br>   ulong magicNumber<br>);<br>``` | Returns the total profit of all the open sell positions matching the specified magic number. | ```<br>//Get total profit of sell positions matching magic number 1<br>MagicSellPositionsProfit(1);<br>``` |
| ```<br>int SymbolPositionsTotal(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Returns the total number of all the open positions matching the specified symbol and magic number. | ```<br>//Get total number of positions matching symbol and magic number 1<br>MagicPositionsTotal(_Symbol, 1);<br>``` |
| ```<br>int SymbolBuyPositionsTotal(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Returns the total number of all the open buy positions matching the specified symbol and magic number. | ```<br>//Get total number of buy positions matching symbol and magic number 1<br>SymbolBuyPositionsTotal(_Symbol, 1);<br>``` |
| ```<br>int SymbolSellPositionsTotal(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Returns the total number of all the open sell positions matching the specified symbol and magic number. | ```<br>//Get total number of sell positions matching symbol and magic number 1<br>SymbolSellPositionsTotal(_Symbol, 1);<br>``` |
| ```<br>double SymbolPositionsTotalVolume(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Returns the total volume of all the open positions matching the specified symbol and magic number. | ```<br>//Get the volume of positions matching symbol and magic number 1<br>SymbolPositionsTotalVolume(_Symbol, 1);<br>``` |
| ```<br>double SymbolBuyPositionsTotalVolume(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Returns the total volume of all the open buy positions matching the specified symbol and magic number. | ```<br>//Get the volume of buy positions matching symbol and magic number 1<br>SymbolBuyPositionsTotalVolume(_Symbol, 1);<br>``` |
| ```<br>double SymbolSellPositionsTotalVolume(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Returns the total volume of all the open sell positions matching the specified symbol and magic number. | ```<br>//Get the volume of sell positions matching symbol and magic number 1<br>SymbolSellPositionsTotalVolume(_Symbol, 1);<br>``` |
| ```<br>double SymbolPositionsProfit(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Returns the total profit of all the open positions matching the specified symbol and magic number. | ```<br>//Get the profit of all positions matching symbol and magic number 1<br>SymbolPositionsProfit(_Symbol, 1);<br>``` |
| ```<br>double SymbolBuyPositionsProfit(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Returns the total profit of all the open buy positions matching the specified symbol and magic number. | ```<br>//Get the profit of all buy positions matching symbol and magic number 1<br>SymbolBuyPositionsProfit(_Symbol, 1);<br>``` |
| ```<br>double SymbolSellPositionsProfit(<br>   string symbol,<br>   ulong magicNumber<br>);<br>``` | Returns the total profit of all the open sell positions matching the specified symbol and magic number. | ```<br>//Get the profit of all sell positions matching symbol and magic number 1<br>SymbolSellPositionsProfit(_Symbol, 1);<br>``` |
| ```<br>string AccountPositionsStatus(<br>   bool formatForComment<br>);<br>``` | Prints a string formatted status of all open positions on the symbol chart or Experts tab in MetaTrader5. | ```<br>//Print the status of all open positions formatted for the chart comments<br>AccountPositionsStatus(true);<br>//Print the status of all open positions formatted for the Experts tab<br>AccountPositionsStatus(false);<br>``` |
| ```<br>string MagicPositionsStatus(<br>   ulong magicNumber,<br>   bool formatForComment<br>);<br>``` | Prints a string formatted status of all open positions matching the specified magic number on the symbol chart or Experts tab in MetaTrader5. | ```<br>//Print the status of all open positions matching<br>//the magic number 1 formatted for the chart comments<br>MagicPositionsStatus(1, true);<br>//Print the status of all open positions matching<br>//the magic number 1 formatted for the Experts tab<br>MagicPositionsStatus(1, false);<br>``` |
| ```<br>string SymbolPositionsStatus(<br>   string symbol,<br>   ulong magicNumber,<br>   bool formatForComment<br>);<br>``` | Prints a string formatted status of all open positions matching the specified symbol and magic number on the symbol chart or Experts tab in MetaTrader5. | ```<br>//Print the status of all open positions matching<br>//the symbol and magic number 1 formatted for the chart comments<br>SymbolPositionsStatus(_Symbol, 1, true);<br>//Print the status of all open positions matching<br>//the symbol and magic number 1 formatted for the Experts tab<br>SymbolPositionsStatus(_Symbol, 1, false);<br>``` |

Integrating the library in your MQL5 projects is straightforward. Follow these two steps to import _PositionsManager.ex5_ into your MQL5 code:

- **Step 1: Copy the library executable file ( _PositionsManager.ex5_)**

> Place the _PositionsManager.ex5_ file in the _MQL5/Libraries/Toolkit_ folder. Ensure you have downloaded and copied this file to the specified location if it's not already present. A copy of _PositionsManager.ex5_ is attached at the end of this article for your convenience.

- **Step 2: Import the function prototype descriptions**

> Add the import directives of the Positions Manager library, and it's function prototype descriptions in the header section of your source code file. Use the code segment below to efficiently import all functions or modules from the _PositionsManager.ex5_ library. I have also created a blank Expert Advisor template ( _PositionsManager\_Imports\_Template.mq5_) that includes the code segment below. You are welcome to selectively comment out or remove any function descriptions not required for your project. The _PositionsManager\_Imports\_Template.mq5_ file is also attached at the end of this article.
>
> ```
> //+-------------------------------------------------------------------------------------+
> //| PositionsManager.ex5 imports template                                               |
> //+-------------------------------------------------------------------------------------+
> #import "Toolkit/PositionsManager.ex5" //-- Opening import directive
> //-- Function descriptions for the imported function prototypes
>
> //-- Error Handling and Permission Status Functions
> bool   ErrorAdvisor(string callingFunc, string symbol, int tradeServerErrorCode);
> bool   TradingIsAllowed();
>
> //-- Position Execution and Modification Functions
> bool   OpenBuyPosition(ulong magicNumber, string symbol, double lotSize, int sl, int tp, string positionComment);
> bool   OpenSellPosition(ulong magicNumber, string symbol, double lotSize, int sl, int tp, string positionComment);
> bool   SetSlTpByTicket(ulong positionTicket, int sl, int tp);
> bool   ClosePositionByTicket(ulong positionTicket);
> bool   SetTrailingStopLoss(ulong positionTicket, int trailingStopLoss);
> bool   CloseAllPositions(string symbol, ulong magicNumber);
> bool   CloseAllPositions();
> bool   CloseAllBuyPositions(string symbol, ulong magicNumber);
> bool   CloseAllSellPositions(string symbol, ulong magicNumber);
> bool   CloseAllMagicPositions(ulong magicNumber);
> bool   CloseAllProfitablePositions(string symbol, ulong magicNumber);
> bool   CloseAllProfitableBuyPositions(string symbol, ulong magicNumber);
> bool   CloseAllProfitableSellPositions(string symbol, ulong magicNumber);
> bool   CloseAllLossPositions(string symbol, ulong magicNumber);
> bool   CloseAllLossBuyPositions(string symbol, ulong magicNumber);
> bool   CloseAllLossSellPositions(string symbol, ulong magicNumber);
>
> //-- Position Status Monitoring Functions
> int    BuyPositionsTotal();
> int    SellPositionsTotal();
> double PositionsTotalVolume();
> double BuyPositionsTotalVolume();
> double SellPositionsTotalVolume();
> double BuyPositionsProfit();
> double SellPositionsProfit();
>
> //-- Positions Filtered By Magic Number Status Monitoring Functions
> int    MagicPositionsTotal(ulong magicNumber);
> int    MagicBuyPositionsTotal(ulong magicNumber);
> int    MagicSellPositionsTotal(ulong magicNumber);
> double MagicPositionsTotalVolume(ulong magicNumber);
> double MagicBuyPositionsTotalVolume(ulong magicNumber);
> double MagicSellPositionsTotalVolume(ulong magicNumber);
> double MagicPositionsProfit(ulong magicNumber);
> double MagicBuyPositionsProfit(ulong magicNumber);
> double MagicSellPositionsProfit(ulong magicNumber);
>
> //-- Positions Filtered By Symbol and/or Magic Number Status Monitoring Functions
> int    SymbolPositionsTotal(string symbol, ulong magicNumber);
> int    SymbolBuyPositionsTotal(string symbol, ulong magicNumber);
> int    SymbolSellPositionsTotal(string symbol, ulong magicNumber);
> double SymbolPositionsTotalVolume(string symbol, ulong magicNumber);
> double SymbolBuyPositionsTotalVolume(string symbol, ulong magicNumber);
> double SymbolSellPositionsTotalVolume(string symbol, ulong magicNumber);
> double SymbolPositionsProfit(string symbol, ulong magicNumber);
> double SymbolBuyPositionsProfit(string symbol, ulong magicNumber);
> double SymbolSellPositionsProfit(string symbol, ulong magicNumber);
>
> //-- Log and Data Display Functions
> string AccountPositionsStatus(bool formatForComment);
> string MagicPositionsStatus(ulong magicNumber, bool formatForComment);
> string SymbolPositionsStatus(string symbol, ulong magicNumber, bool formatForComment);
>
> #import //--- Closing import directive
> ```

With the library imported, you can now effortlessly open, close, modify, or retrieve position status data using simple function calls. To illustrate this, let us create three basic Expert Advisors in the following sections.

### Developing A Dual VIDyA Trailing Stop Expert Advisor Powered by The Positions Manager EX5 Library

In this section, we will develop a trailing stop trading strategy Expert Advisor based on the Variable Index Dynamic Average ( _VIDyA_) Technical Indicator as a practical example of implementing the Positions Manager EX5 library in a real-world trading application.

The VIDyA Trailing Stop strategy will use a pair of Variable Index Dynamic Average Technical Indicators to generate buy and sell signals. Since the VIDyA indicator is displayed as a line on the chart, we will use a line crossover strategy to signal a new trade entry. For a crossover strategy, the indicator pairs must have different settings. The first VIDyA indicator, with lower input values, will be called " _fast VIDyA_" as it will react and generate signals faster. The second, with higher input values, will be called " _slow VIDyA_" as it responds to price changes more slowly. For a buy entry, the " _fast VIDyA_" line must be above the " _slow VIDyA_" line, and for a sell entry, the " _fast VIDyA_" line must be below the " _slow VIDyA_" line.

![Dual VIDyA Strategy](https://c.mql5.com/2/85/Dual_VIDyA_Strategy.png)

Create a new Expert Advisor using the _MetaEditor IDE_ new file _MQL Wizard_ and call it " _DualVidyaTrader.mq5_". Since our Expert Advisor will utilize the _PositionsManager.ex5_ library, the first step is to import and insert the library function prototype descriptions as described earlier. Place the library import code segment below the _#property_ directives. Because the library contains many functions, we will not be importing or using all of them; only import the function prototypes listed in the code below.

```
//--- Import the PositionsManager EX5 Library
#import "Toolkit/PositionsManager.ex5" //-- Open the ex5 import directive
//-- Prototype function descriptions of the EX5 PositionsManager library
bool   OpenBuyPosition(ulong magicNumber, string symbol, double lotSize, int sl, int tp, string positionComment);
bool   OpenSellPosition(ulong magicNumber, string symbol, double lotSize, int sl, int tp, string positionComment);
bool   SetTrailingStopLoss(ulong positionTicket, int trailingStopLoss);
int    MagicPositionsTotal(ulong magicNumber);
int    MagicBuyPositionsTotal(ulong magicNumber);
int    MagicSellPositionsTotal(ulong magicNumber);
bool   CloseAllProfitableBuyPositions(string symbol, ulong magicNumber);
bool   CloseAllProfitableSellPositions(string symbol, ulong magicNumber);
string MagicPositionsStatus(ulong magicNumber, bool formatForComment);
#import //-- Close the ex5 import directive
```

After the _PositionsManager.ex5_ library import code segment, insert the user input global variables as shown.

```
input group ""
input ulong magicNo = 1234;
input ENUM_TIMEFRAMES timeframe = PERIOD_H1;
input ENUM_APPLIED_PRICE  appliedPrice = PRICE_CLOSE; // Applied VIDyA Price

//-- Fast Vidya user inputs
input group "-- FAST VIDyA INPUTS"
input int fast_cmoPeriod = 5; // Fast Chande Momentum Period
input int fast_maPeriod = 10; // Fast MA Smoothing Period
input int fast_emaShift = 0; // Fast Horizontal Shift

//-- Slow Vidya user inputs
input group "-- SLOW VIDyA INPUTS"
input int slow_cmoPeriod = 9; //  Slow Chande Momentum Period
input int slow_maPeriod = 12; // Slow MA Smoothing Period
input int slow_emaShift = 0; // Slow Horizontal Shift

input group "-- PROFIT MANAGEMENT"
input bool liquidateProfitOnCrossover = false; // Liquidate Profit On VIDyA Signal
input bool enableTrailingStops = true; // Use Trailing Stop Losses
```

Let us create more global variables to store the stop loss, take profit, trailing stop loss, and lot or volume values.

```
//-- Get and save the SL, trailingSL and TP values from the spread
int spreadMultiForSl = 1000;
int spreadMultiForTrailingSl = 300;
int spreadMultiForTp = 1000;
int sl = int(SymbolInfoInteger(_Symbol, SYMBOL_SPREAD)) * spreadMultiForSl;
int trailingSl = int(SymbolInfoInteger(_Symbol, SYMBOL_SPREAD)) * spreadMultiForTrailingSl;
int tp = int(SymbolInfoInteger(_Symbol, SYMBOL_SPREAD)) * spreadMultiForTp;

//-- Set the lot or volume to the symbol allowed min value
double lotSize = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
```

Create the VIDyA technical indicator variables in the global scope to ensure they are also accessible from any part of our expert advisor code.

```
//-- Vidya indicator variables
double fastVidya[], slowVidya[];
int fastVidyaHandle, slowVidyaHandle;
bool buyOk, sellOk, vidyaBuy, vidyaSell;
string vidyaTrend;
```

After the global variables section, create the Expert Advisor initialization function that will be called from the standard MQL5 _OnInit()_ event handling function, which is first executed during the Expert Advisor's start-up or initialization. Name this function _GetInit()_. We will perform all our data initializations in this function, including the VIDyA indicator initializations and handling start-up processing.

```
int GetInit()
  {
   int returnVal = 1;

//-- Helps to regulate and prevent openning multiple trades on a single signal trigger
   buyOk = true;
   sellOk = true;

//-- Create the fast iVIDyA indicator handle
   fastVidyaHandle = iVIDyA(_Symbol, timeframe, fast_cmoPeriod, fast_maPeriod, fast_emaShift, appliedPrice);
   if(fastVidyaHandle < 0)
     {
      Print("Error creating fastVidyaHandle = ", INVALID_HANDLE);
      Print("Handle creation: Runtime error = ", GetLastError());
      //-- Close the EA if the handle is not properly loaded
      return(-1);
     }
   ArraySetAsSeries(fastVidya, true); //-- set the vidya array to series access

//-- Create the slow iVIDyA indicator handle
   slowVidyaHandle = iVIDyA(_Symbol, timeframe, slow_cmoPeriod, slow_maPeriod, slow_emaShift, appliedPrice);
   if(slowVidyaHandle < 0)
     {
      Print("Error creating vidyaHandle = ", INVALID_HANDLE);
      Print("Handle creation: Runtime error = ", GetLastError());
      //-- Close the EA if the handle is not properly loaded
      return(-1);
     }
   ArraySetAsSeries(slowVidya, true); //-- set the vidya array to series access

   return(returnVal);
  }
```

After the initialization function, create the de-initialization function and call it _GetDeinit()_. This function will be called within the standard MQL5 _OnDeinit()_ event handling function to perform a system-wide cleanup of all the resources used by our Expert Advisor. This includes releasing the VIDyA indicator, freeing all resources tied to the indicator handle arrays, and removing any chart comments or objects.

```
void GetDeinit()  //-- De-initialize the robot on shutdown and clean everything up
  {
//-- Delete the vidya handles and de-allocate the memory spaces occupied
   IndicatorRelease(fastVidyaHandle);
   ArrayFree(fastVidya);
   IndicatorRelease(slowVidyaHandle);
   ArrayFree(slowVidya);

//-- Delete and clear all chart displayed messages
   Comment("");
  }
```

Next, we need to create a custom function to detect and retrieve the trade signals generated by our VIDyA indicator pair. This function will be called _GetVidya()_. It will be executed and updated every time there is a new incoming tick to ensure that the signal generation is accurate and real-time.

```
void GetVidya()
  {
//-- Get vidya line directions
   if(CopyBuffer(fastVidyaHandle, 0, 0, 100, fastVidya) <= 0 || CopyBuffer(slowVidyaHandle, 0, 0, 100, slowVidya) <= 0)
     {
      return;
     }

//-- Reset vidya status variables
   vidyaBuy = false;
   vidyaSell = false;
   vidyaTrend = "FLAT";

//-- Scan for vidya crossover buy signal
   if(fastVidya[1] > slowVidya[1])
     {
      //-- Save the vidya signal
      vidyaTrend = "BUY/LONG";
      vidyaBuy = true;
      vidyaSell = false;
     }

//-- Scan for vidya crossover sell signal
   if(fastVidya[1] < slowVidya[1])
     {
      //-- Save the vidya signal
      vidyaTrend = "SELL/SHORT";
      vidyaSell = true;
      vidyaBuy = false;
     }
  }
```

Now that we have created the function to get and update the VIDyA signal, let us create another custom function that will be executed on every new tick to scan and open new positions based on the current VIDyA signal. This function will be called _ScanForTradeOpportunities()_. We will call and execute the position opening and status prototype functions imported from our _PositionsManager.ex5_ library in this function.

```
void ScanForTradeOpportunities()
  {
//-- Get the VIDyA signal
   GetVidya();

   if(MagicPositionsTotal(magicNo) == 0)
     {
      buyOk = true;
      sellOk = true;
     }

//-- Check for a buy entry when a VIDyA buy signal is found
   if(buyOk && vidyaBuy) //-- Open a new buy position
     {
      if(OpenBuyPosition(magicNo, _Symbol, lotSize, sl, tp, "Vidya_BUY: " + IntegerToString(MagicBuyPositionsTotal(magicNo) + 1)))
        {
         buyOk = false;
         sellOk = true;
        }
      //-- Market has a strong buy trend, close all profitable sell positions
      if(liquidateProfitOnCrossover)
        {
         CloseAllProfitableSellPositions(_Symbol, magicNo);
        }
     }

//-- Check for a sell entry when a VIDyA sell signal is found
   if(sellOk && vidyaSell) //-- Open a new sell position
     {
      if(OpenSellPosition(magicNo, _Symbol, lotSize, sl, tp, "Vidya_SELL: " + IntegerToString(MagicSellPositionsTotal(magicNo) + 1)))
        {
         sellOk = false;
         buyOk = true;
        }
      //-- Market has a strong sell trend, close all profitable buy positions
      if(liquidateProfitOnCrossover)
        {
         CloseAllProfitableBuyPositions(_Symbol, magicNo);
        }
     }
  }
```

We also need to check and set trailing stop losses on all open positions. This will be simple as our _PositionsManager.ex5_ library contains a trailing stop loss prototype function to help us accomplish this. Let us create a new function called _CheckAndSetTrailingSl()_ to scan all the open positions and retrieve their tickets for use as parameters in the imported _SetTrailingStopLoss()_ prototype function.

```
void CheckAndSetTrailingSl()
  {
   int totalOpenPostions = PositionsTotal();
   for(int x = 0; x < totalOpenPostions; x++)
     {
      //--- Get position properties
      ulong  positionTicket = PositionGetTicket(x); //-- Get ticket to select the position
      string selectedSymbol = PositionGetString(POSITION_SYMBOL);
      ulong positionMagicNo = PositionGetInteger(POSITION_MAGIC);
      int positionType = int(PositionGetInteger(POSITION_TYPE));

      //-- modify only the positions we have opened with this EA (magic number)
      if(selectedSymbol != _Symbol && positionMagicNo != magicNo)
        {
         continue;
        }
      //-- Only set the trailing stop loss when the market trend is in the opposing direction of the position type
      if((positionType == POSITION_TYPE_BUY && vidyaBuy) || (positionType == POSITION_TYPE_SELL && vidyaSell))
        {
         continue;
        }
      //--- set the trailing stop loss
      SetTrailingStopLoss(positionTicket, trailingSl); //-- call the imported function from our ex5 library
     }
  }
```

Now that we have created all the important modules for our Expert Advisor, let us bring them together in the MQL5 _OnTick()_ event-handling function that is executed on every new tick. Arrange them in the order specified below to ensure they systematically execute our trading system in the correct sequence.

```
void OnTick()
  {
//-- Scan and open new positions based on the vidya signal
   ScanForTradeOpportunities();

//-- Check and set the trailing stop
   if(enableTrailingStops)
     {
      CheckAndSetTrailingSl();
     }

//-- Display the vidya trend and positions status for the EA's magicNo
   Comment(
      "\nvidyaTrend: ", vidyaTrend,
      MagicPositionsStatus(magicNo, true)
   );
  }
```

Finally, don't forget to place the initialization and de-initialization functions in their respective standard MQL5 event-handling functions.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   if(GetInit() <= 0)
     {
      return(INIT_FAILED);
     }
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   GetDeinit();
  }
```

You can access and download the source code file for the _DualVidyaTrader.mq5_ Expert Advisor at the end of this article. I have attached the full source code file to ensure you have all the necessary components to implement and customize the trading strategy as needed.

When you compile and load our new _DualVidyaTrader_ Expert Advisor in _MetaTrader 5_, you will notice that in the _"Dependencies"_ tab, all the library function prototypes imported from _PositionsManager.ex5_ are listed, along with the full file path where the EX5 library is saved. This ensures that all required dependencies are correctly referenced before the Expert Advisor is loaded onto the chart. If any library referencing errors, like the ones discussed earlier in the article, are encountered, they will be logged in the _Experts_ or _Journal_ tabs of the _MetaTrader 5 Toolbox_ window.

![Dual Vidya Trader Dependencies Tab](https://c.mql5.com/2/85/Dual_Vidya_Trader_Dependencies.png)

### Positions Manager Trade Panel Powered by The Positions Manager EX5 Library

In this second example, we will create a basic Expert Advisor graphical user interface (GUI) trade panel that is also powered by our _PositionsManager.ex5_ library.

![Positions Manager Panel GUI](https://c.mql5.com/2/85/Positions_Manager_Panel_GUI.png)

Create a new Expert Advisor using the _MetaEditor IDE_ new file _MQL Wizard_ and call it _PositionsManagerPanel.mq5_. Under the _#property_ directives code segment, import the _PositionsManager.ex5_ library. In the imports functions descriptions section only import the following function prototypes.

```
//+------------------------------------------------------------------+
//| EX5 PositionsManager imports                                     |
//+------------------------------------------------------------------+
#import "Toolkit/PositionsManager.ex5" //-- Open import directive
//-- Function descriptions for the imported function prototypes

//--Position Execution and Modification Functions
bool   OpenBuyPosition(ulong magicNumber, string symbol, double lotSize, int sl, int tp, string positionComment);
bool   OpenSellPosition(ulong magicNumber, string symbol, double lotSize, int sl, int tp, string positionComment);
bool   SetSlTpByTicket(ulong positionTicket, int sl, int tp);
bool   ClosePositionByTicket(ulong positionTicket);
bool   SetTrailingStopLoss(ulong positionTicket, int trailingStopLoss);
bool   CloseAllPositions(string symbol, ulong magicNumber);
bool   CloseAllBuyPositions(string symbol, ulong magicNumber);
bool   CloseAllSellPositions(string symbol, ulong magicNumber);
bool   CloseAllMagicPositions(ulong magicNumber);
bool   CloseAllProfitablePositions(string symbol, ulong magicNumber);
bool   CloseAllLossPositions(string symbol, ulong magicNumber);

//--Position Status Monitoring Functions
int    BuyPositionsTotal();
int    SellPositionsTotal();
double PositionsTotalVolume();
double BuyPositionsTotalVolume();
double SellPositionsTotalVolume();
double BuyPositionsProfit();
double SellPositionsProfit();
int    MagicPositionsTotal(ulong magicNumber);
int    MagicBuyPositionsTotal(ulong magicNumber);
int    MagicSellPositionsTotal(ulong magicNumber);
double MagicPositionsTotalVolume(ulong magicNumber);
double MagicBuyPositionsTotalVolume(ulong magicNumber);
double MagicSellPositionsTotalVolume(ulong magicNumber);
double MagicPositionsProfit(ulong magicNumber);
double MagicBuyPositionsProfit(ulong magicNumber);
double MagicSellPositionsProfit(ulong magicNumber);
int    SymbolPositionsTotal(string symbol, ulong magicNumber);
int    SymbolBuyPositionsTotal(string symbol, ulong magicNumber);
int    SymbolSellPositionsTotal(string symbol, ulong magicNumber);
double SymbolPositionsTotalVolume(string symbol, ulong magicNumber);
double SymbolBuyPositionsTotalVolume(string symbol, ulong magicNumber);
double SymbolSellPositionsTotalVolume(string symbol, ulong magicNumber);
double SymbolPositionsProfit(string symbol, ulong magicNumber);
double SymbolBuyPositionsProfit(string symbol, ulong magicNumber);
double SymbolSellPositionsProfit(string symbol, ulong magicNumber);
string AccountPositionsStatus(bool formatForComment);
string MagicPositionsStatus(ulong magicNumber, bool formatForComment);
string SymbolPositionsStatus(string symbol, ulong magicNumber, bool formatForComment);
#import //--- Close import directive
```

The _PositionsManagerPanel.mq5_ will only contain one user input (magic number).

```
//--User input variables
input ulong magicNo = 101010;
```

Next, we create the global variables to store the _volumeLot_, _sl_, and _tp_.

```
//-- Global variables
//-----------------------
//-- Get the current symbol spread and multiply it by a significant number
//-- to simulate user-input SL and TP values
double volumeLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
int sl = (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * 50;
int tp = (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * 100;
```

Let's create our first graphical object, which will serve as the background for our graphical user interface. We will use a rectangle label for this purpose. To accomplish this, we will create a custom function called _CreateRectangleLabel()_.

```
//+-----------------------------------------------------------------------+
//| CreateRectangleLabel(): Creates a rectangle label on the chart window |
//+-----------------------------------------------------------------------+
void CreateRectangleLabel()
  {
//--- Detect if we have an object named the same as our rectangle label
   if(ObjectFind(0, "mainRectangleLabel") >= 0)
     {
      //--- Delete the specified object if it is not a rectangle label
      if(ObjectGetInteger(0, "mainRectangleLabel", OBJPROP_TYPE) != OBJ_RECTANGLE_LABEL)
        {
         ObjectDelete(0, "mainRectangleLabel");
        }
     }
   else
     {
      //-- Create the mainRectangleLabel
      ObjectCreate(0, "mainRectangleLabel", OBJ_RECTANGLE_LABEL, 0, 0, 0);
     }
//--- Set up the new rectangle label properties
   ObjectSetInteger(0, "mainRectangleLabel", OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, "mainRectangleLabel", OBJPROP_XDISTANCE, 240);
   ObjectSetInteger(0, "mainRectangleLabel", OBJPROP_YDISTANCE, 2);
   ObjectSetInteger(0, "mainRectangleLabel", OBJPROP_XSIZE, 460);
   ObjectSetInteger(0, "mainRectangleLabel", OBJPROP_YSIZE, 520);
   ObjectSetInteger(0, "mainRectangleLabel", OBJPROP_BGCOLOR, clrMintCream);
   ObjectSetInteger(0, "mainRectangleLabel", OBJPROP_BACK, false);
   ObjectSetInteger(0, "mainRectangleLabel", OBJPROP_HIDDEN, true);
  }
```

We also need labels to display text in our trade panel. We will create another custom function called _CreateLabel()_ to handle this task.

```
//+---------------------------------------------------------+
//| CreateLabel(): Creates a text label on the chart window |
//+---------------------------------------------------------+
void CreateLabel(
   string labelName, int xDistance, int yDistance, int xSize, int ySize,
   string labelText, color textColor, string fontType, int fontSize
)
  {
//--- Detect if we have an object with the same name as our label
   if(ObjectFind(0, labelName) >= 0)
     {
      //--- Delete the specified object if it is not a label
      if(ObjectGetInteger(0, labelName, OBJPROP_TYPE) != OBJ_LABEL)
        {
         ObjectDelete(0, labelName);
        }
     }
   else
     {
      //-- Create the label
      ObjectCreate(0, labelName, OBJ_LABEL, 0, 0, 0);
     }
//--- Set up the new rectangle label properties
   ObjectSetInteger(0, labelName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, labelName, OBJPROP_XDISTANCE, xDistance);
   ObjectSetInteger(0, labelName, OBJPROP_YDISTANCE, yDistance);
   ObjectSetInteger(0, labelName, OBJPROP_XSIZE, xSize);
   ObjectSetInteger(0, labelName, OBJPROP_YSIZE, ySize);
   ObjectSetString(0, labelName, OBJPROP_TEXT, labelText);
   ObjectSetInteger(0, labelName, OBJPROP_COLOR, textColor);
   ObjectSetString(0, labelName, OBJPROP_FONT, fontType);
   ObjectSetInteger(0, labelName, OBJPROP_FONTSIZE, fontSize);
   ObjectSetInteger(0, labelName, OBJPROP_BACK, false);
   ObjectSetInteger(0, labelName, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, labelName, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, labelName, OBJPROP_SELECTED, false);
  }
```

Clickable or responsive buttons are needed for our trade panel to perform different trade operations. Therefore, we must create a custom function to handle the creation of buttons. Let us call this function _CreateButton()_.

```
//+------------------------------------------------------+
//| CreateButton(): Creates buttons on the chart window  |
//+------------------------------------------------------+
void CreateButton(
   string btnName, int xDistance, int yDistance, int xSize, int ySize, string btnText,
   string tooltip, color textColor, string fontType, int fontSize, color bgColor
)
  {
//--- Detect if we have an object named the same as our button
   if(ObjectFind(0, btnName) >= 0)
     {
      //--- Delete the specified object if it is not a button
      if(ObjectGetInteger(0, btnName, OBJPROP_TYPE) != OBJ_BUTTON)
        {
         ObjectDelete(0, btnName);
        }
     }
   else
     {
      //-- Create the button
      ObjectCreate(0, btnName, OBJ_BUTTON, 0, 0, 0);
     }
//--- Set up the new button properties
   ObjectSetInteger(0, btnName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, btnName, OBJPROP_XDISTANCE, xDistance);
   ObjectSetInteger(0, btnName, OBJPROP_YDISTANCE, yDistance);
   ObjectSetInteger(0, btnName, OBJPROP_XSIZE, xSize);
   ObjectSetInteger(0, btnName, OBJPROP_YSIZE, ySize);
   ObjectSetString(0, btnName, OBJPROP_TEXT, btnText);
   ObjectSetString(0, btnName, OBJPROP_TOOLTIP, tooltip);
   ObjectSetInteger(0, btnName, OBJPROP_COLOR, textColor);
   ObjectSetString(0, btnName, OBJPROP_FONT, fontType);
   ObjectSetInteger(0, btnName, OBJPROP_FONTSIZE, fontSize);
   ObjectSetInteger(0, btnName, OBJPROP_BGCOLOR, bgColor);
  }
```

We now need another custom function to load all the different chart objects and user-input graphical components created with the functions we have written above. Let us call this function _LoadChartObjects()_.

```
//+--------------------------------------------------------------------+
//| LoadChartObjects(): Create and load the buttons and chart objects  |
//| for demonstrating how the imported library functions work          |
//+--------------------------------------------------------------------+
void LoadChartObjects()
  {
//-- Create the rectangle label first
   CreateRectangleLabel();

//-- Create the heading label
   CreateLabel(
      "headingLabel", 250, 10, 440, 60,
      "PositionsManager ex5 Library Demo Trade Panel",
      clrMidnightBlue, "Calibri", 10
   );

//-- Create the second heading label
   CreateLabel(
      "headingLabel2", 250, 30, 440, 60,
      ("Trading " + _Symbol + " with Magic Number: " + (string)magicNo),
      clrBlack, "Consolas", 11
   );

//-- "BUY": Button to call the imported ex5 OpenBuyPosition() function
   CreateButton(
      "OpenBuyPositionBtn", 250, 50, 215, 35, "BUY",
      "OpenBuyPosition() Function", clrMintCream, "Arial Black", 10, clrDodgerBlue
   );

//-- "SELL": Button to call the imported ex5 OpenSellPosition() function
   CreateButton(
      "OpenSellPositionBtn", 475, 50, 215, 35, "SELL",
      "OpenSellPosition() Function", clrMintCream, "Arial Black", 10, clrCrimson
   );

//-- "SetSlTpByTicket": Button to call the imported ex5 SetSlTpByTicket() function
   CreateButton(
      "SetSlTpBtn", 250, 90, 215, 35, "SetSlTpByTicket",
      "SetSlTpByTicket() Function", clrMintCream, "Arial Black", 10, clrDarkSlateGray
   );

//-- "SetTrailingStopLoss": Button to call the imported ex5 SetTrailingStopLoss() function when clicked
   CreateButton(
      "SetTrailingStopLossBtn", 475, 90, 215, 35, "SetTrailingStopLoss",
      "SetTrailingStopLoss Function", clrMintCream, "Arial Black", 10, clrDarkSlateGray
   );

//-- "ClosePositionsByTicket": Button to call the imported ex5 ClosePositionByTicket() function
   CreateButton(
      "ClosePositionsBtn", 250, 130, 215, 35, "ClosePositionsByTicket",
      "ClosePositionByTicket() Function", clrMintCream, "Arial Black", 10, clrMaroon
   );

//-- "CloseAllSymbolPositions": Button to call the imported ex5 CloseAllSymbolPositions() function
   CreateButton(
      "CloseAllPositionsBtn", 475, 130, 215, 35, "CloseAllPositions",
      "CloseAllPositions() Function", clrMintCream, "Arial Black", 10, clrMaroon
   );

//-- "CloseAllBuySymbolPositions": Button to call the imported ex5 CloseAllBuySymbolPositions() function
   CreateButton(
      "CloseAllBuyPositionsBtn", 250, 170, 215, 35, "CloseAllBuyPositions",
      "CloseAllBuyPositions() Function", clrMintCream, "Arial Black", 10, clrBrown
   );

//-- "CloseAllSellSymbolPositions": Button to call the imported ex5 CloseAllSellSymbolPositions() function
   CreateButton(
      "CloseAllSellPositionsBtn", 475, 170, 215, 35, "CloseAllSellPositions",
      "CloseAllSellPositions() Function", clrMintCream, "Arial Black", 10, clrBrown
   );

//-- "CloseAllMagicPositions": Button to call the imported ex5 CloseAllMagicPositions() function
   CreateButton(
      "CloseAllMagicPositionsBtn", 250, 210, 440, 35, "CloseAllMagicPositions",
      "CloseAllMagicPositions() Function", clrMintCream, "Arial Black", 10, C'203,18,55'
   );

//-- "CloseAllProfitablePositions": Button to call the imported ex5 CloseAllMagicPositions() function
   CreateButton(
      "CloseAllProfitablePositionsBtn", 250, 250, 215, 35, "CloseAllProfitablePositions",
      "CloseAllProfitablePositions() Function", clrMintCream, "Arial Black", 10, clrSeaGreen
   );

//-- "CloseAllLossPositions": Button to call the imported ex5 CloseAllLossPositions() function
   CreateButton(
      "CloseAllLossPositionsBtn", 475, 250, 215, 35, "CloseAllLossPositions",
      "CloseAllLossPositions() Function", clrMintCream, "Arial Black", 10, C'179,45,0'
   );

//-- Create the bottomHeadingLabel
   CreateLabel(
      "bottomHeadingLabel", 250, 310, 440, 60,
      (_Symbol + " - (Magic Number: " + (string)magicNo + ") Positions Status"),
      clrBlack, "Calibri", 12
   );

//-- Create totalOpenPositionsLabel
   CreateLabel(
      "totalOpenPositionsLabel", 250, 340, 440, 60,
      ("  Total Open:   " + (string)MagicPositionsTotal(magicNo)),
      clrNavy, "Consolas", 11
   );

//-- Create totalPositionsVolumeLabel
   CreateLabel(
      "totalPositionsVolumeLabel", 250, 360, 440, 60,
      ("  Total Volume: " + (string)NormalizeDouble(MagicPositionsTotalVolume(magicNo), 2)),
      clrNavy, "Consolas", 11
   );

//-- Create the totalPositionsProfitLabel
   CreateLabel(
      "totalPositionsProfitLabel", 250, 380, 100, 60,
      (
         "  Total Profit: " + (string)(NormalizeDouble(MagicPositionsProfit(magicNo), 2)) +
         " " + AccountInfoString(ACCOUNT_CURRENCY)
      ),
      clrNavy, "Consolas", 11
   );

//-- Create the buyPositionsHeadingLabel
   CreateLabel(
      "buyPositionsHeadingLabel", 250, 410, 440, 60,
      ("BUY POSITIONS:"),
      clrBlack, "Calibri", 12
   );

//-- Create the totalBuyPositionsLabel
   CreateLabel(
      "totalBuyPositionsLabel", 250, 430, 440, 60,
      ("  Total Open:   " + (string)MagicBuyPositionsTotal(magicNo)),
      clrNavy, "Consolas", 11
   );

//-- Create the totalBuyPositionsVolumeLabel
   CreateLabel(
      "totalBuyPositionsVolumeLabel", 250, 450, 440, 60,
      ("  Total Volume: " + (string)NormalizeDouble(MagicBuyPositionsTotalVolume(magicNo), 2)),
      clrNavy, "Consolas", 11
   );

//-- Create the totalBuyPositionsProfitLabel
   CreateLabel(
      "totalBuyPositionsProfitLabel", 250, 470, 440, 60,
      (
         "  Total Profit: " + (string)(NormalizeDouble(MagicBuyPositionsProfit(magicNo), 2)) +
         " " + AccountInfoString(ACCOUNT_CURRENCY)
      ),
      clrNavy, "Consolas", 11
   );

//-- Create the sellPositionsHeadingLabel
   CreateLabel(
      "sellPositionsHeadingLabel", 475, 410, 440, 60,
      ("SELL POSITIONS:"),
      clrBlack, "Calibri", 12
   );

//-- Create the totalSellPositionsLabel
   CreateLabel(
      "totalSellPositionsLabel", 475, 430, 440, 60,
      ("  Total Open:   " + (string)MagicSellPositionsTotal(magicNo)),
      clrNavy, "Consolas", 11
   );

//-- Create the totalSellPositionsVolumeLabel
   CreateLabel(
      "totalSellPositionsVolumeLabel", 475, 450, 440, 60,
      ("  Total Volume: " + (string)NormalizeDouble(MagicSellPositionsTotalVolume(magicNo), 2)),
      clrNavy, "Consolas", 11
   );

//-- Create the totalSellPositionsProfitLabel
   CreateLabel(
      "totalSellPositionsProfitLabel", 475, 470, 100, 60,
      (
         "  Total Profit: " + (string)(NormalizeDouble(MagicSellPositionsProfit(magicNo), 2)) +
         " " + AccountInfoString(ACCOUNT_CURRENCY)
      ),
      clrNavy, "Consolas", 11
   );

//--- Redraw the chart to refresh it so that it loads our new chart objects
   ChartRedraw();
//---
  }
```

This next custom function will be responsible for cleaning up and deleting all the chart objects and data when the Expert Advisor is terminated or de-initialized. Let us name it _DeleteChartObjects()_. It will be placed and executed in the _OnDeinit()_ standard event handling MQL5 function.

```
//+------------------------------------------------------------------------------+
//| DeleteChartObjects(): Delete all the chart objects when the EA is terminated |
//| on De-initialization                                                         |
//+------------------------------------------------------------------------------+
void DeleteChartObjects()
  {
//---
//--- Clean up and delete all the buttons or graphical objects
   ObjectDelete(0, "OpenBuyPositionBtn");
   ObjectDelete(0, "OpenSellPositionBtn");
   ObjectDelete(0, "SetSlTpBtn");
   ObjectDelete(0, "SetTrailingStopLossBtn");
   ObjectDelete(0, "ClosePositionsBtn");
   ObjectDelete(0, "CloseAllPositionsBtn");
   ObjectDelete(0, "CloseAllBuyPositionsBtn");
   ObjectDelete(0, "CloseAllSellPositionsBtn");
   ObjectDelete(0, "CloseAllMagicPositionsBtn");
   ObjectDelete(0, "CloseAllProfitablePositionsBtn");
   ObjectDelete(0, "CloseAllLossPositionsBtn");
   ObjectDelete(0, "mainRectangleLabel");

   ObjectDelete(0, "headingLabel");
   ObjectDelete(0, "headingLabel2");

   ObjectDelete(0, "bottomHeadingLabel");
   ObjectDelete(0, "totalOpenPositionsLabel");
   ObjectDelete(0, "totalPositionsVolumeLabel");
   ObjectDelete(0, "totalPositionsProfitLabel");

   ObjectDelete(0, "buyPositionsHeadingLabel");
   ObjectDelete(0, "totalBuyPositionsLabel");
   ObjectDelete(0, "totalBuyPositionsVolumeLabel");
   ObjectDelete(0, "totalBuyPositionsProfitLabel");

   ObjectDelete(0, "sellPositionsHeadingLabel");
   ObjectDelete(0, "totalSellPositionsLabel");
   ObjectDelete(0, "totalSellPositionsVolumeLabel");
   ObjectDelete(0, "totalSellPositionsProfitLabel");
  }
```

Now that we have finished creating and managing the graphical objects, let us create the custom functions that will implement some of the imported functions from the _PositionsManager.ex5_ library. The first function in this group, which we will call _ModifySlTp()_, will be responsible for modifying the stop loss ( _sl_) and take profit ( _tp_) of all positions opened by this Expert Advisor that match the user-inputted magic number. This function will be executed every time the _setSLTP_ button on the chart is clicked.

```
//+-------------------------------------------------------------------------+
// ModifySlTp(): This function demonstrates how to use the imported ex5     |
// bool SetSlTpByTicket(ulong positionTicket, int sl, int tp);              |
// It runs this function when the setSLTP button on the chart is clicked.   |
//+-------------------------------------------------------------------------+
void ModifySlTp()
  {
//-- Get positions that we have openend with the chart buy and sell buttons to test the imported function with
   int totalOpenPostions = PositionsTotal();
//--- Scan open positions
   for(int x = 0; x < totalOpenPostions; x++)
     {
      //--- Get position properties
      ulong  positionTicket = PositionGetTicket(x); //-- Get ticket to select the position
      string selectedSymbol = PositionGetString(POSITION_SYMBOL);
      ulong positionMagicNo = PositionGetInteger(POSITION_MAGIC);

      //-- modify only the positions we have opened with this EA (magic number) using the BUY and SELL buttons on the chart
      if(selectedSymbol != _Symbol && positionMagicNo != magicNo)
        {
         continue;
        }
      //--- modify the sl and tp of the position
      SetSlTpByTicket(positionTicket, sl, tp);//-- call the imported function from our ex5 library
     }
  }
```

The next function, _SetTrailingSl()_, will be responsible for updating the trailing stop loss. It will be executed both when the _SetTrailingStopLoss_ button on the chart is clicked and on every new incoming tick in the _OnTick()_ event handling function.

```
//+-----------------------------------------------------------------------------------+
// SetTrailingSl(): This function demonstrates how to use the imported ex5            |
// bool SetTrailingStopLoss(ulong positionTicket, int trailingStopLoss);              |
// It runs this function when the SetTrailingStopLoss button on the chart is clicked. |
//+-----------------------------------------------------------------------------------+
void SetTrailingSl()
  {
   int trailingSl = sl;

//-- Get positions that we have openend with the chart buy and sell buttons to test the imported function with
   int totalOpenPostions = PositionsTotal();
//--- Scan open positions
   for(int x = 0; x < totalOpenPostions; x++)
     {
      //--- Get position properties
      ulong  positionTicket = PositionGetTicket(x); //-- Get ticket to select the position
      string selectedSymbol = PositionGetString(POSITION_SYMBOL);
      ulong positionMagicNo = PositionGetInteger(POSITION_MAGIC);

      //-- modify only the positions we have opened with this EA (magic number) using the BUY and SELL buttons on the chart
      if(selectedSymbol != _Symbol && positionMagicNo != magicNo)
        {
         continue;
        }
      //--- set the trailing stop loss
      SetTrailingStopLoss(positionTicket, trailingSl); //-- call the imported function from our ex5 library
     }
  }
```

The _ClosePositionWithTicket()_ function will be responsible for closing all open positions and will be executed when the _ClosePositions_ button on the chart is pressed.

```
//+-----------------------------------------------------------------------------------+
// ClosePositionWithTicket(): This function demonstrates how to use the imported ex5  |
// bool ClosePositionByTicket(ulong positionTicket)                                   |
// It runs this function when the ClosePositions button on the chart is clicked.      |
//+-----------------------------------------------------------------------------------+
void ClosePositionWithTicket()
  {
//-- Get positions that we have openend with the chart buy and sell buttons to test the imported function with
   int totalOpenPostions = PositionsTotal();
//--- Scan open positions
   for(int x = 0; x < totalOpenPostions; x++)
     {
      //--- Get position properties
      ulong  positionTicket = PositionGetTicket(x); //-- Get ticket to select the position
      string selectedSymbol = PositionGetString(POSITION_SYMBOL);
      ulong positionMagicNo = PositionGetInteger(POSITION_MAGIC);

      //-- close only the positions we have opened with this EA (magic number) using the BUY and SELL buttons on the chart
      if(selectedSymbol != _Symbol && positionMagicNo != magicNo)
        {
         continue;
        }

      //--- Close the position
      ClosePositionByTicket(positionTicket);//-- call the imported function from our ex5 library
     }
  }
```

The final function is the standard _OnChartEvent_() MQL5 function, which will detect when our various buttons are pressed and perform the corresponding actions. Almost all imported position management functions from the _PositionsManager.ex5_ library will be called and executed from this function.

```
//+------------------------------------------------------------------+
//| ChartEvent function to detect when the buttons are clicked       |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
//--- Detected a CHARTEVENT_CLICK event
   if(id == CHARTEVENT_OBJECT_CLICK)
     {
      Print(__FUNCTION__, ": ", sparam);

      //--- Buy when OpenBuyPositionBtn button (BUY) is pressed or clicked
      if(sparam == "OpenBuyPositionBtn")
        {
         //-- Call our imported function from the Toolkit/PositionsManager ex5 library
         OpenBuyPosition(magicNo, _Symbol, volumeLot, sl, tp, "ex5 PositionsManager");

         //--- Release and unpress the button
         ObjectSetInteger(0, "OpenBuyPositionBtn", OBJPROP_STATE, false);
        }

      //--- Sell when OpenSellPositionBtn button (SELL) is pressed
      if(sparam == "OpenSellPositionBtn")
        {
         //-- Call our imported function from the Toolkit/PositionsManager ex5 library
         OpenSellPosition(magicNo, _Symbol, volumeLot, sl, tp, "ex5 PositionsManager");
         //OpenSellPosition(magicNo, "NON-EXISTENT-Symbol-Name"/*_Symbol*/, volumeLot, sl, tp, "ex5 PositionsManager");

         //--- Release and unpress the button
         ObjectSetInteger(0, "OpenSellPositionBtn", OBJPROP_STATE, false);
        }

      //--- Modify specified positions SL and TP when SetSlTpBtn button (setSLTP) is pressed
      if(sparam == "SetSlTpBtn")
        {
         ModifySlTp();//-- Modify the SL and TP of the positions generated by the BUY and SELL buttons
         //--- Release and unpress the button
         ObjectSetInteger(0, "SetSlTpBtn", OBJPROP_STATE, false);
        }

      //--- Set the Trailing Stop Loss when SetSlTpBtn button (SetTrailingStopLossBtn) is pressed
      if(sparam == "SetTrailingStopLossBtn")
        {
         SetTrailingSl();//-- Set the Trailing Stop Loss for the positions generated by the BUY and SELL buttons
         //--- Release and unpress the button
         ObjectSetInteger(0, "SetTrailingStopLossBtn", OBJPROP_STATE, false);
        }

      //--- Close specified positions when SetSlTpBtn button (setSLTP) is pressed
      if(sparam == "ClosePositionsBtn")
        {
         ClosePositionWithTicket();//-- Close all the positions generated by the BUY and SELL buttons
         //--- Release and unpress the button
         ObjectSetInteger(0, "ClosePositionsBtn", OBJPROP_STATE, false);
        }

      //--- Close all positions for the current symbol when the CloseAllPositionsBtn button is pressed
      if(sparam == "CloseAllPositionsBtn")
        {
         CloseAllPositions(_Symbol, 0);//-- Close all the open symbol positions
         //--- Release and unpress the button
         ObjectSetInteger(0, "CloseAllPositionsBtn", OBJPROP_STATE, false);
        }

      //--- Close all buy positions for the current symbol when the CloseAllBuyPositionsBtn button is pressed
      if(sparam == "CloseAllBuyPositionsBtn")
        {
         CloseAllBuyPositions(_Symbol, magicNo);//-- Close all the open symbol buy positions
         //--- Release and unpress the button
         ObjectSetInteger(0, "CloseAllBuyPositionsBtn", OBJPROP_STATE, false);
        }

      //--- Close all sell positions for the current symbol when the CloseAllSellPositionsBtn button is pressed
      if(sparam == "CloseAllSellPositionsBtn")
        {
         CloseAllSellPositions(_Symbol, magicNo);//-- Close all the open symbol sell positions
         //--- Release and unpress the button
         ObjectSetInteger(0, "CloseAllSellPositionsBtn", OBJPROP_STATE, false);
        }

      //--- Close all positions with the specified magic number when the CloseAllMagicPositionsBtn button is pressed
      if(sparam == "CloseAllMagicPositionsBtn")
        {
         CloseAllMagicPositions(magicNo);//-- Close all the open positions with the specified magic number
         //--- Release and unpress the button
         ObjectSetInteger(0, "CloseAllMagicPositionsBtn", OBJPROP_STATE, false);
        }

      //--- Close all profitable positions with the specified symbol and magic number when the CloseAllProfitablePositionsBtn button is pressed
      if(sparam == "CloseAllProfitablePositionsBtn")
        {
         CloseAllProfitablePositions(_Symbol, magicNo);//-- Close all the open profitable positions with the specified symbol and magic number
         //--- Release and unpress the button
         ObjectSetInteger(0, "CloseAllProfitablePositionsBtn", OBJPROP_STATE, false);
        }

      //--- Close all loss positions with the specified symbol and magic number when the CloseAllLossPositionsBtn button is pressed
      if(sparam == "CloseAllLossPositionsBtn")
        {
         CloseAllLossPositions(_Symbol, magicNo);//-- Close all the open loss positions with the specified symbol and magic number
         //--- Release and unpress the button
         ObjectSetInteger(0, "CloseAllLossPositionsBtn", OBJPROP_STATE, false);
        }

      //--- Redraw the chart to refresh it
      ChartRedraw();
     }
//---
  }
```

Before running the _PositionsTradePanel.mq5_ Expert Advisor, we need to incorporate all the necessary components into the _OnInit()_, _OnDeinit()_, and _OnTick()_ event handling functions. Start by loading all the graphical objects during the Expert initialization with the _LoadChartObjects()_ function in the _OnInit()_ function.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
//create buttons to demonstrate how the different ex5 library functions work
   LoadChartObjects();
//---
   return(INIT_SUCCEEDED);
  }
```

Call and execute the following functions in the _OnTick()_ function.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//-- Check for profitable positions and set the trailing stop loss on every new tick
   SetTrailingSl(); //-- Calls the ex5 library function responsible for setting Trailing stops
   LoadChartObjects(); //--- Update chart objects
  }
```

Add the deinitialization function to the _OnDeinit()_ event handler to delete all graphical and chart objects when the Expert Advisor is closed or terminated.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
//-- Clean up the chart
   DeleteChartObjects();

//-- Clear any chart comments
   Comment("");
  }
```

You will find the complete source code file for the _PositionsManagerPanel.mq5_ Expert Advisor attached at the end of this article for a more detailed examination.

Save, compile and load the _PositionsManagerPanel.mq5_ Expert Advisor in any MetaTrader 5 chart. In the ' _Dependencies_' tab, you will see all the library function prototypes imported from _PositionsManager.ex5_, along with the full file path where the EX5 library is saved. Once it is loaded on the chart, test it by opening a few positions and checking the ' _Experts_' tab log in the trading terminal's _Toolbox panel_ for the printed log data from the _PositionsManager.ex5_ prototype functions.

![Positions Manager Panel EX5 Dependencies Tab](https://c.mql5.com/2/85/Positions_Manager_Panel_Dependencies.png)

### Conclusion

You have now gained a comprehensive understanding of MQL5 EX5 libraries. We have covered their creation, integration, and implementation into external MQL5 projects, including how to debug various EX5 library errors, and how to update and redeploy them. Additionally, we have created a powerful, feature-rich EX5 Positions Management library, complete with detailed documentation and practical use case examples. I have also demonstrated how to import and implement this library in two different practical MQL5 Expert Advisors, providing real-world examples of effective EX5 library deployment. In the next article, we'll follow a similar approach to develop a comprehensive Pending Orders Management EX5 library, designed to simplify pending order processing tasks in your MQL5 applications. Thank you for following along, and I wish you all the success in your trading and programming journey.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15224.zip "Download all attachments in the single ZIP archive")

[PositionsManager.mq5](https://www.mql5.com/en/articles/download/15224/positionsmanager.mq5 "Download PositionsManager.mq5")(83.52 KB)

[PositionsManager.ex5](https://www.mql5.com/en/articles/download/15224/positionsmanager.ex5 "Download PositionsManager.ex5")(44.65 KB)

[DualVidyaTrader.mq5](https://www.mql5.com/en/articles/download/15224/dualvidyatrader.mq5 "Download DualVidyaTrader.mq5")(10.77 KB)

[PositionsManagerPanel.mq5](https://www.mql5.com/en/articles/download/15224/positionsmanagerpanel.mq5 "Download PositionsManagerPanel.mq5")(27.21 KB)

[PositionsManager\_Imports\_Template.mq5](https://www.mql5.com/en/articles/download/15224/positionsmanager_imports_template.mq5 "Download PositionsManager_Imports_Template.mq5")(4.53 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Trading Toolkit (Part 8): How to Implement and Use the History Manager EX5 Library in Your Codebase](https://www.mql5.com/en/articles/17015)
- [MQL5 Trading Toolkit (Part 7): Expanding the History Management EX5 Library with the Last Canceled Pending Order Functions](https://www.mql5.com/en/articles/16906)
- [MQL5 Trading Toolkit (Part 6): Expanding the History Management EX5 Library with the Last Filled Pending Order Functions](https://www.mql5.com/en/articles/16742)
- [MQL5 Trading Toolkit (Part 5): Expanding the History Management EX5 Library with Position Functions](https://www.mql5.com/en/articles/16681)
- [MQL5 Trading Toolkit (Part 4): Developing a History Management EX5 Library](https://www.mql5.com/en/articles/16528)
- [MQL5 Trading Toolkit (Part 3): Developing a Pending Orders Management EX5 Library](https://www.mql5.com/en/articles/15888)

**[Go to discussion](https://www.mql5.com/en/forum/470377)**

![Build Self Optimizing Expert Advisors With MQL5 And Python](https://c.mql5.com/2/85/Build_Self_Optimizing_Expert_Advisors_With_MQL5_And_Python__LOGO.png)[Build Self Optimizing Expert Advisors With MQL5 And Python](https://www.mql5.com/en/articles/15040)

In this article, we will discuss how we can build Expert Advisors capable of autonomously selecting and changing trading strategies based on prevailing market conditions. We will learn about Markov Chains and how they can be helpful to us as algorithmic traders.

![Neural networks made easy (Part 80): Graph Transformer Generative Adversarial Model (GTGAN)](https://c.mql5.com/2/72/Neural_networks_are_easy_Part_80___LOGO.png)[Neural networks made easy (Part 80): Graph Transformer Generative Adversarial Model (GTGAN)](https://www.mql5.com/en/articles/14445)

In this article, I will get acquainted with the GTGAN algorithm, which was introduced in January 2024 to solve complex problems of generation architectural layouts with graph constraints.

![Population optimization algorithms: Whale Optimization Algorithm (WOA)](https://c.mql5.com/2/73/Whale_Optimization_Algorithm___LOGO.png)[Population optimization algorithms: Whale Optimization Algorithm (WOA)](https://www.mql5.com/en/articles/14414)

Whale Optimization Algorithm (WOA) is a metaheuristic algorithm inspired by the behavior and hunting strategies of humpback whales. The main idea of WOA is to mimic the so-called "bubble-net" feeding method, in which whales create bubbles around prey and then attack it in a spiral motion.

![Reimagining Classic Strategies (Part II): Bollinger Bands Breakouts](https://c.mql5.com/2/85/Reimagining_Classic_Strategies_Part_II__LOGO.png)[Reimagining Classic Strategies (Part II): Bollinger Bands Breakouts](https://www.mql5.com/en/articles/15336)

This article explores a trading strategy that integrates Linear Discriminant Analysis (LDA) with Bollinger Bands, leveraging categorical zone predictions for strategic market entry signals.

[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ddzixhmvivdopiisuegocbjhdjmxvesu&ssn=1769178521550047116&ssn_dr=0&ssn_sr=0&fv_date=1769178521&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15224&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Trading%20Toolkit%20(Part%202)%3A%20Expanding%20and%20Implementing%20the%20Positions%20Management%20EX5%20Library%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917852124234463&fz_uniq=5068284000604190616&sv=2552)

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