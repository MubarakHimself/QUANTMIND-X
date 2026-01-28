---
title: MQL5 Trading Toolkit (Part 1): Developing A Positions Management EX5 Library
url: https://www.mql5.com/en/articles/14822
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T18:01:36.485485
---

[![](https://www.mql5.com/ff/sh/7h2yc16rtqsn2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Market analysis\\
\\
Dozens of channels, thousands of subscribers and daily updates. Learn more about trading.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=glufvbpblsoxonicqfngsyuzwfebnilr&s=103cc3ab372a16872ca1698fc86368ffe3b3eaa21b59b4006d5c6c10f48ad545&uid=&ref=https://www.mql5.com/en/articles/14822&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049570269749685709)

MetaTrader 5 / Examples


### Introduction

As a software developer, I often find it convenient and efficient to create my own code libraries or toolkits. This saves me time because I don't have to repeatedly rewrite code for common tasks required in my various MQL5 development projects. In this article series, we will create an MQL5 trading library responsible for performing repetitive tasks in common MQL5 development projects.

In this first article, we will discuss what developer libraries are, why they are important, and the different types of code libraries you can create with MQL5. Then, we'll proceed to create a library of MQL5 functions to handle various position operations as a practical example to help solidify your understanding of how you can use a code library for a real-world project.

### What Are Code Libraries in MQL5?

MQL5 code libraries are pre-written code functions ( _ex5_) or dynamically linked libraries ( _DLLs_) that we can use to efficiently speed up the development process of Expert Advisors, custom indicators, scripts, or services for the MetaTrader 5 platform.

Imagine a code library as a mechanic's toolbox. Just as the mechanic's toolbox contains a variety of tools (wrenches, screwdrivers, etc.) for specific tasks, a code library contains pre-written functions that serve similar roles. Each function addresses a particular task within your program, such as opening, closing, or modifying positions, sending push notifications, database management, etc. much like how a wrench tightens bolts or a screwdriver turns screws.

### Types of Code Libraries in MQL5

As an MQL5 developer, you have several options for building your code library or toolkit:

- **Function Libraries(ex5)**: These libraries implement a procedural coding style, offering a collection of pre-written functions for specific tasks. They also provide the added security advantage of code encapsulation. Think of them as individual tools, each designed for a particular job.
- **Third-Party C++ DLLs**: You can integrate pre-written C++ libraries as DLLs ( _Dynamic Linked Libraries_). This expands MQL5's capabilities by allowing you to leverage external functionalities.

MetaTrader 5 also offers additional ways to extend your toolkit:

- **.NET Libraries**: The MetaEditor provides seamless integration with .NET libraries through "smart" function import, eliminating the need for custom wrappers.
- **Python Language Module**: The newly supported Python language module allows you to leverage Python functionalities within your MQL5 projects.

If you're comfortable with C++, you can create custom DLLs that integrate easily into your MQL5 projects. You can utilize tools like Microsoft Visual Studio to develop C++ source code files (CPP and H), compile them into DLLs, and then import them into MetaEditor for use with your MQL5 code.

#### Other Code Resources Similar To MQL5 Libraries

- **Class/Include(\*.mqh)**: MQL5 include files utilize object-oriented programming, offering pre-built classes that encapsulate data and functionality. Imagine them as more complex tools that combine functions/methods and data structures. In essence, a class or structure can not be exported to create an MQL5 library (ex5) but you can use pointers and references to classes or structures in MQL5 library functions.

### Why Do You Need to Create or Use an ex5 MQL5 Library?

Creating your own code libraries for MQL5 projects can make your development process much more efficient. These libraries, saved as .ex5 files, act like a personal toolbox filled with functions you've optimized for specific tasks.

> #### Easy Reusability and Modular Design
>
> Save time by having not to rewrite common functions every time you start a new project. With .ex5 libraries, you write the code once, optimize it for best performance, export the functions, and then easily import them into any project. This modular approach keeps your code clean and organized by separating core functionalities from project-specific logic. Each part of the library becomes a building block for creating powerful trading systems.
>
> #### Secure Functions with Encapsulation
>
> Creating MQL5 libraries helps you share your functions while keeping the source code hidden. Encapsulation ensures that the details of your code are secure and not visible to users, while still providing a clear interface for functionality. You only need to share the .ex5 library file along with clear documentation of the exported function definitions with other developers for them to import the library functions into their projects. The .ex5 library file effectively hides the source code and how the exported functions are coded, maintaining a secure and encapsulated workspace in your main project code.
>
> #### Easy Upgrades and Long-Term Benefits
>
> When new language features come out or old ones become deprecated, updating your code is easy with .ex5 libraries. Just update the library code, redeploy, and recompile all projects using it to automatically get the changes. This saves you a lot of time and effort, especially for large projects. The library acts as a central system for your codebase and one update or change will affect all related projects, and thus make your work more efficient in the long run.

### How to Create an ex5 Library in MQL5?

All . _ex5_ libraries begin as .mq5 source code files with the _#property_ library directive added at the beginning of the file and one or more functions that are designated as exportable using the special keyword _export_. The . _mq5_ library source code file transforms into a . _ex5_ library file after it is compiled, securely encapsulating or hiding the source code and making it ready for _import_ and use in other MQL5 projects.

Creating a new MQL5 library is easy with the MetaEditor IDE. Follow the steps below to create a new library source code file (. _mq5_) that will contain the position management functions and later be compiled into a .ex5 library.

**Step 1**: Open the _MetaEditor IDE_ and launch the ' _MQL Wizard_' using the ' _New_' menu item button.

![Mql5Wizard new library file](https://c.mql5.com/2/78/Article_04_Mql5Wizard_New_Library.png)

**Step 2**: Select the ' _Library_' option and click ' _Next_.'

![Mql5 Wizard New Library File](https://c.mql5.com/2/78/Article_04_Toolkit_2MQL5_Wizard_New_Library.png)

**Step 3**: In the ' _General Properties of the Library file_' section, fill in the folder and name for your new library "_Libraries\\Toolkit\\PositionsManager_" and proceed by clicking ' _Finish_' to generate our new library.

![Mql5 Wizard new library file general properties](https://c.mql5.com/2/78/Article_04_Toolkit_0MQL5_Wizard_New_Library2p.png)

An MQL5 Library is typically stored in a file with the " _.mq5_" extension. This file contains the source code of different functions written for various specific tasks. Code libraries are stored in the _MQL5\\Libraries_ folder by default within the MetaTrader 5 installation directory. A quick way to gain access to the Libraries folder is by using the _Navigator panel_ in MetaEditor.

![MQL5 Default Libraries folder in MetaEditor Navigator Panel](https://c.mql5.com/2/78/Article_04_Libraries_Folder.png)

We now have a newly created blank MQL5 library _PositionsManager.mq5_ file which contains the head property directives and a commented-out function. Remember to save the new file before you proceed. This is what our newly generated library file looks like:

```
//+------------------------------------------------------------------+
//|                                             PositionsManager.mq5 |
//|                          Copyright 2024, Wanateki Solutions Ltd. |
//|                                         https://www.wanateki.com |
//+------------------------------------------------------------------+
#property library
#property copyright "Copyright 2024, Wanateki Solutions Ltd."
#property link      "https://www.wanateki.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| My function                                                      |
//+------------------------------------------------------------------+
// int MyCalculator(int value,int value2) export
//   {
//    return(value+value2);
//   }
//+------------------------------------------------------------------+
```

### Components Of An MQL5 Library Source Code File

An MQL5 Library source code file ( _.mq5_) consists of two main components:

**1\. The #property library directive**: This property directive must be added at the top of the library _.mq5_ source code file. The library property lets the compiler know that the given file is a library and an indication of this is stored in the header of the compiled library in the . _ex5_ file that results from the . _mq5_ library compilation.

```
#property library
```

**2. Exported Functions**: At the heart of MQL5 libraries are exported functions. These are the main components of a library as they are in charge of performing all the heavy lifting of the tasks that are to be executed by the library. An exported MQL5 function is similar to an ordinary MQL5 function except that it is declared with the _export postmodifier_ enabling it to be imported and used in other MQL5 programs after compilation. The _export_ modifier instructs the compiler to add the specified function to the table of _ex5_ functions _exported_ by this library file. Only functions that are declared with the _export_ modifier are accessible and detectable from other mql5 programs where they can be called after importation with the special _#import_ directive.

```
//+------------------------------------------------------------------+
//| Example of an exported function with the export postmodifier         |
//+------------------------------------------------------------------+
int ExportedFunction(int a, int b) export
  {
   return(a + b);
  }
//+------------------------------------------------------------------+
```

MQL5 libraries are not expected or required to directly execute any standard events handling like expert advisors, custom indicators, or scripts. This means they do not have any of the standard functions like _OnInit()_, _OnDeinit()_, or _OnTick()_.

### MQL5 Functions Library for Position Management

Position management is a fundamental task for every Expert Advisor under development. These essential operations form the core of any algorithmic trading system. To avoid repetitive coding, MQL5 developers should use libraries to manage positions efficiently. This ensures developers don't rewrite the same position management code for each Expert Advisor.

In this section, we will add some code to our newly created _PositionsManager.mq5_ file to create a position management library using MQL5. This library will handle all position-related operations. By importing it into your Expert Advisor code, you can execute and manage positions effectively, keeping your codebase clean and organized.

#### Global Variables

Trade request deals in MQL5 are represented by a special predefined structure known as _[MqlTradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderequest)_. This structure includes all the necessary fields required to execute trade operations and ensures that all required order request data is packaged within a single data structure. When a trade request is processed, the result is saved in another predefined structure called _[MqlTradeResult](https://www.mql5.com/en/docs/constants/structures/mqltraderesult)_. The _MqlTradeResult_ is responsible for giving us detailed information about the result of the trade request, including whether the request was successful and any relevant data regarding the execution of the trade.

Since we will be using these two special data structures in most of our functions, let us begin by declaring them as global variables so that they are available throughout the library.

```
//---Global variables
//-------------------------------
//-- Trade operations request and result data structures
MqlTradeRequest tradeRequest;
MqlTradeResult  tradeResult;
//-------------------------------
```

#### Position Error Management Function

The first function in our library will be an _Error Management function_. When opening, modifying, and closing positions with MQL5, we often get different types of errors that either require us to abort the operation or re-send the position management request to the trade server another time. Creating a dedicated function to monitor and recommend the right action is a necessity to have a properly coded and optimized library.

Before we can create the error handling function, we need to understand the various MQL5 position management error codes that we might encounter. The table below highlights some of the returned trade server and runtime error codes we need to detect and overcome when managing different position operations. You can find a full list of all the codes of [errors and warnings](https://www.mql5.com/en/docs/constants/errorswarnings) in the MQL5 documentation.

| Code | Code Constant | Description | Action Taken | Type Of Code |
| --- | --- | --- | --- | --- |
| 10004 | TRADE\_RETCODE\_REQUOTE | Requote | Resend the order request again. | Trade server return code (RETCODE) |
| 10008 | TRADE\_RETCODE\_PLACED | Order placed | No action to take. Operations successful. | Trade server return code (RETCODE) |
| 10009 | TRADE\_RETCODE\_DONE | Request completed | No action to take. Operations completed. | Trade server return code (RETCODE) |
| 10013 | TRADE\_RETCODE\_INVALID | Invalid request | Stop resending the order initialization request and update order details. | Trade server return code (RETCODE) |
| 10014 | TRADE\_RETCODE\_INVALID\_VOLUME | Invalid volume in the request | Stop resending the order initialization request and update order details. | Trade server return code (RETCODE) |
| 10016 | TRADE\_RETCODE\_INVALID\_STOPS | Invalid stops in the request | Stop resending the order initialization request and update order details. | Trade server return code (RETCODE) |
| 10017 | TRADE\_RETCODE\_TRADE\_DISABLED | Trade is disabled | Terminate all trade operations and stop resending the order initialization request. | Trade server return code (RETCODE) |
| 10018 | TRADE\_RETCODE\_MARKET\_CLOSED | Market is closed | Stop resending the order initialization request. | Trade server return code (RETCODE) |
| 10019 | TRADE\_RETCODE\_NO\_MONEY | There is not enough money to complete the request | Stop resending the order initialization request and update the order details. | Trade server return code (RETCODE) |
| 10026 | TRADE\_RETCODE\_SERVER\_DISABLES\_AT | Autotrading disabled by server | Trading is not allowed by server. Stop resending the order initialization request. | Trade server return code (RETCODE) |
| 10027 | TRADE\_RETCODE\_CLIENT\_DISABLES\_AT | Autotrading disabled by client terminal | Client terminal has disabled EA trading. Stop resending the order initialization request. | Trade server return code (RETCODE) |
| 10034 | TRADE\_RETCODE\_LIMIT\_VOLUME | The volume of orders and positions for the symbol has reached the limit | Stop resending the order initialization request. | Trade server return code (RETCODE) |
| 10011 | TRADE\_RETCODE\_ERROR | Request processing error | Keep resending the order initialization request. | Trade server return code (RETCODE) |
| 10012 | TRADE\_RETCODE\_TIMEOUT | Request canceled by timeout | Pause execution for a few milliseconds and then keep resending the order initialization request. | Trade server return code (RETCODE) |
| 10015 | TRADE\_RETCODE\_INVALID\_PRICE | Invalid price in the request | Update the order entry price and resend the order initialization request. | Trade server return code (RETCODE) |
| 10020 | TRADE\_RETCODE\_PRICE\_CHANGED | Prices changed | Update the order entry price and resend the order initialization request. | Trade server return code (RETCODE) |
| 10021 | TRADE\_RETCODE\_PRICE\_OFF | There are no quotes to process the request | Pause execution for a few milliseconds and then resend the order initialization request. | Trade server return code (RETCODE) |
| 10024 | TRADE\_RETCODE\_TOO\_MANY\_REQUESTS | Too frequent requests | Pause execution for a few seconds and then resend the order initialization request. | Trade server return code (RETCODE) |
| 10031 | TRADE\_RETCODE\_CONNECTION | No connection with the trade server | Pause execution for a few milliseconds and then resend the order initialization request again. | Trade server return code (RETCODE) |
| 0 | ERR\_SUCCESS | The operation completed successfully | Stop resending the request. Order sent ok. | Runtime error code |
| 4752 | ERR\_TRADE\_DISABLED | Trading by Expert Advisors prohibited | Stop resending the order initialization request. | Runtime error code |
| 4753 | ERR\_TRADE\_POSITION\_NOT\_FOUND | Position not found | Stop resending the trade operation request. | Runtime error code |
| 4754 | ERR\_TRADE\_ORDER\_NOT\_FOUND | Order not found | Stop resending the order request. | Runtime error code |
| 4755 | ERR\_TRADE\_DEAL\_NOT\_FOUND | Deal not found | Stop resending the order request. | Runtime error code |

Let Us create our first function in the library to process the errors mentioned above. The error processing function, named _ErrorAdvisor()_, will be of boolean type, meaning it will return _True_ or _False_ depending on the encountered error type. It will take two arguments to aid in data processing:

1. **callingFunc (string)**: This parameter stores the name or identifier of the function that calls _ErrorAdvisor()_.
2. **symbol (string)**: This parameter stores the symbol name of the asset being worked on.
3. **tradeServerErrorCode (integer)**: This parameter stores the type of error encountered.

If the error is recoverable and not critical, the _ErrorAdvisor()_ function will return _True_. This indicates to the calling function that the order has not been executed yet and it should resend the order request. If _ErrorAdvisor()_ returns _False_, it means the calling function should stop sending any more order requests since the order has already been successfully executed or a critical, unrecoverable error has been encountered.

Remember to place the _export_ postmodifier before the function opening curly bracket to indicate that the function belongs to a library and is intended to be used in other MQL5 programs.

```
//------------------------------------------------------------------+
// ErrorAdvisor(): Error analysis and processing function.          |
// Returns true if order opening failed and order can be re-sent    |
// Returns false if the error is critical and can not be executed   |
//------------------------------------------------------------------+
bool ErrorAdvisor(string callingFunc, string symbol, int tradeServerErrorCode)  export
  {
//-- place the function body here
  }
```

Let us begin by declaring and initializing an integer variable to store the current runtime error. Name the integer _runtimeErrorCode_ and call the _GetLastError()_ function to store the most recent runtime error. We will use this variable to process any runtime error we might encounter in the second nested switch operator.

```
//-- save the current runtime error code
   int runtimeErrorCode = GetLastError();
```

We will scan and process the trade server return errors ( _retcode_) and runtime errors using nested _switch_ operators. This is convenient because it allows us to quickly identify the error type, print a description for the user in the ExpertAdvisor log, and instruct the calling function on how to proceed. You'll notice that I've grouped the errors into two categories:

1. **Errors indicating order completion or non-execution**: These errors will return _False_, instructing the calling function to stop sending order requests.
2. **Errors indicating incomplete orders**: These errors will return _True_, instructing the calling function to resend the order request.

The first switch operator will tackle the trade server return codes and the second nested switch operator will handle the runtime error codes. This approach minimizes the function's code by avoiding sequential checks for every error code or warning.

Now, let's code the first switch statement to examine the encountered _tradeServerErrorCode_ and see what kind of error the trading server reported.

```
switch(tradeServerErrorCode)//-- check for trade server errors
     {
        //--- Cases to scan different retcodes/server return codes
     }
```

Inside the switch statement, we'll add cases for different error codes returned by the trading server. Here are a few of them:

**Requote (code 10004)**: This means the price has changed since the user tried to open the order. In this case, we'll want to print a message to the log, wait a few milliseconds using the _Sleep()_ function, and then tell the calling function to retry opening the order.

```
case 10004:
    Print(symbol, " - ", callingFunc, " ->(TradeServer_Code: ", tradeServerErrorCode, ") Requote!");
    Sleep(10);
    return(true);    //--- Exit the function and retry opening the order again
```

**Order Placed Successfully (code 10008)**: If this code is returned, everything went well, and the order was placed. We can print a message to the log saying the order was successful and then tell the calling function to stop trying to open the order.

```
case 10008:
    Print(symbol, " - ", callingFunc, " ->(TradeServer_Code: ", tradeServerErrorCode, ") Order placed!");
    return(false);    //--- success - order placed ok. exit function
```

We'll add similar cases for other trade server errors (codes 10009, 10011, 10012, etc.), following the same logic of printing a message for the ExpertAdvisor log, waiting a bit if needed, and instructing the calling function on whether to retry sending the trade request again.

If the switch statement for trade server errors doesn't find a match, it means the error might be a runtime error and can only be found through creating another switch statement that scans the current runtime error returned by the _GetLastError()_ function. To tackle this challenge, we'll use a nested switch statement in the default section of the previous switch statement to examine the value of the _runtimeErrorCode_ we saved earlier.

```
default:
  switch(runtimeErrorCode)//-- check for runtime errors
    //-- Add cases for different runtime errors here
  }
}
```

Repeat the same process as we did with the trade server errors and add cases for different runtime error codes:

**No Errors (code 0):** This means everything worked well from our program's perspective. We can print a message to the log saying there were no errors and then tell the calling function to stop trying.

```
case 0:
    Print(symbol, " - ", callingFunc, " ->(Runtime_Code: ", runtimeErrorCode, ") The operation completed successfully!");
    ResetLastError(); //--- reset error cache
    return(false);    //--- Exit the function and stop trying to open order
```

We'll continue adding similar cases for other runtime errors (codes 4752, 4753, 4754, etc.), following the same logic of printing a message for the ExpertAdvisor error log and telling the calling function to stop or proceed with the trade request.

Since we have only accounted for the most important error codes that can affect the order execution process and have not scanned or processed all the possibly existing error codes, we may encounter an error that is currently not processed or accounted for in our current code. In this case, we'll print a message to the log indicating an unknown (other) error occurred, specify the server return code error and runtime error encountered, and then tell the calling function to stop trying to open the order.

```
default: //--- All other error codes
    Print(symbol, " - ", callingFunc, " *OTHER* Error occurred \r\nTrade Server RetCode: ", tradeServerErrorCode, ", Runtime Error Code = ", runtimeErrorCode);
    ResetLastError(); //--- reset error cache
    return(false);    //--- Exit the function and stop trying to open order
    break;
```

Here is the complete error management function _ErrorAdvisor()_ with all the code segments completed:

```
bool ErrorAdvisor(string callingFunc, string symbol, int tradeServerErrorCode)  export
  {
//-- save the current runtime error code
   int runtimeErrorCode = GetLastError();

   switch(tradeServerErrorCode)//-- check for trade server errors
     {
      case 10004:
         Print(symbol, " - ", callingFunc, " ->(TradeServer_Code: ", tradeServerErrorCode, ") Requote!");
         Sleep(10);
         return(true);    //--- Exit the function and retry opening the order again
      case 10008:
         Print(symbol, " - ", callingFunc, " ->(TradeServer_Code: ", tradeServerErrorCode, ") Order placed!");
         return(false);    //--- success - order placed ok. exit function

      case 10009:
         Print(symbol, " - ", callingFunc, " ->(TradeServer_Code: ", tradeServerErrorCode, ") Request completed!");
         return(false);    //--- success - order placed ok. exit function

      case 10011:
         Print(symbol, " - ", callingFunc, " ->(TradeServer_Code: ", tradeServerErrorCode, ") Request processing error!");
         Sleep(10);
         return(true);    //--- Exit the function and retry opening the order again

      case 10012:
         Print(symbol, " - ", callingFunc, " ->(TradeServer_Code: ", tradeServerErrorCode, ") Request canceled by timeout!");
         Sleep(100);
         return(true);    //--- Exit the function and retry opening the order again

      case 10015:
         Print(symbol, " - ", callingFunc, " ->(TradeServer_Code: ", tradeServerErrorCode, ") Invalid price in the request!");
         Sleep(10);
         return(true);    //--- Exit the function and retry opening the order again

      case 10020:
         Print(symbol, " - ", callingFunc, " ->(TradeServer_Code: ", tradeServerErrorCode, ") Prices changed!");
         Sleep(10);
         return(true);    //--- Exit the function and retry opening the order again

      case 10021:
         Print(symbol, " - ", callingFunc, " ->(TradeServer_Code: ", tradeServerErrorCode, ") There are no quotes to process the request!");
         Sleep(100);
         return(true);    //--- Exit the function and retry opening the order again

      case 10024:
         Print(symbol, " - ", callingFunc, " ->(TradeServer_Code: ", tradeServerErrorCode, ") Too frequent requests!");
         Sleep(1000);
         return(true);    //--- Exit the function and retry opening the order again

      case 10031:
         Print(symbol, " - ", callingFunc, " ->(TradeServer_Code: ", tradeServerErrorCode, ") No connection with the trade server!");
         Sleep(100);
         return(true);    //--- Exit the function and retry opening the order again

      default:
         switch(runtimeErrorCode)//-- check for runtime errors
            case 0:
               Print(symbol, " - ", callingFunc, " ->(Runtime_Code: ", runtimeErrorCode, ") The operation completed successfully!");
               ResetLastError(); //--- reset error cache
               return(false);    //--- Exit the function and stop trying to open order

            case 4752:
               Print(symbol, " - ", callingFunc, " ->(Runtime_Code: ", runtimeErrorCode, ") Trading by Expert Advisors prohibited!");
               ResetLastError(); //--- reset error cache
               return(false);    //--- Exit the function and stop trying to open order

            case 4753:
               Print(symbol, " - ", callingFunc, " ->(Runtime_Code: ", runtimeErrorCode, ") Position not found!");
               ResetLastError(); //--- reset error cache
               return(false);    //--- Exit the function and stop trying to open order

            case 4754:
               Print(symbol, " - ", callingFunc, " ->(Runtime_Code: ", runtimeErrorCode, ") Order not found!");
               ResetLastError(); //--- reset error cache
               return(false);    //--- Exit the function and stop trying to open order

            case 4755:
               Print(symbol, " - ", callingFunc, " ->(Runtime_Code: ", runtimeErrorCode, ") Deal not found!");
               ResetLastError(); //--- reset error cache
               return(false);    //--- Exit the function and stop trying to open order

            default: //--- All other error codes
               Print(symbol, " - ", callingFunc, " *OTHER* Error occurred \r\nTrade Server RetCode: ", tradeServerErrorCode, ", Runtime Error Code = ", runtimeErrorCode);
               ResetLastError(); //--- reset error cache
               return(false);    //--- Exit the function and stop trying to open order
               break;
           }
     }
  }
```

#### Trade Permissions Function

This function verifies whether trading is currently allowed in the trading terminal. It considers authorization from the user, the trade server, and the broker. The function is called before any position operations or order requests are sent for processing to the trading server.

We will name the function _TradingIsAllowed()_ and give it a return type of _boolean_. If trading is allowed and enabled, it will return a _boolean_ value of _True_ and a _boolean_ value of _False_ if auto-trading trading is disabled or disallowed. It will not have any parameters or arguments and will contain the code segment below:

```
//+-----------------------------------------------------------------------+
//| TradingIsAllowed() verifies whether auto-trading is currently allowed |                                                                 |
//+-----------------------------------------------------------------------+
bool TradingIsAllowed() export
  {
   if(
      !IsStopped() &&
      MQLInfoInteger(MQL_TRADE_ALLOWED) && TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) &&
      AccountInfoInteger(ACCOUNT_TRADE_ALLOWED) && AccountInfoInteger(ACCOUNT_TRADE_EXPERT)
   )
     {
      return(true);//-- trading is allowed, exit and return true
     }
   return(false);//-- trading is not allowed, exit and return false
  }
```

#### Order Details Logging and Printing Function

This is a simple function to log and print the properties of the different trade operations or position opening requests. It provides a simple way for an ExpertAdvisor user to keep updated on the status of the ExpertAdvisor's operations through the MetaTrader 5 EA log tab. You will notice that this function is not exported and thus it is only accessible through other exported functions that explicitly call it or execute it. We will name it _PrintOrderDetails()_ and specify that it won't return any data making it a void type function and will take one string variable as its input parameter or argument.

```
//+-----------------------------------------------------------------------+
//| PrintOrderDetails() prints the order details for the EA log           |
//+-----------------------------------------------------------------------+
void PrintOrderDetails(string header)
  {
   string orderDescription;
//-- Print the order details
   orderDescription += "_______________________________________________________________________________________\r\n";
   orderDescription += "--> "  + tradeRequest.symbol + " " + EnumToString(tradeRequest.type) + " " + header +
                       " <--\r\n";
   orderDescription += "Order ticket: " + (string)tradeRequest.order + "\r\n";
   orderDescription += "Volume: " + StringFormat("%G", tradeRequest.volume) + "\r\n";
   orderDescription += "Price: " + StringFormat("%G", tradeRequest.price) + "\r\n";
   orderDescription += "Stop Loss: " + StringFormat("%G", tradeRequest.sl) + "\r\n";
   orderDescription += "Take Profit: " + StringFormat("%G", tradeRequest.tp) + "\r\n";
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

#### Position Opening Functions

We will group this functions in two categories:

1. _OpenBuyPositions() function_: This exportable function will be responsible for opening new buy positions as its name implies.
2. _OpenSellPositions() function_: This exportable function will take on the sole task of opening new sell positions as its name also implies.

#### OpenBuyPositions() Function

This function is of type bool and returns a _true_ value if it is successful in opening a new buy position as directed and _false_ if it is not possible to open a new buy position. It takes in six parameters or arguments:

1. **ulong magicNumber**: Used to save the ExpertAdvisors magic number for easier position modification or termination with easy filtering of positions.
2. **string symbol**: Saves the name of the symbol or asset for which the request is being executed.
3. **double lotSize**: Stores the volume or quantity of the buy position to be opened.
4. **int sl**: Stores the stop loss value in points/pips of the buy position.
5. **int tp**: Stores the take profit value in points/pips of the buy position.
6. **string positionComment**: Used to save or store the buy position's comment.

We will begin by coding the function definition. Notice that we have placed the _export_ post modifier before the function opening curly brace to direct the compiler to make this function exportable for use in other MQL5 projects that implement this library.

```
bool OpenBuyPosition(ulong magicNumber, string symbol, double lotSize, int sl, int tp, string positionComment) export
  {
        //--- Function body
  }
```

Before we attempt to open an order, let's check if our Expert Advisor is even allowed to trade. We will do this by calling the _TradingIsAllowed()_ function we created earlier. The _TradingIsAllowed()_ function will scan for various settings and permissions to ensure automated trading is enabled.

If _TradingIsAllowed()_ returns _false_, it means that trading is disabled and our ExpertAdvisor cannot open orders. In this case, we'll immediately return _false_ from this function as well and exit it without opening a new buy order.

```
//-- first check if the ea is allowed to trade
if(!TradingIsAllowed())
  {
   return(false); //--- algo trading is disabled, exit function
  }
```

The next step will be preparing for the order request by clearing out any leftover data from previous trade attempts. To do this, let's use the _ZeroMemory()_ function on the two global trade data structures we created at the beginning of our file: _tradeRequest_ and _tradeResult_. These will store the details of the buy order we want to open and the results returned by the trade server, respectively.

```
//-- reset the the tradeRequest and tradeResult values by zeroing them
   ZeroMemory(tradeRequest);
   ZeroMemory(tradeResult);
```

Now, let's set the parameters for opening a buy position in the _tradeRequest_ data structure variable:

- **tradeRequest.type**: Set this to _ORDER\_TYPE\_BUY_ to indicate a buy order.
- **tradeRequest.action**: Set this to _TRADE\_ACTION\_DEAL_ to specify opening a new position.
- **tradeRequest.magic**: We'll assign the _magicNumber_ provided as an argument here. This helps identify orders opened by our ExpertAdvisor.
- **tradeRequest.symbol**: Assign the symbol provided as an argument here, specifying the currency pair to trade.
- **tradeRequest.tp** and **tradeRequest.sl**: We'll set these to 0 for now, as we'll handle take profit (TP) and stop loss (SL) levels later.
- **tradeRequest.comment**: Assign the _positionComment_ provided as an argument here, which can be used to add a text comment to the order.
- **tradeRequest.deviation**: Set this to allow a deviation of up to twice the current spread for the symbol being traded. This gives the platform some flexibility in finding a matching order price and limits order requotes.

```
//-- initialize the parameters to open a buy position
   tradeRequest.type = ORDER_TYPE_BUY;
   tradeRequest.action = TRADE_ACTION_DEAL;
   tradeRequest.magic = magicNumber;
   tradeRequest.symbol = symbol;
   tradeRequest.tp = 0;
   tradeRequest.sl = 0;
   tradeRequest.comment = positionComment;
   tradeRequest.deviation = SymbolInfoInteger(symbol, SYMBOL_SPREAD) * 2;
```

Normalization for the order volume or lot size is a very important step. We will do this by adjusting the _lotSize_ variable argument or parameter to ensure that it falls within the allowed range for the chosen symbol. Here's how we'll adjust it:

```
//-- set and moderate the lot size or volume
   lotSize = MathMax(lotSize, SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN));  //-- Verify that volume is not less than allowed minimum
   lotSize = MathMin(lotSize, SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX));  //-- Verify that volume is not more than allowed maximum
   lotSize = MathFloor(lotSize / SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP)) * SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP); //-- Round down to nearest volume step
   tradeRequest.volume = lotSize;
```

Next, we will use the _ResetLastError()_ function to clear any previous runtime error codes from the platform. This ensures we will get an accurate error code from the _ErrorAdvisor()_ function later.

```
//--- Reset error cache so that we get an accurate runtime error code in the ErrorAdvisor function
   ResetLastError();
```

Now, we'll enter a loop that can try opening the order up to 100 times ( _with a maximum of 101 iterations_). This loop acts as a failsafe in case the order fails to open on the first try due to temporary market fluctuations or other reasons.

```
for(int loop = 0; loop <= 100; loop++) //-- try opening the order untill it is successful (100 max tries)
   {
    //--- Place the order request code to open a new buy position
   }
```

The first task in the for loop will be to update the order opening price on each iteration in case of an order price requote. We'll use _SymbolInfoDouble(symbol, SYMBOL\_ASK)_ to get the current asking price for the symbol and assign it to _tradeRequest.price_. This ensures our order request reflects the latest market price.

```
//--- update order opening price on each iteration
   tradeRequest.price = SymbolInfoDouble(symbol, SYMBOL_ASK);
```

Next, we will update the take profit and stop loss values on each iteration to match the updated entry price while also normalizing their values to ensure they conform to the required precision requirements and then assign them to the _tradeRequest.tp_ and _tradeRequest.tp_ data structure.

```
//-- set the take profit and stop loss on each iteration
   if(tp > 0)
     {
      tradeRequest.tp = NormalizeDouble(tradeRequest.price + (tp * _Point), _Digits);
     }
   if(sl > 0)
     {
      tradeRequest.sl = NormalizeDouble(tradeRequest.price - (sl * _Point), _Digits);
     }
```

Next, we will send the order to the trade server for execution. For this task, we will use the _OrderSend()_ function, passing the prepared _tradeRequest_ and an empty _tradeResult_ variable as arguments or parameters. This function attempts to open the order based on the specifications stored in _tradeRequest_. The results, including success or error codes, will be stored in the _tradeResult_ variable after it completes its execution.

The _if_ statement we have placed for the _OrderSend()_ function will allow us to check and confirm if the order request is successful or not. If _OrderSend()_ returns _true_, it signifies the order request was sent successfully and if it returns _false_ it signifies that the request failed.

We then call our earlier coded function _PrintOrderDetails()_ with the message " _Sent OK_" to log this information in the Expert Advisor log.

Also, check the _tradeResult.retcode_ to confirm successful order execution. Return _true_ from the _OpenBuyPosition()_ function to indicate success, and use _break_ to exit the loop altogether. If _OrderSend()_ returns false (meaning the order request failed), it signifies an issue occurred. We'll call _PrintOrderDetails()_ with the message " _Sending Failed_" to log this information. We'll also print an error message to highlight the different error codes that we have encountered and return _false_ from the _OpenBuyPosition()_ function to indicate failure, and use _break_ to exit the loop.

```
//--- send order to the trade server
      if(OrderSend(tradeRequest, tradeResult))
        {
         //-- Print the order details
         PrintOrderDetails("Sent OK");

         //-- Confirm order execution
         if(tradeResult.retcode == 10008 || tradeResult.retcode == 10009)
           {
            Print(__FUNCTION__, ": CONFIRMED: Successfully openend a ", symbol, " BUY POSITION #", tradeResult.order, ", Price: ", tradeResult.price);
            PrintFormat("retcode=%u  deal=%I64u  order=%I64u", tradeResult.retcode, tradeResult.deal, tradeResult.order);
            Print("_______________________________________________________________________________________");
            return(true); //-- exit the function
            break; //--- success - order placed ok. exit the for loop
           }
        }
      else //-- Order request failed
        {
         //-- Print the order details
         PrintOrderDetails("Sending Failed");

         //-- order not sent or critical error found
         if(!ErrorAdvisor(__FUNCTION__, symbol, tradeResult.retcode) || IsStopped())
           {
            Print(__FUNCTION__, ": ", symbol, " ERROR opening a BUY POSITION at: ", tradeRequest.price, ", Lot\\Vol: ", tradeRequest.volume);
            Print("_______________________________________________________________________________________");
            return(false); //-- exit the function
            break; //-- exit the for loop
           }
        }
```

Here is the _OpenBuyPosition()_ with all the code segments and their proper sequence:

```
//-------------------------------------------------------------------+
// OpenBuyPosition(): Function to open a new buy entry order.        |
//+------------------------------------------------------------------+
bool OpenBuyPosition(ulong magicNumber, string symbol, double lotSize, int sl, int tp, string positionComment) export
  {
//-- first check if the ea is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

//-- reset the the tradeRequest and tradeResult values by zeroing them
   ZeroMemory(tradeRequest);
   ZeroMemory(tradeResult);

//-- initialize the parameters to open a buy position
   tradeRequest.type = ORDER_TYPE_BUY;
   tradeRequest.action = TRADE_ACTION_DEAL;
   tradeRequest.magic = magicNumber;
   tradeRequest.symbol = symbol;
   tradeRequest.tp = 0;
   tradeRequest.sl = 0;
   tradeRequest.comment = positionComment;
   tradeRequest.deviation = SymbolInfoInteger(symbol, SYMBOL_SPREAD) * 2;

//-- set and moderate the lot size or volume
   lotSize = MathMax(lotSize, SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN));  //-- Verify that volume is not less than allowed minimum
   lotSize = MathMin(lotSize, SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX));  //-- Verify that volume is not more than allowed maximum
   lotSize = MathFloor(lotSize / SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP)) * SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP); //-- Round down to nearest volume step
   tradeRequest.volume = lotSize;

//--- Reset error cache so that we get an accurate runtime error code in the ErrorAdvisor function
   ResetLastError();

   for(int loop = 0; loop <= 100; loop++) //-- try opening the order untill it is successful (100 max tries)
     {
      //--- update order opening price on each iteration
      tradeRequest.price = SymbolInfoDouble(symbol, SYMBOL_ASK);

      //-- set the take profit and stop loss on each iteration
      if(tp > 0)
        {
         tradeRequest.tp = NormalizeDouble(tradeRequest.price + (tp * _Point), _Digits);
        }
      if(sl > 0)
        {
         tradeRequest.sl = NormalizeDouble(tradeRequest.price - (sl * _Point), _Digits);
        }

      //--- send order to the trade server
      if(OrderSend(tradeRequest, tradeResult))
        {
         //-- Print the order details
         PrintOrderDetails("Sent OK");

         //-- Confirm order execution
         if(tradeResult.retcode == 10008 || tradeResult.retcode == 10009)
           {
            Print(__FUNCTION__, ": CONFIRMED: Successfully openend a ", symbol, " BUY POSITION #", tradeResult.order, ", Price: ", tradeResult.price);
            PrintFormat("retcode=%u  deal=%I64u  order=%I64u", tradeResult.retcode, tradeResult.deal, tradeResult.order);
            Print("_______________________________________________________________________________________");
            return(true); //-- exit the function
            break; //--- success - order placed ok. exit the for loop
           }
        }
      else //-- Order request failed
        {
         //-- Print the order details
         PrintOrderDetails("Sending Failed");

         //-- order not sent or critical error found
         if(!ErrorAdvisor(__FUNCTION__, symbol, tradeResult.retcode) || IsStopped())
           {
            Print(__FUNCTION__, ": ", symbol, " ERROR opening a BUY POSITION at: ", tradeRequest.price, ", Lot\\Vol: ", tradeRequest.volume);
            Print("_______________________________________________________________________________________");
            return(false); //-- exit the function
            break; //-- exit the for loop
           }
        }
     }
   return(false);
  }
```

#### OpenSellPositios() Function

This function is very similar to the _OpenBuyPosition()_ function we finished coding above and follows the same procedures, with a few differences such as the type of order being processed. The _OpenSellPosition()_ function is designed to open a new sell position. It includes a for loop that makes multiple attempts in case of failure, significantly increasing its success rate as long as it is provided with valid trade request parameters. Here is the _OpenSellPosition()_ function code:

```
//-------------------------------------------------------------------+
// OpenSellPosition(): Function to open a new sell entry order.      |
//+------------------------------------------------------------------+
bool OpenSellPosition(ulong magicNumber, string symbol, double lotSize, int sl, int tp, string positionComment) export
  {
//-- first check if the ea is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

//-- reset the the tradeRequest and tradeResult values by zeroing them
   ZeroMemory(tradeRequest);
   ZeroMemory(tradeResult);

//-- initialize the parameters to open a sell position
   tradeRequest.type = ORDER_TYPE_SELL;
   tradeRequest.action = TRADE_ACTION_DEAL;
   tradeRequest.magic = magicNumber;
   tradeRequest.symbol = symbol;
   tradeRequest.tp = 0;
   tradeRequest.sl = 0;
   tradeRequest.comment = positionComment;
   tradeRequest.deviation = SymbolInfoInteger(symbol, SYMBOL_SPREAD) * 2;

//-- set and moderate the lot size or volume
   lotSize = MathMax(lotSize, SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN));  //-- Verify that volume is not less than allowed minimum
   lotSize = MathMin(lotSize, SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX));  //-- Verify that volume is not more than allowed maximum
   lotSize = MathFloor(lotSize / SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP)) * SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP); //-- Round down to nearest volume step
   tradeRequest.volume = lotSize;

   ResetLastError(); //--- reset error cache so that we get an accurate runtime error code in the ErrorAdvisor function

   for(int loop = 0; loop <= 100; loop++) //-- try opening the order (101 max) times untill it is successful
     {
      //--- update order opening price on each iteration
      tradeRequest.price = SymbolInfoDouble(symbol, SYMBOL_BID);

      //-- set the take profit and stop loss on each iteration
      if(tp > 0)
        {
         tradeRequest.tp = NormalizeDouble(tradeRequest.price - (tp * _Point), _Digits);
        }
      if(sl > 0)
        {
         tradeRequest.sl = NormalizeDouble(tradeRequest.price + (sl * _Point), _Digits);
        }

      //--- send order to the trade server
      if(OrderSend(tradeRequest, tradeResult))
        {
         //-- Print the order details
         PrintOrderDetails("Sent OK");

         //-- Confirm order execution
         if(tradeResult.retcode == 10008 || tradeResult.retcode == 10009)
           {
            Print("CONFIRMED: Successfully openend a ", symbol, " SELL POSITION #", tradeResult.order, ", Price: ", tradeResult.price);
            PrintFormat("retcode=%u  deal=%I64u  order=%I64u", tradeResult.retcode, tradeResult.deal, tradeResult.order);
            Print("_______________________________________________________________________________________");
            return(true); //-- exit function
            break; //--- success - order placed ok. exit for loop
           }
        }
      else  //-- Order request failed
        {
         //-- Print the order details
         PrintOrderDetails("Sending Failed");

         //-- order not sent or critical error found
         if(!ErrorAdvisor(__FUNCTION__, symbol, tradeResult.retcode) || IsStopped())
           {
            Print(symbol, " ERROR opening a SELL POSITION at: ", tradeRequest.price, ", Lot\\Vol: ", tradeRequest.volume);
            Print("_______________________________________________________________________________________");
            return(false); //-- exit function
            break; //-- exit for loop
           }
        }
     }
   return(false);
  }
```

#### Position Stop Loss and Take Profit Modification Function

Our next function in the library will be called _SetSlTpByTicket()_ and will be responsible for modifying the Stop Loss (SL) and Take Profit (TP) levels for an existing open position using the position's ticket as a filtering mechanism. It takes the position's ticket number, desired SL in pips (points), and desired TP in pips (points) as arguments and attempts to update the position's SL and TP on the trading server. The function will return a boolean value of ( _true_ or _false_). If the Stop Loss and Take Profit levels are successfully modified for the position, it will return _true_, and if it is unable to successfully set or modify the Stop Loss or Take Profit levels it will return _false_.

Here is a breakdown of the _SetSlTpByTicket()_ function arguments or parameters:

1. **ulong positionTicket**: This is a unique identifier for the position we will be modifying.
2. **int sl**: This is the desired Stop Loss level in pips (points) from the opening price.
3. **int tp**: This is the desired Take Profit level in pips (points) from the opening price.

Remember to use the export post modifier in the function definition to make it externally accessible for our library.

```
bool SetSlTpByTicket(ulong positionTicket, int sl, int tp) export
  {
//-- Function body
  }
```

Just like with our other functions above, we will first check if trading for our ExpertAdvisor is allowed using the _TradingIsAllowed()_ function. If trading is disabled, the function exits and returns false.

```
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }
```

Before we select the position specified using the _positionTicket_ argument, we will first reset the runtime error code system variable to get an accurate actionable error response for the _ErrorAdvisor()_ function later. If the selection is successful, a message is printed indicating that the position is selected giving us the green light to access the position's properties. If the selection fails, an error message is printed along with the error code retrieved using _GetLastError()_. The function then exits by returning false.

```
//--- Confirm and select the position using the provided positionTicket
   ResetLastError(); //--- Reset error cache incase of ticket selection errors
   if(PositionSelectByTicket(positionTicket))
     {
      //---Position selected
      Print("\r\n_______________________________________________________________________________________");
      Print(__FUNCTION__, ": Position with ticket:", positionTicket, " selected and ready to set SLTP.");
     }
   else
     {
      Print("\r\n_______________________________________________________________________________________");
      Print(__FUNCTION__, ": Selecting position with ticket:", positionTicket, " failed. ERROR: ", GetLastError());
      return(false); //-- Exit the function
     }
```

After selecting the position, we need to gather and save some details about it. We'll use this information for calculations and later reference.

```
//-- create variables to store the calculated tp and sl prices to send to the trade server
   double tpPrice = 0.0, slPrice = 0.0;
   double newTpPrice = 0.0, newSlPrice = 0.0;

//--- Position ticket selected, save the position properties
   string positionSymbol = PositionGetString(POSITION_SYMBOL);
   double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   double volume = PositionGetDouble(POSITION_VOLUME);
   double currentPositionSlPrice = PositionGetDouble(POSITION_SL);
   double currentPositionTpPrice = PositionGetDouble(POSITION_TP);
   ENUM_POSITION_TYPE positionType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
```

We'll also need some symbol-specific information:

```
//-- Get some information about the positions symbol
   int symbolDigits = (int)SymbolInfoInteger(positionSymbol, SYMBOL_DIGITS); //-- Number of symbol decimal places
   int symbolStopLevel = (int)SymbolInfoInteger(positionSymbol, SYMBOL_TRADE_STOPS_LEVEL);
   double symbolPoint = SymbolInfoDouble(positionSymbol, SYMBOL_POINT);
   double positionPriceCurrent = PositionGetDouble(POSITION_PRICE_CURRENT);
   int spread = (int)SymbolInfoInteger(positionSymbol, SYMBOL_SPREAD);
```

Now, let's calculate the new Stop Loss (SL) and Take Profit (TP) prices based on the position's opening price, desired SL/TP in pips (points), and position type (Buy or Sell). We'll store these initial calculations before validating them:

```
//--Save the non-validated tp and sl prices
   if(positionType == POSITION_TYPE_BUY) //-- Calculate and store the non-validated sl and tp prices
     {
      newSlPrice = entryPrice - (sl * symbolPoint);
      newTpPrice = entryPrice + (tp * symbolPoint);
     }
   else  //-- SELL POSITION
     {
      newSlPrice = entryPrice + (sl * symbolPoint);
      newTpPrice = entryPrice - (tp * symbolPoint);
     }
```

Next, we print a summary of the position details we gathered earlier:

```
//-- Print position properties before modification
   string positionProperties = "--> "  + positionSymbol + " " + EnumToString(positionType) + " SLTP Modification Details" +
   " <--\r\n";
   positionProperties += "------------------------------------------------------------\r\n";
   positionProperties += "Ticket: " + (string)positionTicket + "\r\n";
   positionProperties += "Volume: " + StringFormat("%G", volume) + "\r\n";
   positionProperties += "Price Open: " + StringFormat("%G", entryPrice) + "\r\n";
   positionProperties += "Current SL: " + StringFormat("%G", currentPositionSlPrice) + "   -> New Proposed SL: " + (string)newSlPrice + "\r\n";
   positionProperties += "Current TP: " + StringFormat("%G", currentPositionTpPrice) + "   -> New Proposed TP: " + (string)newTpPrice + "\r\n";
   positionProperties += "Comment: " + PositionGetString(POSITION_COMMENT) + "\r\n";
   positionProperties += "Magic Number: " + (string)PositionGetInteger(POSITION_MAGIC) + "\r\n";
   positionProperties += "---";
   Print(positionProperties);
```

Since the values provided by the library user for SL and TP might not be directly usable by the _OrderSend()_ function. We need to do a simple validation of their values before we can continue:

```
//-- validate the sl and tp to a proper double that can be used in the OrderSend() function
   if(sl == 0)
     {
      slPrice = 0.0;
     }
   if(tp == 0)
     {
      tpPrice = 0.0;
     }
```

Now, we need to perform a more complex validation based on the symbol details we had saved earlier. We will group the validation logic into two groups, one for the Buy positions and another for the Sell positions. The SL and TP validation will be based on the symbol's current price, minimum symbol stop level restrictions, and the symbol's spread.

If a specified TP or SL price is invalid and found to be outside the required range, the original TP or SL price will be retained, and a message will be printed explaining why the modification failed. After we finish validating the SL and TP values, we will print another summary to log the confirmed and verified values for reference:

```
//--- Check if the sl and tp are valid in relation to the current price and set the tpPrice
   if(positionType == POSITION_TYPE_BUY)
     {
      //-- calculate the new sl and tp prices
      newTpPrice = 0.0;
      newSlPrice = 0.0;
      if(tp > 0)
        {
         newTpPrice = entryPrice + (tp * symbolPoint);
        }
      if(sl > 0)
        {
         newSlPrice = entryPrice - (sl * symbolPoint);
        }

      //-- save the new sl and tp prices incase they don't change afte validation below
      tpPrice = newTpPrice;
      slPrice = newSlPrice;

      if( //-- Check if specified TP is valid
         tp > 0 &&
         (
            newTpPrice <= entryPrice + (spread * symbolPoint) ||
            newTpPrice <= positionPriceCurrent ||
            (
               newTpPrice - entryPrice < symbolStopLevel * symbolPoint ||
               (positionPriceCurrent > entryPrice && newTpPrice - positionPriceCurrent < symbolStopLevel * symbolPoint)
            )
         )
      )
        {
         //-- Specified TP price is invalid, don't modify the TP
         Print(
            "Specified proposed ", positionSymbol,
            " TP Price at ", newTpPrice,
            " is invalid since current ", positionSymbol, " price is at ", positionPriceCurrent,
            "\r\nCurrent TP at ", StringFormat("%G", currentPositionTpPrice), " will not be changed!"
         );
         tpPrice = currentPositionTpPrice;
        }

      if( //-- Check if specified SL price is valid
         sl > 0 &&
         (
            newSlPrice >= positionPriceCurrent ||
            entryPrice - newSlPrice < symbolStopLevel * symbolPoint ||
            positionPriceCurrent - newSlPrice < symbolStopLevel * symbolPoint
         )
      )
        {
         //-- Specified SL price is invalid, don't modify the SL
         Print(
            "Specified proposed ", positionSymbol,
            " SL Price at ", newSlPrice,
            " is invalid since current ", positionSymbol, " price is at ", positionPriceCurrent,
            "\r\nCurrent SL at ", StringFormat("%G", currentPositionSlPrice), " will not be changed!"
         );
         slPrice = currentPositionSlPrice;
        }
     }
   if(positionType == POSITION_TYPE_SELL)
     {
      //-- calculate the new sl and tp prices
      newTpPrice = 0.0;
      newSlPrice = 0.0;
      if(tp > 0)
        {
         newTpPrice = entryPrice - (tp * symbolPoint);
        }
      if(sl > 0)
        {
         newSlPrice = entryPrice + (sl * symbolPoint);
        }

      //-- save the new sl and tp prices incase they don't change afte validation below
      tpPrice = newTpPrice;
      slPrice = newSlPrice;

      if( //-- Check if specified TP price is valid
         tp > 0 &&
         (
            newTpPrice >= entryPrice - (spread * symbolPoint) ||
            newTpPrice >= positionPriceCurrent ||
            (
               entryPrice - newTpPrice < symbolStopLevel * symbolPoint ||
               (positionPriceCurrent < entryPrice && positionPriceCurrent - newTpPrice < symbolStopLevel * symbolPoint)
            )
         )
      )
        {
         //-- Specified TP price is invalid, don't modify the TP
         Print(
            "Specified proposed ", positionSymbol,
            " TP Price at ", newTpPrice,
            " is invalid since current ", positionSymbol, " price is at ", positionPriceCurrent,
            "\r\nCurrent TP at ", StringFormat("%G", currentPositionTpPrice), " will not be changed!"
         );
         tpPrice = currentPositionTpPrice;
        }

      if( //-- Check if specified SL price is valid
         sl > 0 &&
         (
            newSlPrice <= positionPriceCurrent ||
            newSlPrice - entryPrice < symbolStopLevel * symbolPoint ||
            newSlPrice - positionPriceCurrent < symbolStopLevel * symbolPoint
         )
      )
        {
         //-- Specified SL price is invalid, don't modify the SL
         Print(
            "Specified proposed ", positionSymbol,
            " SL Price at ", newSlPrice,
            " is invalid since current ", positionSymbol, " price is at ", positionPriceCurrent,
            "\r\nCurrent SL at ", StringFormat("%G", currentPositionSlPrice), " will not be changed!"
         );
         slPrice = currentPositionSlPrice;
        }
     }

//-- Print verified position properties before modification
   positionProperties = "---\r\n";
   positionProperties += "--> Validated and Confirmed SL and TP: <--\r\n";
   positionProperties += "Price Open: " + StringFormat("%G", entryPrice) + ", Price Current: " + StringFormat("%G", positionPriceCurrent) + "\r\n";
   positionProperties += "Current SL: " + StringFormat("%G", currentPositionSlPrice) + "   -> New SL: " + (string)slPrice + "\r\n";
   positionProperties += "Current TP: " + StringFormat("%G", currentPositionTpPrice) + "   -> New TP: " + (string)tpPrice + "\r\n";
   Print(positionProperties);
```

Now that we have the validated SL and TP values, it's time to send a request to the trading server to modify them. We use the _ZeroMemory()_ functionto clear the _tradeRequest_ and _tradeResult_ structures, ensuring they contain no residual data from previous operations. Then, we initialize the _tradeRequest_ structure with the following information:

- **action**: Set to _TRADE\_ACTION\_SLTP_ to indicate modifying Stop Loss and Take Profit.
- **position**: Set to the _positionTicket_ to specify the position we're working on.
- **symbol**: Set to the _positionSymbol_ to identify the symbol for this position.
- **sl**: Set to the _slPrice_ which contains the validated Stop Loss value.
- **tp**: Set to the _tpPrice_ which contains the validated Take Profit value.

We next call the _ResetLastError()_ function to clear any previous error codes stored internally. This ensures we get accurate error codes during the order-sending process.

```
//-- reset the the tradeRequest and tradeResult values by zeroing them
   ZeroMemory(tradeRequest);
   ZeroMemory(tradeResult);
```

We're ready to send the order to the trading server. However, experience has taught me that order execution might fail occasionally due to temporary network issues or server overload. This means that we need to find a smart way to handle sending the order with retries. To resolve this, we will use a for loop that iterates up to 101 times ( _loop <= 100_). This retry mechanism helps handle potential temporary errors during order execution.

Inside the for loop, we use _OrderSend()_ to send the order request contained in _tradeRequest_ and store the result in _tradeResult_. If _OrderSend()_ returns true, it indicates that the SL and TP prices have been successfully changed and the order request was completed without any problems.

We will also do a final confirmation by checking the _tradeResult.retcode_  for specific codes (10008 or 10009) that indicate successful SL/TP modification for this position. If the codes match, we print a confirmation message with details like the position ticket, symbol, and return codes. We then use return( _true_) to exit the function successfully. The break statement exits the loop just to be fully sure that we exit the for loop to avoid unnecessary iterations. If _OrderSend()_ returns _false_ or the _retcode_ doesn't match success codes, it indicates an error.

```
//-- initialize the parameters to set the sltp
   tradeRequest.action = TRADE_ACTION_SLTP; //-- Trade operation type for setting sl and tp
   tradeRequest.position = positionTicket;
   tradeRequest.symbol = positionSymbol;
   tradeRequest.sl = slPrice;
   tradeRequest.tp = tpPrice;

   ResetLastError(); //--- reset error cache so that we get an accurate runtime error code in the ErrorAdvisor function

   for(int loop = 0; loop <= 100; loop++) //-- try modifying the sl and tp 101 times untill the request is successful
     {
      //--- send order to the trade server
      if(OrderSend(tradeRequest, tradeResult))
        {
         //-- Confirm order execution
         if(tradeResult.retcode == 10008 || tradeResult.retcode == 10009)
           {
            PrintFormat("Successfully modified SLTP for #%I64d %s %s", positionTicket, positionSymbol, EnumToString(positionType));
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
            PrintFormat("ERROR modified SLTP for #%I64d %s %s", positionTicket, positionSymbol, EnumToString(positionType));
            Print("_______________________________________________________________________________________\r\n\r\n");
            return(false); //-- exit function
            break; //-- exit for loop
           }
        }
     }
```

Here is the _SetSlTpByTicket()_function with all the code segments and their proper sequence. Make sure that your function has all the components of the code below:

```
bool SetSlTpByTicket(ulong positionTicket, int sl, int tp) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

//--- Confirm and select the position using the provided positionTicket
   ResetLastError(); //--- Reset error cache incase of ticket selection errors
   if(PositionSelectByTicket(positionTicket))
     {
      //---Position selected
      Print("\r\n_______________________________________________________________________________________");
      Print(__FUNCTION__, ": Position with ticket:", positionTicket, " selected and ready to set SLTP.");
     }
   else
     {
      Print("\r\n_______________________________________________________________________________________");
      Print(__FUNCTION__, ": Selecting position with ticket:", positionTicket, " failed. ERROR: ", GetLastError());
      return(false); //-- Exit the function
     }

//-- create variables to store the calculated tp and sl prices to send to the trade server
   double tpPrice = 0.0, slPrice = 0.0;
   double newTpPrice = 0.0, newSlPrice = 0.0;

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

//--Save the non-validated tp and sl prices
   if(positionType == POSITION_TYPE_BUY) //-- Calculate and store the non-validated sl and tp prices
     {
      newSlPrice = entryPrice - (sl * symbolPoint);
      newTpPrice = entryPrice + (tp * symbolPoint);
     }
   else  //-- SELL POSITION
     {
      newSlPrice = entryPrice + (sl * symbolPoint);
      newTpPrice = entryPrice - (tp * symbolPoint);
     }

//-- Print position properties before modification
   string positionProperties = "--> "  + positionSymbol + " " + EnumToString(positionType) + " SLTP Modification Details" +
   " <--\r\n";
   positionProperties += "------------------------------------------------------------\r\n";
   positionProperties += "Ticket: " + (string)positionTicket + "\r\n";
   positionProperties += "Volume: " + StringFormat("%G", volume) + "\r\n";
   positionProperties += "Price Open: " + StringFormat("%G", entryPrice) + "\r\n";
   positionProperties += "Current SL: " + StringFormat("%G", currentPositionSlPrice) + "   -> New Proposed SL: " + (string)newSlPrice + "\r\n";
   positionProperties += "Current TP: " + StringFormat("%G", currentPositionTpPrice) + "   -> New Proposed TP: " + (string)newTpPrice + "\r\n";
   positionProperties += "Comment: " + PositionGetString(POSITION_COMMENT) + "\r\n";
   positionProperties += "Magic Number: " + (string)PositionGetInteger(POSITION_MAGIC) + "\r\n";
   positionProperties += "---";
   Print(positionProperties);

//-- validate the sl and tp to a proper double that can be used in the OrderSend() function
   if(sl == 0)
     {
      slPrice = 0.0;
     }
   if(tp == 0)
     {
      tpPrice = 0.0;
     }

//--- Check if the sl and tp are valid in relation to the current price and set the tpPrice
   if(positionType == POSITION_TYPE_BUY)
     {
      //-- calculate the new sl and tp prices
      newTpPrice = 0.0;
      newSlPrice = 0.0;
      if(tp > 0)
        {
         newTpPrice = entryPrice + (tp * symbolPoint);
        }
      if(sl > 0)
        {
         newSlPrice = entryPrice - (sl * symbolPoint);
        }

      //-- save the new sl and tp prices incase they don't change afte validation below
      tpPrice = newTpPrice;
      slPrice = newSlPrice;

      if( //-- Check if specified TP is valid
         tp > 0 &&
         (
            newTpPrice <= entryPrice + (spread * symbolPoint) ||
            newTpPrice <= positionPriceCurrent ||
            (
               newTpPrice - entryPrice < symbolStopLevel * symbolPoint ||
               (positionPriceCurrent > entryPrice && newTpPrice - positionPriceCurrent < symbolStopLevel * symbolPoint)
            )
         )
      )
        {
         //-- Specified TP price is invalid, don't modify the TP
         Print(
            "Specified proposed ", positionSymbol,
            " TP Price at ", newTpPrice,
            " is invalid since current ", positionSymbol, " price is at ", positionPriceCurrent,
            "\r\nCurrent TP at ", StringFormat("%G", currentPositionTpPrice), " will not be changed!"
         );
         tpPrice = currentPositionTpPrice;
        }

      if( //-- Check if specified SL price is valid
         sl > 0 &&
         (
            newSlPrice >= positionPriceCurrent ||
            entryPrice - newSlPrice < symbolStopLevel * symbolPoint ||
            positionPriceCurrent - newSlPrice < symbolStopLevel * symbolPoint
         )
      )
        {
         //-- Specified SL price is invalid, don't modify the SL
         Print(
            "Specified proposed ", positionSymbol,
            " SL Price at ", newSlPrice,
            " is invalid since current ", positionSymbol, " price is at ", positionPriceCurrent,
            "\r\nCurrent SL at ", StringFormat("%G", currentPositionSlPrice), " will not be changed!"
         );
         slPrice = currentPositionSlPrice;
        }
     }
   if(positionType == POSITION_TYPE_SELL)
     {
      //-- calculate the new sl and tp prices
      newTpPrice = 0.0;
      newSlPrice = 0.0;
      if(tp > 0)
        {
         newTpPrice = entryPrice - (tp * symbolPoint);
        }
      if(sl > 0)
        {
         newSlPrice = entryPrice + (sl * symbolPoint);
        }

      //-- save the new sl and tp prices incase they don't change afte validation below
      tpPrice = newTpPrice;
      slPrice = newSlPrice;

      if( //-- Check if specified TP price is valid
         tp > 0 &&
         (
            newTpPrice >= entryPrice - (spread * symbolPoint) ||
            newTpPrice >= positionPriceCurrent ||
            (
               entryPrice - newTpPrice < symbolStopLevel * symbolPoint ||
               (positionPriceCurrent < entryPrice && positionPriceCurrent - newTpPrice < symbolStopLevel * symbolPoint)
            )
         )
      )
        {
         //-- Specified TP price is invalid, don't modify the TP
         Print(
            "Specified proposed ", positionSymbol,
            " TP Price at ", newTpPrice,
            " is invalid since current ", positionSymbol, " price is at ", positionPriceCurrent,
            "\r\nCurrent TP at ", StringFormat("%G", currentPositionTpPrice), " will not be changed!"
         );
         tpPrice = currentPositionTpPrice;
        }

      if( //-- Check if specified SL price is valid
         sl > 0 &&
         (
            newSlPrice <= positionPriceCurrent ||
            newSlPrice - entryPrice < symbolStopLevel * symbolPoint ||
            newSlPrice - positionPriceCurrent < symbolStopLevel * symbolPoint
         )
      )
        {
         //-- Specified SL price is invalid, don't modify the SL
         Print(
            "Specified proposed ", positionSymbol,
            " SL Price at ", newSlPrice,
            " is invalid since current ", positionSymbol, " price is at ", positionPriceCurrent,
            "\r\nCurrent SL at ", StringFormat("%G", currentPositionSlPrice), " will not be changed!"
         );
         slPrice = currentPositionSlPrice;
        }
     }

//-- Print verified position properties before modification
   positionProperties = "---\r\n";
   positionProperties += "--> Validated and Confirmed SL and TP: <--\r\n";
   positionProperties += "Price Open: " + StringFormat("%G", entryPrice) + ", Price Current: " + StringFormat("%G", positionPriceCurrent) + "\r\n";
   positionProperties += "Current SL: " + StringFormat("%G", currentPositionSlPrice) + "   -> New SL: " + (string)slPrice + "\r\n";
   positionProperties += "Current TP: " + StringFormat("%G", currentPositionTpPrice) + "   -> New TP: " + (string)tpPrice + "\r\n";
   Print(positionProperties);

//-- reset the the tradeRequest and tradeResult values by zeroing them
   ZeroMemory(tradeRequest);
   ZeroMemory(tradeResult);

//-- initialize the parameters to set the sltp
   tradeRequest.action = TRADE_ACTION_SLTP; //-- Trade operation type for setting sl and tp
   tradeRequest.position = positionTicket;
   tradeRequest.symbol = positionSymbol;
   tradeRequest.sl = slPrice;
   tradeRequest.tp = tpPrice;

   ResetLastError(); //--- reset error cache so that we get an accurate runtime error code in the ErrorAdvisor function

   for(int loop = 0; loop <= 100; loop++) //-- try modifying the sl and tp 101 times untill the request is successful
     {
      //--- send order to the trade server
      if(OrderSend(tradeRequest, tradeResult))
        {
         //-- Confirm order execution
         if(tradeResult.retcode == 10008 || tradeResult.retcode == 10009)
           {
            PrintFormat("Successfully modified SLTP for #%I64d %s %s", positionTicket, positionSymbol, EnumToString(positionType));
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
            PrintFormat("ERROR modified SLTP for #%I64d %s %s", positionTicket, positionSymbol, EnumToString(positionType));
            Print("_______________________________________________________________________________________\r\n\r\n");
            return(false); //-- exit function
            break; //-- exit for loop
           }
        }
     }
   return(false);
  }
```

#### Position Closing Function

This function will be called _ClosePositionByTicket()_ and will use a structured approach to ensure that we can effectively close positions based on their ticket numbers. It takes the position's ticket number as an argument. It will check if trading is allowed, select the position using the provided ticket, retrieve and print its properties, prepare a trade request, and attempt to close the position while handling any errors that occur.

First, we define the function and specify that it will return a _boolean_ value ( _true_ or _false_) and take one parameter as an argument.

```
bool ClosePositionByTicket(ulong positionTicket) export
  {
//--- Function body
  }
```

We will then check if the Expert Advisor is allowed to trade.

```
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }
```

Next, we reset any previous errors using the _ResetLastError()_ function and then select the position using the provided ticket number. If the position is selected, we print a message confirming the selection and if the selection fails, we print an error message and exit the function by returning _false_.

```
//--- Confirm and select the position using the provided positionTicket
   ResetLastError(); //--- Reset error cache incase of ticket selection errors
   if(PositionSelectByTicket(positionTicket))
     {
      //---Position selected
      Print("...........................................................................................");
      Print(__FUNCTION__, ": Position with ticket:", positionTicket, " selected and ready to be closed.");
     }
   else
     {
      Print("...........................................................................................");
      Print(__FUNCTION__, ": Selecting position with ticket:", positionTicket, " failed. ERROR: ", GetLastError());
      return(false); //-- Exit the function
     }
```

Once the position is selected successfully, we save its properties and print them.

```
//--- Position ticket selected, save the position properties
   string positionSymbol = PositionGetString(POSITION_SYMBOL);
   double positionVolume = PositionGetDouble(POSITION_VOLUME);
   ENUM_POSITION_TYPE positionType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

//-- Print position properties before closing it
   string positionProperties;
   positionProperties += "-- "  + positionSymbol + " " + EnumToString(positionType) + " Details" +
   " -------------------------------------------------------------\r\n";
   positionProperties += "Ticket: " + (string)positionTicket + "\r\n";
   positionProperties += "Volume: " + StringFormat("%G", PositionGetDouble(POSITION_VOLUME)) + "\r\n";
   positionProperties += "Price Open: " + StringFormat("%G", PositionGetDouble(POSITION_PRICE_OPEN)) + "\r\n";
   positionProperties += "SL: " + StringFormat("%G", PositionGetDouble(POSITION_SL)) + "\r\n";
   positionProperties += "TP: " + StringFormat("%G", PositionGetDouble(POSITION_TP)) + "\r\n";
   positionProperties += "Comment: " + PositionGetString(POSITION_COMMENT) + "\r\n";
   positionProperties += "Magic Number: " + (string)PositionGetInteger(POSITION_MAGIC) + "\r\n";
   positionProperties += "_______________________________________________________________________________________";
   Print(positionProperties);
```

Next, we reset the _tradeRequest_ and _tradeResult_ data structure values by using the _ZeroMemory()_ function to clear any previous data in them. We then initialize the trade request parameters to close the position by setting the trade action to _TRADE\_ACTION\_DEAL_ to indicate a trade termination operation, the position ticket, symbol, volume, and price deviation.

```
//-- reset the the tradeRequest and tradeResult values by zeroing them
   ZeroMemory(tradeRequest);
   ZeroMemory(tradeResult);

//-- initialize the trade reqiest parameters to close the position
   tradeRequest.action = TRADE_ACTION_DEAL; //-- Trade operation type for closing a position
   tradeRequest.position = positionTicket;
   tradeRequest.symbol = positionSymbol;
   tradeRequest.volume = positionVolume;
   tradeRequest.deviation = SymbolInfoInteger(positionSymbol, SYMBOL_SPREAD) * 2;
```

We now need to determine the position's closing price and order type based on whether the position is a buy or sell.

```
//--- Set the price and order type of the position being closed
   if(positionType == POSITION_TYPE_BUY)
     {
      tradeRequest.price = SymbolInfoDouble(positionSymbol, SYMBOL_BID);
      tradeRequest.type = ORDER_TYPE_SELL;
     }
   else//--- For sell type positions
     {
      tradeRequest.price = SymbolInfoDouble(positionSymbol, SYMBOL_ASK);
      tradeRequest.type = ORDER_TYPE_BUY;
     }
```

Finally, we reset any previous errors using _ResetLastError()_ function and then attempt to close the position by sending the trade request. We will use a for loop to try sending the position closing request multiple times to ensure that the position is closed even with a weak internet connection or when non-critical errors occur. If the order is successfully sent and executed (return codes 10008 or 10009), we print a success message and return _true_. If the order fails, we call _ErrorAdvisor()_ function to handle the error. If _ErrorAdvisor()_ function indicates a critical error or if the ExpertAdvisor is stopped, we print an error message and return _false_ to signify the position closing failed.

```
ResetLastError(); //--- reset error cache so that we get an accurate runtime error code in the ErrorAdvisor function

   for(int loop = 0; loop <= 100; loop++) //-- try closing the position 101 times untill the request is successful
     {
      //--- send order to the trade server
      if(OrderSend(tradeRequest, tradeResult))
        {
         //-- Confirm order execution
         if(tradeResult.retcode == 10008 || tradeResult.retcode == 10009)
           {
            Print(__FUNCTION__, "_________________________________________________________________________");
            PrintFormat("Successfully closed position #%I64d %s %s", positionTicket, positionSymbol, EnumToString(positionType));
            PrintFormat("retcode=%u  runtime_code=%u", tradeResult.retcode, GetLastError());
            Print("_______________________________________________________________________________________");
            return(true); //-- exit function
            break; //--- success - order placed ok. exit for loop
           }
        }
      else  //-- position closing request failed
        {
         //-- order not sent or critical error found
         if(!ErrorAdvisor(__FUNCTION__, positionSymbol, tradeResult.retcode) || IsStopped())
           {
            Print(__FUNCTION__, "_________________________________________________________________________");
            PrintFormat("ERROR closing position #%I64d %s %s", positionTicket, positionSymbol, EnumToString(positionType));
            Print("_______________________________________________________________________________________");
            return(false); //-- exit function
            break; //-- exit for loop
           }
        }
     }
```

Make sure to arrange all the code segments above in the following sequence. Here are all the _ClosePositionByTicket()_ function code segments together in their appropriate order:

```
bool ClosePositionByTicket(ulong positionTicket) export
  {
//-- first check if the EA is allowed to trade
   if(!TradingIsAllowed())
     {
      return(false); //--- algo trading is disabled, exit function
     }

//--- Confirm and select the position using the provided positionTicket
   ResetLastError(); //--- Reset error cache incase of ticket selection errors
   if(PositionSelectByTicket(positionTicket))
     {
      //---Position selected
      Print("...........................................................................................");
      Print(__FUNCTION__, ": Position with ticket:", positionTicket, " selected and ready to be closed.");
     }
   else
     {
      Print("...........................................................................................");
      Print(__FUNCTION__, ": Selecting position with ticket:", positionTicket, " failed. ERROR: ", GetLastError());
      return(false); //-- Exit the function
     }

//--- Position ticket selected, save the position properties
   string positionSymbol = PositionGetString(POSITION_SYMBOL);
   double positionVolume = PositionGetDouble(POSITION_VOLUME);
   ENUM_POSITION_TYPE positionType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

//-- Print position properties before closing it
   string positionProperties;
   positionProperties += "-- "  + positionSymbol + " " + EnumToString(positionType) + " Details" +
   " -------------------------------------------------------------\r\n";
   positionProperties += "Ticket: " + (string)positionTicket + "\r\n";
   positionProperties += "Volume: " + StringFormat("%G", PositionGetDouble(POSITION_VOLUME)) + "\r\n";
   positionProperties += "Price Open: " + StringFormat("%G", PositionGetDouble(POSITION_PRICE_OPEN)) + "\r\n";
   positionProperties += "SL: " + StringFormat("%G", PositionGetDouble(POSITION_SL)) + "\r\n";
   positionProperties += "TP: " + StringFormat("%G", PositionGetDouble(POSITION_TP)) + "\r\n";
   positionProperties += "Comment: " + PositionGetString(POSITION_COMMENT) + "\r\n";
   positionProperties += "Magic Number: " + (string)PositionGetInteger(POSITION_MAGIC) + "\r\n";
   positionProperties += "_______________________________________________________________________________________";
   Print(positionProperties);

//-- reset the the tradeRequest and tradeResult values by zeroing them
   ZeroMemory(tradeRequest);
   ZeroMemory(tradeResult);

//-- initialize the trade reqiest parameters to close the position
   tradeRequest.action = TRADE_ACTION_DEAL; //-- Trade operation type for closing a position
   tradeRequest.position = positionTicket;
   tradeRequest.symbol = positionSymbol;
   tradeRequest.volume = positionVolume;
   tradeRequest.deviation = SymbolInfoInteger(positionSymbol, SYMBOL_SPREAD) * 2;

//--- Set the price and order type of the position being closed
   if(positionType == POSITION_TYPE_BUY)
     {
      tradeRequest.price = SymbolInfoDouble(positionSymbol, SYMBOL_BID);
      tradeRequest.type = ORDER_TYPE_SELL;
     }
   else//--- For sell type positions
     {
      tradeRequest.price = SymbolInfoDouble(positionSymbol, SYMBOL_ASK);
      tradeRequest.type = ORDER_TYPE_BUY;
     }

   ResetLastError(); //--- reset error cache so that we get an accurate runtime error code in the ErrorAdvisor function

   for(int loop = 0; loop <= 100; loop++) //-- try closing the position 101 times untill the request is successful
     {
      //--- send order to the trade server
      if(OrderSend(tradeRequest, tradeResult))
        {
         //-- Confirm order execution
         if(tradeResult.retcode == 10008 || tradeResult.retcode == 10009)
           {
            Print(__FUNCTION__, "_________________________________________________________________________");
            PrintFormat("Successfully closed position #%I64d %s %s", positionTicket, positionSymbol, EnumToString(positionType));
            PrintFormat("retcode=%u  runtime_code=%u", tradeResult.retcode, GetLastError());
            Print("_______________________________________________________________________________________");
            return(true); //-- exit function
            break; //--- success - order placed ok. exit for loop
           }
        }
      else  //-- position closing request failed
        {
         //-- order not sent or critical error found
         if(!ErrorAdvisor(__FUNCTION__, positionSymbol, tradeResult.retcode) || IsStopped())
           {
            Print(__FUNCTION__, "_________________________________________________________________________");
            PrintFormat("ERROR closing position #%I64d %s %s", positionTicket, positionSymbol, EnumToString(positionType));
            Print("_______________________________________________________________________________________");
            return(false); //-- exit function
            break; //-- exit for loop
           }
        }
     }
   return(false);
  }
```

Save and compile the _PositionsManager.mq5_ source code file of the library, and you will notice that a new _PositionsManager.ex5_ library file is generated in the _Libraries\\Toolkit\_ folder where we have saved our library.

### Conclusion

By now, you've gained a solid understanding of MQL5 _ex5_ libraries and their creation process. In the next article, we'll expand our positions management library with additional functionalities for various position management tasks and then proceed to demonstrate how to implement _ex5_ libraries in any MQL5 project with different practical examples. Please find the source code file _PositionsManager.mq5_ of the library, which includes all the functions we have created above, attached at the bottom of this article.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14822.zip "Download all attachments in the single ZIP archive")

[PositionsManager.mq5](https://www.mql5.com/en/articles/download/14822/positionsmanager.mq5 "Download PositionsManager.mq5")(31.46 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Trading Toolkit (Part 8): How to Implement and Use the History Manager EX5 Library in Your Codebase](https://www.mql5.com/en/articles/17015)
- [MQL5 Trading Toolkit (Part 7): Expanding the History Management EX5 Library with the Last Canceled Pending Order Functions](https://www.mql5.com/en/articles/16906)
- [MQL5 Trading Toolkit (Part 6): Expanding the History Management EX5 Library with the Last Filled Pending Order Functions](https://www.mql5.com/en/articles/16742)
- [MQL5 Trading Toolkit (Part 5): Expanding the History Management EX5 Library with Position Functions](https://www.mql5.com/en/articles/16681)
- [MQL5 Trading Toolkit (Part 4): Developing a History Management EX5 Library](https://www.mql5.com/en/articles/16528)
- [MQL5 Trading Toolkit (Part 3): Developing a Pending Orders Management EX5 Library](https://www.mql5.com/en/articles/15888)
- [MQL5 Trading Toolkit (Part 2): Expanding and Implementing the Positions Management EX5 Library](https://www.mql5.com/en/articles/15224)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/468562)**
(1)


![wcmycml](https://c.mql5.com/avatar/avatar_na2.png)

**[wcmycml](https://www.mql5.com/en/users/wcmycml)**
\|
16 Oct 2025 at 08:48

I'm a guy who just recently got into EA writing and this article is just timely.


![Developing a multi-currency Expert Advisor (Part 3): Architecture revision](https://c.mql5.com/2/70/Developing_a_multi-currency_advisor_0Part_1g___LOGO__3.png)[Developing a multi-currency Expert Advisor (Part 3): Architecture revision](https://www.mql5.com/en/articles/14148)

We have already made some progress in developing a multi-currency EA with several strategies working in parallel. Considering the accumulated experience, let's review the architecture of our solution and try to improve it before we go too far ahead.

![A Step-by-Step Guide on Trading the Break of Structure (BoS) Strategy](https://c.mql5.com/2/80/A_Step-by-Step_Guide_on_Trading_the_Break_of_Structure____LOGO_.png)[A Step-by-Step Guide on Trading the Break of Structure (BoS) Strategy](https://www.mql5.com/en/articles/15017)

A comprehensive guide to developing an automated trading algorithm based on the Break of Structure (BoS) strategy. Detailed information on all aspects of creating an advisor in MQL5 and testing it in MetaTrader 5 — from analyzing price support and resistance to risk management

![Neural networks made easy (Part 74): Trajectory prediction with adaptation](https://c.mql5.com/2/65/Neural_networks_are_easy_4Part_74w_Adaptive_trajectory_prediction____LOGO.png)[Neural networks made easy (Part 74): Trajectory prediction with adaptation](https://www.mql5.com/en/articles/14143)

This article introduces a fairly effective method of multi-agent trajectory forecasting, which is able to adapt to various environmental conditions.

![Building A Candlestick Trend Constraint Model (Part 4): Customizing Display Style For Each Trend Wave](https://c.mql5.com/2/80/Building_A_Candlestick_Trend_Constraint_Model_Part_4___LOGO.png)[Building A Candlestick Trend Constraint Model (Part 4): Customizing Display Style For Each Trend Wave](https://www.mql5.com/en/articles/14899)

In this article, we will explore the capabilities of the powerful MQL5 language in drawing various indicator styles on Meta Trader 5. We will also look at scripts and how they can be used in our model.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/14822&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049570269749685709)

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