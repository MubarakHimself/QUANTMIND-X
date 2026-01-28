---
title: Understanding Programming Paradigms (Part 1): A Procedural Approach to Developing a Price Action Expert Advisor
url: https://www.mql5.com/en/articles/13771
categories: Trading Systems, Expert Advisors
relevance_score: 12
scraped_at: 2026-01-22T17:15:14.241286
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/13771&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5048997480026186480)

MetaTrader 5 / Examples


### Introduction

In the world of software development, programming paradigms are the guiding blueprints for writing and organizing code. Much like choosing from different routes to reach a destination, different programming approaches or paradigms exist for accomplishing tasks using MQL5.

In two articles, we will explore the basic programming paradigms necessary to build trading tools with MQL5. My goal is to share effective and best-practice methods that produce great results with short and efficient code. I'll explain each programming style and demonstrate it by creating a fully functional Expert Advisor.

### Types of Programming Paradigms

There are three major programming paradigms that every MQL5 developer or programmer should be aware of:

1. **Procedural Programming**: This article will focus on this paradigm.
2. **Functional Programming**: This paradigm will also be discussed in this article, as it is very similar to procedural programming.
3. **Object-Oriented Programming (OOP)**: This paradigm will be discussed in the next article.

Each paradigm comes with its own unique rules and properties, designed to solve specific problems and shape how you effectively develop various trading tools with MQL5.

### Understanding Procedural Programming

Procedural programming is a systematic, step-by-step approach to writing code. It involves breaking down any problem into a sequence of precise instructions, much like following a recipe. Programmers create a clear path for the computer, guiding it through each step line by line to achieve the desired result.

Whether you're new to programming or just curious about code organization, procedural programming provides a straightforward and intuitive entry point into the world of coding.

#### Main Properties Of Procedural Programming

Here are the main properties that characterize procedural programming:

1. Functions:

At the core of procedural programming are functions. These are sets of instructions grouped together to perform a specific task. Functions encapsulate functionality, promoting modularity and code reuse.
2. Top-Down Design:

Procedural programming often employs a top-down design approach. Developers break down a problem into smaller, more manageable sub-tasks. Each sub-task is solved individually, contributing to the overall solution.
3. Imperative Style:

    The imperative or command-style nature of procedural programming emphasizes the explicit statements that change a program's state. Developers specify how the program should achieve a task through a series of procedural commands.
4. Variables and Data:

Procedures or functions in procedural programming manipulate variables and data. These variables can hold values that change during the program's execution. State changes are a fundamental aspect of procedural programming.
5. Sequential Execution:

The program's execution follows a sequential flow. Statements are executed one after another, and control structures like loops and conditionals guide the flow of the program.
6. Modularity:

Procedural programming promotes modularity by organizing code into procedures or functions. Each module handles a specific aspect of the program's functionality, enhancing code organization and maintainability.
7. Reusability:

Code reusability is a key benefit of procedural programming. Once a function is written and tested, it can be used wherever that specific functionality is needed in the program, reducing redundancy and promoting efficiency.
8. Readability:

Procedural code tends to be more readable, especially for those accustomed to a step-by-step approach. The linear flow of execution makes it easy to follow the logic of the program.

### Understanding Functional Programming

Functional programming revolves around the concept of functions as first-class citizens and immutability. It is similar to procedural programming except for the main principle of how it treats data and executes tasks.

Unlike procedural programming, where data can change its appearance and role during the program's execution, functional programming prefers a more stable environment. Once data is created, it stays as is. This commitment to immutability ensures a level of predictability and helps prevent unexpected side effects in your code.

#### Main Properties Of Functional Programming

Here are some key defining characteristics of functional programming:

1. Immutability:

In functional programming, immutability is a core principle. Once data is created, it remains unchanged. Rather than modifying existing data, new data is created with the desired changes. This ensures predictability and helps avoid unintended side effects.
2. Functions as First-Class Citizens:

Functions are treated as first-class citizens, meaning they can be assigned to variables, passed as arguments to other functions, and returned as results from other functions. This flexibility allows for the creation of higher-order functions and promotes a more modular and expressive coding style.
3. Declarative Style:

Functional programming favors a declarative programming style, where the focus is on what the program should accomplish rather than how to achieve it. This contributes to more concise and readable code.
4. Avoidance of Mutable State:

A mutable state is minimized or eliminated in functional programming. Data is treated as immutable, and functions avoid modifying external state. This characteristic simplifies reasoning about the behavior of functions.
5. Recursion and Higher-Order Functions:

Recursion and higher-order functions, where functions take other functions as arguments or return them as results, are commonly employed in functional programming. This leads to more modular and reusable code.

### The Procedural Approach To Developing a Price Action EA

Now that we've delved into the essence of the procedural and functional programming paradigms, let's put the theory into practice with a hands-on example: the creation of a price action-based expert advisor. First, I'll provide insights into the trading strategy we're set to automate. Later, we'll navigate through the various components of the code, unraveling their functionality and how they seamlessly work together.

#### Price Action Strategy with EMA Indicator

Our trading strategy relies on a single indicator known as the exponential moving average (EMA). This indicator is widely used in technical analysis and helps determine the market direction based on your chosen trading setup. You can easily find the [moving average as a standard indicator on MQL5](https://www.mql5.com/en/docs/indicators/ima), making it straightforward to incorporate into our code.

**Buy Entry:** Open a buy position when the most recently closed candle is a buy candle, and both it's low and high prices are above the exponential moving average (EMA).

![Price action EMA strategy buy signal](https://c.mql5.com/2/62/Price_Action_EMA_Strategy_BUY_Illustration.png)

**Sell Entry:** Open a sell position when the most recently closed candle is a sell candle, and both it's low and high prices are below the exponential moving average (EMA).

![Price action EMA strategy sell signal](https://c.mql5.com/2/62/Price_Action_EMA_Strategy_SELL_Illustration.png)

**Exit:** Automatically close all open positions and realize the associated profit or loss when the user-specified percentage profit or loss for the account is achieved or use the traditional stop loss or take profit orders.

| Setup | Condition |
| --- | --- |
| Buy Entry | When the most recently closed candle is a buy candle (close > open), and both it's low and high prices are above the Exponential Moving Average (EMA). |
| Sell Entry | When the most recently closed candle is a sell candle (close < open), and both it's low and high prices are below the Exponential Moving Average (EMA). |
| Exit | Close all open positions and realize profit or loss when the user-specified percentage is reached or when the stop loss or take profit orders are triggered. |

### Coding and Implementing The Trading Strategy

Now that we have established our trading rules and plan, let's bring our trading strategy to life by writing our MQL5 code in the MetaEditor IDE. Follow these steps to ensure that we start with a clean expert advisor template containing only the required mandatory functions:

**Step 1:** Open the MetaEditor IDE and launch 'MQL Wizard' using the 'New' menu item button.

![MQL5 wizard new expert advisor](https://c.mql5.com/2/62/MQL5_Wizard_NewFile_01.png)

**Step 2:** Select the 'Expert Advisor (template)' option and click 'Next.'

![MQL5 wizard new expert advisor](https://c.mql5.com/2/62/MQL5_Wizard_NewFile_02.png)

**Step 3:**  In the 'General Properties' section, fill in the expert advisor name and proceed by clicking 'Next.'

![MQL5 wizard new expert advisor](https://c.mql5.com/2/62/MQL5_Wizard_NewFile_03.png)

**Step 4:**  In the 'Event Handlers' section, ensure no options are selected. Uncheck any options if they are selected, and then click 'Next.'

![MQL5 wizard new expert advisor](https://c.mql5.com/2/62/MQL5_Wizard_NewFile_04.png)

**Step 5:** In the 'Tester Event Handlers' section, ensure no options are selected. Uncheck any options if they are selected, and then click 'Finish' to generate our MQL5 expert advisor template.

![MQL5 wizard new expert advisor](https://c.mql5.com/2/62/MQL5_Wizard_NewFile_05.png)

We now have a clean MQL5 expert advisor template with only the mandatory functions (OnInit, OnDeinit, and OnTick). Remember to save the new file before you proceed.

Here is how our newly generated expert advisor code looks like:

```
//+------------------------------------------------------------------+
//|                                               PriceActionEMA.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
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

First, let's write a brief description of the EA, declare and initialize the user input variables, and then declare all the global variables. Place this code right before the OnInit() function.

```
#property description "A price action EA to demonstrate how to "
#property description "implement the procedural programming paradigm."

//--User input variables
input long magicNumber = 101;//Magic Number (Set 0 [Zero] to disable

input group ""
input ENUM_TIMEFRAMES tradingTimeframe = PERIOD_H1;//Trading Timeframe
input int emaPeriod = 20;//Moving Average Period
input int emaShift = 0;//Moving Average Shift

input group ""
input bool enableTrading = true;//Enable Trading
input bool enableAlerts = false;//Enable Alerts

input group ""
input double accountPercentageProfitTarget = 10.0;//Account Percentage (%) Profit Target
input double accountPercentageLossTarget = 10.0;//Account Percentage (%) Loss Target

input group ""
input int maxPositions = 3;//Max Positions (Max open positions in one direction)
input int tp = 5000;//TP (Take Profit Points/Pips [Zero (0) to diasable])
input int sl = 10000;//SL (Stop Loss Points/Pips [Zero (0) to diasable])
```

As we learned earlier, _"At the core of procedural programming are functions. These are sets of instructions grouped together to perform a specific task. Functions encapsulate functionality, promoting modularity and code reuse."_ We are going to implement this by creating our own custom functions.

**GetInit function:**

This function is responsible for initializing all global variables and performing any other tasks when the EA is loaded or initialized.

```
int GetInit() //Function to initialize the robot and all the variables
  {
   int returnVal = 1;
//create the iMA indicator
   emaHandle = iMA(Symbol(), tradingTimeframe, emaPeriod, emaShift, MODE_EMA, PRICE_CLOSE);
   if(emaHandle < 0)
     {
      Print("Error creating emaHandle = ", INVALID_HANDLE);
      Print("Handle creation: Runtime error = ", GetLastError());
      //force program termination if the handle is not properly set
      return(-1);
     }
   ArraySetAsSeries(movingAverage, true);

//reset the count for positions
   totalOpenBuyPositions = 0;
   totalOpenSellPositions = 0;
   buyPositionsProfit = 0.0;
   sellPositionsProfit = 0.0;
   buyPositionsVol = 0.0;
   sellPositionsVol = 0.0;

   closedCandleTime = iTime(_Symbol, tradingTimeframe, 1);

   startingCapital = AccountInfoDouble(ACCOUNT_EQUITY);//used to calculate the account percentage profit

   if(enableAlerts)
     {
      Alert(MQLInfoString(MQL_PROGRAM_NAME), " has just been LOADED in the ", Symbol(), " ", EnumToString(Period()), " period chart.");
     }

//structure our comment string
   commentString = "\n\n" +
                   "Account No: " + IntegerToString(AccountInfoInteger(ACCOUNT_LOGIN)) +
                   "\nAccount Type: " + EnumToString((ENUM_ACCOUNT_TRADE_MODE)AccountInfoInteger(ACCOUNT_TRADE_MODE)) +
                   "\nAccount Leverage: " + IntegerToString(AccountInfoInteger(ACCOUNT_LEVERAGE)) +
                   "\n-----------------------------------------------------------------------------------------------------";
   return(returnVal);
  }
```

**GetDeinit function:**

This function is responsible for freeing any used memory, clearing chart comments, and handling other deinitialization tasks before the EA is shut down.

```
void GetDeinit()  //De-initialize the robot on shutdown and clean everything up
  {
   IndicatorRelease(emaHandle); //delete the moving average handle and deallocate the memory spaces occupied
   ArrayFree(movingAverage); //free the dynamic arrays containing the moving average buffer data

   if(enableAlerts)
     {
      Alert(MQLInfoString(MQL_PROGRAM_NAME), " has just been REMOVED from the ", Symbol(), " ", EnumToString(Period()), " period chart.");
     }
//delete and clear all chart displayed messages
   Comment("");
  }
```

**GetEma function:**

This function retrieves the value of the exponential moving average and determines the market direction by comparing the EMA value with the open, close, high, and low prices of the recently closed candle. It saves the market direction value in global variables for further processing by other functions.

```
void GetEma()
  {
//Get moving average direction
   if(CopyBuffer(emaHandle, 0, 0, 100, movingAverage) <= 0)
     {
      return;
     }
   movingAverageTrend = "FLAT";
   buyOk = false;
   sellOk = false;
   if(movingAverage[1] > iHigh(_Symbol, tradingTimeframe, 1) && movingAverage[1] > iLow(_Symbol, tradingTimeframe, 1))
     {
      movingAverageTrend = "SELL/SHORT";
      if(iClose(_Symbol, tradingTimeframe, 1) < iOpen(_Symbol, tradingTimeframe, 1))
        {
         sellOk = true;
         buyOk = false;
        }
     }
   if(movingAverage[1] < iHigh(_Symbol, tradingTimeframe, 1) && movingAverage[1] < iLow(_Symbol, tradingTimeframe, 1))
     {
      movingAverageTrend = "BUY/LONG";
      if(iClose(_Symbol, tradingTimeframe, 1) > iOpen(_Symbol, tradingTimeframe, 1))
        {
         buyOk = true;
         sellOk = false;
        }
     }
  }
```

**GetPositionsData function:**

This function scans all open positions, saving their properties, such as profit amount, total number of positions opened, total buy and sell positions opened, and the total volume/lot of each position type. It excludes data for positions not opened by our EA or those that do not match the EA magic number.

```
void GetPositionsData()
  {
//get the total number of all open positions and their status
   if(PositionsTotal() > 0)
     {
      //variables for storing position properties values
      ulong positionTicket;
      long positionMagic, positionType;
      string positionSymbol;
      int totalPositions = PositionsTotal();

      //reset the count
      totalOpenBuyPositions = 0;
      totalOpenSellPositions = 0;
      buyPositionsProfit = 0.0;
      sellPositionsProfit = 0.0;
      buyPositionsVol = 0.0;
      sellPositionsVol = 0.0;

      //scan all the open positions
      for(int x = totalPositions - 1; x >= 0; x--)
        {
         positionTicket = PositionGetTicket(x);//gain access to other position properties by selecting the ticket
         positionMagic = PositionGetInteger(POSITION_MAGIC);
         positionSymbol = PositionGetString(POSITION_SYMBOL);
         positionType = PositionGetInteger(POSITION_TYPE);

         if(positionMagic == magicNumber && positionSymbol == _Symbol)
           {
            if(positionType == POSITION_TYPE_BUY)
              {
               ++totalOpenBuyPositions;
               buyPositionsProfit += PositionGetDouble(POSITION_PROFIT);
               buyPositionsVol += PositionGetDouble(POSITION_VOLUME);
              }
            if(positionType == POSITION_TYPE_SELL)
              {
               ++totalOpenSellPositions;
               sellPositionsProfit += PositionGetDouble(POSITION_PROFIT);
               sellPositionsVol += PositionGetDouble(POSITION_VOLUME);
              }
           }
        }
      //Get and save the account percentage profit
      accountPercentageProfit = ((buyPositionsProfit + sellPositionsProfit) * 100) / startingCapital;
     }
   else  //if no positions are open then the account percentage profit should be zero
     {
      startingCapital = AccountInfoDouble(ACCOUNT_EQUITY);
      accountPercentageProfit = 0.0;

      //reset position counters too
      totalOpenBuyPositions = 0;
      totalOpenSellPositions = 0;
     }
  }
```

**TradingIsAllowed function:**

This function checks if the user, terminal, and broker have given the EA permission to trade.

```
bool TradingIsAllowed()
  {
//check if trading is enabled
   if(enableTrading &&
      MQLInfoInteger(MQL_TRADE_ALLOWED) && TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) &&
      AccountInfoInteger(ACCOUNT_TRADE_ALLOWED) && AccountInfoInteger(ACCOUNT_TRADE_EXPERT)
     )
     {
      tradingStatus = "\n-----------------------------------------------------------------------------------------" +
                      "\nTRADING IS FULLY ENABLED! *** SCANNING FOR ENTRY ***";
      return(true);
     }
   else  //trading is disabled
     {
      tradingStatus = "\n-----------------------------------------------------------------------------------------" +
                      "\nTRADING IS NOT FULLY ENABLED! *** GIVE EA PERMISSION TO TRADE ***";
      return(false);
     }
  }
```

**TradeNow function:**

Responsible for opening new positions when all required checks and signals indicate it's okay to proceed with a new trade initialization.

```
void TradeNow()
  {
//Detect new candle formation and open a new position
   if(closedCandleTime != iTime(_Symbol, tradingTimeframe, 1))  //-- New candle found
     {
      //use the candle time as the position comment to prevent opening dublicate trades on one candle
      string positionComment = IntegerToString(iTime(_Symbol, tradingTimeframe, 1));

      //open a buy position
      if(buyOk && totalOpenBuyPositions < maxPositions)
        {
         //Use the positionComment string to check if we had already have a position open on this candle
         if(!PositionFound(_Symbol, POSITION_TYPE_BUY, positionComment)) //no position has been openend on this candle, open a buy position now
           {
            BuySellPosition(POSITION_TYPE_BUY, positionComment);
           }
        }

      //open a sell position
      if(sellOk && totalOpenSellPositions < maxPositions)
        {
         //Use the positionComment string to check if we had already have a position open on this candle
         if(!PositionFound(_Symbol, POSITION_TYPE_SELL, positionComment)) //no position has been openend on this candle, open a sell position now
           {
            BuySellPosition(POSITION_TYPE_SELL, positionComment);
           }
        }

      //reset closedCandleTime value to prevent new entry orders from opening before a new candle is formed
      closedCandleTime = iTime(_Symbol, tradingTimeframe, 1);
     }
  }
```

**ManageProfitAndLoss function:**

This function checks if the user-inputted profit and loss thresholds are reached. If the conditions are met, it liquidates all profits or losses by closing all open positions.

```
void ManageProfitAndLoss()
  {
//if the account percentage profit or loss target is hit, delete all positions
   double lossLevel = -accountPercentageLossTarget;
   if(
      (accountPercentageProfit >= accountPercentageProfitTarget || accountPercentageProfit <= lossLevel) ||
      ((totalOpenBuyPositions >= maxPositions || totalOpenSellPositions >= maxPositions) && accountPercentageProfit > 0)
   )
     {
      //delete all open positions
      if(PositionsTotal() > 0)
        {
         //variables for storing position properties values
         ulong positionTicket;
         long positionMagic, positionType;
         string positionSymbol;
         int totalPositions = PositionsTotal();

         //scan all the open positions
         for(int x = totalPositions - 1; x >= 0; x--)
           {
            positionTicket = PositionGetTicket(x);//gain access to other position properties by selecting the ticket
            positionMagic = PositionGetInteger(POSITION_MAGIC);
            positionSymbol = PositionGetString(POSITION_SYMBOL);
            positionType = PositionGetInteger(POSITION_TYPE);
            int positionDigits= (int)SymbolInfoInteger(positionSymbol, SYMBOL_DIGITS);
            double positionVolume = PositionGetDouble(POSITION_VOLUME);
            ENUM_POSITION_TYPE positionType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

            if(positionMagic == magicNumber && positionSymbol == _Symbol)
              {
               //print the position details
               Print("*********************************************************************");
               PrintFormat(
                  "#%I64u %s  %s  %.2f  %s [%I64d]",
                  positionTicket, positionSymbol, EnumToString(positionType), positionVolume,
                  DoubleToString(PositionGetDouble(POSITION_PRICE_OPEN), positionDigits), positionMagic
               );

               //reset the the tradeRequest and tradeResult values by zeroing them
               ZeroMemory(tradeRequest);
               ZeroMemory(tradeResult);
               //set the operation parameters
               tradeRequest.action = TRADE_ACTION_DEAL;//type of trade operation
               tradeRequest.position = positionTicket;//ticket of the position
               tradeRequest.symbol = positionSymbol;//symbol
               tradeRequest.volume = positionVolume;//volume of the position
               tradeRequest.deviation = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);//allowed deviation from the price
               tradeRequest.magic = magicNumber;//MagicNumber of the position

               //set the price and order type depending on the position type
               if(positionType == POSITION_TYPE_BUY)
                 {
                  tradeRequest.price = SymbolInfoDouble(positionSymbol, SYMBOL_BID);
                  tradeRequest.type = ORDER_TYPE_SELL;
                 }
               else
                 {
                  tradeRequest.price = SymbolInfoDouble(positionSymbol, SYMBOL_ASK);
                  tradeRequest.type = ORDER_TYPE_BUY;
                 }

               //print the position close details
               PrintFormat("Close #%I64d %s %s", positionTicket, positionSymbol, EnumToString(positionType));
               //send the tradeRequest
               if(OrderSend(tradeRequest, tradeResult)) //trade tradeRequest success, position has been closed
                 {
                  if(enableAlerts)
                    {
                     Alert(
                        _Symbol + " PROFIT LIQUIDATION: Just successfully closed POSITION (#" +
                        IntegerToString(positionTicket) + "). Check the EA journal for more details."
                     );
                    }
                  PrintFormat("Just successfully closed position: #%I64d %s %s", positionTicket, positionSymbol, EnumToString(positionType));
                  PrintFormat("retcode=%u  deal=%I64u  order=%I64u", tradeResult.retcode, tradeResult.deal, tradeResult.order);
                 }
               else  //trade tradeRequest failed
                 {
                  //print the information about the operation
                  if(enableAlerts)
                    {
                     Alert(
                        _Symbol + " ERROR ** PROFIT LIQUIDATION: closing POSITION (#" +
                        IntegerToString(positionTicket) + "). Check the EA journal for more details."
                     );
                    }
                  PrintFormat("Position clossing failed: #%I64d %s %s", positionTicket, positionSymbol, EnumToString(positionType));
                  PrintFormat("OrderSend error %d", GetLastError());//print the error code
                 }
              }
           }
        }
     }
  }
```

**PrintOnChart function:**

Formats and displays the EA status on the chart, providing the user with a visual text representation of the account and EA status.

```
void PrintOnChart()
  {
//update account status strings and display them on the chart
   accountStatus = "\nAccount Balance: " + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2) + accountCurrency +
                   "\nAccount Equity: " + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2) + accountCurrency +
                   "\nAccount Profit: " + DoubleToString(AccountInfoDouble(ACCOUNT_PROFIT), 2) + accountCurrency +
                   "\nAccount Percentage Profit: " + DoubleToString(accountPercentageProfit, 2) + "%" +
                   "\n-----------------------------------------------------------------------------------------" +
                   "\nTotal Buy Positions Open: " + IntegerToString(totalOpenBuyPositions) +
                   "        Total Vol/Lots: " + DoubleToString(buyPositionsVol, 2) +
                   "        Profit: " + DoubleToString(buyPositionsProfit, 2) + accountCurrency +
                   "\nTotal Sell Positions Open: " + IntegerToString(totalOpenSellPositions) +
                   "        Total Vol/Lots: " + DoubleToString(sellPositionsVol, 2) +
                   "        Profit: " + DoubleToString(sellPositionsProfit, 2) + accountCurrency +
                   "\nPositionsTotal(): " + IntegerToString(PositionsTotal()) +
                   "\n-----------------------------------------------------------------------------------------" +
                   "\nJust Closed Candle:     Open: " + DoubleToString(iOpen(_Symbol, tradingTimeframe, 1), _Digits) +
                   "     Close: " + DoubleToString(iClose(_Symbol, tradingTimeframe, 1), _Digits) +
                   "     High: " + DoubleToString(iHigh(_Symbol, tradingTimeframe, 1), _Digits) +
                   "     Low: " + DoubleToString(iLow(_Symbol, tradingTimeframe, 1), _Digits) +
                   "\n-----------------------------------------------------------------------------------------" +
                   "\nMovingAverage (EMA): " + DoubleToString(movingAverage[1], _Digits) +
                   "     movingAverageTrend = " + movingAverageTrend +
                   "\nsellOk: " + IntegerToString(sellOk) +
                   "\nbuyOk: " + IntegerToString(buyOk);

//show comments on the chart
   Comment(commentString + accountStatus + tradingStatus);
  }
```

**BuySellPosition function:**

This function opens new buy and sell positions.

```
bool BuySellPosition(int positionType, string positionComment)
  {
//reset the the tradeRequest and tradeResult values by zeroing them
   ZeroMemory(tradeRequest);
   ZeroMemory(tradeResult);
//initialize the parameters to open a position
   tradeRequest.action = TRADE_ACTION_DEAL;
   tradeRequest.symbol = Symbol();
   tradeRequest.deviation = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   tradeRequest.magic = magicNumber;
   tradeRequest.comment = positionComment;
   double volumeLot = NormalizeDouble(((SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN) * AccountInfoDouble(ACCOUNT_EQUITY)) / 10000), 2);

   if(positionType == POSITION_TYPE_BUY)
     {
      if(sellPositionsVol > volumeLot && AccountInfoDouble(ACCOUNT_MARGIN_LEVEL) > 200)
        {
         volumeLot = NormalizeDouble((sellPositionsVol + volumeLot), 2);
        }
      if(volumeLot < 0.01)
        {
         volumeLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
        }
      if(volumeLot > SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX))
        {
         volumeLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
        }
      tradeRequest.volume = NormalizeDouble(volumeLot, 2);
      tradeRequest.type = ORDER_TYPE_BUY;
      tradeRequest.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      if(tp > 0)
        {
         tradeRequest.tp = NormalizeDouble(tradeRequest.price + (tp * _Point), _Digits);
        }
      if(sl > 0)
        {
         tradeRequest.sl = NormalizeDouble(tradeRequest.price - (sl * _Point), _Digits);
        }
      if(OrderSend(tradeRequest, tradeResult)) //successfully openend the position
        {
         if(enableAlerts)
           {
            Alert(_Symbol, " Successfully openend BUY POSITION #", tradeResult.order, ", Price: ", tradeResult.price);
           }
         PrintFormat("retcode=%u  deal=%I64u  order=%I64u", tradeResult.retcode, tradeResult.deal, tradeResult.order);
         return(true);
        }
      else
        {
         if(enableAlerts)
           {
            Alert(_Symbol, " ERROR opening a BUY POSITION at: ", SymbolInfoDouble(_Symbol, SYMBOL_ASK));
           }
         PrintFormat("ERROR: Opening a BUY POSITION: ErrorCode = %d",GetLastError());//OrderSend failed, output the error code
         return(false);
        }
     }

   if(positionType == POSITION_TYPE_SELL)
     {
      if(buyPositionsVol > volumeLot && AccountInfoDouble(ACCOUNT_MARGIN_LEVEL) > 200)
        {
         volumeLot = NormalizeDouble((buyPositionsVol + volumeLot), 2);
        }
      if(volumeLot < 0.01)
        {
         volumeLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
        }
      if(volumeLot > SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX))
        {
         volumeLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
        }
      tradeRequest.volume = NormalizeDouble(volumeLot, 2);
      tradeRequest.type = ORDER_TYPE_SELL;
      tradeRequest.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      if(tp > 0)
        {
         tradeRequest.tp = NormalizeDouble(tradeRequest.price - (tp * _Point), _Digits);
        }
      if(sl > 0)
        {
         tradeRequest.sl = NormalizeDouble(tradeRequest.price + (sl * _Point), _Digits);
        }
      if(OrderSend(tradeRequest, tradeResult)) //successfully openend the position
        {
         if(enableAlerts)
           {
            Alert(_Symbol, " Successfully openend SELL POSITION #", tradeResult.order, ", Price: ", tradeResult.price);
           }
         PrintFormat("retcode=%u  deal=%I64u  order=%I64u", tradeResult.retcode, tradeResult.deal, tradeResult.order);
         return(true);
        }
      else
        {
         if(enableAlerts)
           {
            Alert(_Symbol, " ERROR opening a SELL POSITION at: ", SymbolInfoDouble(_Symbol, SYMBOL_ASK));
           }
         PrintFormat("ERROR: Opening a SELL POSITION: ErrorCode = %d",GetLastError());//OrderSend failed, output the error code
         return(false);
        }
     }
   return(false);
  }
```

**PositionFound function:**

This function checks if a specified position exists so that the EA does’nt open multiple duplicate positions on one candle.

```
bool PositionFound(string symbol, int positionType, string positionComment)
  {
   if(PositionsTotal() > 0)
     {
      ulong positionTicket;
      int totalPositions = PositionsTotal();
      //scan all the open positions
      for(int x = totalPositions - 1; x >= 0; x--)
        {
         positionTicket = PositionGetTicket(x);//gain access to other position properties by selecting the ticket
         if(
            PositionGetInteger(POSITION_MAGIC) == magicNumber && PositionGetString(POSITION_SYMBOL) == symbol &&
            PositionGetInteger(POSITION_TYPE) == positionType && PositionGetString(POSITION_COMMENT) == positionComment
         )
           {
            return(true);//a similar position exists, don't open another position on this candle
            break;
           }
        }
     }
   return(false);
  }
```

Now that we have defined our custom functions, let's call them to perform their intended tasks.

- Place and call the GetInit function in the OnInit function.

```
int OnInit()
  {
//---
   if(GetInit() <= 0)
     {
      return(INIT_FAILED);
     }
//---
   return(INIT_SUCCEEDED);
  }
```

- Place and call the GetDeinit function in the OnDeinit function.

```
void OnDeinit(const int reason)
  {
//---
   GetDeinit();
  }
```

- Place and call the following functions in the OnTick function in the appropriate order. Since some functions modify global variables relied upon by other functions to make key decisions, we will ensure to call them first, ensuring data is processed and updated before being accessed by other functions.

```
void OnTick()
  {
//---
   GetEma();
   GetPositionsData();
   if(TradingIsAllowed())
     {
      TradeNow();
      ManageProfitAndLoss();
     }
   PrintOnChart();
  }
```

With the EA code complete, save and compile it. This allows you to access your EA directly from the trading terminal. The full code is attached at the bottom of the article.

### Testing Our EA in the Strategy Tester

It's crucial to ensure that our EA operates according to our plan. We can achieve this by either loading it on an active symbol chart and trading it in a demo account or by utilizing the strategy tester for a comprehensive evaluation. While you can test it on a demo account, for now, we'll use the strategy tester to assess its performance.

Here are the settings we'll apply in the strategy tester:

- **Broker:** MT5 Metaquotes demo account (Automatically created upon MT5 installation)

- **Symbol:** EURUSD

- **Testing Period (Date):** 1 year (Nov 2022 to Nov 2023)

- **Modeling:** Every tick based on real ticks

- **Deposit:** $10,000 USD

- **Leverage:** 1:100


![PriceActionEA strategy tester settings](https://c.mql5.com/2/61/bandicam_2023-11-30_12-25-57-595.png)

![PriceActionEA strategy tester settings](https://c.mql5.com/2/61/bandicam_2023-11-30_12-26-32-305.png)

Reviewing our backtesting results, our EA not only generated a profit but also maintained a remarkably low drawdown. This strategy exhibits promise and can be further modified and optimized to yield even better results, especially when applied to multiple symbols simultaneously.

![PriceActionEMA tester results](https://c.mql5.com/2/61/bandicam_2023-11-30_13-12-14-448.png)

![PriceActionEMA tester results](https://c.mql5.com/2/61/bandicam_2023-11-30_13-11-29-996.png)

![PriceActionEMA tester results](https://c.mql5.com/2/61/bandicam_2023-11-30_13-13-01-022.png)

### Conclusion

It's straightforward for even beginner MQL5 programmers to understand the procedural code of the EA we've just created above. This simplicity arises from the clear and direct nature of procedural programming, particularly when utilizing functions to organize the code based on specific tasks and global variables to pass modified data to different functions.

However, you may notice that a drawback of procedural code is its tendency to expand significantly as your EA becomes more complicated, rendering it suitable primarily for less complex projects. In cases where your project is highly complex, opting for the object-oriented programming approach proves more advantageous than procedural programming.

In our upcoming article, we'll introduce you to object-oriented programming and transform our recently created procedural price action EA code into object-oriented code. This will provide a distinct comparison between these paradigms, offering a clearer understanding of the differences.

Thank you for investing the time to read this article, I wish you the very best in your MQL5 development journey and trading endeavors.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13771.zip "Download all attachments in the single ZIP archive")

[PriceActionEMA.mq5](https://www.mql5.com/en/articles/download/13771/priceactionema.mq5 "Download PriceActionEMA.mq5")(23.33 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/459384)**
(6)


![Jay Allen](https://c.mql5.com/avatar/2022/10/6338F655-6E3C.jpg)

**[Jay Allen](https://www.mql5.com/en/users/mrjayallen)**
\|
5 Jan 2024 at 19:49

Excellent Article on Procedural Programming!


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
5 Jan 2024 at 20:13

Good article. I expected some procedural price action coding like ABCD waves structure or a conditional zigzag with steps like in step 1 find a peak, step 2 find a trough etc... I don't think a candle low high above or below EMA is procedural "price action" if we leave trading functions apart.


![Wanateki Solutions LTD](https://c.mql5.com/avatar/2025/1/677afe52-ae53.png)

**[Kelvin Muturi Muigua](https://www.mql5.com/en/users/kelmut)**
\|
11 Jan 2024 at 18:25

**Altan Karakaya [#](https://www.mql5.com/en/forum/459384#comment_51302333):**

Very informative and interesting

Thank you. I'm glad you liked it! Your feedback is appreciated.


![Wanateki Solutions LTD](https://c.mql5.com/avatar/2025/1/677afe52-ae53.png)

**[Kelvin Muturi Muigua](https://www.mql5.com/en/users/kelmut)**
\|
11 Jan 2024 at 18:25

**Jay Allen [#](https://www.mql5.com/en/forum/459384#comment_51532845):**

Excellent Article on Procedural Programming!

Thank you. I appreciate your kind words and feedback!


![Wanateki Solutions LTD](https://c.mql5.com/avatar/2025/1/677afe52-ae53.png)

**[Kelvin Muturi Muigua](https://www.mql5.com/en/users/kelmut)**
\|
11 Jan 2024 at 18:28

**Arpit T [#](https://www.mql5.com/en/forum/459384#comment_51533866):**

Good article. I expected some procedural price action coding like ABCD waves structure or a conditional zigzag with steps like in step 1 find a peak, step 2 find a trough etc... I don't think a candle low high above or below EMA is procedural "price action" if we leave trading functions apart.

Thanks for your feedback! The article primarily centered on the procedural programming paradigm as a style of writing and organizing code, using the trading strategy as a practical example for implementation in MQL5. In a future article, I will demonstrate how to create an ABCD wave or zigzag steps strategy with MQL5, as you have suggested. Feel free to recommend any other areas you'd like me to cover in upcoming articles!


![Neural networks made easy (Part 55): Contrastive intrinsic control (CIC)](https://c.mql5.com/2/57/cic-055-avatar.png)[Neural networks made easy (Part 55): Contrastive intrinsic control (CIC)](https://www.mql5.com/en/articles/13212)

Contrastive training is an unsupervised method of training representation. Its goal is to train a model to highlight similarities and differences in data sets. In this article, we will talk about using contrastive training approaches to explore different Actor skills.

![Brute force approach to patterns search (Part VI): Cyclic optimization](https://c.mql5.com/2/57/bruteforce_approach_cyclic_optimization_avatar.png)[Brute force approach to patterns search (Part VI): Cyclic optimization](https://www.mql5.com/en/articles/9305)

In this article I will show the first part of the improvements that allowed me not only to close the entire automation chain for MetaTrader 4 and 5 trading, but also to do something much more interesting. From now on, this solution allows me to fully automate both creating EAs and optimization, as well as to minimize labor costs for finding effective trading configurations.

![Data Science and Machine Learning (Part 17): Money in the Trees? The Art and Science of Random Forests in Forex Trading](https://c.mql5.com/2/63/midjourney_image_13765_54_491__3-logo.png)[Data Science and Machine Learning (Part 17): Money in the Trees? The Art and Science of Random Forests in Forex Trading](https://www.mql5.com/en/articles/13765)

Discover the secrets of algorithmic alchemy as we guide you through the blend of artistry and precision in decoding financial landscapes. Unearth how Random Forests transform data into predictive prowess, offering a unique perspective on navigating the complex terrain of stock markets. Join us on this journey into the heart of financial wizardry, where we demystify the role of Random Forests in shaping market destiny and unlocking the doors to lucrative opportunities

![MQL5 Wizard Techniques you should know (Part 09): Pairing K-Means Clustering with Fractal Waves](https://c.mql5.com/2/62/midjourney_image_13915_50_439__5-logo.png)[MQL5 Wizard Techniques you should know (Part 09): Pairing K-Means Clustering with Fractal Waves](https://www.mql5.com/en/articles/13915)

K-Means clustering takes the approach to grouping data points as a process that’s initially focused on the macro view of a data set that uses random generated cluster centroids before zooming in and adjusting these centroids to accurately represent the data set. We will look at this and exploit a few of its use cases.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/13771&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5048997480026186480)

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