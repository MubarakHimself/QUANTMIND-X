---
title: Building A Candlestick Trend Constraint Model (Part 8): Expert Advisor Development (II)
url: https://www.mql5.com/en/articles/15322
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 8
scraped_at: 2026-01-22T17:44:52.642996
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/15322&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049349461186030051)

MetaTrader 5 / Tester


### Contents:

> 1. [Introduction](https://www.mql5.com/en/articles/15322#para1)
> 2. [Creating the Expert Advisor:](https://www.mql5.com/en/articles/15322#para2)
>
>    - (i) [Launching the default EA (template) in MetaEditor](https://www.mql5.com/en/articles/15322#para3).
>    - (ii) [Customizing the Template.](https://www.mql5.com/en/articles/15322#para4)
>    - (iii) [Writing Trend Constraint Expert logic into the prepared Template.](https://www.mql5.com/en/articles/15322#para5)
>
> 4. [Tester](https://www.mql5.com/en/articles/15322#para6)
> 5. [Conclusion](https://www.mql5.com/en/articles/15322#para7)

### Introduction

In our [previous](https://www.mql5.com/en/articles/15321) article, we explored creating an expert advisor (EA) using the Trend Constraint V1.09 indicator, complemented by the manually executed Trend Constraint R-R script for placing risk and reward rectangles. While this setup provided insightful trading signals and enhanced visualization, it required manual intervention that could be streamlined. With the fast-paced nature of trading environments, the need for a more efficient solution becomes apparent. Many traders seek integrated systems that function autonomously, reducing the need for constant supervision and manual execution.

This article takes the next step in our series by guiding you through the development of an independent expert advisor (EA) that not only incorporates the trend analysis capabilities of Trend Constraint V1.09, but also integrates risk-reward functionalities directly into the EA. Our goal is to empower traders with an all-in-one solution using MQL5 on the MetaTrader 5 platform, offering enhanced automation and seamless operation to keep pace with market demands.

To achieve this, we will revisit the foundational stages covered in [Part 1](https://www.mql5.com/en/articles/14347), [Part 2](https://www.mql5.com/en/articles/14803), and [Part 3](https://www.mql5.com/en/articles/14853), borrowing the logic necessary for the EA to perform its trading tasks. Now, we'll combine all these parts and their logic to write the Expert Advisor.

Below is a summary of the conditions that generated signals in the indicator as implemented in MQL5:

**Buying (Long) Condition:**

> 1. The day candlestick must be Bullish.
> 2. On a lower time frame within a Bullish D1 candlestick, we need a confluence of signals. Specifically, the inbuilt indicator, such as the Relative Strength Index (RSI), should indicate an oversold condition, and the fast-moving average must cross above the slow-moving average.

**Selling (Short) Condition:**

> 1. The day candlestick must be Bearish.
> 2. On a lower time frame within a Bearish D1 candlestick, we need a confluence of signals. Specifically, the inbuilt indicator, such as the Relative Strength Index (RSI), should indicate an overbought condition, and the fast-moving average must cross below the slow-moving average.

### Creating the Expert Advisor

We use the MetaEditor app to write our MQL5 Expert Program. The most important thing is to keep the foundation in mind. Below is the summarized structure of our EA.

Architecture summary of most EA programs:

- Initialization (OnInit): Set up necessary indicators and variables.
- Main Loop (OnTick): Process incoming ticks to evaluate market conditions and make trading decisions.
- Trade Management (OnTrade): Handle trade-related events.
- Testing (OnTester and related functions): Provide structure for optimizing and evaluating the EA's performance in the Strategy Tester.
- User Interaction (OnChartEvent): Optional, but allows interaction with the EA via chart events.

I believe the rough draft in the Block Diagram below will help clarify our desired outcome, making it easier to understand.

![Trend Constraint Expert](https://c.mql5.com/2/91/EA_flow.drawio.png)

A rough block diagram of the EA parts and how they interlink

This structure ensures that our EA is well-organized and capable of effectively handling the various tasks needed to implement our trading strategy. I will divide the development process into subsections, but advanced developers can skip stages (i) and (ii) and proceed directly to subsections (iii).

> (i) [Launching the default EA (template) in MetaEditor](https://www.mql5.com/en/articles/15322#para3).
>
> (ii) [Customizing the Template](https://www.mql5.com/en/articles/15322#para4).
>
> (iii)  [Writing Trend Constraint Expert logic into the prepared Template](https://www.mql5.com/en/articles/15322#para5).

### (i) Launching the EA template

In MetaEditor, press Ctrl + N to launch a new file, and choose "Expert Advisor (Template)" as shown in the image below:

![Launching an Expert Advisor (Template) ](https://c.mql5.com/2/90/MetaEditor64_MtzZho9lTX.gif)

Launch a new Expert Advisor (Template)

Please note that the author details and links are already available when I launch a new project because of previous projects on my computer. You can customize them for yourself.

### (ii) Customizing the Template

The intention behind this writing is not only to demonstrate my coding ability, but also to impact those who are just beginning to understand the process so that they can apply the acquired skills in the future. At this point, we are looking at the template, which may not seem meaningful to a newbie. Before we start coding into the skeleton template, I will extract and briefly explain the most important functions needed. We will then proceed to fill in the details, writing the EA code to meet our needs. This approach is crucial as it will enable you to start other projects of your own, different from what we're focusing on now, without any difficulty.

The skeleton provides the foundation that we will later build upon with the logic for our EA.

Here, is the generated template from MetaEditor:

```
//+------------------------------------------------------------------+
//|                                      Trend Constraint Expert.mq5 |
//|                                Copyright 2024, Clemence Benjamin |
//|             https://www.mql5.com/en/users/billionaire2024/seller |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
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
//| Trade function                                                   |
//+------------------------------------------------------------------+
void OnTrade()
  {
//---

  }
//+------------------------------------------------------------------+
//| Tester function                                                  |
//+------------------------------------------------------------------+
double OnTester()
  {
//---
   double ret=0.0;
//---

//---
   return(ret);
  }
//+------------------------------------------------------------------+
//| TesterInit function                                              |
//+------------------------------------------------------------------+
void OnTesterInit()
  {
//---

  }
//+------------------------------------------------------------------+
//| TesterPass function                                              |
//+------------------------------------------------------------------+
void OnTesterPass()
  {
//---

  }
//+------------------------------------------------------------------+
//| TesterDeinit function                                            |
//+------------------------------------------------------------------+
void OnTesterDeinit()
  {
//---

  }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---

  }
//+------------------------------------------------------------------+
```

From the template, let's discuss the key functions essential for our Expert Advisors:

- **[OnInit() Function](https://www.mql5.com/en/docs/event_handlers/oninit)**: This function runs once when the EA is initialized, setting up indicators, variables, or resources required for operation. Proper initialization ensures that all necessary resources are ready before the EA starts processing market data. For your "Trend Constraint Expert Advisor," the RSI indicator and other crucial variables would typically be initialized here.

- **[OnDeinit() Function](https://www.mql5.com/en/docs/event_handlers/ondeinit)**: This function is called when the EA is deinitialized, such as when it's removed from a chart or the terminal is shut down. It's used to clean up resources, like releasing indicator handles or closing files, to prevent memory leaks or other issues.

- **[OnTick() Function](https://www.mql5.com/en/docs/event_handlers/ontick)**: This main function is triggered every time a new tick (price update) is received for the symbol to which the EA is attached. In your "Trend Constraint Expert Advisor," it would include the logic for checking market conditions, such as the D1 trend and RSI levels, and making trading decisions like opening or closing positions.

- **[OnTrade() Function](https://www.mql5.com/en/docs/event_handlers/ontrade)**: Called when a trade event occurs, such as placing, modifying, or closing an order, this function is vital for monitoring trade status and reacting to changes. For instance, you could use it to track when a trade is opened and adjust the EA's behavior accordingly.

- **[OnTester()](https://www.mql5.com/en/docs/event_handlers/ontester)** Function: This function is used during strategy testing to return a double value, serving as a custom criterion for optimization. It allows you to define a custom metric, such as profit factor or drawdown, to evaluate the EA's performance when testing in the Strategy Tester.

- **[OnTesterInit(), OnTesterPass(), and OnTesterDeinit()](https://www.mql5.com/en/docs/event_handlers/ontesterinit) Functions**: These functions are specifically for strategy testing and optimization, managing the start, ongoing passes, and end of tests in the Strategy Tester. They provide greater control over the testing process by initializing resources, collecting data, and cleaning up after tests.

- **[OnChartEvent() Function](https://www.mql5.com/en/docs/event_handlers/onchartevent)**: This function handles chart events, like mouse clicks or key presses, enabling interaction with the EA while it's running. If your EA includes user interaction, such as changing parameters or triggering actions through chart events, this function is essential.

With an understanding of these functions, we can see how to structure the logic within them. However, the functions provided by the template might not be sufficient for the complexity of the EA we're developing. Additional functions may be necessary, and we'll need to justify and explain their inclusion to meet the specific requirements of our "Trend Constraint Expert."

Additional functions not included in the Expert Advisor template but necessary for our project:

- Indicator Functions (e.g., iRSI): Used to calculate indicators like RSI that are integral to your trading strategy.
- Trade Functions (CTrade class): Used for managing orders and positions, such as placing buy/sell orders, setting stop losses, and modifying positions.

Thus, we have outlined the logic that we want our EA to use within specific, suitable functions. This is a skeletal layout of the functions I intend to include in our Expert Advisor, and I have explained how each is essential in this development with comments in the code.

```
//+------------------------------------------------------------------+
//|                               Trend Constraint Expert Advisor.mq5|
//|                                 Copyright 2024, Clemence Benjamin|
//|              https://www.mql5.com/en/users/billionaire2024/seller|
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   // Initialization code here
   rsi_handle = iRSI(_Symbol, PERIOD_CURRENT, RSI_Period, PRICE_CLOSE);
   if (rsi_handle == INVALID_HANDLE)
     {
      Print("Failed to create RSI indicator handle");
      return(INIT_FAILED);
     }

   // Any other initialization tasks

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   // Cleanup code here
   IndicatorRelease(rsi_handle);  // Release RSI indicator handle
   // Any other deinitialization tasks
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Main trading logic goes here

   // Determine market conditions (e.g., daily trend, RSI levels)

   // Check for trade conditions and execute orders if necessary

   // Implement trailing stop logic if necessary
  }

//+------------------------------------------------------------------+
//| Trade event function                                             |
//+------------------------------------------------------------------+
void OnTrade()
  {
   // Handle trade events (e.g., order placement, modification, closure)
  }

//+------------------------------------------------------------------+
//| Tester function                                                  |
//+------------------------------------------------------------------+
double OnTester()
  {
   double ret = 0.0;
   // Custom optimization criteria (if any) go here
   return(ret);
  }

//+------------------------------------------------------------------+
//| TesterInit function                                              |
//+------------------------------------------------------------------+
void OnTesterInit()
  {
   // Initialization for testing (if needed)
  }

//+------------------------------------------------------------------+
//| TesterPass function                                              |
//+------------------------------------------------------------------+
void OnTesterPass()
  {
   // Actions after each optimization pass (if needed)
  }

//+------------------------------------------------------------------+
//| TesterDeinit function                                            |
//+------------------------------------------------------------------+
void OnTesterDeinit()
  {
   // Cleanup after testing (if needed)
  }

//+------------------------------------------------------------------+
//| Chart event function                                             |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
   // Handle chart events (e.g., mouse clicks, key presses) here
  }

//+------------------------------------------------------------------+
```

### (iii)  Writing Trend Constraint Expert logic.

Now, we can move on to writing the logic for each function that constitutes our Expert Advisor. Let's construct our program step by step:

> **Including the Trade Library:**
>
> We begin by including the trade library, which is necessary because the CTrade class within this library provides the functions needed for executing trade operations such as opening, modifying, and closing positions. By including this library, we enable the EA to interact with the market and manage trades programmatically.
>
> ```
> #include <Trade\Trade.mqh>  // Include the trade library
> ```
>
> **Defining Input Parameters:**
>
> Next, we define the input parameters, which users can adjust to match their trading preferences. These parameters include the RSI period, overbought and oversold levels, lot size, stop loss, take profit, and trailing stop. By writing these input lines, we ensure that the EA can be customized for different market conditions. The program will use these values to make decisions on when to enter and exit trades.
>
> ```
> // Input parameters
> input int    RSI_Period = 14;            // RSI period
> input double RSI_Overbought = 70.0;      // RSI overbought level
> input double RSI_Oversold = 30.0;        // RSI oversold level
> input double Lots = 0.1;                 // Lot size
> input double StopLoss = 100;             // Stop Loss in points
> input double TakeProfit = 200;           // Take Profit in points
> input double TrailingStop = 50;          // Trailing Stop in points
> ```
>
> **Declaring Global Variables:**
>
> We then declare global variables such as RSI\_value and RS1\_handle, which will store the RSI value and handle, respectively, as well as an instance of the CTrade class. By declaring these variables, we ensure that the EA can maintain state across different functions, allowing the program to access and modify these values as needed during its operation.
>
> ```
> // Global variables
> double rsi_value;
> int rsi_handle;
> CTrade trade;  // Declare an instance of the CTrade class
> ```
>
> **Initializing the Expert Advisor:**
>
> In the OnInit function, we create the RSI indicator handle using the iRSI function. This step is crucial because the program requires this handle to fetch the RSI values during each tick. If the handle cannot be created, we have written the program to return INIT\_FAILED, preventing the EA from running without this critical component. This ensures that the program only runs when fully equipped to analyze market data.
>
> ```
> int OnInit()
>   {
>    // Create an RSI indicator handle
>    rsi_handle = iRSI(_Symbol, PERIOD_CURRENT, RSI_Period, PRICE_CLOSE);
>    if (rsi_handle == INVALID_HANDLE)
>      {
>       Print("Failed to create RSI indicator handle");
>       return(INIT_FAILED);
>      }
>
>    return(INIT_SUCCEEDED);
>   }
> ```
>
> **Deinitializing the Expert Advisor:**
>
> To manage resources effectively, we implement the OnDeinit function to release the RSI indicator handle when the EA is removed from the chart. By writing this cleanup code, we prevent memory leaks and ensure that resources are properly released. The program will execute this cleanup automatically when it is deinitialized, maintaining optimal performance.
>
> ```
> void OnDeinit(const int reason)
>   {
>    // Release the RSI indicator handle
>    IndicatorRelease(rsi_handle);
>   }
> ```
>
> **Implementing Core Trading Logic:**
>
> The core trading logic resides in the OnTick function, which we design to be executed on every market tick. First, we write code to determine the current daily trend by comparing the opening and closing prices of the daily candle. This analysis allows the program to identify whether the market is bullish or bearish, which is vital for making informed trading decisions.
>
> ```
> void OnTick()
>   {
>    // Determine current daily trend (bullish or bearish)
>    double daily_open = iOpen(_Symbol, PERIOD_D1, 0);
>    double daily_close = iClose(_Symbol, PERIOD_D1, 0);
>
>    bool is_bullish = daily_close > daily_open;
>    bool is_bearish = daily_close < daily_open;
> ```
>
> **Retrieving RSI Values:**
>
> We then retrieve the RSI value using CopyBuffer and the RSI handle that was created earlier. By writing this, we ensure that the program can assess whether the market is in an overbought or oversold state. The program will use this RSI value in its decision-making process, determining whether it meets the conditions for entering a trade.
>
> ```
>    // Get the RSI value for the current bar
>    double rsi_values[];
>    if (CopyBuffer(rsi_handle, 0, 0, 1, rsi_values) <= 0)
>      {
>       Print("Failed to get RSI value");
>       return;
>      }
>    rsi_value = rsi_values[0];
> ```
>
> **Closing Positions on Trend Change:**
>
> We also include logic to close any open positions if the market trend changes. For instance, if the trend shifts from bullish to bearish, the EA will close any open buy positions, and vice versa. By writing this safeguard, we ensure that the program maintains alignment with the prevailing market sentiment, which is crucial for minimizing risk.
>
> ```
>    // Close open positions if the trend changes
>    for (int i = PositionsTotal() - 1; i >= 0; i--)
>      {
>       if (PositionSelect(PositionGetSymbol(i)))  // Corrected usage
>         {
>          int position_type = PositionGetInteger(POSITION_TYPE);
>          ulong ticket = PositionGetInteger(POSITION_TICKET);  // Get the position ticket
>
>          if ((position_type == POSITION_TYPE_BUY && is_bearish) ||
>              (position_type == POSITION_TYPE_SELL && is_bullish))
>            {
>             trade.PositionClose(ticket);  // Use the ulong variable directly
>            }
>         }
>      }
> ```
>
> **Checking for Buy and Sell Conditions:**
>
> For the buy and sell conditions, we write logic to check for a bullish trend combined with an oversold RSI for buy orders, and a bearish trend combined with an overbought RSI for short orders. By programming these conditions, we ensure that the EA only enters trades when both trend and momentum indicators are in agreement. The program will monitor these conditions and execute trades accordingly, ensuring a disciplined approach to trading.
>
> ```
>    // Check for buy condition (bullish trend + RSI oversold)
>    if (is_bullish && rsi_value < RSI_Oversold)
>      {
>       // No open positions? Place a buy order
>       if (PositionsTotal() == 0)
>         {
>          double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
>          double sl = price - StopLoss * _Point;
>          double tp = price + TakeProfit * _Point;
>
>          // Open a buy order
>          trade.Buy(Lots, _Symbol, price, sl, tp, "TrendConstraintExpert Buy");
>         }
>      }
>
>    // Check for sell condition (bearish trend + RSI overbought)
>    if (is_bearish && rsi_value > RSI_Overbought)
>      {
>       // No open positions? Place a sell order
>       if (PositionsTotal() == 0)
>         {
>          double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
>          double sl = price + StopLoss * _Point;
>          double tp = price - TakeProfit * _Point;
>
>          // Open a sell order
>          trade.Sell(Lots, _Symbol, price, sl, tp, "TrendConstraintExpert Sell");
>         }
>      }
> ```
>
> **Implementing a Trailing Stop Mechanism:**
>
> Finally, we implement a trailing stop mechanism to protect profits as the market moves in favor of an open position. By writing this trailing stop logic, we ensure that the EA dynamically adjusts the stop loss to lock in profits while allowing the trade to continue as long as the market remains favorable. The program will automatically manage the trailing stop, ensuring that it responds to market movements to maximize gains and minimize losses.
>
> ```
>    // Apply trailing stop
>    for (int i = PositionsTotal() - 1; i >= 0; i--)
>      {
>       if (PositionSelect(PositionGetSymbol(i)))  // Corrected usage
>         {
>          double price = PositionGetDouble(POSITION_PRICE_OPEN);
>          double stopLoss = PositionGetDouble(POSITION_SL);
>          double current_price;
>
>          if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
>            {
>             current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
>             if (current_price - price > TrailingStop * _Point)
>               {
>                if (stopLoss < current_price - TrailingStop * _Point)
>                  {
>                   trade.PositionModify(PositionGetInteger(POSITION_TICKET), current_price - TrailingStop * _Point, PositionGetDouble(POSITION_TP));
>                  }
>               }
>            }
>          else if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
>            {
>             current_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
>             if (price - current_price > TrailingStop * _Point)
>               {
>                if (stopLoss > current_price + TrailingStop * _Point || stopLoss == 0)
>                  {
>                   trade.PositionModify(PositionGetInteger(POSITION_TICKET), current_price + TrailingStop * _Point, PositionGetDouble(POSITION_TP));
>                  }
>               }
>            }
>         }
>      }
>   }
> ```

> Our final program with header and other properties:
>
> ```
> //+------------------------------------------------------------------+
> //|                                      Trend Constraint Expert.mq5 |
> //|                                Copyright 2024, Clemence Benjamin |
> //|             https://www.mql5.com/en/users/billionaire2024/seller |
> //+------------------------------------------------------------------+
> #property copyright "Copyright 2024, Clemence Benjamin"
> #property link      "https://www.mql5.com/en/users/billionaire2024/seller"
> #property version   "1.00"
> #property description "A System that seeks to Long D1 Bullish sentiment and short D1 Bearish sentiment"
> #property strict
>
> #include <Trade\Trade.mqh>  // Include the trade library
>
> // Input parameters
> input int    RSI_Period = 14;            // RSI period
> input double RSI_Overbought = 70.0;      // RSI overbought level
> input double RSI_Oversold = 30.0;        // RSI oversold level
> input double Lots = 0.1;                 // Lot size
> input double StopLoss = 100;             // Stop Loss in points
> input double TakeProfit = 200;           // Take Profit in points
> input double TrailingStop = 50;          // Trailing Stop in points
>
> // Global variables
> double rsi_value;
> int rsi_handle;
> CTrade trade;  // Declare an instance of the CTrade class
>
> //+------------------------------------------------------------------+
> //| Expert initialization function                                   |
> //+------------------------------------------------------------------+
> int OnInit()
>   {
>    // Create an RSI indicator handle
>    rsi_handle = iRSI(_Symbol, PERIOD_CURRENT, RSI_Period, PRICE_CLOSE);
>    if (rsi_handle == INVALID_HANDLE)
>      {
>       Print("Failed to create RSI indicator handle");
>       return(INIT_FAILED);
>      }
>
>    return(INIT_SUCCEEDED);
>   }
>
> //+------------------------------------------------------------------+
> //| Expert deinitialization function                                 |
> //+------------------------------------------------------------------+
> void OnDeinit(const int reason)
>   {
>    // Release the RSI indicator handle
>    IndicatorRelease(rsi_handle);
>   }
>
> //+------------------------------------------------------------------+
> //| Expert tick function                                             |
> //+------------------------------------------------------------------+
> void OnTick()
>   {
>    // Determine current daily trend (bullish or bearish)
>    double daily_open = iOpen(_Symbol, PERIOD_D1, 0);
>    double daily_close = iClose(_Symbol, PERIOD_D1, 0);
>
>    bool is_bullish = daily_close > daily_open;
>    bool is_bearish = daily_close < daily_open;
>
>    // Get the RSI value for the current bar
>    double rsi_values[];
>    if (CopyBuffer(rsi_handle, 0, 0, 1, rsi_values) <= 0)
>      {
>       Print("Failed to get RSI value");
>       return;
>      }
>    rsi_value = rsi_values[0];
>
>    // Close open positions if the trend changes
>    for (int i = PositionsTotal() - 1; i >= 0; i--)
>      {
>       if (PositionSelect(PositionGetSymbol(i)))  // Corrected usage
>         {
>          int position_type = PositionGetInteger(POSITION_TYPE);
>          ulong ticket = PositionGetInteger(POSITION_TICKET);  // Get the position ticket
>
>          if ((position_type == POSITION_TYPE_BUY && is_bearish) ||
>              (position_type == POSITION_TYPE_SELL && is_bullish))
>            {
>             trade.PositionClose(ticket);  // Use the ulong variable directly
>            }
>         }
>      }
>
>    // Check for buy condition (bullish trend + RSI oversold)
>    if (is_bullish && rsi_value < RSI_Oversold)
>      {
>       // No open positions? Place a buy order
>       if (PositionsTotal() == 0)
>         {
>          double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
>          double sl = price - StopLoss * _Point;
>          double tp = price + TakeProfit * _Point;
>
>          // Open a buy order
>          trade.Buy(Lots, _Symbol, price, sl, tp, "TrendConstraintExpert Buy");
>         }
>      }
>
>    // Check for sell condition (bearish trend + RSI overbought)
>    if (is_bearish && rsi_value > RSI_Overbought)
>      {
>       // No open positions? Place a sell order
>       if (PositionsTotal() == 0)
>         {
>          double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
>          double sl = price + StopLoss * _Point;
>          double tp = price - TakeProfit * _Point;
>
>          // Open a sell order
>          trade.Sell(Lots, _Symbol, price, sl, tp, "TrendConstraintExpert Sell");
>         }
>      }
>
>    // Apply trailing stop
>    for (int i = PositionsTotal() - 1; i >= 0; i--)
>      {
>       if (PositionSelect(PositionGetSymbol(i)))  // Corrected usage
>         {
>          double price = PositionGetDouble(POSITION_PRICE_OPEN);
>          double stopLoss = PositionGetDouble(POSITION_SL);
>          double current_price;
>
>          if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
>            {
>             current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
>             if (current_price - price > TrailingStop * _Point)
>               {
>                if (stopLoss < current_price - TrailingStop * _Point)
>                  {
>                   trade.PositionModify(PositionGetInteger(POSITION_TICKET), current_price - TrailingStop * _Point, PositionGetDouble(POSITION_TP));
>                  }
>               }
>            }
>          else if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
>            {
>             current_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
>             if (price - current_price > TrailingStop * _Point)
>               {
>                if (stopLoss > current_price + TrailingStop * _Point || stopLoss == 0)
>                  {
>                   trade.PositionModify(PositionGetInteger(POSITION_TICKET), current_price + TrailingStop * _Point, PositionGetDouble(POSITION_TP));
>                  }
>               }
>            }
>         }
>      }
>   }
> //+------------------------------------------------------------------+
> //HAPPY DEVELOPING!
> ```

Having reached this stage, we can proceed to test our program. I have included my testing experience below.

### Tester

To test the "Trend Constraint Expert" in MetaTrader 5's Strategy Tester, we will begin by setting up a back-test using historical data to evaluate the EA's performance. This process will allow us to simulate its trading strategy under various market conditions, helping to analyze profitability, risk management, and overall effectiveness. We have to select our desired time frame (in this case, M1 time frame), input parameters, and trading environment, and observe how well the EA adheres to the trend-following logic and RSI conditions. This test is crucial for fine-tuning the expert before considering live trading. I am, a fan of Boom 500 Index and I loved to test the EA on this beautiful pair.

![Strategy Tester](https://c.mql5.com/2/91/terminal64_RhBujdkqVt.gif)

Strategy Tester Settings: Trend Constraint Expert

![Tester Performance](https://c.mql5.com/2/91/metatester64_fJRQGfc2m2.gif)

Trend constraint Performance in the Tester

![Tester Result](https://c.mql5.com/2/91/tester.PNG)

Tester Result 01/2023-12/2023

### Conclusion

I am glad that, we have reached the conclusion of this wonderful tutorial-like discussion. Our goal was to create an independent expert advisor (EA) that doesn’t require any specific indicators to be installed. I hope this discussion has inspired you to understand the structure of EA development and provided a solid starting point. We focused on the most fundamental concepts to ensure the foundation is easy to grasp. The ability to customize the EA by inputting various factors allows for experimenting with values to find the most profitable settings.

We have successfully developed a working expert advisor based on our initial idea, and we can observe order executions in the Tester. However, there is still significant room for improvement. The Trend Constraint Expert needs further refinement, particularly in its entry conditions, to improve profitability in line with prevailing daily trends. This time, we did not include a magic number, and when testing it on a real demo account, I realized that the EA was affecting other orders that were already placed.

One advantage of this development, compared to our previous one that relied on an installed indicator, is that we now have a portable, single file to run. This doesn’t mean that our previous efforts were in vain; rather, we learned valuable lessons, and the tools from that project remain useful for future endeavors. Our goal is to continue refining the EA for the best possible outcomes as we proceed with our series.

Next, we’ll code in a magic number, enhance our entry techniques, and further increase our creativity in this development. Happy trading!

| File | Description |
| --- | --- |
| Trend Constraint Expert.mq5 | Source code. |

[Back to the Top](https://www.mql5.com/en/articles/15322#content)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15322.zip "Download all attachments in the single ZIP archive")

[Trend\_Constraint\_Expert.mq5](https://www.mql5.com/en/articles/download/15322/trend_constraint_expert.mq5 "Download Trend_Constraint_Expert.mq5")(5.76 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [From Novice to Expert: Higher Probability Signals](https://www.mql5.com/en/articles/20658)
- [From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)
- [From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)
- [From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)
- [From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)
- [The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)
- [The MQL5 Standard Library Explorer (Part 4): Custom Signal Library](https://www.mql5.com/en/articles/20266)

**[Go to discussion](https://www.mql5.com/en/forum/472300)**

![Neural Networks Made Easy (Part 83): The "Conformer" Spatio-Temporal Continuous Attention Transformer Algorithm](https://c.mql5.com/2/74/Neural_networks_are_easy_0Part_83a___LOGO.png)[Neural Networks Made Easy (Part 83): The "Conformer" Spatio-Temporal Continuous Attention Transformer Algorithm](https://www.mql5.com/en/articles/14615)

This article introduces the Conformer algorithm originally developed for the purpose of weather forecasting, which in terms of variability and capriciousness can be compared to financial markets. Conformer is a complex method. It combines the advantages of attention models and ordinary differential equations.

![Developing a multi-currency Expert Advisor (Part 8): Load testing and handling a new bar](https://c.mql5.com/2/75/Developing_a_multi-currency_advisor_8Part_8f_Conducting_load_testing____LOGO.png)[Developing a multi-currency Expert Advisor (Part 8): Load testing and handling a new bar](https://www.mql5.com/en/articles/14574)

As we progressed, we used more and more simultaneously running instances of trading strategies in one EA. Let's try to figure out how many instances we can get to before we hit resource limitations.

![Brain Storm Optimization algorithm (Part II): Multimodality](https://c.mql5.com/2/75/Brain_Storm_Optimization_ePart_Ie_____LOGO_2.png)[Brain Storm Optimization algorithm (Part II): Multimodality](https://www.mql5.com/en/articles/14622)

In the second part of the article, we will move on to the practical implementation of the BSO algorithm, conduct tests on test functions and compare the efficiency of BSO with other optimization methods.

![Example of Causality Network Analysis (CNA) and Vector Auto-Regression Model for Market Event Prediction](https://c.mql5.com/2/91/Vector_Auto-Regression_Model_for_Market_Event_Prediction___LOGO.png)[Example of Causality Network Analysis (CNA) and Vector Auto-Regression Model for Market Event Prediction](https://www.mql5.com/en/articles/15665)

This article presents a comprehensive guide to implementing a sophisticated trading system using Causality Network Analysis (CNA) and Vector Autoregression (VAR) in MQL5. It covers the theoretical background of these methods, provides detailed explanations of key functions in the trading algorithm, and includes example code for implementation.

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/15322&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049349461186030051)

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