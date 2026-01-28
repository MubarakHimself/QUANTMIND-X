---
title: Automating Trading Strategies in MQL5 (Part 11): Developing a Multi-Level Grid Trading System
url: https://www.mql5.com/en/articles/17350
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:27:22.679083
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/17350&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049145828196591047)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 10)](https://www.mql5.com/en/articles/17247), we developed an Expert Advisor to automate the [Trend Flat Momentum Strategy](https://www.mql5.com/go?link=https://www.learn-forextrading.org/2017/10/trend-flat-momentum-trading-system.html%23%3a%7e%3atext%3dTrend%2520flat%2520momentum%2520is%2520a%2520trend%2520following%2520system%2cis%2520designed%2520for%2520works%2520a%2520higher%2520time%2520frame. "https://www.learn-forextrading.org/2017/10/trend-flat-momentum-trading-system.html#:~:text=Trend%20flat%20momentum%20is%20a%20trend%20following%20system,is%20designed%20for%20works%20a%20higher%20time%20frame.") using a blend of moving averages and momentum filters in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5). Now, in Part 11, we focus on building a multi-level grid trading system that leverages a layered grid approach to capitalize on market fluctuations. We will structure the article via the following topics:

1. [Introduction](https://www.mql5.com/en/articles/17350#para1)
2. [Understanding the Architecture of a Multi-Level Grid System](https://www.mql5.com/en/articles/17350#para2)
3. [Implementation in MQL5](https://www.mql5.com/en/articles/17350#para3)
4. [Backtesting](https://www.mql5.com/en/articles/17350#para4)
5. [Conclusion](https://www.mql5.com/en/articles/17350#para5)

By the end of this article, you'll have a comprehensive understanding and a fully functional program ready for live trading. Let’s dive in!

### Understanding the Architecture of a Multi-Level Grid System

A multi-level grid trading system is a structured approach that capitalizes on market volatility by placing a series of buy and sell orders at predetermined intervals across a range of price levels. This strategy we're about to implement isn’t about predicting the market’s direction but rather about profiting from the natural flow of prices, capturing gains whether the market moves up, down, or sideways.

Building on this concept, our program will implement the multi-level grid strategy through a modular design that separates signal detection, order execution, and risk management. In our system development, we will first initialize key parameters—such as moving averages for identifying trade signals—and set up a basket structure that encapsulates trade details like initial lot sizes, grid spacing, and take-profit levels.

As the market evolves, the program will monitor price movements to trigger new trades and manage existing positions, adding orders at each grid level based on predefined conditions and dynamically adjusting risk parameters. The architecture also will include functions for recalculating break-even points, modifying take-profit targets, and closing positions when profit targets or risk thresholds are met. This structured plan will not only organize the program into distinct, manageable components but also ensure that every layer of the grid contributes to a cohesive, risk-managed trading strategy ready for robust backtesting and trading deployment. In a nutshell, this is what the architecture will look like.

![GRIDS ARCHITECTURE](https://c.mql5.com/2/122/Screenshot_2025-03-02_121217.png)

### Implementation in MQL5

[To create the program in MQL5, open the](https://www.mql5.com/en/articles/15927#para1) [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Indicators folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some metadata and [global variables](https://www.mql5.com/en/docs/basis/variables/global) that we will use throughout the program.

```
//+------------------------------------------------------------------+
//|                        Copyright 2025, Forex Algo-Trader, Allan. |
//|                                 "https://t.me/Forex_Algo_Trader" |
//+------------------------------------------------------------------+
#property copyright "Forex Algo-Trader, Allan"
#property link      "https://t.me/Forex_Algo_Trader"
#property version   "1.00"
#property description "This EA trades multiple signals with grid strategy using baskets"
#property strict

#include <Trade/Trade.mqh> //--- Includes the standard trading library for executing trades
CTrade obj_Trade; //--- Instantiates the CTrade object used for managing trade operations

//--- Closure Mode Enumeration and Inputs
enum ClosureMode {
   CLOSE_BY_PROFIT,      //--- Use total profit (in currency) to close positions
   CLOSE_BY_POINTS       //--- Use points threshold from breakeven to close positions
};

input group "General EA Settings"
input ClosureMode closureMode = CLOSE_BY_POINTS;
input double inpLotSize = 0.01;
input long inpMagicNo = 1234567;
input int inpTp_Points = 100;
input int inpGridSize = 100;
input double inpMultiplier = 2.0;
input int inpBreakevenPts = 50;
input int maxBaskets = 5;

input group "MA Indicator Settings" //--- Begins the input group for Moving Average indicator settings
input int inpMAPeriod = 21;                         //--- Period used for the Moving Average calculation
```

Here, we establish the foundational components of our program, ensuring seamless trade execution and strategic position management. We begin by including the "Trade/Trade.mqh" library, which grants access to essential trade execution functions. To facilitate trade operations, we instantiate the "CTrade" object as "obj\_Trade", allowing us to place, modify, and close orders efficiently within our automated strategy.

We define the "ClosureMode" [enumeration](https://www.mql5.com/en/book/basis/builtin_types/enums) to provide flexibility in managing trade exits. The program can operate in two modes: "CLOSE\_BY\_PROFIT", which triggers closure when total accumulated profit reaches a specified threshold in account currency, and "CLOSE\_BY\_POINTS", which closes positions based on a predefined distance from the breakeven level. This ensures that the user can dynamically adjust its exit strategy based on market behavior and risk tolerance.

Next, we introduce a structured [input](https://www.mql5.com/en/docs/basis/variables/inputvariables) section under "General EA Settings" to allow user-defined customization of the trading strategy. We specify "inpLotSize" to control the initial trade volume and use "inpMagicNo" to uniquely identify the EA's trades, preventing conflicts with other active strategies. For grid-based execution, we set "inpTp\_Points" to determine the take profit level per trade, while "inpGridSize" defines the spacing between successive grid orders. The "inpMultiplier" parameter scales trade sizes progressively, implementing an adaptive grid expansion to maximize profit potential while managing risk exposure. To further refine risk control, we configure "inpBreakevenPts", which moves trades to breakeven after a certain threshold, and "maxBaskets", which limits the number of independent grid structures the EA can manage simultaneously.

To enhance trade filtering, we incorporate a Moving Average mechanism under "MA Indicator Settings". Here, we define "inpMAPeriod", which determines the number of periods used to compute the Moving Average. This helps align grid trading with prevailing market trends, filtering out unfavorable conditions and ensuring that trade entries align with broader market momentum. Next, since we will need to handle many signal instances, we can define a basket structure.

```
//--- Basket Structure
struct BasketInfo {
   int basketId;            //--- Unique basket identifier (e.g., 1, 2, 3...)
   long magic;              //--- Unique magic number for this basket to differentiate its trades
   int direction;           //--- Direction of the basket: POSITION_TYPE_BUY or POSITION_TYPE_SELL
   double initialLotSize;   //--- The initial lot size assigned to the basket
   double currentLotSize;   //--- The current lot size for subsequent grid trades
   double gridSize;         //--- The next grid level price for the basket
   double takeProfit;       //--- The current take-profit price for the basket
   datetime signalTime;     //--- Timestamp of the signal to avoid duplicate trade entries
};
```

Here, we define the "BasketInfo" [structure](https://www.mql5.com/en/docs/basis/types/classes) to organize and manage each grid basket independently. We assign a unique "basketId" to track every basket and use "magic" to ensure that our trades remain distinct from others. We determine the trade direction with "direction," deciding whether we are executing a buy or sell strategy.

We set "initialLotSize" for the first trade in the basket, while "currentLotSize" adjusts dynamically for subsequent trades. We use "gridSize" to establish the spacing between trades and "takeProfit" to define our profit target. To prevent duplicate entries, we track the signal's timing using "signalTime." We then can declare a storage array using the defined structure and some initial global variables.

```
BasketInfo baskets[];       //--- Dynamic array to store active basket information
int nextBasketId = 1;       //--- Counter for assigning unique IDs to new baskets
long baseMagic = inpMagicNo;//--- Base magic number obtained from user input
double takeProfitPts = inpTp_Points * _Point; //--- Convert take profit points into price units
double gridSize_Spacing = inpGridSize * _Point; //--- Convert grid size spacing from points into price units
double profitTotal_inCurrency = 100; //--- Target profit in account currency for closing positions

//--- Global Variables
int totalBars = 0;          //--- Stores the total number of bars processed so far
int handle;                 //--- Handle for the Moving Average indicator
double maData[];            //--- Array to store Moving Average indicator data
```

We use the [dynamic array](https://www.mql5.com/en/docs/basis/types/dynamic_array) "baskets\[\]" to store active basket information, ensuring we can track multiple positions efficiently. The variable "nextBasketId" assigns unique identifiers to each new basket, while "baseMagic" ensures that all trades within the system are distinguishable using the user-defined magic number. We convert user inputs into price units by multiplying "inpTp\_Points" and "inpGridSize" by "\_Point," allowing precise control over "takeProfitPts" and "gridSize\_Spacing." The variable "profitTotal\_inCurrency" defines the profit threshold required to close all positions when using a currency-based closure mode.

For technical analysis, we initialize "totalBars" to track the number of price bars processed, "handle" to store the Moving Average indicator handle, and "maData\[\]" as an array for storing the computed Moving Average values. With that, we can define some arbitrary function prototypes that we will use throughout the program when needed.

```
//--- Function Prototypes
void InitializeBaskets(); //--- Prototype for basket initialization function (if used)
void CheckAndCloseProfitTargets(); //--- Prototype to check and close positions if profit target is reached
void CheckForNewSignal(double ask, double bid); //--- Prototype to check for new trading signals based on price
bool ExecuteInitialTrade(int basketIdx, double ask, double bid, int direction); //--- Prototype to execute the initial trade for a basket
void ManageGridPositions(int basketIdx, double ask, double bid); //--- Prototype to manage and add grid positions for an active basket
void UpdateMovingAverage(); //--- Prototype to update the Moving Average indicator data
bool IsNewBar(); //--- Prototype to check whether a new bar has formed
double CalculateBreakevenPrice(int basketId); //--- Prototype to calculate the weighted breakeven price for a basket
void CheckBreakevenClose(int basketIdx, double ask, double bid); //--- Prototype to check and close positions based on breakeven criteria
void CloseBasketPositions(int basketId); //--- Prototype to close all positions within a basket
string GetPositionComment(int basketId, bool isInitial); //--- Prototype to generate a comment for a position based on basket and trade type
int CountBasketPositions(int basketId); //--- Prototype to count the number of open positions in a basket
```

Here, we define function prototypes that outline the core operations of our multi-level grid trading system. These functions will ensure modularity, allowing us to structure trade execution, position management, and risk handling efficiently. We begin with "InitializeBaskets()", which prepares the system for tracking active baskets. The function "CheckAndCloseProfitTargets()" ensures that positions are closed once predefined profit conditions are met. To detect trade opportunities, "CheckForNewSignal()" evaluates price levels to determine if a new trading signal should be executed.

The function "ExecuteInitialTrade()" will manage the first trade within a basket, while "ManageGridPositions()" will ensure that grid levels are systematically expanded as the market moves. "UpdateMovingAverage()" retrieves and processes the Moving Average indicator data to support signal generation. For trade management, "IsNewBar()" helps optimize execution by ensuring actions are performed only on fresh price data. "CalculateBreakevenPrice()" computes the weighted breakeven price for a basket, while "CheckBreakevenClose()" determines whether conditions are met to exit positions based on breakeven criteria.

To manage basket positions, "CloseBasketPositions()" facilitates controlled exits, ensuring all positions within a basket are closed when required. "GetPositionComment()" provides structured trade annotations, improving trade tracking, and "CountBasketPositions()" helps monitor the number of active positions within a basket, ensuring the system operates within defined risk limits.

We can now start by initializing the moving average since we will use it solely for signal generation.

```
//+------------------------------------------------------------------+
//--- Expert initialization function
//+------------------------------------------------------------------+
int OnInit() {
   handle = iMA(_Symbol, _Period, inpMAPeriod, 0, MODE_SMA, PRICE_CLOSE); //--- Initialize the Moving Average indicator with specified period and parameters
   if(handle == INVALID_HANDLE) {
      Print("ERROR: Unable to initialize Moving Average indicator!"); //--- Log error if indicator initialization fails
      return(INIT_FAILED); //--- Terminate initialization with a failure code
   }
   ArraySetAsSeries(maData, true); //--- Set the moving average data array as a time series (newest data at index 0)
   ArrayResize(baskets, 0); //--- Initialize the baskets array as empty at startup
   obj_Trade.SetExpertMagicNumber(baseMagic); //--- Set the default magic number for trade operations
   return(INIT_SUCCEEDED); //--- Signal that initialization completed successfully
}
```

On the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, we start by initializing the Moving Average indicator using the [iMA()](https://www.mql5.com/en/docs/indicators/ima) function, where we apply the specified period and parameters to retrieve trend-based data. If the handle is invalid ( [INVALID\_HANDLE](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants)), we log an error message and terminate the initialization process with [INIT\_FAILED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode) to prevent the EA from running with missing data.

Next, we configure the moving average data array using the [ArraySetAsSeries](https://www.mql5.com/en/docs/array/arraysetasseries) function, ensuring that the most recent values are stored at index 0 for efficient access. We then resize the "baskets" array to zero, preparing it for dynamic allocation as new trades are opened. Lastly, we assign the base magic number to the trading object using the "SetExpertMagicNumber()" method, allowing the EA to track and manage trades with a unique identifier. If all components are successfully initialized, we return [INIT\_SUCCEEDED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode) to confirm that the EA is ready to begin execution.

Since we stored data, we can free the resources when we no longer need the program on the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) event handler, by calling the [IndicatorRelease](https://www.mql5.com/en/docs/series/indicatorrelease) function.

```
//+------------------------------------------------------------------+
//--- Expert deinitialization function
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   IndicatorRelease(handle); //--- Release the indicator handle to free up resources when the EA is removed
}
```

We can then proceed to process the data on every tick on the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler. However, we want to run the program once per bar, so we will need to define a function for that.

```
//+------------------------------------------------------------------+
//--- Expert tick function
//+------------------------------------------------------------------+
void OnTick() {
   if(IsNewBar()) { //--- Execute logic only when a new bar is detected

   }
}
```

The function's prototype is as below.

```
//+------------------------------------------------------------------+
//--- Check for New Bar
//+------------------------------------------------------------------+
bool IsNewBar() {
   int bars = iBars(_Symbol, _Period); //--- Get the current number of bars on the chart for the symbol and period
   if(bars > totalBars) { //--- Compare the current number of bars with the previously stored total
      totalBars = bars; //--- Update the stored bar count to the new value
      return true; //--- Return true to indicate a new bar has formed
   }
   return false; //--- Return false if no new bar has been detected
}
```

Here, we define the "IsNewBar()" function, which checks whether a new bar has formed on the chart, which is essential for ensuring that our EA processes new price data only when a fresh bar appears, preventing unnecessary recalculations. We begin by retrieving the current number of bars on the chart using the [iBars](https://www.mql5.com/en/docs/series/ibars) function, which provides the total count of historical bars for the active symbol and timeframe. We then compare this value with the "totalBars" variable, which stores the previously recorded bar count.

If the current bar count is greater than the value stored in the "totalBars" variable, it means a new bar has appeared. In this case, we update the "totalBars" variable using the new count and return "true", signaling that the EA should proceed with bar-based calculations or trade logic. If no new bar is detected, the function returns "false", ensuring that the EA does not perform redundant operations on the same bar.

Now once we detect a new bar, we need to retrieve the moving average data for further processing. For this, we use a function.

```
//+------------------------------------------------------------------+
//--- Update Moving Average
//+------------------------------------------------------------------+
void UpdateMovingAverage() {
   if(CopyBuffer(handle, 0, 1, 3, maData) < 0) { //--- Copy the latest 3 values from the Moving Average indicator buffer into the maData array
      Print("Error: Unable to update Moving Average data."); //--- Log an error if copying the indicator data fails
   }
}
```

For the "UpdateMovingAverage()" function, which ensures that our EA retrieves the latest values from the Moving Average indicator, we use the [CopyBuffer()](https://www.mql5.com/en/docs/series/copybuffer) function to extract the most recent three values from the Moving Average indicator buffer and store them in the "maData" array. The parameters specify the indicator handle ("handle"), buffer index (0 for the main line), starting position (1 to skip the current forming bar), number of values (3), and the target array ("maData").

If we fail to retrieve the data, we log an error message using the [Print()](https://www.mql5.com/en/docs/common/print) function to alert us of potential issues with indicator data retrieval, safeguarding the EA against incomplete or missing moving average values, and ensuring reliability in decision-making. We then can call the function and use the retrieved data for signal generation.

```
UpdateMovingAverage(); //--- Update the Moving Average data for the current bar
double ask = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK), _Digits); //--- Get and normalize the current ask price
double bid = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID), _Digits); //--- Get and normalize the current bid price

//--- Check for new signals and create baskets accordingly
CheckForNewSignal(ask, bid);
```

After retrieving the indicator data, we retrieve the current ask and bid prices using the [SymbolInfoDouble()](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) function with the [SYMBOL\_ASK](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double) and [SYMBOL\_BID](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double) constants, respectively. Since price values often have multiple decimal places, we use the [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble) function with the [\_Digits](https://www.mql5.com/en/docs/predefined/_digits) parameter to ensure that they are formatted correctly according to the symbol’s price precision.

Finally, we call the "CheckForNewSignal()" function, passing in the normalized ask and bid prices. Here is the function's code snippet.

```
//+------------------------------------------------------------------+
//--- Check for New Crossover Signal
//+------------------------------------------------------------------+
void CheckForNewSignal(double ask, double bid) {
   double close1 = iClose(_Symbol, _Period, 1); //--- Retrieve the close price of the previous bar
   double close2 = iClose(_Symbol, _Period, 2); //--- Retrieve the close price of the bar before the previous one
   datetime currentBarTime = iTime(_Symbol, _Period, 1); //--- Get the time of the current bar

   if(ArraySize(baskets) >= maxBaskets) return; //--- Exit if the maximum allowed baskets are already active

   //--- Buy signal: current bar closes above the MA while the previous closed below it
   if(close1 > maData[1] && close2 < maData[1]) {
      //--- Check if this signal was already processed by comparing signal times in existing baskets
      for(int i = 0; i < ArraySize(baskets); i++) {
         if(baskets[i].signalTime == currentBarTime) return; //--- Signal already acted upon; exit the function
      }
      int basketIdx = ArraySize(baskets); //--- Index for the new basket equals the current array size
      ArrayResize(baskets, basketIdx + 1); //--- Increase the size of the baskets array to add a new basket
      if (ExecuteInitialTrade(basketIdx, ask, bid, POSITION_TYPE_BUY)){
         baskets[basketIdx].signalTime = currentBarTime; //--- Record the time of the signal after a successful trade
      }
   }
   //--- Sell signal: current bar closes below the MA while the previous closed above it
   else if(close1 < maData[1] && close2 > maData[1]) {
      //--- Check for duplicate signals by verifying the signal time in active baskets
      for(int i = 0; i < ArraySize(baskets); i++) {
         if(baskets[i].signalTime == currentBarTime) return; //--- Signal already acted upon; exit the function
      }
      int basketIdx = ArraySize(baskets); //--- Determine the index for the new basket
      ArrayResize(baskets, basketIdx + 1); //--- Resize the baskets array to accommodate the new basket
      if (ExecuteInitialTrade(basketIdx, ask, bid, POSITION_TYPE_SELL)){
         baskets[basketIdx].signalTime = currentBarTime; //--- Record the signal time for the new sell basket
      }
   }
}
```

For the "CheckForNewSignal()" function, we first retrieve the closing prices of the previous two bars using the [iClose()](https://www.mql5.com/en/docs/series/iclose) function. This helps us determine if a crossover has occurred. We also use the [iTime()](https://www.mql5.com/en/docs/series/itime) function to get the timestamp of the most recent bar, ensuring that we do not process the same signal multiple times.

Before proceeding, we check if the number of active baskets has reached the "maxBaskets" limit. If so, the function returns to prevent excessive trade stacking. For buy signals, we check whether the most recent closing price is above the Moving Average while the previous closing price was below it. If this crossover condition is met, we iterate through the existing baskets to ensure that the same signal has not been processed already. If the signal is new, we increase the "baskets" array size, store the new basket at the next available index, and call the "ExecuteInitialTrade()" function with a [POSITION\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type) order. If the trade is executed successfully, we record the signal time to prevent duplicate entries.

Similarly, for sell signals, we check whether the most recent closing price is below the Moving Average while the previous closing price was above it. If this condition is met and no duplicate signal is found, we expand the "baskets" array, execute an initial sell trade using the "ExecuteInitialTrade()" function with a [POSITION\_TYPE\_SELL](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type) order, and store the signal time to maintain uniqueness. The function to execute the trades is as follows.

```
//+------------------------------------------------------------------+
//--- Execute Initial Trade
//+------------------------------------------------------------------+
bool ExecuteInitialTrade(int basketIdx, double ask, double bid, int direction) {
   baskets[basketIdx].basketId = nextBasketId++; //--- Assign a unique basket ID and increment the counter
   baskets[basketIdx].magic = baseMagic + baskets[basketIdx].basketId * 10000; //--- Calculate a unique magic number for the basket
   baskets[basketIdx].initialLotSize = inpLotSize; //--- Set the initial lot size for the basket from input
   baskets[basketIdx].currentLotSize = inpLotSize; //--- Initialize current lot size to the same as the initial lot size
   baskets[basketIdx].direction = direction; //--- Set the trade direction (buy or sell) for the basket
   bool isTradeExecuted = false; //--- Initialize flag to track if the trade was successfully executed
   string comment = GetPositionComment(baskets[basketIdx].basketId, true); //--- Generate a comment string indicating an initial trade
   obj_Trade.SetExpertMagicNumber(baskets[basketIdx].magic); //--- Set the trade object's magic number to the basket's unique value

   if(direction == POSITION_TYPE_BUY) {
      baskets[basketIdx].gridSize = ask - gridSize_Spacing; //--- Set the grid level for subsequent buy orders below the current ask price
      baskets[basketIdx].takeProfit = ask + takeProfitPts; //--- Calculate the take profit level for the buy order
      if(obj_Trade.Buy(baskets[basketIdx].currentLotSize, _Symbol, ask, 0, baskets[basketIdx].takeProfit, comment)) {
         Print("Basket ", baskets[basketIdx].basketId, ": Initial BUY at ", ask, " | Magic: ", baskets[basketIdx].magic); //--- Log the successful buy order details
         isTradeExecuted = true; //--- Mark the trade as executed successfully
      } else {
         Print("Basket ", baskets[basketIdx].basketId, ": Initial BUY failed, error: ", GetLastError()); //--- Log the error if the buy order fails
         ArrayResize(baskets, ArraySize(baskets) - 1); //--- Remove the basket if trade execution fails
      }
   } else if(direction == POSITION_TYPE_SELL) {
      baskets[basketIdx].gridSize = bid + gridSize_Spacing; //--- Set the grid level for subsequent sell orders above the current bid price
      baskets[basketIdx].takeProfit = bid - takeProfitPts; //--- Calculate the take profit level for the sell order
      if(obj_Trade.Sell(baskets[basketIdx].currentLotSize, _Symbol, bid, 0, baskets[basketIdx].takeProfit, comment)) {
         Print("Basket ", baskets[basketIdx].basketId, ": Initial SELL at ", bid, " | Magic: ", baskets[basketIdx].magic); //--- Log the successful sell order details
         isTradeExecuted = true; //--- Mark the trade as executed successfully
      } else {
         Print("Basket ", baskets[basketIdx].basketId, ": Initial SELL failed, error: ", GetLastError()); //--- Log the error if the sell order fails
         ArrayResize(baskets, ArraySize(baskets) - 1); //--- Remove the basket if trade execution fails
      }
   }
   return (isTradeExecuted); //--- Return the status of the trade execution
}
```

We define the "ExecuteInitialTrade()" function to ensure each basket has a unique identifier, assigns a distinct magic number, and initializes key trading parameters before placing the order. First, we assign a "basketId" by incrementing the "nextBasketId" variable. We then generate a unique magic number for the basket by adding a scaled offset to the "baseMagic" value, ensuring that each basket operates independently. The initial and current lot sizes are both set to "inpLotSize" to establish the base trade size for this basket. The "direction" is stored to differentiate between buy and sell baskets.

To ensure trades are identifiable, we call the "GetPositionComment()" function to generate a descriptive comment, and we apply the basket's magic number to the trade object using the "SetExpertMagicNumber()" method. The function is defined as below, where we use the [StringFormat](https://www.mql5.com/en/docs/convert/stringformat) function to get the comment via a [ternary operator](https://www.mql5.com/en/docs/basis/operators/Ternary).

```
//+------------------------------------------------------------------+
//--- Generate Position Comment
//+------------------------------------------------------------------+
string GetPositionComment(int basketId, bool isInitial) {
   return StringFormat("Basket_%d_%s", basketId, isInitial ? "Initial" : "Grid"); //--- Generate a standardized comment string for a position indicating basket ID and trade type
}
```

If the direction is [POSITION\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type), we calculate the grid level by subtracting "gridSize\_Spacing" from the asking price and determine the take profit level by adding "takeProfitPts" to the asking price. We then use the "Buy()" function from the "CTrade" class to place the order. If successful, we log the trade details using the [Print()](https://www.mql5.com/en/docs/common/print) function and mark the trade as executed. If the trade fails, we log the error using the [GetLastError()](https://www.mql5.com/en/docs/check/getlasterror) function and use the [ArrayResize()](https://www.mql5.com/en/docs/array/arrayresize) function to reduce the size of the "baskets" array, removing the failed basket.

For a sell trade ( [POSITION\_TYPE\_SELL](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type)), we calculate the grid level by adding "gridSize\_Spacing" to the bid price and determine the take profit level by subtracting "takeProfitPts" from the bid price. The trade is executed using the "Sell()" function. As with buy trades, successful execution is logged using the "Print()" function, and failure results in an error log with [GetLastError](https://www.mql5.com/en/docs/check/getlasterror), followed by resizing the "baskets" array using "ArrayResize()" to remove the failed basket.

Before executing any trade, the function ensures the array has enough space by calling "ArrayResize()" to increase its size. Finally, the function returns "true" if the trade was executed successfully and "false" otherwise. Upon running the program, we have the following outcome.

![CONFIRMED INITIAL POSITIONS IN BASKETS](https://c.mql5.com/2/122/Screenshot_2025-03-02_011027.png)

From the image, we can see that we have confirmed initial positions as per the baskets or signals realized. We then need to move on to managing these positions by managing each basket individually. To achieve that, we use a [for loop](https://www.mql5.com/en/docs/basis/operators/for) for iteration.

```
//--- Loop through all active baskets to manage grid positions and potential closures
for(int i = 0; i < ArraySize(baskets); i++) {
   ManageGridPositions(i, ask, bid); //--- Manage grid trading for the current basket
}
```

Here, we iterate through all active baskets using a [for](https://www.mql5.com/en/docs/basis/operators/for) loop, ensuring that each basket is managed accordingly. The [ArraySize](https://www.mql5.com/en/docs/array/arraysize) function determines the current number of baskets in the "baskets" array, providing the loop's upper limit. This ensures that we process all existing baskets without exceeding the array boundaries. For each basket, we call the "ManageGridPositions()" function, passing the basket's index along with the normalized "ask" and "bid" prices. The function is as below.

```
//+------------------------------------------------------------------+
//--- Manage Grid Positions
//+------------------------------------------------------------------+
void ManageGridPositions(int basketIdx, double ask, double bid) {
   bool newPositionOpened = false; //--- Flag to track if a new grid position has been opened
   string comment = GetPositionComment(baskets[basketIdx].basketId, false); //--- Generate a comment for grid trades in this basket
   obj_Trade.SetExpertMagicNumber(baskets[basketIdx].magic); //--- Ensure the trade object uses the basket's unique magic number

   if(baskets[basketIdx].direction == POSITION_TYPE_BUY) {
      if(ask <= baskets[basketIdx].gridSize) { //--- Check if the ask price has reached the grid level for a buy order
         baskets[basketIdx].currentLotSize *= inpMultiplier; //--- Increase the lot size based on the defined multiplier
         if(obj_Trade.Buy(baskets[basketIdx].currentLotSize, _Symbol, ask, 0, baskets[basketIdx].takeProfit, comment)) {
            newPositionOpened = true; //--- Set flag if the grid buy order is successfully executed
            Print("Basket ", baskets[basketIdx].basketId, ": Grid BUY at ", ask); //--- Log the grid buy execution details
            baskets[basketIdx].gridSize = ask - gridSize_Spacing; //--- Adjust the grid level for the next potential buy order
         } else {
            Print("Basket ", baskets[basketIdx].basketId, ": Grid BUY failed, error: ", GetLastError()); //--- Log an error if the grid buy order fails
         }
      }
   } else if(baskets[basketIdx].direction == POSITION_TYPE_SELL) {
      if(bid >= baskets[basketIdx].gridSize) { //--- Check if the bid price has reached the grid level for a sell order
         baskets[basketIdx].currentLotSize *= inpMultiplier; //--- Increase the lot size based on the multiplier for grid orders
         if(obj_Trade.Sell(baskets[basketIdx].currentLotSize, _Symbol, bid, 0, baskets[basketIdx].takeProfit, comment)) {
            newPositionOpened = true; //--- Set flag if the grid sell order is successfully executed
            Print("Basket ", baskets[basketIdx].basketId, ": Grid SELL at ", bid); //--- Log the grid sell execution details
            baskets[basketIdx].gridSize = bid + gridSize_Spacing; //--- Adjust the grid level for the next potential sell order
         } else {
            Print("Basket ", baskets[basketIdx].basketId, ": Grid SELL failed, error: ", GetLastError()); //--- Log an error if the grid sell order fails
         }
      }
   }

   //--- If a new grid position was opened and there are multiple positions, adjust the take profit to breakeven
   if(newPositionOpened && CountBasketPositions(baskets[basketIdx].basketId) > 1) {
      double breakevenPrice = CalculateBreakevenPrice(baskets[basketIdx].basketId); //--- Calculate the weighted breakeven price for the basket
      double newTP = (baskets[basketIdx].direction == POSITION_TYPE_BUY) ?
                     breakevenPrice + (inpBreakevenPts * _Point) : //--- Set new TP for buy positions
                     breakevenPrice - (inpBreakevenPts * _Point);  //--- Set new TP for sell positions
      baskets[basketIdx].takeProfit = newTP; //--- Update the basket's take profit level with the new value
      for(int j = PositionsTotal() - 1; j >= 0; j--) { //--- Loop through all open positions to update TP where necessary
         ulong ticket = PositionGetTicket(j); //--- Get the ticket number for the current position
         if(PositionSelectByTicket(ticket) &&
            PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == baskets[basketIdx].magic) { //--- Identify positions that belong to the current basket
            if(!obj_Trade.PositionModify(ticket, 0, newTP)) { //--- Attempt to modify the position's take profit level
               Print("Basket ", baskets[basketIdx].basketId, ": Failed to modify TP for ticket ", ticket); //--- Log error if modifying TP fails
            }
         }
      }
      Print("Basket ", baskets[basketIdx].basketId, ": Breakeven = ", breakevenPrice, ", New TP = ", newTP); //--- Log the new breakeven and take profit levels
   }
}
```

Here, we implement the "ManageGridPositions()" function to dynamically manage grid-based trading within each active basket. We ensure that new grid positions are executed at the correct price levels and that profit adjustments are made when needed. We begin by initializing the "newPositionOpened" flag to track whether a new grid trade has been executed. Using the "GetPositionComment()" function, we generate a comment string specific to the trade type (initial or grid). We then call the "SetExpertMagicNumber()" function to assign the basket's unique magic number, ensuring that all trades within the basket are properly tracked.

For buy baskets, we check whether the asking price has dropped to or below the "gridSize" threshold. If this condition is met, we adjust the lot size by multiplying "currentLotSize" with the "inpMultiplier" input parameter. Next, we attempt to place a buy order using the "Buy()" method from the "obj\_Trade" trade object. If the trade executes successfully, we update "gridSize" by subtracting "gridSize\_Spacing", ensuring the next buy trade is positioned correctly. We also log the successful execution using the [Print()](https://www.mql5.com/en/docs/common/print) function. If the buy order fails, we retrieve and log the error using the [GetLastError()](https://www.mql5.com/en/docs/check/getlasterror) function.

For sell baskets, we follow a similar process but instead, check whether the bid price has risen to or above the gridSize threshold. If this condition is met, we adjust the lot size by applying the "inpMultiplier" to "currentLotSize". We then execute a sell order using the "Sell()" function, updating the gridSize by adding "gridSize\_Spacing" to define the next sell level. If the order is successful, we log the details with "Print()", and if it fails, we log the error using "GetLastError()".

If a new grid position is opened and the basket now holds multiple trades, we proceed to adjust the take profit to a breakeven level. We first determine the breakeven price by calling the "CalculateBreakevenPrice()" function. We then compute a new take profit level based on the direction of the basket:

- For buy baskets, the take profit is set by adding "inpBreakevenPts" (converted into price points) to the breakeven price.
- For sell baskets, the take profit is adjusted by subtracting "inpBreakevenPts" from the breakeven price.

Next, we loop through all open positions using the [PositionsTotal()](https://www.mql5.com/en/docs/trading/positionstotal) function, retrieving each position's ticket number with [PositionGetTicket()](https://www.mql5.com/en/docs/trading/positiongetticket). We use [PositionSelectByTicket()](https://www.mql5.com/en/docs/trading/positionselectbyticket) to select the position and verify its symbol with the "PositionGetString" function. We also ensure that the position belongs to the correct basket by checking its magic number with the "POSITION\_MAGIC" parameter. Once verified, we attempt to modify its take profit using the "PositionModify()" method. If this modification fails, we log the error.

Finally, we log the newly calculated breakeven price and updated take-profit level using the [Print()](https://www.mql5.com/en/docs/common/print) function. This ensures that the grid trading strategy adapts dynamically while maintaining efficient exit points. The function responsible for calculating the averaging price is as follows.

```
//+------------------------------------------------------------------+
//--- Calculate Weighted Breakeven Price for a Basket
//+------------------------------------------------------------------+
double CalculateBreakevenPrice(int basketId) {
   double weightedSum = 0.0; //--- Initialize sum for weighted prices
   double totalLots = 0.0;   //--- Initialize sum for total lot sizes
   for(int i = 0; i < PositionsTotal(); i++) { //--- Loop over all open positions
      ulong ticket = PositionGetTicket(i); //--- Retrieve the ticket for the current position
      if(PositionSelectByTicket(ticket) && PositionGetString(POSITION_SYMBOL) == _Symbol &&
         StringFind(PositionGetString(POSITION_COMMENT), "Basket_" + IntegerToString(basketId)) >= 0) { //--- Check if the position belongs to the specified basket
         double lot = PositionGetDouble(POSITION_VOLUME); //--- Get the lot size of the position
         double openPrice = PositionGetDouble(POSITION_PRICE_OPEN); //--- Get the open price of the position
         weightedSum += openPrice * lot; //--- Add the weighted price to the sum
         totalLots += lot; //--- Add the lot size to the total lots
      }
   }
   return (totalLots > 0) ? (weightedSum / totalLots) : 0; //--- Return the weighted average price (breakeven) or 0 if no positions found
}
```

We implement the "CalculateBreakevenPrice()" function to determine the weighted breakeven price for a given basket of trades, ensuring that the take profit level can be adjusted dynamically based on the volume-weighted entry prices of all open positions within the basket. We start by initializing "weightedSum" to store the sum of weighted prices and "totalLots" to track the total lot size of all positions in the basket. We then iterate through all open positions.

For each position, we retrieve its ticket number using [PositionGetTicket()](https://www.mql5.com/en/docs/trading/positiongetticket) and select the position with [PositionSelectByTicket()](https://www.mql5.com/en/docs/trading/positionselectbyticket). We verify that the position belongs to the current trading symbol. Additionally, we check whether the position is part of the specified basket by searching for the basket ID in the comment string using the [StringFind()](https://www.mql5.com/en/docs/strings/stringfind) function. The comment must contain "Basket\_" + [IntegerToString](https://www.mql5.com/en/docs/convert/IntegerToString)(basketId) to be classified under the same basket.

Once the position is verified, we extract its lot size using " [PositionGetDouble](https://www.mql5.com/en/docs/trading/positiongetdouble)(POSITION\_VOLUME)" and its open price using [POSITION\_PRICE\_OPEN](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_double). We then multiply the open price by the lot size and add the result to "weightedSum", ensuring that larger lot sizes have a greater influence on the final breakeven price. Simultaneously, we accumulate the total lot size in "totalLots".

After looping through all positions, we compute the weighted average breakeven price by dividing "weightedSum" by "totalLots". If no positions exist in the basket ("totalLots" == 0), we return 0 to prevent zero division errors. Upon running the program, we have the following outcome.

![GRIDS OPENED IMAGE 1](https://c.mql5.com/2/122/Screenshot_2025-03-02_014126.png)

From the image, we can see that the baskets are managed independently, by opening grids and averaging the prices. For example, basket 2 has the same take profit levels of 0.68074. We can confirm this in the journal as visualized below.

![GRID POSITIONS JOURNAL](https://c.mql5.com/2/122/Screenshot_2025-03-02_014219.png)

From the image, we can see that once we open the grid buy position for basket 4, we modify the take profit as well. Now, we need to close the positions based on the modes just for security reasons, though not necessary since we have already the levels modified, as follows.

```
if(closureMode == CLOSE_BY_PROFIT) CheckAndCloseProfitTargets(); //--- If using profit target closure mode, check for profit conditions
if(closureMode == CLOSE_BY_POINTS && CountBasketPositions(baskets[i].basketId) > 1) {
   CheckBreakevenClose(i, ask, bid); //--- If using points-based closure and multiple positions exist, check breakeven conditions
}
```

Here, we manage trade closures based on the selected "closureMode". If set to "CLOSE\_BY\_PROFIT", we call "CheckAndCloseProfitTargets()" to close baskets that hit their profit targets. If set to "CLOSE\_BY\_POINTS", we ensure the basket has multiple positions using "CountBasketPositions()" before calling "CheckBreakevenClose()" to close trades at breakeven when conditions are met. The functions are as below.

```
//+------------------------------------------------------------------+
//--- Check and Close Profit Targets (for CLOSE_BY_PROFIT mode)
//+------------------------------------------------------------------+
void CheckAndCloseProfitTargets() {
   for(int i = 0; i < ArraySize(baskets); i++) { //--- Loop through each active basket
      int posCount = CountBasketPositions(baskets[i].basketId); //--- Count how many positions belong to the current basket
      if(posCount <= 1) continue; //--- Skip baskets with only one position as profit target checks apply to multiple positions
      double totalProfit = 0; //--- Initialize the total profit accumulator for the basket
      for(int j = PositionsTotal() - 1; j >= 0; j--) { //--- Loop through all open positions to sum their profits
         ulong ticket = PositionGetTicket(j); //--- Get the ticket for the current position
         if(PositionSelectByTicket(ticket) &&
            StringFind(PositionGetString(POSITION_COMMENT), "Basket_" + IntegerToString(baskets[i].basketId)) >= 0) { //--- Check if the position is part of the current basket
            totalProfit += PositionGetDouble(POSITION_PROFIT); //--- Add the position's profit to the basket's total profit
         }
      }
      if(totalProfit >= profitTotal_inCurrency) { //--- Check if the accumulated profit meets or exceeds the profit target
         Print("Basket ", baskets[i].basketId, ": Profit target reached (", totalProfit, ")"); //--- Log that the profit target has been reached for the basket
         CloseBasketPositions(baskets[i].basketId); //--- Close all positions in the basket to secure the profits
      }
   }
}
```

Here, we check and close baskets when they reach the profit target in "CLOSE\_BY\_PROFIT" mode. We loop through "baskets" and use "CountBasketPositions()" to ensure multiple positions exist. Then, we sum profits using " [PositionGetDouble](https://www.mql5.com/en/docs/trading/positiongetdouble)(POSITION\_PROFIT)" for all positions in the basket. If total profit meets or exceeds "profitTotal\_inCurrency", we log the event and call "CloseBasketPositions()" to secure the gains. The "CountBasketPositions" function is defined as below.

```
//+------------------------------------------------------------------+
//--- Count Positions in a Basket
//+------------------------------------------------------------------+
int CountBasketPositions(int basketId) {
   int count = 0; //--- Initialize the counter for positions in the basket
   for(int i = 0; i < PositionsTotal(); i++) { //--- Loop through all open positions
      ulong ticket = PositionGetTicket(i); //--- Retrieve the ticket for the current position
      if(PositionSelectByTicket(ticket) &&
         StringFind(PositionGetString(POSITION_COMMENT), "Basket_" + IntegerToString(basketId)) >= 0) { //--- Check if the position belongs to the specified basket
         count++; //--- Increment the counter if a matching position is found
      }
   }
   return count; //--- Return the total number of positions in the basket
}
```

We use the "CountBasketPositions()" function to count the positions in a specific basket. We loop through all positions, retrieve each "ticket" with the [PositionGetTicket()](https://www.mql5.com/en/docs/trading/positiongetticket) function, and check if the [POSITION\_COMMENT](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_string) contains the basket ID. If a match is found, we increment "count". Finally, we return the total number of positions in the basket. The "CloseBasketPositions()" function definition is also as follows.

```
//+------------------------------------------------------------------+
//--- Close All Positions in a Basket
//+------------------------------------------------------------------+
void CloseBasketPositions(int basketId) {
   for(int i = PositionsTotal() - 1; i >= 0; i--) { //--- Loop backwards through all open positions
      ulong ticket = PositionGetTicket(i); //--- Retrieve the ticket of the current position
      if(PositionSelectByTicket(ticket) &&
         StringFind(PositionGetString(POSITION_COMMENT), "Basket_" + IntegerToString(basketId)) >= 0) { //--- Identify if the position belongs to the specified basket
         if(obj_Trade.PositionClose(ticket)) { //--- Attempt to close the position
            Print("Basket ", basketId, ": Closed position ticket ", ticket); //--- Log the successful closure of the position
         }
      }
   }
}
```

We use the same logic to iterate via all the positions, verify them, and close them using the "PositionClose" method. Finally, we have the function responsible for forcing the closure of positions when they surpass the defined target levels.

```
//+------------------------------------------------------------------+
//--- Check Breakeven Close
//+------------------------------------------------------------------+
void CheckBreakevenClose(int basketIdx, double ask, double bid) {
   double breakevenPrice = CalculateBreakevenPrice(baskets[basketIdx].basketId); //--- Calculate the breakeven price for the basket
   if(baskets[basketIdx].direction == POSITION_TYPE_BUY) {
      if(bid >= breakevenPrice + (inpBreakevenPts * _Point)) { //--- Check if the bid price exceeds breakeven plus threshold for buy positions
         Print("Basket ", baskets[basketIdx].basketId, ": Closing BUY positions at breakeven + points"); //--- Log that breakeven condition is met for closing positions
         CloseBasketPositions(baskets[basketIdx].basketId); //--- Close all positions for the basket
      }
   } else if(baskets[basketIdx].direction == POSITION_TYPE_SELL) {
      if(ask <= breakevenPrice - (inpBreakevenPts * _Point)) { //--- Check if the ask price is below breakeven minus threshold for sell positions
         Print("Basket ", baskets[basketIdx].basketId, ": Closing SELL positions at breakeven + points"); //--- Log that breakeven condition is met for closing positions
         CloseBasketPositions(baskets[basketIdx].basketId); //--- Close all positions for the basket
      }
   }
}
```

Here, we implement breakeven-based closures using "CheckBreakevenClose()". We first determine the breakeven price with "CalculateBreakevenPrice()". If the basket is in a BUY direction and the bid price exceeds breakeven plus the defined threshold ("inpBreakevenPts \* [\_Point](https://www.mql5.com/en/docs/predefined/_point)"), we log the event and execute "CloseBasketPositions()" to lock in profits. Similarly, for SELL baskets, we check if the ask price drops below breakeven minus the threshold, triggering closure. This ensures positions are secured once price movement aligns with breakeven conditions.

Finally, since we close the positions by take-profit at first, it means that we have empty "shells" or position baskets that litter the system. So to ensure cleanup, we need to identify the empty baskets that do not contain any elements and remove them. We implement the following logic.

```
//--- Remove inactive baskets that no longer have any open positions
for(int i = ArraySize(baskets) - 1; i >= 0; i--) {
   if(CountBasketPositions(baskets[i].basketId) == 0) {
      Print("Removing inactive basket ID: ", baskets[i].basketId); //--- Log the removal of an inactive basket
      for(int j = i; j < ArraySize(baskets) - 1; j++) {
         baskets[j] = baskets[j + 1]; //--- Shift basket elements down to fill the gap
      }
      ArrayResize(baskets, ArraySize(baskets) - 1); //--- Resize the baskets array to remove the empty slot
   }
}
```

Here, we ensure that inactive baskets, which no longer contain open positions, are removed efficiently. We iterate through the "baskets" array in reverse to avoid index-shifting issues during removal. Using "CountBasketPositions()", we check if a basket has no remaining trades. If empty, we log its removal and shift subsequent elements downward to maintain the array structure. Finally, we call [ArrayResize()](https://www.mql5.com/en/docs/array/arrayresize) to adjust the array size, preventing unnecessary memory usage and ensuring that only active baskets are tracked. This approach keeps basket management efficient and prevents clutter in the system. Upon run, we have the following outcome.

![CLUTTER CLEARING](https://c.mql5.com/2/122/Screenshot_2025-03-02_021733.png)

From the image, we can see that we efficiently handle clutter removal, and we can manage the grid positions, hence achieving our objective. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, for 1 year, 2023, using the default settings, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/122/Screenshot_2025-03-02_124804.png)

Backtest report:

![REPORT](https://c.mql5.com/2/122/Screenshot_2025-03-02_124930.png)

### Conclusion

In conclusion, we have developed an MQL5 Multi-Level Grid Trading Expert Advisor that efficiently manages layered trade entries, dynamic grid adjustments, and structured recovery. By integrating scalable grid spacing, controlled lot progression, and breakeven exits, the system adapts to market fluctuations while optimizing risk and reward.

Disclaimer: This article is for educational purposes only. Trading involves significant financial risk, and market conditions can be unpredictable. Proper backtesting and risk management are essential before live deployment.

By applying these techniques, you can enhance your algorithmic trading skills and refine your grid-based strategy. Keep testing and optimizing for long-term success. Best of luck!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17350.zip "Download all attachments in the single ZIP archive")

[1.\_Multi-Level\_Grid\_System\_EA.mq5](https://www.mql5.com/en/articles/download/17350/1._multi-level_grid_system_ea.mq5 "Download 1._Multi-Level_Grid_System_EA.mq5")(24.4 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/482772)**
(4)


![johnsteed](https://c.mql5.com/avatar/avatar_na2.png)

**[johnsteed](https://www.mql5.com/en/users/johnsteed)**
\|
12 Mar 2025 at 11:49

A very good code and very fast EA!

Unfortunately there is a problem with the lot size calculation - multipliers with a decimal (like 1.3, 1.5 etc) may cause trouble with MQL order functions as the lot size gives sometimes  [error codes](https://www.mql5.com/en/articles/70 "Article: OOP in MQL5 by Example: Processing Warning and Error Codes ") 4756 when the multiplier is not 1 oder 2.

It would be too nice if the lot size calculation could be changed marginally to ensure that lot sizes are appropriately calculated for feeding into the order functions for all multiplier values.

![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
17 Mar 2025 at 19:19

**johnsteed [#](https://www.mql5.com/en/forum/482772#comment_56147243):**

A very good code and very fast EA!

Unfortunately there is a problem with the lot size calculation - multipliers with a decimal (like 1.3, 1.5 etc) may cause trouble with MQL order functions as the lot size gives sometimes  [error codes](https://www.mql5.com/en/articles/70 "Article: OOP in MQL5 by Example: Processing Warning and Error Codes ") 4756 when the multiplier is not 1 oder 2.

It would be too nice if the lot size calculation could be changed marginally to ensure that lot sizes are appropriately calculated for feeding into the order functions for all multiplier values.

Thanks for the kind feedback. Sure.

![cbkiri](https://c.mql5.com/avatar/2025/12/69423690-0109.png)

**[cbkiri](https://www.mql5.com/en/users/cbkiri)**
\|
6 Jun 2025 at 03:18

Hi,

After reading the article, found it useful and will definitely testing it out. However, seem I am not seeing or perhaps I missed out from the article on the seperation of first position TP which I believe it is also useful and sustainable for the [trading strategy](https://www.mql5.com/en/articles/3074 "Article: Comparative Analysis of 10 Trending Strategies ").

Thank You.

![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
6 Jun 2025 at 11:19

**cbkiri [#](https://www.mql5.com/en/forum/482772#comment_56881112):**

Hi,

After reading the article, found it useful and will definitely testing it out. However, seem I am not seeing or perhaps I missed out from the article on the seperation of first position TP which I believe it is also useful and sustainable for the [trading strategy](https://www.mql5.com/en/articles/3074 "Article: Comparative Analysis of 10 Trending Strategies ").

Thank You.

Sure. Thanks.

![Data Science and ML (Part 34): Time series decomposition, Breaking the stock market down to the core](https://c.mql5.com/2/124/Data_Science_and_ML_Part_34___LOGO__4.png)[Data Science and ML (Part 34): Time series decomposition, Breaking the stock market down to the core](https://www.mql5.com/en/articles/17361)

In a world overflowing with noisy and unpredictable data, identifying meaningful patterns can be challenging. In this article, we'll explore seasonal decomposition, a powerful analytical technique that helps separate data into its key components: trend, seasonal patterns, and noise. By breaking data down this way, we can uncover hidden insights and work with cleaner, more interpretable information.

![An introduction to Receiver Operating Characteristic curves](https://c.mql5.com/2/124/An_introduction_to_Receiver_Operating_Characteristic_curves___LOGO__1.png)[An introduction to Receiver Operating Characteristic curves](https://www.mql5.com/en/articles/17390)

ROC curves are graphical representations used to evaluate the performance of classifiers. Despite ROC graphs being relatively straightforward, there exist common misconceptions and pitfalls when using them in practice. This article aims to provide an introduction to ROC graphs as a tool for practitioners seeking to understand classifier performance evaluation.

![From Basic to Intermediate: Passing by Value or by Reference](https://c.mql5.com/2/90/logo-15345.png)[From Basic to Intermediate: Passing by Value or by Reference](https://www.mql5.com/en/articles/15345)

In this article, we will practically understand the difference between passing by value and passing by reference. Although this seems like something simple and common and not causing any problems, many experienced programmers often face real failures in working on the code precisely because of this small detail. Knowing when, how, and why to use pass by value or pass by reference will make a huge difference in our lives as programmers. The content presented here is intended solely for educational purposes. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

![Developing a Replay System (Part 60): Playing the Service (I)](https://c.mql5.com/2/89/logo-midjourney_image_12086_394_3792__2.png)[Developing a Replay System (Part 60): Playing the Service (I)](https://www.mql5.com/en/articles/12086)

We have been working on just the indicators for a long time now, but now it's time to get the service working again and see how the chart is built based on the data provided. However, since the whole thing is not that simple, we will have to be attentive to understand what awaits us ahead.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/17350&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049145828196591047)

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