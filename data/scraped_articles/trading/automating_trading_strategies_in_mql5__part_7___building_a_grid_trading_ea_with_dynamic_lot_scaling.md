---
title: Automating Trading Strategies in MQL5 (Part 7): Building a Grid Trading EA with Dynamic Lot Scaling
url: https://www.mql5.com/en/articles/17190
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T17:58:27.451202
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/17190&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049526929234701585)

MetaTrader 5 / Trading


### Introduction

In the [previous article (Part 6)](https://www.mql5.com/en/articles/17135), we developed an automated [Order Block](https://www.mql5.com/go?link=https://tradingkit.net/articles/order-block/ "https://tradingkit.net/articles/order-block/") Detection System in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5). Now, in Part 7, we focus on grid trading, a strategy that places trades at fixed price intervals, combined with dynamic lot scaling to optimize risk and reward. This approach adapts position sizing based on market conditions, aiming to enhance profitability and risk management. We will cover:

1. [Strategy Blueprint](https://www.mql5.com/en/articles/17190#para2)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/17190#para3)
3. [Backtesting](https://www.mql5.com/en/articles/17190#para4)
4. [Conclusion](https://www.mql5.com/en/articles/17190#para5)

By the end, you'll have a fully functional Grid Trading program with dynamic lot scaling, ready for testing and optimization. Let’s begin!

### Strategy Blueprint

Grid trading is a systematic approach that places buy and sell orders at predetermined price intervals, allowing traders to capitalize on market fluctuations without requiring precise trend predictions. This strategy benefits from market volatility by continuously opening and closing trades within a defined price range. To enhance its performance, we will integrate dynamic lot scaling, which will adjust position sizes based on predefined conditions, such as account balance, volatility, or previous trade outcomes. Our Grid Trading system will operate with the following key components:

- **Grid Structure** – We will define the spacing between orders.
- **Entry and Execution Rules** – We will determine when to open grid trades based on fixed distances using a [Moving Average indicator](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma") strategy.
- **Dynamic Lot Scaling** – We will implement an adaptive lot-sizing mechanism that adjusts position sizes based on market conditions or predefined risk parameters.
- **Trade Management** – We will incorporate stop-loss, take-profit, and optional breakeven mechanisms to manage risk effectively.
- **Exit Strategy** – We will develop logic to close positions based on profit targets, risk limits, or trend reversals.

In a nutshell, here is the whole strategy blueprint visualization for ease of understanding.

![GRID LAYOUT](https://c.mql5.com/2/118/Screenshot_2025-02-13_162031.png)

By combining a structured grid system with adaptive lot sizing, we will create an EA that maximizes returns while effectively managing risk. Next, we will implement these concepts in MQL5.

### Implementation in MQL5

To create the program in MQL5, open the MetaEditor, go to the Navigator, locate the Indicators folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some global variables that we will use throughout the program.

```
//+------------------------------------------------------------------+
//|                        Copyright 2025, Forex Algo-Trader, Allan. |
//|                                 "https://t.me/Forex_Algo_Trader" |
//+------------------------------------------------------------------+
#property copyright "Forex Algo-Trader, Allan"
#property link      "https://t.me/Forex_Algo_Trader"
#property version   "1.00"
#property description "This EA trades based on Grid Strategy"
#property strict

#include <Trade/Trade.mqh>                     //--- Include trading library
CTrade obj_Trade;                            //--- Trading object instance

//--- Closure Mode Enumeration and Inputs
enum ClosureMode {
   CLOSE_BY_PROFIT,      //--- Use total profit (in currency) to close positions
   CLOSE_BY_POINTS       //--- Use a points threshold from breakeven to close positions
};

input group "General EA Inputs"
input ClosureMode closureMode = CLOSE_BY_POINTS;    //Select closure mode

double breakevenPoints = 50 * _Point;                 //--- Points offset to add/subtract to/from breakeven

//--- Global Variables
double TakeProfit;                           //--- Current take profit level
double initialLotsize      = 0.1;            //--- Initial lot size for the first trade
double takeProfitPts       = 200 * _Point;    //--- Take profit distance in points
double profitTotal_inCurrency = 100;          //--- Profit target (in currency) to close positions

double gridSize;                             //--- Price level at which grid orders are triggered
double gridSize_Spacing   = 500 * _Point;      //--- Grid spacing in points
double LotSize;                              //--- Current lot size (increased with grid orders)

bool isTradeAllowed      = true;              //--- Flag to allow trade on a new bar
int totalBars            = 0;                 //--- Count of bars seen so far
int handle;                                  //--- Handle for the Moving Average indicator
double maData[];                             //--- Array for Moving Average data
```

Here, we include the " [Trade/Trade.mqh](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade)" library using [#include](https://www.mql5.com/en/docs/basis/preprosessor/include) and instantiate the object " [obj\_Trade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade)" to handle our trades. We define a "ClosureMode" [enumeration](https://www.mql5.com/en/book/basis/builtin_types/enums) with options for closing positions and set up user inputs like "closureMode" and "breakevenPoints". Next, we declare variables to manage our take profit levels, initial lot size, grid spacing, and dynamic lot sizing, along with flags and counters for trade control and moving average indicator data. We then need to declare the prototypes for our key functions that will structure the program as follows.

```
//--- Function Prototypes
void   CheckAndCloseProfitTargets();         //--- Closes all positions if total profit meets target
void   ExecuteInitialTrade(double ask, double bid); //--- Executes the initial BUY/SELL trade (initial positions)
void   ManageGridPositions(double ask, double bid); //--- Adds grid orders when market moves to grid level (grid positions)
void   UpdateMovingAverage();                //--- Updates MA indicator data from its buffer
bool   IsNewBar();                           //--- Checks if a new bar has formed
double CalculateWeightedBreakevenPrice();    //--- Calculates the weighted average entry price for positions
void   CheckBreakevenClose(double ask, double bid); //--- Closes positions if price meets breakeven+/- threshold
void   CloseAllPositions();                  //--- Closes all open positions
```

For the functions, we will implement "CheckAndCloseProfitTargets" to monitor overall profitability and close positions once our target is reached, and "ExecuteInitialTrade" to kick off the strategy with the initial BUY or SELL order. "ManageGridPositions" will add additional orders at set grid intervals as the market moves, while "UpdateMovingAverage" ensures our indicator data is current for decision-making. "IsNewBar" detects new bars to prevent multiple trades on the same candle, "CalculateWeightedBreakevenPrice" computes the average entry price across positions, and "CheckBreakevenClose" uses that information to exit trades when favorable conditions are met. Lastly, "CloseAllPositions" will methodically close all open trades when necessary.

After setting all these on the "global scope", we are set to continue with the program initialization, which is on the "OnInit" event handler.

```
//+------------------------------------------------------------------+
//--- Expert initialization function
//+------------------------------------------------------------------+
int OnInit(){
   //--- Initialize the Moving Average indicator (Period: 21, SMA, Price: Close)
   handle = iMA(_Symbol, _Period, 21, 0, MODE_SMA, PRICE_CLOSE);
   if (handle == INVALID_HANDLE){
      Print("ERROR: UNABLE TO INITIALIZE THE INDICATOR. REVERTING NOW!");
      return (INIT_FAILED);
   }
   ArraySetAsSeries(maData, true);            //--- Ensure MA data array is in series order
   return(INIT_SUCCEEDED);
}
```

Here, we initialize the program by setting up our Moving Average indicator using the [iMA](https://www.mql5.com/en/docs/indicators/ima) function with a period of 21, [SMA](https://www.mql5.com/en/docs/constants/indicatorconstants/enum_ma_method) type, and [PRICE\_CLOSE](https://www.mql5.com/en/docs/constants/indicatorconstants/prices#enum_applied_price_enum) to capture closing prices. We check if the indicator handle is valid—if it isn't ( [INVALID\_HANDLE](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants)), we print an error message and return [INIT\_FAILED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode) to stop the program from running. Finally, we call the [ArraySetAsSeries](https://www.mql5.com/en/docs/array/arraysetasseries) function on the "maData" array to ensure the Moving Average data is arranged in the correct order before returning [INIT\_SUCCEEDED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode) to confirm successful initialization. Once initialized correctly, we can proceed to the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler to build the logic for opening and managing the positions.

```
//+------------------------------------------------------------------+
//--- Expert tick function
//+------------------------------------------------------------------+
void OnTick(){

   //--- Allow new trade signals on a new bar
   if(IsNewBar())
      isTradeAllowed = true;

   //--- Update the Moving Average data
   UpdateMovingAverage();

}
```

Since we don't want to check for trades on every tick, but on every bar, we call the function "IsNewBar" and use it to set the "isTradeAllowed" variable to true if a new bar is formed. We then call the function responsible for getting the moving average values. The function's definitions are as follows.

```
//+-------------------------------------------------------------------+
//--- Function: UpdateMovingAverage
//--- Description: Copies the latest data from the MA indicator buffer.
//+-------------------------------------------------------------------+
void UpdateMovingAverage(){
   if(CopyBuffer(handle, 0, 1, 3, maData) < 0)
      Print("Error: Unable to update Moving Average data.");
}

//+-------------------------------------------------------------------+
//--- Function: IsNewBar
//--- Description: Checks if a new bar has been formed.
//+-------------------------------------------------------------------+
bool IsNewBar(){
   int bars = iBars(_Symbol, _Period);
   if(bars > totalBars){
      totalBars = bars;
      return true;
   }
   return false;
}
```

Here, we implement "UpdateMovingAverage" to refresh our indicator data by copying the latest values from the Moving Average buffer using the [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer) function. If this function call fails, we print an error message to alert us that the update was unsuccessful. In the "IsNewBar" function, we check whether a new bar has formed by comparing the current number of bars, obtained via the [iBars](https://www.mql5.com/en/docs/series/ibars) function, to our stored "totalBars" count; if the number has increased, we update "totalBars" and return "true", indicating that a new bar is available for trading decisions. We then continue with the tick function to execute trades based on retrieved indicator values.

```
//--- Reset lot size if no positions are open
if(PositionsTotal() == 0)
   LotSize = initialLotsize;

//--- Retrieve recent bar prices for trade signal logic
double low1  = iLow(_Symbol, _Period, 1);
double low2  = iLow(_Symbol, _Period, 2);
double high1 = iHigh(_Symbol, _Period, 1);
double high2 = iHigh(_Symbol, _Period, 2);

//--- Get current Ask and Bid prices (normalized)
double ask = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK), _Digits);
double bid = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID), _Digits);

//--- If no positions are open and trading is allowed, check for an initial trade signal
if(PositionsTotal() == 0 && isTradeAllowed){
   ExecuteInitialTrade(ask, bid);
}
```

Here, we first check if no positions are open using the [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) function, and if so, we reset the "LotSize" to "initialLotsize". Next, we retrieve recent bar prices by calling [iLow](https://www.mql5.com/en/docs/series/ilow) and [iHigh](https://www.mql5.com/en/docs/series/ihigh) to capture the highs and lows of the previous two bars, which will help form our trade signals. We then obtain the current "ask" and "bid" prices using [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble), normalizing them with [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble) to ensure accuracy. Finally, if trading is allowed (as indicated by "isTradeAllowed") and no positions are currently open, we call the "ExecuteInitialTrade" function with the "ask" and "bid" prices to initiate our first trade. The function definition is as below.

```
//+---------------------------------------------------------------------------+
//--- Function: ExecuteInitialTrade
//--- Description: Executes the initial BUY or SELL trade based on MA criteria.
//---              (These are considered "initial positions.")
//+---------------------------------------------------------------------------+
void ExecuteInitialTrade(double ask, double bid){
   //--- BUY Signal: previous bar's low above MA and bar before that below MA
   if(iLow(_Symbol, _Period, 1) > maData[1] && iLow(_Symbol, _Period, 2) < maData[1]){
      gridSize = ask - gridSize_Spacing;     //--- Set grid trigger below current ask
      TakeProfit = ask + takeProfitPts;      //--- Set TP for BUY
      if(obj_Trade.Buy(LotSize, _Symbol, ask, 0, TakeProfit,"Initial Buy"))
         Print("Initial BUY order executed at ", ask, " with LotSize: ", LotSize);
      else
         Print("Initial BUY order failed at ", ask);
      isTradeAllowed = false;
   }
   //--- SELL Signal: previous bar's high below MA and bar before that above MA
   else if(iHigh(_Symbol, _Period, 1) < maData[1] && iHigh(_Symbol, _Period, 2) > maData[1]){
      gridSize = bid + gridSize_Spacing;     //--- Set grid trigger above current bid
      TakeProfit = bid - takeProfitPts;      //--- Set TP for SELL
      if(obj_Trade.Sell(LotSize, _Symbol, bid, 0, TakeProfit,"Initial Sell"))
         Print("Initial SELL order executed at ", bid, " with LotSize: ", LotSize);
      else
         Print("Initial SELL order failed at ", bid);
      isTradeAllowed = false;
   }
}
```

Here, we implement the "ExecuteInitialTrade" function to open an initial trade based on the "maData" values. We retrieve the low prices of the previous two bars using the function [iLow](https://www.mql5.com/en/docs/series/ilow) and the high prices using the function [iHigh](https://www.mql5.com/en/docs/series/ihigh). For a BUY signal, we check if the low of the previous bar is above "maData" while the bar before that was below it. If this condition is met, we set "gridSize" below the current "ask" using "gridSize\_Spacing" to determine the next grid level, calculate "TakeProfit" by adding "takeProfitPts" to the "ask", and execute a BUY trade using "obj\_Trade.Buy" method.

For a SELL signal, we check if the high of the previous bar is below "maData" while the bar before that was above it. If true, we set "gridSize" above the "bid", determine "TakeProfit" by subtracting "takeProfitPts" from the "bid", and attempt to execute a SELL trade using "obj\_Trade.Sell". Once a trade is executed, we set "isTradeAllowed" to false to prevent additional entries until further conditions are met. Here is the outcome.

![TRADE EXECUTION](https://c.mql5.com/2/119/Screenshot_2025-02-13_174402.png)

From the image, we can see that we have the confirmed trades being executed. We now need to move on to manage the trades by opening the grid positions.

```
//--- If positions exist, manage grid orders
if(PositionsTotal() > 0){
   ManageGridPositions(ask, bid);
}
```

We check if there are open positions using the function [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal). If the number of positions is greater than zero, we call the function "ManageGridPositions" to handle additional grid trades. The function takes "ask" and "bid" as parameters to determine the appropriate price levels for placing new grid orders based on market movement. The function's code snippet implementation is as below.

```
//+------------------------------------------------------------------------+
//--- Function: ManageGridPositions
//--- Description: When an initial position exists, grid orders are added
//---              if the market moves to the grid level. (These orders are
//---              considered "grid positions.") The lot size is doubled
//---              with each grid order.
//+------------------------------------------------------------------------+
void ManageGridPositions(double ask, double bid){
   for(int i = PositionsTotal()-1; i >= 0; i--){
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket)){
         int positionType = (int)PositionGetInteger(POSITION_TYPE);
         //--- Grid management for BUY positions
         if(positionType == POSITION_TYPE_BUY){
            if(ask <= gridSize){
               LotSize *= 2;              //--- Increase lot size for grid order
               if(obj_Trade.Buy(LotSize, _Symbol, ask, 0, TakeProfit,"Grid Position BUY"))
                  Print("Grid BUY order executed at ", ask, " with LotSize: ", LotSize);
               else
                  Print("Grid BUY order failed at ", ask);
               gridSize = ask - gridSize_Spacing; //--- Update grid trigger
            }
         }
         //--- Grid management for SELL positions
         else if(positionType == POSITION_TYPE_SELL){
            if(bid >= gridSize){
               LotSize *= 2;              //--- Increase lot size for grid order
               if(obj_Trade.Sell(LotSize, _Symbol, bid, 0, TakeProfit,"Grid Position SELL"))
                  Print("Grid SELL order executed at ", bid, " with LotSize: ", LotSize);
               else
                  Print("Grid SELL order failed at ", bid);
               gridSize = bid + gridSize_Spacing; //--- Update grid trigger
            }
         }
      }
   }
}
```

We implement the "ManageGridPositions" function to manage grid orders. We iterate through all open positions in reverse using a [for loop](https://www.mql5.com/en/docs/basis/operators/for) and retrieve each position’s ticket with the function [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket). We then select the position using [PositionSelectByTicket](https://www.mql5.com/en/docs/trading/positionselectbyticket) and determine whether it is a BUY or SELL trade using [PositionGetInteger](https://www.mql5.com/en/docs/trading/positiongetinteger) with the parameter [POSITION\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type). If the position is a BUY, we check if the market price "ask" has reached or dropped below "gridSize". If true, we double "LotSize" and execute a new grid BUY order using the function "obj\_Trade.Buy". If the order is successful, we print a confirmation message; otherwise, we print an error message. We then update "gridSize" to the next grid level below.

Similarly, if the position is a SELL, we check if "bid" has reached or exceeded "gridSize". If true, we double "LotSize" and place a new grid SELL order using "obj\_Trade.Sell". The grid trigger "gridSize" is then updated to the next level above. After opening the grid positions, we need to track and manage the positions by closing them once we hit the defined as below.

```
//--- Check if total profit meets the target (only used if closureMode == CLOSE_BY_PROFIT)
if(closureMode == CLOSE_BY_PROFIT)
   CheckAndCloseProfitTargets();
```

If "closureMode" is set to "CLOSE\_BY\_PROFIT", we call the function "CheckAndCloseProfitTargets" to check if the total profit has reached the predefined target and close all positions accordingly. The function declaration is as below.

```
//+----------------------------------------------------------------------------+
//--- Function: CheckAndCloseProfitTargets
//--- Description: Closes all positions if the combined profit meets or exceeds
//---              the user-defined profit target.
//+----------------------------------------------------------------------------+
void CheckAndCloseProfitTargets(){
   if(PositionsTotal() > 1){
      double totalProfit = 0;
      for(int i = PositionsTotal()-1; i >= 0; i--){
         ulong tkt = PositionGetTicket(i);
         if(PositionSelectByTicket(tkt))
            totalProfit += PositionGetDouble(POSITION_PROFIT);
      }
      if(totalProfit >= profitTotal_inCurrency){
         Print("Profit target reached (", totalProfit, "). Closing all positions.");
         CloseAllPositions();
      }
   }
}
```

To ensure that all positions are closed if the total accumulated profit meets or exceeds the predefined profit target, we first, check if there is more than one open position using [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal). We initialize "totalProfit" to track the combined profit of all positions. We then [loop](https://www.mql5.com/en/docs/basis/operators/for) through all open positions, retrieving each position’s ticket using [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket) and selecting it with [PositionSelectByTicket](https://www.mql5.com/en/docs/trading/positionselectbyticket). For each selected position, we retrieve its profit using [PositionGetDouble](https://www.mql5.com/en/docs/trading/positiongetdouble) with the parameter [POSITION\_PROFIT](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_double) and add it to "totalProfit". If "totalProfit" meets or exceeds "profitTotal\_inCurrency", we print a message indicating that the profit target has been reached and call the "CloseAllPositions" function, whose definition is as below, to close all open trades.

```
//+------------------------------------------------------------------+
//--- Function: CloseAllPositions
//--- Description: Iterates through and closes all open positions.
//+------------------------------------------------------------------+
void CloseAllPositions(){
   for(int i = PositionsTotal()-1; i >= 0; i--){
      ulong posTkt = PositionGetTicket(i);
      if(PositionSelectByTicket(posTkt)){
         if(obj_Trade.PositionClose(posTkt))
            Print("Closed position ticket: ", posTkt);
         else
            Print("Failed to close position ticket: ", posTkt);
      }
   }
}
```

The function just iterates over all open positions and for every selected position, is closed using the "obj\_Trade.PositionClose" method. Finally, we define the logic to close the positions on breakeven.

```
//--- If using CLOSE_BY_POINTS and more than one position exists (i.e. grid), check breakeven closure
if(closureMode == CLOSE_BY_POINTS && PositionsTotal() > 1)
   CheckBreakevenClose(ask, bid);
```

If "closureMode" is set to "CLOSE\_BY\_POINTS" and there is more than one open position, we call the function "CheckBreakevenClose" with parameters "ask" and "bid" to determine whether the price has reached the breakeven threshold, allowing positions to be closed based on predefined points from breakeven. The following is the function definition.

```
//+----------------------------------------------------------------------------+
//--- Function: CalculateWeightedBreakevenPrice
//--- Description: Calculates the weighted average entry price (breakeven)
//---              of all open positions (assumed to be in the same direction).
//+----------------------------------------------------------------------------+
double CalculateWeightedBreakevenPrice(){
   double totalCost = 0;
   double totalVolume = 0;
   int posType = -1;
   //--- Determine the type from the first position
   for(int i = 0; i < PositionsTotal(); i++){
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket)){
         posType = (int)PositionGetInteger(POSITION_TYPE);
         break;
      }
   }
   //--- Sum the cost and volume for positions matching the type
   for(int i = 0; i < PositionsTotal(); i++){
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket)){
         if(PositionGetInteger(POSITION_TYPE) == posType){
            double price = PositionGetDouble(POSITION_PRICE_OPEN);
            double volume = PositionGetDouble(POSITION_VOLUME);
            totalCost += price * volume;
            totalVolume += volume;
         }
      }
   }
   if(totalVolume > 0)
      return(totalCost / totalVolume);
   else
      return(0);
}

//+-----------------------------------------------------------------------------+
//--- Function: CheckBreakevenClose
//--- Description: When using CLOSE_BY_POINTS and multiple positions exist,
//---              calculates the weighted breakeven price and checks if the
//---              current price has moved the specified points in a profitable
//---              direction relative to breakeven. If so, closes all positions.
//+-----------------------------------------------------------------------------+
void CheckBreakevenClose(double ask, double bid){
   //--- Ensure we have more than one position (grid positions)
   if(PositionsTotal() <= 1)
      return;

   double weightedBreakeven = CalculateWeightedBreakevenPrice();
   int posType = -1;
   //--- Determine the trade type from one of the positions
   for(int i = 0; i < PositionsTotal(); i++){
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket)){
         posType = (int)PositionGetInteger(POSITION_TYPE);
         break;
      }
   }
   if(posType == -1)
      return;

   //--- For BUY positions, profit when Bid >= breakeven + threshold
   if(posType == POSITION_TYPE_BUY){
      if(bid >= weightedBreakeven + breakevenPoints){
         Print("Closing BUY positions: Bid (", bid, ") >= Breakeven (", weightedBreakeven, ") + ", breakevenPoints);
         CloseAllPositions();
      }
   }
   //--- For SELL positions, profit when Ask <= breakeven - threshold
   else if(posType == POSITION_TYPE_SELL){
      if(ask <= weightedBreakeven - breakevenPoints){
         Print("Closing SELL positions: Ask (", ask, ") <= Breakeven (", weightedBreakeven, ") - ", breakevenPoints);
         CloseAllPositions();
      }
   }
}
```

Here, we calculate the breakeven price for all open positions and determine if the market price has moved a specified distance beyond it to close positions for profit. In "CalculateWeightedBreakevenPrice", we compute the weighted breakeven price by summing the total cost of all open positions using [POSITION\_PRICE\_OPEN](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_double) and weighting it by "POSITION\_VOLUME". We first determine the position type (BUY or SELL) from the first open position using [POSITION\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type). We then loop through all positions, summing the total cost and volume for positions matching the identified type. If the total volume is greater than zero, we return the weighted breakeven price by dividing the total cost by the total volume. Otherwise, we return zero.

In "CheckBreakevenClose", we first confirm there are multiple open positions using the [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) function. We then retrieve the weighted breakeven price by calling "CalculateWeightedBreakevenPrice". We determine the position type by selecting a position and retrieving [POSITION\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type). If the type is invalid, we exit the function. For BUY positions, we check if the "bid" price has reached or exceeded "weightedBreakeven" plus "breakevenPoints". If so, we print a message and call "CloseAllPositions". For SELL positions, we check if the "ask" price has dropped below "weightedBreakeven" minus "breakevenPoints". If this condition is met, we also print a message and call the "CloseAllPositions" function to secure profits. Upon compilation and running the program, we have the following outcome.

![GRID GIF](https://c.mql5.com/2/119/GRID_GIF.gif)

From the visualization, we can see that the positions are opened and managed via the grid system and closed when the defined closure levels are hit, hence achieving our objective of creating a grid system with dynamic lot sizing. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/119/Screenshot_2025-02-13_191933.png)

Backtest report:

![REPORT](https://c.mql5.com/2/119/Screenshot_2025-02-13_191734.png)

Here is also a video format showcasing the whole strategy backtest within a period of 1 year, 2024.

YouTube

### Conclusion

In conclusion, we have demonstrated the process of developing an [MQL5](https://www.mql5.com/) Expert Advisor (EA) utilizing a dynamic grid trading strategy. By combining key elements such as grid order placement, dynamic lot scaling, and targeted profit and breakeven management, we created a system that adapts to market fluctuations, aiming to optimize risk-to-reward ratios and recover from adverse price movements.

Disclaimer: This article is for educational purposes only. Trading involves significant financial risk, and market behavior can be highly unpredictable. While the strategies outlined offer a structured approach to grid trading, they do not guarantee future profitability. Rigorous backtesting and risk management are essential before live trading.

By implementing these techniques, you can refine your grid trading systems, enhance your market analysis, and elevate your algorithmic trading strategies. Best of luck on your trading journey!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17190.zip "Download all attachments in the single ZIP archive")

[Grid\_EA\_With\_Dynamic\_Lot-Sizing.mq5](https://www.mql5.com/en/articles/download/17190/grid_ea_with_dynamic_lot-sizing.mq5 "Download Grid_EA_With_Dynamic_Lot-Sizing.mq5")(13.97 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/481648)**
(12)


![testtestmio71](https://c.mql5.com/avatar/avatar_na2.png)

**[testtestmio71](https://www.mql5.com/en/users/testtestmio71)**
\|
4 Apr 2025 at 15:19

is ok.....best EA .

The 4 lines are confusing for a newbie

![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
4 Apr 2025 at 16:35

**testtestmio71 [#](https://www.mql5.com/en/forum/481648#comment_56362845):**

is ok.....best EA .

The 4 lines are confusing for a newbie

Okay

![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
20 Oct 2025 at 02:54

great article - thank you very much... I am studying the trading approach I will put with edits on my own trades on [custom](https://www.mql5.com/en/articles/3540 "Article: Creating and Testing Custom Symbols in MetaTrader 5 ") hashedge [symbols](https://www.mql5.com/en/articles/3540 "Article: Creating and Testing Custom Symbols in MetaTrader 5 ")!

checking....

![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
20 Oct 2025 at 14:12

**Roman Shiredchenko [#](https://www.mql5.com/en/forum/481648#comment_58308527):**

great article - thank you very much... I am studying the trading approach I will put with edits on my own trades on [custom](https://www.mql5.com/en/articles/3540 "Article: Creating and Testing Custom Symbols in MetaTrader 5 ") hashedge [symbols](https://www.mql5.com/en/articles/3540 "Article: Creating and Testing Custom Symbols in MetaTrader 5 ")!

checking....

Good. Welcome.


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
30 Oct 2025 at 11:32

**Roman Shiredchenko custom hash [symbols](https://www.mql5.com/en/articles/3540 "Article: Creating and Testing Custom Symbols in MetaTrader 5") in my own trading! [Article: Creating and Testing Custom Symbols in MetaTrader 5](https://www.mql5.com/en/articles/3540 "Article: Creating and Testing Custom Symbols in MetaTrader 5")[Article: Creating and Testing Custom Symbols in MetaTrader 5](https://www.mql5.com/en/articles/3540 "Article: Creating and Testing Custom Symbols in MetaTrader 5")**
**examine....**

Thanks and welcome.


![Animal Migration Optimization (AMO) algorithm](https://c.mql5.com/2/90/logo-amo_15543.png)[Animal Migration Optimization (AMO) algorithm](https://www.mql5.com/en/articles/15543)

The article is devoted to the AMO algorithm, which models the seasonal migration of animals in search of optimal conditions for life and reproduction. The main features of AMO include the use of topological neighborhood and a probabilistic update mechanism, which makes it easy to implement and flexible for various optimization tasks.

![Price Action Analysis Toolkit Development (Part 13): RSI Sentinel Tool](https://c.mql5.com/2/119/Price_Action_Analysis_Toolkit_Development_Part_13___LOGO.png)[Price Action Analysis Toolkit Development (Part 13): RSI Sentinel Tool](https://www.mql5.com/en/articles/17198)

Price action can be effectively analyzed by identifying divergences, with technical indicators such as the RSI providing crucial confirmation signals. In the article below, we explain how automated RSI divergence analysis can identify trend continuations and reversals, thereby offering valuable insights into market sentiment.

![Master MQL5 from beginner to pro (Part IV): About Arrays, Functions and Global Terminal Variables](https://c.mql5.com/2/87/Learning_MQL5_-_from_beginner_to_pro_Part_IV___LOGO.png)[Master MQL5 from beginner to pro (Part IV): About Arrays, Functions and Global Terminal Variables](https://www.mql5.com/en/articles/15357)

The article is a continuation of the series for beginners. It covers in detail data arrays, the interaction of data and functions, as well as global terminal variables that allow data exchange between different MQL5 programs.

![Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (II): Modularization](https://c.mql5.com/2/119/Creating_a_Trading_Administrator_Panel_in_MQL5_Part_IX___LOGO.png)[Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (II): Modularization](https://www.mql5.com/en/articles/16562)

In this discussion, we take a step further in breaking down our MQL5 program into smaller, more manageable modules. These modular components will then be integrated into the main program, enhancing its organization and maintainability. This approach simplifies the structure of our main program and makes the individual components reusable in other Expert Advisors (EAs) and indicator developments. By adopting this modular design, we create a solid foundation for future enhancements, benefiting both our project and the broader developer community.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/17190&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049526929234701585)

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