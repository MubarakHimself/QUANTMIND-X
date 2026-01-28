---
title: Automating Trading Strategies in MQL5 (Part 10): Developing the Trend Flat Momentum Strategy
url: https://www.mql5.com/en/articles/17247
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 7
scraped_at: 2026-01-22T17:47:46.522095
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=bpplosnyudjkiacefplqktfjtvqwvisi&ssn=1769093265634691747&ssn_dr=0&ssn_sr=0&fv_date=1769093265&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F17247&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2010)%3A%20Developing%20the%20Trend%20Flat%20Momentum%20Strategy%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909326523797644&fz_uniq=5049385487371709073&sv=2552)

MetaTrader 5 / Trading


### Introduction

In the [previous article (Part 9)](https://www.mql5.com/en/articles/17239), we developed an Expert Advisor to automate the [Asian Breakout Strategy](https://www.mql5.com/go?link=https://www.dolphintrader.com/asian-range-breakout-forex-strategy/ "https://www.dolphintrader.com/asian-range-breakout-forex-strategy/") using key session levels and dynamic risk management in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5). Now, in Part 10, we shift our focus to the [Trend Flat Momentum Strategy](https://www.mql5.com/go?link=https://www.learn-forextrading.org/2017/10/trend-flat-momentum-trading-system.html%23%3a%7e%3atext%3dTrend%2520flat%2520momentum%2520is%2520a%2520trend%2520following%2520system%2cis%2520designed%2520for%2520works%2520a%2520higher%2520time%2520frame. "https://www.learn-forextrading.org/2017/10/trend-flat-momentum-trading-system.html#:~:text=Trend%20flat%20momentum%20is%20a%20trend%20following%20system,is%20designed%20for%20works%20a%20higher%20time%20frame.")—a method that combines a two [moving averages](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma") crossover with momentum filters like [Relative Strength Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi") (RSI) and [Commodity Channel Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/cci "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/cci") (CCI) to capture trending movements with precision. We will structure the article via the following topics:

1. [Strategy Blueprint](https://www.mql5.com/en/articles/17247#para2)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/17247#para3)
3. [Backtesting and Optimization](https://www.mql5.com/en/articles/17247#para4)
4. [Conclusion](https://www.mql5.com/en/articles/17247#para5)

By the end, we’ll have a fully functional Expert Advisor that automates the Trend Flat Momentum Strategy. Let’s dive in!

### Strategy Blueprint

The Trend Flat Momentum Strategy is designed to capture market trends by blending a simple moving average crossover system with robust momentum filtering. The core idea is to generate buy signals when a fast-moving average crosses above a slower-moving average—suggesting a bullish trend—while confirming the signal with momentum indicators, which are an [RSI](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi") and two different [CCI](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/cci "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/cci") values. Conversely, a short trade is signaled when the slow-moving average exceeds the fast-moving average and the momentum indicators confirm bearish conditions. The indicator settings are:

- Commodity Channel Index (CCI) (36 Periods, Close)
- Commodity Channel Index (CCI) (55 Periods, Close)
- Slow Relative Strength Index (RSI) (27 Periods, Close)
- Moving Average Fast (11 Periods, Smoothed, Median Price)
- Moving Slow Fast (25 Periods, Smoothed, Median Price)

As for the exit strategy, we will place the stop loss at the previous swing low for a long trade and the previous swing high for a short trade. Take profit will be at a predetermined level, 300 points from the entry price. This multi-faceted approach will help filter out false signals and aims to improve the quality of trade entries by ensuring that trend direction and momentum are aligned. In a nutshell, the visualization below depicts the simplified strategic plan.

![STRATEGY PLAN](https://c.mql5.com/2/119/Screenshot_2025-02-17_160926.png)

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Indicators folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some [global variables](https://www.mql5.com/en/docs/basis/variables/global) that we will use throughout the program.

```
//+------------------------------------------------------------------+
//|                        Copyright 2025, Forex Algo-Trader, Allan. |
//|                                 "https://t.me/Forex_Algo_Trader" |
//+------------------------------------------------------------------+
#property copyright "Forex Algo-Trader, Allan"
#property link      "https://t.me/Forex_Algo_Trader"
#property version   "1.00"
#property description "This EA trades based on Trend Flat Momentum Strategy"
#property strict

#include <Trade\Trade.mqh> //--- Include the Trade library for order management.
CTrade obj_Trade; //--- Create an instance of the CTrade class to handle trading operations.

// Input parameters
input int    InpCCI36Period      = 36;              //--- CCI period 1
input int    InpCCI55Period      = 55;              //--- CCI period 2
input int    InpRSIPeriod        = 27;              //--- RSI period
input int    InpMAFastPeriod     = 11;              //--- Fast MA period
input int    InpMASlowPeriod     = 25;              //--- Slow MA period
input double InpRSIThreshold     = 58.0;            //--- RSI threshold for Buy signal (Sell uses 100 - Threshold)
input int    InpTakeProfitPoints = 300;             //--- Take profit in points
input double InpLotSize          = 0.1;             //--- Trade lot size

// Pivot parameters for detecting swing highs/lows
input int    PivotLeft  = 2;  //--- Number of bars to the left for pivot detection
input int    PivotRight = 2;  //--- Number of bars to the right for pivot detection

// Global indicator handles
int handleCCI36;  //--- Handle for the CCI indicator with period InpCCI36Period
int handleCCI55;  //--- Handle for the CCI indicator with period InpCCI55Period
int handleRSI;    //--- Handle for the RSI indicator with period InpRSIPeriod
int handleMA11;   //--- Handle for the fast moving average (MA) with period InpMAFastPeriod
int handleMA25;   //--- Handle for the slow moving average (MA) with period InpMASlowPeriod

// Global dynamic storage buffers
double ma11_buffer[];  //--- Dynamic array to store fast MA values
double ma25_buffer[];  //--- Dynamic array to store slow MA values
double rsi_buffer[];   //--- Dynamic array to store RSI values
double cci36_buffer[]; //--- Dynamic array to store CCI values (period 36)
double cci55_buffer[]; //--- Dynamic array to store CCI values (period 55)

// To detect a new bar
datetime lastBarTime = 0;  //--- Variable to store the time of the last processed bar
```

We start by including the file "Trade\\Trade.mqh" using [#include](https://www.mql5.com/en/docs/basis/preprosessor/include) to access built-in trading functions and create "obj\_Trade", an instance of the "CTrade" class, for executing buy and sell orders. We define key [input](https://www.mql5.com/en/docs/basis/variables/inputvariables) parameters for strategy configuration, including "InpCCI36Period" and "InpCCI55Period" for "CCI" indicators, "InpRSIPeriod" for "RSI", and "InpMAFastPeriod" and "InpMASlowPeriod" for two moving averages. "InpRSIThreshold" sets a condition for trade filtering, while "InpTakeProfitPoints" determines the fixed take-profit level, and "InpLotSize" controls the position size.

To improve trade execution, we introduce "PivotLeft" and "PivotRight", which define the number of bars used to detect swing highs and lows for stop-loss placement. Global indicator handles, such as "handleCCI36", "handleCCI55", "handleRSI", "handleMA11", and "handleMA25", allow us to retrieve indicator values efficiently. Dynamic [arrays](https://www.mql5.com/en/book/basis/arrays) store these values in "ma11\_buffer", "ma25\_buffer", "rsi\_buffer", "cci36\_buffer", and "cci55\_buffer", ensuring smooth data processing. Finally, "lastBarTime" tracks the last processed bar to prevent multiple trades on the same candle, ensuring accurate trade execution. We can then initialize the indicators in the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   //--- Create CCI handle for period InpCCI36Period using the close price.
   handleCCI36 = iCCI(_Symbol, _Period, InpCCI36Period, PRICE_CLOSE); //--- Create the CCI36 indicator handle.
   if (handleCCI36 == INVALID_HANDLE) { //--- Check if the CCI36 handle is valid.
      Print("Error creating CCI36 handle"); //--- Print an error message if invalid.
      return (INIT_FAILED); //--- Return failure if handle creation failed.
   }

   //--- Create CCI handle for period InpCCI55Period using the close price.
   handleCCI55 = iCCI(_Symbol, _Period, InpCCI55Period, PRICE_CLOSE); //--- Create the CCI55 indicator handle.
   if (handleCCI55 == INVALID_HANDLE) { //--- Check if the CCI55 handle is valid.
      Print("Error creating CCI55 handle"); //--- Print an error message if invalid.
      return (INIT_FAILED); //--- Return failure if handle creation failed.
   }

   //--- Create RSI handle for period InpRSIPeriod using the close price.
   handleRSI = iRSI(_Symbol, _Period, InpRSIPeriod, PRICE_CLOSE); //--- Create the RSI indicator handle.
   if (handleRSI == INVALID_HANDLE) { //--- Check if the RSI handle is valid.
      Print("Error creating RSI handle"); //--- Print an error message if invalid.
      return (INIT_FAILED); //--- Return failure if handle creation failed.
   }

   //--- Create fast MA handle using MODE_SMMA on the median price with period InpMAFastPeriod.
   handleMA11 = iMA(_Symbol, _Period, InpMAFastPeriod, 0, MODE_SMMA, PRICE_MEDIAN); //--- Create the fast MA handle.
   if (handleMA11 == INVALID_HANDLE) { //--- Check if the fast MA handle is valid.
      Print("Error creating MA11 handle"); //--- Print an error message if invalid.
      return (INIT_FAILED); //--- Return failure if handle creation failed.
   }

   //--- Create slow MA handle using MODE_SMMA on the median price with period InpMASlowPeriod.
   handleMA25 = iMA(_Symbol, _Period, InpMASlowPeriod, 0, MODE_SMMA, PRICE_MEDIAN); //--- Create the slow MA handle.
   if (handleMA25 == INVALID_HANDLE) { //--- Check if the slow MA handle is valid.
      Print("Error creating MA25 handle"); //--- Print an error message if invalid.
      return (INIT_FAILED); //--- Return failure if handle creation failed.
   }

   //--- Set the dynamic arrays as time series (index 0 = most recent closed bar).
   ArraySetAsSeries(ma11_buffer, true); //--- Set ma11_buffer as a time series.
   ArraySetAsSeries(ma25_buffer, true); //--- Set ma25_buffer as a time series.
   ArraySetAsSeries(rsi_buffer, true);  //--- Set rsi_buffer as a time series.
   ArraySetAsSeries(cci36_buffer, true); //--- Set cci36_buffer as a time series.
   ArraySetAsSeries(cci55_buffer, true); //--- Set cci55_buffer as a time series.

   return (INIT_SUCCEEDED); //--- Return success after initialization.
}
```

On the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, we initialize the Expert Advisor by creating and validating indicator handles. We use the [iCCI](https://www.mql5.com/en/docs/indicators/icci) function to create CCI handles with periods "InpCCI36Period" and "InpCCI55Period," the [iRSI](https://www.mql5.com/en/docs/indicators/irsi) function for the RSI handle, and the [iMA](https://www.mql5.com/en/docs/indicators/ima) function for fast and slow [SMMA](https://www.mql5.com/en/docs/constants/indicatorconstants/enum_ma_method) handles with periods "InpMAFastPeriod" and "InpMASlowPeriod." If any handle is invalid ( [INVALID\_HANDLE](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants)), we print an error and return failure ( [INIT\_FAILED](https://www.mql5.com/en/docs/event_handlers/oninit)). Finally, we use the [ArraySetAsSeries](https://www.mql5.com/en/docs/array/arraysetasseries) function to format buffers as time series and return success upon successful initialization. To save resources, we need to release the created handles when the program is removed as follows.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   if (handleCCI36 != INVALID_HANDLE) { //--- If the CCI36 handle is valid,
      IndicatorRelease(handleCCI36); //--- release the CCI36 indicator.
   }
   if (handleCCI55 != INVALID_HANDLE) { //--- If the CCI55 handle is valid,
      IndicatorRelease(handleCCI55); //--- release the CCI55 indicator.
   }
   if (handleRSI != INVALID_HANDLE) { //--- If the RSI handle is valid,
      IndicatorRelease(handleRSI); //--- release the RSI indicator.
   }
   if (handleMA11 != INVALID_HANDLE) { //--- If the fast MA handle is valid,
      IndicatorRelease(handleMA11); //--- release the fast MA indicator.
   }
   if (handleMA25 != INVALID_HANDLE) { //--- If the slow MA handle is valid,
      IndicatorRelease(handleMA25); //--- release the slow MA indicator.
   }
}
```

Here, we handle the deinitialization of the Expert Advisor by releasing indicator resources in the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit). We check if each indicator handles—"CCI36", "CCI55", "RSI", fast MA, and slow MA—is valid. If so, we use the [IndicatorRelease](https://www.mql5.com/en/docs/series/indicatorrelease) function to free the allocated resources, ensuring efficient memory management. We can now proceed to the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler where all the data procession and decision-making will take place.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   //--- Get the time of the current bar.
   datetime currentBarTime = iTime(_Symbol, _Period, 0); //--- Retrieve the current bar's time.
   if (currentBarTime != lastBarTime) { //--- If a new bar has formed,
      lastBarTime = currentBarTime; //--- update lastBarTime with the current bar's time.
      OnNewBar(); //--- Process the new bar.
   }
}
```

Here, we use the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler to monitor price updates and detect new bars. We retrieve the time of the current bar using the [iTime](https://www.mql5.com/en/docs/series/itime) function and compare it with the stored "lastBarTime" value. If a new bar is detected, we update "lastBarTime" ensuring we trade only once per bar, and call the "OnNewBar" function to process the new bar's data. This is the function where we handle all signal generation and it is as follows.

```
//+------------------------------------------------------------------+
//| Function called on every new bar (bar close)                     |
//+------------------------------------------------------------------+
void OnNewBar() {

   //--- Resize the dynamic arrays to hold 2 values (last closed bar and the one before).
   ArrayResize(ma11_buffer, 2); //--- Resize ma11_buffer to 2 elements.
   ArrayResize(ma25_buffer, 2); //--- Resize ma25_buffer to 2 elements.
   ArrayResize(rsi_buffer, 2);  //--- Resize rsi_buffer to 2 elements.
   ArrayResize(cci36_buffer, 2); //--- Resize cci36_buffer to 2 elements.
   ArrayResize(cci55_buffer, 2); //--- Resize cci55_buffer to 2 elements.

   //--- Copy indicator values into the dynamic arrays.
   if (CopyBuffer(handleMA11, 0, 1, 2, ma11_buffer) != 2) { //--- Copy 2 values from the fast MA indicator.
      return; //--- Exit the function if copying fails.
   }
   if (CopyBuffer(handleMA25, 0, 1, 2, ma25_buffer) != 2) { //--- Copy 2 values from the slow MA indicator.
      return; //--- Exit the function if copying fails.
   }
   if (CopyBuffer(handleRSI, 0, 1, 2, rsi_buffer) != 2) { //--- Copy 2 values from the RSI indicator.
      return; //--- Exit the function if copying fails.
   }
   if (CopyBuffer(handleCCI36, 0, 1, 2, cci36_buffer) != 2) { //--- Copy 2 values from the CCI36 indicator.
      return; //--- Exit the function if copying fails.
   }
   if (CopyBuffer(handleCCI55, 0, 1, 2, cci55_buffer) != 2) { //--- Copy 2 values from the CCI55 indicator.
      return; //--- Exit the function if copying fails.
   }

   //--- For clarity, assign the values from the arrays.
   //--- Index 0: last closed bar, Index 1: bar before.
   double ma11_current   = ma11_buffer[0]; //--- Fast MA value for the last closed bar.
   double ma11_previous  = ma11_buffer[1]; //--- Fast MA value for the previous bar.
   double ma25_current   = ma25_buffer[0]; //--- Slow MA value for the last closed bar.
   double ma25_previous  = ma25_buffer[1]; //--- Slow MA value for the previous bar.
   double rsi_current    = rsi_buffer[0];  //--- RSI value for the last closed bar.
   double cci36_current  = cci36_buffer[0]; //--- CCI36 value for the last closed bar.
   double cci55_current  = cci55_buffer[0]; //--- CCI55 value for the last closed bar.
}
```

In the void function "OnNewBar" that we create, we first ensure that the dynamic arrays holding indicator values are resized using the [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) function to store the last two closed bars. We then retrieve indicator values using the [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer) function for the fast-moving average, slow-moving average, RSI, and two CCI indicators. If any of these operations fail, the function exits to prevent errors. Once the values are successfully copied, we assign them to variables for easier reference, distinguishing between the last closed bar and the bar before it. This setup ensures that we always have the most recent market data available for making trading decisions. If we retrieve the data successfully, we can proceed to make trading decisions. We start with the buy logic.

```
//--- Check for Buy Conditions:
bool maCrossoverBuy    = (ma11_previous < ma25_previous) && (ma11_current > ma25_current); //--- True if fast MA crosses above slow MA.
bool rsiConditionBuy   = (rsi_current > InpRSIThreshold); //--- True if RSI is above the Buy threshold.
bool cci36ConditionBuy = (cci36_current > 0); //--- True if CCI36 is positive.
bool cci55ConditionBuy = (cci55_current > 0); //--- True if CCI55 is positive.

if (maCrossoverBuy) { //--- If crossover for MA Buy is true...
   bool conditionsOk = true; //--- Initialize a flag to track if all conditions are met.

   //--- Check RSI condition for Buy.
   if (!rsiConditionBuy) { //--- If the RSI condition is not met...
      Print("Buy signal rejected: RSI condition not met. RSI=", rsi_current, " Threshold=", InpRSIThreshold); //--- Notify the user.
      conditionsOk = false; //--- Mark the conditions as not met.
   }
   //--- Check CCI36 condition for Buy.
   if (!cci36ConditionBuy) { //--- If the CCI36 condition is not met...
      Print("Buy signal rejected: CCI36 condition not met. CCI36=", cci36_current); //--- Notify the user.
      conditionsOk = false; //--- Mark the conditions as not met.
   }
   //--- Check CCI55 condition for Buy.
   if (!cci55ConditionBuy) { //--- If the CCI55 condition is not met...
      Print("Buy signal rejected: CCI55 condition not met. CCI55=", cci55_current); //--- Notify the user.
      conditionsOk = false; //--- Mark the conditions as not met.
   }
```

Here, we evaluate the conditions for entering a buy trade. The "maCrossoverBuy" variable checks if the fast-moving average ("ma11") has crossed above the slow-moving average ("ma25"), indicating a potential buy signal. The "rsiConditionBuy" ensures that the RSI value is above the defined "InpRSIThreshold," confirming strong bullish momentum. The "cci36ConditionBuy" and "cci55ConditionBuy" check if both CCI indicators are positive, suggesting that the market is in a favorable trend. If the "maCrossoverBuy" condition is true, we proceed to validate the remaining conditions. If any condition fails, we print a message indicating why the buy signal is rejected and set the "conditionsOk" flag to false to prevent further trade execution. This comprehensive check ensures that only trades with strong bullish confirmation are taken. If we then detect one, we can proceed to open the position.

```
if (conditionsOk) { //--- If all Buy conditions are met...
   //--- Get stop loss from previous swing low.
   double stopLoss = GetPivotLow(PivotLeft, PivotRight); //--- Use pivot low as the stop loss.
   if (stopLoss <= 0) { //--- If no valid pivot low is found...
      stopLoss = SymbolInfoDouble(_Symbol, SYMBOL_BID); //--- Fallback to current bid price.
   }

   double entryPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK); //--- Determine entry price as current ask price.
   double tp         = entryPrice + InpTakeProfitPoints * _Point; //--- Calculate take profit based on fixed points.

   //--- Print the swing point (pivot low) used as stop loss.
   Print("Buy signal: Swing Low used as Stop Loss = ", stopLoss); //--- Notify the user of the pivot low used.

   if (obj_Trade.Buy(InpLotSize, NULL, entryPrice, stopLoss, tp, "Buy Order")) { //--- Attempt to open a Buy order.
      Print("Buy order opened at ", entryPrice); //--- Notify the user if the order is opened successfully.
   }
   else {
      Print("Buy order failed: ", obj_Trade.ResultRetcodeDescription()); //--- Notify the user if the order fails.
   }

   return; //--- Exit after processing a valid Buy signal.
}
else {
   Print("Buy signal not executed due to failed condition(s)."); //--- Notify the user if Buy conditions failed.
}
```

After confirming that all buy conditions are met, we proceed to determine the stop loss for the trade. We use the "GetPivotLow" function to find the previous swing low, which is set as the stop loss. If no valid pivot low is found (i.e., stop loss is less than or equal to 0), the current bid price is used as a fallback. The entry price is taken from the current ask price using the [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) function with the [SYMBOL\_ASK](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double) parameter. We calculate the take profit ("tp") by adding the specified "InpTakeProfitPoints" to the entry price, adjusted by the market's point value ( [\_Point](https://www.mql5.com/en/docs/predefined/_point)).

Once the entry price, stop loss, and take profit are determined, a buy order is attempted using the "obj\_Trade.Buy" method. If the buy order is successfully opened, we [Print](https://www.mql5.com/en/docs/common/print) a confirmation message. If the order fails, we provide the failure message and error description using "obj\_Trade.ResultRetcodeDescription". If the buy conditions are not met, a message indicating that the buy signal was rejected is printed, and no trade is opened. We used a "GetPivotLow" custom function and its implementation is as below.

```
//+------------------------------------------------------------------+
//| Function to find the most recent swing low (pivot low)           |
//+------------------------------------------------------------------+
double GetPivotLow(int left, int right) {
   MqlRates rates[]; //--- Declare an array to store rate data.
   int copied = CopyRates(_Symbol, _Period, 0, 100, rates); //--- Copy the last 100 bars into the rates array.
   if (copied <= (left + right)) { //--- Check if sufficient data was copied.
      return (0); //--- Return 0 if there are not enough bars.
   }

   //--- Loop through the bars to find a pivot low.
   for (int i = left; i <= copied - right - 1; i++) {
      bool isPivot = true; //--- Assume the current bar is a pivot low.
      double currentLow = rates[i].low; //--- Get the low value of the current bar.
      for (int j = i - left; j <= i + right; j++) { //--- Loop through neighboring bars.
         if (j == i) { //--- Skip the current bar.
            continue;
         }
         if (rates[j].low <= currentLow) { //--- If any neighbor's low is lower or equal,
            isPivot = false; //--- then the current bar is not a pivot low.
            break;
         }
      }
      if (isPivot) { //--- If a pivot low is confirmed,
         return (currentLow); //--- return the low value of the pivot.
      }
   }
   return (0); //--- Return 0 if no pivot low is found.
}
```

In the function, we aim to identify the most recent swing low (pivot low) by scanning the last 100 bars of market data using the [CopyRates](https://www.mql5.com/en/docs/series/copyrates) function. The array "rates" of the [MqlRates](https://www.mql5.com/en/docs/constants/structures/mqlrates) structure, holds the price data, and we ensure there are enough bars to perform the calculation by checking if the data copied is greater than or equal to the sum of the "left" and "right" parameters. The function then loops through the bars, checking for a pivot low by comparing the current bar's low value with the neighboring bars within the specified "left" and "right" range. If any neighboring bar's low is lower or equal to the current bar's low, the current bar is not a pivot low. If a pivot low is found, its low value is returned. If no pivot low is found after checking all the bars, the function returns 0.

To get the pivot high, we use a similar approach, only with inversed logic.

```
//+------------------------------------------------------------------+
//| Function to find the most recent swing high (pivot high)         |
//+------------------------------------------------------------------+
double GetPivotHigh(int left, int right) {
   MqlRates rates[]; //--- Declare an array to store rate data.
   int copied = CopyRates(_Symbol, _Period, 0, 100, rates); //--- Copy the last 100 bars into the rates array.
   if (copied <= (left + right)) { //--- Check if sufficient data was copied.
      return (0); //--- Return 0 if there are not enough bars.
   }

   //--- Loop through the bars to find a pivot high.
   for (int i = left; i <= copied - right - 1; i++) {
      bool isPivot = true; //--- Assume the current bar is a pivot high.
      double currentHigh = rates[i].high; //--- Get the high value of the current bar.
      for (int j = i - left; j <= i + right; j++) { //--- Loop through neighboring bars.
         if (j == i) { //--- Skip the current bar.
            continue;
         }
         if (rates[j].high >= currentHigh) { //--- If any neighbor's high is higher or equal,
            isPivot = false; //--- then the current bar is not a pivot high.
            break;
         }
      }
      if (isPivot) { //--- If a pivot high is confirmed,
         return (currentHigh); //--- return the high value of the pivot.
      }
   }
   return (0); //--- Return 0 if no pivot high is found.
}
```

Armed with the functions, we can process the sell trade signals using a similar inversed approach as we did for the buy positions.

```
//--- Check for Sell Conditions:
bool maCrossoverSell    = (ma11_previous > ma25_previous) && (ma11_current < ma25_current); //--- True if fast MA crosses below slow MA.
bool rsiConditionSell   = (rsi_current < (100.0 - InpRSIThreshold)); //--- True if RSI is below the Sell threshold.
bool cci36ConditionSell = (cci36_current < 0); //--- True if CCI36 is negative.
bool cci55ConditionSell = (cci55_current < 0); //--- True if CCI55 is negative.

if (maCrossoverSell) { //--- If crossover for MA Sell is true...
   bool conditionsOk = true; //--- Initialize a flag to track if all conditions are met.

   //--- Check RSI condition for Sell.
   if (!rsiConditionSell) { //--- If the RSI condition is not met...
      Print("Sell signal rejected: RSI condition not met. RSI=", rsi_current, " Required below=", (100.0 - InpRSIThreshold)); //--- Notify the user.
      conditionsOk = false; //--- Mark the conditions as not met.
   }
   //--- Check CCI36 condition for Sell.
   if (!cci36ConditionSell) { //--- If the CCI36 condition is not met...
      Print("Sell signal rejected: CCI36 condition not met. CCI36=", cci36_current); //--- Notify the user.
      conditionsOk = false; //--- Mark the conditions as not met.
   }
   //--- Check CCI55 condition for Sell.
   if (!cci55ConditionSell) { //--- If the CCI55 condition is not met...
      Print("Sell signal rejected: CCI55 condition not met. CCI55=", cci55_current); //--- Notify the user.
      conditionsOk = false; //--- Mark the conditions as not met.
   }

   if (conditionsOk) { //--- If all Sell conditions are met...
      //--- Get stop loss from previous swing high.
      double stopLoss = GetPivotHigh(PivotLeft, PivotRight); //--- Use pivot high as the stop loss.
      if (stopLoss <= 0) { //--- If no valid pivot high is found...
         stopLoss = SymbolInfoDouble(_Symbol, SYMBOL_ASK); //--- Fallback to current ask price.
      }

      double entryPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID); //--- Determine entry price as current bid price.
      double tp         = entryPrice - InpTakeProfitPoints * _Point; //--- Calculate take profit based on fixed points.

      //--- Print the swing point (pivot high) used as stop loss.
      Print("Sell signal: Swing High used as Stop Loss = ", stopLoss); //--- Notify the user of the pivot high used.

      if (obj_Trade.Sell(InpLotSize, NULL, entryPrice, stopLoss, tp, "Sell Order")) { //--- Attempt to open a Sell order.
         Print("Sell order opened at ", entryPrice); //--- Notify the user if the order is opened successfully.
      }
      else {
         Print("Sell order failed: ", obj_Trade.ResultRetcodeDescription()); //--- Notify the user if the order fails.
      }

      return; //--- Exit after processing a valid Sell signal.
   }
   else {
      Print("Sell signal not executed due to failed condition(s)."); //--- Notify the user if Sell conditions failed.
   }
}
```

Here, we check the conditions for a Sell signal by reversing the logic used for Buy signals. We start by checking if the fast-moving average crosses below the slow-moving average, which is captured by the "maCrossoverSell" condition. Next, we verify if the RSI is below the Sell threshold, and check that both CCI36 and CCI55 values are negative.

If all conditions are met, we calculate the stop loss using the "GetPivotHigh" function to find the most recent swing high and determine the take profit based on a fixed point distance. We then attempt to open a Sell order using the "obj\_Trade.Sell" method. If the order is successful, we print a confirmation message; if not, we display an error message. If any condition fails, we notify the user that the Sell signal has been rejected. Upon compilation and running the program, we get the following outcome.

![BUY TRADE CONFIRMED.](https://c.mql5.com/2/119/Screenshot_2025-02-17_173529.png)

From the image, we can see that the program identifies and verifies all the entry conditions and if validated, opens the respective position with the respective entry parameters, hence achieving our objective. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

On backtesting the program intensively, we did notice that when finding swing points, we used the oldest data first for comparison, which led to invalid swing points sometimes though rare, and hence resulting in invalid stops for the stop loss.

![INVALID STOPS ERROR](https://c.mql5.com/2/119/Screenshot_2025-02-17_180540.png)

To mitigate the issue, we adopted an approach where we set the search data as a time series using the [ArraySetAsSeries](https://www.mql5.com/en/docs/array/arraysetasseries) function, so we have the latest data at the first position in the storage arrays, and hence we use the latest data first for analysis as follows.

```
//+------------------------------------------------------------------+
//| Function to find the most recent swing low (pivot low)           |
//+------------------------------------------------------------------+
double GetPivotLow(int left, int right) {
   MqlRates rates[]; //--- Declare an array to store rate data.
   ArraySetAsSeries(rates, true);

//---

}
```

Upon further testing for confirmation, we have the following outcome.

![CORRECTED SWING POINT](https://c.mql5.com/2/119/Screenshot_2025-02-17_181920.png)

From the image, we can see that we correctly get the actual recent swing points, hence getting rid of the "invalid stops" error. Thus, we will not get locked out of trades and after thorough testing, from 2023 to 2024, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/119/TesterGraphReport2025.02.17.png)

Backtest report:

![REPORT](https://c.mql5.com/2/121/Screenshot_2025-02-23_152633.png)

### Conclusion

In conclusion, we have successfully developed an [MQL5](https://www.mql5.com/) Expert Advisor designed to automate a comprehensive Trend Flat Momentum trading strategy that combines multiple trend and momentum indicators for both Buy and Sell signals. By incorporating key conditions such as indicator crossovers and threshold checks, we have created a dynamic system that reacts to market trends with precise entry and exit points.

Disclaimer: This article is for educational purposes only. Trading involves significant financial risk, and market conditions can change rapidly. While the strategy provided offers a structured approach to trading, it does not guarantee profitability. Thorough backtesting and sound risk management are crucial before applying this system in live environment.

By implementing these concepts, you can enhance your algorithmic trading skills and refine your approach to technical analysis. Best of luck as you continue developing and improving your trading strategies!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17247.zip "Download all attachments in the single ZIP archive")

[Trend\_Flat\_Momentum\_EA.mq5](https://www.mql5.com/en/articles/download/17247/trend_flat_momentum_ea.mq5 "Download Trend_Flat_Momentum_EA.mq5")(18.36 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/482155)**

![Neural Network in Practice: Sketching a Neuron](https://c.mql5.com/2/88/Neural_network_in_practice_Sketching_a_neuron___LOGO.png)[Neural Network in Practice: Sketching a Neuron](https://www.mql5.com/en/articles/13744)

In this article we will build a basic neuron. And although it looks simple, and many may consider this code completely trivial and meaningless, I want you to have fun studying this simple sketch of a neuron. Don't be afraid to modify the code, understanding it fully is the goal.

![The Kalman Filter for Forex Mean-Reversion Strategies](https://c.mql5.com/2/121/The_Kalman_Filter_for_Forex_Mean-Reversion_Strategies__LOGO__2.png)[The Kalman Filter for Forex Mean-Reversion Strategies](https://www.mql5.com/en/articles/17273)

The Kalman filter is a recursive algorithm used in algorithmic trading to estimate the true state of a financial time series by filtering out noise from price movements. It dynamically updates predictions based on new market data, making it valuable for adaptive strategies like mean reversion. This article first introduces the Kalman filter, covering its calculation and implementation. Next, we apply the filter to a classic mean-reversion forex strategy as an example. Finally, we conduct various statistical analyses by comparing the filter with a moving average across different forex pairs.

![Artificial Algae Algorithm (AAA)](https://c.mql5.com/2/89/logo-midjourney_image_15565_402_3881__3.png)[Artificial Algae Algorithm (AAA)](https://www.mql5.com/en/articles/15565)

The article considers the Artificial Algae Algorithm (AAA) based on biological processes characteristic of microalgae. The algorithm includes spiral motion, evolutionary process and adaptation, which allows it to solve optimization problems. The article provides an in-depth analysis of the working principles of AAA and its potential in mathematical modeling, highlighting the connection between nature and algorithmic solutions.

![William Gann methods (Part II): Creating Gann Square indicator](https://c.mql5.com/2/89/logo-midjourney_image_15566_400_3863__3.png)[William Gann methods (Part II): Creating Gann Square indicator](https://www.mql5.com/en/articles/15566)

We will create an indicator based on the Gann's Square of 9, built by squaring time and price. We will prepare the code and test the indicator in the platform on different time intervals.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/17247&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049385487371709073)

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