---
title: Formulating Dynamic Multi-Pair EA (Part 2): Portfolio Diversification and Optimization
url: https://www.mql5.com/en/articles/16089
categories: Trading Systems, Indicators
relevance_score: 9
scraped_at: 2026-01-22T17:34:28.197411
---

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/16089&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049225048868366193)

MetaTrader 5 / Tester


### Introduction

One of the most persistent challenges in professional trading lies in maintaining portfolio consistency and robust risk management protocols. Traders often exhibit over-reliance on singular assets or strategies, heightening exposure to substantial draw-downs during abrupt shifts in market regimes. Compounding this risk is the prevalent tendency to over-leverage correlated instruments, which amplifies the likelihood of concurrent losses and undermines the stability of returns. Absent a rigorously diversified and optimized portfolio, traders face erratic performance outcomes, often precipitating emotion-driven decision-making and volatile profitability. A systematic framework that strategically balances risk-adjusted returns across a spectrum of uncorrelated assets is therefore indispensable for sustainable long-term performance.

To mitigate these challenges, a quantitative methodology integrating portfolio optimization, multi-asset diversification, and a breakout trading strategy enhanced by oscillator-based confirmation offers a robust solution. By deploying a breakout strategy across multiple currency pairs, traders can capitalize on high-probability momentum-driven price movements while dispersing risk through exposure to non-correlated markets. The integration of an oscillator indicator serves to validate entry signals, minimizing false breakout participation and curtailing unproductive trades. This approach not only elevates profit potential but also fortifies portfolio stability by systematically exploiting opportunities across divergent market phases. The resultant strategy demonstrates heightened resilience to volatility, ensuring consistent performance alignment with evolving macroeconomic and technical conditions.

### Expert Logic

Buy Model:

The Expert Advisor begins by calculating the price range between 10:00 AM and 12:00 PM UTC+2, identifying the highest high and lowest low within this timeframe. At 12:00 PM or any time after, if the price breaks above the previously established high, a potential buy opportunity is triggered. However, for confirmation, the Stochastic Oscillator must be at or below the 20 level, indicating an oversold market condition. This ensures that the breakout is not overextended and has room for upward momentum. Once both conditions are met, the EA executes a buy trade, aiming to capitalize on the breakout while minimizing false signals through oscillator-based filtering.

![](https://c.mql5.com/2/127/33.png)

Sell Model:

For the sell model, the Expert Advisor follows the same range calculation process from 10:00 AM to 12:00 PM UTC+2, identifying the session’s high and low. If at 12:00 PM or later, the price breaks below the established low, a sell opportunity is considered. However, the trade is only executed if the Stochastic Oscillator is at or above the 80 level, indicating an overbought market condition. This ensures that the downward breakout is backed by potential selling pressure rather than a false move. Once both conditions align, the EA enters a sell trade, capitalizing on bearish momentum while avoiding premature entries.

![](https://c.mql5.com/2/127/selll.png)

Getting started:

```
//+------------------------------------------------------------------+
//|                                                      Dyna MP.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <Trade/Trade.mqh>
CTrade trade;

enum Signal_Breakout{
   Normal_Signal,
   Reversed_Signal,
};
```

\`#include <Trade/Trade.mqh>\` This line includes the MQL5 Trade library. The \`Trade.mqh\` header file defines classes and functions that streamline trading tasks such as initiating, adjusting, or closing orders. Here, an object named \`trade\` is instantiated from the \`CTrade\` class. This class contains methods for executing trading operations (e.g., submitting buy/sell requests, modifying open trades, or closing positions) in a structured manner.

```
input group "--------------General Inputs--------------"
input string Symbols = "XAUUSD, GBPUSD, USDCAD, USDJPY";
input Signal_Breakout BreakOutMode = Reversed_Signal;
input double  In_Lot = 0.01;
input int TakeProfit = 500;
 double StopLoss = 500;
input bool TrailYourStop = false;
input int TrailingStop = 50;

// Stochastic
input int KPeriod = 21;
input int upprer_level = 80;
```

\`Symbols\` specifies the assets to trade (e.g., "XAUUSD, GBPUSD"), \`In\_Lot\` sets the fixed trade size (0.01 lots), **\`** TakeProfit **\`** (500 points) and \`StopLoss\` (1000 points) define profit-taking and risk limits, while \`TrailYourStop\` and \`Trailing Stop\` (70 points) activate and adjust dynamic stop-loss trailing. For the Stochastic Oscillator, **\`** KPeriod **\`** (20) determines the calculation period for the percentage line, and\`upper\_level\` (80) marks the overbought threshold. These inputs collectively balance trade execution rules, risk management, and technical signal generation.

```
//+------------------------------------------------------------------+
//|                           Global vars                            |
//+------------------------------------------------------------------+
int handles[];
double bufferM[];

int RangeStart = 600;
int RangeDuration = 120;
int RangeClose = 1200;

int Num_symbs = 0;
string symb_List[];
string Formatted_Symbs[];
```

This section defines global variables. The \`handles\` array stores handles for the Stochastic Oscillator, which enables the EA to manage multiple assets efficiently. The \`buffer\` array store calculated values, such as indicator readings or historical price levels. The \`RangeStart\` variable is set to 600 minutes (10:00 AM UTC+2), marking the beginning of the range measurement, while \`RangeDuration\` is set to 120 minutes, defining a 2-hour period for capturing the high and low. Once the \`RangeClose\` time of 1200 minutes (12:00 PM UTC+2) is reached, the EA stops calculating the range and begins monitoring for breakout conditions.

For symbol management, the \`symb\_List\` array holds the raw list of symbols to be processed. Additionally, the \`Formatted\_Symbs\` array holds the total number of symbols to be used after parsing the \`Symbols\` input. These variables collectively allow the EA to execute trades dynamically across multiple assets while ensuring precise range calculations for breakout detection.

```
//+------------------------------------------------------------------+
//|                    Ranger Global Vars                            |
//+------------------------------------------------------------------+
struct RANGER{
   datetime start_time;
   datetime end_time;
   datetime close_time;
   double high;
   double low;
   bool b_entry;
   bool b_high_breakout;
   bool b_low_breakout;

   RANGER() : start_time(0), end_time(0), close_time(0), high(0), low(999999), b_entry(false), b_high_breakout(false), b_low_breakout(false) {};
};
```

The \`Ranger\` struct stores and manages key data points for the range breakout strategy. It defines a structured way to track the range's time boundaries, price levels, and breakout conditions. The variables \`start\_time\`, \`end\_time\`, and \`close\_time\` represent the start, end, and closing time of the range calculation, ensuring the EA correctly identifies the breakout window. The \`high\` and \`low\` variables store the highest and lowest prices recorded within the range, with \`low\` being initialized to a very high value (999999) to ensure accurate price updates. The \`low (999999)\` is a standard pattern for finding minimum values in price series.

Additionally, we three boolean flags. The \`b\_entry\` tracks whether a trade entry has been executed, preventing multiple trades from being placed within the same breakout event. \`b\_high\_breakout\` signals whether the price has broken above the range high, confirming a potential buy setup, and \`b\_low\_breakout\` indicates if the price has broken below the range low, confirming a sell setup. The constructor \`Ranger()\` initializes all values to default states, ensuring the struct starts with clean data before being updated dynamically during live trading.

```
RANGER rangeArray[];
MqlTick prevTick[], currTick[];
```

These variables play crucial roles in tracking price action and managing the range breakout logic in the Expert Advisor. The \`rangeArray is an array of \`RANGER\` structures, it stores multiple instances of the defined range breakout parameters. This allows the EA to track multiple symbols simultaneously, ensuring that each symbol has its own range data, including start and end times, high and low prices, and breakout conditions.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){

   string separator = ",";
   ushort usprtr;
   usprtr = StringGetCharacter(separator, 0);
   StringSplit(Symbols, usprtr, symb_List);
   Num_symbs = ArraySize(symb_List);
   ArrayResize(Formatted_Symbs, Num_symbs);

   for(int i = 0; i < Num_symbs; i++){
      Formatted_Symbs[i] = symb_List[i];
   }

   ArrayResize(rangeArray, Num_symbs);
   ArrayResize(prevTick, Num_symbs);
   ArrayResize(currTick, Num_symbs);
   ArrayResize(handles, Num_symbs);
   ArraySetAsSeries(bufferM, true);

   // Calculate initial ranges for each symbol
   for (int i = 0; i < ArraySize(Formatted_Symbs); i++) {
      CalculateRange(i, Formatted_Symbs[i]);  // Pass the symbol index
      handles[i] = iStochastic(Formatted_Symbs[i], PERIOD_CURRENT, KPeriod, 1, 3, MODE_SMA, STO_LOWHIGH);

      if(handles[i] == INVALID_HANDLE){
         Alert("Failed to create indicator handle");
         return INIT_FAILED;
      }

      StopLoss = SymbolInfoDouble(Formatted_Symbs[i], SYMBOL_POINT)*TrailingStop;
   }
   return(INIT_SUCCEEDED);
}
```

The \`OnInit\` function is responsible for initializing and setting up key variables, handling symbols, and preparing the required data structures. We still use separator \`,\` to split the symbol list into individual tradable assets, which are stored in \`symb\_List\`. We determine the total number of symbols \`Num\_symbs\` and then resize the \`Formatted\_Symbs\` array accordingly, and ensuring that each symbol is properly formatted and accessible for trading.

Next, the EA dynamically resizes several arrays, including \`rangeArray\` to store range breakout data, \`prevTick\` and \`currTick\` to track price updates, \`handles\` for Stochastic Oscillator, and \`bufferM\` for indicator values, ensuring that all required data structures are properly allocated before the EA starts execution.

Once the arrays are set up, the EA loops through each symbol in \`Formatted\_Symbs\` to calculate the initial range using the \`CalculateRange()\` function. It also creates a Stochastic Oscillator handle (istochastic) for each symbol, which we will use later to confirm breakouts by checking overbought and oversold conditions. If the indicator handle fails to initialize (INVALID\_HANDLE), the EA triggers an alert and exits with an INIT\_FAILED status. Additionally, we calculate the Stop Loss for each symbol based on its **point value** and a trailing stop parameter. If everything is successfully initialized, the function returns INIT\_SUCCEEDED, allowing the EA to proceed with monitoring price movements and executing trades.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason){
   for(int i = 0; i < ArraySize(Formatted_Symbs); i++){
      if(handles[i] != INVALID_HANDLE){
         IndicatorRelease(handles[i]);
      }
   }
}
```

The OnDeinit() function is the de-initialization function of the Expert Advisor (EA), responsible for properly releasing resources when the EA is removed from the chart or stops running.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){

   for(int i = 0; i < ArraySize(Formatted_Symbs); i++){
      string symbol = Formatted_Symbs[i];
      prevTick[i] = currTick[i];
      SymbolInfoTick(symbol, currTick[i]);

      // Range Cal
      if(currTick[i].time > rangeArray[i].start_time && currTick[i].time < rangeArray[i].end_time){
         // flag
         rangeArray[i].b_entry = true;

         // high
         if(currTick[i].ask > rangeArray[i].high){
            rangeArray[i].high = currTick[i].ask;
         }

         // low
         if(currTick[i].bid < rangeArray[i].low){
            rangeArray[i].low = currTick[i].bid;
         }
      }

      // now calculate range
      if(((RangeClose >= 0 && currTick[i].time >= rangeArray[i].close_time)
         || (rangeArray[i].b_high_breakout && rangeArray[i].b_low_breakout)
         || (rangeArray[i].end_time == 0)
         || (rangeArray[i].end_time != 0 && currTick[i].time > rangeArray[i].end_time && !rangeArray[i].b_entry))){
         CalculateRange(i, Formatted_Symbs[i]);
      }
      checkBreak(i, Formatted_Symbs[i]);
   }

}
```

The \`OnTick\` function is the core execution loop of the Expert Advisor (EA), running on every new tick update, as we all know. It manages live price analysis, adjusts range parameters, and assesses breakout opportunities across all configured trading instruments. The process initiates by sequentially evaluating each symbol in the \`Formatted\_Symbs\` list. For each symbol, it archives the current tick data into a historical buffer ( _prevTick_) and retrieves the most recent market state via _\`SymbolInfoTick\`._

During the predefined active trading window ( _start\_time_ to _end\_time_), the system dynamically tracks price extremes. It elevates the session high using the peak _ask_ value and reduces the session low using the trough _bid_ value. If the current time falls within this range, the _b\_entry_ flag activates, designating the range as valid for trading.

Post trading window, the logic evaluates three renewal triggers:

1. Scheduled session closure \`(close\_time)\`
2. Concurrent high and low breakout occurrences
3. Invalid or expired \`end\_time\` timestamps

If any condition is met, _\`CalculateRange()\`_ regenerates fresh range parameters. The cycle concludes by invoking _\`checkBreak()\`_, which scans for price breaches beyond range boundaries and initiates trades accordingly. This framework enables continuous market surveillance and strategic order placement aligned with breakout dynamics, ensuring responsiveness to real-time price movements.

```
//+------------------------------------------------------------------+
//|                  Range Calculation function                      |
//+------------------------------------------------------------------+
void CalculateRange(int index, string symbol) {
   for(index = 0; index < ArraySize(Formatted_Symbs); index++){
      symbol = Formatted_Symbs[index];

      // Reset all the range variables
      rangeArray[index].start_time = 0;
      rangeArray[index].end_time = 0;
      rangeArray[index].close_time = 0;
      rangeArray[index].high = 0.0;
      rangeArray[index].low = 999999;
      rangeArray[index].b_entry = false;
      rangeArray[index].b_high_breakout = false;
      rangeArray[index].b_low_breakout = false;

      // Calculate range start time
      int time_cycle = 86400;
      rangeArray[index].start_time = (currTick[index].time - (currTick[index].time % time_cycle)) + RangeStart * 60;
      for(int i = 0; i < 8; i++){
         MqlDateTime tmp;
         TimeToStruct(rangeArray[index].start_time, tmp);
         int dotw = tmp.day_of_week;
         if(currTick[index].time >= rangeArray[index].start_time || dotw == 6 || dotw == 0){
            rangeArray[index].start_time += time_cycle;
         }
      }

      // Calculate range end time
      rangeArray[index].end_time = rangeArray[index].start_time + RangeDuration * 60;
      for(int i = 0 ; i < 2; i++){
         MqlDateTime tmp;
         TimeToStruct(rangeArray[index].end_time, tmp);
         int dotw = tmp.day_of_week;
         if(dotw == 6 || dotw == 0){
            rangeArray[index].end_time += time_cycle;
         }
      }

      // Calculate range close
      rangeArray[index].close_time = (rangeArray[index].end_time - (rangeArray[index].end_time % time_cycle)) + RangeClose * 60;
      for(int i = 0; i < 3; i++){
         MqlDateTime tmp;
         TimeToStruct(rangeArray[index].close_time, tmp);
         int dotw = tmp.day_of_week;
         if(rangeArray[index].close_time <= rangeArray[index].end_time || dotw == 6 || dotw == 0){
            rangeArray[index].close_time += time_cycle;
         }
      }

   }
}
```

The \`CalculateRange\` function initializes and configures time-based trading ranges for multiple symbols. For each symbol in the \`Formatted\_Symbs\` array, it first resets critical range parameters—start/end/close times, high/low price thresholds, and breakout flags—to default values. The \`start\_time\` is calculated by aligning the current tick time to a daily boundary (using a 24-hour time\_cycle), then offsetting it by a user-defined \`RangeStart\` (in minutes). A loop ensures the start time avoids weekends (Saturday/Sunday) and remains chronologically valid by incrementing the timestamp by full days if conflicts arise. This creates a baseline for the trading window while respecting market closure periods.

After setting the range start time, the function calculates the range end time by adding \`RangeDuration\` to \`start\_time\`. Similar to the start time calculation, it ensures that the end time does not fall on a weekend by iterating through a validation loop. Finally, the function determines the range close time, which marks the point at which the EA stops monitoring for breakouts. This time is derived from \`RangeClose\` and is adjusted to avoid weekends. By maintaining these dynamic calculations, the function ensures that the EA accurately sets up range trading conditions across different symbols, avoiding trading during non-trading hours and weekends while ensuring precise breakout detection.

```
bool CLots(double sl, double &lots){
   lots = In_Lot;

   if(!CHLots(lots)){return false;}
   return true;
}
```

The \`CLots()\` function sets and validates the lot size before a trade is executed, it takes two parameters which are your stop loss (sl) and lots (a reference variable that stores the final lot size).

```
bool CHLots(double &lots){
   for(int i = 0; i < ArraySize(Formatted_Symbs); i++){
      string symbol = Formatted_Symbs[i];
      double min = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
      double max = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
      double step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
      if(lots < min){
         lots = min;
         return true;
      }
      if(lots > max){
         return false;
      }

      lots = (int)MathFloor(lots / step)  * step;

   }
   return true;

}
```

The \`CHLots()\` function validates lot sizes to comply with broker-specific trading rules for each instrument. It iterates through the list of symbols in \`Formatted\_Symbs\`, extracting the broker-defined constraints—minimum lot size, maximum lot size, and permissible increment step. These parameters establish the operational boundaries for order volumes. When the proposed lot value falls below the minimum threshold, the function automatically corrects it to the minimum allowable volume and confirms validity by returning true. Conversely, if the requested lot size exceeds the broker’s maximum limit, the function aborts the request by returning false, thereby blocking non-compliant trades.

To ensure precision, the function enforces step alignment by rounding down the lot size using the formula \`MathFloor(lots/step) \* step\`. This eliminates fractional or irregular increments that could trigger broker rejections. If the lot size adheres to all constraints without adjustments, the function returns true, confirming its acceptability. By rigorously enforcing these checks, \`CHLots()\` acts as a critical safeguard, preventing order rejections due to volume violations and bolstering the EA’s operational reliability in live trading environments.

```
//+------------------------------------------------------------------+
//|                      Check for Breakout                          |
//+------------------------------------------------------------------+
void checkBreak(int i, string symbol) {
   for (i = 0; i < ArraySize(Formatted_Symbs); i++) {
      symbol = Formatted_Symbs[i];

      //get indicator vals
      if(CopyBuffer(handles[i], 0, 1, 2, bufferM) != 2){
         Print("Failed to get indicator values");
         return;
      }

      int stopLevel = (int)SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
      int spread = (int)SymbolInfoInteger(symbol, SYMBOL_SPREAD);
      double Bid = SymbolInfoDouble(symbol, SYMBOL_BID);
      double Ask = SymbolInfoDouble(symbol, SYMBOL_ASK);

      if (currTick[i].time >= rangeArray[i].end_time && rangeArray[i].end_time > 0 && rangeArray[i].b_entry) {
         double rangeSize = rangeArray[i].high - rangeArray[i].low;

         // High Breakout (BUY/SELL)
         bool upperBreak = bufferM[0] >= upprer_level && bufferM[1] < upprer_level;
         bool lowerBreak = bufferM[0] <= (100 - upprer_level) && bufferM[1] > (100 - upprer_level);
         bool HighSigType,LowSigType;

         if(BreakOutMode == Normal_Signal){
            HighSigType = upperBreak;
         }else{HighSigType = lowerBreak;}
         if (!rangeArray[i].b_high_breakout && currTick[i].ask >= rangeArray[i].high && HighSigType) {
            rangeArray[i].b_high_breakout = true;

            double entry = NormalizeDouble(Ask + 100 * _Point, _Digits);
            double sl = rangeArray[i].low;
            //sl = NormalizeDouble(sl, true);
            double tp = entry + TakeProfit * _Point;

            double lots;
            if (!CLots(entry - sl, lots)) continue;

            if (!trade.PositionOpen(symbol, ORDER_TYPE_BUY, lots, currTick[i].ask, sl, tp, "High Breakout"))
               Print("Buy Order Failed: ", GetLastError());
         }

         if(BreakOutMode == Normal_Signal){
            LowSigType = upperBreak;
         }else{LowSigType = lowerBreak; }
         // Low Breakout (SELL)
         if (!rangeArray[i].b_low_breakout && currTick[i].bid <= rangeArray[i].low  && LowSigType) {
            rangeArray[i].b_low_breakout = true;

            double entry = NormalizeDouble(Bid - 100 * _Point, _Digits);
            double sl = rangeArray[i].high;
            //sl = NormalizeDouble(sl,true);
            double tp = entry - TakeProfit * _Point;

            double lots;
            if (!CLots(sl - entry, lots)) continue;

            if (!trade.PositionOpen(symbol, ORDER_TYPE_SELL, lots, currTick[i].bid, sl, tp, "Low Breakout"))
               Print("Sell Order Failed: ", GetLastError());
         }
      }
   }
}
```

This function monitors price breakouts and manages trade execution by combining price movements with Stochastic Oscillator signals. It cycles through each symbol in _\`Formatted\_Symbs_ \`, first attempting to retrieve the oscillator’s values via _\`CopyBuffer()\`_. If this data retrieval fails, the function logs an error and terminates to avoid flawed decisions. For each symbol, critical parameters like stop levels, spread, and current bid/ask prices are gathered. Breakout evaluations occur only after the range period concludes, verified by checking if the current time exceeds the range’s _\`end\_time\`_ and the _\`b\_entry\`_ flag is active, ensuring analysis is confined to valid trading windows.

For high breakouts (buy signal), the function confirms two criteria, which are the ask price surpasses the session’s high, and the oscillator reflects oversold conditions (below _100 - upper\_level_). Upon validation, it triggers a buy order—calculating entry price, stop-loss (based on range volatility), and take-profit levels—then verifies lot size compliance via _\`CLots()_ \`.

Conversely, for low breakout (sell signal) require the bid price to drop below the session’s low while the oscillator indicates overbought status (above _upper\_level_). If met, a sell order is generated with analogous risk parameters. Both scenarios include error logging for failed orders, ensuring transparency. By synchronizing price thresholds with oscillator-based confirmation, the function enforces disciplined, criteria-driven trade execution.

```
//+------------------------------------------------------------------+
//|                      Trailing Stoploss                           |
//+------------------------------------------------------------------+
void Trailler(){
   if(!TrailYourStop) return;

   for(int i = PositionsTotal()-1; i >= 0; i--){
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0) continue;

      if(!PositionSelectByTicket(ticket)) continue;

      // Get position details
      string symbol = PositionGetString(POSITION_SYMBOL);
      long magic;
      if(!PositionGetInteger(POSITION_MAGIC, magic)) continue;
      if(magic != MagicNumber) continue;

      // Get current prices
      MqlTick latestTick;
      if(!SymbolInfoTick(symbol, latestTick)) continue;

      long type;
      double openPrice, currentSl, currentTp;
      PositionGetInteger(POSITION_TYPE, type);
      PositionGetDouble(POSITION_PRICE_OPEN, openPrice);
      PositionGetDouble(POSITION_SL, currentSl);
      PositionGetDouble(POSITION_TP, currentTp);

      // Calculate pip values
      double pipSize = 10 * SymbolInfoDouble(symbol, SYMBOL_POINT);
      double currentPrice = type == POSITION_TYPE_BUY ? latestTick.bid : latestTick.ask;
      double priceMove = MathAbs(currentPrice - openPrice);

      // Calculate required moves
      double requiredMove = 70 * pipSize; // 20 pips
      double trailAmount = 10 * pipSize;  // 10 pips

      // Calculate new stop loss
      double newSl = currentSl;
      bool inProfit = type == POSITION_TYPE_BUY ?
                     (currentPrice > openPrice) :
                     (currentPrice < openPrice);

      if(inProfit && priceMove >= requiredMove){
         int steps = int(priceMove / requiredMove);
         if(type == POSITION_TYPE_BUY){
             newSl = openPrice + (steps * trailAmount);
             newSl = MathMax(newSl, currentSl + trailAmount);
         }
         else{
             newSl = openPrice - (steps * trailAmount);
             newSl = MathMin(newSl, currentSl - trailAmount);
         }
      }

      // Validate and modify SL
      if(newSl != currentSl){
         // Check stop levels
         double minDist = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
         newSl = NormalizeDouble(newSl, _Digits);

         if(type == POSITION_TYPE_BUY && (currentPrice - newSl) >= minDist){
             if(!trade.PositionModify(ticket, newSl, currentTp))
                 Print("Buy Trailing Failed: ", GetLastError());
         }
         else if(type == POSITION_TYPE_SELL && (newSl - currentPrice) >= minDist){
             if(!trade.PositionModify(ticket, newSl, currentTp))
                 Print("Sell Trailing Failed: ", GetLastError());
         }
      }
   }
}
```

The \`Trailing Stop Loss\` (TSL) function in trading ensures that once a trade moves favorably, the stop-loss automatically adjusts to **lock in profits** while minimizing risks. The \`Trailer()\` function recalculates and updates the stop-loss level based on price movement and a percentage of the high-low range, preventing premature exits while securing gains.

### Conclusion

In summary, portfolio optimization and diversification are essential strategies in trading that aim to maximize returns while minimizing risk by spreading investments across multiple assets. Traditional trading methods often focus on single-pair strategies, exposing traders to higher volatility and market-specific risks. Diversification strengthens portfolio resilience, while optimization methods strategically deploy capital by analyzing historical performance patterns, asset volatility metrics, and inter-market relationships to achieve an efficient risk-reward balance.

In conclusion, incorporating portfolio optimization and diversification into a trading strategy provides a more resilient and adaptive approach to market fluctuations. By combining a breakout trading strategy with an oscillator indicator, traders can identify high-probability setups while managing risk dynamically. This approach increases the chances of consistent profitability but also enhances long-term sustainability by mitigating draw-downs.

To achieve the following results, the Expert Advisor was tested using the **EURUSD** as a base symbol with **zero latency delays**, **ideal execution**, and **every tick modeling** for maximum accuracy. The testing period ranged from **2022.02.01 to 2022.03.22**. In the input settings, the **BreakoutMode** was set to **Reversed\_signal**, and **TrailYourStop** was enabled ( **set to true**) to allow dynamic stop-loss adjustments. All other input parameters were left at their default values.

![](https://c.mql5.com/2/128/AT3.png)

![](https://c.mql5.com/2/128/ATb.png)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16089.zip "Download all attachments in the single ZIP archive")

[Dyna\_MP.mq5](https://www.mql5.com/en/articles/download/16089/dyna_mp.mq5 "Download Dyna_MP.mq5")(37.35 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Adaptive Smart Money Architecture (ASMA): Merging SMC Logic With Market Sentiment for Dynamic Strategy Switching](https://www.mql5.com/en/articles/20414)
- [Fortified Profit Architecture: Multi-Layered Account Protection](https://www.mql5.com/en/articles/20449)
- [Analytical Volume Profile Trading (AVPT): Liquidity Architecture, Market Memory, and Algorithmic Execution](https://www.mql5.com/en/articles/20327)
- [Automating Black-Scholes Greeks: Advanced Scalping and Microstructure Trading](https://www.mql5.com/en/articles/20287)
- [Integrating MQL5 with Data Processing Packages (Part 6): Merging Market Feedback with Model Adaptation](https://www.mql5.com/en/articles/20235)
- [Formulating Dynamic Multi-Pair EA (Part 5): Scalping vs Swing Trading Approaches](https://www.mql5.com/en/articles/19989)
- [Black-Scholes Greeks: Gamma and Delta](https://www.mql5.com/en/articles/20054)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/484856)**
(8)


![Alberto Tortella](https://c.mql5.com/avatar/2015/9/55E56F96-BB76.jpg)

**[Alberto Tortella](https://www.mql5.com/en/users/alberto_jazz)**
\|
19 Apr 2025 at 13:48

Ok I wrote [EURUSD](https://www.mql5.com/en/quotes/currencies/eurusd "EURUSD chart: technical analysis") in Input/Symbols and now it works.

Thank you

![CapeCoddah](https://c.mql5.com/avatar/avatar_na2.png)

**[CapeCoddah](https://www.mql5.com/en/users/capecoddah)**
\|
19 Apr 2025 at 20:10

Great Article!  I'm going to try it tomorrow.  I'm interested to know why you used such a strange time period for the [strategy tester](https://www.mql5.com/en/articles/239 "Article: The Fundamentals of Testing in MetaTrader 5 ").  I would have expected full months in 2024.  I like your concept of the trailing stop loss, I am using the same technique. One wrinkle I have made is to also attempt to minimize the loss if the trade turns negative after almost reaching breakeven.

Cheers and keep the articles coming, they are great

CapeCoddah

![Brian Pereira](https://c.mql5.com/avatar/2025/2/67B35DAF-A05C.png)

**[Brian Pereira](https://www.mql5.com/en/users/brianpereira123)**
\|
23 Apr 2025 at 08:07

**Alberto Tortella [#](https://www.mql5.com/en/forum/484856#comment_56491475):**

Good morning, I'm trying to run the expert but I have the following error when the tester starts.

I seem [stochastic oscillator](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/so "MetaTrader 5 Help: Stochastic Oscillator Indicator") works well on graph.

Could you help me? Thank you

Each input currency should be separated only by a comma. Do not put a space between currencies

![CapeCoddah](https://c.mql5.com/avatar/avatar_na2.png)

**[CapeCoddah](https://www.mql5.com/en/users/capecoddah)**
\|
30 Apr 2025 at 11:33

Hi Again,

I have tried to use your system on an active chart and have found a couple of improvements

Alberto's problem was probably he did not have all pairs in the symbol list on his market watch window, ctlM.  I got that error also with XAUUSD

Instead of ArraySize(... for  For statements, use Num\_symbls, a little quicker.  Also I have found that spelling out full names helps others understand  your code better and also prevents a lot of yntax errors, e.g. Number\_Symbols is better in my opinion than Num\_symbls.

DisplayObjects was not in the code, I added it.

In display objects, I added a condition to select only the chart symbol.  Enumerating through the others is not required and will clutter up the screen.  But maybe I am missing something.

Finally there is a problem with the Range Calculation.  On an active chart, not in the [Strategy Tester](https://www.mql5.com/en/articles/239 "Article: The Fundamentals of Testing in MetaTrader 5 "), Starting the EA produces a ray that is beyond the current date in the future.  For example starting on 4/30 produces a ray that starts on 4/30 10am and ends on 5/1.  This results in a non visible ray that doesn't show on the chart but does show in the Objects List.  I'll let you fix this one.

I'm attaching my code for your use

Cheers, CapeCoddah

![CapeCoddah](https://c.mql5.com/avatar/avatar_na2.png)

**[CapeCoddah](https://www.mql5.com/en/users/capecoddah)**
\|
30 Apr 2025 at 13:02

I think something got messed up as the eas from part 1 & part 2 are identical.  It looks like Part is identical to part 1


![Neural Networks in Trading: Transformer for the Point Cloud (Pointformer)](https://c.mql5.com/2/92/Neural_Networks_in_Trading_Transformer_for_Point_Cloud____LOGO.png)[Neural Networks in Trading: Transformer for the Point Cloud (Pointformer)](https://www.mql5.com/en/articles/15820)

In this article, we will talk about algorithms for using attention methods in solving problems of detecting objects in a point cloud. Object detection in point clouds is important for many real-world applications.

![Feature Engineering With Python And MQL5 (Part IV): Candlestick Pattern Recognition With UMAP Regression](https://c.mql5.com/2/134/Feature_Engineering_With_Python_And_MQL5_Part_IV___LOGO__2.png)[Feature Engineering With Python And MQL5 (Part IV): Candlestick Pattern Recognition With UMAP Regression](https://www.mql5.com/en/articles/17631)

Dimension reduction techniques are widely used to improve the performance of machine learning models. Let us discuss a relatively new technique known as Uniform Manifold Approximation and Projection (UMAP). This new technique has been developed to explicitly overcome the limitations of legacy methods that create artifacts and distortions in the data. UMAP is a powerful dimension reduction technique, and it helps us group similar candle sticks in a novel and effective way that reduces our error rates on out of sample data and improves our trading performance.

![Decoding Opening Range Breakout Intraday Trading Strategies](https://c.mql5.com/2/134/Decoding_Opening_Range_Breakout_Intraday_Trading_Strategies__LOGO.png)[Decoding Opening Range Breakout Intraday Trading Strategies](https://www.mql5.com/en/articles/17745)

Opening Range Breakout (ORB) strategies are built on the idea that the initial trading range established shortly after the market opens reflects significant price levels where buyers and sellers agree on value. By identifying breakouts above or below a certain range, traders can capitalize on the momentum that often follows as the market direction becomes clearer. In this article, we will explore three ORB strategies adapted from the Concretum Group.

![Manual Backtesting Made Easy: Building a Custom Toolkit for Strategy Tester in MQL5](https://c.mql5.com/2/134/Manual_Backtesting_Made_Easy_Building_a_Custom_Toolkit_for_Strategy_Tester_in_MQL5__LOGO.png)[Manual Backtesting Made Easy: Building a Custom Toolkit for Strategy Tester in MQL5](https://www.mql5.com/en/articles/17751)

In this article, we design a custom MQL5 toolkit for easy manual backtesting in the Strategy Tester. We explain its design and implementation, focusing on interactive trade controls. We then show how to use it to test strategies effectively

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/16089&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049225048868366193)

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