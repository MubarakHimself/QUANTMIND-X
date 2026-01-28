---
title: Developing Zone Recovery Martingale strategy in MQL5
url: https://www.mql5.com/en/articles/15067
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T18:01:10.780687
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/15067&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049565498041019829)

MetaTrader 5 / Trading


### Introduction

In this article we will create the zone recovery martingale forex trading strategy expert advisor (EA) in [MetaQuotes Language 5](https://www.mql5.com/en) (MQL5) for [MetaTrader 5](https://www.metatrader5.com/en "https://www.metatrader5.com/en") (MT5) step by step. The zone recovery martingale strategy is a common strategy that is aimed at countering losing positions by opening opposite positions with a slightly greater trading volume that cancels the losing positions. It is basically a trend following strategy that does not care about the direction of the market, with the hope that at one point, the market will be trending either downwards or upwards and eventually particular targets will be hit. Join us as we not only discuss but also automate this system in the MQL5.

This journey will cover the following topics;

1. Definition of Zone Recovery strategy
2. Zone Recovery strategy description
3. Implementation in MQL5
4. Backtest results
5. Conclusion

### Definition of Zone Recovery strategy

The zone recovery trading strategy is sophisticated method that is mostly used in forex trading to control and reduce losses. Determining precise price ranges that a trader anticipates the market to fluctuate within is the core idea of this method. The trader launches a series of countertrades to build a recovery zone when a trade moves unfavorably and reaches a predetermined loss zone, as opposed to closing the position at a loss. Ideally, these trades should be placed so that a move back into the recovery zone enables the position as a whole to be closed at breakeven or even profit.

This technique is primarily focused on hedging and averaging down. When the market goes against the first deal, the trader creates an equal-sized opposing position to act as a hedge. If the market continues to move downward, new trades are opened at crucial intervals to average down the entry price. The deals are designed so that the aggregate profit from the recovery trades can equal the losses from the initial trade when the market returns to its mean. This approach necessitates a well-defined risk management strategy, as the possibility of amassing holdings might result in high margin requirements and exposure to volatile market circumstances.

The zone recovery strategy's ability to turn a losing transaction into a successful one without requiring precise market direction prediction is one of its main advantages. It enables traders to profit from market turbulence and reversals, converting unfavorable trends into chances for a rebound.

The zone recovery trading method is most appropriate for seasoned traders with a thorough understanding of risk management strategies and market dynamics. It is a powerful instrument in a trader's toolbox that is especially helpful in erratic markets with frequent price swings. Although it has the ability to recover from failed trades and turn a profit, its complexity and associated hazards necessitate thoughtful planning and calculated execution.

### Zone Recovery strategy description

Based on a market analysis, the zone recovery trading method starts with the opening of an initial position. Imagine a trader who opens a buy position because they believe the market will rise. Making money from the increasing trend is the first goal. The trader closes the position and secures the gains if the market moves positively and the price rises to a predetermined profit objective. The trader can profit from favorable market movements with this simple strategy without having to use any more intricate ones.

On the other hand, the zone recovery method initiates its loss mitigation mechanism if the market moves against the initial long position and hits a predetermined loss point. At this point, the trader opens a sell position with a larger lot size, usually double the size of the previous long position , rather than closing the buy position at a loss. By offsetting the losses from the first trade with the possible gains from the new position, this counter trade seeks to establish a hedge. Using the market's inherent oscillation, the strategy anticipates a reversal or at the very least a stabilization within a given range.

The trader keeps an eye on the fresh sell position as the market moves forward. The combined impact of the initial purchase and larger sell positions ideally results in a breakeven or profit situation if the market continues to decrease and reaches another predetermined point. After then, the trader can close both positions, using the winnings from the larger following trade to offset the losses from the first trade. In order to guarantee that the entire position may be closed profitably within the recovery zone, this strategy necessitates exact computation and timing.

### Implementation in MQL5

To create an expert advisor in MQL5 that is oriented towards visualizing the zone recovery levels, typically four, we will first need to define the levels. We do this by defining them as early as possible, as they will be crucial in visualizing the trading system. This is achieved by using the keyword "#define," which is an in-built directive in MQL5 that can be used to assign mnemonic names to constants. This is as below:

```
#define ZONE_H "ZH"            // Define a constant for the high zone line name
#define ZONE_L "ZL"            // Define a constant for the low zone line name
#define ZONE_T_H "ZTH"         // Define a constant for the target high zone line name
#define ZONE_T_L "ZTL"         // Define a constant for the target low zone line name
```

Here, we define the constants of our zone boundaries: ZONE\_H as 'ZH' (Zone High), ZONE\_L as 'ZL' (Zone Low),  ZONE\_T\_H as 'ZTH' (Zone Target High), and ZONE\_T\_L as 'ZTL' (Zone Target Low). This constants represents the respective levels in our system.

After our definitions, we will need to open trading positions. The easiest way to open positions is to include a trading instance, typically achieved via the inclusion of another file that is dedicated to open positions. We use the include directive to include the trade library, which contains functions for trading operations.

```
#include <Trade/Trade.mqh>
CTrade obj_trade;
```

First, we use the angle brackets to signify that the file we want to include is contained in the include folder and provide the Trade folder, followed by a normal slash or backslash, and then the target file name, in this case, "Trade.MQH". CTrade is a class for handling trade operations, and obj\_trade is an instance of this class, typically a pointer object created from the CTrade class to provide access to the member variables of the class.

Afterward, we need some control logic to generate signals to open the positions. In our case, we use the RSI (Relative Strength Indicator), but you can use any as you deem fit.

```
int rsi_handle;                // Handle for the RSI indicator
double rsiData[];              // Array to store RSI data
int totalBars = 0;             // Variable to keep track of the total number of bars
```

The rsi\_handle stores the reference for the RSI (Relative Strength Index) indicator, which is initialized in the OnInit function, allowing the EA to retrieve RSI values. The rsiData array stores these RSI values, fetched using CopyBuffer, and is used to determine trading signals based on RSI thresholds. The _totalBars_ variable keeps track of the total number of bars on the chart, ensuring the trading logic executes only once per new bar, preventing multiple executions within a single bar. Together, these variables enable the EA to generate trading signals based on RSI values while maintaining proper execution timing.

Finally, after defining the indicator values, we define the indicator signal generation levels and zone levels.

```
double overBoughtLevel = 70.0; // Overbought level for RSI
double overSoldLevel = 30.0;   // Oversold level for RSI
double zoneHigh = 0;           // Variable to store the high zone price
double zoneLow = 0;            // Variable to store the low zone price
double zoneTargetHigh = 0;     // Variable to store the target high zone price
double zoneTargetLow = 0;      // Variable to store the target low zone price
```

Again, we define two double variables, overBoughtLevel and overSoldLevel, and initialize them to 70 and 30, respectively. These serve as our extreme levels for signal production. Moreover, we define four extra double data type variables, zoneHigh, zoneLow, zoneTradegetHigh, and zoneTargetLow, and initialize them to zero. They will hold our recovery setup levels later on in the code.

Up to this point, we have defined all the global variables, which are crucial to the system. We can now freely graduate to the OnInit event handler, which is called whenever the expert advisor is initialized. It is in this instance that we need to initialize the indicator handle that we will later copy data from for further analysis. To initialize the indicator, we use the built-in function to return its handle by providing the correct parameters.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- Initialize the RSI indicator
   rsi_handle = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);
   //--- Return initialization result
   return(INIT_SUCCEEDED);
  }
```

On the expert deinitialization event handler, we need to release the indicator data as well as free the stored data. We do this to typically free the indicator from the computer memory to save resources since it will not be used again, and if it is, the indicator handles and data will be created on the initialization of the expert advisor.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   //--- Remove RSI indicator from memory
   IndicatorRelease(rsi_handle);
   ArrayFree(rsiData); // Free the RSI data array
  }
```

We then move on to the OnTick event handler, which is a function that is called on every tick, which is a quote price change. This is again our heart function or section, as it contains all the crucial code snippets for a successful implementation of the trading strategy. Since we will open positions, we need to define our asking and bidding prices so that we can use their most current values for analysis purposes.

```
   double ask = SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol,SYMBOL_BID);
```

Defining the ask and bid prices by using symbol information as a double data type helps us get the latest price quotes for further analysis. The function that we use is an overloading function that contains two variants, but we use the first one that takes just two parameters. The first parameter is the symbol; we use \_Symbol to automatically retrieve the symbol on the current chart and SYMBOL\_ASK to get the enumeration of the double type.

After getting the latest price quotes, technically generated on each tick, we are all set. We just continue to define the zone range and the zone target for the expert advisor. We define them as double data type variables and multiply them with the \_Point variable, which contains the point size of the current symbol in the quote currency.

```
   double zoneRange = 200*_Point;
   double zoneTarget = 400*_Point;
```

Finally, we need utility variables that will ensure we open recovery positions correctly. In this case, we define four of them. LastDirection will hold the type of the last opened position to ensure we alternate between buy and sell orders. We intend the sell to be -1 and the buy to be +1, just an arbitrary value that can be changed or implemented as per one's choice. The recovery lot is then used and initialized to zero, whose function will be to hold and store our computed recovery points so we don't lose track of the next trade volumes, provided a zone recovery system is still in play. Still, we have two boolean variables, isBuyDone, and isSellDone, to typically store flags to aid us in not opening several positions at once. Note that all of our variables are static to ensure they are not updated unless we update them individually by ourselves. This is because local variables declared with the static keyword retain their values throughout the function's lifetime. Thus, with each next OnTick function call, our local variables will contain the values that they had during the previous call.

```
   static int lastDirection = 0; //-1 = sell, 1 = buy
   static double recovery_lot = 0.0;
   static bool isBuyDone = false, isSellDone = false;
```

After defining all the utility variables, we then proceed to open positions, from which we will later implement the zone recovery logic. We intend to achieve this by using the RSI indicator, which is entirely changeable as per users' preferences, so you can just go ahead and apply your entry technique. Okay, now, to save resources, we want to get the data from the indicator on each candlestick and not on every tick. This is achieved as follows:

```
   int bars = iBars(_Symbol,PERIOD_CURRENT);
   if (totalBars == bars) return;
   totalBars = bars;
```

Here, we define integer variable bars and initialize them to the number of the current bars on the chart, achieved via the use of the iBars in-built function, which takes two arguments: a symbol and a period. We then proceed to check if the number of the previously defined bars is equal to the current bars, and if so, it then means that we are still on the current bar, and thus we return, meaning we break the operation and return control to the calling program. Otherwise, if the two variables do not match, it means that we have graduated to a new candlestick, and we can continue. So we update the value of _totalBars_ to the current bars so that on the next tick, we do have an updated value of the _totalBars_ variable.

For the zone recovery to have effect and open just one position and manage it, we need to open one position per instance. Thus, if the number of positions is greater than one, we don't need to add any other positions, and we just return early. This is as below again.

If we do not return to this point, it means we don't have any position yet, and we can proceed to open one. Thus, we copy the data from the indicator handle and store it in the indicator data array for further analysis. This is achieved via the copy buffer function.

```
   if (PositionsTotal() > 0) return;
```

```
   if (!CopyBuffer(rsi_handle,0,1,2,rsiData)) return;
```

As visualized, the copy buffer function is an overloading function that returns an integer. For security reasons, we use the if statement to check whether it returns the requested data, and if not, we don't have enough data to return since further analysis cannot be done. However, let us see what the function does. It contains five arguments. First is the indicator handle from which to copy data, and second is the buffer number of the indicator; in this case, it is 0, but can entirely vary based on the indicator you are using. Third is the starting position, or the index of the bar to copy the data from. Here we use 1 to signify that we start at the bar before the current bar on the chart. Fourth is the count—the number of data counts to store. Two is enough for us here, as we don't do detailed analysis. Finally, we provide the target array of the retrieved data storage.

After retrieving data from the indicator handle, we then proceed to use the data for trading purposes, or rather, signal generation. First, we look for buy-in signals. We achieve this by using an if statement and opening a buy position.

```
   if (rsiData[1] < overSoldLevel && rsiData[0] > overSoldLevel){
      obj_trade.Buy(0.01);
```

If the data at index 1 on the stored array is less than the defined oversold level and the data at index 0 is greater than the oversold level, it then means we had a crossover between the RSI line and the oversold level, signifying we have a buy signal. We then make use of the trade object and the dot operator to get access to the buy method contained in the CTrade class. In this instance, we open a buy position of volume 0.01 and ignore the rest of the parameters like stop loss and take profit since we will not be using them, as they would make our system not work as intended because we implement the x=zone recovery strategy that does not need to close the positions on stop loss levels.

However, to set the zone recovery levels, we will need the ticket for the position so we can get access to its properties. To get the ticket, we use the result order of the previously opened position. After getting the ticket, we want to check that it is greater than 0, signifying that it was a success opening the position, and then select the position by ticket. If we can select it by the ticket, we can then get the position's properties, but we are only interested in the opening price. From the opening price, we set the positions as follows: The open price of the buy position is our zone high, and to get the zone low, we just subtract the zone range from the zone high. For the zone targets high and low, we just add the zone target to the zone high and subtract the zone target from the zone low, respectively. Finally, we just need to normalize the values to the digits of the symbol for accuracy, and we are all done.

```
      ulong pos_ticket = obj_trade.ResultOrder();
      if (pos_ticket > 0){
         if (PositionSelectByTicket(pos_ticket)){
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            zoneHigh = NormalizeDouble(openPrice,_Digits);
            zoneLow = NormalizeDouble(zoneHigh - zoneRange,_Digits);
            zoneTargetHigh = NormalizeDouble(zoneHigh + zoneTarget,_Digits);
            zoneTargetLow = NormalizeDouble(zoneLow - zoneTarget,_Digits);
```

Up to this point, we can set the levels, but they are not visible on the chart. To achieve this, we create a function that we will use to visualize the four defined levels on the chart.

```
            drawZoneLevel(ZONE_H,zoneHigh,clrGreen,2);
            drawZoneLevel(ZONE_L,zoneLow,clrRed,2);
            drawZoneLevel(ZONE_T_H,zoneTargetHigh,clrBlue,3);
            drawZoneLevel(ZONE_T_L,zoneTargetLow,clrBlue,3);
```

We create just a simple void function that takes four input parameters or arguments, viz., levelName, price, CLR, and width. We use the ObjectCreate inbuilt function to create a horizontal line that spans throughout the chart length, attaching it to the provided time and price. Finally, we use the ObjectSetInteger to set the color of the object for uniqueness and the width for easier adjustable visibility.

```
void drawZoneLevel(string levelName, double price, color clr, int width) {
   ObjectCreate(0, levelName, OBJ_HLINE, 0, TimeCurrent(), price); // Create a horizontal line object
   ObjectSetInteger(0, levelName, OBJPROP_COLOR, clr); // Set the line color
   ObjectSetInteger(0, levelName, OBJPROP_WIDTH, width); // Set the line width
}
```

Finally, we set the last direction to the value of 1 to show that we did open a buy position, set the next recovery volume as the initial volume multiplied by a multiplier constant, in this case, 2, meaning we double the volume, and lastly, set the isBuyDone flag to true and isSellDone to false.

```
            lastDirection = 1;
            recovery_lot = 0.01*2;
            isBuyDone = true; isSellDone = false;
```

The full code to open the position and set up the zone recovery levels is as below.

```
   //--- Check for oversold condition and open a buy position
   if (rsiData[1] < overSoldLevel && rsiData[0] > overSoldLevel) {
      obj_trade.Buy(0.01); // Open a buy trade with 0.01 lots
      ulong pos_ticket = obj_trade.ResultOrder();
      if (pos_ticket > 0) {
         if (PositionSelectByTicket(pos_ticket)) {
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            zoneHigh = NormalizeDouble(openPrice, _Digits); // Set the high zone price
            zoneLow = NormalizeDouble(zoneHigh - zoneRange, _Digits); // Set the low zone price
            zoneTargetHigh = NormalizeDouble(zoneHigh + zoneTarget, _Digits); // Set the target high zone price
            zoneTargetLow = NormalizeDouble(zoneLow - zoneTarget, _Digits); // Set the target low zone price
            drawZoneLevel(ZONE_H, zoneHigh, clrGreen, 2); // Draw the high zone line
            drawZoneLevel(ZONE_L, zoneLow, clrRed, 2); // Draw the low zone line
            drawZoneLevel(ZONE_T_H, zoneTargetHigh, clrBlue, 3); // Draw the target high zone line
            drawZoneLevel(ZONE_T_L, zoneTargetLow, clrBlue, 3); // Draw the target low zone line

            lastDirection = 1;       // Set the last direction to buy
            recovery_lot = 0.01 * 2; // Set the initial recovery lot size
            isBuyDone = true;        // Mark buy trade as done
            isSellDone = false;      // Reset sell trade flag
         }
      }
   }
```

To open the sell position and set up the zone recovery levels, the control logic remains, but with inverse conditions as below.

```
   else if (rsiData[1] > overBoughtLevel && rsiData[0] < overBoughtLevel) {
      obj_trade.Sell(0.01); // Open a sell trade with 0.01 lots
      ulong pos_ticket = obj_trade.ResultOrder();
      if (pos_ticket > 0) {
         if (PositionSelectByTicket(pos_ticket)) {
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            zoneLow = NormalizeDouble(openPrice, _Digits); // Set the low zone price
            zoneHigh = NormalizeDouble(zoneLow + zoneRange, _Digits); // Set the high zone price
            zoneTargetHigh = NormalizeDouble(zoneHigh + zoneTarget, _Digits); // Set the target high zone price
            zoneTargetLow = NormalizeDouble(zoneLow - zoneTarget, _Digits); // Set the target low zone price
            drawZoneLevel(ZONE_H, zoneHigh, clrGreen, 2); // Draw the high zone line
            drawZoneLevel(ZONE_L, zoneLow, clrRed, 2); // Draw the low zone line
            drawZoneLevel(ZONE_T_H, zoneTargetHigh, clrBlue, 3); // Draw the target high zone line
            drawZoneLevel(ZONE_T_L, zoneTargetLow, clrBlue, 3); // Draw the target low zone line

            lastDirection = -1;      // Set the last direction to sell
            recovery_lot = 0.01 * 2; // Set the initial recovery lot size
            isBuyDone = false;       // Reset buy trade flag
            isSellDone = true;       // Mark sell trade as done
         }
      }
   }
```

Here, we check if the sell signal conditions are met, and if so, we open a sell position instantly. Then, we get its ticket and use the ticket to retrieve the position's opening price, which we use to set the zone recovery levels. Since it is a sell position, its price becomes the zone low, and to get the zone high, we just add the zone range to the zone high. Similarly, we add the zone target to the zone high to get the zone target high and subtract the zone target from the zone low to get the zone target low. For visualization, we again draw the four levels using the functions. Lastly, we just set up our utility variables.

Up to this point, we managed to open the positions based on the presented signal and set up the zone recovery system. Here is the full code that enables that.

```
void OnTick()
  {
   int bars = iBars(_Symbol,PERIOD_CURRENT);
   if (totalBars == bars) return;
   totalBars = bars;

   if (PositionsTotal() > 0) return;

   if (!CopyBuffer(rsi_handle,0,1,2,rsiData)) return;

   //--- Check for oversold condition and open a buy position
   if (rsiData[1] < overSoldLevel && rsiData[0] > overSoldLevel) {
      obj_trade.Buy(0.01); // Open a buy trade with 0.01 lots
      ulong pos_ticket = obj_trade.ResultOrder();
      if (pos_ticket > 0) {
         if (PositionSelectByTicket(pos_ticket)) {
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            zoneHigh = NormalizeDouble(openPrice, _Digits); // Set the high zone price
            zoneLow = NormalizeDouble(zoneHigh - zoneRange, _Digits); // Set the low zone price
            zoneTargetHigh = NormalizeDouble(zoneHigh + zoneTarget, _Digits); // Set the target high zone price
            zoneTargetLow = NormalizeDouble(zoneLow - zoneTarget, _Digits); // Set the target low zone price
            drawZoneLevel(ZONE_H, zoneHigh, clrGreen, 2); // Draw the high zone line
            drawZoneLevel(ZONE_L, zoneLow, clrRed, 2); // Draw the low zone line
            drawZoneLevel(ZONE_T_H, zoneTargetHigh, clrBlue, 3); // Draw the target high zone line
            drawZoneLevel(ZONE_T_L, zoneTargetLow, clrBlue, 3); // Draw the target low zone line

            lastDirection = 1;       // Set the last direction to buy
            recovery_lot = 0.01 * 2; // Set the initial recovery lot size
            isBuyDone = true;        // Mark buy trade as done
            isSellDone = false;      // Reset sell trade flag
         }
      }
   }
   //--- Check for overbought condition and open a sell position
   else if (rsiData[1] > overBoughtLevel && rsiData[0] < overBoughtLevel) {
      obj_trade.Sell(0.01); // Open a sell trade with 0.01 lots
      ulong pos_ticket = obj_trade.ResultOrder();
      if (pos_ticket > 0) {
         if (PositionSelectByTicket(pos_ticket)) {
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            zoneLow = NormalizeDouble(openPrice, _Digits); // Set the low zone price
            zoneHigh = NormalizeDouble(zoneLow + zoneRange, _Digits); // Set the high zone price
            zoneTargetHigh = NormalizeDouble(zoneHigh + zoneTarget, _Digits); // Set the target high zone price
            zoneTargetLow = NormalizeDouble(zoneLow - zoneTarget, _Digits); // Set the target low zone price
            drawZoneLevel(ZONE_H, zoneHigh, clrGreen, 2); // Draw the high zone line
            drawZoneLevel(ZONE_L, zoneLow, clrRed, 2); // Draw the low zone line
            drawZoneLevel(ZONE_T_H, zoneTargetHigh, clrBlue, 3); // Draw the target high zone line
            drawZoneLevel(ZONE_T_L, zoneTargetLow, clrBlue, 3); // Draw the target low zone line

            lastDirection = -1;      // Set the last direction to sell
            recovery_lot = 0.01 * 2; // Set the initial recovery lot size
            isBuyDone = false;       // Reset buy trade flag
            isSellDone = true;       // Mark sell trade as done
         }
      }
   }
}
```

We can set up the zone recovery system, but we need to monitor it and ensure we escape out of it once we either hit the target levels or open respective positions once we hit the pre-defined levels. We will have to do this on every tick, and thus, the logic has to be implemented before the candlestick loop restriction. First, let us check the condition where the prices hit the target levels, and later on, check the condition where the prices hit the zone levels. So let us dance then.

```
         if (zoneTargetHigh > 0 && zoneTargetLow > 0){
            if (bid > zoneTargetHigh || bid < zoneTargetLow){
               obj_trade.PositionClose(_Symbol);
               deleteZoneLevels();
               ...
            }
         }
```

Here, we check that the zone recovery system is set by having the logic that the target levels are above zero, which means we already have a system in play. So if that is the case, we check if the bid price is above the target high or below the target low, an indication that we can comfortably destroy the zone recovery as its objective has been attained. Thus, we delete the zone levels via the function deleteZoneLevels. The function that we use is a void type since we do not need to return anything, and the inbuilt ObjectDelete function is implemented to delete the levels by taking up two arguments, the chart index and the object name.

```
void deleteZoneLevels(){
   ObjectDelete(0,ZONE_H);
   ObjectDelete(0,ZONE_L);
   ObjectDelete(0,ZONE_T_H);
   ObjectDelete(0,ZONE_T_L);
}
```

To close the positions, since there could be several of them at this point, we use a loop that considers all of them and then deletes them individually. That is achieved via the below code.

```
               for (int i = PositionsTotal()-1; i >= 0; i--){
                  ulong ticket = PositionGetTicket(i);
                  if (ticket > 0){
                     if (PositionSelectByTicket(ticket)){
                        obj_trade.PositionClose(ticket);
                     }
                  }
               }
```

After closing all the positions and deleting the levels, we reset the system to default, which does not have any zone recovery system.

```
               //closed all, reset all
               zoneHigh=0;zoneLow=0;zoneTargetHigh=0;zoneTargetLow=0;
               lastDirection=0;
               recovery_lot = 0;
```

That is achieved by setting the zone levels and targets to zero, besides the last direction and recovery lot. These are static variables, which is why we have to manually reset them. For the dynamic variables, there is no need since they are often updated automatically.

The full code responsible for destroying the recovery system after its objectives are met is as below:

```
   //--- Close all positions if the bid price is outside target zones
   if (zoneTargetHigh > 0 && zoneTargetLow > 0) {
      if (bid > zoneTargetHigh || bid < zoneTargetLow) {
         obj_trade.PositionClose(_Symbol); // Close the current position
         deleteZoneLevels();               // Delete all drawn zone levels
         for (int i = PositionsTotal() - 1; i >= 0; i--) {
            ulong ticket = PositionGetTicket(i);
            if (ticket > 0) {
               if (PositionSelectByTicket(ticket)) {
                  obj_trade.PositionClose(ticket); // Close positions by ticket
               }
            }
         }
         //--- Reset all zone and direction variables
         zoneHigh = 0;
         zoneLow = 0;
         zoneTargetHigh = 0;
         zoneTargetLow = 0;
         lastDirection = 0;
         recovery_lot = 0;
      }
   }
```

Moving on now to open recovery positions, we will need to check if the system is still in play, as shown by the instance when the zone levels are above zero. If so, we set it by declaring a variable lots\_rec, which we simply use to store our recovery lots. We then normalize it to 3 decimal places for precision, since the trading account that we are using is a microlot account. This value can change according to the account type that you are using. For example, if you are using a standard account, its minimum lot is 1, and thus your value will be 0, to get rid of the decimal places. Most have 2 decimal places, but you could have a 0.001 account type, and thus your value is 3 to round off the lots to the nearest 3 decimal places.

```
   if (zoneHigh > 0 && zoneLow > 0){
      double lots_Rec = 0;
      lots_Rec = NormalizeDouble(recovery_lot,2);
      ...
   }
```

Then, we check if the bid price is above the zone high, and if either the previous isBuyDone flag is false or the last direction value is less than zero, we open a buy recovery position. After opening the position, we set the lastDirection to 1, meaning that the previously opened position is a buy, compute the recovery lots and store them in the recovery\_lot variable for use on the next recovery position call, and then set the isBuyDone flag to true and isSellDone to false, indicating that a buy position has already been opened.

```
  if (bid > zoneHigh) {
         if (isBuyDone == false || lastDirection < 0) {
            obj_trade.Buy(lots_Rec); // Open a buy trade

            lastDirection = 1;       // Set the last direction to buy
            recovery_lot = recovery_lot * 2; // Double the recovery lot size
            isBuyDone = true;        // Mark buy trade as done
            isSellDone = false;      // Reset sell trade flag
         }
      }
```

Otherwise, if the bid price is below the zone low, we open the sell recovery position, respectively, as shown.

```
else if (bid < zoneLow) {
         if (isSellDone == false || lastDirection > 0) {
            obj_trade.Sell(lots_Rec); // Open a sell trade

            lastDirection = -1;      // Set the last direction to sell
            recovery_lot = recovery_lot * 2; // Double the recovery lot size
            isBuyDone = false;       // Reset buy trade flag
            isSellDone = true;       // Mark sell trade as done
         }
      }
   }
```

The full code now responsible for opening the recovery positions is as below:

```
   //--- Check if price is within defined zones and take action
   if (zoneHigh > 0 && zoneLow > 0) {
      double lots_Rec = NormalizeDouble(recovery_lot, 2); // Normalize the recovery lot size to 2 decimal places
      if (bid > zoneHigh) {
         if (isBuyDone == false || lastDirection < 0) {
            obj_trade.Buy(lots_Rec); // Open a buy trade

            lastDirection = 1;       // Set the last direction to buy
            recovery_lot = recovery_lot * 2; // Double the recovery lot size
            isBuyDone = true;        // Mark buy trade as done
            isSellDone = false;      // Reset sell trade flag
         }
      } else if (bid < zoneLow) {
         if (isSellDone == false || lastDirection > 0) {
            obj_trade.Sell(lots_Rec); // Open a sell trade

            lastDirection = -1;      // Set the last direction to sell
            recovery_lot = recovery_lot * 2; // Double the recovery lot size
            isBuyDone = false;       // Reset buy trade flag
            isSellDone = true;       // Mark sell trade as done
         }
      }
   }
```

This is the milestone that we have achieved up to this point.

![Current Milestone](https://c.mql5.com/2/82/MILESTONE.png)

The full code needed to automate the zone recovery system is as below:

```
//+------------------------------------------------------------------+
//|                                                MARTINGALE EA.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

//--- Define utility variables for later use

#define ZONE_H "ZH"            // Define a constant for the high zone line name
#define ZONE_L "ZL"            // Define a constant for the low zone line name
#define ZONE_T_H "ZTH"         // Define a constant for the target high zone line name
#define ZONE_T_L "ZTL"         // Define a constant for the target low zone line name

//--- Include trade instance class

#include <Trade/Trade.mqh>     // Include the trade class for trading functions
CTrade obj_trade;              // Create an instance of the CTrade class for trading operations

//--- Declare variables to hold indicator data

int rsi_handle;                // Handle for the RSI indicator
double rsiData[];              // Array to store RSI data
int totalBars = 0;             // Variable to keep track of the total number of bars

double overBoughtLevel = 70.0; // Overbought level for RSI
double overSoldLevel = 30.0;   // Oversold level for RSI
double zoneHigh = 0;           // Variable to store the high zone price
double zoneLow = 0;            // Variable to store the low zone price
double zoneTargetHigh = 0;     // Variable to store the target high zone price
double zoneTargetLow = 0;      // Variable to store the target low zone price

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- Initialize the RSI indicator
   rsi_handle = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);
   //--- Return initialization result
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   //--- Remove RSI indicator from memory
   IndicatorRelease(rsi_handle);
   ArrayFree(rsiData); // Free the RSI data array
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //--- Retrieve the current Ask and Bid prices
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double zoneRange = 200 * _Point;       // Define the range for the zones
   double zoneTarget = 400 * _Point;      // Define the target range for the zones

   //--- Variables to track trading status
   static int lastDirection = 0;          // -1 = sell, 1 = buy
   static double recovery_lot = 0.0;      // Lot size for recovery trades
   static bool isBuyDone = false, isSellDone = false; // Flags to track trade completion

   //--- Close all positions if the bid price is outside target zones
   if (zoneTargetHigh > 0 && zoneTargetLow > 0) {
      if (bid > zoneTargetHigh || bid < zoneTargetLow) {
         obj_trade.PositionClose(_Symbol); // Close the current position
         deleteZoneLevels();               // Delete all drawn zone levels
         for (int i = PositionsTotal() - 1; i >= 0; i--) {
            ulong ticket = PositionGetTicket(i);
            if (ticket > 0) {
               if (PositionSelectByTicket(ticket)) {
                  obj_trade.PositionClose(ticket); // Close positions by ticket
               }
            }
         }
         //--- Reset all zone and direction variables
         zoneHigh = 0;
         zoneLow = 0;
         zoneTargetHigh = 0;
         zoneTargetLow = 0;
         lastDirection = 0;
         recovery_lot = 0;
      }
   }

   //--- Check if price is within defined zones and take action
   if (zoneHigh > 0 && zoneLow > 0) {
      double lots_Rec = NormalizeDouble(recovery_lot, 2); // Normalize the recovery lot size to 2 decimal places
      if (bid > zoneHigh) {
         if (isBuyDone == false || lastDirection < 0) {
            obj_trade.Buy(lots_Rec); // Open a buy trade

            lastDirection = 1;       // Set the last direction to buy
            recovery_lot = recovery_lot * 2; // Double the recovery lot size
            isBuyDone = true;        // Mark buy trade as done
            isSellDone = false;      // Reset sell trade flag
         }
      } else if (bid < zoneLow) {
         if (isSellDone == false || lastDirection > 0) {
            obj_trade.Sell(lots_Rec); // Open a sell trade

            lastDirection = -1;      // Set the last direction to sell
            recovery_lot = recovery_lot * 2; // Double the recovery lot size
            isBuyDone = false;       // Reset buy trade flag
            isSellDone = true;       // Mark sell trade as done
         }
      }
   }

   //--- Update bars and check for new bars
   int bars = iBars(_Symbol, PERIOD_CURRENT);
   if (totalBars == bars) return; // Exit if no new bars
   totalBars = bars; // Update the total number of bars

   //--- Exit if there are open positions
   if (PositionsTotal() > 0) return;

   //--- Copy RSI data and check for oversold/overbought conditions
   if (!CopyBuffer(rsi_handle, 0, 1, 2, rsiData)) return;

   //--- Check for oversold condition and open a buy position
   if (rsiData[1] < overSoldLevel && rsiData[0] > overSoldLevel) {
      obj_trade.Buy(0.01); // Open a buy trade with 0.01 lots
      ulong pos_ticket = obj_trade.ResultOrder();
      if (pos_ticket > 0) {
         if (PositionSelectByTicket(pos_ticket)) {
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            zoneHigh = NormalizeDouble(openPrice, _Digits); // Set the high zone price
            zoneLow = NormalizeDouble(zoneHigh - zoneRange, _Digits); // Set the low zone price
            zoneTargetHigh = NormalizeDouble(zoneHigh + zoneTarget, _Digits); // Set the target high zone price
            zoneTargetLow = NormalizeDouble(zoneLow - zoneTarget, _Digits); // Set the target low zone price
            drawZoneLevel(ZONE_H, zoneHigh, clrGreen, 2); // Draw the high zone line
            drawZoneLevel(ZONE_L, zoneLow, clrRed, 2); // Draw the low zone line
            drawZoneLevel(ZONE_T_H, zoneTargetHigh, clrBlue, 3); // Draw the target high zone line
            drawZoneLevel(ZONE_T_L, zoneTargetLow, clrBlue, 3); // Draw the target low zone line

            lastDirection = 1;       // Set the last direction to buy
            recovery_lot = 0.01 * 2; // Set the initial recovery lot size
            isBuyDone = true;        // Mark buy trade as done
            isSellDone = false;      // Reset sell trade flag
         }
      }
   }
   //--- Check for overbought condition and open a sell position
   else if (rsiData[1] > overBoughtLevel && rsiData[0] < overBoughtLevel) {
      obj_trade.Sell(0.01); // Open a sell trade with 0.01 lots
      ulong pos_ticket = obj_trade.ResultOrder();
      if (pos_ticket > 0) {
         if (PositionSelectByTicket(pos_ticket)) {
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            zoneLow = NormalizeDouble(openPrice, _Digits); // Set the low zone price
            zoneHigh = NormalizeDouble(zoneLow + zoneRange, _Digits); // Set the high zone price
            zoneTargetHigh = NormalizeDouble(zoneHigh + zoneTarget, _Digits); // Set the target high zone price
            zoneTargetLow = NormalizeDouble(zoneLow - zoneTarget, _Digits); // Set the target low zone price
            drawZoneLevel(ZONE_H, zoneHigh, clrGreen, 2); // Draw the high zone line
            drawZoneLevel(ZONE_L, zoneLow, clrRed, 2); // Draw the low zone line
            drawZoneLevel(ZONE_T_H, zoneTargetHigh, clrBlue, 3); // Draw the target high zone line
            drawZoneLevel(ZONE_T_L, zoneTargetLow, clrBlue, 3); // Draw the target low zone line

            lastDirection = -1;      // Set the last direction to sell
            recovery_lot = 0.01 * 2; // Set the initial recovery lot size
            isBuyDone = false;       // Reset buy trade flag
            isSellDone = true;       // Mark sell trade as done
         }
      }
   }
}

//+------------------------------------------------------------------+
//|      FUNCTION TO DRAW HORIZONTAL ZONE LINES                      |
//+------------------------------------------------------------------+

void drawZoneLevel(string levelName, double price, color clr, int width) {
   ObjectCreate(0, levelName, OBJ_HLINE, 0, TimeCurrent(), price); // Create a horizontal line object
   ObjectSetInteger(0, levelName, OBJPROP_COLOR, clr); // Set the line color
   ObjectSetInteger(0, levelName, OBJPROP_WIDTH, width); // Set the line width
}

//+------------------------------------------------------------------+
//|       FUNCTION TO DELETE DRAWN ZONE LINES                        |
//+------------------------------------------------------------------+

void deleteZoneLevels() {
   ObjectDelete(0, ZONE_H); // Delete the high zone line
   ObjectDelete(0, ZONE_L); // Delete the low zone line
   ObjectDelete(0, ZONE_T_H); // Delete the target high zone line
   ObjectDelete(0, ZONE_T_L); // Delete the target low zone line
}
```

Up to this point now, we have successfully automated the zone recovery forex trading system as intended, and we will proceed to test and see its performance, as shown below, to ascertain whether it meets its designated objectives.

### Backtest results

After the testing on the strategy tester, below are its results.

Graph:

![Graph](https://c.mql5.com/2/82/GRAPH.png)

Results:

![Results](https://c.mql5.com/2/82/RESULTS.png)

### Conclusion

In this article, we have looked at the basic steps that need to be implemented towards the automation of the famous zone recovery martingale strategy in MQL5. We have provided the basic definition and description of the strategy and shown how it can be implemented in MQL5. Traders can now use the knowledge shown to develop more complex zone recovery systems that can, later on, be optimized to produce better results at the end.

Disclaimer: This code is only intended to help you get the basics of creating a zone recovery forex trading system, and the results demonstrated do not guarantee future performance. Thus, carefully implement the knowledge to create and optimize your systems to fit your trading styles.

The article contains all the steps in a periodic manner towards creating the system. We do hope you find it useful and a stepping stone towards creating a better and fully optimized zone recovery system. Attachments of the necessary files are made to provide the examples that were used to demonstrate these examples. You should be able to study the code and apply it to your specific strategy to achieve optimal results.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15067.zip "Download all attachments in the single ZIP archive")

[Zone\_Recovery.mq5](https://www.mql5.com/en/articles/download/15067/zone_recovery.mq5 "Download Zone_Recovery.mq5")(9.7 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Implementing a Bollinger Bands Trading Strategy with MQL5: A Step-by-Step Guide](https://www.mql5.com/en/articles/15394)
- [Cascade Order Trading Strategy Based on EMA Crossovers for MetaTrader 5](https://www.mql5.com/en/articles/15250)
- [Creating a Daily Drawdown Limiter EA in MQL5](https://www.mql5.com/en/articles/15199)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/469287)**
(1)


![Ivan Titov](https://c.mql5.com/avatar/2024/9/66d71f0c-3796.png)

**[Ivan Titov](https://www.mql5.com/en/users/goldrat)**
\|
22 Nov 2024 at 07:01

Zone Recovery is not a strategy, but rather a money-management method. This method makes sense to use in combination with [range exit](https://www.mql5.com/ru/search#!keyword=%D0%92%D1%8B%D1%85%D0%BE%D0%B4%20%D0%B8%D0%B7%20%D0%B4%D0%B8%D0%B0%D0%BF%D0%B0%D0%B7%D0%BE%D0%BD%D0%B0) strategies.


![Automated Parameter Optimization for Trading Strategies Using Python and MQL5](https://c.mql5.com/2/82/Automated_Parameter_Optimization_for_Trading_Strategies_Using_Python_and_MQL5__LOGO.png)[Automated Parameter Optimization for Trading Strategies Using Python and MQL5](https://www.mql5.com/en/articles/15116)

There are several types of algorithms for self-optimization of trading strategies and parameters. These algorithms are used to automatically improve trading strategies based on historical and current market data. In this article we will look at one of them with python and MQL5 examples.

![Mastering Market Dynamics: Creating a Support and Resistance Strategy Expert Advisor (EA)](https://c.mql5.com/2/82/Creating_a_Support_and_Resistance_Strategy_Expert_Advisor__LOGO_2.png)[Mastering Market Dynamics: Creating a Support and Resistance Strategy Expert Advisor (EA)](https://www.mql5.com/en/articles/15107)

A comprehensive guide to developing an automated trading algorithm based on the Support and Resistance strategy. Detailed information on all aspects of creating an expert advisor in MQL5 and testing it in MetaTrader 5 – from analyzing price range behaviors to risk management.

![Propensity score in causal inference](https://c.mql5.com/2/72/Propensity_score_in_causal_inference____LOGO.png)[Propensity score in causal inference](https://www.mql5.com/en/articles/14360)

The article examines the topic of matching in causal inference. Matching is used to compare similar observations in a data set. This is necessary to correctly determine causal effects and get rid of bias. The author explains how this helps in building trading systems based on machine learning, which become more stable on new data they were not trained on. The propensity score plays a central role and is widely used in causal inference.

![Creating Time Series Predictions using LSTM Neural Networks: Normalizing Price and Tokenizing Time](https://c.mql5.com/2/82/Creating_Time_Series_Predictions_using_LSTM_Neural_Networks___LOGO.png)[Creating Time Series Predictions using LSTM Neural Networks: Normalizing Price and Tokenizing Time](https://www.mql5.com/en/articles/15063)

This article outlines a simple strategy for normalizing the market data using the daily range and training a neural network to enhance market predictions. The developed models may be used in conjunction with an existing technical analysis frameworks or on a standalone basis to assist in predicting the overall market direction. The framework outlined in this article may be further refined by any technical analyst to develop models suitable for both manual and automated trading strategies.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=zgthytpfzeqvesljeoubhxovzszurvbn&ssn=1769094069885964372&ssn_dr=0&ssn_sr=0&fv_date=1769094069&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15067&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20Zone%20Recovery%20Martingale%20strategy%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909406942338225&fz_uniq=5049565498041019829&sv=2552)

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