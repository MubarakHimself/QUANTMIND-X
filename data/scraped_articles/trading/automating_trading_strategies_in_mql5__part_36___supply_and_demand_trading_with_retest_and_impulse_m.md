---
title: Automating Trading Strategies in MQL5 (Part 36): Supply and Demand Trading with Retest and Impulse Model
url: https://www.mql5.com/en/articles/19674
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T17:53:02.568571
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=vyrkshbuuvoftchndznfnmmtahslhygf&ssn=1769179980986815204&ssn_dr=0&ssn_sr=0&fv_date=1769179980&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19674&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2036)%3A%20Supply%20and%20Demand%20Trading%20with%20Retest%20and%20Impulse%20Model%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917998070589501&fz_uniq=5068756730474593611&sv=2552)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 35)](https://www.mql5.com/en/articles/19638), we developed a [Breaker Block](https://www.mql5.com/go?link=https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/breaker-blocks "https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/breaker-blocks") Trading System in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) that identified consolidation ranges, validated breaker blocks with swing points, and traded retests with customizable risk parameters and visual feedback. In Part 36, we develop a [Supply and Demand Trading System](https://www.mql5.com/go?link=https://www.quantifiedstrategies.com/supply-and-demand-trading-strategy/ "https://www.quantifiedstrategies.com/supply-and-demand-trading-strategy/") utilizing a retest and impulse model. This model detects supply and demand zones through consolidation, validates them with impulsive moves, and executes trades on retests with trend confirmation and dynamic chart visualizations. We will cover the following topics:

1. [Understanding the Supply and Demand Strategy Framework](https://www.mql5.com/en/articles/19674#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/19674#para2)
3. [Backtesting](https://www.mql5.com/en/articles/19674#para3)
4. [Conclusion](https://www.mql5.com/en/articles/19674#para4)

By the end, you’ll have a functional MQL5 strategy for trading supply and demand zone retests, ready for customization—let’s dive in!

### Understanding the Supply and Demand Strategy Framework

[The](https://www.mql5.com/en/articles/19674#para4) [supply and demand strategy](https://www.mql5.com/go?link=https://www.quantifiedstrategies.com/supply-and-demand-trading-strategy/ "https://www.quantifiedstrategies.com/supply-and-demand-trading-strategy/") identifies key price zones where significant buying (demand) or selling (supply) has occurred, typically after periods of consolidation. After an impulsive price move confirms a zone's validity, traders aim to trade its retest. They may enter buy trades when price revisits a demand zone in a downtrend, or initiate sell trades at a supply zone in an uptrend, expecting a bounce. By defining risk and reward levels, traders capitalize on high-probability setups. Have a look below at the different setups we could have.

Supply Zone Setup:

![SUPPLY ZONE SETUP](https://c.mql5.com/2/171/Screenshot_2025-09-23_003350.png)

Demand Zone Setup:

![DEMAND ZONE SETUP](https://c.mql5.com/2/171/Screenshot_2025-09-23_003420.png)

Our plan is to detect consolidation ranges over a set number of bars, validate zones with impulsive moves using a multiplier-based threshold, and confirm trade entries with optional trend checks. We will implement logic to track zone status, execute trades on retests with customizable stop-loss and take-profit settings, and visualize zones with dynamic labels and colors, creating a system for precise supply and demand trading. In brief, here is a visual representation of our objectives.

![SUPPLY AND DEMAND FRAMEWORK](https://c.mql5.com/2/171/Screenshot_2025-09-23_004336.png)

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Experts folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some [input parameters](https://www.mql5.com/en/docs/basis/variables/inputvariables) and [global variables](https://www.mql5.com/en/docs/basis/variables/global) that we will use throughout the program.

```
//+------------------------------------------------------------------+
//|                                         Supply and Demand EA.mq5 |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//|                                   https://t.me/Forex_Algo_Trader |
//+------------------------------------------------------------------+
#property copyright "Forex Algo-Trader, Allan"
#property link "https://t.me/Forex_Algo_Trader"
#property version "1.00"
#property strict

#include <Trade/Trade.mqh>                         //--- Include Trade library for position management
CTrade obj_Trade;                                  //--- Instantiate trade object for order operations
```

We begin the implementation by including the trade library with "#include <Trade/Trade.mqh>", which provides built-in functions for managing trade operations. We then initialize the trade object "obj\_Trade" using the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class, allowing the Expert Advisor to execute buy and sell orders programmatically. This setup will ensure that trade execution is handled efficiently without requiring manual intervention. Then we can declare some [enumerations](https://www.mql5.com/en/docs/basis/types/integer/enumeration) that will enable classification of some user inputs.

```
//+------------------------------------------------------------------+
//| Enum for trading tested zones                                    |
//+------------------------------------------------------------------+
enum TradeTestedZonesMode {                        // Define modes for trading tested zones
   NoRetrade,                                      // Trade zones only once
   LimitedRetrade,                                 // Trade zones up to a maximum number of times
   UnlimitedRetrade                                // Trade zones as long as they are valid
};

//+------------------------------------------------------------------+
//| Enum for broken zones validation                                 |
//+------------------------------------------------------------------+
enum BrokenZonesMode {                             // Define modes for broken zones validation
   AllowBroken,                                    // Zones can be marked as broken
   NoBroken                                        // Zones remain testable regardless of price break
};

//+------------------------------------------------------------------+
//| Enum for zone size restriction                                   |
//+------------------------------------------------------------------+
enum ZoneSizeMode {                                // Define modes for zone size restrictions
   NoRestriction,                                  // No restriction on zone size
   EnforceLimits                                   // Enforce minimum and maximum zone points
};

//+------------------------------------------------------------------+
//| Enum for trend confirmation                                      |
//+------------------------------------------------------------------+
enum TrendConfirmationMode {                       // Define modes for trend confirmation
   NoConfirmation,                                 // No trend confirmation required
   ConfirmTrend                                    // Confirm trend before trading on tap
};
```

We forward declare some key [enumerations](https://www.mql5.com/en/docs/basis/types/integer/enumeration) to configure trading behavior and zone validation. First, we create the "TradeTestedZonesMode" enum with options: "NoRetrade" (trade zones once), "LimitedRetrade" (trade up to a set limit), and "UnlimitedRetrade" (trade while valid), which control how often zones can be traded. Then, we define the "BrokenZonesMode" enum with "AllowBroken" (mark zones as broken if price breaches them) and "NoBroken" (keep zones testable), determining zone validity after breakouts. Next, we implement the "ZoneSizeMode" [enum](https://www.mql5.com/en/docs/basis/types/integer/enumeration) with "NoRestriction" (no size limits) and "EnforceLimits" (restrict zone size within bounds), ensuring zones meet size criteria.

Finally, we add the "TrendConfirmationMode" enum with "NoConfirmation" (no trend check) and "ConfirmTrend" (require trend validation), enabling optional trend-based trade filtering. This will make the system have a flexible configuration for zone trading and validation rules. We can use these enumerations to create our user inputs.

```
//+------------------------------------------------------------------+
//| Input Parameters                                                 |
//+------------------------------------------------------------------+
input double tradeLotSize = 0.01;                   // Trade size in lots
input bool   enableTrading = true;                  // Enable automated trading
input bool   enableTrailingStop = true;             // Enable trailing stop
input double trailingStopPoints = 30;               // Trailing stop points
input double minProfitToTrail = 50;                 // Minimum trailing points
input int    uniqueMagicNumber = 12345;             // Magic Number
input int    consolidationBars = 5;                 // Consolidation range bars
input double maxConsolidationSpread = 30;           // Maximum allowed spread in points for consolidation
input double stopLossDistance = 200;                // Stop loss in points
input double takeProfitDistance = 400;              // Take profit in points
input double minMoveAwayPoints = 50;                // Minimum points price must move away before zone is ready
input bool   deleteBrokenZonesFromChart = false;    // Delete broken zones from chart
input bool   deleteExpiredZonesFromChart = false;   // Delete expired zones from chart
input int    zoneExtensionBars = 150;               // Number of bars to extend zones to the right
input bool   enableImpulseValidation = true;        // Enable impulse move validation
input int    impulseCheckBars = 3;                  // Number of bars to check for impulsive move
input double impulseMultiplier = 1.0;               // Multiplier for impulsive threshold
input TradeTestedZonesMode tradeTestedMode = NoRetrade; // Mode for trading tested zones
input int    maxTradesPerZone = 2;                  // Maximum trades per zone for LimitedRetrade
input BrokenZonesMode brokenZoneMode = AllowBroken; // Mode for broken zones validation
input color  demandZoneColor = clrBlue;             // Color for untested demand zones
input color  supplyZoneColor = clrRed;              // Color for untested supply zones
input color  testedDemandZoneColor = clrBlueViolet; // Color for tested demand zones
input color  testedSupplyZoneColor = clrOrange;     // Color for tested supply zones
input color  brokenZoneColor = clrDarkGray;         // Color for broken zones
input color  labelTextColor = clrBlack;             // Color for text labels
input ZoneSizeMode zoneSizeRestriction = NoRestriction; // Zone size restriction mode
input double minZonePoints = 50;                    // Minimum zone size in points
input double maxZonePoints = 300;                   // Maximum zone size in points
input TrendConfirmationMode trendConfirmation = NoConfirmation; // Trend confirmation mode
input int    trendLookbackBars = 10;                // Number of bars for trend confirmation
input double minTrendPoints = 1;                    // Minimum points for trend confirmation
```

Here, we establish the configuration [input parameters](https://www.mql5.com/en/docs/basis/variables/inputvariables) for our system to define its trading and visualization behavior. We have added self-explanatory comments to make everything easy and straightforward. Finally, since we will be managing several supply and demand zones, we need to declare a [structure](https://www.mql5.com/en/docs/basis/types/classes) where we will store the zones' information for ease of management.

```
//+------------------------------------------------------------------+
//| Structure for zone information                                   |
//+------------------------------------------------------------------+
struct SDZone {                                    //--- Define structure for supply/demand zones
   double   high;                                  //--- Store zone high price
   double   low;                                   //--- Store zone low price
   datetime startTime;                             //--- Store zone start time
   datetime endTime;                               //--- Store zone end time
   datetime breakoutTime;                          //--- Store breakout time
   bool     isDemand;                              //--- Indicate demand (true) or supply (false)
   bool     tested;                                //--- Track if zone was tested
   bool     broken;                                //--- Track if zone was broken
   bool     readyForTest;                          //--- Track if zone is ready for testing
   int      tradeCount;                            //--- Track number of trades on zone
   string   name;                                  //--- Store zone object name
};

//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
SDZone zones[];                                    //--- Store active supply/demand zones
SDZone potentialZones[];                           //--- Store potential zones awaiting validation
int    maxZones = 50;                              //--- Set maximum number of zones to track

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   obj_Trade.SetExpertMagicNumber(uniqueMagicNumber); //--- Set magic number for trade identification
   return(INIT_SUCCEEDED);                        //--- Return initialization success
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   ObjectsDeleteAll(0, "SDZone_");                //--- Remove all zone objects from chart
   ChartRedraw(0);                                //--- Redraw chart to clear objects
}
```

First, we create the "SDZone" [structure](https://www.mql5.com/en/docs/basis/types/classes) to store zone details, including high and low prices, start, end, and breakout times, flags for demand/supply type ("isDemand"), tested status ("tested"), broken status ("broken"), readiness for testing ("readyForTest"), trade count ("tradeCount"), and object name ("name"). Then, we initialize [global variables](https://www.mql5.com/en/docs/basis/variables/global): "zones" [array](https://www.mql5.com/en/book/basis/arrays/arrays_usage) to hold active supply and demand zones, "potentialZones" array for zones awaiting validation, and "maxZones" set to 50 to limit tracked zones. You can increase or decrease this value based on your timeframe and settings; we just chose an arbitrary standard value.

In the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, we call "SetExpertMagicNumber" on "obj\_Trade" with "uniqueMagicNumber" to tag trades and return [INIT\_SUCCEEDED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode) for successful initialization. In the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) function, we use [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/ObjectDeleteAll) to remove all chart objects with the "SDZone\_" prefix, as we will be naming all our objects with this prefix, and call [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to refresh the chart, ensuring clean resource cleanup. We can now define some helper functions that will help us detect and manage the zones. We will start with the logic that will detect the zones, but first, let us have a helper function that will help in debugging the zones.

```
//+------------------------------------------------------------------+
//| Print zones for debugging                                        |
//+------------------------------------------------------------------+
void PrintZones(SDZone &arr[]) {
   Print("Current zones count: ", ArraySize(arr)); //--- Log total number of zones
   for (int i = 0; i < ArraySize(arr); i++) {     //--- Iterate through zones
      Print("Zone ", i, ": ", arr[i].name, " endTime: ", TimeToString(arr[i].endTime)); //--- Log zone details
   }
}
```

To monitor zone states, we develop the "PrintZones" [function](https://www.mql5.com/en/docs/basis/function), which takes an "SDZone" [array](https://www.mql5.com/en/book/basis/arrays/arrays_usage), logs the total number of zones using [Print](https://www.mql5.com/en/docs/common/print) with [ArraySize](https://www.mql5.com/en/docs/array/arraysize), and iterates through the array to log each zone’s index, name, and end time with "Print" and [TimeToString](https://www.mql5.com/en/docs/convert/timetostring) for clear tracking. We can now develop the core logic to detect the zones.

```
//+------------------------------------------------------------------+
//| Detect supply and demand zones                                   |
//+------------------------------------------------------------------+
void DetectZones() {
   int startIndex = consolidationBars + 1;                 //--- Set start index for consolidation check
   if (iBars(_Symbol, _Period) < startIndex + 1) return;   //--- Exit if insufficient bars
   bool isConsolidated = true;                             //--- Assume consolidation
   double highPrice = iHigh(_Symbol, _Period, startIndex); //--- Initialize high price
   double lowPrice = iLow(_Symbol, _Period, startIndex);   //--- Initialize low price
   for (int i = startIndex - 1; i >= 2; i--) {             //--- Iterate through consolidation bars
      highPrice = MathMax(highPrice, iHigh(_Symbol, _Period, i)); //--- Update highest high
      lowPrice = MathMin(lowPrice, iLow(_Symbol, _Period, i)); //--- Update lowest low
      if (highPrice - lowPrice > maxConsolidationSpread * _Point) { //--- Check spread limit
         isConsolidated = false;                           //--- Mark as not consolidated
         break;                                            //--- Exit loop
      }
   }
   if (isConsolidated) {                                   //--- Confirm consolidation
      double closePrice = iClose(_Symbol, _Period, 1);     //--- Get last closed bar price
      double breakoutLow = iLow(_Symbol, _Period, 1);      //--- Get breakout bar low
      double breakoutHigh = iHigh(_Symbol, _Period, 1);    //--- Get breakout bar high
      bool isDemandZone = closePrice > highPrice && breakoutLow >= lowPrice; //--- Check demand zone
      bool isSupplyZone = closePrice < lowPrice && breakoutHigh <= highPrice; //--- Check supply zone
      if (isDemandZone || isSupplyZone) {                   //--- Validate zone type
         double zoneSize = (highPrice - lowPrice) / _Point; //--- Calculate zone size
         if (zoneSizeRestriction == EnforceLimits && (zoneSize < minZonePoints || zoneSize > maxZonePoints)) return; //--- Check size restrictions
         datetime lastClosedBarTime = iTime(_Symbol, _Period, 1); //--- Get last bar time
         bool overlaps = false;                             //--- Initialize overlap flag
         for (int j = 0; j < ArraySize(zones); j++) {       //--- Check existing zones
            if (lastClosedBarTime < zones[j].endTime) {     //--- Check time overlap
               double maxLow = MathMax(lowPrice, zones[j].low); //--- Find max low
               double minHigh = MathMin(highPrice, zones[j].high); //--- Find min high
               if (maxLow <= minHigh) {                     //--- Check price overlap
                  overlaps = true;                          //--- Mark as overlapping
                  break;                                    //--- Exit loop
               }
            }
         }
         bool duplicate = false;                        //--- Initialize duplicate flag
         for (int j = 0; j < ArraySize(zones); j++) {   //--- Check for duplicates
            if (lastClosedBarTime < zones[j].endTime) { //--- Check time
               if (MathAbs(zones[j].high - highPrice) < _Point && MathAbs(zones[j].low - lowPrice) < _Point) { //--- Check price match
                  duplicate = true;                     //--- Mark as duplicate
                  break;                                //--- Exit loop
               }
            }
         }
         if (overlaps || duplicate) return;             //--- Skip overlapping or duplicate zones
         if (enableImpulseValidation) {                 //--- Check impulse validation
            bool pot_overlaps = false;                  //--- Initialize potential overlap flag
            for (int j = 0; j < ArraySize(potentialZones); j++) { //--- Check potential zones
               if (lastClosedBarTime < potentialZones[j].endTime) { //--- Check time overlap
                  double maxLow = MathMax(lowPrice, potentialZones[j].low); //--- Find max low
                  double minHigh = MathMin(highPrice, potentialZones[j].high); //--- Find min high
                  if (maxLow <= minHigh) {              //--- Check price overlap
                     pot_overlaps = true;               //--- Mark as overlapping
                     break;                             //--- Exit loop
                  }
               }
            }
            bool pot_duplicate = false;           //--- Initialize potential duplicate flag
            for (int j = 0; j < ArraySize(potentialZones); j++) { //--- Check potential duplicates
               if (lastClosedBarTime < potentialZones[j].endTime) { //--- Check time
                  if (MathAbs(potentialZones[j].high - highPrice) < _Point && MathAbs(potentialZones[j].low - lowPrice) < _Point) { //--- Check price match
                     pot_duplicate = true;      //--- Mark as duplicate
                     break;                     //--- Exit loop
                  }
               }
            }
            if (pot_overlaps || pot_duplicate) return; //--- Skip overlapping or duplicate potential zones
            int potCount = ArraySize(potentialZones); //--- Get potential zones count
            ArrayResize(potentialZones, potCount + 1); //--- Resize potential zones array
            potentialZones[potCount].high = highPrice; //--- Set zone high
            potentialZones[potCount].low = lowPrice; //--- Set zone low
            potentialZones[potCount].startTime = iTime(_Symbol, _Period, startIndex); //--- Set start time
            potentialZones[potCount].endTime = TimeCurrent() + PeriodSeconds(_Period) * zoneExtensionBars; //--- Set end time
            potentialZones[potCount].breakoutTime = iTime(_Symbol, _Period, 1); //--- Set breakout time
            potentialZones[potCount].isDemand = isDemandZone; //--- Set zone type
            potentialZones[potCount].tested = false; //--- Set untested
            potentialZones[potCount].broken = false; //--- Set not broken
            potentialZones[potCount].readyForTest = false; //--- Set not ready
            potentialZones[potCount].tradeCount = 0; //--- Initialize trade count
            potentialZones[potCount].name = "PotentialZone_" + TimeToString(potentialZones[potCount].startTime, TIME_DATE|TIME_SECONDS); //--- Set zone name
            Print("Potential zone created: ", (isDemandZone ? "Demand" : "Supply"), " at ", lowPrice, " - ", highPrice, " endTime: ", TimeToString(potentialZones[potCount].endTime)); //--- Log potential zone
         } else {                                 //--- No impulse validation
            int zoneCount = ArraySize(zones);     //--- Get zones count
            if (zoneCount >= maxZones) {          //--- Check max zones limit
               ArrayRemove(zones, 0, 1);          //--- Remove oldest zone
               zoneCount--;                       //--- Decrease count
            }
            ArrayResize(zones, zoneCount + 1);    //--- Resize zones array
            zones[zoneCount].high = highPrice;    //--- Set zone high
            zones[zoneCount].low = lowPrice;      //--- Set zone low
            zones[zoneCount].startTime = iTime(_Symbol, _Period, startIndex); //--- Set start time
            zones[zoneCount].endTime = TimeCurrent() + PeriodSeconds(_Period) * zoneExtensionBars; //--- Set end time
            zones[zoneCount].breakoutTime = iTime(_Symbol, _Period, 1); //--- Set breakout time
            zones[zoneCount].isDemand = isDemandZone; //--- Set zone type
            zones[zoneCount].tested = false;      //--- Set untested
            zones[zoneCount].broken = false;      //--- Set not broken
            zones[zoneCount].readyForTest = false; //--- Set not ready
            zones[zoneCount].tradeCount = 0;      //--- Initialize trade count
            zones[zoneCount].name = "SDZone_" + TimeToString(zones[zoneCount].startTime, TIME_DATE|TIME_SECONDS); //--- Set zone name
            Print("Zone created: ", (isDemandZone ? "Demand" : "Supply"), " zone: ", zones[zoneCount].name, " at ", lowPrice, " - ", highPrice, " endTime: ", TimeToString(zones[zoneCount].endTime)); //--- Log zone creation
            PrintZones(zones);                    //--- Print zones for debugging
         }
      }
   }
}
```

Here, we implement the zone detection logic for our system. In the "DetectZones" [function](https://www.mql5.com/en/docs/basis/function), we set "startIndex" to "consolidationBars + 1", exiting if insufficient bars exist via the [iBars](https://www.mql5.com/en/docs/series/ibars) function. We assume consolidation ("isConsolidated" true), initialize "highPrice" and "lowPrice" with [iHigh](https://www.mql5.com/en/docs/series/ihigh) and [iLow](https://www.mql5.com/en/docs/series/ilow) at "startIndex", and iterate backward through bars, updating with [MathMax](https://www.mql5.com/en/docs/math/mathmax) and [MathMin](https://www.mql5.com/en/docs/math/mathmin), marking "isConsolidated" false if the range exceeds "maxConsolidationSpread \* Point". If consolidated, we check the last bar’s close ("iClose"), low ("iLow"), and high ("iHigh") to identify demand ("closePrice > highPrice" and "breakoutLow >= lowPrice") or supply zones ("closePrice < lowPrice" and "breakoutHigh <= highPrice").

For valid zones, we verify size restrictions with "zoneSizeRestriction" and "minZonePoints"/"maxZonePoints", check for overlaps or duplicates in "zones" and "potentialZones" using "MathMax" and "MathMin", and, if "enableImpulseValidation" is true, add to "potentialZones" with [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize), setting fields like "high", "low", "startTime" ( [iTime](https://www.mql5.com/en/docs/series/itime)), "endTime" (" [TimeCurrent](https://www.mql5.com/en/docs/dateandtime/timecurrent) \+ zoneExtensionBars"), and "name" ("PotentialZone"), logging with "Print"; otherwise, we add directly to "zones", removing the oldest if "maxZones" is reached, and log with "Print" and "PrintZones" for debugging so we keep track of our zones, hence creating the core logic for detecting and storing supply and demand zones. We can run this in the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler to detect the zones.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   static datetime lastBarTime = 0;                      //--- Store last processed bar time
   datetime currentBarTime = iTime(_Symbol, _Period, 0); //--- Get current bar time
   bool isNewBar = (currentBarTime != lastBarTime);      //--- Check for new bar
   if (isNewBar) {                                       //--- Process new bar
      lastBarTime = currentBarTime;                      //--- Update last bar time
      DetectZones();                                     //--- Detect new zones
   }
}
```

In the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler, we track new bars by comparing the current bar’s time from [iTime](https://www.mql5.com/en/docs/series/itime) (for the symbol and period at shift 0) with a static "lastBarTime", setting "isNewBar" to true, and updating "lastBarTime" if different. If a new bar is detected, we call our function, "DetectZones", to identify new supply and demand zones based on consolidation patterns. We are now able to detect the zones as below.

![POTENTIAL ZONES DETECTED](https://c.mql5.com/2/171/Screenshot_2025-09-23_012914.png)

Now that we can detect potential supply and demand zones, we just have to validate them via the rally up or down movements, which we will refer to as impulse movements. For modularization, we can have the entire logic in a function.

```
//+------------------------------------------------------------------+
//| Validate potential zones based on impulsive move                 |
//+------------------------------------------------------------------+
void ValidatePotentialZones() {
   datetime lastClosedBarTime = iTime(_Symbol, _Period, 1);   //--- Get last closed bar time
   for (int p = ArraySize(potentialZones) - 1; p >= 0; p--) { //--- Iterate potential zones backward
      if (lastClosedBarTime >= potentialZones[p].endTime) {   //--- Check for expired zone
         Print("Potential zone expired and removed from array: ", potentialZones[p].name, " endTime: ", TimeToString(potentialZones[p].endTime)); //--- Log expiration
         ArrayRemove(potentialZones, p, 1);                   //--- Remove expired zone
         continue;                                            //--- Skip to next
      }
      if (TimeCurrent() > potentialZones[p].breakoutTime + impulseCheckBars * PeriodSeconds(_Period)) { //--- Check impulse window
         bool isImpulsive = false;                            //--- Initialize impulsive flag
         int breakoutShift = iBarShift(_Symbol, _Period, potentialZones[p].breakoutTime, false); //--- Get breakout bar shift
         double range = potentialZones[p].high - potentialZones[p].low; //--- Calculate zone range
         double threshold = range * impulseMultiplier;        //--- Calculate impulse threshold
         for (int shift = 1; shift <= impulseCheckBars; shift++) { //--- Check bars after breakout
            if (shift + breakoutShift >= iBars(_Symbol, _Period)) continue; //--- Skip out-of-bounds
            double cl = iClose(_Symbol, _Period, shift);      //--- Get close price
            if (potentialZones[p].isDemand) {                 //--- Check demand zone
               if (cl >= potentialZones[p].high + threshold) { //--- Check bullish impulse
                  isImpulsive = true;                         //--- Set impulsive flag
                  break;                                      //--- Exit loop
               }
            } else {                                          //--- Check supply zone
               if (cl <= potentialZones[p].low - threshold) { //--- Check bearish impulse
                  isImpulsive = true;                         //--- Set impulsive flag
                  break;                                      //--- Exit loop
               }
            }
         }
         if (isImpulsive) {                                  //--- Process impulsive zone
            double zoneSize = (potentialZones[p].high - potentialZones[p].low) / _Point; //--- Calculate zone size
            if (zoneSizeRestriction == EnforceLimits && (zoneSize < minZonePoints || zoneSize > maxZonePoints)) { //--- Check size limits
               ArrayRemove(potentialZones, p, 1);            //--- Remove invalid zone
               continue;                                     //--- Skip to next
            }
            bool overlaps = false;                           //--- Initialize overlap flag
            for (int j = 0; j < ArraySize(zones); j++) {     //--- Check existing zones
               if (lastClosedBarTime < zones[j].endTime) {   //--- Check time overlap
                  double maxLow = MathMax(potentialZones[p].low, zones[j].low); //--- Find max low
                  double minHigh = MathMin(potentialZones[p].high, zones[j].high); //--- Find min high
                  if (maxLow <= minHigh) {                   //--- Check price overlap
                     overlaps = true;                        //--- Mark as overlapping
                     break;                                  //--- Exit loop
                  }
               }
            }
            bool duplicate = false;                          //--- Initialize duplicate flag
            for (int j = 0; j < ArraySize(zones); j++) {     //--- Check for duplicates
               if (lastClosedBarTime < zones[j].endTime) {   //--- Check time
                  if (MathAbs(zones[j].high - potentialZones[p].high) < _Point && MathAbs(zones[j].low - potentialZones[p].low) < _Point) { //--- Check price match
                     duplicate = true;                       //--- Mark as duplicate
                     break;                                  //--- Exit loop
                  }
               }
            }
            if (overlaps || duplicate) {                     //--- Check overlap or duplicate
               Print("Validated zone overlaps or duplicates, discarded: ", potentialZones[p].low, " - ", potentialZones[p].high); //--- Log discard
               ArrayRemove(potentialZones, p, 1);            //--- Remove zone
               continue;                                     //--- Skip to next
            }
            int zoneCount = ArraySize(zones);                //--- Get zones count
            if (zoneCount >= maxZones) {                     //--- Check max zones limit
               ArrayRemove(zones, 0, 1);                     //--- Remove oldest zone
               zoneCount--;                                  //--- Decrease count
            }
            ArrayResize(zones, zoneCount + 1);               //--- Resize zones array
            zones[zoneCount] = potentialZones[p];            //--- Copy potential zone
            zones[zoneCount].name = "SDZone_" + TimeToString(zones[zoneCount].startTime, TIME_DATE|TIME_SECONDS); //--- Set zone name
            zones[zoneCount].endTime = TimeCurrent() + PeriodSeconds(_Period) * zoneExtensionBars; //--- Update end time
            Print("Zone validated: ", (zones[zoneCount].isDemand ? "Demand" : "Supply"), " zone: ", zones[zoneCount].name, " at ", zones[zoneCount].low, " - ", zones[zoneCount].high, " endTime: ", TimeToString(zones[zoneCount].endTime)); //--- Log validation
            ArrayRemove(potentialZones, p, 1);               //--- Remove validated zone
            PrintZones(zones);                               //--- Print zones for debugging
         } else {                                            //--- Zone not impulsive
            Print("Potential zone not impulsive, discarded: ", potentialZones[p].low, " - ", potentialZones[p].high); //--- Log discard
            ArrayRemove(potentialZones, p, 1);               //--- Remove non-impulsive zone
         }
      }
   }
}
```

Here, we create a [function](https://www.mql5.com/en/docs/basis/function) to implement the validation logic for potential supply and demand zones. In the "ValidatePotentialZones" function, we iterate backward through "potentialZones", checking if the last closed bar’s time ( [iTime](https://www.mql5.com/en/docs/series/itime) at shift 1) exceeds a zone’s "endTime", removing expired zones with [ArrayRemove](https://www.mql5.com/en/docs/array/arrayremove), and logging the action. For zones within the impulse window (" [TimeCurrent](https://www.mql5.com/en/docs/dateandtime/timecurrent) \> breakoutTime + impulseCheckBars \* [PeriodSeconds](https://www.mql5.com/en/docs/common/periodseconds)"), we calculate the zone range ("high - low") and impulse threshold ("range \* impulseMultiplier"), then check bars after the breakout ( [iBarShift](https://www.mql5.com/en/docs/series/ibarshift)) for a close price ( [iClose](https://www.mql5.com/en/docs/series/iclose)) exceeding the high plus threshold for demand zones or below the low minus threshold for supply zones, setting "isImpulsive" if met.

If impulsive, we verify zone size against "minZonePoints" and "maxZonePoints" if "zoneSizeRestriction" is "EnforceLimits", check for overlaps or duplicates in "zones" using [MathMax](https://www.mql5.com/en/docs/math/mathmax) and [MathMin](https://www.mql5.com/en/docs/math/mathmin), and, if valid, move the zone to "zones" with [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize), updating its name to "SDZone\_" and end time, logging with the "Print" and "PrintZones" functions, then remove it from "potentialZones"; non-impulsive zones are discarded with [ArrayRemove](https://www.mql5.com/en/docs/array/arrayremove) and logged, creating a system for validating zones based on impulsive moves and ensuring unique, valid zones. When you call the function in the tick event handler, you should get something that depicts the following.

![SUPPLY AND DEMAND ZONES VALIDATION](https://c.mql5.com/2/171/Screenshot_2025-09-23_014021.png)

Now that we can validate the zones, let us manage and visualize the zones on the chart for easier tracking.

```
//+------------------------------------------------------------------+
//| Update and draw zones                                            |
//+------------------------------------------------------------------+
void UpdateZones() {
   datetime lastClosedBarTime = iTime(_Symbol, _Period, 1); //--- Get last closed bar time
   for (int i = ArraySize(zones) - 1; i >= 0; i--) { //--- Iterate zones backward
      if (lastClosedBarTime >= zones[i].endTime) { //--- Check for expired zone
         Print("Zone expired and removed from array: ", zones[i].name, " endTime: ", TimeToString(zones[i].endTime)); //--- Log expiration
         if (deleteExpiredZonesFromChart) {    //--- Check if deleting expired
            ObjectDelete(0, zones[i].name);    //--- Delete zone rectangle
            ObjectDelete(0, zones[i].name + "Label"); //--- Delete zone label
         }
         ArrayRemove(zones, i, 1);             //--- Remove expired zone
         continue;                             //--- Skip to next
      }
      bool wasReady = zones[i].readyForTest;   //--- Store previous ready status
      if (!zones[i].readyForTest) {            //--- Check if not ready
         double currentClose = iClose(_Symbol, _Period, 1); //--- Get current close
         double zoneLevel = zones[i].isDemand ? zones[i].high : zones[i].low; //--- Get zone level
         double distance = zones[i].isDemand ? (currentClose - zoneLevel) : (zoneLevel - currentClose); //--- Calculate distance
         if (distance > minMoveAwayPoints * _Point) { //--- Check move away distance
            zones[i].readyForTest = true;      //--- Set ready for test
         }
      }
      if (!wasReady && zones[i].readyForTest) { //--- Check if newly ready
         Print("Zone ready for test: ", zones[i].name); //--- Log ready status
      }
      if (brokenZoneMode == AllowBroken && !zones[i].tested) { //--- Check if breakable
         double currentClose = iClose(_Symbol, _Period, 1); //--- Get current close
         bool wasBroken = zones[i].broken;     //--- Store previous broken status
         if (zones[i].isDemand) {              //--- Check demand zone
            if (currentClose < zones[i].low) { //--- Check if broken
               zones[i].broken = true;         //--- Mark as broken
            }
         } else {                              //--- Check supply zone
            if (currentClose > zones[i].high) { //--- Check if broken
               zones[i].broken = true;         //--- Mark as broken
            }
         }
         if (!wasBroken && zones[i].broken) {  //--- Check if newly broken
            Print("Zone broken in UpdateZones: ", zones[i].name); //--- Log broken zone
            ObjectSetInteger(0, zones[i].name, OBJPROP_COLOR, brokenZoneColor); //--- Update zone color
            string labelName = zones[i].name + "Label"; //--- Get label name
            string labelText = zones[i].isDemand ? "Demand Zone (Broken)" : "Supply Zone (Broken)"; //--- Set broken label
            ObjectSetString(0, labelName, OBJPROP_TEXT, labelText); //--- Update label text
            if (deleteBrokenZonesFromChart) {  //--- Check if deleting broken
               ObjectDelete(0, zones[i].name); //--- Delete zone rectangle
               ObjectDelete(0, labelName);     //--- Delete zone label
            }
         }
      }
      if (ObjectFind(0, zones[i].name) >= 0 || (!zones[i].broken || !deleteBrokenZonesFromChart)) { //--- Check if drawable
         color zoneColor;                        //--- Initialize zone color
         if (zones[i].tested) {                  //--- Check if tested
            zoneColor = zones[i].isDemand ? testedDemandZoneColor : testedSupplyZoneColor; //--- Set tested color
         } else if (zones[i].broken) {           //--- Check if broken
            zoneColor = brokenZoneColor;         //--- Set broken color
         } else {                                //--- Untested zone
            zoneColor = zones[i].isDemand ? demandZoneColor : supplyZoneColor; //--- Set untested color
         }
         ObjectCreate(0, zones[i].name, OBJ_RECTANGLE, 0, zones[i].startTime, zones[i].high, zones[i].endTime, zones[i].low); //--- Create zone rectangle
         ObjectSetInteger(0, zones[i].name, OBJPROP_COLOR, zoneColor); //--- Set zone color
         ObjectSetInteger(0, zones[i].name, OBJPROP_FILL, true); //--- Enable fill
         ObjectSetInteger(0, zones[i].name, OBJPROP_BACK, true); //--- Set to background
         string labelName = zones[i].name + "Label"; //--- Generate label name
         string labelText = zones[i].isDemand ? "Demand Zone" : "Supply Zone"; //--- Set base label
         if (zones[i].tested) labelText += " (Tested)"; //--- Append tested status
         else if (zones[i].broken) labelText += " (Broken)"; //--- Append broken status
         datetime labelTime = zones[i].startTime + (zones[i].endTime - zones[i].startTime) / 2; //--- Calculate label time
         double labelPrice = (zones[i].high + zones[i].low) / 2; //--- Calculate label price
         ObjectCreate(0, labelName, OBJ_TEXT, 0, labelTime, labelPrice); //--- Create label
         ObjectSetString(0, labelName, OBJPROP_TEXT, labelText); //--- Set label text
         ObjectSetInteger(0, labelName, OBJPROP_COLOR, labelTextColor); //--- Set label color
         ObjectSetInteger(0, labelName, OBJPROP_ANCHOR, ANCHOR_CENTER); //--- Set label anchor
      }
   }
   ChartRedraw(0);                                //--- Redraw chart
}
```

We proceed to implement the zone management and visualization logic for the system. In the "UpdateZones" function, we iterate backward through "zones", checking if the last closed bar’s time ( [iTime](https://www.mql5.com/en/docs/series/itime) at shift 1) exceeds a zone’s "endTime", removing expired zones with [ArrayRemove](https://www.mql5.com/en/docs/array/arrayremove), deleting their chart objects ( [OBJ\_RECTANGLE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_rectangle) and "Label") if "deleteExpiredZonesFromChart" is true, and logging, ensuring that if a zone is expired, it is no longer our concern. For non-ready zones, we calculate the distance from the current close ( [iClose](https://www.mql5.com/en/docs/series/iclose)) to the zone’s high (demand) or low (supply), marking "readyForTest" true if it exceeds "minMoveAwayPoints \* [\_Point](https://www.mql5.com/en/docs/predefined/_point)", logging if newly ready.

If "brokenZoneMode" is "AllowBroken" and the zone is untested, we mark it broken if the close falls below the low (demand) or above the high (supply), updating the color to "brokenZoneColor" with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) and label to "Demand/Supply Zone (Broken)" with [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring), deleting objects if "deleteBrokenZonesFromChart" is true, and logging the instance. For drawable zones (existing or not broken with "deleteBrokenZonesFromChart" false), we set colors ("demandZoneColor", "supplyZoneColor", "testedDemandZoneColor", "testedSupplyZoneColor", or "brokenZoneColor") based on status, draw rectangles with [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) ("OBJ\_RECTANGLE") using "startTime", "high", "endTime", and "low", and add centered labels with "ObjectCreate" ( [OBJ\_TEXT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_text)) using "labelTextColor", then redraw the chart with the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function, thus updating zone states and rendering them dynamically. We can now call this function in the tick event handler, and when we do, we get the following outcome.

![MANAGED AND VISUALIZED ZONES](https://c.mql5.com/2/171/Screenshot_2025-09-23_015308.png)

Now that we can manage the zones and visualize them on the chart, we just need to track them and trade based on fulfilled trading conditions. We will create a function that will loop through the valid zones and check the trading conditions.

```
//+------------------------------------------------------------------+
//| Trade on zones                                                   |
//+------------------------------------------------------------------+
void TradeOnZones(bool isNewBar) {
   static datetime lastTradeCheck = 0;                   //--- Store last trade check time
   datetime currentBarTime = iTime(_Symbol, _Period, 0); //--- Get current bar time
   if (!isNewBar || lastTradeCheck == currentBarTime) return; //--- Exit if not new bar or checked
   lastTradeCheck = currentBarTime;                      //--- Update last trade check
   double currentBid = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID), _Digits); //--- Get current bid
   double currentAsk = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK), _Digits); //--- Get current ask
   for (int i = 0; i < ArraySize(zones); i++) {          //--- Iterate through zones
      if (zones[i].broken) continue;                     //--- Skip broken zones
      if (tradeTestedMode == NoRetrade && zones[i].tested) continue; //--- Skip tested zones
      if (tradeTestedMode == LimitedRetrade && zones[i].tested && zones[i].tradeCount >= maxTradesPerZone) continue; //--- Skip max trades
      if (!zones[i].readyForTest) continue;              //--- Skip not ready zones
      double prevHigh = iHigh(_Symbol, _Period, 1);      //--- Get previous high
      double prevLow = iLow(_Symbol, _Period, 1);        //--- Get previous low
      double prevClose = iClose(_Symbol, _Period, 1);    //--- Get previous close
      bool tapped = false;                               //--- Initialize tap flag
      bool overlap = (prevLow <= zones[i].high && prevHigh >= zones[i].low); //--- Check candle overlap
      if (zones[i].isDemand) {                           //--- Check demand zone
         if (overlap && prevClose > zones[i].high) {     //--- Confirm demand tap
            tapped = true;                               //--- Set tapped flag
         }
      } else {                                           //--- Check supply zone
         if (overlap && prevClose < zones[i].low) {      //--- Confirm supply tap
            tapped = true;                               //--- Set tapped flag
         }
      }
      if (tapped) {                                      //--- Process tapped zone
         bool trendConfirmed = (trendConfirmation == NoConfirmation); //--- Assume no trend confirmation
         if (trendConfirmation == ConfirmTrend) {        //--- Check trend confirmation
            int oldShift = 2 + trendLookbackBars - 1;    //--- Calculate lookback shift
            if (oldShift >= iBars(_Symbol, _Period)) continue; //--- Skip if insufficient bars
            double oldClose = iClose(_Symbol, _Period, oldShift); //--- Get old close
            double recentClose = iClose(_Symbol, _Period, 2); //--- Get recent close
            double minChange = minTrendPoints * _Point; //--- Calculate min trend change
            if (zones[i].isDemand) {                    //--- Check demand trend
               trendConfirmed = (oldClose > recentClose + minChange); //--- Confirm downtrend
            } else {                                    //--- Check supply trend
               trendConfirmed = (oldClose < recentClose - minChange); //--- Confirm uptrend
            }
         }
         if (!trendConfirmed) continue;                 //--- Skip if trend not confirmed
         bool wasTested = zones[i].tested;              //--- Store previous tested status
         if (zones[i].isDemand) {                       //--- Handle demand trade
            double entryPrice = currentAsk;             //--- Set entry at ask
            double stopLossPrice = NormalizeDouble(zones[i].low - stopLossDistance * _Point, _Digits); //--- Set stop loss
            double takeProfitPrice = NormalizeDouble(entryPrice + takeProfitDistance * _Point, _Digits); //--- Set take profit
            obj_Trade.Buy(tradeLotSize, _Symbol, entryPrice, stopLossPrice, takeProfitPrice, "Buy at Demand Zone"); //--- Execute buy trade
            Print("Buy trade entered at Demand Zone: ", zones[i].name); //--- Log buy trade
         } else {                                       //--- Handle supply trade
            double entryPrice = currentBid;             //--- Set entry at bid
            double stopLossPrice = NormalizeDouble(zones[i].high + stopLossDistance * _Point, _Digits); //--- Set stop loss
            double takeProfitPrice = NormalizeDouble(entryPrice - takeProfitDistance * _Point, _Digits); //--- Set take profit
            obj_Trade.Sell(tradeLotSize, _Symbol, entryPrice, stopLossPrice, takeProfitPrice, "Sell at Supply Zone"); //--- Execute sell trade
            Print("Sell trade entered at Supply Zone: ", zones[i].name); //--- Log sell trade
         }
         zones[i].tested = true;                        //--- Mark zone as tested
         zones[i].tradeCount++;                         //--- Increment trade count
         if (!wasTested && zones[i].tested) {           //--- Check if newly tested
            Print("Zone tested: ", zones[i].name, ", Trade count: ", zones[i].tradeCount); //--- Log tested zone
         }
         color zoneColor = zones[i].isDemand ? testedDemandZoneColor : testedSupplyZoneColor; //--- Set tested color
         ObjectSetInteger(0, zones[i].name, OBJPROP_COLOR, zoneColor); //--- Update zone color
         string labelName = zones[i].name + "Label";                   //--- Get label name
         string labelText = zones[i].isDemand ? "Demand Zone (Tested)" : "Supply Zone (Tested)"; //--- Set tested label
         ObjectSetString(0, labelName, OBJPROP_TEXT, labelText);       //--- Update label text
      }
   }
   ChartRedraw(0);                                       //--- Redraw chart
}
```

To implement the trading logic for zone retests or taps, we create the "TradeOnZones" function. First, we track new bars with a [static](https://www.mql5.com/en/docs/basis/variables/static) "lastTradeCheck" and exit if not new or already checked, updating "lastTradeCheck" with [iTime](https://www.mql5.com/en/docs/series/itime) if true, and retrieve bid and ask prices normalized with the [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) and [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble) functions. We iterate through "zones", skipping broken, over-tested (based on "tradeTestedMode" and "maxTradesPerZone"), or unready zones, then check the previous bar’s high ( [iHigh](https://www.mql5.com/en/docs/series/ihigh)), low ("iLow"), and close ( [iClose](https://www.mql5.com/en/docs/series/iclose)) for overlap with the zone; for demand zones ("isDemand"), we confirm a tap if "overlap" is true and "prevClose > high", for supply if "overlap" and "prevClose < low", setting "tapped" accordingly.

If tapped, we confirm trend if "trendConfirmation" is "ConfirmTrend" by comparing old and recent closes ("iClose") over "trendLookbackBars" against "minTrendPoints \* [\_Point](https://www.mql5.com/en/docs/predefined/_point)", skipping if not confirmed. For valid taps, we execute trades: for demand, buy at ask with stop loss below "low" by "stopLossDistance \* \_Point" and take profit above entry by "takeProfitDistance \* \_Point" using "obj\_Trade.Buy", logging with [Print](https://www.mql5.com/en/docs/common/print); for supply, sell at bid with stop loss above "high" and take profit below entry, using "obj\_Trade.Sell". We mark the zone as "tested", increment "tradeCount", log if newly tested, update the zone color to "testedDemandZoneColor" or "testedSupplyZoneColor" with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger), and refresh the label text to "Demand/Supply Zone (Tested)" with the [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) function. Finally, we redraw the chart. When we call the function, we get the following outcome.

![CONFIRMED SIGNALS](https://c.mql5.com/2/171/Screenshot_2025-09-23_020706.png)

From the image, we can see that we detect the zone tapping and store the number of trades or taps essentially for that zone, so that we can trade it on some other taps if needed. What now remains is adding a trailing stop to maximize the gains. We will have it in a function as well.

```
//+------------------------------------------------------------------+
//| Apply trailing stop to open positions                            |
//+------------------------------------------------------------------+
void ApplyTrailingStop() {
   double point = _Point;                               //--- Get point value
   for (int i = PositionsTotal() - 1; i >= 0; i--) {    //--- Iterate through positions
      if (PositionGetTicket(i) > 0) {                   //--- Check valid ticket
         if (PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == uniqueMagicNumber) { //--- Verify symbol and magic
            double sl = PositionGetDouble(POSITION_SL); //--- Get current stop loss
            double tp = PositionGetDouble(POSITION_TP); //--- Get current take profit
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN); //--- Get open price
            if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) { //--- Check buy position
               double newSL = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID) - trailingStopPoints * point, _Digits); //--- Calculate new stop loss
               if (newSL > sl && SymbolInfoDouble(_Symbol, SYMBOL_BID) - openPrice > minProfitToTrail * point) { //--- Check trailing condition
                  obj_Trade.PositionModify(PositionGetInteger(POSITION_TICKET), newSL, tp); //--- Update stop loss
               }
            } else if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) { //--- Check sell position
               double newSL = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK) + trailingStopPoints * point, _Digits); //--- Calculate new stop loss
               if (newSL < sl && openPrice - SymbolInfoDouble(_Symbol, SYMBOL_ASK) > minProfitToTrail * point) { //--- Check trailing condition
                  obj_Trade.PositionModify(PositionGetInteger(POSITION_TICKET), newSL, tp); //--- Update stop loss
               }
            }
         }
      }
   }
}
```

Here, we implement the trailing stop logic to manage open positions dynamically. In the "ApplyTrailingStop" function, we retrieve the point value with [\_Point](https://www.mql5.com/en/docs/predefined/_point) and iterate backward through open positions using [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal), verifying each position’s ticket with [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket), symbol with [PositionGetString](https://www.mql5.com/en/docs/trading/positiongetstring), and magic number with [PositionGetInteger](https://www.mql5.com/en/docs/trading/positiongetinteger) against the magic number.

For buy positions ( [POSITION\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type)), we calculate a new stop loss as the bid price ( [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) with [SYMBOL\_BID](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double)) minus "trailingStopPoints \* point", normalized with [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble), and update it with "obj\_Trade.PositionModify" if higher than the current stop loss (" [PositionGetDouble(POSITION\_SL)](https://www.mql5.com/en/docs/trading/positiongetdouble)") and the profit exceeds "minProfitToTrail \* point". For sell positions, we calculate the new stop loss as the ask price ("SYMBOL\_ASK") plus "trailingStopPoints \* point", updating if lower than the current stop loss and profit exceeds the threshold, thus adjusting stop losses to lock in profits during favorable price movements. We can just call it on every tick now to do the management as follows.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   if (enableTrailingStop) {                      //--- Check if trailing stop enabled
      ApplyTrailingStop();                        //--- Apply trailing stop to positions
   }
   static datetime lastBarTime = 0;               //--- Store last processed bar time
   datetime currentBarTime = iTime(_Symbol, _Period, 0); //--- Get current bar time
   bool isNewBar = (currentBarTime != lastBarTime); //--- Check for new bar
   if (isNewBar) {                                //--- Process new bar
      lastBarTime = currentBarTime;               //--- Update last bar time
      DetectZones();                              //--- Detect new zones
      ValidatePotentialZones();                   //--- Validate potential zones
      UpdateZones();                              //--- Update existing zones
   }
   if (enableTrading) {                           //--- Check if trading enabled
      TradeOnZones(isNewBar);                     //--- Execute trades on zones
   }
}
```

When we run the program, we get the following outcome.

![TRAILING STOP ACTIVATED](https://c.mql5.com/2/171/Screenshot_2025-09-23_022532.png)

From the image, we can see that the trailing stop is fully enabled when the price goes in our favour. Here is a unified test for both zones in the previous month.

![UNIFIED SUPPLY AND DEMAND TEST GIF](https://c.mql5.com/2/171/Supply_and_demand_GIF.gif)

From the visualization, we can see that the program identifies and verifies all the entry conditions, and if validated, opens the respective position with the respective entry parameters, hence achieving our objective. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/171/Screenshot_2025-09-23_025057.png)

Backtest report:

![REPORT](https://c.mql5.com/2/171/Screenshot_2025-09-23_025126.png)

### Conclusion

In conclusion, we’ve created a [supply and demand](https://www.mql5.com/go?link=https://www.quantifiedstrategies.com/supply-and-demand-trading-strategy/ "https://www.quantifiedstrategies.com/supply-and-demand-trading-strategy/") trading system in MQL5 for detecting supply and demand zones through consolidation, validating them with impulsive moves, and trading retests with trend confirmation and customizable risk settings. The system visualizes zones with dynamic labels and colors, incorporating trailing stops for effective risk management.

Disclaimer: This article is for educational purposes only. Trading carries significant financial risks, and market volatility may result in losses. Thorough backtesting and careful risk management are crucial before deploying this program in live markets.

With this supply and demand strategy, you’re equipped to trade retest opportunities, ready for further optimization in your trading journey. Happy trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19674.zip "Download all attachments in the single ZIP archive")

[Supply\_and\_Demand\_EA.mq5](https://www.mql5.com/en/articles/download/19674/Supply_and_Demand_EA.mq5 "Download Supply_and_Demand_EA.mq5")(36.79 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/496768)**
(4)


![jiyanazad](https://c.mql5.com/avatar/avatar_na2.png)

**[jiyanazad](https://www.mql5.com/en/users/jiyanazad)**
\|
27 Nov 2025 at 20:59

Hi,

i compiled it withou any errors.Unfortunately, you can't see anything in the [backtest](https://www.mql5.com/en/articles/2612 "Article: Testing trading strategies on real ticks ") or on the chart. Neither supply & demand zones are plotted, nor are signals recognized or trades executed. The debug and journal are also empty. Has anyone gotten it to work? Can the auto check if the code is complete and up to date?


![bzaranyika](https://c.mql5.com/avatar/avatar_na2.png)

**[bzaranyika](https://www.mql5.com/en/users/bzaranyika)**
\|
4 Dec 2025 at 19:04

Worked on 1m time frame


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
5 Dec 2025 at 07:52

**bzaranyika [#](https://www.mql5.com/en/forum/496768#comment_58657951):**

Worked on 1m time frame

Thanks for the kind feedback


![Olowogbade Sunday](https://c.mql5.com/avatar/2021/2/601767D3-D824.png)

**[Olowogbade Sunday](https://www.mql5.com/en/users/olowogbadesunday)**
\|
6 Jan 2026 at 23:44

Thanks for this great help

But it doesn’t work on some quotes, is it an error on my end.

If yes, pls what and what are the criteria for the code to work on any currency or tradable asset

![Neural Networks in Trading: A Hybrid Trading Framework with Predictive Coding (Final Part)](https://c.mql5.com/2/107/Neural_networks_in_trading_Hybrid_trading_framework_ending_LOGO.png)[Neural Networks in Trading: A Hybrid Trading Framework with Predictive Coding (Final Part)](https://www.mql5.com/en/articles/16713)

We continue our examination of the StockFormer hybrid trading system, which combines predictive coding and reinforcement learning algorithms for financial time series analysis. The system is based on three Transformer branches with a Diversified Multi-Head Attention (DMH-Attn) mechanism that enables the capturing of complex patterns and interdependencies between assets. Previously, we got acquainted with the theoretical aspects of the framework and implemented the DMH-Attn mechanisms. Today, we will talk about the model architecture and training.

![Developing Advanced ICT Trading Systems: Implementing Signals in the Order Blocks Indicator](https://c.mql5.com/2/100/Final_Logo.png)[Developing Advanced ICT Trading Systems: Implementing Signals in the Order Blocks Indicator](https://www.mql5.com/en/articles/16268)

In this article, you will learn how to develop an Order Blocks indicator based on order book volume (market depth) and optimize it using buffers to improve accuracy. This concludes the current stage of the project and prepares for the next phase, which will include the implementation of a risk management class and a trading bot that uses signals generated by the indicator.

![MQL5 Wizard Techniques you should know (Part 81):  Using Patterns of Ichimoku and the ADX-Wilder with Beta VAE Inference Learning](https://c.mql5.com/2/173/19781-mql5-wizard-techniques-you-logo.png)[MQL5 Wizard Techniques you should know (Part 81): Using Patterns of Ichimoku and the ADX-Wilder with Beta VAE Inference Learning](https://www.mql5.com/en/articles/19781)

This piece follows up ‘Part-80’, where we examined the pairing of Ichimoku and the ADX under a Reinforcement Learning framework. We now shift focus to Inference Learning. Ichimoku and ADX are complimentary as already covered, however we are going to revisit the conclusions of the last article related to pipeline use. For our inference learning, we are using the Beta algorithm of a Variational Auto Encoder. We also stick with the implementation of a custom signal class designed for integration with the MQL5 Wizard.

![Price Action Analysis Toolkit Development (Part 43): Candlestick Probability and Breakouts](https://c.mql5.com/2/173/19738-price-action-analysis-toolkit-logo__1.png)[Price Action Analysis Toolkit Development (Part 43): Candlestick Probability and Breakouts](https://www.mql5.com/en/articles/19738)

Enhance your market analysis with the MQL5-native Candlestick Probability EA, a lightweight tool that transforms raw price bars into real-time, instrument-specific probability insights. It classifies Pinbars, Engulfing, and Doji patterns at bar close, uses ATR-aware filtering, and optional breakout confirmation. The EA calculates raw and volume-weighted follow-through percentages, helping you understand each pattern's typical outcome on specific symbols and timeframes. On-chart markers, a compact dashboard, and interactive toggles allow easy validation and focus. Export detailed CSV logs for offline testing. Use it to develop probability profiles, optimize strategies, and turn pattern recognition into a measurable edge.

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/19674&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068756730474593611)

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