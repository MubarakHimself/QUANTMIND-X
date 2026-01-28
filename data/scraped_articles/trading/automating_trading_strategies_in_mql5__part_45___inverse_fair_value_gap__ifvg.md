---
title: Automating Trading Strategies in MQL5 (Part 45): Inverse Fair Value Gap (IFVG)
url: https://www.mql5.com/en/articles/20361
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T17:52:10.150460
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=qjsfcqmvqpgwupuqtwscvpkhsltyfsrz&ssn=1769179927372575347&ssn_dr=0&ssn_sr=0&fv_date=1769179927&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F20361&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2045)%3A%20Inverse%20Fair%20Value%20Gap%20(IFVG)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917992755649296&fz_uniq=5068740641527102742&sv=2552)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 44)](https://www.mql5.com/en/articles/20355), we developed a [Change of Character (CHoCH)](https://www.mql5.com/go?link=https://innercircletrader.net/tutorials/change-of-character-choch-in-trading/ "https://innercircletrader.net/tutorials/change-of-character-choch-in-trading/") detection system in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5). This system scanned bars to identify and label swing highs and lows for trend determination. It triggered trades on breaks signaling reversals. The system supported per-bar and tick modes, and visualized with icons, labels, break lines, and dynamic fonts. In Part 45, we develop an [Inverse Fair Value Gap (IFVG)](https://www.mql5.com/go?link=https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/Inversion-Fair-Value-Gaps "https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/Inversion-Fair-Value-Gaps") system.

This system identifies bullish or bearish [Fair Value Gaps (FVGs)](https://www.mql5.com/go?link=https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/Fair-Value-Gaps "https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/Fair-Value-Gaps") on recent bars, applying a minimum gap size filter. It tracks states as normal, mitigated, or inverted based on price interactions. Mitigation occurs on far-side breaks, retracement on re-entry, and inversion on close beyond the far side from inside. The system ignores overlaps while limiting tracked FVGs. It supports one, limited, or unlimited trades per FVG. It opens buys on bullish IFVGs or sells on bearish. Zones are visualized as colored rectangles with state/trade labels, as well as mitigation icons. We will cover the following topics:

1. [Understanding the Inverse Fair Value Gap (IFVG) Framework](https://www.mql5.com/en/articles/2361#para2)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/2361#para3)
3. [Backtesting](https://www.mql5.com/en/articles/2361#para4)
4. [Conclusion](https://www.mql5.com/en/articles/2361#para1)

By the end, you’ll have a functional MQL5 strategy for detecting and trading IFVGs with state tracking, adaptive visuals, and configurable modes—let’s dive in!

### Understanding the Inverse Fair Value Gap (IFVG) Framework

The [Fair Value Gap (FVG)](https://www.mql5.com/go?link=https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/Fair-Value-Gaps "https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/Fair-Value-Gaps") is a price action concept representing imbalances or gaps between candles where buying or selling pressure created an unfilled void, often seen as bullish FVGs (low of later candle above high of earlier) or bearish FVGs (high of later below low of earlier), acting as potential support/resistance zones that price may retrace to fill. An Inverse Fair Value Gap (IFVG) occurs when a mitigated FVG (price broke the far side) is retraced into and then inverted by price closing beyond the far side from inside, signaling a reversal: a mitigated bullish FVG inverting bearish (price closes below low after re-entry) or mitigated bearish inverting bullish (closes above high). States track progression — normal (initial gap), mitigated (far-side break), retraced (re-entry after mitigation), inverted (close beyond far from inside post-retrace) — with inversion as the key trade signal.

In a mitigated bearish FVG (original down gap), a bullish IFVG triggers when price re-enters after mitigation and closes above the high, entering buy with stop-loss below low and take-profit at fixed points. Conversely, for a mitigated bullish FVG (original up), a bearish IFVG on close below low enters sell with stop-loss above high. Have a look below at the different setups we could have.

![INVERSE FAIR VALUE GAP (IFVG) SETUPS](https://c.mql5.com/2/182/Screenshot_2025-11-20_093807.png)

Our plan is to detect FVGs on recent bars with minimum gap filtering, load historical FVGs on initialization, track/update states (normal/mitigated/inverted) based on price interactions, ignore overlaps, limit tracked FVGs with cleanup of expired, trade inversions with buys on bullish IFVGs (orig down inverted) or sells on bearish (orig up inverted), fixed trade levels, trade modes/counts per FVG, trailing stops, and visualize colored rectangles (normal/mitigated/inverted shades) with state/trade labels and mitigation icons. In brief, here is a visual representation of our objectives.

![IFVGS FRAMEWORK](https://c.mql5.com/2/182/Screenshot_2025-11-20_094135.png)

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Experts folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some [input parameters](https://www.mql5.com/en/docs/basis/variables/inputvariables) and [global variables](https://www.mql5.com/en/docs/basis/variables/global) that we will use throughout the program.

```
//+------------------------------------------------------------------+
//|                                                  FVG Inverse.mq5 |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//|                                   https://t.me/Forex_Algo_Trader |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Allan Munene Mutiiria."
#property link      "https://t.me/Forex_Algo_Trader"
#property version   "1.00"

#include <Trade/Trade.mqh>

//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+
CTrade obj_Trade;                                                 //--- Trade object
#define FVG_Prefix "IFVG REC "                                    //--- FVG prefix
// Normal FVGs
#define CLR_UP clrGreen                                           // Green for normal up (Bullish FVG)
#define CLR_DOWN clrRed                                           // Red for normal down (Bearish FVG)
// Mitigated FVGs
#define CLR_MIT_UP clrPurple                                      // Purple for mitigated up (Mitigated Bullish FVG)
#define CLR_MIT_DOWN clrOrange                                    // Orange for mitigated down (Mitigated Bearish FVG)
// Inverted FVGs
#define CLR_INV_UP clrRed                                         // Red for inverted up (Bearish IFVG)
#define CLR_INV_DOWN clrGreen                                     // Green for inverted down (Bullish IFVG)

//+------------------------------------------------------------------+
//| Enums                                                            |
//+------------------------------------------------------------------+
enum TradeMode {                                                  // Define trade mode enum
   TradeOnce,                                                     // Trade Once
   LimitedTrades,                                                 // Limited Trades
   UnlimitedTrades                                                // Unlimited Trades
};

enum FVGState {                                                   // Define FVG state enum
   Normal,                                                        // Normal
   Mitigated,                                                     // Mitigated
   Inverted                                                       // Inverted
};

enum TrailingTypeEnum {                                           // Define enum for trailing stop types
   Trailing_None   = 0,                                           // None
   Trailing_Points = 2                                            // By Points
};

//+------------------------------------------------------------------+
//| Input Parameters                                                 |
//+------------------------------------------------------------------+
input group "EA GENERAL SETTINGS"
input double inpLot             = 0.01;                           // Lotsize
input int    sl_pts             = 300;                            // Stop Loss Points
input int    tp_pts             = 300;                            // Take Profit Points
input int    minPts             = 100;                            // Minimum Gap Size in Points
input int    FVG_Rec_Ext_Bars   = 30;                             // FVG Extension Bars
input bool   prt                = true;                           // Print Statements
input long   magic_number       = 123456789;                      // Magic Number
input bool   ignoreOverlaps     = true;                           // Ignore new FVGs that overlap existing ones
input TradeMode tradeMode       = TradeOnce;                      // Mode for trading FVGs
input int    maxTradesPerFVG    = 2;                              // Maximum trades per FVG for LimitedTrades
input int    maxFVGs            = 50;                             // Maximum FVGs to track in array
input TrailingTypeEnum TrailingType = Trailing_None;              // Trailing Stop Type
input double Trailing_Stop_Pips = 30.0;                           // Trailing Stop in Pips (for Points type)
input double Min_Profit_To_Trail_Pips = 50.0;                     // Min Profit to Start Trailing in Pips
```

We begin the implementation by including the trade library with " [#include](https://www.mql5.com/en/docs/basis/preprosessor/include) <Trade/Trade.mqh>", which provides the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class for managing orders and positions. We declare "obj\_Trade" as a global instance of "CTrade" to handle all trading operations. We [define](https://www.mql5.com/en/docs/basis/preprosessor/constant) a string constant "FVG\_Prefix" as "IFVG REC " for naming FVG rectangle objects. We set color constants for different FVG states and directions: "CLR\_UP" as green for normal bullish FVGs, "CLR\_DOWN" as red for normal bearish, "CLR\_MIT\_UP" as purple for mitigated bullish, "CLR\_MIT\_DOWN" as orange for mitigated bearish, "CLR\_INV\_UP" as red for inverted bearish (orig up), and "CLR\_INV\_DOWN" as green for inverted bullish. You can change these to your desired ones; they are just arbitrary colors we used for our white background chart.

Then, we create three [enumerations](https://www.mql5.com/en/docs/basis/types/integer/enumeration) for configuration. The "TradeMode" enum offers "TradeOnce" to limit to one trade per FVG, "LimitedTrades" for a user-defined max per FVG, and "UnlimitedTrades" with no per-FVG limit. This is important if you want to trade multiple taps. The "FVGState" enum defines "Normal" for initial gaps, "Mitigated" after far-side breaks, and "Inverted" on inversion signals. The "TrailingTypeEnum" enum provides "Trailing\_None" to disable trailing and "Trailing\_Points" for points-based trailing stops.

We group the [input parameters](https://www.mql5.com/en/docs/basis/variables/inputvariables) under "EA GENERAL SETTINGS" for the properties dialog. These include "inpLot" for lot size, "sl\_pts" and "tp\_pts" for stop-loss and take-profit distances in points, "minPts" as the minimum gap size to qualify as an FVG, "FVG\_Rec\_Ext\_Bars" for how many bars to extend FVG rectangles rightward, "prt" to toggle print logging, "magic\_number" for trade identification, "ignoreOverlaps" to skip new FVGs overlapping existing ones (we thought this would be extrememly important for visual clarity, but you can ignore), "tradeMode" using the enum for trading limits, "maxTradesPerFVG" for the limit in limited mode, "maxFVGs" to cap tracked FVGs in the array, "TrailingType" with its enum, "Trailing\_Stop\_Pips" for trailing distance, and "Min\_Profit\_To\_Trail\_Pips" for the profit threshold before trailing starts. With the inputs in place, we will define some structure and helper functions to help in managing our setups.

```
//+------------------------------------------------------------------+
//| Structure for FVG zone information                               |
//+------------------------------------------------------------------+
struct FVGZone {                                                  // Define FVG zone structure
   string   name;                                                 //--- Zone name
   datetime startTime;                                            //--- Start time
   datetime origEndTime;                                          //--- Original end time
   datetime mitTime;                                              //--- Mitigation time
   bool     signal;                                               //--- Signal flag
   bool     inverted;                                             //--- Inverted flag
   bool     mit;                                                  //--- Mitigated flag
   bool     ret;                                                  //--- Retraced flag
   bool     origUp;                                               //--- Original up flag
   int      tradeCount;                                           //--- Trade count
   FVGState state;                                                //--- State
   bool     newSignal;                                            //--- New signal flag
};
FVGZone fvgs[];                                                   //--- FVG zones array

//+------------------------------------------------------------------+
//| Get color based on state and direction                           |
//+------------------------------------------------------------------+
color GetFVGColor(bool isUp, FVGState currentState) {
   if (currentState == Normal) return isUp ? CLR_UP : CLR_DOWN;   //--- Return normal color
   if (currentState == Mitigated) return isUp ? CLR_MIT_UP : CLR_MIT_DOWN; //--- Return mitigated color
   if (currentState == Inverted) return isUp ? CLR_INV_UP : CLR_INV_DOWN; //--- Return inverted color
   return clrNONE;                                                //--- Return none
}

//+------------------------------------------------------------------+
//| Print FVGs for debugging                                         |
//+------------------------------------------------------------------+
void PrintFVGs() {
   if (!prt) return;                                              //--- Return if no print
   Print("Current FVGs count: ", ArraySize(fvgs));                //--- Print count
   for (int i = 0; i < ArraySize(fvgs); i++) {                    //--- Iterate FVGs
      Print("FVG ", i, ": ", fvgs[i].name, " state=", EnumToString(fvgs[i].state), " mit=", fvgs[i].mit, " ret=", fvgs[i].ret, " inverted=", fvgs[i].inverted, " tradeCount=", fvgs[i].tradeCount, " newSignal=", fvgs[i].newSignal, " endTime=", TimeToString(fvgs[i].origEndTime)); //--- Print details
   }
}
```

First, we define the "FVGZone" [structure](https://www.mql5.com/en/docs/basis/types/classes) to hold all relevant information for each detected fair value gap, including "name" as a string for the object identifier, "startTime" and "origEndTime" as [datetimes](https://www.mql5.com/en/docs/basis/types/integer/datetime) for the gap's initial span, "mitTime" for when mitigation occurs, [boolean](https://www.mql5.com/en/docs/basis/types/integer/boolconst) flags like "signal" for inversion triggers, "inverted" for inversion status, "mit" for mitigation, "ret" for retracement, "origUp" to indicate if it was originally a bullish gap, "tradeCount" as an integer to track trades on this FVG, "state" using the state enum, and "newSignal" as a boolean for fresh inversion signals. We then declare a global array "fvgs\[\]" of type "FVGZone" to store all active FVGs, allowing us to track multiple gaps efficiently with their states.

We implement the "GetFVGColor" [function](https://www.mql5.com/en/docs/basis/function) to determine the appropriate color for an FVG rectangle based on its direction "isUp" and "currentState": for "Normal", we return "CLR\_UP" (green) if up or "CLR\_DOWN" (red) if down; for "Mitigated", "CLR\_MIT\_UP" (purple) or "CLR\_MIT\_DOWN" (orange); for "Inverted", "CLR\_INV\_UP" (red) or "CLR\_INV\_DOWN" (green); otherwise, none. We also create the "PrintFVGs" function for debugging, which returns early if "prt" is false, otherwise prints the current count from [ArraySize](https://www.mql5.com/en/docs/array/arraysize), then loops through each entry to print details like index, name, state via [EnumToString](https://www.mql5.com/en/docs/convert/enumtostring), flags for mit/ret/inverted, trade count, new signal, and end time with the [TimeToString](https://www.mql5.com/en/docs/convert/timetostring) function. We can now proceed to defining the visual functions for the rectangles and labels.

```
//+------------------------------------------------------------------+
//| Create Rectangle                                                 |
//+------------------------------------------------------------------+
void CreateRec(string objName, datetime time1, double price1, datetime time2, double price2, color clr) {
   ObjectCreate(0, objName, OBJ_RECTANGLE, 0, time1, price1, time2, price2); //--- Create rectangle
   ObjectSetInteger(0, objName, OBJPROP_FILL, true);              //--- Set fill
   ObjectSetInteger(0, objName, OBJPROP_COLOR, clr);              //--- Set color
   ObjectSetInteger(0, objName, OBJPROP_BACK, false);             //--- Set foreground
   datetime midTime = time1 + (time2 - time1) / 2;                //--- Calc mid time
   double midPrice = (price1 + price2) / 2;                       //--- Calc mid price
   CreateLabel(objName, midTime, midPrice);                       //--- Create label
   ChartRedraw(0);                                                //--- Redraw chart
}

//+------------------------------------------------------------------+
//| Update Rectangle                                                 |
//+------------------------------------------------------------------+
void UpdateRec(string objName, datetime time1, double price1, datetime time2, double price2, color clr) {
   if (ObjectFind(0, objName) >= 0) {                             //--- Check exists
      ObjectSetInteger(0, objName, OBJPROP_TIME, 0, time1);       //--- Set time1
      ObjectSetDouble(0, objName, OBJPROP_PRICE, 0, price1);      //--- Set price1
      ObjectSetInteger(0, objName, OBJPROP_TIME, 1, time2);       //--- Set time2
      ObjectSetDouble(0, objName, OBJPROP_PRICE, 1, price2);      //--- Set price2
      ObjectSetInteger(0, objName, OBJPROP_COLOR, clr);           //--- Set color
      datetime midTime = time1 + (time2 - time1) / 2;             //--- Calc mid time
      double midPrice = (price1 + price2) / 2;                    //--- Calc mid price
      UpdateLabel(objName, midTime, midPrice);                    //--- Update label
      ChartRedraw(0);                                             //--- Redraw chart
   }
}

//+------------------------------------------------------------------+
//| Create label                                                     |
//+------------------------------------------------------------------+
void CreateLabel(string zoneName, datetime time, double price) {
   string lblName = zoneName + "_Label";                          //--- Label name
   ObjectCreate(0, lblName, OBJ_TEXT, 0, time, price);            //--- Create text
   ObjectSetInteger(0, lblName, OBJPROP_ANCHOR, ANCHOR_CENTER);   //--- Set anchor
   ObjectSetInteger(0, lblName, OBJPROP_COLOR, clrBlack);         //--- Set color
   UpdateLabelText(lblName, zoneName);                            //--- Update text
}

//+------------------------------------------------------------------+
//| Update label position                                            |
//+------------------------------------------------------------------+
void UpdateLabel(string zoneName, datetime time, double price) {
   string lblName = zoneName + "_Label";                          //--- Label name
   if (ObjectFind(0, lblName) >= 0) {                             //--- Check exists
      ObjectSetInteger(0, lblName, OBJPROP_TIME, 0, time);        //--- Set time
      ObjectSetDouble(0, lblName, OBJPROP_PRICE, 0, price);       //--- Set price
      UpdateLabelText(lblName, zoneName);                         //--- Update text
   }
}

//+------------------------------------------------------------------+
//| Update label text                                                |
//+------------------------------------------------------------------+
void UpdateLabelText(string lblName, string zoneName) {
   string text = "";                                              //--- Init text
   int tradeCnt = 0;                                              //--- Init count
   FVGState state = Normal;                                       //--- Init state
   bool origUp = false;                                           //--- Init orig up
   for (int idx = 0; idx < ArraySize(fvgs); idx++) {              //--- Iterate FVGs
      if (fvgs[idx].name == zoneName) {                           //--- Check match
         tradeCnt = fvgs[idx].tradeCount;                         //--- Get count
         state = fvgs[idx].state;                                 //--- Get state
         origUp = fvgs[idx].origUp;                               //--- Get orig up
         break;                                                   //--- Break loop
      }
   }
   if (state == Normal) {                                         //--- Check normal
      text = origUp ? "Bullish FVG" : "Bearish FVG";              //--- Set text
   } else if (state == Mitigated) {                               //--- Check mitigated
      text = origUp ? "Mitigated Bullish FVG" : "Mitigated Bearish FVG"; //--- Set text
   } else if (state == Inverted) {                                //--- Check inverted
      text = origUp ? "Bearish Inversed FVG" : "Bullish Inversed FVG"; //--- Set text
   }
   if (tradeCnt > 0) {                                            //--- Check traded
      text += " (Traded " + IntegerToString(tradeCnt) + " times)"; //--- Add traded
   }
   ObjectSetString(0, lblName, OBJPROP_TEXT, text);               //--- Set text
}

//+------------------------------------------------------------------+
//| Draw mitigation icon                                              |
//+------------------------------------------------------------------+
void DrawMitIcon(string fvgNAME, datetime mitTime, double fvgHigh, double fvgLow, bool isUp) {
   string iconName = fvgNAME + "_MitIcon";                        //--- Icon name
   double iconPrice = isUp ? fvgLow : fvgHigh;                    //--- Icon price
   ObjectCreate(0, iconName, OBJ_ARROW, 0, mitTime, iconPrice);   //--- Create arrow
   ObjectSetInteger(0, iconName, OBJPROP_ARROWCODE, 251);         //--- Set code
   ObjectSetInteger(0, iconName, OBJPROP_COLOR, clrBlue);         //--- Set color
   ObjectSetInteger(0, iconName, OBJPROP_ANCHOR, isUp ? ANCHOR_TOP : ANCHOR_BOTTOM); //--- Set anchor
   ChartRedraw(0);                                                //--- Redraw chart
}
```

For the visualization, first, we define the "CreateRec" function to draw a new rectangle representing an FVG zone on the chart. We create the rectangle object with [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) using [OBJ\_RECTANGLE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_rectangle), spanning from time1 at price1 to time2 at price2. We enable filling with [OBJPROP\_FILL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) set to true, apply the provided color via "OBJPROP\_COLOR", position it in the foreground by setting [OBJPROP\_BACK](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to false, calculate the midpoint time as time1 plus half the duration and midpoint price as the average of price1 and price2, then call "CreateLabel" to add a descriptive text at the midpoint, and finally redraw the chart using the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function. We implement the "UpdateRec" function to modify an existing FVG rectangle when its state or color changes. If the object exists per [ObjectFind](https://www.mql5.com/en/docs/objects/objectfind), we update its coordinates by setting "OBJPROP\_TIME" and "OBJPROP\_PRICE" for both anchors, apply the new color, recalculate the midpoint time and price as before, call "UpdateLabel" to reposition and refresh the text, and redraw the chart.

Next,we create the "CreateLabel" function to add a text label inside the FVG rectangle. We form the label name by appending "\_Label" to the zone name, create an [OBJ\_TEXT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_text) object at the given time and price with "ObjectCreate", set its anchor to center via [OBJPROP\_ANCHOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer), color to black, then call "UpdateLabelText" to set the initial descriptive text. We define the "UpdateLabel" function to move an existing label when the rectangle adjusts. If the label exists, we update its position, then call "UpdateLabelText" to refresh its content based on the current state. We then implement the "UpdateLabelText" function to build and set the label's string dynamically. We initialize an empty text, then loop through the "fvgs" array to find the matching zone by name and retrieve its trade count, state, and original up direction.

Based on the state, we set text to "Bullish FVG" or "Bearish FVG" for normal, "Mitigated Bullish FVG" or "Mitigated Bearish FVG" for mitigated, "Bearish Inversed FVG" or "Bullish Inversed FVG" for inverted; if trade count exceeds 0, we append "(Traded X times)". We apply this text to the label with [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) and [OBJPROP\_TEXT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_string). Finally, we create the "DrawMitIcon" function to place a visual marker at mitigation events. We form the icon name by appending "\_MitIcon" to the FVG name, create an [OBJ\_ARROW](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_arrow) at the mitigation time and appropriate price (low for up gaps, high for down), set its arrow code to 251, color to blue, anchor to top for up or bottom for down based on "isUp", and redraw the chart. You can change the arrow code per your liking based on the [MQL5 Wingdings](https://www.mql5.com/en/docs/constants/objectconstants/wingdings) font as below.

![MQL5 WINGDINGS](https://c.mql5.com/2/182/C_MQL5_WINGDINGS__2.png)

Armed with these functions, we can create the initial markup of the setups on the chart using the available bars on the chart to create that indicator styling, so we are able to see the setups on initialization of the program. We will house the logic in a function, too.

```
//+------------------------------------------------------------------+
//| Process historical mitigation, retracement, signal for an FVG    |
//+------------------------------------------------------------------+
void ProcessHistoricalState(int idx) {
   string fvgNAME = fvgs[idx].name;                               //--- Get name
   datetime timeSTART = fvgs[idx].startTime;                      //--- Get start time
   datetime endTime = fvgs[idx].origEndTime;                      //--- Get end time
   double fvgLow = MathMin(ObjectGetDouble(0, fvgNAME, OBJPROP_PRICE, 0), ObjectGetDouble(0, fvgNAME, OBJPROP_PRICE, 1)); //--- Calc low
   double fvgHigh = MathMax(ObjectGetDouble(0, fvgNAME, OBJPROP_PRICE, 0), ObjectGetDouble(0, fvgNAME, OBJPROP_PRICE, 1)); //--- Calc high
   int fvgBar = iBarShift(_Symbol, _Period, timeSTART);           //--- Get bar
   if (fvgBar < 0) return;                                        //--- Return invalid
   bool isMit = false, isRet = false, isSig = false;              //--- Init flags
   datetime mitTime = 0;                                          //--- Init mit time
   int mitK = -1, sigK = -1;                                      //--- Init indices
   for (int k = fvgBar - 1; k >= 0; k--) {                        //--- Iterate bars
      double barLow = iLow(_Symbol, _Period, k);                  //--- Get bar low
      double barHigh = iHigh(_Symbol, _Period, k);                //--- Get bar high
      double barClose = iClose(_Symbol, _Period, k);              //--- Get bar close
      if (!isMit) {                                               //--- Check not mit
         bool breakFar = (fvgs[idx].origUp && barLow < fvgLow) || (!fvgs[idx].origUp && barHigh > fvgHigh); //--- Check break far
         if (breakFar) {                                          //--- Break far
            isMit = true;                                         //--- Set mit
            mitK = k;                                             //--- Set mit k
            mitTime = iTime(_Symbol, _Period, k);                 //--- Set mit time
            if (prt) Print("Historical Mitigated: ", fvgNAME, " at bar ", k, " time=", TimeToString(mitTime)); //--- Log mitigated
         }
      }
      if (isMit && !isRet) {                                      //--- Check mit and not ret
         bool inside = (barHigh > fvgLow && barLow < fvgHigh);    //--- Check inside
         if (inside) {                                            //--- Inside
            isRet = true;                                         //--- Set ret
            if (prt) Print("Historical Retraced: ", fvgNAME, " at bar ", k); //--- Log retraced
         }
      }
      if (isMit && isRet && !isSig) {                             //--- Check mit ret not sig
         bool signal = (fvgs[idx].origUp && barClose < fvgLow) || (!fvgs[idx].origUp && barClose > fvgHigh); //--- Check signal
         if (signal) {                                            //--- Signal
            if (k + 1 < iBars(_Symbol, _Period)) {                //--- Check prev bar
               double prevClose = iClose(_Symbol, _Period, k + 1); //--- Get prev close
               bool prevInside = (prevClose > fvgLow && prevClose < fvgHigh); //--- Check prev inside
               if (prevInside) {                                  //--- Prev inside
                  isSig = true;                                   //--- Set sig
                  sigK = k;                                       //--- Set sig k
                  if (prt) Print("Historical Signal/Inverted: ", fvgNAME, " at bar ", k, " time=", TimeToString(iTime(_Symbol, _Period, k))); //--- Log signal
               }
            }
         }
      }
   }
   fvgs[idx].mit = isMit;                                         //--- Set mit
   fvgs[idx].ret = isRet;                                         //--- Set ret
   fvgs[idx].inverted = isSig;                                    //--- Set inverted
   fvgs[idx].signal = isSig;                                      //--- Set signal
   fvgs[idx].mitTime = mitTime;                                   //--- Set mit time
   fvgs[idx].state = isSig ? Inverted : (isMit ? Mitigated : Normal); //--- Set state
   fvgs[idx].newSignal = false;                                   //--- Set no new signal
   color currentClr = GetFVGColor(fvgs[idx].origUp, fvgs[idx].state); //--- Get color
   UpdateRec(fvgs[idx].name, fvgs[idx].startTime, fvgLow, fvgs[idx].origEndTime, fvgHigh, currentClr); //--- Update rec
   if (mitTime > 0) DrawMitIcon(fvgs[idx].name, mitTime, fvgHigh, fvgLow, fvgs[idx].origUp); //--- Draw mit icon
}
```

Here, we define the "ProcessHistoricalState" function to analyze and update the state of a specific FVG in the array during initialization, checking for mitigation, retracement, and inversion based on historical bars after the gap's start. We start by retrieving the FVG's name, start time, and original end time from the structure at the given index, then calculate its low and high prices using [MathMin](https://www.mql5.com/en/docs/math/mathmin) and [MathMax](https://www.mql5.com/en/docs/math/mathmax) on the rectangle object's prices via [ObjectGetDouble](https://www.mql5.com/en/docs/objects/objectgetdouble) with [OBJPROP\_PRICE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_double). We find the bar index of the start time with [iBarShift](https://www.mql5.com/en/docs/series/ibarshift) and return early if invalid. We initialize flags for mitigation, retracement, and signal to false, along with a mitigation time and bar indices to -1. We then loop backward from the bar before the FVG to bar 0: for each bar, we get its low, high, and close with the [iLow](https://www.mql5.com/en/docs/series/ilow), [iHigh](https://www.mql5.com/en/docs/series/ihigh), and [iClose](https://www.mql5.com/en/docs/series/iclose) functions. If not yet mitigated, we check for a far-side break — for original up gaps (bullish FVG), if bar low drops below FVG low; for down gaps, if bar high exceeds FVG high — setting the mitigation flag, bar index, time via [iTime](https://www.mql5.com/en/docs/series/itime), and logging if "prt" is true.

If mitigated but not retraced, we check if the bar overlaps the FVG (high above low and low below high), setting the retracement flag and logging. If both mitigated and retraced but no signal, we detect inversion: for up gaps, if bar close is below FVG low; for down gaps, above FVG high — then verify the previous bar's close was inside the FVG (above low and below high) to confirm from-inside exit, setting the signal flag, bar index, and logging.

We update the structure fields: mitigation, retracement, inverted, and signal to the flag values, mitigation time, state as "Inverted" if signal, "Mitigated" if mitigated, or "Normal", and reset the new signal to false. We get the current color with "GetFVGColor" based on original direction and state, update the rectangle via "UpdateRec" with possibly adjusted coordinates and new color, and if mitigated, draw the mitigation icon with "DrawMitIcon" at the mitigation time and appropriate edge price (low for up gaps, high for down). We can now use this function in the initialization function to draw our first setups.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   obj_Trade.SetExpertMagicNumber(magic_number);                  //--- Set magic number
   ObjectsDeleteAll(0, FVG_Prefix);                               //--- Delete FVG objects
   ArrayResize(fvgs, 0);                                          //--- Reset array
   if (prt) Print("Initializing: Deleted all existing FVG objects and reset array."); //--- Log init
   int visibleBars = (int)ChartGetInteger(0, CHART_VISIBLE_BARS); //--- Get visible bars
   if (prt) Print("Total Visible Bars On Chart = ", visibleBars); //--- Log visible bars
   // Detect historical FVGs from older to newer
   for (int i = visibleBars - 3; i >= 0; i--) {                   //--- Iterate bars
      double low0 = iLow(_Symbol, _Period, i);                    //--- Get low0
      double high2 = iHigh(_Symbol, _Period, i + 2);              //--- Get high2
      double gap_L0_H2 = NormalizeDouble((low0 - high2) / _Point, _Digits); //--- Calc gap L0 H2
      double high0 = iHigh(_Symbol, _Period, i);                  //--- Get high0
      double low2 = iLow(_Symbol, _Period, i + 2);                //--- Get low2
      double gap_H0_L2 = NormalizeDouble((low2 - high0) / _Point, _Digits); //--- Calc gap H0 L2
      bool FVG_UP = low0 > high2 && gap_L0_H2 > minPts;           //--- Check up FVG
      bool FVG_DOWN = low2 > high0 && gap_H0_L2 > minPts;         //--- Check down FVG
      if (FVG_UP || FVG_DOWN) {                                   //--- Check FVG
         datetime time1 = iTime(_Symbol, _Period, i + 1);         //--- Get time1
         double price1 = FVG_UP ? high2 : high0;                  //--- Set price1
         double price2 = FVG_UP ? low0 : low2;                    //--- Set price2
         double newLow = MathMin(price1, price2);                 //--- Calc new low
         double newHigh = MathMax(price1, price2);                //--- Calc new high
         bool overlaps = false;                                   //--- Init overlaps
         if (ignoreOverlaps) {                                    //--- Check ignore overlaps
            for (int ex = 0; ex < ArraySize(fvgs); ex++) {        //--- Iterate existing
               double exLow = ObjectGetDouble(0, fvgs[ex].name, OBJPROP_PRICE, 0); //--- Get ex low
               double exHigh = ObjectGetDouble(0, fvgs[ex].name, OBJPROP_PRICE, 1); //--- Get ex high
               exLow = MathMin(exLow, exHigh);                    //--- Min ex low
               exHigh = MathMax(exLow, exHigh);                   //--- Max ex high
               if (MathMax(newLow, exLow) < MathMin(newHigh, exHigh)) { //--- Check overlap
                  overlaps = true;                                //--- Set overlaps
                  if (prt) Print("Historical: Skipping overlapping FVG at ", TimeToString(time1)); //--- Log skip
                  break;                                          //--- Break loop
               }
            }
         }
         if (overlaps) continue;                                  //--- Continue if overlaps
         string fvgNAME = FVG_Prefix + "(" + TimeToString(time1) + ")"; //--- FVG name
         color fvgClr = FVG_UP ? CLR_UP : CLR_DOWN;               //--- Set color
         CreateRec(fvgNAME, time1, price1, time1 + PeriodSeconds(_Period) * FVG_Rec_Ext_Bars, price2, fvgClr); //--- Create rec
         int size = ArraySize(fvgs);                              //--- Get size
         if (size >= maxFVGs) {                                   //--- Check max
            if (prt) Print("Historical: Max FVGs reached, removing oldest."); //--- Log max
            ArrayRemove(fvgs, 0, 1);                              //--- Remove oldest
            PrintFVGs();                                          //--- Print FVGs
         }
         ArrayResize(fvgs, size + 1);                             //--- Resize array
         fvgs[size].name = fvgNAME;                               //--- Set name
         fvgs[size].startTime = time1;                            //--- Set start time
         fvgs[size].origEndTime = time1 + PeriodSeconds(_Period) * FVG_Rec_Ext_Bars; //--- Set end time
         fvgs[size].mitTime = 0;                                  //--- Set mit time
         fvgs[size].signal = false;                               //--- Set signal
         fvgs[size].inverted = false;                             //--- Set inverted
         fvgs[size].mit = false;                                  //--- Set mit
         fvgs[size].ret = false;                                  //--- Set ret
         fvgs[size].origUp = FVG_UP;                              //--- Set orig up
         fvgs[size].tradeCount = 0;                               //--- Set trade count
         fvgs[size].state = Normal;                               //--- Set state
         fvgs[size].newSignal = false;                            //--- Set new signal
         if (prt) Print("Historical FVG created: ", fvgNAME, " origUp=", FVG_UP, " endTime=", TimeToString(fvgs[size].origEndTime)); //--- Log created
         PrintFVGs();                                             //--- Print FVGs
      }
   }
   // Process historical states
   for (int j = 0; j < ArraySize(fvgs); j++) {                    //--- Iterate FVGs
      ProcessHistoricalState(j);                                  //--- Process state
   }
   PrintFVGs();                                                   //--- Print FVGs
   return(INIT_SUCCEEDED);                                        //--- Return success
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   for (int i = 0; i < ArraySize(fvgs); i++) {                    //--- Iterate FVGs
      ObjectDelete(0, fvgs[i].name);                              //--- Delete name
      ObjectDelete(0, fvgs[i].name + "_Label");                   //--- Delete label
      ObjectDelete(0, fvgs[i].name + "_MitIcon");                 //--- Delete mit icon
   }
   ArrayResize(fvgs, 0);                                          //--- Reset array
   ChartRedraw(0);                                                //--- Redraw chart
   if (prt) Print("Deinit: Deleted all FVG objects and reset array."); //--- Log deinit
}
```

In the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, which runs when the program starts, we first assign the input "magic\_number" to "obj\_Trade" with "SetExpertMagicNumber" for trade identification. We then clear any existing FVG rectangles by calling [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/ObjectDeleteAll) with the current chart and "FVG\_Prefix", reset the "fvgs" array size to 0 using [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize), and log the initialization if "prt" is true. We retrieve the number of visible bars on the chart with [ChartGetInteger](https://www.mql5.com/en/docs/chart_operations/chartgetinteger) using [CHART\_VISIBLE\_BARS](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer) into "visibleBars", logging it if "prt".

To detect historical FVGs from oldest to newest, we loop from "visibleBars - 3" down to 0: for each bar i, we calculate potential gaps by comparing low of i to high of i+2 (normalized gap in points into "gap\_L0\_H2") and high of i to low of i+2 ("gap\_H0\_L2"), setting "FVG\_UP" if low0 > high2 and gap > "minPts" (bullish gap) or "FVG\_DOWN" if low2 > high0 and gap > "minPts" (bearish gap). We need at least 3 complete bars to detect a gap. It is extra important that you understand this once and for all. We visualized it for you for clarity as below.

![REQUIREMENTS FOR A FVG SETUP](https://c.mql5.com/2/182/Screenshot_2025-11-20_105033.png)

Continuing, if either type is found, we get the time of bar i+1 into "time1", set prices accordingly, calculate the gap's low and high with the [MathMin](https://www.mql5.com/en/docs/math/mathmin) and [MathMax](https://www.mql5.com/en/docs/math/mathmax) functions. If "ignoreOverlaps" is true, we check against all existing in "fvgs" by fetching their prices with [ObjectGetDouble](https://www.mql5.com/en/docs/objects/objectgetdouble) and comparing ranges — if overlapping (max of lows < min of highs), we set "overlaps" true and log skip if "prt", continuing to next. If no overlap, we form the name as "FVG\_Prefix + (TimeToString(time1))", choose color based on up or down, call "CreateRec" to draw the rectangle extended by "FVG\_Rec\_Ext\_Bars" periods. We check if the array size reaches "maxFVGs", removing the oldest with [ArrayRemove](https://www.mql5.com/en/docs/array/arrayremove) and logging if "prt", then resize "fvgs" up by one, populate the new entry with name, times, flags all false except "origUp" as "FVG\_UP", trade count 0, state "Normal", new signal false, log creation if "prt", and call "PrintFVGs". See how the overlaps could impact the visual appeal, but have no issue with the trading activity.

![OVERLAPPING SETUPS SAMPLE](https://c.mql5.com/2/182/Screenshot_2025-11-20_110824.png)

You can see that the overlapping instances are not visually appealing. Then, after detecting all historical FVGs, we loop through "fvgs" and call "ProcessHistoricalState" on each index to set initial states based on past price action, then call "PrintFVGs" again, and return [INIT\_SUCCEEDED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode). Finally, in the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) function, we loop through "fvgs", deleting each rectangle, label (suffixed "\_Label"), and mitigation icon ("\_MitIcon") with [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete), resize "fvgs" to 0, redraw the chart, and log the deinit if "prt". When we compile, we get the following outcome.

![IFVG INITIAL LOAD HISTORICAL SETUP](https://c.mql5.com/2/182/IFVG_INIT_GIF.gif)

From the visualization, we can see that we scan, map, and update all the FVGs on loadup. What we now need to do is continue the detection and setup updates on new bars. Let us start with the detection logic.

```
//+------------------------------------------------------------------+
//| Detect new FVGs on recent bars                                   |
//+------------------------------------------------------------------+
void DetectFVGs() {
   for (int i = 3; i >= 1; i--) {                                 //--- Iterate recent bars
      double low0 = iLow(_Symbol, _Period, i);                    //--- Get low0
      double high2 = iHigh(_Symbol, _Period, i + 2);              //--- Get high2
      double gap_L0_H2 = NormalizeDouble((low0 - high2) / _Point, _Digits); //--- Calc gap L0 H2
      double high0 = iHigh(_Symbol, _Period, i);                  //--- Get high0
      double low2 = iLow(_Symbol, _Period, i + 2);                //--- Get low2
      double gap_H0_L2 = NormalizeDouble((low2 - high0) / _Point, _Digits); //--- Calc gap H0 L2
      bool FVG_UP = low0 > high2 && gap_L0_H2 > minPts;           //--- Check up FVG
      bool FVG_DOWN = low2 > high0 && gap_H0_L2 > minPts;         //--- Check down FVG
      if (FVG_UP || FVG_DOWN) {                                   //--- Check FVG
         datetime time1 = iTime(_Symbol, _Period, i + 1);         //--- Get time1
         double price1 = FVG_UP ? high2 : high0;                  //--- Set price1
         double price2 = FVG_UP ? low0 : low2;                    //--- Set price2
         double newLow = MathMin(price1, price2);                 //--- Calc new low
         double newHigh = MathMax(price1, price2);                //--- Calc new high
         bool overlaps = false;                                   //--- Init overlaps
         if (ignoreOverlaps) {                                    //--- Check ignore overlaps
            for (int ex = 0; ex < ArraySize(fvgs); ex++) {        //--- Iterate existing
               double exLow = MathMin(ObjectGetDouble(0, fvgs[ex].name, OBJPROP_PRICE, 0), ObjectGetDouble(0, fvgs[ex].name, OBJPROP_PRICE, 1)); //--- Calc ex low
               double exHigh = MathMax(ObjectGetDouble(0, fvgs[ex].name, OBJPROP_PRICE, 0), ObjectGetDouble(0, fvgs[ex].name, OBJPROP_PRICE, 1)); //--- Calc ex high
               if (MathMax(newLow, exLow) < MathMin(newHigh, exHigh)) { //--- Check overlap
                  overlaps = true;                                //--- Set overlaps
                  if (prt) Print("Detect: Skipping overlapping FVG at ", TimeToString(time1)); //--- Log skip
                  break;                                          //--- Break loop
               }
            }
         }
         if (overlaps) continue;                                  //--- Continue if overlaps
         string fvgNAME = FVG_Prefix + "(" + TimeToString(time1) + ")"; //--- FVG name
         if (ObjectFind(0, fvgNAME) >= 0) continue;               //--- Skip duplicate
         color fvgClr = FVG_UP ? CLR_UP : CLR_DOWN;               //--- Set color
         datetime endTime = time1 + PeriodSeconds(_Period) * FVG_Rec_Ext_Bars; //--- Calc end time
         CreateRec(fvgNAME, time1, price1, endTime, price2, fvgClr); //--- Create rec
         int size = ArraySize(fvgs);                              //--- Get size
         if (size >= maxFVGs) {                                   //--- Check max
            if (prt) Print("Detect: Max FVGs reached, removing oldest."); //--- Log max
            ArrayRemove(fvgs, 0, 1);                              //--- Remove oldest
            PrintFVGs();                                          //--- Print FVGs
         }
         ArrayResize(fvgs, size + 1);                             //--- Resize array
         fvgs[size].name = fvgNAME;                               //--- Set name
         fvgs[size].startTime = time1;                            //--- Set start time
         fvgs[size].origEndTime = endTime;                        //--- Set end time
         fvgs[size].mitTime = 0;                                  //--- Set mit time
         fvgs[size].signal = false;                               //--- Set signal
         fvgs[size].inverted = false;                             //--- Set inverted
         fvgs[size].mit = false;                                  //--- Set mit
         fvgs[size].ret = false;                                  //--- Set ret
         fvgs[size].origUp = FVG_UP;                              //--- Set orig up
         fvgs[size].tradeCount = 0;                               //--- Set trade count
         fvgs[size].state = Normal;                               //--- Set state
         fvgs[size].newSignal = false;                            //--- Set new signal
         if (prt) Print("New FVG added to storage: ", fvgNAME, " origUp=", FVG_UP, " endTime=", TimeToString(endTime)); //--- Log added
         PrintFVGs();                                             //--- Print FVGs
      }
   }
}
```

We define the "DetectFVGs" function to scan the most recent bars for new fair value gaps on every new bar, adding them to our tracking system if they meet criteria. We loop from shift 3 down to 1 to check the last few completed bars (this is not new now): for each i, we fetch low of i with [iLow](https://www.mql5.com/en/docs/series/ilow) into "low0", high of i+2 into "high2", and calculate the normalized gap in points into "gap\_L0\_H2"; similarly for high of i into "high0", low of i+2 into "low2", and "gap\_H0\_L2". We set "FVG\_UP" true if "low0 > high2" and gap exceeds "minPts" (bullish gap), or "FVG\_DOWN" if "low2 > high0" and gap > "minPts" (bearish gap).

If either is detected, we get the time of i+1 into "time1", set "price1" to "high2" for up or "high0" for down, "price2" to "low0" or "low2", and calculate the gap's low and high. If "ignoreOverlaps" is true, we check against all existing in "fvgs" by fetching their prices with [ObjectGetDouble](https://www.mql5.com/en/docs/objects/objectgetdouble) and using [MathMin](https://www.mql5.com/en/docs/math/mathmin)/ [MathMax](https://www.mql5.com/en/docs/math/mathmax) to get ranges — if a new gap overlaps any (max lows < min highs), we set "overlaps" true, log skip if "prt", and continue. If no overlap and no duplicate object per [ObjectFind](https://www.mql5.com/en/docs/objects/ObjectFind), we form the name as "FVG\_Prefix + (TimeToString(time1))", choose color based on up (green) or down (red), calculate "endTime" as "time1 + [PeriodSeconds](https://www.mql5.com/en/docs/common/periodseconds)(\_Period) \* FVG\_Rec\_Ext\_Bars", and call "CreateRec" to draw the rectangle. We then manage the "fvgs" array: if size reaches "maxFVGs", remove the oldest with [ArrayRemove](https://www.mql5.com/en/docs/array/arrayremove) and log if "prt", then resize up by one with [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize), populate the new entry with name, times, mitigation time 0, all flags false except "origUp" as "FVG\_UP", trade count 0, state "Normal", new signal false, log addition if "prt", and call "PrintFVGs". This detects and stores only valid, non-overlapping new FVGs extending rightward for visibility. We can call this function in the tick event handler to do the heavy lifting as below.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   static datetime lastBarTime = 0;                               //--- Last bar time
   datetime curBarTime = iTime(_Symbol, _Period, 0);              //--- Current bar time
   bool newBar = (curBarTime != lastBarTime);                     //--- Check new bar
   if (!newBar) return;                                           //--- Return if not new
   lastBarTime = curBarTime;                                      //--- Update last time
   DetectFVGs();                                                  //--- Detect FVGs
}
```

In the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler, which executes on every price tick to handle real-time updates, we use a static "lastBarTime" to track the previous bar's open time, fetch the current bar's time with [iTime](https://www.mql5.com/en/docs/series/itime) at shift 0 into "curBarTime", and set "newBar" to true if they differ, indicating a new bar has formed. If not a new bar, we return early to avoid redundant processing. Otherwise, we update "lastBarTime" to "curBarTime" and call "DetectFVGs" to scan for new gaps on recent bars. We get the following outcome.

![INITIAL ONTICK IFVGS DETECTION](https://c.mql5.com/2/182/IFVG_TICK_1.gif)

With the initial detection handled, we can proceed to updating the setups. We use the following logic.

```
//+------------------------------------------------------------------+
//| Update states for all FVGs                                       |
//+------------------------------------------------------------------+
void UpdateFVGs() {
   double prevClose = iClose(_Symbol, _Period, 1);                //--- Get prev close
   double prevLow = iLow(_Symbol, _Period, 1);                    //--- Get prev low
   double prevHigh = iHigh(_Symbol, _Period, 1);                  //--- Get prev high
   double bar2Close = iClose(_Symbol, _Period, 2);                //--- Get bar2 close
   datetime curBarTime = iTime(_Symbol, _Period, 1);              //--- Get prev bar time
   bool removed = false;                                          //--- Init removed
   for (int j = ArraySize(fvgs) - 1; j >= 0; j--) {               //--- Iterate reverse
      if (ObjectFind(0, fvgs[j].name) < 0) {                      //--- Check no object
         if (prt) Print("Update: Removed non-existent FVG from storage: ", fvgs[j].name); //--- Log removed
         ArrayRemove(fvgs, j, 1);                                 //--- Remove from array
         removed = true;                                          //--- Set removed
         continue;                                                //--- Continue
      }
      double fvgLow = MathMin(ObjectGetDouble(0, fvgs[j].name, OBJPROP_PRICE, 0), ObjectGetDouble(0, fvgs[j].name, OBJPROP_PRICE, 1)); //--- Calc low
      double fvgHigh = MathMax(ObjectGetDouble(0, fvgs[j].name, OBJPROP_PRICE, 0), ObjectGetDouble(0, fvgs[j].name, OBJPROP_PRICE, 1)); //--- Calc high
      if (!fvgs[j].mit) {                                         //--- Check not mit
         bool breakFar = (fvgs[j].origUp && prevLow < fvgLow) || (!fvgs[j].origUp && prevHigh > fvgHigh); //--- Check break far
         if (breakFar) {                                          //--- Break far
            fvgs[j].mit = true;                                   //--- Set mit
            fvgs[j].mitTime = curBarTime;                         //--- Set mit time
            fvgs[j].state = Mitigated;                            //--- Set state
            if (prt) Print("Mitigated FVG: ", fvgs[j].name, " at time=", TimeToString(curBarTime)); //--- Log mitigated
            color mitClr = GetFVGColor(fvgs[j].origUp, fvgs[j].state); //--- Get color
            UpdateRec(fvgs[j].name, fvgs[j].startTime, fvgLow, fvgs[j].origEndTime, fvgHigh, mitClr); //--- Update rec
            DrawMitIcon(fvgs[j].name, curBarTime, fvgHigh, fvgLow, fvgs[j].origUp); //--- Draw icon
         }
      }
      if (fvgs[j].mit && !fvgs[j].ret) {                          //--- Check mit not ret
         bool inside = (prevHigh > fvgLow && prevLow < fvgHigh);  //--- Check inside
         if (inside) {                                            //--- Inside
            fvgs[j].ret = true;                                   //--- Set ret
            if (prt) Print("Retraced into FVG: ", fvgs[j].name);  //--- Log retraced
         }
      }
      if (fvgs[j].mit && fvgs[j].ret) {                           //--- Check mit ret
         bool signal = (fvgs[j].origUp && prevClose < fvgLow) || (!fvgs[j].origUp && prevClose > fvgHigh); //--- Check signal
         bool prevInside = (bar2Close > fvgLow && bar2Close < fvgHigh); //--- Check prev inside
         if (signal && curBarTime != fvgs[j].mitTime && prevInside) { //--- Check signal conditions
            fvgs[j].newSignal = true;                             //--- Set new signal
            if (!fvgs[j].inverted) {                              //--- Check not inverted
               fvgs[j].inverted = true;                           //--- Set inverted
               fvgs[j].state = Inverted;                          //--- Set state
               if (prt) Print("Signal/Inverted FVG: ", fvgs[j].name, " at time=", TimeToString(curBarTime)); //--- Log signal
               color sigClr = GetFVGColor(fvgs[j].origUp, fvgs[j].state); //--- Get color
               UpdateRec(fvgs[j].name, fvgs[j].startTime, fvgLow, fvgs[j].origEndTime, fvgHigh, sigClr); //--- Update rec
            }
         }
      }
   }
   if (removed) PrintFVGs();                                      //--- Print if removed
}
```

Here, we define the "UpdateFVGs" function to refresh the states of all tracked fair value gaps on each new bar, using the previous bar's data to detect mitigation, retracement, and inversion in real time. We start by fetching the previous bar's close with [iClose](https://www.mql5.com/en/docs/series/iclose) at shift 1 into "prevClose", its low with [iLow](https://www.mql5.com/en/docs/series/ilow) into "prevLow", high with [iHigh](https://www.mql5.com/en/docs/series/ihigh) into "prevHigh", the bar before that's close into "bar2Close" at shift 2, and the previous bar's time with "iTime" at shift 1 into "curBarTime". We initialize a "removed" flag to false, then loop backward through the "fvgs" array to safely remove entries if needed. For each FVG at index j, if the rectangle object is missing per "ObjectFind" returning negative, we log the removal if "prt" is true, delete the entry with [ArrayRemove](https://www.mql5.com/en/docs/array/arrayremove), set "removed" true, and continue to the next. Otherwise, we calculate the current low and high from the object's prices using "MathMin" and "MathMax" on [ObjectGetDouble](https://www.mql5.com/en/docs/objects/objectgetdouble) with [OBJPROP\_PRICE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_double) for anchors 0 and 1.

If not yet mitigated, we check for a far-side break: for original up gaps, if "prevLow" drops below FVG low; for down gaps, if "prevHigh" exceeds FVG high — setting the mitigation flag true, "mitTime" to "curBarTime", state to "Mitigated", logging if "prt", getting the new color with "GetFVGColor", updating the rectangle via "UpdateRec" with adjusted coordinates and color, and drawing the mitigation icon with "DrawMitIcon" at "curBarTime" and the far edge (high for down gaps, low for up). If mitigated but not retraced, we verify if the previous bar overlaps the FVG (high above low and low below high), setting the retracement flag true and logging.

If both mitigated and retraced, we detect inversion: for up gaps, if "prevClose" is below FVG low; for down gaps, above FVG high — also ensuring it's not the mitigation bar itself and that "bar2Close" was inside the FVG (above low and below high) to confirm from-inside exit. We set "newSignal" true, and if not already inverted, set inverted true, state to "Inverted", log, get inversion color, and update the rectangle. Finally, if any removals occurred, we call "PrintFVGs" for debugging. This keeps all FVGs' states current, enabling accurate inversion signals for trading while handling orphaned objects gracefully. We get the following outcome when we call the function.

![UPDATED IFVG SETUPS GIF](https://c.mql5.com/2/182/IFVG_TICK_2.gif)

We can see that we update the setups when prices interact with them. What now remains is trading the inversion setups, and that will be all. Here is the logic we implemented to achieve that in a function to maintain modularization.

```
//+------------------------------------------------------------------+
//| Trade on FVGs with signals                                       |
//+------------------------------------------------------------------+
void TradeOnFVGs() {
   double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK), _Digits); //--- Get ask
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID), _Digits); //--- Get bid
   for (int j = 0; j < ArraySize(fvgs); j++) {                    //--- Iterate FVGs
      if (!fvgs[j].newSignal || fvgs[j].mitTime == 0) continue;   //--- Skip no signal or no mit
      if (tradeMode == TradeOnce && fvgs[j].tradeCount >= 1) {    //--- Check once and traded
         fvgs[j].newSignal = false;                               //--- Reset signal
         continue;                                                //--- Continue
      }
      if (tradeMode == LimitedTrades && fvgs[j].tradeCount >= maxTradesPerFVG) { //--- Check limited and max
         fvgs[j].newSignal = false;                               //--- Reset signal
         continue;                                                //--- Continue
      }
      double fvgLow = MathMin(ObjectGetDouble(0, fvgs[j].name, OBJPROP_PRICE, 0), ObjectGetDouble(0, fvgs[j].name, OBJPROP_PRICE, 1)); //--- Calc low
      double fvgHigh = MathMax(ObjectGetDouble(0, fvgs[j].name, OBJPROP_PRICE, 0), ObjectGetDouble(0, fvgs[j].name, OBJPROP_PRICE, 1)); //--- Calc high
      if (!fvgs[j].origUp) {                                      //--- Check orig down: Bullish IFVG, Buy
         if (prt) Print("BULLISH IFVG TRADE SIGNAL For ", fvgs[j].name, " at ", Bid); //--- Log buy signal
         double SL_Buy = NormalizeDouble(fvgLow - sl_pts * _Point, _Digits); //--- Calc buy SL
         double TP_Buy = NormalizeDouble(Ask + tp_pts * _Point, _Digits); //--- Calc buy TP
         obj_Trade.Buy(inpLot, _Symbol, Ask, SL_Buy, TP_Buy, "IFVG Buy"); //--- Open buy
      } else {                                                    //--- Orig up: Bearish IFVG, Sell
         if (prt) Print("BEARISH IFVG TRADE SIGNAL For ", fvgs[j].name, " at ", Ask); //--- Log sell signal
         double SL_Sell = NormalizeDouble(fvgHigh + sl_pts * _Point, _Digits); //--- Calc sell SL
         double TP_Sell = NormalizeDouble(Bid - tp_pts * _Point, _Digits); //--- Calc sell TP
         obj_Trade.Sell(inpLot, _Symbol, Bid, SL_Sell, TP_Sell, "IFVG Sell"); //--- Open sell
      }
      fvgs[j].tradeCount++;                                       //--- Increment count
      fvgs[j].newSignal = false;                                  //--- Reset signal
      fvgs[j].ret = false;                                        //--- Reset ret
      if (prt) Print("Trade executed on ", fvgs[j].name, ", tradeCount now=", fvgs[j].tradeCount); //--- Log executed
      double midPrice = (fvgLow + fvgHigh) / 2;                   //--- Calc mid price
      datetime midTime = fvgs[j].startTime + (fvgs[j].origEndTime - fvgs[j].startTime) / 2; //--- Calc mid time
      UpdateLabel(fvgs[j].name, midTime, midPrice);               //--- Update label
   }
}

//+------------------------------------------------------------------+
//| Cleanup expired FVGs from array (keep on chart)                  |
//+------------------------------------------------------------------+
void CleanupExpiredFVGs(datetime curBarTime) {
   bool removed = false;                                          //--- Init removed
   for (int j = ArraySize(fvgs) - 1; j >= 0; j--) {               //--- Iterate reverse
      if (curBarTime > fvgs[j].origEndTime) {                     //--- Check expired
         if (prt) Print("Expired FVG removed from storage (kept on chart): ", fvgs[j].name, " endTime=", TimeToString(fvgs[j].origEndTime)); //--- Log expired
         ArrayRemove(fvgs, j, 1);                                 //--- Remove from array
         removed = true;                                          //--- Set removed
      }
   }
   if (removed) PrintFVGs();                                      //--- Print if removed
}
```

First, we define the "TradeOnFVGs" function to execute trades on fresh inversion signals from the FVGs, respecting the configured trade modes and limits. We first retrieve the current ask price with [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) using [SYMBOL\_ASK](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double) and normalize it to digits into "Ask", doing the same for bid with [SYMBOL\_BID](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double) into "Bid". We then loop through the "fvgs" array: for each entry, we skip if no new signal or mitigation time is zero. We check trade modes — for "TradeOnce", if trade count is 1 or more, or for "LimitedTrades", if at or above "maxTradesPerFVG", we reset the new signal flag and continue.

For valid signals, we calculate the FVG's low and high from the rectangle's prices using [MathMin](https://www.mql5.com/en/docs/math/mathmin) and [MathMax](https://www.mql5.com/en/docs/math/mathmax) on the [ObjectGetDouble](https://www.mql5.com/en/docs/objects/objectgetdouble) function. If not original up (bearish FVG, inverting bullish), we log the buy signal if "prt" is true, set stop-loss below low minus "sl\_pts \* [\_Point](https://www.mql5.com/en/docs/predefined/_point)" normalized, take-profit above ask plus "tp\_pts \* \_Point" normalized, and open a buy with "obj\_Trade.Buy" using "inpLot", symbol, "Ask", calculated levels, and comment "IFVG Buy". Same logic for a bearish one. We increment the trade count, reset new signal and retracement flags, log the execution with current count if "prt", calculate midpoint price and time, and call "UpdateLabel" to refresh the label position.

Finally, we implement the "CleanupExpiredFVGs" function to remove outdated FVGs from the array while keeping their visuals on the chart, called with the previous bar's time. We initialize a removed flag to false, then loop backward through "fvgs": if the provided time exceeds the original end time, we log the expiration, remove the entry with [ArrayRemove](https://www.mql5.com/en/docs/array/arrayremove), and set removed to true. If any removals happened, we call "PrintFVGs" for debugging. This is very important to make sure that we track only setups that matter to us. When we call these and run, we get the following outcome.

![TRADE EXECUTIONS](https://c.mql5.com/2/182/Screenshot_2025-11-20_121526.png)

Since we can act on the signals generated and open positions, we just need to manage the trades by applying a trailing stop. Here is the logic we use to achieve that.

```
//+------------------------------------------------------------------+
//| Apply Points Trailing Stop                                       |
//+------------------------------------------------------------------+
void ApplyPointsTrailing() {
   double point = _Point;                                            //--- Get point value
   for (int i = PositionsTotal() - 1; i >= 0; i--) {                 //--- Iterate positions reverse
      if (PositionGetTicket(i) > 0) {                                //--- Check valid ticket
         if (PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == magic_number) { //--- Check symbol and magic
            double sl = PositionGetDouble(POSITION_SL);              //--- Get SL
            double tp = PositionGetDouble(POSITION_TP);              //--- Get TP
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN); //--- Get open price
            ulong ticket = PositionGetInteger(POSITION_TICKET);      //--- Get ticket
            if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) { //--- Check buy
               double newSL = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID) - Trailing_Stop_Pips * point, _Digits); //--- Calc new SL
               if (newSL > sl && SymbolInfoDouble(_Symbol, SYMBOL_BID) - openPrice > Min_Profit_To_Trail_Pips * point) { //--- Check conditions
                  obj_Trade.PositionModify(ticket, newSL, tp);       //--- Modify position
               }
            } else if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) { //--- Check sell
               double newSL = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK) + Trailing_Stop_Pips * point, _Digits); //--- Calc new SL
               if (newSL < sl && openPrice - SymbolInfoDouble(_Symbol, SYMBOL_ASK) > Min_Profit_To_Trail_Pips * point) { //--- Check conditions
                  obj_Trade.PositionModify(ticket, newSL, tp);       //--- Modify position
               }
            }
         }
      }
   }
}

//--- Call the function in the tick event handler per tick
if (TrailingType == Trailing_Points && PositionsTotal() > 0) { //--- Check trailing
   ApplyPointsTrailing();                                      //--- Apply trailing
}
```

Here, we define the "ApplyPointsTrailing" function to implement points-based trailing stops when selected, adjusting stop-loss levels in real time as price moves profitably. We start by assigning the symbol's point value to "point" using [\_Point](https://www.mql5.com/en/docs/predefined/_point). We then loop backward through all open positions with [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) to avoid index issues during modifications, checking each ticket's validity via the [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket) function. For positions matching our symbol and "magic\_number", we retrieve stop-loss with [PositionGetDouble](https://www.mql5.com/en/docs/trading/positiongetdouble) and [POSITION\_SL](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_double), take-profit with "POSITION\_TP", open price via "POSITION\_PRICE\_OPEN", and ticket with "POSITION\_TICKET". For buy positions ( [POSITION\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type)), we calculate a new stop-loss as the current bid minus "Trailing\_Stop\_Pips \* point", normalized to digits — if this is tighter than the existing stop-loss and unrealized profit exceeds "Min\_Profit\_To\_Trail\_Pips \* point", we update the position using "obj\_Trade.PositionModify" while keeping take-profit unchanged. We apply similar logic for sell positions: new stop-loss as ask plus trailing distance, modified only if it tightens and profit meets the threshold.

Within "OnTick", if "TrailingType" is "Trailing\_Points" and positions exist per "PositionsTotal", we call "ApplyPointsTrailing" on every tick to ensure timely adjustments. Upon compilation, we get the following outcome.

![IFVG FINAL TESTING GIF](https://c.mql5.com/2/182/IFVG_FINAL_GIF.gif)

From the visualization, we can see that we detect, trade, and manage the inverse fair value gap setups, hence achieving our objectives. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/182/Screenshot_2025-11-20_124301.png)

Backtest report:

![REPORT](https://c.mql5.com/2/182/Screenshot_2025-11-20_124312.png)

### Conclusion

In conclusion, we’ve developed an [Inverse Fair Value Gap (IFVG)](https://www.mql5.com/go?link=https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/Inversion-Fair-Value-Gaps "https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/Inversion-Fair-Value-Gaps") system in [MQL5](https://www.mql5.com/) that detects bullish/bearish [Fair Value Gaps (FVGs)](https://www.mql5.com/go?link=https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/Fair-Value-Gaps "https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/Fair-Value-Gaps") on recent bars with minimum gap filtering, tracks states as normal/mitigated/inverted based on price interactions, ignores overlaps while limiting tracked FVGs, and loads historical FVGs on initialization with real-time updates and expired cleanup. The system supports once/limited/unlimited trades per setup, opens buys on bullish IFVGs or sells on bearish with fixed trade levels, position limits, trailing stops, and visualizes colored rectangles with state/trade labels and mitigation icons.

Disclaimer: This article is for educational purposes only. Trading carries significant financial risks, and market volatility may result in losses. Thorough backtesting and careful risk management are crucial before deploying this program in live markets.

With this Inverse Fair Value Gap strategy offering state tracking and inversion signals, you’re equipped to trade gap imbalances, ready for further optimization in your trading journey. Happy trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20361.zip "Download all attachments in the single ZIP archive")

[FVG\_Inverse.mq5](https://www.mql5.com/en/articles/download/20361/FVG_Inverse.mq5 "Download FVG_Inverse.mq5")(81.62 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/501475)**

![The View and Controller components for tables in the MQL5 MVC paradigm: Containers](https://c.mql5.com/2/155/18658-komponenti-view-i-controller-logo.png)[The View and Controller components for tables in the MQL5 MVC paradigm: Containers](https://www.mql5.com/en/articles/18658)

In this article, we will discuss creating a "Container" control that supports scrolling its contents. Within the process, the already implemented classes of graphics library controls will be improved.

![Chaos Game Optimization (CGO)](https://c.mql5.com/2/122/Chaos_Game_Optimization___LOGO.png)[Chaos Game Optimization (CGO)](https://www.mql5.com/en/articles/17047)

The article presents a new metaheuristic algorithm, Chaos Game Optimization (CGO), which demonstrates a unique ability to maintain high efficiency when dealing with high-dimensional problems. Unlike most optimization algorithms, CGO not only does not lose, but sometimes even increases performance when scaling a problem, which is its key feature.

![Fortified Profit Architecture: Multi-Layered Account Protection](https://c.mql5.com/2/184/20449-fortified-profit-architecture-logo.png)[Fortified Profit Architecture: Multi-Layered Account Protection](https://www.mql5.com/en/articles/20449)

In this discussion, we introduce a structured, multi-layered defense system designed to pursue aggressive profit targets while minimizing exposure to catastrophic loss. The focus is on blending offensive trading logic with protective safeguards at every level of the trading pipeline. The idea is to engineer an EA that behaves like a “risk-aware predator”—capable of capturing high-value opportunities, but always with layers of insulation that prevent blindness to sudden market stress.

![From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://c.mql5.com/2/184/20417-from-novice-to-expert-developing-logo.png)[From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)

Trading without session awareness is like navigating without a compass—you're moving, but not with purpose. Today, we're revolutionizing how traders perceive market timing by transforming ordinary charts into dynamic geographical displays. Using MQL5's powerful visualization capabilities, we'll build a live world map that illuminates active trading sessions in real-time, turning abstract market hours into intuitive visual intelligence. This journey sharpens your trading psychology and reveals professional-grade programming techniques that bridge the gap between complex market structure and practical, actionable insight.

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/20361&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068740641527102742)

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