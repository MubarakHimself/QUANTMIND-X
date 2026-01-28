---
title: MQL5 Trading Tools (Part 6): Dynamic Holographic Dashboard with Pulse Animations and Controls
url: https://www.mql5.com/en/articles/18880
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T17:53:44.316658
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=glzzsmccgtaxktcvpcyycqnlzcrcvhfx&ssn=1769180022315610424&ssn_dr=0&ssn_sr=0&fv_date=1769180022&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18880&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Trading%20Tools%20(Part%206)%3A%20Dynamic%20Holographic%20Dashboard%20with%20Pulse%20Animations%20and%20Controls%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918002268788546&fz_uniq=5068770212376935798&sv=2552)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 5)](https://www.mql5.com/en/articles/18844), we created a [rolling ticker tape](https://en.wikipedia.org/wiki/Ticker_tape "https://en.wikipedia.org/wiki/Ticker_tape") in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) for real-time symbol monitoring with scrolling prices, spreads, and changes to keep traders informed efficiently. In Part 6, we develop a Dynamic Holographic Dashboard that displays multi-symbol and timeframe indicators, such as the [Relative Strength Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi") (RSI) and volatility (based on the [Average True Range](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/atr "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/atr") (ATR)), with pulse animations, sorting, and interactive controls, creating an engaging analysis tool. We will cover the following topics:

1. [Understanding the Holographic Dashboard Architecture](https://www.mql5.com/en/articles/18880#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/18880#para2)
3. [Backtesting](https://www.mql5.com/en/articles/18880#para3)
4. [Conclusion](https://www.mql5.com/en/articles/18880#para4)

By the end, you’ll have a customizable holographic dashboard ready for your trading setup—let’s get started!

### Understanding the Holographic Dashboard Architecture

The holographic dashboard we’re building is a visual tool that monitors multiple symbols and timeframes, displaying indicators such as [RSI](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi") and volatility, along with sorting and alerts, to help us quickly spot opportunities. This architecture is important because it combines real-time data with interactive controls and animations, making analysis more engaging and efficient in a cluttered chart environment.

We will achieve this by using arrays for data management, handles for indicators like [ATR](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/atr "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/atr") and RSI, and functions for sorting and pulsing effects, with buttons for toggling visibility and switching views. We plan to centralize updates in a loop that refreshes the [User Interface](https://en.wikipedia.org/wiki/User_interface "https://en.wikipedia.org/wiki/User_interface") (UI) dynamically, ensuring the dashboard remains adaptable and responsive for strategic trading. View the visualization below to get what we want to achieve before we proceed to the implementation.

![COMPLETE ARCHITECTURE](https://c.mql5.com/2/157/HOLOGRAPH_PLAN.gif)

### Implementation in MQL5

To create the program in [MQL5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5"), we will need to define the program metadata and then define some [inputs](https://www.mql5.com/en/docs/basis/variables/inputvariables) that will enable us to easily modify the functioning of the program without interfering with the code directly.

```
//+------------------------------------------------------------------+
//|                                  Holographic Dashboard EA.mq5    |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//|                                   https://t.me/Forex_Algo_Trader |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Allan Munene Mutiiria."
#property link      "https://t.me/Forex_Algo_Trader"
#property version   "1.00"

#include <Arrays\ArrayString.mqh> //--- Include ArrayString library for string array operations
#include <Files\FileTxt.mqh>      //--- Include FileTxt library for text file operations

// Input Parameters
input int BaseFontSize = 9;           // Base Font Size
input string FontType = "Calibri";    // Font Type (Professional)
input int X_Offset = 30;              // X Offset
input int Y_Offset = 30;              // Y Offset
input color PanelColor = clrDarkSlateGray; // Panel Background Color
input color TitleColor = clrWhite;    // Title/Symbol Color
input color DataColor = clrLightGray; // Bid/Neutral Color
input color ActiveColor = clrLime;    // Active Timeframe/Symbol Color
input color UpColor = clrDeepSkyBlue; // Uptrend Color
input color DownColor = clrCrimson;   // Downtrend Color
input color LineColor = clrSilver;    // Grid Line Color
input bool EnableAnimations = true;   // Enable Pulse Animations
input int PanelWidth = 730;           // Panel Width (px)
input int ATR_Period = 14;            // ATR Period for Volatility
input int RSI_Period = 14;            // RSI Period
input double Vol_Alert_Threshold = 2.0; // Volatility Alert Threshold (%)
input color GlowColor = clrDodgerBlue; // Glow Color for holographic effect
input int AnimationSpeed = 30; // Animation delay in ms for pulse
input int PulseCycles = 3; // Number of pulse cycles for animations
```

We start by including libraries for string arrays and text file logging, and setting up inputs to customize the UI and indicators. We include "<Arrays\\ArrayString.mqh>" for handling symbol lists and "<Files\\FileTxt.mqh>" for logging errors to a file. The inputs will allow us to adjust the base font size to 9, choose a professional font like Calibri, set offsets for x and y positioning at 30 pixels each, and select colors such as dark slate gray for the panel background, white for titles, light gray for data, lime for active elements, deep sky blue for uptrends, crimson for downtrends, and silver for grid lines.

We enable pulse animations by default, define the panel width at 730 pixels, set ATR and RSI periods to 14 for volatility and momentum calculations, establish a 2.0% threshold for volatility alerts, choose Dodger Blue for the holographic glow, and configure animation speed at 30 ms with 3 pulse cycles. These settings will make the dashboard highly adaptable for visual and functional preferences. We then need to define some global variables that we will use throughout the program.

```
// Global Variables
double prices_PrevArray[];            //--- Array for previous prices
double volatility_Array[];            //--- Array for volatility values
double bid_array[];                   //--- Array for bid prices
long spread_array[];                  //--- Array for spreads
double change_array[];                //--- Array for percentage changes
double vol_array[];                   //--- Array for volumes
double rsi_array[];                   //--- Array for RSI values
int indices[];                        //--- Array for sorted indices
ENUM_TIMEFRAMES periods[] = {PERIOD_M1, PERIOD_M5, PERIOD_H1, PERIOD_H2, PERIOD_H4, PERIOD_D1, PERIOD_W1}; //--- Array of timeframes
string logFileName = "Holographic_Dashboard_Log.txt"; //--- Log file name
int sortMode = 0;                     //--- Current sort mode
string sortNames[] = {"Name ASC", "Vol DESC", "Change ABS DESC", "RSI DESC"}; //--- Sort mode names
int atr_handles_sym[];                //--- ATR handles for symbols
int rsi_handles_sym[];                //--- RSI handles for symbols
int atr_handles_tf[];                 //--- ATR handles for timeframes
int rsi_handles_tf[];                 //--- RSI handles for timeframes
int totalSymbols;                     //--- Total number of symbols
bool dashboardVisible = true;         //--- Dashboard visibility flag
```

Here, we define [global variables](https://www.mql5.com/en/docs/basis/variables/global) to manage data and indicators in our program, supporting real-time monitoring, sorting, and animations. We create arrays like "prices\_PrevArray" for previous prices to calculate changes, "volatility\_Array" for volatility values, "bid\_array" for current bids, "spread\_array" for spreads as longs, "change\_array" for percentage changes, "vol\_array" for volumes, "rsi\_array" for RSI values, and "indices" for sorting indices. We set "periods" as an array of timeframes from [PERIOD\_M1](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes) to PERIOD\_W1, "logFileName" to "Holographic\_Dashboard\_Log.txt" for error logging, "sortMode" to 0 for initial sorting, "sortNames" as strings for sort options like "Name ASC" or "Vol DESC", and arrays for ATR and RSI handles ("atr\_handles\_sym", "rsi\_handles\_sym" for symbols, "atr\_handles\_tf", "rsi\_handles\_tf" for timeframes).

The "totalSymbols" integer tracks the number of symbols, and "dashboardVisible" is true, which controls the dashboard's state. To better manage objects, we will create a class.

```
// Object Manager Class
class CObjectManager : public CArrayString {
public:
   void AddObject(string name) {      //--- Add object name to manager
      if (!Add(name)) {               //--- Check if add failed
         LogError(__FUNCTION__ + ": Failed to add object name: " + name); //--- Log error
      }
   }

   void DeleteAllObjects() {          //--- Delete all managed objects
      for (int i = Total() - 1; i >= 0; i--) { //--- Iterate through objects
         string name = At(i);         //--- Get object name
         if (ObjectFind(0, name) >= 0) { //--- Check if object exists
            if (!ObjectDelete(0, name)) { //--- Delete object
               LogError(__FUNCTION__ + ": Failed to delete object: " + name + ", Error: " + IntegerToString(GetLastError())); //--- Log deletion failure
            }
         }
         Delete(i);                   //--- Remove from array
      }
      ChartRedraw(0);                 //--- Redraw chart
   }
};
```

To manage the dashboard objects efficiently, we create the "CObjectManager" class, extending the [CArrayString](https://www.mql5.com/en/docs/standardlibrary/datastructures/CArrayString) class. In the "AddObject" method, we add the object "name" to the array with "Add", logging failures via "LogError" if unsuccessful. We use the "DeleteAllObjects" method looping backward through the array with "Total", get each "name" with "At", check existence with the [ObjectFind](https://www.mql5.com/en/docs/objects/ObjectFind) function, delete with [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete) and log errors if failed, remove it from the array with "Delete", and redraw the chart with the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function. With the class extension, we can create some helper functions that we will call throughout the program to reuse.

```
CObjectManager objManager;            //--- Object manager instance

//+------------------------------------------------------------------+
//| Utility Functions                                                |
//+------------------------------------------------------------------+
void LogError(string message) {
   CFileTxt file;                     //--- Create file object
   if (file.Open(logFileName, FILE_WRITE | FILE_TXT | FILE_COMMON, true) >= 0) { //--- Open log file
      file.WriteString(message + "\n"); //--- Write message
      file.Close();                   //--- Close file
   }
   Print(message);                    //--- Print message
}

string Ask(string symbol) {
   double value;                      //--- Variable for ask price
   if (SymbolInfoDouble(symbol, SYMBOL_ASK, value)) { //--- Get ask price
      return DoubleToString(value, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)); //--- Return formatted ask
   }
   LogError(__FUNCTION__ + ": Failed to get ask price for " + symbol + ", Error: " + IntegerToString(GetLastError())); //--- Log error
   return "N/A";                      //--- Return N/A on failure
}

string Bid(string symbol) {
   double value;                      //--- Variable for bid price
   if (SymbolInfoDouble(symbol, SYMBOL_BID, value)) { //--- Get bid price
      return DoubleToString(value, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)); //--- Return formatted bid
   }
   LogError(__FUNCTION__ + ": Failed to get bid price for " + symbol + ", Error: " + IntegerToString(GetLastError())); //--- Log error
   return "N/A";                      //--- Return N/A on failure
}

string Spread(string symbol) {
   long value;                        //--- Variable for spread
   if (SymbolInfoInteger(symbol, SYMBOL_SPREAD, value)) { //--- Get spread
      return IntegerToString(value);  //--- Return spread as string
   }
   LogError(__FUNCTION__ + ": Failed to get spread for " + symbol + ", Error: " + IntegerToString(GetLastError())); //--- Log error
   return "N/A";                      //--- Return N/A on failure
}

string PercentChange(double current, double previous) {
   if (previous == 0) return "0.00%"; //--- Handle zero previous value
   return StringFormat("%.2f%%", ((current - previous) / previous) * 100); //--- Calculate and format percentage
}

string TruncPeriod(ENUM_TIMEFRAMES period) {
   return StringSubstr(EnumToString(period), 7); //--- Truncate timeframe string
}
```

To manage objects and data, we instantiate "objManager" as a "CObjectManager" to track UI elements. We create the "LogError" function to log errors, opening "logFileName" with "CFileTxt" using " [FILE\_WRITE \| FILE\_TXT \| FILE\_COMMON](https://www.mql5.com/en/docs/constants/io_constants/fileflags)", writing the "message" with "WriteString", closing the file, and printing it. The "Ask" function retrieves the ask price for a "symbol" with [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble), formats it with [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring) using "SymbolInfoInteger" for digits, logs errors with "LogError" if failed, and returns "N/A" on failure. Similarly, the "Bid" function fetches the bid price, formats it, and handles errors.

The "Spread" function gets the spread with "SymbolInfoInteger", returns it as a string with [IntegerToString](https://www.mql5.com/en/docs/convert/integertostring), or "N/A" on failure. The "PercentChange" function calculates the percentage change between "current" and "previous" prices using [StringFormat](https://www.mql5.com/en/docs/convert/stringformat), returning "0.00%" if "previous" is zero. The "TruncPeriod" function truncates [ENUM\_TIMEFRAMES](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes) strings with [StringSubstr](https://www.mql5.com/en/docs/strings/stringsubstr) for concise timeframe display, ensuring we have clean outputs. We can now create the function for the holographic pulses.

```
//+------------------------------------------------------------------+
//| Holographic Animation Function                                   |
//+------------------------------------------------------------------+
void HolographicPulse(string objName, color mainClr, color glowClr) {
   if (!EnableAnimations) return;     //--- Exit if animations disabled
   int cycles = PulseCycles;          //--- Set pulse cycles
   int delay = AnimationSpeed;        //--- Set animation delay
   for (int i = 0; i < cycles; i++) { //--- Iterate through cycles
      ObjectSetInteger(0, objName, OBJPROP_COLOR, glowClr); //--- Set glow color
      ChartRedraw(0);                 //--- Redraw chart
      Sleep(delay);                   //--- Delay
      ObjectSetInteger(0, objName, OBJPROP_COLOR, mainClr); //--- Set main color
      ChartRedraw(0);                 //--- Redraw chart
      Sleep(delay / 2);               //--- Shorter delay
   }
}
```

Here, we implement the "HolographicPulse" function to create a pulse animation effect for dashboard elements. We exit early if "EnableAnimations" is false to skip animations. We set "cycles" to "PulseCycles" and "delay" to "AnimationSpeed", then loop through "cycles" with a for loop. In each iteration, we set the object's color to "glowClr" with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) for [OBJPROP\_COLOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer), redraw the chart with [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw), pause with "Sleep" for "delay", switch back to "mainClr", redraw again, and pause for "delay / 2" for a shorter effect. This will add the holographic pulse to highlight active or alert elements visually. Armed with these functions, we can graduate to creating the core initialization dashboard. For that, we will need some helper functions to keep the program modular.

```
//+------------------------------------------------------------------+
//| Create Text Label Function                                       |
//+------------------------------------------------------------------+
bool createText(string objName, string text, int x, int y, color clrTxt, int fontsize, string font, bool animate = false, double opacity = 1.0) {
   ResetLastError();                  //--- Reset error code
   if (ObjectFind(0, objName) < 0) {  //--- Check if object exists
      if (!ObjectCreate(0, objName, OBJ_LABEL, 0, 0, 0)) { //--- Create label
         LogError(__FUNCTION__ + ": Failed to create label: " + objName + ", Error: " + IntegerToString(GetLastError())); //--- Log error
         return false;                //--- Return failure
      }
      objManager.AddObject(objName);  //--- Add to manager
   }
   ObjectSetInteger(0, objName, OBJPROP_XDISTANCE, x); //--- Set x-coordinate
   ObjectSetInteger(0, objName, OBJPROP_YDISTANCE, y); //--- Set y-coordinate
   ObjectSetInteger(0, objName, OBJPROP_CORNER, CORNER_LEFT_UPPER); //--- Set corner
   ObjectSetString(0, objName, OBJPROP_TEXT, text); //--- Set text
   ObjectSetInteger(0, objName, OBJPROP_COLOR, clrTxt); //--- Set color
   ObjectSetInteger(0, objName, OBJPROP_FONTSIZE, fontsize); //--- Set font size
   ObjectSetString(0, objName, OBJPROP_FONT, font); //--- Set font
   ObjectSetInteger(0, objName, OBJPROP_BACK, false); //--- Set foreground
   ObjectSetInteger(0, objName, OBJPROP_SELECTABLE, false); //--- Disable selection
   ObjectSetInteger(0, objName, OBJPROP_ZORDER, StringFind(objName, "Glow") >= 0 ? -1 : 0); //--- Set z-order

   if (animate && EnableAnimations) { //--- Check for animation
      ObjectSetInteger(0, objName, OBJPROP_COLOR, DataColor); //--- Set temporary color
      ChartRedraw(0);                 //--- Redraw chart
      Sleep(50);                      //--- Delay
      ObjectSetInteger(0, objName, OBJPROP_COLOR, clrTxt); //--- Set final color
   }

   ChartRedraw(0);                    //--- Redraw chart
   return true;                       //--- Return success
}

//+------------------------------------------------------------------+
//| Create Button Function                                           |
//+------------------------------------------------------------------+
bool createButton(string objName, string text, int x, int y, int width, int height, color textColor, color bgColor, color borderColor, bool animate = false) {
   ResetLastError();                  //--- Reset error code
   if (ObjectFind(0, objName) < 0) {  //--- Check if object exists
      if (!ObjectCreate(0, objName, OBJ_BUTTON, 0, 0, 0)) { //--- Create button
         LogError(__FUNCTION__ + ": Failed to create button: " + objName + ", Error: " + IntegerToString(GetLastError())); //--- Log error
         return false;                //--- Return failure
      }
      objManager.AddObject(objName);  //--- Add to manager
   }
   ObjectSetInteger(0, objName, OBJPROP_XDISTANCE, x); //--- Set x-coordinate
   ObjectSetInteger(0, objName, OBJPROP_YDISTANCE, y); //--- Set y-coordinate
   ObjectSetInteger(0, objName, OBJPROP_XSIZE, width); //--- Set width
   ObjectSetInteger(0, objName, OBJPROP_YSIZE, height); //--- Set height
   ObjectSetString(0, objName, OBJPROP_TEXT, text); //--- Set text
   ObjectSetInteger(0, objName, OBJPROP_COLOR, textColor); //--- Set text color
   ObjectSetInteger(0, objName, OBJPROP_BGCOLOR, bgColor); //--- Set background color
   ObjectSetInteger(0, objName, OBJPROP_BORDER_COLOR, borderColor); //--- Set border color
   ObjectSetInteger(0, objName, OBJPROP_FONTSIZE, BaseFontSize + (StringFind(objName, "SwitchTFBtn") >= 0 ? 3 : 0)); //--- Set font size
   ObjectSetString(0, objName, OBJPROP_FONT, FontType); //--- Set font
   ObjectSetInteger(0, objName, OBJPROP_ZORDER, 1); //--- Set z-order
   ObjectSetInteger(0, objName, OBJPROP_STATE, false); //--- Reset state

   if (animate && EnableAnimations) { //--- Check for animation
      ObjectSetInteger(0, objName, OBJPROP_BGCOLOR, clrLightGray); //--- Set temporary background
      ChartRedraw(0);                 //--- Redraw chart
      Sleep(50);                      //--- Delay
      ObjectSetInteger(0, objName, OBJPROP_BGCOLOR, bgColor); //--- Set final background
   }

   ChartRedraw(0);                    //--- Redraw chart
   return true;                       //--- Return success
}

//+------------------------------------------------------------------+
//| Create Panel Function                                            |
//+------------------------------------------------------------------+
bool createPanel(string objName, int x, int y, int width, int height, color clr, double opacity = 1.0) {
   ResetLastError();                  //--- Reset error code
   if (ObjectFind(0, objName) < 0) {  //--- Check if object exists
      if (!ObjectCreate(0, objName, OBJ_RECTANGLE_LABEL, 0, 0, 0)) { //--- Create panel
         LogError(__FUNCTION__ + ": Failed to create panel: " + objName + ", Error: " + IntegerToString(GetLastError())); //--- Log error
         return false;                //--- Return failure
      }
      objManager.AddObject(objName);  //--- Add to manager
   }
   ObjectSetInteger(0, objName, OBJPROP_XDISTANCE, x); //--- Set x-coordinate
   ObjectSetInteger(0, objName, OBJPROP_YDISTANCE, y); //--- Set y-coordinate
   ObjectSetInteger(0, objName, OBJPROP_XSIZE, width); //--- Set width
   ObjectSetInteger(0, objName, OBJPROP_YSIZE, height); //--- Set height
   ObjectSetInteger(0, objName, OBJPROP_BGCOLOR, clr); //--- Set background color
   ObjectSetInteger(0, objName, OBJPROP_BORDER_TYPE, BORDER_FLAT); //--- Set border type
   ObjectSetInteger(0, objName, OBJPROP_ZORDER, -1); //--- Set z-order
   ChartRedraw(0);                    //--- Redraw chart
   return true;                       //--- Return success
}

//+------------------------------------------------------------------+
//| Create Line Function                                             |
//+------------------------------------------------------------------+
bool createLine(string objName, int x1, int y1, int x2, int y2, color clrLine, double opacity = 1.0) {
   ResetLastError();                  //--- Reset error code
   if (ObjectFind(0, objName) < 0) {  //--- Check if object exists
      if (!ObjectCreate(0, objName, OBJ_RECTANGLE_LABEL, 0, 0, 0)) { //--- Create line as rectangle
         LogError(__FUNCTION__ + ": Failed to create line: " + objName + ", Error: " + IntegerToString(GetLastError())); //--- Log error
         return false;                //--- Return failure
      }
      objManager.AddObject(objName);  //--- Add to manager
   }
   ObjectSetInteger(0, objName, OBJPROP_XDISTANCE, x1); //--- Set x1
   ObjectSetInteger(0, objName, OBJPROP_YDISTANCE, y1); //--- Set y1
   ObjectSetInteger(0, objName, OBJPROP_XSIZE, x2 - x1); //--- Set width
   ObjectSetInteger(0, objName, OBJPROP_YSIZE, StringFind(objName, "Glow") >= 0 ? 3 : 1); //--- Set height
   ObjectSetInteger(0, objName, OBJPROP_BGCOLOR, clrLine); //--- Set color
   ObjectSetInteger(0, objName, OBJPROP_ZORDER, StringFind(objName, "Glow") >= 0 ? -1 : 0); //--- Set z-order
   ChartRedraw(0);                    //--- Redraw chart
   return true;                       //--- Return success
}
```

Here, we define the "createText" function to generate a text label. We start by calling [ResetLastError](https://www.mql5.com/en/docs/common/ResetLastError) to clear any prior error. If the object doesn't exist (checked via "ObjectFind(0, objName) < 0"), we create it using [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) with type [OBJ\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object). On failure, we log the error and return false. We add it to "objManager" via "AddObject". We set properties: [OBJPROP\_XDISTANCE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to "x", "OBJPROP\_YDISTANCE" to "y", and the same to the others. If "animate" and "EnableAnimations" are true, we temporarily set "OBJPROP\_COLOR" to "DataColor", redraw, delay via "Sleep(50)", then set to the color of the text. Finally, redraw and return true.

Next, we define "createButton" similarly: reset error, check existence, create with [OBJ\_BUTTON](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) if needed, log on failure, and add to manager. We then set the object properties and, if animations are enabled, temporarily set [OBJPROP\_BGCOLOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to "clrLightGray", redraw, sleep 50ms, then set to "bgColor". Redraw and return true. For "createPanel", we use a similar approach.

Lastly, "createLine" uses a similar pattern: reset, check, create as [OBJ\_RECTANGLE\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) (simulating a line), log on failure, and add to the manager. Set "OBJPROP\_XDISTANCE" to "x1", "OBJPROP\_YDISTANCE" to "y1", "OBJPROP\_XSIZE" to "x2-x1", "OBJPROP\_YSIZE" to 3 if "Glow" in name else 1, "OBJPROP\_BGCOLOR" to "clrLine", [OBJPROP\_ZORDER](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to -1 if "Glow" else 0. Redraw and return true. We now use these functions to create the core function that will enable us to craft the main dashboard as follows.

```
//+------------------------------------------------------------------+
//| Dashboard Creation Function with Holographic Effects             |
//+------------------------------------------------------------------+
void InitDashboard() {
   // Get chart dimensions
   long chartWidth, chartHeight;      //--- Variables for chart dimensions
   if (!ChartGetInteger(0, CHART_WIDTH_IN_PIXELS, 0, chartWidth) || !ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS, 0, chartHeight)) { //--- Get chart size
      LogError(__FUNCTION__ + ": Failed to get chart dimensions, Error: " + IntegerToString(GetLastError())); //--- Log error
      return;                         //--- Exit on failure
   }
   int fontSize = (int)(BaseFontSize * (chartWidth / 800.0)); //--- Calculate font size
   int cellWidth = PanelWidth / 8;    //--- Calculate cell width
   int cellHeight = 18;               //--- Set cell height
   int panelHeight = 70 + (ArraySize(periods) + 1) * cellHeight + 40 + (totalSymbols + 1) * cellHeight + 50; //--- Calculate panel height

   // Create Dark Panel
   createPanel("DashboardPanel", X_Offset, Y_Offset, PanelWidth, panelHeight, PanelColor); //--- Create dashboard panel

   // Create Header with Glow
   createText("Header", "HOLOGRAPHIC DASHBOARD", X_Offset + 10, Y_Offset + 10, TitleColor, fontSize + 4, FontType); //--- Create header text
   createText("HeaderGlow", "HOLOGRAPHIC DASHBOARD", X_Offset + 11, Y_Offset + 11, GlowColor, fontSize + 4, FontType, true); //--- Create header glow
   createText("SubHeader", StringFormat("%s | TF: %s", _Symbol, TruncPeriod(_Period)), X_Offset + 10, Y_Offset + 30, DataColor, fontSize, FontType); //--- Create subheader

   // Timeframe Grid
   int y = Y_Offset + 50;             //--- Set y-coordinate for timeframe grid
   createText("TF_Label", "Timeframe", X_Offset + 10, y, TitleColor, fontSize, FontType); //--- Create timeframe label
   createText("Trend_Label", "Trend", X_Offset + 10 + cellWidth, y, TitleColor, fontSize, FontType); //--- Create trend label
   createText("Vol_Label", "Vol", X_Offset + 10 + cellWidth * 2, y, TitleColor, fontSize, FontType); //--- Create vol label
   createText("RSI_Label", "RSI", X_Offset + 10 + cellWidth * 3, y, TitleColor, fontSize, FontType); //--- Create RSI label
   createLine("TF_Separator", X_Offset + 5, y + cellHeight + 2, X_Offset + PanelWidth - 5, y + cellHeight + 2, LineColor, 0.6); //--- Create separator line
   createLine("TF_Separator_Glow", X_Offset + 4, y + cellHeight + 1, X_Offset + PanelWidth - 4, y + cellHeight + 3, GlowColor, 0.3); //--- Create glow separator
   if (EnableAnimations) HolographicPulse("TF_Separator", LineColor, GlowColor); //--- Animate separator if enabled

   y += cellHeight + 5;               //--- Update y-coordinate
   for (int i = 0; i < ArraySize(periods); i++) { //--- Iterate through timeframes
      color periodColor = (periods[i] == _Period) ? ActiveColor : DataColor; //--- Set period color
      createText("Period_" + IntegerToString(i), TruncPeriod(periods[i]), X_Offset + 10, y, periodColor, fontSize, FontType); //--- Create period text
      createText("Trend_" + IntegerToString(i), "-", X_Offset + 10 + cellWidth, y, DataColor, fontSize, FontType); //--- Create trend text
      createText("Vol_" + IntegerToString(i), "0.00%", X_Offset + 10 + cellWidth * 2, y, DataColor, fontSize, FontType); //--- Create vol text
      createText("RSI_" + IntegerToString(i), "0.0", X_Offset + 10 + cellWidth * 3, y, DataColor, fontSize, FontType); //--- Create RSI text
      y += cellHeight;                //--- Update y-coordinate
   }

   // Symbol Grid
   y += 30;                           //--- Update y-coordinate for symbol grid
   createText("Symbol_Label", "Symbol", X_Offset + 10, y, TitleColor, fontSize, FontType); //--- Create symbol label
   createText("Bid_Label", "Bid", X_Offset + 10 + cellWidth, y, TitleColor, fontSize, FontType); //--- Create bid label
   createText("Spread_Label", "Spread", X_Offset + 10 + cellWidth * 2, y, TitleColor, fontSize, FontType); //--- Create spread label
   createText("Change_Label", "% Change", X_Offset + 10 + cellWidth * 3, y, TitleColor, fontSize, FontType); //--- Create change label
   createText("Vol_Label_Symbol", "Vol", X_Offset + 10 + cellWidth * 4, y, TitleColor, fontSize, FontType); //--- Create vol label
   createText("RSI_Label_Symbol", "RSI", X_Offset + 10 + cellWidth * 5, y, TitleColor, fontSize, FontType); //--- Create RSI label
   createText("UpArrow_Label", CharToString(236), X_Offset + 10 + cellWidth * 6, y, TitleColor, fontSize, "Wingdings"); //--- Create up arrow label
   createText("DownArrow_Label", CharToString(238), X_Offset + 10 + cellWidth * 7, y, TitleColor, fontSize, "Wingdings"); //--- Create down arrow label
   createLine("Symbol_Separator", X_Offset + 5, y + cellHeight + 2, X_Offset + PanelWidth - 5, y + cellHeight + 2, LineColor, 0.6); //--- Create separator line
   createLine("Symbol_Separator_Glow", X_Offset + 4, y + cellHeight + 1, X_Offset + PanelWidth - 4, y + cellHeight + 3, GlowColor, 0.3); //--- Create glow separator
   if (EnableAnimations) HolographicPulse("Symbol_Separator", LineColor, GlowColor); //--- Animate separator if enabled

   y += cellHeight + 5;               //--- Update y-coordinate
   for (int i = 0; i < totalSymbols; i++) { //--- Iterate through symbols
      string symbol = SymbolName(i, true); //--- Get symbol name
      string displaySymbol = (symbol == _Symbol) ? "*" + symbol : symbol; //--- Format display symbol
      color symbolColor = (symbol == _Symbol) ? ActiveColor : DataColor; //--- Set symbol color
      createText("Symbol_" + IntegerToString(i), displaySymbol, X_Offset + 10, y, symbolColor, fontSize, FontType); //--- Create symbol text
      createText("Bid_" + IntegerToString(i), Bid(symbol), X_Offset + 10 + cellWidth, y, DataColor, fontSize, FontType); //--- Create bid text
      createText("Spread_" + IntegerToString(i), Spread(symbol), X_Offset + 10 + cellWidth * 2, y, DataColor, fontSize, FontType); //--- Create spread text
      createText("Change_" + IntegerToString(i), "0.00%", X_Offset + 10 + cellWidth * 3, y, DataColor, fontSize, FontType); //--- Create change text
      createText("Vol_" + IntegerToString(i), "0.00%", X_Offset + 10 + cellWidth * 4, y, DataColor, fontSize, FontType); //--- Create vol text
      createText("RSI_" + IntegerToString(i), "0.0", X_Offset + 10 + cellWidth * 5, y, DataColor, fontSize, FontType); //--- Create RSI text
      createText("ArrowUp_" + IntegerToString(i), CharToString(236), X_Offset + 10 + cellWidth * 6, y, UpColor, fontSize, "Wingdings"); //--- Create up arrow
      createText("ArrowDown_" + IntegerToString(i), CharToString(238), X_Offset + 10 + cellWidth * 7, y, DownColor, fontSize, "Wingdings"); //--- Create down arrow
      y += cellHeight;                //--- Update y-coordinate
   }

   // Interactive Buttons with Pulse Animation
   createButton("ToggleBtn", "TOGGLE DASHBOARD", X_Offset + 10, y + 20, 150, 25, TitleColor, PanelColor, UpColor); //--- Create toggle button
   createButton("SwitchTFBtn", "NEXT TF", X_Offset + 170, y + 20, 120, 25, UpColor, PanelColor, UpColor); //--- Create switch TF button
   createButton("SortBtn", "SORT: " + sortNames[sortMode], X_Offset + 300, y + 20, 150, 25, TitleColor, PanelColor, UpColor); //--- Create sort button

   ChartRedraw(0);                    //--- Redraw chart
}
```

Here, we initialize the dashboard by first retrieving the chart's dimensions using the "chartWidth" and "chartHeight" variables. We achieve this by calling the [ChartGetInteger](https://www.mql5.com/en/docs/chart_operations/chartgetinteger) function twice: once with [CHART\_WIDTH\_IN\_PIXELS](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer) to get the width and once with "CHART\_HEIGHT\_IN\_PIXELS" to get the height. Next, we calculate the "fontSize" by scaling the "BaseFontSize" based on the chart's width relative to 800 pixels, casting the result to an integer. We then determine the "cellWidth" by dividing "PanelWidth" by 8 and set "cellHeight" to a fixed value of 18. The "panelHeight" is computed by adding 70, the product of (" [ArraySize(periods)](https://www.mql5.com/en/docs/array/arraysize)" \+ 1) and "cellHeight", 40, the product of ("totalSymbols" + 1) and "cellHeight", and 50—this accounts for the overall layout, including timeframes and symbols.

We proceed to create the dark panel background by invoking the "createPanel" function with the name "DashboardPanel", positioned at "X\_Offset" and "Y\_Offset", with dimensions "PanelWidth" and "panelHeight", and colored with "PanelColor". For the header, we create the main text label "Header" with the string "HOLOGRAPHIC DASHBOARD" using the "createText" function at coordinates "X\_Offset" + 10" and "Y\_Offset" + 10", styled with "TitleColor", a font size of "fontSize" + 4", and "FontType". To add a glow effect, we create another text label "HeaderGlow"" with the same string, but offset by 1 pixel in both x and y directions, using "GlowColor", the same font size, "FontType", and setting the opacity flag to true.

We then add a subheader label "SubHeader", formatted with the current symbol [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol) and truncated period from "TruncPeriod(\_Period)" using "StringFormat", positioned at "X\_Offset" + 10" and "Y\_Offset" + 30", colored with "DataColor", "fontSize", and "FontType".

Moving to the timeframe grid section, we set "y" to "Y\_Offset" + 50". We create labels for "Timeframe", "Trend", "Vol", and "RSI" using "createText"" for each, positioned horizontally with offsets based on "cellWidth", all using "TitleColor", "fontSize", and "FontType"". Below these, we draw a separator line "TF\_Separator"" using "createLine" function from "X\_Offset + 5" to "X\_Offset + PanelWidth - 5"", at height "y + cellHeight + 2"", colored "LineColor" with opacity 0.6". For glow, we add "TF\_Separator\_Glow"" as another line slightly offset and wider, with "GlowColor" and opacity 0.3". If "EnableAnimations" is true, we apply animation via "HolographicPulse" with "LineColor"" and "GlowColor". We use a similar logic for all the other label objects.

Finally, we create interactive buttons: "ToggleBtn"" as "TOGGLE DASHBOARD" at "X\_Offset" + 10", "y + 20", size 150x25, with "TitleColor", "PanelColor", "UpColor"; "SwitchTFBtn"" as "NEXT TF" at "X\_Offset" + 170", same y, size 120x25, with "UpColor", "PanelColor", "UpColor"; and "SortBtn"" as "SORT: " + sortNames\[sortMode\]" at "X\_Offset" + 300", same y, size 150x25, with "TitleColor", "PanelColor", "UpColor". We conclude by redrawing the chart with the "ChartRedraw(0)" function. With the function, we can call it in the initialization event handler, and we can have the heavy lifting done.

```
//+------------------------------------------------------------------+
//| Expert Initialization Function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   // Clear existing objects
   if (ObjectsDeleteAll(0, -1, -1) < 0) { //--- Delete all objects
      LogError(__FUNCTION__ + ": Failed to delete objects, Error: " + IntegerToString(GetLastError())); //--- Log error
   }
   objManager.DeleteAllObjects();     //--- Delete managed objects

   // Initialize arrays
   totalSymbols = SymbolsTotal(true); //--- Get total symbols
   if (totalSymbols == 0) {           //--- Check for symbols
      LogError(__FUNCTION__ + ": No symbols available"); //--- Log error
      return INIT_FAILED;             //--- Return failure
   }
   ArrayResize(prices_PrevArray, totalSymbols); //--- Resize previous prices array
   ArrayResize(volatility_Array, totalSymbols); //--- Resize volatility array
   ArrayResize(bid_array, totalSymbols); //--- Resize bid array
   ArrayResize(spread_array, totalSymbols); //--- Resize spread array
   ArrayResize(change_array, totalSymbols); //--- Resize change array
   ArrayResize(vol_array, totalSymbols); //--- Resize vol array
   ArrayResize(rsi_array, totalSymbols); //--- Resize RSI array
   ArrayResize(indices, totalSymbols);   //--- Resize indices array
   ArrayResize(atr_handles_sym, totalSymbols); //--- Resize ATR symbol handles
   ArrayResize(rsi_handles_sym, totalSymbols); //--- Resize RSI symbol handles
   ArrayResize(atr_handles_tf, ArraySize(periods)); //--- Resize ATR timeframe handles
   ArrayResize(rsi_handles_tf, ArraySize(periods)); //--- Resize RSI timeframe handles
   ArrayInitialize(prices_PrevArray, 0); //--- Initialize previous prices
   ArrayInitialize(volatility_Array, 0); //--- Initialize volatility

   // Create indicator handles for timeframes
   for (int i = 0; i < ArraySize(periods); i++) { //--- Iterate through timeframes
      atr_handles_tf[i] = iATR(_Symbol, periods[i], ATR_Period); //--- Create ATR handle
      if (atr_handles_tf[i] == INVALID_HANDLE) { //--- Check for invalid handle
         LogError(__FUNCTION__ + ": Failed to create ATR handle for TF " + EnumToString(periods[i])); //--- Log error
         return INIT_FAILED;          //--- Return failure
      }
      rsi_handles_tf[i] = iRSI(_Symbol, periods[i], RSI_Period, PRICE_CLOSE); //--- Create RSI handle
      if (rsi_handles_tf[i] == INVALID_HANDLE) { //--- Check for invalid handle
         LogError(__FUNCTION__ + ": Failed to create RSI handle for TF " + EnumToString(periods[i])); //--- Log error
         return INIT_FAILED;          //--- Return failure
      }
   }

   // Create indicator handles for symbols on H1
   for (int i = 0; i < totalSymbols; i++) { //--- Iterate through symbols
      string symbol = SymbolName(i, true); //--- Get symbol name
      atr_handles_sym[i] = iATR(symbol, PERIOD_H1, ATR_Period); //--- Create ATR handle
      if (atr_handles_sym[i] == INVALID_HANDLE) { //--- Check for invalid handle
         LogError(__FUNCTION__ + ": Failed to create ATR handle for symbol " + symbol); //--- Log error
         return INIT_FAILED;          //--- Return failure
      }
      rsi_handles_sym[i] = iRSI(symbol, PERIOD_H1, RSI_Period, PRICE_CLOSE); //--- Create RSI handle
      if (rsi_handles_sym[i] == INVALID_HANDLE) { //--- Check for invalid handle
         LogError(__FUNCTION__ + ": Failed to create RSI handle for symbol " + symbol); //--- Log error
         return INIT_FAILED;          //--- Return failure
      }
   }

   InitDashboard();                   //--- Initialize dashboard
   dashboardVisible = true;           //--- Set dashboard visible

   return INIT_SUCCEEDED;             //--- Return success
}
```

In the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) function, we clear existing objects with the [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/ObjectDeleteAll) function for all charts and types, logging failures with "LogError" if unsuccessful, and call "objManager.DeleteAllObjects" to remove managed items. We get "totalSymbols" from [SymbolsTotal](https://www.mql5.com/en/docs/marketinformation/symbolstotal) with true for market watch symbols, returning [INIT\_FAILED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode) if zero, and logging with "LogError". We resize arrays like "prices\_PrevArray", "volatility\_Array", "bid\_array", "spread\_array", "change\_array", "vol\_array", "rsi\_array", "indices", "atr\_handles\_sym", "rsi\_handles\_sym", "atr\_handles\_tf", and "rsi\_handles\_tf" to match "totalSymbols" or "ArraySize(periods)" using [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize), and initialize "prices\_PrevArray" and "volatility\_Array" to zero with the [ArrayInitialize](https://www.mql5.com/en/docs/array/arrayinitialize) function.

For timeframes, we loop through "periods" and create "atr\_handles\_tf\[i\]" with [iATR](https://www.mql5.com/en/docs/indicators/IATR) on "\_Symbol", "periods\[i\]", and "ATR\_Period", and "rsi\_handles\_tf\[i\]" with "iRSI" on [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol), "periods\[i\]", "RSI\_Period", and "PRICE\_CLOSE", logging and returning [INIT\_FAILED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode) if "INVALID\_HANDLE". Similarly for symbols, we loop through "totalSymbols", get "symbol" with "SymbolName" and true, create "atr\_handles\_sym\[i\]" with "iATR" on "symbol", "PERIOD\_H1", and "ATR\_Period", and "rsi\_handles\_sym\[i\]" with [iRSI](https://www.mql5.com/en/docs/indicators/irsi) on "symbol", "PERIOD\_H1", "RSI\_Period", and "PRICE\_CLOSE", logging and returning "INIT\_FAILED" if invalid. We call "InitDashboard" to build the UI, set "dashboardVisible" to true, and return success. When we run the program, we get the following outcome.

![INITIAL DASHBOARD](https://c.mql5.com/2/157/Screenshot_2025-07-17_232336.png)

From the image, we can see that we initialized the program successfully. We can take care of the program deinitialization, where we will need to delete the created objects and release the indicator handles.

```
//+------------------------------------------------------------------+
//| Expert Deinitialization Function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   if (ObjectsDeleteAll(0, -1, -1) < 0) { //--- Delete all objects
      LogError(__FUNCTION__ + ": Failed to delete objects, Error: " + IntegerToString(GetLastError())); //--- Log error
   }
   objManager.DeleteAllObjects();     //--- Delete managed objects

   // Release indicator handles
   for (int i = 0; i < ArraySize(atr_handles_tf); i++) { //--- Iterate through timeframe ATR handles
      if (atr_handles_tf[i] != INVALID_HANDLE) IndicatorRelease(atr_handles_tf[i]); //--- Release handle
      if (rsi_handles_tf[i] != INVALID_HANDLE) IndicatorRelease(rsi_handles_tf[i]); //--- Release handle
   }
   for (int i = 0; i < ArraySize(atr_handles_sym); i++) { //--- Iterate through symbol ATR handles
      if (atr_handles_sym[i] != INVALID_HANDLE) IndicatorRelease(atr_handles_sym[i]); //--- Release handle
      if (rsi_handles_sym[i] != INVALID_HANDLE) IndicatorRelease(rsi_handles_sym[i]); //--- Release handle
   }
}
```

In the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) event handler, we clean up resources when the EA is removed. We delete all chart objects with [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/ObjectDeleteAll) using -1 for all charts and types, logging failures with "LogError" if the result is negative. We call "objManager.DeleteAllObjects" to remove managed items. For timeframe handles, we loop through "atr\_handles\_tf" and "rsi\_handles\_tf" with [ArraySize](https://www.mql5.com/en/docs/array/arraysize), releasing valid handles with [IndicatorRelease](https://www.mql5.com/en/docs/series/indicatorrelease) if not "INVALID\_HANDLE". Similarly, for symbol handles in "atr\_handles\_sym" and "rsi\_handles\_sym". This ensures complete cleanup of objects and indicators. Here is an illustration.

![REMOVAL GIF](https://c.mql5.com/2/157/HOLOGRAPHIC_DEINIT.gif)

With the created objects taken care of completely, we can now go to the updates. We intend to do the updates in the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler to keep everything simple, but you could do them in the [OnTimer](https://www.mql5.com/en/docs/event_handlers/ontimer) event handler. Let's first start with the time frame section.

```
//+------------------------------------------------------------------+
//| Expert Tick Function with Holographic Updates                    |
//+------------------------------------------------------------------+
void OnTick() {
   if (!dashboardVisible) return;     //--- Exit if dashboard hidden

   long chartWidth;                   //--- Variable for chart width
   ChartGetInteger(0, CHART_WIDTH_IN_PIXELS, 0, chartWidth); //--- Get chart width
   int fontSize = (int)(BaseFontSize * (chartWidth / 800.0)); //--- Calculate font size
   int cellWidth = PanelWidth / 8;    //--- Calculate cell width
   int cellHeight = 18;               //--- Set cell height
   int y = Y_Offset + 75;             //--- Set y-coordinate for timeframe data

   // Update Timeframe Data with Pulse
   for (int i = 0; i < ArraySize(periods); i++) { //--- Iterate through timeframes
      double open = iOpen(_Symbol, periods[i], 0); //--- Get open price
      double close = iClose(_Symbol, periods[i], 0); //--- Get close price
      double atr_buf[1];              //--- Buffer for ATR
      if (CopyBuffer(atr_handles_tf[i], 0, 0, 1, atr_buf) != 1) { //--- Copy ATR data
         LogError(__FUNCTION__ + ": Failed to copy ATR buffer for TF " + EnumToString(periods[i])); //--- Log error
         continue;                    //--- Skip on failure
      }
      double vol = (close > 0) ? (atr_buf[0] / close) * 100 : 0.0; //--- Calculate volatility
      double rsi_buf[1];              //--- Buffer for RSI
      if (CopyBuffer(rsi_handles_tf[i], 0, 0, 1, rsi_buf) != 1) { //--- Copy RSI data
         LogError(__FUNCTION__ + ": Failed to copy RSI buffer for TF " + EnumToString(periods[i])); //--- Log error
         continue;                    //--- Skip on failure
      }
      double rsi = rsi_buf[0];        //--- Get RSI value
      color clr = DataColor;          //--- Set default color
      string trend = "-";             //--- Set default trend
      if (rsi > 50) { clr = UpColor; trend = "↑"; } //--- Set up trend
      else if (rsi < 50) { clr = DownColor; trend = "↓"; } //--- Set down trend
      createText("Trend_" + IntegerToString(i), trend, X_Offset + 10 + cellWidth, y, clr, fontSize, FontType, EnableAnimations); //--- Update trend text
      createText("Vol_" + IntegerToString(i), StringFormat("%.2f%%", vol), X_Offset + 10 + cellWidth * 2, y, vol > Vol_Alert_Threshold ? UpColor : DataColor, fontSize, FontType, vol > Vol_Alert_Threshold && EnableAnimations); //--- Update vol text
      color rsi_clr = (rsi > 70 ? DownColor : (rsi < 30 ? UpColor : DataColor)); //--- Set RSI color
      createText("RSI_" + IntegerToString(i), StringFormat("%.1f", rsi), X_Offset + 10 + cellWidth * 3, y, rsi_clr, fontSize, FontType, (rsi > 70 || rsi < 30) && EnableAnimations); //--- Update RSI text
      HolographicPulse("Period_" + IntegerToString(i), (periods[i] == _Period) ? ActiveColor : DataColor, GlowColor); //--- Pulse period text
      y += cellHeight;                //--- Update y-coordinate
   }

   // Update Symbol Data with Advanced Glow
   y += 50;                           //--- Update y-coordinate for symbol data
   for (int i = 0; i < totalSymbols; i++) { //--- Iterate through symbols
      string symbol = SymbolName(i, true); //--- Get symbol name
      double bidPrice;                //--- Variable for bid price
      if (!SymbolInfoDouble(symbol, SYMBOL_BID, bidPrice)) { //--- Get bid price
         LogError(__FUNCTION__ + ": Failed to get bid for " + symbol + ", Error: " + IntegerToString(GetLastError())); //--- Log error
         continue;                    //--- Skip on failure
      }
      long spread;                    //--- Variable for spread
      if (!SymbolInfoInteger(symbol, SYMBOL_SPREAD, spread)) { //--- Get spread
         LogError(__FUNCTION__ + ": Failed to get spread for " + symbol + ", Error: " + IntegerToString(GetLastError())); //--- Log error
         continue;                    //--- Skip on failure
      }
      double change = (prices_PrevArray[i] == 0 ? 0 : (bidPrice - prices_PrevArray[i]) / prices_PrevArray[i] * 100); //--- Calculate change
      double close = iClose(symbol, PERIOD_H1, 0); //--- Get close price
      double atr_buf[1];              //--- Buffer for ATR
      if (CopyBuffer(atr_handles_sym[i], 0, 0, 1, atr_buf) != 1) { //--- Copy ATR data
         LogError(__FUNCTION__ + ": Failed to copy ATR buffer for symbol " + symbol); //--- Log error
         continue;                    //--- Skip on failure
      }
      double vol = (close > 0) ? (atr_buf[0] / close) * 100 : 0.0; //--- Calculate volatility
      double rsi_buf[1];              //--- Buffer for RSI
      if (CopyBuffer(rsi_handles_sym[i], 0, 0, 1, rsi_buf) != 1) { //--- Copy RSI data
         LogError(__FUNCTION__ + ": Failed to copy RSI buffer for symbol " + symbol); //--- Log error
         continue;                    //--- Skip on failure
      }
      double rsi = rsi_buf[0];        //--- Get RSI value
      bid_array[i] = bidPrice;        //--- Store bid
      spread_array[i] = spread;       //--- Store spread
      change_array[i] = change;       //--- Store change
      vol_array[i] = vol;             //--- Store vol
      rsi_array[i] = rsi;             //--- Store RSI
      volatility_Array[i] = vol;      //--- Store volatility
      prices_PrevArray[i] = bidPrice; //--- Update previous price
   }
}
```

In the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) function, we handle updates on every market tick, ensuring real-time data refresh for timeframes and symbols. We exit early if "dashboardVisible" is false to skip unnecessary processing. We retrieve "chartWidth" with [ChartGetInteger](https://www.mql5.com/en/docs/chart_operations/chartgetinteger) using [CHART\_WIDTH\_IN\_PIXELS](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer), calculate "fontSize" scaled by "chartWidth / 800.0", "cellWidth" as "PanelWidth / 8", and "cellHeight" as 18. We set "y" to "Y\_Offset + 75" for the timeframe grid and loop through "periods" with the [ArraySize](https://www.mql5.com/en/docs/array/arraysize) function. For each timeframe, we get "open" with [iOpen](https://www.mql5.com/en/docs/series/iopen) and "close" with "iClose" at shift 0, copy "atr\_buf" with [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer) from "atr\_handles\_tf\[i\]", and calculate "vol" as percentage ATR over "close" if positive.

We copy "rsi\_buf" from "rsi\_handles\_tf\[i\]" and get "rsi", setting "clr" and "trend" based on RSI > 50 for up ("↑" in "UpColor") or < 50 for down ("↓" in "DownColor"). We could have used the arrows from fonts, but these hand-coded ones blend in smoothly and increase the holographic appeal. We update texts with "createText" for trend, vol (colored "UpColor" if > "Vol\_Alert\_Threshold" with animation), and RSI (colored based on overbought/oversold with animation), and call "HolographicPulse" on the period text with "ActiveColor" if matching [\_Period](https://www.mql5.com/en/docs/predefined/_period). We increment "y" by "cellHeight".

We update "y" by 50 for the symbol grid and loop through "totalSymbols". For each symbol from [SymbolName](https://www.mql5.com/en/docs/MarketInformation/SymbolName) with true, we fetch "bidPrice" with "SymbolInfoDouble" using "SYMBOL\_BID" and "spread" with [SymbolInfoInteger](https://www.mql5.com/en/docs/marketinformation/symbolinfointeger) using [SYMBOL\_SPREAD](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_integer), logging and skipping on failure. We calculate "change" as percentage over "prices\_PrevArray\[i\]", get "close" with [iClose](https://www.mql5.com/en/docs/series/iclose) on "PERIOD\_H1" at shift 0, copy "atr\_buf" from "atr\_handles\_sym\[i\]" to compute "vol", and "rsi\_buf" from "rsi\_handles\_sym\[i\]" to get "rsi". We store values in "bid\_array", "spread\_array", "change\_array", "vol\_array", "rsi\_array", and "volatility\_array", and update "prices\_PrevArray\[i\]" to "bidPrice". We can now move on to the symbols section, where we will need to sort and display them with effects.

```
// Sort indices
for (int i = 0; i < totalSymbols; i++) indices[i] = i; //--- Initialize indices
bool swapped = true;               //--- Swap flag
while (swapped) {                  //--- Loop until no swaps
   swapped = false;                //--- Reset flag
   for (int j = 0; j < totalSymbols - 1; j++) { //--- Iterate through indices
      bool do_swap = false;        //--- Swap decision
      int a = indices[j], b = indices[j + 1]; //--- Get indices
      if (sortMode == 0) {         //--- Sort by name ASC
         string na = SymbolName(a, true), nb = SymbolName(b, true); //--- Get names
         if (na > nb) do_swap = true; //--- Swap if needed
      } else if (sortMode == 1) {  //--- Sort by vol DESC
         if (vol_array[a] < vol_array[b]) do_swap = true; //--- Swap if needed
      } else if (sortMode == 2) {  //--- Sort by change ABS DESC
         if (MathAbs(change_array[a]) < MathAbs(change_array[b])) do_swap = true; //--- Swap if needed
      } else if (sortMode == 3) {  //--- Sort by RSI DESC
         if (rsi_array[a] < rsi_array[b]) do_swap = true; //--- Swap if needed
      }
      if (do_swap) {               //--- Perform swap
         int temp = indices[j];    //--- Temporary store
         indices[j] = indices[j + 1]; //--- Swap
         indices[j + 1] = temp;    //--- Complete swap
         swapped = true;           //--- Set flag
      }
   }
}

// Display sorted symbols with pulse on high vol
for (int j = 0; j < totalSymbols; j++) { //--- Iterate through sorted indices
   int i = indices[j];                   //--- Get index
   string symbol = SymbolName(i, true);  //--- Get symbol
   double bidPrice = bid_array[i];       //--- Get bid
   long spread = spread_array[i];        //--- Get spread
   double change = change_array[i];      //--- Get change
   double vol = vol_array[i];            //--- Get vol
   double rsi = rsi_array[i];            //--- Get RSI
   color clr_s = (symbol == _Symbol) ? ActiveColor : DataColor; //--- Set symbol color
   color clr_p = DataColor, clr_sp = DataColor, clr_ch = DataColor, clr_vol = DataColor, clr_rsi = DataColor; //--- Set default colors
   color clr_a1 = DataColor, clr_a2 = DataColor; //--- Set arrow colors

   // Price Change
   if (change > 0) {               //--- Check positive change
      clr_p = UpColor; clr_ch = UpColor; clr_a1 = UpColor; clr_a2 = DataColor; //--- Set up colors
   } else if (change < 0) {        //--- Check negative change
      clr_p = DownColor; clr_ch = DownColor; clr_a1 = DataColor; clr_a2 = DownColor; //--- Set down colors
   }

   // Volatility Alert
   if (vol > Vol_Alert_Threshold) { //--- Check high volatility
      clr_vol = UpColor;            //--- Set vol color
      clr_s = (symbol == _Symbol) ? ActiveColor : UpColor; //--- Set symbol color
   }

   // RSI Color
   clr_rsi = (rsi > 70 ? DownColor : (rsi < 30 ? UpColor : DataColor)); //--- Set RSI color

   // Update Texts
   string displaySymbol = (symbol == _Symbol) ? "*" + symbol : symbol; //--- Format display symbol
   createText("Symbol_" + IntegerToString(j), displaySymbol, X_Offset + 10, y, clr_s, fontSize, FontType, vol > Vol_Alert_Threshold && EnableAnimations); //--- Update symbol text
   createText("Bid_" + IntegerToString(j), Bid(symbol), X_Offset + 10 + cellWidth, y, clr_p, fontSize, FontType, EnableAnimations); //--- Update bid text
   createText("Spread_" + IntegerToString(j), Spread(symbol), X_Offset + 10 + cellWidth * 2, y, clr_sp, fontSize, FontType); //--- Update spread text
   createText("Change_" + IntegerToString(j), StringFormat("%.2f%%", change), X_Offset + 10 + cellWidth * 3, y, clr_ch, fontSize, FontType); //--- Update change text
   createText("Vol_" + IntegerToString(j), StringFormat("%.2f%%", vol), X_Offset + 10 + cellWidth * 4, y, clr_vol, fontSize, FontType, vol > Vol_Alert_Threshold && EnableAnimations); //--- Update vol text
   createText("RSI_" + IntegerToString(j), StringFormat("%.1f", rsi), X_Offset + 10 + cellWidth * 5, y, clr_rsi, fontSize, FontType, (rsi > 70 || rsi < 30) && EnableAnimations); //--- Update RSI text
   createText("ArrowUp_" + IntegerToString(j), CharToString(236), X_Offset + 10 + cellWidth * 6, y, clr_a1, fontSize, "Wingdings"); //--- Update up arrow
   createText("ArrowDown_" + IntegerToString(j), CharToString(238), X_Offset + 10 + cellWidth * 7, y, clr_a2, fontSize, "Wingdings"); //--- Update down arrow

   // Pulse on high volatility
   if (vol > Vol_Alert_Threshold) { //--- Check high volatility
      HolographicPulse("Symbol_" + IntegerToString(j), clr_s, GlowColor); //--- Pulse symbol text
   }

   y += cellHeight;                //--- Update y-coordinate
}

ChartRedraw(0);                    //--- Redraw chart
}
```

Here, we sort the indices by first initializing the "indices" array from 0 to "totalSymbols" - 1 in a loop. We use a bubble sort approach with the "swapped" flag set to true initially, entering a while loop until no more swaps occur. Inside, we reset "swapped" to false, then loop from 0 to "totalSymbols" - 2, setting "do\_swap" to false and getting "a" and "b" as "indices\[j\]" and "indices\[j+1\]". Depending on "sortMode": for 0 (name ASC), we get names via " [SymbolName(a, true)](https://www.mql5.com/en/docs/MarketInformation/SymbolName)" and "SymbolName(b, true)", swap if "na > nb"; for 1 (vol DESC), swap if "vol\_array\[a\] < vol\_array\[b\]"; for 2 (change ABS DESC), swap if "MathAbs(change\_array\[a\]) < [MathAbs(change\_array\[b\])](https://www.mql5.com/en/docs/math/mathabs)"; for 3 (RSI DESC), swap if "rsi\_array\[a\] < rsi\_array\[b\]". If "do\_swap", we swap "indices\[j\]" and "indices\[j+1\]" using a "temp" variable and set "swapped" to true.

Next, we display sorted symbols by looping over "totalSymbols", getting "i" as "indices\[j\]", then fetching "symbol" via "SymbolName(i, true)", "bidPrice" from "bid\_array\[i\]", "spread" from "spread\_array\[i\]", "change" from "change\_array\[i\]", "vol" from "vol\_array\[i\]", and "rsi" from "rsi\_array\[i\]". We set "clr\_s" to "ActiveColor" if it matches "\_Symbol", else "DataColor"; default other colors to "DataColor". For price change: if "change > 0", set "clr\_p", "clr\_ch", "clr\_a1" to "UpColor" and "clr\_a2" to "DataColor"; if < 0, set to "DownColor" with "clr\_a1" as "DataColor". For volatility alert: if "vol > Vol\_Alert\_Threshold", set "clr\_vol" to "UpColor" and update "clr\_s" if not the current symbol. For RSI: set "clr\_rsi" to "DownColor" if >70, "UpColor" if <30, else "DataColor".

We format "displaySymbol" with "\*" if it matches "\_Symbol". Update texts via "createText": symbol ("Symbol\_j") with "displaySymbol", "clr\_s", animate if high vol and enabled; bid ("Bid\_j") with "Bid(symbol)", "clr\_p", animate if enabled; spread ("Spread\_j") with "Spread(symbol)", "clr\_sp"; change ("Change\_j") formatted "%.2f%%" via "StringFormat", "clr\_ch"; vol ("Vol\_j") formatted "%.2f%%", "clr\_vol", animate if high vol and enabled; RSI ("RSI\_j") formatted "%.1f", "clr\_rsi", animate if overbought/oversold and enabled; up arrow ("ArrowUp\_j") with "CharToString(236)", "clr\_a1", "Wingdings"; down arrow ("ArrowDown\_j") with " [CharToString(238)](https://www.mql5.com/en/docs/convert/chartostring)", "clr\_a2", "Wingdings". If high vol, apply "HolographicPulse" on symbol text with "clr\_s" and "GlowColor". Increment "y" by "cellHeight" each iteration and finally redraw. When we compile, we get the following outcome.

![PULSIVE UPDATES IN ONTICK](https://c.mql5.com/2/157/HOLOGRAPHIC_ontick.gif)

From the visualization, we can see that the updates are taking effect on every market tick. We can now graduate to adding life to the buttons we created. We will achieve that via the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler.

```
//+------------------------------------------------------------------+
//| Chart Event Handler                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam) {
   if (id == CHARTEVENT_OBJECT_CLICK) { //--- Handle click event
      if (sparam == "ToggleBtn") {    //--- Check toggle button
         dashboardVisible = !dashboardVisible; //--- Toggle visibility
         objManager.DeleteAllObjects(); //--- Delete objects
         if (dashboardVisible) {      //--- Check if visible
            InitDashboard();          //--- Reinitialize dashboard
         } else {
            createButton("ToggleBtn", "TOGGLE DASHBOARD", X_Offset + 10, Y_Offset + 10, 150, 25, TitleColor, PanelColor, UpColor); //--- Create toggle button
         }
      }
      else if (sparam == "SwitchTFBtn") { //--- Check switch TF button
         int currentIdx = -1;         //--- Initialize current index
         for (int i = 0; i < ArraySize(periods); i++) { //--- Find current timeframe
            if (periods[i] == _Period) { //--- Match found
               currentIdx = i;        //--- Set index
               break;                 //--- Exit loop
            }
         }
         int nextIdx = (currentIdx + 1) % ArraySize(periods); //--- Calculate next index
         if (!ChartSetSymbolPeriod(0, _Symbol, periods[nextIdx])) { //--- Switch timeframe
            LogError(__FUNCTION__ + ": Failed to switch timeframe, Error: " + IntegerToString(GetLastError())); //--- Log error
         }
         createButton("SwitchTFBtn", "NEXT TF", X_Offset + 170, (int)ObjectGetInteger(0, "SwitchTFBtn", OBJPROP_YDISTANCE), 120, 25, UpColor, PanelColor, UpColor, EnableAnimations); //--- Update button
      }
      else if (sparam == "SortBtn") { //--- Check sort button
         sortMode = (sortMode + 1) % 4; //--- Cycle sort mode
         createButton("SortBtn", "SORT: " + sortNames[sortMode], X_Offset + 300, (int)ObjectGetInteger(0, "SortBtn", OBJPROP_YDISTANCE), 150, 25, TitleColor, PanelColor, UpColor, EnableAnimations); //--- Update button
      }
      ObjectSetInteger(0, sparam, OBJPROP_STATE, false); //--- Reset button state
      ChartRedraw(0);                 //--- Redraw chart
   }
}
```

We implement the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler to handle interactive events, responding to button clicks for toggling visibility, switching timeframes, and cycling sort modes. For [CHARTEVENT\_OBJECT\_CLICK](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents), we check "sparam" against "ToggleBtn", toggling "dashboardVisible", deleting objects with "objManager.DeleteAllObjects", and reinitializing with "InitDashboard" if visible or creating a new "ToggleBtn" with "createButton" if hidden. If "sparam" is "SwitchTFBtn", we find the current timeframe index in "periods" with a loop, calculate "nextIdx" as "(currentIdx + 1) % ArraySize(periods)", switch the chart with "ChartSetSymbolPeriod" using "periods\[nextIdx\]", log failures with "LogError", and update the button with "createButton" including animation if "EnableAnimations".

For "SortBtn", we cycle "sortMode" with "(sortMode + 1) % 4" and update the button text to "SORT: " + "sortNames\[sortMode\]" using "createButton" with animation. We reset the button state with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/ObjectSetInteger) for [OBJPROP\_STATE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to false and redraw the chart with the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function. This enables control over the dashboard's display and data organization. Upon compilation, we have the following output.

![RESPONSIVE BUTTON CLICKS](https://c.mql5.com/2/157/HOLOGRAPHIC_clicks.gif)

We can see that we can update the dashboard on every market tick and respond to the button clicks for the dashboard toggle, timeframe change, and the sorting of the indices for the symbol metrics, hence achieving our objectives. What now remains is testing the workability of the project, and that is handled in the preceding section.

### Backtesting

We did the testing, and below is the compiled visualization in a single [Graphics Interchange Format](https://en.wikipedia.org/wiki/GIF "https://en.wikipedia.org/wiki/GIF") (GIF) bitmap image format.

![BACKTESTING](https://c.mql5.com/2/157/HOLOGRAPH_TESTING.gif)

### Conclusion

In conclusion, we’ve created a Dynamic Holographic Dashboard in [MQL5](https://www.mql5.com/) that monitors symbols and timeframes with RSI, volatility alerts, and sorting, featuring pulse animations and interactive buttons for an immersive trading experience. We’ve detailed the architecture and implementation, using [class components](https://www.mql5.com/en/docs/basis/types/classes) like "CObjectManager" and [functions](https://www.mql5.com/en/docs/basis/function) such as "HolographicPulse" to deliver real-time, visually engaging insights. You can customize this dashboard to fit your trading needs, elevating your analysis with holographic effects and controls.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18880.zip "Download all attachments in the single ZIP archive")

[Holographic\_Dashboard\_EA.mq5](https://www.mql5.com/en/articles/download/18880/holographic_dashboard_ea.mq5 "Download Holographic_Dashboard_EA.mq5")(79.18 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/491711)**

![Price Action Analysis Toolkit Development (Part 33): Candle Range Theory Tool](https://c.mql5.com/2/159/18911-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 33): Candle Range Theory Tool](https://www.mql5.com/en/articles/18911)

Upgrade your market reading with the Candle-Range Theory suite for MetaTrader 5, a fully MQL5-native solution that converts raw price bars into real-time volatility intelligence. The lightweight CRangePattern library benchmarks each candle’s true range against an adaptive ATR and classifies it the instant it closes; the CRT Indicator then projects those classifications on your chart as crisp, color-coded rectangles and arrows that reveal tightening consolidations, explosive breakouts, and full-range engulfment the moment they occur.

![Introduction to MQL5 (Part 19): Automating Wolfe Wave Detection](https://c.mql5.com/2/158/18884-introduction-to-mql5-part-19-logo.png)[Introduction to MQL5 (Part 19): Automating Wolfe Wave Detection](https://www.mql5.com/en/articles/18884)

This article shows how to programmatically identify bullish and bearish Wolfe Wave patterns and trade them using MQL5. We’ll explore how to identify Wolfe Wave structures programmatically and execute trades based on them using MQL5. This includes detecting key swing points, validating pattern rules, and preparing the EA to act on the signals it finds.

![From Novice to Expert: Animated News Headline Using MQL5 (VII) — Post Impact Strategy for News Trading](https://c.mql5.com/2/159/18817-from-novice-to-expert-animated-logo.png)[From Novice to Expert: Animated News Headline Using MQL5 (VII) — Post Impact Strategy for News Trading](https://www.mql5.com/en/articles/18817)

The risk of whipsaw is extremely high during the first minute following a high-impact economic news release. In that brief window, price movements can be erratic and volatile, often triggering both sides of pending orders. Shortly after the release—typically within a minute—the market tends to stabilize, resuming or correcting the prevailing trend with more typical volatility. In this section, we’ll explore an alternative approach to news trading, aiming to assess its effectiveness as a valuable addition to a trader’s toolkit. Continue reading for more insights and details in this discussion.

![Population ADAM (Adaptive Moment Estimation)](https://c.mql5.com/2/104/Adaptive_Moment_Estimation___LOGO.png)[Population ADAM (Adaptive Moment Estimation)](https://www.mql5.com/en/articles/16443)

The article presents the transformation of the well-known and popular ADAM gradient optimization method into a population algorithm and its modification with the introduction of hybrid individuals. The new approach allows creating agents that combine elements of successful decisions using probability distribution. The key innovation is the formation of hybrid population individuals that adaptively accumulate information from the most promising solutions, increasing the efficiency of search in complex multidimensional spaces.

[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=gzzvtfizkloknockarxzmsxarzjnkwbe&ssn=1769180022315610424&ssn_dr=0&ssn_sr=0&fv_date=1769180022&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18880&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Trading%20Tools%20(Part%206)%3A%20Dynamic%20Holographic%20Dashboard%20with%20Pulse%20Animations%20and%20Controls%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918002268796428&fz_uniq=5068770212376935798&sv=2552)

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