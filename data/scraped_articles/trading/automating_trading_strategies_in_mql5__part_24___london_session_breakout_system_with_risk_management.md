---
title: Automating Trading Strategies in MQL5 (Part 24): London Session Breakout System with Risk Management and Trailing Stops
url: https://www.mql5.com/en/articles/18867
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 7
scraped_at: 2026-01-22T17:47:03.729213
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/18867&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049376717048490597)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 23)](https://www.mql5.com/en/articles/18778), we enhanced the [Zone Recovery System](https://www.mql5.com/go?link=https://tradingliteracy.com/zone-recovery-in-trading/ "https://tradingliteracy.com/zone-recovery-in-trading/") for Envelopes Trend Trading in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) with trailing stops and multi-basket trading for better profit protection and signal handling. In Part 24, we develop a [London Session Breakout System](https://www.mql5.com/go?link=https://tradingstrategyguides.com/london-breakout-strategy/ "https://tradingstrategyguides.com/london-breakout-strategy/") that identifies pre-session ranges, places pending orders, and incorporates risk management tools, including risk-to-reward ratios, drawdown limits, and a control panel for real-time monitoring. We will cover the following topics:

1. [Understanding the London Session Breakout Strategy](https://www.mql5.com/en/articles/18867#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/18867#para2)
3. [Backtesting](https://www.mql5.com/en/articles/18867#para3)
4. [Conclusion](https://www.mql5.com/en/articles/18867#para4)

By the end, you’ll have a complete MQL5 breakout program with advanced risk controls, ready for testing and refinement—let’s dive in!

### Understanding the London Session Breakout Strategy

The [London Session Breakout Strategy](https://www.mql5.com/go?link=https://tradingstrategyguides.com/london-breakout-strategy/ "https://tradingstrategyguides.com/london-breakout-strategy/") targets the increased volatility during the London market open by identifying the price range formed in the pre-London hours and placing pending orders to capture breakouts from that range. This strategy is important because the London session often experiences high liquidity and price movements, offering reliable opportunities for profit; however, it requires careful risk management to avoid false breakouts and drawdowns.

We will achieve this by calculating the pre-London high and low to set buy and sell stop orders with offsets, incorporating risk-to-reward ratios for take-profits, trailing stops for profit locking, and limits on open trades and daily drawdown to protect capital. We plan to use a control panel for real-time monitoring and session-specific checks to ensure trades only occur within defined ranges, making the system adaptable to varying market conditions. In a nutshell, here is a representation of the system we want to achieve.

![STRATEGY BLUEPRINT](https://c.mql5.com/2/157/Screenshot_2025-07-16_001547.png)

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Indicators folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will start by declaring some [inputs](https://www.mql5.com/en/docs/basis/variables/inputvariables) and [structures](https://www.mql5.com/en/docs/basis/types/classes) that will make the program more dynamic.

```
//+------------------------------------------------------------------+
//|                                        London Breakout EA.mq5    |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//|                                   https://t.me/Forex_Algo_Trader |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Allan Munene Mutiiria."
#property link      "https://t.me/Forex_Algo_Trader"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh> //--- Include Trade library for trading operations

//--- Enumerations
enum ENUM_TRADE_TYPE {     //--- Enumeration for trade types
   TRADE_ALL,              // All Trades (Buy and Sell)
   TRADE_BUY_ONLY,         // Buy Trades Only
   TRADE_SELL_ONLY         // Sell Trades Only
};

//--- Input parameters
sinput group "General EA Settings"
input double inpTradeLotsize = 0.01; // Lotsize
input ENUM_TRADE_TYPE TradeType = TRADE_ALL; // Trade Type Selection
sinput int MagicNumber = 12345;     // Magic Number
input double RRRatio = 1.0;        // Risk to Reward Ratio
input int StopLossPoints = 500;    // Stop loss in points
input int OrderOffsetPoints = 10;   // Points offset for Orders
input bool DeleteOppositeOrder = true; // Delete opposite order when one is activated?
input bool UseTrailing = false;    // Use Trailing Stop?
input int TrailingPoints = 50;     // Trailing Points (distance)
input int MinProfitPoints = 100;   // Minimum Profit Points to start trailing

sinput group "London Session Settings"
input int LondonStartHour = 9;        // London Start Hour
input int LondonStartMinute = 0;      // London Start Minute
input int LondonEndHour = 8;          // London End Hour
input int LondonEndMinute = 0;        // London End Minute
input int MinRangePoints = 100;       // Min Pre-London Range in points
input int MaxRangePoints = 300;       // Max Pre-London Range in points

sinput group "Risk Management"
input int MaxOpenTrades = 2;       // Maximum simultaneous open trades
input double MaxDailyDrawdownPercent = 5.0; // Max daily drawdown % to stop trading

//--- Structures
struct PositionInfo {      //--- Structure for position information
   ulong ticket;           // Position ticket
   double openPrice;      // Entry price
   double londonRange;    // Pre-London range in points for this position
   datetime sessionID;    // Session identifier (day)
   bool trailingActive;   // Trailing active flag
};
```

We begin implementing our [London Session Breakout System](https://www.mql5.com/go?link=https://tradingstrategyguides.com/london-breakout-strategy/ "https://tradingstrategyguides.com/london-breakout-strategy/") by including the "<Trade\\Trade.mqh>" library and defining key [enumerations](https://www.mql5.com/en/docs/basis/types/integer/enumeration), inputs, and a [structure](https://www.mql5.com/en/docs/basis/types/classes) for position tracking. We include "<Trade\\Trade.mqh>" to access the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class for executing trading operations like placing orders and modifying positions. We define the "ENUM\_TRADE\_TYPE" enumeration with options "TRADE\_ALL" for both buy and sell trades, "TRADE\_BUY\_ONLY" for buys only, and "TRADE\_SELL\_ONLY" for sells only, allowing us to restrict trade directions.

We then set up input parameters in groups: under "General EA Settings", "inpTradeLotsize" at 0.01 for lot size, "TradeType" using the enumeration with default "TRADE\_ALL", "MagicNumber" at 12345 to identify EA trades, "RRRatio" at 1.0 for risk-reward ratio, "StopLossPoints" at 500 for stop-loss distance, "OrderOffsetPoints" at 10 for entry offsets, "DeleteOppositeOrder" as true to remove opposite pending orders, "UseTrailing" as false to enable trailing stops, "TrailingPoints" at 50 for trailing distance, and "MinProfitPoints" at 100 to start trailing.

Under "London Session Settings", "LondonStartHour" at 9 and "LondonStartMinute" at 0 for session start, "LondonEndHour" at 8 and "LondonEndMinute" at 0 for end, "MinRangePoints" at 100 and "MaxRangePoints" at 300 for pre-London range validation. Under "Risk Management", "MaxOpenTrades" is set at 2 to limit simultaneous positions, and "MaxDailyDrawdownPercent" at 5.0 to halt trading on excessive drawdown. We define the "PositionInfo" [structure](https://www.mql5.com/en/docs/basis/types/classes) to track open trades, with "ticket" for the position ticket, "openPrice" for entry price, "londonRange" for the pre-London range, "sessionID" for the day identifier, and "trailingActive" as a boolean for trailing status. Upon compilation, we have the following output.

![INPUT SET](https://c.mql5.com/2/157/Screenshot_2025-07-16_003436.png)

With the inputs set in that structured manner, we can now define some extra [global variables](https://www.mql5.com/en/docs/basis/variables/global) that we will use throughout the program.

```
//--- Global variables
CTrade obj_Trade;                 //--- Trade object
double PreLondonHigh = 0.0;       //--- Pre-London session high
double PreLondonLow = 0.0;        //--- Pre-London session low
datetime PreLondonHighTime = 0;   //--- Time of Pre-London high
datetime PreLondonLowTime = 0;    //--- Time of Pre-London low
ulong buyOrderTicket = 0;         //--- Buy stop order ticket
ulong sellOrderTicket = 0;        //--- Sell stop order ticket
bool panelVisible = true;         //--- Panel visibility flag
double LondonRangePoints = 0.0;   //--- Current session's Pre-London range
PositionInfo positionList[];      //--- Array to store position info
datetime lastCheckedDay = 0;      //--- Last checked day
bool noTradeToday = false;        //--- Flag to prevent trading today
bool sessionChecksDone = false;   //--- Flag for session checks completion
datetime analysisTime = 0;        //--- Time for London analysis
double dailyDrawdown = 0.0;       //--- Current daily drawdown
bool isTrailing = false;          //--- Global flag for any trailing active
const int PreLondonStartHour = 3; //--- Fixed Pre-London Start Hour
const int PreLondonStartMinute = 0; //--- Fixed Pre-London Start Minute
```

Here, we define global variables for our program: "obj\_Trade" as [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) for trading, "PreLondonHigh" and "PreLondonLow" as doubles for ranges, "PreLondonHighTime" and "PreLondonLowTime" as [datetimes](https://www.mql5.com/en/docs/basis/types/integer/datetime) for timings, "buyOrderTicket" and "sellOrderTicket" as ulongs for orders, "panelVisible" as true for the panel, "LondonRangePoints" as 0.0 for current range, "positionList" as "PositionInfo" array for positions, "lastCheckedDay" as 0 for daily tracking, "noTradeToday" and "sessionChecksDone" as false for trading flags, "analysisTime" as 0 for session timing, "dailyDrawdown" as 0.0 for risk, "isTrailing" as false for trailing, and constants "PreLondonStartHour" as 3 and "PreLondonStartMinute" as 0.

With that done, we will proceed to create the panel, which is the easiest step, then advance to the more complex trading logic. Let's start with the necessary creation functions.

```
//+------------------------------------------------------------------+
//| Create a rectangle label for the panel background                |
//+------------------------------------------------------------------+
bool createRecLabel(string objName, int xD, int yD, int xS, int yS,
                    color clrBg, int widthBorder, color clrBorder = clrNONE,
                    ENUM_BORDER_TYPE borderType = BORDER_FLAT, ENUM_LINE_STYLE borderStyle = STYLE_SOLID) {
    ResetLastError();              //--- Reset last error
    if (!ObjectCreate(0, objName, OBJ_RECTANGLE_LABEL, 0, 0, 0)) { //--- Create rectangle label
        Print(__FUNCTION__, ": failed to create rec label! Error code = ", _LastError); //--- Log creation failure
        return false;              //--- Return failure
    }
    ObjectSetInteger(0, objName, OBJPROP_XDISTANCE, xD); //--- Set x-distance
    ObjectSetInteger(0, objName, OBJPROP_YDISTANCE, yD); //--- Set y-distance
    ObjectSetInteger(0, objName, OBJPROP_XSIZE, xS); //--- Set x-size
    ObjectSetInteger(0, objName, OBJPROP_YSIZE, yS); //--- Set y-size
    ObjectSetInteger(0, objName, OBJPROP_CORNER, CORNER_LEFT_UPPER); //--- Set corner
    ObjectSetInteger(0, objName, OBJPROP_BGCOLOR, clrBg); //--- Set background color
    ObjectSetInteger(0, objName, OBJPROP_BORDER_TYPE, borderType); //--- Set border type
    ObjectSetInteger(0, objName, OBJPROP_STYLE, borderStyle); //--- Set border style
    ObjectSetInteger(0, objName, OBJPROP_WIDTH, widthBorder); //--- Set border width
    ObjectSetInteger(0, objName, OBJPROP_COLOR, clrBorder); //--- Set border color
    ObjectSetInteger(0, objName, OBJPROP_BACK, false); //--- Set foreground
    ObjectSetInteger(0, objName, OBJPROP_STATE, false); //--- Set state
    ObjectSetInteger(0, objName, OBJPROP_SELECTABLE, false); //--- Disable selectable
    ObjectSetInteger(0, objName, OBJPROP_SELECTED, false); //--- Disable selected
    ChartRedraw(0);                //--- Redraw chart
    return true;                   //--- Return success
}

//+------------------------------------------------------------------+
//| Create a text label for panel elements                           |
//+------------------------------------------------------------------+
bool createLabel(string objName, int xD, int yD,
                 string txt, color clrTxt = clrBlack, int fontSize = 10,
                 string font = "Arial") {
    ResetLastError();              //--- Reset last error
    if (!ObjectCreate(0, objName, OBJ_LABEL, 0, 0, 0)) { //--- Create label
        Print(__FUNCTION__, ": failed to create the label! Error code = ", _LastError); //--- Log creation failure
        return false;              //--- Return failure
    }
    ObjectSetInteger(0, objName, OBJPROP_XDISTANCE, xD); //--- Set x-distance
    ObjectSetInteger(0, objName, OBJPROP_YDISTANCE, yD); //--- Set y-distance
    ObjectSetInteger(0, objName, OBJPROP_CORNER, CORNER_LEFT_UPPER); //--- Set corner
    ObjectSetString(0, objName, OBJPROP_TEXT, txt); //--- Set text
    ObjectSetInteger(0, objName, OBJPROP_COLOR, clrTxt); //--- Set color
    ObjectSetInteger(0, objName, OBJPROP_FONTSIZE, fontSize); //--- Set font size
    ObjectSetString(0, objName, OBJPROP_FONT, font); //--- Set font
    ObjectSetInteger(0, objName, OBJPROP_BACK, false); //--- Set foreground
    ObjectSetInteger(0, objName, OBJPROP_STATE, false); //--- Set state
    ObjectSetInteger(0, objName, OBJPROP_SELECTABLE, false); //--- Disable selectable
    ObjectSetInteger(0, objName, OBJPROP_SELECTED, false); //--- Disable selected
    ChartRedraw(0);                //--- Redraw chart
    return true;                   //--- Return success
}
```

Here, we implement utility functions to create the control panel [User Interface](https://en.wikipedia.org/wiki/User_interface "https://en.wikipedia.org/wiki/User_interface") (UI) elements using rectangle labels for backgrounds and text labels for display. We start with the "createRecLabel" function to generate rectangle labels for panel backgrounds, taking some parameters. We reset errors with the [ResetLastError](https://www.mql5.com/en/docs/common/ResetLastError) function and create the object with [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) as [OBJ\_RECTANGLE\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object), logging failures with [Print](https://www.mql5.com/en/docs/common/print), and returning false if unsuccessful. We set properties using [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) for [OBJPROP\_XDISTANCE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) and the same case to all the other necessary integer properties, then redraw the chart with [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) and return true.

Next, we create the "createLabel" function for text labels in the panel, taking "objName", "xD", "yD", "txt", "clrTxt", "fontSize", and "font" as parameters. We reset errors with "ResetLastError" and create the object with "ObjectCreate" as [OBJ\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object), logging failures with "Print", and returning false if unsuccessful. We set properties using the "ObjectSetInteger" function just as we did with the rectangle label function, but here, we use the additional [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) function for "OBJPROP\_TEXT" and [OBJPROP\_FONT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_string) properties, then redraw and return true. These functions will enable us to build a dynamic control panel for monitoring session data and program status. We can now use the functions to create and update the panel.

```
string panelPrefix = "LondonPanel_"; //--- Prefix for panel objects

//+------------------------------------------------------------------+
//| Create the information panel                                     |
//+------------------------------------------------------------------+
void CreatePanel() {
   createRecLabel(panelPrefix + "Background", 10, 10, 270, 200, clrMidnightBlue, 1, clrSilver); //--- Create background
   createLabel(panelPrefix + "Title", 20, 15, "London Breakout Control Center", clrGold, 12); //--- Create title
   createLabel(panelPrefix + "RangePoints", 20, 40, "Range (points): ", clrWhite, 10); //--- Create range label
   createLabel(panelPrefix + "HighPrice", 20, 60, "High Price: ", clrWhite); //--- Create high price label
   createLabel(panelPrefix + "LowPrice", 20, 80, "Low Price: ", clrWhite); //--- Create low price label
   createLabel(panelPrefix + "BuyLevel", 20, 100, "Buy Level: ", clrWhite); //--- Create buy level label
   createLabel(panelPrefix + "SellLevel", 20, 120, "Sell Level: ", clrWhite); //--- Create sell level label
   createLabel(panelPrefix + "AccountBalance", 20, 140, "Balance: ", clrWhite); //--- Create balance label
   createLabel(panelPrefix + "AccountEquity", 20, 160, "Equity: ", clrWhite); //--- Create equity label
   createLabel(panelPrefix + "CurrentDrawdown", 20, 180, "Drawdown (%): ", clrWhite); //--- Create drawdown label
   createRecLabel(panelPrefix + "Hide", 250, 10, 30, 22, clrCrimson, 1, clrNONE); //--- Create hide button
   createLabel(panelPrefix + "HideText", 258, 12, CharToString(251), clrWhite, 13, "Wingdings"); //--- Create hide text
   ObjectSetInteger(0, panelPrefix + "Hide", OBJPROP_SELECTABLE, true); //--- Make hide selectable
   ObjectSetInteger(0, panelPrefix + "Hide", OBJPROP_STATE, true); //--- Set hide state
}

//+------------------------------------------------------------------+
//| Update panel with current data                                   |
//+------------------------------------------------------------------+
void UpdatePanel() {
   string rangeText = "Range (points): " + (LondonRangePoints > 0 ? DoubleToString(LondonRangePoints, 0) : "Calculating..."); //--- Format range text
   ObjectSetString(0, panelPrefix + "RangePoints", OBJPROP_TEXT, rangeText); //--- Update range text

   string highText = "High Price: " + (LondonRangePoints > 0 ? DoubleToString(PreLondonHigh, _Digits) : "N/A"); //--- Format high text
   ObjectSetString(0, panelPrefix + "HighPrice", OBJPROP_TEXT, highText); //--- Update high text

   string lowText = "Low Price: " + (LondonRangePoints > 0 ? DoubleToString(PreLondonLow, _Digits) : "N/A"); //--- Format low text
   ObjectSetString(0, panelPrefix + "LowPrice", OBJPROP_TEXT, lowText); //--- Update low text

   string buyText = "Buy Level: " + (LondonRangePoints > 0 ? DoubleToString(PreLondonHigh + OrderOffsetPoints * _Point, _Digits) : "N/A"); //--- Format buy text
   ObjectSetString(0, panelPrefix + "BuyLevel", OBJPROP_TEXT, buyText); //--- Update buy text

   string sellText = "Sell Level: " + (LondonRangePoints > 0 ? DoubleToString(PreLondonLow - OrderOffsetPoints * _Point, _Digits) : "N/A"); //--- Format sell text
   ObjectSetString(0, panelPrefix + "SellLevel", OBJPROP_TEXT, sellText); //--- Update sell text

   string balanceText = "Balance: " + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2); //--- Format balance text
   ObjectSetString(0, panelPrefix + "AccountBalance", OBJPROP_TEXT, balanceText); //--- Update balance text

   string equityText = "Equity: " + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2); //--- Format equity text
   ObjectSetString(0, panelPrefix + "AccountEquity", OBJPROP_TEXT, equityText); //--- Update equity text

   string ddText = "Drawdown (%): " + DoubleToString(dailyDrawdown, 2); //--- Format drawdown text
   ObjectSetString(0, panelPrefix + "CurrentDrawdown", OBJPROP_TEXT, ddText); //--- Update drawdown text
   ObjectSetInteger(0, panelPrefix + "CurrentDrawdown", OBJPROP_COLOR, dailyDrawdown > MaxDailyDrawdownPercent / 2 ? clrYellow : clrWhite); //--- Set drawdown color
}
```

Here, we define the "panelPrefix" string as "LondonPanel\_" to prefix all panel object names, ensuring organized identification for the control panel. We create the "CreatePanel" function to build the information panel UI. We call "createRecLabel" for "panelPrefix + "Background" to make the panel background at position 10,10 with size 270x200, "clrMidnightBlue" background, width 1, and a silver border. We use "createLabel" to add the title "London Breakout Control Center" at 20,15 with gold color and size 12, and labels for range, high price, low price, buy level, sell level, balance, equity, and drawdown at respective positions with white color and size 10.

For the hide button, we call "createRecLabel" for "panelPrefix + 'Hide' at 250,10 with size 30x22 and "clrCrimson" background, and "createLabel" for "panelPrefix + 'HideText' with [CharToString](https://www.mql5.com/en/docs/convert/chartostring)(251) from [Wingdings](https://en.wikipedia.org/wiki/Wingdings "https://en.wikipedia.org/wiki/Wingdings") at 258,12 with "clrWhite" color and size 13. We set [OBJPROP\_SELECTABLE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) and "OBJPROP\_STATE" to true for the hide button with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) to make it interactive. The choice of the Wingdings code to use is dependent on your aesthetic feel. Here is a list of code you can choose from.

![WINGDINGS CODES](https://c.mql5.com/2/157/C_MQL5_WINGDINGS.png)

We then implement the "UpdatePanel" function to refresh the panel with the current data. We format "rangeText" with "LondonRangePoints" using [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring) or "Calculating..." if zero, and update "panelPrefix + "RangePoints" text with the [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) function. We similarly format and update texts for high price, low price, buy level (adding "OrderOffsetPoints \* \_Point" to "PreLondonHigh"), sell level (subtracting "OrderOffsetPoints \* \_Point" from "PreLondonLow"), balance with [AccountInfoDouble(ACCOUNT\_BALANCE)](https://www.mql5.com/en/docs/account/accountinfodouble), equity with "AccountInfoDouble(ACCOUNT\_EQUITY)", and drawdown with "dailyDrawdown".

We set the drawdown color with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) to yellow if daily drawdown exceeds "MaxDailyDrawdownPercent / 2", else white. To make the functions usable, we call them in the initialization function as follows.

```
//+------------------------------------------------------------------+
//| Initialize EA                                                    |
//+------------------------------------------------------------------+
int OnInit() {
   obj_Trade.SetExpertMagicNumber(MagicNumber); //--- Set magic number
   ArrayFree(positionList);           //--- Free position list
   CreatePanel();                     //--- Create panel
   panelVisible = true;               //--- Set panel visible
   return(INIT_SUCCEEDED);            //--- Return success
}

//+------------------------------------------------------------------+
//| Deinitialize EA                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   ObjectsDeleteAll(0, "LondonPanel_"); //--- Delete panel objects
   ArrayFree(positionList);           //--- Free position list
}
```

On the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, we set the magic number using the trade object, use the [ArrayFree](https://www.mql5.com/en/docs/array/ArrayFree) function to free the position list, call the "CreatePanel" function to create the panel, and set the panel visibility flag to true after its creation. Then, on the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) event handler, we use the [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/ObjectDeleteAll) function to delete all the objects with the specified prefix and free the position list array since we no longer need it. When we compile, we get the following outcome.

![INITIAL PANEL](https://c.mql5.com/2/157/Screenshot_2025-07-16_122518.png)

Since we have created the panel, let us now add some life to it by making the cancel button responsive, where we will destroy the panel once it is clicked. We will achieve this in the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler.

```
//+------------------------------------------------------------------+
//| Handle chart events (e.g., panel close)                          |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam) {
   if (id == CHARTEVENT_OBJECT_CLICK && sparam == panelPrefix + "Hide") { //--- Check hide click
      panelVisible = false;           //--- Set panel hidden
      ObjectsDeleteAll(0, "LondonPanel_"); //--- Delete panel objects
      ChartRedraw(0);                 //--- Redraw chart
   }
}
```

In the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler, we check if the event ID is an object click and the object is the hide or cancel button, and then set the visibility flag to false. We then delete the panel objects and redraw the chart for changes to take effect. Upon compilation, we get the following outcome.

![PANEL CLOSE](https://c.mql5.com/2/157/LONDON_SESSION.gif)

From the visualization, we can see that the panel is now all set. Let us update it so that it initializes all parts. We will need some functions to set the daily ranges and to calculate the drawdown features.

```
//+------------------------------------------------------------------+
//| Check if it's a new trading day                                  |
//+------------------------------------------------------------------+
bool IsNewDay(datetime currentBarTime) {
   MqlDateTime barTime;               //--- Bar time structure
   TimeToStruct(currentBarTime, barTime); //--- Convert time
   datetime currentDay = StringToTime(StringFormat("%04d.%02d.%02d", barTime.year, barTime.mon, barTime.day)); //--- Get current day
   if (currentDay != lastCheckedDay) { //--- Check new day
      lastCheckedDay = currentDay;    //--- Update last day
      sessionChecksDone = false;      //--- Reset checks
      noTradeToday = false;           //--- Reset no trade
      buyOrderTicket = 0;             //--- Reset buy ticket
      sellOrderTicket = 0;            //--- Reset sell ticket
      LondonRangePoints = 0.0;        //--- Reset range
      return true;                    //--- Return new day
   }
   return false;                      //--- Return not new day
}

//+------------------------------------------------------------------+
//| Update daily drawdown                                            |
//+------------------------------------------------------------------+
void UpdateDailyDrawdown() {
   static double maxEquity = 0.0;     //--- Max equity tracker
   double equity = AccountInfoDouble(ACCOUNT_EQUITY); //--- Get equity
   if (equity > maxEquity) maxEquity = equity; //--- Update max equity
   dailyDrawdown = (maxEquity - equity) / maxEquity * 100; //--- Calculate drawdown
   if (dailyDrawdown >= MaxDailyDrawdownPercent) noTradeToday = true; //--- Set no trade if exceeded
}
```

Here, we implement the "IsNewDay" function to check for a new trading day. We create a [MqlDateTime](https://www.mql5.com/en/docs/constants/structures/mqldatetime) structure "barTime" and convert "currentBarTime" to it with the [TimeToStruct](https://www.mql5.com/en/docs/dateandtime/timetostruct) function. We calculate "currentDay" using [StringToTime](https://www.mql5.com/en/docs/convert/stringtotime) and [StringFormat](https://www.mql5.com/en/docs/convert/stringformat) from "barTime.year", "barTime.mon", and "barTime.day". If "currentDay" differs from "lastCheckedDay", we update "lastCheckedDay", reset "sessionChecksDone" and "noTradeToday" to false, clear "buyOrderTicket" and "sellOrderTicket" to 0, set "LondonRangePoints" to 0.0, and return true; otherwise, return false. This function ensures daily resets for session analysis and trading flags.

Next, we implement the "UpdateDailyDrawdown" function to monitor daily risk. We use a static "maxEquity" initialized to 0.0 to track peak equity. We get "equity" with [AccountInfoDouble](https://www.mql5.com/en/docs/account/accountinfodouble) using "ACCOUNT\_EQUITY", update "maxEquity" if "equity" is higher, and calculate "dailyDrawdown" as the percentage drop from "maxEquity". If "dailyDrawdown" meets or exceeds "MaxDailyDrawdownPercent", we set "noTradeToday" to true to halt trading, providing essential drawdown protection. To make the functions usable, we call them in the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler to populate the panel with updated data.

```
//+------------------------------------------------------------------+
//| Main tick handler                                                |
//+------------------------------------------------------------------+
void OnTick() {
   datetime currentBarTime = iTime(_Symbol, _Period, 0); //--- Get current bar time
   IsNewDay(currentBarTime);          //--- Check new day

   UpdatePanel();                     //--- Update panel
   UpdateDailyDrawdown();             //--- Update drawdown
}
```

Here, we just call the new day function to set the range parameters, update the panel, and daily drawdown levels, respectively. When we run the program now, we have the following outcome.

![INITIALIZED PANEL](https://c.mql5.com/2/157/Screenshot_2025-07-16_124501.png)

From the visualization, we can see the panel is now populated with recent data and shows the status. Let us now move on to the more complex logic of defining the session ranges. Let's first check the trading conditions and place the orders if the trading conditions are met, and then manage the positions later. So we will need some functions to first let us define the range, visualize the ranges, and then place the orders. Here is the logic we use to achieve that.

```
//+------------------------------------------------------------------+
//| Fixed lot size                                                   |
//+------------------------------------------------------------------+
double CalculateLotSize(double entryPrice, double stopLossPrice) {
   return NormalizeDouble(inpTradeLotsize, 2); //--- Normalize lot size
}

//+------------------------------------------------------------------+
//| Calculate session range (high-low) in points                     |
//+------------------------------------------------------------------+
double GetRange(datetime startTime, datetime endTime, double &highVal, double &lowVal, datetime &highTime, datetime &lowTime) {
   int startBar = iBarShift(_Symbol, _Period, startTime, true); //--- Get start bar
   int endBar = iBarShift(_Symbol, _Period, endTime, true); //--- Get end bar
   if (startBar == -1 || endBar == -1 || startBar < endBar) return -1; //--- Invalid bars

   int highestBar = iHighest(_Symbol, _Period, MODE_HIGH, startBar - endBar + 1, endBar); //--- Get highest bar
   int lowestBar = iLowest(_Symbol, _Period, MODE_LOW, startBar - endBar + 1, endBar); //--- Get lowest bar
   highVal = iHigh(_Symbol, _Period, highestBar); //--- Set high value
   lowVal = iLow(_Symbol, _Period, lowestBar); //--- Set low value
   highTime = iTime(_Symbol, _Period, highestBar); //--- Set high time
   lowTime = iTime(_Symbol, _Period, lowestBar); //--- Set low time
   return (highVal - lowVal) / _Point; //--- Return range in points
}

//+------------------------------------------------------------------+
//| Place pending buy/sell stop orders                               |
//+------------------------------------------------------------------+
void PlacePendingOrders(double preLondonHigh, double preLondonLow, datetime sessionID) {
   double buyPrice = preLondonHigh + OrderOffsetPoints * _Point; //--- Calculate buy price
   double sellPrice = preLondonLow - OrderOffsetPoints * _Point; //--- Calculate sell price
   double slPoints = StopLossPoints; //--- Set SL points
   double buySL = buyPrice - slPoints * _Point; //--- Calculate buy SL
   double sellSL = sellPrice + slPoints * _Point; //--- Calculate sell SL
   double tpPoints = slPoints * RRRatio; //--- Calculate TP points
   double buyTP = buyPrice + tpPoints * _Point; //--- Calculate buy TP
   double sellTP = sellPrice - tpPoints * _Point; //--- Calculate sell TP
   double lotSizeBuy = CalculateLotSize(buyPrice, buySL); //--- Calculate buy lot
   double lotSizeSell = CalculateLotSize(sellPrice, sellSL); //--- Calculate sell lot

   if (TradeType == TRADE_ALL || TradeType == TRADE_BUY_ONLY) { //--- Check buy trade
      obj_Trade.BuyStop(lotSizeBuy, buyPrice, _Symbol, buySL, buyTP, 0, 0, "Buy Stop - London"); //--- Place buy stop
      buyOrderTicket = obj_Trade.ResultOrder(); //--- Get buy ticket
   }

   if (TradeType == TRADE_ALL || TradeType == TRADE_SELL_ONLY) { //--- Check sell trade
      obj_Trade.SellStop(lotSizeSell, sellPrice, _Symbol, sellSL, sellTP, 0, 0, "Sell Stop - London"); //--- Place sell stop
      sellOrderTicket = obj_Trade.ResultOrder(); //--- Get sell ticket
   }
}

//+------------------------------------------------------------------+
//| Draw session ranges on the chart                                 |
//+------------------------------------------------------------------+
void DrawSessionRanges(datetime preLondonStart, datetime londonEnd) {
   string sessionID = "Sess_" + IntegerToString(lastCheckedDay); //--- Session ID

   string preRectName = "PreRect_" + sessionID; //--- Rectangle name
   ObjectCreate(0, preRectName, OBJ_RECTANGLE, 0, PreLondonHighTime, PreLondonHigh, PreLondonLowTime, PreLondonLow); //--- Create rectangle
   ObjectSetInteger(0, preRectName, OBJPROP_COLOR, clrTeal); //--- Set color
   ObjectSetInteger(0, preRectName, OBJPROP_FILL, true); //--- Enable fill
   ObjectSetInteger(0, preRectName, OBJPROP_BACK, true); //--- Set background

   string preTopLineName = "PreTopLine_" + sessionID; //--- Top line name
   ObjectCreate(0, preTopLineName, OBJ_TREND, 0, preLondonStart, PreLondonHigh, londonEnd, PreLondonHigh); //--- Create top line
   ObjectSetInteger(0, preTopLineName, OBJPROP_COLOR, clrBlack); //--- Set color
   ObjectSetInteger(0, preTopLineName, OBJPROP_WIDTH, 1); //--- Set width
   ObjectSetInteger(0, preTopLineName, OBJPROP_RAY_RIGHT, false); //--- Disable ray
   ObjectSetInteger(0, preTopLineName, OBJPROP_BACK, true); //--- Set background

   string preBotLineName = "PreBottomLine_" + sessionID; //--- Bottom line name
   ObjectCreate(0, preBotLineName, OBJ_TREND, 0, preLondonStart, PreLondonLow, londonEnd, PreLondonLow); //--- Create bottom line
   ObjectSetInteger(0, preBotLineName, OBJPROP_COLOR, clrRed); //--- Set color
   ObjectSetInteger(0, preBotLineName, OBJPROP_WIDTH, 1); //--- Set width
   ObjectSetInteger(0, preBotLineName, OBJPROP_RAY_RIGHT, false); //--- Disable ray
   ObjectSetInteger(0, preBotLineName, OBJPROP_BACK, true); //--- Set background
}
```

To enable range calculation, order placement, and chart drawing, we start with the "CalculateLotSize" function to compute fixed lot sizes, taking "entryPrice" and "stopLossPrice" as parameters (unused for fixed sizing). We return "inpTradeLotsize" normalized to 2 decimals with [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble), ensuring consistent lot sizes for all trades. You can use another decimal depending on your account type.

Next, we create the "GetRange" function to calculate the pre-London session range. We get "startBar" and "endBar" with the [iBarShift](https://www.mql5.com/en/docs/series/ibarshift) function using "startTime" and "endTime", returning -1 if invalid or "startBar" < "endBar". We find "highestBar" with [iHighest](https://www.mql5.com/en/docs/series/ihighest) on [MODE\_HIGH](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes#enum_seriesmode) and "lowestBar" with "iLowest" on [MODE\_LOW](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes#enum_seriesmode) over the bar range. We set "highVal" with "iHigh" on "highestBar", "lowVal" with "iLow" on "lowestBar", "highTime" with "iTime" on "highestBar", and "lowTime" with "iTime" on "lowestBar". We return the range as the "(highVal-lowVal) / [\_Point](https://www.mql5.com/en/docs/predefined/_point)" value.

We then define the "PlacePendingOrders" function to set up buy and sell stops. We calculate "buyPrice" as "preLondonHigh + OrderOffsetPoints \* \_Point" and "sellPrice" as "preLondonLow - OrderOffsetPoints \* \_Point". We set "slPoints" to "StopLossPoints", "buySL" as "buyPrice - slPoints \* \_Point", "sellSL" as "sellPrice + slPoints \* [\_Point](https://www.mql5.com/en/docs/predefined/_point)", "tpPoints" as "slPoints \* RRRatio", "buyTP" as "buyPrice + tpPoints \* \_Point", and "sellTP" as "sellPrice - tpPoints \* \_Point". We compute "lotSizeBuy" and "lotSizeSell" with "CalculateLotSize".

If "TradeType" is "TRADE\_ALL" or "TRADE\_BUY\_ONLY", we place a buy stop with "obj\_Trade.BuyStop" using "lotSizeBuy", "buyPrice", "buySL", "buyTP", and label "Buy Stop - London", storing the ticket in "buyOrderTicket" from "ResultOrder". Similarly, for sell if "TRADE\_ALL" or "TRADE\_SELL\_ONLY".

Finally, we implement the "DrawSessionRanges" function to visualize the session on the chart. We create "sessionID" as "Sess\_" plus "lastCheckedDay" with the [IntegerToString](https://www.mql5.com/en/docs/convert/IntegerToString) function. For the rectangle "preRectName" as "PreRect\_" plus "sessionID", we use [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) as [OBJ\_RECTANGLE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) from "PreLondonHighTime", "PreLondonHigh" to "PreLondonLowTime", "PreLondonLow", setting [OBJPROP\_COLOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to "clrTeal", "OBJPROP\_FILL" to true, and "OBJPROP\_BACK" to true.

For the top line "preTopLineName" as "PreTopLine\_" plus "sessionID", we create [OBJ\_TREND](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) from "preLondonStart", "PreLondonHigh" to "londonEnd", "PreLondonHigh", setting "OBJPROP\_COLOR" to "clrBlack", "OBJPROP\_WIDTH" to 1, "OBJPROP\_RAY\_RIGHT" to false, and "OBJPROP\_BACK" to true. Similarly, for the bottom line "preBotLineName" as "PreBottomLine\_" plus "sessionID" from "preLondonStart", "PreLondonLow" to "londonEnd", "PreLondonLow", with red color. We can now define a function to check for the trading conditions using these functions.

```
//+------------------------------------------------------------------+
//| Check trading conditions and place orders                        |
//+------------------------------------------------------------------+
void CheckTradingConditions(datetime currentTime) {
   MqlDateTime timeStruct;            //--- Time structure
   TimeToStruct(currentTime, timeStruct); //--- Convert time
   datetime today = StringToTime(StringFormat("%04d.%02d.%02d", timeStruct.year, timeStruct.mon, timeStruct.day)); //--- Get today

   datetime preLondonStart = today + PreLondonStartHour * 3600 + PreLondonStartMinute * 60; //--- Pre-London start
   datetime londonStart = today + LondonStartHour * 3600 + LondonStartMinute * 60; //--- London start
   datetime londonEnd = today + LondonEndHour * 3600 + LondonEndMinute * 60; //--- London end
   analysisTime = londonStart;        //--- Set analysis time

   if (currentTime < analysisTime) return; //--- Exit if before analysis

   double preLondonRange = GetRange(preLondonStart, currentTime, PreLondonHigh, PreLondonLow, PreLondonHighTime, PreLondonLowTime); //--- Get range
   if (preLondonRange < MinRangePoints || preLondonRange > MaxRangePoints) { //--- Check range limits
      noTradeToday = true;            //--- Set no trade
      sessionChecksDone = true;       //--- Set checks done
      DrawSessionRanges(preLondonStart, londonEnd); //--- Draw ranges
      return;                         //--- Exit
   }

   LondonRangePoints = preLondonRange; //--- Set range points
   PlacePendingOrders(PreLondonHigh, PreLondonLow, today); //--- Place orders
   noTradeToday = true;               //--- Set no trade
   sessionChecksDone = true;          //--- Set checks done
   DrawSessionRanges(preLondonStart, londonEnd); //--- Draw ranges
}
```

We implement the "CheckTradingConditions" function to evaluate session conditions and place orders in our London Session Breakout System. We create a [MqlDateTime](https://www.mql5.com/en/docs/constants/structures/mqldatetime) structure, "timeStruct," and convert the current time to it with the [TimeToStruct](https://www.mql5.com/en/docs/dateandtime/timetostruct) function. We calculate "today" using "StringToTime" and "StringFormat" from "timeStruct.year", "timeStruct.mon", and "timeStruct.day". We set "preLondonStart" as "today" plus "PreLondonStartHour" and "PreLondonStartMinute" in seconds, "londonStart" as "today" plus "LondonStartHour" and "LondonStartMinute", and "londonEnd" as "today" plus "LondonEndHour" and "LondonEndMinute". We assign "analysisTime" to "londonStart" and exit if the current time is before it.

We get "preLondonRange" with the "GetRange" function, passing "preLondonStart", "currentTime", and references for "PreLondonHigh", "PreLondonLow", "PreLondonHighTime", and "PreLondonLowTime". If "preLondonRange" is below "MinRangePoints" or above "MaxRangePoints", we set "noTradeToday" and "sessionChecksDone" to true, call "DrawSessionRanges" with "preLondonStart" and "londonEnd", and exit. Otherwise, we set "LondonRangePoints" to "preLondonRange", call "PlacePendingOrders" with "PreLondonHigh", "PreLondonLow", and "today", set "noTradeToday" and "sessionChecksDone" to true, and call "DrawSessionRanges", ensuring trades occur only within valid session ranges. We can call the function now in the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler for the signals.

```
if (!noTradeToday && !sessionChecksDone) { //--- Check trading conditions
   CheckTradingConditions(TimeCurrent()); //--- Check conditions
}
```

If there are no trades done for today and no session checks done already, we call the function to check for the conditions for the current time. When we run the program, we get the following outcome.

![RANGES SET](https://c.mql5.com/2/157/Screenshot_2025-07-16_131448.png)

We can see that we set the ranges and placed the pending orders. The range is 100 points, which meets our trading conditions. We can now move on to managing the trading trades, but first, let us check and delete the pending orders when one is activated, and add the positions to the position list for management.

```
//+------------------------------------------------------------------+
//| Delete opposite pending order when one is filled                 |
//+------------------------------------------------------------------+
void CheckAndDeleteOppositeOrder() {
   if (!DeleteOppositeOrder || TradeType != TRADE_ALL) return; //--- Exit if not applicable

   bool buyOrderExists = false;       //--- Buy exists flag
   bool sellOrderExists = false;      //--- Sell exists flag

   for (int i = OrdersTotal() - 1; i >= 0; i--) { //--- Iterate through orders
      ulong orderTicket = OrderGetTicket(i); //--- Get ticket
      if (OrderSelect(orderTicket)) { //--- Select order
         if (OrderGetString(ORDER_SYMBOL) == _Symbol && OrderGetInteger(ORDER_MAGIC) == MagicNumber) { //--- Check symbol and magic
            if (orderTicket == buyOrderTicket) buyOrderExists = true; //--- Set buy exists
            if (orderTicket == sellOrderTicket) sellOrderExists = true; //--- Set sell exists
         }
      }
   }

   if (!buyOrderExists && sellOrderExists && sellOrderTicket != 0) { //--- Check delete sell
      obj_Trade.OrderDelete(sellOrderTicket); //--- Delete sell order
   } else if (!sellOrderExists && buyOrderExists && buyOrderTicket != 0) { //--- Check delete buy
      obj_Trade.OrderDelete(buyOrderTicket); //--- Delete buy order
   }
}

//+------------------------------------------------------------------+
//| Add position to tracking list when opened                        |
//+------------------------------------------------------------------+
void AddPositionToList(ulong ticket, double openPrice, double londonRange, datetime sessionID) {
   if (londonRange <= 0) return;      //--- Exit if invalid range
   int index = ArraySize(positionList); //--- Get current size
   ArrayResize(positionList, index + 1); //--- Resize array
   positionList[index].ticket = ticket; //--- Set ticket
   positionList[index].openPrice = openPrice; //--- Set open price
   positionList[index].londonRange = londonRange; //--- Set range
   positionList[index].sessionID = sessionID; //--- Set session ID
   positionList[index].trailingActive = false; //--- Set trailing inactive
}
```

We start with the "CheckAndDeleteOppositeOrder" function to remove the opposite pending order when one is triggered. We exit early if "DeleteOppositeOrder" is false or "TradeType" is not "TRADE\_ALL". We initialize "buyOrderExists" and "sellOrderExists" as false. We loop backward through orders with [OrdersTotal](https://www.mql5.com/en/docs/trading/orderstotal) and [OrderGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket), selecting each with "OrderSelect". If the order matches "\_Symbol" and "MagicNumber" via "OrderGetString" and [OrderGetInteger](https://www.mql5.com/en/docs/trading/positiongetinteger), we set "buyOrderExists" or "sellOrderExists" if the ticket matches "buyOrderTicket" or "sellOrderTicket".

If the buy order is gone and the sell order exists, we delete "sellOrderTicket" with "obj\_Trade.OrderDelete"; similarly, if the sell order is gone and the buy order exists. This function will ensure only one direction of trade after activation.

Next, we create the "AddPositionToList" function to track opened positions. We exit if "londonRange" is <= 0 because the range is not set yet. We get the current "index" with [ArraySize](https://www.mql5.com/en/docs/array/arraysize) of "positionList", resize it with [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) to add one slot, and set "positionList\[index\]. ticket", "openPrice", "londonRange", "sessionID", and "trailingActive" to false, helping to maintain a list for managing trailing stops and session-specific data. We can now implement this logic on the tick event handler.

```
CheckAndDeleteOppositeOrder();     //--- Delete opposite order

// Add untracked positions
for (int i = 0; i < PositionsTotal(); i++) { //--- Iterate through positions
   ulong ticket = PositionGetTicket(i); //--- Get ticket
   if (PositionSelectByTicket(ticket) && PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == MagicNumber) { //--- Check position
      bool tracked = false;        //--- Tracked flag
      for (int j = 0; j < ArraySize(positionList); j++) { //--- Check list
         if (positionList[j].ticket == ticket) tracked = true; //--- Set tracked
      }
      if (!tracked) {              //--- If not tracked
         double openPrice = PositionGetDouble(POSITION_PRICE_OPEN); //--- Get open price
         AddPositionToList(ticket, openPrice, LondonRangePoints, lastCheckedDay); //--- Add to list
      }
   }
}
```

Here, we call the "CheckAndDeleteOppositeOrder" function to manage pending orders, ensuring that if one direction's order is filled, the opposite is deleted as per the "DeleteOppositeOrder" input, preventing conflicting trades.

Next, we add untracked positions to the "positionList" to ensure all relevant open trades are monitored for trailing stops. We loop through all positions with [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) and [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket) to get each "ticket". If [PositionSelectByTicket](https://www.mql5.com/en/docs/trading/positionselectbyticket) succeeds and the position matches [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol) and "MagicNumber" using "PositionGetString" and "PositionGetInteger", we set a "tracked" flag to false and check the "positionList" array with "ArraySize" and an inner loop to see if the "ticket" is already in "positionList\[j\].ticket". If not "tracked", we get "openPrice" with [PositionGetDouble](https://www.mql5.com/en/docs/trading/positiongetdouble) using [POSITION\_PRICE\_OPEN](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_double) and call "AddPositionToList" with "ticket", "openPrice", "LondonRangePoints", and "lastCheckedDay". This ensures every matching position is added to the list for management without duplicates. Here is the outcome.

![PENDING ORDER DELETED](https://c.mql5.com/2/157/Screenshot_2025-07-16_134539.png)

Since everything is perfect up to this point, we can now define the function to manage these positions as follows.

```
//+------------------------------------------------------------------+
//| Remove position from tracking list when closed                   |
//+------------------------------------------------------------------+
void RemovePositionFromList(ulong ticket) {
   for (int i = 0; i < ArraySize(positionList); i++) { //--- Iterate through list
      if (positionList[i].ticket == ticket) { //--- Match ticket
         for (int j = i; j < ArraySize(positionList) - 1; j++) { //--- Shift elements
            positionList[j] = positionList[j + 1]; //--- Copy next
         }
         ArrayResize(positionList, ArraySize(positionList) - 1); //--- Resize array
         break;                   //--- Exit loop
      }
   }
}

//+------------------------------------------------------------------+
//| Manage trailing stops                                            |
//+------------------------------------------------------------------+
void ManagePositions() {
   if (PositionsTotal() == 0 || !UseTrailing) return; //--- Exit if no positions or no trailing
   isTrailing = false;                //--- Reset trailing flag

   double currentBid = SymbolInfoDouble(_Symbol, SYMBOL_BID); //--- Get bid
   double currentAsk = SymbolInfoDouble(_Symbol, SYMBOL_ASK); //--- Get ask
   double point = _Point;             //--- Get point value

   for (int i = 0; i < ArraySize(positionList); i++) { //--- Iterate through positions
      ulong ticket = positionList[i].ticket; //--- Get ticket
      if (!PositionSelectByTicket(ticket)) { //--- Select position
         RemovePositionFromList(ticket); //--- Remove if not selected
         continue;                    //--- Skip
      }

      if (PositionGetString(POSITION_SYMBOL) != _Symbol || PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue; //--- Skip if not matching

      double openPrice = positionList[i].openPrice; //--- Get open price
      long positionType = PositionGetInteger(POSITION_TYPE); //--- Get type
      double currentPrice = (positionType == POSITION_TYPE_BUY) ? currentBid : currentAsk; //--- Get current price
      double profitPoints = (positionType == POSITION_TYPE_BUY) ? (currentPrice - openPrice) / point : (openPrice - currentPrice) / point; //--- Calculate profit points

      if (profitPoints >= MinProfitPoints + TrailingPoints) { //--- Check for trailing
         double newSL = 0.0;          //--- New SL variable
         if (positionType == POSITION_TYPE_BUY) { //--- Buy position
            newSL = currentPrice - TrailingPoints * point; //--- Calculate new SL
         } else {                     //--- Sell position
            newSL = currentPrice + TrailingPoints * point; //--- Calculate new SL
         }
         double currentSL = PositionGetDouble(POSITION_SL); //--- Get current SL
         if ((positionType == POSITION_TYPE_BUY && newSL > currentSL + point) || (positionType == POSITION_TYPE_SELL && newSL < currentSL - point)) { //--- Check move condition
            if (obj_Trade.PositionModify(ticket, NormalizeDouble(newSL, _Digits), PositionGetDouble(POSITION_TP))) { //--- Modify position
               positionList[i].trailingActive = true; //--- Set trailing active
               isTrailing = true;        //--- Set global trailing
            }
         }
      }
   }
}
```

Here, we implement functions to remove closed positions from the tracking list and manage trailing stops. We start with the "RemovePositionFromList" function to clean up the "positionList" array when a position closes, taking "ticket" as a parameter. We loop through "positionList" with [ArraySize](https://www.mql5.com/en/docs/array/arraysize), and if "positionList\[i\].ticket" matches "ticket", we shift subsequent elements with an inner loop copying "positionList\[j + 1\]" to "positionList\[j\]", then resize the array with [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) to reduce its size by one, and [break](https://www.mql5.com/en/docs/basis/operators/break) the loop. This function will ensure the list remains current, avoiding unnecessary checks on closed positions, especially when we trail and close them.

Next, we create the "ManagePositions" function to handle trailing stops for open trades. We exit early if [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) is 0 or "UseTrailing" is false. We reset "isTrailing" to false, get "currentBid" and "currentAsk" with [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) using "SYMBOL\_BID" and "SYMBOL\_ASK", and "point" as "\_Point". We loop through "positionList" with "ArraySize", getting "ticket" and selecting it with the [PositionSelectByTicket](https://www.mql5.com/en/docs/trading/positionselectbyticket) function. If selection fails, we call "RemovePositionFromList" and continue. If the position doesn't match "\_Symbol" or "MagicNumber" with "PositionGetString" and "PositionGetInteger", we skip. We get "openPrice" from "positionList\[i\]", "positionType" with [PositionGetInteger](https://www.mql5.com/en/docs/trading/positiongetinteger), "currentPrice" based on type, and "profitPoints" as the difference divided by point.

If "profitPoints" meets or exceeds "MinProfitPoints + TrailingPoints", we calculate "newSL" as "currentPrice - TrailingPoints \* point" for buys or plus for sells. We get "currentSL" with [PositionGetDouble](https://www.mql5.com/en/docs/trading/positiongetdouble), and if "newSL" is better than "currentSL" by at least "point", we modify the position with "obj\_Trade.PositionModify" using normalized "newSL" and current TP from "PositionGetDouble". On success, we set "positionList\[i\].trailingActive" and "isTrailing" to true, allowing us to dynamically adjust stops to lock in profits while allowing winning trades to run. Now, just call the function to manage the positions on every tick. Upon compilation, we have the following outcome.

![POSITION MANAGEMENT](https://c.mql5.com/2/157/Screenshot_2025-07-16_134901.png)

From the image, we can see that we check for trading conditions, place orders, and manage them as per the trading strategy, hence achieving our objectives. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/157/Screenshot_2025-07-16_162229.png)

Backtest report:

![REPORT](https://c.mql5.com/2/157/Screenshot_2025-07-16_162156.png)

### Conclusion

In conclusion, we have developed a [London Session Breakout System](https://www.mql5.com/go?link=https://tradingstrategyguides.com/london-breakout-strategy/ "https://tradingstrategyguides.com/london-breakout-strategy/") in MQL5 that analyzes pre-session ranges to place pending orders with customizable risk-to-reward ratios, trailing stops, and multi-trade limits, complemented by a control panel for real-time monitoring of ranges, levels, and drawdown. Through modular components like the "PositionInfo" [structure](https://www.mql5.com/en/docs/basis/types/classes), this program offers a disciplined approach to breakout trading that you can refine by adjusting session times or risk parameters.

Disclaimer: This article is for educational purposes only. Trading carries significant financial risks, and market volatility may result in losses. Thorough backtesting and careful risk management are crucial before deploying this program in live markets.

By leveraging the concepts and implementation presented, you can adapt this breakout system to your trading style, empowering your algorithmic strategies. Happy trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18867.zip "Download all attachments in the single ZIP archive")

[1.\_London\_Breakout\_EA.mq5](https://www.mql5.com/en/articles/download/18867/1._london_breakout_ea.mq5 "Download 1._London_Breakout_EA.mq5")(28.52 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/491635)**
(14)


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
10 Aug 2025 at 22:20

**Kyle Young Sangster [#](https://www.mql5.com/en/forum/491635#comment_57771750):**

I downloaded and ran this code as-is through the [Strategy Tester](https://www.mql5.com/en/articles/239 "Article: The Fundamentals of Testing in MetaTrader 5 "). It finds the range each day, and draws the box on the chart. However, it doesn't take a trade each day, (assuming that it should). Over the course of 1.5 months, it took only 3 trades.

The second issue is the high / low levels and the buy / sell levels in the control panel do not update.

[https://c.mql5.com/3/471/Screenshot_2025-08-10_at_13.11.07__1.png](https://c.mql5.com/3/471/Screenshot_2025-08-10_at_13.11.07__1.png "https://c.mql5.com/3/471/Screenshot_2025-08-10_at_13.11.07__1.png")

The  high /low range levels are clearly being found on the chart, so I'm guessing the buy / sell levels should also be displayed on the chart and updated in the control panel, as they are derived directly from the high / low range levels.

What suggestions do you have to get this working correctly?

Thanks in advance.

As for your second issue, the article explains that, but assuming that your issue emanates from poor testing data and giving a hint, when the range is in calculation, you will always see "Calculating..." status until there is enough data to set the London range session or whichever session you define in the inputs. Assuming you are using the default settings, with prelondon hours being 3, and your time from the shared screenshot is 13, Feb, 2 bars after 22:00 which is 2\*15 minutes = 30, hence giving 22:30, is outside the range calculation time, so the data on the panel should be  visible still since the previous set range is still in play unless the first session has not yet been found, and will be cleared as the range calculation is reached from midnight. See below:

```
const int PreLondonStartHour = 3; //--- Fixed Pre-London Start Hour
const int PreLondonStartMinute = 0; //--- Fixed Pre-London Start Minute
```

You might need to see the logic below for finding the range

```
//+------------------------------------------------------------------+
//| Check trading conditions and place orders                        |
//+------------------------------------------------------------------+
void CheckTradingConditions(datetime currentTime) {
   MqlDateTime timeStruct;            //--- Time structure
   TimeToStruct(currentTime, timeStruct); //--- Convert time
   datetime today = StringToTime(StringFormat("%04d.%02d.%02d", timeStruct.year, timeStruct.mon, timeStruct.day)); //--- Get today

   datetime preLondonStart = today + PreLondonStartHour * 3600 + PreLondonStartMinute * 60; //--- Pre-London start
   datetime londonStart = today + LondonStartHour * 3600 + LondonStartMinute * 60; //--- London start
   datetime londonEnd = today + LondonEndHour * 3600 + LondonEndMinute * 60; //--- London end
   analysisTime = londonStart;        //--- Set analysis time

   if (currentTime < analysisTime) return; //--- Exit if before analysis

   double preLondonRange = GetRange(preLondonStart, currentTime, PreLondonHigh, PreLondonLow, PreLondonHighTime, PreLondonLowTime); //--- Get range
   if (preLondonRange < MinRangePoints || preLondonRange > MaxRangePoints) { //--- Check range limits
      noTradeToday = true;            //--- Set no trade
      sessionChecksDone = true;       //--- Set checks done
      DrawSessionRanges(preLondonStart, londonEnd); //--- Draw ranges
      return;                         //--- Exit
   }

   LondonRangePoints = preLondonRange; //--- Set range points
   PlacePendingOrders(PreLondonHigh, PreLondonLow, today); //--- Place orders
   noTradeToday = true;               //--- Set no trade
   sessionChecksDone = true;          //--- Set checks done
   DrawSessionRanges(preLondonStart, londonEnd); //--- Draw ranges
}
```

And how it is set.

```
//+------------------------------------------------------------------+
//| Update panel with current data                                   |
//+------------------------------------------------------------------+
void UpdatePanel() {
   string rangeText = "Range (points): " + (LondonRangePoints > 0 ? DoubleToString(LondonRangePoints, 0) : "Calculating..."); //--- Format range text
   ObjectSetString(0, panelPrefix + "RangePoints", OBJPROP_TEXT, rangeText); //--- Update range text

   //---

}
```

See the image below, though we don't know the year of your testing, we will take the 2025, if it is 2020 as in your case, we don't have quality data for that so either way, we use 2025 and thus range calculation should start at midnight.

[![23:55](https://c.mql5.com/3/471/Screenshot_2025-08-11_010921__1.png)](https://c.mql5.com/3/471/Screenshot_2025-08-11_010921.png)

From the image, you can see the data at 23:55 is stil intact. However, when it is midnight, we shpuld reset. See below.

[![MIDNIGHT DATA 00:00](https://c.mql5.com/3/471/Screenshot_2025-08-11_010951__1.png)](https://c.mql5.com/3/471/Screenshot_2025-08-11_010951.png "https://c.mql5.com/3/471/Screenshot_2025-08-11_010951.png")

You can see that we reset the data at midnight for the other range calculation. Actaully, when the range calculation is done, the visualization can help you know what really went on. For example, in your case using the default settings, we will see the raneg bars from 0300 hrs to 0800 hrs because that is what we defined. See below:

[![RANGE HOURS](https://c.mql5.com/3/471/Screenshot_2025-08-11_011512__1.png)](https://c.mql5.com/3/471/Screenshot_2025-08-11_011512.png "https://c.mql5.com/3/471/Screenshot_2025-08-11_011512.png")

Hope this clarifies things again. You can adjust everything as per your trading style. To avoid the issues you are facing, it is advisable to use reliable testing data. Thanks.

![Kyle Young Sangster](https://c.mql5.com/avatar/2024/11/6736F47E-D362.png)

**[Kyle Young Sangster](https://www.mql5.com/en/users/ksngstr)**
\|
10 Aug 2025 at 22:34

Where in the code did you intent to use the variable "MaxOpenTrades"? It's defined but never referenced.


![Kyle Young Sangster](https://c.mql5.com/avatar/2024/11/6736F47E-D362.png)

**[Kyle Young Sangster](https://www.mql5.com/en/users/ksngstr)**
\|
10 Aug 2025 at 22:52

**Allan Munene Mutiiria [#](https://www.mql5.com/en/forum/491635#comment_57772653):**

As for your second issue, the article explains that, but assuming that your issue emanates from poor testing data and giving a hint, when the range is in calculation, you will always see "Calculating..." status until there is enough data to set the London range session or whichever session you define in the inputs. Assuming you are using the default settings, with prelondon hours being 3, and your time from the shared screenshot is 13, Feb, 2 bars after 22:00 which is 2\*15 minutes = 30, hence giving 22:30, is outside the range calculation time, so the data on the panel should be  visible still since the previous set range is still in play unless the first session has not yet been found, and will be cleared as the range calculation is reached from midnight. See below:

You might need to see the logic below for finding the range

And how it is set.

See the image below, though we don't know the year of your testing, we will take the 2025, if it is 2020 as in your case, we don't have quality data for that so either way, we use 2025 and thus range calculation should start at midnight.

From the image, you can see the data at 23:55 is stil intact. However, when it is midnight, we shpuld reset. See below.

You can see that we reset the data at midnight for the other range calculation. Actaully, when the range calculation is done, the visualization can help you know what really went on. For example, in your case using the default settings, we will see the raneg bars from 0300 hrs to 0800 hrs because that is what we defined. See below:

Hope this clarifies things again. You can adjust everything as per your trading style. To avoid the issues you are facing, it is advisable to use reliable testing data. Thanks.

Thanks very much for the comprehensive reply.

Yes, I did read the article, followed along coding my own copy until I ran into the issues I outlined. What I saw was the panel didn't update, even during the default times. My screenshot was meant to show that, even though the box had been drawn on the chart, the data was collected, but the panel had not been updated. Additionally, there was no error messages in logs regarding invalid prices or levels.

I added logging messages to my version; from that I can see that the panel doesn't update when the range is too big or too small; so that may be part of the reason.

I will double-check the testing data quality. And thank you for pointing out which pair you tested on; I will certainly make adjustments for my chosen pairs.

Many thanks for your help.

![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
10 Aug 2025 at 23:49

**Kyle Young Sangster [#](https://www.mql5.com/en/forum/491635/page2#comment_57772723):**

Thanks very much for the comprehensive reply.

Yes, I did read the article, followed along coding my own copy until I ran into the issues I outlined. What I saw was the panel didn't update, even during the default times. My screenshot was meant to show that, even though the box had been drawn on the chart, the data was collected, but the panel had not been updated. Additionally, there was no error messages in logs regarding invalid prices or levels.

I added logging messages to my version; from that I can see that the panel doesn't update when the range is too big or too small; so that may be part of the reason.

I will double-check the testing data quality. And thank you for pointing out which pair you tested on; I will certainly make adjustments for my chosen pairs.

Many thanks for your help.

Sure. Welcome.

![Torsten Busch](https://c.mql5.com/avatar/2025/4/681143F8-FB43.png)

**[Torsten Busch](https://www.mql5.com/en/users/tortibusch)**
\|
11 Nov 2025 at 21:01

Thank you for sharing your code with us.

As I have written session dependent EAs myself, I can tell you that the code only works if your broker is always in the GMT+1 time zone and also uses British [Summer Time](https://www.mql5.com/en/docs/dateandtime/timedaylightsavings "Reference book MQL5 : TimeDaylightSavings function").

In all other cases your start time will not work. Why? Because the London session starts at 8:00 a.m. UK time. In winter this is 8:00 GMT and in summer 7:00 GMT.

TimeCurrent() does not return your local time, but always the time from the trade server.

![MQL5 Wizard Techniques you should know (Part 76):  Using Patterns of Awesome Oscillator and the Envelope Channels with Supervised Learning](https://c.mql5.com/2/158/18878-mql5-wizard-techniques-you-logo.png)[MQL5 Wizard Techniques you should know (Part 76): Using Patterns of Awesome Oscillator and the Envelope Channels with Supervised Learning](https://www.mql5.com/en/articles/18878)

We follow up on our last article, where we introduced the indicator couple of the Awesome-Oscillator and the Envelope Channel, by looking at how this pairing could be enhanced with Supervised Learning. The Awesome-Oscillator and Envelope-Channel are a trend-spotting and support/resistance complimentary mix. Our supervised learning approach is a CNN that engages the Dot Product Kernel with Cross-Time-Attention to size its kernels and channels. As per usual, this is done in a custom signal class file that works with the MQL5 wizard to assemble an Expert Advisor.

![Reimagining Classic Strategies (Part 14): Multiple Strategy Analysis](https://c.mql5.com/2/158/18847-reimagining-classic-strategies-logo.png)[Reimagining Classic Strategies (Part 14): Multiple Strategy Analysis](https://www.mql5.com/en/articles/18847)

In this article, we continue our exploration of building an ensemble of trading strategies and using the MT5 genetic optimizer to tune the strategy parameters. Today, we analyzed the data in Python, showing our model could better predict which strategy would outperform, achieving higher accuracy than forecasting market returns directly. However, when we tested our application with its statistical models, our performance levels fell dismally. We subsequently discovered that the genetic optimizer unfortunately favored highly correlated strategies, prompting us to revise our method to keep vote weights fixed and focus optimization on indicator settings instead.

![Population ADAM (Adaptive Moment Estimation)](https://c.mql5.com/2/104/Adaptive_Moment_Estimation___LOGO.png)[Population ADAM (Adaptive Moment Estimation)](https://www.mql5.com/en/articles/16443)

The article presents the transformation of the well-known and popular ADAM gradient optimization method into a population algorithm and its modification with the introduction of hybrid individuals. The new approach allows creating agents that combine elements of successful decisions using probability distribution. The key innovation is the formation of hybrid population individuals that adaptively accumulate information from the most promising solutions, increasing the efficiency of search in complex multidimensional spaces.

![MQL5 Trading Tools (Part 5): Creating a Rolling Ticker Tape for Real-Time Symbol Monitoring](https://c.mql5.com/2/158/18844-mql5-trading-tools-part-5-creating-logo.png)[MQL5 Trading Tools (Part 5): Creating a Rolling Ticker Tape for Real-Time Symbol Monitoring](https://www.mql5.com/en/articles/18844)

In this article, we develop a rolling ticker tape in MQL5 for real-time monitoring of multiple symbols, displaying bid prices, spreads, and daily percentage changes with scrolling effects. We implement customizable fonts, colors, and scroll speeds to highlight price movements and trends effectively.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=elejtlgvznxtyhpnpfkfkmlhcljqrqpr&ssn=1769093222234055930&ssn_dr=0&ssn_sr=0&fv_date=1769093222&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18867&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2024)%3A%20London%20Session%20Breakout%20System%20with%20Risk%20Management%20and%20Trailing%20Stops%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909322205116841&fz_uniq=5049376717048490597&sv=2552)

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