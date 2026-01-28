---
title: Manual Backtesting Made Easy: Building a Custom Toolkit for Strategy Tester in MQL5
url: https://www.mql5.com/en/articles/17751
categories: Trading Systems, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:34:39.004685
---

[![](https://www.mql5.com/ff/sh/a27a2kwmtszm2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Subscribe to traders' channels or create your own.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=vpcudokyepxfrcxrpjcktglhsjlemtza&s=f08ad2c1289e29bd5630f1ef977aef297d5cdbfcb686faed4a4b0f1e276d3c4a&uid=&ref=https://www.mql5.com/en/articles/17751&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049227140517439356)

MetaTrader 5 / Tester


### Introduction

Backtesting trading strategies is a cornerstone of successful trading, but automating every idea can feel restrictive, while manual testing often lacks structure and precision. What if you could combine the control of manual trading with the power of [MetaTrader 5’s](https://www.metatrader5.com/ "https://www.metatrader5.com/") Strategy Tester? In this article, we introduce a custom [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) Expert Advisor (EA) that transforms manual backtesting into an intuitive, efficient process—equipping you with a toolkit to test strategies on your terms. We’ll cover these steps in this order:

1. [The Plan: Designing a Manual Backtesting Toolkit](https://www.mql5.com/en/articles/17751#para1)
2. [Implementation in MQL5: Bringing the Toolkit to Life](https://www.mql5.com/en/articles/17751#para2)
3. [Backtesting in Action: Using the Toolkit](https://www.mql5.com/en/articles/17751#para3)
4. [Conclusion](https://www.mql5.com/en/articles/17751#para4)

By the end, you’ll have a practical solution to backtest and refine your trading ideas quickly and confidently in the Strategy Tester.

### The Plan: Designing a Manual Backtesting Toolkit

We aim to create a toolkit that merges manual control with the [Strategy Tester’s](https://www.metatrader5.com/en/automated-trading/strategy-tester "https://www.metatrader5.com/en/automated-trading/strategy-tester") fast backtesting speed in MetaTrader 5, sidestepping the slow real-time ticks of traditional manual testing. We will design the program with on-chart buttons to trigger Buy or Sell trades, adjust lot sizes, set Stop Loss (SL) and Take Profit (TP) levels, and close all positions via a Panic Button—fully integrable with any strategy, from indicators and Japanese candlestick patterns to price action, all working at the Tester’s accelerated pace. This flexible setup will let us test any trading approach interactively with speed and precision, streamlining strategy refinement in a simulated environment. In a nutshell, here is a visualization of what we are aiming for:

![IMAGE PLAN](https://c.mql5.com/2/132/Screenshot_2025-04-09_174847.png)

### Implementation in MQL5: Bringing the Toolkit to Life

To create the program in MQL5, we will need to define the program metadata, and then define some user input parameters, and lastly, we include some library files that will enable us to do the trading activity.

```
//+------------------------------------------------------------------+
//|                       Manual backtest toolkit in Strategy Tester |
//|                        Copyright 2025, Forex Algo-Trader, Allan. |
//|                                 "https://t.me/Forex_Algo_Trader" |
//+------------------------------------------------------------------+
#property copyright "Forex Algo-Trader, Allan"
#property link      "https://t.me/Forex_Algo_Trader"
#property version   "1.00"
#property description "This EA Enables manual backtest in the strategy tester"
#property strict //--- Enforce strict coding rules to catch errors early

#define BTN_BUY "BTN BUY" //--- Define the name for the Buy button
#define BTN_SELL "BTN SELL" //--- Define the name for the Sell button
#define BTN_P "BTN P" //--- Define the name for the button that increase lot size
#define BTN_M "BTN M" //--- Define the name for the button that decrease lot size
#define BTN_LOT "BTN LOT" //--- Define the name for the lot size display button
#define BTN_CLOSE "BTN CLOSE" //--- Define the name for the button that close all positions
#define BTN_SL "BTN SL" //--- Define the name for the Stop Loss display button
#define BTN_SL1M "BTN SL1M" //--- Define the name for the button that slightly lower Stop Loss
#define BTN_SL2M "BTN SL2M" //--- Define the name for the button that greatly lower Stop Loss
#define BTN_SL1P "BTN SL1P" //--- Define the name for the button that slightly raise Stop Loss
#define BTN_SL2P "BTN SL2P" //--- Define the name for the button that greatly raise Stop Loss
#define BTN_TP "BTN TP" //--- Define the name for the Take Profit display button
#define BTN_TP1M "BTN TP1M" //--- Define the name for the button that slightly lower Take Profit
#define BTN_TP2M "BTN TP2M" //--- Define the name for the button that greatly lower Take Profit
#define BTN_TP1P "BTN TP1P" //--- Define the name for the button that slightly raise Take Profit
#define BTN_TP2P "BTN TP2P" //--- Define the name for the button that greatly raise Take Profit
#define BTN_YES "BTN YES" //--- Define the name for the button that confirm a trade
#define BTN_NO "BTN NO" //--- Define the name for the button that cancel a trade
#define BTN_IDLE "BTN IDLE" //--- Define the name for the idle button between Yes and No
#define HL_SL "HL SL" //--- Define the name for the Stop Loss horizontal line
#define HL_TP "HL TP" //--- Define the name for the Take Profit horizontal line

#include <Trade/Trade.mqh> //--- Bring in the Trade library needed for trading functions
CTrade obj_Trade; //--- Create a trading object to handle trade operations

bool tradeInAction = false; //--- Track whether a trade setup is currently active
bool isHaveTradeLevels = false; //--- Track whether Stop Loss and Take Profit levels are shown

input double init_lot = 0.03;
input int slow_pts = 10;
input int fast_pts = 100;
```

Here, we start by defining a set of interactive buttons like "BTN\_BUY" and "BTN\_SELL" using [#define](https://www.mql5.com/en/docs/basis/preprosessor/constant) keyword to kick off trades whenever we want, giving us direct control over entry points, while "BTN\_P" and "BTN\_M" let us tweak the "init\_lot" size—set initially at 0.03—up or down to match our risk appetite. We also include "BTN\_CLOSE" as our emergency exit, a quick way to shut down all positions in a snap, and we rely on "tradeInAction" to keep tabs on whether we’re in the middle of setting up a trade and "isHaveTradeLevels" to signal when Stop Loss and Take Profit visuals are active.

We then tap into the "CTrade" class from "<Trade/Trade.mqh>" to create an "obj\_Trade" object to handle trade execution smoothly and efficiently. To give us even more flexibility, we add adjustable [inputs](https://www.mql5.com/en/docs/basis/variables/inputvariables) like "slow\_pts" at 10 and "fast\_pts" at 100, so we can fine-tune our Stop Loss and Take Profit levels on the fly, ensuring our toolkit adapts to whatever strategy we’re testing. Now, since we will need to create the panel buttons, let us create a function with all the possible inputs for reusability and customization.

```
//+------------------------------------------------------------------+
//| Create button function                                           |
//+------------------------------------------------------------------+
void CreateBtn(string objName,int xD,int yD,int xS,int yS,string txt,
               int fs=13,color clrTxt=clrWhite,color clrBg=clrBlack,
               color clrBd=clrBlack,string font="Calibri"){
   ObjectCreate(0,objName,OBJ_BUTTON,0,0,0); //--- Create a new button object on the chart
   ObjectSetInteger(0,objName,OBJPROP_XDISTANCE, xD); //--- Set the button's horizontal position
   ObjectSetInteger(0,objName,OBJPROP_YDISTANCE, yD); //--- Set the button's vertical position
   ObjectSetInteger(0,objName,OBJPROP_XSIZE, xS); //--- Set the button's width
   ObjectSetInteger(0,objName,OBJPROP_YSIZE, yS); //--- Set the button's height
   ObjectSetInteger(0,objName,OBJPROP_CORNER, CORNER_LEFT_UPPER); //--- Position the button from the top-left corner
   ObjectSetString(0,objName,OBJPROP_TEXT, txt); //--- Set the text displayed on the button
   ObjectSetInteger(0,objName,OBJPROP_FONTSIZE, fs); //--- Set the font size of the button text
   ObjectSetInteger(0,objName,OBJPROP_COLOR, clrTxt); //--- Set the color of the button text
   ObjectSetInteger(0,objName,OBJPROP_BGCOLOR, clrBg); //--- Set the background color of the button
   ObjectSetInteger(0,objName,OBJPROP_BORDER_COLOR,clrBd); //--- Set the border color of the button
   ObjectSetString(0,objName,OBJPROP_FONT,font); //--- Set the font style of the button text

   ChartRedraw(0); //--- Refresh the chart to show the new button
}
```

Here, we define the "CreateBtn" function to build every button—like "BTN\_BUY" or "BTN\_SELL"—on the chart, taking inputs such as "objName" for the button’s identity, "xD" and "yD" for its horizontal and vertical positions, "xS" and "yS" for its width and height, and "txt" for the label we want to display, like “BUY” or “SELL.” To make this happen, we use the [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) function to place a new [OBJ\_BUTTON](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) object on the chart, setting its base at coordinates (0,0,0) for simplicity. Then, we position it precisely with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) to adjust "OBJPROP\_XDISTANCE" to "xD" and "OBJPROP\_YDISTANCE" to "yD," ensuring it sits exactly where we need it, and we size it using "OBJPROP\_XSIZE" for "xS" and "OBJPROP\_YSIZE" for "yS" to fit our design.

We anchor it to the top-left corner with "OBJPROP\_CORNER" set to [CORNER\_LEFT\_UPPER](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer), making the layout consistent, and we use [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) to assign "OBJPROP\_TEXT" as "txt" so the button shows its purpose clearly. For style, we tweak "OBJPROP\_FONTSIZE" to "fs" (defaulting to 13), "OBJPROP\_COLOR" to "clrTxt" (defaulting to white) for text, [OBJPROP\_BGCOLOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to "clrBg" (defaulting to black) for the background, and "OBJPROP\_BORDER\_COLOR" to "clrBd" (defaulting to black) for the outline, while "OBJPROP\_FONT" gets "font" (defaulting to "Calibri") for a clean look. Finally, we use the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function to refresh the chart with "0" as the window ID, instantly displaying our new button so we can interact with it in the Strategy Tester. We can now call the function whenever we want to create a button, and we will start by calling it in the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){
   CreateBtn(BTN_P,150,45,40,25,CharToString(217),15,clrBlack,clrWhite,clrBlack,"Wingdings"); //--- Make the button to increase lot size with an up arrow
   CreateBtn(BTN_LOT,190,45,60,25,string(init_lot),12,clrWhite,clrGray,clrBlack); //--- Make the button showing the current lot size
   CreateBtn(BTN_M,250,45,40,25,CharToString(218),15,clrBlack,clrWhite,clrBlack,"Wingdings"); //--- Make the button to decrease lot size with a down arrow
   CreateBtn(BTN_BUY,110,70,110,30,"BUY",15,clrWhite,clrGreen,clrBlack); //--- Make the Buy button with a green background
   CreateBtn(BTN_SELL,220,70,110,30,"SELL",15,clrWhite,clrRed,clrBlack); //--- Make the Sell button with a red background
   CreateBtn(BTN_CLOSE,110,100,220,30,"PANIC BUTTON (X)",15,clrWhite,clrBlack,clrBlack); //--- Make the emergency button to close all trades

   return(INIT_SUCCEEDED); //--- Tell the system the EA start up successfully
}
```

Here, we kick off our manual backtesting toolkit with the OnInit event handler, setting up its interface in the Strategy Tester. We use the "CreateBtn" function to place "BTN\_P" at "xD" 150, "yD" 45 with an up arrow from [CharToString(217)](https://www.mql5.com/en/docs/convert/chartostring) in "Wingdings," "BTN\_LOT" at "xD" 190 showing "init\_lot," and "BTN\_M" at "xD" 250 with a down arrow from "CharToString(218)"—all styled for lot size control. Then, we add "BTN\_BUY" at "xD" 110, "yD" 70 with "BUY" on "clrGreen," "BTN\_SELL" at "xD" 220 with "SELL" on "clrRed," and "BTN\_CLOSE" at "xD" 110, "yD" 100 as "PANIC BUTTON (X)" on "clrBlack," before signaling success with "return" and [INIT\_SUCCEEDED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode). The Wingdings font we use for the icons from the MQL5 already defined table for characters, which is as below.

![WINGDINGS](https://c.mql5.com/2/132/Screenshot_2025-04-09_185338.png)

When we run the program, we get the following output.

![BUTTONS INTERFACE](https://c.mql5.com/2/132/Screenshot_2025-04-09_190537.png)

Since we have set the foundational background, we need to read the button states and the values so we can use them for trading purposes. Thus, we need some functions for that too.

```
int GetState(string Name){return (int)ObjectGetInteger(0,Name,OBJPROP_STATE);} //--- Get whether a button is pressed or not
string GetValue(string Name){return ObjectGetString(0,Name,OBJPROP_TEXT);} //--- Get the text shown on an object
double GetValueHL(string Name){return ObjectGetDouble(0,Name,OBJPROP_PRICE);} //--- Get the price level of a horizontal line
```

Here, we define the "GetState" function to check button clicks, where we use the [ObjectGetInteger](https://www.mql5.com/en/docs/objects/objectgetinteger) function with "OBJPROP\_STATE" to return if "Name" is pressed, "GetValue" to fetch text from "Name" using "ObjectGetString" with "OBJPROP\_TEXT," and "GetValueHL" to grab price levels of "Name" with [ObjectGetDouble](https://www.mql5.com/en/docs/objects/objectgetdouble) using [OBJPROP\_PRICE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_double) for precise trade control. We can now use the functions to get the button states in the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler since we can't directly use the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler in the Strategy Tester. Here is how we achieve that.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){
   double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits); //--- Get the current Ask price and adjust it to the right decimal places
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits); //--- Get the current Bid price and adjust it to the right decimal places

   if (GetState(BTN_BUY)==true || GetState(BTN_SELL)){ //--- Check if either the Buy or Sell button is clicked
      tradeInAction = true; //--- Set trade setup to active
   }
}
```

Here, we use the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler to drive our toolkit’s real-time actions in the Strategy Tester, where we use the [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble) function with [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) to set "Ask" to the current "SYMBOL\_ASK" price and "Bid" to the [SYMBOL\_BID](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double) price, both adjusted to [\_Digits](https://www.mql5.com/en/docs/predefined/_Digits) for accuracy and if "GetState" shows "BTN\_BUY" or "BTN\_SELL" as true, we set "tradeInAction" to true to start our trade setup. This is the point we need to extra trade levels to enable us to set the levels and adjust dynamically. Let us have a function for that.

```
//+------------------------------------------------------------------+
//| Create high low function                                         |
//+------------------------------------------------------------------+
void createHL(string objName,datetime time1,double price1,color clr){
   if (ObjectFind(0,objName) < 0){ //--- Check if the horizontal line doesn’t already exist
      ObjectCreate(0,objName,OBJ_HLINE,0,time1,price1); //--- Create a new horizontal line at the specified price
      ObjectSetInteger(0,objName,OBJPROP_TIME,time1); //--- Set the time property (though not critical for HLINE)
      ObjectSetDouble(0,objName,OBJPROP_PRICE,price1); //--- Set the price level of the horizontal line
      ObjectSetInteger(0,objName,OBJPROP_COLOR,clr); //--- Set the color of the line (red for SL, green for TP)
      ObjectSetInteger(0,objName,OBJPROP_STYLE,STYLE_DASHDOTDOT); //--- Set the line style to dash-dot-dot
      ChartRedraw(0); //--- Refresh the chart to display the new line
   }
}
```

First, we define the "createHL" function to draw horizontal lines for our toolkit in the Strategy Tester, where we use the [ObjectFind](https://www.mql5.com/en/docs/objects/objectfind) function to check if "objName" exists and, if it’s less than 0, we use the [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) function to make an "OBJ\_HLINE" at "time1" and "price1," we use the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) function to set "OBJPROP\_TIME" to "time1" and "OBJPROP\_COLOR" to "clr" and "OBJPROP\_STYLE" to "STYLE\_DASHDOTDOT," we use the [ObjectSetDouble](https://www.mql5.com/en/docs/objects/objectsetdouble) function to set "OBJPROP\_PRICE" to "price1," and we use the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function to refresh the chart with "0" to display it. Then, we integrate this function into another function for creating the trade levels seamlessly as below.

```
//+------------------------------------------------------------------+
//| Create trade levels function                                     |
//+------------------------------------------------------------------+
void CreateTradeLevels(){
   double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits); //--- Get unnoticed the current Ask price, adjusted for digits
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits); //--- Get the current Bid price, adjusted for digits

   string level_SL,level_TP; //--- Declare variables to hold SL and TP levels as strings
   if (GetState(BTN_BUY)==true){ //--- Check if the Buy button is active
      level_SL = string(Bid-100*_Point); //--- Set initial Stop Loss 100 points below Bid for Buy
      level_TP = string(Bid+100*_Point); //--- Set initial Take Profit 100 points above Bid for Buy
   }
   else if (GetState(BTN_SELL)==true){ //--- Check if the Sell button is active
      level_SL = string(Ask+100*_Point); //--- Set initial Stop Loss 100 points above Ask for Sell
      level_TP = string(Ask-100*_Point); //--- Set initial Take Profit 100 points below Ask for Sell
   }

   createHL(HL_SL,0,double(level_SL),clrRed); //--- Create a red Stop Loss line at the calculated level
   createHL(HL_TP,0,double(level_TP),clrGreen); //--- Create a green Take Profit line at the calculated level

   CreateBtn(BTN_SL,110,135,110,23,"SL: "+GetValue(HL_SL),13,clrRed,clrWhite,clrRed); //--- Make a button showing the Stop Loss level
   CreateBtn(BTN_TP,220,135,110,23,"TP: "+GetValue(HL_TP),13,clrGreen,clrWhite,clrGreen); //--- Make a button showing the Take Profit level

   CreateBtn(BTN_SL1M,110,158,27,20,"-",17,clrBlack,clrWhite,clrGray); //--- Make a button to slightly lower Stop Loss
   CreateBtn(BTN_SL2M,137,158,27,20,"--",17,clrBlack,clrWhite,clrGray); //--- Make a button to greatly lower Stop Loss
   CreateBtn(BTN_SL2P,164,158,27,20,"++",17,clrBlack,clrWhite,clrGray); //--- Make a button to greatly raise Stop Loss
   CreateBtn(BTN_SL1P,191,158,27,20,"+",17,clrBlack,clrWhite,clrGray); //--- Make a button to slightly raise Stop Loss

   CreateBtn(BTN_TP1P,222,158,27,20,"+",17,clrBlack,clrWhite,clrGray); //--- Make a button to slightly raise Take Profit
   CreateBtn(BTN_TP2P,249,158,27,20,"++",17,clrBlack,clrWhite,clrGray); //--- Make a button to greatly raise Take Profit
   CreateBtn(BTN_TP2M,276,158,27,20,"--",17,clrBlack,clrWhite,clrGray); //--- Make a button to greatly lower Take Profit
   CreateBtn(BTN_TP1M,303,158,27,20,"-",17,clrBlack,clrWhite,clrGray); //--- Make a button to slightly lower Take Profit

   CreateBtn(BTN_YES,110,178,70,30,CharToString(254),20,clrWhite,clrDarkGreen,clrWhite,"Wingdings"); //--- Make a green checkmark button to confirm the trade
   CreateBtn(BTN_NO,260,178,70,30,CharToString(253),20,clrWhite,clrDarkRed,clrWhite,"Wingdings"); //--- Make a red X button to cancel the trade
   CreateBtn(BTN_IDLE,180,183,80,25,CharToString(40),20,clrWhite,clrBlack,clrWhite,"Wingdings"); //--- Make a neutral button between Yes and No
}
```

Here, we define the "CreateTradeLevels" function to set up our trade levels, where we use the [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble) function with [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) to set "Ask" to "SYMBOL\_ASK" and "Bid" to "SYMBOL\_BID," adjusted by [\_Digits](https://www.mql5.com/en/docs/predefined/_Digits), and declare "level\_SL" and "level\_TP" as strings. If "GetState" shows "BTN\_BUY" as true, we set "level\_SL" to "Bid-100\_Point" and "level\_TP" to "Bid+100\_Point," but if "BTN\_SELL" is true, we set "level\_SL" to "Ask+100\_Point" and "level\_TP" to "Ask-100\_Point".

We use the "createHL" function to draw "HL\_SL" at "double(level\_SL)" in "clrRed" and "HL\_TP" at "double(level\_TP)" in "clrGreen," then use the "CreateBtn" function to make buttons like "BTN\_SL" with "GetValue(HL\_SL)" text, "BTN\_TP" with "GetValue(HL\_TP)," and adjustment buttons "BTN\_SL1M," "BTN\_SL2M," "BTN\_SL2P," "BTN\_SL1P," "BTN\_TP1P," "BTN\_TP2P," "BTN\_TP2M," and "BTN\_TP1M" with symbols like "-" and "+," plus "BTN\_YES," "BTN\_NO," and "BTN\_IDLE" using [CharToString](https://www.mql5.com/en/docs/convert/chartostring) for confirm, cancel, and neutral options in " [Wingdings](https://www.mql5.com/en/docs/constants/objectconstants/wingdings)". With the function, we can call it when the buy or sell buttons are clicked to initialize the trade level setup.

```
if (!isHaveTradeLevels){ //--- Check if trade levels aren't already on the chart
   CreateTradeLevels(); //--- Add Stop Loss and Take Profit levels and controls to the chart
   isHaveTradeLevels = true; //--- Mark that trade levels are now present
}
```

Here, we set up a check, where we test if "isHaveTradeLevels" is false with "!isHaveTradeLevels," and when it is, we use the "CreateTradeLevels" function to place Stop Loss and Take Profit controls on the chart, then update "isHaveTradeLevels" to true to show they’re active. Upon compilation, we have the following outcome.

![TRADE LEVELS](https://c.mql5.com/2/132/Screenshot_2025-04-09_193759.png)

Next, we need to bring the buttons for the trade levels to life by making them responsive and doing what they need to do. Here is how we achieve that.

```
if (tradeInAction){ //--- Continue if a trade setup is active

   // SL SLOW/FAST BUTTONS
   if (GetState(BTN_SL1M)){ //--- Check if the small Stop Loss decrease button is clicked
      ObjectSetDouble(0,HL_SL,OBJPROP_PRICE,GetValueHL(HL_SL)-slow_pts*_Point); //--- Move the Stop Loss down by a small amount
      ObjectSetInteger(0,BTN_SL1M,OBJPROP_STATE,false); //--- Turn off the button press state
      ChartRedraw(0); //--- Refresh the chart to show the change
   }
   if (GetState(BTN_SL2M)){ //--- Check if the large Stop Loss decrease button is clicked
      ObjectSetDouble(0,HL_SL,OBJPROP_PRICE,GetValueHL(HL_SL)-fast_pts*_Point); //--- Move the Stop Loss down by a large amount
      ObjectSetInteger(0,BTN_SL2M,OBJPROP_STATE,false); //--- Turn off the button press state
      ChartRedraw(0); //--- Refresh the chart to show the change
   }
   if (GetState(BTN_SL1P)){ //--- Check if the small Stop Loss increase button is clicked
      ObjectSetDouble(0,HL_SL,OBJPROP_PRICE,GetValueHL(HL_SL)+slow_pts*_Point); //--- Move the Stop Loss up by a small amount
      ObjectSetInteger(0,BTN_SL1P,OBJPROP_STATE,false); //--- Turn off the button press state
      ChartRedraw(0); //--- Refresh the chart to show the change
   }
   if (GetState(BTN_SL2P)){ //--- Check if the large Stop Loss increase button is clicked
      ObjectSetDouble(0,HL_SL,OBJPROP_PRICE,GetValueHL(HL_SL)+fast_pts*_Point); //--- Move the Stop Loss up by a large amount
      ObjectSetInteger(0,BTN_SL2P,OBJPROP_STATE,false); //--- Turn off the button press state
      ChartRedraw(0); //--- Refresh the chart to show the change
   }

   // TP SLOW/FAST BUTTONS
   if (GetState(BTN_TP1M)){ //--- Check if the small Take Profit decrease button is clicked
      ObjectSetDouble(0,HL_TP,OBJPROP_PRICE,GetValueHL(HL_TP)-slow_pts*_Point); //--- Move the Take Profit down by a small amount
      ObjectSetInteger(0,BTN_TP1M,OBJPROP_STATE,false); //--- Turn off the button press state
      ChartRedraw(0); //--- Refresh the chart to show the change
   }
   if (GetState(BTN_TP2M)){ //--- Check if the large Take Profit decrease button is clicked
      ObjectSetDouble(0,HL_TP,OBJPROP_PRICE,GetValueHL(HL_TP)-fast_pts*_Point); //--- Move the Take Profit down by a large amount
      ObjectSetInteger(0,BTN_TP2M,OBJPROP_STATE,false); //--- Turn off the button press state
      ChartRedraw(0); //--- Refresh the chart to show the change
   }
   if (GetState(BTN_TP1P)){ //--- Check if the small Take Profit increase button is clicked
      ObjectSetDouble(0,HL_TP,OBJPROP_PRICE,GetValueHL(HL_TP)+slow_pts*_Point); //--- Move the Take Profit up by a small amount
      ObjectSetInteger(0,BTN_TP1P,OBJPROP_STATE,false); //--- Turn off the button press state
      ChartRedraw(0); //--- Refresh the chart to show the change
   }
   if (GetState(BTN_TP2P)){ //--- Check if the large Take Profit increase button is clicked
      ObjectSetDouble(0,HL_TP,OBJPROP_PRICE,GetValueHL(HL_TP)+fast_pts*_Point); //--- Move the Take Profit up by a large amount
      ObjectSetInteger(0,BTN_TP2P,OBJPROP_STATE,false); //--- Turn off the button press state
      ChartRedraw(0); //--- Refresh the chart to show the change
   }

}
```

Here, we manage Stop Loss and Take Profit adjustments in our toolkit when "tradeInAction" is true, where we use the "GetState" function to check if buttons like "BTN\_SL1M," "BTN\_SL2M," "BTN\_SL1P," or "BTN\_SL2P" are clicked and adjust "HL\_SL" by "slow\_pts\_Point" or "fast\_pts\_Point" using the [ObjectSetDouble](https://www.mql5.com/en/docs/objects/objectsetdouble) function with "OBJPROP\_PRICE" and "GetValueHL," then use the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) function to reset [OBJPROP\_STATE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to false and the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function to update the chart, and similarly handle "BTN\_TP1M," "BTN\_TP2M," "BTN\_TP1P," or "BTN\_TP2P" for "HL\_TP" adjustments. Finally, once the levels are set, we can confirm the placement and open respective positions, and then clean off the trade level setup, but first, we will need to get a function to delete the levels setup panel.

```
//+------------------------------------------------------------------+
//| Delete objects function                                          |
//+------------------------------------------------------------------+
void DeleteObjects_SLTP(){
   ObjectDelete(0,HL_SL); //--- Remove the Stop Loss line from the chart
   ObjectDelete(0,HL_TP); //--- Remove the Take Profit line from the chart
   ObjectDelete(0,BTN_SL); //--- Remove the Stop Loss display button
   ObjectDelete(0,BTN_TP); //--- Remove the Take Profit display button
   ObjectDelete(0,BTN_SL1M); //--- Remove the small Stop Loss decrease button
   ObjectDelete(0,BTN_SL2M); //--- Remove the large Stop Loss decrease button
   ObjectDelete(0,BTN_SL1P); //--- Remove the small Stop Loss increase button
   ObjectDelete(0,BTN_SL2P); //--- Remove the large Stop Loss increase button
   ObjectDelete(0,BTN_TP1P); //--- Remove the small Take Profit increase button
   ObjectDelete(0,BTN_TP2P); //--- Remove the large Take Profit increase button
   ObjectDelete(0,BTN_TP2M); //--- Remove the large Take Profit decrease button
   ObjectDelete(0,BTN_TP1M); //--- Remove the small Take Profit decrease button
   ObjectDelete(0,BTN_YES); //--- Remove the confirm trade button
   ObjectDelete(0,BTN_NO); //--- Remove the cancel trade button
   ObjectDelete(0,BTN_IDLE); //--- Remove the idle button

   ChartRedraw(0); //--- Refresh the chart to show all objects removed
}
```

Here, we handle cleanup in our toolkit with the "DeleteObjects\_SLTP" function, where we use the [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete) function to remove "HL\_SL," "HL\_TP," "BTN\_SL," "BTN\_TP," "BTN\_SL1M," "BTN\_SL2M," "BTN\_SL1P," "BTN\_SL2P," "BTN\_TP1P," "BTN\_TP2P," "BTN\_TP2M," "BTN\_TP1M," "BTN\_YES," "BTN\_NO," and "BTN\_IDLE" from the chart, then use the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function with "0" to refresh and show everything cleared. We can now use this function in our order placement logic.

```
// BUY ORDER PLACEMENT
if (GetState(BTN_BUY) && GetState(BTN_YES)){ //--- Check if both Buy and Yes buttons are clicked
   obj_Trade.Buy(double(GetValue(BTN_LOT)),_Symbol,Ask,GetValueHL(HL_SL),GetValueHL(HL_TP)); //--- Place a Buy order with set lot size, SL, and TP
   DeleteObjects_SLTP(); //--- Remove all trade level objects from the chart
   isHaveTradeLevels = false; //--- Mark that trade levels are no longer present
   ObjectSetInteger(0,BTN_YES,OBJPROP_STATE,false); //--- Turn off the Yes button press state
   ObjectSetInteger(0,BTN_BUY,OBJPROP_STATE,false); //--- Turn off the Buy button press state
   tradeInAction = false; //--- Mark the trade setup as complete
   ChartRedraw(0); //--- Refresh the chart to reflect changes
}
// SELL ORDER PLACEMENT
else if (GetState(BTN_SELL) && GetState(BTN_YES)){ //--- Check if both Sell and Yes buttons are clicked
   obj_Trade.Sell(double(GetValue(BTN_LOT)),_Symbol,Bid,GetValueHL(HL_SL),GetValueHL(HL_TP)); //--- Place a Sell order with set lot size, SL, and TP
   DeleteObjects_SLTP(); //--- Remove all trade level objects from the chart
   isHaveTradeLevels = false; //--- Mark that trade levels are no longer present
   ObjectSetInteger(0,BTN_YES,OBJPROP_STATE,false); //--- Turn off the Yes button press state
   ObjectSetInteger(0,BTN_SELL,OBJPROP_STATE,false); //--- Turn off the Sell button press state
   tradeInAction = false; //--- Mark the trade setup as complete
   ChartRedraw(0); //--- Refresh the chart to reflect changes
}
else if (GetState(BTN_NO)){ //--- Check if the No button is clicked to cancel
   DeleteObjects_SLTP(); //--- Remove all trade level objects from the chart
   isHaveTradeLevels = false; //--- Mark that trade levels are no longer present
   ObjectSetInteger(0,BTN_NO,OBJPROP_STATE,false); //--- Turn off the No button press state
   ObjectSetInteger(0,BTN_BUY,OBJPROP_STATE,false); //--- Turn off the Buy button press state
   ObjectSetInteger(0,BTN_SELL,OBJPROP_STATE,false); //--- Turn off the Sell button press state
   tradeInAction = false; //--- Mark the trade setup as canceled
   ChartRedraw(0); //--- Refresh the chart to reflect changes
}
```

We execute trades in our toolkit within the Strategy Tester, where we use the "GetState" function to check if "BTN\_BUY" and "BTN\_YES" are true and then use the "obj\_Trade.Buy" method with "double(GetValue(BTN\_LOT))", [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol), "Ask", "GetValueHL(HL\_SL)", and "GetValueHL(HL\_TP)" to place a Buy order, or if "BTN\_SELL" and "BTN\_YES" are true, we use "obj\_Trade.Sell" with "Bid" instead, and in either case, we use the "DeleteObjects\_SLTP" function to clear objects, set "isHaveTradeLevels" and "tradeInAction" to false, use the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) function to reset [OBJPROP\_STATE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) on "BTN\_YES", "BTN\_BUY", or "BTN\_SELL" to false, and use the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function to update the chart, but if "BTN\_NO" is true, we cancel by clearing objects and resetting states similarly. Similarly, we handle the increase or decrease of the trading volume buttons as follows.

```
if (GetState(BTN_P)==true){ //--- Check if the lot size increase button is clicked
   double newLot = (double)GetValue(BTN_LOT); //--- Get the current lot size as a number
   double lotStep = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP); //--- Get the minimum lot size change allowed
   newLot += lotStep; //--- Increase the lot size by one step
   newLot = NormalizeDouble(newLot,2); //--- Round the new lot size to 2 decimal places
   newLot = newLot > 0.1 ? lotStep : newLot; //--- Ensure lot size doesn't exceed 0.1, otherwise reset to step
   ObjectSetString(0,BTN_LOT,OBJPROP_TEXT,string(newLot)); //--- Update the lot size display with the new value
   ObjectSetInteger(0,BTN_P,OBJPROP_STATE,false); //--- Turn off the increase button press state
   ChartRedraw(0); //--- Refresh the chart to show the new lot size
}
if (GetState(BTN_M)==true){ //--- Check if the lot size decrease button is clicked
   double newLot = (double)GetValue(BTN_LOT); //--- Get the current lot size as a number
   double lotStep = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP); //--- Get the minimum lot size change allowed
   newLot -= lotStep; //--- Decrease the lot size by one step
   newLot = NormalizeDouble(newLot,2); //--- Round the new lot size to 2 decimal places
   newLot = newLot < lotStep ? lotStep : newLot; //--- Ensure lot size doesn't go below minimum, otherwise set to step
   ObjectSetString(0,BTN_LOT,OBJPROP_TEXT,string(newLot)); //--- Update the lot size display with the new value
   ObjectSetInteger(0,BTN_M,OBJPROP_STATE,false); //--- Turn off the decrease button press state
   ChartRedraw(0); //--- Refresh the chart to show the new lot size
}
```

Here, we adjust lot sizes starting with the increase where we use the "GetState" function to check if "BTN\_P" is true, then use "GetValue" to set "newLot" from "BTN\_LOT," use [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) to get "lotStep" from [SYMBOL\_VOLUME\_STEP](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double), add "lotStep" to "newLot," and use [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble) to round it to 2 decimals, capping it at "lotStep" if over 0.1 before using [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) to update "BTN\_LOT"’s "OBJPROP\_TEXT" and "ObjectSetInteger" to reset "BTN\_P"’s "OBJPROP\_STATE" to false, followed by [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to refresh.

For the decrease, we use "GetState" to check "BTN\_M," subtract "lotStep" from "newLot" after fetching it the same way, keep it at least "lotStep," and apply the same [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring), "ObjectSetInteger," and "ChartRedraw" function steps to update "BTN\_LOT" and reset "BTN\_M". As for the panic button, we will need to define a function to close all the open positions when it is clicked.

```
//+------------------------------------------------------------------+
//| Close all positions function                                     |
//+------------------------------------------------------------------+
void closeAllPositions(){
   for (int i=PositionsTotal()-1; i>=0; i--){ //--- Loop through all open positions, starting from the last one
      ulong ticket = PositionGetTicket(i); //--- Get the ticket number of the current position
      if (ticket > 0){ //--- Check if the ticket is valid
         if (PositionSelectByTicket(ticket)){ //--- Select the position by its ticket number
            if (PositionGetString(POSITION_SYMBOL)==_Symbol){ //--- Check if the position is for the current chart symbol
               obj_Trade.PositionClose(ticket); //--- Close the selected position
            }
         }
      }
   }
}
```

We handle closing all positions with the "closeAllPositions" function, where we use the [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) function to loop from "i" as the last position minus 1 down to 0, use the [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket) function to get "ticket" for each index "i," and if "ticket" is valid, we use [PositionSelectByTicket](https://www.mql5.com/en/docs/trading/positionselectbyticket) to select it, then use [PositionGetString](https://www.mql5.com/en/docs/objects/objectgetstring) to check if [POSITION\_SYMBOL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_string) matches [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol) before using "obj\_Trade.PositionClose" method to close the position with "ticket". Then we can call this function when the panic button is clicked to close all the positions.

```
if (GetState(BTN_CLOSE)==true){ //--- Check if the close all positions button is clicked
   closeAllPositions(); //--- Close all open trades
   ObjectSetInteger(0,BTN_CLOSE,OBJPROP_STATE,false); //--- Turn off the close button press state
   ChartRedraw(0); //--- Refresh the chart to reflect closed positions
}
```

To manage the closing all trades, we use the "GetState" function to check if "BTN\_CLOSE" is true, and if so, we use the "closeAllPositions" function to shut down all open positions, then use the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) function to set "BTN\_CLOSE"’s [OBJPROP\_STATE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to false and the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function with "0" to update the chart. Upon compilation and running the program, we have the following outcome.

![FINAL OUTCOME](https://c.mql5.com/2/132/Screenshot_2025-04-09_201149.png)

From the image, we can see that we set the trade levels and we can open positions dynamically, achieving our objective. What now remains is testing the program thoroughly, and that is handled in the next topic below.

### Backtesting in Action: Using the Toolkit

We test our toolkit in [MetaTrader 5’s Strategy Tester](https://www.metatrader5.com/en/automated-trading/strategy-tester "https://www.metatrader5.com/en/automated-trading/strategy-tester") by loading the program, selecting our settings, and starting it—watch the [Graphics Interchange Format](https://en.wikipedia.org/wiki/GIF "https://en.wikipedia.org/wiki/GIF") (GIF) below to see Buy, Sell, and Adjustment buttons in action at lightning speed. Click Buy or Sell, tweak Stop Loss, Take Profit, and lot size, then confirm with Yes or cancel with No, with the Panic Button ready to close all trades fast. Here it is.

![TESTER BACKTEST GIF](https://c.mql5.com/2/132/TESTER_GIF.gif)

### Conclusion

In conclusion, we’ve crafted a manual backtesting toolkit that merges hands-on control with the [Strategy Tester’s](https://www.metatrader5.com/en/automated-trading/strategy-tester "https://www.metatrader5.com/en/automated-trading/strategy-tester") speed in MQL5, simplifying how we test trading ideas. We’ve shown how to design it, code it, and use it to adjust trades with buttons—all tailored for quick, precise simulations. You can adapt this toolkit to your needs and enhance your backtesting experience with it.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17751.zip "Download all attachments in the single ZIP archive")

[Manual\_Backtest\_in\_Strategy\_Tester.mq5](https://www.mql5.com/en/articles/download/17751/manual_backtest_in_strategy_tester.mq5 "Download Manual_Backtest_in_Strategy_Tester.mq5")(42.72 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/484838)**
(6)


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
16 Apr 2025 at 14:31

**Mogulh Chilyalya Kiti [#](https://www.mql5.com/en/forum/484838#comment_56469733):**

What about different timeframes

Hello. Currently just one timeframe. Maybe try that in the near future.

![Blessing Dumbura](https://c.mql5.com/avatar/2021/1/5FFC20BB-41D2.png)

**[Blessing Dumbura](https://www.mql5.com/en/users/beetrader)**
\|
16 Apr 2025 at 15:47

Thank you its a useful tool


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
16 Apr 2025 at 18:26

**Blessing Dumbura [#](https://www.mql5.com/en/forum/484838#comment_56473744):**

Thank you its a useful tool

Sure. Welcome and thanks too for the feedback.

![Dontrace](https://c.mql5.com/avatar/avatar_na2.png)

**[Dontrace](https://www.mql5.com/en/users/dontrace)**
\|
7 May 2025 at 11:08

was wondering how to convert mq4 indicator to mq5

![zigooo](https://c.mql5.com/avatar/2023/10/651b8490-6cfa.jpg)

**[zigooo](https://www.mql5.com/en/users/fauzidox)**
\|
17 Jun 2025 at 17:16

**MetaQuotes:**

Check out the new article: [Manual Backtesting Made Easy: Building a Custom Toolkit for Strategy Tester in MQL5](https://www.mql5.com/en/articles/17751).

Author: [Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372 "29210372")

I tried to modify it so that it can open orders quickly with lotsize and tp/sl, according to the EA input parameters. It works in the real market, but it doesn't work in backtest mode. What is the solution?


![Feature Engineering With Python And MQL5 (Part IV): Candlestick Pattern Recognition With UMAP Regression](https://c.mql5.com/2/134/Feature_Engineering_With_Python_And_MQL5_Part_IV___LOGO__2.png)[Feature Engineering With Python And MQL5 (Part IV): Candlestick Pattern Recognition With UMAP Regression](https://www.mql5.com/en/articles/17631)

Dimension reduction techniques are widely used to improve the performance of machine learning models. Let us discuss a relatively new technique known as Uniform Manifold Approximation and Projection (UMAP). This new technique has been developed to explicitly overcome the limitations of legacy methods that create artifacts and distortions in the data. UMAP is a powerful dimension reduction technique, and it helps us group similar candle sticks in a novel and effective way that reduces our error rates on out of sample data and improves our trading performance.

![Mastering Log Records (Part 6): Saving logs to database](https://c.mql5.com/2/131/Mastering_Log_Records_Part_6__LOGO.png)[Mastering Log Records (Part 6): Saving logs to database](https://www.mql5.com/en/articles/17709)

This article explores the use of databases to store logs in a structured and scalable way. It covers fundamental concepts, essential operations, configuration and implementation of a database handler in MQL5. Finally, it validates the results and highlights the benefits of this approach for optimization and efficient monitoring.

![Formulating Dynamic Multi-Pair EA (Part 2): Portfolio Diversification and Optimization](https://c.mql5.com/2/134/Formulating_Dynamic_Multi-Pair_EA_Part_2___LOGO.png)[Formulating Dynamic Multi-Pair EA (Part 2): Portfolio Diversification and Optimization](https://www.mql5.com/en/articles/16089)

Portfolio Diversification and Optimization strategically spreads investments across multiple assets to minimize risk while selecting the ideal asset mix to maximize returns based on risk-adjusted performance metrics.

![Developing a Replay System (Part 64): Playing the service (V)](https://c.mql5.com/2/93/Desenvolvendo_um_sistema_de_Replay_Parte_64____LOGO.png)[Developing a Replay System (Part 64): Playing the service (V)](https://www.mql5.com/en/articles/12250)

In this article, we will look at how to fix two errors in the code. However, I will try to explain them in a way that will help you, beginner programmers, understand that things don't always go as you expect. Anyway, this is an opportunity to learn. The content presented here is intended solely for educational purposes. In no way should this application be considered as a final document with any purpose other than to explore the concepts presented.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/17751&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049227140517439356)

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