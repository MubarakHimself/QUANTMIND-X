---
title: MQL5 Trading Tools (Part 1): Building an Interactive Visual Pending Orders Trade Assistant Tool
url: https://www.mql5.com/en/articles/17931
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T17:56:47.712569
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/17931&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049503813720714412)

MetaTrader 5 / Trading


### Introduction

Developing effective trading tools is essential for simplifying complex Forex trading tasks, yet creating intuitive interfaces that enhance decision-making remains a challenge. What if you could design a visual, interactive tool streamlining pending order placement within [MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/")? In this article, we introduce a custom [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) Expert Advisor (EA) that empowers traders with a Trade Assistant Tool, combining graphical precision with user-friendly controls to place Buy/Sell Stop and Limit orders efficiently. We’ll cover these steps in this order:

1. [Conceptual Design and Objectives of the Trade Assistant Tool](https://www.mql5.com/en/articles/17931#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/17931#para2)
3. [Backtesting](https://www.mql5.com/en/articles/17931#para3)
4. [Conclusion](https://www.mql5.com/en/articles/17931#para4)

By the end, you’ll have a clear understanding of how to build and test this tool, paving the way for advanced enhancements in the preceding parts.

### Conceptual Design and Objectives of the Trade Assistant Tool

We aim to develop a Trade Assistant Tool that delivers a seamless and efficient experience for us by simplifying the process of placing pending orders in Forex trading. We envision the tool as a [graphical user interface](https://en.wikipedia.org/wiki/Graphical_user_interface "https://en.wikipedia.org/wiki/Graphical_user_interface") (GUI) that integrates directly with MetaTrader 5, enabling us to set up Buy Stop, Sell Stop, Buy Limit, and Sell Limit orders through an intuitive control panel. Our design is to include buttons for selecting the desired order type and an input field for specifying the lot size. We prioritize visual interaction, allowing us to define entry price, stop-loss (SL), and take-profit (TP) levels by dragging interactive elements on the chart, which provides immediate feedback on price levels and the point differences between them.

Our focus is to ensure the tool is accessible and responsive. We will design the interface to be responsive, enabling us to adjust price levels with accuracy and confirm orders with a single click, minimizing the time spent on setup. Additionally, we will incorporate options to cancel or close the interface, offering us flexibility to adapt quickly to changing market conditions. By creating a visually appealing and responsive tool, we seek to enhance our decision-making, reduce errors in order placement, and provide a foundation for future enhancements, such as advanced risk management features, that we will explore in subsequent iterations. In a nutshell, here is a visualization of what we envision to create.

![GUI PLAN](https://c.mql5.com/2/137/Screenshot_2025-04-27_004219.png)

### Implementation in MQL5

To create the program in [MQL5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5"), we will need to define the program metadata, and then define some object name constants, and lastly, we include some library files that will enable us to do the trading activity.

```
//+------------------------------------------------------------------+
//|                                         TRADE ASSISTANT GUI TOOL |
//|      Copyright 2025, ALLAN MUNENE MUTIIRIA. #@Forex Algo-Trader. |
//|                           https://youtube.com/@ForexAlgo-Trader? |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, ALLAN MUNENE MUTIIRIA. #@Forex Algo-Trader"
#property link      "https://youtube.com/@ForexAlgo-Trader?"
#property version   "1.00"

#include <Trade/Trade.mqh> //--- Include the Trade library for trading operations

// Control panel object names
#define PANEL_BG       "PANEL_BG"         //--- Define constant for panel background object name
#define LOT_EDIT       "LOT_EDIT"         //--- Define constant for lot size edit field object name
#define PRICE_LABEL    "PRICE_LABEL"      //--- Define constant for price label object name
#define SL_LABEL       "SL_LABEL"         //--- Define constant for stop-loss label object name
#define TP_LABEL       "TP_LABEL"         //--- Define constant for take-profit label object name
#define BUY_STOP_BTN   "BUY_STOP_BTN"     //--- Define constant for buy stop button object name
#define SELL_STOP_BTN  "SELL_STOP_BTN"    //--- Define constant for sell stop button object name
#define BUY_LIMIT_BTN  "BUY_LIMIT_BTN"    //--- Define constant for buy limit button object name
#define SELL_LIMIT_BTN "SELL_LIMIT_BTN"   //--- Define constant for sell limit button object name
#define PLACE_ORDER_BTN "PLACE_ORDER_BTN" //--- Define constant for place order button object name
#define CANCEL_BTN     "CANCEL_BTN"       //--- Define constant for cancel button object name
#define CLOSE_BTN      "CLOSE_BTN"        //--- Define constant for close button object name

#define REC1 "REC1" //--- Define constant for rectangle 1 (TP) object name
#define REC2 "REC2" //--- Define constant for rectangle 2 object name
#define REC3 "REC3" //--- Define constant for rectangle 3 (Entry) object name
#define REC4 "REC4" //--- Define constant for rectangle 4 object name
#define REC5 "REC5" //--- Define constant for rectangle 5 (SL) object name

#define TP_HL "TP_HL" //--- Define constant for take-profit horizontal line object name
#define SL_HL "SL_HL" //--- Define constant for stop-loss horizontal line object name
#define PR_HL "PR_HL" //--- Define constant for price (entry) horizontal line object name

double Get_Price_d(string name) { return ObjectGetDouble(0, name, OBJPROP_PRICE); }                          //--- Function to get price as double for an object
string Get_Price_s(string name) { return DoubleToString(ObjectGetDouble(0, name, OBJPROP_PRICE), _Digits); } //--- Function to get price as string with proper digits
string update_Text(string name, string val) { return (string)ObjectSetString(0, name, OBJPROP_TEXT, val); }  //--- Function to update text of an object

int
   xd1, yd1, xs1, ys1, //--- Variables for rectangle 1 position and size
   xd2, yd2, xs2, ys2, //--- Variables for rectangle 2 position and size
   xd3, yd3, xs3, ys3, //--- Variables for rectangle 3 position and size
   xd4, yd4, xs4, ys4, //--- Variables for rectangle 4 position and size
   xd5, yd5, xs5, ys5; //--- Variables for rectangle 5 position and size

// Control panel variables
bool tool_visible = false;       //--- Flag to track if trading tool is visible
string selected_order_type = ""; //--- Variable to store selected order type
double lot_size = 0.01;          //--- Default lot size for trades
CTrade obj_Trade;                //--- Trade object for executing trading operations
int panel_x = 10, panel_y = 30;  //--- Panel position coordinates
```

Here, we lay the foundation for our Trade Assistant Tool by defining essential components, variables, and functions that enable the tool’s graphical and trading functionalities. We begin by including the "Trade.mqh" library, which provides the "CTrade" class for executing trading operations, such as placing pending orders. We then define a series of constants using [#define](https://www.mql5.com/en/docs/basis/preprosessor/constant) to assign unique names for GUI elements, such as "PANEL\_BG" for the control panel background, "LOT\_EDIT" for the lot size input field, and buttons like "BUY\_STOP\_BTN" and "SELL\_STOP\_BTN" for order type selection, and so much more.

We implement three utility functions to manage chart object properties: the "Get\_Price\_d" function retrieves the price of an object as a double, the "Get\_Price\_s" function converts this price to a string formatted with the appropriate number of decimal places using the [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring) function, and the "update\_Text" function updates the text of an object using the [ObjectSetString](https://www.mql5.com/en/docs/objects/ObjectSetString) function to display real-time price information.

To handle the positioning and sizing of the interactive rectangles, we declare sets of integer variables like "xd1", "yd1", "xs1", and "ys1" for each rectangle ("REC1" through "REC5"), representing their x-distance, y-distance, x-size, and y-size on the chart.

Finally, we define key control panel variables: "tool\_visible" as a boolean to track the tool’s visibility, "selected\_order\_type" as a string to store the chosen order type, "lot\_size" as a double initialized to 0.01 for trade volume, "obj\_Trade" as a "CTrade" object for executing trades, and "panel\_x" and "panel\_y" as integers to set the control panel’s position at coordinates (10, 30).

These elements collectively establish the structural backbone for our tool’s interactive interface and trading capabilities. So we can now move on to creating the control panel, but first, we will need a function to create a custom button.

```
//+------------------------------------------------------------------+
//| Create button                                                    |
//+------------------------------------------------------------------+
bool createButton(string objName, string text, int xD, int yD, int xS, int yS,
                  color clrTxt, color clrBG, int fontsize = 12,
                  color clrBorder = clrNONE, bool isBack = false, string font = "Calibri") {
   ResetLastError();                                                               //--- Reset last error code
   if(!ObjectCreate(0, objName, OBJ_BUTTON, 0, 0, 0)) {                            //--- Create button object
      Print(__FUNCTION__, ": Failed to create Btn: Error Code: ", GetLastError()); //--- Print error message
      return false;                                                                //--- Return failure
   }
   ObjectSetInteger(0, objName, OBJPROP_XDISTANCE, xD);             //--- Set button x-position
   ObjectSetInteger(0, objName, OBJPROP_YDISTANCE, yD);             //--- Set button y-position
   ObjectSetInteger(0, objName, OBJPROP_XSIZE, xS);                 //--- Set button width
   ObjectSetInteger(0, objName, OBJPROP_YSIZE, yS);                 //--- Set button height
   ObjectSetInteger(0, objName, OBJPROP_CORNER, CORNER_LEFT_UPPER); //--- Set button corner
   ObjectSetString(0, objName, OBJPROP_TEXT, text);                 //--- Set button text
   ObjectSetInteger(0, objName, OBJPROP_FONTSIZE, fontsize);        //--- Set font size
   ObjectSetString(0, objName, OBJPROP_FONT, font);                 //--- Set font
   ObjectSetInteger(0, objName, OBJPROP_COLOR, clrTxt);             //--- Set text color
   ObjectSetInteger(0, objName, OBJPROP_BGCOLOR, clrBG);            //--- Set background color
   ObjectSetInteger(0, objName, OBJPROP_BORDER_COLOR, clrBorder);   //--- Set border color
   ObjectSetInteger(0, objName, OBJPROP_BACK, isBack);              //--- Set background/foreground
   ObjectSetInteger(0, objName, OBJPROP_STATE, false);              //--- Reset button state
   ObjectSetInteger(0, objName, OBJPROP_SELECTABLE, false);         //--- Disable selection
   ObjectSetInteger(0, objName, OBJPROP_SELECTED, false);           //--- Disable selected state

   ChartRedraw(0); //--- Redraw chart
   return true;    //--- Return success
}
```

We define the "createButton" function to build customizable buttons for our tool. It accepts parameters like "objName" for the button’s name, "text" for its label, "xD" and "yD" for position, "xS" and "yS" for size, "clrTxt" and "clrBG" for text and background colors, "fontsize" (default 12), "clrBorder" (default "clrNONE"), "isBack" (default false), and "font" (default "Calibri").

We use the [ResetLastError](https://www.mql5.com/en/docs/common/ResetLastError) function to clear error codes, and then the [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) function to create an [OBJ\_BUTTON](https://www.mql5.com/en/docs/constants/objectconstants/enum_object). If it fails, we call the [Print](https://www.mql5.com/en/docs/common/print) function with [\_\_FUNCTION\_\_](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros) and [GetLastError](https://www.mql5.com/en/docs/check/getlasterror) to log the error and return false.

On success, we set properties like position, size, and colors using the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/ObjectSetInteger) and [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) functions, disable state and selection, and call the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function to update the chart, returning true. This will enable us to create interactive buttons. So with this function, we can create the control panel function.

```
//+------------------------------------------------------------------+
//| Create control panel                                             |
//+------------------------------------------------------------------+
void createControlPanel() {
   // Background rectangle
   ObjectCreate(0, PANEL_BG, OBJ_RECTANGLE_LABEL, 0, 0, 0);        //--- Create panel background rectangle
   ObjectSetInteger(0, PANEL_BG, OBJPROP_XDISTANCE, panel_x);      //--- Set background x-position
   ObjectSetInteger(0, PANEL_BG, OBJPROP_YDISTANCE, panel_y);      //--- Set background y-position
   ObjectSetInteger(0, PANEL_BG, OBJPROP_XSIZE, 250);              //--- Set background width
   ObjectSetInteger(0, PANEL_BG, OBJPROP_YSIZE, 280);              //--- Set background height
   ObjectSetInteger(0, PANEL_BG, OBJPROP_BGCOLOR, C'070,070,070'); //--- Set background color
   ObjectSetInteger(0, PANEL_BG, OBJPROP_BORDER_COLOR, clrWhite);  //--- Set border color
   ObjectSetInteger(0, PANEL_BG, OBJPROP_BACK, false);             //--- Set background to foreground

   createButton(CLOSE_BTN, CharToString(203), panel_x + 209, panel_y + 1, 40, 25, clrWhite, clrCrimson, 12, C'070,070,070', false, "Wingdings"); //--- Create close button

   // Lot size input
   ObjectCreate(0, LOT_EDIT, OBJ_EDIT, 0, 0, 0); //--- Create lot size edit field
   ObjectSetInteger(0, LOT_EDIT, OBJPROP_XDISTANCE, panel_x + 70); //--- Set edit field x-position
   ObjectSetInteger(0, LOT_EDIT, OBJPROP_YDISTANCE, panel_y + 40); //--- Set edit field y-position
   ObjectSetInteger(0, LOT_EDIT, OBJPROP_XSIZE, 110);              //--- Set edit field width
   ObjectSetInteger(0, LOT_EDIT, OBJPROP_YSIZE, 25);               //--- Set edit field height
   ObjectSetString(0, LOT_EDIT, OBJPROP_TEXT, "0.01");             //--- Set default lot size text
   ObjectSetInteger(0, LOT_EDIT, OBJPROP_COLOR, clrBlack);         //--- Set text color
   ObjectSetInteger(0, LOT_EDIT, OBJPROP_BGCOLOR, clrWhite);       //--- Set background color
   ObjectSetInteger(0, LOT_EDIT, OBJPROP_BORDER_COLOR, clrBlack);  //--- Set border color
   ObjectSetInteger(0, LOT_EDIT, OBJPROP_ALIGN, ALIGN_CENTER);     //--- Center text
   ObjectSetString(0, LOT_EDIT, OBJPROP_FONT, "Arial");            //--- Set font
   ObjectSetInteger(0, LOT_EDIT, OBJPROP_FONTSIZE, 13);            //--- Set font size
   ObjectSetInteger(0, LOT_EDIT, OBJPROP_BACK, false);             //--- Set to foreground

   // Entry price label
   ObjectCreate(0, PRICE_LABEL, OBJ_LABEL, 0, 0, 0);                  //--- Create entry price label
   ObjectSetInteger(0, PRICE_LABEL, OBJPROP_XDISTANCE, panel_x + 10); //--- Set label x-position
   ObjectSetInteger(0, PRICE_LABEL, OBJPROP_YDISTANCE, panel_y + 70); //--- Set label y-position
   ObjectSetInteger(0, PRICE_LABEL, OBJPROP_XSIZE, 230);              //--- Set label width
   ObjectSetString(0, PRICE_LABEL, OBJPROP_TEXT, "Entry: -");         //--- Set default text
   ObjectSetString(0, PRICE_LABEL, OBJPROP_FONT, "Arial Bold");       //--- Set font
   ObjectSetInteger(0, PRICE_LABEL, OBJPROP_FONTSIZE, 13);            //--- Set font size
   ObjectSetInteger(0, PRICE_LABEL, OBJPROP_COLOR, clrWhite);         //--- Set text color
   ObjectSetInteger(0, PRICE_LABEL, OBJPROP_ALIGN, ALIGN_CENTER);     //--- Center text
   ObjectSetInteger(0, PRICE_LABEL, OBJPROP_BACK, false); //--- Set to foreground

   // SL and TP labels
   ObjectCreate(0, SL_LABEL, OBJ_LABEL, 0, 0, 0);                  //--- Create stop-loss label
   ObjectSetInteger(0, SL_LABEL, OBJPROP_XDISTANCE, panel_x + 10); //--- Set label x-position
   ObjectSetInteger(0, SL_LABEL, OBJPROP_YDISTANCE, panel_y + 95); //--- Set label y-position
   ObjectSetInteger(0, SL_LABEL, OBJPROP_XSIZE, 110);              //--- Set label width
   ObjectSetString(0, SL_LABEL, OBJPROP_TEXT, "SL: -");            //--- Set default text
   ObjectSetString(0, SL_LABEL, OBJPROP_FONT, "Arial Bold");       //--- Set font
   ObjectSetInteger(0, SL_LABEL, OBJPROP_FONTSIZE, 12);            //--- Set font size
   ObjectSetInteger(0, SL_LABEL, OBJPROP_COLOR, clrYellow);        //--- Set text color
   ObjectSetInteger(0, SL_LABEL, OBJPROP_ALIGN, ALIGN_CENTER);     //--- Center text
   ObjectSetInteger(0, SL_LABEL, OBJPROP_BACK, false);             //--- Set to foreground

   ObjectCreate(0, TP_LABEL, OBJ_LABEL, 0, 0, 0);                   //--- Create take-profit label
   ObjectSetInteger(0, TP_LABEL, OBJPROP_XDISTANCE, panel_x + 130); //--- Set label x-position
   ObjectSetInteger(0, TP_LABEL, OBJPROP_YDISTANCE, panel_y + 95);  //--- Set label y-position
   ObjectSetInteger(0, TP_LABEL, OBJPROP_XSIZE, 110);               //--- Set label width
   ObjectSetString(0, TP_LABEL, OBJPROP_TEXT, "TP: -");             //--- Set default text
   ObjectSetString(0, TP_LABEL, OBJPROP_FONT, "Arial Bold");        //--- Set font
   ObjectSetInteger(0, TP_LABEL, OBJPROP_FONTSIZE, 12);             //--- Set font size
   ObjectSetInteger(0, TP_LABEL, OBJPROP_COLOR, clrLime);           //--- Set text color
   ObjectSetInteger(0, TP_LABEL, OBJPROP_ALIGN, ALIGN_CENTER);      //--- Center text
   ObjectSetInteger(0, TP_LABEL, OBJPROP_BACK, false);              //--- Set to foreground

   // Order type buttons
   createButton(BUY_STOP_BTN, "Buy Stop", panel_x + 10, panel_y + 140, 110, 30, clrWhite, clrForestGreen, 10, clrBlack, false, "Arial");    //--- Create Buy Stop button
   createButton(SELL_STOP_BTN, "Sell Stop", panel_x + 130, panel_y + 140, 110, 30, clrWhite, clrFireBrick, 10, clrBlack, false, "Arial");   //--- Create Sell Stop button
   createButton(BUY_LIMIT_BTN, "Buy Limit", panel_x + 10, panel_y + 180, 110, 30, clrWhite, clrForestGreen, 10, clrBlack, false, "Arial");  //--- Create Buy Limit button
   createButton(SELL_LIMIT_BTN, "Sell Limit", panel_x + 130, panel_y + 180, 110, 30, clrWhite, clrFireBrick, 10, clrBlack, false, "Arial"); //--- Create Sell Limit button

   // Place Order and Cancel buttons
   createButton(PLACE_ORDER_BTN, "Place Order", panel_x + 10, panel_y + 240, 110, 30, clrWhite, clrDodgerBlue, 10, clrBlack, false, "Arial"); //--- Create Place Order button
   createButton(CANCEL_BTN, "Cancel", panel_x + 130, panel_y + 240, 110, 30, clrWhite, clrSlateGray, 10, clrBlack, false, "Arial");           //--- Create Cancel button
}
```

Here, we define the "createControlPanel" function to construct the main [graphical user interface](https://en.wikipedia.org/wiki/Graphical_user_interface "https://en.wikipedia.org/wiki/Graphical_user_interface") (GUI) for our Trade Assistant Tool. We start by using the [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) function to create a background rectangle named "PANEL\_BG" with type [OBJ\_RECTANGLE\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object), positioned at "panel\_x" and "panel\_y" (set to 10 and 30), with a size of 250x280 pixels, a dark gray background ("C'070,070,070'"), a white border ("clrWhite"), and foreground placement ( [OBJPROP\_BACK](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property) set to false).

We then call the "createButton" function to add a close button ("CLOSE\_BTN") at the top-right corner, displaying a cross symbol (character 203 from "Wingdings"), styled in crimson. The input is defined in MQL5 documentation as below but you can use one of your liking.

![MQL5 WINGDINGS TABLE](https://c.mql5.com/2/137/Screenshot_2025-04-27_020328.png)

For lot size input, we use the ObjectCreate function to create an edit field ("LOT\_EDIT") of type [OBJ\_EDIT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) at "panel\_x + 70", "panel\_y + 40", sized 110x25 pixels, initialized with "0.01", and styled with black text, white background, and centered Arial font using [ObjectSetInteger](https://www.mql5.com/en/docs/objects/ObjectSetInteger) and [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) functions.

We create three labels for displaying trade information using the "ObjectCreate" function: "PRICE\_LABEL" for entry price at "panel\_x + 10", "panel\_y + 70", spanning 230 pixels with default text "Entry: -"; "SL\_LABEL" for stop-loss at "panel\_x + 10", "panel\_y + 95", with yellow text and default "SL: -"; and "TP\_LABEL" for take-profit at "panel\_x + 130", "panel\_y + 95", with lime text and default "TP: -", all using bold Arial font and centered alignment.

Finally, we use the "createButton" function to add order type buttons—"BUY\_STOP\_BTN" and "SELL\_STOP\_BTN" at "panel\_y + 140", "BUY\_LIMIT\_BTN" and "SELL\_LIMIT\_BTN" at "panel\_y + 180"—in green and red, respectively, and action buttons "PLACE\_ORDER\_BTN" and "CANCEL\_BTN" at "panel\_y + 240" in blue and gray, all sized 110x30 pixels with Arial font. This setup forms the interactive control panel for our tool. We can call the function in the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler to initialize the panel.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {

// Create control panel
   createControlPanel();   //--- Call function to create the control panel
   ChartRedraw(0);         //--- Redraw chart to display panel
   return(INIT_SUCCEEDED); //--- Return successful initialization
}
```

On the OnInit event handler, we call the "createControlPanel" function to construct the graphical user interface, setting up the control panel with buttons, labels, and input fields. Next, we use the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function to refresh the chart, ensuring the panel is displayed immediately. Finally, we return [INIT\_SUCCEEDED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode) to indicate successful initialization. Upon compilation, we have the following output.

![THE CONTROL PANEL](https://c.mql5.com/2/137/Screenshot_2025-04-27_020918.png)

From the image, we can see that we have created the control panel. What we need to create now is the assistant panel that we can use to dynamically get the chart prices and fill automatically to the control panel and trade based on them. This will require integrating the chart scale and the screen scale but we got you covered. What we will do is have everything in a function. Then, we will add event listeners that will call respective functions when the control button is interacted with.

```
//+------------------------------------------------------------------+
//| Expert onchart event function                                    |
//+------------------------------------------------------------------+
void OnChartEvent(
   const int id,         //--- Event ID
   const long& lparam,   //--- Long parameter (e.g., x-coordinate for mouse)
   const double& dparam, //--- Double parameter (e.g., y-coordinate for mouse)
   const string& sparam  //--- String parameter (e.g., object name)
) {
   if(id == CHARTEVENT_OBJECT_CLICK) {     //--- Handle object click events
      // Handle order type buttons
      if(sparam == BUY_STOP_BTN) {         //--- Check if Buy Stop button clicked
         selected_order_type = "BUY_STOP"; //--- Set order type to Buy Stop
      }
   }
}
```

Here, we implement the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler to handle user interactions with the tool. The function receives parameters: "id" for the event type, "lparam" for data like x-coordinates, "dparam" for data like y-coordinates, and "sparam" for string data like object names. We check if "id" equals [CHARTEVENT\_OBJECT\_CLICK](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents) to detect object clicks, and if "sparam" matches "BUY\_STOP\_BTN", we set the "selected\_order\_type" variable to "BUY\_STOP", enabling us to register the user’s selection of a Buy Stop order. If that is the case, we will need a function to show the tool.

```
//+------------------------------------------------------------------+
//| Show main tool                                                   |
//+------------------------------------------------------------------+
void showTool() {
   // Hide panel
   ObjectSetInteger(0, PANEL_BG, OBJPROP_BACK, false);        //--- Hide panel background
   ObjectSetInteger(0, LOT_EDIT, OBJPROP_BACK, false);        //--- Hide lot edit field
   ObjectSetInteger(0, PRICE_LABEL, OBJPROP_BACK, false);     //--- Hide price label
   ObjectSetInteger(0, SL_LABEL, OBJPROP_BACK, false);        //--- Hide SL label
   ObjectSetInteger(0, TP_LABEL, OBJPROP_BACK, false);        //--- Hide TP label
   ObjectSetInteger(0, BUY_STOP_BTN, OBJPROP_BACK, false);    //--- Hide Buy Stop button
   ObjectSetInteger(0, SELL_STOP_BTN, OBJPROP_BACK, false);   //--- Hide Sell Stop button
   ObjectSetInteger(0, BUY_LIMIT_BTN, OBJPROP_BACK, false);   //--- Hide Buy Limit button
   ObjectSetInteger(0, SELL_LIMIT_BTN, OBJPROP_BACK, false);  //--- Hide Sell Limit button
   ObjectSetInteger(0, PLACE_ORDER_BTN, OBJPROP_BACK, false); //--- Hide Place Order button
   ObjectSetInteger(0, CANCEL_BTN, OBJPROP_BACK, false);      //--- Hide Cancel button
   ObjectSetInteger(0, CLOSE_BTN, OBJPROP_BACK, false);       //--- Hide Close button

   // Create main tool 150 pixels from the right edge
   int chart_width = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS); //--- Get chart width
   int tool_x = chart_width - 400 - 50;                              //--- Calculate tool x-position (400 is REC1 width, 50 is margin)

   if(selected_order_type == "BUY_STOP" || selected_order_type == "BUY_LIMIT") { //--- Check for buy orders
      // Buy orders: TP at top, entry in middle, SL at bottom
      createButton(REC1, "", tool_x, 20, 350, 30, clrWhite, clrGreen, 13, clrBlack, false, "Arial Black"); //--- Create TP rectangle

      xd1 = (int)ObjectGetInteger(0, REC1, OBJPROP_XDISTANCE); //--- Get REC1 x-distance
      yd1 = (int)ObjectGetInteger(0, REC1, OBJPROP_YDISTANCE); //--- Get REC1 y-distance
      xs1 = (int)ObjectGetInteger(0, REC1, OBJPROP_XSIZE);     //--- Get REC1 x-size
      ys1 = (int)ObjectGetInteger(0, REC1, OBJPROP_YSIZE);     //--- Get REC1 y-size

      xd2 = xd1;       //--- Set REC2 x-distance
      yd2 = yd1 + ys1; //--- Set REC2 y-distance
      xs2 = xs1;       //--- Set REC2 x-size
      ys2 = 100;       //--- Set REC2 y-size

      xd3 = xd2;       //--- Set REC3 x-distance
      yd3 = yd2 + ys2; //--- Set REC3 y-distance
      xs3 = xs2;       //--- Set REC3 x-size
      ys3 = 30;        //--- Set REC3 y-size

      xd4 = xd3;       //--- Set REC4 x-distance
      yd4 = yd3 + ys3; //--- Set REC4 y-distance
      xs4 = xs3;       //--- Set REC4 x-size
      ys4 = 100;       //--- Set REC4 y-size

      xd5 = xd4;       //--- Set REC5 x-distance
      yd5 = yd4 + ys4; //--- Set REC5 y-distance
      xs5 = xs4;       //--- Set REC5 x-size
      ys5 = 30;        //--- Set REC5 y-size
   }
   else { //--- Handle sell orders
      // Sell orders: SL at top, entry in middle, TP at bottom
      createButton(REC5, "", tool_x, 20, 350, 30, clrWhite, clrRed, 13, clrBlack, false, "Arial Black"); //--- Create SL rectangle

      xd5 = (int)ObjectGetInteger(0, REC5, OBJPROP_XDISTANCE); //--- Get REC5 x-distance
      yd5 = (int)ObjectGetInteger(0, REC5, OBJPROP_YDISTANCE); //--- Get REC5 y-distance
      xs5 = (int)ObjectGetInteger(0, REC5, OBJPROP_XSIZE);     //--- Get REC5 x-size
      ys5 = (int)ObjectGetInteger(0, REC5, OBJPROP_YSIZE);     //--- Get REC5 y-size

      xd2 = xd5;       //--- Set REC2 x-distance
      yd2 = yd5 + ys5; //--- Set REC2 y-distance
      xs2 = xs5;       //--- Set REC2 x-size
      ys2 = 100;       //--- Set REC2 y-size

      xd3 = xd2;       //--- Set REC3 x-distance
      yd3 = yd2 + ys2; //--- Set REC3 y-distance
      xs3 = xs2;       //--- Set REC3 x-size
      ys3 = 30;        //--- Set REC3 y-size

      xd4 = xd3;       //--- Set REC4 x-distance
      yd4 = yd3 + ys3; //--- Set REC4 y-distance
      xs4 = xs3;       //--- Set REC4 x-size
      ys4 = 100;       //--- Set REC4 y-size

      xd1 = xd4;       //--- Set REC1 x-distance
      yd1 = yd4 + ys4; //--- Set REC1 y-distance
      xs1 = xs4;       //--- Set REC1 x-size
      ys1 = 30;        //--- Set REC1 y-size
   }

   datetime dt_tp = 0, dt_sl = 0, dt_prc = 0;        //--- Variables for time
   double price_tp = 0, price_sl = 0, price_prc = 0; //--- Variables for price
   int window = 0;                                   //--- Chart window

   ChartXYToTimePrice(0, xd1, yd1 + ys1, window, dt_tp, price_tp);   //--- Convert REC1 coordinates to time and price
   ChartXYToTimePrice(0, xd3, yd3 + ys3, window, dt_prc, price_prc); //--- Convert REC3 coordinates to time and price
   ChartXYToTimePrice(0, xd5, yd5 + ys5, window, dt_sl, price_sl);   //--- Convert REC5 coordinates to time and price

   createHL(TP_HL, dt_tp, price_tp, clrTeal);   //--- Create TP horizontal line
   createHL(PR_HL, dt_prc, price_prc, clrBlue); //--- Create entry horizontal line
   createHL(SL_HL, dt_sl, price_sl, clrRed);    //--- Create SL horizontal line

   if(selected_order_type == "BUY_STOP" || selected_order_type == "BUY_LIMIT") {                              //--- Check for buy orders
      createButton(REC2, "", xd2, yd2, xs2, ys2, clrWhite, clrHoneydew, 12, clrBlack, true);                  //--- Create REC2
      createButton(REC3, "", xd3, yd3, xs3, ys3, clrBlack, clrLightGray, 13, clrBlack, false, "Arial Black"); //--- Create REC3
      createButton(REC4, "", xd4, yd4, xs4, ys4, clrWhite, clrLinen, 12, clrBlack, true);                     //--- Create REC4
      createButton(REC5, "", xd5, yd5, xs5, ys5, clrWhite, clrRed, 13, clrBlack, false, "Arial Black");       //--- Create REC5
   }
   else { //--- Handle sell orders
      createButton(REC2, "", xd2, yd2, xs2, ys2, clrWhite, clrHoneydew, 12, clrBlack, true);                  //--- Create REC2
      createButton(REC3, "", xd3, yd3, xs3, ys3, clrBlack, clrLightGray, 13, clrBlack, false, "Arial Black"); //--- Create REC3
      createButton(REC4, "", xd4, yd4, xs4, ys4, clrWhite, clrLinen, 12, clrBlack, true);                     //--- Create REC4
      createButton(REC1, "", xd1, yd1, xs1, ys1, clrWhite, clrGreen, 13, clrBlack, false, "Arial Black");     //--- Create REC1
   }

   update_Text(REC1, "TP: " + DoubleToString(MathAbs((Get_Price_d(TP_HL) - Get_Price_d(PR_HL)) / _Point), 0) + " Points | " + Get_Price_s(TP_HL)); //--- Update REC1 text
   update_Text(REC3, selected_order_type + ": | Lot: " + DoubleToString(lot_size, 2) + " | " + Get_Price_s(PR_HL));                                //--- Update REC3 text
   update_Text(REC5, "SL: " + DoubleToString(MathAbs((Get_Price_d(PR_HL) - Get_Price_d(SL_HL)) / _Point), 0) + " Points | " + Get_Price_s(SL_HL)); //--- Update REC5 text
   update_Text(PRICE_LABEL, "Entry: " + Get_Price_s(PR_HL)); //--- Update entry label text
   update_Text(SL_LABEL, "SL: " + Get_Price_s(SL_HL));       //--- Update SL label text
   update_Text(TP_LABEL, "TP: " + Get_Price_s(TP_HL));       //--- Update TP label text

   tool_visible = true;                              //--- Set tool visibility flag
   ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, true); //--- Enable mouse move events
   ChartRedraw(0);                                   //--- Redraw chart
}
```

To show the chart price tool, we implement the "showTool" function and begin by hiding the control panel using the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/ObjectSetInteger) function, setting [OBJPROP\_BACK](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property) to false for objects like "PANEL\_BG", "LOT\_EDIT", "PRICE\_LABEL", "SL\_LABEL", "TP\_LABEL", "BUY\_STOP\_BTN", "SELL\_STOP\_BTN", "BUY\_LIMIT\_BTN", "SELL\_LIMIT\_BTN", "PLACE\_ORDER\_BTN", "CANCEL\_BTN", and "CLOSE\_BTN".

We calculate the tool’s x-position with the [ChartGetInteger](https://www.mql5.com/en/docs/chart_operations/chartgetinteger) function to get [CHART\_WIDTH\_IN\_PIXELS](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer), setting "tool\_x" 450 pixels from the right edge. For "BUY\_STOP" or "BUY\_LIMIT" orders, we use the "createButton" function to create "REC1" (TP) at "tool\_x", y=20, sized 350x30 in green, and set variables "xd1", "yd1", "xs1", "ys1" via "ObjectGetInteger", then position "REC2" to "REC5" vertically (TP, entry, SL) with "xd2" to "xd5", "yd2" to "yd5", "xs2" to "xs5", "ys2" to "ys5".

For sell orders, we create "REC5" (SL) in red and arrange "REC2" to "REC1" (SL, entry, TP).

We declare "dt\_tp", "dt\_sl", "dt\_prc" for time, "price\_tp", "price\_sl", and "price\_prc" for prices, and "window" for the chart, using the [ChartXYToTimePrice](https://www.mql5.com/en/docs/chart_operations/chartxytotimeprice) function to convert coordinates of "REC1", "REC3", and "REC5" to price and time. We call the "createHL" function to draw "TP\_HL", "PR\_HL", and "SL\_HL" in teal, blue, and red.

Depending on "selected\_order\_type", we use "createButton" to create the remaining rectangles ("REC2", "REC3", "REC4", "REC5" for buy; "REC2", "REC3", "REC4", "REC1" for sell) with appropriate colors. We update text using the "update\_Text" function for "REC1", "REC3", "REC5", "PRICE\_LABEL", "SL\_LABEL", and "TP\_LABEL", calculating point differences with "Get\_Price\_d", "Get\_Price\_s", [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring), and [MathAbs](https://www.mql5.com/en/docs/math/mathabs).

Finally, we set "tool\_visible" to true, enable mouse events with [ChartSetInteger](https://www.mql5.com/en/docs/chart_operations/chartsetinteger), and call [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to display the tool. To create the horizontal lines, we use the following function.

```
//+------------------------------------------------------------------+
//| Create horizontal line                                           |
//+------------------------------------------------------------------+
bool createHL(string objName, datetime time1, double price1, color clr) {
   ResetLastError();                                                              //--- Reset last error code
   if(!ObjectCreate(0, objName, OBJ_HLINE, 0, time1, price1)) {                   //--- Create horizontal line
      Print(__FUNCTION__, ": Failed to create HL: Error Code: ", GetLastError()); //--- Print error message
      return false; //--- Return failure
   }
   ObjectSetInteger(0, objName, OBJPROP_TIME, time1);             //--- Set line time
   ObjectSetDouble(0, objName, OBJPROP_PRICE, price1);            //--- Set line price
   ObjectSetInteger(0, objName, OBJPROP_COLOR, clr);              //--- Set line color
   ObjectSetInteger(0, objName, OBJPROP_BACK, false);             //--- Set to foreground
   ObjectSetInteger(0, objName, OBJPROP_STYLE, STYLE_DASHDOTDOT); //--- Set line style

   ChartRedraw(0); //--- Redraw chart
   return true;    //--- Return success
}
```

Here, we just create the [OBJ\_HLINE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) object and set the necessary object parameters as we did with the function for creating buttons earlier on. We will also need a function to show the panel as below.

```
//+------------------------------------------------------------------+
//| Show control panel                                               |
//+------------------------------------------------------------------+
void showPanel() {
   // Show panel
   ObjectSetInteger(0, PANEL_BG, OBJPROP_BACK, false);        //--- Show panel background
   ObjectSetInteger(0, LOT_EDIT, OBJPROP_BACK, false);        //--- Show lot edit field
   ObjectSetInteger(0, PRICE_LABEL, OBJPROP_BACK, false);     //--- Show price label
   ObjectSetInteger(0, SL_LABEL, OBJPROP_BACK, false);        //--- Show SL label
   ObjectSetInteger(0, TP_LABEL, OBJPROP_BACK, false);        //--- Show TP label
   ObjectSetInteger(0, BUY_STOP_BTN, OBJPROP_BACK, false);    //--- Show Buy Stop button
   ObjectSetInteger(0, SELL_STOP_BTN, OBJPROP_BACK, false);   //--- Show Sell Stop button
   ObjectSetInteger(0, BUY_LIMIT_BTN, OBJPROP_BACK, false);   //--- Show Buy Limit button
   ObjectSetInteger(0, SELL_LIMIT_BTN, OBJPROP_BACK, false);  //--- Show Sell Limit button
   ObjectSetInteger(0, PLACE_ORDER_BTN, OBJPROP_BACK, false); //--- Show Place Order button
   ObjectSetInteger(0, CANCEL_BTN, OBJPROP_BACK, false);      //--- Show Cancel button
   ObjectSetInteger(0, CLOSE_BTN, OBJPROP_BACK, false);       //--- Show Close button

   // Reset panel state
   update_Text(PRICE_LABEL, "Entry: -");              //--- Reset entry label text
   update_Text(SL_LABEL, "SL: -");                    //--- Reset SL label text
   update_Text(TP_LABEL, "TP: -");                    //--- Reset TP label text
   update_Text(PLACE_ORDER_BTN, "Place Order");       //--- Reset Place Order button text
   selected_order_type = "";                          //--- Clear selected order type
   tool_visible = false;                              //--- Hide tool
   ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, false); //--- Disable mouse move events
   ChartRedraw(0);                                    //--- Redraw chart
}
```

We define the "showPanel" function to display our control panel. We use the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/ObjectSetInteger) function to set "OBJPROP\_BACK" to false for "PANEL\_BG", "LOT\_EDIT", "PRICE\_LABEL", "SL\_LABEL", "TP\_LABEL", "BUY\_STOP\_BTN", "SELL\_STOP\_BTN", "BUY\_LIMIT\_BTN", "SELL\_LIMIT\_BTN", "PLACE\_ORDER\_BTN", "CANCEL\_BTN", and "CLOSE\_BTN", making them visible.

We reset the state with the "update\_Text" function, setting "PRICE\_LABEL" to "Entry: -", "SL\_LABEL" to "SL: -", "TP\_LABEL" to "TP: -", and "PLACE\_ORDER\_BTN" to "Place Order", clear "selected\_order\_type", set "tool\_visible" to false, disable mouse events via [ChartSetInteger](https://www.mql5.com/en/docs/chart_operations/chartsetinteger), and call [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to update the chart.

To delete the tool and panel, we use the following functions, by calling the [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete) function on respective objects.

```
//+------------------------------------------------------------------+
//| Delete main tool objects                                         |
//+------------------------------------------------------------------+
void deleteObjects() {
   ObjectDelete(0, REC1);  //--- Delete REC1 object
   ObjectDelete(0, REC2);  //--- Delete REC2 object
   ObjectDelete(0, REC3);  //--- Delete REC3 object
   ObjectDelete(0, REC4);  //--- Delete REC4 object
   ObjectDelete(0, REC5);  //--- Delete REC5 object
   ObjectDelete(0, TP_HL); //--- Delete TP horizontal line
   ObjectDelete(0, SL_HL); //--- Delete SL horizontal line
   ObjectDelete(0, PR_HL); //--- Delete entry horizontal line
   ChartRedraw(0);         //--- Redraw chart
}

//+------------------------------------------------------------------+
//| Delete control panel objects                                     |
//+------------------------------------------------------------------+
void deletePanel() {
   ObjectDelete(0, PANEL_BG);        //--- Delete panel background
   ObjectDelete(0, LOT_EDIT);        //--- Delete lot edit field
   ObjectDelete(0, PRICE_LABEL);     //--- Delete price label
   ObjectDelete(0, SL_LABEL);        //--- Delete SL label
   ObjectDelete(0, TP_LABEL);        //--- Delete TP label
   ObjectDelete(0, BUY_STOP_BTN);    //--- Delete Buy Stop button
   ObjectDelete(0, SELL_STOP_BTN);   //--- Delete Sell Stop button
   ObjectDelete(0, BUY_LIMIT_BTN);   //--- Delete Buy Limit button
   ObjectDelete(0, SELL_LIMIT_BTN);  //--- Delete Sell Limit button
   ObjectDelete(0, PLACE_ORDER_BTN); //--- Delete Place Order button
   ObjectDelete(0, CANCEL_BTN);      //--- Delete Cancel button
   ObjectDelete(0, CLOSE_BTN);       //--- Delete Close button
   ChartRedraw(0);                   //--- Redraw chart
}
```

To place the orders when the respective button is clicked, we use the following function.

```
//+------------------------------------------------------------------+
//| Place order based on selected type                               |
//+------------------------------------------------------------------+
void placeOrder() {
   double price = Get_Price_d(PR_HL);               //--- Get entry price
   double sl = Get_Price_d(SL_HL);                  //--- Get stop-loss price
   double tp = Get_Price_d(TP_HL);                  //--- Get take-profit price
   string symbol = Symbol();                        //--- Get current symbol
   datetime expiration = TimeCurrent() + 3600 * 24; //--- Set 24-hour order expiration

   // Validate lot size
   if(lot_size <= 0) {                       //--- Check if lot size is valid
      Print("Invalid lot size: ", lot_size); //--- Print error message
      return;                                //--- Exit function
   }

   // Validate prices
   if(price <= 0 || sl <= 0 || tp <= 0) {                                                          //--- Check if prices are valid
      Print("Invalid prices: Entry=", price, ", SL=", sl, ", TP=", tp, " (all must be positive)"); //--- Print error message
      return;                                                                                      //--- Exit function
   }

   // Validate price relationships based on order type
   if(selected_order_type == "BUY_STOP" || selected_order_type == "BUY_LIMIT") {                     //--- Check for buy orders
      if(sl >= price) {                                                                              //--- Check if SL is below entry
         Print("Invalid SL for ", selected_order_type, ": SL=", sl, " must be below Entry=", price); //--- Print error message
         return;                                                                                     //--- Exit function
      }
      if(tp <= price) {                                                                              //--- Check if TP is above entry
         Print("Invalid TP for ", selected_order_type, ": TP=", tp, " must be above Entry=", price); //--- Print error message
         return;                                                                                     //--- Exit function
      }
   }
   else if(selected_order_type == "SELL_STOP" || selected_order_type == "SELL_LIMIT") {              //--- Check for sell orders
      if(sl <= price) {                                                                              //--- Check if SL is above entry
         Print("Invalid SL for ", selected_order_type, ": SL=", sl, " must be above Entry=", price); //--- Print error message
         return;                                                                                     //--- Exit function
      }
      if(tp >= price) {                                                                              //--- Check if TP is below entry
         Print("Invalid TP for ", selected_order_type, ": TP=", tp, " must be below Entry=", price); // AMPK--- Print error message
         return;                                                                                     //--- Exit function
      }
   }
   else {                                                 //--- Handle invalid order type
      Print("Invalid order type: ", selected_order_type); //--- Print error message
      return;                                             //--- Exit function
   }

   // Place the order
   if(selected_order_type == "BUY_STOP") {                                                              //--- Handle Buy Stop order
      if(!obj_Trade.BuyStop(lot_size, price, symbol, sl, tp, ORDER_TIME_DAY, expiration)) {             //--- Attempt to place Buy Stop order
         Print("Buy Stop failed: Entry=", price, ", SL=", sl, ", TP=", tp, ", Error=", GetLastError()); //--- Print error message
      }
      else {                                                                //--- Order placed successfully
         Print("Buy Stop placed: Entry=", price, ", SL=", sl, ", TP=", tp); //--- Print success message
      }
   }
   else if(selected_order_type == "SELL_STOP") {                                                         //--- Handle Sell Stop order
      if(!obj_Trade.SellStop(lot_size, price, symbol, sl, tp, ORDER_TIME_DAY, expiration)) {             //--- Attempt to place Sell Stop order
         Print("Sell Stop failed: Entry=", price, ", SL=", sl, ", TP=", tp, ", Error=", GetLastError()); //--- Print error message
      }
      else {                                                                 //--- Order placed successfully
         Print("Sell Stop placed: Entry=", price, ", SL=", sl, ", TP=", tp); //--- Print success message
      }
   }
   else if(selected_order_type == "BUY_LIMIT") {                                                         //--- Handle Buy Limit order
      if(!obj_Trade.BuyLimit(lot_size, price, symbol, sl, tp, ORDER_TIME_DAY, expiration)) {             //--- Attempt to place Buy Limit order
         Print("Buy Limit failed: Entry=", price, ", SL=", sl, ", TP=", tp, ", Error=", GetLastError()); //--- Print error message
      }
      else {                                                                 //--- Order placed successfully
         Print("Buy Limit placed: Entry=", price, ", SL=", sl, ", TP=", tp); //--- Print success message
      }
   }
   else if(selected_order_type == "SELL_LIMIT") {                                                         //--- Handle Sell Limit order
      if(!obj_Trade.SellLimit(lot_size, price, symbol, sl, tp, ORDER_TIME_DAY, expiration)) {             //--- Attempt to place Sell Limit order
         Print("Sell Limit failed: Entry=", price, ", SL=", sl, ", TP=", tp, ", Error=", GetLastError()); //--- Print error message
      }
      else {                                                                  //--- Order placed successfully
         Print("Sell Limit placed: Entry=", price, ", SL=", sl, ", TP=", tp); //--- Print success message
      }
   }
}
```

We implement the "placeOrder" function to execute pending orders for our tool and we start by retrieving the entry price ("price"), stop-loss ("sl"), and take-profit ("tp") using the "Get\_Price\_d" function for "PR\_HL", "SL\_HL", and "TP\_HL", get the current "symbol" with the [Symbol](https://www.mql5.com/en/docs/check/symbol) function, and set a 24-hour "expiration" using the [TimeCurrent](https://www.mql5.com/en/docs/dateandtime/timecurrent) function.

We validate "lot\_size" (>0) and ensure "price", "sl", and "tp" are positive, exiting with the [Print](https://www.mql5.com/en/docs/common/print) function if invalid. For "BUY\_STOP" or "BUY\_LIMIT", we check "sl" is below "price" and "tp" is above, and for "SELL\_STOP" or "SELL\_LIMIT", "sl" is above and "tp" is below, using "Print" to log errors and exit if conditions fail. If "selected\_order\_type" is invalid, we exit with a "Print" message.

We then use "obj\_Trade" methods—"BuyStop", "SellStop", "BuyLimit", or "SellLimit"—to place the order, logging success or failure with "Print" and [GetLastError](https://www.mql5.com/en/docs/common/ResetLastError) if it fails. Armed with these functions, we can call them on respective button clicks as below.

```
if(id == CHARTEVENT_OBJECT_CLICK) {                    //--- Handle object click events
   // Handle order type buttons
   if(sparam == BUY_STOP_BTN) {                        //--- Check if Buy Stop button clicked
      selected_order_type = "BUY_STOP";                //--- Set order type to Buy Stop
      showTool();                                      //--- Show trading tool
      update_Text(PLACE_ORDER_BTN, "Place Buy Stop");  //--- Update place order button text
   }
   else if(sparam == SELL_STOP_BTN) {                  //--- Check if Sell Stop button clicked
      selected_order_type = "SELL_STOP";               //--- Set order type to Sell Stop
      showTool();                                      //--- Show trading tool
      update_Text(PLACE_ORDER_BTN, "Place Sell Stop"); //--- Update place order button text
   }
   else if(sparam == BUY_LIMIT_BTN) {                  //--- Check if Buy Limit button clicked
      selected_order_type = "BUY_LIMIT";               //--- Set order type to Buy Limit
      showTool();                                      //--- Show trading tool
      update_Text(PLACE_ORDER_BTN, "Place Buy Limit"); //--- Update place order button text
   }
   else if(sparam == SELL_LIMIT_BTN) {                 //--- Check if Sell Limit button clicked
      selected_order_type = "SELL_LIMIT";              //--- Set order type to Sell Limit
      showTool();                                      //--- Show trading tool
      update_Text(PLACE_ORDER_BTN, "Place Sell Limit");//--- Update place order button text
   }
   else if(sparam == PLACE_ORDER_BTN) {                //--- Check if Place Order button clicked
      placeOrder();                                    //--- Execute order placement
      deleteObjects();                                 //--- Delete tool objects
      showPanel();                                     //--- Show control panel
   }
   else if(sparam == CANCEL_BTN) {                     //--- Check if Cancel button clicked
      deleteObjects();                                 //--- Delete tool objects
      showPanel();                                     //--- Show control panel
   }
   else if(sparam == CLOSE_BTN) {                      //--- Check if Close button clicked
      deleteObjects();                                 //--- Delete tool objects
      deletePanel();                                   //--- Delete control panel
   }
   ObjectSetInteger(0, sparam, OBJPROP_STATE, false);  //--- Reset button state
   ChartRedraw(0);                                     //--- Redraw chart
}
```

Upon compilation, we have the following outcome.

![PANEL + PRICE CHART TOOL](https://c.mql5.com/2/137/Screenshot_2025-04-27_023509.png)

From the image, we can see that we can create the respective price chart tool dynamically. What we need to do next is make the tool intractable by being able to move it along the chart. Here is the logic we apply in the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler.

```
//+------------------------------------------------------------------+
//| Chart event handler                                              |
//+------------------------------------------------------------------+
int prevMouseState = 0; //--- Variable to track previous mouse state

int mlbDownX1 = 0, mlbDownY1 = 0, mlbDownXD_R1 = 0, mlbDownYD_R1 = 0; //--- Variables for mouse down coordinates for REC1
int mlbDownX2 = 0, mlbDownY2 = 0, mlbDownXD_R2 = 0, mlbDownYD_R2 = 0; //--- Variables for mouse down coordinates for REC2
int mlbDownX3 = 0, mlbDownY3 = 0, mlbDownXD_R3 = 0, mlbDownYD_R3 = 0; //--- Variables for mouse down coordinates for REC3
int mlbDownX4 = 0, mlbDownY4 = 0, mlbDownXD_R4 = 0, mlbDownYD_R4 = 0; //--- Variables for mouse down coordinates for REC4
int mlbDownX5 = 0, mlbDownY5 = 0, mlbDownXD_R5 = 0, mlbDownYD_R5 = 0; //--- Variables for mouse down coordinates for REC5

bool movingState_R1 = false; //--- Flag for REC1 movement state
bool movingState_R3 = false; //--- Flag for REC3 movement state
bool movingState_R5 = false; //--- Flag for REC5 movement state
```

First, we define variables for the OnChartEvent function to enable drag-and-drop in our Trade Assistant Tool. "prevMouseState" tracks mouse state changes, while "mlbDownX1", "mlbDownY1", "mlbDownXD\_R1", "mlbDownYD\_R1" (and similar for "REC2" to "REC5") store mouse and rectangle coordinates for "REC1" (TP), "REC3" (entry), and "REC5" (SL) during clicks. Boolean flags "movingState\_R1", "movingState\_R3", and "movingState\_R5" indicate if these rectangles are being dragged. We can then use these control variables to define the movement of the price tool.

```
if(id == CHARTEVENT_MOUSE_MOVE && tool_visible) { //--- Handle mouse move events when tool is visible
   int MouseD_X = (int)lparam; //--- Get mouse x-coordinate
   int MouseD_Y = (int)dparam; //--- Get mouse y-coordinate
   int MouseState = (int)sparam; //--- Get mouse state

   int XD_R1 = (int)ObjectGetInteger(0, REC1, OBJPROP_XDISTANCE); //--- Get REC1 x-distance
   int YD_R1 = (int)ObjectGetInteger(0, REC1, OBJPROP_YDISTANCE); //--- Get REC1 y-distance
   int XS_R1 = (int)ObjectGetInteger(0, REC1, OBJPROP_XSIZE);     //--- Get REC1 x-size
   int YS_R1 = (int)ObjectGetInteger(0, REC1, OBJPROP_YSIZE);     //--- Get REC1 y-size

   int XD_R2 = (int)ObjectGetInteger(0, REC2, OBJPROP_XDISTANCE); //--- Get REC2 x-distance
   int YD_R2 = (int)ObjectGetInteger(0, REC2, OBJPROP_YDISTANCE); //--- Get REC2 y-distance
   int XS_R2 = (int)ObjectGetInteger(0, REC2, OBJPROP_XSIZE);     //--- Get REC2 x-size
   int YS_R2 = (int)ObjectGetInteger(0, REC2, OBJPROP_YSIZE);     //--- Get REC2 y-size

   int XD_R3 = (int)ObjectGetInteger(0, REC3, OBJPROP_XDISTANCE); //--- Get REC3 x-distance
   int YD_R3 = (int)ObjectGetInteger(0, REC3, OBJPROP_YDISTANCE); //--- Get REC3 y-distance
   int XS_R3 = (int)ObjectGetInteger(0, REC3, OBJPROP_XSIZE);     //--- Get REC3 x-size
   int YS_R3 = (int)ObjectGetInteger(0, REC3, OBJPROP_YSIZE);     //--- Get REC3 y-size

   int XD_R4 = (int)ObjectGetInteger(0, REC4, OBJPROP_XDISTANCE); //--- Get REC4 x-distance
   int YD_R4 = (int)ObjectGetInteger(0, REC4, OBJPROP_YDISTANCE); //--- Get REC4 y-distance
   int XS_R4 = (int)ObjectGetInteger(0, REC4, OBJPROP_XSIZE);     //--- Get REC4 x-size
   int YS_R4 = (int)ObjectGetInteger(0, REC4, OBJPROP_YSIZE);     //--- Get REC4 y-size

   int XD_R5 = (int)ObjectGetInteger(0, REC5, OBJPROP_XDISTANCE); //--- Get REC5 x-distance
   int YD_R5 = (int)ObjectGetInteger(0, REC5, OBJPROP_YDISTANCE); //--- Get REC5 y-distance
   int XS_R5 = (int)ObjectGetInteger(0, REC5, OBJPROP_XSIZE);     //--- Get REC5 x-size
   int YS_R5 = (int)ObjectGetInteger(0, REC5, OBJPROP_YSIZE);     //--- Get REC5 y-size

   if(prevMouseState == 0 && MouseState == 1) { //--- Check for mouse button down
      mlbDownX1 = MouseD_X; //--- Store mouse x-coordinate for REC1
      mlbDownY1 = MouseD_Y; //--- Store mouse y-coordinate for REC1
      mlbDownXD_R1 = XD_R1; //--- Store REC1 x-distance
      mlbDownYD_R1 = YD_R1; //--- Store REC1 y-distance

      mlbDownX2 = MouseD_X; //--- Store mouse x-coordinate for REC2
      mlbDownY2 = MouseD_Y; //--- Store mouse y-coordinate for REC2
      mlbDownXD_R2 = XD_R2; //--- Store REC2 x-distance
      mlbDownYD_R2 = YD_R2; //--- Store REC2 y-distance

      mlbDownX3 = MouseD_X; //--- Store mouse x-coordinate for REC3
      mlbDownY3 = MouseD_Y; //--- Store mouse y-coordinate for REC3
      mlbDownXD_R3 = XD_R3; //--- Store REC3 x-distance
      mlbDownYD_R3 = YD_R3; //--- Store REC3 y-distance

      mlbDownX4 = MouseD_X; //--- Store mouse x-coordinate for REC4
      mlbDownY4 = MouseD_Y; //--- Store mouse y-coordinate for REC4
      mlbDownXD_R4 = XD_R4; //--- Store REC4 x-distance
      mlbDownYD_R4 = YD_R4; //--- Store REC4 y-distance

      mlbDownX5 = MouseD_X; //--- Store mouse x-coordinate for REC5
      mlbDownY5 = MouseD_Y; //--- Store mouse y-coordinate for REC5
      mlbDownXD_R5 = XD_R5; //--- Store REC5 x-distance
      mlbDownYD_R5 = YD_R5; //--- Store REC5 y-distance

      if(MouseD_X >= XD_R1 && MouseD_X <= XD_R1 + XS_R1 && //--- Check if mouse is within REC1 bounds
         MouseD_Y >= YD_R1 && MouseD_Y <= YD_R1 + YS_R1) {
         movingState_R1 = true;                            //--- Enable REC1 movement
      }
      if(MouseD_X >= XD_R3 && MouseD_X <= XD_R3 + XS_R3 && //--- Check if mouse is within REC3 bounds
         MouseD_Y >= YD_R3 && MouseD_Y <= YD_R3 + YS_R3) {
         movingState_R3 = true;                            //--- Enable REC3 movement
      }
      if(MouseD_X >= XD_R5 && MouseD_X <= XD_R5 + XS_R5 && //--- Check if mouse is within REC5 bounds
         MouseD_Y >= YD_R5 && MouseD_Y <= YD_R5 + YS_R5) {
         movingState_R5 = true;                            //--- Enable REC5 movement
      }
   }
   if(movingState_R1) {                                                                        //--- Handle REC1 (TP) movement
      ChartSetInteger(0, CHART_MOUSE_SCROLL, false);                                           //--- Disable chart scrolling
      bool canMove = false;                                                                    //--- Flag to check if movement is valid
      if(selected_order_type == "BUY_STOP" || selected_order_type == "BUY_LIMIT") {            //--- Check for buy orders
         if(YD_R1 + YS_R1 < YD_R3) {                                                           //--- Ensure TP is above entry for buy orders
            canMove = true;                                                                    //--- Allow movement
            ObjectSetInteger(0, REC1, OBJPROP_YDISTANCE, mlbDownYD_R1 + MouseD_Y - mlbDownY1); //--- Update REC1 y-position
            ObjectSetInteger(0, REC2, OBJPROP_YDISTANCE, YD_R1 + YS_R1);                       //--- Update REC2 y-position
            ObjectSetInteger(0, REC2, OBJPROP_YSIZE, YD_R3 - (YD_R1 + YS_R1));                 //--- Update REC2 y-size
         }
      }
      else {                                                                                   //--- Handle sell orders
         if(YD_R1 > YD_R3 + YS_R3) {                                                           //--- Ensure TP is below entry for sell orders
            canMove = true;                                                                    //--- Allow movement
            ObjectSetInteger(0, REC1, OBJPROP_YDISTANCE, mlbDownYD_R1 + MouseD_Y - mlbDownY1); //--- Update REC1 y-position
            ObjectSetInteger(0, REC4, OBJPROP_YDISTANCE, YD_R3 + YS_R3);                       //--- Update REC4 y-position
            ObjectSetInteger(0, REC4, OBJPROP_YSIZE, YD_R1 - (YD_R3 + YS_R3));                 //--- Update REC4 y-size
         }
      }

      if(canMove) {           //--- If movement is valid
         datetime dt_TP = 0;  //--- Variable for TP time
         double price_TP = 0; //--- Variable for TP price
         int window = 0;      //--- Chart window

         ChartXYToTimePrice(0, XD_R1, YD_R1 + YS_R1, window, dt_TP, price_TP); //--- Convert chart coordinates to time and price
         ObjectSetInteger(0, TP_HL, OBJPROP_TIME, dt_TP);                      //--- Update TP horizontal line time
         ObjectSetDouble(0, TP_HL, OBJPROP_PRICE, price_TP);                   //--- Update TP horizontal line price

         update_Text(REC1, "TP: " + DoubleToString(MathAbs((Get_Price_d(TP_HL) - Get_Price_d(PR_HL)) / _Point), 0) + " Points | " + Get_Price_s(TP_HL)); //--- Update REC1 text
         update_Text(TP_LABEL, "TP: " + Get_Price_s(TP_HL));                                                                                             //--- Update TP label text
      }

      ChartRedraw(0); //--- Redraw chart
   }

   if(movingState_R5) {                                                                        //--- Handle REC5 (SL) movement
      ChartSetInteger(0, CHART_MOUSE_SCROLL, false);                                           //--- Disable chart scrolling
      bool canMove = false;                                                                    //--- Flag to check if movement is valid
      if(selected_order_type == "BUY_STOP" || selected_order_type == "BUY_LIMIT") {            //--- Check for buy orders
         if(YD_R5 > YD_R4) {                                                                   //--- Ensure SL is below entry for buy orders
            canMove = true;                                                                    //--- Allow movement
            ObjectSetInteger(0, REC5, OBJPROP_YDISTANCE, mlbDownYD_R5 + MouseD_Y - mlbDownY5); //--- Update REC5 y-position
            ObjectSetInteger(0, REC4, OBJPROP_YDISTANCE, YD_R3 + YS_R3);                       //--- Update REC4 y-position
            ObjectSetInteger(0, REC4, OBJPROP_YSIZE, YD_R5 - (YD_R3 + YS_R3));                 //--- Update REC4 y-size
         }
      }
      else {                                                                                   //--- Handle sell orders
         if(YD_R5 + YS_R5 < YD_R3) {                                                           //--- Ensure SL is above entry for sell orders
            canMove = true;                                                                    //--- Allow movement
            ObjectSetInteger(0, REC5, OBJPROP_YDISTANCE, mlbDownYD_R5 + MouseD_Y - mlbDownY5); //--- Update REC5 y-position
            ObjectSetInteger(0, REC2, OBJPROP_YDISTANCE, YD_R5 + YS_R5);                       //--- Update REC2 y-position
            ObjectSetInteger(0, REC2, OBJPROP_YSIZE, YD_R3 - (YD_R5 + YS_R5));                 //--- Update REC2 y-size
         }
      }

      if(canMove) {           //--- If movement is valid
         datetime dt_SL = 0;  //--- Variable for SL time
         double price_SL = 0; //--- Variable for SL price
         int window = 0;      //--- Chart window

         ChartXYToTimePrice(0, XD_R5, YD_R5 + YS_R5, window, dt_SL, price_SL); //--- Convert chart coordinates to time and price
         ObjectSetInteger(0, SL_HL, OBJPROP_TIME, dt_SL);                      //--- Update SL horizontal line time
         ObjectSetDouble(0, SL_HL, OBJPROP_PRICE, price_SL);                   //--- Update SL horizontal line price

         update_Text(REC5, "SL: " + DoubleToString(MathAbs((Get_Price_d(PR_HL) - Get_Price_d(SL_HL)) / _Point), 0) + " Points | " + Get_Price_s(SL_HL)); //--- Update REC5 text
         update_Text(SL_LABEL, "SL: " + Get_Price_s(SL_HL));                                                                                             //--- Update SL label text
      }

      ChartRedraw(0); //--- Redraw chart
   }

   if(movingState_R3) {                                                                  //--- Handle REC3 (Entry) movement
      ChartSetInteger(0, CHART_MOUSE_SCROLL, false);                                     //--- Disable chart scrolling
      ObjectSetInteger(0, REC3, OBJPROP_XDISTANCE, mlbDownXD_R3 + MouseD_X - mlbDownX3); //--- Update REC3 x-position
      ObjectSetInteger(0, REC3, OBJPROP_YDISTANCE, mlbDownYD_R3 + MouseD_Y - mlbDownY3); //--- Update REC3 y-position

      ObjectSetInteger(0, REC1, OBJPROP_XDISTANCE, mlbDownXD_R1 + MouseD_X - mlbDownX1); //--- Update REC1 x-position
      ObjectSetInteger(0, REC1, OBJPROP_YDISTANCE, mlbDownYD_R1 + MouseD_Y - mlbDownY1); //--- Update REC1 y-position

      ObjectSetInteger(0, REC2, OBJPROP_XDISTANCE, mlbDownXD_R2 + MouseD_X - mlbDownX2); //--- Update REC2 x-position
      ObjectSetInteger(0, REC2, OBJPROP_YDISTANCE, mlbDownYD_R2 + MouseD_Y - mlbDownY2); //--- Update REC2 y-position

      ObjectSetInteger(0, REC4, OBJPROP_XDISTANCE, mlbDownXD_R4 + MouseD_X - mlbDownX4); //--- Update REC4 x-position
      ObjectSetInteger(0, REC4, OBJPROP_YDISTANCE, mlbDownYD_R4 + MouseD_Y - mlbDownY4); //--- Update REC4 y-position

      ObjectSetInteger(0, REC5, OBJPROP_XDISTANCE, mlbDownXD_R5 + MouseD_X - mlbDownX5); //--- Update REC5 x-position
      ObjectSetInteger(0, REC5, OBJPROP_YDISTANCE, mlbDownYD_R5 + MouseD_Y - mlbDownY5); //--- Update REC5 y-position

      datetime dt_PRC = 0, dt_SL1 = 0, dt_TP1 = 0;        //--- Variables for time
      double price_PRC = 0, price_SL1 = 0, price_TP1 = 0; //--- Variables for price
      int window = 0;                                     //--- Chart window

      ChartXYToTimePrice(0, XD_R3, YD_R3 + YS_R3, window, dt_PRC, price_PRC); //--- Convert REC3 coordinates to time and price
      ChartXYToTimePrice(0, XD_R5, YD_R5 + YS_R5, window, dt_SL1, price_SL1); //--- Convert REC5 coordinates to time and price
      ChartXYToTimePrice(0, XD_R1, YD_R1 + YS_R1, window, dt_TP1, price_TP1); //--- Convert REC1 coordinates to time and price

      ObjectSetInteger(0, PR_HL, OBJPROP_TIME, dt_PRC);    //--- Update entry horizontal line time
      ObjectSetDouble(0, PR_HL, OBJPROP_PRICE, price_PRC); //--- Update entry horizontal line price

      ObjectSetInteger(0, TP_HL, OBJPROP_TIME, dt_TP1);    //--- Update TP horizontal line time
      ObjectSetDouble(0, TP_HL, OBJPROP_PRICE, price_TP1); //--- Update TP horizontal line price

      ObjectSetInteger(0, SL_HL, OBJPROP_TIME, dt_SL1);    //--- Update SL horizontal line time
      ObjectSetDouble(0, SL_HL, OBJPROP_PRICE, price_SL1); //--- Update SL horizontal line price

      update_Text(REC1, "TP: " + DoubleToString(MathAbs((Get_Price_d(TP_HL) - Get_Price_d(PR_HL)) / _Point), 0) + " Points | " + Get_Price_s(TP_HL)); //--- Update REC1 text
      update_Text(REC3, selected_order_type + ": | Lot: " + DoubleToString(lot_size, 2) + " | " + Get_Price_s(PR_HL));                                //--- Update REC3 text
      update_Text(REC5, "SL: " + DoubleToString(MathAbs((Get_Price_d(PR_HL) - Get_Price_d(SL_HL)) / _Point), 0) + " Points | " + Get_Price_s(SL_HL)); //--- Update REC5 text
      update_Text(PRICE_LABEL, "Entry: " + Get_Price_s(PR_HL));                                                                                       //--- Update entry label text
      update_Text(SL_LABEL, "SL: " + Get_Price_s(SL_HL));                                                                                             //--- Update SL label text
      update_Text(TP_LABEL, "TP: " + Get_Price_s(TP_HL));                                                                                             //--- Update TP label text

      ChartRedraw(0); //--- Redraw chart
   }

   if(MouseState == 0) {                            //--- Check if mouse button is released
      movingState_R1 = false;                       //--- Disable REC1 movement
      movingState_R3 = false;                       //--- Disable REC3 movement
      movingState_R5 = false;                       //--- Disable REC5 movement
      ChartSetInteger(0, CHART_MOUSE_SCROLL, true); //--- Enable chart scrolling
   }
   prevMouseState = MouseState;                     //--- Update previous mouse state
}
```

Here, we extend the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) function to handle mouse movement for dragging chart objects in our tool when "tool\_visible" is true and "id" is [CHARTEVENT\_MOUSE\_MOVE](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents). We extract "MouseD\_X", "MouseD\_Y", and "MouseState" from "lparam", "dparam", and "sparam", and use the [ObjectGetInteger](https://www.mql5.com/en/docs/objects/objectgetinteger) function to get position and size ("XD\_R1", "YD\_R1", "XS\_R1", "YS\_R1" for "REC1", and similarly for "REC2" to "REC5").

On mouse click ("prevMouseState" 0 to "MouseState" 1), we store mouse coordinates in "mlbDownX1", "mlbDownY1" and rectangle positions in "mlbDownXD\_R1", "mlbDownYD\_R1" (and for "REC2" to "REC5"), setting "movingState\_R1", "movingState\_R3", or "movingState\_R5" to true if the click is within "REC1", "REC3", or "REC5" bounds.

For "movingState\_R1" (TP), we disable scrolling with [ChartSetInteger](https://www.mql5.com/en/docs/chart_operations/chartsetinteger), validate TP position (above entry for "BUY\_STOP"/"BUY\_LIMIT", below for sell), update "REC1" and "REC2"/"REC4" positions and sizes with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger), convert coordinates to price using [ChartXYToTimePrice](https://www.mql5.com/en/docs/chart_operations/chartxytotimeprice), update "TP\_HL" with [ObjectSetDouble](https://www.mql5.com/en/docs/objects/objectsetdouble), and refresh text with "update\_Text", "Get\_Price\_d", "Get\_Price\_s", [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring), and [MathAbs](https://www.mql5.com/en/docs/math/mathabs).

Similarly, for "movingState\_R5" (SL), we adjust "REC5" and "REC4"/"REC2", update "SL\_HL", and refresh text. For "movingState\_R3" (entry), we move all rectangles and update "PR\_HL", "TP\_HL", "SL\_HL", and text.

On mouse release ("MouseState" 0), we reset "movingState" flags, enable scrolling, and update "prevMouseState", calling [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to reflect changes. Finally, what we need to do is delete the objects completely when we remove the program and update the lot size on the tick to make the changes reflect, though you can leave this as it is not very necessary.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   deleteObjects(); //--- Delete tool objects
   deletePanel();   //--- Delete control panel objects
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   // Update lot size from edit field
   string lot_text = ObjectGetString(0, LOT_EDIT, OBJPROP_TEXT); //--- Get lot size text from edit field
   double new_lot = StringToDouble(lot_text);                    //--- Convert lot size text to double
   if(new_lot > 0) lot_size = new_lot;                           //--- Update lot size if valid
}
```

Here, we implement two essential event handlers for our tool: [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) and [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick). In the OnDeinit function, triggered when the Expert Advisor is removed from the chart, we call the "deleteObjects" function to remove chart objects like "REC1" to "REC5", "TP\_HL", "SL\_HL", and "PR\_HL", and the "deletePanel" function to delete control panel objects such as "PANEL\_BG", "LOT\_EDIT", and buttons like "BUY\_STOP\_BTN", ensuring a clean exit.

In the OnTick function, executed on each price tick, we use the [ObjectGetString](https://www.mql5.com/en/docs/objects/objectgetstring) function to retrieve the text from the "LOT\_EDIT" field, convert it to a double using the [StringToDouble](https://www.mql5.com/en/docs/convert/StringToDouble) function, and update the "lot\_size" variable if the "new\_lot" value is positive, keeping our tool’s lot size synchronized with user input.

Upon compilation, we have the following output.

![FINAL OUTPUT](https://c.mql5.com/2/138/TRADE_TOOL_1_GIF_1.gif)

From the visualization, we can see that when we click on any of the trade buttons, the respective trade price tool is generated and then when dragged, it is updated in real-time and prices reflect in the trade panel ready to be used for trading purposes when the place trade button is clicked, the respective trades are placed dynamically. This verifies that we have achieved our objective, and what remains is testing the panel to ensure it is perfect, and that is handled in the next section.

### Backtesting

We did the testing and below is the compiled visualization in a single [Graphics Interchange Format](https://en.wikipedia.org/wiki/GIF "https://en.wikipedia.org/wiki/GIF") (GIF) bitmap image format.

![BACKTESTING OUTPUT](https://c.mql5.com/2/138/TRADE_TOOL_1_GIF_2_FULL.gif)

### Conclusion

In conclusion, we’ve developed an interactive Trade Assistant Tool in MQL5 that combines visual precision with intuitive controls, streamlining how we place pending orders. We’ve demonstrated how to design its user-friendly Graphical User Interface (GUI), implement it with functions like "createControlPanel" and "placeOrder", and ensure its reliability through structured implementation and validation. You can customize this tool to suit your trading style, enhancing your order placement efficiency, and look forward to the preceding parts where we’ll introduce advanced features like risk management and draggable panels. Keep tuned.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17931.zip "Download all attachments in the single ZIP archive")

[Trade\_Assistant\_GUI\_Tool.mq5](https://www.mql5.com/en/articles/download/17931/trade_assistant_gui_tool.mq5 "Download Trade_Assistant_GUI_Tool.mq5")(44.77 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/486090)**
(1)


![zhanglei 张](https://c.mql5.com/avatar/avatar_na2.png)

**[zhanglei 张](https://www.mql5.com/en/users/qq34373430)**
\|
22 Jan 2026 at 01:23

**MetaQuotes:**

New article [MQL5 Trading Tools (Part 1): building an interactive visual pending order trading assistant tool](https://www.mql5.com/en/articles/17931) has been released:

By [Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372 "29210372")

Is it possible to use the market price as the bid price and open an order directly at the market price when the stop loss is determined by adjusting the take profit and stop loss?


![Overcoming The Limitation of Machine Learning (Part 1): Lack of Interoperable Metrics](https://c.mql5.com/2/140/Overcoming_The_Limitation_of_Machine_Learning_Part_1_Lack_of_Interoperable_Metrics__LOGO.png)[Overcoming The Limitation of Machine Learning (Part 1): Lack of Interoperable Metrics](https://www.mql5.com/en/articles/17906)

There is a powerful and pervasive force quietly corrupting the collective efforts of our community to build reliable trading strategies that employ AI in any shape or form. This article establishes that part of the problems we face, are rooted in blind adherence to "best practices". By furnishing the reader with simple real-world market-based evidence, we will reason to the reader why we must refrain from such conduct, and rather adopt domain-bound best practices if our community should stand any chance of recovering the latent potential of AI.

![From Basic to Intermediate: Arrays and Strings (I)](https://c.mql5.com/2/94/Do_bfsico_ao_intermedi9rio_Array_e_Strings_I__LOGO.png)[From Basic to Intermediate: Arrays and Strings (I)](https://www.mql5.com/en/articles/15441)

In today's article, we'll start exploring some special data types. To begin, we'll define what a string is and explain how to use some basic procedures. This will allow us to work with this type of data, which can be interesting, although sometimes a little confusing for beginners. The content presented here is intended solely for educational purposes. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

![MQL5 Wizard Techniques you should know (Part 63): Using Patterns of DeMarker and Envelope Channels](https://c.mql5.com/2/140/MQL5_Wizard_Techniques_you_should_know_Part_63___LOGO.png)[MQL5 Wizard Techniques you should know (Part 63): Using Patterns of DeMarker and Envelope Channels](https://www.mql5.com/en/articles/17987)

The DeMarker Oscillator and the Envelope indicator are momentum and support/resistance tools that can be paired when developing an Expert Advisor. We therefore examine on a pattern by pattern basis what could be of use and what potentially avoid. We are using, as always, a wizard assembled Expert Advisor together with the Patterns-Usage functions that are built into the Expert Signal Class.

![Data Science and ML (Part 38): AI Transfer Learning in Forex Markets](https://c.mql5.com/2/140/Data_Science_and_ML_lPart_385_AI_Transfer_Learning_in_Forex_Markets___LOGO.png)[Data Science and ML (Part 38): AI Transfer Learning in Forex Markets](https://www.mql5.com/en/articles/17886)

The AI breakthroughs dominating headlines, from ChatGPT to self-driving cars, aren’t built from isolated models but through cumulative knowledge transferred from various models or common fields. Now, this same "learn once, apply everywhere" approach can be applied to help us transform our AI models in algorithmic trading. In this article, we are going to learn how we can leverage the information gained across various instruments to help in improving predictions on others using transfer learning.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/17931&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049503813720714412)

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