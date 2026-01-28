---
title: Automating Trading Strategies in MQL5 (Part 17): Mastering the Grid-Mart Scalping Strategy with a Dynamic Dashboard
url: https://www.mql5.com/en/articles/18038
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 4
scraped_at: 2026-01-23T17:36:31.096886
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/18038&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068437468375611756)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 16)](https://www.mql5.com/en/articles/17876), we automated the Midnight Range Breakout with the [Break of Structure](https://www.mql5.com/go?link=https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/Break-of-Structures "https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/Break-of-Structures") strategy to capture price breakouts. Now, in Part 17, we focus on automating the Grid-Mart Scalping Strategy in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5), developing an Expert Advisor that executes grid-based Martingale trades and features a dynamic dashboard for real-time monitoring. We will cover the following topics:

1. [Understanding the Grid-Mart Scalping Strategy](https://www.mql5.com/en/articles/18038#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/18038#para2)
3. [Backtesting](https://www.mql5.com/en/articles/18038#para3)
4. [Conclusion](https://www.mql5.com/en/articles/18038#para4)

By the end of this article, you will have a fully functional MQL5 program that scalps markets with precision and visualizes trading metrics—let’s dive in!

### Understanding the Grid-Mart Scalping Strategy

The Grid-Mart Scalping Strategy employs a grid-based Martingale approach, placing buy or sell orders at fixed price intervals (e.g., 2.0 pips) to capture small profits from market fluctuations, while increasing lot sizes after losses to recover capital quickly. It relies on high-frequency trading, targeting modest gains (e.g., 4 pips) per trade. However, it requires careful risk management due to the exponential lot size growth, which is capped by configurable limits like maximum grid levels and daily drawdown thresholds. This strategy thrives in volatile markets but demands precise configuration to avoid significant drawdowns during prolonged trends.

Our implementation plan involves creating an MQL5 Expert Advisor to automate the Grid-Mart strategy by calculating grid intervals, managing lot size progression, and executing trades with predefined stop-loss and take-profit levels. The program will feature a dynamic dashboard to display real-time metrics, such as spread, active lot sizes, and account status, with color-coded visuals to aid decision-making. Robust risk controls, including drawdown limits and grid size restrictions, will ensure consistent performance across market conditions. In a nutshell, this is what we aim to create.

![STRATEGY PLAN](https://c.mql5.com/2/140/Screenshot_2025-05-05_173636.png)

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Indicators folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some [global variables](https://www.mql5.com/en/docs/basis/variables/global) that we will use throughout the program.

```
//+------------------------------------------------------------------+
//|                                      GridMart Scalper MT5 EA.mq5 |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//|                                   https://t.me/Forex_Algo_Trader |
//+------------------------------------------------------------------+

#property copyright "Copyright 2025, Allan Munene Mutiiria."
#property link      "https://t.me/Forex_Algo_Trader"
#property version   "1.00"

#include <Trade/Trade.mqh>

//--- Declare trade object to execute trading operations
CTrade obj_Trade;

//--- Trading state variables
double dailyBalance = 0;           //--- Store the daily starting balance for drawdown monitoring
datetime dailyResetTime = 0;       //--- Track the last daily reset time
bool tradingEnabled = true;        //--- Indicate if trading is allowed based on drawdown limits
datetime lastBarTime = 0;          //--- Store the timestamp of the last processed bar
bool hasBuyPosition = false;       //--- Flag if there is an active buy position
bool hasSellPosition = false;      //--- Flag if there is an active sell position
bool openNewTrade = false;         //--- Signal when a new trade should be opened
int activeOrders = 0;              //--- Count the number of active orders
double latestBuyPrice = 0;         //--- Store the price of the latest buy order
double latestSellPrice = 0;        //--- Store the price of the latest sell order
double calculatedLot = 0;          //--- Hold the calculated lot size for new orders
bool modifyPositions = false;      //--- Indicate if position SL/TP need modification
double weightedPrice = 0;          //--- Track the weighted average price of open positions
double targetTP = 0;               //--- Store the target take-profit price
double targetSL = 0;               //--- Store the target stop-loss price
bool updateSLTP = false;           //--- Signal when SL/TP updates are needed
int cycleCount = 0;                //--- Count the number of trading cycles
double totalVolume = 0;            //--- Accumulate the total volume of open positions
bool dashboardVisible = true;      //--- Control visibility of the dashboard

//--- Dashboard dragging and hover state variables
bool panelDragging = false;        //--- Indicate if the dashboard is being dragged
int panelDragX = 0;                //--- Store the X-coordinate of the mouse during dragging
int panelDragY = 0;                //--- Store the Y-coordinate of the mouse during dragging
int panelStartX = 0;               //--- Store the initial X-position of the dashboard
int panelStartY = 0;               //--- Store the initial Y-position of the dashboard
bool closeButtonHovered = false;   //--- Track if the close button is hovered
bool headerHovered = false;        //--- Track if the header is hovered

input group "Main EA Settings"
input string EA_NAME = "GridMart Scalper";    // EA Name
input bool CONTINUE_TRADING = true;           // Continue Trading After Cycle
input int MAX_CYCLES = 1000;                  // Max Trading Cycles
input int START_HOUR = 0;                     // Start Trading Hour
input int END_HOUR = 23;                      // End Trading Hour
input int LOT_MODE = 1;                       // Lot Mode (1=Multiplier, 2=Fixed)
input double BASE_LOT = 0.01;                 // Base Lot Size
input int STOP_LOSS_PIPS = 100;               // Stop Loss (Pips)
input int TAKE_PROFIT_PIPS =
3 ;               // Take Profit (Pips)
input double GRID_DISTANCE = 3.0;             // Grid Distance (Pips)
input double LOT_MULTIPLIER = 1.3;            // Lot Multiplier
input int MAX_GRID_LEVELS = 30;               // Max Grid Levels
input int LOT_PRECISION = 2;                  // Lot Decimal Precision
input int MAGIC = 1234567890;                 // Magic Number
input color TEXT_COLOR = clrWhite;            // Dashboard Text Color

input group "EA Risk Management Settings"
input bool ENABLE_DAILY_DRAWDOWN = false;     // Enable Daily Drawdown Limiter
input double DRAWDOWN_LIMIT = -1.0;           // Daily Drawdown Threshold (-%)
input bool CLOSE_ON_DRAWDOWN = false;         // Close Positions When Threshold Hit

input group "Dashboard Settings"
input int PANEL_X = 30;                       // Initial X Distance (pixels)
input int PANEL_Y = 50;                       // Initial Y Distance (pixels)

//--- Pip value for price calculations
double pipValue;

//--- Dashboard constants
const int DASHBOARD_WIDTH = 300;                   //--- Width of the dashboard in pixels
const int DASHBOARD_HEIGHT = 260;                  //--- Height of the dashboard in pixels
const int HEADER_HEIGHT = 30;                      //--- Height of the header section
const int CLOSE_BUTTON_WIDTH = 40;                 //--- Width of the close button
const int CLOSE_BUTTON_HEIGHT = 28;                //--- Height of the close button
const color HEADER_NORMAL_COLOR = clrGold;         //--- Normal color of the header
const color HEADER_HOVER_COLOR = C'200,150,0';     //--- Header color when hovered
const color BACKGROUND_COLOR = clrDarkSlateGray;   //--- Background color of the dashboard
const color BORDER_COLOR = clrBlack;               //--- Border color of dashboard elements
const color SECTION_TITLE_COLOR = clrLightGray;    //--- Color for section titles
const color CLOSE_BUTTON_NORMAL_BG = clrCrimson;   //--- Normal background color of the close button
const color CLOSE_BUTTON_HOVER_BG = clrDodgerBlue; //--- Hover background color of the close button
const color CLOSE_BUTTON_NORMAL_BORDER = clrBlack; //--- Normal border color of the close button
const color CLOSE_BUTTON_HOVER_BORDER = clrBlue;   //--- Hover border color of the close button
const color VALUE_POSITIVE_COLOR = clrLimeGreen;   //--- Color for positive values (e.g., profit, low spread)
const color VALUE_NEGATIVE_COLOR = clrOrange;      //--- Color for negative or warning values (e.g., loss, high spread)
const color VALUE_LOSS_COLOR = clrHotPink;         //--- Color for negative profit
const color VALUE_ACTIVE_COLOR = clrGold;          //--- Color for active states (e.g., open orders, medium spread)
const color VALUE_DRAWDOWN_INACTIVE = clrAqua;     //--- Color for inactive drawdown state
const color VALUE_DRAWDOWN_ACTIVE = clrRed;        //--- Color for active drawdown state
const int FONT_SIZE_HEADER = 12;                   //--- Header text font size (pt)
const int FONT_SIZE_SECTION_TITLE = 11;            //--- Section title font size (pt)
const int FONT_SIZE_METRIC = 9;                    //--- Metric label/value font size (pt)
const int FONT_SIZE_BUTTON = 12;                   //--- Button font size (pt)
```

Here, we implement the strategy in MQL5, initializing the program’s core components to automate grid-based Martingale trades and support a dynamic dashboard. We declare the "CTrade" object as "obj\_Trade" using " [#include](https://www.mql5.com/en/docs/basis/preprosessor/include) <Trade/Trade.mqh>" to manage trade execution. We define variables like "dailyBalance" to track account balance, "lastBarTime" to store bar timestamps with the [iTime](https://www.mql5.com/en/docs/series/itime) function, "hasBuyPosition" and "hasSellPosition" to flag active trades, and "activeOrders" to count open positions.

We set [inputs](https://www.mql5.com/en/docs/basis/variables/inputvariables) like "GRID\_DISTANCE = 3.0" for grid intervals, "LOT\_MULTIPLIER = 1.3" for lot scaling, and "TAKE\_PROFIT\_PIPS = 3" for profit targets, using "MAGIC = 1234567890" to identify trades. We include "ENABLE\_DAILY\_DRAWDOWN" and "DRAWDOWN\_LIMIT" for risk control, with "cycleCount" and "MAX\_CYCLES" limiting trade cycles. We configure the dashboard with "DASHBOARD\_WIDTH = 300", "FONT\_SIZE\_METRIC = 9", "panelDragging" for drag functionality, "closeButtonHovered" for hover effects, and "VALUE\_POSITIVE\_COLOR = clrLimeGreen" for visuals, using "pipValue" for precise pricing. This gives us the user interface as follows.

![USER INTERFACE](https://c.mql5.com/2/140/Screenshot_2025-05-05_144248.png)

From the image, we can see that we can control the program from the defined user interface. We now need to continue to define some helper functions that we will use when needing basic and frequent actions like the currency pair or position type. Here is the logic we adapted for that.

```
//+------------------------------------------------------------------+
//| Retrieve the current account balance                             |
//+------------------------------------------------------------------+
double GetAccountBalance() {
   //--- Return the current account balance
   return AccountInfoDouble(ACCOUNT_BALANCE);
}

//+------------------------------------------------------------------+
//| Retrieve the magic number of the selected position               |
//+------------------------------------------------------------------+
long GetPositionMagic() {
   //--- Return the magic number of the selected position
   return PositionGetInteger(POSITION_MAGIC);
}

//+------------------------------------------------------------------+
//| Retrieve the open price of the selected position                 |
//+------------------------------------------------------------------+
double GetPositionOpenPrice() {
   //--- Return the open price of the selected position
   return PositionGetDouble(POSITION_PRICE_OPEN);
}

//+------------------------------------------------------------------+
//| Retrieve the stop-loss price of the selected position            |
//+------------------------------------------------------------------+
double GetPositionSL() {
   //--- Return the stop-loss price of the selected position
   return PositionGetDouble(POSITION_SL);
}

//+------------------------------------------------------------------+
//| Retrieve the take-profit price of the selected position          |
//+------------------------------------------------------------------+
double GetPositionTP() {
   //--- Return the take-profit price of the selected position
   return PositionGetDouble(POSITION_TP);
}

//+------------------------------------------------------------------+
//| Retrieve the symbol of the selected position                     |
//+------------------------------------------------------------------+
string GetPositionSymbol() {
   //--- Return the symbol of the selected position
   return PositionGetString(POSITION_SYMBOL);
}

//+------------------------------------------------------------------+
//| Retrieve the ticket number of the selected position              |
//+------------------------------------------------------------------+
ulong GetPositionTicket() {
   //--- Return the ticket number of the selected position
   return PositionGetInteger(POSITION_TICKET);
}

//+------------------------------------------------------------------+
//| Retrieve the open time of the selected position                  |
//+------------------------------------------------------------------+
datetime GetPositionOpenTime() {
   //--- Return the open time of the selected position as a datetime
   return (datetime)PositionGetInteger(POSITION_TIME);
}

//+------------------------------------------------------------------+
//| Retrieve the type of the selected position                       |
//+------------------------------------------------------------------+
int GetPositionType() {
   //--- Return the type of the selected position (buy/sell)
   return (int)PositionGetInteger(POSITION_TYPE);
}
```

We use the "GetAccountBalance" function with the [AccountInfoDouble](https://www.mql5.com/en/docs/account/accountinfodouble) function to retrieve the current account balance, enabling balance tracking for risk management.

We implement the "GetPositionMagic" function using the [PositionGetInteger](https://www.mql5.com/en/docs/trading/positiongetinteger) function to fetch a position’s magic number, the "GetPositionOpenPrice" function with [PositionGetDouble](https://www.mql5.com/en/docs/trading/positiongetdouble) to obtain the open price, and the "GetPositionSL" and "GetPositionTP" functions with "PositionGetDouble" to access stop-loss and take-profit levels, respectively, supporting precise trade calculations.

Additionally, we define the "GetPositionSymbol" function with [PositionGetString](https://www.mql5.com/en/docs/trading/positiongetstring) to verify a position’s symbol, the "GetPositionTicket" function and "GetPositionOpenTime" function with "PositionGetInteger" to track position identifiers and opening times, and the "GetPositionType" function to determine buy or sell status, facilitating accurate position monitoring and trade logic. We can now move on to creating the dashboard, and we will need helper functions to make the work easier.

```
//+------------------------------------------------------------------+
//| Create a rectangular object for the dashboard                    |
//+------------------------------------------------------------------+
void CreateRectangle(string name, int x, int y, int width, int height, color bgColor, color borderColor) {
   //--- Check if object does not exist
   if (ObjectFind(0, name) < 0) {
      //--- Create a rectangle label object
      ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   }
   //--- Set X-coordinate
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   //--- Set Y-coordinate
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   //--- Set width
   ObjectSetInteger(0, name, OBJPROP_XSIZE, width);
   //--- Set height
   ObjectSetInteger(0, name, OBJPROP_YSIZE, height);
   //--- Set background color
   ObjectSetInteger(0, name, OBJPROP_BGCOLOR, bgColor);
   //--- Set border type to flat
   ObjectSetInteger(0, name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   //--- Set border color
   ObjectSetInteger(0, name, OBJPROP_COLOR, borderColor);
   //--- Set border width
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
   //--- Set object to foreground
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
   //--- Disable object selection
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   //--- Hide object from object list
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
}

//+------------------------------------------------------------------+
//| Create a text label for the dashboard                            |
//+------------------------------------------------------------------+
void CreateTextLabel(string name, int x, int y, string text, color clr, int fontSize, string font = "Arial") {
   //--- Check if object does not exist
   if (ObjectFind(0, name) < 0) {
      //--- Create a text label object
      ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   }
   //--- Set X-coordinate
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   //--- Set Y-coordinate
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   //--- Set label text
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   //--- Set font
   ObjectSetString(0, name, OBJPROP_FONT, font);
   //--- Set font size
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, fontSize);
   //--- Set text color
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   //--- Set object to foreground
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
   //--- Disable object selection
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   //--- Hide object from object list
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
}

//+------------------------------------------------------------------+
//| Create a button for the dashboard                                |
//+------------------------------------------------------------------+
void CreateButton(string name, string text, int x, int y, int width, int height, color textColor, color bgColor, int fontSize, color borderColor, bool isBack, string font = "Arial") {
   //--- Check if object does not exist
   if (ObjectFind(0, name) < 0) {
      //--- Create a button object
      ObjectCreate(0, name, OBJ_BUTTON, 0, 0, 0);
   }
   //--- Set X-coordinate
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   //--- Set Y-coordinate
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   //--- Set width
   ObjectSetInteger(0, name, OBJPROP_XSIZE, width);
   //--- Set height
   ObjectSetInteger(0, name, OBJPROP_YSIZE, height);
   //--- Set button text
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   //--- Set font
   ObjectSetString(0, name, OBJPROP_FONT, font);
   //--- Set font size
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, fontSize);
   //--- Set text color
   ObjectSetInteger(0, name, OBJPROP_COLOR, textColor);
   //--- Set background color
   ObjectSetInteger(0, name, OBJPROP_BGCOLOR, bgColor);
   //--- Set border color
   ObjectSetInteger(0, name, OBJPROP_BORDER_COLOR, borderColor);
   //--- Set background rendering
   ObjectSetInteger(0, name, OBJPROP_BACK, isBack);
   //--- Reset button state
   ObjectSetInteger(0, name, OBJPROP_STATE, false);
   //--- Disable button selection
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   //--- Hide button from object list
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
}
```

Here, we define the "CreateRectangle" function with [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) and [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) functions to draw rectangular elements like the dashboard background and header, setting properties such as position, size, and colors for a polished layout. We implement the "CreateTextLabel" function, utilizing "ObjectCreate" and [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) to display metrics like spread and lot sizes, with customizable font and color settings for clear readability.

Additionally, we define the "CreateButton" function to add interactive buttons, such as the close button, enabling user actions with tailored styling and hover effects, ensuring a seamless and visually intuitive dashboard experience. We can now use these functions to create the dashboard elements in a new function but since we will need the total profit count, let's define the function for that.

```
//+------------------------------------------------------------------+
//| Calculate the total unrealized profit of open positions          |
//+------------------------------------------------------------------+
double CalculateTotalProfit() {
   //--- Initialize profit accumulator
   double profit = 0;
   //--- Iterate through open positions
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      //--- Get position ticket
      ulong ticket = PositionGetTicket(i);
      //--- Select position by ticket
      if (PositionSelectByTicket(ticket) && GetPositionSymbol() == _Symbol && GetPositionMagic() == MAGIC) {
         //--- Accumulate unrealized profit
         profit += PositionGetDouble(POSITION_PROFIT);
      }
   }
   //--- Return total profit
   return profit;
}
```

Here, we implement the "CalculateTotalProfit" function with [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) and [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket) to iterate through open positions, selecting each via [PositionSelectByTicket](https://www.mql5.com/en/docs/trading/positionselectbyticket) and verifying its symbol and magic number with "GetPositionSymbol" and "GetPositionMagic" to get total profit. We accumulate unrealized profits using the [PositionGetDouble](https://www.mql5.com/en/docs/trading/positiongetdouble) function to retrieve each position’s profit, storing the sum in the "profit" variable, which enables accurate monitoring of trading outcomes. We can then continue to compose the dashboard creation function as follows.

```
//+------------------------------------------------------------------+
//| Update the dashboard with real-time trading metrics              |
//+------------------------------------------------------------------+
void UpdateDashboard() {
   //--- Exit if dashboard is not visible
   if (!dashboardVisible) return;

   //--- Create dashboard background rectangle
   CreateRectangle("Dashboard", panelStartX, panelStartY, DASHBOARD_WIDTH, DASHBOARD_HEIGHT, BACKGROUND_COLOR, BORDER_COLOR);

   //--- Create header rectangle
   CreateRectangle("Header", panelStartX, panelStartY, DASHBOARD_WIDTH, HEADER_HEIGHT, headerHovered ? HEADER_HOVER_COLOR : HEADER_NORMAL_COLOR, BORDER_COLOR);
   //--- Create header text label
   CreateTextLabel("HeaderText", panelStartX + 10, panelStartY + 8, EA_NAME, clrBlack, FONT_SIZE_HEADER, "Arial Bold");

   //--- Create close button
   CreateButton("CloseButton", CharToString(122), panelStartX + DASHBOARD_WIDTH - CLOSE_BUTTON_WIDTH, panelStartY + 1, CLOSE_BUTTON_WIDTH, CLOSE_BUTTON_HEIGHT, clrWhite, closeButtonHovered ? CLOSE_BUTTON_HOVER_BG : CLOSE_BUTTON_NORMAL_BG, FONT_SIZE_BUTTON, closeButtonHovered ? CLOSE_BUTTON_HOVER_BORDER : CLOSE_BUTTON_NORMAL_BORDER, false, "Wingdings");

   //--- Initialize dashboard content layout
   //--- Set initial Y-position below header
   int sectionY = panelStartY + HEADER_HEIGHT + 15;
   //--- Set left column X-position for labels
   int labelXLeft = panelStartX + 15;
   //--- Set right column X-position for values
   int valueXRight = panelStartX + 160;
   //--- Set row height for metrics
   int rowHeight = 15;

   //--- Pre-calculate values for conditional coloring
   //--- Calculate total unrealized profit
   double profit = CalculateTotalProfit();
   //--- Set profit color based on value
   color profitColor = (profit > 0) ? VALUE_POSITIVE_COLOR : (profit < 0) ? VALUE_LOSS_COLOR : TEXT_COLOR;

   //--- Get current equity
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   //--- Get current balance
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   //--- Set equity color based on comparison with balance
   color equityColor = (equity > balance) ? VALUE_POSITIVE_COLOR : (equity < balance) ? VALUE_NEGATIVE_COLOR : TEXT_COLOR;

   //--- Set balance color based on comparison with daily balance
   color balanceColor = (balance > dailyBalance) ? VALUE_POSITIVE_COLOR : (balance < dailyBalance) ? VALUE_NEGATIVE_COLOR : TEXT_COLOR;

   //--- Set open orders color based on active orders
   color ordersColor = (activeOrders > 0) ? VALUE_ACTIVE_COLOR : TEXT_COLOR;

   //--- Set drawdown active color based on trading state
   color drawdownColor = tradingEnabled ? VALUE_DRAWDOWN_INACTIVE : VALUE_DRAWDOWN_ACTIVE;

   //--- Set lot sizes color based on active orders
   color lotsColor = (activeOrders > 0) ? VALUE_ACTIVE_COLOR : TEXT_COLOR;

   //--- Calculate dynamic spread and color
   //--- Get current ask price
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   //--- Get current bid price
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   //--- Calculate spread in points
   double spread = (ask - bid) / Point();
   //--- Format spread with 1 decimal place for display
   string spreadDisplay = DoubleToString(spread, 1);
   //--- Initialize spread color
   color spreadColor;
   //--- Check if spread is low (favorable)
   if (spread <= 2.0) {
      //--- Set color to lime green for low spread
      spreadColor = VALUE_POSITIVE_COLOR;
   }
   //--- Check if spread is medium (moderate)
   else if (spread <= 5.0) {
      //--- Set color to gold for medium spread
      spreadColor = VALUE_ACTIVE_COLOR;
   }
   //--- Spread is high (costly)
   else {
      //--- Set color to orange for high spread
      spreadColor = VALUE_NEGATIVE_COLOR;
   }

   //--- Account Information Section
   //--- Create section title
   CreateTextLabel("SectionAccount", labelXLeft, sectionY, "Account Information", SECTION_TITLE_COLOR, FONT_SIZE_SECTION_TITLE, "Arial Bold");
   //--- Move to next row
   sectionY += rowHeight + 5;
   //--- Create account number label
   CreateTextLabel("AccountNumberLabel", labelXLeft, sectionY, "Account:", TEXT_COLOR, FONT_SIZE_METRIC);
   //--- Create account number value
   CreateTextLabel("AccountNumberValue", valueXRight, sectionY, DoubleToString(AccountInfoInteger(ACCOUNT_LOGIN), 0), TEXT_COLOR, FONT_SIZE_METRIC);
   //--- Move to next row
   sectionY += rowHeight;
   //--- Create account name label
   CreateTextLabel("AccountNameLabel", labelXLeft, sectionY, "Name:", TEXT_COLOR, FONT_SIZE_METRIC);
   //--- Create account name value
   CreateTextLabel("AccountNameValue", valueXRight, sectionY, AccountInfoString(ACCOUNT_NAME), TEXT_COLOR, FONT_SIZE_METRIC);
   //--- Move to next row
   sectionY += rowHeight;
   //--- Create leverage label
   CreateTextLabel("LeverageLabel", labelXLeft, sectionY, "Leverage:", TEXT_COLOR, FONT_SIZE_METRIC);
   //--- Create leverage value
   CreateTextLabel("LeverageValue", valueXRight, sectionY, "1:" + DoubleToString(AccountInfoInteger(ACCOUNT_LEVERAGE), 0), TEXT_COLOR, FONT_SIZE_METRIC);
   //--- Move to next row
   sectionY += rowHeight;

   //--- Market Information Section
   //--- Create section title
   CreateTextLabel("SectionMarket", labelXLeft, sectionY, "Market Information", SECTION_TITLE_COLOR, FONT_SIZE_SECTION_TITLE, "Arial Bold");
   //--- Move to next row
   sectionY += rowHeight + 5;
   //--- Create spread label
   CreateTextLabel("SpreadLabel", labelXLeft, sectionY, "Spread:", TEXT_COLOR, FONT_SIZE_METRIC);
   //--- Create spread value with dynamic color
   CreateTextLabel("SpreadValue", valueXRight, sectionY, spreadDisplay, spreadColor, FONT_SIZE_METRIC);
   //--- Move to next row
   sectionY += rowHeight;

   //--- Trading Statistics Section
   //--- Create section title
   CreateTextLabel("SectionTrading", labelXLeft, sectionY, "Trading Statistics", SECTION_TITLE_COLOR, FONT_SIZE_SECTION_TITLE, "Arial Bold");
   //--- Move to next row
   sectionY += rowHeight + 5;
   //--- Create balance label
   CreateTextLabel("BalanceLabel", labelXLeft, sectionY, "Balance:", TEXT_COLOR, FONT_SIZE_METRIC);
   //--- Create balance value with dynamic color
   CreateTextLabel("BalanceValue", valueXRight, sectionY, DoubleToString(balance, 2), balanceColor, FONT_SIZE_METRIC);
   //--- Move to next row
   sectionY += rowHeight;
   //--- Create equity label
   CreateTextLabel("EquityLabel", labelXLeft, sectionY, "Equity:", TEXT_COLOR, FONT_SIZE_METRIC);
   //--- Create equity value with dynamic color
   CreateTextLabel("EquityValue", valueXRight, sectionY, DoubleToString(equity, 2), equityColor, FONT_SIZE_METRIC);
   //--- Move to next row
   sectionY += rowHeight;
   //--- Create profit label
   CreateTextLabel("ProfitLabel", labelXLeft, sectionY, "Profit:", TEXT_COLOR, FONT_SIZE_METRIC);
   //--- Create profit value with dynamic color
   CreateTextLabel("ProfitValue", valueXRight, sectionY, DoubleToString(profit, 2), profitColor, FONT_SIZE_METRIC);
   //--- Move to next row
   sectionY += rowHeight;
   //--- Create open orders label
   CreateTextLabel("OrdersLabel", labelXLeft, sectionY, "Open Orders:", TEXT_COLOR, FONT_SIZE_METRIC);
   //--- Create open orders value with dynamic color
   CreateTextLabel("OrdersValue", valueXRight, sectionY, IntegerToString(activeOrders), ordersColor, FONT_SIZE_METRIC);
   //--- Move to next row
   sectionY += rowHeight;
   //--- Create drawdown active label
   CreateTextLabel("DrawdownLabel", labelXLeft, sectionY, "Drawdown Active:", TEXT_COLOR, FONT_SIZE_METRIC);
   //--- Create drawdown active value with dynamic color
   CreateTextLabel("DrawdownValue", valueXRight, sectionY, tradingEnabled ? "No" : "Yes", drawdownColor, FONT_SIZE_METRIC);
   //--- Move to next row
   sectionY += rowHeight;

   //--- Active Lot Sizes
   //--- Create active lots label
   CreateTextLabel("ActiveLotsLabel", labelXLeft, sectionY, "Active Lots:", TEXT_COLOR, FONT_SIZE_METRIC);
   //--- Create active lots value with dynamic color
   CreateTextLabel("ActiveLotsValue", valueXRight, sectionY, GetActiveLotSizes(), lotsColor, FONT_SIZE_METRIC);

   //--- Redraw the chart to update display
   ChartRedraw(0);
}
```

Here, we define the "UpdateDashboard" function to render a dynamic dashboard for real-time trading metrics, providing a comprehensive interface for monitoring performance. We start by checking the "dashboardVisible" variable to ensure updates occur only when the dashboard is active, preventing unnecessary processing. We use the "CreateRectangle" function to draw the main dashboard panel and header, setting dimensions with "DASHBOARD\_WIDTH" and "HEADER\_HEIGHT", and applying colors like "BACKGROUND\_COLOR" and "HEADER\_NORMAL\_COLOR" or "HEADER\_HOVER\_COLOR" based on the "headerHovered" state for visual feedback.

We employ the "CreateTextLabel" function to display critical metrics, including account balance, equity, profit, spread, open orders, drawdown status, and active lot sizes, organizing them into sections like Account Information, Market Information, and Trading Statistics. We calculate the spread using the [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) function to retrieve ask and bid prices, applying conditional color-coding with "VALUE\_POSITIVE\_COLOR" for low spreads (≤ 2.0 points), "VALUE\_ACTIVE\_COLOR" for medium spreads (2.1–5.0 points), and "VALUE\_NEGATIVE\_COLOR" for high spreads (> 5.0 points). You can change this to your preferred ranges. We then use [AccountInfoDouble](https://www.mql5.com/en/docs/account/accountinfodouble) to fetch balance and equity, and "CalculateTotalProfit" to determine unrealized profit, assigning colors like "profitColor" based on profit value for intuitive monitoring.

We integrate the "CreateButton" function to add an interactive close button, styled with "CLOSE\_BUTTON\_WIDTH", "FONT\_SIZE\_BUTTON", and dynamic colors ("CLOSE\_BUTTON\_NORMAL\_BG" or "CLOSE\_BUTTON\_HOVER\_BG") based on "closeButtonHovered", enhancing user interaction. We call the "GetActiveLotSizes" function to display up to three lot sizes in ascending order, using "lotsColor" for visual distinction, and we manage layout with variables like "sectionY", "labelXLeft", and "valueXRight" for precise positioning. We finalize updates with the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function to ensure smooth rendering. We can now call this function on the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler to create the first display.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   //--- Calculate pip value based on symbol digits (3 or 5 digits: multiply by 10)
   pipValue = (_Digits == 3 || _Digits == 5) ? 10.0 * Point() : Point();
   //--- Set the magic number for trade operations
   obj_Trade.SetExpertMagicNumber(MAGIC);
   //--- Initialize dashboard visibility
   dashboardVisible = true;
   //--- Set initial X-coordinate of the dashboard
   panelStartX = PANEL_X;
   //--- Set initial Y-coordinate of the dashboard
   panelStartY = PANEL_Y;
   //--- Initialize the dashboard display
   UpdateDashboard();
   //--- Enable mouse move events for dragging and hovering
   ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, true);
   //--- Return successful initialization
   return INIT_SUCCEEDED;
}
```

On the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, we calculate "pipValue" with the [Point](https://www.mql5.com/en/docs/check/point) function, set "obj\_Trade" with "SetExpertMagicNumber" for "MAGIC", and enable "dashboardVisible". We position the dashboard using "panelStartX" and "panelStartY" with "PANEL\_X" and "PANEL\_Y", call the "UpdateDashboard" function to display it, and use [ChartSetInteger](https://www.mql5.com/en/docs/chart_operations/chartsetinteger) for mouse interactions, returning [INIT\_SUCCEEDED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode). On the [OnDeInit](https://www.mql5.com/en/docs/event_handlers/ondeinit) event handler, we make sure to delete the created objects as follows.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   //--- Remove all graphical objects from the chart
   ObjectsDeleteAll(0);
   //--- Disable mouse move events
   ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, false);
}
```

We simply use the [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/ObjectDeleteAll) function to delete all the chart objects since we don't need them any longer. Upon compilation, here is the outcome.

![INITIAL PANEL](https://c.mql5.com/2/140/Screenshot_2025-05-05_160400.png)

To make the panel intractable, we call the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler to handle hover and dragging effects. Here is the logic we implement to make that work.

```
//+------------------------------------------------------------------+
//| Expert chart event handler                                       |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam) {
   //--- Exit if dashboard is not visible
   if (!dashboardVisible) return;

   //--- Handle mouse click events
   if (id == CHARTEVENT_CLICK) {
      //--- Get X-coordinate of the click
      int x = (int)lparam;
      //--- Get Y-coordinate of the click
      int y = (int)dparam;
      //--- Calculate close button X-position
      int buttonX = panelStartX + DASHBOARD_WIDTH - CLOSE_BUTTON_WIDTH - 5;
      //--- Calculate close button Y-position
      int buttonY = panelStartY + 1;
      //--- Check if click is within close button bounds
      if (x >= buttonX && x <= buttonX + CLOSE_BUTTON_WIDTH && y >= buttonY && y <= buttonY + CLOSE_BUTTON_HEIGHT) {
         //--- Hide the dashboard
         dashboardVisible = false;
         //--- Remove all graphical objects
         ObjectsDeleteAll(0);
         //--- Redraw the chart
         ChartRedraw(0);
      }
   }

   //--- Handle mouse move events
   if (id == CHARTEVENT_MOUSE_MOVE) {
      //--- Get X-coordinate of the mouse
      int mouseX = (int)lparam;
      //--- Get Y-coordinate of the mouse
      int mouseY = (int)dparam;
      //--- Get mouse state (e.g., button pressed)
      int mouseState = (int)sparam;

      //--- Update close button hover state
      //--- Calculate close button X-position
      int buttonX = panelStartX + DASHBOARD_WIDTH - CLOSE_BUTTON_WIDTH - 5;
      //--- Calculate close button Y-position
      int buttonY = panelStartY + 1;
      //--- Check if mouse is over the close button
      bool isCloseHovered = (mouseX >= buttonX && mouseX <= buttonX + CLOSE_BUTTON_WIDTH && mouseY >= buttonY && mouseY <= buttonY + CLOSE_BUTTON_HEIGHT);
      //--- Update close button hover state if changed
      if (isCloseHovered != closeButtonHovered) {
         //--- Set new hover state
         closeButtonHovered = isCloseHovered;
         //--- Update close button background color
         ObjectSetInteger(0, "CloseButton", OBJPROP_BGCOLOR, isCloseHovered ? CLOSE_BUTTON_HOVER_BG : CLOSE_BUTTON_NORMAL_BG);
         //--- Update close button border color
         ObjectSetInteger(0, "CloseButton", OBJPROP_BORDER_COLOR, isCloseHovered ? CLOSE_BUTTON_HOVER_BORDER : CLOSE_BUTTON_NORMAL_BORDER);
         //--- Redraw the chart
         ChartRedraw(0);
      }

      //--- Update header hover state
      //--- Set header X-position
      int headerX = panelStartX;
      //--- Set header Y-position
      int headerY = panelStartY;
      //--- Check if mouse is over the header
      bool isHeaderHovered = (mouseX >= headerX && mouseX <= headerX + DASHBOARD_WIDTH && mouseY >= headerY && mouseY <= headerY + HEADER_HEIGHT);
      //--- Update header hover state if changed
      if (isHeaderHovered != headerHovered) {
         //--- Set new hover state
         headerHovered = isHeaderHovered;
         //--- Update header background color
         ObjectSetInteger(0, "Header", OBJPROP_BGCOLOR, isHeaderHovered ? HEADER_HOVER_COLOR : HEADER_NORMAL_COLOR);
         //--- Redraw the chart
         ChartRedraw(0);
      }

      //--- Handle panel dragging
      //--- Store previous mouse state for click detection
      static int prevMouseState = 0;
      //--- Check for mouse button press (start dragging)
      if (prevMouseState == 0 && mouseState == 1) {
         //--- Check if header is hovered to initiate dragging
         if (isHeaderHovered) {
            //--- Enable dragging mode
            panelDragging = true;
            //--- Store initial mouse X-coordinate
            panelDragX = mouseX;
            //--- Store initial mouse Y-coordinate
            panelDragY = mouseY;
            //--- Get current dashboard X-position
            panelStartX = (int)ObjectGetInteger(0, "Dashboard", OBJPROP_XDISTANCE);
            //--- Get current dashboard Y-position
            panelStartY = (int)ObjectGetInteger(0, "Dashboard", OBJPROP_YDISTANCE);
            //--- Disable chart scrolling during dragging
            ChartSetInteger(0, CHART_MOUSE_SCROLL, false);
         }
      }

      //--- Update dashboard position during dragging
      if (panelDragging && mouseState == 1) {
         //--- Calculate X movement delta
         int dx = mouseX - panelDragX;
         //--- Calculate Y movement delta
         int dy = mouseY - panelDragY;
         //--- Update dashboard X-position
         panelStartX += dx;
         //--- Update dashboard Y-position
         panelStartY += dy;
         //--- Refresh the dashboard with new position
         UpdateDashboard();
         //--- Update stored mouse X-coordinate
         panelDragX = mouseX;
         //--- Update stored mouse Y-coordinate
         panelDragY = mouseY;
         //--- Redraw the chart
         ChartRedraw(0);
      }

      //--- Stop dragging when mouse button is released
      if (mouseState == 0) {
         //--- Check if dragging is active
         if (panelDragging) {
            //--- Disable dragging mode
            panelDragging = false;
            //--- Re-enable chart scrolling
            ChartSetInteger(0, CHART_MOUSE_SCROLL, true);
         }
      }

      //--- Update previous mouse state
      prevMouseState = mouseState;
   }
}
```

On the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler, we begin by checking the "dashboardVisible" variable to ensure event processing occurs only when the dashboard is active, optimizing performance by skipping unnecessary updates. For mouse click events, we calculate the close button’s position using "panelStartX", "DASHBOARD\_WIDTH", "CLOSE\_BUTTON\_WIDTH", and "panelStartY", and if a click falls within its bounds, we set "dashboardVisible" to false, call the [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/ObjectDeleteAll) function to remove all graphical objects, and use the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function to refresh the chart, effectively hiding the dashboard.

For mouse move events, we track the cursor’s position with "mouseX" and "mouseY", and monitor "mouseState" to detect actions like clicks or releases. We update the close button’s hover state by comparing "mouseX" and "mouseY" against its coordinates, setting "closeButtonHovered" and using the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) function to adjust [OBJPROP\_BGCOLOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) and "OBJPROP\_BORDER\_COLOR" to "CLOSE\_BUTTON\_HOVER\_BG" or "CLOSE\_BUTTON\_NORMAL\_BG" for visual feedback. Similarly, we manage the header’s hover state with "headerHovered", applying "HEADER\_HOVER\_COLOR" or "HEADER\_NORMAL\_COLOR" via "ObjectSetInteger" to enhance interactivity.

To enable dashboard dragging, we use a static "prevMouseState" to detect mouse button presses, initiating drag mode with "panelDragging" when the header is hovered, and store initial coordinates in "panelDragX" and "panelDragY". We retrieve the dashboard’s current position with the [ObjectGetInteger](https://www.mql5.com/en/docs/objects/objectgetinteger) function for "OBJPROP\_XDISTANCE" and [OBJPROP\_YDISTANCE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer), disable chart scrolling using the [ChartSetInteger](https://www.mql5.com/en/docs/chart_operations/chartsetinteger) function, and update "panelStartX" and "panelStartY" based on mouse movement deltas, calling the "UpdateDashboard" function to reposition the dashboard in real-time. When the mouse button is released, we reset "panelDragging" and re-enable scrolling with "ChartSetInteger", finalizing each update with [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to ensure a seamless user experience. Upon compilation, here is the outcome.

![DASHBOARD UI GIF](https://c.mql5.com/2/140/DRAG_EFFECT_GIF.gif)

From the visualization, we can see that we can drag, and hover over the buttons, update the metrics, and close the entire dashboard. We can now move on to the critical part which is opening and managing the positions, and the dashboard will ease the work by visualizing the progress dynamically. To achieve this, we will need some helper functions to calculate the lot sizes and more as below.

```
//+------------------------------------------------------------------+
//| Calculate the lot size for a new trade                           |
//+------------------------------------------------------------------+
double CalculateLotSize(ENUM_POSITION_TYPE tradeType) {
   //--- Initialize lot size
   double lotSize = 0;
   //--- Select lot size calculation mode
   switch (LOT_MODE) {
      //--- Fixed lot size mode
      case 0:
         //--- Use base lot size
         lotSize = BASE_LOT;
         break;
      //--- Multiplier-based lot size mode
      case 1:
         //--- Calculate lot size with multiplier based on active orders
         lotSize = NormalizeDouble(BASE_LOT * MathPow(LOT_MULTIPLIER, activeOrders), LOT_PRECISION);
         break;
      //--- Fixed lot size with multiplier on loss mode
      case 2: {
         //--- Initialize last close time
         datetime lastClose = 0;
         //--- Set default lot size
         lotSize = BASE_LOT;
         //--- Select trade history for the last 24 hours
         HistorySelect(TimeCurrent() - 24 * 60 * 60, TimeCurrent());
         //--- Iterate through trade history
         for (int i = HistoryDealsTotal() - 1; i >= 0; i--) {
            //--- Get deal ticket
            ulong ticket = HistoryDealGetTicket(i);
            //--- Select deal by ticket
            if (HistoryDealSelect(ticket) && HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT &&
                HistoryDealGetString(ticket, DEAL_SYMBOL) == _Symbol && HistoryDealGetInteger(ticket, DEAL_MAGIC) == MAGIC) {
               //--- Check if deal is more recent
               if (lastClose < HistoryDealGetInteger(ticket, DEAL_TIME)) {
                  //--- Update last close time
                  lastClose = (int)HistoryDealGetInteger(ticket, DEAL_TIME);
                  //--- Check if deal resulted in a loss
                  if (HistoryDealGetDouble(ticket, DEAL_PROFIT) < 0) {
                     //--- Increase lot size by multiplier
                     lotSize = NormalizeDouble(HistoryDealGetDouble(ticket, DEAL_VOLUME) * LOT_MULTIPLIER, LOT_PRECISION);
                  } else {
                     //--- Reset to base lot size
                     lotSize = BASE_LOT;
                  }
               }
            }
         }
         break;
      }
   }
   //--- Return calculated lot size
   return lotSize;
}

//+------------------------------------------------------------------+
//| Count the number of active orders                                |
//+------------------------------------------------------------------+
int CountActiveOrders() {
   //--- Initialize order counter
   int count = 0;
   //--- Iterate through open positions
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      //--- Get position ticket
      ulong ticket = PositionGetTicket(i);
      //--- Select position by ticket
      if (PositionSelectByTicket(ticket) && GetPositionSymbol() == _Symbol && GetPositionMagic() == MAGIC) {
         //--- Check if position is buy or sell
         if (GetPositionType() == POSITION_TYPE_BUY || GetPositionType() == POSITION_TYPE_SELL) {
            //--- Increment order counter
            count++;
         }
      }
   }
   //--- Return total active orders
   return count;
}

//+------------------------------------------------------------------+
//| Return a formatted string of active lot sizes in ascending order |
//+------------------------------------------------------------------+
string GetActiveLotSizes() {
   //--- Check if no active orders
   if (activeOrders == 0) {
      //--- Return waiting message
      return "[Waiting]";
   }
   //--- Initialize array for lot sizes
   double lotSizes[];
   //--- Resize array to match active orders
   ArrayResize(lotSizes, activeOrders);
   //--- Initialize counter
   int count = 0;
   //--- Iterate through open positions
   for (int i = PositionsTotal() - 1; i >= 0 && count < activeOrders; i--) {
      //--- Get position ticket
      ulong ticket = PositionGetTicket(i);
      //--- Select position by ticket
      if (PositionSelectByTicket(ticket) && GetPositionSymbol() == _Symbol && GetPositionMagic() == MAGIC) {
         //--- Store position volume (lot size)
         lotSizes[count] = PositionGetDouble(POSITION_VOLUME);
         //--- Increment counter
         count++;
      }
   }
   //--- Sort lot sizes in ascending order
   ArraySort(lotSizes);
   //--- Initialize result string
   string result = "";
   //--- Determine maximum number of lots to display (up to 3)
   int maxDisplay = (activeOrders > 3) ? 3 : activeOrders;
   //--- Format lot sizes
   for (int i = 0; i < maxDisplay; i++) {
      //--- Add comma and space for subsequent entries
      if (i > 0) result += ", ";
      //--- Convert lot size to string with specified precision
      result += DoubleToString(lotSizes[i], LOT_PRECISION);
   }
   //--- Append ellipsis if more than 3 orders
   if (activeOrders > 3) result += ", ...";
   //--- Return formatted lot sizes in brackets
   return "[" + result + "]";
}
```

Here, we define the "CalculateLotSize", "CountActiveOrders", and "GetActiveLotSizes" functions to help in trade sizing, position tracking, and dashboard updates, ensuring precise execution and comprehensive real-time monitoring.

We implement the "CalculateLotSize" function to determine trade volumes based on the "LOT\_MODE" input, supporting three modes: fixed lot sizing by returning "BASE\_LOT" directly, Martingale scaling by applying the [MathPow](https://www.mql5.com/en/docs/math/mathpow) function with "LOT\_MULTIPLIER" and "activeOrders" to increase lots exponentially, or loss-based adjustments where we use the [HistorySelect](https://www.mql5.com/en/docs/trading/historyselect) function to review the last 24 hours of trades.

In the loss-based mode, we iterate with [HistoryDealsTotal](https://www.mql5.com/en/docs/trading/historydealstotal), retrieve deal details via [HistoryDealGetTicket](https://www.mql5.com/en/docs/trading/historydealgetticket) and [HistoryDealGetDouble](https://www.mql5.com/en/docs/trading/historydealgetdouble), and adjust "lotSize" with [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble) if the most recent trade was a loss, resetting to "BASE\_LOT" otherwise, ensuring precise lot calculations tailored to trading outcomes.

We utilize the "CountActiveOrders" function to maintain an accurate count of open positions, critical for Martingale scaling and dashboard accuracy. We iterate through all positions using the [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) function, select each with [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket), and verify relevance with the "GetPositionSymbol" and "GetPositionMagic" functions, incrementing the "count" variable for buy or sell positions identified by "GetPositionType", thus updating "activeOrders" reliably.

Additionally, we craft the "GetActiveLotSizes" function to format and display lot sizes on the dashboard, enhancing user visibility into active trades. We check "activeOrders" to return "\[Waiting\]" if none exist, otherwise initialize an array with [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) to store lot sizes. We iterate positions with PositionsTotal, select them via "PositionGetTicket", and use [PositionGetDouble](https://www.mql5.com/en/docs/trading/positiongetdouble) to collect volumes in "lotSizes", sorting them in ascending order with the [ArraySort](https://www.mql5.com/en/docs/array/arraysort) function. We format up to three lots using [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring) with "LOT\_PRECISION", adding commas and an ellipsis for more than three orders, and return the result in brackets, providing a clear, professional display of trade volumes for real-time monitoring. Still, we need to define functions to help in order placement as below.

```
//+------------------------------------------------------------------+
//| Place a new trade order                                          |
//+------------------------------------------------------------------+
int PlaceOrder(ENUM_ORDER_TYPE orderType, double lot, double price, double slPrice, int gridLevel) {
   //--- Calculate stop-loss price
   double sl = CalculateSL(slPrice, STOP_LOSS_PIPS);
   //--- Calculate take-profit price
   double tp = CalculateTP(price, TAKE_PROFIT_PIPS, orderType);
   //--- Initialize ticket number
   int ticket = 0;
   //--- Set maximum retry attempts
   int retries = 100;
   //--- Create trade comment with grid level
   string comment = "GridMart Scalper-" + IntegerToString(gridLevel);

   //--- Attempt to place order with retries
   for (int i = 0; i < retries; i++) {
      //--- Open position with specified parameters
      ticket = obj_Trade.PositionOpen(_Symbol, orderType, lot, price, sl, tp, comment);
      //--- Get last error code
      int error = GetLastError();
      //--- Exit loop if order is successful
      if (error == 0) break;
      //--- Check for retryable errors (server busy, trade context busy, etc.)
      if (!(error == 4 || error == 137 || error == 146 || error == 136)) break;
      //--- Wait before retrying
      Sleep(5000);
   }
   //--- Return order ticket (or error code if negative)
   return ticket;
}

//+------------------------------------------------------------------+
//| Calculate the stop-loss price for a trade                        |
//+------------------------------------------------------------------+
double CalculateSL(double price, int points) {
   //--- Check if stop-loss is disabled
   if (points == 0) return 0;
   //--- Calculate stop-loss price (subtract points from price)
   return price - points * pipValue;
}

//+------------------------------------------------------------------+
//| Calculate the take-profit price for a trade                      |
//+------------------------------------------------------------------+
double CalculateTP(double price, int points, ENUM_ORDER_TYPE orderType) {
   //--- Check if take-profit is disabled
   if (points == 0) return 0;
   //--- Calculate take-profit for buy order (add points)
   if (orderType == ORDER_TYPE_BUY) return price + points * pipValue;
   //--- Calculate take-profit for sell order (subtract points)
   return price - points * pipValue;
}

//+------------------------------------------------------------------+
//| Retrieve the latest buy order price                              |
//+------------------------------------------------------------------+
double GetLatestBuyPrice() {
   //--- Initialize price
   double price = 0;
   //--- Initialize latest ticket
   int latestTicket = 0;
   //--- Iterate through open positions
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      //--- Get position ticket
      ulong ticket = PositionGetTicket(i);
      //--- Select position by ticket
      if (PositionSelectByTicket(ticket) && GetPositionSymbol() == _Symbol && GetPositionMagic() == MAGIC && GetPositionType() == POSITION_TYPE_BUY) {
         //--- Check if ticket is more recent
         if ((int)ticket > latestTicket) {
            //--- Update price
            price = GetPositionOpenPrice();
            //--- Update latest ticket
            latestTicket = (int)ticket;
         }
      }
   }
   //--- Return latest buy price
   return price;
}

//+------------------------------------------------------------------+
//| Retrieve the latest sell order price                             |
//+------------------------------------------------------------------+
double GetLatestSellPrice() {
   //--- Initialize price
   double price = 0;
   //--- Initialize latest ticket
   int latestTicket = 0;
   //--- Iterate through open positions
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      //--- Get position ticket
      ulong ticket = PositionGetTicket(i);
      //--- Select position by ticket
      if (PositionSelectByTicket(ticket) && GetPositionSymbol() == _Symbol && GetPositionMagic() == MAGIC && GetPositionType() == POSITION_TYPE_SELL) {
         //--- Check if ticket is more recent
         if ((int)ticket > latestTicket) {
            //--- Update price
            price = GetPositionOpenPrice();
            //--- Update latest ticket
            latestTicket = (int)ticket;
         }
      }
   }
   //--- Return latest sell price
   return price;
}

//+------------------------------------------------------------------+
//| Check if trading is allowed                                      |
//+------------------------------------------------------------------+
int IsTradingAllowed() {
   //--- Always allow trading
   return 1;
}
```

Here, we implement the "PlaceOrder" function to initiate trades, using the "PositionOpen" function from "obj\_Trade" to open positions with parameters like "lot", "price", and a comment formatted with [IntegerToString](https://www.mql5.com/en/docs/convert/integertostring) for "gridLevel", incorporating retry logic with [GetLastError](https://www.mql5.com/en/docs/check/getlasterror) and [Sleep](https://www.mql5.com/en/docs/common/sleep) for robust execution. We utilize the "CalculateSL" function to compute stop-loss prices by subtracting "STOP\_LOSS\_PIPS" multiplied by "pipValue" from "slPrice", and the "CalculateTP" function to set take-profit levels, adding or subtracting "TAKE\_PROFIT\_PIPS" based on "orderType" for buy or sell trades, respectively.

We craft the "GetLatestBuyPrice" function to identify the most recent buy position’s price, iterating with [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal), selecting positions via [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket), and verifying with "GetPositionSymbol", "GetPositionMagic", and "GetPositionType", updating "price" using "GetPositionOpenPrice" for the highest "ticket".

Similarly, we implement the "GetLatestSellPrice" function to retrieve the latest sell position’s price, following the same logic to ensure accurate grid placement. We define the "IsTradingAllowed" function to return a constant value of 1, enabling continuous trading without restrictions, which supports the strategy’s high-frequency approach. We can now use these functions to define the actual trading logic in the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   //--- Get current ask price
   double ask = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK), _Digits);
   //--- Get current bid price
   double bid = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID), _Digits);

   //--- Update dashboard if visible
   if (dashboardVisible) UpdateDashboard();

   //--- Check if trading is allowed
   if (IsTradingAllowed()) {
      //--- Get current bar time
      datetime currentBarTime = iTime(_Symbol, _Period, 0);
      //--- Exit if the bar hasn’t changed
      if (lastBarTime == currentBarTime) return;
      //--- Update last bar time
      lastBarTime = currentBarTime;
      //--- Count active orders
      activeOrders = CountActiveOrders();
      //--- Reset SL/TP update flag if no active orders
      if (activeOrders == 0) updateSLTP = false;

      //--- Reset position flags
      hasBuyPosition = false;
      hasSellPosition = false;
      //--- Iterate through open positions
      for (int i = PositionsTotal() - 1; i >= 0; i--) {
         //--- Get position ticket
         ulong ticket = PositionGetTicket(i);
         //--- Select position by ticket
         if (PositionSelectByTicket(ticket) && GetPositionSymbol() == _Symbol && GetPositionMagic() == MAGIC) {
            //--- Check for buy position
            if (GetPositionType() == POSITION_TYPE_BUY) {
               //--- Set buy position flag
               hasBuyPosition = true;
               //--- Clear sell position flag
               hasSellPosition = false;
               //--- Exit loop after finding a position
               break;
            }
            //--- Check for sell position
            else if (GetPositionType() == POSITION_TYPE_SELL) {
               //--- Set sell position flag
               hasSellPosition = true;
               //--- Clear buy position flag
               hasBuyPosition = false;
               //--- Exit loop after finding a position
               break;
            }
         }
      }

      //--- Check conditions to open new trades
      if (activeOrders > 0 && activeOrders <= MAX_GRID_LEVELS) {
         //--- Get latest buy price
         latestBuyPrice = GetLatestBuyPrice();
         //--- Get latest sell price
         latestSellPrice = GetLatestSellPrice();
         //--- Check if a new buy trade is needed
         if (hasBuyPosition && latestBuyPrice - ask >= GRID_DISTANCE * pipValue) openNewTrade = true;
         //--- Check if a new sell trade is needed
         if (hasSellPosition && bid - latestSellPrice >= GRID_DISTANCE * pipValue) openNewTrade = true;
      }
      //--- Allow new trades if no active orders
      if (activeOrders < 1) {
         //--- Clear position flags
         hasBuyPosition = false;
         hasSellPosition = false;
         //--- Signal to open a new trade
         openNewTrade = true;
      }

      //--- Execute new trade if signaled
      if (openNewTrade) {
         //--- Update latest buy price
         latestBuyPrice = GetLatestBuyPrice();
         //--- Update latest sell price
         latestSellPrice = GetLatestSellPrice();
         //--- Handle sell position
         if (hasSellPosition) {
            //--- Calculate lot size for sell order
            calculatedLot = CalculateLotSize(POSITION_TYPE_SELL);
            //--- Check if lot size is valid and trading is enabled
            if (calculatedLot > 0 && tradingEnabled) {
               //--- Place sell order
               int ticket = PlaceOrder(ORDER_TYPE_SELL, calculatedLot, bid, ask, activeOrders);
               //--- Check for order placement errors
               if (ticket < 0) {
                  //--- Log error message
                  Print("Sell Order Error: ", GetLastError());
                  return;
               }
               //--- Update latest sell price
               latestSellPrice = GetLatestSellPrice();
               //--- Clear new trade signal
               openNewTrade = false;
               //--- Signal to modify positions
               modifyPositions = true;
            }
         }
         //--- Handle buy position
         else if (hasBuyPosition) {
            //--- Calculate lot size for buy order
            calculatedLot = CalculateLotSize(POSITION_TYPE_BUY);
            //--- Check if lot size is valid and trading is enabled
            if (calculatedLot > 0 && tradingEnabled) {
               //--- Place buy order
               int ticket = PlaceOrder(ORDER_TYPE_BUY, calculatedLot, ask, bid, activeOrders);
               //--- Check for order placement errors
               if (ticket < 0) {
                  //--- Log error message
                  Print("Buy Order Error: ", GetLastError());
                  return;
               }
               //--- Update latest buy price
               latestBuyPrice = GetLatestBuyPrice();
               //--- Clear new trade signal
               openNewTrade = false;
               //--- Signal to modify positions
               modifyPositions = true;
            }
         }

         //---

      }
   }
}
```

We define the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler to drive the core trading logic of the program, orchestrating real-time trade execution and position management. We begin by retrieving current market prices with the [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) function to capture "ask" and "bid" values, normalized via [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble), and update the dashboard using the "UpdateDashboard" function if "dashboardVisible" is true. We check trading permissions with the "IsTradingAllowed" function, ensuring trades proceed only when conditions allow, and use the [iTime](https://www.mql5.com/en/docs/series/itime) function to fetch the current bar’s timestamp, storing it in "currentBarTime" to prevent redundant processing by comparing with "lastBarTime".

We manage position tracking by calling the "CountActiveOrders" function to update "activeOrders", resetting "updateSLTP" if no orders exist, and iterating positions with [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) and [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket) to set "hasBuyPosition" or "hasSellPosition" based on "GetPositionType".

We evaluate grid conditions using "GetLatestBuyPrice" and "GetLatestSellPrice", triggering "openNewTrade" when price movements exceed "GRID\_DISTANCE" times "pipValue" or if "activeOrders" is below "MAX\_GRID\_LEVELS". When "openNewTrade" is true, we calculate lot sizes with the "CalculateLotSize" function, execute trades via the "PlaceOrder" function for buy or sell orders, log errors with "Print" and [GetLastError](https://www.mql5.com/en/docs/check/getlasterror) if placement fails, and update "latestBuyPrice", "latestSellPrice", and "modifyPositions" to manage ongoing trades, ensuring precise and efficient scalping operations. To open new positions, we use the following logic for signal generation.

```
//--- Check conditions to open a new trade without existing positions
MqlDateTime timeStruct;
//--- Get current time
TimeCurrent(timeStruct);
//--- Verify trading hours, cycle limit, and new trade conditions
if (timeStruct.hour >= START_HOUR && timeStruct.hour < END_HOUR && cycleCount < MAX_CYCLES && CONTINUE_TRADING && openNewTrade && activeOrders < 1) {
   //--- Get previous bar close price
   double closePrev = iClose(_Symbol, PERIOD_CURRENT, 2);
   //--- Get current bar close price
   double closeCurrent = iClose(_Symbol, PERIOD_CURRENT, 1);
   //--- Check if no existing positions
   if (!hasSellPosition && !hasBuyPosition) {
      //--- Check for bearish signal (previous close > current close)
      if (closePrev > closeCurrent) {
         //--- Calculate lot size for sell order
         calculatedLot = CalculateLotSize(POSITION_TYPE_SELL);
         //--- Check if lot size is valid and trading is enabled
         if (calculatedLot > 0 && tradingEnabled) {
            //--- Place sell order
            int ticket = PlaceOrder(ORDER_TYPE_SELL, calculatedLot, bid, bid, activeOrders);
            //--- Check for order placement errors
            if (ticket < 0) {
               //--- Log error message
               Print("Sell Order Error: ", GetLastError());
               return;
            }
            //--- Increment cycle count
            cycleCount++;
            //--- Update latest buy price
            latestBuyPrice = GetLatestBuyPrice();
            //--- Signal to modify positions
            modifyPositions = true;
         }
      }
      //--- Check for bullish signal (previous close <= current close)
      else {
         //--- Calculate lot size for buy order
         calculatedLot = CalculateLotSize(POSITION_TYPE_BUY);
         //--- Check if lot size is valid and trading is enabled
         if (calculatedLot > 0 && tradingEnabled) {
            //--- Place buy order
            int ticket = PlaceOrder(ORDER_TYPE_BUY, calculatedLot, ask, ask, activeOrders);
            //--- Check for order placement errors
            if (ticket < 0) {
               //--- Log error message
               Print("Buy Order Error: ", GetLastError());
               return;
            }
            //--- Increment cycle count
            cycleCount++;
            //--- Update latest sell price
            latestSellPrice = GetLatestSellPrice();
            //--- Signal to modify positions
            modifyPositions = true;
         }
      }
   }
}
```

To initiate new trades, we start by using the [TimeCurrent](https://www.mql5.com/en/docs/dateandtime/timecurrent) function to retrieve the current time into the "timeStruct" variable, checking if "timeStruct.hour" falls within "START\_HOUR" and "END\_HOUR", "cycleCount" is below "MAX\_CYCLES", and "CONTINUE\_TRADING" and "openNewTrade" are true, with "activeOrders" less than 1. We fetch the previous and current bar close prices with the [iClose](https://www.mql5.com/en/docs/series/iclose) function, storing them in "closePrev" and "closeCurrent", and confirm no positions exist with "hasSellPosition" and "hasBuyPosition".

For a bearish signal (when "closePrev" exceeds "closeCurrent"), we calculate the lot size using the "CalculateLotSize" function for a sell order, verify "calculatedLot" and "tradingEnabled", and execute the trade with the "PlaceOrder" function, logging errors via "Print" and [GetLastError](https://www.mql5.com/en/docs/check/getlasterror) if needed. For a bullish signal (when "closePrev" is less than or equal to "closeCurrent"), we perform the same process for a buy order, updating "cycleCount" and "latestBuyPrice" or "latestSellPrice" with "GetLatestBuyPrice" or "GetLatestSellPrice", and setting "modifyPositions" to true, enabling precise trade initiation and position management.

You can replace this with any of your trading strategies. We just used a simple signal generation strategy since the main aim is to manage the positions by applying the strategy. Once we open positions, we need to check and modify them when upon price advancing as below.

```
//--- Update active orders count
activeOrders = CountActiveOrders();
//--- Reset weighted price and total volume
weightedPrice = 0;
totalVolume = 0;
//--- Calculate weighted price and total volume
for (int i = PositionsTotal() - 1; i >= 0; i--) {
   //--- Get position ticket
   ulong ticket = PositionGetTicket(i);
   //--- Select position by ticket
   if (PositionSelectByTicket(ticket) && GetPositionSymbol() == _Symbol && GetPositionMagic() == MAGIC) {
      //--- Accumulate weighted price (price * volume)
      weightedPrice += GetPositionOpenPrice() * PositionGetDouble(POSITION_VOLUME);
      //--- Accumulate total volume
      totalVolume += PositionGetDouble(POSITION_VOLUME);
   }
}

//--- Normalize weighted price if there are active orders
if (activeOrders > 0) weightedPrice = NormalizeDouble(weightedPrice / totalVolume, _Digits);
//--- Check if positions need SL/TP modification
if (modifyPositions) {
   //--- Iterate through open positions
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      //--- Get position ticket
      ulong ticket = PositionGetTicket(i);
      //--- Select position by ticket
      if (PositionSelectByTicket(ticket) && GetPositionSymbol() == _Symbol && GetPositionMagic() == MAGIC) {
         //--- Handle buy positions
         if (GetPositionType() == POSITION_TYPE_BUY) {
            //--- Set take-profit for buy position
            targetTP = weightedPrice + TAKE_PROFIT_PIPS * pipValue;
            //--- Set stop-loss for buy position
            targetSL = weightedPrice - STOP_LOSS_PIPS * pipValue;
            //--- Signal SL/TP update
            updateSLTP = true;
         }
         //--- Handle sell positions
         else if (GetPositionType() == POSITION_TYPE_SELL) {
            //--- Set take-profit for sell position
            targetTP = weightedPrice - TAKE_PROFIT_PIPS * pipValue;
            //--- Set stop-loss for sell position
            targetSL = weightedPrice + STOP_LOSS_PIPS * pipValue;
            //--- Signal SL/TP update
            updateSLTP = true;
         }
      }
   }
}

//--- Apply SL/TP modifications if needed
if (modifyPositions && updateSLTP) {
   //--- Iterate through open positions
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      //--- Get position ticket
      ulong ticket = PositionGetTicket(i);
      //--- Select position by ticket
      if (PositionSelectByTicket(ticket) && GetPositionSymbol() == _Symbol && GetPositionMagic() == MAGIC) {
         //--- Modify position with new SL/TP
         if (obj_Trade.PositionModify(ticket, targetSL, targetTP)) {
            //--- Clear modification signal on success
            modifyPositions = false;
         }
      }
   }
}
```

Here, we update "activeOrders" with "CountActiveOrders", reset "weightedPrice" and "totalVolume", and iterate positions using [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) and [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket), accumulating "weightedPrice" and "totalVolume" via "GetPositionOpenPrice" and [PositionGetDouble](https://www.mql5.com/en/docs/trading/positiongetdouble) functions. We normalize "weightedPrice" with [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble) if "activeOrders" exists, set "targetTP" and "targetSL" using "GetPositionType" and "pipValue" when "modifyPositions" is true, and apply updates with "PositionModify" from "obj\_Trade" if "updateSLTP", resetting "modifyPositions" on success. Upon compilation, we have the following outcome.

![ORDERS OPENED](https://c.mql5.com/2/140/Screenshot_2025-05-05_165934.png)

From the image, we can see that we have opened orders that are already in management mode. We just now need to apply the risk management logic for monitoring daily drawdown and limiting it. For this, we will a function to do all the tracking and stop the program when necessary as follows.

```
//+------------------------------------------------------------------+
//| Monitor daily drawdown and control trading state                 |
//+------------------------------------------------------------------+
void MonitorDailyDrawdown() {
   //--- Initialize daily profit accumulator
   double totalDayProfit = 0.0;
   //--- Get current time
   datetime end = TimeCurrent();
   //--- Get current date as string
   string sdate = TimeToString(TimeCurrent(), TIME_DATE);
   //--- Convert date to datetime (start of day)
   datetime start = StringToTime(sdate);
   //--- Set end of day (24 hours later)
   datetime to = start + (1 * 24 * 60 * 60);

   //--- Check if daily reset is needed
   if (dailyResetTime < to) {
      //--- Update reset time
      dailyResetTime = to;
      //--- Store current balance as daily starting balance
      dailyBalance = GetAccountBalance();
   }
   //--- Select trade history for the day
   HistorySelect(start, end);
   //--- Get total number of deals
   int totalDeals = HistoryDealsTotal();

   //--- Iterate through trade history
   for (int i = 0; i < totalDeals; i++) {
      //--- Get deal ticket
      ulong ticket = HistoryDealGetTicket(i);
      //--- Check if deal is a position close
      if (HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT) {
         //--- Calculate deal profit (including commission and swap)
         double latestDayProfit = (HistoryDealGetDouble(ticket, DEAL_PROFIT) +
                                  HistoryDealGetDouble(ticket, DEAL_COMMISSION) +
                                  HistoryDealGetDouble(ticket, DEAL_SWAP));
         //--- Accumulate daily profit
         totalDayProfit += latestDayProfit;
      }
   }

   //--- Calculate starting balance for the day
   double startingBalance = GetAccountBalance() - totalDayProfit;
   //--- Calculate daily profit/drawdown percentage
   double dailyProfitOrDrawdown = NormalizeDouble((totalDayProfit * 100 / startingBalance), 2);

   //--- Check if drawdown limit is exceeded
   if (dailyProfitOrDrawdown <= DRAWDOWN_LIMIT) {
      //--- Close all positions if configured
      if (CLOSE_ON_DRAWDOWN) CloseAllPositions();
      //--- Disable trading
      tradingEnabled = false;
   } else {
      //--- Enable trading
      tradingEnabled = true;
   }
}

//+------------------------------------------------------------------+
//| Close all open positions managed by the EA                       |
//+------------------------------------------------------------------+
void CloseAllPositions() {
   //--- Iterate through open positions
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      //--- Get position ticket
      ulong ticket = PositionGetTicket(i);
      //--- Select position by ticket
      if (ticket > 0 && PositionSelectByTicket(ticket)) {
         //--- Check if position belongs to this EA
         if (GetPositionSymbol() == _Symbol && GetPositionMagic() == MAGIC) {
            //--- Close the position
            obj_Trade.PositionClose(ticket);
         }
      }
   }
}
```

To enforce risk management, we implement the "MonitorDailyDrawdown" function to track daily performance, using [TimeCurrent](https://www.mql5.com/en/docs/dateandtime/timecurrent) and [TimeToString](https://www.mql5.com/en/docs/convert/timetostring) to set the day’s time range, and [HistorySelect](https://www.mql5.com/en/docs/trading/historyselect) to access trade history. We calculate total profit with [HistoryDealGetTicket](https://www.mql5.com/en/docs/trading/historydealgetticket) and [HistoryDealGetDouble](https://www.mql5.com/en/docs/trading/historydealgetdouble) functions, normalize drawdown percentage via [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble), and adjust "tradingEnabled" based on "DRAWDOWN\_LIMIT", calling the "CloseAllPositions" function if "CLOSE\_ON\_DRAWDOWN" is true.

We define "CloseAllPositions" to iterate positions with "PositionsTotal" and [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket), closing relevant trades using "PositionClose" from "obj\_Trade" after verifying with "GetPositionSymbol" and "GetPositionMagic", ensuring robust drawdown control. We can then call this function on every tick to make the risk management when eligible.

```
//--- Monitor daily drawdown if enabled
if (ENABLE_DAILY_DRAWDOWN) MonitorDailyDrawdown();
```

We check the "ENABLE\_DAILY\_DRAWDOWN" condition to determine if drawdown control is active and if true, we call the "MonitorDailyDrawdown" function to evaluate daily profit and loss, adjust "tradingEnabled", and potentially close positions, safeguarding the account against excessive losses. Upon compilation, we have the following outcome.

![FINAL POSITIONS MANAGEMENT GIF](https://c.mql5.com/2/140/GRIDMART_GIF.gif)

From the visualization, we can see that we can open the positions, manage them dynamically, and close them once the targets are reached, hence achieving our objective of creating the grid-mart strategy setup. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/140/Screenshot_2025-05-05_213653.png)

Backtest report:

![REPORT](https://c.mql5.com/2/140/Screenshot_2025-05-05_213618.png)

### Conclusion

In conclusion, we have developed an MQL5 program that automates the Grid-Mart Scalping Strategy, executing grid-based Martingale trades with a dynamic [dashboard](https://en.wikipedia.org/wiki/Dashboard_(computing) "https://en.wikipedia.org/wiki/Dashboard_(computing)") for real-time monitoring of key metrics like spread, profit, and lot sizes. With precise trade execution, robust risk management through drawdown controls, and an interactive interface, you can further enhance this program by tailoring its parameters or integrating additional strategies to suit your trading preferences.

Disclaimer: This article is for educational purposes only. Trading carries significant financial risks, and market volatility may result in losses. Thorough backtesting and careful risk management are crucial before deploying this program in live markets.

By mastering these techniques, you can enhance this program further to make it more robust or use it as a backbone for developing other trading strategies, empowering your algorithmic trading journey.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18038.zip "Download all attachments in the single ZIP archive")

[GridMart\_Scalper\_MT5\_EA.mq5](https://www.mql5.com/en/articles/download/18038/gridmart_scalper_mt5_ea.mq5 "Download GridMart_Scalper_MT5_EA.mq5")(108.01 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/486403)**
(5)


![Ahmet Parlakbilek](https://c.mql5.com/avatar/2017/12/5A294D45-ADD7.JPG)

**[Ahmet Parlakbilek](https://www.mql5.com/en/users/ahmetp)**
\|
14 May 2025 at 21:27

Nice article, but there is a significant bug in CalculateSL.


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
14 May 2025 at 22:45

**Ahmet Parlakbilek [#](https://www.mql5.com/en/forum/486403#comment_56703291):**

Nice article, but there is a significant bug in CalculateSL.

Sure. Thanks. What's up with it?

![Ahmet Parlakbilek](https://c.mql5.com/avatar/2017/12/5A294D45-ADD7.JPG)

**[Ahmet Parlakbilek](https://www.mql5.com/en/users/ahmetp)**
\|
15 May 2025 at 06:36

**Allan Munene Mutiiria [#](https://www.mql5.com/en/forum/486403#comment_56703611):**

Sure. Thanks. What's up with it?

You forgot handling sell side. Attached is the corrected version.

![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
15 May 2025 at 15:44

**Ahmet Parlakbilek [#](https://www.mql5.com/en/forum/486403#comment_56705887):**

You forgot handling sell side. Attached is the corrected version.

Ooh. Yeah. Sure. It will be of great help to others. Thanks.

![Pramendra](https://c.mql5.com/avatar/avatar_na2.png)

**[Pramendra](https://www.mql5.com/en/users/pssola)**
\|
18 May 2025 at 11:28

Sir, Please fix.


![Data Science and ML (Part 39): News + Artificial Intelligence, Would You Bet on it?](https://c.mql5.com/2/141/17986-data-science-and-ml-part-39-logo.png)[Data Science and ML (Part 39): News + Artificial Intelligence, Would You Bet on it?](https://www.mql5.com/en/articles/17986)

News drives the financial markets, especially major releases like Non-Farm Payrolls (NFPs). We've all witnessed how a single headline can trigger sharp price movements. In this article, we dive into the powerful intersection of news data and Artificial Intelligence.

![Custom Debugging and Profiling Tools for MQL5 Development (Part I): Advanced Logging](https://c.mql5.com/2/141/17933-custom-debugging-and-profiling-logo.png)[Custom Debugging and Profiling Tools for MQL5 Development (Part I): Advanced Logging](https://www.mql5.com/en/articles/17933)

Learn how to implement a powerful custom logging framework for MQL5 that goes beyond simple Print() statements by supporting severity levels, multiple output handlers, and automated file rotation—all configurable on‐the‐fly. Integrate the singleton CLogger with ConsoleLogHandler and FileLogHandler to capture contextual, timestamped logs in both the Experts tab and persistent files. Streamline debugging and performance tracing in your Expert Advisors with clear, customizable log formats and centralized control.

![Artificial Ecosystem-based Optimization (AEO) algorithm](https://c.mql5.com/2/97/Artificial_Ecosystem_based_Optimization__LOGO.png)[Artificial Ecosystem-based Optimization (AEO) algorithm](https://www.mql5.com/en/articles/16058)

The article considers a metaheuristic Artificial Ecosystem-based Optimization (AEO) algorithm, which simulates interactions between ecosystem components by creating an initial population of solutions and applying adaptive update strategies, and describes in detail the stages of AEO operation, including the consumption and decomposition phases, as well as different agent behavior strategies. The article introduces the features and advantages of this algorithm.

![Neural Networks in Trading: Mask-Attention-Free Approach to Price Movement Forecasting](https://c.mql5.com/2/95/Neural_Networks_in_Trading_A_Maskless_Approach_to_Price_Movement_Forecasting__LOGO_2.png)[Neural Networks in Trading: Mask-Attention-Free Approach to Price Movement Forecasting](https://www.mql5.com/en/articles/15973)

In this article, we will discuss the Mask-Attention-Free Transformer (MAFT) method and its application in the field of trading. Unlike traditional Transformers that require data masking when processing sequences, MAFT optimizes the attention process by eliminating the need for masking, significantly improving computational efficiency.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=sqwzwxwvtismshycwfjfniewhuhexqrh&ssn=1769178988459962760&ssn_dr=0&ssn_sr=0&fv_date=1769178988&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18038&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2017)%3A%20Mastering%20the%20Grid-Mart%20Scalping%20Strategy%20with%20a%20Dynamic%20Dashboard%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917898889454625&fz_uniq=5068437468375611756&sv=2552)

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