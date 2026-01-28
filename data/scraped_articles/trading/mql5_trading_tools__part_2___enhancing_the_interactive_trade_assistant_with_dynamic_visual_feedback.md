---
title: MQL5 Trading Tools (Part 2): Enhancing the Interactive Trade Assistant with Dynamic Visual Feedback
url: https://www.mql5.com/en/articles/17972
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T17:55:40.992667
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=yztwrvyaikkdagymuaycnesqgitkphyn&ssn=1769180138781407680&ssn_dr=0&ssn_sr=0&fv_date=1769180138&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F17972&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Trading%20Tools%20(Part%202)%3A%20Enhancing%20the%20Interactive%20Trade%20Assistant%20with%20Dynamic%20Visual%20Feedback%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918013893686153&fz_uniq=5068806887102676457&sv=2552)

MetaTrader 5 / Trading


### Introduction

In our previous article, Part 1, we built a Trade Assistant Tool in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) for [MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/") to simplify placing pending orders. Now, we take it further by enhancing its interactivity with dynamic visual feedback. We introduce features like a draggable control panel, hover effects for intuitive navigation, and real-time order validation to ensure our trade setups are precise and market-relevant. We cover these advancements through the following subtopics:

1. [Conceptual Enhancements for Improved Interactivity](https://www.mql5.com/en/articles/17972#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/17972#para2)
3. [Backtesting](https://www.mql5.com/en/articles/17972#para3)
4. [Conclusion](https://www.mql5.com/en/articles/17972#para4)

These sections guide us toward a more responsive, intuitive, and user-friendly trading tool.

### Conceptual Enhancements for Improved Interactivity

We strive to elevate our trade assistant tool by making it more intuitive and adaptable. We start with a draggable control panel that we can freely position on the trading chart. This flexibility will allow us to customize the interface to fit our workflow, whether we’re managing multiple charts or focusing on a single trade setup. Additionally, we will integrate hover effects to highlight buttons and chart elements as our cursor moves over them, providing instant visual feedback that streamlines navigation and minimizes errors.

Real-time order validation will be another key enhancement, ensuring our entry, stop-loss, and take-profit levels are logically aligned with current market prices before execution. This feature will boost our confidence by preventing invalid trade setups, and maintaining simplicity while enhancing precision. Together, these improvements will create a responsive, user-centric tool that supports our trading decisions and sets the stage for future advancements like risk management features. In a nutshell, below is a visualization of what we aim to achieve.

![OBJECTIVES VISUALIZATION](https://c.mql5.com/2/138/Screenshot_2025-04-29_211021.png)

### Implementation in MQL5

To achieve our objectives in [MQL5](https://www.metatrader5.com/ "https://www.metatrader5.com/"), we will first need to define some extra panel objects, drag, and hover confirmation variables that we will use to keep track of the user interactions with either the panel or the price tool as below.

```
// Control panel object names
#define PANEL_BG        "PANEL_BG"        //--- Define constant for panel background object name
#define PANEL_HEADER    "PANEL_HEADER"    //--- Define constant for panel header object name
#define LOT_EDIT        "LOT_EDIT"        //--- Define constant for lot size edit field object name
#define PRICE_LABEL     "PRICE_LABEL"     //--- Define constant for price label object name
#define SL_LABEL        "SL_LABEL"        //--- Define constant for stop-loss label object name
#define TP_LABEL        "TP_LABEL"        //--- Define constant for take-profit label object name
#define BUY_STOP_BTN    "BUY_STOP_BTN"    //--- Define constant for buy stop button object name
#define SELL_STOP_BTN   "SELL_STOP_BTN"   //--- Define constant for sell stop button object name
#define BUY_LIMIT_BTN   "BUY_LIMIT_BTN"   //--- Define constant for buy limit button object name
#define SELL_LIMIT_BTN  "SELL_LIMIT_BTN"  //--- Define constant for sell limit button object name
#define PLACE_ORDER_BTN "PLACE_ORDER_BTN" //--- Define constant for place order button object name
#define CANCEL_BTN      "CANCEL_BTN"      //--- Define constant for cancel button object name
#define CLOSE_BTN       "CLOSE_BTN"       //--- Define constant for close button object name

// Variables for dragging panel
bool panel_dragging   = false;            //--- Flag to track if panel is being dragged
int  panel_drag_x     = 0,
     panel_drag_y     = 0;                //--- Mouse coordinates when drag starts
int  panel_start_x    = 0,
     panel_start_y    = 0;                //--- Panel coordinates when drag starts

// Button and rectangle hover states
bool buy_stop_hovered    = false;         //--- Buy Stop button hover state
bool sell_stop_hovered   = false;         //--- Sell Stop button hover state
bool buy_limit_hovered   = false;         //--- Buy Limit button hover state
bool sell_limit_hovered  = false;         //--- Sell Limit button hover state
bool place_order_hovered = false;         //--- Place Order button hover state
bool cancel_hovered      = false;         //--- Cancel button hover state
bool close_hovered       = false;         //--- Close button hover state
bool header_hovered      = false;         //--- Header hover state
bool rec1_hovered        = false;         //--- REC1 (TP) hover state
bool rec3_hovered        = false;         //--- REC3 (Entry) hover state
bool rec5_hovered        = false;         //--- REC5 (SL) hover state
```

We begin implementing the enhanced interactivity features for our tool by defining key variables that enable panel dragging and hover effects in the [MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/") interface. We use the [#define](https://www.mql5.com/en/docs/basis/preprosessor/constant) directive to create a constant "PANEL\_HEADER" for the panel’s header object, which serves as the draggable region of the control panel. To support dragging, we declare "panel\_dragging" as a boolean flag to track when the panel is being moved, and integers "panel\_drag\_x", "panel\_drag\_y" to store the mouse coordinates at the start of a drag, and "panel\_start\_x", "panel\_start\_y" to record the panel’s initial position, allowing us to calculate its new position during movement.

We also introduce boolean variables to manage hover states for buttons and chart rectangles, including "buy\_stop\_hovered", "sell\_stop\_hovered", "buy\_limit\_hovered", "sell\_limit\_hovered", "place\_order\_hovered", "cancel\_hovered", "close\_hovered", and "header\_hovered" for the respective buttons and panel header, as well as "rec1\_hovered", "rec3\_hovered", and "rec5\_hovered" for the take-profit, entry, and stop-loss rectangles. These [variables](https://www.mql5.com/en/docs/basis/variables) will enable us to detect when our cursor hovers over these elements, triggering visual feedback like color changes to enhance navigation and interaction within the tool’s interface. Next, we need to get the values of the price tool and validate them for trading.

```
//+------------------------------------------------------------------+
//| Check if order setup is valid                                    |
//+------------------------------------------------------------------+
bool isOrderValid() {
   if(!tool_visible) return true;                                     //--- No validation needed if tool is not visible
   double current_price = SymbolInfoDouble(Symbol(), SYMBOL_BID);     //--- Get current bid price
   double entry_price   = Get_Price_d(PR_HL);                         //--- Get entry price
   double sl_price      = Get_Price_d(SL_HL);                         //--- Get stop-loss price
   double tp_price      = Get_Price_d(TP_HL);                         //--- Get take-profit price

   if(selected_order_type == "BUY_STOP") {
      //--- Buy Stop: Entry must be above current price, TP above entry, SL below entry
      if(entry_price <= current_price || tp_price <= entry_price || sl_price >= entry_price) {
         return false;
      }
   }
   else if(selected_order_type == "SELL_STOP") {
      //--- Sell Stop: Entry must be below current price, TP below entry, SL above entry
      if(entry_price >= current_price || tp_price >= entry_price || sl_price <= entry_price) {
         return false;
      }
   }
   else if(selected_order_type == "BUY_LIMIT") {
      //--- Buy Limit: Entry must be below current price, TP above entry, SL below entry
      if(entry_price >= current_price || tp_price <= entry_price || sl_price >= entry_price) {
         return false;
      }
   }
   else if(selected_order_type == "SELL_LIMIT") {
      //--- Sell Limit: Entry must be above current price, TP below entry, SL above entry
      if(entry_price <= current_price || tp_price >= entry_price || sl_price <= entry_price) {
         return false;
      }
   }
   return true;                                                       //--- Order setup is valid
}
```

Here, we implement the "isOrderValid" function to enhance our tool by validating the order setup in real time, ensuring our trades align with market conditions. We start by checking if "tool\_visible" is false, returning true to skip validation when the tool isn’t active. We retrieve the current market price using the [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) function with [SYMBOL\_BID](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double), and get the entry ("entry\_price"), stop-loss ("sl\_price"), and take-profit ("tp\_price") prices using the "Get\_Price\_d" function for "PR\_HL", "SL\_HL", and "TP\_HL".

For "BUY\_STOP", we verify "entry\_price" is above "current\_price", "tp\_price" above "entry\_price", and "sl\_price" below "entry\_price"; for "SELL\_STOP", "entry\_price" below "current\_price", "tp\_price" below "entry\_price", and "sl\_price" above "entry\_price"; for "BUY\_LIMIT", "entry\_price" below "current\_price", "tp\_price" above "entry\_price", and "sl\_price" below "entry\_price"; and for "SELL\_LIMIT", "entry\_price" above "current\_price", "tp\_price" below "entry\_price", and "sl\_price" above "entry\_price", returning false if any condition fails, or true if the setup is valid. Then we can update the rectangle colors as per the order validity.

```
//+------------------------------------------------------------------+
//| Update rectangle colors based on order validity and hover        |
//+------------------------------------------------------------------+
void updateRectangleColors() {
   if(!tool_visible) return;                                                             //--- Skip if tool is not visible
   bool is_valid = isOrderValid();                                                       //--- Check order validity

   if(!is_valid) {
      //--- Gray out REC1 and REC5 if order is invalid, with hover effect
      ObjectSetInteger(0, REC1, OBJPROP_BGCOLOR, rec1_hovered ? C'100,100,100' : clrGray);
      ObjectSetInteger(0, REC5, OBJPROP_BGCOLOR, rec5_hovered ? C'100,100,100' : clrGray);
   }
   else {
      //--- Restore original colors based on order type and hover state
      if(selected_order_type == "BUY_STOP" || selected_order_type == "BUY_LIMIT") {
         ObjectSetInteger(0, REC1, OBJPROP_BGCOLOR, rec1_hovered ? C'0,100,0'   : clrGreen); //--- TP rectangle (dark green on hover)
         ObjectSetInteger(0, REC5, OBJPROP_BGCOLOR, rec5_hovered ? C'139,0,0'   : clrRed);   //--- SL rectangle (dark red on hover)
      }
      else {
         ObjectSetInteger(0, REC1, OBJPROP_BGCOLOR, rec1_hovered ? C'0,100,0'   : clrGreen); //--- TP rectangle (dark green on hover)
         ObjectSetInteger(0, REC5, OBJPROP_BGCOLOR, rec5_hovered ? C'139,0,0'   : clrRed);   //--- SL rectangle (dark red on hover)
      }
   }

   ObjectSetInteger(0, REC3, OBJPROP_BGCOLOR, rec3_hovered ? C'105,105,105' : clrLightGray); //--- Entry rectangle (darker gray on hover)
   ChartRedraw(0);                                                                          //--- Redraw chart
}
```

We implement the "updateRectangleColors" function to enhance our tool’s visual feedback by updating chart rectangle colors based on order validity and hover states. We skip if "tool\_visible" is false, check validity with the "isOrderValid" function, and use the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/ObjectSetInteger) function to set "REC1" (TP) and "REC5" (SL) to gray ("clrGray" or "C'100,100,100'" if "rec1\_hovered"/"rec5\_hovered") if invalid, or green/red ("clrGreen"/"clrRed" or "C'0,100,0'"/"C'139,0,0'" on hover) is valid for "BUY\_STOP"/"BUY\_LIMIT"/sell orders, and "REC3" (entry) to light gray ("clrLightGray" or "C'105,105,105'" if "rec3\_hovered"), calling [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to refresh the chart.

After that, we need to get the hover states of the buttons as below.

```
//+------------------------------------------------------------------+
//| Update button and header hover state                             |
//+------------------------------------------------------------------+
void updateButtonHoverState(int mouse_x, int mouse_y) {
   // Define button names and their properties
   string buttons[] = {BUY_STOP_BTN, SELL_STOP_BTN, BUY_LIMIT_BTN, SELL_LIMIT_BTN, PLACE_ORDER_BTN, CANCEL_BTN, CLOSE_BTN};
   bool hover_states[] = {buy_stop_hovered, sell_stop_hovered, buy_limit_hovered, sell_limit_hovered, place_order_hovered, cancel_hovered, close_hovered};
   color normal_colors[] = {clrForestGreen, clrFireBrick, clrForestGreen, clrFireBrick, clrDodgerBlue, clrSlateGray, clrCrimson};
   color hover_color = clrDodgerBlue;                      //--- Bluish color for hover
   color hover_border = clrBlue;                           //--- Bluish border for hover

   for(int i = 0; i < ArraySize(buttons); i++) {
      int x = (int)ObjectGetInteger(0, buttons[i], OBJPROP_XDISTANCE);
      int y = (int)ObjectGetInteger(0, buttons[i], OBJPROP_YDISTANCE);
      int width = (int)ObjectGetInteger(0, buttons[i], OBJPROP_XSIZE);
      int height = (int)ObjectGetInteger(0, buttons[i], OBJPROP_YSIZE);

      bool is_hovered = (mouse_x >= x && mouse_x <= x + width && mouse_y >= y && mouse_y <= y + height);

      if(is_hovered && !hover_states[i]) {
         // Mouse entered button
         ObjectSetInteger(0, buttons[i], OBJPROP_BGCOLOR, hover_color);
         ObjectSetInteger(0, buttons[i], OBJPROP_BORDER_COLOR, hover_border);
         hover_states[i] = true;
      }
      else if(!is_hovered && hover_states[i]) {
         // Mouse left button
         ObjectSetInteger(0, buttons[i], OBJPROP_BGCOLOR, normal_colors[i]);
         ObjectSetInteger(0, buttons[i], OBJPROP_BORDER_COLOR, clrBlack);
         hover_states[i] = false;
      }
   }

   // Update header hover state
   int header_x = (int)ObjectGetInteger(0, PANEL_HEADER, OBJPROP_XDISTANCE);
   int header_y = (int)ObjectGetInteger(0, PANEL_HEADER, OBJPROP_YDISTANCE);
   int header_width = (int)ObjectGetInteger(0, PANEL_HEADER, OBJPROP_XSIZE);
   int header_height = (int)ObjectGetInteger(0, PANEL_HEADER, OBJPROP_YSIZE);

   bool is_header_hovered = (mouse_x >= header_x && mouse_x <= header_x + header_width && mouse_y >= header_y && mouse_y <= header_y + header_height);

   if(is_header_hovered && !header_hovered) {
      ObjectSetInteger(0, PANEL_HEADER, OBJPROP_BGCOLOR, C'030,030,030'); //--- Darken header
      header_hovered = true;
   }
   else if(!is_header_hovered && header_hovered) {
      ObjectSetInteger(0, PANEL_HEADER, OBJPROP_BGCOLOR, C'050,050,050'); //--- Restore header color
      header_hovered = false;
   }

   // Update tool rectangle hover states
   if(tool_visible) {
      int x1 = (int)ObjectGetInteger(0, REC1, OBJPROP_XDISTANCE);
      int y1 = (int)ObjectGetInteger(0, REC1, OBJPROP_YDISTANCE);
      int width1 = (int)ObjectGetInteger(0, REC1, OBJPROP_XSIZE);
      int height1 = (int)ObjectGetInteger(0, REC1, OBJPROP_YSIZE);

      int x3 = (int)ObjectGetInteger(0, REC3, OBJPROP_XDISTANCE);
      int y3 = (int)ObjectGetInteger(0, REC3, OBJPROP_YDISTANCE);
      int width3 = (int)ObjectGetInteger(0, REC3, OBJPROP_XSIZE);
      int height3 = (int)ObjectGetInteger(0, REC3, OBJPROP_YSIZE);

      int x5 = (int)ObjectGetInteger(0, REC5, OBJPROP_XDISTANCE);
      int y5 = (int)ObjectGetInteger(0, REC5, OBJPROP_YDISTANCE);
      int width5 = (int)ObjectGetInteger(0, REC5, OBJPROP_XSIZE);
      int height5 = (int)ObjectGetInteger(0, REC5, OBJPROP_YSIZE);

      bool is_rec1_hovered = (mouse_x >= x1 && mouse_x <= x1 + width1 && mouse_y >= y1 && mouse_y <= y1 + height1);
      bool is_rec3_hovered = (mouse_x >= x3 && mouse_x <= x3 + width3 && mouse_y >= y3 && mouse_y <= y3 + height3);
      bool is_rec5_hovered = (mouse_x >= x5 && mouse_x <= x5 + width5 && mouse_y >= y5 && mouse_y <= y5 + height5);

      if(is_rec1_hovered != rec1_hovered || is_rec3_hovered != rec3_hovered || is_rec5_hovered != rec5_hovered) {
         rec1_hovered = is_rec1_hovered;
         rec3_hovered = is_rec3_hovered;
         rec5_hovered = is_rec5_hovered;
         updateRectangleColors();                            //--- Update colors based on hover state
      }
   }

   // Update hover state variables
   buy_stop_hovered = hover_states[0];
   sell_stop_hovered = hover_states[1];
   buy_limit_hovered = hover_states[2];
   sell_limit_hovered = hover_states[3];
   place_order_hovered = hover_states[4];
   cancel_hovered = hover_states[5];
   close_hovered = hover_states[6];

   ChartRedraw(0);                                           //--- Redraw chart
}
```

To enhance our tool’s interactivity by managing hover effects for buttons and chart elements, we implement the "updateButtonHoverState" function. We define arrays "buttons" for button names ("BUY\_STOP\_BTN" to "CLOSE\_BTN"), "hover\_states" for their hover flags ("buy\_stop\_hovered" to "close\_hovered"), and "normal\_colors" for default colors, with "hover\_color" ( [clrDodgerBlue](https://www.mql5.com/en/docs/constants/objectconstants/webcolors)) and "hover\_border" ("clrBlue") for hover states.

For each button, we use the [ObjectGetInteger](https://www.mql5.com/en/docs/objects/objectgetinteger) function to get position and size, check if "mouse\_x" and "mouse\_y" are within bounds, and update "OBJPROP\_BGCOLOR" and "OBJPROP\_BORDER\_COLOR" to "hover\_color" or "normal\_colors" using [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger), toggling "hover\_states".

For the "PANEL\_HEADER", we similarly check the hover state, darkening it to "C'030,030,030'" or restoring "C'050,050,050'" with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger). When "tool\_visible", we check "REC1", "REC3", and "REC5" bounds, updating "rec1\_hovered", "rec3\_hovered", and "rec5\_hovered", and call "updateRectangleColors" if changed. We sync "buy\_stop\_hovered" to "close\_hovered" with "hover\_states" and call [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to refresh the chart. Then we can call these functions within the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler so that we can get real-time updates.

```
//+------------------------------------------------------------------+
//| Expert onchart event function                                    |
//+------------------------------------------------------------------+
void OnChartEvent(
   const int id,          //--- Event ID
   const long& lparam,    //--- Long parameter (e.g., x-coordinate for mouse)
   const double& dparam,  //--- Double parameter (e.g., y-coordinate for mouse)
   const string& sparam   //--- String parameter (e.g., object name)
) {
   if(id == CHARTEVENT_OBJECT_CLICK) {                           //--- Handle object click events
      // Handle order type buttons
      if(sparam == BUY_STOP_BTN) {                               //--- Check if Buy Stop button clicked
         selected_order_type = "BUY_STOP";                       //--- Set order type to Buy Stop
         showTool();                                             //--- Show trading tool
         update_Text(PLACE_ORDER_BTN, "Place Buy Stop");         //--- Update place order button text
         updateRectangleColors();                                //--- Update rectangle colors
      }
      else if(sparam == SELL_STOP_BTN) {                         //--- Check if Sell Stop button clicked
         selected_order_type = "SELL_STOP";                      //--- Set order type to Sell Stop
         showTool();                                             //--- Show trading tool
         update_Text(PLACE_ORDER_BTN, "Place Sell Stop");        //--- Update place order button text
         updateRectangleColors();                                //--- Update rectangle colors
      }
      else if(sparam == BUY_LIMIT_BTN) {                         //--- Check if Buy Limit button clicked
         selected_order_type = "BUY_LIMIT";                      //--- Set order type to Buy Limit
         showTool();                                             //--- Show trading tool
         update_Text(PLACE_ORDER_BTN, "Place Buy Limit");        //--- Update place order button text
         updateRectangleColors();                                //--- Update rectangle colors
      }
      else if(sparam == SELL_LIMIT_BTN) {                        //--- Check if Sell Limit button clicked
         selected_order_type = "SELL_LIMIT";                     //--- Set order type to Sell Limit
         showTool();                                             //--- Show trading tool
         update_Text(PLACE_ORDER_BTN, "Place Sell Limit");       //--- Update place order button text
         updateRectangleColors();                                //--- Update rectangle colors
      }
      else if(sparam == PLACE_ORDER_BTN) {                       //--- Check if Place Order button clicked
         if(isOrderValid()) {
            placeOrder();                                        //--- Execute order placement
            deleteObjects();                                     //--- Delete tool objects
            showPanel();                                         //--- Show control panel
         }
         else {
            Print("Cannot place order: Invalid price setup for ", selected_order_type);
         }
      }
      else if(sparam == CANCEL_BTN) {                            //--- Check if Cancel button clicked
         deleteObjects();                                        //--- Delete tool objects
         showPanel();                                            //--- Show control panel
      }
      else if(sparam == CLOSE_BTN) {                             //--- Check if Close button clicked
         deleteObjects();                                        //--- Delete tool objects
         deletePanel();                                          //--- Delete control panel
         ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, false);      //--- Disable mouse move events
      }
      ObjectSetInteger(0, sparam, OBJPROP_STATE, false);         //--- Reset button state click
      ChartRedraw(0);                                            //--- Redraw chart
   }

   if(id == CHARTEVENT_MOUSE_MOVE) {                             //--- Handle mouse move events
      int MouseD_X = (int)lparam;                                //--- Get mouse x-coordinate
      int MouseD_Y = (int)dparam;                                //--- Get mouse y-coordinate
      int MouseState = (int)sparam;                              //--- Get mouse state

      // Update button and rectangle hover states
      updateButtonHoverState(MouseD_X, MouseD_Y);

      // Handle panel dragging
      int header_xd = (int)ObjectGetInteger(0, PANEL_HEADER, OBJPROP_XDISTANCE);
      int header_yd = (int)ObjectGetInteger(0, PANEL_HEADER, OBJPROP_YDISTANCE);
      int header_xs = (int)ObjectGetInteger(0, PANEL_HEADER, OBJPROP_XSIZE);
      int header_ys = (int)ObjectGetInteger(0, PANEL_HEADER, OBJPROP_YSIZE);

      if(prevMouseState == 0 && MouseState == 1) {               //--- Mouse button down
         if(MouseD_X >= header_xd && MouseD_X <= header_xd + header_xs &&
            MouseD_Y >= header_yd && MouseD_Y <= header_yd + header_ys) {
            panel_dragging = true;                               //--- Start dragging
            panel_drag_x = MouseD_X;                             //--- Store mouse x-coordinate
            panel_drag_y = MouseD_Y;                             //--- Store mouse y-coordinate
            panel_start_x = header_xd;                           //--- Store panel x-coordinate
            panel_start_y = header_yd;                           //--- Store panel y-coordinate
            ChartSetInteger(0, CHART_MOUSE_SCROLL, false);       //--- Disable chart scrolling
         }
      }

      if(panel_dragging && MouseState == 1) {                    //--- Dragging panel
         int dx = MouseD_X - panel_drag_x;                       //--- Calculate x displacement
         int dy = MouseD_Y - panel_drag_y;                       //--- Calculate y displacement
         panel_x = panel_start_x + dx;                           //--- Update panel x-position
         panel_y = panel_start_y + dy;                           //--- Update panel y-position

         // Update all panel objects' positions
         ObjectSetInteger(0, PANEL_BG, OBJPROP_XDISTANCE, panel_x);
         ObjectSetInteger(0, PANEL_BG, OBJPROP_YDISTANCE, panel_y);
         ObjectSetInteger(0, PANEL_HEADER, OBJPROP_XDISTANCE, panel_x);
         ObjectSetInteger(0, PANEL_HEADER, OBJPROP_YDISTANCE, panel_y+2);
         ObjectSetInteger(0, CLOSE_BTN, OBJPROP_XDISTANCE, panel_x + 209);
         ObjectSetInteger(0, CLOSE_BTN, OBJPROP_YDISTANCE, panel_y + 1);
         ObjectSetInteger(0, LOT_EDIT, OBJPROP_XDISTANCE, panel_x + 70);
         ObjectSetInteger(0, LOT_EDIT, OBJPROP_YDISTANCE, panel_y + 40);
         ObjectSetInteger(0, PRICE_LABEL, OBJPROP_XDISTANCE, panel_x + 10);
         ObjectSetInteger(0, PRICE_LABEL, OBJPROP_YDISTANCE, panel_y + 70);
         ObjectSetInteger(0, SL_LABEL, OBJPROP_XDISTANCE, panel_x + 10);
         ObjectSetInteger(0, SL_LABEL, OBJPROP_YDISTANCE, panel_y + 95);
         ObjectSetInteger(0, TP_LABEL, OBJPROP_XDISTANCE, panel_x + 130);
         ObjectSetInteger(0, TP_LABEL, OBJPROP_YDISTANCE, panel_y + 95);
         ObjectSetInteger(0, BUY_STOP_BTN, OBJPROP_XDISTANCE, panel_x + 10);
         ObjectSetInteger(0, BUY_STOP_BTN, OBJPROP_YDISTANCE, panel_y + 140);
         ObjectSetInteger(0, SELL_STOP_BTN, OBJPROP_XDISTANCE, panel_x + 130);
         ObjectSetInteger(0, SELL_STOP_BTN, OBJPROP_YDISTANCE, panel_y + 140);
         ObjectSetInteger(0, BUY_LIMIT_BTN, OBJPROP_XDISTANCE, panel_x + 10);
         ObjectSetInteger(0, BUY_LIMIT_BTN, OBJPROP_YDISTANCE, panel_y + 180);
         ObjectSetInteger(0, SELL_LIMIT_BTN, OBJPROP_XDISTANCE, panel_x + 130);
         ObjectSetInteger(0, SELL_LIMIT_BTN, OBJPROP_YDISTANCE, panel_y + 180);
         ObjectSetInteger(0, PLACE_ORDER_BTN, OBJPROP_XDISTANCE, panel_x + 10);
         ObjectSetInteger(0, PLACE_ORDER_BTN, OBJPROP_YDISTANCE, panel_y + 240);
         ObjectSetInteger(0, CANCEL_BTN, OBJPROP_XDISTANCE, panel_x + 130);
         ObjectSetInteger(0, CANCEL_BTN, OBJPROP_YDISTANCE, panel_y + 240);

         ChartRedraw(0);                                                   //--- Redraw chart
      }

      if(MouseState == 0) {                                                //--- Mouse button released
         if(panel_dragging) {
            panel_dragging = false;                                        //--- Stop dragging
            ChartSetInteger(0, CHART_MOUSE_SCROLL, true);                  //--- Re-enable chart scrolling
         }
      }

      if(tool_visible) {                                                   //--- Handle tool movement
         int XD_R1 = (int)ObjectGetInteger(0, REC1, OBJPROP_XDISTANCE);    //--- Get REC1 x-distance
         int YD_R1 = (int)ObjectGetInteger(0, REC1, OBJPROP_YDISTANCE);    //--- Get REC1 y-distance
         int XS_R1 = (int)ObjectGetInteger(0, REC1, OBJPROP_XSIZE);        //--- Get REC1 x-size
         int YS_R1 = (int)ObjectGetInteger(0, REC1, OBJPROP_YSIZE);        //--- Get REC1 y-size

         int XD_R2 = (int)ObjectGetInteger(0, REC2, OBJPROP_XDISTANCE);    //--- Get REC2 x-distance
         int YD_R2 = (int)ObjectGetInteger(0, REC2, OBJPROP_YDISTANCE);    //--- Get REC2 y-distance
         int XS_R2 = (int)ObjectGetInteger(0, REC2, OBJPROP_XSIZE);        //--- Get REC2 x-size
         int YS_R2 = (int)ObjectGetInteger(0, REC2, OBJPROP_YSIZE);        //--- Get REC2 y-size

         int XD_R3 = (int)ObjectGetInteger(0, REC3, OBJPROP_XDISTANCE);    //--- Get REC3 x-distance
         int YD_R3 = (int)ObjectGetInteger(0, REC3, OBJPROP_YDISTANCE);    //--- Get REC3 y-distance
         int XS_R3 = (int)ObjectGetInteger(0, REC3, OBJPROP_XSIZE);        //--- Get REC3 x-size
         int YS_R3 = (int)ObjectGetInteger(0, REC3, OBJPROP_YSIZE);        //--- Get REC3 y-size

         int XD_R4 = (int)ObjectGetInteger(0, REC4, OBJPROP_XDISTANCE);    //--- Get REC4 x-distance
         int YD_R4 = (int)ObjectGetInteger(0, REC4, OBJPROP_YDISTANCE);    //--- Get REC4 y-distance
         int XS_R4 = (int)ObjectGetInteger(0, REC4, OBJPROP_XSIZE);        //--- Get REC4 x-size
         int YS_R4 = (int)ObjectGetInteger(0, REC4, OBJPROP_YSIZE);        //--- Get REC4 y-size

         int XD_R5 = (int)ObjectGetInteger(0, REC5, OBJPROP_XDISTANCE);    //--- Get REC5 x-distance
         int YD_R5 = (int)ObjectGetInteger(0, REC5, OBJPROP_YDISTANCE);    //--- Get REC5 y-distance
         int XS_R5 = (int)ObjectGetInteger(0, REC5, OBJPROP_XSIZE);        //--- Get REC5 x-size
         int YS_R5 = (int)ObjectGetInteger(0, REC5, OBJPROP_YSIZE);        //--- Get REC5 y-size

         if(prevMouseState == 0 && MouseState == 1 && !panel_dragging) {   //--- Check for mouse button down, avoid dragging conflict
            mlbDownX1 = MouseD_X;                                          //--- Store mouse x-coordinate for REC1
            mlbDownY1 = MouseD_Y;                                          //--- Store mouse y-coordinate for REC1
            mlbDownXD_R1 = XD_R1;                                          //--- Store REC1 x-distance
            mlbDownYD_R1 = YD_R1;                                          //--- Store REC1 y-distance

            mlbDownX2 = MouseD_X;                                          //--- Store mouse x-coordinate for REC2
            mlbDownY2 = MouseD_Y;                                          //--- Store mouse y-coordinate for REC2
            mlbDownXD_R2 = XD_R2;                                          //--- Store REC2 x-distance
            mlbDownYD_R2 = YD_R2;                                          //--- Store REC2 y-distance

            mlbDownX3 = MouseD_X;                                          //--- Store mouse x-coordinate for REC3
            mlbDownY3 = MouseD_Y;                                          //--- Store mouse y-coordinate for REC3
            mlbDownXD_R3 = XD_R3;                                          //--- Store REC3 x-distance
            mlbDownYD_R3 = YD_R3;                                          //--- Store REC3 y-distance

            mlbDownX4 = MouseD_X;                                          //--- Store mouse x-coordinate for REC4
            mlbDownY4 = MouseD_Y;                                          //--- Store mouse y-coordinate for REC4
            mlbDownXD_R4 = XD_R4;                                          //--- Store REC4 x-distance
            mlbDownYD_R4 = YD_R4;                                          //--- Store REC4 y-distance

            mlbDownX5 = MouseD_X;                                          //--- Store mouse x-coordinate for REC5
            mlbDownY5 = MouseD_Y;                                          //--- Store mouse y-coordinate for REC5
            mlbDownXD_R5 = XD_R5;                                          //--- Store REC5 x-distance
            mlbDownYD_R5 = YD_R5;                                          //--- Store REC5 y-distance

            if(MouseD_X >= XD_R1 && MouseD_X <= XD_R1 + XS_R1 &&           //--- Check if mouse is within REC1 bounds
               MouseD_Y >= YD_R1 && MouseD_Y <= YD_R1 + YS_R1) {
               movingState_R1 = true;                                      //--- Enable REC1 movement
               ChartSetInteger(0, CHART_MOUSE_SCROLL, false);              //--- Disable chart scrolling
            }
            if(MouseD_X >= XD_R3 && MouseD_X <= XD_R3 + XS_R3 &&           //--- Check if mouse is within REC3 bounds
               MouseD_Y >= YD_R3 && MouseD_Y <= YD_R3 + YS_R3) {
               movingState_R3 = true;                                      //--- Enable REC3 movement
               ChartSetInteger(0, CHART_MOUSE_SCROLL, false);              //--- Disable chart scrolling
            }
            if(MouseD_X >= XD_R5 && MouseD_X <= XD_R5 + XS_R5 &&           //--- Check if mouse is within REC5 bounds
               MouseD_Y >= YD_R5 && MouseD_Y <= YD_R5 + YS_R5) {
               movingState_R5 = true;                                      //--- Enable REC5 movement
               ChartSetInteger(0, CHART_MOUSE_SCROLL, false);              //--- Disable chart scrolling
            }
         }
         if(movingState_R1) {                                                                           //--- Handle REC1 (TP) movement
            bool canMove = false;                                                                       //--- Flag to check if movement is valid
            if(selected_order_type == "BUY_STOP" || selected_order_type == "BUY_LIMIT") {               //--- Check for buy orders
               if(YD_R1 + YS_R1 < YD_R3) {                                                              //--- Ensure TP is above entry for buy orders
                  canMove = true;                                                                       //--- Allow movement
                  ObjectSetInteger(0, REC1, OBJPROP_YDISTANCE, mlbDownYD_R1 + MouseD_Y - mlbDownY1);    //--- Update REC1 y-position
                  ObjectSetInteger(0, REC2, OBJPROP_YDISTANCE, YD_R1 + YS_R1);                          //--- Update REC2 y-position
                  ObjectSetInteger(0, REC2, OBJPROP_YSIZE, YD_R3 - (YD_R1 + YS_R1));                    //--- Update REC2 y-size
               }
            }
            else {                                                                                      //--- Handle sell orders
               if(YD_R1 > YD_R3 + YS_R3) {                                                              //--- Ensure TP is below entry for sell orders
                  canMove = true;                                                                       //--- Allow movement
                  ObjectSetInteger(0, REC1, OBJPROP_YDISTANCE, mlbDownYD_R1 + MouseD_Y - mlbDownY1);    //--- Update REC1 y-position
                  ObjectSetInteger(0, REC4, OBJPROP_YDISTANCE, YD_R3 + YS_R3);                          //--- Update REC4 y-position
                  ObjectSetInteger(0, REC4, OBJPROP_YSIZE, YD_R1 - (YD_R3 + YS_R3));                    //--- Update REC4 y-size
               }
            }

            if(canMove) {                                                                               //--- If movement is valid
               datetime dt_TP = 0;                                                                      //--- Variable for TP time
               double price_TP = 0;                                                                     //--- Variable for TP price
               int window = 0;                                                                          //--- Chart window

               ChartXYToTimePrice(0, XD_R1, YD_R1 + YS_R1, window, dt_TP, price_TP);                    //--- Convert chart coordinates to time and price
               ObjectSetInteger(0, TP_HL, OBJPROP_TIME, dt_TP);                                         //--- Update TP horizontal line time
               ObjectSetDouble(0, TP_HL, OBJPROP_PRICE, price_TP);                                      //--- Update TP horizontal line price

               update_Text(REC1, "TP: " + DoubleToString(MathAbs((Get_Price_d(TP_HL) - Get_Price_d(PR_HL)) / _Point), 0) + " Points | " + Get_Price_s(TP_HL)); //--- Update REC1 text
               update_Text(TP_LABEL, "TP: " + Get_Price_s(TP_HL));                                      //--- Update TP label text
            }

            updateRectangleColors();                                                                    //--- Update rectangle colors
            ChartRedraw(0);                                                                             //--- Redraw chart
         }

         if(movingState_R5) {                                                                           //--- Handle REC5 (SL) movement
            bool canMove = false;                                                                       //--- Flag to check if movement is valid
            if(selected_order_type == "BUY_STOP" || selected_order_type == "BUY_LIMIT") {               //--- Check for buy orders
               if(YD_R5 > YD_R4) {                                                                      //--- Ensure SL is below entry for buy orders
                  canMove = true;                                                                       //--- Allow movement
                  ObjectSetInteger(0, REC5, OBJPROP_YDISTANCE, mlbDownYD_R5 + MouseD_Y - mlbDownY5);    //--- Update REC5 y-position
                  ObjectSetInteger(0, REC4, OBJPROP_YDISTANCE, YD_R3 + YS_R3);                          //--- Update REC4 y-position
                  ObjectSetInteger(0, REC4, OBJPROP_YSIZE, YD_R5 - (YD_R3 + YS_R3));                    //--- Update REC4 y-size
               }
            }
            else {                                                                                      //--- Handle sell orders
               if(YD_R5 + YS_R5 < YD_R3) {                                                              //--- Ensure SL is above entry for sell orders
                  canMove = true;                                                                       //--- Allow movement
                  ObjectSetInteger(0, REC5, OBJPROP_YDISTANCE, mlbDownYD_R5 + MouseD_Y - mlbDownY5);    //--- Update REC5 y-position
                  ObjectSetInteger(0, REC2, OBJPROP_YDISTANCE, YD_R5 + YS_R5);                          //--- Update REC2 y-position
                  ObjectSetInteger(0, REC2, OBJPROP_YSIZE, YD_R3 - (YD_R5 + YS_R5));                    //--- Update REC2 y-size
               }
            }

            if(canMove) {                                                                               //--- If movement is valid
               datetime dt_SL = 0;                                                                      //--- Variable for SL time
               double price_SL = 0;                                                                     //--- Variable for SL price
               int window = 0;                                                                          //--- Chart window

               ChartXYToTimePrice(0, XD_R5, YD_R5 + YS_R5, window, dt_SL, price_SL);                    //--- Convert chart coordinates to time and price
               ObjectSetInteger(0, SL_HL, OBJPROP_TIME, dt_SL);                                         //--- Update SL horizontal line time
               ObjectSetDouble(0, SL_HL, OBJPROP_PRICE, price_SL);                                      //--- Update SL horizontal line price

               update_Text(REC5, "SL: " + DoubleToString(MathAbs((Get_Price_d(PR_HL) - Get_Price_d(SL_HL)) / _Point), 0) + " Points | " + Get_Price_s(SL_HL)); //--- Update REC5 text
               update_Text(SL_LABEL, "SL: " + Get_Price_s(SL_HL));                                      //--- Update SL label text
            }

            updateRectangleColors();                                                                    //--- Update rectangle colors
            ChartRedraw(0);                                                                             //--- Redraw chart
         }

         if(movingState_R3) { //--- Handle REC3 (Entry) movement
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

            datetime dt_PRC = 0, dt_SL1 = 0, dt_TP1 = 0;                                       //--- Variables for time
            double price_PRC = 0, price_SL1 = 0, price_TP1 = 0;                                //--- Variables for price
            int window = 0;                                                                    //--- Chart window

            ChartXYToTimePrice(0, XD_R3, YD_R3 + YS_R3, window, dt_PRC, price_PRC);            //--- Convert REC3 coordinates to time and price
            ChartXYToTimePrice(0, XD_R5, YD_R5 + YS_R5, window, dt_SL1, price_SL1);            //--- Convert REC5 coordinates to time and price
            ChartXYToTimePrice(0, XD_R1, YD_R1 + YS_R1, window, dt_TP1, price_TP1);            //--- Convert REC1 coordinates to time and price

            ObjectSetInteger(0, PR_HL, OBJPROP_TIME, dt_PRC);                                  //--- Update entry horizontal line time
            ObjectSetDouble(0, PR_HL, OBJPROP_PRICE, price_PRC);                               //--- Update entry horizontal line price

            ObjectSetInteger(0, TP_HL, OBJPROP_TIME, dt_TP1);                                  //--- Update TP horizontal line time
            ObjectSetDouble(0, TP_HL, OBJPROP_PRICE, price_TP1);                               //--- Update TP horizontal line price

            ObjectSetInteger(0, SL_HL, OBJPROP_TIME, dt_SL1);                                  //--- Update SL horizontal line time
            ObjectSetDouble(0, SL_HL, OBJPROP_PRICE, price_SL1);                               //--- Update SL horizontal line price

            update_Text(REC1, "TP: " + DoubleToString(MathAbs((Get_Price_d(TP_HL) - Get_Price_d(PR_HL)) / _Point), 0) + " Points | " + Get_Price_s(TP_HL)); //--- Update REC1 text
            update_Text(REC3, selected_order_type + ": | Lot: " + DoubleToString(lot_size, 2) + " | " + Get_Price_s(PR_HL));                                //--- Update REC3 text
            update_Text(REC5, "SL: " + DoubleToString(MathAbs((Get_Price_d(PR_HL) - Get_Price_d(SL_HL)) / _Point), 0) + " Points | " + Get_Price_s(SL_HL)); //--- Update REC5 text
            update_Text(PRICE_LABEL, "Entry: " + Get_Price_s(PR_HL));                                                                                       //--- Update entry label text
            update_Text(SL_LABEL, "SL: " + Get_Price_s(SL_HL));                                                                                             //--- Update SL label text
            update_Text(TP_LABEL, "TP: " + Get_Price_s(TP_HL));                                                                                             //--- Update TP label text

            updateRectangleColors(); //--- Update rectangle colors
            ChartRedraw(0); //--- Redraw chart
         }

         if(MouseState == 0) {                               //--- Check if mouse button is released
            movingState_R1 = false;                          //--- Disable REC1 movement
            movingState_R3 = false;                          //--- Disable REC3 movement
            movingState_R5 = false;                          //--- Disable REC5 movement
            ChartSetInteger(0, CHART_MOUSE_SCROLL, true);    //--- Enable chart scrolling
         }
      }
      prevMouseState = MouseState;                           //--- Update previous mouse state
   }
}
```

Since we had already defined the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) function, we will just concentrate on pinpointing the enhanced logic we have added to incorporate the new interactivity features, such as panel dragging, hover state updates, and order validation. For [CHARTEVENT\_OBJECT\_CLICK](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents), we extend button click handling for "BUY\_STOP\_BTN", "SELL\_STOP\_BTN", "BUY\_LIMIT\_BTN", and "SELL\_LIMIT\_BTN" by calling the "updateRectangleColors" function to reflect order validity visually, and for "PLACE\_ORDER\_BTN", we add a check with the "isOrderValid" function, logging an error via the [Print](https://www.mql5.com/en/docs/common/print) function if invalid, preventing erroneous trades as visualized below.

![ORDER VALIDITY](https://c.mql5.com/2/138/PART_2.1.gif)

We also introduce the "updateButtonHoverState" function after clicks to refresh hover effects, using "lparam" and "dparam" for mouse coordinates. For [CHARTEVENT\_MOUSE\_MOVE](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents), we add panel dragging by checking if the mouse click is within "PANEL\_HEADER" bounds (obtained via [ObjectGetInteger](https://www.mql5.com/en/docs/objects/objectgetinteger)), setting "panel\_dragging" to true, storing coordinates in "panel\_drag\_x", "panel\_drag\_y", "panel\_start\_x", and "panel\_start\_y", and disabling scrolling with [ChartSetInteger](https://www.mql5.com/en/docs/chart_operations/chartsetinteger).

While dragging ("panel\_dragging" and "MouseState" 1), we calculate displacement ("dx", "dy"), update "panel\_x" and "panel\_y", and reposition all panel objects ("PANEL\_BG", "PANEL\_HEADER", "CLOSE\_BTN", "LOT\_EDIT", etc.) using [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger), calling [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to update the chart. On mouse release, we reset "panel\_dragging" and re-enable scrolling. We ensure rectangle dragging (for "REC1", "REC3", "REC5") avoids conflicts with panel dragging by checking "!panel\_dragging", and update colors with "updateRectangleColors" during "movingState\_R1", "movingState\_R5", and "movingState\_R3" to reflect hover and validity states.

We have highlighted some of the snippets that are crucial to note. Here is a visualization.

![PANEL DRAGGING AND HOVER](https://c.mql5.com/2/138/PART_2.2.gif)

Also, since we have used the header panel object, here is how we create it.

```
//+------------------------------------------------------------------+
//| Create control panel                                             |
//+------------------------------------------------------------------+
void createControlPanel() {
   // Background rectangle
   ObjectCreate(0, PANEL_BG, OBJ_RECTANGLE_LABEL, 0, 0, 0);                   //--- Create panel background rectangle
   ObjectSetInteger(0, PANEL_BG, OBJPROP_XDISTANCE, panel_x);                 //--- Set background x-position
   ObjectSetInteger(0, PANEL_BG, OBJPROP_YDISTANCE, panel_y);                 //--- Set background y-position
   ObjectSetInteger(0, PANEL_BG, OBJPROP_XSIZE, 250);                         //--- Set background width
   ObjectSetInteger(0, PANEL_BG, OBJPROP_YSIZE, 280);                         //--- Set background height
   ObjectSetInteger(0, PANEL_BG, OBJPROP_BGCOLOR, C'070,070,070');            //--- Set background color
   ObjectSetInteger(0, PANEL_BG, OBJPROP_BORDER_COLOR, clrWhite);             //--- Set border color
   ObjectSetInteger(0, PANEL_BG, OBJPROP_BACK, false);                        //--- Set background to foreground

   // Header rectangle (inside panel)

   createButton(PANEL_HEADER,"",panel_x+2,panel_y+2,250-4,28-3,clrBlue,C'050,050,050',12,C'050,050,050',false);

   createButton(CLOSE_BTN, CharToString(203), panel_x + 209, panel_y + 1, 40, 25, clrWhite, clrCrimson, 12, clrBlack, false, "Wingdings"); //--- Create close button

//---

}
```

In the "createControlPanel" function, we add a "PANEL\_HEADER" button to our tool’s control panel using the "createButton" function, placed at "panel\_x+2", "panel\_y+2" with size 246x25, styled with blue text ( [clrBlue](https://www.mql5.com/en/docs/constants/objectconstants/webcolors)), dark gray background/border ("C'050,050,050'"), and no label, enabling panel dragging in [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent). The other thing that we need to do is destroy the panel as below.

```
//+------------------------------------------------------------------+
//| Delete control panel objects                                     |
//+------------------------------------------------------------------+
void deletePanel() {
   ObjectDelete(0, PANEL_BG);        //--- Delete panel background
   ObjectDelete(0, PANEL_HEADER);    //--- Delete panel header
   ObjectDelete(0, LOT_EDIT);        //--- Delete lot edit field
   ObjectDelete(0, PRICE_LABEL);     //--- Delete price label
   ObjectDelete(0, SL_LABEL);        //--- Delete SL label
   ObjectDelete(0, TP_LABEL);        //--- Delete TP label
   ObjectDelete(0, BUY_STOP_BTN);    //--- Delete Buy Stop button
   ObjectDelete(0, SELL_STOP_BTN);   //--- Delete Sell Stop button
   ObjectDelete(0, BUY_LIMIT_BTN);   //--- Delete Buy Limit button
   ObjectDelete(0, SELL_LIMIT_BTN);  //--- Delete Sell Limit button
   ObjectDelete(0, PLACE_ORDER_BTN); //--- Delete Place Order button
   ObjectDelete(0, CANCEL_BTN);      //--- Delete Cancel button
   ObjectDelete(0, CLOSE_BTN);       //--- Delete Close button
   ChartRedraw(0);                   //--- Redraw chart
}
```

Here, we update the "deletePanel" function to ensure our tool’s control panel is properly cleaned up by removing all associated objects, including the new header that we have introduced recently. We use the [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete) function to remove the panel background ("PANEL\_BG"), the newly added header ("PANEL\_HEADER"), the lot size input field ("LOT\_EDIT"), labels ("PRICE\_LABEL", "SL\_LABEL", "TP\_LABEL"), and buttons ("BUY\_STOP\_BTN", "SELL\_STOP\_BTN", "BUY\_LIMIT\_BTN", "SELL\_LIMIT\_BTN", "PLACE\_ORDER\_BTN", "CANCEL\_BTN", "CLOSE\_BTN") from the MetaTrader 5 chart.

Finally, we call the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function to refresh the chart, ensuring a clean interface after deletion. Also, when showing the tool, we have to consider the hover effects to ensure they remain salient as below.

```
//+------------------------------------------------------------------+
//| Show control panel                                               |
//+------------------------------------------------------------------+
void showPanel() {
   // Ensure panel is in foreground
   ObjectSetInteger(0, PANEL_BG,        OBJPROP_BACK, false); //--- Show panel background
   ObjectSetInteger(0, PANEL_HEADER,    OBJPROP_BACK, false); //--- Show panel header
   ObjectSetInteger(0, LOT_EDIT,        OBJPROP_BACK, false); //--- Show lot edit field
   ObjectSetInteger(0, PRICE_LABEL,     OBJPROP_BACK, false); //--- Show price label
   ObjectSetInteger(0, SL_LABEL,        OBJPROP_BACK, false); //--- Show SL label
   ObjectSetInteger(0, TP_LABEL,        OBJPROP_BACK, false); //--- Show TP label
   ObjectSetInteger(0, BUY_STOP_BTN,    OBJPROP_BACK, false); //--- Show Buy Stop button
   ObjectSetInteger(0, SELL_STOP_BTN,   OBJPROP_BACK, false); //--- Show Sell Stop button
   ObjectSetInteger(0, BUY_LIMIT_BTN,   OBJPROP_BACK, false); //--- Show Buy Limit button
   ObjectSetInteger(0, SELL_LIMIT_BTN,  OBJPROP_BACK, false); //--- Show Sell Limit button
   ObjectSetInteger(0, PLACE_ORDER_BTN, OBJPROP_BACK, false); //--- Show Place Order button
   ObjectSetInteger(0, CANCEL_BTN,      OBJPROP_BACK, false); //--- Show Cancel button
   ObjectSetInteger(0, CLOSE_BTN,       OBJPROP_BACK, false); //--- Show Close button

   // Reset button hover states
   buy_stop_hovered     = false;
   sell_stop_hovered    = false;
   buy_limit_hovered    = false;
   sell_limit_hovered   = false;
   place_order_hovered  = false;
   cancel_hovered       = false;
   close_hovered        = false;
   header_hovered       = false;

   // Reset button colors
   ObjectSetInteger(0, BUY_STOP_BTN,    OBJPROP_BGCOLOR,       clrForestGreen);
   ObjectSetInteger(0, BUY_STOP_BTN,    OBJPROP_BORDER_COLOR,  clrBlack);
   ObjectSetInteger(0, SELL_STOP_BTN,   OBJPROP_BGCOLOR,       clrFireBrick);
   ObjectSetInteger(0, SELL_STOP_BTN,   OBJPROP_BORDER_COLOR,  clrBlack);
   ObjectSetInteger(0, BUY_LIMIT_BTN,   OBJPROP_BGCOLOR,       clrForestGreen);
   ObjectSetInteger(0, BUY_LIMIT_BTN,   OBJPROP_BORDER_COLOR,  clrBlack);
   ObjectSetInteger(0, SELL_LIMIT_BTN,  OBJPROP_BGCOLOR,       clrFireBrick);
   ObjectSetInteger(0, SELL_LIMIT_BTN,  OBJPROP_BORDER_COLOR,  clrBlack);
   ObjectSetInteger(0, PLACE_ORDER_BTN, OBJPROP_BGCOLOR,       clrDodgerBlue);
   ObjectSetInteger(0, PLACE_ORDER_BTN, OBJPROP_BORDER_COLOR,  clrBlack);
   ObjectSetInteger(0, CANCEL_BTN,      OBJPROP_BGCOLOR,       clrSlateGray);
   ObjectSetInteger(0, CANCEL_BTN,      OBJPROP_BORDER_COLOR,  clrBlack);
   ObjectSetInteger(0, CLOSE_BTN,       OBJPROP_BGCOLOR,       clrCrimson);
   ObjectSetInteger(0, CLOSE_BTN,       OBJPROP_BORDER_COLOR,  clrBlack);
   ObjectSetInteger(0, PANEL_HEADER,    OBJPROP_BGCOLOR,       C'050,050,050');

   // Reset panel state
   update_Text(PRICE_LABEL,     "Entry: -");        //--- Reset entry label text
   update_Text(SL_LABEL,        "SL: -");           //--- Reset SL label text
   update_Text(TP_LABEL,        "TP: -");           //--- Reset TP label text
   update_Text(PLACE_ORDER_BTN, "Place Order");     //--- Reset Place Order button text
   selected_order_type = "";                        //--- Clear selected order type
   tool_visible        = false;                     //--- Hide tool
   ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, true); //--- Ensure mouse move events are enabled
   ChartRedraw(0);                                   //--- Redraw chart
}
```

We enhance the "showPanel" function to manage the display and state reset of our tool’s control panel, incorporating the new "PANEL\_HEADER" and hover state management introduced to support the tool’s improved interactivity. We begin by ensuring all panel elements are visible by using the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/ObjectSetInteger) function to set the [OBJPROP\_BACK](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) property to false for the panel background ("PANEL\_BG"), the newly added header ("PANEL\_HEADER"), the lot size input field ("LOT\_EDIT"), price-related labels ("PRICE\_LABEL", "SL\_LABEL", "TP\_LABEL"), and all buttons ("BUY\_STOP\_BTN", "SELL\_STOP\_BTN", "BUY\_LIMIT\_BTN", "SELL\_LIMIT\_BTN", "PLACE\_ORDER\_BTN", "CANCEL\_BTN", "CLOSE\_BTN"), bringing them to the foreground of the chart.

To maintain a clean and predictable interface, we reset the hover states by setting the boolean variables "buy\_stop\_hovered", "sell\_stop\_hovered", "buy\_limit\_hovered", "sell\_limit\_hovered", "place\_order\_hovered", "cancel\_hovered", "close\_hovered", and "header\_hovered" to false, ensuring no residual hover effects persist when the panel is shown.

We then restore the default visual appearance of the buttons and header by using the "ObjectSetInteger" function to set [OBJPROP\_BGCOLOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to their original colors: "clrForestGreen" for "BUY\_STOP\_BTN" and "BUY\_LIMIT\_BTN", " [clrFireBrick](https://www.mql5.com/en/docs/constants/objectconstants/webcolors)" for "SELL\_STOP\_BTN" and "SELL\_LIMIT\_BTN", "clrDodgerBlue" for "PLACE\_ORDER\_BTN", "clrSlateGray" for "CANCEL\_BTN", "clrCrimson" for "CLOSE\_BTN", and a dark gray ("C'050,050,050'") for "PANEL\_HEADER".

We also set [OBJPROP\_BORDER\_COLOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to "clrBlack" for all buttons to ensure a consistent, non-hovered appearance.

To reset the panel’s functional state, we call the "update\_Text" function to set the "PRICE\_LABEL" to "Entry: -", "SL\_LABEL" to "SL: -", "TP\_LABEL" to "TP: -", and "PLACE\_ORDER\_BTN" to "Place Order", clearing any previous trade setup information. We clear the "selected\_order\_type" variable to ensure no order type is pre-selected, set "tool\_visible" to false to hide the chart tool, and use the [ChartSetInteger](https://www.mql5.com/en/docs/chart_operations/chartsetinteger) function to enable [CHART\_EVENT\_MOUSE\_MOVE](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer) events, ensuring the panel is ready for hover and drag interactions.

Finally, we call the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function to refresh the chart, rendering the panel in its default state, fully prepared for our next interaction. Upon compilation, here is the outcome.

![FINAL OUTCOME](https://c.mql5.com/2/138/PART_2.3.gif)

From the visualization, we can see that we can be able to dynamically validate the orders via the price tool and change its colors to alert the user that prices are out of bounds. Also, we can see that we can be able to drag the panel and the price tool dynamically, and upon hovering the buttons, we can get their ranges and change colors based on the button ranges dynamically, hence achieving our objective. What now remains is testing the interactivity of the project, and that is handled in the preceding section.

### Backtesting

We did the testing and below is the compiled visualization in a single [Graphics Interchange Format](https://en.wikipedia.org/wiki/GIF "https://en.wikipedia.org/wiki/GIF") (GIF) bitmap image format.

![BACKTEST GIF](https://c.mql5.com/2/138/PART_2.4.gif)

### Conclusion

In conclusion, we’ve enhanced our Trade Assistant Tool in [MQL5](https://www.mql5.com/) with dynamic visual feedback, incorporating a draggable panel, hover effects, and real-time order validation to make our pending order placement more intuitive and precise. We’ve demonstrated the design and implementation of these improvements, ensuring their reliability through thorough backtesting tailored to our trading needs. You can customize this tool to suit your style, significantly improving your order placement efficiency in the trading charts.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17972.zip "Download all attachments in the single ZIP archive")

[Trade\_Assistant\_GUI\_Tool\_Part\_2.mq5](https://www.mql5.com/en/articles/download/17972/trade_assistant_gui_tool_part_2.mq5 "Download Trade_Assistant_GUI_Tool_Part_2.mq5")(59.4 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/486190)**

![Raw Code Optimization and Tweaking for Improving Back-Test Results](https://c.mql5.com/2/140/Raw_Code_Optimization_and_Tweaking_for_Improving_Back-Test_Results___logo.png)[Raw Code Optimization and Tweaking for Improving Back-Test Results](https://www.mql5.com/en/articles/17702)

Enhance your MQL5 code by optimizing logic, refining calculations, and reducing execution time to improve back-test accuracy. Fine-tune parameters, optimize loops, and eliminate inefficiencies for better performance.

![MQL5 Wizard Techniques you should know (Part 64): Using Patterns of DeMarker and Envelope Channels with the White-Noise Kernel](https://c.mql5.com/2/141/MQL5_Wizard_Techniques_you_should_know_cPart_64i_Using_Patterns_of_DeMarker_and_Envelope_Channels_wi.png)[MQL5 Wizard Techniques you should know (Part 64): Using Patterns of DeMarker and Envelope Channels with the White-Noise Kernel](https://www.mql5.com/en/articles/18033)

The DeMarker Oscillator and the Envelopes' indicator are momentum and support/ resistance tools that can be paired when developing an Expert Advisor. We continue from our last article that introduced these pair of indicators by adding machine learning to the mix. We are using a recurrent neural network that uses the white-noise kernel to process vectorized signals from these two indicators. This is done in a custom signal class file that works with the MQL5 wizard to assemble an Expert Advisor.

![From Basic to Intermediate: Arrays and Strings (II)](https://c.mql5.com/2/95/Do_bisico_ao_intermedi2rio_Array_e_Strings_I__LOGO.png)[From Basic to Intermediate: Arrays and Strings (II)](https://www.mql5.com/en/articles/15442)

In this article I will show that although we are still at a very basic stage of programming, we can already implement some interesting applications. In this case, we will create a fairly simple password generator. This way we will be able to apply some of the concepts that have been explained so far. In addition, we will look at how solutions can be developed for some specific problems.

![Developing a Replay System (Part 67): Refining the Control Indicator](https://c.mql5.com/2/95/Desenvolvendo_um_sistema_de_Replay_Parte_67____LOGO.png)[Developing a Replay System (Part 67): Refining the Control Indicator](https://www.mql5.com/en/articles/12293)

In this article, we'll look at what can be achieved with a little code refinement. This refinement is aimed at simplifying our code, making more use of MQL5 library calls and, above all, making it much more stable, secure and easy to use in other projects that we may develop in the future.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/17972&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068806887102676457)

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