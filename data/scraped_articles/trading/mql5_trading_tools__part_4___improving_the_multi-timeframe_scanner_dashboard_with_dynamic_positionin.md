---
title: MQL5 Trading Tools (Part 4): Improving the Multi-Timeframe Scanner Dashboard with Dynamic Positioning and Toggle Features
url: https://www.mql5.com/en/articles/18786
categories: Trading, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T17:53:54.491507
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/18786&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068773592516197760)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 3)](https://www.mql5.com/en/articles/18319), we developed a Multi-Timeframe Scanner Dashboard in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5), displaying [Relative Strength Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi") (RSI), Stochastic, Commodity Channel Index (CCI), Average Directional Index (ADX), and [Awesome Oscillator](https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/awesome "https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/awesome") (AO) indicators across multiple timeframes to identify trading signals for the current symbol. In Part 4, we enhance this dashboard by adding dynamic positioning to allow dragging across the chart and a toggle feature to minimize or maximize the display, improving usability and screen management. We will cover the following topics:

1. [Understanding the Dynamic Positioning and Toggle Architecture](https://www.mql5.com/en/articles/18786#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/18786#para2)
3. [Backtesting](https://www.mql5.com/en/articles/18786#para3)
4. [Conclusion](https://www.mql5.com/en/articles/18786#para4)

By the end, you’ll have an advanced MQL5 dashboard with flexible positioning and toggle functionality, ready for testing and further customization—let’s get started!

### Understanding the Dynamic Positioning and Toggle Architecture

We’re enhancing our Multi-Timeframe Scanner Dashboard by adding dynamic positioning, allowing it to be dragged across the chart, and a toggle to minimize or maximize it, thereby improving usability. These features are essential for avoiding chart clutter and optimizing screen space for efficient trading analysis. We will implement mouse-driven dragging to reposition the dashboard and a toggle button to switch between compact and full views, maintaining seamless indicator updates for better trading decisions. In a nutshell, here is a visualization of what we aim to achieve.

![POSITIONING & DRAG STATES ARCHITECTURE](https://c.mql5.com/2/155/Screenshot_2025-07-09_000429.png)

### Implementation in MQL5

To implement the enhancements in [MQL5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5"), we will define the extra toggle button object that we will use later for switching between maximized and minimized states, as the hover and drag will be done on the header object, which we already have in the implementation.

```
// Define identifiers and properties for UI elements
#define MAIN_PANEL              "PANEL_MAIN"                     //--- Main panel rectangle identifier

//--- THE REST OF THE EXISTING OBJECTS

#define TOGGLE_BUTTON           "BUTTON_TOGGLE"                  //--- Toggle (minimize/maximize) button identifier

//--- THE REST OF THE EXISTING OBJECTS

#define COLOR_DARK_GRAY         C'105,105,105'                   //--- Dark gray color for indicator backgrounds
```

We start the enhancement of our Multi-Timeframe Scanner Dashboard by updating the [User Interface](https://en.wikipedia.org/wiki/User_interface "https://en.wikipedia.org/wiki/User_interface") (UI) element definitions to include a new identifier for the toggle button, supporting our goal of adding minimize/maximize functionality. We retain the existing definitions to maintain the dashboard’s core structure. The key change is the addition of the "TOGGLE\_BUTTON" identifier, defined as "BUTTON\_TOGGLE", which will allow us to create a button for toggling the dashboard between minimized and maximized states.

This addition is crucial as it will enable the new toggle feature, giving us the ability to collapse the dashboard to save screen space or expand it for full visibility, improving usability without altering the existing indicator display logic. Then, we will need some more control over global variables, which we will use to store the dashboard states.

```
bool panel_minimized = false;                                    //--- Flag to control minimized state
int panel_x = 632, panel_y = 40;                                 //--- Panel position coordinates
bool panel_dragging = false;                                     //--- Flag to track if panel is being dragged
int panel_drag_x = 0, panel_drag_y = 0;                          //--- Mouse coordinates when drag starts
int panel_start_x = 0, panel_start_y = 0;                        //--- Panel coordinates when drag starts
int prev_mouse_state = 0;                                        //--- Variable to track previous mouse state
bool header_hovered = false;                                     //--- Header hover state
bool toggle_hovered = false;                                     //--- Toggle button hover state
bool close_hovered = false;                                      //--- Close button hover state
int last_mouse_x = 0, last_mouse_y = 0;                          //--- Track last mouse position for optimization
bool prev_header_hovered = false;                                //--- Track previous header hover state
bool prev_toggle_hovered = false;                                //--- Track previous toggle hover state
bool prev_close_hovered = false;                                 //--- Track previous close button hover state
```

To help track the dashboard states, we add some global variables. We introduce "panel\_minimized" to track the minimized state, "panel\_x" and "panel\_y" for the dashboard’s position, and "panel\_dragging", "panel\_drag\_x", "panel\_drag\_y", "panel\_start\_x", and "panel\_start\_y" for dragging. We also add "prev\_mouse\_state", "header\_hovered", "toggle\_hovered", "close\_hovered", "last\_mouse\_x", "last\_mouse\_y", "prev\_header\_hovered", "prev\_toggle\_hovered", and "prev\_close\_hovered" to manage mouse events and hover states. These will enable a draggable, interactive dashboard with toggle support.

We can then move on to adding the toggle button to the dashboard. To manage the switch states, we will need to have the creation logic in separate functions for both the maximized and minimized panels. Let's start with the maximized dashboard.

```
//+------------------------------------------------------------------+
//| Create full dashboard UI                                         |
//+------------------------------------------------------------------+
void create_full_dashboard() {
   create_rectangle(MAIN_PANEL, panel_x, panel_y, 617, 374, C'30,30,30', BORDER_FLAT); //--- Create main panel background
   create_rectangle(HEADER_PANEL, panel_x, panel_y, 617, 27, C'60,60,60', BORDER_FLAT); //--- Create header panel background
   create_label(HEADER_PANEL_ICON, CharToString(91), panel_x - 12, panel_y + 14, 18, clrAqua, "Wingdings"); //--- Create header icon
   create_label(HEADER_PANEL_TEXT, "TimeframeScanner", panel_x - 105, panel_y + 12, 13, COLOR_WHITE); //--- Create header title
   create_label(CLOSE_BUTTON, CharToString('r'), panel_x - 600, panel_y + 14, 18, clrYellow, "Webdings"); //--- Create close button
   create_label(TOGGLE_BUTTON, CharToString('r'), panel_x - 570, panel_y + 14, 18, clrYellow, "Wingdings"); //--- Create minimize button (-)

   // Create header rectangle and label
   create_rectangle(SYMBOL_RECTANGLE, panel_x - 2, panel_y + 35, WIDTH_TIMEFRAME, HEIGHT_RECTANGLE, clrGray); //--- Create symbol rectangle
   create_label(SYMBOL_TEXT, _Symbol, panel_x - 47, panel_y + 45, 11, COLOR_WHITE); //--- Create symbol label

   // Create summary and indicator headers (rectangles and labels)
   string header_names[] = {"BUY", "SELL", "RSI", "STOCH", "CCI", "ADX", "AO"}; //--- Define header titles
   for(int header_index = 0; header_index < ArraySize(header_names); header_index++) { //--- Loop through headers
      int x_offset = panel_x - WIDTH_TIMEFRAME - (header_index < 2 ? header_index * WIDTH_SIGNAL : 2 * WIDTH_SIGNAL + (header_index - 2) * WIDTH_INDICATOR) + (1 + header_index); //--- Calculate x position
      int width = (header_index < 2 ? WIDTH_SIGNAL : WIDTH_INDICATOR); //--- Set width based on header type
      create_rectangle(HEADER_RECTANGLE + IntegerToString(header_index), x_offset, panel_y + 35, width, HEIGHT_RECTANGLE, clrGray); //--- Create header rectangle
      create_label(HEADER_TEXT + IntegerToString(header_index), header_names[header_index], x_offset - width/2, panel_y + 45, 11, COLOR_WHITE); //--- Create header label
   }

   // Create timeframe rectangles and labels, and summary/indicator cells
   for(int timeframe_index = 0; timeframe_index < ArraySize(timeframes_array); timeframe_index++) { //--- Loop through timeframes
      // Highlight current timeframe
      color timeframe_background = (timeframes_array[timeframe_index] == _Period) ? clrLimeGreen : clrGray; //--- Set background color for current timeframe
      color timeframe_text_color = (timeframes_array[timeframe_index] == _Period) ? COLOR_BLACK : COLOR_WHITE; //--- Set text color for current timeframe

      create_rectangle(TIMEFRAME_RECTANGLE + IntegerToString(timeframe_index), panel_x - 2, (panel_y + 35 + HEIGHT_RECTANGLE) + timeframe_index * HEIGHT_RECTANGLE - (1 + timeframe_index), WIDTH_TIMEFRAME, HEIGHT_RECTANGLE, timeframe_background); //--- Create timeframe rectangle
      create_label(TIMEFRAME_TEXT + IntegerToString(timeframe_index), truncate_timeframe_name(timeframe_index), panel_x - 47, (panel_y + 45 + HEIGHT_RECTANGLE) + timeframe_index * HEIGHT_RECTANGLE - (1 + timeframe_index), 11, timeframe_text_color); //--- Create timeframe label

      // Create summary and indicator cells
      for(int header_index = 0; header_index < ArraySize(header_names); header_index++) { //--- Loop through headers for cells
         string cell_rectangle_name, cell_text_name;              //--- Declare cell name and label variables
         color cell_background = (header_index < 2) ? COLOR_LIGHT_GRAY : COLOR_BLACK; //--- Set cell background color
         switch(header_index) {                                   //--- Select cell type
            case 0: cell_rectangle_name = BUY_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = BUY_TEXT + IntegerToString(timeframe_index); break; //--- Buy cell
            case 1: cell_rectangle_name = SELL_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = SELL_TEXT + IntegerToString(timeframe_index); break; //--- Sell cell
            case 2: cell_rectangle_name = RSI_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = RSI_TEXT + IntegerToString(timeframe_index); break; //--- RSI cell
            case 3: cell_rectangle_name = STOCH_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = STOCH_TEXT + IntegerToString(timeframe_index); break; //--- Stochastic cell
            case 4: cell_rectangle_name = CCI_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = CCI_TEXT + IntegerToString(timeframe_index); break; //--- CCI cell
            case 5: cell_rectangle_name = ADX_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = ADX_TEXT + IntegerToString(timeframe_index); break; //--- ADX cell
            case 6: cell_rectangle_name = AO_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = AO_TEXT + IntegerToString(timeframe_index); break; //--- AO cell
         }
         int x_offset = panel_x - WIDTH_TIMEFRAME - (header_index < 2 ? header_index * WIDTH_SIGNAL : 2 * WIDTH_SIGNAL + (header_index - 2) * WIDTH_INDICATOR) + (1 + header_index); //--- Calculate x position
         int width = (header_index < 2 ? WIDTH_SIGNAL : WIDTH_INDICATOR); //--- Set width based on cell type
         create_rectangle(cell_rectangle_name, x_offset, (panel_y + 35 + HEIGHT_RECTANGLE) + timeframe_index * HEIGHT_RECTANGLE - (1 + timeframe_index), width, HEIGHT_RECTANGLE, cell_background); //--- Create cell rectangle
         create_label(cell_text_name, "-/-", x_offset - width/2, (panel_y + 45 + HEIGHT_RECTANGLE) + timeframe_index * HEIGHT_RECTANGLE - (1 + timeframe_index), 10, COLOR_WHITE); //--- Create cell label
      }
   }
   ChartRedraw(0);
}
```

We implement the dashboard states by creating the "create\_full\_dashboard" function, moving the static creation logic to it, and updating it to support dynamic positioning and toggle functionality. The primary change is integrating the "panel\_x" and "panel\_y" variables into the positioning of all dashboard elements, allowing the dashboard to be placed anywhere on the chart. We create the main panel with "create\_rectangle" using "MAIN\_PANEL", "panel\_x", and "panel\_y" for the position, maintaining the 617x374 size. Similarly, we position the header panel, icon, and title using "create\_rectangle" and "create\_label" for "HEADER\_PANEL", "HEADER\_PANEL\_ICON", and "HEADER\_PANEL\_TEXT", adjusting their x and y coordinates relative to "panel\_x" and "panel\_y".

We add the new "TOGGLE\_BUTTON" with "create\_label", placing it at "panel\_x - 570" and "panel\_y + 14", displaying a minimize symbol ('r' in Webdings). The symbol rectangle and label ("SYMBOL\_RECTANGLE" and "SYMBOL\_TEXT") and header rectangles and labels ("HEADER\_RECTANGLE" and "HEADER\_TEXT") use "panel\_x" and "panel\_y" for their x and y offsets, ensuring they move with the panel. For each timeframe in "timeframes\_array", we create timeframe rectangles and labels ("TIMEFRAME\_RECTANGLE" and "TIMEFRAME\_TEXT") and indicator cells ("RSI\_RECTANGLE", "STOCH\_RECTANGLE", etc.) with positions calculated relative to "panel\_x" and "panel\_y", preserving the layout but making it relocatable. We call the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function to update the display. Unlike the previous version’s fixed coordinates (e.g., 632, 40), this function uses dynamic coordinates, enabling the dashboard to be dragged, and adds a toggle button for minimizing/maximizing. We will get something that resembles the below sample.

![MAXIMIZED STATE](https://c.mql5.com/2/155/Screenshot_2025-07-09_005846.png)

We can then have a function to create the minimized panel.

```
//+------------------------------------------------------------------+
//| Create minimized dashboard UI                                    |
//+------------------------------------------------------------------+
void create_minimized_dashboard() {
   create_rectangle(HEADER_PANEL, panel_x, panel_y, 617, 27, C'60,60,60', BORDER_FLAT); //--- Create header panel background
   create_label(HEADER_PANEL_ICON, CharToString(91), panel_x - 12, panel_y + 14, 18, clrAqua, "Wingdings"); //--- Create header icon
   create_label(HEADER_PANEL_TEXT, "TimeframeScanner", panel_x - 105, panel_y + 12, 13, COLOR_WHITE); //--- Create header title
   create_label(CLOSE_BUTTON, CharToString('r'), panel_x - 600, panel_y + 14, 18, clrYellow, "Webdings"); //--- Create close button
   create_label(TOGGLE_BUTTON, CharToString('o'), panel_x - 570, panel_y + 14, 18, clrYellow, "Wingdings"); //--- Create maximize button (+)
   ChartRedraw(0);
}
```

To support toggling to a compact view, we define the "create\_minimized\_dashboard" function. We create the header panel with "create\_rectangle" for "HEADER\_PANEL" at "panel\_x" and "panel\_y", add "HEADER\_PANEL\_ICON" and "HEADER\_PANEL\_TEXT" with "create\_label" for the icon and title, include "CLOSE\_BUTTON", and add "TOGGLE\_BUTTON" with a maximize symbol ('o' in Webdings) at "panel\_x - 570" and "panel\_y + 14". We call "ChartRedraw" to update the display, enabling a movable, minimized dashboard state. As you might have noticed, we chose the [Wingdings](https://en.wikipedia.org/wiki/Wingdings "https://en.wikipedia.org/wiki/Wingdings") font for both the maximize and minimize buttons because of consistency. You can choose any of your like. In our case, the icon characters are 'r' and 'o' respectively. Here is a centralized view of them.

![WINGDINGS FONT CHARACTERS](https://c.mql5.com/2/155/C_SYMBOL_FONTS_-_Copy.png)

When you run the minimized panel state, you get the following outcome.

![MINIMIZED STATE](https://c.mql5.com/2/155/Screenshot_2025-07-09_010000.png)

Armed with the functions, we can choose whichever to use based on the user command. Now, since we created several objects, we can centralize the deletion of all the objects in a function because we will need to delete the objects and create them again automatically.

```
//+------------------------------------------------------------------+
//| Delete all dashboard objects                                     |
//+------------------------------------------------------------------+
void delete_all_objects() {
   ObjectDelete(0, MAIN_PANEL);                                  //--- Delete main panel
   ObjectDelete(0, HEADER_PANEL);                                //--- Delete header panel
   ObjectDelete(0, HEADER_PANEL_ICON);                           //--- Delete header icon
   ObjectDelete(0, HEADER_PANEL_TEXT);                           //--- Delete header title
   ObjectDelete(0, CLOSE_BUTTON);                                //--- Delete close button
   ObjectDelete(0, TOGGLE_BUTTON);                               //--- Delete toggle button

   ObjectsDeleteAll(0, SYMBOL_RECTANGLE);                        //--- Delete all symbol rectangles
   ObjectsDeleteAll(0, SYMBOL_TEXT);                             //--- Delete all symbol labels
   ObjectsDeleteAll(0, TIMEFRAME_RECTANGLE);                     //--- Delete all timeframe rectangles
   ObjectsDeleteAll(0, TIMEFRAME_TEXT);                          //--- Delete all timeframe labels
   ObjectsDeleteAll(0, HEADER_RECTANGLE);                        //--- Delete all header rectangles
   ObjectsDeleteAll(0, HEADER_TEXT);                             //--- Delete all header labels
   ObjectsDeleteAll(0, RSI_RECTANGLE);                           //--- Delete all RSI rectangles
   ObjectsDeleteAll(0, RSI_TEXT);                                //--- Delete all RSI labels
   ObjectsDeleteAll(0, STOCH_RECTANGLE);                         //--- Delete all Stochastic rectangles
   ObjectsDeleteAll(0, STOCH_TEXT);                              //--- Delete all Stochastic labels
   ObjectsDeleteAll(0, CCI_RECTANGLE);                           //--- Delete all CCI rectangles
   ObjectsDeleteAll(0, CCI_TEXT);                                //--- Delete all CCI labels
   ObjectsDeleteAll(0, ADX_RECTANGLE);                           //--- Delete all ADX rectangles
   ObjectsDeleteAll(0, ADX_TEXT);                                //--- Delete all ADX labels
   ObjectsDeleteAll(0, AO_RECTANGLE);                            //--- Delete all AO rectangles
   ObjectsDeleteAll(0, AO_TEXT);                                 //--- Delete all AO labels
   ObjectsDeleteAll(0, BUY_RECTANGLE);                           //--- Delete all buy rectangles
   ObjectsDeleteAll(0, BUY_TEXT);                                //--- Delete all buy labels
   ObjectsDeleteAll(0, SELL_RECTANGLE);                          //--- Delete all sell rectangles
   ObjectsDeleteAll(0, SELL_TEXT);                               //--- Delete all sell labels
}
```

To have more control over when to delete our objects, we create and update the "delete\_all\_objects" function to include the removal of the new "TOGGLE\_BUTTON", supporting our toggle functionality improvement. We add the [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete) function for "TOGGLE\_BUTTON" to the list of objects being removed, ensuring the toggle button is properly deleted when the dashboard is closed or toggled. We retain the deletion of all other objects, such as "MAIN\_PANEL", "HEADER\_PANEL", "HEADER\_PANEL\_ICON", "HEADER\_PANEL\_TEXT", "CLOSE\_BUTTON", and all symbol, timeframe, header, indicator, and signal rectangles and labels using the [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/objectdeleteall) function. This change ensures our movable and minimizable dashboard cleans up all components, including the new toggle button, maintaining a tidy chart when the dashboard is hidden or reinitialized.

To enhance dynamic positioning, we will need to create a function that will create and update the dashboard elements based on the cursor position. Here is the logic we use to achieve that.

```
//+------------------------------------------------------------------+
//| Update panel object positions                                    |
//+------------------------------------------------------------------+
void update_panel_positions() {
   // Update header and buttons
   ObjectSetInteger(0, HEADER_PANEL, OBJPROP_XDISTANCE, panel_x); //--- Set header panel x position
   ObjectSetInteger(0, HEADER_PANEL, OBJPROP_YDISTANCE, panel_y); //--- Set header panel y position
   ObjectSetInteger(0, HEADER_PANEL_ICON, OBJPROP_XDISTANCE, panel_x - 12); //--- Set header icon x position
   ObjectSetInteger(0, HEADER_PANEL_ICON, OBJPROP_YDISTANCE, panel_y + 14); //--- Set header icon y position
   ObjectSetInteger(0, HEADER_PANEL_TEXT, OBJPROP_XDISTANCE, panel_x - 105); //--- Set header title x position
   ObjectSetInteger(0, HEADER_PANEL_TEXT, OBJPROP_YDISTANCE, panel_y + 12); //--- Set header title y position
   ObjectSetInteger(0, CLOSE_BUTTON, OBJPROP_XDISTANCE, panel_x - 600); //--- Set close button x position
   ObjectSetInteger(0, CLOSE_BUTTON, OBJPROP_YDISTANCE, panel_y + 14); //--- Set close button y position
   ObjectSetInteger(0, TOGGLE_BUTTON, OBJPROP_XDISTANCE, panel_x - 570); //--- Set toggle button x position
   ObjectSetInteger(0, TOGGLE_BUTTON, OBJPROP_YDISTANCE, panel_y + 14); //--- Set toggle button y position

   if (!panel_minimized) {
      // Update main panel
      ObjectSetInteger(0, MAIN_PANEL, OBJPROP_XDISTANCE, panel_x); //--- Set main panel x position
      ObjectSetInteger(0, MAIN_PANEL, OBJPROP_YDISTANCE, panel_y); //--- Set main panel y position

      // Update symbol rectangle and label
      ObjectSetInteger(0, SYMBOL_RECTANGLE, OBJPROP_XDISTANCE, panel_x - 2); //--- Set symbol rectangle x position
      ObjectSetInteger(0, SYMBOL_RECTANGLE, OBJPROP_YDISTANCE, panel_y + 35); //--- Set symbol rectangle y position
      ObjectSetInteger(0, SYMBOL_TEXT, OBJPROP_XDISTANCE, panel_x - 47); //--- Set symbol text x position
      ObjectSetInteger(0, SYMBOL_TEXT, OBJPROP_YDISTANCE, panel_y + 45); //--- Set symbol text y position

      // Update header rectangles and labels
      string header_names[] = {"BUY", "SELL", "RSI", "STOCH", "CCI", "ADX", "AO"};
      for(int header_index = 0; header_index < ArraySize(header_names); header_index++) { //--- Loop through headers
         int x_offset = panel_x - WIDTH_TIMEFRAME - (header_index < 2 ? header_index * WIDTH_SIGNAL : 2 * WIDTH_SIGNAL + (header_index - 2) * WIDTH_INDICATOR) + (1 + header_index); //--- Calculate x position
         ObjectSetInteger(0, HEADER_RECTANGLE + IntegerToString(header_index), OBJPROP_XDISTANCE, x_offset); //--- Set header rectangle x position
         ObjectSetInteger(0, HEADER_RECTANGLE + IntegerToString(header_index), OBJPROP_YDISTANCE, panel_y + 35); //--- Set header rectangle y position
         ObjectSetInteger(0, HEADER_TEXT + IntegerToString(header_index), OBJPROP_XDISTANCE, x_offset - (header_index < 2 ? WIDTH_SIGNAL/2 : WIDTH_INDICATOR/2)); //--- Set header text x position
         ObjectSetInteger(0, HEADER_TEXT + IntegerToString(header_index), OBJPROP_YDISTANCE, panel_y + 45); //--- Set header text y position
      }

      // Update timeframe rectangles, labels, and cells
      for(int timeframe_index = 0; timeframe_index < ArraySize(timeframes_array); timeframe_index++) { //--- Loop through timeframes
         int y_offset = (panel_y + 35 + HEIGHT_RECTANGLE) + timeframe_index * HEIGHT_RECTANGLE - (1 + timeframe_index); //--- Calculate y position
         ObjectSetInteger(0, TIMEFRAME_RECTANGLE + IntegerToString(timeframe_index), OBJPROP_XDISTANCE, panel_x - 2); //--- Set timeframe rectangle x position
         ObjectSetInteger(0, TIMEFRAME_RECTANGLE + IntegerToString(timeframe_index), OBJPROP_YDISTANCE, y_offset); //--- Set timeframe rectangle y position
         ObjectSetInteger(0, TIMEFRAME_TEXT + IntegerToString(timeframe_index), OBJPROP_XDISTANCE, panel_x - 47); //--- Set timeframe text x position
         ObjectSetInteger(0, TIMEFRAME_TEXT + IntegerToString(timeframe_index), OBJPROP_YDISTANCE, y_offset + 10); //--- Set timeframe text y position

         for(int header_index = 0; header_index < ArraySize(header_names); header_index++) { //--- Loop through cells
            string cell_rectangle_name, cell_text_name;
            switch(header_index) { //--- Select cell type
               case 0: cell_rectangle_name = BUY_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = BUY_TEXT + IntegerToString(timeframe_index); break; //--- Buy cell
               case 1: cell_rectangle_name = SELL_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = SELL_TEXT + IntegerToString(timeframe_index); break; //--- Sell cell
               case 2: cell_rectangle_name = RSI_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = RSI_TEXT + IntegerToString(timeframe_index); break; //--- RSI cell
               case 3: cell_rectangle_name = STOCH_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = STOCH_TEXT + IntegerToString(timeframe_index); break; //--- Stochastic cell
               case 4: cell_rectangle_name = CCI_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = CCI_TEXT + IntegerToString(timeframe_index); break; //--- CCI cell
               case 5: cell_rectangle_name = ADX_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = ADX_TEXT + IntegerToString(timeframe_index); break; //--- ADX cell
               case 6: cell_rectangle_name = AO_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = AO_TEXT + IntegerToString(timeframe_index); break; //--- AO cell
            }
            int x_offset = panel_x - WIDTH_TIMEFRAME - (header_index < 2 ? header_index * WIDTH_SIGNAL : 2 * WIDTH_SIGNAL + (header_index - 2) * WIDTH_INDICATOR) + (1 + header_index); //--- Calculate x position
            int width = (header_index < 2 ? WIDTH_SIGNAL : WIDTH_INDICATOR); //--- Set cell width
            ObjectSetInteger(0, cell_rectangle_name, OBJPROP_XDISTANCE, x_offset); //--- Set cell rectangle x position
            ObjectSetInteger(0, cell_rectangle_name, OBJPROP_YDISTANCE, y_offset); //--- Set cell rectangle y position
            ObjectSetInteger(0, cell_text_name, OBJPROP_XDISTANCE, x_offset - width/2); //--- Set cell text x position
            ObjectSetInteger(0, cell_text_name, OBJPROP_YDISTANCE, y_offset + 10); //--- Set cell text y position
         }
      }
   }
   ChartRedraw(0);                                               //--- Redraw chart
}
```

To support dynamic positioning of the dashboard, we introduce a new "update\_panel\_positions" function. The function adjusts the positions of all dashboard elements based on the current "panel\_x" and "panel\_y" coordinates, enabling the dashboard to be dragged across the chart. We update the header panel, icon, title, close button, and toggle button ("HEADER\_PANEL", "HEADER\_PANEL\_ICON", "HEADER\_PANEL\_TEXT", "CLOSE\_BUTTON", "TOGGLE\_BUTTON") using the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/ObjectSetInteger) function with [OBJPROP\_XDISTANCE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) and "OBJPROP\_YDISTANCE", setting their positions relative to "panel\_x" and "panel\_y".

If "panel\_minimized" is false, we reposition the main panel ("MAIN\_PANEL"), symbol rectangle and label ("SYMBOL\_RECTANGLE", "SYMBOL\_TEXT"), header rectangles and labels ("HEADER\_RECTANGLE", "HEADER\_TEXT"), and timeframe rectangles, labels, and indicator cells ("TIMEFRAME\_RECTANGLE", "TIMEFRAME\_TEXT", "BUY\_RECTANGLE", "RSI\_RECTANGLE", etc.) using calculated offsets from "panel\_x" and "panel\_y". We call the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function to refresh the display. This function will ensure all elements move together when the dashboard is dragged, a critical feature for our movable dashboard enhancement. We can then try testing the dashboard. For this, we will call the function to create the full dashboard in the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler and call the destroy function in the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) event handler.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()                                                     //--- Initialize EA
{
   create_full_dashboard();                                      //--- Create full dashboard
   ArraySetAsSeries(rsi_values, true);                           //--- Set RSI array as timeseries
   ArraySetAsSeries(stochastic_values, true);                    //--- Set Stochastic array as timeseries
   ArraySetAsSeries(cci_values, true);                           //--- Set CCI array as timeseries
   ArraySetAsSeries(adx_values, true);                           //--- Set ADX array as timeseries
   ArraySetAsSeries(ao_values, true);                            //--- Set AO array as timeseries

   ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, true);             //--- Enable mouse move events
   return(INIT_SUCCEEDED);                                       //--- Return initialization success
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)                                  //--- Deinitialize EA
{
   delete_all_objects();                                         //--- Delete all objects
   ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, false);            //--- Disable mouse move events
   ChartRedraw(0);                                               //--- Redraw chart
}
```

Here, we just call the respective functions during initialization and deinitialization of the program to enable us to test the basic responses before we move on to the next parts. It is always a better programming manner to compile your program stepwise to ascertain that it is working. We have the following outcome upon compilation.

![INIT AND DEINIT GIF](https://c.mql5.com/2/155/Scanner_GIF_1.gif)

From the visualization, we can see that we can initialize and remove the panel successfully. Now, let us move on to making the dashboard responsive. We first need to get the cursor position to determine if it is in the header or in the buttons, so that we know specifically what the user might want to do. We will also get the button movement path so we can know and visualize hover states and click states by changing the color, but first, let us define a function to determine the cursor position relative to the dashboard elements.

```
//+------------------------------------------------------------------+
//| Check if cursor is inside header or buttons                      |
//+------------------------------------------------------------------+
bool is_cursor_in_header_or_buttons(int mouse_x, int mouse_y) {
   int chart_width = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);

   // Header panel bounds
   int header_x = (int)ObjectGetInteger(0, HEADER_PANEL, OBJPROP_XDISTANCE);
   int header_y = (int)ObjectGetInteger(0, HEADER_PANEL, OBJPROP_YDISTANCE);
   int header_width = (int)ObjectGetInteger(0, HEADER_PANEL, OBJPROP_XSIZE);
   int header_height = (int)ObjectGetInteger(0, HEADER_PANEL, OBJPROP_YSIZE);
   int header_left = chart_width - header_x;
   int header_right = header_left + header_width;
   bool in_header = (mouse_x >= header_left && mouse_x <= header_right &&
                     mouse_y >= header_y && mouse_y <= header_y + header_height);

   // Close button bounds
   int close_x = (int)ObjectGetInteger(0, CLOSE_BUTTON, OBJPROP_XDISTANCE);
   int close_y = (int)ObjectGetInteger(0, CLOSE_BUTTON, OBJPROP_YDISTANCE);
   int close_width = 20;
   int close_height = 20;
   int close_left = chart_width - close_x;
   int close_right = close_left + close_width;
   bool in_close = (mouse_x >= close_left && mouse_x <= close_right &&
                    mouse_y >= close_y && mouse_y <= close_y + close_height);

   // Toggle button bounds
   int toggle_x = (int)ObjectGetInteger(0, TOGGLE_BUTTON, OBJPROP_XDISTANCE);
   int toggle_y = (int)ObjectGetInteger(0, TOGGLE_BUTTON, OBJPROP_YDISTANCE);
   int toggle_width = 20;
   int toggle_height = 20;
   int toggle_left = chart_width - toggle_x;
   int toggle_right = toggle_left + toggle_width;
   bool in_toggle = (mouse_x >= toggle_left && mouse_x <= toggle_right &&
                     mouse_y >= toggle_y && mouse_y <= toggle_y + toggle_height);

   return in_header || in_close || in_toggle;
}
```

First, we introduce a new "is\_cursor\_in\_header\_or\_buttons" function to support dynamic positioning and toggle functionality. The function will check if the mouse cursor is over the header panel, close button, or toggle button, enabling interactive dragging and button actions. We start by retrieving the chart width using [ChartGetInteger](https://www.mql5.com/en/docs/chart_operations/chartgetinteger) with [CHART\_WIDTH\_IN\_PIXELS](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer). For the header panel, we get "header\_x", "header\_y", "header\_width", and "header\_height" using [ObjectGetInteger](https://www.mql5.com/en/docs/objects/objectgetinteger) with "OBJPROP\_XDISTANCE", "OBJPROP\_YDISTANCE", "OBJPROP\_XSIZE", and [OBJPROP\_YSIZE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) for "HEADER\_PANEL", calculating "header\_left" and "header\_right" relative to the chart width. We check if "mouse\_x" and "mouse\_y" fall within these bounds, setting "in\_header" to true if they do.

For the close button, we fetch "close\_x" and "close\_y" for "CLOSE\_BUTTON", define a 20x20 pixel area, and calculate "close\_left" and "close\_right", setting "in\_close" to true if the cursor is within this area. Similarly, for the toggle button, we get "toggle\_x" and "toggle\_y" for "TOGGLE\_BUTTON", define a 20x20 area, and set "in\_toggle" to true if the cursor is inside. We return true if the cursor is in any of these areas ("in\_header", "in\_close", or "in\_toggle"). This function will be crucial for detecting mouse interactions with the dashboard’s draggable header and buttons, enabling our enhanced movable and interactive features. We can then update the hover states by visualizing the colors for easier recognition.

```
//+------------------------------------------------------------------+
//| Update button hover states                                       |
//+------------------------------------------------------------------+
void update_button_hover_states(int mouse_x, int mouse_y) {
   int chart_width = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);

   // Close button hover
   int close_x = (int)ObjectGetInteger(0, CLOSE_BUTTON, OBJPROP_XDISTANCE);
   int close_y = (int)ObjectGetInteger(0, CLOSE_BUTTON, OBJPROP_YDISTANCE);
   int close_width = 20;
   int close_height = 20;
   int close_left = chart_width - close_x;
   int close_right = close_left + close_width;
   bool is_close_hovered = (mouse_x >= close_left && mouse_x <= close_right &&
                            mouse_y >= close_y && mouse_y <= close_y + close_height);

   if (is_close_hovered != prev_close_hovered) {
      ObjectSetInteger(0, CLOSE_BUTTON, OBJPROP_COLOR, is_close_hovered ? clrWhite : clrYellow);
      ObjectSetInteger(0, CLOSE_BUTTON, OBJPROP_BGCOLOR, is_close_hovered ? clrDodgerBlue : clrNONE);
      prev_close_hovered = is_close_hovered;
      ChartRedraw(0);
   }

   // Toggle button hover
   int toggle_x = (int)ObjectGetInteger(0, TOGGLE_BUTTON, OBJPROP_XDISTANCE);
   int toggle_y = (int)ObjectGetInteger(0, TOGGLE_BUTTON, OBJPROP_YDISTANCE);
   int toggle_width = 20;
   int toggle_height = 20;
   int toggle_left = chart_width - toggle_x;
   int toggle_right = toggle_left + toggle_width;
   bool is_toggle_hovered = (mouse_x >= toggle_left && mouse_x <= toggle_right &&
                             mouse_y >= toggle_y && mouse_y <= toggle_y + toggle_height);

   if (is_toggle_hovered != prev_toggle_hovered) {
      ObjectSetInteger(0, TOGGLE_BUTTON, OBJPROP_COLOR, is_toggle_hovered ? clrWhite : clrYellow);
      ObjectSetInteger(0, TOGGLE_BUTTON, OBJPROP_BGCOLOR, is_toggle_hovered ? clrDodgerBlue : clrNONE);
      prev_toggle_hovered = is_toggle_hovered;
      ChartRedraw(0);
   }
}

//+------------------------------------------------------------------+
//| Update header hover state                                        |
//+------------------------------------------------------------------+
void update_header_hover_state(int mouse_x, int mouse_y) {
   int chart_width = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
   int header_x = (int)ObjectGetInteger(0, HEADER_PANEL, OBJPROP_XDISTANCE);
   int header_y = (int)ObjectGetInteger(0, HEADER_PANEL, OBJPROP_YDISTANCE);
   int header_width = (int)ObjectGetInteger(0, HEADER_PANEL, OBJPROP_XSIZE);
   int header_height = (int)ObjectGetInteger(0, HEADER_PANEL, OBJPROP_YSIZE);
   int header_left = chart_width - header_x;
   int header_right = header_left + header_width;

   // Exclude button areas from header hover
   int close_x = (int)ObjectGetInteger(0, CLOSE_BUTTON, OBJPROP_XDISTANCE);
   int close_y = (int)ObjectGetInteger(0, CLOSE_BUTTON, OBJPROP_YDISTANCE);
   int close_width = 20;
   int close_height = 20;
   int close_left = chart_width - close_x;
   int close_right = close_left + close_width;

   int toggle_x = (int)ObjectGetInteger(0, TOGGLE_BUTTON, OBJPROP_XDISTANCE);
   int toggle_y = (int)ObjectGetInteger(0, TOGGLE_BUTTON, OBJPROP_YDISTANCE);
   int toggle_width = 20;
   int toggle_height = 20;
   int toggle_left = chart_width - toggle_x;
   int toggle_right = toggle_left + toggle_width;

   bool is_header_hovered = (mouse_x >= header_left && mouse_x <= header_right &&
                             mouse_y >= header_y && mouse_y <= header_y + header_height &&
                             !(mouse_x >= close_left && mouse_x <= close_right &&
                               mouse_y >= close_y && mouse_y <= close_y + close_height) &&
                             !(mouse_x >= toggle_left && mouse_x <= toggle_right &&
                               mouse_y >= toggle_y && mouse_y <= toggle_y + toggle_height));

   if (is_header_hovered != prev_header_hovered && !panel_dragging) {
      ObjectSetInteger(0, HEADER_PANEL, OBJPROP_BGCOLOR, is_header_hovered ? clrRed : C'60,60,60');
      prev_header_hovered = is_header_hovered;
      ChartRedraw(0);
   }

   update_button_hover_states(mouse_x, mouse_y);
}
```

Here, we introduce two new functions, "update\_button\_hover\_states" and "update\_header\_hover\_state," to add visual feedback for user interactions, improving the dashboard’s usability. We start with the "update\_button\_hover\_states" function, which takes "mouse\_x" and "mouse\_y" to detect hover over the close and toggle buttons. For the "CLOSE\_BUTTON", we fetch "close\_x" and "close\_y" using [ObjectGetInteger](https://www.mql5.com/en/docs/objects/objectgetinteger) with "OBJPROP\_XDISTANCE" and "OBJPROP\_YDISTANCE", calculate a 20x20 pixel area relative to the chart width from [ChartGetInteger](https://www.mql5.com/en/docs/chart_operations/chartgetinteger) with "CHART\_WIDTH\_IN\_PIXELS", and set "is\_close\_hovered" if the cursor is within this area.

If "is\_close\_hovered" differs from "prev\_close\_hovered", we update "CLOSE\_BUTTON" with "ObjectSetInteger", setting "OBJPROP\_COLOR" to "clrWhite" and [OBJPROP\_BGCOLOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to "clrDodgerBlue" when hovered, or "clrYellow" and "clrNONE" when not, update "prev\_close\_hovered", and call the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function. Similarly, for "TOGGLE\_BUTTON", we fetch "toggle\_x" and "toggle\_y", check the 20x20 area, and update its colors and "prev\_toggle\_hovered" if "is\_toggle\_hovered" changes, ensuring responsive button feedback.

Next, we create the "update\_header\_hover\_state" function, also taking "mouse\_x" and "mouse\_y". We get "header\_x", "header\_y", "header\_width", and "header\_height" for "HEADER\_PANEL" using "ObjectGetInteger", calculate the header’s bounds, and exclude the areas of "CLOSE\_BUTTON" and "TOGGLE\_BUTTON" (20x20 pixels each) to avoid overlap. If "is\_header\_hovered" differs from "prev\_header\_hovered" and "panel\_dragging" is false, we update "HEADER\_PANEL"’s "OBJPROP\_BGCOLOR" to "clrRed" when hovered or "C'60,60,60'" when not, update "prev\_header\_hovered", and call "ChartRedraw". We then call "update\_button\_hover\_states" to ensure button states are updated. These functions will provide visual cues for dragging and button interactions, enhancing the dashboard’s interactivity. We can then use the functions in the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler for full implementation. Here is the logic we apply.

```
//+------------------------------------------------------------------+
//| Expert chart event handler                                       |
//+------------------------------------------------------------------+
void OnChartEvent(const int event_id, const long& long_param, const double& double_param, const string& string_param)
{
   if (event_id == CHARTEVENT_OBJECT_CLICK) {                    //--- Handle object click event
      if (string_param == CLOSE_BUTTON) {                        //--- Check if close button clicked
         Print("Closing the panel now");                         //--- Log panel closure
         PlaySound("alert.wav");                                 //--- Play alert sound
         panel_is_visible = false;                               //--- Hide panel
         delete_all_objects();                                   //--- Delete all objects
         ChartRedraw(0);                                         //--- Redraw chart
      } else if (string_param == TOGGLE_BUTTON) {                //--- Toggle button clicked
         delete_all_objects();                                   //--- Delete current UI
         panel_minimized = !panel_minimized;                     //--- Toggle minimized state
         if (panel_minimized) {
            Print("Minimizing the panel");                       //--- Log minimization
            create_minimized_dashboard();                        //--- Create minimized UI
         } else {
            Print("Maximizing the panel");                       //--- Log maximization
            create_full_dashboard();                             //--- Create full UI
         }
         // Reset hover states after toggle
         prev_header_hovered = false;
         prev_close_hovered = false;
         prev_toggle_hovered = false;
         ObjectSetInteger(0, HEADER_PANEL, OBJPROP_BGCOLOR, C'60,60,60');
         ObjectSetInteger(0, CLOSE_BUTTON, OBJPROP_COLOR, clrYellow);
         ObjectSetInteger(0, CLOSE_BUTTON, OBJPROP_BGCOLOR, clrNONE);
         ObjectSetInteger(0, TOGGLE_BUTTON, OBJPROP_COLOR, clrYellow);
         ObjectSetInteger(0, TOGGLE_BUTTON, OBJPROP_BGCOLOR, clrNONE);
         ChartRedraw(0);
      }
   }
   else if (event_id == CHARTEVENT_MOUSE_MOVE && panel_is_visible) { //--- Handle mouse move events
      int mouse_x = (int)long_param;                             //--- Get mouse x-coordinate
      int mouse_y = (int)double_param;                           //--- Get mouse y-coordinate
      int mouse_state = (int)string_param;                       //--- Get mouse state

      if (mouse_x == last_mouse_x && mouse_y == last_mouse_y && !panel_dragging) { //--- Skip redundant updates
         return;
      }
      last_mouse_x = mouse_x;                                    //--- Update last mouse x position
      last_mouse_y = mouse_y;                                    //--- Update last mouse y position

      update_header_hover_state(mouse_x, mouse_y);               //--- Update header and button hover states

      int chart_width = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
      int header_x = (int)ObjectGetInteger(0, HEADER_PANEL, OBJPROP_XDISTANCE);
      int header_y = (int)ObjectGetInteger(0, HEADER_PANEL, OBJPROP_YDISTANCE);
      int header_width = (int)ObjectGetInteger(0, HEADER_PANEL, OBJPROP_XSIZE);
      int header_height = (int)ObjectGetInteger(0, HEADER_PANEL, OBJPROP_YSIZE);
      int header_left = chart_width - header_x;
      int header_right = header_left + header_width;

      int close_x = (int)ObjectGetInteger(0, CLOSE_BUTTON, OBJPROP_XDISTANCE);
      int close_width = 20;
      int close_left = chart_width - close_x;
      int close_right = close_left + close_width;

      int toggle_x = (int)ObjectGetInteger(0, TOGGLE_BUTTON, OBJPROP_XDISTANCE);
      int toggle_width = 20;
      int toggle_left = chart_width - toggle_x;
      int toggle_right = toggle_left + toggle_width;

      if (prev_mouse_state == 0 && mouse_state == 1) {           //--- Detect mouse button down
         if (mouse_x >= header_left && mouse_x <= header_right &&
             mouse_y >= header_y && mouse_y <= header_y + header_height &&
             !(mouse_x >= close_left && mouse_x <= close_right) &&
             !(mouse_x >= toggle_left && mouse_x <= toggle_right)) { //--- Exclude button areas
            panel_dragging = true;                              //--- Start dragging
            panel_drag_x = mouse_x;                             //--- Store mouse x-coordinate
            panel_drag_y = mouse_y;                             //--- Store mouse y-coordinate
            panel_start_x = header_x;                           //--- Store panel x-coordinate
            panel_start_y = header_y;                           //--- Store panel y-coordinate
            ObjectSetInteger(0, HEADER_PANEL, OBJPROP_BGCOLOR, clrMediumBlue); //--- Set header to blue on drag start
            ChartSetInteger(0, CHART_MOUSE_SCROLL, false);      //--- Disable chart scrolling
         }
      }

      if (panel_dragging && mouse_state == 1) {                  //--- Handle dragging
         int dx = mouse_x - panel_drag_x;                        //--- Calculate x displacement
         int dy = mouse_y - panel_drag_y;                        //--- Calculate y displacement
         panel_x = panel_start_x - dx;                           //--- Update panel x-position (inverted for CORNER_RIGHT_UPPER)
         panel_y = panel_start_y + dy;                           //--- Update panel y-position

         int chart_width = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS); //--- Get chart width
         int chart_height = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS); //--- Get chart height
         panel_x = MathMax(617, MathMin(chart_width, panel_x));  //--- Keep panel within right edge
         panel_y = MathMax(0, MathMin(chart_height - (panel_minimized ? 27 : 374), panel_y)); //--- Adjust height based on state

         update_panel_positions();                               //--- Update all panel object positions
         ChartRedraw(0);                                         //--- Redraw chart during dragging
      }

      if (mouse_state == 0 && prev_mouse_state == 1) {           //--- Detect mouse button release
         if (panel_dragging) {
            panel_dragging = false;                              //--- Stop dragging
            update_header_hover_state(mouse_x, mouse_y);          //--- Update hover state immediately after drag
            ChartSetInteger(0, CHART_MOUSE_SCROLL, true);        //--- Re-enable chart scrolling
            ChartRedraw(0);                                      //--- Redraw chart
         }
      }

      prev_mouse_state = mouse_state;                            //--- Update previous mouse state
   }
}
```

Here, we enhance our program by updating the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler to support dynamic positioning and toggle functionality, significantly improving on the previous version. We retain the handling of [CHARTEVENT\_OBJECT\_CLICK](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents) for the "CLOSE\_BUTTON", which logs closure with [Print](https://www.mql5.com/en/docs/common/print), plays a sound with [PlaySound](https://www.mql5.com/en/docs/common/playsound), sets "panel\_is\_visible" to false, calls the "delete\_all\_objects" function, and redraws the chart.

The new addition is the handling of clicks on the "TOGGLE\_BUTTON", where we call the "delete\_all\_objects" function, toggle "panel\_minimized", and either create a minimized dashboard with "create\_minimized\_dashboard" (logging "Minimizing the panel") or a full dashboard with "create\_full\_dashboard" (logging "Maximizing the panel"). We reset hover states ("prev\_header\_hovered", "prev\_close\_hovered", "prev\_toggle\_hovered") to false, restore default colors for "HEADER\_PANEL", "CLOSE\_BUTTON", and "TOGGLE\_BUTTON" using the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/ObjectSetInteger) function, and redraw the chart.

For dynamic positioning, we add [CHARTEVENT\_MOUSE\_MOVE](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents) handling when "panel\_is\_visible" is true. We get "mouse\_x", "mouse\_y", and "mouse\_state" from event parameters, skip redundant updates if coordinates match "last\_mouse\_x" and "last\_mouse\_y" and "panel\_dragging" is false, and update these coordinates. We call "update\_header\_hover\_state" to manage hover effects. If "prev\_mouse\_state" is 0 and "mouse\_state" is 1, we check if the cursor is over "HEADER\_PANEL" (excluding "CLOSE\_BUTTON" and "TOGGLE\_BUTTON" areas) using "ObjectGetInteger" and [ChartGetInteger](https://www.mql5.com/en/docs/chart_operations/chartgetinteger), then set "panel\_dragging" to true, store coordinates in "panel\_drag\_x", "panel\_drag\_y", "panel\_start\_x", and "panel\_start\_y", set "HEADER\_PANEL" color to "clrMediumBlue", and disable chart scrolling with "ChartSetInteger".

While "panel\_dragging" and "mouse\_state" are 1, we calculate displacement, update "panel\_x" and "panel\_y" within chart bounds, call "update\_panel\_positions", and redraw. On mouse release, we stop dragging, update hover states, re-enable scrolling, and redraw. We update "prev\_mouse\_state". These changes will now enable dragging and toggling, unlike the previous static click-only handling. Upon compilation, we get the following outcome.

![DRAG, HOVER AND VISIBILITY STATES](https://c.mql5.com/2/155/Scanner_GIF_2.gif)

From the visualization, we can see the dashboard is coming to life, but we need to take care of the conflicts between the event handlers. We need to give one event handler the right of way so it can be prioritized to increase productivity. For example, when we are hovering or dragging, or even in minimized mode, we need to prioritize the chart events. If we are in the dormant state or maximized state, we need to prioritize the dashboard updates. We will achieve that by altering the tick processing event, since that is where we update the dashboard.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()                                                     //--- Handle tick events
{
   if (panel_is_visible && !panel_minimized && !is_cursor_in_header_or_buttons(last_mouse_x, last_mouse_y)) { //--- Update indicators only if panel is visible, not minimized, and cursor is not in header/buttons
      updateIndicators();                                        //--- Update indicators
   }
}
```

We update the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler to optimize indicator updates. Unlike the previous version, which called the "updateIndicators" function solely based on "panel\_is\_visible", we now add conditions to only update indicators when "panel\_is\_visible" is true, "panel\_minimized" is false, and the cursor is not over the header or buttons, checked using "is\_cursor\_in\_header\_or\_buttons" with "last\_mouse\_x" and "last\_mouse\_y". This change will ensure that indicator updates are paused when the dashboard is minimized or the user is interacting with the header, close button, or toggle button, reducing unnecessary processing and improving performance during dragging or toggling actions. Upon compilation, we have the following outcome.

![IMPROVED PERFORMANCE GIF](https://c.mql5.com/2/155/Scanner_GIF_3.gif)

From the visualization, we can see that there is improved performance in the dashboard, and all the objectives have been met. What now remains is testing the workability of the project, and that is handled in the preceding section.

### Backtesting

We did the testing, and below is the compiled visualization in a single [Graphics Interchange Format](https://en.wikipedia.org/wiki/GIF "https://en.wikipedia.org/wiki/GIF") (GIF) bitmap image format.

![DASHBOARD BACKTEST](https://c.mql5.com/2/155/SCANNER_GIF_FINAL.gif)

### Conclusion

In conclusion, we’ve enhanced our Multi-Timeframe Scanner Dashboard in [MQL5](https://www.mql5.com/) by adding dynamic positioning and toggle functionality, building on [Part 3](https://www.mql5.com/en/articles/18319) with a movable interface, minimize/maximize toggle, and interactive hover effects for better user control. We’ve demonstrated how to implement these improvements using functions like "create\_minimized\_dashboard" and "update\_header\_hover\_state", ensuring seamless integration with the existing indicator grid for real-time trading insights. You can further customize this dashboard to suit your trading needs, boosting your ability to monitor market signals efficiently across multiple timeframes.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18786.zip "Download all attachments in the single ZIP archive")

[TimeframeScanner\_Dashboard\_-\_Movable\_EA.mq5](https://www.mql5.com/en/articles/download/18786/timeframescanner_dashboard_-_movable_ea.mq5 "Download TimeframeScanner_Dashboard_-_Movable_EA.mq5")(50.66 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/491248)**
(2)


![linfo2](https://c.mql5.com/avatar/2023/4/6438c14d-e2f0.png)

**[linfo2](https://www.mql5.com/en/users/neilhazelwood)**
\|
17 Jul 2025 at 17:38

Thank you Allan , That is pretty cool , well documented and covers features I didn't know much appreciated


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
17 Jul 2025 at 18:58

**linfo2 [#](https://www.mql5.com/en/forum/491248#comment_57562960):**

Thank you Allan , That is pretty cool , well documented and covers features I didn't know much appreciated

[@linfo2](https://www.mql5.com/en/users/neilhazelwood) so much welcomed. Thank you.


![Implementing Practical Modules from Other Languages in MQL5 (Part 02): Building the REQUESTS Library, Inspired by Python](https://c.mql5.com/2/157/18728-implementing-practical-modules-logo.png)[Implementing Practical Modules from Other Languages in MQL5 (Part 02): Building the REQUESTS Library, Inspired by Python](https://www.mql5.com/en/articles/18728)

In this article, we implement a module similar to requests offered in Python to make it easier to send and receive web requests in MetaTrader 5 using MQL5.

![Cycles and trading](https://c.mql5.com/2/103/Cycles_and_Trading___LOGO.png)[Cycles and trading](https://www.mql5.com/en/articles/16494)

This article is about using cycles in trading. We will consider building a trading strategy based on cyclical models.

![MQL5 Wizard Techniques you should know (Part 75): Using Awesome Oscillator and the Envelopes](https://c.mql5.com/2/157/18842-mql5-wizard-techniques-you-logo__1.png)[MQL5 Wizard Techniques you should know (Part 75): Using Awesome Oscillator and the Envelopes](https://www.mql5.com/en/articles/18842)

The Awesome Oscillator by Bill Williams and the Envelopes Channel are a pairing that could be used complimentarily within an MQL5 Expert Advisor. We use the Awesome Oscillator for its ability to spot trends, while the envelopes channel is incorporated to define our support/resistance levels. In exploring this indicator pairing, we use the MQL5 wizard to build and test any potential these two may possess.

![From Basic to Intermediate: Recursion](https://c.mql5.com/2/102/Do_bbsico_ao_intermedilrio_Recursividade__LOGO.png)[From Basic to Intermediate: Recursion](https://www.mql5.com/en/articles/15504)

In this article we will look at a very interesting and quite challenging programming concept, although it should be treated with great caution, as its misuse or misunderstanding can turn relatively simple programs into something unnecessarily complex. But when used correctly and adapted perfectly to equally suitable situations, recursion becomes an excellent ally in solving problems that would otherwise be much more laborious and time-consuming. The materials presented here are intended for educational purposes only. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=nfekgrjxdktlwizwrrcrdyapacvdhrpv&ssn=1769180032824991536&ssn_dr=0&ssn_sr=0&fv_date=1769180032&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18786&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Trading%20Tools%20(Part%204)%3A%20Improving%20the%20Multi-Timeframe%20Scanner%20Dashboard%20with%20Dynamic%20Positioning%20and%20Toggle%20Features%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918003275794021&fz_uniq=5068773592516197760&sv=2552)

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