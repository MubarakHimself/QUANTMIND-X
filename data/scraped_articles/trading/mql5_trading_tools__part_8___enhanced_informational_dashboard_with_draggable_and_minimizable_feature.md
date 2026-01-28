---
title: MQL5 Trading Tools (Part 8): Enhanced Informational Dashboard with Draggable and Minimizable Features
url: https://www.mql5.com/en/articles/19059
categories: Trading, Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T17:53:34.318812
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/19059&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068766754928262508)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 7)](https://www.mql5.com/en/articles/18986), we developed an informational dashboard in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) to monitor multi-symbol positions and account metrics, such as "Balance", "Equity", and "Free Margin", with sortable columns and [Comma Separated Values](https://en.wikipedia.org/wiki/Comma-separated_values "https://en.wikipedia.org/wiki/Comma-separated_values") (CSV) export capabilities. In Part 8, we enhance this dashboard by adding draggable and minimizable features, interactive buttons for closing, toggling, and exporting, as well as hover effects for a more dynamic user experience. This enhancement retains real-time position tracking and a header glow effect. We will cover the following topics:

1. [Understanding the Enhanced Dashboard Architecture](https://www.mql5.com/en/articles/19059#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/19059#para2)
3. [Backtesting](https://www.mql5.com/en/articles/19059#para3)
4. [Conclusion](https://www.mql5.com/en/articles/19059#para4)

By the end, you’ll have a versatile, user-friendly MQL5 dashboard tailored for efficient trading oversight—let’s get started!

### Understanding the Enhanced Dashboard Architecture

We're upgrading the [Informational Dashboard](https://www.mql5.com/en/articles/18986) from Part 7 by adding draggable and minimizable features, along with interactive buttons and hover effects, to make it more flexible and user-friendly for managing multiple positions. These upgrades are relevant because they will allow the dashboard to be moved anywhere on the chart, minimizing clutter during analysis, while the minimize option will save screen space, and the interactive elements will provide immediate visual feedback, improving the overall trading experience in fast-paced environments.

We will achieve this by incorporating mouse event handling for dragging and button clicks, ensuring the dashboard remains responsive and adaptable without losing its core position tracking capabilities. We will also add an icon in the header for the export feature, so it is easy to find, but still keep the keyboard key feature. We plan to maintain the sortable grid and real-time updates while adding these enhancements, creating a tool that feels intuitive and efficient for daily use. Have a look below at what we aim to achieve, and then we can proceed to the implementation!

![IMPLEMENTATION PLAN](https://c.mql5.com/2/161/Screenshot_2025-08-04_204234.png)

### Implementation in MQL5

To enhance the program in [MQL5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5"), we will need to define the new dashboard components, typically 4 of them.

```
//--- existing components

#define HEADER_PANEL_TEXT   "HEADER_PANEL_TEXT"//--- Header title label
#define CLOSE_BUTTON        "BUTTON_CLOSE"     //--- Close button identifier
#define EXPORT_BUTTON       "BUTTON_EXPORT"    //--- Export button identifier
#define TOGGLE_BUTTON       "BUTTON_TOGGLE"    //--- Toggle (minimize/maximize) button

//--- the rest of the components
```

We start by adding new [defines](https://www.mql5.com/en/docs/basis/preprosessor/constant) to support the enhanced features of the informational dashboard, introducing identifiers for interactive [User Interface](https://en.wikipedia.org/wiki/User_interface "https://en.wikipedia.org/wiki/User_interface") (UI) elements. We define "HEADER\_PANEL\_TEXT" as "HEADER\_PANEL\_TEXT" for the dashboard’s title label, providing a clear visual header. The "CLOSE\_BUTTON" is defined as "BUTTON\_CLOSE", which creates an identifier for a button to close the dashboard, allowing us to remove it from the chart. The "EXPORT\_BUTTON" is defined as "BUTTON\_EXPORT" and sets up a button for triggering CSV export, enhancing data accessibility. The "TOGGLE\_BUTTON" is defined as "BUTTON\_TOGGLE", enabling a button to minimize or maximize the dashboard, improving screen space management.

These definitions ensure organized naming for the new interactive components, supporting the draggable and minimizable upgrades. The next thing we change is the color of the header shades by replacing them with clearly defined MQL5 constants.

```
// Dashboard settings
struct DashboardSettings {                     //--- Structure for dashboard settings
   int      panel_x;                           //--- X-coordinate of panel
   int      panel_y;                           //--- Y-coordinate of panel
   int      row_height;                        //--- Height of each row
   int      font_size;                         //--- Font size for labels
   string   font;                              //--- Font type for labels
   color    bg_color;                          //--- Background color of main panel
   color    border_color;                      //--- Border color of panels
   color    header_color;                      //--- Default color for header text
   color    text_color;                        //--- Default color for text
   color    section_bg_color;                  //--- Background color for header/footer panels
   int      zorder_panel;                      //--- Z-order for main panel
   int      zorder_subpanel;                   //--- Z-order for sub-panels
   int      zorder_labels;                     //--- Z-order for labels
   int      label_y_offset;                    //--- Y-offset for label positioning
   int      label_x_offset;                    //--- X-offset for label positioning
   int      header_x_distances[9];             //--- X-distances for header labels (9 columns)
   color    header_shades[12];                 //--- Array of header color shades for glow effect
} settings = {                                 //--- Initialize settings with default values
   20,                                         //--- panel_x
   20,                                         //--- panel_y
   24,                                         //--- row_height
   11,                                         //--- font_size
   "Calibri Bold",                             //--- font
   C'240,240,240',                             //--- bg_color (light gray)
   clrBlack,                                   //--- border_color
   C'0,50,70',                                 //--- header_color (dark teal)
   clrBlack,                                   //--- text_color
   C'200,220,230',                             //--- section_bg_color (light blue-gray)
   100,                                        //--- zorder_panel
   101,                                        //--- zorder_subpanel
   102,                                        //--- zorder_labels
   3,                                          //--- label_y_offset
   25,                                         //--- label_x_offset
   {10, 120, 170, 220, 280, 330, 400, 470, 530}, //--- header_x_distances
   {clrBlack, clrRed, clrBlue, clrGreen, clrMagenta, clrDarkOrchid,
    clrDeepPink, clrSkyBlue, clrDodgerBlue, clrDarkViolet, clrOrange, clrCrimson} //--- header_shades
};

//--- the previous one was as below
/*
   //---
   {C'0,0,0', C'255,0,0', C'0,255,0', C'0,0,255', C'255,255,0', C'0,255,255',
    C'255,0,255', C'255,255,255', C'255,0,255', C'0,255,255', C'255,255,0', C'0,0,255'}
*/
```

Here, we enhance the "DashboardSettings" [structure](https://www.mql5.com/en/docs/basis/types/classes) by updating the "header\_shades" array to improve the header glow effect for a more visually appealing experience. Previously, "header\_shades" used a mix of basic [RGB colors](https://en.wikipedia.org/wiki/RGB_color_model "https://en.wikipedia.org/wiki/RGB_color_model") (e.g., pure black, red, green, blue, yellow, cyan, magenta, white) for the glow cycle. We now define "header\_shades" with a curated set of 12 colors: " [clrBlack](https://www.mql5.com/en/docs/constants/objectconstants/webcolors)", "clrRed", "clrBlue", "clrGreen", "clrMagenta", "clrDarkOrchid", "clrDeepPink", "clrSkyBlue", "clrDodgerBlue", "clrDarkViolet", "clrOrange", and "clrCrimson".

This upgrade will provide a richer, more varied palette that cycles through vibrant and nuanced shades, enhancing the dashboard’s aesthetic while maintaining the glow effect’s functionality for highlighting headers. Finally, we add more [global variables](https://www.mql5.com/en/docs/basis/variables/global) to help us have a dynamic dashboard, to take care of the hover and drag states.

```
//--- added global variables

int          prev_num_symbols = 0;             //--- Previous number of active symbols
bool         panel_is_visible = true;          //--- Flag to control panel visibility
bool         panel_minimized = false;          //--- Flag to control minimized state
bool         panel_dragging = false;           //--- Flag to track if panel is being dragged
int          panel_drag_x = 0;                 //--- Mouse x-coordinate when drag starts
int          panel_drag_y = 0;                 //--- Mouse y-coordinate when drag starts
int          panel_start_x = 0;                //--- Panel x-coordinate when drag starts
int          panel_start_y = 0;                //--- Panel y-coordinate when drag starts
int          prev_mouse_state = 0;             //--- Previous mouse state
bool         header_hovered = false;           //--- Header hover state
bool         toggle_hovered = false;           //--- Toggle button hover state
bool         close_hovered = false;            //--- Close button hover state
bool         export_hovered = false;           //--- Export button hover state
int          last_mouse_x = 0;                 //--- Last mouse x position
int          last_mouse_y = 0;                 //--- Last mouse y position
bool         prev_header_hovered = false;      //--- Previous header hover state
bool         prev_toggle_hovered = false;      //--- Previous toggle hover state
bool         prev_close_hovered = false;       //--- Previous close button hover state
bool         prev_export_hovered = false;      //--- Previous export button hover state
```

Finally, we introduce additional [global variables](https://www.mql5.com/en/docs/basis/variables/global) to support the enhanced interactivity and draggable features. We define "prev\_num\_symbols" as 0 to track the previous number of active symbols for dynamic resizing, "panel\_is\_visible" as true to control dashboard visibility, and "panel\_minimized" as false to manage the minimized state. To enable dragging, we add "panel\_dragging" as false to track drag status, "panel\_drag\_x" and "panel\_drag\_y" as 0 for mouse coordinates at drag start, and "panel\_start\_x" and "panel\_start\_y" as 0 for panel coordinates at drag start.

We include "prev\_mouse\_state" as 0 to monitor mouse click states, and for hover effects, we define "header\_hovered", "toggle\_hovered", "close\_hovered", and "export\_hovered" as false to track hover states for the header and buttons, with "last\_mouse\_x" and "last\_mouse\_y" as 0 to store the last mouse position, and "prev\_header\_hovered", "prev\_toggle\_hovered", "prev\_close\_hovered", and "prev\_export\_hovered" as false to detect hover state changes.

These variables will enable dynamic UI interactions like dragging, minimizing, and hover feedback. Since we now have the updated variables, let us update the functions as well to standardize object creation, since this is taking an advanced way for modularization. Let us start with the function to create a label and add a [tooltip](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/cchartobject/cchartobjecttooltip) feature.

```
//+------------------------------------------------------------------+
//| Creating label object                                            |
//+------------------------------------------------------------------+
bool createLABEL(string objName, string txt, int xD, int yD, color clrTxt, int fontSize, string font, int anchor, string tooltip = "", bool selectable = false) {
   if(!ObjectCreate(0, objName, OBJ_LABEL, 0, 0, 0)) { //--- Creating label object
      Print(__FUNCTION__, ": Failed to create label '", objName, "'. Error code = ", GetLastError()); //--- Logging creation failure
      return(false);                                   //--- Returning failure
   }
   ObjectSetInteger(0, objName, OBJPROP_XDISTANCE, xD);                  //--- Setting x-coordinate
   ObjectSetInteger(0, objName, OBJPROP_YDISTANCE, yD);                  //--- Setting y-coordinate
   ObjectSetInteger(0, objName, OBJPROP_CORNER, CORNER_LEFT_UPPER);      //--- Setting corner alignment
   ObjectSetString(0, objName, OBJPROP_TEXT, txt);                       //--- Setting text content
   ObjectSetInteger(0, objName, OBJPROP_FONTSIZE, fontSize);             //--- Setting font size
   ObjectSetString(0, objName, OBJPROP_FONT, font);                      //--- Setting font type
   ObjectSetInteger(0, objName, OBJPROP_COLOR, clrTxt);                  //--- Setting text color
   ObjectSetInteger(0, objName, OBJPROP_BACK, false);                    //--- Setting to foreground
   ObjectSetInteger(0, objName, OBJPROP_STATE, selectable);              //--- Setting selectable state
   ObjectSetInteger(0, objName, OBJPROP_SELECTABLE, selectable);         //--- Setting selectability
   ObjectSetInteger(0, objName, OBJPROP_SELECTED, false);                //--- Setting not selected
   ObjectSetInteger(0, objName, OBJPROP_ANCHOR, anchor);                 //--- Setting anchor point
   ObjectSetInteger(0, objName, OBJPROP_ZORDER, settings.zorder_labels); //--- Setting z-order

   //--- added this tooltip feature
   ObjectSetString(0, objName, OBJPROP_TOOLTIP, tooltip == "" ? (selectable ? "Click to sort" : "Position data") : tooltip); //--- Setting tooltip text
   //---
   //--- the existing was a hardcoded to this
   // ObjectSetString(0, objName, OBJPROP_TOOLTIP, selectable ? "Click to sort" : "Position data"); //--- Set tooltip
   //---

   ChartRedraw(0);                                    //--- Redrawing chart
   return(true);                                      //--- Returning success
}
```

For the "createLABEL" function, we improve the [tooltip](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/cchartobject/cchartobjecttooltip) logic to make it more flexible and reusable for various UI elements. Previously, the tooltip was hardcoded with [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) setting [OBJPROP\_TOOLTIP](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_string) to either "Click to sort" for selectable labels or "Position data" for non-selectable ones, limiting customization. We now modify this by adding a "tooltip" parameter with a default empty string, and use a ternary condition in "ObjectSetString" for "OBJPROP\_TOOLTIP": if "tooltip" is empty, it defaults to "Click to sort" for selectable labels or "Position data" for others; otherwise, it uses the provided "tooltip" value. This change will allow specific tooltips for elements like buttons (e.g., "Minimize dashboard" or "Close dashboard") while maintaining defaults for headers and data labels, improving user guidance and interaction clarity.

Then, to standardize panel creation, we will replace the inline object creation calls in the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler with a function for easier maintenance. Here is the logic we adopt for that.

```
//+------------------------------------------------------------------+
//| Creating rectangle object                                        |
//+------------------------------------------------------------------+
bool createRectangle(string object_name, int x_distance, int y_distance, int x_size, int y_size,
                     color background_color, color border_color = clrBlack) {
   if(!ObjectCreate(0, object_name, OBJ_RECTANGLE_LABEL, 0, 0, 0)) { //--- Creating rectangle object
      Print(__FUNCTION__, ": Failed to create Rectangle: '", object_name, "'. Error code = ", GetLastError()); //--- Logging creation failure
      return(false);                                  //--- Returning failure
   }
   ObjectSetInteger(0, object_name, OBJPROP_XDISTANCE, x_distance);            //--- Setting x-coordinate
   ObjectSetInteger(0, object_name, OBJPROP_YDISTANCE, y_distance);            //--- Setting y-coordinate
   ObjectSetInteger(0, object_name, OBJPROP_XSIZE, x_size);                    //--- Setting width
   ObjectSetInteger(0, object_name, OBJPROP_YSIZE, y_size);                    //--- Setting height
   ObjectSetInteger(0, object_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);        //--- Setting corner alignment
   ObjectSetInteger(0, object_name, OBJPROP_BGCOLOR, background_color);        //--- Setting background color
   ObjectSetInteger(0, object_name, OBJPROP_BORDER_COLOR, border_color);       //--- Setting border color
   ObjectSetInteger(0, object_name, OBJPROP_BORDER_TYPE, BORDER_FLAT);         //--- Setting border type
   ObjectSetInteger(0, object_name, OBJPROP_BACK, false);                      //--- Setting to foreground
   ObjectSetInteger(0, object_name, OBJPROP_ZORDER, settings.zorder_subpanel); //--- Setting z-order
   return(true);                                                               //--- Returning success
}
```

Here, we just create a boolean "createRectangle" function and use a similar structure for definition as we did with the labels. This is not new to you, so we will just save time and move on to the next update, which is a minor fix in the counting functions to change the loop direction.

```
//+------------------------------------------------------------------+
//| Counting total positions for a symbol                            |
//+------------------------------------------------------------------+
string countPositionsTotal(string symbol) {
   int totalPositions = 0;                               //--- Initializing position counter
   int count_Total_Pos = PositionsTotal();               //--- Getting total positions
   for(int i = count_Total_Pos - 1; i >= 0; i--) {       //--- Iterating through positions
      ulong ticket = PositionGetTicket(i);               //--- Getting position ticket
      if(ticket > 0 && PositionSelectByTicket(ticket)) { //--- Checking if position selected
         if(PositionGetString(POSITION_SYMBOL) == symbol && (MagicNumber < 0 || PositionGetInteger(POSITION_MAGIC) == MagicNumber)) totalPositions++; //--- Checking symbol and magic
      }
   }
   return IntegerToString(totalPositions);               //--- Returning total as string
}

//+------------------------------------------------------------------+
//| Counting buy or sell positions for a symbol                      |
//+------------------------------------------------------------------+
string countPositions(string symbol, ENUM_POSITION_TYPE pos_type) {
   int totalPositions = 0;                               //--- Initializing position counter
   int count_Total_Pos = PositionsTotal();               //--- Getting total positions
   for(int i = count_Total_Pos - 1; i >= 0; i--) {       //--- Iterating through positions
      ulong ticket = PositionGetTicket(i);               //--- Getting position ticket
      if(ticket > 0 && PositionSelectByTicket(ticket)) { //--- Checking if position selected
         if(PositionGetString(POSITION_SYMBOL) == symbol && PositionGetInteger(POSITION_TYPE) == pos_type && (MagicNumber < 0 || PositionGetInteger(POSITION_MAGIC) == MagicNumber)) { //--- Checking symbol, type, magic
            totalPositions++;                            //--- Incrementing counter
         }
      }
   }
   return IntegerToString(totalPositions);               //--- Returning total as string
}

//+------------------------------------------------------------------+
//| Counting pending orders for a symbol                             |
//+------------------------------------------------------------------+
string countOrders(string symbol) {
   int total = 0;                                        //--- Initializing counter
   int tot = OrdersTotal();                              //--- Getting total orders
   for(int i = tot - 1; i >= 0; i--) {                   //--- Iterating through orders
      ulong ticket = OrderGetTicket(i);                  //--- Getting order ticket
      if(ticket > 0 && OrderSelect(ticket)) {            //--- Checking if order selected
         if(OrderGetString(ORDER_SYMBOL) == symbol && (MagicNumber < 0 || OrderGetInteger(ORDER_MAGIC) == MagicNumber)) total++; //--- Checking symbol and magic
      }
   }
   return IntegerToString(total);                        //--- Returning total as string
}

//+------------------------------------------------------------------+
//| Summing double property for positions of a symbol                |
//+------------------------------------------------------------------+
string sumPositionDouble(string symbol, ENUM_POSITION_PROPERTY_DOUBLE prop) {
   double total = 0.0;                                   //--- Initializing total
   int count_Total_Pos = PositionsTotal();               //--- Getting total positions
   for(int i = count_Total_Pos - 1; i >= 0; i--) {       //--- Iterating through positions
      ulong ticket = PositionGetTicket(i);               //--- Getting position ticket
      if(ticket > 0 && PositionSelectByTicket(ticket)) { //--- Checking if position selected
         if(PositionGetString(POSITION_SYMBOL) == symbol && (MagicNumber < 0 || PositionGetInteger(POSITION_MAGIC) == MagicNumber)) { //--- Checking symbol and magic
            total += PositionGetDouble(prop);            //--- Adding property value
         }
      }
   }
   return DoubleToString(total, 2);                      //--- Returning total as string
}

//+------------------------------------------------------------------+
//| Summing commission for positions of a symbol from history        |
//+------------------------------------------------------------------+
double sumPositionCommission(string symbol) {
   double total_comm = 0.0;                             //--- Initializing total commission
   int pos_total = PositionsTotal();                    //--- Getting total positions
   for(int p = 0; p < pos_total; p++) {                 //--- Iterating through positions
      ulong ticket = PositionGetTicket(p);              //--- Getting position ticket
      if(ticket > 0 && PositionSelectByTicket(ticket)) { //--- Checking if selected
         if(PositionGetString(POSITION_SYMBOL) == symbol && (MagicNumber < 0 || PositionGetInteger(POSITION_MAGIC) == MagicNumber)) { //--- Checking symbol and magic
            long pos_id = PositionGetInteger(POSITION_IDENTIFIER); //--- Getting position ID
            if(HistorySelectByPosition(pos_id)) {       //--- Selecting history by position
               int deals_total = HistoryDealsTotal();   //--- Getting total deals
               for(int d = 0; d < deals_total; d++) {   //--- Iterating through deals
                  ulong deal_ticket = HistoryDealGetTicket(d); //--- Getting deal ticket
                  if(deal_ticket > 0) {                 //--- Checking valid
                     total_comm += HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION); //--- Adding commission
                  }
               }
            }
         }
      }
   }
   return total_comm;                                  //--- Returning total commission
}

//+------------------------------------------------------------------+
//| Collecting active symbols with positions or orders               |
//+------------------------------------------------------------------+
void CollectActiveSymbols() {
   string symbols_temp[];                             //--- Temporary array for symbols
   int added = 0;                                     //--- Counter for added symbols
   // Collecting from positions
   int pos_total = PositionsTotal();                  //--- Getting total positions
   for(int i = pos_total - 1; i >= 0; i--) {          //--- Iterating through positions
      ulong ticket = PositionGetTicket(i);            //--- Getting position ticket
      if(ticket > 0 && PositionSelectByTicket(ticket)) { //--- Checking if position selected
         if(MagicNumber < 0 || PositionGetInteger(POSITION_MAGIC) == MagicNumber) { //--- Checking magic number
            string sym = PositionGetString(POSITION_SYMBOL); //--- Getting symbol
            bool found = false;                       //--- Flag for symbol found
            for(int k = 0; k < added; k++) {          //--- Checking existing symbols
               if(symbols_temp[k] == sym) {           //--- Symbol already added
                  found = true;                       //--- Setting found flag
                  break;                              //--- Exiting loop
               }
            }
            if(!found) {                              //--- If not found
               ArrayResize(symbols_temp, added + 1);  //--- Resizing array
               symbols_temp[added] = sym;             //--- Adding symbol
               added++;                               //--- Incrementing counter
            }
         }
      }
   }
   // Collecting from orders
   int ord_total = OrdersTotal();                     //--- Getting total orders
   for(int i = ord_total - 1; i >= 0; i--) {          //--- Iterating through orders
      ulong ticket = OrderGetTicket(i);               //--- Getting order ticket
      if(ticket > 0 && OrderSelect(ticket)) {         //--- Checking if order selected
         if(MagicNumber < 0 || OrderGetInteger(ORDER_MAGIC) == MagicNumber) { //--- Checking magic number
            string sym = OrderGetString(ORDER_SYMBOL); //--- Getting symbol
            bool found = false;                       //--- Flag for symbol found
            for(int k = 0; k < added; k++) {          //--- Checking existing symbols
               if(symbols_temp[k] == sym) {           //--- Symbol already added
                  found = true;                       //--- Setting found flag
                  break;                              //--- Exiting loop
               }
            }
            if(!found) {                             //--- If not found
               ArrayResize(symbols_temp, added + 1); //--- Resizing array
               symbols_temp[added] = sym;            //--- Adding symbol
               added++;                              //--- Incrementing counter
            }
         }
      }
   }
   // Setting symbol_data
   ArrayResize(symbol_data, added);                   //--- Resizing symbol data array
   for(int i = 0; i < added; i++) {                   //--- Iterating through added symbols
      symbol_data[i].name = symbols_temp[i];          //--- Setting symbol name
      symbol_data[i].buys = 0;                        //--- Initializing buys
      symbol_data[i].sells = 0;                       //--- Initializing sells
      symbol_data[i].trades = 0;                      //--- Initializing trades
      symbol_data[i].lots = 0.0;                      //--- Initializing lots
      symbol_data[i].profit = 0.0;                    //--- Initializing profit
      symbol_data[i].pending = 0;                     //--- Initializing pending
      symbol_data[i].swaps = 0.0;                     //--- Initializing swaps
      symbol_data[i].comm = 0.0;                      //--- Initializing commission
      symbol_data[i].buys_str = "0";                  //--- Initializing buys string
      symbol_data[i].sells_str = "0";                 //--- Initializing sells string
      symbol_data[i].trades_str = "0";                //--- Initializing trades string
      symbol_data[i].lots_str = "0.00";               //--- Initializing lots string
      symbol_data[i].profit_str = "0.00";             //--- Initializing profit string
      symbol_data[i].pending_str = "0";               //--- Initializing pending string
      symbol_data[i].swaps_str = "0.00";              //--- Initializing swaps string
      symbol_data[i].comm_str = "0.00";               //--- Initializing commission string
   }
}
```

To strengthen the counting functions' logic, we change the loop direction from an incremental one to a decremental one, which is typically safer in MQL5, but you can choose to keep the original. However, we added ticket checks before selecting orders and positions to prevent potential errors with invalid tickets or list modifications during iterations. Since we want full dynamicity, let us move the initial dashboard creation into a function and subdivide it into two, one for maximized and the other for minimized creation.

```
//+------------------------------------------------------------------+
//| Creating full dashboard UI                                       |
//+------------------------------------------------------------------+
void createFullDashboard() {
   CollectActiveSymbols();                            //--- Collecting active symbols
   int num_rows = ArraySize(symbol_data);             //--- Getting number of rows
   int num_columns = ArraySize(headers);              //--- Getting number of columns
   int column_width_sum = 0;                          //--- Initializing column width sum
   for(int i = 0; i < num_columns; i++)               //--- Iterating through columns
      column_width_sum += column_widths[i];           //--- Adding column width
   int panel_width = MathMax(settings.header_x_distances[num_columns - 1] + column_widths[num_columns - 1], column_width_sum) + 20 + settings.label_x_offset; //--- Calculating panel width
   // Creating main panel
   string panel_name = PREFIX + PANEL;                //--- Defining main panel name
   createRectangle(panel_name, settings.panel_x, settings.panel_y, panel_width, (num_rows + 3) * settings.row_height, settings.bg_color, settings.border_color); //--- Creating main panel
   // Creating header panel
   string header_panel = PREFIX + HEADER_PANEL;       //--- Defining header panel name
   createRectangle(header_panel, settings.panel_x, settings.panel_y, panel_width, settings.row_height, settings.section_bg_color, settings.border_color); //--- Creating header panel
   // Creating header title
   createLABEL(PREFIX + HEADER_PANEL_TEXT, "Trading Dashboard", settings.panel_x + 10, settings.panel_y + 8 + settings.label_y_offset, clrBlack, 14, settings.font, ANCHOR_LEFT, "Dashboard Title"); //--- Creating header title
   // Creating export button
   createLABEL(PREFIX + EXPORT_BUTTON, CharToString(60), settings.panel_x + panel_width - 90, settings.panel_y + 12, clrBlack, 18, "Wingdings", ANCHOR_CENTER, "Click to export or press 'E' key to export", true); //--- Creating export button
   // Creating toggle button
   createLABEL(PREFIX + TOGGLE_BUTTON, CharToString('r'), settings.panel_x + panel_width - 60, settings.panel_y + 12, clrBlack, 18, "Wingdings", ANCHOR_CENTER, "Minimize dashboard", true); //--- Creating toggle button
   // Creating close button
   createLABEL(PREFIX + CLOSE_BUTTON, CharToString('r'), settings.panel_x + panel_width - 30, settings.panel_y + 12, clrBlack, 18, "Webdings", ANCHOR_CENTER, "Close dashboard", true); //--- Creating close button
   // Creating headers
   int header_y = settings.panel_y + settings.row_height + 8 + settings.label_y_offset; //--- Calculating header y-coordinate
   for(int i = 0; i < num_columns; i++) {             //--- Iterating through headers
      string header_name = PREFIX + HEADER + IntegerToString(i); //--- Defining header label name
      int header_x = settings.panel_x + settings.header_x_distances[i] + settings.label_x_offset; //--- Calculating header x-coordinate
      createLABEL(header_name, headers[i], header_x, header_y, settings.header_color, 12, settings.font, ANCHOR_LEFT, "Click to sort", true); //--- Creating header label
   }
   // Creating symbol and data labels
   int first_row_y = header_y + settings.row_height;  //--- Calculating first row y-coordinate
   int symbol_x = settings.panel_x + 10 + settings.label_x_offset; //--- Setting symbol x-coordinate
   for(int i = 0; i < num_rows; i++) {                //--- Iterating through rows
      string symbol_name = PREFIX + SYMB + IntegerToString(i); //--- Defining symbol label name
      createLABEL(symbol_name, symbol_data[i].name, symbol_x, first_row_y + i * settings.row_height + settings.label_y_offset, settings.text_color, settings.font_size, settings.font, ANCHOR_LEFT, "Symbol name"); //--- Creating symbol label
      int x_offset = settings.panel_x + 10 + column_widths[0] + settings.label_x_offset; //--- Setting data x-offset
      for(int j = 0; j < num_columns - 1; j++) {      //--- Iterating through data columns
         string data_name = PREFIX + DATA + IntegerToString(i) + "_" + IntegerToString(j); //--- Defining data label name
         color initial_color = data_default_colors[j]; //--- Setting initial color
         string initial_txt = (j <= 2 || j == 5) ? "0" : "0.00"; //--- Setting initial text
         createLABEL(data_name, initial_txt, x_offset, first_row_y + i * settings.row_height + settings.label_y_offset, initial_color, settings.font_size, settings.font, ANCHOR_RIGHT, "Data value"); //--- Creating data label
         x_offset += column_widths[j + 1];            //--- Updating x-offset
      }
   }
   // Creating footer panel
   int footer_y = settings.panel_y + (num_rows + 3) * settings.row_height - settings.row_height; //--- Calculating footer y-coordinate
   string footer_panel = PREFIX + FOOTER_PANEL;       //--- Defining footer panel name
   createRectangle(footer_panel, settings.panel_x, footer_y, panel_width, settings.row_height, settings.section_bg_color, settings.border_color); //--- Creating footer panel
   // Creating footer text and data
   int footer_text_x = settings.panel_x + 10 + settings.label_x_offset; //--- Setting footer text x-coordinate
   createLABEL(PREFIX + FOOTER_TEXT, "Total:", footer_text_x, footer_y + 8 + settings.label_y_offset, settings.text_color, settings.font_size, settings.font, ANCHOR_LEFT, "Totals"); //--- Creating footer text label
   int x_offset = settings.panel_x + 10 + column_widths[0] + settings.label_x_offset; //--- Setting footer data x-offset
   for(int j = 0; j < num_columns - 1; j++) {         //--- Iterating through footer data
      string footer_data_name = PREFIX + FOOTER_DATA + IntegerToString(j); //--- Defining footer data label name
      color footer_color = data_default_colors[j];    //--- Setting footer data color
      string initial_txt = (j <= 2 || j == 5) ? "0" : "0.00"; //--- Setting initial text
      createLABEL(footer_data_name, initial_txt, x_offset, footer_y + 8 + settings.label_y_offset, footer_color, settings.font_size, settings.font, ANCHOR_RIGHT, "Total value"); //--- Creating footer data label
      x_offset += column_widths[j + 1];               //--- Updating x-offset
   }
   // Creating account panel
   int account_panel_y = footer_y + settings.row_height + 5; //--- Calculating account panel y-coordinate
   string account_panel_name = PREFIX + ACCOUNT_PANEL; //--- Defining account panel name
   createRectangle(account_panel_name, settings.panel_x, account_panel_y, panel_width, settings.row_height, settings.section_bg_color, settings.border_color); //--- Creating account panel
   // Creating account text and data labels
   int acc_x = settings.panel_x + 10 + settings.label_x_offset; //--- Setting account label x-coordinate
   int acc_data_offset = 160;                         //--- Setting data offset
   int acc_spacing = (panel_width - 45) / ArraySize(account_items); //--- Calculating spacing
   for(int k = 0; k < ArraySize(account_items); k++) { //--- Iterating through account items
      string acc_text_name = PREFIX + ACC_TEXT + IntegerToString(k); //--- Defining account text label name
      int text_x = acc_x + k * acc_spacing;          //--- Calculating text x-coordinate
      createLABEL(acc_text_name, account_items[k] + ":", text_x, account_panel_y + 8 + settings.label_y_offset, settings.text_color, settings.font_size, settings.font, ANCHOR_LEFT, "Account info"); //--- Creating account text label
      string acc_data_name = PREFIX + ACC_DATA + IntegerToString(k); //--- Defining account data label name
      int data_x = text_x + acc_data_offset;         //--- Calculating data x-coordinate
      createLABEL(acc_data_name, "0.00", data_x, account_panel_y + 8 + settings.label_y_offset, settings.text_color, settings.font_size, settings.font, ANCHOR_RIGHT, "Account value"); //--- Creating account data label
   }
   ChartRedraw(0);                                    //--- Redrawing chart
}

//+------------------------------------------------------------------+
//| Creating minimized dashboard UI                                  |
//+------------------------------------------------------------------+
void createMinimizedDashboard() {
   int num_columns = ArraySize(headers);              //--- Getting number of columns
   int column_width_sum = 0;                          //--- Initializing column width sum
   for(int i = 0; i < num_columns; i++)               //--- Iterating through columns
      column_width_sum += column_widths[i];           //--- Adding column width
   int panel_width = MathMax(settings.header_x_distances[num_columns - 1] + column_widths[num_columns - 1], column_width_sum) + 20 + settings.label_x_offset; //--- Calculating panel width
   // Creating header panel
   createRectangle(PREFIX + HEADER_PANEL, settings.panel_x, settings.panel_y, panel_width, settings.row_height, settings.section_bg_color, settings.border_color); //--- Creating header panel
   // Creating header title
   createLABEL(PREFIX + HEADER_PANEL_TEXT, "Trading Dashboard", settings.panel_x + 10, settings.panel_y + 8 + settings.label_y_offset, clrBlack, 14, settings.font, ANCHOR_LEFT, "Dashboard Title"); //--- Creating header title
   // Creating export button
   createLABEL(PREFIX + EXPORT_BUTTON, CharToString(60), settings.panel_x + panel_width - 90, settings.panel_y + 12, clrBlack, 18, "Wingdings", ANCHOR_CENTER, "Click to export or press 'E' key to export", true); //--- Creating export button
   // Creating toggle button (maximize)
   createLABEL(PREFIX + TOGGLE_BUTTON, CharToString('o'), settings.panel_x + panel_width - 60, settings.panel_y + 12, clrBlack, 18, "Wingdings", ANCHOR_CENTER, "Maximize dashboard", true); //--- Creating toggle button
   // Creating close button
   createLABEL(PREFIX + CLOSE_BUTTON, CharToString('r'), settings.panel_x + panel_width - 30, settings.panel_y + 12, clrBlack, 18, "Webdings", ANCHOR_CENTER, "Close dashboard", true); //--- Creating close button
   ChartRedraw(0);                                    //--- Redrawing chart
}

//+------------------------------------------------------------------+
//| Deleting all dashboard objects                                   |
//+------------------------------------------------------------------+
void deleteAllObjects() {
   ObjectsDeleteAll(0, PREFIX, -1, -1);               //--- Deleting all objects with prefix
}
```

We implement the "createFullDashboard", "createMinimizedDashboard", and "deleteAllObjects" functions to manage the UI supporting full and minimized views with interactive elements. In "createFullDashboard", we call "CollectActiveSymbols" to populate "symbol\_data", calculate "num\_rows" and "num\_columns" with [ArraySize](https://www.mql5.com/en/docs/array/arraysize), and compute "panel\_width" using "column\_widths" and "settings.header\_x\_distances". We create the main panel with "createRectangle" for "PREFIX + PANEL", header panel with "PREFIX + HEADER\_PANEL", and header title with "createLABEL" for "PREFIX + HEADER\_PANEL\_TEXT" as "Trading Dashboard".

We add buttons with "createLABEL" for "PREFIX + EXPORT\_BUTTON" (Wingdings 60), "PREFIX + TOGGLE\_BUTTON" ( [Wingdings](https://en.wikipedia.org/wiki/Wingdings "https://en.wikipedia.org/wiki/Wingdings") 'r' for minimize), and "PREFIX + CLOSE\_BUTTON" (Webdings 'r'), all with specific tooltips and selectable true. The choice of the icon styles depends on you. Here is a compact view of what you could use for the fonts. Just use the accurate symbol and character type.

![SYMBOL FONTS](https://c.mql5.com/2/161/C_SYMBOL_FONTS.png)

Then, we create header labels for "headers" at calculated positions, symbol labels for "symbol\_data\[i\].name", data labels with initial values, footer panel with "PREFIX + FOOTER\_PANEL", footer text "Total:", footer data labels, account panel with "PREFIX + ACCOUNT\_PANEL", and account labels for "account\_items", all using "createRectangle" and "createLABEL" with appropriate coordinates and colors, followed by the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function.

In "createMinimizedDashboard", we create a compact UI with only the header panel using "createRectangle" for "PREFIX + HEADER\_PANEL", header title with "createLABEL" for "PREFIX + HEADER\_PANEL\_TEXT", and buttons for export (Wingdings 60), toggle (Wingdings 'o' for maximize), and close (Webdings 'r'), ensuring minimal screen usage, and redraw.

The "deleteAllObjects" function removes all dashboard objects with [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/ObjectDeleteAll) using "PREFIX" for all charts and types, ensuring a clean slate for UI updates or closure. These functions will enable a flexible dashboard with full and minimized states, supporting user interactions like dragging and toggling. We will now proceed to update the dashboard function using these dynamic functions.

```
//+------------------------------------------------------------------+
//| Updating dashboard data and visuals                              |
//+------------------------------------------------------------------+
void UpdateDashboard() {
   bool needs_redraw = false;                //--- Initializing redraw flag
   CollectActiveSymbols();                   //--- Collecting active symbols
   int current_num = ArraySize(symbol_data); //--- Getting current number of symbols
   if(current_num != prev_num_symbols) {     //--- Checking if symbol count changed
      deleteAllObjects();                    //--- Deleting all objects
      if(panel_minimized) {                  //--- Checking if minimized
         createMinimizedDashboard();         //--- Creating minimized dashboard
      } else {
         createFullDashboard();              //--- Creating full dashboard
      }
      prev_num_symbols = current_num;        //--- Updating previous symbol count
      needs_redraw = true;                   //--- Setting redraw flag
   }
   if(!panel_is_visible || panel_minimized) return; //--- Exiting if not visible or minimized
   // Resetting totals
   totalBuys = 0;     //--- Resetting total buys
   totalSells = 0;    //--- Resetting total sells
   totalTrades = 0;   //--- Resetting total trades
   totalLots = 0.0;   //--- Resetting total lots
   totalProfit = 0.0; //--- Resetting total profit
   totalPending = 0;  //--- Resetting total pending
   totalSwap = 0.0;   //--- Resetting total swap
   totalComm = 0.0;   //--- Resetting total commission
   // Calculating symbol data and totals
   for(int i = 0; i < current_num; i++) {           //--- Iterating through symbols
      string symbol = symbol_data[i].name;          //--- Getting symbol name
      for(int j = 0; j < 8; j++) {                  //--- Iterating through data columns
         string value = "";                         //--- Initializing value
         color data_color = data_default_colors[j]; //--- Setting default color
         double dval = 0.0;                         //--- Initializing double value
         int ival = 0;                              //--- Initializing integer value
         switch(j) {                                //--- Handling data type
            case 0:                                 // Buy positions
               value = countPositions(symbol, POSITION_TYPE_BUY); //--- Getting buy positions
               ival = (int)StringToInteger(value);  //--- Converting to integer
               if(value != symbol_data[i].buys_str) { //--- Checking if changed
                  symbol_data[i].buys_str = value;  //--- Updating buys string
                  symbol_data[i].buys = ival;       //--- Updating buys count
               }
               totalBuys += ival;                   //--- Adding to total buys
               break;
            case 1:                                 // Sell positions
               value = countPositions(symbol, POSITION_TYPE_SELL); //--- Getting sell positions
               ival = (int)StringToInteger(value);  //--- Converting to integer
               if(value != symbol_data[i].sells_str) { //--- Checking if changed
                  symbol_data[i].sells_str = value; //--- Updating sells string
                  symbol_data[i].sells = ival;      //--- Updating sells count
               }
               totalSells += ival; //--- Adding to total sells
               break;
            case 2: // Total trades
               value = countPositionsTotal(symbol); //--- Getting total trades
               ival = (int)StringToInteger(value);  //--- Converting to integer
               if(value != symbol_data[i].trades_str) { //--- Checking if changed
                  symbol_data[i].trades_str = value; //--- Updating trades string
                  symbol_data[i].trades = ival;     //--- Updating trades count
               }
               totalTrades += ival;                 //--- Adding to total trades
               break;
            case 3: // Lots
               value = sumPositionDouble(symbol, POSITION_VOLUME); //--- Getting total lots
               dval = StringToDouble(value);        //--- Converting to double
               if(value != symbol_data[i].lots_str) { //--- Checking if changed
                  symbol_data[i].lots_str = value; //--- Updating lots string
                  symbol_data[i].lots = dval;      //--- Updating lots value
               }
               totalLots += dval;                  //--- Adding to total lots
               break;
            case 4: // Profit
               value = sumPositionDouble(symbol, POSITION_PROFIT); //--- Getting total profit
               dval = StringToDouble(value);      //--- Converting to double
               data_color = (dval > 0) ? clrGreen : (dval < 0) ? clrRed : clrGray; //--- Setting color based on value
               if(value != symbol_data[i].profit_str) { //--- Checking if changed
                  symbol_data[i].profit_str = value; //--- Updating profit string
                  symbol_data[i].profit = dval;  //--- Updating profit value
               }
               totalProfit += dval;              //--- Adding to total profit
               break;
            case 5: // Pending
               value = countOrders(symbol);      //--- Getting pending orders
               ival = (int)StringToInteger(value); //--- Converting to integer
               if(value != symbol_data[i].pending_str) { //--- Checking if changed
                  symbol_data[i].pending_str = value; //--- Updating pending string
                  symbol_data[i].pending = ival; //--- Updating pending count
               }
               totalPending += ival;             //--- Adding to total pending
               break;
            case 6: // Swap
               value = sumPositionDouble(symbol, POSITION_SWAP); //--- Getting total swap
               dval = StringToDouble(value);     //--- Converting to double
               data_color = (dval > 0) ? clrGreen : (dval < 0) ? clrRed : clrPurple; //--- Setting color based on value
               if(value != symbol_data[i].swaps_str) { //--- Checking if changed
                  symbol_data[i].swaps_str = value;    //--- Updating swap string
                  symbol_data[i].swaps = dval;  //--- Updating swap value
               }
               totalSwap += dval; //--- Adding to total swap
               break;
            case 7: // Comm
               dval = sumPositionCommission(symbol); //--- Getting total commission
               value = DoubleToString(dval, 2);      //--- Formatting commission
               data_color = (dval > 0) ? clrGreen : (dval < 0) ? clrRed : clrBrown; //--- Setting color based on value
               if(value != symbol_data[i].comm_str) { //--- Checking if changed
                  symbol_data[i].comm_str = value;   //--- Updating commission string
                  symbol_data[i].comm = dval;        //--- Updating commission value
               }
               totalComm += dval;                    //--- Adding to total commission
               break;
         }
      }
   }
   // Sort after calculating values
   SortDashboard(); //--- Sorting dashboard data
   // Update header breathing effect
   glow_counter += MathMax(UpdateIntervalMs, 10); //--- Incrementing glow counter
   if(glow_counter >= GLOW_INTERVAL_MS) { //--- Checking if glow interval reached
      if(glow_direction) { //--- Checking if glowing forward
         glow_index++; //--- Incrementing glow index
         if(glow_index >= ArraySize(settings.header_shades) - 1) //--- Checking if at end
            glow_direction = false; //--- Reversing glow direction
      } else { //--- Glow backward
         glow_index--; //--- Decrementing glow index
         if(glow_index <= 0) //--- Checking if at start
            glow_direction = true; //--- Reversing glow direction
      }
      glow_counter = 0; //--- Resetting glow counter
   }
   color header_shade = settings.header_shades[glow_index]; //--- Getting current header shade
   for(int i = 0; i < ArraySize(headers); i++) { //--- Iterating through headers
      string header_name = PREFIX + HEADER + IntegerToString(i); //--- Defining header name
      ObjectSetInteger(0, header_name, OBJPROP_COLOR, header_shade); //--- Updating header color
      needs_redraw = true; //--- Setting redraw flag
   }
   // Update symbol and data labels
   bool labels_updated = false; //--- Initializing label update flag
   for(int i = 0; i < current_num; i++) { //--- Iterating through symbols
      string symbol = symbol_data[i].name; //--- Getting symbol name
      string symb_name = PREFIX + SYMB + IntegerToString(i); //--- Defining symbol label name
      string current_symb_txt = ObjectGetString(0, symb_name, OBJPROP_TEXT); //--- Getting current symbol text
      if(current_symb_txt != symbol) { //--- Checking if symbol changed
         ObjectSetString(0, symb_name, OBJPROP_TEXT, symbol); //--- Updating symbol text
         labels_updated = true; //--- Setting label updated flag
      }
      for(int j = 0; j < 8; j++) { //--- Iterating through data columns
         string data_name = PREFIX + DATA + IntegerToString(i) + "_" + IntegerToString(j); //--- Defining data label name
         string value; //--- Initializing value
         color data_color = data_default_colors[j]; //--- Setting default color
         switch(j) { //--- Handling data type
            case 0: //--- Buy positions
               value = symbol_data[i].buys_str; //--- Getting buys string
               data_color = clrRed; //--- Setting color to red
               break;
            case 1: //--- Sell positions
               value = symbol_data[i].sells_str; //--- Getting sells string
               data_color = clrGreen; //--- Setting color to green
               break;
            case 2: // Total trades
               value = symbol_data[i].trades_str;
               data_color = clrDarkGray;
               break;
            case 3: //--- Lots
               value = symbol_data[i].lots_str; //--- Getting lots string
               data_color = clrOrange; //--- Setting color to orange
               break;
            case 4: //--- Profit
               value = symbol_data[i].profit_str; //--- Getting profit string
               data_color = (symbol_data[i].profit > 0) ? clrGreen : (symbol_data[i].profit < 0) ? clrRed : clrGray; //--- Setting color based on profit
               break;
            case 5: //--- Pending
               value = symbol_data[i].pending_str; //--- Getting pending string
               data_color = clrBlue; //--- Setting color to blue
               break;
            case 6: //--- Swap
               value = symbol_data[i].swaps_str; //--- Getting swap string
               data_color = (symbol_data[i].swaps > 0) ? clrGreen : (symbol_data[i].swaps < 0) ? clrRed : clrPurple; //--- Setting color based on swap
               break;
            case 7: //--- Comm
               value = symbol_data[i].comm_str; //--- Getting commission string
               data_color = (symbol_data[i].comm > 0) ? clrGreen : (symbol_data[i].comm < 0) ? clrRed : clrBrown; //--- Setting color based on commission
               break;
         }
         if(updateLABEL(data_name, value, data_color)) labels_updated = true; //--- Updating label if changed
      }
   }
   if(labels_updated) needs_redraw = true; //--- Setting redraw flag if labels updated
   // Updating totals
   string new_total_buys = IntegerToString(totalBuys); //--- Formatting total buys
   if(new_total_buys != total_buys_str) { //--- Checking if changed
      total_buys_str = new_total_buys; //--- Updating buys string
      if(updateLABEL(PREFIX + FOOTER_DATA + "0", new_total_buys, clrRed)) needs_redraw = true; //--- Updating label
   }
   string new_total_sells = IntegerToString(totalSells); //--- Formatting total sells
   if(new_total_sells != total_sells_str) { //--- Checking if changed
      total_sells_str = new_total_sells; //--- Updating sells string
      if(updateLABEL(PREFIX + FOOTER_DATA + "1", new_total_sells, clrGreen)) needs_redraw = true; //--- Updating label
   }
   string new_total_trades = IntegerToString(totalTrades); //--- Formatting total trades
   if(new_total_trades != total_trades_str) { //--- Checking if changed
      total_trades_str = new_total_trades; //--- Updating trades string
      if(updateLABEL(PREFIX + FOOTER_DATA + "2", new_total_trades, clrDarkGray)) needs_redraw = true; //--- Updating label
   }
   string new_total_lots = DoubleToString(totalLots, 2); //--- Formatting total lots
   if(new_total_lots != total_lots_str) { //--- Checking if changed
      total_lots_str = new_total_lots; //--- Updating lots string
      if(updateLABEL(PREFIX + FOOTER_DATA + "3", new_total_lots, clrOrange)) needs_redraw = true; //--- Updating label
   }
   string new_total_profit = DoubleToString(totalProfit, 2); //--- Formatting total profit
   color total_profit_color = (totalProfit > 0) ? clrGreen : (totalProfit < 0) ? clrRed : clrGray; //--- Setting color based on profit
   if(new_total_profit != total_profit_str) { //--- Checking if changed
      total_profit_str = new_total_profit; //--- Updating profit string
      if(updateLABEL(PREFIX + FOOTER_DATA + "4", new_total_profit, total_profit_color)) needs_redraw = true; //--- Updating label
   }
   string new_total_pending = IntegerToString(totalPending); //--- Formatting total pending
   if(new_total_pending != total_pending_str) { //--- Checking if changed
      total_pending_str = new_total_pending; //--- Updating pending string
      if(updateLABEL(PREFIX + FOOTER_DATA + "5", new_total_pending, clrBlue)) needs_redraw = true; //--- Updating label
   }
   string new_total_swap = DoubleToString(totalSwap, 2); //--- Formatting total swap
   color total_swap_color = (totalSwap > 0) ? clrGreen : (totalSwap < 0) ? clrRed : clrPurple; //--- Setting color based on swap
   if(new_total_swap != total_swap_str) { //--- Checking if changed
      total_swap_str = new_total_swap; //--- Updating swap string
      if(updateLABEL(PREFIX + FOOTER_DATA + "6", new_total_swap, total_swap_color)) needs_redraw = true; //--- Updating label
   }
   string new_total_comm = DoubleToString(totalComm, 2); //--- Formatting total commission
   color total_comm_color = (totalComm > 0) ? clrGreen : (totalComm < 0) ? clrRed : clrBrown; //--- Setting color based on commission
   if(new_total_comm != total_comm_str) { //--- Checking if changed
      total_comm_str = new_total_comm; //--- Updating commission string
      if(updateLABEL(PREFIX + FOOTER_DATA + "7", new_total_comm, total_comm_color)) needs_redraw = true; //--- Updating label
   }
   // Updating account info
   double balance = AccountInfoDouble(ACCOUNT_BALANCE); //--- Getting account balance
   double equity = AccountInfoDouble(ACCOUNT_EQUITY); //--- Getting account equity
   double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE); //--- Getting free margin
   string new_bal = DoubleToString(balance, 2); //--- Formatting balance
   if(new_bal != acc_bal_str) { //--- Checking if changed
      acc_bal_str = new_bal; //--- Updating balance string
      if(updateLABEL(PREFIX + ACC_DATA + "0", new_bal, clrBlack)) needs_redraw = true; //--- Updating label
   }
   string new_eq = DoubleToString(equity, 2); //--- Formatting equity
   color eq_color = (equity > balance) ? clrGreen : (equity < balance) ? clrRed : clrBlack; //--- Setting color based on equity
   if(new_eq != acc_eq_str) { //--- Checking if changed
      acc_eq_str = new_eq; //--- Updating equity string
      if(updateLABEL(PREFIX + ACC_DATA + "1", new_eq, eq_color)) needs_redraw = true; //--- Updating label
   }
   string new_free = DoubleToString(free_margin, 2); //--- Formatting free margin
   if(new_free != acc_free_str) { //--- Checking if changed
      acc_free_str = new_free; //--- Updating free margin string
      if(updateLABEL(PREFIX + ACC_DATA + "2", new_free, clrBlack)) needs_redraw = true; //--- Updating label
   }
   if(needs_redraw) { //--- Checking if redraw needed
      ChartRedraw(0); //--- Redrawing chart
   }
}
```

To the "UpdateDashboard" function, we add calls to the "CollectActiveSymbols" at the start, so we always update the totals and balance fields. Then, when the symbol count changes, we call the "deleteAllObjects" function to destroy the dashboard and recreate it via the "createFullDashboard" or "createMinimizedDashboard" function. In the data calculation loop, we added color logic for profits/swap/commission, which was previously partial. We have highlighted the areas of change for easier identification and clarity. Finally, we can now call our logic on initialization to see the milestone achievement.

```
//+------------------------------------------------------------------+
//| Initializing expert                                              |
//+------------------------------------------------------------------+
int OnInit() {
   createFullDashboard();                             //--- Creating full dashboard
   prev_num_symbols = ArraySize(symbol_data);         //--- Setting initial symbol count
   ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, true);  //--- Enabling mouse move events
   EventSetMillisecondTimer(MathMax(UpdateIntervalMs, 10)); //--- Setting timer with minimum 10ms
   UpdateDashboard();                                 //--- Updating dashboard
   return(INIT_SUCCEEDED);                            //--- Returning success
}

//+------------------------------------------------------------------+
//| Deinitializing expert                                            |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   deleteAllObjects();                                //--- Deleting all objects
   ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, false); //--- Disabling mouse move events
   EventKillTimer();                                  //--- Stopping timer
}

//+------------------------------------------------------------------+
//| Handling timer for millisecond-based updates                     |
//+------------------------------------------------------------------+
void OnTimer() {
   if(panel_is_visible && !panel_minimized) {         //--- Checking if visible and not minimized
      UpdateDashboard();                              //--- Updating dashboard
   }
}
```

Here, we implement the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit), "OnDeinit", and "OnTimer" event handlers to manage the lifecycle and updates of the dashboard, enabling its interactive and dynamic functionality. In the "OnInit" function, we call "createFullDashboard" to build the complete UI, set "prev\_num\_symbols" to the size of "symbol\_data" using [ArraySize](https://www.mql5.com/en/docs/array/arraysize) to track initial symbols, enable mouse move events with [ChartSetInteger](https://www.mql5.com/en/docs/chart_operations/chartsetinteger) setting [CHART\_EVENT\_MOUSE\_MOVE](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer) to true for dragging and hover effects, set a timer with [EventSetMillisecondTimer](https://www.mql5.com/en/docs/eventfunctions/eventsetmillisecondtimer) using the maximum of "UpdateIntervalMs" and 10ms for periodic updates, and call "UpdateDashboard" to populate initial data, returning "INIT\_SUCCEEDED" for successful initialization.

The [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) function cleans up by calling "deleteAllObjects" to remove all dashboard objects with "PREFIX", disabling mouse move events with "ChartSetInteger" setting "CHART\_EVENT\_MOUSE\_MOVE" to false, and stopping the timer with [EventKillTimer](https://www.mql5.com/en/docs/eventfunctions/eventkilltimer) to free resources.

In the [OnTimer](https://www.mql5.com/en/docs/event_handlers/ontimer) function, we check if "panel\_is\_visible" is true and "panel\_minimized" is false, then call "UpdateDashboard" to refresh data only when the dashboard is fully visible, ensuring efficient updates without processing in minimized or hidden states. Upon compilation, we have the following outcome.

![INITIALIZATION FEATURES](https://c.mql5.com/2/161/Screenshot_2025-08-04_182311.png)

From the image, we can see that the new features pop up successfully. We will now move on to creating a function for updating the panel positions when we are dragging to avoid recreating the objects.

```
//+------------------------------------------------------------------+
//| Updating panel object positions                                  |
//+------------------------------------------------------------------+
void updatePanelPositions() {
   int num_rows = ArraySize(symbol_data);             //--- Getting number of rows
   int num_columns = ArraySize(headers);              //--- Getting number of columns
   int column_width_sum = 0;                          //--- Initializing column width sum
   for(int i = 0; i < num_columns; i++)               //--- Iterating through columns
      column_width_sum += column_widths[i];           //--- Adding column width
   int panel_width = MathMax(settings.header_x_distances[num_columns - 1] + column_widths[num_columns - 1], column_width_sum) + 20 + settings.label_x_offset; //--- Calculating panel width
   // Updating header panel and buttons
   ObjectSetInteger(0, PREFIX + HEADER_PANEL, OBJPROP_XDISTANCE, settings.panel_x); //--- Updating header panel x-coordinate
   ObjectSetInteger(0, PREFIX + HEADER_PANEL, OBJPROP_YDISTANCE, settings.panel_y); //--- Updating header panel y-coordinate
   ObjectSetInteger(0, PREFIX + HEADER_PANEL_TEXT, OBJPROP_XDISTANCE, settings.panel_x + 10); //--- Updating header text x-coordinate
   ObjectSetInteger(0, PREFIX + HEADER_PANEL_TEXT, OBJPROP_YDISTANCE, settings.panel_y + 8 + settings.label_y_offset); //--- Updating header text y-coordinate
   ObjectSetInteger(0, PREFIX + EXPORT_BUTTON, OBJPROP_XDISTANCE, settings.panel_x + panel_width - 90); //--- Updating export button x-coordinate
   ObjectSetInteger(0, PREFIX + EXPORT_BUTTON, OBJPROP_YDISTANCE, settings.panel_y + 12); //--- Updating export button y-coordinate
   ObjectSetInteger(0, PREFIX + TOGGLE_BUTTON, OBJPROP_XDISTANCE, settings.panel_x + panel_width - 60); //--- Updating toggle button x-coordinate
   ObjectSetInteger(0, PREFIX + TOGGLE_BUTTON, OBJPROP_YDISTANCE, settings.panel_y + 12); //--- Updating toggle button y-coordinate
   ObjectSetInteger(0, PREFIX + CLOSE_BUTTON, OBJPROP_XDISTANCE, settings.panel_x + panel_width - 30); //--- Updating close button x-coordinate
   ObjectSetInteger(0, PREFIX + CLOSE_BUTTON, OBJPROP_YDISTANCE, settings.panel_y + 12); //--- Updating close button y-coordinate
   if(!panel_minimized) {                                               //--- Checking if not minimized
      // Updating main panel
      ObjectSetInteger(0, PREFIX + PANEL, OBJPROP_XDISTANCE, settings.panel_x); //--- Updating main panel x-coordinate
      ObjectSetInteger(0, PREFIX + PANEL, OBJPROP_YDISTANCE, settings.panel_y); //--- Updating main panel y-coordinate
      // Updating headers
      int header_y = settings.panel_y + settings.row_height + 8 + settings.label_y_offset; //--- Calculating header y-coordinate
      for(int i = 0; i < num_columns; i++) {                            //--- Iterating through headers
         string header_name = PREFIX + HEADER + IntegerToString(i);     //--- Defining header name
         int header_x = settings.panel_x + settings.header_x_distances[i] + settings.label_x_offset; //--- Calculating header x-coordinate
         ObjectSetInteger(0, header_name, OBJPROP_XDISTANCE, header_x); //--- Updating header x-coordinate
         ObjectSetInteger(0, header_name, OBJPROP_YDISTANCE, header_y); //--- Updating header y-coordinate
      }
      // Updating symbol and data labels
      int first_row_y = header_y + settings.row_height;                 //--- Calculating first row y-coordinate
      int symbol_x = settings.panel_x + 10 + settings.label_x_offset;   //--- Setting symbol x-coordinate
      for(int i = 0; i < num_rows; i++) {                               //--- Iterating through rows
         string symbol_name = PREFIX + SYMB + IntegerToString(i);       //--- Defining symbol label name
         ObjectSetInteger(0, symbol_name, OBJPROP_XDISTANCE, symbol_x); //--- Updating symbol x-coordinate
         ObjectSetInteger(0, symbol_name, OBJPROP_YDISTANCE, first_row_y + i * settings.row_height + settings.label_y_offset); //--- Updating symbol y-coordinate
         int x_offset = settings.panel_x + 10 + column_widths[0] + settings.label_x_offset; //--- Setting data x-offset
         for(int j = 0; j < num_columns - 1; j++) {                      //--- Iterating through data columns
            string data_name = PREFIX + DATA + IntegerToString(i) + "_" + IntegerToString(j); //--- Defining data label name
            ObjectSetInteger(0, data_name, OBJPROP_XDISTANCE, x_offset); //--- Updating data x-coordinate
            ObjectSetInteger(0, data_name, OBJPROP_YDISTANCE, first_row_y + i * settings.row_height + settings.label_y_offset); //--- Updating data y-coordinate
            x_offset += column_widths[j + 1];                            //--- Updating x-offset
         }
      }
      // Updating footer panel and labels
      int footer_y = settings.panel_y + (num_rows + 3) * settings.row_height - settings.row_height; //--- Calculating footer y-coordinate
      ObjectSetInteger(0, PREFIX + FOOTER_PANEL, OBJPROP_XDISTANCE, settings.panel_x); //--- Updating footer panel x-coordinate
      ObjectSetInteger(0, PREFIX + FOOTER_PANEL, OBJPROP_YDISTANCE, footer_y); //--- Updating footer panel y-coordinate
      ObjectSetInteger(0, PREFIX + FOOTER_TEXT, OBJPROP_XDISTANCE, settings.panel_x + 10 + settings.label_x_offset); //--- Updating footer text x-coordinate
      ObjectSetInteger(0, PREFIX + FOOTER_TEXT, OBJPROP_YDISTANCE, footer_y + 8 + settings.label_y_offset); //--- Updating footer text y-coordinate
      int x_offset = settings.panel_x + 10 + column_widths[0] + settings.label_x_offset; //--- Setting footer data x-offset
      for(int j = 0; j < num_columns - 1; j++) {                        //--- Iterating through footer data
         string footer_data_name = PREFIX + FOOTER_DATA + IntegerToString(j); //--- Defining footer data name
         ObjectSetInteger(0, footer_data_name, OBJPROP_XDISTANCE, x_offset);  //--- Updating footer data x-coordinate
         ObjectSetInteger(0, footer_data_name, OBJPROP_YDISTANCE, footer_y + 8 + settings.label_y_offset); //--- Updating footer data y-coordinate
         x_offset += column_widths[j + 1];                              //--- Updating x-offset
      }
      // Updating account panel and labels
      int account_panel_y = footer_y + settings.row_height + 5;         //--- Calculating account panel y-coordinate
      ObjectSetInteger(0, PREFIX + ACCOUNT_PANEL, OBJPROP_XDISTANCE, settings.panel_x); //--- Updating account panel x-coordinate
      ObjectSetInteger(0, PREFIX + ACCOUNT_PANEL, OBJPROP_YDISTANCE, account_panel_y);  //--- Updating account panel y-coordinate
      int acc_x = settings.panel_x + 10 + settings.label_x_offset;      //--- Setting account label x-coordinate
      int acc_data_offset = 160;                                        //--- Setting data offset
      int acc_spacing = (panel_width - 45) / ArraySize(account_items);  //--- Calculating spacing
      for(int k = 0; k < ArraySize(account_items); k++) { //--- Iterating through account items
         string acc_text_name = PREFIX + ACC_TEXT + IntegerToString(k); //--- Defining account text name
         int text_x = acc_x + k * acc_spacing;                          //--- Calculating text x-coordinate
         ObjectSetInteger(0, acc_text_name, OBJPROP_XDISTANCE, text_x); //--- Updating account text x-coordinate
         ObjectSetInteger(0, acc_text_name, OBJPROP_YDISTANCE, account_panel_y + 8 + settings.label_y_offset); //--- Updating account text y-coordinate
         string acc_data_name = PREFIX + ACC_DATA + IntegerToString(k); //--- Defining account data name
         int data_x = text_x + acc_data_offset;                         //--- Calculating data x-coordinate
         ObjectSetInteger(0, acc_data_name, OBJPROP_XDISTANCE, data_x); //--- Updating account data x-coordinate
         ObjectSetInteger(0, acc_data_name, OBJPROP_YDISTANCE, account_panel_y + 8 + settings.label_y_offset); //--- Updating account data y-coordinate
      }
   }
   ChartRedraw(0);                                    //--- Redrawing chart
}
```

Here, we implement the "updatePanelPositions" function to enable the draggable feature of the enhanced dashboard, ensuring all UI elements move cohesively when the dashboard is dragged. We calculate "num\_rows" and "num\_columns" using "ArraySize" on "symbol\_data" and "headers", and compute "panel\_width" by summing "column\_widths" and using [MathMax](https://www.mql5.com/en/docs/math/mathmax) with "settings.header\_x\_distances" plus padding. We update the header panel and buttons by setting [OBJPROP\_XDISTANCE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) and "OBJPROP\_YDISTANCE" for "PREFIX + HEADER\_PANEL", "PREFIX + HEADER\_PANEL\_TEXT", "PREFIX + EXPORT\_BUTTON", "PREFIX + TOGGLE\_BUTTON", and "PREFIX + CLOSE\_BUTTON" using [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) with "settings.panel\_x" and "settings.panel\_y" coordinates.

If "panel\_minimized" is false, we update the main panel’s position with "PREFIX + PANEL", headers at "header\_y" calculated from "settings.panel\_y + settings.row\_height + 8 + settings.label\_y\_offset", symbol and data labels at "first\_row\_y" with "symbol\_x" and "x\_offset" adjusted by "column\_widths", footer panel and labels at "footer\_y" calculated for "num\_rows + 3", and account panel and labels at "account\_panel\_y" with "acc\_x" and "acc\_spacing" for alignment, all using the [ObjectSetInteger](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) function. We call [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/chartredraw) to refresh the display. This will ensure the entire dashboard moves seamlessly during dragging, maintaining layout integrity. We will need to define a logic to track the position of the cursor over the header or buttons for hover considerations. Here is the logic we used to implement that.

```
//+------------------------------------------------------------------+
//| Checking if cursor is inside header or buttons                   |
//+------------------------------------------------------------------+
bool isCursorInHeaderOrButtons(int mouse_x, int mouse_y) {
   int header_x = (int)ObjectGetInteger(0, PREFIX + HEADER_PANEL, OBJPROP_XDISTANCE);  //--- Getting header x-coordinate
   int header_y = (int)ObjectGetInteger(0, PREFIX + HEADER_PANEL, OBJPROP_YDISTANCE);  //--- Getting header y-coordinate
   int header_width = (int)ObjectGetInteger(0, PREFIX + HEADER_PANEL, OBJPROP_XSIZE);  //--- Getting header width
   int header_height = (int)ObjectGetInteger(0, PREFIX + HEADER_PANEL, OBJPROP_YSIZE); //--- Getting header height
   bool in_header = (mouse_x >= header_x && mouse_x <= header_x + header_width &&
                     mouse_y >= header_y && mouse_y <= header_y + header_height);      //--- Checking if in header
   int close_x = (int)ObjectGetInteger(0, PREFIX + CLOSE_BUTTON, OBJPROP_XDISTANCE);   //--- Getting close button x-coordinate
   int close_y = (int)ObjectGetInteger(0, PREFIX + CLOSE_BUTTON, OBJPROP_YDISTANCE);   //--- Getting close button y-coordinate
   int close_width = 20;                              //--- Setting close button width
   int close_height = 20;                             //--- Setting close button height
   bool in_close = (mouse_x >= close_x && mouse_x <= close_x + close_width &&
                    mouse_y >= close_y && mouse_y <= close_y + close_height);          //--- Checking if in close button
   int export_x = (int)ObjectGetInteger(0, PREFIX + EXPORT_BUTTON, OBJPROP_XDISTANCE); //--- Getting export button x-coordinate
   int export_y = (int)ObjectGetInteger(0, PREFIX + EXPORT_BUTTON, OBJPROP_YDISTANCE); //--- Getting export button y-coordinate
   int export_width = 20;                             //--- Setting export button width
   int export_height = 20;                            //--- Setting export button height
   bool in_export = (mouse_x >= export_x && mouse_x <= export_x + export_width &&
                     mouse_y >= export_y && mouse_y <= export_y + export_height);      //--- Checking if in export button
   int toggle_x = (int)ObjectGetInteger(0, PREFIX + TOGGLE_BUTTON, OBJPROP_XDISTANCE); //--- Getting toggle button x-coordinate
   int toggle_y = (int)ObjectGetInteger(0, PREFIX + TOGGLE_BUTTON, OBJPROP_YDISTANCE); //--- Getting toggle button y-coordinate
   int toggle_width = 20;                             //--- Setting toggle button width
   int toggle_height = 20;                            //--- Setting toggle button height
   bool in_toggle = (mouse_x >= toggle_x && mouse_x <= toggle_x + toggle_width &&
                     mouse_y >= toggle_y && mouse_y <= toggle_y + toggle_height);      //--- Checking if in toggle button
   return in_header || in_close || in_export || in_toggle;                             //--- Returning combined check
}
```

We implement the "isCursorInHeaderOrButtons" function to detect mouse cursor presence over interactive elements, enabling dragging and button interactions. We retrieve coordinates and dimensions for the header panel using [ObjectGetInteger](https://www.mql5.com/en/docs/objects/objectgetinteger) for "PREFIX + HEADER\_PANEL" with "OBJPROP\_XDISTANCE", "OBJPROP\_YDISTANCE", "OBJPROP\_XSIZE", and "OBJPROP\_YSIZE", storing them in "header\_x", "header\_y", "header\_width", and "header\_height", and check if the cursor ("mouse\_x", "mouse\_y") is within the header bounds with "in\_header".

Similarly, we get coordinates for "PREFIX + CLOSE\_BUTTON", "PREFIX + EXPORT\_BUTTON", and "PREFIX + TOGGLE\_BUTTON" using "OBJPROP\_XDISTANCE" and [OBJPROP\_YDISTANCE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer), setting "close\_width", "close\_height", "export\_width", "export\_height", "toggle\_width", and "toggle\_height" to 20, and verify if the cursor is within each button’s bounds with "in\_close", "in\_export", and "in\_toggle". We return true if the cursor is in the header or any button, combining conditions with the [OR operator](https://www.mql5.com/en/docs/basis/operations/bool). After hover detection, we will need to update the detected header or buttons for visual feedback. Here is the logic we implement to achieve that.

```
//+------------------------------------------------------------------+
//| Updating button hover states                                     |
//+------------------------------------------------------------------+
void updateButtonHoverStates(int mouse_x, int mouse_y) {
   int close_x = (int)ObjectGetInteger(0, PREFIX + CLOSE_BUTTON, OBJPROP_XDISTANCE);     //--- Getting close button x-coordinate
   int close_y = (int)ObjectGetInteger(0, PREFIX + CLOSE_BUTTON, OBJPROP_YDISTANCE);     //--- Getting close button y-coordinate
   int close_width = 20;                              //--- Setting close button width
   int close_height = 20;                             //--- Setting close button height
   bool is_close_hovered = (mouse_x >= close_x && mouse_x <= close_x + close_width &&
                            mouse_y >= close_y && mouse_y <= close_y + close_height);     //--- Checking if close button hovered
   if(is_close_hovered != prev_close_hovered) {       //--- Checking hover change
      ObjectSetInteger(0, PREFIX + CLOSE_BUTTON, OBJPROP_COLOR, is_close_hovered ? clrRed : clrBlack); //--- Updating close button color
      ObjectSetInteger(0, PREFIX + CLOSE_BUTTON, OBJPROP_BGCOLOR, is_close_hovered ? clrDodgerBlue : clrNONE); //--- Updating close button background
      prev_close_hovered = is_close_hovered;          //--- Updating previous hover state
      ChartRedraw(0);                                 //--- Redrawing chart
   }
   int export_x = (int)ObjectGetInteger(0, PREFIX + EXPORT_BUTTON, OBJPROP_XDISTANCE);    //--- Getting export button x-coordinate
   int export_y = (int)ObjectGetInteger(0, PREFIX + EXPORT_BUTTON, OBJPROP_YDISTANCE);    //--- Getting export button y-coordinate
   int export_width = 20;                             //--- Setting export button width
   int export_height = 20;                            //--- Setting export button height
   bool is_export_hovered = (mouse_x >= export_x && mouse_x <= export_x + export_width &&
                             mouse_y >= export_y && mouse_y <= export_y + export_height); //--- Checking if export button hovered
   if(is_export_hovered != prev_export_hovered) {     //--- Checking hover change
      ObjectSetInteger(0, PREFIX + EXPORT_BUTTON, OBJPROP_COLOR, is_export_hovered ? clrOrange : clrBlack); //--- Updating export button color
      ObjectSetInteger(0, PREFIX + EXPORT_BUTTON, OBJPROP_BGCOLOR, is_export_hovered ? clrDodgerBlue : clrNONE); //--- Updating export button background
      prev_export_hovered = is_export_hovered;        //--- Updating previous hover state
      ChartRedraw(0);                                 //--- Redrawing chart
   }
   int toggle_x = (int)ObjectGetInteger(0, PREFIX + TOGGLE_BUTTON, OBJPROP_XDISTANCE);    //--- Getting toggle button x-coordinate
   int toggle_y = (int)ObjectGetInteger(0, PREFIX + TOGGLE_BUTTON, OBJPROP_YDISTANCE);    //--- Getting toggle button y-coordinate
   int toggle_width = 20;                             //--- Setting toggle button width
   int toggle_height = 20;                            //--- Setting toggle button height
   bool is_toggle_hovered = (mouse_x >= toggle_x && mouse_x <= toggle_x + toggle_width &&
                             mouse_y >= toggle_y && mouse_y <= toggle_y + toggle_height); //--- Checking if toggle button hovered
   if(is_toggle_hovered != prev_toggle_hovered) {     //--- Checking hover change
      ObjectSetInteger(0, PREFIX + TOGGLE_BUTTON, OBJPROP_COLOR, is_toggle_hovered ? clrBlue : clrBlack); //--- Updating toggle button color
      ObjectSetInteger(0, PREFIX + TOGGLE_BUTTON, OBJPROP_BGCOLOR, is_toggle_hovered ? clrDodgerBlue : clrNONE); //--- Updating toggle button background
      prev_toggle_hovered = is_toggle_hovered;        //--- Updating previous hover state
      ChartRedraw(0);                                 //--- Redrawing chart
   }
}

//+------------------------------------------------------------------+
//| Updating header hover state                                      |
//+------------------------------------------------------------------+
void updateHeaderHoverState(int mouse_x, int mouse_y) {
   int header_x = (int)ObjectGetInteger(0, PREFIX + HEADER_PANEL, OBJPROP_XDISTANCE);  //--- Getting header x-coordinate
   int header_y = (int)ObjectGetInteger(0, PREFIX + HEADER_PANEL, OBJPROP_YDISTANCE);  //--- Getting header y-coordinate
   int header_width = (int)ObjectGetInteger(0, PREFIX + HEADER_PANEL, OBJPROP_XSIZE);  //--- Getting header width
   int header_height = (int)ObjectGetInteger(0, PREFIX + HEADER_PANEL, OBJPROP_YSIZE); //--- Getting header height
   int close_x = (int)ObjectGetInteger(0, PREFIX + CLOSE_BUTTON, OBJPROP_XDISTANCE);   //--- Getting close button x-coordinate
   int close_y = (int)ObjectGetInteger(0, PREFIX + CLOSE_BUTTON, OBJPROP_YDISTANCE);   //--- Getting close button y-coordinate
   int close_width = 20;                              //--- Setting close button width
   int close_height = 20;                             //--- Setting close button height
   int export_x = (int)ObjectGetInteger(0, PREFIX + EXPORT_BUTTON, OBJPROP_XDISTANCE); //--- Getting export button x-coordinate
   int export_y = (int)ObjectGetInteger(0, PREFIX + EXPORT_BUTTON, OBJPROP_YDISTANCE); //--- Getting export button y-coordinate
   int export_width = 20;                             //--- Setting export button width
   int export_height = 20;                            //--- Setting export button height
   int toggle_x = (int)ObjectGetInteger(0, PREFIX + TOGGLE_BUTTON, OBJPROP_XDISTANCE); //--- Getting toggle button x-coordinate
   int toggle_y = (int)ObjectGetInteger(0, PREFIX + TOGGLE_BUTTON, OBJPROP_YDISTANCE); //--- Getting toggle button y-coordinate
   int toggle_width = 20;                             //--- Setting toggle button width
   int toggle_height = 20;                            //--- Setting toggle button height
   bool is_header_hovered = (mouse_x >= header_x && mouse_x <= header_x + header_width &&
                             mouse_y >= header_y && mouse_y <= header_y + header_height &&
                             !(mouse_x >= close_x && mouse_x <= close_x + close_width &&
                               mouse_y >= close_y && mouse_y <= close_y + close_height) &&
                             !(mouse_x >= export_x && mouse_x <= export_x + export_width &&
                               mouse_y >= export_y && mouse_y <= export_y + export_height) &&
                             !(mouse_x >= toggle_x && mouse_x <= toggle_x + toggle_width &&
                               mouse_y >= toggle_y && mouse_y <= toggle_y + toggle_height)); //--- Checking if header hovered
   if(is_header_hovered != prev_header_hovered && !panel_dragging) {                         //--- Checking hover change
      ObjectSetInteger(0, PREFIX + HEADER_PANEL, OBJPROP_BGCOLOR, is_header_hovered ? clrRed : settings.section_bg_color); //--- Updating header background
      prev_header_hovered = is_header_hovered;        //--- Updating previous hover state
      ChartRedraw(0);                                 //--- Redrawing chart
   }
   updateButtonHoverStates(mouse_x, mouse_y);         //--- Updating button hover states
}
```

Finally, we implement the "updateButtonHoverStates" and "updateHeaderHoverState" functions to add visual feedback for user interactions, enhancing button and header responsiveness. In "updateButtonHoverStates", we check hover states for buttons by retrieving coordinates for "PREFIX + CLOSE\_BUTTON", "PREFIX + EXPORT\_BUTTON", and "PREFIX + TOGGLE\_BUTTON" using [ObjectGetInteger](https://www.mql5.com/en/docs/objects/objectgetinteger) with "OBJPROP\_XDISTANCE" and "OBJPROP\_YDISTANCE", setting "close\_width", "close\_height", "export\_width", "export\_height", "toggle\_width", and "toggle\_height" to 20.

For the close button, we set "is\_close\_hovered" if "mouse\_x" and "mouse\_y" are within its bounds, and if different from "prev\_close\_hovered", update "OBJPROP\_COLOR" to "clrRed" or "clrBlack" and "OBJPROP\_BGCOLOR" to "clrDodgerBlue" or "clrNONE" with "ObjectSetInteger", update "prev\_close\_hovered", and call the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/chartredraw) function. Similarly, for the export button, we set "is\_export\_hovered" to "clrOrange" or "clrBlack" and "clrDodgerBlue" or "clrNONE", update "prev\_export\_hovered", and redraw; for the toggle button, we use "clrBlue" or "clrBlack", update "prev\_toggle\_hovered", and redraw.

In "updateHeaderHoverState", we get "header\_x", "header\_y", "header\_width", and "header\_height" for "PREFIX + HEADER\_PANEL", and button coordinates, checking "is\_header\_hovered" if the cursor is within the header but outside button bounds. If "is\_header\_hovered" differs from "prev\_header\_hovered" and "panel\_dragging" is false, we update "OBJPROP\_BGCOLOR" of "PREFIX + HEADER\_PANEL" to "clrRed" or "settings.section\_bg\_color" with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger), update "prev\_header\_hovered", call "ChartRedraw", and invoke "updateButtonHoverStates". These functions will provide dynamic hover effects for intuitive user interaction. To make use of the functions, we will expand the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) function to house the visual feedback logic.

```
//+------------------------------------------------------------------+
//| Handling chart events for sorting, export, and UI interactions   |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam) {
   if(id == CHARTEVENT_OBJECT_CLICK) {                //--- Handling object click
      if(sparam == PREFIX + CLOSE_BUTTON) {           //--- Checking close button click
         Print("Closing the dashboard");              //--- Logging closing
         PlaySound("alert.wav");                      //--- Playing alert sound
         panel_is_visible = false;                    //--- Setting panel invisible
         deleteAllObjects();                          //--- Deleting all objects
         ChartRedraw(0);                              //--- Redrawing chart
      }
      else if(sparam == PREFIX + EXPORT_BUTTON) {     //--- Checking export button click
         Print("Exporting dashboard to CSV");         //--- Logging exporting
         ExportToCSV();                               //--- Exporting to CSV
         ChartRedraw(0);                              //--- Redrawing chart
      }
      else if(sparam == PREFIX + TOGGLE_BUTTON) {     //--- Checking toggle button click
         deleteAllObjects();                          //--- Deleting all objects
         panel_minimized = !panel_minimized;          //--- Toggling minimized state
         if(panel_minimized) {                        //--- Checking if minimized
            Print("Minimizing the dashboard");        //--- Logging minimizing
            createMinimizedDashboard();               //--- Creating minimized dashboard
         } else {
            Print("Maximizing the dashboard");        //--- Logging maximizing
            createFullDashboard();                    //--- Creating full dashboard
            // Resetting string variables to force update
            total_buys_str = "";                      //--- Resetting buys string
            total_sells_str = "";                     //--- Resetting sells string
            total_trades_str = "";                    //--- Resetting trades string
            total_lots_str = "";                      //--- Resetting lots string
            total_profit_str = "";                    //--- Resetting profit string
            total_pending_str = "";                   //--- Resetting pending string
            total_swap_str = "";                      //--- Resetting swap string
            total_comm_str = "";                      //--- Resetting commission string
            acc_bal_str = "";                         //--- Resetting balance string
            acc_eq_str = "";                          //--- Resetting equity string
            acc_free_str = "";                        //--- Resetting free margin string
            UpdateDashboard();                        //--- Updating dashboard
         }
         prev_header_hovered = false;                 //--- Resetting header hover
         prev_close_hovered = false;                  //--- Resetting close hover
         prev_export_hovered = false;                 //--- Resetting export hover
         prev_toggle_hovered = false;                 //--- Resetting toggle hover
         ObjectSetInteger(0, PREFIX + HEADER_PANEL, OBJPROP_BGCOLOR, settings.section_bg_color); //--- Resetting header background
         ObjectSetInteger(0, PREFIX + CLOSE_BUTTON, OBJPROP_COLOR, clrBlack); //--- Resetting close button color
         ObjectSetInteger(0, PREFIX + CLOSE_BUTTON, OBJPROP_BGCOLOR, clrNONE); //--- Resetting close button background
         ObjectSetInteger(0, PREFIX + EXPORT_BUTTON, OBJPROP_COLOR, clrBlack); //--- Resetting export button color
         ObjectSetInteger(0, PREFIX + EXPORT_BUTTON, OBJPROP_BGCOLOR, clrNONE); //--- Resetting export button background
         ObjectSetInteger(0, PREFIX + TOGGLE_BUTTON, OBJPROP_COLOR, clrBlack); //--- Resetting toggle button color
         ObjectSetInteger(0, PREFIX + TOGGLE_BUTTON, OBJPROP_BGCOLOR, clrNONE); //--- Resetting toggle button background
         ChartRedraw(0);                              //--- Redrawing chart
      }
      else {
         for(int i = 0; i < ArraySize(headers); i++) { //--- Iterating through headers
            if(sparam == PREFIX + HEADER + IntegerToString(i)) { //--- Checking header click
               if(sort_column == i)                   //--- Checking if same column
                  sort_ascending = !sort_ascending;   //--- Toggling sort direction
               else {
                  sort_column = i;                    //--- Setting new sort column
                  sort_ascending = true;              //--- Setting to ascending
               }
               UpdateDashboard();                     //--- Updating dashboard
               break;                                 //--- Exiting loop
            }
         }
      }
   }
   else if(id == CHARTEVENT_KEYDOWN && lparam == 'E') { //--- Handling 'E' key press
      ExportToCSV();                                  //--- Exporting to CSV
   }
   else if(id == CHARTEVENT_MOUSE_MOVE && panel_is_visible) { //--- Handling mouse move
      int mouse_x = (int)lparam;                      //--- Getting mouse x-coordinate
      int mouse_y = (int)dparam;                      //--- Getting mouse y-coordinate
      int mouse_state = (int)sparam;                  //--- Getting mouse state
      if(mouse_x == last_mouse_x && mouse_y == last_mouse_y && !panel_dragging) { //--- Checking if mouse moved
         return;                                      //--- Exiting if no movement
      }
      last_mouse_x = mouse_x;                         //--- Updating last mouse x
      last_mouse_y = mouse_y;                         //--- Updating last mouse y
      updateHeaderHoverState(mouse_x, mouse_y);       //--- Updating header hover state
      int header_x = (int)ObjectGetInteger(0, PREFIX + HEADER_PANEL, OBJPROP_XDISTANCE); //--- Getting header x-coordinate
      int header_y = (int)ObjectGetInteger(0, PREFIX + HEADER_PANEL, OBJPROP_YDISTANCE); //--- Getting header y-coordinate
      int header_width = (int)ObjectGetInteger(0, PREFIX + HEADER_PANEL, OBJPROP_XSIZE); //--- Getting header width
      int header_height = (int)ObjectGetInteger(0, PREFIX + HEADER_PANEL, OBJPROP_YSIZE);//--- Getting header height
      int close_x = (int)ObjectGetInteger(0, PREFIX + CLOSE_BUTTON, OBJPROP_XDISTANCE);  //--- Getting close button x-coordinate
      int close_width = 20;                           //--- Setting close button width
      int export_x = (int)ObjectGetInteger(0, PREFIX + EXPORT_BUTTON, OBJPROP_XDISTANCE);//--- Getting export button x-coordinate
      int export_width = 20;                          //--- Setting export button width
      int toggle_x = (int)ObjectGetInteger(0, PREFIX + TOGGLE_BUTTON, OBJPROP_XDISTANCE);//--- Getting toggle button x-coordinate
      int toggle_width = 20;                          //--- Setting toggle button width
      if(prev_mouse_state == 0 && mouse_state == 1) { //--- Checking mouse click start
         if(mouse_x >= header_x && mouse_x <= header_x + header_width &&
            mouse_y >= header_y && mouse_y <= header_y + header_height &&
            !(mouse_x >= close_x && mouse_x <= close_x + close_width) &&
            !(mouse_x >= export_x && mouse_x <= export_x + export_width) &&
            !(mouse_x >= toggle_x && mouse_x <= toggle_x + toggle_width)) { //--- Checking if in draggable area
            panel_dragging = true;                     //--- Starting dragging
            panel_drag_x = mouse_x;                    //--- Setting drag start x
            panel_drag_y = mouse_y;                    //--- Setting drag start y
            panel_start_x = header_x;                  //--- Setting panel start x
            panel_start_y = header_y;                  //--- Setting panel start y
            ObjectSetInteger(0, PREFIX + HEADER_PANEL, OBJPROP_BGCOLOR, clrMediumBlue); //--- Setting dragging color
            ChartSetInteger(0, CHART_MOUSE_SCROLL, false); //--- Disabling chart scroll
         }
      }
      if(panel_dragging && mouse_state == 1) {        //--- Handling dragging
         int dx = mouse_x - panel_drag_x;             //--- Calculating x change
         int dy = mouse_y - panel_drag_y;             //--- Calculating y change
         settings.panel_x = panel_start_x + dx;       //--- Updating panel x
         settings.panel_y = panel_start_y + dy;       //--- Updating panel y
         int chart_width = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);   //--- Getting chart width
         int chart_height = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS); //--- Getting chart height
         int num_columns = ArraySize(headers);        //--- Getting number of columns
         int column_width_sum = 0;                    //--- Initializing column width sum
         for(int i = 0; i < num_columns; i++) column_width_sum += column_widths[i]; //--- Adding column width
         int panel_width = MathMax(settings.header_x_distances[num_columns - 1] + column_widths[num_columns - 1], column_width_sum) + 20 + settings.label_x_offset; //--- Calculating panel width
         int panel_height = panel_minimized ? settings.row_height : (ArraySize(symbol_data) + 3) * settings.row_height; //--- Calculating panel height
         settings.panel_x = MathMax(0, MathMin(chart_width - panel_width, settings.panel_x)); //--- Constraining x
         settings.panel_y = MathMax(0, MathMin(chart_height - panel_height, settings.panel_y)); //--- Constraining y
         updatePanelPositions();                      //--- Updating object positions
         ChartRedraw(0);                              //--- Redrawing chart
      }
      if(mouse_state == 0 && prev_mouse_state == 1) { //--- Handling mouse release
         if(panel_dragging) {                         //--- Checking if was dragging
            panel_dragging = false;                   //--- Stopping dragging
            updateHeaderHoverState(mouse_x, mouse_y); //--- Updating hover state
            ChartSetInteger(0, CHART_MOUSE_SCROLL, true); //--- Enabling chart scroll
            ChartRedraw(0);                           //--- Redrawing chart
         }
      }
      prev_mouse_state = mouse_state;                //--- Updating previous mouse state
   }
}
```

Here, we expand the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) function to handle interactive events, managing clicks for closing, exporting, toggling, sorting, and mouse movements for dragging. For [CHARTEVENT\_OBJECT\_CLICK](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents), we check if "sparam" is "PREFIX + CLOSE\_BUTTON", logging with "Print", playing "alert.wav" with [PlaySound](https://www.mql5.com/en/docs/common/playsound), setting "panel\_is\_visible" to false, calling "deleteAllObjects", and redrawing with "ChartRedraw". If "sparam" is "PREFIX + EXPORT\_BUTTON", we log and call "ExportToCSV".

For "PREFIX + TOGGLE\_BUTTON", we delete objects, toggle "panel\_minimized", log "Minimizing" or "Maximizing" with "Print", call "createMinimizedDashboard" or "createFullDashboard", reset string variables like "total\_buys\_str" and "acc\_bal\_str", call "UpdateDashboard", reset hover states ("prev\_header\_hovered", "prev\_close\_hovered", etc.), and reset colors for "PREFIX + HEADER\_PANEL", "PREFIX + CLOSE\_BUTTON", "PREFIX + EXPORT\_BUTTON", and "PREFIX + TOGGLE\_BUTTON" using the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) function. That will give you something as follows.

![MINIMIZED STATE](https://c.mql5.com/2/161/Screenshot_2025-08-04_185243.png)

For header clicks, we loop through "headers", toggle "sort\_ascending" if "sort\_column" matches, or set new "sort\_column" and "sort\_ascending" to true, then call "UpdateDashboard". For [CHARTEVENT\_KEYDOWN](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents) with 'E', we call "ExportToCSV". For [CHARTEVENT\_MOUSE\_MOVE](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents) when "panel\_is\_visible", we get "mouse\_x", "mouse\_y", and "mouse\_state", exit if unchanged and not dragging, update "last\_mouse\_x" and "last\_mouse\_y", and call "updateHeaderHoverState".

If "prev\_mouse\_state" is 0 and "mouse\_state" is 1, we check for draggable area clicks (excluding buttons), set "panel\_dragging" to true, store coordinates, set header color to "clrMediumBlue", and disable scroll with the [ChartSetInteger](https://www.mql5.com/en/docs/chart_operations/chartsetinteger) function. If dragging and "mouse\_state" is 1, we calculate "dx" and "dy", update "settings.panel\_x" and "settings.panel\_y" within chart bounds, call "updatePanelPositions", and redraw. On mouse release, we stop dragging, update hover, re-enable scroll, and redraw. This enables dynamic UI interactions for a user-friendly dashboard. Upon compilation, we get the following button hover state outcome.

![BUTTON HOVER STATES](https://c.mql5.com/2/161/Screenshot_2025-08-04_185439.png)

The outcome for a maximized and drag state is as follows.

![FINAL DRAG STATE](https://c.mql5.com/2/161/Screenshot_2025-08-04_185335.png)

From the image, we can see that we have added the dashboard components for the hover, drag, and minimization logic, hence achieving our objectives. What now remains is testing the workability of the project, and that is handled in the preceding section.

### Backtesting

We did the testing, and below is the compiled visualization in a single [Graphics Interchange Format](https://en.wikipedia.org/wiki/GIF "https://en.wikipedia.org/wiki/GIF") (GIF) bitmap image format.

![BACKTEST](https://c.mql5.com/2/161/ID_2_Test.gif)

### Conclusion

In conclusion, we’ve enhanced the Informational Dashboard in MQL5 for Part 8, adding draggable and minimizable features, interactive buttons like "CLOSE\_BUTTON" and "TOGGLE\_BUTTON", and hover effects to improve user experience while maintaining robust multi-symbol position and account monitoring. We’ve detailed the architecture and implementation, using functions like "createFullDashboard", "updatePanelPositions", and [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) to deliver a flexible, visually responsive tool with real-time updates and [Excel](https://en.wikipedia.org/wiki/Microsoft_Excel "https://en.wikipedia.org/wiki/Microsoft_Excel") CSV export. You can customize this dashboard to optimize your trading workflow, making position analysis more intuitive and efficient.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19059.zip "Download all attachments in the single ZIP archive")

[Informational\_Dashboard\_PART\_2.mq5](https://www.mql5.com/en/articles/download/19059/informational_dashboard_part_2.mq5 "Download Informational_Dashboard_PART_2.mq5")(86.08 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/492931)**

![Mastering Log Records (Part 10): Avoiding Log Replay by Implementing a Suppression](https://c.mql5.com/2/160/19014-mastering-log-records-part-logo.png)[Mastering Log Records (Part 10): Avoiding Log Replay by Implementing a Suppression](https://www.mql5.com/en/articles/19014)

We created a log suppression system in the Logify library. It details how the CLogifySuppression class reduces console noise by applying configurable rules to avoid repetitive or irrelevant messages. We also cover the external configuration framework, validation mechanisms, and comprehensive testing to ensure robustness and flexibility in log capture during bot or indicator development.

![Python-MetaTrader 5 Strategy Tester (Part 01): Trade Simulator](https://c.mql5.com/2/162/18971-python-metatrader-5-strategy-logo__1.png)[Python-MetaTrader 5 Strategy Tester (Part 01): Trade Simulator](https://www.mql5.com/en/articles/18971)

The MetaTrader 5 module offered in Python provides a convenient way of opening trades in the MetaTrader 5 app using Python, but it has a huge problem, it doesn't have the strategy tester capability present in the MetaTrader 5 app, In this article series, we will build a framework for back testing your trading strategies in Python environments.

![Formulating Dynamic Multi-Pair EA (Part 4): Volatility and Risk Adjustment](https://c.mql5.com/2/162/18165-formulating-dynamic-multi-pair-logo__1.png)[Formulating Dynamic Multi-Pair EA (Part 4): Volatility and Risk Adjustment](https://www.mql5.com/en/articles/18165)

This phase fine-tunes your multi-pair EA to adapt trade size and risk in real time using volatility metrics like ATR boosting consistency, protection, and performance across diverse market conditions.

![Building a Trading System (Part 2): The Science of Position Sizing](https://c.mql5.com/2/162/18991-building-a-profitable-trading-logo.png)[Building a Trading System (Part 2): The Science of Position Sizing](https://www.mql5.com/en/articles/18991)

Even with a positive-expectancy system, position sizing determines whether you thrive or collapse. It’s the pivot of risk management—translating statistical edges into real-world results while safeguarding your capital.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=wvpqenchzfabwffnqgjwhneyekjpicmx&ssn=1769180012277102327&ssn_dr=0&ssn_sr=0&fv_date=1769180012&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19059&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Trading%20Tools%20(Part%208)%3A%20Enhanced%20Informational%20Dashboard%20with%20Draggable%20and%20Minimizable%20Features%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918001229760449&fz_uniq=5068766754928262508&sv=2552)

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