---
title: MQL5 Trading Tools (Part 3): Building a Multi-Timeframe Scanner Dashboard for Strategic Trading
url: https://www.mql5.com/en/articles/18319
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T17:56:16.385589
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/18319&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049497401334541454)

MetaTrader 5 / Trading


### Introduction

In our [previous article, Part 2](https://www.mql5.com/en/articles/17972), we enhanced a Trade Assistant Tool in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) with dynamic visual feedback for improved interactivity. Now, we focus on building a multi-timeframe scanner dashboard to deliver real-time trading signals for strategic decision-making. We introduce a grid-based interface with indicator-driven signals and a close button, covering these advancements through the following subtopics:

1. [The plan of the Scanner Dashboard](https://www.mql5.com/en/articles/18319#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/18319#para2)
3. [Backtesting](https://www.mql5.com/en/articles/18319#para3)
4. [Conclusion](https://www.mql5.com/en/articles/18319#para4)

These sections guide us toward creating an intuitive and powerful trading dashboard.

### The plan of the Scanner Dashboard

We aim to create a multi-timeframe scanner [dashboard](https://en.wikipedia.org/wiki/Dashboard_(computing) "https://en.wikipedia.org/wiki/Dashboard") that provides clear, real-time trading signals to enhance strategic decision-making. The dashboard will feature a grid layout displaying buy and sell signals across multiple timeframes, allowing us to quickly assess market conditions without switching charts. A close button will be included to enable easy panel dismissal, ensuring a clean and flexible user experience that adapts to our trading needs.

We will incorporate signals from key indicators, including the [Relative Strength Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi") (RSI), Stochastic Oscillator (STOCH), Commodity Channel Index (CCI), Average Directional Index (ADX), and [Awesome Oscillator](https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/awesome "https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/awesome") (AO), which are designed to identify potential trade opportunities with customizable strength thresholds. Still, the choice of the indicators or price action data to use is upon you. This setup will help us spot trends and reversals across timeframes, supporting both short-term and long-term strategies. Our goal is a streamlined, intuitive tool that delivers actionable insights while remaining user-friendly, paving the way for future enhancements like automated alerts or additional indicators. Below is a visualization of what we aim to achieve.

![IMPLEMENTATION PLAN](https://c.mql5.com/2/146/Screenshot_2025-05-28_192440.png)

### Implementation in MQL5

To create the program in [MQL5](https://www.metatrader5.com/ "https://www.metatrader5.com/"), we will need to define the program metadata, and then define some object name constants, that will help us refer to the dashboard objects and manage them easily.

```
//+------------------------------------------------------------------+
//|                             TimeframeScanner Dashboard EA.mq5    |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Allan Munene Mutiiria."
#property link      "https://t.me/Forex_Algo_Trader"
#property version   "1.00"

// Define identifiers and properties for UI elements
#define MAIN_PANEL              "PANEL_MAIN"                     //--- Main panel rectangle identifier
#define HEADER_PANEL            "PANEL_HEADER"                   //--- Header panel rectangle identifier
#define HEADER_PANEL_ICON       "PANEL_HEADER_ICON"              //--- Header icon label identifier
#define HEADER_PANEL_TEXT       "PANEL_HEADER_TEXT"              //--- Header title label identifier
#define CLOSE_BUTTON            "BUTTON_CLOSE"                   //--- Close button identifier
#define SYMBOL_RECTANGLE        "SYMBOL_HEADER"                  //--- Symbol rectangle identifier
#define SYMBOL_TEXT             "SYMBOL_TEXT"                    //--- Symbol text label identifier
#define TIMEFRAME_RECTANGLE     "TIMEFRAME_"                     //--- Timeframe rectangle prefix
#define TIMEFRAME_TEXT          "TIMEFRAME_TEXT_"                //--- Timeframe text label prefix
#define HEADER_RECTANGLE        "HEADER_"                        //--- Header rectangle prefix
#define HEADER_TEXT             "HEADER_TEXT_"                   //--- Header text label prefix
#define RSI_RECTANGLE           "RSI_"                           //--- RSI rectangle prefix
#define RSI_TEXT                "RSI_TEXT_"                      //--- RSI text label prefix
#define STOCH_RECTANGLE         "STOCH_"                         //--- Stochastic rectangle prefix
#define STOCH_TEXT              "STOCH_TEXT_"                    //--- Stochastic text label prefix
#define CCI_RECTANGLE           "CCI_"                           //--- CCI rectangle prefix
#define CCI_TEXT                "CCI_TEXT_"                      //--- CCI text label prefix
#define ADX_RECTANGLE           "ADX_"                           //--- ADX rectangle prefix
#define ADX_TEXT                "ADX_TEXT_"                      //--- ADX text label prefix
#define AO_RECTANGLE            "AO_"                            //--- AO rectangle prefix
#define AO_TEXT                 "AO_TEXT_"                       //--- AO text label prefix
#define BUY_RECTANGLE           "BUY_"                           //--- Buy rectangle prefix
#define BUY_TEXT                "BUY_TEXT_"                      //--- Buy text label prefix
#define SELL_RECTANGLE          "SELL_"                          //--- Sell rectangle prefix
#define SELL_TEXT               "SELL_TEXT_"                     //--- Sell text label prefix
#define WIDTH_TIMEFRAME         90                               //--- Width of timeframe and symbol rectangles
#define WIDTH_INDICATOR         70                               //--- Width of indicator rectangles
#define WIDTH_SIGNAL            90                               //--- Width of BUY/SELL signal rectangles
#define HEIGHT_RECTANGLE        25                               //--- Height of all rectangles
#define COLOR_WHITE             clrWhite                         //--- White color for text and backgrounds
#define COLOR_BLACK             clrBlack                         //--- Black color for borders and text
#define COLOR_LIGHT_GRAY        C'230,230,230'                   //--- Light gray color for signal backgrounds
#define COLOR_DARK_GRAY         C'105,105,105'                   //--- Dark gray color for indicator backgrounds
```

We start by establishing the user interface framework for our multi-timeframe scanner dashboard by using the [#define](https://www.mql5.com/en/docs/basis/preprosessor/constant) directive to create constants like "MAIN\_PANEL" and "HEADER\_PANEL" for the main and header panel rectangles, and "HEADER\_PANEL\_ICON", "HEADER\_PANEL\_TEXT", and "CLOSE\_BUTTON" for the header’s icon, title, and close button elements.

We define identifiers for the dashboard’s grid structure. For the symbol, we set "SYMBOL\_RECTANGLE" and "SYMBOL\_TEXT", while "TIMEFRAME\_RECTANGLE" and "TIMEFRAME\_TEXT" prefixes handle timeframe rows. We use "HEADER\_RECTANGLE" and "HEADER\_TEXT" prefixes for column headers, and prefixes like "RSI\_RECTANGLE", "STOCH\_RECTANGLE", "BUY\_RECTANGLE", with corresponding "RSI\_TEXT", "STOCH\_TEXT", and "BUY\_TEXT" for indicator and signal cells.

We configure sizes with "WIDTH\_TIMEFRAME" (90 pixels), "WIDTH\_INDICATOR" (70 pixels), "WIDTH\_SIGNAL" (90 pixels), and "HEIGHT\_RECTANGLE" (25 pixels). We define colors using "COLOR\_WHITE" and "COLOR\_BLACK" for text and borders, "COLOR\_LIGHT\_GRAY" ("C'230,230,230'") for signal backgrounds, and "COLOR\_DARK\_GRAY" ("C'105,105,105'") for indicators, ensuring a uniform and clear layout. We then need to define some more global variables that we will throughout the program.

```
bool panel_is_visible = true;                                    //--- Flag to control panel visibility

// Define the timeframes to be used
ENUM_TIMEFRAMES timeframes_array[] = {PERIOD_M1, PERIOD_M5, PERIOD_M15, PERIOD_M20, PERIOD_M30,
                                      PERIOD_H1, PERIOD_H2, PERIOD_H3, PERIOD_H4, PERIOD_H8,
                                      PERIOD_H12, PERIOD_D1, PERIOD_W1}; //--- Array of timeframes for scanning

// Global variables for indicator values
double rsi_values[];                                             //--- Array to store RSI values
double stochastic_values[];                                      //--- Array to store Stochastic signal line values
double cci_values[];                                             //--- Array to store CCI values
double adx_values[];                                             //--- Array to store ADX values
double ao_values[];                                              //--- Array to store AO values
```

Here, we declare the [boolean](https://www.mql5.com/en/docs/basis/types/integer/boolconst) variable "panel\_is\_visible" and set it to true, which determines whether the dashboard is displayed on the chart. This flag allows us to toggle the dashboard’s visibility as needed, especially when we don't need data updates. We then define the array "timeframes\_array" using the [ENUM\_TIMEFRAMES](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes) type, listing periods from "PERIOD\_M1" (1-minute) to "PERIOD\_W1" (weekly). This array specifies the timeframes the dashboard will analyze, enabling us to scan market signals across multiple time horizons in a structured manner. If you don't need some or need twisting, just modify the enumerations.

To store indicator data, we create double arrays "rsi\_values", "stochastic\_values", "cci\_values", "adx\_values", and "ao\_values". These arrays hold the calculated values for the [Relative Strength Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi"), Stochastic Oscillator, Commodity Channel Index, Average Directional Index, and Awesome Oscillator, respectively, allowing us to process and display trading signals for each timeframe efficiently. We can now define some helper functions that we will use to determine the direction and truncation of the retrieved chart symbol as follows.

```
//+------------------------------------------------------------------+
//| Truncate timeframe enum to display string                        |
//+------------------------------------------------------------------+
string truncate_timeframe_name(int timeframe_index)              //--- Function to format timeframe name
{
   string timeframe_string = StringSubstr(EnumToString(timeframes_array[timeframe_index]), 7); //--- Extract timeframe name
   return timeframe_string;                                      //--- Return formatted name
}

//+------------------------------------------------------------------+
//| Calculate signal strength for buy/sell                           |
//+------------------------------------------------------------------+
string calculate_signal_strength(double rsi, double stochastic, double cci, double adx, double ao, bool is_buy) //--- Function to compute signal strength
{
   int signal_strength = 0;                                      //--- Initialize signal strength counter

   if(is_buy && rsi < 40) signal_strength++;                     //--- Increment for buy if RSI is oversold
   else if(!is_buy && rsi > 60) signal_strength++;               //--- Increment for sell if RSI is overbought

   if(is_buy && stochastic < 40) signal_strength++;              //--- Increment for buy if Stochastic is oversold
   else if(!is_buy && stochastic > 60) signal_strength++;        //--- Increment for sell if Stochastic is overbought

   if(is_buy && cci < -70) signal_strength++;                    //--- Increment for buy if CCI is oversold
   else if(!is_buy && cci > 70) signal_strength++;               //--- Increment for sell if CCI is overbought

   if(adx > 40) signal_strength++;                               //--- Increment if ADX indicates strong trend

   if(is_buy && ao > 0) signal_strength++;                       //--- Increment for buy if AO is positive
   else if(!is_buy && ao < 0) signal_strength++;                 //--- Increment for sell if AO is negative

   if(signal_strength >= 3) return is_buy ? "Strong Buy" : "Strong Sell"; //--- Return strong signal if 3+ conditions met
   if(signal_strength >= 2) return is_buy ? "Buy" : "Sell";      //--- Return regular signal if 2 conditions met
   return "Neutral";                                             //--- Return neutral if insufficient conditions
}
```

Here, we define the "truncate\_timeframe\_name" function, which takes an integer parameter "timeframe\_index" to format timeframe names for display. Inside, we use the [StringSubstr](https://www.mql5.com/en/docs/strings/stringsubstr) function to extract a substring from the result of the [EnumToString](https://www.mql5.com/en/docs/convert/enumtostring) function applied to "timeframes\_array\[timeframe\_index\]", starting at position 7, and store it in "timeframe\_string". We then return "timeframe\_string", providing a clean, user-readable timeframe name.

We create the "calculate\_signal\_strength" function to determine buy or sell signals based on indicator values. We initialize an integer "signal\_strength" to zero to count matching conditions. For the [Relative Strength Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi"), we increment "signal\_strength" if "is\_buy" is true and "rsi" is below 40 (oversold) or if "is\_buy" is false and "rsi" exceeds 60 (overbought). Similarly, we check "stochastic" (below 40 or above 60), "CCI" (below -70 or above 70), and "ao" (positive for buy, negative for sell), incrementing "signal\_strength" for each met condition.

We also evaluate the Average Directional Index, incrementing "signal\_strength" if "adx" exceeds 40, indicating a strong trend for both buy and sell scenarios. If "signal\_strength" reaches 3 or more, we return “Strong Buy” for "is\_buy" true or “Strong Sell” otherwise. If it’s 2, we return “Buy” or “Sell”, and for fewer, we return “Neutral”, enabling clear signal classification for the dashboard. Finally, we can now define the functions that will enable us to create the objects.

```
//+------------------------------------------------------------------+
//| Create a rectangle for the UI                                    |
//+------------------------------------------------------------------+
bool create_rectangle(string object_name, int x_distance, int y_distance, int x_size, int y_size,
                      color background_color, color border_color = COLOR_BLACK) //--- Function to create a rectangle
{
   ResetLastError();                                                            //--- Reset error code
   if(!ObjectCreate(0, object_name, OBJ_RECTANGLE_LABEL, 0, 0, 0)) {            //--- Create rectangle object
      Print(__FUNCTION__, ": failed to create Rectangle: ERR Code: ", GetLastError()); //--- Log creation failure
      return(false);                                                            //--- Return failure
   }
   ObjectSetInteger(0, object_name, OBJPROP_XDISTANCE, x_distance);             //--- Set x position
   ObjectSetInteger(0, object_name, OBJPROP_YDISTANCE, y_distance);             //--- Set y position
   ObjectSetInteger(0, object_name, OBJPROP_XSIZE, x_size);                     //--- Set width
   ObjectSetInteger(0, object_name, OBJPROP_YSIZE, y_size);                     //--- Set height
   ObjectSetInteger(0, object_name, OBJPROP_CORNER, CORNER_RIGHT_UPPER);        //--- Set corner to top-right
   ObjectSetInteger(0, object_name, OBJPROP_BGCOLOR, background_color);         //--- Set background color
   ObjectSetInteger(0, object_name, OBJPROP_BORDER_COLOR, border_color);        //--- Set border color
   ObjectSetInteger(0, object_name, OBJPROP_BORDER_TYPE, BORDER_FLAT);          //--- Set flat border style
   ObjectSetInteger(0, object_name, OBJPROP_BACK, false);                       //--- Set to foreground

   ChartRedraw(0);                                                              //--- Redraw chart
   return(true);                                                                //--- Return success
}

//+------------------------------------------------------------------+
//| Create a text label for the UI                                   |
//+------------------------------------------------------------------+
bool create_label(string object_name, string text, int x_distance, int y_distance, int font_size = 12,
                  color text_color = COLOR_BLACK, string font = "Arial Rounded MT Bold") //--- Function to create a label
{
   ResetLastError();                                                               //--- Reset error code
   if(!ObjectCreate(0, object_name, OBJ_LABEL, 0, 0, 0)) {                         //--- Create label object
      Print(__FUNCTION__, ": failed to create Label: ERR Code: ", GetLastError()); //--- Log creation failure
      return(false);                                                               //--- Return failure
   }
   ObjectSetInteger(0, object_name, OBJPROP_XDISTANCE, x_distance);                //--- Set x position
   ObjectSetInteger(0, object_name, OBJPROP_YDISTANCE, y_distance);                //--- Set y position
   ObjectSetInteger(0, object_name, OBJPROP_CORNER, CORNER_RIGHT_UPPER);           //--- Set corner to top-right
   ObjectSetString(0, object_name, OBJPROP_TEXT, text);                            //--- Set label text
   ObjectSetString(0, object_name, OBJPROP_FONT, font);                            //--- Set font
   ObjectSetInteger(0, object_name, OBJPROP_FONTSIZE, font_size);                  //--- Set font size
   ObjectSetInteger(0, object_name, OBJPROP_COLOR, text_color);                    //--- Set text color
   ObjectSetInteger(0, object_name, OBJPROP_ANCHOR, ANCHOR_CENTER);                //--- Center text

   ChartRedraw(0);                                                                 //--- Redraw chart
   return(true);                                                                   //--- Return success
}
```

To enable the creation of the objects, we define the "create\_rectangle" function with parameters "object\_name", "x\_distance", "y\_distance", "x\_size", "y\_size", "background\_color", and "border\_color". We use the [ResetLastError](https://www.mql5.com/en/docs/common/ResetLastError) function, create an [OBJ\_RECTANGLE\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) with the [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) function, and log errors with the "Print" function if it fails, returning false.

We set rectangle properties with the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/ObjectSetInteger) function for the position, size, "CORNER\_RIGHT\_UPPER", "background\_color", "border\_color", and "BORDER\_FLAT", ensuring foreground display. We use the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function and return true. For text, we define the "create\_label" function with "object\_name", "text", "x\_distance", "y\_distance", "font\_size", "text\_color", and "font".

We use the "ResetLastError" function, create an "OBJ\_LABEL" with the "ObjectCreate" function, and log errors if it fails. We use the "ObjectSetInteger" function for position, size, color, and "ANCHOR\_CENTER", and the [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) function for "text" and "font". We use the "ChartRedraw" function and return true. Armed with these functions, we can now create the initial panel objects to give us a starting point in the "OnInit" event handler.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()                                                                             //--- Initialize EA
{
   create_rectangle(MAIN_PANEL, 632, 40, 617, 374, C'30,30,30', BORDER_FLAT);            //--- Create main panel background
   create_rectangle(HEADER_PANEL, 632, 40, 617, 27, C'60,60,60', BORDER_FLAT);           //--- Create header panel background
   create_label(HEADER_PANEL_ICON, CharToString(91), 620, 54, 18, clrAqua, "Wingdings"); //--- Create header icon
   create_label(HEADER_PANEL_TEXT, "TimeframeScanner", 527, 52, 13, COLOR_WHITE);        //--- Create header title
   create_label(CLOSE_BUTTON, CharToString('r'), 32, 54, 18, clrYellow, "Webdings");     //--- Create close button

   // Create header rectangle and label
   create_rectangle(SYMBOL_RECTANGLE, 630, 75, WIDTH_TIMEFRAME, HEIGHT_RECTANGLE, clrGray); //--- Create symbol rectangle
   create_label(SYMBOL_TEXT, _Symbol, 585, 85, 11, COLOR_WHITE); //--- Create symbol label

   // Create summary and indicator headers (rectangles and labels)
   string header_names[] = {"BUY", "SELL", "RSI", "STOCH", "CCI", "ADX", "AO"};            //--- Define header titles
   for(int header_index = 0; header_index < ArraySize(header_names); header_index++) {     //--- Loop through headers
      int x_offset = (630 - WIDTH_TIMEFRAME) - (header_index < 2 ? header_index * WIDTH_SIGNAL : 2 * WIDTH_SIGNAL + (header_index - 2) * WIDTH_INDICATOR) + (1 + header_index); //--- Calculate x position
      int width = (header_index < 2 ? WIDTH_SIGNAL : WIDTH_INDICATOR);                     //--- Set width based on header type
      create_rectangle(HEADER_RECTANGLE + IntegerToString(header_index), x_offset, 75, width, HEIGHT_RECTANGLE, clrGray);             //--- Create header rectangle
      create_label(HEADER_TEXT + IntegerToString(header_index), header_names[header_index], x_offset - width/2, 85, 11, COLOR_WHITE); //--- Create header label
   }

   // Create timeframe rectangles and labels, and summary/indicator cells
   for(int timeframe_index = 0; timeframe_index < ArraySize(timeframes_array); timeframe_index++) {            //--- Loop through timeframes
      // Highlight current timeframe
      color timeframe_background = (timeframes_array[timeframe_index] == _Period) ? clrLimeGreen : clrGray;    //--- Set background color for current timeframe
      color timeframe_text_color = (timeframes_array[timeframe_index] == _Period) ? COLOR_BLACK : COLOR_WHITE; //--- Set text color for current timeframe

      create_rectangle(TIMEFRAME_RECTANGLE + IntegerToString(timeframe_index), 630, (75 + HEIGHT_RECTANGLE) + timeframe_index * HEIGHT_RECTANGLE - (1 + timeframe_index), WIDTH_TIMEFRAME, HEIGHT_RECTANGLE, timeframe_background);   //--- Create timeframe rectangle
      create_label(TIMEFRAME_TEXT + IntegerToString(timeframe_index), truncate_timeframe_name(timeframe_index), 585, (85 + HEIGHT_RECTANGLE) + timeframe_index * HEIGHT_RECTANGLE - (1 + timeframe_index), 11, timeframe_text_color); //--- Create timeframe label

      // Create summary and indicator cells
      for(int header_index = 0; header_index < ArraySize(header_names); header_index++) { //--- Loop through headers for cells
         string cell_rectangle_name, cell_text_name;                                      //--- Declare cell name and label variables
         color cell_background = (header_index < 2) ? COLOR_LIGHT_GRAY : COLOR_BLACK;     //--- Set cell background color
         switch(header_index) {                                   //--- Select cell type
            case 0: cell_rectangle_name = BUY_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = BUY_TEXT + IntegerToString(timeframe_index); break;     //--- Buy cell
            case 1: cell_rectangle_name = SELL_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = SELL_TEXT + IntegerToString(timeframe_index); break;   //--- Sell cell
            case 2: cell_rectangle_name = RSI_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = RSI_TEXT + IntegerToString(timeframe_index); break;     //--- RSI cell
            case 3: cell_rectangle_name = STOCH_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = STOCH_TEXT + IntegerToString(timeframe_index); break; //--- Stochastic cell
            case 4: cell_rectangle_name = CCI_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = CCI_TEXT + IntegerToString(timeframe_index); break;     //--- CCI cell
            case 5: cell_rectangle_name = ADX_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = ADX_TEXT + IntegerToString(timeframe_index); break;     //--- ADX cell
            case 6: cell_rectangle_name = AO_RECTANGLE + IntegerToString(timeframe_index); cell_text_name = AO_TEXT + IntegerToString(timeframe_index); break;       //--- AO cell
         }
         int x_offset = (630 - WIDTH_TIMEFRAME) - (header_index < 2 ? header_index * WIDTH_SIGNAL : 2 * WIDTH_SIGNAL + (header_index - 2) * WIDTH_INDICATOR) + (1 + header_index);        //--- Calculate x position
         int width = (header_index < 2 ? WIDTH_SIGNAL : WIDTH_INDICATOR); //--- Set width based on cell type
         create_rectangle(cell_rectangle_name, x_offset, (75 + HEIGHT_RECTANGLE) + timeframe_index * HEIGHT_RECTANGLE - (1 + timeframe_index), width, HEIGHT_RECTANGLE, cell_background); //--- Create cell rectangle
         create_label(cell_text_name, "-/-", x_offset - width/2, (85 + HEIGHT_RECTANGLE) + timeframe_index * HEIGHT_RECTANGLE - (1 + timeframe_index), 10, COLOR_WHITE);                  //--- Create cell label
      }
   }

   // Initialize indicator arrays
   ArraySetAsSeries(rsi_values, true);                       //--- Set RSI array as timeseries
   ArraySetAsSeries(stochastic_values, true);                //--- Set Stochastic array as timeseries
   ArraySetAsSeries(cci_values, true);                       //--- Set CCI array as timeseries
   ArraySetAsSeries(adx_values, true);                       //--- Set ADX array as timeseries
   ArraySetAsSeries(ao_values, true);                        //--- Set AO array as timeseries

   return(INIT_SUCCEEDED);                                   //--- Return initialization success
}
```

In the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, we initialize the multi-timeframe scanner dashboard’s user interface. We use the "create\_rectangle" function to draw the "MAIN\_PANEL" at (632, 40) with size 617x374 pixels in “C’30,30,30’” and the "HEADER\_PANEL" at the same position with a 27-pixel height in “C’60,60,60’”. We use the "create\_label" function to add the "HEADER\_PANEL\_ICON" with a Wingdings character at (620, 54). We use the default characters in MQL5 and use the [CharToString](https://www.mql5.com/en/docs/convert/chartostring) function to convert the character code to a string. Here is the character code we used, 91, but you can use any of your liking.

![MQL5 WINGDINGS](https://c.mql5.com/2/146/C_MQL5_WINGDINGS.png)

We then create the "HEADER\_PANEL\_TEXT" with “TimeframeScanner” at (527, 52), and "CLOSE\_BUTTON" at (32, 54), but this time round we advance to using a different font and map the letter "r" to the string. Here is a visualization of the different font symbols you can use.

![SYMBOLS FONT](https://c.mql5.com/2/146/C_SYMBOL_FONTS.png)

We set up the symbol display using the "create\_rectangle" function for the "SYMBOL\_RECTANGLE" at (630, 75), sized "WIDTH\_TIMEFRAME" by "HEIGHT\_RECTANGLE", in gray. We use the "create\_label" function to place the "SYMBOL\_TEXT" at (585, 85) with the current symbol. For headers, we define the "header\_names" array with titles like “BUY” and “RSI”, looping to create "HEADER\_RECTANGLE" at y=75 with x-offsets based on "WIDTH\_SIGNAL" and "WIDTH\_INDICATOR", and "HEADER\_TEXT" labels at y=85 using the "create\_label" function.

We build the timeframe grid by looping through "timeframes\_array". We use the "create\_rectangle" function for "TIMEFRAME\_RECTANGLE" at x=630, y-offsets from (75 + "HEIGHT\_RECTANGLE") adjusted by -(1 + "timeframe\_index"), colored with "timeframe\_background". We use the "create\_label" function for "TIMEFRAME\_TEXT" with names from the "truncate\_timeframe\_name" function. For cells, we loop to create "BUY\_RECTANGLE", "RSI\_RECTANGLE", etc., with the "create\_rectangle" function, using "cell\_background", and add “-/-” labels with the "create\_label" function. We initialize indicator arrays like "rsi\_values" using the [ArraySetAsSeries](https://www.mql5.com/en/docs/array/arraysetasseries) function, setting them as time series for data handling. We return [INIT\_SUCCEEDED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode) to confirm successful initialization, establishing the dashboard’s layout and data structure. Upon compilation, we have the following outcome.

![STATIC DASHBOARD](https://c.mql5.com/2/146/Screenshot_2025-05-28_224407.png)

From the image, we can see that we have the dashboard ready. All we have to do now is add the indicator values and use them for analysis. We already have a function for the analysis, we just need to get the data. To achieve that easily, we create a function to handle all the dynamic update logic.

```
//+------------------------------------------------------------------+
//| Update indicator values                                          |
//+------------------------------------------------------------------+
void updateIndicators()                                                                               //--- Update dashboard indicators
{
   for(int timeframe_index = 0; timeframe_index < ArraySize(timeframes_array); timeframe_index++) {   //--- Loop through timeframes
      // Initialize indicator handles
      int rsi_indicator_handle = iRSI(_Symbol, timeframes_array[timeframe_index], 14, PRICE_CLOSE);   //--- Create RSI handle
      int stochastic_indicator_handle = iStochastic(_Symbol, timeframes_array[timeframe_index], 14, 3, 3, MODE_SMA, STO_LOWHIGH); //--- Create Stochastic handle
      int cci_indicator_handle = iCCI(_Symbol, timeframes_array[timeframe_index], 20, PRICE_TYPICAL); //--- Create CCI handle
      int adx_indicator_handle = iADX(_Symbol, timeframes_array[timeframe_index], 14);                //--- Create ADX handle
      int ao_indicator_handle = iAO(_Symbol, timeframes_array[timeframe_index]);                      //--- Create AO handle

      // Check for valid handles
      if(rsi_indicator_handle == INVALID_HANDLE || stochastic_indicator_handle == INVALID_HANDLE ||
         cci_indicator_handle == INVALID_HANDLE || adx_indicator_handle == INVALID_HANDLE ||
         ao_indicator_handle == INVALID_HANDLE) {                                                             //--- Check if any handle is invalid
         Print("Failed to create indicator handle for timeframe ", truncate_timeframe_name(timeframe_index)); //--- Log failure
         continue;                                                                                            //--- Skip to next timeframe
      }

      // Copy indicator values
      if(CopyBuffer(rsi_indicator_handle, 0, 0, 1, rsi_values) <= 0 ||                            //--- Copy RSI value
         CopyBuffer(stochastic_indicator_handle, 1, 0, 1, stochastic_values) <= 0 ||              //--- Copy Stochastic signal line value
         CopyBuffer(cci_indicator_handle, 0, 0, 1, cci_values) <= 0 ||                            //--- Copy CCI value
         CopyBuffer(adx_indicator_handle, 0, 0, 1, adx_values) <= 0 ||                            //--- Copy ADX value
         CopyBuffer(ao_indicator_handle, 0, 0, 1, ao_values) <= 0) {                              //--- Copy AO value
         Print("Failed to copy buffer for timeframe ", truncate_timeframe_name(timeframe_index)); //--- Log copy failure
         continue;                                                                                //--- Skip to next timeframe
      }

      // Update RSI
      color rsi_text_color = (rsi_values[0] < 30) ? clrBlue : (rsi_values[0] > 70) ? clrRed : COLOR_WHITE;         //--- Set RSI text color
      update_label(RSI_TEXT + IntegerToString(timeframe_index), DoubleToString(rsi_values[0], 2), rsi_text_color); //--- Update RSI label

      // Update Stochastic (Signal Line only)
      color stochastic_text_color = (stochastic_values[0] < 20) ? clrBlue : (stochastic_values[0] > 80) ? clrRed : COLOR_WHITE;    //--- Set Stochastic text color
      update_label(STOCH_TEXT + IntegerToString(timeframe_index), DoubleToString(stochastic_values[0], 2), stochastic_text_color); //--- Update Stochastic label

      // Update CCI
      color cci_text_color = (cci_values[0] < -100) ? clrBlue : (cci_values[0] > 100) ? clrRed : COLOR_WHITE;      //--- Set CCI text color
      update_label(CCI_TEXT + IntegerToString(timeframe_index), DoubleToString(cci_values[0], 2), cci_text_color); //--- Update CCI label

      // Update ADX
      color adx_text_color = (adx_values[0] > 25) ? clrBlue : COLOR_WHITE;                                         //--- Set ADX text color
      update_label(ADX_TEXT + IntegerToString(timeframe_index), DoubleToString(adx_values[0], 2), adx_text_color); //--- Update ADX label

      // Update AO
      color ao_text_color = (ao_values[0] > 0) ? clrGreen : (ao_values[0] < 0) ? clrRed : COLOR_WHITE;          //--- Set AO text color
      update_label(AO_TEXT + IntegerToString(timeframe_index), DoubleToString(ao_values[0], 2), ao_text_color); //--- Update AO label

      // Update Buy/Sell signals
      string buy_signal = calculate_signal_strength(rsi_values[0], stochastic_values[0], cci_values[0],
                                                   adx_values[0], ao_values[0], true);                          //--- Calculate buy signal
      string sell_signal = calculate_signal_strength(rsi_values[0], stochastic_values[0], cci_values[0],
                                                    adx_values[0], ao_values[0], false);                        //--- Calculate sell signal

      color buy_text_color = (buy_signal == "Strong Buy") ? COLOR_WHITE : COLOR_WHITE;                          //--- Set buy text color
      color buy_background = (buy_signal == "Strong Buy") ? clrGreen :
                            (buy_signal == "Buy") ? clrSeaGreen : COLOR_DARK_GRAY;                              //--- Set buy background color
      update_rectangle(BUY_RECTANGLE + IntegerToString(timeframe_index), buy_background);                       //--- Update buy rectangle
      update_label(BUY_TEXT + IntegerToString(timeframe_index), buy_signal, buy_text_color);                    //--- Update buy label

      color sell_text_color = (sell_signal == "Strong Sell") ? COLOR_WHITE : COLOR_WHITE;                       //--- Set sell text color
      color sell_background = (sell_signal == "Strong Sell") ? clrRed :
                             (sell_signal == "Sell") ? clrSalmon : COLOR_DARK_GRAY;                             //--- Set sell background color
      update_rectangle(SELL_RECTANGLE + IntegerToString(timeframe_index), sell_background);                     //--- Update sell rectangle
      update_label(SELL_TEXT + IntegerToString(timeframe_index), sell_signal, sell_text_color);                 //--- Update sell label

      // Release indicator handles
      IndicatorRelease(rsi_indicator_handle);                   //--- Release RSI handle
      IndicatorRelease(stochastic_indicator_handle);            //--- Release Stochastic handle
      IndicatorRelease(cci_indicator_handle);                   //--- Release CCI handle
      IndicatorRelease(adx_indicator_handle);                   //--- Release ADX handle
      IndicatorRelease(ao_indicator_handle);                    //--- Release AO handle
   }
}
```

To easily manage the dashboard values update, we implement the "updateIndicators" function to refresh the indicator values and signals. We loop through "timeframes\_array" using "timeframe\_index", processing each timeframe. We use the [iRSI](https://www.mql5.com/en/docs/indicators/irsi), "iStochastic", "iCCI", [iADX](https://www.mql5.com/en/docs/indicators/iadx), and "iAO" functions to create indicator handles like "rsi\_indicator\_handle" for the current symbol and timeframe, configuring parameters such as a 14-period RSI and a 20-period CCI. All the indicator settings are customizable to fit your needs, so don't limit yourself to the default values.

We then check if any handle, such as "rsi\_indicator\_handle", equals [INVALID\_HANDLE](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants), indicating a creation failure. If so, we use the "Print" function to log the error with the "truncate\_timeframe\_name" function’s output and skip to the next timeframe. We use the [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer) function to fetch the latest values into arrays like "rsi\_values", and if any fail, we log the error and continue. We update indicator displays using the "update\_label" function. For example, we set "rsi\_text\_color" based on "rsi\_values\[0\]" (blue if <30, red if >70, else "COLOR\_WHITE") and update "RSI\_TEXT" with the "DoubleToString" function’s formatted value. We repeat this for "stochastic\_values", "cci\_values", "adx\_values", and "ao\_values", applying color logic (e.g., green for positive "ao\_values").

We calculate signals using the "calculate\_signal\_strength" function, passing "rsi\_values\[0\]" and others, to get "buy\_signal" and "sell\_signal". We set "buy\_background" (e.g., green for “Strong Buy”) and use the "update\_rectangle" function for "BUY\_RECTANGLE", updating "BUY\_TEXT" with "update\_label". We do the same for "sell\_background" and "SELL\_TEXT". Finally, we use the [IndicatorRelease](https://www.mql5.com/en/docs/series/indicatorrelease) function to free handles like "rsi\_indicator\_handle", ensuring efficient resource management. The helper functions that we used, are defined as below.

```
//+------------------------------------------------------------------+
//| Update rectangle background color                                |
//+------------------------------------------------------------------+
bool update_rectangle(string object_name, color background_color)//--- Function to update rectangle color
{
   int found = ObjectFind(0, object_name);                       //--- Find rectangle object
   if(found < 0) {                                               //--- Check if object not found
      ResetLastError();                                          //--- Reset error code
      Print("UNABLE TO FIND THE RECTANGLE: ", object_name, ". ERR Code: ", GetLastError()); //--- Log error
      return(false);                                             //--- Return failure
   }
   ObjectSetInteger(0, object_name, OBJPROP_BGCOLOR, background_color); //--- Set background color

   ChartRedraw(0);                                               //--- Redraw chart
   return(true);                                                 //--- Return success
}

//+------------------------------------------------------------------+
//| Update label text and color                                      |
//+------------------------------------------------------------------+
bool update_label(string object_name, string text, color text_color) //--- Function to update label
{
   int found = ObjectFind(0, object_name);                       //--- Find label object
   if(found < 0) {                                               //--- Check if object not found
      ResetLastError();                                          //--- Reset error code
      Print("UNABLE TO FIND THE LABEL: ", object_name, ". ERR Code: ", GetLastError()); //--- Log error
      return(false);                                             //--- Return failure
   }
   ObjectSetString(0, object_name, OBJPROP_TEXT, text);          //--- Set label text
   ObjectSetInteger(0, object_name, OBJPROP_COLOR, text_color);  //--- Set text color

   ChartRedraw(0);                                               //--- Redraw chart
   return(true);                                                 //--- Return success
}
```

We define the "update\_rectangle" function, taking "object\_name" and "background\_color" as parameters to modify a rectangle’s appearance. We use the [ObjectFind](https://www.mql5.com/en/docs/objects/ObjectFind) function to locate the rectangle, storing the result in "found". If "found" is less than 0, indicating the object is missing, we use the [ResetLastError](https://www.mql5.com/en/docs/common/ResetLastError) function, log the error with the "Print" function and "GetLastError", and return false. We update the rectangle’s background by using the "ObjectSetInteger" function to set "OBJPROP\_BGCOLOR" to "background\_color". We use the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function to refresh the chart and return true for success. For text updates, we define the "update\_label" function with parameters "object\_name", "text", and "text\_color".

We use the "ObjectFind" function to check for the label’s existence, and if "found" is negative, we use the "ResetLastError" function, log the error with the "Print" function, and return false. We use the [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) function to set "OBJPROP\_TEXT" to "text" and the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/ObjectSetInteger) function to set "OBJPROP\_COLOR" to "text\_color". We use the "ChartRedraw" function to update the chart and return true, enabling dynamic label updates. We can now call the update function on the tick to make updates to the dashboard.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()                                                    //--- Handle tick events
{
   if (panel_is_visible) {                                       //--- Check if panel is visible
      updateIndicators();                                        //--- Update indicators
   }
}
```

Here, in the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler, we just call the "updateIndicators" function if the panel is visible to apply the updates. We finally need to delete the objects that we created so that they are removed from that chart when we no longer need them.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)                                  //--- Deinitialize EA
{
   ObjectDelete(0, MAIN_PANEL);                                  //--- Delete main panel
   ObjectDelete(0, HEADER_PANEL);                                //--- Delete header panel
   ObjectDelete(0, HEADER_PANEL_ICON);                           //--- Delete header icon
   ObjectDelete(0, HEADER_PANEL_TEXT);                           //--- Delete header title
   ObjectDelete(0, CLOSE_BUTTON);                                //--- Delete close button

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

   ChartRedraw(0);                                               //--- Redraw chart
}
```

Finally, we implement the cleanup process using the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) function, which runs when the Expert Advisor is removed. We use the [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete) function to remove individual UI elements, starting with the "MAIN\_PANEL" rectangle, followed by the "HEADER\_PANEL", "HEADER\_PANEL\_ICON", "HEADER\_PANEL\_TEXT", and "CLOSE\_BUTTON", ensuring the main panel and header components are cleared from the chart.

We systematically remove all dashboard objects by using the [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/objectdeleteall) function for each element type. We delete all rectangles and labels associated with "SYMBOL\_RECTANGLE" and "SYMBOL\_TEXT", "TIMEFRAME\_RECTANGLE" and "TIMEFRAME\_TEXT", and "HEADER\_RECTANGLE" and "HEADER\_TEXT", clearing the symbol, timeframe, and header displays. We also remove indicator-related objects, including "RSI\_RECTANGLE", "STOCH\_RECTANGLE", "CCI\_RECTANGLE", "ADX\_RECTANGLE", and "AO\_RECTANGLE", along with their respective text labels like "RSI\_TEXT".

We complete the cleanup by using the "ObjectsDeleteAll" function to delete all "BUY\_RECTANGLE" and "SELL\_RECTANGLE" objects, along with their "BUY\_TEXT" and "SELL\_TEXT" labels, removing all signal-related elements. Finally, we use the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function to refresh the chart, ensuring a clean visual state after deinitialization. Finally, we need to take care of the cancel button so that when clicked, we close the dashboard and disable further updates.

```
//+------------------------------------------------------------------+
//| Expert chart event handler                                       |
//+------------------------------------------------------------------+
void OnChartEvent(const int       event_id,                      //--- Event ID
                  const long&     long_param,                    //--- Long parameter
                  const double&   double_param,                  //--- Double parameter
                  const string&   string_param)                  //--- String parameter
{
   if (event_id == CHARTEVENT_OBJECT_CLICK) {                    //--- Check for object click event
      if (string_param == CLOSE_BUTTON) {                        //--- Check if close button clicked
         Print("Closing the panel now");                         //--- Log panel closure
         PlaySound("alert.wav");                                 //--- Play alert sound
         panel_is_visible = false;                               //--- Hide panel

         ObjectDelete(0, MAIN_PANEL);                            //--- Delete main panel
         ObjectDelete(0, HEADER_PANEL);                          //--- Delete header panel
         ObjectDelete(0, HEADER_PANEL_ICON);                     //--- Delete header icon
         ObjectDelete(0, HEADER_PANEL_TEXT);                     //--- Delete header title
         ObjectDelete(0, CLOSE_BUTTON);                          //--- Delete close button

         ObjectsDeleteAll(0, SYMBOL_RECTANGLE);                  //--- Delete all symbol rectangles
         ObjectsDeleteAll(0, SYMBOL_TEXT);                       //--- Delete all symbol labels
         ObjectsDeleteAll(0, TIMEFRAME_RECTANGLE);               //--- Delete all timeframe rectangles
         ObjectsDeleteAll(0, TIMEFRAME_TEXT);                    //--- Delete all timeframe labels
         ObjectsDeleteAll(0, HEADER_RECTANGLE);                  //--- Delete all header rectangles
         ObjectsDeleteAll(0, HEADER_TEXT);                       //--- Delete all header labels
         ObjectsDeleteAll(0, RSI_RECTANGLE);                     //--- Delete all RSI rectangles
         ObjectsDeleteAll(0, RSI_TEXT);                          //--- Delete all RSI labels
         ObjectsDeleteAll(0, STOCH_RECTANGLE);                   //--- Delete all Stochastic rectangles
         ObjectsDeleteAll(0, STOCH_TEXT);                        //--- Delete all Stochastic labels
         ObjectsDeleteAll(0, CCI_RECTANGLE);                     //--- Delete all CCI rectangles
         ObjectsDeleteAll(0, CCI_TEXT);                          //--- Delete all CCI labels
         ObjectsDeleteAll(0, ADX_RECTANGLE);                     //--- Delete all ADX rectangles
         ObjectsDeleteAll(0, ADX_TEXT);                          //--- Delete all ADX labels
         ObjectsDeleteAll(0, AO_RECTANGLE);                      //--- Delete all AO rectangles
         ObjectsDeleteAll(0, AO_TEXT);                           //--- Delete all AO labels
         ObjectsDeleteAll(0, BUY_RECTANGLE);                     //--- Delete all buy rectangles
         ObjectsDeleteAll(0, BUY_TEXT);                          //--- Delete all buy labels
         ObjectsDeleteAll(0, SELL_RECTANGLE);                    //--- Delete all sell rectangles
         ObjectsDeleteAll(0, SELL_TEXT);                         //--- Delete all sell labels

         ChartRedraw(0);                                         //--- Redraw chart
      }
   }
}
```

In the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler, we listen to object clicks when the event id is [CHARTEVENT\_OBJECT\_CLICK](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents), and the object clicked is the cancel button, and we play an alert sound using the [PlaySound](https://www.mql5.com/en/docs/common/playsound) function to alert the user that the panel is being disabled, then disable the visibility of the panel, and use the same logic we used to clear the chart on [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) to clear the dashboard. Upon compilation, we have the following outcome.

![UPDATED DASHBOARD](https://c.mql5.com/2/146/Screenshot_2025-05-28_231802.png)

From the image, we can see that we have the dashboard updated with the indicator data and trading directions indicated. What now remains is testing the workability of the project, and that is handledin in the section following this.

### Backtesting

We did the testing and below is the compiled visualization in a single Graphics Interchange Format (GIF) bitmap image format.

![TESTING GIF](https://c.mql5.com/2/146/Timeframes_SCANNER_GIF.gif)

### Conclusion

In conclusion, we’ve developed a multi-timeframe scanner dashboard in MQL5, integrating a structured grid layout, real-time indicator signals, and an interactive close button to enhance strategic trading decisions. We’ve shown the design and implementation of these features, ensuring their effectiveness through robust initialization and dynamic updates tailored to our trading requirements. You can adapt this dashboard to fit your preferences, greatly improving your ability to monitor and act on market signals across multiple timeframes.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18319.zip "Download all attachments in the single ZIP archive")

[TimeframeScanner\_Dashboard\_EA.mq5](https://www.mql5.com/en/articles/download/18319/timeframescanner_dashboard_ea.mq5 "Download TimeframeScanner_Dashboard_EA.mq5")(31.59 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/488209)**

![Introduction to MQL5 (Part 17): Building Expert Advisors for Trend Reversals](https://c.mql5.com/2/147/18259-introduction-to-mql5-part-17-logo.png)[Introduction to MQL5 (Part 17): Building Expert Advisors for Trend Reversals](https://www.mql5.com/en/articles/18259)

This article teaches beginners how to build an Expert Advisor (EA) in MQL5 that trades based on chart pattern recognition using trend line breakouts and reversals. By learning how to retrieve trend line values dynamically and compare them with price action, readers will be able to develop EAs capable of identifying and trading chart patterns such as ascending and descending trend lines, channels, wedges, triangles, and more.

![Neural Networks in Trading: Contrastive Pattern Transformer](https://c.mql5.com/2/98/Atom-Motif_Contrastive_Transformer___LOGO.png)[Neural Networks in Trading: Contrastive Pattern Transformer](https://www.mql5.com/en/articles/16163)

The Contrastive Transformer is designed to analyze markets both at the level of individual candlesticks and based on entire patterns. This helps improve the quality of market trend modeling. Moreover, the use of contrastive learning to align representations of candlesticks and patterns fosters self-regulation and improves the accuracy of forecasts.

![Data Science and ML (Part 42): Forex Time series Forecasting using ARIMA in Python, Everything you need to Know](https://c.mql5.com/2/147/18247-data-science-and-ml-part-42-logo.png)[Data Science and ML (Part 42): Forex Time series Forecasting using ARIMA in Python, Everything you need to Know](https://www.mql5.com/en/articles/18247)

ARIMA, short for Auto Regressive Integrated Moving Average, is a powerful traditional time series forecasting model. With the ability to detect spikes and fluctuations in a time series data, this model can make accurate predictions on the next values. In this article, we are going to understand what is it, how it operates, what you can do with it when it comes to predicting the next prices in the market with high accuracy and much more.

![Price Action Analysis Toolkit Development (Part 25): Dual EMA Fractal Breaker](https://c.mql5.com/2/147/18297-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 25): Dual EMA Fractal Breaker](https://www.mql5.com/en/articles/18297)

Price action is a fundamental approach for identifying profitable trading setups. However, manually monitoring price movements and patterns can be challenging and time-consuming. To address this, we are developing tools that analyze price action automatically, providing timely signals whenever potential opportunities are detected. This article introduces a robust tool that leverages fractal breakouts alongside EMA 14 and EMA 200 to generate reliable trading signals, helping traders make informed decisions with greater confidence.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=vxrjshtpcvqtlieivvuqcrbnhfadvwqy&ssn=1769093774809921707&ssn_dr=0&ssn_sr=0&fv_date=1769093774&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18319&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Trading%20Tools%20(Part%203)%3A%20Building%20a%20Multi-Timeframe%20Scanner%20Dashboard%20for%20Strategic%20Trading%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909377458126561&fz_uniq=5049497401334541454&sv=2552)

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