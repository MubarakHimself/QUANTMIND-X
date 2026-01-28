---
title: Automating Trading Strategies in MQL5 (Part 16): Midnight Range Breakout with Break of Structure (BoS) Price Action
url: https://www.mql5.com/en/articles/17876
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 7
scraped_at: 2026-01-22T17:47:26.345568
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/17876&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049381183814478459)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 15)](https://www.mql5.com/en/articles/17865), we automated a trading strategy leveraging the [Cypher harmonic pattern](https://www.mql5.com/go?link=https://howtotrade.com/chart-patterns/cypher-harmonic-pattern/ "https://howtotrade.com/chart-patterns/cypher-harmonic-pattern/") to capture market reversals. Now, in Part 16, we focus on automating the Midnight Range Breakout with Break of Structure strategy in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5), developing an Expert Advisor that identifies the midnight to 6 AM price range, detects Break of Structure (BoS), and executes trades. We will cover the following topics:

1. [Understanding the Midnight Range Breakout with Break of Structure Strategy](https://www.mql5.com/en/articles/17876#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/17876#para2)
3. [Backtesting](https://www.mql5.com/en/articles/17876#para3)
4. [Conclusion](https://www.mql5.com/en/articles/17876#para4)

By the end of this article, you will have a fully functional [MQL5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") program that visualizes key price levels, confirms breakouts, and executes trades with defined risk parameters—let us begin!

### Understanding the Midnight Range Breakout with Break of Structure Strategy

The Midnight Range Breakout with Break of Structure strategy capitalizes on the low-volatility price range formed between midnight and 6 AM, using the highest and lowest prices as breakout levels, while requiring Break of Structure confirmation to validate trade signals. Break of Structure identifies trend shifts by detecting when the price surpasses a swing high (bullish) or falls below a swing low (bearish), filtering out false breakouts and aligning trades with market momentum. This approach suits markets during session transitions, such as London’s opening, or any other of your liking. Still, it requires timezone alignment and caution during high-impact news events to avoid whipsaws.

Our implementation plan will involve creating an MQL5 Expert Advisor to automate the strategy by calculating the midnight to 6 AM range, monitoring for breakouts within a set time window, and confirming them with Break of Structure on a specified timeframe, usually 5, 10, or 15 minutes, so we will have this in input so the user can choose dynamically. The system will execute trades with stop-loss and take-profit levels derived from the range, visualize key levels on the chart for clarity, and ensure robust risk management to maintain consistent performance across market conditions. In a nutshell, this is what we mean.

![STRATEGY IN A NUTSHELL](https://c.mql5.com/2/136/Screenshot_2025-04-21_193816.png)

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Indicators folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some [global variables](https://www.mql5.com/en/docs/basis/variables/global) that we will use throughout the program.

```
//+------------------------------------------------------------------+
//|                   Midnight Range Break of Structure Breakout.mq5 |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//|                                   https://t.me/Forex_Algo_Trader |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Allan Munene Mutiiria."
#property link      "https://t.me/Forex_Algo_Trader"
#property version   "1.00"

#include <Trade/Trade.mqh> //--- Include the Trade library for handling trade operations
CTrade obj_Trade;          //--- Create an instance of the CTrade class for trade execution

double maximum_price = -DBL_MAX;      //--- Initialize the maximum price variable to negative infinity
double minimum_price = DBL_MAX;       //--- Initialize the minimum price variable to positive infinity
datetime maximum_time, minimum_time;  //--- Declare variables to store the times of maximum and minimum prices
bool isHaveDailyRange_Prices = false; //--- Initialize flag to indicate if daily range prices are calculated
bool isHaveRangeBreak = false;        //--- Initialize flag to indicate if a range breakout has occurred
bool isTakenTrade = false;            //--- Initialize flag to indicate if a trade is taken for the current day

#define RECTANGLE_PREFIX "RANGE RECTANGLE " //--- Define a prefix for rectangle object names
#define UPPER_LINE_PREFIX "UPPER LINE "     //--- Define a prefix for upper line object names
#define LOWER_LINE_PREFIX "LOWER LINE "     //--- Define a prefix for lower line object names

// bos
input ENUM_TIMEFRAMES timeframe_bos = PERIOD_M5; // Input the timeframe for Break of Structure (BoS) analysis
```

Here, we begin implementing the strategy by setting up the foundational components of the program. We include the <Trade/Trade.mqh> library to enable trade operations and instantiate the "CTrade" class as the "obj\_Trade" object, which will handle trade execution, such as opening buy or sell positions with specified parameters.

We define several [global variables](https://www.mql5.com/en/docs/basis/variables/global) to track critical data for the strategy. The "maximum\_price" and "minimum\_price" variables, initialized to [-DBL\_MAX](https://www.mql5.com/en/docs/constants/namedconstants/typeconstants) and [DBL\_MAX](https://www.mql5.com/en/docs/constants/namedconstants/typeconstants) respectively, store the highest and lowest prices within the midnight to 6 AM range, allowing us to identify the range boundaries. The "maximum\_time" and "minimum\_time" variables, of type datetime, record the times when these extreme prices occur, which is essential for visualizing the range on the chart. We also use boolean flags: "isHaveDailyRange\_Prices" to indicate whether the daily range has been calculated, "isHaveRangeBreak" to track if a breakout has occurred, and "isTakenTrade" to ensure only one trade is taken per day, preventing overtrading.

To facilitate chart visualization, we [define](https://www.mql5.com/en/docs/basis/preprosessor/constant) constants for object naming: "RECTANGLE\_PREFIX" as "RANGE RECTANGLE ", "UPPER\_LINE\_PREFIX" as "UPPER LINE ", and "LOWER\_LINE\_PREFIX" as "LOWER LINE ". These prefixes ensure unique names for chart objects like rectangles and lines, which will mark the range and breakout levels, making the strategy’s actions visually clear. Additionally, we introduce a user [input](https://www.mql5.com/en/docs/basis/variables/inputvariables) parameter, "timeframe\_bos", set to [PERIOD\_M5](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes) by default, allowing traders to specify the timeframe for Break of Structure analysis, such as the 5-minute chart, to detect swing highs and lows. With that, we are all set. What need to do is define two functions to enable us to control trading instances between new days and new bars.

```
//+------------------------------------------------------------------+
//| Function to check for a new bar                                  |
//+------------------------------------------------------------------+
bool isNewBar(){ //--- Define a function to detect a new bar on the current timeframe
   static int prevBars = 0;                //--- Store the previous number of bars
   int currBars = iBars(_Symbol,_Period);  //--- Get the current number of bars
   if (prevBars==currBars) return (false); //--- Return false if no new bar has formed
   prevBars = currBars;                    //--- Update the previous bar count
   return (true);                          //--- Return true if a new bar has formed
}

//+------------------------------------------------------------------+
//| Function to check for a new day                                  |
//+------------------------------------------------------------------+
bool isNewDay(){ //--- Define a function to detect a new trading day
   bool newDay = false;  //--- Initialize the new day flag

   MqlDateTime Str_DateTime;                 //--- Declare a structure to hold date and time information
   TimeToStruct(TimeCurrent(),Str_DateTime); //--- Convert the current time to the structure

   static int prevDay = 0;         //--- Store the previous day's date
   int currDay = Str_DateTime.day; //--- Get the current day's date

   if (prevDay == currDay){ //--- Check if the current day is the same as the previous day
      newDay = false;       //--- Set the flag to false (no new day)
   }
   else if (prevDay != currDay){ //--- Check if a new day has started
      Print("WE HAVE A NEW DAY WITH DATE ",currDay); //--- Log the new day
      prevDay = currDay;                             //--- Update the previous day
      newDay = true;                                 //--- Set the flag to true (new day)
   }

   return (newDay); //--- Return the new day status
}
```

Here, we implement the "isNewBar" and "isNewDay" functions to synchronize our Midnight Range Breakout with the Break of Structure strategy in MQL5 with market timing. In "isNewBar", we track bar counts using a static "prevBars" and the [iBars](https://www.mql5.com/en/docs/series/ibars) function with [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol) and [\_Period](https://www.mql5.com/en/docs/predefined/_period), returning true when a new bar forms to trigger price updates. In "isNewDay", we use a [MqlDateTime](https://www.mql5.com/en/docs/constants/structures/mqldatetime) structure, the [TimeToStruct](https://www.mql5.com/en/docs/dateandtime/timetostruct) function with [TimeCurrent](https://www.mql5.com/en/docs/dateandtime/timecurrent), and a static "prevDay" to detect new days, resetting range calculations when "currDay" changes, logging via [Print](https://www.mql5.com/en/docs/common/print). Armed with these functions, we can define the logic directly in the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){
   //---
   static datetime midnight = iTime(_Symbol,PERIOD_D1,0);            //--- Store the current day's midnight time
   static datetime sixAM = midnight + 6 * 3600;                      //--- Calculate 6 AM time by adding 6 hours to midnight
   static datetime scanBarTime = sixAM + 1 * PeriodSeconds(_Period); //--- Set the time of the next bar after 6 AM for scanning
   static double midnight_price = iClose(_Symbol,PERIOD_D1,0);       //--- Store the closing price at midnight

   static datetime validBreakTime_start = scanBarTime;               //--- Set the start time for valid breakout detection
   static datetime validBreakTime_end = midnight + (6+5) * 3600;     //--- Set the end time for valid breakouts to 11 AM

   if (isNewDay()){ //--- Check if a new trading day has started
      midnight = iTime(_Symbol,PERIOD_D1,0);                          //--- Update midnight time for the new day
      sixAM = midnight + 6 * 3600;                                    //--- Recalculate 6 AM time for the new day
      scanBarTime = sixAM + 1 * PeriodSeconds(_Period);               //--- Update the scan bar time to the next bar after 6 AM
      midnight_price = iClose(_Symbol,PERIOD_D1,0);                   //--- Update the midnight closing price
      Print("Midnight price = ",midnight_price,", Time = ",midnight); //--- Log the midnight price and time

      validBreakTime_start = scanBarTime;           //--- Reset the start time for valid breakouts
      validBreakTime_end = midnight + (6+5) * 3600; //--- Reset the end time for valid breakouts to 11 AM

      maximum_price = -DBL_MAX; //--- Reset the maximum price to negative infinity
      minimum_price = DBL_MAX;  //--- Reset the minimum price to positive infinity

      isHaveDailyRange_Prices = false; //--- Reset the flag indicating daily range calculation
      isHaveRangeBreak = false;        //--- Reset the flag indicating a range breakout
      isTakenTrade = false;            //--- Reset the flag indicating a trade is taken
   }
}
```

Here, we develop the core logic of our strategy within the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) function, the main event handler in our program that executes on every price tick. We initialize static variables to track critical timings: "midnight" stores the current day’s start time using the [iTime](https://www.mql5.com/en/docs/series/itime) function with [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol), [PERIOD\_D1](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes), and index 0; "sixAM" is calculated by adding 6 hours (21,600 seconds) to "midnight"; "scanBarTime" sets the time of the next bar after 6 AM using the [PeriodSeconds](https://www.mql5.com/en/docs/common/periodseconds) function with "\_Period"; and "midnight\_price" captures the day’s closing price at midnight via the [iClose](https://www.mql5.com/en/docs/series/iclose) function. We also define "validBreakTime\_start" as "scanBarTime" and "validBreakTime\_end" as 11 AM (midnight plus 11 hours) to establish a time window for valid breakouts.

When a new trading day begins, detected by our "isNewDay" function, we update these timing variables to reflect the new day’s data, ensuring our range calculations remain current. We reset "midnight", "sixAM", "scanBarTime", and "midnight\_price" using the same [iTime](https://www.mql5.com/en/docs/series/itime) and [iClose](https://www.mql5.com/en/docs/series/iclose) functions, and log the midnight details with the [Print](https://www.mql5.com/en/docs/common/print) function for debugging. We also reset "validBreakTime\_start" and "validBreakTime\_end" for the new breakout window, and reinitialized "maximum\_price" to "-DBL\_MAX", "minimum\_price" to "DBL\_MAX", and the flags "isHaveDailyRange\_Prices", "isHaveRangeBreak", and "isTakenTrade" to false, preparing the EA to calculate fresh midnight to 6 AM range and monitor for new breakouts. We can now check the calculation of the time ranges.

```
if (isNewBar()){ //--- Check if a new bar has formed on the current timeframe
   datetime currentBarTime = iTime(_Symbol,_Period,0); //--- Get the time of the current bar

   if (currentBarTime == scanBarTime && !isHaveDailyRange_Prices){              //--- Check if it's time to scan for daily range and range is not yet calculated
      Print("WE HAVE ENOUGH BARS DATA FOR DOCUMENTATION. MAKE THE EXTRACTION"); //--- Log that the scan for daily range is starting
      int total_bars = int((sixAM - midnight)/PeriodSeconds(_Period))+1;        //--- Calculate the number of bars from midnight to 6 AM
      Print("Total Bars for scan = ",total_bars);                               //--- Log the total number of bars to scan
      int highest_price_bar_index = -1;                                         //--- Initialize the index of the bar with the highest price
      int lowest_price_bar_index = -1;                                          //--- Initialize the index of the bar with the lowest price

      for (int i=1; i<=total_bars ; i++){ //--- Loop through each bar from midnight to 6 AM
         double open_i = open(i);   //--- Get the open price of the i-th bar
         double close_i = close(i); //--- Get the close price of the i-th bar

         double highest_price_i = (open_i > close_i) ? open_i : close_i; //--- Determine the highest price (open or close) of the bar
         double lowest_price_i = (open_i < close_i) ? open_i : close_i;  //--- Determine the lowest price (open or close) of the bar

         if (highest_price_i > maximum_price){ //--- Check if the bar's highest price exceeds the current maximum
            maximum_price = highest_price_i;   //--- Update the maximum price
            highest_price_bar_index = i;       //--- Store the bar index of the maximum price
            maximum_time = time(i);            //--- Store the time of the maximum price
         }
         if (lowest_price_i < minimum_price){ //--- Check if the bar's lowest price is below the current minimum
            minimum_price = lowest_price_i; //--- Update the minimum price
            lowest_price_bar_index = i;     //--- Store the bar index of the minimum price
            minimum_time = time(i);         //--- Store the time of the minimum price
         }
      }
      Print("Maximum Price = ",maximum_price,", Bar index = ",highest_price_bar_index,", Time = ",maximum_time); //--- Log the maximum price, its bar index, and time
      Print("Minimum Price = ",minimum_price,", Bar index = ",lowest_price_bar_index,", Time = ",minimum_time);  //--- Log the minimum price, its bar index, and time

      isHaveDailyRange_Prices = true; //--- Set the flag to indicate that the daily range is calculated
   }
}
```

To calculate the midnight to 6 AM price range when a new bar forms, we use the "isNewBar" function to check for a new bar, then retrieve the bar’s time with [iTime](https://www.mql5.com/en/docs/series/itime) for [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol), [\_Period](https://www.mql5.com/en/docs/predefined/_period), index 0, storing it in "currentBarTime". If "currentBarTime" equals "scanBarTime" and "isHaveDailyRange\_Prices" is false, we log the range scan starting with [Print](https://www.mql5.com/en/docs/common/print), compute "total\_bars" using [PeriodSeconds](https://www.mql5.com/en/docs/common/periodseconds) for [\_Period](https://www.mql5.com/en/docs/predefined/_period), and loop through bars to find the highest and lowest prices with "open" and "close" functions, updating "maximum\_price", "minimum\_price", "maximum\_time", "minimum\_time", and their indices. We log results and set "isHaveDailyRange\_Prices" to true, enabling breakout monitoring.

For simplicity, we used predefined functions to get the prices and they are as follows.

```
//+------------------------------------------------------------------+
//| Helper functions for price and time data                         |
//+------------------------------------------------------------------+
double open(int index){return (iOpen(_Symbol,_Period,index));}   //--- Return the open price of the specified bar index on the current timeframe
double high(int index){return (iHigh(_Symbol,_Period,index));}   //--- Return the high price of the specified bar index on the current timeframe
double low(int index){return (iLow(_Symbol,_Period,index));}     //--- Return the low price of the specified bar index on the current timeframe
double close(int index){return (iClose(_Symbol,_Period,index));} //--- Return the close price of the specified bar index on the current timeframe
datetime time(int index){return (iTime(_Symbol,_Period,index));} //--- Return the time of the specified bar index on the current timeframe

double high(int index,ENUM_TIMEFRAMES tf_bos){return (iHigh(_Symbol,tf_bos,index));}   //--- Return the high price of the specified bar index on the BoS timeframe
double low(int index,ENUM_TIMEFRAMES tf_bos){return (iLow(_Symbol,tf_bos,index));}     //--- Return the low price of the specified bar index on the BoS timeframe
datetime time(int index,ENUM_TIMEFRAMES tf_bos){return (iTime(_Symbol,tf_bos,index));} //--- Return the time of the specified bar index on the BoS timeframe
```

We implement helper functions to efficiently retrieve price and time data by defining the "open", "high", "low", "close", and "time" functions, each taking an "index" parameter, to fetch data for the current timeframe using iOpen, " [iHigh](https://www.mql5.com/en/docs/series/ihigh)", "iLow", "iClose", and [iTime](https://www.mql5.com/en/docs/series/itime) respectively, with "\_Symbol" and [\_Period](https://www.mql5.com/en/docs/predefined/_period) as inputs, returning the corresponding open price, high price, low price, close price, or bar time for the specified bar index.

Additionally, we overload the "high", "low", and "time" functions to accept an [ENUM\_TIMEFRAMES](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes) "tf\_bos" parameter, enabling us to retrieve the high price, low price, or bar time for the Break of Structure timeframe using "iHigh", "iLow", and "iTime" with [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol) and "tf\_bos". Since we did define the range, let us visualize it on the chart. For that, we will need to define some extra helper functions too.

```
//+------------------------------------------------------------------+
//| Function to create a rectangle object                            |
//+------------------------------------------------------------------+
void create_Rectangle(string objName,datetime time1,double price1,
               datetime time2,double price2,color clr){ //--- Define a function to draw a rectangle on the chart
   if (ObjectFind(0,objName) < 0){                                       //--- Check if the rectangle object does not already exist
      ObjectCreate(0,objName,OBJ_RECTANGLE,0,time1,price1,time2,price2); //--- Create a rectangle object with specified coordinates

      ObjectSetInteger(0,objName,OBJPROP_TIME,0,time1);  //--- Set the first time coordinate of the rectangle
      ObjectSetDouble(0,objName,OBJPROP_PRICE,0,price1); //--- Set the first price coordinate of the rectangle
      ObjectSetInteger(0,objName,OBJPROP_TIME,1,time2);  //--- Set the second time coordinate of the rectangle
      ObjectSetDouble(0,objName,OBJPROP_PRICE,1,price2); //--- Set the second price coordinate of the rectangle
      ObjectSetInteger(0,objName,OBJPROP_FILL,true);     //--- Enable filling the rectangle with color
      ObjectSetInteger(0,objName,OBJPROP_COLOR,clr);     //--- Set the color of the rectangle
      ObjectSetInteger(0,objName,OBJPROP_BACK,false);    //--- Ensure the rectangle is drawn in the foreground

      ChartRedraw(0); //--- Redraw the chart to display the rectangle
   }
}

//+------------------------------------------------------------------+
//| Function to create a line object with text                       |
//+------------------------------------------------------------------+
void create_Line(string objName,datetime time1,double price1,
               datetime time2,double price2,int width,color clr,string text){ //--- Define a function to draw a trend line with text
   if (ObjectFind(0,objName) < 0){                                   //--- Check if the line object does not already exist
      ObjectCreate(0,objName,OBJ_TREND,0,time1,price1,time2,price2); //--- Create a trend line object with specified coordinates

      ObjectSetInteger(0,objName,OBJPROP_TIME,0,time1);  //--- Set the first time coordinate of the line
      ObjectSetDouble(0,objName,OBJPROP_PRICE,0,price1); //--- Set the first price coordinate of the line
      ObjectSetInteger(0,objName,OBJPROP_TIME,1,time2);  //--- Set the second time coordinate of the line
      ObjectSetDouble(0,objName,OBJPROP_PRICE,1,price2); //--- Set the second price coordinate of the line
      ObjectSetInteger(0,objName,OBJPROP_WIDTH,width);   //--- Set the width of the line
      ObjectSetInteger(0,objName,OBJPROP_COLOR,clr);     //--- Set the color of the line
      ObjectSetInteger(0,objName,OBJPROP_BACK,false);    //--- Ensure the line is drawn in the foreground

      long scale = 0; //--- Initialize a variable to store the chart scale
      if(!ChartGetInteger(0,CHART_SCALE,0,scale)){                                   //--- Attempt to get the chart scale
         Print("UNABLE TO GET THE CHART SCALE. DEFAULT OF ",scale," IS CONSIDERED"); //--- Log if the chart scale cannot be retrieved
      }

      int fontsize = 11; //--- Set the default font size for the text
      if (scale==0){fontsize=5;} //--- Adjust font size for minimized chart scale
      else if (scale==1){fontsize=6;}  //--- Adjust font size for scale 1
      else if (scale==2){fontsize=7;}  //--- Adjust font size for scale 2
      else if (scale==3){fontsize=9;}  //--- Adjust font size for scale 3
      else if (scale==4){fontsize=11;} //--- Adjust font size for scale 4
      else if (scale==5){fontsize=13;} //--- Adjust font size for maximized chart scale

      string txt = " Right Price";         //--- Define the text suffix for the price label
      string objNameDescr = objName + txt; //--- Create a unique name for the text object
      ObjectCreate(0,objNameDescr,OBJ_TEXT,0,time2,price2);        //--- Create a text object at the line's end
      ObjectSetInteger(0,objNameDescr,OBJPROP_COLOR,clr);          //--- Set the color of the text
      ObjectSetInteger(0,objNameDescr,OBJPROP_FONTSIZE,fontsize);  //--- Set the font size of the text
      ObjectSetInteger(0,objNameDescr,OBJPROP_ANCHOR,ANCHOR_LEFT); //--- Set the text anchor to the left
      ObjectSetString(0,objNameDescr,OBJPROP_TEXT," "+text);       //--- Set the text content (price value)
      ObjectSetString(0,objNameDescr,OBJPROP_FONT,"Calibri");      //--- Set the font type to Calibri

      ChartRedraw(0); //--- Redraw the chart to display the line and text
   }
}
```

To visualize the range, we define two functions. In the "create\_Rectangle" function, we draw a filled rectangle to represent the midnight to 6 AM price range, using parameters "objName", "time1", "price1", "time2", "price2", and "clr" for customization. We first check if the object doesn’t exist using the [ObjectFind](https://www.mql5.com/en/docs/objects/ObjectFind) function with chart ID 0, ensuring we avoid duplicates.

If absent, we create the rectangle with [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) using [OBJ\_RECTANGLE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object), setting its coordinates with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) for OBJPROP\_TIME and [ObjectSetDouble](https://www.mql5.com/en/docs/objects/objectsetdouble) for "OBJPROP\_PRICE". We enable color filling with "ObjectSetInteger" for "OBJPROP\_FILL", set the rectangle’s color, and ensure it appears in the foreground by setting "OBJPROP\_BACK" to false, followed by a [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to update the chart display.

In the "create\_Line" function, we draw a trend line with a descriptive text label to mark range boundaries, accepting parameters "objName", "time1", "price1", "time2", "price2", "width", "clr", and "text". We verify the line’s absence with "ObjectFind", then create it using "ObjectCreate" with [OBJ\_TREND](https://www.mql5.com/en/docs/constants/objectconstants/enum_object), configuring coordinates, line width, and color via "ObjectSetInteger" and "ObjectSetDouble". To ensure readable text, we retrieve the chart scale with "ChartGetInteger", logging any failures with "Print", and adjust the font size dynamically based on the scale (from 5 to 13).

We create a text object with "ObjectCreate" using "OBJ\_TEXT", named "objNameDescr", and set its properties with "ObjectSetInteger" for color, font size, and left anchor, and [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) for the "Calibri" font and price text, before redrawing the chart with "ChartRedraw". With these functions, we can call them when we define the ranges.

```
create_Rectangle(RECTANGLE_PREFIX+TimeToString(maximum_time),maximum_time,maximum_price,minimum_time,minimum_price,clrBlue);                       //--- Draw a rectangle to mark the daily range
create_Line(UPPER_LINE_PREFIX+TimeToString(midnight),midnight,maximum_price,sixAM,maximum_price,3,clrBlack,DoubleToString(maximum_price,_Digits)); //--- Draw the upper line for the range
create_Line(LOWER_LINE_PREFIX+TimeToString(midnight),midnight,minimum_price,sixAM,minimum_price,3,clrRed,DoubleToString(minimum_price,_Digits));   //--- Draw the lower line for the range
```

We complete the visualization of the midnight to 6 AM price range by calling "create\_Rectangle" with "RECTANGLE\_PREFIX+TimeToString(maximum\_time)", "maximum\_time", "maximum\_price", "minimum\_time", "minimum\_price", and "clrBlue" to draw a rectangle marking the range. We then use "create\_Line" twice: first for the upper line with "UPPER\_LINE\_PREFIX+ [TimeToString](https://www.mql5.com/en/docs/convert/timetostring)(midnight)", "midnight", "maximum\_price", "sixAM", width 3, "clrBlack", and " [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring)(maximum\_price,\_Digits)"; and second for the lower line with "LOWER\_LINE\_PREFIX", "minimum\_price", "clrRed", and matching parameters. Here is the current outcome.

![RANGE WITH PRICES](https://c.mql5.com/2/136/Screenshot_2025-04-21_225345.png)

From the image, we can see that we have accurately visualized the range. The next thing that we need to do is track that there is a break within a predefined time range and visualize it on the chart too. We will need a custom function to show the break on the chart.

```
//+------------------------------------------------------------------+
//| Function to draw a breakpoint marker                             |
//+------------------------------------------------------------------+
void drawBreakPoint(string objName,datetime time,double price,int arrCode,
   color clr,int direction){ //--- Define a function to draw a breakpoint marker
   if (ObjectFind(0,objName) < 0){ //--- Check if the breakpoint object does not already exist
      ObjectCreate(0,objName,OBJ_ARROW,0,time,price);        //--- Create an arrow object at the specified time and price
      ObjectSetInteger(0,objName,OBJPROP_ARROWCODE,arrCode); //--- Set the arrow code for the marker
      ObjectSetInteger(0,objName,OBJPROP_COLOR,clr);         //--- Set the color of the arrow
      ObjectSetInteger(0,objName,OBJPROP_FONTSIZE,12);       //--- Set the font size for the arrow
      if (direction > 0) ObjectSetInteger(0,objName,OBJPROP_ANCHOR,ANCHOR_TOP);    //--- Set the anchor to top for upward breaks
      if (direction < 0) ObjectSetInteger(0,objName,OBJPROP_ANCHOR,ANCHOR_BOTTOM); //--- Set the anchor to bottom for downward breaks

      string txt = " Break"; //--- Define the text suffix for the breakpoint label
      string objNameDescr = objName + txt; //--- Create a unique name for the text object
      ObjectCreate(0,objNameDescr,OBJ_TEXT,0,time,price);   //--- Create a text object at the breakpoint
      ObjectSetInteger(0,objNameDescr,OBJPROP_COLOR,clr);   //--- Set the color of the text
      ObjectSetInteger(0,objNameDescr,OBJPROP_FONTSIZE,12); //--- Set the font size of the text
      if (direction > 0) { //--- Check if the breakout is upward
         ObjectSetInteger(0,objNameDescr,OBJPROP_ANCHOR,ANCHOR_LEFT_UPPER); //--- Set the text anchor to left upper
         ObjectSetString(0,objNameDescr,OBJPROP_TEXT, " " + txt);           //--- Set the text content
      }
      if (direction < 0) { //--- Check if the breakout is downward
         ObjectSetInteger(0,objNameDescr,OBJPROP_ANCHOR,ANCHOR_LEFT_LOWER); //--- Set the text anchor to left lower
         ObjectSetString(0,objNameDescr,OBJPROP_TEXT, " " + txt);           //--- Set the text content
      }
   }
   ChartRedraw(0); //--- Redraw the chart to display the breakpoint
}
```

Here, we define the "drawBreakPoint" function to visually mark breakout points. Using parameters "objName", "time", "price", "arrCode", "clr", and "direction", we create an arrow with [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) and [OBJ\_ARROW](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) if absent via [ObjectFind](https://www.mql5.com/en/docs/objects/ObjectFind), setting its style, color, and font size 12 with "ObjectSetInteger", anchoring to "ANCHOR\_TOP" or "ANCHOR\_BOTTOM" based on "direction".

We add a text label " Break" with "ObjectCreate" and "OBJ\_TEXT", named "objNameDescr", configuring color, font size, and anchor ( [ANCHOR\_LEFT\_UPPER](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) or [ANCHOR\_LEFT\_LOWER](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer)) using "ObjectSetInteger" and [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring). We finalize with "ChartRedraw" to display these markers, ensuring clear breakout visualization. We can now use this function to visualize the breakpoints on the chart.

```
double barClose = close(1); //--- Get the closing price of the previous bar
datetime barTime = time(1); //--- Get the time of the previous bar

if (barClose > maximum_price && isHaveDailyRange_Prices && !isHaveRangeBreak
    && barTime >= validBreakTime_start && barTime <= validBreakTime_end
){ //--- Check for a breakout above the maximum price within the valid time window
   Print("CLOSE Price broke the HIGH range. ",barClose," > ",maximum_price); //--- Log the breakout above the high range
   isHaveRangeBreak = true;                                                //--- Set the flag to indicate a range breakout has occurred
   drawBreakPoint(TimeToString(barTime),barTime,barClose,234,clrBlack,-1); //--- Draw a breakpoint marker for the high breakout
}
else if (barClose < minimum_price && isHaveDailyRange_Prices && !isHaveRangeBreak
         && barTime >= validBreakTime_start && barTime <= validBreakTime_end
){ //--- Check for a breakout below the minimum price within the valid time window
   Print("CLOSE Price broke the LOW range. ",barClose," < ",minimum_price); //--- Log the breakout below the low range
   isHaveRangeBreak = true;                                              //--- Set the flag to indicate a range breakout has occurred
   drawBreakPoint(TimeToString(barTime),barTime,barClose,233,clrBlue,1); //--- Draw a breakpoint marker for the low breakout
}
```

To detect and visualize the valid breakouts, we fetch the previous bar’s close with "close(1)" into "barClose" and time with "time(1)" into "barTime". If "barClose" exceeds "maximum\_price", "isHaveDailyRange\_Prices" is true, "isHaveRangeBreak" is false, and "barTime" is within "validBreakTime\_start" to "validBreakTime\_end", we log the high breakout with "Print", set "isHaveRangeBreak" to true, and call "drawBreakPoint" with " [TimeToString](https://www.mql5.com/en/docs/convert/timetostring)(barTime)", "barClose", arrow 234, "clrBlack", and -1.

If "barClose" is below "minimum\_price" under the same conditions, we log the low breakout, set "isHaveRangeBreak", and call "drawBreakPoint" with arrow 233, "clrBlue", and 1. This marks valid breakouts. We used the MQL5 predefined arrows, specifically 233 and 23 as you can see below, but you can use any of your liking.

![ARROWS TABLE](https://c.mql5.com/2/136/Screenshot_2025-04-21_231107.png)

When we run the program, we have the following output.

![THE BREAK](https://c.mql5.com/2/136/Screenshot_2025-04-21_232001.png)

From the image, we can see that we can accurately identify and visualize the break. What we need to do next is define the structure shift and its break logic. We will thus need a function to draw the identified swing point.

```
//+------------------------------------------------------------------+
//| Function to draw a swing point marker                            |
//+------------------------------------------------------------------+
void drawSwingPoint(string objName,datetime time,double price,int arrCode,
   color clr,int direction){ //--- Define a function to draw a swing point marker
   if (ObjectFind(0,objName) < 0){ //--- Check if the swing point object does not already exist
      ObjectCreate(0,objName,OBJ_ARROW,0,time,price);        //--- Create an arrow object at the specified time and price
      ObjectSetInteger(0,objName,OBJPROP_ARROWCODE,arrCode); //--- Set the arrow code for the marker
      ObjectSetInteger(0,objName,OBJPROP_COLOR,clr);         //--- Set the color of the arrow
      ObjectSetInteger(0,objName,OBJPROP_FONTSIZE,10);       //--- Set the font size for the arrow

      if (direction > 0) {ObjectSetInteger(0,objName,OBJPROP_ANCHOR,ANCHOR_TOP);}    //--- Set the anchor to top for swing lows
      if (direction < 0) {ObjectSetInteger(0,objName,OBJPROP_ANCHOR,ANCHOR_BOTTOM);} //--- Set the anchor to bottom for swing highs

      string text = "BoS";                   //--- Define the text label for Break of Structure
      string objName_Descr = objName + text; //--- Create a unique name for the text object
      ObjectCreate(0,objName_Descr,OBJ_TEXT,0,time,price);   //--- Create a text object at the swing point
      ObjectSetInteger(0,objName_Descr,OBJPROP_COLOR,clr);   //--- Set the color of the text
      ObjectSetInteger(0,objName_Descr,OBJPROP_FONTSIZE,10); //--- Set the font size of the text

      if (direction > 0) { //--- Check if the swing is a low
         ObjectSetString(0,objName_Descr,OBJPROP_TEXT,"  "+text);            //--- Set the text content
         ObjectSetInteger(0,objName_Descr,OBJPROP_ANCHOR,ANCHOR_LEFT_UPPER); //--- Set the text anchor to left upper
      }
      if (direction < 0) { //--- Check if the swing is a high
         ObjectSetString(0,objName_Descr,OBJPROP_TEXT,"  "+text);            //--- Set the text content
         ObjectSetInteger(0,objName_Descr,OBJPROP_ANCHOR,ANCHOR_LEFT_LOWER); //--- Set the text anchor to left lower
      }
   }
   ChartRedraw(0); //--- Redraw the chart to display the swing point
}
```

We implement the "drawSwingPoint" function to mark swing highs and lows that have been identified. Using parameters "objName", "time", "price", "arrCode", "clr", and "direction", we verify absence with [ObjectFind](https://www.mql5.com/en/docs/objects/ObjectFind), create an arrow with [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) using [OBJ\_ARROW](https://www.mql5.com/en/docs/constants/objectconstants/enum_object), and set style, color, and font size 10 via [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger), anchoring to "ANCHOR\_TOP" for lows or "ANCHOR\_BOTTOM" for highs. We add a "BoS" text label with "ObjectCreate" using "OBJ\_TEXT", configuring color, font size, and anchor ( [ANCHOR\_LEFT\_UPPER](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) or [ANCHOR\_LEFT\_LOWER](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer)) via "ObjectSetInteger" and "ObjectSetString". We call [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to display these markers, highlighting key swing points. With this function, we then continue to structure the logic for swing point identification.

```
// bos logic
if (isHaveDailyRange_Prices){ //--- Proceed with BoS logic only if the daily range is calculated
   static bool isNewBar_bos = false;            //--- Initialize flag to indicate a new bar on the BoS timeframe
   int currBars = iBars(_Symbol,timeframe_bos); //--- Get the current number of bars on the BoS timeframe
   static int prevBars = currBars;              //--- Store the previous number of bars for comparison

   if (prevBars == currBars){isNewBar_bos = false;}                          //--- Set flag to false if no new bar has formed
   else if (prevBars != currBars){isNewBar_bos = true; prevBars = currBars;} //--- Set flag to true and update prevBars if a new bar has formed

   const int length = 4;                         //--- Define the number of bars to check for swing high/low (must be > 2)
   int right_index, left_index;                  //--- Declare variables to store indices for bars to the right and left
   int curr_bar = length;                        //--- Set the current bar index for swing analysis
   bool isSwingHigh = true, isSwingLow = true;   //--- Initialize flags to determine if the current bar is a swing high or low
   static double swing_H = -1.0, swing_L = -1.0; //--- Initialize variables to store the latest swing high and low prices

   if (isNewBar_bos){ //--- Check if a new bar has formed on the BoS timeframe
      for (int a=1; a<=length; a++){ //--- Loop through the specified number of bars to check for swings
         right_index = curr_bar - a; //--- Calculate the right-side bar index
         left_index = curr_bar + a;  //--- Calculate the left-side bar index
         if ( (high(curr_bar,timeframe_bos) <= high(right_index,timeframe_bos)) || (high(curr_bar,timeframe_bos) < high(left_index,timeframe_bos)) ){ //--- Check if the current bar's high is not the highest
            isSwingHigh = false; //--- Set flag to false if the bar is not a swing high
         }
         if ( (low(curr_bar,timeframe_bos) >= low(right_index,timeframe_bos)) || (low(curr_bar,timeframe_bos) > low(left_index,timeframe_bos)) ){ //--- Check if the current bar's low is not the lowest
            isSwingLow = false; //--- Set flag to false if the bar is not a swing low
         }
      }

      if (isSwingHigh){ //--- Check if the current bar is a swing high
         swing_H = high(curr_bar,timeframe_bos); //--- Store the swing high price
         Print("WE DO HAVE A SWING HIGH @ BAR INDEX ",curr_bar," H: ",high(curr_bar,timeframe_bos)); //--- Log the swing high details
         drawSwingPoint(TimeToString(time(curr_bar,timeframe_bos)),time(curr_bar,timeframe_bos),high(curr_bar,timeframe_bos),77,clrBlue,-1); //--- Draw a marker for the swing high
      }
      if (isSwingLow){ //--- Check if the current bar is a swing low
         swing_L = low(curr_bar,timeframe_bos); //--- Store the swing low price
         Print("WE DO HAVE A SWING LOW @ BAR INDEX ",curr_bar," L: ",low(curr_bar,timeframe_bos)); //--- Log the swing low details
         drawSwingPoint(TimeToString(time(curr_bar,timeframe_bos)),time(curr_bar,timeframe_bos),low(curr_bar,timeframe_bos),77,clrRed,+1); //--- Draw a marker for the swing low
      }
   }
}
```

If we have the daily prices, meaning we have the daily range already defined, we track new bars on the Break of Structure timeframe using a static "isNewBar\_bos" flag, comparing the current bar count from [iBars](https://www.mql5.com/en/docs/series/ibars) with [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol) and "timeframe\_bos" against a static "prevBars", updating "isNewBar\_bos" to true and "prevBars" when a new bar forms.

When "isNewBar\_bos" is true, we analyze the bar at index "curr\_bar" (set to "length" = 4) for swings, checking "length" bars on either side using "right\_index" and "left\_index". We use "high" and "low" functions with "timeframe\_bos" to compare the current bar’s high and low against neighboring bars, setting "isSwingHigh" or "isSwingLow" to false if not the highest or lowest.

If "isSwingHigh", we store the price in "swing\_H", log it with "Print", and call "drawSwingPoint" with "TimeToString", the bar’s time, price, arrow code 77, "clrBlue", and -1; if "isSwingLow", we update "swing\_L", log, and call "drawSwingPoint" with "clrRed" and +1. Upon compilation, we have the following outcome.

![CONFIRMED SWING POINTS](https://c.mql5.com/2/136/Screenshot_2025-04-21_233629.png)

From the image, you can see that we draw the swing points. The next thing that we need to do is track the break of the swing points meaning a structure shift and so a break of structure. To visualize these again, we will need a custom function as follows.

```
//+------------------------------------------------------------------+
//| Function to draw a break level line                              |
//+------------------------------------------------------------------+
void drawBreakLevel(string objName,datetime time1,double price1,
   datetime time2,double price2,color clr,int direction){ //--- Define a function to draw a break level line
   if (ObjectFind(0,objName) < 0){ //--- Check if the break level object does not already exist
      ObjectCreate(0,objName,OBJ_ARROWED_LINE,0,time1,price1,time2,price2); //--- Create an arrowed line object
      ObjectSetInteger(0,objName,OBJPROP_TIME,0,time1);                     //--- Set the first time coordinate of the line
      ObjectSetDouble(0,objName,OBJPROP_PRICE,0,price1);                    //--- Set the first price coordinate of the line
      ObjectSetInteger(0,objName,OBJPROP_TIME,1,time2);                     //--- Set the second time coordinate of the line
      ObjectSetDouble(0,objName,OBJPROP_PRICE,1,price2);                    //--- Set the second price coordinate of the line

      ObjectSetInteger(0,objName,OBJPROP_COLOR,clr); //--- Set the color of the line
      ObjectSetInteger(0,objName,OBJPROP_WIDTH,2);   //--- Set the width of the line

      string text = "Break";                 //--- Define the text label for the break
      string objName_Descr = objName + text; //--- Create a unique name for the text object
      ObjectCreate(0,objName_Descr,OBJ_TEXT,0,time2,price2); //--- Create a text object at the line's end
      ObjectSetInteger(0,objName_Descr,OBJPROP_COLOR,clr);   //--- Set the color of the text
      ObjectSetInteger(0,objName_Descr,OBJPROP_FONTSIZE,10); //--- Set the font size of the text

      if (direction > 0) { //--- Check if the break is upward
         ObjectSetString(0,objName_Descr,OBJPROP_TEXT,text+"  ");             //--- Set the text content
         ObjectSetInteger(0,objName_Descr,OBJPROP_ANCHOR,ANCHOR_RIGHT_UPPER); //--- Set the text anchor to right upper
      }
      if (direction < 0) { //--- Check if the break is downward
         ObjectSetString(0,objName_Descr,OBJPROP_TEXT,text+"  ");             //--- Set the text content
         ObjectSetInteger(0,objName_Descr,OBJPROP_ANCHOR,ANCHOR_RIGHT_LOWER); //--- Set the text anchor to right lower
      }
   }
   ChartRedraw(0); //--- Redraw the chart to display the break level
}
```

We just define the "drawBreakLevel" function to visualize the break of the structure. We use a similar logic for the visualization just as we did with the prior visualization functions, so we wont invest much time explaining what everything does. We will use this function to visualize the levels.

```
double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits); //--- Get and normalize the current Ask price
double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits); //--- Get and normalize the current Bid price

if (swing_H > 0 && Ask > swing_H){ //--- Check if the Ask price breaks above the swing high
   Print("$$$$$$$$$ BUY SIGNAL NOW. BREAK OF SWING HIGH"); //--- Log a buy signal due to swing high breakout
   int swing_H_index = 0;                //--- Initialize the index of the swing high bar
   for (int i=0; i<=length*2+1000; i++){ //--- Loop through bars to find the swing high
      double high_sel = high(i,timeframe_bos); //--- Get the high price of the i-th bar
      if (high_sel == swing_H){ //--- Check if the high matches the swing high
         swing_H_index = i;     //--- Store the bar index
         Print("BREAK HIGH FOUND @ BAR INDEX ",swing_H_index); //--- Log the swing high bar index
         break;                 //--- Exit the loop once found
      }
   }
   drawBreakLevel(TimeToString(time(0,timeframe_bos)),time(swing_H_index,timeframe_bos),high(swing_H_index,timeframe_bos),
   time(0,timeframe_bos),high(swing_H_index,timeframe_bos),clrBlue,-1); //--- Draw a line to mark the swing high breakout

   if (isTakenTrade == false){                                     //--- Check if no trade is taken yet
      obj_Trade.Buy(0.01,_Symbol,Ask,minimum_price,maximum_price); //--- Execute a buy trade with 0.01 lots, using minimum price as SL and maximum as TP
      isTakenTrade = true;                                         //--- Set the flag to indicate a trade is taken
   }

   swing_H = -1.0; //--- Reset the swing high price
   return;         //--- Exit the OnTick function to avoid further processing
}
if (swing_L > 0 && Bid < swing_L){ //--- Check if the Bid price breaks below the swing low
   Print("$$$$$$$$$ SELL SIGNAL NOW. BREAK OF SWING LOW"); //--- Log a sell signal due to swing low breakout
   int swing_L_index = 0; //--- Initialize the index of the swing low bar
   for (int i=0; i<=length*2+1000; i++){ //--- Loop through bars to find the swing low
      double low_sel = low(i,timeframe_bos); //--- Get the low price of the i-th bar
      if (low_sel == swing_L){ //--- Check if the low matches the swing low
         swing_L_index = i;    //--- Store the bar index
         Print("BREAK LOW FOUND @ BAR INDEX ",swing_L_index); //--- Log the swing low bar index
         break;                //--- Exit the loop once found
      }
   }
   drawBreakLevel(TimeToString(time(0,timeframe_bos)),time(swing_L_index,timeframe_bos),low(swing_L_index,timeframe_bos),
   time(0,timeframe_bos),low(swing_L_index,timeframe_bos),clrRed,+1); //--- Draw a line to mark the swing low breakout

   if (isTakenTrade == false){ //--- Check if no trade is taken yet
      obj_Trade.Sell(0.01,_Symbol,Bid,maximum_price,minimum_price); //--- Execute a sell trade with 0.01 lots, using maximum price as SL and minimum as TP
      isTakenTrade = true;                                          //--- Set the flag to indicate a trade is taken
   }

   swing_L = -1.0; //--- Reset the swing low price
   return;         //--- Exit the OnTick function to avoid further processing
}
```

Here, we implement trade execution logic when we have a valid breakout. We get normalized "Ask" and "Bid" prices using [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) and [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble) with [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol) and [\_Digits](https://www.mql5.com/en/docs/predefined/_Digits).

If "swing\_H" is positive and "Ask" exceeds "swing\_H", we log with "Print", find the swing high index with "high" and "timeframe\_bos", mark it with "drawBreakLevel" using [TimeToString](https://www.mql5.com/en/docs/convert/timetostring) and "time", and call "obj\_Trade.Buy" with 0.01 lots, "minimum\_price" stop-loss, and "maximum\_price" take-profit if "isTakenTrade" is false, setting it true and resetting "swing\_H".

If "swing\_L" is positive and "Bid" falls below "swing\_L", we log, find the swing low index with "low", mark with "drawBreakLevel", and call "obj\_Trade.Sell", resetting "swing\_L". We exit with "return" after each trade for precise Break of Structure trading. Here is the outcome.

![TRADED SETUP](https://c.mql5.com/2/136/Screenshot_2025-04-22_001241.png)

We are now able to open trades for the confirmed setups. However, what about when there is a breakout that happens but it is not within the range. That is price pushes higher or lower from the range boundaries. We need to wait till the price comes back within the range for us to consider the break as valid. To achieve that, we will need to get the maximum and minimum range prices and add them for a firmer restriction to avoid false signals.

```
if (swing_H > 0 && Ask > swing_H && swing_H <= maximum_price && swing_H >= minimum_price){ //--- Check if the Ask price breaks above the swing high within the range
   Print("$$$$$$$$$ BUY SIGNAL NOW. BREAK OF SWING HIGH WITHIN RANGE"); //--- Log a buy signal due to swing high breakout
   int swing_H_index = 0;                                               //--- Initialize the index of the swing high bar
   for (int i=0; i<=length*2+1000; i++){                                //--- Loop through bars to find the swing high
      double high_sel = high(i,timeframe_bos); //--- Get the high price of the i-th bar
      if (high_sel == swing_H){                //--- Check if the high matches the swing high
         swing_H_index = i; //--- Store the bar index
         Print("BREAK HIGH FOUND @ BAR INDEX ",swing_H_index); //--- Log the swing high bar index
         break;             //--- Exit the loop once found
      }
   }
   drawBreakLevel(TimeToString(time(0,timeframe_bos)),time(swing_H_index,timeframe_bos),high(swing_H_index,timeframe_bos),
   time(0,timeframe_bos),high(swing_H_index,timeframe_bos),clrBlue,-1); //--- Draw a line to mark the swing high breakout

   if (isTakenTrade == false){ //--- Check if no trade is taken yet
      obj_Trade.Buy(0.01,_Symbol,Ask,minimum_price,maximum_price); //--- Execute a buy trade with 0.01 lots, using minimum price as SL and maximum as TP
      isTakenTrade = true;                                         //--- Set the flag to indicate a trade is taken
   }

   swing_H = -1.0; //--- Reset the swing high price
   return;         //--- Exit the OnTick function to avoid further processing
}
if (swing_L > 0 && Bid < swing_L && swing_L <= maximum_price && swing_L >= minimum_price){ //--- Check if the Bid price breaks below the swing low within the range
   Print("$$$$$$$$$ SELL SIGNAL NOW. BREAK OF SWING LOW WITHIN RANGE"); //--- Log a sell signal due to swing low breakout
   int swing_L_index = 0;                //--- Initialize the index of the swing low bar
   for (int i=0; i<=length*2+1000; i++){ //--- Loop through bars to find the swing low
      double low_sel = low(i,timeframe_bos); //--- Get the low price of the i-th bar
      if (low_sel == swing_L){ //--- Check if the low matches the swing low
         swing_L_index = i;    //--- Store the bar index
         Print("BREAK LOW FOUND @ BAR INDEX ",swing_L_index); //--- Log the swing low bar index
         break;                //--- Exit the loop once found
      }
   }
   drawBreakLevel(TimeToString(time(0,timeframe_bos)),time(swing_L_index,timeframe_bos),low(swing_L_index,timeframe_bos),
   time(0,timeframe_bos),low(swing_L_index,timeframe_bos),clrRed,+1); //--- Draw a line to mark the swing low breakout

   if (isTakenTrade == false){ //--- Check if no trade is taken yet
      obj_Trade.Sell(0.01,_Symbol,Bid,maximum_price,minimum_price); //--- Execute a sell trade with 0.01 lots, using maximum price as SL and maximum as TP
      isTakenTrade = true;                                          //--- Set the flag to indicate a trade is taken
   }

   swing_L = -1.0; //--- Reset the swing low price
   return;         //--- Exit the OnTick function to avoid further processing
}
```

We are okay by discarding false signals and we can see that we can open trades for a confirmed setup, hence achieving our objective of identifying, visualizing, and trading the strategy setup. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/136/Screenshot_2025-04-22_010443.png)

Backtest report:

![REPORT](https://c.mql5.com/2/136/Screenshot_2025-04-22_010513.png)

### Conclusion

In conclusion, we have built a MQL5 Expert Advisor that automates the Midnight Range Breakout with [Break of Structure](https://www.mql5.com/go?link=https://forexbee.co/break-of-structure/ "https://forexbee.co/break-of-structure/") strategy, trading breakouts within the midnight range confirmed by current-day swing points. With precise range detection and visualization, you can advance it more and define some more strategies to make it robust and geared towards your trading style.

Disclaimer: This article is for educational purposes only. Trading involves substantial financial risks, and market volatility can lead to losses. Comprehensive backtesting and prudent risk management are essential before using this Expert Advisor in live markets.

By mastering these techniques, you can advance your algorithmic trading skills and approach markets with greater confidence. Best of luck in your trading journey!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17876.zip "Download all attachments in the single ZIP archive")

[Midnight\_Range\_Break\_of\_Structure\_Breakout.mq5](https://www.mql5.com/en/articles/download/17876/midnight_range_break_of_structure_breakout.mq5 "Download Midnight_Range_Break_of_Structure_Breakout.mq5")(29.89 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/485811)**
(7)


![CapeCoddah](https://c.mql5.com/avatar/avatar_na2.png)

**[CapeCoddah](https://www.mql5.com/en/users/capecoddah)**
\|
3 May 2025 at 19:47

Hi Allan.

I'm going to download your system and begin testing.  Interestingly the [strategy testers](https://www.mql5.com/en/articles/239 "Article: The Fundamentals of Testing in MetaTrader 5 ") report does not identify either the trading pair nor the time frame.  As you illustrated the AUUSD M15, I am assuming it is what you used and I will begin testing with.  Do you have any feeling for using other pairs or time frames.   I suspect that this ea may work better on Asian trading pairs, am I correct?

Cheers, CapeCoddah

The testing was a bust.  I tried AUDUSD, AUDJPY, & USDJPY and all produced losses with the Sharpe Ratios of -3.00 to -5.00. All except USDJPY went negative immediately and never recovered.  USDJPY had 2 periods of positive gains but eventually turned negative and never returned.

Adios

![Juan Guirao](https://c.mql5.com/avatar/2023/10/6520fda6-f3b7.jpg)

**[Juan Guirao](https://www.mql5.com/en/users/freenrg)**
\|
5 May 2025 at 17:02

Excellent work. Thank you Allan!


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
5 May 2025 at 18:52

**Juan Guirao [#](https://www.mql5.com/en/forum/485811#comment_56625124):**

Excellent work. Thank you Allan!

Sure. Welcome. Thanks for the kind feedback.

![sevkoff](https://c.mql5.com/avatar/avatar_na2.png)

**[sevkoff](https://www.mql5.com/en/users/sevkoff)**
\|
5 Oct 2025 at 16:26

Thank You Allan!. you are the best


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
6 Oct 2025 at 13:07

**sevkoff [#](https://www.mql5.com/en/forum/485811#comment_58193926):**

Thank You Allan!. you are the best

Thank you for the kind feedback. Welcome.


![MQL5 Wizard Techniques you should know (Part 62): Using Patterns of ADX and CCI with Reinforcement-Learning TRPO](https://c.mql5.com/2/139/article_17938_image-logo.png)[MQL5 Wizard Techniques you should know (Part 62): Using Patterns of ADX and CCI with Reinforcement-Learning TRPO](https://www.mql5.com/en/articles/17938)

The ADX Oscillator and CCI oscillator are trend following and momentum indicators that can be paired when developing an Expert Advisor. We continue where we left off in the last article by examining how in-use training, and updating of our developed model, can be made thanks to reinforcement-learning. We are using an algorithm we are yet to cover in these series, known as Trusted Region Policy Optimization. And, as always, Expert Advisor assembly by the MQL5 Wizard allows us to set up our model(s) for testing much quicker and also in a way where it can be distributed and tested with different signal types.

![Creating Dynamic MQL5 Graphical Interfaces through Resource-Driven Image Scaling with Bicubic Interpolation on Trading Charts](https://c.mql5.com/2/138/logo-17892-2.png)[Creating Dynamic MQL5 Graphical Interfaces through Resource-Driven Image Scaling with Bicubic Interpolation on Trading Charts](https://www.mql5.com/en/articles/17892)

In this article, we explore dynamic MQL5 graphical interfaces, using bicubic interpolation for high-quality image scaling on trading charts. We detail flexible positioning options, enabling dynamic centering or corner anchoring with custom offsets.

![Developing a Replay System (Part 66): Playing the service (VII)](https://c.mql5.com/2/94/Desenvolvendo_um_sistema_de_Replay_Parte_66__LOGO.png)[Developing a Replay System (Part 66): Playing the service (VII)](https://www.mql5.com/en/articles/12286)

In this article, we will implement the first solution that will allow us to determine when a new bar may appear on the chart. This solution is applicable in a wide variety of situations. Understanding its development will help you grasp several important aspects. The content presented here is intended solely for educational purposes. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

![DoEasy. Service functions (Part 3): Outside Bar pattern](https://c.mql5.com/2/75/DoEasy._Service_functions_Part_1___LOGO.png)[DoEasy. Service functions (Part 3): Outside Bar pattern](https://www.mql5.com/en/articles/14710)

In this article, we will develop the Outside Bar Price Action pattern in the DoEasy library and optimize the methods of access to price pattern management. In addition, we will fix errors and shortcomings identified during library tests.

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/17876&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049381183814478459)

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