---
title: Manual charting and trading toolkit (Part I). Preparation: structure description and helper class
url: https://www.mql5.com/en/articles/7468
categories: Trading, Indicators
relevance_score: 0
scraped_at: 2026-01-24T13:34:19.398911
---

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/7468&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5082961793640305193)

MetaTrader 5 / Examples


### Introduction

I am a manual trader. I prefer to analyze charts not using complex formulas and indicators, but manually, i.e. graphically. This helps me be more flexible in trading, notice some things which are hard to formalize, easily switch between timeframes if I see that the movement is increasing or slowing down, know the probable price behavior long before I enter a trade.

Basically, I use different combinations of trend lines (pitchfork, fan, levels and so on). So, I created a convenient tool enabling quick drawing of trend lines, literally with one keystroke. Now I would like to share this tool with the community.

### Video presentation. How it works

Презентация библиотеки Shortcuts - YouTube

[Photo image of Oleg Fedorov](https://www.youtube.com/channel/UCKjTq_OmRkCP2ZBLCimpoqQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7468)

Oleg Fedorov

10 subscribers

[Презентация библиотеки Shortcuts](https://www.youtube.com/watch?v=1so5g-cI3Eo)

Oleg Fedorov

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

More videos

## More videos

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=1so5g-cI3Eo&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7468)

0:00

0:00 / 3:11

•Live

•

### General concept. Setting a task

So, we are creating a tool that helps to perform the most frequent operations using keyboard shortcuts.

Which tasks can be implemented? For example:

- draw a simple horizontal line by pressing the "**H**" key ("Horizontal"), and to draw a vertical line by pressing "i" (simply because it looks like a vertical line)

- draw horizontal lines of a certain length (NOT infinite) starting at any point on the chart
- draw vertical levels at a certain (arbitrary) distance from the starting point

- switch timeframes and rearrange layers on the chart using keys
- draw Fibonacci fan with the preset levels, at the nearest extreme points
- draw trend lines at the nearest extreme points; their length should be a multiple of the distance between the extreme points; in some cases the line can also be a ray
- draw Andrews' pitchforks of various types (standard, Schiff, reverse Schiff pitchfork - for fast trends) (see. the video)
- for extreme points of pitchforks, lines and fans, there should be an ability to customize the order of extreme points (the number of bars, on the left and on the right separately)
- a graphical interface that allows customizing the parameters of required lines and extreme points without opening the settings window
- a set of order managing functions: open an order based on percent of deposit, automatically set stop levels when opening a market order or when a pending order having no stop level triggers, enable partial position closure by levels, trailing stop and so on


Of course, most of the tasks can be automated using scripts. I have a few of them. You can find them below, in the article attachment. However, I prefer a different approach.

_In any Expert Advisor or indicator_, we can create the [**OnChartEvent**](https://www.mql5.com/en/docs/event_handlers/onchartevent) method containing description of reactions to any events: keystrokes, mouse movements, creation or deletion of graphical objects.

That is why I decided to create a program as include files. All functions and variables are distributed over several classes to make them easier to access. At this point, I only need classes for convenient grouping of functions. That is why, in this first implementation, we will not use complex things like inheritance or factories. It is only a collection.

Furthermore, I want to create a cross-platform class which can run both in MQL4 and MQL5.

### Structure of the Program

![List of files](https://c.mql5.com/2/38/files.png)

The library contains five related files. All files are located under the "Shortcuts" folder in the Include directory. Their names are shown in the figure: GlobalVariables.mqh, Graphics.mqh, Mouse.mqh, Shortcuts.mqh, Utilites.mqh.

### The main library file (Shortcuts.mqh)

The main file of the program is "Shortcuts.mqh". Logic of reaction to keystrokes will be written in this file. This is the file that should be connected to an Expert Advisor. All auxiliary files will also be included in it.

```
//+------------------------------------------------------------------+
//|                                                    Shortcuts.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                            https://www.mql5.com/ru/articles/7468 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://www.mql5.com/en/articles/7468"

#include "GlobalVariables.mqh"
#include "Mouse.mqh"
#include "Utilites.mqh"
#include "Graphics.mqh"

//+------------------------------------------------------------------+
//| The main control class of the program. It should be connected    |
//| to the Expert Advisor                                            |
//+------------------------------------------------------------------+
class CShortcuts
  {
private:
   CGraphics         m_graphics;   // Object for drawing m_graphics
public:
                     CShortcuts();
   void              OnChartEvent(const int id,
                                  const long &lparam,
                                  const double &dparam,
                                  const string &sparam);
  };
//+------------------------------------------------------------------+
//| Default constructor                                              |
//+------------------------------------------------------------------+
CShortcuts::CShortcuts(void)
  {
   ChartSetInteger(0,CHART_EVENT_MOUSE_MOVE,true);
  }

//+------------------------------------------------------------------+
//| Event handling function                                          |
//+------------------------------------------------------------------+
void CShortcuts::OnChartEvent(
   const int id,
   const long &lparam,
   const double &dparam,
   const string &sparam
)
  {
//---

   // This will contain the description of the events related to keystrokes
   // and mouse movements

   //  ...

  }
}

CShortcuts shortcuts;
```

The file contains the **CShortcuts** class description.

All helper classes are connected at the beginning of the file

The class has only two methods. The first one is the **OnChartEvent** event handler, in which all keystroke and mouse movement events will be handled. The second one is the default constructor, in which mouse movements can be handled.

After class description, a shortcuts variable is created, which should be used in the **OnChartEvent** method of the main Expert Advisor, when the library is connected.

The connection requires two lines:

```
//+------------------------------------------------------------------+
//| The main Expert (file "Shortcuts-Main-Expert.mq5")               |
//+------------------------------------------------------------------+
#include <Shortcuts\Shortcuts.mqh>

// ...

//+------------------------------------------------------------------+
//| The ChartEvent function                                          |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   shortcuts.OnChartEvent(id,lparam,dparam,sparam);
  }
```

The first line connects the class file. The second line transfers event handling control to the class.

After that the Expert Advisor is ready to work - it can be compiled and used for drawing lines.

### Mouse movement handling class

The class will store all basic parameters of the current cursor position: coordinates X, Y in pixels and in price/time, bar number on which the pointer is positioned, and so one - all these are stored in the "Mouse.mqh" file.

```
//+------------------------------------------------------------------+
//|                                                        Mouse.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                            https://www.mql5.com/ru/articles/7468 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://www.mql5.com/en/articles/7468"
//+------------------------------------------------------------------+
//| Mouse movement handling class                                    |
//+------------------------------------------------------------------+
class CMouse
  {
   //--- Members
private:
   static int        m_x;           // X
   static int        m_y;           // Y
   static int        m_barNumber;   // Bar number
   static bool       m_below;       // Indication of a cursor above the price
   static bool       m_above;       // Indication of a cursor below the price

   static datetime   m_currentTime; // Current time
   static double     m_currentPrice;// Current price
   //--- Methods
public:

   //--- Remembers the main parameters of the mouse cursor
   static void       SetCurrentParameters(
      const int id,
      const long &lparam,
      const double &dparam,
      const string &sparam
   );
   //--- Returns the X coordinate (in pixels) of the current cursor position
   static int        X(void) {return m_x;}
   //--- Returns the Y coordinate (in pixels) of the current cursor position
   static int        Y(void) {return m_y;}
   //--- Returns the price of the current cursor position
   static double     Price(void) {return m_currentPrice;}
   //--- Returns the time of the current cursor position
   static datetime   Time(void) {return m_currentTime;}
   //--- Returns the bar number of the current cursor position
   static int        Bar(void) {return m_barNumber;}
   //--- Returns a flag showing that the price is below the Low of the current bar
   static bool       Below(void) {return m_below;}
   //--- Returns a flag showing that the price is above the High of the current bar
   static bool       Above(void) {return m_above;}
  };
//---

int CMouse::m_x=0;
int CMouse::m_y=0;
int CMouse::m_barNumber=0;
bool CMouse::m_below=false;
bool CMouse::m_above=false;
datetime CMouse::m_currentTime=0;
double CMouse::m_currentPrice=0;

//+------------------------------------------------------------------+
//| Remembers the main parameters for a mouse movement: coordinates  |
//| of cursor in pixels and price/time, whether the cursor is        |
//| above or below the price.                                        |
//|+-----------------------------------------------------------------+
static void CMouse::SetCurrentParameters(
   const int id,
   const long &lparam,
   const double &dparam,
   const string &sparam
)
  {
//---
   int window = 0;
//---
   ChartXYToTimePrice(
      0,
      (int)lparam,
      (int)dparam,
      window,
      m_currentTime,
      m_currentPrice
   );
   m_x=(int)lparam;
   m_y=(int)dparam;
   m_barNumber=iBarShift(
                  Symbol(),
                  PERIOD_CURRENT,
                  m_currentTime
               );
   m_below=m_currentPrice<iLow(Symbol(),PERIOD_CURRENT,m_barNumber);
   m_above=m_currentPrice>iHigh(Symbol(),PERIOD_CURRENT,m_barNumber);
  }

//+------------------------------------------------------------------+
```

This class can be used from any place within a program, because its methods are declared as static. There is no need to create an instance of this class to use it.

### Description of the Expert Advisor settings block. The GlobalVariables.mqh file

All settings available to the user are stored in the GlobalVariables.mqh file.

The next article will provide descriptions of much more settings, as we will add more objects for drawing.

Below is the code of settings which are used in the current version:

```
//+------------------------------------------------------------------+
//|                                              GlobalVariables.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                            https://www.mql5.com/ru/articles/7468 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://www.mql5.com/en/articles/7468"
//+------------------------------------------------------------------+
//| File describing parameters available to the user                 |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Key settings                                                     |
//+------------------------------------------------------------------+
input string   Keys="=== Key settings ===";
input string   Up_Key="U";            // Switch timeframe up
input string   Down_Key="D";          // Switch timeframe down
input string   Trend_Line_Key="T";              // Trend line
input string   Switch_Trend_Ray_Key="R";        // Indication of a trend line ray
input string   Z_Index_Key="Z";                 // Indication of the chart on top

//+------------------------------------------------------------------+
//| Size settings                                                    |
//+------------------------------------------------------------------+
input string   Dimensions="=== Size settings ===";
input int      Trend_Line_Width=2;        // Trend line width

//+------------------------------------------------------------------+
//| Display styles                                                   |
//+------------------------------------------------------------------+
input string   Styles="=== Display styles ===";
input ENUM_LINE_STYLE      Trend_Line_Style=STYLE_SOLID;       // Trend line style

//+------------------------------------------------------------------+
//| Other parameters                                                 |
//+------------------------------------------------------------------+
input string               Others="=== Other parameters ===";

input bool                 Is_Trend_Ray=false;                      // Trend line - ray
input bool                 Is_Change_Timeframe_On_Create = true;    // Hide objects on higher timeframes?
                                                                    //   (true - hide, false - show)
input bool                 Is_Select_On_Create=true;                // Select upon creation
input bool                 Is_Different_Colors=true;                // Change colors for times

// Number of bars on the left and on the right
// for trend line and fan extreme points
input int                  Fractal_Size_Left=1;                     // Size of the left fractal
input int                  Fractal_Size_Right=1;                    // Size of the right fractal

//+------------------------------------------------------------------+
//| Name prefixes of drawn shapes (can be change only in code,       |
//| not visible in EA parameters)                                    |
//+------------------------------------------------------------------+
// string   Prefixes="=== Prefixes ===";

string   Trend_Line_Prefix="Trend_";                     // Trend line prefix

//+------------------------------------------------------------------+
//| Colors for objects of one timeframe (can be changed only in code,|
//| not visible in EA parameters)                                    |
//+------------------------------------------------------------------+
// string TimeframeColors="=== Time frame colors ===";
color mn1_color=clrCrimson;
color w1_color=clrDarkOrange;
color d1_color=clrGoldenrod;
color h4_color=clrLimeGreen;
color h1_color=clrLime;
color m30_color=clrDeepSkyBlue;
color m15_color=clrBlue;
color m5_color=clrViolet;
color m1_color=clrDarkViolet;
color common_color=clrGray;

//--- Auxiliary constant for displaying error messages
#define DEBUG_MESSAGE_PREFIX "=== ",__FUNCTION__," === "

//--- Constants for describing the main timeframes when drawing
#define PERIOD_LOWER_M5 OBJ_PERIOD_M1|OBJ_PERIOD_M5
#define PERIOD_LOWER_M15 PERIOD_LOWER_M5|OBJ_PERIOD_M15
#define PERIOD_LOWER_M30 PERIOD_LOWER_M15|OBJ_PERIOD_M30
#define PERIOD_LOWER_H1 PERIOD_LOWER_M30|OBJ_PERIOD_H1
#define PERIOD_LOWER_H4 PERIOD_LOWER_H1|OBJ_PERIOD_H4
#define PERIOD_LOWER_D1 PERIOD_LOWER_H4|OBJ_PERIOD_D1
#define PERIOD_LOWER_W1 PERIOD_LOWER_D1|OBJ_PERIOD_W1
//+------------------------------------------------------------------+
```

### Auxiliary functions

The program has a lot of functions which are not directly related to charting, but they help find extreme points, switch timeframes and so on. All these functions are implemented in the "Utilites.mqh" file.

#### The Utilites.mqh file header

```
//+------------------------------------------------------------------+
//|                                                     Utilites.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                            https://www.mql5.com/ru/articles/7468 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://www.mql5.com/en/articles/7468"

//+------------------------------------------------------------------+
//| Class describing auxiliary functions                             |
//+------------------------------------------------------------------+
class CUtilites
  {
public:
   //--- Changes the timeframe to the next one on the toolbar
   static void       ChangeTimeframes(bool isUp);
   //--- Converts string command constants to keycodes
   static int        GetCurrentOperationChar(string keyString);
   //--- Switches layers in charts (the chart is on top of all objects)
   static void       ChangeChartZIndex(void);
   //--- Returns the number of the nearest extreme bar
   static int        GetNearestExtremumBarNumber(
      int starting_number=0,
      bool is_search_right=false,
      bool is_up=false,
      int left_side_bars=1,
      int right_side_bars=1,
      string symbol=NULL,
      ENUM_TIMEFRAMES timeframe=PERIOD_CURRENT
   );

   //--- Returns the color for the current timeframe
   static color      GetTimeFrameColor(long allDownPeriodsValue);
   //--- Returns a list of all timeframes lower than the current one (including the current one)
   static long       GetAllLowerTimeframes(int NeededTimeframe=PERIOD_CURRENT);
   //--- Coordinates of the straight line. Writes numbers of extreme bars to points p1 and p2
   static void       SetExtremumsBarsNumbers(bool _is_up,int &p1, int &p2);
   //--- Converts a string to an array of floating point numbers
   static void       StringToDoubleArray(
      string _haystack,
      double &_result[],
      const string _delimiter=","
   );
   //--- Determines the name of the current object
   static string            GetCurrentObjectName(
      const string _prefix,
      const ENUM_OBJECT _type=OBJ_TREND,
      int _number = -1
   );
   //--- Obtains the number of the next object
   static int               GetNextObjectNumber(
      const string _prefix,
      const ENUM_OBJECT _object_type,
      bool true
   );
   //--- Returns the distance, in screen pixels, between adjacent bars
   static int               GetBarsPixelDistance(void);
   //--- Converts a numeric value of the timeframe to its string name
   static string            GetTimeframeSymbolName(
      ENUM_TIMEFRAMES _timeframe=PERIOD_CURRENT  // Desired timeframe
   );
  };
```

#### The function sequentially changes the period of the chart

First functions in this file are simple. For example, the function that sequentially changes the periods of the current graph, looks as follows:

```
//+------------------------------------------------------------------+
//| Sequentially changes the current chart period                    |
//|                                                                  |
//| In this implementation, only the timeframes shown in the         |
//| standard panel are used.                                         |
//|                                                                  |
//| Parameters:                                                      |
//|    _isUp - timeframe switch direction: up (true)                 |
//|            or down (false)                                       |
//+------------------------------------------------------------------+
static void CUtilites::ChangeTimeframes(bool _isUp)
  {
   ENUM_TIMEFRAMES timeframes[] =
     {
      PERIOD_CURRENT,
      PERIOD_M1,
      PERIOD_M5,
      PERIOD_M15,
      PERIOD_M30,
      PERIOD_H1,
      PERIOD_H4,
      PERIOD_D1,
      PERIOD_W1,
      PERIOD_MN1
     };
   int period = Period();
   int shift = ArrayBsearch(timeframes,period);
   if(_isUp && shift < ArraySize(timeframes)-1)
     {
      ChartSetSymbolPeriod(0,NULL,timeframes[++shift]);
     }
   else
      if(!_isUp && shift > 1)
        {
         ChartSetSymbolPeriod(0,NULL,timeframes[--shift]);
        }
  }
```

Firstly, an array of all timeframes specified in the default toolbar, is described in the function. If you wish to switch between _all_ timeframes available in the MetaTrader 5, appropriate constants should be written to the array. However, in this case compatibility may be lost and the library may stop working with MQL4.

Next, we use standard functions to obtain the current period and to find it on the list.

Then chart timeframes are switched using the standard [ChartSetSymbolPeriod](https://www.mql5.com/en/docs/chart_operations/chartsetsymbolperiod) function, to which the period following the current one is passed.

Other functions used in the code should be clear without explanation. This is simply code.

#### A few simple functions

```
//+------------------------------------------------------------------+
//| Converts string command constants to keycodes                    |
//+------------------------------------------------------------------+
static int CUtilites::GetCurrentOperationChar(string keyString)
  {
   string keyValue = keyString;
   StringToUpper(keyValue);
   return(StringGetCharacter(keyValue,0));
  }
//+------------------------------------------------------------------+
//| Switch chart position to display on top of other objects         |
//+------------------------------------------------------------------+
static void CUtilites::ChangeChartZIndex(void)
  {
   ChartSetInteger(
      0,
      CHART_FOREGROUND,
      !(bool)ChartGetInteger(0,CHART_FOREGROUND)
   );
   ChartRedraw(0);
  }
//+------------------------------------------------------------------+
//| Returns a string name for any standard timeframe                 |
//| Parameters:                                                      |
//|    _timeframe - ENUM_TIMEFRAMES numeric value for which we need  |
//|                 to find a string name                            |
//| Return value:                                                    |
//|   string name of the required timeframe                          |
//+------------------------------------------------------------------+
static string CUtilites::GetTimeframeSymbolName(
   ENUM_TIMEFRAMES _timeframe=PERIOD_CURRENT  // Desired timeframe
)
  {
   ENUM_TIMEFRAMES current_timeframe; // current timeframe
   string result = "";
//---
   if(_timeframe == PERIOD_CURRENT)
     {
      current_timeframe = Period();
     }
   else
     {
      current_timeframe = _timeframe;
     }
//---
   switch(current_timeframe)
     {
      case PERIOD_M1:
         return "M1";
      case PERIOD_M2:
         return "M2";
      case PERIOD_M3:
         return "M3";
      case PERIOD_M4:
         return "M4";
      case PERIOD_M5:
         return "M5";
      case PERIOD_M6:
         return "M6";
      case PERIOD_M10:
         return "M10";
      case PERIOD_M12:
         return "M12";
      case PERIOD_M15:
         return "M15";
      case PERIOD_M20:
         return "M20";
      case PERIOD_M30:
         return "M30";
      case PERIOD_H1:
         return "H1";
      case PERIOD_H2:
         return "M1";
      case PERIOD_H3:
         return "H3";
      case PERIOD_H4:
         return "H4";
      case PERIOD_H6:
         return "H6";
      case PERIOD_H8:
         return "H8";
      case PERIOD_D1:
         return "D1";
      case PERIOD_W1:
         return "W1";
      case PERIOD_MN1:
         return "MN1";
      default:
         return "Unknown";
     }
  }
//+------------------------------------------------------------------+
//| Returns the standard color for the current timeframe             |
//+------------------------------------------------------------------+
static color CUtilites::GetTimeFrameColor(long _all_down_periods_value)
  {
   if(Is_Different_Colors)
     {
      switch((int)_all_down_periods_value)
        {
         case OBJ_PERIOD_M1:
            return (m1_color);
         case PERIOD_LOWER_M5:
            return (m5_color);
         case PERIOD_LOWER_M15:
            return (m15_color);
         case PERIOD_LOWER_M30:
            return (m30_color);
         case PERIOD_LOWER_H1:
            return (h1_color);
         case PERIOD_LOWER_H4:
            return (h4_color);
         case PERIOD_LOWER_D1:
            return (d1_color);
         case PERIOD_LOWER_W1:
            return (w1_color);
         case OBJ_ALL_PERIODS:
            return (mn1_color);
         default:
            return (common_color);
        }
     }
   else
     {
      return (common_color);
     }
  }
```

#### Extremum searching function and its application

The next function helps to find extreme points. Depending on parameters, the extremum points can be either to the right or to the left of the mouse pointer. Also, you can specify the desired number of bars to the left and to the right of the extremum.

```
//+------------------------------------------------------------------+
//| Returns the number of the nearest fractal in the selected        |
//|   direction                                                      |
//| Parameters:                                                      |
//|   starting_number - bar number at which the search starts        |
//|   is_search_right - search to the right (true) or left (false)   |
//|   is_up - if "true", search by High levels, otherwise - Low      |
//|   left_side_bars - number of bars on the left                    |
//|   right_side_bars - number of bars on the right                  |
//|   symbol - symbol name for search                                |
//|   timeframe - period for search                                  |
//| Return value:                                                    |
//|   the number of the extremum closest to starting_number,         |
//|   matching the specified parameters                              |                 |
//+------------------------------------------------------------------+
static int CUtilites::GetNearestExtremumBarNumber(
   int starting_number=0,              // Initial bar number
   const bool is_search_right=false,   // Direction to the right
   const bool is_up=false,             // Search by Highs
   const int left_side_bars=1,         // Number of bars to the left
   const int right_side_bars=1,        // Number of bars to the right
   const string symbol=NULL,           // The required symbol
   const ENUM_TIMEFRAMES timeframe=PERIOD_CURRENT // The required timeframe
)
  {
//---
   int   i,
         nextExtremum,
         sign = is_search_right ? -1 : 1;

//--- If the starting bar is specified incorrectly
//--- (is beyond the current chart borders)
//--- and search - towards the border...
   if((starting_number-right_side_bars<0
       && is_search_right)
      || (starting_number+left_side_bars>iBars(symbol,timeframe)
          && !is_search_right)
     )
     { //--- ...it is necessary to display an error message
      if(Print_Warning_Messages)
        {
         Print(DEBUG_MESSAGE_PREFIX,
               "Can't find extremum: ",
               "wrong direction");
         Print("left_side_bars = ",left_side_bars,"; ",
               "right_side_bars = ",right_side_bars);
        }
      return (-2);
     }
   else
     {
      //--- otherwise - the direction allows you to select the correct bar.
      //---   check all bars in the required direction,
      //---   as long as we are beyond the known chart borders
      while((starting_number-right_side_bars<0
             && !is_search_right)
            || (starting_number+left_side_bars>iBars(symbol,timeframe)
                && is_search_right)
           )
        {
         starting_number +=sign;
        }
     }
//---
   i=starting_number;

   //--- Preparation is complete. Proceed to search
   while(i-right_side_bars>=0
         && i+left_side_bars<iBars(symbol,timeframe)
        )
     {
      //--- Depending on the level, check the required extremum
      if(is_up)
        { //--- either the upper one
         nextExtremum = iHighest(
                           Symbol(),
                           Period(),
                           MODE_HIGH,
                           left_side_bars+right_side_bars+1,
                           i-right_side_bars
                        );
        }
      else
        {  //--- or the lower one
         nextExtremum = iLowest(
                           Symbol(),
                           Period(),
                           MODE_LOW,
                           left_side_bars+right_side_bars+1,
                           i-right_side_bars
                        );
        }
      if(nextExtremum == i) // If the current bar is an extremum,
        {
         return nextExtremum;  // the problem is solved
        }
      else // Otherwise - continue to shift the counter of the checked candlestick to the desired direction
         if(is_search_right)
           {
            if(nextExtremum<i)
              {
               i=nextExtremum;
              }
            else
              {
               i--;
              }
           }
         else
           {
            if(nextExtremum>i)
              {
               i=nextExtremum;
              }
            else
              {
               i++;
              }
           }
     }
//--- If the edge is reached but no extremum has been found,
   if(Print_Warning_Messages)
     {
      //--- show an error message.
      Print(DEBUG_MESSAGE_PREFIX,
            "Can't find extremum: ",
            "an incorrect starting point or wrong border conditions.");
      Print("left_side_bars = ",left_side_bars,"; ",
            "right_side_bars = ",right_side_bars);
     }
   return (-1);
  }
```

To draw trend lines, we need a function that finds the two nearest extreme points to the right of the mouse. This function can use the previous one:

```
//+------------------------------------------------------------------+
//| Finds 2 nearest extreme points to the right of the current       |
//| mouse pointer position                                           |
//| Parameters:                                                      |
//|    _is_up - search by High (true) or Low (false)                 |
//|    int &_p1 - bar number of the first point                      |
//|    int &_p2 - bar number of the second point                     |
//+------------------------------------------------------------------+
static void CUtilites::SetExtremumsBarsNumbers(
   bool _is_up, // search by High (true) or by Low (false)
   int &_p1,    // bar number of the first point
   int &_p2     // bar number of second point
)
  {
   int dropped_bar_number=CMouse::Bar();
//---
   _p1=CUtilites::GetNearestExtremumBarNumber(
          dropped_bar_number,
          true,
          _is_up,
          Fractal_Size_Left,
          Fractal_Size_Right
       );
   _p2=CUtilites::GetNearestExtremumBarNumber(
          _p1-1, // Bar to the left of the previous found extremum
          true,
          _is_up,
          Fractal_Size_Left,
          Fractal_Size_Right
       );
   if(_p2<0)
     {
      _p2=0;
     }
  }
```

#### Generating object names

In order to be able to draw series of the same objects, the objects must have unique names. The most efficient way is to use a prefix corresponding to this object type, and to add a unique number to it. Prefixes for different object types are listed in GlobalVariables.mqh.

Numbers are generated by the appropriate function.

```
//+------------------------------------------------------------------+
//| Returns the number of the next object in the series              |
//| Parameters:                                                      |
//|   prefix - name prefix for this group of objects.                |
//|   object_type - the type of objects to search in.                |
//|   only_prefixed - if "false", search in all objects              |
//|                      of this type, "true" - only the objects     |
//|                      with the specified prefix                   |
//| Return value:                                                    |
//|   number of the next object in series. If search by prefix,      |
//|      and the existing numbering has a gap, the next number       |
//|      will be at gap beginning.                                   |
//+------------------------------------------------------------------+
int               CUtilites::GetNextObjectNumber(
   const string prefix,
   const ENUM_OBJECT object_type,
   bool true
)
  {
   int count = ObjectsTotal(0,0,object_type),
       i,
       current_element_number,
       total_elements = 0;
   string current_element_name = "",
          comment_text = "";
//---
   if(only_prefixed)
     {
      for(i=0; i<count; i++)
        {
         current_element_name=ObjectName(0,i,0,object_type);
         if(StringSubstr(current_element_name,0,StringLen(prefix))==prefix)
           {
            current_element_number=
               (int)StringToInteger(
                  StringSubstr(current_element_name,
                               StringLen(prefix),
                               -1)
               );
            if(current_element_number!=total_elements)
              {
               break;
              }
            total_elements++;
           }
        }
     }
   else
     {
      total_elements = ObjectsTotal(0,-1,object_type);
      do
        {
         current_element_name = GetCurrentObjectName(
                                   prefix,
                                   object_type,
                                   total_elements
                                );
         if(ObjectFind(0,current_element_name)>=0)
           {
            total_elements++;
           }
        }
      while(ObjectFind(0,current_element_name)>=0);
     }
//---
   return(total_elements);
  }
```

Two search algorithms are implemented in the code. The first algorithm (the main one for this library) checks all objects corresponding to the type and having the specified prefix. Once it finds a free number, the algorithm returns it to the user. This allows filling "gaps" in numbering.

However, this algorithm is not suitable for the case when there can be several objects with the same number but with different suffixes. In earlier versions, when I used scripts to draw objects, I used such naming for pitchfork sets.

Therefore, the library has the second search method. The algorithm takes the total number of objects of this type and checks if there is a name starting with the same prefix and having the same index. If yes, the number is increased by 1 until a free value is found.

And when there is a number (or when it can be easily obtained using a function), a name can be easily created.

```
//+------------------------------------------------------------------+
//| Generates the current element name                               |
//| Parameters:                                                      |
//|    _prefix - name prefix for this group of objects.              |
//|    _type - the type of objects to search in.                     |
//|    _number - the number of the current object if it is ready.    |
//| Return value:                                                    |
//|    current object name string.                                   |
//+------------------------------------------------------------------+
string            CUtilites::GetCurrentObjectName(
   string _prefix,
   ENUM_OBJECT _type=OBJ_TREND,
   int _number = -1
)
  {
   int Current_Line_Number;

//--- Addition to the prefix - current timeframe
   string Current_Line_Name=IntegerToString(PeriodSeconds()/60)+"_"+_prefix;

//--- Get the element number
   if(_number<0)
     {
      Current_Line_Number = GetNextObjectNumber(Current_Line_Name,_type);
     }
   else
     {
      Current_Line_Number = _number;
     }

//--- Generate the name
   Current_Line_Name +=IntegerToString(Current_Line_Number,4,StringGetCharacter("0",0));

//---
   return (Current_Line_Name);
  }
```

#### The distance between adjacent bars (in pixels)

Sometimes it is necessary to calculate the distance to a certain point in the future. One of the most reliable ways is to calculate the distance in pixels between two adjacent bars and then to multiply it by the required coefficient (the desired number of bars for the indent).

The distance between adjacent bars can be calculated using the following function:

```
//+------------------------------------------------------------------+
//| Calculates a distance in pixels between two adjacent bars        |
//+------------------------------------------------------------------+
int        CUtilites::GetBarsPixelDistance(void)
  {
   double price; // arbitrary price on the chart (used for
                 // standard functions calculating coordinates
   datetime time1,time2;  // the time of the current and adjacent candlesticks
   int x1,x2,y1,y2;       // screen coordinates of two points
                          //   on adjacent candlesticks
   int deltha;            // result - the desired distance

//--- Initial settings
   price = iHigh(Symbol(),PERIOD_CURRENT,CMouse::Bar());
   time1 = CMouse::Time();

//--- Get the time of the current candlestick
   if(CMouse::Bar()<Bars(Symbol(),PERIOD_CURRENT)){ // if at the middle of the chart,
      time2 = time1+PeriodSeconds();                // take the candlestick on the left
   }
   else {                                           // otherwise
      time2 = time1;
      time1 = time1-PeriodSeconds();                // take the candlestick on the right
   }

//--- Convert coordinates form price/time values to screen pixels
   ChartTimePriceToXY(0,0,time1,price,x1,y1);
   ChartTimePriceToXY(0,0,time2,price,x2,y2);

//--- Calculate the distance
   deltha = MathAbs(x2-x1);
//---
   return (deltha);
  }
```

#### Function converting a string to an array of doubles

The most convenient way to set up Fibonacci levels using EA parameters, is to use strings which consist of comma separated values. However, MQL requires double number to be used for setting the level.

The following function can help in extracting number from a string.

```
//+------------------------------------------------------------------+
//| Converts a string with separators to an array of double          |
//|    numbers                                                       |
//| Parameters:                                                      |
//|    _haystack - source string for conversion                      |
//|    _result[] - resulting array                                   |
//|    _delimiter - separator character                              |
//+------------------------------------------------------------------+
static void CUtilites::StringToDoubleArray(
   string _haystack,             // source string
   double &_result[],            // array of results
   const string _delimiter=","   // separator
)
  {
//---
   string haystack_pieces[]; // Array of string fragments
   int pieces_count,         // Number of fragments
       i;                    // Counter
   string current_number=""; // The current string fragment (estimated number)

//--- Split the string into fragments
   pieces_count=StringSplit(_haystack,StringGetCharacter(_delimiter,0),haystack_pieces);
//--- Convert
   if(pieces_count>0)
     {
      ArrayResize(_result,pieces_count);
      for(i=0; i<pieces_count; i++)
        {
         StringTrimLeft(haystack_pieces[i]);
         StringTrimRight(haystack_pieces[i]);
         _result[i]=StringToDouble(haystack_pieces[i]);
        }
     }
   else
     {
      ArrayResize(_result,1);
      _result[0]=0;
     }
  }
```

### Drawing class: example of using utility functions

The article turns out to be lengthy, that is why most of the drawing functions will be described in the next article. However, in order to test some of the created functions, I will add here the code that draws simple straight lines (based on two nearest extrema).

#### Graphics.mqh file header

```
//+------------------------------------------------------------------+
//|                                                     Graphics.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                            https://www.mql5.com/ru/articles/7468 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://www.mql5.com/en/articles/7468"

//+------------------------------------------------------------------+
//| Class for plotting graphic objects                               |
//+------------------------------------------------------------------+
class CGraphics
  {
   //--- Fields
private:
   bool              m_Is_Trend_Ray;
   bool              m_Is_Change_Timeframe_On_Create;
   //--- Methods
private:
   //--- Sets general parameters for any newly created object
   void              CurrentObjectDecorate(
      const string _name,
      const color _color=clrNONE,
      const int _width = 1,
      const ENUM_LINE_STYLE _style = STYLE_SOLID
   );
public:
   //--- Default constructor
                     CGraphics();
   //--- Universal function for creating trend lines with specified parameters
   bool              TrendCreate(
      const long            chart_ID=0,        // chart ID
      const string          name="TrendLine",  // line name
      const int             sub_window=0,      // subwindow number
      datetime              time1=0,           // time of the first point
      double                price1=0,          // price of the first point
      datetime              time2=0,           // time of the second point
      double                price2=0,          // price of the second point
      const color           clr=clrRed,        // line color
      const ENUM_LINE_STYLE style=STYLE_SOLID, // line style
      const int             width=1,           // line width
      const bool            back=false,        // in the background
      const bool            selection=true,    // if the line is selected
      const bool            ray_right=false,   // ray to the right
      const bool            hidden=true,       // hide in the list of objects
      const long            z_order=0          // Z-Index
   );
   //--- Plots a trend line based on the two nearest (to the right of the mouse) extreme points
   void              DrawTrendLine(void);
   //--- Checks if the straight line is a ray
   bool              IsRay() {return m_Is_Trend_Ray;}
   //--- Sets the ray indication (whether the line should be extended to the right)
   void              IsRay(bool _is_ray) {m_Is_Trend_Ray = _is_ray;}
   //--- Checks if the display of objects created by the program should be changed on higher timeframes
   bool              IsChangeTimeframe(void) {return m_Is_Change_Timeframe_On_Create;}
   //--- Sets flags for the display of objects created by the program on higher timeframes
   void              IsChangeTimeframe(bool _is_tf_change) {m_Is_Change_Timeframe_On_Create = _is_tf_change;}
  };
```

#### Function setting general parameters for any newly created object

```
//+------------------------------------------------------------------+
//| Sets general parameters for all new objects                      |
//| Parameters:                                                      |
//|    _name - the name of the object being modified                 |
//|    _color - the color of the object being modified. If not set,  |
//|             standard color of the current period is used         |
//|    _width - object line width                                    |
//|    _style - object line type                                     |
//+------------------------------------------------------------------+
void CGraphics::CurrentObjectDecorate(
   const string _name,            // the name of the object being modified
   const color _color=clrNONE,    // the color of the object being modified
   const int _width = 1,          // object line width
   const ENUM_LINE_STYLE _style = STYLE_SOLID  // object line style
)
  {
   long timeframes;         // timeframes on which the object will be displayed
   color currentColor;      // object color
//--- Specify timeframes on which the object will be displayed
   if(Is_Change_Timeframe_On_Create)
     {
      timeframes = CUtilites::GetAllLowerTimeframes();
     }
   else
     {
      timeframes = OBJ_ALL_PERIODS;
     }
//--- Specify the object color
   if(_color != clrNONE)
     {
      currentColor = _color;
     }
   else
     {
      currentColor = CUtilites::GetTimeFrameColor(timeframes);
     }
//--- Set attributes
   ObjectSetInteger(0,_name,OBJPROP_COLOR,currentColor);            // Color
   ObjectSetInteger(0,_name,OBJPROP_TIMEFRAMES,timeframes);         // Periods
   ObjectSetInteger(0,_name,OBJPROP_HIDDEN,true);                   // Hide in the list of objects
   ObjectSetInteger(0,_name,OBJPROP_SELECTABLE,true);               // Ability to select
   ObjectSetInteger(0,_name,OBJPROP_SELECTED,Is_Select_On_Create);  // Stay selected after creation?
   ObjectSetInteger(0,_name,OBJPROP_WIDTH,_width);                  // Line width
   ObjectSetInteger(0,_name,OBJPROP_STYLE,_style);                  // Line style
  }
```

#### Straight line plotting function

```
//+------------------------------------------------------------------+
//| Universal function for creating trend lines with specified       |
//|    parameters                                                    |
//| Parameters:                                                      |
//|    chart_ID - chart ID                                           |
//|    name - line name                                              |
//|    sub_window - subwindow number                                 |
//|    time1 - time of the first point                               |
//|    price1 - price of the first point                             |
//|    time2 - time of the second point                              |
//|    price2 - price of the second point                            |
//|    clr - line color                                              |
//|    style - line style                                            |
//|    width - line width                                            |
//|    back - in the background                                      |
//|    selection - if the line is selected                           |
//|    ray_right - ray to the right                                  |
//|    hidden - hide in the list of objects                          |
//|    z_order - priority of mouse clicks (Z-Index)                  |
//| Return value:                                                    |
//|    indication of success. If line drawing failed,                |
//|    false is returned, otherwise - true                           |
//+------------------------------------------------------------------+
bool              CGraphics::TrendCreate(
      const long            chart_ID=0,        // chart ID
      const string          name="TrendLine",  // line name
      const int             sub_window=0,      // subwindow number
      datetime              time1=0,           // time of the first point
      double                price1=0,          // price of the first point
      datetime              time2=0,           // time of the second point
      double                price2=0,          // price of the second point
      const color           clr=clrRed,        // line color
      const ENUM_LINE_STYLE style=STYLE_SOLID, // line style
      const int             width=1,           // line width
      const bool            back=false,        // in the background
      const bool            selection=true,    // if the line is selected
      const bool            ray_right=false,   // ray to the right
      const bool            hidden=true,       // hide in the list of objects
      const long            z_order=0          // Z-Index
)
  {

//--- Reset the last error message
   ResetLastError();
//--- Create a line
   if(!ObjectCreate(chart_ID,name,OBJ_TREND,sub_window,time1,price1,time2,price2))
     {
      if(Print_Warning_Messages) // if failed, show an error message
        {
         Print(__FUNCTION__,
               ": Can't create trend line! Error code = ",GetLastError());
        }
      return(false);
     }

//--- Set additional attributes
   CurrentObjectDecorate(name,clr,width,style);

//--- line in the foreground (false) or in the background (true)
   ObjectSetInteger(chart_ID,name,OBJPROP_BACK,back);
//--- ray to the right (true) or exact borders (false)
   ObjectSetInteger(chart_ID,name,OBJPROP_RAY_RIGHT,ray_right);
//--- mouse click priority (Z-index)
   ObjectSetInteger(chart_ID,name,OBJPROP_ZORDER,z_order);
//--- Update the chart image
   ChartRedraw(0);
//--- Everything is good. The line has been drawn successfully.
   return(true);
  }
```

Let's use this common function as a basis and create another function that draws a straight line by two adjacent extreme points.

```
//+------------------------------------------------------------------+
//| Draws a trend line by nearest extreme points in the right        |
//+------------------------------------------------------------------+
void              CGraphics::DrawTrendLine(void)
  {
   int dropped_bar_number=CMouse::Bar(); // candlestick number under the mouse
   int p1=0,p2=0;                        // numbers of the first and seconds points
   string trend_name =                   // trend line name
      CUtilites::GetCurrentObjectName(Trend_Line_Prefix,OBJ_TREND);
   double
      price1=0,   // price of the first point
      price2=0,   // price of the second point
      tmp_price;  // variable for temporary storing of the price
   datetime
      time1=0,    // time of the first point
      time2=0,    // time of the second point
      tmp_time;
   int x1,x2,y1,y2;   // Point coordinates
   int window=0;      // Subwindow number

//--- Setting initial parameters
   if(CMouse::Below()) // If a mouse cursor is below the candlestick Low
     {
      //--- Find two extreme points below
      CUtilites::SetExtremumsBarsNumbers(false,p1,p2);

      //--- Determine point coordinates
      time1=iTime(Symbol(),PERIOD_CURRENT,p1);
      price1=iLow(Symbol(),PERIOD_CURRENT,p1);
      time2=iTime(Symbol(),PERIOD_CURRENT,p2);
      price2=iLow(Symbol(),PERIOD_CURRENT,p2);
     }
   else // otherwise
      if(CMouse::Above()) // If a mouse cursor is below the candlestick High
        {
         //--- Find two extreme points above
         CUtilites::SetExtremumsBarsNumbers(true,p1,p2);

         //--- Determine point coordinates
         time1=iTime(Symbol(),PERIOD_CURRENT,p1);
         price1=iHigh(Symbol(),PERIOD_CURRENT,p1);
         time2=iTime(Symbol(),PERIOD_CURRENT,p2);
         price2=iHigh(Symbol(),PERIOD_CURRENT,p2);

        }
//--- Draw the line
   TrendCreate(0,trend_name,0,
               time1,price1,time2,price2,
               CUtilites::GetTimeFrameColor(CUtilites::GetAllLowerTimeframes()),
               0,Trend_Line_Width,false,true,m_Is_Trend_Ray
              );

//--- Redraw the chart
   ChartRedraw(0);
  }
```

Please pay attention to the CUtilites::SetExtremumsBarsNumbers function call, which obtains bar numbers for points 1 and 2. Its code was described earlier. The rest seems clear so there is no need to add a long description

The final function draws a simple line based on two points. Depending on the **Is\_Trend\_Ray** global parameter, (described in the GlobalVariables.mqh file), the line will be either a ray extended to the right or a short segment between two extreme points.

![ Is_Trend_Ray = true](https://c.mql5.com/2/39/Line-Ray-GBPUSDH1.png)![ Is_Trend_Ray = false](https://c.mql5.com/2/39/LIne-Short-GBPUSDH1.png)

Let's add the possibility to extend the line using a keyboard.

### Creating a control block: setting the OnChartEvent method

Now that the basic functions are ready, we can customize keyboard shortcuts.

In Shortcuts.mqh, write the **CShortcuts::OnChartEvent** method.

```
//+------------------------------------------------------------------+
//| Event handling function                                          |
//+------------------------------------------------------------------+
void CShortcuts::OnChartEvent(
   const int id,
   const long &lparam,
   const double &dparam,
   const string &sparam
)
  {
//---
   switch(id)
     {
      //--- Save the coordinates of the mouse cursor
      case CHARTEVENT_MOUSE_MOVE:
         CMouse::SetCurrentParameters(id,lparam,dparam,sparam);
         break;

      //--- Handle keystrokes
      case CHARTEVENT_KEYDOWN:
         //--- Change the timeframe 1 level up
         if(CUtilites::GetCurrentOperationChar(Up_Key) == lparam)
           {
            CUtilites::ChangeTimeframes(true);
           };
         //--- Change the timeframe 1 level down
         if(CUtilites::GetCurrentOperationChar(Down_Key) == lparam)
           {
            CUtilites::ChangeTimeframes(false);
           };
         //--- Change the Z Index of the chart (chart on top of all objects)
         if(CUtilites::GetCurrentOperationChar(Z_Index_Key) == lparam)
           {
            CUtilites::ChangeChartZIndex();
           }
         //--- Draw a trend line
         if(CUtilites::GetCurrentOperationChar(Trend_Line_Key) == lparam)
           {
            m_graphics.DrawTrendLine();
           }
         //--- Switch the Is_Trend_Ray parameter
         if(CUtilites::GetCurrentOperationChar(Switch_Trend_Ray_Key) == lparam)
           {
            m_graphics.IsRay(!m_graphics.IsRay());
           }

         break;
         //---

     }
  }
```

### Keys used in the current library implementation

| Action | Key | Means |
| --- | --- | --- |
| Move **timeframe** **up** by main TFs (from the panel of TFs) | **U** | Up |
| Move **timeframe** **down** | **D** | Down |
| **Change chart Z level** (chart on top of all objects or not) | **Z** | Z order |
| Draw **a sloping trend line** based on two unidirectional extreme points closest to the mouse | **T** | Trend line |
| Switch **ray** mode for new lines | **R key** | Ray |

### Conclusion

The attached file contains the current version of the library. Also, the attachment includes three scripts.

- The first one is **Del-All-Graphics**. It deletes all graphic objects from the current window. In my terminal I set the **Ctrl+A keyboard shortcut (All) for this script.**
- The second script is **Del-All-Prefixed**. It allows deleting all objects with a prefix (for example, all trend lines or objects starting with H1). I call it using **Alt+R** (Remove).
- And finally, the third script ( **DeselectAllObjects**) allows deselecting all objects in the current window. My keyboard shortcut is **Ctrl+D** (Deselect as in Photoshop).

It is better to connect the library to an Expert Advisor, not an indicator, because if you connect to an indicator and try to use this indicator together with some other Expert Advisor, this may cause severe slowdowns. At least this was in my case. Of course, there might have been some other errors.

What will be implemented further.

The second version of the library will describe the implementation of the useful objects shown in the video. Some objects are primitive (like vertical or horizontal lines), other objects, such as lines of a specific the length, required more effort. Some of them still do not always work properly because of the "output error" or for some other reason. I will describe my decisions, and, of course, your feedback is welcome.

The third version will contain a graphical interface for configuring the parameters.

The fourth version, if it ever appears, will pr4esent a full-fledged assistant EA which will facilitate manual trading. I need advice from the community. I am not sure that I will apply any new ideas compared to existing solutions. Of course, I will draw the interface myself. However, all this applies to manual trading. So, do you think this development will be useful?

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7468](https://www.mql5.com/ru/articles/7468)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7468.zip "Download all attachments in the single ZIP archive")

[Shortcuts\_v1\_20200503.zip](https://www.mql5.com/en/articles/download/7468/shortcuts_v1_20200503.zip "Download Shortcuts_v1_20200503.zip")(36.18 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Master MQL5 from Beginner to Pro (Part VI): Basics of Developing Expert Advisors](https://www.mql5.com/en/articles/15727)
- [Master MQL5 from beginner to pro (Part V): Fundamental control flow operators](https://www.mql5.com/en/articles/15499)
- [Master MQL5 from beginner to pro (Part IV): About Arrays, Functions and Global Terminal Variables](https://www.mql5.com/en/articles/15357)
- [Master MQL5 from Beginner to Pro (Part III): Complex Data Types and Include Files](https://www.mql5.com/en/articles/14354)
- [Master MQL5 from beginner to pro (Part II): Basic data types and use of variable](https://www.mql5.com/en/articles/13749)
- [Master MQL5 from beginner to pro (Part I): Getting started with programming](https://www.mql5.com/en/articles/13594)
- [DRAKON visual programming language — communication tool for MQL developers and customers](https://www.mql5.com/en/articles/13324)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/351995)**
(17)


![Oleh Fedorov](https://c.mql5.com/avatar/2017/12/5A335A41-73FE.jpg)

**[Oleh Fedorov](https://www.mql5.com/en/users/certain)**
\|
26 Nov 2020 at 14:21

In case anyone hasn't seen it - there's a second version (https://www.mql5.com/en/articles/7908)


![Facundo Laje](https://c.mql5.com/avatar/2023/1/63b21560-ca73.jpg)

**[Facundo Laje](https://www.mql5.com/en/users/velasforexpips)**
\|
13 Feb 2021 at 12:21

Excellent! Please continue with this kind of articles.


![Alexey Brusenskiy](https://c.mql5.com/avatar/2022/10/633EEC6F-5856.jpg)

**[Alexey Brusenskiy](https://www.mql5.com/en/users/zondik86)**
\|
16 Aug 2023 at 12:21

Help to implement the following thing:

There is a figure (line, rectangle).

Activate the tool by pressing the hotkey, set the beginning with the left mouse button and the end with the left mouse button.

as here

линии и прямоугольники \- YouTube

[Photo image of Alexey Brusenskiy](https://www.youtube.com/channel/UCpDa83zoB39wSFRk5f5r2pA?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7468)

Alexey Brusenskiy

6 subscribers

[линии и прямоугольники](https://www.youtube.com/watch?v=s10QDSrsWpY)

Alexey Brusenskiy

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

0:00

0:00 / 0:16

•Live

•

![Oleh Fedorov](https://c.mql5.com/avatar/2017/12/5A335A41-73FE.jpg)

**[Oleh Fedorov](https://www.mql5.com/en/users/certain)**
\|
17 Aug 2023 at 14:50

**Alexey Brusenskiy [#](https://www.mql5.com/ru/forum/339640#comment_48775583):**

Help to implement the following thing:

There is a figure (line, rectangle).

Activate the tool by pressing the hotkey, set the beginning with the left mouse button and the end with the left mouse button.

as here

Okay.

Are you only interested in straight lines and rectangles? Or do you need something else?

There are still a couple of non-obvious bugs to fix, so I'll try to post all the changes to kodobase soon.

![Алексей Брукс](https://c.mql5.com/avatar/avatar_na2.png)

**[Алексей Брукс](https://www.mql5.com/en/users/zonda86)**
\|
17 Aug 2023 at 22:26

**Oleh Fedorov [#](https://www.mql5.com/ru/forum/339640#comment_48802286):**

All right. (chuckles)

Are you only interested in straight lines and rectangles? Or do you need something else?

There are still a couple of non-obvious bugs to fix, so I'll try to post all the changes to kodobase soon.

They're the only ones.

![Practical application of neural networks in trading](https://c.mql5.com/2/37/neural_DLL.png)[Practical application of neural networks in trading](https://www.mql5.com/en/articles/7031)

In this article, we will consider the main aspects of integration of neural networks and the trading terminal, with the purpose of creating a fully featured trading robot.

![Timeseries in DoEasy library (part 44): Collection class of indicator buffer objects](https://c.mql5.com/2/39/MQL5-avatar-doeasy-library.png)[Timeseries in DoEasy library (part 44): Collection class of indicator buffer objects](https://www.mql5.com/en/articles/7886)

The article deals with creating a collection class of indicator buffer objects. I am going to test the ability to create and work with any number of buffers for indicators (the maximum number of buffers that can be created in MQL indicators is 512).

![Practical application of neural networks in trading. It's time to practice](https://c.mql5.com/2/39/neural_DLL.png)[Practical application of neural networks in trading. It's time to practice](https://www.mql5.com/en/articles/7370)

The article provides a description and instructions for the practical use of neural network modules on the Matlab platform. It also covers the main aspects of creation of a trading system using the neural network module. In order to be able to introduce the complex within one article, I had to modify it so as to combine several neural network module functions in one program.

![MQL as a Markup Tool for the Graphical Interface of MQL Programs (Part 3). Form Designer](https://c.mql5.com/2/38/MQL5-avatar-dialog_form__2.png)[MQL as a Markup Tool for the Graphical Interface of MQL Programs (Part 3). Form Designer](https://www.mql5.com/en/articles/7795)

In this paper, we are completing the description of our concept of building the window interface of MQL programs, using the structures of MQL. Specialized graphical editor will allow to interactively set up the layout that consists of the basic classes of the GUI elements and then export it into the MQL description to use it in your MQL project. The paper presents the internal design of the editor and a user guide. Source codes are attached.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/7468&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082961793640305193)

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