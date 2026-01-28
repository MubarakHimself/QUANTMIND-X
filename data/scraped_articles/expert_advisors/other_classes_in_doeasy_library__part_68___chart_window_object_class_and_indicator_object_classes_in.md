---
title: Other classes in DoEasy library (Part 68): Chart window object class and indicator object classes in the chart window
url: https://www.mql5.com/en/articles/9236
categories: Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:27:44.328408
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/9236&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071849042503217053)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/9236#node01)
- [Class of the chart window with indicator objects](https://www.mql5.com/en/articles/9236#node02)
- [Test](https://www.mql5.com/en/articles/9236#node03)
- [What's next?](https://www.mql5.com/en/articles/9236#node04)


### Concept

[In the previous article](https://www.mql5.com/en/articles/9213), I started the development of the chart object class preparing its first version. The object describes one terminal chart with all its parameters. It allows managing its properties — getting and installing the window size and chart elements display parameters.

In our case, a chart may feature several windows, which in turn may contain indicators. The windows have their own size, while for now our chart object can simply return the parameters of the specified subwindow located on it and manage its height. One subwindow (just like the main chart window) may contain different number of indicators. We should consider this to be able to request a necessary window located on the chart while accessing the chart object. Using the obtained window object, we are then able to request the list of its indicators obtaining the handle of the necessary one for further processing.

Today I will create two objects — the indicator object in the chart window, which will describe some indicator parameters for its identification, and the chart window object, which is to store its size and the list of indicators (indicator objects in the chart window) attached to it. The chart object I started developing in the previous article is to have the list of window objects attached to it (including the main chart window).

In the future, such a hierarchy will greatly facilitate our work with many charts and their subwindows containing the lists of indicators. Unfortunately, it is still too early to create the collection class of chart objects announced in the previous article. First, we need to complete all main chart object improvements. This is what I am going to do here.

### Class of the chart window with indicator objects

First, let's add all the necessary messages to the library.

In \\MQL5\\Include\\DoEasy\ **Data.mqh**, add the indices of the new messages:

```
   MSG_CHART_OBJ_CHART_WINDOW,                        // Main chart window
   MSG_CHART_OBJ_CHART_SUBWINDOW,                     // Chart subwindow
   MSG_CHART_OBJ_CHART_SUBWINDOWS_NUM,                // Subwindows
   MSG_CHART_OBJ_INDICATORS_MW_NAME_LIST,             // Indicators in the main chart window
   MSG_CHART_OBJ_INDICATORS_SW_NAME_LIST,             // Indicators in the chart window
   MSG_CHART_OBJ_INDICATOR,                           // Indicator
   MSG_CHART_OBJ_INDICATORS_TOTAL,                    // Indicators
   MSG_CHART_OBJ_WINDOW_N,                            // Window
   MSG_CHART_OBJ_INDICATORS_NONE,                     // No indicators

  };
//+------------------------------------------------------------------+
```

and the message texts corresponding to the newly added indices:

```
   {"Главное окно графика","Main chart window"},
   {"Подокно графика","Chart subwindow"},
   {"Подокон","Subwindows"},
   {"Индикаторы в главном окне графика","Indicators in the main chart window"},
   {"Индикаторы в окне графика","Indicators in the chart window"},
   {"Индикатор","Indicator"},
   {"Индикаторов","Indicators total"},
   {"Окно","Window"},
   {"Отсутствуют","No indicators"},

  };
//+---------------------------------------------------------------------+
```

While creating the enumeration of integer chart object properties, I deliberately skipped three properties inherent not only to the main window object, but also to all chart subwindows:

- Subwindow visibility,

- Distance in Y axis pixels between the upper frame of the indicator subwindow and the upper frame of the chart main window,

- Chart height in pixels.

These are to be the main properties of the chart window object (apart from the subwindow visibility we are going to define from the chart object). The list of chart windows is to be present in each chart object, and each object is to have its own values for these properties. Each window object is to feature the list of indicators attached to the window. So we will need to set additional constants in the enumerations of integer and string chart object properties.

In \\MQL5\\Include\\DoEasy\\Defines.mqh, uncomment previously defined and commented out properties — these are the chart window object properties and add the new ones — for the indicator object in the chart window:

```
//+------------------------------------------------------------------+
//| Chart integer property                                           |
//+------------------------------------------------------------------+
enum ENUM_CHART_PROP_INTEGER
  {
   CHART_PROP_ID = 0,                                 // Chart ID
   CHART_PROP_TIMEFRAME,                              // Chart timeframe
   CHART_PROP_SHOW,                                   // Price chart drawing
   CHART_PROP_IS_OBJECT,                              // Chart object (OBJ_CHART) identification attribute
   CHART_PROP_BRING_TO_TOP,                           // Show chart above all others
   CHART_PROP_CONTEXT_MENU,                           // Enable/disable access to the context menu using the right click
   CHART_PROP_CROSSHAIR_TOOL,                         // Enable/disable access to the Crosshair tool using the middle click
   CHART_PROP_MOUSE_SCROLL,                           // Scroll the chart horizontally using the left mouse button
   CHART_PROP_EVENT_MOUSE_WHEEL,                      // Send messages about mouse wheel events (CHARTEVENT_MOUSE_WHEEL) to all MQL5 programs on a chart
   CHART_PROP_EVENT_MOUSE_MOVE,                       // Send messages about mouse button click and movement events (CHARTEVENT_MOUSE_MOVE) to all MQL5 programs on a chart
   CHART_PROP_EVENT_OBJECT_CREATE,                    // Send messages about the graphical object creation event (CHARTEVENT_OBJECT_CREATE) to all MQL5 programs on a chart
   CHART_PROP_EVENT_OBJECT_DELETE,                    // Send messages about the graphical object destruction event (CHARTEVENT_OBJECT_DELETE) to all MQL5 programs on a chart
   CHART_PROP_MODE,                                   // Type of the chart (candlesticks, bars or line (ENUM_CHART_MODE))
   CHART_PROP_FOREGROUND,                             // Price chart in the foreground
   CHART_PROP_SHIFT,                                  // Mode of shift of the price chart from the right border
   CHART_PROP_AUTOSCROLL,                             // The mode of automatic shift to the right border of the chart
   CHART_PROP_KEYBOARD_CONTROL,                       // Allow managing the chart using a keyboard
   CHART_PROP_QUICK_NAVIGATION,                       // Allow the chart to intercept Space and Enter key strokes to activate the quick navigation bar
   CHART_PROP_SCALE,                                  // Scale
   CHART_PROP_SCALEFIX,                               // Fixed scale mode
   CHART_PROP_SCALEFIX_11,                            // 1:1 scale mode
   CHART_PROP_SCALE_PT_PER_BAR,                       // The mode of specifying the scale in points per bar
   CHART_PROP_SHOW_TICKER,                            // Display a symbol ticker in the upper left corner
   CHART_PROP_SHOW_OHLC,                              // Display OHLC values in the upper left corner
   CHART_PROP_SHOW_BID_LINE,                          // Display Bid value as a horizontal line on the chart
   CHART_PROP_SHOW_ASK_LINE,                          // Display Ask value as a horizontal line on a chart
   CHART_PROP_SHOW_LAST_LINE,                         // Display Last value as a horizontal line on a chart
   CHART_PROP_SHOW_PERIOD_SEP,                        // Display vertical separators between adjacent periods
   CHART_PROP_SHOW_GRID,                              // Display a grid on the chart
   CHART_PROP_SHOW_VOLUMES,                           // Display volumes on a chart
   CHART_PROP_SHOW_OBJECT_DESCR,                      // Display text descriptions of objects
   CHART_PROP_VISIBLE_BARS,                           // Number of bars on a chart that are available for display
   CHART_PROP_WINDOWS_TOTAL,                          // The total number of chart windows including indicator subwindows
   CHART_PROP_WINDOW_IS_VISIBLE,                      // Subwindow visibility
   CHART_PROP_WINDOW_HANDLE,                          // Chart window handle
   CHART_PROP_WINDOW_YDISTANCE,                       // Distance in Y axis pixels between the upper frame of the indicator subwindow and the upper frame of the chart main window
   CHART_PROP_FIRST_VISIBLE_BAR,                      // Number of the first visible bar on the chart
   CHART_PROP_WIDTH_IN_BARS,                          // Width of the chart in bars
   CHART_PROP_WIDTH_IN_PIXELS,                        // Width of the chart in pixels
   CHART_PROP_HEIGHT_IN_PIXELS,                       // Height of the chart in pixels
   CHART_PROP_COLOR_BACKGROUND,                       // Color of background of the chart
   CHART_PROP_COLOR_FOREGROUND,                       // Color of axes, scale and OHLC line
   CHART_PROP_COLOR_GRID,                             // Grid color
   CHART_PROP_COLOR_VOLUME,                           // Color of volumes and position opening levels
   CHART_PROP_COLOR_CHART_UP,                         // Color for the up bar, shadows and body borders of bull candlesticks
   CHART_PROP_COLOR_CHART_DOWN,                       // Color of down bar, its shadow and border of body of the bullish candlestick
   CHART_PROP_COLOR_CHART_LINE,                       // Color of the chart line and the Doji candlesticks
   CHART_PROP_COLOR_CANDLE_BULL,                      // Color of body of a bullish candlestick
   CHART_PROP_COLOR_CANDLE_BEAR,                      // Color of body of a bearish candlestick
   CHART_PROP_COLOR_BID,                              // Color of the Bid price line
   CHART_PROP_COLOR_ASK,                              // Color of the Ask price line
   CHART_PROP_COLOR_LAST,                             // Color of the last performed deal's price line (Last)
   CHART_PROP_COLOR_STOP_LEVEL,                       // Color of stop order levels (Stop Loss and Take Profit)
   CHART_PROP_SHOW_TRADE_LEVELS,                      // Display trade levels on the chart (levels of open positions, Stop Loss, Take Profit and pending orders)
   CHART_PROP_DRAG_TRADE_LEVELS,                      // Enable the ability to drag trading levels on a chart using mouse
   CHART_PROP_SHOW_DATE_SCALE,                        // Display the time scale on a chart
   CHART_PROP_SHOW_PRICE_SCALE,                       // Display a price scale on a chart
   CHART_PROP_SHOW_ONE_CLICK,                         // Display the quick trading panel on the chart
   CHART_PROP_IS_MAXIMIZED,                           // Chart window maximized
   CHART_PROP_IS_MINIMIZED,                           // Chart window minimized
   CHART_PROP_IS_DOCKED,                              // Chart window docked
   CHART_PROP_FLOAT_LEFT,                             // Left coordinate of the undocked chart window relative to the virtual screen
   CHART_PROP_FLOAT_TOP,                              // Upper coordinate of the undocked chart window relative to the virtual screen
   CHART_PROP_FLOAT_RIGHT,                            // Right coordinate of the undocked chart window relative to the virtual screen
   CHART_PROP_FLOAT_BOTTOM,                           // Bottom coordinate of the undocked chart window relative to the virtual screen
   //--- CWndInd
   CHART_PROP_WINDOW_IND_HANDLE,                      // Indicator handle in the chart window
   CHART_PROP_WINDOW_IND_INDEX,                       // Indicator index in the chart window
  };
#define CHART_PROP_INTEGER_TOTAL (67)                 // Total number of integer properties
#define CHART_PROP_INTEGER_SKIP  (0)                  // Number of integer DOM properties not used in sorting
//+------------------------------------------------------------------+
//| Chart real properties                                            |
//+------------------------------------------------------------------+
enum ENUM_CHART_PROP_DOUBLE
  {
   CHART_PROP_SHIFT_SIZE = CHART_PROP_INTEGER_TOTAL,  // Shift size of the zero bar from the right border in %
   CHART_PROP_FIXED_POSITION,                         // Chart fixed position from the left border in %
   CHART_PROP_FIXED_MAX,                              // Chart fixed maximum
   CHART_PROP_FIXED_MIN,                              // Chart fixed minimum
   CHART_PROP_POINTS_PER_BAR,                         // Scale in points per bar
   CHART_PROP_PRICE_MIN,                              // Chart minimum
   CHART_PROP_PRICE_MAX,                              // Chart maximum
  };
#define CHART_PROP_DOUBLE_TOTAL  (7)                  // Total number of real properties
#define CHART_PROP_DOUBLE_SKIP   (0)                  // Number of real properties not used in sorting
//+------------------------------------------------------------------+
//| Chart string properties                                          |
//+------------------------------------------------------------------+
enum ENUM_CHART_PROP_STRING
  {
   CHART_PROP_COMMENT = (CHART_PROP_INTEGER_TOTAL+CHART_PROP_DOUBLE_TOTAL), // Chart comment text
   CHART_PROP_EXPERT_NAME,                            // Name of an EA launched on the chart
   CHART_PROP_SCRIPT_NAME,                            // Name of a script launched on the chart
   CHART_PROP_INDICATOR_NAME,                         // Name of an indicator launched on the chart
   CHART_PROP_SYMBOL,                                 // Chart symbol
  };
#define CHART_PROP_STRING_TOTAL  (5)                  // Total number of string properties
//+------------------------------------------------------------------+
```

Let's change the number of properties featuring enumerations with new constants — the number of integer properties increased **from 62 to 67**, while the number of string ones is increased **from 4 to 5**.

In the enumeration of possible chart object sorting criteria, add the new criteria corresponding to the newly added properties:

```
//+------------------------------------------------------------------+
//| Possible chart sorting criteria                                  |
//+------------------------------------------------------------------+
#define FIRST_CHART_DBL_PROP  (CHART_PROP_INTEGER_TOTAL-CHART_PROP_INTEGER_SKIP)
#define FIRST_CHART_STR_PROP  (CHART_PROP_INTEGER_TOTAL-CHART_PROP_INTEGER_SKIP+CHART_PROP_DOUBLE_TOTAL-CHART_PROP_DOUBLE_SKIP)
enum ENUM_SORT_CHART_MODE
  {
//--- Sort by integer properties
   SORT_BY_CHART_SHOW = 0,                            // Sort by the price chart drawing attribute
   SORT_BY_CHART_IS_OBJECT,                           // Sort by chart object (OBJ_CHART) identification attribute
   SORT_BY_CHART_BRING_TO_TOP,                        // Sort by the flag of displaying a chart above all others
   SORT_BY_CHART_CONTEXT_MENU,                        // Sort by the flag of enabling/disabling access to the context menu using the right click
   SORT_BY_CHART_CROSSHAIR_TOO,                       // Sort by the flag of enabling/disabling access to the Crosshair tool using the middle click
   SORT_BY_CHART_MOUSE_SCROLL,                        // Sort by the flag of scrolling the chart horizontally using the left mouse button
   SORT_BY_CHART_EVENT_MOUSE_WHEEL,                   // Sort by the flag of sending messages about mouse wheel events to all MQL5 programs on a chart
   SORT_BY_CHART_EVENT_MOUSE_MOVE,                    // Sort by the flag of sending messages about mouse button click and movement events to all MQL5 programs on a chart
   SORT_BY_CHART_EVENT_OBJECT_CREATE,                 // Sort by the flag of sending messages about the graphical object creation event to all MQL5 programs on a chart
   SORT_BY_CHART_EVENT_OBJECT_DELETE,                 // Sort by the flag of sending messages about the graphical object destruction event to all MQL5 programs on a chart
   SORT_BY_CHART_MODE,                                // Sort by chart type
   SORT_BY_CHART_FOREGROUND,                          // Sort by the "Price chart in the foreground" flag
   SORT_BY_CHART_SHIFT,                               // Sort by the "Mode of shift of the price chart from the right border" flag
   SORT_BY_CHART_AUTOSCROLL,                          // Sort by the "The mode of automatic shift to the right border of the chart" flag
   SORT_BY_CHART_KEYBOARD_CONTROL,                    // Sort by the flag allowing the chart management using a keyboard
   SORT_BY_CHART_QUICK_NAVIGATION,                    // Sort by the flag allowing the chart to intercept Space and Enter key strokes to activate the quick navigation bar
   SORT_BY_CHART_SCALE,                               // Sort by scale
   SORT_BY_CHART_SCALEFIX,                            // Sort by the fixed scale flag
   SORT_BY_CHART_SCALEFIX_11,                         // Sort by the 1:1 scale flag
   SORT_BY_CHART_SCALE_PT_PER_BAR,                    // Sort by the flag of specifying the scale in points per bar
   SORT_BY_CHART_SHOW_TICKER,                         // Sort by the flag displaying a symbol ticker in the upper left corner
   SORT_BY_CHART_SHOW_OHLC,                           // Sort by the flag displaying OHLC values in the upper left corner
   SORT_BY_CHART_SHOW_BID_LINE,                       // Sort by the flag displaying Bid value as a horizontal line on the chart
   SORT_BY_CHART_SHOW_ASK_LINE,                       // Sort by the flag displaying Ask value as a horizontal line on the chart
   SORT_BY_CHART_SHOW_LAST_LINE,                      // Sort by the flag displaying Last value as a horizontal line on the chart
   SORT_BY_CHART_SHOW_PERIOD_SEP,                     // Sort by the flag displaying vertical separators between adjacent periods
   SORT_BY_CHART_SHOW_GRID,                           // Sort by the flag of displaying a grid on the chart
   SORT_BY_CHART_SHOW_VOLUMES,                        // Sort by the mode of displaying volumes on a chart
   SORT_BY_CHART_SHOW_OBJECT_DESCR,                   // Sort by the flag of displaying object text descriptions
   SORT_BY_CHART_VISIBLE_BARS,                        // Sort by the number of bars on a chart that are available for display
   SORT_BY_CHART_WINDOWS_TOTAL,                       // Sort by the total number of chart windows including indicator subwindows
   SORT_BY_CHART_WINDOW_IS_VISIBLE,                   // Sort by the subwindow visibility flag
   SORT_BY_CHART_WINDOW_HANDLE,                       // Sort by the chart handle
   SORT_BY_CHART_WINDOW_YDISTANCE,                    // Sort by the distance in Y axis pixels between the upper frame of the indicator subwindow and the upper frame of the chart main window
   SORT_BY_CHART_FIRST_VISIBLE_BAR,                   // Sort by the number of the first visible bar on the chart
   SORT_BY_CHART_WIDTH_IN_BARS,                       // Sort by the width of the chart in bars
   SORT_BY_CHART_WIDTH_IN_PIXELS,                     // Sort by the width of the chart in pixels
   SORT_BY_CHART_HEIGHT_IN_PIXELS,                    // Sort by the height of the chart in pixels
   SORT_BY_CHART_COLOR_BACKGROUND,                    // Sort by the color of the chart background
   SORT_BY_CHART_COLOR_FOREGROUND,                    // Sort by color of axes, scale and OHLC line
   SORT_BY_CHART_COLOR_GRID,                          // Sort by grid color
   SORT_BY_CHART_COLOR_VOLUME,                        // Sort by the color of volumes and position opening levels
   SORT_BY_CHART_COLOR_CHART_UP,                      // Sort by the color for the up bar, shadows and body borders of bull candlesticks
   SORT_BY_CHART_COLOR_CHART_DOWN,                    // Sort by the color of down bar, its shadow and border of body of the bullish candlestick
   SORT_BY_CHART_COLOR_CHART_LINE,                    // Sort by the color of the chart line and the Doji candlesticks
   SORT_BY_CHART_COLOR_CANDLE_BULL,                   // Sort by the color of a bullish candlestick body
   SORT_BY_CHART_COLOR_CANDLE_BEAR,                   // Sort by the color of a bearish candlestick body
   SORT_BY_CHART_COLOR_BID,                           // Sort by the color of the Bid price line
   SORT_BY_CHART_COLOR_ASK,                           // Sort by the color of the Ask price line
   SORT_BY_CHART_COLOR_LAST,                          // Sort by the color of the last performed deal's price line (Last)
   SORT_BY_CHART_COLOR_STOP_LEVEL,                    // Sort by the color of stop order levels (Stop Loss and Take Profit)
   SORT_BY_CHART_SHOW_TRADE_LEVELS,                   // Sort by the flag of displaying trading levels on the chart
   SORT_BY_CHART_DRAG_TRADE_LEVELS,                   // Sort by the flag enabling the ability to drag trading levels on a chart using mouse
   SORT_BY_CHART_SHOW_DATE_SCALE,                     // Sort by the flag of displaying the time scale on the chart
   SORT_BY_CHART_SHOW_PRICE_SCALE,                    // Sort by the flag of displaying the price scale on the chart
   SORT_BY_CHART_SHOW_ONE_CLICK,                      // Sort by the flag of displaying the quick trading panel on the chart
   SORT_BY_CHART_IS_MAXIMIZED,                        // Sort by the "Chart window maximized" flag
   SORT_BY_CHART_IS_MINIMIZED,                        // Sort by the "Chart window minimized" flag
   SORT_BY_CHART_IS_DOCKED,                           // Sort by the "Chart window docked" flag
   SORT_BY_CHART_FLOAT_LEFT,                          // Sort by the left coordinate of the undocked chart window relative to the virtual screen
   SORT_BY_CHART_FLOAT_TOP,                           // Sort by the upper coordinate of the undocked chart window relative to the virtual screen
   SORT_BY_CHART_FLOAT_RIGHT,                         // Sort by the right coordinate of the undocked chart window relative to the virtual screen
   SORT_BY_CHART_FLOAT_BOTTOM,                        // Sort by the bottom coordinate of the undocked chart window relative to the virtual screen
   SORT_BY_CHART_WINDOW_IND_HANDLE,                   // Sort by the indicator handle in the chart window
   SORT_BY_CHART_WINDOW_IND_INDEX,                    // Sort by the indicator index in the chart window
//--- Sort by real properties
   SORT_BY_CHART_SHIFT_SIZE = FIRST_CHART_DBL_PROP,   // Sort by the shift size of the zero bar from the right border in %
   SORT_BY_CHART_FIXED_POSITION,                      // Sort by the chart fixed position from the left border in %
   SORT_BY_CHART_FIXED_MAX,                           // Sort by the fixed chart maximum
   SORT_BY_CHART_FIXED_MIN,                           // Sort by the fixed chart minimum
   SORT_BY_CHART_POINTS_PER_BAR,                      // Sort by the scale value in points per bar
   SORT_BY_CHART_PRICE_MIN,                           // Sort by the chart minimum
   SORT_BY_CHART_PRICE_MAX,                           // Sort by the chart maximum
//--- Sort by string properties
   SORT_BY_CHART_COMMENT = FIRST_CHART_STR_PROP,      // Sort by a comment text on the chart
   SORT_BY_CHART_EXPERT_NAME,                         // Sort by a name of an EA launched on the chart
   SORT_BY_CHART_SCRIPT_NAME,                         // Sort by a name of a script launched on the chart
   SORT_BY_CHART_INDICATOR_NAME,                      // Sort by a name of an indicator launched on the chart
   SORT_BY_CHART_SYMBOL,                              // Sort by chart symbol
  };
//+------------------------------------------------------------------+
```

In the first chart object version, the list of criteria was incorrect because commented out integer chart properties were added to the sorting criteria and there was no sorting by symbol name. I have fixed this in the current article.

Now I need to create two classes — the class of the indicator object in the chart window and the class of the chart window object. Let's add them into a single file at once.

In \\MQL5\\Include\\DoEasy\\Objects\ **Chart\**, create a new file **ChartWnd.mqh** of the **CWndInd** (indicator in the chart window) and **CChartWnd** (chart window) classes.

The CWndInd class should be inherited [from the base class of the CObject standard library](https://www.mql5.com/en/docs/standardlibrary/cobject), while the CChartWnd class — [from the base object of all CBaseObj library objects](https://www.mql5.com/en/articles/7071#node01).

Many different indicators can be attached to the chart window. The chart window object should know about them so that we are always able to get the handle of the necessary indicator from the chart window to work with it. We do not need many different parameters to identify indicators. The indicator handle, its short name and an index of the chart window the indicator is attached to are sufficient. Therefore, the indicator object class in the chart window will be the simplest one and will be inherited from the basic object of the standard library so that we can add all these objects [to the list of pointers to CArrayObj objects](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj), which is located in the chart window object.

Let's write the code of the new class in the newly created ChartWnd.mqh file:

```
//+------------------------------------------------------------------+
//|                                                     ChartWnd.mqh |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\..\Objects\BaseObj.mqh"
//+------------------------------------------------------------------+
//| Chart window indicator object class                              |
//+------------------------------------------------------------------+
class CWndInd : public CObject
  {
private:
   long              m_chart_id;                         // Chart ID
   string            m_name;                             // Indicator short name
   int               m_index;                            // Window index on the chart
   int               m_handle;                           // Indicator handle
public:
//--- Return itself
   CWndInd          *GetObject(void)                     { return &this;         }
//--- Return (1) indicator name, (2) window index and (3) indicator handle
   string            Name(void)                    const { return this.m_name;   }
   int               Index(void)                   const { return this.m_index;  }
   int               Handle(void)                  const { return this.m_handle; }

//--- Display the description of object properties in the journal (dash=true - hyphen before the description, false - description only)
   void              Print(const bool dash=false)        { ::Print((dash ? "- " : "")+this.Header());                      }
//--- Return the object short name
   string            Header(void)                  const { return CMessage::Text(MSG_CHART_OBJ_INDICATOR)+" "+this.Name(); }

//--- Compare CWndInd objects with each other by the specified property
   virtual int       Compare(const CObject *node,const int mode=0) const;

//--- Constructors
                     CWndInd(void);
                     CWndInd(const int handle,const string name,const int index) : m_handle(handle),m_name(name),m_index(index) {}
  };
//+------------------------------------------------------------------+
//| Compare CWndInd objects with each other by the specified property|
//+------------------------------------------------------------------+
int CWndInd::Compare(const CObject *node,const int mode=0) const
  {
   const CWndInd *obj_compared=node;
   if(mode==CHART_PROP_WINDOW_IND_HANDLE) return(this.Handle()>obj_compared.Handle() ? 1 : this.Handle()<obj_compared.Handle() ? -1 : 0);
   else if(mode==CHART_PROP_WINDOW_IND_INDEX) return(this.Index()>obj_compared.Index() ? 1 : this.Index()<obj_compared.Index() ? -1 : 0);
   return(this.Name()==obj_compared.Name() ? 0 : this.Name()<obj_compared.Name() ? -1 : 1);
  }
//+------------------------------------------------------------------+
```

This is an entire indicator object class in the chart window.

The private section features the class member variables to be stored:

- ID of the chart the window with the indicator is located at,

- indicator short name (in the chart windows, the indicators are identified by the terminal by its short names),

- index of the chart window where the indicator is located (0 — main chart window, 1, etc. — chart subwindows),
- indicator handle.


This data is sufficient to store the list of all indicators attached to the window in the chart window object. The list itself is to contain the objects allowing us to find the necessary indicator and return its handle for further work.

The public methods returning the values of the variables mentioned above are self-explanatory. Let's consider some other class methods.

**The method returning the short name of the indicator object:**

```
//--- Return the object short name
   string            Header(void)                  const { return CMessage::Text(MSG_CHART_OBJ_INDICATOR)+" "+this.Name(); }
```

Simply return the Indicator header + the indicator short name.

**The method displaying the description of the indicator object properties in the journal:**

```
//--- Display the description of object properties in the journal (dash=true - hyphen before the description, false - description only)
   void              Print(const bool dash=false)        { ::Print((dash ? "- " : "")+this.Header());                      }
```

Since there may be several indicators in the chart window, they are to be displayed as a list with a header. To make the list in the journal more visually appealing, I will use the hyphen before an indicator name. The method input will indicate the necessity to display the hyphen.

We have two constructors: the default and the parametric one. The default constructor may be helpful for creating an "empty" indicator object in the window, while the parametric one is to be used as the main class constructor when creating the list of indicators in the chart window object class.

The **parametric constructor** receives the indicator handle, its short name and the index of the subwindow the indicator is located in.

```
//--- Constructors
                     CWndInd(void);
                     CWndInd(const int handle,const string name,const int index) : m_handle(handle),m_name(name),m_index(index) {}
```

All parameter values passed to the method are immediately assigned to the class variables in its initialization list.

**The method for comparing the chart window indicator objects by the specified property:**

```
//+------------------------------------------------------------------+
//| Compare CWndInd objects with each other by the specified property|
//+------------------------------------------------------------------+
int CWndInd::Compare(const CObject *node,const int mode=0) const
  {
   const CWndInd *obj_compared=node;
   if(mode==CHART_PROP_WINDOW_IND_HANDLE) return(this.Handle()>obj_compared.Handle() ? 1 : this.Handle()<obj_compared.Handle() ? -1 : 0);
   else if(mode==CHART_PROP_WINDOW_IND_INDEX) return(this.Index()>obj_compared.Index() ? 1 : this.Index()<obj_compared.Index() ? -1 : 0);
   return(this.Name()==obj_compared.Name() ? 0 : this.Name()<obj_compared.Name() ? -1 : 1);
  }
//+------------------------------------------------------------------+
```

We do not have custom enumerations of properties for this object. All its properties are present in the enumerations of chart object properties. Therefore, the method may receive any chart object properties. However, the comparison by the passed properties is performed only if **mode** receives the "indicator handle" property or "chart window index". In any other cases, the comparison is performed by the indicator short name.

The comparison method is standard for all library objects: if the parameter value of the current object exceeds the value of the compared one, 1 is returned. If it is less, then -1 is returned, otherwise 0.

**Let's start the development of the chart window object class.**

I will continue developing the code and write the chart window object class in the file containing the class of the indicator object in the chart window (ChartWnd.mqh). The class should be derived [from the basic object of all CBaseObj library objects](https://www.mql5.com/en/articles/7071#node01):

```
//+------------------------------------------------------------------+
//| Chart window object class                                        |
//+------------------------------------------------------------------+
class CChartWnd : public CBaseObj
  {
  }
```

The private section of the class is to contain the list of pointers to indicator objects in this window, an index of the subwindow described by the object and additional methods for arranging the class operation:

```
//+------------------------------------------------------------------+
//| Chart window object class                                        |
//+------------------------------------------------------------------+
class CChartWnd : public CBaseObj
  {
private:
   CArrayObj         m_list_ind;                                        // Indicator list
   int               m_window_num;                                      // Subwindow index
//--- Return the flag indicating the presence of an indicator from the list in the window
   bool              IsPresentInWindow(const CWndInd *ind);
//--- Remove indicators not present in the window from the list
   void              IndicatorsDelete(void);
//--- Add new indicators to the list
   void              IndicatorsAdd(void);
//--- Set a subwindow index
   void              SetWindowNum(const int num)                        { this.m_window_num=num;   }
public:
```

The public section of the class features standard methods (except for the methods setting the property values — since the object has no custom property enumeration lists, so we will only return the required values described by some chart object properties belonging to the chart window). These methods were considered in the previous articles multiple times.

Besides, the class features the methods for setting and returning window property values, as well as for working with the class. I will consider then in more detail below.

```
//+------------------------------------------------------------------+
//| Chart window object class                                        |
//+------------------------------------------------------------------+
class CChartWnd : public CBaseObj
  {
private:
   CArrayObj         m_list_ind;                                        // Indicator list
   int               m_window_num;                                      // Subwindow index
//--- Return the flag indicating the presence of an indicator from the list in the window
   bool              IsPresentInWindow(const CWndInd *ind);
//--- Remove indicators not present in the window from the list
   void              IndicatorsDelete(void);
//--- Add new indicators to the list
   void              IndicatorsAdd(void);
//--- Set a subwindow index
   void              SetWindowNum(const int num)                        { this.m_window_num=num;   }
public:
//--- Return itself
   CChartWnd        *GetObject(void)                                    { return &this;            }

//--- Return the flag of the object supporting this property
   virtual bool      SupportProperty(ENUM_CHART_PROP_INTEGER property)  { return(property==CHART_PROP_WINDOW_YDISTANCE || property==CHART_PROP_HEIGHT_IN_PIXELS ? true : false); }
   virtual bool      SupportProperty(ENUM_CHART_PROP_DOUBLE property)   { return false; }
   virtual bool      SupportProperty(ENUM_CHART_PROP_STRING property)   { return (property==CHART_PROP_INDICATOR_NAME ? true : false); }

//--- Get description of (1) integer, (2) real and (3) string properties
   string            GetPropertyDescription(ENUM_CHART_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_CHART_PROP_DOUBLE property)  { return CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED);  }
   string            GetPropertyDescription(ENUM_CHART_PROP_STRING property)  { return CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED);  }

//--- Display the description of object properties in the journal (full_prop=true - all properties, false - supported ones only)
   void              Print(const bool full_prop=false);
//--- Display a short description of the object in the journal
   virtual void      PrintShort(const bool dash=false);
//--- Return the object short name
   virtual string    Header(void);

//--- Compare CChartWnd objects by a specified property (to sort the list by an MQL5 signal object)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CChartWnd objects by all properties (to search for equal MQL5 signal objects)
   bool              IsEqual(CChartWnd* compared_obj) const;

//--- Constructors
                     CChartWnd(void);
                     CChartWnd(const long chart_id,const int wnd_num);

//--- Return the distance in pixels between the window borders
   int               YDistance(void)                              const { return (int)::ChartGetInteger(this.m_chart_id,CHART_WINDOW_YDISTANCE,this.m_window_num);}
//--- (1) Return and (2) set the window height in pixels
   int               HeightInPixels(void)                         const { return (int)::ChartGetInteger(this.m_chart_id,CHART_HEIGHT_IN_PIXELS,this.m_window_num);}
   bool              SetHeightInPixels(const int value,const bool redraw=false);
//--- Return (1) the subwindow index and (2) the number of indicators attached to the window
   int               WindowNum(void)                              const { return this.m_window_num;}
   int               IndicatorsTotal(void)                        const { return this.m_list_ind.Total();   }

//--- Return (1) the indicator list and (2) the window indicator object from the list by index
   CArrayObj        *GetIndicatorsList(void)                            { return &this.m_list_ind;          }
   CWndInd          *GetIndicator(const int index)                      { return this.m_list_ind.At(index); }

//--- Display the description of indicators attached to the chart window in the journal
   void              PrintIndicators(const bool dash=false);
//--- Display the description of the window parameters in the journal
   void              PrintParameters(const bool dash=false);

//--- Create the list of indicators attached to the window
   void              IndicatorsListCreate(void);
//--- Update data on attached indicators
   void              Refresh(void);

  };
//+------------------------------------------------------------------+
```

**The method returning the distance in pixels between the window borders:**

```
//--- Return the distance in pixels between the window borders
   int YDistance(void) const { return (int)::ChartGetInteger(this.m_chart_id,CHART_WINDOW_YDISTANCE,this.m_window_num);}
```

Since the [CHART\_WINDOW\_YDISTANCE](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property) chart property is read-only, there is no method setting the value here. The method simply returns the value of the property for this particular chart subwindow.

**The method returning the window height in pixels:**

```
int HeightInPixels(void) const { return (int)::ChartGetInteger(this.m_chart_id,CHART_HEIGHT_IN_PIXELS,this.m_window_num);}
```

The method works similarly to the above one and returns the property value for the index of the window specified in the **m\_window\_num** variable.

**The method setting the window height in pixels** (declared in the class body, while implemented outside the class body):

```
//+------------------------------------------------------------------+
//| Set the window height in pixels                                  |
//+------------------------------------------------------------------+
bool CChartWnd::SetHeightInPixels(const int value,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.m_chart_id,CHART_HEIGHT_IN_PIXELS,this.m_window_num,value))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   if(redraw)
      ::ChartRedraw(this.m_chart_id);
   return true;
  }
//+------------------------------------------------------------------+
```

The method is similar to the ones setting the values of chart object properties considered in the previous article. The method receives the value required for setting, then an attempt is made to assign it to the window using the [ChartSetInteger()](https://www.mql5.com/en/docs/chart_operations/chartsetinteger) function. If the chart change event is not queued, inform of that and return false. If the event is successfully queued, return true by preliminarily redrawing the chart with the **redraw** flag activated. The forced chart redrawing is necessary to avoid waiting for any chart event (quote arrival, size change, mouse click, etc.) for displaying changes, but instead to redraw the chart and see the result.

**The method comparing chart window objects by a specified property:**

```
//+------------------------------------------------------------------+
//| Compare CChartWnd objects with each other by a specified property|
//+------------------------------------------------------------------+
int CChartWnd::Compare(const CObject *node,const int mode=0) const
  {
   const CChartWnd *obj_compared=node;
   if(mode==CHART_PROP_WINDOW_YDISTANCE)
      return(this.YDistance()>obj_compared.YDistance() ? 1 : this.YDistance()<obj_compared.YDistance() ? -1 : 0);
   else if(mode==CHART_PROP_HEIGHT_IN_PIXELS)
      return(this.HeightInPixels()>obj_compared.HeightInPixels() ? 1 : this.HeightInPixels()<obj_compared.HeightInPixels() ? -1 : 0);
   return -1;
  }
//+------------------------------------------------------------------+
```

Just like in the method for comparing in the chart window indicator class considered above, we compare only some properties specified in the enumerations of the chart object properties:

- Distance in Y axis pixels between the upper frame of the indicator subwindow and the upper frame of the main chart window,
- Chart height in pixels,
- In all other cases, return -1


**The method** **comparing chart window objects by all properties:**

```
//+------------------------------------------------------------------+
//| Compare the CChartWnd objects by all properties                  |
//+------------------------------------------------------------------+
bool CChartWnd::IsEqual(CChartWnd *compared_obj) const
  {
   return(this.YDistance()!=compared_obj.YDistance() || this.HeightInPixels()!=compared_obj.HeightInPixels() ? false : true);
  }
//+------------------------------------------------------------------+
```

Here if at least one of the two properties of compared objects returns inequality, return false  — the objects are not equal. Otherwise, return true  — the objects are identical.

**The method returning the description of the integer property:**

```
//+------------------------------------------------------------------+
//| Return description of object's integer property                  |
//+------------------------------------------------------------------+
string CChartWnd::GetPropertyDescription(ENUM_CHART_PROP_INTEGER property)
  {
   return
     (
      property==CHART_PROP_WINDOW_YDISTANCE  ?  CMessage::Text(MSG_CHART_OBJ_WINDOW_YDISTANCE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.YDistance()
         )  :
      property==CHART_PROP_HEIGHT_IN_PIXELS  ?  CMessage::Text(MSG_CHART_OBJ_HEIGHT_IN_PIXELS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.HeightInPixels()
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

Depending on which chart window object integer property is passed to the method, the string with its description is created and returned.

**The method displaying object properties in the journal:**

```
//+------------------------------------------------------------------+
//| Display object properties in the journal                         |
//+------------------------------------------------------------------+
void CChartWnd::Print(const bool full_prop=false)
  {
   ::Print("============= ",CMessage::Text(MSG_LIB_PARAMS_LIST_BEG)," (",this.Header(),") =============");
   int beg=0, end=CHART_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_CHART_PROP_INTEGER prop=(ENUM_CHART_PROP_INTEGER)i;
      if(prop!=CHART_PROP_WINDOW_YDISTANCE && prop!=CHART_PROP_HEIGHT_IN_PIXELS) continue;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=CHART_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      //ENUM_CHART_PROP_DOUBLE prop=(ENUM_CHART_PROP_DOUBLE)i;
      //if(!full_prop && !this.SupportProperty(prop)) continue;
      //::Print(this.GetPropertyDescription(prop));
     }
   beg=end; end+=CHART_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_CHART_PROP_STRING prop=(ENUM_CHART_PROP_STRING)i;
      if(prop==CHART_PROP_INDICATOR_NAME)
        {
         this.PrintIndicators();
         continue;
        }
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("============= ",CMessage::Text(MSG_LIB_PARAMS_LIST_END)," (",this.Header(),") =============\n");
  }
//+------------------------------------------------------------------+
```

In three loops by all chart object properties, receive the next property and display its description in the journal. Since the chart window object does not have custom enumeration lists of its properties, we need to display only the chart object properties inherent in the chart window object — integer and string ones. The object has no real properties meaning they cannot be displayed. I could have set a rigid framework for the beginning and end of each loop... But such a solution is not the best one in terms of subsequent modifications of the chart object properties as I would have had to get back to the object and edit the values of each loop start and end. Therefore, I will simply make the real property loop empty (may be this is temporary till real properties are needed for the chart window object in the future). Thus, at any changes of the number of properties, the start and end of each loop will always be correct.

**The method returning a short description of the chart window object:**

```
//+------------------------------------------------------------------+
//| Return the object short name                                     |
//+------------------------------------------------------------------+
string CChartWnd::Header(void)
  {
   return(this.m_window_num==0 ? CMessage::Text(MSG_CHART_OBJ_CHART_WINDOW) : (string)this.WindowNum()+" "+CMessage::Text(MSG_CHART_OBJ_CHART_SUBWINDOW));
  }
//+------------------------------------------------------------------+
```

The method checks the subwindow index. If this is the main chart window (index 0), the "Main chart window" string is returned. If this is the main chart subwindow, the subwindow index + "Chart subwindow" string are returned.

**The method displaying the short chart window object description in the journal:**

```
//+------------------------------------------------------------------+
//| Display a short description of the object in the journal         |
//+------------------------------------------------------------------+
void CChartWnd::PrintShort(const bool dash=false)
  {
   ::Print((dash ? "- " : ""),this.Header()," ID: ",(string)this.GetChartID(),", ",CMessage::Text(MSG_CHART_OBJ_INDICATORS_TOTAL),": ",this.IndicatorsTotal());
  }
//+------------------------------------------------------------------+
```

Here we create a string consisting of a short object name, chart ID and the number of indicators attached to the window. If the method receives the flag indicating the necessity of displaying the hyphen before an object description ( **dash**), the hyphen is displayed before the created string.

**The method displaying the descriptions of all indicators attached to the window in the journal:**

```
//+------------------------------------------------------------------+
//| Display the description of indicators attached to the window     |
//+------------------------------------------------------------------+
void CChartWnd::PrintIndicators(const bool dash=false)
  {
   string header=
     (
      this.WindowNum()==0 ? CMessage::Text(MSG_CHART_OBJ_INDICATORS_MW_NAME_LIST) :
      CMessage::Text(MSG_CHART_OBJ_INDICATORS_SW_NAME_LIST)+" "+(string)this.WindowNum()
     );
   ::Print(header,":");
   int total=this.IndicatorsTotal();
   if(total==0)
      ::Print("- ",CMessage::Text(MSG_CHART_OBJ_INDICATORS_NONE));
   else for(int i=0;i<total;i++)
     {
      CWndInd *ind=this.m_list_ind.At(i);
      if(ind==NULL)
         continue;
      ind.Print(dash);
     }
  }
//+------------------------------------------------------------------+
```

First, create a header matching the window and display it in the journal.

If this is the main chart window, the header text should be "Indicators in the main chart window", otherwise the header text will be "Indicators in the chart window" + the window index.

Next, define the number of indicators attached to the window. If there are no such indicators, display "No indicators".

Otherwise, in the loop by the list of all indicators, get the next indicator object in the chart window and print its data.

**The method displaying the description of the window parameters in the journal:**

```
//+------------------------------------------------------------------+
//| Display the description of the window parameters in the journal  |
//+------------------------------------------------------------------+
void CChartWnd::PrintParameters(const bool dash=false)
  {
   string header=
     (
      this.WindowNum()==0 ? CMessage::Text(MSG_CHART_OBJ_CHART_WINDOW) :
      CMessage::Text(MSG_CHART_OBJ_CHART_SUBWINDOW)+" "+(string)this.WindowNum()
     );
   ::Print((dash ? " " : ""),header,":");
   if(this.WindowNum()>0)
      ::Print((dash ? " - " : ""),GetPropertyDescription(CHART_PROP_WINDOW_YDISTANCE));
   ::Print((dash ? " - " : ""),GetPropertyDescription(CHART_PROP_HEIGHT_IN_PIXELS));
  }
//+------------------------------------------------------------------+
```

First, create a header matching the window and display it in the journal.

If this is the main chart window, the header text should be "Main chart window", otherwise, the header text will be "Chart subwindow" + the window index.

If this is a chart subwindow (its index exceeds zero), display the vertical (Y axis) distance in pixels between the upper border of the indicator subwindow and the upper border of the main chart window (for the main window it is always 0, and it is not displayed). Next, display the second object property (chart height in pixels) in the journal.

**The method creating the list of indicators attached to the window:**

```
//+------------------------------------------------------------------+
//| Create the list of indicators attached to the window             |
//+------------------------------------------------------------------+
void CChartWnd::IndicatorsListCreate(void)
  {
   //--- Clear the list of indicators
   this.m_list_ind.Clear();
   //--- Get the total number of indicators in the window
   int total=::ChartIndicatorsTotal(this.m_chart_id,this.m_window_num);
   //--- In the loop by the number of indicators,
   for(int i=0;i<total;i++)
     {
      //--- obtain and save the short indicator name,
      string name=::ChartIndicatorName(this.m_chart_id,this.m_window_num,i);
      //--- get and save the indicator handle by its short name
      int handle=::ChartIndicatorGet(this.m_chart_id,this.m_window_num,name);
      //--- Free the indicator handle
      ::IndicatorRelease(handle);
      //--- Create the new indicator object in the chart window
      CWndInd *ind=new CWndInd(handle,name,i);
      if(ind==NULL)
         continue;
      //--- set the sorted list flag to the list
      this.m_list_ind.Sort();
      //--- If failed to add the object to the list, remove it
      if(!this.m_list_ind.Add(ind))
         delete ind;
     }
  }
//+------------------------------------------------------------------+
```

The method is commented in detail in the code. When obtaining the list of indicators in the window, we obtain the indicator handle by its short name using [ChartIndicatorGet()](https://www.mql5.com/en/docs/chart_operations/chartindicatorget), which imposes some "responsibilities" on us. The terminal keeps track of the use of each indicator. With each handle receipt, the internal counter of using the indicator is increased. If we do not take care of releasing the indicator handle that is already unnecessary to us in our program, then it will be impossible to catch the "lost" handle later. Therefore, I release the handle immediately after obtaining all the necessary indicator data, thereby decreasing the internal counter of the indicator usage.

**The method adding new indicators to the list:**

```
//+------------------------------------------------------------------+
//| Add new indicators to the list                                   |
//+------------------------------------------------------------------+
void CChartWnd::IndicatorsAdd(void)
  {
   //--- Get the total number of indicators in the window
   int total=::ChartIndicatorsTotal(this.m_chart_id,this.m_window_num);
   //--- In the loop by the number of indicators,
   for(int i=0;i<total;i++)
     {
      //--- obtain and save the short indicator name,
      string name=::ChartIndicatorName(this.m_chart_id,this.m_window_num,i);
      //--- get and save the indicator handle by its short name
      int handle=::ChartIndicatorGet(this.m_chart_id,this.m_window_num,name);
      //--- Release the indicator handle
      ::IndicatorRelease(handle);
      //--- Create the new indicator object in the chart window
      CWndInd *ind=new CWndInd(handle,name,i);
      if(ind==NULL)
         continue;
      //--- set the sorted list flag to the list
      this.m_list_ind.Sort();
      //--- If the object is already in the list or an attempt to add it to the list failed, remove it
      if(this.m_list_ind.Search(ind)>WRONG_VALUE || !this.m_list_ind.Add(ind))
         delete ind;
     }
  }
//+------------------------------------------------------------------+
```

The method logic is identical to the one considered above. It is also commented in the code. The only difference is that the list is initially not cleared in the method. When an indicator is added to the list, the presence of such an indicator in the list is first checked. If it is already present, the indicator object is removed.

To be able to synchronize our list of indicators in the window with their real number (after all, indicators can be added to the window and removed from the chart window), we need to compare their number in the window with the number in the list. We will do this in subsequent articles.

**However, I will implement the method returning the presence of an indicator from the list in the terminal window** here:

```
//+--------------------------------------------------------------------------------------+
//| Return the flag indicating the presence of an indicator from the list in the window  |
//+--------------------------------------------------------------------------------------+
bool CChartWnd::IsPresentInWindow(const CWndInd *ind)
  {
   int total=::ChartIndicatorsTotal(this.m_chart_id,this.m_window_num);
   for(int i=0;i<total;i++)
     {
      string name=::ChartIndicatorName(this.m_chart_id,this.m_window_num,i);
      int handle=::ChartIndicatorGet(this.m_chart_id,this.m_window_num,name);
      ::IndicatorRelease(handle);
      if(ind.Name()==name && ind.Handle()==handle)
         return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

The method receives the pointer to the indicator object in the chart window whose actual presence should be checked. Next, in the loop by the total number of indicators in the chart window, get the name of the next indicator, get its handle and release it right away. If the current indicator's short name and handle match the ones of the checked object, return true. The indicator is still present in the chart window. Upon the loop completion, return false — no matches are found, which means there is no such indicator in the chart window.

If the list features indicators that are not present on the chart, they should be removed from the list.

**The method removing indicators not present in the window from the list:**

```
//+------------------------------------------------------------------+
//| Remove indicators not present in the window from the list        |
//+------------------------------------------------------------------+
void CChartWnd::IndicatorsDelete(void)
  {
   int total=this.m_list_ind.Total();
   for(int i=total-1;i>WRONG_VALUE;i--)
     {
      CWndInd *ind=this.m_list_ind.At(i);
      if(!this.IsPresentInWindow(ind))
         this.m_list_ind.Delete(i);
     }
  }
//+------------------------------------------------------------------+
```

Here in the loop by the list of indicator objects,obtain the next indicator object by the loop index and check its presence in the real chart window. If it is absent, remove the pointer to it from the list using the [Delete()](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjdelete) method.

Later on, while implementing the chart collection class, we will need to track the indicator status in subwindows of all chart objects. If there is a mismatch between a real number of indicators in chart subwindows with their descriptions in the lists of chart window objects, we will need to remove unnecessary indicators from the list and add new ones if available.

To achieve this, let's create **the method updating data on indicators attached to chart windows:**

```
//+------------------------------------------------------------------+
//| Update data on attached indicators                               |
//+------------------------------------------------------------------+
void CChartWnd::Refresh(void)
  {
   this.IndicatorsDelete();
   this.IndicatorsAdd();
  }
//+------------------------------------------------------------------+
```

Here we first detect indicators not present in a real chart window and remove them from the list. Next, we search and add new indicators that are present in the window but not in the list using the methods mentioned above.

Finally, let's consider **the parametric class constructor:**

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CChartWnd::CChartWnd(const long chart_id,const int wnd_num) : m_window_num(wnd_num)
  {
   CBaseObj::SetChartID(chart_id);
   this.IndicatorsListCreate();
  }
//+------------------------------------------------------------------+
```

Here we first set the chart ID in the CBaseObj parent object. Then we create the list of indicators attached to the chart window. The chart subwindow index is set in the constructor initialization list.

This concludes creation of the indicator object classes in the chart window and the chart window object.

The full listing of both classes located in one file can be viewed in the files attached below.

Since we now have the object describing the chart window and its subwindows, it is time to improve the **CChartObj** chart object class in \\MQL5\\Include\\DoEasy\\Objects\\Chart\ **ChartObj.mqh**. Now it will have the list of all subwindows located in its main window. To obtain window properties, we need to refer to the pointer to the necessary chart window object created above. From the obtained window object, we are able to get the list of all indicators attached to it. In turn, from that indicators, we can receive the handle of a necessary indicator to work with it.

First of all, let's include the file of window objects and indicators in the window to the chart object file and declare the list object, in which we will store the pointers to all chart object windows:

```
//+------------------------------------------------------------------+
//|                                                     ChartObj.mqh |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\..\Objects\BaseObj.mqh"
#include "ChartWnd.mqh"
//+------------------------------------------------------------------+
//| Chart object class                                               |
//+------------------------------------------------------------------+
class CChartObj : public CBaseObj
  {
private:
   CArrayObj         m_list_wnd;                                  // List of chart window objects
   long              m_long_prop[CHART_PROP_INTEGER_TOTAL];       // Integer properties
   double            m_double_prop[CHART_PROP_DOUBLE_TOTAL];      // Real properties
   string            m_string_prop[CHART_PROP_STRING_TOTAL];      // String properties
   int               m_digits;                                    // Symbol's Digits()
```

In the list of the methods for setting property values of the class private section, add the method setting the chart window visibility:

```
//--- The methods of setting property values
   bool              SetMode(const string source,const ENUM_CHART_MODE mode,const bool redraw=false);
   bool              SetScale(const string source,const int scale,const bool redraw=false);
   bool              SetModeVolume(const string source,const ENUM_CHART_VOLUME_MODE mode,const bool redraw=false);
   void              SetVisibleBars(void);
   void              SetWindowsTotal(void);
   void              SetVisible(void);
   void              SetFirstVisibleBars(void);
   void              SetWidthInBars(void);
   void              SetWidthInPixels(void);
   void              SetMaximizedFlag(void);
   void              SetMinimizedFlag(void);
   void              SetExpertName(void);
   void              SetScriptName(void);

public:
```

Since the "Distance in Y axis pixels between the upper frame of the indicator subwindow and the upper frame of the chart main window" property is not applicable to the main window (it is always 0 there), let's change the value return in case this is the CHART\_PROP\_WINDOW\_YDISTANCE property in the method returning the flag of an object supporting an integer property:

```
//--- Return the flag of the object supporting this property
   virtual bool      SupportProperty(ENUM_CHART_PROP_INTEGER property)           { return (property!=CHART_PROP_WINDOW_YDISTANCE ? true : false); }
   virtual bool      SupportProperty(ENUM_CHART_PROP_DOUBLE property)            { return true; }
   virtual bool      SupportProperty(ENUM_CHART_PROP_STRING property)            { return true; }
```

If the property passed to the method is not equal to CHART\_PROP\_WINDOW\_YDISTANCE, return true, otherwise – false.

In the list of methods for a simplified access to the object properties, add the method returning the window visibility:

```
//--- Return the total number of chart windows including indicator subwindows
   int               WindowsTotal(void)                              const { return (int)this.GetProperty(CHART_PROP_WINDOWS_TOTAL);      }

//--- Return the window visibility
   bool              Visible(void)                                   const { return (bool)this.GetProperty(CHART_PROP_WINDOW_IS_VISIBLE); }

//--- Return the chart window handle
   int               Handle(void)                                    const { return (int)this.GetProperty(CHART_PROP_WINDOW_HANDLE);      }
```

Here the method is to return the property of only the main chart window.

The methods returning the distance in Y axis pixels between the upper frame of the indicator subwindow and the upper frame of the chart main window, as well as the ones returning and setting the height of a specified chart in pixels have been changed. Now we need to find the necessary window object for setting and returning these properties to set or get these values in the object properties. The implementation of the methods will be considered a bit later.

```
//--- Return the name of a script launched on the chart
   string            ScriptName(void)                                const { return this.GetProperty(CHART_PROP_SCRIPT_NAME);             }

//--- Return the distance in Y axis pixels between the upper frame of the indicator subwindow and the upper frame of the chart main window
   int               WindowYDistance(const int sub_window)           const;

//--- (1) Return and (2) set the height of the specified chart in pixels
   int               WindowHeightInPixels(const int sub_window)      const;
   bool              SetWindowHeightInPixels(const int height,const int sub_window,const bool redraw=false);

//--- Return the specified subwindow visibility
```

At the very bottom of the class body, declare additional methods for receiving the necessary window object and displaying the properties of all chart subwindows and data of all indicators attached to the specified chart window in the journal:

```
//--- Emulate a tick (chart updates - similar to the terminal Refresh command)
   void              EmulateTick(void)                                     { ::ChartSetSymbolPeriod(this.ID(),this.Symbol(),this.Timeframe());}

//--- Return the chart window specified by index
   CChartWnd        *GetWindowByIndex(const int index)               const { return this.m_list_wnd.At(index); }
//--- Return the window object by its subwindow index
   CChartWnd        *GetWindowByNum(const int win_num)               const;

//--- Display data of all indicators of all chart windows in the journal
   void              PrintWndIndicators(void);
//--- Display the properties of all chart windows in the journal
   void              PrintWndParameters(void);

  };
//+------------------------------------------------------------------+
```

**In the parametric class constructor**, set the chart ID in the class parent object and specify three main window properties for the chart object (this is in fact unnecessary since the main window object is now located in the list of all chart windows but since the object features such properties, I will fill them with the correct data). At the very end of the method, add all window objects belonging to the chart to the list of all chart windows:

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CChartObj::CChartObj(const long chart_id)
  {
//--- Set chart ID to the base object
   CBaseObj::SetChartID(chart_id);
//--- Set integer properties
   this.SetProperty(CHART_PROP_ID,chart_id);                                                             // Chart ID
   this.SetProperty(CHART_PROP_TIMEFRAME,::ChartPeriod(this.ID()));                                      // Chart timeframe
   this.SetProperty(CHART_PROP_SHOW,::ChartGetInteger(this.ID(),CHART_SHOW));                            // Price chart drawing attribute
   this.SetProperty(CHART_PROP_IS_OBJECT,::ChartGetInteger(this.ID(),CHART_IS_OBJECT));                  // Chart object identification attribute
   this.SetProperty(CHART_PROP_BRING_TO_TOP,false);                                                      // Show chart above all others
   this.SetProperty(CHART_PROP_CONTEXT_MENU,::ChartGetInteger(this.ID(),CHART_CONTEXT_MENU));            // Access to the context menu using the right click
   this.SetProperty(CHART_PROP_CROSSHAIR_TOOL,::ChartGetInteger(this.ID(),CHART_CROSSHAIR_TOOL));        // Access the Crosshair tool by pressing the middle mouse button
   this.SetProperty(CHART_PROP_MOUSE_SCROLL,::ChartGetInteger(this.ID(),CHART_MOUSE_SCROLL));            // Scroll the chart horizontally using the left mouse button
   this.SetProperty(CHART_PROP_EVENT_MOUSE_WHEEL,::ChartGetInteger(this.ID(),CHART_EVENT_MOUSE_WHEEL));  // Send messages about mouse wheel events to all MQL5 programs on a chart
   this.SetProperty(CHART_PROP_EVENT_MOUSE_MOVE,::ChartGetInteger(this.ID(),CHART_EVENT_MOUSE_MOVE));    // Send messages about mouse button click and movement events to all MQL5 programs on a chart
   this.SetProperty(CHART_PROP_EVENT_OBJECT_CREATE,::ChartGetInteger(this.ID(),CHART_EVENT_OBJECT_CREATE)); // Send messages about the graphical object creation event to all MQL5 programs on a chart
   this.SetProperty(CHART_PROP_EVENT_OBJECT_DELETE,::ChartGetInteger(this.ID(),CHART_EVENT_OBJECT_DELETE)); // Send messages about the graphical object destruction event to all MQL5 programs on a chart
   this.SetProperty(CHART_PROP_MODE,::ChartGetInteger(this.ID(),CHART_MODE));                            // Type of the chart (candlesticks, bars or line)
   this.SetProperty(CHART_PROP_FOREGROUND,::ChartGetInteger(this.ID(),CHART_FOREGROUND));                // Price chart in the foreground
   this.SetProperty(CHART_PROP_SHIFT,::ChartGetInteger(this.ID(),CHART_SHIFT));                          // Mode of shift of the price chart from the right border
   this.SetProperty(CHART_PROP_AUTOSCROLL,::ChartGetInteger(this.ID(),CHART_AUTOSCROLL));                // The mode of automatic shift to the right border of the chart
   this.SetProperty(CHART_PROP_KEYBOARD_CONTROL,::ChartGetInteger(this.ID(),CHART_KEYBOARD_CONTROL));    // Allow managing the chart using a keyboard
   this.SetProperty(CHART_PROP_QUICK_NAVIGATION,::ChartGetInteger(this.ID(),CHART_QUICK_NAVIGATION));    // Allow the chart to intercept Space and Enter key strokes to activate the quick navigation bar
   this.SetProperty(CHART_PROP_SCALE,::ChartGetInteger(this.ID(),CHART_SCALE));                          // Scale
   this.SetProperty(CHART_PROP_SCALEFIX,::ChartGetInteger(this.ID(),CHART_SCALEFIX));                    // Fixed scale mode
   this.SetProperty(CHART_PROP_SCALEFIX_11,::ChartGetInteger(this.ID(),CHART_SCALEFIX_11));              // 1:1 scale mode
   this.SetProperty(CHART_PROP_SCALE_PT_PER_BAR,::ChartGetInteger(this.ID(),CHART_SCALE_PT_PER_BAR));    // Mode for specifying the scale in points per bar
   this.SetProperty(CHART_PROP_SHOW_TICKER,::ChartGetInteger(this.ID(),CHART_SHOW_TICKER));              // Display a symbol ticker in the upper left corner
   this.SetProperty(CHART_PROP_SHOW_OHLC,::ChartGetInteger(this.ID(),CHART_SHOW_OHLC));                  // Display OHLC values in the upper left corner
   this.SetProperty(CHART_PROP_SHOW_BID_LINE,::ChartGetInteger(this.ID(),CHART_SHOW_BID_LINE));          // Display Bid value as a horizontal line on the chart
   this.SetProperty(CHART_PROP_SHOW_ASK_LINE,::ChartGetInteger(this.ID(),CHART_SHOW_ASK_LINE));          // Display Ask value as a horizontal line on the chart
   this.SetProperty(CHART_PROP_SHOW_LAST_LINE,::ChartGetInteger(this.ID(),CHART_SHOW_LAST_LINE));        // Display Last value as a horizontal line on the chart
   this.SetProperty(CHART_PROP_SHOW_PERIOD_SEP,::ChartGetInteger(this.ID(),CHART_SHOW_PERIOD_SEP));      // Display vertical separators between adjacent periods
   this.SetProperty(CHART_PROP_SHOW_GRID,::ChartGetInteger(this.ID(),CHART_SHOW_GRID));                  // Display the chart grid
   this.SetProperty(CHART_PROP_SHOW_VOLUMES,::ChartGetInteger(this.ID(),CHART_SHOW_VOLUMES));            // Display volumes on the chart
   this.SetProperty(CHART_PROP_SHOW_OBJECT_DESCR,::ChartGetInteger(this.ID(),CHART_SHOW_OBJECT_DESCR));  // Display text descriptions of the objects
   this.SetProperty(CHART_PROP_VISIBLE_BARS,::ChartGetInteger(this.ID(),CHART_VISIBLE_BARS));            // Number of bars on a chart that are available for display
   this.SetProperty(CHART_PROP_WINDOWS_TOTAL,::ChartGetInteger(this.ID(),CHART_WINDOWS_TOTAL));          // The total number of chart windows including indicator subwindows
   this.SetProperty(CHART_PROP_WINDOW_IS_VISIBLE,::ChartGetInteger(this.ID(),CHART_WINDOW_IS_VISIBLE,0));// Window visibility
   this.SetProperty(CHART_PROP_WINDOW_HANDLE,::ChartGetInteger(this.ID(),CHART_WINDOW_HANDLE));          // Chart window handle
   this.SetProperty(CHART_PROP_WINDOW_YDISTANCE,::ChartGetInteger(this.ID(),CHART_WINDOW_YDISTANCE,0));  // Distance in Y axis pixels between the upper frame of the indicator subwindow and the upper frame of the chart main window
   this.SetProperty(CHART_PROP_FIRST_VISIBLE_BAR,::ChartGetInteger(this.ID(),CHART_FIRST_VISIBLE_BAR));  // Number of the first visible bar on the chart
   this.SetProperty(CHART_PROP_WIDTH_IN_BARS,::ChartGetInteger(this.ID(),CHART_WIDTH_IN_BARS));          // Chart width in bars
   this.SetProperty(CHART_PROP_WIDTH_IN_PIXELS,::ChartGetInteger(this.ID(),CHART_WIDTH_IN_PIXELS));      // Chart width in pixels
   this.SetProperty(CHART_PROP_HEIGHT_IN_PIXELS,::ChartGetInteger(this.ID(),CHART_HEIGHT_IN_PIXELS,0));  // Chart height in pixels
   this.SetProperty(CHART_PROP_COLOR_BACKGROUND,::ChartGetInteger(this.ID(),CHART_COLOR_BACKGROUND));    // Chart background color
   this.SetProperty(CHART_PROP_COLOR_FOREGROUND,::ChartGetInteger(this.ID(),CHART_COLOR_FOREGROUND));    // Color of axes, scale and OHLC line
   this.SetProperty(CHART_PROP_COLOR_GRID,::ChartGetInteger(this.ID(),CHART_COLOR_GRID));                // Grid color
   this.SetProperty(CHART_PROP_COLOR_VOLUME,::ChartGetInteger(this.ID(),CHART_COLOR_VOLUME));            // Color of volumes and position opening levels
   this.SetProperty(CHART_PROP_COLOR_CHART_UP,::ChartGetInteger(this.ID(),CHART_COLOR_CHART_UP));        // Color for the up bar, shadows and body borders of bullish candlesticks
   this.SetProperty(CHART_PROP_COLOR_CHART_DOWN,::ChartGetInteger(this.ID(),CHART_COLOR_CHART_DOWN));    // Color for the down bar, shadows and body borders of bearish candlesticks
   this.SetProperty(CHART_PROP_COLOR_CHART_LINE,::ChartGetInteger(this.ID(),CHART_COLOR_CHART_LINE));    // Color of the chart line and the Doji candlesticks
   this.SetProperty(CHART_PROP_COLOR_CANDLE_BULL,::ChartGetInteger(this.ID(),CHART_COLOR_CANDLE_BULL));  // Color of the bullish candle body
   this.SetProperty(CHART_PROP_COLOR_CANDLE_BEAR,::ChartGetInteger(this.ID(),CHART_COLOR_CANDLE_BEAR));  // Color of the bearish candle body
   this.SetProperty(CHART_PROP_COLOR_BID,::ChartGetInteger(this.ID(),CHART_COLOR_BID));                  // Bid price line color
   this.SetProperty(CHART_PROP_COLOR_ASK,::ChartGetInteger(this.ID(),CHART_COLOR_ASK));                  // Ask price line color
   this.SetProperty(CHART_PROP_COLOR_LAST,::ChartGetInteger(this.ID(),CHART_COLOR_LAST));                // Color of the last performed deal's price line (Last)
   this.SetProperty(CHART_PROP_COLOR_STOP_LEVEL,::ChartGetInteger(this.ID(),CHART_COLOR_STOP_LEVEL));    // Color of stop order levels (Stop Loss and Take Profit)
   this.SetProperty(CHART_PROP_SHOW_TRADE_LEVELS,::ChartGetInteger(this.ID(),CHART_SHOW_TRADE_LEVELS));  // Display trade levels on the chart (levels of open positions, Stop Loss, Take Profit and pending orders)
   this.SetProperty(CHART_PROP_DRAG_TRADE_LEVELS,::ChartGetInteger(this.ID(),CHART_DRAG_TRADE_LEVELS));  // Enable the ability to drag trading levels on a chart using mouse
   this.SetProperty(CHART_PROP_SHOW_DATE_SCALE,::ChartGetInteger(this.ID(),CHART_SHOW_DATE_SCALE));      // Display the time scale on the chart
   this.SetProperty(CHART_PROP_SHOW_PRICE_SCALE,::ChartGetInteger(this.ID(),CHART_SHOW_PRICE_SCALE));    // Display the price scale on the chart
   this.SetProperty(CHART_PROP_SHOW_ONE_CLICK,::ChartGetInteger(this.ID(),CHART_SHOW_ONE_CLICK));        // Display the quick trading panel on the chart
   this.SetProperty(CHART_PROP_IS_MAXIMIZED,::ChartGetInteger(this.ID(),CHART_IS_MAXIMIZED));            // Chart window maximized
   this.SetProperty(CHART_PROP_IS_MINIMIZED,::ChartGetInteger(this.ID(),CHART_IS_MINIMIZED));            // Chart window minimized
   this.SetProperty(CHART_PROP_IS_DOCKED,::ChartGetInteger(this.ID(),CHART_IS_DOCKED));                  // Chart window docked
   this.SetProperty(CHART_PROP_FLOAT_LEFT,::ChartGetInteger(this.ID(),CHART_FLOAT_LEFT));                // Left coordinate of the undocked chart window relative to the virtual screen
   this.SetProperty(CHART_PROP_FLOAT_TOP,::ChartGetInteger(this.ID(),CHART_FLOAT_TOP));                  // Upper coordinate of the undocked chart window relative to the virtual screen
   this.SetProperty(CHART_PROP_FLOAT_RIGHT,::ChartGetInteger(this.ID(),CHART_FLOAT_RIGHT));              // Right coordinate of the undocked chart window relative to the virtual screen
   this.SetProperty(CHART_PROP_FLOAT_BOTTOM,::ChartGetInteger(this.ID(),CHART_FLOAT_BOTTOM));            // Bottom coordinate of the undocked chart window relative to the virtual screen
//--- Set real properties
   this.SetProperty(CHART_PROP_SHIFT_SIZE,::ChartGetDouble(this.ID(),CHART_SHIFT_SIZE));                 // Shift size of the zero bar from the right border in %
   this.SetProperty(CHART_PROP_FIXED_POSITION,::ChartGetDouble(this.ID(),CHART_FIXED_POSITION));         // Chart fixed position from the left border in %
   this.SetProperty(CHART_PROP_FIXED_MAX,::ChartGetDouble(this.ID(),CHART_FIXED_MAX));                   // Fixed chart maximum
   this.SetProperty(CHART_PROP_FIXED_MIN,::ChartGetDouble(this.ID(),CHART_FIXED_MIN));                   // Fixed chart minimum
   this.SetProperty(CHART_PROP_POINTS_PER_BAR,::ChartGetDouble(this.ID(),CHART_POINTS_PER_BAR));         // Scale in points per bar
   this.SetProperty(CHART_PROP_PRICE_MIN,::ChartGetDouble(this.ID(),CHART_PRICE_MIN));                   // Chart minimum
   this.SetProperty(CHART_PROP_PRICE_MAX,::ChartGetDouble(this.ID(),CHART_PRICE_MAX));                   // Chart maximum
//--- Set string properties
   this.SetProperty(CHART_PROP_COMMENT,::ChartGetString(this.ID(),CHART_COMMENT));                       // Comment text on the chart
   this.SetProperty(CHART_PROP_EXPERT_NAME,::ChartGetString(this.ID(),CHART_EXPERT_NAME));               // name of an EA launched on the chart
   this.SetProperty(CHART_PROP_SCRIPT_NAME,::ChartGetString(this.ID(),CHART_SCRIPT_NAME));               // name of a script launched on the chart
   this.SetProperty(CHART_PROP_SYMBOL,::ChartSymbol(this.ID()));                                         // Chart symbol

   this.m_digits=(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS);
   int total=this.WindowsTotal();
   for(int i=0;i<total;i++)
     {
      CChartWnd *wnd=new CChartWnd(m_chart_id,i);
      if(wnd==NULL)
         continue;
      m_list_wnd.Sort();
      if(!m_list_wnd.Add(wnd))
         delete wnd;
     }
  }
//+------------------------------------------------------------------+
```

To create the chart window lists, get the total number of all chart windows. In the loop by all windows, create a new chart window object and add it to the list.

A logical mistake was made in the method of comparing two chart objects developed [in the previous article](https://www.mql5.com/en/articles/9213): each chart has a unique chart ID and window handle. Therefore, comparing by these two properties always returns false. However, the objective of the method is to make sure the charts match, while comparing their IDs and handles will always show that they are not identical.

Let's fix this by skipping these two properties while comparing:

```
//+------------------------------------------------------------------+
//| Compare the CChartObj objects by all properties                  |
//+------------------------------------------------------------------+
bool CChartObj::IsEqual(CChartObj *compared_obj) const
  {
   int beg=0, end=CHART_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_CHART_PROP_INTEGER prop=(ENUM_CHART_PROP_INTEGER)i;
      if(prop==CHART_PROP_ID || prop==CHART_PROP_WINDOW_HANDLE) continue;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=CHART_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_CHART_PROP_DOUBLE prop=(ENUM_CHART_PROP_DOUBLE)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=CHART_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_CHART_PROP_STRING prop=(ENUM_CHART_PROP_STRING)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

**In the method returning the description of an integer object property, add returning the description of three added properties:**

```
//+------------------------------------------------------------------+
//| Return description of object's integer property                  |
//+------------------------------------------------------------------+
string CChartObj::GetPropertyDescription(ENUM_CHART_PROP_INTEGER property)
  {
   return
     (
      property==CHART_PROP_ID     ?  CMessage::Text(MSG_CHART_OBJ_ID)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_TIMEFRAME      ?  CMessage::Text(MSG_LIB_TEXT_BAR_PERIOD)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+TimeframeDescription((ENUM_TIMEFRAMES)this.GetProperty(property))
         )  :
      property==CHART_PROP_SHOW   ?  CMessage::Text(MSG_CHART_OBJ_SHOW)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_IS_OBJECT      ?  CMessage::Text(MSG_CHART_OBJ_IS_OBJECT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_BRING_TO_TOP   ?  CMessage::Text(MSG_CHART_OBJ_BRING_TO_TOP)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_CONTEXT_MENU   ?  CMessage::Text(MSG_CHART_OBJ_CONTEXT_MENU)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_CROSSHAIR_TOOL ?  CMessage::Text(MSG_CHART_OBJ_CROSSHAIR_TOOL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_MOUSE_SCROLL   ?  CMessage::Text(MSG_CHART_OBJ_MOUSE_SCROLL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_EVENT_MOUSE_WHEEL ?  CMessage::Text(MSG_CHART_OBJ_EVENT_MOUSE_WHEEL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_EVENT_MOUSE_MOVE  ?  CMessage::Text(MSG_CHART_OBJ_EVENT_MOUSE_MOVE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_EVENT_OBJECT_CREATE  ?  CMessage::Text(MSG_CHART_OBJ_EVENT_OBJECT_CREATE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_EVENT_OBJECT_DELETE  ?  CMessage::Text(MSG_CHART_OBJ_EVENT_OBJECT_DELETE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_MODE           ?  CMessage::Text(MSG_CHART_OBJ_MODE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+ChartModeDescription((ENUM_CHART_MODE)this.GetProperty(property))
         )  :
      property==CHART_PROP_FOREGROUND     ?  CMessage::Text(MSG_CHART_OBJ_FOREGROUND)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHIFT        ?  CMessage::Text(MSG_CHART_OBJ_SHIFT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_AUTOSCROLL        ?  CMessage::Text(MSG_CHART_OBJ_AUTOSCROLL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_KEYBOARD_CONTROL        ?  CMessage::Text(MSG_CHART_OBJ_KEYBOARD_CONTROL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_QUICK_NAVIGATION        ?  CMessage::Text(MSG_CHART_OBJ_QUICK_NAVIGATION)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SCALE        ?  CMessage::Text(MSG_CHART_OBJ_SCALE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_SCALEFIX        ?  CMessage::Text(MSG_CHART_OBJ_SCALEFIX)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SCALEFIX_11        ?  CMessage::Text(MSG_CHART_OBJ_SCALEFIX_11)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SCALE_PT_PER_BAR  ?  CMessage::Text(MSG_CHART_OBJ_SCALE_PT_PER_BAR)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_TICKER        ?  CMessage::Text(MSG_CHART_OBJ_SHOW_TICKER)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_OHLC        ?  CMessage::Text(MSG_CHART_OBJ_SHOW_OHLC)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_BID_LINE  ?  CMessage::Text(MSG_CHART_OBJ_SHOW_BID_LINE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_ASK_LINE  ?  CMessage::Text(MSG_CHART_OBJ_SHOW_ASK_LINE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_LAST_LINE ?  CMessage::Text(MSG_CHART_OBJ_SHOW_LAST_LINE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_PERIOD_SEP   ?  CMessage::Text(MSG_CHART_OBJ_SHOW_PERIOD_SEP)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_GRID        ?  CMessage::Text(MSG_CHART_OBJ_SHOW_GRID)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_VOLUMES   ?  CMessage::Text(MSG_CHART_OBJ_SHOW_VOLUMES)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+ChartModeVolumeDescription((ENUM_CHART_VOLUME_MODE)this.GetProperty(property))
         )  :
      property==CHART_PROP_SHOW_OBJECT_DESCR ?  CMessage::Text(MSG_CHART_OBJ_SHOW_OBJECT_DESCR)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_VISIBLE_BARS   ?  CMessage::Text(MSG_CHART_OBJ_VISIBLE_BARS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_WINDOWS_TOTAL  ?  CMessage::Text(MSG_CHART_OBJ_WINDOWS_TOTAL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_WINDOW_IS_VISIBLE  ?  CMessage::Text(MSG_CHART_OBJ_WINDOW_IS_VISIBLE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_WINDOW_HANDLE  ?  CMessage::Text(MSG_CHART_OBJ_WINDOW_HANDLE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_WINDOW_YDISTANCE  ?  CMessage::Text(MSG_CHART_OBJ_WINDOW_YDISTANCE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_FIRST_VISIBLE_BAR ?  CMessage::Text(MSG_CHART_OBJ_FIRST_VISIBLE_BAR)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_WIDTH_IN_BARS  ?  CMessage::Text(MSG_CHART_OBJ_WIDTH_IN_BARS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_WIDTH_IN_PIXELS   ?  CMessage::Text(MSG_CHART_OBJ_WIDTH_IN_PIXELS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_HEIGHT_IN_PIXELS   ?  CMessage::Text(MSG_CHART_OBJ_HEIGHT_IN_PIXELS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_COLOR_BACKGROUND        ?  CMessage::Text(MSG_CHART_OBJ_COLOR_BACKGROUND)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_FOREGROUND  ?  CMessage::Text(MSG_CHART_OBJ_COLOR_FOREGROUND)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_GRID     ?  CMessage::Text(MSG_CHART_OBJ_COLOR_GRID)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_VOLUME   ?  CMessage::Text(MSG_CHART_OBJ_COLOR_VOLUME)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_CHART_UP ?  CMessage::Text(MSG_CHART_OBJ_COLOR_CHART_UP)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_CHART_DOWN  ?  CMessage::Text(MSG_CHART_OBJ_COLOR_CHART_DOWN)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_CHART_LINE  ?  CMessage::Text(MSG_CHART_OBJ_COLOR_CHART_LINE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_CANDLE_BULL ?  CMessage::Text(MSG_CHART_OBJ_COLOR_CANDLE_BULL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_CANDLE_BEAR ?  CMessage::Text(MSG_CHART_OBJ_COLOR_CANDLE_BEAR)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_BID        ?  CMessage::Text(MSG_CHART_OBJ_COLOR_BID)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_ASK        ?  CMessage::Text(MSG_CHART_OBJ_COLOR_ASK)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_LAST        ?  CMessage::Text(MSG_CHART_OBJ_COLOR_LAST)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_COLOR_STOP_LEVEL  ?  CMessage::Text(MSG_CHART_OBJ_COLOR_STOP_LEVEL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::ColorToString((color)this.GetProperty(property),true)
         )  :
      property==CHART_PROP_SHOW_TRADE_LEVELS ?  CMessage::Text(MSG_CHART_OBJ_SHOW_TRADE_LEVELS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_DRAG_TRADE_LEVELS ?  CMessage::Text(MSG_CHART_OBJ_DRAG_TRADE_LEVELS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_DATE_SCALE   ?  CMessage::Text(MSG_CHART_OBJ_SHOW_DATE_SCALE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_PRICE_SCALE  ?  CMessage::Text(MSG_CHART_OBJ_SHOW_PRICE_SCALE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_SHOW_ONE_CLICK ?  CMessage::Text(MSG_CHART_OBJ_SHOW_ONE_CLICK)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_IS_MAXIMIZED   ?  CMessage::Text(MSG_CHART_OBJ_IS_MAXIMIZED)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_IS_MINIMIZED   ?  CMessage::Text(MSG_CHART_OBJ_IS_MINIMIZED)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_IS_DOCKED      ?  CMessage::Text(MSG_CHART_OBJ_IS_DOCKED)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_FLOAT_LEFT     ?  CMessage::Text(MSG_CHART_OBJ_FLOAT_LEFT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_FLOAT_TOP      ?  CMessage::Text(MSG_CHART_OBJ_FLOAT_TOP)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_FLOAT_RIGHT    ?  CMessage::Text(MSG_CHART_OBJ_FLOAT_RIGHT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_FLOAT_BOTTOM   ?  CMessage::Text(MSG_CHART_OBJ_FLOAT_BOTTOM)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

The blocks of the added code are identical to all the others — depending on the property passed to the method, the string with its description and value is created and returned.

**Let's improve the method displaying all object properties in the journal:**

```
//+------------------------------------------------------------------+
//| Display object properties in the journal                         |
//+------------------------------------------------------------------+
void CChartObj::Print(const bool full_prop=false)
  {
   ::Print("============= ",CMessage::Text(MSG_LIB_PARAMS_LIST_BEG)," (",this.Header(),") =============");
   int beg=0, end=CHART_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_CHART_PROP_INTEGER prop=(ENUM_CHART_PROP_INTEGER)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      if(prop==CHART_PROP_WINDOW_IND_HANDLE || prop==CHART_PROP_WINDOW_IND_INDEX) continue;
      if(prop==CHART_PROP_HEIGHT_IN_PIXELS)
        {
         this.PrintWndParameters();
         continue;
        }
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=CHART_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_CHART_PROP_DOUBLE prop=(ENUM_CHART_PROP_DOUBLE)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=CHART_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_CHART_PROP_STRING prop=(ENUM_CHART_PROP_STRING)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      if(prop==CHART_PROP_INDICATOR_NAME)
        {
         this.PrintWndIndicators();
         continue;
        }
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("============= ",CMessage::Text(MSG_LIB_PARAMS_LIST_END)," (",this.Header(),") =============\n");
  }
//+------------------------------------------------------------------+
```

If the loop features the "Indicator handle" and "Indicator index in the window" properties, they are skipped. They are not related to the chart object.

In case of the "Chart height in pixels" property, the method displaying the description of the property for all chart windows is called.

In case of the "Indicator name in the window" property, the method displaying the description of all indicators attached to all chart windows is called. We will consider these methods later.

**The method displaying a short object description** has been improved as well.

Now it also displays the number of chart window subwindows if they are presentor informs that there are no subwindows:

```
//+------------------------------------------------------------------+
//| Display a short description of the object in the journal         |
//+------------------------------------------------------------------+
void CChartObj::PrintShort(const bool dash=false)
  {
   ::Print
     (
      (dash ? "- " : ""),this.Header()," ID: ",(string)this.ID(),", HWND: ",(string)this.Handle(),
      ", ",CMessage::Text(MSG_CHART_OBJ_CHART_SUBWINDOWS_NUM),": ",(this.WindowsTotal()>1 ? string(this.WindowsTotal()-1) : CMessage::Text(MSG_LIB_TEXT_NO))
     );
  }
//+------------------------------------------------------------------+
```

**The method displaying data of all indicators of all chart windows in the journal:**

```
//+-------------------------------------------------------------------+
//| Display data of all indicators of all chart windows in the journal|
//+-------------------------------------------------------------------+
void CChartObj::PrintWndIndicators(void)
  {
   for(int i=0;i<this.WindowsTotal();i++)
     {
      CChartWnd *wnd=m_list_wnd.At(i);
      if(wnd==NULL)
         continue;
      wnd.PrintIndicators(true);
     }
  }
//+------------------------------------------------------------------+
```

Here in the loop by all chart window objects, get the next chart window object and display the descriptions of all indicators attached to the window in the journal. The method receives trueso that the hyphen is displayed before an indicator description.

**The method displaying the properties of all chart windows in the journal:**

```
//+------------------------------------------------------------------+
//| Display the properties of all chart windows in the journal       |
//+------------------------------------------------------------------+
void CChartObj::PrintWndParameters(void)
  {
   for(int i=0;i<this.WindowsTotal();i++)
     {
      CChartWnd *wnd=m_list_wnd.At(i);
      if(wnd==NULL)
         continue;
      wnd.PrintParameters(true);
     }
  }
//+------------------------------------------------------------------+
```

Here in the loop by all chart window objects, get the next chart window object anddisplay the description of its parameters in the journal. The method receives trueso that the hyphen is displayed before a window description.

**The method returning the window object by its subwindow index:**

```
//+------------------------------------------------------------------+
//| Return the window object by its subwindow index                  |
//+------------------------------------------------------------------+
CChartWnd *CChartObj::GetWindowByNum(const int win_num) const
  {
   int total=m_list_wnd.Total();
   for(int i=0;i<total;i++)
     {
      CChartWnd *wnd=m_list_wnd.At(i);
      if(wnd==NULL)
         continue;
      if(wnd.WindowNum()==win_num)
         return wnd;
     }
   return NULL;
  }
//+------------------------------------------------------------------+
```

Here in the loop by the total number of chart window objects, get the next window object.If its index matches the one passed to the method, return the pointer to the object found in the list. Upon the loop completion, return NULL — the object is not found.

**The method setting the "Subwindow visibility" property to the chart object:**

```
//+------------------------------------------------------------------+
//| Set the "Subwindow visibility" property                          |
//+------------------------------------------------------------------+
void CChartObj::SetVisible(void)
  {
   this.SetProperty(CHART_PROP_WINDOW_IS_VISIBLE,::ChartGetInteger(this.ID(),CHART_WINDOW_IS_VISIBLE,0));
  }
//+------------------------------------------------------------------+
```

Here, simply add the appropriate read-only property of the main chart window.

**The method returning the distance in Y axis pixels between the upper frame of the indicator subwindow and the upper frame of the main window:**

```
//+-----------------------------------------------------------------------------------+
//| Return the distance in pixels by Y axis between                                   |
//| the upper frame of the indicator subwindow and the upper frame of the main window |
//+-----------------------------------------------------------------------------------+
int CChartObj::WindowYDistance(const int sub_window) const
  {
   CChartWnd *wnd=GetWindowByNum(sub_window);
   return(wnd!=NULL ? wnd.YDistance() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
```

Here we receive the chart window object by its subwindow index and return the property value from the obtained object.

If the object is not received, return -1.

**The method returning the height of the specified chart in pixels:**

```
//+------------------------------------------------------------------+
//| Return the height of the specified chart in pixels               |
//+------------------------------------------------------------------+
int CChartObj::WindowHeightInPixels(const int sub_window) const
  {
   CChartWnd *wnd=GetWindowByNum(sub_window);
   return(wnd!=NULL ? wnd.HeightInPixels() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
```

Here we receive the chart window object by its subwindow index and return the property value from the obtained object.

If the object is not received, return -1.

**The method setting the height of the specified chart in pixels:**

```
//+------------------------------------------------------------------+
//| Set the height of the specified chart in pixels                  |
//+------------------------------------------------------------------+
bool CChartObj::SetWindowHeightInPixels(const int height,const int sub_window,const bool redraw=false)
  {
   CChartWnd *wnd=GetWindowByNum(sub_window);
   return(wnd!=NULL ? wnd.SetHeightInPixels(height,redraw) : false);
  }
//+------------------------------------------------------------------+
```

Here we receive the chart window object by its subwindow index and return the result of setting the appropriate property for the obtained object.

If the object is not received, return false.

Let's remove the old method I added in the previous article:

```
//+------------------------------------------------------------------+
//| Set the height of the specified chart in pixels                  |
//+------------------------------------------------------------------+
bool CChartObj::SetWindowHeightInPixels(const int height,const int sub_window,const bool redraw=false)
  {
   ::ResetLastError();
   if(!::ChartSetInteger(this.ID(),CHART_HEIGHT_IN_PIXELS,sub_window,height))
     {
      CMessage::ToLog(DFUN,::GetLastError(),true);
      return false;
     }
   if(redraw)
      ::ChartRedraw(this.ID());
   return true;
  }
//+------------------------------------------------------------------+
```

**This concludes the improvement of the chart object class.**

### Test

In order to check the performance of created objects, simply open any three charts. Add the fractals indicator to the chart with the EA + add the indicator window, for example, DeMarker containing another indicator, for example AMA based on DeMarker data.

On the second chart, I am going to place Stochastic window, while the third window will be undocked (Alt+D):

![](https://c.mql5.com/2/42/terminal64_u0AKKor0NN.png)

I will display short descriptions of all three chart objects and the full description of the current chart the EA is located at in the journal.

To perform the test, I will use the [EA from the previous article](https://www.mql5.com/en/articles/9213#node04) and save it in \\MQL5\\Experts\\TestDoEasy\ **Part68\** as **TestDoEasyPart68.mq5**.

The EA remains almost unchanged. All we need is adding the logic to the OnTick() handler:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Handle the NewTick event in the library
   engine.OnTick(rates_data);

//--- If working in the tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer(rates_data);   // Working in the timer
      PressButtonsControl();        // Button pressing control
      engine.EventsHandling();      // Working with events
     }

//--- If the trailing flag is set
   if(trailing_on)
     {
      TrailingPositions();          // Trailing positions
      TrailingOrders();             // Trailing pending orders
     }

//--- If it is the first launch
   static bool done=false;
   if(!done)
     {
      //--- Create the list object for storing chart objects
      CArrayObj *list=new CArrayObj();
      if(list==NULL)
         return;
      //--- Declare the variables and get the first chart ID
      long currChart,prevChart=ChartFirst();
      int i=0;
      //--- Create the chart object and add it to the list
      CChartObj *chart_first=new CChartObj(prevChart);
      list.Add(chart_first);
      //--- In the loop by the total number of terminal charts (not more than 100)
      while(i<CHARTS_MAX)
        {
         //--- based on the previous one, get the new chart
         currChart=ChartNext(prevChart);
         //--- When reaching the end of the chart list, complete the loop
         if(currChart<0) break;
         //--- Create the chart object based on the current chart ID in the loop and add it to the list
         CChartObj *chart=new CChartObj(currChart);
         list.Add(chart);
         //--- remember the current chart ID for ChartNext() and increase the loop counter
         prevChart=currChart;
         i++;
        }
      Print("");
      //--- From the filled list in the loop, receive the next chart object and display its short description
      int total=list.Total();
      for(int j=0;j<total;j++)
        {
         CChartObj *chart_obj=list.At(j);
         if(chart_obj!=NULL)
            chart_obj.PrintShort();
        }
      Print("");
      //--- Display the full description of the current chart: in the loop by all objects of the created list
      for(int j=0;j<total;j++)
        {
         //--- get the next chart object and
         CChartObj *chart_obj=list.At(j);
         //--- if its symbol matches the current chart symbol, display its full description in the journal
         if(chart_obj!=NULL && chart_obj.Symbol()==Symbol())
            chart_obj.Print();
        }
      //--- Destroy the list of chart objects
      delete list;
      done=true;
     }
//---
  }
//+------------------------------------------------------------------+
```

Compile the EA and launch it on the chart, while preliminarily creating the required terminal environment described at the very beginning of the section.

The journal displays short descriptions of all three open charts:

```
Main chart window EURUSD H4 ID: 131733844391938630, HWND: 5179646, Subwindows: 1
Main chart window AUDUSD H4 ID: 131733844391938634, HWND: 3672036, Subwindows: 1
Main chart window GBPUSD H4 ID: 131733844391938633, HWND: 3473910, Subwindows: No
```

and the full description of the current one. The result of the classes developed in the current article is displayed in the journal as ordered strings of window object propertiesand indicators attached to them:

```
============= The beginning of the parameter list (Main chart window EURUSD H4) =============
Chart ID: 131733844391938630
Timeframe: H4
Drawing attributes of a price chart: Yes
Object "Chart": No
Chart on top of other charts: No
Accessing the context menu by pressing the right mouse button: Yes
Accessing the "Crosshair tool" by pressing the middle mouse button: Yes
Scrolling the chart horizontally using the left mouse button: Yes
Sending messages about mouse wheel events to all mql5 programs on a chart: No
Send notifications of mouse move and mouse click events to all mql5 programs on a chart: No
Send a notification of an event of new object creation to all mql5-programs on a chart: No
Send a notification of an event of object deletion to all mql5-programs on a chart: No
Chart type: Display as Japanese candlesticks
Price chart in the foreground: No
Price chart indent from the right border: Yes
Automatic moving to the right border of the chart: Yes
Managing the chart using a keyboard: Yes
Allowed to intercept Space and Enter key presses on the chart to activate the quick navigation bar: Yes
Scale: 2
Fixed scale mode: No
Scale 1:1 mode: No
Scale to be specified in points per bar: No
Display a symbol ticker in the upper left corner: Yes
Display OHLC values in the upper left corner: Yes
Display Bid values as a horizontal line in a chart: Yes
Display Ask values as a horizontal line in a chart: Yes
Display Last values as a horizontal line in a chart: No
Display vertical separators between adjacent periods: No
Display grid in the chart: No
Display volume in the chart: Trade volumes
Display textual descriptions of objects: Yes
The number of bars on the chart that can be displayed: 137
The total number of chart windows, including indicator subwindows: 2
Visibility of subwindow: Yes
Chart window handle: 5179646
Number of the first visible bar in the chart: 136
Chart width in bars: 168
Chart width in pixels: 670
 Main chart window:
 - Chart height in pixels: 301
 Chart subwindow 1:
 - The distance between the upper frame of the indicator subwindow and the upper frame of the main chart window: 303
 - Chart height in pixels: 13
Chart background color: clrWhite
Color of axes, scales and OHLC line: clrBlack
Grid color: clrSilver
Color of volumes and position opening levels: clrGreen
Color for the up bar, shadows and body borders of bull candlesticks: clrBlack
Color for the down bar, shadows and body borders of bear candlesticks: clrBlack
Line chart color and color of "Doji" Japanese candlesticks: clrBlack
Body color of a bull candlestick: clrWhite
Body color of a bear candlestick: clrBlack
Bid price level color: clrLightSkyBlue
Ask price level color: clrCoral
Line color of the last executed deal price (Last): clrSilver
Color of stop order levels (Stop Loss and Take Profit): clrOrangeRed
Displaying trade levels in the chart (levels of open positions, Stop Loss, Take Profit and pending orders): Yes
Permission to drag trading levels on a chart with a mouse: Yes
Showing the time scale on a chart: Yes
Showing the price scale on a chart: Yes
Showing the "One click trading" panel on a chart: No
Chart window is maximized: Yes
Chart window is minimized: No
The chart window is docked: Yes
The left coordinate of the undocked chart window relative to the virtual screen: 0
The top coordinate of the undocked chart window relative to the virtual screen: 0
The right coordinate of the undocked chart window relative to the virtual screen: 0
The bottom coordinate of the undocked chart window relative to the virtual screen: 0
------
The size of the zero bar indent from the right border in percents: 18.93
Chart fixed position from the left border in percent value: 0.00
Fixed  chart maximum: 1.22620
Fixed  chart minimum : 1.17940
Scale in points per bar: 1.00
Chart minimum: 1.17940
Chart maximum: 1.22620
------
Text of a comment in a chart: ""
The name of the Expert Advisor running on the chart: "TestDoEasyPart68"
The name of the script running on the chart: ""
Indicators in the main chart window:
- Indicator Fractals
Indicators in the chart window 1:
- Indicator DeM(14)
- Indicator AMA(14,2,30)
Symbol: "EURUSD"
============= End of the parameter list (Main chart window EURUSD H4) =============
```

### What's next?

In the next article, I will start the development of the chart object collection class.

All files of the current version of the library are attached below together with the test EA file for MQL5 for you to test and download.

I do not recommend using chart objects in your work in their current state since they are to be changed further.

Leave your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/9236#node00)

**\*Previous articles within the series:**

[Prices in DoEasy library (Part 62): Updating tick series in real time, preparation for working with Depth of Market](https://www.mql5.com/en/articles/8988)

[Prices in DoEasy library (Part 63): Depth of Market and its abstract request class](https://www.mql5.com/en/articles/9010)

[Prices in DoEasy library (Part 64): Depth of Market, classes of DOM snapshot and snapshot series objects](https://www.mql5.com/en/articles/9044)

[Prices and Signals in DoEasy library (Part 65): Depth of Market collection and the class for working with MQL5.com Signals](https://www.mql5.com/en/articles/9095)

[Other classes in DoEasy library (Part 66): MQL5.com Signals collection class](https://www.mql5.com/en/articles/9146)

[Other classes in DoEasy library (Part 67): Chart object class](https://www.mql5.com/en/articles/9213)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/9236](https://www.mql5.com/ru/articles/9236)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/9236.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/9236/mql5.zip "Download MQL5.zip")(3945.98 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Tables in the MVC Paradigm in MQL5: Customizable and sortable table columns](https://www.mql5.com/en/articles/19979)
- [How to publish code to CodeBase: A practical guide](https://www.mql5.com/en/articles/19441)
- [Tables in the MVC Paradigm in MQL5: Integrating the Model Component into the View Component](https://www.mql5.com/en/articles/19288)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Resizable elements](https://www.mql5.com/en/articles/18941)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Containers](https://www.mql5.com/en/articles/18658)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Simple controls](https://www.mql5.com/en/articles/18221)
- [The View component for tables in the MQL5 MVC paradigm: Base graphical element](https://www.mql5.com/en/articles/17960)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/369338)**
(9)


![ゆうじ 保坂](https://c.mql5.com/avatar/2021/7/60F06EBE-82A4.jpg)

**[ゆうじ 保坂](https://www.mql5.com/en/users/popaipapa111-gmail)**
\|
7 Aug 2021 at 07:05

Please return the money!


![ゆうじ 保坂](https://c.mql5.com/avatar/2021/7/60F06EBE-82A4.jpg)

**[ゆうじ 保坂](https://www.mql5.com/en/users/popaipapa111-gmail)**
\|
7 Aug 2021 at 07:06

If you have a settlement offer, please present it!


![ゆうじ 保坂](https://c.mql5.com/avatar/2021/7/60F06EBE-82A4.jpg)

**[ゆうじ 保坂](https://www.mql5.com/en/users/popaipapa111-gmail)**
\|
7 Aug 2021 at 07:06

**ゆうじ 保坂:**

Please return the money!

![ゆうじ 保坂](https://c.mql5.com/avatar/2021/7/60F06EBE-82A4.jpg)

**[ゆうじ 保坂](https://www.mql5.com/en/users/popaipapa111-gmail)**
\|
7 Aug 2021 at 07:06

**ゆうじ 保坂:**

![Shino Unada](https://c.mql5.com/avatar/2017/6/5933685B-0DA7.png)

**[Shino Unada](https://www.mql5.com/en/users/unadajapon)**
\|
7 Aug 2021 at 07:29

**ゆうじ 保坂:**

If you have a settlement offer, please present it!

1\. you have no idea what you are talking about.

2\. if you are in trouble with MetaQuotes, tell the service desk.

3\. no one sees what you write on the forum. Don't waste any more time posting.

![Other classes in DoEasy library (Part 69): Chart object collection class](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__7.png)[Other classes in DoEasy library (Part 69): Chart object collection class](https://www.mql5.com/en/articles/9260)

With this article, I start the development of the chart object collection class. The class will store the collection list of chart objects with their subwindows and indicators providing the ability to work with any selected charts and their subwindows or with a list of several charts at once.

![Neural networks made easy (Part 13): Batch Normalization](https://c.mql5.com/2/48/Neural_networks_made_easy_013.png)[Neural networks made easy (Part 13): Batch Normalization](https://www.mql5.com/en/articles/9207)

In the previous article, we started considering methods aimed at improving neural network training quality. In this article, we will continue this topic and will consider another approach — batch data normalization.

![MVC design pattern and its possible application](https://c.mql5.com/2/42/MVC.png)[MVC design pattern and its possible application](https://www.mql5.com/en/articles/9168)

The article discusses a popular MVC pattern, as well as the possibilities, pros and cons of its usage in MQL programs. The idea is to split an existing code into three separate components: Model, View and Controller.

![Other classes in DoEasy library (Part 67): Chart object class](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__5.png)[Other classes in DoEasy library (Part 67): Chart object class](https://www.mql5.com/en/articles/9213)

In this article, I will create the chart object class (of a single trading instrument chart) and improve the collection class of MQL5 signal objects so that each signal object stored in the collection updates all its parameters when updating the list.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/9236&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071849042503217053)

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