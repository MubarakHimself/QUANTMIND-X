---
title: Other classes in DoEasy library (Part 71): Chart object collection events
url: https://www.mql5.com/en/articles/9360
categories: Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:12:23.568464
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/9360&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083403290508532500)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/9360#node01)
- [Improving library classes](https://www.mql5.com/en/articles/9360#node02)
- [Track chart events](https://www.mql5.com/en/articles/9360#node03)
- [Test](https://www.mql5.com/en/articles/9360#node04)
- [What's next?](https://www.mql5.com/en/articles/9360#node05)


### Concept

[In the previous article](https://www.mql5.com/en/articles/9293), I introduced the auto update of some chart object properties and related objects — opening a new/closing an existing symbol chart (chart object), adding a new/closing an existing indicator window from the chart object, as well as adding a new/removing/changing an existing indicator in the chart window.

In the current article, I am going to create an event functionality for chart objects, chart window objects and indicator objects in the chart window. This functionality will allow us to send custom events to the control program chart when registering the object events considered above.

Working with charts is implemented with manual chart control in mind, i.e. when users make changes in charts manually or using control elements of their program, operations are performed step by step one object at a time. A programmatic change of several objects at once within a single timer tick may cause an incorrect definition of occurring events. Only the last event will be recorded at best. At worst, changed objects will be defined incorrectly. I will certainly consider such a probability, while writing the code, and insert the code allowing to batch change object parameters wherever possible. However, I am not going to fine tune and test programmatic simultaneous handling of multiple chart elements. I will return to the task if there are requests from the library users.

### Improving library classes

Let's add new messages to the library. In \\MQL5\\Include\\DoEasy\ **Data.mqh**, add new message indices:

```
   MSG_CHART_OBJ_TEMPLATE_SAVED,                      // Chart template saved
   MSG_CHART_OBJ_TEMPLATE_APPLIED,                    // Template applied to chart

   MSG_CHART_OBJ_INDICATOR_ADDED,                     // Indicator added
   MSG_CHART_OBJ_INDICATOR_REMOVED,                   // Indicator removed
   MSG_CHART_OBJ_INDICATOR_CHANGED,                   // Indicator changed
   MSG_CHART_OBJ_WINDOW_ADDED,                        // Subwindow added
   MSG_CHART_OBJ_WINDOW_REMOVED,                      // Subwindow removed

//--- CChartObjCollection
   MSG_CHART_COLLECTION_TEXT_CHART_COLLECTION,        // Chart collection
   MSG_CHART_COLLECTION_ERR_FAILED_CREATE_CHART_OBJ,  // Failed to create a new chart object
   MSG_CHART_COLLECTION_ERR_FAILED_ADD_CHART,         // Failed to add a chart object to the collection
   MSG_CHART_COLLECTION_ERR_CHARTS_MAX,               // Cannot open new chart. Number of open charts at maximum

   MSG_CHART_COLLECTION_CHART_OPENED,                 // Chart opened
   MSG_CHART_COLLECTION_CHART_CLOSED,                 // Chart closed

  };
//+------------------------------------------------------------------+
```

and message texts corresponding to newly added indices:

```
   {"Шаблон графика сохранён","Chart template saved"},
   {"Шаблон применён к графику","Template applied to the chart"},

   {"Добавлен индикатор","Added indicator"},
   {"Удалён индикатор","Removed indicator"},
   {"Изменён индикатор","Changed indicator"},

   {"Добавлено подокно","Added subwindow"},
   {"Удалено подокно","Removed subwindow"},

//--- CChartObjCollection
   {"Коллекция чартов","Chart collection"},
   {"Не удалось создать новый объект-чарт","Failed to create new chart object"},
   {"Не удалось добавить объект-чарт в коллекцию","Failed to add chart object to collection"},
   {"Нельзя открыть новый график, так как количество открытых графиков уже максимальное","You cannot open a new chart, since the number of open charts is already maximum"},

   {"Открыт график","Open chart"},
   {"Закрыт график","Closed chart"},

  };
//+---------------------------------------------------------------------+
```

In the current article, we will handle some chart events. In order to track them and specify a certain event, create a new enumeration of possible chart events in \\MQL5\\Include\\DoEasy\ **Defines.mqh**:

```
//+------------------------------------------------------------------+
//| Data for working with charts                                     |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| List of possible chart events                                    |
//+------------------------------------------------------------------+
enum ENUM_CHART_OBJ_EVENT
  {
   CHART_OBJ_EVENT_NO_EVENT = SIGNAL_MQL5_EVENTS_NEXT_CODE, // No event
   CHART_OBJ_EVENT_CHART_OPEN,                        // "New chart opening" event
   CHART_OBJ_EVENT_CHART_CLOSE,                       // "Chart closure" event
   CHART_OBJ_EVENT_CHART_WND_ADD,                     // "Adding a new window on the chart" event
   CHART_OBJ_EVENT_CHART_WND_DEL,                     // "Removing a window from the chart" event
   CHART_OBJ_EVENT_CHART_WND_IND_ADD,                 // "Adding a new indicator to the chart window" event
   CHART_OBJ_EVENT_CHART_WND_IND_DEL,                 // "Removing an indicator from the chart window" event
   CHART_OBJ_EVENT_CHART_WND_IND_CHANGE,              // "Changing indicator parameters in the chart window" event
  };
#define CHART_OBJ_EVENTS_NEXT_CODE  (CHART_OBJ_EVENT_CHART_WND_IND_CHANGE+1)  // The code of the next event after the last chart event code
//+------------------------------------------------------------------+
```

When registering chart events specified in this enumeration, a custom event is sent to the program chart. This custom event is to contain the type of an occurred event using the appropriate constant from the enumeration. Next, the program analyzes the event code and handles it accordingly.

While implementing the event handling, I bumped into an unexpected challenge. It turned out difficult to define an index of an already deleted chart subwindow and the indicator inside it. To define them more conveniently, I decided to implement a new property for the indicator object in the chart window — an index of the window it is located in.

All deleted charts, chart windows and appropriate indicators (objects describing them) are to be stored in special lists allowing us to receive a previously deleted object at any time. This is the object we are able to get an indicator window index from (if the indicator was deleted).

Add a new property constant for an indicator object in the chart window to the enumeration of the chart object integer properties:

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
   CHART_PROP_WINDOW_NUM,                             // Chart window index
  };
#define CHART_PROP_INTEGER_TOTAL (67)                 // Total number of integer properties
#define CHART_PROP_INTEGER_SKIP  (0)                  // Number of integer DOM properties not used in sorting
//+------------------------------------------------------------------+
```

Since the number of integer properties has increased, increase their number from 66 to **67**.

Add sorting by the chart window index to the enumeration of chart object sorting criteria:

```
//+------------------------------------------------------------------+
//| Possible chart sorting criteria                                  |
//+------------------------------------------------------------------+
#define FIRST_CHART_DBL_PROP  (CHART_PROP_INTEGER_TOTAL-CHART_PROP_INTEGER_SKIP)
#define FIRST_CHART_STR_PROP  (CHART_PROP_INTEGER_TOTAL-CHART_PROP_INTEGER_SKIP+CHART_PROP_DOUBLE_TOTAL-CHART_PROP_DOUBLE_SKIP)
enum ENUM_SORT_CHART_MODE
  {
//--- Sort by integer properties
   SORT_BY_CHART_ID = 0,                              // Sort by chart ID
   SORT_BY_CHART_TIMEFRAME,                           // Sort by chart timeframe
   SORT_BY_CHART_SHOW,                                // Sort by the price chart drawing attribute
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
   SORT_BY_CHART_WINDOW_NUM,                          // Sort by chart window index
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
   SORT_BY_CHART_WINDOW_IND_NAME,                     // Sort by a name of an indicator launched in the chart window
   SORT_BY_CHART_SYMBOL,                              // Sort by chart symbol
  };
//+------------------------------------------------------------------+
```

As mentioned above, I will use special lists to store object copies of the removed indicators, windows and charts. We will need access to these lists in each chart object, its windows and indicators belonging to the windows. To avoid storing custom lists in each object class and arrange access to them from other objects requiring data on a certain deleted object, declare all these lists in the chart object collection class (the class provides access to all chart objects), while pointers to the lists are passed to other objects (chart, chart window, indicator in the chart window). Thus, each object stored in the collection has access to all lists.

This imposes some restrictions on the use of these lists within collection objects, other than the collection object itself. We cannot delete objects in the lists, reset the lists while working inside all collection objects, except for the collection itself, and modify pointers to the lists in any way. However, if we keep in mind this feature of using the pointer to the list (in fact, applying it in read-only mode), arranging the access to the list from different objects is simplified since passing the pointer to the list object is sufficient to read its contents.

Let's improve the indicator object class file in the chart window and the chart window object file (both classes are located in \\MQL5\\Include\\DoEasy\\Objects\\Chart\ **ChartWnd.mqh**).

In the private class section **CWndInd,** declare the variable for storing the subwindow index the indicator is located in, while in the public section of the class, write the methods for setting and returning all object properties (previously, we were able to set a single property — the indicator index in the window list):

```
//+------------------------------------------------------------------+
//| Chart window indicator object class                              |
//+------------------------------------------------------------------+
class CWndInd : public CObject
  {
private:
   long              m_chart_id;                         // Chart ID
   string            m_name;                             // Indicator short name
   int               m_index;                            // indicator index in the list
   int               m_window_num;                       // Indicator subwindow index
   int               m_handle;                           // Indicator handle
public:
//--- Return itself
   CWndInd          *GetObject(void)                     { return &this;               }
//--- Return (1) indicator name, (2) index in the list, (3) indicator handle and (4) subwindow index
   string            Name(void)                    const { return this.m_name;         }
   int               Index(void)                   const { return this.m_index;        }
   int               Handle(void)                  const { return this.m_handle;       }
   int               WindowNum(void)               const { return this.m_window_num;   }
//--- Set (1) subwindow name, (2) window index on the chart, (3) handle, (4) index
   void              SetName(const string name)          { this.m_name=name;           }
   void              SetIndex(const int index)           { this.m_index=index;         }
   void              SetHandle(const int handle)         { this.m_handle=handle;       }
   void              SetWindowNum(const int win_num)     { this.m_window_num=win_num;  }

//--- Display the description of object properties in the journal (dash=true - hyphen before the description, false - description only)
   void              Print(const bool dash=false)        { ::Print((dash ? "- " : "")+this.Header());                      }
//--- Return the object short name
   string            Header(void)                  const { return CMessage::Text(MSG_CHART_OBJ_INDICATOR)+" "+this.Name(); }

//--- Compare CWndInd objects with each other by the specified property
   virtual int       Compare(const CObject *node,const int mode=0) const;

//--- Constructors
                     CWndInd(void){;}
                     CWndInd(const int handle,const string name,const int index,const int win_num) : m_handle(handle),
                                                                                                     m_name(name),
                                                                                                     m_index(index),
                                                                                                     m_window_num(win_num) {}
  };
//+------------------------------------------------------------------+
```

In addition, add passing the chart subwindow index the indicator is located in and assigning the passed value to the appropriate variable to the parametric constructor.

Now when creating such an object, we should additionally specify the index of the subwindow featuring the indicator the object is created for. Thus, we are able to find the window featuring the indicator after removing the indicator and the window. This will make it easier for us to find the index of an already absent window in order to understand which of them contained the indicator removed along with the window, while their objects are located in the lists of removed chart objects.

Let's make changes in the chart window object as well. In the private section of the class, declare the pointers to the lists of indicators changed in the windowand removed from it, as well as declare the variable for storing a symbol of the chart the window belongs to. Declare the method returning the flag indicating the presence of the indicator from the window in the chart object list, the method returning the indicator object present in the list but not present on the chart window and the method for checking the changes of parameters of indicators present in the window:

```
//+------------------------------------------------------------------+
//| Chart window object class                                        |
//+------------------------------------------------------------------+
class CChartWnd : public CBaseObjExt
  {
private:
   CArrayObj         m_list_ind;                                        // Indicator list
   CArrayObj        *m_list_ind_del;                                    // Pointer to the list of indicators removed from the indicator window
   CArrayObj        *m_list_ind_param;                                  // Pointer to the list of changed indicators
   int               m_window_num;                                      // Subwindow index
   int               m_wnd_coord_x;                                     // The X coordinate for the time on the chart in the window
   int               m_wnd_coord_y;                                     // The Y coordinate for the price on the chart in the window
   string            m_symbol;                                          // Symbol of a chart the window belongs to
//--- Return the flag indicating the presence of an indicator (1) from the list in the window and (2) from the window in the list
   bool              IsPresentInWindow(const CWndInd *ind);
   bool              IsPresentInList(const string name);
//--- Return the indicator object present in the list but not present on the chart
   CWndInd          *GetMissingInd(void);
//--- Remove indicators not present in the window from the list
   void              IndicatorsDelete(void);
//--- Add new indicators to the list
   void              IndicatorsAdd(void);
//--- Check the changes of the parameters of existing indicators
   void              IndicatorsChangeCheck(void);

public:
```

Now the class is inherited [from the expanded base class of all library objects](https://www.mql5.com/en/articles/7663#node02) providing an event functionality that can be easily done for each of such objects.

In the public section of the class, namely, in the method returning the flag indicating that an object supports a specified property, add yet another supported property — chart symbol. Move the implementation of the method returning the string property description beyond the class body (and consider it further):

```
public:
//--- Return itself
   CChartWnd        *GetObject(void)                                    { return &this;            }

//--- Return the flag of the object supporting this property
   virtual bool      SupportProperty(ENUM_CHART_PROP_INTEGER property)  { return(property==CHART_PROP_WINDOW_YDISTANCE || property==CHART_PROP_HEIGHT_IN_PIXELS ? true : false); }
   virtual bool      SupportProperty(ENUM_CHART_PROP_DOUBLE property)   { return false; }
   virtual bool      SupportProperty(ENUM_CHART_PROP_STRING property)   { return (property==CHART_PROP_WINDOW_IND_NAME || property==CHART_PROP_SYMBOL ? true : false);  }

//--- Get description of (1) integer, (2) real and (3) string properties
   string            GetPropertyDescription(ENUM_CHART_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_CHART_PROP_DOUBLE property)  { return CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED);        }
   string            GetPropertyDescription(ENUM_CHART_PROP_STRING property);
```

In the public section of the class, declare the method for creating and sending a chart event to the control program chart. The class parametric constructor will now receive the chart symbol name and the pointers to the lists of removedand changed indicators of the window. Also, declare the class destructor:

```
//--- Return the object short name
   virtual string    Header(void);

//--- Create and send the chart event to the control program chart
   void              SendEvent(ENUM_CHART_OBJ_EVENT event);

//--- Compare CChartWnd objects by a specified property (to sort the list by an MQL5 signal object)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CChartWnd objects by all properties (to search for equal MQL5 signal objects)
   bool              IsEqual(CChartWnd* compared_obj) const;

//--- Constructors
                     CChartWnd(void){;}
                     CChartWnd(const long chart_id,const int wnd_num,const string symbol,CArrayObj *list_ind_del,CArrayObj *list_ind_param);
//--- Destructor
                    ~CChartWnd(void);
```

Next, add other necessary methods whose purpose is described in the listing comments:

```
//--- Return (1) the subwindow index, (2) the number of indicators attached to the window and (3) the name of a symbol chart
   int               WindowNum(void)                              const { return this.m_window_num;                                       }
   int               IndicatorsTotal(void)                        const { return this.m_list_ind.Total();                                 }
   string            Symbol(void)                                 const { return m_symbol;}
//--- Set (1) the subwindow index and (2) the chart symbol
   void              SetWindowNum(const int num)                        { this.m_window_num=num;                                          }
   void              SetSymbol(const string symbol)                     { this.m_symbol=symbol;                                           }

//--- Return (1) the indicator list, the window indicator object from the list by (2) index in the list and (3) by handle
   CArrayObj        *GetIndicatorsList(void)                            { return &this.m_list_ind;                                        }
   CWndInd          *GetIndicatorByIndex(const int index);
   CWndInd          *GetIndicatorByHandle(const int handle);
//--- Return (1) the last one added to the window, (2) the last one removed from the window and (3) the changed indicator
   CWndInd          *GetLastAddedIndicator(void)                        { return this.m_list_ind.At(this.m_list_ind.Total()-1);           }
   CWndInd          *GetLastDeletedIndicator(void)                      { return this.m_list_ind_del.At(this.m_list_ind_del.Total()-1);   }
   CWndInd          *GetLastChangedIndicator(void)                      { return this.m_list_ind_param.At(this.m_list_ind_param.Total()-1);}
```

Let's consider the implementation of new and improved methods in detail.

In the initialization list of the parametric class constructor, assign the value passed in the parameters to **m\_symbol**, while in the class body, assign the values of pointers (passed to the method) to the variable pointers to the lists:

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CChartWnd::CChartWnd(const long chart_id,const int wnd_num,const string symbol,CArrayObj *list_ind_del,CArrayObj *list_ind_param) : m_window_num(wnd_num),
                                                                                                                                    m_symbol(symbol),
                                                                                                                                    m_wnd_coord_x(0),
                                                                                                                                    m_wnd_coord_y(0)
  {
   this.m_list_ind_del=list_ind_del;
   this.m_list_ind_param=list_ind_param;
   CBaseObj::SetChartID(chart_id);
   this.IndicatorsListCreate();
  }
//+------------------------------------------------------------------+
```

In the previous article (namely, in the methods providing data on indicators attached to the window), the indicator handles were taken from the indicator list. After reading the indicator data, the handle was immediately released. As a result, a new handle was created each time for the same indicator (due to my incorrect understanding of Help information — the indicator handle should be released only when there is really no need for it anymore, namely when the program operation is completed, and not immediately after receiving data by the handle, while the indicator is to be further used by the program). I will fix this in the current article — handles of all indicators are to be released in the class destructor.

**The class destructor:**

```
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CChartWnd::~CChartWnd(void)
  {
   int total=this.m_list_ind.Total();
   for(int i=total-1;i>WRONG_VALUE;i--)
     {
      CWndInd *ind=this.m_list_ind.At(i);
      if(ind==NULL)
         continue;
      ::IndicatorRelease(ind.Handle());
      this.m_list_ind.Delete(i);
     }
  }
//+------------------------------------------------------------------+
```

Here in the loop by the list of window indicator objects, get the next object, release the indicator handleset in the object properties and remove the object itself.

**The virtual method of comparing two objects by a specified property** receives comparing by window index and by chart symbol:

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
   else if(mode==CHART_PROP_WINDOW_NUM)
      return(this.WindowNum()>obj_compared.WindowNum() ? 1 : this.WindowNum()<obj_compared.WindowNum() ? -1 : 0);
   else if(mode==CHART_PROP_SYMBOL)
      return(this.Symbol()==obj_compared.Symbol() ? 0 : this.Symbol()>obj_compared.Symbol() ? 1 : -1);
   return -1;
  }
//+------------------------------------------------------------------+
```

This is necessary to sort and search objects in the lists by chart symbol and the newly implemented object property — window index.

**Implementation of the method returning the description of the object string property** is moved outside the class body:

```
//+------------------------------------------------------------------+
//| Return description of object's string property                   |
//+------------------------------------------------------------------+
string CChartWnd::GetPropertyDescription(ENUM_CHART_PROP_STRING property)
  {
   return
     (
      property==CHART_PROP_SYMBOL  ?  CMessage::Text(MSG_LIB_PROP_SYMBOL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.Symbol()
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

Like in all similar methods of all library objects, here we create a text description of the property passed to the method:

if this is a "Chart symbol" property, return its string description, return an empty string for any other property.

If the property has the "property not supported" flag, the string with the same contents is returned.

**In the method creating the list of indicators attached to the window**, remove the string, in which the handle of the currently selected indicator is released (the reason was discussed above):

```
      //--- get and save the indicator handle by its short name
      int handle=::ChartIndicatorGet(this.m_chart_id,this.m_window_num,name);
      //--- Free the indicator handle
      ::IndicatorRelease(handle);
      //--- Create the new indicator object in the chart window
```

the string, in which we create a new indicator object in the chart window, receives the ability to pass the current window index class to the constructor:

```
//+------------------------------------------------------------------+
//| Create the list of indicators attached to the window             |
//+------------------------------------------------------------------+
void CChartWnd::IndicatorsListCreate(void)
  {
   //--- Clear the indicator lists
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
      //--- Create the new indicator object in the chart window
      CWndInd *ind=new CWndInd(handle,name,i,this.WindowNum());
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

Now each indicator object in the chart window will "know" the window it is located in.

The same is done to the **method adding new indicators to the list:**

Remove the string

```
      int handle=::ChartIndicatorGet(this.m_chart_id,this.m_window_num,name);
      //--- Free the indicator handle
      ::IndicatorRelease(handle);
      //--- Create the new indicator object in the chart window
```

and add passing the window index when creating a new indicator object in the chart window:

```
//+------------------------------------------------------------------+
//| Add new indicators to the list                                   |
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
      //--- Create the new indicator object in the chart window
      CWndInd *ind=new CWndInd(handle,name,i,this.WindowNum());
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

**In the method returning the flag of the indicator presence from the list in the window**, get rid of the string as well:

```
      int handle=::ChartIndicatorGet(this.m_chart_id,this.m_window_num,name);
      ::IndicatorRelease(handle);
```

Now we do not free the indicator handle in this method as well:

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
      if(ind.Name()==name && ind.Handle()==handle)
         return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

**The method checking changes in the parameters of existing indicators:**

```
//+------------------------------------------------------------------+
//| Check the changes of the parameters of existing indicators       |
//+------------------------------------------------------------------+
void CChartWnd::IndicatorsChangeCheck(void)
  {
   //--- Get the total number of indicators in the window
   int total=::ChartIndicatorsTotal(this.m_chart_id,this.m_window_num);
   //--- In the loop by all window indicators,
   for(int i=0;i<total;i++)
     {
      //--- get the indicator name and get its handle by a name
      string name=::ChartIndicatorName(this.m_chart_id,this.m_window_num,i);
      int handle=::ChartIndicatorGet(this.m_chart_id,this.m_window_num,name);
      //--- If the indicator with such a name is present in the object indicator list, move on to the next one
      if(this.IsPresentInList(name))
         continue;
      //--- Get the indicator object present in the list but not present in the window
      CWndInd *ind=this.GetMissingInd();
      if(ind==NULL)
         continue;
      //--- If the indicator and the detected object have the same index, this is the indicator with changed parameters
      if(ind.Index()==i)
        {
         //--- Create a new indicator object based on the detected indicator object,
         CWndInd *changed=new CWndInd(ind.Handle(),ind.Name(),ind.Index(),ind.WindowNum());
         if(changed==NULL)
            continue;
         //--- set the sorted list flag to the list of changed indicators
         this.m_list_ind_param.Sort();
         //--- If failed to add a newly created indicator object to the list of changed indicators,
         //--- remove the created object and move on to the next indicator in the window
         if(!this.m_list_ind_param.Add(changed))
           {
            delete changed;
            continue;
           }
         //--- Set the new parameters for the detected "lost" indicator - short name and handle
         ind.SetName(name);
         ind.SetHandle(handle);
         //--- and call the method of sending a custom event to the control program chart
         this.SendEvent(CHART_OBJ_EVENT_CHART_WND_IND_CHANGE);
        }
     }
  }
//+------------------------------------------------------------------+
```

The entire method logic is fully described in its listing. Indicators in the chart window are identified by their short names. If an indicator features a changed parameter, its short name should be changed (this is the case for correctly made custom indicators, while standard indicators take this feature into account). Therefore, the search for a changed indicator here is based on the fact that the index of the changed indicator in the window remains the same, unlike its short name. So, if we detect an indicator present in the window object list but not in the client terminal chart window, we need to check if the indices match — if the indicator and the detected object have the same index (when deleting the indicator, the indices of other indicators in the window are altered), then this is the indicator with changed parameters we were looking for.

Since the indicators removed from the window are to be stored in a special list (for their subsequent search when handling events), we need to improve **the method removing indicators not present in the window from the list:**

```
//+------------------------------------------------------------------+
//| Remove indicators not present in the window from the list        |
//+------------------------------------------------------------------+
void CChartWnd::IndicatorsDelete(void)
  {
   //--- In the loop by the list of window indicator objects,
   int total=this.m_list_ind.Total();
   for(int i=total-1;i>WRONG_VALUE;i--)
     {
      //--- get the next indicator object
      CWndInd *ind=this.m_list_ind.At(i);
      if(ind==NULL)
         continue;
      //--- If such an indicator is present in the chart window, move on to the next object in the list
      if(this.IsPresentInWindow(ind))
         continue;
      //--- Create a copy of a removed indicator
      CWndInd *ind_del=new CWndInd(ind.Handle(),ind.Name(),ind.Index(),ind.WindowNum());
      if(ind_del==NULL)
         continue;
      //--- If failed to place a created object to the list of indicators removed from the window,
      //--- remove it and go to the next object in the list
      if(!this.m_list_ind_del.Add(ind_del))
        {
         delete ind_del;
         continue;
        }
      //--- Remove the indicator, which was deleted from the window, from the list
      this.m_list_ind.Delete(i);
     }
  }
//+------------------------------------------------------------------+
```

The method logic is described in the code listing in detail. I believe, it requires no special explanations. If you have any questions, feel free to ask them in the comments below.

**The method returning the flag of the presence of an indicator from the window in the list:**

```
//+-----------------------------------------------------------------------------+
//| Return the flag of the presence of an indicator from the window in the list |
//+-----------------------------------------------------------------------------+
bool CChartWnd::IsPresentInList(const string name)
  {
   CWndInd *ind=new CWndInd();
   if(ind==NULL)
      return false;
   ind.SetName(name);
   this.m_list_ind.Sort(SORT_BY_CHART_WINDOW_IND_NAME);
   int index=this.m_list_ind.Search(ind);
   delete ind;
   return(index>WRONG_VALUE);
  }
//+------------------------------------------------------------------+
```

The method allows using a short name to define the presence of an appropriate indicator object in the window object list.

Here we create a temporary indicator object and assign a short name passed to the method. Set the flag of sorting by indicator name to the list of indicator objects and get the indicator index with such a name in the list. Make sure to delete the temporary object and return the flag indicating that the index of a detected indicator in the list exceeds -1 (if the indicator with such a name is found, its index exceeds -1).

**The method returning an indicator object present in the list but not in the chart window:**

```
//+------------------------------------------------------------------+
//| Return the indicator object present in the list                  |
//| but not on the chart                                             |
//+------------------------------------------------------------------+
CWndInd *CChartWnd::GetMissingInd(void)
  {
   for(int i=0;i<this.m_list_ind.Total();i++)
     {
      CWndInd *ind=this.m_list_ind.At(i);
      if(!this.IsPresentInWindow(ind))
         return ind;
     }
   return NULL;
  }
//+------------------------------------------------------------------+
```

Here, in the loop by all indicator objects in the list, get the next indicator object. If there is no such indicator on the chart, return the pointer to the detected indicator object. Otherwise, return NULL.

**The method returning the indicator object from the object list by the indicator index in the chart window list:**

```
//+------------------------------------------------------------------+
//| Return the indicator object by the index in the window list      |
//+------------------------------------------------------------------+
CWndInd *CChartWnd::GetIndicatorByIndex(const int index)
  {
   CWndInd *ind=new CWndInd();
   if(ind==NULL)
      return NULL;
   ind.SetIndex(index);
   this.m_list_ind.Sort(SORT_BY_CHART_WINDOW_IND_INDEX);
   int n=this.m_list_ind.Search(ind);
   delete ind;
   return this.m_list_ind.At(n);
  }
//+------------------------------------------------------------------+
```

Here, we create a temporary indicator object and assign the index passed to the method. Set the flag of sorting by indicator index on a chart to the list of indicator objects and get the indicator index in the window object list with such an index in the chart window list. Make sure to delete the temporary object and return the pointer to the object by a specified index in the list of indicator objects.

If the object is not found, [the Search() method](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjsearch) returns -1, while [the At() method](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjat) returns NULL in case of such index value. Thus, the method returns either a pointer to the detected object in the list, or NULL, if the indicator object with the specified index in the chart window is not present in the list.

**The method returning the window indicator object from the list by handle:**

```
//+------------------------------------------------------------------+
//| Return the window indicator object from the list by handle       |
//+------------------------------------------------------------------+
CWndInd *CChartWnd::GetIndicatorByHandle(const int handle)
  {
   CWndInd *ind=new CWndInd();
   if(ind==NULL)
      return NULL;
   ind.SetHandle(handle);
   this.m_list_ind.Sort(SORT_BY_CHART_WINDOW_IND_HANDLE);
   int index=this.m_list_ind.Search(ind);
   delete ind;
   return this.m_list_ind.At(index);
  }
//+------------------------------------------------------------------+
```

The method is identical to the one considered above, except that it receives a value of a handle of a necessary indicator object. Thus, the list is sorted by the "indicator handle" property, so that the search in the list is performed by this object property.

Improve **the method updating data by indicators attached to the window** for sending custom events to the control program chart in case of any changes in the number of their parameters or in the parameters themselves:

```
//+------------------------------------------------------------------+
//| Update data on attached indicators                               |
//+------------------------------------------------------------------+
void CChartWnd::Refresh(void)
  {
   //--- Calculate the change of the indicator number in the "now and during the previous check" window
   int change=::ChartIndicatorsTotal(this.m_chart_id,this.m_window_num)-this.m_list_ind.Total();
   //--- If there is no change in the number of indicators in the window,
   if(change==0)
     {
      //--- check the change of parameters of all indicators and exit
      this.IndicatorsChangeCheck();
      return;
     }
   //--- If indicators are added
   if(change>0)
     {
      //--- Call the method of adding new indicators to the list
      this.IndicatorsAdd();
      //--- In the loop by the number of indicators added to the window,
      for(int i=0;i<change;i++)
        {
         //--- get the new indicator in the list by the index calculated from the end of the list
         int index=this.m_list_ind.Total()-(1+i);
         //--- and if failed to obtain the object, move on to the next one
         CWndInd *ind=this.m_list_ind.At(index);
         if(ind==NULL)
            continue;
         //--- call the method of sending an event to the control program chart
         this.SendEvent(CHART_OBJ_EVENT_CHART_WND_IND_ADD);
        }
     }
   //--- If there are removed indicators
   if(change<0)
     {
      //--- Call the method of removing unnecessary indicators from the list
      this.IndicatorsDelete();
      //--- In the loop by the number of indicators removed from the window,
      for(int i=0;i<-change;i++)
        {
         //--- get a new removed indicator in the list of removed indicators by index calculated from the end of the list
         int index=this.m_list_ind_del.Total()-(1+i);
         //--- and if failed to obtain the object, move on to the next one
         CWndInd *ind=this.m_list_ind_del.At(index);
         if(ind==NULL)
            continue;
         //--- call the method of sending an event to the control program chart
         this.SendEvent(CHART_OBJ_EVENT_CHART_WND_IND_DEL);
        }
     }
  }
//+------------------------------------------------------------------+
```

The method logic is described in the code listing. I should note that the loops by newly added indicator objects in the indicator (or removed indicator) list are not needed in the current implementation of event handling. Now it is possible to immediately call the method of sending events. But the loop of searching for indicator objects newly added to the lists may be necessary in case we need to re-arrange the class for a correct tracking of changes of multiple indicators at once within a single timer tick. It is much easier to implement that programmatically but not manually. Currently, I am implementing handling manual changes on the chart. Therefore, I do not need these loops for now, although they may turn out to be of use during the future development.

**The method creating and sending a chart window event to the control program chart:**

```
//+------------------------------------------------------------------+
//| Create and send a chart window event                             |
//| to the control program chart                                     |
//+------------------------------------------------------------------+
void CChartWnd::SendEvent(ENUM_CHART_OBJ_EVENT event)
  {
   //--- If an indicator is added
   if(event==CHART_OBJ_EVENT_CHART_WND_IND_ADD)
     {
      //--- Get the last indicator object added to the list
      CWndInd *ind=this.GetLastAddedIndicator();
      if(ind==NULL)
         return;
      //--- Send the CHART_OBJ_EVENT_CHART_WND_IND_ADD event to the control program chart
      //--- pass the chart ID to lparam,
      //--- pass the chart window index to dparam,
      //--- pass the short name of the added indicator to sparam
      ::EventChartCustom(this.m_chart_id_main,(ushort)event,this.m_chart_id,this.WindowNum(),ind.Name());
     }
   //--- If the indicator is removed
   else if(event==CHART_OBJ_EVENT_CHART_WND_IND_DEL)
     {
      //--- Get the last indicator object added to the list of removed indicators
      CWndInd *ind=this.GetLastDeletedIndicator();
      if(ind==NULL)
         return;
      //--- Send the CHART_OBJ_EVENT_CHART_WND_IND_DEL event to the control program chart
      //--- pass the chart ID to lparam,
      //--- pass the chart window index to dparam,
      //--- pass the short name of a deleted indicator to sparam
      ::EventChartCustom(this.m_chart_id_main,(ushort)event,this.m_chart_id,this.WindowNum(),ind.Name());
     }
   //--- If the indicator has changed
   else if(event==CHART_OBJ_EVENT_CHART_WND_IND_CHANGE)
     {
      //--- Get the last indicator object added to the list of changed indicators
      CWndInd *ind=this.GetLastChangedIndicator();
      if(ind==NULL)
         return;
      //--- Send the CHART_OBJ_EVENT_CHART_WND_IND_CHANGE event to the control program chart
      //--- pass the chart ID to lparam,
      //--- pass the chart window index to dparam,
      //--- pass the short name of a changed indicator to sparam
      ::EventChartCustom(this.m_chart_id_main,(ushort)event,this.m_chart_id,this.WindowNum(),ind.Name());
     }
  }
//+------------------------------------------------------------------+
```

The entire method logic is fully described in its listing. If you have any questions, feel free to ask them in the comments below.

Now let's improve the chart object class in \\MQL5\\Include\\DoEasy\\Objects\\Chart\ **ChartObj.mqh**.

Similar to the chart window object class, make this class a descendant [of the class of the extended object of all library objects](https://www.mql5.com/en/articles/7663#node02), while in the private section of the class, declare the pointers to removed chart windows, removed and changed indicators, as well as the method for re-creating chart windows:

```
//+------------------------------------------------------------------+
//| Chart object class                                               |
//+------------------------------------------------------------------+
class CChartObj : public CBaseObjExt
  {
private:
   CArrayObj         m_list_wnd;                                  // List of chart window objects
   CArrayObj        *m_list_wnd_del;                              // Pointer to the list of chart window objects
   CArrayObj        *m_list_ind_del;                              // Pointer to the list of indicators removed from the indicator window
   CArrayObj        *m_list_ind_param;                            // Pointer to the list of changed indicators
   long              m_long_prop[CHART_PROP_INTEGER_TOTAL];       // Integer properties
   double            m_double_prop[CHART_PROP_DOUBLE_TOTAL];      // Real properties
   string            m_string_prop[CHART_PROP_STRING_TOTAL];      // String properties
   int               m_digits;                                    // Symbol's Digits()
   datetime          m_wnd_time_x;                                // Time for X coordinate on the windowed chart
   double            m_wnd_price_y;                               // Price for Y coordinate on the windowed chart

//--- Return the index of the array the (1) double and (2) string properties are actually located at
   int               IndexProp(ENUM_CHART_PROP_DOUBLE property)   const { return(int)property-CHART_PROP_INTEGER_TOTAL;                         }
   int               IndexProp(ENUM_CHART_PROP_STRING property)   const { return(int)property-CHART_PROP_INTEGER_TOTAL-CHART_PROP_DOUBLE_TOTAL; }

//--- The methods of setting parameter flags
   bool              SetShowFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetBringToTopFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetContextMenuFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetCrosshairToolFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetMouseScrollFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetEventMouseWhellFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetEventMouseMoveFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetEventObjectCreateFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetEventObjectDeleteFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetForegroundFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShiftFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetAutoscrollFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetKeyboardControlFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetQuickNavigationFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetScaleFixFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetScaleFix11Flag(const string source,const bool flag,const bool redraw=false);
   bool              SetScalePTPerBarFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowTickerFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowOHLCFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowBidLineFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowAskLineFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowLastLineFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowPeriodSeparatorsFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowGridFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowObjectDescriptionsFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowTradeLevelsFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetDragTradeLevelsFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowDateScaleFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowPriceScaleFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetShowOneClickPanelFlag(const string source,const bool flag,const bool redraw=false);
   bool              SetDockedFlag(const string source,const bool flag,const bool redraw=false);

//--- The methods of setting property values
   bool              SetMode(const string source,const ENUM_CHART_MODE mode,const bool redraw=false);
   bool              SetScale(const string source,const int scale,const bool redraw=false);
   bool              SetModeVolume(const string source,const ENUM_CHART_VOLUME_MODE mode,const bool redraw=false);
   void              SetVisibleBars(void);
   void              SetWindowsTotal(void);
   void              SetFirstVisibleBars(void);
   void              SetWidthInBars(void);
   void              SetWidthInPixels(void);
   void              SetMaximizedFlag(void);
   void              SetMinimizedFlag(void);
   void              SetExpertName(void);
   void              SetScriptName(void);

//--- (1) Create, (2) check and re-create the chart window list
   void              CreateWindowsList(void);
   void              RecreateWindowsList(const int change);
//--- Add an extension to the screenshot file if it is missing
   string            FileNameWithExtention(const string filename);

public:
```

In the public section of the class, write and declare new methods for working with class events.

Pass the pointers to new lists to the parametric constructor:

```
public:
//--- Set object's (1) integer, (2) real and (3) string properties
   void              SetProperty(ENUM_CHART_PROP_INTEGER property,long value)    { this.m_long_prop[property]=value;                      }
   void              SetProperty(ENUM_CHART_PROP_DOUBLE property,double value)   { this.m_double_prop[this.IndexProp(property)]=value;    }
   void              SetProperty(ENUM_CHART_PROP_STRING property,string value)   { this.m_string_prop[this.IndexProp(property)]=value;    }
//--- Return object’s (1) integer, (2) real and (3) string property from the properties array
   long              GetProperty(ENUM_CHART_PROP_INTEGER property)         const { return this.m_long_prop[property];                     }
   double            GetProperty(ENUM_CHART_PROP_DOUBLE property)          const { return this.m_double_prop[this.IndexProp(property)];   }
   string            GetProperty(ENUM_CHART_PROP_STRING property)          const { return this.m_string_prop[this.IndexProp(property)];   }
//--- Return (1) itself, (2) the window object list and (3) the list of removed window objects
   CChartObj        *GetObject(void)                                             { return &this;               }
   CArrayObj        *GetList(void)                                               { return &this.m_list_wnd;    }

//--- Return the last (1) added (removed) chart window
   CChartWnd        *GetLastAddedWindow(void)                                    { return this.m_list_wnd.At(this.m_list_wnd.Total()-1);           }
   CChartWnd        *GetLastDeletedWindow(void)                                  { return this.m_list_wnd_del.At(this.m_list_wnd_del.Total()-1);   }

//--- Return (1) the last one added to the window, (2) the last one removed from the window and (3) the changed indicator,
   CWndInd          *GetLastAddedIndicator(const int win_num);
   CWndInd          *GetLastDeletedIndicator(void)                               { return this.m_list_ind_del.At(this.m_list_ind_del.Total()-1);   }
   CWndInd          *GetLastChangedIndicator(void)                               { return this.m_list_ind_param.At(this.m_list_ind_param.Total()-1);}
//--- Return the indicator by index from the specified chart window
   CWndInd          *GetIndicator(const int win_num,const int ind_index);

//--- Return the flag of the object supporting this property
   virtual bool      SupportProperty(ENUM_CHART_PROP_INTEGER property)           { return (property!=CHART_PROP_WINDOW_YDISTANCE ? true : false);  }
   virtual bool      SupportProperty(ENUM_CHART_PROP_DOUBLE property)            { return true; }
   virtual bool      SupportProperty(ENUM_CHART_PROP_STRING property)            { return true; }

//--- Get description of (1) integer, (2) real and (3) string properties
   string            GetPropertyDescription(ENUM_CHART_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_CHART_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_CHART_PROP_STRING property);

//--- Display the description of object properties in the journal (full_prop=true - all properties, false - supported ones only)
   void              Print(const bool full_prop=false);
//--- Display a short description of the object in the journal
   virtual void      PrintShort(const bool dash=false);
//--- Return the object short name
   virtual string    Header(void);

//--- Create and send the chart event to the control program chart
   void              SendEvent(ENUM_CHART_OBJ_EVENT event);

//--- Compare CChartObj objects by a specified property (to sort the list by a specified chart object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CChartObj objects by all properties (to search for equal chart objects)
   bool              IsEqual(CChartObj* compared_obj) const;

//--- Update the chart object and its list of indicator windows
   void              Refresh(void);

//--- Constructors
                     CChartObj(){;}
                     CChartObj(const long chart_id,CArrayObj *list_wnd_del,CArrayObj *list_ind_del,CArrayObj *list_ind_param);
//+------------------------------------------------------------------+
```

Let's consider new and improved class methods.

**In the parametric constructor**, let the variables storing the pointers to list objects receive the pointers to them passed to the method:

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CChartObj::CChartObj(const long chart_id,CArrayObj *list_wnd_del,CArrayObj *list_ind_del,CArrayObj *list_ind_param) : m_wnd_time_x(0),m_wnd_price_y(0)
  {
   this.m_list_wnd_del=list_wnd_del;
   this.m_list_ind_del=list_ind_del;
   this.m_list_ind_param=list_ind_param;
//--- Set chart ID to the base object
```

At the end of the constructor listing, set the flag of sorting by chart window index to the list of removed chart object windows:

```
   this.m_digits=(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS);
   this.m_list_wnd_del.Sort(SORT_BY_CHART_WINDOW_NUM);
   this.CreateWindowsList();
  }
//+------------------------------------------------------------------+
```

**The method returning the last indicator added to the window:**

```
//+------------------------------------------------------------------+
//| Return the last indicator added to the window                    |
//+------------------------------------------------------------------+
CWndInd *CChartObj::GetLastAddedIndicator(const int win_num)
  {
   CChartWnd *wnd=this.GetWindowByNum(win_num);
   return(wnd!=NULL ? wnd.GetLastAddedIndicator() : NULL);
  }
//+------------------------------------------------------------------+
```

The method receives the window index of the chart, from which we want to get the last added indicator. Use the GetWindowByNum() method to get the required chart window. In turn, get the last added indicator from that chart window. If failed to get the chart window object, return NULL. Keep in mind that the GetLastAddedIndicator() chart window object method can also return NULL.

**The method returning the indicator by index from the specified chart window:**

```
//+------------------------------------------------------------------+
//| Return the indicator by index from the specified chart window    |
//+------------------------------------------------------------------+
CWndInd *CChartObj::GetIndicator(const int win_num,const int ind_index)
  {
   CChartWnd *wnd=this.GetWindowByNum(win_num);
   return(wnd!=NULL ? wnd.GetIndicatorByIndex(ind_index) : NULL);
  }
//+------------------------------------------------------------------+
```

The method receives the chart window index, from which we want to get the last added indicator, and the indicator index in the window list.

Use the GetWindowByNum() method to get the required chart window and get the indicator by its index in the chart window. If failed to get the chart window object, return NULL. Keep in mind that the GetIndicatorByIndex() chart window object method can also return NULL.

**In the method updating the chart object and the list of its windows, replace the CreateWindowsList() method with the new method RecreateWindowsList():**

```
//+------------------------------------------------------------------+
//| Update the chart object and its window list                      |
//+------------------------------------------------------------------+
void CChartObj::Refresh(void)
  {
   for(int i=0;i<this.m_list_wnd.Total();i++)
     {
      CChartWnd *wnd=this.m_list_wnd.At(i);
      if(wnd==NULL)
         continue;
      wnd.Refresh();
     }
   int change=(int)::ChartGetInteger(this.m_chart_id,CHART_WINDOWS_TOTAL)-this.WindowsTotal();
   if(change==0)
      return;
   this.RecreateWindowsList(change);
  }
//+------------------------------------------------------------------+
```

The method for creating the list of chart windows CreateWindowsList() is to be used only for building windows when launching the program. The method for re-constructing changed windows (considered below) is to be used when updating the chart object.

Now, when creating a new chart window object, we need to pass the chart symbol name and the pointers to the lists to it. So, let's implement passing them to the chart window object class constructor when creating a new chart window object **in the method of creating the chart window list:**

```
//+------------------------------------------------------------------+
//| Create the list of chart windows                                 |
//+------------------------------------------------------------------+
void CChartObj::CreateWindowsList(void)
  {
   //--- Clear the chart window list
   this.m_list_wnd.Clear();
   //--- Get the total number of chart windows from the environment
   int total=(int)::ChartGetInteger(this.m_chart_id,CHART_WINDOWS_TOTAL);
   //--- In the loop by the total number of windows
   for(int i=0;i<total;i++)
     {
      //--- Create a new chart window object
      CChartWnd *wnd=new CChartWnd(this.m_chart_id,i,this.Symbol(),this.m_list_ind_del,this.m_list_ind_param);
      if(wnd==NULL)
         continue;
      //--- If the window index exceeds 0 (not the main chart window) and it still has no indicator,
      //--- remove the newly created chart window object and go to the next loop iteration
      if(wnd.WindowNum()!=0 && wnd.IndicatorsTotal()==0)
        {
         delete wnd;
         continue;
        }
      //--- If the object was not added to the list, remove that object
      this.m_list_wnd.Sort();
      if(!this.m_list_wnd.Add(wnd))
         delete wnd;
     }
   //--- If the number of objects in the list corresponds to the number of windows on the chart,
   //--- write that value to the chart object property
   //--- If the number of objects in the list does not correspond to the number of windows on the chart,
   //--- write the number of objects in the list to the chart object property.
   int value=int(this.m_list_wnd.Total()==total ? total : this.m_list_wnd.Total());
   this.SetProperty(CHART_PROP_WINDOWS_TOTAL,value);
  }
//+------------------------------------------------------------------+
```

**The method for checking the changes of chart windows and re-creating their list:**

```
//+------------------------------------------------------------------+
//| Check and re-create the chart window list                        |
//+------------------------------------------------------------------+
void CChartObj::RecreateWindowsList(const int change)
  {
//--- If the window is removed
   if(change<0)
     {
      //--- If the chart has only one window, this means we have only the main chart with no subwindows,
      //--- while the change in the number of chart windows indicates the removal of a symbol chart in the terminal.
      //--- This situation is handled in the collection class of chart objects - leave the method
      if(this.WindowsTotal()==1)
         return;
      //--- Get the last removed indicator from the list of removed indicators
      CWndInd *ind=this.m_list_ind_del.At(this.m_list_ind_del.Total()-1);
      //--- If managed to get the indicator,
      if(ind!=NULL)
        {
         //--- create a new chart window object
         CChartWnd *wnd=new CChartWnd();
         if(wnd!=NULL)
           {
            //--- Set the subwindow index from the last removed indicator object,
            //--- ID and the chart object symbol name for a new object
            wnd.SetWindowNum(ind.WindowNum());
            wnd.SetChartID(this.ID());
            wnd.SetSymbol(this.Symbol());
            //--- If failed to add the created object to the list of removed chart window objects, remove it
            if(!this.m_list_wnd_del.Add(wnd))
               delete wnd;
           }
        }
      //--- Call the method of sending an event to the control program chart and re-create the chart window list
      this.SendEvent(CHART_OBJ_EVENT_CHART_WND_DEL);
      this.CreateWindowsList();
      return;
     }
//--- If there are no changes, leave
   else if(change==0)
      return;

//--- If a window is added
   //--- Get the total number of chart windows from the environment
   int total=(int)::ChartGetInteger(this.m_chart_id,CHART_WINDOWS_TOTAL);
   //--- In the loop by the total number of windows
   for(int i=0;i<total;i++)
     {
      //--- Create a new chart window object
      CChartWnd *wnd=new CChartWnd(this.m_chart_id,i,this.Symbol(),this.m_list_ind_del,this.m_list_ind_param);
      if(wnd==NULL)
         continue;
      this.m_list_wnd.Sort(SORT_BY_CHART_WINDOW_NUM);
      //--- If the window index exceeds 0 (not the main chart window) and it still has no indicator,
      //--- or such a window is already present in the list or the window object is not added to the list
      //--- remove the newly created chart window object and go to the next loop iteration
      if((wnd.WindowNum()!=0 && wnd.IndicatorsTotal()==0) || this.m_list_wnd.Search(wnd)>WRONG_VALUE || !this.m_list_wnd.Add(wnd))
        {
         delete wnd;
         continue;
        }
      //--- If added the window, call the method of sending an event to the control program chart
      this.SendEvent(CHART_OBJ_EVENT_CHART_WND_ADD);
     }
   //--- If the number of objects in the list corresponds to the number of windows on the chart,
   //--- write that value to the chart object property
   //--- If the number of objects in the list does not correspond to the number of windows on the chart,
   //--- write the number of objects in the list to the chart object property.
   int value=int(this.m_list_wnd.Total()==total ? total : this.m_list_wnd.Total());
   this.SetProperty(CHART_PROP_WINDOWS_TOTAL,value);
  }
//+------------------------------------------------------------------+
```

**The method creating and sending a chart event to the control program chart:**

```
//+------------------------------------------------------------------+
//| Create and send a chart event                                    |
//| to the control program chart                                     |
//+------------------------------------------------------------------+
void CChartObj::SendEvent(ENUM_CHART_OBJ_EVENT event)
  {
   //--- If a window is added
   if(event==CHART_OBJ_EVENT_CHART_WND_ADD)
     {
      //--- Get the last chart window object added to the list
      CChartWnd *wnd=this.GetLastAddedWindow();
      if(wnd==NULL)
         return;
      //--- Send the CHART_OBJ_EVENT_CHART_WND_ADD event to the control program chart
      //--- pass the chart ID to lparam,
      //--- pass the chart window index to dparam,
      //--- pass the chart symbol to sparam
      ::EventChartCustom(this.m_chart_id_main,(ushort)event,this.m_chart_id,wnd.WindowNum(),this.Symbol());
     }
   //--- If the window is removed
   else if(event==CHART_OBJ_EVENT_CHART_WND_DEL)
     {
      //--- Get the last chart window object added to the list of removed windows
      CChartWnd *wnd=this.GetLastDeletedWindow();
      if(wnd==NULL)
         return;
      //--- Send the CHART_OBJ_EVENT_CHART_WND_DEL event to the control program chart
      //--- pass the chart ID to lparam,
      //--- pass the chart window index to dparam,
      //--- pass the chart symbol to sparam
      ::EventChartCustom(this.m_chart_id_main,(ushort)event,this.m_chart_id,wnd.WindowNum(),this.Symbol());
     }
  }
//+------------------------------------------------------------------+
```

The entire logic of the last two methods is completely described in their listings and requires no explanations.

If you have any questions related to the methods, feel free to ask them in the comments.

Improve the chart object collection class in \\MQL5\\Include\\DoEasy\\Collections\ **ChartObjCollection.mqh**.

**In the private section of the class**, declare the list objects, the pointers to which were passed to the chart window and indicator objects in the chart windows:

```
//+------------------------------------------------------------------+
//| MQL5 signal object collection                                    |
//+------------------------------------------------------------------+
class CChartObjCollection : public CBaseObj
  {
private:
   CListObj                m_list;                                   // List of chart objects
   CListObj                m_list_del;                               // List of deleted chart objects
   CArrayObj               m_list_wnd_del;                           // List of deleted chart window objects
   CArrayObj               m_list_ind_del;                           // List of indicators removed from the indicator window
   CArrayObj               m_list_ind_param;                         // List of changed indicators
   int                     m_charts_total_prev;                      // Previous number of charts in the terminal
   //--- Return the number of charts in the terminal
   int                     ChartsTotal(void) const;
   //--- Return the flag indicating the existence of (1) a chart object and (2) a chart
   bool                    IsPresentChartObj(const long chart_id);
   bool                    IsPresentChart(const long chart_id);
   //--- Create a new chart object and add it to the list
   bool                    CreateNewChartObj(const long chart_id,const string source);
   //--- Find the missing chart object, create it and add it to the collection list
   bool                    FindAndCreateMissingChartObj(void);
   //--- Find a chart object not present in the terminal and remove it from the list
   void                    FindAndDeleteExcessChartObj(void);
public:
```

Instead of the pointers to the lists, here we declare the [CArrayObj objects](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj), in which we will store all deleted chart objects.

**In the public section of the class,** declare new methods required for working with chart object events:

```
public:
//--- Return (1) itself, (2) chart object collection list, (3) the list of deleted chart objects,
//--- the list (4) of deleted window objects, (5) deleted and (6) changed indicators
   CChartObjCollection    *GetObject(void)                                 { return &this;                  }
   CArrayObj              *GetList(void)                                   { return &this.m_list;           }
   CArrayObj              *GetListDeletedCharts(void)                      { return &this.m_list_del;       }
   CArrayObj              *GetListDeletedWindows(void)                     { return &this.m_list_wnd_del;   }
   CArrayObj              *GetListDeletedIndicators(void)                  { return &this.m_list_ind_del;   }
   CArrayObj              *GetListChangedIndicators(void)                  { return &this.m_list_ind_param; }
   //--- Return the list by selected (1) integer, (2) real and (3) string properties meeting the compared criterion
   CArrayObj              *GetList(ENUM_CHART_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByChartProperty(this.GetList(),property,value,mode);  }
   CArrayObj              *GetList(ENUM_CHART_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByChartProperty(this.GetList(),property,value,mode);  }
   CArrayObj              *GetList(ENUM_CHART_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByChartProperty(this.GetList(),property,value,mode);  }
//--- Return the number of chart objects in the list
   int                     DataTotal(void)                           const { return this.m_list.Total();    }
//--- Display (1) the complete and (2) short collection description in the journal
   void                    Print(void);
   void                    PrintShort(void);
//--- Constructor
                           CChartObjCollection();

//--- Return the list of chart objects by (1) symbol and (2) timeframe
   CArrayObj              *GetChartsList(const string symbol)              { return this.GetList(CHART_PROP_SYMBOL,symbol,EQUAL);      }
   CArrayObj              *GetChartsList(const ENUM_TIMEFRAMES timeframe)  { return this.GetList(CHART_PROP_TIMEFRAME,timeframe,EQUAL);}
//--- Return the pointer to the chart object (1) by ID and (2) by an index in the list
   CChartObj              *GetChart(const long id);
   CChartObj              *GetChart(const int index)                       { return this.m_list.At(index);                             }
//--- Return (1) the last added chart and (2) the last removed chart
   CChartObj              *GetLastAddedChart(void)                         { return this.m_list.At(this.m_list.Total()-1);             }
   CChartObj              *GetLastDeletedChart(void)                       { return this.m_list_del.At(this.m_list_del.Total()-1);     }

//--- Return (1) the last added window on the chart by chart ID and (2) the last removed chart window
   CChartWnd              *GetLastAddedChartWindow(const long chart_id);
   CChartWnd              *GetLastDeletedChartWindow(void)                 { return this.m_list_wnd_del.At(this.m_list_wnd_del.Total()-1);}

//--- Return (1) the last one added to the specified window of the specified chart, (2) the last one removed from the window and (3) the changed indicator
   CWndInd                *GetLastAddedIndicator(const long chart_id,const int win_num);
   CWndInd                *GetLastDeletedIndicator(void)                   { return this.m_list_ind_del.At(this.m_list_ind_del.Total()-1);   }
   CWndInd                *GetLastChangedIndicator(void)                   { return this.m_list_ind_param.At(this.m_list_ind_param.Total()-1);}
//--- Return the indicator by index from the specified window of the specified chart
   CWndInd                *GetIndicator(const long chart_id,const int win_num,const int ind_index);

//--- Return the chart ID with the program
   long                    GetMainChartID(void)                      const { return CBaseObj::GetMainChartID();                        }
//--- Create the collection list of chart objects
   bool                    CreateCollection(void);
//--- Update (1) the chart object collection list and (2) the specified chart object
   void                    Refresh(void);
   void                    Refresh(const long chart_id);

//--- (1) Open a new chart with the specified symbol and period, (2) close the specified chart
   bool                    Open(const string symbol,const ENUM_TIMEFRAMES timeframe);
   bool                    Close(const long chart_id);

//--- Create and send the chart event to the control program chart
   void                    SendEvent(ENUM_CHART_OBJ_EVENT event);

  };
//+------------------------------------------------------------------+
```

Let's consider the implementation of new methods and improvement of the exisiting ones.

**In the class constructor**, clear the new lists and set the sorted list flags to them:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CChartObjCollection::CChartObjCollection()
  {
   this.m_list.Clear();
   this.m_list.Sort();
   this.m_list_ind_del.Clear();
   this.m_list_ind_del.Sort();
   this.m_list_ind_param.Clear();
   this.m_list_ind_param.Sort();
   this.m_list_wnd_del.Clear();
   this.m_list_wnd_del.Sort();
   this.m_list_del.Clear();
   this.m_list_del.Sort();
   this.m_list.Type(COLLECTION_CHARTS_ID);
   this.m_charts_total_prev=this.ChartsTotal();
  }
//+------------------------------------------------------------------+
```

**In the method declaring the chart object collection list**, add calling the method of sending events to the control program chart:

```
//+------------------------------------------------------------------+
//| Update the collection list of chart objects                      |
//+------------------------------------------------------------------+
void CChartObjCollection::Refresh(void)
  {
   //--- In the loop by the number of chart objects in the list,
   for(int i=0;i<this.m_list.Total();i++)
     {
      //--- get the next chart object and
      CChartObj *chart=this.m_list.At(i);
      if(chart==NULL)
         continue;
      //--- update it
      chart.Refresh();
     }
   //--- Get the number of open charts in the terminal and
   int charts_total=this.ChartsTotal();
   //--- calculate the difference between the number of open charts in the terminal
   //--- and chart objects in the collection list
   int change=charts_total-this.m_list.Total();
   //--- If there are no changes, leave
   if(change==0)
      return;
   //--- If a chart is added in the terminal
   if(change>0)
     {
      //--- Find the missing chart object, create and add it to the collection list
      this.FindAndCreateMissingChartObj();
      //--- Get the current chart and return to it since
      //--- adding a new chart switches the focus to it
      CChartObj *chart=this.GetChart(GetMainChartID());
      if(chart!=NULL)
         chart.SetBringToTopON(true);
      for(int i=0;i<change;i++)
        {
         chart=m_list.At(m_list.Total()-(1+i));
         if(chart==NULL)
            continue;
         this.SendEvent(CHART_OBJ_EVENT_CHART_OPEN);
        }
     }
   //--- If a chart is removed in the terminal
   else if(change<0)
     {
      //--- Find an extra chart object in the collection list and remove it from the list
      this.FindAndDeleteExcessChartObj();
      for(int i=0;i<-change;i++)
        {
         CChartObj *chart=this.m_list_del.At(this.m_list_del.Total()-(1+i));
         if(chart==NULL)
            continue;
         this.SendEvent(CHART_OBJ_EVENT_CHART_CLOSE);
        }
     }
  }
//+------------------------------------------------------------------+
```

The code blocks, in which the SendEvent() method is called, are identical to the ones previously considered for the chart window object class and are also implemented with their possible future improvement in mind. The loops are not required to handle manual changes in terminal charts — we can call the method of sending events immediately.

**In the method creating a new chart object and adding it to the list**, it is now possible to pass the pointers to new lists when creating a new chart object. Let's add these changes:

```
//+------------------------------------------------------------------+
//| Create a new chart object and add it to the list                 |
//+------------------------------------------------------------------+
bool CChartObjCollection::CreateNewChartObj(const long chart_id,const string source)
  {
   ::ResetLastError();
   CChartObj *chart_obj=new CChartObj(chart_id,this.GetListDeletedWindows(),this.GetListDeletedIndicators(),this.GetListChangedIndicators());
   if(chart_obj==NULL)
     {
      CMessage::ToLog(source,MSG_CHART_COLLECTION_ERR_FAILED_CREATE_CHART_OBJ,true);
      return false;
     }
   this.m_list.Sort(SORT_BY_CHART_ID);
   if(!this.m_list.InsertSort(chart_obj))
     {
      CMessage::ToLog(source,MSG_CHART_COLLECTION_ERR_FAILED_ADD_CHART,true);
      delete chart_obj;
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

**The method returning the last added window to the chart by chart ID :**

```
//+------------------------------------------------------------------+
//| Return the last added window to the chart by ID                  |
//+------------------------------------------------------------------+
CChartWnd* CChartObjCollection::GetLastAddedChartWindow(const long chart_id)
  {
   CChartObj *chart=this.GetChart(chart_id);
   if(chart==NULL)
      return NULL;
   CArrayObj *list=chart.GetList();
   return(list!=NULL ? list.At(list.Total()-1) : NULL);
  }
//+------------------------------------------------------------------+
```

The method receives the ID of the chart whose last added window we want to get.

Get the chart object by the ID passed to the method and retrieve the list of its windows.

The last added window is always located at the end of the list — return the object at the very end of the list of chart windows or NULL if failed to do that.

**The method returning the last indicator added to the specified window of a specified chart:**

```
//+------------------------------------------------------------------+
//| Return the last indicator added                                  |
//| to the specified window of the specified chart                   |
//+------------------------------------------------------------------+
CWndInd* CChartObjCollection::GetLastAddedIndicator(const long chart_id,const int win_num)
  {
   CChartObj *chart=this.GetChart(chart_id);
   return(chart!=NULL ? chart.GetLastAddedIndicator(win_num) : NULL);
  }
//+------------------------------------------------------------------+
```

The method receives the chart ID and the number of the subwindow whose last added indicator we want to get.

Get the chart object by the ID passed to the method and use the GetLastAddedIndicator() chart object method to return the pointer to the last indicator added to the specified chart window. If failed, return NULL.

**The method returning the indicator by index from the specified window of the specified chart:**

```
//+------------------------------------------------------------------+
//| Return the indicator by index                                    |
//| from the specified window of the specified chart                 |
//+------------------------------------------------------------------+
CWndInd* CChartObjCollection::GetIndicator(const long chart_id,const int win_num,const int ind_index)
  {
   CChartObj *chart=this.GetChart(chart_id);
   return(chart!=NULL ? chart.GetIndicator(win_num,ind_index) : NULL);
  }
//+------------------------------------------------------------------+
```

The method receives the chart ID, subwindow index and indicator index we want to get.

Get the chart object by the ID passed to the method and use the GetIndicator() chart object method to return the pointer to the indicator from the specified chart window. If failed, return NULL.

**In the method, which finds and removes a chart object not present in the terminal from the list**, add the code block placing the detected chart object to the list of removed charts:

```
//+-----------------------------------------------------------------------------+
//|Find a chart object not present in the terminal and remove it from the list  |
//+-----------------------------------------------------------------------------+
void CChartObjCollection::FindAndDeleteExcessChartObj(void)
  {
   for(int i=this.m_list.Total()-1;i>WRONG_VALUE;i--)
     {
      CChartObj *chart=this.m_list.At(i);
      if(chart==NULL)
         continue;
      if(!this.IsPresentChart(chart.ID()))
        {
         chart=this.m_list.Detach(i);
         if(chart!=NULL)
           {
            if(!this.m_list_del.Add(chart))
               this.m_list.Delete(i);
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

Here, if the chart object obtained from the list is not present in the client terminal, use [the Detach() standard library method](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjdetach) to remove the object from the list and add it to the list of removed charts. If failed to place the removed object to the new list, delete it to avoid a memory leak.

**The method creating and sending a chart event to the control program chart:**

```
//+------------------------------------------------------------------+
//| Create and send a chart event                                    |
//| to the control program chart                                     |
//+------------------------------------------------------------------+
void CChartObjCollection::SendEvent(ENUM_CHART_OBJ_EVENT event)
  {
   //--- If a chart is added
   if(event==CHART_OBJ_EVENT_CHART_OPEN)
     {
      //--- Get the last chart object added to the list
      CChartObj *chart=this.GetLastAddedChart();
      if(chart==NULL)
         return;
      //--- Send the CHART_OBJ_EVENT_CHART_OPEN event to the control program chart
      //--- pass the chart ID to lparam,
      //--- pass the chart timeframe to dparam,
      //--- pass the chart symbol to sparam
      ::EventChartCustom(this.m_chart_id_main,(ushort)event,chart.ID(),chart.Timeframe(),chart.Symbol());
     }
   //--- If a chart is removed
   else if(event==CHART_OBJ_EVENT_CHART_CLOSE)
     {
      //--- Get the last chart object added to the list of removed charts
      CChartObj *chart=this.GetLastDeletedChart();
      if(chart==NULL)
         return;
      //--- Send the CHART_OBJ_EVENT_CHART_CLOSE event to the control program chart
      //--- pass the chart ID to lparam,
      //--- pass the chart timeframe to dparam,
      //--- pass the chart symbol to sparam
      ::EventChartCustom(this.m_chart_id_main,(ushort)event,chart.ID(),chart.Timeframe(),chart.Symbol());
     }
  }
//+------------------------------------------------------------------+
```

The method logic is described in the listing in detail. I believe, it requires no explanations.

### Track chart events

I have created the functionality for tracking some chart events. Now we need to provide the control program with access to it. To achieve this, I will use the **CEngine** library main class. It should feature the new methods providing access to the chart object collection class methods I have implemented in the current article. Since we already have [the functionality updating the chart object collection list](https://www.mql5.com/en/articles/9293#node03) described in the previous article, all is ready for tracking events in the control program. We only need to implement handling incoming events (namely, events arriving from the chart object collection class) from the library in the test EA.

The library is not yet able to track changes in all chart properties, their windows and indicators. However, since all objects are already descendants [of the extended class of the base object of all library objects](https://www.mql5.com/en/articles/7663#node02) (which automatically endows its descendants with event functionality), then all we have to do is add the methods for managing object properties. I will leave it for the next article. Now, let's connect the new functionality with the "outside world" and test all I have implemented in the current article.

In the \\MQL5\\Include\\DoEasy\ **Engine.mqh** file of the library main object class, add the methods of accessing the new methods of the chart object collection:

```
//--- Return the list of chart objects by (1) symbol and (2) timeframe
   CArrayObj           *GetListCharts(const string symbol)                             { return this.m_charts.GetChartsList(symbol);         }
   CArrayObj           *GetListCharts(const ENUM_TIMEFRAMES timeframe)                 { return this.m_charts.GetChartsList(timeframe);      }
//--- Return the list of removed (1) chart objects, (2) chart windows, (3) indicators in the chart window and (4) changed indicators in the chart window
   CArrayObj           *GetListChartsClosed(void)                                      { return this.m_charts.GetListDeletedCharts();        }
   CArrayObj           *GetListChartWindowsDeleted(void)                               { return this.m_charts.GetListDeletedWindows();       }
   CArrayObj           *GetListChartWindowsIndicatorsDeleted(void)                     { return this.m_charts.GetListDeletedIndicators();    }
   CArrayObj           *GetListChartWindowsIndicatorsChanged(void)                     { return this.m_charts.GetListChangedIndicators();    }

//--- Return (1) the specified chart object and (2) the chart object with the program
   CChartObj           *ChartGetChartObj(const long chart_id)                          { return this.m_charts.GetChart(chart_id);            }
   CChartObj           *ChartGetMainChart(void)                                        { return this.m_charts.GetChart(this.m_charts.GetMainChartID());}
//--- Reutrn the chart object of the last (1) open and (2) closed chart
   CChartObj           *ChartGetLastOpenedChart(void)                                  { return this.m_charts.GetLastAddedChart();           }
   CChartObj           *ChartGetLastClosedChart(void)                                  { return this.m_charts.GetLastDeletedChart();         }

//--- Return the object (1) of the last added window of the specified chart and (2) the last removed chart window
   CChartWnd           *ChartGetLastAddedChartWindow(const long chart_id)              { return this.m_charts.GetLastAddedChartWindow(chart_id);}
   CChartWnd           *ChartGetLastDeletedChartWindow(void)                           { return this.m_charts.GetLastDeletedChartWindow();   }

//--- Return (1) the last one added to the specified window of the specified chart, (2) the last one removed from the window and (3) the changed indicator
   CWndInd             *ChartGetLastAddedIndicator(const long id,const int win)        { return m_charts.GetLastAddedIndicator(id,win);      }
   CWndInd             *ChartGetLastDeletedIndicator(void)                             { return this.m_charts.GetLastDeletedIndicator();     }
   CWndInd             *ChartGetLastChangedIndicator(void)                             { return this.m_charts.GetLastChangedIndicator();     }
//--- Return the indicator by index from the specified window of the specified chart
   CWndInd             *ChartGetIndicator(const long chart_id,const int win_num,const int ind_index)
                          {
                           return m_charts.GetIndicator(chart_id,win_num,ind_index);
                          }

//--- Return the number of charts in the collection list
   int                  ChartsTotal(void)                                              { return this.m_charts.DataTotal();                   }
```

All newly added methods return the result of calling the appropriate chart object collection methods.

### Test

To perform the test, I will use the [EA from the previous article](https://www.mql5.com/en/articles/9293#node04) and save it in \\MQL5\\Experts\\TestDoEasy\ **Part71\** as **TestDoEasyPart71.mq5**.

All we need to do is add handling new event codes to the **OnDoEasyEvent()** library event handler.

There is no point in considering the full function code. It is large and requires splitting into separate handlers of events from different library objects. I will deal with this much later.

Now let's consider the code block to be added to the OnDoEasyEvent() EA function:

```
//--- Handling timeseries events
   else if(idx>SERIES_EVENTS_NO_EVENT && idx<SERIES_EVENTS_NEXT_CODE)
     {
      //--- "New bar" event
      if(idx==SERIES_EVENTS_NEW_BAR)
        {
         Print(TextByLanguage("Новый бар на ","New Bar on "),sparam," ",TimeframeDescription((ENUM_TIMEFRAMES)dparam),": ",TimeToString(lparam));
        }
     }

//--- Handle chart events
   else if(idx>CHART_OBJ_EVENT_NO_EVENT && idx<CHART_OBJ_EVENTS_NEXT_CODE)
     {
      //--- "New chart opening" event
      if(idx==CHART_OBJ_EVENT_CHART_OPEN)
        {
         //::EventChartCustom(this.m_chart_id_main,(ushort)event,chart.ID(),chart.Timeframe(),chart.Symbol());
         CChartObj *chart=engine.ChartGetLastOpenedChart();
         if(chart!=NULL)
           {
            string symbol=sparam;
            long chart_id=lparam;
            ENUM_TIMEFRAMES timeframe=(ENUM_TIMEFRAMES)dparam;
            string header=symbol+" "+TimeframeDescription(timeframe)+", ID "+(string)chart_id;
            Print(DFUN,CMessage::Text(MSG_CHART_COLLECTION_CHART_OPENED),": ",header);
           }
        }
      //--- "Chart closure" event
      if(idx==CHART_OBJ_EVENT_CHART_CLOSE)
        {
         //::EventChartCustom(this.m_chart_id_main,(ushort)event,chart.ID(),chart.Timeframe(),chart.Symbol());
         CChartObj *chart=engine.ChartGetLastClosedChart();
         if(chart!=NULL)
           {
            string symbol=sparam;
            long   chart_id=lparam;
            ENUM_TIMEFRAMES timeframe=(ENUM_TIMEFRAMES)dparam;
            string header=symbol+" "+TimeframeDescription(timeframe)+", ID "+(string)chart_id;
            Print(DFUN,CMessage::Text(MSG_CHART_COLLECTION_CHART_CLOSED),": ",header);
           }
        }
      //--- "Adding a new window on the chart" event
      if(idx==CHART_OBJ_EVENT_CHART_WND_ADD)
        {
         //::EventChartCustom(this.m_chart_id_main,(ushort)event,this.m_chart_id,wnd.WindowNum(),this.Symbol());
         ENUM_TIMEFRAMES timeframe=WRONG_VALUE;
         string ind_name="";
         string symbol=sparam;
         long   chart_id=lparam;
         int    win_num=(int)dparam;
         string header=symbol+" "+TimeframeDescription(timeframe)+", ID "+(string)chart_id+": ";

         CChartObj *chart=engine.ChartGetLastOpenedChart();
         if(chart!=NULL)
           {
            timeframe=chart.Timeframe();
            CChartWnd *wnd=engine.ChartGetLastAddedChartWindow(chart.ID());
            if(wnd!=NULL)
              {
               CWndInd *ind=wnd.GetLastAddedIndicator();
               if(ind!=NULL)
                  ind_name=ind.Name();
              }
           }
         Print(DFUN,header,CMessage::Text(MSG_CHART_OBJ_WINDOW_ADDED)," ",(string)win_num," ",ind_name);
        }
      //--- "Removing a window from the chart" event
      if(idx==CHART_OBJ_EVENT_CHART_WND_DEL)
        {
         //::EventChartCustom(this.m_chart_id_main,(ushort)event,this.m_chart_id,wnd.WindowNum(),this.Symbol());
         CChartWnd *wnd=engine.ChartGetLastDeletedChartWindow();
         ENUM_TIMEFRAMES timeframe=WRONG_VALUE;
         string symbol=sparam;
         long   chart_id=lparam;
         int    win_num=(int)dparam;
         string header=symbol+" "+TimeframeDescription(timeframe)+", ID "+(string)chart_id+": ";
         Print(DFUN,header,CMessage::Text(MSG_CHART_OBJ_WINDOW_REMOVED)," ",(string)win_num);
        }
      //--- "Adding a new indicator to the chart window" event
      if(idx==CHART_OBJ_EVENT_CHART_WND_IND_ADD)
        {
         //::EventChartCustom(this.m_chart_id_main,(ushort)event,this.m_chart_id,this.WindowNum(),ind.Name());
         ENUM_TIMEFRAMES timeframe=WRONG_VALUE;
         string ind_name=sparam;
         string symbol=NULL;
         long   chart_id=lparam;
         int    win_num=(int)dparam;
         string header=NULL;

         CWndInd *ind=engine.ChartGetLastAddedIndicator(chart_id,win_num);
         if(ind!=NULL)
           {
            CChartObj *chart=engine.ChartGetChartObj(chart_id);
            if(chart!=NULL)
              {
               symbol=chart.Symbol();
               timeframe=chart.Timeframe();
               CChartWnd *wnd=chart.GetWindowByNum(win_num);
               if(wnd!=NULL)
                  header=wnd.Header();
              }
           }
         Print(DFUN,symbol," ",TimeframeDescription(timeframe),", ID ",chart_id,", ",header,": ",CMessage::Text(MSG_CHART_OBJ_INDICATOR_ADDED)," ",ind_name);
        }
      //--- "Removing an indicator from the chart window" event
      if(idx==CHART_OBJ_EVENT_CHART_WND_IND_DEL)
        {
         //::EventChartCustom(this.m_chart_id_main,(ushort)event,this.m_chart_id,this.WindowNum(),ind.Name());
         ENUM_TIMEFRAMES timeframe=WRONG_VALUE;
         string ind_name=sparam;
         string symbol=NULL;
         long   chart_id=lparam;
         int    win_num=(int)dparam;
         string header=NULL;

         CWndInd *ind=engine.ChartGetLastDeletedIndicator();
         if(ind!=NULL)
           {
            CChartObj *chart=engine.ChartGetChartObj(chart_id);
            if(chart!=NULL)
              {
               symbol=chart.Symbol();
               timeframe=chart.Timeframe();
               CChartWnd *wnd=chart.GetWindowByNum(win_num);
               if(wnd!=NULL)
                  header=wnd.Header();
              }
           }
         Print(DFUN,symbol," ",TimeframeDescription(timeframe),", ID ",chart_id,", ",header,": ",CMessage::Text(MSG_CHART_OBJ_INDICATOR_REMOVED)," ",ind_name);
        }
      //--- "Changing indicator parameters in the chart window" event
      if(idx==CHART_OBJ_EVENT_CHART_WND_IND_CHANGE)
        {
         //::EventChartCustom(this.m_chart_id_main,(ushort)event,this.m_chart_id,this.WindowNum(),ind.Name());
         ENUM_TIMEFRAMES timeframe=WRONG_VALUE;
         string ind_name=sparam;
         string symbol=NULL;
         long   chart_id=lparam;
         int    win_num=(int)dparam;
         string header=NULL;

         CWndInd *ind=NULL;
         CWndInd *ind_changed=engine.ChartGetLastChangedIndicator();
         if(ind_changed!=NULL)
           {
            ind=engine.ChartGetIndicator(chart_id,win_num,ind_changed.Index());
            if(ind!=NULL)
              {
               CChartObj *chart=engine.ChartGetChartObj(chart_id);
               if(chart!=NULL)
                 {
                  symbol=chart.Symbol();
                  timeframe=chart.Timeframe();
                  CChartWnd *wnd=chart.GetWindowByNum(win_num);
                  if(wnd!=NULL)
                     header=wnd.Header();
                 }
              }
           }
         Print(DFUN,symbol," ",TimeframeDescription(timeframe),", ID ",chart_id,", ",header,": ",CMessage::Text(MSG_CHART_OBJ_INDICATOR_CHANGED)," ",ind_name," >>> ",ind.Name());
        }
     }

//--- Handling trading events
```

Each of the events arriving from the chart object collection features an example of sending that event from the library classes in the comments.

This example clearly shows what kind of data we get in each of the three handler parameters ( **lparam**, **dparam** and **sparam**). This data is used to search for the necessary objects in the library. It is also used as a basis when creating a common message to be shown in the journal. I will not implement another handling of events from the chart object collection in the test EA. This example is quite sufficient to understand how to handle incoming events for your own needs.

Compile the EA and launch it on a symbol chart.

**Open a new symbol chart** — get the following journal message from the OnDoEasyEvent() handler:

```
OnDoEasyEvent: Open chart: AUDNZD H4, ID 131733844391938634
```

**Add a new window of any oscillator to the open chart** — get the following journal message from the OnDoEasyEvent() handler:

```
OnDoEasyEvent: AUDNZD H1, ID 131733844391938634: Added subwindow 1 Momentum(14)
```

**Add any indicator drawn in the main window to the open chart** — get the following journal message from the OnDoEasyEvent() handler:

```
OnDoEasyEvent: AUDNZD H4, ID 131733844391938634, Main chart window: Added indicator AMA(14,2,30)
```

**Change the oscillator parameters** — get the following journal message from the OnDoEasyEvent() handler:

```
OnDoEasyEvent: AUDNZD H4, ID 131733844391938634, Chart subwindow 1: Changed indicator Momentum(14) >>> Momentum(20)
```

**Change the indicator parameters in the main window** — get the following journal message from the OnDoEasyEvent() handler:

```
OnDoEasyEvent: AUDNZD H4, ID 131733844391938634, Main chart window: Changed indicator AMA(14,2,30) >>> AMA(20,2,30)
```

**Remove the oscillator window** — get the following two journal messages from the OnDoEasyEvent() handler:

```
OnDoEasyEvent: AUDNZD H4, ID 131733844391938634: Removed indicator Momentum(20)
OnDoEasyEvent: AUDNZD H1, ID 131733844391938634: Removed subwindow 1
```

**Remove the indicator from the main window** — get the following journal message from the OnDoEasyEvent() handler:

```
OnDoEasyEvent: AUDNZD H4, ID 131733844391938634, Main chart window: Removed indicator AMA(20,2,30)
```

**Close the previously opened chart window** — get the following journal message from the OnDoEasyEvent() handler:

```
OnDoEasyEvent: Closed chart: AUDNZD H4, ID 131733844391938634
```

As we can see, all events are handled correctly and sent to the control program.

### What's next?

In the next article, I will implement auto tracking of changes and managing the properties of all chart objects.

All files of the current version of the library are attached below together with the test EA file for MQL5 for you to test and download.

Leave your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/9360#node00)

**\*Previous articles within the series:**

[Other classes in DoEasy library (Part 67): Chart object class](https://www.mql5.com/en/articles/9213)

[Other classes in DoEasy library (Part 68): Chart window object class and indicator object classes in the chart window](https://www.mql5.com/en/articles/9236)

[Other classes in DoEasy library (Part 69): Chart object collection class](https://www.mql5.com/en/articles/9260)

[Other classes in DoEasy library (Part 70): Expanding functionality and auto updating the chart object collection](https://www.mql5.com/en/articles/9293)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/9360](https://www.mql5.com/ru/articles/9360)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/9360.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/9360/mql5.zip "Download MQL5.zip")(3962.05 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/370585)**

![Tips from a professional programmer (Part II): Storing and exchanging parameters between an Expert Advisor, scripts and external programs](https://c.mql5.com/2/42/tipstricks__1.png)[Tips from a professional programmer (Part II): Storing and exchanging parameters between an Expert Advisor, scripts and external programs](https://www.mql5.com/en/articles/9327)

These are some tips from a professional programmer about methods, techniques and auxiliary tools which can make programming easier. We will discuss parameters which can be restored after terminal restart (shutdown). All examples are real working code segments from my Cayman project.

![Swaps (Part I): Locking and Synthetic Positions](https://c.mql5.com/2/42/33201.png)[Swaps (Part I): Locking and Synthetic Positions](https://www.mql5.com/en/articles/9198)

In this article I will try to expand the classic concept of swap trading methods. I will explain why I have come to the conclusion that this concept deserves special attention and is absolutely recommended for study.

![Other classes in DoEasy library (Part 72): Tracking and recording chart object parameters in the collection](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__10.png)[Other classes in DoEasy library (Part 72): Tracking and recording chart object parameters in the collection](https://www.mql5.com/en/articles/9385)

In this article, I will complete working with chart object classes and their collection. I will also implement auto tracking of changes in chart properties and their windows, as well as saving new parameters to the object properties. Such a revision allows the future implementation of an event functionality for the entire chart collection.

![Other classes in DoEasy library (Part 70): Expanding functionality and auto updating the chart object collection](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__8.png)[Other classes in DoEasy library (Part 70): Expanding functionality and auto updating the chart object collection](https://www.mql5.com/en/articles/9293)

In this article, I will expand the functionality of chart objects and arrange navigation through charts, creation of screenshots, as well as saving and applying templates to charts. Also, I will implement auto update of the collection of chart objects, their windows and indicators within them.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/9360&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083403290508532500)

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