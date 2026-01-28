---
title: Timeseries in DoEasy library (part 47): Multi-period multi-symbol standard indicators
url: https://www.mql5.com/en/articles/8207
categories: Trading Systems, Indicators
relevance_score: 3
scraped_at: 2026-01-23T19:35:11.550683
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=kyxgujwbbftsbsdjcrccimfjlivpswki&ssn=1769186105940125189&ssn_dr=4&ssn_sr=0&fv_date=1769186105&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8207&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Timeseries%20in%20DoEasy%20library%20(part%2047)%3A%20Multi-period%20multi-symbol%20standard%20indicators%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691861092278071&fz_uniq=5070403171827782968&sv=2552)

MetaTrader 5 / Trading systems


### Contents

- [Concept](https://www.mql5.com/en/articles/8207#node01)
- [Improving library classes](https://www.mql5.com/en/articles/8207#node02)
- [Methods of working with standard indicators](https://www.mql5.com/en/articles/8207#node03)
- [Test](https://www.mql5.com/en/articles/8207#node04)
- [What's next?](https://www.mql5.com/en/articles/8207#node05)


### Concept

I believe, everyone knows about standard indicators from the conventional terminal delivery. These indicators use the current symbol/period chart to display data for the same symbol/period.

The thing I will start implementing in this article is the ability to create custom indicators displaying data on all standard indicators, calculated for the specified symbols/periods, on the current symbol/period chart.

In this article, I will consider creating the necessary methods for creating a custom indicator based on the standard AC ( [Accelerator Oscillator](https://www.metatrader5.com/en/mobile-trading/iphone/help/chart/indicators/bw_indicators/accelerator_oscillator "https://www.metatrader5.com/en/mobile-trading/iphone/help/chart/indicators/bw_indicators/accelerator_oscillator")) indicator. All methods are to be usable by other standard indicators as well, although with minor modifications - I will implement them in the following articles.

Let's add new properties for the buffer object to create and identify buffer objects for working with standard indicator data:

- Identifier of multiple buffers of one indicator allows identifying and selecting all buffer objects belonging to a single standard indicator using these buffers. One custom indicator may apply several identical standard indicators with different parameters (when creating a complex custom indicator based on several standard ones). This identifier allows defining each of the applied buffer objects by its belonging to the standard indicator.
- Handle of an indicator using a buffer — each buffer object used to calculate the standard indicator is to feature the handle of the created standard indicator for working with it from any buffer object belonging to this indicator.
- Type of an indicator using a buffer — indicator type from the [ENUM\_INDICATOR indicator type enumeration](https://www.mql5.com/en/docs/constants/indicatorconstants/enum_indicator) is specified here. This also allows defining and selecting buffer objects by their belonging to the standard indicator type.

- Name of an indicator using a buffer — name of a standard indicator applying a buffer object to display its description is to be stored here.


In addition to creating a database for working with standard indicator data, I will slightly improve the "New bar" object and timeseries classes for tracking skipped history bars and sending the "Skipped bars" event into the program.

In case of connection loss, enabling/disabling sleep mode and other abnormal events that require time to recover, we can see that some history bars are skipped in the library database after the program resumes its operation. Let's create the methods tracking the number of skipped bars and sending the "Skipped bars" event to the program so that users are able to handle such a situation in their programs.

### Improving library classes

First, let's add the data for displaying messages to \\MQL5\\Include\\DoEasy\ **Datas.mqh**.

Add new message IDs:

```
   MSG_LIB_TEXT_BUFFER_TEXT_INDEX_NEXT_PLOT,          // Index of the next drawn buffer
   MSG_LIB_TEXT_BUFFER_TEXT_ID,                       // Indicator buffer ID
   MSG_LIB_TEXT_BUFFER_TEXT_IND_HANDLE,               // Handle of an indicator using a buffer
   MSG_LIB_TEXT_BUFFER_TEXT_IND_TYPE,                 // Type of an indicator using a buffer
   MSG_LIB_TEXT_BUFFER_TEXT_TIMEFRAME,                // Buffer (timeframe) data period
   MSG_LIB_TEXT_BUFFER_TEXT_STATUS,                   // Buffer status
   MSG_LIB_TEXT_BUFFER_TEXT_TYPE,                     // Buffer type
   MSG_LIB_TEXT_BUFFER_TEXT_ACTIVE,                   // Active
   MSG_LIB_TEXT_BUFFER_TEXT_ARROW_CODE,               // Arrow code
   MSG_LIB_TEXT_BUFFER_TEXT_ARROW_SHIFT,              // The vertical shift of the arrows
   MSG_LIB_TEXT_BUFFER_TEXT_DRAW_BEGIN,               // The number of initial bars that are not drawn and values in DataWindow
   MSG_LIB_TEXT_BUFFER_TEXT_DRAW_TYPE,                // Graphical construction type
   MSG_LIB_TEXT_BUFFER_TEXT_SHOW_DATA,                // Display construction values in DataWindow
   MSG_LIB_TEXT_BUFFER_TEXT_SHIFT,                    // Indicator graphical construction shift by time axis in bars
   MSG_LIB_TEXT_BUFFER_TEXT_LINE_STYLE,               // Line style
   MSG_LIB_TEXT_BUFFER_TEXT_LINE_WIDTH,               // Line width
   MSG_LIB_TEXT_BUFFER_TEXT_ARROW_SIZE,               // Arrow size
   MSG_LIB_TEXT_BUFFER_TEXT_COLOR_NUM,                // Number of colors
   MSG_LIB_TEXT_BUFFER_TEXT_COLOR,                    // Drawing color
   MSG_LIB_TEXT_BUFFER_TEXT_EMPTY_VALUE,              // Empty value for plotting where nothing will be drawn
   MSG_LIB_TEXT_BUFFER_TEXT_SYMBOL,                   // Buffer symbol
   MSG_LIB_TEXT_BUFFER_TEXT_LABEL,                    // Name of the graphical indicator series displayed in DataWindow
   MSG_LIB_TEXT_BUFFER_TEXT_IND_NAME,                 // Name of an indicator using a buffer

```

and text messages corresponding to newly added IDs:

```
   {"Индекс следующего по счёту рисуемого буфера","Index of the next drawable buffer"},
   {"Идентификатор буферов индикатора","Indicator Buffer Id"},
   {"Хэндл индикатора, использующего буфер","Indicator handle that uses buffer"},
   {"Тип индикатора, использующего буфер","Indicator type that uses buffer"},
   {"Период данных буфера (таймфрейм)","Buffer data Period (Timeframe)"},
   {"Статус буфера","Buffer status"},
   {"Тип буфера","Buffer type"},
   {"Активен","Active"},
   {"Код стрелки","Arrow code"},
   {"Смещение стрелок по вертикали","Vertical shift of arrows"},
   {"Количество начальных баров без отрисовки и значений в DataWindow","Number of initial bars without drawing and values in DataWindow"},
   {"Тип графического построения","Type of graphical construction"},
   {"Отображение значений построения в окне DataWindow","Display construction values in DataWindow"},
   {"Сдвиг графического построения индикатора по оси времени в барах","Shift of indicator plotting along time axis in bars"},
   {"Стиль линии отрисовки","Drawing line style "},
   {"Толщина линии отрисовки","Thickness of drawing line"},
   {"Размер значка стрелки","Arrow icon size"},
   {"Количество цветов","Number of colors"},
   {"Цвет отрисовки","Index of buffer containing drawing color"},
   {"Пустое значение для построения, для которого нет отрисовки","Empty value for plotting, for which there is no drawing"},
   {"Символ буфера","Buffer Symbol"},
   {"Имя индикаторной графической серии, отображаемое в окне DataWindow","Name of indicator graphical series to display in DataWindow"},
   {"Наименование индикатора, использующего буфер","Name of indicator that uses buffer"},
   {"Индикаторный буфер с типом графического построения","Indicator buffer with graphic plot type"},
   {"Неправильно указано количество буферов индикатора (#property indicator_buffers)","Number of indicator buffers incorrect (#property indicator_buffers)"},
   {"Достигнуто максимально возможное количество индикаторных буферов","Maximum number of indicator buffers reached"},
```

Set all the necessary additions for the current tasks in \\MQL5\\Include\\DoEasy\ **Defines.mqh**.

In the "Macro substitutions" section, change the name of the constant storing the value of default trading attempts to a more informative one:

```
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
//--- Describe the function with the error line number
#define DFUN_ERR_LINE                  (__FUNCTION__+(TerminalInfoString(TERMINAL_LANGUAGE)=="Russian" ? ", Page " : ", Line ")+(string)__LINE__+": ")
#define DFUN                           (__FUNCTION__+": ")        // "Function description"
#define COUNTRY_LANG                   ("Russian")                // Country language
#define END_TIME                       (D'31.12.3000 23:59:59')   // End date for account history data requests
#define TIMER_FREQUENCY                (16)                       // Minimal frequency of the library timer in milliseconds
#define TOTAL_TRADE_TRY                (5)                        // Default number of trading attempts
#define IND_COLORS_TOTAL               (64)                       // Maximum possible number of indicator buffer colors
#define IND_BUFFERS_MAX                (512)                      // Maximum possible number of indicator buffers
//--- Standard sounds
```

Previously, the constant was named TOTAL\_TRY, which was not informative. Since we may have other constants specifying the number of attempts, adding the affiliation of attempts to a certain action (here it is "TRADE" — affiliation with trading attempts) to the constant name is more informative. It relieves us of the necessity to change the name of the constant when adding new constants for other "numbers of attempts"

Add a new event to the enumeration of possible timeseries events:

```
//+------------------------------------------------------------------+
//| List of possible timeseries events                               |
//+------------------------------------------------------------------+
enum ENUM_SERIES_EVENT
  {
   SERIES_EVENTS_NO_EVENT = SYMBOL_EVENTS_NEXT_CODE,        // no event
   SERIES_EVENTS_NEW_BAR,                                   // "New bar" event
   SERIES_EVENTS_MISSING_BARS,                              // "Bars skipped" event
  };
#define SERIES_EVENTS_NEXT_CODE  (SERIES_EVENTS_MISSING_BARS+1)   // Code of the next event after the "Bars skipped" event
//+------------------------------------------------------------------+
```

Correspondingly, the code of the next event is now based on a new constant.

I have already mentioned adding new properties to the buffer object. Let's set them in the enumerations of buffer object integer and string properties:

```
//+------------------------------------------------------------------+
//| Buffer integer properties                                        |
//+------------------------------------------------------------------+
enum ENUM_BUFFER_PROP_INTEGER
  {
   BUFFER_PROP_INDEX_PLOT = 0,                              // Plotted buffer serial number
   BUFFER_PROP_STATUS,                                      // Buffer status (by drawing style) from the ENUM_BUFFER_STATUS enumeration
   BUFFER_PROP_TYPE,                                        // Buffer type (from the ENUM_BUFFER_TYPE enumeration)
   BUFFER_PROP_TIMEFRAME,                                   // Buffer period data (timeframe)
   BUFFER_PROP_ACTIVE,                                      // Buffer usage flag
   BUFFER_PROP_DRAW_TYPE,                                   // Graphical construction type (from the ENUM_DRAW_TYPE enumeration)
   BUFFER_PROP_ARROW_CODE,                                  // Arrow code for DRAW_ARROW style
   BUFFER_PROP_ARROW_SHIFT,                                 // The vertical shift of the arrows for DRAW_ARROW style
   BUFFER_PROP_LINE_STYLE,                                  // Line style
   BUFFER_PROP_LINE_WIDTH,                                  // Line width
   BUFFER_PROP_DRAW_BEGIN,                                  // The number of initial bars that are not drawn and values in DataWindow
   BUFFER_PROP_SHOW_DATA,                                   // Flag of displaying construction values in DataWindow
   BUFFER_PROP_SHIFT,                                       // Indicator graphical construction shift by time axis in bars
   BUFFER_PROP_COLOR_INDEXES,                               // Number of colors
   BUFFER_PROP_COLOR,                                       // Drawing color
   BUFFER_PROP_INDEX_BASE,                                  // Base data buffer index
   BUFFER_PROP_INDEX_NEXT_BASE,                             // Index of the array to be assigned as the next indicator buffer
   BUFFER_PROP_INDEX_NEXT_PLOT,                             // Index of the next drawn buffer
   BUFFER_PROP_ID,                                          // ID of multiple buffers of a single indicator
   BUFFER_PROP_IND_HANDLE,                                  // Handle of an indicator using a buffer
   BUFFER_PROP_IND_TYPE,                                    // Type of an indicator using a buffer
   BUFFER_PROP_NUM_DATAS,                                   // Number of data buffers
   BUFFER_PROP_INDEX_COLOR,                                 // Color buffer index
  };
#define BUFFER_PROP_INTEGER_TOTAL (23)                      // Total number of integer bar properties
#define BUFFER_PROP_INTEGER_SKIP  (2)                       // Number of buffer properties not used in sorting
//+------------------------------------------------------------------+
//| Buffer real properties                                           |
//+------------------------------------------------------------------+
enum ENUM_BUFFER_PROP_DOUBLE
  {
   BUFFER_PROP_EMPTY_VALUE = BUFFER_PROP_INTEGER_TOTAL,     // Empty value for plotting where nothing will be drawn
  };
#define BUFFER_PROP_DOUBLE_TOTAL  (1)                       // Total number of real buffer properties
#define BUFFER_PROP_DOUBLE_SKIP   (0)                       // Number of buffer properties not used in sorting
//+------------------------------------------------------------------+
//| Buffer string properties                                         |
//+------------------------------------------------------------------+
enum ENUM_BUFFER_PROP_STRING
  {
   BUFFER_PROP_SYMBOL = (BUFFER_PROP_INTEGER_TOTAL+BUFFER_PROP_DOUBLE_TOTAL), // Buffer symbol
   BUFFER_PROP_LABEL,                                       // Name of the graphical indicator series displayed in DataWindow
   BUFFER_PROP_IND_NAME,                                    // Name of an indicator using a buffer
  };
#define BUFFER_PROP_STRING_TOTAL  (3)                       // Total number of string buffer properties
//+------------------------------------------------------------------+
```

Increase the total number of integer properties from 20 to **23**, as well as the number of string properties from 2 to **3**.

Since we added new properties, we also need to add the ability to sort and select by these properties.

Add new types of sorting buffer objects to the enumeration of possible sorting criteria:

```
//+------------------------------------------------------------------+
//| Possible buffer sorting criteria                                 |
//+------------------------------------------------------------------+
#define FIRST_BUFFER_DBL_PROP          (BUFFER_PROP_INTEGER_TOTAL-BUFFER_PROP_INTEGER_SKIP)
#define FIRST_BUFFER_STR_PROP          (BUFFER_PROP_INTEGER_TOTAL-BUFFER_PROP_INTEGER_SKIP+BUFFER_PROP_DOUBLE_TOTAL-BUFFER_PROP_DOUBLE_SKIP)
enum ENUM_SORT_BUFFER_MODE
  {
//--- Sort by integer properties
   SORT_BY_BUFFER_INDEX_PLOT = 0,                           // Sort by the plotted buffer serial number
   SORT_BY_BUFFER_STATUS,                                   // Sort by buffer drawing style (status) from the ENUM_BUFFER_STATUS enumeration
   SORT_BY_BUFFER_TYPE,                                     // Sort by buffer type (from the ENUM_BUFFER_TYPE enumeration)
   SORT_BY_BUFFER_TIMEFRAME,                                // Sort by the buffer data period (timeframe)
   SORT_BY_BUFFER_ACTIVE,                                   // Sort by the buffer usage flag
   SORT_BY_BUFFER_DRAW_TYPE,                                // Sort by graphical construction type (from the ENUM_DRAW_TYPE enumeration)
   SORT_BY_BUFFER_ARROW_CODE,                               // Sort by the arrow code for DRAW_ARROW style
   SORT_BY_BUFFER_ARROW_SHIFT,                              // Sort by the vertical shift of the arrows for DRAW_ARROW style
   SORT_BY_BUFFER_LINE_STYLE,                               // Sort by the line style
   SORT_BY_BUFFER_LINE_WIDTH,                               // Sort by the line width
   SORT_BY_BUFFER_DRAW_BEGIN,                               // Sort by the number of initial bars that are not drawn and values in DataWindow
   SORT_BY_BUFFER_SHOW_DATA,                                // Sort by the flag of displaying construction values in DataWindow
   SORT_BY_BUFFER_SHIFT,                                    // Sort by the indicator graphical construction shift by time axis in bars
   SORT_BY_BUFFER_COLOR_INDEXES,                            // Sort by a number of attempts
   SORT_BY_BUFFER_COLOR,                                    // Sort by the drawing color
   SORT_BY_BUFFER_INDEX_BASE,                               // Sort by the basic data buffer index
   SORT_BY_BUFFER_INDEX_NEXT_BASE,                          // Sort by the index of the array to be assigned as the next indicator buffer
   SORT_BY_BUFFER_INDEX_NEXT_PLOT,                          // Sort by the index of the next drawn buffer
   SORT_BY_BUFFER_ID,                                       // Sort by ID of multiple buffers of a single indicator
   SORT_BY_BUFFER_IND_HANDLE,                               // Sort by handle of an indicator using a buffer
   SORT_BY_BUFFER_IND_TYPE,                                 // Sort by type of an indicator using a buffer
//--- Sort by real properties
   SORT_BY_BUFFER_EMPTY_VALUE = FIRST_BUFFER_DBL_PROP,      // Sort by the empty value for plotting where nothing will be drawn
//--- Sort by string properties
   SORT_BY_BUFFER_SYMBOL = FIRST_BUFFER_STR_PROP,           // Sort by the buffer symbol
   SORT_BY_BUFFER_LABEL,                                    // Sort by the name of the graphical indicator series displayed in DataWindow
   SORT_BY_BUFFER_IND_NAME,                                 // Sort by name of an indicator using a buffer
  };
//+------------------------------------------------------------------+
```

To detect skipped bars (for example, after a connection loss), we need to slightly improve the "New bar" object class in \\MQL5\\Include\\DoEasy\\Objects\\Series\ **NewBarObj.mqh**. All we need to do is add a count of the number of bars between the two "New Bar" events. The value exceeding 1 indicates history bars have been skipped or are not present on the server at all (this situation is not yet considered).

In the private class section, add four new class member variables for storing the time of the previous "New bar" event time for manual and automatic time management, as well as for storing the number of seconds and bars between the two "New bar" events

```
//+------------------------------------------------------------------+
//| "New bar" object class                                           |
//+------------------------------------------------------------------+
class CNewBarObj : public CBaseObj
  {
private:
   string            m_symbol;                                    // Symbol
   ENUM_TIMEFRAMES   m_timeframe;                                 // Timeframe
   datetime          m_new_bar_time;                              // New bar time for auto time management
   datetime          m_prev_time;                                 // Previous time for auto time management
   datetime          m_new_bar_time_manual;                       // New bar time for manual time management
   datetime          m_prev_time_manual;                          // Previous time for manual time management
   datetime          m_prev_new_bar_time;                         // Previous new bar time for auto time management
   datetime          m_prev_new_bar_time_manual;                  // Previous new bar time for manual time management
   long              m_seconds_between;                           // Number of seconds between two "New bar" events
   int               m_bars_between;                              // Number of bars between two "New bar" events
//--- Return the current bar data
   datetime          GetLastBarDate(const datetime time);
public:
```

In the public section of the class, rename methods for setting and returning the object timeframe (Period has been used previously, however using Timeframe for storing a timeframe is more informative) and add methods of returning values of newly declared variables:

```
public:
//--- Set (1) symbol and (2) timeframe
   void              SetSymbol(const string symbol)               { this.m_symbol=(symbol==NULL || symbol==""   ? ::Symbol() : symbol);                     }
   void              SetTimeframe(const ENUM_TIMEFRAMES timeframe){ this.m_timeframe=(timeframe==PERIOD_CURRENT ? (ENUM_TIMEFRAMES)::Period() : timeframe); }
//--- Save the new bar time during the manual time management
   void              SaveNewBarTime(const datetime time)          { this.m_prev_time_manual=this.GetLastBarDate(time);                                      }
//--- Return (1) symbol and (2) timeframe
   string            Symbol(void)                           const { return this.m_symbol;             }
   ENUM_TIMEFRAMES   Timeframe(void)                        const { return this.m_timeframe;          }
//--- Return (1) new bar time, (2) previous new bar time, number of (3) seconds, (4) number of bars between the two last events
   datetime          TimeNewBar(void)                       const { return this.m_new_bar_time;       }
   datetime          TimePrevNewBar(void)                   const { return this.m_prev_new_bar_time;  }
   long              SecondsBetweenNewBars(void)            const { return this.m_seconds_between;    }
   int               BarsBetweenNewBars(void)               const { return this.m_bars_between;       }
//--- Return the new bar opening flag during the time (1) auto, (2) manual management
   bool              IsNewBar(const datetime time);
   bool              IsNewBarManual(const datetime time);
//--- Constructors
                     CNewBarObj(void) : m_symbol(::Symbol()),
                                        m_timeframe((ENUM_TIMEFRAMES)::Period()),
                                        m_prev_time(0),m_new_bar_time(0),
                                        m_prev_time_manual(0),m_new_bar_time_manual(0) {}
                     CNewBarObj(const string symbol,const ENUM_TIMEFRAMES timeframe);
  };
//+------------------------------------------------------------------+
```

In the initialization list of the paramteric class constructor, set initializing values for the number of seconds and bars, while other new variables are initialized using zero in the constructor body:

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CNewBarObj::CNewBarObj(const string symbol,const ENUM_TIMEFRAMES timeframe) : m_symbol(symbol),
                                                                              m_timeframe(timeframe),
                                                                              m_seconds_between(0),
                                                                              m_bars_between(0)
  {
   this.m_prev_new_bar_time=this.m_prev_new_bar_time_manual=this.m_prev_time=this.m_prev_time_manual=this.m_new_bar_time=this.m_new_bar_time_manual=0;
  }
//+------------------------------------------------------------------+
```

In the method returning the flag of opening a new bar during the automatic time management, save the time of the previous new bar when a new bar has formed and calculate the number of seconds and bars between two "New bar" events:

```
//+------------------------------------------------------------------+
//| Return new bar opening flag                                      |
//+------------------------------------------------------------------+
bool CNewBarObj::IsNewBar(const datetime time)
  {
//--- Get the current bar time
   datetime tm=this.GetLastBarDate(time);
   if(tm<=0)
      return false;
//--- If the previous and current time are equal to zero, this is the first launch
   if(this.m_prev_time+this.m_new_bar_time==0)
     {
      //--- set the new bar opening time,
      //--- set the previous bar time as the current one and return 'false'
      this.m_new_bar_time=this.m_prev_time=tm;
      return false;
     }
//--- If the previous time is less than the current bar open time, this is a new bar
   if(this.m_prev_time>0 && this.m_prev_time<tm)
     {
      this.m_prev_new_bar_time=this.m_prev_time;
      this.m_seconds_between=tm-m_prev_time;
      this.m_bars_between=int(this.m_seconds_between/::PeriodSeconds(this.m_timeframe));
      //--- set the new bar opening time,
      //--- set the previous time as the current one and return 'true'
      this.m_new_bar_time=this.m_prev_time=tm;
      return true;
     }
//--- in other cases, return 'false'
   return false;
  }
//+------------------------------------------------------------------+
```

In the method returning a new bar flag in case of the manual management, there is no need to calculate the data. The data on skipped bars is always calculated automatically. However, in this method, we will save the time of the previous "new bar" in case of the manual management and fix the error of assigning the new bar time (previously, the time was saved to the variable for auto time management):

```
//+------------------------------------------------------------------+
//| Return the new bar opening flag during the manual management     |
//+------------------------------------------------------------------+
bool CNewBarObj::IsNewBarManual(const datetime time)
  {
//--- Get the current bar time
   datetime tm=this.GetLastBarDate(time);
   if(tm<=0)
      return false;
//--- If the previous and current time are equal to zero, this is the first launch
   if(this.m_prev_time_manual+this.m_new_bar_time_manual==0)
     {
      //--- set the new bar opening time,
      //--- set the previous bar time as the current one and return 'false'
      this.m_new_bar_time_manual=this.m_prev_time_manual=tm;
      return false;
     }
//--- If the previous time is less than the current bar open time, this is a new bar
   if(this.m_prev_time_manual>0 && this.m_prev_time_manual<tm)
     {
      this.m_prev_new_bar_time_manual=this.m_prev_time_manual;
      //--- set the new bar opening time and return 'true'
      //--- Save the previous time as the current one from the program using the SaveNewBarTime() method
      //--- Till the previous time is forcibly set as the current one from the program,
      //--- the method returns the new bar flag allowing the completion of all the necessary actions on the new bar.
      this.m_new_bar_time_manual=tm;
      return true;
     }
//--- in other cases, return 'false'
   return false;
  }
//+------------------------------------------------------------------+
```

We can often see the library entries about errors of receiving history bars in the terminal journal. This happens because the library views the entire history even if a certain symbol has no historical data on a certain symbol. The appropriate entry is displayed and the system moves on to the next history bar. This is done for the ability to debug library methods when working with timeseries. I will remove these entries where it is definitely not necessary to view the errors of obtaining historical data. To do this, \\MQL5\\Include\\DoEasy\\Objects\\Series\ **Bar.mqh** of the Bar object class should receive yet another constructor with no parameters:

```
//+------------------------------------------------------------------+
//| Bar class                                                        |
//+------------------------------------------------------------------+
class CBar : public CBaseObj
  {
private:
   MqlDateTime       m_dt_struct;                                 // Date structure
   int               m_digits;                                    // Symbol's digits value
   string            m_period_description;                        // Timeframe string description
   long              m_long_prop[BAR_PROP_INTEGER_TOTAL];         // Integer properties
   double            m_double_prop[BAR_PROP_DOUBLE_TOTAL];        // Real properties
   string            m_string_prop[BAR_PROP_STRING_TOTAL];        // String properties

//--- Return the index of the array the bar's (1) double and (2) string properties are located at
   int               IndexProp(ENUM_BAR_PROP_DOUBLE property)     const { return(int)property-BAR_PROP_INTEGER_TOTAL;                        }
   int               IndexProp(ENUM_BAR_PROP_STRING property)     const { return(int)property-BAR_PROP_INTEGER_TOTAL-BAR_PROP_DOUBLE_TOTAL;  }

//--- Return the bar type (bullish/bearish/zero)
   ENUM_BAR_BODY_TYPE BodyType(void)                              const;
//--- Calculate and return the size of (1) candle, (2) candle body,
//--- (3) upper, (4) lower candle wick,
//--- (5) candle body top and (6) bottom
   double            CandleSize(void)                             const { return(this.High()-this.Low());                                    }
   double            BodySize(void)                               const { return(this.BodyHigh()-this.BodyLow());                            }
   double            ShadowUpSize(void)                           const { return(this.High()-this.BodyHigh());                               }
   double            ShadowDownSize(void)                         const { return(this.BodyLow()-this.Low());                                 }
   double            BodyHigh(void)                               const { return ::fmax(this.Close(),this.Open());                           }
   double            BodyLow(void)                                const { return ::fmin(this.Close(),this.Open());                           }

//--- Return the (1) year and (2) month the bar belongs to, (3) week day,
//--- (4) bar serial number in a year, (5) day, (6) hour, (7) minute,
   int               TimeYear(void)                               const { return this.m_dt_struct.year;                                      }
   int               TimeMonth(void)                              const { return this.m_dt_struct.mon;                                       }
   int               TimeDayOfWeek(void)                          const { return this.m_dt_struct.day_of_week;                               }
   int               TimeDayOfYear(void)                          const { return this.m_dt_struct.day_of_year;                               }
   int               TimeDay(void)                                const { return this.m_dt_struct.day;                                       }
   int               TimeHour(void)                               const { return this.m_dt_struct.hour;                                      }
   int               TimeMinute(void)                             const { return this.m_dt_struct.min;                                       }

public:
//--- Set bar's (1) integer, (2) real and (3) string properties
   void              SetProperty(ENUM_BAR_PROP_INTEGER property,long value) { this.m_long_prop[property]=value;                              }
   void              SetProperty(ENUM_BAR_PROP_DOUBLE property,double value){ this.m_double_prop[this.IndexProp(property)]=value;            }
   void              SetProperty(ENUM_BAR_PROP_STRING property,string value){ this.m_string_prop[this.IndexProp(property)]=value;            }
//--- Return (1) integer, (2) real and (3) string bar properties from the properties array
   long              GetProperty(ENUM_BAR_PROP_INTEGER property)  const { return this.m_long_prop[property];                                 }
   double            GetProperty(ENUM_BAR_PROP_DOUBLE property)   const { return this.m_double_prop[this.IndexProp(property)];               }
   string            GetProperty(ENUM_BAR_PROP_STRING property)   const { return this.m_string_prop[this.IndexProp(property)];               }

//--- Return the flag of the bar supporting the property
   virtual bool      SupportProperty(ENUM_BAR_PROP_INTEGER property)    { return true; }
   virtual bool      SupportProperty(ENUM_BAR_PROP_DOUBLE property)     { return true; }
   virtual bool      SupportProperty(ENUM_BAR_PROP_STRING property)     { return true; }
//--- Return itself
   CBar             *GetObject(void)                                    { return &this;}
//--- Set (1) bar symbol, timeframe and time, (2) bar object parameters
   void              SetSymbolPeriod(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time);
   void              SetProperties(const MqlRates &rates);

//--- Compare CBar objects by all possible properties (for sorting the lists by a specified bar object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CBar objects by all properties (to search for equal bar objects)
   bool              IsEqual(CBar* compared_bar) const;
//--- Constructors
                     CBar(){;}
                     CBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time,const string source);
                     CBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const MqlRates &rates);

//+------------------------------------------------------------------+
```

When creating timeseries lists by symbols, we will use the constructor for creating a new bar object belonging to the specified symbol timeseries. Previously, parametric constructors tried to retrieve the necessary newly created bar object data in history on their own, and the debugging entry was sent to the journal in case of an error while retrieving history from the constructor. A simple constructor with no parameters will create an empty bar object that you need to fill with data after it has been successfully created. This will happen in the CSeriesDE class methods.

Let's consider the changes that need to be made to the class listing in \\MQL5\\Include\\DoEasy\\Objects\\Series\ **SeriesDE.mqh**.

In the public section of the class, add the method returning the pointer to the "New bar" class object belonging to the class timeseries:

```
//+------------------------------------------------------------------+
//| Timeseries class                                                 |
//+------------------------------------------------------------------+
class CSeriesDE : public CBaseObj
  {
private:
   ENUM_TIMEFRAMES   m_timeframe;                                       // Timeframe
   string            m_symbol;                                          // Symbol
   string            m_period_description;                              // Timeframe string description
   datetime          m_firstdate;                                       // The very first date by a period symbol at the moment
   datetime          m_lastbar_date;                                    // Time of opening the last bar by period symbol
   uint              m_amount;                                          // Amount of applied timeseries data
   uint              m_required;                                        // Required amount of applied timeseries data
   uint              m_bars;                                            // Number of bars in history by symbol and timeframe
   bool              m_sync;                                            // Synchronized data flag
   CArrayObj         m_list_series;                                     // Timeseries list
   CNewBarObj        m_new_bar_obj;                                     // "New bar" object
//--- Set the very first date by a period symbol at the moment and the new time of opening the last bar by a period symbol
   void              SetServerDate(void)
                       {
                        this.m_firstdate=(datetime)::SeriesInfoInteger(this.m_symbol,this.m_timeframe,SERIES_FIRSTDATE);
                        this.m_lastbar_date=(datetime)::SeriesInfoInteger(this.m_symbol,this.m_timeframe,SERIES_LASTBAR_DATE);
                       }

public:
//--- Return (1) itself, (2) timeseries list, (3) timeseries "New bar" object
   CSeriesDE        *GetObject(void)                                    { return &this;               }
   CArrayObj        *GetList(void)                                      { return &m_list_series;      }
   CNewBarObj       *GetNewBarObj(void)                                 { return &this.m_new_bar_obj; }

//--- Return the list of bars by selected (1) double, (2) integer and (3) string property fitting a compared condition
```

Since now we have two timeseries events ("New bar" and "Bars skipped"), the method of creating and sending the timeseries event to the control program chart should be improved. In the method declaration, add the input parameter, in which we are going to pass the timeseries event to be created and sent:

```
//--- Create and send the timeseries event to the control program chart
   void              SendEvent(ENUM_SERIES_EVENT event);
```

Improve the method located outside the class body:

```
//+------------------------------------------------------------------+
//| Create and send the timeseries event                             |
//| to the control program chart                                     |
//+------------------------------------------------------------------+
void CSeriesDE::SendEvent(ENUM_SERIES_EVENT event)
  {
   if(event==SERIES_EVENTS_NEW_BAR)
     {
      int index=CSelect::FindBarMax(this.GetList(),BAR_PROP_TIME);
      CBar *bar=this.m_list_series.At(index);
      if(bar==NULL)
         return;
      ::EventChartCustom(this.m_chart_id_main,SERIES_EVENTS_NEW_BAR,bar.Time(),this.Timeframe(),this.Symbol());
     }
   else if(event==SERIES_EVENTS_MISSING_BARS)
     {
      ::EventChartCustom(this.m_chart_id_main,SERIES_EVENTS_MISSING_BARS,this.m_new_bar_obj.BarsBetweenNewBars(),this.Timeframe(),this.Symbol());
     }
  }
//+------------------------------------------------------------------+
```

Here, depending on the value passed to the method, we create the necessary event and send it to the control program chart. If the "Bars skipped" event is created, pass the number of skipped history bars in **lparam value** of the [EventChartCustom()](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom) function.

To get rid of the unnecessary history data receiving errors in the journal, we need to develop the method of returning the bar object by time in the timeseries:

```
//+------------------------------------------------------------------+
//| Return the bar object by time in the timeseries                  |
//+------------------------------------------------------------------+
CBar *CSeriesDE::GetBar(const datetime time)
  {
   CBar *obj=new CBar();
   if(obj==NULL)
      return NULL;
   obj.SetSymbolPeriod(this.m_symbol,this.m_timeframe,time);
   this.m_list_series.Sort(SORT_BY_BAR_TIME);
   int index=this.m_list_series.Search(obj);
   delete obj;
   return this.m_list_series.At(index);
  }
//+------------------------------------------------------------------+
```

Since now we have the constructor with no parameters in the CBar class, we will use creation of a new bar object using the constructor to search for the necessary bar.

Here we simply create a temporary empty bar object, as well as set the required symbol, timeframe and bar time.

The rest is simple: sort the list of bar objects by time and search the list of bar objects for the object whose data matches the one that we set for the created temporary bar object.

The [Search()](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjsearch) method returns the obtained object index in the list, while the [At()](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjat) method returns the pointer to the object by index. If no object is found, the index has the value of -1, while At() returns NULL.

New bar events, as well as bar skipping events, are now detected in the methods of updating all existing CTimeSeriesDE class timeseries in \\MQL5\\Include\\DoEasy\\Objects\\Series\ **TimeSeriesDE.mqh**.

Let's improve two methods of updating timeseries by adding code blocks for defining "Bars skipped" events:

```
//+------------------------------------------------------------------+
//| Update a specified timeseries list                               |
//+------------------------------------------------------------------+
void CTimeSeriesDE::Refresh(const ENUM_TIMEFRAMES timeframe,SDataCalculate &data_calculate)
  {
//--- Reset the timeseries event flag and clear the list of all timeseries events
   this.m_is_event=false;
   this.m_list_events.Clear();
//--- Get the timeseries from the list by its timeframe
   CSeriesDE *series_obj=this.m_list_series.At(this.IndexTimeframe(timeframe));
   if(series_obj==NULL || series_obj.DataTotal()==0 || !series_obj.IsAvailable())
      return;
//--- Update the timeseries list
   series_obj.Refresh(data_calculate);
   datetime time=
     (
      this.m_program==PROGRAM_INDICATOR && series_obj.Symbol()==::Symbol() && series_obj.Timeframe()==(ENUM_TIMEFRAMES)::Period() ?
      data_calculate.rates.time :
      series_obj.LastBarDate()
     );
//--- If the timeseries object features the New bar event
   if(series_obj.IsNewBar(time))
     {
      //--- send the "New bar" event to the control program chart
      series_obj.SendEvent(SERIES_EVENTS_NEW_BAR);
      //--- set the values of the first date in history on the server and in the terminal
      this.SetTerminalServerDate();
      //--- add the "New bar" event to the list of timeseries events
      //--- in case of successful addition, set the event flag for the timeseries
      if(this.EventAdd(SERIES_EVENTS_NEW_BAR,time,series_obj.Timeframe(),series_obj.Symbol()))
         this.m_is_event=true;

      //--- Check skipped bars
      int missing=series_obj.GetNewBarObj().BarsBetweenNewBars();
      if(missing>1)
        {
         //--- send the "Bars skipped" event to the control program chart
         series_obj.SendEvent(SERIES_EVENTS_MISSING_BARS);
         //--- add the "Bars skipped" event to the list of timeseries events
         this.EventAdd(SERIES_EVENTS_MISSING_BARS,missing,series_obj.Timeframe(),series_obj.Symbol());
        }
     }
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Update all timeseries lists                                      |
//+------------------------------------------------------------------+
void CTimeSeriesDE::RefreshAll(SDataCalculate &data_calculate)
  {
//--- Reset the flags indicating the necessity to set the first date in history on the server and in the terminal
//--- and the timeseries event flag, and clear the list of all timeseries events
   bool upd=false;
   this.m_is_event=false;
   this.m_list_events.Clear();
//--- In the loop by the list of all used timeseries,
   int total=this.m_list_series.Total();
   for(int i=0;i<total;i++)
     {
      //--- get the next timeseries object by the loop index
      CSeriesDE *series_obj=this.m_list_series.At(i);
      if(series_obj==NULL || !series_obj.IsAvailable() || series_obj.DataTotal()==0)
         continue;
      //--- update the timeseries list
      series_obj.Refresh(data_calculate);
      datetime time=
        (
         this.m_program==PROGRAM_INDICATOR && series_obj.Symbol()==::Symbol() && series_obj.Timeframe()==(ENUM_TIMEFRAMES)::Period() ?
         data_calculate.rates.time :
         series_obj.LastBarDate()
        );
      //--- If the timeseries object features the New bar event
      if(series_obj.IsNewBar(time))
        {
         //--- send the "New bar" event to the control program chart,
         series_obj.SendEvent(SERIES_EVENTS_NEW_BAR);
         //--- set the flag indicating the necessity to set the first date in history on the server and in the terminal
         upd=true;
         //--- add the "New bar" event to the list of timeseries events
         //--- in case of successful addition, set the event flag for the timeseries
         if(this.EventAdd(SERIES_EVENTS_NEW_BAR,time,series_obj.Timeframe(),series_obj.Symbol()))
            this.m_is_event=true;

         //--- Check skipped bars
         int missing=series_obj.GetNewBarObj().BarsBetweenNewBars();
         if(missing>1)
           {
            //--- send the "Bars skipped" event to the control program chart
            series_obj.SendEvent(SERIES_EVENTS_MISSING_BARS);
            //--- add the "Bars skipped" event to the list of timeseries events
            this.EventAdd(SERIES_EVENTS_MISSING_BARS,missing,series_obj.Timeframe(),series_obj.Symbol());
           }
        }
     }
//--- if the flag indicating the necessity to set the first date in history on the server and in the terminal is enabled,
//--- set the values of the first date in history on the server and in the terminal
   if(upd)
      this.SetTerminalServerDate();
  }
//+------------------------------------------------------------------+
```

When defining the "New bar" event, we call the previously changed method for creating a new timeseries event, to which we pass the "new bar" event. If there are missing bars, create the appropriate event as well.

In the public section of the CTimeSeriesCollection collection class of all timeseries objects in \\MQL5\\Include\\DoEasy\\Collections\ **TimeSeriesCollection.mqh**, add declaration of the method for re-creating all timeseries:

```
//--- (1) Create, (2) re-create a specified timeseries of a specified symbol, (3) re-create all timeseries
   bool                    CreateSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,const int rates_total=0,const uint required=0);
   bool                    ReCreateSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,const int rates_total=0,const uint required=0);
   bool                    ReCreateSeriesAll(const int rates_total=0,const uint required=0);
//--- Return (1) an empty, (2) partially filled timeseries
```

Let's write its implementation outside the class body:

```
//+------------------------------------------------------------------+
//| Re-create all timeseries                                         |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::ReCreateSeriesAll(const int rates_total=0,const uint required=0)
  {
//--- In the loop by all symbol timeseries objects in the collection,
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      //--- get the next symbol timeseries object
      CTimeSeriesDE *timeseries=this.m_list.At(i);
      if(timeseries==NULL)
         continue;
      //--- Get the list of all symbol timeseries
      CArrayObj *list=timeseries.GetListSeries();
      if(list==NULL)
         continue;
      //--- In a loop by all symbol timeseries
      int total_series=list.Total();
      for(int j=0;j<total_series;j++)
        {
         //--- Get the next timeseries
         CSeriesDE *series=list.At(j);
         if(series==NULL)
            continue;
         //--- check timeseries synchronization and re-create it
         if(!series.SyncData(required,rates_total))
            return false;
         if(series.Create(required)==0)
            return false;
        }
     }
   return true;
  }
//+------------------------------------------------------------------+
```

The method simply recreates all available timeseries in the collection. So far, this method is not applied anywhere, but it can be useful in the future if it is necessary to re-create the existing timeseries collections. For example, it may be required when defining skipping a large number of bars when the program uses many symbols/periods. In this case, it is much easier to re-create all collection timeseries calling one method, rather than define the number of skipped bars in each timeseries and re-create each one separately. Moreover, this will happen only when restoring connection to the server or at a new bar.

I have completed all preparatory steps slightly improving handling timeseries and bars. It is time to start creating methods for working with standard indicators.

### Methods of working with standard indicators

First of all, let's improve [the abstract buffer object class](https://www.mql5.com/en/articles/7821) in \\MQL5\\Include\\DoEasy\\Objects\\Indicators\ **Buffer.mqh**.

In the public section of the class, add methods for settingand returning four new buffer object properties:

```
//--- Set (1) the arrow code, (2) vertical shift of arrows, (3) symbol, (4) timeframe, (5) buffer activity flag
//--- (6) drawing type, (7) number of initial bars without drawing, (8) flag of displaying construction values in DataWindow,
//--- (9) shift of the indicator graphical construction along the time axis, (10) line style, (11) line width,
//--- (12) total number of colors, (13) one drawing color, (14) color of drawing in the specified color index,
//--- (15) drawing colors from the color array, (16) empty value, (17) name of the graphical series displayed in DataWindow
   virtual void      SetArrowCode(const uchar code)                  { return;                                                               }
   virtual void      SetArrowShift(const int shift)                  { return;                                                               }
   void              SetSymbol(const string symbol)                  { this.SetProperty(BUFFER_PROP_SYMBOL,symbol);                          }
   void              SetTimeframe(const ENUM_TIMEFRAMES timeframe)   { this.SetProperty(BUFFER_PROP_TIMEFRAME,timeframe);                    }
   void              SetActive(const bool flag)                      { this.SetProperty(BUFFER_PROP_ACTIVE,flag);                            }
   void              SetDrawType(const ENUM_DRAW_TYPE draw_type);
   void              SetDrawBegin(const int value);
   void              SetShowData(const bool flag);
   void              SetShift(const int shift);
   void              SetStyle(const ENUM_LINE_STYLE style);
   void              SetWidth(const int width);
   void              SetColorNumbers(const int number);
   void              SetColor(const color colour);
   void              SetColor(const color colour,const uchar index);
   void              SetColors(const color &array_colors[]);
   void              SetEmptyValue(const double value);
   virtual void      SetLabel(const string label);
   void              SetID(const int id)                             { this.SetProperty(BUFFER_PROP_ID,id);                                  }
   void              SetIndicatorHandle(const int handle)            { this.SetProperty(BUFFER_PROP_IND_HANDLE,handle);                      }
   void              SetIndicatorType(const ENUM_INDICATOR type)     { this.SetProperty(BUFFER_PROP_IND_TYPE,type);                          }
   void              SetIndicatorName(const string name)             { this.SetProperty(BUFFER_PROP_IND_NAME,name);                          }

//--- Return (1) the serial number of the drawn buffer, (2) bound array index, (3) color buffer index,
//--- (4) index of the first free bound array, (5) index of the next drawn buffer, (6) buffer data period, (7) buffer status,
//--- (8) buffer type, (9) buffer usage flag, (10) arrow code, (11) arrow shift for DRAW_ARROW style,
//--- (12) number of initial bars that are not drawn and values in DataWindow, (13) graphical construction type,
//--- (14) flag of displaying construction values in DataWindow, (15) indicator graphical construction shift along the time axis,
//--- (16) drawing line style, (17) drawing line width, (18) number of colors, (19) drawing color, number of buffers for construction
//--- (20) set empty value, (21) buffer symbol, (22) name of the indicator graphical series displayed in DataWindow
   int               IndexPlot(void)                           const { return (int)this.GetProperty(BUFFER_PROP_INDEX_PLOT);                 }
   int               IndexBase(void)                           const { return (int)this.GetProperty(BUFFER_PROP_INDEX_BASE);                 }
   int               IndexColor(void)                          const { return (int)this.GetProperty(BUFFER_PROP_INDEX_COLOR);                }
   int               IndexNextBaseBuffer(void)                 const { return (int)this.GetProperty(BUFFER_PROP_INDEX_NEXT_BASE);            }
   int               IndexNextPlotBuffer(void)                 const { return (int)this.GetProperty(BUFFER_PROP_INDEX_NEXT_PLOT);            }
   ENUM_TIMEFRAMES   Timeframe(void)                           const { return (ENUM_TIMEFRAMES)this.GetProperty(BUFFER_PROP_TIMEFRAME);      }
   ENUM_BUFFER_STATUS Status(void)                             const { return (ENUM_BUFFER_STATUS)this.GetProperty(BUFFER_PROP_STATUS);      }
   ENUM_BUFFER_TYPE  TypeBuffer(void)                          const { return (ENUM_BUFFER_TYPE)this.GetProperty(BUFFER_PROP_TYPE);          }
   bool              IsActive(void)                            const { return (bool)this.GetProperty(BUFFER_PROP_ACTIVE);                    }
   uchar             ArrowCode(void)                           const { return (uchar)this.GetProperty(BUFFER_PROP_ARROW_CODE);               }
   int               ArrowShift(void)                          const { return (int)this.GetProperty(BUFFER_PROP_ARROW_SHIFT);                }
   int               DrawBegin(void)                           const { return (int)this.GetProperty(BUFFER_PROP_DRAW_BEGIN);                 }
   ENUM_DRAW_TYPE    DrawType(void)                            const { return (ENUM_DRAW_TYPE)this.GetProperty(BUFFER_PROP_DRAW_TYPE);       }
   bool              IsShowData(void)                          const { return (bool)this.GetProperty(BUFFER_PROP_SHOW_DATA);                 }
   int               Shift(void)                               const { return (int)this.GetProperty(BUFFER_PROP_SHIFT);                      }
   ENUM_LINE_STYLE   LineStyle(void)                           const { return (ENUM_LINE_STYLE)this.GetProperty(BUFFER_PROP_LINE_STYLE);     }
   int               LineWidth(void)                           const { return (int)this.GetProperty(BUFFER_PROP_LINE_WIDTH);                 }
   int               ColorsTotal(void)                         const { return (int)this.GetProperty(BUFFER_PROP_COLOR_INDEXES);              }
   color             Color(void)                               const { return (color)this.GetProperty(BUFFER_PROP_COLOR);                    }
   int               BuffersTotal(void)                        const { return (int)this.GetProperty(BUFFER_PROP_NUM_DATAS);                  }
   double            EmptyValue(void)                          const { return this.GetProperty(BUFFER_PROP_EMPTY_VALUE);                     }
   string            Symbol(void)                              const { return this.GetProperty(BUFFER_PROP_SYMBOL);                          }
   string            Label(void)                               const { return this.GetProperty(BUFFER_PROP_LABEL);                           }
   int               ID(void)                                  const { return (int)this.GetProperty(BUFFER_PROP_ID);                         }
   int               IndicatorHandle(void)                     const { return (int)this.GetProperty(BUFFER_PROP_IND_HANDLE);                 }
   ENUM_INDICATOR    IndicatorType(void)                       const { return (ENUM_INDICATOR)this.GetProperty(BUFFER_PROP_IND_TYPE);        }
   string            IndicatorName(void)                       const { return this.GetProperty(BUFFER_PROP_IND_NAME);                        }
   int               IndicatorBarsCalculated(void)             const { return ::BarsCalculated((int)this.GetProperty(BUFFER_PROP_IND_HANDLE));}
```

In the class constructor, set the default values to the new properties:

```
//+------------------------------------------------------------------+
//| Closed parametric constructor                                    |
//+------------------------------------------------------------------+
CBuffer::CBuffer(ENUM_BUFFER_STATUS buffer_status,
                 ENUM_BUFFER_TYPE buffer_type,
                 const uint index_plot,
                 const uint index_base_array,
                 const int num_datas,
                 const uchar total_arrays,
                 const int width,
                 const string label)
  {
   this.m_type=COLLECTION_BUFFERS_ID;
   this.m_act_state_trigger=true;
   this.m_total_arrays=total_arrays;
//--- Save integer properties
   this.m_long_prop[BUFFER_PROP_STATUS]                        = buffer_status;
   this.m_long_prop[BUFFER_PROP_TYPE]                          = buffer_type;
   this.m_long_prop[BUFFER_PROP_ID]                            = WRONG_VALUE;
   this.m_long_prop[BUFFER_PROP_IND_HANDLE]                    = INVALID_HANDLE;
   this.m_long_prop[BUFFER_PROP_IND_TYPE]                      = WRONG_VALUE;
   ENUM_DRAW_TYPE type=
     (
      !this.TypeBuffer() || !this.Status() ? DRAW_NONE      :
      this.Status()==BUFFER_STATUS_FILLING ? DRAW_FILLING   :
      ENUM_DRAW_TYPE(this.Status()+8)
     );
   this.m_long_prop[BUFFER_PROP_DRAW_TYPE]                     = type;
   this.m_long_prop[BUFFER_PROP_TIMEFRAME]                     = PERIOD_CURRENT;
   this.m_long_prop[BUFFER_PROP_ACTIVE]                        = true;
   this.m_long_prop[BUFFER_PROP_ARROW_CODE]                    = 0x9F;
   this.m_long_prop[BUFFER_PROP_ARROW_SHIFT]                   = 0;
   this.m_long_prop[BUFFER_PROP_DRAW_BEGIN]                    = 0;
   this.m_long_prop[BUFFER_PROP_SHOW_DATA]                     = (buffer_type>BUFFER_TYPE_CALCULATE ? true : false);
   this.m_long_prop[BUFFER_PROP_SHIFT]                         = 0;
   this.m_long_prop[BUFFER_PROP_LINE_STYLE]                    = STYLE_SOLID;
   this.m_long_prop[BUFFER_PROP_LINE_WIDTH]                    = width;
   this.m_long_prop[BUFFER_PROP_COLOR_INDEXES]                 = (this.Status()>BUFFER_STATUS_NONE ? (this.Status()!=BUFFER_STATUS_FILLING ? 1 : 2) : 0);
   this.m_long_prop[BUFFER_PROP_COLOR]                         = clrRed;
   this.m_long_prop[BUFFER_PROP_NUM_DATAS]                     = num_datas;
   this.m_long_prop[BUFFER_PROP_INDEX_PLOT]                    = index_plot;
   this.m_long_prop[BUFFER_PROP_INDEX_BASE]                    = index_base_array;
   this.m_long_prop[BUFFER_PROP_INDEX_COLOR]                   = this.GetProperty(BUFFER_PROP_INDEX_BASE)+
                                                                   (this.TypeBuffer()!=BUFFER_TYPE_CALCULATE ? this.GetProperty(BUFFER_PROP_NUM_DATAS) : 0);
   this.m_long_prop[BUFFER_PROP_INDEX_NEXT_BASE]               = index_base_array+this.m_total_arrays;
   this.m_long_prop[BUFFER_PROP_INDEX_NEXT_PLOT]               = (this.TypeBuffer()>BUFFER_TYPE_CALCULATE ? index_plot+1 : index_plot);

//--- Save real properties
   this.m_double_prop[this.IndexProp(BUFFER_PROP_EMPTY_VALUE)] = (this.TypeBuffer()>BUFFER_TYPE_CALCULATE ? EMPTY_VALUE : 0);
//--- Save string properties
   this.m_string_prop[this.IndexProp(BUFFER_PROP_SYMBOL)]      = ::Symbol();
   this.m_string_prop[this.IndexProp(BUFFER_PROP_LABEL)]       = (this.TypeBuffer()>BUFFER_TYPE_CALCULATE ? label : NULL);
   this.m_string_prop[this.IndexProp(BUFFER_PROP_IND_NAME)]    = NULL;

//--- If failed to change the size of the indicator buffer array, display the appropriate message indicating the string
```

Such values of these new properties will belong to buffer objects that do not work with standard indicators. If we create a buffer object belonging to a standard indicator, these parameters will be filled in by the library at the moment of its creation (to be implemented later).

Add displaying descriptions for new integer properties to the method returning the buffer integer property description:

```
//+------------------------------------------------------------------+
//| Return description of a buffer's integer property                |
//+------------------------------------------------------------------+
string CBuffer::GetPropertyDescription(ENUM_BUFFER_PROP_INTEGER property)
  {
   return
     (
      property==BUFFER_PROP_INDEX_PLOT    ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_INDEX_PLOT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_STATUS        ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_STATUS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetStatusDescription()
         )  :
      property==BUFFER_PROP_TYPE          ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_TYPE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetTypeBufferDescription()
         )  :
      property==BUFFER_PROP_TIMEFRAME     ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_TIMEFRAME)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetTimeframeDescription()
         )  :
      property==BUFFER_PROP_ACTIVE        ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_ACTIVE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetActiveDescription()
         )  :
      property==BUFFER_PROP_DRAW_TYPE     ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_DRAW_TYPE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetDrawTypeDescription()
         )  :
      property==BUFFER_PROP_ARROW_CODE    ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_ARROW_CODE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_ARROW_SHIFT   ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_ARROW_SHIFT)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_LINE_STYLE    ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_LINE_STYLE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetLineStyleDescription()
         )  :
      property==BUFFER_PROP_LINE_WIDTH    ?
         (this.Status()==BUFFER_STATUS_ARROW ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_ARROW_SIZE) :
          CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_LINE_WIDTH))+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_DRAW_BEGIN    ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_DRAW_BEGIN)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_SHOW_DATA     ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_SHOW_DATA)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetShowDataDescription()
         )  :
      property==BUFFER_PROP_SHIFT         ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_SHIFT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_COLOR_INDEXES ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_COLOR_NUM)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_INDEX_COLOR   ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_INDEX_COLOR)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_INDEX_BASE    ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_INDEX_BASE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_INDEX_NEXT_BASE ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_INDEX_NEXT_BASE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_INDEX_NEXT_PLOT ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_INDEX_NEXT_PLOT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_ID ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_ID)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_IND_HANDLE ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_IND_HANDLE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_IND_TYPE ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_IND_TYPE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_NUM_DATAS     ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_NUM_DATAS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_COLOR         ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_COLOR)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetColorsDescription()
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

Add displaying descriptions for a new string property to the method returning the buffer string property description:

```
//+------------------------------------------------------------------+
//| Return description of a buffer's string property                 |
//+------------------------------------------------------------------+
string CBuffer::GetPropertyDescription(ENUM_BUFFER_PROP_STRING property)
  {
   return
     (
      property==BUFFER_PROP_SYMBOL    ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_SYMBOL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.Symbol()
         )  :
      property==BUFFER_PROP_LABEL    ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_LABEL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.Label()==NULL || this.Label()=="" ? CMessage::Text(MSG_LIB_PROP_NOT_SET) : "\""+this.Label()+"\"")
         )  :
      property==BUFFER_PROP_IND_NAME   ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_IND_NAME)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.IndicatorName()==NULL || this.IndicatorName()=="" ? CMessage::Text(MSG_LIB_PROP_NOT_SET) : "\""+this.IndicatorName()+"\"")
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

Let's make changes in the methods of setting an empty value and the graphical series name. Previously, these values were not set for the calculated buffer. Let's make it so that the values are set only to the buffer object properties in case of a calculated buffer.

In case of a drawn buffer, the values should be set both to objectand buffer properties:

```
//+------------------------------------------------------------------+
//| Set the "empty" value for construction                           |
//| without drawing                                                  |
//+------------------------------------------------------------------+
void CBuffer::SetEmptyValue(const double value)
  {
   this.SetProperty(BUFFER_PROP_EMPTY_VALUE,value);
   if(this.TypeBuffer()!=BUFFER_TYPE_CALCULATE)
      ::PlotIndexSetDouble((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_EMPTY_VALUE,value);
  }
//+------------------------------------------------------------------+
//| Set the indicator graphical series name                          |
//+------------------------------------------------------------------+
void CBuffer::SetLabel(const string label)
  {
   this.SetProperty(BUFFER_PROP_LABEL,label);
   if(this.TypeBuffer()!=BUFFER_TYPE_CALCULATE)
      ::PlotIndexSetString((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LABEL,label);
  }
//+------------------------------------------------------------------+
```

Add control for index value less than zero to the methods returning values by the timeseries index:

```
//+------------------------------------------------------------------+
//| Return the value from the specified timeseries index             |
//| of the specified data buffer array                               |
//+------------------------------------------------------------------+
double CBuffer::GetDataBufferValue(const uint buffer_index,const int series_index) const
  {
   int correct_buff_index=this.GetCorrectIndexBuffer(buffer_index);
   int data_total=this.GetDataTotal(correct_buff_index);
   if(data_total==0 || series_index<0)
      return this.EmptyValue();
   int data_index=((int)series_index<data_total ? (int)series_index : data_total-1);
   return this.DataBuffer[correct_buff_index].Array[data_index];
  }
//+------------------------------------------------------------------+
//| Return the color index value from the specified timeseries index |
//| of the specified color buffer array                              |
//+------------------------------------------------------------------+
int CBuffer::GetColorBufferValueIndex(const int series_index) const
  {
   int data_total=this.GetDataTotal(0);
   if(data_total==0 || series_index<0)
      return WRONG_VALUE;
   int data_index=((int)series_index<data_total ? (int)series_index : data_total-1);
   return(this.ColorsTotal()==1 ? 0 : (int)this.ColorBufferArray[data_index]);
  }
//+------------------------------------------------------------------+
//| Return the color value from the specified timeseries index       |
//| of the specified color buffer array                              |
//+------------------------------------------------------------------+
color CBuffer::GetColorBufferValueColor(const int series_index) const
  {
   int data_total=this.GetDataTotal(0);
   if(data_total==0 || series_index<0)
      return clrNONE;
   int color_index=this.GetColorBufferValueIndex(series_index);
   return(color_index>WRONG_VALUE ? (color)this.ArrayColors[color_index] : clrNONE);
  }
//+------------------------------------------------------------------+
```

Thus, if a wrong index is passed to the method, exit from the method is performed while returning the "empty" value, which is different for each of the methods.

Now let's improve the calculated buffer object class in \\MQL5\\Include\\DoEasy\\Objects\\Indicators\ **BufferCalculate.mqh**.

The methods returning the flag of supporting real and string properties by the buffer object previously returned false — i.e. the calculated buffer did not support properties of this type. Let's make it support each of these properties. In the method returning the flag of supporting integer properties by the object, add new integer properties for supporting them using the calculated buffer object:

```
//+------------------------------------------------------------------+
//| Return 'true' if a buffer supports a passed                      |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CBufferCalculate::SupportProperty(ENUM_BUFFER_PROP_INTEGER property)
  {
   if(
      property==BUFFER_PROP_INDEX_PLOT       ||
      property==BUFFER_PROP_STATUS           ||
      property==BUFFER_PROP_TYPE             ||
      property==BUFFER_PROP_INDEX_BASE       ||
      property==BUFFER_PROP_ID               ||
      property==BUFFER_PROP_IND_HANDLE       ||
      property==BUFFER_PROP_IND_TYPE         ||
      property==BUFFER_PROP_INDEX_NEXT_BASE
     ) return true;
   return false;
  }
//+------------------------------------------------------------------+
//| Return 'true' if a buffer supports a passed                      |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CBufferCalculate::SupportProperty(ENUM_BUFFER_PROP_DOUBLE property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if a buffer supports a passed                      |
//| string property, otherwise return 'false'                        |
//+------------------------------------------------------------------+
bool CBufferCalculate::SupportProperty(ENUM_BUFFER_PROP_STRING property)
  {
   return true;
  }
//+------------------------------------------------------------------+
```

The object supporting the calculated buffer of all real and string properties is mostly a fast temporary solution for creating methods of working with buffers handling standard indicators. Later, I will remove some of them from the list of supported properties.

The entire handling of indicator buffers for standard indicators is arranged in the CBuffersCollection collection class of indicator buffers in \\MQL5\\Include\\DoEasy\\Collections\ **BuffersCollection.mqh**.

Today I am going to create and maintain multi-symbol multi-period indicator buffers of AC (Accelerator Oscillator) standard indicator. In the following articles, I will add the ability to create and work with other standard indicators based on the tested functionality.

All buffer objects working with standard indicators obtain the ID allowing us to find the necessary buffers and work with them.

In the public class section, declare the method returning the list of buffer objects with such an ID:

```
//+------------------------------------------------------------------+
//| Collection of indicator buffers                                  |
//+------------------------------------------------------------------+
class CBuffersCollection : public CObject
  {
private:
   CListObj                m_list;                       // Buffer object list
   CTimeSeriesCollection  *m_timeseries;                 // Pointer to the timeseries collection object

//--- Return the index of the (1) last, (2) next drawn and (3) basic buffer
   int                     GetIndexLastPlot(void);
   int                     GetIndexNextPlot(void);
   int                     GetIndexNextBase(void);
//--- Create a new buffer object and place it to the collection list
   bool                    CreateBuffer(ENUM_BUFFER_STATUS status);
//--- Get data of the necessary timeseries and bars for working with a single buffer bar, and return the number of bars
   int                     GetBarsData(CBuffer *buffer,const int series_index,int &index_bar_period);

public:
//--- Return (1) itself, (2) timeseries list, (3) indicator buffer list (featuring the ID of belonging to an indicator)
   CBuffersCollection     *GetObject(void)               { return &this;                                       }
   CArrayObj              *GetList(void)                 { return &this.m_list;                                }
   CArrayObj              *GetListBuffersWithID(void);
```

Let's write its implementation outside the class body:

```
//+------------------------------------------------------------------+
//| Return the list of indicator buffers                             |
//| (featuring the ID of belonging to an indicator)                  |
//+------------------------------------------------------------------+
CArrayObj *CBuffersCollection::GetListBuffersWithID(void)
  {
   CArrayObj *list=CSelect::ByBufferProperty(this.GetList(),BUFFER_PROP_ID,WRONG_VALUE,NO_EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
```

Here all is simple: using the CSelect class, get the list of buffer objects with the ID value not equal to -1 and return the pointer to the obtained list.

If the list is obtained successfully, it features all buffer objects having an ID not equal to -1. This means the list will contain all created buffer objects for working with standard indicators, including calculated and drawn ones for any standard indicator type.

To search for buffer objects belonging to a specific indicator, the list should be additionally sorted by standard indicator type, ID and buffer type.

Add declarations of methods for creating buffer objects handling standard indicators to the public class section:

```
//--- Create the new buffer (1) "Drawing with arrows", (2) "Line", (3) "Sections", (4) "Histogram from the zero line",
//--- (5) "Histogram on two indicator buffers", (6) "Zigzag", (7) "Color filling between two levels",
//--- (8) "Display as bars", (9) "Display as candles", calculated buffer
   bool                    CreateArrow(void)             { return this.CreateBuffer(BUFFER_STATUS_ARROW);      }
   bool                    CreateLine(void)              { return this.CreateBuffer(BUFFER_STATUS_LINE);       }
   bool                    CreateSection(void)           { return this.CreateBuffer(BUFFER_STATUS_SECTION);    }
   bool                    CreateHistogram(void)         { return this.CreateBuffer(BUFFER_STATUS_HISTOGRAM);  }
   bool                    CreateHistogram2(void)        { return this.CreateBuffer(BUFFER_STATUS_HISTOGRAM2); }
   bool                    CreateZigZag(void)            { return this.CreateBuffer(BUFFER_STATUS_ZIGZAG);     }
   bool                    CreateFilling(void)           { return this.CreateBuffer(BUFFER_STATUS_FILLING);    }
   bool                    CreateBars(void)              { return this.CreateBuffer(BUFFER_STATUS_BARS);       }
   bool                    CreateCandles(void)           { return this.CreateBuffer(BUFFER_STATUS_CANDLES);    }
   bool                    CreateCalculate(void)         { return this.CreateBuffer(BUFFER_STATUS_NONE);       }

//--- Create a multi-symbol multi-period indicator
   int                     CreateAC(const string symbol,const ENUM_TIMEFRAMES timeframe,const int id=WRONG_VALUE);
   int                     CreateAD(const string symbol,const ENUM_TIMEFRAMES timeframe,const ENUM_APPLIED_VOLUME applied_volume,const int id=WRONG_VALUE);
   int                     CreateADX(const string symbol,const ENUM_TIMEFRAMES timeframe,const int adx_period,const int id=WRONG_VALUE);
   int                     CreateADXWilder(const string symbol,const ENUM_TIMEFRAMES timeframe,const int adx_period,const int id=WRONG_VALUE);
   int                     CreateAlligator(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int jaw_period,
                                       const int jaw_shift,
                                       const int teeth_period,
                                       const int teeth_shift,
                                       const int lips_period,
                                       const int lips_shift,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_PRICE applied_price,
                                       const int id=WRONG_VALUE);
   int                     CreateAMA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ama_period,
                                       const int fast_ma_period,
                                       const int slow_ma_period,
                                       const int ama_shift,
                                       const ENUM_APPLIED_PRICE applied_price,
                                       const int id=WRONG_VALUE);
   int                     CreateAO(const string symbol,const ENUM_TIMEFRAMES timeframe,const int id=WRONG_VALUE);
   int                     CreateATR(const string symbol,const ENUM_TIMEFRAMES timeframe,const int ma_period,const int id=WRONG_VALUE);
   int                     CreateBearsPower(const string symbol,const ENUM_TIMEFRAMES timeframe,const int ma_period,const int id=WRONG_VALUE);
   int                     CreateBands(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int bands_period,
                                       const int bands_shift,
                                       const double deviation,
                                       const ENUM_APPLIED_PRICE applied_price,
                                       const int id=WRONG_VALUE);
   int                     CreateBullsPower(const string symbol,const ENUM_TIMEFRAMES timeframe,const int ma_period,const int id=WRONG_VALUE);
   int                     CreateCCI(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const ENUM_APPLIED_PRICE applied_price,
                                       const int id=WRONG_VALUE);
   int                     CreateChaikin(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int fast_ma_period,
                                       const int slow_ma_period,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_VOLUME applied_volume,
                                       const int id=WRONG_VALUE);
   int                     CreateDEMA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const ENUM_APPLIED_PRICE applied_price,
                                       const int id=WRONG_VALUE);
   int                     CreateDeMarker(const string symbol,const ENUM_TIMEFRAMES timeframe,const int ma_period,const int id=WRONG_VALUE);
   int                     CreateEnvelopes(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_PRICE applied_price,
                                       const double deviation,
                                       const int id=WRONG_VALUE);
   int                     CreateForce(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_VOLUME applied_volume,
                                       const int id=WRONG_VALUE);
   int                     CreateFractals(const string symbol,const ENUM_TIMEFRAMES timeframe,const int id=WRONG_VALUE);
   int                     CreateFrAMA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const ENUM_APPLIED_PRICE applied_price,
                                       const int id=WRONG_VALUE);
   int                     CreateGator(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int jaw_period,
                                       const int jaw_shift,
                                       const int teeth_period,
                                       const int teeth_shift,
                                       const int lips_period,
                                       const int lips_shift,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_PRICE applied_price,
                                       const int id=WRONG_VALUE);
   int                     CreateIchimoku(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int tenkan_sen,
                                       const int kijun_sen,
                                       const int senkou_span_b,
                                       const int id=WRONG_VALUE);
   int                     CreateBWMFI(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const ENUM_APPLIED_VOLUME applied_volume,
                                       const int id=WRONG_VALUE);
   int                     CreateMomentum(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int mom_period,
                                       const ENUM_APPLIED_PRICE applied_price,
                                       const int id=WRONG_VALUE);
   int                     CreateMFI(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const ENUM_APPLIED_VOLUME applied_volume,
                                       const int id=WRONG_VALUE);
   int                     CreateMA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_PRICE applied_price,
                                       const int id=WRONG_VALUE);
   int                     CreateOsMA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int fast_ema_period,
                                       const int slow_ema_period,
                                       const int signal_period,
                                       const ENUM_APPLIED_PRICE applied_price,
                                       const int id=WRONG_VALUE);
   int                     CreateMACD(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int fast_ema_period,
                                       const int slow_ema_period,
                                       const int signal_period,
                                       const ENUM_APPLIED_PRICE applied_price,
                                       const int id=WRONG_VALUE);
   int                     CreateOBV(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const ENUM_APPLIED_VOLUME applied_volume,
                                       const int id=WRONG_VALUE);
   int                     CreateSAR(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const double step,
                                       const double maximum,
                                       const int id=WRONG_VALUE);
   int                     CreateRSI(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const ENUM_APPLIED_PRICE applied_price,
                                       const int id=WRONG_VALUE);
   int                     CreateRVI(const string symbol,const ENUM_TIMEFRAMES timeframe,const int ma_period,const int id=WRONG_VALUE);
   int                     CreateStdDev(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_PRICE applied_price,
                                       const int id=WRONG_VALUE);
   int                     CreateStochastic(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int Kperiod,
                                       const int Dperiod,
                                       const int slowing,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_STO_PRICE price_field,
                                       const int id=WRONG_VALUE);
   int                     CreateTEMA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const ENUM_APPLIED_PRICE applied_price,
                                       const int id=WRONG_VALUE);
   int                     CreateTriX(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const ENUM_APPLIED_PRICE applied_price,
                                       const int id=WRONG_VALUE);
   int                     CreateWPR(const string symbol,const ENUM_TIMEFRAMES timeframe,const int calc_period,const int id=WRONG_VALUE);
   int                     CreateVIDYA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int cmo_period,
                                       const int ema_period,
                                       const int ma_shift,
                                       const ENUM_APPLIED_PRICE applied_price,
                                       const int id=WRONG_VALUE);
   int                     CreateVolumes(const string symbol,const ENUM_TIMEFRAMES timeframe,const ENUM_APPLIED_VOLUME applied_volume,const int id=WRONG_VALUE);
```

Each specific standard indicator type is to use its own method of creating an appropriate indicator and necessary buffer objects.

As an example, here I will implement handling AC indicator. Let's write the method of creating AC indicator and its buffers beyond the class body:

```
//+------------------------------------------------------------------+
//| Create multi-symbol multi-period AC                              |
//+------------------------------------------------------------------+
int CBuffersCollection::CreateAC(const string symbol,const ENUM_TIMEFRAMES timeframe,const int id=WRONG_VALUE)
  {
//--- Create the indicator handle and set the default ID
   int handle=::iAC(symbol,timeframe);
   int identifier=(id==WRONG_VALUE ? IND_AC : id);
   if(handle!=INVALID_HANDLE)
     {
      //--- Create the histogram buffer from the zero line
      this.CreateHistogram();
      //--- Get the last created (drawn) buffer object and set all the necessary parameters to it
      CBuffer *buff=this.GetLastCreateBuffer();
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_AC);
      buff.SetShowData(true);
      buff.SetLabel("AC("+symbol+","+TimeframeDescription(timeframe)+")");
      buff.SetIndicatorName("Accelerator Oscillator");

      //--- Create a calculated buffer storing standard indicator data
      this.CreateCalculate();
      //--- Get the last created (calculated) buffer object and set all the necessary parameters to it
      buff=this.GetLastCreateBuffer();
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_AC);
      buff.SetEmptyValue(EMPTY_VALUE);
      buff.SetLabel("AC("+symbol+","+TimeframeDescription(timeframe)+")");
      buff.SetIndicatorName("Accelerator Oscillator");
     }
   return handle;
  }
//+------------------------------------------------------------------+
```

As you can see, all is simple here. If -1 is passed as the ID, the ID is equal to the [standard indicator type constant](https://www.mql5.com/en/docs/constants/indicatorconstants/enum_indicator). If the indicator is created successfully (its handle is not equal to INVALID\_HANDLE), create the buffer object of "Histogram from zero line" drawing type and use the GetLastCreateBuffer() method returning the pointer to the last created buffer (the method is to be considered later) to get the pointer to the histogram buffer object and set the necessary parameters for its identification as a buffer for drawing data of the standard AC indicator.

Next, do the same for the calculated buffer as well. In the calculated buffer, write AC indicator data obtained when accessing its handle. The handle of the created indicator is set in the buffer object properties. This is true both for drawn and calculated one, i.e. we can get any of these buffer objects, access the indicator by the handle set in the objects and work with the indicator.

Add implementation of the method of creating the AD indicator with the necessary buffer objects to see the differences in implementing the methods for standard indicators of different types:

```
//+------------------------------------------------------------------+
//| Create multi-symbol multi-period AD                              |
//+------------------------------------------------------------------+
int CBuffersCollection::CreateAD(const string symbol,const ENUM_TIMEFRAMES timeframe,const ENUM_APPLIED_VOLUME applied_volume,const int id=WRONG_VALUE)
  {
//--- Create the indicator handle and set the default ID
   int handle=::iAD(symbol,timeframe,applied_volume);
   int identifier=(id==WRONG_VALUE ? IND_AD : id);
   if(handle!=INVALID_HANDLE)
     {
      //--- Create the line buffer
      this.CreateLine();
      //--- Get the last created (drawn) buffer object and set all the necessary parameters to it
      CBuffer *buff=this.GetLastCreateBuffer();
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_AD);
      buff.SetShowData(true);
      buff.SetLabel("AD("+symbol+","+TimeframeDescription(timeframe)+")");
      buff.SetIndicatorName("Accumulation/Distribution");

      //--- Create a calculated buffer storing standard indicator data
      this.CreateCalculate();
      //--- Get the last created (calculated) buffer object and set all the necessary parameters to it
      buff=this.GetLastCreateBuffer();
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_AD);
      buff.SetEmptyValue(EMPTY_VALUE);
      buff.SetLabel("AD("+symbol+","+TimeframeDescription(timeframe)+")");
      buff.SetIndicatorName("Accumulation/Distribution");
     }
   return handle;
  }
//+------------------------------------------------------------------+
```

The differences are small and are mostly related to the drawn buffer type, standard indicator type, graphical series name and indicator name. In other types of standard indicators, there will be a different number of drawn and calculated buffer objects (if necessary) for handling the standard indicator.

Add declaration of the remaining methods directly after declaring the methods of creating standard indicators:

```
//--- Prepare calculated buffer data of the specified standard indicator
   int                     PreparingDataBufferStdInd(const ENUM_INDICATOR std_ind,const int id,const int total_copy);
//--- Clear buffer data of the specified standard indicator by the timeseries index
   void                    ClearDataBufferStdInd(const ENUM_INDICATOR std_ind,const int id,const int series_index);
//--- Set the values for the current chart to the specified standard indicator buffer by the timeseries index according to the buffer object period/symbol
   bool                    SetDataBufferStdInd(const ENUM_INDICATOR std_ind,const int id,const int series_index,const datetime series_time,const char color_index=WRONG_VALUE);

//--- Return the buffer (1) by the graphical series name, (2) by timeframe,
//--- (3) by Plot index, (4) by object index in the collection list, (5) the last created one,
//--- list of buffers (6) by ID, (7) by standard indicator type, (8) by type and ID
   CBuffer                *GetBufferByLabel(const string plot_label);
   CBuffer                *GetBufferByTimeframe(const ENUM_TIMEFRAMES timeframe);
   CBuffer                *GetBufferByPlot(const int plot_index);
   CBuffer                *GetBufferByListIndex(const int index_list);
   CBuffer                *GetLastCreateBuffer(void);
   CArrayObj              *GetListBufferByID(const int id);
   CArrayObj              *GetListBufferByIndType(const ENUM_INDICATOR indicator_type);
   CArrayObj              *GetListBufferByTypeID(const ENUM_INDICATOR indicator_type,const int id);
```

All declared methods are named in the comments. Let's consider their implementations outside the class body.

**The method returning the last created buffer object:**

```
//+------------------------------------------------------------------+
//| Return the last created buffer                                   |
//+------------------------------------------------------------------+
CBuffer *CBuffersCollection::GetLastCreateBuffer(void)
  {
   return this.m_list.At(this.m_list.Total()-1);
  }
//+------------------------------------------------------------------+
```

The method simply returns the pointer to the buffer object which is the last in the list of buffer objects.

**The method returning the list of buffer objects by ID:**

```
//+------------------------------------------------------------------+
//| Return the list of buffers by ID                                 |
//+------------------------------------------------------------------+
CArrayObj *CBuffersCollection::GetListBufferByID(const int id)
  {
   CArrayObj *list=CSelect::ByBufferProperty(this.GetList(),BUFFER_PROP_ID,id,EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
```

Get the list of buffer objects featuring the IDequal to the one passed to the method.

Return the pointer to the obtained list from the method.

**The method returning the list of buffer objects by the standard indicator type:**

```
//+------------------------------------------------------------------+
//| Return the list of buffers by the standard indicator type        |
//+------------------------------------------------------------------+
CArrayObj *CBuffersCollection::GetListBufferByIndType(const ENUM_INDICATOR indicator_type)
  {
   CArrayObj *list=CSelect::ByBufferProperty(this.GetList(),BUFFER_PROP_IND_TYPE,indicator_type,EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
```

Get the list of buffer objects featuring the standard indicator typeequal tothe one passed to the method.

Return the pointer to the obtained list from the method.

**The method returning the list of buffer objects by the standard indicator type and ID:**

```
//+------------------------------------------------------------------+
//| Return the list of buffers by type and ID                        |
//+------------------------------------------------------------------+
CArrayObj *CBuffersCollection::GetListBufferByTypeID(const ENUM_INDICATOR indicator_type,const int id)
  {
   CArrayObj *list=this.GetListBufferByIndType(indicator_type);
   list=CSelect::ByBufferProperty(list,BUFFER_PROP_ID,id,EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
```

First, get the list of buffer objects with the specified type of standard indicator in their properties. Next, sort the obtained list by buffer objects whose properties feature the specified ID.

The pointer to the resulting list is returned from the method.

**The method returning the list of buffer objects belonging to some standard indicator:**

```
//+------------------------------------------------------------------+
//| Return the list of indicator buffers                             |
//| (featuring the ID of belonging to an indicator)                  |
//+------------------------------------------------------------------+
CArrayObj *CBuffersCollection::GetListBuffersWithID(void)
  {
   CArrayObj *list=CSelect::ByBufferProperty(this.GetList(),BUFFER_PROP_ID,WRONG_VALUE,NO_EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
```

From the collection list of buffer objects, get the list of objects whose ID propertyis not equal to -1.

Return the pointer to the obtained list from the method.

The CEngine library main object class is the link between the program and the library.

Let's make the necessary improvements in the class file \\MQL5\\Include\\DoEasy\ **Engine.mqh**.

Add the method of re-creating all timeseries to the public class section:

```
//--- Re-create (1) the specified timeseries of the specified symbol, (2) all collection timeseries
   bool                 SeriesReCreate(const string symbol,const ENUM_TIMEFRAMES timeframe,const int rates_total=0,const uint required=0)
                          { return this.m_time_series.ReCreateSeries(symbol,timeframe,rates_total,required);      }
   bool                 SeriesReCreateAll(const int rates_total=0,const uint required=0)
                          { return this.m_time_series.ReCreateSeriesAll(rates_total,required);                    }
```

The method simply returns the result of the timeseries collection method of the same name I have added above.

In the public section of the class, add the method returning the number of bars of the specified timeseries:

```
//--- Return (1) an empty, (2) partially filled timeseries
   CSeriesDE           *SeriesGetSeriesEmpty(void)       { return this.m_time_series.GetSeriesEmpty();            }
   CSeriesDE           *SeriesGetSeriesIncompleted(void) { return this.m_time_series.GetSeriesIncompleted();      }
//--- Return the umber of bars of the timeseries of the specified symbol/period
   int                  SeriesGetBarsTotal(const string symbol,const ENUM_TIMEFRAMES timeframe);
```

Implement the method outside the class body:

```
//+---------------------------------------------------------------------------+
//| Return the umber of bars of the timeseries of the specified symbol/period |
//+---------------------------------------------------------------------------+
int CEngine::SeriesGetBarsTotal(const string symbol,const ENUM_TIMEFRAMES timeframe)
  {
   CSeriesDE *series=this.SeriesGetSeries(symbol,timeframe);
   if(series==NULL)
      return WRONG_VALUE;
   return (int)series.Bars();
  }
//+------------------------------------------------------------------+
```

Get the specified timeseries from the timeseries collection class and return the number of timeseries bars.

Previously, I had the method returning the last created buffer:

```
//--- Return the buffer by (1) the graphical series name, (2) timeframe, (3) Plot index, (4) collection list and (5) the last one in the list
   CBuffer             *GetBufferByLabel(const string plot_label)                      { return this.m_buffers.GetBufferByLabel(plot_label); }
   CBuffer             *GetBufferByTimeframe(const ENUM_TIMEFRAMES timeframe)          { return this.m_buffers.GetBufferByTimeframe(timeframe);}
   CBuffer             *GetBufferByPlot(const int plot_index)                          { return this.m_buffers.GetBufferByPlot(plot_index);  }
   CBuffer             *GetBufferByListIndex(const int index_list)                     { return this.m_buffers.GetBufferByListIndex(index_list);}
   CBuffer             *GetLastBuffer(void);
```

and its implementation:

```
//+------------------------------------------------------------------+
//| Return the last indicator buffer                                 |
//| in the indicator buffer collection list                          |
//+------------------------------------------------------------------+
CBuffer *CEngine::GetLastBuffer(void)
  {
   CArrayObj *list=this.GetListBuffers();
   if(list==NULL)
      return NULL;
   return list.At(list.Total()-1);
  }
//+------------------------------------------------------------------+
```

Remove the method implementation from the class listingreplacing its declaration with the new method:

```
//--- Return the buffer by (1) the graphical series name, (2) timeframe, (3) Plot index, (4) collection list and (5) the last one in the list
   CBuffer             *GetBufferByLabel(const string plot_label)                      { return this.m_buffers.GetBufferByLabel(plot_label); }
   CBuffer             *GetBufferByTimeframe(const ENUM_TIMEFRAMES timeframe)          { return this.m_buffers.GetBufferByTimeframe(timeframe);}
   CBuffer             *GetBufferByPlot(const int plot_index)                          { return this.m_buffers.GetBufferByPlot(plot_index);  }
   CBuffer             *GetBufferByListIndex(const int index_list)                     { return this.m_buffers.GetBufferByListIndex(index_list);}
   CBuffer             *GetLastCreateBuffer(void)                                      { return this.m_buffers.GetLastCreateBuffer();        }
```

The method returns the result of the buffer collection class method of the same name considered above.

In the public class section, add one method for creating AC standard indicator and buffers for its operation:

```
//--- Create the new buffer (1) "Drawing with arrows", (2) "Line", (3) "Sections", (4) "Histogram from the zero line",
//--- (5) "Histogram on two indicator buffers", (6) "Zigzag", (7) "Color filling between two levels",
//--- (8) "Display as bars", (9) "Display as candles", calculated buffer
   bool                 BufferCreateArrow(void)                                        { return this.m_buffers.CreateArrow();                }
   bool                 BufferCreateLine(void)                                         { return this.m_buffers.CreateLine();                 }
   bool                 BufferCreateSection(void)                                      { return this.m_buffers.CreateSection();              }
   bool                 BufferCreateHistogram(void)                                    { return this.m_buffers.CreateHistogram();            }
   bool                 BufferCreateHistogram2(void)                                   { return this.m_buffers.CreateHistogram2();           }
   bool                 BufferCreateZigZag(void)                                       { return this.m_buffers.CreateZigZag();               }
   bool                 BufferCreateFilling(void)                                      { return this.m_buffers.CreateFilling();              }
   bool                 BufferCreateBars(void)                                         { return this.m_buffers.CreateBars();                 }
   bool                 BufferCreateCandles(void)                                      { return this.m_buffers.CreateCandles();              }
   bool                 BufferCreateCalculate(void)                                    { return this.m_buffers.CreateCalculate();            }

//--- The methods of creating standard indicators and buffer objects for them
   bool                 BufferCreateAC(const string symbol,const ENUM_TIMEFRAMES timeframe,const int id)
                          { return(this.m_buffers.CreateAC(symbol,timeframe,id)!=INVALID_HANDLE);  }

//--- Initialize all drawn buffers by a (1) specified value, (2) empty value set for the buffer object
```

The method returns the result of AC indicator creation method of the same name from the indicator buffer collection class I have considered above. The methods for creating other standard indicators are to be added in the following articles.

**Implementing the method of preparing calculated buffer data for the standard indicator (only for AC so far):**

```
//+------------------------------------------------------------------+
//| Prepare the calculated buffer data                               |
//| of the specified standard indicator                              |
//+------------------------------------------------------------------+
int CBuffersCollection::PreparingDataBufferStdInd(const ENUM_INDICATOR std_ind,const int id,const int total_copy)
  {
   CArrayObj *list=this.GetListBufferByTypeID(std_ind,id);
   list=CSelect::ByBufferProperty(list,BUFFER_PROP_TYPE,BUFFER_TYPE_CALCULATE,EQUAL);
   if(list==NULL || list.Total()==0)
      return 0;
   CBufferCalculate *buffer=NULL;
   int copies=WRONG_VALUE;
   switch((int)std_ind)
     {
      case IND_AC :
        buffer=list.At(0);
        if(buffer==NULL) return 0;
        copies=buffer.FillAsSeries(buffer.IndicatorHandle(),0,0,total_copy);
        return copies;

      case IND_AD :
        break;
      case IND_ADX :
        break;
      case IND_ADXW :
        break;
      case IND_ALLIGATOR :
        break;
      case IND_AMA :
        break;
      case IND_AO :
        break;
      case IND_ATR :
        break;
      case IND_BANDS :
        break;
      case IND_BEARS :
        break;
      case IND_BULLS :
        break;
      case IND_BWMFI :
        break;
      case IND_CCI :
        break;
      case IND_CHAIKIN :
        break;
      case IND_DEMA :
        break;
      case IND_DEMARKER :
        break;
      case IND_ENVELOPES :
        break;
      case IND_FORCE :
        break;
      case IND_FRACTALS :
        break;
      case IND_FRAMA :
        break;
      case IND_GATOR :
        break;
      case IND_ICHIMOKU :
        break;
      case IND_MA :
        break;
      case IND_MACD :
        break;
      case IND_MFI :
        break;
      case IND_MOMENTUM :
        break;
      case IND_OBV :
        break;
      case IND_OSMA :
        break;
      case IND_RSI :
        break;
      case IND_RVI :
        break;
      case IND_SAR :
        break;
      case IND_STDDEV :
        break;
      case IND_STOCHASTIC :
        break;
      case IND_TEMA :
        break;
      case IND_TRIX :
        break;
      case IND_VIDYA :
        break;
      case IND_VOLUMES :
        break;
      case IND_WPR :
        break;

      default:
        break;
     }
   return 0;
  }
//+------------------------------------------------------------------+
```

Get the list of buffer objects by indicator type and ID, leave only calculated buffers in the obtained list, get the very first (and only for AC) calculated buffer from the list, fill in the calculated buffer with the specified amount of data and return the number of successfully copied data from the indicator handle to the calculated buffer.

**Implementing the method of clearing the calculated buffer data for the standard indicator by the specified index (only for AC so far):**

```
//+------------------------------------------------------------------+
//| Clear buffer data of the specified standard indicator            |
//| by the timeseries index                                          |
//+------------------------------------------------------------------+
void CBuffersCollection::ClearDataBufferStdInd(const ENUM_INDICATOR std_ind,const int id,const int series_index)
  {
   CArrayObj *list=this.GetListBufferByID(id);
   if(list==NULL)
      return;
   list=CSelect::ByBufferProperty(list,BUFFER_PROP_TYPE,BUFFER_TYPE_DATA,EQUAL);
   if(list.Total()==0)
      return;
   CBuffer *buffer=NULL;
   switch((int)std_ind)
     {
      case IND_AC :
        buffer=list.At(0);
        if(buffer==NULL) return;
        buffer.SetBufferValue(0,series_index,buffer.EmptyValue());
        break;

      case IND_AD :
        break;
      case IND_ADX :
        break;
      case IND_ADXW :
        break;
      case IND_ALLIGATOR :
        break;
      case IND_AMA :
        break;
      case IND_AO :
        break;
      case IND_ATR :
        break;
      case IND_BANDS :
        break;
      case IND_BEARS :
        break;
      case IND_BULLS :
        break;
      case IND_BWMFI :
        break;
      case IND_CCI :
        break;
      case IND_CHAIKIN :
        break;
      case IND_DEMA :
        break;
      case IND_DEMARKER :
        break;
      case IND_ENVELOPES :
        break;
      case IND_FORCE :
        break;
      case IND_FRACTALS :
        break;
      case IND_FRAMA :
        break;
      case IND_GATOR :
        break;
      case IND_ICHIMOKU :
        break;
      case IND_MA :
        break;
      case IND_MACD :
        break;
      case IND_MFI :
        break;
      case IND_MOMENTUM :
        break;
      case IND_OBV :
        break;
      case IND_OSMA :
        break;
      case IND_RSI :
        break;
      case IND_RVI :
        break;
      case IND_SAR :
        break;
      case IND_STDDEV :
        break;
      case IND_STOCHASTIC :
        break;
      case IND_TEMA :
        break;
      case IND_TRIX :
        break;
      case IND_VIDYA :
        break;
      case IND_VOLUMES :
        break;
      case IND_WPR :
        break;

      default:
        break;
     }
  }
//+------------------------------------------------------------------+
```

The method works similarly to the data preparation method. But instead of copying data from the indicator handle to the calculated buffer, the empty value set for the buffer object by the specified index to the drawn buffer is specified here.

**Implementing the method of filling the drawn buffer on the current chart with standard indicator data from any symbol/timeframe (only for AC so far):**

```
//+------------------------------------------------------------------+
//| Set values for the current chart to the specified buffer         |
//| of the standard indicator by the timeseries index according to   |
//| the buffer object symbol/period                                  |
//+------------------------------------------------------------------+
bool CBuffersCollection::SetDataBufferStdInd(const ENUM_INDICATOR ind_type,const int id,const int series_index,const datetime series_time,const char color_index=WRONG_VALUE)
  {
//--- Get the list of buffer objects with ID
   CArrayObj *list=this.GetListBufferByTypeID(ind_type,id);
   if(list==NULL)
      return false;
//--- Get the list of drawn objects with ID
   CArrayObj *list_data=CSelect::ByBufferProperty(list,BUFFER_PROP_TYPE,BUFFER_TYPE_DATA,EQUAL);
   list_data=CSelect::ByBufferProperty(list_data,BUFFER_PROP_IND_TYPE,ind_type,EQUAL);
//--- Get the list of calculated buffers with ID
   CArrayObj *list_calc=CSelect::ByBufferProperty(list,BUFFER_PROP_TYPE,BUFFER_TYPE_CALCULATE,EQUAL);
   list_calc=CSelect::ByBufferProperty(list_calc,BUFFER_PROP_IND_TYPE,ind_type,EQUAL);
//--- Exit if any of the lists is empty
   if(list_data.Total()==0 || list_calc.Total()==0)
      return false;
//--- Declare the necessary objects and variables
   CBuffer *buffer_data=NULL;
   CBuffer *buffer_calc=NULL;
   int index_period=0;
   int series_index_start=0;
   int num_bars=1,index=0;
   datetime time_period=0;
   double value0=EMPTY_VALUE, value1=EMPTY_VALUE;

//--- Depending on the standard indicator type
   switch((int)ind_type)
     {
      case IND_AC :
        //--- Get drawn and calculated buffer objects
        buffer_data=list_data.At(0);
        buffer_calc=list_calc.At(0);
        if(buffer_calc==NULL || buffer_data==NULL || buffer_calc.GetDataTotal(0)==0) return false;

        //--- Find the bar index corresponding to the current bar start time
        index_period=::iBarShift(buffer_calc.Symbol(),buffer_calc.Timeframe(),series_time,true);
        if(index_period==WRONG_VALUE || index_period>buffer_calc.GetDataTotal()-1) return false;
        //--- Get the value by the index from the indicator buffer
        value0=buffer_calc.GetDataBufferValue(0,index_period);
        if(buffer_calc.Symbol()==::Symbol() && buffer_calc.Timeframe()==::Period())
          {
           series_index_start=series_index;
           num_bars=1;
          }
        else
          {
           //--- Get the bar time the bar with the index_period index falls into on the calculated buffer period and symbol
           time_period=::iTime(buffer_calc.Symbol(),buffer_calc.Timeframe(),index_period);
           if(time_period==0) return false;
           //--- Get the appropriate current chart bar
           series_index_start=::iBarShift(::Symbol(),::Period(),time_period,true);
           if(series_index_start==WRONG_VALUE) return false;
           //--- Calculate the number of bars on the current chart which should be filled with calculated buffer data
           num_bars=::PeriodSeconds(buffer_calc.Timeframe())/::PeriodSeconds(PERIOD_CURRENT);
           if(num_bars==0) num_bars=1;
          }
        //--- Take values to calculate colors
        value1=(series_index_start+num_bars>buffer_data.GetDataTotal()-1 ? value0 : buffer_data.GetDataBufferValue(0,series_index_start+num_bars));
        //--- In the loop by the number of bars in num_bars, fill in the drawn buffer with the calculated buffer value taken by the index_period index
        //--- and set the color of the drawn buffer depending on the value0 and value1 values ratio
        for(int i=0;i<num_bars;i++)
          {
           index=series_index_start-i;
           buffer_data.SetBufferValue(0,index,value0);
           buffer_data.SetBufferColorIndex(index,uchar(value0>value1 ? 0 : value0<value1 ? 1 : 2));
          }
        break;

      case IND_AD :
        break;
      case IND_ADX :
        break;
      case IND_ADXW :
        break;
      case IND_ALLIGATOR :
        break;
      case IND_AMA :
        break;
      case IND_AO :
        break;
      case IND_ATR :
        break;
      case IND_BANDS :
        break;
      case IND_BEARS :
        break;
      case IND_BULLS :
        break;
      case IND_BWMFI :
        break;
      case IND_CCI :
        break;
      case IND_CHAIKIN :
        break;
      case IND_DEMA :
        break;
      case IND_DEMARKER :
        break;
      case IND_ENVELOPES :
        break;
      case IND_FORCE :
        break;
      case IND_FRACTALS :
        break;
      case IND_FRAMA :
        break;
      case IND_GATOR :
        break;
      case IND_ICHIMOKU :
        break;
      case IND_MA :
        break;
      case IND_MACD :
        break;
      case IND_MFI :
        break;
      case IND_MOMENTUM :
        break;
      case IND_OBV :
        break;
      case IND_OSMA :
        break;
      case IND_RSI :
        break;
      case IND_RVI :
        break;
      case IND_SAR :
        break;
      case IND_STDDEV :
        break;
      case IND_STOCHASTIC :
        break;
      case IND_TEMA :
        break;
      case IND_TRIX :
        break;
      case IND_VIDYA :
        break;
      case IND_VOLUMES :
        break;
      case IND_WPR :
        break;

      default:
        break;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

The entire method logic related to data calculation for AC is thoroughly described in comments.

At the very end of the class body, declare two methods for handling library events:

```
public:
//--- Create and return the composite magic number from the specified magic number value, the first and second group IDs and the pending request ID
   uint                 SetCompositeMagicNumber(ushort magic_id,const uchar group_id1=0,const uchar group_id2=0,const uchar pending_req_id=0);

//--- Handling DoEasy library events
void                    OnDoEasyEvent(const int id,const long &lparam,const double &dparam,const string &sparam);
//--- Working with events in the tester
void                    EventsHandling(void);

  };
//+------------------------------------------------------------------+
```

Previously, I used the functions having the same names as the newly declared methods in order to handle library events in custom programs. We passed these functions from one program to another without any changes. This suggests that these handlers can be transferred to the library, while in the program, we can simply receive the flags of the occurred events (receiving the flags, as well as the event flags themselves and the ability to handle events in custom programs are to be implemented later).

I have already moved these functions from the test indicator to the CEngine class listing using them to implement the methods declared above:

```
//+------------------------------------------------------------------+
//| Handling DoEasy library events                                   |
//+------------------------------------------------------------------+
void CEngine::OnDoEasyEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
  {
   int idx=id-CHARTEVENT_CUSTOM;
//--- Retrieve (1) event time milliseconds, (2) reason and (3) source from lparam, as well as (4) set the exact event time
   ushort msc=this.EventMSC(lparam);
   ushort reason=this.EventReason(lparam);
   ushort source=this.EventSource(lparam);
   long time=::TimeCurrent()*1000+msc;

//--- Handling symbol events
   if(source==COLLECTION_SYMBOLS_ID)
     {
      CSymbol *symbol=this.GetSymbolObjByName(sparam);
      if(symbol==NULL)
         return;
      //--- Number of decimal places in the event value - in case of a 'long' event, it is 0, otherwise - Digits() of a symbol
      int digits=(idx<SYMBOL_PROP_INTEGER_TOTAL ? 0 : symbol.Digits());
      //--- Event text description
      string id_descr=(idx<SYMBOL_PROP_INTEGER_TOTAL ? symbol.GetPropertyDescription((ENUM_SYMBOL_PROP_INTEGER)idx) : symbol.GetPropertyDescription((ENUM_SYMBOL_PROP_DOUBLE)idx));
      //--- Property change text value
      string value=::DoubleToString(dparam,digits);

      //--- Check event reasons and display its description in the journal
      if(reason==BASE_EVENT_REASON_INC)
        {
         ::Print(DFUN,symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_DEC)
        {
         ::Print(DFUN,symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_MORE_THEN)
        {
         ::Print(DFUN,symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_LESS_THEN)
        {
         ::Print(DFUN,symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_EQUALS)
        {
         ::Print(DFUN,symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
     }

//--- Handling account events
   else if(source==COLLECTION_ACCOUNT_ID)
     {
      CAccount *account=this.GetAccountCurrent();
      if(account==NULL)
         return;
      //--- Number of decimal places in the event value - in case of a 'long' event, it is 0, otherwise - Digits() of a symbol
      int digits=int(idx<ACCOUNT_PROP_INTEGER_TOTAL ? 0 : account.CurrencyDigits());
      //--- Event text description
      string id_descr=(idx<ACCOUNT_PROP_INTEGER_TOTAL ? account.GetPropertyDescription((ENUM_ACCOUNT_PROP_INTEGER)idx) : account.GetPropertyDescription((ENUM_ACCOUNT_PROP_DOUBLE)idx));
      //--- Property change text value
      string value=::DoubleToString(dparam,digits);

      //--- Checking event reasons and handling the increase of funds by a specified value,

      //--- Display an event in the journal
      if(reason==BASE_EVENT_REASON_INC)
        {
         ::Print(DFUN,account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_DEC)
        {
         ::Print(DFUN,account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_MORE_THEN)
        {
         ::Print(DFUN,account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_LESS_THEN)
        {
         ::Print(DFUN,account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_EQUALS)
        {
         ::Print(DFUN,account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
     }

//--- Handling market watch window events
   else if(idx>MARKET_WATCH_EVENT_NO_EVENT && idx<SYMBOL_EVENTS_NEXT_CODE)
     {
      //--- Market Watch window event
      string descr=this.GetMWEventDescription((ENUM_MW_EVENT)idx);
      string name=(idx==MARKET_WATCH_EVENT_SYMBOL_SORT ? "" : ": "+sparam);
      Print(TimeMSCtoString(lparam)," ",descr,name);
     }

//--- Handling timeseries events
   else if(idx>SERIES_EVENTS_NO_EVENT && idx<SERIES_EVENTS_NEXT_CODE)
     {
      //--- "New bar" event
      if(idx==SERIES_EVENTS_NEW_BAR)
        {
         ::Print(DFUN,TextByLanguage("Новый бар на ","New Bar on "),sparam," ",TimeframeDescription((ENUM_TIMEFRAMES)dparam),": ",TimeToString(lparam));
         CArrayObj *list=this.m_buffers.GetListBuffersWithID();
         if(list!=NULL)
           {
            int total=list.Total();
            for(int i=0;i<total;i++)
              {
               CBuffer *buff=list.At(i);
               if(buff==NULL)
                  continue;
               string symbol=sparam;
               ENUM_TIMEFRAMES timeframe=(ENUM_TIMEFRAMES)dparam;
               if(buff.TypeBuffer()==BUFFER_TYPE_DATA || buff.IndicatorType()==WRONG_VALUE)
                  continue;
               if(buff.Symbol()==symbol && buff.Timeframe()==timeframe )
                 {
                  CSeriesDE *series=this.SeriesGetSeries(symbol,timeframe);
                  if(series==NULL)
                     continue;
                  int count=::fmin(buff.GetDataTotal(),buff.IndicatorBarsCalculated());
                  this.m_buffers.PreparingDataBufferStdInd(buff.IndicatorType(),buff.ID(),count);
                 }
              }
           }
        }
      //--- "Bars skipped" event
      if(idx==SERIES_EVENTS_MISSING_BARS)
        {
         ::Print(DFUN,TextByLanguage("Пропущены бары на ","Missed bars on "),sparam," ",TimeframeDescription((ENUM_TIMEFRAMES)dparam),": ",(string)lparam);
        }
     }

//--- Handling trading events
   else if(idx>TRADE_EVENT_NO_EVENT && idx<TRADE_EVENTS_NEXT_CODE)
     {
      //--- Get the list of trading events
      CArrayObj *list=this.GetListAllOrdersEvents();
      if(list==NULL)
         return;
      //--- get the event index shift relative to the end of the list
      //--- in the tester, the shift is passed by the lparam parameter to the event handler
      //--- outside the tester, events are sent one by one and handled in OnChartEvent()
      int shift=(this.IsTester() ? (int)lparam : 0);
      CEvent *event=list.At(list.Total()-1-shift);
      if(event==NULL)
      return;
      //--- Accrue the credit
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_CREDIT)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Additional charges
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_CHARGE)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Correction
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_CORRECTION)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Enumerate bonuses
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_BONUS)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Additional commissions
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_COMISSION)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Daily commission
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_COMISSION_DAILY)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Monthly commission
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_COMISSION_MONTHLY)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Daily agent commission
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_COMISSION_AGENT_DAILY)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Monthly agent commission
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_COMISSION_AGENT_MONTHLY)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Interest rate
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_INTEREST)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Canceled buy deal
      if(event.TypeEvent()==TRADE_EVENT_BUY_CANCELLED)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Canceled sell deal
      if(event.TypeEvent()==TRADE_EVENT_SELL_CANCELLED)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Dividend operations
      if(event.TypeEvent()==TRADE_EVENT_DIVIDENT)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Accrual of franked dividend
      if(event.TypeEvent()==TRADE_EVENT_DIVIDENT_FRANKED)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Tax charges
      if(event.TypeEvent()==TRADE_EVENT_TAX)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Replenishing account balance
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_BALANCE_REFILL)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Withdrawing funds from balance
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_BALANCE_WITHDRAWAL)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }

      //--- Pending order placed
      if(event.TypeEvent()==TRADE_EVENT_PENDING_ORDER_PLASED)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Pending order removed
      if(event.TypeEvent()==TRADE_EVENT_PENDING_ORDER_REMOVED)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Pending order activated by price
      if(event.TypeEvent()==TRADE_EVENT_PENDING_ORDER_ACTIVATED)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Pending order partially activated by price
      if(event.TypeEvent()==TRADE_EVENT_PENDING_ORDER_ACTIVATED_PARTIAL)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Position opened
      if(event.TypeEvent()==TRADE_EVENT_POSITION_OPENED)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Position opened partially
      if(event.TypeEvent()==TRADE_EVENT_POSITION_OPENED_PARTIAL)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed by an opposite one
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_BY_POS)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed by StopLoss
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_BY_SL)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed by TakeProfit
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_BY_TP)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Position reversal by a new deal (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_REVERSED_BY_MARKET)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Position reversal by activating a pending order (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_REVERSED_BY_PENDING)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Position reversal by partial market order execution (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_REVERSED_BY_MARKET_PARTIAL)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Position reversal by activating a pending order (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_REVERSED_BY_PENDING_PARTIAL)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Added volume to a position by a new deal (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Added volume to a position by partial execution of a market order (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET_PARTIAL)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Added volume to a position by activating a pending order (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Added volume to a position by partial activation of a pending order (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING_PARTIAL)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed partially
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_PARTIAL)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Position partially closed by an opposite one
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed partially by StopLoss
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_SL)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed partially by TakeProfit
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_TP)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- StopLimit order activation
      if(event.TypeEvent()==TRADE_EVENT_TRIGGERED_STOP_LIMIT_ORDER)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order price
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_PRICE)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order and StopLoss price
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_PRICE_SL)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order and TakeProfit price
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_PRICE_TP)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order, StopLoss and TakeProfit price
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_PRICE_SL_TP)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order's StopLoss and TakeProfit price
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_SL_TP)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order's StopLoss
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_SL)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order's TakeProfit
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_TP)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing position's StopLoss and TakeProfit
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_POSITION_SL_TP)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing position StopLoss
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_POSITION_SL)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing position TakeProfit
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_POSITION_TP)
        {
         ::Print(DFUN,event.TypeEventDescription());
        }
     }
  }
//+------------------------------------------------------------------+
//| Working with events in the tester                                |
//+------------------------------------------------------------------+
void CEngine::EventsHandling(void)
  {
//--- If a trading event is present
   if(this.IsTradeEvent())
     {
      //--- Number of trading events occurred simultaneously
      int total=this.GetTradeEventsTotal();
      for(int i=0;i<total;i++)
        {
         //--- Get the next event from the list of simultaneously occurred events by index
         CEventBaseObj *event=this.GetTradeEventByIndex(i);
         if(event==NULL)
            continue;
         long   lparam=i;
         double dparam=event.DParam();
         string sparam=event.SParam();
         this.OnDoEasyEvent(CHARTEVENT_CUSTOM+event.ID(),lparam,dparam,sparam);
        }
     }
//--- If there is an account event
   if(this.IsAccountsEvent())
     {
      //--- Get the list of all account events occurred simultaneously
      CArrayObj* list=this.GetListAccountEvents();
      if(list!=NULL)
        {
         //--- Get the next event in a loop
         int total=list.Total();
         for(int i=0;i<total;i++)
           {
            //--- take an event from the list
            CEventBaseObj *event=list.At(i);
            if(event==NULL)
               continue;
            //--- Send an event to the event handler
            long lparam=event.LParam();
            double dparam=event.DParam();
            string sparam=event.SParam();
            this.OnDoEasyEvent(CHARTEVENT_CUSTOM+event.ID(),lparam,dparam,sparam);
           }
        }
     }
//--- If there is a symbol collection event
   if(this.IsSymbolsEvent())
     {
      //--- Get the list of all symbol events occurred simultaneously
      CArrayObj* list=this.GetListSymbolsEvents();
      if(list!=NULL)
        {
         //--- Get the next event in a loop
         int total=list.Total();
         for(int i=0;i<total;i++)
           {
            //--- take an event from the list
            CEventBaseObj *event=list.At(i);
            if(event==NULL)
               continue;
            //--- Send an event to the event handler
            long lparam=event.LParam();
            double dparam=event.DParam();
            string sparam=event.SParam();
            this.OnDoEasyEvent(CHARTEVENT_CUSTOM+event.ID(),lparam,dparam,sparam);
           }
        }
     }
//--- If there is a timeseries collection event
   if(this.IsSeriesEvent())
     {
      //--- Get the list of all timeseries events occurred simultaneously
      CArrayObj* list=this.GetListSeriesEvents();
      if(list!=NULL)
        {
         //--- Get the next event in a loop
         int total=list.Total();
         for(int i=0;i<total;i++)
           {
            //--- take an event from the list
            CEventBaseObj *event=list.At(i);
            if(event==NULL)
               continue;
            //--- Send an event to the event handler
            long lparam=event.LParam();
            double dparam=event.DParam();
            string sparam=event.SParam();
            this.OnDoEasyEvent(CHARTEVENT_CUSTOM+event.ID(),lparam,dparam,sparam);
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

I have already considered these functions (which are now CEngine class methods) in the starting library description articles when developing test EAs. The method listing clearly shows that almost each event is accompanied by a journal entry. Accordingly, it is possible to create the list of event flags in the global visibility scope and simply set the required flags. In custom programs, it is easier to implement handlers of each of the activated flags. I will do that later.

Thus, we no longer need to specify these handlers in each custom program.

The class features the [**calculate**](https://www.mql5.com/en/docs/runtime/event_fire#calculate) event handler called from the indicator. If the value returned by the handler is equal to zero, this means that not all timeseries used in the indicator have been constructed yet. The indicator should exit OnCalculate() with the return code of 0, which means waiting for the next tick and indicating that no data has been calculated.

Since I am adding handling standard indicators, it is necessary to make sure that the created indicator has been calculated.

To find the amount of calculated data, we may use the [BarsCalculated()](https://www.mql5.com/en/docs/series/barscalculated) function returning the amount of data already calculated by the indicator. If data has not been calculated yet, the function returns -1.

Add the check for successful calculation of all created standard indicators in the buffer collection to the method handling the 'calculate' event:

```
//+------------------------------------------------------------------+
//| Calculate event handler                                          |
//+------------------------------------------------------------------+
int CEngine::OnCalculate(SDataCalculate &data_calculate,const uint required=0)
  {
//--- If this is not an indicator, exit
   if(this.m_program!=PROGRAM_INDICATOR)
      return 0;
//--- Re-create empty timeseries
//--- If at least one of the timeseries is not synchronized, return zero
   if(!this.SeriesSync(data_calculate,required))
      return 0;
//--- Update the timeseries of the current symbol (not in the tester) and
//--- return either 0 (in case there are empty timeseries), or rates_total
   if(!this.IsTester())
      this.SeriesRefresh(NULL,data_calculate);
   int res=(this.SeriesGetSeriesEmpty()==NULL ? data_calculate.rates_total : 0);

//--- Check the amount of calculated standard indicator data
   CArrayObj *list=m_buffers.GetListBuffersWithID();
   if(list!=NULL)
     {
      //--- In a loop by the number of buffers having an ID
      int total=list.Total();
      for(int i=0;i<total;i++)
        {
         //--- get the next calculated buffer using the standard indicator
         CBuffer *buff=list.At(i);
         if(buff==NULL || buff.TypeBuffer()==BUFFER_TYPE_DATA || buff.IndicatorHandle()==INVALID_HANDLE)
            continue;
         //--- if the indicator data is not calculated yet, return zero
         if(buff.IndicatorBarsCalculated()==WRONG_VALUE)
            return 0;
        }
     }
   return res;
  }
//+------------------------------------------------------------------+
```

The logic of handling created indicator data is described in the method listing.

As the final touch in the revision of the library in the current article, I am going to add the current chart period to the list of used timeframes.

The file of the library service functions E:\\MetaQuotes\\MetaTrader 5\\MQL5\\Include\\DoEasy\\Services\ **DELib.mqh** features the function preparing the list of used timeframes. If the current chart period is not specified in the program settings, the library does not create its timeseries. But we constantly need the timeseries for our work.

Let's improve the CreateUsedTimeframesArray() function by adding the code block for specifying the current chart period in the list of used timeframes:

```
//+------------------------------------------------------------------+
//| Prepare the array of timeframes for the timeseries collection    |
//+------------------------------------------------------------------+
bool CreateUsedTimeframesArray(const ENUM_TIMEFRAMES_MODE mode_used_periods,string defined_used_periods,string &used_periods_array[])
  {
//--- If working with the current chart period, fill the array with the current timeframe description string
   if(mode_used_periods==TIMEFRAMES_MODE_CURRENT)
     {
      ArrayResize(used_periods_array,1,21);
      used_periods_array[0]=TimeframeDescription((ENUM_TIMEFRAMES)Period());
      return true;
     }
//--- If working with a predefined set of chart periods (from the defined_used_periods string)
   else if(mode_used_periods==TIMEFRAMES_MODE_LIST)
     {
      //--- Set comma as a separator (defined in the Datas.mqh file, page 11)
      string separator=INPUT_SEPARATOR;
      //--- Fill in the array of parameters from the string with predefined timeframes
      int n=StringParamsPrepare(defined_used_periods,separator,used_periods_array);
      //--- if nothing is found, display the appropriate message (working with the current period is selected automatically)
      if(n<1)
        {
         int err_code=GetLastError();
         string err=
           (n==0  ?
            DFUN_ERR_LINE+CMessage::Text(MSG_LIB_SYS_ERROR_EMPTY_PERIODS_STRING)+TimeframeDescription((ENUM_TIMEFRAMES)Period()) :
            DFUN_ERR_LINE+CMessage::Text(MSG_LIB_SYS_FAILED_PREPARING_PERIODS_ARRAY)+(string)err_code+": "+CMessage::Text(err_code)
           );
         Print(err);
         //--- Set the current period to the array
         ArrayResize(used_periods_array,1,21);
         used_periods_array[0]=TimeframeDescription((ENUM_TIMEFRAMES)Period());
         return false;
        }
     }
//--- If working with the full list of timeframes, fill in the array with strings describing all timeframes
   else
     {
      ArrayResize(used_periods_array,21,21);
      for(int i=0;i<21;i++)
         used_periods_array[i]=TimeframeDescription(TimeframeByEnumIndex(uchar(i+1)));
     }

//--- Add the current chart timeframe to the list of used periods
   bool f=false;
   for(int i=0;i<ArraySize(used_periods_array);i++)
     {
      if(used_periods_array[i]==TimeframeDescription((ENUM_TIMEFRAMES)Period()))
        {
         f=true;
         break;
        }
     }
   //--- If the list of used periods features no timeframe of the current chart
   if(!f)
     {
      //--- Increase the array of used periods by 1 and add the current chart period to it
      ArrayResize(used_periods_array,ArraySize(used_periods_array)+1);
      used_periods_array[ArraySize(used_periods_array)-1]=TimeframeDescription((ENUM_TIMEFRAMES)Period());
     }
//--- All is successful
   return true;
  }
//+------------------------------------------------------------------+
```

**This concludes improvements of library classes.**

It is time to test the development of the Accelerator Oscillator multi-symbol multi-period standard indicator.

### Test

To perform the test, I will use the [indicator from the previous article](https://www.mql5.com/en/articles/8115#node04) and save it in \\MQL5\\Indicators\\TestDoEasy\ **Part47\** as **TestDoEasyPart47.mq5**.

We need to specify which symbol and timeframe to use when calculating the standard AcceleratorOscillator indicator in the indicator settings. The indicator is to display that data in the current chart subwindow.

The indicator header is to be as follows:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart47.mq5 |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
//--- properties
#property indicator_separate_window
#property indicator_buffers 3
#property indicator_plots   1

//--- classes

//--- enums

//--- defines

//--- structures

//--- input variables
sinput   string               InpUsedSymbols    =  "GBPUSD";      // Used symbol (one only)
sinput   ENUM_TIMEFRAMES      InpPeriod         =  PERIOD_M30;    // Used chart period
//---
sinput   bool                 InpUseSounds      =  true;          // Use sounds
//--- indicator buffers

//--- global variables
ENUM_SYMBOLS_MODE    InpModeUsedSymbols=  SYMBOLS_MODE_DEFINES;   // Mode of used symbols list
ENUM_TIMEFRAMES_MODE InpModeUsedTFs    =  TIMEFRAMES_MODE_LIST;   // Mode of used timeframes list
string               InpUsedTFs;                                  // List of used timeframes
CEngine              engine;                                      // CEngine library main object
string               prefix;                                      // Prefix of graphical object names
int                  min_bars;                                    // The minimum number of bars for the indicator calculation
int                  used_symbols_mode;                           // Mode of working with symbols
string               array_used_symbols[];                        // The array for passing used symbols to the library
string               array_used_periods[];                        // The array for passing used timeframes to the library
//+------------------------------------------------------------------+
```

Specify only one symbol and symbol chart period to be used to calculate AC indicator.

In the OnInit() handler, create standard AC indicator featuring the parameters specified in the indicator inputs, its ID (equal to 1) and the buffers for working with it:

```
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Write the name of the working timeframe selected in the settings to the InpUsedTFs variable
   InpUsedTFs=TimeframeDescription(InpPeriod);
//--- Initialize DoEasy library
   OnInitDoEasy();

//--- Set indicator global variables
   prefix=engine.Name()+"_";
   //--- calculate the number of bars of the current period fitting in the maximum used period
   //--- Use the obtained value if it exceeds 2, otherwise use 2
   int num_bars=NumberBarsInTimeframe(InpPeriod);
   min_bars=(num_bars>2 ? num_bars : 2);

//--- Check and remove remaining indicator graphical objects
   if(IsPresentObectByPrefix(prefix))
      ObjectsDeleteAll(0,prefix);

//--- Create the button panel

//--- Check playing a standard sound using macro substitutions
   engine.PlaySoundByDescription(SND_OK);
//--- Wait for 600 milliseconds
   engine.Pause(600);
   engine.PlaySoundByDescription(SND_NEWS);

//--- indicator buffers mapping
//--- Create all the necessary buffer objects for constructing AO
   engine.BufferCreateAC(InpUsedSymbols,InpPeriod,1);

//--- Check the number of buffers specified in the 'properties' block
   if(engine.BuffersPropertyPlotsTotal()!=indicator_plots)
      Alert(TextByLanguage("Внимание! Значение \"indicator_plots\" должно быть ","Attention! Value of \"indicator_plots\" should be "),engine.BuffersPropertyPlotsTotal());
   if(engine.BuffersPropertyBuffersTotal()!=indicator_buffers)
      Alert(TextByLanguage("Внимание! Значение \"indicator_buffers\" должно быть ","Attention! Value of \"indicator_buffers\" should be "),engine.BuffersPropertyBuffersTotal());

//--- Create the color array and set non-default colors to all buffers within the collection
   color array_colors[]={clrGreen,clrRed,clrGray};
   engine.BuffersSetColors(array_colors);

//--- Display short descriptions of created indicator buffers
   engine.BuffersPrintShort();
//--- Set the short name for the indicator and bit depth
   IndicatorSetString(INDICATOR_SHORTNAME,"AC("+InpUsedSymbols+","+TimeframeDescription(InpPeriod)+")");
   IndicatorSetInteger(INDICATOR_DIGITS,(int)SymbolInfoInteger(InpUsedSymbols,SYMBOL_DIGITS)+2);
//--- Successful
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

In OnCalculate(), first prepare AC indicator calculated buffer data. Next, in the main indicator loop, fill in drawn buffer data on the current chart with data from the AC indicator calculated buffer:

```
//+------------------------------------------------------------------+
//| OnCalculate code block for working with the library:             |
//+------------------------------------------------------------------+
//--- Pass the current symbol data from OnCalculate() to the price structure and set the "as timeseries" flag to the arrays
   CopyDataAsSeries(rates_total,prev_calculated,time,open,high,low,close,tick_volume,volume,spread);

//--- Check for the minimum number of bars for calculation
   if(rates_total<min_bars || Point()==0) return 0;
//--- Handle the Calculate event in the library
//--- If the OnCalculate() method of the library returns zero, not all timeseries are ready - leave till the next tick
   if(engine.0)
      return 0;

//--- If working in the tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer(rates_data);   // Working in the library timer
      engine.EventsHandling();      // Working with library events
     }
//+------------------------------------------------------------------+
//| OnCalculate code block for working with the indicator:           |
//+------------------------------------------------------------------+
//--- Check and calculate the number of calculated bars
//--- If limit = 0, there are no new bars - calculate the current one
//--- If limit = 1, a new bar has appeared - calculate the first and the current ones
//--- limit > 1 means the first launch or changes in history - the full recalculation of all data
   int limit=rates_total-prev_calculated;

//--- Recalculate the entire history
   if(limit>1)
     {
      limit=rates_total-1;
      engine.BuffersInitPlots();
      engine.BuffersInitCalculates();
     }

//--- Prepare data
   int bars_total=engine.SeriesGetBarsTotal(InpUsedSymbols,InpPeriod);
   int total_copy=(limit<min_bars ? min_bars : fmin(limit,bars_total));

//--- Fill in the calculated buffer with AO data
   CArrayObj *list=engine.GetBuffersCollection().GetListBuffersWithID();
   if(list!=NULL)
     {
      for(int i=0;i<list.Total();i++)
        {
         CBuffer *buff=list.At(i);
         if(buff==NULL || buff.TypeBuffer()==BUFFER_TYPE_DATA || buff.IndicatorType()==WRONG_VALUE)
            continue;
         CSeriesDE *series=engine.SeriesGetSeries(buff.Symbol(),buff.Timeframe());
         if(series==NULL)
            return 0;
         ulong used_data=series.AvailableUsedData();
         int copied=engine.GetBuffersCollection().PreparingDataBufferStdInd(IND_AC,1,(int)used_data);
         if(copied<(int)used_data)
            return 0;
        }
     }

//--- Calculate the indicator
   CBar *bar=NULL;         // Bar object for defining the candle direction
   uchar color_index=0;    // Color index to be set for the buffer depending on the candle direction

//--- Main calculation loop of the indicator
   for(int i=limit; i>WRONG_VALUE && !IsStopped(); i--)
     {
      engine.GetBuffersCollection().SetDataBufferStdInd(IND_AC,1,i,time[i]);
     }
//--- return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

This is all we need to calculate and display the standard AC indicator on the current chart calculated on any symbol/timeframe.

The standard indicator data preparation block will be improved (it is not optimal in the current implementation, as it has been developed only to check the concept) and moved to the library in subsequent articles.

The full indicator code is provided in the files attached below.

Compile the indicator and launch it on EURUSD M1 after setting GBPUSD M5 in the indicator settings, which means displaying AC indicator data (calculated on GBPUSD M5) on the current EURUSD minute chart:

![](https://c.mql5.com/2/39/Ig33RjnXVV.gif)

GBPUSD M5 with standard AC indicator is also opened for comparison.

### What's next?

In the next article, I will continue the development of multi-symbol multi-period standard indicators.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave your questions, comments and suggestions in the comments.

Please keep in mind that here I have developed the MQL5 test indicator for MetaTrader 5.

The attached files are intended only for MetaTrader 5. The current library version has not been tested in MetaTrader 4.

After developing and testing the functionality for working with indicator buffers, I will try to implement some MQL5 features in MetaTrader 4.

[Back to contents](https://www.mql5.com/en/articles/8207#node00)

**Previous articles within the series:**

[Timeseries in DoEasy library (part 35): Bar object and symbol timeseries list](https://www.mql5.com/en/articles/7594)

[Timeseries in DoEasy library (part 36): Object of timeseries for all used symbol periods](https://www.mql5.com/en/articles/7627)

[Timeseries in DoEasy library (part 37): Timeseries collection - database of timeseries by symbols and periods](https://www.mql5.com/en/articles/7663)

[Timeseries in DoEasy library (part 38): Timeseries collection - real-time updates and accessing data from the program](https://www.mql5.com/en/articles/7695)

[Timeseries in DoEasy library (part 39): Library-based indicators - preparing data and timeseries events](https://www.mql5.com/en/articles/7724)

[Timeseries in DoEasy library (part 40): Library-based indicators - updating data in real time](https://www.mql5.com/en/articles/7771)

[Timeseries in DoEasy library (part 41): Sample multi-symbol multi-period indicator](https://www.mql5.com/en/articles/7804)

[Timeseries in DoEasy library (part 42): Abstract indicator buffer object class](https://www.mql5.com/en/articles/7821)

[Timeseries in DoEasy library (part 43): Classes of indicator buffer objects](https://www.mql5.com/en/articles/7868)

[Timeseries in DoEasy library (part 44): Collection class of indicator buffer objects](https://www.mql5.com/en/articles/7886)

[Timeseries in DoEasy library (part 45): Multi-period indicator buffers](https://www.mql5.com/en/articles/8023)

[Timeseries in DoEasy library (part 46): Multi-period multi-symbol indicator buffers](https://www.mql5.com/en/articles/8115)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8207](https://www.mql5.com/ru/articles/8207)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8207.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/8207/mql5.zip "Download MQL5.zip")(3748.84 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/355354)**
(4)


![tuma_news](https://c.mql5.com/avatar/avatar_na2.png)

**[tuma\_news](https://www.mql5.com/en/users/tuma_news)**
\|
26 Jul 2020 at 18:09

Artem, hi there !

Thanks for your contribution to kodobase and for your work !

I understand correctly that this is not a template, in which I can enter any symbol and time I need ?

It is suitable only for inbuilt indicators in MT ?

What about [custom indicators](https://www.mql5.com/en/articles/5 "Article Switching to New Rails: Custom Indicators in MQL5")?

I would very much like to have a template, in which you can enter any indicator, including custom indicators.

Thank you!

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
26 Jul 2020 at 18:19

**tuma\_news:**

Artem, hi !

Thanks for your contribution to kodobase and for the work done !

I understand correctly that this is not a template, in which I can enter any symbol and time I need ?

It is suitable only for inbuilt indicators in MT ?

What about [custom indicators](https://www.mql5.com/en/articles/5 "Article Switching to New Rails: Custom Indicators in MQL5")?

I would like to have a template where any indicator, including custom indicators, can be inserted.

Thank you!

The library is not a template and is intended to help you develop your own programmes for MetaTrader.

The article describes how to use the library to create a multi-period multi-character custom indicator, which displays data of a standard indicator (on the example of AC) with a symbol and period different from the symbol/period of the current chart. Other standard indicators will be described in subsequent articles.

And it is the standard indicators that this part is devoted to.

To work with third-party (not created with the help of the library) custom indicators, you need a slightly different approach. And this will be described in further articles.

![SergeiKrasnoff](https://c.mql5.com/avatar/2020/7/5F242D86-206E.jpg)

**[SergeiKrasnoff](https://www.mql5.com/en/users/sergeikrasnoff)**
\|
31 Jul 2020 at 14:42

Thank you for helping the newbies


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
31 Jul 2020 at 14:47

**SergeiKrasnoff:**

Thank you for helping newcomers

You're welcome.


![A system of voice notifications for trade events and signals](https://c.mql5.com/2/39/logo.png)[A system of voice notifications for trade events and signals](https://www.mql5.com/en/articles/8111)

Nowadays, voice assistants play a prominent role in human life, as we often use navigators, voice search and translators. In this article, I will try to develop a simple and user friendly system of voice notifications for various trade events, market states or signals generated by trading signals.

![On Methods to Detect Overbought/Oversold Zones. Part I](https://c.mql5.com/2/39/logo_200x200.png)[On Methods to Detect Overbought/Oversold Zones. Part I](https://www.mql5.com/en/articles/7782)

Overbought/oversold zones characterize a certain state of the market, differentiating through weaker changes in the prices of securities. This adverse change in the synamics is pronounced most at the final stage in the development of trends of any scales. Since the profit value in trading depends directly on the capability of covering as large trend amplitude as possible, the accuracy of detecting such zones is a key task in trading with any securities whatsoever.

![Timeseries in DoEasy library (part 48): Multi-period multi-symbol indicators on one buffer in a subwindow](https://c.mql5.com/2/40/MQL5-avatar-doeasy-library.png)[Timeseries in DoEasy library (part 48): Multi-period multi-symbol indicators on one buffer in a subwindow](https://www.mql5.com/en/articles/8257)

The article considers an example of creating multi-symbol multi-period standard indicators using a single indicator buffer for construction and working in the indicator subwindow. I am going to prepare the library classes for working with standard indicators working in the program main window and having more than one buffer for displaying their data.

![Probability theory and mathematical statistics with examples (part I): Fundamentals and elementary theory](https://c.mql5.com/2/39/Probability_theory_1.png)[Probability theory and mathematical statistics with examples (part I): Fundamentals and elementary theory](https://www.mql5.com/en/articles/8038)

Trading is always about making decisions in the face of uncertainty. This means that the results of the decisions are not quite obvious at the time these decisions are made. This entails the importance of theoretical approaches to the construction of mathematical models allowing us to describe such cases in meaningful manner.

[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=rtslrdebrexznwhulstdbnepkrdurlqc&ssn=1769186105940125189&ssn_dr=4&ssn_sr=0&fv_date=1769186105&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8207&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Timeseries%20in%20DoEasy%20library%20(part%2047)%3A%20Multi-period%20multi-symbol%20standard%20indicators%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918610922613352&fz_uniq=5070403171827782968&sv=2552)

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