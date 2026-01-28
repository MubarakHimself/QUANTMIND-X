---
title: Timeseries in DoEasy library (part 42): Abstract indicator buffer object class
url: https://www.mql5.com/en/articles/7821
categories: Trading Systems, Indicators
relevance_score: 3
scraped_at: 2026-01-23T19:36:04.761885
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/7821&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070416310132741490)

MetaTrader 5 / Trading systems


### Contents

- [Concept](https://www.mql5.com/en/articles/7821#node01)
- [CBuffer abstract buffer class](https://www.mql5.com/en/articles/7821#node02)
- [Checking creation of buffer objects in the indicator](https://www.mql5.com/en/articles/7821#node03)
- [What's next?](https://www.mql5.com/en/articles/7821#node04)


### Concept

[In the previous article,](https://www.mql5.com/en/articles/7804) I have developed a sample indicator using DoEasy library timeseries objects for its work. In order to store and access the buffer data, I have created the buffer structure featuring all the necessary data for correct identification of the buffer belonging to a chart symbol and period for plotting and drawing indicator lines in the form of candlesticks. However, it would be inconvenient and impractical to create the structures, whose fields correspond to the required type of line drawing, for each indicator. It is much more convenient to use the object class of the indicator buffer which allows us to easily create any types of buffers by style and drawing method.

I will start developing such a tool in this article.

The object structure is to consist of the basic indicator buffer class (the so called "abstract buffer object") containing all general properties inherent in all indicator buffers regardless of their drawing type. The buffer object classes containing the clarifying data on specific buffer type are to be inherited from this abstract buffer. These descendant classes will precisely define the drawing type and feature individual properties unique to this type of indicator buffer.

In the current article, we will write the object class of the abstract indicator buffer. The object is to contain all indicator buffer properties from the [ENUM\_PLOT\_PROPERTY\_INTEGER,](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles#enum_plot_property_integer) ENUM\_PLOT\_PROPERTY\_DOUBLE and [ENUM\_PLOT\_PROPERTY\_STRING](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles#enum_plot_property_integer) enumerations, as well as a few additional properties:

- for specifying a buffer type:

  - data buffer,
  - calculated buffer;


- buffer status (define the drawing type of indicator lines)

descendant objects are to be sorted by the buffer status — a separate descendant class of the abstract buffer class is created for each status;
- buffer working symbol:

  - current,
  - strictly specified;

- buffer timeframe:

  - current,
  - strictly specified;

- The serial number of the drawn buffer is a buffer number displayed in the terminal's DataWindow (these numbers do not coincide with the indices of double arrays assigned to the buffer);
- The buffer usage flag — the indicator is to make it possible to enable/disable the display of buffer lines and values in the DataWindow by checking/unchecking the flag;
- The index of the base data buffer — auxiliary data on the index of the first array from all arrays used to draw the buffer lines;

- The amount of buffer data — auxiliary data on the real amount of arrays used to draw buffer graphics. It is necessary to calculate the color buffer index and the index of the next array the index can be assigned to as the base data buffer;
- Color buffer index is an auxiliary data on the array index used as the color buffer;
- The free array index for assigning as the next indicator buffer — auxiliary data on the calculated index of the base data buffer for the next buffer object.

As a result, after creating all the necessary classes for working with the indicator buffers, we obtain the ability to simply "tell" our program "create the buffer of this type", and the library will create the buffer and any subsequent buffers without the need to independently declare arrays, assign their properties and binding to indicator buffers. We simply obtain the buffer list, from which we will access any of the previously created indicator buffers and buffer data.

Since MQL5 provides the ability to create two similar [drawing styles](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles) — monochrome and colored ones, all buffers created by the library are to be colored. If you want to create a monochrome line, a single color for all data displayed by the buffer is set.

When using different colors for coloring the bars, the specified number of colors will be used to display the line on each bar.

### CBuffer abstract buffer class

For the class operation, we will need text messages with descriptions of buffer properties.

Open \\MQL5\\Include\\DoEasy\ **Datas.mqh** and add indices of new messages to it:

```
   MSG_LIB_TEXT_WAIT,                                 // Wait
   MSG_LIB_TEXT_END,                                  // End
   MSG_LIB_TEXT_PERIOD_CURRENT,                       // Current chart period

```

...

```
   MSG_LIB_TEXT_TS_TEXT_ATTEMPT,                      // Attempt:
   MSG_LIB_TEXT_TS_TEXT_WAIT_FOR_SYNC,                // Waiting for data synchronization ...

//--- CBuffer
   MSG_LIB_TEXT_BUFFER_TEXT_INDEX_BASE,               // Base data buffer index
   MSG_LIB_TEXT_BUFFER_TEXT_INDEX_PLOT,               // Plotted buffer serial number
   MSG_LIB_TEXT_BUFFER_TEXT_INDEX_COLOR,              // Color buffer index
   MSG_LIB_TEXT_BUFFER_TEXT_NUM_DATAS,                // Number of data buffers
   MSG_LIB_TEXT_BUFFER_TEXT_INDEX_NEXT,               // Index of the free array to be assigned as the next indicator buffer
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
   MSG_LIB_TEXT_BUFFER_TEXT_COLOR_NUM,                // Number of colors
   MSG_LIB_TEXT_BUFFER_TEXT_COLOR,                    // Drawing color
   MSG_LIB_TEXT_BUFFER_TEXT_EMPTY_VALUE,              // Empty value for plotting where nothing will be drawn
   MSG_LIB_TEXT_BUFFER_TEXT_SYMBOL,                   // Buffer symbol
   MSG_LIB_TEXT_BUFFER_TEXT_LABEL,                    // Name of the graphical indicator series displayed in DataWindow
   MSG_LIB_TEXT_BUFFER_TEXT_STATUS_NAME,              // Indicator buffer with graphical construction type

   MSG_LIB_TEXT_BUFFER_TEXT_STATUS_NONE,              // No drawing
   MSG_LIB_TEXT_BUFFER_TEXT_STATUS_FILLING,           // Color filling between two levels
   MSG_LIB_TEXT_BUFFER_TEXT_STATUS_LINE,              // Line
   MSG_LIB_TEXT_BUFFER_TEXT_STATUS_HISTOGRAM,         // Histogram from the zero line
   MSG_LIB_TEXT_BUFFER_TEXT_STATUS_ARROW,             // Drawing with arrows
   MSG_LIB_TEXT_BUFFER_TEXT_STATUS_SECTION,           // Segments
   MSG_LIB_TEXT_BUFFER_TEXT_STATUS_HISTOGRAM2,        // Histogram on two indicator buffers
   MSG_LIB_TEXT_BUFFER_TEXT_STATUS_ZIGZAG,            // Zigzag
   MSG_LIB_TEXT_BUFFER_TEXT_STATUS_BARS,              // Display as bars
   MSG_LIB_TEXT_BUFFER_TEXT_STATUS_CANDLES,           // Display as candles

   MSG_LIB_TEXT_BUFFER_TEXT_TYPE_CALCULATE,           // Calculated buffer
   MSG_LIB_TEXT_BUFFER_TEXT_TYPE_DATA,                // Colored data buffer

   MSG_LIB_TEXT_BUFFER_TEXT_STYLE_SOLID,              // Solid line
   MSG_LIB_TEXT_BUFFER_TEXT_STYLE_DASH,               // Dashed line
   MSG_LIB_TEXT_BUFFER_TEXT_STYLE_DOT,                // Dotted line
   MSG_LIB_TEXT_BUFFER_TEXT_STYLE_DASHDOT,            // Dot-dash line
   MSG_LIB_TEXT_BUFFER_TEXT_STYLE_DASHDOTDOT,         // Dash - two dots

  };
//+------------------------------------------------------------------+
```

and message texts corresponding to newly added indices:

```
   {"Ожидание","Wait"},
   {"Окончание","End"},
   {"Текущий период графика","Current chart period"},

```

...

```
   {"Попытка: ","Attempt: "},
   {"Ожидание синхронизации данных ...","Waiting for data synchronization ..."},

   {"Индекс базового буфера данных","Index of Base data buffer"},
   {"Порядковый номер рисуемого буфера","Plot buffer sequence number"},
   {"Индекс буфера цвета","Color buffer index"},
   {"Количество буферов данных","Number of data buffers"},
   {"Индекс массива для назначения следующим индикаторным буфером","Array index for assignment as the next indicator buffer"},
   {"Период данных буфера (таймфрейм)","Buffer data Period (Timeframe)"},
   {"Статус буфера","Buffer status"},
   {"Тип буфера","Buffer type"},
   {"Активен","Active"},
   {"Код стрелки","Arrow code"},
   {"Смещение стрелок по вертикали","Vertical shift of arrows"},
   {"Количество начальных баров без отрисовки и значений в DataWindow","Number of initial bars without drawing and values in the DataWindow"},
   {"Тип графического построения","Type of graphical construction"},
   {"Отображение значений построения в окне DataWindow","Display construction values in the DataWindow"},
   {"Сдвиг графического построения индикатора по оси времени в барах","Shift of indicator plotting along the time axis in bars"},
   {"Стиль линии отрисовки","Drawing line style "},
   {"Толщина линии отрисовки","The thickness of the drawing line"},
   {"Количество цветов","The number of colors"},
   {"Цвет отрисовки","The index of a buffer containing the drawing color"},
   {"Пустое значение для построения, для которого нет отрисовки","An empty value for plotting, for which there is no drawing"},
   {"Символ буфера","Buffer Symbol"},
   {"Имя индикаторной графической серии, отображаемое в окне DataWindow","The name of the indicator graphical series to display in the DataWindow"},
   {"Индикаторный буфер с типом графического построения","Indicator buffer with graphic plot type"},

   {"Нет отрисовки","No drawing"},
   {"Цветовая заливка между двумя уровнями","Color fill between the two levels"},
   {"Линия","Line"},
   {"Гистограмма от нулевой линии","Histogram from the zero line"},
   {"Отрисовка стрелками","Drawing arrows"},
   {"Отрезки","Section"},
   {"Гистограмма на двух индикаторных буферах","Histogram of the two indicator buffers"},
   {"Зигзаг","Zigzag"},
   {"Отображение в виде баров","Display as a sequence of bars"},
   {"Отображение в виде свечей","Display as a sequence of candlesticks"},

   {"Расчётный буфер","Calculated buffer"},
   {"Цветной буфер данных","Colored Data buffer"},

   {"Сплошная линия","Solid line"},
   {"Прерывистая линия","Broken line"},
   {"Пунктирная линия","Dotted line"},
   {"Штрих-пунктирная линия","Dash-dot line"},
   {"Штрих - две точки","Dash - two points"},

  };
//+---------------------------------------------------------------------+
```

When creating library objects to be stored in the object collection, ID of the appropriate collection is assigned to each object. The same is done for buffer objects: create indicator buffer collection ID and assign it to the abstract buffer object. Accordingly, each object inheriting from an abstract buffer will have an ID indicating that it belongs to the buffer collection.

To create an abstract buffer object, we need to define and describe all its properties. Using these properties, we are always able to find the required buffer object in the buffer collection for further work.

Open \\MQL5\\Include\\DoEasy\ **Defines.mqh** and add the indicator buffer collection ID:

```
//--- Collection list IDs
#define COLLECTION_HISTORY_ID          (0x777A)                   // Historical collection list ID
#define COLLECTION_MARKET_ID           (0x777B)                   // Market collection list ID
#define COLLECTION_EVENTS_ID           (0x777C)                   // Event collection list ID
#define COLLECTION_ACCOUNT_ID          (0x777D)                   // Account collection list ID
#define COLLECTION_SYMBOLS_ID          (0x777E)                   // Symbol collection list ID
#define COLLECTION_SERIES_ID           (0x777F)                   // Timeseries collection list ID
#define COLLECTION_BUFFERS_ID          (0x7780)                   // Indicator buffer collection list ID
//--- Data parameters for file operations
```

At the very end of the file, add enumerations of the buffer object status and its type, define integer, real and string object properties, as well as write possible criteria of buffer object sorting in their collection list:

```
//+------------------------------------------------------------------+
//| Data for working with indicator buffers                          |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Abstract buffer status (by drawing style)                        |
//+------------------------------------------------------------------+
enum ENUM_BUFFER_STATUS
  {
   BUFFER_STATUS_NONE,                                         // No drawing
   BUFFER_STATUS_FILLING,                                      // Color filling between two levels (MQL5)
   BUFFER_STATUS_LINE,                                         // Line
   BUFFER_STATUS_HISTOGRAM,                                    // Histogram from the zero line
   BUFFER_STATUS_ARROW,                                        // Drawing with arrows
   BUFFER_STATUS_SECTION,                                      // Segments
   BUFFER_STATUS_HISTOGRAM2,                                   // Histogram on two indicator buffers
   BUFFER_STATUS_ZIGZAG,                                       // Zigzag style
   BUFFER_STATUS_BARS,                                         // Display as bars (MQL5)
   BUFFER_STATUS_CANDLES,                                      // Display as candles (MQL5)
  };
//+------------------------------------------------------------------+
//| Buffer type                                                      |
//+------------------------------------------------------------------+
enum ENUM_BUFFER_TYPE
  {
   BUFFER_TYPE_CALCULATE,                                      // Calculated buffer
   BUFFER_TYPE_DATA,                                           // Colored data buffer
  };
//+------------------------------------------------------------------+
//| Buffer integer properties                                        |
//+------------------------------------------------------------------+
enum ENUM_BUFFER_PROP_INTEGER
  {
   BUFFER_PROP_INDEX_PLOT = 0,                                 // Plotted buffer serial number
   BUFFER_PROP_STATUS,                                         // Buffer status (by drawing style) from the ENUM_BUFFER_STATUS enumeration
   BUFFER_PROP_TYPE,                                           // Buffer type (from the ENUM_BUFFER_TYPE enumeration)
   BUFFER_PROP_TIMEFRAME,                                      // Buffer period data (timeframe)
   BUFFER_PROP_ACTIVE,                                         // Buffer usage flag
   BUFFER_PROP_ARROW_CODE,                                     // Arrow code for DRAW_ARROW style
   BUFFER_PROP_ARROW_SHIFT,                                    // The vertical shift of the arrows for DRAW_ARROW style
   BUFFER_PROP_DRAW_BEGIN,                                     // The number of initial bars that are not drawn and values in DataWindow
   BUFFER_PROP_SHOW_DATA,                                      // Flag of displaying construction values in DataWindow
   BUFFER_PROP_DRAW_TYPE,                                      // Graphical construction type (from the ENUM_DRAW_TYPE enumeration)
   BUFFER_PROP_SHIFT,                                          // Indicator graphical construction shift by time axis in bars
   BUFFER_PROP_LINE_STYLE,                                     // Line style
   BUFFER_PROP_LINE_WIDTH,                                     // Line width
   BUFFER_PROP_COLOR_INDEXES,                                  // Number of colors
   BUFFER_PROP_COLOR,                                          // Drawing color
   BUFFER_PROP_NUM_DATAS,                                      // Number of data buffers
   BUFFER_PROP_INDEX_BASE,                                     // Base data buffer index
   BUFFER_PROP_INDEX_COLOR,                                    // Color buffer index
   BUFFER_PROP_INDEX_NEXT,                                     // Index of the free array to be assigned as the next indicator buffer
  };
#define BUFFER_PROP_INTEGER_TOTAL (19)                         // Total number of integer bar properties
#define BUFFER_PROP_INTEGER_SKIP  (6)                          // Number of buffer properties not used in sorting
//+------------------------------------------------------------------+
//| Buffer real properties                                           |
//+------------------------------------------------------------------+
enum ENUM_BUFFER_PROP_DOUBLE
  {
   BUFFER_PROP_EMPTY_VALUE = BUFFER_PROP_INTEGER_TOTAL,        // Empty value for plotting where nothing will be drawn
  };
#define BUFFER_PROP_DOUBLE_TOTAL  (1)                          // Total number of real buffer properties
#define BUFFER_PROP_DOUBLE_SKIP   (0)                          // Number of buffer properties not used in sorting
//+------------------------------------------------------------------+
//| Buffer string properties                                         |
//+------------------------------------------------------------------+
enum ENUM_BUFFER_PROP_STRING
  {
   BUFFER_PROP_SYMBOL = (BUFFER_PROP_INTEGER_TOTAL+BUFFER_PROP_DOUBLE_TOTAL), // Buffer symbol
   BUFFER_PROP_LABEL,                                          // Name of the graphical indicator series displayed in DataWindow
  };
#define BUFFER_PROP_STRING_TOTAL  (2)                          // Total number of string buffer properties
//+------------------------------------------------------------------+
//| Possible buffer sorting criteria                                 |
//+------------------------------------------------------------------+
#define FIRST_BUFFER_DBL_PROP          (BUFFER_PROP_INTEGER_TOTAL-BUFFER_PROP_INTEGER_SKIP)
#define FIRST_BUFFER_STR_PROP          (BUFFER_PROP_INTEGER_TOTAL-BUFFER_PROP_INTEGER_SKIP+BUFFER_PROP_DOUBLE_TOTAL-BUFFER_PROP_DOUBLE_SKIP)
enum ENUM_SORT_BUFFER_MODE
  {
//--- Sort by integer properties
   SORT_BY_BUFFER_INDEX_PLOT = 0,                              // Sort by the plotted buffer serial number
   SORT_BY_BUFFER_STATUS,                                      // Sort by buffer drawing style (status) from the ENUM_BUFFER_STATUS enumeration
   SORT_BY_BUFFER_TYPE,                                        // Sort by buffer type (from the ENUM_BUFFER_TYPE enumeration)
   SORT_BY_BUFFER_TIMEFRAME,                                   // Sort by the buffer data period (timeframe)
   SORT_BY_BUFFER_ACTIVE,                                      // Sort by the buffer usage flag
   SORT_BY_BUFFER_ARROW_CODE,                                  // Sort by the arrow code for DRAW_ARROW style
   SORT_BY_BUFFER_ARROW_SHIFT,                                 // Sort by the vertical shift of the arrows for DRAW_ARROW style
   SORT_BY_BUFFER_DRAW_BEGIN,                                  // Sort by the number of initial bars that are not drawn and values in DataWindow
   SORT_BY_BUFFER_SHOW_DATA,                                   // Sort by the flag of displaying construction values in DataWindow
   SORT_BY_BUFFER_DRAW_TYPE,                                   // Sort by graphical construction type (from the ENUM_DRAW_TYPE enumeration)
   SORT_BY_BUFFER_SHIFT,                                       // Sort by the indicator graphical construction shift by time axis in bars
   SORT_BY_BUFFER_LINE_STYLE,                                  // Sort by the line style
   SORT_BY_BUFFER_LINE_WIDTH,                                  // Sort by the line width
   SORT_BY_BUFFER_COLOR_INDEXES,                               // Sort by a number of attempts
   SORT_BY_BUFFER_COLOR,                                       // Sort by the drawing color
//--- Sort by real properties
   SORT_BY_BUFFER_EMPTY_VALUE = FIRST_BUFFER_DBL_PROP,         // Sort by the empty value for plotting where nothing will be drawn
//--- Sort by string properties
   SORT_BY_BUFFER_SYMBOL = FIRST_BUFFER_STR_PROP,              // Sort by the buffer symbol
   SORT_BY_BUFFER_LABEL,                                       // Sort by the name of the graphical indicator series displayed in DataWindow
  };
//+------------------------------------------------------------------+
```

All these properties, as well as their definition, assignment and usage, are standard for the library. We have considered them many times, so there is no point in delving into each individual enumeration and macro substitution here. I hope, you are already familiar with their purpose and no explanations are required. However, if you have any questions, feel free to ask them in the comments below.

**We have prepared all the necessary data. Now it is time to start creating the abstract buffer object class.**

In \\MQL5\\Include\\DoEasy\ **Objects\**, create **Indicators\** folder containing the **Buffer.mqh** file of the CBuffer class.

[The class of the base object of all CBaseObj library objects](https://www.mql5.com/en/articles/7071) serves as the base class for the abstract buffer object class.

Include the class file to the buffer class file:

```
//+------------------------------------------------------------------+
//|                                                       Buffer.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\..\Objects\BaseObj.mqh"
//+------------------------------------------------------------------+
//| Abstract indicator buffer class                                  |
//+------------------------------------------------------------------+
class CBuffer : public CBaseObj
  {
```

The set of basic methods of library objects is standard for each object. It allows working with the properties inherent in each object in the same way for each individual object. I have considered the object construction principles [at the very beginning of the library description](https://www.mql5.com/en/articles/5654).

Here everything is standard as well: three object property arrays (integer, real and string ones), methods of returning real indices of real and string properties, as well as methods of setting, receiving and  retrieving the description of the specified property from these arrays. Three virtual methods returning the flags of using specified properties and two comparison methods — for searching and sorting in the collection by the specified property as well as for comparing two objects for equality. Two constructors — the default and closed parametric one and two methods for displaying all buffer object properties and its short description in the journal:

```
//+------------------------------------------------------------------+
//| Abstract indicator buffer class                                  |
//+------------------------------------------------------------------+
class CBuffer : public CBaseObj
  {
private:
   long              m_long_prop[BUFFER_PROP_INTEGER_TOTAL];                     // Integer properties
   double            m_double_prop[BUFFER_PROP_DOUBLE_TOTAL];                    // Real properties
   string            m_string_prop[BUFFER_PROP_STRING_TOTAL];                    // String properties
//--- Return the index of the array the buffer's (1) double and (2) string properties are located at
   int               IndexProp(ENUM_BUFFER_PROP_DOUBLE property)           const { return(int)property-BUFFER_PROP_INTEGER_TOTAL;                           }
   int               IndexProp(ENUM_BUFFER_PROP_STRING property)           const { return(int)property-BUFFER_PROP_INTEGER_TOTAL-BUFFER_PROP_DOUBLE_TOTAL;  }
//--- Set the graphical construction type
   void              SetDrawType(void);

public:
//--- Array of the (1) drawn indicator buffer and (2) color buffer
   double            DataArray[];
   double            ColorArray[];

//--- Set buffer's (1) integer, (2) real and (3) string properties
   void              SetProperty(ENUM_BUFFER_PROP_INTEGER property,long value)   { this.m_long_prop[property]=value;                                        }
   void              SetProperty(ENUM_BUFFER_PROP_DOUBLE property,double value)  { this.m_double_prop[this.IndexProp(property)]=value;                      }
   void              SetProperty(ENUM_BUFFER_PROP_STRING property,string value)  { this.m_string_prop[this.IndexProp(property)]=value;                      }
//--- Return (1) integer, (2) real and (3) string buffer properties from the properties array
   long              GetProperty(ENUM_BUFFER_PROP_INTEGER property)        const { return this.m_long_prop[property];                                       }
   double            GetProperty(ENUM_BUFFER_PROP_DOUBLE property)         const { return this.m_double_prop[this.IndexProp(property)];                     }
   string            GetProperty(ENUM_BUFFER_PROP_STRING property)         const { return this.m_string_prop[this.IndexProp(property)];                     }
//--- Get description of buffer's (1) integer, (2) real and (3) string properties
   string            GetPropertyDescription(ENUM_BUFFER_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_BUFFER_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_BUFFER_PROP_STRING property);
//--- Return the flag of the buffer supporting the property
   virtual bool      SupportProperty(ENUM_BUFFER_PROP_INTEGER property)          { return true; }
   virtual bool      SupportProperty(ENUM_BUFFER_PROP_DOUBLE property)           { return true; }
   virtual bool      SupportProperty(ENUM_BUFFER_PROP_STRING property)           { return true; }

//--- Compare CBuffer objects by all possible properties (for sorting the lists by a specified buffer object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CBuffer objects by all properties (to search for equal buffer objects)
   bool              IsEqual(CBuffer* compared_obj) const;

//--- Default constructor
                     CBuffer(void){;}
//protected:
//--- Protected parametric constructor
                     CBuffer(ENUM_BUFFER_STATUS status_buffer,ENUM_BUFFER_TYPE buffer_type,const uint index_plot,const uint index_base_array);
public:
//--- Send description of buffer properties to the journal (full_prop=true - all properties, false - only supported ones)
   void              Print(const bool full_prop=false);
//--- Display a short buffer description in the journal (implementation in the descendants)
   virtual void      PrintShort(void) {;}

```

Standard data and methods are highlighted here.

The protected parametric constructor should be present in the protected class section but, in the current article, the constructor will be made public to check the created object operation.

This is why the protected: access specifier has been commented out.

One private method of setting the buffer graphical construction type, two public arrays for their binding as data array and color array of the indicator buffer have remained undescribed:

```
//--- Set the graphical construction type
   void              SetDrawType(void);

public:
//--- Array of the (1) drawn indicator buffer and (2) color buffer
   double            DataArray[];
   double            ColorArray[];
```

Depending on the buffer status (inherited objects are "bound" to its status), we will define the type of the buffer graphical construction.

If we look closely at the buffer status enumeration constants

```
//+------------------------------------------------------------------+
//| Abstract buffer status (by drawing style)                        |
//+------------------------------------------------------------------+
enum ENUM_BUFFER_STATUS
  {
   BUFFER_STATUS_NONE,                                         // No drawing
   BUFFER_STATUS_FILLING,                                      // Color filling between two levels (MQL5)
   BUFFER_STATUS_LINE,                                         // Line
   BUFFER_STATUS_HISTOGRAM,                                    // Histogram from the zero line
   BUFFER_STATUS_ARROW,                                        // Drawing with arrows
   BUFFER_STATUS_SECTION,                                      // Segments
   BUFFER_STATUS_HISTOGRAM2,                                   // Histogram on two indicator buffers
   BUFFER_STATUS_ZIGZAG,                                       // Zigzag style
   BUFFER_STATUS_BARS,                                         // Display as bars (MQL5)
   BUFFER_STATUS_CANDLES,                                      // Display as candles (MQL5)
  };
//+------------------------------------------------------------------+
```

and compare the order of the constants with the order of constants of almost the same name in the [ENUM\_DRAW\_TYPE](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles) enumeration using a simple loop by the number of constants in the enumeration,

```
for(int i=0;i<18;i++)
   Print(EnumToString((ENUM_DRAW_TYPE)i)," = ",i);

2020.04.15 12:51:53.725 DRAW_NONE = 0
2020.04.15 12:51:53.725 DRAW_LINE = 1
2020.04.15 12:51:53.725 DRAW_HISTOGRAM = 2
2020.04.15 12:51:53.725 DRAW_ARROW = 3
2020.04.15 12:51:53.725 DRAW_SECTION = 4
2020.04.15 12:51:53.725 DRAW_HISTOGRAM2 = 5
2020.04.15 12:51:53.725 DRAW_ZIGZAG = 6
2020.04.15 12:51:53.725 DRAW_FILLING = 7
2020.04.15 12:51:53.725 DRAW_BARS = 8
2020.04.15 12:51:53.725 DRAW_CANDLES = 9
2020.04.15 12:51:53.725 DRAW_COLOR_LINE = 10
2020.04.15 12:51:53.725 DRAW_COLOR_HISTOGRAM = 11
2020.04.15 12:51:53.725 DRAW_COLOR_ARROW = 12
2020.04.15 12:51:53.725 DRAW_COLOR_SECTION = 13
2020.04.15 12:51:53.725 DRAW_COLOR_HISTOGRAM2 = 14
2020.04.15 12:51:53.725 DRAW_COLOR_ZIGZAG = 15
2020.04.15 12:51:53.725 DRAW_COLOR_BARS = 16
2020.04.15 12:51:53.725 DRAW_COLOR_CANDLES = 17
```

we will see that the orders match by the graphical construction type.

The difference is that the enumeration of drawing styles has constants for monochrome and color buffers, and there is one style that does not need a color buffer (the color filling between the two levels).

All buffers are to be colored. To set the drawing style, check the buffer status and drawing type passed to the **SetDrawType()** method and, depending on them, either set no drawing, or set filling the space between the two levels with color, or shift the status enumeration index by 8 units so that the constant value corresponds to that of the color buffer from the drawing style enumeration.

Beyond the class body, **implement the method:**

```
//+------------------------------------------------------------------+
//| Set the graphical construction type                              |
//+------------------------------------------------------------------+
void CBuffer::SetDrawType(void)
  {
   ENUM_DRAW_TYPE type=(!this.TypeBuffer() || !this.Status() ? DRAW_NONE : this.Status()==BUFFER_STATUS_FILLING ? DRAW_FILLING : ENUM_DRAW_TYPE(this.Status()+8));
   this.SetProperty(BUFFER_PROP_DRAW_TYPE,type);
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_DRAW_TYPE,type);
  }
//+------------------------------------------------------------------+
```

If the buffer type is BUFFER\_TYPE\_CALCULATE (0) or BUFFER\_STATUS\_NONE (0), the drawing style is set to "No drawing". If the buffer status is BUFFER\_STATUS\_FILLING (filling with color), the appropriate drawing style is set. All the remaining values are simply increased by 8. This shift indicates the constant of the colored drawing style.

Below is an example of increasing the ENUM\_BUFFER\_STATUS enumeration constant value by 8 and the value in the ENUM\_DRAW\_TYPE enumeration it falls on in this case:

```
enum ENUM_BUFFER_STATUS
  {
   BUFFER_STATUS_NONE,       //     0           // No drawing
   BUFFER_STATUS_FILLING,    //     1           // Color filling between two levels (MQL5)
   BUFFER_STATUS_LINE,       //     2 +8 = 10   // Line
   BUFFER_STATUS_HISTOGRAM,  //     3 +8 = 11   // Histogram from the zero line
   BUFFER_STATUS_ARROW,      //     4 +8 = 12   // Drawing with arrows
   BUFFER_STATUS_SECTION,    //     5 +8 = 13   // Segments
   BUFFER_STATUS_HISTOGRAM2, //     6 +8 = 14   // Histogram on two indicator buffers
   BUFFER_STATUS_ZIGZAG,     //     7 +8 = 15   // Zigzag style
   BUFFER_STATUS_BARS,       //     8 +8 = 16   // Display as bars (MQL5)
   BUFFER_STATUS_CANDLES,    //     9 +8 = 17   // Display as candles (MQL5)
  };
2020.04.15 12:51:53.725 DRAW_NONE = 0
2020.04.15 12:51:53.725 DRAW_LINE = 1
2020.04.15 12:51:53.725 DRAW_HISTOGRAM = 2
2020.04.15 12:51:53.725 DRAW_ARROW = 3
2020.04.15 12:51:53.725 DRAW_SECTION = 4
2020.04.15 12:51:53.725 DRAW_HISTOGRAM2 = 5
2020.04.15 12:51:53.725 DRAW_ZIGZAG = 6
2020.04.15 12:51:53.725 DRAW_FILLING = 7
2020.04.15 12:51:53.725 DRAW_BARS = 8
2020.04.15 12:51:53.725 DRAW_CANDLES = 9
2020.04.15 12:51:53.725 DRAW_COLOR_LINE = 10
2020.04.15 12:51:53.725 DRAW_COLOR_HISTOGRAM = 11
2020.04.15 12:51:53.725 DRAW_COLOR_ARROW = 12
2020.04.15 12:51:53.725 DRAW_COLOR_SECTION = 13
2020.04.15 12:51:53.725 DRAW_COLOR_HISTOGRAM2 = 14
2020.04.15 12:51:53.725 DRAW_COLOR_ZIGZAG = 15
2020.04.15 12:51:53.725 DRAW_COLOR_BARS = 16
2020.04.15 12:51:53.725 DRAW_COLOR_CANDLES = 17
```

Thus, its drawing style is set based on the buffer status value.

**Add all the remaining methods for setting and returning the buffer properties and displaying the description of its properties to the journal to the public section of the class:**

```
public:

//--- Set (1) the arrow code, (2) vertical shift of arrows, (3) symbol, (4) timeframe, (5) buffer activity flag
//--- (6) number of initial bars without drawing, (7) flag of displaying construction values in DataWindow,
//--- (8) shift of the indicator graphical construction along the time axis, (9) line style, (10) line width,
//--- (11) number of colors, (12) drawing color, (13) empty value and (14) graphical series name displayed in DataWindow
   virtual void      SetArrowCode(const uchar code)                  { return;                                                            }
   virtual void      SetArrowShift(const int shift)                  { return;                                                            }
   void              SetSymbol(const string symbol)                  { this.SetProperty(BUFFER_PROP_SYMBOL,symbol);                       }
   void              SetTimeframe(const ENUM_TIMEFRAMES timeframe)   { this.SetProperty(BUFFER_PROP_TIMEFRAME,timeframe);                 }
   void              SetActive(const bool flag)                      { this.SetProperty(BUFFER_PROP_ACTIVE,flag);                         }
   void              SetDrawBegin(const int value);
   void              SetShowData(const bool flag);
   void              SetShift(const int shift);
   void              SetStyle(const ENUM_LINE_STYLE style);
   void              SetWidth(const int width);
   void              SetColorNumbers(const int number);
   void              SetColor(const color colour);
   void              SetEmptyValue(const double value);
   void              SetLabel(const string label);

//--- Return (1) the serial number of the drawn buffer, (2) bound array index, (3) color buffer index,
//--- (4) index of the first free bound array, (5) buffer data period (timeframe) (6) buffer status,
//--- (7) buffer type, (8) buffer usage flag, (9) arrow code, (10) arrow shift for DRAW_ARROW style,
//--- (11) Number of initial bars that are not drawn and values in DataWindow, (12) graphical construction type,
//--- (13) flag of displaying construction values in DataWindow, (14) indicator graphical construction shift along the time axis,
//--- (15) drawing line style, (16) drawing line width, (17) number of colors, (18) drawing color,
//--- (19) set empty value, (20) buffer symbol and (21) name of the indicator graphical series displayed in DataWindow
   int               IndexPlot(void)                           const { return (int)this.GetProperty(BUFFER_PROP_INDEX_PLOT);              }
   int               IndexBase(void)                           const { return (int)this.GetProperty(BUFFER_PROP_INDEX_BASE);              }
   int               IndexColor(void)                          const { return (int)this.GetProperty(BUFFER_PROP_INDEX_COLOR);             }
   int               IndexNextBuffer(void)                     const { return (int)this.GetProperty(BUFFER_PROP_INDEX_NEXT);              }
   ENUM_TIMEFRAMES   Timeframe(void)                           const { return (ENUM_TIMEFRAMES)this.GetProperty(BUFFER_PROP_TIMEFRAME);   }
   ENUM_BUFFER_STATUS Status(void)                             const { return (ENUM_BUFFER_STATUS)this.GetProperty(BUFFER_PROP_STATUS);   }
   ENUM_BUFFER_TYPE  TypeBuffer(void)                          const { return (ENUM_BUFFER_TYPE)this.GetProperty(BUFFER_PROP_TYPE);       }
   bool              IsActive(void)                            const { return (bool)this.GetProperty(BUFFER_PROP_ACTIVE);                 }
   uchar             ArrowCode(void)                           const { return (uchar)this.GetProperty(BUFFER_PROP_ARROW_CODE);            }
   int               ArrowShift(void)                          const { return (int)this.GetProperty(BUFFER_PROP_ARROW_SHIFT);             }
   int               DrawBegin(void)                           const { return (int)this.GetProperty(BUFFER_PROP_DRAW_BEGIN);              }
   ENUM_DRAW_TYPE    DrawType(void)                            const { return (ENUM_DRAW_TYPE)this.GetProperty(BUFFER_PROP_DRAW_TYPE);    }
   bool              IsShowData(void)                          const { return (bool)this.GetProperty(BUFFER_PROP_SHOW_DATA);              }
   int               Shift(void)                               const { return (int)this.GetProperty(BUFFER_PROP_SHIFT);                   }
   ENUM_LINE_STYLE   LineStyle(void)                           const { return (ENUM_LINE_STYLE)this.GetProperty(BUFFER_PROP_LINE_STYLE);  }
   int               LineWidth(void)                           const { return (int)this.GetProperty(BUFFER_PROP_LINE_WIDTH);              }
   int               NumberColors(void)                        const { return (int)this.GetProperty(BUFFER_PROP_COLOR_INDEXES);           }
   color             Color(void)                               const { return (color)this.GetProperty(BUFFER_PROP_COLOR);                 }
   double            EmptyValue(void)                          const { return this.GetProperty(BUFFER_PROP_EMPTY_VALUE);                  }
   string            Symbol(void)                              const { return this.GetProperty(BUFFER_PROP_SYMBOL);                       }
   string            Label(void)                               const { return this.GetProperty(BUFFER_PROP_LABEL);                        }

//--- Return descriptions of the (1) buffer status, (2) buffer type, (3) buffer usage flag, (4) flag of displaying construction values in DataWindow,
//--- (5) drawing line style, (6) set empty value, (7) graphical construction type and (8) used timeframe
   string            GetStatusDescription(bool draw_type=false)const;
   string            GetTypeBufferDescription(void)            const;
   string            GetActiveDescription(void)                const;
   string            GetShowDataDescription(void)              const;
   string            GetLineStyleDescription(void)             const;
   string            GetEmptyValueDescription(void)            const;
   string            GetDrawTypeDescription(void)              const;
   string            GetTimeframeDescription(void)             const;

//--- Return the size of the data buffer array
   int               GetDataTotal(void)                        const { return ::ArraySize(this.DataArray);                                }

  };
//+------------------------------------------------------------------+
```

Let's consider the implementation of declared methods and closed parametric constructor.

**Closed parametric constructor:**

```
//+------------------------------------------------------------------+
//| Closed parametric constructor                                    |
//+------------------------------------------------------------------+
CBuffer::CBuffer(ENUM_BUFFER_STATUS buffer_status,ENUM_BUFFER_TYPE buffer_type,const uint index_plot,const uint index_base_array)
  {
   this.m_type=COLLECTION_BUFFERS_ID;
//--- Save integer properties
   this.m_long_prop[BUFFER_PROP_STATUS]                        = buffer_status;
   this.m_long_prop[BUFFER_PROP_TYPE]                          = buffer_type;
   this.m_long_prop[BUFFER_PROP_TIMEFRAME]                     = PERIOD_CURRENT;
   this.m_long_prop[BUFFER_PROP_ACTIVE]                        = false;
   this.m_long_prop[BUFFER_PROP_ARROW_CODE]                    = 0xFB;
   this.m_long_prop[BUFFER_PROP_ARROW_SHIFT]                   = 0;
   this.m_long_prop[BUFFER_PROP_DRAW_BEGIN]                    = 0;
   this.m_long_prop[BUFFER_PROP_SHOW_DATA]                     = (buffer_type>BUFFER_TYPE_CALCULATE ? true : false);
   this.m_long_prop[BUFFER_PROP_SHIFT]                         = 0;
   this.m_long_prop[BUFFER_PROP_LINE_STYLE]                    = STYLE_SOLID;
   this.m_long_prop[BUFFER_PROP_LINE_WIDTH]                    = 1;
   this.m_long_prop[BUFFER_PROP_COLOR_INDEXES]                 = 1;
   this.m_long_prop[BUFFER_PROP_COLOR]                         = clrRed;
   this.m_long_prop[BUFFER_PROP_NUM_DATAS]                     = 1;
   this.m_long_prop[BUFFER_PROP_INDEX_PLOT]                    = index_plot;
   this.m_long_prop[BUFFER_PROP_INDEX_BASE]                    = index_base_array;
   this.m_long_prop[BUFFER_PROP_INDEX_COLOR]                   = this.GetProperty(BUFFER_PROP_INDEX_BASE)+this.GetProperty(BUFFER_PROP_NUM_DATAS);
   this.m_long_prop[BUFFER_PROP_INDEX_NEXT]                    = this.GetProperty(BUFFER_PROP_INDEX_COLOR)+1;
   this.SetDrawType();
//--- Save real properties
   this.m_double_prop[this.IndexProp(BUFFER_PROP_EMPTY_VALUE)] = EMPTY_VALUE;
//--- Save string properties
   this.m_string_prop[this.IndexProp(BUFFER_PROP_SYMBOL)]      = ::Symbol();
   this.m_string_prop[this.IndexProp(BUFFER_PROP_LABEL)]       = (this.TypeBuffer()>BUFFER_TYPE_CALCULATE ? "Buffer "+(string)this.IndexPlot() : NULL);

//--- Bind indicator buffers with arrays
   ::SetIndexBuffer((int)this.GetProperty(BUFFER_PROP_INDEX_BASE),this.DataArray,INDICATOR_DATA);
   ::SetIndexBuffer((int)this.GetProperty(BUFFER_PROP_INDEX_COLOR),this.ColorArray,INDICATOR_COLOR_INDEX);

//--- Set integer buffer parameters
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_DRAW_TYPE,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_DRAW_TYPE));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_ARROW,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_ARROW_CODE));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_ARROW_SHIFT,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_ARROW_SHIFT));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_DRAW_BEGIN,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_DRAW_BEGIN));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_SHOW_DATA,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_SHOW_DATA));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_SHIFT,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_SHIFT));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LINE_STYLE,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_LINE_STYLE));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LINE_WIDTH,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_LINE_WIDTH));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_COLOR_INDEXES,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_COLOR_INDEXES));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LINE_COLOR,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_COLOR));
//--- Set real buffer parameters
   ::PlotIndexSetDouble((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_EMPTY_VALUE,this.GetProperty(BUFFER_PROP_EMPTY_VALUE));
//--- Set string buffer parameters
   ::PlotIndexSetString((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LABEL,this.GetProperty(BUFFER_PROP_LABEL));
  }
//+------------------------------------------------------------------+
```

The constructor inputs pass the status and type of the created buffer object, drawn buffer index (Market Watch index) and base array index (the base array is the very first array from the general list of arrays used to construct the buffer).

Assign the type of belonging to the buffer object collection to the object and fill in integer, real and string object properties with default values.

Next, bind the indicator buffers with the arrays and set integer, real and string properties from the newly filled default values of same-type buffer object properties to the drawn buffer.

These actions are enough for creating a one-color indicator buffer with the specified drawing type. No arrays should be declared, set and assigned when creating such a buffer from the indicator — all the necessary buffer objects are already present and assigned (if necessary) in the buffer object. The BUFFER\_PROP\_INDEX\_PLOT property contains the drawn buffer index used to set and receive the buffer data from its indicator program.

I will mention the standard methods here without delving into them. I have already described similar methods of other objects in all previous articles. Besides, I have considered the methods in detail when creating the very first library object:

**Comparison methods for searching and sorting, as well as for returning the flag of the equality of two compared objects:**

```
//+------------------------------------------------------------------+
//| Class methods                                                    |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Compare CBuffer objects by all possible properties               |
//+------------------------------------------------------------------+
int CBuffer::Compare(const CObject *node,const int mode=0) const
  {
   const CBuffer *compared_obj=node;
//--- compare integer properties of two buffers
   if(mode<BUFFER_PROP_INTEGER_TOTAL)
     {
      long value_compared=compared_obj.GetProperty((ENUM_BUFFER_PROP_INTEGER)mode);
      long value_current=this.GetProperty((ENUM_BUFFER_PROP_INTEGER)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare real properties of two buffers
   else if(mode<BUFFER_PROP_INTEGER_TOTAL+BUFFER_PROP_DOUBLE_TOTAL)
     {
      double value_compared=compared_obj.GetProperty((ENUM_BUFFER_PROP_DOUBLE)mode);
      double value_current=this.GetProperty((ENUM_BUFFER_PROP_DOUBLE)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare string properties of two buffers
   else if(mode<BUFFER_PROP_INTEGER_TOTAL+BUFFER_PROP_DOUBLE_TOTAL+BUFFER_PROP_STRING_TOTAL)
     {
      string value_compared=compared_obj.GetProperty((ENUM_BUFFER_PROP_STRING)mode);
      string value_current=this.GetProperty((ENUM_BUFFER_PROP_STRING)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
   return 0;
  }
//+------------------------------------------------------------------+
//| Compare CBuffer objects by all properties                        |
//+------------------------------------------------------------------+
bool CBuffer::IsEqual(CBuffer *compared_obj) const
  {
   int beg=0, end=BUFFER_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_BUFFER_PROP_INTEGER prop=(ENUM_BUFFER_PROP_INTEGER)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=BUFFER_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_BUFFER_PROP_DOUBLE prop=(ENUM_BUFFER_PROP_DOUBLE)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=BUFFER_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_BUFFER_PROP_STRING prop=(ENUM_BUFFER_PROP_STRING)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

**The method for displaying all buffer object properties:**

```
//+------------------------------------------------------------------+
//| Display buffer properties in the journal                         |
//+------------------------------------------------------------------+
void CBuffer::Print(const bool full_prop=false)
  {
   ::Print("============= ",
           CMessage::Text(MSG_LIB_PARAMS_LIST_BEG),": ",
           this.GetTypeBufferDescription(),"[",(string)this.IndexPlot(),"] \"",this.GetStatusDescription(true),"\"",
           " =================="
          );
   int beg=0, end=BUFFER_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_BUFFER_PROP_INTEGER prop=(ENUM_BUFFER_PROP_INTEGER)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=BUFFER_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {

      ENUM_BUFFER_PROP_DOUBLE prop=(ENUM_BUFFER_PROP_DOUBLE)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=BUFFER_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_BUFFER_PROP_STRING prop=(ENUM_BUFFER_PROP_STRING)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("================== ",
           CMessage::Text(MSG_LIB_PARAMS_LIST_END),": ",
           this.GetTypeBufferDescription(),"[",(string)this.IndexPlot(),"] \"",this.GetStatusDescription(true),"\"",
           " ==================\n"
          );
  }
//+------------------------------------------------------------------+
```

**The methods returning the description of integer, real and string buffer object properties:**

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
      property==BUFFER_PROP_NUM_DATAS     ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_NUM_DATAS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_INDEX_NEXT    ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_INDEX_NEXT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_TIMEFRAME     ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_TIMEFRAME)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetTimeframeDescription()
         )  :
      property==BUFFER_PROP_STATUS        ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_STATUS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetStatusDescription()
         )  :
      property==BUFFER_PROP_TYPE          ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_TYPE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetTypeBufferDescription()
         )  :
      property==BUFFER_PROP_ACTIVE        ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_ACTIVE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetActiveDescription()
         )  :
      property==BUFFER_PROP_ARROW_CODE    ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_ARROW_CODE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_ARROW_SHIFT   ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_ARROW_SHIFT)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_DRAW_BEGIN    ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_DRAW_BEGIN)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_DRAW_TYPE     ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_DRAW_TYPE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetDrawTypeDescription()
         )  :
      property==BUFFER_PROP_SHOW_DATA     ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_SHOW_DATA)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetShowDataDescription()
         )  :
      property==BUFFER_PROP_SHIFT         ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_SHIFT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_LINE_STYLE    ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_LINE_STYLE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetLineStyleDescription()
         )  :
      property==BUFFER_PROP_LINE_WIDTH    ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_LINE_WIDTH)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_COLOR_INDEXES ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_COLOR_NUM)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_COLOR         ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_COLOR)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+ColorToString(this.Color(),true)
         )  :
      property==BUFFER_PROP_INDEX_BASE    ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_INDEX_BASE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_NUM_DATAS ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_NUM_DATAS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_INDEX_COLOR   ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_INDEX_COLOR)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_INDEX_NEXT         ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_INDEX_NEXT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+ColorToString(this.Color(),true)
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return description of a buffer's real property                   |
//+------------------------------------------------------------------+
string CBuffer::GetPropertyDescription(ENUM_BUFFER_PROP_DOUBLE property)
  {
   return
     (
      property==BUFFER_PROP_EMPTY_VALUE    ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_EMPTY_VALUE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetEmptyValueDescription()
         )  :
      ""
     );
  }
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
          ": "+(this.Label()==NULL || this.Label()=="" ? CMessage::Text(MSG_LIB_PROP_NOT_SET) : this.Label())
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

Let's have a look at the implementation of the remaining added methods to the class public section.

**The method returning the buffer status description:**

```
//+------------------------------------------------------------------+
//| Return the buffer status description                             |
//+------------------------------------------------------------------+
string CBuffer::GetStatusDescription(bool draw_type=false) const
  {
   string type=
     (
      this.Status()==BUFFER_STATUS_NONE         ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_STATUS_NONE)         :
      this.Status()==BUFFER_STATUS_ARROW        ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_STATUS_ARROW)        :
      this.Status()==BUFFER_STATUS_BARS         ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_STATUS_BARS)         :
      this.Status()==BUFFER_STATUS_CANDLES      ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_STATUS_CANDLES)      :
      this.Status()==BUFFER_STATUS_FILLING      ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_STATUS_FILLING)      :
      this.Status()==BUFFER_STATUS_HISTOGRAM    ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_STATUS_HISTOGRAM)    :
      this.Status()==BUFFER_STATUS_HISTOGRAM2   ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_STATUS_HISTOGRAM2)   :
      this.Status()==BUFFER_STATUS_LINE         ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_STATUS_LINE)         :
      this.Status()==BUFFER_STATUS_SECTION      ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_STATUS_SECTION)      :
      this.Status()==BUFFER_STATUS_ZIGZAG       ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_STATUS_ZIGZAG)       :
      "Unknown"
     );
   return(!draw_type ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_STATUS_NAME)+" \""+type+"\"" : type);
  }
//+------------------------------------------------------------------+
```

Since the buffer status also defines the drawing style, the method receives the flag specifying how to return the description:

— if the buffer status is requested ( **draw\_type** set to false), the buffer status is returned in the following form

```
Buffer status: Indicator buffer with graphical construction type "Line"
```

— if the drawing type is requested ( **draw\_type** is set to true), the drawing type is returned in the following form

```
Graphical construction type: Line
```

All **methods returning object property descriptions** are quite simple, leave them for independent study:

```
//+------------------------------------------------------------------+
//| Return the buffer type description                               |
//+------------------------------------------------------------------+
string CBuffer::GetTypeBufferDescription(void) const
  {
   return
     (
      this.TypeBuffer()==BUFFER_TYPE_DATA       ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_TYPE_DATA)        :
      this.TypeBuffer()==BUFFER_TYPE_CALCULATE  ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_TYPE_CALCULATE)   :
      "Unknown"
     );
  }
//+------------------------------------------------------------------+
//| Return description of the buffer usage flag                      |
//+------------------------------------------------------------------+
string CBuffer::GetActiveDescription(void) const
  {
   return(this.IsActive() ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO));
  }
//+---------------------------------------------------------------------+
//|Return description of displaying construction values in DataWindow   |
//+---------------------------------------------------------------------+
string CBuffer::GetShowDataDescription(void) const
  {
   return(this.IsShowData() ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO));
  }
//+------------------------------------------------------------------+
//| Return description of the drawing line style                     |
//+------------------------------------------------------------------+
string CBuffer::GetLineStyleDescription(void) const
  {
   return
     (
      this.LineStyle()==STYLE_SOLID       ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_STYLE_SOLID)      :
      this.LineStyle()==STYLE_DASH        ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_STYLE_DASH)       :
      this.LineStyle()==STYLE_DOT         ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_STYLE_DOT)        :
      this.LineStyle()==STYLE_DASHDOT     ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_STYLE_DASHDOT)    :
      this.LineStyle()==STYLE_DASHDOTDOT  ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_STYLE_DASHDOTDOT) :
      "Unknown"
     );
  }
//+------------------------------------------------------------------+
//| Return description of the set empty value                        |
//+------------------------------------------------------------------+
string CBuffer::GetEmptyValueDescription(void) const
  {
   return(this.EmptyValue()<EMPTY_VALUE ? ::DoubleToString(this.EmptyValue(),(this.EmptyValue()==0 ? 1 : 8)) : "EMPTY_VALUE");
  }
//+------------------------------------------------------------------+
//| Return description of the graphical construction type            |
//+------------------------------------------------------------------+
string CBuffer::GetDrawTypeDescription(void) const
  {
   return this.GetStatusDescription(true);
  }
//+------------------------------------------------------------------+
//| Return description of the used timeframe                         |
//+------------------------------------------------------------------+
string CBuffer::GetTimeframeDescription(void) const
  {
   string timeframe=TimeframeDescription(this.Timeframe());
   return(this.Timeframe()==PERIOD_CURRENT ? CMessage::Text(MSG_LIB_TEXT_PERIOD_CURRENT)+" ("+timeframe+")" : timeframe);
  }
//+------------------------------------------------------------------+
```

**The methods for setting different buffer object properties:**

```
//+------------------------------------------------------------------+
//| Set the number of initial bars                                   |
//| without drawing and values in DataWindow                         |
//+------------------------------------------------------------------+
void CBuffer::SetDrawBegin(const int value)
  {
   this.SetProperty(BUFFER_PROP_DRAW_BEGIN,value);
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_DRAW_BEGIN,value);
  }
//+------------------------------------------------------------------+
//| Set the flag of displaying                                       |
//| construction values in DataWindow                                |
//+------------------------------------------------------------------+
void CBuffer::SetShowData(const bool flag)
  {
   this.SetProperty(BUFFER_PROP_SHOW_DATA,flag);
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_SHOW_DATA,flag);
  }
//+------------------------------------------------------------------+
//| Set the indicator graphical construction shift                   |
//+------------------------------------------------------------------+
void CBuffer::SetShift(const int shift)
  {
   this.SetProperty(BUFFER_PROP_SHIFT,shift);
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_SHIFT,shift);
  }
//+------------------------------------------------------------------+
//| Set the line style                                               |
//+------------------------------------------------------------------+
void CBuffer::SetStyle(const ENUM_LINE_STYLE style)
  {
   this.SetProperty(BUFFER_PROP_LINE_STYLE,style);
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LINE_STYLE,style);
  }
//+------------------------------------------------------------------+
//| Set the line width                                               |
//+------------------------------------------------------------------+
void CBuffer::SetWidth(const int width)
  {
   this.SetProperty(BUFFER_PROP_LINE_WIDTH,width);
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LINE_WIDTH,width);
  }
//+------------------------------------------------------------------+
//| Set the number of colors                                         |
//+------------------------------------------------------------------+
void CBuffer::SetColorNumbers(const int number)
  {
   this.SetProperty(BUFFER_PROP_COLOR_INDEXES,number);
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_COLOR_INDEXES,number);
  }
//+------------------------------------------------------------------+
//| Set the drawing color                                            |
//+------------------------------------------------------------------+
void CBuffer::SetColor(const color colour)
  {
   this.SetProperty(BUFFER_PROP_COLOR,colour);
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LINE_COLOR,colour);
  }
//+------------------------------------------------------------------+
//| Set the "empty" value for construction                           |
//| without drawing                                                  |
//+------------------------------------------------------------------+
void CBuffer::SetEmptyValue(const double value)
  {
   this.SetProperty(BUFFER_PROP_EMPTY_VALUE,value);
   ::PlotIndexSetDouble((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_EMPTY_VALUE,value);
  }
//+------------------------------------------------------------------+
//| Set the drawing color                                            |
//+------------------------------------------------------------------+
void CBuffer::SetLabel(const string label)
  {
   this.SetProperty(BUFFER_PROP_LABEL,label);
   ::PlotIndexSetString((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LABEL,label);
  }
//+------------------------------------------------------------------+
```

The value passed to the method is first written to the appropriate buffer object property, then the property is set for the drawn buffer by its index.

Currently, this is all that needs to be done for creating the abstract indicator buffer object.

### Checking creation of buffer objects in the indicator

To check the abstract buffer object operation, use [the indicator from the previous article](https://www.mql5.com/en/articles/7804#node03) and save it in \\MQL5\\Indicators\\TestDoEasy\ **Part42\** as **TestDoEasyPart42.mq5**.

Remove all unnecessary things from the indicator.

We do not need any buttons, button pressing handling functions and buffer data filling functions, so remove them from the code leaving only the elements directly related to the library operation.

Besides, let's move the functions for copying data from OnCalculate() to the library price structure from the indicator code to the file of the library service functions. Open \\MQL5\\Include\\DoEasy\\Services\ **DELib.mqh** and add two functions to it:

```
//+------------------------------------------------------------------+
//| Copy data from the first OnCalculate() form to the structure     |
//+------------------------------------------------------------------+
void CopyData(const int rates_total,
              const int prev_calculated,
              const int begin,
              const double &price[])
  {
//--- Get the array indexing flag as in the timeseries. If failed,
//--- set the indexing direction for the array as in the timeseries
   bool as_series_price=ArrayGetAsSeries(price);
   if(!as_series_price)
      ArraySetAsSeries(price,true);
//--- Copy the array zero bar to the OnCalculate() SDataCalculate data structure
   rates_data.rates_total=rates_total;
   rates_data.prev_calculated=prev_calculated;
   rates_data.begin=begin;
   rates_data.price=price[0];
//--- Return the array's initial indexing direction
   if(!as_series_price)
      ArraySetAsSeries(price,false);
  }
//+------------------------------------------------------------------+
//| Copy data from the second OnCalculate() form to the structure    |
//+------------------------------------------------------------------+
void CopyData(const int rates_total,
              const int prev_calculated,
              const datetime &time[],
              const double &open[],
              const double &high[],
              const double &low[],
              const double &close[],
              const long &tick_volume[],
              const long &volume[],
              const int &spread[])
  {
//--- Get the array indexing flags as in the timeseries. If failed,
//--- set the indexing direction or the arrays as in the timeseries
   bool as_series_time=ArrayGetAsSeries(time);
   if(!as_series_time)
      ArraySetAsSeries(time,true);
   bool as_series_open=ArrayGetAsSeries(open);
   if(!as_series_open)
      ArraySetAsSeries(open,true);
   bool as_series_high=ArrayGetAsSeries(high);
   if(!as_series_high)
      ArraySetAsSeries(high,true);
   bool as_series_low=ArrayGetAsSeries(low);
   if(!as_series_low)
      ArraySetAsSeries(low,true);
   bool as_series_close=ArrayGetAsSeries(close);
   if(!as_series_close)
      ArraySetAsSeries(close,true);
   bool as_series_tick_volume=ArrayGetAsSeries(tick_volume);
   if(!as_series_tick_volume)
      ArraySetAsSeries(tick_volume,true);
   bool as_series_volume=ArrayGetAsSeries(volume);
   if(!as_series_volume)
      ArraySetAsSeries(volume,true);
   bool as_series_spread=ArrayGetAsSeries(spread);
   if(!as_series_spread)
      ArraySetAsSeries(spread,true);
//--- Copy the arrays' zero bar to the OnCalculate() SDataCalculate data structure
   rates_data.rates_total=rates_total;
   rates_data.prev_calculated=prev_calculated;
   rates_data.rates.time=time[0];
   rates_data.rates.open=open[0];
   rates_data.rates.high=high[0];
   rates_data.rates.low=low[0];
   rates_data.rates.close=close[0];
   rates_data.rates.tick_volume=tick_volume[0];
   rates_data.rates.real_volume=(#ifdef __MQL5__ volume[0] #else 0 #endif);
   rates_data.rates.spread=(#ifdef __MQL5__ spread[0] #else 0 #endif);
//--- Return the arrays' initial indexing direction
   if(!as_series_time)
      ArraySetAsSeries(time,false);
   if(!as_series_open)
      ArraySetAsSeries(open,false);
   if(!as_series_high)
      ArraySetAsSeries(high,false);
   if(!as_series_low)
      ArraySetAsSeries(low,false);
   if(!as_series_close)
      ArraySetAsSeries(close,false);
   if(!as_series_tick_volume)
      ArraySetAsSeries(tick_volume,false);
   if(!as_series_volume)
      ArraySetAsSeries(volume,false);
   if(!as_series_spread)
      ArraySetAsSeries(spread,false);
  }
//+------------------------------------------------------------------+
```

As a result, we cleaned up the indicator code for the current verification and future use. The two moved functions are necessary to work with indicators within the library, so the file of the library service functions is the best place for them.

**The indicator "header" is as follows:**

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart42.mq5 |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
#include <DoEasy\Objects\Indicators\Buffer.mqh>
//--- properties
#property indicator_chart_window
#property indicator_buffers 4
#property indicator_plots   2

//--- classes

//--- enums

//--- defines

//--- structures

//--- input variables
/*sinput*/ ENUM_SYMBOLS_MODE  InpModeUsedSymbols=  SYMBOLS_MODE_DEFINES;            // Mode of used symbols list
sinput   string               InpUsedSymbols    =  "EURUSD,AUDUSD,EURAUD,EURGBP,EURCAD,EURJPY,EURUSD,GBPUSD,NZDUSD,USDCAD,USDJPY";  // List of used symbols (comma - separator)
sinput   ENUM_TIMEFRAMES_MODE InpModeUsedTFs    =  TIMEFRAMES_MODE_LIST;            // Mode of used timeframes list
sinput   string               InpUsedTFs        =  "M1,M5,M15,M30,H1,H4,D1,W1,MN1"; // List of used timeframes (comma - separator)
sinput   bool                 InpUseSounds      =  true; // Use sounds
//--- indicator buffers
CArrayObj      list_buffers;                    // Temporary list for storing two buffer objects
//--- global variables
CEngine        engine;                          // CEngine library main object
string         prefix;                          // Prefix of graphical object names
int            min_bars;                        // The minimum number of bars for the indicator calculation
int            used_symbols_mode;               // Mode of working with symbols
string         array_used_symbols[];            // The array for passing used symbols to the library
string         array_used_periods[];            // The array for passing used timeframes to the library
//+------------------------------------------------------------------+
```

Here we order the compiler to create the indicator in a separate window and set four buffers for it (two drawn and two color ones).

The remaining buffer parameters are set during and after creating indicator buffer objects.

Add [the dynamic array of CArrayObj pointers](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) as the indicator buffer. It is to store created buffers.

When working with buffer objects, we do not need to declare double arrays for assigning them as indicator buffers. Everything is located inside the created buffer objects and assigned when creating them in the indicator's OnInit() handler:

```
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Initialize DoEasy library
   OnInitDoEasy();

//--- Set indicator global variables
   prefix=engine.Name()+"_";
   //--- Get the index of the maximum used timeframe in the array,
   //--- calculate the number of bars of the current period fitting in the maximum used period
   //--- Use the obtained value if it exceeds 2, otherwise use 2
   int index=ArrayMaximum(ArrayUsedTimeframes);
   int num_bars=NumberBarsInTimeframe(ArrayUsedTimeframes[index]);
   min_bars=(index>WRONG_VALUE ? (num_bars>2 ? num_bars : 2) : 2);

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

//--- Create two buffer objects
   CBuffer *buffer0=new CBuffer(BUFFER_STATUS_ARROW,BUFFER_TYPE_DATA, 0, 0);
   CBuffer *buffer1=new CBuffer(BUFFER_STATUS_LINE,BUFFER_TYPE_DATA, 1, buffer0.IndexNextBuffer());
//--- Set non-default values for the second buffer's "empty" value and color
   buffer1.SetEmptyValue(0);
   buffer1.SetColor(clrBlue);
//--- Add both buffers to the list of indicator buffers
   list_buffers.Add(buffer0);
   list_buffers.Add(buffer1);
//--- Print data of the created buffers
   buffer0.Print();
   buffer1.Print();
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

Here I have created two objects of abstract indicator buffers.

The first one has the "Drawing with arrows" status. Set 0 as the drawn buffer index and the buffer base array index — this is the very first buffer, and all indices should have initial values here.

We are going to get rid of specifying array indices while creating buffer objects when creating the indicator buffer collection — all indices are to be set automatically there, no user intervention is to be required.

The second object has the "Line" status. Set 1 (following zero) as the drawn buffer index. Also, set the value returned from the first buffer object as the base array index. This is the value indicating the index of the next free array for specifying it as the base array for the next buffer.

Besides, 0 (zero) is assigned to the second created buffer as the "empty" value, while the line color is set to "Blue" (in order to check how the values are set to buffer object properties).

Add both buffers to the previously declared **list\_buffers** list and send to the journal all properties of both newly created buffers.

Here we do not check if the buffer objects have been successfully created and added to the list since this is a test indicator, and we are free to neglect the control over object creation and adding to the list for the sake of fast verification of buffer objects.

Let's "clear" the OnCalculate() handler from the previous article leaving only the most necessary elements for our test: we only need to check the results of creating two buffer objects and their correct assignment as indicator buffers.

How can we do that?

The buffer creation is checked in OnInit() when we print all the properties of each created buffer to the journal. Successful assignment of the created objects as indicator buffers can be checked only in OnCalculate(). To do this, simply compare the size of arrays used in objects as indicator buffers with the number of bars on the symbol.

As soon as we assign an array as an indicator buffer, the executed terminal subsystem takes these arrays under its wing, allocates memory for them and manages the array size. Therefore, we only need to get each of the objects from the **list\_buffers list** and compare the size of the array assigned as a buffer with the **rates\_total** value in OnCalculate(). Their equality indicates that the terminal subsystem has taken the arrays from the buffer objects under its control.

To avoid displaying entries at each tick, assigning buffer object arrays as indicator buffers is verified during the very first indicator calculation when the calculated **limit** value exceeds one:

```
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
//+------------------------------------------------------------------+
//| OnCalculate code block for working with the library:             |
//+------------------------------------------------------------------+

//--- Pass the current symbol data from OnCalculate() to the price structure
   CopyData(rates_total,prev_calculated,time,open,high,low,close,tick_volume,volume,spread);

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
      EventsHandling();             // Working with library events
     }
//+------------------------------------------------------------------+
//| OnCalculate code block for working with the indicator:           |
//+------------------------------------------------------------------+
//--- Set OnCalculate arrays as timeseries
   ArraySetAsSeries(open,true);
   ArraySetAsSeries(high,true);
   ArraySetAsSeries(low,true);
   ArraySetAsSeries(close,true);
   ArraySetAsSeries(time,true);
   ArraySetAsSeries(tick_volume,true);
   ArraySetAsSeries(volume,true);
   ArraySetAsSeries(spread,true);

//--- Check and calculate the number of calculated bars
//--- If limit = 0, there are no new bars - calculate the current one
//--- If limit = 1, a new bar has appeared - calculate the first and the current ones
//--- limit > 1 means the first launch or changes in history - the full recalculation of all data
   int limit=rates_total-prev_calculated;

//--- Recalculate the entire history
   if(limit>1)
     {
      //--- In a loop by the number of buffers in the list
      for(int i=0;i<list_buffers.Total();i++)
        {
         //--- get the next buffer and display the type of its graphical construction to the journal
         //--- together with the double array assigned to the buffer (if all is correct, the size is equal to rates_total)
         CBuffer *buff=list_buffers.At(i);
         Print(buff.Label()," type = ",EnumToString(buff.DrawType()),", data total = ",buff.GetDataTotal(),", rates_total=",rates_total);
        }

      limit=rates_total-1;
     }
//--- Prepare data

//--- Calculate the indicator
   for(int i=limit; i>WRONG_VALUE && !IsStopped(); i--)
     {
      CalculateSeries(i,time[i]);
     }
//--- return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

Find the entire code of the test indicator in the files attached below.

**Compile the indicator and launch it on the chart after setting the following parameters:**

![](https://c.mql5.com/2/40/terminal64_ZQQ911Khgn.png)

The journal displays the following entries:

```
Account 8550475: Artyom Trishkin (MetaQuotes Software Corp.) 10425.23 USD, 1:100, Hedge, MetaTrader 5 demo
--- Initializing "DoEasy" library ---
Working with the current symbol only. The number of used symbols: 1
"EURUSD"
Working with the specified timeframe list:
"M5"  "M15" "M30" "H1"
EURUSD symbol timeseries:
- Timeseries "EURUSD" M5: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3684
- Timeseries "EURUSD" M15: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3042
- Timeseries "EURUSD" M30: Requested: 1000, Actual: 0, Created: 0, On the server: 0
- Timeseries "EURUSD" H1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 6240
Library initialization time: 00:00:00.156

============= Parameter list start: Colored data buffer[0] "Drawing with arrows" ==================
Plotted buffer serial number: 0
Buffer status: Indicator buffer with graphical construction type "Drawing with arrows"
Buffer type: Colored data buffer
Buffer data period (timeframe): Current chart period (M30)
Active: No
Arrow code: 251
The vertical shift of the arrows: 0
The number of initial bars that are not drawn and values in DataWindow: 0
Display construction values in DataWindow: Yes
Graphical construction type: Drawing with arrows
Indicator graphical construction shift by time axis in bars: 0
Line style: Solid line
Line width: 1
Number of colors: 1
Drawing color: clrRed
Number of data buffers: 1
Base data buffer index: 0
Color buffer index: 1
Index of the array to be assigned as the next indicator buffer: 2
------
Empty value for plotting where nothing will be drawn: EMPTY_VALUE
------
Buffer symbol: EURUSD
Name of the graphical indicator series displayed in DataWindow: Buffer 0
================== Parameter list end: Colored data buffer[0] "Drawing with arrows" ==================

============= Parameter list start: Colored data buffer[1] "Line" ==================
Plotted buffer serial number: 1
Buffer status: Indicator buffer with graphical construction type "Line"
Buffer type: Colored data buffer
Buffer data period (timeframe): Current chart period (M30)
Active: No
Arrow code: 251
The vertical shift of the arrows: 0
The number of initial bars that are not drawn and values in DataWindow: 0
Display construction values in DataWindow: Yes
Graphical construction type: Line
Indicator graphical construction shift by time axis in bars: 0
Line style: Solid line
Line width: 1
Number of colors: 1
Drawing color: clrBlue
Number of data buffers: 1
Base data buffer index: 2
Color buffer index: 3
Index of the array to be assigned as the next indicator buffer: 4
------
Empty value for plotting where nothing will be drawn: 0.0
------
Buffer symbol: EURUSD
Name of the graphical indicator series displayed in DataWindow: Buffer 1
================== Parameter list end: Colored data buffer[1] "Line" ==================

"EURUSD" M30 timeseries created successfully:
- Timeseries "EURUSD" M30: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5111

Buffer 0 type = DRAW_COLOR_ARROW, data total = 5111, rates_total=5111
Buffer 1 type = DRAW_COLOR_LINE, data total = 5111, rates_total=5111
```

After the library messages about timseries creation, the block displaying all properties of each of the two created buffer objects is printed from OnInit(). Next, two messages about the drawing type of each of the created buffers are displayed from OnCalculate(), the size of arrays from buffer objects assigned as indicator buffers are printed and the **rates\_total** value at the moment of the indicator launch is specified.

As we can see, the size of arrays and **rates\_total** match. This means that the arrays of the created buffer objects are controlled by the terminal, and they are used as indicator buffers.

To make sure once again, simply open the indicator properties ( **Ctrl+I**) and go to the Colors tab:

![](https://c.mql5.com/2/40/terminal64_cjh5iXsOeF.png)

Names and colors are set for both indicator buffers. The name and color have not been specified, except for the default ones set in the buffer object class constructor. In case of the second buffer, we have reset the color to blue after creating it in OnInit().

Everything works as expected. However, this is just the beginning. In order to create various types of indicator buffers, we need to create inherited classes for each of the graphical construction types and work with these classes from the indicator buffer collection.

### What's next?

In the next article, we will create descendant objects of the abstract buffer class. These objects are to be used by the library for creating and using indicator buffers in indicator programs based on the DoEasy library.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave your questions, comments and suggestions in the comments.

Please keep in mind that here I have developed the MQL5 test indicator for MetaTrader 5.

The attached files are intended only for MetaTrader 5. The current library version has not been tested in MetaTrader 4.

After creating all classes of indicator buffers and their collections, I will try to implement some MQL5 features in MetaTrader 4.

[Back to contents](https://www.mql5.com/en/articles/7821#node00)

**Previous articles within the series:**

[Timeseries in DoEasy library (part 35): Bar object and symbol timeseries list](https://www.mql5.com/en/articles/7594)

[Timeseries in DoEasy library (part 36): Object of timeseries for all used symbol periods](https://www.mql5.com/en/articles/7627)

[Timeseries in DoEasy library (part 37): Timeseries collection - database of timeseries by symbols and periods](https://www.mql5.com/en/articles/7663)

[Timeseries in DoEasy library (part 38): Timeseries collection - real-time updates and accessing data from the program](https://www.mql5.com/en/articles/7695)

[Timeseries in DoEasy library (part 39): Library-based indicators - preparing data and timeseries events](https://www.mql5.com/en/articles/7724)

[Timeseries in DoEasy library (part 40): Library-based indicators - updating data in real time](https://www.mql5.com/en/articles/7771)

[Timeseries in DoEasy library (part 41): Sample multi-symbol multi-period indicator](https://www.mql5.com/en/articles/7804)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7821](https://www.mql5.com/ru/articles/7821)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7821.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7821/mql5.zip "Download MQL5.zip")(3711.1 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/348883)**
(12)


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
16 Apr 2020 at 19:33

**Sergey Chalyshev:**

It's just the [DoEasy (Part **42**).](https://www.mql5.com/en/articles/7821)

everything will happen ))

Is it supposed to be funny?

![Aliaksandr Hryshyn](https://c.mql5.com/avatar/2016/2/56CF9FD9-71DB.jpg)

**[Aliaksandr Hryshyn](https://www.mql5.com/en/users/greshnik1)**
\|
17 Apr 2020 at 21:09

An article for the sake of an article, like so many others.

What is the idea? How to use properties of [indicator buffers](https://www.mql5.com/en/articles/180 "Article: Averaging price series without additional buffers for intermediate calculations")? That's in the help. How to use classes, inheritance, methods, properties and others? So there were already OOP examples in previous articles. Shall I share code? There is a kodobase for that.

Let's search for good codes in kodobase and make a couple of hundreds of such articles based on them, there is a lot of good stuff there.

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
17 Apr 2020 at 23:03

**Aliaksandr Hryshyn:**

Please show with deeds what is right, what is necessary, useful and ideological.

![Andrey Khatimlianskii](https://c.mql5.com/avatar/2022/10/63528ae6-0b8c.jpg)

**[Andrey Khatimlianskii](https://www.mql5.com/en/users/komposter)**
\|
18 Apr 2020 at 08:56

**Aliaksandr Hryshyn:**

An article for the sake of an article, like so many others.

This idea is in the air and has been voiced many times, nobody cares. The main thing is that the articles are interesting to MQ. The rest is not important.

![Mykola Revych](https://c.mql5.com/avatar/2010/12/4D1345D2-BE9D.jpg)

**[Mykola Revych](https://www.mql5.com/en/users/1cmaster)**
\|
2 Dec 2020 at 17:26

[@Artyom Trishkin](https://www.mql5.com/en/users/artmedia70) Good afternoon, Artem. Maybe you know and can give me a hint.

I created an indicator based on a class. I also wrapped the arrays in a separate class dto.

I attach the indicator to the chart - everything is fine.

But during visual testing there are holes in the bars in the arrays.

The 0-th bar comes first, then the 1st bar, and then immediately the data of yesterday's bar.

As far as I understand, I am accessing the data array before it is ready?

Could you help me with advice? I attach the indicator code.

The indicator is simple - the channel is calculated, if the bar crosses the upper boundary, a buy signal is inserted into the indicator buffer.

If the bar crosses the lower boundary, a sell signal is inserted into a separate buffer.

What is the correct way to check that the indicator buffers are ready for calculations?


![Native Twitter Client: Part 2](https://c.mql5.com/2/40/mql_twitter__1.png)[Native Twitter Client: Part 2](https://www.mql5.com/en/articles/8318)

A Twitter client implemented as MQL class to allow you to send tweets with photos. All you need is to include a single self contained include file and off you go to tweet all your wonderful charts and signals.

![Native Twitter Client for MT4 and MT5 without DLL](https://c.mql5.com/2/41/mql5_twitter__1.png)[Native Twitter Client for MT4 and MT5 without DLL](https://www.mql5.com/en/articles/8270)

Ever wanted to access tweets and/or post your trade signals on Twitter ? Search no more, these on-going article series will show you how to do it without using any DLL. Enjoy the journey of implementing Twitter API using MQL. In this first part, we will follow the glory path of authentication and authorization in accessing Twitter API.

![Multicurrency monitoring of trading signals (Part 5): Composite signals](https://c.mql5.com/2/39/Article_Logo.png)[Multicurrency monitoring of trading signals (Part 5): Composite signals](https://www.mql5.com/en/articles/7759)

In the fifth article related to the creation of a trading signal monitor, we will consider composite signals and will implement the necessary functionality. In earlier versions, we used simple signals, such as RSI, WPR and CCI, and we also introduced the possibility to use custom indicators.

![Timeseries in DoEasy library (part 41): Sample multi-symbol multi-period indicator](https://c.mql5.com/2/38/MQL5-avatar-doeasy-library__6.png)[Timeseries in DoEasy library (part 41): Sample multi-symbol multi-period indicator](https://www.mql5.com/en/articles/7804)

In the article, we will consider a sample multi-symbol multi-period indicator using the timeseries classes of the DoEasy library displaying the chart of a selected currency pair on a selected timeframe as candles in a subwindow. I am going to modify the library classes a bit and create a separate file for storing enumerations for program inputs and selecting a compilation language.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/7821&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070416310132741490)

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