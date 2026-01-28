---
title: Timeseries in DoEasy library (part 45): Multi-period indicator buffers
url: https://www.mql5.com/en/articles/8023
categories: Trading Systems, Indicators
relevance_score: 3
scraped_at: 2026-01-23T19:35:32.925564
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/8023&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070408768170169682)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/8023#node01)
- [Improving classes for working with indicator buffers in multi-period mode](https://www.mql5.com/en/articles/8023#node02)
- [Testing](https://www.mql5.com/en/articles/8023#node03)
- [What's next?](https://www.mql5.com/en/articles/8023#node04)


### Concept

In the [previous article](https://www.mql5.com/en/articles/7886), I started the development of the indicator buffer collection class. Today, I will resume my work and arrange the method of creating indicator buffers and accessing their data. I am going to implement display of buffer values to the current symbol/period chart depending on the period property (data timeframe) set in the buffer object. If the timeframe value not corresponding to the current chart is set for the buffer, all buffer data is to be displayed on the chart correctly. For example, if the current chart has the data period of M15, while the buffer object chart is set to H1, the buffer data is displayed on each bar of the current M15 chart whose time is located inside the H1 chart bar. In the current example, four bars of the current chart are filled with the same value corresponding to the requested value from the bar with the chart period of H1.

These improvements allow us to create multi-period indicators comfortably since we only need to set a timeframe for a buffer object. The library does the rest. Besides, I am going to gradually facilitate creation of indicator buffers. It is to be reduced to a single code string, in which the required indicator buffer object with the necessary properties is to be created.

Any indicator buffer is to be accessed via the buffer index with a certain drawing type.

The "buffer index" concept is as follows: if the arrow buffer is created first followed by the line one, and then the arrow buffer is created again, the buffer object indices are arranged in the sequence of their creation. Each drawing style features its own index sequence. In the provided example, the indices are as follows: the first created arrow buffer has the index of 0, the second one has the index of 1, while the line buffer created immediately after the first arrow buffer has the index of 0 because it is the very first buffer featuring the "line" drawing style, although it is created the second in a row.

Let's illustrate the concept of buffer object indices:

01. Creating the arrow buffer. Its index is 0,
02. Creating the line buffer. Its index is 0,
03. Creating the arrow buffer. Its index is 1,
04. Creating the ZigZag buffer. Its index is 0,
05. Creating the ZigZag buffer. Its index is 1,
06. Creating the arrow buffer. Its index is 2,
07. Creating the arrow buffer. Its index is 3,
08. Creating the line buffer. Its index is 1,
09. Creating the candle buffer. Its index is 0,
10. Creating the arrow buffer. Its index is 4

Apart from accessing the buffer by its serial index, it can also be accessed by its name in the graphical series, time, data window index and collection list index. We can also get the buffer object which was the last one to be created and added to the collection list. This simplifies setting parameters for the newly created buffer object — just create an object, get it and set additional properties to it.

### Improving classes for working with indicator buffers in multi-period mode

Starting from build 2430, MetaTrader 5 features an increased number of symbols that can be located in the Market Watch window. Now it is 5000 instead of 1000 before build 2430. To work with the symbol list, let's introduce a new macro substitution in the **Defines.mqh** file:

```
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
//--- Describe the function with the error line number
....
//--- Symbol parameters
#define CLR_DEFAULT                    (0xFF000000)               // Default symbol background color in the navigator
#ifdef __MQL5__
   #define SYMBOLS_COMMON_TOTAL        (TerminalInfoInteger(TERMINAL_BUILD)<2430 ? 1000 : 5000)   // Total number of MQL5 working symbols
#else
   #define SYMBOLS_COMMON_TOTAL        (1000)                     // Total number of MQL4 working symbols
#endif
//--- Pending request type IDs
#define PENDING_REQUEST_ID_TYPE_ERR    (1)                        // Type of a pending request created based on the server return code
#define PENDING_REQUEST_ID_TYPE_REQ    (2)                        // Type of a pending request created by request
//--- Timeseries parameters
#define SERIES_DEFAULT_BARS_COUNT      (1000)                     // Required default amount of timeseries data
#define PAUSE_FOR_SYNC_ATTEMPTS        (16)                       // Amount of pause milliseconds between synchronization attempts
#define ATTEMPTS_FOR_SYNC              (5)                        // Number of attempts to receive synchronization with the server
//+------------------------------------------------------------------+
//| Enumerations                                                     |
//+------------------------------------------------------------------+
```

In \\MQL5\\Include\\DoEasy\\Collections\ **SymbolsCollection.mqh**, replace the reserved symbol array size from 1000 to the new macro substitution value in the method of setting the list of used symbols:

```
//+------------------------------------------------------------------+
//| Set the list of used symbols                                     |
//+------------------------------------------------------------------+
bool CSymbolsCollection::SetUsedSymbols(const string &symbol_used_array[])
  {
   ::ArrayResize(this.m_array_symbols,0,SYMBOLS_COMMON_TOTAL);
   ::ArrayCopy(this.m_array_symbols,symbol_used_array);
```

In the method of creating a symbol list, replace the reference value of 1000 with the new macro substitution in the loop index:

```
//+------------------------------------------------------------------+
//| Creating the symbol list (Market Watch or the complete list)     |
//+------------------------------------------------------------------+
bool CSymbolsCollection::CreateSymbolsList(const bool flag)
  {
   bool res=true;
   int total=::SymbolsTotal(flag);
   for(int i=0;i<total && i<SYMBOLS_COMMON_TOTAL;i++)
     {
```

In the same way, replace 1000 to the macro substitution in the method of writing all used symbols and timeframes in \\MQL5\\Include\\DoEasy\Engine.mqh:

```
//+------------------------------------------------------------------+
//| Write all used symbols and timeframes                            |
//| to the ArrayUsedSymbols and ArrayUsedTimeframes arrays           |
//+------------------------------------------------------------------+
void CEngine::WriteSymbolsPeriodsToArrays(void)
  {
//--- Get the list of all created timeseries (created by the number of used symbols)
   CArrayObj *list_timeseries=this.GetListTimeSeries();
   if(list_timeseries==NULL)
      return;
//--- Get the total number of created timeseries
   int total_timeseries=list_timeseries.Total();
   if(total_timeseries==0)
      return;
//--- Set the size of the array of used symbols equal to the number of created timeseries, while
//--- the size of the array of used timeframes is set equal to the maximum possible number of timeframes in the terminal
   if(::ArrayResize(ArrayUsedSymbols,total_timeseries,SYMBOLS_COMMON_TOTAL)!=total_timeseries || ::ArrayResize(ArrayUsedTimeframes,21,21)!=21)
      return;
//--- Set both arrays to zero
   ::ZeroMemory(ArrayUsedSymbols);
   ::ZeroMemory(ArrayUsedTimeframes);
//--- Reset the number of added symbols and timeframes to zero and,
//--- in a loop by the total number of timeseries,
   int num_symbols=0,num_periods=0;
   for(int i=0;i<total_timeseries;i++)
     {
      //--- get the next object of all timeseries of a single symbol
      CTimeSeriesDE *timeseries=list_timeseries.At(i);
      if(timeseries==NULL || this.IsExistSymbol(timeseries.Symbol()))
         continue;
      //--- increase the number of used symbols and (num_symbols variable), and
      //--- write the timeseries symbol name to the array of used symbols by the num_symbols-1 index
      num_symbols++;
      ArrayUsedSymbols[num_symbols-1]=timeseries.Symbol();
      //--- Get the list of all its timeseries from the object of all symbol timeseries
      CArrayObj *list_series=timeseries.GetListSeries();
      if(list_series==NULL)
         continue;
      //--- In the loop by the total number of symbol timeseries,
      int total_series=list_series.Total();
      for(int j=0;j<total_series;j++)
        {
         //--- get the next timeseries object
         CSeriesDE *series=list_series.At(j);
         if(series==NULL || this.IsExistTimeframe(series.Timeframe()))
            continue;
         //--- increase the number of used timeframes and (num_periods variable), and
         //--- write the timeseries timeframe value to the array of used timeframes by num_periods-1 index
         num_periods++;
         ArrayUsedTimeframes[num_periods-1]=series.Timeframe();
        }
     }
//--- Upon the loop completion, change the size of both arrays to match the exact number of added symbols and timeframes
   ::ArrayResize(ArrayUsedSymbols,num_symbols,SYMBOLS_COMMON_TOTAL);
   ::ArrayResize(ArrayUsedTimeframes,num_periods,21);
  }
//+------------------------------------------------------------------+
```

The buffer object parameters feature the parameter specifying the array index to be assigned the next indicator buffer. This allows us to immediately set the next array index to be the first (basic) array for the next buffer when creating a single buffer and adding the necessary number of symbols to it. On the contrary, in case of the indicator buffer (graphical construction) index, this value is used when setting the necessary values for the buffer using the [PlotIndexSetDouble()](https://www.mql5.com/en/docs/customind/plotindexsetdouble), [PlotIndexSetInteger()](https://www.mql5.com/en/docs/customind/plotindexsetinteger) and [PlotIndexSetString()](https://www.mql5.com/en/docs/customind/plotindexsetstring) functions. We do not have the property in the buffer object parameters. To calculate the index of the next graphical construction conveniently, add the new property to the enumeration of integer properties of the buffer object in the **Defines.mqh** file. Also, rename the property storing the index of the base array of the next buffer object:

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
   BUFFER_PROP_NUM_DATAS,                                   // Number of data buffers
   BUFFER_PROP_INDEX_COLOR,                                 // Color buffer index
  };
#define BUFFER_PROP_INTEGER_TOTAL (20)                      // Total number of integer bar properties
#define BUFFER_PROP_INTEGER_SKIP  (2)                       // Number of buffer properties not used in sorting
//+------------------------------------------------------------------+
```

Increase the number of buffer object integer properties from 19 to **20**.

While enumerating possible buffer sorting criteria, add sorting by the newly added property and rename the constant name of sorting by the array index for assigning as the next basic buffer array:

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
//--- Sort by real properties
   SORT_BY_BUFFER_EMPTY_VALUE = FIRST_BUFFER_DBL_PROP,      // Sort by the empty value for plotting where nothing will be drawn
//--- Sort by string properties
   SORT_BY_BUFFER_SYMBOL = FIRST_BUFFER_STR_PROP,           // Sort by the buffer symbol
   SORT_BY_BUFFER_LABEL,                                    // Sort by the name of the graphical indicator series displayed in DataWindow
  };
//+------------------------------------------------------------------+
```

Now when creating any buffer object, we will write the basic array indices and the index of the next graphical series to the current buffer parameters. Thus, in order to find out which indices should be set in the newly created buffer object, we only need to access the previous one (the last created buffer object) and get the necessary values.

This is easier and better since each buffer may have from one to five arrays assigned to it, and we do not have to re-calculate all values of the necessary array indices for a new indicator buffer — all is directly set in the properties of the last created buffer object. These values are recalculated with each newly added indicator buffer depending on the number of arrays used by it.

In \\MQL5\\Include\\DoEasy\ **Datas.mqh**, add the new message index for the newly added buffer property and rename the index constant name of the message concerning the property of the next array for the indicator buffer:

```
//--- CBuffer
   MSG_LIB_TEXT_BUFFER_TEXT_INDEX_BASE,               // Base data buffer index
   MSG_LIB_TEXT_BUFFER_TEXT_INDEX_PLOT,               // Plotted buffer serial number
   MSG_LIB_TEXT_BUFFER_TEXT_INDEX_COLOR,              // Color buffer index
   MSG_LIB_TEXT_BUFFER_TEXT_NUM_DATAS,                // Number of data buffers
   MSG_LIB_TEXT_BUFFER_TEXT_INDEX_NEXT_BASE,          // Index of the array to be assigned as the next indicator buffer
   MSG_LIB_TEXT_BUFFER_TEXT_INDEX_NEXT_PLOT,          // Index of the next drawn buffer
   MSG_LIB_TEXT_BUFFER_TEXT_TIMEFRAME,                // Buffer (timeframe) data period
```

Also, add the message text corresponding to the newly added index:

```
   {"Индекс базового буфера данных","Index of Base data buffer"},
   {"Порядковый номер рисуемого буфера","Plot buffer sequence number"},
   {"Индекс буфера цвета","Color buffer index"},
   {"Количество буферов данных","Number of data buffers"},
   {"Индекс массива для назначения следующим индикаторным буфером","Array index for assignment as the next indicator buffer"},
   {"Индекс следующего по счёту рисуемого буфера","Index of the next drawable buffer"},
   {"Период данных буфера (таймфрейм)","Buffer data Period (Timeframe)"},
```

Since we have changed the name of the constant storing the index of the next basic buffer object array, replace all instances of the BUFFER\_PROP\_INDEX\_NEXT string with BUFFER\_PROP\_INDEX\_NEXT\_BASE in all files of the buffer object classes ( **BufferArrow.mqh, BufferBars.mqh, BufferCandles.mqh, BufferFilling.mqh, BufferHistogram.mqh, BufferHistogram2.mqh, BufferLine.mqh, BufferArrow.mqh, BufferSection.mqh** and **BufferZigZag.mqh**) and improve the virtual method displaying the short buffer object description in the journal.

Let's consider the changes of the method using the CBufferArrow class in \\MQL5\\Include\\DoEasy\\Objects\\Indicators\ **BufferArrow.mqh** as an example:

```
//+------------------------------------------------------------------+
//| Display short buffer description in the journal                  |
//+------------------------------------------------------------------+
void CBufferArrow::PrintShort(void)
  {
   ::Print
     (
      CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_BUFFER),"(P",(string)this.IndexPlot(),"/B",(string)this.IndexBase(),"/C",(string)this.IndexColor(),"): ",
      this.GetStatusDescription(true)," ",this.Symbol()," ",TimeframeDescription(this.Timeframe())
     );
  }
//+------------------------------------------------------------------+
```

The message text has been made as informative as possible. For example, the short description of the arrow buffer class displayed by the improved method looks as follows:

```
Buffer(P0/B0/C1): Drawing with arrows EURUSD H1
```

where:

- "P" means "Plot" — graphical construction index,
- "B" means "Base" — index of the basic array which is the first one to be assigned to the buffer (the remaining buffers of the array are arranged in increasing index starting from the basic array index),
- "C" means "Color" — index of the color array assigned to the buffer.


Thus, if we create all possible buffer objects and display their short descriptions in the journal, we will see the following entries there:

```
Buffer(P0/B0/C1): Drawing with arrows EURUSD H1
Buffer(P1/B2/C3): EURUSD H1 line
Buffer(P2/B4/C5): EURUSD H1 sections
Buffer(P3/B6/C7): Histogram from the zero line EURUSD H1
Buffer(P4/B8/C10): Histogram on two indicator buffers EURUSD H1
Buffer(P5/B11/C13): EURUSD H1 zigzag
Buffer(P6/B14/C16): Color filling between two levels EURUSD H1
Buffer(P7/B16/C20): Display as EURUSD H1 bars
Buffer(P8/B21/C25): Display as EURUSD H1 candles
Buffer[P8/B26/C27]: Calculated buffer
```

Now we can clearly see indicator buffers and arrays with indices (B and C) assigned to them, as well as graphical series indices assigned to the buffers (P).

Here we can see that the graphical series index assigned to the calculated buffer (to be introduced in this article later) matches the one assigned to the previous buffer. In fact, the calculated buffer has no graphical series and color buffer array. Currently, it only contains debugging data. P8 and C27 show which indices of the graphical series and the color buffer array should have the next buffer created after the calculated one.

After creating the indicator buffers, I will change the data, for example, to PN, CN or PX, CX which means the calculated buffer has no graphical series and color buffer array or even remove it leaving only B — the index of the array assigned to the buffer.

We have descendant objects of the [basic abstract buffer object](https://www.mql5.com/en/articles/7821), which clarify the type of the created indicator buffer by its drawing type. However, we lack yet another buffer object — the calculated one used to perform calculations requiring the array but without displaying anything on the chart. Such an array is an indicator buffer. The terminal subsystem is responsible for its distribution along with the drawn indicator buffer arrays.

Create such a descendant object with the CBufferCalculate class name. In \\MQL5\\Include\\DoEasy\\Objects\\Indicators\\, create the new **BufferCalculate.mqh** file and fill it with the necessary content right away:

```
//+------------------------------------------------------------------+
//|                                              BufferCalculate.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Buffer.mqh"
//+------------------------------------------------------------------+
//| Calculated buffer                                                |
//+------------------------------------------------------------------+
class CBufferCalculate : public CBuffer
  {
private:

public:
//--- Constructor
                     CBufferCalculate(const uint index_plot,const uint index_array) :
                        CBuffer(BUFFER_STATUS_NONE,BUFFER_TYPE_CALCULATE,index_plot,index_array,1,0,"Calculate") {}
//--- Supported integer properties of a buffer
   virtual bool      SupportProperty(ENUM_BUFFER_PROP_INTEGER property);
//--- Supported real properties of a buffer
   virtual bool      SupportProperty(ENUM_BUFFER_PROP_DOUBLE property);
//--- Supported string properties of a buffer
   virtual bool      SupportProperty(ENUM_BUFFER_PROP_STRING property);
//--- Display a short buffer description in the journal
   virtual void      PrintShort(void);

//--- Set the value to the data buffer array
   void              SetData(const uint series_index,const double value)               { this.SetBufferValue(0,series_index,value);       }
//--- Return the value from the data buffer array
   double            GetData(const uint series_index)                            const { return this.GetDataBufferValue(0,series_index);  }

  };
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
   return false;
  }
//+------------------------------------------------------------------+
//| Return 'true' if a buffer supports a passed                      |
//| string property, otherwise return 'false'                        |
//+------------------------------------------------------------------+
bool CBufferCalculate::SupportProperty(ENUM_BUFFER_PROP_STRING property)
  {
   return false;
  }
//+------------------------------------------------------------------+
//| Display short buffer description in the journal                  |
//+------------------------------------------------------------------+
void CBufferCalculate::PrintShort(void)
  {
   ::Print
     (
      CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_BUFFER),"[P",(string)this.IndexPlot(),"/B",(string)this.IndexBase(),"/C",(string)this.IndexColor(),"]: ",
      this.GetTypeBufferDescription()
     );
  }
//+------------------------------------------------------------------+
```

I have already considered the structure and operation principles of the descendant classes of the basic abstract buffer [in the article 43](https://www.mql5.com/en/articles/7868). They are similar to the ones described here.

Now that we have a complete set of all descendant classes of the basic buffer object (CBuffer), let's modify it as well.

Sometimes, it is necessary to perform some one-time actions with the indicator buffer by request or conditions. For example, you may want to clear or update the buffer data by pressing a button. To do this, let's introduce the flag specifying that no required action has been performed yet.

In the private class section, declare the class member variable for storing such a flag, while in the public section, declare two methods for settingand getting the value variable:

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
   bool              m_act_state_trigger;                                        // Auxiliary buffer status switch flag
//--- Return the index of the array the buffer's (1) double and (2) string properties are located at
   int               IndexProp(ENUM_BUFFER_PROP_DOUBLE property)           const { return(int)property-BUFFER_PROP_INTEGER_TOTAL;                           }
   int               IndexProp(ENUM_BUFFER_PROP_STRING property)           const { return(int)property-BUFFER_PROP_INTEGER_TOTAL-BUFFER_PROP_DOUBLE_TOTAL;  }
//--- Set the graphical construction type by buffer type and status
   void              SetDrawType(void);
//--- Return the adjusted buffer array index
   int               GetCorrectIndexBuffer(const uint buffer_index) const;

protected:
   struct SDataBuffer { double Array[]; };                                       // Structure for storing buffer object buffer arrays
   SDataBuffer       DataBuffer[];                                               // Array of all object indicator buffers
   double            ColorBufferArray[];                                         // Color buffer array
   int               ArrayColors[];                                              // Array for storing colors

public:
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
   virtual bool      SupportProperty(ENUM_BUFFER_PROP_INTEGER property)          { return true;       }
   virtual bool      SupportProperty(ENUM_BUFFER_PROP_DOUBLE property)           { return true;       }
   virtual bool      SupportProperty(ENUM_BUFFER_PROP_STRING property)           { return true;       }

//--- Compare CBuffer objects by all possible properties (for sorting the lists by a specified buffer object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CBuffer objects by all properties (to search for equal buffer objects)
   bool              IsEqual(CBuffer* compared_obj) const;

//--- Set the buffer name
   void              SetName(const string name)                                  { this.m_name=name;  }
//--- (1) Set and (2) return the buffer status switch flag
   void              SetActStateFlag(const bool flag)                            { this.m_act_state_trigger=flag;    }
   bool              GetActStateFlag(void)                                 const { return this.m_act_state_trigger;  }

//--- Default constructor
                     CBuffer(void){;}
```

For example, if we disable the buffer display by pressing the button, we can also set the buffer [drawing style](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles) to DRAW\_NONE in addition to clearing all array data. To achieve this, introduce the additional method setting the drawing type to the buffer object in the public section of the class:

```
public:
//--- Send description of buffer properties to the journal (full_prop=true - all properties, false - only supported ones)
   void              Print(const bool full_prop=false);
//--- Display a short buffer description in the journal (implementation in the descendants)
   virtual void      PrintShort(void) {;}

//--- Set (1) the arrow code, (2) vertical shift of arrows, (3) symbol, (4) timeframe, (5) buffer activity flag
//--- (6) drawing type, (7) number of initial bars without drawing, (8) flag of displaying construction values in DataWindow,
//--- (9) shift of the indicator graphical construction along the time axis, (10) line style, (11) line width,
//--- (12) total number of colors, (13) one drawing color, (14) color of drawing in the specified color index,
//--- (15) drawing colors from the color array, (16) empty value, (17) name of the graphical series displayed in DataWindow
   virtual void      SetArrowCode(const uchar code)                  { return;                                                            }
   virtual void      SetArrowShift(const int shift)                  { return;                                                            }
   void              SetSymbol(const string symbol)                  { this.SetProperty(BUFFER_PROP_SYMBOL,symbol);                       }
   void              SetTimeframe(const ENUM_TIMEFRAMES timeframe)   { this.SetProperty(BUFFER_PROP_TIMEFRAME,timeframe);                 }
   void              SetActive(const bool flag)                      { this.SetProperty(BUFFER_PROP_ACTIVE,flag);                         }
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
```

Let's write its implementation outside the class body:

```
//+------------------------------------------------------------------+
//| Set the passed graphical construction type                       |
//+------------------------------------------------------------------+
void CBuffer::SetDrawType(const ENUM_DRAW_TYPE draw_type)
  {
   if(this.TypeBuffer()==BUFFER_TYPE_CALCULATE)
      return;
   this.SetProperty(BUFFER_PROP_DRAW_TYPE,draw_type);
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_DRAW_TYPE,draw_type);
  }
//+------------------------------------------------------------------+
```

Here, if the buffer type is Calculation, it has no rendering — exit the method.

Set the drawing type (passed to the method) to the object property and set the style for the buffer.

Since I have replaced the BUFFER\_PROP\_INDEX\_NEXT constant with BUFFER\_PROP\_INDEX\_NEXT\_BASE, rename the appropriate IndexNextBuffer() method to IndexNextBaseBuffer() and add yet another method returning the index of the next drawn buffer:

```
//--- Return (1) the serial number of the drawn buffer, (2) bound array index, (3) color buffer index,
//--- (4) index of the first free bound array, (5) index of the next drawn buffer, (6) buffer data period, (7) buffer status,
//--- (8) buffer type, (9) buffer usage flag, (10) arrow code, (11) arrow shift for DRAW_ARROW style,
//--- (12) number of initial bars that are not drawn and values in DataWindow, (13) graphical construction type,
//--- (14) flag of displaying construction values in DataWindow, (15) indicator graphical construction shift along the time axis,
//--- (16) drawing line style, (17) drawing line width, (18) number of colors, (19) drawing color, number of buffers for construction
//--- (20) set empty value, (21) buffer symbol, (22) name of the indicator graphical series displayed in DataWindow
   int               IndexPlot(void)                           const { return (int)this.GetProperty(BUFFER_PROP_INDEX_PLOT);              }
   int               IndexBase(void)                           const { return (int)this.GetProperty(BUFFER_PROP_INDEX_BASE);              }
   int               IndexColor(void)                          const { return (int)this.GetProperty(BUFFER_PROP_INDEX_COLOR);             }
   int               IndexNextBaseBuffer(void)                 const { return (int)this.GetProperty(BUFFER_PROP_INDEX_NEXT_BASE);         }
   int               IndexNextPlotBuffer(void)                 const { return (int)this.GetProperty(BUFFER_PROP_INDEX_NEXT_PLOT);         }
   ENUM_TIMEFRAMES   Timeframe(void)                           const { return (ENUM_TIMEFRAMES)this.GetProperty(BUFFER_PROP_TIMEFRAME);   }
```

Finally, add four methods at the very end of the class body — two methods for initializing object arraysand two methods for filling them:

```
//--- Return the size of the data buffer array
   virtual int       GetDataTotal(const uint buffer_index=0)   const;
//--- Return the value from the specified index of the specified (1) data, (2) color index and (3) color buffer arrays
   double            GetDataBufferValue(const uint buffer_index,const uint series_index) const;
   int               GetColorBufferValueIndex(const uint series_index) const;
   color             GetColorBufferValueColor(const uint series_index) const;
//--- Set the value to the specified index of the specified (1) data and (2) color buffer arrays
   void              SetBufferValue(const uint buffer_index,const uint series_index,const double value);
   void              SetBufferColorIndex(const uint series_index,const uchar color_index);
//--- Initialize all object buffers by (1) a specified value, (2) the empty value specified for the object
   void              InitializeAll(const double value,const uchar color_index);
   void              InitializeAll(void);
//--- Fill all data buffers with empty values in the specified timeseries index
   void              ClearData(const int series_index);
//--- Fill the specified buffer with data from (1) the passed timeseries and (2) the array
   void              FillAsSeries(const int buffer_index,CSeriesDE *series,const ENUM_SORT_BAR_MODE property);
   void              FillAsSeries(const int buffer_index,const double &array[]);

  };
//+------------------------------------------------------------------+
```

In fact, these are two overloaded methods accepting different set of parameters for different situations.

The parametric method of initializing all object buffers as parameters takes the value used to initialize all buffer object arrays assigned by indicator buffers and the color index used to initialize the array assigned by the color buffer in the buffer object.

The method with no inputs initializes all buffer object arrays with the value set as "empty" in the buffer object properties, while the color array is initialized by zero — the very first color set to the buffer object (or the only one if there is only one color).

Let's implement the methods outside the class body:

```
//+------------------------------------------------------------------+
//| Initialize all object buffers by the specified value             |
//+------------------------------------------------------------------+
void CBuffer::InitializeAll(const double value,const uchar color_index)
  {
   for(int i=0;i<this.GetProperty(BUFFER_PROP_NUM_DATAS);i++)
      ::ArrayInitialize(this.DataBuffer[i].Array,value);
   if(this.Status()!=BUFFER_STATUS_FILLING && this.TypeBuffer()!=BUFFER_TYPE_CALCULATE)
      ::ArrayInitialize(this.ColorBufferArray,(color_index>this.ColorsTotal()-1 ? 0 : color_index));
  }
//+------------------------------------------------------------------+
//| Initialize all object buffers                                    |
//| by the empty value set for the object                            |
//+------------------------------------------------------------------+
void CBuffer::InitializeAll(void)
  {
   for(int i=0;i<this.GetProperty(BUFFER_PROP_NUM_DATAS);i++)
      ::ArrayInitialize(this.DataBuffer[i].Array,this.EmptyValue());
   if(this.Status()!=BUFFER_STATUS_FILLING && this.TypeBuffer()!=BUFFER_TYPE_CALCULATE)
      ::ArrayInitialize(this.ColorBufferArray,0);
  }
//+------------------------------------------------------------------+
```

First, in the loop by the number of buffer object arrays assigned by indicator buffers, initialize each next array using the value passed to the method in the first method and the value set for the buffer in the second method. Next, if the buffer object is not of "Color filling between two levels" drawing style and is not a calculation buffer (the buffers have no color buffer), the color array is initialized using the value passed to the method in the first method and using zero in the second method.

```
//+-----------------------------------------------------------------------+
//| Fill the specified buffer with data from the passed timeseries' bars  |
//+-----------------------------------------------------------------------+
void CBuffer::FillAsSeries(const int buffer_index,CSeriesDE *series,const ENUM_SORT_BAR_MODE property)
  {
//--- If an empty timeseries is passed or a property to fill the array is a string one, leave the method
   if(series==NULL || property>FIRST_BAR_STR_PROP-1)
      return;
//--- Get the list of all its timeseries from the passed timeseries object
   CArrayObj *list=series.GetList();
   if(list==NULL || list.Total()==0)
      return;
//--- Get the number of bars in the timeseries list
   int total=list.Total();
//--- In the loop from the very last timeseries list element (from the current bar)
   int n=0;
   for(int i=total-1;i>WRONG_VALUE && !::IsStopped();i--)
     {
      //--- get the next bar object by the loop index,
      CBar *bar=list.At(i);
      //--- Set the value of the copied property
      double value=
        (bar==NULL ? this.EmptyValue()                                                    :
         property<FIRST_BAR_DBL_PROP ? bar.GetProperty((ENUM_BAR_PROP_INTEGER)property)   :
         property<FIRST_BAR_STR_PROP ? bar.GetProperty((ENUM_BAR_PROP_DOUBLE)property)    :
         this.EmptyValue()
        );
      //--- calculate the index, based on which the bar property is saved to the buffer, and
      //--- write the value of the obtained bar property to the buffer array using the calculated index
      n=total-1-i;
      this.SetBufferValue(buffer_index,n,value);
     }
  }
//+------------------------------------------------------------------+
//| Fill the specified buffer with data from the passed array        |
//+------------------------------------------------------------------+
void CBuffer::FillAsSeries(const int buffer_index,const double &array[])
  {
//--- Get the copied array size
   int total=::ArraySize(array);
   if(total==0)
      return;
//--- In the loop from the very last array element (from the current bar)
   int n=0;
   for(int i=total-1;i>WRONG_VALUE && !::IsStopped();i--)
     {
      //--- calculate the index, based on which the array value is saved to the buffer, and
      //--- write the value of the array cell by i index using the calculated index
      n=total-1-i;
      this.SetBufferValue(buffer_index,n,array[i]);
     }
  }
//+------------------------------------------------------------------+
```

These two methods are commented in details in the code listing. The first method receives the pointer to the timeseries object whose data is to be used to fill the indicator buffer, while the second method receives a double array. Its data is also used to fill the indicator buffer.

Let's adjust the closed class constructor:

```
//+------------------------------------------------------------------+
//| Closed parametric constructor                                    |
//+------------------------------------------------------------------+
CBuffer::CBuffer(ENUM_BUFFER_STATUS buffer_status,
                 ENUM_BUFFER_TYPE buffer_type,
                 const uint index_plot,
                 const uint index_base_array,
                 const int num_datas,
                 const int width,
                 const string label)
  {
   this.m_type=COLLECTION_BUFFERS_ID;
   this.m_act_state_trigger=true;
//--- Save integer properties
   this.m_long_prop[BUFFER_PROP_STATUS]                        = buffer_status;
   this.m_long_prop[BUFFER_PROP_TYPE]                          = buffer_type;
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
   this.m_long_prop[BUFFER_PROP_INDEX_COLOR]                   = this.GetProperty(BUFFER_PROP_INDEX_BASE)+this.GetProperty(BUFFER_PROP_NUM_DATAS);
   this.m_long_prop[BUFFER_PROP_INDEX_NEXT_BASE]               = this.GetProperty(BUFFER_PROP_INDEX_COLOR)+
                                                                    (this.Status()==BUFFER_STATUS_FILLING || this.TypeBuffer()==BUFFER_TYPE_CALCULATE ? 0 : 1);
   this.m_long_prop[BUFFER_PROP_INDEX_NEXT_PLOT]               = (this.TypeBuffer()>BUFFER_TYPE_CALCULATE ? index_plot+1 : index_plot);

//--- Save real properties
   this.m_double_prop[this.IndexProp(BUFFER_PROP_EMPTY_VALUE)] = (this.TypeBuffer()>BUFFER_TYPE_CALCULATE ? EMPTY_VALUE : 0);
//--- Save string properties
   this.m_string_prop[this.IndexProp(BUFFER_PROP_SYMBOL)]      = ::Symbol();
   this.m_string_prop[this.IndexProp(BUFFER_PROP_LABEL)]       = (this.TypeBuffer()>BUFFER_TYPE_CALCULATE ? label : NULL);

//--- If failed to change the size of the indicator buffer array, display the appropriate message indicating the string
   if(::ArrayResize(this.DataBuffer,(int)this.GetProperty(BUFFER_PROP_NUM_DATAS))==WRONG_VALUE)
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_FAILED_DRAWING_ARRAY_RESIZE),". ",CMessage::Text(MSG_LIB_SYS_ERROR),": ",(string)::GetLastError());

//--- If failed to change the size of the color array (only for a non-calculated buffer), display the appropriate message indicating the string
   if(this.TypeBuffer()>BUFFER_TYPE_CALCULATE)
      if(::ArrayResize(this.ArrayColors,(int)this.ColorsTotal())==WRONG_VALUE)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_FAILED_COLORS_ARRAY_RESIZE),". ",CMessage::Text(MSG_LIB_SYS_ERROR),": ",(string)::GetLastError());

//--- For DRAW_FILLING, fill in the color array with two default colors
   if(this.Status()==BUFFER_STATUS_FILLING)
     {
      this.SetColor(clrBlue,0);
      this.SetColor(clrRed,1);
     }

//--- Bind indicator buffers with arrays
//--- In a loop by the number of indicator buffers
   int total=::ArraySize(DataBuffer);
   for(int i=0;i<total;i++)
     {
      //--- calculate the index of the next array and
      //--- bind the indicator buffer by the calculated index with the dynamic array
      //--- located by the i loop index in the DataBuffer array
      int index=(int)this.GetProperty(BUFFER_PROP_INDEX_BASE)+i;
      ::SetIndexBuffer(index,this.DataBuffer[i].Array,(this.TypeBuffer()==BUFFER_TYPE_DATA ? INDICATOR_DATA : INDICATOR_CALCULATIONS));
      //--- Set indexation flag as in the timeseries to all buffer arrays
      ::ArraySetAsSeries(this.DataBuffer[i].Array,true);
     }
//--- Bind the color buffer with the array (only for a non-calculated buffer and not for the filling buffer)
   if(this.Status()!=BUFFER_STATUS_FILLING && this.TypeBuffer()!=BUFFER_TYPE_CALCULATE)
     {
      ::SetIndexBuffer((int)this.GetProperty(BUFFER_PROP_INDEX_COLOR),this.ColorBufferArray,INDICATOR_COLOR_INDEX);
      ::ArraySetAsSeries(this.ColorBufferArray,true);
     }
//--- If this is a calculated buffer, all is done
   if(this.TypeBuffer()==BUFFER_TYPE_CALCULATE)
      return;
//--- Set integer parameters of the drawn buffer
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_DRAW_TYPE,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_DRAW_TYPE));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_ARROW,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_ARROW_CODE));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_ARROW_SHIFT,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_ARROW_SHIFT));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_DRAW_BEGIN,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_DRAW_BEGIN));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_SHOW_DATA,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_SHOW_DATA));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_SHIFT,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_SHIFT));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LINE_STYLE,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_LINE_STYLE));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LINE_WIDTH,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_LINE_WIDTH));
   this.SetColor((color)this.GetProperty(BUFFER_PROP_COLOR));
//--- Set real buffer parameters
   ::PlotIndexSetDouble((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_EMPTY_VALUE,this.GetProperty(BUFFER_PROP_EMPTY_VALUE));
//--- Set string buffer parameters
   ::PlotIndexSetString((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LABEL,this.GetProperty(BUFFER_PROP_LABEL));
  }
//+------------------------------------------------------------------+
```

The changes have affected setting the **m\_act\_state\_trigger** variable to true, which means the buffer was already handled while being created (for example, when pressing the button, we are going to check "whether the necessary action was performed with the buffer object", if not, then it should be performed. The set flag does not allow a false action to be performed).

The remaining improvements are related to the fact that we already have a calculated buffer featuring no color array. So there is no need to set graphical series properties for it. Thus, when calculating array indices, the buffer object type is now considered. If this is a calculated buffer, the appropriate adjustments are introduced in the index calculation. Before setting the graphical series values for indicator buffers, we check the buffer object type. If this is a calculated buffer, leave the method — the calculated buffer has no graphical series.

For methods of setting buffer object graphical series properties, introduce the check for the buffer type. If this is a calculated buffer, leave the method at once:

```
//+------------------------------------------------------------------+
//| Set the number of initial bars                                   |
//| without drawing and values in DataWindow                         |
//+------------------------------------------------------------------+
void CBuffer::SetDrawBegin(const int value)
  {
   if(this.TypeBuffer()==BUFFER_TYPE_CALCULATE)
      return;
   this.SetProperty(BUFFER_PROP_DRAW_BEGIN,value);
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_DRAW_BEGIN,value);
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Set the number of colors                                         |
//+------------------------------------------------------------------+
void CBuffer::SetColorNumbers(const int number)
  {
   if(number>IND_COLORS_TOTAL || this.TypeBuffer()==BUFFER_TYPE_CALCULATE)
      return;
   int n=(this.Status()!=BUFFER_STATUS_FILLING ? number : 2);
   this.SetProperty(BUFFER_PROP_COLOR_INDEXES,n);
   ::ArrayResize(this.ArrayColors,n);
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_COLOR_INDEXES,n);
  }
//+------------------------------------------------------------------+
```

Similar checks were introduced to all methods that are not needed for the calculated buffer — we will not consider them here.

Sometimes, when calculating the bar index the indicator buffer value should be set in, the resulting calculated value turns out to be less than zero. This happens when calculating the bar index on the current chart while the data of a chart period not equal to the current one is displayed on it. In order to avoid additional checks for the validity of the calculated index, it would be much easier to check the passed index for a negative value and exit the method in the methods responsible for writing values to arrays by a specified index:

```
//+------------------------------------------------------------------+
//| Set the value to the specified timeseries index                  |
//| for the specified data buffer array                              |
//+------------------------------------------------------------------+
void CBuffer::SetBufferValue(const uint buffer_index,const uint series_index,const double value)
  {
   if(this.GetDataTotal(buffer_index)==0)
      return;
   int correct_buff_index=this.GetCorrectIndexBuffer(buffer_index);
   int data_total=this.GetDataTotal(buffer_index);
   int data_index=((int)series_index<data_total ? (int)series_index : data_total-1);
   if(data_index<0)
      return;
   this.DataBuffer[correct_buff_index].Array[data_index]=value;
  }
//+------------------------------------------------------------------+
//| Set the color index to the specified timeseries index            |
//| of the color buffer array                                        |
//+------------------------------------------------------------------+
void CBuffer::SetBufferColorIndex(const uint series_index,const uchar color_index)
  {
   if(this.GetDataTotal(0)==0 || color_index>this.ColorsTotal()-1 || this.Status()==BUFFER_STATUS_FILLING || this.Status()==BUFFER_STATUS_NONE)
      return;
   int data_total=this.GetDataTotal(0);
   int data_index=((int)series_index<data_total ? (int)series_index : data_total-1);
   if(::ArraySize(this.ColorBufferArray)==0)
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_INVALID_PROPERTY_BUFF));
   if(data_index<0)
      return;
   this.ColorBufferArray[data_index]=color_index;
  }
//+------------------------------------------------------------------+
```

I have also introduced some minor changes in the class listing. There is no point in describing them here since they do not affect the code performance. You can see all the changes in the files attached below.

Now let's make additions to the [CBuffersCollection collection class of indicator buffers](https://www.mql5.com/en/articles/7886) in \\MQL5\\Include\\DoEasy\\Collections\ **BuffersCollection.mqh**.

Since I am introducing the ability to work with chart periods different from the current one, the buffer collection class should have access to the [timeseries collection class](https://www.mql5.com/en/articles/7663). From the timeseries collection, get all the timeseries necessary for the indicator buffer construction calculation on the current chart using the data from other period charts.

To let the buffer collection class access the timeseries collection class, simply pass the pointer to the timeseries collection to the collection of indicator buffers. Let's use this pointer to select the necessary data and methods for receiving it in the class.

Similar to changing the constant name in all buffer object class files, all instances of the BUFFER\_PROP\_INDEX\_PLOT string have already been replaced with BUFFER\_PROP\_INDEX\_NEXT\_PLOT here, while accessing the IndexNextBuffer() method has been replaced with its renamed version IndexNextBaseBuffer().

Include the calculated buffer object class file and the timeseries collection class file to the indicator buffer collection class file and define the private class member variable for storing the pointer to the timeseries collection class object:

```
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Objects\Indicators\BufferArrow.mqh"
#include "..\Objects\Indicators\BufferLine.mqh"
#include "..\Objects\Indicators\BufferSection.mqh"
#include "..\Objects\Indicators\BufferHistogram.mqh"
#include "..\Objects\Indicators\BufferHistogram2.mqh"
#include "..\Objects\Indicators\BufferZigZag.mqh"
#include "..\Objects\Indicators\BufferFilling.mqh"
#include "..\Objects\Indicators\BufferBars.mqh"
#include "..\Objects\Indicators\BufferCandles.mqh"
#include "..\Objects\Indicators\BufferCalculate.mqh"
#include "TimeSeriesCollection.mqh"
//+------------------------------------------------------------------+
//| Collection of indicator buffers                                  |
//+------------------------------------------------------------------+
class CBuffersCollection : public CObject
  {
private:
   CListObj                m_list;                       // Buffer object list
   CTimeSeriesCollection  *m_timeseries;                 // Pointer to the timeseries collection object
```

Although there are multiple methods to be added, most of them have similar functions for different buffer objects. So to avoid describing each method (especially since they are all described in the appropriate comments), let's declare all new methods in the class body and analyze their structure and principles selectively:

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
//--- Return (1) oneself and (2) the timeseries list
   CBuffersCollection     *GetObject(void)               { return &this;                                       }
   CArrayObj              *GetList(void)                 { return &this.m_list;                                }
//--- Return the number of (1) drawn buffers, (2) all arrays used to build all buffers in the collection
   int                     PropertyPlotsTotal(void);
   int                     PropertyBuffersTotal(void);

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

//--- Return the buffer by (1) the graphical series name, (2) timeframe, (2) Plot index and (3) collection list object
   CBuffer                *GetBufferByLabel(const string plot_label);
   CBuffer                *GetBufferByTimeframe(const ENUM_TIMEFRAMES timeframe);
   CBuffer                *GetBufferByPlot(const int plot_index);
   CBuffer                *GetBufferByListIndex(const int index_list);
//--- Return buffers by their status by the specified serial number
//--- (0 - the very first created buffer with the ХХХ drawing style, 1,2,N - subsequent ones)
   CBufferArrow           *GetBufferArrow(const int number);
   CBufferLine            *GetBufferLine(const int number);
   CBufferSection         *GetBufferSection(const int number);
   CBufferHistogram       *GetBufferHistogram(const int number);
   CBufferHistogram2      *GetBufferHistogram2(const int number);
   CBufferZigZag          *GetBufferZigZag(const int number);
   CBufferFilling         *GetBufferFilling(const int number);
   CBufferBars            *GetBufferBars(const int number);
   CBufferCandles         *GetBufferCandles(const int number);
   CBufferCalculate       *GetBufferCalculate(const int number);

//--- Initialize all drawn buffers by a (1) specified value, (2) empty value set for the buffer object
   void                    InitializePlots(const double value,const uchar color_index);
   void                    InitializePlots(void);
//--- Initialize all calculated buffers by a (1) specified value, (2) empty value set for the buffer object
   void                    InitializeCalculates(const double value);
   void                    InitializeCalculates(void);
//--- Set color values from the passed color array for all indicator buffers within the collection
   void                    SetColors(const color &array_colors[]);

//--- Set the value by the timeseries index for the (1) arrow, (2) line, (3) section, (4) zero line histogram,
//--- (5) two buffer histogram, (6) zigzag, (7) filling, (8) bar, (9) candle, (10) calculated buffer
   void                    SetBufferArrowValue(const int number,const int series_index,const double value,const uchar color_index,bool as_current=false);
   void                    SetBufferLineValue(const int number,const int series_index,const double value,const uchar color_index,bool as_current=false);
   void                    SetBufferSectionValue(const int number,const int series_index,const double value,const uchar color_index,bool as_current=false);
   void                    SetBufferHistogramValue(const int number,const int series_index,const double value,const uchar color_index,bool as_current=false);
   void                    SetBufferHistogram2Value(const int number,const int series_index,const double value1,const double value2,const uchar color_index,bool as_current=false);
   void                    SetBufferZigZagValue(const int number,const int series_index,const double value1,const double value2,const uchar color_index,bool as_current=false);
   void                    SetBufferFillingValue(const int number,const int series_index,const double value1,const double value2,bool as_current=false);
   void                    SetBufferBarsValue(const int number,const int series_index,const double open,const double high,const double low,const double close,const uchar color_index,bool as_current=false);
   void                    SetBufferCandlesValue(const int number,const int series_index,const double open,const double high,const double low,const double close,const uchar color_index,bool as_current=false);
   void                    SetBufferCalculateValue(const int number,const int series_index,const double value);

//--- Set the color index to the color buffer by its serial number of
//--- (1) arrows, (2) lines, (3) sections, (4) zero histogram
//--- (5) histogram on two buffers, (6) zigzag, (7) filling, (8) bars and (9) candles
//--- (0 - the very first created buffer with the ХХХ drawing style, 1,2,N - subsequent ones)
   void                    SetBufferArrowColorIndex(const int number,const int series_index,const uchar color_index);
   void                    SetBufferLineColorIndex(const int number,const int series_index,const uchar color_index);
   void                    SetBufferSectionColorIndex(const int number,const int series_index,const uchar color_index);
   void                    SetBufferHistogramColorIndex(const int number,const int series_index,const uchar color_index);
   void                    SetBufferHistogram2ColorIndex(const int number,const int series_index,const uchar color_index);
   void                    SetBufferZigZagColorIndex(const int number,const int series_index,const uchar color_index);
   void                    SetBufferFillingColorIndex(const int number,const int series_index,const uchar color_index);
   void                    SetBufferBarsColorIndex(const int number,const int series_index,const uchar color_index);
   void                    SetBufferCandlesColorIndex(const int number,const int series_index,const uchar color_index);

//--- Clear buffer data by its index in the list in the specified timeseries bar
   void                    Clear(const int buffer_list_index,const int series_index);
//--- Clear data by the timeseries index for the (1) arrow, (2) line, (3) section, (4) zero line histogram,
//--- (5) histogram on two buffers, (6) zigzag, (7) filling, (8) bars and (9) candles
   void                    ClearBufferArrow(const int number,const int series_index);
   void                    ClearBufferLine(const int number,const int series_index);
   void                    ClearBufferSection(const int number,const int series_index);
   void                    ClearBufferHistogram(const int number,const int series_index);
   void                    ClearBufferHistogram2(const int number,const int series_index);
   void                    ClearBufferZigZag(const int number,const int series_index);
   void                    ClearBufferFilling(const int number,const int series_index);
   void                    ClearBufferBars(const int number,const int series_index);
   void                    ClearBufferCandles(const int number,const int series_index);

//--- Constructor
                           CBuffersCollection();
//--- Get pointers to the timeseries collection (the method is called in the CollectionOnInit() method of the CEngine object)
   void                    OnInit(CTimeSeriesCollection *timeseries) { this.m_timeseries=timeseries;  }
  };
//+------------------------------------------------------------------+
```

Let's consider the assignments of the newly added methods. While describing the method assignment, I will immediately consider their implementation which has been moved outside the class body, just like most methods of library classes.

**GetIndexLastPlot()** returns the index of the graphical series of the last created buffer object:

```
//+------------------------------------------------------------------+
//| Return the index of the last drawn buffer                        |
//+------------------------------------------------------------------+
int CBuffersCollection::GetIndexLastPlot(void)
  {
//--- Return the pointer to the list. If the list is not created for some reason, return -1
   CArrayObj *list=this.GetList();
   if(list==NULL)
      return WRONG_VALUE;
//--- Get the index of the drawn buffer with the highest value. If the FindBufferMax() method returns -1,
//--- the list is empty, return index 0 for the very first buffer in the list
   int index=CSelect::FindBufferMax(list,BUFFER_PROP_INDEX_PLOT);
   if(index==WRONG_VALUE)
      return 0;
//--- if the index is not -1,
//--- get the buffer object from the list by its index
   CBuffer *buffer=this.m_list.At(index);
   if(buffer==NULL)
      return WRONG_VALUE;
//--- Return the Plot index of the buffer object
   return buffer.IndexPlot();
  }
//+------------------------------------------------------------------+
```

The method is quite specific and is intended mostly for the method of creating a new buffer, since the value returned by it is tied to the size of the buffer list (if the list is empty, then zero, rather than -1, is returned). If this method turns out to be necessary for other purposes later, then I will slightly correct it. If it is no longer needed except for the current application, then the method will be made private.

The **GetBarsData()** private method is used to get data from other timeseries (non-native for the buffer object) and returns the number of bars of the current timeframe included into one bar of the buffer object chart period:

```
//+------------------------------------------------------------------+
//| Get data of the necessary timeseries and bars                    |
//| for working with a single bar of the buffer                      |
//+------------------------------------------------------------------+
int CBuffersCollection::GetBarsData(CBuffer *buffer,const int series_index,int &index_bar_period)
  {
//--- Get timeseries of the current chart and the chart of the buffer timeframe
   CSeriesDE *series_current=this.m_timeseries.GetSeries(buffer.Symbol(),PERIOD_CURRENT);
   CSeriesDE *series_period=this.m_timeseries.GetSeries(buffer.Symbol(),buffer.Timeframe());
   if(series_current==NULL || series_period==NULL)
      return WRONG_VALUE;
//--- Get the bar object of the current timeseries corresponding to the required timeseries index
   CBar *bar_current=series_current.GetBar(series_index);
   if(bar_current==NULL)
      return WRONG_VALUE;
//--- Get the timeseries bar object of the buffer chart period corresponding to the time the timeseries bar of the current chart falls into
   CBar *bar_period=m_timeseries.GetBarSeriesFirstFromSeriesSecond(NULL,PERIOD_CURRENT,bar_current.Time(),NULL,series_period.Timeframe());
   if(bar_period==NULL)
      return WRONG_VALUE;
//--- Write down the bar index on the current timeframe which falls into the bar start time of the buffer object chart
   index_bar_period=bar_period.Index(PERIOD_CURRENT);
//--- Calculate the amount of bars of the current timeframe included into one bar of the buffer object chart period
//--- and return this value (1 if the result is 0)
   int num_bars=::PeriodSeconds(bar_period.Timeframe())/::PeriodSeconds(bar_current.Timeframe());
   return(num_bars>0 ? num_bars : 1);
  }
//+------------------------------------------------------------------+
```

The method will be widely used for calculating the bars of the current chart, in which you need to write the value of the indicator buffer with data from the chart period different from the current timeframe.

The **PropertyPlotsTotal()** and **PropertyBuffersTotal()** methods are former PlotsTotal() and BuffersTotal() methods renamed for more clarity. They return correct values to [#property indicator\_plots and #property indicator\_buffers](https://www.mql5.com/en/docs/basis/preprosessor/compilation) accordingly for specifying them as indicator program properties.

In the PropertyBuffersTotal() method, the constant names have also been replaced with the renamed ones (BUFFER\_PROP\_INDEX\_NEXT have been replaced with BUFFER\_PROP\_INDEX\_NEXT\_BASE).

The PropertyPlotsTotal() method has been rewritten due to the emergence of a calculated buffer. So the data now should be retrieved in a different way:

```
//+------------------------------------------------------------------+
//| Return the number of drawn buffers                               |
//+------------------------------------------------------------------+
int CBuffersCollection::PropertyPlotsTotal(void)
  {
   CArrayObj *list=CSelect::ByBufferProperty(this.GetList(),BUFFER_PROP_TYPE,BUFFER_TYPE_DATA,EQUAL);
   return(list!=NULL ? list.Total() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
```

Get the list of drawn-only buffer objects and return its size. If the list is empty, -1 is returned.

**CreateCalculate()** serves for creating the object of the calculated (not drawn) buffer.

The method is implemented directly in the class body and returns the result of the method for creating a new buffer (considered [in the previous article](https://www.mql5.com/en/articles/7886)) the calculated buffer status is passed to:

```
bool CreateCalculate(void) { return this.CreateBuffer(BUFFER_STATUS_NONE); }
```

The **GetBufferByLabel()** method returns the pointer to the buffer object by the graphical series name:

```
//+------------------------------------------------------------------+
//| Return the buffer by the graphical series name                   |
//+------------------------------------------------------------------+
CBuffer *CBuffersCollection::GetBufferByLabel(const string plot_label)
  {
   CArrayObj *list=CSelect::ByBufferProperty(this.GetList(),BUFFER_PROP_LABEL,plot_label,EQUAL);
   return(list!=NULL && list.Total()>0 ? list.At(list.Total()-1) : NULL);
  }
//+------------------------------------------------------------------+
```

Get the list of buffer objects, in which the "Graphical series name" corresponds to the values passed to the method, and return the pointer to the last object from the list. If the list is empty, return NULL. If there are several buffer objects with the same graphical series name, the method returns the very last buffer object created with such a name.

The **GetBufferByTimeframe()** method returns the pointer to the buffer object by the timeframe set to it:

```
//+------------------------------------------------------------------+
//| Return the buffer by timeframe                                   |
//+------------------------------------------------------------------+
CBuffer *CBuffersCollection::GetBufferByTimeframe(const ENUM_TIMEFRAMES timeframe)
  {
   CArrayObj *list=CSelect::ByBufferProperty(this.GetList(),BUFFER_PROP_TIMEFRAME,timeframe,EQUAL);
   return(list!=NULL && list.Total()>0 ? list.At(list.Total()-1) : NULL);
  }
//+------------------------------------------------------------------+
```

Get the list of buffer objects, in which the "Timeframe" corresponds to the values passed to the method, and return the pointer to the last object from the list. If the list is empty, return NULL. If there are several buffer objects with the same timeframe, the method returns the very last buffer object created with the same chart period.

The **GetBufferByListIndex()** method returns the pointer to the buffer object by its index in the collection list:

```
//+------------------------------------------------------------------+
//| Return the buffer by the collection list index                   |
//+------------------------------------------------------------------+
CBuffer *CBuffersCollection::GetBufferByListIndex(const int index_list)
  {
   return this.m_list.At(index_list);
  }
//+------------------------------------------------------------------+
```

Simply return the pointer to the very last object in the buffer collection list. If the list is empty, [the At() method](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjat) returns NULL.

The **GetBufferCalculate()** method returns the pointer to the buffer object by its serial number (in the order of creating calculated buffers):

```
//+------------------------------------------------------------------+
//|Return the calculated buffer by serial number                     |
//| (0 - the very first candle buffer, 1,2,N - subsequent ones)      |
//+------------------------------------------------------------------+
CBufferCalculate *CBuffersCollection::GetBufferCalculate(const int number)
  {
   CArrayObj *list=CSelect::ByBufferProperty(this.GetList(),BUFFER_PROP_TYPE,BUFFER_TYPE_CALCULATE,EQUAL);
   return(list!=NULL && list.Total()>0 ? list.At(number) : NULL);
  }
//+------------------------------------------------------------------+
```

From the buffer object collection list, get only the buffers of the "Calculated buffer" type and return the pointer to the object by the index passed to the method from the obtained list. If the resulting sorted list is empty or the index of the necessary object passed to the method exceeds the obtained list, the [At() method](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjat) returns NULL.

The **InitializePlots()** overloaded methods are used to initialize all drawn buffers in the collection:

```
//+------------------------------------------------------------------+
//| Initialize all drawn buffers by a specified empty value          |
//+------------------------------------------------------------------+
void CBuffersCollection::InitializePlots(const double value,const uchar color_index)
  {
   CArrayObj *list=CSelect::ByBufferProperty(this.GetList(),BUFFER_PROP_TYPE,BUFFER_TYPE_DATA,EQUAL);
   if(list==NULL || list.Total()==0)
      return;
   int total=list.Total();
   for(int i=0;i<total;i++)
     {
      CBuffer *buff=list.At(i);
      if(buff==NULL)
         continue;
      buff.InitializeAll(value,color_index);
     }
  }
//+------------------------------------------------------------------+
```

The method initializes all drawn collection buffers using the buffer and color index values passed in the parameters.

Only drawn buffers are selected for the list first. Then we get the next buffer object by the obtained list in a loop and initialize all its arrays using the InitializeAll() buffer object class method passing the method inputs to it.

```
//+------------------------------------------------------------------+
//| Initialize all drawn buffers using                               |
//| the empty value set for the buffer object                        |
//+------------------------------------------------------------------+
void CBuffersCollection::InitializePlots(void)
  {
   CArrayObj *list=CSelect::ByBufferProperty(this.GetList(),BUFFER_PROP_TYPE,BUFFER_TYPE_DATA,EQUAL);
   if(list==NULL || list.Total()==0)
      return;
   int total=list.Total();
   for(int i=0;i<total;i++)
     {
      CBuffer *buff=list.At(i);
      if(buff==NULL)
         continue;
      buff.InitializeAll();
     }
  }
//+------------------------------------------------------------------+
```

The method initializes all drawn collection buffers using the buffer value set for each buffer object and color index value equal to zero.

Only drawn buffers are selected for the list first. Then we get the next buffer object by the obtained list in a loop and initialize all its arrays using the InitializeAll() buffer object class method with no parameters.

The **InitializeCalculates()** overloaded methods are used to initialize all calculated buffers in the collection:

```
//+------------------------------------------------------------------+
//| Initialize all calculated buffers by a specified empty value     |
//+------------------------------------------------------------------+
void CBuffersCollection::InitializeCalculates(const double value)
  {
   CArrayObj *list=CSelect::ByBufferProperty(this.GetList(),BUFFER_PROP_TYPE,BUFFER_TYPE_CALCULATE,EQUAL);
   if(list==NULL || list.Total()==0)
      return;
   int total=list.Total();
   for(int i=0;i<total;i++)
     {
      CBuffer *buff=list.At(i);
      if(buff==NULL)
         continue;
      buff.InitializeAll(value,0);
     }
  }
//+------------------------------------------------------------------+
```

The method initializes all calculated collection buffers using the buffer value passed in the parameters.

Only calculated buffers are selected for the list first. Then we get the next buffer object by the obtained list in a loop and initialize all its array using the InitializeAll() buffer object class method passing the method input to it.

```
//+------------------------------------------------------------------+
//| Initialize all calculated buffers using                          |
//| the empty value set for the buffer object                        |
//+------------------------------------------------------------------+
void CBuffersCollection::InitializeCalculates(void)
  {
   CArrayObj *list=CSelect::ByBufferProperty(this.GetList(),BUFFER_PROP_TYPE,BUFFER_TYPE_CALCULATE,EQUAL);
   if(list==NULL || list.Total()==0)
      return;
   int total=list.Total();
   for(int i=0;i<total;i++)
     {
      CBuffer *buff=list.At(i);
      if(buff==NULL)
         continue;
      buff.InitializeAll();
     }
  }
//+------------------------------------------------------------------+
```

The method initializes all calculated collection buffers using the buffer value passed for it.

Only calculated buffers are selected for the list first. Then we get the next buffer object by the obtained list in a loop and initialize its array using the InitializeAll() buffer object class method with no parameters.

The **SetColors()** method sets color values from the color array for all drawn collection buffers.

```
//+------------------------------------------------------------------+
//| Set color values from the passed color array                     |
//| for all collection indicator buffers                             |
//+------------------------------------------------------------------+
void CBuffersCollection::SetColors(const color &array_colors[])
  {
   CArrayObj *list=CSelect::ByBufferProperty(this.GetList(),BUFFER_PROP_TYPE,BUFFER_TYPE_DATA,EQUAL);
   if(list==NULL || list.Total()==0)
      return;
   int total=list.Total();
   for(int i=0;i<total;i++)
     {
      CBuffer *buff=list.At(i);
      if(buff==NULL)
         continue;
      buff.SetColors(array_colors);
     }
  }
//+------------------------------------------------------------------+
```

The method sets color values passed in the array parameters for all drawn collection buffers.

Only drawn buffers are selected for the list first. Then we get the next buffer object by the obtained list in a loop and specify the set of colors for it using the SetColors() buffer object class method passing the array sent to the method inputs.

The **Clear()** method clears the buffer data by its index in the collection list in the specified timeseries bar:

```
//+------------------------------------------------------------------+
//| Clear buffer data by its index in the list                       |
//| in the specified timeseries bar                                  |
//+------------------------------------------------------------------+
void CBuffersCollection::Clear(const int buffer_list_index,const int series_index)
  {
   CBuffer *buff=this.GetBufferByListIndex(buffer_list_index);
   if(buff==NULL)
      return;
   buff.ClearData(series_index);
  }
//+------------------------------------------------------------------+
```

First, get the pointer to the buffer object by its index in the list using the GetBufferByListIndex() method considered above. Then clear all its arrays using the method of the obtained ClearData() buffer object.

To be able to calculate the necessary bar index data when drawing indicator buffer lines on the current chart using data from other timeframes, we need to pass the pointer to the collection list of all used timeseries located [in the timeseries collection class](https://www.mql5.com/en/articles/7695) to the collection class.

The easiest way to do this is during program initialization, when all objects of the library classes have already been created.

The OnInit() class method assigns the passed pointer to the timeseries collection class to the **m\_timeseries** variable:

```
void OnInit(CTimeSeriesCollection *timeseries) { this.m_timeseries=timeseries; }
```

Further on, I will call the method in the library initialization functions within programs. I already have all the necessary functionality for that. Below, I will simply add calling the method in the required CEngine class method.

The class already features the methods for setting the values for indicator buffers by buffer type (arrows, line, etc.), as well as for setting the color index for these buffers and clearing all buffer arrays. I have added 28 methods in total. There is no point in describing each one here — all methods are presented in the files attached below. They are identical to each other in terms of logic. The only difference is the number of arrays buffers of different types have. Therefore, we will consider only the arrow buffer methods here.

**The method setting the value by the arrow buffer timeseries index:**

```
//+------------------------------------------------------------------+
//| Set the value to the arrow buffer by the timeseries index        |
//+------------------------------------------------------------------+
void CBuffersCollection::SetBufferArrowValue(const int number,const int series_index,const double value,const uchar color_index,bool as_current=false)
  {
//--- Get the arrow buffer object
   CBufferArrow *buff=this.GetBufferArrow(number);
   if(buff==NULL)
      return;
//--- If the buffer usage flag is set only as for the current timeframe,
//--- write the passed value to the current buffer bar and exit (the color is not used)
   if(as_current)
     {
      buff.SetBufferValue(0,series_index,value);
      return;
     }
//--- Get data on the necessary timeseries and bars, and calculate the amount of bars of the current timeframe included into one bar of the buffer object chart period
   int index_bar_period=series_index;
   int num_bars=this.GetBarsData(buff,series_index,index_bar_period);
   if(num_bars==WRONG_VALUE)
      return;
//--- Calculate the index of the next bar for the current chart in the loop by the number of bars and
//--- set the value and color passed to the method by the calculated index
   for(int i=0;i<num_bars;i++)
     {
      int index=index_bar_period-i;
      if(index<0)
         break;
      buff.SetBufferValue(0,index,value);
      buff.SetBufferColorIndex(index,color_index);
     }
  }
//+------------------------------------------------------------------+
```

The method logic is described in the code comments. However, let me clarify some nuances.

The method receives the number of the arrow buffer object. The number means the serial number consisting of all created arrow buffers. The very first arrow buffer we have created in the custom program has the number of 0, the second one is 1, the third one is 2, etc. Do not confuse the numbers of created buffer objects with buffer object indices in the collection list. Indices of newly added buffers always go in increasing order and do not depend on a buffer object type. The numbers of buffer objects depend on the buffer type. I have described the concept in detail at the very beginning of the article.

The method receives the timeseries bar index accepting the value which is also passed by the next parameter, index of the color used to paint the indicator buffer line by the specified timeseries index, as well as the additional parameter "as for the current timeseries".

The methods are arranged so that if the buffer object has the "timeframe" property value different from the current chart period, the indicator line is drawn on the current chart taking into account the buffer object timeframe data, regardless of the timeseries index passed to the method. I.e. the method correctly displays data from other chart periods on the current one provided that the **as\_current** flag has the value of false (by default). If the flag is set to true, the method works only with the current timeframe data regardless of whether the buffer object has another specified chart period in its properties.

**The method setting the color index** **for the arrow buffer object** **by its serial number into the color buffer array** **:**

```
//+------------------------------------------------------------------+
//| Set the color index to the color buffer                          |
//| of the arrow buffer by the timeseries index                      |
//+------------------------------------------------------------------+
void CBuffersCollection::SetBufferArrowColorIndex(const int number,const int series_index,const uchar color_index)
  {
   CBufferArrow *buff=this.GetBufferArrow(number);
   if(buff==NULL)
      return;
   buff.SetBufferColorIndex(series_index,color_index);
  }
//+------------------------------------------------------------------+
```

Here all is simple: get the arrow buffer object by its index and set the specified color index to the passed timeseries index.

**The method clearing the data by the arrow buffer timeseries index** **by its index:**

```
//+------------------------------------------------------------------+
//| Clear the arrow buffer data by the timeseries index              |
//+------------------------------------------------------------------+
void CBuffersCollection::ClearBufferArrow(const int number,const int series_index)
  {
   CBufferArrow *buff=this.GetBufferArrow(number);
   if(buff==NULL)
      return;
   buff.SetBufferValue(0,series_index,buff.EmptyValue());
  }
//+------------------------------------------------------------------+
```

Here we get the arrow buffer object by its index and set the "empty" value specified for the buffer object to the passed timeseries index.

All the remaining methods for working with other buffer object types do not fundamentally differ from the ones discussed above except for the number of arrays and buffer object type. You can examine them on your own.

**This concludes the improvements of the buffer object collection class**.

Find the full listing with all the changes in the files attached below.

Now let's complement and improve the CEngine library main object class in \\MQL5\\Include\\DoEasy\ **Engine.mqh**. I will proceed in the same way as when describing the improvements of the buffer collection class: first, we will have a look at the additions and changes in the class body and then I will analyze each of the methods.

There is no point in displaying the full listing here, since it is quite voluminous. Instead, I will display extracts with implemented changes and additions:

```
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine
  {
private:
//--- The code has been removed for the sake of space
//--- ...
public:
//--- The code has been removed for the sake of space
//--- ...
//--- Return the bar type of the specified timeframe's symbol by (1) index and (2) time
   ENUM_BAR_BODY_TYPE   SeriesBarType(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index);
   ENUM_BAR_BODY_TYPE   SeriesBarType(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time);

//--- Copy the specified double property of the specified timeseries of the specified symbol to the array
//--- Regardless of the array indexing direction, copying is performed the same way as copying to a timeseries array
   bool                 SeriesCopyToBufferAsSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,const ENUM_BAR_PROP_DOUBLE property,
                                                   double &array[],const double empty=EMPTY_VALUE)
                          { return this.m_time_series.CopyToBufferAsSeries(symbol,timeframe,property,array,empty);}

//--- Return (1) the buffer collection and (2) the buffer list from the collection
   CBuffersCollection  *GetBuffersCollection(void)                                     { return &this.m_buffers;                             }
   CArrayObj           *GetListBuffers(void)                                           { return this.m_buffers.GetList();                    }
//--- Return the buffer by (1) the graphical series name, (2) timeframe, (3) Plot index, (4) collection list and (5) the last one in the list
   CBuffer             *GetBufferByLabel(const string plot_label)                      { return this.m_buffers.GetBufferByLabel(plot_label); }
   CBuffer             *GetBufferByTimeframe(const ENUM_TIMEFRAMES timeframe)          { return this.m_buffers.GetBufferByTimeframe(timeframe);}
   CBuffer             *GetBufferByPlot(const int plot_index)                          { return this.m_buffers.GetBufferByPlot(plot_index);  }
   CBuffer             *GetBufferByListIndex(const int index_list)                     { return this.m_buffers.GetBufferByListIndex(index_list);}
   CBuffer             *GetLastBuffer(void);
//--- Return buffers by drawing style by a serial number
//--- (0 - the very first created buffer with the ХХХ drawing style, 1,2,N - subsequent ones)
   CBufferArrow        *GetBufferArrow(const int number)                               { return this.m_buffers.GetBufferArrow(number);       }
   CBufferLine         *GetBufferLine(const int number)                                { return this.m_buffers.GetBufferLine(number);        }
   CBufferSection      *GetBufferSection(const int number)                             { return this.m_buffers.GetBufferSection(number);     }
   CBufferHistogram    *GetBufferHistogram(const int number)                           { return this.m_buffers.GetBufferHistogram(number);   }
   CBufferHistogram2   *GetBufferHistogram2(const int number)                          { return this.m_buffers.GetBufferHistogram2(number);  }
   CBufferZigZag       *GetBufferZigZag(const int number)                              { return this.m_buffers.GetBufferZigZag(number);      }
   CBufferFilling      *GetBufferFilling(const int number)                             { return this.m_buffers.GetBufferFilling(number);     }
   CBufferBars         *GetBufferBars(const int number)                                { return this.m_buffers.GetBufferBars(number);        }
   CBufferCandles      *GetBufferCandles(const int number)                             { return this.m_buffers.GetBufferCandles(number);     }
   CBufferCalculate    *GetBufferCalculate(const int number)                           { return this.m_buffers.GetBufferCalculate(number);   }

//--- Return the number of (1) drawn buffers and (2) all indicator arrays
   int                  BuffersPropertyPlotsTotal(void)                                { return this.m_buffers.PropertyPlotsTotal();         }
   int                  BuffersPropertyBuffersTotal(void)                              { return this.m_buffers.PropertyBuffersTotal();       }

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

//--- Initialize all drawn buffers by a (1) specified value, (2) empty value set for the buffer object
   void                 BuffersInitPlots(const double value,const uchar color_index)   { this.m_buffers.InitializePlots(value,color_index);  }
   void                 BuffersInitPlots(void)                                         { this.m_buffers.InitializePlots();                   }
//--- Initialize all calculated buffers by a (1) specified value, (2) empty value set for the buffer object
   void                 BuffersInitCalculates(const double value)                      { this.m_buffers.InitializeCalculates(value);         }
   void                 BuffersInitCalculates(void)                                    { this.m_buffers.InitializeCalculates();              }

//--- Return buffer data by its serial number of (1) arrows, (2) line, (3) sections and (4) histogram from zero
//--- (0 - the very first created buffer with the ХХХ drawing style, 1,2,N - subsequent ones)
   double               BufferDataArrow(const int number,const int series_index);
   double               BufferDataLine(const int number,const int series_index);
   double               BufferDataSection(const int number,const int series_index);
   double               BufferDataHistogram(const int number,const int series_index);
//--- Return buffer data by its serial number of (1) the zero and (2) the first histogram buffer on two buffers
//--- (0 - the very first created buffer with the ХХХ drawing style, 1,2,N - subsequent ones)
   double               BufferDataHistogram20(const int number,const int series_index);
   double               BufferDataHistogram21(const int number,const int series_index);
//--- Return buffer data by its serial number of (1) the zero and (2) the first zigzag buffer
//--- (0 - the very first created buffer with the ХХХ drawing style, 1,2,N - subsequent ones)
   double               BufferDataZigZag0(const int number,const int series_index);
   double               BufferDataZigZag1(const int number,const int series_index);
//--- Return buffer data by its serial number of (1) the zero and (2) the first filling buffer
//--- (0 - the very first created buffer with the ХХХ drawing style, 1,2,N - subsequent ones)
   double               BufferDataFilling0(const int number,const int series_index);
   double               BufferDataFilling1(const int number,const int series_index);
//--- Return buffer data by its serial number of (1) Open, (2) High, (3) Low and (4) Close bar buffers
//--- (0 - the very first created buffer with the ХХХ drawing style, 1,2,N - subsequent ones)
   double               BufferDataBarsOpen(const int number,const int series_index);
   double               BufferDataBarsHigh(const int number,const int series_index);
   double               BufferDataBarsLow(const int number,const int series_index);
   double               BufferDataBarsClose(const int number,const int series_index);
//--- Return buffer data by its serial number of (1) Open, (2) High, (3) Low and (4) Close candle buffers
//--- (0 - the very first created buffer with the ХХХ drawing style, 1,2,N - subsequent ones)
   double               BufferDataCandlesOpen(const int number,const int series_index);
   double               BufferDataCandlesHigh(const int number,const int series_index);
   double               BufferDataCandlesLow(const int number,const int series_index);
   double               BufferDataCandlesClose(const int number,const int series_index);

//--- Set buffer data by its serial number of (1) arrows, (2) line, (3) sections, (4) histogram from zero and the (5) calculated buffer
//--- (0 - the very first created buffer with the ХХХ drawing style, 1,2,N - subsequent ones)
   void                 BufferSetDataArrow(const int number,const int series_index,const double value,const uchar color_index,bool as_current=false);
   void                 BufferSetDataLine(const int number,const int series_index,const double value,const uchar color_index,bool as_current=false);
   void                 BufferSetDataSection(const int number,const int series_index,const double value,const uchar color_index,bool as_current=false);
   void                 BufferSetDataHistogram(const int number,const int series_index,const double value,const uchar color_index,bool as_current=false);
   void                 BufferSetDataCalculate(const int number,const int series_index,const double value);
//--- Set data of the (1) zero, (2) first and (3) all histogram buffers on two buffers by a serial number of a created buffer
//--- (0 - the very first created buffer with the HISTOGRAM2 drawing style, 1,2,N - subsequent ones)
   void                 BufferSetDataHistogram20(const int number,const int series_index,const double value);
   void                 BufferSetDataHistogram21(const int number,const int series_index,const double value);
   void                 BufferSetDataHistogram2(const int number,const int series_index,const double value0,const double value1,const uchar color_index,bool as_current=false);
//--- Set data of the (1) zero, (2) first and (3) all zigzag buffers by a serial number of a created buffer
//--- (0 - the very first created buffer with the ZIGZAG drawing style, 1,2,N - subsequent ones)
   void                 BufferSetDataZigZag0(const int number,const int series_index,const double value);
   void                 BufferSetDataZigZag1(const int number,const int series_index,const double value);
   void                 BufferSetDataZigZag(const int number,const int series_index,const double value0,const double value1,const uchar color_index,bool as_current=false);
//--- Set data of the (1) zero, (2) first and (3) all filling buffers by a serial number of a created buffer
//--- (0 - the very first created buffer with the FILLING drawing style, 1,2,N - subsequent ones)
   void                 BufferSetDataFilling0(const int number,const int series_index,const double value);
   void                 BufferSetDataFilling1(const int number,const int series_index,const double value);
   void                 BufferSetDataFilling(const int number,const int series_index,const double value0,const double value1,bool as_current=false);
//--- Set data of the (1) Open, (2) High, (3) Low, (4) Close and (5) all bar buffers by a serial number of a created buffer
//--- (0 - the very first created buffer with the BARS drawing style, 1,2,N - subsequent ones)
   void                 BufferSetDataBarsOpen(const int number,const int series_index,const double value);
   void                 BufferSetDataBarsHigh(const int number,const int series_index,const double value);
   void                 BufferSetDataBarsLow(const int number,const int series_index,const double value);
   void                 BufferSetDataBarsClose(const int number,const int series_index,const double value);
   void                 BufferSetDataBars(const int number,const int series_index,const double open,const double high,const double low,const double close,const uchar color_index,bool as_current=false);
//--- Set data of the (1) Open, (2) High, (3) Low, (4) Close and (5) all candle buffers by a serial number of a created buffer
//--- (0 - the very first created buffer with the CANDLES drawing style, 1,2,N - subsequent ones)
   void                 BufferSetDataCandlesOpen(const int number,const int series_index,const double value);
   void                 BufferSetDataCandlesHigh(const int number,const int series_index,const double value);
   void                 BufferSetDataCandlesLow(const int number,const int series_index,const double value);
   void                 BufferSetDataCandlesClose(const int number,const int series_index,const double value);
   void                 BufferSetDataCandles(const int number,const int series_index,const double open,const double high,const double low,const double close,const uchar color_index,bool as_current=false);

//--- Return buffer color by its serial number of (1) arrows, (2) line, (3) sections, (4) histogram from zero
//--- (5) histogram on two buffers, (6) zigzag, (7) filling, (8) bars and (9) candles
//--- (0 - the very first created buffer with the ХХХ drawing style, 1,2,N - subsequent ones)
   color                BufferArrowColor(const int number,const int series_index);
   color                BufferLineColor(const int number,const int series_index);
   color                BufferSectionColor(const int number,const int series_index);
   color                BufferHistogramColor(const int number,const int series_index);
   color                BufferHistogram2Color(const int number,const int series_index);
   color                BufferZigZagColor(const int number,const int series_index);
   color                BufferFillingColor(const int number,const int series_index);
   color                BufferBarsColor(const int number,const int series_index);
   color                BufferCandlesColor(const int number,const int series_index);

//--- Return buffer color index by its serial number of (1) arrows, (2) line, (3) sections, (4) histogram from zero
//--- (5) histogram on two buffers, (6) zigzag, (7) filling, (8) bars and (9) candles
//--- (0 - the very first created buffer with the ХХХ drawing style, 1,2,N - subsequent ones)
   int                  BufferArrowColorIndex(const int number,const int series_index);
   int                  BufferLineColorIndex(const int number,const int series_index);
   int                  BufferSectionColorIndex(const int number,const int series_index);
   int                  BufferHistogramColorIndex(const int number,const int series_index);
   int                  BufferHistogram2ColorIndex(const int number,const int series_index);
   int                  BufferZigZagColorIndex(const int number,const int series_index);
   int                  BufferFillingColorIndex(const int number,const int series_index);
   int                  BufferBarsColorIndex(const int number,const int series_index);
   int                  BufferCandlesColorIndex(const int number,const int series_index);

//--- Set color values from the passed color array for all indicator buffers within the collection
   void                 BuffersSetColors(const color &array_colors[])                     { this.m_buffers.SetColors(array_colors);                   }
//--- Set the color index to the color buffer by its serial number of (1) arrows, (2) line, (3) sections, (4) histogram from zero
//--- (5) histogram on two buffers, (6) zigzag, (7) filling, (8) bars and (9) candles
//--- (0 - the very first created buffer with the ХХХ drawing style, 1,2,N - subsequent ones)
   void                 BufferArrowSetColorIndex(const int number,const int series_index,const uchar color_index)
                          { this.m_buffers.SetBufferArrowColorIndex(number,series_index,color_index);       }
   void                 BufferLineSetColorIndex(const int number,const int series_index,const uchar color_index)
                          { this.m_buffers.SetBufferLineColorIndex(number,series_index,color_index);        }
   void                 BufferSectionSetColorIndex(const int number,const int series_index,const uchar color_index)
                          { this.m_buffers.SetBufferSectionColorIndex(number,series_index,color_index);     }
   void                 BufferHistogramSetColorIndex(const int number,const int series_index,const uchar color_index)
                          { this.m_buffers.SetBufferHistogramColorIndex(number,series_index,color_index);   }
   void                 BufferHistogram2SetColorIndex(const int number,const int series_index,const uchar color_index)
                          { this.m_buffers.SetBufferHistogram2ColorIndex(number,series_index,color_index);  }
   void                 BufferZigZagSetColorIndex(const int number,const int series_index,const uchar color_index)
                          { this.m_buffers.SetBufferZigZagColorIndex(number,series_index,color_index);      }
   void                 BufferFillingSetColorIndex(const int number,const int series_index,const uchar color_index)
                          { this.m_buffers.SetBufferFillingColorIndex(number,series_index,color_index);     }
   void                 BufferBarsSetColorIndex(const int number,const int series_index,const uchar color_index)
                          { this.m_buffers.SetBufferBarsColorIndex(number,series_index,color_index);        }
   void                 BufferCandlesSetColorIndex(const int number,const int series_index,const uchar color_index)
                          { this.m_buffers.SetBufferCandlesColorIndex(number,series_index,color_index);     }

//--- Clear buffer data by its index in the list in the specified timeseries bar
   void                 BufferClear(const int buffer_list_index,const int series_index)   { this.m_buffers.Clear(buffer_list_index,series_index);     }
//--- Clear data by the timeseries index for the (1) arrow, (2) line, (3) section, (4) zero line histogram,
//--- (5) histogram on two buffers, (6) zigzag, (7) filling, (8) bars and (9) candles
   void                 BufferArrowClear(const int number,const int series_index)         { this.m_buffers.ClearBufferArrow(number,series_index);     }
   void                 BufferLineClear(const int number,const int series_index)          { this.m_buffers.ClearBufferLine(number,series_index);      }
   void                 BufferSectionClear(const int number,const int series_index)       { this.m_buffers.ClearBufferSection(number,series_index);   }
   void                 BufferHistogramClear(const int number,const int series_index)     { this.m_buffers.ClearBufferHistogram(number,series_index); }
   void                 BufferHistogram2Clear(const int number,const int series_index)    { this.m_buffers.ClearBufferHistogram2(number,series_index);}
   void                 BufferZigZagClear(const int number,const int series_index)        { this.m_buffers.ClearBufferZigZag(number,series_index);    }
   void                 BufferFillingClear(const int number,const int series_index)       { this.m_buffers.ClearBufferFilling(number,series_index);   }
   void                 BufferBarsClear(const int number,const int series_index)          { this.m_buffers.ClearBufferBars(number,series_index);      }
   void                 BufferCandlesClear(const int number,const int series_index)       { this.m_buffers.ClearBufferCandles(number,series_index);   }

//--- Display short description of all indicator buffers of the buffer collection
   void                 BuffersPrintShort(void);

//--- Set the following for the trading classes:
//--- (1) correct filling policy, (2) filling policy,
//--- (3) correct order expiration type, (4) order expiration type,
//--- (5) magic number, (6) comment, (7) slippage, (8) volume, (9) order expiration date,
//--- (10) the flag of asynchronous sending of a trading request, (11) logging level, (12) number of trading attempts
   void                 TradingSetCorrectTypeFilling(const ENUM_ORDER_TYPE_FILLING type=ORDER_FILLING_FOK,const string symbol_name=NULL);
   void                 TradingSetTypeFilling(const ENUM_ORDER_TYPE_FILLING type=ORDER_FILLING_FOK,const string symbol_name=NULL);
   void                 TradingSetCorrectTypeExpiration(const ENUM_ORDER_TYPE_TIME type=ORDER_TIME_GTC,const string symbol_name=NULL);
   void                 TradingSetTypeExpiration(const ENUM_ORDER_TYPE_TIME type=ORDER_TIME_GTC,const string symbol_name=NULL);
   void                 TradingSetMagic(const uint magic,const string symbol_name=NULL);
   void                 TradingSetComment(const string comment,const string symbol_name=NULL);
   void                 TradingSetDeviation(const ulong deviation,const string symbol_name=NULL);
   void                 TradingSetVolume(const double volume=0,const string symbol_name=NULL);
   void                 TradingSetExpiration(const datetime expiration=0,const string symbol_name=NULL);
   void                 TradingSetAsyncMode(const bool async_mode=false,const string symbol_name=NULL);
   void                 TradingSetLogLevel(const ENUM_LOG_LEVEL log_level=LOG_LEVEL_ERROR_MSG,const string symbol_name=NULL);
   void                 TradingSetTotalTry(const uchar attempts)                       { this.m_trading.SetTotalTry(attempts);                     }

//--- Return the logging level of a trading class symbol trading object
   ENUM_LOG_LEVEL       TradingGetLogLevel(const string symbol_name)                   { return this.m_trading.GetTradeObjLogLevel(symbol_name);   }

//--- Set standard sounds (symbol==NULL) for a symbol trading object, (symbol!=NULL) for trading objects of all symbols
   void                 SetSoundsStandart(const string symbol=NULL)
                          {
                           this.m_trading.SetSoundsStandart(symbol);
                          }
//--- Set the flag of using sounds
   void                 SetUseSounds(const bool flag) { this.m_trading.SetUseSounds(flag);   }
//--- Set a sound for a specified order/position type and symbol. 'mode' specifies an event a sound is set for
//--- (symbol=NULL) for trading objects of all symbols, (symbol!=NULL) for a trading object of a specified symbol
   void                 SetSound(const ENUM_MODE_SET_SOUND mode,const ENUM_ORDER_TYPE action,const string sound,const string symbol=NULL)
                          {
                           this.m_trading.SetSound(mode,action,sound,symbol);
                          }
//--- Play a sound by its description
   bool                 PlaySoundByDescription(const string sound_description);

//--- Pass the pointers to all the necessary collections to the trading class and the indicator buffer collection class
   void                 CollectionOnInit(void)
                          {
                           this.m_trading.OnInit(this.GetAccountCurrent(),m_symbols.GetObject(),m_market.GetObject(),m_history.GetObject(),m_events.GetObject());
                           this.m_buffers.OnInit(this.m_time_series.GetObject());
                          }
```

The **SeriesBarType()** overloaded method returns the type of the specified bar on the specified symbol and chart period **by the specified index of the appropriate timeseries:**

```
//+------------------------------------------------------------------+
//| Return the bar type of the specified symbol                      |
//| of the specified timeframe by index                              |
//+------------------------------------------------------------------+
ENUM_BAR_BODY_TYPE CEngine::SeriesBarType(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index)
  {
   CBar *bar=this.m_time_series.GetBar(symbol,timeframe,index);
   return(bar!=NULL ? bar.TypeBody() : (ENUM_BAR_BODY_TYPE)WRONG_VALUE);
  }
//+------------------------------------------------------------------+
```

Get the required bar object from the timeseries collection by the specified timeseries index and return its type (bullish/bearish/zero/with a zero body). If failed to get a bar, -1 is returned.

**by time:**

```
//+------------------------------------------------------------------+
//| Return the bar type of the specified symbol                      |
//| of the specified timeframe by time                               |
//+------------------------------------------------------------------+
ENUM_BAR_BODY_TYPE CEngine::SeriesBarType(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time)
  {
   CBar *bar=this.m_time_series.GetBar(symbol,timeframe,time);
   return(bar!=NULL ? bar.TypeBody() : (ENUM_BAR_BODY_TYPE)WRONG_VALUE);
  }
//+------------------------------------------------------------------+
```

Get the required bar object from the timeseries collection by the specified time and return its type (bullish/bearish/zero/with a zero body).

If failed to get a bar, -1 is returned.

It is convenient to use the methods in indicators to determine a color of the required bar (or the required bars when drawing data from other timeseries).

The **GetBufferByLabel()** method returns the pointer to the buffer object by the name of its graphical series:

```
CBuffer *GetBufferByLabel(const string plot_label) { return this.m_buffers.GetBufferByLabel(plot_label); }
```

Return the result of the buffer collection class method of the same name discussed above.

The **GetBufferByTimeframe()** method returns the pointer to the buffer object by its "timeframe" property:

```
CBuffer *GetBufferByTimeframe(const ENUM_TIMEFRAMES timeframe) { return this.m_buffers.GetBufferByTimeframe(timeframe);}
```

Return the result of the buffer collection class method of the same name discussed above.

The **GetBufferByListIndex()** method returns the pointer to the buffer object by its index in the buffer object collection list of the buffer collection class:

```
CBuffer *GetBufferByListIndex(const int index_list) { return this.m_buffers.GetBufferByListIndex(index_list);}
```

Return the result of the buffer collection class method of the same name discussed above.

The **GetLastBuffer()** method returns the pointer to the buffer object created last:

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

Get the pointer to the buffer collection list and return the last buffer object from the list.

The **GetBufferCalculate()** method returns the result of the buffer collection class method of the same name discussed above:

```
CBufferCalculate *GetBufferCalculate(const int number) { return this.m_buffers.GetBufferCalculate(number); }
```

The **BuffersPropertyPlotsTotal()** and **BuffersPropertyBuffersTotal()** methods have been renamed from the BufferPlotsTotal() and BuffersTotal() ones accordingly. They return the results of the PropertyPlotsTotal() and PropertyBuffersTotal() methods of the buffer collection class I have considered above:

```
int BuffersPropertyPlotsTotal(void)   { return this.m_buffers.PropertyPlotsTotal();   }
int BuffersPropertyBuffersTotal(void) { return this.m_buffers.PropertyBuffersTotal(); }
```

The **BufferCreateCalculate()** method returns the result of the CreateCalculate() method of the buffer collection class I have considered above:

```
bool BufferCreateCalculate(void) { return this.m_buffers.CreateCalculate(); }
```

Currently, there is one limitation for using the method for creating a calculated buffer: all calculated buffers should be created when all the necessary drawn buffers are already present. If we create a calculated buffer between several drawn ones, this violates the sequence of setting arrays for indicator buffers. As a result, all drawn indicator buffers created after the calculated buffer are not displayed. I have not found the reason for this behavior yet, but I firmly intend to fix this later.

The overloaded **BuffersInitPlots()** method initializes all drawn buffers using the InitializePlots() method of the buffer collection class I have considered above:

```
void BuffersInitPlots(const double value,const uchar color_index) { this.m_buffers.InitializePlots(value,color_index); }
void BuffersInitPlots(void)                                       { this.m_buffers.InitializePlots();                  }
```

The overloaded **BuffersInitCalculates()** method initializes all calculated buffers using the InitializeCalculates() method of the buffer collection class I have considered above:

```
void BuffersInitCalculates(const double value) { this.m_buffers.InitializeCalculates(value); }
void BuffersInitCalculates(void)               { this.m_buffers.InitializeCalculates();      }
```

The **BufferSetDataCalculate()** method sets data for the calculated buffer using the SetBufferCalculateValue() method of the buffer object collection class I have considered above:

```
//+------------------------------------------------------------------+
//| Set the calculated buffer data by its serial number              |
//| (0 - the very first buffer, 1,2,N - subsequent ones)             |
//+------------------------------------------------------------------+
void CEngine::BufferSetDataCalculate(const int number,const int series_index,const double value)
  {
   this.m_buffers.SetBufferCalculateValue(number,series_index,value);
  }
//+------------------------------------------------------------------+
```

The **BufferSetDataArrow()**, **BufferSetDataLine()**, **BufferSetDataSection()**, **BufferSetDataHistogram()**, **BufferSetDataHistogram2()**, **BufferSetDataZigZag()**, **BufferSetDataFilling()**, **BufferSetDataBars()** and **BufferSetDataCandles()** methods have been changed. Now they set the data of the appropriate buffers using the buffer object collection class methods considered above:

```
//+------------------------------------------------------------------+
//| Set arrow buffer data by its serial number                       |
//| (0 - the very first arrow buffer, 1,2,N - subsequent ones)       |
//+------------------------------------------------------------------+
void CEngine::BufferSetDataArrow(const int number,const int series_index,const double value,const uchar color_index,bool as_current=false)
  {
   this.m_buffers.SetBufferArrowValue(number,series_index,value,color_index,as_current);
  }
//+------------------------------------------------------------------+
//| Set line buffer data by its serial number                        |
//| (0 - the very first line buffer, 1,2,N - subsequent ones)        |
//+------------------------------------------------------------------+
void CEngine::BufferSetDataLine(const int number,const int series_index,const double value,const uchar color_index,bool as_current=false)
  {
   this.m_buffers.SetBufferLineValue(number,series_index,value,color_index,as_current);
  }
//+------------------------------------------------------------------+
//| Set section buffer data by its serial number                     |
//| (0 - the very first sections buffer, 1,2,N - subsequent ones)    |
//+------------------------------------------------------------------+
void CEngine::BufferSetDataSection(const int number,const int series_index,const double value,const uchar color_index,bool as_current=false)
  {
   this.m_buffers.SetBufferSectionValue(number,series_index,value,color_index,as_current);
  }
//+------------------------------------------------------------------+
//| Set histogram buffer data from zero                              |
//| by its serial number                                             |
//| (0 - the very first buffer, 1,2,N - subsequent ones)             |
//+------------------------------------------------------------------+
void CEngine::BufferSetDataHistogram(const int number,const int series_index,const double value,const uchar color_index,bool as_current=false)
  {
   this.m_buffers.SetBufferHistogramValue(number,series_index,value,color_index,as_current);
  }
//+------------------------------------------------------------------+
```

...

```
//+------------------------------------------------------------------+
//| Set data of all histogram buffers on two buffers                 |
//| by its serial number                                             |
//| (0 - the very first buffer, 1,2,N - subsequent ones)             |
//+------------------------------------------------------------------+
void CEngine::BufferSetDataHistogram2(const int number,const int series_index,const double value1,const double value2,const uchar color_index,bool as_current=false)
  {
   this.m_buffers.SetBufferHistogram2Value(number,series_index,value1,value2,color_index,as_current);
  }
//+------------------------------------------------------------------+
```

...

```
//+------------------------------------------------------------------+
//| Set data of all zizag buffers                                    |
//| by its serial number                                             |
//| (0 - the very first zigzag buffer, 1,2,N - subsequent ones)      |
//+------------------------------------------------------------------+
void CEngine::BufferSetDataZigZag(const int number,const int series_index,const double value1,const double value2,const uchar color_index,bool as_current=false)
  {
   this.m_buffers.SetBufferZigZagValue(number,series_index,value1,value2,color_index,as_current);
  }
//+------------------------------------------------------------------+
```

...

```
//+------------------------------------------------------------------+
//| Set data of all filling buffers                                  |
//| by its serial number                                             |
//| (0 - the very first filling buffer, 1,2,N - subsequent ones)     |
//+------------------------------------------------------------------+
void CEngine::BufferSetDataFilling(const int number,const int series_index,const double value1,const double value2,bool as_current=false)
  {
   this.m_buffers.SetBufferFillingValue(number,series_index,value1,value2,as_current);
  }
//+------------------------------------------------------------------+
```

...

```
//+------------------------------------------------------------------+
//| Set data of all bar buffers                                      |
//| by its serial number                                             |
//| (0 - the very first bar buffer, 1,2,N - subsequent ones)         |
//+------------------------------------------------------------------+
void CEngine::BufferSetDataBars(const int number,const int series_index,const double open,const double high,const double low,const double close,const uchar color_index,bool as_current=false)
  {
   CBufferBars *buff=this.m_buffers.GetBufferBars(number);
   if(buff==NULL)
      return;
   this.m_buffers.SetBufferBarsValue(number,series_index,open,high,low,close,color_index,as_current);
  }
//+------------------------------------------------------------------+
```

...

```
//+------------------------------------------------------------------+
//| Set data of all candle buffers                                   |
//| by its serial number                                             |
//| (0 - the very first candle buffer, 1,2,N - subsequent ones)      |
//+------------------------------------------------------------------+
void CEngine::BufferSetDataCandles(const int number,const int series_index,const double open,const double high,const double low,const double close,const uchar color_index,bool as_current=false)
  {
   CBufferCandles *buff=this.m_buffers.GetBufferCandles(number);
   if(buff==NULL)
      return;
   this.m_buffers.SetBufferCandlesValue(number,series_index,open,high,low,close,color_index,as_current);
  }
//+------------------------------------------------------------------+
```

All these methods are multi-period allowing us to display requested data from other periods on the current chart within one call.

The **BufferArrowColor()**, **BufferLineColor()**, **BufferSectionColor()**, **BufferHistogramColor()**, **BufferHistogram2Color()**, **BufferZigZagColor()**, **BufferFillingColor()**, **BufferBarsColor()**, **BufferCandlesColor()**, **BufferArrowColorIndex()**, **BufferLineColorIndex()**, **BufferSectionColorIndex()**, **BufferHistogramColorIndex()**, **BufferHistogram2ColorIndex()**, **BufferZigZagColorIndex()**, **BufferFillingColorIndex()**, **BufferBarsColorIndex()** and **BufferCandlesColorIndex()** methods have been renamed simply for better readability within a program.

For example, the BufferArrowColor() method has been previously called BufferColorArrow(), which could cause confusion when determining its purpose. Now all these methods describe the buffer type first, next they return what seems to be more visually comprehensive and correct.

The **BuffersSetColors()** method sets color values to all indicator buffers from the passed color array using the SetColors() method of the buffer object collection class I have considered above:

```
void BuffersSetColors(const color &array_colors[]) { this.m_buffers.SetColors(array_colors); }
```

**The methods of setting the color index to certain buffer objects** have been revised. Now they call the appropriate buffer object collection class methods considered above:

```
void BufferArrowSetColorIndex(const int number,const int series_index,const uchar color_index)
       { this.m_buffers.SetBufferArrowColorIndex(number,series_index,color_index);       }
void BufferLineSetColorIndex(const int number,const int series_index,const uchar color_index)
       { this.m_buffers.SetBufferLineColorIndex(number,series_index,color_index);        }
void BufferSectionSetColorIndex(const int number,const int series_index,const uchar color_index)
       { this.m_buffers.SetBufferSectionColorIndex(number,series_index,color_index);     }
void BufferHistogramSetColorIndex(const int number,const int series_index,const uchar color_index)
       { this.m_buffers.SetBufferHistogramColorIndex(number,series_index,color_index);   }
void BufferHistogram2SetColorIndex(const int number,const int series_index,const uchar color_index)
       { this.m_buffers.SetBufferHistogram2ColorIndex(number,series_index,color_index);  }
void BufferZigZagSetColorIndex(const int number,const int series_index,const uchar color_index)
       { this.m_buffers.SetBufferZigZagColorIndex(number,series_index,color_index);      }
void BufferFillingSetColorIndex(const int number,const int series_index,const uchar color_index)
       { this.m_buffers.SetBufferFillingColorIndex(number,series_index,color_index);     }
void BufferBarsSetColorIndex(const int number,const int series_index,const uchar color_index)
       { this.m_buffers.SetBufferBarsColorIndex(number,series_index,color_index);        }
void BufferCandlesSetColorIndex(const int number,const int series_index,const uchar color_index)
       { this.m_buffers.SetBufferCandlesColorIndex(number,series_index,color_index);     }
```

The **BufferClear()** method clears the buffer data by its index in the list in the specified timeseries bar using the **Clear()** method of the buffer object collection class:

```
void BufferClear(const int buffer_list_index,const int series_index) { this.m_buffers.Clear(buffer_list_index,series_index); }
```

**The methods of clearing buffer objects** clear data of all arrays of the specified buffer using the appropriate methods of the buffer object collection class considered above:

```
void BufferArrowClear(const int number,const int series_index)      { this.m_buffers.ClearBufferArrow(number,series_index);     }
void BufferLineClear(const int number,const int series_index)       { this.m_buffers.ClearBufferLine(number,series_index);      }
void BufferSectionClear(const int number,const int series_index)    { this.m_buffers.ClearBufferSection(number,series_index);   }
void BufferHistogramClear(const int number,const int series_index)  { this.m_buffers.ClearBufferHistogram(number,series_index); }
void BufferHistogram2Clear(const int number,const int series_index) { this.m_buffers.ClearBufferHistogram2(number,series_index);}
void BufferZigZagClear(const int number,const int series_index)     { this.m_buffers.ClearBufferZigZag(number,series_index);    }
void BufferFillingClear(const int number,const int series_index)    { this.m_buffers.ClearBufferFilling(number,series_index);   }
void BufferBarsClear(const int number,const int series_index)       { this.m_buffers.ClearBufferBars(number,series_index);      }
void BufferCandlesClear(const int number,const int series_index)    { this.m_buffers.ClearBufferCandles(number,series_index);   }
```

The **BuffersPrintShort()** method displays the short description of all indicator buffers created in the program and stored in the buffer object collection:

```
//+-------------------------------------------------------------------------+
//| Display short description of all indicator buffers within the collection|
//+-------------------------------------------------------------------------+
void CEngine::BuffersPrintShort(void)
  {
//--- Get the pointer to the collection list of buffer objects
   CArrayObj *list=this.GetListBuffers();
   if(list==NULL)
      return;
   int total=list.Total();
//--- In a loop by the number of buffers in the list,
//--- get the next buffer and display its brief description in the journal
   for(int i=0;i<total;i++)
     {
      CBuffer *buff=list.At(i);
      if(buff==NULL)
         continue;
      buff.PrintShort();
     }
  }
//+------------------------------------------------------------------+
```

The method logic is described in the code comments and is quite simple: in a loop by the list of all collection buffer objects, get the next buffer and display its short description using the PrintShort() method of the buffer object class.

I have already mentioned that we need to pass the pointer to the timeseries collection class object to the buffer object collection class.

We already have the **CollectionOnInit()** method. We pass all the necessary collections to the library by calling the method. Now it is time to add passing the necessary pointer to it. We do this by calling the previously considered OnInit() method of the buffer object collection class:

```
//--- Pass the pointers to all the necessary collections to the trading class and the indicator buffer collection class
   void                 CollectionOnInit(void)
                          {
                           this.m_trading.OnInit(this.GetAccountCurrent(),m_symbols.GetObject(),m_market.GetObject(),m_history.GetObject(),m_events.GetObject());
                           this.m_buffers.OnInit(this.m_time_series.GetObject());
                          }
```

This method is called in OnInit() of library-based programs.

To be more accurate, the function of initializing the OnInitDoEasy() library is called in OnInit(). In turn, the OnInitDoEasy() library contains calling this method along with other configuring actions.

**This concludes the improvement of the library classes**.

Let's test creation of the multi-period indicator.

### Testing

To perform the test, let's take the [test indicator from the previous article](https://www.mql5.com/en/articles/7886#node04) and save it in \\MQL5\\Indicators\\TestDoEasy\ **Part45\** as **TestDoEasyPart45.mq5**.

The indicator features a timeframe in its settings which should be used in its work. Besides, it will also have the ability to select various drawing types. The buffers displayed on the chart are to depend on selected drawing types.

The indicator string from the previous article

```
/*sinput*/   ENUM_TIMEFRAMES_MODE InpModeUsedTFs    =  TIMEFRAMES_MODE_CURRENT;            // Mode of used timeframes list
```

is replaced with the new one:

```
/*sinput*/   ENUM_TIMEFRAMES_MODE InpModeUsedTFs    =  TIMEFRAMES_MODE_LIST;            // Mode of used timeframes list
```

Here I have replaced the indicator working with the current timeframe with the one working with the timeframe list. Now the indicator waits for the list of timeframes, located in the array intended for their storage, to start working.

Next, add the input, where the necessary timeframe is set, to the settings (default is the current one). Then the selected timeframe is written to the array in the OnInit() handler. Thus, we will set the required chart period the indicator should take data from for displaying on the current one.

```
sinput   ENUM_TIMEFRAMES      InpPeriod         =  PERIOD_CURRENT;                  // Used chart period

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Write the name of the working timeframe selected in the settings to the InpUsedTFs variable
   InpUsedTFs=TimeframeDescription(InpPeriod);
```

After writing the timeframe to the array, the library uses this value to create required timeseries.

Next, we simply create buffers with all drawing types, set display flags for them in the data window depending on the appropriate input settings and set the timeframe specified in the settings for them:

```
//--- indicator buffers mapping
//--- Create all the necessary buffer objects
   engine.BufferCreateArrow();         // 2 arrays
   engine.BufferCreateLine();          // 2 arrays
   engine.BufferCreateSection();       // 2 arrays
   engine.BufferCreateHistogram();     // 2 arrays
   engine.BufferCreateHistogram2();    // 3 arrays
   engine.BufferCreateZigZag();        // 3 arrays
   engine.BufferCreateFilling();       // 2 arrays
   engine.BufferCreateBars();          // 5 arrays
   engine.BufferCreateCandles();       // 5 arrays
   engine.BufferCreateCalculate();     // 1 array

//--- Check the number of buffers specified in the 'properties' block
   if(engine.BuffersPropertyPlotsTotal()!=indicator_plots)
      Alert(TextByLanguage("Внимание! Значение \"indicator_plots\" должно быть ","Attention! Value of \"indicator_plots\" should be "),engine.BuffersPropertyPlotsTotal());
   if(engine.BuffersPropertyBuffersTotal()!=indicator_buffers)
      Alert(TextByLanguage("Внимание! Значение \"indicator_buffers\" должно быть ","Attention! Value of \"indicator_buffers\" should be "),engine.BuffersPropertyBuffersTotal());

//--- Create the color array and set non-default colors to all buffers within the collection
   color array_colors[]={clrDodgerBlue,clrRed,clrGray};
   engine.BuffersSetColors(array_colors);
//--- Set the line width for ZigZag (the sixth drawn buffer)
//--- It has the index of 5 considering that the starting point is zero
   CBuffer *buff_zz=engine.GetBufferByPlot(5);
   if(buff_zz!=NULL)
     {
      buff_zz.SetWidth(2);
     }

//--- In a loop by the list of collection buffer objects,
   for(int i=0;i<engine.GetListBuffers().Total();i++)
     {
      //--- get the next buffer
      CBuffer *buff=engine.GetListBuffers().At(i);
      if(buff==NULL)
         continue;
      //--- and set its display in the data window depending on its specified usage
      //--- and also the chart period selected in the settings
      buff.SetShowData(IsUse(buff.Status()));
      buff.SetTimeframe(InpPeriod);
     }
```

Since I am going to use the indexing direction as in the timeseries within indicators (due to how storing timeseries is arranged in the library), create yet another function for passing all data arrays from OnCalculate() to the library. Previously, this was done by the CopyData() function, in which the "as in the timeseries" flag was set for arrays and its previous status was then returned:

```
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

Next, I have "turned over" these arrays again for their correct operation with the library timeseries in OnCalculate():

```
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
```

To avoid unnecessary work, let's create yet another function in the file of the library service functions where the arrays passed to it receive the indexing direction as in the timeseries. Open \\MQL5\\Include\\DoEasy\\Services\ **DELib.mqh** and add the new function:

```
//+------------------------------------------------------------------+
//| Copy data from the second OnCalculate() form to the structure    |
//| and set the "as timeseries" flag to all arrays                   |
//+------------------------------------------------------------------+
void CopyDataAsSeries(const int rates_total,
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
//--- set the indexing direction or the arrays as in the timeseries
   ArraySetAsSeries(time,true);
   ArraySetAsSeries(open,true);
   ArraySetAsSeries(high,true);
   ArraySetAsSeries(low,true);
   ArraySetAsSeries(close,true);
   ArraySetAsSeries(tick_volume,true);
   ArraySetAsSeries(volume,true);
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
  }
//+------------------------------------------------------------------+
```

The function does the same as the previous one, but does not return the arrays to their original indexing. Now instead of calling the CopyData() function from the indicator, let's call the new one - CopyDataAsSeries(). This saves us from re-setting the desired indexing for the arrays:

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
      EventsHandling();             // Working with library events
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

//--- Calculate the indicator
   CBar *bar=NULL;         // Bar object for defining the candle direction
   uchar color_index=0;    // Color index to be set for the buffer depending on the candle direction

//--- Main calculation loop of the indicator
   for(int i=limit; i>WRONG_VALUE && !IsStopped(); i--)
     {
      //--- Clear the current bar of all created buffers
      engine.BufferArrowClear(0,0);
      engine.BufferLineClear(0,0);
      engine.BufferSectionClear(0,0);
      engine.BufferHistogramClear(0,0);
      engine.BufferHistogram2Clear(0,0);
      engine.BufferZigZagClear(0,0);
      engine.BufferFillingClear(0,0);
      engine.BufferBarsClear(0,0);
      engine.BufferCandlesClear(0,0);

      //--- Get the timeseries bar corresponding to the loop index time on the chart period specified in the settings
      bar=engine.SeriesGetBar(NULL,InpPeriod,time[i]);
      if(bar==NULL)
         continue;
      //--- Calculate the color index depending on the candle direction on the timeframe specified in the settings
      color_index=(bar.TypeBody()==BAR_BODY_TYPE_BULLISH ? 0 : bar.TypeBody()==BAR_BODY_TYPE_BEARISH ? 1 : 2);
      //--- Check the settings and calculate the arrow buffer
      if(IsUse(BUFFER_STATUS_ARROW))
         engine.BufferSetDataArrow(0,i,bar.Close(),color_index);
      //--- Check the settings and calculate the line buffer
      if(IsUse(BUFFER_STATUS_LINE))
         engine.BufferSetDataLine(0,i,bar.Open(),color_index);
      //--- Check the settings and calculate the section buffer
      if(IsUse(BUFFER_STATUS_SECTION))
         engine.BufferSetDataSection(0,i,bar.Close(),color_index);
      //--- Check the settings and calculate the histogram from zero buffer
      if(IsUse(BUFFER_STATUS_HISTOGRAM))
         engine.BufferSetDataHistogram(0,i,open[i],color_index);
      //--- Check the settings and calculate the 'histogram on two buffers' buffer
      if(IsUse(BUFFER_STATUS_HISTOGRAM2))
         engine.BufferSetDataHistogram2(0,i,bar.Open(),bar.Close(),color_index);
      //--- Check the settings and calculate the zigzag buffer
      if(IsUse(BUFFER_STATUS_ZIGZAG))
        {
         //--- Set the bar's Low value value for the zigzag (for bullish candles)
         double value1=bar.Low();
         double value2=value1;
         //--- If the candle is bearish (color index = 1), set the bar's High value for the zigzag
         if(color_index==1)
           {
            value1=value2=bar.High();
           }
         engine.BufferSetDataZigZag(0,i,value1,value2,color_index);
        }
      //--- Check the settings and calculate the filling buffer
      if(IsUse(BUFFER_STATUS_FILLING))
        {
         //--- Set filling border values for bullish candles
         double value1=bar.High();
         double value2=bar.Low();
         //--- In case of the bearish candle (color index = 1), swap the filling borders to change the color
         if(color_index==1)
           {
            value1=bar.Low();
            value2=bar.High();
           }
         engine.BufferSetDataFilling(0,i,value1,value2,color_index);
        }
      //--- Check the settings and calculate the bar buffer
      if(IsUse(BUFFER_STATUS_BARS))
         engine.BufferSetDataBars(0,i,bar.Open(),bar.High(),bar.Low(),bar.Close(),color_index);
      //--- Check the settings and calculate the candle buffer
      if(IsUse(BUFFER_STATUS_CANDLES))
         engine.BufferSetDataCandles(0,i,bar.Open(),bar.High(),bar.Low(),bar.Close(),color_index);
     }
//--- return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

The remaining actions and the OnCalculate() handler logic are described in detail in the code comments. Working with the multi-period indicator has become much easier. We do not need to calculate anything on our own. Instead, we can simply write data to the buffer to let the library define where to put the data and how to display it:

![](https://c.mql5.com/2/39/sqkXC8xitG.gif)

### What's next?

In the next article, we will continue the development of the indicator buffer collection class and arrange the indicator operation in multi-symbol mode.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave your questions, comments and suggestions in the comments.

Please keep in mind that here I have developed the MQL5 test indicator for MetaTrader 5.

The attached files are intended only for MetaTrader 5. The current library version has not been tested in MetaTrader 4.

After creating and testing the indicator buffer collection, I will try to implement some MQL5 features in MetaTrader 4.

[Back to contents](https://www.mql5.com/en/articles/8023#node00)

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

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8023](https://www.mql5.com/ru/articles/8023)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8023.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/8023/mql5.zip "Download MQL5.zip")(3772.33 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/353453)**
(6)


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
17 Jun 2020 at 18:34

**Igor Makanu:**

can this library already perform any practical tasks or is it still a work in progress?

I would like to see a practical example, for example - MACD from senior TFs in the subwindow, i.e. throw an indicator written with the help of the library on charts M1.... Н1... and see in the subwindow MACD on D1

A little less than half of the planned work is done.

If MACD calculation will be included in the code of the indicator made with the help of the library, then yes, it will work. If you need to output the standard one through the library, then... I think you can do it too - the data of the standard MACD should be written to the indicator buffers created on the basis of the library in OnCalculate(). In the example of this article, prices are written to the indicator buffers. Instead of prices - write MACD values. It is true that you need to take MACD from the required timeframe, which is natural. Further, and later, there will be classes for working with standard and [custom indicators](https://www.mql5.com/en/articles/5 "Article Switching to New Rails: Custom Indicators in MQL5") \- there will be easier than described here.

![Igor Makanu](https://c.mql5.com/avatar/2018/10/5BB56740-A283.jpg)

**[Igor Makanu](https://www.mql5.com/en/users/igorm)**
\|
17 Jun 2020 at 18:46

**Artyom Trishkin:**

A little less than half of what is planned is still done.

If MACD calculation will be included in the code of the indicator made with the help of the library, then yes, it will work. If you need to output the standard one through the library, then... I think you can do it too - the data of the standard MACD should be written to the indicator buffers created on the basis of the library in OnCalculate(). In the example of this article, prices are written to the indicator buffers. Instead of prices - write MACD values. It is true that you need to take MACD from the required timeframe, which is natural. Further, and later, there will be classes for working with standard and [custom indicators](https://www.mql5.com/en/articles/5 "Article Switching to New Rails: Custom Indicators in MQL5") \- it will be easier there than described here.

OK, so we need to wait

such tasks - to see an indicator from a senior TF (stochastic, MACD or just MA) are always in demand on traders' forums, the question is of course not how to do it, but how fast (in terms of code writing speed or convenient to write) your library can solve such tasks.

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
17 Jun 2020 at 20:06

**Igor Makanu:**

OK, so we'll just have to wait

such tasks - to watch an indicator from a senior TF (stochastic, MACD or just MA) are always in demand on traders' forums, the question of course is not how to do it, but how fast (in terms of code writing speed or convenient to write) your library can solve such tasks.

I tried to perform such a task without modifying the library. It can be done, but it requires extra steps.

Thanks for the tip - this is just the missing functionality of the calculation buffer object. It will be in the next article.

![thebeno](https://c.mql5.com/avatar/2022/2/62081ADF-8642.png)

**[thebeno](https://www.mql5.com/en/users/thebeno)**
\|
12 Feb 2022 at 18:35

Hi thanks for this library and your work.

I was not able to compile without errors. ( I have latest includes from part 90)

I did this changes:

MQL5\\Indicators\\TestDoEasy\\Part45\\TestDoEasyPart45.mq5

line 403:

-    engine.SetSoundsStandart(); --> **engine.SetSoundsStandard();**

MQL5\\Include\\DoEasy\\Objects\\Indicators\\Buffer.mqh

line 84:

-    virtual void      PrintShort(const bool dash=false,const bool symbol=false) ;  -->    **virtual void      PrintShort(const bool dash=false,const bool symbol=false) {return;}**

is it OK?

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
12 Feb 2022 at 23:46

**thebeno [#](https://www.mql5.com/en/forum/353453#comment_27677797):**

Hi thanks for this library and your work.

I was not able to compile without errors. ( I have latest includes from part 90)

I did this changes:

MQL5\\Indicators\\TestDoEasy\\Part45\\TestDoEasyPart45.mq5

line 403:

-    engine.SetSoundsStandart(); --> **engine.SetSoundsStandard();**

MQL5\\Include\\DoEasy\\Objects\\Indicators\\Buffer.mqh

line 84:

-    virtual void      PrintShort(const bool dash=false,const bool symbol=false) ;  -->    **virtual void      PrintShort(const bool dash=false,const bool symbol=false) {return;}**

is it OK?

Hi.

Yes, it is normal. The library is under development and is constantly changing. Accordingly, when connecting the 90th part of the library, it will not be possible to compile a program written for the 45th version.

You need to edit the code in the program itself for new changes.

In any case, at the end of the development of the library, all examples will be brought into line with its latest version.


![Calculating mathematical expressions (Part 2). Pratt and shunting yard parsers](https://c.mql5.com/2/39/MQL5-avatar-analysis__1.png)[Calculating mathematical expressions (Part 2). Pratt and shunting yard parsers](https://www.mql5.com/en/articles/8028)

In this article, we consider the principles of mathematical expression parsing and evaluation using parsers based on operator precedence. We will implement Pratt and shunting-yard parser, byte-code generation and calculations by this code, as well as view how to use indicators as functions in expressions and how to set up trading signals in Expert Advisors based on these indicators.

![Calculating mathematical expressions (Part 1). Recursive descent parsers](https://c.mql5.com/2/39/MQL5-avatar-analysis.png)[Calculating mathematical expressions (Part 1). Recursive descent parsers](https://www.mql5.com/en/articles/8027)

The article considers the basic principles of mathematical expression parsing and calculation. We will implement recursive descent parsers operating in the interpreter and fast calculation modes, based on a pre-built syntax tree.

![Timeseries in DoEasy library (part 46): Multi-period multi-symbol indicator buffers](https://c.mql5.com/2/39/MQL5-avatar-doeasy-library__2.png)[Timeseries in DoEasy library (part 46): Multi-period multi-symbol indicator buffers](https://www.mql5.com/en/articles/8115)

In this article, I am going to improve the classes of indicator buffer objects to work in the multi-symbol mode. This will pave the way for creating multi-symbol multi-period indicators in custom programs. I will add the missing functionality to the calculated buffer objects allowing us to create multi-symbol multi-period standard indicators.

![Quick Manual Trading Toolkit: Basic Functionality](https://c.mql5.com/2/39/Frame_1.png)[Quick Manual Trading Toolkit: Basic Functionality](https://www.mql5.com/en/articles/7892)

Today, many traders switch to automated trading systems which can require additional setup or can be fully automated and ready to use. However, there is a considerable part of traders who prefer trading manually, in the old fashioned way. In this article, we will create toolkit for quick manual trading, using hotkeys, and for performing typical trading actions in one click.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=koackivhrxhfntrlzhnzfsbwgotdfzsc&ssn=1769186130045204585&ssn_dr=0&ssn_sr=0&fv_date=1769186130&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8023&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Timeseries%20in%20DoEasy%20library%20(part%2045)%3A%20Multi-period%20indicator%20buffers%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918613084185447&fz_uniq=5070408768170169682&sv=2552)

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