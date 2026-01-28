---
title: Timeseries in DoEasy library (part 40): Library-based indicators - updating data in real time
url: https://www.mql5.com/en/articles/7771
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:36:17.562376
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/7771&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070418809803707772)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/7771#node01)
- [Improving timeseries classes](https://www.mql5.com/en/articles/7771#node02)
- [Creating and testing a multi-period indicator](https://www.mql5.com/en/articles/7771#node03)
- [What's next?](https://www.mql5.com/en/articles/7771#node04)


### Concept

In the [previous article](https://www.mql5.com/en/articles/7724), we set our eyes on the [DoEasy library](https://www.mql5.com/en/articles/5654) working in indicators. This type of programs requires a slightly different approach to building and updating timeseries due to some features of resource-saving calculation in indicators and limitations in obtaining data for the current symbol and chart period in an indicator launched on the same chart.

We managed to achieve the correct request and loading of history data. Now we need to develop the functionality for real-time update of all data from all timeseries used in the indicator (we assume that the indicator is multi-period and receives data for its operation from the specified chart timeframes it is launched on).

While constructing the indicator buffer data, we moved in the loop from the bar with the specified history data depth to the current (zero) bar. The most simple solution here is to take data by the loop index — the data has already been created in the library timeseries object, so nothing prevents us from receiving it by index. However, this works only when constructing static data. We face the issue of indexing by bar number during real-time data update in the timeseries lists. When adding a new bar to the timeseries list, the new bar has the index of 0 since the new bar is to become the zero (current) one. The indices of all previous bars in the timeseries are to be increased by 1. Thus, every time a new bar is opened on the chart, we need to add this newly appeared bar to the timeseries list and increase the numbers of all other bars in the updated timeseries list by one.

This is extremely impractical. Instead, we should apply working with bar time in the timeseries list — open time of each bar in the timeseries list always remains the same. The bar time is the starting point in any reference to any of the bars in the library timeseries collection. I am going to leave indexing by bar numbers as well. But the bar number is to be received not from bar properties. Instead, when requesting a bar by the timeseries index, the bar time is to be calculated by the requested index and the necessary bar is to be received from the timeseries list by the calculated bar time for its further use.

### Improving timeseries classes

In \\MQL5\\Include\\DoEasy\ **Defines.mqh**, remove the bar index property from the enumeration of bar object integer properties:

```
//+------------------------------------------------------------------+
//| Bar integer properties                                           |
//+------------------------------------------------------------------+
enum ENUM_BAR_PROP_INTEGER
  {
   BAR_PROP_INDEX = 0,                                      // Bar index in timeseries
   BAR_PROP_TYPE,                                           // Bar type (from the ENUM_BAR_BODY_TYPE enumeration)
```

Replace it with the "Bar time" property and decrease the number of bar object integer properties by 1 (from **14 to 13**):

```
//+------------------------------------------------------------------+
//| Bar integer properties                                           |
//+------------------------------------------------------------------+
enum ENUM_BAR_PROP_INTEGER
  {
   BAR_PROP_TIME = 0,                                       // Bar period start time
   BAR_PROP_TYPE,                                           // Bar type (from the ENUM_BAR_BODY_TYPE enumeration)
   BAR_PROP_PERIOD,                                         // Bar period (timeframe)
   BAR_PROP_SPREAD,                                         // Bar spread
   BAR_PROP_VOLUME_TICK,                                    // Bar tick volume
   BAR_PROP_VOLUME_REAL,                                    // Bar exchange volume

   BAR_PROP_TIME_DAY_OF_YEAR,                               // Bar day serial number in a year
   BAR_PROP_TIME_YEAR,                                      // A year the bar belongs to
   BAR_PROP_TIME_MONTH,                                     // A month the bar belongs to
   BAR_PROP_TIME_DAY_OF_WEEK,                               // Bar week day
   BAR_PROP_TIME_DAY,                                       // Bar day of month (number)
   BAR_PROP_TIME_HOUR,                                      // Bar hour
   BAR_PROP_TIME_MINUTE,                                    // Bar minute
  };
#define BAR_PROP_INTEGER_TOTAL (13)                         // Total number of integer bar properties
#define BAR_PROP_INTEGER_SKIP  (0)                          // Number of bar properties not used in sorting
//+------------------------------------------------------------------+
```

Accordingly, in the enumeration of possible bar sorting criteria, we need to remove sorting by index and replace it with sorting by bar time:

```
//+------------------------------------------------------------------+
//| Possible bar sorting criteria                                    |
//+------------------------------------------------------------------+
#define FIRST_BAR_DBL_PROP          (BAR_PROP_INTEGER_TOTAL-BAR_PROP_INTEGER_SKIP)
#define FIRST_BAR_STR_PROP          (BAR_PROP_INTEGER_TOTAL-BAR_PROP_INTEGER_SKIP+BAR_PROP_DOUBLE_TOTAL-BAR_PROP_DOUBLE_SKIP)
enum ENUM_SORT_BAR_MODE
  {
//--- Sort by integer properties
   SORT_BY_BAR_TIME = 0,                                    // Sort by bar period start time
   SORT_BY_BAR_TYPE,                                        // Sort by bar type (from the ENUM_BAR_BODY_TYPE enumeration)
   SORT_BY_BAR_PERIOD,                                      // Sort by bar period (timeframe)
   SORT_BY_BAR_SPREAD,                                      // Sort by bar spread
   SORT_BY_BAR_VOLUME_TICK,                                 // Sort by bar tick volume
   SORT_BY_BAR_VOLUME_REAL,                                 // Sort by bar exchange volume
   SORT_BY_BAR_TIME_DAY_OF_YEAR,                            // Sort by bar day number in a year
   SORT_BY_BAR_TIME_YEAR,                                   // Sort by a year the bar belongs to
   SORT_BY_BAR_TIME_MONTH,                                  // Sort by a month the bar belongs to
   SORT_BY_BAR_TIME_DAY_OF_WEEK,                            // Sort by a bar week day
   SORT_BY_BAR_TIME_DAY,                                    // Sort by a bar day
   SORT_BY_BAR_TIME_HOUR,                                   // Sort by a bar hour
   SORT_BY_BAR_TIME_MINUTE,                                 // Sort by a bar minute
//--- Sort by real properties
   SORT_BY_BAR_OPEN = FIRST_BAR_DBL_PROP,                   // Sort by bar open price
   SORT_BY_BAR_HIGH,                                        // Sort by the highest price for the bar period
   SORT_BY_BAR_LOW,                                         // Sort by the lowest price for the bar period
   SORT_BY_BAR_CLOSE,                                       // Sort by a bar close price
   SORT_BY_BAR_CANDLE_SIZE,                                 // Sort by a candle price
   SORT_BY_BAR_CANDLE_SIZE_BODY,                            // Sort by a candle body size
   SORT_BY_BAR_CANDLE_BODY_TOP,                             // Sort by a candle body top
   SORT_BY_BAR_CANDLE_BODY_BOTTOM,                          // Sort by a candle body bottom
   SORT_BY_BAR_CANDLE_SIZE_SHADOW_UP,                       // Sort by candle upper wick size
   SORT_BY_BAR_CANDLE_SIZE_SHADOW_DOWN,                     // Sort by candle lower wick size
//--- Sort by string properties
   SORT_BY_BAR_SYMBOL = FIRST_BAR_STR_PROP,                 // Sort by a bar symbol
  };
//+------------------------------------------------------------------+
```

Re-build the **CBar** class in \\MQL5\\Include\\DoEasy\\Objects\\Series\ **Bar.mqh** to work with the bar time.

Previously, the SetSymbolPeriod() method set a specified symbol, chart period and bar index for a bar object. Now, the index is replaced with the bar time:

```
//--- Set (1) bar symbol, timeframe and time, (2) bar object parameters
   void              SetSymbolPeriod(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time);
   void              SetProperties(const MqlRates &rates);
```

Let's fix the method implementation as well:

```
//+------------------------------------------------------------------+
//| Set bar symbol, timeframe and index                              |
//+------------------------------------------------------------------+
void CBar::SetSymbolPeriod(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time)
  {
   this.SetProperty(BAR_PROP_TIME,time);
   this.SetProperty(BAR_PROP_SYMBOL,symbol);
   this.SetProperty(BAR_PROP_PERIOD,timeframe);
   this.m_digits=(int)::SymbolInfoInteger(symbol,SYMBOL_DIGITS);
   this.m_period_description=TimeframeDescription(timeframe);
  }
//+------------------------------------------------------------------+
```

Instead of the bar index, the first parametric class constructor now receives the bar time the CBar class constructor was called from for more data. Add the variable used to pass the description of the class method the bar object creation is called in:

```
//--- Constructors
                     CBar(){;}
                     CBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time,const string source);
                     CBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const MqlRates &rates);
```

Let's also fix the implementation of the constructor — instead of the index, use the bar time, and add the variable specifying the class method the constructor was called from to the text describing the error of getting history data:

```
//+------------------------------------------------------------------+
//| Constructor 1                                                    |
//+------------------------------------------------------------------+
CBar::CBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time,const string source)
  {
   this.m_type=COLLECTION_SERIES_ID;
   MqlRates rates_array[1];
   this.SetSymbolPeriod(symbol,timeframe,time);
   ::ResetLastError();
//--- If failed to get the requested data by time and write bar data to the MqlRates array,
//--- display an error message, create and fill the structure with zeros, and write it to the rates_array array
   if(::CopyRates(symbol,timeframe,time,1,rates_array)<1)
     {
      int err_code=::GetLastError();
      ::Print
        (
         DFUN,"(1)-> ",source,symbol," ",TimeframeDescription(timeframe)," ",::TimeToString(time),": ",
         CMessage::Text(MSG_LIB_TEXT_BAR_FAILED_GET_BAR_DATA),". ",
         CMessage::Text(MSG_LIB_SYS_ERROR),"> ",CMessage::Text(err_code)," ",
         CMessage::Retcode(err_code)
        );
      //--- Set the requested bar time to the structure with zero fields
      MqlRates err={0};
      err.time=time;
      rates_array[0]=err;
     }
   ::ResetLastError();
//--- If failed to set time to the time structure, display the error message
   if(!::TimeToStruct(rates_array[0].time,this.m_dt_struct))
     {
      int err_code=::GetLastError();
      ::Print
        (
         DFUN,"(1) ",symbol," ",TimeframeDescription(timeframe)," ",::TimeToString(time),": ",
         CMessage::Text(MSG_LIB_TEXT_BAR_FAILED_DT_STRUCT_WRITE),". ",
         CMessage::Text(MSG_LIB_SYS_ERROR),"> ",CMessage::Text(err_code)," ",
         CMessage::Retcode(err_code)
        );
     }
//--- Set the bar properties
   this.SetProperties(rates_array[0]);
  }
//+------------------------------------------------------------------+
```

Adding the **source** variable value to the displayed message of an error when obtaining history data allows finding the class and its method an attempt to create a new bar object was made from.

The second parametric constructor now also applies the bar time instead of its index:

```
//+------------------------------------------------------------------+
//| Constructor 2                                                    |
//+------------------------------------------------------------------+
CBar::CBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const MqlRates &rates)
  {
   this.m_type=COLLECTION_SERIES_ID;
   this.SetSymbolPeriod(symbol,timeframe,rates.time);
   ::ResetLastError();
//--- If failed to set time to the time structure, display the error message,
//--- create and fill the structure with zeros, set the bar properties from this structure and exit
   if(!::TimeToStruct(rates.time,this.m_dt_struct))
     {
      int err_code=::GetLastError();
      ::Print
        (
         DFUN,"(2) ",symbol," ",TimeframeDescription(timeframe)," ",::TimeToString(rates.time),": ",
         CMessage::Text(MSG_LIB_TEXT_BAR_FAILED_DT_STRUCT_WRITE),". ",
         CMessage::Text(MSG_LIB_SYS_ERROR),"> ",CMessage::Text(err_code)," ",
         CMessage::Retcode(err_code)
        );
      //--- Set the requested bar time to the structure with zero fields
      MqlRates err={0};
      err.time=rates.time;
      this.SetProperties(err);
      return;
     }
//--- Set the bar properties
   this.SetProperties(rates);
  }
//+------------------------------------------------------------------+
```

In the block of methods for a simplified access to bar object properties of the public class section, rename the Period() method to Timeframe() and remove the Index() method returning the already removed bar property:

```
//+------------------------------------------------------------------+
//| Methods of simplified access to bar object properties            |
//+------------------------------------------------------------------+
//--- Return the (1) type, (2) period, (3) spread, (4) tick, (5) exchange volume,
//--- (6) bar period start time, (7) year, (8) month the bar belongs to
//--- (9) week number since the year start, (10) week number since the month start
//--- (11) day, (12) hour, (13) minute
   ENUM_BAR_BODY_TYPE TypeBody(void)                                    const { return (ENUM_BAR_BODY_TYPE)this.GetProperty(BAR_PROP_TYPE);  }
   ENUM_TIMEFRAMES   Timeframe(void)                                    const { return (ENUM_TIMEFRAMES)this.GetProperty(BAR_PROP_PERIOD);   }
   int               Spread(void)                                       const { return (int)this.GetProperty(BAR_PROP_SPREAD);               }
   long              VolumeTick(void)                                   const { return this.GetProperty(BAR_PROP_VOLUME_TICK);               }
   long              VolumeReal(void)                                   const { return this.GetProperty(BAR_PROP_VOLUME_REAL);               }
   datetime          Time(void)                                         const { return (datetime)this.GetProperty(BAR_PROP_TIME);            }
   long              Year(void)                                         const { return this.GetProperty(BAR_PROP_TIME_YEAR);                 }
   long              Month(void)                                        const { return this.GetProperty(BAR_PROP_TIME_MONTH);                }
   long              DayOfWeek(void)                                    const { return this.GetProperty(BAR_PROP_TIME_DAY_OF_WEEK);          }
   long              DayOfYear(void)                                    const { return this.GetProperty(BAR_PROP_TIME_DAY_OF_YEAR);          }
   long              Day(void)                                          const { return this.GetProperty(BAR_PROP_TIME_DAY);                  }
   long              Hour(void)                                         const { return this.GetProperty(BAR_PROP_TIME_HOUR);                 }
   long              Minute(void)                                       const { return this.GetProperty(BAR_PROP_TIME_MINUTE);               }
   long              Index(void)                                        const { return this.GetProperty(BAR_PROP_INDEX);                     }
```

Instead of returning a non-existent bar object property, the Index() method now returns the calculated value by bar time:

```
//--- Return bar symbol
   string            Symbol(void)                                       const { return this.GetProperty(BAR_PROP_SYMBOL);                    }
//--- Return bar index on the specified timeframe the bar time falls into
   int               Index(const ENUM_TIMEFRAMES timeframe=PERIOD_CURRENT)  const
                       { return ::iBarShift(this.Symbol(),(timeframe>PERIOD_CURRENT ? timeframe : this.Timeframe()),this.Time());            }
//+------------------------------------------------------------------+
```

The method returns the bar index of the current timeseries for the timeframe specified in the method input calculated by the [iBarShift()](https://www.mql5.com/en/docs/series/ibarshift) function.

In the method returning a short bar object name, we now call the newly described method with the default value of [PERIOD\_CURRENT](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes), which returns the index for the timeseries the bar object belongs to:

```
//+------------------------------------------------------------------+
//| Return the bar object short name                                 |
//+------------------------------------------------------------------+
string CBar::Header(void)
  {
   return
     (
      CMessage::Text(MSG_LIB_TEXT_BAR)+" \""+this.GetProperty(BAR_PROP_SYMBOL)+"\" "+
      TimeframeDescription((ENUM_TIMEFRAMES)this.GetProperty(BAR_PROP_PERIOD))+"["+(string)this.Index()+"]"
     );
  }
//+------------------------------------------------------------------+
```

Remove the block returning the bar index description from the method returning the description of the bar object integer property:

```
//+------------------------------------------------------------------+
//| Return the description of the bar integer property               |
//+------------------------------------------------------------------+
string CBar::GetPropertyDescription(ENUM_BAR_PROP_INTEGER property)
  {
   return
     (
      property==BAR_PROP_INDEX               ?  CMessage::Text(MSG_LIB_TEXT_BAR_INDEX)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BAR_PROP_TYPE                ?  CMessage::Text(MSG_ORD_TYPE)+
```

Instead, set a code block returning a bar time (full method listing):

```
//+------------------------------------------------------------------+
//| Return the description of the bar integer property               |
//+------------------------------------------------------------------+
string CBar::GetPropertyDescription(ENUM_BAR_PROP_INTEGER property)
  {
   return
     (
      property==BAR_PROP_TIME                ?  CMessage::Text(MSG_LIB_TEXT_BAR_TIME)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::TimeToString(this.GetProperty(property),TIME_DATE|TIME_MINUTES|TIME_SECONDS)
         )  :
      property==BAR_PROP_TYPE                ?  CMessage::Text(MSG_ORD_TYPE)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.BodyTypeDescription()
         )  :
      property==BAR_PROP_PERIOD              ?  CMessage::Text(MSG_LIB_TEXT_BAR_PERIOD)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.m_period_description
         )  :
      property==BAR_PROP_SPREAD              ?  CMessage::Text(MSG_LIB_TEXT_BAR_SPREAD)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BAR_PROP_VOLUME_TICK         ?  CMessage::Text(MSG_LIB_TEXT_BAR_VOLUME_TICK)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BAR_PROP_VOLUME_REAL         ?  CMessage::Text(MSG_LIB_TEXT_BAR_VOLUME_REAL)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BAR_PROP_TIME_YEAR           ?  CMessage::Text(MSG_LIB_TEXT_BAR_TIME_YEAR)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.Year()
         )  :
      property==BAR_PROP_TIME_MONTH          ?  CMessage::Text(MSG_LIB_TEXT_BAR_TIME_MONTH)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+MonthDescription((int)this.Month())
         )  :
      property==BAR_PROP_TIME_DAY_OF_YEAR    ?  CMessage::Text(MSG_LIB_TEXT_BAR_TIME_DAY_OF_YEAR)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)::IntegerToString(this.DayOfYear(),3,'0')
         )  :
      property==BAR_PROP_TIME_DAY_OF_WEEK    ?  CMessage::Text(MSG_LIB_TEXT_BAR_TIME_DAY_OF_WEEK)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+DayOfWeekDescription((ENUM_DAY_OF_WEEK)this.DayOfWeek())
         )  :
      property==BAR_PROP_TIME_DAY            ?  CMessage::Text(MSG_LIB_TEXT_BAR_TIME_DAY)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)::IntegerToString(this.Day(),2,'0')
         )  :
      property==BAR_PROP_TIME_HOUR           ?  CMessage::Text(MSG_LIB_TEXT_BAR_TIME_HOUR)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)::IntegerToString(this.Hour(),2,'0')
         )  :
      property==BAR_PROP_TIME_MINUTE         ?  CMessage::Text(MSG_LIB_TEXT_BAR_TIME_MINUTE)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)::IntegerToString(this.Minute(),2,'0')
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

This completes the changes in the bar object class.

If we take a close look at the lists of the standard library classes, we will see two files in MQL5\\Include\ **Indicators\**: **Series.mqh** and **TimeSeries.mqh**.

The library also has class files of the same name, which is wrong. Let's rename our two classes — add DE (DoEasy) to their names, as well as to the names of their files. Also, change the names where access to these files and classes occurs. These changes have affected three files: Series.mqh (now renamed to SeriesDE.mqh and the CSeriesDE class), TimeSeries.mqh (now renamed to TimeSeriesDE.mqh and the CTimeSeriesDE class) and TimeSeriesCollection.mqh (applies both renamed classes). Let's consider all these files and their classes in order.

The Series.mqh file is now saved as \\MQL5\\Include\\DoEasy\\Objects\\Series\ **SeriesDE.mqh** and the class name changes appropriately as well:

```
//+------------------------------------------------------------------+
//| Timeseries class                                                 |
//+------------------------------------------------------------------+
class CSeriesDE : public CBaseObj
  {
private:
```

Accordingly, the method returning the class object now features the new class type:

```
public:
//--- Return (1) oneself and (2) the timeseries list
   CSeriesDE        *GetObject(void)                                    { return &this;         }
```

The public method returning the bar object by index as in the GetBarBySeriesIndex timeseries has been renamed to GetBar(). Let's add one more method of the same kind to return the bar object by its open time in the timeseries:

```
//--- Return the bar object by (1) a real index in the list, (2) an index as in the timeseries, (3) time and (4) the real list size
   CBar             *GetBarByListIndex(const uint index);
   CBar             *GetBar(const uint index);
   CBar             *GetBar(const datetime time);
   int               DataTotal(void)                                       const { return this.m_list_series.Total();               }
```

Thus, we now have two overloaded methods for returning a bar object by time and by index.

**Implementing the method for returning a bar object by its open time:**

```
//+------------------------------------------------------------------+
//| Return the bar object by time in the timeseries                  |
//+------------------------------------------------------------------+
CBar *CSeriesDE::GetBar(const datetime time)
  {
   CBar *obj=new CBar(this.m_symbol,this.m_timeframe,time,DFUN_ERR_LINE);
   if(obj==NULL)
      return NULL;
   this.m_list_series.Sort(SORT_BY_BAR_TIME);
   int index=this.m_list_series.Search(obj);
   delete obj;
   CBar *bar=this.m_list_series.At(index);
   return bar;
  }
//+------------------------------------------------------------------+
```

The method receives the time to be used to find and return the appropriate bar object.

Create a temporary bar object for the current timeseries with the time property equal to the one passed to the method.

Set the flag of sorting the list of bar objects by time and search the list for the bar object with the time property equal to the one passed to the method.

The search yields the bar index in the list or -1 if unsuccessful.

Remove a temporary bar object and get the necessary bar from the list by the obtained index. If the index is less than zero, the [At() method of the CArrayObj class](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjat) returns NULL.

Return either an object bar if the object is found by time, or NULL from the method.

**Implementing the method for returning a bar object by index:**

```
//+------------------------------------------------------------------+
//| Return the bar object by index as in the timeseries              |
//+------------------------------------------------------------------+
CBar *CSeriesDE::GetBar(const uint index)
  {
   datetime time=::iTime(this.m_symbol,this.m_timeframe,index);
   if(time==0)
      return NULL;
   return this.GetBar(time);
  }
//+------------------------------------------------------------------+
```

The method receives the necessary bar index.

Use the [iTime()](https://www.mql5.com/en/docs/series/itime) function to get the bar time by index and return the result of the GetBar() method operation considered above, which returns the bar object by obtained time.

In the public class section, along with the methods returning the main bar properties by index, declare the methods returning the same properties by bar time:

```
//--- Return (1) Open, (2) High, (3) Low, (4) Close, (5) time, (6) tick volume, (7) real volume, (8) bar spread by index
   double            Open(const uint index,const bool from_series=true);
   double            High(const uint index,const bool from_series=true);
   double            Low(const uint index,const bool from_series=true);
   double            Close(const uint index,const bool from_series=true);
   datetime          Time(const uint index,const bool from_series=true);
   long              TickVolume(const uint index,const bool from_series=true);
   long              RealVolume(const uint index,const bool from_series=true);
   int               Spread(const uint index,const bool from_series=true);

//--- Return (1) Open, (2) High, (3) Low, (4) Close, (5) time, (6) tick volume, (7) real volume, (8) bar spread by index
   double            Open(const datetime time);
   double            High(const datetime time);
   double            Low(const datetime time);
   double            Close(const datetime time);
   datetime          Time(const datetime time);
   long              TickVolume(const datetime time);
   long              RealVolume(const datetime time);
   int               Spread(const datetime time);
```

Implementation of the declared methods will be considered later.

In the same public class section, declare the method allowing to write the specified timeseries object data to the array passed to the method:

```
//--- (1) Create and (2) update the timeseries list
   int               Create(const uint required=0);
   void              Refresh(SDataCalculate &data_calculate);
//--- Copy the specified double property of the timeseries to the array
//--- Regardless of the array indexing direction, copying is performed the same way as copying to a timeseries array
   bool              CopyToBufferAsSeries(const ENUM_BAR_PROP_DOUBLE property,double &array[],const double empty=EMPTY_VALUE);

//--- Create and send the "New bar" event to the control program chart
   void              SendEvent(void);
```

Suppose that we need to write the timeseries data to the indicator buffer in one go. The bar object is able to contain many different properties — both integer and real. Any of the bar object real properties can be written to the array using this method. All data is written to the array the same way as when writing to the timeseries array — current bar data stored in the timeseries object at the end of the list is written to the zero index of the recipient array, i.e. writing is done backwards.

Let's have a look at its implementation:

```
//+------------------------------------------------------------------+
//| Copy the specified double property of the timeseries to the array|
//+------------------------------------------------------------------+
bool CSeriesDE::CopyToBufferAsSeries(const ENUM_BAR_PROP_DOUBLE property,double &array[],const double empty=EMPTY_VALUE)
  {
//--- Get the number of bars in the timeseries list
   int total=this.m_list_series.Total();
   if(total==0)
      return false;
//--- If a dynamic array is passed to the method and its size is not equal to that of the timeseries list,
//--- set the new size of the passed array equal to that of the timeseries list
   if(::ArrayIsDynamic(array) && ::ArraySize(array)!=total)
      if(::ArrayResize(array,total,this.m_required)==WRONG_VALUE)
         return false;
//--- In the loop from the very last timeseries list element (from the current bar)
   int n=0;
   for(int i=total-1;i>WRONG_VALUE && !::IsStopped();i--)
     {
      //--- get the next bar object by the loop index,
      CBar *bar=this.m_list_series.At(i);
      //--- calculate the index, based on which the bar property is saved to the passed array
      n=total-1-i;
      //--- write the value of the obtained bar property using the calculated index
      //--- if the bar is not received or the property is equal to zero, write the value passed to the method as "empty" to the array
      array[n]=(bar==NULL ? empty : (bar.GetProperty(property)>0 && bar.GetProperty(property)<EMPTY_VALUE ? bar.GetProperty(property) : empty));
     }
   return true;
  }
//+------------------------------------------------------------------+
```

As we can see, the recipient array index is calculated so that the very last value from the source array falls into the zero cell of the recipient array. Therefore, our timeseries list (requested bar property) is written to the array (for example, the indicator buffer) in the order of bars numbering on a symbol chart, while bar objects in the timeseries list are located in reverse order — the bar with the most recent time (the current bar) is located at the end of the list. This allows us to quickly copy the properties of all bars from the timeseries list to the indicator buffer if the timeframe of the copied timeseries matches the chart timeframe, for which we copy the timeseries to the buffer using the method.

In both class constructors, set the flag of sorting the timeseries list by bar time:

```
//+------------------------------------------------------------------+
//| Constructor 1 (current symbol and period timeseries)             |
//+------------------------------------------------------------------+
CSeriesDE::CSeriesDE(void) : m_bars(0),m_amount(0),m_required(0),m_sync(false)
  {
   this.m_list_series.Clear();
   this.m_list_series.Sort(SORT_BY_BAR_TIME);
   this.SetSymbolPeriod(NULL,(ENUM_TIMEFRAMES)::Period());
   this.m_period_description=TimeframeDescription(this.m_timeframe);
  }
//+------------------------------------------------------------------+
//| Constructor 2 (specified symbol and period timeseries)           |
//+------------------------------------------------------------------+
CSeriesDE::CSeriesDE(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint required=0) : m_bars(0), m_amount(0),m_required(0),m_sync(false)
  {
   this.m_list_series.Clear();
   this.m_list_series.Sort(SORT_BY_BAR_TIME);
   this.SetSymbolPeriod(symbol,timeframe);
   this.m_sync=this.SetRequiredUsedData(required,0);
   this.m_period_description=TimeframeDescription(this.m_timeframe);
  }
//+------------------------------------------------------------------+
```

In the method of creating the timeseries list, replace sorting by index with sorting by time and complement the text displayed in case of bar object creation errors and  errors occurred while adding it to the timeseries list:

```
//+------------------------------------------------------------------+
//| Create the timeseries list                                       |
//+------------------------------------------------------------------+
int CSeriesDE::Create(const uint required=0)
  {
//--- If the required history depth is not set for the list yet,
//--- display the appropriate message and return zero,
   if(this.m_amount==0)
     {
      ::Print(DFUN,this.m_symbol," ",TimeframeDescription(this.m_timeframe),": ",CMessage::Text(MSG_LIB_TEXT_BAR_TEXT_FIRS_SET_AMOUNT_DATA));
      return 0;
     }
//--- otherwise, if the passed 'required' value exceeds zero and is not equal to the one already set,
//--- while being lower than the available bar number,
//--- set the new value of the required history depth for the list
   else if(required>0 && this.m_amount!=required && required<this.m_bars)
     {
      //--- If failed to set a new value, return zero
      if(!this.SetRequiredUsedData(required,0))
         return 0;
     }
//--- For the rates[] array we are to receive historical data to,
//--- set the flag of direction like in the timeseries,
//--- clear the bar object list and set the flag of sorting by bar index
   MqlRates rates[];
   ::ArraySetAsSeries(rates,true);
   this.m_list_series.Clear();
   this.m_list_series.Sort(SORT_BY_BAR_TIME);
   ::ResetLastError();
//--- Get historical data of the MqlRates structure to the rates[] array starting from the current bar in the amount of m_amount,
//--- if failed to get data, display the appropriate message and return zero
   int copied=::CopyRates(this.m_symbol,this.m_timeframe,0,(uint)this.m_amount,rates),err=ERR_SUCCESS;
   if(copied<1)
     {
      err=::GetLastError();
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_BAR_FAILED_GET_SERIES_DATA)," ",this.m_symbol," ",TimeframeDescription(this.m_timeframe),". ",
                   CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(err),CMessage::Retcode(err));
      return 0;
     }
//--- Historical data is received in the rates[] array
//--- In the rates[] array loop,
   for(int i=0; i<copied; i++)
     {
      //--- create a new bar object out of the current MqlRates structure by the loop index
      ::ResetLastError();
      CBar* bar=new CBar(this.m_symbol,this.m_timeframe,rates[i]);
      if(bar==NULL)
        {
         ::Print
           (
            DFUN,CMessage::Text(MSG_LIB_SYS_FAILED_CREATE_BAR_OBJ)," ",this.Header()," ",::TimeToString(rates[i].time),". ",
            CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(::GetLastError())
           );
         continue;
        }
      //--- If failed to add bar object to the list,
      //--- display the appropriate message with the error description in the journal
      if(!this.m_list_series.Add(bar))
        {
         err=::GetLastError();
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_BAR_FAILED_ADD_TO_LIST)," ",bar.Header()," ",::TimeToString(rates[i].time),". ",
                      CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(err),CMessage::Retcode(err));
        }
     }
//--- Return the size of the created bar object list
   return this.m_list_series.Total();
  }
//+------------------------------------------------------------------+
```

The method of updating the list and timeseries data has also been slightly updated:

```
//+------------------------------------------------------------------+
//| Update timeseries list and data                                  |
//+------------------------------------------------------------------+
void CSeriesDE::Refresh(SDataCalculate &data_calculate)
  {
//--- If the timeseries is not used, exit
   if(!this.m_available)
      return;
   MqlRates rates[1];
//--- Set the flag of sorting the list of bars by time
   this.m_list_series.Sort(SORT_BY_BAR_TIME);
//--- If a new bar is present on a symbol and period,
   if(this.IsNewBarManual(data_calculate.rates.time))
     {
      //--- create a new bar object and add it to the end of the list
      CBar *new_bar=new CBar(this.m_symbol,this.m_timeframe,this.m_new_bar_obj.TimeNewBar(),DFUN_ERR_LINE);
      if(new_bar==NULL)
         return;
      if(!this.m_list_series.InsertSort(new_bar))
        {
         delete new_bar;
         return;
        }
      //--- Write the very first date by a period symbol at the moment and the new time of opening the last bar by a period symbol
      this.SetServerDate();
      //--- if the timeseries exceeds the requested number of bars, remove the earliest bar
      if(this.m_list_series.Total()>(int)this.m_required)
         this.m_list_series.Delete(0);
      //--- save the new bar time as the previous one for the subsequent new bar check
      this.SaveNewBarTime(data_calculate.rates.time);
     }

//--- Get the bar index with the maximum time (zero bar) and bar object from the list by the obtained index
   int index=CSelect::FindBarMax(this.GetList(),BAR_PROP_TIME);
   CBar *bar=this.m_list_series.At(index);
   if(bar==NULL)
      return;
//--- if the work is performed in an indicator and the timeseries belongs to the current symbol and timeframe,
//--- copy price parameters (passed to the method from the outside) to the bar price structure
   int copied=1;
   if(this.m_program==PROGRAM_INDICATOR && this.m_symbol==::Symbol() && this.m_timeframe==(ENUM_TIMEFRAMES)::Period())
     {
      rates[0].time=data_calculate.rates.time;
      rates[0].open=data_calculate.rates.open;
      rates[0].high=data_calculate.rates.high;
      rates[0].low=data_calculate.rates.low;
      rates[0].close=data_calculate.rates.close;
      rates[0].tick_volume=data_calculate.rates.tick_volume;
      rates[0].real_volume=data_calculate.rates.real_volume;
      rates[0].spread=data_calculate.rates.spread;
     }
//--- otherwise, get data to the bar price structure from the environment
   else
      copied=::CopyRates(this.m_symbol,this.m_timeframe,0,1,rates);
//--- If the prices are obtained, set the new properties from the price structure for the bar object
   if(copied==1)
      bar.SetProperties(rates[0]);
  }
//+------------------------------------------------------------------+
```

Here the list sorting is now also set by time. When creating a new bar object, pass the bar time from the "New bar" object to the class constructor since we add the new bar to the list only when defining the fact of opening a new bar, while the "New bar" object already contains the bar open time. Pass it to the constructor. In addition, pass the description of the method, where a new bar object is created, to the constructor. If failed to create a new bar object, the message is sent to the journal from the constructor containing the CSeriesDE::Refresh method and the code string the CBar class constructor was called from.

To get the most recent (current) bar from the timeseries list, find it by the maximum time of all bar objects in the timeseries list. To do this, first find the bar object index with the maximum time using the FindBarMax() method of the CSelect class. Using the obtained index, take the very last bar from the list. That bar will be the current one. If, for some reason, we are unable to get the current bar index, the index value is -1. [When receiving the list element using the At() method](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjat) in case of a negative index, we get NULL. If it is actually null, simply exit the update method.

**The methods for returning main bar object properties by time:**

```
//+------------------------------------------------------------------+
//| Return bar's Open by time                                        |
//+------------------------------------------------------------------+
double CSeriesDE::Open(const datetime time)
  {
   CBar *bar=this.GetBar(time);
   return(bar!=NULL ? bar.Open() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//| Return bar's High by time                                        |
//+------------------------------------------------------------------+
double CSeriesDE::High(const datetime time)
  {
   CBar *bar=this.GetBar(time);
   return(bar!=NULL ? bar.High() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//| Return bar's Low by time                                         |
//+------------------------------------------------------------------+
double CSeriesDE::Low(const datetime time)
  {
   CBar *bar=this.GetBar(time);
   return(bar!=NULL ? bar.Low() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//| Return bar's Close by time                                       |
//+------------------------------------------------------------------+
double CSeriesDE::Close(const datetime time)
  {
   CBar *bar=this.GetBar(time);
   return(bar!=NULL ? bar.Close() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//| Return bar time by time                                          |
//+------------------------------------------------------------------+
datetime CSeriesDE::Time(const datetime time)
  {
   CBar *bar=this.GetBar(time);
   return(bar!=NULL ? bar.Time() : 0);
  }
//+------------------------------------------------------------------+
//| Return bar tick volume by time                                   |
//+------------------------------------------------------------------+
long CSeriesDE::TickVolume(const datetime time)
  {
   CBar *bar=this.GetBar(time);
   return(bar!=NULL ? bar.VolumeTick() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//| Return bar real volume by time                                   |
//+------------------------------------------------------------------+
long CSeriesDE::RealVolume(const datetime time)
  {
   CBar *bar=this.GetBar(time);
   return(bar!=NULL ? bar.VolumeReal() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//| Return bar spread by time                                        |
//+------------------------------------------------------------------+
int CSeriesDE::Spread(const datetime time)
  {
   CBar *bar=this.GetBar(time);
   return(bar!=NULL ? bar.Spread() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
```

They all work the same way:

get the bar object from the timeseries list by time and return the value of the appropriate property considering the error receiving the bar object.

The method of creating and sending the "New bar" event on the control program chart has also been improved considering the need to get the current bar object by time:

```
//+------------------------------------------------------------------+
//| Create and send the "New bar" event                              |
//| to the control program chart                                     |
//+------------------------------------------------------------------+
void CSeriesDE::SendEvent(void)
  {
   int index=CSelect::FindBarMax(this.GetList(),BAR_PROP_TIME);
   CBar *bar=this.m_list_series.At(index);
   if(bar==NULL)
      return;
   ::EventChartCustom(this.m_chart_id_main,SERIES_EVENTS_NEW_BAR,bar.Time(),this.Timeframe(),this.Symbol());
  }
//+------------------------------------------------------------------+
```

Like in the Refresh() method, here we get the current bar object from the timeseries list and the bar time is passed to the **lparam** parameter when sending a [custom event](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom) to the control program chart.

**This completes the timeseries class. Now let's improve the class of all timeseries of a single symbol**.

As mentioned earlier, the CTimeSerirs class may cause a conflict with the same-name class of the standard library. Therefore, I have renamed it to CTimeSerirsDE. Inside the class listing, I have replaced all instances of the CTimeSerirs string with CTimeSerirsDE and CSerirs to CSerirsDE. Instead of delving into detailed description, consider the following brief example:

```
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "SeriesDE.mqh"
#include "..\Ticks\NewTickObj.mqh"
//+------------------------------------------------------------------+
//| Symbol timeseries class                                          |
//+------------------------------------------------------------------+
class CTimeSeriesDE : public CBaseObjExt
  {
private:
```

In the public section of the class, declare the method for copying the specified real property of the specified timeseries' bars to the passed array:

```
//--- Copy the specified double property of the specified timeseries to the array
//--- Regardless of the array indexing direction, copying is performed the same way as copying to a timeseries array
   bool              CopyToBufferAsSeries(const ENUM_TIMEFRAMES timeframe,
                                          const ENUM_BAR_PROP_DOUBLE property,
                                          double &array[],
                                          const double empty=EMPTY_VALUE);

//--- Compare CTimeSeriesDE objects (by symbol)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Display (1) description and (2) short symbol timeseries description in the journal
   void              Print(const bool created=true);
   void              PrintShort(const bool created=true);

//--- Constructors
                     CTimeSeriesDE(void){;}
                     CTimeSeriesDE(const string symbol);
  };
//+------------------------------------------------------------------+
```

We have considered this method above while improving the CSeriesDE class. Let's implement the method:

```
//+------------------------------------------------------------------+
//| Copy the specified double property of the specified timeseries   |
//+------------------------------------------------------------------+
bool CTimeSeriesDE::CopyToBufferAsSeries(const ENUM_TIMEFRAMES timeframe,
                                         const ENUM_BAR_PROP_DOUBLE property,
                                         double &array[],
                                         const double empty=EMPTY_VALUE)
  {
   CSeriesDE *series=this.GetSeries(timeframe);
   if(series==NULL)
      return false;
   return series.CopyToBufferAsSeries(property,array,empty);
  }
//+------------------------------------------------------------------+
```

Here all is simple: first, get the necessary timeseries by the specified timeframe, then return the method call result from the obtained timeseries object.

In the list of all symbol timeseries of the method returning the timeseries index, implement verification of the timeframe specified for searching:

```
//+------------------------------------------------------------------+
//| Return the timeframe index in the list                           |
//+------------------------------------------------------------------+
int CTimeSeriesDE::IndexTimeframe(const ENUM_TIMEFRAMES timeframe)
  {
   const CSeriesDE *obj=new CSeriesDE(this.m_symbol,(timeframe==PERIOD_CURRENT ? (ENUM_TIMEFRAMES)::Period() : timeframe));
   if(obj==NULL)
      return WRONG_VALUE;
   this.m_list_series.Sort();
   int index=this.m_list_series.Search(obj);
   delete obj;
   return index;
  }
//+------------------------------------------------------------------+
```

When creating a temporary object for a search, check the specified timeframe, and if it is CURRENT\_PERIOD, use the current timeframe for the search.

In the method of updating the specified timeseries list, use the new bar open time from the **data\_calculate** structure as the **lparam** parameter value when adding a new event to the event list:

```
//+------------------------------------------------------------------+
//| Update a specified timeseries list                               |
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
//--- If the timeseries object features the New bar event
   if(series_obj.IsNewBar(data_calculate.rates.time))
     {
      //--- send the "New bar" event to the control program chart
      series_obj.SendEvent();
      //--- set the values of the first date in history on the server and in the terminal
      this.SetTerminalServerDate();
      //--- add the "New bar" event to the list of timeseries events
      //--- in case of successful addition, set the event flag for the timeseries
      if(this.EventAdd(SERIES_EVENTS_NEW_BAR,series_obj.Time(data_calculate.rates.time),series_obj.Timeframe(),series_obj.Symbol()))
         this.m_is_event=true;
     }
  }
//+------------------------------------------------------------------+
```

**This completes the CTimeSeriesDE class.** Move to the CTimeSeriesCollection class of the collection object of objects of all timeseries of all symbols.

Currently, we have two renamed classes: CSeriesDE and CTimeSerirsDE. Inside the CTimeSeriesCollection class listing, replace all instances of the CTimeSerirs string to CTimeSerirsDE and CSerirs to CSerirsDE.

Instead of delving into detailed description, consider the following brief example:

```
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Objects\Series\TimeSeriesDE.mqh"
#include "..\Objects\Symbols\Symbol.mqh"
//+------------------------------------------------------------------+
//| Symbol timeseries collection                                     |
//+------------------------------------------------------------------+
class CTimeSeriesCollection : public CBaseObjExt
  {
private:
   CListObj                m_list;                    // List of applied symbol timeseries
//--- Return the timeseries index by symbol name
   int                     IndexTimeSeries(const string symbol);
public:
//--- Return (1) oneself and (2) the timeseries list
   CTimeSeriesCollection  *GetObject(void)            { return &this;         }
   CArrayObj              *GetList(void)              { return &this.m_list;  }
//--- Return (1) the timeseries object of the specified symbol and (2) the timeseries object of the specified symbol/period
   CTimeSeriesDE          *GetTimeseries(const string symbol);
   CSeriesDE              *GetSeries(const string symbol,const ENUM_TIMEFRAMES timeframe);

//--- Create the symbol timeseries list collection
```

In the public section of the class, declare three new methods:

the method returning the bar object of the specified timeseries of the specified symbol by bar open time and two methods returning the bar object of a single timeseries corresponding to the open time of the bar in another timeseries by bar index and bar time:

```
//--- Return the bar object of the specified timeseries of the specified symbol of the specified position (1) by index, (2) by time
//--- bar object of the first timeseries corresponding to the bar open time on the second timeseries (3) by index, (4) by time
   CBar                   *GetBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index,const bool from_series=true);
   CBar                   *GetBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime bar_time);
   CBar                   *GetBarSeriesFirstFromSeriesSecond(const string symbol_first,const ENUM_TIMEFRAMES timeframe_first,const int index,
                                                             const string symbol_second=NULL,const ENUM_TIMEFRAMES timeframe_second=PERIOD_CURRENT);
   CBar                   *GetBarSeriesFirstFromSeriesSecond(const string symbol_first,const ENUM_TIMEFRAMES timeframe_first,const datetime first_bar_time,
                                                             const string symbol_second=NULL,const ENUM_TIMEFRAMES timeframe_second=PERIOD_CURRENT);
```

Also, declare yet another two methods in the public section: the method for updating all timeseries of the specified symbol and the method copying the specified double property of the specified timeseries of the specified symbol to the array:

```
//--- Update (1) the specified timeseries of the specified symbol, (2) all timeseries of the specified symbol, (3) all timeseries of all symbols
   void                    Refresh(const string symbol,const ENUM_TIMEFRAMES timeframe,SDataCalculate &data_calculate);
   void                    Refresh(const string symbol,SDataCalculate &data_calculate);
   void                    Refresh(SDataCalculate &data_calculate);

//--- Get events from the timeseries object and add them to the list
   bool                    SetEvents(CTimeSeriesDE *timeseries);

//--- Display (1) the complete and (2) short collection description in the journal
   void                    Print(const bool created=true);
   void                    PrintShort(const bool created=true);

//--- Copy the specified double property of the specified timeseries of the specified symbol to the array
//--- Regardless of the array indexing direction, copying is performed the same way as copying to a timeseries array
   bool                    CopyToBufferAsSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                                const ENUM_BAR_PROP_DOUBLE property,
                                                double &array[],
                                                const double empty=EMPTY_VALUE);
//--- Constructor
                           CTimeSeriesCollection();
  };
//+------------------------------------------------------------------+
```

**Implementing the method returning the bar object of the specified timeseries of the specified symbol of the specified position by time:**

```
//+------------------------------------------------------------------+
//| Return the bar object of the specified timeseries                |
//| of the specified symbol of the specified position by time        |
//+------------------------------------------------------------------+
CBar *CTimeSeriesCollection::GetBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime bar_time)
  {
   CSeriesDE *series=this.GetSeries(symbol,timeframe);
   if(series==NULL)
      return NULL;
   return series.GetBar(bar_time);
  }
//+------------------------------------------------------------------+
```

The method passes symbol and timeframe of the timeseries, from which we should get a bar with the specified open time.

Get the timeseries object with the specified symbol and timeframe and return the bar object taken from the obtained timeseries by bar time.

If failed to get the bar, return NULL.

**Implementing the method returning the bar object of the first timeseries by index corresponding to the bar open time on the second timeseries:**

```
//+------------------------------------------------------------------+
//| Return the bar object of the first timeseries by index           |
//| corresponding to the bar open time on the second timeseries      |
//+------------------------------------------------------------------+
CBar *CTimeSeriesCollection::GetBarSeriesFirstFromSeriesSecond(const string symbol_first,const ENUM_TIMEFRAMES timeframe_first,const int index,
                                                               const string symbol_second=NULL,const ENUM_TIMEFRAMES timeframe_second=PERIOD_CURRENT)
  {
   CBar *bar_first=this.GetBar(symbol_first,timeframe_first,index);
   if(bar_first==NULL)
      return NULL;
   CBar *bar_second=this.GetBar(symbol_second,timeframe_second,bar_first.Time());
   return bar_second;
  }
//+------------------------------------------------------------------+
```

The method receives a symbol and timeframe of the first chart, bar index on the first chart, as well as a symbol and period of the second chart.

Get the first bar object from the timeseries of the first symbolperiod by the specified index, get and return the second bar object of the second symbolperiod by the time of the first obtained bar.

The method allows receiving a bar position specified by index on the specified first chart symbol period matching the bar position on the second specified chart period symbol by open time.

What's in it for us? As an example, we are able to quickly mark all Н1 bars on M15 chart.

Simply pass the current symbol, М15 chart period, bar position by its index on the chart (for example, indicator calculation loop index), the current symbol and Н1 period to the method. The method returns the bar object from the current symbol chart and Н1 period, whose open time includes the time of opening the first specified bar.

**Implementing the method returning the bar object of the first timeseries by time corresponding to the bar open time on the second timeseries:**

```
//+------------------------------------------------------------------+
//| Return the bar object of the first timeseries by time            |
//| corresponding to the bar open time on the second timeseries      |
//+------------------------------------------------------------------+
CBar *CTimeSeriesCollection::GetBarSeriesFirstFromSeriesSecond(const string symbol_first,const ENUM_TIMEFRAMES timeframe_first,const datetime first_bar_time,
                                                               const string symbol_second=NULL,const ENUM_TIMEFRAMES timeframe_second=PERIOD_CURRENT)
  {
   CBar *bar_first=this.GetBar(symbol_first,timeframe_first,first_bar_time);
   if(bar_first==NULL)
      return NULL;
   CBar *bar_second=this.GetBar(symbol_second,timeframe_second,bar_first.Time());
   return bar_second;
  }
//+------------------------------------------------------------------+
```

The method is similar to the method of receiving the bar object by index I have just described above. Here, instead of the bar index in the timeseries, the system sets the time of its opening in the specified first timeseries.

As you may have noticed, both methods receive the periods and symbols of both charts. This means that the methods are able to get back the bar object from any period symbol corresponding to the bar object of the first period symbol with its specified position in the timeseries. This allows us to easily match two bars from any period symbol to compare them by any of the bar object properties.

Add the check for a "non-native symbol" to the method of updating the specified timeseries of the specified symbol:

```
//+------------------------------------------------------------------+
//| Update the specified timeseries of the specified symbol          |
//+------------------------------------------------------------------+
void CTimeSeriesCollection::Refresh(const string symbol,const ENUM_TIMEFRAMES timeframe,SDataCalculate &data_calculate)
  {
//--- Reset the flag of an event in the timeseries collection and clear the event list
   this.m_is_event=false;
   this.m_list_events.Clear();
//--- Get the object of all symbol timeseries by a symbol name
   CTimeSeriesDE *timeseries=this.GetTimeseries(symbol);
   if(timeseries==NULL)
      return;
//--- If a symbol is non-native and there is no new tick on the timeseries object symbol, exit
   if(symbol!=::Symbol() && !timeseries.IsNewTick())
      return;
//--- Update the required object timeseries of all symbol timeseries
   timeseries.Refresh(timeframe,data_calculate);
//--- If the timeseries has the enabled event flag,
//--- get events from symbol timeseries, write them to the collection event list
//--- and set the event flag in the collection
   if(timeseries.IsEvent())
      this.m_is_event=this.SetEvents(timeseries);
  }
//+------------------------------------------------------------------+
```

Why do we need this? We update all the timeseries not belonging to the current period symbol in the library timer. Timeseries belonging to the symbol the program is launched on should be updated from the program's [Start, NewTick or Calculate](https://www.mql5.com/en/docs/runtime/event_fire) event handler. To avoid the new tick event for the current symbol in the timer (the current symbol timeseries is updated by tick anyway), we check if the timeseries symbol matches the current one and check the "new tick" timeseries event only if the timeseries does not belong to the current symbol.

**Implementing the method of updating all timeseries of the specified symbol:**

```
//+------------------------------------------------------------------+
//| Update all timeseries of the specified symbol                    |
//+------------------------------------------------------------------+
void CTimeSeriesCollection::Refresh(const string symbol,SDataCalculate &data_calculate)
  {
//--- Reset the flag of an event in the timeseries collection and clear the event list
   this.m_is_event=false;
   this.m_list_events.Clear();
//--- Get the object of all symbol timeseries by a symbol name
   CTimeSeriesDE *timeseries=this.GetTimeseries(symbol);
   if(timeseries==NULL)
      return;
//--- If a symbol is non-native and there is no new tick on the timeseries object symbol, exit
   if(symbol!=::Symbol() && !timeseries.IsNewTick())
      return;
//--- Update all object timeseries of all symbol timeseries
   timeseries.RefreshAll(data_calculate);
//--- If the timeseries has the enabled event flag,
//--- get events from symbol timeseries, write them to the collection event list
//--- and set the event flag in the collection
   if(timeseries.IsEvent())
      this.m_is_event=this.SetEvents(timeseries);
  }
//+------------------------------------------------------------------+
```

Each string of the method logic is described in the code comments, so I hope, all is clear here.

**Implementing the method writing the specified real bar data of the specified timeseries object to the array passed to the method:**

```
//+------------------------------------------------------------------+
//| Copy the specified double property to the array                  |
//| for a specified timeseries of a specified symbol                 |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::CopyToBufferAsSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                                 const ENUM_BAR_PROP_DOUBLE property,
                                                 double &array[],
                                                 const double empty=EMPTY_VALUE)
  {
   CSeriesDE *series=this.GetSeries(symbol,timeframe);
   if(series==NULL)
      return false;
   return series.CopyToBufferAsSeries(property,array,empty);
  }
//+------------------------------------------------------------------+
```

We have considered the method operation above while improving the CSeriesDE class.

Here we simply get the required timeseries object by the specified symbol and period, and return the result of calling the same-name method of the obtained timeseries.

**This completes the work on the timeseries collection class.**

Now we need to provide access to newly created methods from library-based programs. Such an access is provided by the CEngine library main object.

Open \\MQL5\\Include\\DoEasy\ **Engine.mqh** and replace all instances of the CSerirs string to CSerirsDE and CTimeSerirs to CTimeSerirsDE.

In the private class section, declare the class member variable for storing the program name:

```
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine
  {
private:
   CHistoryCollection   m_history;                       // Collection of historical orders and deals
   CMarketCollection    m_market;                        // Collection of market orders and deals
   CEventsCollection    m_events;                        // Event collection
   CAccountsCollection  m_accounts;                      // Account collection
   CSymbolsCollection   m_symbols;                       // Symbol collection
   CTimeSeriesCollection m_time_series;                  // Timeseries collection
   CResourceCollection  m_resource;                      // Resource list
   CTradingControl      m_trading;                       // Trading management object
   CPause               m_pause;                         // Pause object
   CArrayObj            m_list_counters;                 // List of timer counters
   int                  m_global_error;                  // Global error code
   bool                 m_first_start;                   // First launch flag
   bool                 m_is_hedge;                      // Hedge account flag
   bool                 m_is_tester;                     // Flag of working in the tester
   bool                 m_is_market_trade_event;         // Account trading event flag
   bool                 m_is_history_trade_event;        // Account history trading event flag
   bool                 m_is_account_event;              // Account change event flag
   bool                 m_is_symbol_event;               // Symbol change event flag
   ENUM_TRADE_EVENT     m_last_trade_event;              // Last account trading event
   int                  m_last_account_event;            // Last event in the account properties
   int                  m_last_symbol_event;             // Last event in the symbol properties
   ENUM_PROGRAM_TYPE    m_program;                       // Program type
   string               m_name;                          // Program name
```

In the class constructor, assign the program name to the variable:

```
//+------------------------------------------------------------------+
//| CEngine constructor                                              |
//+------------------------------------------------------------------+
CEngine::CEngine() : m_first_start(true),
                     m_last_trade_event(TRADE_EVENT_NO_EVENT),
                     m_last_account_event(WRONG_VALUE),
                     m_last_symbol_event(WRONG_VALUE),
                     m_global_error(ERR_SUCCESS)
  {
   this.m_is_hedge=#ifdef __MQL4__ true #else bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING) #endif;
   this.m_is_tester=::MQLInfoInteger(MQL_TESTER);
   this.m_program=(ENUM_PROGRAM_TYPE)::MQLInfoInteger(MQL_PROGRAM_TYPE);
   this.m_name=::MQLInfoString(MQL_PROGRAM_NAME);

...
```

In the public class section, add the method returning the bar object of the specified timeseries of the specified symbol of the specified position by bar time,

two methods returning the bar object of the first timeseries corresponding to the bar open time on the secod timeseries by index and time,

the method updating all timeseries of the specified symbol,

the methods returning the bar base properties by time,

the method for copying the specified double property of the specified timeseries of the specified symbol to the array and

the method returning the name of the library-based program.

```
//--- Return the bar object of the specified timeseries of the specified symbol of the specified position (1) by index, (2) by time
   CBar                *SeriesGetBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index,const bool from_series=true)
                          { return this.m_time_series.GetBar(symbol,timeframe,index,from_series);                 }
   CBar                *SeriesGetBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time)
                          { return this.m_time_series.GetBar(symbol,timeframe,time);                              }
//--- Return the bar object of the first timeseries corresponding to the bar open time on the second timeseries (1) by index, (2) by time
   CBar                *SeriesGetBarSeriesFirstFromSeriesSecond(const string symbol_first,const ENUM_TIMEFRAMES timeframe_first,const int index,
                                                                const string symbol_second=NULL,const ENUM_TIMEFRAMES timeframe_second=PERIOD_CURRENT)
                          { return this.m_time_series.GetBarSeriesFirstFromSeriesSecond(symbol_first,timeframe_first,index,symbol_second,timeframe_second); }

   CBar                *SeriesGetBarSeriesFirstFromSeriesSecond(const string symbol_first,const ENUM_TIMEFRAMES timeframe_first,const datetime time,
                                                                const string symbol_second=NULL,const ENUM_TIMEFRAMES timeframe_second=PERIOD_CURRENT)
                          { return this.m_time_series.GetBarSeriesFirstFromSeriesSecond(symbol_first,timeframe_first,time,symbol_second,timeframe_second); }

//--- Return the flag of opening a new bar of the specified timeseries of the specified symbol
   bool                 SeriesIsNewBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time=0)
                          { return this.m_time_series.IsNewBar(symbol,timeframe,time);                            }

//--- Update (1) the specified timeseries of the specified symbol, (2) all timeseries of the specified symbol, (3) all timeseries of all symbols
   void                 SeriesRefresh(const string symbol,const ENUM_TIMEFRAMES timeframe,SDataCalculate &data_calculate)
                          { this.m_time_series.Refresh(symbol,timeframe,data_calculate);                          }
   void                 SeriesRefresh(const string symbol,SDataCalculate &data_calculate)
                          { this.m_time_series.Refresh(symbol,data_calculate);                                    }
   void                 SeriesRefresh(SDataCalculate &data_calculate)
                          { this.m_time_series.Refresh(data_calculate);                                           }

//--- Return (1) the timeseries object of the specified symbol and (2) the timeseries object of the specified symbol/period
   CTimeSeriesDE       *SeriesGetTimeseries(const string symbol)
                          { return this.m_time_series.GetTimeseries(symbol);                                      }
   CSeriesDE           *SeriesGetSeries(const string symbol,const ENUM_TIMEFRAMES timeframe)
                          { return this.m_time_series.GetSeries(symbol,timeframe);                                }
//--- Return (1) an empty, (2) partially filled timeseries
   CSeriesDE           *SeriesGetSeriesEmpty(void)       { return this.m_time_series.GetSeriesEmpty();            }
   CSeriesDE           *SeriesGetSeriesIncompleted(void) { return this.m_time_series.GetSeriesIncompleted();      }

//--- Return (1) Open, (2) High, (3) Low, (4) Close, (5) Time, (6) TickVolume,
//--- (7) RealVolume, (8) Spread of the bar, specified by index, of the specified symbol of the specified timeframe
   double               SeriesOpen(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index);
   double               SeriesHigh(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index);
   double               SeriesLow(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index);
   double               SeriesClose(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index);
   datetime             SeriesTime(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index);
   long                 SeriesTickVolume(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index);
   long                 SeriesRealVolume(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index);
   int                  SeriesSpread(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index);

//--- Return (1) Open, (2) High, (3) Low, (4) Close, (5) Time, (6) TickVolume,
//--- (7) RealVolume, (8) Spread of the bar, specified by time, of the specified symbol of the specified timeframe
   double               SeriesOpen(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time);
   double               SeriesHigh(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time);
   double               SeriesLow(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time);
   double               SeriesClose(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time);
   datetime             SeriesTime(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time);
   long                 SeriesTickVolume(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time);
   long                 SeriesRealVolume(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time);
   int                  SeriesSpread(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time);

//--- Copy the specified double property of the specified timeseries of the specified symbol to the array
//--- Regardless of the array indexing direction, copying is performed the same way as copying to a timeseries array
   bool                 SeriesCopyToBufferAsSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,const ENUM_BAR_PROP_DOUBLE property,
                                                   double &array[],const double empty=EMPTY_VALUE)
                          { return this.m_time_series.CopyToBufferAsSeries(symbol,timeframe,property,array,empty);}
```

...

```
//--- Return the program name
   string               Name(void)                                const { return this.m_name;                                 }
```

All methods whose implementation is set in the class body return the result of calling same-name methods of the collection of the **TimeSeriesCollection** timeseries considered above.

**Implementing the methods returning bar base properties by time:**

```
//+------------------------------------------------------------------+
//| Return Open of the specified bar by time                         |
//| of the specified symbol of the specified timeframe               |
//+------------------------------------------------------------------+
double CEngine::SeriesOpen(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time)
  {
   CBar *bar=this.m_time_series.GetBar(symbol,timeframe,time);
   return(bar!=NULL ? bar.Open() : 0);
  }
//+------------------------------------------------------------------+
//| Return High of the specified bar by time                         |
//| of the specified symbol of the specified timeframe               |
//+------------------------------------------------------------------+
double CEngine::SeriesHigh(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time)
  {
   CBar *bar=this.m_time_series.GetBar(symbol,timeframe,time);
   return(bar!=NULL ? bar.High() : 0);
  }
//+------------------------------------------------------------------+
//| Return Low of the specified bar by time                          |
//| of the specified symbol of the specified timeframe               |
//+------------------------------------------------------------------+
double CEngine::SeriesLow(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time)
  {
   CBar *bar=this.m_time_series.GetBar(symbol,timeframe,time);
   return(bar!=NULL ? bar.Low() : 0);
  }
//+------------------------------------------------------------------+
//| Return Close of the specified bar by time                        |
//| of the specified symbol of the specified timeframe               |
//+------------------------------------------------------------------+
double CEngine::SeriesClose(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time)
  {
   CBar *bar=this.m_time_series.GetBar(symbol,timeframe,time);
   return(bar!=NULL ? bar.Close() : 0);
  }
//+------------------------------------------------------------------+
//| Return Time of the specified bar by time                         |
//| of the specified symbol of the specified timeframe               |
//+------------------------------------------------------------------+
datetime CEngine::SeriesTime(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time)
  {
   CBar *bar=this.m_time_series.GetBar(symbol,timeframe,time);
   return(bar!=NULL ? bar.Time() : 0);
  }
//+------------------------------------------------------------------+
//| Return TickVolume of the specified bar by time                   |
//| of the specified symbol of the specified timeframe               |
//+------------------------------------------------------------------+
long CEngine::SeriesTickVolume(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time)
  {
   CBar *bar=this.m_time_series.GetBar(symbol,timeframe,time);
   return(bar!=NULL ? bar.VolumeTick() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//| Return RealVolume of the specified bar by time                   |
//| of the specified symbol of the specified timeframe               |
//+------------------------------------------------------------------+
long CEngine::SeriesRealVolume(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time)
  {
   CBar *bar=this.m_time_series.GetBar(symbol,timeframe,time);
   return(bar!=NULL ? bar.VolumeReal() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//| Return Spread of the specified bar by time                       |
//| of the specified symbol of the specified timeframe               |
//+------------------------------------------------------------------+
int CEngine::SeriesSpread(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time)
  {
   CBar *bar=this.m_time_series.GetBar(symbol,timeframe,time);
   return(bar!=NULL ? bar.Spread() : INT_MIN);
  }
//+------------------------------------------------------------------+
```

Here, all is simple:

get the bar object from the timeseries collection class using the GetBar() method specifying the timeseries symbol and period and the time of opening the requested bar in the timeseries, and return the value of the appropriate property of the obtained bar considering the error of receiving bar from the timeseries.

Add the update of all timeseries of the current symbol to the NewTick event handler of the current symbol:

```
//+------------------------------------------------------------------+
//| NewTick event handler                                            |
//+------------------------------------------------------------------+
void CEngine::OnTick(SDataCalculate &data_calculate,const uint required=0)
  {
//--- If this is not a EA, exit
   if(this.m_program!=PROGRAM_EXPERT)
      return;
//--- Re-create empty timeseries and update the current symbol timeseries
   this.SeriesSync(data_calculate,required);
   this.SeriesRefresh(NULL,data_calculate);
//--- end
  }
//+------------------------------------------------------------------+
```

This allows updating all applied timeseries of the current symbol in EAs immediately after synchronization attempt, so that we do not have to wait for the current symbol's timeseries update in the library timer since this sometimes causes data desynchronization when data update in the timer is called after a new tick arrives on the current symbol.

Add the update of all timeseries of the current symbol after symchronizing all timeseries to the Calculate event handler of the current symbol:

```
//+------------------------------------------------------------------+
//| Calculate event handler                                          |
//+------------------------------------------------------------------+
int CEngine::OnCalculate(SDataCalculate &data_calculate,const uint required=0)
  {
//--- If this is not an indicator, exit
   if(this.m_program!=PROGRAM_INDICATOR)
      return data_calculate.rates_total;
//--- Re-create empty timeseries
//--- If at least one of the timeseries is not synchronized, return zero
   if(!this.SeriesSync(data_calculate,required))
     {
      return 0;
     }
//--- Update the timeseries of the current symbol and return rates_total
   this.SeriesRefresh(NULL,data_calculate);
   return data_calculate.rates_total;
  }
//+------------------------------------------------------------------+
```

Here are the differences from the OnTick() handler — the method returns zero till all applied timeseries of the current symbol are synchronized, which, in turn, informs the OnCalculate() handler of the indicator about the necessity to fully re-calculate historical data.

Accordingly, the method of synchronizing data of all timeseries should now return boolean values:

```
//+------------------------------------------------------------------+
//| Synchronize timeseries data with the server                      |
//+------------------------------------------------------------------+
bool CEngine::SeriesSync(SDataCalculate &data_calculate,const uint required=0)
  {
//--- If the timeseries data is not calculated, try re-creating the timeseries
//--- Get the pointer to the empty timeseries
   CSeriesDE *series=this.SeriesGetSeriesEmpty();
//--- If there is an empty timeseries
   if(series!=NULL)
     {
      //--- Display the empty timeseries data as a chart comment and try synchronizing the timeseries with the server data
      ::Comment(series.Header(),": ",CMessage::Text(MSG_LIB_TEXT_TS_TEXT_WAIT_FOR_SYNC));
      ::ChartRedraw(::ChartID());
      //--- if the data has been synchronized
      if(series.SyncData(required,data_calculate.rates_total))
        {
         //--- if managed to re-create the timeseries
         if(this.m_time_series.ReCreateSeries(series.Symbol(),series.Timeframe(),data_calculate.rates_total))
           {
            //--- display the chart comment and the journal entry with the re-created timeseries data
            ::Comment(series.Header(),": OK");
            ::ChartRedraw(::ChartID());
            Print(series.Header()," ",CMessage::Text(MSG_LIB_TEXT_TS_TEXT_CREATED_OK),":");
            series.PrintShort();
            return true;
           }
        }
      //--- Data is not yet synchronized or failed to re-create the timeseries
      return false;
     }
//--- There are no empty timeseries - all is synchronized, delete all comments
   else
     {
      ::Comment("");
      ::ChartRedraw(::ChartID());
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

**This completes the CEngine class for now.**

Now let's check how all this works in the indicators. Since we use several different timeseries in a single indicator and we are able to obtain single bar data corresponding to data of another bar with the time falling within the boundaries of the first bar from other timeseries, the first thing that comes to mind is creating an indicator displaying OHLC lines of bars from other timeframes on the current chart.

### Creating and testing a multi-period indicator

To perform the test, let's use the indicator we have developed [in the previous article](https://www.mql5.com/en/articles/7724#node03) and save it in \\MQL5\\Indicators\\TestDoEasy\ **Part40\** as **TestDoEasyPart40.mq5**.

We may use 21 timeseries by the number of standard available chart periods. The settings feature the standard set of used timeframes, while the chart is to display the buttons corresponding to the used timeframes selected in the settings. To avoid excessive code meant for the indicator buffers, simply assign the buffers to each chart period present in the terminal using the structure array.

The visibility of buffer lines on the chart and its data in the indicator data window is enabled/disabled by enabling/disabling the appropriate button. Two buffers (drawn and calculated) are to be assigned to each timeframe. The calculated buffer allows storing intermediate data of the corresponding timeseries. However, in the current implementation, the calculated buffer is not used. To avoid writing all 42 buffers (21 drawn and 21 calculated ones), we will create the structure which is to store the parameters for each of the timeframes:

- The array assigned by the drawn indicator buffer
- The array assigned by the calculated indicator buffer
- Buffer ID (timeframe of the timeseries whose data is to be displayed by the buffer)
- The index of the indicator buffer related to the drawn buffer array
- The index of the indicator buffer related to the calculated buffer array
- The flag of using the buffer in the indicator (button pressed/not pressed)
- The flag of displaying the buffer in the indicator before enabling/disabling the buffer display by the chart button


The indicator settings allow you to decide on whether each of the timeframes should be used and, accordingly, which of the timeseries is selected. The chart buttons plotted according to the selected timeseries allow enabling/disabling the display of corresponding indicator buffers on the chart. The flag of displaying the buffer in the indicator till its display is enabled/disabled by the button allows us to decide on removing or displaying the buffer data on the chart only when the appropriate button is pressed.

Set all parameters of each indicator buffer (we could have set it programmatically, but the current method is faster):

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart40.mq5 |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
//--- properties
#property indicator_chart_window
#property indicator_buffers 43
#property indicator_plots   21
//--- plot M1
#property indicator_label1  " M1"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrGray
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1
//--- plot M2
#property indicator_label2  " M2"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrGray
#property indicator_style2  STYLE_SOLID
#property indicator_width2  1
//--- plot M3
#property indicator_label3  " M3"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrGray
#property indicator_style3  STYLE_SOLID
#property indicator_width3  1
//--- plot M4
#property indicator_label4  " M4"
#property indicator_type4  DRAW_LINE
#property indicator_color4  clrGray
#property indicator_style4  STYLE_SOLID
#property indicator_width4  1
//--- plot M5
#property indicator_label5  " M5"
#property indicator_type5   DRAW_LINE
#property indicator_color5  clrGray
#property indicator_style5  STYLE_SOLID
#property indicator_width5  1
//--- plot M6
#property indicator_label6  " M6"
#property indicator_type6   DRAW_LINE
#property indicator_color6  clrGray
#property indicator_style6  STYLE_SOLID
#property indicator_width6  1
//--- plot M10
#property indicator_label7  " M10"
#property indicator_type7   DRAW_LINE
#property indicator_color7  clrGray
#property indicator_style7  STYLE_SOLID
#property indicator_width7  1
//--- plot M12
#property indicator_label8  " M12"
#property indicator_type8   DRAW_LINE
#property indicator_color8  clrGray
#property indicator_style8  STYLE_SOLID
#property indicator_width8  1
//--- plot M15
#property indicator_label9  " M15"
#property indicator_type9   DRAW_LINE
#property indicator_color9  clrGray
#property indicator_style9  STYLE_SOLID
#property indicator_width9  1
//--- plot M20
#property indicator_label10 " M20"
#property indicator_type10  DRAW_LINE
#property indicator_color10 clrGray
#property indicator_style10 STYLE_SOLID
#property indicator_width10 1
//--- plot M30
#property indicator_label11 " M30"
#property indicator_type11  DRAW_LINE
#property indicator_color11 clrGray
#property indicator_style11 STYLE_SOLID
#property indicator_width11 1
//--- plot H1
#property indicator_label12 " H1"
#property indicator_type12  DRAW_LINE
#property indicator_color12 clrGray
#property indicator_style12 STYLE_SOLID
#property indicator_width12 1
//--- plot H2
#property indicator_label13 " H2"
#property indicator_type13  DRAW_LINE
#property indicator_color13 clrGray
#property indicator_style13 STYLE_SOLID
#property indicator_width13 1
//--- plot H3
#property indicator_label14 " H3"
#property indicator_type14  DRAW_LINE
#property indicator_color14 clrGray
#property indicator_style14 STYLE_SOLID
#property indicator_width14 1
//--- plot H4
#property indicator_label15 " H4"
#property indicator_type15  DRAW_LINE
#property indicator_color15 clrGray
#property indicator_style15 STYLE_SOLID
#property indicator_width15 1
//--- plot H6
#property indicator_label16 " H6"
#property indicator_type16  DRAW_LINE
#property indicator_color16 clrGray
#property indicator_style16 STYLE_SOLID
#property indicator_width16 1
//--- plot H8
#property indicator_label17 " H8"
#property indicator_type17  DRAW_LINE
#property indicator_color17 clrGray
#property indicator_style17 STYLE_SOLID
#property indicator_width17 1
//--- plot H12
#property indicator_label18 " H12"
#property indicator_type18  DRAW_LINE
#property indicator_color18 clrGray
#property indicator_style18 STYLE_SOLID
#property indicator_width18 1
//--- plot D1
#property indicator_label19 " D1"
#property indicator_type19  DRAW_LINE
#property indicator_color19 clrGray
#property indicator_style19 STYLE_SOLID
#property indicator_width19 1
//--- plot W1
#property indicator_label20 " W1"
#property indicator_type20  DRAW_LINE
#property indicator_color20 clrGray
#property indicator_style20 STYLE_SOLID
#property indicator_width20 1
//--- plot MN1
#property indicator_label21 " MN1"
#property indicator_type21  DRAW_LINE
#property indicator_color21 clrGray
#property indicator_style21 STYLE_SOLID
#property indicator_width21 1

//--- classes
```

As we can see, the total amount of buffers is set to 43, while the amount of drawn buffers is set to 21. Since I decided to add one calculated buffer to each of the drawn buffers, the result is 21+21=42. Where does one extra buffer come from? We need it to store data on time from the time\[\] OnCalculate() array. Since some functions require bar time by index, while the time\[\] array exists only within the OnCalculate() handler visibility scope, the simplest solution for having time data for each bar of the current timeframe is to save the time\[\] array in one of the indicator calculated buffers. This is why I have set one more buffer.

The indicator provides the ability to display four bar prices: Open, High, Low and Close. The bar object has more real properties:

- Bar open price (Open)

- Highest price within the bar period (High)

- Lowest price within the bar period (Low)

- Bar close price (Close)

- Candle size
- Candle body size
- Candle body top
- Candle body bottom
- Candle upper wick size
- Candle lower wick size

Therefore, we cannot use the value of the enumeration (ENUM\_BAR\_PROP\_DOUBLE) in the settings. Let's create another enumeration featuring the necessary properties matching the enumeration properties of real properties of the ENUM\_BAR\_PROP\_DOUBLE bar object that can be selected in the settings for display and set macro substitution with the total amount of available chart periods:

```
//--- classes

//--- enums
enum ENUM_BAR_PRICE
  {
   BAR_PRICE_OPEN    =  BAR_PROP_OPEN,    // Bar Open
   BAR_PRICE_HIGH    =  BAR_PROP_HIGH,    // Bar High
   BAR_PRICE_LOW     =  BAR_PROP_LOW,     // Bar Low
   BAR_PRICE_CLOSE   =  BAR_PROP_CLOSE,   // Bar Close
  };
//--- defines
#define PERIODS_TOTAL   (21)              // Total amount of available chart periods
//--- structures
```

Now let's create the data structure of one drawn and one calculated buffers assigned for a single timeseries (chart period):

```
//--- structures
struct SDataBuffer
  {
private:
   int               m_buff_id;           // Buffer ID (timeframe)
   int               m_buff_data_index;   // The index of the indicator buffer related to the Data[] array
   int               m_buff_tmp_index;    // The index of the indicator buffer related to the Temp[] array
   bool              m_used;              // The flag of using the buffer in the indicator
   bool              m_show_data;         // The flag of displaying the buffer on the chart before enabling/disabling its display
public:
   double            Data[];              // The array assigned as INDICATOR_DATA by the indicator buffer
   double            Temp[];              // The array assigned as INDICATOR_CALCULATIONS by the indicator buffer
//--- Set indices for the drawn and calculated buffers assigned to the timeframe
   void              SetIndex(const int index)
                       {
                        this.m_buff_data_index=index;
                        this.m_buff_tmp_index=index+PERIODS_TOTAL;
                       }
//--- Methods of setting and returning values of the private structure members
   void              SetID(const int id)              { this.m_buff_id=id;             }
   void              SetUsed(const bool flag)         { this.m_used=flag;              }
   void              SetShowData(const bool flag)     { this.m_show_data=flag;         }
   int               IndexDataBuffer(void)      const { return this.m_buff_data_index; }
   int               IndexTempBuffer(void)      const { return this.m_buff_tmp_index;  }
   int               ID(void)                   const { return this.m_buff_id;         }
   bool              IsUsed(void)               const { return this.m_used;            }
   bool              GetShowDataFlag(void)      const { return this.m_show_data;       }
   void              Print(void);
  };
//--- Display structure data to the journal
void SDataBuffer::Print(void)
  {
   ::Print
     (
      "Buffer[",this.IndexDataBuffer(),"], ID: ",(string)this.ID(),
      " (",TimeframeDescription((ENUM_TIMEFRAMES)this.ID()),
      "), temp buffer index: ",(string)this.IndexTempBuffer(),
      ", used: ",this.IsUsed()
     );
  }
//--- input variables
```

The structure is to store all the data for working with a single timeframe. A separate structure is to be assigned to each of the used indicator timeframes. The array of the appropriate structures is the most optimal solution for that. Let's create it in the block for defining the indicator buffers.

Write the indicator inputs:

```
//--- input variables
/*sinput*/ENUM_SYMBOLS_MODE   InpModeUsedSymbols=  SYMBOLS_MODE_CURRENT;            // Mode of used symbols list
/*sinput*/string              InpUsedSymbols    =  "EURUSD,AUDUSD,EURAUD,EURCAD,EURGBP,EURJPY,EURUSD,GBPUSD,NZDUSD,USDCAD,USDJPY";  // List of used symbols (comma - separator)
sinput   ENUM_TIMEFRAMES_MODE InpModeUsedTFs    =  TIMEFRAMES_MODE_LIST;            // Mode of used timeframes list
sinput   string               InpUsedTFs        =  "M1,M5,M15,M30,H1,H4,D1,W1,MN1"; // List of used timeframes (comma - separator)
sinput   ENUM_BAR_PRICE       InpBarPrice       =  BAR_PRICE_OPEN;                  // Applied bar price
sinput   bool                 InpShowBarTimes   =  false;                           // Show bar time comments
sinput   uint                 InpControlBar     =  1;                               // Control bar
sinput   uint                 InpButtShiftX     =  0;    // Buttons X shift
sinput   uint                 InpButtShiftY     =  10;   // Buttons Y shift
sinput   bool                 InpUseSounds      =  true; // Use sounds
//--- indicator buffers
```

Here all is similar to test EAs and indicators I provide for each article. Since I am going to test working with a single symbol, comment out the sinput modifiers in the symbol settings indicating that the variable is an indicator input (sinput modifier indicates that parameter optimization is disabled for the variable). Thus, these parameters cannot be selected in the settings, while the SYMBOLS\_MODE\_CURRENT value is assigned to the InpModeUsedSymbols variable — working with the current symbol only.

The **InpShowBarTimes** **variable** allows displaying/hiding comments on the chart — displaying the bar on the current chart period matching the bar with the same time on the charts of tested timeseries. The **InpControlBar** **variable** is used to specify the index of the bar whose value can be tracked via the chart comments.

Finally, write the indicator buffers and global variables:

```
//--- indicator buffers
SDataBuffer    Buffers[PERIODS_TOTAL];          // Array of the indicator buffer data structures assigned to the timeseries
double         BufferTime[];                    // The calculated buffer for storing and passing data from the time[] array
//--- global variables
CEngine        engine;                          // CEngine library main object
string         prefix;                          // Prefix of graphical object names
bool           testing;                         // Flag of working in the tester
int            used_symbols_mode;               // Mode of working with symbols
string         array_used_symbols[];            // Array of used symbols
string         array_used_periods[];            // Array of used timeframes
//+------------------------------------------------------------------+
```

As you can see, I have set the structure array described above as the indicator buffers definition. When initializing an indicator, we will assign data to the structure arrays and bind the structure arrays to the indicator buffers. The calculated buffer is defined here for storing and passing the time to the indicator functions.

The indicator global variables are commented on and, I believe, are quite comprehensible.

In the indicator's OnInit() handler, first create the panel with the buttons corresponding to the timeframes selected in the settings. Then assign all indicator buffers and set all indicator buffer parameters to structures located in the array of the indicator buffer structures:

```
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Set indicator global variables
   prefix=engine.Name()+"_";
   testing=engine.IsTester();
   ZeroMemory(rates_data);

//--- Initialize DoEasy library
   OnInitDoEasy();

//--- Check and remove remaining indicator graphical objects
   if(IsPresentObectByPrefix(prefix))
      ObjectsDeleteAll(0,prefix);

//--- Create the button panel
   if(!CreateButtons(InpButtShiftX,InpButtShiftY))
      return INIT_FAILED;

//--- Check playing a standard sound using macro substitutions
   engine.PlaySoundByDescription(SND_OK);
//--- Wait for 600 milliseconds
   engine.Pause(600);
   engine.PlaySoundByDescription(SND_NEWS);

//--- indicator buffers mapping

   //--- In the loop by the total number of available timeframes,
   for(int i=0;i<PERIODS_TOTAL;i++)
     {
      //--- get the next timeframe
      ENUM_TIMEFRAMES timeframe=TimeframeByEnumIndex(uchar(i+1));
      //--- Bind the drawn indicator buffer by the buffer index equal to the loop index with the structure Data[] array
      SetIndexBuffer(i,Buffers[i].Data);
      //--- set "the empty value" for the Data[] buffer,
      //--- set the name of the graphical series displayed in the data window for the Data[] buffer
      //--- set the direction of indexing the Data[] drawn buffer as in the timeseries
      PlotIndexSetDouble(i,PLOT_EMPTY_VALUE,EMPTY_VALUE);
      PlotIndexSetString(i,PLOT_LABEL,"Buffer "+TimeframeDescription(timeframe));
      ArraySetAsSeries(Buffers[i].Data,true);
      //--- Setting the drawn buffer according to the button status
      bool state=false;
      //--- Set the name of the button correspondign to the buffer with the loop index and its timeframe
      string name=prefix+"BUTT_"+TimeframeDescription(timeframe);
      //--- If not in the tester, while the chart features the button with the specified name,
      if(!engine.IsTester() && ObjectFind(ChartID(),name)==0)
        {
         //--- set the name of the terminal global variable for storing the button status
         string name_gv=(string)ChartID()+"_"+name;
         //--- if no global variable with such a name is found, create it set to 'false',
         if(!GlobalVariableCheck(name_gv))
            GlobalVariableSet(name_gv,false);
         //--- get the button status from the terminal global variable
         state=GlobalVariableGet(name_gv);
        }
      //--- Set the values for all structure fields
      Buffers[i].SetID(timeframe);
      Buffers[i].SetIndex(i);
      Buffers[i].SetUsed(state);
      Buffers[i].SetShowData(state);
      //--- Set the button status
      ButtonState(name,state);
      //--- Depending on the button status, specify whether the buffer data should be displayed should be displayed in the data window
      PlotIndexSetInteger(i,PLOT_SHOW_DATA,state);
      //--- Bind the calculated indicator buffer by the buffer index from IndexTempBuffer() with the Temp[] array of the structure
      SetIndexBuffer(Buffers[i].IndexTempBuffer(),Buffers[i].Temp,INDICATOR_CALCULATIONS);
      //--- set the direction of indexing the Temp[] calculated buffer as in the timeseries
      ArraySetAsSeries(Buffers[i].Temp,true);
     }
   //--- Bind the calculated indicator buffer by the PERIODS_TOTAL*2 buffer index with the BufferTime[] array of the indicator
   SetIndexBuffer(PERIODS_TOTAL*2,BufferTime,INDICATOR_CALCULATIONS);
   //--- set the direction of indexing the BufferTime[] calculated buffer as in the timeseries
   ArraySetAsSeries(BufferTime,true);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

Here I have commented on all strings of the loop where the indicator buffers are bound to the loop index by the structure array and the remaining parameters are set for each structure stored in each structure array cell. If you have any questions, feel free to ask them in the comments:

**The button functions:**

```
//+------------------------------------------------------------------+
//| Create the buttons panel                                         |
//+------------------------------------------------------------------+
bool CreateButtons(const int shift_x=20,const int shift_y=0)
  {
   int total=ArraySize(array_used_periods);
   uint w=30,h=20,x=InpButtShiftX+1, y=InpButtShiftY+h+1;
   //--- In the loop by the amount of used timeframes
   for(int i=0;i<total;i++)
     {
      //--- create the name of the next button
      string butt_name=prefix+"BUTT_"+array_used_periods[i];
      //--- create a new button with the offset by ((button width + 1) * loop index)
      if(!ButtonCreate(butt_name,x+(w+1)*i,y,w,h,array_used_periods[i],clrGray))
        {
         Alert(TextByLanguage("Не удалось создать кнопку \"","Could not create button \""),array_used_periods[i]);
         return false;
        }
     }
   ChartRedraw(0);
   return true;
  }
//+------------------------------------------------------------------+
//| Create the button                                                |
//+------------------------------------------------------------------+
bool ButtonCreate(const string name,const int x,const int y,const int w,const int h,const string text,const color clr,const string font="Calibri",const int font_size=8)
  {
   if(ObjectFind(0,name)<0)
     {
      if(!ObjectCreate(0,name,OBJ_BUTTON,0,0,0))
        {
         Print(DFUN,TextByLanguage("не удалось создать кнопку! Код ошибки=","Could not create button! Error code="),GetLastError());
         return false;
        }
      ObjectSetInteger(0,name,OBJPROP_SELECTABLE,false);
      ObjectSetInteger(0,name,OBJPROP_HIDDEN,true);
      ObjectSetInteger(0,name,OBJPROP_XDISTANCE,x);
      ObjectSetInteger(0,name,OBJPROP_YDISTANCE,y);
      ObjectSetInteger(0,name,OBJPROP_XSIZE,w);
      ObjectSetInteger(0,name,OBJPROP_YSIZE,h);
      ObjectSetInteger(0,name,OBJPROP_CORNER,CORNER_LEFT_LOWER);
      ObjectSetInteger(0,name,OBJPROP_ANCHOR,ANCHOR_LEFT_LOWER);
      ObjectSetInteger(0,name,OBJPROP_FONTSIZE,font_size);
      ObjectSetString(0,name,OBJPROP_FONT,font);
      ObjectSetString(0,name,OBJPROP_TEXT,text);
      ObjectSetInteger(0,name,OBJPROP_COLOR,clr);
      ObjectSetString(0,name,OBJPROP_TOOLTIP,"\n");
      ObjectSetInteger(0,name,OBJPROP_BORDER_COLOR,clrGray);
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
//| Set the terminal's global variable value                         |
//+------------------------------------------------------------------+
bool SetGlobalVariable(const string gv_name,const double value)
  {
//--- If the variable name length exceeds 63 symbols, return 'false'
   if(StringLen(gv_name)>63)
      return false;
   return(GlobalVariableSet(gv_name,value)>0);
  }
//+------------------------------------------------------------------+
//| Return the button status                                         |
//+------------------------------------------------------------------+
bool ButtonState(const string name)
  {
   return (bool)ObjectGetInteger(0,name,OBJPROP_STATE);
  }
//+------------------------------------------------------------------+
//| Return the button status by the timeframe name                   |
//+------------------------------------------------------------------+
bool ButtonState(const ENUM_TIMEFRAMES timeframe)
  {
   string name=prefix+"BUTT_"+TimeframeDescription(timeframe);
   return ButtonState(name);
  }
//+------------------------------------------------------------------+
//| Set the button status                                            |
//+------------------------------------------------------------------+
void ButtonState(const string name,const bool state)
  {
   ObjectSetInteger(0,name,OBJPROP_STATE,state);
   if(state)
      ObjectSetInteger(0,name,OBJPROP_BGCOLOR,C'220,255,240');
   else
      ObjectSetInteger(0,name,OBJPROP_BGCOLOR,C'240,240,240');
  }
//+------------------------------------------------------------------+
//| Track the buttons' status                                        |
//+------------------------------------------------------------------+
void PressButtonsControl(void)
  {
   int total=ObjectsTotal(0,0);
   for(int i=0;i<total;i++)
     {
      string obj_name=ObjectName(0,i);
      if(StringFind(obj_name,prefix+"BUTT_")<0)
         continue;
      PressButtonEvents(obj_name);
     }
  }
//+------------------------------------------------------------------+
//| Handle pressing the buttons                                      |
//+------------------------------------------------------------------+
void PressButtonEvents(const string button_name)
  {
//--- Convert button name into its string ID
   string button=StringSubstr(button_name,StringLen(prefix));
//--- Create the button name for the terminal's global variable
   string name_gv=(string)ChartID()+"_"+prefix+button;
//--- Get the button status (pressed/released). If not in the tester,
//--- write the status to the button global variable (1 or 0)
   bool state=ButtonState(button_name);
   if(!engine.IsTester())
      SetGlobalVariable(name_gv,state);
//--- Get the timeframe from the button string ID and
//--- the drawn buffer index by timeframe
   ENUM_TIMEFRAMES timeframe=TimeframeByDescription(StringSubstr(button,5));
   int buffer_index=IndexBuffer(timeframe);
//--- Set the button color depending on its status,
//--- write its status to the buffer structure depending on the button status (used/not used)
//--- initialize the buffer corresponding to the button timeframe by the buffer index received earlier
   ButtonState(button_name,state);
   Buffers[buffer_index].SetUsed(state);
   if(Buffers[buffer_index].GetShowDataFlag()!=state)
     {
      InitBuffer(buffer_index);
      BufferFill(buffer_index);
      Buffers[buffer_index].SetShowData(state);
     }

//--- Here you can add additional handling of button pressing:
//--- If the button is pressed
   if(state)
     {
      //--- If M1 button is pressed
      if(button=="BUTT_M1")
        {

        }
      //--- If button M2 is pressed
      else if(button=="BUTT_M2")
        {

        }
      //---
      // Remaining buttons ...
      //---
     }
   //--- Not pressed
   else
     {
      //--- M1 button
      if(button=="BUTT_M1")
        {

        }
      //--- M2 button
      if(button=="BUTT_M2")
        {

        }
      //---
      // Remaining buttons ...
      //---
     }
//--- re-draw the chart
   ChartRedraw();
  }
//+------------------------------------------------------------------+
```

All these functions are quite simple and straightforward, besides, some of their strings are commented on.

**Let's have a look at the indicator's OnCalculate() handler:**

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
   CopyData(rates_data,rates_total,prev_calculated,time,open,high,low,close,tick_volume,volume,spread);

//--- Handle the Calculate event in the library
   engine.OnCalculate(rates_data);

//--- If working in the tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer(rates_data);   // Working in the timer
      PressButtonsControl();        // Button pressing control
      EventsHandling();             // Working with events
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

//--- Setting buffer arrays as timeseries

//--- Check for the minimum number of bars for calculation
   if(rates_total<2 || Point()==0) return 0;

//--- Display reference data on bar open time
   if(InpShowBarTimes)
     {
      string txt="";
      int total=ArraySize(array_used_periods);
      //--- In the loop by the amount of used timeframes
      for(int i=0;i<total;i++)
        {
         //--- get the next timeframe, buffer index and timeseries object by timeframe
         ENUM_TIMEFRAMES timeframe=TimeframeByDescription(array_used_periods[i]);
         int buffer_index=IndexBuffer(timeframe);
         CSeriesDE *series=engine.SeriesGetSeries(NULL,timeframe);
         //--- If failed to get the timeseries or the buffer is not used (the button is released), move on to the next one
         if(series==NULL || !Buffers[buffer_index].IsUsed())
            continue;
         //--- Get the reference bar from the timeseries list
         CBar *bar=series.GetBar(InpControlBar);
         if(bar==NULL)
            continue;
         //--- Collect data for the comment text
         string t1=TimeframeDescription((ENUM_TIMEFRAMES)Period());
         string t2=TimeframeDescription(bar.Timeframe());
         string t3=(string)InpControlBar;
         string t4=TimeToString(bar.Time());
         string t5=(string)bar.Index((ENUM_TIMEFRAMES)Period());
         //--- Set the comment text depending on the terminal language
         string tn=TextByLanguage
           (
            "Бар на "+t1+", соответствующий бару "+t2+"["+t3+"] со временеи открытия "+t4+", расположен на баре "+t5,
            "The bar on "+t1+", corresponding to the "+t2+"["+t3+"] bar since the opening time of "+t4+", is located on bar "+t5
           );
         txt+=tn+"\n";
        }
      //--- Display the comment on the chart
      Comment(txt);
     }

//--- Check and calculate the number of calculated bars
   int limit=rates_total-prev_calculated;

//--- Recalculate the entire history
   if(limit>1)
     {
      limit=rates_total-1;
      InitBuffersAll();
     }
//--- Prepare data

//--- Calculate the indicator
   for(int i=limit; i>WRONG_VALUE && !IsStopped(); i--)
     {
      BufferTime[i]=(double)time[i];
      CalculateSeries((ENUM_BAR_PROP_DOUBLE)InpBarPrice,i,time[i]);
     }
//--- return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

If the "Show bar time comments" parameter (InpShowBarTimes variable) is set to true, the code block displays data on the current chart's bar specified in the InpControlBar ("ControlBar") variable indicating that it matches the bar on timeframes of all used timeseries.

If the calculated **limit** value exceeds one (which means the need to re-draw the entire history due to changes in the history), set **limit** equal to the start of history on the current chart and call the function of initializing all indicator buffers.

The indicator is calculated from the **limit** value (in normal conditions, it is equal to 1 (new bar) or zero — calculate the current bar) to zero.

In the main indicator calculation loop, fill in the calculated time buffer from the time\[\] array (we need the time buffer for other indicator functions obtaining time by index where the time\[\] array is unavailable) and call the function of calculating a single bar for all used indicator buffers.

**The function of initializing the indicator buffers:**

```
//+------------------------------------------------------------------+
//| Initialize the timeseries and the appropriate buffers by index   |
//+------------------------------------------------------------------+
bool InitBuffer(const int buffer_index)
  {
//--- Leave if the wrong index is passed
   if(buffer_index==WRONG_VALUE)
      return false;
Initialize the variables using the "Not rendered" drawing style and disable the display in the data window
   int draw_type=DRAW_NONE;
   bool show_data=false;
//--- If the buffer is used (button pressed)
//--- Set the "Line" drawing style for variables and enable display in the data window
   if(Buffers[buffer_index].IsUsed())
     {
      draw_type=DRAW_LINE;
      show_data=true;
     }
//--- Set the drawing style and display in the data window for the buffer by its index
   PlotIndexSetInteger(Buffers[buffer_index].IndexDataBuffer(),PLOT_DRAW_TYPE,draw_type);
   PlotIndexSetInteger(Buffers[buffer_index].IndexDataBuffer(),PLOT_SHOW_DATA,show_data);
//--- Initialize the calculated buffer using zero, while the drawn one is initialized using the "empty" value
   ArrayInitialize(Buffers[buffer_index].Temp,0);
   ArrayInitialize(Buffers[buffer_index].Data,EMPTY_VALUE);
   return true;
  }
//+------------------------------------------------------------------+
//|Initialize the timeseries and the appropriate buffers by timeframe|
//+------------------------------------------------------------------+
bool InitBuffer(const ENUM_TIMEFRAMES timeframe)
  {
   return InitBuffer(IndexBuffer(timeframe));
  }
//+------------------------------------------------------------------+
//| Initialize all timeseries and the appropriate buffers            |
//+------------------------------------------------------------------+
void InitBuffersAll(void)
  {
//--- Initialize the next buffer in the loop by the total number of chart periods
   for(int i=0;i<PERIODS_TOTAL;i++)
      if(!InitBuffer(i))
         continue;
  }
//+------------------------------------------------------------------+
```

**The function of calculating a single specified bar of all used indicator buffers (the button is pressed for):**

```
//+------------------------------------------------------------------+
//| Calculating a single bar of all active buffers                   |
//+------------------------------------------------------------------+
void CalculateSeries(const ENUM_BAR_PROP_DOUBLE property,const int index,const datetime time)
  {
//--- Get the next buffer in the loop by the total number of chart periods
   for(int i=0;i<PERIODS_TOTAL;i++)
     {
      //--- if the buffer is not used (the button is released), move on to the next one
      if(!Buffers[i].IsUsed())
         continue;
      //--- get the timeseries object by the buffer timeframe
      CSeriesDE *series=engine.SeriesGetSeries(NULL,(ENUM_TIMEFRAMES)Buffers[i].ID());
      //--- if the timeseries is not received
      //--- or the bar index passed to the function is beyond the total number of bars in the timeseries, move on to the next buffer
      if(series==NULL || index>series.GetList().Total()-1)
         continue;
      //--- get the bar object from the timeseries corresponding to the one passed to the bar time function on the current chart
      CBar *bar=engine.SeriesGetBarSeriesFirstFromSeriesSecond(NULL,PERIOD_CURRENT,time,NULL,series.Timeframe());
      if(bar==NULL)
         continue;
      //--- get the specified property from the obtained bar and
      //--- call the function of writing the value to the buffer by i index
      double value=bar.GetProperty(property);
      SetBufferData(i,value,index,bar);
     }
  }
//+------------------------------------------------------------------+
```

**The function of writing the bar object property to the indicator buffer by several bar indices on the current chart:**

```
//+------------------------------------------------------------------+
//| Write data on a single bar to the specified buffer               |
//+------------------------------------------------------------------+
void SetBufferData(const int buffer_index,const double value,const int index,const CBar *bar)
  {
//--- Get the bar index by its time falling within the time limits on the current chart
   int n=iBarShift(NULL,PERIOD_CURRENT,bar.Time());
//--- If the passed index on the current chart (index) is less than the calculated time of bar start on another timeframe
   if(index<n)
      //--- in the loop from the n bar on the current chart to zero
      while(n>WRONG_VALUE && !IsStopped())
        {
         //--- fill in the n index buffer with the 'value' passed to the function (0 - EMPTY_VALUE)
         //--- and decrease the n value
         Buffers[buffer_index].Data[n]=(value>0 ? value : EMPTY_VALUE);
         n--;
        }
//--- If the passed index on the current chart (index) is not less than the calculated time of bar start on another timeframe
//--- Set 'value' for the buffer by the 'index' passed to the function (0 - EMPTY_VALUE)
   else
      Buffers[buffer_index].Data[index]=(value>0 ? value : EMPTY_VALUE);
  }
//+------------------------------------------------------------------+
```

For correct bar data display from another timeframe on the current chart, find the start of the specified candle (bar) period on the current chart and fill in all buffer indices on the current chart with the value of the bar on another period. This is what the function does.

When pressing the timeframe activation button, we need to either fill in the appropriate displayed buffer with an empty value (if the button is released) or fully re-calculate all data of the buffer indicated by the button (if the button is pressed). The buffer initialization function deletes the data, while the following function fills in the buffer with the specified timeseries data:

```
//+------------------------------------------------------------------+
//| Fill in the entire buffer with historical data                   |
//+------------------------------------------------------------------+
void BufferFill(const int buffer_index)
  {
//--- Leave if the wrong index is passed
   if(buffer_index==WRONG_VALUE)
      return;
//--- Leave if the buffer is not used (the button is released)
   if(!Buffers[buffer_index].IsUsed())
      return;
//--- Get the timeseries object by the buffer timeframe
   CSeriesDE *series=engine.SeriesGetSeries(NULL,(ENUM_TIMEFRAMES)Buffers[buffer_index].ID());
   if(series==NULL)
      return;
//--- If the buffer belongs to the current chart, copy the bar data from the timeseries to the buffer
   if(Buffers[buffer_index].ID()==Period())
      series.CopyToBufferAsSeries((ENUM_BAR_PROP_DOUBLE)InpBarPrice,Buffers[buffer_index].Data,EMPTY_VALUE);
//--- Otherwise, calculate each next timeseries bar and write it to the buffer in the loop by the number of the current chart bars
   else
      for(int i=rates_data.rates_total-1;i>WRONG_VALUE && !IsStopped();i--)
         CalculateSeries((ENUM_BAR_PROP_DOUBLE)InpBarPrice,i,(datetime)BufferTime[i]);
  }
//+------------------------------------------------------------------+
```

The full indicator code is provided in the files attached below.

Please keep in mind that this test indicator was developed in MQL5. It works on MQL4 as well but not in a normal way — when pressing the appropriate button, the current chart period is not displayed. It is only displayed when activating yet another timeframe. When setting non-standard chart periods in MetaTrader 4 settings, the indicator endlessly waits for their synchronization.

Also some data is displayed incorrectly in the terminal data window — all the indicator buffers are displayed (including the calculated ones), which is natural since not all MQL5 functions work in MQL4 and should be replaced with their MQL4 counterparts.

Moreover, the indicator may incorrectly handle changes in historical data in MetaTrader 5 as well since the indicator is made for test purposes, namely to check the operation in multi-period mode. All detected bugs are to be gradually fixed in subsequent articles. When all shortcomings are eliminated in MetaTrader 5, the library is to be adjusted for MetaTrader 4 indicators.

Compile the indicator and launch it on the chart:

![](https://c.mql5.com/2/38/Jj3R8ki983.gif)

As we can see, on М15, the data buffer from М5 shows the М5 bar close prices in one of a third of the current chart candles, which is understandable since a single М15 bar contains three М5 bars, and the М5 bar close price is displayed on М15 bar.

Launch the indicator in the tester with the enabled parameter of displaying the timeseries data on the current chart period:

![](https://c.mql5.com/2/38/8ISivtMkRf.gif)

### What's next?

In the next article, we will continue our work on handling library timeseries objects in indicators.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7771#node00)

**Previous articles within the series:**

[Timeseries in DoEasy library (part 35): Bar object and symbol timeseries list](https://www.mql5.com/en/articles/7594)

[Timeseries in DoEasy library (part 36): Object of timeseries for all used symbol periods](https://www.mql5.com/en/articles/7627)

[Timeseries in DoEasy library (part 37): Timeseries collection - database of timeseries by symbols and periods](https://www.mql5.com/en/articles/7663)

[Timeseries in DoEasy library (part 38): Timeseries collection - real-time updates and accessing data from the program](https://www.mql5.com/en/articles/7695)

[Timeseries in DoEasy library (part 39): Library-based indicators - preparing data and timeseries events](https://www.mql5.com/en/articles/7724)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7771](https://www.mql5.com/ru/articles/7771)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7771.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7771/mql5.zip "Download MQL5.zip")(3700.41 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7771/mql4.zip "Download MQL4.zip")(3700.4 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/348043)**
(4)


![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
4 Sep 2020 at 02:49

H there Artyom,

I noticed that, in this article, you renamed of the files **SeriesDE.mqh** and **TimeSeriesDE.mqh** to avoid collisions with the Standard Library (MQL5\\Include\\Indicators\\). I'd like to know if you are very familiar with the Strategy Modules of the Standard Library ( [https://www.mql5.com/en/docs/standardlibrary/expertclasses](https://www.mql5.com/en/docs/standardlibrary/expertclasses)) and whether it is a good idea to use them in conjunction with your DoEasy library? I am considering to rewrite **CExpert**, **CExpertTrade** and any other necessary classes to use your DoEasy code instead of **Ctrade** from the Standard Library...

Hopefully, this will provide me with a robust and modular EA development framework that is part of MT5 and with DoEasy can be adapted for compatibilitywith MT4 as well. I understand that this is not a very simple task, but would love to hear your views and recommendation for the best approach?

Thanks, /dima

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
4 Sep 2020 at 04:31

**ddiall :**

H there Artyom,

I noticed that, in this article, you renamed of the files  **SeriesDE.mqh**  and  **TimeSeriesDE.mqh** to avoid collisions with the Standard Library (MQL5\\Include\\Indicators\\). I'd like to know if you are very familiar with the Strategy Modules of the Standard Library ( [https://www.mql5.com/en/docs/standardlibrary/expertclasses](https://www.mql5.com/en/docs/standardlibrary/expertclasses) ) and whether it is a good idea to use them in conjunction with your DoEasy library? I am considering to rewrite **CExpert** , **CExpertTrade** and any other necessary classes to use your DoEasy code instead of **Ctrade** from the Standard Library...

Hopefully, this will provide me with a robust and modular  EA development framework that is part of MT5 and with DoEasy can be adapted for compatibility with MT4 as well.  I understand that this is not a very simple task, but would love to hear your views and recommendation for the best  approach?

Thanks, /dima

Hey. I didn't understand this set of classes. You can try to study them and combine them with the library, but later, when the same functionality is implemented in the library, this will be overkill. Either way, trying to figure out a set of classes will be a good experience for you.

![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
4 Sep 2020 at 12:08

Yes, I tend to agree that it looks like overkill combining the two; those [Standard Library](https://www.mql5.com/en/docs/standardlibrary "MQL5 Documentation: Standard Library") classes seem overly complicated and particularly tailored for the EA Wizard...

Maybe I will create my own **CExpertAdvisor** wrapper class around your DoEasy library for my initial needs and keep things simple for now. Can you give me an idea of the upcoming features you plan to implement specifically for EA development, and when you target to release that?

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
4 Sep 2020 at 13:39

**ddiall :**

Yes, I tend to agree that it looks like overkill combining the two; those [Standard Library](https://www.mql5.com/en/docs/standardlibrary "MQL5 Documentation: Standard Library") classes seem overly complicated and particularly tailored for the EA Wizard...

Maybe I will create my own **CExpertAdvisor** wrapper class around your DoEasy library for my initial needs and keep things simple for now. Can you give me an idea of the upcoming features you plan to implement specifically for EA development, and when you target to release that?

All the "auxiliary" functionality of the library will be implemented after the creation of all the necessary basic functionality. And at this stage, the basic functionality is being created. It is possible that I will use some of the classes provided in the standard library - if they fit well with the library's concept. That is why I told you about some excess when using these classes when making your own additions to this library.

![Continuous Walk-Forward Optimization (Part 7): Binding Auto Optimizer's logical part with graphics and controlling graphics from the program](https://c.mql5.com/2/38/MQL5-avatar-continuous_optimization__4.png)[Continuous Walk-Forward Optimization (Part 7): Binding Auto Optimizer's logical part with graphics and controlling graphics from the program](https://www.mql5.com/en/articles/7747)

This article describes the connection of the graphical part of the auto optimizer program with its logical part. It considers the optimization launch process, from a button click to task redirection to the optimization manager.

![Developing a cross-platform grid EA: testing a multi-currency EA](https://c.mql5.com/2/38/mql5_ea_adviser_grid.png)[Developing a cross-platform grid EA: testing a multi-currency EA](https://www.mql5.com/en/articles/7777)

Markets dropped down by more that 30% within one month. It seems to be the best time for testing grid- and martingale-based Expert Advisors. This article is an unplanned continuation of the series "Creating a Cross-Platform Grid EA". The current market provides an opportunity to arrange a stress rest for the grid EA. So, let's use this opportunity and test our Expert Advisor.

![Timeseries in DoEasy library (part 41): Sample multi-symbol multi-period indicator](https://c.mql5.com/2/38/MQL5-avatar-doeasy-library__6.png)[Timeseries in DoEasy library (part 41): Sample multi-symbol multi-period indicator](https://www.mql5.com/en/articles/7804)

In the article, we will consider a sample multi-symbol multi-period indicator using the timeseries classes of the DoEasy library displaying the chart of a selected currency pair on a selected timeframe as candles in a subwindow. I am going to modify the library classes a bit and create a separate file for storing enumerations for program inputs and selecting a compilation language.

![Continuous Walk-Forward Optimization (Part 6): Auto optimizer's logical part and structure](https://c.mql5.com/2/38/MQL5-avatar-continuous_optimization__3.png)[Continuous Walk-Forward Optimization (Part 6): Auto optimizer's logical part and structure](https://www.mql5.com/en/articles/7718)

We have previously considered the creation of automatic walk-forward optimization. This time, we will proceed to the internal structure of the auto optimizer tool. The article will be useful for all those who wish to further work with the created project and to modify it, as well as for those who wish to understand the program logic. The current article contains UML diagrams which present the internal structure of the project and the relationships between objects. It also describes the process of optimization start, but it does not contain the description of the optimizer implementation process.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=fnkvsdixwczkmpikehoxutknklxmmptk&ssn=1769186174723742345&ssn_dr=0&ssn_sr=0&fv_date=1769186174&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7771&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Timeseries%20in%20DoEasy%20library%20(part%2040)%3A%20Library-based%20indicators%20-%20updating%20data%20in%20real%20time%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918617476899438&fz_uniq=5070418809803707772&sv=2552)

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