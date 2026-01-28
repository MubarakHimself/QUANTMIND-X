---
title: Timeseries in DoEasy library (part 36): Object of timeseries for all used symbol periods
url: https://www.mql5.com/en/articles/7627
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:52:05.344826
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=tcxwgwnszxxpeyfdlxfeiebwpvgkeofu&ssn=1769251923959366178&ssn_dr=0&ssn_sr=0&fv_date=1769251923&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7627&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Timeseries%20in%20DoEasy%20library%20(part%2036)%3A%20Object%20of%20timeseries%20for%20all%20used%20symbol%20periods%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925192396193943&fz_uniq=5083169373704689235&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/7627#node01)
- [Symbol timeseries object](https://www.mql5.com/en/articles/7627#node02)
- [Testing](https://www.mql5.com/en/articles/7627#node03)
- [What's next?](https://www.mql5.com/en/articles/7627#node04)


### Concept

[In the previous article](https://www.mql5.com/en/articles/7594), we started a new series of DoEasy library descriptions and examined the creation of the Bar object and the bar object list. In relation to the MetaTrader platform terminology, we created a single timeseries for one symbol at one timeframe and filled it with data for each bar of this timeseries.

In relation to the DoEasy library terminology, we created the bar collection object for a single symbol at one timeframe. Now we are able to perform any search and sorting within the created collection (within the specified collection history depth) by any property of bar objects present in the collection. In simple words, we are able to search for various parameters of timeseries bars and their various combinations (various combinations of bars are to be added later), as well as detect a new bar in the created collection and update the current data in it.

This is good but insufficient since we can use multiple timeframes and symbols for our programs. Therefore, the number of symbol timeseries collections at our disposal should be equal to the number of its timeframes we need to use in our programs.

The created timeseries collection allows this to be done — it is created in relation to a symbol and timeframe, which means we can create as many collections of one symbol as we need.

It would be convenient to store all collections of one symbol but different timeframes in one object — the symbol timeseries object. A single common timeseries collection for different symbols and their timeframes is then created of these objects.

### Symbol timeseries object

In the foreseeable future, many library classes will require knowledge of the type of a program they are running in. To do this, use the [MQLInfoInteger()](https://www.mql5.com/en/docs/check/mqlinfointeger) function with the [MQL\_PROGRAM\_TYPE](https://www.mql5.com/en/docs/constants/environment_state/mql5_programm_info#enum_mql_info_integer) specifier. In this case, the function returns the [type of the running mql5 program](https://www.mql5.com/en/docs/constants/environment_state/mql5_programm_info#enum_program_type).

To avoid writing the variables storing the program type in each class, we will declare the variable [in the base class of all program objects](https://www.mql5.com/en/articles/7071). All classes derived from the base class will have the variable storing the type of an executed program.

In the protected section of the CBaseObj class located in \\MQL5\\Include\\DoEasy\\Objects\\BaseObj.mqh, **declare the class member variable storing the type of the running program**:

```
//+------------------------------------------------------------------+
//| Base object class for all library objects                        |
//+------------------------------------------------------------------+
#define  CONTROLS_TOTAL    (10)
class CBaseObj : public CObject
  {
private:
   int               m_long_prop_total;
   int               m_double_prop_total;
   //--- Fill in the object property array
   template<typename T> bool  FillPropertySettings(const int index,T &array[][CONTROLS_TOTAL],T &array_prev[][CONTROLS_TOTAL],int &event_id);
protected:
   CArrayObj         m_list_events_base;                       // Object base event list
   CArrayObj         m_list_events;                            // Object event list
   ENUM_LOG_LEVEL    m_log_level;                              // Logging level
   ENUM_PROGRAM_TYPE m_program;                                // Program type
   MqlTick           m_tick;                                   // Tick structure for receiving quote data
```

while **in the class constructor, set the value to it**:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CBaseObj::CBaseObj() : m_global_error(ERR_SUCCESS),
                       m_hash_sum(0),m_hash_sum_prev(0),
                       m_is_event(false),m_event_code(WRONG_VALUE),
                       m_chart_id_main(::ChartID()),
                       m_chart_id(::ChartID()),
                       m_folder_name(DIRECTORY),
                       m_name(__FUNCTION__),
                       m_long_prop_total(0),
                       m_double_prop_total(0),
                       m_first_start(true)
  {
   ::ArrayResize(this.m_long_prop_event,0,100);
   ::ArrayResize(this.m_double_prop_event,0,100);
   ::ArrayResize(this.m_long_prop_event_prev,0,100);
   ::ArrayResize(this.m_double_prop_event_prev,0,100);
   ::ZeroMemory(this.m_tick);
   this.m_program=(ENUM_PROGRAM_TYPE)::MQLInfoInteger(MQL_PROGRAM_TYPE);
   this.m_digits_currency=(#ifdef __MQL5__ (int)::AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS) #else 2 #endif);
   this.m_list_events.Clear();
   this.m_list_events.Sort();
   this.m_list_events_base.Clear();
   this.m_list_events_base.Sort();
  }
//+------------------------------------------------------------------+
```

Now all objects derived from the base class of all library objects "know" about the type of program they are running in.

In the **CNewBarObj** clas listing in \\MQL5\\Include\\DoEasy\\Objects\\Series\\NewBarObj.mqh, **include the base object class file**:

```
//+------------------------------------------------------------------+
//|                                                    NewBarObj.mqh |
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
//| "New bar" object class                                           |
//+------------------------------------------------------------------+
```

inherit the "New bar" object from the base object:

```
//+------------------------------------------------------------------+
//| "New bar" object class                                           |
//+------------------------------------------------------------------+
class CNewBarObj : public CBaseObj
  {
private:
```

and remove "all mentions" of the program type variable from the listing — now the program type is set in CBaseObj:

```
//+------------------------------------------------------------------+
//| "New bar" object class                                           |
//+------------------------------------------------------------------+
class CNewBarObj
  {
private:
   ENUM_PROGRAM_TYPE m_program;                                   // Program type
   string            m_symbol;                                    // Symbol
   ENUM_TIMEFRAMES   m_timeframe;                                 // Timeframe
   datetime          m_new_bar_time;                              // New bar time for auto time management
   datetime          m_prev_time;                                 // Previous time for auto time management
   datetime          m_new_bar_time_manual;                       // New bar time for manual time management
   datetime          m_prev_time_manual;                          // Previous time for manual time management
//--- Return the current bar data
   datetime          GetLastBarDate(const datetime time);
public:
//--- Set (1) symbol and (2) timeframe
   void              SetSymbol(const string symbol)               { this.m_symbol=(symbol==NULL || symbol==""   ? ::Symbol() : symbol);                     }
   void              SetPeriod(const ENUM_TIMEFRAMES timeframe)   { this.m_timeframe=(timeframe==PERIOD_CURRENT ? (ENUM_TIMEFRAMES)::Period() : timeframe); }
//--- Save the new bar time during the manual time management
   void              SaveNewBarTime(const datetime time)          { this.m_prev_time_manual=this.GetLastBarDate(time);                                      }
//--- Return (1) symbol and (2) timeframe
   string            Symbol(void)                           const { return this.m_symbol;       }
   ENUM_TIMEFRAMES   Period(void)                           const { return this.m_timeframe;    }
//--- Return the (1) new bar time
   datetime          TimeNewBar(void)                       const { return this.m_new_bar_time; }
//--- Return the new bar opening flag during the time (1) auto, (2) manual management
   bool              IsNewBar(const datetime time);
   bool              IsNewBarManual(const datetime time);
//--- Constructors
                     CNewBarObj(void) : m_program((ENUM_PROGRAM_TYPE)::MQLInfoInteger(MQL_PROGRAM_TYPE)),
                                        m_symbol(::Symbol()),
                                        m_timeframe((ENUM_TIMEFRAMES)::Period()),
                                        m_prev_time(0),m_new_bar_time(0),
                                        m_prev_time_manual(0),m_new_bar_time_manual(0) {}
                     CNewBarObj(const string symbol,const ENUM_TIMEFRAMES timeframe);
  };
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CNewBarObj::CNewBarObj(const string symbol,const ENUM_TIMEFRAMES timeframe) : m_symbol(symbol),m_timeframe(timeframe)
  {
   this.m_program=(ENUM_PROGRAM_TYPE)::MQLInfoInteger(MQL_PROGRAM_TYPE);
   this.m_prev_time=this.m_prev_time_manual=this.m_new_bar_time=this.m_new_bar_time_manual=0;
  }
//+------------------------------------------------------------------+
```

Find the corrected full class listing in the files attached to the article.

**Let's slightly modify the CSeries class created [in the previous article](https://www.mql5.com/en/articles/7594#node04).**

Inherit the class from the CBaseObj base object instead of CObject and remove the variable storing the program type from it:

```
//+------------------------------------------------------------------+
//| Timeseries class                                                 |
//+------------------------------------------------------------------+
class CSeries : public CBaseObj
  {
private:
   ENUM_PROGRAM_TYPE m_program;                                         // Program type
   ENUM_TIMEFRAMES   m_timeframe;                                       // Timeframe
   string            m_symbol;                                          // Symbol
   uint              m_amount;                                          // Amount of applied timeseries data
   uint              m_bars;                                            // Number of bars in history by symbol and timeframe
   bool              m_sync;                                            // Synchronized data flag
   CArrayObj         m_list_series;                                     // Timeseries list
   CNewBarObj        m_new_bar_obj;                                     // "New bar" object
public:
```

In the public section of the class, declare the methods of setting a symboland timeframe, as well as complete implementing the method returning the total amount of available data:

```
//+------------------------------------------------------------------+
//| Timeseries class                                                 |
//+------------------------------------------------------------------+
class CSeries : public CBaseObj
  {
private:
   ENUM_TIMEFRAMES   m_timeframe;                                       // Timeframe
   string            m_symbol;                                          // Symbol
   uint              m_amount;                                          // Amount of applied timeseries data
   uint              m_bars;                                            // Number of bars in history by symbol and timeframe
   bool              m_sync;                                            // Synchronized data flag
   CArrayObj         m_list_series;                                     // Timeseries list
   CNewBarObj        m_new_bar_obj;                                     // "New bar" object
public:
//--- Return the timeseries list
   CArrayObj        *GetList(void)                                      { return &m_list_series;}
//--- Return the list of bars by selected (1) double, (2) integer and (3) string property fitting a compared condition
   CArrayObj        *GetList(ENUM_BAR_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL){ return CSelect::ByBarProperty(this.GetList(),property,value,mode); }
   CArrayObj        *GetList(ENUM_BAR_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByBarProperty(this.GetList(),property,value,mode); }
   CArrayObj        *GetList(ENUM_BAR_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL){ return CSelect::ByBarProperty(this.GetList(),property,value,mode); }

//--- Set (1) symbol, (2) timeframe, (3) symbol and timeframe, (4) amount of applied timeseries data
   void              SetSymbol(const string symbol);
   void              SetTimeframe(const ENUM_TIMEFRAMES timeframe);
   void              SetSymbolPeriod(const string symbol,const ENUM_TIMEFRAMES timeframe);
   bool              SetAmountUsedData(const uint amount,const uint rates_total);

//--- Return (1) symbol, (2) timeframe, (3) number of applied timeseries data,
//--- (4) number of bars in the timeseries, the new bar flag with the (5) auto, (6) manual time management
   string            Symbol(void)                                          const { return this.m_symbol;                            }
   ENUM_TIMEFRAMES   Timeframe(void)                                       const { return this.m_timeframe;                         }
   uint              AmountUsedData(void)                                  const { return this.m_amount;                            }
   uint              Bars(void)                                            const { return this.m_bars;                              }
   bool              IsNewBar(const datetime time)                               { return this.m_new_bar_obj.IsNewBar(time);        }
   bool              IsNewBarManual(const datetime time)                         { return this.m_new_bar_obj.IsNewBarManual(time);  }
//--- Return the bar object by index (1) in the list and (2) in the timeseries, as well as (3) the real list size
   CBar             *GetBarByListIndex(const uint index);
   CBar             *GetBarBySeriesIndex(const uint index);
   int               DataTotal(void)                                       const { return this.m_list_series.Total();               }
//--- Return (1) Open, (2) High, (3) Low, (4) Close, (5) time, (6) tick volume, (7) real volume, (8) bar spread by index
   double            Open(const uint index,const bool from_series=true);
   double            High(const uint index,const bool from_series=true);
   double            Low(const uint index,const bool from_series=true);
   double            Close(const uint index,const bool from_series=true);
   datetime          Time(const uint index,const bool from_series=true);
   long              TickVolume(const uint index,const bool from_series=true);
   long              RealVolume(const uint index,const bool from_series=true);
   int               Spread(const uint index,const bool from_series=true);

//--- Save the new bar time during the manual time management
   void              SaveNewBarTime(const datetime time)                         { this.m_new_bar_obj.SaveNewBarTime(time);         }
//--- Synchronize symbol and timeframe data with server data
   bool              SyncData(const uint amount,const uint rates_total);
//--- (1) Create and (2) update the timeseries list
   int               Create(const uint amount=0);
   void              Refresh(const datetime time=0,
                             const double open=0,
                             const double high=0,
                             const double low=0,
                             const double close=0,
                             const long tick_volume=0,
                             const long volume=0,
                             const int spread=0);

//--- Constructors
                     CSeries(void);
                     CSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint amount=0);
  };
//+------------------------------------------------------------------+
```

Beyond the class body, **implement the methods of setting a symbol and timeframe:**

```
//+------------------------------------------------------------------+
//| Set a symbol                                                     |
//+------------------------------------------------------------------+
void CSeries::SetSymbol(const string symbol)
  {
   this.m_symbol=(symbol==NULL || symbol==""   ? ::Symbol() : symbol);
   this.m_new_bar_obj.SetSymbol(this.m_symbol);
  }
//+------------------------------------------------------------------+
//| Set a timeframe                                                  |
//+------------------------------------------------------------------+
void CSeries::SetTimeframe(const ENUM_TIMEFRAMES timeframe)
  {
   this.m_timeframe=(timeframe==PERIOD_CURRENT ? (ENUM_TIMEFRAMES)::Period() : timeframe);
   this.m_new_bar_obj.SetPeriod(this.m_timeframe);
  }
//+------------------------------------------------------------------+
```

The values passed to the methods are checked and adjusted if necessary. Then they are sent to the variables.

After that, the value is set in the "New bar" class object.

The CSelect class listing in \\MQL5\\Include\\DoEasy\\Services\\Select.mqh is to include the CSeries class file instead of the Bar.mqh file of the CBar class:

```
//+------------------------------------------------------------------+
//|                                                       Select.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Arrays\ArrayObj.mqh>
#include "..\Objects\Orders\Order.mqh"
#include "..\Objects\Events\Event.mqh"
#include "..\Objects\Accounts\Account.mqh"
#include "..\Objects\Symbols\Symbol.mqh"
#include "..\Objects\PendRequest\PendRequest.mqh"
#include "..\Objects\Series\Series.mqh"
//+------------------------------------------------------------------+
//| Storage list                                                     |
//+------------------------------------------------------------------+
CArrayObj   ListStorage; // Storage object for storing sorted collection lists
//+------------------------------------------------------------------+
//| Class for sorting objects meeting the criterion                  |
//+------------------------------------------------------------------+
class CSelect
  {
```

**Now all is ready for creating the class of the object of all symbol timeseries.**

What is a symbol timeseries object? In the previous article, we created the timeseries object of one period for a single symbol. Now all bar objects inside that list can be sorted by any of the bar object properties, any bar object can be detected by any of its properties, etc. However, the programs often require the use of multi-period analysis of history of one or several symbols. The symbol timeseries object contains multiple timeseries of all possible timeframes of one symbol. The number of timeframes can be equal to the number of available chart periods in the terminal described in the [ENUM\_TIMEFRAMES](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes) enumeration.

In fact, the object is [the array of pointers to CArrayObj objects](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj), in which objects are symbol timeseries lists I have created [in the previous article](https://www.mql5.com/en/articles/7594#node04). In turn, they contain [bar objects](https://www.mql5.com/en/articles/7594#node02).

In the current article, we will create the object of all symbol timeseries allowing to:

- manually set the use of:


  - specified chart periods of a single symbol
  - all possible chart periods of a single symbol

- create:

  - specified timeseries objects of a single symbol
  - all possible timeseries objects of a single symbol

- update data of the:


  - specified timeseries objects of a single symbol
  - all timeseries objects of a single symbol

The remaining object functionality is added when creating the object of all timeseries of all used symbols in subsequent articles.

As usual, we first add all the messages necessary for the new class — message indices and their appropriate texts.

In the Datas.mqh file located in \\MQL5\\Include\\DoEasy\\Datas.mqh, add new message indices:

```
   MSG_LIB_TEXT_BAR_TEXT_FIRS_SET_AMOUNT_DATA,        // First, we need to set the required amount of data using SetAmountUsedData()

//--- CTimeSeries
   MSG_LIB_TEXT_TS_TEXT_FIRS_SET_SYMBOL,              // First, set a symbol using SetSymbol()
   MSG_LIB_TEXT_TS_TEXT_UNKNOWN_TIMEFRAME,            // Unknown timeframe
   MSG_LIB_TEXT_TS_FAILED_GET_SERIES_OBJ,             // Failed to receive the timeseries object
  };
//+------------------------------------------------------------------+
```

and message texts corresponding to newly added indices:

```
   {"Сначала нужно установить требуемое количество данных при помощи SetAmountUsedData()","First you need to set required amount of data using SetAmountUsedData()"},

   {"Сначала нужно установить символ при помощи SetSymbol()","First you need to set Symbol using SetSymbol()"},
   {"Неизвестный таймфрейм","Unknown timeframe"},
   {"Не удалось получить объект-таймсерию ","Failed to get timeseries object "},

  };
//+---------------------------------------------------------------------+
```

In \\MQL5\\Include\\DoEasy\\Objects\ **Series\**, create the **TimeSeries.mqh** file of the CTimeSeries class with the Series.mqh timeseries object file connected to it and derived from the [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject) standard library base object.

Fill the class body with the required content and then consider all variables and methods separately:

```
//+------------------------------------------------------------------+
//|                                                   TimeSeries.mqh |
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
#include "Series.mqh"
//+------------------------------------------------------------------+
//| Timeseries class                                                 |
//+------------------------------------------------------------------+
class CTimeSeries : public CObject
  {
private:
   string            m_symbol;                                             // Timeseries symbol
   CArrayObj         m_list_series;                                        // List of timeseries by timeframes
//--- Return (1) the timeframe index in the list and (2) the timeframe by index
   char              IndexTimeframe(const ENUM_TIMEFRAMES timeframe) const;
   ENUM_TIMEFRAMES   TimeframeByIndex(const uchar index)             const;
public:
//--- Return (1) the full list of timeseries, (2) specified timeseries object and (3) timeseries object by index
   CArrayObj        *GetList(void)                                         { return &this.m_list_series;                                        }
   CSeries          *GetSeries(const ENUM_TIMEFRAMES timeframe)            { return this.m_list_series.At(this.IndexTimeframe(timeframe));      }
   CSeries          *GetSeriesByIndex(const uchar index)                   { return this.m_list_series.At(index);                               }
//--- Set/return timeseries symbol
   void              SetSymbol(const string symbol)                        { this.m_symbol=(symbol==NULL || symbol=="" ? ::Symbol() : symbol);  }
   string            Symbol(void)                                    const { return this.m_symbol;                                              }
//--- Set the history depth (1) of a specified timeseries and (2) of all applied symbol timeseries
   bool              SetAmountUsedData(const ENUM_TIMEFRAMES timeframe,const uint amount=0,const int rates_total=0);
   bool              SetAmountAllUsedData(const uint amount=0,const int rates_total=0);
//--- Return the flag of data synchronization with the server data of the (1) specified timeseries, (2) all timeseries
   bool              SyncData(const ENUM_TIMEFRAMES timeframe,const uint amount=0,const uint rates_total=0);
   bool              SyncAllData(const uint amount=0,const uint rates_total=0);

//--- Create (1) the specified timeseries list and (2) all timeseries lists
   bool              SeriesCreate(const ENUM_TIMEFRAMES timeframe,const uint amount=0);
   bool              SeriesCreateAll(const uint amount=0);
//--- Update (1) the specified timeseries list and (2) all timeseries lists
   void              Refresh(const ENUM_TIMEFRAMES timeframe,
                             const datetime time=0,
                             const double open=0,
                             const double high=0,
                             const double low=0,
                             const double close=0,
                             const long tick_volume=0,
                             const long volume=0,
                             const int spread=0);
   void              RefreshAll(const datetime time=0,
                             const double open=0,
                             const double high=0,
                             const double low=0,
                             const double close=0,
                             const long tick_volume=0,
                             const long volume=0,
                             const int spread=0);
//--- Constructor
                     CTimeSeries(void);
  };
//+------------------------------------------------------------------+
```

The **m\_symbol** class member variable stores a symbol name, for which the necessary timeseries are created, stored and handled in the object. Subsequently, the variable value is used to select the necessary objects with timeseries of required symbols.

The [array of pointers to CObject class instances](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) **m\_list\_series** is designed to store timeseries objects created in the previous article. The number of objects in the list can be equal to the number of all timeframes available in the platform, and they are arranged in the list in the order they are listed in the [ENUM\_TIMEFRAMES](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes) enumeration, which in turn allows us to know exactly the index of each timeseries object in the list. Two methods are created to return the index in the timeseries object list:

The **IndexTimeframe()** method returns the timeseries object index in the list by a timeframe value.

Its implementation beyond the class body:

```
//+------------------------------------------------------------------+
//| Return the timeframe index in the list                           |
//+------------------------------------------------------------------+
char CTimeSeries::IndexTimeframe(ENUM_TIMEFRAMES timeframe) const
  {
   int statement=(timeframe==PERIOD_CURRENT ? ::Period() : timeframe);
   switch(statement)
     {
      case PERIOD_M1    :  return 0;
      case PERIOD_M2    :  return 1;
      case PERIOD_M3    :  return 2;
      case PERIOD_M4    :  return 3;
      case PERIOD_M5    :  return 4;
      case PERIOD_M6    :  return 5;
      case PERIOD_M10   :  return 6;
      case PERIOD_M12   :  return 7;
      case PERIOD_M15   :  return 8;
      case PERIOD_M20   :  return 9;
      case PERIOD_M30   :  return 10;
      case PERIOD_H1    :  return 11;
      case PERIOD_H2    :  return 12;
      case PERIOD_H3    :  return 13;
      case PERIOD_H4    :  return 14;
      case PERIOD_H6    :  return 15;
      case PERIOD_H8    :  return 16;
      case PERIOD_H12   :  return 17;
      case PERIOD_D1    :  return 18;
      case PERIOD_W1    :  return 19;
      case PERIOD_MN1   :  return 20;
      default           :  ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_TS_TEXT_UNKNOWN_TIMEFRAME)); return WRONG_VALUE;
     }
  }
//+------------------------------------------------------------------+
```

Everything is quite clear here. Depending on a timeframe passed to the method, its serial number in the [ENUM\_TIMEFRAMES](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes) enumeration (and accordingly, its index in the **m\_list\_series** list) is returned.

The **TimeframeByIndex()** method returns a timeframe by a timeseries object index in the list.

Its implementation beyond the class body:

```
//+------------------------------------------------------------------+
//| Return a timeframe by index                                      |
//+------------------------------------------------------------------+
ENUM_TIMEFRAMES CTimeSeries::TimeframeByIndex(const uchar index) const
  {
   switch(index)
     {
      case 0   :  return PERIOD_M1;
      case 1   :  return PERIOD_M2;
      case 2   :  return PERIOD_M3;
      case 3   :  return PERIOD_M4;
      case 4   :  return PERIOD_M5;
      case 5   :  return PERIOD_M6;
      case 6   :  return PERIOD_M10;
      case 7   :  return PERIOD_M12;
      case 8   :  return PERIOD_M15;
      case 9   :  return PERIOD_M20;
      case 10  :  return PERIOD_M30;
      case 11  :  return PERIOD_H1;
      case 12  :  return PERIOD_H2;
      case 13  :  return PERIOD_H3;
      case 14  :  return PERIOD_H4;
      case 15  :  return PERIOD_H6;
      case 16  :  return PERIOD_H8;
      case 17  :  return PERIOD_H12;
      case 18  :  return PERIOD_D1;
      case 19  :  return PERIOD_W1;
      case 20  :  return PERIOD_MN1;
      default  :  ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_NOT_GET_DATAS),"... ",CMessage::Text(MSG_SYM_STATUS_INDEX),": ",(string)index); return WRONG_VALUE;
     }
  }
//+------------------------------------------------------------------+
```

This method is the opposite of IndexTimeframe(). Depending on the index passed to the method, the appropriate timeframe is returned in the order of its location in the [ENUM\_TIMEFRAMES](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes) enumeration.

The **Getlist ()** method returns the full list of all timeseries 'as is' to the control program. While in the program, you are able to select the necessary timeseries from the obtained list.

The **GetSeries()** method returns the specified timeseries object from the **m\_list\_series** list by the name of the required timeseries from the [ENUM\_TIMEFRAMES](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes) enumeration. The previously considered IndexTimeframe() method is used to obtain the timeseries index in the list

The **GetSeriesByIndex()** method returns the timeseries object by its index in the **m\_list\_series** list.

**Implementing the method setting the history depth of the specified timeseries:**

```
//+------------------------------------------------------------------+
//| Set a history depth of a specified timeseries                    |
//+------------------------------------------------------------------+
bool CTimeSeries::SetAmountUsedData(const ENUM_TIMEFRAMES timeframe,const uint amount=0,const int rates_total=0)
  {
   if(this.m_symbol==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_TS_TEXT_FIRS_SET_SYMBOL));
      return false;
     }
   CSeries *series_obj=this.m_list_series.At(this.IndexTimeframe(timeframe));
   return series_obj.SetAmountUsedData(amount,rates_total);
  }
//+------------------------------------------------------------------+
```

The method receives the timeseries timeframe whose history depth should be set, the required size of history timeseries data (history depth; if 0, the depth of 1000 bars is used) and the number of the current timeseries bars (only for indicators when setting the history depth for the current symbol on the current timeframe — pass the **rates\_total** parameter to [OnCalculate()](https://www.mql5.com/en/docs/event_handlers/oncalculate); in other cases, the parameter is of no importance).

If a symbol is not set for the class object yet,  display the appropriate message and return false.

Get the requested timeseries object from the list by its index obtained by a timeframe name and return the result of setting the history depth using the timeseries object class method of the same name we considered [in the previous article](https://www.mql5.com/en/articles/7594#node04).

**Implementing the method setting a history depth for all used symbol timeseries:**

```
//+------------------------------------------------------------------+
//| Set the history depth of all applied symbol timeseries           |
//+------------------------------------------------------------------+
bool CTimeSeries::SetAmountAllUsedData(const uint amount=0,const int rates_total=0)
  {
   if(this.m_symbol==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_TS_TEXT_FIRS_SET_SYMBOL));
      return false;
     }
   bool res=true;
   for(int i=0;i<21;i++)
     {
      CSeries *series_obj=this.m_list_series.At(i);
      if(series_obj==NULL)
         continue;
      res &=series_obj.SetAmountUsedData(amount,rates_total);
     }
   return res;
  }
//+------------------------------------------------------------------+
```

The method receives the required size of historical timeseries data (history depth; if 0, the depth of 1000 bars is used) and the number of bars of the current timeseries (only for indicators when setting the history depth for the current symbol on the current timeframe — pass the **rates\_total** parameter to [OnCalculate()](https://www.mql5.com/en/docs/event_handlers/oncalculate), in other cases, the parameter is of no importance).

If a symbol is not set for the class object yet,  display the appropriate message and return false.

In a loop by the full list of all existing timeframes,get the next timeseries list by the loop index from the list and write the result of setting the history depth using the timeseries object class method of the same name we considered [in the previous article](https://www.mql5.com/en/articles/7594#node04) to the local **res** variable. If at least one of the methods for setting the history depth of any of the available timeseries objects returns false, false is set in the variable.

Upon the loop completion, return the result of all settings written to the **res** variable.

**Implementing the method returning the flag of specified timeseries data synchronization with the server data:**

```
//+------------------------------------------------------------------+
//| Return the flag of data synchronization                          |
//| with the server data                                             |
//+------------------------------------------------------------------+
bool CTimeSeries::SyncData(const ENUM_TIMEFRAMES timeframe,const uint amount=0,const uint rates_total=0)
  {
   if(this.m_symbol==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_TS_TEXT_FIRS_SET_SYMBOL));
      return false;
     }
   CSeries *series_obj=this.m_list_series.At(this.IndexTimeframe(timeframe));
   if(series_obj==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_TS_FAILED_GET_SERIES_OBJ),this.m_symbol," ",TimeframeDescription(timeframe));
      return false;
     }
   return series_obj.SyncData(amount,rates_total);
  }
//+------------------------------------------------------------------+
```

The method receives the timeseries timeframe whose synchronization flag should be returned, the required size of history timeseries data (history depth; if 0, the depth of 1000 bars is used) and the number of the current timeseries bars (only for indicators when setting the history depth for the current symbol on the current timeframe — pass the **rates\_total** parameter to [OnCalculate()](https://www.mql5.com/en/docs/event_handlers/oncalculate); in other cases, the parameter is of no importance).

If a symbol is not set for the class object yet,  display the appropriate message and return false.

Get the requested timeseries object from the list by its index obtained by a timeframe name and return the result of checking data synchronization result using the timeseries object class method of the same name we considered [in the previous article](https://www.mql5.com/en/articles/7594#node04).

**Implementing the method returning the flag of data synchronization with the server data for all timeseries:**

```
//+------------------------------------------------------------------+
//| Return the flag of data synchronization                          |
//| of all timeseries with the server data                           |
//+------------------------------------------------------------------+
bool CTimeSeries::SyncAllData(const uint amount=0,const uint rates_total=0)
  {
   if(this.m_symbol==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_TS_TEXT_FIRS_SET_SYMBOL));
      return false;
     }
   bool res=true;
   for(int i=0;i<21;i++)
     {
      CSeries *series_obj=this.m_list_series.At(i);
      if(series_obj==NULL)
         continue;
      res &=series_obj.SyncData(amount,rates_total);
     }
   return res;
  }
//+------------------------------------------------------------------+
```

The method receives the required size of historical timeseries data (history depth; if 0, the depth of 1000 bars is used) and the number of bars of the current timeseries (only for indicators when setting the history depth for the current symbol on the current timeframe — pass the **rates\_total** parameter to [OnCalculate()](https://www.mql5.com/en/docs/event_handlers/oncalculate), in other cases, the parameter is of no importance).

If a symbol is not set for the class object yet,  display the appropriate message and return false.

In a loop by the full list of all existing timeframes,get the next timeseries list by the loop index from the list and write the flag of checking data synchronization with the server using the timeseries object class method of the same name we considered [in the previous article](https://www.mql5.com/en/articles/7594#node04) to the local **res** variable. If at least one of the methods of checking timeseries object synchronization returns false, false is set in the variable.

Upon the loop completion, return the result of all checks set in the **res** variable.

**Implementing the method creating a specified timeseries list:**

```
//+------------------------------------------------------------------+
//| Create a specified timeseries list                               |
//+------------------------------------------------------------------+
bool CTimeSeries::Create(const ENUM_TIMEFRAMES timeframe,const uint amount=0)
  {
   CSeries *series_obj=this.m_list_series.At(this.IndexTimeframe(timeframe));
   if(series_obj==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_TS_FAILED_GET_SERIES_OBJ),this.m_symbol," ",TimeframeDescription(timeframe));
      return false;
     }
   if(series_obj.AmountUsedData()==0)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_BAR_TEXT_FIRS_SET_AMOUNT_DATA));
      return false;
     }
   return(series_obj.Create(amount)>0);
  }
//+------------------------------------------------------------------+
```

The method receives the timeframe whose timeseries should be created and the history depth of the created timeseries (the default value is zero — create a timeseries with the history depth previously set for the timeseries object using the SetAmountUsedData() method; if the value exceeds zero and less than the value of available timeseries bars of the specified chart period, the created history depth passed to the method is used)

Get the necessary timeseries object by its index received by the timeframe name. If failed to get the objector the history depth is not set for it yet, display the appropriate messages and return false.

Return the timeseries creation result from the method using a timeseries object method of the same name we considered [in the previous article](https://www.mql5.com/en/articles/7594#node04). Since the timeseries creation method described in the previous article returns the number of object bars added to the timeseries list, while the current method returns the boolean value, it is sufficient to return the result of comparing the number of added elements to the "above zero" list to return true or false. This is exactly what we do here.

**Implementing the method creating all timeseries lists:**

```
//+------------------------------------------------------------------+
//| Create all timeseries lists                                      |
//+------------------------------------------------------------------+
bool CTimeSeries::CreateAll(const uint amount=0)
  {
   bool res=true;
   for(int i=0;i<21;i++)
     {
      CSeries *series_obj=this.m_list_series.At(i);
      if(series_obj==NULL || series_obj.AmountUsedData()==0)
         continue;
      res &=(series_obj.Create(amount)>0);
     }
   return res;
  }
//+------------------------------------------------------------------+
```

The method receives the history depth of a created timeseries (the default value is zero — create a timeseries with the history depth previously set for the timeseries object using the SetAmountUsedData() method; if the value exceeds zero and less than the value of available timeseries bars of the specified chart period, the created history depth passed to the method is used).

In a loop by the list of all timeframes, get the next timeseries object by the loop index. If failed to get the object or a history depth is not set for it yet, move on to the next timeframe.

The local **res** variable receives the result of creating the timeseries using the timeseries object class method of the same name we considered [in the previous article](https://www.mql5.com/en/articles/7594#node04) expressed as a result of "above zero" comparison of the number of elements added to the list. If at least one of the methods of creating timeseries objects returns false, the variable is set to false.

Upon the loop completion, return the result of creating all timeseries written to the **res** variable.

**Implementing the method, updating a specified timeseries list:**

```
//+------------------------------------------------------------------+
//| Update a specified timeseries list                               |
//+------------------------------------------------------------------+
void CTimeSeries::Refresh(const ENUM_TIMEFRAMES timeframe,
                          const datetime time=0,
                          const double open=0,
                          const double high=0,
                          const double low=0,
                          const double close=0,
                          const long tick_volume=0,
                          const long volume=0,
                          const int spread=0)
  {
   CSeries *series_obj=this.m_list_series.At(this.IndexTimeframe(timeframe));
   if(series_obj==NULL || series_obj.DataTotal()==0)
      return;
   series_obj.Refresh(time,open,high,low,close,tick_volume,volume,spread);
  }
//+------------------------------------------------------------------+
```

The method receives the updated timeseries timeframe and the current bar price data (only for indicators when updating data of the current symbol on the current timeframe — pass data from price arraysto [OnCalculate()](https://www.mql5.com/en/docs/event_handlers/oncalculate); in other cases, the values of passed parameters do not matter).

Get the necessary timeseries object by its index received by the timeframe name. If failed to get the objector the size of a created timeframe history is zero (the timeframe is not used or was not created using the Create() method), exit the method.

Next, call the method of updating the timeseries object of the same name we considered [in the previous article](https://www.mql5.com/en/articles/7594#node04).

**Implementing the method updating all timeseries lists:**

```
//+------------------------------------------------------------------+
//| Update all timeseries lists                                      |
//+------------------------------------------------------------------+
void CTimeSeries::RefreshAll(const datetime time=0,
                             const double open=0,
                             const double high=0,
                             const double low=0,
                             const double close=0,
                             const long tick_volume=0,
                             const long volume=0,
                             const int spread=0)
  {
   for(int i=0;i<21;i++)
     {
      CSeries *series_obj=this.m_list_series.At(i);
      if(series_obj==NULL || series_obj.DataTotal()==0)
         continue;
      series_obj.Refresh(time,open,high,low,close,tick_volume,volume,spread);
     }
  }
//+------------------------------------------------------------------+
```

The method receives the current bar price data (only for indicators when updating data of the current symbol on the current timeframe — pass data from price arraysto [OnCalculate()](https://www.mql5.com/en/docs/event_handlers/oncalculate); in other cases, the values of passed parameters do not matter).

In a loop by the list of all timeframes, get the next timeseries object by the loop index. If failed to get the objector the size of a created timeframe history is zero (the timeframe is not used or was not created using the Create() method), move on to the next timeframe.

Next, call the method of updating the timeseries object of the same name we considered [in the previous article](https://www.mql5.com/en/articles/7594#node04).

The first version of the object class of all timeseries of a single symbol is ready. The current class functionality is sufficient to test working with several timeframes of a single symbol. In the future, we will refine it when creating a common timeseries collection class for multiple symbols.

In the CEngine class file located in \\MQL5\\Include\\DoEasy\\Engine.mqh, replace the string of including the timeseries object class file:

```
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Services\TimerCounter.mqh"
#include "Collections\HistoryCollection.mqh"
#include "Collections\MarketCollection.mqh"
#include "Collections\EventsCollection.mqh"
#include "Collections\AccountsCollection.mqh"
#include "Collections\SymbolsCollection.mqh"
#include "Collections\ResourceCollection.mqh"
#include "TradingControl.mqh"
#include "Objects\Series\Series.mqh"
//+------------------------------------------------------------------+
```

with the file of the object of timeseries for all used symbol periods:

```
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Services\TimerCounter.mqh"
#include "Collections\HistoryCollection.mqh"
#include "Collections\MarketCollection.mqh"
#include "Collections\EventsCollection.mqh"
#include "Collections\AccountsCollection.mqh"
#include "Collections\SymbolsCollection.mqh"
#include "Collections\ResourceCollection.mqh"
#include "TradingControl.mqh"
#include "Objects\Series\TimeSeries.mqh"
//+------------------------------------------------------------------+
```

Now the newly created class is visible from a library-based program.

### Testing

To test working with a single symbol on timeseries of different periods, we are going to use [the EA from the previous article](https://www.mql5.com/en/articles/7594#node05) and save it in \\MQL5\\Experts\\TestDoEasy\ **Part36\** under the name **TestDoEasyPart36.mq5**.

To test the class, create two versions of the test EA using [the conditional compilation directives](https://www.mql5.com/en/docs/basis/preprosessor/conditional_compilation).

- The first EA version creates two timeseries of the current symbol:

the first one is for M15 consisting of only two bars,

another one consists of ten bars — for the current chart period the EA is launched on.
- The second EA version creates all timeseries of the current symbol with the default values:

either 1000 bars for each timeseries, or the maximum possible value provided that the available number of bars for the timeseries is less than 1000.


In the block of the EA global variables, leave a single object of the timeseries class. Instead of the CSeries class, define the CTimeSeries class variable since the CSeries class objects are now a part of the CTimeSeries class object.

Remove one CSeries object:

```
//--- global variables
CEngine        engine;
CSeries        series;
CSeries        series_m1;
SDataButt      butt_data[TOTAL_BUTT];
```

also, rename the second object and define its type as CTimeSeries:

```
//--- global variables
CEngine        engine;
CTimeSeries    timeseries;
SDataButt      butt_data[TOTAL_BUTT];
```

At the very end of the EA's OnInit() handler, write the code block for creating timeseries depending on the existence/absence of the specified TIMESERIES\_ALL ID:

```
//--- Check playing a standard sound by macro substitution and a custom sound by description
   engine.PlaySoundByDescription(SND_OK);
   Sleep(600);
   engine.PlaySoundByDescription(TextByLanguage("Звук упавшей монетки 2","Falling coin 2"));

//--- Set a symbol for created timeseries
   timeseries.SetSymbol(Symbol());
//#define TIMESERIES_ALL
//--- Create two timeseries
   #ifndef TIMESERIES_ALL
      timeseries.SyncData(PERIOD_CURRENT,10);
      timeseries.Create(PERIOD_CURRENT);
      timeseries.SyncData(PERIOD_M15,2);
      timeseries.Create(PERIOD_M15);
//--- Create all timeseries
   #else
      timeseries.SyncAllData();
      timeseries.CreateAll();
   #endif
//--- Check created timeseries
   CArrayObj *list=timeseries.GetList();
   Print(TextByLanguage("Данные созданных таймсерий:","Data of created timeseries:"));
   for(int i=0;i<list.Total();i++)
     {
      CSeries *series_obj=timeseries.GetSeriesByIndex((uchar)i);
      if(series_obj==NULL || series_obj.AmountUsedData()==0 || series_obj.DataTotal()==0)
         continue;
      Print(
            DFUN,i,": ",series_obj.Symbol()," ",TimeframeDescription(series_obj.Timeframe()),
            ": AmountUsedData=",series_obj.AmountUsedData(),", DataTotal=",series_obj.DataTotal(),", Bars=",series_obj.Bars()
           );
     }
   Print("");
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

First, make sure to set a symbol name for the symbol timeseries class. Next, we have the defined and commented out ID as a [macro substitution](https://www.mql5.com/en/docs/basis/preprosessor/constant). Its presence/absence defines the version of the code to compile — the one for creation of two timeseries (if there is no ID) or the one for creating all timeseries (if there is an ID).

In general and as of the current version of the object class of all symbol timeseries, we need the following to create a timeseries:

1. set a symbol of all timeseries object,

2. call the method of setting the timeseries history depth and checking the timeseries data synchronization with the server,
3. create a timeseries based on the specified history depth.


Normally, the method of setting the history depth and checking server synchronization should be verified for a return result. The method can return false if failed to set the history depth or if the data is not synchronized with the server yet.

However, we can skip this check for now to perform a test on the current symbol. Most likely, the entire data will be available. Even if the data is not available, the timeseries is simply not created. It will be possible to simply restart the EA since the first access to the functions, initiating history data loading, launches the loading, and the data should become synchronized during the first EA launch.

After the necessary timeseries are created, display the full list of created timeseries to the journal to check if creation is successful.

To do this, receive the full list of all timeseries. In a loop, receive the next timeseries object from the list. If the timeseries is created (features the history depth and is filled with data), display the data in the journal.

At the very end of the OnTick() handler, insert the code block to update data of all created symbol timeseries:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- If working in the tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer();       // Working in the timer
      PressButtonsControl();  // Button pressing control
      EventsHandling();       // Working with events
     }
//--- If the trailing flag is set
   if(trailing_on)
     {
      TrailingPositions();    // Trailing positions
      TrailingOrders();       // Trailing of pending orders
     }
//--- Update created timeseries
   CArrayObj *list=timeseries.GetList();
   for(int i=0;i<list.Total();i++)
     {
      CSeries *series_obj=timeseries.GetSeriesByIndex((uchar)i);
      if(series_obj==NULL || series_obj.DataTotal()==0)
         continue;
      series_obj.Refresh();
      if(series_obj.IsNewBar(0))
        {
         Print(TextByLanguage("Новый бар на ","New bar on "),series_obj.Symbol()," ",TimeframeDescription(series_obj.Timeframe())," ",TimeToString(series_obj.Time(0)));
         if(series_obj.Timeframe()==Period())
            engine.PlaySoundByDescription(SND_NEWS);
        }
     }
  }
//+------------------------------------------------------------------+
```

Here, obtain the list of timeseries from the object of all symbol timeseries. In a loop by the list of timeseries,get the next timeseries object by the loop index. If failed to receive the timeseries or it has no data (bars), move on to the next timeseries of the next timeframe. If the timeseries object is received, update it. If the new bar flag is set for the timeseries, display the appropriate message (also play the news.wav sound for the current period timeseries)

Compile the EA (string 176 with the **#define TIMESERIES\_ALL** macro substitution definition should be commented out) — the EA version creating two timeseries is compiled.

Launch it in the terminal on M30 chart. The entries about the parameters of the two created timeseries  are displayed in the journal. The entries about opening new bars on M15 and M30 charts appear after some time:

```
Account 8550475: Artyom Trishkin (MetaQuotes Software Corp.) 10425.23 USD, 1:100, Hedge, Demo account MetaTrader 5
Work only with the current symbol. The number of symbols used: 1
Data of created timeseries:
OnInit: 8: EURUSD M15: AmountUsedData=2, DataTotal=2, Bars=5000
OnInit: 10: EURUSD M30: AmountUsedData=10, DataTotal=10, Bars=5030

New bar on EURUSD M15 2020.02.20 20:45
New bar on EURUSD M15 2020.02.20 21:00
New bar on EURUSD M30 2020.02.20 21:00
New bar on EURUSD M15 2020.02.20 21:15
New bar on EURUSD M15 2020.02.20 21:30
New bar on EURUSD M30 2020.02.20 21:30
```

Now uncomment the string 176 defining the **#define TIMESERIES\_ALL** macro substitution and compile the EA — the EA version creating all timeseries with the default values is generated.

Launch it on the symbol chart. The entries about the parameters of all created timeseries are displayed in the journal. The entries about opening new bars on created timeseries chart periods appear after some time:

```
Account 8550475: Artyom Trishkin (MetaQuotes Software Corp.) 10425.23 USD, 1:100, Hedge, Demo account MetaTrader 5
Work only with the current symbol. The number of symbols used: 1
Data of created timeseries:
OnInit: 0: EURUSD M1: AmountUsedData=1000, DataTotal=1000, Bars=5140
OnInit: 1: EURUSD M2: AmountUsedData=1000, DataTotal=1000, Bars=4010
OnInit: 2: EURUSD M3: AmountUsedData=1000, DataTotal=1000, Bars=3633
OnInit: 3: EURUSD M4: AmountUsedData=1000, DataTotal=1000, Bars=3445
OnInit: 4: EURUSD M5: AmountUsedData=1000, DataTotal=1000, Bars=3332
OnInit: 5: EURUSD M6: AmountUsedData=1000, DataTotal=1000, Bars=3256
OnInit: 6: EURUSD M10: AmountUsedData=1000, DataTotal=1000, Bars=3106
OnInit: 7: EURUSD M12: AmountUsedData=1000, DataTotal=1000, Bars=3068
OnInit: 8: EURUSD M15: AmountUsedData=1000, DataTotal=1000, Bars=5004
OnInit: 9: EURUSD M20: AmountUsedData=1000, DataTotal=1000, Bars=2993
OnInit: 10: EURUSD M30: AmountUsedData=1000, DataTotal=1000, Bars=5032
OnInit: 11: EURUSD H1: AmountUsedData=1000, DataTotal=1000, Bars=5352
OnInit: 12: EURUSD H2: AmountUsedData=1000, DataTotal=1000, Bars=6225
OnInit: 13: EURUSD H3: AmountUsedData=1000, DataTotal=1000, Bars=6212
OnInit: 14: EURUSD H4: AmountUsedData=1000, DataTotal=1000, Bars=5292
OnInit: 15: EURUSD H6: AmountUsedData=1000, DataTotal=1000, Bars=5182
OnInit: 16: EURUSD H8: AmountUsedData=1000, DataTotal=1000, Bars=5443
OnInit: 17: EURUSD H12: AmountUsedData=1000, DataTotal=1000, Bars=5192
OnInit: 18: EURUSD D1: AmountUsedData=1000, DataTotal=1000, Bars=5080
OnInit: 19: EURUSD W1: AmountUsedData=1000, DataTotal=1000, Bars=2562
OnInit: 20: EURUSD MN1: AmountUsedData=589, DataTotal=589, Bars=589

New bar on EURUSD M1 2020.02.20 21:41
New bar on EURUSD M1 2020.02.20 21:42
New bar on EURUSD M2 2020.02.20 21:42
New bar on EURUSD M3 2020.02.20 21:42
New bar on EURUSD M6 2020.02.20 21:42
New bar on EURUSD M1 2020.02.20 21:43
New bar on EURUSD M1 2020.02.20 21:44
New bar on EURUSD M2 2020.02.20 21:44
New bar on EURUSD M4 2020.02.20 21:44
New bar on EURUSD M1 2020.02.20 21:45
New bar on EURUSD M3 2020.02.20 21:45
New bar on EURUSD M5 2020.02.20 21:45
New bar on EURUSD M15 2020.02.20 21:45
```

Launch the EA in the tester's visual mode on M5:

![](https://c.mql5.com/2/38/LuUj01oIsa.gif)

First, the tester downloads historical data for all timeframes, then the EA displays data of the created timeseries. The messages are then sent to the journal notifying of opening new bars on the created timeseries during the test.

All works as intended at this stage of creating the functionality for working with a single symbol timeseries.

### What's next?

In the next article, we will create the common timeseries collection class storing the required amount of data for different symbols and their timeframes.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7627#node00)

**Previous articles within the series:**

[Timeseries in DoEasy library (part 35): Bar object and symbol timeseries list](https://www.mql5.com/en/articles/7594)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7627](https://www.mql5.com/ru/articles/7627)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7627.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7627/mql5.zip "Download MQL5.zip")(3686.04 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7627/mql4.zip "Download MQL4.zip")(3686.04 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/342992)**
(23)


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
28 Feb 2020 at 15:26

**Sergey Pavlov:**

Artyom. where did you get that from?

I was talking about trolling talent.


![Sergey Pavlov](https://c.mql5.com/avatar/2010/2/4B7AECD8-6F67.jpg)

**[Sergey Pavlov](https://www.mql5.com/en/users/dc2008)**
\|
28 Feb 2020 at 15:29

**Artyom Trishkin:**

No one has called you a second-class person, except yourself.

But I have no desire to discuss the topic of paying for articles. One thing I will say is that it is good that they are paid. Without it, there would not be an overwhelming number of articles here. And I am not ashamed to be paid for my work. If you wanted to discuss it, I think I have answered you.

Too bad... You still don't get it....

Good luck to you.

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
28 Feb 2020 at 16:53

**Fast235:**

Artem, I have great respect for your hard work (capitalised exclusively)

If you think I'm a nub, believe me, I've read here in aggregate, probably more than any of the participants.

actually written well, it's just scary to use such giant shupals instead of - buy/lot/sl/tk/slider/comment.

That's what MQL5 is accused of.

Thanks. The library implements what is most often required. Just buy ... - you don't need a library for that.


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
28 Feb 2020 at 16:53

**Sergey Pavlov:**

That's too bad. You still don't understand...

Good luck to you.

I apologise if I offended you. Thanks for your help.


![Fast235](https://c.mql5.com/avatar/2019/11/5DDBA4DA-BF3F.png)

**[Fast235](https://www.mql5.com/en/users/igann)**
\|
29 Feb 2020 at 02:49

**Artyom Trishkin:**

Thank you. The library implements what is most often needed. Just buy ... - you don't need a library for that.

Yes, I remember, working with history,

the developer (MQL5) is a big enthusiast, he made it for himself.

Maybe there is no better solution

![Continuous Walk-Forward Optimization (Part 4): Optimization Manager (Auto Optimizer)](https://c.mql5.com/2/38/MQL5-avatar-continuous_optimization.png)[Continuous Walk-Forward Optimization (Part 4): Optimization Manager (Auto Optimizer)](https://www.mql5.com/en/articles/7538)

The main purpose of the article is to describe the mechanism of working with our application and its capabilities. Thus the article can be treated as an instruction on how to use the application. It covers all possible pitfalls and specifics of the application usage.

![Forecasting Time Series (Part 1): Empirical Mode Decomposition (EMD) Method](https://c.mql5.com/2/38/mql5-avatar-emd.png)[Forecasting Time Series (Part 1): Empirical Mode Decomposition (EMD) Method](https://www.mql5.com/en/articles/7601)

This article deals with the theory and practical use of the algorithm for forecasting time series, based on the empirical decomposition mode. It proposes the MQL implementation of this method and presents test indicators and Expert Advisors.

![Projects assist in creating profitable trading robots! Or at least, so it seems](https://c.mql5.com/2/39/mql5-avatar-thumbs_up.png)[Projects assist in creating profitable trading robots! Or at least, so it seems](https://www.mql5.com/en/articles/7863)

A big program starts with a small file, which then grows in size as you keep adding more functions and objects. Most robot developers utilize include files to handle this problem. However, there is a better solution: start developing any trading application in a project. There are so many reasons to do so.

![Timeseries in DoEasy library (part 35): Bar object and symbol timeseries list](https://c.mql5.com/2/38/MQL5-avatar-doeasy-library.png)[Timeseries in DoEasy library (part 35): Bar object and symbol timeseries list](https://www.mql5.com/en/articles/7594)

This article starts a new series about the creation of the DoEasy library for easy and fast program development. In the current article, we will implement the library functionality for accessing and working with symbol timeseries data. We are going to create the Bar object storing the main and extended timeseries bar data, and place bar objects to the timeseries list for convenient search and sorting of the objects.

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/7627&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083169373704689235)

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