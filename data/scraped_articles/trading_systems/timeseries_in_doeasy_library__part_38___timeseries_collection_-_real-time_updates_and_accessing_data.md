---
title: Timeseries in DoEasy library (part 38): Timeseries collection - real-time updates and accessing data from the program
url: https://www.mql5.com/en/articles/7695
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:51:45.432432
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/7695&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083164962773276223)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/7695#node01)
- [Improving timeseries classes](https://www.mql5.com/en/articles/7695#node02)
- ["New tick" class and data update](https://www.mql5.com/en/articles/7695#node03)
- [Testing](https://www.mql5.com/en/articles/7695#node04)
- [What's next?](https://www.mql5.com/en/articles/7695#node05)


### Concept

In the previous articles devoted to creating timeseries of any chart periods and of any symbols, we have created a full-fledged [timeseries collection class of all symbols used in the program](https://www.mql5.com/en/articles/7663) and learned to fill timeseries with history data for its fast search and sorting.

Such a tool will allow us to search and compare various combinations of price data in history. But we also need to think about updating the current data which should be done at each new tick for each used symbol.

Even the simplest version allows us to update all timeseries in the program's [OnTimer()](https://www.mql5.com/en/docs/event_handlers/ontimer) millisecond handler. However, this gives rise to a question whether the timeseries data should always be updated exactly according to the timer counter. After all, the data is changed in the program upon arrival of a new tick. It would be wrong to simply update the data regardless of a new tick arrival — this would be irrational in terms of performance.

While we always know about a new tick arrival in the EA's [OnTick()](https://www.mql5.com/en/docs/event_handlers/ontick) handlers or indicator's [OnCalculate()](https://www.mql5.com/en/docs/event_handlers/oncalculate) ones on the current symbol, this is not the case for any other symbol tracked by the program launched on another symbol. This task requires tracking the necessary events [in an EA](https://www.mql5.com/en/docs/runtime/event_fire#newtick) or [an indicator](https://www.mql5.com/en/docs/runtime/event_fire#calculate).

Here, the simplest possible option satisfying the current library needs is comparing the previous tick time with the current one. If the previous tick time is different from the current one, a new tick is considered to have arrived on a symbol tracked by the program but not "native" to it (the program is launched on another symbol's chart).

Let's slightly improve the existing classes before developing the "New tick" class and the real-time update of all timeseries used in the program.

### Improving timeseries classes

First of all, the **Datas.mqh** file receives the library's new message index:

```
//--- CTimeSeries
   MSG_LIB_TEXT_TS_TEXT_FIRST_SET_SYMBOL,             // First, set a symbol using SetSymbol()
   MSG_LIB_TEXT_TS_TEXT_IS_NOT_USE,                   // Timeseries is not used. Set the flag using SetAvailable()
   MSG_LIB_TEXT_TS_TEXT_UNKNOWN_TIMEFRAME,            // Unknown timeframe
```

and the message text corresponding to the newly added index:

```
   {"Сначала нужно установить символ при помощи SetSymbol()","First you need to set Symbol using SetSymbol()"},
   {"Таймсерия не используется. Нужно установить флаг использования при помощи SetAvailable()","Timeseries not used. Need to set usage flag using SetAvailable()"},
   {"Неизвестный таймфрейм","Unknown timeframe"},
```

The [CBaseObj](https://www.mql5.com/en/articles/7071) class of the base object of all library objects features two variables:

```
//+------------------------------------------------------------------+
//| Base object class for all library objects                        |
//+------------------------------------------------------------------+
class CBaseObj : public CObject
  {
protected:
   ENUM_LOG_LEVEL    m_log_level;                              // Logging level
   ENUM_PROGRAM_TYPE m_program;                                // Program type
   bool              m_first_start;                            // First launch flag
   bool              m_use_sound;                              // Flag of playing the sound set for an object
   bool              m_available;                              // Flag of using a descendant object in the program
   int               m_global_error;                           // Global error code
   long              m_chart_id_main;                          // Control program chart ID
   long              m_chart_id;                               // Chart ID
   string            m_name;                                   // Object name
   string            m_folder_name;                            // Name of the folder storing CBaseObj descendant objects
   string            m_sound_name;                             // Object sound file name
   int               m_type;                                   // Object type (corresponds to the collection IDs)

public:
```

The **m\_chart\_id\_main** variable stores the control program chart ID — this is a chart of a symbol the program has been launched on. The chart is to get all events registered in the library collections and objects.

The **m\_chart\_id** stores the ID of the chart the object derived from the CBaseObj class is somehow related to. This property is not used anywhere yet. Its time will come later.

Since we added the m\_chart\_id\_main variable later than m\_chart\_id, all messages are sent to the chart ID set in the m\_chart\_id variable. I have fixed this. Now all current chart IDs are set in the m\_chart\_id\_main variable. All classes sending messages from the library to the control program chart have been changed — all instances of "m\_chart\_id" have been replaced with "m\_chart\_id\_main".

Such changes have been made to all event classes from the \\MQL5\\Include\\DoEasy\\Objects\ **Events\** folder, as well as to the files of the AccountsCollection.mqh, EventsCollection.mqh and SymbolsCollection.mqh collection classes.

You can see all the changes in the attached files.

To display the data of the specified bar from the timeseries collection, add the text description of the CBar class bar parameters to \\MQL5\\Include\\DoEasy\\Objects\\Series\\Bar.mqh.

In the code block containing the description of the object properties, declare the method for creating the text description of the bar parameters:

```
//+------------------------------------------------------------------+
//| Descriptions of bar object properties                            |
//+------------------------------------------------------------------+
//--- Get description of a bar's (1) integer, (2) real and (3) string properties
   string            GetPropertyDescription(ENUM_BAR_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_BAR_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_BAR_PROP_STRING property);

//--- Return the bar type description
   string            BodyTypeDescription(void)  const;
//--- Send description of bar properties to the journal (full_prop=true - all properties, false - only supported ones)
   void              Print(const bool full_prop=false);
//--- Display a short bar description in the journal
   virtual void      PrintShort(void);
//--- Return the (1) short name and (2) description of bar object parameters
   virtual string    Header(void);
   string            ParameterDescription(void);
//---
  };
//+------------------------------------------------------------------+
```

Beyond the class body, implement the method for creating the text description of the bar parameters and change implementation of the method displaying the short bar description in the journal:

```
//+------------------------------------------------------------------+
//| Return the description of the bar object parameters              |
//+------------------------------------------------------------------+
string CBar::ParameterDescription(void)
  {
   int dg=(this.m_digits>0 ? this.m_digits : 1);
   return
     (
      ::TimeToString(this.Time(),TIME_DATE|TIME_MINUTES|TIME_SECONDS)+", "+
      "O: "+::DoubleToString(this.Open(),dg)+", "+
      "H: "+::DoubleToString(this.High(),dg)+", "+
      "L: "+::DoubleToString(this.Low(),dg)+", "+
      "C: "+::DoubleToString(this.Close(),dg)+", "+
      "V: "+(string)this.VolumeTick()+", "+
      (this.VolumeReal()>0 ? "R: "+(string)this.VolumeReal()+", " : "")+
      this.BodyTypeDescription()
     );
  }
//+------------------------------------------------------------------+
//| Display a short bar description in the journal                   |
//+------------------------------------------------------------------+
void CBar::PrintShort(void)
  {
   ::Print(this.Header(),": ",this.ParameterDescription());
  }
//+------------------------------------------------------------------+
```

Here, I have simply removed the parameter description code from the method displaying the bar parameters in the journal and placed it in the new method returning the text message. When displaying the bar parameters in the journal, display the composite message consisting of a short bar object name and its parameters whose text description is now generated in the new method ParameterDescription().

In order to update the "non-native" timeseries (that are not the ones the program is launched on), we decided to create the "New tick" class and update data of such symbols only upon arrival of the "New tick" event for each symbol used in the program.

### "New tick" class and data update

In \\MQL5\\Include\\DoEasy\ **Objects\**, create the **Ticks\** folder featuring the **NewTickObj.mqh** file of the CNewTickObj class derived from the base object of all CBaseObj library objects (whose file is included into the class file) and fill in the necessary data:

```
//+------------------------------------------------------------------+
//|                                                   NewTickObj.mqh |
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
//| "New tick" class                                                 |
//+------------------------------------------------------------------+
class CNewTickObj : public CBaseObj
  {
private:
   MqlTick           m_tick;                          // Structure of the current prices
   MqlTick           m_tick_prev;                     // Structure of the current prices during the previous check
   string            m_symbol;                        // Object symbol
   bool              m_new_tick;                      // New tick flag
public:
//--- Set a symbol
   void              SetSymbol(const string symbol)   { this.m_symbol=symbol;             }
//--- Return the new tick flag
   bool              IsNewTick(void);
//--- Update price data in the tick structure and set the "New tick" event flag if necessary
   void              Refresh(void)                    { this.m_new_tick=this.IsNewTick(); }
//--- Constructors
                     CNewTickObj(void){;}
                     CNewTickObj(const string symbol);
  };
//+------------------------------------------------------------------+
```

The **m\_tick** variable stores price data of the last arrived tick.

The **m\_tick\_prev** variable stores price data of the previous tick.

The **m\_symbol** variable stores a symbol whose new tick is to be tracked.

The new tick flag in the **m\_new\_tick** variable is to be used later.

For the current library needs, the "New tick" event on a symbol is defined by the **IsNewTick()** method:

```
//+------------------------------------------------------------------+
//| Return the new tick flag                                         |
//+------------------------------------------------------------------+
bool CNewTickObj::IsNewTick(void)
  {
//--- If failed to get the current prices to the tick structure, return 'false'
   if(!::SymbolInfoTick(this.m_symbol,this.m_tick))
      return false;
//--- If this is the first launch, copy data of the obtained tick to the previous tick data
//--- reset the first launch flag and return 'false'
   if(this.m_first_start)
     {
      this.m_tick_prev=this.m_tick;
      this.m_first_start=false;
      return false;
     }
//--- If the time of a new tick is not equal to the time of a tick during the previous check -
//--- copy data of the obtained tick to the previous tick data and return 'true'
   if(this.m_tick.time_msc!=this.m_tick_prev.time_msc)
     {
      this.m_tick_prev=this.m_tick;
      return true;
     }
//--- In all other cases, return 'false'
   return false;
  }
//+------------------------------------------------------------------+
```

The class has two defined constructors:

- the default constructor with no parameters is used to define the "New tick" object within another class. In this case, use the SetSymbol() class method to set a symbol for the CNewTickObj class object the "New tick" events are defined for.
- the paramteric constructor is used to create the class object via the [new](https://www.mql5.com/en/docs/basis/operators/newoperator) operator. In this case, a symbol the object is being created for can be specified immediately when creating the object.


```
//+------------------------------------------------------------------+
//| Parametric constructor CNewTickObj                               |
//+------------------------------------------------------------------+
CNewTickObj::CNewTickObj(const string symbol) : m_symbol(symbol)
  {
//--- Reset the structures of the new and previous ticks
   ::ZeroMemory(this.m_tick);
   ::ZeroMemory(this.m_tick_prev);
//--- If managed to get the current prices to the tick structure,
//--- copy data of the obtained tick to the previous tick data and reset the first launch flag
  if(::SymbolInfoTick(this.m_symbol,this.m_tick))
     {
      this.m_tick_prev=this.m_tick;
      this.m_first_start=false;
     }
  }
//+------------------------------------------------------------------+
```

This is the entire class of the new tick object. The idea is simple: get the prices to the tick structure and compare the time of the arrived tick with the time of the previous one.

If these times are not equal, then a new tick has arrived.

Ticks may be skipped in the EAs, but this is not important here. We are able to track a new tick on a "non-native" symbol in the timer in order to update the data only when a new tick arrives rather than doing it constantly by the timer.

If an indicator tracks all the ticks which can arrive in batches, the update of the current timeseries data for a symbol the indicator is launched on should be done in the OnCalculate() handler. The new ticks for "non-native" symbols are tracked in the timer (new tick events for a "non-native" symbol cannot be received in OnOnCalculate()), therefore, it would be sufficient to track only the time difference between the new and previous ticks for "non-native" symbols to update the timeseries data in time.

Let the [CSeries](https://www.mql5.com/en/articles/7594#node04) timeseries object send its "New bar" event to the control program. This will allow us to get such events from any timeseries in the program and respond to them.

At the end of the **Defines.mqh** listing file, add the new enumeration with the list of possible timeseries object events:

```
//+------------------------------------------------------------------+
//| List of possible timeseries events                               |
//+------------------------------------------------------------------+
enum ENUM_SERIES_EVENT
  {
   SERIES_EVENTS_NO_EVENT = SYMBOL_EVENTS_NEXT_CODE,        // no event
   SERIES_EVENTS_NEW_BAR,                                   // "New bar" event
  };
#define SERIES_EVENTS_NEXT_CODE  (SERIES_EVENTS_NEW_BAR+1)  // Code of the next event after the "New bar" event
//+------------------------------------------------------------------+
```

Here we have only two states of the timeseries events yet: "No event" and "New bar" event. We need these enumeration constants to search for the bar object [by specified properties](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjsearch) in the bar collection list (in the CSeries timeseries).

Since the timeseries objects are updated in the library timer, add the parameters of the timeseries object collection update timer to the **Defines.mqh** listing file together with the timeseries collection list ID:

```
//--- Trading class timer parameters
#define COLLECTION_REQ_PAUSE           (300)                      // Trading class timer pause in milliseconds
#define COLLECTION_REQ_COUNTER_STEP    (16)                       // Trading class timer counter increment
#define COLLECTION_REQ_COUNTER_ID      (5)                        // Trading class timer counter ID
//--- Parameters of the timeseries collection timer
#define COLLECTION_TS_PAUSE            (32)                       // Timeseries collection timer pause in milliseconds
#define COLLECTION_TS_COUNTER_STEP     (16)                       // Account timer counter increment
#define COLLECTION_TS_COUNTER_ID       (6)                        // Timeseries timer counter ID
//--- Collection list IDs
#define COLLECTION_HISTORY_ID          (0x777A)                   // Historical collection list ID
#define COLLECTION_MARKET_ID           (0x777B)                   // Market collection list ID
#define COLLECTION_EVENTS_ID           (0x777C)                   // Event collection list ID
#define COLLECTION_ACCOUNT_ID          (0x777D)                   // Account collection list ID
#define COLLECTION_SYMBOLS_ID          (0x777E)                   // Symbol collection list ID
#define COLLECTION_SERIES_ID           (0x777F)                   // Timeseries collection list ID
//--- Data parameters for file operations
```

We have considered the collection timer parameters when creating the [CEngine library base object](https://www.mql5.com/en/articles/5687#node02), while the purpose of collection IDs have been described when [re-arranging the library structure](https://www.mql5.com/en/articles/6211#node01).

Assign the timeseries collection ID to the bar object right away since the timeseries object is a list containing pointers to bar objects belonging to the list.

Open \\MQL5\\Include\\DoEasy\\Objects\\Series\ **Bar.mqh** once again and add the object type to both constructors:

```
//+------------------------------------------------------------------+
//| Constructor 1                                                    |
//+------------------------------------------------------------------+
CBar::CBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index)
  {
   this.m_type=COLLECTION_SERIES_ID;
   MqlRates rates_array[1];
   this.SetSymbolPeriod(symbol,timeframe,index);
   ::ResetLastError();
//--- If failed to write bar data to the MqlRates array by index or set the time to the time structure,
//--- display an error message, create and fill the structure with zeros, and write it to the rates_array array
   if(::CopyRates(symbol,timeframe,index,1,rates_array)<1 || !::TimeToStruct(rates_array[0].time,this.m_dt_struct))
     {
      int err_code=::GetLastError();
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_BAR_FAILED_GET_BAR_DATA),". ",CMessage::Text(MSG_LIB_SYS_ERROR)," ",CMessage::Text(err_code)," ",CMessage::Retcode(err_code));
      MqlRates err={0};
      rates_array[0]=err;
     }
//--- Set the bar properties
   this.SetProperties(rates_array[0]);
  }
//+------------------------------------------------------------------+
//| Constructor 2                                                    |
//+------------------------------------------------------------------+
CBar::CBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index,const MqlRates &rates)
  {
   this.m_type=COLLECTION_SERIES_ID;
   this.SetSymbolPeriod(symbol,timeframe,index);
   ::ResetLastError();
//--- If failed to set time to the time structure, display the error message,
//--- create and fill the structure with zeros, set the bar properties from this structure and exit
   if(!::TimeToStruct(rates.time,this.m_dt_struct))
     {
      int err_code=::GetLastError();
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_BAR_FAILED_GET_BAR_DATA),". ",CMessage::Text(MSG_LIB_SYS_ERROR)," ",CMessage::Text(err_code)," ",CMessage::Retcode(err_code));
      MqlRates err={0};
      this.SetProperties(err);
      return;
     }
//--- Set the bar properties
   this.SetProperties(rates);
  }
//+------------------------------------------------------------------+
```

**Now, let's improve the [CSeries](https://www.mql5.com/en/articles/7594#node04) timeseries object class located in \\MQL5\\Include\\DoEasy\\Objects\\Series\\Series.mqh.**

In the public class section, declare the new method for sending an event to the control program chart:

```
//--- (1) Create and (2) update the timeseries list
   int               Create(const uint required=0);
   void              Refresh(const datetime time=0,
                             const double open=0,
                             const double high=0,
                             const double low=0,
                             const double close=0,
                             const long tick_volume=0,
                             const long volume=0,
                             const int spread=0);

//--- Create and send the "New bar" event to the control program chart
   void              SendEvent(void);

//--- Return the timeseries name
   string            Header(void);
//--- Display (1) the timeseries description and (2) the brief timeseries description in the journal
   void              Print(void);
   void              PrintShort(void);

//--- Constructors
                     CSeries(void);
                     CSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint required=0);
  };
//+------------------------------------------------------------------+
```

In the end of the class listing, implement the declared method:

```
//+------------------------------------------------------------------+
//| Create and send the "New bar" event                              |
//| to the control program chart                                     |
//+------------------------------------------------------------------+
void CSeries::SendEvent(void)
  {
   ::EventChartCustom(this.m_chart_id_main,SERIES_EVENTS_NEW_BAR,this.Time(0),this.Timeframe(),this.Symbol());
  }
//+------------------------------------------------------------------+
```

Here we create [and send an event to the control program chart](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom). The event consists of:

event receiver chart ID,

event ID (New bar),

send the new bar opening time as a **long** event parameter,

send the timeframe of the chart where the event has occurred as a **double** event parameter and

send a name of a symbol (on whose timeseries the event has occurred) as a **string** parameter.

Add the check of the flag indicating the use of the timeseries in the program to the timeseries data synchronization method:

```
//+------------------------------------------------------------------+
//|Synchronize symbol and timeframe data with server data            |
//+------------------------------------------------------------------+
bool CSeries::SyncData(const uint required,const uint rates_total)
  {
//--- If the timeseries is not used, notify of that and exit
   if(!this.m_available)
     {
      ::Print(DFUN,this.m_symbol," ",TimeframeDescription(this.m_timeframe),": ",CMessage::Text(MSG_LIB_TEXT_TS_TEXT_IS_NOT_USE));
      return false;
     }
//--- If managed to obtain the available number of bars in the timeseries
```

In other words, if the flag of the timeseries usage in the program is not set, there is no need to synchronize it. There may also be a situation when we need the timeseries while the usage flag is not set. Therefore, the appropriate message is sent to the journal.

Implement the same check to the timeseries creation method:

```
//+------------------------------------------------------------------+
//| Create the timeseries list                                       |
//+------------------------------------------------------------------+
int CSeries::Create(const uint required=0)
  {
//--- If the timeseries is not used, notify of that and return zero
   if(!this.m_available)
     {
      ::Print(DFUN,this.m_symbol," ",TimeframeDescription(this.m_timeframe),": ",CMessage::Text(MSG_LIB_TEXT_TS_TEXT_IS_NOT_USE));
      return 0;
     }
//--- If the required history depth is not set for the list yet,
```

The method returning the bar object by the timeseries index has been revised in the class. Previously, the method looked as follows:

```
//+------------------------------------------------------------------+
//| Return the bar object by index in the timeseries                 |
//+------------------------------------------------------------------+
CBar *CSeries::GetBarBySeriesIndex(const uint index)
  {
   CArrayObj *list=this.GetList(BAR_PROP_INDEX,index);
   return(list==NULL || list.Total()==0 ? NULL : list.At(0));
  }
//+------------------------------------------------------------------+
```

In other words, a new list featuring a copy of the necessary bar has been created and that copy has been returned. This is sufficient if we simply want to receive data of a requested bar, but if we need to change the bar properties, then this method does not work as the changes are made to the properties of the bar copy rather than to the properties of the original object.

Since we want the current bar to update in real time upon a new tick arrival, I have changed the method to return the pointer to the original bar object located in the bar collection list rather than to the bar from the list copy:

```
//+------------------------------------------------------------------+
//| Return the bar object by index in the timeseries                 |
//+------------------------------------------------------------------+
CBar *CSeries::GetBarBySeriesIndex(const uint index)
  {
   CBar *tmp=new CBar(this.m_symbol,this.m_timeframe,index);
   if(tmp==NULL)
      return NULL;
   this.m_list_series.Sort(SORT_BY_BAR_INDEX);
   int idx=this.m_list_series.Search(tmp);
   delete tmp;
   CBar *bar=this.m_list_series.At(idx);
   return(bar!=NULL ? bar : NULL);
  }
//+------------------------------------------------------------------+
```

Here we create the temporary bar object with a symbol and period of the current timeseries object chart and the bar index passed to the method. The bar index in the chart timeseries is necessary to search for the same object in the timeseries listsorted by bar indices. While searching for a bar with the same timeseries index, we get its index in the list (this index is used to get the pointer to the bar object in the list) and return the pointer to the object.

Now the method returns the pointer to the original bar object in the timeseries list. It can be changed during real-time data update.

**Now improve the [CTimeSeries](https://www.mql5.com/en/articles/7627)** timeseries object class to track new ticks and update data when defining such an event. The class object is a set of timeseries of all used chart periods of a single symbol. This means the object is the best place for the "New tick" class object since obtaining a new tick by the CTimeSeries timeseries object symbol launches the update of the [CSeries](https://www.mql5.com/en/articles/7594#node04) timeseries object data of all periods belonging to the object.

Include the "New tick" object class file to the timeseries object class file. In the private class section, define the "New tick" class object.

In the public section of the class, add the method returning the new tick flag on the current timeseries object symbol:

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
#include "..\Ticks\NewTickObj.mqh"
//+------------------------------------------------------------------+
//| Symbol timeseries class                                          |
//+------------------------------------------------------------------+
class CTimeSeries : public CBaseObj
  {
private:
   string            m_symbol;                                             // Timeseries symbol
   CNewTickObj       m_new_tick;                                           // "New tick" object
   CArrayObj         m_list_series;                                        // List of timeseries by timeframes
   datetime          m_server_firstdate;                                   // The very first date in history by a server symbol
   datetime          m_terminal_firstdate;                                 // The very first date in history by a symbol in the client terminal
//--- Return (1) the timeframe index in the list and (2) the timeframe by the list index
   char              IndexTimeframe(const ENUM_TIMEFRAMES timeframe) const { return IndexEnumTimeframe(timeframe)-1;                            }
   ENUM_TIMEFRAMES   TimeframeByIndex(const uchar index)             const { return TimeframeByEnumIndex(uchar(index+1));                       }
//--- Set the very first date in history by symbol on the server and in the client terminal
   void              SetTerminalServerDate(void)
                       {
                        this.m_server_firstdate=(datetime)::SeriesInfoInteger(this.m_symbol,::Period(),SERIES_SERVER_FIRSTDATE);
                        this.m_terminal_firstdate=(datetime)::SeriesInfoInteger(this.m_symbol,::Period(),SERIES_TERMINAL_FIRSTDATE);
                       }
public:
//--- Return (1) oneself, (2) the full list of timeseries, (3) specified timeseries object and (4) timeseries object by index
   CTimeSeries      *GetObject(void)                                       { return &this;                                                      }
   CArrayObj        *GetListSeries(void)                                   { return &this.m_list_series;                                        }
   CSeries          *GetSeries(const ENUM_TIMEFRAMES timeframe)            { return this.m_list_series.At(this.IndexTimeframe(timeframe));      }
   CSeries          *GetSeriesByIndex(const uchar index)                   { return this.m_list_series.At(index);                               }
//--- Set/return timeseries symbol
   void              SetSymbol(const string symbol)                        { this.m_symbol=(symbol==NULL || symbol=="" ? ::Symbol() : symbol);  }
   string            Symbol(void)                                    const { return this.m_symbol;                                              }
//--- Set the history depth (1) of a specified timeseries and (2) of all applied symbol timeseries
   bool              SetRequiredUsedData(const ENUM_TIMEFRAMES timeframe,const uint required=0,const int rates_total=0);
   bool              SetRequiredAllUsedData(const uint required=0,const int rates_total=0);
//--- Return the flag of data synchronization with the server data of the (1) specified timeseries, (2) all timeseries
   bool              SyncData(const ENUM_TIMEFRAMES timeframe,const uint required=0,const int rates_total=0);
   bool              SyncAllData(const uint required=0,const int rates_total=0);
//--- Return the very first date in history by symbol (1) on the server, (2) in the client terminal and (3) the new tick flag
   datetime          ServerFirstDate(void)                           const { return this.m_server_firstdate;                                    }
   datetime          TerminalFirstDate(void)                         const { return this.m_terminal_firstdate;                                  }
   bool              IsNewTick(void)                                       { return this.m_new_tick.IsNewTick();                                }
//--- Create (1) the specified timeseries list and (2) all timeseries lists
   bool              Create(const ENUM_TIMEFRAMES timeframe,const uint required=0);
   bool              CreateAll(const uint required=0);
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

//--- Compare CTimeSeries objects (by symbol)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Display (1) description and (2) short symbol timeseries description in the journal
   void              Print(const bool created=true);
   void              PrintShort(const bool created=true);

//--- Constructors
                     CTimeSeries(void){;}
                     CTimeSeries(const string symbol);
  };
//+------------------------------------------------------------------+
```

The **IsNewTick()** method returns the result of requesting data on the new tick from the **m\_new\_tick** "New tick" object.

To let the "New tick" class object know about the symbol whose data is to be returned, we should set the symbol for the "New tick" class object in the class constructor and immediately update data for reading the current tick prices:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CTimeSeries::CTimeSeries(const string symbol) : m_symbol(symbol)
  {
   this.m_list_series.Clear();
   this.m_list_series.Sort();
   for(int i=0;i<21;i++)
     {
      ENUM_TIMEFRAMES timeframe=this.TimeframeByIndex((uchar)i);
      CSeries *series_obj=new CSeries(this.m_symbol,timeframe);
      this.m_list_series.Add(series_obj);
     }
   this.SetTerminalServerDate();
   this.m_new_tick.SetSymbol(this.m_symbol);
   this.m_new_tick.Refresh();
  }
//+------------------------------------------------------------------+
```

We are now going to check the timeseries usage flag in the methods returning the data synchronization flag. If the flag is unchecked, the timeseries is not used in the program and should not be handled:

```
//+------------------------------------------------------------------+
//| Return the flag of data synchronization                          |
//| with the server data                                             |
//+------------------------------------------------------------------+
bool CTimeSeries::SyncData(const ENUM_TIMEFRAMES timeframe,const uint required=0,const int rates_total=0)
  {
   if(this.m_symbol==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_TS_TEXT_FIRST_SET_SYMBOL));
      return false;
     }
   CSeries *series_obj=this.m_list_series.At(this.IndexTimeframe(timeframe));
   if(series_obj==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_TS_FAILED_GET_SERIES_OBJ),this.m_symbol," ",TimeframeDescription(timeframe));
      return false;
     }
   if(!series_obj.IsAvailable())
      return false;
   return series_obj.SyncData(required,rates_total);
  }
//+------------------------------------------------------------------+
//| Return the flag of data synchronization                          |
//| of all timeseries with the server data                           |
//+------------------------------------------------------------------+
bool CTimeSeries::SyncAllData(const uint required=0,const int rates_total=0)
  {
   if(this.m_symbol==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_TS_TEXT_FIRST_SET_SYMBOL));
      return false;
     }
   bool res=true;
   for(int i=0;i<21;i++)
     {
      CSeries *series_obj=this.m_list_series.At(i);
      if(series_obj==NULL || !series_obj.IsAvailable())
         continue;
      res &=series_obj.SyncData(required,rates_total);
     }
   return res;
  }
//+------------------------------------------------------------------+
```

Forcibly set the timeseries usage flag in the timeseries creation methods:

```
//+------------------------------------------------------------------+
//| Create a specified timeseries list                               |
//+------------------------------------------------------------------+
bool CTimeSeries::Create(const ENUM_TIMEFRAMES timeframe,const uint required=0)
  {
   CSeries *series_obj=this.m_list_series.At(this.IndexTimeframe(timeframe));
   if(series_obj==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_TS_FAILED_GET_SERIES_OBJ),this.m_symbol," ",TimeframeDescription(timeframe));
      return false;
     }
   if(series_obj.RequiredUsedData()==0)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_BAR_TEXT_FIRS_SET_AMOUNT_DATA));
      return false;
     }
   series_obj.SetAvailable(true);
   return(series_obj.Create(required)>0);
  }
//+------------------------------------------------------------------+
//| Create all timeseries lists                                      |
//+------------------------------------------------------------------+
bool CTimeSeries::CreateAll(const uint required=0)
  {
   bool res=true;
   for(int i=0;i<21;i++)
     {
      CSeries *series_obj=this.m_list_series.At(i);
      if(series_obj==NULL || series_obj.RequiredUsedData()==0)
         continue;
      series_obj.SetAvailable(true);
      res &=(series_obj.Create(required)>0);
     }
   return res;
  }
//+------------------------------------------------------------------+
```

In the timeseries update methods (in case the "New bar" event is detected in it), add sending a message about the event to the control program chart using the **SendEvent()** method of the CSeries timeseries object considered above:

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
   if(series_obj.IsNewBar(time))
     {
      series_obj.SendEvent();
      this.SetTerminalServerDate();
     }
  }
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
   bool upd=false;
   for(int i=0;i<21;i++)
     {
      CSeries *series_obj=this.m_list_series.At(i);
      if(series_obj==NULL || series_obj.DataTotal()==0)
         continue;
      series_obj.Refresh(time,open,high,low,close,tick_volume,volume,spread);
      if(series_obj.IsNewBar(time))
        {
         series_obj.SendEvent();
         upd &=true;
        }
     }
   if(upd)
      this.SetTerminalServerDate();
  }
//+------------------------------------------------------------------+
```

**Let's improve the CTimeSeriesCollection timeseries collection class** in \\MQL5\\Include\\DoEasy\\Collections\\TimeSeriesCollection.mqh.

Set the timeseries collection list type to be [CListObj class](https://www.mql5.com/en/articles/6211#node01) type.

To do this, include the CListObj class file and change the collection list type from CArrayObj to **CListObj**:

```
//+------------------------------------------------------------------+
//|                                         TimeSeriesCollection.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Objects\Series\TimeSeries.mqh"
#include "..\Objects\Symbols\Symbol.mqh"
//+------------------------------------------------------------------+
//| Symbol timeseries collection                                     |
//+------------------------------------------------------------------+
class CTimeSeriesCollection : public CObject
  {
private:
   CListObj                m_list;                    // List of applied symbol timeseries
//--- Return the timeseries index by symbol name
   int                     IndexTimeSeries(const string symbol);
public:
```

In the public section of the class, declare the method for returning the specified timeseries bar by the chart timeseries index, the method returning the flag of opening a new bar of a specified timeseries and the method for updating timeseries that do not belong to the current symbol:

```
//--- Return the flag of data synchronization with the server data of the (1) specified timeseries of the specified symbol,
//--- (2) the specified timeseries of all symbols, (3) all timeseries of the specified symbol and (4) all timeseries of all symbols
   bool                    SyncData(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint required=0,const int rates_total=0);
   bool                    SyncData(const ENUM_TIMEFRAMES timeframe,const uint required=0,const int rates_total=0);
   bool                    SyncData(const string symbol,const uint required=0,const int rates_total=0);
   bool                    SyncData(const uint required=0,const int rates_total=0);

//--- Return the bar of the specified timeseries of the specified symbol of the specified position
   CBar                   *GetBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index,const bool from_series=true);
//--- Return the flag of opening a new bar of the specified timeseries of the specified symbol
   bool                    IsNewBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time=0);

//--- Create (1) the specified timeseries of the specified symbol, (2) the specified timeseries of all symbols,
//--- (3) all timeseries of the specified symbol and (4) all timeseries of all symbols

   void                    RefreshOther(void);

//--- Display (1) the complete and (2) short collection description in the journal
   void                    Print(const bool created=true);
   void                    PrintShort(const bool created=true);


//--- Constructor
                           CTimeSeriesCollection();
  };
//+------------------------------------------------------------------+
```

In the class constructor, set the timeseries collection ID for the list of timeseries objects:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CTimeSeriesCollection::CTimeSeriesCollection()
  {
   this.m_list.Clear();
   this.m_list.Sort();
   this.m_list.Type(COLLECTION_SERIES_ID);
  }
//+------------------------------------------------------------------+
```

Implementing the methods of returning the bar object by the timeseries index and the new bar event from the specified timeseries list:

```
//+-----------------------------------------------------------------------+
//| Return the bar of the specified timeseries                            |
//| of the specified symbol of the specified position                     |
//| from_series=true - by the timeseries index, false - by the list index |
//+-----------------------------------------------------------------------+
CBar *CTimeSeriesCollection::GetBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index,const bool from_series=true)
  {
//--- Get the timeseries object index in the timeseries collection list by a symbol name
   int idx=this.IndexTimeSeries(symbol);
   if(idx==WRONG_VALUE)
      return NULL;
//--- Get the pointer to the timeseries object from the collection list of timeseries objects by the obtained index
   CTimeSeries *timeseries=this.m_list.At(idx);
   if(timeseries==NULL)
      return NULL;
//--- Get the specified timeseries from the symbol timeseries object by the specified timeframe
   CSeries *series=timeseries.GetSeries(timeframe);
   if(series==NULL)
      return NULL;
//--- Depending on the from_series flag, return the pointer to the bar
//--- either by the chart timeseries index or by the bar index in the timeseries list
   return(from_series ? series.GetBarBySeriesIndex(index) : series.GetBarByListIndex(index));
  }
//+------------------------------------------------------------------+
//| Return new bar opening flag                                      |
//| for a specified timeseries of a specified symbol                 |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::IsNewBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time=0)
  {
//--- Get the timeseries object index in the timeseries collection list by a symbol name
   int index=this.IndexTimeSeries(symbol);
   if(index==WRONG_VALUE)
      return false;
//--- Get the pointer to the timeseries object from the collection list of timeseries objects by the obtained index
   CTimeSeries *timeseries=this.m_list.At(index);
   if(timeseries==NULL)
      return false;
//--- Get the specified timeseries from the symbol timeseries object by the specified timeframe
   CSeries *series=timeseries.GetSeries(timeframe);
   if(series==NULL)
      return false;
//--- Return the result of checking the new bar of the specified timeseries
   return series.IsNewBar(time);
  }
//+------------------------------------------------------------------+
```

Implementing the method of updating all timeseries except for the current symbol timeseries:

```
//+------------------------------------------------------------------+
//| Update all timeseries except the current symbol                  |
//+------------------------------------------------------------------+
void CTimeSeriesCollection::RefreshOther(void)
  {
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL)
         continue;
      if(timeseries.Symbol()==::Symbol() || !timeseries.IsNewTick())
         continue;
      timeseries.RefreshAll();
     }
  }
//+------------------------------------------------------------------+
```

In the loop by the list of all timeseries objects,get the next timeseries object. If the object symbol is equal to a symbol of a chart the program is launched on, such timeseries object is skipped.

This method, as well as the timeseries update methods described below, feature the check for the new tick flag. If there is no new tick, the timeseries is skipped and its data is not updated:

```
//+------------------------------------------------------------------+
//| Update the specified timeseries of the specified symbol          |
//+------------------------------------------------------------------+
void CTimeSeriesCollection::Refresh(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                    const datetime time=0,
                                    const double open=0,
                                    const double high=0,
                                    const double low=0,
                                    const double close=0,
                                    const long tick_volume=0,
                                    const long volume=0,
                                    const int spread=0)
  {
   int index=this.IndexTimeSeries(symbol);
   if(index==WRONG_VALUE)
      return;
   CTimeSeries *timeseries=this.m_list.At(index);
   if(timeseries==NULL)
      return;
   if(!timeseries.IsNewTick())
      return;
   timeseries.Refresh(timeframe,time,open,high,low,close,tick_volume,volume,spread);
  }
//+------------------------------------------------------------------+
//| Update the specified timeseries of all symbols                   |
//+------------------------------------------------------------------+
void CTimeSeriesCollection::Refresh(const ENUM_TIMEFRAMES timeframe,
                                    const datetime time=0,
                                    const double open=0,
                                    const double high=0,
                                    const double low=0,
                                    const double close=0,
                                    const long tick_volume=0,
                                    const long volume=0,
                                    const int spread=0)
  {
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL)
         continue;
      if(!timeseries.IsNewTick())
         continue;
      timeseries.Refresh(timeframe,time,open,high,low,close,tick_volume,volume,spread);
     }
  }
//+------------------------------------------------------------------+
//| Update all timeseries of the specified symbol                    |
//+------------------------------------------------------------------+
void CTimeSeriesCollection::Refresh(const string symbol,
                                    const datetime time=0,
                                    const double open=0,
                                    const double high=0,
                                    const double low=0,
                                    const double close=0,
                                    const long tick_volume=0,
                                    const long volume=0,
                                    const int spread=0)
  {
   int index=this.IndexTimeSeries(symbol);
   if(index==WRONG_VALUE)
      return;
   CTimeSeries *timeseries=this.m_list.At(index);
   if(timeseries==NULL)
      return;
   if(!timeseries.IsNewTick())
      return;
   timeseries.RefreshAll(time,open,high,low,close,tick_volume,volume,spread);
  }
//+------------------------------------------------------------------+
//| Update all timeseries of all symbols                             |
//+------------------------------------------------------------------+
void CTimeSeriesCollection::Refresh(const datetime time=0,
                                    const double open=0,
                                    const double high=0,
                                    const double low=0,
                                    const double close=0,
                                    const long tick_volume=0,
                                    const long volume=0,
                                    const int spread=0)
  {
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL)
         continue;
      if(!timeseries.IsNewTick())
         continue;
      timeseries.RefreshAll(time,open,high,low,close,tick_volume,volume,spread);
     }
  }
//+------------------------------------------------------------------+
```

**The final step is to make the necessary improvements to the file of the CEngine library main object class.**

Open the class file in \\MQL5\\Include\\DoEasy\\Engine.mqh.

In the private class section, declare the variable for storing the type of a program based on the library:

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
   CTimeSeriesCollection m_series;                       // Timeseries collection
   CResourceCollection  m_resource;                      // Resource list
   CTradingControl      m_trading;                       // Trading management object
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
```

In the public section of the class, declare the method for handling [NewTick EA events](https://www.mql5.com/en/docs/runtime/event_fire#newtick):

```
//--- (1) NewTick event timer and (2) handler
   void                 OnTimer(void);
   void                 OnTick(void);
```

In the same public section, declare the method returning the bar object of the specified timeseries of the specified symbol by the chart timeseries index and the method returning the flag of opening a new bar of the specified timeseries of the specified symbol:

```
//--- Create (1) the specified timeseries of the specified symbol, (2) the specified timeseries of all symbols,
//--- (3) all timeseries of the specified symbol and (4) all timeseries of all symbols
   bool                 SeriesCreate(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint required=0)
                          { return this.m_series.CreateSeries(symbol,timeframe,required);          }
   bool                 SeriesCreate(const ENUM_TIMEFRAMES timeframe,const uint required=0)
                          { return this.m_series.CreateSeries(timeframe,required);                 }
   bool                 SeriesCreate(const string symbol,const uint required=0)
                          { return this.m_series.CreateSeries(symbol,required);                    }
   bool                 SeriesCreate(const uint required=0)
                          { return this.m_series.CreateSeries(required);                           }

//--- Return the bar of the specified timeseries of the specified symbol of the specified position
   CBar                *SeriesGetBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index,const bool from_series=true)
                          { return this.m_series.GetBar(symbol,timeframe,index,from_series);                   }
//--- Return the flag of opening a new bar of the specified timeseries of the specified symbol
   bool                 SeriesIsNewBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time=0)
                          { return this.m_series.IsNewBar(symbol,timeframe,time);                  }

//--- Update (1) the specified timeseries of the specified symbol, (2) the specified timeseries of all symbols,
//--- (3) all timeseries of the specified symbol and (4) all timeseries of all symbols
```

In the same class section, declare the methods returning standard bar properties for a specified symbol, timeseries and its position in the chart timeseries (bar index):

```
//--- Return (1) Open, (2) High, (3) Low, (4) Close, (5) Time, (6) TickVolume,
//--- (7) RealVolume, (8) Spread of the specified bar of the specified symbol of the specified timeframe
   double               SeriesOpen(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index);
   double               SeriesHigh(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index);
   double               SeriesLow(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index);
   double               SeriesClose(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index);
   datetime             SeriesTime(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index);
   long                 SeriesTickVolume(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index);
   long                 SeriesRealVolume(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index);
   int                  SeriesSpread(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index);

//--- Set the following for the trading classes:
//--- (1) correct filling policy, (2) filling policy,
//--- (3) correct order expiration type, (4) order expiration type,
//--- (5) magic number, (6) comment, (7) slippage, (8) volume, (9) order expiration date,
//--- (10) the flag of asynchronous sending of a trading request, (11) logging level, (12) number of trading attempts
```

In the class constructor, set the type of the running program and create the counter of the timeseries collection timer:

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

   this.m_list_counters.Sort();
   this.m_list_counters.Clear();
   this.CreateCounter(COLLECTION_ORD_COUNTER_ID,COLLECTION_ORD_COUNTER_STEP,COLLECTION_ORD_PAUSE);
   this.CreateCounter(COLLECTION_ACC_COUNTER_ID,COLLECTION_ACC_COUNTER_STEP,COLLECTION_ACC_PAUSE);
   this.CreateCounter(COLLECTION_SYM_COUNTER_ID1,COLLECTION_SYM_COUNTER_STEP1,COLLECTION_SYM_PAUSE1);
   this.CreateCounter(COLLECTION_SYM_COUNTER_ID2,COLLECTION_SYM_COUNTER_STEP2,COLLECTION_SYM_PAUSE2);
   this.CreateCounter(COLLECTION_REQ_COUNTER_ID,COLLECTION_REQ_COUNTER_STEP,COLLECTION_REQ_PAUSE);
   this.CreateCounter(COLLECTION_TS_COUNTER_ID,COLLECTION_TS_COUNTER_STEP,COLLECTION_TS_PAUSE);

   ::ResetLastError();
   #ifdef __MQL5__
      if(!::EventSetMillisecondTimer(TIMER_FREQUENCY))
        {
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_FAILED_CREATE_TIMER),(string)::GetLastError());
         this.m_global_error=::GetLastError();
        }
   //---__MQL4__
   #else
      if(!this.IsTester() && !::EventSetMillisecondTimer(TIMER_FREQUENCY))
        {
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_FAILED_CREATE_TIMER),(string)::GetLastError());
         this.m_global_error=::GetLastError();
        }
   #endif
   //---
  }
//+------------------------------------------------------------------+
```

In the OnTimer() handler of the library, add working with the timeseries collection timer (excessive code removed):

```
//+------------------------------------------------------------------+
//| CEngine timer                                                    |
//+------------------------------------------------------------------+
void CEngine::OnTimer(void)
  {
//--- Timer of the collections of historical orders and deals, as well as of market orders and positions
//...

//--- Account collection timer
//...

//--- Timer 1 of the symbol collection (updating symbol quote data in the collection)
//...

//--- Timer 2 of the symbol collection (updating all data of all symbols in the collection and tracking symbl and symbol search events in the market watch window)
//...

//--- Trading class timer
//...

//--- Timeseries collection timer
   index=this.CounterIndex(COLLECTION_TS_COUNTER_ID);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter!=NULL)
        {
         //--- If this is not a tester
         if(!this.IsTester())
           {
            //--- If the pause is over, work with the timeseries list (except for the current symbol timeseries)
            if(counter.IsTimeDone())
               this.m_series.RefreshOther();
           }
         //--- In case of the tester, work with the timeseries list by tick (except for the current symbol timeseries)
         else
            this.m_series.RefreshOther();
        }
     }
  }
//+------------------------------------------------------------------+
```

Working with the collection timer counters and with the timer itself, [has been considered when creating the CEngine library main object](https://www.mql5.com/en/articles/5687#node02). Everything else is described in the comments to the code.

Keep in mind that the timer handles only the timeseries whose symbol does not match the symbol of a chart the program is launched on.

In the timer, we update timeseries when registering "New tick" events for "non-native" symbols. Therefore, these are the events we detect in the timer.

The OnTick() method to be launched from the EA's [OnTick()](https://www.mql5.com/en/docs/event_handlers/ontick) handler is used to update the current symbol timeseries:

```
//+------------------------------------------------------------------+
//| NewTick event handler                                            |
//+------------------------------------------------------------------+
void CEngine::OnTick(void)
  {
//--- If this is not a EA, exit
   if(this.m_program!=PROGRAM_EXPERT)
      return;
//--- Update the current symbol timeseries
   this.SeriesRefresh(NULL,PERIOD_CURRENT);
  }
//+------------------------------------------------------------------+
```

Implementing the methods for receiving the main properties of the specified bar of the specified timeseries:

```
//+------------------------------------------------------------------+
//| Return the specified bar's Open                                  |
//| of the specified symbol of the specified timeframe               |
//+------------------------------------------------------------------+
double CEngine::SeriesOpen(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index)
  {
   CBar *bar=this.m_series.GetBar(symbol,timeframe,index);
   return(bar!=NULL ? bar.Open() : 0);
  }
//+------------------------------------------------------------------+
//| Return the specified bar's High                                  |
//| of the specified symbol of the specified timeframe               |
//+------------------------------------------------------------------+
double CEngine::SeriesHigh(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index)
  {
   CBar *bar=this.m_series.GetBar(symbol,timeframe,index);
   return(bar!=NULL ? bar.High() : 0);
  }
//+------------------------------------------------------------------+
//| Return the specified bar's Low                                   |
//| of the specified symbol of the specified timeframe               |
//+------------------------------------------------------------------+
double CEngine::SeriesLow(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index)
  {
   CBar *bar=this.m_series.GetBar(symbol,timeframe,index);
   return(bar!=NULL ? bar.Low() : 0);
  }
//+------------------------------------------------------------------+
//| Return the specified bar's Close                                 |
//| of the specified symbol of the specified timeframe               |
//+------------------------------------------------------------------+
double CEngine::SeriesClose(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index)
  {
   CBar *bar=this.m_series.GetBar(symbol,timeframe,index);
   return(bar!=NULL ? bar.Close() : 0);
  }
//+------------------------------------------------------------------+
//| Return the specified bar's Time                                  |
//| of the specified symbol of the specified timeframe               |
//+------------------------------------------------------------------+
datetime CEngine::SeriesTime(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index)
  {
   CBar *bar=this.m_series.GetBar(symbol,timeframe,index);
   return(bar!=NULL ? bar.Time() : 0);
  }
//+------------------------------------------------------------------+
//| Return the specified bar's TickVolume                            |
//| of the specified symbol of the specified timeframe               |
//+------------------------------------------------------------------+
long CEngine::SeriesTickVolume(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index)
  {
   CBar *bar=this.m_series.GetBar(symbol,timeframe,index);
   return(bar!=NULL ? bar.VolumeTick() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//| Return the specified bar's RealVolume                            |
//| of the specified symbol of the specified timeframe               |
//+------------------------------------------------------------------+
long CEngine::SeriesRealVolume(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index)
  {
   CBar *bar=this.m_series.GetBar(symbol,timeframe,index);
   return(bar!=NULL ? bar.VolumeReal() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//| Return the specified bar's Spread                                |
//| of the specified symbol of the specified timeframe               |
//+------------------------------------------------------------------+
int CEngine::SeriesSpread(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index)
  {
   CBar *bar=this.m_series.GetBar(symbol,timeframe,index);
   return(bar!=NULL ? bar.Spread() : INT_MIN);
  }
//+------------------------------------------------------------------+
```

Here all is simple: receive the bar objectby timeseries symbol and timeframe from the specified index of the chart timeseries (0 — current bar) and return the appropriate bar property.

These are all the improvements needed today to create an auto update of the timeseries price data used in the program, send events to the control program chart and receive data from the created timeseries in the program.

### Testing

Let's perform the test the following way:

create three timeseries for the current timeframes of three symbols, get the zero bar object (CBar class) from the timeseries collection object (CTimeSeriesCollection) and display the bar data in the chart comment using the methods returning the short name of the bar object + description of the bar object parameters. The second comment line is to contain the zero bar data in a similar format. In this case, however, the data is generated using the methods of the CEngine library main object returning the data of the specified bar of the specified symbol of the specified timeframe.

The data is to be updated in real time in the tester and on the chart the EA is launched on.

We are also going to implement handling receiving of events from the CSeries class objects sending the "New bar" event to the control program chart and observe receiving these events in the program launched on a symbol chart.

To perform the test, we will use the EA from the previous article and save it in \\MQL5\\Experts\\TestDoEasy\ **Part38\** under the name **TestDoEasyPart38.mq5**.

Check the EA's OnTick() handler the following way:

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
//--- Handle the NewTick event in the library
   engine.OnTick();

//--- If the trailing flag is set
   if(trailing_on)
     {
      TrailingPositions();    // Trailing positions
      TrailingOrders();       // Trailing of pending orders
     }

//--- Bet the zero bar of the current timeseries
   CBar *bar=engine.SeriesGetBar(NULL,PERIOD_CURRENT,0);
   if(bar==NULL)
      return;
//--- Create a string of parameters of the current bar similar to the one
//--- displayed by the bar object description:
//--- bar.Header()+": "+bar.ParameterDescription()
   string parameters=
     (TextByLanguage("Бар \"","Bar \"")+Symbol()+"\" "+TimeframeDescription((ENUM_TIMEFRAMES)Period())+"[0]: "+TimeToString(bar.Time(),TIME_DATE|TIME_MINUTES|TIME_SECONDS)+
      ", O: "+DoubleToString(engine.SeriesOpen(NULL,PERIOD_CURRENT,0),Digits())+
      ", H: "+DoubleToString(engine.SeriesHigh(NULL,PERIOD_CURRENT,0),Digits())+
      ", L: "+DoubleToString(engine.SeriesLow(NULL,PERIOD_CURRENT,0),Digits())+
      ", C: "+DoubleToString(engine.SeriesClose(NULL,PERIOD_CURRENT,0),Digits())+
      ", V: "+(string)engine.SeriesTickVolume(NULL,PERIOD_CURRENT,0)+
      ", Real: "+(string)engine.SeriesRealVolume(NULL,PERIOD_CURRENT,0)+
      ", Spread: "+(string)engine.SeriesSpread(NULL,PERIOD_CURRENT,0)
     );
//--- Display the data received from the bar object in the first line of the chart comment,
//--- while the second line contains the methods of receiving timeseries price data
   Comment(bar.Header(),": ",bar.ParameterDescription(),"\n",parameters);
  }
//+------------------------------------------------------------------+
```

Here all is simple: the code block is a standard template when working with the DoEasy library. The current implementation features calling the NewTick event handler handled by the library on each tick (currently, it perofrms the update of created timeseries). All missing timeseries (declared but not created by the Create() methods) are skipped (not updated by the library). In the future, calling this method from the OnTick() handler for EAs will be required to update the current timeseries data.

Next, we receive the bar object from the current symbol and period timeseries, create the string featuring the description of the obtained bar data and display two lines in the comment:

the first line is displayed using the bar object methods,

the second one consists of data obtained by the methods of the library main object returning the requested bar data.

The **OnInitDoEasy()** library initialization function features the slightly changed code block for creating the timeseries of all used symbols:

```
//--- Implement displaying the list of used timeframes only for MQL5 - MQL4 has no ArrayPrint() function
#ifdef __MQL5__
   if(InpModeUsedTFs!=TIMEFRAMES_MODE_CURRENT)
      ArrayPrint(array_used_periods);
#endif
//--- Create timeseries of all used symbols
   CArrayObj *list_timeseries=engine.GetListTimeSeries();
   if(list_timeseries!=NULL)
     {
      int total=list_timeseries.Total();
      for(int i=0;i<total;i++)
        {
         CTimeSeries *timeseries=list_timeseries.At(i);
         int total_periods=ArraySize(array_used_periods);
         for(int j=0;j<total_periods;j++)
           {
            ENUM_TIMEFRAMES timeframe=TimeframeByDescription(array_used_periods[j]);
            timeseries.SyncData(timeframe);
            timeseries.Create(timeframe);
           }
        }
     }
//--- Check created timeseries - display descriptions of all created timeseries in the journal
//--- (true - only created ones, false - created and declared ones)
   engine.GetTimeSeriesCollection().PrintShort(true); // Short descriptions
   //engine.GetTimeSeriesCollection().Print(true);      // Full descriptions
```

Here we obtain the list of all timeseries and, in the loop by the timeseries list,get the next timeseries object by the loop index. Then in the loop by the number of used timeframes, create the required timeseries listafter synchronizing the timeseries and history data.

In the function handling **OnDoEasyEvent()** library events, add the code block for handling timeseries events (the redundant code has been removed):

```
//+------------------------------------------------------------------+
//| Handling DoEasy library events                                   |
//+------------------------------------------------------------------+
void OnDoEasyEvent(const int id,
                   const long &lparam,
                   const double &dparam,
                   const string &sparam)
  {
   int idx=id-CHARTEVENT_CUSTOM;
//--- Retrieve (1) event time milliseconds, (2) reason and (3) source from lparam, as well as (4) set the exact event time
   ushort msc=engine.EventMSC(lparam);
   ushort reason=engine.EventReason(lparam);
   ushort source=engine.EventSource(lparam);
   long time=TimeCurrent()*1000+msc;

//--- Handling symbol events
//...

//--- Handling account events
//...

//--- Handling market watch window events
//...

//--- Handling timeseries events
   else if(idx>SERIES_EVENTS_NO_EVENT && idx<SERIES_EVENTS_NEXT_CODE)
     {
      //--- "New bar" event
      if(idx==SERIES_EVENTS_NEW_BAR)
        {
         Print(TextByLanguage("Новый бар на ","New Bar on "),sparam," ",TimeframeDescription((ENUM_TIMEFRAMES)dparam),": ",TimeToString(lparam));
        }
     }

//--- Handling trading events
//...

  }
//+------------------------------------------------------------------+
```

Here, if the obtained event ID is located within the timeseries event IDs and if this is the "New bar" event, display the message about the event in the terminal journal.

Compile the EA and set its parameters the following way:

- set **Mode of used symbols list** for using a specified symbol list,
- in the **List of used symbols (comma - separator)**, leave only three symbols, one of them being EURUSD and
- in **Mode of used timeframes list**, select working with the current timeframe only, for example:

![](https://c.mql5.com/2/39/Image_1.png)

Launch the EA on the chart. After a while, the journal displays the "New bar" event messages on used symbols for the current symbol chart:

```
New bar on EURUSD M5: 2020.03.11 12:55
New bar on EURAUD M5: 2020.03.11 12:55
New bar on AUDUSD M5: 2020.03.11 12:55
New bar on EURUSD M5: 2020.03.11 13:00
New bar on AUDUSD M5: 2020.03.11 13:00
New bar on EURAUD M5: 2020.03.11 13:00
```

Launch the EA in the visual tester mode on the chart of one of the symbols selected in the settings, for example on EURUSD, and see how the zero bar data changes in the chart comment:

![](https://c.mql5.com/2/38/7Pj1WOu2GA.gif)

As we can see, both lines containing data obtained in different ways, have identical values of received zero bar properties and are updated in real time on each tick.

### What's next?

In the next article, we will fix some shortcomings of the current library version detected upon completion of the current article and continue the development of the concept of working with timeseries by preparing the library for working as part of indicators.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7695#node00)

**Previous articles within the series:**

[Timeseries in DoEasy library (part 35): Bar object and symbol timeseries list](https://www.mql5.com/en/articles/7594)

[Timeseries in DoEasy library (part 36): Object of timeseries for all used symbol periods](https://www.mql5.com/en/articles/7627)

[Timeseries in DoEasy library (part 37): Timeseries collection - database of timeseries by symbols and periods](https://www.mql5.com/en/articles/7663)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7695](https://www.mql5.com/ru/articles/7695)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7695.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7695/mql5.zip "Download MQL5.zip")(3698.04 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7695/mql4.zip "Download MQL4.zip")(3697.98 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/345731)**
(15)


![BillionerClub](https://c.mql5.com/avatar/avatar_na2.png)

**[BillionerClub](https://www.mql5.com/en/users/billionerclub)**
\|
23 Jan 2021 at 20:11

**Artyom Trishkin:**

Okay. In which EA is this happening?

mt4 do easy part 38

server FXOpen-Real1

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
24 Jan 2021 at 00:25

**BillionerClub:**

mt4 do easy part 38

server FXOpen-Real1

If my memory serves me correctly, this **test** EA is specially designed not to open positions with the minimum lot so that partial [closures](https://www.metatrader5.com/en/terminal/help/trading/performing_deals#position_manage "Help: Opening and closing positions in MetaTrader 5 trading terminal") can be **tested**.

![BillionerClub](https://c.mql5.com/avatar/avatar_na2.png)

**[BillionerClub](https://www.mql5.com/en/users/billionerclub)**
\|
24 Jan 2021 at 10:03

**Artyom Trishkin:**

If my memory serves me correctly, this **test** Expert Advisor is specially designed not to open positions with the minimum lot - so that partial position [closures](https://www.metatrader5.com/en/terminal/help/trading/performing_deals#position_manage "Help: Opening and closing positions in MetaTrader 5 trading terminal") can be **tested**.

I thought it was a feature,  like  "  he _who does not take risks_ _does not drink champagne_", when you open a lot and you get 15 bang. Of course, it's a cool feature))

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
24 Jan 2021 at 13:02

**BillionerClub:**

I thought a chip, like  " _who does not risk_ \- _he does not drink champagne_" open a lot and you 15 so bang. Ficha of course cool)))

Once again I emphasise that this is a TEST advisor, and it is not intended for use not in the context of the article.


![BillionerClub](https://c.mql5.com/avatar/avatar_na2.png)

**[BillionerClub](https://www.mql5.com/en/users/billionerclub)**
\|
24 Jan 2021 at 16:20

**Artyom Trishkin:**

Once again I emphasise that this is a TEST advisor, and it is not intended to be used outside the context of the article.

What I will create is even worse, and kodobase, you can not say that there is better

![MQL as a Markup Tool for the Graphical Interface of MQL Programs. Part 1](https://c.mql5.com/2/38/MQL5-avatar-dialog_form__1.png)[MQL as a Markup Tool for the Graphical Interface of MQL Programs. Part 1](https://www.mql5.com/en/articles/7734)

This paper proposes a new conception to describe the window interface of MQL programs, using the structures of MQL. Special classes transform the viewable MQL markup into the GUI elements and allow manage them, set up their properties, and process the events in a unified manner. It also provides some examples of using the markup for the dialogs and elements of a standard library.

![Multicurrency monitoring of trading signals (Part 3): Introducing search algorithms](https://c.mql5.com/2/38/Article_Logo.png)[Multicurrency monitoring of trading signals (Part 3): Introducing search algorithms](https://www.mql5.com/en/articles/7600)

In the previous article, we developed the visual part of the application, as well as the basic interaction of GUI elements. This time we are going to add internal logic and the algorithm of trading signal data preparation, as well us the ability to set up signals, to search them and to visualize them in the monitor.

![Multicurrency monitoring of trading signals (Part 4): Enhancing functionality and improving the signal search system](https://c.mql5.com/2/38/Article_Logo__1.png)[Multicurrency monitoring of trading signals (Part 4): Enhancing functionality and improving the signal search system](https://www.mql5.com/en/articles/7678)

In this part, we expand the trading signal searching and editing system, as well as introduce the possibility to use custom indicators and add program localization. We have previously created a basic system for searching signals, but it was based on a small set of indicators and a simple set of search rules.

![Applying OLAP in trading (part 4): Quantitative and visual analysis of tester reports](https://c.mql5.com/2/38/OLAP_in_trading.png)[Applying OLAP in trading (part 4): Quantitative and visual analysis of tester reports](https://www.mql5.com/en/articles/7656)

The article offers basic tools for the OLAP analysis of tester reports relating to single passes and optimization results. The tool can work with standard format files (tst and opt), and it also provides a graphical interface. MQL source codes are attached below.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=fpqbhcyvkkgpicyxbyiiaxfewifbuims&ssn=1769251903541448805&ssn_dr=0&ssn_sr=0&fv_date=1769251903&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7695&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Timeseries%20in%20DoEasy%20library%20(part%2038)%3A%20Timeseries%20collection%20-%20real-time%20updates%20and%20accessing%20data%20from%20the%20program%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925190351679504&fz_uniq=5083164962773276223&sv=2552)

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