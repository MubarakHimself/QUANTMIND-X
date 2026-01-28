---
title: Timeseries in DoEasy library (part 39): Library-based indicators - preparing data and timeseries events
url: https://www.mql5.com/en/articles/7724
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:36:28.407055
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/7724&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070421833460684169)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/7724#node01)
- [Improving classes for working with indicators, creating timeseries events](https://www.mql5.com/en/articles/7724#node02)

- [Testing timeseries and their events in indicators](https://www.mql5.com/en/articles/7724#node03)
- [What's next?](https://www.mql5.com/en/articles/7724#node04)


### Concept

Everything we did up to this point was related to EAs and scripts only. In no way was it related to indicators. However, the timeseries can also be actively used as a data source for various calculations in indicators. So, it is time to consider them as well.

Unlike EAs, indicators feature a completely different architecture. Each indicator is executed in a single stream of a single symbol it is launched on. This means that if we launch different indicators on several charts of the same symbol, they are all executed in the same symbol thread all these charts belong to.

Accordingly, if one of the indicators has flawed architecture, it slows down the entire symbol thread. In this case, all the remaining indicators working in the same thread freeze while waiting for the "slow" indicator.

In order to avoid delays while waiting for historical data when working with indicators, the terminal features sequential return of requested data — the functions activating the loading of historical data immediately return the function result without waiting.

[When requesting data of any timeseries of any symbol using Copy functions](https://www.mql5.com/en/docs/series/copyrates), an indicator and an EA show different behavior when the terminal sends historical data:

When requesting data from an indicator, the function immediately returns -1 if requested timeseries are not constructed yet or they should be downloaded from the server, but loading/constructing itself is initiated.

When requesting data from an EA or a script, [download from the server](https://www.mql5.com/en/docs/series/timeseries_access#synchronized) is initiated if the terminal does not have the appropriate data locally, or construction of the necessary timeseries starts if the data can be constructed from the local history but they are not ready yet. The function returns the amount that will be ready by the time the timeout expires, however the history download continues, and the function returns more data during the next similar request.

Thus, we can see that when requesting data from the EA, the terminal starts downloading data (if there is no locally requested data yet or it is not sufficient). Upon timeout expiration, the function returns the amount of history already present at the moment of waiting for history download to complete — the terminal immediately attempts to provide us with the requested history. If the local data is insufficient, it attempts to download it in the necessary amount.

Meanwhile, the program waits for the data to be downloaded.

In case of indicators, we cannot wait, so the terminal sends us what it has (or reports that it has nothing). If there is no local history or it is insufficient during the first data request, its download begins. Here, the system does not wait till the missing data is downloaded before the timeout.

In the current situation, the program should exit its calculation part before the next tick. During the next launch of the indicator's OnCalculate() handler on a new tick, the data may already be partially or fully loaded and available for calculations. Here we should decide on how much data will be enough to run the program algorithm seamlessly.

Besides, [the indicator should not try to download its own data](https://www.mql5.com/en/docs/series/timeseries_access#synchronized) — the data whose symbol and period it is launched on. Otherwise, such a request may lead to a conflict. The terminal subsystem downloads such data for indicators. It provides us with all the data on their amount and status in the **rates\_total** and **prev\_calculated** variables of the [OnCalculate()](https://www.mql5.com/en/docs/event_handlers/oncalculate) handler.

Based on these minimum requirements, we need to adjust some classes for working with timeseries and arrange the correct initial loading of the data necessary for calculations in our indicators.

In the current article, we are going to adjust the classes that have already been created, arrange the correct initial data loading of all used timeseries in our programs and send any events of all used timeseries to the control program chart during their real-time update.

### Improving classes for working with indicators, creating timeseries events

First of all, let's add the new messages to the **Datas.mqh** file — message indices:

```
   MSG_LIB_SYS_FAILED_PREPARING_SYMBOLS_ARRAY,        // Failed to prepare array of used symbols. Error
   MSG_LIB_SYS_FAILED_GET_SYMBOLS_ARRAY,              // Failed to get array of used symbols.
   MSG_LIB_SYS_ERROR_EMPTY_PERIODS_STRING,            // Error. The string of predefined periods is empty and is to be used
```

...

```
//--- CBar
   MSG_LIB_TEXT_BAR_FAILED_GET_BAR_DATA,              // Failed to receive bar data
   MSG_LIB_TEXT_BAR_FAILED_DT_STRUCT_WRITE,           // Failed to write time to time structure
   MSG_LIB_TEXT_BAR_FAILED_GET_SERIES_DATA,           // Failed to receive timeseries data
```

...

```
   MSG_LIB_TEXT_TS_TEXT_SYMBOL_TERMINAL_FIRSTDATE,    // The very first date in history by a symbol in the client terminal
   MSG_LIB_TEXT_TS_TEXT_CREATED_OK,                   // successfully created
   MSG_LIB_TEXT_TS_TEXT_NOT_CREATED,                  // not created
   MSG_LIB_TEXT_TS_TEXT_IS_SYNC,                      // synchronized
   MSG_LIB_TEXT_TS_TEXT_ATTEMPT,                      // Attempt:
   MSG_LIB_TEXT_TS_TEXT_WAIT_FOR_SYNC,                // Waiting for data synchronization ...

  };
//+------------------------------------------------------------------+
```

and message texts corresponding to newly added indices:

```
   {"Не удалось подготовить массив используемых символов. Ошибка ","Failed to create an array of used symbols. Error "},
   {"Не удалось получить массив используемых символов","Failed to get array of used symbols"},
   {"Ошибка. Строка предопределённых периодов пустая, будет использоваться ","Error. String of predefined periods is empty, the Period will be used: "},
```

...

```
   {"Не удалось получить данные бара","Failed to get bar data"},
   {"Не удалось записать время в структуру времени","Failed to write time to datetime structure"},
   {"Не удалось получить данные таймсерии","Failed to get timeseries data"},
```

...

```
   {"Самая первая дата в истории по символу в клиентском терминале","Very first date in history of symbol in client terminal"},
   {"создана успешно","created successfully"},
   {"не создана","not created"},
   {"синхронизирована","synchronized"},
   {"Попытка: ","Attempt: "},
   {"Ожидание синхронизации данных ...","Waiting for data synchronization ..."},

  };
//+---------------------------------------------------------------------+
```

In the class constructor of the CBaseObj base object of all library objects in \\MQL5\\Include\\DoEasy\\Objects\ **BaseObj.mqh**, I have changed the initialization of the **m\_available** variable. Right during the creation, all objects derived from the CBaseObj base object feature the availability property for working in the program with the "used" status (true). Previously, the value was installed when initializing into "not used" false status:

```
//--- Constructor
                     CBaseObj() : m_program((ENUM_PROGRAM_TYPE)::MQLInfoInteger(MQL_PROGRAM_TYPE)),
                                  m_global_error(ERR_SUCCESS),
                                  m_log_level(LOG_LEVEL_ERROR_MSG),
                                  m_chart_id_main(::ChartID()),
                                  m_chart_id(::ChartID()),
                                  m_folder_name(DIRECTORY),
                                  m_sound_name(""),
                                  m_name(__FUNCTION__),
                                  m_type(0),
                                  m_use_sound(false),
                                  m_available(true),
                                  m_first_start(true) {}
  };
//+------------------------------------------------------------------+
```

The name of the method setting the flag of an event detected in the object has been changed in the class of the extended base object of all CBaseObjExt library objects in \\MQL5\\Include\\DoEasy\\Objects\ **BaseObj.mqh**:

```
//--- Set/return the occurred event flag to the object data
   void              SetEventFlag(const bool flag)                   { this.m_is_event=flag;                   }
```

Previously, the method was called SetEvent() which could cause some confusion since SetEvent can mean creating, setting, sending, etc. of any event rather than setting a signal flag of the event presence.

Therefore, the files of the classes applying the method have also been changed — calling the SetEvent() method has been replaced with SetEventFlag(). Find the details in the attached files.

**Since trading functions are disabled in indicators, make changes in trading object classes.**

In the cross-platform trading object class in \\MQL5\\Include\\DoEasy\\Objects\\Trade\ **TradeObj.mqh** at the beginning of all trading methods, enter a check for the program type. If this is an indicator or a service, leave the method and return true:

```
//+------------------------------------------------------------------+
//| Open a position                                                  |
//+------------------------------------------------------------------+
bool CTradeObj::OpenPosition(const ENUM_POSITION_TYPE type,
                             const double volume,
                             const double sl=0,
                             const double tp=0,
                             const ulong magic=ULONG_MAX,
                             const string comment=NULL,
                             const ulong deviation=ULONG_MAX,
                             const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   if(this.m_program==PROGRAM_INDICATOR || this.m_program==PROGRAM_SERVICE)
      return true;
   ::ResetLastError();
```

...

```
//+------------------------------------------------------------------+
//| Close a position                                                 |
//+------------------------------------------------------------------+
bool CTradeObj::ClosePosition(const ulong ticket,
                              const string comment=NULL,
                              const ulong deviation=ULONG_MAX)
  {
   if(this.m_program==PROGRAM_INDICATOR || this.m_program==PROGRAM_SERVICE)
      return true;
   ::ResetLastError();
```

...

```
//+------------------------------------------------------------------+
//| Close a position partially                                       |
//+------------------------------------------------------------------+
bool CTradeObj::ClosePositionPartially(const ulong ticket,
                                       const double volume,
                                       const string comment=NULL,
                                       const ulong deviation=ULONG_MAX)
  {
   if(this.m_program==PROGRAM_INDICATOR || this.m_program==PROGRAM_SERVICE)
      return true;
   ::ResetLastError();
```

...

```
//+------------------------------------------------------------------+
//| Close a position by an opposite one                              |
//+------------------------------------------------------------------+
bool CTradeObj::ClosePositionBy(const ulong ticket,const ulong ticket_by)
  {
   if(this.m_program==PROGRAM_INDICATOR || this.m_program==PROGRAM_SERVICE)
      return true;
   ::ResetLastError();
```

...

```
//+------------------------------------------------------------------+
//| Modify a position                                                |
//+------------------------------------------------------------------+
bool CTradeObj::ModifyPosition(const ulong ticket,const double sl=WRONG_VALUE,const double tp=WRONG_VALUE)
  {
   if(this.m_program==PROGRAM_INDICATOR || this.m_program==PROGRAM_SERVICE)
      return true;
   ::ResetLastError();
```

...

```
//+------------------------------------------------------------------+
//| Set an order                                                     |
//+------------------------------------------------------------------+
bool CTradeObj::SetOrder(const ENUM_ORDER_TYPE type,
                         const double volume,
                         const double price,
                         const double sl=0,
                         const double tp=0,
                         const double price_stoplimit=0,
                         const ulong magic=ULONG_MAX,
                         const string comment=NULL,
                         const datetime expiration=0,
                         const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                         const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   if(this.m_program==PROGRAM_INDICATOR || this.m_program==PROGRAM_SERVICE)
      return true;
   ::ResetLastError();
```

...

```
//+------------------------------------------------------------------+
//| Remove an order                                                  |
//+------------------------------------------------------------------+
bool CTradeObj::DeleteOrder(const ulong ticket)
  {
   if(this.m_program==PROGRAM_INDICATOR || this.m_program==PROGRAM_SERVICE)
      return true;
   ::ResetLastError();
```

...

```
//+------------------------------------------------------------------+
//| Modify an order                                                  |
//+------------------------------------------------------------------+
bool CTradeObj::ModifyOrder(const ulong ticket,
                            const double price=WRONG_VALUE,
                            const double sl=WRONG_VALUE,
                            const double tp=WRONG_VALUE,
                            const double price_stoplimit=WRONG_VALUE,
                            const datetime expiration=WRONG_VALUE,
                            const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                            const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   if(this.m_program==PROGRAM_INDICATOR || this.m_program==PROGRAM_SERVICE)
      return true;
   ::ResetLastError();
```

The same changes have been made in all same-name trading methods of the library's main trading class in \\MQL5\\Include\\DoEasy\ **Trading.mqh**.

Exiting trading methods in such a way does not allow calling trading functions in programs where they are disabled and returns the method successful execution preventing handling library errors.

**Now let's consider the changes that directly affected the classes of timeseries objects.**

In the bar object class, I have slightly changed the texts displayed from the class constructor in case of an error when receiving historical data while creating a bar object. The displayed text now also features constructor number, symbol and timeframe of the timeseries the bar object is created for.

In the first form constructor,checking data retrieval errors and writing time to the time structure were set in separate blocks:

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
//--- If ailed to get the requested data by index and write bar data to the MqlRates array,
//--- display an error message, create and fill the structure with zeros, and write it to the rates_array array
   if(::CopyRates(symbol,timeframe,index,1,rates_array)<1)
     {
      int err_code=::GetLastError();
      ::Print
        (
         DFUN,"(1) ",symbol," ",TimeframeDescription(timeframe)," ",
         CMessage::Text(MSG_LIB_TEXT_BAR_FAILED_GET_BAR_DATA),". ",
         CMessage::Text(MSG_LIB_SYS_ERROR)," ",CMessage::Text(err_code)," ",
         CMessage::Retcode(err_code)
        );
      MqlRates err={0};
      rates_array[0]=err;
     }
   ::ResetLastError();
//--- If failed to set time to the time structure, display the error message
   if(!::TimeToStruct(rates_array[0].time,this.m_dt_struct))
     {
      int err_code=::GetLastError();
      ::Print
        (
         DFUN,"(1) ",symbol," ",TimeframeDescription(timeframe)," ",
         CMessage::Text(MSG_LIB_TEXT_BAR_FAILED_DT_STRUCT_WRITE),". ",
         CMessage::Text(MSG_LIB_SYS_ERROR)," ",CMessage::Text(err_code)," ",
         CMessage::Retcode(err_code)
        );
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
      ::Print
        (
         DFUN,"(2) ",symbol," ",TimeframeDescription(timeframe)," ",
         CMessage::Text(MSG_LIB_TEXT_BAR_FAILED_DT_STRUCT_WRITE),". ",
         CMessage::Text(MSG_LIB_SYS_ERROR)," ",CMessage::Text(err_code)," ",
         CMessage::Retcode(err_code)
        );
      MqlRates err={0};
      this.SetProperties(err);
      return;
     }
//--- Set the bar properties
   this.SetProperties(rates);
  }
//+------------------------------------------------------------------+
```

These actions provide us with more data in case of a bar object creation error.

Since we need to use timeseries arrays provided by the OnCalculate() handler to request data about the number of bars and their values on the current period symbol, we need to somehow pass these arrays and values to the library classes.

To do this, create the structure in \\MQL5\\Include\\DoEasy\ **Defines.mqh**. The structure is to store variables to be used to pass all the necessary data calculated for the current timeseries to the library timeseries:

```
//+------------------------------------------------------------------+
//| Structures                                                       |
//+------------------------------------------------------------------+
struct SDataCalculate
  {
   int         rates_total;                                 // size of input time series
   int         prev_calculated;                             // number of handled bars at the previous call
   int         begin;                                       // where significant data start
   double      price;                                       // current array value for calculation
   MqlRates    rates;                                       // Price structure
  } rates_data;
//+------------------------------------------------------------------+
//| Enumerations                                                     |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Search and sorting data                                          |
//+------------------------------------------------------------------+
```

As we can see, the structure contains all the necessary fields for passing data to the library for any implementation of the indicator's OnCalculate() handler.

**For the handler first form**

```
int OnCalculate(
   const int        rates_total,       // price[] array size
   const int        prev_calculated,   // number of handled bars at the previous call
   const int        begin,             // index number in the price[] array meaningful data starts from
   const double&    price[]            // array of values for calculation
   );
```

rates\_total, prev\_calculated, begin and price variable structures are used.

**For the handler second form**

```
int OnCalculate(
   const int        rates_total,       // size of input time series
   const int        prev_calculated,   // number of handled bars at the previous call
   const datetime&  time{},            // Time array
   const double&    open[],            // Open array
   const double&    high[],            // High array
   const double&    low[],             // Low array
   const double&    close[],           // Close array
   const long&      tick_volume[],     // Tick Volume array
   const long&      volume[],          // Real Volume array
   const int&       spread[]           // Spread array
   );
```

rates\_total and prev\_calculated variable structures, as well as the MqlRates rates structure are used to store array values.

The current structure implementation is suitable for passing the value of only a single bar to the library.

In the CSeries class in \\MQL5\\Include\\DoEasy\\Objects\\Series\ **Series.mqh**, add the flag of setting server dates to the methods of setting a symbol and a timeframe:

```
//--- Set (1) symbol, (2) timeframe, (3) symbol and timeframe, (4) amount of applied timeseries data
   void              SetSymbol(const string symbol,const bool set_server_date=false);
   void              SetTimeframe(const ENUM_TIMEFRAMES timeframe,const bool set_server_date=false);
```

By default, the flag is disabled. This prevents setting server dates when calling the method, since, in order to call the method for setting server dates, the flag status is checked first:

```
//+------------------------------------------------------------------+
//| Set a symbol                                                     |
//+------------------------------------------------------------------+
void CSeries::SetSymbol(const string symbol,const bool set_server_date=false)
  {
   if(this.m_symbol==symbol)
      return;
   this.m_symbol=(symbol==NULL || symbol==""   ? ::Symbol() : symbol);
   this.m_new_bar_obj.SetSymbol(this.m_symbol);
   if(set_server_date)
      this.SetServerDate();
  }
//+------------------------------------------------------------------+
//| Set a timeframe                                                  |
//+------------------------------------------------------------------+
void CSeries::SetTimeframe(const ENUM_TIMEFRAMES timeframe,const bool set_server_date=false)
  {
   if(this.m_timeframe==timeframe)
      return;
   this.m_timeframe=(timeframe==PERIOD_CURRENT ? (ENUM_TIMEFRAMES)::Period() : timeframe);
   this.m_new_bar_obj.SetPeriod(this.m_timeframe);
   this.m_period_description=TimeframeDescription(this.m_timeframe);
   if(set_server_date)
      this.SetServerDate();
  }
//+------------------------------------------------------------------+
```

This has been done to avoid multiple resetting of server dates when calling the method of setting a symbol and a timeframe simultaneously:

```
//+------------------------------------------------------------------+
//| Set a symbol and timeframe                                       |
//+------------------------------------------------------------------+
void CSeries::SetSymbolPeriod(const string symbol,const ENUM_TIMEFRAMES timeframe)
  {
   if(this.m_symbol==symbol && this.m_timeframe==timeframe)
      return;
   this.SetSymbol(symbol);
   this.SetTimeframe(timeframe,true);
  }
//+------------------------------------------------------------------+
```

Here, the symbol setting method is called first (flag disabled) followed by the method of setting a timeframe with the enabled flag for calling the method of setting server dates from the timeframe setting method.

The method of updating the timeseries data now passes the new structure of the OnCalculate() handler data instead the full list of its arrays:

```
//--- (1) Create and (2) update the timeseries list
   int               Create(const uint required=0);
   void              Refresh(SDataCalculate &data_calculate);

//--- Create and send the "New bar" event to the control program chart
   void              SendEvent(void);
```

Thus, the Refresh() method implementation now features access to the structure data rather than to the arrays:

```
//+------------------------------------------------------------------+
//| Update timeseries list and data                                  |
//+------------------------------------------------------------------+
void CSeries::Refresh(SDataCalculate &data_calculate)
  {
//--- If the timeseries is not used, exit
   if(!this.m_available)
      return;
   MqlRates rates[1];
//--- Set the flag of sorting the list of bars by index
   this.m_list_series.Sort(SORT_BY_BAR_INDEX);
//--- If a new bar is present on a symbol and period,
   if(this.IsNewBarManual(data_calculate.rates.time))
     {
      //--- create a new bar object and add it to the end of the list
      CBar *new_bar=new CBar(this.m_symbol,this.m_timeframe,0);
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
//--- Get the bar object from the list by the terminal timeseries index (zero bar)
   CBar *bar=this.GetBarBySeriesIndex(0);
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

A search in the list of timeseries objects by timeframe can now be done via the virtual method of comparing two timeseries objects:

```
//--- Comparison method to search for identical timeseries objects by timeframe
   virtual int       Compare(const CObject *node,const int mode=0) const
                       {
                        const CSeries *compared_obj=node;
                        return(this.Timeframe()>compared_obj.Timeframe() ? 1 : this.Timeframe()<compared_obj.Timeframe() ? -1 : 0);
                       }
//--- Constructors
                     CSeries(void);
                     CSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint required=0);
  };
//+------------------------------------------------------------------+
```

The method compares the "timeframe" property of the two compared timeseries objects (the current one and the one passed to the method) and returns zero if they are equal.

We have already examined the logic of similar [methods for searching and sorting](https://www.mql5.com/en/docs/standardlibrary/cobject/cobjectcompare) various objects derived from the [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject) standard library base object. The method is defined as a virtual one in the base object of the standard library. Therefore, it should be implemented in descendant objects, and the method should return zero in case of equality or 1/-1 if the value of the current object's compared property is greater/less than the property value of the compared object.

Since the first access to the functions returning historical data activates downloading data in case it is absent/insufficient locally, add accessing the required historical data (simply request the current bar date) to the very beginning of the method for setting the amount of required data. This starts the download of the required data (in case it is absent locally):

```
//+------------------------------------------------------------------+
//| Set the number of required data                                  |
//+------------------------------------------------------------------+
bool CSeries::SetRequiredUsedData(const uint required,const uint rates_total)
  {
   this.m_required=(required<1 ? SERIES_DEFAULT_BARS_COUNT : required);
//--- Launch downloading historical data
   if(this.m_program!=PROGRAM_INDICATOR || (this.m_program==PROGRAM_INDICATOR && (this.m_symbol!=::Symbol() || this.m_timeframe!=::Period())))
     {
      datetime array[1];
      ::CopyTime(this.m_symbol,this.m_timeframe,0,1,array);
     }
//--- Set the number of available timeseries bars
```

When we created [the object storing the lists of all timeseries of a single symbol](https://www.mql5.com/en/articles/7627) (CTimeSeries class), we made it so that this object always has a list featuring the full set of all timeframes that are possible in the terminal. The timeseries lists are immediately added to the list. However, they are only created if necessary. Accessing the pointers to the necessary timeseries was performed by the constant index corresponding to the list timeframe index position in the ENUM\_TIMEFRAMES enumeration with the offset of 1 ( [described in the article](https://www.mql5.com/en/articles/7627)).

This was done to accelerate access to the pointer to the necessary timeseries object in the list. But it turns out that instant access to the pointer is accompanied by tester issues — the visual tester created charts of absolutely all timeframes regardless of whether they were actually used in the program and whether their timeseries lists were created.

Besides, we have another issue when switching the chart period during the program operation — previously created lists are not re-created and the program resumes tracking events of non-existing objects replacing them with others.

To avoid further accumulation of hidden errors and prolonged search for their causes, I decided to store pointers to actually used and created timeseries lists only in the CTimeSeries class object storing timeseries lists of all used timeframes. In other words, the pointers to each timeseries of each chart period are added to the list only if the program explicitly indicates the need for its use and such a timeseries object is physically created.

Open \\MQL5\\Include\\DoEasy\\Objects\\Series\ **TimeSeries.mqh** and add the necessary improvements to it.

Now the class of timeseries of a single symbol is derived from the class of the extended base object of all library objects.

This is done to be able to use the event functionality of the CBaseObjExt class:

```
//+------------------------------------------------------------------+
//| Symbol timeseries class                                          |
//+------------------------------------------------------------------+
class CTimeSeries : public CBaseObjExt
  {
```

The method returning the timeseries index in the list by timeframe name is now simply declared in the private section of the class:

```
//+------------------------------------------------------------------+
class CTimeSeries : public CBaseObjExt
  {
private:
   string            m_symbol;                                             // Timeseries symbol
   CNewTickObj       m_new_tick;                                           // "New tick" object
   CArrayObj         m_list_series;                                        // List of timeseries by timeframes
   datetime          m_server_firstdate;                                   // The very first date in history by a server symbol
   datetime          m_terminal_firstdate;                                 // The very first date in history by a symbol in the client terminal
//--- Return (1) the timeframe index in the list and (2) the timeframe by the list index
   int               IndexTimeframe(const ENUM_TIMEFRAMES timeframe);
   ENUM_TIMEFRAMES   TimeframeByIndex(const uchar index)             const { return TimeframeByEnumIndex(uchar(index+1));                       }
//--- Set the very first date in history by symbol on the server and in the client terminal
   void              SetTerminalServerDate(void)
                       {
                        this.m_server_firstdate=(datetime)::SeriesInfoInteger(this.m_symbol,::Period(),SERIES_SERVER_FIRSTDATE);
                        this.m_terminal_firstdate=(datetime)::SeriesInfoInteger(this.m_symbol,::Period(),SERIES_TERMINAL_FIRSTDATE);
                       }
public:
```

The method is now implemented outside the class body:

```
//+------------------------------------------------------------------+
//| Return the timeframe index in the list                           |
//+------------------------------------------------------------------+
int CTimeSeries::IndexTimeframe(const ENUM_TIMEFRAMES timeframe)
  {
   const CSeries *obj=new CSeries(this.m_symbol,timeframe);
   if(obj==NULL)
      return WRONG_VALUE;
   this.m_list_series.Sort();
   int index=this.m_list_series.Search(obj);
   delete obj;
   return index;
  }
//+------------------------------------------------------------------+
```

The method receives a timeframe. The pointer to the timeframe's timeseries should be returned.

Next, create a temporary timeseries object with the necessary timeframe.

Set the sorted list flag for the list of timeseries objects and get the timeseries object index in the list whose timeframe is equal to the temporary object timeframe.

If such an object exists in the list, its index is received, otherwise — WRONG\_VALUE (-1).

Remove the temporary object and return the obtained index.

Instead of the Create() and CreateAll() methods, declare the methods for adding the specified timeseries to the list and the method of creating the specified timeseries object, while the methods of updating timeseries lists now receive the structure of parameter values and OnCalculate() arrays instead of the full list of arrays:

```
//--- (1) Add the specified timeseries list to the list and create (2) the specified timeseries list
   bool              AddSeries(const ENUM_TIMEFRAMES timeframe,const uint required=0);
   bool              CreateSeries(const ENUM_TIMEFRAMES timeframe,const uint required=0);
//--- Update (1) the specified timeseries list and (2) all timeseries lists
   void              Refresh(const ENUM_TIMEFRAMES timeframe,SDataCalculate &data_calculate);
   void              RefreshAll(SDataCalculate &data_calculate);

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

Remove the loop of creating timeseries lists from the class constructor:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
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

Now the necessary timeseries are created after creating the array of used timeseries in the program's OnInit() handler. Any change in the number of chart periods used in the program causes EA re-initialization or re-creation of an indicator leading to a complete re-creation of the list of used timeseries objects and their correct accounting in the future.

In the methods of setting the history depth of all used timeseries **SetRequiredAllUsedData()** and returning the synchronization flag of all applied timeseries **SyncAllData()**, replace the loop for the total number of all possible timeframes

```
//+------------------------------------------------------------------+
//| Set the history depth of all applied symbol timeseries           |
//+------------------------------------------------------------------+
bool CTimeSeries::SetRequiredAllUsedData(const uint required=0,const int rates_total=0)
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
      if(series_obj==NULL)
         continue;
      res &=series_obj.SetRequiredUsedData(required,rates_total);
     }
   return res;
  }
//+------------------------------------------------------------------+
```

with the loop for the number of real timeseries objects in the list:

```
   int total=this.m_list_series.Total();
   for(int i=0;i<total;i++)
```

Now the list consists of actually created timeseries objects, and the loops are performed according to their actual number.

**Implementing the method of adding the specified timeseries object to the list:**

```
//+------------------------------------------------------------------+
//| Add the specified timeseries list to the list                    |
//+------------------------------------------------------------------+
bool CTimeSeries::AddSeries(const ENUM_TIMEFRAMES timeframe,const uint required=0)
  {
   bool res=false;
   CSeries *series=new CSeries(this.m_symbol,timeframe,required);
   if(series==NULL)
      return res;
   this.m_list_series.Sort();
   if(this.m_list_series.Search(series)==WRONG_VALUE)
      res=this.m_list_series.Add(series);
   if(!res)
      delete series;
   series.SetAvailable(true);
   return res;
  }
//+------------------------------------------------------------------+
```

The method receives the timeseries chart period to be added to the symbol timeseries list.

Create a timeseries object featuring a timeframe whose value is passed to the method.

Set the sorted list flag for the timeseries list and search the list for a timeseries object equal to the newly created one.

If the list contains no such object (the search returns -1), add the created timeseries object to the list.

Otherwise, remove the created object since such a timeseries object is already on the list.

Set the flag of using the timeseries in the program and return the result of adding the timeseries to the list.

Successful adding returns true, unsuccessful — false.

The library features the event functionality in the extended object of all library objects for sending events occurring to library's various objects. In the [articles 16](https://www.mql5.com/en/articles/7071) and [17](https://www.mql5.com/en/articles/7124), we considered the principles and logic of working with library events.

In short, each object derived from the CBaseObj library base object (currently, it is CBaseObjExt) has the list registering all events that may occur to the object within one loop of the program operation on a single tick or a single timer iteration.

When identifying any object event, the flag of an occurred event is set for it. Next, the lists of collection objects can be viewed in the collection classes. In turn, the flags are checked in the lists. If an object with the enabled event flag is found, the collection class of these objects receives the list of all object events with the event flag enabled and sends all events from the list to the control program chart.

The program itself features the functionality for handling all incoming events. In the tester, all events are handled by ticks. Beyond the tester, they are processed in the OnChartEvent() handler.

In the considered object class of all timeseries of a single symbol CTimeSeries, the best place for defining events of all its timeseries lists is a method of updating the specified Refresh() timeseries and the method of updating all symbol timeseries RefreshAll().

**Let's consider implementing the methods of updating timeseries lists:**

```
//+------------------------------------------------------------------+
//| Update a specified timeseries list                               |
//+------------------------------------------------------------------+
void CTimeSeries::Refresh(const ENUM_TIMEFRAMES timeframe,SDataCalculate &data_calculate)
  {
//--- Reset the timeseries event flag and clear the list of all timeseries events
   this.m_is_event=false;
   this.m_list_events.Clear();
//--- Get the timeseries from the list by its timeframe
   CSeries *series_obj=this.m_list_series.At(this.IndexTimeframe(timeframe));
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
      if(this.EventAdd(SERIES_EVENTS_NEW_BAR,series_obj.Time(0),series_obj.Timeframe(),series_obj.Symbol()))
         this.m_is_event=true;
     }
  }
//+------------------------------------------------------------------+
//| Update all timeseries lists                                      |
//+------------------------------------------------------------------+
void CTimeSeries::RefreshAll(SDataCalculate &data_calculate)
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
      CSeries *series_obj=this.m_list_series.At(i);
      if(series_obj==NULL || !series_obj.IsAvailable() || series_obj.DataTotal()==0)
         continue;
      //--- update the timeseries list
      series_obj.Refresh(data_calculate);
      //--- If the timeseries object features the New bar event
      if(series_obj.IsNewBar(data_calculate.rates.time))
        {
         //--- send the "New bar" event to the control program chart,
         series_obj.SendEvent();
         //--- set the flag indicating the necessity to set the first date in history on the server and in the terminal
         upd=true;
         //--- add the "New bar" event to the list of timeseries events
         //--- in case of successful addition, set the event flag for the timeseries
         if(this.EventAdd(SERIES_EVENTS_NEW_BAR,series_obj.Time(0),series_obj.Timeframe(),series_obj.Symbol()))
            this.m_is_event=true;
        }
     }
//--- if the flag indicating the necessity to set the first date in history on the server and in the terminal is enabled,
//--- set the values of the first date in history on the server and in the terminal
   if(upd)
      this.SetTerminalServerDate();
  }
//+------------------------------------------------------------------+
```

Here I commented on every method code string, so everything should be clear. If you have any questions, feel free to ask them in the comments below.

**This completes the CTimeSeries class of the object of all timeseries for a single symbol.**

The next class is the **CTimeSeriesCollection** collection class of symbol timeseries objects. It also should feature the event functionality since it is "responsible" for obtaining lists with events from all objects storing all timeseries of each symbol used in the program.

Open \\MQL5\\Include\\DoEasy\\Collections\ **TimeSeriesCollection.mqh** and derive it from the extended base class of all library objects:

```
//+------------------------------------------------------------------+
//| Symbol timeseries collection                                     |
//+------------------------------------------------------------------+
class CTimeSeriesCollection : public CBaseObjExt
  {
```

In the public section of the class, declare two methods for returning the object of all timeseries of the specified symbol and returning the timeseries object of the specified symbol and period:

```
public:
//--- Return (1) oneself and (2) the timeseries list
   CTimeSeriesCollection  *GetObject(void)            { return &this;         }
   CArrayObj              *GetList(void)              { return &this.m_list;  }
//--- Return (1) the timeseries object of the specified symbol and (2) the timeseries object of the specified symbol/period
   CTimeSeries            *GetTimeseries(const string symbol);
   CSeries                *GetSeries(const string symbol,const ENUM_TIMEFRAMES timeframe);
```

Let's write its implementation outside the class body.

**The method returning the timesries object of the specified symbol:**

```
//+------------------------------------------------------------------+
//| Return the timeseries object of the specified symbol             |
//+------------------------------------------------------------------+
CTimeSeries *CTimeSeriesCollection::GetTimeseries(const string symbol)
  {
   int index=this.IndexTimeSeries(symbol);
   if(index==WRONG_VALUE)
      return NULL;
   CTimeSeries *timeseries=this.m_list.At(index);
   return timeseries;
  }
//+------------------------------------------------------------------+
```

Here we obtain the index of the timeseries object for naming a symbol using the IndexTimeSeries() method we considered [in the part 37](https://www.mql5.com/en/articles/7663#node03). The obtained index is used to get the timeseries object from the list. If failed to get the index or an object from the list, NULL is returned. Otherwise, we get the pointer to the requested object in the list.

**The method returning the timeseries object of the specified symbol/period:**

```
//+------------------------------------------------------------------+
//| Return the timeseries object of the specified symbol/period      |
//+------------------------------------------------------------------+
CSeries *CTimeSeriesCollection::GetSeries(const string symbol,const ENUM_TIMEFRAMES timeframe)
  {
   CTimeSeries *timeseries=this.GetTimeseries(symbol);
   if(timeseries==NULL)
      return NULL;
   CSeries *series=timeseries.GetSeries(timeframe);
   return series;
  }
//+-----------------------------------------------------------------------+
```

Here we obtain the timeseries object using the GetTimeseries() method (considered above) by a symbol passed to the method.

From the obtained timeseries object, get the timeseries list by a specified timeframe and return the pointer to the obtained timeseries object.

The GetSeries() method of the timeseries object uses the above mentioned **IndexTimeframe()** method to return the required timeseries, while the GetSeries() method of the CTimeSeries timeseries object looks as follows:

```
CSeries *GetSeries(const ENUM_TIMEFRAMES timeframe) { return this.m_list_series.At(this.IndexTimeframe(timeframe)); }
```

In the public section of the class, remove three methods for creating timeseriesleaving only one for creating the specified timeseries of the specified symbol:

```
//--- Create (1) the specified timeseries of the specified symbol, (2) the specified timeseries of all symbols,
//--- (3) all timeseries of the specified symbol and (4) all timeseries of all symbols
   bool                    CreateSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint required=0);
   bool                    CreateSeries(const ENUM_TIMEFRAMES timeframe,const uint required=0);
   bool                    CreateSeries(const string symbol,const uint required=0);
   bool                    CreateSeries(const uint required=0);
//--- Update (1) the specified timeseries of the specified symbol, (2) the specified timeseries of all symbols,
//--- (3) all timeseries of the specified symbol and (4) all timeseries of all symbols and (5) all timeseries except for the current symbol
```

Three removed methods seem redundant here so far. So, let's declare three new methods instead — for re-creating a specified timeseries, for returning an empty timeseries and returning a partially filled timeseries:

```
//--- (1) Create and (2) re-create a specified timeseries of a specified symbol
   bool                    CreateSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,const int rates_total=0,const uint required=0);
   bool                    ReCreateSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,const int rates_total=0,const uint required=0);
//--- Return (1) an empty, (2) partially filled timeseries
   CSeries                *GetSeriesEmpty(void);
   CSeries                *GetSeriesIncompleted(void);
```

Why do we need to re-create a timeseries? When initializing the library and creating all applied timeseries of all symbols, we use the functions initiating the download of historical data. As I have said more than once, if a program is an indicator and refers to a symbol and a timeframe it is launched on, this may cause a conflict. Therefore, such situations are skipped. Upon completion and entering the OnCalculate() handler, we should first revise the created timeseries, get the empty one (skipped during initialization) and re-create it using data from the **rates\_total** variable in OnCalculate().

Now instead of getting timeseries array data from OnCalculate(), the timeseries update methods receive the data structure. Declare the method for getting events from the timeseries object and adding them to the event list of all objects of symbol timeseries collection:

```
//--- Update (1) the specified timeseries of the specified symbol, (2) all timeseries of all symbols
   void                    Refresh(const string symbol,const ENUM_TIMEFRAMES timeframe,SDataCalculate &data_calculate);
   void                    Refresh(SDataCalculate &data_calculate);

//--- Get events from the timeseries object and add them to the list
   bool                    SetEvents(CTimeSeries *timeseries);

//--- Display (1) the complete and (2) short collection description in the journal
   void                    Print(const bool created=true);
   void                    PrintShort(const bool created=true);

//--- Constructor
                           CTimeSeriesCollection();
  };
//+------------------------------------------------------------------+
```

**Implementing methods returning empty and partially filled timeseries:**

```
//+------------------------------------------------------------------+
//|Return the empty (created but not filled with data) timeseries    |
//+------------------------------------------------------------------+
CSeries *CTimeSeriesCollection::GetSeriesEmpty(void)
  {
//--- In the loop by the timeseries object list
   int total_timeseries=this.m_list.Total();
   for(int i=0;i<total_timeseries;i++)
     {
      //--- get the next object of all symbol timeseries by the loop index
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL || !timeseries.IsAvailable())
         continue;
      //--- get the list of timeseries objects from the object of all symbol timeseries
      CArrayObj *list_series=timeseries.GetListSeries();
      if(list_series==NULL)
         continue;
      //--- in the loop by the symbol timeseries list
      int total_series=list_series.Total();
      for(int j=0;j<total_series;j++)
        {
         //--- get the next timeseries
         CSeries *series=list_series.At(j);
         if(series==NULL || !series.IsAvailable())
            continue;
         //--- if the timeseries has no bar objects,

         //--- return the pointer to the timeseries
         if(series.DataTotal()==0)
            return series;
        }
     }
   return NULL;
  }
//+------------------------------------------------------------------+
//| Return partially filled timeseries                               |
//+------------------------------------------------------------------+
CSeries *CTimeSeriesCollection::GetSeriesIncompleted(void)
  {
//--- In the loop by the timeseries object list
   int total_timeseries=this.m_list.Total();
   for(int i=0;i<total_timeseries;i++)
     {
      //--- get the next object of all symbol timeseries by the loop index
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL || !timeseries.IsAvailable())
         continue;
      //--- get the list of timeseries objects from the object of all symbol timeseries
      CArrayObj *list_series=timeseries.GetListSeries();
      if(list_series==NULL)
         continue;
      //--- in the loop by the symbol timeseries list
      int total_series=list_series.Total();
      for(int j=0;j<total_series;j++)
        {
         //--- get the next timeseries
         CSeries *series=list_series.At(j);
         if(series==NULL || !series.IsAvailable())
            continue;
         //--- if the timeseries has bar objects,
         //--- but their number is not equal to the requested and available one for the symbol,
         //--- return the pointer to the timeseries
         if(series.DataTotal()>0 && series.AvailableUsedData()!=series.DataTotal())
            return series;
        }
     }
   return NULL;
  }
//+------------------------------------------------------------------+
```

Each method string is commented and the methods are similar except for the empty and partially filled timeseries check.

The methods return the first oncoming timeseries satisfying the search conditions. This has been done in order to successively get all possible empty or partially filled timeseries on each subsequent tick (entering OnCalculate). This corresponds to MetaQuotes recommendations for the correct handling of insufficient data in indicators — exiting the handler and checking the presence of the data on the next tick.

**Implementing the method for creating the specified timeseries of the specified symbol:**

```
//+------------------------------------------------------------------+
//| Create the specified timeseries of the specified symbol          |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::CreateSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,const int rates_total=0,const uint required=0)
  {
   CTimeSeries *timeseries=this.GetTimeseries(symbol);
   if(timeseries==NULL)
      return false;
   if(!timeseries.AddSeries(timeframe,required))
      return false;
   if(!timeseries.SyncData(timeframe,required,rates_total))
      return false;
   return timeseries.CreateSeries(timeframe,required);
  }
//+------------------------------------------------------------------+
```

The method adds data to the timeseries object of a single symbol — a new timeseries with the specified chart period.

The method receives a symbol and the required timeseries period.

Get the timeseries object and add the new timeseries of the specified chart period to it.

Request symbol/period data and set the necessary amount of data in the timeseries.

If all previous actions are successful, return the result of creating a new timeseries and adding data to it.

We have considered all these methods in the previous articles. Here, I have introduced a new logic of creating the required symbol/period timeseries. The logic is different from the one described [in the article 37](https://www.mql5.com/en/articles/7663#node03).

**Implementing the method for re-creating the specified timeseries of the specified symbol:**

```
//+------------------------------------------------------------------+
//| Re-create a specified timeseries of a specified symbol           |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::ReCreateSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,const int rates_total=0,const uint required=0)
  {
   CTimeSeries *timeseries=this.GetTimeseries(symbol);
   if(timeseries==NULL)
      return false;
   if(!timeseries.SyncData(timeframe,rates_total,required))
      return false;
   return timeseries.CreateSeries(timeframe,required);
  }
//+------------------------------------------------------------------+
```

Here everything is exactly the same with only one difference — the timeseries has already been created, so the step of adding a new timeseries to the object of all symbol timeseries is skipped.

**Implementing the method receiving events from the timeseries object and adding them to the list of timeseries collection events:**

```
//+------------------------------------------------------------------+
//| Get events from the timeseries object and add them to the list   |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::SetEvents(CTimeSeries *timeseries)
  {
//--- Set the flag of successfully adding an event to the list and
//--- get the list of symbol timeseries object events
   bool res=true;
   CArrayObj *list=timeseries.GetListEvents();
   if(list==NULL)
      return false;
//--- In the loop by the obtained list of events,
   int total=list.Total();
   for(int i=0;i<total;i++)
     {
      //--- get the next event by the loop index and
      CEventBaseObj *event=timeseries.GetEvent(i);
      if(event==NULL)
         continue;
      //--- add the result of adding the obtained event to the flag value
      //--- from the symbol timeseries list to the timeseries collection list
      res &=this.EventAdd(event.ID(),event.LParam(),event.DParam(),event.SParam());
     }
//--- Return the result of adding events to the list
   return res;
  }
//+------------------------------------------------------------------+
```

The method receives the pointer to the symbol timeseries object. All its events are added to the list of timeseries collection events in a loop by the object event list.

**Implementing the method of updating a specified timeseries of the specified symbol and adding its events to the list of timeseries collection events:**

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
   CTimeSeries *timeseries=this.GetTimeseries(symbol);
   if(timeseries==NULL)
      return;
//--- If there is no new tick on the timeseries object symbol, exit
   if(!timeseries.IsNewTick())
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

**Implementing the method of updating all timeseries of all symbols and adding their events to the list of timeseries collection events:**

```
//+------------------------------------------------------------------+
//| Update all timeseries of all symbols                             |
//+------------------------------------------------------------------+
void CTimeSeriesCollection::Refresh(SDataCalculate &data_calculate)
  {
//--- Reset the flag of an event in the timeseries collection and clear the event list
   this.m_is_event=false;
   this.m_list_events.Clear();
//--- In the loop by all symbol timeseries objects in the collection,
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      //--- get the next symbol timeseries object
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL)
         continue;
      //--- if there is no new tick on a timeseries symbol, move to the next object in the list
      if(!timeseries.IsNewTick())
         continue;
      //--- Update all symbol timeseries
      timeseries.RefreshAll(data_calculate);
      //--- If the event flag enabled for the symbol timeseries object,
      //--- get events from symbol timeseries, write them to the collection event list
      //--- and set the event flag in the collection
      if(timeseries.IsEvent())
         this.m_is_event=this.SetEvents(timeseries);
     }
  }
//+------------------------------------------------------------------+
```

All these methods are commented in detail and their logic is easy to understand.

**This completes improving all timeseries classes at the current stage.**

Now let's improve the **CEngine** library main object(\\MQL5\\Include\\DoEasy\\Engine.mqh) to work with the timeseries collection from programs.

In the private section of the class, declare the pause object:

```
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
```

In the public section of the class, add the method returning the flag of the event presence in the timeseries collection:

```
//--- Return the (1) hedge account, (2) working in the tester, (3) account event, (4) symbol event and (5) trading event flag
   bool                 IsHedge(void)                             const { return this.m_is_hedge;                             }
   bool                 IsTester(void)                            const { return this.m_is_tester;                            }
   bool                 IsAccountsEvent(void)                     const { return this.m_accounts.IsEvent();                   }
   bool                 IsSymbolsEvent(void)                      const { return this.m_symbols.IsEvent();                    }
   bool                 IsTradeEvent(void)                        const { return this.m_events.IsEvent();                     }
   bool                 IsSeriesEvent(void)                       const { return this.m_time_series.IsEvent();                }
```

The method returns the result of the IsEvent() method operation of the timeseries collection object.

Since the array data from the OnCalculate() handler of the indicator should now be sent to the timeseries update methods for handling the current timeseries data, add passing the OnCalculate() array data structure to the Timer and Tick event handling methods, as well as declare the method of handling the Calculate event:

```
//--- (1) Timer, (2) NewTick event handler and (3) Calculate event handler
   void                 OnTimer(SDataCalculate &data_calculate);
   void                 OnTick(SDataCalculate &data_calculate,const uint required=0);
   int                  OnCalculate(SDataCalculate &data_calculate,const uint required=0);
```

In the same public section of the class, add the method returning the timeseries event list:

```
//--- Return (1) the timeseries collection and (2) the list of timeseries from the timeseries collection and (3) the list of timeseries events
   CTimeSeriesCollection *GetTimeSeriesCollection(void)                       { return &this.m_time_series;                                     }
   CArrayObj           *GetListTimeSeries(void)                               { return this.m_time_series.GetList();                            }
   CArrayObj           *GetListSeriesEvents(void)                             { return this.m_time_series.GetListEvents();                      }
```

The method returns the pointer to the list of timeseries collection events using the GetListEvents() timeseries collection method.

The public section of the class features the four methods for creating various timeseries. Let's temporarily remove three methods we do not need yet:

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
```

and replace them with declaring the method for creating all timeseries of all used collection symbols. Also, write the method for re-creating the specified timeseries and declare the method for requesting the timeseries synchronization with the server:

```
//--- Create (1) the specified timeseries of the specified symbol and (2) all used timeseries of all used symbols
   bool                 SeriesCreate(const string symbol,const ENUM_TIMEFRAMES timeframe,const int rates_total=0,const uint required=0)
                          { return this.m_time_series.CreateSeries(symbol,timeframe,rates_total,required);        }
   bool                 SeriesCreateAll(const string &array_periods[],const int rates_total=0,const uint required=0);
//--- Re-create a specified timeseries of a specified symbol
   bool                 SeriesReCreate(const string symbol,const ENUM_TIMEFRAMES timeframe,const int rates_total=0,const uint required=0)
                          { return this.m_time_series.ReCreateSeries(symbol,timeframe,rates_total,required);      }
//--- Synchronize timeseries data with the server
   void                 SeriesSync(SDataCalculate &data_calculate,const uint required=0);
```

There we also have four methods for updating the timeseries collection.

Leave only two methods — the first one for updating the specified timeseries and the second one for updating all collection timeseries:

```
//--- Update (1) the specified timeseries of the specified symbol, (2) all timeseries of all symbols
   void                 SeriesRefresh(const string symbol,const ENUM_TIMEFRAMES timeframe,SDataCalculate &data_calculate)
                          { this.m_time_series.Refresh(symbol,timeframe,data_calculate);                          }
   void                 SeriesRefresh(SDataCalculate &data_calculate)
                          { this.m_time_series.Refresh(data_calculate);                                           }
```

The structure featuring the data on variables and OnCalculate() arrays is passed to the methods instead of OnCalculate() array values.

Let's add four new methods — for returning the pointer to the timeseries object of the specified symbol, for the specified timeseries object, as well as the methods returning the pointers to an empty and partially filled timeseries:

```
//--- Return (1) the timeseries object of the specified symbol and (2) the timeseries object of the specified symbol/period
   CTimeSeries         *SeriesGetTimeseries(const string symbol)
                          { return this.m_time_series.GetTimeseries(symbol);                                      }
   CSeries             *SeriesGetSeries(const string symbol,const ENUM_TIMEFRAMES timeframe)
                          { return this.m_time_series.GetSeries(symbol,timeframe);                                }
//--- Return (1) an empty, (2) partially filled timeseries
   CSeries             *SeriesGetSeriesEmpty(void)       { return this.m_time_series.GetSeriesEmpty();            }
   CSeries             *SeriesGetSeriesIncompleted(void) { return this.m_time_series.GetSeriesIncompleted();      }
```

The methods return the result of returning same-name methods of timeseries collection we considered above.

The **TradingOnInit()** method passing the pointers to all the necessary collections into the trading class has been renamed to **CollectionOnInit()** since such a name is more suitable for it as necessary initializations of all collection classes are performed in it.

In the end of the class body code, add the block with the methods for working with the pause object:

```
//--- Set the new (1) pause countdown start time and (2) pause in milliseconds
   void                 PauseSetTimeBegin(const ulong time)             { this.m_pause.SetTimeBegin(time);                    }
   void                 PauseSetWaitingMSC(const ulong pause)           { this.m_pause.SetWaitingMSC(pause);                  }
//--- Return (1) the time passed from the pause countdown start in milliseconds, (2) waiting completion flag
//--- (3) pause countdown start time, (4) pause in milliseconds
   ulong                PausePassed(void)                         const { return this.m_pause.Passed();                       }
   bool                 PauseIsCompleted(void)                    const { return this.m_pause.IsCompleted();                  }
   ulong                PauseTimeBegin(void)                      const { return this.m_pause.TimeBegin();                    }
   ulong                PauseTimeWait(void)                       const { return this.m_pause.TimeWait();                     }
//--- Return the description (1) of the time passed till the countdown starts in milliseconds,
//--- (2) pause countdown start time, (3) pause in milliseconds
   string               PausePassedDescription(void)              const { return this.m_pause.PassedDescription();            }
   string               PauseTimeBeginDescription(void)           const { return this.m_pause.TimeBeginDescription();         }
   string               PauseWaitingMSCDescription(void)          const { return this.m_pause.WaitingMSCDescription();        }
   string               PauseWaitingSECDescription(void)          const { return this.m_pause.WaitingSECDescription();        }
//--- Launch the new pause countdown
   void                 Pause(const ulong pause_msc,const datetime time_start=0)
                          {
                           this.PauseSetWaitingMSC(pause_msc);
                           this.PauseSetTimeBegin(time_start*1000);
                           while(!this.PauseIsCompleted() && !IsStopped()){}
                          }

//--- Constructor/destructor
                        CEngine();
                       ~CEngine();
```

The Pause class was described [in the article 30](https://www.mql5.com/en/articles/7481#node02). The class is meant for inserting pauses instead of the [Sleep()](https://www.mql5.com/en/docs/common/sleep) function that does not work in the indicators.

In addition to the already described CPause class methods called from these methods, we added yet another Pause() method allowing us to launch a new waiting for pause without preliminary initialization of its parameters — all parameters are passed to the method while the method features waiting for the pause completion in milliseconds passed to the method as an input. These methods can be useful in programs for organizing pauses in indicators.

Keep in mind that this pause object delays the main thread the indicator has been launched on, just like the Sleep() function.

This pause should be applied in indicators where necessary.

CEngine class timer has been re-arranged — previously we checked where each handler works — in the tester or not. Each handler of all collections had to perform such checks which was unreasonable.

Now we first check where the work is done — not in the tetser or in the tester. The handling of all collections is then performed inside the blocks (non-tester and tester):

```
//+------------------------------------------------------------------+
//| CEngine timer                                                    |
//+------------------------------------------------------------------+
void CEngine::OnTimer(SDataCalculate &data_calculate)
  {
//--- If this is not a tester, work with collection events by timer
   if(!this.IsTester())
     {
   //--- Timer of the collections of historical orders and deals, as well as of market orders and positions
      int index=this.CounterIndex(COLLECTION_ORD_COUNTER_ID);
      CTimerCounter* cnt1=this.m_list_counters.At(index);
      if(cnt1!=NULL)
        {
         //--- If unpaused, work with the order, deal and position collections events
         if(cnt1.IsTimeDone())
            this.TradeEventsControl();
        }
   //--- Account collection timer
      index=this.CounterIndex(COLLECTION_ACC_COUNTER_ID);
      CTimerCounter* cnt2=this.m_list_counters.At(index);
      if(cnt2!=NULL)
        {
         //--- If unpaused, work with the account collection events
         if(cnt2.IsTimeDone())
            this.AccountEventsControl();
        }
   //--- Timer 1 of the symbol collection (updating symbol quote data in the collection)
      index=this.CounterIndex(COLLECTION_SYM_COUNTER_ID1);
      CTimerCounter* cnt3=this.m_list_counters.At(index);
      if(cnt3!=NULL)
        {
         //--- If the pause is over, update quote data of all symbols in the collection
         if(cnt3.IsTimeDone())
            this.m_symbols.RefreshRates();
        }
   //--- Timer 2 of the symbol collection (updating all data of all symbols in the collection and tracking symbl and symbol search events in the market watch window)
      index=this.CounterIndex(COLLECTION_SYM_COUNTER_ID2);
      CTimerCounter* cnt4=this.m_list_counters.At(index);
      if(cnt4!=NULL)
        {
         //--- If the pause is over
         if(cnt4.IsTimeDone())
           {
            //--- update data and work with events of all symbols in the collection
            this.SymbolEventsControl();
            //--- When working with the market watch list, check the market watch window events
            if(this.m_symbols.ModeSymbolsList()==SYMBOLS_MODE_MARKET_WATCH)
               this.MarketWatchEventsControl();
           }
        }
   //--- Trading class timer
      index=this.CounterIndex(COLLECTION_REQ_COUNTER_ID);
      CTimerCounter* cnt5=this.m_list_counters.At(index);
      if(cnt5!=NULL)
        {
         //--- If unpaused, work with the list of pending requests
         if(cnt5.IsTimeDone())
            this.m_trading.OnTimer();
        }
   //--- Timeseries collection timer
      index=this.CounterIndex(COLLECTION_TS_COUNTER_ID);
      CTimerCounter* cnt6=this.m_list_counters.At(index);
      if(cnt6!=NULL)
        {
         //--- If unpaused, work with the timeseries list
         if(cnt6.IsTimeDone())
            this.SeriesRefresh(data_calculate);
        }
     }
//--- If this is a tester, work with collection events by tick
   else
     {
      //--- work with events of collections of orders, deals and positions by tick
      this.TradeEventsControl();
      //--- work with events of collections of accounts by tick
      this.AccountEventsControl();
      //--- update quote data of all collection symbols by tick
      this.m_symbols.RefreshRates();
      //--- work with events of all symbols in the collection by tick
      this.SymbolEventsControl();
      //--- work with the list of pending orders by tick
      this.m_trading.OnTimer();
      //--- work with the timeseries list by tick
      this.SeriesRefresh(data_calculate);
     }
  }
//+------------------------------------------------------------------+
```

The handler has become more compact and features more comprehensible logic. Besides, it is now relieved of unnecessary repeating checks.

**The method synchronizing empty timeseries data with the server and recreating the empty timeseries:**

```
//+------------------------------------------------------------------+
//| Synchronize timeseries data with the server                      |
//+------------------------------------------------------------------+
void CEngine::SeriesSync(SDataCalculate &data_calculate,const uint required=0)
  {
//--- If the timeseries data is not calculated, try re-creating the timeseries
//--- Get the pointer to the empty timeseries
   CSeries *series=this.SeriesGetSeriesEmpty();
   if(series!=NULL)
     {
      //--- Display the empty timeseries data as a chart comment and try synchronizing the timeseries with the server data
      ::Comment(series.Header(),": ",CMessage::Text(MSG_LIB_TEXT_TS_TEXT_WAIT_FOR_SYNC));
      ::ChartRedraw(::ChartID());
      //--- if the data has been synchronized
      if(series.SyncData(0,data_calculate.rates_total))
        {
         //--- if managed to re-create the timeseries
         if(this.m_time_series.ReCreateSeries(series.Symbol(),series.Timeframe(),data_calculate.rates_total))
           {
            //--- display the chart comment and the journal entry with the re-created timeseries data
            ::Comment(series.Header(),": OK");
            ::ChartRedraw(::ChartID());
            Print(series.Header()," ",CMessage::Text(MSG_LIB_TEXT_TS_TEXT_CREATED_OK),":");
            series.PrintShort();
           }
        }
     }
//--- Delete all comments
   else
     {
      ::Comment("");
      ::ChartRedraw(::ChartID());
     }
  }
//+------------------------------------------------------------------+
```

The method is a cornerstone for the correct loading of historical data of any timeseries used — any symbols and any periods of the charts.

The method receives the first unfilled timeseries from the timeseries collection, which means it had no data one tick before. The attempt to synchronize the timeseries data with the server data is performed immediately. If failed, exit the method till the next tick. If the data has been synchronized, the timeseries is re-created — filled by all available (but not more than the requested quantity) bars from history.

The process is performed on every tick — we get the next empty timeseries, synchronize and re-create it till no empty timeseries remain.

Implementing NewTick and Calculate event handlers:

```
//+------------------------------------------------------------------+
//| NewTick event handler                                            |
//+------------------------------------------------------------------+
void CEngine::OnTick(SDataCalculate &data_calculate,const uint required=0)
  {
//--- If this is not a EA, exit
   if(this.m_program!=PROGRAM_EXPERT)
      return;
//--- Re-create empty timeseries
   this.SeriesSync(data_calculate,required);
//--- end
  }
//+------------------------------------------------------------------+
//| Calculate event handler                                          |
//+------------------------------------------------------------------+
int CEngine::OnCalculate(SDataCalculate &data_calculate,const uint required=0)
  {
//--- If this is not an indicator, exit
   if(this.m_program!=PROGRAM_INDICATOR)
      return data_calculate.rates_total;
//--- Re-create empty timeseries
   this.SeriesSync(data_calculate,required);
//--- return rates_total
   return data_calculate.rates_total;
  }
//+------------------------------------------------------------------+
```

The method for re-creating empty timeseries is called in both methods.

The methods themselves are to be called from same-name program handlers based on the library.

**Implementing the methods for creating all applied timeseries of all used symbols:**

```
//+------------------------------------------------------------------+
//| Create all applied timeseries of all used symbols                |
//+------------------------------------------------------------------+
bool CEngine::SeriesCreateAll(const string &array_periods[],const int rates_total=0,const uint required=0)
  {
//--- Set the flag of successful creation of all timeseries of all symbols
   bool res=true;
//--- Get the list of all used symbols
   CArrayObj* list_symbols=this.GetListAllUsedSymbols();
   if(list_symbols==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_FAILED_GET_SYMBOLS_ARRAY));
      return false;
     }
   //--- In the loop by the total number of symbols
   for(int i=0;i<list_symbols.Total();i++)
     {
      //--- get the next symbol object
      CSymbol *symbol=list_symbols.At(i);
      if(symbol==NULL)
        {
         ::Print(DFUN,"index ",i,": ",CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
         continue;
        }
      //--- In the loop by the total number of used timeframes,
      int total_periods=::ArraySize(array_periods);
      for(int j=0;j<total_periods;j++)
        {
         //--- create the timeseries object of the next symbol.
         //--- Add the timeseries creation result to the res variable
         ENUM_TIMEFRAMES timeframe=TimeframeByDescription(array_periods[j]);
         res &=this.SeriesCreate(symbol.Name(),timeframe,rates_total,required);
        }
     }
//--- Return the result of creating all timeseries for all symbols
   return res;
  }
//+------------------------------------------------------------------+
```

The method is to be called during the program initialization after creating the list of all used symbols.

The method receives the array created during initialization. The array contains the names of used chart periods and parameters for creating timeseries — the number of the current timeseries bars (only for indicators — rates\_total) and the necessary history depth for created timeseries (the default is 1000, but not more than the symbol's Bars() value and not more than rates\_total for indicators).

**Currently, these are all the necessary improvements for working with timeseries.**

### Testing timeseries and their events in indicators

To test the work of the timeseries collection class in indicators, create a new folder in the terminal indicator directory \\MQL5\\Indicators\ **TestDoEasy\**. Let's create a new subfolder **Part39\** there with a new indicator **TestDoEasyPart39.mq5** inside.

The number and type of drawn indicator buffers does not matter to us so far since we are not going to draw anything in it. However, I have set two drawn buffers of the [DRAW\_LINE](https://www.mql5.com/en/docs/customind/indicators_examples/draw_line) drawing type for future use.

The necessary indicator inputs for setting the necessary symbols and timeframes, as well as some other inputs have been taken [from the test EA described in the previous article](https://www.mql5.com/en/articles/7695#node04). Here is how it looks now:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart39.mq5 |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
//--- enums
//--- defines
//--- structures
//--- properties
#property indicator_chart_window
#property indicator_buffers 2
#property indicator_plots   2
//--- plot Label1
#property indicator_label1  "Label1"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrRed
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1
//--- plot Label2
#property indicator_label2  "Label2"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrGreen
#property indicator_style2  STYLE_SOLID
#property indicator_width2  1
//--- indicator buffers
double         Buffer1[];
double         Buffer2[];
//--- input variables
sinput   ENUM_SYMBOLS_MODE InpModeUsedSymbols   =  SYMBOLS_MODE_CURRENT;            // Mode of used symbols list
sinput   string            InpUsedSymbols       =  "EURUSD,AUDUSD,EURAUD,EURCAD,EURGBP,EURJPY,EURUSD,GBPUSD,NZDUSD,USDCAD,USDJPY";  // List of used symbols (comma - separator)
sinput   ENUM_TIMEFRAMES_MODE InpModeUsedTFs    =  TIMEFRAMES_MODE_LIST;            // Mode of used timeframes list
sinput   string            InpUsedTFs           =  "M1,M5,M15,M30,H1,H4,D1,W1,MN1"; // List of used timeframes (comma - separator)
sinput   bool              InpUseSounds         =  true; // Use sounds
//--- global variables
CEngine        engine;                          // CEngine library main object
string         prefix;                          // Prefix of graphical object names
bool           testing;                         // Flag of working in the tester
int            used_symbols_mode;               // Mode of working with symbols
string         array_used_symbols[];            // Array of used symbols
string         array_used_periods[];            // Array of used timeframes
//+------------------------------------------------------------------+
```

In the indicator's OnInit() handler, implement setting indicator global variables and calling the library initialization function:

```
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping
   SetIndexBuffer(0,Buffer1,INDICATOR_DATA);
   SetIndexBuffer(1,Buffer2,INDICATOR_DATA);

//--- Set indicator global variables
   prefix=MQLInfoString(MQL_PROGRAM_NAME)+"_";
   testing=engine.IsTester();
   ZeroMemory(rates_data);

//--- Initialize DoEasy library
   OnInitDoEasy();

//--- Check and remove remaining indicator graphical objects
   if(IsPresentObectByPrefix(prefix))
      ObjectsDeleteAll(0,prefix);

//--- Check playing a standard sound using macro substitutions
   engine.PlaySoundByDescription(SND_OK);
//--- Wait for 600 milliseconds
   engine.Pause(600);
   engine.PlaySoundByDescription(SND_NEWS);

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

The indicator's OnDeinit() handler is taken from the test EA described in the previous article:

```
//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                       |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Remove indicator graphical objects by an object name prefix
   ObjectsDeleteAll(0,prefix);
   Comment("");
  }
//+------------------------------------------------------------------+
```

Let's take the OnTimer() and OnChartEvent() handlers from the EA as well:

```
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
//--- Launch the library timer (only not in the tester)
   if(!MQLInfoInteger(MQL_TESTER))
      engine.OnTimer(rates_data);
  }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//--- If working in the tester, exit
   if(MQLInfoInteger(MQL_TESTER))
      return;
//--- Handling mouse events
   if(id==CHARTEVENT_OBJECT_CLICK)
     {
      //--- Handling pressing the buttons in the panel
      if(StringFind(sparam,"BUTT_")>0)
         PressButtonEvents(sparam);
     }
//--- Handling DoEasy library events
   if(id>CHARTEVENT_CUSTOM-1)
     {
      OnDoEasyEvent(id,lparam,dparam,sparam);
     }
  }
//+------------------------------------------------------------------+
```

Create two functions for filling the structure of array and variable data from the indicator's first and second OnCalculate() forms:

```
//+------------------------------------------------------------------+
//| Copy data from the first OnCalculate() form to the structure     |
//+------------------------------------------------------------------+
void CopyData(SDataCalculate &data_calculate,
              const int rates_total,
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
   data_calculate.rates_total=rates_total;
   data_calculate.prev_calculated=prev_calculated;
   data_calculate.begin=begin;
   data_calculate.price=price[0];
//--- Return the array's initial indexing direction
   if(!as_series_price)
      ArraySetAsSeries(price,false);
  }
//+------------------------------------------------------------------+
//| Copy data from the second OnCalculate() form to the structure    |
//+------------------------------------------------------------------+
void CopyData(SDataCalculate &data_calculate,
              const int rates_total,
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
   data_calculate.rates_total=rates_total;
   data_calculate.prev_calculated=prev_calculated;
   data_calculate.rates.time=time[0];
   data_calculate.rates.open=open[0];
   data_calculate.rates.high=high[0];
   data_calculate.rates.low=low[0];
   data_calculate.rates.close=close[0];
   data_calculate.rates.tick_volume=tick_volume[0];
   data_calculate.rates.real_volume=(#ifdef __MQL5__ volume[0] #else 0 #endif);
   data_calculate.rates.spread=(#ifdef __MQL5__ spread[0] #else 0 #endif);
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

Move the function of handling DoEasy library events from the test EA:

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
   if(source==COLLECTION_SYMBOLS_ID)
     {
      CSymbol *symbol=engine.GetSymbolObjByName(sparam);
      if(symbol==NULL)
         return;
      //--- Number of decimal places in the event value - in case of a 'long' event, it is 0, otherwise - Digits() of a symbol
      int digits=(idx<SYMBOL_PROP_INTEGER_TOTAL ? 0 : symbol.Digits());
      //--- Event text description
      string id_descr=(idx<SYMBOL_PROP_INTEGER_TOTAL ? symbol.GetPropertyDescription((ENUM_SYMBOL_PROP_INTEGER)idx) : symbol.GetPropertyDescription((ENUM_SYMBOL_PROP_DOUBLE)idx));
      //--- Property change text value
      string value=DoubleToString(dparam,digits);

      //--- Check event reasons and display its description in the journal
      if(reason==BASE_EVENT_REASON_INC)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_DEC)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_MORE_THEN)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_LESS_THEN)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_EQUALS)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
     }

//--- Handling account events
   else if(source==COLLECTION_ACCOUNT_ID)
     {
      CAccount *account=engine.GetAccountCurrent();
      if(account==NULL)
         return;
      //--- Number of decimal places in the event value - in case of a 'long' event, it is 0, otherwise - Digits() of a symbol
      int digits=int(idx<ACCOUNT_PROP_INTEGER_TOTAL ? 0 : account.CurrencyDigits());
      //--- Event text description
      string id_descr=(idx<ACCOUNT_PROP_INTEGER_TOTAL ? account.GetPropertyDescription((ENUM_ACCOUNT_PROP_INTEGER)idx) : account.GetPropertyDescription((ENUM_ACCOUNT_PROP_DOUBLE)idx));
      //--- Property change text value
      string value=DoubleToString(dparam,digits);

      //--- Checking event reasons and handling the increase of funds by a specified value,

      //--- In case of a property value increase
      if(reason==BASE_EVENT_REASON_INC)
        {
         //--- Display an event in the journal
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
         //--- if this is an equity increase
         if(idx==ACCOUNT_PROP_EQUITY)
           {
            //--- Get the list of all open positions for the current symbol
            CArrayObj* list_positions=engine.GetListMarketPosition();
            list_positions=CSelect::ByOrderProperty(list_positions,ORDER_PROP_SYMBOL,Symbol(),EQUAL);
            //--- Select positions with the profit exceeding zero
            list_positions=CSelect::ByOrderProperty(list_positions,ORDER_PROP_PROFIT_FULL,0,MORE);
            if(list_positions!=NULL)
              {
               //--- Sort the list by profit considering commission and swap
               list_positions.Sort(SORT_BY_ORDER_PROFIT_FULL);
               //--- Get the position index with the highest profit
               int index=CSelect::FindOrderMax(list_positions,ORDER_PROP_PROFIT_FULL);
               if(index>WRONG_VALUE)
                 {
                  COrder* position=list_positions.At(index);
                  if(position!=NULL)
                    {
                     //--- Get a ticket of a position with the highest profit and close the position by a ticket
                     engine.ClosePosition(position.Ticket());
                    }
                 }
              }
           }
        }
      //--- Other events are simply displayed in the journal
      if(reason==BASE_EVENT_REASON_DEC)
        {
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_MORE_THEN)
        {
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_LESS_THEN)
        {
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_EQUALS)
        {
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
     }

//--- Handling market watch window events
   else if(idx>MARKET_WATCH_EVENT_NO_EVENT && idx<SYMBOL_EVENTS_NEXT_CODE)
     {
      //--- Market Watch window event
      string descr=engine.GetMWEventDescription((ENUM_MW_EVENT)idx);
      string name=(idx==MARKET_WATCH_EVENT_SYMBOL_SORT ? "" : ": "+sparam);
      Print(TimeMSCtoString(lparam)," ",descr,name);
     }

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
   else if(idx>TRADE_EVENT_NO_EVENT && idx<TRADE_EVENTS_NEXT_CODE)
     {
      //--- Get the list of trading events
      CArrayObj *list=engine.GetListAllOrdersEvents();
      if(list==NULL)
         return;
      //--- get the event index shift relative to the end of the list
      //--- in the tester, the shift is passed by the lparam parameter to the event handler
      //--- outside the tester, events are sent one by one and handled in OnChartEvent()
      int shift=(testing ? (int)lparam : 0);
      CEvent *event=list.At(list.Total()-1-shift);
      if(event==NULL)
      return;
      //--- Accrue the credit
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_CREDIT)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Additional charges
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_CHARGE)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Correction
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_CORRECTION)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Enumerate bonuses
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_BONUS)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Additional commissions
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_COMISSION)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Daily commission
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_COMISSION_DAILY)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Monthly commission
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_COMISSION_MONTHLY)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Daily agent commission
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_COMISSION_AGENT_DAILY)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Monthly agent commission
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_COMISSION_AGENT_MONTHLY)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Interest rate
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_INTEREST)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Canceled buy deal
      if(event.TypeEvent()==TRADE_EVENT_BUY_CANCELLED)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Canceled sell deal
      if(event.TypeEvent()==TRADE_EVENT_SELL_CANCELLED)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Dividend operations
      if(event.TypeEvent()==TRADE_EVENT_DIVIDENT)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Accrual of franked dividend
      if(event.TypeEvent()==TRADE_EVENT_DIVIDENT_FRANKED)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Tax charges
      if(event.TypeEvent()==TRADE_EVENT_TAX)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Replenishing account balance
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_BALANCE_REFILL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Withdrawing funds from balance
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_BALANCE_WITHDRAWAL)
        {
         Print(DFUN,event.TypeEventDescription());
        }

      //--- Pending order placed
      if(event.TypeEvent()==TRADE_EVENT_PENDING_ORDER_PLASED)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Pending order removed
      if(event.TypeEvent()==TRADE_EVENT_PENDING_ORDER_REMOVED)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Pending order activated by price
      if(event.TypeEvent()==TRADE_EVENT_PENDING_ORDER_ACTIVATED)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Pending order partially activated by price
      if(event.TypeEvent()==TRADE_EVENT_PENDING_ORDER_ACTIVATED_PARTIAL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position opened
      if(event.TypeEvent()==TRADE_EVENT_POSITION_OPENED)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position opened partially
      if(event.TypeEvent()==TRADE_EVENT_POSITION_OPENED_PARTIAL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed by an opposite one
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_BY_POS)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed by StopLoss
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_BY_SL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed by TakeProfit
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_BY_TP)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position reversal by a new deal (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_REVERSED_BY_MARKET)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position reversal by activating a pending order (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_REVERSED_BY_PENDING)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position reversal by partial market order execution (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_REVERSED_BY_MARKET_PARTIAL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position reversal by activating a pending order (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_REVERSED_BY_PENDING_PARTIAL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Added volume to a position by a new deal (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Added volume to a position by partial execution of a market order (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET_PARTIAL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Added volume to a position by activating a pending order (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Added volume to a position by partial activation of a pending order (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING_PARTIAL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed partially
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_PARTIAL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position partially closed by an opposite one
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed partially by StopLoss
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_SL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed partially by TakeProfit
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_TP)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- StopLimit order activation
      if(event.TypeEvent()==TRADE_EVENT_TRIGGERED_STOP_LIMIT_ORDER)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order price
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_PRICE)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order and StopLoss price
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_PRICE_SL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order and TakeProfit price
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_PRICE_TP)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order, StopLoss and TakeProfit price
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_PRICE_SL_TP)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order's StopLoss and TakeProfit price
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_SL_TP)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order's StopLoss
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_SL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order's TakeProfit
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_TP)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing position's StopLoss and TakeProfit
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_POSITION_SL_TP)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing position StopLoss
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_POSITION_SL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing position TakeProfit
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_POSITION_TP)
        {
         Print(DFUN,event.TypeEventDescription());
        }
     }
  }
//+------------------------------------------------------------------+
```

The function of working with the library events in the tester from the EA:

```
//+------------------------------------------------------------------+
//| Working with events in the tester                                |
//+------------------------------------------------------------------+
void EventsHandling(void)
  {
//--- If a trading event is present
   if(engine.IsTradeEvent())
     {
      //--- Number of trading events occurred simultaneously
      int total=engine.GetTradeEventsTotal();
      for(int i=0;i<total;i++)
        {
         //--- Get the next event from the list of simultaneously occurred events by index
         CEventBaseObj *event=engine.GetTradeEventByIndex(i);
         if(event==NULL)
            continue;
         long   lparam=i;
         double dparam=event.DParam();
         string sparam=event.SParam();
         OnDoEasyEvent(CHARTEVENT_CUSTOM+event.ID(),lparam,dparam,sparam);
        }
     }
//--- If there is an account event
   if(engine.IsAccountsEvent())
     {
      //--- Get the list of all account events occurred simultaneously
      CArrayObj* list=engine.GetListAccountEvents();
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
            OnDoEasyEvent(CHARTEVENT_CUSTOM+event.ID(),lparam,dparam,sparam);
           }
        }
     }
//--- If there is a symbol collection event
   if(engine.IsSymbolsEvent())
     {
      //--- Get the list of all symbol events occurred simultaneously
      CArrayObj* list=engine.GetListSymbolsEvents();
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
            OnDoEasyEvent(CHARTEVENT_CUSTOM+event.ID(),lparam,dparam,sparam);
           }
        }
     }
//--- If there is a timeseries collection event
   if(engine.IsSeriesEvent())
     {
      //--- Get the list of all timeseries events occurred simultaneously
      CArrayObj* list=engine.GetListSeriesEvents();
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
            OnDoEasyEvent(CHARTEVENT_CUSTOM+event.ID(),lparam,dparam,sparam);
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

We do not need to relocate the EA functions for working with the trading panel buttons. However, let's do that anyway with some slight changes to be able to use buttons in the indicator (two buttons are to be implemented):

```
//+------------------------------------------------------------------+
//| Return the button status                                         |
//+------------------------------------------------------------------+
bool ButtonState(const string name)
  {
   return (bool)ObjectGetInteger(0,name,OBJPROP_STATE);
  }
//+------------------------------------------------------------------+
//| Set the button status                                            |
//+------------------------------------------------------------------+
void ButtonState(const string name,const bool state)
  {
   ObjectSetInteger(0,name,OBJPROP_STATE,state);
//--- Button 1
   if(name=="BUTT_1")
     {
      if(state)
         ObjectSetInteger(0,name,OBJPROP_BGCOLOR,C'220,255,240');
      else
         ObjectSetInteger(0,name,OBJPROP_BGCOLOR,C'240,240,240');
     }
//--- Button 2
   if(name=="BUTT_2")
     {
      if(state)
         ObjectSetInteger(0,name,OBJPROP_BGCOLOR,C'255,220,90');
      else
         ObjectSetInteger(0,name,OBJPROP_BGCOLOR,C'240,240,240');
     }
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
//| Handle pressing the buttons                                      |
//+------------------------------------------------------------------+
void PressButtonEvents(const string button_name)
  {
   //--- Convert button name into its string ID
   string button=StringSubstr(button_name,StringLen(prefix));
   //--- If the button is pressed
   if(ButtonState(button_name))
     {
      //--- If button 1 is pressed
      if(button=="BUTT_1")
        {

        }
      //--- If button 2 is pressed
      else if(button=="BUTT_2")
        {

        }
      //--- Wait for 1/10 of a second
      engine.Pause(100);
      //--- "Unpress" the button (if this is neither a trailing button, nor the buttons enabling pending requests)
      ButtonState(button_name,false);
      //--- re-draw the chart
      ChartRedraw();
     }
   //--- Not pressed
   else
     {
      //--- button 1
      if(button=="BUTT_1")
        {
         ButtonState(button_name,false);
        }
      //--- button 2
      if(button=="BUTT_2")
        {
         ButtonState(button_name,false);
        }
      //--- re-draw the chart
      ChartRedraw();
     }
  }
//+------------------------------------------------------------------+
```

As we can see, most EA functions can be used in indicators with no need for adjustments. This suggests that all necessary functions for working with the library from EAs and indicators should be moved to the library include file. But this will be done later. Currently, we need to create the OnCalculate() handler of the indicator.

The handler is to consist of the essential code block for preparing library data and the optional (for now) code block for working with the indicator:

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
//| OnCalculate code block for working with the library:             |
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
//| OnCalculate code block for working with the indicator:           |
//+------------------------------------------------------------------+
//--- Arrange resource-saving indicator calculations
//--- Set OnCalculate arrays as timeseries
   ArraySetAsSeries(open,true);
   ArraySetAsSeries(high,true);
   ArraySetAsSeries(low,true);
   ArraySetAsSeries(close,true);
   ArraySetAsSeries(tick_volume,true);
   ArraySetAsSeries(volume,true);
   ArraySetAsSeries(spread,true);

//--- Setting buffer arrays as timeseries
   ArraySetAsSeries(Buffer1,true);
   ArraySetAsSeries(Buffer2,true);

//--- Check for the minimum number of bars for calculation
   if(rates_total<2 || Point()==0) return 0;

//--- Check and calculate the number of calculated bars
   int limit=rates_total-prev_calculated;
   if(limit>1)
     {
      limit=rates_total-1;
      ArrayInitialize(Buffer1,EMPTY_VALUE);
      ArrayInitialize(Buffer2,EMPTY_VALUE);
     }
//--- Prepare data
   for(int i=limit; i>=0 && !IsStopped(); i--)
     {
      // the code for preparing indicator calculation buffers
     }

//--- Calculate the indicator
   for(int i=limit; i>=0 && !IsStopped(); i--)
     {
      Buffer1[i]=high[i];
      Buffer2[i]=low[i];
     }

//--- return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

As we can see, everything related to the library operation fits into a small code block in the OnCalculate() handler. In fact, the difference between an EA is that we fill in the price structure of the current array data from OnCalculate() using the CopyData() function, while everything else is absolutely identical to working in an EA — the library works in the timer if the indicator is launched on a symbol chart and in OnCalculate() by ticks if the indicator is launched in the tester.

Fill the indicator buffers in the OnCalculate() calculation part with high\[\] and low\[\] array data.

The full indicator code can be viewed in the files attached below.

Compile the indicator and launch it on the symbol chart we have not worked with for a long time (while setting working with the current symbol in the settings beforehand) and select working with the specified timeframe list. Launching the indicator on long unused symbols makes the indicator to download missing data and inform of that in the journal and on the chart:

![](https://c.mql5.com/2/38/5KTQtoIjOK.gif)

Here we can see that each next empty timeseries has been synchronized and created at each new tick. The following entries have been displayed in the journal:

```
Account 8550475: Artyom Trishkin (MetaQuotes Software Corp.) 10425.23 USD, 1:100, Hedge, MetaTrader 5 demo
--- Initializing "DoEasy" library ---
Working with the current symbol only: "USDCAD"
Working with the specified timeframe list:
"M1"  "M5"  "M15" "M30" "H1"  "H4"  "D1"  "W1"  "MN1"
USDCAD symbol timeseries:
- Timeseries "USDCAD" M1: Requested: 1000, Actual: 0, Created: 0, On the server: 0
- Timeseries "USDCAD" M5: Requested: 1000, Actual: 0, Created: 0, On the server: 0
- Timeseries "USDCAD" M15: Requested: 1000, Actual: 0, Created: 0, On the server: 0
- Timeseries "USDCAD" M30: Requested: 1000, Actual: 0, Created: 0, On the server: 0
- Timeseries "USDCAD" H1: Requested: 1000, Actual: 0, Created: 0, On the server: 0
- Timeseries "USDCAD" H4: Requested: 1000, Actual: 0, Created: 0, On the server: 0
- Timeseries "USDCAD" D1: Requested: 1000, Actual: 0, Created: 0, On the server: 0
- Timeseries "USDCAD" W1: Requested: 1000, Actual: 0, Created: 0, On the server: 0
- Timeseries "USDCAD" MN1: Requested: 1000, Actual: 0, Created: 0, On the server: 0
Library initialization time: 00:00:01.406
"USDCAD" M1 timeseries created successfully:
- Timeseries "USDCAD" M1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5001
"USDCAD" M5 timeseries created successfully:
- Timeseries "USDCAD" M5: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5741
"USDCAD" M15 timeseries created successfully:
- Timeseries "USDCAD" M15: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5247
"USDCAD" M30 timeseries created successfully:
- Timeseries "USDCAD" M30: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5123
"USDCAD" H1 timeseries created successfully:
- Timeseries "USDCAD" H1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 6257
"USDCAD" H4 timeseries created successfully:
- Timeseries "USDCAD" H4: Requested: 1000, Actual: 1000, Created: 1000, On the server: 6232
"USDCAD" D1 timeseries created successfully:
- Timeseries "USDCAD" D1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5003
"USDCAD" W1 timeseries created successfully:
- Timeseries "USDCAD" W1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 1403
"USDCAD" MN1 timeseries created successfully:
- Timeseries "USDCAD" MN1: Requested: 1000, Actual: 323, Created: 323, On the server: 323
New bar on USDCAD M1: 2020.03.19 12:18
New bar on USDCAD M1: 2020.03.19 12:19
New bar on USDCAD M1: 2020.03.19 12:20
New bar on USDCAD M5: 2020.03.19 12:20
```

Here we can see that all requested timeseries have been created when initializing the library. However, they have not been filled with data due to its absence. During the first access to the requested data, data download by the terminal has been initiated. Upon arrival of each subsequent tick, we have received another empty timeseries object, synchronized its data with the server and filled the timeseries object with bar data in the requested quantity. Only 323 bars are actually available on MN1. All of them have been added to the timeseries list.

Now let's launch the indicator in the tester visual mode with the same settings:

![](https://c.mql5.com/2/38/ScoFhgLbwR.gif)

The tester loads all the necessary history for all used timeframes, the library informs of creating all timeseries except the current one. The timeseries for the current symbol and period is successfully recreated on the first entry in OnCalculate(). After unpausing the tester, we can see how the "New bar" events of used timeseries are triggered in the tester.

Everything works as expected.

### What's next?

In the next article, we will continue our work with indicator timeseries and test using the created timeseries for displaying info on a chart.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7724#node00)

**Previous articles within the series:**

[Timeseries in DoEasy library (part 35): Bar object and symbol timeseries list](https://www.mql5.com/en/articles/7594)

[Timeseries in DoEasy library (part 36): Object of timeseries for all used symbol periods](https://www.mql5.com/en/articles/7627)

[Timeseries in DoEasy library (part 37): Timeseries collection - database of timeseries by symbols and periods](https://www.mql5.com/en/articles/7663)

[Timeseries in DoEasy library (part 38): Timeseries collection - real-time updates and accessing data from the program](https://www.mql5.com/en/articles/7695)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7724](https://www.mql5.com/ru/articles/7724)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7724.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7724/mql5.zip "Download MQL5.zip")(3715.65 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7724/mql4.zip "Download MQL4.zip")(3715.65 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/346874)**
(21)


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
11 Apr 2021 at 08:47

Everything is working:

![](https://c.mql5.com/3/352/dvWpQqNPJB.gif)

![jewelnguyen](https://c.mql5.com/avatar/avatar_na2.png)

**[jewelnguyen](https://www.mql5.com/en/users/jewelnguyen)**
\|
11 Apr 2021 at 09:33

I'm just like you, why doesn't it work? Do you try on mt4 or mt5? I try on mt5, here you:

2021.04.11 14:29:21.6462017.01.02 09:01:18   failed market sell 0.1 GBPUSD sl: 1.23561 tp: 1.23261 \[Unsupported filling mode\]

2021.04.11 14:29:21.6462017.01.02 09:01:18   Trading attempt #2. Error : Invalid [order filling type](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_property_integer "MQL5 documentation:")

I need it to test my manual strategy, please help me

Thank you very much

Jewel

PS: I tried on mt4 working very well, but on mt5 it got the same error as above


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
11 Apr 2021 at 19:04

**jewelnguyen :**

I'm just like you, why doesn't it work? Do you try on mt4 or mt5? I try on mt5, here you:

2021.04.11 14:29:21.6462017.01.02 09:01:18   failed market sell 0.1 GBPUSD sl: 1.23561 tp: 1.23261 \[Unsupported filling mode\]

2021.04.11 14:29:21.6462017.01.02 09:01:18   Trading attempt #2. Error : Invalid [order filling type](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_property_integer "MQL5 documentation:")

I need it to test my manual strategy, please help me

Thank you very much

Jewel

PS: I tried on mt4 working very well, but on mt5 it got the same error as above

You need to set the correct order execution policy yourself ( [ENUM\_ORDER\_TYPE\_FILLING](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type_filling)). Use for this in the OnInit () handler:

```
...
...
...
 //---
   engine.TradingSetTypeFilling(ORDER_FILLING_XXX);
   return (INIT_SUCCEEDED);
  }
 //+------------------------------------------------------------------+
```

There are only three possible values:

- ORDER\_FILLING\_FOK \- This filling policy means that an order can be filled only in the specified amount. If the necessary amount of a financial instrument is currently unavailable in the market, the order will not be executed. The required volume can be filled using several offers available on the market at the moment.

- ORDER\_FILLING\_IOC \- This mode means that a trader agrees to execute a deal with the volume maximally available in the market within that indicated in the order. In case the entire volume of an order cannot be filled, the available volume of it will be filled, and the remaining volume will be canceled.

- ORDER\_FILLING\_RETURN \- This policy is used only for market orders (ORDER\_TYPE\_BUY and ORDER\_TYPE\_SELL), limit and stop limit orders (ORDER\_TYPE\_BUY\_LIMIT, ORDER\_TYPE\_SELL\_LIMIT, ORDER\_TYPE\_LIMIT\_STYPELL) In case of partial filling a market or limit order with remaining volume is not canceled but processed further.

For the activation of the ORDER\_TYPE\_BUY\_STOP\_LIMIT and ORDER\_TYPE\_SELL\_STOP\_LIMIT orders, a corresponding limit order ORDER\_TYPE\_BUY\_LIMIT / ORDER\_TYPE\_SELL\_LIMIT with the ORDER\_FILLING\_RETURN execution type is created.


![jewelnguyen](https://c.mql5.com/avatar/avatar_na2.png)

**[jewelnguyen](https://www.mql5.com/en/users/jewelnguyen)**
\|
12 Apr 2021 at 03:34

**Artyom Trishkin:**

You need to set the correct order execution policy yourself ( [ENUM\_ORDER\_TYPE\_FILLING](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type_filling)). Use for this in the OnInit () handler:

There are only three possible values:

- ORDER\_FILLING\_FOK \- This filling policy means that an order can be filled only in the specified amount. If the necessary amount of a financial instrument is currently unavailable in the market, the order will not be executed. The required volume can be filled using several offers available on the market at the moment.

- ORDER\_FILLING\_IOC \- This mode means that a trader agrees to execute a deal with the volume maximally available in the market within that indicated in the order. In case the entire volume of an order cannot be filled, the available volume of it will be filled, and the remaining volume will be canceled.

- ORDER\_FILLING\_RETURN \- This policy is used only for market orders (ORDER\_TYPE\_BUY and ORDER\_TYPE\_SELL), limit and stop limit orders (ORDER\_TYPE\_BUY\_LIMIT, ORDER\_TYPE\_SELL\_LIMIT, ORDER\_TYPE\_LIMIT\_STYPELL) In case of partial filling a market or limit order with remaining volume is not canceled but processed further.

For the activation of the ORDER\_TYPE\_BUY\_STOP\_LIMIT and ORDER\_TYPE\_SELL\_STOP\_LIMIT orders, a corresponding limit order ORDER\_TYPE\_BUY\_LIMIT / ORDER\_TYPE\_SELL\_LIMIT with the ORDER\_FILLING\_RETURN execution type is created.


OK, thank you very much


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
12 Apr 2021 at 05:20

**jewelnguyen :**

OK, thank you very much

Please report the results

![MQL as a Markup Tool for the Graphical Interface of MQL Programs. Part 2](https://c.mql5.com/2/38/MQL5-avatar-dialog_form.png)[MQL as a Markup Tool for the Graphical Interface of MQL Programs. Part 2](https://www.mql5.com/en/articles/7739)

This paper continues checking the new conception to describe the window interface of MQL programs, using the structures of MQL. Automatically creating GUI based on the MQL markup provides additional functionality for caching and dynamically generating the elements and controlling the styles and new schemes for processing the events. Attached is an enhanced version of the standard library of controls.

![Multicurrency monitoring of trading signals (Part 4): Enhancing functionality and improving the signal search system](https://c.mql5.com/2/38/Article_Logo__1.png)[Multicurrency monitoring of trading signals (Part 4): Enhancing functionality and improving the signal search system](https://www.mql5.com/en/articles/7678)

In this part, we expand the trading signal searching and editing system, as well as introduce the possibility to use custom indicators and add program localization. We have previously created a basic system for searching signals, but it was based on a small set of indicators and a simple set of search rules.

![Continuous Walk-Forward Optimization (Part 6): Auto optimizer's logical part and structure](https://c.mql5.com/2/38/MQL5-avatar-continuous_optimization__3.png)[Continuous Walk-Forward Optimization (Part 6): Auto optimizer's logical part and structure](https://www.mql5.com/en/articles/7718)

We have previously considered the creation of automatic walk-forward optimization. This time, we will proceed to the internal structure of the auto optimizer tool. The article will be useful for all those who wish to further work with the created project and to modify it, as well as for those who wish to understand the program logic. The current article contains UML diagrams which present the internal structure of the project and the relationships between objects. It also describes the process of optimization start, but it does not contain the description of the optimizer implementation process.

![MQL as a Markup Tool for the Graphical Interface of MQL Programs. Part 1](https://c.mql5.com/2/38/MQL5-avatar-dialog_form__1.png)[MQL as a Markup Tool for the Graphical Interface of MQL Programs. Part 1](https://www.mql5.com/en/articles/7734)

This paper proposes a new conception to describe the window interface of MQL programs, using the structures of MQL. Special classes transform the viewable MQL markup into the GUI elements and allow manage them, set up their properties, and process the events in a unified manner. It also provides some examples of using the markup for the dialogs and elements of a standard library.

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/7724&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070421833460684169)

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