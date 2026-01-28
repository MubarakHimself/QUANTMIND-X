---
title: Prices in DoEasy library (part 62): Updating tick series in real time, preparation for working with Depth of Market
url: https://www.mql5.com/en/articles/8988
categories: Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:50:47.780243
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=rvhrqarzmbshisihavnzfjozfishrjpi&ssn=1769251845574117591&ssn_dr=0&ssn_sr=0&fv_date=1769251845&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8988&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Prices%20in%20DoEasy%20library%20(part%2062)%3A%20Updating%20tick%20series%20in%20real%20time%2C%20preparation%20for%20working%20with%20Depth%20of%20Market%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17692518455085069&fz_uniq=5083153581109941764&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/8988#node01)
- [Improving library classes](https://www.mql5.com/en/articles/8988#node02)
- [Updating tick series](https://www.mql5.com/en/articles/8988#node03)
- [Improving the symbol class for working with the Depth of Market](https://www.mql5.com/en/articles/8988#node04)
- [Test](https://www.mql5.com/en/articles/8988#node05)
- [What's next?](https://www.mql5.com/en/articles/8988#node06)


### Concept

I have created the tick data collection of all symbols used in the program. The library is able to obtain the required amount of tick data for each of the symbols used by the program and stores all of them in the collection of tick data. The tick data collection allows finding any required tick object and receiving its data. We are able to sort out the lists for conducting statistical research. However, new ticks are not entered into the tick database when new ticks arrive for symbols. In this article, I am going to implement this feature.

Each new tick will increase the number of stored objects in the collection. To limit their number, as well as the number of used memory, let's introduce a constant allowing us to set the maximum possible number of ticks stored in the library database for one instrument. This will protect us from running out of memory. If many symbols are used in the program and the database already contains a sufficient number of ticks, the library automatically removes the necessary amount of the oldest ticks. Thus, we will always have the specified number of ticks for the instrument. The default number is 200,000. This number should be sufficient for conducting statistical research for about the last two days. In any case, the maximum number of ticks stored in the collection for a single instrument can always be changed if necessary.

Also, I will start preparations for working with the Depth of Market (DOM). I am going to introduce the ability to subscribe to the DOM broadcast in the symbol object class. In the next articles, I will start implementing the functionality for working with DOM.

### Improving library classes

As usual, let's start by adding new library text messages.

In \\MQL5\\Include\\DoEasy\ **Data.mqh**, add the indices of new messages:

```
   MSG_SYM_EVENT_SYMBOL_ADD,                          // Added symbol to Market Watch window
   MSG_SYM_EVENT_SYMBOL_DEL,                          // Symbol removed from Market Watch window
   MSG_SYM_EVENT_SYMBOL_SORT,                         // Changed location of symbols in Market Watch window
   MSG_SYM_SYMBOLS_MODE_CURRENT,                      // Work with current symbol only
   MSG_SYM_SYMBOLS_MODE_DEFINES,                      // Work with predefined symbol list
   MSG_SYM_SYMBOLS_MODE_MARKET_WATCH,                 // Work with Market Watch window symbols
   MSG_SYM_SYMBOLS_MODE_ALL,                          // Work with full list of all available symbols
   MSG_SYM_SYMBOLS_BOOK_ADD,                          // Subscribed to Depth of Market
   MSG_SYM_SYMBOLS_BOOK_DEL,                          // Unsubscribed from Depth of Market
   MSG_SYM_SYMBOLS_MODE_BOOK,                         // Subscription to Depth of Market

//--- CAccount
```

and text messages corresponding to newly added indices:

```
   {"В окно \"Обзор рынка\" добавлен символ","Added symbol to \"Market Watch\" window"},
   {"Из окна \"Обзор рынка\" удалён символ","Removed from \"Market Watch\" window"},
   {"Изменено расположение символов в окне \"Обзор рынка\"","Changed arrangement of symbols in \"Market Watch\" window"},
   {"Работа только с текущим символом","Work only with the current symbol"},
   {"Работа с предопределённым списком символов","Work with predefined list of symbols"},
   {"Работа с символами из окна \"Обзор рынка\"","Working with symbols from \"Market Watch\" window"},
   {"Работа с полным списком всех доступных символов","Work with full list of all available symbols"},
   {"Осуществлена подписка на стакан цен ","Subscribed to Depth of Market"},
   {"Осуществлена отписка от стакан цен ","Unsubscribed from Depth of Market"},
   {"Подписка на стакан цен","Subscription to Depth of Market"},

//--- CAccount
```

When a new tick arrives to the current symbol, we need to add it to the [MqlTick](https://www.mql5.com/en/docs/constants/structures/mqltick) structure. A new tick object is created based on the structure and added to the tick series list stored in the collection together with the lists of other symbols. However, we cannot obtain ticks for other symbols in the program's OnTick() handler as the handler is activated when a new tick arrives to the current symbol. Therefore, in order to obtain new ticks on other used symbols, we need to control them in the library timer using the previously created "New tick" class object. To do this, we need yet another library timer, in which ticks for all instruments except the current one are tracked for updating the lists of tick data of these instruments.

In \\MQL5\\Include\\DoEasy\ **Defines.mqh**, add the parameters of the tick data collection timer and the constant for specifying the maximum possible number of tick objects on a single symbol:

```
//--- Parameters of the timer of indicator data timeseries collection
#define COLLECTION_IND_TS_PAUSE        (64)                       // Pause of the timer of indicator data timeseries collection in milliseconds
#define COLLECTION_IND_TS_COUNTER_STEP (16)                       // Increment of indicator data timeseries timer counter
#define COLLECTION_IND_TS_COUNTER_ID   (7)                        // ID of indicator data timeseries timer counter
//--- Parameters of the tick series collection timer
#define COLLECTION_TICKS_PAUSE         (64)                       // Tick series collection timer pause in milliseconds
#define COLLECTION_TICKS_COUNTER_STEP  (16)                       // Tick series timer counter increment step
#define COLLECTION_TICKS_COUNTER_ID    (8)                        // Tick series timer counter ID
//--- Collection list IDs
#define COLLECTION_HISTORY_ID          (0x777A)                   // Historical collection list ID
#define COLLECTION_MARKET_ID           (0x777B)                   // Market collection list ID
#define COLLECTION_EVENTS_ID           (0x777C)                   // Event collection list ID
#define COLLECTION_ACCOUNT_ID          (0x777D)                   // Account collection list ID
#define COLLECTION_SYMBOLS_ID          (0x777E)                   // Symbol collection list ID
#define COLLECTION_SERIES_ID           (0x777F)                   // Timeseries collection list ID
#define COLLECTION_BUFFERS_ID          (0x7780)                   // Indicator buffer collection list ID
#define COLLECTION_INDICATORS_ID       (0x7781)                   // Indicator collection list ID
#define COLLECTION_INDICATORS_DATA_ID  (0x7782)                   // Indicator data collection list ID
#define COLLECTION_TICKSERIES_ID       (0x7783)                   // Tick series collection list ID
//--- Data parameters for file operations
#define DIRECTORY                      ("DoEasy\\")               // Library directory for storing object folders
#define RESOURCE_DIR                   ("DoEasy\\Resource\\")     // Library directory for storing resource folders
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
//--- Tick series parameters
#define TICKSERIES_DEFAULT_DAYS_COUNT  (1)                        // Required number of days for tick data in default series
#define TICKSERIES_MAX_DATA_TOTAL      (200000)                   // Maximum number of stored tick data of a single symbol
//+------------------------------------------------------------------+
```

To be able to understand whether we are subscribed to a DOM broadcast by a symbol, we need to add a parameter to the symbol properties indicating the subscription status. To achieve this, add yet another parameter to the symbol integer properties and increase the number of integer properties from 36 to **37**:

```
//+------------------------------------------------------------------+
//| Symbol integer properties                                        |
//+------------------------------------------------------------------+
enum ENUM_SYMBOL_PROP_INTEGER
  {
   SYMBOL_PROP_STATUS = 0,                                  // Symbol status
   SYMBOL_PROP_INDEX_MW,                                    // Symbol index in the Market Watch window
   SYMBOL_PROP_CUSTOM,                                      // Custom symbol flag
   SYMBOL_PROP_CHART_MODE,                                  // The price type used for generating bars – Bid or Last (from the ENUM_SYMBOL_CHART_MODE enumeration)
   SYMBOL_PROP_EXIST,                                       // Flag indicating that the symbol under this name exists
   SYMBOL_PROP_SELECT,                                      // The indication that the symbol is selected in Market Watch
   SYMBOL_PROP_VISIBLE,                                     // The indication that the symbol is displayed in Market Watch
   SYMBOL_PROP_SESSION_DEALS,                               // The number of deals in the current session
   SYMBOL_PROP_SESSION_BUY_ORDERS,                          // The total number of Buy orders at the moment
   SYMBOL_PROP_SESSION_SELL_ORDERS,                         // The total number of Sell orders at the moment
   SYMBOL_PROP_VOLUME,                                      // Last deal volume
   SYMBOL_PROP_VOLUMEHIGH,                                  // Maximum volume within a day
   SYMBOL_PROP_VOLUMELOW,                                   // Minimum volume within a day
   SYMBOL_PROP_TIME,                                        // Latest quote time
   SYMBOL_PROP_DIGITS,                                      // Number of decimal places
   SYMBOL_PROP_DIGITS_LOTS,                                 // Number of decimal places for a lot
   SYMBOL_PROP_SPREAD,                                      // Spread in points
   SYMBOL_PROP_SPREAD_FLOAT,                                // Floating spread flag
   SYMBOL_PROP_TICKS_BOOKDEPTH,                             // Maximum number of orders displayed in the Depth of Market
   SYMBOL_PROP_BOOKDEPTH_STATE,                             // Flag of subscription to DOM
   SYMBOL_PROP_TRADE_CALC_MODE,                             // Contract price calculation method (from the ENUM_SYMBOL_CALC_MODE enumeration)
   SYMBOL_PROP_TRADE_MODE,                                  // Order execution type (from the ENUM_SYMBOL_TRADE_MODE enumeration)
   SYMBOL_PROP_START_TIME,                                  // Symbol trading start date (usually used for futures)
   SYMBOL_PROP_EXPIRATION_TIME,                             // Symbol trading end date (usually used for futures)
   SYMBOL_PROP_TRADE_STOPS_LEVEL,                           // Minimum distance in points from the current close price for setting Stop orders
   SYMBOL_PROP_TRADE_FREEZE_LEVEL,                          // Freeze distance for trading operations (in points)
   SYMBOL_PROP_TRADE_EXEMODE,                               // Deal execution mode (from the ENUM_SYMBOL_TRADE_EXECUTION enumeration)
   SYMBOL_PROP_SWAP_MODE,                                   // Swap calculation model (from the ENUM_SYMBOL_SWAP_MODE enumeration)
   SYMBOL_PROP_SWAP_ROLLOVER3DAYS,                          // Triple-day swap (from the ENUM_DAY_OF_WEEK enumeration)
   SYMBOL_PROP_MARGIN_HEDGED_USE_LEG,                       // Calculating hedging margin using the larger leg (Buy or Sell)
   SYMBOL_PROP_EXPIRATION_MODE,                             // Flags of allowed order expiration modes
   SYMBOL_PROP_FILLING_MODE,                                // Flags of allowed order filling modes
   SYMBOL_PROP_ORDER_MODE,                                  // Flags of allowed order types
   SYMBOL_PROP_ORDER_GTC_MODE,                              // Expiration of Stop Loss and Take Profit orders if SYMBOL_EXPIRATION_MODE=SYMBOL_EXPIRATION_GTC (from the ENUM_SYMBOL_ORDER_GTC_MODE enumeration)
   SYMBOL_PROP_OPTION_MODE,                                 // Option type (from the ENUM_SYMBOL_OPTION_MODE enumeration)
   SYMBOL_PROP_OPTION_RIGHT,                                // Option right (Call/Put) (from the ENUM_SYMBOL_OPTION_RIGHT enumeration)
   //--- skip the property
   SYMBOL_PROP_BACKGROUND_COLOR                             // The color of the background used for the symbol in Market Watch
  };
#define SYMBOL_PROP_INTEGER_TOTAL    (37)                   // Total number of integer properties
#define SYMBOL_PROP_INTEGER_SKIP     (1)                    // Number of symbol integer properties not used in sorting
//+------------------------------------------------------------------+
```

Add sorting by a new integer property to the enumerations of possible criteria of sorting symbol objects:

```
//+------------------------------------------------------------------+
//| Possible symbol sorting criteria                                 |
//+------------------------------------------------------------------+
#define FIRST_SYM_DBL_PROP          (SYMBOL_PROP_INTEGER_TOTAL-SYMBOL_PROP_INTEGER_SKIP)
#define FIRST_SYM_STR_PROP          (SYMBOL_PROP_INTEGER_TOTAL-SYMBOL_PROP_INTEGER_SKIP+SYMBOL_PROP_DOUBLE_TOTAL-SYMBOL_PROP_DOUBLE_SKIP)
enum ENUM_SORT_SYMBOLS_MODE
  {
//--- Sort by integer properties
   SORT_BY_SYMBOL_STATUS = 0,                               // Sort by symbol status
   SORT_BY_SYMBOL_INDEX_MW,                                 // Sort by index in the Market Watch window
   SORT_BY_SYMBOL_CUSTOM,                                   // Sort by custom symbol property
   SORT_BY_SYMBOL_CHART_MODE,                               // Sort by price type for constructing bars – Bid or Last (from the ENUM_SYMBOL_CHART_MODE enumeration)
   SORT_BY_SYMBOL_EXIST,                                    // Sort by the flag that a symbol with such a name exists
   SORT_BY_SYMBOL_SELECT,                                   // Sort by the flag indicating that a symbol is selected in Market Watch
   SORT_BY_SYMBOL_VISIBLE,                                  // Sort by the flag indicating that a selected symbol is displayed in Market Watch
   SORT_BY_SYMBOL_SESSION_DEALS,                            // Sort by the number of deals in the current session
   SORT_BY_SYMBOL_SESSION_BUY_ORDERS,                       // Sort by the total number of current buy orders
   SORT_BY_SYMBOL_SESSION_SELL_ORDERS,                      // Sort by the total number of current sell orders
   SORT_BY_SYMBOL_VOLUME,                                   // Sort by last deal volume
   SORT_BY_SYMBOL_VOLUMEHIGH,                               // Sort by maximum volume for a day
   SORT_BY_SYMBOL_VOLUMELOW,                                // Sort by minimum volume for a day
   SORT_BY_SYMBOL_TIME,                                     // Sort by the last quote time
   SORT_BY_SYMBOL_DIGITS,                                   // Sort by a number of decimal places
   SORT_BY_SYMBOL_DIGITS_LOT,                               // Sort by a number of decimal places in a lot
   SORT_BY_SYMBOL_SPREAD,                                   // Sort by spread in points
   SORT_BY_SYMBOL_SPREAD_FLOAT,                             // Sort by floating spread
   SORT_BY_SYMBOL_TICKS_BOOKDEPTH,                          // Sort by a maximum number of requests displayed in the market depth
   SORT_BY_SYMBOL_BOOKDEPTH_STATE,                          // Sort by the DOM subscription flag
   SORT_BY_SYMBOL_TRADE_CALC_MODE,                          // Sort by contract price calculation method (from the ENUM_SYMBOL_CALC_MODE enumeration)
   SORT_BY_SYMBOL_TRADE_MODE,                               // Sort by order execution type (from the ENUM_SYMBOL_TRADE_MODE enumeration)
   SORT_BY_SYMBOL_START_TIME,                               // Sort by an instrument trading start date (usually used for futures)
   SORT_BY_SYMBOL_EXPIRATION_TIME,                          // Sort by an instrument trading end date (usually used for futures)
   SORT_BY_SYMBOL_TRADE_STOPS_LEVEL,                        // Sort by the minimum indent from the current close price (in points) for setting Stop orders
   SORT_BY_SYMBOL_TRADE_FREEZE_LEVEL,                       // Sort by trade operation freeze distance (in points)
   SORT_BY_SYMBOL_TRADE_EXEMODE,                            // Sort by trade execution mode (from the ENUM_SYMBOL_TRADE_EXECUTION enumeration)
   SORT_BY_SYMBOL_SWAP_MODE,                                // Sort by swap calculation model (from the ENUM_SYMBOL_SWAP_MODE enumeration)
   SORT_BY_SYMBOL_SWAP_ROLLOVER3DAYS,                       // Sort by week day for accruing a triple swap (from the ENUM_DAY_OF_WEEK enumeration)
   SORT_BY_SYMBOL_MARGIN_HEDGED_USE_LEG,                    // Sort by the calculation mode of a hedged margin using the larger leg (Buy or Sell)
   SORT_BY_SYMBOL_EXPIRATION_MODE,                          // Sort by flags of allowed order expiration modes
   SORT_BY_SYMBOL_FILLING_MODE,                             // Sort by flags of allowed order filling modes
   SORT_BY_SYMBOL_ORDER_MODE,                               // Sort by flags of allowed order types
   SORT_BY_SYMBOL_ORDER_GTC_MODE,                           // Sort by StopLoss and TakeProfit orders lifetime
   SORT_BY_SYMBOL_OPTION_MODE,                              // Sort by option type (from the ENUM_SYMBOL_OPTION_MODE enumeration)
   SORT_BY_SYMBOL_OPTION_RIGHT,                             // Sort by option right (Call/Put) (from the ENUM_SYMBOL_OPTION_RIGHT enumeration)
//--- Sort by real properties
```

### Updating tick series

Since ticks can come simultaneously in a bundle, we cannot add them one by one (tick by tick) to the list of tick series. In order to save all ticks received in one batch, we need to control the millisecond time of the last received tick and copy the ticks from this time to the very end of the history data. After copying all newly arrived ticks (this can be either one tick or several at once in one pack), we need to save the last tick time to start copying ticks from this time + 1 millisecond (in order not to copy the previous last tick again) to the very end of the history data — up to the current time. Thus, we will always be able to obtain all the necessary data that appeared with the arrival of a new tick on each new OnTick() activation. After copying, we need to remember the new time of the last tick for subsequent copying.

When creating the method for updating the tick series, it turned out that creating a new tick data object and adding it to the tick series list is identical to creating a new tick data object and adding it to the list in the already developed method of creating a tick series. Therefore, this code block was moved to the new method returning the pointer to a newly created object added to the list or NULL. The changed method of creating the list and the new method for updating the list will be considered below.

In the private section of the class in \\MQL5\\Include\\DoEasy\\Objects\\Ticks\ **TickSeries.mqh**, declare the class member variable for storing the millisecond time of the last tick and the method for creating a new tick object and adding it to the tick series list:

```
//+------------------------------------------------------------------+
//| "Tick data series" class                                         |
//+------------------------------------------------------------------+
class CTickSeries : public CBaseObj
  {
private:
   string            m_symbol;                                          // Symbol
   ulong             m_last_time;                                       // Last tick time
   uint              m_amount;                                          // Amount of applied tick series data
   uint              m_required;                                        // Required number of days for tick series data
   CArrayObj         m_list_ticks;                                      // List of tick data
   CNewTickObj       m_new_tick_obj;                                    // "New tick" object
//--- Create a new tick data object
   CDataTick        *CreateNewTickObj(const MqlTick &tick);

public:
```

In the public section of the class, declare the method returning the pointer to the last object of tick data in the list:

```
//--- Return the object of tick data by (1) index in the list, (2) time,
//--- (3) time in milliseconds, (4) the last one in the list and (5) the list size
   CDataTick        *GetTickByListIndex(const uint index);
   CDataTick        *GetTick(const datetime time);
   CDataTick        *GetTick(const ulong time_msc);
   CDataTick        *GetLastTick(void);
   int               DataTotal(void)                              const { return this.m_list_ticks.Total();       }

//--- The comparison method for searching identical tick series objects by a symbol
```

Since we will need the "New tick" object, we should specify a symbol it should work with so that it operates correctly.

Let's do this in the class constructor:

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CTickSeries::CTickSeries(const string symbol,const uint required=0) : m_symbol(symbol),m_last_time(0)
  {
   this.m_list_ticks.Clear();
   this.m_list_ticks.Sort(SORT_BY_TICK_TIME_MSC);
   this.SetRequiredUsedDays(required);
   this.m_new_tick_obj.SetSymbol(this.m_symbol);
   this.m_new_tick_obj.Refresh();
  }
//+------------------------------------------------------------------+
```

Immediately after setting the symbol, update the data in the "New tick" object once to remember the last tick time in the object.

**The method creating a new tick data object and placing it to the list:**

```
//+------------------------------------------------------------------+
//| Create a new tick data object                                    |
//+------------------------------------------------------------------+
CDataTick *CTickSeries::CreateNewTickObj(const MqlTick &tick)
  {
   //--- create a new object of tick data out of the MqlTick structure passed to the method
   int err=ERR_SUCCESS;
   ::ResetLastError();
   //--- If failed to create an object, inform of that and return NULL
   CDataTick* tick_obj=new CDataTick(this.m_symbol,tick);
   if(tick_obj==NULL)
     {
      ::Print
        (
         DFUN,CMessage::Text(MSG_TICKSERIES_FAILED_CREATE_TICK_DATA_OBJ)," ",this.Header()," ",::TimeMSCtoString(tick.time_msc),". ",
         CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(::GetLastError())
        );
      return NULL;
     }
   //--- If failed to add a new tick data object to the list,
   //--- display the appropriate message with the error description in the journal,
   //--- remove the newly created object and return NULL
   this.m_list_ticks.Sort();
   if(!this.m_list_ticks.InsertSort(tick_obj))
     {
      err=::GetLastError();
      ::Print(DFUN,CMessage::Text(MSG_TICKSERIES_FAILED_ADD_TO_LIST)," ",tick_obj.Header()," ",
                   CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(err),CMessage::Retcode(err));
      delete tick_obj;
      return NULL;
     }
   //--- Return the pointer to the tick data object that was created and added to the list
   return tick_obj;
  }
//+------------------------------------------------------------------+
```

In the method, all its logic is described in the comments. This code block was moved to the new method from the method of creating the tick series list I made [in the previous article](https://www.mql5.com/en/articles/8952). Now the method receives the tick structure used to create a new tick data object. Upon its creation and adding to the list, the pointer to the object is returned (or NULL if failed to create the object or add it to the list).

**The method creating the tick data series list:**

```
//+------------------------------------------------------------------+
//| Create the series list of tick data                              |
//+------------------------------------------------------------------+
int CTickSeries::Create(const uint required=0)
  {
//--- If the tick series is not used, inform of that and exit
   if(!this.m_available)
     {
      ::Print(DFUN,this.m_symbol,": ",CMessage::Text(MSG_TICKSERIES_TEXT_IS_NOT_USE));
      return false;
     }
//--- Declare the ticks[] array we are to receive historical data to,
//--- clear the list of tick data objects and set the flag of sorting by time in milliseconds
   MqlTick ticks_array[];
   this.m_list_ticks.Clear();
   this.m_list_ticks.Sort(SORT_BY_TICK_TIME_MSC);
   this.m_last_time=0;
   ::ResetLastError();
   int err=ERR_SUCCESS;
//--- Calculate the day start time in milliseconds the ticks should be copied from
   MqlDateTime date_str={0};
   datetime date=::iTime(m_symbol,PERIOD_D1,this.m_required);
   ::TimeToStruct(date,date_str);
   date_str.hour=date_str.min=date_str.sec=0;
   date=::StructToTime(date_str);
   long date_from=(long)date*1000;
   if(date_from<1) date_from=1;
//--- Get historical data of the MqlTick structure to the tick[] array
//--- from the calculated date to the current time and save the obtained number in m_amount.
//--- If failed to get data, display the appropriate message and return zero
   this.m_amount=::CopyTicksRange(m_symbol,ticks_array,COPY_TICKS_ALL,date_from);
   if(this.m_amount<1)
     {
      err=::GetLastError();
      ::Print(DFUN,CMessage::Text(MSG_TICKSERIES_ERR_GET_TICK_DATA),": ",CMessage::Text(err),CMessage::Retcode(err));
      return 0;
     }
//--- Historical data is received in the rates[] array
//--- In the ticks[] array loop
   for(int i=0; i<(int)this.m_amount; i++)
     {
      //--- Create the tick object and add it to the list
      CDataTick *tick_obj=this.CreateNewTickObj(ticks_array[i]);
      if(tick_obj==NULL)
         continue;
      //--- If the tick time exceeds the previous one, write the new tick time to m_last_time as the starting one
      //--- to copy the newly arrived ticks in the tick series update method
      if(this.m_last_time<(ulong)tick_obj.TimeMSC())
         this.m_last_time=tick_obj.TimeMSC();
     }
//--- Return the size of the created tick object list
   return this.m_list_ticks.Total();
  }
//+------------------------------------------------------------------+
```

[I described the method in the article devoted to creating tick series](https://www.mql5.com/en/articles/8912#node03). Here, I have changed it so that a new tick data object is created and added using the new method considered above. After the object has been successfully added to the list, save the last tick time for its subsequent use in the tick series update method.

**The method for updating the tick series:**

```
//+------------------------------------------------------------------+
//| Update the tick series list                                      |
//+------------------------------------------------------------------+
void CTickSeries::Refresh(void)
  {
   MqlTick ticks_array[];
   if(IsNewTick())
     {
      //--- Copy ticks from m_last_time time+1 ms to the end of history
      int err=ERR_SUCCESS;
      int total=::CopyTicksRange(this.Symbol(),ticks_array,COPY_TICKS_ALL,this.m_last_time+1,0);
      //--- If the ticks have been copied, create new tick data objects and add them to the list in the loop by their number
      if(total>0)
        {
         for(int i=0;i<total;i++)
           {
            //--- Create the tick object and add it to the list
            CDataTick *tick_obj=this.CreateNewTickObj(ticks_array[i]);
            if(tick_obj==NULL)
               break;
            //--- Write the last tick time for subsequent copying of newly arrived ticks
            long end_time=ticks_array[::ArraySize(ticks_array)-1].time_msc;
            if(this.Symbol()=="AUDUSD")
               Comment(DFUN,this.Symbol(),", copied=",total,", m_last_time=",TimeMSCtoString(m_last_time),", end_time=",TimeMSCtoString(end_time),", total=",DataTotal());
            this.m_last_time=end_time;
           }
         //--- If the number of ticks in the list exceeds the default maximum number,
         //--- remove the calculated number of tick objects from the end of the list
         if(this.DataTotal()>TICKSERIES_MAX_DATA_TOTAL)
           {
            int total_del=m_list_ticks.Total()-TICKSERIES_MAX_DATA_TOTAL;
            for(int j=0;j<total_del;j++)
               this.m_list_ticks.Delete(j);
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

Here all is simple. We have the last tick time fixed during the previous OnTick() activation and stored in the **m\_last\_time** variable. Now, in order to start copying new ticks, we need to add one millisecond to this time since the [CopyTicksRange()](https://www.mql5.com/en/docs/series/copyticksrange) function copies based on the specified time, **including** this time. To avoid inserting the already copied previous tick object to copied ticks during a new OnTick() activation, I start copying not from the time fixed at the previous tick but from the time with the difference of one millisecond. If after copying new ticks and adding them to the list, their total number exceeds the maximum value set for them, calculate the number of unnecessary objects in the list and remove them from the list — these are the oldest tick objects in the list.

The method receives the code blocks displaying data by the number of copied ticks, past and current time and the total number of tick data in the tick series list to check the correctness of their copying and only for AUDUSD.

In the next articles, we will remove these test strings. Now these strings are able to show us that ticks on a "non-native" symbol are copied in the library timer, while tick data objects are added to the tick series list.

**The method returning the most recent tick data object from the list:**

```
//+------------------------------------------------------------------+
//| Return the most recent tick data object from the list            |
//+------------------------------------------------------------------+
CDataTick *CTickSeries::GetLastTick(void)
  {
   return this.m_list_ticks.At(this.m_list_ticks.Total()-1);
  }
//+------------------------------------------------------------------+
```

The method returns the pointer to the most recent object in the list.

Now let's slightly improve the tick series collection class in \\MQL5\\Include\\DoEasy\\Collections\ **TickSeriesCollection.mqh**.

Since we will need to separately update tick series of the current symbol and the remaining ones, the series on the current symbol is updated in the library's OnTick() handler, while the remaining tick data of other symbols is updated in the library timer, create the method updating all tick series of the collection except for the current symbol. Declare the method in the public section of the class:

```
//--- Update (1) a tick series of a specified symbol, (2) all symbols and (3) all symbols except the current one
   void                    Refresh(const string symbol);
   void                    Refresh(void);
   void                    RefreshExpectCurrent(void);

//--- Display (1) the complete and (2) short collection description in the journal
```

and write its implementation beyond the class body:

```
//+------------------------------------------------------------------+
//| Update tick series of all symbols except the current one         |
//+------------------------------------------------------------------+
void CTickSeriesCollection::RefreshExpectCurrent(void)
  {
   for(int i=0;i<this.m_list.Total();i++)
     {
      CTickSeries *tickseries=this.m_list.At(i);
      if(tickseries==NULL || tickseries.Symbol()==::Symbol())
         continue;
      tickseries.Refresh();
     }
  }
//+------------------------------------------------------------------+
```

In the loop by the total number of tick series in the collection, get the next tick series object from the list.If its symbol is equal to a symbol of a chart the program is launched on, the series is skipped. Update all other tick series.

Add three methods to the **CEngine** library main object in \\MQL5\\Include\\DoEasy\ **Engine.mqh** — for updating the tick series of the specified symbol, for updating all tick series and for updating all tick series of the collection except for the current symbol series:

```
//--- Return (1) the tick series collection, (2) the list of tick series from the tick series collection
   CTickSeriesCollection *GetTickSeriesCollection(void)                                { return &this.m_tick_series;                         }
   CArrayObj           *GetListTickSeries(void)                                        { return this.m_tick_series.GetList();                }

//--- Update (1) a tick series of a specified symbol, (2) all symbols and (3) all symbols except the current one
   void                 TickSeriesRefresh(const string symbol)                         { this.m_tick_series.Refresh(symbol);                 }
   void                 TickSeriesRefreshAll(void)                                     { this.m_tick_series.Refresh();                       }
   void                 TickSeriesRefreshAllExceptCurrent(void)                        { this.m_tick_series.RefreshExpectCurrent();          }

//--- Return (1) the buffer collection and (2) the buffer list from the collection
```

The methods simply call the same-name methods of the tick series collection class.

**In the class constructor, implement creating the tick data collection timer:**

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

   this.m_list_counters.Sort();
   this.m_list_counters.Clear();
   this.CreateCounter(COLLECTION_ORD_COUNTER_ID,COLLECTION_ORD_COUNTER_STEP,COLLECTION_ORD_PAUSE);
   this.CreateCounter(COLLECTION_ACC_COUNTER_ID,COLLECTION_ACC_COUNTER_STEP,COLLECTION_ACC_PAUSE);
   this.CreateCounter(COLLECTION_SYM_COUNTER_ID1,COLLECTION_SYM_COUNTER_STEP1,COLLECTION_SYM_PAUSE1);
   this.CreateCounter(COLLECTION_SYM_COUNTER_ID2,COLLECTION_SYM_COUNTER_STEP2,COLLECTION_SYM_PAUSE2);
   this.CreateCounter(COLLECTION_REQ_COUNTER_ID,COLLECTION_REQ_COUNTER_STEP,COLLECTION_REQ_PAUSE);
   this.CreateCounter(COLLECTION_TS_COUNTER_ID,COLLECTION_TS_COUNTER_STEP,COLLECTION_TS_PAUSE);
   this.CreateCounter(COLLECTION_IND_TS_COUNTER_ID,COLLECTION_IND_TS_COUNTER_STEP,COLLECTION_IND_TS_PAUSE);
   this.CreateCounter(COLLECTION_TICKS_COUNTER_ID,COLLECTION_TICKS_COUNTER_STEP,COLLECTION_TICKS_PAUSE);

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

Now that the collection timer of tick series is created, we need to implement the block of working with the timer in the class timer:

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
         //--- If the pause is over, work with the timeseries list (update all except the current one)
         if(cnt6.IsTimeDone())
            this.SeriesRefreshAllExceptCurrent(data_calculate);
        }

   //--- Timer of timeseries collection of indicator buffer data
      index=this.CounterIndex(COLLECTION_IND_TS_COUNTER_ID);
      CTimerCounter* cnt7=this.m_list_counters.At(index);
      if(cnt7!=NULL)
        {
         //--- If the pause is over, work with the timeseries list of indicator data (update all except for the current one)
         if(cnt7.IsTimeDone())
            this.IndicatorSeriesRefreshAll();
        }
   //--- Tick series collection timer
      index=this.CounterIndex(COLLECTION_TICKS_COUNTER_ID);
      CTimerCounter* cnt8=this.m_list_counters.At(index);
      if(cnt8!=NULL)
        {
         //--- If the pause is over, work with the tick series list (update all except the current one)
         if(cnt8.IsTimeDone())
            this.TickSeriesRefreshAllExceptCurrent();
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
      //--- work with the timeseries list of indicator buffers by tick
      this.IndicatorSeriesRefreshAll();
      //--- work with the list of tick series by tick
      this.TickSeriesRefreshAll();
     }
  }
//+------------------------------------------------------------------+
```

Now all tick series of all "non-native" symbols is updated in the library timer.

To update the tick series of the current symbol, add calling the method of updating the tick series of the specified symbol to the OnTick() class handler:

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
   this.TickSeriesRefresh(NULL);
//--- end
  }
//+------------------------------------------------------------------+
```

Now tick series of the current symbol is updated upon arrival of the new tick, while the remaining tick series of other symbols are updated in the library timer.

### Improving the symbol class for working with the Depth of Market

In the next article, I will start implementing the library functionality for working with DOM.

To get the [BookEvent](https://www.mql5.com/en/docs/runtime/event_fire#bookevent) events for any symbol, simply subscribe to receive them for this symbol using the [MarketBookAdd()](https://www.mql5.com/en/docs/marketinformation/marketbookadd) function. To cancel subscription for receiving BookEvent for a certain symbol, call the [MarketBookRelease()](https://www.mql5.com/en/docs/marketinformation/marketbookrelease) function.

Each DOM connection should correspond to its disconnection. This can easily be done in the class — enable DOM in the constructor and disable it in the destructor. A separate symbol object is created for each symbol. For each of the symbol objects, we can clearly find out when DOM was connected and when it should be disabled.

Open the symbol object class in \\MQL5\\Include\\DoEasy\\Objects\\Symbols\ **Symbol.mqh** and add the necessary changes.

In the private class section, declare the variable for storing the flag of subscribing to DOM, while in the public section, declare the class destructor:

```
//+------------------------------------------------------------------+
//| Abstract symbol class                                            |
//+------------------------------------------------------------------+
class CSymbol : public CBaseObjExt
  {
private:
   struct MqlMarginRate
     {
      double         Initial;                                  // initial margin rate
      double         Maintenance;                              // maintenance margin rate
     };
   struct MqlMarginRateMode
     {
      MqlMarginRate  Long;                                     // MarginRate of long positions
      MqlMarginRate  Short;                                    // MarginRate of short positions
      MqlMarginRate  BuyStop;                                  // MarginRate of BuyStop orders
      MqlMarginRate  BuyLimit;                                 // MarginRate of BuyLimit orders
      MqlMarginRate  BuyStopLimit;                             // MarginRate of BuyStopLimit orders
      MqlMarginRate  SellStop;                                 // MarginRate of SellStop orders
      MqlMarginRate  SellLimit;                                // MarginRate of SellLimit orders
      MqlMarginRate  SellStopLimit;                            // MarginRate of SellStopLimit orders
     };
   MqlMarginRateMode m_margin_rate;                            // Margin ratio structure
   MqlBookInfo       m_book_info_array[];                      // Array of the market depth data structures

   long              m_long_prop[SYMBOL_PROP_INTEGER_TOTAL];   // Integer properties
   double            m_double_prop[SYMBOL_PROP_DOUBLE_TOTAL];  // Real properties
   string            m_string_prop[SYMBOL_PROP_STRING_TOTAL];  // String properties
   bool              m_is_change_trade_mode;                   // Flag of changing trading mode for a symbol
   bool              m_book_subscribed;                        // Flag of subscribing to the Depth of Market by symbol
   CTradeObj         m_trade;                                  // Trading class object
//--- Return the index of the array the symbol's (1) double and (2) string properties are located at
   int               IndexProp(ENUM_SYMBOL_PROP_DOUBLE property)  const { return(int)property-SYMBOL_PROP_INTEGER_TOTAL;                                    }
   int               IndexProp(ENUM_SYMBOL_PROP_STRING property)  const { return(int)property-SYMBOL_PROP_INTEGER_TOTAL-SYMBOL_PROP_DOUBLE_TOTAL;           }
//--- (1) Fill in all the "margin ratio" symbol properties, (2) initialize the ratios
-+/   bool              MarginRates(void);
   void              InitMarginRates(void);
//--- Reset all symbol object data
   void              Reset(void);
//--- Return the current day of the week
   ENUM_DAY_OF_WEEK  CurrentDayOfWeek(void)              const;

public:
//--- Default constructor
                     CSymbol(void){;}
//--- Destructor
                    ~CSymbol(void);
protected:
//--- Protected parametric constructor
                     CSymbol(ENUM_SYMBOL_STATUS symbol_status,const string name,const int index);
```

In the methods of the simplified access to symbol properties of the public class section, add the new method returning the status of subscription to DOM:

```
public:
//+------------------------------------------------------------------+
//| Methods of a simplified access to the order object properties    |
//+------------------------------------------------------------------+
//--- Integer properties
   long              Status(void)                                 const { return this.GetProperty(SYMBOL_PROP_STATUS);                                      }
   int               IndexInMarketWatch(void)                     const { return (int)this.GetProperty(SYMBOL_PROP_INDEX_MW);                               }
   bool              IsCustom(void)                               const { return (bool)this.GetProperty(SYMBOL_PROP_CUSTOM);                                }
   color             ColorBackground(void)                        const { return (color)this.GetProperty(SYMBOL_PROP_BACKGROUND_COLOR);                     }
   ENUM_SYMBOL_CHART_MODE ChartMode(void)                         const { return (ENUM_SYMBOL_CHART_MODE)this.GetProperty(SYMBOL_PROP_CHART_MODE);          }
   bool              IsExist(void)                                const { return (bool)this.GetProperty(SYMBOL_PROP_EXIST);                                 }
   bool              IsExist(const string name)                   const { return this.SymbolExists(name);                                                   }
   bool              IsSelect(void)                               const { return (bool)this.GetProperty(SYMBOL_PROP_SELECT);                                }
   bool              IsVisible(void)                              const { return (bool)this.GetProperty(SYMBOL_PROP_VISIBLE);                               }
   long              SessionDeals(void)                           const { return this.GetProperty(SYMBOL_PROP_SESSION_DEALS);                               }
   long              SessionBuyOrders(void)                       const { return this.GetProperty(SYMBOL_PROP_SESSION_BUY_ORDERS);                          }
   long              SessionSellOrders(void)                      const { return this.GetProperty(SYMBOL_PROP_SESSION_SELL_ORDERS);                         }
   long              Volume(void)                                 const { return this.GetProperty(SYMBOL_PROP_VOLUME);                                      }
   long              VolumeHigh(void)                             const { return this.GetProperty(SYMBOL_PROP_VOLUMEHIGH);                                  }
   long              VolumeLow(void)                              const { return this.GetProperty(SYMBOL_PROP_VOLUMELOW);                                   }
   long              Time(void)                                   const { return (datetime)this.GetProperty(SYMBOL_PROP_TIME);                              }
   int               Digits(void)                                 const { return (int)this.GetProperty(SYMBOL_PROP_DIGITS);                                 }
   int               DigitsLot(void)                              const { return (int)this.GetProperty(SYMBOL_PROP_DIGITS_LOTS);                            }
   int               Spread(void)                                 const { return (int)this.GetProperty(SYMBOL_PROP_SPREAD);                                 }
   bool              IsSpreadFloat(void)                          const { return (bool)this.GetProperty(SYMBOL_PROP_SPREAD_FLOAT);                          }
   int               TicksBookdepth(void)                         const { return (int)this.GetProperty(SYMBOL_PROP_TICKS_BOOKDEPTH);                        }
   bool              BookdepthSubscription(void)                  const { return (bool)this.GetProperty(SYMBOL_PROP_BOOKDEPTH_STATE);                       }
   ENUM_SYMBOL_CALC_MODE TradeCalcMode(void)                      const { return (ENUM_SYMBOL_CALC_MODE)this.GetProperty(SYMBOL_PROP_TRADE_CALC_MODE);      }
   ENUM_SYMBOL_TRADE_MODE TradeMode(void)                         const { return (ENUM_SYMBOL_TRADE_MODE)this.GetProperty(SYMBOL_PROP_TRADE_MODE);          }
   datetime          StartTime(void)                              const { return (datetime)this.GetProperty(SYMBOL_PROP_START_TIME);                        }
   datetime          ExpirationTime(void)                         const { return (datetime)this.GetProperty(SYMBOL_PROP_EXPIRATION_TIME);                   }
   int               TradeStopLevel(void)                         const { return (int)this.GetProperty(SYMBOL_PROP_TRADE_STOPS_LEVEL);                      }
   int               TradeFreezeLevel(void)                       const { return (int)this.GetProperty(SYMBOL_PROP_TRADE_FREEZE_LEVEL);                     }
   ENUM_SYMBOL_TRADE_EXECUTION TradeExecutionMode(void)           const { return (ENUM_SYMBOL_TRADE_EXECUTION)this.GetProperty(SYMBOL_PROP_TRADE_EXEMODE);  }
   ENUM_SYMBOL_SWAP_MODE SwapMode(void)                           const { return (ENUM_SYMBOL_SWAP_MODE)this.GetProperty(SYMBOL_PROP_SWAP_MODE);            }
   ENUM_DAY_OF_WEEK  SwapRollover3Days(void)                      const { return (ENUM_DAY_OF_WEEK)this.GetProperty(SYMBOL_PROP_SWAP_ROLLOVER3DAYS);        }
   bool              IsMarginHedgedUseLeg(void)                   const { return (bool)this.GetProperty(SYMBOL_PROP_MARGIN_HEDGED_USE_LEG);                 }
   int               ExpirationModeFlags(void)                    const { return (int)this.GetProperty(SYMBOL_PROP_EXPIRATION_MODE);                        }
   int               FillingModeFlags(void)                       const { return (int)this.GetProperty(SYMBOL_PROP_FILLING_MODE);                           }
   int               OrderModeFlags(void)                         const { return (int)this.GetProperty(SYMBOL_PROP_ORDER_MODE);                             }
   ENUM_SYMBOL_ORDER_GTC_MODE OrderModeGTC(void)                  const { return (ENUM_SYMBOL_ORDER_GTC_MODE)this.GetProperty(SYMBOL_PROP_ORDER_GTC_MODE);  }
   ENUM_SYMBOL_OPTION_MODE OptionMode(void)                       const { return (ENUM_SYMBOL_OPTION_MODE)this.GetProperty(SYMBOL_PROP_OPTION_MODE);        }
   ENUM_SYMBOL_OPTION_RIGHT OptionRight(void)                     const { return (ENUM_SYMBOL_OPTION_RIGHT)this.GetProperty(SYMBOL_PROP_OPTION_RIGHT);      }
//--- Real properties
```

In the class constructor, initialize the subscription flag to falseand add the flag value to the symbol property:

```
//+------------------------------------------------------------------+
//| Closed parametric constructor                                    |
//+------------------------------------------------------------------+
CSymbol::CSymbol(ENUM_SYMBOL_STATUS symbol_status,const string name,const int index)
  {
   this.m_name=name;
   this.m_book_subscribed=false;
   this.m_type=COLLECTION_SYMBOLS_ID;
   if(!this.Exist())
     {
      ::Print(DFUN_ERR_LINE,"\"",this.m_name,"\"",": ",CMessage::Text(MSG_LIB_SYS_NOT_SYMBOL_ON_SERVER));
      this.m_global_error=ERR_MARKET_UNKNOWN_SYMBOL;
     }
   bool select=::SymbolInfoInteger(this.m_name,SYMBOL_SELECT);
   ::ResetLastError();
   if(!select)
     {
      if(!this.SetToMarketWatch())
        {
         this.m_global_error=::GetLastError();
         ::Print(DFUN_ERR_LINE,"\"",this.m_name,"\": ",CMessage::Text(MSG_LIB_SYS_FAILED_PUT_SYMBOL),this.m_global_error);
        }
     }
   ::ResetLastError();
   if(!::SymbolInfoTick(this.m_name,this.m_tick))
     {
      this.m_global_error=::GetLastError();
      ::Print(DFUN_ERR_LINE,"\"",this.m_name,"\": ",CMessage::Text(MSG_LIB_SYS_NOT_GET_PRICE),this.m_global_error);
     }
//--- Initializing base object data arrays
   this.SetControlDataArraySizeLong(SYMBOL_PROP_INTEGER_TOTAL);
   this.SetControlDataArraySizeDouble(SYMBOL_PROP_DOUBLE_TOTAL);
   this.ResetChangesParams();
   this.ResetControlsParams();

//--- Initialize symbol data
   this.Reset();
   this.InitMarginRates();
#ifdef __MQL5__
   ::ResetLastError();
   if(!this.MarginRates())
     {
      this.m_global_error=::GetLastError();
      ::Print(DFUN_ERR_LINE,this.Name(),": ",CMessage::Text(MSG_LIB_SYS_NOT_GET_MARGIN_RATES),this.m_global_error);
      return;
     }
#endif

//--- Save integer properties
   this.m_long_prop[SYMBOL_PROP_STATUS]                                             = symbol_status;
   this.m_long_prop[SYMBOL_PROP_INDEX_MW]                                           = index;
   this.m_long_prop[SYMBOL_PROP_VOLUME]                                             = (long)this.m_tick.volume;
   this.m_long_prop[SYMBOL_PROP_SELECT]                                             = ::SymbolInfoInteger(this.m_name,SYMBOL_SELECT);
   this.m_long_prop[SYMBOL_PROP_VISIBLE]                                            = ::SymbolInfoInteger(this.m_name,SYMBOL_VISIBLE);
   this.m_long_prop[SYMBOL_PROP_SESSION_DEALS]                                      = ::SymbolInfoInteger(this.m_name,SYMBOL_SESSION_DEALS);
   this.m_long_prop[SYMBOL_PROP_SESSION_BUY_ORDERS]                                 = ::SymbolInfoInteger(this.m_name,SYMBOL_SESSION_BUY_ORDERS);
   this.m_long_prop[SYMBOL_PROP_SESSION_SELL_ORDERS]                                = ::SymbolInfoInteger(this.m_name,SYMBOL_SESSION_SELL_ORDERS);
   this.m_long_prop[SYMBOL_PROP_VOLUMEHIGH]                                         = ::SymbolInfoInteger(this.m_name,SYMBOL_VOLUMEHIGH);
   this.m_long_prop[SYMBOL_PROP_VOLUMELOW]                                          = ::SymbolInfoInteger(this.m_name,SYMBOL_VOLUMELOW);
   this.m_long_prop[SYMBOL_PROP_DIGITS]                                             = ::SymbolInfoInteger(this.m_name,SYMBOL_DIGITS);
   this.m_long_prop[SYMBOL_PROP_SPREAD]                                             = ::SymbolInfoInteger(this.m_name,SYMBOL_SPREAD);
   this.m_long_prop[SYMBOL_PROP_SPREAD_FLOAT]                                       = ::SymbolInfoInteger(this.m_name,SYMBOL_SPREAD_FLOAT);
   this.m_long_prop[SYMBOL_PROP_TICKS_BOOKDEPTH]                                    = ::SymbolInfoInteger(this.m_name,SYMBOL_TICKS_BOOKDEPTH);
   this.m_long_prop[SYMBOL_PROP_TRADE_MODE]                                         = ::SymbolInfoInteger(this.m_name,SYMBOL_TRADE_MODE);
   this.m_long_prop[SYMBOL_PROP_START_TIME]                                         = ::SymbolInfoInteger(this.m_name,SYMBOL_START_TIME);
   this.m_long_prop[SYMBOL_PROP_EXPIRATION_TIME]                                    = ::SymbolInfoInteger(this.m_name,SYMBOL_EXPIRATION_TIME);
   this.m_long_prop[SYMBOL_PROP_TRADE_STOPS_LEVEL]                                  = ::SymbolInfoInteger(this.m_name,SYMBOL_TRADE_STOPS_LEVEL);
   this.m_long_prop[SYMBOL_PROP_TRADE_FREEZE_LEVEL]                                 = ::SymbolInfoInteger(this.m_name,SYMBOL_TRADE_FREEZE_LEVEL);
   this.m_long_prop[SYMBOL_PROP_TRADE_EXEMODE]                                      = ::SymbolInfoInteger(this.m_name,SYMBOL_TRADE_EXEMODE);
   this.m_long_prop[SYMBOL_PROP_SWAP_ROLLOVER3DAYS]                                 = ::SymbolInfoInteger(this.m_name,SYMBOL_SWAP_ROLLOVER3DAYS);
   this.m_long_prop[SYMBOL_PROP_TIME]                                               = this.TickTime();
   this.m_long_prop[SYMBOL_PROP_EXIST]                                              = this.SymbolExists();
   this.m_long_prop[SYMBOL_PROP_CUSTOM]                                             = this.SymbolCustom();
   this.m_long_prop[SYMBOL_PROP_MARGIN_HEDGED_USE_LEG]                              = this.SymbolMarginHedgedUseLEG();
   this.m_long_prop[SYMBOL_PROP_ORDER_MODE]                                         = this.SymbolOrderMode();
   this.m_long_prop[SYMBOL_PROP_FILLING_MODE]                                       = this.SymbolOrderFillingMode();
   this.m_long_prop[SYMBOL_PROP_EXPIRATION_MODE]                                    = this.SymbolExpirationMode();
   this.m_long_prop[SYMBOL_PROP_ORDER_GTC_MODE]                                     = this.SymbolOrderGTCMode();
   this.m_long_prop[SYMBOL_PROP_OPTION_MODE]                                        = this.SymbolOptionMode();
   this.m_long_prop[SYMBOL_PROP_OPTION_RIGHT]                                       = this.SymbolOptionRight();
   this.m_long_prop[SYMBOL_PROP_BACKGROUND_COLOR]                                   = this.SymbolBackgroundColor();
   this.m_long_prop[SYMBOL_PROP_CHART_MODE]                                         = this.SymbolChartMode();
   this.m_long_prop[SYMBOL_PROP_TRADE_CALC_MODE]                                    = this.SymbolCalcMode();
   this.m_long_prop[SYMBOL_PROP_SWAP_MODE]                                          = this.SymbolSwapMode();
   this.m_long_prop[SYMBOL_PROP_BOOKDEPTH_STATE]                                    = this.m_book_subscribed;
//--- Save real properties
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASKHIGH)]                          = ::SymbolInfoDouble(this.m_name,SYMBOL_ASKHIGH);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASKLOW)]                           = ::SymbolInfoDouble(this.m_name,SYMBOL_ASKLOW);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LASTHIGH)]                         = ::SymbolInfoDouble(this.m_name,SYMBOL_LASTHIGH);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LASTLOW)]                          = ::SymbolInfoDouble(this.m_name,SYMBOL_LASTLOW);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_POINT)]                            = ::SymbolInfoDouble(this.m_name,SYMBOL_POINT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE)]                 = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_TICK_VALUE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE_PROFIT)]          = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_TICK_VALUE_PROFIT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE_LOSS)]            = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_TICK_VALUE_LOSS);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_SIZE)]                  = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_TICK_SIZE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_CONTRACT_SIZE)]              = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_CONTRACT_SIZE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_MIN)]                       = ::SymbolInfoDouble(this.m_name,SYMBOL_VOLUME_MIN);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_MAX)]                       = ::SymbolInfoDouble(this.m_name,SYMBOL_VOLUME_MAX);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_STEP)]                      = ::SymbolInfoDouble(this.m_name,SYMBOL_VOLUME_STEP);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_LIMIT)]                     = ::SymbolInfoDouble(this.m_name,SYMBOL_VOLUME_LIMIT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SWAP_LONG)]                        = ::SymbolInfoDouble(this.m_name,SYMBOL_SWAP_LONG);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SWAP_SHORT)]                       = ::SymbolInfoDouble(this.m_name,SYMBOL_SWAP_SHORT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_INITIAL)]                   = ::SymbolInfoDouble(this.m_name,SYMBOL_MARGIN_INITIAL);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_MAINTENANCE)]               = ::SymbolInfoDouble(this.m_name,SYMBOL_MARGIN_MAINTENANCE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_VOLUME)]                   = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_VOLUME);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_TURNOVER)]                 = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_TURNOVER);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_INTEREST)]                 = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_INTEREST);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME)]        = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_BUY_ORDERS_VOLUME);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME)]       = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_SELL_ORDERS_VOLUME);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_OPEN)]                     = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_OPEN);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_CLOSE)]                    = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_CLOSE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_AW)]                       = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_AW);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_PRICE_SETTLEMENT)]         = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_PRICE_SETTLEMENT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_PRICE_LIMIT_MIN)]          = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_PRICE_LIMIT_MIN);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_PRICE_LIMIT_MAX)]          = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_PRICE_LIMIT_MAX);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BID)]                              = this.m_tick.bid;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASK)]                              = this.m_tick.ask;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LAST)]                             = this.m_tick.last;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BIDHIGH)]                          = this.SymbolBidHigh();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BIDLOW)]                           = this.SymbolBidLow();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_REAL)]                      = this.SymbolVolumeReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUMEHIGH_REAL)]                  = this.SymbolVolumeHighReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUMELOW_REAL)]                   = this.SymbolVolumeLowReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_OPTION_STRIKE)]                    = this.SymbolOptionStrike();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_ACCRUED_INTEREST)]           = this.SymbolTradeAccruedInterest();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_FACE_VALUE)]                 = this.SymbolTradeFaceValue();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_LIQUIDITY_RATE)]             = this.SymbolTradeLiquidityRate();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_HEDGED)]                    = this.SymbolMarginHedged();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_LONG_INITIAL)]              = this.m_margin_rate.Long.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOP_INITIAL)]          = this.m_margin_rate.BuyStop.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_LIMIT_INITIAL)]         = this.m_margin_rate.BuyLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_INITIAL)]     = this.m_margin_rate.BuyStopLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_LONG_MAINTENANCE)]          = this.m_margin_rate.Long.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOP_MAINTENANCE)]      = this.m_margin_rate.BuyStop.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_LIMIT_MAINTENANCE)]     = this.m_margin_rate.BuyLimit.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_MAINTENANCE)] = this.m_margin_rate.BuyStopLimit.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SHORT_INITIAL)]             = this.m_margin_rate.Short.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOP_INITIAL)]         = this.m_margin_rate.SellStop.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_LIMIT_INITIAL)]        = this.m_margin_rate.SellLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_INITIAL)]    = this.m_margin_rate.SellStopLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SHORT_MAINTENANCE)]         = this.m_margin_rate.Short.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOP_MAINTENANCE)]     = this.m_margin_rate.SellStop.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_LIMIT_MAINTENANCE)]    = this.m_margin_rate.SellLimit.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_MAINTENANCE)]= this.m_margin_rate.SellStopLimit.Maintenance;
//--- Save string properties
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_NAME)]                             = this.m_name;
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_CURRENCY_BASE)]                    = ::SymbolInfoString(this.m_name,SYMBOL_CURRENCY_BASE);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_CURRENCY_PROFIT)]                  = ::SymbolInfoString(this.m_name,SYMBOL_CURRENCY_PROFIT);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_CURRENCY_MARGIN)]                  = ::SymbolInfoString(this.m_name,SYMBOL_CURRENCY_MARGIN);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_DESCRIPTION)]                      = ::SymbolInfoString(this.m_name,SYMBOL_DESCRIPTION);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_PATH)]                             = ::SymbolInfoString(this.m_name,SYMBOL_PATH);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_BASIS)]                            = this.SymbolBasis();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_BANK)]                             = this.SymbolBank();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_ISIN)]                             = this.SymbolISIN();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_FORMULA)]                          = this.SymbolFormula();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_PAGE)]                             = this.SymbolPage();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_CATEGORY)]                         = this.SymbolCategory();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_EXCHANGE)]                         = this.SymbolExchange();
//--- Save additional integer properties
   this.m_long_prop[SYMBOL_PROP_DIGITS_LOTS]                                        = this.SymbolDigitsLot();

//--- Fill in the symbol current data
   for(int i=0;i<SYMBOL_PROP_INTEGER_TOTAL;i++)
      this.m_long_prop_event[i][3]=this.m_long_prop[i];
   for(int i=0;i<SYMBOL_PROP_DOUBLE_TOTAL;i++)
      this.m_double_prop_event[i][3]=this.m_double_prop[i];

//--- Update the base object data and search for changes
   CBaseObjExt::Refresh();
//---
   if(!select)
      this.RemoveFromMarketWatch();

//--- Initializing default values of a trading object
   this.m_trade.Init(this.Name(),0,this.LotsMin(),5,0,0,false,this.GetCorrectTypeFilling(),this.GetCorrectTypeExpiration(),LOG_LEVEL_ERROR_MSG);
  }
//+------------------------------------------------------------------+
```

**Implementing the class destructor:**

```
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CSymbol::~CSymbol(void)
  {
   if(this.m_book_subscribed)
      this.BookClose();
  }
//+------------------------------------------------------------------+
```

If the DOM subscription flag is enabled, unsubscribe from DOM broadcast.

Thus, the rule one subscription = one subscription cancellation is fulfilled for any symbol since the method of unsubscribing from DOM is called in the class destructor upon the program operation completion only if a subscription to a symbol has been activated. This works for all symbols with enabled subscription used in the program.

**In the method returning the description of a symbol integer property**, **add the code block for displaying the DOM subscription status property description:**

```
//+------------------------------------------------------------------+
//| Return the description of the symbol integer property            |
//+------------------------------------------------------------------+
string CSymbol::GetPropertyDescription(ENUM_SYMBOL_PROP_INTEGER property)
  {
   return
     (
      property==SYMBOL_PROP_STATUS              ?  CMessage::Text(MSG_ORD_STATUS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetStatusDescription()
         )  :
      property==SYMBOL_PROP_INDEX_MW            ?  CMessage::Text(MSG_SYM_PROP_INDEX)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SYMBOL_PROP_CUSTOM              ?  CMessage::Text(MSG_SYM_PROP_CUSTOM)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property)   ?  CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==SYMBOL_PROP_CHART_MODE          ?  CMessage::Text(MSG_SYM_PROP_CHART_MODE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetChartModeDescription()
         )  :
      property==SYMBOL_PROP_EXIST               ?  CMessage::Text(MSG_SYM_PROP_EXIST)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property)   ?  CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==SYMBOL_PROP_SELECT  ?  CMessage::Text(MSG_SYM_PROP_SELECT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property)   ?  CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==SYMBOL_PROP_VISIBLE ?  CMessage::Text(MSG_SYM_PROP_VISIBLE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property)   ?  CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==SYMBOL_PROP_SESSION_DEALS       ?  CMessage::Text(MSG_SYM_PROP_SESSION_DEALS)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED_MQL4) #endif
         )  :
      property==SYMBOL_PROP_SESSION_BUY_ORDERS  ?  CMessage::Text(MSG_SYM_PROP_SESSION_BUY_ORDERS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED_MQL4) #endif
         )  :
      property==SYMBOL_PROP_SESSION_SELL_ORDERS ?  CMessage::Text(MSG_SYM_PROP_SESSION_SELL_ORDERS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED_MQL4) #endif
         )  :
      property==SYMBOL_PROP_VOLUME              ?  CMessage::Text(MSG_SYM_PROP_VOLUME)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED_MQL4) #endif
         )  :
      property==SYMBOL_PROP_VOLUMEHIGH          ?  CMessage::Text(MSG_SYM_PROP_VOLUMEHIGH)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED_MQL4) #endif
         )  :
      property==SYMBOL_PROP_VOLUMELOW           ?  CMessage::Text(MSG_SYM_PROP_VOLUMELOW)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED_MQL4) #endif
         )  :
      property==SYMBOL_PROP_TIME                ?  CMessage::Text(MSG_SYM_PROP_TIME)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property)==0 ? "("+CMessage::Text(MSG_LIB_SYS_NO_TICKS_YET)+")" : TimeMSCtoString(this.GetProperty(property)))
         )  :
      property==SYMBOL_PROP_DIGITS              ?  CMessage::Text(MSG_SYM_PROP_DIGITS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SYMBOL_PROP_DIGITS_LOTS         ?  CMessage::Text(MSG_SYM_PROP_DIGITS_LOTS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SYMBOL_PROP_SPREAD              ?  CMessage::Text(MSG_SYM_PROP_SPREAD)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SYMBOL_PROP_SPREAD_FLOAT        ?  CMessage::Text(MSG_SYM_PROP_SPREAD_FLOAT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property)   ?  CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_YES))
         )  :
      property==SYMBOL_PROP_TICKS_BOOKDEPTH     ?  CMessage::Text(MSG_SYM_PROP_TICKS_BOOKDEPTH)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED_MQL4) #endif
         )  :
      property==SYMBOL_PROP_BOOKDEPTH_STATE     ?  CMessage::Text(MSG_SYM_SYMBOLS_MODE_BOOK)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+ #ifdef __MQL5__
                  (this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
                #else
                  CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED_MQL4)
                #endif
         )  :
      property==SYMBOL_PROP_TRADE_CALC_MODE     ?  CMessage::Text(MSG_SYM_PROP_TRADE_CALC_MODE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetCalcModeDescription()
         )  :
      property==SYMBOL_PROP_TRADE_MODE ?  CMessage::Text(MSG_SYM_PROP_TRADE_MODE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetTradeModeDescription()
         )  :
      property==SYMBOL_PROP_START_TIME          ?  CMessage::Text(MSG_SYM_PROP_START_TIME)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
         (this.GetProperty(property)==0  ?  ": ("+CMessage::Text(MSG_LIB_PROP_EMPTY)+")" : ": "+TimeMSCtoString(this.GetProperty(property)*1000))
         )  :
      property==SYMBOL_PROP_EXPIRATION_TIME     ?  CMessage::Text(MSG_SYM_PROP_EXPIRATION_TIME)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
         (this.GetProperty(property)==0  ?  ": ("+CMessage::Text(MSG_LIB_PROP_EMPTY)+")" : ": "+TimeMSCtoString(this.GetProperty(property)*1000))
         )  :
      property==SYMBOL_PROP_TRADE_STOPS_LEVEL   ?  CMessage::Text(MSG_SYM_PROP_TRADE_STOPS_LEVEL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SYMBOL_PROP_TRADE_FREEZE_LEVEL  ?  CMessage::Text(MSG_SYM_PROP_TRADE_FREEZE_LEVEL)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SYMBOL_PROP_TRADE_EXEMODE       ?  CMessage::Text(MSG_SYM_PROP_TRADE_EXEMODE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetTradeExecDescription()
         )  :
      property==SYMBOL_PROP_SWAP_MODE           ?  CMessage::Text(MSG_SYM_PROP_SWAP_MODE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetSwapModeDescription()
         )  :
      property==SYMBOL_PROP_SWAP_ROLLOVER3DAYS  ?  CMessage::Text(MSG_SYM_PROP_SWAP_ROLLOVER3DAYS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+DayOfWeekDescription(this.SwapRollover3Days())
         )  :
      property==SYMBOL_PROP_MARGIN_HEDGED_USE_LEG  ?  CMessage::Text(MSG_SYM_PROP_MARGIN_HEDGED_USE_LEG)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED)   :
          ": "+(this.GetProperty(property)   ?  CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==SYMBOL_PROP_EXPIRATION_MODE     ?  CMessage::Text(MSG_SYM_PROP_EXPIRATION_MODE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetExpirationModeFlagsDescription()
         )  :
      property==SYMBOL_PROP_FILLING_MODE        ?  CMessage::Text(MSG_SYM_PROP_FILLING_MODE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetFillingModeFlagsDescription()
         )  :
      property==SYMBOL_PROP_ORDER_MODE          ?  CMessage::Text(MSG_SYM_PROP_ORDER_MODE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetOrderModeFlagsDescription()
         )  :
      property==SYMBOL_PROP_ORDER_GTC_MODE      ?  CMessage::Text(MSG_SYM_PROP_ORDER_GTC_MODE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetOrderGTCModeDescription()
         )  :
      property==SYMBOL_PROP_OPTION_MODE         ?  CMessage::Text(MSG_SYM_PROP_OPTION_MODE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetOptionTypeDescription()
         )  :
      property==SYMBOL_PROP_OPTION_RIGHT        ?  CMessage::Text(MSG_SYM_PROP_OPTION_RIGHT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetOptionRightDescription()
         )  :
      property==SYMBOL_PROP_BACKGROUND_COLOR    ?  CMessage::Text(MSG_SYM_PROP_BACKGROUND_COLOR)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
         #ifdef __MQL5__
         (this.GetProperty(property)==CLR_DEFAULT || this.GetProperty(property)==CLR_NONE ?  ": ("+CMessage::Text(MSG_LIB_PROP_EMPTY)+")" : ": "+::ColorToString((color)this.GetProperty(property),true))
         #else ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED_MQL4) #endif
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

**The method performing subscription to DOM:**

```
//+------------------------------------------------------------------+
//| Subscribe to the Depth of Market                                 |
//+------------------------------------------------------------------+
bool CSymbol::BookAdd(void)
  {
   this.m_book_subscribed=(#ifdef __MQL5__ ::MarketBookAdd(this.m_name) #else false #endif);
   if(this.m_book_subscribed)
     {
      this.m_long_prop[SYMBOL_PROP_BOOKDEPTH_STATE]=this.m_book_subscribed;
      ::Print(CMessage::Text(MSG_SYM_SYMBOLS_BOOK_ADD)+" "+this.m_name);
     }
   return this.m_book_subscribed;
  }
//+------------------------------------------------------------------+
```

The **m\_book\_subscribed** variable storing the subscription status flag receives the result of the [MarketBookAdd()](https://www.mql5.com/en/docs/marketinformation/marketbookadd) function, which opens DOM for a specified instrument and subscribes to notifications about changing the DOM.

If subscription is performed, display the appropriate message.

The method returns the result set in **m\_book\_subscribed.**

**The method for closing the DOM:**

```
//+------------------------------------------------------------------+
//| Close the market depth                                           |
//+------------------------------------------------------------------+
bool CSymbol::BookClose(void)
  {
//--- If the DOM subscription flag is off, subscription is disabled (or not enabled yet). Return 'true'
   if(!this.m_book_subscribed)
      return true;
//--- Save the result of unsubscribing from the DOM
   bool res=( #ifdef __MQL5__ ::MarketBookRelease(this.m_name) #else true #endif );
//--- If unsubscribed successfully, reset the DOM subscription flag and write the status to the object property
   if(res)
     {
      this.m_long_prop[SYMBOL_PROP_BOOKDEPTH_STATE]=this.m_book_subscribed=false;
      ::Print(CMessage::Text(MSG_SYM_SYMBOLS_BOOK_DEL)+" "+this.m_name);
     }
//--- Return the result of unsubscribing from DOM
   return res;
  }
//+------------------------------------------------------------------+
```

The method logic is described in the code comments. The method should return the successful unsubscribing flag. If no DOM subscription has been performed yet, the method immediately returns true indicating unsubscribing from DOM broadcast although there has actually been no subscription.

In the method of updating all symbol properties, add updating the subscription status matching the flag in the **m\_book\_subscribed** variable:

```
//+------------------------------------------------------------------+
//| Update all symbol data                                           |
//+------------------------------------------------------------------+
void CSymbol::Refresh(void)
  {
//--- Update quote data
   if(!this.RefreshRates())
      return;
#ifdef __MQL5__
   ::ResetLastError();
   if(!this.MarginRates())
     {
      this.m_global_error=::GetLastError();
      return;
     }
#endif
//--- Initialize event data
   this.m_is_event=false;
   this.m_hash_sum=0;
//--- Update integer properties
   this.m_long_prop[SYMBOL_PROP_SELECT]                                             = ::SymbolInfoInteger(this.m_name,SYMBOL_SELECT);
   this.m_long_prop[SYMBOL_PROP_VISIBLE]                                            = ::SymbolInfoInteger(this.m_name,SYMBOL_VISIBLE);
   this.m_long_prop[SYMBOL_PROP_SESSION_DEALS]                                      = ::SymbolInfoInteger(this.m_name,SYMBOL_SESSION_DEALS);
   this.m_long_prop[SYMBOL_PROP_SESSION_BUY_ORDERS]                                 = ::SymbolInfoInteger(this.m_name,SYMBOL_SESSION_BUY_ORDERS);
   this.m_long_prop[SYMBOL_PROP_SESSION_SELL_ORDERS]                                = ::SymbolInfoInteger(this.m_name,SYMBOL_SESSION_SELL_ORDERS);
   this.m_long_prop[SYMBOL_PROP_VOLUMEHIGH]                                         = ::SymbolInfoInteger(this.m_name,SYMBOL_VOLUMEHIGH);
   this.m_long_prop[SYMBOL_PROP_VOLUMELOW]                                          = ::SymbolInfoInteger(this.m_name,SYMBOL_VOLUMELOW);
   this.m_long_prop[SYMBOL_PROP_SPREAD]                                             = ::SymbolInfoInteger(this.m_name,SYMBOL_SPREAD);
   this.m_long_prop[SYMBOL_PROP_TICKS_BOOKDEPTH]                                    = ::SymbolInfoInteger(this.m_name,SYMBOL_TICKS_BOOKDEPTH);
   this.m_long_prop[SYMBOL_PROP_START_TIME]                                         = ::SymbolInfoInteger(this.m_name,SYMBOL_START_TIME);
   this.m_long_prop[SYMBOL_PROP_EXPIRATION_TIME]                                    = ::SymbolInfoInteger(this.m_name,SYMBOL_EXPIRATION_TIME);
   this.m_long_prop[SYMBOL_PROP_TRADE_STOPS_LEVEL]                                  = ::SymbolInfoInteger(this.m_name,SYMBOL_TRADE_STOPS_LEVEL);
   this.m_long_prop[SYMBOL_PROP_TRADE_FREEZE_LEVEL]                                 = ::SymbolInfoInteger(this.m_name,SYMBOL_TRADE_FREEZE_LEVEL);
   this.m_long_prop[SYMBOL_PROP_BACKGROUND_COLOR]                                   = this.SymbolBackgroundColor();
   this.m_long_prop[SYMBOL_PROP_BOOKDEPTH_STATE]                                    = this.m_book_subscribed;

//--- Update real properties
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE)]                 = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_TICK_VALUE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE_PROFIT)]          = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_TICK_VALUE_PROFIT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE_LOSS)]            = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_TICK_VALUE_LOSS);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_SIZE)]                  = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_TICK_SIZE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_CONTRACT_SIZE)]              = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_CONTRACT_SIZE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_MIN)]                       = ::SymbolInfoDouble(this.m_name,SYMBOL_VOLUME_MIN);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_MAX)]                       = ::SymbolInfoDouble(this.m_name,SYMBOL_VOLUME_MAX);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_STEP)]                      = ::SymbolInfoDouble(this.m_name,SYMBOL_VOLUME_STEP);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_LIMIT)]                     = ::SymbolInfoDouble(this.m_name,SYMBOL_VOLUME_LIMIT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SWAP_LONG)]                        = ::SymbolInfoDouble(this.m_name,SYMBOL_SWAP_LONG);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SWAP_SHORT)]                       = ::SymbolInfoDouble(this.m_name,SYMBOL_SWAP_SHORT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_INITIAL)]                   = ::SymbolInfoDouble(this.m_name,SYMBOL_MARGIN_INITIAL);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_MAINTENANCE)]               = ::SymbolInfoDouble(this.m_name,SYMBOL_MARGIN_MAINTENANCE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_VOLUME)]                   = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_VOLUME);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_TURNOVER)]                 = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_TURNOVER);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_INTEREST)]                 = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_INTEREST);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME)]        = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_BUY_ORDERS_VOLUME);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME)]       = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_SELL_ORDERS_VOLUME);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_OPEN)]                     = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_OPEN);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_CLOSE)]                    = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_CLOSE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_AW)]                       = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_AW);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_PRICE_SETTLEMENT)]         = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_PRICE_SETTLEMENT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_PRICE_LIMIT_MIN)]          = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_PRICE_LIMIT_MIN);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_PRICE_LIMIT_MAX)]          = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_PRICE_LIMIT_MAX);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_REAL)]                      = this.SymbolVolumeReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUMEHIGH_REAL)]                  = this.SymbolVolumeHighReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUMELOW_REAL)]                   = this.SymbolVolumeLowReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_OPTION_STRIKE)]                    = this.SymbolOptionStrike();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_ACCRUED_INTEREST)]           = this.SymbolTradeAccruedInterest();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_FACE_VALUE)]                 = this.SymbolTradeFaceValue();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_LIQUIDITY_RATE)]             = this.SymbolTradeLiquidityRate();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_HEDGED)]                    = this.SymbolMarginHedged();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_LONG_INITIAL)]              = this.m_margin_rate.Long.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOP_INITIAL)]          = this.m_margin_rate.BuyStop.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_LIMIT_INITIAL)]         = this.m_margin_rate.BuyLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_INITIAL)]     = this.m_margin_rate.BuyStopLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_LONG_MAINTENANCE)]          = this.m_margin_rate.Long.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOP_MAINTENANCE)]      = this.m_margin_rate.BuyStop.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_LIMIT_MAINTENANCE)]     = this.m_margin_rate.BuyLimit.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_MAINTENANCE)] = this.m_margin_rate.BuyStopLimit.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SHORT_INITIAL)]             = this.m_margin_rate.Short.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOP_INITIAL)]         = this.m_margin_rate.SellStop.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_LIMIT_INITIAL)]        = this.m_margin_rate.SellLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_INITIAL)]    = this.m_margin_rate.SellStopLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SHORT_MAINTENANCE)]         = this.m_margin_rate.Short.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOP_MAINTENANCE)]     = this.m_margin_rate.SellStop.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_LIMIT_MAINTENANCE)]    = this.m_margin_rate.SellLimit.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_MAINTENANCE)]= this.m_margin_rate.SellStopLimit.Maintenance;

//--- Fill in the symbol current data
   for(int i=0;i<SYMBOL_PROP_INTEGER_TOTAL;i++)
      this.m_long_prop_event[i][3]=this.m_long_prop[i];
   for(int i=0;i<SYMBOL_PROP_DOUBLE_TOTAL;i++)
      this.m_double_prop_event[i][3]=this.m_double_prop[i];

//--- Update the base object data and search for changes
   CBaseObjExt::Refresh();
   this.CheckEvents();
  }
//+------------------------------------------------------------------+
```

**These are all the changes in the symbol object class enabling us to work with the DOM subscription in subsequent articles.**

### Test

To perform the test, let's use the EA from the previous article and save it in \\MQL5\\Experts\ **TestDoEasy\\Part62\** as **TestDoEasyPart62.mq5**.

In the test, we will select working only with symbols specified in the settings (two symbols). Let's add the parameter specifying the flag of using the subscription on DOMs for all symbols selected for work and see how the DOM subscription is activated, how the tick series data is updated, how new tick data objects are added and the size of their lists is managed according to their specified maximum possible number.

In the input area, add a new input allowing us to decide on whether to subscribe to DOMs for selected EA working symbols:

```
//--- input variables
input    ushort            InpMagic             =  123;  // Magic number
input    double            InpLots              =  0.1;  // Lots
input    uint              InpStopLoss          =  150;  // StopLoss in points
input    uint              InpTakeProfit        =  150;  // TakeProfit in points
input    uint              InpDistance          =  50;   // Pending orders distance (points)
input    uint              InpDistanceSL        =  50;   // StopLimit orders distance (points)
input    uint              InpDistancePReq      =  50;   // Distance for Pending Request's activate (points)
input    uint              InpBarsDelayPReq     =  5;    // Bars delay for Pending Request's activate (current timeframe)
input    uint              InpSlippage          =  5;    // Slippage in points
input    uint              InpSpreadMultiplier  =  1;    // Spread multiplier for adjusting stop-orders by StopLevel
input    uchar             InpTotalAttempts     =  5;    // Number of trading attempts
sinput   double            InpWithdrawal        =  10;   // Withdrawal funds (in tester)
sinput   uint              InpButtShiftX        =  0;    // Buttons X shift
sinput   uint              InpButtShiftY        =  10;   // Buttons Y shift
input    uint              InpTrailingStop      =  50;   // Trailing Stop (points)
input    uint              InpTrailingStep      =  20;   // Trailing Step (points)
input    uint              InpTrailingStart     =  0;    // Trailing Start (points)
input    uint              InpStopLossModify    =  20;   // StopLoss for modification (points)
input    uint              InpTakeProfitModify  =  60;   // TakeProfit for modification (points)
sinput   ENUM_SYMBOLS_MODE InpModeUsedSymbols   =  SYMBOLS_MODE_CURRENT;            // Mode of used symbols list
sinput   string            InpUsedSymbols       =  "EURUSD,AUDUSD,EURAUD,EURCAD,EURGBP,EURJPY,EURUSD,GBPUSD,NZDUSD,USDCAD,USDJPY";  // List of used symbols (comma - separator)
sinput   ENUM_INPUT_YES_NO InpUseBook           =  INPUT_YES;                       // Use Depth of Market
sinput   ENUM_TIMEFRAMES_MODE InpModeUsedTFs    =  TIMEFRAMES_MODE_LIST;            // Mode of used timeframes list
sinput   string            InpUsedTFs           =  "M1,M5,M15,M30,H1,H4,D1,W1,MN1"; // List of used timeframes (comma - separator)
sinput   bool              InpUseSounds         =  true; // Use sounds
```

In the **OnInitDoEasy()** library initialization function, namely in the area of setting reference values for symbols, add the code block for subscribing to the DOM on each of the working symbols. Additionally, implement the display of all properties to the journal for the current symbol to see if the symbol property I have just added works correctly.

```
//+------------------------------------------------------------------+
//| Initializing DoEasy library                                      |
//+------------------------------------------------------------------+
void OnInitDoEasy()
  {
//--- Check if working with the full list is selected
   used_symbols_mode=InpModeUsedSymbols;
   if((ENUM_SYMBOLS_MODE)used_symbols_mode==SYMBOLS_MODE_ALL)
     {
      int total=SymbolsTotal(false);
      string ru_n="\nКоличество символов на сервере "+(string)total+".\nМаксимальное количество: "+(string)SYMBOLS_COMMON_TOTAL+" символов.";
      string en_n="\nNumber of symbols on server "+(string)total+".\nMaximum number: "+(string)SYMBOLS_COMMON_TOTAL+" symbols.";
      string caption=TextByLanguage("Внимание!","Attention!");
      string ru="Выбран режим работы с полным списком.\nВ этом режиме первичная подготовка списков коллекций символов и таймсерий может занять длительное время."+ru_n+"\nПродолжить?\n\"Нет\" - работа с текущим символом \""+Symbol()+"\"";
      string en="Full list mode selected.\nIn this mode, the initial preparation of lists of symbol collections and timeseries can take a long time."+en_n+"\nContinue?\n\"No\" - working with the current symbol \""+Symbol()+"\"";
      string message=TextByLanguage(ru,en);
      int flags=(MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2);
      int mb_res=MessageBox(message,caption,flags);
      switch(mb_res)
        {
         case IDNO :
           used_symbols_mode=SYMBOLS_MODE_CURRENT;
           break;
         default:
           break;
        }
     }
//--- Set the counter start point to measure the approximate library initialization time
   ulong begin=GetTickCount();
   Print(TextByLanguage("--- Инициализация библиотеки \"DoEasy\" ---","--- Initializing the \"DoEasy\" library ---"));
//--- Fill in the array of used symbols
   CreateUsedSymbolsArray((ENUM_SYMBOLS_MODE)used_symbols_mode,InpUsedSymbols,array_used_symbols);
//--- Set the type of the used symbol list in the symbol collection and fill in the list of symbol timeseries
   engine.SetUsedSymbols(array_used_symbols);
//--- Displaying the selected mode of working with the symbol object collection in the journal
   string num=
     (
      used_symbols_mode==SYMBOLS_MODE_CURRENT ? ": \""+Symbol()+"\"" :
      TextByLanguage(". Количество используемых символов: ",". The number of symbols used: ")+(string)engine.GetSymbolsCollectionTotal()
     );
   Print(engine.ModeSymbolsListDescription(),num);
//--- Implement displaying the list of used symbols only for MQL5 - MQL4 has no ArrayPrint() function
#ifdef __MQL5__
   if(InpModeUsedSymbols!=SYMBOLS_MODE_CURRENT)
     {
      string array_symbols[];
      CArrayObj* list_symbols=engine.GetListAllUsedSymbols();
      for(int i=0;i<list_symbols.Total();i++)
        {
         CSymbol *symbol=list_symbols.At(i);
         if(symbol==NULL)
            continue;
         ArrayResize(array_symbols,ArraySize(array_symbols)+1,SYMBOLS_COMMON_TOTAL);
         array_symbols[ArraySize(array_symbols)-1]=symbol.Name();
        }
      ArrayPrint(array_symbols);
     }
#endif
//--- Set used timeframes
   CreateUsedTimeframesArray(InpModeUsedTFs,InpUsedTFs,array_used_periods);
//--- Display the selected mode of working with the timeseries object collection
   string mode=
     (
      InpModeUsedTFs==TIMEFRAMES_MODE_CURRENT   ?
         TextByLanguage("Работа только с текущим таймфреймом: ","Work only with the current Period: ")+TimeframeDescription((ENUM_TIMEFRAMES)Period())   :
      InpModeUsedTFs==TIMEFRAMES_MODE_LIST      ?
         TextByLanguage("Работа с заданным списком таймфреймов:","Work with a predefined list of Periods:")                                              :
      TextByLanguage("Работа с полным списком таймфреймов:","Work with the full list of all Periods:")
     );
   Print(mode);
//--- Implement displaying the list of used timeframes only for MQL5 - MQL4 has no ArrayPrint() function
#ifdef __MQL5__
   if(InpModeUsedTFs!=TIMEFRAMES_MODE_CURRENT)
      ArrayPrint(array_used_periods);
#endif
//--- Create timeseries of all used symbols
   engine.SeriesCreateAll(array_used_periods);

//--- Check created timeseries - display descriptions of all created timeseries in the journal
//--- (true - only created ones, false - created and declared ones)
   engine.GetTimeSeriesCollection().PrintShort(false); // Short descriptions
   //engine.GetTimeSeriesCollection().Print(true);      // Full descriptions

//--- Create tick series of all used symbols
   engine.GetTickSeriesCollection().CreateTickSeriesAll();
//--- Check created tick series - display descriptions of all created tick series in the journal
   engine.GetTickSeriesCollection().Print();

//--- Create resource text files
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_01",TextByLanguage("Звук упавшей монетки 1","Falling coin 1"),sound_array_coin_01);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_02",TextByLanguage("Звук упавших монеток","Falling coins"),sound_array_coin_02);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_03",TextByLanguage("Звук монеток","Coins"),sound_array_coin_03);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_04",TextByLanguage("Звук упавшей монетки 2","Falling coin 2"),sound_array_coin_04);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_01",TextByLanguage("Звук щелчка по кнопке 1","Button click 1"),sound_array_click_01);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_02",TextByLanguage("Звук щелчка по кнопке 2","Button click 2"),sound_array_click_02);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_03",TextByLanguage("Звук щелчка по кнопке 3","Button click 3"),sound_array_click_03);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_cash_machine_01",TextByLanguage("Звук кассового аппарата","Cash machine"),sound_array_cash_machine_01);
   engine.CreateFile(FILE_TYPE_BMP,"img_array_spot_green",TextByLanguage("Изображение \"Зелёный светодиод\"","Image \"Green Spot lamp\""),img_array_spot_green);
   engine.CreateFile(FILE_TYPE_BMP,"img_array_spot_red",TextByLanguage("Изображение \"Красный светодиод\"","Image \"Red Spot lamp\""),img_array_spot_red);

//--- Pass all existing collections to the main library class
   engine.CollectionOnInit();

//--- Set the default magic number for all used symbols
   engine.TradingSetMagic(engine.SetCompositeMagicNumber(magic_number));
//--- Set synchronous passing of orders for all used symbols
   engine.TradingSetAsyncMode(false);
//--- Set the number of trading attempts in case of an error
   engine.TradingSetTotalTry(InpTotalAttempts);
//--- Set correct order expiration and filling types to all trading objects
   engine.TradingSetCorrectTypeExpiration();
   engine.TradingSetCorrectTypeFilling();

//--- Set standard sounds for trading objects of all used symbols
   engine.SetSoundsStandart();
//--- Set the general flag of using sounds
   engine.SetUseSounds(InpUseSounds);
//--- Set the spread multiplier for symbol trading objects in the symbol collection
   engine.SetSpreadMultiplier(InpSpreadMultiplier);

//--- Set controlled values for symbols
   //--- Get the list of all collection symbols
   CArrayObj *list=engine.GetListAllUsedSymbols();
   if(list!=NULL && list.Total()!=0)
     {
      //--- In a loop by the list, set the necessary values for tracked symbol properties
      //--- By default, the LONG_MAX value is set to all properties, which means "Do not track this property"
      //--- It can be enabled or disabled (by setting the value less than LONG_MAX or vice versa - set the LONG_MAX value) at any time and anywhere in the program

      for(int i=0;i<list.Total();i++)
        {
         CSymbol* symbol=list.At(i);
         if(symbol==NULL)
            continue;
         if(InpUseBook)
            symbol.BookAdd();
         if(symbol.Name()==Symbol())
            symbol.Print();
         /*
         //--- Set control of the symbol price increase by 100 points
         symbol.SetControlBidInc(100000*symbol.Point());
         //--- Set control of the symbol price decrease by 100 points
         symbol.SetControlBidDec(100000*symbol.Point());
         //--- Set control of the symbol spread increase by 40 points
         symbol.SetControlSpreadInc(400);
         //--- Set control of the symbol spread decrease by 40 points
         symbol.SetControlSpreadDec(400);
         //--- Set control of the current spread by the value of 40 points
         symbol.SetControlSpreadLevel(400);
         */
        }

     }
//--- Set controlled values for the current account
   CAccount* account=engine.GetAccountCurrent();
   if(account!=NULL)
     {
      //--- Set control of the profit increase to 10
      account.SetControlledValueINC(ACCOUNT_PROP_PROFIT,10.0);
      //--- Set control of the funds increase to 15
      account.SetControlledValueINC(ACCOUNT_PROP_EQUITY,15.0);
      //--- Set profit control level to 20
      account.SetControlledValueLEVEL(ACCOUNT_PROP_PROFIT,20.0);
     }

//--- Get the end of the library initialization time counting and display it in the journal
   ulong end=GetTickCount();
   Print(TextByLanguage("Время инициализации библиотеки: ","Library initialization time: "),TimeMSCtoString(end-begin,TIME_MINUTES|TIME_SECONDS));
  }
//+------------------------------------------------------------------+
```

Compile the EA, launch it on EURUSD chart, while preliminarily selecting EURUSD and AUDUSD from the list and subscription to DOMs of all selected symbols:

![](https://c.mql5.com/2/42/z3mH31CVgK.png)

After launching the EA, the journal receives messages about an activated subscription to DOMs of two symbols. All EURUSD properties are then displayed. The line of the new property showing the DOM subscription status is among them:

```
Account 8550475: Artyom Trishkin (MetaQuotes Software Corp.) 10426.13 USD, 1:100, Hedge, MetaTrader 5 demo
--- Initializing "DoEasy" library ---
Working with predefined symbol list. The number of used symbols: 2
"AUDUSD" "EURUSD"
Working with the current timeframe only: H1
AUDUSD symbol timeseries:
- Timeseries "AUDUSD" H1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 6325
EURUSD symbol timeseries:
- Timeseries "EURUSD" H1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5806
Tick series "AUDUSD": Requested number of days: 1, Historical data created: 183398
Tick series "EURUSD": Requested number of days: 1, Historical data created: 148089
Subscribed to Depth of Market  AUDUSD
Subscribed to Depth of Market  EURUSD
============= Beginning of parameter list: "EURUSD" (Euro vs US Dollar) ==================
Status: Major Forex symbol
Index in Market Watch: 2
Custom symbol: No
Price type used for generating bars: Bars are built based on Bid prices
Symbol selected in Market Watch: Yes
Symbol visible in Market Watch: Yes
Number of deals in the current session: 0
Total number of Buy orders at the moment: 0
Total number of Sell orders at the moment: 0
Volume of the last deal: 0
Maximal day volume: 0
Minimal day volume: 0
Time of the last quote: 2021.01.26 22:41:04.852
Number of decimal places: 5
Digits after a decimal point in the value of the lot: 2
Spread value in points: 2
Floating spread: Yes
Maximum number of requests displayed in DOM: 10
Subscription to DOM: Yes
Contract price calculation mode: Forex mode
Order execution type: No trading limitations
Trading start date for an instrument: (No)
Trading end date for an instrument: (No)
Minimal indention from the close price to place Stop orders: 0
Freeze distance for trading operations: 0
Deal execution mode: Instant execution
Swap calculation model: Swaps charged in points
Triple-day swap: Wednesday
Calculating hedging margin using the larger leg: No
Flags of allowed order expiration modes:
 - Unlimited (Yes)
 - Valid till the end of the day (Yes)
 - Time is specified in the order (Yes)
 - Date specified in order (Yes)
Flags of allowed order filling modes:
 - Return (Yes)
 - Fill or Kill (Yes)
 - Immediate or Cancel order (No)
Flags of allowed order types:
 - Market order (Yes)
 - Limit order (Yes)
 - Stop order (Yes)
 - Stop limit order (Yes)
 - StopLoss (Yes)
 - TakeProfit (Yes)
 - Close by (Yes)
StopLoss and TakeProfit order validity periods: Pending orders and Stop Loss/Take Profit levels valid for unlimited period until their explicit cancellation
Option type: European option may only be exercised on a specified date
Option right: Call option gives you right to buy asset at specified price
Background color of the symbol in Market Watch: (No)
------
Bid price: 1.21665
Highest Bid price of the day: 1.21760
Lowest Bid price of the day: 1.21078
Ask price: 1.21667
Highest Ask price of the day: 1.21760
Lowest Ask price of the day: 1.21081
Real volume of the day: 0.00
Maximum real volume of the day: 0.00
Minimum real volume of the day: 0.00
Option execution price: 0.00000
Point value: 0.00001
Calculated tick value for a position: 1.00
Calculated tick value for a winning position: 1.00
Calculated tick value for a losing position: 1.00
Minimum price change: 0.00001
Trade contract size: 100000.00
Accrued interest: 0.00
Initial bond value set by the issuer: 0.00
Liquidity rate: 0.00
Minimum volume for deal execution: 0.01
Maximum volume for deal execution: 500.00
Minimal volume change step for deal execution: 0.01
Maximum allowed aggregate volume of an open position and pending orders in one direction: 0.00
Long swap value: -0.70
Short swap value: -1.00
Initial margin: 0.00000000
Maintenance margin for an instrument: 0.00000000
Initial margin requirement applicable to long positions: 1.00000000
Initial margin requirement applicable to BuyStop orders: (Value not set)
Initial margin requirement applicable to BuyLimit orders: (Value not set)
Initial margin requirement applicable to BuyStopLimit orders: (Value not set)
Maintenance margin requirement applicable to long positions: (Value not set)
Maintenance margin requirement applicable to BuyStop orders: (Value not set)
Maintenance margin requirement applicable to BuyLimit orders: (Value not set)
Maintenance margin requirement applicable to BuyStopLimit orders: (Value not set)
Initial margin requirement applicable to short positions: 1.00000000
Initial margin requirement applicable to SellStop orders: (Value not set)
Initial margin requirement applicable to SellLimit orders: (Value not set)
Initial margin requirement applicable to SellStopLimit orders: (Value not set)
Maintenance margin requirement applicable to short positions: (Value not set)
Maintenance margin requirement applicable to SellStop orders: (Value not set)
Maintenance margin requirement applicable to SellLimit orders: (Value not set)
Maintenance margin requirement applicable to SellStopLimit orders: (Value not set)
Total volume of deals in the current session: 0.00
Total turnover in the current session: 0.00
Total volume of open positions: 0.00
Total volume of Buy orders at the moment: 0.00
Total volume of Sell orders at the moment: 0.00
Open price of the session: 1.21371
Close price of the session: 1.21413
Average weighted price of the session: 0.00000
Settlement price of the current session: 0.00000
Minimum allowable price value for the session: 0.00000
Maximum allowable price value for the session: 0.00000
Size of a contract or margin for one lot of hedged positions: 100000.00
------
Symbol name: EURUSD
Name of the underlaying asset for a derivative symbol: (No)
Instrument base currency: "EUR"
Profit currency: "USD"
Margin funds currency: "EUR"
Source of the current quote: (No)
Symbol description: "Euro vs US Dollar"
Symbol name in ISIN system: (No)
Address of the web page containing symbol information: "http://www.google.com/finance?q=EURUSD"
Location in the symbol tree: "Forex\EURUSD"
Name of the category or sector the symbol belongs to: (No)
Name of the exchange in which the security is traded: (No)
================== End of parameter list: "EURUSD" (Euro vs US Dollar) ==================

Library initialization time: 00:00:09.953
```

The string from the Refresh() method of the tick series class for AUDUSD is displayed in the comment of the chart — number of newly copied ticks, previous time current time and the total number of tick data objects present in the tick series list:

![](https://c.mql5.com/2/42/terminal64_mqeDjoh0KV.png)

### What's next?

In the next article, I will start creating the library functionality allowing to work with symbol DOMs.

All files of the current version of the library are attached below together with the test EA file for MQL5 for you to test and download.

Leave your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/8988#node00)

**\*Previous articles within the series:**

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

[Timeseries in DoEasy library (part 47): Multi-period multi-symbol standard indicators](https://www.mql5.com/en/articles/8207)

[Timeseries in DoEasy library (part 48): Multi-period multi-symbol indicators on one buffer in a subwindow](https://www.mql5.com/en/articles/8257)

[Timeseries in DoEasy library (part 49): Multi-period multi-symbol multi-buffer standard indicators](https://www.mql5.com/en/articles/8292)

[Timeseries in DoEasy library (part 50): Multi-period multi-symbol standard indicators with a shift](https://www.mql5.com/en/articles/8331)

[Timeseries in DoEasy library (part 51): Composite multi-period multi-symbol standard indicators](https://www.mql5.com/en/articles/8354)

[Timeseries in DoEasy library (part 52): Cross-platform nature of multi-period multi-symbol single-buffer standard indicators](https://www.mql5.com/en/articles/8399)

[Timeseries in DoEasy library (part 53): Abstract base indicator class](https://www.mql5.com/en/articles/8464)

[Timeseries in DoEasy library (part 54): Descendant classes of abstract base indicator](https://www.mql5.com/en/articles/8508)

[Timeseries in DoEasy library (part 55): Indicator collection class](https://www.mql5.com/en/articles/8576/)

[Timeseries in DoEasy library (part 56): Custom indicator object, get data from indicator objects in the collection](https://www.mql5.com/en/articles/8646)

[Timeseries in DoEasy library (part 57): Indicator buffer data object](https://www.mql5.com/en/articles/8705)

[Timeseries in DoEasy library (part 58): Timeseries of indicator buffer data](https://www.mql5.com/en/articles/8787)

[Prices in DoEasy library (part 59): Object to store data of one tick](https://www.mql5.com/en/articles/8818)

[Prices in DoEasy library (part 60): Series list of symbol tick data](https://www.mql5.com/en/articles/8912)

[Prices in DoEasy library (part 61): Collection of symbol tick series](https://www.mql5.com/en/articles/8952)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8988](https://www.mql5.com/ru/articles/8988)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8988.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/8988/mql5.zip "Download MQL5.zip")(3881.55 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/365590)**

![Neural networks made easy (Part 11): A take on GPT](https://c.mql5.com/2/48/Neural_networks_made_easy_011.png)[Neural networks made easy (Part 11): A take on GPT](https://www.mql5.com/en/articles/9025)

Perhaps one of the most advanced models among currently existing language neural networks is GPT-3, the maximal variant of which contains 175 billion parameters. Of course, we are not going to create such a monster on our home PCs. However, we can view which architectural solutions can be used in our work and how we can benefit from them.

![Prices in DoEasy library (part 61): Collection of symbol tick series](https://c.mql5.com/2/41/MQL5-avatar-doeasy-library__5.png)[Prices in DoEasy library (part 61): Collection of symbol tick series](https://www.mql5.com/en/articles/8952)

Since a program may use different symbols in its work, a separate list should be created for each of them. In this article, I will combine such lists into a tick data collection. In fact, this will be a regular list based on the class of dynamic array of pointers to instances of CObject class and its descendants of the Standard library.

![Prices in DoEasy library (part 63): Depth of Market and its abstract request class](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__1.png)[Prices in DoEasy library (part 63): Depth of Market and its abstract request class](https://www.mql5.com/en/articles/9010)

In the article, I will start developing the functionality for working with the Depth of Market. I will also create the class of the Depth of Market abstract order object and its descendants.

![Prices in DoEasy library (part 60): Series list of symbol tick data](https://c.mql5.com/2/41/MQL5-avatar-doeasy-library__4.png)[Prices in DoEasy library (part 60): Series list of symbol tick data](https://www.mql5.com/en/articles/8912)

In this article, I will create the list for storing tick data of a single symbol and check its creation and retrieval of required data in an EA. Tick data lists that are individual for each used symbol will further constitute a collection of tick data.

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/8988&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083153581109941764)

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