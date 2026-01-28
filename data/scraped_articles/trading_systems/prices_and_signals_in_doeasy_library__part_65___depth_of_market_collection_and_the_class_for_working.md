---
title: Prices and Signals in DoEasy library (Part 65): Depth of Market collection and the class for working with MQL5.com Signals
url: https://www.mql5.com/en/articles/9095
categories: Trading Systems
relevance_score: 4
scraped_at: 2026-01-23T17:47:08.590401
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/9095&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068646014807636969)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/9095#node01)
- [Improving library classes](https://www.mql5.com/en/articles/9095#node02)
- [DOM collection class](https://www.mql5.com/en/articles/9095#node03)
- [MQL5 signal object class](https://www.mql5.com/en/articles/9095#node04)
- [Test](https://www.mql5.com/en/articles/9095#node05)
- [What's next?](https://www.mql5.com/en/articles/9095#node06)


### Concept

We already have the functionality for working with a DOM of any symbol — in the previous articles, I have created the classes of the DOM abstract order objects and its descendants, the DOM snapshot class and the DOM snapshot series class. Now it remains to create a common storage for DOM snapshot series objects — the snapshot series collection class, which is to store all these series with a convenient access to any DOM snapshot stored in the collection lists and auto update (adding new snapshots and removing old ones) for supporting specified series size.

Apart from creating the DOM snapshot series collection class, I will also start the new library section — other library classes.

I will start with the functionality for working with the [MQL5.com signal service](https://www.mql5.com/en/signals), namely I will create the signal object class storing all data of a single signal broadcast by the MQL5.com Signals service.

### Improving library classes

Let's add the new messages to the library right away. In \\MQL5\\Include\\DoEasy\ **Data.mqh**, add new message indices:

```
//--- CMBookSeries
   MSG_MBOOK_SERIES_TEXT_MBOOKSERIES,                 // DOM snapshot series
   MSG_MBOOK_SERIES_ERR_ADD_TO_LIST,                  // Error. Failed to add DOM snapshot series to the list

//--- CMBookSeriesCollection
   MSG_MB_COLLECTION_TEXT_MBCOLLECTION,               // DOM snapshot series collection

//--- CMQLSignal
   MSG_SIGNAL_MQL5_TEXT_SIGNAL,                       // Signal
   MSG_SIGNAL_MQL5_TEXT_SIGNAL_MQL5,                  // MQL5.com Signals service signal
   MSG_SIGNAL_MQL5_TRADE_MODE,                        // Account type
   MSG_SIGNAL_MQL5_DATE_PUBLISHED,                    // Publication date
   MSG_SIGNAL_MQL5_DATE_STARTED,                      // Monitoring start date
   MSG_SIGNAL_MQL5_DATE_UPDATED,                      // Date of the latest update of the trading statistics
   MSG_SIGNAL_MQL5_ID,                                // ID
   MSG_SIGNAL_MQL5_LEVERAGE,                          // Trading account leverage
   MSG_SIGNAL_MQL5_PIPS,                              // Trading result in pips
   MSG_SIGNAL_MQL5_RATING,                            // Position in the signal rating
   MSG_SIGNAL_MQL5_SUBSCRIBERS,                       // Number of subscribers
   MSG_SIGNAL_MQL5_TRADES,                            // Number of trades
   MSG_SIGNAL_MQL5_SUBSCRIPTION_STATUS,               // Status of account subscription to a signal

   MSG_SIGNAL_MQL5_EQUITY,                            // Account equity
   MSG_SIGNAL_MQL5_GAIN,                              // Account growth in %
   MSG_SIGNAL_MQL5_MAX_DRAWDOWN,                      // Maximum drawdown
   MSG_SIGNAL_MQL5_PRICE,                             // Signal subscription price
   MSG_SIGNAL_MQL5_ROI,                               // Signal ROI (Return on Investment) in %

   MSG_SIGNAL_MQL5_AUTHOR_LOGIN,                      // Author login
   MSG_SIGNAL_MQL5_BROKER,                            // Broker (company) name
   MSG_SIGNAL_MQL5_BROKER_SERVER,                     // Broker server
   MSG_SIGNAL_MQL5_NAME,                              // Name
   MSG_SIGNAL_MQL5_CURRENCY,                          // Account currency

   MSG_SIGNAL_MQL5_TEXT_GAIN,                         // Growth
   MSG_SIGNAL_MQL5_TEXT_DRAWDOWN,                     // Drawdown

  };
//+------------------------------------------------------------------+
```

and message texts corresponding to newly added indices:

```
//--- CMBookSeries
   {"Серия снимков стакана цен","Series of shots of the Depth of Market"},
   {"Ошибка. Не удалось добавить серию снимков стакана цен в список","Error. Failed to add a shots series of the Depth of Market to the list"},

//--- CMBookSeriesCollection
   {"Коллекция серий снимков стакана цен","Collection of series of the Depth of Market shot"},

//--- CMQLSignal
   {"Сигнал","Signal"},
   {"Сигнал сервиса сигналов MQL5.com","Signal from MQL5.com signal service"},
   {"Тип счета","Account type"},
   {"Дата публикации","Publication date"},
   {"Дата начала мониторинга","Monitoring starting date"},
   {"Дата последнего обновления торговой статистики","The date of the last update of the signal's trading statistics"},
   {"ID","ID"},
   {"Плечо торгового счета","Account leverage"},
   {"Результат торговли в пипсах","Profit in pips"},
   {"Позиция в рейтинге сигналов","Position in rating"},
   {"Количество подписчиков","Number of subscribers"},
   {"Количество трейдов","Number of trades"},
   {"Состояние подписки счёта на этот сигнал","Account subscription status for this signal"},

   {"Средства на счете","Account equity"},
   {"Прирост счета в процентах","Account gain"},
   {"Максимальная просадка","Account maximum drawdown"},
   {"Цена подписки на сигнал","Signal subscription price"},
   {"Значение ROI (Return on Investment) сигнала в %","Return on Investment (%)"},

   {"Логин автора","Author login"},
   {"Наименование брокера (компании)","Broker name (company)"},
   {"Сервер брокера","Broker server"},
   {"Имя","Name"},
   {"Валюта счета","Base currency"},

   {"Прирост","Gain"},
   {"Просадка","Drawdown"},

  };
//+---------------------------------------------------------------------+
```

Since I am going to develop a new collection today, we need to set its ID. In the ID section in \\MQL5\\Include\\DoEasy\ **Defines.mqh**, add the DOM snapshot series collection ID:

```
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
#define COLLECTION_MBOOKSERIES_ID      (0x7784)                   // DOM series collection list ID
//--- Data parameters for file operations
```

Let's improve the file of the DOM snapshot series object class. In some instances, the object description should be displayed once, while sometimes, we need to display descriptions of all DOM snapshot series at once. In this case, the list looks more visually appealing if a hyphen is added before the description of each series. In the file of the DOM snapshot series class \\MQL5\\Include\\DoEasy\\Objects\\Book\ **MBookSeries.mqh**, add the changes in the description of the methods:

```
//--- Display (1) description and (2) short description of a DOM snapshot series
   void              Print(const bool dash=false);
   void              PrintShort(const bool dash=false);

//--- Constructors
```

Simply use the flag indicating the necessity of displaying a hyphen before the object description.

Add the same changes in the implementation of the methods:

```
//+------------------------------------------------------------------+
//| Display the description of the DOM snapshot series in the journal|
//+------------------------------------------------------------------+
void CMBookSeries::Print(const bool dash=false)
  {
   string txt=
     (
      CMessage::Text(MSG_TICKSERIES_REQUIRED_HISTORY_DAYS)+(string)this.RequiredUsedDays()+", "+
      CMessage::Text(MSG_LIB_TEXT_TS_ACTUAL_DEPTH)+(string)this.DataTotal()
     );
   ::Print((dash ? "- " : ""),this.Header(),": ",txt);
  }
//+------------------------------------------------------------------+
//| Display a short description of a DOM snapshot series             |
//+------------------------------------------------------------------+
void CMBookSeries::PrintShort(const bool dash=false)
  {
   ::Print((dash ? "- " : ""),this.Header());
  }
//+------------------------------------------------------------------+
```

By default, the flag is set to false, so no hyphen is displayed before the object description.

### DOM collection class

We have all the necessary objects for creating the DOM collection. Namely, we have the DOM order object represented by the [MqlBookInfo](https://www.mql5.com/en/docs/constants/structures/mqlbookinfo) structure in the terminal. When the [BookEvent](https://www.mql5.com/en/docs/runtime/event_fire#bookevent) event arrives, the [OnBookEvent()](https://www.mql5.com/en/docs/event_handlers/onbookevent) handler is called. A symbol, on which the DOM change event has occurred, is specified as its parameter. We are able to receive all current DOM data to the MqlBookInfo structure array. The collection of this data in the library is described by the DOM snapshot object featuring order objects we obtained when handling the DOM change event. During each DOM change, we create a new snapshot object we add to the list. Such a list is represented by the DOM snapshot series object. We create such lists for each symbol we have subscribed to in order to receive DOM events. The lists are created in real time (unfortunately, the terminal has no DOM change history for a symbol). As a result, we now need to combine all existing lists into a single DOM collection object.

The symbol DOM collection is to store constantly updated DOM snapshot lists allowing us to create the history of DOM changes for each symbol while the program is running. We will be able to get data on any snapshot present in the collection lists. Besides, we will retrieve any order out of each snapshot. We will be able to search for the required data, sort out the lists by specified criteria and conduct statistical studies with the available collection lists.

In \\MQL5\\Include\\DoEasy\ **Collections\**, create the new class **CMBookSeriesCollection** in **BookSeriesCollection.mqh**.

The class of the basic object of all CBaseObj library objects should be used as the base one, while all the necessary files should be included into the class listing:

```
//+------------------------------------------------------------------+
//|                                         BookSeriesCollection.mqh |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://MQL5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://MQL5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Objects\Book\MBookSeries.mqh"
#include "..\Objects\Symbols\Symbol.mqh"
//+------------------------------------------------------------------+
//| Collection of symbol DOM series                                  |
//+------------------------------------------------------------------+
class CMBookSeriesCollection : public CBaseObj
  {
  }
```

Let's have a look at the class body and its methods, and analyze the implementation of the methods and their purpose:

```
//+------------------------------------------------------------------+
//| Collection of symbol DOM series                                  |
//+------------------------------------------------------------------+
class CMBookSeriesCollection : public CBaseObj
  {
private:
   CListObj                m_list;                                   // List of used symbol DOM series
//--- Return the DOM series index by a symbol name
   int                     IndexMBookSeries(const string symbol);
public:
//--- Return (1) itself, (2) DOM series collection list and (3) the number of DOM series in the list
   CMBookSeriesCollection *GetObject(void)                              { return &this;               }
   CArrayObj              *GetList(void)                                { return &this.m_list;        }
   int                     DataTotal(void)                        const { return this.m_list.Total(); }
//--- Return the pointer to the DOM series object (1) by symbol and (2) by index in the list
   CMBookSeries           *GetMBookseries(const string symbol);
   CMBookSeries           *GetMBookseries(const int index)              { return this.m_list.At(index);}
//--- Create symbol DOM series collection list
   bool                    CreateCollection(const CArrayObj *list_symbols,const uint required=0);
//--- Set the flag of using the DOM series of (1) a specified symbol and (2) all symbols
   void                    SetAvailableMBookSeries(const string symbol,const bool flag=true);
   void                    SetAvailableMBookSeries(const bool flag=true);
//--- Return the flag of using the DOM series of (1) a specified symbol and (2) all symbols
   bool                    IsAvailableMBookSeries(const string symbol);
   bool                    IsAvailableMBookSeries(void);

//--- Set the number of DOM history days of (1) a specified symbol and (2) all symbols
   bool                    SetRequiredUsedDays(const string symbol,const uint required=0);
   bool                    SetRequiredUsedDays(const uint required=0);

//--- Return the DOM snapshot object of a specified symbol (1) by index and (2) by time in milliseconds
   CMBookSnapshot         *GetMBook(const string symbol,const int index);
   CMBookSnapshot         *GetMBook(const string symbol,const long time_msc);

//--- Update the DOM series of a specified symbol
   bool                    Refresh(const string symbol,const long time_msc);

//--- Display (1) the complete and (2) short collection description in the journal
   void                    Print(void);
   void                    PrintShort(void);

//--- Constructor
                           CMBookSeriesCollection();
  };
//+------------------------------------------------------------------+
```

The method descriptions make it clear that the collection itself is represented by **m\_list**, which is to store series objects of snapshots of DOMs for different symbols represented by the CMBookSeries class. These are the symbols featuring DOM snapshot objects represented by the CMBookSnapshot class.

Let's consider the implementation of the class methods in more details.

**In the class constructor**, clear the collection list, set the sorted list flag to the list and assign the DOM series collection list ID:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CMBookSeriesCollection::CMBookSeriesCollection()
  {
   this.m_list.Clear();
   this.m_list.Sort();
   this.m_list.Type(COLLECTION_MBOOKSERIES_ID);
  }
//+------------------------------------------------------------------+
```

By default, all series lists are sorted in the collection list by symbol name of DOM snapshot series.

**The method of creating the symbol DOM series collection list:**

```
//+------------------------------------------------------------------+
//| Create symbol DOM series collection list                         |
//+------------------------------------------------------------------+
bool CMBookSeriesCollection::CreateCollection(const CArrayObj *list_symbols,const uint required=0)
  {
//--- If an empty list of symbol objects is passed, exit
   if(list_symbols==NULL)
      return false;
//--- Get the number of symbol objects in the passed list
   int total=list_symbols.Total();
//--- Clear the DOM series collection list
   this.m_list.Clear();
//--- In a loop by all symbol objects
   for(int i=0;i<total;i++)
     {
      //--- get the next symbol object
      CSymbol *symbol_obj=list_symbols.At(i);
      //--- if failed to get a symbol object, move on to the next one in the list
      if(symbol_obj==NULL)
         continue;
      //--- Create a new DOM series object
      CMBookSeries *bookseries=new CMBookSeries(symbol_obj.Name(),required);
      //--- If failed to create the DOM series object, move on to the next symbol in the list
      if(bookseries==NULL)
         continue;
      //--- Set the sorted list flag for the DOM series collection list
      this.m_list.Sort();
      //--- If the object with the same symbol name is already present in the DOM series collection list, remove the DOM series object
      if(this.m_list.Search(bookseries)>WRONG_VALUE)
         delete bookseries;
      //--- otherwise, there is no object with such a symbol name in the collection yet
      else
        {
         //--- if failed to add the DOM series object to the collection list, remove the series object,
         //--- inform of that and return 'false'
         if(!this.m_list.Add(bookseries))
           {
            delete bookseries;
            ::Print(DFUN,"\"",symbol_obj.Name(),"\": ",CMessage::Text(MSG_MBOOK_SERIES_ERR_ADD_TO_LIST));
            return false;
           }
        }
     }
//--- Collection created successfully, return 'true'
   return true;
  }
//+------------------------------------------------------------------+
```

All method logic is described in detail in its listing. Briefly, the method receives the list of all symbols used in the program and the number of DOM snapshots than can be stored in the collection lists. In the loop by the list passed to the method, receive the next symbol object from the list, create a new DOM snapshot series object by the symbol and add it to the collection list. As a result, we get the list of snapshot series objects of DOMs for different symbols — the ones present in the list passed to the method. Each created snapshot series list is initially empty.

The lists created in the list collection are filled **in the method of updating the snapshot series list of a DOM of a specified symbol:**

```
//+------------------------------------------------------------------+
//| Update the snapshot series list of a specified symbol DOM        |
//+------------------------------------------------------------------+
bool CMBookSeriesCollection::Refresh(const string symbol,const long time_msc)
  {
   CMBookSeries *bookseries=this.GetMBookseries(symbol);
   if(bookseries==NULL)
      return false;
   return bookseries.Refresh(time_msc);
  }
//+------------------------------------------------------------------+
```

Since the OnBookEvent() handler is activated for each symbol, the update method is used only for one symbol from the collection rather than for the entire collection. The method receives the name of a symbol, on which the DOM change event happened, and the time of registering a DOM change event in milliseconds. Further on, we receive the DOM snashot series object by a symbol passed to the method and return the Refresh() method result [of the DOM snapshot series object we have considered in the previous article](https://www.mql5.com/en/articles/9044#node04).

**The method of returning the DOM series index by symbol name:**

```
//+------------------------------------------------------------------+
//| Return the DOM series index by a symbol name                     |
//+------------------------------------------------------------------+
int CMBookSeriesCollection::IndexMBookSeries(const string symbol)
  {
   const CMBookSeries *obj=new CMBookSeries(symbol==NULL || symbol=="" ? ::Symbol() : symbol);
   if(obj==NULL)
      return WRONG_VALUE;
   this.m_list.Sort();
   int index=this.m_list.Search(obj);
   delete obj;
   return index;
  }
//+------------------------------------------------------------------+
```

Here, we create a new temporary DOM snapshot series object with a symbol specified in the inputs. Set the sorted list flag (the search is performed only in the sorted list) and obtain the object index with the same symbol in the list. Make sure to remove the temporary object and return the obtained index. If the DOM snapshot series object with a specified symbol is present in the list, the [Search()](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjsearch) method return its index, otherwise, the method returns -1.

**The method returning the specified symbol DOM series object:**

```
//+------------------------------------------------------------------+
//| Return the specified symbol DOM series object                    |
//+------------------------------------------------------------------+
CMBookSeries *CMBookSeriesCollection::GetMBookseries(const string symbol)
  {
   int index=this.IndexMBookSeries(symbol);
   return this.m_list.At(index);
  }
//+------------------------------------------------------------------+
```

Here we use the method specified above to get the index of the DOM snapshot series object in the list by symbol and return the pointer to the object in the list by the specified index. If the object with a specified symbol is not present in the list (the index is -1), the [At()](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjat) method returns NULL.

To use the functionality of working with DOMs, we need to subscribe to receive DOM change events for each symbol. This feature is already present in the symbol object class. Besides, the DOM snapshot series object features the flag indicating the need to enable/disable handling of DOM change events. In other words, even if subscription to BookEvent events is active, we may temporarily disable working with DOM on this symbol using the flag. **Use the following method to set the flag indicating the necessity to handle BookEvent events by symbol:**

```
//+------------------------------------------------------------------+
//| Set the flag of using a series                                   |
//| of a specified symbol's DOM                                      |
//+------------------------------------------------------------------+
void CMBookSeriesCollection::SetAvailableMBookSeries(const string symbol,const bool flag=true)
  {
   CMBookSeries *bookseries=this.GetMBookseries(symbol);
   if(bookseries==NULL)
      return;
   bookseries.SetAvailable(flag);
  }
//+------------------------------------------------------------------+
```

Here we receive the series object of a DOM for a specified symbol and set the flag passed to the method for it.

The default flag value is true.

If our program works with multiple symbols, while we need to manage the DOM snapshot series flags mentioned above, then **the method defining the flag value for all symbols used in the program at once can be of help:**

```
//+------------------------------------------------------------------+
//|Set the flag of using the DOM series for all symbols              |
//+------------------------------------------------------------------+
void CMBookSeriesCollection::SetAvailableMBookSeries(const bool flag=true)
  {
   for(int i=0;i<this.m_list.Total();i++)
     {
      CMBookSeries *bookseries=this.m_list.At(i);
      if(bookseries==NULL)
         continue;
      bookseries.SetAvailable(flag);
     }
  }
//+------------------------------------------------------------------+
```

Here, in the loop by the total number of DOM snapshot series objects in the collection, get the next series list and set the flag passed to the method for it. The default flag value is true.

The methods opposite to those considered above return the values of flags for working with the series list of either a specified symbol or all of them at once.

**The method returning the flag of using the DOM series of a specified symbol:**

```
//+------------------------------------------------------------------+
//|Return the flag of using the DOM series of a specified symbol     |
//+------------------------------------------------------------------+
bool CMBookSeriesCollection::IsAvailableMBookSeries(const string symbol)
  {
   CMBookSeries *bookseries=this.GetMBookseries(symbol);
   if(bookseries==NULL)
      return false;
   return bookseries.IsAvailable();
  }
//+------------------------------------------------------------------+
```

Here, we obtain the object series of DOM snapshots by a symbol passed to the method and return the flag value specified for the object.

**The method returning the flag of using the DOM series of all symbols:**

```
//+------------------------------------------------------------------+
//| Return the flag of using the DOM series of all symbols           |
//+------------------------------------------------------------------+
bool CMBookSeriesCollection::IsAvailableMBookSeries(void)
  {
   bool res=true;
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CMBookSeries *bookseries=this.m_list.At(i);
      if(bookseries==NULL)
         continue;
      res &=bookseries.IsAvailable();
     }
   return res;
  }
//+------------------------------------------------------------------+
```

The method returns true only if the flag indicating the necessity of the symbol use is activated for each of the used symbols in their DOM series. Here, in the loop by all collection series objects, we obtain the next series and add the series flag value to the variable. If the flag is set to false for at least one of the symbols, the method returns false.

**The method setting the number of snapshots in the DOM history of the specified symbol:**

```
//+------------------------------------------------------------------+
//| Set the number of snapshots in history                           |
//| of a specified symbol's DOM                                      |
//+------------------------------------------------------------------+
bool CMBookSeriesCollection::SetRequiredUsedDays(const string symbol,const uint required=0)
  {
   CMBookSeries *bookseries=this.GetMBookseries(symbol);
   if(bookseries==NULL)
      return false;
   bookseries.SetRequiredUsedDays(required);
   return true;
  }
//+------------------------------------------------------------------+
```

The method is used to limit the total number of DOM snapshots for each of the symbols and avoid spending extra memory on outdated data. Here we obtain the snapshot series list of a specified symbol and set the maximum number of snapshots in the list for it. If failed to obtain the snapshot series list, return false. If the required number is successfully set, return true.

**The method returning the number of snapshots in the DOM history of all symbols:**

```
//+------------------------------------------------------------------+
//|Set the number of snapshots in the DOM history of all symbols     |
//+------------------------------------------------------------------+
bool CMBookSeriesCollection::SetRequiredUsedDays(const uint required=0)
  {
   bool res=true;
   for(int i=0;i<this.m_list.Total();i++)
     {
      CMBookSeries *bookseries=this.m_list.At(i);
      if(bookseries==NULL)
        {
         res &=false;
         continue;
        }
      bookseries.SetRequiredUsedDays(required);
     }
   return res;
  }
//+------------------------------------------------------------------+
```

The method sets a similar number of required DOM history data for all symbols used in the program at once.

In the loop by the total number of series objects in the collection, receive the next DOM snapshot series object and set the specified amount of data for it. If failed to obtain the series, false is added to the result. Thus, if at least one series fails to set the necessary amount, the method returns false. Upon the loop completion, return the result obtained in the **res** variable.

**The method returning the DOM snapshot object of a specified symbol by index:**

```
//+------------------------------------------------------------------+
//|Return the DOM snapshot object of a specified symbol by index     |
//+------------------------------------------------------------------+
CMBookSnapshot *CMBookSeriesCollection::GetMBook(const string symbol,const int index)
  {
   CMBookSeries *bookseries=this.GetMBookseries(symbol);
   if(bookseries==NULL)
      return NULL;
   return bookseries.GetMBookByListIndex(index);
  }
//+------------------------------------------------------------------+
```

Here, we receive the DOM snapshot series object of a specified symbol and return the pointer to the DOM snapshot object by a specified index using the GetMBookByListIndex() method of the CMBookSeries class [described in the previous article](https://www.mql5.com/en/articles/9044#node04).

**The method returning the DOM snapshot object of a specified symbol by time in milliseconds:**

```
//+------------------------------------------------------------------+
//| Return the DOM snapshot object of a specified symbol             |
//| by time in milliseconds                                          |
//+------------------------------------------------------------------+
CMBookSnapshot *CMBookSeriesCollection::GetMBook(const string symbol,const long time_msc)
  {
   CMBookSeries *bookseries=this.GetMBookseries(symbol);
   if(bookseries==NULL)
      return NULL;
   return bookseries.GetMBook(time_msc);
  }
//+------------------------------------------------------------------+
```

Here, we receive the DOM snapshot series object of a specified symbol and return the pointer to the DOM snapshot object by a specified index using the GetMBook() method of the CMBookSeries class [described in the previous article](https://www.mql5.com/en/articles/9044#node04).

The distinguishing feature of the method is that, in order to get the pointer to the DOM snapshot object, we need to know its exact time in milliseconds.

**The method displaying the full description of the DOM snapshot series collection in the journal:**

```
//+------------------------------------------------------------------+
//| Display complete collection description to the journal           |
//+------------------------------------------------------------------+
void CMBookSeriesCollection::Print(void)
  {
   ::Print(CMessage::Text(MSG_MB_COLLECTION_TEXT_MBCOLLECTION),":");
   for(int i=0;i<this.m_list.Total();i++)
     {
      CMBookSeries *bookseries=this.m_list.At(i);
      if(bookseries==NULL)
         continue;
      bookseries.Print(true);
     }
  }
//+------------------------------------------------------------------+
```

First, display the header, next in the loop, obtain each subsequent DOM series object of the collection and display its full description. In the Print() method of the snapshot series object, we pass true  so that the method sets a hyphen before the description.

**The method returning the short collection list to the journal:**

```
//+------------------------------------------------------------------+
//| Display the short collection description in the journal          |
//+------------------------------------------------------------------+
void CMBookSeriesCollection::PrintShort(void)
  {
   ::Print(CMessage::Text(MSG_MB_COLLECTION_TEXT_MBCOLLECTION),":");
   for(int i=0;i<this.m_list.Total();i++)
     {
      CMBookSeries *bookseries=this.m_list.At(i);
      if(bookseries==NULL)
         continue;
      bookseries.PrintShort(true);
     }
  }
//+------------------------------------------------------------------+
```

The method is identical to the one described above except that the method of displaying a short description of the DOM series object is used to display the collection series description.

Let's improve the class of the **CEngine** library main object in \\MQL5\\Include\\DoEasy\ **Engine.mqh**.

Include the DOM snapshot series collection clas into the file:

```
//+------------------------------------------------------------------+
//|                                                       Engine.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://MQL5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://MQL5.com/en/users/artmedia70"
#property version   "1.00"
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
#include "Collections\TimeSeriesCollection.mqh"
#include "Collections\BuffersCollection.mqh"
#include "Collections\IndicatorsCollection.mqh"
#include "Collections\TickSeriesCollection.mqh"
#include "Collections\BookSeriesCollection.mqh"
#include "TradingControl.mqh"
//+------------------------------------------------------------------+
```

In the class object list, add a new object — the DOM snapshot series collection class:

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
   CBuffersCollection   m_buffers;                       // Collection of indicator buffers
   CIndicatorsCollection m_indicators;                   // Indicator collection
   CTickSeriesCollection m_tick_series;                  // Collection of tick series
   CMBookSeriesCollection m_book_series;                 // Collection of DOM series
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

Add the new BookEvent event handler. In the method setting the list of used symbols in the symbol collection, apart from passing the number of days for the tick series, add passing the maximum number of DOM snapshots:

```
//--- (1) Timer, event handler (2) NewTick, (3) Calculate, (4) BookEvent, (4) Deinit
   void                 OnTimer(SDataCalculate &data_calculate);
   void                 OnTick(SDataCalculate &data_calculate,const uint required=0);
   int                  OnCalculate(SDataCalculate &data_calculate,const uint required=0);
   bool                 OnBookEvent(const string &symbol);
   void                 OnDeinit(void);

//--- Set the list of used symbols in the symbol collection and create the collection of symbol timeseries
   bool                 SetUsedSymbols(const string &array_symbols[],const uint required_ticks=0,const uint required_books=0);
```

In the list of methods for working with the collection of tick series, add the new methods for simplifying the work with the tick series collection and write the methods for working with the DOM series collection:

```
//--- Return (1) the tick series collection, (2) the list of tick series from the tick series collection
   CTickSeriesCollection *GetTickSeriesCollection(void)                                { return &this.m_tick_series;                                  }
   CArrayObj           *GetListTickSeries(void)                                        { return this.m_tick_series.GetList();                         }

//--- Create (1) a specified tick series and (2) all tick series
   bool                 TickSeriesCreate(const string symbol,const uint required=0)    { return this.m_tick_series.CreateTickSeries(symbol,required); }
   bool                 TickSeriesCreateAll(const uint required=0)                     { return this.m_tick_series.CreateTickSeriesAll(required);     }

//--- Update (1) a tick series of a specified symbol, (2) all symbols and (3) all symbols except the current one
   void                 TickSeriesRefresh(const string symbol)                         { this.m_tick_series.Refresh(symbol);                          }
   void                 TickSeriesRefreshAll(void)                                     { this.m_tick_series.Refresh();                                }
   void                 TickSeriesRefreshAllExceptCurrent(void)                        { this.m_tick_series.RefreshExpectCurrent();                   }

//--- Return (1) the DOM series collection and (2) the list of DOM series from the DOM series collection
   CMBookSeriesCollection *GetMBookSeriesCollection(void)                              { return &this.m_book_series;                         }
   CArrayObj           *GetListMBookSeries(void)                                       { return this.m_book_series.GetList();                }

//--- Update the DOM series of a specified symbol
   void                 MBookSeriesRefresh(const string symbol,const long time_msc)    { this.m_book_series.Refresh(symbol,time_msc);        }

//--- Return the DOM series of a specified symbol
   CMBookSeries        *GetMBookSeries(const string symbol)                            { return this.m_book_series.GetMBookseries(symbol);   }
```

All these methods simply call the methods of the corresponding collections of the same name.

**Implement the OnBookEvent() handler of the library outside the class body:**

```
//+------------------------------------------------------------------+
//| BookEvent event handler                                          |
//+------------------------------------------------------------------+
bool CEngine::OnBookEvent(const string &symbol)
  {
   CSymbol *sym=this.m_symbols.GetSymbolObjByName(symbol);
   if(sym==NULL || !sym.BookdepthSubscription())
      return false;
   return this.m_book_series.Refresh(sym.Name(),sym.Time());
  }
//+------------------------------------------------------------------+
```

Since the OnBookEvent() handler is always activated during the DOM change event on a single symbol only, we can define the symbol inside the handler itself.

The method receives the name of a symbol, at which the DOM change event occurred. Next, we receive the appropriate symbol object from the symbol collection class and return the result of the method for updating the appropriate series of DOM snapshots. Besides, we specify the time in milliseconds to be set for a newly created DOM snapshot in the Refresh() method.

**Let's improve the method setting the list of used symbols in the symbol collection:**

```
//+------------------------------------------------------------------+
//| Set the list of used symbols in the symbol collection            |
//| and create the collections of timeseries, tick series            |
//| and symbol DOM series                                            |
//+------------------------------------------------------------------+
bool CEngine::SetUsedSymbols(const string &array_symbols[],const uint required_ticks=0,const uint required_books=0)
  {
   bool res=this.m_symbols.SetUsedSymbols(array_symbols);
   CArrayObj *list=this.GetListAllUsedSymbols();
   if(list==NULL)
      return false;
   res&=this.m_time_series.CreateCollection(list);
   res&=this.m_tick_series.CreateCollection(list,required_ticks);
   res&=this.m_book_series.CreateCollection(list,required_books);
   return res;
  }
//+------------------------------------------------------------------+
```

Now the method receives the number of days of tick series and the maximum number of DOM snapshots in their series.

Here we added the creation of DOM snapshot series collection for all symbols.

**This concludes the improvement of the classes for working with DOM series.**

In the future, I may get back to this topic for further improvements but now let's deal with other necessary library functionality.

[MQL5.com Signals service](https://www.mql5.com/en/signals) is a copy-trading service allowing you to automatically copy provider's deals on your trading account.

Here, the MQL5 signal object is the trading account where trading operations are monitored to be publicly transmitted — a signal source.

A signal source has its own parameters that can be obtained using the [SignalBaseGetInteger()](https://www.mql5.com/en/docs/signals/signalbasegetinteger), [SignalBaseGetDouble()](https://www.mql5.com/en/docs/signals/signalbasegetdouble) and [SignalBaseGetString()](https://www.mql5.com/en/docs/signals/signalbasegetstring) functions. Besides, there are subscription parameters — signal copying for a specific account. These parameters can be obtained using the [SignalInfoGetDouble()](https://www.mql5.com/en/docs/signals/signalinfogetdouble), [SignalInfoGetInteger()](https://www.mql5.com/en/docs/signals/signalinfogetinteger) and [SignalInfoGetString()](https://www.mql5.com/en/docs/signals/signalinfogetstring) functions.

An MQL5 signal object is to describe one signal source available in the terminal for subscription. It is possible to subscribe to only one signal from one account but we still can have the full list of all signals (MQL5 signal objects) available for subscription with the ability to sort them by any property, compare and subscribe to a selected signal. This is a very convenient and useful functionality for those willing to benefit from the MQL5.com Signals service.

### MQL5 signal object class

In \\MQL5\\Include\\DoEasy\ **Defines.mqh**, add the enumerations of integer, real and string properties of the MQL5 signal object:

```
//+------------------------------------------------------------------+
//| Data for working with MQL5.com signals                           |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| The list of possible signal events                               |
//+------------------------------------------------------------------+
#define SIGNAL_MQL5_EVENTS_NEXT_CODE  (MBOOK_ORD_EVENTS_NEXT_CODE+1)   // The code of the next event after the last signal event code
//+------------------------------------------------------------------+
//| MQL5 signal integer properties                                   |
//+------------------------------------------------------------------+
enum ENUM_SIGNAL_MQL5_PROP_INTEGER
  {
   SIGNAL_MQL5_PROP_TRADE_MODE = 0,                   // Account type
   SIGNAL_MQL5_PROP_DATE_PUBLISHED,                   // Signal publication date
   SIGNAL_MQL5_PROP_DATE_STARTED,                     // Start date of signal monitoring
   SIGNAL_MQL5_PROP_DATE_UPDATED,                     // Date of the latest update of the signal trading statistics
   SIGNAL_MQL5_PROP_ID,                               // Signal ID
   SIGNAL_MQL5_PROP_LEVERAGE,                         // Trading account leverage
   SIGNAL_MQL5_PROP_PIPS,                             // Trading result in pips
   SIGNAL_MQL5_PROP_RATING,                           // Position in the signal rating
   SIGNAL_MQL5_PROP_SUBSCRIBERS,                      // Number of subscribers
   SIGNAL_MQL5_PROP_TRADES,                           // Number of trades
   SIGNAL_MQL5_PROP_SUBSCRIPTION_STATUS,              // Status of account subscription to a signal
  };
#define SIGNAL_MQL5_PROP_INTEGER_TOTAL (11)           // Total number of integer properties
#define SIGNAL_MQL5_PROP_INTEGER_SKIP  (0)            // Number of integer DOM properties not used in sorting
//+------------------------------------------------------------------+
//| MQL5 signal real properties                                      |
//+------------------------------------------------------------------+
enum ENUM_SIGNAL_MQL5_PROP_DOUBLE
  {
   SIGNAL_MQL5_PROP_BALANCE = SIGNAL_MQL5_PROP_INTEGER_TOTAL, // Account balance
   SIGNAL_MQL5_PROP_EQUITY,                           // Account equity
   SIGNAL_MQL5_PROP_GAIN,                             // Account growth in %
   SIGNAL_MQL5_PROP_MAX_DRAWDOWN,                     // Maximum drawdown
   SIGNAL_MQL5_PROP_PRICE,                            // Signal subscription price
   SIGNAL_MQL5_PROP_ROI,                              // Signal's ROI (Return on Investment) in %
  };
#define SIGNAL_MQL5_PROP_DOUBLE_TOTAL  (6)            // Total number of real properties
#define SIGNAL_MQL5_PROP_DOUBLE_SKIP   (0)            // Number of real properties not used in sorting
//+------------------------------------------------------------------+
//| MQL5 signal string properties                                    |
//+------------------------------------------------------------------+
enum ENUM_SIGNAL_MQL5_PROP_STRING
  {
   SIGNAL_MQL5_PROP_AUTHOR_LOGIN = (SIGNAL_MQL5_PROP_INTEGER_TOTAL+SIGNAL_MQL5_PROP_DOUBLE_TOTAL), // Signal author's login
   SIGNAL_MQL5_PROP_BROKER,                           // Broker (company) name
   SIGNAL_MQL5_PROP_BROKER_SERVER,                    // Broker server
   SIGNAL_MQL5_PROP_NAME,                             // Signal name
   SIGNAL_MQL5_PROP_CURRENCY,                         // Signal account currency
  };
#define SIGNAL_MQL5_PROP_STRING_TOTAL  (5)            // Total number of string properties
//+------------------------------------------------------------------+
```

I am not going to introduce the list of possible mql5 signal events yet. Instead, I will only introduce a constant specifying the code of the next event after MQL5 signal event codes. I will most probably change this further on.

To be able to search and sort by the properties of MQL5 signals, define the enumeration of all possible sorting criteria:

```
//+------------------------------------------------------------------+
//| Possible sorting criteria of MQL5 signals                        |
//+------------------------------------------------------------------+
#define FIRST_SIGNAL_MQL5_DBL_PROP  (SIGNAL_MQL5_PROP_INTEGER_TOTAL-SIGNAL_MQL5_PROP_INTEGER_SKIP)
#define FIRST_SIGNAL_MQL5_STR_PROP  (SIGNAL_MQL5_PROP_INTEGER_TOTAL-SIGNAL_MQL5_PROP_INTEGER_SKIP+SIGNAL_MQL5_PROP_DOUBLE_TOTAL-SIGNAL_MQL5_PROP_DOUBLE_SKIP)
enum ENUM_SORT_SIGNAL_MQL5_MODE
  {
//--- Sort by integer properties
   SORT_BY_SIGNAL_MQL5_TRADE_MODE = 0,                // Sort by signal type
   SORT_BY_SIGNAL_MQL5_DATE_PUBLISHED,                // Sort by signal publication date
   SORT_BY_SIGNAL_MQL5_DATE_STARTED,                  // Sort by signal monitoring start date
   SORT_BY_SIGNAL_MQL5_DATE_UPDATED,                  // Sort by date of the latest update of the signal trading statistics
   SORT_BY_SIGNAL_MQL5_ID,                            // Sort by signal ID
   SORT_BY_SIGNAL_MQL5_LEVERAGE,                      // Sort by trading account leverage
   SORT_BY_SIGNAL_MQL5_PIPS,                          // Sort by trading result in pips
   SORT_BY_SIGNAL_MQL5_RATING,                        // Sort by a position in the signal rating
   SORT_BY_SIGNAL_MQL5_SUBSCRIBERS,                   // Sort by the number of subscribers
   SORT_BY_SIGNAL_MQL5_TRADES,                        // Sort by the number of trades
   SORT_BY_SIGNAL_MQL5_SUBSCRIPTION_STATUS,           // Sort by the status of account subscription to a signal
//--- Sort by real properties
   SORT_BY_SIGNAL_MQL5_BALANCE = FIRST_SIGNAL_MQL5_DBL_PROP, // Sort by account balance
   SORT_BY_SIGNAL_MQL5_EQUITY,                        // Sort by account equity
   SORT_BY_SIGNAL_MQL5_GAIN,                          // Sort by account growth in %
   SORT_BY_SIGNAL_MQL5_MAX_DRAWDOWN,                  // Sort by maximum drawdown
   SORT_BY_SIGNAL_MQL5_PRICE,                         // Sort by signal subscription price
   SORT_BY_SIGNAL_MQL5_ROI,                           // Sort by ROI
//--- Sort by string properties
   SORT_BY_SIGNAL_MQL5_AUTHOR_LOGIN = FIRST_SIGNAL_MQL5_STR_PROP, // Sort by signal author login
   SORT_BY_SIGNAL_MQL5_BROKER,                        // Sort by Broker (company) name
   SORT_BY_SIGNAL_MQL5_BROKER_SERVER,                 // Sort by broker server
   SORT_BY_SIGNAL_MQL5_NAME,                          // Sort by signal name
   SORT_BY_SIGNAL_MQL5_CURRENCY,                      // Sort by signal account currency
  };
//+------------------------------------------------------------------+
```

This is a set of the most necessary parameters for creating a new class of the new library object.

In \\MQL5\\Include\\DoEasy\ **Objects\**, create a new folder **MQLSignalBase\** featuring a new file **MQLSignal.mqh** of the CMQLSignal class.

The class of the basic object of all CBaseObj library objects should be used as the base class:

```
//+------------------------------------------------------------------+
//|                                                    MQLSignal.mqh |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://MQL5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://MQL5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\..\Objects\BaseObj.mqh"
//+------------------------------------------------------------------+
//| Abstract MQL5 signal class                                       |
//+------------------------------------------------------------------+
class CMQLSignal : public CBaseObj
  {
  }
```

Let's consider the class composition and implementation of its methods:

```
//+------------------------------------------------------------------+
//|                                                    MQLSignal.mqh |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://MQL5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://MQL5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\..\Objects\BaseObj.mqh"
//+------------------------------------------------------------------+
//| Abstract MQL5 signal class                                       |
//+------------------------------------------------------------------+
class CMQLSignal : public CBaseObj
  {
private:
   long              m_long_prop[SIGNAL_MQL5_PROP_INTEGER_TOTAL];       // Integer properties
   double            m_double_prop[SIGNAL_MQL5_PROP_DOUBLE_TOTAL];      // Real properties
   string            m_string_prop[SIGNAL_MQL5_PROP_STRING_TOTAL];      // String properties

//--- Return the index of the array the (1) double and (2) string properties are actually located at
   int               IndexProp(ENUM_SIGNAL_MQL5_PROP_DOUBLE property)   const { return(int)property-SIGNAL_MQL5_PROP_INTEGER_TOTAL;                               }
   int               IndexProp(ENUM_SIGNAL_MQL5_PROP_STRING property)   const { return(int)property-SIGNAL_MQL5_PROP_INTEGER_TOTAL-SIGNAL_MQL5_PROP_DOUBLE_TOTAL; }

public:
//--- Set object's (1) integer, (2) real and (3) string properties
   void              SetProperty(ENUM_SIGNAL_MQL5_PROP_INTEGER property,long value)    { this.m_long_prop[property]=value;                      }
   void              SetProperty(ENUM_SIGNAL_MQL5_PROP_DOUBLE property,double value)   { this.m_double_prop[this.IndexProp(property)]=value;    }
   void              SetProperty(ENUM_SIGNAL_MQL5_PROP_STRING property,string value)   { this.m_string_prop[this.IndexProp(property)]=value;    }
//--- Return object’s (1) integer, (2) real and (3) string property from the properties array
   long              GetProperty(ENUM_SIGNAL_MQL5_PROP_INTEGER property)         const { return this.m_long_prop[property];                     }
   double            GetProperty(ENUM_SIGNAL_MQL5_PROP_DOUBLE property)          const { return this.m_double_prop[this.IndexProp(property)];   }
   string            GetProperty(ENUM_SIGNAL_MQL5_PROP_STRING property)          const { return this.m_string_prop[this.IndexProp(property)];   }
//--- Return itself
   CMQLSignal       *GetObject(void)                                                   { return &this;}

//--- Return the flag of the object supporting this property
   virtual bool      SupportProperty(ENUM_SIGNAL_MQL5_PROP_INTEGER property)           { return true; }
   virtual bool      SupportProperty(ENUM_SIGNAL_MQL5_PROP_DOUBLE property)            { return true; }
   virtual bool      SupportProperty(ENUM_SIGNAL_MQL5_PROP_STRING property)            { return true; }

//--- Get description of (1) integer, (2) real and (3) string properties
   string            GetPropertyDescription(ENUM_SIGNAL_MQL5_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_SIGNAL_MQL5_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_SIGNAL_MQL5_PROP_STRING property);

//--- Display the description of object properties in the journal (full_prop=true - all properties, false - supported ones only)
   void              Print(const bool full_prop=false);
//--- Display a short description of the object in the journal
   virtual void      PrintShort(void);
//--- Return the object short name
   virtual string    Header(const bool shrt=false);

//--- Compare CMQLSignal objects by a specified property (to sort the list by an MQL5 signal object)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CMQLSignal objects by all properties (to search for equal MQL5 signal objects)
   bool              IsEqual(CMQLSignal* compared_obj) const;

//--- Constructors
                     CMQLSignal(){;}
                     CMQLSignal(const long signal_id);

public:
//+------------------------------------------------------------------+
//| Methods of simplified access to MQL5 signal object               |
//+------------------------------------------------------------------+
//--- Returns the dates of (1) publication, (2) monitoring start, (3) trading statistics last update,
//--- (4) ID, (5) trading account leverage, (6) trading result in pips, (7) position in the signal rating,
//--- (8) number of subscribers, (9) number of trades, (10) account type, (11) flag of the current account subscription to a signal
   datetime          DatePublished(void)     const { return (datetime)this.GetProperty(SIGNAL_MQL5_PROP_DATE_PUBLISHED);         }
   datetime          DateStarted(void)       const { return (datetime)this.GetProperty(SIGNAL_MQL5_PROP_DATE_STARTED);           }
   datetime          DateUpdated(void)       const { return (datetime)this.GetProperty(SIGNAL_MQL5_PROP_DATE_UPDATED);           }
   long              ID(void)                const { return this.GetProperty(SIGNAL_MQL5_PROP_ID);                               }
   long              Leverage(void)          const { return this.GetProperty(SIGNAL_MQL5_PROP_LEVERAGE);                         }
   long              Pips(void)              const { return this.GetProperty(SIGNAL_MQL5_PROP_PIPS);                             }
   long              Rating(void)            const { return this.GetProperty(SIGNAL_MQL5_PROP_RATING);                           }
   long              Subscribers(void)       const { return this.GetProperty(SIGNAL_MQL5_PROP_SUBSCRIBERS);                      }
   long              Trades(void)            const { return this.GetProperty(SIGNAL_MQL5_PROP_TRADES);                           }
   long              TradeMode(void)         const { return (long)this.GetProperty(SIGNAL_MQL5_PROP_TRADE_MODE);                 }
   bool              SubscriptionStatus(void)const { return (bool)this.GetProperty(SIGNAL_MQL5_PROP_SUBSCRIPTION_STATUS);        }
//--- Return (1) the account balance, (2) account funds, (3) account growth in %,
//--- (4) maximum drawdown, (5) subscription price and (6) signal ROI value (Return on Investment) in %
   double            Balance(void)           const { return this.GetProperty(SIGNAL_MQL5_PROP_BALANCE);                          }
   double            Equity(void)            const { return this.GetProperty(SIGNAL_MQL5_PROP_EQUITY);                           }
   double            Gain(void)              const { return this.GetProperty(SIGNAL_MQL5_PROP_GAIN);                             }
   double            MaxDrawdown(void)       const { return this.GetProperty(SIGNAL_MQL5_PROP_MAX_DRAWDOWN);                     }
   double            Price(void)             const { return this.GetProperty(SIGNAL_MQL5_PROP_PRICE);                            }
   double            ROI(void)               const { return this.GetProperty(SIGNAL_MQL5_PROP_ROI);                              }
//--- Return (1) signal author's login, (2) broker (company) name,
//--- (3) broker server, (4) signal name and (5) signal account currency
   string            AuthorLogin(void)       const { return this.GetProperty(SIGNAL_MQL5_PROP_AUTHOR_LOGIN);                     }
   string            Broker(void)            const { return this.GetProperty(SIGNAL_MQL5_PROP_BROKER);                           }
   string            BrokerServer(void)      const { return this.GetProperty(SIGNAL_MQL5_PROP_BROKER_SERVER);                    }
   string            Name(void)              const { return this.GetProperty(SIGNAL_MQL5_PROP_NAME);                             }
   string            Currency(void)          const { return this.GetProperty(SIGNAL_MQL5_PROP_CURRENCY);                         }

//--- Set the dates of (1) publication, (2) monitoring start, (3) trading statistics last update,
//--- (4) ID, (5) trading account leverage, (6) trading result in pips, (7) position in the signal rating,
//--- (8) number of subscribers, (9) number of trades, (10) account type, (11) flag of the current account subscription to a signal
   void              SetDatePublished(const datetime date)  { this.SetProperty(SIGNAL_MQL5_PROP_DATE_PUBLISHED,date);            }
   void              SetDateStarted(const datetime date)    { this.SetProperty(SIGNAL_MQL5_PROP_DATE_STARTED,date);              }
   void              SetDateUpdated(const datetime date)    { this.SetProperty(SIGNAL_MQL5_PROP_DATE_UPDATED,date);              }
   void              SetID(const long id)                   { this.SetProperty(SIGNAL_MQL5_PROP_ID,id);                          }
   void              SetLeverage(const long value)          { this.SetProperty(SIGNAL_MQL5_PROP_LEVERAGE,value);                 }
   void              SetPips(const long value)              { this.SetProperty(SIGNAL_MQL5_PROP_PIPS,value);                     }
   void              SetRating(const long value)            { this.SetProperty(SIGNAL_MQL5_PROP_RATING,value);                   }
   void              SetSubscribers(const long value)       { this.SetProperty(SIGNAL_MQL5_PROP_SUBSCRIBERS,value);              }
   void              SetTrades(const long value)            { this.SetProperty(SIGNAL_MQL5_PROP_TRADES,value);                   }
   void              SetTradeMode(const long mode)          { this.SetProperty(SIGNAL_MQL5_PROP_TRADE_MODE,mode);                }
   void              SetSubscriptionStatus(const bool flag) { this.SetProperty(SIGNAL_MQL5_PROP_SUBSCRIPTION_STATUS,flag);       }
//--- Set (1) the account balance, (2) account funds, (3) account growth in %,
//--- (4) maximum drawdown, (5) subscription price and (6) signal ROI value (Return on Investment) in %
   void              SetBalance(const double value)         { this.SetProperty(SIGNAL_MQL5_PROP_BALANCE,value);                  }
   void              SetEquity(const double value)          { this.SetProperty(SIGNAL_MQL5_PROP_EQUITY,value);                   }
   void              SetGain(const double value)            { this.SetProperty(SIGNAL_MQL5_PROP_GAIN,value);                     }
   void              SetMaxDrawdown(const double value)     { this.SetProperty(SIGNAL_MQL5_PROP_MAX_DRAWDOWN,value);             }
   void              SetPrice(const double value)           { this.SetProperty(SIGNAL_MQL5_PROP_PRICE,value);                    }
   void              SetROI(const double value)             { this.SetProperty(SIGNAL_MQL5_PROP_ROI,value);                      }
//--- Set (1) signal author's login, (2) broker (company) name,
//--- (3) broker server, (4) signal name and (5) signal account currency
   void              SetAuthorLogin(const string value)     { this.SetProperty(SIGNAL_MQL5_PROP_AUTHOR_LOGIN,value);             }
   void              SetBroker(const string value)          { this.SetProperty(SIGNAL_MQL5_PROP_BROKER,value);                   }
   void              SetBrokerServer(const string value)    { this.SetProperty(SIGNAL_MQL5_PROP_BROKER_SERVER,value);            }
   void              SetName(const string value)            { this.SetProperty(SIGNAL_MQL5_PROP_NAME,value);                     }
   void              SetCurrency(const string value)        { this.SetProperty(SIGNAL_MQL5_PROP_CURRENCY,value);                 }

//--- Return the account type name
   string            TradeModeDescription(void);

  };
//+------------------------------------------------------------------+
```

The arrays for storing integer, real and string object properties and methods returning the real index of the object real and string properties are declared in the private section.

In the public section of the class, we can see the standard methods of setting and receiving object properties, searching and sorting, displaying object descriptions and class constructors.

The public section also features the methods for a simplified access to object properties — the methods simply have self-explanatory names so that library users are able to access any property without the need to memorize the names of all object enumerations.

The composition of the object classes was considered in details [in the first part of the library description](https://www.mql5.com/en/articles/5654). Let's have a look at the implementation of the class methods.

**The parametric class constructor** receives the signal ID. All object properties are then filled with values obtained from the appropriate functions (considering that this signal is selected):

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CMQLSignal::CMQLSignal(const long signal_id)
  {
   this.m_long_prop[SIGNAL_MQL5_PROP_ID]                             = signal_id;
   this.m_long_prop[SIGNAL_MQL5_PROP_SUBSCRIPTION_STATUS]            = false;
   this.m_long_prop[SIGNAL_MQL5_PROP_TRADE_MODE]                     = ::SignalBaseGetInteger(SIGNAL_BASE_TRADE_MODE);
   this.m_long_prop[SIGNAL_MQL5_PROP_DATE_PUBLISHED]                 = ::SignalBaseGetInteger(SIGNAL_BASE_DATE_PUBLISHED);
   this.m_long_prop[SIGNAL_MQL5_PROP_DATE_STARTED]                   = ::SignalBaseGetInteger(SIGNAL_BASE_DATE_STARTED);
   this.m_long_prop[SIGNAL_MQL5_PROP_DATE_UPDATED]                   = ::SignalBaseGetInteger(SIGNAL_BASE_DATE_UPDATED);
   this.m_long_prop[SIGNAL_MQL5_PROP_LEVERAGE]                       = ::SignalBaseGetInteger(SIGNAL_BASE_LEVERAGE);
   this.m_long_prop[SIGNAL_MQL5_PROP_PIPS]                           = ::SignalBaseGetInteger(SIGNAL_BASE_PIPS);
   this.m_long_prop[SIGNAL_MQL5_PROP_RATING]                         = ::SignalBaseGetInteger(SIGNAL_BASE_RATING);
   this.m_long_prop[SIGNAL_MQL5_PROP_SUBSCRIBERS]                    = ::SignalBaseGetInteger(SIGNAL_BASE_SUBSCRIBERS);
   this.m_long_prop[SIGNAL_MQL5_PROP_TRADES]                         = ::SignalBaseGetInteger(SIGNAL_BASE_TRADES);

   this.m_double_prop[this.IndexProp(SIGNAL_MQL5_PROP_BALANCE)]      = ::SignalBaseGetDouble(SIGNAL_BASE_BALANCE);
   this.m_double_prop[this.IndexProp(SIGNAL_MQL5_PROP_EQUITY)]       = ::SignalBaseGetDouble(SIGNAL_BASE_EQUITY);
   this.m_double_prop[this.IndexProp(SIGNAL_MQL5_PROP_GAIN)]         = ::SignalBaseGetDouble(SIGNAL_BASE_GAIN);
   this.m_double_prop[this.IndexProp(SIGNAL_MQL5_PROP_MAX_DRAWDOWN)] = ::SignalBaseGetDouble(SIGNAL_BASE_MAX_DRAWDOWN);
   this.m_double_prop[this.IndexProp(SIGNAL_MQL5_PROP_PRICE)]        = ::SignalBaseGetDouble(SIGNAL_BASE_PRICE);
   this.m_double_prop[this.IndexProp(SIGNAL_MQL5_PROP_ROI)]          = ::SignalBaseGetDouble(SIGNAL_BASE_ROI);

   this.m_string_prop[this.IndexProp(SIGNAL_MQL5_PROP_AUTHOR_LOGIN)] = ::SignalBaseGetString(SIGNAL_BASE_AUTHOR_LOGIN);
   this.m_string_prop[this.IndexProp(SIGNAL_MQL5_PROP_BROKER)]       = ::SignalBaseGetString(SIGNAL_BASE_BROKER);
   this.m_string_prop[this.IndexProp(SIGNAL_MQL5_PROP_BROKER_SERVER)]= ::SignalBaseGetString(SIGNAL_BASE_BROKER_SERVER);
   this.m_string_prop[this.IndexProp(SIGNAL_MQL5_PROP_NAME)]         = ::SignalBaseGetString(SIGNAL_BASE_NAME);
   this.m_string_prop[this.IndexProp(SIGNAL_MQL5_PROP_CURRENCY)]     = ::SignalBaseGetString(SIGNAL_BASE_CURRENCY);
  }
//+------------------------------------------------------------------+
```

The signal subscription flag is set to false — subscription to a selected signal is performed from another class.

**The method for comparing MQL5 signal objects by a specified property:**

```
//+----------------------------------------------------------------------+
//| Compare CMQLSignal objects with each other by the specified property |
//+----------------------------------------------------------------------+
int CMQLSignal::Compare(const CObject *node,const int mode=0) const
  {
   const CMQLSignal *obj_compared=node;
//--- compare integer properties of two objects
   if(mode<SIGNAL_MQL5_PROP_INTEGER_TOTAL)
     {
      long value_compared=obj_compared.GetProperty((ENUM_SIGNAL_MQL5_PROP_INTEGER)mode);
      long value_current=this.GetProperty((ENUM_SIGNAL_MQL5_PROP_INTEGER)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare real properties of two objects
   else if(mode<SIGNAL_MQL5_PROP_DOUBLE_TOTAL+SIGNAL_MQL5_PROP_INTEGER_TOTAL)
     {
      double value_compared=obj_compared.GetProperty((ENUM_SIGNAL_MQL5_PROP_DOUBLE)mode);
      double value_current=this.GetProperty((ENUM_SIGNAL_MQL5_PROP_DOUBLE)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare string properties of two objects
   else if(mode<SIGNAL_MQL5_PROP_DOUBLE_TOTAL+SIGNAL_MQL5_PROP_INTEGER_TOTAL+SIGNAL_MQL5_PROP_STRING_TOTAL)
     {
      string value_compared=obj_compared.GetProperty((ENUM_SIGNAL_MQL5_PROP_STRING)mode);
      string value_current=this.GetProperty((ENUM_SIGNAL_MQL5_PROP_STRING)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
   return 0;
  }
//+------------------------------------------------------------------+
```

In brief, the method receives an object for comparison as well as a property, by which two objects are to be compared. Integer, real or string properties of two objects are compared depending on the passed property. The comparison result of -1, 1 or 0 (less than, greater than, equal) is returned.

**The method for comparing MQL5 signal objects by all properties:**

```
//+------------------------------------------------------------------+
//| Compare CMQLSignal objects by all properties                     |
//+------------------------------------------------------------------+
bool CMQLSignal::IsEqual(CMQLSignal *compared_obj) const
  {
   int beg=0, end=SIGNAL_MQL5_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_SIGNAL_MQL5_PROP_INTEGER prop=(ENUM_SIGNAL_MQL5_PROP_INTEGER)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=SIGNAL_MQL5_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_SIGNAL_MQL5_PROP_DOUBLE prop=(ENUM_SIGNAL_MQL5_PROP_DOUBLE)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=SIGNAL_MQL5_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_SIGNAL_MQL5_PROP_STRING prop=(ENUM_SIGNAL_MQL5_PROP_STRING)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

In three loops by three object properties, compare each subsequent property of two objects and, if the properties of two objects are not equal, return false. After comparing all properties of two objects in three arrays, return true  if non-equal properties are not found.

**The methods returning the description of the specified integer, real and string object property:**

```
//+------------------------------------------------------------------+
//| Return description of object's integer property                  |
//+------------------------------------------------------------------+
string CMQLSignal::GetPropertyDescription(ENUM_SIGNAL_MQL5_PROP_INTEGER property)
  {
   return
     (
      property==SIGNAL_MQL5_PROP_TRADE_MODE     ?  CMessage::Text(MSG_SIGNAL_MQL5_TRADE_MODE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.TradeModeDescription()
         )  :
      property==SIGNAL_MQL5_PROP_DATE_PUBLISHED ?  CMessage::Text(MSG_SIGNAL_MQL5_DATE_PUBLISHED)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::TimeToString(this.DatePublished())
         )  :
      property==SIGNAL_MQL5_PROP_DATE_STARTED   ?  CMessage::Text(MSG_SIGNAL_MQL5_DATE_STARTED)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::TimeToString(this.DateStarted())
         )  :
      property==SIGNAL_MQL5_PROP_DATE_UPDATED   ?  CMessage::Text(MSG_SIGNAL_MQL5_DATE_UPDATED)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::TimeToString(this.DateUpdated())
         )  :
      property==SIGNAL_MQL5_PROP_ID             ?  CMessage::Text(MSG_SIGNAL_MQL5_ID)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SIGNAL_MQL5_PROP_LEVERAGE       ?  CMessage::Text(MSG_SIGNAL_MQL5_LEVERAGE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SIGNAL_MQL5_PROP_PIPS           ?  CMessage::Text(MSG_SIGNAL_MQL5_PIPS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SIGNAL_MQL5_PROP_RATING         ?  CMessage::Text(MSG_SIGNAL_MQL5_RATING)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SIGNAL_MQL5_PROP_SUBSCRIBERS    ?  CMessage::Text(MSG_SIGNAL_MQL5_SUBSCRIBERS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SIGNAL_MQL5_PROP_TRADES        ?  CMessage::Text(MSG_SIGNAL_MQL5_TRADES)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SIGNAL_MQL5_PROP_SUBSCRIPTION_STATUS  ?  CMessage::Text(MSG_SIGNAL_MQL5_SUBSCRIPTION_STATUS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return description of object's real property                     |
//+------------------------------------------------------------------+
string CMQLSignal::GetPropertyDescription(ENUM_SIGNAL_MQL5_PROP_DOUBLE property)
  {
   return
     (
      property==SIGNAL_MQL5_PROP_BALANCE         ?  CMessage::Text(MSG_ACC_PROP_BALANCE)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),2)
         )  :
      property==SIGNAL_MQL5_PROP_EQUITY   ?  CMessage::Text(MSG_SIGNAL_MQL5_EQUITY)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),2)
         )  :

      property==SIGNAL_MQL5_PROP_GAIN   ?  CMessage::Text(MSG_SIGNAL_MQL5_GAIN)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),2)
         )  :
      property==SIGNAL_MQL5_PROP_MAX_DRAWDOWN   ?  CMessage::Text(MSG_SIGNAL_MQL5_MAX_DRAWDOWN)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),2)
         )  :
      property==SIGNAL_MQL5_PROP_PRICE   ?  CMessage::Text(MSG_SIGNAL_MQL5_PRICE)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),2)
         )  :
      property==SIGNAL_MQL5_PROP_ROI   ?  CMessage::Text(MSG_SIGNAL_MQL5_ROI)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),2)
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return description of object's string property                   |
//+------------------------------------------------------------------+
string CMQLSignal::GetPropertyDescription(ENUM_SIGNAL_MQL5_PROP_STRING property)
  {
   return
     (
      property==SIGNAL_MQL5_PROP_AUTHOR_LOGIN   ?  CMessage::Text(MSG_SIGNAL_MQL5_AUTHOR_LOGIN)+": \""+this.GetProperty(property)+"\""    :
      property==SIGNAL_MQL5_PROP_BROKER         ?  CMessage::Text(MSG_SIGNAL_MQL5_BROKER)+": \""+this.GetProperty(property)+"\""          :
      property==SIGNAL_MQL5_PROP_BROKER_SERVER  ?  CMessage::Text(MSG_SIGNAL_MQL5_BROKER_SERVER)+": \""+this.GetProperty(property)+"\""   :
      property==SIGNAL_MQL5_PROP_NAME           ?  CMessage::Text(MSG_SIGNAL_MQL5_NAME)+": \""+this.GetProperty(property)+"\""            :
      property==SIGNAL_MQL5_PROP_CURRENCY       ?  CMessage::Text(MSG_SIGNAL_MQL5_CURRENCY)+": \""+this.GetProperty(property)+"\""        :
      ""
     );
  }
//+------------------------------------------------------------------+
```

Depending on the property passed to the method, the string with its description and value is created and returned.

**The method displaying all MQL5 signal object properties in the journal:**

```
//+------------------------------------------------------------------+
//| Display object properties in the journal                         |
//+------------------------------------------------------------------+
void CMQLSignal::Print(const bool full_prop=false)
  {
   ::Print("============= ",CMessage::Text(MSG_LIB_PARAMS_LIST_BEG)," (",this.Header(),") =============");
   int beg=0, end=SIGNAL_MQL5_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_SIGNAL_MQL5_PROP_INTEGER prop=(ENUM_SIGNAL_MQL5_PROP_INTEGER)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=SIGNAL_MQL5_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_SIGNAL_MQL5_PROP_DOUBLE prop=(ENUM_SIGNAL_MQL5_PROP_DOUBLE)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=SIGNAL_MQL5_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_SIGNAL_MQL5_PROP_STRING prop=(ENUM_SIGNAL_MQL5_PROP_STRING)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("============= ",CMessage::Text(MSG_LIB_PARAMS_LIST_END)," (",this.Header(),") =============\n");
  }
//+------------------------------------------------------------------+
```

A description of each subsequent object property is displayed in three loops by integer, real and string properties using the appropriate GetPropertyDescription() method.

**The method returning the short name of the MQL5 signal object:**

```
//+------------------------------------------------------------------+
//| Return the object short name                                     |
//+------------------------------------------------------------------+
string CMQLSignal::Header(const bool shrt=false)
  {
   return(shrt ? CMessage::Text(MSG_SIGNAL_MQL5_TEXT_SIGNAL) : CMessage::Text(MSG_SIGNAL_MQL5_TEXT_SIGNAL_MQL5));
  }
//+------------------------------------------------------------------+
```

The method receives the flag specifying either short or long description. The long description is shown by default:

```
The signal of the MQL5.com Signal service
```

If the flag is true, a short description is displayed:

```
Signal
```

**The method displaying the short object description in the journal:**

```
//+------------------------------------------------------------------+
//| Display a short description of the object in the journal         |
//+------------------------------------------------------------------+
void CMQLSignal::PrintShort(void)
  {
   ::Print
     (
      this.Header(true),
      " \"",this.Name(),"\". ",
      CMessage::Text(MSG_SIGNAL_MQL5_AUTHOR_LOGIN),": ",this.AuthorLogin(),
      ", ID ",this.ID(),
      ", ",CMessage::Text(MSG_SIGNAL_MQL5_TEXT_GAIN),": ",::DoubleToString(this.Gain(),2),
      ", ",CMessage::Text(MSG_SIGNAL_MQL5_TEXT_DRAWDOWN),": ",::DoubleToString(this.MaxDrawdown(),2),
      ", ",CMessage::Text(MSG_LIB_TEXT_REQUEST_PRICE),": ",::DoubleToString(this.Price(),2)
     );
  }
//+------------------------------------------------------------------+
```

The short Signal header is created first. Some parameters displayed in the terminal journal are added to it.

For example, it may look like this:

```
Signal "DemoForArticle". Author login: login, ID XXXXXX, Growth: XX.XX, Drawdown: XX.XX, Price: XX.XX
```

**The method returning the account type name:**

```
//+------------------------------------------------------------------+
//| Return the account type name                                     |
//+------------------------------------------------------------------+
string CMQLSignal::TradeModeDescription(void)
  {
   return
     (
      this.TradeMode()==0 ? CMessage::Text(MSG_ACC_TRADE_MODE_REAL)     :
      this.TradeMode()==1 ? CMessage::Text(MSG_ACC_TRADE_MODE_DEMO)     :
      this.TradeMode()==2 ? CMessage::Text(MSG_ACC_TRADE_MODE_CONTEST)  :
      CMessage::Text(MSG_ACC_TRADE_MODE_UNKNOWN)
     );
  }
//+------------------------------------------------------------------+
```

Depending on a signal source account, a string with an account type (real, demo or contest) is returned.

**This completes the creation of the MQL5 signal object.**

### Test

To perform the test, let's use [the EA from the previous article](https://www.mql5.com/en/articles/9044#node05) and save it to \\MQL5\\Experts\\TestDoEasy\ **Part65\** as **TestDoEasyPart65.mq5**.

The DOM snapshot series collection class is now available from the main library object, while the MQL5 signal object is not connected to CEngine yet. Therefore, let's replace the inclusion string of the DOM snapshot series object class

```
#include <DoEasy\Objects\Book\MBookSeries.mqh>
```

with the one for the inclusion of the MQL5 signal object:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart65.mq5 |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://MQL5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://MQL5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
#include <DoEasy\Objects\MQLSignalBase\MQLSignal.mqh>
//--- enums
```

Remove the declaration of the DOM snapshot series object from the list of EA global variables:

```
//--- global variables
CEngine        engine;
SDataButt      butt_data[TOTAL_BUTT];
string         prefix;
double         lot;
double         withdrawal=(InpWithdrawal<0.1 ? 0.1 : InpWithdrawal);
ushort         magic_number;
uint           stoploss;
uint           takeprofit;
uint           distance_pending;
uint           distance_stoplimit;
uint           distance_pending_request;
uint           bars_delay_pending_request;
uint           slippage;
bool           trailing_on;
bool           pressed_pending_buy;
bool           pressed_pending_buy_limit;
bool           pressed_pending_buy_stop;
bool           pressed_pending_buy_stoplimit;
bool           pressed_pending_close_buy;
bool           pressed_pending_close_buy2;
bool           pressed_pending_close_buy_by_sell;
bool           pressed_pending_sell;
bool           pressed_pending_sell_limit;
bool           pressed_pending_sell_stop;
bool           pressed_pending_sell_stoplimit;
bool           pressed_pending_close_sell;
bool           pressed_pending_close_sell2;
bool           pressed_pending_close_sell_by_buy;
bool           pressed_pending_delete_all;
bool           pressed_pending_close_all;
bool           pressed_pending_sl;
bool           pressed_pending_tp;
double         trailing_stop;
double         trailing_step;
uint           trailing_start;
uint           stoploss_to_modify;
uint           takeprofit_to_modify;
int            used_symbols_mode;
string         array_used_symbols[];
string         array_used_periods[];
bool           testing;
uchar          group1;
uchar          group2;
double         g_point;
int            g_digits;
//---
CMBookSeries   book_series;
//+------------------------------------------------------------------+
```

[MQL5 help features an example](https://www.mql5.com/en/docs/signals/signalbaseselect) of receiving the list of profitable free signals with non-zero number of subscribers:

```
void OnStart()
  {
//--- request the total number of signals in the signal database
   int total=SignalBaseTotal();
//--- loop through all signals
   for(int i=0;i<total;i++)
     {
      //--- select a signal for further operation
      if(SignalBaseSelect(i))
        {
         //--- receive signal properties
         long   id    =SignalBaseGetInteger(SIGNAL_BASE_ID);          // signal ID
         long   pips  =SignalBaseGetInteger(SIGNAL_BASE_PIPS);        // trading result in pips
         long   subscr=SignalBaseGetInteger(SIGNAL_BASE_SUBSCRIBERS); // number of subscribers
         string name  =SignalBaseGetString(SIGNAL_BASE_NAME);         // signal name
         double price =SignalBaseGetDouble(SIGNAL_BASE_PRICE);        // signal subscription price
         string curr  =SignalBaseGetString(SIGNAL_BASE_CURRENCY);     // signal currency
         //--- display all profitable free signals with a non-zero number of subscribers
         if(price==0.0 && pips>0 && subscr>0)
            PrintFormat("id=%d, name=\"%s\", currency=%s, pips=%d, subscribers=%d",id,name,curr,pips,subscr);
        }
      else PrintFormat("Signal selection error. Error code=%d",GetLastError());
     }
  }
```

Let's do the same with the help of the new MQL5 signal object. Write the following code block in the EA's OnInit() handler:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Calling the function displays the list of enumeration constants in the journal
//--- (the list is set in the strings 22 and 25 of the DELib.mqh file) for checking the constants validity
   //EnumNumbersTest();

//--- Set EA global variables
   prefix=MQLInfoString(MQL_PROGRAM_NAME)+"_";
   testing=engine.IsTester();
   for(int i=0;i<TOTAL_BUTT;i++)
     {
      butt_data[i].name=prefix+EnumToString((ENUM_BUTTONS)i);
      butt_data[i].text=EnumToButtText((ENUM_BUTTONS)i);
     }
   lot=NormalizeLot(Symbol(),fmax(InpLots,MinimumLots(Symbol())*2.0));
   magic_number=InpMagic;
   stoploss=InpStopLoss;
   takeprofit=InpTakeProfit;
   distance_pending=InpDistance;
   distance_stoplimit=InpDistanceSL;
   slippage=InpSlippage;
   trailing_stop=InpTrailingStop*Point();
   trailing_step=InpTrailingStep*Point();
   trailing_start=InpTrailingStart;
   stoploss_to_modify=InpStopLossModify;
   takeprofit_to_modify=InpTakeProfitModify;
   distance_pending_request=(InpDistancePReq<5 ? 5 : InpDistancePReq);
   bars_delay_pending_request=(InpBarsDelayPReq<1 ? 1 : InpBarsDelayPReq);
   g_point=SymbolInfoDouble(NULL,SYMBOL_POINT);
   g_digits=(int)SymbolInfoInteger(NULL,SYMBOL_DIGITS);
//--- Initialize random group numbers
   group1=0;
   group2=0;
   srand(GetTickCount());

//--- Initialize DoEasy library
   OnInitDoEasy();

//--- Check and remove remaining EA graphical objects
   if(IsPresentObectByPrefix(prefix))
      ObjectsDeleteAll(0,prefix);

//--- Create the button panel
   if(!CreateButtons(InpButtShiftX,InpButtShiftY))
      return INIT_FAILED;
//--- Set trailing activation button status
   ButtonState(butt_data[TOTAL_BUTT-1].name,trailing_on);
//--- Reset states of the buttons for working using pending requests
   for(int i=0;i<14;i++)
     {
      ButtonState(butt_data[i].name+"_PRICE",false);
      ButtonState(butt_data[i].name+"_TIME",false);
     }

//--- Check playing a standard sound by macro substitution and a custom sound by description
   engine.PlaySoundByDescription(SND_OK);
//--- Wait for 600 milliseconds
   engine.Pause(600);
   engine.PlaySoundByDescription(TextByLanguage("Звук упавшей монетки 2","Falling coin 2"));

   CArrayObj *list=new CArrayObj();
   if(list!=NULL)
     {
      //--- request the total number of signals in the signal database
      int total=SignalBaseTotal();
      //--- loop through all signals
      for(int i=0;i<total;i++)
        {
         //--- select a signal for further operation
         if(!SignalBaseSelect(i))
            continue;
         long id=SignalBaseGetInteger(SIGNAL_BASE_ID);
         CMQLSignal *signal=new CMQLSignal(id);
         if(signal==NULL)
            continue;
         if(!list.Add(signal))
           {
            delete signal;
            continue;
           }
        }
      //--- display all profitable free signals with a non-zero number of subscribers
      Print("");
      static bool done=false;
      for(int i=0;i<list.Total();i++)
        {
         CMQLSignal *signal=list.At(i);
         if(signal==NULL)
            continue;
         if(signal.Price()>0 || signal.Subscribers()==0)
            continue;
         //--- The very first suitable signal is fully displayed in the journal
         if(!done)
           {
            signal.Print();
            done=true;
           }
         //--- Short descriptions are displayed for the rest
         else
            signal.PrintShort();
        }
      delete list;
     }

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

At first glance, the resulting amount of code is larger than the code from the help. But this is a temporary phenomenon... As soon as the signal collection class is ready, it will be much more concise. In the meantime, I will use the code to check whether the MQL5 signal object works correctly.

Since we now have access to the DOM snapshot series collection class from the main library object, the OnBookEvent() handler has undergone some changes. Now it looks as follows:

```
//+------------------------------------------------------------------+
//| OnBookEvent function                                             |
//+------------------------------------------------------------------+
void OnBookEvent(const string& symbol)
  {
   static bool first=true;
   //--- Leave if failed to update the symbol snapshot series
   if(!engine.OnBookEvent(symbol))
      return;
   //--- Work by the current symbol
   if(symbol==Symbol())
     {
      //--- Get the DOM snapshot series of the current symbol
      CMBookSeries *book_series=engine.GetMBookSeries(symbol);
      if(book_series==NULL)
         return;
      //--- Get the last DOM snapshot object from the DOM snapshot series object
      CMBookSnapshot *book=book_series.GetLastMBook();
      if(book==NULL)
         return;
      //--- Get the very first and last DOM order objects from the DOM snapshot object
      CMarketBookOrd *ord_0=book.GetMBookByListIndex(0);
      CMarketBookOrd *ord_N=book.GetMBookByListIndex(book.DataTotal()-1);
      if(ord_0==NULL || ord_N==NULL) return;
      //--- Display the time of the current DOM snapshot in the chart comment,
      //--- the maximum number of displayed orders in DOM for a symbol,
      //--- the obtained number of orders in the current DOM snapshot,
      //--- the total number of DOM snapshots set in the series list and
      //--- the highest and lowest orders of the current DOM snapshot
      Comment
        (
         DFUN,book.Symbol(),": ",TimeMSCtoString(book.Time()),
         //", symbol book size=",sym.TicksBookdepth(),
         ", last book data total: ",book.DataTotal(),
         ", series books total: ",book_series.DataTotal(),
         "\nMax: ",ord_N.Header(),"\nMin: ",ord_0.Header()
        );
      //--- Display the first DOM snapshot in the journal
      if(first)
        {
         //--- series description
         book_series.Print();
         //--- snapshot description
         book.Print();
         first=false;
        }
     }
  }
//+------------------------------------------------------------------+
```

Let's make some corrections in the OnInitDoEasy() function as well. Now, creation of the tick series of all used symbols is implemented as a direct access to the appropriate method of the library main object. Besides, I have added the check of created DOM series:

```
//--- Check created timeseries - display descriptions of all created timeseries in the journal
//--- (true - only created ones, false - created and declared ones)
   engine.GetTimeSeriesCollection().PrintShort(false); // Short descriptions

//--- Create tick series of all used symbols
   engine.TickSeriesCreateAll();

//--- Check created tick series - display descriptions of all created tick series in the journal
   engine.GetTickSeriesCollection().Print();

//--- Check created DOM series - display descriptions of all created DOM series in the journal
   engine.GetMBookSeriesCollection().Print();
```

Compile the EA and launch it having preliminary set in the settings the usage of two symbols and the current chart period:

Data on the created DOM snapshot collection, full data on the very first suitable signal and short data on all profitable free signals are to be displayed in the journal. The end of the list features the data on the snapshot series and the first obtained DOM snapshot of the current chart:

```
Account 8550475: Artyom Trishkin (MetaQuotes Software Corp.) 10428.13 USD, 1:100, Hedge, MetaTrader 5 demo
--- Initializing "DoEasy" library ---
Working with predefined symbol list. The number of used symbols: 2
"AUDUSD" "EURUSD"
Working with the current timeframe only: H1
AUDUSD symbol timeseries:
- Timeseries "AUDUSD" H1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5385
EURUSD symbol timeseries:
- Timeseries "EURUSD" H1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 6310
Tick series "AUDUSD": Requested number of days: 1, Historical data created: 294221
Tick series "EURUSD": Requested number of days: 1, Historical data created: 216048
DOM snapshot series collection:
- "AUDUSD" DOM snapshot series: Requested number of days: 1, Actual history depth: 0
"EURUSD" DOM snapshot series: Requested number of days: 1, Actual history depth: 0
Subscribed to Depth of Market  AUDUSD
Subscribed to Depth of Market  EURUSD
Library initialization time: 00:00:25.015

============= Beginning of parameter list (Signal from the MQL5.com Signal service) =============
Account type: Demo
Publication date: 2018.06.04 16:45
Monitoring start date: 2018.06.04 16:45
Date of the latest update of the trading statistics: 2021.02.20 02:01
ID: 435626
Trading account leverage: 200
Trading result in pips: -72124
Position in the Rating of Signals: 12
Number of subscribers: 132
Number of trades: 7866
Status of account subscription to a signal: No
------
Account balance: 43144.27
Account equity: 43150.55
Account growth in %: 331.44
Maximum drawdown: 43.93
Signal subscription price: 0.00
Signal ROI (Return on Investment) in %: 331.51
------
Author login: "robots4forex"
Broker (company) name: "MetaQuotes Software Corp."
Broker server: "MetaQuotes-Demo"
Name: "Prospector Scalper EA"
Account currency: "GBP"
============= End of parameter list (Signal from the MQL5.com Signal service) =============

Signal "ADS MT5". Author login: vluxus, ID 478235, Growth: 251.84, Drawdown: 40.26, Price: 0.00
Signal "Sparrow USD ForexClub c 01012019". Author login: Tradotrade, ID 519975, Growth: 12.45, Drawdown: 14.98, Price: 0.00
Signal "RAZRED v03". Author login: joaoluiz_sa, ID 545382, Growth: 96.17, Drawdown: 28.18, Price: 0.00
Signal "NRDemo". Author login: v3sare, ID 655353, Growth: 2.94, Drawdown: 25.37, Price: 0.00
Signal "Amega 1000085182". Author login: AmegaTrust, ID 722001, Growth: 1.92, Drawdown: 5.13, Price: 0.00
Signal "Wns My strategy". Author login: WaldeliN, ID 727851, Growth: 103.69, Drawdown: 47.01, Price: 0.00
Signal "Bk EA". Author login: worknet, ID 749557, Growth: 506.88, Drawdown: 59.78, Price: 0.00
Signal "ROBIN 24". Author login: juanca034, ID 752873, Growth: 926.29, Drawdown: 30.25, Price: 0.00
Signal "Deny Forex". Author login: deny.mendonca, ID 759729, Growth: 149.06, Drawdown: 39.24, Price: 0.00
Signal "T Strategy". Author login: tonarino210, ID 760343, Growth: 28.87, Drawdown: 20.37, Price: 0.00
Signal "IC MT5 Demo". Author login: InvestForce, ID 760539, Growth: 67.01, Drawdown: 35.99, Price: 0.00
Signal "Gridingale". Author login: Myxx, ID 766073, Growth: 21.10, Drawdown: 15.55, Price: 0.00
Signal "Marcos Monteiro". Author login: slovenwill, ID 783988, Growth: 85.08, Drawdown: 17.59, Price: 0.00
Signal "Multi currency trend". Author login: mj2019, ID 785447, Growth: 54.42, Drawdown: 18.52, Price: 0.00
Signal "W7 901074879 Campeonato MT5". Author login: dramos236, ID 787269, Growth: 91.99, Drawdown: 21.20, Price: 0.00
Signal "Ramon Fx". Author login: viniciusramon18, ID 788732, Growth: 54.31, Drawdown: 9.46, Price: 0.00
Signal "Douglas demo w7". Author login: douglas.o.carne, ID 792392, Growth: 219.94, Drawdown: 43.61, Price: 0.00
"Suelen" signal. Author login: suelenacca, ID 794655, Growth: 67.40, Drawdown: 20.97, Price: 0.00
Signal "Conquers". Author login: borgesti, ID 795133, Growth: 37.23, Drawdown: 11.09, Price: 0.00
Signal "Conta demo torneio". Author login: Tiagoximenes, ID 798798, Growth: 42.36, Drawdown: 17.94, Price: 0.00
Signal "Conta demo de mil". Author login: Tiagoximenes, ID 798802, Growth: 132.02, Drawdown: 27.87, Price: 0.00
Signal "The art of Forex". Author login: Myxx, ID 801685, Growth: 170.29, Drawdown: 40.95, Price: 0.00
Signal "BB29 ICM". Author login: desmondpylow, ID 806971, Growth: 2.28, Drawdown: 41.60, Price: 0.00
Signal "Prometheus". Author login: g0079, ID 808538, Growth: 91.44, Drawdown: 22.98, Price: 0.00
Signal "Prueba robot 2 0 automatico". Author login: richwolfcompany, ID 809986, Growth: 76.76, Drawdown: 44.45, Price: 0.00
Signal "Deep Takeover Hedge StressTest 5M Candle". Author login: johnnypasado, ID 811819, Growth: 10.08, Drawdown: 13.58, Price: 0.00
Signal "Campeonato". Author login: AndreAutotecnic, ID 812233, Growth: 87.47, Drawdown: 13.79, Price: 0.00
Signal "OPM PRO". Author login: herinata, ID 812856, Growth: 38.55, Drawdown: 32.35, Price: 0.00
Signal "Slowly but surely 2". Author login: gyurmanz, ID 815467, Growth: 53.73, Drawdown: 13.08, Price: 0.00
Signal "Beef Waves". Author login: vladimir0005, ID 819055, Growth: 50.46, Drawdown: 32.69, Price: 0.00
Signal "Adriano Garcia". Author login: agarcia_ag, ID 823082, Growth: 111.62, Drawdown: 36.00, Price: 0.00
Signal "Max ScalperSpeed MT5". Author login: paran1615, ID 824333, Growth: 74.51, Drawdown: 40.62, Price: 0.00
Signal "SyH". Author login: gtrader2017, ID 826520, Growth: 42.78, Drawdown: 36.81, Price: 0.00
Signal "ECmp5s free". Author login: VallaLorenzo, ID 830456, Growth: 146.90, Drawdown: 27.64, Price: 0.00
Signal "MaxScalperSpeed MT5". Author login: paran1615, ID 835890, Growth: 64.33, Drawdown: 35.14, Price: 0.00
Signal "YEARNSIGNALS". Author login: yearnsignal2k19, ID 837512, Growth: 11.10, Drawdown: 2.54, Price: 0.00
Signal "AGS test 2". Author login: alireza.akbari, ID 838427, Growth: 7.93, Drawdown: 10.89, Price: 0.00
Signal "Waldeli003". Author login: WaldeliN, ID 838605, Growth: 32.98, Drawdown: 5.54, Price: 0.00
Signal "Michele". Author login: michele-m-r, ID 843351, Growth: 49.27, Drawdown: 13.90, Price: 0.00
Signal "SNAILER". Author login: 8F117EE2, ID 843458, Growth: 83.65, Drawdown: 11.86, Price: 0.00
Signal "Juniornicks". Author login: juniornicks, ID 845611, Growth: 100.25, Drawdown: 43.93, Price: 0.00
Signal "Black Hunter". Author login: christianlara, ID 845761, Growth: 51.94, Drawdown: 24.44, Price: 0.00
Signal "Master dizicheh1". Author login: awdtghuoilp, ID 857594, Growth: 5.04, Drawdown: 37.93, Price: 0.00
Signal "EUROS". Author login: Marketsystem, ID 858449, Growth: 5.31, Drawdown: 2.94, Price: 0.00
Signal "Scalpers risk10 pairs7 leverage100". Author login: leron34, ID 861750, Growth: 27.98, Drawdown: 20.53, Price: 0.00
Signal "EUREKA". Author login: Edmed933, ID 861927, Growth: 59.89, Drawdown: 7.32, Price: 0.00
Signal "Nadando Com Tubaroes". Author login: jun152, ID 862191, Growth: 21.18, Drawdown: 5.45, Price: 0.00
Signal "Demo using a grid system". Author login: RyanAfriansyah, ID 865900, Growth: 20.56, Drawdown: 8.38, Price: 0.00
Signal "Pilares". Author login: ValterCezar, ID 866672, Growth: 29.87, Drawdown: 18.96, Price: 0.00
Signal "EUROUSD". Author login: fxtrader036, ID 866719, Growth: 303.28, Drawdown: 40.70, Price: 0.00
Signal "LanzPower 25 S". Author login: sirlanz, ID 868027, Growth: 36.64, Drawdown: 45.53, Price: 0.00
Signal "Amadeu Volpato Desafio Internacional". Author login: Amadeu1971, ID 868928, Growth: 19.79, Drawdown: 12.57, Price: 0.00
Signal "Fernando correia W7". Author login: nandooo_123-hotmail, ID 870169, Growth: 41.70, Drawdown: 25.16, Price: 0.00
Signal "MAK GO". Author login: 9489631, ID 870413, Growth: 469.22, Drawdown: 36.31, Price: 0.00
Signal "Adriano Garcia W7bt 4 Pilares". Author login: agarcia_ag, ID 871868, Growth: 42.84, Drawdown: 13.19, Price: 0.00
Signal "Albertofxsemstop". Author login: albertosuga, ID 871969, Growth: 27.84, Drawdown: 19.36, Price: 0.00
Signal "BetoSTCDemo". Author login: betoabcsp, ID 872141, Growth: 29.03, Drawdown: 18.07, Price: 0.00
Signal "DESAFIOSEMSTOPLOSSCDS". Author login: cdsantos42, ID 873575, Growth: 19.47, Drawdown: 13.24, Price: 0.00
Signal "MrGeek7421". Author login: KamranAhmadi, ID 873583, Growth: 86.74, Drawdown: 16.33, Price: 0.00
Signal "Douglastorneio2w7". Author login: douglas.o.carne, ID 876302, Growth: 18.13, Drawdown: 15.34, Price: 0.00
Signal "Douglasw7demo1". Author login: douglas.o.carne, ID 876303, Growth: 148.80, Drawdown: 26.47, Price: 0.00
Signal "Douglastorneio1w7". Author login: douglas.o.carne, ID 876932, Growth: 136.86, Drawdown: 41.86, Price: 0.00
Signal "Campeonato mundial sem stop". Author login: Lpontes835, ID 878082, Growth: 23.52, Drawdown: 14.93, Price: 0.00
Signal "ALPHA IA v3". Author login: avaalpha, ID 878517, Growth: 2.77, Drawdown: 0.77, Price: 0.00
Signal "Gold x10". Author login: DynamixFX, ID 878540, Growth: 6.47, Drawdown: 8.87, Price: 0.00
Signal "MultiBolbandsRealM5". Author login: 11BREATH11, ID 879072, Growth: 83.18, Drawdown: 20.09, Price: 0.00
Signal "Ticols Stable profit". Author login: ticols, ID 879609, Growth: -56.37, Drawdown: 68.68, Price: 0.00
Signal "EA SkyBot MultiPares CENT". Author login: 4PerformanceFx, ID 882222, Growth: 248.38, Drawdown: 41.29, Price: 0.00
Signal "Trader Unity M15 100 rec". Author login: crifalo, ID 882268, Growth: 24.36, Drawdown: 26.11, Price: 0.00
Signal "Mad Piper Bill Millin". Author login: DiXOVERS, ID 882495, Growth: 251.06, Drawdown: 38.78, Price: 0.00
Signal "ProfitGuy STAR M Demo". Author login: justbond, ID 882847, Growth: 27.45, Drawdown: 24.89, Price: 0.00
Signal "EA GrayRock". Author login: serggray, ID 883235, Growth: 49.42, Drawdown: 28.68, Price: 0.00
Signal "FX FLASH". Author login: tradedeal, ID 883322, Growth: 9.17, Drawdown: 2.88, Price: 0.00
Signal "Optimizer". Author login: alama1, ID 884765, Growth: 73.53, Drawdown: 28.58, Price: 0.00
Signal "NnaFX Demo 02". Author login: 12259468, ID 886070, Growth: 136.64, Drawdown: 30.54, Pric: 0.00
Signal "Phantom5000 DEMO". Author login: JosephSmith, ID 887046, Growth: 43.41, Drawdown: 17.73, Price: 0.00
Signal "Art of Forex MadCat The G". Author login: The_G, ID 888018, Growth: 215.67, Drawdown: 40.86, Price: 0.00
Signal "ICMarkets MT5 AK 05". Author login: A.Klimkovsky, ID 889370, Growth: 13.03, Drawdown: 8.55, Price: 0.00
Signal "ProfitGuy STAR G Demo". Author login: justbond, ID 890551, Growth: 58.84, Drawdown: 58.80, Price: 0.00
Signal "BetoSemStopDM". Author login: betoabcsp, ID 892251, Growth: 94.96, Drawdown: 6.30, Price: 0.00
Signal "Smart Grid 26980 Demo". Author login: tm3912, ID 892313, Growth: 86.30, Drawdown: 37.19, Price: 0.00
Signal "Rel Vigor PSar n Red Dragon n mad max". Author login: DynamixFX, ID 892523, Growth: 73.74, Drawdown: 38.55, Price: 0.00
Signal "SolangeL". Author login: Mulherdeletras1, ID 894019, Growth: 108.40, Drawdown: 20.84, Price: 0.00
Signal "Paulotbone 1". Author login: paulotbone, ID 894062, Growth: 35.14, Drawdown: 23.93, Price: 0.00
Signal "GOLD MASTER". Author login: THEICD, ID 894983, Growth: 218.90, Drawdown: 22.80, Price: 0.00
Signal "Fxfrance". Author login: fxfrance, ID 895838, Growth: 369.48, Drawdown: 41.48, Price: 0.00
Signal "Marcos Paulo Serigatti". Author login: mpm4rcos, ID 895960, Growth: 45.62, Drawdown: 18.21, Price: 0.00
Signal "Roma Forex Desafio 4 Pilares". Author login: Jhonesroma, ID 896016, Growth: 29.60, Drawdown: 43.45, Price: 0.00
Signal "BOSBM OTC Test 5W". Author login: houli, ID 896563, Growth: 144.65, Drawdown: 22.94, Price: 0.00
Signal "FSTickTrade". Author login: onlyforsignup, ID 897751, Growth: 24.68, Drawdown: 16.20, Price: 0.00
Signal "Sun AI". Author login: Myxx, ID 899179, Growth: 16.82, Drawdown: 21.07, Price: 0.00
Signal "Equinox Demo". Author login: Kratoner, ID 905773, Growth: 44.46, Drawdown: 20.55, Price: 0.00
Signal "STRAGA Tornado VM1". Author login: stragapede, ID 906398, Growth: 36.70, Drawdown: 26.00, Price: 0.00
Signal "DiegoT". Author login: DiegoTT, ID 910230, Growth: 28.20, Drawdown: 11.49, Price: 0.00
Signal "Breaking Bad". Author login: Myxx, ID 911569, Growth: 7.63, Drawdown: 8.42, Price: 0.00
Signal "TradesaovivoBr". Author login: dhyegorodrigo1988, ID 913924, Growth: 16.77, Drawdown: 5.60, Price: 0.00
Signal "VantageFX Sunphone Dragon". Author login: sunphone, ID 916421, Growth: 345.56, Drawdown: 36.04, Price: 0.00
Signal "BYQS121". Author login: kadirryildiz, ID 916600, Growth: 32.85, Drawdown: 4.18, Price: 0.00
Signal "FBSLevelUP". Author login: manoj, ID 919106, Growth: 15.29, Drawdown: 10.49, Price: 0.00
Signal "Omega". Author login: zyntherius74, ID 922043, Growth: 70.18, Drawdown: 25.42, Price: 0.00
Signal "Best Bingo". Author login: kalnov-an, ID 926010, Growth: 59.90, Drawdown: 20.63, Price: 0.00
Signal "LCF 1". Author login: yosuf, ID 931735, Growth: 22.42, Drawdown: 36.26, Price: 0.00
Signal "Bao365". Author login: bao365, ID 933208, Growth: 28.41, Drawdown: 11.49, Price: 0.00
The "EURUSD" DOM snapshot series: Requested number of days: 1, Actual history depth: 1
"EURUSD" DOM snapshot (2021.02.24 22:22:54.654):
 - Sell order: 1.21576 [250.00]
 - Sell order: 1.21567 [100.00]
 - Sell order: 1.21566 [50.00]
 - Sell order: 1.21565 [36.00]
 - Buy order: 1.21563 [36.00]
 - Buy order: 1.21561 [50.00]
 - Buy order: 1.21559 [100.00]
 - Buy order: 1.21555 [250.00]
```

### What's next?

In the next article, I will create a collection of MQL5 signals.

All files of the current version of the library are attached below together with the test EA file for MQL5 for you to test and download.

Leave your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/9095#node00)

**\*Previous articles within the series:**

[Prices in DoEasy library (Part 62): Updating tick series in real time, preparation for working with Depth of Market](https://www.mql5.com/en/articles/8988)

[Prices in DoEasy library (Part 63): Depth of Market and its abstract request class](https://www.mql5.com/en/articles/9010)

[Prices in DoEasy library (Part 64): Depth of Market, classes of DOM snapshot and snapshot series objects](https://www.mql5.com/en/articles/9044)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/9095](https://www.mql5.com/ru/articles/9095)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/9095.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/9095/mql5.zip "Download MQL5.zip")(3907.3 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/367627)**
(5)


![leonerd](https://c.mql5.com/avatar/2017/5/5919A02E-9AEB.jpg)

**[leonerd](https://www.mql5.com/en/users/leonerd)**
\|
2 Mar 2021 at 11:35

And I take it the library is not cross-platform?


![BillionerClub](https://c.mql5.com/avatar/avatar_na2.png)

**[BillionerClub](https://www.mql5.com/en/users/billionerclub)**
\|
2 Mar 2021 at 11:49

**leonerd:**

And the library, I understand, is not cross-platform?

Ended at article 38-40, MT4 didn't have enough power to be on equal footing with MT5.

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
2 Mar 2021 at 15:28

**leonerd:**

And the library, I understand, is not cross-platform?

Unfortunately, MeteQuotes policy has become less loyal to the fourth version of the terminal, and articles about the fourth version are not accepted. Since many things made for the fifth version have to be written from scratch to be ported to the fourth, it takes almost the whole volume of the article and it is not accepted.

So it was decided to write only for five. Creating cross-platform is through freelancing to me. Unfortunately...

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
2 Mar 2021 at 15:29

**BillionerClub:**

Ended on article 38-40, MT4 did not have enough power to be on an equal footing with MT5

It can be done. But not with two lines of code :)


![nielsato](https://c.mql5.com/avatar/avatar_na2.png)

**[nielsato](https://www.mql5.com/en/users/nielsato)**
\|
1 May 2021 at 08:29

how to install the indicator


![Neural networks made easy (Part 12): Dropout](https://c.mql5.com/2/48/Neural_networks_made_easy_012.png)[Neural networks made easy (Part 12): Dropout](https://www.mql5.com/en/articles/9112)

As the next step in studying neural networks, I suggest considering the methods of increasing convergence during neural network training. There are several such methods. In this article we will consider one of them entitled Dropout.

![Prices in DoEasy library (Part 64): Depth of Market, classes of DOM snapshot and snapshot series objects](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__2.png)[Prices in DoEasy library (Part 64): Depth of Market, classes of DOM snapshot and snapshot series objects](https://www.mql5.com/en/articles/9044)

In this article, I will create two classes (the class of DOM snapshot object and the class of DOM snapshot series object) and test creation of the DOM data series.

![Other classes in DoEasy library (Part 66): MQL5.com Signals collection class](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__4.png)[Other classes in DoEasy library (Part 66): MQL5.com Signals collection class](https://www.mql5.com/en/articles/9146)

In this article, I will create the signal collection class of the MQL5.com Signals service with the functions for managing signals. Besides, I will improve the Depth of Market snapshot object class for displaying the total DOM buy and sell volumes.

![Machine learning in Grid and Martingale trading systems. Would you bet on it?](https://c.mql5.com/2/42/yandex_catboost__3.png)[Machine learning in Grid and Martingale trading systems. Would you bet on it?](https://www.mql5.com/en/articles/8826)

This article describes the machine learning technique applied to grid and martingale trading. Surprisingly, this approach has little to no coverage in the global network. After reading the article, you will be able to create your own trading bots.

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/9095&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068646014807636969)

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