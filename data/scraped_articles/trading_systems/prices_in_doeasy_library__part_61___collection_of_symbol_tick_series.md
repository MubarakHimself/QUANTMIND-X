---
title: Prices in DoEasy library (part 61): Collection of symbol tick series
url: https://www.mql5.com/en/articles/8952
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:50:57.404868
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/8952&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083155303391827471)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/8952#node01)
- [Class collection of tick data](https://www.mql5.com/en/articles/8952#node02)
- [Test](https://www.mql5.com/en/articles/8952#node03)
- [What's next?](https://www.mql5.com/en/articles/8952#node04)


### Concept

[In the previous article,](https://www.mql5.com/en/articles/8912) I created the tick data list object class to collect and store symbol ticks for a specified number of days. Since a program may use different symbols in its work, a separate list should be created for each of them. In this article, I will combine such lists into a tick data collection. This will be a regular list based on the class of dynamic array of pointers to instances of the [CObject class](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) and its descendants of the Standard library. The list is to store the pointers to created tick data lists for each symbol whose object class I prepared in the previous article.

The concept is identical to the one of constructing previous collection classes in the library. It will allow us to save, store, update, receive and use tick data of any symbols present in the library database in statistical analysis.

### Class collection of tick data

In \\MQL5\\Include\\DoEasy\ **Collections\**, create a new tick data collection class file named **TickSeriesCollection.mqh**.

The class is to be a descendant of the [class of the basic object of all library objects](https://www.mql5.com/en/articles/7071#node01).

Let's have a look at the class body and analyze its variables and methods:

```
//+------------------------------------------------------------------+
//|                                         TickSeriesCollection.mqh |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Objects\Ticks\TickSeries.mqh"
#include "..\Objects\Symbols\Symbol.mqh"
//+------------------------------------------------------------------+
//| Collection of symbol tick series                                 |
//+------------------------------------------------------------------+
class CTickSeriesCollection : public CBaseObj
  {
private:
   CListObj                m_list;                                   // List of used symbol tick series
//--- Return the tick series index by symbol name
   int                     IndexTickSeries(const string symbol);
public:
//--- Return (1) itself and (2) tick series collection list and (3) the number of tick series in the list
   CTickSeriesCollection  *GetObject(void)                              { return &this;               }
   CArrayObj              *GetList(void)                                { return &this.m_list;        }
   int                     DataTotal(void)                        const { return this.m_list.Total(); }
//--- Return the pointer to the tick series object (1) by symbol and (2) by index in the list
   CTickSeries            *GetTickseries(const string symbol);
   CTickSeries            *GetTickseries(const int index);
//--- Create a collection list of symbol tick series
   bool                    CreateCollection(const CArrayObj *list_symbols,const uint required=0);
//--- Set the flag of using the tick series of (1) a specified symbol and (2) all symbols
   void                    SetAvailableTickSeries(const string symbol,const bool flag=true);
   void                    SetAvailableTickSeries(const bool flag=true);
//--- Return the flag of using the tick series of (1) a specified symbol and (2) all symbols
   bool                    IsAvailableTickSeries(const string symbol);
   bool                    IsAvailableTickSeries(void);

//--- Set the number of days of the tick history of (1) a specified symbol and (2) all symbols
   bool                    SetRequiredUsedDays(const string symbol,const uint required=0);
   bool                    SetRequiredUsedDays(const uint required=0);

//--- Return the last tick object of a specified symbol (1) by index, (2) by time and (4) by time in milliseconds
   CDataTick              *GetTick(const string symbol,const int index);
   CDataTick              *GetTick(const string symbol,const datetime tick_time);
   CDataTick              *GetTick(const string symbol,const long tick_time_msc);

//--- Return the new tick flag of a specified symbol
   bool                    IsNewTick(const string symbol);

//--- Create a tick series of (1) a specified symbol and (2) all symbols
   bool                    CreateTickSeries(const string symbol,const uint required=0);
   bool                    CreateTickSeriesAll(const uint required=0);
//--- Update (1) a tick series of a specified symbol and (2) all symbols
   void                    Refresh(const string symbol);
   void                    Refresh(void);

//--- Display (1) the complete and (2) short collection description in the journal
   void                    Print(void);
   void                    PrintShort(void);

//--- Constructor
                           CTickSeriesCollection();
  };
//+------------------------------------------------------------------+
```

The member variable of the **m\_list** class is of CListObj type — the class, which is a descendant of the CArrayObj class of the standard library, just like many other lists created in the library. The only objective of the CListObj class is to implement the work of the Type() virtual method of the [CObject class](https://www.mql5.com/en/docs/standardlibrary/cobject) — the basic class of the standard library objects. The method should return the class type ID. In this case, it is the array type ID.

The Type() virtual method is implemented in the CListObj class, which was added to the library a long time ago:

```
//+------------------------------------------------------------------+
//|                                                      ListObj.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Arrays\ArrayObj.mqh>
//+------------------------------------------------------------------+
//| Class of collection lists                                        |
//+------------------------------------------------------------------+
class CListObj : public CArrayObj
  {
private:
   int               m_type;                    // List type
public:
   void              Type(const int type)       { this.m_type=type;     }
   virtual int       Type(void)           const { return(this.m_type);  }
                     CListObj()                 { this.m_type=0x7778;   }
  };
//+------------------------------------------------------------------+
```

Here the Type() method sets the passed value to the **m\_type** variable, while the virtual Type() method returns the value set by this variable.

By default (in the class constructor) the variable receives the same value of the array type ID as for CArrayObj — 0x7778.

The purpose of all class methods is described in the code comments. I will describe the implementation of these methods below.

**In the class constructor,** clear the list, set the sorted list flag for it and define the tick data collection list ID for it:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CTickSeriesCollection::CTickSeriesCollection()
  {
   this.m_list.Clear();
   this.m_list.Sort();
   this.m_list.Type(COLLECTION_TICKSERIES_ID);
  }
//+------------------------------------------------------------------+
```

The **IndexTickSeries()** private method returns the tick series index by symbol name:

```
//+------------------------------------------------------------------+
//| Return the tick series index by symbol name                      |
//+------------------------------------------------------------------+
int CTickSeriesCollection::IndexTickSeries(const string symbol)
  {
   const CTickSeries *obj=new CTickSeries(symbol==NULL || symbol=="" ? ::Symbol() : symbol);
   if(obj==NULL)
      return WRONG_VALUE;
   this.m_list.Sort();
   int index=this.m_list.Search(obj);
   delete obj;
   return index;
  }
//+------------------------------------------------------------------+
```

The method receives a name of the symbol whose tick series index should be returned from the list.

Next, create a temporary empty object of the tick series. It is to have the name of a symbol passed to the method.

Set the sorted list flag and and search for the object index in the list.

Next, remove a temporary object and return the obtained index. If the object is not found or creation of a temporary object failed, the method returns NULL.

**The method returning the pointer to the tick series object by symbol:**

```
//+------------------------------------------------------------------+
//| Return the object of tick series of a specified symbol           |
//+------------------------------------------------------------------+
CTickSeries *CTickSeriesCollection::GetTickseries(const string symbol)
  {
   int index=this.IndexTickSeries(symbol);
   return this.m_list.At(index);
  }
//+------------------------------------------------------------------+
```

The method receives a name of the symbol whose tick series object should be returned from the list.

Let's use the method I have just described to find the index of the tick series object in the list, get the pointer to the object by the found index and return it. If the index is not found, it is equal to -1, while the [At() method of the CArrayObj class](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjat) returns NULL.

**The method setting the flag of using the tick series of the specified symbol:**

```
//+------------------------------------------------------------------+
//| Set the flag of using the tick series of a specified symbol      |
//+------------------------------------------------------------------+
void CTickSeriesCollection::SetAvailableTickSeries(const string symbol,const bool flag=true)
  {
   CTickSeries *tickseries=this.GetTickseries(symbol);
   if(tickseries==NULL)
      return;
   tickseries.SetAvailable(flag);
  }
//+------------------------------------------------------------------+
```

The method receives a name of the symbol whose tick series object should receive the usage flag.

Using the GetTickseries() method described above, get the pointer to the tick series object from the list and set the flag passed to the method to it.

**The method setting the usage flag of tick series of all collection symbols:**

```
//+------------------------------------------------------------------+
//| Set the flag of using the tick series of all symbols             |
//+------------------------------------------------------------------+
void CTickSeriesCollection::SetAvailableTickSeries(const bool flag=true)
  {
   for(int i=0;i<this.m_list.Total();i++)
     {
      CTickSeries *tickseries=this.m_list.At(i);
      if(tickseries==NULL)
         continue;
      tickseries.SetAvailable(flag);
     }
  }
//+------------------------------------------------------------------+
```

In the loop by the total number of tick series in the list, get the next tick series object by the loop index and set the flag passed to the method for it.

**The method returning the flag of using the tick series of the specified symbol:**

```
//+------------------------------------------------------------------+
//| Return the flag of using the tick series of a specified symbol   |
//+------------------------------------------------------------------+
bool CTickSeriesCollection::IsAvailableTickSeries(const string symbol)
  {
   CTickSeries *tickseries=this.GetTickseries(symbol);
   if(tickseries==NULL)
      return false;
   return tickseries.IsAvailable();
  }
//+------------------------------------------------------------------+
```

The method receives a name of the symbol whose tick series object usage flag should be returned.

Use the GetTickseries() method to get the pointer to the tick series object of the necessary symbol and return the usage flag set for this object. If failed to get the object from the list, the method returns false.

**The method returning the flag of using tick series of all symbols:**

```
//+------------------------------------------------------------------+
//| Return the flag of using tick series of all symbols              |
//+------------------------------------------------------------------+
bool CTickSeriesCollection::IsAvailableTickSeries(void)
  {
   bool res=true;
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CTickSeries *tickseries=this.m_list.At(i);
      if(tickseries==NULL)
         continue;
      res &=tickseries.IsAvailable();
     }
   return res;
  }
//+------------------------------------------------------------------+
```

Declare the **res** variable and initialize it using the true value.

Next, in the loop by the total number of objects in the list,get the pointer to the next tick series object and add the usage flag defined for the current object to the **res** variable.

Upon the loop completion, return the obtained **res** value.

If a usage flag is not set for at least one of the objects in the list (false), **res** stores false upon the loop completion. Thus, the method allows to know whether usage flags are set for all tick series. True is returned only if the usage flags are set to true for each tick series object in the collection.

**The method setting the number of days of the tick history of the specified symbol:**

```
//+------------------------------------------------------------------+
//| Set the number of days of the tick history of a specified symbol |
//+------------------------------------------------------------------+
bool CTickSeriesCollection::SetRequiredUsedDays(const string symbol,const uint required=0)
  {
   CTickSeries *tickseries=this.GetTickseries(symbol);
   if(tickseries==NULL)
      return false;
   tickseries.SetRequiredUsedDays(required);
   return true;
  }
//+------------------------------------------------------------------+
```

The method receives a name of the symbol whose number of tick data days should be set.

Get the pointer to the tick series object using the previously considered method, set the number of days for it and return true.

If failed to get the pointer to the tick series object from the list, the method returns false.

**The method setting the number of days of the tick history of all symbols:**

```
//+------------------------------------------------------------------+
//| Set the number of days of the tick history of all symbols        |
//+------------------------------------------------------------------+
bool CTickSeriesCollection::SetRequiredUsedDays(const uint required=0)
  {
   bool res=true;
   for(int i=0;i<this.m_list.Total();i++)
     {
      CTickSeries *tickseries=this.m_list.At(i);
      if(tickseries==NULL)
        {
         res &=false;
         continue;
        }
      tickseries.SetRequiredUsedDays(required);
     }
   return res;
  }
//+------------------------------------------------------------------+
```

Declare the **res** variable and initialize it using the true value.

Next, in the loop by the total number of objects in the list,get the pointer to the next tick series object and, if failed to get the pointer to the object, the **res** variable gets false. Next, move on to the next object in the collection list.

Otherwise, set the number of tick data days for the current object.

Upon the loop completion, return the obtained **res** value.

If the number of tick data days is not set for at least one of the objects in the list, **res** stores false upon the loop completion. Thus, the method allows setting the number of days for all tick series in the collection and returns successful execution only if the number of days is set for each tick data object stored in the list.

**The method returning the tick object of the specified symbol by index in the tick series list:**

```
//+------------------------------------------------------------------+
//| Return the tick object of the specified symbol by index          |
//+------------------------------------------------------------------+
CDataTick *CTickSeriesCollection::GetTick(const string symbol,const int index)
  {
   CTickSeries *tickseries=this.GetTickseries(symbol);
   if(tickseries==NULL)
      return NULL;
   return tickseries.GetTickByListIndex(index);
  }
//+------------------------------------------------------------------+
```

The method passes the tick series symbol of the [CTickSeries class](https://www.mql5.com/en/articles/8912#node03) and the index of the necessary tick object stored in the tick series list.

Get the pointer to the tick series object from the symbol collection using the previously described GetTickseries() method and return the pointer to the tick object from the tick series list using the GetTickByListIndex() method I considered in the previous article.

If failed to get the tick series object, the method returns NULL. NULL may also be returned by the GetTickByListIndex() method of the CTickSeries class.

**The method returning the last tick object of the specified symbol by time from the tick series list:**

```
//+------------------------------------------------------------------+
//| Return the last tick object of the specified symbol by time      |
//+------------------------------------------------------------------+
CDataTick *CTickSeriesCollection::GetTick(const string symbol,const datetime tick_time)
  {
   CTickSeries *tickseries=this.GetTickseries(symbol);
   if(tickseries==NULL)
      return NULL;
   return tickseries.GetTick(tick_time);
  }
//+------------------------------------------------------------------+
```

The method passes the tick series symbol of the [CTickSeries class](https://www.mql5.com/en/articles/8912#node03) and the time of the necessary tick object stored in the tick series list.

Get the pointer to the tick series object from the symbol collection using the previously described GetTickseries() method and return the pointer to the tick object from the tick series list using the GetTick() method I considered in the previous article.

If failed to get the tick series object, the method returns NULL. Besides, NULL may also be returned by the GetTick() method of the CTickSeries class.

**The method returning the last tick object of the specified symbol by time in milliseconds from the tick series list:**

```
//+------------------------------------------------------------------+
//| Return the last tick object of the specified symbol              |
//| by time in milliseconds                                          |
//+------------------------------------------------------------------+
CDataTick *CTickSeriesCollection::GetTick(const string symbol,const long tick_time_msc)
  {
   CTickSeries *tickseries=this.GetTickseries(symbol);
   if(tickseries==NULL)
      return NULL;
   return tickseries.GetTick(tick_time_msc);
  }
//+------------------------------------------------------------------+
```

The method receives the tick series symbol of the [CTickSeries class](https://www.mql5.com/en/articles/8912#node03) and the time of the necessary tick object stored in the tick series list in milliseconds.

Get the pointer to the tick series object from the symbol collection using the previously described GetTickseries() method and return the pointer to the tick object from the tick series list using the GetTick() method I considered in the previous article.

If failed to get the tick series object, the method returns NULL. Besides, NULL may also be returned by the GetTick() method of the CTickSeries class.

There may be several ticks with a similar time for the last two methods returning tick objects by time, therefore the GetTick() method of the CTickSeries class returns the very last of them (with the latest time) as the most relevant one.

**The method returning the new tick flag of a specified symbol:**

```
//+------------------------------------------------------------------+
//| Return the new tick flag of a specified symbol                   |
//+------------------------------------------------------------------+
bool CTickSeriesCollection::IsNewTick(const string symbol)
  {
   CTickSeries *tickseries=this.GetTickseries(symbol);
   if(tickseries==NULL)
      return false;
   return tickseries.IsNewTick();
  }
//+------------------------------------------------------------------+
```

The method receives a name of the symbol whose new tick appearance flag should be returned.

Get the pointer to the tick series object from the symbol collection using the previously described GetTickseries() method and return the flag of the tick series' new tick using the IsNewTick() method of the CTickSeries class we considered in the previous article.

If failed to get the tick series object, the method returns false.

This ability is not implemented in the CTickSeries class yet. This will be done in the coming articles.

**The method creating a tick series of the specified symbol:**

```
//+------------------------------------------------------------------+
//| Create a tick series of a specified symbol                       |
//+------------------------------------------------------------------+
bool CTickSeriesCollection::CreateTickSeries(const string symbol,const uint required=0)
  {
   CTickSeries *tickseries=this.GetTickseries(symbol);
   if(tickseries==NULL)
      return false;
   return(tickseries.Create(required)>0);
  }
//+------------------------------------------------------------------+
```

The method receives a name of the symbol, whose tick series should be created, and the number of tick data days.

Get the pointer to the tick series object from the symbol collection using the previously described GetTickseries() method and return the flag indicating that the Create() method of the CTickSeries class has returned a value greater than zero (the number of created tick objects is not zero).

**The method creating tick series of all used symbols:**

```
//+------------------------------------------------------------------+
//| Create tick series of all symbols                                |
//+------------------------------------------------------------------+
bool CTickSeriesCollection::CreateTickSeriesAll(const uint required=0)
  {
   bool res=true;
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CTickSeries *tickseries=this.m_list.At(i);
      if(tickseries==NULL)
         continue;
      res &=(tickseries.Create(required)>0);
     }
   return res;
  }
//+------------------------------------------------------------------+
```

The method receives the number of tick data days.

Declare the **res** variable and initialize it using the true value.

Next, in the loop by the total number of objects in the list, get the pointer to the next tick series object andadd the flag indicating that the value returned by the Create() method of the CTickSeries class is greater than zero (the tick series has been created) to the **res** variable.

Upon the loop completion, return the obtained **res** value.

If a tick series is not created for at least one of the objects in the list, **res** stores false upon the loop completion. Thus, the method allows creating tick series collections for all symbols and returns successful execution only if tick series are created for each tick data object stored in the list.

**The method updating the tick series of the specified symbol:**

```
//+------------------------------------------------------------------+
//| Update a tick series of a specified symbol                       |
//+------------------------------------------------------------------+
void CTickSeriesCollection::Refresh(const string symbol)
  {
   CTickSeries *tickseries=this.GetTickseries(symbol);
   if(tickseries==NULL)
      return;
   tickseries.Refresh();
  }
//+------------------------------------------------------------------+
```

The method receives a symbol name whose tick series should be updated.

Get the pointer to the tick series object from the symbol collection using the previously described GetTickseries() method and update it using the Refresh() method of the CTickSeries class.

**The method updating tick series of all symbols:**

```
//+------------------------------------------------------------------+
//| Update tick series of all symbols                                |
//+------------------------------------------------------------------+
void CTickSeriesCollection::Refresh(void)
  {
   for(int i=0;i<this.m_list.Total();i++)
     {
      CTickSeries *tickseries=this.m_list.At(i);
      if(tickseries==NULL)
         continue;
      tickseries.Refresh();
     }
  }
//+------------------------------------------------------------------+
```

In the loop by the total number of objects in the list,get the pointer to the next tick series object by the loop index and update the seriesusing the Refresh() method of the CTickSeries class.

Updating tick series is not implemented in the CTickSeries class yet. This will be done in the coming articles.

**The method returning the full collection list to the journal:**

```
//+------------------------------------------------------------------+
//| Display complete collection description to the journal           |
//+------------------------------------------------------------------+
void CTickSeriesCollection::Print(void)
  {
   for(int i=0;i<this.m_list.Total();i++)
     {
      CTickSeries *tickseries=this.m_list.At(i);
      if(tickseries==NULL)
         continue;
      tickseries.Print();
     }
  }
//+------------------------------------------------------------------+
```

In the loop by the total number of objects in the list, get the pointer to the next tick series object by the loop index and display the full description of the tick series in the journal.

**The method returning the short collection list to the journal:**

```
//+------------------------------------------------------------------+
//| Display the short collection description in the journal          |
//+------------------------------------------------------------------+
void CTickSeriesCollection::PrintShort(void)
  {
   for(int i=0;i<this.m_list.Total();i++)
     {
      CTickSeries *tickseries=this.m_list.At(i);
      if(tickseries==NULL)
         continue;
      tickseries.PrintShort();
     }
  }
//+------------------------------------------------------------------+
```

In the loop by the total number of objects in the list,get the pointer to the next tick series object by the loop index and display the brief tick series description in the journal.

The methods considered above are meant for working with the already created collection list of pointers to tick data objects of different symbols. Our programs may use different symbols. The following method is used to create the collection object itself to place all the required tick series in it and get the appropriate pointers from the collection list to work with them.

**The method creating the collection list of symbol tick series:**

```
//+------------------------------------------------------------------+
//| Create a collection list of symbol tick series                   |
//+------------------------------------------------------------------+
bool CTickSeriesCollection::CreateCollection(const CArrayObj *list_symbols,const uint required=0)
  {
//--- If an empty list of symbol objects is passed, exit
   if(list_symbols==NULL)
      return false;
//--- Get the number of symbol objects in the passed list
   int total=list_symbols.Total();
//--- Clear the tick series collection list
   this.m_list.Clear();
//--- In a loop by all symbol objects
   for(int i=0;i<total;i++)
     {
      //--- get the next symbol object
      CSymbol *symbol_obj=list_symbols.At(i);
      //--- if failed to get a symbol object, move on to the next one in the list
      if(symbol_obj==NULL)
         continue;
      //--- Create a new empty tick series object
      CTickSeries *tickseries=new CTickSeries();
      //--- If failed to create the tick series object, move on to the next symbol in the list
      if(tickseries==NULL)
         continue;
      //--- Set a symbol name for a tick series object
      tickseries.SetSymbol(symbol_obj.Name());
      //--- Set the sorted list flag for the tick series collection list
      this.m_list.Sort();
      //--- If the object with the same symbol name is already present in the tick series collection list, remove the tick series object
      if(this.m_list.Search(tickseries)>WRONG_VALUE)
         delete tickseries;
      //--- otherwise, there is no object with such a symbol name in the collection yet
      else
        {
         //--- Set the number of tick data days for a tick series object
         tickseries.SetRequiredUsedDays(required);
         //--- if failed to add the tick series object to the collection list, remove the tick series object
         if(!this.m_list.Add(tickseries))
            delete tickseries;
        }
     }
//--- Return the flag indicating that the created collection list has a size greater than zero
   return this.m_list.Total()>0;
  }
//+------------------------------------------------------------------+
```

The method is quite simple — it receives the list of symbols used in the program (the list exists for quite a long time already and is used when creating the symbol timeseries collection). Next, in the loop by the total number of symbols, create a new tick series object and set its symbol name from the list of symbols in the current loop position. If the tick series object with such a symbol is not present in the list yet, set the number of tick data days passed to the method and add the object to the collection list. This should be done for each symbol in the list. Here I have outlined the system in brief. If you dig deeper, you will see the checks for successful creation and adding tick series objects to the list and deleting unnecessary objects if needed. The entire method logic is described in detail in its listing. I will leave it for you to analyze.

The **CEngine** main library class is used to connect the created collection with the "outside world".

The class is stored in \\MQL5\\Include\\DoEasy\ **Engine.mqh**.

Connect the file of a newly created class to it:

```
//+------------------------------------------------------------------+
//|                                                       Engine.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
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
#include "TradingControl.mqh"
//+------------------------------------------------------------------+
```

In the private section of the class, declare the tick series collection class object:

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
   CResourceCollection  m_resource;                      // Resource list
   CTradingControl      m_trading;                       // Trading management object
   CPause               m_pause;                         // Pause object
   CArrayObj            m_list_counters;                 // List of timer counters
```

The class features the **SetUsedSymbols()** method allowing us to set the list of symbols to be used in the program.

Add passing the number of days that should have tick data in the library:

```
//--- Set the list of used symbols in the symbol collection and create the collection of symbol timeseries
   bool                 SetUsedSymbols(const string &array_symbols[],const uint required=0);
```

By default, zero is passed (which means one day) and set in \\MQL5\\Include\\DoEasy\ **Defines.mqh** by the TICKSERIES\_DEFAULT\_DAYS\_COUNT constant.

In the method implementation, add creating the collection of tick series.

```
//+------------------------------------------------------------------+
//| Set the list of used symbols in the symbol collection            |
//| and create the symbol timeseries collection                      |
//+------------------------------------------------------------------+
bool CEngine::SetUsedSymbols(const string &array_symbols[],const uint required=0)
  {
   bool res=this.m_symbols.SetUsedSymbols(array_symbols);
   CArrayObj *list=this.GetListAllUsedSymbols();
   if(list==NULL)
      return false;
   res&=this.m_time_series.CreateCollection(list);
   res&=this.m_tick_series.CreateCollection(list,required);
   return res;
  }
//+------------------------------------------------------------------+
```

Now two collections (timeseries and tick series collections) are created when calling the method from the program.

In the public class section, add the methods for accessing the tick series collection class from custom programs:

```
//--- Copy the specified double property of the specified timeseries of the specified symbol to the array
//--- Regardless of the array indexing direction, copying is performed the same way as copying to a timeseries array
   bool                 SeriesCopyToBufferAsSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,const ENUM_BAR_PROP_DOUBLE property,
                                                   double &array[],const double empty=EMPTY_VALUE)
                          { return this.m_time_series.CopyToBufferAsSeries(symbol,timeframe,property,array,empty);}

//--- Return (1) the tick series collection, (2) the list of tick series from the tick series collection
   CTickSeriesCollection *GetTickSeriesCollection(void)                       { return &this.m_tick_series;                                     }
   CArrayObj           *GetListTickSeries(void)                               { return this.m_tick_series.GetList();                            }

//--- Return (1) the buffer collection and (2) the buffer list from the collection
```

For now, it is sufficient to return the tick series collection object itself and the collection list from it to the program.

**Currently, this is all we need to create the tick series collection.**

### Test

To test the creation of the tick series collection for the symbol program operation, I will take the EA from the previous article and save it in \\MQL5\\Experts\ **TestDoEasy\\Part61\** as **TestDoEasyPart61.mq5**.

Since now all tick series are available from the library itself, let's remove the inclusion of their class file in the program:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart60.mq5 |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
#include <DoEasy\Objects\Ticks\TickSeries.mqh>
//--- enums
```

In the area of program global variables, remove "New tick" object variables and the current symbol's tick series data object:

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

//--- "New tick" object
CNewTickObj    check_tick;
//--- Object of the current symbol tick series data
CTickSeries    tick_series;
//+------------------------------------------------------------------+
```

At the end of OnInit() handler, remove setting the current symbol for the "New tick" object:

```
//--- Wait for 600 milliseconds
   engine.Pause(600);
   engine.PlaySoundByDescription(TextByLanguage("Звук упавшей монетки 2","Falling coin 2"));

//--- Set the current symbol for "New tick" object
   check_tick.SetSymbol(Symbol());
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

Remove the code block for checking the creation of a tick series of the current symbol from the OnInitDoEasy() function:

```
//--- Check created timeseries - display descriptions of all created timeseries in the journal
//--- (true - only created ones, false - created and declared ones)
   engine.GetTimeSeriesCollection().PrintShort(false); // Short descriptions
   //engine.GetTimeSeriesCollection().Print(true);      // Full descriptions

//--- Code block for checking the tick list creation and working with it
   Print("");
//--- Since the tick series object is created with the default constructor,
//--- set a symbol, usage flag and the number of days (the default is 1) to copy the ticks
//--- Create the tick series and printed data in the journal
   tick_series.SetSymbol(Symbol());
   tick_series.SetAvailable(true);
   tick_series.SetRequiredUsedDays();
   tick_series.Create();
   tick_series.Print();

   Print("");
//--- Get and display in the journal the data of an object with the highest Ask price in the daily price range
   int index_max=CSelect::FindTickDataMax(tick_series.GetList(),TICK_PROP_ASK);
   CDataTick *tick_max=tick_series.GetList().At(index_max);
   if(tick_max!=NULL)
      tick_max.Print();
//--- Get and display in the journal the data of an object with the lowest Bid price in the daily price range
   int index_min=CSelect::FindTickDataMin(tick_series.GetList(),TICK_PROP_BID);
   CDataTick *tick_min=tick_series.GetList().At(index_min);
   if(tick_min!=NULL)
      tick_min.Print();

//--- Create resource text files
```

Now we need to set the creation of tick series for all symbols of the created tick data collection here:

```
//--- Check created timeseries - display descriptions of all created timeseries in the journal
//--- (true - only created ones, false - created and declared ones)
   engine.GetTimeSeriesCollection().PrintShort(false); // Short descriptions
   //engine.GetTimeSeriesCollection().Print(true);      // Full descriptions

//--- Create tick series of all used symbols
   engine.GetTickSeriesCollection().CreateTickSeriesAll();
//--- Check created tick series - display descriptions of all created tick series in the journal
   engine.GetTickSeriesCollection().Print();

//--- Create resource text files
```

In the OnTick() handler, upon arrival of a new tick, try to find a tick object with the highest Ask and lowest Bid price in the tick data lists for each symbol, as well as display the parameters of each detected tick object in the journal:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Handle the NewTick event in the library
   engine.OnTick(rates_data);

//--- If working in the tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer(rates_data);   // Working in the timer
      PressButtonsControl();        // Button pressing control
      engine.EventsHandling();      // Working with events
     }

//--- If the trailing flag is set
   if(trailing_on)
     {
      TrailingPositions();          // Trailing positions
      TrailingOrders();             // Trailing pending orders
     }

//--- Check created tick data on the first tick
//--- Get and display in the journal the data of an object with the highest Ask price and the lowest Bid price in the daily price range
   static bool check=false;
   if(!check)
     {
      Print("");
      //--- Get the pointer to the list of tick data of all symbols from the tick collection
      CArrayObj* list=engine.GetTickSeriesCollection().GetList();
      int total=engine.GetTickSeriesCollection().DataTotal();
      //--- In the loop by the number of tick series in the collection
      for(int i=0;i<list.Type();i++)
        {
         //--- Get the next tick series from the collection by index
         CTickSeries *tick_series=engine.GetTickSeriesCollection().GetTickseries(i);
         if(tick_series!=NULL)
           {
            //--- In the obtained tick series, find the indices of tick objects with the highest Ask and the lowest Bid
            int index_max=CSelect::FindTickDataMax(tick_series.GetList(),TICK_PROP_ASK);
            int index_min=CSelect::FindTickDataMin(tick_series.GetList(),TICK_PROP_BID);
            //--- Display the data of the tick objects obtained from the tick series in the journal
            engine.GetTickSeriesCollection().GetTick(tick_series.Symbol(),index_max).Print();
            engine.GetTickSeriesCollection().GetTick(tick_series.Symbol(),index_min).Print();
           }
        }
      check=true;
     }
  }
//+------------------------------------------------------------------+
```

Compile the EA and launch it on a chart of any symbol. Prior to that, make sure to enable the current timeframe and symbols from the predefined list, in which only the two first symbols are left out of the entire proposed symbols:

![](https://c.mql5.com/2/42/terminal64_YlypVHSa3Q.png)

After a short time required to create tick data for two used symbols in the OnInit() handler, the journal receives data on the program parameters, created timeseries and created tick data. Upon arrival of the new tick, the journal receives the data on four detected ticks with the highest Ask and lowest Bid for each of the two symbols:

```
Account 8550475: Artyom Trishkin (MetaQuotes Software Corp.) 10426.13 USD, 1:100, Hedge, MetaTrader 5 demo
--- Initializing "DoEasy" library ---
Working with predefined symbol list. The number of used symbols: 2
"AUDUSD" "EURUSD"
Working with the current timeframe only: H1
AUDUSD symbol timeseries:
- Timeseries "AUDUSD" H1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 6194
EURUSD symbol timeseries:
- Timeseries "EURUSD" H1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5675
Tick series "AUDUSD": Requested number of days: 1, Historical data created: 142712
Tick series "EURUSD": Requested number of days: 1, Historical data created: 113985
Library initialization time: 00:00:06.156

============= Beginning of parameter list (Tick "AUDUSD" 2021.01.19 10:06:53.387) =============
Last price update time in milliseconds: 2021.01.19 10:06:53.387
Last price update time: 2021.01.19 10:06:53
Volume for the current Last price: 0
Flags: 6
Changed data on the tick:
 - Ask price change
 - Bid price change
------
Bid price: 0.77252
Ask price: 0.77256
Last price: 0.00000
Volume for the current Last price with greater accuracy: 0.00
Spread: 0.00004
------
Symbol: "AUDUSD"
============= End of parameter list (Tick "AUDUSD" 2021.01.19 10:06:53.387) =============

============= Beginning of parameter list (Tick "AUDUSD" 2021.01.18 11:51:48.662) =============
Last price update time in milliseconds: 2021.01.18 11:51:48.662
Last price update time: 2021.01.18 11:51:48
Volume for the current Last price: 0
Flags: 130
Changed data on the tick:
 - Bid price change
------
Bid price: 0.76589
Ask price: 0.76593
Last price: 0.00000
Volume for the current Last price with greater accuracy: 0.00
Spread: 0.00004
------
Symbol: "AUDUSD"
============= End of parameter list (Tick "AUDUSD" 2021.01.18 11:51:48.662) =============

============= Beginning of parameter list (Tick "EURUSD" 2021.01.19 10:05:07.246) =============
Last price update time in milliseconds: 2021.01.19 10:05:07.246
Last price update time: 2021.01.19 10:05:07
Volume for the current Last price: 0
Flags: 6
Changed data on the tick:
 - Ask price change
 - Bid price change
------
Bid price: 1.21189
Ask price: 1.21189
Last price: 0.00000
Volume for the current Last price with greater accuracy: 0.00
Spread: 0.00000
------
Symbol: "EURUSD"
============= End of parameter list (Tick "EURUSD" 2021.01.19 10:05:07.246) =============

============= Beginning of parameter list (Tick "EURUSD" 2021.01.18 14:57:53.847) =============
Last price update time in milliseconds: 2021.01.18 14:57:53.847
Last price update time: 2021.01.18 14:57:53
Volume for the current Last price: 0
Flags: 134
Changed data on the tick:
 - Ask price change
 - Bid price change
------
Bid price: 1.20536
Ask price: 1.20536
Last price: 0.00000
Volume for the current Last price with greater accuracy: 0.00
Spread: 0.00000
------
Symbol: "EURUSD"
============= End of parameter list (Tick "EURUSD" 2021.01.18 14:57:53.847) =============
```

According to the journal, the library initialization and creation of tick data lists took 16 seconds. Upon the arrival of a new tick, we found two ticks — with the highest Ask and lowest Bid prices — per each of the used symbols for the current day.

### What's next?

In the next article, I will start creating realtime update and control of data events in the tick collection created today.

All files of the current version of the library are attached below together with the test EA file for MQL5 for you to test and download.

The tick series collection class is under development, therefore its use in custom programs at this stage is strongly not recommended.

Leave your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/8952#node00)

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

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8952](https://www.mql5.com/ru/articles/8952)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8952.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/8952/mql5.zip "Download MQL5.zip")(3880.08 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/365416)**
(2)


![Christian](https://c.mql5.com/avatar/2016/3/56F90C3B-A503.gif)

**[Christian](https://www.mql5.com/en/users/collider)**
\|
27 Mar 2021 at 18:56

Another good article.

Found a small error. MetaEditor hangs up.

If [the](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer "Reference book MQL5 : Object properties") TickSeries [object](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer "Reference book MQL5 : Object properties") is not filled with ticks. (m\_amount = 0 )

the GetTick() function returns NULL and the .Print() method causes the editor to crash.

TestDoEasyPart61.mq5

```
237           engine.GetTickSeriesCollection().GetTick(tick_series.Symbol(),index_max).Print();
238           engine.GetTickSeriesCollection().GetTick(tick_series.Symbol(),index_min).Print();
```

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
27 Mar 2021 at 19:29

**Christian :**

Another good article.

Found a small error. MetaEditor hangs up.

If [the](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer "Reference book MQL5: Object properties") TickSeries [object](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer "Reference book MQL5: Object properties") is not filled with ticks. (m\_amount = 0 )

the GetTick() function returns NULL and the .Print() method causes the editor to crash.

TestDoEasyPart61.mq5

Thanks, I will fix it.

![Prices in DoEasy library (part 62): Updating tick series in real time, preparation for working with Depth of Market](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library.png)[Prices in DoEasy library (part 62): Updating tick series in real time, preparation for working with Depth of Market](https://www.mql5.com/en/articles/8988)

In this article, I will implement updating tick data in real time and prepare the symbol object class for working with Depth of Market (DOM itself is to be implemented in the next article).

![Prices in DoEasy library (part 60): Series list of symbol tick data](https://c.mql5.com/2/41/MQL5-avatar-doeasy-library__4.png)[Prices in DoEasy library (part 60): Series list of symbol tick data](https://www.mql5.com/en/articles/8912)

In this article, I will create the list for storing tick data of a single symbol and check its creation and retrieval of required data in an EA. Tick data lists that are individual for each used symbol will further constitute a collection of tick data.

![Neural networks made easy (Part 11): A take on GPT](https://c.mql5.com/2/48/Neural_networks_made_easy_011.png)[Neural networks made easy (Part 11): A take on GPT](https://www.mql5.com/en/articles/9025)

Perhaps one of the most advanced models among currently existing language neural networks is GPT-3, the maximal variant of which contains 175 billion parameters. Of course, we are not going to create such a monster on our home PCs. However, we can view which architectural solutions can be used in our work and how we can benefit from them.

![Multilayer perceptron and backpropagation algorithm](https://c.mql5.com/2/41/Sem_tbtulo.png)[Multilayer perceptron and backpropagation algorithm](https://www.mql5.com/en/articles/8908)

The popularity of these two methods grows, so a lot of libraries have been developed in Matlab, R, Python, C++ and others, which receive a training set as input and automatically create an appropriate network for the problem. Let us try to understand how the basic neural network type works (including single-neuron perceptron and multilayer perceptron). We will consider an exciting algorithm which is responsible for network training - gradient descent and backpropagation. Existing complex models are often based on such simple network models.

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/8952&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083155303391827471)

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