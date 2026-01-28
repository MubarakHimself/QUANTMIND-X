---
title: Prices in DoEasy library (Part 64): Depth of Market, classes of DOM snapshot and snapshot series objects
url: https://www.mql5.com/en/articles/9044
categories: Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:50:26.899166
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/9044&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083149251782907376)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/9044#node01)
- [Improving library classes](https://www.mql5.com/en/articles/9044#node02)
- [Depth of Market snapshot object class](https://www.mql5.com/en/articles/9044#node03)
- [Depth of Market snapshot series object class](https://www.mql5.com/en/articles/9044#node04)
- [Test](https://www.mql5.com/en/articles/9044#node05)
- [What's next?](https://www.mql5.com/en/articles/9044#node06)


### Concept

In the [previous article](https://www.mql5.com/en/articles/9010), I have created the class of the Depth of Market (DOM) abstract order object and its descendants. The multitude of these objects constitutes one DOM snapshot obtained during one call of the [MarketBookGet()](https://www.mql5.com/en/docs/marketinformation/marketbookget) function at the moment the [OnBookEvent()](https://www.mql5.com/en/docs/event_handlers/onbookevent) handler is activated. Data obtained by the MarketBookGet() function is set in the [MqlBookInfo](https://www.mql5.com/en/docs/constants/structures/mqlbookinfo) structure array. Based on the obtained data, we are able to create the DOM snapshot object which is to store all DOM orders obtained in the MqlBookInfo structure array. In other words, the data in the mentioned structure array will form the DOM snapshot, in which each structure member is represented by a single DOM order object. Each activation of the OnBookEvent() handler leads to creating another DOM snapshot which is, in turn, entered to the object of the DOM snapshot series class.

The standard library's [class of dynamic array of pointers to instances of the CObject class and its descendants](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) is to be used as the list of storing DOM snapshot objects. The size of the list is to be limited by a specified number of objects. By default, the size of the list does not exceed 200,000 DOM snapshot objects, which should cover approximately one or two trading days. Such DOM snapshot series lists are to be created for each symbol used in the program. As a result, every such list is to be stored in the DOM data collection.

Here, I am going to create two classes — the class of DOM snapshot object of a single symbol and the class of DOM snapshot series of a single symbol. In the next article, I will create and test the class of DOM snapshot series collection.

### Improving library classes

In \\MQL5\\Include\\DoEasy\ **Data.mqh**, add the library's new message indices:

```
//--- CMarketBookOrd
   MSG_MBOOK_ORD_TEXT_MBOOK_ORD,                      // Order in DOM
   MSG_MBOOK_ORD_VOLUME,                              // Volume
   MSG_MBOOK_ORD_VOLUME_REAL,                         // Extended accuracy volume
   MSG_MBOOK_ORD_STATUS_BUY,                          // Buy side
   MSG_MBOOK_ORD_STATUS_SELL,                         // Sell side
   MSG_MBOOK_ORD_TYPE_SELL,                           // Sell order
   MSG_MBOOK_ORD_TYPE_BUY,                            // Buy order
   MSG_MBOOK_ORD_TYPE_SELL_MARKET,                    // Sell order by Market
   MSG_MBOOK_ORD_TYPE_BUY_MARKET,                     // Buy order by Market

//--- CMarketBookSnapshot
   MSG_MBOOK_SNAP_TEXT_SNAPSHOT,                      // DOM snapshot

//--- CMBookSeries
   MSG_MBOOK_SERIES_TEXT_MBOOKSERIES,                 // DOM snapshot series

  };
//+------------------------------------------------------------------+
```

and text messages corresponding to newly added indices:

```
//--- CMarketBookOrd
   {"Заявка в стакане цен","Order in Depth of Market"},
   {"Объем","Volume"},
   {"Объем c повышенной точностью","Volume Real"},
   {"Сторона Buy","Buy side"},
   {"Сторона Sell","Sell side"},
   {"Заявка на продажу","Sell order"},
   {"Заявка на покупку","Buy order"},
   {"Заявка на продажу по рыночной цене","Sell order at market price"},
   {"Заявка на покупку по рыночной цене","Buy order at market price"},

//--- CMarketBookSnapshot
   {"Снимок стакана цен","Depth of Market Snapshot"},

//--- CMBookSeries
   {"Серия снимков стакана цен","Series of shots of the Depth of Market"},

  };
//+---------------------------------------------------------------------+
```

To be able to specify the criterion for sorting by time when searching for the necessary DOM snapshot objects in the series list, we need to add a new property to the DOM order object — the time of receiving a snapshot in milliseconds. The order itself has no such property but we are able to track the time of receiving a DOM snapshot. To avoid introducing new enumerations for the DOM snapshot series list containing a single integer property (time of obtaining a snapshot in milliseconds), let's add this property to the DOM order object properties. The time of obtaining a snapshot is assigned to each order in a single snapshot object. We will use this newly updated constant for searching and sorting in the series list.

In \\MQL5\\Include\\DoEasy\ **Defines.mqh**, enter the parameters of the DOM snapshot series so that we able to set the necessary amount of data days and the maximum possible number of snapshots in the list:

```
//--- Tick series parameters
#define TICKSERIES_DEFAULT_DAYS_COUNT  (1)                        // Required number of days for tick data in default series
#define TICKSERIES_MAX_DATA_TOTAL      (200000)                   // Maximum number of stored tick data of a single symbol
//--- Parameters of the DOM snapshot series
#define MBOOKSERIES_DEFAULT_DAYS_COUNT (1)                        // The default required number of days for DOM snapshots in the series
#define MBOOKSERIES_MAX_DATA_TOTAL     (200000)                   // Maximum number of stored DOM snapshots of a single symbol
//+------------------------------------------------------------------+
```

I am not going to use the first parameter (number of days) yet — later on, I will try to link the data to the number of tick data days. Currently, I will use the second parameter only — maximum possible amount of DOM snapshot data.

In the same file, add the new integer DOM order object property (the time in milliseconds) and increase the number of integer object properties up to **4**:

```
//+------------------------------------------------------------------+
//| Integer properties of DOM order                                  |
//+------------------------------------------------------------------+
enum ENUM_MBOOK_ORD_PROP_INTEGER
  {
   MBOOK_ORD_PROP_STATUS = 0,                         // Order status
   MBOOK_ORD_PROP_TYPE,                               // Order type
   MBOOK_ORD_PROP_VOLUME,                             // Order volume
   MBOOK_ORD_PROP_TIME_MSC,                           // Time of making a DOM snapshot in milliseconds
  };
#define MBOOK_ORD_PROP_INTEGER_TOTAL (4)              // Total number of integer properties
#define MBOOK_ORD_PROP_INTEGER_SKIP  (0)              // Number of integer DOM properties not used in sorting
//+------------------------------------------------------------------+
```

Since I have added a new integer property, I should also add a new criterion for sorting by integer properties:

```
//+------------------------------------------------------------------+
//| Possible sorting criteria of DOM orders                          |
//+------------------------------------------------------------------+
#define FIRST_MB_DBL_PROP  (MBOOK_ORD_PROP_INTEGER_TOTAL-MBOOK_ORD_PROP_INTEGER_SKIP)
#define FIRST_MB_STR_PROP  (MBOOK_ORD_PROP_INTEGER_TOTAL-MBOOK_ORD_PROP_INTEGER_SKIP+MBOOK_ORD_PROP_DOUBLE_TOTAL-MBOOK_ORD_PROP_DOUBLE_SKIP)
enum ENUM_SORT_MBOOK_ORD_MODE
  {
//--- Sort by integer properties
   SORT_BY_MBOOK_ORD_STATUS = 0,                      // Sort by order status
   SORT_BY_MBOOK_ORD_TYPE,                            // Sort by order type
   SORT_BY_MBOOK_ORD_VOLUME,                          // Sort by order volume
   SORT_BY_MBOOK_ORD_TIME_MSC,                        // Sort by time of making a DOM snapshot in milliseconds
//--- Sort by real properties
   SORT_BY_MBOOK_ORD_PRICE = FIRST_MB_DBL_PROP,       // Sort by order price
   SORT_BY_MBOOK_ORD_VOLUME_REAL,                     // Sort by extended accuracy order volume
//--- Sort by string properties
   SORT_BY_MBOOK_ORD_SYMBOL = FIRST_MB_STR_PROP,      // Sort by symbol name
  };
//+------------------------------------------------------------------+
```

This constant will be specified as a parameter used to sort DOM snapshot objects in the currently developed class of DOM snapshot series object.

Thus, there is a need for the methods of searching and sorting DOM order objects in the **CSelect** class file stored in \\MQL5\\Include\\DoEasy\\Services\ **Select.mqh** and described in details [in the third article](https://www.mql5.com/en/articles/5687). Now I will simply describe all the necessary modifications of this class for organizing the search and sorting by properties of DOM order objects.

Include the class of DOM abstract order object to the file:

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
#include "..\Objects\Series\SeriesDE.mqh"
#include "..\Objects\Indicators\Buffer.mqh"
#include "..\Objects\Indicators\IndicatorDE.mqh"
#include "..\Objects\Indicators\DataInd.mqh"
#include "..\Objects\Ticks\DataTick.mqh"
#include "..\Objects\Book\MarketBookOrd.mqh"
//+------------------------------------------------------------------+
```

Declare all the necessary methods at the end of the class body:

```
//+------------------------------------------------------------------+
//| Methods of working with DOM data                                 |
//+------------------------------------------------------------------+
   //--- Return the list of DOM data with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByMBookProperty(CArrayObj *list_source,ENUM_MBOOK_ORD_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByMBookProperty(CArrayObj *list_source,ENUM_MBOOK_ORD_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByMBookProperty(CArrayObj *list_source,ENUM_MBOOK_ORD_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the DOM data index in the list with the maximum value of (1) integer, (2) real and (3) string property of data
   static int        FindMBookMax(CArrayObj *list_source,ENUM_MBOOK_ORD_PROP_INTEGER property);
   static int        FindMBookMax(CArrayObj *list_source,ENUM_MBOOK_ORD_PROP_DOUBLE property);
   static int        FindMBookMax(CArrayObj *list_source,ENUM_MBOOK_ORD_PROP_STRING property);
   //--- Return the DOM data index in the list with the minimum value of (1) integer, (2) real and (3) string property of data
   static int        FindMBookMin(CArrayObj *list_source,ENUM_MBOOK_ORD_PROP_INTEGER property);
   static int        FindMBookMin(CArrayObj *list_source,ENUM_MBOOK_ORD_PROP_DOUBLE property);
   static int        FindMBookMin(CArrayObj *list_source,ENUM_MBOOK_ORD_PROP_STRING property);
//---
  };
//+------------------------------------------------------------------+
```

Implement them beyond the class body:

```
//+------------------------------------------------------------------+
//| Methods of working with DOM data                                 |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Return the list of DOM data with one of integer                  |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByMBookProperty(CArrayObj *list_source,ENUM_MBOOK_ORD_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   int total=list_source.Total();
   for(int i=0; i<total; i++)
     {
      CMarketBookOrd *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      long obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of DOM data with one of real                     |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByMBookProperty(CArrayObj *list_source,ENUM_MBOOK_ORD_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CMarketBookOrd *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      double obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of DOM data with one of string                   |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByMBookProperty(CArrayObj *list_source,ENUM_MBOOK_ORD_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CMarketBookOrd *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      string obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the DOM data index in the list                            |
//| with the maximum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindMBookMax(CArrayObj *list_source,ENUM_MBOOK_ORD_PROP_INTEGER property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CMarketBookOrd *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CMarketBookOrd *obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      long obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the DOM data index in the list                            |
//| with the maximum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindMBookMax(CArrayObj *list_source,ENUM_MBOOK_ORD_PROP_DOUBLE property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CMarketBookOrd *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CMarketBookOrd *obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      double obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the DOM data index in the list                            |
//| with the maximum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindMBookMax(CArrayObj *list_source,ENUM_MBOOK_ORD_PROP_STRING property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CMarketBookOrd *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CMarketBookOrd *obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      string obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the DOM data index in the list                            |
//| with the minimum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindMBookMin(CArrayObj* list_source,ENUM_MBOOK_ORD_PROP_INTEGER property)
  {
   int index=0;
   CMarketBookOrd *min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CMarketBookOrd *obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      long obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the DOM data index in the list                            |
//| with the minimum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindMBookMin(CArrayObj* list_source,ENUM_MBOOK_ORD_PROP_DOUBLE property)
  {
   int index=0;
   CMarketBookOrd *min_obj=NULL;
   int total=list_source.Total();
   if(total== 0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CMarketBookOrd *obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      double obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the DOM data index in the list                            |
//| with the minimum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindMBookMin(CArrayObj* list_source,ENUM_MBOOK_ORD_PROP_STRING property)
  {
   int index=0;
   CMarketBookOrd *min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CMarketBookOrd *obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      string obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
```

The logic of the methods' operation is similar to that of other methods of the class I created earlier for other library objects. Therefore, I will not dwell on describing the operation of the methods. You can find all the necessary data [here](https://www.mql5.com/en/articles/5687).

Since I have introduced a new parameter — the time of obtaining a DOM snapshot, it will be set in the object of a DOM abstract order. The [MqlBookInfo](https://www.mql5.com/en/docs/constants/structures/mqlbookinfo) structure describing a DOM order features no time parameter. This means we need to describe a time of receiving a snapshot for each DOM order object on our own.

To achieve this, the file of the DOM abstract order class \\MQL5\\Include\\DoEasy\\Objects\\Book\ **MarketBookOrd.mqh** receives a new public method:

```
public:
//+-------------------------------------------------------------------+
//|Methods of a simplified access to the DOM request object properties|
//+-------------------------------------------------------------------+
//--- Set a snapshot time - all orders of a single DOM snapshot have the same name
   void              SetTime(const long time_msc)  { this.SetProperty(MBOOK_ORD_PROP_TIME_MSC,time_msc);                      }

//--- Return order (1) status, (2) type and (3) order volume
```

The method simply sets the obtained time value in milliseconds in the object's new property.

Initialize the time of obtaining a snapshot in the parametric constructor of the DOM abstract order class:

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CMarketBookOrd::CMarketBookOrd(const ENUM_MBOOK_ORD_STATUS status,const MqlBookInfo &book_info,const string symbol)
  {
//--- Save symbol’s Digits
   this.m_digits=(int)::SymbolInfoInteger(symbol,SYMBOL_DIGITS);
//--- Save integer object properties
   this.SetProperty(MBOOK_ORD_PROP_STATUS,status);
   this.SetProperty(MBOOK_ORD_PROP_TYPE,book_info.type);
   this.SetProperty(MBOOK_ORD_PROP_VOLUME,book_info.volume);
//--- Save real object properties
   this.SetProperty(MBOOK_ORD_PROP_PRICE,book_info.price);
   this.SetProperty(MBOOK_ORD_PROP_VOLUME_REAL,book_info.volume_real);
//--- Save additional object properties
   this.SetProperty(MBOOK_ORD_PROP_SYMBOL,(symbol==NULL || symbol=="" ? ::Symbol() : symbol));
//--- Order time is not present in the parameters and is considered in the DOM snapshot class. Reset the time
   this.SetProperty(MBOOK_ORD_PROP_TIME_MSC,0);
  }
//+------------------------------------------------------------------+
```

The time of each of the orders present in a single DOM snapshot is set at the moment of receiving the DOM snapshot.

Some minor changes have been made to the virtual object description methods in the **MarketBookOrd.mqh** DOM abstract order class file and its descendants **MarketBookBuy.mqh**, **MarketBookBuyMarket.mqh**, **MarketBookSell.mqh** and **MarketBookSellMarket.mqh**:

```
//--- Display a short description of the object in the journal
   virtual void      PrintShort(const bool symbol=false);
//--- Return the object short name
   virtual string    Header(const bool symbol=false);
```

Each of the methods has obtained the flags indicating the need to display a symbol name in the object description. By default, a symbol is not displayed in the order object description. The reason is that the order object is not independent, so to speak, but instead is a part of a DOM snapshot whose class I am to develop in the current article. Displaying a symbol for each order looks redundant when displaying the DOM snapshot object description together with descriptions of all orders since the symbol is already shown in the header of the DOM snapshot object description.

The refinement of these methods looks the same for all of the above classes.

**For the CMarketBookOrd class:**

```
//+------------------------------------------------------------------+
//| Return the object short name                                     |
//+------------------------------------------------------------------+
string CMarketBookOrd::Header(const bool symbol=false)
  {
   return this.TypeDescription()+(symbol ? " \""+this.Symbol()+"\"" : "");
  }
//+------------------------------------------------------------------+
//| Display a short description of the object in the journal         |
//+------------------------------------------------------------------+
void CMarketBookOrd::PrintShort(const bool symbol=false)
  {
   ::Print(this.Header(symbol));
  }
//+------------------------------------------------------------------+
```

For the **CMarketBookBuy**, **CMarketBookBuyMarket**, **CMarketBookSell** and **CMarketBookSellMarket** classes:

```
//+------------------------------------------------------------------+
//| Return the object short name                                     |
//+------------------------------------------------------------------+
string CMarketBookBuy::Header(const bool symbol=false)
  {
   return CMessage::Text(MSG_MBOOK_ORD_TYPE_BUY)+(symbol ? " \""+this.Symbol() : "")+
          ": "+::DoubleToString(this.Price(),this.Digits())+" ["+::DoubleToString(this.VolumeReal(),2)+"]";
  }
//+------------------------------------------------------------------+
```

For the **CMarketBookBuyMarket** class:

```
//+------------------------------------------------------------------+
//| Return the object short name                                     |
//+------------------------------------------------------------------+
string CMarketBookBuyMarket::Header(const bool symbol=false)
  {
   return CMessage::Text(MSG_MBOOK_ORD_TYPE_BUY_MARKET)+(symbol ? " \""+this.Symbol() : "")+
          ": "+::DoubleToString(this.Price(),this.Digits())+" ["+::DoubleToString(this.VolumeReal(),2)+"]";
  }
//+------------------------------------------------------------------+
```

For the **CMarketBookSell** class:

```
//+------------------------------------------------------------------+
//| Return the object short name                                     |
//+------------------------------------------------------------------+
string CMarketBookSell::Header(const bool symbol=false)
  {
   return CMessage::Text(MSG_MBOOK_ORD_TYPE_SELL)+(symbol ? " \""+this.Symbol() : "")+
          ": "+::DoubleToString(this.Price(),this.Digits())+" ["+::DoubleToString(this.VolumeReal(),2)+"]";
  }
//+------------------------------------------------------------------+
```

For the **CMarketBookSellMarket** class:

```
//+------------------------------------------------------------------+
//| Return the object short name                                     |
//+------------------------------------------------------------------+
string CMarketBookSellMarket::Header(const bool symbol=false)
  {
   return CMessage::Text(MSG_MBOOK_ORD_TYPE_SELL_MARKET)+(symbol ? " \""+this.Symbol() : "")+
          ": "+::DoubleToString(this.Price(),this.Digits())+" ["+::DoubleToString(this.VolumeReal(),2)+"]";
  }
//+------------------------------------------------------------------+
```

Accordingly, the flags were added to the declaration of all these methods in all descendant classes:

```
//--- Return the object short name
   virtual string    Header(const bool symbol=false);
```

By default, a symbol is not displayed in the object description.

### Depth of Market snapshot object class

Now all is ready for developing the DOM snapshot object class. In fact, this is a list of DOM requests passed to the MqlBookInfo structure array when the OnBookEvent() handler is activated. However, each of the array requests in the class is represented by the CMarketBookOrd class object — its descendants. They are all added to the CArrayObj list which is [a class of a dynamic array of pointers to the standard library's instances of the CObject class and its descendants](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj). Apart from the list storing the DOM request objects, the class is to provide the features for handling objects and their lists that are standard for all library objects — searching and sorting by their various properties — for convenient collection of any statistical data when working with the DOM in their programs.

In \\MQL5\\Include\\DoEasy\\Objects\ **Book\**, create the new file **MarketBookSnapshot.mqh** of the **CMBookSnapshot** class.

The class of the basic object of all CBaseObj library objects should be used as the base class.

The files of descendant object classes of the DOM abstract order object should be included into the file.

**Let's consider the class listing and implementation of its methods:**

```
//+------------------------------------------------------------------+
//|                                           MarketBookSnapshot.mqh |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\..\Services\Select.mqh"
#include "MarketBookBuy.mqh"
#include "MarketBookSell.mqh"
#include "MarketBookBuyMarket.mqh"
#include "MarketBookSellMarket.mqh"
//+------------------------------------------------------------------+
//| "DOM snapshot" class                                             |
//+------------------------------------------------------------------+
class CMBookSnapshot : public CBaseObj
  {
private:
   string            m_symbol;                  // Symbol
   long              m_time;                    // Snapshot time
   int               m_digits;                  // Symbol's Digits
   CArrayObj         m_list;                    // List of DOM order objects
public:
//--- Return (1) itself and (2) the list of DOM order objects
   CMBookSnapshot   *GetObject(void)                                    { return &this;   }
   CArrayObj        *GetList(void)                                      { return &m_list; }

//--- Return the list of DOM order objects by selected (1) double, (2) integer and (3) string property satisfying the compared condition
   CArrayObj        *GetList(ENUM_MBOOK_ORD_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL){ return CSelect::ByMBookProperty(this.GetList(),property,value,mode); }
   CArrayObj        *GetList(ENUM_MBOOK_ORD_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByMBookProperty(this.GetList(),property,value,mode); }
   CArrayObj        *GetList(ENUM_MBOOK_ORD_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL){ return CSelect::ByMBookProperty(this.GetList(),property,value,mode); }
//--- (1) Return the DOM order object by index in the list and (2) the order list size
   CMarketBookOrd   *GetMBookByListIndex(const uint index)              { return this.m_list.At(index);  }
   int               DataTotal(void)                              const { return this.m_list.Total();    }

//--- The comparison method for searching and sorting DOM snapshot objects by time
   virtual int       Compare(const CObject *node,const int mode=0) const
                       {
                        const CMBookSnapshot *compared_obj=node;
                        return(this.Time()<compared_obj.Time() ? -1 : this.Time()>compared_obj.Time() ? 1 : 0);
                       }

//--- Return the DOM snapshot change
   string            Header(void);
//--- Display (1) description and (2) short description of a DOM snapshot
   void              Print(void);
   void              PrintShort(void);

//--- Constructors
                     CMBookSnapshot(){;}
                     CMBookSnapshot(const string symbol,const long time,MqlBookInfo &book_array[]);
//+--------------------------------------------------------------------+
//|Methods of a simplified access to the DOM snapshot object properties|
//+--------------------------------------------------------------------+
//--- Set (1) a symbol, (2) a DOM snapshot time and (3) the specified time for all DOM orders
   void              SetSymbol(const string symbol)   { this.m_symbol=(symbol==NULL || symbol=="" ? ::Symbol() : symbol); }
   void              SetTime(const long time_msc)     { this.m_time=time_msc; }
   void              SetTimeToOrders(const long time_msc);
//--- Return (1) a DOM symbol, (2) symbol's Digits and (3) a snapshot time
   string            Symbol(void)               const { return this.m_symbol; }
   int               Digits(void)               const { return this.m_digits; }
   long              Time(void)                 const { return this.m_time;   }
  };
//+------------------------------------------------------------------+
```

Here we can see a usual arrangement of the class similar to all library objects: the class member variables are declared in the private section, while the public section features the standard methods of returning the lists by specified order object properties, the method of comparing two DOM snapshot objects for searching and sorting them in the list (the list of DOM snapshot series object which is to store the objects later on), the methods for describing the DOM snapshot object, as well as two constructors — the default and the parametric one (the parametric constructor is to be used when creating new DOM snapshot objects with all their properties known, while the default one is to be used for the fast creation of a new object and adding the required property for searching the objects in the list with the specified property value). The methods of a simplified access to object properties serve for setting and returning some object properties I will need later.

Let's have a look at the implementation of the class methods.

**In the parametric class constructor,** view the obtained MqlBookInfo structure array, create the appropriate types of DOM order objects and add them to the list.

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CMBookSnapshot::CMBookSnapshot(const string symbol,const long time,MqlBookInfo &book_array[]) : m_time(time)
  {
   //--- Set a symbol
   this.SetSymbol(symbol);
   //--- Clear the list
   this.m_list.Clear();
   //--- In the loop by the structure array
   int total=::ArraySize(book_array);
   for(int i=0;i<total;i++)
     {
      //--- Create order objects of the current DOM snapshot depending on the order type
      CMarketBookOrd *mbook_ord=NULL;
      switch(book_array[i].type)
        {
         case BOOK_TYPE_BUY         : mbook_ord=new CMarketBookBuy(this.m_symbol,book_array[i]);         break;
         case BOOK_TYPE_SELL        : mbook_ord=new CMarketBookSell(this.m_symbol,book_array[i]);        break;
         case BOOK_TYPE_BUY_MARKET  : mbook_ord=new CMarketBookBuyMarket(this.m_symbol,book_array[i]);   break;
         case BOOK_TYPE_SELL_MARKET : mbook_ord=new CMarketBookSellMarket(this.m_symbol,book_array[i]);  break;
         default: break;
        }
      if(mbook_ord==NULL)
         continue;
      //--- Set the DOM snapshot time for the order
      mbook_ord.SetTime(this.m_time);
      //--- Set the sorted list flag for the list (by the price value) and add the current order object to it
      this.m_list.Sort(SORT_BY_MBOOK_ORD_PRICE);
      if(!this.m_list.InsertSort(mbook_ord))
         delete mbook_ord;
     }
  }
//+------------------------------------------------------------------+
```

**The method returning a short name of the DOM snapshot object:**

```
//+------------------------------------------------------------------+
//| Return the object short name                                     |
//+------------------------------------------------------------------+
string CMBookSnapshot::Header(void)
  {
   return CMessage::Text(MSG_MBOOK_SNAP_TEXT_SNAPSHOT)+" \""+this.Symbol();
  }
//+------------------------------------------------------------------+
```

Here we simply create a string made of the text message with the object and symbol description approximately looking as follows:

```
EURUSD DOM snapshot
```

**The method displaying a short description of the DOM snapshot object in the journal:**

```
//+------------------------------------------------------------------+
//| Display a short description of the object in the journal         |
//+------------------------------------------------------------------+
void CMBookSnapshot::PrintShort(void)
  {
   ::Print(this.Header()," ("+TimeMSCtoString(this.m_time),")");
  }
//+------------------------------------------------------------------+
```

In the journal, the method prints a string consisting of the object name plus the DOM snapshot time in milliseconds, for example:

```
"EURUSD" DOM snapshot (2021.02.09 22:16:24.557)
```

**The method displaying the DOM snapshot object properties in the journal:**

```
//+------------------------------------------------------------------+
//| Display object properties in the journal                         |
//+------------------------------------------------------------------+
void CMBookSnapshot::Print(void)
  {
   ::Print(this.Header()," ("+TimeMSCtoString(this.m_time),"):");
   this.m_list.Sort(SORT_BY_MBOOK_ORD_PRICE);
   for(int i=this.m_list.Total()-1;i>WRONG_VALUE;i--)
     {
      CMarketBookOrd *ord=this.m_list.At(i);
      if(ord==NULL)
         continue;
      ::Print(" - ",ord.Header());
     }
  }
//+------------------------------------------------------------------+
```

The header featuring the DOM snapshot description is displayed first followed by the descriptions of all DOM order objects in a loop.

Since order objects in the DOM snapshot object list are unable to obtain their time, I will implement **the method setting the time in milliseconds specified in the object to all order objects in the list:**

```
//+------------------------------------------------------------------+
//| Set the specified time to all DOM orders                         |
//+------------------------------------------------------------------+
void CMBookSnapshot::SetTimeToOrders(const long time_msc)
  {
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CMarketBookOrd *ord=this.m_list.At(i);
      if(ord==NULL)
         continue;
      ord.SetTime(time_msc);
     }
  }
//+------------------------------------------------------------------+
```

In the loop by the list of all DOM order objects, get the next order object and assign the time specified in the DOM snapshot object properties to it. Thus, all order objects in the DOM list have the same time of their receipt. This is logical since we obtain them at the moment of the OnBookEvent() handler activation. The activation time is set for the DOM snapshot object and all its orders.

The DOM snapshot object is ready. Now it is time to place these objects in the list since we are going to obtain a new DOM snapshot and create the appropriate object at each activation of the OnBookEvent() handler.

All these objects are to be stored in the DOM series object class.

### Depth of Market snapshot series object class

In terms of its "ideology", the class of DOM snapshot series is similar to the symbol timeseries classes or tick data. In this classes, the data can be obtained from the environment, while in the DOM snapshot list class, we are unable to obtain historical data — it has to be accumulated in real time. Therefore, the class will not feature the list creation method but rather the list update method only.

In \\MQL5\\Include\\DoEasy\\Objects\ **Book\**, create the new file **MBookSeries.mqh** of the **CMBookSeries** class.

The class of the basic object of all CBaseObj library objects should be used as the base class.

The file of the DOM snapshot object class should be included in the file.

Let's consider the class listing and analyze its methods afterwards:

```
//+------------------------------------------------------------------+
//|                                                  MBookSeries.mqh |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "MarketBookSnapshot.mqh"
//+------------------------------------------------------------------+
//| "DOM snapshot series" class                                      |
//+------------------------------------------------------------------+
class CMBookSeries : public CBaseObj
  {
private:
   string            m_symbol;                                          // Symbol
   uint              m_amount;                                          // Number of used DOM snapshots in the series
   uint              m_required;                                        // Required number of days for DOM snapshot series
   CArrayObj         m_list;                                            // DOM snapshot series list
   MqlBookInfo       m_book_info[];                                     // DOM structure array
public:
//--- Return (1) itself and (2) the series list
   CMBookSeries     *GetObject(void)                                    { return &this;   }
   CArrayObj        *GetList(void)                                      { return &m_list; }

//--- Return the DOM snapshot list by selected (1) double, (2) integer and (3) string properties fitting the compared condition
   CArrayObj        *GetList(ENUM_MBOOK_ORD_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL){ return CSelect::ByMBookProperty(this.GetList(),property,value,mode); }
   CArrayObj        *GetList(ENUM_MBOOK_ORD_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByMBookProperty(this.GetList(),property,value,mode); }
   CArrayObj        *GetList(ENUM_MBOOK_ORD_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL){ return CSelect::ByMBookProperty(this.GetList(),property,value,mode); }

//--- Return the DOM snapshot object by (1) index in the list, (2) time and (3) actual list size
   CMBookSnapshot   *GetMBookByListIndex(const uint index)        const { return this.m_list.At(index);              }
   CMBookSnapshot   *GetLastMBook(void)                           const { return this.m_list.At(this.DataTotal()-1); }
   CMBookSnapshot   *GetMBook(const long time_msc);
   int               DataTotal(void)                              const { return this.m_list.Total();                }

//--- Set a (1) symbol, (2) a number of days for DOM snapshots
   void              SetSymbol(const string symbol);
   void              SetRequiredUsedDays(const uint required=0);

//--- The comparison method for searching and sorting DOM snapshot series objects by symbol
   virtual int       Compare(const CObject *node,const int mode=0) const
                       {
                        const CMBookSeries *compared_obj=node;
                        return(this.Symbol()==compared_obj.Symbol() ? 0 : this.Symbol()>compared_obj.Symbol() ? 1 : -1);
                       }
//--- Return the name of the DOM  snapshot series
   string            Header(void);
//--- Display (1) description and (2) short description of a DOM snapshot series
   void              Print(void);
   void              PrintShort(void);

//--- Constructors
                     CMBookSeries(){;}
                     CMBookSeries(const string symbol,const uint required=0);

//+------------------------------------------------------------------+
//| Methods of working with objects and accessing their properties   |
//+------------------------------------------------------------------+
//--- Return (1) a symbol, a number of (2) used and (3) requested DOM snapshots in the series and
//--- (4) the time of a DOM snapshot specified by the index in milliseconds
   string            Symbol(void)                                          const { return this.m_symbol;    }
   ulong             AvailableUsedData(void)                               const { return this.m_amount;    }
   ulong             RequiredUsedDays(void)                                const { return this.m_required;  }
   long              MBookTime(const int index) const;
//--- update the list of DOM snapshot series
   bool              Refresh(const long time_msc);
  };
//+------------------------------------------------------------------+
```

Here we can see:

- standard methods of receiving object properties and object lists by specified properties,
- methods of placing some properties,
- the method of comparing two list objects for sorting them only by a symbol name,
- methods of returning object names and class constructors.

Let's have a look at the implementation of the class methods.

The list symbol is set in the initialization list **of the parametric constructor**, while the list is cleared in the method body. The flag of a list sorted by time in milliseconds is set for it and the required number of DOM snapshot data days is specified.

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CMBookSeries::CMBookSeries(const string symbol,const uint required=0) : m_symbol(symbol)
  {
   this.m_list.Clear();
   this.m_list.Sort(SORT_BY_MBOOK_ORD_TIME_MSC);
   this.SetRequiredUsedDays(required);
  }
//+------------------------------------------------------------------+
```

**The method updating the DOM snapshot list:**

```
//+------------------------------------------------------------------+
//| Update the list of DOM snapshot series                           |
//+------------------------------------------------------------------+
bool CMBookSeries::Refresh(const long time_msc)
  {
//--- Get DOM entries to the structure array
   if(!::MarketBookGet(this.m_symbol,this.m_book_info))
      return false;
//--- Create a new DOM snapshot object
   CMBookSnapshot *book=new CMBookSnapshot(this.m_symbol,time_msc,this.m_book_info);
   if(book==NULL)
      return false;
//--- Set the flag of a list sorted by time for the list and add the created DOM snapshot to it
   this.m_list.Sort(SORT_BY_MBOOK_ORD_TIME_MSC);
   if(!this.m_list.InsertSort(book))
     {
      delete book;
      return false;
     }
//--- Set time in milliseconds to all DOM snapshot order objects
   book.SetTimeToOrders(time_msc);
//--- If the number of snapshots in the list exceeds the default maximum number,
//--- remove the calculated number of snapshot objects from the end of the list
   if(this.DataTotal()>MBOOKSERIES_MAX_DATA_TOTAL)
     {
      int total_del=this.m_list.Total()-MBOOKSERIES_MAX_DATA_TOTAL;
      for(int i=0;i<total_del;i++)
         this.m_list.Delete(i);
     }
   return true;
  }
//+------------------------------------------------------------------+
```

The method is called when the OnBookEvent() handler is activated. The method receives the handler activation time. Use [MarketBookGet()](https://www.mql5.com/en/docs/marketinformation/marketbookget) to obtain the DOM structure array. Use the structure to create a new DOM snapshot objects and add it to the snapshot series list.

The entire logic is detailed in the comments to the method code. I believe, it is clear.

**The method for placing a symbol name to the snapshot series list:**

```
//+------------------------------------------------------------------+
//| Set a symbol                                                     |
//+------------------------------------------------------------------+
void CMBookSeries::SetSymbol(const string symbol)
  {
   if(this.m_symbol==symbol)
      return;
   this.m_symbol=(symbol==NULL || symbol=="" ? ::Symbol() : symbol);
  }
//+------------------------------------------------------------------+
```

Here all is transparent. If NULL or an empty string is passed, the current symbol is set. Otherwise, set the one passed to the method.

**The method defining the required number of days for DOM snapshot series:**

```
//+------------------------------------------------------------------+
//| Set the number of days for DOM snapshots in the series           |
//+------------------------------------------------------------------+
void CMBookSeries::SetRequiredUsedDays(const uint required=0)
  {
   this.m_required=(required<1 ? MBOOKSERIES_DEFAULT_DAYS_COUNT : required);
  }
//+------------------------------------------------------------------+
```

If a zero value is passed, set the default number of days, otherwise, set the one passed to the method. The method is not used anywhere yet.

**The method returning the DOM snapshot object by the specified time:**

```
//+------------------------------------------------------------------+
//| Return the DOM snapshot object by its time                       |
//+------------------------------------------------------------------+
CMBookSnapshot *CMBookSeries::GetMBook(const long time_msc)
  {
   CMBookSnapshot *book=new CMBookSnapshot();
   if(book==NULL)
      return NULL;
   book.SetTime(time_msc);
   this.m_list.Sort();
   int index=this.m_list.Search(book);
   delete book;
   return this.m_list.At(index);
  }
//+------------------------------------------------------------------+
```

Here we create a temporary DOM snapshot object, set the required time to it, as well as the sorted list flag and use the [Search()](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjsearch) method to receive the object index in the list with the necessary time. Make sure to delete the temporary object and return the pointer to the detected object by index in the list.

**The method returning the time of the DOM snapshot specified by index in milliseconds:**

```
//+------------------------------------------------------------------+
//| Returns the time in milliseconds                                 |
//| of a DOM snapshot specified by index                             |
//+------------------------------------------------------------------+
long CMBookSeries::MBookTime(const int index) const
  {
   CMBookSnapshot *book=this.m_list.At(index);
   return(book!=NULL ? book.Time() : 0);
  }
//+------------------------------------------------------------------+
```

Get the pointer to the DOM snapshot object by the specified index and return its time in milliseconds or NULL in case of a failure.

**The method returning the name of the DOM snapshot series:**

```
//+------------------------------------------------------------------+
//| Return the name of the DOM snapshot series                       |
//+------------------------------------------------------------------+
string CMBookSeries::Header(void)
  {
   return CMessage::Text(MSG_MBOOK_SERIES_TEXT_MBOOKSERIES)+" \""+this.m_symbol+"\"";
  }
//+------------------------------------------------------------------+
```

The method returns the string consisting of the description of an object and symbol, for example:

```
Series of "EURUSD" DOM snapshots
```

**The method displaying the description of the DOM snapshot series in the journal:**

```
//+------------------------------------------------------------------+
//| Display the description of the DOM snapshot series in the journal|
//+------------------------------------------------------------------+
void CMBookSeries::Print(void)
  {
   string txt=
     (
      CMessage::Text(MSG_TICKSERIES_REQUIRED_HISTORY_DAYS)+(string)this.RequiredUsedDays()+", "+
      CMessage::Text(MSG_LIB_TEXT_TS_ACTUAL_DEPTH)+(string)this.DataTotal()
     );
   ::Print(this.Header(),": ",txt);
  }
//+------------------------------------------------------------------+
```

The header featuring the snapshot series description, a requested number of data days and a number of the actually collected DOM snapshots is created first. Then all orders of the snapshot object are displayed in the loop.

**This completes the creation of the DOM snapshot object and object series classes.**

### Test

To perform the test, let's use the EA from the previous article and save it in \\MQL5\\Experts\\TestDoEasy\ **Part64\** as **TestDoEasyPart64.mq5**.

In the EA, create the DOM snapshot series object for the current symbol and add a new DOM snapshot object at each activation of the OnBoolEvent() handler on the current symbol. Display data on the number of snapshot objects added to the list and two extreme orders of the current snapshot (the highest sell and the lowest buy ones) in the chart comment. When receiving DOM data for the first time, print it in the terminal journal.

Remove connecting order object classes from the EA listing — they are now included into the new class files I have created today:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart63.mq5 |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
#include <DoEasy\Objects\Book\MarketBookBuy.mqh>
#include <DoEasy\Objects\Book\MarketBookSell.mqh>
#include <DoEasy\Objects\Book\MarketBookBuyMarket.mqh>
#include <DoEasy\Objects\Book\MarketBookSellMarket.mqh>
//--- enums
```

Instead, add the inclusion of the DOM snapshot series class file:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart64.mq5 |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
#include <DoEasy\Objects\Book\MBookSeries.mqh>
//--- enums
```

In the list of the EA global variables, declare the object of the DOM snapshot series class:

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

The entire work on creating the list of DOM snapshot series is performed in the OnBoolEvent() handler:

```
//+------------------------------------------------------------------+
//| OnBookEvent function                                             |
//+------------------------------------------------------------------+
void OnBookEvent(const string& symbol)
  {
   static bool first=true;
   //--- Get a symbol object
   CSymbol *sym=engine.GetSymbolCurrent();
   //--- If failed to get a symbol object or it is not subscribed to DOM, exit
   if(sym==NULL || !sym.BookdepthSubscription()) return;
   //--- Work by the current symbol
   if(symbol==sym.Name())
     {
      //--- Set a symbol and a required number of data days for the DOM snapshot series object
      book_series.SetSymbol(sym.Name());
      book_series.SetRequiredUsedDays();
      //--- Update the DOM snapshot series
      if(!book_series.Refresh(sym.Time()))
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
         DFUN,sym.Name(),": ",TimeMSCtoString(book.Time()),
         ", symbol book size=",sym.TicksBookdepth(),

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

All code strings are described in the comments here. I hope, they do not need additional explanation.

If you have any questions, feel free to ask them in the comments.

Compile the EA and launch it on a symbol chart having preliminarily defined in the settings the work on two specified symbols and the current timeframe:

![](https://c.mql5.com/2/42/TPKdIv2dTG.png)

The journal displays data on the created DOM snapshot series and the very first snapshot:

```
Account 8550475: Artyom Trishkin (MetaQuotes Software Corp.) 10428.13 USD, 1:100, Hedge, MetaTrader 5 demo
--- Initializing "DoEasy" library ---
Working with predefined symbol list. The number of used symbols: 2
"AUDUSD" "EURUSD"
Working with the current timeframe only: H1
AUDUSD symbol timeseries:
- Timeseries "AUDUSD" H1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5121
EURUSD symbol timeseries:
- Timeseries "EURUSD" H1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 6046
Tick series "AUDUSD": Requested number of days: 1, Historical data created: 176033
Tick series "EURUSD": Requested number of days: 1, Historical data created: 181969
Subscribed to Depth of Market  AUDUSD
Subscribed to Depth of Market  EURUSD
Library initialization time: 00:00:12.516
The "EURUSD" DOM snapshot series: Requested number of days: 1, Actual history depth: 1
"EURUSD" DOM snapshot (2021.02.09 22:16:24.557):
 - Sell order: 1.21198 [250.00]
 - Sell order: 1.21193 [100.00]
 - Sell order: 1.21192 [50.00]
 - Sell order: 1.21191 [30.00]
 - Sell order: 1.21190 [6.00]
 - Buy order: 1.21188 [36.00]
 - Buy order: 1.21186 [50.00]
 - Buy order: 1.21185 [100.00]
 - Buy order: 1.21180 [250.00]
```

The number of the last DOM snapshot, the number of orders for a symbol, the number of orders in the current snapshot and the total number of DOM snapshots added to the DOM snapshot list are to be displayed in a symbol chart:

![](https://c.mql5.com/2/42/terminal64_W8obhOfUF9.png)

The figure displays data on the EA that has already been working for some time (5019 snapshots have been added to the list)

### What's next?

In the next article, I will create the collection of DOM snapshot series allowing users to fully work with DOMs of any symbols having active subscription to the DOM and enabled broadcast.

All files of the current version of the library are attached below together with the test EA file for MQL5 for you to test and download.

The classes for working with DOM are under development, therefore their use in custom programs at this stage is strongly not recommended.

Leave your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/9044#node00)

**\*Previous articles within the series:**

[Prices in DoEasy library (Part 62): Updating tick series in real time, preparation for working with Depth of Market](https://www.mql5.com/en/articles/8988)

[Prices in DoEasy library (Part 63): Depth of Market and its abstract request class](https://www.mql5.com/en/articles/9010)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/9044](https://www.mql5.com/ru/articles/9044)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/9044.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/9044/mql5.zip "Download MQL5.zip")(3897 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/367161)**
(6)


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
16 Apr 2021 at 19:45

**Christian :**

Very diligent Artyom !

This seems to be a life's work 👌😊.

I assume that the libraries you have developed will be integrated into the Metatrader5 [standard library](https://www.mql5.com/en/docs/standardlibrary "Reference book MQL5: Standard library") at some point.

When will that be?

Thank you :)

I don't think so. It's just a separate project and doesn't fit the concept of the standard library.

![Christian](https://c.mql5.com/avatar/2016/3/56F90C3B-A503.gif)

**[Christian](https://www.mql5.com/en/users/collider)**
\|
16 Apr 2021 at 20:09

**Artyom Trishkin:**

Thank you :)

I don't think so. It's just a separate project and doesn't fit the concept of the standard library.

You derive everything from CObject. Where are the conceptual differences to the [standard library](https://www.mql5.com/en/docs/standardlibrary "Reference book MQL5 : Standard Library")?

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
16 Apr 2021 at 20:24

**Christian :**

You derive everything from CObject. Where are the conceptual differences to the [standard library](https://www.mql5.com/en/docs/standardlibrary "Reference book MQL5: Standard library")?

The MQL5 standard library is written in the MQL5 language and is intended to make it easier for end users to write programmes (indicators, scripts, expert advisors). The library provides convenient access to most of the internal functions of MQL5.

My library is created using the tools provided by the standard library - this is already an add-on to it.

![Christian](https://c.mql5.com/avatar/2016/3/56F90C3B-A503.gif)

**[Christian](https://www.mql5.com/en/users/collider)**
\|
16 Apr 2021 at 20:30

**Artyom Trishkin:**

The MQL5 standard library is written in the MQL5 language and is designed to make it easier for end users to write programmes (indicators, scripts, expert advisors). The library provides convenient access to most of MQL5's internal functions.

My library is created using the tools provided by the standard library - this is already an add-on to it.

Ok, I see.

I still hope that your work will be permanently integrated at some point.

There are some real gems in there.

Keep up the good work!

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
16 Apr 2021 at 20:37

**Christian :**

Ok, I see.

I still hope that your work will be permanently integrated at some point.

There are some real gems in there.

Keep up the good work!

Thank you


![Prices and Signals in DoEasy library (Part 65): Depth of Market collection and the class for working with MQL5.com Signals](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__3.png)[Prices and Signals in DoEasy library (Part 65): Depth of Market collection and the class for working with MQL5.com Signals](https://www.mql5.com/en/articles/9095)

In this article, I will create the collection class of Depths of Market of all symbols and start developing the functionality for working with the MQL5.com Signals service by creating the signal object class.

![Machine learning in Grid and Martingale trading systems. Would you bet on it?](https://c.mql5.com/2/42/yandex_catboost__3.png)[Machine learning in Grid and Martingale trading systems. Would you bet on it?](https://www.mql5.com/en/articles/8826)

This article describes the machine learning technique applied to grid and martingale trading. Surprisingly, this approach has little to no coverage in the global network. After reading the article, you will be able to create your own trading bots.

![Neural networks made easy (Part 12): Dropout](https://c.mql5.com/2/48/Neural_networks_made_easy_012.png)[Neural networks made easy (Part 12): Dropout](https://www.mql5.com/en/articles/9112)

As the next step in studying neural networks, I suggest considering the methods of increasing convergence during neural network training. There are several such methods. In this article we will consider one of them entitled Dropout.

![Self-adapting algorithm (Part IV): Additional functionality and tests](https://c.mql5.com/2/41/50_percents__4.png)[Self-adapting algorithm (Part IV): Additional functionality and tests](https://www.mql5.com/en/articles/8859)

I continue filling the algorithm with the minimum necessary functionality and testing the results. The profitability is quite low but the articles demonstrate the model of the fully automated profitable trading on completely different instruments traded on fundamentally different markets.

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/9044&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083149251782907376)

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