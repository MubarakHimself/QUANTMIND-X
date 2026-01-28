---
title: Library for easy and quick development of MetaTrader programs (part III). Collection of market orders and positions, search and sorting
url: https://www.mql5.com/en/articles/5687
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:42:44.563296
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/5687&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070501526578861825)

MetaTrader 5 / Examples


### Contents

[Arranging the search](https://www.mql5.com/en/articles/5687#node01)

[Engine base object is the library core](https://www.mql5.com/en/articles/5687#node02)

[Objects of active market orders and positions](https://www.mql5.com/en/articles/5687#node03)

[Collection of active market orders and positions](https://www.mql5.com/en/articles/5687#node04)

[What's next](https://www.mql5.com/en/articles/5687#node05)

In [the first part of the article series](https://www.mql5.com/en/articles/5654), we started creating a large cross-platform library simplifying the development of programs for [MetaTrader 5](https://www.metaquotes.net/en/metatrader5 "https://www.metaquotes.net/en/metatrader5") and MetaTrader 4 platforms.

In the [second part](https://www.mql5.com/en/articles/5669), we resumed the development of the library and implemented the collection of historical orders and deals.

Here we are going to create a class for a convenient selection and sorting of orders, deals and positions in collection lists, implement the base library object called Engine and add collection of market orders and positions to the library.

At the moment, a certain data storage structure is already emerging. We are going to adhere to it when creating collections of various object types:

![](https://c.mql5.com/2/36/Struct_eng.png)

A single Engine object will be created for storing and managing collections, as well as for exchanging data between the program and the library. Engine is to become the base object of the entire library. Programs based on the library are to refer to it to obtain data. Besides, it is to accumulate the entire library automation.

### Arranging the search

For easy and convenient use of data from library collections, we are going to implement a convenient data search, sorting and display on request. To achieve this, let's create a special class and name it **CSelect**.

All data requests are to pass through it.

In the **Collections** library folder, create the new **CSelect** class. There is no need to set the base class. After completion of the MQL Wizard operation, the new Select.mqh file is generated in the Collections folder:

```
//+------------------------------------------------------------------+
//|                                                       Select.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CSelect
  {
private:

public:
                     CSelect();
                    ~CSelect();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CSelect::CSelect()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CSelect::~CSelect()
  {
  }
//+------------------------------------------------------------------+
```

To perform the search, set all its modes. To do that, create the enumeration describing object comparison modes during the search. The enumeration is to be created in the Defines.mqh file:

```
//+------------------------------------------------------------------+
//|                                                      Defines.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
#define COUNTRY_LANG   ("Russian")              // Country language
#define DFUN           (__FUNCTION__+": ")      // "Function description"
#define END_TIME       (D'31.12.3000 23:59:59') // Final data for requesting account history data
//+------------------------------------------------------------------+
//| Search                                                           |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Search data                                                      |
//+------------------------------------------------------------------+
enum ENUM_COMPARER_TYPE
  {
   EQUAL,                                                   // Equal
   MORE,                                                    // More
   LESS,                                                    // Less
   NO_EQUAL,                                                // Not equal
   EQUAL_OR_MORE,                                           // Equal or more
   EQUAL_OR_LESS                                            // Equal or less
  };
//+------------------------------------------------------------------+
```

In the CSelect class file, from thestandard library, connect the [class](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) [of the list of dynamic pointers to object instances](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj), COrder class and the library of DELib.mqh service functions (together with the Defines.mqh file). Besides, declare a special storage object available to the entire library on the global level. It is to store the copies of lists created during sorting. If newly created lists are not attached to a storage object, they should be deleted after the need for them disappears. This means we need to allocate extra resources on tracking where, when and why we needed a certain list that may cause a loss of a logic chain and memory leaks due to undeleted objects. If we attach them to the list, the tracking is performed by the terminal subsystem, which always deletes both the storage list and its contents in time.

To make a comparison, we need to create such a method. Let's declare a template static method for comparing two values.

```
//+------------------------------------------------------------------+
//|                                                       Select.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Arrays\ArrayObj.mqh>
#include "..\Objects\Order.mqh"
//+------------------------------------------------------------------+
//| Storage list                                                     |
//+------------------------------------------------------------------+
CArrayObj   ListStorage; // Storage object for storing sorted collection lists
//+------------------------------------------------------------------+
//| Class for sorting objects meeting the criterion                  |
//+------------------------------------------------------------------+
class CSelect
  {
private:
   //--- Method for comparing two values
   template<typename T>
   static bool       CompareValues(T value1,T value2,ENUM_COMPARER_TYPE mode);
public:

  };
//+------------------------------------------------------------------+
```

**Implementing the comparison method:**

```
//+------------------------------------------------------------------+
//| Method of comparing two values                                   |
//+------------------------------------------------------------------+
template<typename T>
bool CSelect::CompareValues(T value1,T value2,ENUM_COMPARER_TYPE mode)
  {
   return
     (
      mode==EQUAL && value1==value2          ?  true  :
      mode==NO_EQUAL && value1!=value2       ?  true  :
      mode==MORE && value1>value2            ?  true  :
      mode==LESS && value1<value2            ?  true  :
      mode==EQUAL_OR_MORE && value1>=value2  ?  true  :
      mode==EQUAL_OR_LESS && value1<=value2  ?  true  :  false
     );
  }
//+------------------------------------------------------------------+
```

Two valuesof the same type and the comparison mode are passed to the method.

Next, a simple comparison depending on the applied method is made (equal/not equal/more/less/equal or more/equal or less) and the result is returned.

Now let's create several methods for searching the list. In the public section of the CSelect class, declare three static methods for searching for an order by a given criterion:

```
//+------------------------------------------------------------------+
//| Class for sorting objects meeting the criterion                  |
//+------------------------------------------------------------------+
class CSelect
  {
private:
   //--- Two values comparison method
   template<typename T>
   static bool       CompareValues(T value1,T value2,ENUM_COMPARER_TYPE mode);
public:
   //--- Return the list of orders with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
  };
//+------------------------------------------------------------------+
```

Let's implement them immediately outside the class body:

```
//+------------------------------------------------------------------+
//| Return the list of orders having one of the integer              |
//| properties meeting a specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   int total=list_source.Total();
   for(int i=0; i<total; i++)
     {
      COrder *order=list_source.At(i);
      if(!order.SupportProperty(property)) continue;
      long order_prop=order.GetProperty(property);
      if(CompareValues(order_prop,value,mode)) list.Add(order);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of orders having one of the real                 |
//| properties meeting a specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      COrder *order=list_source.At(i);
      if(!order.SupportProperty(property)) continue;
      double order_prop=order.GetProperty(property);
      if(CompareValues(order_prop,value,mode)) list.Add(order);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of orders having one of the string               |
//| properties meeting a specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      COrder *order=list_source.At(i);
      if(!order.SupportProperty(property)) continue;
      string order_prop=order.GetProperty(property);
      if(CompareValues(order_prop,value,mode)) list.Add(order);
     }
   return list;
  }
//+------------------------------------------------------------------+
```

Let's have a closer look using the search by string criteria as an example:

- The method receives the pointer to the collection list, a property the new list is to be based on and the comparison mode. The property should meet the search criterion.
- After that, the list is checked for validity, and NULL is returned if it is invalid.
- If the check is passed, create a new list object and set the manual memory management flag for it. If this is not done, then when this list object is deleted, all pointers to the order objects stored in the collection are deleted as well, which is unacceptable. You can find the details in the help on the dynamic list of pointers, in particular, [this method](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjfreemodeconst).
- Next, add the newly created list to the storage list and start sorting objects in a loop:
  - get an order from the list. If it does not support a property specified in the search, skip it and select the next one.
  - Next, the order property is compared with the property passed for comparing according to the specified mode (more/less/equal, etc.). If the comparison criterion is met, this order is added to a newly created list.
  - At the end of the loop, the list is returned to the calling program.

Add another six methods for searching and returning an order index with the maximum and minimum value of the specified property:

```
//+------------------------------------------------------------------+
//| Class for sorting objects meeting the criterion                  |
//+------------------------------------------------------------------+
class CSelect
  {
private:
   //--- Two values comparison method
   template<typename T>
   static bool       CompareValues(T value1,T value2,ENUM_COMPARER_TYPE mode);
public:
   //--- Return the list of orders with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the order index with the maximum value of the (1) integer, (2) real and (3) string properties
   static int        FindOrderMax(CArrayObj* list_source,ENUM_ORDER_PROP_INTEGER property);
   static int        FindOrderMax(CArrayObj* list_source,ENUM_ORDER_PROP_DOUBLE property);
   static int        FindOrderMax(CArrayObj* list_source,ENUM_ORDER_PROP_STRING property);
   //--- Return the order index with the minimum value of the (1) integer, (2) real and (3) string properties
   static int        FindOrderMin(CArrayObj* list_source,ENUM_ORDER_PROP_INTEGER property);
   static int        FindOrderMin(CArrayObj* list_source,ENUM_ORDER_PROP_DOUBLE property);
   static int        FindOrderMin(CArrayObj* list_source,ENUM_ORDER_PROP_STRING property);
  };
//+------------------------------------------------------------------+
```

and their implementation:

```
//+------------------------------------------------------------------+
//| Return the order index in the list                               |
//| with the maximum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindOrderMax(CArrayObj *list_source,ENUM_ORDER_PROP_INTEGER property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   COrder *max_order=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      COrder *order=list_source.At(i);
      long order1_prop=order.GetProperty(property);
      max_order=list_source.At(index);
      long order2_prop=max_order.GetProperty(property);
      if(CompareValues(order1_prop,order2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the order index in the list                               |
//| with the maximum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindOrderMax(CArrayObj *list_source,ENUM_ORDER_PROP_DOUBLE property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   COrder *max_order=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      COrder *order=list_source.At(i);
      double order1_prop=order.GetProperty(property);
      max_order=list_source.At(index);
      double order2_prop=max_order.GetProperty(property);
      if(CompareValues(order1_prop,order2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the order index in the list                               |
//| with the maximum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindOrderMax(CArrayObj *list_source,ENUM_ORDER_PROP_STRING property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   COrder *max_order=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      COrder *order=list_source.At(i);
      string order1_prop=order.GetProperty(property);
      max_order=list_source.At(index);
      string order2_prop=max_order.GetProperty(property);
      if(CompareValues(order1_prop,order2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
```

Let's have a closer look using the search for the order index with the maximum string value:

- The method receives the pointer to the collection list and the property for searching for an order with the maximum value of that property.
- After that, the list is checked for validity, and WRONG\_VALUE (-1) is returned if it is invalid.
- Declare the order index with the maximum value, initialize it with zero and create an empty order object that is to store comparison values.
- Next, move along the collection list from the second order in a loop:
  - Get the target value with the loop index from the order, get the target value with the 'index' index from the order and compare two obtained values. If the target value of the first order (with the loop index) exceeds the second order's one (with the 'index' index), assign the index of the order having the **g** reater value to the 'index' variable.
  - Upon the loop's completion, the 'index' variable will have the index of the order having the highest target value. Return it to the calling program.

**Methods returning the order index with the lowest value of the specified property are arranged similarly:**

```
//+------------------------------------------------------------------+
//| Return the order index in the list                               |
//| with the minimum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindOrderMin(CArrayObj* list_source,ENUM_ORDER_PROP_INTEGER property)
  {
   int index=0;
   COrder* min_order=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++){
      COrder* order=list_source.At(i);
      long order1_prop=order.GetProperty(property);
      min_order=list_source.At(index);
      long order2_prop=min_order.GetProperty(property);
      if(CompareValues(order1_prop,order2_prop,LESS)) index=i;
      }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the order index in the list                               |
//| with the minimum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindOrderMin(CArrayObj* list_source,ENUM_ORDER_PROP_DOUBLE property)
  {
   int index=0;
   COrder* min_order=NULL;
   int total=list_source.Total();
   if(total== 0) return WRONG_VALUE;
   for(int i=1; i<total; i++){
      COrder* order=list_source.At(i);
      double order1_prop=order.GetProperty(property);
      min_order=list_source.At(index);
      double order2_prop=min_order.GetProperty(property);
      if(CompareValues(order1_prop,order2_prop,LESS)) index=i;
      }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the order index in the list                               |
//| with the minimum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindOrderMin(CArrayObj* list_source,ENUM_ORDER_PROP_STRING property)
  {
   int index=0;
   COrder* min_order=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++){
      COrder* order=list_source.At(i);
      string order1_prop=order.GetProperty(property);
      min_order=list_source.At(index);
      string order2_prop=min_order.GetProperty(property);
      if(CompareValues(order1_prop,order2_prop,LESS)) index=i;
      }
   return index;
  }
//+------------------------------------------------------------------+
```

**Now we can add sorting the collection list by time and specified criteria to the collection class of historical orders.**

First, add options of sorting by time to the Defines.mqh file:

```
//+------------------------------------------------------------------+
//|                                                      Defines.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"

//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
#define COUNTRY_LANG   ("Russian")              // Country language
#define DFUN           (__FUNCTION__+": ")      // "Function description"
#define END_TIME       (D'31.12.3000 23:59:59') // Final data for requesting account history data
//+------------------------------------------------------------------+
//| Search                                                           |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Search and sorting data                                          |
//+------------------------------------------------------------------+
enum ENUM_COMPARER_TYPE
  {
   EQUAL,                                                   // Equal
   MORE,                                                    // More
   LESS,                                                    // Less
   NO_EQUAL,                                                // Not equal
   EQUAL_OR_MORE,                                           // Equal or more
   EQUAL_OR_LESS                                            // Equal or less
  };
//+------------------------------------------------------------------+
//| Possible options of selecting by time                            |
//+------------------------------------------------------------------+
enum ENUM_SELECT_BY_TIME
  {
   SELECT_BY_TIME_OPEN,                                     // By open time
   SELECT_BY_TIME_CLOSE,                                    // By close time
   SELECT_BY_TIME_OPEN_MSC,                                 // By open time in milliseconds
   SELECT_BY_TIME_CLOSE_MSC,                                // By close time in milliseconds
  };
//+------------------------------------------------------------------+
```

Include the CSelect class to the HistoryCollection.mqh file. To do this, replace the service functions inclusion string with the CSelect class file inclusion one:

```
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Arrays\ArrayObj.mqh>
#include "..\DELib.mqh"
#include "..\Objects\HistoryOrder.mqh"
#include "..\Objects\HistoryPending.mqh"
#include "..\Objects\HistoryDeal.mqh"
//+------------------------------------------------------------------+
```

```
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Arrays\ArrayObj.mqh>
#include "Select.mqh"
#include "..\Objects\HistoryOrder.mqh"
#include "..\Objects\HistoryPending.mqh"
#include "..\Objects\HistoryDeal.mqh"
//+------------------------------------------------------------------+
```

Now, we have included the CSelect class file instead of the service functions one. We have included Order.mqh to Select.mqh, while the service functions file is already included to Order.mqh file.

In the public section of the CHistoryCollection class, declare the method of selecting orders from the collection by the specified time in the date range, while in the private section, add the COrder abstract order class which is to act as a sample order for searching for values:


```
//+------------------------------------------------------------------+
//| Collection of historical orders and deals                        |
//+------------------------------------------------------------------+
class CHistoryCollection
  {
private:
   CArrayObj         m_list_all_orders;      // List of all historical orders and deals
   COrder            m_order_instance;       // Order object to search by a property
   bool              m_is_trade_event;       // Trading event flag
   int               m_index_order;          // Index of the last order added to the collection from the terminal history list (MQL4, MQL5)
   int               m_index_deal;           // Index of the last deal added to the collection from the terminal history list (MQL5)
   int               m_delta_order;          // Difference in the number of orders as compared to the past check
   int               m_delta_deal;           // Difference in the number of deals as compared to the past check
public:
   //--- Return the full collection list 'as is'
   CArrayObj        *GetList(void) { return &m_list_all_orders;  }
   //--- Select orders from the collection with time from begin_time to end_time
   CArrayObj        *GetListByTime(const datetime begin_time=0,const datetime end_time=0,
                                   const ENUM_SELECT_BY_TIME select_time_mode=SELECT_BY_TIME_CLOSE);
   //--- Constructor
                     CHistoryCollection();
   //--- Update the list of orders, fill data on the number of new ones and set the trading event flag
   void              Refresh(void);
  };
//+------------------------------------------------------------------+
```

Before implementing the method for selecting orders from the collection within the range of dates, add the methods of placing integer, real and string properties to the COrder abstract order class in the Order.mqh file (in this case, the methods of writing the integer property are required to add parameters to the sample time data order):

```
public:
   //--- Set (1) integer, (2) real and (3) string order properties
   void              SetProperty(ENUM_ORDER_PROP_INTEGER property,long value) { m_long_prop[property]=value;                     }
   void              SetProperty(ENUM_ORDER_PROP_DOUBLE property,long value)  { m_long_prop[property]=value;                     }
   void              SetProperty(ENUM_ORDER_PROP_STRING property,long value)  { m_long_prop[property]=value;                     }
   //--- Return (1) integer, (2) real and (3) string order properties from the properties array
   long              GetProperty(ENUM_ORDER_PROP_INTEGER property)      const { return m_long_prop[property];                    }
   double            GetProperty(ENUM_ORDER_PROP_DOUBLE property)       const { return m_double_prop[this.IndexProp(property)];  }
   string            GetProperty(ENUM_ORDER_PROP_STRING property)       const { return m_string_prop[this.IndexProp(property)];  }

   //--- Return the flag of the order supporting the property
   virtual bool      SupportProperty(ENUM_ORDER_PROP_INTEGER property)        { return true; }
   virtual bool      SupportProperty(ENUM_ORDER_PROP_DOUBLE property)         { return true; }
   virtual bool      SupportProperty(ENUM_ORDER_PROP_STRING property)         { return true; }

   //--- Compare COrder objects with one another by all possible properties
   virtual int       Compare(const CObject *node,const int mode=0) const;

//+------------------------------------------------------------------+
```

In the HistoryCollection.mqh file, **implement the method of selecting orders from the collection within the range of dates**:

```
//+------------------------------------------------------------------+
//| Select orders from the collection                                |
//| from begin_time to end_time                                      |
//+------------------------------------------------------------------+
CArrayObj *CHistoryCollection::GetListByTime(const datetime begin_time=0,const datetime end_time=0,
                                             const ENUM_SELECT_BY_TIME select_time_mode=SELECT_BY_TIME_CLOSE)
  {
   ENUM_ORDER_PROP_INTEGER property=
     (
      select_time_mode==SELECT_BY_TIME_CLOSE       ?  ORDER_PROP_TIME_CLOSE      :
      select_time_mode==SELECT_BY_TIME_OPEN        ?  ORDER_PROP_TIME_OPEN       :
      select_time_mode==SELECT_BY_TIME_CLOSE_MSC   ?  ORDER_PROP_TIME_CLOSE_MSC  :
      ORDER_PROP_TIME_OPEN_MSC
     );

   CArrayObj *list=new CArrayObj();
   if(list==NULL)
     {
      ::Print(DFUN+TextByLanguage("Ошибка создания временного списка","Error creating temporary list"));
      return NULL;
     }
   datetime begin=begin_time,end=(end_time==0 ? END_TIME : end_time);
   if(begin_time>end_time) begin=0;
   list.FreeMode(false);
   ListStorage.Add(list);
   //---
   m_order_instance.SetProperty(property,begin);
   int index_begin=m_list_all_orders.SearchGreatOrEqual(&m_order_instance);
   if(index_begin==WRONG_VALUE)
      return list;
   m_order_instance.SetProperty(property,end);
   int index_end=m_list_all_orders.SearchLessOrEqual(&m_order_instance);
   if(index_end==WRONG_VALUE)
      return list;
   for(int i=index_begin; i<=index_end; i++)
      list.Add(m_list_all_orders.At(i));
   return list;
  }
//+------------------------------------------------------------------+
```

So, what do we have here?

- The method receives the required history's range start time, range end time and the mode of selecting the range dates from the ENUM\_SELECT\_BY\_TIME enumeration.
- The necessary property for searching and comparing in the order properties is set depending on the time the list has been sorted by. If it has been sorted by the open time, the search property will be the open time. If it has been sorted by the close time, the close time is compared in the orders, etc.
- The new list is then created. It will contain orders corresponding to the date range criterion and eventually will be returned to the calling program.
- The range start and end dates are then checked.
- If zero is passed as the range end date, set the maximum future date, then check the range start date. If it exceeds the range end date, the earliest date is set as the range start one. In this case, if the dates are set incorrectly, the list from the start of the account history up to the range end is provided. An earlier date is considered the range start, while a date closer to the current time is considered the range end.
- The flag of the manual memory management (discussed above) is then set for the newly created list, and the list is attached to the storage object.
- Next, the initial date is set to the sample order for searching in the collection list and the search for the order index with the initial date is performed using the [SearchGreatOrEqual()](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjsearchgreatorequal) method inherited by the COrder class from the parent CObject.
- If the index is not found, this means there are no orders later than the specified date, and the empty list is returned.
- Next, the same is done when searching for the end date: the end date to be used in the search is set to the sample, and the search for the order index with the end date using the [SearchLessOrEqual()](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjsearchlessorequal) method is performed. If the index is not found, this means there are no orders earlier than the target date, and the empty list is returned.
- Next, all orders located in the range are added to the listin the loop from the start date order index to the end date order one, and the filled list is returned to the calling program.

In the public section, declare the methods returning the list by the selected integer, real and string properties meeting the compared criterion:

```
//+------------------------------------------------------------------+
//| Collection of historical orders and deals                        |
//+------------------------------------------------------------------+
class CHistoryCollection
  {
private:
   CArrayObj         m_list_all_orders;      // List of all historical orders and deals
   COrder            m_order_instance;       // Order object to search by a property
   bool              m_is_trade_event;       // Trading event flag
   int               m_index_order;          // Index of the last order added to the collection from the terminal history list (MQL4, MQL5)
   int               m_index_deal;           // Index of the last deal added to the collection from the terminal history list (MQL5)
   int               m_delta_order;          // Difference in the number of orders as compared to the past check
   int               m_delta_deal;           // Difference in the number of deals as compared to the past check
public:
   //--- Return the full collection list 'as is'
   CArrayObj        *GetList(void) { return &m_list_all_orders;  }
   //--- Select orders from the collection with time from begin_time to end_time
   CArrayObj        *GetListByTime(const datetime begin_time=0,const datetime end_time=0,
                                   const ENUM_SELECT_BY_TIME select_time_mode=SELECT_BY_TIME_CLOSE);
   //--- Return the list by selected (1) integer, (2) real and (3) string properties meeting the compared criterion
   CArrayObj*        GetList(ENUM_ORDER_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByOrderProperty(this.GetList(),property,value,mode); }
   CArrayObj*        GetList(ENUM_ORDER_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByOrderProperty(this.GetList(),property,value,mode); }
   CArrayObj*        GetList(ENUM_ORDER_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByOrderProperty(this.GetList(),property,value,mode); }
   //--- Constructor
                     CHistoryCollection();
   //--- Update the list of orders, fill data on the number of new ones and set the trading event flag
   void              Refresh(void);
  };
//+------------------------------------------------------------------+
```

The method receives the order's target property, the value to compare to and the comparison mode (more/less/equal/not equal/equal or more/equal or less). The list sorted by the required properties, values and comparison method is returned with the help of the previously described CSelect class methods.

**Let's test receiving the required lists using different methods.**

Use the test EA TestDoEasyPart02.mq5 from the [second part](https://www.mql5.com/en/articles/5669) and save it in the new **Part03** subfolder of MQL5\\Experts\\TestDoEasy under the name **TestDoEasyPart03\_1.mq5**. Add selection of the date range's start and end to its input parameters and change the code in the OnInit() handler where we are going to request history in the date range:

```
//+------------------------------------------------------------------+
//|                                           TestDoEasyPart03_1.mq5 |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Collections\HistoryCollection.mqh>
//--- enums
enum ENUM_TYPE_ORDERS
  {
   TYPE_ORDER_MARKET,   // Market orders
   TYPE_ORDER_PENDING,  // Pending orders
   TYPE_ORDER_DEAL      // Deals
  };
//--- input parameters
input ENUM_TYPE_ORDERS  InpOrderType   =  TYPE_ORDER_DEAL;  // Show type:
input datetime          InpTimeBegin   =  0;                // Start date of required range
input datetime          InpTimeEnd     =  END_TIME;         // End date of required range
//--- global variables
CHistoryCollection history;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- update history
   history.Refresh();
//--- get collection list in the date range
   CArrayObj* list=history.GetListByTime(InpTimeBegin,InpTimeEnd,SELECT_BY_TIME_CLOSE);
   if(list==NULL)
     {
      Print("Could not get collection list");
      return INIT_FAILED;
     }
   int total=list.Total();
   for(int i=0;i<total;i++)
     {
      //--- get the order from the list
      COrder* order=list.At(i);
      if(order==NULL) continue;
      //--- if this is a deal
      if(order.Status()==ORDER_STATUS_DEAL && InpOrderType==TYPE_ORDER_DEAL)
         order.Print();
      //--- if this is a historical market order
      if(order.Status()==ORDER_STATUS_HISTORY_ORDER && InpOrderType==TYPE_ORDER_MARKET)
         order.Print();
      //--- if this is a removed pending order
      if(order.Status()==ORDER_STATUS_HISTORY_PENDING && InpOrderType==TYPE_ORDER_PENDING)
         order.Print();
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

  }
//+------------------------------------------------------------------+
```

Now, instead of the full list, we get the list selected by the specified date range using the GetListByTime() method. Compile and launch the EA with the default settings. All deals for the entire account history are displayed in the journal:

![](https://c.mql5.com/2/36/Result1_en__1.gif)

Press F7 and set the end date of the required range in the settings. Personally, I entered the account history, determined when a replenishment took place, defined the date of the next deal (the first deal that occurred after the account replenishment)

![](https://c.mql5.com/2/36/BalanceDeal_en.png)

and selected the range so that the first deal was outside of it: 2018.01.22 - 2018.02.01.

As a result, only one deal (account replenishment) was displayed in the journal:

![](https://c.mql5.com/2/36/Balance1_en.gif)

Now let's save the TestDoEasyPart03\_1.mq5 EA under the name **TestDoEasyPart03\_2mq5**. Remove the inputs and change the way the data on deals is received:

```
//+------------------------------------------------------------------+
//|                                           TestDoEasyPart03_2.mq5 |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Collections\HistoryCollection.mqh>
//--- global variables
CHistoryCollection history;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- update history
   history.Refresh();
//--- receive only deals from the collection list
   CArrayObj* list=history.GetList(ORDER_PROP_STATUS,ORDER_STATUS_DEAL,EQUAL);
//--- sort the obtained list by balance operations
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,DEAL_TYPE_BALANCE,EQUAL);
   if(list==NULL)
     {
      Print("Could not get collection list");
      return INIT_FAILED;
     }
   int total=list.Total();
   for(int i=0;i<total;i++)
     {
      //--- get the order from the list
      COrder* order=list.At(i);
      if(order==NULL) continue;
      order.Print();
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

  }
//+------------------------------------------------------------------+
```

First, get the list of all deals (highlight order states of the deal type in the list) and sort the obtained list by the "balance operation" type. The Equal comparison mode is used in both cases.

As a result, only the "Balance replenishment" operation is displayed in the journal:

![](https://c.mql5.com/2/36/Balance1_3_en.gif)

While in the previous example, we had to look at the deal range in the terminal's account history tab to display the balance operation, here we have obtained it immediately after sorting the list by the required criteria.

Any other ways of obtaining the required data follow the same principle. For example, in order to obtain the same "balance replenishment", we can find the indices of the most and least profitable deals.

Example in the **TestDoEasyPart03\_3.mq5** test EA:

```
//+------------------------------------------------------------------+
//|                                           TestDoEasyPart03_3.mq5 |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Collections\HistoryCollection.mqh>
//--- global variables
CHistoryCollection history;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- update history
   history.Refresh();
//--- receive only deals from the collection list
   CArrayObj* list=history.GetList(ORDER_PROP_STATUS,ORDER_STATUS_DEAL,EQUAL);
   if(list==NULL)
     {
      Print(TextByLanguage("Не удалось получить список","Could not get list"));
      return INIT_FAILED;
     }
//--- Get the index of the most profitable deal (first balance replenishment)
   int index=CSelect::FindOrderMax(list,ORDER_PROP_PROFIT);
   if(index!=WRONG_VALUE)
     {
      //--- get the deal from the list by index
      COrder* order=list.At(index);
      if(order!=NULL)
         order.Print();
     }
   else
      Print(TextByLanguage("Не найден индекс ордера с максимальным значением профита","Order index with maximum profit value not found"));
//--- Get the index of the least profitable deal
   index=CSelect::FindOrderMin(list,ORDER_PROP_PROFIT);
   if(index!=WRONG_VALUE)
     {
      //--- get the deal from the list by index
      COrder* order=list.At(index);
      if(order!=NULL)
         order.Print();
     }
   else
      Print(TextByLanguage("Не найден индекс ордера с минимальным значением профита","Order index with minimum profit value not found"));
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

  }
//+------------------------------------------------------------------+
```

Upon completion, two deals are displayed in the journal — the one having the highest profit (balance replenishment) and the one with the least profit.

![](https://c.mql5.com/2/36/MaxMinDeals_en.gif)

### Engine base object is the library core

A custom program working in conjunction with the library should send data to the library and receive data from it. To achieve this, it is more convenient to have a single class connecting to the program and accumulating all possible actions for communication between the library and the program. Of course, the class should take on all possible service functions that will work in the background and will not require any costly actions from a user. Therefore, we will create a basic class as the library basis and name it **Engine**.

In the root folder of the library, create the new CEngine class based on the basic CObject and connect historical collection class to it:

```
//+------------------------------------------------------------------+
//|                                                       Engine.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Collections\HistoryCollection.mqh"
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine : public CObject
  {
private:
//--- Collection of historical orders and deals
   CHistoryCollection   m_history;
public:
                        CEngine();
                       ~CEngine();
  };
//+------------------------------------------------------------------+
//| CEngine constructor                                              |
//+------------------------------------------------------------------+
CEngine::CEngine()
  {
  }
//+------------------------------------------------------------------+
//| CEngine destructor                                               |
//+------------------------------------------------------------------+
CEngine::~CEngine()
  {
  }
//+------------------------------------------------------------------+
```

All actions we performed while working on the collection of historical orders in test EAs were carried out in the OnInit() handler. In other words, they were executed only once when launching the EA, re-compiling it or changing its parameters. This is enough for a fast check but is not acceptable in the working program. So let's start putting everything in order.

First, create the OnTimer() handler in the public section of the class, as well as its implementation outside of the class body to update all collections:


```
//+------------------------------------------------------------------+
//|                                                       Engine.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Collections\HistoryCollection.mqh"
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine : public CObject
  {
private:
//--- Collection of historical orders and deals
   CHistoryCollection   m_history;
public:
//--- Timer
   void                 OnTimer(void);
                        CEngine();
                       ~CEngine();
  };
//+------------------------------------------------------------------+
//| CEngine constructor                                              |
//+------------------------------------------------------------------+
CEngine::CEngine()
  {
  }
//+------------------------------------------------------------------+
//| CEngine destructor                                               |
//+------------------------------------------------------------------+
CEngine::~CEngine()
  {
  }
//+------------------------------------------------------------------+
//| CEngine timer                                                    |
//+------------------------------------------------------------------+
void CEngine::OnTimer(void)
  {

  }
//+------------------------------------------------------------------+
```

Let's make the service class (timer counter) since it is quite likely different timer delays will be required for different events. The class is to count the required delay time, and a separate counter instance is to be declared for each of the timer counters.

First, add new macro substitutions to the **Defines.mqh** files. Specify the frequency of the library timer and the collections timer counter pause in milliseconds, collections timer counter increment and the ID of historical orders and deals update timer counter in them (when developing the library, you may need several counters requiring individual IDs)


```
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
#define COUNTRY_LANG             ("Russian")                // Country language
#define DFUN                     (__FUNCTION__+": ")        // "Function description"
#define END_TIME                 (D'31.12.3000 23:59:59')   // End date for requesting account history data
#define TIMER_FREQUENCY          (16)                       // Minimal frequency of the library timer in milliseconds
#define COLLECTION_PAUSE         (250)                      // Orders and deals collection timer pause in milliseconds
#define COLLECTION_COUNTER_STEP  (16)                       // Increment of the orders and deals collection timer counter
#define COLLECTION_COUNTER_ID    (1)                        // Orders and deals collection timer counter ID
//+------------------------------------------------------------------+
```

In the root folder of the library, create the new **Services** folder, as well as the new CTimerCounter class in it. Move the DELib.mqh service functions file to this folder right away. This is the right place for it.

After moving DELib.mqh to the new folder, change the address of the service functions file in the Order.mqh file:


replace the address

```
//+------------------------------------------------------------------+
//|                                                        Order.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Object.mqh>
#include "..\DELib.mqh"
//+------------------------------------------------------------------+
```

with the address:

```
//+------------------------------------------------------------------+
//|                                                        Order.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Object.mqh>
#include "..\Services\DELib.mqh"
//+------------------------------------------------------------------+
```

**Now let's consider the timer counter class**. The class is simple. So let's look at its listing and dwell on its operation:

```
//+------------------------------------------------------------------+
//|                                                 TimerCounter.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Object.mqh>
#include "DELib.mqh"
//+------------------------------------------------------------------+
//| Timer counter class                                              |
//+------------------------------------------------------------------+
class CTimerCounter : public CObject
  {
private:
   int               m_counter_id;
   ulong             m_counter;
   ulong             m_counter_step;
   ulong             m_counter_pause;
public:
   //--- Return the waiting completion flag
   bool              IsTimeDone(void);
   //--- Set the counter parameters
   void              SetParams(const ulong step,const ulong pause)         { this.m_counter_step=step; this.m_counter_pause=pause;  }
   //--- Return th counter id
   virtual  int      Type(void)                                      const { return this.m_counter_id;                              }
   //--- Compare counter objects
   virtual int       Compare(const CObject *node,const int mode=0)   const;
   //--- Constructor
                     CTimerCounter(const int id);
  };
//+------------------------------------------------------------------+
//| CTimerCounter constructor                                        |
//+------------------------------------------------------------------+
CTimerCounter::CTimerCounter(const int id) : m_counter(0),m_counter_step(16),m_counter_pause(16)
  {
   this.m_counter_id=id;
  }
//+------------------------------------------------------------------+
//| CTimerCounter returns the pause completion flag                  |
//+------------------------------------------------------------------+
bool CTimerCounter::IsTimeDone(void)
  {
   if(this.m_counter>=ULONG_MAX)
      this.m_counter=0;
   if(this.m_counter<this.m_counter_pause)
     {
      this.m_counter+=this.m_counter_step;
      return false;
     }
   this.m_counter=0;
   return true;
  }
//+------------------------------------------------------------------+
//| Compare CTimerCounter objects by id                              |
//+------------------------------------------------------------------+
int CTimerCounter::Compare(const CObject *node,const int mode=0) const
  {
   const CTimerCounter *counter_compared=node;
   int value_compared=counter_compared.Type();
   int value_current=this.Type();
   return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
   return 0;
  }

//+------------------------------------------------------------------+
```

Since we have moved DELib.mqh to the same folder where the counter class is located, we should include it directly from the same folder. Defines.mqh is included into DELib.mqh, which means the class will see all macro substitutions.

- Four class member variables are declared in the private section: timer ID, timer counter, timer increment and pause.
- The method of setting required counter parameters is located in the public section. The method passes the timer step and pause. The passed values are immediately assigned to class member variables.
- In the class constructor, the default parameters of the timer counter (0), timer step (16) and timer pause (16) are specified in the initialization list. The step and pause parameters allow the timer to work with no delay, while waiting for the pause duration to be reached.
- The value passed by the input is assigned to the counter ID in the class constructor body.

The method returning the pause completion flag is arranged in a simple way:

- First, the system checks for the variable counter overflow. If the value exceeds the maximum possible one, the counter is reset to zero.
- Next, the system compares timer counter and pause values. If the counter value is less than that of the pause, the increment is added to the counter and 'false' is returned.
- If the counter value exceeds or is equal to the pause, the counter is reset and the counter waiting time completion event is returned.

Method returning the Type() counter ID is made virtual. In the CObject class delivery, there is a [virtual method returning the object type](https://www.mql5.com/en/docs/standardlibrary/cobject/cobjecttype):

```
   //--- method of identifying the object
   virtual int       Type(void)                                    const { return(0);      }
```

It is assumed that this method is to be re-defined in the descendant classes and is to return the ID of the CObject descendant class object (the type=0 is returned for CObject itself). Let's use this opportunity and return the counter ID by re-defining the virtual method:

```
   virtual  int      Type(void)                                    const { return this.m_counter_id; }
```

**The virtual method of comparing two counter objects** is simple:

```
//+------------------------------------------------------------------+
//| Compare CTimerCounter objects by id                              |
//+------------------------------------------------------------------+
int CTimerCounter::Compare(const CObject *node,const int mode=0) const
  {
   const CTimerCounter *counter_compared=node;
   int value_compared=counter_compared.Type();
   int value_current=this.Type();
   return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
   return 0;
  }
//+------------------------------------------------------------------+
```

Get a link to the source object, take its ID and take ID of the current counter. Next, return the result of a simple comparison by more/less/equal.

* * *

Let's continue to fill in the CEngine class. Include the timer counter class into the Engine.mqh file:

```
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Collections\HistoryCollection.mqh"
#include "Services\TimerCounter.mqh"
//+------------------------------------------------------------------+
```

Add the list object of the timer counters, as well as the method returning the counter index in the list by its ID to the private section, and the counter creation method to the public one (so that the method is accessible from the outside, and it is possible to create custom counters in programs).

In the class constructor, initialize the millisecond timer, set the sorted list flag and create the counter of the historical orders and deals collection update timer. Set destroying the timer in the class destructor:


```
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine : public CObject
  {
private:
//--- List of timer counters
   CArrayObj            m_list_counters;
//--- Collection of historical orders and deals
   CHistoryCollection   m_history;
//--- Return the counter index by id
   int                  CounterIndex(const int id) const;
public:
//--- Create the timer counter
   void                 CreateCounter(const int counter_id,const ulong frequency,const ulong pause);
//--- Timer
   void                 OnTimer(void);
                        CEngine();
                       ~CEngine();
  };
//+------------------------------------------------------------------+
//| CEngine constructor                                              |
//+------------------------------------------------------------------+
CEngine::CEngine()
  {
   ::EventSetMillisecondTimer(TIMER_FREQUENCY);
   this.m_list_counters.Sort();
   this.CreateCounter(COLLECTION_COUNTER_ID,COLLECTION_COUNTER_STEP,COLLECTION_MIN_PAUSE);
  }
//+------------------------------------------------------------------+
//| CEngine destructor                                               |
//+------------------------------------------------------------------+
CEngine::~CEngine()
  {
   ::EventKillTimer();
  }
//+------------------------------------------------------------------+
```

**Implementing the method returning the counter index by its ID:**

```
//+------------------------------------------------------------------+
//| Return the counter index in the list by id                       |
//+------------------------------------------------------------------+
int CEngine::CounterIndex(const int id) const
  {
   int total=this.m_list_counters.Total();
   for(int i=0;i<total;i++)
     {
      CTimerCounter* counter=this.m_list_counters.At(i);
      if(counter==NULL) continue;
      if(counter.Type()==id)
         return i;
     }
   return WRONG_VALUE;
  }
//+------------------------------------------------------------------+
```

Since there may hardly be numerous counters, I have arranged the simplest enumeration featuring the values search and comparison. If the counter with this ID is found in the list, the system returns its index in the list, otherwise, the system returns -1.

**Let's consider the method of creating the timer counter:**

```
//+------------------------------------------------------------------+
//| Create the timer counter                                         |
//+------------------------------------------------------------------+
void CEngine::CreateCounter(const int id,const ulong step,const ulong pause)
  {
   if(this.CounterIndex(id)>WRONG_VALUE)
     {
      ::Print(TextByLanguage("Ошибка. Уже создан счётчик с идентификатором ","Error. Already created counter with id "),(string)id);
      return;
     }
   m_list_counters.Sort();
   CTimerCounter* counter=new CTimerCounter(id);
   if(counter==NULL)
      ::Print(TextByLanguage("Не удалось создать счётчик таймера ","Failed to create timer counter "),(string)id);
   counter.SetParams(step,pause);
   if(this.m_list_counters.Search(counter)==WRONG_VALUE)
      this.m_list_counters.Add(counter);
   else
     {
      string t1=TextByLanguage("Ошибка. Счётчик с идентификатором ","Error. Counter with ID ")+(string)id;
      string t2=TextByLanguage(", шагом ",", step ")+(string)step;
      string t3=TextByLanguage(" и паузой "," and pause ")+(string)pause;
      ::Print(t1+t2+t3+TextByLanguage(" уже существует"," already exists"));
      delete counter;
     }
  }
//+------------------------------------------------------------------+
```

First, check the counter ID passed to the method. If such an ID is already present, display the appropriate message in the journal and exit the method — the counter with this ID already exists.

Since the [search](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjsearch) can only be conducted in a sorted list, the sorting flag is set for the list, a new counter object is created, its successful creation is checked and the counter's required properties are set.

After that, the search for the same counter is performed in the list. If it is not found, the new counter is added to the list.

Otherwise, the message containing all parameters is formed and displayed in the journal. Then the counter object is removed since we already have a similar one.

The last check for all parameters of the newly created counter matching the already existing one is currently redundant — ID check at the very beginning of the method prevents creating an object with an ID already present in the list. I have left it for possible future changes.

To let the CEngine class know when to handle the trading situation, it should be aware of the occurred changes in the number of historical orders and deals. To do this, add the methods returning the number of newly appeared orders and deals in the list to the CHistoryCollection class:

```
//+------------------------------------------------------------------+
//| Collection of historical orders and deals                        |
//+------------------------------------------------------------------+
class CHistoryCollection
  {
private:
   CArrayObj         m_list_all_orders;      // List of all historical orders and deals
   COrder            m_order_instance;       // Order object to search by a property
   bool              m_is_trade_event;       // Trading event flag
   int               m_index_order;          // Index of the last order added to the collection from the terminal history list (MQL4, MQL5)
   int               m_index_deal;           // Index of the last deal added to the collection from the terminal history list (MQL5)
   int               m_delta_order;          // Difference in the number of orders as compared to the past check
   int               m_delta_deal;           // Difference in the number of deals as compared to the past check
public:
   //--- Select orders from the collection with time from begin_time to end_time
   CArrayObj        *GetListByTime(const datetime begin_time=0,const datetime end_time=0,
                                   const ENUM_SELECT_BY_TIME select_time_mode=SELECT_BY_TIME_CLOSE);
   //--- Return the full collection list 'as is'
   CArrayObj        *GetList(void)                                                                       { return &m_list_all_orders;                                            }
   //--- Return the list by selected (1) integer, (2) real and (3) string properties meeting the compared criterion
   CArrayObj        *GetList(ENUM_ORDER_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByOrderProperty(this.GetList(),property,value,mode);  }
   CArrayObj        *GetList(ENUM_ORDER_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByOrderProperty(this.GetList(),property,value,mode);  }
   CArrayObj        *GetList(ENUM_ORDER_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByOrderProperty(this.GetList(),property,value,mode);  }
   //--- Return the number of (1) new orders and (2) new deals
   int               NewOrders(void)                                                                     { return m_delta_order; }
   int               NewDeals(void)                                                                      { return m_delta_deal;  }

   //--- Constructor
                     CHistoryCollection();
   //--- Update the list of orders, fill data on the number of new ones and set the trading event flag
   void              Refresh(void);
  };
//+------------------------------------------------------------------+
```

The methods simply return the values of the appropriate class member variables.

Now in CEngine, we may check their status in the class timer:

```
//+------------------------------------------------------------------+
//| CEngine timer                                                    |
//+------------------------------------------------------------------+
void CEngine::OnTimer(void)
  {
   //--- Timer of the historical orders and deals collection
   int index=this.CounterIndex(COLLECTION_COUNTER_ID);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter.IsTimeDone())
        {
         this.m_history.Refresh();
         if(this.m_history.NewOrders()>0)
           {
            Print(DFUN,TextByLanguage("Изменилось количество исторических ордеров: NewOrders=","Number of historical orders changed: NewOrders="),this.m_history.NewOrders());
           }
         if(this.m_history.NewDeals()>0)
           {
            Print(DFUN,TextByLanguage("Изменилось количество сделок: NewDeals=","Number of deals changed: NewDeals="),this.m_history.NewOrders());
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

Get the index of the historical orders and deals collection timer's counter, get the pointer to the timer counter by its index, check the completion of the timer delay time and update the collection (only the last added order or deal if any).

If the number of historical orders changed, display the message in the journal. The same goes for deals.


Let's make a simple EA for a check. In Experts\\TestDoEasy\\Part3, create the EA with the timer named TestDoEasyPart03\_4.mq5. To create the template of the EA with the timer, check OnTimer on the second page of MQL Wizard:

![](https://c.mql5.com/2/36/TimerCB_en.png)

Click Next till the Wizard's operation completion. As a result, an empty EA template is created. Connect the main library file to it, create the library class object and call the library timer in the EA timer.

```
//+------------------------------------------------------------------+
//|                                           TestDoEasyPart03_4.mq5 |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
//--- global variables
CEngine        engine;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

  }
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
   engine.OnTimer();
  }
//+------------------------------------------------------------------+
```

**This is all that is required to be done in the EA to obtain data on changes in the account history.**

If we launch the EA now, place a pending order and delete it, an entry concerning the change of the number of historical orders appears in the journal.

If we open a position, two entries appear in the journal:

1. concerning the change in the number of orders (the open market order has been sent) and

2. concerning the change in the number of deals (the market order has been activated and generated the "Market entry" deal).


If we close a position, two entries appear in the journal again:

1. about the appearance of the closing market order
2. concerning the appearance of the new deal (the closing market order has been activated and generated the "Market exit" deal)


During the first launch, messages on orders and deals changes in the account history are displayed in the journal. This happens because the library reads the entire history during the first launch, therefore the difference between the zero number of orders and deals (the library knows nothing about them during the first launch) and the number of all orders and deals calculated during the full account history loop will be equal to their full number. This is inconvenient and unnecessary. Therefore, we need to eliminate such false number change messages. We can do that in two ways:

1. display nothing in the journal during the first launch

2. display account status messages during the first launch

Since we are going to implement the collection and display of the necessary statistics in the subsequent parts of the description, let's use the first option for now (displaying nothing in the journal during the first launch) and just make a stub for the first launch:

In the private section of the CEngine class, add the flag of the first launch and the method that checks and resets the flag:

```
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine : public CObject
  {
private:
   CHistoryCollection   m_history;                       // Collection of historical orders and deals
   CArrayObj            m_list_counters;                 // List of timer counters
   bool                 m_first_start;                   // First launch flag
//--- Return the counter index by id
   int                  CounterIndex(const int id) const;
//--- Return the first launch flag
   bool                 IsFirstStart(void);
public:
//--- Create the timer counter
   void                 CreateCounter(const int id,const ulong frequency,const ulong pause);
//--- Timer
   void                 OnTimer(void);
                        CEngine();
                       ~CEngine();
  };
//+------------------------------------------------------------------+
```

and add the implementation of the check method and resetting the first launch flag beyond the class body:

```
//+------------------------------------------------------------------+
//| Return the first launch flag, reset the flag                     |
//+------------------------------------------------------------------+
bool CEngine::IsFirstStart(void)
  {
   if(this.m_first_start)
     {
      this.m_first_start=false;
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

All is very simple here: if the flag is set, reset it and return 'true', otherwise return 'false'.

Now we need to set the flag in the initialization list, so that it always remains ready to be activated.

```
//+------------------------------------------------------------------+
//| CEngine constructor                                              |
//+------------------------------------------------------------------+
CEngine::CEngine() : m_first_start(true)
  {
   ::EventSetMillisecondTimer(TIMER_FREQUENCY);
   this.m_list_counters.Sort();
   this.CreateCounter(COLLECTION_COUNTER_ID,COLLECTION_COUNTER_STEP,COLLECTION_PAUSE);
  }
//+------------------------------------------------------------------+
```

All is set. From now on, no unnecessary account history event related to the very first history calculation will appear during the first program launch.

We can check this by launching the test EA from MQL5\\Experts\\TestDoEasy\\Part2\\TestDoEasyPart03\_4.mq5 and ensuring that no messages about adding orders and deals to the account history appear in the Experts journal during the first launch.

### Objects of active market orders and positions

I believe, now it is time to temporarily stop adding the functionality to the CEngine class and start implementing  objects and the collection of market orders and positions. After the implementation is complete, we will continue developing the functionality of the Engine base object, since this functionality affects both the history of the account and its current status.


In the **Objects** folder of the library, create the new CMarketPosition class based on the abstract order of the COrder library — this is to be a market position object:

![](https://c.mql5.com/2/36/NewCMarketOrder_en.png)

After clicking Finish, a class template named MarketPosition.mqh is created. Let's add inclusion of the COrder class to it right away:

```
//+------------------------------------------------------------------+
//|                                               MarketPosition.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Order.mqh"
//+------------------------------------------------------------------+
//| Market position                                                  |
//+------------------------------------------------------------------+
class CMarketPosition : public COrder
  {
private:

public:
                     CMarketPosition();
                    ~CMarketPosition();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CMarketPosition::CMarketPosition()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CMarketPosition::~CMarketPosition()
  {
  }
//+------------------------------------------------------------------+
```

Change the constructor so that the position ticket is passed to it, set the "market position" status for the parent class (COrder) in its initialization list and send the ticket to it. Declare three virtual methods returning the flag of supporting integer, real and string properties by the position:

```
//+------------------------------------------------------------------+
//| Market position                                                  |
//+------------------------------------------------------------------+
class CMarketPosition : public COrder
  {
public:
   //--- Constructor
                     CMarketPosition(const ulong ticket=0) : COrder(ORDER_STATUS_MARKET_POSITION,ticket) {}
   //--- Supported position properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_ORDER_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_ORDER_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_ORDER_PROP_STRING property);
  };
//+------------------------------------------------------------------+
```

and add implementation of these methods outside of the class body:

```
//+------------------------------------------------------------------+
//| Return 'true' if the position supports the passed                |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CMarketPosition::SupportProperty(ENUM_ORDER_PROP_INTEGER property)
  {
   if(property==ORDER_PROP_TIME_CLOSE     ||
      property==ORDER_PROP_TIME_CLOSE_MSC ||
      property==ORDER_PROP_TIME_EXP       ||
      property==ORDER_PROP_POSITION_BY_ID ||
      property==ORDER_PROP_DEAL_ORDER     ||
      property==ORDER_PROP_DEAL_ENTRY     ||
      property==ORDER_PROP_CLOSE_BY_SL    ||
      property==ORDER_PROP_CLOSE_BY_TP
     #ifdef __MQL5__                      ||
      property==ORDER_PROP_TICKET_FROM    ||
      property==ORDER_PROP_TICKET_TO
     #endif
     ) return false;
   return true;
}
//+------------------------------------------------------------------+
//| Return 'true' if the position supports the passed                |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CMarketPosition::SupportProperty(ENUM_ORDER_PROP_DOUBLE property)
  {
   if(property==ORDER_PROP_PRICE_CLOSE || property==ORDER_PROP_PRICE_STOP_LIMIT) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if the position supports the passed                |
//| string property, otherwise return 'false'                        |
//+------------------------------------------------------------------+
bool CMarketPosition::SupportProperty(ENUM_ORDER_PROP_STRING property)
  {
   if(property==ORDER_PROP_EXT_ID) return false;
   return true;
  }
//+------------------------------------------------------------------+
```

Everything here is similar to creating objects of historical orders and deals [discussed in the second part of the library description](https://www.mql5.com/en/articles/5669#node01).

Now let's create a market pending order object in a similar way. Let's create a new class CMarketPending based on the abstract order of the COrder library in the Objects folder and enter the already familiar changes of the class template created by MQL Wizard to it:

```
//+------------------------------------------------------------------+
//|                                                MarketPending.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Order.mqh"
//+------------------------------------------------------------------+
//| Market pending order                                             |
//+------------------------------------------------------------------+
class CMarketPending : public COrder
  {
public:
   //--- Constructor
                     CMarketPending(const ulong ticket=0) : COrder(ORDER_STATUS_MARKET_PENDING,ticket) {}
   //--- Supported order properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_ORDER_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_ORDER_PROP_INTEGER property);
  };
//+------------------------------------------------------------------+
//| Return 'true' if the order supports the passed                   |
//| integer property, otherwise, return 'false'                      |
//+------------------------------------------------------------------+
bool CMarketPending::SupportProperty(ENUM_ORDER_PROP_INTEGER property)
  {
   if(property==ORDER_PROP_PROFIT_PT         ||
      property==ORDER_PROP_DEAL_ORDER        ||
      property==ORDER_PROP_DEAL_ENTRY        ||
      property==ORDER_PROP_TIME_UPDATE       ||
      property==ORDER_PROP_TIME_CLOSE        ||
      property==ORDER_PROP_TIME_CLOSE_MSC    ||
      property==ORDER_PROP_TIME_UPDATE_MSC   ||
      property==ORDER_PROP_TICKET_FROM       ||
      property==ORDER_PROP_TICKET_TO         ||
      property==ORDER_PROP_CLOSE_BY_SL       ||
      property==ORDER_PROP_CLOSE_BY_TP
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if the order supports the passed                   |
//| real property, otherwise, return 'false'                         |
//+------------------------------------------------------------------+
bool CMarketPending::SupportProperty(ENUM_ORDER_PROP_DOUBLE property)
  {
   if(property==ORDER_PROP_COMMISSION  ||
      property==ORDER_PROP_SWAP        ||
      property==ORDER_PROP_PROFIT      ||
      property==ORDER_PROP_PROFIT_FULL ||
      property==ORDER_PROP_PRICE_CLOSE
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
```

Pass the "pending order" status to the basic COrder in the class constructor's initialization list.

We have completed the development of the objects necessary for creating the collection of market orders and positions.

### Collection of active market orders and positions

While creating the collection of historical orders and deals, we adhered to the rule that there is no point in constantly checking the entire history. So, we added new orders and deals to the single previously created list only when their numbers changed. When using the list of market positions, a completely different rule should be kept in mind — the list on each tick should be relevant.

To achieve this:

1. track the changes in the number of pending orders, in the number of active positions for hedge accounts (since there is only one position on netting accounts) and in the position volume (increasing or decreasing volume in the netting position, partial closing of one of the hedging positions),
2. make sure to update data on each existing hedging position or a single netting one at each tick to always have relevant position status data.

Remember about the values specified on the last tick to compare them with the same data on the current one and update the position or re-create the entire list again if there are changes. Fortunately, the list is not large, and its re-creation does not take much time.

Let's start. In the **Collections** library folder, create the new class **CMarketCollection**. To do this, right-click the Collection folder and select "New file". In the newly opened MQL Wizard, select "New Class" and click Next.

![](https://c.mql5.com/2/36/CreateMarketCollection_en.png)

Enter the name of the class CMarketCollection and click Finish. The MarketCollection.mqh class template is created:

```
//+------------------------------------------------------------------+
//|                                             MarketCollection.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CMarketCollection
  {
private:

public:
                     CMarketCollection();
                    ~CMarketCollection();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CMarketCollection::CMarketCollection()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CMarketCollection::~CMarketCollection()
  {
  }
//+------------------------------------------------------------------+
```

Let's fill it in.

First, include all the prepared classes, as well as the ones required for implementing the collection of market orders and positions and searching it:

```
//+------------------------------------------------------------------+
//|                                             MarketCollection.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Arrays\ArrayObj.mqh>
#include "Select.mqh"
#include "..\Objects\MarketPending.mqh"
#include "..\Objects\MarketPosition.mqh"
//+------------------------------------------------------------------+
```

In the private section of the class, create the structure, specify the variables for storing all previously mentioned tracked values in it (number of orders and positions, etc.) and create two class member variables with this structure type for storing current and previous data:

```
//+------------------------------------------------------------------+
//|                                             MarketCollection.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Arrays\ArrayObj.mqh>
#include "Select.mqh"
#include "..\Objects\MarketPending.mqh"
#include "..\Objects\MarketPosition.mqh"
//+------------------------------------------------------------------+
//| Collection of historical orders and deals                        |
//+------------------------------------------------------------------+
class CMarketCollection
  {
private:
   struct MqlDataCollection
     {
      long           hash_sum_acc;           // Hash sum of all orders and positions on the account
      int            total_pending;          // Number of pending orders on the account
      int            total_positions;        // Number of positions on the account
      double         total_volumes;          // Total volume of orders and positions on the account
     };
   MqlDataCollection m_struct_curr_market;   // Current data on market orders and positions on the account
   MqlDataCollection m_struct_prev_market;   // Previous data on market orders and positions on the account
public:
                     CMarketCollection();
                    ~CMarketCollection();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CMarketCollection::CMarketCollection()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CMarketCollection::~CMarketCollection()
  {
  }
//+------------------------------------------------------------------+
```

Let us dwell on the hash sum in the structure.

The number of orders and positions is insufficient if we want to accurately define an occurred account event. A pending order may be removed changing the total number of orders and positions on the account. On the other hand, a pending order may be activated and become a position. In this case, the total sum of orders and positions remains unchanged (for hedging accounts and MQL4) — the number of positions increases, but the number of orders decreases. As a result, the total number remains the same. This is not suitable for us.

Let's consider the ticket. Adding/removing a pending order changes the total sum of tickets on the account, while a pending order activation does not change the total sum of tickets on the account.

Let's consider the total volume. Placing/removing a pending order — the total volume on the account has changed, opening, closing or modifying a position —  the total volume on the account has changed. This option seems to fit but activating a pending orders does not change the total volume.

So, let's have a look at yet another position property — time of its change in milliseconds: opening a new position alters the total position change time, partial closing alters the position change time and adding the volume on a netting account changes the total position change time.


What are the most suitable options for accurate defining of an occurred account change? Ticket+position change time. Let's check:

- Opening a position — sum of tickets has changed \+ sum of position change time has changed — there is a change

- Closing a position — sum of tickets has changed \+ sum of position change time has changed — there is a change
- Placed a pending order — sum of tickets has changed \+ sum of position change time has not changed — there is a change
- Removed a pending order — sum of tickets has changed \+ sum of position change time has not changed — there is a change
- Activated a pending order — sum of tickets has not changed \+ sum of position change time has changed — there is a change
- Closed a position partially — sum of tickets has changed \+ sum of position change time has changed — there is a change
- Adding a volume to a position — sum of tickets has not changed \+ sum of position change time has changed — there is a change

**Thus, we will use the ticket + position change time in milliseconds for the hash sum.**

In the private section of the class, create the [dynamic list of pointers to objects](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) to be used as a collection list of market pending orders and positions. Also, create two flags: trading event flag on the account and the flag indicating an occurred position volume changefor a simplified identification of a trading event in the CEngine class, as well as three class member variables for setting the volume change value, number of new positions and pending orders.

In the public section, declare the method for updating the collection list and write the class constructor implementation outside the class body:

```
//+------------------------------------------------------------------+
//| Collection of market orders and positions                        |
//+------------------------------------------------------------------+
class CMarketCollection
  {
private:
   struct MqlDataCollection
     {
      long           hash_sum_acc;           // Hash sum of all orders and positions on the account
      int            total_pending;          // Number of pending orders on the account
      int            total_positions;        // Number of positions on the account
      double         total_volumes;          // Total volume of orders and positions on the account
     };
   MqlDataCollection m_struct_curr_market;   // Current data on market orders and positions on the account
   MqlDataCollection m_struct_prev_market;   // Previous data on market orders and positions on the account
   CArrayObj         m_list_all_orders;      // List of pending orders and positions on the account
   bool              m_is_trade_event;       // Trading event flag
   bool              m_is_change_volume;     // Total volume change flag
   double            m_change_volume_value;  // Total volume change value
   int               m_new_positions;        // Number of new positions
   int               m_new_pendings;         // Number of new pending orders
public:
   //--- Constructor
                     CMarketCollection(void);
   //--- Update the list of pending orders and positions
   void              Refresh(void);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CMarketCollection::CMarketCollection(void) : m_is_trade_event(false),m_is_change_volume(false),m_change_volume_value(0)
  {
   m_list_all_orders.Sort(SORT_BY_ORDER_TIME_OPEN);
   ::ZeroMemory(this.m_struct_prev_market);
   this.m_struct_prev_market.hash_sum_acc=WRONG_VALUE;
  }
//+------------------------------------------------------------------+
```

Reset the trading event and position volume change flags and reset the volume change value in the initialization list of the class constructor.

In the constructor body, set the sorting of the market order and position list by open time, reset all variable structures of the account's previous status except for the previous hash sum — set -1 for it (for identifying the first launch).

Add the method for saving the currently collected account data in the previous data structure to the private section of the class for subsequent check of changes in the number of orders and positions on the account. Add the three methods returning the number of new pending orders,  the number of new positions and the method returning the flag of a trading event occurred on the account to the public section of the class:

```
//+------------------------------------------------------------------+
//| Collection of market orders and positions                        |
//+------------------------------------------------------------------+
class CMarketCollection
  {
private:
   struct MqlDataCollection
     {
      long           hash_sum_acc;           // Hash sum of all orders and positions on the account
      int            total_pending;          // Number of pending orders on the account
      int            total_positions;        // Number of positions on the account
      double         total_volumes;          // Total volume of orders and positions on the account
     };
   MqlDataCollection m_struct_curr_market;   // Current data on market orders and positions on the account
   MqlDataCollection m_struct_prev_market;   // Previous data on market orders and positions on the account
   CArrayObj         m_list_all_orders;      // List of pending orders and positions on the account
   bool              m_is_trade_event;       // Trading event flag
   bool              m_is_change_volume;     // Total volume change flag
   double            m_change_volume_value;  // Total volume change value
   int               m_new_positions;        // Number of new positions
   int               m_new_pendings;         // Number of new pending orders
   //--- Save the current values of the account data status as previous ones
   void              SavePrevValues(void)             { this.m_struct_prev_market=this.m_struct_curr_market;   }
public:
   //--- Return the number of (1) new pending orders, (2) new positions, (3) occurred trading event flag
   int               NewOrders(void)    const         { return this.m_new_pendings;                            }
   int               NewPosition(void)  const         { return this.m_new_positions;                           }
   bool              IsTradeEvent(void) const         { return this.m_is_trade_event;                          }
   //--- Constructor
                     CMarketCollection(void);
   //--- Update the list of pending orders and positions
   void              Refresh(void);
  };
//+------------------------------------------------------------------+
```

Let's implement the method for updating the current market status:

```
//+------------------------------------------------------------------+
//| Update the order list                                            |
//+------------------------------------------------------------------+
void CMarketCollection::Refresh(void)
  {
   ::ZeroMemory(this.m_struct_curr_market);
   this.m_is_trade_event=false;
   this.m_is_change_volume=false;
   this.m_new_pendings=0;
   this.m_new_positions=0;
   this.m_change_volume_value=0;
   m_list_all_orders.Clear();
#ifdef __MQL4__
   int total=::OrdersTotal();
   for(int i=0; i<total; i++)
     {
      if(!::OrderSelect(i,SELECT_BY_POS)) continue;
      long ticket=::OrderTicket();
      ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)::OrderType();
      if(type==ORDER_TYPE_BUY || type==ORDER_TYPE_SELL)
        {
         CMarketPosition *position=new CMarketPosition(ticket);
         if(position==NULL) continue;
         if(this.m_list_all_orders.InsertSort(position))
           {
            this.m_struct_market.hash_sum_acc+=ticket;
            this.m_struct_market.total_volumes+=::OrderLots();
            this.m_struct_market.total_positions++;
           }
         else
           {
            ::Print(DFUN,TextByLanguage("Не удалось добавить позицию в список","Failed to add position to list"));
            delete position;
           }
        }
      else
        {
         CMarketPending *order=new CMarketPending(ticket);
         if(order==NULL) continue;
         if(this.m_list_all_orders.InsertSort(order))
           {
            this.m_struct_market.hash_sum_acc+=ticket;
            this.m_struct_market.total_volumes+=::OrderLots();
            this.m_struct_market.total_pending++;
           }
         else
           {
            ::Print(DFUN,TextByLanguage("Не удалось добавить ордер в список","Failed to add order to list"));
            delete order;
           }
        }
     }
//--- MQ5
#else
//--- Positions
   int total_positions=::PositionsTotal();
   for(int i=0; i<total_positions; i++)
     {
      ulong ticket=::PositionGetTicket(i);
      if(ticket==0) continue;
      CMarketPosition *position=new CMarketPosition(ticket);
      if(position==NULL) continue;
      if(this.m_list_all_orders.InsertSort(position))
        {
         this.m_struct_curr_market.hash_sum_acc+=(long)::PositionGetInteger(POSITION_TIME_UPDATE_MSC);
         this.m_struct_curr_market.total_volumes+=::PositionGetDouble(POSITION_VOLUME);
         this.m_struct_curr_market.total_positions++;
        }
      else
        {
         ::Print(DFUN,TextByLanguage("Не удалось добавить позицию в список","Failed to add position to list"));
         delete position;
        }
     }
//--- Orders
   int total_orders=::OrdersTotal();
   for(int i=0; i<total_orders; i++)
     {
      ulong ticket=::OrderGetTicket(i);
      if(ticket==0) continue;
      CMarketPending *order=new CMarketPending(ticket);
      if(order==NULL) continue;
      if(this.m_list_all_orders.InsertSort(order))
        {
         this.m_struct_curr_market.hash_sum_acc+=(long)ticket;
         this.m_struct_curr_market.total_volumes+=::OrderGetDouble(ORDER_VOLUME_INITIAL);
         this.m_struct_curr_market.total_pending++;
        }
      else
        {
         ::Print(DFUN,TextByLanguage("Не удалось добавить ордер в список","Failed to add order to list"));
         delete order;
        }
     }
#endif
//--- First launch
   if(this.m_struct_prev_market.hash_sum_acc==WRONG_VALUE)
     {
      this.SavePrevValues();
     }
//--- If the hash sum of all orders and positions changed
   if(this.m_struct_curr_market.hash_sum_acc!=this.m_struct_prev_market.hash_sum_acc)
     {
      this.m_new_pendings=this.m_struct_curr_market.total_pending-this.m_struct_prev_market.total_pending;
      this.m_new_positions=this.m_struct_curr_market.total_positions-this.m_struct_prev_market.total_positions;
      this.m_change_volume_value=::NormalizeDouble(this.m_struct_curr_market.total_volumes-this.m_struct_prev_market.total_volumes,4);
      this.m_is_change_volume=(this.m_change_volume_value!=0 ? true : false);
      this.m_is_trade_event=true;
      this.SavePrevValues();
     }
  }
//+------------------------------------------------------------------+
```

Before analyzing the method, let's make a small digression: since we constantly need relevant data on all market orders and positions, we can clear the list at each tick and fill it with data from the market environment. Alternatively, we can fill in the list once and alter only the data that could change. At first sight, it seems that it would be faster to alter only changing data. But to do this, we need to:

1. go through the list of market orders and terminal positions, and fill in the library list with them,

2. at each tick, go through the list of terminal market orders and positions, take the changing data, look for orders and positions with the same ticket in the library list and update the existing data,
3. if an order is removed or a position is closed, remove it from the library list.

This seems more costly than simply clearing the library list and filling it by market orders and terminal positions in one loop.

Therefore, let's follow a simpler way: clear the list and fill it in again. Of course, nothing keeps us from trying the method involving the data search and update in the already existing library list. We will try it when working with the library and improving its working speed ("simple-to-complex" basis).

**Now let's see how the method of updating the collection list of market orders and positions is arranged.**

At the very start of the method, the structure of the current market data, event flags, volume change value, as well as all variables concerning the number of orders and positions are reset, and the collection list is cleared.

Then the check for belonging to MQL4or MQL5 is conducted.

Since at this stage, we prepare the MQL5 code, let's have a look at the MQL5 version:

Unlike MQL4, in MQL5, orders and positions are stored in different lists.

Therefore, get the total number of positions on the account, then move along all terminal positions in a loop, select the ticket of the next position, create a position object and add it to the collection list of active orders and library positions.

In the same way, add all pending orders currently present on the account. Get the total number of orders on the account for pending orders only and move along the terminal order list in a loop receiving an order ticket and adding an order object to the collection list of the library active orders and positions.

After both loops are complete, the collection list of the library's active orders and positions will contain objects of orders and positions currently present on the account. Next, check the first launch flag (the value of the "previous" hash sum equal to -1 is used as the flag here). If this is the first launch, the values of all "past" values are copied to the structure storing these values using the SavePrevValues() method. If this is not the first run, then the value of the past hash sum is compared with the value of the current one calculated when passing the loops of collecting account data to the collection list. If the previous hash sum is not equal to the current one, then a change has occurred on the account

In this case, the difference between the current and previous number of orders is set in the variable storing the new number of account orders, while the difference between the current and previous number of positions is set in the variable storing the new number of account positions. Save the value, by which the total account volume has changed, set the volume change flag, as well as the flag of an occurred trading event, and finally, add new values to the structure of "previous" values using the SavePrevValues() method for subsequent verification.

The SavePrevValues() method simply copies the structure with the current values to the structure with the previous ones.

**To check the operation of the market orders and positions list update method and its joint work with the historical orders and deals list update method**, use the last test EA from the Part03 folder named TestDoEasyPart03\_4.mq5:

```
//+------------------------------------------------------------------+
//|                                           TestDoEasyPart03_4.mq5 |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
//--- global variables
CEngine        engine;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

  }
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
   engine.OnTimer();
  }
//+------------------------------------------------------------------+
```

In order to see the changes implemented when adding and tracking collections of market orders and positions, add the following strings to the Timer event handler of the CEngine class:

```
//+------------------------------------------------------------------+
//| CEngine timer                                                    |
//+------------------------------------------------------------------+
void CEngine::OnTimer(void)
  {
   //--- Timer of the collections of historical orders and deals, as well as of market orders and positions
   int index=this.CounterIndex(COLLECTION_COUNTER_ID);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter!=NULL && counter.IsTimeDone())
        {
         //--- Update the lists
         this.m_market.Refresh();
         this.m_history.Refresh();
         //--- First launch actions
         if(this.IsFirstStart())
           {
            return;
           }
         //--- Check the market status change
         if(this.m_market.IsTradeEvent())
           {
            Print(DFUN,TextByLanguage("Новое торговое событие на счёте","New trading event on account"));
           }
         //--- Check the account history change
         if(this.m_history.IsTradeEvent())
           {
            Print(DFUN,TextByLanguage("Новое торговое событие в истории счёта","New trading event in account history"));
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

Everything is simple here. Completion of waiting in the collection timer counter is checked first. If the pause is over, the lists of market orders and positions, as well as of historical orders and deals are updated. No actions are required yet during the first launch. When receiving a flag of an occurred account event, display the appropriate method in the journal. The same is done when receiving the event flag in the account history.

Compile the test EA and launch it. Now, if we open a position, two entries appear in the journal:

```
2019.02.28 17:36:24.678 CEngine::OnTimer: New trading event on the account
2019.02.28 17:36:24.678 CEngine::OnTimer: New trading event in the account history
```

The first one indicates that a trading event has occurred on an account, while the second one informs of adding a new event to the account history. In this case, a trading event on the account consists of increasing the number of positions by 1, while a new event in the account history means the appearance of a single new opening market order and one new deal — "market entry".

If we now close a position, the same two entries appear in the journal, but now the readings are different: a trading event on the account is decreasing the number of positions by 1, while a new event in the account history is adding a single new closing market order and a single new deal — "market exit".

### What's next?

In the next article, we will continue the development of the main library element (the CEngine class) and implement handling events coming from the collection and sending them to the program.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/5687#node00)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/5687](https://www.mql5.com/ru/articles/5687)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/5687.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/5687/mql5.zip "Download MQL5.zip")(36.13 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/312964)**
(9)


![Aleksandr Brown](https://c.mql5.com/avatar/2013/5/51829572-CC9A.JPG)

**[Aleksandr Brown](https://www.mql5.com/en/users/brown-aleks)**
\|
2 Aug 2020 at 18:17

Somehow **ORDER\_STATUS\_MARKET\_ACTIVE** was replaced by **ORDER\_STATUS\_MARKET\_POSITION** in **Defines.mqh**. And everywhere, and throughout the [project](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it's not for sure "), where previously written **ORDER\_STATUS\_MARKET\_ACTIVE** should be replaced with **ORDER\_STATUS\_MARKET\_POSITION**.

This is not a big note, for those who will also scrutinise and pump their skills... On this series of articles. =)

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
2 Aug 2020 at 22:17

**Aleksandr Brown:**

Somehow **ORDER\_STATUS\_MARKET\_ACTIVE** was replaced by **ORDER\_STATUS\_MARKET\_POSITION** in **Defines.mqh**. And everywhere and throughout the project, where **ORDER\_STATUS\_MARKET\_ACTIVE** was written earlier, it should be replaced with **ORDER\_STATUS\_MARKET\_POSITION**.

This is not a big note, for those who will also scrutinise and pump their skills... On this series of articles. =)

Yes. Sometimes some things are changed "quietly" - there is no point in describing them, but it is not difficult to replace them. And it is often said in the articles that some minor tweaks were made.

Everything is in the attached files, and the articles describe the essence.

![BillionerClub](https://c.mql5.com/avatar/avatar_na2.png)

**[BillionerClub](https://www.mql5.com/en/users/billionerclub)**
\|
15 Dec 2020 at 11:23

Greatest work! Blessings and prosperity to you.

There is no criticism, but there should be, in the sense of useful criticism. Still in the work on the project we ignore already standard functions.

|     |     |
| --- | --- |
| [OnTrade](https://www.mql5.com/en/docs/event_handlers/ontrade) | The function is called in EAs during the [trade](https://www.mql5.com/en/docs/runtime/event_fire#trade) event  generated at the end of a trading operation on a trade server |
| [OnTradeTransaction](https://www.mql5.com/en/docs/event_handlers/ontradetransaction) | The function is called in EAs when the [TradeTransaction](https://www.mql5.com/en/docs/runtime/event_fire#tradetransaction) event  occurs to process a trade request execution results |

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
15 Dec 2020 at 12:54

**BillionerClub:**

Great work! Blessings and prosperity to you.

There is no criticism, but there should be, in the sense of useful criticism. Still, in the work on the project we ignore the already standard functions.

|     |     |
| --- | --- |
| [OnTrade](https://www.mql5.com/en/docs/event_handlers/ontrade) | The function is called in EAs during the [trade](https://www.mql5.com/en/docs/runtime/event_fire#trade) event  generated at the end of a trading operation on a trade server |
| [OnTradeTransaction](https://www.mql5.com/en/docs/event_handlers/ontradetransaction) | The function is called in EAs when the [TradeTransaction](https://www.mql5.com/en/docs/runtime/event_fire#tradetransaction) event  occurs to process a trade request execution results |

I bypassed them on purpose.

1\. Compatibility with MQL4

2\. Avoiding occasional loss of events when using them.

![Luiz Marin](https://c.mql5.com/avatar/2023/3/64158A72-0AF6.jpg)

**[Luiz Marin](https://www.mql5.com/en/users/luizcmarin)**
\|
14 Apr 2023 at 18:42

no arquivo "TimerCounter.mqh" a frase "const CTimerCounter \*counter\_compared = node;" dá um erro de compilação: "'=' - type mismatch".

Entendo o que você quer dizer, mas sou novo no mql. Alguém pode resolver isso para mim?

in the "TimerCounter.mqh" file, the sentence "const CTimerCounter \*counter\_compared = node;" throws error when compiling: "'=' - type mismatch".

I understand what you mean, but I'm new to mql. Could someone solve this for me?

![MTF indicators as the technical analysis tool](https://c.mql5.com/2/35/mtf-avatar.png)[MTF indicators as the technical analysis tool](https://www.mql5.com/en/articles/2837)

Most of traders agree that the current market state analysis starts with the evaluation of higher chart timeframes. The analysis is performed downwards to lower timeframes until the one, at which deals are performed. This analysis method seems to be a mandatory part of professional approach for successful trading. In this article, we will discuss multi-timeframe indicators and their creation ways, as well as we will provide MQL5 code examples. In addition to the general evaluation of advantages and disadvantages, we will propose a new indicator approach using the MTF mode.

![Extracting structured data from HTML pages using CSS selectors](https://c.mql5.com/2/35/MQL5-CSS_selectors.png)[Extracting structured data from HTML pages using CSS selectors](https://www.mql5.com/en/articles/5706)

The article provides a description of a universal method for analyzing and converting data from HTML documents based on CSS selectors. Trading reports, tester reports, your favorite economic calendars, public signals, account monitoring and additional online quote sources will become available straight from MQL.

![Using MATLAB 2018 computational capabilities in MetaTrader 5](https://c.mql5.com/2/35/ext_infin2.png)[Using MATLAB 2018 computational capabilities in MetaTrader 5](https://www.mql5.com/en/articles/5572)

After the upgrade of the MATLAB package in 2015, it is necessary to consider a modern way of creating DLL libraries. The article uses a sample predictive indicator to illustrate the peculiarities of linking MetaTrader 5 and MATLAB using modern 64-bit versions of the platforms, which are utilized nowadays. With the entire sequence of connecting MATLAB considered, MQL5 developers will be able to create applications with advanced computational capabilities much faster, avoiding «pitfalls».

![Library for easy and quick development of MetaTrader programs (part II). Collection of historical orders and deals](https://c.mql5.com/2/35/MQL5-avatar-doeasy__1.png)[Library for easy and quick development of MetaTrader programs (part II). Collection of historical orders and deals](https://www.mql5.com/en/articles/5669)

In the first part, we started creating a large cross-platform library simplifying the development of programs for MetaTrader 5 and MetaTrader 4 platforms. We created the COrder abstract object which is a base object for storing data on history orders and deals, as well as on market orders and positions. Now we will develop all the necessary objects for storing account history data in collections.

[![](https://www.mql5.com/ff/si/0nfwvn6yhmgzf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F117%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dorder.expert%26utm_content%3Dorder.freelance%26utm_campaign%3D0622.MQL5.com.Internal&a=tunpwtbhegzufrqocbwiszessdutnobs&s=d9e7484e15300021b4066b1df77a94a1352f9e7c326d5113006bb4f6476bafeb&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=zhkbjhogdcnquwujnveuccxspnkdkxfg&ssn=1769186562880427803&ssn_dr=0&ssn_sr=0&fv_date=1769186562&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F5687&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20III).%20Collection%20of%20market%20orders%20and%20positions%2C%20search%20and%20sorting%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918656218940504&fz_uniq=5070501526578861825&sv=2552)

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