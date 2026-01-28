---
title: Library for easy and quick development of MetaTrader programs (part II). Collection of historical orders and deals
url: https://www.mql5.com/en/articles/5669
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:42:54.869700
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/5669&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070503708422248204)

MetaTrader 5 / Examples


### Contents

- [Objects of historical orders and deals](https://www.mql5.com/en/articles/5669#node01)
- [Collection of historical orders and deals](https://www.mql5.com/en/articles/5669#node02)
- [What's next?](https://www.mql5.com/en/articles/5669#node03)

[In the previous article](https://www.mql5.com/en/articles/5654), we started creating a large cross-platform library simplifying the development of programs for [MetaTrader 5](https://www.metaquotes.net/en/metatrader5 "https://www.metaquotes.net/en/metatrader5") and MetaTrader 4 platforms. We created the COrder abstract object which is a base object for storing data on history orders and deals, as well as on market orders and positions.

In this part, we will proceed developing all the necessary objects for storing account history data in collections, prepare the collection of history orders and deals, as well as modify and improve already created objects and enumerations.

### Objects of historical orders and deals

COrder base object contains all data on any account object, be it a market order (an order to perform an action), a pending order, a deal or a position. In order for us to freely operate all these objects separately from each other, we will develop several classes based on the abstract COrder. These classes will accurately indicate the object's affiliation to its type.

The list of history orders and deals may feature several types of such objects: removed pending order, placed market order and deal (market order execution result). There are also two more object types in MQL4: balance and credit operations (in MQL5, these data are stored in the deal properties).

Create the new class CHistoryOrder in the Objects folder of the library.

To do this, right-click the **Objects** folder and select "New file" menu item (Ctrl+N). In the newly opened MQL5 Wizard, select "New Class" and click Next. Enter CHistoryOrder **(1)** in the class name field, specify the name of the COrder **(2)** abstract class in the base class field and click Finish.

![](https://c.mql5.com/2/36/CreateCHistoryOrder_en.png)

After that, the HistoryOrder.mqh **(3)** file is generated in the Objects folder. Open it:

```
//+------------------------------------------------------------------+
//|                                                 HistoryOrder.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CHistoryOrder : public COrder
  {
private:

public:
                     CHistoryOrder();
                    ~CHistoryOrder();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHistoryOrder::CHistoryOrder()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHistoryOrder::~CHistoryOrder()
  {
  }
//+------------------------------------------------------------------+
```

For now, this is just a class template. If we try to compile it, we will see the five already familiar errors— the new class derived from COrder knows nothing about its parent. Add including the Order.mqh file:

```
//+------------------------------------------------------------------+
//|                                                 HistoryOrder.mqh |
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
//|                                                                  |
//+------------------------------------------------------------------+
class CHistoryOrder : public COrder
  {
private:

public:
                     CHistoryOrder();
                    ~CHistoryOrder();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHistoryOrder::CHistoryOrder()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHistoryOrder::~CHistoryOrder()
  {
  }
//+------------------------------------------------------------------+
```

Now all is compiled with no issues.

The class will be quite small. We need to redefine the COrder parent class methods, which return the order properties maintenance flags, as well as set passing the order ticket to the class in the constructor:

```
//+------------------------------------------------------------------+
//|                                                 HistoryOrder.mqh |
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
//| History market order                                             |
//+------------------------------------------------------------------+
class CHistoryOrder : public COrder
  {
public:
   //--- Constructor
                     CHistoryOrder(const ulong ticket) : COrder(ORDER_STATUS_HISTORY_ORDER,ticket) {}
   //--- Supported integer order properties
   virtual bool      SupportProperty(ENUM_ORDER_PROP_INTEGER property);
  };
//+------------------------------------------------------------------+
//| Return 'true' if the order supports the passed property,         |
//| otherwise, return 'false'                                        |
//+------------------------------------------------------------------+
bool CHistoryOrder::SupportProperty(ENUM_ORDER_PROP_INTEGER property)
  {
   if(property==ORDER_PROP_TIME_EXP       ||
      property==ORDER_PROP_DEAL_ENTRY     ||
      property==ORDER_PROP_TIME_UPDATE    ||
      property==ORDER_PROP_TIME_UPDATE_MSC
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
```

Thus, a ticket of a selected order is passed to the class constructor, while an order status (history order) and its ticket are passed to the protected constructor of the COrder parent object.

We have also redefined the virtual method of the parent class returning support for the integer order properties. The methods returning the support for the real and string properties of the order have been left unchanged — these parent class methods always return 'true', and we will assume that the historical order supports all real and string properties, so we will not redefine them yet.

Check the property in the method supporting order integer properties. If this is an expiration time, deal direction or position change time, return 'false'. Such properties are not supported by market orders. All the remaining properties are supported and 'true' is returned.

In a similar fashion, create the CHistoryPending class of the historic (removed) pending order and the CHistoryDeal class of historical deal:

```
//+------------------------------------------------------------------+
//|                                               HistoryPending.mqh |
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
//| Removed pending order                                            |
//+------------------------------------------------------------------+
class CHistoryPending : public COrder
  {
public:
   //--- Constructor
                     CHistoryPending(const ulong ticket) : COrder(ORDER_STATUS_HISTORY_PENDING,ticket) {}
   //--- Supported order properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_ORDER_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_ORDER_PROP_INTEGER property);
  };
//+------------------------------------------------------------------+
//| Return 'true' if the order supports the passed property,         |
//| otherwise, return 'false'                                        |
//+------------------------------------------------------------------+
bool CHistoryPending::SupportProperty(ENUM_ORDER_PROP_INTEGER property)
  {
   if(property==ORDER_PROP_PROFIT_PT         ||
      property==ORDER_PROP_DEAL_ORDER        ||
      property==ORDER_PROP_DEAL_ENTRY        ||
      property==ORDER_PROP_TIME_UPDATE       ||
      property==ORDER_PROP_TIME_UPDATE_MSC   ||
      property==ORDER_PROP_TICKET_FROM       ||
      property==ORDER_PROP_TICKET_TO         ||
      property==ORDER_PROP_CLOSE_BY_SL       ||
      property==ORDER_PROP_CLOSE_BY_TP
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if the order supports the passed property,         |
//| otherwise, returns 'false'                                       |
//+------------------------------------------------------------------+
bool CHistoryPending::SupportProperty(ENUM_ORDER_PROP_DOUBLE property)
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

```
//+------------------------------------------------------------------+
//|                                                  HistoryDeal.mqh |
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
//| Historical deal                                                  |
//+------------------------------------------------------------------+
class CHistoryDeal : public COrder
  {
public:
   //--- Constructor
                     CHistoryDeal(const ulong ticket) : COrder(ORDER_STATUS_DEAL,ticket) {}
   //--- Supported deal properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_ORDER_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_ORDER_PROP_INTEGER property);
  };
//+------------------------------------------------------------------+
//| Return 'true' if the order supports the passed property,         |
//| otherwise, return 'false'                                        |
//+------------------------------------------------------------------+
bool CHistoryDeal::SupportProperty(ENUM_ORDER_PROP_INTEGER property)
  {
   if(property==ORDER_PROP_TIME_EXP          ||
      property==ORDER_PROP_PROFIT_PT         ||
      property==ORDER_PROP_POSITION_BY_ID    ||
      property==ORDER_PROP_TIME_UPDATE       ||
      property==ORDER_PROP_TIME_UPDATE_MSC   ||
      property==ORDER_PROP_STATE             ||
      (
       this.OrderType()==DEAL_TYPE_BALANCE &&
       (
        property==ORDER_PROP_POSITION_ID     ||
        property==ORDER_PROP_POSITION_BY_ID  ||
        property==ORDER_PROP_TICKET_FROM     ||
        property==ORDER_PROP_TICKET_TO       ||
        property==ORDER_PROP_DEAL_ORDER      ||
        property==ORDER_PROP_MAGIC           ||
        property==ORDER_PROP_TIME_CLOSE      ||
        property==ORDER_PROP_TIME_CLOSE_MSC  ||
        property==ORDER_PROP_CLOSE_BY_SL     ||
        property==ORDER_PROP_CLOSE_BY_TP
       )
      )
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
bool CHistoryDeal::SupportProperty(ENUM_ORDER_PROP_DOUBLE property)
  {
   if(property==ORDER_PROP_TP                ||
      property==ORDER_PROP_SL                ||
      property==ORDER_PROP_PRICE_CLOSE       ||
      property==ORDER_PROP_VOLUME_CURRENT    ||
      property==ORDER_PROP_PRICE_STOP_LIMIT  ||
      (
       this.OrderType()==DEAL_TYPE_BALANCE &&
       (
        property==ORDER_PROP_PRICE_OPEN      ||
        property==ORDER_PROP_COMMISSION      ||
        property==ORDER_PROP_SWAP            ||
        property==ORDER_PROP_VOLUME
       )
      )
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
```

We have created three order objects the collection of historical orders is to be based on. All of them are inherited from the COrder abstract order base class. They have its properties but allow returning only the properties supported by these order type. All of them are to be located in a single collection list (collection of historical orders), from which we are going to receive all necessary data on account history in any composition and order.

Not all supported or unsupported properties are taken into account in the SupportProperty () methods for displaying order properties in the journal. For example, only three types are considered for deals: buy, sell and balance operation.

Properties that are not considered yet and not explicitly specified in the methods returning support for them will always be printed. Then you can add them to the method in order not to print the properties that always return zero values regardless of the situation (unsupported ones).

* * *

### Collection of historical orders and deals

It is always helpful to have the account history at hand. The terminal provides it and gives the tools to get it in programs. However, our current tasks require a custom list we are able to sort and re-arrange to return necessary data to our programs. This means the change of the previous account history status should be checked at each tick. If a change is detected, the list of historical orders and deals is to be recalculated. But sorting the entire history at each tick is too resource-intensive. Therefore, we will only make additions to our list of new data, while previous data is already stored in the list.

Let's create the new class **CHistoryCollection** in the **Collections** folder:

Right-click on the **Collections** folder, select "New File", select "New Class" in MQL Wizard window and click Next. Enter CHistoryCollection class name, leave the base class field blank and click Finish.

![](https://c.mql5.com/2/36/CreateCHistoryCollection_en.png)

New file HistoryCollection is generated in the Collections folder:

```
//+------------------------------------------------------------------+
//|                                            HistoryCollection.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CHistoryCollection
  {
private:

public:
                     CHistoryCollection();
                    ~CHistoryCollection();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHistoryCollection::CHistoryCollection()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHistoryCollection::~CHistoryCollection()
  {
  }
//+------------------------------------------------------------------+
```

Let's fill it in.

We will use the [dynamic list of pointers to object instances](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) of the CArrayObj standard library for the list. Include it to the file and define it right away (you can use the right-click context menu to do that):

![](https://c.mql5.com/2/36/IncludeSB_en.png)

Include CArrayObj and define the list of historical orders and deals in the class private section:

```
//+------------------------------------------------------------------+
//|                                            HistoryCollection.mqh |
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
//+------------------------------------------------------------------+
//| Collection of historical orders and deals                        |
//+------------------------------------------------------------------+
class CHistoryCollection
  {
private:
   CArrayObj         m_list_all_orders;      // List of historical orders and deals

public:
                     CHistoryCollection();
                    ~CHistoryCollection();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHistoryCollection::CHistoryCollection()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHistoryCollection::~CHistoryCollection()
  {
  }
//+------------------------------------------------------------------+
```

We will need to save indices of the last orders and deals added to the collection. Besides, we need to know the difference between the past and present number of orders and deals, therefore we will create private class members for storing them:

```
//+------------------------------------------------------------------+
//|                                            HistoryCollection.mqh |
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
//+------------------------------------------------------------------+
//| Collection of historical orders and deals                        |
//+------------------------------------------------------------------+
class CHistoryCollection
  {
private:
   CArrayObj         m_list_all_orders;      // List of historical orders and deals
   int               m_index_order;          // Index of the last order added to the collection from the terminal history list (MQL4, MQL5)
   int               m_index_deal;           // Index of the last deal added to the collection from the terminal history list (MQL5)
   int               m_delta_order;          // Difference in the number of orders as compared to the past check
   int               m_delta_deal;           // Difference in the number of deals as compared to the past check
public:
                     CHistoryCollection();
                    ~CHistoryCollection();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHistoryCollection::CHistoryCollection()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHistoryCollection::~CHistoryCollection()
  {
  }
//+------------------------------------------------------------------+
```

During the first launch, all private class members are reset and the history is re-calculated again. To do this, simply add the list of class members initialization in the class constructor and set the default criterion the collection list is to be sorted by.

Currently, we have the default constructor. Before writing its implementation, we should create an enumeration containing all possible criteria for sorting orders and deals in the collection list.

But first, let's arrange integer, real and string order properties for their more logical display in the journal. Open the **Defines.mqh** file from the library root folder and place enumeration members in the required order:

```
//+------------------------------------------------------------------+
//| Order, deal, position integer properties                         |
//+------------------------------------------------------------------+
enum ENUM_ORDER_PROP_INTEGER
  {
   ORDER_PROP_TICKET = 0,                                   // Order ticket
   ORDER_PROP_MAGIC,                                        // Order magic number
   ORDER_PROP_TIME_OPEN,                                    // Open time (MQL5 Deal time)
   ORDER_PROP_TIME_CLOSE,                                   // Close time (MQL5 Execution or removal time - ORDER_TIME_DONE)
   ORDER_PROP_TIME_OPEN_MSC,                                // Open time in milliseconds (MQL5 Deal time in msc)
   ORDER_PROP_TIME_CLOSE_MSC,                               // Close time in milliseconds (MQL5 Execution or removal time - ORDER_TIME_DONE_MSC)
   ORDER_PROP_TIME_EXP,                                     // Order expiration date (for pending orders)
   ORDER_PROP_STATUS,                                       // Order status (from the ENUM_ORDER_STATUS enumeration)
   ORDER_PROP_TYPE,                                         // Order type (MQL5 deal type)
   ORDER_PROP_DIRECTION,                                    // Direction (Buy, Sell)
   ORDER_PROP_REASON,                                       // Deal/order/position reason or source
   ORDER_PROP_POSITION_ID,                                  // Position ID
   ORDER_PROP_POSITION_BY_ID,                               // Opposite position ID
   ORDER_PROP_DEAL_ORDER,                                   // Order a deal is based on
   ORDER_PROP_DEAL_ENTRY,                                   // Deal direction – IN, OUT or IN/OUT
   ORDER_PROP_TIME_UPDATE,                                  // Position change time in seconds
   ORDER_PROP_TIME_UPDATE_MSC,                              // Position change time in milliseconds
   ORDER_PROP_TICKET_FROM,                                  // Parent order ticket
   ORDER_PROP_TICKET_TO,                                    // Derived order ticket
   ORDER_PROP_PROFIT_PT,                                    // Profit in points
   ORDER_PROP_CLOSE_BY_SL,                                  // Flag of closing by StopLoss
   ORDER_PROP_CLOSE_BY_TP,                                  // Flag of closing by TakeProfit
  };
#define ORDER_PROP_INTEGER_TOTAL    (22)                    // Total number of integer properties
//+------------------------------------------------------------------+
//| Order, deal, position real properties                            |
//+------------------------------------------------------------------+
enum ENUM_ORDER_PROP_DOUBLE
  {
   ORDER_PROP_PRICE_OPEN = ORDER_PROP_INTEGER_TOTAL,        // Open price (MQL5 deal price)
   ORDER_PROP_PRICE_CLOSE,                                  // Close price
   ORDER_PROP_SL,                                           // StopLoss price
   ORDER_PROP_TP,                                           // TakeProfit price
   ORDER_PROP_PROFIT,                                       // Profit
   ORDER_PROP_COMMISSION,                                   // Commission
   ORDER_PROP_SWAP,                                         // Swap
   ORDER_PROP_VOLUME,                                       // Volume
   ORDER_PROP_VOLUME_CURRENT,                               // Unexecuted volume
   ORDER_PROP_PROFIT_FULL,                                  // Profit+commission+swap
   ORDER_PROP_PRICE_STOP_LIMIT,                             // Limit order price when StopLimit order is activated
  };
#define ORDER_PROP_DOUBLE_TOTAL     (11)                    // Total number of real properties
//+------------------------------------------------------------------+
//| Order, deal, position string properties                          |
//+------------------------------------------------------------------+
enum ENUM_ORDER_PROP_STRING
  {
   ORDER_PROP_SYMBOL = (ORDER_PROP_INTEGER_TOTAL+ORDER_PROP_DOUBLE_TOTAL), // Order symbol
   ORDER_PROP_COMMENT,                                      // Order comment
   ORDER_PROP_EXT_ID                                        // Order ID in an external trading system
  };
#define ORDER_PROP_STRING_TOTAL     (3)                     // Total number of string properties
//+------------------------------------------------------------------+
```

Now let's add the enumeration with all possible types of order and deal sorting to the same file:

```
//+------------------------------------------------------------------+
//| Possible criteria of orders and deals sorting                    |
//+------------------------------------------------------------------+
enum ENUM_SORT_ORDERS_MODE
  {
   //--- Sort by integer properties
   SORT_BY_ORDER_TICKET          =  0,                      // Sort by order ticket
   SORT_BY_ORDER_MAGIC           =  1,                      // Sort by order magic number
   SORT_BY_ORDER_TIME_OPEN       =  2,                      // Sort by order open time
   SORT_BY_ORDER_TIME_CLOSE      =  3,                      // Sort by order close time
   SORT_BY_ORDER_TIME_OPEN_MSC   =  4,                      // Sort by order open time in milliseconds
   SORT_BY_ORDER_TIME_CLOSE_MSC  =  5,                      // Sort by order close time in milliseconds
   SORT_BY_ORDER_TIME_EXP        =  6,                      // Sort by order expiration date
   SORT_BY_ORDER_STATUS          =  7,                      // Sort by order status (market order/pending order/deal)
   SORT_BY_ORDER_TYPE            =  8,                      // Sort by order type
   SORT_BY_ORDER_REASON          =  10,                     // Sort by deal/order/position reason/source
   SORT_BY_ORDER_POSITION_ID     =  11,                     // Sort by position ID
   SORT_BY_ORDER_POSITION_BY_ID  =  12,                     // Sort by opposite position ID
   SORT_BY_ORDER_DEAL_ORDER      =  13,                     // Sort by order a deal is based on
   SORT_BY_ORDER_DEAL_ENTRY      =  14,                     // Sort by deal direction – IN, OUT or IN/OUT
   SORT_BY_ORDER_TIME_UPDATE     =  15,                     // Sort by position change time in seconds
   SORT_BY_ORDER_TIME_UPDATE_MSC =  16,                     // Sort by position change time in milliseconds
   SORT_BY_ORDER_TICKET_FROM     =  17,                     // Sort by parent order ticket
   SORT_BY_ORDER_TICKET_TO       =  18,                     // Sort by derived order ticket
   SORT_BY_ORDER_PROFIT_PT       =  19,                     // Sort by order profit in points
   SORT_BY_ORDER_CLOSE_BY_SL     =  20,                     // Sort by order closing by StopLoss flag
   SORT_BY_ORDER_CLOSE_BY_TP     =  21,                     // Sort by order closing by TakeProfit flag
   //--- Sort by real properties
   SORT_BY_ORDER_PRICE_OPEN      =  ORDER_PROP_INTEGER_TOTAL,// Sort by open price
   SORT_BY_ORDER_PRICE_CLOSE     =  23,                     // Sort by close price
   SORT_BY_ORDER_SL              =  24,                     // Sort by StopLoss price
   SORT_BY_ORDER_TP              =  25,                     // Sort by TakeProfit price
   SORT_BY_ORDER_PROFIT          =  26,                     // Sort by profit
   SORT_BY_ORDER_COMMISSION      =  27,                     // Sort by commission
   SORT_BY_ORDER_SWAP            =  28,                     // Sort by swap
   SORT_BY_ORDER_VOLUME          =  29,                     // Sort by volume
   SORT_BY_ORDER_VOLUME_CURRENT  =  30,                     // Sort by unexecuted volume
   SORT_BY_ORDER_PROFIT_FULL     =  31,                     // Sort by profit+commission+swap criterion
   SORT_BY_ORDER_PRICE_STOP_LIMIT=  32,                     // Sort by Limit order when StopLimit order is activated
   //--- Sort by string properties
   SORT_BY_ORDER_SYMBOL          =  ORDER_PROP_INTEGER_TOTAL+ORDER_PROP_DOUBLE_TOTAL,// Sort by symbol
   SORT_BY_ORDER_COMMENT         =  34,                     // Sort by comment
   SORT_BY_ORDER_EXT_ID          =  35                      // Sort by order ID in an external trading system
  };
//+------------------------------------------------------------------+
```

**Note**: Indices of the sorting enumeration members should coincide with the ones of the properties enumeration members, since the list should be sorted by the same value that is used for searching in that list.

As we can see, the ORDER\_PROP\_DIRECTION sorting property has been skipped in that list since this is a service property used for the library needs, just like other custom properties we have added previously. However, they may need sorting. This is why they have been left.

Now we can implement the constructor of the CHistoryCollection class:

```
//+------------------------------------------------------------------+
//|                                            HistoryCollection.mqh |
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
#include "..\DELib.mqh"
//+------------------------------------------------------------------+
//| Collection of historical orders and deals                        |
//+------------------------------------------------------------------+
class CHistoryCollection
  {
private:
   CArrayObj         m_list_all_orders;      // List of historical orders and deals
   int               m_index_order;          // Index of the last order added to the collection from the terminal history list (MQL4, MQL5)
   int               m_index_deal;           // Index of the last deal added to the collection from the terminal history list (MQL5)
   int               m_delta_order;          // Difference in the number of orders as compared to the past check
   int               m_delta_deal;           // Difference in the number of deals as compared to the past check
public:
                     CHistoryCollection();
                    ~CHistoryCollection();
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHistoryCollection::CHistoryCollection(void) : m_index_deal(0),
                                               m_delta_deal(0),
                                               m_index_order(0),
                                               m_delta_order(0)
  {
   m_list_all_orders.Sort(SORT_BY_ORDER_TIME_CLOSE);
  }
//+------------------------------------------------------------------+
```

Let's analyze the listing.

Since the class constructor uses only the value of a newly added enumeration, we should include the Defines.mqh file to the class file.

While preparing the base class of the abstract order in the first article, we developed the **DELib.mqh** library of service functions and included the Defines.mqh file with all the necessary enumerations and macro substitutions to it. Therefore, we are going to include the library of service functions.

In the initialization list of the constructor, all indices and values of the differences between the current and previous numbers are reset, and the default sorting by close time is specified in the constructor body.

Now it is time to start collecting information from the account and save it in the collection list. To do this, go through the account history in loops and specify each order in the list. If the number of orders or deals has changed compared to the previous check, then set the flag of the occurred trading event. It is required to send messages about the occurred new event in the account history to an external program.

Declare the trading event flag in the private class section, and Refresh() method for updating the historical collection in the public one:

```
//+------------------------------------------------------------------+
//| Collection of historical orders and deals                        |
//+------------------------------------------------------------------+
class CHistoryCollection
  {
private:
   CArrayObj         m_list_all_orders;      // List of all historical orders and deals
   bool              m_is_trade_event;       // Trading event flag
   int               m_index_order;          // Index of the last order added to the collection from the terminal history list (MQL4, MQL5)
   int               m_index_deal;           // Index of the last deal added to the collection from the terminal history list (MQL5)
   int               m_delta_order;          // Difference in the number of orders as compared to the past check
   int               m_delta_deal;           // Difference in the number of deals as compared to the past check
public:
                     CHistoryCollection();
   //--- Update the list of orders, fill data on the number of new ones and set the trading event flag
   void              Refresh(void);
  };
//+------------------------------------------------------------------+
```

To implement updating the list of collection orders, we need one more macro substitution to request the full history data. The [HistorySelect()](https://www.mql5.com/en/docs/trading/historyselect) function is used for that. The dates of the beginning and end of the required data are passed in its parameters. To obtain the full list of the account history, the initial date should be passed as 0, while the end one should be passed as [TimeCurrent()](https://www.mql5.com/en/docs/dateandtime/timecurrent). However, in that case, the returned history data may sometimes be incomplete. To avoid that, enter a date exceeding the current server time instead of TimeCurrent(). I am going to enter the maximum possible date: 31.12.3000 23:59:59. Another advantage here is that [custom symbols](https://www.mql5.com/en/docs/customsymbols) may contain such a date and obtaining history will still work.

Let's insert a new macro substitution to the Defines.mqh file:

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
#define END_TIME       (D'31.12.3000 23:59:59') // End time to request account history data
//+------------------------------------------------------------------+
```

Now, instead of entering TimeCurrent() as an end time, we are going to enter the END\_TIME macro.

**Implementing the update of the collection order list:**

```
//+------------------------------------------------------------------+
//| Update the order list                                            |
//+------------------------------------------------------------------+
void CHistoryCollection::Refresh(void)
  {
#ifdef __MQL4__
   int total=::OrdersHistoryTotal(),i=m_index_order;
   for(; i<total; i++)
     {
      if(!::OrderSelect(i,SELECT_BY_POS,MODE_HISTORY)) continue;
      ENUM_ORDER_TYPE order_type=(ENUM_ORDER_TYPE)::OrderType();
      //--- Closed positions and balance/credit operations
      if(order_type<ORDER_TYPE_BUY_LIMIT || order_type>ORDER_TYPE_SELL_STOP)
        {
         CHistoryOrder *order=new CHistoryOrder(::OrderTicket());
         if(order==NULL) continue;
         m_list_all_orders.InsertSort(order);
        }
      else
        {
         //--- Removed pending orders
         CHistoryPending *order=new CHistoryPending(::OrderTicket());
         if(order==NULL) continue;
         m_list_all_orders.InsertSort(order);
        }
     }
//---
   int delta_order=i-m_index_order;
   this.m_index_order=i;
   this.m_delta_order=delta_order;
   this.m_is_trade_event=(this.m_delta_order!=0 ? true : false);
//--- __MQL5__
#else
   if(!::HistorySelect(0,END_TIME)) return;
//--- Orders
   int total_orders=::HistoryOrdersTotal(),i=m_index_order;
   for(; i<total_orders; i++)
     {
      ulong order_ticket=::HistoryOrderGetTicket(i);
      if(order_ticket==0) continue;
      ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)::HistoryOrderGetInteger(order_ticket,ORDER_TYPE);
      if(type==ORDER_TYPE_BUY || type==ORDER_TYPE_SELL)
        {
         CHistoryOrder *order=new CHistoryOrder(order_ticket);
         if(order==NULL) continue;
         m_list_all_orders.InsertSort(order);
        }
      else
        {
         CHistoryPending *order=new CHistoryPending(order_ticket);
         if(order==NULL) continue;
         m_list_all_orders.InsertSort(order);
        }
     }
//--- save the index of the last added order and the difference as compared to the previous check
   int delta_order=i-this.m_index_order;
   this.m_index_order=i;
   this.m_delta_order=delta_order;

//--- Deals
   int total_deals=::HistoryDealsTotal(),j=m_index_deal;
   for(; j<total_deals; j++)
     {
      ulong deal_ticket=::HistoryDealGetTicket(j);
      if(deal_ticket==0) continue;
      CHistoryDeal *deal=new CHistoryDeal(deal_ticket);
      if(deal==NULL) continue;
      m_list_all_orders.InsertSort(deal);
     }
//--- save the index of the last added deal and the difference as compared to the previous check
   int delta_deal=j-this.m_index_deal;
   this.m_index_deal=j;
   this.m_delta_deal=delta_deal;
//--- Set the new event flag in history
   this.m_is_trade_event=(this.m_delta_order+this.m_delta_deal);
#endif
  }
//+------------------------------------------------------------------+
```

Check affiliation with MQL4 or MQL5. Let's analyze the code using MQL5 as an example as it is a bit more complicated.

First, request the full account history. If failed, exit until the next tick. After a successful history request, two lists are created — list of orders and deals.

First, move along the list of all orders in a loop. The initial index the loop moves from is a result of the previous loop operation (at the first start = 0). This allows us to move only through new orders that have appeared in the history since the last check instead of moving the resource-intensive loop along the entire history.

Next, get order ticket and its type. Create a new object according to the result of the order type check (either a historical market order or a removed pending order) and place it to the collection list [in sorted form](https://www.mql5.com/en/docs/standardlibrary/datastructures/carraystring/carraystringinsertsort) right away (we have already set sorting by close time before).

Upon completion of the loop operation, save the order new index so that a new loop starts from it. The difference between the previous and current loops operation results is a number of newly added orders.

The loop for working with deals is the same except that there is no need to divide deals by types. Instead, you can add a deal object to the collection list immediately.

The loop in the MQL4 code is almost the same except that the loop is taken from the entire possible history available for the account. The history length is specified by a user in the terminal's History tab, which means it is up to users to make sure the entire history is available if the program requires it. Another option is to use WinAPI for obtaining the entire history, which is beyond the scope of the articles.

Upon completion of the loops, the number of new orders and deals is checked. If the number is greater than zero, the flag of the occurred trading event is set.

We have implemented obtaining data in the historical collection class similar to the one implemented in the [test EA in the first part](https://www.mql5.com/en/articles/5654#node05), except that in the current case, different objects of historical orders are created separated by their status. In order to check how it all works, we need to get the created collection list of historical orders from the outside (from the program that is to use this library; in our case, this is a test EA).

To do this, add the GetList() method with no parameters to the public section of the CHistoryCollection class (in the next article, I will add methods for receiving the list with parameters):


```
//+------------------------------------------------------------------+
//| Collection of historical orders and deals                        |
//+------------------------------------------------------------------+
class CHistoryCollection
  {
private:
   CArrayObj         m_list_all_orders;      // List of all historical orders and deals
   bool              m_is_trade_event;       // Trading event flag
   int               m_index_order;          // Index of the last order added to the collection from the terminal history list (MQL4, MQL5)
   int               m_index_deal;           // Index of the last deal added to the collection from the terminal history list (MQL5)
   int               m_delta_order;          // Difference in the number of orders as compared to the past check
   int               m_delta_deal;           // Difference in the number of deals as compared to the past check
public:
   //--- Return the full collection list 'as is'
   CArrayObj*        GetList(void)           { return &m_list_all_orders;  }
//--- Constructor
                     CHistoryCollection();
   //--- Update the list of orders, fill data on the number of new ones and set the trading event flag
   void              Refresh(void);
  };
//+------------------------------------------------------------------+
```

As can be seen from the listing, the pointer to the list (not the object itself) is returned. This is sufficient for using the list in the programs. However, this is insufficient for quick and easy use of the library. In the next article devoted to the library, I will implement an easy access to the necessary data on request. In the coming articles, I am going to simplify the access even further by creating special functions for working with the library in custom programs.

Let's see how the list is filled. To do this, create a small EA. In the **TestDoEasy** folder (that we have already created in the first part), generate the **Part02** subfolder. It will contain the file of the second test EA named TestDoEasyPart02 and connect the created collection to it:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart02.mq5 |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Collections\HistoryCollection.mqh>
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
```

Since the list now stores the objects of various types inherited from a single COrder parent, let's implement the display of descriptions of selected order objects in the journal. To do this, create the enumeration with selection of displayed order types in the inputs, as well as a collection object to read account data from:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart02.mq5 |
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
//--- global variables
CHistoryCollection history;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
```

Like in the previous test EA from the first part of the library description, read account history in the OnInit() handler and display information about orders to the journal in a loop:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- update history
   history.Refresh();
//--- get the pointer to the full collection list
   CArrayObj* list=history.GetList();
   if(list==NULL)
     {
      Print("Could not get collection list");
      return INIT_FAILED;
     }
   int total=list.Total();
   for(int i=0;i<total;i++)
     {
      //--- get order from the list
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
```

Here, create a pointer to the collection list and receive it from CHistoryCollection using the GetList() method created there.

Next, in a loop, get the order object from the list, check its status and permission to display data in the journal in the EA settings.

Depending on the results, display data in the journal or not.

Although, we get the COrder base object from the list, only the data inherent to the descendants of the base order are displayed in the journal — market order **s**, pending orders and deals with redefined virtual methods that return the flag indicating the order supports its inherent properties only.

Compile and launch the EA. The journal displays data on selected order and deal types:

![](https://c.mql5.com/2/36/Result1_en.gif)

If you look closely at the types of displayed properties, we will see the properties that are not typical for MQL5 orders: profit, swap, commission and profit in points.

**Let's make some improvements and additions in the already created objects.**

Add the unsupported property MQL5 profit in points to the SupportProperty(ENUM\_ORDER\_PROP\_INTEGER property) method of the CHistoryOrder class:

```
//+------------------------------------------------------------------+
//| Return 'true' if an order supports the passed property,          |
//| otherwise, return 'false'                                        |
//+------------------------------------------------------------------+
bool CHistoryOrder::SupportProperty(ENUM_ORDER_PROP_INTEGER property)
  {
   if(property==ORDER_PROP_TIME_EXP       ||
      property==ORDER_PROP_DEAL_ENTRY     ||
      property==ORDER_PROP_TIME_UPDATE    ||
      property==ORDER_PROP_TIME_UPDATE_MSC
      #ifdef __MQL5__                     ||
      property==ORDER_PROP_PROFIT_PT
      #endif
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
```

and add yet another virtual method returning support of real properties by orders:

```
//+------------------------------------------------------------------+
//| Historical market order                                          |
//+------------------------------------------------------------------+
class CHistoryOrder : public COrder
  {
public:
   //--- Constructor
                     CHistoryOrder(const ulong ticket) : COrder(ORDER_STATUS_HISTORY_ORDER,ticket) {}
   //--- Supported integer properties of an order
   virtual bool      SupportProperty(ENUM_ORDER_PROP_INTEGER property);
   //--- Supported real properties of an order
   virtual bool      SupportProperty(ENUM_ORDER_PROP_DOUBLE property);
  };
//+------------------------------------------------------------------+
```

as well as its implementation:

```
//+------------------------------------------------------------------+
//| Return 'true' if an order supports the passed property,          |
//| otherwise, return 'false'                                        |
//+------------------------------------------------------------------+
bool CHistoryOrder::SupportProperty(ENUM_ORDER_PROP_DOUBLE property)
  {
#ifdef __MQL5__
   if(property==ORDER_PROP_PROFIT      ||
      property==ORDER_PROP_PROFIT_FULL ||
      property==ORDER_PROP_SWAP        ||
      property==ORDER_PROP_COMMISSION  ||
      property==ORDER_PROP_PRICE_STOP_LIMIT
     ) return false;
#endif
   return true;
  }
//+------------------------------------------------------------------+
```

If this is MQL5, profit, swap, commission, full profit and StopLimit order price are not supported. For MQL4 and other real order properties in MQL5, return the flag of order's support for real properties.

MQL5 orders also have the ORDER\_STATE property. This is an order status set in the [ENUM\_ORDER\_STATE](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_state) enumeration.

Add it to the list of integer order properties (the ENUM\_ORDER\_PROP\_INTEGER enumeration in the Defines.mqh files):

```
//+------------------------------------------------------------------+
//| Order, deal, position integer properties                         |
//+------------------------------------------------------------------+
enum ENUM_ORDER_PROP_INTEGER
  {
   ORDER_PROP_TICKET = 0,                                   // Order ticket
   ORDER_PROP_MAGIC,                                        // Order magic number
   ORDER_PROP_TIME_OPEN,                                    // Open time (MQL5 Deal time)
   ORDER_PROP_TIME_CLOSE,                                   // Close time (MQL5 Execution or removal time - ORDER_TIME_DONE)
   ORDER_PROP_TIME_OPEN_MSC,                                // Open time in milliseconds (MQL5 Deal time in msc)
   ORDER_PROP_TIME_CLOSE_MSC,                               // Close time in milliseconds (MQL5 Execution or removal time - ORDER_TIME_DONE_MSC)
   ORDER_PROP_TIME_EXP,                                     // Order expiration date (for pending orders)
   ORDER_PROP_STATUS,                                       // Order status (from the ENUM_ORDER_STATUS enumeration)
   ORDER_PROP_TYPE,                                         // Order type (MQL5 deal type)
   ORDER_PROP_DIRECTION,                                    // Direction (Buy, Sell)
   ORDER_PROP_REASON,                                       // Deal/order/position reason or source
   ORDER_PROP_STATE,                                        // Order status (from the ENUM_ORDER_STATE enumeration)
   ORDER_PROP_POSITION_ID,                                  // Position ID
   ORDER_PROP_POSITION_BY_ID,                               // Opposite position ID
   ORDER_PROP_DEAL_ORDER,                                   // Order a deal is based on
   ORDER_PROP_DEAL_ENTRY,                                   // Deal direction – IN, OUT or IN/OUT
   ORDER_PROP_TIME_UPDATE,                                  // Position change time in seconds
   ORDER_PROP_TIME_UPDATE_MSC,                              // Position change time in milliseconds
   ORDER_PROP_TICKET_FROM,                                  // Parent order ticket
   ORDER_PROP_TICKET_TO,                                    // Derived order ticket
   ORDER_PROP_PROFIT_PT,                                    // Profit in points
   ORDER_PROP_CLOSE_BY_SL,                                  // Flag of closing by StopLoss
   ORDER_PROP_CLOSE_BY_TP,                                  // Flag of closing by TakeProfit
  };
#define ORDER_PROP_INTEGER_TOTAL    (23)                    // Total number of integer properties
//+------------------------------------------------------------------+
```

and make sure to change the number of order's integer properties from 22 to **23** in the ORDER\_PROP\_INTEGER\_TOTAL macro substitution indicating the number of integer order properties and used to calculate an exact "address" of the necessary order property.

In the same file, add the new property to the possible order sorting criteria, so that we are able to sort orders by this new property, and implement more convenient calculation of indices in case of a subsequent change of enumerations:

```
//+------------------------------------------------------------------+
//| Possible criteria of orders and deals sorting                    |
//+------------------------------------------------------------------+
#define FIRST_DBL_PROP              (ORDER_PROP_INTEGER_TOTAL)
#define FIRST_STR_PROP              (ORDER_PROP_INTEGER_TOTAL+ORDER_PROP_DOUBLE_TOTAL)
enum ENUM_SORT_ORDERS_MODE
  {
   //--- Sort by integer properties
   SORT_BY_ORDER_TICKET          =  0,                      // Sort by order ticket
   SORT_BY_ORDER_MAGIC           =  1,                      // Sort by order magic number
   SORT_BY_ORDER_TIME_OPEN       =  2,                      // Sort by order open time
   SORT_BY_ORDER_TIME_CLOSE      =  3,                      // Sort by order close time
   SORT_BY_ORDER_TIME_OPEN_MSC   =  4,                      // Sort by order open time in milliseconds
   SORT_BY_ORDER_TIME_CLOSE_MSC  =  5,                      // Sort by order close time in milliseconds
   SORT_BY_ORDER_TIME_EXP        =  6,                      // Sort by order expiration date
   SORT_BY_ORDER_STATUS          =  7,                      // Sort by order status (market order/pending order/deal)
   SORT_BY_ORDER_TYPE            =  8,                      // Sort by order type
   SORT_BY_ORDER_REASON          =  10,                     // Sort by deal/order/position reason/source
   SORT_BY_ORDER_STATE           =  11,                     // Sort by order status
   SORT_BY_ORDER_POSITION_ID     =  12,                     // Sort by position ID
   SORT_BY_ORDER_POSITION_BY_ID  =  13,                     // Sort by opposite position ID
   SORT_BY_ORDER_DEAL_ORDER      =  14,                     // Sort by order a deal is based on
   SORT_BY_ORDER_DEAL_ENTRY      =  15,                     // Sort by deal direction – IN, OUT or IN/OUT
   SORT_BY_ORDER_TIME_UPDATE     =  16,                     // Sort by position change time in seconds
   SORT_BY_ORDER_TIME_UPDATE_MSC =  17,                     // Sort by position change time in milliseconds
   SORT_BY_ORDER_TICKET_FROM     =  18,                     // Sort by parent order ticket
   SORT_BY_ORDER_TICKET_TO       =  19,                     // Sort by derived order ticket
   SORT_BY_ORDER_PROFIT_PT       =  20,                     // Sort by order profit in points
   SORT_BY_ORDER_CLOSE_BY_SL     =  21,                     // Sort by order closing by StopLoss flag
   SORT_BY_ORDER_CLOSE_BY_TP     =  22,                     // Sort by order closing by TakeProfit flag
   //--- Sort by real properties
   SORT_BY_ORDER_PRICE_OPEN      =  FIRST_DBL_PROP,         // Sort by open price
   SORT_BY_ORDER_PRICE_CLOSE     =  FIRST_DBL_PROP+1,       // Sort by close price
   SORT_BY_ORDER_SL              =  FIRST_DBL_PROP+2,       // Sort by StopLoss price
   SORT_BY_ORDER_TP              =  FIRST_DBL_PROP+3,       // Sort by TakeProfit price
   SORT_BY_ORDER_PROFIT          =  FIRST_DBL_PROP+4,       // Sort by profit
   SORT_BY_ORDER_COMMISSION      =  FIRST_DBL_PROP+5,       // Sort by commission
   SORT_BY_ORDER_SWAP            =  FIRST_DBL_PROP+6,       // Sort by swap
   SORT_BY_ORDER_VOLUME          =  FIRST_DBL_PROP+7,       // Sort by volume
   SORT_BY_ORDER_VOLUME_CURRENT  =  FIRST_DBL_PROP+8,       // Sort by unexecuted volume
   SORT_BY_ORDER_PROFIT_FULL     =  FIRST_DBL_PROP+9,       // Sort by profit+commission+swap criterion
   SORT_BY_ORDER_PRICE_STOP_LIMIT=  FIRST_DBL_PROP+10,      // Sort by Limit order when StopLimit order is activated
   //--- Sort by string properties
   SORT_BY_ORDER_SYMBOL          =  FIRST_STR_PROP,         // Sort by symbol
   SORT_BY_ORDER_COMMENT         =  FIRST_STR_PROP+1,       // Sort by comment
   SORT_BY_ORDER_EXT_ID          =  FIRST_STR_PROP+2        // Sort by order ID in an external trading system
  };
//+------------------------------------------------------------------+
```

In the protected section of the COrder abstract order class of the Order.mqh file, declare the OrderState() method writing its status from the [ENUM\_ORDER\_STATE](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_state) enumeration to the order properties:

```
protected:
   //--- Protected parametric constructor
                     COrder(ENUM_ORDER_STATUS order_status,const ulong ticket);

   //--- Get and return integer properties of a selected order from its parameters
   long              OrderMagicNumber(void)        const;
   long              OrderTicket(void)             const;
   long              OrderTicketFrom(void)         const;
   long              OrderTicketTo(void)           const;
   long              OrderPositionID(void)         const;
   long              OrderPositionByID(void)       const;
   long              OrderOpenTimeMSC(void)        const;
   long              OrderCloseTimeMSC(void)       const;
   long              OrderType(void)               const;
   long              OrderState(void)              const;
   long              OrderTypeByDirection(void)    const;
   long              OrderTypeFilling(void)        const;
   long              OrderTypeTime(void)           const;
   long              OrderReason(void)             const;
   long              DealOrder(void)               const;
   long              DealEntry(void)               const;
   bool              OrderCloseByStopLoss(void)    const;
   bool              OrderCloseByTakeProfit(void)  const;
   datetime          OrderOpenTime(void)           const;
   datetime          OrderCloseTime(void)          const;
   datetime          OrderExpiration(void)         const;
   datetime          PositionTimeUpdate(void)      const;
   datetime          PositionTimeUpdateMSC(void)   const;
```

and add its implementation:

```
//+------------------------------------------------------------------+
//| Return the order status                                          |
//+------------------------------------------------------------------+
long COrder::OrderState(void) const
  {
#ifdef __MQL4__
   return ORDER_STATE_FILLED;
#else
   long res=0;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetInteger(m_ticket,ORDER_STATE); break;
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetInteger(ORDER_STATE);                 break;
      case ORDER_STATUS_MARKET_ACTIVE     :
      case ORDER_STATUS_DEAL              :
      default                             : res=0;                                              break;
     }
   return res;
#endif
  }
//+------------------------------------------------------------------+
```

In case of MQL4, let's return the order full execution for now. In case of MQL5,  either return 0 (if this is a deal or a position), or order state (if this is a market or pending order) depending on the order status.

In the public section of the COrder class, declare the method returning the order status description:

```
//+------------------------------------------------------------------+
//| Descriptions of order object properties                          |
//+------------------------------------------------------------------+
   //--- Get description of an order's (1) integer, (2) real and (3) string property
   string            GetPropertyDescription(ENUM_ORDER_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_ORDER_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_ORDER_PROP_STRING property);
   //--- Return order status name
   string            StatusDescription(void)    const;
   //---  Return order or position name
   string            TypeDescription(void)      const;
   //--- Return order status description
   string            StateDescription(void)     const;
   //--- Return deal direction name
   string            DealEntryDescription(void) const;
   //--- Return order/position direction type
   string            DirectionDescription(void) const;
   //--- Send description of order properties to the journal (full_prop=true - all properties, false - only supported ones)
   void              Print(const bool full_prop=false);
```

as well as its implementation:

```
//+------------------------------------------------------------------+
//| Return order status description                                  |
//+------------------------------------------------------------------+
string COrder::StateDescription(void) const
  {
   if(this.Status()==ORDER_STATUS_DEAL || this.Status()==ORDER_STATUS_MARKET_ACTIVE)
      return "";
   else switch(this.StateOrder())
     {
      case ORDER_STATE_STARTED         :  return TextByLanguage("Ордер проверен на корректность, но еще не принят брокером","Order checked for correctness, but not yet accepted by broker");
      case ORDER_STATE_PLACED          :  return TextByLanguage("Ордер принят","Order accepted");
      case ORDER_STATE_CANCELED        :  return TextByLanguage("Ордер снят клиентом","Order withdrawn by client");
      case ORDER_STATE_PARTIAL         :  return TextByLanguage("Ордер выполнен частично","Order filled partially");
      case ORDER_STATE_FILLED          :  return TextByLanguage("Ордер выполнен полностью","Order filled");
      case ORDER_STATE_REJECTED        :  return TextByLanguage("Ордер отклонен","Order rejected");
      case ORDER_STATE_EXPIRED         :  return TextByLanguage("Ордер снят по истечении срока его действия","Order withdrawn upon expiration");
      case ORDER_STATE_REQUEST_ADD     :  return TextByLanguage("Ордер в состоянии регистрации (выставление в торговую систему)","Order in state of registration (placing in trading system)");
      case ORDER_STATE_REQUEST_MODIFY  :  return TextByLanguage("Ордер в состоянии модификации","Order in state of modification.");
      case ORDER_STATE_REQUEST_CANCEL  :  return TextByLanguage("Ордер в состоянии удаления","Order in deletion state");
      default                          :  return TextByLanguage("Неизвестное состояние","Unknown state");
     }
  }
//+------------------------------------------------------------------+
```

If this is a deal or a position, return an empty string, otherwise, check the order status and return its description.

Add return of the order status description into the GetPropertyDescription(ENUM\_ORDER\_PROP\_INTEGER property) method implementation:

```
//+------------------------------------------------------------------+
//| Return description of an order's integer property                |
//+------------------------------------------------------------------+
string COrder::GetPropertyDescription(ENUM_ORDER_PROP_INTEGER property)
  {
   return
     (
   //--- General properties
      property==ORDER_PROP_MAGIC             ?  TextByLanguage("Магик","Magic number")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TICKET            ?  TextByLanguage("Тикет","Ticket")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          " #"+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TICKET_FROM       ?  TextByLanguage("Тикет родительского ордера","Ticket of parent order")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          " #"+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TICKET_TO         ?  TextByLanguage("Тикет наследуемого ордера","Inherited order ticket")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          " #"+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TIME_OPEN         ?  TextByLanguage("Время открытия","Open time")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+::TimeToString(this.GetProperty(property),TIME_DATE|TIME_MINUTES|TIME_SECONDS)
         )  :
      property==ORDER_PROP_TIME_CLOSE        ?  TextByLanguage("Время закрытия","Close time")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+::TimeToString(this.GetProperty(property),TIME_DATE|TIME_MINUTES|TIME_SECONDS)
         )  :
      property==ORDER_PROP_TIME_EXP          ?  TextByLanguage("Дата экспирации","Expiration date")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          (this.GetProperty(property)==0     ?  TextByLanguage(": Не задана",": Not set") :
          ": "+::TimeToString(this.GetProperty(property),TIME_DATE|TIME_MINUTES|TIME_SECONDS))
         )  :
      property==ORDER_PROP_TYPE              ?  TextByLanguage("Тип","Type")+": "+this.TypeDescription()                   :
      property==ORDER_PROP_DIRECTION         ?  TextByLanguage("Тип по направлению","Type by direction")+": "+this.DirectionDescription() :

      property==ORDER_PROP_REASON            ?  TextByLanguage("Причина","Reason")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetReasonDescription(this.GetProperty(property))
         )  :
      property==ORDER_PROP_POSITION_ID       ?  TextByLanguage("Идентификатор позиции","Position ID")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": #"+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_DEAL_ORDER        ?  TextByLanguage("Сделка на основании ордера","Deal by order")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": #"+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_DEAL_ENTRY        ?  TextByLanguage("Направление сделки","Deal direction")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetEntryDescription(this.GetProperty(property))
         )  :
      property==ORDER_PROP_POSITION_BY_ID    ?  TextByLanguage("Идентификатор встречной позиции","Opposite position ID")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TIME_OPEN_MSC     ?  TextByLanguage("Время открытия в милисекундах","Open time in milliseconds")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)+" > "+TimeMSCtoString(this.GetProperty(property))
         )  :
      property==ORDER_PROP_TIME_CLOSE_MSC    ?  TextByLanguage("Время закрытия в милисекундах","Close time in milliseconds")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)+" > "+TimeMSCtoString(this.GetProperty(property))
         )  :
      property==ORDER_PROP_TIME_UPDATE       ?  TextByLanguage("Время изменения позиции","Position change time")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)!=0 ? ::TimeToString(this.GetProperty(property),TIME_DATE|TIME_MINUTES|TIME_SECONDS) : "0")
         )  :
      property==ORDER_PROP_TIME_UPDATE_MSC   ?  TextByLanguage("Время изменения позиции в милисекундах","Position change time in milliseconds")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)!=0 ? (string)this.GetProperty(property)+" > "+TimeMSCtoString(this.GetProperty(property)) : "0")
         )  :
      property==ORDER_PROP_STATE             ?  TextByLanguage("Состояние","Statе")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": \""+this.StateDescription()+"\""
         )  :
   //--- Additional property
      property==ORDER_PROP_STATUS            ?  TextByLanguage("Статус","Status")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": \""+this.StatusDescription()+"\""
         )  :
      property==ORDER_PROP_PROFIT_PT         ?  TextByLanguage("Прибыль в пунктах","Profit in points")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_CLOSE_BY_SL       ?  TextByLanguage("Закрытие по StopLoss","Close by StopLoss")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property) ? TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
         )  :
      property==ORDER_PROP_CLOSE_BY_TP       ?  TextByLanguage("Закрытие по TakeProfit","Close by TakeProfit")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property) ? TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

We have completed all the improvements.

It should be borne in mind that the library is being developed "live" and is a beta version, therefore various revisions, changes and additions can be implemented later.

### What's next?

In the next article, I am going to develop a class for a convenient selection and sorting of orders, deals and positions by any of the supported criteria and create a collection of market orders and positions.

All files of the current version of the library are attached below together with test EA files for you to test and download.

Leave your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/5669#node00)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/5669](https://www.mql5.com/ru/articles/5669)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/5669.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/5669/mql5.zip "Download MQL5.zip")(22.89 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/311563)**
(28)


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
23 Jan 2020 at 05:16

**Xiaowei Yan :**

Are you work for metaquote? Why the standard mqh file in mt5 are similar but not the same as yours?

1\. No. 2.

2\. What file are you talking about?

![Aleksandr Brown](https://c.mql5.com/avatar/2013/5/51829572-CC9A.JPG)

**[Aleksandr Brown](https://www.mql5.com/en/users/brown-aleks)**
\|
26 Jul 2020 at 18:57

This is a great idea! Thank you very much for the provided material.

I didn't download the attached files on purpose. As I studied the article, I decided to type everything with my own hands. It is easier to understand and memorise, and it is useful for practice.

And actually I found a small mark in this article. It does not say anywhere that in **HistoryCollection.mqh** in lines 14,15,16 should be connected **HistoryOrder.mqh**, **HistoryPending.** **mqh** and **HistoryDeal** **.mqh**.

For me, as a beginner, I had to strain my brain a lot. =))

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
26 Jul 2020 at 19:26

**Aleksandr Brown:**

This is a great idea! Thank you very much for providing the material.

I didn't download the attached files on purpose. As I studied the article, I decided to type everything with my own hands. It is easier to understand and memorise, and it is useful for practice.

And actually I found a small mark in this article. It does not say anywhere that **HistoryOrder** **.mqh**, **HistoryPending** **.mqh** and **HistoryDeal** **.mqh** should be connected in **HistoryCollection** **.** **mqh** in lines 14,15,16.

For me, as a beginner, I had to strain my brain a lot. =))

Thank you.

Well you see, sometimes "mistakes" make you think and find solutions ;).

At the initial stage, in the articles everything is "chewed up". But the further you go, the less detailed everything is described with the aim that a person carefully read the first articles and further on will understand everything himself about library constructions.

![Ildar Valiullin](https://c.mql5.com/avatar/2016/11/58284563-E435.jpg)

**[Ildar Valiullin](https://www.mql5.com/en/users/ildarvin)**
\|
5 Mar 2022 at 18:34

Very interesting work. I am studying it line by line. I try to rewrite the code manually, and I encountered an error in the string function COrder::StateDescription(void).

The compiler was swearing at StateOrder(). I found the solution in the archive files, and there is a lot of other things there. But it's even good for warming up my brain :)

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
5 Mar 2022 at 18:47

**Ildar Valiullin [#](https://www.mql5.com/ru/forum/305683/page2#comment_28131052):**

Very interesting work. I am studying it line by line. I try to rewrite the code manually, and encountered an error in the function string COrder::StateDescription(void).

The compiler was swearing at StateOrder(). I found the solution in the archive files, and there is a lot of other things there. But it's even good for warming up my brain :)

I'm glad you managed to find a solution yourself.

![Extracting structured data from HTML pages using CSS selectors](https://c.mql5.com/2/35/MQL5-CSS_selectors.png)[Extracting structured data from HTML pages using CSS selectors](https://www.mql5.com/en/articles/5706)

The article provides a description of a universal method for analyzing and converting data from HTML documents based on CSS selectors. Trading reports, tester reports, your favorite economic calendars, public signals, account monitoring and additional online quote sources will become available straight from MQL.

![Studying candlestick analysis techniques (part III): Library for pattern operations](https://c.mql5.com/2/35/Pattern_I__4.png)[Studying candlestick analysis techniques (part III): Library for pattern operations](https://www.mql5.com/en/articles/5751)

The purpose of this article is to create a custom tool, which would enable users to receive and use the entire array of information about patterns discussed earlier. We will create a library of pattern related functions which you will be able to use in your own indicators, trading panels, Expert Advisors, etc.

![Library for easy and quick development of MetaTrader programs (part III). Collection of market orders and positions, search and sorting](https://c.mql5.com/2/35/MQL5-avatar-doeasy__2.png)[Library for easy and quick development of MetaTrader programs (part III). Collection of market orders and positions, search and sorting](https://www.mql5.com/en/articles/5687)

In the first part, we started creating a large cross-platform library simplifying the development of programs for MetaTrader 5 and MetaTrader 4 platforms. Further on, we implemented the collection of history orders and deals. Our next step is creating a class for a convenient selection and sorting of orders, deals and positions in collection lists. We are going to implement the base library object called Engine and add collection of market orders and positions to the library.

![Library for easy and quick development of MetaTrader programs (part I). Concept, data management and first results](https://c.mql5.com/2/35/MQL5-avatar-doeasy.png)[Library for easy and quick development of MetaTrader programs (part I). Concept, data management and first results](https://www.mql5.com/en/articles/5654)

While analyzing a huge number of trading strategies, orders for development of applications for MetaTrader 5 and MetaTrader 4 terminals and various MetaTrader websites, I came to the conclusion that all this diversity is based mostly on the same elementary functions, actions and values appearing regularly in different programs. This resulted in DoEasy cross-platform library for easy and quick development of МetaТrader 5 and МetaТrader 4 applications.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/5669&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070503708422248204)

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