---
title: Library for easy and quick development of MetaTrader programs (part I). Concept, data management and first results
url: https://www.mql5.com/en/articles/5654
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:43:05.520960
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=kpfwnekljxxwwgdcnasjdnmxhayeqzzo&ssn=1769186583602896835&ssn_dr=0&ssn_sr=0&fv_date=1769186583&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F5654&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20I).%20Concept%2C%20data%20management%20and%20first%20results%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918658316460550&fz_uniq=5070505581027989270&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/5654#node01)
- [Concept](https://www.mql5.com/en/articles/5654#node02)
- [Data structure](https://www.mql5.com/en/articles/5654#node03)
- [First implementation](https://www.mql5.com/en/articles/5654#node04)
- [Test](https://www.mql5.com/en/articles/5654#node05)
- [What's next?](https://www.mql5.com/en/articles/5654#node06)

### Introduction

While analyzing a huge number of trading strategies, orders for development of applications for MetaTrader 5 and MetaTrader 4 terminals and various websites concerning scripts, indicators and robots for [MetaTrader](https://www.metaquotes.net/en/metatrader5 "MetaTrader 5 trading platform"), I came to the conclusion that all this diversity is based mostly on the same elementary functions, actions and values appearing regularly in different programs.

In fact, the logic of any program can be divided into many identical actions. The results of these actions are used to build the logic of applications. This has been repeatedly confirmed by the uniformity of questions asked on MQL4/MQL5 forums. Different users ask essentially the same questions about the algorithms and tasks they solve.

Considering all this, I decided to develop a large library featuring built-in functions for requesting and obtaining the necessary data. To use the data set of the proposed library, users only need to resort to the question-answer method for obtaining a huge amount of completely different data (in different combinations and sorting parameters) the library collects and stores in its database.

### Concept

Any data type can be represented as a set of objects featuring the same properties.

For example, a timeseries can be represented as a long ordered list, each subsequent cell of which stores an object having a set of properties that is of the same type as all other sets belonging to similar objects within the timeseries. The data of such an object is represented by the **MqlRates** structure:

_Structure for storing information about prices, volumes and spread_.

```
struct MqlRates
  {
   datetime time;         // period start time
   double   open;         // open price
   double   high;         // highest price per period
   double   low;          // lowest price per period
   double   close;        // close price
   long     tick_volume;  // tick volume
   int      spread;       // spread
   long     real_volume;  // exchange volume
  };
```

A set of tick data can also be portrayed as an ordered list where a tick represented by the **MqlTick** structure is an object with a fixed set of properties:

_Structure for storing the last prices by symbol. The structure is designed for obtaining the most needed data on the current prices in a timely manner._

```
struct MqlTick
  {
   datetime     time;          // Last price update time
   double       bid;           // Current Bid price
   double       ask;           // Current Ask price
   double       last;          // Current price of the last deal (Last)
   ulong        volume;        // Volume for the current Last price
   long         time_msc;      // Time of the last price update in milliseconds
   uint         flags;         // Tick flags
   double       volume_real;   // Volume for the current Last price with increased accuracy
  };
```

Any other data necessary for analyzing and preparing the program logic are also arranged as simple object lists.

All objects in one list will have the same type of data specific to this type of object regardless of whether it is a list of orders, deals or pending orders. For each specific object, we will develop a class featuring the minimally necessary functionality for storing, sorting and displaying data.

### Library data structure

As already mentioned, the library is to consist of the object list and provide the ability to select any of the list items by any custom criterion or property supported by that object. The library will collect the necessary data for storing and handling on its own. No human intervention is needed. Users will only apply results of their library queries.

I am going to describe all the steps of developing the library starting with the simplest topics and gradually adding the new functionality and data to the already existing one. The library is to be developed in "live" mode. Required edits and additions are to be implemented in each article. I believe, this style of presentation is most useful as it involves readers into the development.

The minimum structure of the library data is a set of different objects for describing the properties of the necessary data, while data collections are lists storing the corresponding objects.

We will use the class of the [dynamic array of pointers to CObject class instances and its derived classes](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) from the standard library data collection to arrange the lists. Since [objects of the base class of the CObject standard library](https://www.mql5.com/en/docs/standardlibrary/cobject) are required for storing in such a list, we will simply inherit each class of our objects from the CObject base class.

### First implementation

First, we will develop the library for hedge accounts of the MetaTrader 5 terminal. After preparing the minimum functionality, we will adjust its operation to MetaTrader 4. Next, after implementing the account history, current market positions and pending orders processing, we are going to add the ability to work with MetaTrader 5 netting accounts. Finally, we will fill the library with a variety of entire functions.

Let's start with the account history. Many strategies apply the results of past trades in one way or another, for example, results of previous deals, the methods of their closure (stop loss, take profit), price, etc. Besides, the results for the last day can be used as a starting point for the current work. The types of trading strategies are limitless out there, and our task is to provide quick access to all the diversity.

First, let's define the terminology for working with collections of historical orders, deals, market orders and positions. The MetaTrader 4 order system is different from the MetaTrader 5 one. While MetaTrader 4 features market and pending orders, in MetaTrader 5, an order is basically a trading request (market order) generating a deal, while the deal in turn generates a position. Besides, there are pending orders. In other words, we have at least three objects — an order, a deal and a position. In order to avoid being distracted by names and definitions, let's call the class for storing data by orders, deals and positions simply an abstract order class. Further on, everything is to be sorted by types (orders, deals, etc.) in our list (collection of history orders mentioned at the beginning of the article).

The order abstract class is to contain an order or deal ticket, as well as the entire data on the order's or deal's parameters and its status. The status shows what exactly the object is — an order, a deal or a position.

In the **<terminal data folder>\\MQL5\\Include**, create the **DoEasy** folder where the library files are to be stored. At this stage, we will need two more directories in the DoEasy folder: the **Objects** folder will store object classes, while **Collections** will contain data collections (object lists).

To find the terminal data directory, go to File -> Open Data Folder (Ctrl + Shift + D).

Now we can create the first class (abstract order class). Create a new class in the Objects folder and name it COrder **(1)**. Make sure to set the CObject **(2)** class of the standard library as a base one. In this case, our new object class is inherited from the CObject class, and it can be placed in the list of CArrayObj objects of the standard library.

![](https://c.mql5.com/2/35/CreateNewClass_en.png)

After clicking Finish, the new Order.mqh **(3)** file appears in the Objects folder of the library directory. For now, this is just a class work piece:

```
//+------------------------------------------------------------------+
//|                                                        Order.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/ru/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class COrder : public CObject
  {
private:

public:
                     COrder();
                    ~COrder();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
COrder::COrder()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
COrder::~COrder()
  {
  }
//+------------------------------------------------------------------+
```

When trying to compile the code, we get five errors. They indicate the absence of the CObject class the COrder class is derived from. Include the CObject class file to the listing and compile it once again. Now everything is fine.

```
//+------------------------------------------------------------------+
//|                                                        Order.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#include <Object.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class COrder : public CObject
  {
private:

public:
                     COrder();
                    ~COrder();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
COrder::COrder()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
COrder::~COrder()
  {
  }
//+------------------------------------------------------------------+
```

The object of the class is created in a loop by orders, deals or positions when the next order, deal or position is selected. In order to initialize the object's fields right away, we will create a private [constructor](https://www.mql5.com/en/docs/basis/types/classes#constructor) the object type (status) and ticket are passed to for subsequent identification. But first, let's place the new Defines.mqh include file to the project root folder, which stores all enumerations necessary for the library, as well as [macro substitutions](https://www.mql5.com/en/docs/basis/preprosessor/constant), constants and structures.

Currently, we need enumerations describing an order status and enumerations describing all parameters of orders, deals or positions. There will be three enumerations with order parameters: integer, real and string ones.

```
//+------------------------------------------------------------------+
//|                                                      Defines.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Abstract order type (status)                                     |
//+------------------------------------------------------------------+
enum ENUM_ORDER_STATUS
  {
   ORDER_STATUS_MARKET_PENDING,                             // Current pending order
   ORDER_STATUS_MARKET_ACTIVE,                              // Active market order
   ORDER_STATUS_HISTORY_ORDER,                              // History market order
   ORDER_STATUS_HISTORY_PENDING,                            // Removed pending order
   ORDER_STATUS_BALANCE,                                    // Balance operation
   ORDER_STATUS_CREDIT,                                     // Credit operation
   ORDER_STATUS_DEAL,                                       // Deal
   ORDER_STATUS_UNKNOWN                                     // Unknown status
  };
//+------------------------------------------------------------------+
//| Order, deal, position integer properties                         |
//+------------------------------------------------------------------+
enum ENUM_ORDER_PROP_INTEGER
  {
   ORDER_PROP_TICKET = 0,                                   // Order ticket
   ORDER_PROP_MAGIC,                                        // Order magic
   ORDER_PROP_TIME_OPEN,                                    // Open time
   ORDER_PROP_TIME_CLOSE,                                   // Close time
   ORDER_PROP_TIME_EXP,                                     // Order expiration date (for pending orders)
   ORDER_PROP_TYPE,                                         // Order and deal type
   ORDER_PROP_STATUS,                                       // Order status (from the ENUM_ORDER_STATUS enumeration)
   ORDER_PROP_REASON,                                       // Deal/order/position reason or source
   ORDER_PROP_POSITION_ID,                                  // Position ID
   ORDER_PROP_POSITION_BY_ID,                               // Opposite position ID
   ORDER_PROP_DEAL_ORDER,                                   // Order, based on which a deal is executed
   ORDER_PROP_DEAL_ENTRY,                                   // Deal direction – IN, OUT or IN/OUT
   ORDER_PROP_TIME_OPEN_MSC,                                // Open time in milliseconds
   ORDER_PROP_TIME_CLOSE_MSC,                               // Close time in milliseconds (execution or removal time - ORDER_TIME_DONE_MSC)
   ORDER_PROP_TIME_UPDATE,                                  // Position change time in seconds
   ORDER_PROP_TIME_UPDATE_MSC,                              // Position change time in milliseconds
   ORDER_PROP_TICKET_FROM,                                  // Parent order ticket
   ORDER_PROP_TICKET_TO,                                    // Derived order ticket
   ORDER_PROP_PROFIT_PT,                                    // Profit in points
   ORDER_PROP_CLOSE_BY_SL,                                  // Flag of closing by StopLoss
   ORDER_PROP_CLOSE_BY_TP,                                  // Flag of closing by TakeProfit
   ORDER_PROP_DIRECTION,                                    // Direction (Buy, Sell)
  };
#define ORDER_PROP_INTEGER_TOTAL    (22)                    // Total number of integer properties
//+------------------------------------------------------------------+
//| Order, deal, position real properties                            |
//+------------------------------------------------------------------+
enum ENUM_ORDER_PROP_DOUBLE
  {
   ORDER_PROP_PRICE_OPEN = ORDER_PROP_INTEGER_TOTAL,        // Open price (MQL5 deal price)
   ORDER_PROP_PRICE_CLOSE,                                  // Close price
   ORDER_PROP_PROFIT,                                       // Profit
   ORDER_PROP_COMMISSION,                                   // Commission
   ORDER_PROP_SWAP,                                         // Swap
   ORDER_PROP_VOLUME,                                       // Volume
   ORDER_PROP_VOLUME_CURRENT,                               // Unexecuted volume
   ORDER_PROP_SL,                                           // StopLoss price
   ORDER_PROP_TP,                                           // TakeProfit price
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

Some properties were additionally placed to the enumerations: parent order ticket, derived order ticket, profit in points, properties of closing by stop loss or take profit, direction and full profit — considering commission and swap. These data are often used in trading strategy logic, so they should be stored in the abstract order fields, as well as updated and received in custom programs right away.

In order to put together all order properties (integer, real and string ones), create macro substitutions containing the number of parameters in each of the three enumerations for each property type.

The numbering of enumerations of integer properties starts from zero, while the numbering of enumerations of other properties starts with the total number of previous properties. Thus, we can always get the index of the property we need as a difference between the number of a requested property and the number of the initial property of this enumeration.

After creating the Defines.mqh file, connect it to the current file of the COrder class and create in the private section the class member variable for storing the order ticket and three arrays for storing integer, real and string order properties:

```
//+------------------------------------------------------------------+
//|                                                        Order.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#include <Object.mqh>
#include "..\Defines.mqh"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class COrder : public CObject
  {
private:
   ulong             m_ticket;                                 // Selected order/deal ticket (MQL5)
   long              m_long_prop[ORDER_PROP_INTEGER_TOTAL];    // Integer properties
   double            m_double_prop[ORDER_PROP_DOUBLE_TOTAL];   // Real properties
   string            m_string_prop[ORDER_PROP_STRING_TOTAL];   // String properties
public:
                     COrder();
                    ~COrder();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
COrder::COrder()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
COrder::~COrder()
  {
  }
//+------------------------------------------------------------------+
```

In order to include the Defines.mqh file, we set a relative path to the file in quotes rather than applying angle brackets (<>) we used when including CObject. This is done so that when transferring the library to any other directory, the connection between the files is not lost and always refers to the location of the Defines.mqh file relative to the current directory.

Now, let's create two methods in the same private section. The first one is to return an exact location of the necessary property in the property arrays, while the second one is to return a protected constructor in the corresponding protected section. Leave the default constructor for creating an empty order object without initializing its properties and remove the destructor (we do not need it, and it will be created automatically during compilation):

```
//+------------------------------------------------------------------+
//|                                                        Order.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#include <Object.mqh>
#include "..\Defines.mqh"
//+------------------------------------------------------------------+
//| Abstract order class                                             |
//+------------------------------------------------------------------+
class COrder : public CObject
  {
private:
   ulong             m_ticket;                                    // Selected order/deal ticket (MQL5)
   long              m_long_prop[ORDER_PROP_INTEGER_TOTAL];       // Integer properties
   double            m_double_prop[ORDER_PROP_DOUBLE_TOTAL];      // Real properties
   string            m_string_prop[ORDER_PROP_STRING_TOTAL];      // String properties

   //--- Return the array index the double property is actually located at
   int               IndexProp(ENUM_ORDER_PROP_DOUBLE property)   const { return (int)property-ORDER_PROP_INTEGER_TOTAL;                  }
   //--- Return the array index the string property is actually located at
   int               IndexProp(ENUM_ORDER_PROP_STRING property)   const { return (int)property-ORDER_PROP_INTEGER_TOTAL-ORDER_PROP_DOUBLE_TOTAL;}
public:
   //--- Default constructor
                     COrder(void){;}
protected:
   //--- Protected parametric constructor
                     COrder(ENUM_ORDER_STATUS order_status,const ulong ticket);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
COrder::COrder(ENUM_ORDER_STATUS order_status,const ulong ticket)
  {
  }
//+------------------------------------------------------------------+
```

So far, the protected constructor does nothing. It will be used to immediately initialize all properties of the order, the ticket of which is passed to the class constructor, when creating an object. Since the order has already been selected, we can obtain the necessary properties of a selected order and write them to the arrays of the object properties. We will do this a bit later after creating methods for receiving and returning order data.

Since this is a cross-platform library, it will be more convenient to make separate methods for obtaining order properties.

In the protected section, add descriptions of methods for receiving integer, real and string properties of a selected order.

```
//+------------------------------------------------------------------+
//|                                                        Order.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#include <Object.mqh>
#include "..\Defines.mqh"
//+------------------------------------------------------------------+
//| Abstract order class                                             |
//+------------------------------------------------------------------+
class COrder : public CObject
  {
private:
   ulong             m_ticket;                                    // Selected order/deal ticket (MQL5)
   long              m_long_prop[ORDER_PROP_INTEGER_TOTAL];       // Integer properties
   double            m_double_prop[ORDER_PROP_DOUBLE_TOTAL];      // Real properties
   string            m_string_prop[ORDER_PROP_STRING_TOTAL];      // String properties

   //--- Return the index of the array the double property is actually located at
   int               IndexProp(ENUM_ORDER_PROP_DOUBLE property)   const { return (int)property-ORDER_PROP_INTEGER_TOTAL;                        }
   //--- Return the index of the array the string property is actually located at
   int               IndexProp(ENUM_ORDER_PROP_STRING property)   const { return (int)property-ORDER_PROP_INTEGER_TOTAL-ORDER_PROP_DOUBLE_TOTAL;}
public:
   //--- Default constructor
                     COrder(void){;}
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

   //--- Get and return real properties of a selected order from its parameters: (1) open price, (2) close price, (3) profit,
   //---  (4) commission, (5) swap, (6) volume, (7) unexecuted volume (8) StopLoss price, (9) TakeProfit price (10) StopLimit order price
   double            OrderOpenPrice(void)          const;
   double            OrderClosePrice(void)         const;
   double            OrderProfit(void)             const;
   double            OrderCommission(void)         const;
   double            OrderSwap(void)               const;
   double            OrderVolume(void)             const;
   double            OrderVolumeCurrent(void)      const;
   double            OrderStopLoss(void)           const;
   double            OrderTakeProfit(void)         const;
   double            OrderPriceStopLimit(void)     const;

   //--- Get and return string properties of a selected order from its parameters: (1) symbol, (2) comment, (3) ID at an exchange
   string            OrderSymbol(void)             const;
   string            OrderComment(void)            const;
   string            OrderExternalID(void)         const;

//---
  };
//+------------------------------------------------------------------+
//| Closed parametric constructor                                    |
//+------------------------------------------------------------------+
COrder::COrder(ENUM_ORDER_STATUS order_status,const ulong ticket)
  {
  }
//+------------------------------------------------------------------+
```

Currently, only the methods for receiving properties have been declared. They are not implemented yet, although the class is compiled without errors. Order property arrays will be filled in the class constructor using the methods we have just added. When all is ready, these arrays will allow us to obtain in the program any property upon request.

The [virtual method of comparing objects](https://www.mql5.com/en/docs/standardlibrary/cobject/cobjectcompare) by specified properties is declared in the CObject class of the standard library. However, the method should be implemented in the subclasses. Therefore, let's add to the abstract order class the method for comparing COrder objects according to any of its properties, as well as several public methods for accessing order properties and virtual methods returning flags for supporting an order object's specific property. These methods are to be implemented in the COrder class descendant objects. This will be required later to select orders from the collection list by any of their properties. By default, if any of these methods is not implemented in the descendant class, the flag indicating the order's support for that property is returned.

```
//+------------------------------------------------------------------+
//|                                                        Order.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#include <Object.mqh>
#include "..\Defines.mqh"
//+------------------------------------------------------------------+
//| Abstract order class                                             |
//+------------------------------------------------------------------+
class COrder : public CObject
  {
private:
   ulong             m_ticket;                                    // Selected order/deal ticket (MQL5)
   long              m_long_prop[ORDER_PROP_INTEGER_TOTAL];       // Integer properties
   double            m_double_prop[ORDER_PROP_DOUBLE_TOTAL];      // Real properties
   string            m_string_prop[ORDER_PROP_STRING_TOTAL];      // String properties

   //--- Return the index of the array the double property is actually located at
   int               IndexProp(ENUM_ORDER_PROP_DOUBLE property)   const { return (int)property-ORDER_PROP_INTEGER_TOTAL;                        }
   //--- Return the index of the array the string property is actually located at
   int               IndexProp(ENUM_ORDER_PROP_STRING property)   const { return (int)property-ORDER_PROP_INTEGER_TOTAL-ORDER_PROP_DOUBLE_TOTAL;}
public:
   //--- Default constructor
                     COrder(void){;}
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

   //--- Get and return real properties of a selected order from its parameters: (1) open price, (2) close price, (3) profit,
   //---  (4) commission, (5) swap, (6) volume, (7) unexecuted volume (8) StopLoss price, (9) TakeProfit price (10) StopLimit order price
   double            OrderOpenPrice(void)          const;
   double            OrderClosePrice(void)         const;
   double            OrderProfit(void)             const;
   double            OrderCommission(void)         const;
   double            OrderSwap(void)               const;
   double            OrderVolume(void)             const;
   double            OrderVolumeCurrent(void)      const;
   double            OrderStopLoss(void)           const;
   double            OrderTakeProfit(void)         const;
   double            OrderPriceStopLimit(void)     const;

   //--- Get and return string properties of a selected order from its parameters: (1) symbol, (2) comment, (3) ID at an exchange
   string            OrderSymbol(void)             const;
   string            OrderComment(void)            const;
   string            OrderExternalID(void)         const;

public:
   //--- Return (1) integer, (2) real and (3) string order properties from the property array
   long              GetProperty(ENUM_ORDER_PROP_INTEGER property)      const { return m_long_prop[property];                    }
   double            GetProperty(ENUM_ORDER_PROP_DOUBLE property)       const { return m_double_prop[this.IndexProp(property)];  }
   string            GetProperty(ENUM_ORDER_PROP_STRING property)       const { return m_string_prop[this.IndexProp(property)];  }

   //--- Return the flag of the order supporting the property
   virtual bool      SupportProperty(ENUM_ORDER_PROP_INTEGER property)        { return true; }
   virtual bool      SupportProperty(ENUM_ORDER_PROP_DOUBLE property)         { return true; }
   virtual bool      SupportProperty(ENUM_ORDER_PROP_STRING property)         { return true; }

   //--- Compare COrder objects by all possible properties
   virtual int       Compare(const CObject *node,const int mode=0) const;

//---
  };
//+------------------------------------------------------------------+
//| Closed parametric constructor                                    |
//+------------------------------------------------------------------+
COrder::COrder(ENUM_ORDER_STATUS order_status,const ulong ticket)
  {
  }
//+------------------------------------------------------------------+
```

**Implementing the method for comparing two orders by a specified property:**

```
//+------------------------------------------------------------------+
//| Compare COrder objects by all possible properties                |
//+------------------------------------------------------------------+
int COrder::Compare(const CObject *node,const int mode=0) const
  {
   const COrder *order_compared=node;
//--- compare integer properties of two orders
   if(mode<ORDER_PROP_INTEGER_TOTAL)
     {
      long value_compared=order_compared.GetProperty((ENUM_ORDER_PROP_INTEGER)mode);
      long value_current=this.GetProperty((ENUM_ORDER_PROP_INTEGER)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare real properties of two orders
   else if(mode<ORDER_PROP_DOUBLE_TOTAL+ORDER_PROP_INTEGER_TOTAL)
     {
      double value_compared=order_compared.GetProperty((ENUM_ORDER_PROP_DOUBLE)mode);
      double value_current=this.GetProperty((ENUM_ORDER_PROP_DOUBLE)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare string properties of two orders
   else if(mode<ORDER_PROP_DOUBLE_TOTAL+ORDER_PROP_INTEGER_TOTAL+ORDER_PROP_STRING_TOTAL)
     {
      string value_compared=order_compared.GetProperty((ENUM_ORDER_PROP_STRING)mode);
      string value_current=this.GetProperty((ENUM_ORDER_PROP_STRING)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
   return 0;
  }
//+------------------------------------------------------------------+
```

The method receives the pointer to the order object whose property should be compared with a certain given value, as well as the value itself from the order properties enumeration.

If the order value exceeds the compared one, the system returns 1, if it is less than the compared value, the system returns -1, otherwise - 0. The list of orders where comparison is performed should be preliminarily sorted by the compared property.

Now, let's implement the methods for receiving order properties and writing them to the property arrays. These are the methods that have been previously declared in the private section. Since the methods for receiving order properties are cross-platform, let's analyze them using receiving the EA magic number as an example:

```
//+------------------------------------------------------------------+
//| Return magic number                                              |
//+------------------------------------------------------------------+
long COrder::OrderMagicNumber() const
  {
#ifdef __MQL4__
   return ::OrderMagicNumber();
#else
   long res=0;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_MARKET_ACTIVE     : res=::PositionGetInteger(POSITION_MAGIC);           break;
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetInteger(ORDER_MAGIC);                 break;
      case ORDER_STATUS_DEAL              : res=::HistoryDealGetInteger(m_ticket,DEAL_MAGIC);   break;
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetInteger(m_ticket,ORDER_MAGIC); break;
      default                             : res=0;                                              break;
     }
   return res;
#endif
  }
//+------------------------------------------------------------------+
```

If this is a code for MQL4, the magic number is returned by the OrderMagicNumber() MQL5 function. Otherwise, check the order status. Depending on what we are dealing with, return either a position, an order or a deal magic number.

The remaining methods of reading and writing the properties of a highlighted order/deal/position are made in the same way. You can analyze them on your own.

**Methods for obtaining the order/deal/position integer properties:**

```
//+------------------------------------------------------------------+
//| Return the ticket                                                |
//+------------------------------------------------------------------+
long COrder::OrderTicket(void) const
  {
#ifdef __MQL4__
   return ::OrderTicket();
#else
   long res=0;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_MARKET_ACTIVE     :
      case ORDER_STATUS_MARKET_PENDING    :
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     :
      case ORDER_STATUS_DEAL              : res=(long)m_ticket;                                 break;
      default                             : res=0;                                              break;
     }
   return res;
#endif
  }
//+------------------------------------------------------------------+
//| Return the parent order ticket                                   |
//+------------------------------------------------------------------+
long COrder::OrderTicketFrom(void) const
  {
   long ticket=0;
#ifdef __MQL4__
   string order_comment=::OrderComment();
   if(::StringFind(order_comment,"from #")>WRONG_VALUE) ticket=::StringToInteger(::StringSubstr(order_comment,6));
#endif
   return ticket;
  }
//+------------------------------------------------------------------+
//| Return the child order ticket                                    |
//+------------------------------------------------------------------+
long COrder::OrderTicketTo(void) const
  {
   long ticket=0;
#ifdef __MQL4__
   string order_comment=::OrderComment();
   if(::StringFind(order_comment,"to #")>WRONG_VALUE) ticket=::StringToInteger(::StringSubstr(order_comment,4));
#endif
   return ticket;
  }
//+------------------------------------------------------------------+
//| Return position ID                                               |
//+------------------------------------------------------------------+
long COrder::OrderPositionID(void) const
  {
#ifdef __MQL4__
   return ::OrderMagicNumber();
#else
   long res=0;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_MARKET_ACTIVE     : res=::PositionGetInteger(POSITION_IDENTIFIER);            break;
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetInteger(ORDER_POSITION_ID);                 break;
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetInteger(m_ticket,ORDER_POSITION_ID); break;
      case ORDER_STATUS_DEAL              : res=::HistoryDealGetInteger(m_ticket,DEAL_POSITION_ID);   break;
      default                             : res=0;                                                    break;
     }
   return res;
#endif
  }
//+------------------------------------------------------------------+
//| Return the opposite position ID                                  |
//+------------------------------------------------------------------+
long COrder::OrderPositionByID(void) const
  {
#ifdef __MQL4__
   return 0;
#else
   long res=0;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetInteger(ORDER_POSITION_BY_ID);                 break;
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetInteger(m_ticket,ORDER_POSITION_BY_ID); break;
      default                             : res=0;                                                       break;
     }
   return res;
#endif
  }
//+------------------------------------------------------------------+
//| Return open time in milliseconds                                 |
//+------------------------------------------------------------------+
long COrder::OrderOpenTimeMSC(void) const
  {
#ifdef __MQL4__
   return (long)::OrderOpenTime();
#else
   long res=0;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_MARKET_ACTIVE     : res=::PositionGetInteger(POSITION_TIME_MSC);                 break;
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetInteger(ORDER_TIME_SETUP_MSC);                 break;
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetInteger(m_ticket,ORDER_TIME_SETUP_MSC); break;
      case ORDER_STATUS_DEAL              : res=::HistoryDealGetInteger(m_ticket,DEAL_TIME_MSC);         break;
      default                             : res=0;                                                       break;
     }
   return res;
#endif
  }
//+------------------------------------------------------------------+
//| Return close time in milliseconds                                |
//+------------------------------------------------------------------+
long COrder::OrderCloseTimeMSC(void) const
  {
#ifdef __MQL4__
   return (long)::OrderCloseTime();
#else
   long res=0;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetInteger(m_ticket,ORDER_TIME_DONE_MSC);     break;
      case ORDER_STATUS_DEAL              : res=(datetime)::HistoryDealGetInteger(m_ticket,DEAL_TIME_MSC);  break;
      default                             : res=0;                                                          break;
     }
   return res;
#endif
  }
//+------------------------------------------------------------------+
//| Return the type                                                  |
//+------------------------------------------------------------------+
long COrder::OrderType(void) const
  {
#ifdef __MQL4__
   return (long)::OrderType();
#else
   long res=0;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_MARKET_ACTIVE     : res=::PositionGetInteger(POSITION_TYPE);            break;
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetInteger(ORDER_TYPE);                  break;
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetInteger(m_ticket,ORDER_TYPE);  break;
      case ORDER_STATUS_DEAL              : res=::HistoryDealGetInteger(m_ticket,DEAL_TYPE);    break;
      default                             : res=0;                                              break;
     }
   return res;
#endif
  }
//+------------------------------------------------------------------+
//| Return the type by direction                                     |
//+------------------------------------------------------------------+
long COrder::OrderTypeByDirection(void) const
  {
   ENUM_ORDER_STATUS status=(ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS);
   if(status==ORDER_STATUS_MARKET_ACTIVE)
     {
      return(this.OrderType()==POSITION_TYPE_BUY ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
     }
   if(status==ORDER_STATUS_MARKET_PENDING || status==ORDER_STATUS_HISTORY_PENDING)
     {
      return
        (
         this.OrderType()==ORDER_TYPE_BUY_LIMIT ||
         this.OrderType()==ORDER_TYPE_BUY_STOP
         #ifdef __MQL5__                        ||
         this.OrderType()==ORDER_TYPE_BUY_STOP_LIMIT
         #endif ?
         ORDER_TYPE_BUY :
         ORDER_TYPE_SELL
        );
     }
   if(status==ORDER_STATUS_HISTORY_ORDER)
     {
      return this.OrderType();
     }
   return WRONG_VALUE;
  }
//+------------------------------------------------------------------+
//| Return execution type by residue                                 |
//+------------------------------------------------------------------+
long COrder::OrderTypeFilling(void) const
  {
#ifdef __MQL4__
   return (long)ORDER_FILLING_RETURN;
#else
   long res=0;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetInteger(ORDER_TYPE_FILLING);                break;
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetInteger(m_ticket,ORDER_TYPE_FILLING);break;
      default                             : res=0;                                                    break;
     }
   return res;
#endif
  }
//+------------------------------------------------------------------+
//| Return order lifetime                                            |
//+------------------------------------------------------------------+
long COrder::OrderTypeTime(void) const
  {
#ifdef __MQL4__
   return (long)ORDER_TIME_GTC;
#else
   long res=0;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetInteger(ORDER_TYPE_TIME);                break;
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetInteger(m_ticket,ORDER_TYPE_TIME);break;
      default                             : res=0;                                                 break;
     }
   return res;
#endif
  }
//+------------------------------------------------------------------+
//| Order reason or source                                           |
//+------------------------------------------------------------------+
long COrder::OrderReason(void) const
  {
#ifdef __MQL4__
   return
     (
      this.OrderCloseByStopLoss()   ?  ORDER_REASON_SL      :
      this.OrderCloseByTakeProfit() ?  ORDER_REASON_TP      :
      this.OrderMagicNumber()!=0    ?  ORDER_REASON_EXPERT  : WRONG_VALUE
     );
#else
   long res=WRONG_VALUE;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_MARKET_ACTIVE     : res=::PositionGetInteger(POSITION_REASON);          break;
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetInteger(ORDER_REASON);                break;
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetInteger(m_ticket,ORDER_REASON);break;
      case ORDER_STATUS_DEAL              : res=::HistoryDealGetInteger(m_ticket,DEAL_REASON);  break;
      default                             : res=WRONG_VALUE;                                    break;
     }
   return res;
#endif
  }
//+------------------------------------------------------------------+
//| Order, based on which a deal is performed                        |
//+------------------------------------------------------------------+
long COrder::DealOrder(void) const
  {
#ifdef __MQL4__
   return ::OrderTicket();
#else
   long res=0;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_MARKET_ACTIVE     : res=::PositionGetInteger(POSITION_IDENTIFIER);            break;
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetInteger(ORDER_POSITION_ID);                 break;
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetInteger(m_ticket,ORDER_POSITION_ID); break;
      case ORDER_STATUS_DEAL              : res=::HistoryDealGetInteger(m_ticket,DEAL_ORDER);         break;
      default                             : res=0;                                                    break;
     }
   return res;
#endif
  }
//+------------------------------------------------------------------+
//| Deal direction IN, OUT, IN/OUT                                   |
//+------------------------------------------------------------------+
long COrder::DealEntry(void) const
  {
#ifdef __MQL4__
   return ::OrderType();
#else
   long res=WRONG_VALUE;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_DEAL  : res=::HistoryDealGetInteger(m_ticket,DEAL_ENTRY);break;
      default                 : res=WRONG_VALUE;                                 break;
     }
   return res;
#endif
  }
//+------------------------------------------------------------------+
//| Return the flag of closing a position by StopLoss                |
//+------------------------------------------------------------------+
bool COrder::OrderCloseByStopLoss(void) const
  {
#ifdef __MQL4__
   return(::StringFind(::OrderComment(),"[sl")>WRONG_VALUE);\
#else\
   return\
     (\
      this.Status()==ORDER_STATUS_HISTORY_ORDER ? this.OrderReason()==ORDER_REASON_SL :\
      this.Status()==ORDER_STATUS_DEAL ? this.OrderReason()==DEAL_REASON_SL : false\
     );\
#endif\
  }\
//+------------------------------------------------------------------+\
//| Return the flag of closing position by TakeProfit                |\
//+------------------------------------------------------------------+\
bool COrder::OrderCloseByTakeProfit(void) const\
  {\
#ifdef __MQL4__\
   return(::StringFind(::OrderComment(),"[tp")>WRONG_VALUE);\
#else\
   return\
     (\
      this.Status()==ORDER_STATUS_HISTORY_ORDER ? this.OrderReason()==ORDER_REASON_TP :\
      this.Status()==ORDER_STATUS_DEAL ? this.OrderReason()==DEAL_REASON_TP : false\
     );\
#endif\
  }\
//+------------------------------------------------------------------+\
//| Return open time                                                 |\
//+------------------------------------------------------------------+\
datetime COrder::OrderOpenTime(void) const\
  {\
#ifdef __MQL4__\
   return ::OrderOpenTime();\
#else\
   datetime res=0;\
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))\
     {\
      case ORDER_STATUS_MARKET_ACTIVE     : res=(datetime)::PositionGetInteger(POSITION_TIME);                 break;\
      case ORDER_STATUS_MARKET_PENDING    : res=(datetime)::OrderGetInteger(ORDER_TIME_SETUP);                 break;\
      case ORDER_STATUS_HISTORY_PENDING   :\
      case ORDER_STATUS_HISTORY_ORDER     : res=(datetime)::HistoryOrderGetInteger(m_ticket,ORDER_TIME_SETUP); break;\
      case ORDER_STATUS_DEAL              : res=(datetime)::HistoryDealGetInteger(m_ticket,DEAL_TIME);         break;\
      default                             : res=0;                                                             break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
//| Return close time                                                |\
//+------------------------------------------------------------------+\
datetime COrder::OrderCloseTime(void) const\
  {\
#ifdef __MQL4__\
   return ::OrderCloseTime();\
#else\
   datetime res=0;\
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))\
     {\
      case ORDER_STATUS_HISTORY_PENDING   :\
      case ORDER_STATUS_HISTORY_ORDER     : res=(datetime)::HistoryOrderGetInteger(m_ticket,ORDER_TIME_DONE);  break;\
      case ORDER_STATUS_DEAL              : res=(datetime)::HistoryDealGetInteger(m_ticket,DEAL_TIME);         break;\
      default                             : res=0;                                                             break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
//| Return expiration time                                           |\
//+------------------------------------------------------------------+\
datetime COrder::OrderExpiration(void) const\
  {\
#ifdef __MQL4__\
   return ::OrderExpiration();\
#else\
   datetime res=0;\
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))\
     {\
      case ORDER_STATUS_MARKET_PENDING    : res=(datetime)::OrderGetInteger(ORDER_TIME_EXPIRATION);                  break;\
      case ORDER_STATUS_HISTORY_PENDING   : res=(datetime)::HistoryOrderGetInteger(m_ticket,ORDER_TIME_EXPIRATION);  break;\
      default                             : res=0;                                                                   break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
//| Position change time in seconds                                  |\
//+------------------------------------------------------------------+\
datetime COrder::PositionTimeUpdate(void) const\
  {\
#ifdef __MQL4__\
   return 0;\
#else\
   datetime res=0;\
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))\
     {\
      case ORDER_STATUS_MARKET_ACTIVE  : res=(datetime)::PositionGetInteger(POSITION_TIME_UPDATE); break;\
      default                          : res=0;                                                    break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
//| Position change time in milliseconds                             |\
//+------------------------------------------------------------------+\
datetime COrder::PositionTimeUpdateMSC(void) const\
  {\
#ifdef __MQL4__\
   return 0;\
#else\
   datetime res=0;\
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))\
     {\
      case ORDER_STATUS_MARKET_ACTIVE  : res=(datetime)::PositionGetInteger(POSITION_TIME_UPDATE_MSC);break;\
      default                          : res=0;                                                       break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
```\
\
**Methods for obtaining the order/deal/position real properties:**\
\
```\
//+------------------------------------------------------------------+\
//| Return open price                                                |\
//+------------------------------------------------------------------+\
double COrder::OrderOpenPrice(void) const\
  {\
#ifdef __MQL4__\
   return ::OrderOpenPrice();\
#else\
   double res=0;\
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))\
     {\
      case ORDER_STATUS_MARKET_ACTIVE     : res=::PositionGetDouble(POSITION_PRICE_OPEN);          break;\
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetDouble(ORDER_PRICE_OPEN);                break;\
      case ORDER_STATUS_HISTORY_PENDING   :\
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetDouble(m_ticket,ORDER_PRICE_OPEN);break;\
      case ORDER_STATUS_DEAL              : res=::HistoryDealGetDouble(m_ticket,DEAL_PRICE);       break;\
      default                             : res=0;                                                 break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
//| Return close price                                               |\
//+------------------------------------------------------------------+\
double COrder::OrderClosePrice(void) const\
  {\
#ifdef __MQL4__\
   return ::OrderClosePrice();\
#else\
   double res=0;\
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))\
     {\
      case ORDER_STATUS_HISTORY_PENDING   :\
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetDouble(m_ticket,ORDER_PRICE_OPEN);break;\
      case ORDER_STATUS_DEAL              : res=::HistoryDealGetDouble(m_ticket,DEAL_PRICE);       break;\
      default                             : res=0;                                                 break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
//| Return profit                                                    |\
//+------------------------------------------------------------------+\
double COrder::OrderProfit(void) const\
  {\
#ifdef __MQL4__\
   return ::OrderProfit();\
#else\
   double res=0;\
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))\
     {\
      case ORDER_STATUS_MARKET_ACTIVE  : res=::PositionGetDouble(POSITION_PROFIT);        break;\
      case ORDER_STATUS_DEAL           : res=::HistoryDealGetDouble(m_ticket,DEAL_PROFIT);break;\
      default                          : res=0;                                           break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
//| Return commission                                                |\
//+------------------------------------------------------------------+\
double COrder::OrderCommission(void) const\
  {\
#ifdef __MQL4__\
   return ::OrderCommission();\
#else\
   double res=0;\
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))\
     {\
      case ORDER_STATUS_DEAL  : res=::HistoryDealGetDouble(m_ticket,DEAL_COMMISSION);  break;\
      default                 : res=0;                                                 break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
//| Return swap                                                      |\
//+------------------------------------------------------------------+\
double COrder::OrderSwap(void) const\
  {\
#ifdef __MQL4__\
   return ::OrderSwap();\
#else\
   double res=0;\
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))\
     {\
      case ORDER_STATUS_MARKET_ACTIVE  : res=::PositionGetDouble(POSITION_SWAP);          break;\
      case ORDER_STATUS_DEAL           : res=::HistoryDealGetDouble(m_ticket,DEAL_SWAP);  break;\
      default                          : res=0;                                           break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
//| Return volume                                                    |\
//+------------------------------------------------------------------+\
double COrder::OrderVolume(void) const\
  {\
#ifdef __MQL4__\
   return ::OrderLots();\
#else\
   double res=0;\
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))\
     {\
      case ORDER_STATUS_MARKET_ACTIVE     : res=::PositionGetDouble(POSITION_VOLUME);                    break;\
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetDouble(ORDER_VOLUME_INITIAL);                  break;\
      case ORDER_STATUS_HISTORY_PENDING   :\
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetDouble(m_ticket,ORDER_VOLUME_INITIAL);  break;\
      case ORDER_STATUS_DEAL              : res=::HistoryDealGetDouble(m_ticket,DEAL_VOLUME);            break;\
      default                             : res=0;                                                       break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
//| Return unexecuted volume                                         |\
//+------------------------------------------------------------------+\
double COrder::OrderVolumeCurrent(void) const\
  {\
#ifdef __MQL4__\
   return ::OrderLots();\
#else\
   double res=0;\
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))\
     {\
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetDouble(ORDER_VOLUME_CURRENT);                  break;\
      case ORDER_STATUS_HISTORY_PENDING   :\
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetDouble(m_ticket,ORDER_VOLUME_CURRENT);  break;\
      default                             : res=0;                                                       break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
//| Return StopLoss price                                            |\
//+------------------------------------------------------------------+\
double COrder::OrderStopLoss(void) const\
  {\
#ifdef __MQL4__\
   return ::OrderStopLoss();\
#else\
   double res=0;\
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))\
     {\
      case ORDER_STATUS_MARKET_ACTIVE     : res=::PositionGetDouble(POSITION_SL);            break;\
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetDouble(ORDER_SL);                  break;\
      case ORDER_STATUS_HISTORY_PENDING   :\
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetDouble(m_ticket,ORDER_SL);  break;\
      default                             : res=0;                                           break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
//| Return TakeProfit price                                          |\
//+------------------------------------------------------------------+\
double COrder::OrderTakeProfit(void) const\
  {\
#ifdef __MQL4__\
   return ::OrderTakeProfit();\
#else\
   double res=0;\
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))\
     {\
      case ORDER_STATUS_MARKET_ACTIVE     : res=::PositionGetDouble(POSITION_TP);            break;\
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetDouble(ORDER_TP);                  break;\
      case ORDER_STATUS_HISTORY_PENDING   :\
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetDouble(m_ticket,ORDER_TP);  break;\
      default                             : res=0;                                           break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
//| Return Limit order price                                         |\
//| when StopLimit order is activated                                |\
//+------------------------------------------------------------------+\
double COrder::OrderPriceStopLimit(void) const\
  {\
#ifdef __MQL4__\
   return 0;\
#else\
   double res=0;\
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))\
     {\
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetDouble(ORDER_PRICE_STOPLIMIT);                 break;\
      case ORDER_STATUS_HISTORY_PENDING   :\
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetDouble(m_ticket,ORDER_PRICE_STOPLIMIT); break;\
      default                             : res=0;                                                       break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
```\
\
**Methods for obtaining the order/deal/position string properties:**\
\
```\
//+------------------------------------------------------------------+\
//| Return symbol                                                    |\
//+------------------------------------------------------------------+\
string COrder::OrderSymbol(void) const\
  {\
#ifdef __MQL4__\
   return ::OrderSymbol();\
#else\
   string res="";\
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))\
     {\
      case ORDER_STATUS_MARKET_ACTIVE     : res=::PositionGetString(POSITION_SYMBOL);           break;\
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetString(ORDER_SYMBOL);                 break;\
      case ORDER_STATUS_HISTORY_PENDING   :\
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetString(m_ticket,ORDER_SYMBOL); break;\
      case ORDER_STATUS_DEAL              : res=::HistoryDealGetString(m_ticket,DEAL_SYMBOL);   break;\
      default                             : res="";                                             break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
//| Return comment                                                   |\
//+------------------------------------------------------------------+\
string COrder::OrderComment(void) const\
  {\
#ifdef __MQL4__\
   return ::OrderComment();\
#else\
   string res="";\
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))\
     {\
      case ORDER_STATUS_MARKET_ACTIVE     : res=::PositionGetString(POSITION_COMMENT);          break;\
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetString(ORDER_COMMENT);                break;\
      case ORDER_STATUS_HISTORY_PENDING   :\
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetString(m_ticket,ORDER_COMMENT);break;\
      case ORDER_STATUS_DEAL              : res=::HistoryDealGetString(m_ticket,DEAL_COMMENT);  break;\
      default                             : res="";                                             break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
//| Return ID used by an exchange                                    |\
//+------------------------------------------------------------------+\
string COrder::OrderExternalID(void) const\
  {\
#ifdef __MQL4__\
   return "";\
#else\
   string res="";\
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))\
     {\
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetString(ORDER_EXTERNAL_ID);                  break;\
      case ORDER_STATUS_HISTORY_PENDING   :\
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetString(m_ticket,ORDER_EXTERNAL_ID);  break;\
      case ORDER_STATUS_DEAL              : res=::HistoryDealGetString(m_ticket,DEAL_EXTERNAL_ID);    break;\
      default                             : res="";                                                   break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
```\
\
We have declared and implemented the private methods for receiving properties from the order data.\
\
Now it is time to implement the protected class constructor by writing to it methods that save all properties of the order whose ticket has been passed to the constructor.\
\
**Implementing the protected class constructor:**\
\
```\
//+------------------------------------------------------------------+\
//| Closed parametric constructor                                    |\
//+------------------------------------------------------------------+\
COrder::COrder(ENUM_ORDER_STATUS order_status,const ulong ticket)\
  {\
//--- Save integer properties\
   m_ticket=ticket;\
   m_long_prop[ORDER_PROP_STATUS]                              = order_status;\
   m_long_prop[ORDER_PROP_MAGIC]                               = this.OrderMagicNumber();\
   m_long_prop[ORDER_PROP_TICKET]                              = this.OrderTicket();\
   m_long_prop[ORDER_PROP_TIME_OPEN]                           = (long)(ulong)this.OrderOpenTime();\
   m_long_prop[ORDER_PROP_TIME_CLOSE]                          = (long)(ulong)this.OrderCloseTime();\
   m_long_prop[ORDER_PROP_TIME_EXP]                            = (long)(ulong)this.OrderExpiration();\
   m_long_prop[ORDER_PROP_TYPE]                                = this.OrderType();\
   m_long_prop[ORDER_PROP_DIRECTION]                           = this.OrderTypeByDirection();\
   m_long_prop[ORDER_PROP_POSITION_ID]                         = this.OrderPositionID();\
   m_long_prop[ORDER_PROP_REASON]                              = this.OrderReason();\
   m_long_prop[ORDER_PROP_DEAL_ORDER]                          = this.DealOrder();\
   m_long_prop[ORDER_PROP_DEAL_ENTRY]                          = this.DealEntry();\
   m_long_prop[ORDER_PROP_POSITION_BY_ID]                      = this.OrderPositionByID();\
   m_long_prop[ORDER_PROP_TIME_OPEN_MSC]                       = this.OrderOpenTimeMSC();\
   m_long_prop[ORDER_PROP_TIME_CLOSE_MSC]                      = this.OrderCloseTimeMSC();\
   m_long_prop[ORDER_PROP_TIME_UPDATE]                         = (long)(ulong)this.PositionTimeUpdate();\
   m_long_prop[ORDER_PROP_TIME_UPDATE_MSC]                     = (long)(ulong)this.PositionTimeUpdateMSC();\
\
//--- Save real properties\
   m_double_prop[this.IndexProp(ORDER_PROP_PRICE_OPEN)]        = this.OrderOpenPrice();\
   m_double_prop[this.IndexProp(ORDER_PROP_PRICE_CLOSE)]       = this.OrderClosePrice();\
   m_double_prop[this.IndexProp(ORDER_PROP_PROFIT)]            = this.OrderProfit();\
   m_double_prop[this.IndexProp(ORDER_PROP_COMMISSION)]        = this.OrderCommission();\
   m_double_prop[this.IndexProp(ORDER_PROP_SWAP)]              = this.OrderSwap();\
   m_double_prop[this.IndexProp(ORDER_PROP_VOLUME)]            = this.OrderVolume();\
   m_double_prop[this.IndexProp(ORDER_PROP_SL)]                = this.OrderStopLoss();\
   m_double_prop[this.IndexProp(ORDER_PROP_TP)]                = this.OrderTakeProfit();\
   m_double_prop[this.IndexProp(ORDER_PROP_VOLUME_CURRENT)]    = this.OrderVolumeCurrent();\
   m_double_prop[this.IndexProp(ORDER_PROP_PRICE_STOP_LIMIT)]  = this.OrderPriceStopLimit();\
\
//--- Save string properties\
   m_string_prop[this.IndexProp(ORDER_PROP_SYMBOL)]            = this.OrderSymbol();\
   m_string_prop[this.IndexProp(ORDER_PROP_COMMENT)]           = this.OrderComment();\
   m_string_prop[this.IndexProp(ORDER_PROP_EXT_ID)]            = this.OrderExternalID();\
\
//--- Save additional integer properties\
   m_long_prop[ORDER_PROP_PROFIT_PT]                           = this.ProfitInPoints();\
   m_long_prop[ORDER_PROP_TICKET_FROM]                         = this.OrderTicketFrom();\
   m_long_prop[ORDER_PROP_TICKET_TO]                           = this.OrderTicketTo();\
   m_long_prop[ORDER_PROP_CLOSE_BY_SL]                         = this.OrderCloseByStopLoss();\
   m_long_prop[ORDER_PROP_CLOSE_BY_TP]                         = this.OrderCloseByTakeProfit();\
\
//--- Save additional real properties\
   m_double_prop[this.IndexProp(ORDER_PROP_PROFIT_FULL)]       = this.ProfitFull();\
  }\
//+------------------------------------------------------------------+\
```\
\
When we pass along history or market orders/deals/positions in a loop or select a new order/deal/position, a new object derived from the COrder class is created. The class constructor featuring the order status and ticket is called in the class constructor receiving only the order ticket. Next, order property arrays are simply filled in the COrder class constructor using the methods described above.\
\
Thus, each new order/deal/position is to have its unique property. All of them are to be stored in the lists, while working with such lists is the main activity of the library. The lists can be sorted by any of the order/deal/position properties. Besides, it is possible to generate new lists out of the selected ones.\
\
\
At this stage, the basic functionality of the COrder abstract order class has been implemented. This is a basic class for storing various types of orders, deals and positions. All others will be derived from it for creating objects divided by types of orders, deals and positions.\
\
The library is created for simplified access to data and easy program development.\
\
In the current state, we have three public methods for accessing the properties of the GetProperty(XXX) abstract order. You can use them, but this is not very convenient since you need to remember the member names of the enumerations describing a specific order property. Therefore, we will add several more public methods to obtain the necessary data. These methods will have reasonable names so that it is immediately clear what property can be obtained by using a specific method.\
\
These methods can be used in custom programs to read the properties of a selected order, deal or position:\
\
\
```\
   //--- Return (1) ticket, (2) parent order ticket, (3) derived order ticket, (4) magic number, (5) order reason (6) position ID\
   //--- (7) opposite position ID, (8) flag of closing by StopLoss, (9) flag of closing by TakeProfit (10) open time, (11) close time,\
   //--- (12) open time in milliseconds, (13) close time in milliseconds (14) expiration date, (15) type, (16) status, (17) direction\
   long              Ticket(void)                                       const { return this.GetProperty(ORDER_PROP_TICKET);                     }\
   long              TicketFrom(void)                                   const { return this.GetProperty(ORDER_PROP_TICKET_FROM);                }\
   long              TicketTo(void)                                     const { return this.GetProperty(ORDER_PROP_TICKET_TO);                  }\
   long              Magic(void)                                        const { return this.GetProperty(ORDER_PROP_MAGIC);                      }\
   long              Reason(void)                                       const { return this.GetProperty(ORDER_PROP_REASON);                     }\
   long              PositionID(void)                                   const { return this.GetProperty(ORDER_PROP_POSITION_ID);                }\
   long              PositionByID(void)                                 const { return this.GetProperty(ORDER_PROP_POSITION_BY_ID);             }\
   bool              IsCloseByStopLoss(void)                            const { return (bool)this.GetProperty(ORDER_PROP_CLOSE_BY_SL);          }\
   bool              IsCloseByTakeProfit(void)                          const { return (bool)this.GetProperty(ORDER_PROP_CLOSE_BY_TP);          }\
   datetime          TimeOpen(void)                                     const { return (datetime)this.GetProperty(ORDER_PROP_TIME_OPEN);        }\
   datetime          TimeClose(void)                                    const { return (datetime)this.GetProperty(ORDER_PROP_TIME_CLOSE);       }\
   datetime          TimeOpenMSC(void)                                  const { return (datetime)this.GetProperty(ORDER_PROP_TIME_OPEN_MSC);    }\
   datetime          TimeCloseMSC(void)                                 const { return (datetime)this.GetProperty(ORDER_PROP_TIME_CLOSE_MSC);   }\
   datetime          TimeExpiration(void)                               const { return (datetime)this.GetProperty(ORDER_PROP_TIME_EXP);         }\
   ENUM_ORDER_TYPE   TypeOrder(void)                                    const { return (ENUM_ORDER_TYPE)this.GetProperty(ORDER_PROP_TYPE);      }\
   ENUM_ORDER_STATUS Status(void)                                       const { return (ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS);  }\
   ENUM_ORDER_TYPE   TypeByDirection(void)                              const { return (ENUM_ORDER_TYPE)this.GetProperty(ORDER_PROP_DIRECTION); }\
\
   //--- Return (1) open price, (2) close price, (3) profit, (4) commission, (5) swap, (6) volume,\
   //--- (7) unexecuted volume (8) StopLoss and (9) TakeProfit (10) StopLimit order price\
   double            PriceOpen(void)                                    const { return this.GetProperty(ORDER_PROP_PRICE_OPEN);                 }\
   double            PriceClose(void)                                   const { return this.GetProperty(ORDER_PROP_PRICE_CLOSE);                }\
   double            Profit(void)                                       const { return this.GetProperty(ORDER_PROP_PROFIT);                     }\
   double            Comission(void)                                    const { return this.GetProperty(ORDER_PROP_COMMISSION);                 }\
   double            Swap(void)                                         const { return this.GetProperty(ORDER_PROP_SWAP);                       }\
   double            Volume(void)                                       const { return this.GetProperty(ORDER_PROP_VOLUME);                     }\
   double            VolumeCurrent(void)                                const { return this.GetProperty(ORDER_PROP_VOLUME_CURRENT);             }\
   double            StopLoss(void)                                     const { return this.GetProperty(ORDER_PROP_SL);                         }\
   double            TakeProfit(void)                                   const { return this.GetProperty(ORDER_PROP_TP);                         }\
   double            PriceStopLimit(void)                               const { return this.GetProperty(ORDER_PROP_PRICE_STOP_LIMIT);           }\
\
   //--- Return (1) symbol, (2) comment, (3) ID at an exchange\
   string            Symbol(void)                                       const { return this.GetProperty(ORDER_PROP_SYMBOL);                     }\
   string            Comment(void)                                      const { return this.GetProperty(ORDER_PROP_COMMENT);                    }\
   string            ExternalID(void)                                   const { return this.GetProperty(ORDER_PROP_EXT_ID);                     }\
\
   //--- Get the full order profit\
   double            ProfitFull(void)                                   const { return this.Profit()+this.Comission()+this.Swap();              }\
   //--- Get order profit in points\
   int               ProfitInPoints(void) const;\
```\
\
**Implementing profit receiving method in points:**\
\
```\
//+------------------------------------------------------------------+\
//| Return order profit in points                                    |\
//+------------------------------------------------------------------+\
int COrder::ProfitInPoints(void) const\
  {\
   ENUM_ORDER_TYPE type=this.TypeOrder();\
   string symbol=this.Symbol();\
   double point=::SymbolInfoDouble(symbol,SYMBOL_POINT);\
   if(type>ORDER_TYPE_SELL || point==0) return 0;\
   if(this.Status()==ORDER_STATUS_HISTORY_ORDER)\
      return int(type==ORDER_TYPE_BUY ? (this.PriceClose()-this.PriceOpen())/point : type==ORDER_TYPE_SELL ? (this.PriceOpen()-this.PriceClose())/point : 0);\
   else if(this.Status()==ORDER_STATUS_MARKET_ACTIVE)\
     {\
      if(type==ORDER_TYPE_BUY)\
         return int((::SymbolInfoDouble(symbol,SYMBOL_BID)-this.PriceOpen())/point);\
      else if(type==ORDER_TYPE_SELL)\
         return int((this.PriceOpen()-::SymbolInfoDouble(symbol,SYMBOL_ASK))/point);\
     }\
   return 0;\
  }\
//+------------------------------------------------------------------+\
```\
\
And let's add public methods to describe some properties of the order object so that you can display them on demand in a convenient way:\
\
```\
   //--- Get description of an order's (1) integer, (2) real and (3) string property\
   string            GetPropertyDescription(ENUM_ORDER_PROP_INTEGER property);\
   string            GetPropertyDescription(ENUM_ORDER_PROP_DOUBLE property);\
   string            GetPropertyDescription(ENUM_ORDER_PROP_STRING property);\
   //--- Return order status name\
   string            StatusDescription(void) const;\
   //--- Return order or position name\
   string            TypeDescription(void) const;\
   //--- Return the deal direction name\
   string            DealEntryDescription(void) const;\
   //--- Return order/position direction\
   string            DirectionDescription(void) const;\
   //--- Send description of order properties to journal (full_prop=true - all properties, false - only supported ones)\
   void              Print(const bool full_prop=false);\
```\
\
Before we implement these methods, let's solve yet another issue: the library and the programs based on it require various service functions. For example, for this case, we need the function of displaying time with milliseconds and the function of receiving messages in one of two languages. The language of messages is to depend on the terminal language.\
\
Create a new include file in the library's root directory\
\
![](https://c.mql5.com/2/35/CreateDELib_en.png)\
\
and call it DELib. This will be the library file of service functions available for use by both the library classes themselves and the library-based programs.\
\
![](https://c.mql5.com/2/35/NewMQH_en.png)\
\
Click Finish to create a template file:\
\
```\
//+------------------------------------------------------------------+\
//|                                                        DELib.mqh |\
//|                        Copyright 2018, MetaQuotes Software Corp. |\
//|                             https://mql5.com/en/users/artmedia70 |\
//+------------------------------------------------------------------+\
#property copyright "Copyright 2018, MetaQuotes Software Corp."\
#property link      "https://mql5.com/en/users/artmedia70"\
//+------------------------------------------------------------------+\
//| defines                                                          |\
//+------------------------------------------------------------------+\
// #define MacrosHello   "Hello, world!"\
// #define MacrosYear    2010\
//+------------------------------------------------------------------+\
//| DLL imports                                                      |\
//+------------------------------------------------------------------+\
// #import "user32.dll"\
//   int      SendMessageA(int hWnd,int Msg,int wParam,int lParam);\
// #import "my_expert.dll"\
//   int      ExpertRecalculate(int wParam,int lParam);\
// #import\
//+------------------------------------------------------------------+\
//| EX5 imports                                                      |\
//+------------------------------------------------------------------+\
// #import "stdlib.ex5"\
//   string ErrorDescription(int error_code);\
// #import\
//+------------------------------------------------------------------+\
```\
\
Include the Defines.mqh file to it and modify the template according to our needs:\
\
```\
//+------------------------------------------------------------------+\
//|                                                        DELib.mqh |\
//|                        Copyright 2018, MetaQuotes Software Corp. |\
//|                             https://mql5.com/en/users/artmedia70 |\
//+------------------------------------------------------------------+\
#property copyright "Copyright 2018, MetaQuotes Software Corp."\
#property link      "https://mql5.com/en/users/artmedia70"\
#property strict  // Necessary for mql4\
//+------------------------------------------------------------------+\
//| Include files                                                    |\
//+------------------------------------------------------------------+\
#include "Defines.mqh"\
//+------------------------------------------------------------------+\
//| Service functions                                                |\
//+------------------------------------------------------------------+\
\
//+------------------------------------------------------------------+\
```\
\
Since we have included Defines.mqh into this file, we can include this new file (rather than Defines.mqh) into the COrder class file, so that both of them are available in the library. Besides, we will do it in one string, not two.\
\
Replace the include directive in the Order.mqh file:\
\
```\
//+------------------------------------------------------------------+\
//|                                                        Order.mqh |\
//|                        Copyright 2018, MetaQuotes Software Corp. |\
//|                             https://mql5.com/en/users/artmedia70 |\
//+------------------------------------------------------------------+\
#property copyright "Copyright 2018, MetaQuotes Software Corp."\
#property link      "https://mql5.com/en/users/artmedia70"\
#property version   "1.00"\
#include <Object.mqh>\
#include "..\DELib.mqh"\
//+------------------------------------------------------------------+\
//| Abstract order class                                             |\
//+------------------------------------------------------------------+\
```\
\
Let's also add defining the user's country language as a macro substitution in the Defines.mqh file:\
\
```\
//+------------------------------------------------------------------+\
//|                                                      Defines.mqh |\
//|                        Copyright 2018, MetaQuotes Software Corp. |\
//|                             https://mql5.com/en/users/artmedia70 |\
//+------------------------------------------------------------------+\
#property copyright "Copyright 2018, MetaQuotes Software Corp."\
#property link      "https://mql5.com/en/users/artmedia70"\
//+------------------------------------------------------------------+\
//| Macro substitutions                                              |\
//+------------------------------------------------------------------+\
#define COUNTRY_LANG    "Russian"\
//+------------------------------------------------------------------+\
```\
\
Thus, the user will be able to set their native language for displayed messages if the terminal language is not English. However, to achieve this, we will have to replace all future messages in Russian we are to enter here with the ones in language required by the user.\
\
Add the function for returning messages in one of two languages in the DELib.mqh file:\
\
```\
//+------------------------------------------------------------------+\
//|                                                        DELib.mqh |\
//|                        Copyright 2018, MetaQuotes Software Corp. |\
//|                             https://mql5.com/en/users/artmedia70 |\
//+------------------------------------------------------------------+\
#property copyright "Copyright 2018, MetaQuotes Software Corp."\
#property link      "https://mql5.com/en/users/artmedia70"\
#property strict  // Necessary for mql4\
//+------------------------------------------------------------------+\
//| Include files                                                    |\
//+------------------------------------------------------------------+\
#include "Defines.mqh"\
//+------------------------------------------------------------------+\
//| Service functions                                                |\
//+------------------------------------------------------------------+\
//+------------------------------------------------------------------+\
//| Return the text in one of two languages                          |\
//+------------------------------------------------------------------+\
string TextByLanguage(const string text_country_lang,const string text_en)\
  {\
   return(TerminalInfoString(TERMINAL_LANGUAGE)==COUNTRY_LANG ? text_country_lang : text_en);\
  }\
//+------------------------------------------------------------------+\
```\
\
The function checks the terminal language and if it matches the languages specified in the COUNTRY\_LANG macro substitution, the text passed in the first function parameter is displayed. Otherwise, the text contained in the second function parameter (English) is shown.\
\
Let's also add the function of displaying time with milliseconds:\
\
```\
//+------------------------------------------------------------------+\
//|                                                        DELib.mqh |\
//|                        Copyright 2018, MetaQuotes Software Corp. |\
//|                             https://mql5.com/en/users/artmedia70 |\
//+------------------------------------------------------------------+\
#property copyright "Copyright 2018, MetaQuotes Software Corp."\
#property link      "https://mql5.com/en/users/artmedia70"\
#property strict  // Necessary for mql4\
//+------------------------------------------------------------------+\
//| Include file                                                     |\
//+------------------------------------------------------------------+\
#include "Defines.mqh"\
//+------------------------------------------------------------------+\
//| Service functions                                                |\
//+------------------------------------------------------------------+\
//+------------------------------------------------------------------+\
//| Return text in one of the two languages                          |\
//+------------------------------------------------------------------+\
string TextByLanguage(const string text_country_lang,const string text_en)\
  {\
   return(TerminalInfoString(TERMINAL_LANGUAGE)==COUNTRY_LANG ? text_country_lang : text_en);\
  }\
//+------------------------------------------------------------------+\
//| Return time with milliseconds                                    |\
//+------------------------------------------------------------------+\
string TimeMSCtoString(const long time_msc)\
  {\
   return TimeToString(time_msc/1000,TIME_DATE|TIME_MINUTES|TIME_SECONDS)+"."+IntegerToString(time_msc%1000,3,'0');\
  }\
//+------------------------------------------------------------------+\
```\
\
Here, all is simple: time set in milliseconds is passed to the function. To calculate time in seconds, we need to divide the value passed to the function to 1000. To calculate milliseconds, take the residue of this division. All received values are formatted as strings and returned to the calling program.\
\
Sometimes, you may want to get the number of decimal places in a symbol lot. Let's enter this function to our file of service functions:\
\
```\
//+------------------------------------------------------------------+\
//| Return the number of decimal places in a symbol lot              |\
//+------------------------------------------------------------------+\
uint DigitsLots(const string symbol_name)\
  {\
   return (int)ceil(fabs(log(SymbolInfoDouble(symbol_name,SYMBOL_VOLUME_STEP))/log(10)));\
  }\
//+------------------------------------------------------------------+\
```\
\
In addition to service functions, we will need three methods returning order reason, direction and deal type for displaying messages in the journal. Add the three methods to the protected section of the COrder class:\
\
```\
   //--- Return (1) reason, (2) direction, (3) deal type\
   string            GetReasonDescription(const long reason)            const;\
   string            GetEntryDescription(const long deal_entry)         const;\
   string            GetTypeDealDescription(const long type_deal)       const;\
```\
\
And their implementation:\
\
```\
//+------------------------------------------------------------------+\
//| Reason                                                           |\
//+------------------------------------------------------------------+\
string COrder::GetReasonDescription(const long reason) const\
  {\
#ifdef __MQL4__\
   return\
     (\
      this.IsCloseByStopLoss()            ?  TextByLanguage("Срабатывание StopLoss","Due to StopLoss")                  :\
      this.IsCloseByTakeProfit()          ?  TextByLanguage("Срабатывание TakeProfit","Due to TakeProfit")              :\
      this.Reason()==ORDER_REASON_EXPERT  ?  TextByLanguage("Выставлен из mql4-программы","Placed from mql4 program")   :\
      TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4")\
     );\
#else\
   string res="";\
   switch(this.Status())\
     {\
      case ORDER_STATUS_MARKET_ACTIVE        :\
         res=\
           (\
            reason==POSITION_REASON_CLIENT   ?  TextByLanguage("Позиция открыта из десктопного терминала","Position opened from desktop terminal") :\
            reason==POSITION_REASON_MOBILE   ?  TextByLanguage("Позиция открыта из мобильного приложения","Position opened from mobile app") :\
            reason==POSITION_REASON_WEB      ?  TextByLanguage("Позиция открыта из веб-платформы","Position opened from web platform") :\
            reason==POSITION_REASON_EXPERT   ?  TextByLanguage("Позиция открыта из советника или скрипта","Position opened from EA or script") : ""\
           );\
         break;\
      case ORDER_STATUS_MARKET_PENDING       :\
      case ORDER_STATUS_HISTORY_PENDING      :\
      case ORDER_STATUS_HISTORY_ORDER        :\
         res=\
           (\
            reason==ORDER_REASON_CLIENT      ?  TextByLanguage("Ордер выставлен из десктопного терминала","Order set from desktop terminal") :\
            reason==ORDER_REASON_MOBILE      ?  TextByLanguage("Ордер выставлен из мобильного приложения","Order set from mobile app") :\
            reason==ORDER_REASON_WEB         ?  TextByLanguage("Ордер выставлен из веб-платформы","Oder set from web platform") :\
            reason==ORDER_REASON_EXPERT      ?  TextByLanguage("Ордер выставлен советником или скриптом","Order set from EA or script") :\
            reason==ORDER_REASON_SL          ?  TextByLanguage("Срабатывание StopLoss","Due to StopLoss") :\
            reason==ORDER_REASON_TP          ?  TextByLanguage("Срабатывание TakeProfit","Due to TakeProfit") :\
            reason==ORDER_REASON_SO          ?  TextByLanguage("Ордер выставлен в результате наступления Stop Out","Due to Stop Out") : ""\
           );\
         break;\
      case ORDER_STATUS_DEAL                 :\
         res=\
           (\
            reason==DEAL_REASON_CLIENT       ?  TextByLanguage("Сделка проведена из десктопного терминала","Deal carried out from desktop terminal") :\
            reason==DEAL_REASON_MOBILE       ?  TextByLanguage("Сделка проведена из мобильного приложения","Deal carried out from mobile app") :\
            reason==DEAL_REASON_WEB          ?  TextByLanguage("Сделка проведена из веб-платформы","Deal carried out from web platform") :\
            reason==DEAL_REASON_EXPERT       ?  TextByLanguage("Сделка проведена из советника или скрипта","Deal carried out from EA or script") :\
            reason==DEAL_REASON_SL           ?  TextByLanguage("Срабатывание StopLoss","Due to StopLoss") :\
            reason==DEAL_REASON_TP           ?  TextByLanguage("Срабатывание TakeProfit","Due to TakeProfit") :\
            reason==DEAL_REASON_SO           ?  TextByLanguage("Сделка проведена в результате наступления Stop Out","Due to Stop Out") :\
            reason==DEAL_REASON_ROLLOVER     ?  TextByLanguage("Сделка проведена по причине переноса позиции","Due to position rollover") :\
            reason==DEAL_REASON_VMARGIN      ?  TextByLanguage("Сделка проведена по причине начисления/списания вариационной маржи","Due to variation margin") :\
            reason==DEAL_REASON_SPLIT        ?  TextByLanguage("Сделка проведена по причине сплита (понижения цены) инструмента","Due to split") : ""\
           );\
         break;\
      default                                : res="";   break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
```\
\
Affiliation to MQL4 or MQL5 is checked, the order reason passed in the inputs is verified according to the order status and its description is returned.\
\
```\
//+------------------------------------------------------------------+\
//| Deal direction description                                       |\
//+------------------------------------------------------------------+\
string COrder::GetEntryDescription(const long deal_entry) const\
  {\
#ifdef __MQL4__\
   return TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4");\
#else\
   string res="";\
   switch(this.Status())\
     {\
      case ORDER_STATUS_MARKET_ACTIVE     :\
         res=TextByLanguage("Свойство не поддерживается у позиции","Property not supported for position");\
         break;\
      case ORDER_STATUS_MARKET_PENDING    :\
      case ORDER_STATUS_HISTORY_PENDING   :\
         res=TextByLanguage("Свойство не поддерживается у отложенного ордера","Property not supported for pending order");\
         break;\
      case ORDER_STATUS_HISTORY_ORDER     :\
         res=TextByLanguage("Свойство не поддерживается у исторического ордера","Property not supported for history order");\
         break;\
      case ORDER_STATUS_DEAL              :\
         res=\
           (\
            deal_entry==DEAL_ENTRY_IN     ?  TextByLanguage("Вход в рынок","Entry to the market") :\
            deal_entry==DEAL_ENTRY_OUT    ?  TextByLanguage("Выход из рынка","Out from the market") :\
            deal_entry==DEAL_ENTRY_INOUT  ?  TextByLanguage("Разворот","Reversal") :\
            deal_entry==DEAL_ENTRY_OUT_BY ?  TextByLanguage("Закрытие встречной позицией","Closing by opposite position") : ""\
           );\
         break;\
      default                             : res=""; break;\
     }\
   return res;\
#endif\
  }\
//+------------------------------------------------------------------+\
```\
\
Affiliation to MQL4 or MQL5 is checked, deal direction passed in the inputs is verified according to the order status and its description is returned.\
\
```\
//+------------------------------------------------------------------+\
//| Return deal type name                                            |\
//+------------------------------------------------------------------+\
string COrder::GetTypeDealDescription(const long deal_type) const\
  {\
#ifdef __MQL4__\
   return TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4");\
#else\
   string res="";\
   switch(this.Status())\
     {\
      case ORDER_STATUS_MARKET_ACTIVE     :\
         res=TextByLanguage("Свойство не поддерживается у позиции","Property not supported for position");\
         break;\
      case ORDER_STATUS_MARKET_PENDING    :\
      case ORDER_STATUS_HISTORY_PENDING   :\
         res=TextByLanguage("Свойство не поддерживается у отложенного ордера","Property not supported for pending order");\
         break;\
      case ORDER_STATUS_HISTORY_ORDER     :\
         res=TextByLanguage("Свойство не поддерживается у исторического ордера","Property not supported for history order");\
         break;\
      case ORDER_STATUS_DEAL              :\
         res=\
           (\
            deal_type==DEAL_TYPE_BUY                      ?  TextByLanguage("Сделка на покупку","Buy deal") :\
            deal_type==DEAL_TYPE_SELL                     ?  TextByLanguage("Сделка на продажу","Sell deal") :\
            deal_type==DEAL_TYPE_BALANCE                  ?  TextByLanguage("Начисление баланса","Balance accrual") :\
            deal_type==DEAL_TYPE_CREDIT                   ?  TextByLanguage("Начисление кредита","Credit accrual") :\
            deal_type==DEAL_TYPE_CHARGE                   ?  TextByLanguage("Дополнительные сборы","Extra charges") :\
            deal_type==DEAL_TYPE_CORRECTION               ?  TextByLanguage("Корректирующая запись","Corrective entry") :\
            deal_type==DEAL_TYPE_BONUS                    ?  TextByLanguage("Перечисление бонусов","Bonuses") :\
            deal_type==DEAL_TYPE_COMMISSION               ?  TextByLanguage("Дополнительные комиссии","Additional comissions") :\
            deal_type==DEAL_TYPE_COMMISSION_DAILY         ?  TextByLanguage("Комиссия, начисляемая в конце торгового дня","Commission accrued at the end of a trading day") :\
            deal_type==DEAL_TYPE_COMMISSION_MONTHLY       ?  TextByLanguage("Комиссия, начисляемая в конце месяца","Commission accrued at the end of a month") :\
            deal_type==DEAL_TYPE_COMMISSION_AGENT_DAILY   ?  TextByLanguage("Агентская комиссия, начисляемая в конце торгового дня","Agency commission charged at the end of a trading day") :\
            deal_type==DEAL_TYPE_COMMISSION_AGENT_MONTHLY ?  TextByLanguage("Агентская комиссия, начисляемая в конце месяца","Agency commission charged at the end of a month") :\
            deal_type==DEAL_TYPE_INTEREST                 ?  TextByLanguage("Начисления процентов на свободные средства","Accrued interest on free funds") :\
            deal_type==DEAL_TYPE_BUY_CANCELED             ?  TextByLanguage("Отмененная сделка покупки","Canceled buy transaction") :\
            deal_type==DEAL_TYPE_SELL_CANCELED            ?  TextByLanguage("Отмененная сделка продажи","Canceled sell transaction") :\
            deal_type==DEAL_DIVIDEND                      ?  TextByLanguage("Начисление дивиденда","Accrued dividends") :\
            deal_type==DEAL_DIVIDEND_FRANKED              ?  TextByLanguage("Начисление франкированного дивиденда","Accrual of franked dividend") :\
            deal_type==DEAL_TAX                           ?  TextByLanguage("Начисление налога","Tax accrual") : ""\
           );\
         break;\
      default                             : res=""; break;\
     }\
   return res;\
#endif\
  }\
```\
\
Affiliation to MQL4 or MQL5 is checked, deal type passed in the inputs is verified according to the order status and its description is returned.\
\
**Implementing the methods describing order properties:**\
\
```\
//+------------------------------------------------------------------+\
//| Return description of an order's integer property                |\
//+------------------------------------------------------------------+\
string COrder::GetPropertyDescription(ENUM_ORDER_PROP_INTEGER property)\
  {\
   return\
     (\
   //--- General properties\
      property==ORDER_PROP_MAGIC             ?  TextByLanguage("Магик","Magic")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+(string)this.GetProperty(property)\
         )  :\
      property==ORDER_PROP_TICKET            ?  TextByLanguage("Тикет","Ticket")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          " #"+(string)this.GetProperty(property)\
         )  :\
      property==ORDER_PROP_TICKET_FROM       ?  TextByLanguage("Тикет родительского ордера","Ticket of parent order")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          " #"+(string)this.GetProperty(property)\
         )  :\
      property==ORDER_PROP_TICKET_TO         ?  TextByLanguage("Тикет наследуемого ордера","Inherited order ticket")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          " #"+(string)this.GetProperty(property)\
         )  :\
      property==ORDER_PROP_TIME_OPEN         ?  TextByLanguage("Время открытия","Open time")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+::TimeToString(this.GetProperty(property),TIME_DATE|TIME_MINUTES|TIME_SECONDS)\
         )  :\
      property==ORDER_PROP_TIME_CLOSE        ?  TextByLanguage("Время закрытия","Close time")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+::TimeToString(this.GetProperty(property),TIME_DATE|TIME_MINUTES|TIME_SECONDS)\
         )  :\
      property==ORDER_PROP_TIME_EXP          ?  TextByLanguage("Дата экспирации","Expiration date")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          (this.GetProperty(property)==0     ?  TextByLanguage(": Не задана",": Not set") :\
          ": "+::TimeToString(this.GetProperty(property),TIME_DATE|TIME_MINUTES|TIME_SECONDS))\
         )  :\
      property==ORDER_PROP_TYPE              ?  TextByLanguage("Тип","Type")+": "+this.TypeDescription()                   :\
      property==ORDER_PROP_DIRECTION         ?  TextByLanguage("Тип по направлению","Type by direction")+": "+this.DirectionDescription() :\
\
      property==ORDER_PROP_REASON            ?  TextByLanguage("Причина","Reason")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+this.GetReasonDescription(this.GetProperty(property))\
         )  :\
      property==ORDER_PROP_POSITION_ID       ?  TextByLanguage("Идентификатор позиции","Position ID")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": #"+(string)this.GetProperty(property)\
         )  :\
      property==ORDER_PROP_DEAL_ORDER        ?  TextByLanguage("Сделка на основании ордера","Deal by order")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": #"+(string)this.GetProperty(property)\
         )  :\
      property==ORDER_PROP_DEAL_ENTRY        ?  TextByLanguage("Направление сделки","Deal direction")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+this.GetEntryDescription(this.GetProperty(property))\
         )  :\
      property==ORDER_PROP_POSITION_BY_ID    ?  TextByLanguage("Идентификатор встречной позиции","Opposite position ID")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+(string)this.GetProperty(property)\
         )  :\
      property==ORDER_PROP_TIME_OPEN_MSC     ?  TextByLanguage("Время открытия в милисекундах","Open time in milliseconds")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+(string)this.GetProperty(property)+" > "+TimeMSCtoString(this.GetProperty(property))\
         )  :\
      property==ORDER_PROP_TIME_CLOSE_MSC    ?  TextByLanguage("Время закрытия в милисекундах","Close time in milliseconds")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+(string)this.GetProperty(property)+" > "+TimeMSCtoString(this.GetProperty(property))\
         )  :\
      property==ORDER_PROP_TIME_UPDATE       ?  TextByLanguage("Время изменения позиции","Position change time")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+(this.GetProperty(property)!=0 ? ::TimeToString(this.GetProperty(property),TIME_DATE|TIME_MINUTES|TIME_SECONDS) : "0")\
         )  :\
      property==ORDER_PROP_TIME_UPDATE_MSC   ?  TextByLanguage("Время изменения позиции в милисекундах","Position change time in milliseconds")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+(this.GetProperty(property)!=0 ? (string)this.GetProperty(property)+" > "+TimeMSCtoString(this.GetProperty(property)) : "0")\
         )  :\
   //--- Additional property\
      property==ORDER_PROP_STATUS            ?  TextByLanguage("Статус","Status")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": \""+this.StatusDescription()+"\""\
         )  :\
      property==ORDER_PROP_PROFIT_PT         ?  TextByLanguage("Прибыль в пунктах","Profit in points")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+(string)this.GetProperty(property)\
         )  :\
      property==ORDER_PROP_CLOSE_BY_SL       ?  TextByLanguage("Закрытие по StopLoss","Close by StopLoss")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+(this.GetProperty(property) ? TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))\
         )  :\
      property==ORDER_PROP_CLOSE_BY_TP       ?  TextByLanguage("Закрытие по TakeProfit","Close by TakeProfit")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+(this.GetProperty(property) ? TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))\
         )  :\
      ""\
     );\
  }\
//+------------------------------------------------------------------+\
//| Return description of order's real property                      |\
//+------------------------------------------------------------------+\
string COrder::GetPropertyDescription(ENUM_ORDER_PROP_DOUBLE property)\
  {\
   int dg=(int)::SymbolInfoInteger(this.GetProperty(ORDER_PROP_SYMBOL),SYMBOL_DIGITS);\
   int dgl=(int)DigitsLots(this.GetProperty(ORDER_PROP_SYMBOL));\
   return\
     (\
      //--- General properties\
      property==ORDER_PROP_PRICE_CLOSE       ?  TextByLanguage("Цена закрытия","Close price")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+::DoubleToString(this.GetProperty(property),dg)\
         )  :\
      property==ORDER_PROP_PRICE_OPEN        ?  TextByLanguage("Цена открытия","Open price")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+::DoubleToString(this.GetProperty(property),dg)\
         )  :\
      property==ORDER_PROP_SL                ?  TextByLanguage("Цена StopLoss","StopLoss price")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          (this.GetProperty(property)==0     ?  TextByLanguage(": Отсутствует",": Not set"):": "+::DoubleToString(this.GetProperty(property),dg))\
         )  :\
      property==ORDER_PROP_TP                ?  TextByLanguage("Цена TakeProfit","TakeProfit price")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          (this.GetProperty(property)==0     ?  TextByLanguage(": Отсутствует",": Not set"):": "+::DoubleToString(this.GetProperty(property),dg))\
         )  :\
      property==ORDER_PROP_PROFIT            ?  TextByLanguage("Прибыль","Profit")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+::DoubleToString(this.GetProperty(property),2)\
         )  :\
      property==ORDER_PROP_COMMISSION        ?  TextByLanguage("Комиссия","Comission")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+::DoubleToString(this.GetProperty(property),2)\
         )  :\
      property==ORDER_PROP_SWAP              ?  TextByLanguage("Своп","Swap")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+::DoubleToString(this.GetProperty(property),2)\
         )  :\
      property==ORDER_PROP_VOLUME            ?  TextByLanguage("Объём","Volume")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+::DoubleToString(this.GetProperty(property),dgl)\
          ) :\
      property==ORDER_PROP_VOLUME_CURRENT    ?  TextByLanguage("Невыполненный объём","Unfulfilled volume")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+::DoubleToString(this.GetProperty(property),dgl)\
          ) :\
      property==ORDER_PROP_PRICE_STOP_LIMIT  ?\
         TextByLanguage("Цена постановки Limit ордера при активации StopLimit ордера","Price of placing Limit order when StopLimit order activated")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+::DoubleToString(this.GetProperty(property),dg)\
          ) :\
      //--- Additional property\
      property==ORDER_PROP_PROFIT_FULL       ?  TextByLanguage("Прибыль+комиссия+своп","Profit+Comission+Swap")+\
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
          ": "+::DoubleToString(this.GetProperty(property),2)\
          ) :\
      ""\
     );\
  }\
//+------------------------------------------------------------------+\
//| Return description of the order's string property                |\
//+------------------------------------------------------------------+\
string COrder::GetPropertyDescription(ENUM_ORDER_PROP_STRING property)\
  {\
   return\
     (\
      property==ORDER_PROP_SYMBOL         ?  TextByLanguage("Символ","Symbol")+": \""+this.GetProperty(property)+"\""            :\
      property==ORDER_PROP_COMMENT        ?  TextByLanguage("Комментарий","Comment")+\
         (this.GetProperty(property)==""  ?  TextByLanguage(": Отсутствует",": Not set"):": \""+this.GetProperty(property)+"\"") :\
      property==ORDER_PROP_EXT_ID         ?  TextByLanguage("Идентификатор на бирже","ID on exchange")+\
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :\
         (this.GetProperty(property)==""  ?  TextByLanguage(": Отсутствует",": Not set"):": \""+this.GetProperty(property)+"\"")):\
      ""\
     );\
  }\
//+------------------------------------------------------------------+\
```\
\
**Implementation of the methods for returning a name of an order status, a name of an order or a position, a name of a deal direction, a description of an order/position direction type and the method of convenient display of order properties in the journal.**\
\
First, add yet another macro substitution to the Defines.mqh file for convenient display of a function/method name in the journal:\
\
```\
//+------------------------------------------------------------------+\
//| Macro substitutions                                              |\
//+------------------------------------------------------------------+\
#define COUNTRY_LANG    "Russian"            // Country language\
#define DFUN           (__FUNCTION__+": ")   // "Function description"\
//+------------------------------------------------------------------+\
```\
\
Now, instead of writing the strings in full, we can simply write DFUN, and the compiler changes the description into the strings set in the macro.\
\
```\
//+------------------------------------------------------------------+\
//| Return the order status name                                     |\
//+------------------------------------------------------------------+\
string COrder::StatusDescription(void) const\
  {\
   ENUM_ORDER_STATUS status=this.Status();\
   ENUM_ORDER_TYPE   type=(ENUM_ORDER_TYPE)this.TypeOrder();\
   return\
     (\
      status==ORDER_STATUS_BALANCE        ?  TextByLanguage("Балансная операция","Balance operation") :\
      status==ORDER_STATUS_CREDIT         ?  TextByLanguage("Кредитная операция","Credits operation") :\
      status==ORDER_STATUS_HISTORY_ORDER  || status==ORDER_STATUS_HISTORY_PENDING                     ?\
      (\
       type>ORDER_TYPE_SELL ? TextByLanguage("Отложенный ордер","Pending order")                      :\
       TextByLanguage("Ордер на ","The order to ")+(type==ORDER_TYPE_BUY ? TextByLanguage("покупку","buy") : TextByLanguage("продажу","sell"))\
      )                                                                                               :\
      status==ORDER_STATUS_DEAL           ?  TextByLanguage("Сделка","Deal")                          :\
      status==ORDER_STATUS_MARKET_ACTIVE  ?  TextByLanguage("Позиция","Active position")              :\
      status==ORDER_STATUS_MARKET_PENDING ?  TextByLanguage("Установленный отложенный ордер","Active pending order") :\
      ""\
     );\
  }\
//+------------------------------------------------------------------+\
//| Return order or position name                                    |\
//+------------------------------------------------------------------+\
string COrder::TypeDescription(void) const\
  {\
   if(this.Status()==ORDER_STATUS_DEAL)\
      return this.GetTypeDealDescription(this.TypeOrder());\
   else switch(this.TypeOrder())\
     {\
      case ORDER_TYPE_BUY              :  return "Buy";\
      case ORDER_TYPE_BUY_LIMIT        :  return "Buy Limit";\
      case ORDER_TYPE_BUY_STOP         :  return "Buy Stop";\
      case ORDER_TYPE_SELL             :  return "Sell";\
      case ORDER_TYPE_SELL_LIMIT       :  return "Sell Limit";\
      case ORDER_TYPE_SELL_STOP        :  return "Sell Stop";\
      #ifdef __MQL4__\
      case ORDER_TYPE_BALANCE          :  return TextByLanguage("Балансовая операция","Balance operation");\
      case ORDER_TYPE_CREDIT           :  return TextByLanguage("Кредитная операция","Credit operation");\
      #else\
      case ORDER_TYPE_BUY_STOP_LIMIT   :  return "Buy Stop Limit";\
      case ORDER_TYPE_SELL_STOP_LIMIT  :  return "Sell Stop Limit";\
      #endif\
      default                          :  return TextByLanguage("Неизвестный тип","Unknown type");\
     }\
  }\
//+------------------------------------------------------------------+\
//| Return deal direction name                                       |\
//+------------------------------------------------------------------+\
string COrder::DealEntryDescription(void) const\
  {\
   return(this.Status()==ORDER_STATUS_DEAL ? this.GetEntryDescription(this.GetProperty(ORDER_PROP_DEAL_ENTRY)) : "");\
  }\
//+------------------------------------------------------------------+\
//| Return order/position direction type                             |\
//+------------------------------------------------------------------+\
string COrder::DirectionDescription(void) const\
  {\
   if(this.Status()==ORDER_STATUS_DEAL)\
      return this.TypeDescription();\
   switch(this.TypeByDirection())\
     {\
      case ORDER_TYPE_BUY  :  return "Buy";\
      case ORDER_TYPE_SELL :  return "Sell";\
      default              :  return TextByLanguage("Неизвестный тип","Unknown type");\
     }\
  }\
//+------------------------------------------------------------------+\
//| Send order properties to the journal                             |\
//+------------------------------------------------------------------+\
void COrder::Print(const bool full_prop=false)\
  {\
   ::Print("============= ",TextByLanguage("Начало списка параметров: \"","Beginning of the parameter list: \""),this.StatusDescription(),"\" =============");\
   int beg=0, end=ORDER_PROP_INTEGER_TOTAL;\
   for(int i=beg; i<end; i++)\
     {\
      ENUM_ORDER_PROP_INTEGER prop=(ENUM_ORDER_PROP_INTEGER)i;\
      if(!full_prop && !this.SupportProperty(prop)) continue;\
      ::Print(this.GetPropertyDescription(prop));\
     }\
   ::Print("------");\
   beg=end; end+=ORDER_PROP_DOUBLE_TOTAL;\
   for(int i=beg; i<end; i++)\
     {\
      ENUM_ORDER_PROP_DOUBLE prop=(ENUM_ORDER_PROP_DOUBLE)i;\
      if(!full_prop && !this.SupportProperty(prop)) continue;\
      ::Print(this.GetPropertyDescription(prop));\
     }\
   ::Print("------");\
   beg=end; end+=ORDER_PROP_STRING_TOTAL;\
   for(int i=beg; i<end; i++)\
     {\
      ENUM_ORDER_PROP_STRING prop=(ENUM_ORDER_PROP_STRING)i;\
      if(!full_prop && !this.SupportProperty(prop)) continue;\
      ::Print(this.GetPropertyDescription(prop));\
     }\
   ::Print("================== ",TextByLanguage("Конец списка параметров: \"","End of the parameter list: \""),this.StatusDescription(),"\" ==================\n");\
  }\
//+------------------------------------------------------------------+\
```\
\
Now that all methods of the COrder abstract order class have been implemented, we can see what we have really got at this stage.\
\
### COrder abstract order test\
\
Let's test and see how COrder base objects are created and filled with data. Further on, we will develop objects by types based on it and store them in collections.\
\
In the **Experts** directory or subdirectory, create a new folder and call it **TestDoEasy**. Next, create another folder inside the newly created one: **Part01**. The test EA's file is to be located in it.\
\
Right-click on Part01 in the editor navigator and select "New file" from the pop-up menu, or press Ctrl+N with the Part01 folder highlighted. MQL Wizard window opens with the "Expert Advisor (template)" option already selected, click Next and add the name of the new TestDoEasyPart01 file to the already specified file path in the name input field. Click Next till you reach the end of the wizard operation. We will not need any additional event handlers for the test, so you may leave all checkboxes empty. Click Finish to create the new EA template:\
\
\
```\
//+------------------------------------------------------------------+\
//|                                             TestDoEasyPart01.mq5 |\
//|                        Copyright 2018, MetaQuotes Software Corp. |\
//|                             https://mql5.com/en/users/artmedia70 |\
//+------------------------------------------------------------------+\
#property copyright "Copyright 2018, MetaQuotes Software Corp."\
#property link      "https://mql5.com/en/users/artmedia70"\
#property version   "1.00"\
//+------------------------------------------------------------------+\
//| Expert initialization function                                   |\
//+------------------------------------------------------------------+\
int OnInit()\
  {\
//---\
\
//---\
   return(INIT_SUCCEEDED);\
  }\
//+------------------------------------------------------------------+\
//| Expert deinitialization function                                 |\
//+------------------------------------------------------------------+\
void OnDeinit(const int reason)\
  {\
//---\
\
  }\
//+------------------------------------------------------------------+\
//| Expert tick function                                             |\
//+------------------------------------------------------------------+\
void OnTick()\
  {\
//---\
\
  }\
//+------------------------------------------------------------------+\
```\
\
The test is as simple as possible, so there is no need to reinvent the wheel. So, we simply include the class of our COrder abstract order and [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) class from the [standard library](https://www.mql5.com/en/docs/standardlibrary), and create the object list:\
\
```\
//+------------------------------------------------------------------+\
//|                                             TestDoEasyPart01.mq5 |\
//|                        Copyright 2018, MetaQuotes Software Corp. |\
//|                             https://mql5.com/en/users/artmedia70 |\
//+------------------------------------------------------------------+\
#property copyright "Copyright 2018, MetaQuotes Software Corp."\
#property link      "https://mql5.com/en/users/artmedia70"\
#property version   "1.00"\
//--- includes\
#include <DoEasy\Objects\Order.mqh>\
#include <Arrays\ArrayObj.mqh>\
//--- global variables\
CArrayObj      list_all_orders;\
//+------------------------------------------------------------------+\
//| Expert initialization function                                   |\
//+------------------------------------------------------------------+\
```\
\
Next, set the sorted list flag to the object list in OnInit() and fill it with all history deals and orders in two loops.\
\
Then display data on each of the filled list's objects in the terminal's Experts journal in the loop:\
\
```\
//+------------------------------------------------------------------+\
//| Expert initialization function                                   |\
//+------------------------------------------------------------------+\
int OnInit()\
  {\
//---\
   list_all_orders.Sort();\
   list_all_orders.Clear();\
   if(!HistorySelect(0,TimeCurrent()))\
     {\
      Print(DFUN,TextByLanguage(": Не удалось получить историю сделок и ордеров",": Failed to get history of deals and orders"));\
      return INIT_FAILED;\
     }\
//--- Deals\
   int total_deals=HistoryDealsTotal();\
   for(int i=0; i<total_deals; i++)\
     {\
      ulong deal_ticket=::HistoryDealGetTicket(i);\
      if(deal_ticket==0) continue;\
      COrder *deal=new COrder(ORDER_STATUS_DEAL,deal_ticket);\
      if(deal==NULL) continue;\
      list_all_orders.InsertSort(deal);\
     }\
//--- Orders\
   int total_orders=HistoryOrdersTotal();\
   for(int i=0; i<total_orders; i++)\
     {\
      ulong order_ticket=::HistoryOrderGetTicket(i);\
      if(order_ticket==0) continue;\
      ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)::HistoryOrderGetInteger(order_ticket,ORDER_TYPE);\
      if(type==ORDER_TYPE_BUY || type==ORDER_TYPE_SELL)\
        {\
         COrder *order=new COrder(ORDER_STATUS_HISTORY_ORDER,order_ticket);\
         if(order==NULL) continue;\
         list_all_orders.InsertSort(order);\
        }\
      else\
        {\
         COrder *order=new COrder(ORDER_STATUS_HISTORY_PENDING,order_ticket);\
         if(order==NULL) continue;\
         list_all_orders.InsertSort(order);\
        }\
     }\
//--- Display in the journal\
   int total=list_all_orders.Total();\
   for(int i=0;i<total;i++)\
     {\
      COrder *order=list_all_orders.At(i);\
      if(order==NULL)\
         continue;\
      order.Print();\
     }\
//---\
   return(INIT_SUCCEEDED);\
  }\
//+------------------------------------------------------------------+\
```\
\
If we now try to compile the EA, we get three errors:\
\
![](https://c.mql5.com/2/35/CompileErrors_en.png)\
\
The errors indicate that we cannot access the protected methods of the class from the outside. After all, we have made the COrder constructor in the protected section of the class. When we use the class as a basis to develop objects by types, everything will be fine as they will have open constructors. Now, in order to perform the test, let's simply move the constructor to the public section of the COrder class:\
\
![](https://c.mql5.com/2/35/MoveConstructor_en__1.png)\
\
Now all is compiled without errors, and data on all orders and deals in trading account's history are displayed in the terminal journal.\
\
![](https://c.mql5.com/2/35/Result__2.gif)\
\
All properties of each order/deal, including unsupported ones, are displayed.\
\
The fact is that we have developed the methods returning the flags for supporting specific properties by this order to be virtual, so that they are redefined in the derived classes. These derived classes are then used to display data in the journal. In that case, all should be displayed correctly. If there is a property not supported by the order, it is not displayed in the journal, since the **Print(const bool full\_prop=false)** method of the COrder class has the default flag for disabling a display of unsupported properties in the journal, while the **SupportProperty()** virtual methods of the class simply return 'true' for any property.\
\
### What's next?\
\
The first (and the smallest) part is ready. We have developed a basic object for the collection of history orders and deals as well as for the collection of market orders and positions. So far there is no practical value but this is only the beginning. This single basic object is to become a keystone for the system storing and displaying data on the order system. Our next step is to develop other necessary objects and collections using the same principles. I am also going to automate collection of constantly required data.\
\
In the next part, I am going to develop several classes based on this abstract order. The classes will describe specific order, deal and position types. I will develop the first collection (collection of history orders and deals), as well as the basis and the main object of the library — Engine. I will also get rid of errors while trying to compile the library in MQL4 in the subsequent parts of the library development description.\
\
Below is an archive with all the files discussed in the first part. You can download and analyze them in details. Leave your questions, comments and suggestions in the comments.\
\
Please note that the library is being developed in parallel with writing the articles. Therefore, the necessary revisions are made to the library's methods and variables from article to article. In this regard, there may be minor inconsistencies in the content of the attached files and description texts in the article. For example, there may be edits in the comments, rearrangements of the enumeration members, etc. This does not affect the performance of the attached examples.\
\
[Back to contents](https://www.mql5.com/en/articles/5654#node00).\
\
Translated from Russian by MetaQuotes Ltd.\
\
Original article: [https://www.mql5.com/ru/articles/5654](https://www.mql5.com/ru/articles/5654)\
\
**Attached files** \|\
\
\
[Download ZIP](https://www.mql5.com/en/articles/download/5654.zip "Download all attachments in the single ZIP archive")\
\
[MQL5.zip](https://www.mql5.com/en/articles/download/5654/mql5.zip "Download MQL5.zip")(16.19 KB)\
\
**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.\
\
This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.\
\
#### Other articles by this author\
\
- [Tables in the MVC Paradigm in MQL5: Customizable and sortable table columns](https://www.mql5.com/en/articles/19979)\
- [How to publish code to CodeBase: A practical guide](https://www.mql5.com/en/articles/19441)\
- [Tables in the MVC Paradigm in MQL5: Integrating the Model Component into the View Component](https://www.mql5.com/en/articles/19288)\
- [The View and Controller components for tables in the MQL5 MVC paradigm: Resizable elements](https://www.mql5.com/en/articles/18941)\
- [The View and Controller components for tables in the MQL5 MVC paradigm: Containers](https://www.mql5.com/en/articles/18658)\
- [The View and Controller components for tables in the MQL5 MVC paradigm: Simple controls](https://www.mql5.com/en/articles/18221)\
- [The View component for tables in the MQL5 MVC paradigm: Base graphical element](https://www.mql5.com/en/articles/17960)\
\
**Last comments \|**\
**[Go to discussion](https://www.mql5.com/en/forum/310865)**\
(69)\
\
\
![albertpess](https://c.mql5.com/avatar/2019/5/5CE3FD80-56C2.jpg)\
\
**[albertpess](https://www.mql5.com/en/users/albertpess)**\
\|\
26 Apr 2021 at 00:49\
\
**Artyom Trishkin:**\
\
So, somewhere they did something wrong. Missed something.\
\
Just download the files attached to the article to your terminal folder and compile the EA.\
\
And only then disassemble everything step by step as described in the article.\
\
Thank you.\
\
![Enrique Enguix](https://c.mql5.com/avatar/2025/9/68c108f2-b619.jpg)\
\
**[Enrique Enguix](https://www.mql5.com/en/users/envex)**\
\|\
2 Feb 2022 at 19:40\
\
Very [useful](https://www.mql5.com/en/docs/array/arrayfill "MQL5 Documentation: ArrayFill function") and well done! Thanks\
\
\
![Peng Peng Liu](https://c.mql5.com/avatar/avatar_na2.png)\
\
**[Peng Peng Liu](https://www.mql5.com/en/users/yylnthz)**\
\|\
21 Dec 2023 at 10:46\
\
It's pretty good.\
\
\
![Spoxus Spoxus](https://c.mql5.com/avatar/avatar_na2.png)\
\
**[Spoxus Spoxus](https://www.mql5.com/en/users/spoxus)**\
\|\
26 Dec 2024 at 12:40\
\
**Implementing the protected class constructor:**\
\
```\
//+------------------------------------------------------------------+\
//| Closed parametric constructor                                    |\
//+------------------------------------------------------------------+\
COrder::COrder(ENUM_ORDER_STATUS order_status,const ulong ticket)\
  {\
//--- Save integer properties\
   m_ticket=ticket;\
   m_long_prop[ORDER_PROP_STATUS]                              = order_status;\
   m_long_prop[ORDER_PROP_MAGIC]                               = this.OrderMagicNumber();\
   m_long_prop[ORDER_PROP_TICKET]                              = this.OrderTicket();\
```\
\
Hello Artyom,\
\
can we use:\
\
m\_long\_prop\[ORDER\_PROP\_TICKET\] = (long)ticket;// ticket is the parameter passed in.\
\
![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)\
\
**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**\
\|\
26 Dec 2024 at 13:48\
\
**Ming Ge [#](https://www.mql5.com/en/forum/310865#comment_55469607) :**\
\
this . OrderTicket ();\
\
If\
\
```\
 this . OrderTicket ();\
```\
\
returns the value of m\_ticket set above, then it is possible.\
\
But I wrote this so long ago that I can’t remember right away.\
\
I looked. This method writes the ticket to the order property:\
\
```\
 //+------------------------------------------------------------------+\
 //| Возвращает тикет                                                 |\
 //+------------------------------------------------------------------+\
 long COrder::OrderTicket( void ) const\
  {\
 #ifdef __MQL4__\
   return ::OrderTicket();\
 #else\
   long res= 0 ;\
   switch ((ENUM_ORDER_STATUS) this .GetProperty(ORDER_PROP_STATUS))\
     {\
       case ORDER_STATUS_MARKET_POSITION   :\
       case ORDER_STATUS_MARKET_ORDER      :\
       case ORDER_STATUS_MARKET_PENDING    :\
       case ORDER_STATUS_HISTORY_PENDING   :\
       case ORDER_STATUS_HISTORY_ORDER     :\
       case ORDER_STATUS_DEAL              : res=( long )m_ticket;  break ;\
       default                             : res= 0 ;               break ;\
     }\
   return res;\
 #endif\
   }\
```\
\
Decide for yourself whether you need to replace it with a simple assignment or not.\
\
![Studying candlestick analysis techniques (part III): Library for pattern operations](https://c.mql5.com/2/35/Pattern_I__4.png)[Studying candlestick analysis techniques (part III): Library for pattern operations](https://www.mql5.com/en/articles/5751)\
\
The purpose of this article is to create a custom tool, which would enable users to receive and use the entire array of information about patterns discussed earlier. We will create a library of pattern related functions which you will be able to use in your own indicators, trading panels, Expert Advisors, etc.\
\
![Scraping bond yield data from the web](https://c.mql5.com/2/35/MQL5-avatar-web_scraping.png)[Scraping bond yield data from the web](https://www.mql5.com/en/articles/5204)\
\
Automate the collection of interest rate data to improve the performance of an Expert Advisor.\
\
![Library for easy and quick development of MetaTrader programs (part II). Collection of historical orders and deals](https://c.mql5.com/2/35/MQL5-avatar-doeasy__1.png)[Library for easy and quick development of MetaTrader programs (part II). Collection of historical orders and deals](https://www.mql5.com/en/articles/5669)\
\
In the first part, we started creating a large cross-platform library simplifying the development of programs for MetaTrader 5 and MetaTrader 4 platforms. We created the COrder abstract object which is a base object for storing data on history orders and deals, as well as on market orders and positions. Now we will develop all the necessary objects for storing account history data in collections.\
\
![Developing graphical interfaces for Expert Advisors and indicators based on .Net Framework and C#](https://c.mql5.com/2/35/icon__3.png)[Developing graphical interfaces for Expert Advisors and indicators based on .Net Framework and C#](https://www.mql5.com/en/articles/5563)\
\
The article presents a simple and fast method of creating graphical windows using Visual Studio with subsequent integration into the Expert Advisor's MQL code. The article is meant for non-specialist audiences and does not require any knowledge of C# and .Net technology.\
\
[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=zypswqspmqamvtuawetqwrwwaeunhafl&ssn=1769186583602896835&ssn_dr=0&ssn_sr=0&fv_date=1769186583&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F5654&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20I).%20Concept%2C%20data%20management%20and%20first%20results%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691865831637988&fz_uniq=5070505581027989270&sv=2552)\
\
This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).\
\
![close](https://c.mql5.com/i/close.png)\
\
![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)\
\
You are missing trading opportunities:\
\
- Free trading apps\
- Over 8,000 signals for copying\
- Economic news for exploring financial markets\
\
RegistrationLog in\
\
latin characters without spaces\
\
a password will be sent to this email\
\
An error occurred\
\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)\
\
You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)\
\
If you do not have an account, please [register](https://www.mql5.com/en/auth_register)\
\
Allow the use of cookies to log in to the MQL5.com website.\
\
Please enable the necessary setting in your browser, otherwise you will not be able to log in.\
\
[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)