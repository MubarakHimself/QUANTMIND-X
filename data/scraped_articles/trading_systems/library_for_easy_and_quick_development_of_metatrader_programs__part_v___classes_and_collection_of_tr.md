---
title: Library for easy and quick development of MetaTrader programs (part V): Classes and collection of trading events, sending events to the program
url: https://www.mql5.com/en/articles/6211
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:42:11.766737
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=rzsfznxypesehayrjtvjomsldqzxjglj&ssn=1769186528935500883&ssn_dr=0&ssn_sr=0&fv_date=1769186528&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F6211&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20V)%3A%20Classes%20and%20collection%20of%20trading%20events%2C%20sending%20events%20to%20the%20program%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918652823956115&fz_uniq=5070494929509095135&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Re-arranging the library structure](https://www.mql5.com/en/articles/6211#node01)
- [Event classes](https://www.mql5.com/en/articles/6211#node02)
- [Collection of trading events](https://www.mql5.com/en/articles/6211#node03)
- [Testing the processes of defining, handling and receiving events](https://www.mql5.com/en/articles/6211#node04)
- [What's next?](https://www.mql5.com/en/articles/6211#node05)


### Re-arranging the library structure

In the previous articles, we started creating a large cross-platform library simplifying the development of programs for [MetaTrader \\
5](https://www.metaquotes.net/en/metatrader5 "https://www.metaquotes.net/en/metatrader5") and MetaTrader 4 platforms. In the [fourth part](https://www.mql5.com/en/articles/5724), we tested tracking
trading events on the account. In this article, we will develop trading event classes and place them to the event collections. From there,
they will be sent to the base object of the Engine library and the control program chart.

**But first, let's prepare the ground for the further development of the library structure.**

Since we will have a lot of different collections, and each collection will have its own objects inherent only in this collection, it seems
reasonable to store objects for each collection in separate subfolders.

To do this, let's create **Orders** and **Events** folders in the **Objects** subfolder of the **DoEasy**
library root directory.

Move all previously created classes from the Objects folder to the Orders one, while the Events folder is to store the classes of event
objects we are going to develop in this article.

Also, relocate the Select.mqh file from Collections to Services since we are going to include yet another service class to it. The class
features methods for the fast access to any properties of any objects from the existing and future collections, which means it should be
located in the folder of service classes.

After relocating the file of the CSelect class and moving the order object classes to the new directory, the relative addresses of files
required for their compilation change as well. Therefore, let's move along the listings of moved classes and replace addresses of included
files in them:

**In the Order.mqh file**, replace the path for including the service functions file

```
#include "..\Services\DELib.mqh"
```

with

```
#include "..\..\Services\DELib.mqh"
```

**In the HistoryCollection.mqh file**, replace the paths

```
#include "Select.mqh"
#include "..\Objects\HistoryOrder.mqh"
#include "..\Objects\HistoryPending.mqh"
#include "..\Objects\HistoryDeal.mqh"
```

with

```
#include "..\Services\Select.mqh"
#include "..\Objects\Orders\HistoryOrder.mqh"
#include "..\Objects\Orders\HistoryPending.mqh"
#include "..\Objects\Orders\HistoryDeal.mqh"
```

**In the MarketCollection.mqh file**, replace the paths

```
#include "Select.mqh"
#include "..\Objects\MarketOrder.mqh"
#include "..\Objects\MarketPending.mqh"
#include "..\Objects\MarketPosition.mqh"
```

with

```
#include "..\Services\Select.mqh"
#include "..\Objects\Orders\MarketOrder.mqh"
#include "..\Objects\Orders\MarketPending.mqh"
#include "..\Objects\Orders\MarketPosition.mqh"
```

Now everything should be compiled without errors.

Since the number of upcoming collections is huge, it would be good to distinguish the ownership of the collection list based on [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj)
to identify the list. Each collection features a method returning the pointer to the full collection list. If somewhere there is a method that
receives a certain list of a certain collection, then inside this method, we need to be able to accurately identify the list passed to the
method by its belonging to one collection or another to avoid passing an additional flag indicating the type of the list passed to the method.

Fortunately, the standard library already provides the necessary tool for this in the form of the [Type()](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjtype)
virtual method returning the object ID.

For example, for [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject/cobjecttype),
the returned ID is 0, while for

[CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjtype), the ID is 0x7778.
Since the method is virtual, this allows the descendants of the classes to have their own methods returning specific IDs.

All of our collection lists are based on the **CArrayObj** class. We will create our own **CListObj** class which is a
descendant of the CArrayObj class, and the list ID is returned in its virtual Type() method. The ID itself is set as a constant in the class
constructor. Thus, we will continue to gain access to our collections as to the CArrayObj object, but now each list will have its own specific
ID.

First, let's set the necessary collection lists IDs in the
Defines.mqh file and add the macro describing the function with the error line
number to display debugging messages containing a string this message is sent from to locate the issue in the code during the
debugging:

```
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
//--- Describe the function with the error line number
#define DFUN_ERR_LINE            (__FUNCTION__+(TerminalInfoString(TERMINAL_LANGUAGE)=="Russian" ? ", Page " : ", Line ")+(string)__LINE__+": ")
#define DFUN                     (__FUNCTION__+": ")        // "Function description"
#define COUNTRY_LANG             ("Russian")                // Country language
#define END_TIME                 (D'31.12.3000 23:59:59')   // End date for requesting account history data
#define TIMER_FREQUENCY          (16)                       // Minimal frequency of the library timer in milliseconds
#define COLLECTION_PAUSE         (250)                      // Orders and deals collection timer pause in milliseconds
#define COLLECTION_COUNTER_STEP  (16)                       // Increment of the orders and deals collection timer counter
#define COLLECTION_COUNTER_ID    (1)                        // Orders and deals collection timer counter ID
#define COLLECTION_HISTORY_ID    (0x7778+1)                 // Historical collection list ID
#define COLLECTION_MARKET_ID     (0x7778+2)                 // Market collection list ID
#define COLLECTION_EVENTS_ID     (0x7778+3)                 // Events collection list ID
//+------------------------------------------------------------------+
```

Now create the CListObj class in the **ListObj.mqh** file within the Collections folder. The base class for it is CArrayObj:

```
//+------------------------------------------------------------------+
//|                                                      ListObj.mqh |
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
//| Collection lists class                                           |
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

All we have to do here is to declare a member of the class containing the list
type, add the method for defining the list type and the virtual
method for returning it.

In the class constructor, set the default
list type equal to that of the CArrajObj list. It can then be redefined from a calling program using the Type() method.

Now we need to inherit all collection lists from the class to be able to assign a separate search ID to each list. That ID will allow us to track the
list ownership in any methods the list will be passed to.

Open the **HistoryCollection.mqh** file, add inclusion of the CListObj
class and inherit the CHistoryCollection class from CListObj.

```
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Services\Select.mqh"
#include "..\Objects\Orders\HistoryOrder.mqh"
#include "..\Objects\Orders\HistoryPending.mqh"
#include "..\Objects\Orders\HistoryDeal.mqh"
//+------------------------------------------------------------------+
//| Collection of historical orders and deals                        |
//+------------------------------------------------------------------+
class CHistoryCollection : public CListObj
  {
```

In the class constructor, define the historical collection list type
we have specified as

COLLECTION\_HISTORY\_ID in the Defines.mqh file:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHistoryCollection::CHistoryCollection(void) : m_index_deal(0),m_delta_deal(0),m_index_order(0),m_delta_order(0),m_is_trade_event(false)
  {
   this.m_list_all_orders.Sort(#ifdef __MQL5__ SORT_BY_ORDER_TIME_OPEN #else SORT_BY_ORDER_TIME_CLOSE #endif );
   this.m_list_all_orders.Clear();
   this.m_list_all_orders.Type(COLLECTION_HISTORY_ID);
  }
//+------------------------------------------------------------------+
```

Do the same for the CMarketCollection class **in the MarketCollection.mqh file**:

```
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Services\Select.mqh"
#include "..\Objects\Orders\MarketOrder.mqh"
#include "..\Objects\Orders\MarketPending.mqh"
#include "..\Objects\Orders\MarketPosition.mqh"
//+------------------------------------------------------------------+
//| Collection of market orders and positions                        |
//+------------------------------------------------------------------+
class CMarketCollection : public CListObj
  {
```

In the class constructor, define the market collection type
we have specified in the Defines.mqh file as

COLLECTION\_MARKET\_ID:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CMarketCollection::CMarketCollection(void) : m_is_trade_event(false),m_is_change_volume(false),m_change_volume_value(0)
  {
   this.m_list_all_orders.Sort(SORT_BY_ORDER_TIME_OPEN);
   this.m_list_all_orders.Clear();
   ::ZeroMemory(this.m_struct_prev_market);
   this.m_struct_prev_market.hash_sum_acc=WRONG_VALUE;
   this.m_list_all_orders.Type(COLLECTION_MARKET_ID);
  }
//+------------------------------------------------------------------+
```

Now each of the collection lists has its ID simplifying identification of lists by their types.

* * *

Since we are to add new collections for working with new data types (including the account event collection in the present article), we will use
the new enumerations. To avoid name conflicts, we need to

replace the names of some previously created macro substitutions:

```
//+------------------------------------------------------------------+
//| Possible criteria of orders and deals sorting                    |
//+------------------------------------------------------------------+
#define FIRST_ORD_DBL_PROP          (ORDER_PROP_INTEGER_TOTAL)
#define FIRST_ORD_STR_PROP          (ORDER_PROP_INTEGER_TOTAL+ORDER_PROP_DOUBLE_TOTAL)
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
   SORT_BY_ORDER_STATUS          =  7,                      // Sort by order status (market order/pending order/deal/balance, credit operation)
   SORT_BY_ORDER_TYPE            =  8,                      // Sort by order type
   SORT_BY_ORDER_REASON          =  10,                     // Sort by order/position reason/source
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
   SORT_BY_ORDER_PRICE_OPEN      =  FIRST_ORD_DBL_PROP,     // Sort by open price
   SORT_BY_ORDER_PRICE_CLOSE     =  FIRST_ORD_DBL_PROP+1,   // Sort by close price
   SORT_BY_ORDER_SL              =  FIRST_ORD_DBL_PROP+2,   // Sort by StopLoss price
   SORT_BY_ORDER_TP              =  FIRST_ORD_DBL_PROP+3,   // Sort by TakeProfit price
   SORT_BY_ORDER_PROFIT          =  FIRST_ORD_DBL_PROP+4,   // Sort by profit
   SORT_BY_ORDER_COMMISSION      =  FIRST_ORD_DBL_PROP+5,   // Sort by commission
   SORT_BY_ORDER_SWAP            =  FIRST_ORD_DBL_PROP+6,   // Sort by swap
   SORT_BY_ORDER_VOLUME          =  FIRST_ORD_DBL_PROP+7,   // Sort by volume
   SORT_BY_ORDER_VOLUME_CURRENT  =  FIRST_ORD_DBL_PROP+8,   // Sort by unexecuted volume
   SORT_BY_ORDER_PROFIT_FULL     =  FIRST_ORD_DBL_PROP+9,   // Sort by profit+commission+swap criterion
   SORT_BY_ORDER_PRICE_STOP_LIMIT=  FIRST_ORD_DBL_PROP+10,  // Sort by Limit order when StopLimit order is activated
   //--- Sort by string properties
   SORT_BY_ORDER_SYMBOL          =  FIRST_ORD_STR_PROP,     // Sort by symbol
   SORT_BY_ORDER_COMMENT         =  FIRST_ORD_STR_PROP+1,   // Sort by comment
   SORT_BY_ORDER_EXT_ID          =  FIRST_ORD_STR_PROP+2    // Sort by order ID in an external trading system
  };
//+------------------------------------------------------------------+
```

Since we are currently editing the Defines.mqh files, add all necessary enumerations for the event classes and account event collection:

```
//+------------------------------------------------------------------+
//| Event status                                                     |
//+------------------------------------------------------------------+
enum ENUM_EVENT_STATUS
  {
   EVENT_STATUS_MARKET_POSITION,                            // Market position event (opening, partial opening, partial closing, adding volume, reversal)
   EVENT_STATUS_MARKET_PENDING,                             // Market pending order event (placing)
   EVENT_STATUS_HISTORY_PENDING,                            // Historical pending order event (removal)
   EVENT_STATUS_HISTORY_POSITION,                           // Historical position event (closing)
   EVENT_STATUS_BALANCE,                                    // Balance operation event (accruing balance, withdrawing funds and events from the ENUM_DEAL_TYPE enumeration)
  };
//+------------------------------------------------------------------+
//| Event reason                                                     |
//+------------------------------------------------------------------+
enum ENUM_EVENT_REASON
  {
   EVENT_REASON_ACTIVATED_PENDING               =  0,       // Pending order activation
   EVENT_REASON_ACTIVATED_PENDING_PARTIALLY     =  1,       // Pending order partial activation
   EVENT_REASON_CANCEL                          =  2,       // Cancelation
   EVENT_REASON_EXPIRED                         =  3,       // Order expiration
   EVENT_REASON_DONE                            =  4,       // Request executed in full
   EVENT_REASON_DONE_PARTIALLY                  =  5,       // Request executed partially
   EVENT_REASON_DONE_SL                         =  6,       // Closing by StopLoss
   EVENT_REASON_DONE_SL_PARTIALLY               =  7,       // Partial closing by StopLoss
   EVENT_REASON_DONE_TP                         =  8,       // Closing by TakeProfit
   EVENT_REASON_DONE_TP_PARTIALLY               =  9,       // Partial closing by TakeProfit
   EVENT_REASON_DONE_BY_POS                     =  10,      // Closing by an opposite position
   EVENT_REASON_DONE_PARTIALLY_BY_POS           =  11,      // Partial closing by an opposite position
   EVENT_REASON_DONE_BY_POS_PARTIALLY           =  12,      // Closing an opposite position by a partial volume
   EVENT_REASON_DONE_PARTIALLY_BY_POS_PARTIALLY =  13,      // Partial closing of an opposite position by a partial volume
   //--- Constants related to DEAL_TYPE_BALANCE deal type from the ENUM_DEAL_TYPE enumeration
   EVENT_REASON_BALANCE_REFILL                  =  14,      // Refilling the balance
   EVENT_REASON_BALANCE_WITHDRAWAL              =  15,      // Withdrawing funds from the account
   //--- List of constants is relevant to TRADE_EVENT_ACCOUNT_CREDIT from the ENUM_TRADE_EVENT enumeration and shifted to +13 relative to ENUM_DEAL_TYPE (EVENT_REASON_ACCOUNT_CREDIT-3)
   EVENT_REASON_ACCOUNT_CREDIT                  =  16,      // Accruing credit
   EVENT_REASON_ACCOUNT_CHARGE                  =  17,      // Additional charges
   EVENT_REASON_ACCOUNT_CORRECTION              =  18,      // Correcting entry
   EVENT_REASON_ACCOUNT_BONUS                   =  19,      // Accruing bonuses
   EVENT_REASON_ACCOUNT_COMISSION               =  20,      // Additional commissions
   EVENT_REASON_ACCOUNT_COMISSION_DAILY         =  21,      // Commission charged at the end of a trading day
   EVENT_REASON_ACCOUNT_COMISSION_MONTHLY       =  22,      // Commission charged at the end of a trading month
   EVENT_REASON_ACCOUNT_COMISSION_AGENT_DAILY   =  23,      // Agent commission charged at the end of a trading day
   EVENT_REASON_ACCOUNT_COMISSION_AGENT_MONTHLY =  24,      // Agent commission charged at the end of a month
   EVENT_REASON_ACCOUNT_INTEREST                =  25,      // Accruing interest on free funds
   EVENT_REASON_BUY_CANCELLED                   =  26,      // Canceled buy deal
   EVENT_REASON_SELL_CANCELLED                  =  27,      // Canceled sell deal
   EVENT_REASON_DIVIDENT                        =  28,      // Accruing dividends
   EVENT_REASON_DIVIDENT_FRANKED                =  29,      // Accruing franked dividends
   EVENT_REASON_TAX                             =  30       // Tax
  };
#define REASON_EVENT_SHIFT    (EVENT_REASON_ACCOUNT_CREDIT-3)
//+------------------------------------------------------------------+
//| Event's integer properties                                       |
//+------------------------------------------------------------------+
enum ENUM_EVENT_PROP_INTEGER
  {
   EVENT_PROP_TYPE_EVENT = 0,                               // Account trading event type (from the ENUM_TRADE_EVENT enumeration)
   EVENT_PROP_TIME_EVENT,                                   // Event time in milliseconds
   EVENT_PROP_STATUS_EVENT,                                 // Event status (from the ENUM_EVENT_STATUS enumeration)
   EVENT_PROP_REASON_EVENT,                                 // Event reason (from the ENUM_EVENT_REASON enumeration)
   EVENT_PROP_TYPE_DEAL_EVENT,                              // Deal event type
   EVENT_PROP_TICKET_DEAL_EVENT,                            // Deal event ticket
   EVENT_PROP_TYPE_ORDER_EVENT,                             // Type of an order, based on which a deal event is opened (the last position order)
   EVENT_PROP_TICKET_ORDER_EVENT,                           // Ticket of an order, based on which a deal event is opened (the last position order)
   EVENT_PROP_TIME_ORDER_POSITION,                          // Time of an order, based on which a position deal is opened (the first position order)
   EVENT_PROP_TYPE_ORDER_POSITION,                          // Type of an order, based on which a position deal is opened (the first position order)
   EVENT_PROP_TICKET_ORDER_POSITION,                        // Ticket of an order, based on which a position deal is opened (the first position order)
   EVENT_PROP_POSITION_ID,                                  // Position ID
   EVENT_PROP_POSITION_BY_ID,                               // Opposite position ID
   EVENT_PROP_MAGIC_ORDER,                                  // Order/deal/position magic number
  };
#define EVENT_PROP_INTEGER_TOTAL (14)                       // Total number of integer event properties
//+------------------------------------------------------------------+
//| Event's real properties                                          |
//+------------------------------------------------------------------+
enum ENUM_EVENT_PROP_DOUBLE
  {
   EVENT_PROP_PRICE_EVENT = (EVENT_PROP_INTEGER_TOTAL),     // Price an event occurred at
   EVENT_PROP_PRICE_OPEN,                                   // Order/deal/position open price
   EVENT_PROP_PRICE_CLOSE,                                  // Order/deal/position close price
   EVENT_PROP_PRICE_SL,                                     // StopLoss order/deal/position price
   EVENT_PROP_PRICE_TP,                                     // TakeProfit Order/deal/position
   EVENT_PROP_VOLUME_INITIAL,                               // Requested volume
   EVENT_PROP_VOLUME_EXECUTED,                              // Executed volume
   EVENT_PROP_VOLUME_CURRENT,                               // Remaining volume
   EVENT_PROP_PROFIT                                        // Profit
  };
#define EVENT_PROP_DOUBLE_TOTAL  (9)                        // Total number of event's real properties
//+------------------------------------------------------------------+
//| Event's string properties                                        |
//+------------------------------------------------------------------+
enum ENUM_EVENT_PROP_STRING
  {
   EVENT_PROP_SYMBOL = (EVENT_PROP_INTEGER_TOTAL+EVENT_PROP_DOUBLE_TOTAL), // Order symbol
  };
#define EVENT_PROP_STRING_TOTAL     (1)                     // Total number of event's string properties
//+------------------------------------------------------------------+
//| Possible event sorting criteria                                  |
//+------------------------------------------------------------------+
#define FIRST_EVN_DBL_PROP       (EVENT_PROP_INTEGER_TOTAL)
#define FIRST_EVN_STR_PROP       (EVENT_PROP_INTEGER_TOTAL+EVENT_PROP_DOUBLE_TOTAL)
enum ENUM_SORT_EVENTS_MODE
  {
   //--- Sort by integer properties
   SORT_BY_EVENT_TYPE_EVENT            = 0,                    // Sort by event type
   SORT_BY_EVENT_TIME_EVENT            = 1,                    // Sort by event time
   SORT_BY_EVENT_STATUS_EVENT          = 2,                    // Sort by event status (from the ENUM_EVENT_STATUS enumeration)
   SORT_BY_EVENT_REASON_EVENT          = 3,                    // Sort by event reason (from the ENUM_EVENT_REASON enumeration)
   SORT_BY_EVENT_TYPE_DEAL_EVENT       = 4,                    // Sort by deal event type
   SORT_BY_EVENT_TICKET_DEAL_EVENT     = 5,                    // Sort by deal event ticket
   SORT_BY_EVENT_TYPE_ORDER_EVENT      = 6,                    // Sort by type of an order, based on which a deal event is opened (the last position order)
   SORT_BY_EVENT_TYPE_ORDER_POSITION   = 7,                    // Sort by type of an order, based on which a position deal is opened (the first position order)
   SORT_BY_EVENT_TICKET_ORDER_EVENT    = 8,                    // Sort by a ticket of an order, based on which a deal event is opened (the last position order)
   SORT_BY_EVENT_TICKET_ORDER_POSITION = 9,                    // Sort by a ticket of an order, based on which a position deal is opened (the first position order)
   SORT_BY_EVENT_POSITION_ID           = 10,                   // Sort by position ID
   SORT_BY_EVENT_POSITION_BY_ID        = 11,                   // Sort by opposite position ID
   SORT_BY_EVENT_MAGIC_ORDER           = 12,                   // Sort by order/deal/position magic number
   SORT_BY_EVENT_TIME_ORDER_POSITION   = 13,                   // Sort by time of an order, based on which a position deal is opened (the first position order)
   //--- Sort by real properties
   SORT_BY_EVENT_PRICE_EVENT        =  FIRST_EVN_DBL_PROP,     // Sort by a price an event occurred at
   SORT_BY_EVENT_PRICE_OPEN         =  FIRST_EVN_DBL_PROP+1,   // Sort by position open price
   SORT_BY_EVENT_PRICE_CLOSE        =  FIRST_EVN_DBL_PROP+2,   // Sort by position close price
   SORT_BY_EVENT_PRICE_SL           =  FIRST_EVN_DBL_PROP+3,   // Sort by position's StopLoss price
   SORT_BY_EVENT_PRICE_TP           =  FIRST_EVN_DBL_PROP+4,   // Sort by position's TakeProfit price
   SORT_BY_EVENT_VOLUME_INITIAL     =  FIRST_EVN_DBL_PROP+5,   // Sort by initial volume
   SORT_BY_EVENT_VOLUME             =  FIRST_EVN_DBL_PROP+6,   // Sort by the current volume
   SORT_BY_EVENT_VOLUME_CURRENT     =  FIRST_EVN_DBL_PROP+7,   // Sort by remaining volume
   SORT_BY_EVENT_PROFIT             =  FIRST_EVN_DBL_PROP+8,   // Sort by profit
   //--- Sort by string properties
   SORT_BY_EVENT_SYMBOL             =  FIRST_EVN_STR_PROP      // Sort by order/position/deal symbol
  };
//+------------------------------------------------------------------+
```

Here we have all possible event object states (similar to order states described in the [first \\
article](https://www.mql5.com/en/articles/5654)), event occurrence reasons, all event properties and criteria for sorting events to search by properties. All of this is already
familiar from the previous articles. We can always

[get back to the beginning](https://www.mql5.com/en/articles/5654) and refresh data for clarification.

Apart from the event status providing a general event data, the event reason (ENUM\_EVENT\_REASON) already contains all details on a particular
event origin.

For example, if an event has a market position status (EVENT\_STATUS\_MARKET\_POSITION), then the event occurrence reason is specified
in the EVENT\_PROP\_REASON\_EVENT object field. It can either be a pending order activation (EVENT\_REASON\_ACTIVATED\_PENDING), or opening
a position by a market order (EVENT\_REASON\_DONE). Besides, the following nuances are also considered: if a position is opened partially
(not the entire pending or market order volume was executed), the event reason is EVENT\_REASON\_ACTIVATED\_PENDING\_PARTIALLY or
EVENT\_REASON\_DONE\_PARTIALLY etc.

Thus, an event object contains the entire data on the event and an order that triggered it. Besides, historical events provide data on two
orders — the first position order and closing position order.

Thus, data on orders, deals and position itself in the event object allows us to track the entire chain of position events within the
entire history of its existence — from opening to closing.

The ENUM\_EVENT\_REASON enumeration constants are located and numbered so that when the event status is "deal", the deal type falls at the ENUM\_DEAL\_TYPE
enumeration in case the deal type exceeds DEAL\_TYPE\_SELL. Thus, we end up with the balance operations types. The description of the balance
operation is sent to the event reason when defining the deal type in the class prepared for creation.

The shift to be added to the deal type is calculated in the #define
REASON\_EVENT\_SHIFT macro substitution. It is needed to set the balance operation type in the ENUM\_EVENT\_REASON enumeration.

Let's add the functions returning descriptions of order, position
and deal types, as well as the function returning
the position name depending on the type of an order opening it. All functions are added to the **DELib.mqh** file located
in the

**Services** library. This allows for convenient output of orders, positions and deals.

```
//+------------------------------------------------------------------+
//| Return the order name                                            |
//+------------------------------------------------------------------+
string OrderTypeDescription(const ENUM_ORDER_TYPE type)
  {
   string pref=(#ifdef __MQL5__ "Market order" #else "Position" #endif );
   return
     (
      type==ORDER_TYPE_BUY_LIMIT       ?  "Buy Limit"                                                 :
      type==ORDER_TYPE_BUY_STOP        ?  "Buy Stop"                                                  :
      type==ORDER_TYPE_SELL_LIMIT      ?  "Sell Limit"                                                :
      type==ORDER_TYPE_SELL_STOP       ?  "Sell Stop"                                                 :
   #ifdef __MQL5__
      type==ORDER_TYPE_BUY_STOP_LIMIT  ?  "Buy Stop Limit"                                            :
      type==ORDER_TYPE_SELL_STOP_LIMIT ?  "Sell Stop Limit"                                           :
      type==ORDER_TYPE_CLOSE_BY        ?  TextByLanguage("Закрывающий ордер","Order for closing by")  :
   #else
      type==ORDER_TYPE_BALANCE         ?  TextByLanguage("Балансовая операция","Balance operation")   :
      type==ORDER_TYPE_CREDIT          ?  TextByLanguage("Кредитная операция","Credit operation")     :
   #endif
      type==ORDER_TYPE_BUY             ?  pref+" Buy"                                                 :
      type==ORDER_TYPE_SELL            ?  pref+" Sell"                                                :
      TextByLanguage("Неизвестный тип ордера","Unknown order type")
     );
  }
//+------------------------------------------------------------------+
//| Return the position name                                         |
//+------------------------------------------------------------------+
string PositionTypeDescription(const ENUM_POSITION_TYPE type)
  {
   return
     (
      type==POSITION_TYPE_BUY    ? "Buy"  :
      type==POSITION_TYPE_SELL   ? "Sell" :
      TextByLanguage("Неизвестный тип позиции","Unknown position type")
     );
  }
//+------------------------------------------------------------------+
//| Return the deal name                                             |
//+------------------------------------------------------------------+
string DealTypeDescription(const ENUM_DEAL_TYPE type)
  {
   return
     (
      type==DEAL_TYPE_BUY                       ?  TextByLanguage("Сделка на покупку","Buy deal") :
      type==DEAL_TYPE_SELL                      ?  TextByLanguage("Сделка на продажу","Sell deal") :
      type==DEAL_TYPE_BALANCE                   ?  TextByLanguage("Балансовая операция","Balance operation") :
      type==DEAL_TYPE_CREDIT                    ?  TextByLanguage("Начисление кредита","Credit") :
      type==DEAL_TYPE_CHARGE                    ?  TextByLanguage("Дополнительные сборы","Additional charge") :
      type==DEAL_TYPE_CORRECTION                ?  TextByLanguage("Корректирующая запись","Correction") :
      type==DEAL_TYPE_BONUS                     ?  TextByLanguage("Перечисление бонусов","Bonus") :
      type==DEAL_TYPE_COMMISSION                ?  TextByLanguage("Дополнительные комиссии","Additional comissions") :
      type==DEAL_TYPE_COMMISSION_DAILY          ?  TextByLanguage("Комиссия, начисляемая в конце торгового дня","Daily commission") :
      type==DEAL_TYPE_COMMISSION_MONTHLY        ?  TextByLanguage("Комиссия, начисляемая в конце месяца","Monthly commission") :
      type==DEAL_TYPE_COMMISSION_AGENT_DAILY    ?  TextByLanguage("Агентская комиссия, начисляемая в конце торгового дня","Daily agent commission") :
      type==DEAL_TYPE_COMMISSION_AGENT_MONTHLY  ?  TextByLanguage("Агентская комиссия, начисляемая в конце месяца","Monthly agent commission") :
      type==DEAL_TYPE_INTEREST                  ?  TextByLanguage("Начисления процентов на свободные средства","Agency commission charged at the end of month") :
      type==DEAL_TYPE_BUY_CANCELED              ?  TextByLanguage("Отмененная сделка покупки","Canceled buy transaction") :
      type==DEAL_TYPE_SELL_CANCELED             ?  TextByLanguage("Отмененная сделка продажи","Canceled sell transaction") :
      type==DEAL_DIVIDEND                       ?  TextByLanguage("Начисление дивиденда","Dividend operations") :
      type==DEAL_DIVIDEND_FRANKED               ?  TextByLanguage("Начисление франкированного дивиденда","Franked (non-taxable) dividend operations") :
      type==DEAL_TAX                            ?  TextByLanguage("Начисление налога","Tax charges") :
      TextByLanguage("Неизвестный тип сделки","Unknown deal type")
     );
  }
//+------------------------------------------------------------------+
//| Return the position type by the order type                       |
//+------------------------------------------------------------------+
ENUM_POSITION_TYPE PositionTypeByOrderType(ENUM_ORDER_TYPE type_order)
  {
   if(
      type_order==ORDER_TYPE_BUY             ||
      type_order==ORDER_TYPE_BUY_LIMIT       ||
      type_order==ORDER_TYPE_BUY_STOP
   #ifdef __MQL5__                           ||
      type_order==ORDER_TYPE_BUY_STOP_LIMIT
   #endif
     ) return POSITION_TYPE_BUY;
   else if(
      type_order==ORDER_TYPE_SELL            ||
      type_order==ORDER_TYPE_SELL_LIMIT      ||
      type_order==ORDER_TYPE_SELL_STOP
   #ifdef __MQL5__                           ||
      type_order==ORDER_TYPE_SELL_STOP_LIMIT
   #endif
     ) return POSITION_TYPE_SELL;
   return WRONG_VALUE;
  }
//+------------------------------------------------------------------+
```

When testing the event collection class, a very unpleasant problem was discovered: when creating lists of orders and deals in the terminal
using

[HistorySelect()](https://www.mql5.com/en/docs/trading/historyselect) and the subsequent access to the new elements of the
lists, I found out that the orders are listed not by the order of event occurrence but by their placement time. Let me explain:

1. open a position,

2. place a pending order right away,

3. close part of a position,
4. wait till a pending order is activated

The sequence of the events in history is expected to be as follows:

opening a position, placing an order, partial
closure, order activation — in order of conducting operations over time. But it turned out that the sequence of events in the common order and
deal history is as follows:

1. opening a position
2. placing an order
3. order activation
4. partial closure

In other words, the histories of orders and deals live their own lives within the terminal and do not correlate with each other, which is
also reasonable since these are two lists with their own histories.

The class of collections of orders and deals is made in such a way that when changing any of the lists (orders or deals), the last event on the
account is read in order not to constantly scan the history, which is very expensive. But considering the above and when conducting
trade operations, we do not track the sequence of actions. We simply place an order and wait for its activation. After opening a
position, we work exclusively with it. In that case, all events are to be located in the required order allowing for their tracking.
However, this is insufficient. We need to work in any sequence, and the program should be able to find the right event and point at it
accurately.

Based on the above, I have improved the historical orders and events collection class. Now if an out-of-order event appears, the class
finds the necessary order, creates its object and puts it to the list to be the last one, so that the events collection class is always able
to accurately define the last occurred event.

To implement the feature, let's add three new methods to the private section of the historical orders and deals collection class:

```
//--- Return the flag of the order object by its type and ticket in the list of historical orders and deals
   bool              IsPresentOrderInList(const ulong order_ticket,const ENUM_ORDER_TYPE type);
//--- Return the "lost" order type and ticket
   ulong             OrderSearch(const int start,ENUM_ORDER_TYPE &order_type);
//--- Create the order object and place it to the list
   bool              CreateNewOrder(const ulong order_ticket,const ENUM_ORDER_TYPE order_type);
```

and their implementation beyond the class body as well.

**The method returning the flag of the order object presence in the list by its ticket and type:**

```
//+-----------------------------------------------------------------------------+
//| Return the flag of the order object presence in the list by type and ticket |
//+-----------------------------------------------------------------------------+
bool CHistoryCollection::IsPresentOrderInList(const ulong order_ticket,const ENUM_ORDER_TYPE type)
  {
   CArrayObj* list=dynamic_cast<CListObj*>(&this.m_list_all_orders);
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,type,EQUAL);
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TICKET,order_ticket,EQUAL);
   return(list.Total()>0);
  }
//+------------------------------------------------------------------+
```

Create the pointer to the list using the dynamic typecasting
(send the CArrayObj list to the CSelect class, while collection lists are of CListObj type and are inherited from CArrayObj)

Leave
only ordershaving the type passed to the method by the input.

Leave only orderhaving
the ticket passed to the method by the input.

If such an order exists
(the list is greater than zero), return

true.

**The method returning a type and a method of an order, which is not the last one in the terminal list, but is not in the collection list:**

```
//+------------------------------------------------------------------+
//| Return the "lost" order's type and ticket                        |
//+------------------------------------------------------------------+
ulong CHistoryCollection::OrderSearch(const int start,ENUM_ORDER_TYPE &order_type)
  {
   ulong order_ticket=0;
   for(int i=start-1;i>=0;i--)
     {
      ulong ticket=::HistoryOrderGetTicket(i);
      if(ticket==0)
         continue;
      ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)::HistoryOrderGetInteger(ticket,ORDER_TYPE);
      if(this.IsPresentOrderInList(ticket,type))
         continue;
      order_ticket=ticket;
      order_type=type;
     }
   return order_ticket;
  }
//+------------------------------------------------------------------+
```

The index of the last order is passed to the terminal order list.
Since the index specifies the order already present in the collection, the

search loop should be started from the previous order in the list
(start-1).

Since the necessary order is usually located near the
end of the list, search for an order with a ticket and a type absent in the collection in the loop from the list end using the
IsPresentOrderInList() method.

If the order is present in the collection, check the next one.
As soon as there is an order missing in the collection,

write its ticket and type
and return them to the calling program. The ticket is returned by the
method result, while the type is returned in the variable via the
link.

Since we now need to create order objects in several places within the class (when defining a new order and when looking for a "lost" one), let's
make a separate

**method creating an order object and placing it in the collection list:**

```
//+------------------------------------------------------------------+
//| Create an order object and place it to the list                  |
//+------------------------------------------------------------------+
bool CHistoryCollection::CreateNewOrder(const ulong order_ticket,const ENUM_ORDER_TYPE order_type)
  {
   COrder* order=NULL;
   if(order_type==ORDER_TYPE_BUY)
     {
      order=new CHistoryOrder(order_ticket);
      if(order==NULL)
         return false;
     }
   else if(order_type==ORDER_TYPE_BUY_LIMIT)
     {
      order=new CHistoryPending(order_ticket);
      if(order==NULL)
         return false;
     }
   else if(order_type==ORDER_TYPE_BUY_STOP)
     {
      order=new CHistoryPending(order_ticket);
      if(order==NULL)
         return false;
     }
   else if(order_type==ORDER_TYPE_SELL)
     {
      order=new CHistoryOrder(order_ticket);
      if(order==NULL)
         return false;
     }
   else if(order_type==ORDER_TYPE_SELL_LIMIT)
     {
      order=new CHistoryPending(order_ticket);
      if(order==NULL)
         return false;
     }
   else if(order_type==ORDER_TYPE_SELL_STOP)
     {
      order=new CHistoryPending(order_ticket);
      if(order==NULL)
         return false;
     }
#ifdef __MQL5__
   else if(order_type==ORDER_TYPE_BUY_STOP_LIMIT)
     {
      order=new CHistoryPending(order_ticket);
      if(order==NULL)
         return false;
     }
   else if(order_type==ORDER_TYPE_SELL_STOP_LIMIT)
     {
      order=new CHistoryPending(order_ticket);
      if(order==NULL)
         return false;
     }
   else if(order_type==ORDER_TYPE_CLOSE_BY)
     {
      order=new CHistoryOrder(order_ticket);
      if(order==NULL)
         return false;
     }
#endif
   if(this.m_list_all_orders.InsertSort(order))
      return true;
   else
     {
      delete order;
      return false;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

Here all is simple and clear: the method receives an order ticket and type, and a new order object is created depending on the order type. If the
object could not be created,

false is returned immediately. If the object created successfully, it is placed to the
collection and

true is returned. If it could not be placed to the collection, a newly created object is
removed and

false is returned.

Let's change the **Refresh() collection class method**, since the "loss" of a necessary order should be processed:

```
//+------------------------------------------------------------------+
//| Update the list of orders and deals                              |
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
         if(!this.m_list_all_orders.InsertSort(order))
           {
            ::Print(DFUN,TextByLanguage("Не удалось добавить ордер в список","Failed to add order to list"));
            delete order;
           }
        }
      else
        {
         //--- Removed pending orders
         CHistoryPending *order=new CHistoryPending(::OrderTicket());
         if(order==NULL) continue;
         if(!this.m_list_all_orders.InsertSort(order))this.m_list_all_orders.Type()
           {
            ::Print(DFUN,TextByLanguage("Не удалось добавить ордер в список","Failed to add order to list"));
            delete order;
           }
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
      if(type==ORDER_TYPE_BUY || type==ORDER_TYPE_SELL || type==ORDER_TYPE_CLOSE_BY)
        {
         //--- If there is no order of this type and with this ticket in the list, create an order object and add it to the list
         if(!this.IsPresentOrderInList(order_ticket,type))
           {
            if(!this.CreateNewOrder(order_ticket,type))
               ::Print(DFUN,TextByLanguage("Не удалось добавить ордер в список","Could not add order to list"));
           }
         //--- Such an order is already present in the list, which means the necessary order is not the last one in the history list. Let's find it
         else
           {
            ENUM_ORDER_TYPE type_lost=WRONG_VALUE;
            ulong ticket_lost=this.OrderSearch(i,type_lost);
            if(ticket_lost>0 && !this.CreateNewOrder(ticket_lost,type_lost))
               ::Print(DFUN,TextByLanguage("Не удалось добавить ордер в список","Could not add order to list"));
           }
        }
      else
        {
         //--- If there is no pending order of this type and with this ticket in the list, create an order object and add it to the list
         if(!this.IsPresentOrderInList(order_ticket,type))
           {
            if(!this.CreateNewOrder(order_ticket,type))
               ::Print(DFUN,TextByLanguage("Не удалось добавить ордер в список","Could not add order to list"));
           }
         //--- Such an order is already present in the list, which means the necessary order is not the last one in the history list. Let's find it
         else
           {
            ENUM_ORDER_TYPE type_lost=WRONG_VALUE;
            ulong ticket_lost=this.OrderSearch(i,type_lost);
            if(ticket_lost>0 && !this.CreateNewOrder(ticket_lost,type_lost))
               ::Print(DFUN,TextByLanguage("Не удалось добавить ордер в список","Could not add order to list"));
           }
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
      if(!this.m_list_all_orders.InsertSort(deal))
        {
         ::Print(DFUN,TextByLanguage("Не удалось добавить сделку в список","Could not add deal to list"));
         delete deal;
        }
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

The block of handling new orders for MQL5 has been changed in
the method. All implemented changes are described by comments and

highlighted in the listing text.

Let's add the method definition to search for similar orders in the public section of the **COrder** class:

```
//--- Compare COrder by all properties (to search for equal event objects)
   bool              IsEqual(COrder* compared_order) const;
```

and its implementation beyond the class body as well:

```
//+------------------------------------------------------------------+
//| Compare COrder objects by all properties                         |
//+------------------------------------------------------------------+
bool COrder::IsEqual(COrder *compared_order) const
  {
   int beg=0, end=ORDER_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_ORDER_PROP_INTEGER prop=(ENUM_ORDER_PROP_INTEGER)i;
      if(this.GetProperty(prop)!=compared_order.GetProperty(prop)) return false;
     }
   beg=end; end+=ORDER_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_ORDER_PROP_DOUBLE prop=(ENUM_ORDER_PROP_DOUBLE)i;
      if(this.GetProperty(prop)!=compared_order.GetProperty(prop)) return false;
     }
   beg=end; end+=ORDER_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_ORDER_PROP_STRING prop=(ENUM_ORDER_PROP_STRING)i;
      if(this.GetProperty(prop)!=compared_order.GetProperty(prop)) return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

The method goes through all properties of the current order object and the compared
order passed to the method by the pointer in a loop.

As soon as any of the properties of the current order not equal to the same
property of the compared order is detected,

false is returned meaning the orders are not equal.

### Event classes

The preparatory stage is complete. Let's start creating classes of event objects.

We will do exactly the same as when creating order classes. We will develop a basic event class and five descendant classes described by their
states:

- position open event,

- position close event,

- pending order placement event,

- pending order removal event,

- balance operation event

In the previously created Events folder of the Objects library directory, create a new CEvent class inherited from the CObject base class.
In the newly created class template, set the necessary inclusions of the service functions file, order collection classes, as well as
private and protected class members and methods:

```
//+------------------------------------------------------------------+
//|                                                        Event.mqh |
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
#include "\..\..\Services\DELib.mqh"
#include "..\..\Collections\HistoryCollection.mqh"
#include "..\..\Collections\MarketCollection.mqh"
//+------------------------------------------------------------------+
//| Abstract event class                                             |
//+------------------------------------------------------------------+
class CEvent : public CObject
  {
private:
   int               m_event_code;                                   // Event code
//--- Return the index of the array the event's (1) double and (2) string properties are located at
   int               IndexProp(ENUM_EVENT_PROP_DOUBLE property)const { return(int)property-EVENT_PROP_INTEGER_TOTAL;                         }
   int               IndexProp(ENUM_EVENT_PROP_STRING property)const { return(int)property-EVENT_PROP_INTEGER_TOTAL-EVENT_PROP_DOUBLE_TOTAL; }
protected:
   ENUM_TRADE_EVENT  m_trade_event;                                  // Trading event
   long              m_chart_id;                                     // Control program chart ID
   int               m_digits_acc;                                   // Number of decimal places for the account currency
   long              m_long_prop[EVENT_PROP_INTEGER_TOTAL];          // Event integer properties
   double            m_double_prop[EVENT_PROP_DOUBLE_TOTAL];         // Event real properties
   string            m_string_prop[EVENT_PROP_STRING_TOTAL];         // Event string properties
//--- return the flag presence in the trading event
   bool              IsPresentEventFlag(const int event_code)  const { return (this.m_event_code & event_code)==event_code;            }

   //--- Protected parametric constructor
                     CEvent(const ENUM_EVENT_STATUS event_status,const int event_code,const ulong ticket);
public:
//--- Default constructor
                     CEvent(void){;}

//--- Set event's (1) integer, (2) real and (3) string properties
   void              SetProperty(ENUM_EVENT_PROP_INTEGER property,long value) { this.m_long_prop[property]=value;                      }
   void              SetProperty(ENUM_EVENT_PROP_DOUBLE property,double value){ this.m_double_prop[this.IndexProp(property)]=value;    }
   void              SetProperty(ENUM_EVENT_PROP_STRING property,string value){ this.m_string_prop[this.IndexProp(property)]=value;    }
//--- Return the event's (1) integer, (2) real and (3) string properties from the property array
   long              GetProperty(ENUM_EVENT_PROP_INTEGER property)      const { return this.m_long_prop[property];                     }
   double            GetProperty(ENUM_EVENT_PROP_DOUBLE property)       const { return this.m_double_prop[this.IndexProp(property)];   }
   string            GetProperty(ENUM_EVENT_PROP_STRING property)       const { return this.m_string_prop[this.IndexProp(property)];   }

//--- Return the flag of the event supporting the property
   virtual bool      SupportProperty(ENUM_EVENT_PROP_INTEGER property)        { return true; }
   virtual bool      SupportProperty(ENUM_EVENT_PROP_DOUBLE property)         { return true; }
   virtual bool      SupportProperty(ENUM_EVENT_PROP_STRING property)         { return true; }

//--- Set the control program chart ID
   void              SetChartID(const long id)                                { this.m_chart_id=id;                                    }
//--- Decode the event code and set the trading event, (2) return the trading event
   void              SetTypeEvent(void);
   ENUM_TRADE_EVENT  TradeEvent(void)                                   const { return this.m_trade_event;                             }
//--- Send the event to the chart (implementation in descendant classes)
   virtual void      SendEvent(void) {;}

//--- Compare CEvent objects by a specified property (to sort the lists by a specified event object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CEvent objects by all properties (to search for equal event objects)
   bool              IsEqual(CEvent* compared_event) const;
//+------------------------------------------------------------------+
//| Methods of simplified access to event object properties          |
//+------------------------------------------------------------------+
//--- Return (1) event type, (2) event time in milliseconds, (3) event status, (4) event reason, (5) deal type, (6) deal ticket,
//--- (7) order type, based on which a deal was executed, (8) position opening order type, (9) position last order ticket,
//--- (10) position first order ticket, (11) position ID, (12) opposite position ID, (13) magic number, (14) position open time

   ENUM_TRADE_EVENT  TypeEvent(void)                                    const { return (ENUM_TRADE_EVENT)this.GetProperty(EVENT_PROP_TYPE_EVENT);     }
   long              TimeEvent(void)                                    const { return this.GetProperty(EVENT_PROP_TIME_EVENT);                       }
   ENUM_EVENT_STATUS Status(void)                                       const { return (ENUM_EVENT_STATUS)this.GetProperty(EVENT_PROP_STATUS_EVENT);  }
   ENUM_EVENT_REASON Reason(void)                                       const { return (ENUM_EVENT_REASON)this.GetProperty(EVENT_PROP_REASON_EVENT);  }
   long              TypeDeal(void)                                     const { return this.GetProperty(EVENT_PROP_TYPE_DEAL_EVENT);                  }
   long              TicketDeal(void)                                   const { return this.GetProperty(EVENT_PROP_TICKET_DEAL_EVENT);                }
   long              TypeOrderEvent(void)                               const { return this.GetProperty(EVENT_PROP_TYPE_ORDER_EVENT);                 }
   long              TypeOrderPosition(void)                            const { return this.GetProperty(EVENT_PROP_TYPE_ORDER_POSITION);              }
   long              TicketOrderEvent(void)                             const { return this.GetProperty(EVENT_PROP_TICKET_ORDER_EVENT);               }
   long              TicketOrderPosition(void)                          const { return this.GetProperty(EVENT_PROP_TICKET_ORDER_POSITION);            }
   long              PositionID(void)                                   const { return this.GetProperty(EVENT_PROP_POSITION_ID);                      }
   long              PositionByID(void)                                 const { return this.GetProperty(EVENT_PROP_POSITION_BY_ID);                   }
   long              Magic(void)                                        const { return this.GetProperty(EVENT_PROP_MAGIC_ORDER);                      }
   long              TimePosition(void)                                 const { return this.GetProperty(EVENT_PROP_TIME_ORDER_POSITION);              }

//--- Return (1) the price the event occurred at, (2) open price, (3) close price,
//--- (4) StopLoss price, (5) TakeProfit price, (6) profit, (7) requested volume, (8), executed volume, (9) remaining volume
   double            PriceEvent(void)                                   const { return this.GetProperty(EVENT_PROP_PRICE_EVENT);                      }
   double            PriceOpen(void)                                    const { return this.GetProperty(EVENT_PROP_PRICE_OPEN);                       }
   double            PriceClose(void)                                   const { return this.GetProperty(EVENT_PROP_PRICE_CLOSE);                      }
   double            PriceStopLoss(void)                                const { return this.GetProperty(EVENT_PROP_PRICE_SL);                         }
   double            PriceTakeProfit(void)                              const { return this.GetProperty(EVENT_PROP_PRICE_TP);                         }
   double            Profit(void)                                       const { return this.GetProperty(EVENT_PROP_PROFIT);                           }
   double            VolumeInitial(void)                                const { return this.GetProperty(EVENT_PROP_VOLUME_INITIAL);                   }
   double            VolumeExecuted(void)                               const { return this.GetProperty(EVENT_PROP_VOLUME_EXECUTED);                  }
   double            VolumeCurrent(void)                                const { return this.GetProperty(EVENT_PROP_VOLUME_CURRENT);                   }

//--- Return a symbol
   string            Symbol(void)                                       const { return this.GetProperty(EVENT_PROP_SYMBOL);                           }

//+------------------------------------------------------------------+
//| Descriptions of order object properties                          |
//+------------------------------------------------------------------+
//--- Return description of the order's (1) integer, (2) real and (3) string properties
   string            GetPropertyDescription(ENUM_EVENT_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_EVENT_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_EVENT_PROP_STRING property);
//--- Return the event's (1) status and (2) type
   string            StatusDescription(void)          const;
   string            TypeEventDescription(void)       const;
//--- Return the name of an (1) order/position/deal, (2) parent order, (3) position
   string            TypeOrderDescription(void)       const;
   string            TypeOrderBasedDescription(void)  const;
   string            TypePositionDescription(void)    const;
//--- Return the name of the deal/order/position reason
   string            ReasonDescription(void)          const;

//--- Display (1) description of order properties (full_prop=true - all properties, false - only supported ones),
//--- (2) short event message (implementation in the class descendants) in the journal
   void              Print(const bool full_prop=false);
   virtual void      PrintShort(void) {;}
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CEvent::CEvent(const ENUM_EVENT_STATUS event_status,const int event_code,const ulong ticket) : m_event_code(event_code)
  {
   this.m_long_prop[EVENT_PROP_STATUS_EVENT]       =  event_status;
   this.m_long_prop[EVENT_PROP_TICKET_ORDER_EVENT] =  (long)ticket;
   this.m_digits_acc=(int)::AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS);
   this.m_chart_id=::ChartID();
  }
//+------------------------------------------------------------------+
```

The constructor receives the event
status, trading event code and order
or deal ticket which triggered the event.

Here almost all is similar to a protected constructor of the [previously \\
considered COrder class](https://www.mql5.com/en/articles/5654#node04) in the first part of the library description.

The difference is that only two event properties are filled in the protected
class constructor. These are event status and ticket of an order/deal that triggered the event. The event type is detected and saved
in the SetTypeEvent() class method based on the event code passed to the constructor. All other event properties are detected from the
status of orders and deals involved in the event and set by the corresponding class methods separately. This is done because events are to be
detected in the events collection class featuring the method setting all properties for a newly created event.

We considered the event code ( **m\_event\_code**), as well as
its filling and interpretation in the

[fourth part of the library description](https://www.mql5.com/en/articles/5724). We moved it here from the CEngine class
since it was placed to the library base class on a temporary basis to check working with events. Now it will be calculated in the events
collection class and passed to the class constructor when creating an event object.

The trading event ( **m\_trade\_event**) itself is compiled
by decoding the event code in the SetTypeEvent() method, We already described the event code decoding method in the

[fourth article](https://www.mql5.com/en/articles/5724).

We need the control
program chart ID (

**m\_chart\_id**) to send custom messages about events to it.

The
number of decimal places for the account currency (

**m\_digits\_acc**) is necessary for the correct display of messages about events to the journal.

**The methods of comparing the Compare() and IsEqual()**
object event properties are quite simple and transparent. We considered the Compare() method in the first part of the library description. It was
similar to the one of the COrder object. As compared to the first method that compares two objects only by one of the properties, IsEqual()
compares these two objects by all fields. If all fields of the two objects are similar (each property of the current object is equal to the
appropriate property of the compared one), then both objects are identical. The method checks all properties of the two objects in a loop and
returns

false as soon as a discrepancy is detected. There is no point in further checks since one of
the object properties is no longer equal to the same property of the compared object.

```
//+------------------------------------------------------------------+
//| Compare CEvent objects by a specified property                   |
//+------------------------------------------------------------------+
int CEvent::Compare(const CObject *node,const int mode=0) const
  {
   const CEvent *event_compared=node;
//--- compare integer properties of two events
   if(mode<EVENT_PROP_INTEGER_TOTAL)
     {
      long value_compared=event_compared.GetProperty((ENUM_EVENT_PROP_INTEGER)mode);
      long value_current=this.GetProperty((ENUM_EVENT_PROP_INTEGER)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare integer properties of two objects
   if(mode<EVENT_PROP_DOUBLE_TOTAL+EVENT_PROP_INTEGER_TOTAL)
     {
      double value_compared=event_compared.GetProperty((ENUM_EVENT_PROP_DOUBLE)mode);
      double value_current=this.GetProperty((ENUM_EVENT_PROP_DOUBLE)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare string properties of two objects
   else if(mode<EVENT_PROP_DOUBLE_TOTAL+EVENT_PROP_INTEGER_TOTAL+EVENT_PROP_STRING_TOTAL)
     {
      string value_compared=event_compared.GetProperty((ENUM_EVENT_PROP_STRING)mode);
      string value_current=this.GetProperty((ENUM_EVENT_PROP_STRING)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
   return 0;
  }
//+------------------------------------------------------------------+
//| Compare CEvent events by all properties                          |
//+------------------------------------------------------------------+
bool CEvent::IsEqual(CEvent *compared_event) const
  {
   int beg=0, end=EVENT_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_EVENT_PROP_INTEGER prop=(ENUM_EVENT_PROP_INTEGER)i;
      if(this.GetProperty(prop)!=compared_event.GetProperty(prop)) return false;
     }
   beg=end; end+=EVENT_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_EVENT_PROP_DOUBLE prop=(ENUM_EVENT_PROP_DOUBLE)i;
      if(this.GetProperty(prop)!=compared_event.GetProperty(prop)) return false;
     }
   beg=end; end+=EVENT_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_EVENT_PROP_STRING prop=(ENUM_EVENT_PROP_STRING)i;
      if(this.GetProperty(prop)!=compared_event.GetProperty(prop)) return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

**Let's have a closer look at the SetTypeEvent() method**

All necessary checks and actions are set directly in the code comments:

```
//+------------------------------------------------------------------+
//| Decode the event code and set a trading event                    |
//+------------------------------------------------------------------+
void CEvent::SetTypeEvent(void)
  {
//--- Pending order placed (check for matching the event code since there can only be one flag here)
   if(this.m_event_code==TRADE_EVENT_FLAG_ORDER_PLASED)
     {
      this.m_trade_event=TRADE_EVENT_PENDING_ORDER_PLASED;
      this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
      return;
     }
//--- Pending order removed (check for matching the event code since there can only be one flag here)
   if(this.m_event_code==TRADE_EVENT_FLAG_ORDER_REMOVED)
     {
      this.m_trade_event=TRADE_EVENT_PENDING_ORDER_REMOVED;
      this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
      return;
     }
//--- Position opened (Check the presence of multiple flags in the event code)
   if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_POSITION_OPENED))
     {
      //--- If the pending order is activated by a price
      if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_ORDER_ACTIVATED))
        {
         //--- check the partial closure flag and set the "pending order activated" or "pending order partially activated" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_PENDING_ORDER_ACTIVATED : TRADE_EVENT_PENDING_ORDER_ACTIVATED_PARTIAL);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
      //--- check the partial opening flag and set the "Position opened" or "Position partially opened" trading event
      this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_OPENED : TRADE_EVENT_POSITION_OPENED_PARTIAL);
      this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
      return;
     }
//--- Position closed (Check the presence of multiple flags in the event code)
   if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_POSITION_CLOSED))
     {
      //--- if a position is closed by StopLoss
      if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_SL))
        {
         //--- check the partial closing flag and set the "Position closed by StopLoss" or "Position partially closed by StopLoss" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED_BY_SL : TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_SL);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
      //--- if a position is closed by TakeProfit
      else if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_TP))
        {
         //--- check the partial closure flag and set the "Position closed by TakeProfit" or "Position partially closed by TakeProfit" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED_BY_TP : TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_TP);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
      //--- if a position is closed by an opposite one
      else if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_BY_POS))
        {
         //--- check the partial closure flag and set the "Position closed by opposite one" or "Position partially closed by opposite one" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED_BY_POS : TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
      //--- If a position is closed
      else
        {
         //--- check the partial closure flag and set the "Position closed" or "Position partially closed" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED : TRADE_EVENT_POSITION_CLOSED_PARTIAL);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
     }
//--- Balance operation on the account (clarify the event by deal type)
   if(this.m_event_code==TRADE_EVENT_FLAG_ACCOUNT_BALANCE)
     {
      //--- Initialize a trading event
      this.m_trade_event=TRADE_EVENT_NO_EVENT;
      //--- Take a deal type
      ENUM_DEAL_TYPE deal_type=(ENUM_DEAL_TYPE)this.GetProperty(EVENT_PROP_TYPE_DEAL_EVENT);
      //--- if a deal is a balance operation
      if(deal_type==DEAL_TYPE_BALANCE)
        {
        //--- check the deal profit and set an event (funds deposit or withdrawal)
         this.m_trade_event=(this.GetProperty(EVENT_PROP_PROFIT)>0 ? TRADE_EVENT_ACCOUNT_BALANCE_REFILL : TRADE_EVENT_ACCOUNT_BALANCE_WITHDRAWAL);
        }
      //--- The remaining balance operation types match the ENUM_DEAL_TYPE enumeration starting from DEAL_TYPE_CREDIT
      else if(deal_type>DEAL_TYPE_BALANCE)
        {
        //--- set the event
         this.m_trade_event=(ENUM_TRADE_EVENT)deal_type;
        }
      this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
      return;
     }
  }
//+------------------------------------------------------------------+
```

Here all is simple: an event code is passed to the method and the event code flags are then checked. If the code has the checked flag, the
appropriate trading event is set. Since the event code may have multiple flags, all possible flags for the event are checked and the event
type is defined from their combination. Next, the event type is added to the appropriate class variable and is entered in the property of the
event object (EVENT\_PROP\_TYPE\_EVENT).

**Let's have a look at the listings of the remaining class methods**:

```
//+------------------------------------------------------------------+
//| Return the description of the event's integer property           |
//+------------------------------------------------------------------+
string CEvent::GetPropertyDescription(ENUM_EVENT_PROP_INTEGER property)
  {
   return
     (
      property==EVENT_PROP_TYPE_EVENT              ?  TextByLanguage("Тип события","Event type")+": "+this.TypeEventDescription()                                                       :
      property==EVENT_PROP_TIME_EVENT              ?  TextByLanguage("Время события","Time of event")+": "+TimeMSCtoString(this.GetProperty(property))                                    :
      property==EVENT_PROP_STATUS_EVENT            ?  TextByLanguage("Статус события","Status of event")+": \""+this.StatusDescription()+"\""                                             :
      property==EVENT_PROP_REASON_EVENT            ?  TextByLanguage("Причина события","Reason of event")+": "+this.ReasonDescription()                                                   :
      property==EVENT_PROP_TYPE_DEAL_EVENT         ?  TextByLanguage("Тип сделки","Deal's type")+": "+DealTypeDescription((ENUM_DEAL_TYPE)this.GetProperty(property))                     :
      property==EVENT_PROP_TICKET_DEAL_EVENT       ?  TextByLanguage("Тикет сделки","Deal's ticket")+" #"+(string)this.GetProperty(property)                                              :
      property==EVENT_PROP_TYPE_ORDER_EVENT        ?  TextByLanguage("Тип ордера события","Event's order type")+": "+OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(property))    :
      property==EVENT_PROP_TYPE_ORDER_POSITION     ?  TextByLanguage("Тип ордера позиции","Position's order type")+": "+OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(property)) :
      property==EVENT_PROP_TICKET_ORDER_POSITION   ?  TextByLanguage("Тикет первого ордера позиции","Position's first order ticket")+" #"+(string)this.GetProperty(property)              :
      property==EVENT_PROP_TICKET_ORDER_EVENT      ?  TextByLanguage("Тикет ордера события","Event's order ticket")+" #"+(string)this.GetProperty(property)                               :
      property==EVENT_PROP_POSITION_ID             ?  TextByLanguage("Идентификатор позиции","Position ID")+" #"+(string)this.GetProperty(property)                                       :
      property==EVENT_PROP_POSITION_BY_ID          ?  TextByLanguage("Идентификатор встречной позиции","Opposite position ID")+" #"+(string)this.GetProperty(property)                  :
      property==EVENT_PROP_MAGIC_ORDER             ?  TextByLanguage("Магический номер","Magic number")+": "+(string)this.GetProperty(property)                                           :
      property==EVENT_PROP_TIME_ORDER_POSITION     ?  TextByLanguage("Время открытия позиции","Position open time")+": "+TimeMSCtoString(this.GetProperty(property))                  :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return the description of the event's real property              |
//+------------------------------------------------------------------+
string CEvent::GetPropertyDescription(ENUM_EVENT_PROP_DOUBLE property)
  {
   int dg=(int)::SymbolInfoInteger(this.GetProperty(EVENT_PROP_SYMBOL),SYMBOL_DIGITS);
   int dgl=(int)DigitsLots(this.GetProperty(EVENT_PROP_SYMBOL));
   return
     (
      property==EVENT_PROP_PRICE_EVENT       ?  TextByLanguage("Цена события","Price at the time of event")+": "+::DoubleToString(this.GetProperty(property),dg) :
      property==EVENT_PROP_PRICE_OPEN        ?  TextByLanguage("Цена открытия","Open price")+": "+::DoubleToString(this.GetProperty(property),dg)                    :
      property==EVENT_PROP_PRICE_CLOSE       ?  TextByLanguage("Цена закрытия","Close price")+": "+::DoubleToString(this.GetProperty(property),dg)                   :
      property==EVENT_PROP_PRICE_SL          ?  TextByLanguage("Цена StopLoss","StopLoss price")+": "+::DoubleToString(this.GetProperty(property),dg)                :
      property==EVENT_PROP_PRICE_TP          ?  TextByLanguage("Цена TakeProfit","TakeProfit price")+": "+::DoubleToString(this.GetProperty(property),dg)            :
      property==EVENT_PROP_VOLUME_INITIAL    ?  TextByLanguage("Начальный объём","Initial volume")+": "+::DoubleToString(this.GetProperty(property),dgl)             :
      property==EVENT_PROP_VOLUME_EXECUTED   ?  TextByLanguage("Исполненный объём","Executed volume")+": "+::DoubleToString(this.GetProperty(property),dgl)          :
      property==EVENT_PROP_VOLUME_CURRENT    ?  TextByLanguage("Оставшийся объём","Remaining volume")+": "+::DoubleToString(this.GetProperty(property),dgl)          :
      property==EVENT_PROP_PROFIT            ?  TextByLanguage("Профит","Profit")+": "+::DoubleToString(this.GetProperty(property),this.m_digits_acc)                :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return the description of the event's string property            |
//+------------------------------------------------------------------+
string CEvent::GetPropertyDescription(ENUM_EVENT_PROP_STRING property)
  {
   return TextByLanguage("Символ","Symbol")+": \""+this.GetProperty(property)+"\"";
  }
//+------------------------------------------------------------------+
//| Return the event status name                                     |
//+------------------------------------------------------------------+
string CEvent::StatusDescription(void) const
  {
   ENUM_EVENT_STATUS status=(ENUM_EVENT_STATUS)this.GetProperty(EVENT_PROP_STATUS_EVENT);
   return
     (
      status==EVENT_STATUS_MARKET_PENDING    ?  TextByLanguage("Установлен отложенный ордер","Pending order placed") :
      status==EVENT_STATUS_MARKET_POSITION   ?  TextByLanguage("Открыта позиция","Position opened")                 :
      status==EVENT_STATUS_HISTORY_PENDING   ?  TextByLanguage("Удален отложенный ордер","Pending order removed")    :
      status==EVENT_STATUS_HISTORY_POSITION  ?  TextByLanguage("Закрыта позиция","Position closed")                  :
      status==EVENT_STATUS_BALANCE           ?  TextByLanguage("Балансная операция","Balance operation")             :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return the trading event name                                    |
//+------------------------------------------------------------------+
string CEvent::TypeEventDescription(void) const
  {
   ENUM_TRADE_EVENT event=this.TypeEvent();
   return
     (
      event==TRADE_EVENT_PENDING_ORDER_PLASED            ?  TextByLanguage("Отложенный ордер установлен","Pending order placed")                                  :
      event==TRADE_EVENT_PENDING_ORDER_REMOVED           ?  TextByLanguage("Отложенный ордер удалён","Pending order removed")                                     :
      event==TRADE_EVENT_ACCOUNT_CREDIT                  ?  TextByLanguage("Начисление кредита","Credit")                                                         :
      event==TRADE_EVENT_ACCOUNT_CHARGE                  ?  TextByLanguage("Дополнительные сборы","Additional charge")                                            :
      event==TRADE_EVENT_ACCOUNT_CORRECTION              ?  TextByLanguage("Корректирующая запись","Correction")                                                  :
      event==TRADE_EVENT_ACCOUNT_BONUS                   ?  TextByLanguage("Перечисление бонусов","Bonus")                                                        :
      event==TRADE_EVENT_ACCOUNT_COMISSION               ?  TextByLanguage("Дополнительные комиссии","Additional commission")                                     :
      event==TRADE_EVENT_ACCOUNT_COMISSION_DAILY         ?  TextByLanguage("Комиссия, начисляемая в конце торгового дня","Daily commission")                      :
      event==TRADE_EVENT_ACCOUNT_COMISSION_MONTHLY       ?  TextByLanguage("Комиссия, начисляемая в конце месяца","Monthly commission")                           :
      event==TRADE_EVENT_ACCOUNT_COMISSION_AGENT_DAILY   ?  TextByLanguage("Агентская комиссия, начисляемая в конце торгового дня","Daily agent commission")      :
      event==TRADE_EVENT_ACCOUNT_COMISSION_AGENT_MONTHLY ?  TextByLanguage("Агентская комиссия, начисляемая в конце месяца","Monthly agent commission")           :
      event==TRADE_EVENT_ACCOUNT_INTEREST                ?  TextByLanguage("Начисления процентов на свободные средства","Interest rate")                          :
      event==TRADE_EVENT_BUY_CANCELLED                   ?  TextByLanguage("Отмененная сделка покупки","Canceled buy deal")                                       :
      event==TRADE_EVENT_SELL_CANCELLED                  ?  TextByLanguage("Отмененная сделка продажи","Canceled sell deal")                                      :
      event==TRADE_EVENT_DIVIDENT                        ?  TextByLanguage("Начисление дивиденда","Dividend operations")                                          :
      event==TRADE_EVENT_DIVIDENT_FRANKED                ?  TextByLanguage("Начисление франкированного дивиденда","Franked (non-taxable) dividend operations")    :
      event==TRADE_EVENT_TAX                             ?  TextByLanguage("Начисление налога","Tax charges")                                                     :
      event==TRADE_EVENT_ACCOUNT_BALANCE_REFILL          ?  TextByLanguage("Пополнение средств на балансе","Balance refill")                                      :
      event==TRADE_EVENT_ACCOUNT_BALANCE_WITHDRAWAL      ?  TextByLanguage("Снятие средств с баланса","Withdrawals")                                              :
      event==TRADE_EVENT_PENDING_ORDER_ACTIVATED         ?  TextByLanguage("Отложенный ордер активирован ценой","Pending order activated")                        :
      event==TRADE_EVENT_PENDING_ORDER_ACTIVATED_PARTIAL ?  TextByLanguage("Отложенный ордер активирован ценой частично","Pending order activated partially")     :
      event==TRADE_EVENT_POSITION_OPENED                 ?  TextByLanguage("Позиция открыта","Position opened")                                                  :
      event==TRADE_EVENT_POSITION_OPENED_PARTIAL         ?  TextByLanguage("Позиция открыта частично","Position opened partially")                               :
      event==TRADE_EVENT_POSITION_CLOSED                 ?  TextByLanguage("Позиция закрыта","Position closed")                                                   :
      event==TRADE_EVENT_POSITION_CLOSED_PARTIAL         ?  TextByLanguage("Позиция закрыта частично","Position closed partially")                                :
      event==TRADE_EVENT_POSITION_CLOSED_BY_POS          ?  TextByLanguage("Позиция закрыта встречной","Position closed by opposite position")                    :
      event==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS  ?  TextByLanguage("Позиция закрыта встречной частично","Position closed partially by opposite position") :
      event==TRADE_EVENT_POSITION_CLOSED_BY_SL           ?  TextByLanguage("Позиция закрыта по StopLoss","Position closed by StopLoss")                           :
      event==TRADE_EVENT_POSITION_CLOSED_BY_TP           ?  TextByLanguage("Позиция закрыта по TakeProfit","Position closed by TakeProfit")                       :
      event==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_SL   ?  TextByLanguage("Позиция закрыта частично по StopLoss","Position closed partially by StopLoss")        :
      event==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_TP   ?  TextByLanguage("Позиция закрыта частично по TakeProfit","Position closed partially by TakeProfit")    :
      event==TRADE_EVENT_POSITION_REVERSED               ?  TextByLanguage("Разворот позиции","Position reversal")                                                :
      event==TRADE_EVENT_POSITION_VOLUME_ADD             ?  TextByLanguage("Добавлен объём к позиции","Added volume to position")                                 :
      TextByLanguage("Нет торгового события","No trade event")
     );
  }
//+------------------------------------------------------------------+
//| Return the name of the order/position/deal                       |
//+------------------------------------------------------------------+
string CEvent::TypeOrderDescription(void) const
  {
   ENUM_EVENT_STATUS status=this.Status();
   return
     (
      status==EVENT_STATUS_MARKET_PENDING  || status==EVENT_STATUS_HISTORY_PENDING  ?  OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(EVENT_PROP_TYPE_ORDER_EVENT))      :
      status==EVENT_STATUS_MARKET_POSITION || status==EVENT_STATUS_HISTORY_POSITION ?  PositionTypeDescription((ENUM_POSITION_TYPE)this.GetProperty(EVENT_PROP_TYPE_DEAL_EVENT)) :
      status==EVENT_STATUS_BALANCE  ?  DealTypeDescription((ENUM_DEAL_TYPE)this.GetProperty(EVENT_PROP_TYPE_DEAL_EVENT))  :  "Unknown"
     );
  }
//+------------------------------------------------------------------+
//| Return the name of the parent order                              |
//+------------------------------------------------------------------+
string CEvent::TypeOrderBasedDescription(void) const
  {
   return OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(EVENT_PROP_TYPE_ORDER_POSITION));
  }
//+------------------------------------------------------------------+
//| Return the position name                                         |
//+------------------------------------------------------------------+
string CEvent::TypePositionDescription(void) const
  {
   ENUM_POSITION_TYPE type=PositionTypeByOrderType((ENUM_ORDER_TYPE)this.GetProperty(EVENT_PROP_TYPE_ORDER_POSITION));
   return PositionTypeDescription(type);
  }
//+------------------------------------------------------------------+
//| Return the name of the deal/order/position reason                |
//+------------------------------------------------------------------+
string CEvent::ReasonDescription(void) const
  {
   ENUM_EVENT_REASON reason=this.Reason();
   return
     (
      reason==EVENT_REASON_ACTIVATED_PENDING                ?  TextByLanguage("Активирован отложенный ордер","Pending order activated")                           :
      reason==EVENT_REASON_ACTIVATED_PENDING_PARTIALLY      ?  TextByLanguage("Частичное срабатывание отложенного ордера","Pending order partially triggered")    :
      reason==EVENT_REASON_CANCEL                           ?  TextByLanguage("Отмена","Canceled")                                                                :
      reason==EVENT_REASON_EXPIRED                          ?  TextByLanguage("Истёк срок действия","Expired")                                                    :
      reason==EVENT_REASON_DONE                             ?  TextByLanguage("Запрос выполнен полностью","Request fully executed")                            :
      reason==EVENT_REASON_DONE_PARTIALLY                   ?  TextByLanguage("Запрос выполнен частично","Request partially executed")                         :
      reason==EVENT_REASON_DONE_SL                          ?  TextByLanguage("закрытие по StopLoss","Close by StopLoss triggered")                               :
      reason==EVENT_REASON_DONE_SL_PARTIALLY                ?  TextByLanguage("Частичное закрытие по StopLoss","Partial close by StopLoss triggered")             :
      reason==EVENT_REASON_DONE_TP                          ?  TextByLanguage("закрытие по TakeProfit","Close by TakeProfit triggered")                           :
      reason==EVENT_REASON_DONE_TP_PARTIALLY                ?  TextByLanguage("Частичное закрытие по TakeProfit","Partial close by TakeProfit triggered")         :
      reason==EVENT_REASON_DONE_BY_POS                      ?  TextByLanguage("Закрытие встречной позицией","Closed by opposite position")                        :
      reason==EVENT_REASON_DONE_PARTIALLY_BY_POS            ?  TextByLanguage("Частичное закрытие встречной позицией","Closed partially by opposite position")    :
      reason==EVENT_REASON_DONE_BY_POS_PARTIALLY            ?  TextByLanguage("Закрытие частью объёма встречной позиции","Closed by incomplete volume of opposite position") :
      reason==EVENT_REASON_DONE_PARTIALLY_BY_POS_PARTIALLY  ?  TextByLanguage("Частичное закрытие частью объёма встречной позиции","Closed partially by incomplete volume of opposite position")  :
      reason==EVENT_REASON_BALANCE_REFILL                   ?  TextByLanguage("Пополнение баланса","Balance refill")                                              :
      reason==EVENT_REASON_BALANCE_WITHDRAWAL               ?  TextByLanguage("Снятие средств с баланса","Withdrawals from balance")                          :
      reason==EVENT_REASON_ACCOUNT_CREDIT                   ?  TextByLanguage("Начисление кредита","Credit")                                                      :
      reason==EVENT_REASON_ACCOUNT_CHARGE                   ?  TextByLanguage("Дополнительные сборы","Additional charge")                                         :
      reason==EVENT_REASON_ACCOUNT_CORRECTION               ?  TextByLanguage("Корректирующая запись","Correction")                                               :
      reason==EVENT_REASON_ACCOUNT_BONUS                    ?  TextByLanguage("Перечисление бонусов","Bonus")                                                     :
      reason==EVENT_REASON_ACCOUNT_COMISSION                ?  TextByLanguage("Дополнительные комиссии","Additional commission")                                  :
      reason==EVENT_REASON_ACCOUNT_COMISSION_DAILY          ?  TextByLanguage("Комиссия, начисляемая в конце торгового дня","Daily commission")                   :
      reason==EVENT_REASON_ACCOUNT_COMISSION_MONTHLY        ?  TextByLanguage("Комиссия, начисляемая в конце месяца","Monthly commission")                        :
      reason==EVENT_REASON_ACCOUNT_COMISSION_AGENT_DAILY    ?  TextByLanguage("Агентская комиссия, начисляемая в конце торгового дня","Daily agent commission")   :
      reason==EVENT_REASON_ACCOUNT_COMISSION_AGENT_MONTHLY  ?  TextByLanguage("Агентская комиссия, начисляемая в конце месяца","Monthly agent commission")        :
      reason==EVENT_REASON_ACCOUNT_INTEREST                 ?  TextByLanguage("Начисления процентов на свободные средства","Interest rate")                       :
      reason==EVENT_REASON_BUY_CANCELLED                    ?  TextByLanguage("Отмененная сделка покупки","Canceled buy deal")                                    :
      reason==EVENT_REASON_SELL_CANCELLED                   ?  TextByLanguage("Отмененная сделка продажи","Canceled sell deal")                                   :
      reason==EVENT_REASON_DIVIDENT                         ?  TextByLanguage("Начисление дивиденда","Dividend operations")                                       :
      reason==EVENT_REASON_DIVIDENT_FRANKED                 ?  TextByLanguage("Начисление франкированного дивиденда","Franked (non-taxable) dividend operations") :
      reason==EVENT_REASON_TAX                              ?  TextByLanguage("Начисление налога","Tax charges")                                                  :
      EnumToString(reason)
     );
  }
//+------------------------------------------------------------------+
//| Display the event properties in the journal                      |
//+------------------------------------------------------------------+
void CEvent::Print(const bool full_prop=false)
  {
   ::Print("============= ",TextByLanguage("Начало списка параметров события: \"","Beginning of event parameter list: \""),this.StatusDescription(),"\" =============");
   int beg=0, end=EVENT_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_EVENT_PROP_INTEGER prop=(ENUM_EVENT_PROP_INTEGER)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=EVENT_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_EVENT_PROP_DOUBLE prop=(ENUM_EVENT_PROP_DOUBLE)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=EVENT_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_EVENT_PROP_STRING prop=(ENUM_EVENT_PROP_STRING)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("================== ",TextByLanguage("Конец списка параметров: \"","End of parameter list: \""),this.StatusDescription(),"\" ==================\n");
  }
//+------------------------------------------------------------------+
```

The logic of all these methods is similar to the one of the already described order data output methods. Therefore, we will not focus on them as
everything is quite simple and visually comprehensible here.

The full event class listing:

```
//+------------------------------------------------------------------+
//|                                                        Event.mqh |
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
#include "\..\..\Services\DELib.mqh"
#include "..\..\Collections\HistoryCollection.mqh"
#include "..\..\Collections\MarketCollection.mqh"
//+------------------------------------------------------------------+
//| Abstract event class                                             |
//+------------------------------------------------------------------+
class CEvent : public CObject
  {
private:
   int               m_event_code;                                   // Event code
//--- Return the index of the array the event's (1) double and (2) string properties are located at
   int               IndexProp(ENUM_EVENT_PROP_DOUBLE property)const { return(int)property-EVENT_PROP_INTEGER_TOTAL;                         }
   int               IndexProp(ENUM_EVENT_PROP_STRING property)const { return(int)property-EVENT_PROP_INTEGER_TOTAL-EVENT_PROP_DOUBLE_TOTAL; }
protected:
   ENUM_TRADE_EVENT  m_trade_event;                                  // Trading event
   long              m_chart_id;                                     // Control program chart ID
   int               m_digits_acc;                                   // Number of decimal places for the account currency
   long              m_long_prop[EVENT_PROP_INTEGER_TOTAL];          // Event integer properties
   double            m_double_prop[EVENT_PROP_DOUBLE_TOTAL];         // Event real properties
   string            m_string_prop[EVENT_PROP_STRING_TOTAL];         // Event string properties
//--- return the flag presence in the trading event
   bool              IsPresentEventFlag(const int event_code)  const { return (this.m_event_code & event_code)==event_code;            }

   //--- Protected parametric constructor
                     CEvent(const ENUM_EVENT_STATUS event_status,const int event_code,const ulong ticket);
public:
//--- Default constructor
                     CEvent(void){;}

//--- Set event's (1) integer, (2) real and (3) string properties
   void              SetProperty(ENUM_EVENT_PROP_INTEGER property,long value) { this.m_long_prop[property]=value;                      }
   void              SetProperty(ENUM_EVENT_PROP_DOUBLE property,double value){ this.m_double_prop[this.IndexProp(property)]=value;    }
   void              SetProperty(ENUM_EVENT_PROP_STRING property,string value){ this.m_string_prop[this.IndexProp(property)]=value;    }
//--- Return the event's (1) integer, (2) real and (3) string properties from the property array
   long              GetProperty(ENUM_EVENT_PROP_INTEGER property)      const { return this.m_long_prop[property];                     }
   double            GetProperty(ENUM_EVENT_PROP_DOUBLE property)       const { return this.m_double_prop[this.IndexProp(property)];   }
   string            GetProperty(ENUM_EVENT_PROP_STRING property)       const { return this.m_string_prop[this.IndexProp(property)];   }

//--- Return the flag of the event supporting the property
   virtual bool      SupportProperty(ENUM_EVENT_PROP_INTEGER property)        { return true; }
   virtual bool      SupportProperty(ENUM_EVENT_PROP_DOUBLE property)         { return true; }
   virtual bool      SupportProperty(ENUM_EVENT_PROP_STRING property)         { return true; }

//--- Set the control program chart ID
   void              SetChartID(const long id)                                { this.m_chart_id=id;                                    }
//--- Decode the event code and set the trading event, (2) return the trading event
   void              SetTypeEvent(void);
   ENUM_TRADE_EVENT  TradeEvent(void)                                   const { return this.m_trade_event;                             }
//--- Send the event to the chart (implementation in the class descendants)
   virtual void      SendEvent(void) {;}

//--- Compare CEvent objects by a specified property (to sort the lists by a specified event object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CEvent objects by all properties (to search for equal event objects)
   bool              IsEqual(CEvent* compared_event);
//+------------------------------------------------------------------+
//| Methods of simplified access to event object properties          |
//+------------------------------------------------------------------+
//--- Return (1) event type, (2) event time in milliseconds, (3) event status, (4) event reason, (5) deal type, (6) deal ticket,
//--- (7) order type, based on which a deal was executed, (8) position opening order type, (9) position last order ticket,
//--- (10) position first order ticket, (11) position ID, (12) opposite position ID, (13) magic number, (14) position open time

   ENUM_TRADE_EVENT  TypeEvent(void)                                    const { return (ENUM_TRADE_EVENT)this.GetProperty(EVENT_PROP_TYPE_EVENT);     }
   long              TimeEvent(void)                                    const { return this.GetProperty(EVENT_PROP_TIME_EVENT);                       }
   ENUM_EVENT_STATUS Status(void)                                       const { return (ENUM_EVENT_STATUS)this.GetProperty(EVENT_PROP_STATUS_EVENT);  }
   ENUM_EVENT_REASON Reason(void)                                       const { return (ENUM_EVENT_REASON)this.GetProperty(EVENT_PROP_REASON_EVENT);  }
   long              TypeDeal(void)                                     const { return this.GetProperty(EVENT_PROP_TYPE_DEAL_EVENT);                  }
   long              TicketDeal(void)                                   const { return this.GetProperty(EVENT_PROP_TICKET_DEAL_EVENT);                }
   long              TypeOrderEvent(void)                               const { return this.GetProperty(EVENT_PROP_TYPE_ORDER_EVENT);                 }
   long              TypeOrderPosition(void)                            const { return this.GetProperty(EVENT_PROP_TYPE_ORDER_POSITION);              }
   long              TicketOrderEvent(void)                             const { return this.GetProperty(EVENT_PROP_TICKET_ORDER_EVENT);               }
   long              TicketOrderPosition(void)                          const { return this.GetProperty(EVENT_PROP_TICKET_ORDER_POSITION);            }
   long              PositionID(void)                                   const { return this.GetProperty(EVENT_PROP_POSITION_ID);                      }
   long              PositionByID(void)                                 const { return this.GetProperty(EVENT_PROP_POSITION_BY_ID);                   }
   long              Magic(void)                                        const { return this.GetProperty(EVENT_PROP_MAGIC_ORDER);                      }
   long              TimePosition(void)                                 const { return this.GetProperty(EVENT_PROP_TIME_ORDER_POSITION);              }

//--- Return (1) the price the event occurred at, (2) open price, (3) close price,
//--- (4) StopLoss price, (5) TakeProfit price, (6) profit, (7) requested volume, (8), executed volume, (9) remaining volume
   double            PriceEvent(void)                                   const { return this.GetProperty(EVENT_PROP_PRICE_EVENT);                      }
   double            PriceOpen(void)                                    const { return this.GetProperty(EVENT_PROP_PRICE_OPEN);                       }
   double            PriceClose(void)                                   const { return this.GetProperty(EVENT_PROP_PRICE_CLOSE);                      }
   double            PriceStopLoss(void)                                const { return this.GetProperty(EVENT_PROP_PRICE_SL);                         }
   double            PriceTakeProfit(void)                              const { return this.GetProperty(EVENT_PROP_PRICE_TP);                         }
   double            Profit(void)                                       const { return this.GetProperty(EVENT_PROP_PROFIT);                           }
   double            VolumeInitial(void)                                const { return this.GetProperty(EVENT_PROP_VOLUME_INITIAL);                   }
   double            VolumeExecuted(void)                               const { return this.GetProperty(EVENT_PROP_VOLUME_EXECUTED);                  }
   double            VolumeCurrent(void)                                const { return this.GetProperty(EVENT_PROP_VOLUME_CURRENT);                   }

//--- Return a symbol
   string            Symbol(void)                                       const { return this.GetProperty(EVENT_PROP_SYMBOL);                           }

//+------------------------------------------------------------------+
//| Descriptions of order object properties                          |
//+------------------------------------------------------------------+
//--- Return description of the order's (1) integer, (2) real and (3) string properties
   string            GetPropertyDescription(ENUM_EVENT_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_EVENT_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_EVENT_PROP_STRING property);
//--- Return the event's (1) status and (2) type
   string            StatusDescription(void)          const;
   string            TypeEventDescription(void)       const;
//--- Return the name of an (1) order/position/deal, (2) parent order, (3) position
   string            TypeOrderDescription(void)       const;
   string            TypeOrderBasedDescription(void)  const;
   string            TypePositionDescription(void)    const;
//--- Return the name of the deal/order/position reason
   string            ReasonDescription(void)          const;

//--- Display (1) description of order properties (full_prop=true - all properties, false - only supported ones),
//--- (2) short event message (implementation in the class descendants) in the journal
   void              Print(const bool full_prop=false);
   virtual void      PrintShort(void) {;}
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CEvent::CEvent(const ENUM_EVENT_STATUS event_status,const int event_code,const ulong ticket) : m_event_code(event_code)
  {
   this.m_long_prop[EVENT_PROP_STATUS_EVENT]       =  event_status;
   this.m_long_prop[EVENT_PROP_TICKET_ORDER_EVENT] =  (long)ticket;
   this.m_digits_acc=(int)::AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS);
   this.m_chart_id=::ChartID();
  }
//+------------------------------------------------------------------+
//| Compare CEvent objects by a specified property                   |
//+------------------------------------------------------------------+
int CEvent::Compare(const CObject *node,const int mode=0) const
  {
   const CEvent *event_compared=node;
//--- compare integer properties of two events
   if(mode<EVENT_PROP_INTEGER_TOTAL)
     {
      long value_compared=event_compared.GetProperty((ENUM_EVENT_PROP_INTEGER)mode);
      long value_current=this.GetProperty((ENUM_EVENT_PROP_INTEGER)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare real properties of two events
   if(mode<EVENT_PROP_DOUBLE_TOTAL+EVENT_PROP_INTEGER_TOTAL)
     {
      double value_compared=event_compared.GetProperty((ENUM_EVENT_PROP_DOUBLE)mode);
      double value_current=this.GetProperty((ENUM_EVENT_PROP_DOUBLE)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare string properties of two events
   else if(mode<EVENT_PROP_DOUBLE_TOTAL+EVENT_PROP_INTEGER_TOTAL+EVENT_PROP_STRING_TOTAL)
     {
      string value_compared=event_compared.GetProperty((ENUM_EVENT_PROP_STRING)mode);
      string value_current=this.GetProperty((ENUM_EVENT_PROP_STRING)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
   return 0;
  }
//+------------------------------------------------------------------+
//| Compare CEvent objects by all properties                         |
//+------------------------------------------------------------------+
bool CEvent::IsEqual(CEvent *compared_event)
  {
   int beg=0, end=EVENT_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_EVENT_PROP_INTEGER prop=(ENUM_EVENT_PROP_INTEGER)i;
      if(this.GetProperty(prop)!=compared_event.GetProperty(prop)) return false;
     }
   beg=end; end+=EVENT_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_EVENT_PROP_DOUBLE prop=(ENUM_EVENT_PROP_DOUBLE)i;
      if(this.GetProperty(prop)!=compared_event.GetProperty(prop)) return false;
     }
   beg=end; end+=EVENT_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_EVENT_PROP_STRING prop=(ENUM_EVENT_PROP_STRING)i;
      if(this.GetProperty(prop)!=compared_event.GetProperty(prop)) return false;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Decode the event code and set a trading event                    |
//+------------------------------------------------------------------+
void CEvent::SetTypeEvent(void)
  {
//--- Pending order placed (check for matching the event code since there can only be one flag here)
   if(this.m_event_code==TRADE_EVENT_FLAG_ORDER_PLASED)
     {
      this.m_trade_event=TRADE_EVENT_PENDING_ORDER_PLASED;
      this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
      return;
     }
//--- Pending order removed (check for matching the event code since there can only be one flag here)
   if(this.m_event_code==TRADE_EVENT_FLAG_ORDER_REMOVED)
     {
      this.m_trade_event=TRADE_EVENT_PENDING_ORDER_REMOVED;
      this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
      return;
     }
//--- Position opened (Check the presence of multiple flags in the event code)
   if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_POSITION_OPENED))
     {
      //--- If the pending order is activated by a price
      if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_ORDER_ACTIVATED))
        {
         //--- check the partial closure flag and set the "pending order activated" or "pending order partially activated" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_PENDING_ORDER_ACTIVATED : TRADE_EVENT_PENDING_ORDER_ACTIVATED_PARTIAL);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
      //--- check the partial opening flag and set the "Position opened" or "Position partially opened" trading event
      this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_OPENED : TRADE_EVENT_POSITION_OPENED_PARTIAL);
      this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
      return;
     }
//--- Position closed (Check the presence of multiple flags in the event code)
   if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_POSITION_CLOSED))
     {
      //--- if a position is closed by StopLoss
      if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_SL))
        {
         //--- check the partial closing flag and set the "Position closed by StopLoss" or "Position partially closed by StopLoss" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED_BY_SL : TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_SL);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
      //--- if a position is closed by TakeProfit
      else if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_TP))
        {
         //--- check the partial closure flag and set the "Position closed by TakeProfit" or "Position partially closed by TakeProfit" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED_BY_TP : TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_TP);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
      //--- if a position is closed by an opposite one
      else if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_BY_POS))
        {
         //--- check the partial closure flag and set the "Position closed by opposite one" or "Position partially closed by opposite one" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED_BY_POS : TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
      //--- If a position is closed
      else
        {
         //--- check the partial closure flag and set the "Position closed" or "Position partially closed" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED : TRADE_EVENT_POSITION_CLOSED_PARTIAL);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
     }
//--- Balance operation on the account (clarify the event by deal type)
   if(this.m_event_code==TRADE_EVENT_FLAG_ACCOUNT_BALANCE)
     {
      //--- Initialize a trading event
      this.m_trade_event=TRADE_EVENT_NO_EVENT;
      //--- Take a deal type
      ENUM_DEAL_TYPE deal_type=(ENUM_DEAL_TYPE)this.GetProperty(EVENT_PROP_TYPE_DEAL_EVENT);
      //--- if a deal is a balance operation
      if(deal_type==DEAL_TYPE_BALANCE)
        {
        //--- check the deal profit and set an event (funds deposit or withdrawal)
         this.m_trade_event=(this.GetProperty(EVENT_PROP_PROFIT)>0 ? TRADE_EVENT_ACCOUNT_BALANCE_REFILL : TRADE_EVENT_ACCOUNT_BALANCE_WITHDRAWAL);
        }
      //--- The remaining balance operation types match the ENUM_DEAL_TYPE enumeration starting with DEAL_TYPE_CREDIT
      else if(deal_type>DEAL_TYPE_BALANCE)
        {
        //--- set the event
         this.m_trade_event=(ENUM_TRADE_EVENT)deal_type;
        }
      this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
      return;
     }
  }
//+------------------------------------------------------------------+
//| Return the description of the event's integer property           |
//+------------------------------------------------------------------+
string CEvent::GetPropertyDescription(ENUM_EVENT_PROP_INTEGER property)
  {
   return
     (
      property==EVENT_PROP_TYPE_EVENT              ?  TextByLanguage("Тип события","Event type")+": "+this.TypeEventDescription()                                                       :
      property==EVENT_PROP_TIME_EVENT              ?  TextByLanguage("Время события","Time of event")+": "+TimeMSCtoString(this.GetProperty(property))                                    :
      property==EVENT_PROP_STATUS_EVENT            ?  TextByLanguage("Статус события","Status of event")+": \""+this.StatusDescription()+"\""                                             :
      property==EVENT_PROP_REASON_EVENT            ?  TextByLanguage("Причина события","Reason of event")+": "+this.ReasonDescription()                                                   :
      property==EVENT_PROP_TYPE_DEAL_EVENT         ?  TextByLanguage("Тип сделки","Deal's type")+": "+DealTypeDescription((ENUM_DEAL_TYPE)this.GetProperty(property))                     :
      property==EVENT_PROP_TICKET_DEAL_EVENT       ?  TextByLanguage("Тикет сделки","Deal's ticket")+" #"+(string)this.GetProperty(property)                                              :
      property==EVENT_PROP_TYPE_ORDER_EVENT        ?  TextByLanguage("Тип ордера события","Event's order type")+": "+OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(property))    :
      property==EVENT_PROP_TYPE_ORDER_POSITION     ?  TextByLanguage("Тип ордера позиции","Position's order type")+": "+OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(property)) :
      property==EVENT_PROP_TICKET_ORDER_POSITION   ?  TextByLanguage("Тикет первого ордера позиции","Position's first order ticket")+" #"+(string)this.GetProperty(property)              :
      property==EVENT_PROP_TICKET_ORDER_EVENT      ?  TextByLanguage("Тикет ордера события","Event's order ticket")+" #"+(string)this.GetProperty(property)                               :
      property==EVENT_PROP_POSITION_ID             ?  TextByLanguage("Идентификатор позиции","Position ID")+" #"+(string)this.GetProperty(property)                                       :
      property==EVENT_PROP_POSITION_BY_ID          ?  TextByLanguage("Идентификатор встречной позиции","Opposite position's ID")+" #"+(string)this.GetProperty(property)                  :
      property==EVENT_PROP_MAGIC_ORDER             ?  TextByLanguage("Магический номер","Magic number")+": "+(string)this.GetProperty(property)                                           :
      property==EVENT_PROP_TIME_ORDER_POSITION     ?  TextByLanguage("Время открытия позиции","Position's opened time")+": "+TimeMSCtoString(this.GetProperty(property))                  :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return the description of the event's real property              |
//+------------------------------------------------------------------+
string CEvent::GetPropertyDescription(ENUM_EVENT_PROP_DOUBLE property)
  {
   int dg=(int)::SymbolInfoInteger(this.GetProperty(EVENT_PROP_SYMBOL),SYMBOL_DIGITS);
   int dgl=(int)DigitsLots(this.GetProperty(EVENT_PROP_SYMBOL));
   return
     (
      property==EVENT_PROP_PRICE_EVENT       ?  TextByLanguage("Цена события","Price at the time of event")+": "+::DoubleToString(this.GetProperty(property),dg) :
      property==EVENT_PROP_PRICE_OPEN        ?  TextByLanguage("Цена открытия","Open price")+": "+::DoubleToString(this.GetProperty(property),dg)                    :
      property==EVENT_PROP_PRICE_CLOSE       ?  TextByLanguage("Цена закрытия","Close price")+": "+::DoubleToString(this.GetProperty(property),dg)                   :
      property==EVENT_PROP_PRICE_SL          ?  TextByLanguage("Цена StopLoss","StopLoss price")+": "+::DoubleToString(this.GetProperty(property),dg)                :
      property==EVENT_PROP_PRICE_TP          ?  TextByLanguage("Цена TakeProfit","TakeProfit price")+": "+::DoubleToString(this.GetProperty(property),dg)            :
      property==EVENT_PROP_VOLUME_INITIAL    ?  TextByLanguage("Начальный объём","Initial volume")+": "+::DoubleToString(this.GetProperty(property),dgl)             :
      property==EVENT_PROP_VOLUME_EXECUTED   ?  TextByLanguage("Исполненный объём","Executed volume")+": "+::DoubleToString(this.GetProperty(property),dgl)          :
      property==EVENT_PROP_VOLUME_CURRENT    ?  TextByLanguage("Оставшийся объём","Remaining volume")+": "+::DoubleToString(this.GetProperty(property),dgl)          :
      property==EVENT_PROP_PROFIT            ?  TextByLanguage("Профит","Profit")+": "+::DoubleToString(this.GetProperty(property),this.m_digits_acc)                :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return the description of the event's string property            |
//+------------------------------------------------------------------+
string CEvent::GetPropertyDescription(ENUM_EVENT_PROP_STRING property)
  {
   return TextByLanguage("Символ","Symbol")+": \""+this.GetProperty(property)+"\"";
  }
//+------------------------------------------------------------------+
//| Return the event status name                                     |
//+------------------------------------------------------------------+
string CEvent::StatusDescription(void) const
  {
   ENUM_EVENT_STATUS status=(ENUM_EVENT_STATUS)this.GetProperty(EVENT_PROP_STATUS_EVENT);
   return
     (
      status==EVENT_STATUS_MARKET_PENDING    ?  TextByLanguage("Установлен отложенный ордер","Pending order placed") :
      status==EVENT_STATUS_MARKET_POSITION   ?  TextByLanguage("Открыта позиция","Position opened")                 :
      status==EVENT_STATUS_HISTORY_PENDING   ?  TextByLanguage("Удален отложенный ордер","Pending order removed")    :
      status==EVENT_STATUS_HISTORY_POSITION  ?  TextByLanguage("Закрыта позиция","Position closed")                  :
      status==EVENT_STATUS_BALANCE           ?  TextByLanguage("Балансная операция","Balance operation")             :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return the trading event name                                    |
//+------------------------------------------------------------------+
string CEvent::TypeEventDescription(void) const
  {
   ENUM_TRADE_EVENT event=this.TypeEvent();
   return
     (
      event==TRADE_EVENT_PENDING_ORDER_PLASED            ?  TextByLanguage("Отложенный ордер установлен","Pending order placed")                                  :
      event==TRADE_EVENT_PENDING_ORDER_REMOVED           ?  TextByLanguage("Отложенный ордер удалён","Pending order removed")                                     :
      event==TRADE_EVENT_ACCOUNT_CREDIT                  ?  TextByLanguage("Начисление кредита","Credit")                                                         :
      event==TRADE_EVENT_ACCOUNT_CHARGE                  ?  TextByLanguage("Дополнительные сборы","Additional charge")                                            :
      event==TRADE_EVENT_ACCOUNT_CORRECTION              ?  TextByLanguage("Корректирующая запись","Correction")                                                  :
      event==TRADE_EVENT_ACCOUNT_BONUS                   ?  TextByLanguage("Перечисление бонусов","Bonus")                                                        :
      event==TRADE_EVENT_ACCOUNT_COMISSION               ?  TextByLanguage("Дополнительные комиссии","Additional commission")                                     :
      event==TRADE_EVENT_ACCOUNT_COMISSION_DAILY         ?  TextByLanguage("Комиссия, начисляемая в конце торгового дня","Daily commission")                      :
      event==TRADE_EVENT_ACCOUNT_COMISSION_MONTHLY       ?  TextByLanguage("Комиссия, начисляемая в конце месяца","Monthly commission")                           :
      event==TRADE_EVENT_ACCOUNT_COMISSION_AGENT_DAILY   ?  TextByLanguage("Агентская комиссия, начисляемая в конце торгового дня","Daily agent commission")      :
      event==TRADE_EVENT_ACCOUNT_COMISSION_AGENT_MONTHLY ?  TextByLanguage("Агентская комиссия, начисляемая в конце месяца","Monthly agent commission")           :
      event==TRADE_EVENT_ACCOUNT_INTEREST                ?  TextByLanguage("Начисления процентов на свободные средства","Interest rate")                          :
      event==TRADE_EVENT_BUY_CANCELLED                   ?  TextByLanguage("Отмененная сделка покупки","Canceled buy deal")                                       :
      event==TRADE_EVENT_SELL_CANCELLED                  ?  TextByLanguage("Отмененная сделка продажи","Canceled sell deal")                                      :
      event==TRADE_EVENT_DIVIDENT                        ?  TextByLanguage("Начисление дивиденда","Dividend operations")                                          :
      event==TRADE_EVENT_DIVIDENT_FRANKED                ?  TextByLanguage("Начисление франкированного дивиденда","Franked (non-taxable) dividend operations")    :
      event==TRADE_EVENT_TAX                             ?  TextByLanguage("Начисление налога","Tax charges")                                                     :
      event==TRADE_EVENT_ACCOUNT_BALANCE_REFILL          ?  TextByLanguage("Пополнение средств на балансе","Balance refill")                                      :
      event==TRADE_EVENT_ACCOUNT_BALANCE_WITHDRAWAL      ?  TextByLanguage("Снятие средств с баланса","Withdrawals")                                              :
      event==TRADE_EVENT_PENDING_ORDER_ACTIVATED         ?  TextByLanguage("Отложенный ордер активирован ценой","Pending order activated")                        :
      event==TRADE_EVENT_PENDING_ORDER_ACTIVATED_PARTIAL ?  TextByLanguage("Отложенный ордер активирован ценой частично","Pending order activated partially")     :
      event==TRADE_EVENT_POSITION_OPENED                 ?  TextByLanguage("Позиция открыта","Position opened")                                                  :
      event==TRADE_EVENT_POSITION_OPENED_PARTIAL         ?  TextByLanguage("Позиция открыта частично","Position opened partially")                               :
      event==TRADE_EVENT_POSITION_CLOSED                 ?  TextByLanguage("Позиция закрыта","Position closed")                                                   :
      event==TRADE_EVENT_POSITION_CLOSED_PARTIAL         ?  TextByLanguage("Позиция закрыта частично","Position closed partially")                                :
      event==TRADE_EVENT_POSITION_CLOSED_BY_POS          ?  TextByLanguage("Позиция закрыта встречной","Position closed by opposite position")                    :
      event==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS  ?  TextByLanguage("Позиция закрыта встречной частично","Position closed partially by opposite position") :
      event==TRADE_EVENT_POSITION_CLOSED_BY_SL           ?  TextByLanguage("Позиция закрыта по StopLoss","Position closed by StopLoss")                           :
      event==TRADE_EVENT_POSITION_CLOSED_BY_TP           ?  TextByLanguage("Позиция закрыта по TakeProfit","Position closed by TakeProfit")                       :
      event==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_SL   ?  TextByLanguage("Позиция закрыта частично по StopLoss","Position closed partially by StopLoss")        :
      event==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_TP   ?  TextByLanguage("Позиция закрыта частично по TakeProfit","Position closed partially by TakeProfit")    :
      event==TRADE_EVENT_POSITION_REVERSED               ?  TextByLanguage("Разворот позиции","Position reversal")                                                :
      event==TRADE_EVENT_POSITION_VOLUME_ADD             ?  TextByLanguage("Добавлен объём к позиции","Added volume to position")                                 :
      TextByLanguage("Нет торгового события","No trade event")
     );
  }
//+------------------------------------------------------------------+
//| Return the name of the order/position/deal                       |
//+------------------------------------------------------------------+
string CEvent::TypeOrderDescription(void) const
  {
   ENUM_EVENT_STATUS status=this.Status();
   return
     (
      status==EVENT_STATUS_MARKET_PENDING  || status==EVENT_STATUS_HISTORY_PENDING  ?  OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(EVENT_PROP_TYPE_ORDER_EVENT))      :
      status==EVENT_STATUS_MARKET_POSITION || status==EVENT_STATUS_HISTORY_POSITION ?  PositionTypeDescription((ENUM_POSITION_TYPE)this.GetProperty(EVENT_PROP_TYPE_DEAL_EVENT)) :
      status==EVENT_STATUS_BALANCE  ?  DealTypeDescription((ENUM_DEAL_TYPE)this.GetProperty(EVENT_PROP_TYPE_DEAL_EVENT))  :  "Unknown"
     );
  }
//+------------------------------------------------------------------+
//| Return the name of the parent order                              |
//+------------------------------------------------------------------+
string CEvent::TypeOrderBasedDescription(void) const
  {
   return OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(EVENT_PROP_TYPE_ORDER_POSITION));
  }
//+------------------------------------------------------------------+
//| Return the position name                                         |
//+------------------------------------------------------------------+
string CEvent::TypePositionDescription(void) const
  {
   ENUM_POSITION_TYPE type=PositionTypeByOrderType((ENUM_ORDER_TYPE)this.GetProperty(EVENT_PROP_TYPE_ORDER_POSITION));
   return PositionTypeDescription(type);
  }
//+------------------------------------------------------------------+
//| Return the name of the deal/order/position reason                |
//+------------------------------------------------------------------+
string CEvent::ReasonDescription(void) const
  {
   ENUM_EVENT_REASON reason=this.Reason();
   return
     (
      reason==EVENT_REASON_ACTIVATED_PENDING                ?  TextByLanguage("Активирован отложенный ордер","Pending order activated")                           :
      reason==EVENT_REASON_ACTIVATED_PENDING_PARTIALLY      ?  TextByLanguage("Частичное срабатывание отложенного ордера","Pending order partially triggered")    :
      reason==EVENT_REASON_CANCEL                           ?  TextByLanguage("Отмена","Canceled")                                                                :
      reason==EVENT_REASON_EXPIRED                          ?  TextByLanguage("Истёк срок действия","Expired")                                                    :
      reason==EVENT_REASON_DONE                             ?  TextByLanguage("Запрос выполнен полностью","Request fully executed")                            :
      reason==EVENT_REASON_DONE_PARTIALLY                   ?  TextByLanguage("Запрос выполнен частично","Request partially executed")                         :
      reason==EVENT_REASON_DONE_SL                          ?  TextByLanguage("закрытие по StopLoss","Close by StopLoss triggered")                               :
      reason==EVENT_REASON_DONE_SL_PARTIALLY                ?  TextByLanguage("Частичное закрытие по StopLoss","Partial close by StopLoss triggered")             :
      reason==EVENT_REASON_DONE_TP                          ?  TextByLanguage("закрытие по TakeProfit","Close by TakeProfit triggered")                           :
      reason==EVENT_REASON_DONE_TP_PARTIALLY                ?  TextByLanguage("Частичное закрытие по TakeProfit","Partial close by TakeProfit triggered")         :
      reason==EVENT_REASON_DONE_BY_POS                      ?  TextByLanguage("Закрытие встречной позицией","Closed by opposite position")                        :
      reason==EVENT_REASON_DONE_PARTIALLY_BY_POS            ?  TextByLanguage("Частичное закрытие встречной позицией","Closed partially by opposite position")    :
      reason==EVENT_REASON_DONE_BY_POS_PARTIALLY            ?  TextByLanguage("Закрытие частью объёма встречной позиции","Closed by incomplete volume of opposite position") :
      reason==EVENT_REASON_DONE_PARTIALLY_BY_POS_PARTIALLY  ?  TextByLanguage("Частичное закрытие частью объёма встречной позиции","Closed partially by incomplete volume of opposite position")  :
      reason==EVENT_REASON_BALANCE_REFILL                   ?  TextByLanguage("Пополнение баланса","Balance refill")                                              :
      reason==EVENT_REASON_BALANCE_WITHDRAWAL               ?  TextByLanguage("Снятие средств с баланса","Withdrawals from balance")                          :
      reason==EVENT_REASON_ACCOUNT_CREDIT                   ?  TextByLanguage("Начисление кредита","Credit")                                                      :
      reason==EVENT_REASON_ACCOUNT_CHARGE                   ?  TextByLanguage("Дополнительные сборы","Additional charge")                                         :
      reason==EVENT_REASON_ACCOUNT_CORRECTION               ?  TextByLanguage("Корректирующая запись","Correction")                                               :
      reason==EVENT_REASON_ACCOUNT_BONUS                    ?  TextByLanguage("Перечисление бонусов","Bonus")                                                     :
      reason==EVENT_REASON_ACCOUNT_COMISSION                ?  TextByLanguage("Дополнительные комиссии","Additional commission")                                  :
      reason==EVENT_REASON_ACCOUNT_COMISSION_DAILY          ?  TextByLanguage("Комиссия, начисляемая в конце торгового дня","Daily commission")                   :
      reason==EVENT_REASON_ACCOUNT_COMISSION_MONTHLY        ?  TextByLanguage("Комиссия, начисляемая в конце месяца","Monthly commission")                        :
      reason==EVENT_REASON_ACCOUNT_COMISSION_AGENT_DAILY    ?  TextByLanguage("Агентская комиссия, начисляемая в конце торгового дня","Daily agent commission")   :
      reason==EVENT_REASON_ACCOUNT_COMISSION_AGENT_MONTHLY  ?  TextByLanguage("Агентская комиссия, начисляемая в конце месяца","Monthly agent commission")        :
      reason==EVENT_REASON_ACCOUNT_INTEREST                 ?  TextByLanguage("Начисления процентов на свободные средства","Interest rate")                       :
      reason==EVENT_REASON_BUY_CANCELLED                    ?  TextByLanguage("Отмененная сделка покупки","Canceled buy deal")                                    :
      reason==EVENT_REASON_SELL_CANCELLED                   ?  TextByLanguage("Отмененная сделка продажи","Canceled sell deal")                                   :
      reason==EVENT_REASON_DIVIDENT                         ?  TextByLanguage("Начисление дивиденда","Dividend operations")                                       :
      reason==EVENT_REASON_DIVIDENT_FRANKED                 ?  TextByLanguage("Начисление франкированного дивиденда","Franked (non-taxable) dividend operations") :
      reason==EVENT_REASON_TAX                              ?  TextByLanguage("Начисление налога","Tax charges")                                                  :
      EnumToString(reason)
     );
  }
//+------------------------------------------------------------------+
//| Display the event properties to the journal                      |
//+------------------------------------------------------------------+
void CEvent::Print(const bool full_prop=false)
  {
   ::Print("============= ",TextByLanguage("Начало списка параметров события: \"","Beginning of event parameter list: \""),this.StatusDescription(),"\" =============");
   int beg=0, end=EVENT_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_EVENT_PROP_INTEGER prop=(ENUM_EVENT_PROP_INTEGER)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=EVENT_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_EVENT_PROP_DOUBLE prop=(ENUM_EVENT_PROP_DOUBLE)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=EVENT_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_EVENT_PROP_STRING prop=(ENUM_EVENT_PROP_STRING)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("================== ",TextByLanguage("Конец списка параметров: \"","End of parameter list: \""),this.StatusDescription(),"\" ==================\n");
  }
//+------------------------------------------------------------------+
```

The class of the abstract base event is ready. Now we need to create five descendant classes that will be an event indicating its type: placing a
pending order, deleting a pending order, opening a position, closing a position and a balance operation.

**Create a descendant class having the "Placing pending order" event status.**

In the Events library folder, create a new file of the CEventOrderPlased class named EventOrderPlased.mqh with
the CEvent base class and add all the necessary connections
and methods to it:

```
//+------------------------------------------------------------------+
//|                                             EventOrderPlased.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Event.mqh"
//+------------------------------------------------------------------+
//| Placing a pending order event                                    |
//+------------------------------------------------------------------+
class CEventOrderPlased : public CEvent
  {
public:
//--- Constructor
                     CEventOrderPlased(const int event_code,const ulong ticket=0) : CEvent(EVENT_STATUS_MARKET_PENDING,event_code,ticket) {}
//--- Supported (1) real and (2) integer order properties
   virtual bool      SupportProperty(ENUM_EVENT_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_EVENT_PROP_DOUBLE property);
//--- (1) Display a brief message about the event in the journal, (2) Send the event to the chart
   virtual void      PrintShort(void);
   virtual void      SendEvent(void);
  };
//+------------------------------------------------------------------+
```

Pass the event code and ticket
of an order or deal which triggered the event to the class constructor, and send the "Placing
pending order" (EVENT\_STATUS\_MARKET\_PENDING) event status, event code and an order or deal ticket to the parent class in the
initialization list:

```
CEventOrderPlased(const int event_code,const ulong ticket=0) : CEvent(EVENT_STATUS_MARKET_PENDING,event_code,ticket) {}
```

We have already described the methods returning the flags of an object supporting certain SupportProperty() properties in the first part
of the library description. All is the same here:

```
//+------------------------------------------------------------------+
//| Return 'true' if the event supports the passed                   |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CEventOrderPlased::SupportProperty(ENUM_EVENT_PROP_INTEGER property)
  {
   if(property==EVENT_PROP_TYPE_DEAL_EVENT         ||
      property==EVENT_PROP_TICKET_DEAL_EVENT       ||
      property==EVENT_PROP_TYPE_ORDER_POSITION     ||
      property==EVENT_PROP_TICKET_ORDER_POSITION   ||
      property==EVENT_PROP_POSITION_ID             ||
      property==EVENT_PROP_POSITION_BY_ID          ||
      property==EVENT_PROP_TIME_ORDER_POSITION
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if the event supports the passed                   |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CEventOrderPlased::SupportProperty(ENUM_EVENT_PROP_DOUBLE property)
  {
   if(property==EVENT_PROP_PRICE_CLOSE             ||
      property==EVENT_PROP_PROFIT
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
```

The CEvent parent event object features the Print() method that displays full data on all supported event object's properties and the
virtual PrintShort() method allowing to display sufficient data on the event in the terminal journal in two lines.

The implementation of the PrintShort() method in each descendant class of the basic event object will be individual since events are
also different in their origins:

```
//+------------------------------------------------------------------+
//| Display a brief event message in the journal                     |
//+------------------------------------------------------------------+
void CEventOrderPlased::PrintShort(void)
  {
   string head="- "+this.TypeEventDescription()+": "+TimeMSCtoString(this.TimePosition())+" -\n";
   string sl=(this.PriceStopLoss()>0 ? ", sl "+::DoubleToString(this.PriceStopLoss(),(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS)) : "");
   string tp=(this.PriceTakeProfit()>0 ? ", tp "+::DoubleToString(this.PriceTakeProfit(),(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS)) : "");
   string vol=::DoubleToString(this.VolumeInitial(),DigitsLots(this.Symbol()));
   string magic=(this.Magic()!=0 ? TextByLanguage(", магик ",", magic ")+(string)this.Magic() : "");
   string type=this.TypeOrderDescription()+" #"+(string)this.TicketOrderEvent();
   string price=TextByLanguage(" по цене "," at price ")+::DoubleToString(this.PriceOpen(),(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS));
   string txt=head+this.Symbol()+" "+vol+" "+type+price+sl+tp+magic;
   ::Print(txt);
  }
//+------------------------------------------------------------------+
```

Here we do the following things:

- Create a message header consisting of an event type description and time
- If the order has StopLoss, create the line with its description, otherwise the string remains empty

- If the order has TakeProfit, create the line with its description, otherwise the string remains empty
- Create a line indicating the order volume
- If the order has a magic number, create the line with its description, otherwise the string remains empty
- Create a line specifying the order type and ticket
- Create a line specifying the order price and a symbol the order is placed at
- Create a full line out of all the above descriptions
- Display the created line in the journal

The method sending a custom event to the chart is quite simple:

```
//+------------------------------------------------------------------+
//| Send the event to the chart                                      |
//+------------------------------------------------------------------+
void CEventOrderPlased::SendEvent(void)
  {
   this.PrintShort();
   ::EventChartCustom(this.m_chart_id,(ushort)this.m_trade_event,this.TicketOrderEvent(),this.PriceOpen(),this.Symbol());
  }
//+------------------------------------------------------------------+
```

First, a short message about the event is displayed in the journal,
then the

[EventChartCustom()](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom) custom event is sent to the chart
specified in the

**m\_chart\_id** chart ID of the base CEvent event class.

Send the **m\_trade\_event** event to the event ID,

order
ticket — to the

**long** type parameter,


order price — to the

**double** type parameter,

order
symbol — to the

**string** type parameter.

The class for displaying messages allowing users to set message levels for displaying only necessary data in the journal is to be
developed in the future. At the current library development stage, all messages are displayed by default.

Let's consider the full listings of other event classes.

**"Removing pending order" event class:**

```
//+------------------------------------------------------------------+
//|                                            EventOrderRemoved.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Event.mqh"
//+------------------------------------------------------------------+
//| Event of placing a pending order                                 |
//+------------------------------------------------------------------+
class CEventOrderRemoved : public CEvent
  {
public:
//--- Constructor
                     CEventOrderRemoved(const int event_code,const ulong ticket=0) : CEvent(EVENT_STATUS_HISTORY_PENDING,event_code,ticket) {}
//--- Supported (1) real and (2) integer order properties
   virtual bool      SupportProperty(ENUM_EVENT_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_EVENT_PROP_DOUBLE property);
//--- (1) Display a brief message about the event in the journal, (2) Send the event to the chart
   virtual void      PrintShort(void);
   virtual void      SendEvent(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if the event supports the passed                   |
//| integer property, otherwise, return 'false'                      |
//+------------------------------------------------------------------+
bool CEventOrderRemoved::SupportProperty(ENUM_EVENT_PROP_INTEGER property)
  {
   if(property==EVENT_PROP_TYPE_DEAL_EVENT         ||
      property==EVENT_PROP_TICKET_DEAL_EVENT       ||
      property==EVENT_PROP_TYPE_ORDER_POSITION     ||
      property==EVENT_PROP_TICKET_ORDER_POSITION   ||
      property==EVENT_PROP_TIME_ORDER_POSITION
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if the event supports the passed                   |
//| real property, otherwise, return 'false'                         |
//+------------------------------------------------------------------+
bool CEventOrderRemoved::SupportProperty(ENUM_EVENT_PROP_DOUBLE property)
  {
   return(property==EVENT_PROP_PROFIT ? false : true);
  }
//+------------------------------------------------------------------+
//| Display a brief event message in the journal                     |
//+------------------------------------------------------------------+
void CEventOrderRemoved::PrintShort(void)
  {
   string head="- "+this.TypeEventDescription()+": "+TimeMSCtoString(this.TimePosition())+" -\n";
   string sl=(this.PriceStopLoss()>0 ? ", sl "+::DoubleToString(this.PriceStopLoss(),(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS)) : "");
   string tp=(this.PriceTakeProfit()>0 ? ", tp "+::DoubleToString(this.PriceTakeProfit(),(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS)) : "");
   string vol=::DoubleToString(this.VolumeInitial(),DigitsLots(this.Symbol()));
   string magic=(this.Magic()!=0 ? TextByLanguage(", магик ",", magic ")+(string)this.Magic() : "");
   string type=this.TypeOrderDescription()+" #"+(string)this.TicketOrderEvent();
   string price=TextByLanguage(" по цене "," at price ")+::DoubleToString(this.PriceOpen(),(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS));
   string txt=head+this.Symbol()+" "+vol+" "+type+price+sl+tp+magic;
   ::Print(txt);
  }
//+------------------------------------------------------------------+
//| Send the event to the chart                                      |
//+------------------------------------------------------------------+
void CEventOrderRemoved::SendEvent(void)
  {
   this.PrintShort();
   ::EventChartCustom(this.m_chart_id,(ushort)this.m_trade_event,this.TicketOrderEvent(),this.PriceOpen(),this.Symbol());
  }
//+------------------------------------------------------------------+
```

**"Position opening" event class:**

```
//+------------------------------------------------------------------+
//|                                            EventPositionOpen.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Event.mqh"
//+------------------------------------------------------------------+
//| Position opening event                                           |
//+------------------------------------------------------------------+
class CEventPositionOpen : public CEvent
  {
public:
//--- Constructor
                     CEventPositionOpen(const int event_code,const ulong ticket=0) : CEvent(EVENT_STATUS_MARKET_POSITION,event_code,ticket) {}
//--- Supported (1) real and (2) integer order properties
   virtual bool      SupportProperty(ENUM_EVENT_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_EVENT_PROP_DOUBLE property);
//--- (1) Display a brief message about the event in the journal, (2) Send the event to the chart
   virtual void      PrintShort(void);
   virtual void      SendEvent(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if the event supports the passed                   |
//| integer property, otherwise, return 'false'                      |
//+------------------------------------------------------------------+
bool CEventPositionOpen::SupportProperty(ENUM_EVENT_PROP_INTEGER property)
  {
   return(property==EVENT_PROP_POSITION_BY_ID ? false : true);
  }
//+------------------------------------------------------------------+
//| Return 'true' if the order supports the passed                   |
//| real property, otherwise, return 'false'                         |
//+------------------------------------------------------------------+
bool CEventPositionOpen::SupportProperty(ENUM_EVENT_PROP_DOUBLE property)
  {
   if(property==EVENT_PROP_PRICE_CLOSE ||
      property==EVENT_PROP_PROFIT
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Display a brief event message in the journal                     |
//+------------------------------------------------------------------+
void CEventPositionOpen::PrintShort(void)
  {
   string head="- "+this.TypeEventDescription()+": "+TimeMSCtoString(this.TimePosition())+" -\n";
   string order=(this.IsPresentEventFlag(TRADE_EVENT_FLAG_ORDER_ACTIVATED) ? " #"+(string)this.TicketOrderPosition() : "");
   string activated=(this.IsPresentEventFlag(TRADE_EVENT_FLAG_ORDER_ACTIVATED) ? TextByLanguage(" активацией ордера "," by ")+this.TypeOrderBasedDescription() : "");
   string sl=(this.PriceStopLoss()>0 ? ", sl "+::DoubleToString(this.PriceStopLoss(),(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS)) : "");
   string tp=(this.PriceTakeProfit()>0 ? ", tp "+::DoubleToString(this.PriceTakeProfit(),(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS)) : "");
   string vol=::DoubleToString(this.VolumeInitial(),DigitsLots(this.Symbol()));
   string magic=(this.Magic()!=0 ? TextByLanguage(", магик ",", magic ")+(string)this.Magic() : "");
   string type=this.TypePositionDescription()+" #"+(string)this.PositionID();
   string price=TextByLanguage(" по цене "," at price ")+::DoubleToString(this.PriceOpen(),(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS));
   string txt=head+this.Symbol()+" "+vol+" "+type+activated+order+price+sl+tp+magic;
   ::Print(txt);
  }
//+------------------------------------------------------------------+
//| Send the event to the chart                                      |
//+------------------------------------------------------------------+
void CEventPositionOpen::SendEvent(void)
  {
   this.PrintShort();
   ::EventChartCustom(this.m_chart_id,(ushort)this.m_trade_event,this.PositionID(),this.PriceOpen(),this.Symbol());
  }
//+------------------------------------------------------------------+
```

**"Position closing" event class:**

```
//+------------------------------------------------------------------+
//|                                           EventPositionClose.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Event.mqh"
//+------------------------------------------------------------------+
//| Position opening event                                           |
//+------------------------------------------------------------------+
class CEventPositionClose : public CEvent
  {
public:
//--- Constructor
                     CEventPositionClose(const int event_code,const ulong ticket=0) : CEvent(EVENT_STATUS_HISTORY_POSITION,event_code,ticket) {}
//--- Supported (1) real and (2) integer order properties
   virtual bool      SupportProperty(ENUM_EVENT_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_EVENT_PROP_DOUBLE property);
//--- (1) Display a brief message about the event in the journal, (2) Send the event to the chart
   virtual void      PrintShort(void);
   virtual void      SendEvent(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if the event supports the passed                   |
//| integer property, otherwise, return 'false'                      |
//+------------------------------------------------------------------+
bool CEventPositionClose::SupportProperty(ENUM_EVENT_PROP_INTEGER property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if the event supports the passed                   |
//| real property, otherwise, return 'false'                         |
//+------------------------------------------------------------------+
bool CEventPositionClose::SupportProperty(ENUM_EVENT_PROP_DOUBLE property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Display a brief message about the event in the journal           |
//+------------------------------------------------------------------+
void CEventPositionClose::PrintShort(void)
  {
   string head="- "+this.TypeEventDescription()+": "+TimeMSCtoString(this.TimePosition())+" -\n";
   string opposite=(this.IsPresentEventFlag(TRADE_EVENT_FLAG_BY_POS) ? " by "+this.TypeOrderDescription()+" #"+(string)this.PositionByID() : "");
   string vol=::DoubleToString(this.VolumeExecuted(),DigitsLots(this.Symbol()));
   string magic=(this.Magic()!=0 ? TextByLanguage(", магик ",", magic ")+(string)this.Magic() : "");
   string type=this.TypePositionDescription()+" #"+(string)this.PositionID()+opposite;
   string price=TextByLanguage(" по цене "," at price ")+::DoubleToString(this.PriceClose(),(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS));
   string profit=TextByLanguage(", профит: ",", profit: ")+::DoubleToString(this.Profit(),this.m_digits_acc)+" "+::AccountInfoString(ACCOUNT_CURRENCY);
   string txt=head+this.Symbol()+" "+vol+" "+type+price+magic+profit;
   ::Print(txt);
  }
//+------------------------------------------------------------------+
//| Send the event to the chart                                      |
//+------------------------------------------------------------------+
void CEventPositionClose::SendEvent(void)
  {
   this.PrintShort();
   ::EventChartCustom(this.m_chart_id,(ushort)this.m_trade_event,this.PositionID(),this.PriceClose(),this.Symbol());
  }
//+------------------------------------------------------------------+
```

**"Balance operation" event class:**

```
//+------------------------------------------------------------------+
//|                                        EventBalanceOperation.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Event.mqh"
//+------------------------------------------------------------------+
//| Position opening event                                           |
//+------------------------------------------------------------------+
class CEventBalanceOperation : public CEvent
  {
public:
//--- Constructor
                     CEventBalanceOperation(const int event_code,const ulong ticket=0) : CEvent(EVENT_STATUS_BALANCE,event_code,ticket) {}
//--- Supported (1) real and (2) integer order properties
   virtual bool      SupportProperty(ENUM_EVENT_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_EVENT_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_EVENT_PROP_STRING property);
//--- (1) Display a brief message about the event in the journal, (2) Send the event to the chart
   virtual void      PrintShort(void);
   virtual void      SendEvent(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if the event supports the passed                   |
//| integer property, otherwise, return 'false'                      |
//+------------------------------------------------------------------+
bool CEventBalanceOperation::SupportProperty(ENUM_EVENT_PROP_INTEGER property)
  {
   if(property==EVENT_PROP_TYPE_ORDER_EVENT        ||
      property==EVENT_PROP_TYPE_ORDER_POSITION     ||
      property==EVENT_PROP_TICKET_ORDER_EVENT      ||
      property==EVENT_PROP_TICKET_ORDER_POSITION   ||
      property==EVENT_PROP_POSITION_ID             ||
      property==EVENT_PROP_POSITION_BY_ID          ||
      property==EVENT_PROP_POSITION_ID             ||
      property==EVENT_PROP_MAGIC_ORDER             ||
      property==EVENT_PROP_TIME_ORDER_POSITION
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if the event supports the passed                   |
//| real property, otherwise, return 'false'                         |
//+------------------------------------------------------------------+
bool CEventBalanceOperation::SupportProperty(ENUM_EVENT_PROP_DOUBLE property)
  {
   return(property==EVENT_PROP_PROFIT ? true : false);
  }
//+------------------------------------------------------------------+
//| Return 'true' if the event supports the passed                   |
//| string property, otherwise, return 'false'                       |
//+------------------------------------------------------------------+
bool CEventBalanceOperation::SupportProperty(ENUM_EVENT_PROP_STRING property)
  {
   return false;
  }
//+------------------------------------------------------------------+
//| Display a brief message about the event in the journal           |
//+------------------------------------------------------------------+
void CEventBalanceOperation::PrintShort(void)
  {
   string head="- "+this.StatusDescription()+": "+TimeMSCtoString(this.TimePosition())+" -\n";
   ::Print(head+this.TypeEventDescription()+": "+::DoubleToString(this.Profit(),this.m_digits_acc)+" "+::AccountInfoString(ACCOUNT_CURRENCY));
  }
//+------------------------------------------------------------------+
//| Send the event to the chart                                      |
//+------------------------------------------------------------------+
void CEventBalanceOperation::SendEvent(void)
  {
   this.PrintShort();
   ::EventChartCustom(this.m_chart_id,(ushort)this.m_trade_event,this.TypeEvent(),this.Profit(),::AccountInfoString(ACCOUNT_CURRENCY));
  }
//+------------------------------------------------------------------+
```

As we can see from the listings, the classes are different only by the number of supported properties, the status sent to the parent class
constructor and the PrintShort() methods since each event has its own features that should be reflected in the journal. All this can be
understood from the methods' listings, and you can analyze them on your own, so there is no point in dwelling on them. Let's pass to the
development of the event collection class.

### Collection of trading events

In the fourth part of the library description, we [tested defining account \\
events](https://www.mql5.com/en/articles/5724#node02) and their display in the journal and the EA. However, we were able to track only the latest event. Besides, the entire
functionality was located in the base class of the CEngine library.

The correct decision is to put everything into a separate class and process all occurring events in it.

To achieve this, I have developed event objects. Now we need to make the class handling any number of simultaneously occurred events. After
all, there may be a situation when pending orders are removed or placed, or several positions are closed simultaneously in a single loop.

The principle that we have already tested in such situations would give us only the most recent event out of several ones performed at one
go. I think, this is incorrect. Therefore, let's make the class that saves all events that took place in one go into the event collection list.
Besides, in the future, it will be possible to use the methods of these classes to walk through the history of the account and recreate
everything that happened on it since it was opened.

In the DoEasy\\Collections, create the new file of the **CEventsCollection** class named **EventsCollection.mqh**.
The

**CListObj** class should be made the base one.

Fill in the newly created class template with all the necessary inclusions, members and methods right away:

```
//+------------------------------------------------------------------+
//|                                             EventsCollection.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Services\Select.mqh"
#include "..\Objects\Orders\Order.mqh"
#include "..\Objects\Events\EventBalanceOperation.mqh"
#include "..\Objects\Events\EventOrderPlaced.mqh"
#include "..\Objects\Events\EventOrderRemoved.mqh"
#include "..\Objects\Events\EventPositionOpen.mqh"
#include "..\Objects\Events\EventPositionClose.mqh"
//+------------------------------------------------------------------+
//| Collection of account events                                     |
//+------------------------------------------------------------------+
class CEventsCollection : public CListObj
  {
private:
   CListObj          m_list_events;                   // List of events
   bool              m_is_hedge;                      // Hedge account flag
   long              m_chart_id;                      // Control program chart ID
   ENUM_TRADE_EVENT  m_trade_event;                   // Account trading event
   CEvent            m_event_instance;                // Event object for searching by property

//--- Create a trading event depending on the order status
   void              CreateNewEvent(COrder* order,CArrayObj* list_history,CArrayObj* list_market);
//--- Select and return the list of market pending orders
   CArrayObj*        GetListMarketPendings(CArrayObj* list);
//--- Select and return the list of historical (1) removed pending orders, (2) deals, (3) all closing orders
   CArrayObj*        GetListHistoryPendings(CArrayObj* list);
   CArrayObj*        GetListDeals(CArrayObj* list);
   CArrayObj*        GetListCloseByOrders(CArrayObj* list);
//--- Select and return the list of (1) all position orders by its ID, (2) all deal positions by its ID
//--- (3) all market entry deals by position ID, (4) all market exit deals by position ID
   CArrayObj*        GetListAllOrdersByPosID(CArrayObj* list,const ulong position_id);
   CArrayObj*        GetListAllDealsByPosID(CArrayObj* list,const ulong position_id);
   CArrayObj*        GetListAllDealsInByPosID(CArrayObj* list,const ulong position_id);
   CArrayObj*        GetListAllDealsOutByPosID(CArrayObj* list,const ulong position_id);
//--- Return the total volume of all deals (1) IN, (2) OUT of the position by its ID
   double            SummaryVolumeDealsInByPosID(CArrayObj* list,const ulong position_id);
   double            SummaryVolumeDealsOutByPosID(CArrayObj* list,const ulong position_id);
//--- Return the (1) first, (2) last and (3) closing order from the list of all position order, (4) an order by ticket
   COrder*           GetFirstOrderFromList(CArrayObj* list,const ulong position_id);
   COrder*           GetLastOrderFromList(CArrayObj* list,const ulong position_id);
   COrder*           GetCloseByOrderFromList(CArrayObj* list,const ulong position_id);
   COrder*           GetOrderByTicket(CArrayObj* list,const ulong order_ticket);
//--- Return the flag of the event object presence in the event list
   bool              IsPresentEventInList(CEvent* compared_event);

public:
//--- Select events from the collection with time within the range from begin_time to end_time
   CArrayObj        *GetListByTime(const datetime begin_time=0,const datetime end_time=0);
//--- Return the full event collection list "as is"
   CArrayObj        *GetList(void)                                                                       { return &this.m_list_events;                                           }
//--- Return the list by selected (1) integer, (2) real and (3) string properties meeting the compared criterion
   CArrayObj        *GetList(ENUM_EVENT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByEventProperty(this.GetList(),property,value,mode);  }
   CArrayObj        *GetList(ENUM_EVENT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByEventProperty(this.GetList(),property,value,mode);  }
   CArrayObj        *GetList(ENUM_EVENT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByEventProperty(this.GetList(),property,value,mode);  }
//--- Update the list of events
   void              Refresh(CArrayObj* list_history,
                             CArrayObj* list_market,
                             const bool is_history_event,
                             const bool is_market_event,
                             const int  new_history_orders,
                             const int  new_market_pendings,
                             const int  new_market_positions,
                             const int  new_deals);
//--- Set the control program chart ID
   void              SetChartID(const long id)        { this.m_chart_id=id;         }
//--- Return the last trading event on the account
   ENUM_TRADE_EVENT  GetLastTradeEvent(void)    const { return this.m_trade_event;  }
//--- Reset the last trading event
   void              ResetLastTradeEvent(void)        { this.m_trade_event=TRADE_EVENT_NO_EVENT;   }
//--- Constructor
                     CEventsCollection(void);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CEventsCollection::CEventsCollection(void) : m_trade_event(TRADE_EVENT_NO_EVENT)
  {
   this.m_list_events.Clear();
   this.m_list_events.Sort(SORT_BY_EVENT_TIME_EVENT);
   this.m_list_events.Type(COLLECTION_EVENTS_ID);
   this.m_is_hedge=bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING);
   this.m_chart_id=::ChartID();
  }
//+------------------------------------------------------------------+
```

Reset the trading event in the class constructor's
initialization list,

clear the collection list in the constructor body,


set its sorting by the event time,

set the event collection list ID,

set the hedge account flag and


set the control program chart ID as the current chart.

**Let's consider the methods necessary for the class to work.**

The following class members are declared in the private class section:

```
   CListObj          m_list_events;                   // Event list
   bool              m_is_hedge;                      // Hedge account flag
   long              m_chart_id;                      // Control program chart ID
   ENUM_TRADE_EVENT  m_trade_event;                   // Account trading event
   CEvent            m_event_instance;                // Event object for searching by property
```

The **m\_list\_events** event
list is based on CListObj. It will store events occurring on the account since launching the program. Besides, we will use it to receive the
necessary number of multiple events that occurred at one go.

The **m\_is\_hedge**
**hedge account flag** is used to save and receive the account type. The flag value (account type) defines the block handling the events
occurring on the chart

The **m\_chart\_id**
control program chart ID receives custom events occurring on the account. The ID is sent to event objects and back to the chart. The ID can be set from the
control program using the method created for this purpose.

The **m\_trade\_event** trading
event stores the last event occurred on the account.

**m\_event\_instance** event
object for searching by a property — a special sample object for internal use in the method that returns the list of events with specified
dates of the search range start and end. We have already analyzed a similar method in the

[third part](https://www.mql5.com/en/articles/5687#node01) of the library description when discussing how to arrange the
search in the lists by different criteria.

Here in the private section, you can see the methods necessary for the class operation:

```
//--- Create a trading event depending on the order status
   void              CreateNewEvent(COrder* order,CArrayObj* list_history);
//--- Select and return the list of market pending orders
   CArrayObj*        GetListMarketPendings(CArrayObj* list);
//--- Select and return the list of historical (1) orders, (2) removed pending orders,
//--- (3) deals, (4) all position orders by its ID, (5) all position deals by its ID
//--- (6) all market entry deals by position ID, (7) all market exit deals by position ID
//--- (7) all closing orders
   CArrayObj*        GetListHistoryOrders(CArrayObj* list);
   CArrayObj*        GetListHistoryPendings(CArrayObj* list);
   CArrayObj*        GetListDeals(CArrayObj* list);
   CArrayObj*        GetListAllOrdersByPosID(CArrayObj* list,const ulong position_id);
   CArrayObj*        GetListAllDealsByPosID(CArrayObj* list,const ulong position_id);
   CArrayObj*        GetListAllDealsInByPosID(CArrayObj* list,const ulong position_id);
   CArrayObj*        GetListAllDealsOutByPosID(CArrayObj* list,const ulong position_id);
   CArrayObj*        GetListAllCloseByOrders(CArrayObj* list);
//--- Return the total volume of all deals (1) IN, (2) OUT position by its ID
   double            SummaryVolumeDealsInByPosID(CArrayObj* list,const ulong position_id);
   double            SummaryVolumeDealsOutByPosID(CArrayObj* list,const ulong position_id);
//--- Return the (1) first, (2) last and (3) closing order from the list of all position order, (4) an order by ticket
   COrder*           GetFirstOrderFromList(CArrayObj* list,const ulong position_id);
   COrder*           GetLastOrderFromList(CArrayObj* list,const ulong position_id);
   COrder*           GetCloseByOrderFromList(CArrayObj* list,const ulong position_id);
   COrder*           GetOrderByTicket(CArrayObj* list,const ulong order_ticket);
//--- Return the flag of the event object presence in the event list
   bool              IsPresentEventInList(CEvent* compared_event);
```

The **CreateNewEvent()** method creating a trade event depending on the order status is used in the **Refresh()**
main class method. We will consider it when discussing the Refresh() method.

The methods for receiving the lists of various order types are quite simple — selection by a specified property was discussed [in \\
the third part of the library description](https://www.mql5.com/en/articles/5687). Here we will only briefly mention that some methods consist of several iterations of
selection by necessary properties.

**The method of receiving the list of market pending orders:**

```
//+------------------------------------------------------------------+
//| Select only market pending orders from the list                  |
//+------------------------------------------------------------------+
CArrayObj* CEventsCollection::GetListMarketPendings(CArrayObj* list)
  {
   if(list.Type()!=COLLECTION_MARKET_ID)
     {
      Print(DFUN,TextByLanguage("Ошибка. Список не является списком рыночной коллекции","Error. List is not a list of market collection"));
      return NULL;
     }
   CArrayObj* list_orders=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_PENDING,EQUAL);
   return list_orders;
  }
//+------------------------------------------------------------------+
```

The type of the list passed to the method is checked first. If it is not a list of the market collection, the error message is displayed and the
empty list is returned.

Then orders having the "Market pending order" status are selected from the list passed to the method and the obtained list is returned.

The **methods for receiving lists of removed pending orders,**

**deals and closing**
**orders** placed when closing a position by an opposite one:

```
//+------------------------------------------------------------------+
//| Select only removed pending orders from the list                 |
//+------------------------------------------------------------------+
CArrayObj* CEventsCollection::GetListHistoryPendings(CArrayObj* list)
  {
   if(list.Type()!=COLLECTION_HISTORY_ID)
     {
      Print(DFUN,TextByLanguage("Ошибка. Список не является списком исторической коллекции","Error. The list is not a list of the history collection"));
      return NULL;
     }
   CArrayObj* list_orders=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_HISTORY_PENDING,EQUAL);
   return list_orders;
  }
//+------------------------------------------------------------------+
//| Select only deals from the list                                  |
//+------------------------------------------------------------------+
CArrayObj* CEventsCollection::GetListDeals(CArrayObj* list)
  {
   if(list.Type()!=COLLECTION_HISTORY_ID)
     {
      Print(DFUN,TextByLanguage("Ошибка. Список не является списком исторической коллекции","Error. The list is not a list of the history collection"));
      return NULL;
     }
   CArrayObj* list_deals=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_DEAL,EQUAL);
   return list_deals;
  }
//+------------------------------------------------------------------+
//|  Return the list of all closing CloseBy orders from the list     |
//+------------------------------------------------------------------+
CArrayObj* CEventsCollection::GetListCloseByOrders(CArrayObj *list)
  {
   if(list.Type()!=COLLECTION_HISTORY_ID)
     {
      Print(DFUN,TextByLanguage("Ошибка. Список не является списком исторической коллекции","Error. The list is not a list of the history collection"));
      return NULL;
     }
   CArrayObj* list_orders=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,ORDER_TYPE_CLOSE_BY,EQUAL);
   return list_orders;
  }
//+------------------------------------------------------------------+
```

Just like when returning the list of active pending orders:

The list type is checked and if it is not the history collection one, the message is displayed and NULL is returned.


Then orders having the "Removed pending order" and "Deal" states are selected from the list passed to the method or orders are selected by
ORDER\_TYPE\_CLOSE\_BY type depending on the method, and the obtained list is returned.

**The method for obtaining a list of all orders belonging to a position by its ID:**

```
//+------------------------------------------------------------------+
//|  Return the list of all position orders by its ID                |
//+------------------------------------------------------------------+
CArrayObj* CEventsCollection::GetListAllOrdersByPosID(CArrayObj* list,const ulong position_id)
  {
   CArrayObj* list_orders=CSelect::ByOrderProperty(list,ORDER_PROP_POSITION_ID,position_id,EQUAL);
   list_orders=CSelect::ByOrderProperty(list_orders,ORDER_PROP_STATUS,ORDER_STATUS_DEAL,NO_EQUAL);
   return list_orders;
  }
//+------------------------------------------------------------------+
```

First, using the list passed to the method, form a separate list of all
objects featuring a pointer to the position ID passed to the method by its parameter.

Next, all
deals are removed from the obtained list, and the final list is returned to the calling program. The result of returning the method
may be NULL, therefore we should check what the method returned in the calling program.

**The method for obtaining a list of all deals belonging to a position by its ID:**

```
//+------------------------------------------------------------------+
//| Return the list of all position deals by its ID                  |
//+------------------------------------------------------------------+
CArrayObj* CEventsCollection::GetListAllDealsByPosID(CArrayObj *list,const ulong position_id)
  {
   if(list.Type()!=COLLECTION_HISTORY_ID)
     {
      Print(DFUN,TextByLanguage("Ошибка. Список не является списком исторической коллекции","Error. The list is not a list of the history collection"));
      return NULL;
     }
   CArrayObj* list_deals=CSelect::ByOrderProperty(list,ORDER_PROP_POSITION_ID,position_id,EQUAL);
   list_deals=CSelect::ByOrderProperty(list_deals,ORDER_PROP_STATUS,ORDER_STATUS_DEAL,EQUAL);
   return list_deals;
  }
//+------------------------------------------------------------------+
```

The list type is checked first and if it is not the history collection one, the message is displayed and NULL is returned.

Next, using the
list passed to the method,

form a separate list of all objects featuring a pointer to the position ID
passed to the method by its parameter.

After that, only
deals are left in the obtained list, and the final list is returned to the calling program. The result of returning the method may be
NULL, therefore we should check what the method returned in the calling program.

**The method for obtaining a list of all market entry deals belonging to a position by its ID:**

```
//+------------------------------------------------------------------+
//| Return the list of all market entry deals (IN)                   |
//| by position ID                                                   |
//+------------------------------------------------------------------+
CArrayObj* CEventsCollection::GetListAllDealsInByPosID(CArrayObj *list,const ulong position_id)
  {
   CArrayObj* list_deals=this.GetListAllDealsByPosID(list,position_id);
   list_deals=CSelect::ByOrderProperty(list_deals,ORDER_PROP_DEAL_ENTRY,DEAL_ENTRY_IN,EQUAL);
   return list_deals;
  }
//+------------------------------------------------------------------+
```

First, using the list passed to the method, form a separate list of all
deals featuring a pointer to the position ID passed to the method by its parameter.

Next, only
deals of DEAL\_ENTRY\_IN type are left in the obtained list, and the final list is returned to the calling program. The result of
returning the method may be NULL, therefore we should check what the method returned in the calling program.

**The method for obtaining a list of all market exit deals belonging to a position by its ID:**

```
//+------------------------------------------------------------------+
//| Return the list of all market exit deals (OUT)                   |
//| by position ID                                                   |
//+------------------------------------------------------------------+
CArrayObj* CEventsCollection::GetListAllDealsOutByPosID(CArrayObj *list,const ulong position_id)
  {
   CArrayObj* list_deals=this.GetListAllDealsByPosID(list,position_id);
   list_deals=CSelect::ByOrderProperty(list_deals,ORDER_PROP_DEAL_ENTRY,DEAL_ENTRY_OUT,EQUAL);
   return list_deals;
  }
//+------------------------------------------------------------------+
```

First, using the list passed to the method, form a separate list of all
deals featuring a pointer to the position ID passed to the method by its parameter.

Next, only
deals of DEAL\_ENTRY\_OUT type are left in the obtained list, and the final list is returned to the calling program. The result of
returning the method may be NULL, therefore we should check what exactly the method returned in the calling program.

**The method returning the total volume of all deals of a market entry position by its ID:**

```
//+------------------------------------------------------------------+
//| Return the total volume of all deals of IN position              |
//| by its ID                                                        |
//+------------------------------------------------------------------+
double CEventsCollection::SummaryVolumeDealsInByPosID(CArrayObj *list,const ulong position_id)
  {
   double vol=0.0;
   CArrayObj* list_in=this.GetListAllDealsInByPosID(list,position_id);
   if(list_in==NULL)
      return 0;
   for(int i=0;i<list_in.Total();i++)
     {
      COrder* deal=list_in.At(i);
      if(deal==NULL)
         continue;
      vol+=deal.Volume();
     }
   return vol;
  }
//+------------------------------------------------------------------+
```

First, receive the list of all market entry position deals,
then

sum the volumes of all deals in a loop. The resulting volume is
returned to the calling program. If the list passed to the method is empty, or it is not a history collection list, the method returns zero.

**The method returning the total volume of all deals of a market exit position by its ID:**

```
//+--------------------------------------------------------------------+
//| Return the total volume of all deals of OUT position by its        |
//| ID (participation in closing by an opposite position is considered)|
//+--------------------------------------------------------------------+
double CEventsCollection::SummaryVolumeDealsOutByPosID(CArrayObj *list,const ulong position_id)
  {
   double vol=0.0;
   CArrayObj* list_out=this.GetListAllDealsOutByPosID(list,position_id);
   if(list_out!=NULL)
     {
      for(int i=0;i<list_out.Total();i++)
        {
         COrder* deal=list_out.At(i);
         if(deal==NULL)
            continue;
         vol+=deal.Volume();
        }
     }
   CArrayObj* list_by=this.GetListCloseByOrders(list);
   if(list_by!=NULL)
     {
      for(int i=0;i<list_by.Total();i++)
        {
         COrder* order=list_by.At(i);
         if(order==NULL)
            continue;
         if(order.PositionID()==position_id || order.PositionByID()==position_id)
           {
            vol+=order.Volume();
           }
        }
     }
   return vol;
  }
//+------------------------------------------------------------------+
```

If part of a position, whose ID was passed to the method, participated in closing another position (as an opposite one), or part of the
position was closed by an opposite one, this is not considered in the position deals. Instead, it is considered in the
ORDER\_PROP\_POSITION\_BY\_ID property field of the position's last closing order. Therefore, this method has two searches for closed
volumes — by deals and by closing orders.

First, receive the list of all market exit position deals,
then

sum the volumes of all deals in a loop.

Next, receive
the list of all closing orders present in the historical list, use the loop to check
the belonging of the selected order to the position whose ID was passed to the method. If the selected order participated in closing a
position,

its volume is added to the total one.

The resulting volume is
returned to the calling program. If the list passed to the method is empty, or it is not a history collection list, the method returns zero.

**The method returning the first (opening) position order by its ID:**

```
//+------------------------------------------------------------------+
//| Return the first order from the list of all position orders      |
//+------------------------------------------------------------------+
COrder* CEventsCollection::GetFirstOrderFromList(CArrayObj* list,const ulong position_id)
  {
   CArrayObj* list_orders=this.GetListAllOrdersByPosID(list,position_id);
   if(list_orders==NULL || list_orders.Total()==0) return NULL;
   list_orders.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
   COrder* order=list_orders.At(0);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
```

First, receive the list of all position orders. The obtained
list is

sorted by open time and its
first element is taken. It will be used as the first position order. The obtained order is returned to the calling program. If the lists
are empty, the method returns NULL.

**The method returning the last position order by its ID:**

```
//+------------------------------------------------------------------+
//| Return the last order from the list of all position orders       |
//+------------------------------------------------------------------+
COrder* CEventsCollection::GetLastOrderFromList(CArrayObj* list,const ulong position_id)
  {
   CArrayObj* list_orders=this.GetListAllOrdersByPosID(list,position_id);
   if(list_orders==NULL || list_orders.Total()==0) return NULL;
   list_orders.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
   COrder* order=list_orders.At(list_orders.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
```

First, receive the list of all position orders. The obtained
list is

sorted by open time and its
last element is taken. It will be used as the last position order. The obtained order is returned to the calling program. If the lists
are empty, the method returns NULL.

**The method returning the last closing position order by its ID (ORDER\_TYPE\_CLOSE\_BY type order):**

```
//+------------------------------------------------------------------+
//| Return the last closing order                                    |
//| from the list of all position orders                             |
//+------------------------------------------------------------------+
COrder* CEventsCollection::GetCloseByOrderFromList(CArrayObj *list,const ulong position_id)
  {
   CArrayObj* list_orders=this.GetListAllOrdersByPosID(list,position_id);
   list_orders=CSelect::ByOrderProperty(list_orders,ORDER_PROP_TYPE,ORDER_TYPE_CLOSE_BY,EQUAL);
   if(list_orders==NULL || list_orders.Total()==0) return NULL;
   list_orders.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
   COrder* order=list_orders.At(list_orders.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
```

Since partial closings are possible when closing by an opposite position, and the volumes of the two opposite positions may be unequal, the
closing order may not be the only one within the position orders. Therefore, the method searches for such orders and returns the last one of
them — it is the last order that triggers an event.

First, receive the list of all position orders. Then from the
obtained list,

receive the list containing only closing orders (of ORDER\_TYPE\_CLOSE\_BY type).
The list obtained in such a way is

sorted by open time and its
last element is taken. It will be used as the last closing position order. The obtained order is returned to the calling program. If the
lists are empty, the method returns NULL.

When closing by an opposite position, there may be situations when the library sees two identical events: two positions are closed, and only
one of them has a closing order, plus we have two deals. Therefore, in order not to duplicate the same event in the collection, we should first
check for the presence of exactly the same event in the collection list of events, and if it is not there, add the event to the list.

**The method returning an order by ticket:**

```
//+------------------------------------------------------------------+
//| Return the order by ticket                                       |
//+------------------------------------------------------------------+
COrder* CEventsCollection::GetOrderByTicket(CArrayObj *list,const ulong order_ticket)
  {
   CArrayObj* list_orders=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_DEAL,NO_EQUAL);
   list_orders=CSelect::ByOrderProperty(list_orders,ORDER_PROP_TICKET,order_ticket,EQUAL);
   if(list_orders==NULL || list_orders.Total()==0) return NULL;
   COrder* order=list_orders.At(0);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
```

First, create the list of orders only, then sort
the list by the ticket passed by the method parameter. As a
result, we

return either NULL (when there
is no order with such a ticket), or the ticket number.

**The method returning the event presence in the list is used to check if the event is in the list:**

```
//+------------------------------------------------------------------+
//| Return the flag of the event object presence in the event list   |
//+------------------------------------------------------------------+
bool CEventsCollection::IsPresentEventInList(CEvent *compared_event)
  {
   int total=this.m_list_events.Total();
   if(total==0)
      return false;
   for(int i=total-1;i>=0;i--)
     {
      CEvent* event=this.m_list_events.At(i);
      if(event==NULL)
         continue;
      if(event.IsEqual(compared_event))
         return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

The pointer to the compared event object is passed to the
method.

If the collection list is empty, 'false' is returned immediately
meaning there is no such event in the list. After that, the next event is
taken from the list in a loop and compared to the event passed to the method
using the **IsEqual()** method of the **CEvent** abstract event. If
the method returns 'true', such an event object is present in the event collection list.
Completing the loop or reaching the last method string means there is no event in the list, and 'false' is returned.

**Declare the methods in the public section of the class:**

```
public:
//--- Select events from the collection with time within the range from begin_time to end_time
   CArrayObj        *GetListByTime(const datetime begin_time=0,const datetime end_time=0);
//--- Return the full event collection list "as is"
   CArrayObj        *GetList(void)                                                                       { return &this.m_list_events;                                           }
//--- Return the list by selected (1) integer, (2) real and (3) string properties meeting the compared criterion
   CArrayObj        *GetList(ENUM_EVENT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByEventProperty(this.GetList(),property,value,mode);  }
   CArrayObj        *GetList(ENUM_EVENT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByEventProperty(this.GetList(),property,value,mode);  }
   CArrayObj        *GetList(ENUM_EVENT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByEventProperty(this.GetList(),property,value,mode);  }
//--- Update the list of events
   void              Refresh(CArrayObj* list_history,
                             CArrayObj* list_market,
                             const bool is_history_event,
                             const bool is_market_event,
                             const int  new_history_orders,
                             const int  new_market_pendings,
                             const int  new_market_positions,
                             const int  new_deals);
//--- Set the control program chart ID
   void              SetChartID(const long id)        { this.m_chart_id=id;         }
//--- Return the last trading event on the account
   ENUM_TRADE_EVENT  GetLastTradeEvent(void)    const { return this.m_trade_event;  }
//--- Reset the last trading event
   void              ResetLastTradeEvent(void)        { this.m_trade_event=TRADE_EVENT_NO_EVENT;   }
//--- Constructor
                     CEventsCollection(void);
```

I described the methods for receiving the full list, the lists by a date range and by selected integer, real and string properties in the [third \\
part](https://www.mql5.com/en/articles/5687#node01) of the library description. Here I will show you only the listing of these methods so that you can analyze them on your own.

**The method for receiving the list of events in the specified date range:**

```
//+------------------------------------------------------------------+
//| Select events from the collection with the time                  |
//| within the range from begin_time to end_time                     |
//+------------------------------------------------------------------+
CArrayObj *CEventsCollection::GetListByTime(const datetime begin_time=0,const datetime end_time=0)
  {
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
   this.m_event_instance.SetProperty(EVENT_PROP_TIME_EVENT,begin);
   int index_begin=this.m_list_events.SearchGreatOrEqual(&m_event_instance);
   if(index_begin==WRONG_VALUE)
      return list;
   this.m_event_instance.SetProperty(EVENT_PROP_TIME_EVENT,end);
   int index_end=this.m_list_events.SearchLessOrEqual(&m_event_instance);
   if(index_end==WRONG_VALUE)
      return list;
   for(int i=index_begin; i<=index_end; i++)
      list.Add(this.m_list_events.At(i));
   return list;
  }
//+------------------------------------------------------------------+
```

The main method to be called from the base library object when any of the events occurs is **Refresh()**.


Currently, the method works on hedge accounts for MQL5.



The method receives the pointers to the lists of collections of market and historical orders, deals and positions, as well as the data on the
number of newly appeared or removed orders, open and closed positions and new deals.

Depending on a changed list, the necessary number of orders or deals is taken according to the number of orders/positions/deals in a
loop, and the CreateNewEvent() method for creating events and placing them to the collection list is called for each of them.

Thus, the new event creation method is called for any occurred event, while the event is placed to the collection list and the calling
program is notified of all events by sending a custom message to the calling program's chart.

The **m\_trade\_event** class member variable receives the value of the last occurred event. The **GetLastTradeEvent()** public
method returns the value of the last trading event. There is also the method for resetting the last trading event (similar to GetLastError()
and ResetLastError()).

In addition, there are methods returning the collection list of events both in full, by a time range and specified criteria. The calling
program always knows that an event or several events occurred, and it is possible to request the list of all these events in a required amount
and handle it according to the built-in program's logic.

Let's consider the listings of the **Refresh()** and **CreateNewEvent()** methods.

**The method of updating the event collection list:**

```
//+------------------------------------------------------------------+
//| Update the event list                                            |
//+------------------------------------------------------------------+
void CEventsCollection::Refresh(CArrayObj* list_history,
                                CArrayObj* list_market,
                                const bool is_history_event,
                                const bool is_market_event,
                                const int  new_history_orders,
                                const int  new_market_pendings,
                                const int  new_market_positions,
                                const int  new_deals)
  {
//--- Exit if the lists are empty
   if(list_history==NULL || list_market==NULL)
      return;
//--- In case of a hedging account
   if(this.m_is_hedge)
     {
      //--- If the event is in the market environment
      if(is_market_event)
        {
         //--- if the number of placed pending orders increased
         if(new_market_pendings>0)
           {
            //--- Receive the list of the newly placed pending orders
            CArrayObj* list=this.GetListMarketPendings(list_market);
            if(list!=NULL)
              {
               //--- Sort the new list by order placement time
               list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
               //--- Take the number of orders equal to the number of newly placed ones from the end of the list in a loop (the last N events)
               int total=list.Total(), n=new_market_pendings;
               for(int i=total-1; i>=0 && n>0; i--,n--)
                 {
                  //--- Receive an order from the list, if this is a pending order, set a trading event
                  COrder* order=list.At(i);
                  if(order!=NULL && order.Status()==ORDER_STATUS_MARKET_PENDING)
                     this.CreateNewEvent(order,list_history,list_market);
                 }
              }
           }
        }
      //--- If the event is in the account history
      if(is_history_event)
        {
         //--- If the number of historical orders increased
         if(new_history_orders>0)
           {
            //--- Receive the list of removed pending orders only
            CArrayObj* list=this.GetListHistoryPendings(list_history);
            if(list!=NULL)
              {
               //--- Sort the new list by order removal time
               list.Sort(SORT_BY_ORDER_TIME_CLOSE_MSC);
               //--- Take the number of orders equal to the number of newly removed ones from the end of the list in a loop (the last N events)
               int total=list.Total(), n=new_history_orders;
               for(int i=total-1; i>=0 && n>0; i--,n--)
                 {
                  //--- Receive an order from the list. If this is a removed pending order, set a trading event
                  COrder* order=list.At(i);
                  if(order!=NULL && order.Status()==ORDER_STATUS_HISTORY_PENDING)
                     this.CreateNewEvent(order,list_history,list_market);
                 }
              }
           }
         //--- If the number of deals increased
         if(new_deals>0)
           {
            //--- Receive the list of deals only
            CArrayObj* list=this.GetListDeals(list_history);
            if(list!=NULL)
              {
               //--- Sort the new list by deal time
               list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
               //--- Take the number of deals equal to the number of new ones from the end of the list in a loop (the last N events)
               int total=list.Total(), n=new_deals;
               for(int i=total-1; i>=0 && n>0; i--,n--)
                 {
                  //--- Receive a deal from the list and set a trading event
                  COrder* order=list.At(i);
                  if(order!=NULL)
                     this.CreateNewEvent(order,list_history,list_market);
                 }
              }
           }
        }
     }
   //--- In case of a netting account
   else
     {

     }
  }
//+------------------------------------------------------------------+
```

The simple method listing contains all necessary conditions and actions when these conditions are met. I believe, all is quite transparent
here. Currently, events are handled on a hedging account only.

Let's consider the **method for creating a new event:**

```
//+------------------------------------------------------------------+
//| Create a trading event depending on the order status             |
//+------------------------------------------------------------------+
void CEventsCollection::CreateNewEvent(COrder* order,CArrayObj* list_history,CArrayObj* list_market)
  {
   int trade_event_code=TRADE_EVENT_FLAG_NO_EVENT;
   ENUM_ORDER_STATUS status=order.Status();
//--- Pending order placed
   if(status==ORDER_STATUS_MARKET_PENDING)
     {
      trade_event_code=TRADE_EVENT_FLAG_ORDER_PLASED;
      CEvent* event=new CEventOrderPlased(trade_event_code,order.Ticket());
      if(event!=NULL)
        {
         event.SetProperty(EVENT_PROP_TIME_EVENT,order.TimeOpenMSC());                       // Event time
         event.SetProperty(EVENT_PROP_REASON_EVENT,EVENT_REASON_DONE);                       // Event reason (from the ENUM_EVENT_REASON enumeration)
         event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,order.TypeOrder());                    // Event deal type
         event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,order.Ticket());                     // Event order ticket
         event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,order.TypeOrder());                   // Event order type
         event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,order.TypeOrder());                // Event order type
         event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,order.Ticket());                    // Event order ticket
         event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,order.Ticket());                 // Order ticket
         event.SetProperty(EVENT_PROP_POSITION_ID,order.PositionID());                       // Position ID
         event.SetProperty(EVENT_PROP_POSITION_BY_ID,order.PositionByID());                  // Opposite position ID
         event.SetProperty(EVENT_PROP_MAGIC_ORDER,order.Magic());                            // Order magic number
         event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order.TimeOpenMSC());              // Order time
         event.SetProperty(EVENT_PROP_PRICE_EVENT,order.PriceOpen());                        // Price the event occurred at
         event.SetProperty(EVENT_PROP_PRICE_OPEN,order.PriceOpen());                         // Order placement price
         event.SetProperty(EVENT_PROP_PRICE_CLOSE,order.PriceClose());                       // Order closure price
         event.SetProperty(EVENT_PROP_PRICE_SL,order.StopLoss());                            // StopLoss order price
         event.SetProperty(EVENT_PROP_PRICE_TP,order.TakeProfit());                          // TakeProfit order price
         event.SetProperty(EVENT_PROP_VOLUME_INITIAL,order.Volume());                        // Requested volume
         event.SetProperty(EVENT_PROP_VOLUME_EXECUTED,order.Volume()-order.VolumeCurrent()); // Executed volume
         event.SetProperty(EVENT_PROP_VOLUME_CURRENT,order.VolumeCurrent());                 // Remaining (unexecuted) volume
         event.SetProperty(EVENT_PROP_PROFIT,order.Profit());                                // Profit
         event.SetProperty(EVENT_PROP_SYMBOL,order.Symbol());                                // Order symbol
         //--- Set control program chart ID, decode the event code and set the event type
         event.SetChartID(this.m_chart_id);
         event.SetTypeEvent();
         //--- Add the event object if it is not in the list
         if(!this.IsPresentEventInList(event))
           {
            this.m_list_events.InsertSort(event);
            //--- Send a message about the event and set the value of the last trading event
            event.SendEvent();
            this.m_trade_event=event.TradeEvent();
           }
         //--- If the event is already present in the list, remove a new event object and display a debugging message
         else
           {
            ::Print(DFUN_ERR_LINE,TextByLanguage("Такое событие уже есть в списке","This event already in the list."));
            delete event;
           }
        }
     }
//--- Pending order removed
   if(status==ORDER_STATUS_HISTORY_PENDING)
     {
      trade_event_code=TRADE_EVENT_FLAG_ORDER_REMOVED;
      CEvent* event=new CEventOrderRemoved(trade_event_code,order.Ticket());
      if(event!=NULL)
        {
         ENUM_EVENT_REASON reason=
           (
            order.State()==ORDER_STATE_CANCELED ? EVENT_REASON_CANCEL :
            order.State()==ORDER_STATE_EXPIRED  ? EVENT_REASON_EXPIRED : EVENT_REASON_DONE
           );
         event.SetProperty(EVENT_PROP_TIME_EVENT,order.TimeCloseMSC());             // Event time
         event.SetProperty(EVENT_PROP_REASON_EVENT,reason);                         // Event reason (from the ENUM_EVENT_REASON reason)
         event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,order.TypeOrder());           // Event order type
         event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,order.Ticket());            // Event order ticket
         event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,order.TypeOrder());          // Type of an order that triggered an event deal (the last position order)
         event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,order.TypeOrder());       // Type of an order that triggered a position deal (the first position order)
         event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,order.Ticket());           // Ticket of an order, based on which an event deal is opened (the last position order)
         event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,order.Ticket());        // Ticket of an order, based on which a position deal is opened (the first position order)
         event.SetProperty(EVENT_PROP_POSITION_ID,order.PositionID());              // Position ID
         event.SetProperty(EVENT_PROP_POSITION_BY_ID,order.PositionByID());         // Opposite position ID
         event.SetProperty(EVENT_PROP_MAGIC_ORDER,order.Magic());                   // Order magic number
         event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order.TimeOpenMSC());     // Time of an order, based on which a position deal is opened (the first position order)
         event.SetProperty(EVENT_PROP_PRICE_EVENT,order.PriceOpen());               // Event price
         event.SetProperty(EVENT_PROP_PRICE_OPEN,order.PriceOpen());                // Order open price
         event.SetProperty(EVENT_PROP_PRICE_CLOSE,order.PriceClose());              // Order close price
         event.SetProperty(EVENT_PROP_PRICE_SL,order.StopLoss());                   // StopLoss order price
         event.SetProperty(EVENT_PROP_PRICE_TP,order.TakeProfit());                 // TakeProfit order price
         event.SetProperty(EVENT_PROP_VOLUME_INITIAL,order.Volume());               // Requested volume
         event.SetProperty(EVENT_PROP_VOLUME_EXECUTED,order.Volume()-order.VolumeCurrent()); // Executed volume
         event.SetProperty(EVENT_PROP_VOLUME_CURRENT,order.VolumeCurrent());        // Remaining (unexecuted) volume
         event.SetProperty(EVENT_PROP_PROFIT,order.Profit());                       // Profit
         event.SetProperty(EVENT_PROP_SYMBOL,order.Symbol());                       // Order symbol
         //--- Set the control program chart ID, decode the event code and set the event type
         event.SetChartID(this.m_chart_id);
         event.SetTypeEvent();
         //--- Add the event object if it is not present on the list
         if(!this.IsPresentEventInList(event))
           {
            this.m_list_events.InsertSort(event);
            //--- Send a message about the event and set the last trading event value
            event.SendEvent();
            this.m_trade_event=event.TradeEvent();
           }
         //--- If the event is already in the list, remove the new event object and display the debugging message
         else
           {
            ::Print(DFUN_ERR_LINE,TextByLanguage("Такое событие уже есть в списке","This event already in the list."));
            delete event;
           }
        }
     }
//--- Position opened (__MQL4__)
   if(status==ORDER_STATUS_MARKET_POSITION)
     {
      trade_event_code=TRADE_EVENT_FLAG_POSITION_OPENED;
      CEvent* event=new CEventPositionOpen(trade_event_code,order.Ticket());
      if(event!=NULL)
        {
         event.SetProperty(EVENT_PROP_TIME_EVENT,order.TimeOpen());              // Event time
         event.SetProperty(EVENT_PROP_REASON_EVENT,EVENT_REASON_DONE);           // Event reason (from the ENUM_EVENT_REASON enumeration)
         event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,order.TypeOrder());        // Event deal type
         event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,order.Ticket());         // Event deal ticket
         event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,order.TypeOrder());       // Type of an order, based on which an event deal is opened (the last position order)
         event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,order.TypeOrder());    // Type of an order, based on which a position deal is opened (the first position order)
         event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,order.Ticket());        // Ticket of an order, based on which an event deal is opened (the last position order)
         event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,order.Ticket());     // Ticket of an order, based on which a position deal is opened (the first position order)
         event.SetProperty(EVENT_PROP_POSITION_ID,order.PositionID());           // Position ID
         event.SetProperty(EVENT_PROP_POSITION_BY_ID,order.PositionByID());      // Opposite position ID
         event.SetProperty(EVENT_PROP_MAGIC_ORDER,order.Magic());                // Order/deal/position magic number
         event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order.TimeOpen());     // Time of an order, based on which a position deal is opened (the first position order)
         event.SetProperty(EVENT_PROP_PRICE_EVENT,order.PriceOpen());            // Event price
         event.SetProperty(EVENT_PROP_PRICE_OPEN,order.PriceOpen());             // Order/deal/position open price
         event.SetProperty(EVENT_PROP_PRICE_CLOSE,order.PriceClose());           // Order/deal/position close price
         event.SetProperty(EVENT_PROP_PRICE_SL,order.StopLoss());                // StopLoss position price
         event.SetProperty(EVENT_PROP_PRICE_TP,order.TakeProfit());              // TakeProfit position price
         event.SetProperty(EVENT_PROP_VOLUME_INITIAL,order.Volume());            // Requested volume
         event.SetProperty(EVENT_PROP_VOLUME_EXECUTED,order.Volume());           // Executed volume
         event.SetProperty(EVENT_PROP_VOLUME_CURRENT,order.VolumeCurrent());     // Remaining (unexecuted) volume
         event.SetProperty(EVENT_PROP_PROFIT,order.Profit());                    // Profit
         event.SetProperty(EVENT_PROP_SYMBOL,order.Symbol());                    // Order symbol
         //--- Set the control program chart ID, decode the event code and set the event type
         event.SetChartID(this.m_chart_id);
         event.SetTypeEvent();
         //--- Add the event object if it is not present in the list
         if(!this.IsPresentEventInList(event))
           {
            this.m_list_events.InsertSort(event);
            //--- Send the message about the event and set the value of the last trading event
            event.SendEvent();
            this.m_trade_event=event.TradeEvent();
           }
         //--- If the event is already present in the list, remove the new event object and display the debugging message
         else
           {
            ::Print(DFUN_ERR_LINE,TextByLanguage("Такое событие уже есть в списке","This event already in the list."));
            delete event;
           }
        }
     }
//--- New deal (__MQL5__)
   if(status==ORDER_STATUS_DEAL)
     {
      //--- New balance operation
      if((ENUM_DEAL_TYPE)order.TypeOrder()>DEAL_TYPE_SELL)
        {
         trade_event_code=TRADE_EVENT_FLAG_ACCOUNT_BALANCE;
         CEvent* event=new CEventBalanceOperation(trade_event_code,order.Ticket());
         if(event!=NULL)
           {
            ENUM_EVENT_REASON reason=
              (
               (ENUM_DEAL_TYPE)order.TypeOrder()==DEAL_TYPE_BALANCE ? (order.Profit()>0 ? EVENT_REASON_BALANCE_REFILL : EVENT_REASON_BALANCE_WITHDRAWAL) :
               (ENUM_EVENT_REASON)(order.TypeOrder()+REASON_EVENT_SHIFT)
              );
            event.SetProperty(EVENT_PROP_TIME_EVENT,order.TimeOpenMSC());           // Event time
            event.SetProperty(EVENT_PROP_REASON_EVENT,reason);                      // Event reason (from the ENUM_EVENT_REASON enumeration)
            event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,order.TypeOrder());        // Event deal type
            event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,order.Ticket());         // Event order ticket
            event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,order.TypeOrder());       // Type of an order that triggered an event deal (the last position order)
            event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,order.TypeOrder());    // Type of an order, based on which a position deal is opened (the first position order)
            event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,order.Ticket());        // Ticket of an order, based on which an event deal is opened (the last position order)
            event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,order.Ticket());     // Ticket of an order, based on which a position deal is opened (the first position order)
            event.SetProperty(EVENT_PROP_POSITION_ID,order.PositionID());           // Position ID
            event.SetProperty(EVENT_PROP_POSITION_BY_ID,order.PositionByID());      // Opposite position ID
            event.SetProperty(EVENT_PROP_MAGIC_ORDER,order.Magic());                // Order/deal/position magic number
            event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order.TimeOpenMSC());  // Time of an order, based on which a position deal is opened (the first position order)
            event.SetProperty(EVENT_PROP_PRICE_EVENT,order.PriceOpen());            // Event price
            event.SetProperty(EVENT_PROP_PRICE_OPEN,order.PriceOpen());             // Order/deal/position open price
            event.SetProperty(EVENT_PROP_PRICE_CLOSE,order.PriceClose());           // Order/deal/position close price
            event.SetProperty(EVENT_PROP_PRICE_SL,order.StopLoss());                // StopLoss deal price
            event.SetProperty(EVENT_PROP_PRICE_TP,order.TakeProfit());              // TakeProfit deal price
            event.SetProperty(EVENT_PROP_VOLUME_INITIAL,order.Volume());            // Requested volume
            event.SetProperty(EVENT_PROP_VOLUME_EXECUTED,order.Volume());           // Executed volume
            event.SetProperty(EVENT_PROP_VOLUME_CURRENT,order.VolumeCurrent());     // Remaining (unexecuted) volume
            event.SetProperty(EVENT_PROP_PROFIT,order.Profit());                    // Profit
            event.SetProperty(EVENT_PROP_SYMBOL,order.Symbol());                    // Order symbol
            //--- Set the control program chart ID, decode the event code and set the event type
            event.SetChartID(this.m_chart_id);
            event.SetTypeEvent();
            //--- Add the event object if it is not in the list
            if(!this.IsPresentEventInList(event))
              {
               //--- Send a message about the event and set the last trading event value
               this.m_list_events.InsertSort(event);
               event.SendEvent();
               this.m_trade_event=event.TradeEvent();
              }
            //--- If the event is already in the list, remove the new event object and display the debugging message
            else
              {
               ::Print(DFUN_ERR_LINE,TextByLanguage("Такое событие уже есть в списке","This event already in the list."));
               delete event;
              }
           }
        }
      //--- If this is not a balance operation
      else
        {
         //--- Market entry
         if(order.GetProperty(ORDER_PROP_DEAL_ENTRY)==DEAL_ENTRY_IN)
           {
            trade_event_code=TRADE_EVENT_FLAG_POSITION_OPENED;
            int reason=EVENT_REASON_DONE;
            //--- Look for all position deals in the direction of its opening and count its total volume
            double volume_in=this.SummaryVolumeDealsInByPosID(list_history,order.PositionID());
            //--- Take the first and last position orders from the list of all position orders
            ulong order_ticket=order.GetProperty(ORDER_PROP_DEAL_ORDER);
            COrder* order_first=this.GetOrderByTicket(list_history,order_ticket);
            COrder* order_last=this.GetLastOrderFromList(list_history,order.PositionID());
            //--- If there is no last order, the first and last position orders coincide
            if(order_last==NULL)
               order_last=order_first;
            if(order_first!=NULL)
              {
               //--- If the order volume is opened partially, this is a partial execution
               if(this.SummaryVolumeDealsInByPosID(list_history,order.PositionID())<order_first.Volume())
                 {
                  trade_event_code+=TRADE_EVENT_FLAG_PARTIAL;
                  reason=EVENT_REASON_DONE_PARTIALLY;
                 }
               //--- If an opening order is a pending one, the pending order is activated
               if(order_first.TypeOrder()>ORDER_TYPE_SELL && order_first.TypeOrder()<ORDER_TYPE_CLOSE_BY)
                 {
                  trade_event_code+=TRADE_EVENT_FLAG_ORDER_ACTIVATED;
                  //--- If an order is executed partially, set the partial order execution as an event reason
                  reason=
                    (this.SummaryVolumeDealsInByPosID(list_history,order.PositionID())<order_first.Volume() ?
                     EVENT_REASON_ACTIVATED_PENDING_PARTIALLY :
                     EVENT_REASON_ACTIVATED_PENDING
                    );
                 }
               CEvent* event=new CEventPositionOpen(trade_event_code,order.PositionID());
               if(event!=NULL)
                 {
                  event.SetProperty(EVENT_PROP_TIME_EVENT,order.TimeOpenMSC());                 // Event time (position open time)
                  event.SetProperty(EVENT_PROP_REASON_EVENT,reason);                            // Event reason (from the ENUM_EVENT_REASON enumeration)
                  event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,order.TypeOrder());              // Event deal type
                  event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,order.Ticket());               // Event deal ticket
                  event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,order_first.TypeOrder());    // Type of an order, based on which a position deal is opened (the first position order)
                  event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,order_first.Ticket());     // Ticket of an order, based on which an event deal is opened (the last position order)
                  event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,order_last.TypeOrder());        // Type of an order, based on which an event deal is opened (the last position order)
                  event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,order_last.Ticket());         // Ticket of an order, based on which an event deal is opened (the last position order)
                  event.SetProperty(EVENT_PROP_POSITION_ID,order.PositionID());                 // Position ID
                  event.SetProperty(EVENT_PROP_POSITION_BY_ID,order_last.PositionByID());       // Opposite position ID
                  event.SetProperty(EVENT_PROP_MAGIC_ORDER,order.Magic());                      // Order/deal/position magic number
                  event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order_first.TimeOpenMSC());  // Time of an order, based on which a position deal is opened (the first position order)
                  event.SetProperty(EVENT_PROP_PRICE_EVENT,order.PriceOpen());                  // Event price (position open price)
                  event.SetProperty(EVENT_PROP_PRICE_OPEN,order_first.PriceOpen());             // Order open price (position opening order price)
                  event.SetProperty(EVENT_PROP_PRICE_CLOSE,order_last.PriceClose());            // Order close price (position last order close price)
                  event.SetProperty(EVENT_PROP_PRICE_SL,order_first.StopLoss());                // StopLoss price (Position order StopLoss price)
                  event.SetProperty(EVENT_PROP_PRICE_TP,order_first.TakeProfit());              // TakeProfit price (Position order TakeProfit price)
                  event.SetProperty(EVENT_PROP_VOLUME_INITIAL,order_first.Volume());            // Requested volume
                  event.SetProperty(EVENT_PROP_VOLUME_EXECUTED,volume_in);                      // Executed volume
                  event.SetProperty(EVENT_PROP_VOLUME_CURRENT,order_first.Volume()-volume_in);  // Remaining (unexecuted) volume
                  event.SetProperty(EVENT_PROP_PROFIT,order.ProfitFull());                      // Profit
                  event.SetProperty(EVENT_PROP_SYMBOL,order.Symbol());                          // Order symbol
                  //--- Set the control program chart ID, decode the event code and set the event type
                  event.SetChartID(this.m_chart_id);
                  event.SetTypeEvent();
                  //--- Add the event object if it is not on the list
                  if(!this.IsPresentEventInList(event))
                    {
                     this.m_list_events.InsertSort(event);
                     //--- Send a message about the event and set the last trading event value
                     event.SendEvent();
                     this.m_trade_event=event.TradeEvent();
                    }
                  //--- If the event is already in the list, remove the new event object and display the debugging message
                  else
                    {
                     ::Print(DFUN_ERR_LINE,TextByLanguage("Такое событие уже есть в списке","This event already in the list."));
                     delete event;
                    }
                 }
              }
           }
         //--- Market exit
         else if(order.GetProperty(ORDER_PROP_DEAL_ENTRY)==DEAL_ENTRY_OUT)
           {
            trade_event_code=TRADE_EVENT_FLAG_POSITION_CLOSED;
            int reason=EVENT_REASON_DONE;
            //--- Take the first and last position orders from the list of all position orders
            COrder* order_first=this.GetFirstOrderFromList(list_history,order.PositionID());
            COrder* order_last=this.GetLastOrderFromList(list_history,order.PositionID());
            if(order_first!=NULL && order_last!=NULL)
              {
               //--- Look for all position deals in the direction of its opening and closing and count their total volume
               double volume_in=this.SummaryVolumeDealsInByPosID(list_history,order.PositionID());
               double volume_out=this.SummaryVolumeDealsOutByPosID(list_history,order.PositionID());
               //--- Calculate the current volume of the closed position
               int dgl=(int)DigitsLots(order.Symbol());
               double volume_current=::NormalizeDouble(volume_in-volume_out,dgl);
               //--- If the order volume is closed partially, this is a partial execution
               if(volume_current>0)
                 {
                  trade_event_code+=TRADE_EVENT_FLAG_PARTIAL;
                 }
               //--- If the closing order is executed partially, set the closing order partial execution as an event reason
               if(order_last.VolumeCurrent()>0)
                 {
                  reason=EVENT_REASON_DONE_PARTIALLY;
                 }
               //--- If the closing flag is set to StopLoss for a position's closing order, then closing is performed by StopLoss
               //--- If a StopLoss order is executed partially, set partial StopLoss order execution as the event reason
               if(order_last.IsCloseByStopLoss())
                 {
                  trade_event_code+=TRADE_EVENT_FLAG_SL;
                  reason=(order_last.VolumeCurrent()>0 ? EVENT_REASON_DONE_SL_PARTIALLY : EVENT_REASON_DONE_SL);
                 }
               //--- If the closing flag is set to TakeProfit for a position's closing order, then closing is performed by TakeProfit
               //--- If a TakeProfit order is executed partially, set partial TakeProfit order execution as the event reason
               else if(order_last.IsCloseByTakeProfit())
                 {
                  trade_event_code+=TRADE_EVENT_FLAG_TP;
                  reason=(order_last.VolumeCurrent()>0 ? EVENT_REASON_DONE_TP_PARTIALLY : EVENT_REASON_DONE_TP);
                 }
               //---
               CEvent* event=new CEventPositionClose(trade_event_code,order.PositionID());
               if(event!=NULL)
                 {
                  event.SetProperty(EVENT_PROP_TIME_EVENT,order.TimeOpenMSC());                 // Event time (position closing time)
                  event.SetProperty(EVENT_PROP_REASON_EVENT,reason);                            // Event reason (from the ENUM_EVENT_REASON enumeration)
                  event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,order.TypeOrder());              // Event deal type
                  event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,order.Ticket());               // Event deal ticket
                  event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,order_first.TypeOrder());    // Type of an order, based on which a position deal is opened (the first position order)
                  event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,order_last.TypeOrder());        // Type of an order, based on which an event deal is opened (the last position order)
                  event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,order_first.Ticket());     // Ticket of an order, based on which a position deal is opened (the first position order)
                  event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,order_last.Ticket());         // Ticket of an order, based on which an event deal is opened (the last position order)
                  event.SetProperty(EVENT_PROP_POSITION_ID,order.PositionID());                 // Position ID
                  event.SetProperty(EVENT_PROP_POSITION_BY_ID,order_last.PositionByID());       // Opposite position ID
                  event.SetProperty(EVENT_PROP_MAGIC_ORDER,order.Magic());                      // Order/deal/position magic number
                  event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order_first.TimeOpenMSC());  // Time of an order, based on which a position deal is opened (the first position order)
                  event.SetProperty(EVENT_PROP_PRICE_EVENT,order.PriceOpen());                  // Event price (position closing price)
                  event.SetProperty(EVENT_PROP_PRICE_OPEN,order_first.PriceOpen());             // Order open price (position opening order price)
                  event.SetProperty(EVENT_PROP_PRICE_CLOSE,order_last.PriceClose());            // Order close price (position last order closing price)
                  event.SetProperty(EVENT_PROP_PRICE_SL,order_first.StopLoss());                // StopLoss price (Position order StopLoss price)
                  event.SetProperty(EVENT_PROP_PRICE_TP,order_first.TakeProfit());              // TakeProfit price (Position order TakeProfit price)
                  event.SetProperty(EVENT_PROP_VOLUME_INITIAL,volume_in);                       // Initial volume
                  event.SetProperty(EVENT_PROP_VOLUME_EXECUTED,order.Volume());                 // Closed volume
                  event.SetProperty(EVENT_PROP_VOLUME_CURRENT,volume_in-volume_out);            // Remaining (current) volume
                  event.SetProperty(EVENT_PROP_PROFIT,order.ProfitFull());                      // Profit
                  event.SetProperty(EVENT_PROP_SYMBOL,order.Symbol());                          // Order symbol
                  //--- Set the control program chart ID, decode the event code and set the event type
                  event.SetChartID(this.m_chart_id);
                  event.SetTypeEvent();
                  //--- Add the event object if it is not on the list
                  if(!this.IsPresentEventInList(event))
                    {
                     this.m_list_events.InsertSort(event);
                     //--- Send a message about the event and set the last trading event value
                     event.SendEvent();
                     this.m_trade_event=event.TradeEvent();
                    }
                  //--- If the event is already in the list, remove the new event object and display the debugging message
                  else
                    {
                     ::Print(DFUN_ERR_LINE,TextByLanguage("Такое событие уже есть в списке","This event already in the list."));
                     delete event;
                    }
                 }
              }
           }
         //--- Opposite position
         else if(order.GetProperty(ORDER_PROP_DEAL_ENTRY)==DEAL_ENTRY_OUT_BY)
           {
            trade_event_code=TRADE_EVENT_FLAG_POSITION_CLOSED;
            int reason=EVENT_REASON_DONE_BY_POS;
            //--- Take the first and last position orders from the list of all position orders
            COrder* order_first=this.GetFirstOrderFromList(list_history,order.PositionID());
            COrder* order_close=this.GetCloseByOrderFromList(list_history,order.PositionID());
            if(order_first!=NULL && order_close!=NULL)
              {
               //--- Add the flag of closing by an opposite position
               trade_event_code+=TRADE_EVENT_FLAG_BY_POS;
               //--- Look for all closed position deals in the direction of its opening and closing and count their total volume
               double volume_in=this.SummaryVolumeDealsInByPosID(list_history,order.PositionID());
               double volume_out=this.SummaryVolumeDealsOutByPosID(list_history,order.PositionID());//+order_close.Volume();
               //--- Calculate the current volume of the closed position
               int dgl=(int)DigitsLots(order.Symbol());
               double volume_current=::NormalizeDouble(volume_in-volume_out,dgl);
               //--- Look for all opposite position deals in the direction of its opening and closing and calculate their total volume
               double volume_opp_in=this.SummaryVolumeDealsInByPosID(list_history,order_close.PositionByID());
               double volume_opp_out=this.SummaryVolumeDealsOutByPosID(list_history,order_close.PositionByID());//+order_close.Volume();
               //--- Calculate the current volume of the opposite position
               double volume_opp_current=::NormalizeDouble(volume_opp_in-volume_opp_out,dgl);
               //--- If the closed position volume is closed partially, this is a partial closing
               if(volume_current>0 || order_close.VolumeCurrent()>0)
                 {
                  //--- Add the partial closing flag
                  trade_event_code+=TRADE_EVENT_FLAG_PARTIAL;
                  //--- If the opposite position is closed partially, there is a partial closing by the part of the opposite position volume
                  reason=(volume_opp_current>0 ? EVENT_REASON_DONE_PARTIALLY_BY_POS_PARTIALLY : EVENT_REASON_DONE_PARTIALLY_BY_POS);
                 }
               //--- If the position volume is closed in full and there is a partial execution by the opposite one, there is a closing by the part of the opposite position volume
               else
                 {
                  if(volume_opp_current>0)
                    {
                     reason=EVENT_REASON_DONE_BY_POS_PARTIALLY;
                    }
                 }
               CEvent* event=new CEventPositionClose(trade_event_code,order.PositionID());
               if(event!=NULL)
                 {
                  event.SetProperty(EVENT_PROP_TIME_EVENT,order.TimeOpenMSC());                 // Event time
                  event.SetProperty(EVENT_PROP_REASON_EVENT,reason);                            // Event reason (from the ENUM_EVENT_REASON enumeration)
                  event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,order.TypeOrder());              // Event deal type
                  event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,order.Ticket());               // Event deal ticket
                  event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,order_close.TypeOrder());       // Type of an order, based on which an event deal is opened (the last position order)
                  event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,order_close.Ticket());        // Ticket of an order, based on which an event deal is opened (the last position order)
                  event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order_first.TimeOpenMSC());  // Time of an order, based on which a position deal is opened (the first position order)
                  event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,order_first.TypeOrder());    // Type of an order, based on which a position deal is opened (the first position order)
                  event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,order_first.Ticket());     // Ticket of an order, based on which a position deal is opened (the first position order)
                  event.SetProperty(EVENT_PROP_POSITION_ID,order.PositionID());                 // Position ID
                  event.SetProperty(EVENT_PROP_POSITION_BY_ID,order_close.PositionByID());      // Opposite position ID
                  event.SetProperty(EVENT_PROP_MAGIC_ORDER,order.Magic());                      // Order/deal/position magic number
                  event.SetProperty(EVENT_PROP_PRICE_EVENT,order.PriceOpen());                  // Event price
                  event.SetProperty(EVENT_PROP_PRICE_OPEN,order_first.PriceOpen());             // Order/deal/position open price
                  event.SetProperty(EVENT_PROP_PRICE_CLOSE,order.PriceClose());                 // Order/deal/position close price
                  event.SetProperty(EVENT_PROP_PRICE_SL,order_first.StopLoss());                // StopLoss price (Position order StopLoss price)
                  event.SetProperty(EVENT_PROP_PRICE_TP,order_first.TakeProfit());              // TakeProfit price (Position order TakeProfit price)
                  event.SetProperty(EVENT_PROP_VOLUME_INITIAL,::NormalizeDouble(volume_in,dgl));// Initial volume
                  event.SetProperty(EVENT_PROP_VOLUME_EXECUTED,order.Volume());                 // Closed volume
                  event.SetProperty(EVENT_PROP_VOLUME_CURRENT,volume_current);                  // Remaining (current) volume
                  event.SetProperty(EVENT_PROP_PROFIT,order.ProfitFull());                      // Profit
                  event.SetProperty(EVENT_PROP_SYMBOL,order.Symbol());                          // Order symbol
                  //--- Set the control program chart ID, decode the event code and set the event type
                  event.SetChartID(this.m_chart_id);
                  event.SetTypeEvent();
                  //--- Add the event object if it is not in the list
                  if(!this.IsPresentEventInList(event))
                    {
                     this.m_list_events.InsertSort(event);
                     //--- Send a message about the event and set the value of the last trading event
                     event.SendEvent();
                     this.m_trade_event=event.TradeEvent();
                    }
                  //--- If the event is already present in the list, remove the new event object and display the debugging message
                  else
                    {
                     ::Print(DFUN_ERR_LINE,TextByLanguage("Такое событие уже есть в списке","This event already in the list."));
                     delete event;
                    }
                 }
              }
           }
         //--- Reversal
         else if(order.GetProperty(ORDER_PROP_DEAL_ENTRY)==DEAL_ENTRY_INOUT)
           {
            //--- Position reversal
            Print(DFUN,"Position reversal");
            order.Print();
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

The method turned out to be quite lengthy. Therefore, all descriptions of the necessary checks and corresponding actions are provided
directly in the listing.

The method checks the status of a passed order and all the necessary components of an occurred event depending on its type (placed pending
order, removed pending order, deal). A new event is created and filled with data corresponding to order and event type, while the event
is placed into the event collection, and finally a message about this event is sent to the control program chart and the variable storing
the type of the last occurred event is filled in.

**The events collection class is ready. Now we need to include it to the library's base object.**

After creating the events collection class, some things we did [in \\
the fourth part](https://www.mql5.com/en/articles/5724#node01) of the CEngine base object class to track events are redundant, therefore the base object should be revised.

1. Remove the private member variable of the m\_trade\_event\_code class storing the trading event status code.
2. Remove the private methods:

1. SetTradeEvent() method for decoding an event code,

2. IsTradeEventFlag() method returning the presence of a flag in a trading event,

3. WorkWithHedgeCollections() and WorkWithNettoCollections() methods of working with hedging and netting collections

4. and TradeEventCode() method for returning a trading event code

Add inclusion of the trading event collection class file
to the class body, declare the event collection object, add the
TradeEventsControl() method for working with events to the private class section, change the GetListHistoryDeals() method
name to

**GetListDeals()** in the public section. Deals are always
located in the historical collection so I believe, there is no need to explicitly mention the collection in the method name. Let's
change the implementation of the method for resetting the last trading event: since we now receive the last event from the event
collection class and the method of resetting the last event is present inside the class, we simply need to call the method of the same name
from the event collection class in the

**ResetLastTradeEvent()** method of the class.

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
#include "Collections\MarketCollection.mqh"
#include "Collections\EventsCollection.mqh"
#include "Services\TimerCounter.mqh"
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine : public CObject
  {
private:
   CHistoryCollection   m_history;                       // Collection of historical orders and deals
   CMarketCollection    m_market;                        // Collection of market orders and deals
   CEventsCollection    m_events;                        // Collection of events
   CArrayObj            m_list_counters;                 // List of timer counters
   bool                 m_first_start;                   // First launch flag
   bool                 m_is_hedge;                      // Hedge account flag
   bool                 m_is_market_trade_event;         // Account trading event flag
   bool                 m_is_history_trade_event;        // Account history trading event flag
   ENUM_TRADE_EVENT     m_acc_trade_event;               // Account trading event
//--- Return the counter index by id
   int                  CounterIndex(const int id) const;
//--- Return (1) the first launch flag, (2) flag presence in a trading event
   bool                 IsFirstStart(void);
//--- Working with events
   void                 TradeEventsControl(void);
//--- Return the last (1) market pending order, (2) market order, (3) last position, (4) position by ticket
   COrder*              GetLastMarketPending(void);
   COrder*              GetLastMarketOrder(void);
   COrder*              GetLastPosition(void);
   COrder*              GetPosition(const ulong ticket);
//--- Return the last (1) removed pending order, (2) historical market order, (3) historical order (market or pending one) by its ticket
   COrder*              GetLastHistoryPending(void);
   COrder*              GetLastHistoryOrder(void);
   COrder*              GetHistoryOrder(const ulong ticket);
//--- Return the (1) first and the (2) last historical market orders from the list of all position orders, (3) the last deal
   COrder*              GetFirstOrderPosition(const ulong position_id);
   COrder*              GetLastOrderPosition(const ulong position_id);
   COrder*              GetLastDeal(void);
public:
   //--- Return the list of market (1) positions, (2) pending orders and (3) market orders
   CArrayObj*           GetListMarketPosition(void);
   CArrayObj*           GetListMarketPendings(void);
   CArrayObj*           GetListMarketOrders(void);
   //--- Return the list of historical (1) orders, (2) removed pending orders, (3) deals, (4) all position market orders by its id
   CArrayObj*           GetListHistoryOrders(void);
   CArrayObj*           GetListHistoryPendings(void);
   CArrayObj*           GetListDeals(void);
   CArrayObj*           GetListAllOrdersByPosID(const ulong position_id);
//--- Reset the last trading event
   void                 ResetLastTradeEvent(void)                       { this.m_events.ResetLastTradeEvent(); }
//--- Return the (1) last trading event and (2) hedge account flag
   ENUM_TRADE_EVENT     LastTradeEvent(void)                      const { return this.m_acc_trade_event;       }
   bool                 IsHedge(void)                             const { return this.m_is_hedge;              }
//--- Create the timer account
   void                 CreateCounter(const int id,const ulong frequency,const ulong pause);
//--- Timer
   void                 OnTimer(void);
//--- Constructor/destructor
                        CEngine();
                       ~CEngine();
  };
//+------------------------------------------------------------------+
```

In the CEngine class constructor, add handling the millisecond timer development result. If it is not created, display an appropriate
message in the journal. Next, we are going to develop the class for handling certain errors, set flags visible by a library-based program and
process error situations.

```
//+------------------------------------------------------------------+
//| CEngine constructor                                              |
//+------------------------------------------------------------------+
CEngine::CEngine() : m_first_start(true),m_acc_trade_event(TRADE_EVENT_NO_EVENT)
  {
   ::ResetLastError();
   if(!::EventSetMillisecondTimer(TIMER_FREQUENCY))
      Print(DFUN,"Не удалось создать таймер. Ошибка: ","Could not create timer. Error: ",(string)::GetLastError());
   this.m_list_counters.Sort();
   this.m_list_counters.Clear();
   this.CreateCounter(COLLECTION_COUNTER_ID,COLLECTION_COUNTER_STEP,COLLECTION_PAUSE);
   this.m_is_hedge=bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING);
  }
//+------------------------------------------------------------------+
```

In the class timer, call the **TradeEventsControl()**
method after the timer of orders, deals and positions collection unpauses.

```
//+------------------------------------------------------------------+
//| CEngine timer                                                    |
//+------------------------------------------------------------------+
void CEngine::OnTimer(void)
  {
//--- Timer of historical orders, deals, market orders and positions collections
   int index=this.CounterIndex(COLLECTION_COUNTER_ID);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      //--- If unpaused, work with the collections events
      if(counter!=NULL && counter.IsTimeDone())
        {
         this.TradeEventsControl();
        }
     }
  }
//+------------------------------------------------------------------+
```

Let's improve the method returning a historical order by ticket. Since the historical collection list may contain pending orders, activated
market orders and orders acting as closing ones when closing by an opposite position, we need to consider all order types.

To do this, first, search for an order by ticket in the list of market and closing
orders. If the list is empty, look for a removed pending order with the same ticket. If the list does not contain the order, NULL is
returned. Otherwise, the program returns the first element of the list where the order was found. If failed to receive the order from the
list, NULL is returned.

```
//+------------------------------------------------------------------+
//| Return historical order by its ticket                            |
//+------------------------------------------------------------------+
COrder* CEngine::GetHistoryOrder(const ulong ticket)
  {
   CArrayObj* list=this.GetListHistoryOrders();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TICKET,(long)ticket,EQUAL);
   if(list==NULL || list.Total()==0)
     {
      list=this.GetListHistoryPendings();
      list=CSelect::ByOrderProperty(list,ORDER_PROP_TICKET,(long)ticket,EQUAL);
      if(list==NULL) return NULL;
     }
   COrder* order=list.At(0);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
```

**Let's implement the TradeEventsControl() method for working with account events:**

```
//+------------------------------------------------------------------+
//| Check trading events                                             |
//+------------------------------------------------------------------+
void CEngine::TradeEventsControl(void)
  {
//--- Initialize the code and flags of trading events
   this.m_is_market_trade_event=false;
   this.m_is_history_trade_event=false;
//--- Update the lists
   this.m_market.Refresh();
   this.m_history.Refresh();
//--- Actions during the first launch
   if(this.IsFirstStart())
     {
      this.m_acc_trade_event=TRADE_EVENT_NO_EVENT;
      return;
     }
//--- Check the changes in the market state and account history
   this.m_is_market_trade_event=this.m_market.IsTradeEvent();
   this.m_is_history_trade_event=this.m_history.IsTradeEvent();

//--- In case of any event, send the lists, flags and the number of new orders and deals to the event collection and update it
   if(this.m_is_history_trade_event || this.m_is_market_trade_event)
     {
      this.m_events.Refresh(this.m_history.GetList(),this.m_market.GetList(),
                            this.m_is_history_trade_event,this.m_is_market_trade_event,
                            this.m_history.NewOrders(),this.m_market.NewPendingOrders(),
                            this.m_market.NewMarketOrders(),this.m_history.NewDeals());
      //--- Get the last account trading event
      this.m_acc_trade_event=this.m_events.GetLastTradeEvent();
     }
  }
```

This method is much shorter compared to its predecessor WorkWithHedgeCollections() from the fourth part of the library description.

The method is simple and requires no explanations. The code contains all the comments allowing you to understand its simple logic.

Here is a complete listing of the updated CEngine class:

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
#include "Collections\MarketCollection.mqh"
#include "Collections\EventsCollection.mqh"
#include "Services\TimerCounter.mqh"
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine : public CObject
  {
private:
   CHistoryCollection   m_history;                       // Collection of historical orders and deals
   CMarketCollection    m_market;                        // Collection of market orders and deals
   CEventsCollection    m_events;                        // Collection of events
   CArrayObj            m_list_counters;                 // List of timer counters
   bool                 m_first_start;                   // First launch flag
   bool                 m_is_hedge;                      // Hedge account flag
   bool                 m_is_market_trade_event;         // Account trading event flag
   bool                 m_is_history_trade_event;        // Account history trading event flag
   ENUM_TRADE_EVENT     m_acc_trade_event;               // Account trading event
//--- Return the counter index by id
   int                  CounterIndex(const int id) const;
//--- Return (1) the first launch flag, (2) flag presence in a trading event
   bool                 IsFirstStart(void);
//--- Working with events
   void                 TradeEventsControl(void);
//--- Return the last (1) market pending order, (2) market order, (3) last position, (4) position by ticket
   COrder*              GetLastMarketPending(void);
   COrder*              GetLastMarketOrder(void);
   COrder*              GetLastPosition(void);
   COrder*              GetPosition(const ulong ticket);
//--- Return the last (1) removed pending order, (2) historical market order, (3) historical order (market or pending one) by its ticket
   COrder*              GetLastHistoryPending(void);
   COrder*              GetLastHistoryOrder(void);
   COrder*              GetHistoryOrder(const ulong ticket);
//--- Return the (1) first and the (2) last historical market orders from the list of all position orders, (3) the last deal
   COrder*              GetFirstOrderPosition(const ulong position_id);
   COrder*              GetLastOrderPosition(const ulong position_id);
   COrder*              GetLastDeal(void);
public:
   //--- Return the list of market (1) positions, (2) pending orders and (3) market orders
   CArrayObj*           GetListMarketPosition(void);
   CArrayObj*           GetListMarketPendings(void);
   CArrayObj*           GetListMarketOrders(void);
   //--- Return the list of historical (1) orders, (2) removed pending orders, (3) deals, (4) all position market orders by its id
   CArrayObj*           GetListHistoryOrders(void);
   CArrayObj*           GetListHistoryPendings(void);
   CArrayObj*           GetListDeals(void);
   CArrayObj*           GetListAllOrdersByPosID(const ulong position_id);
//--- Reset the last trading event
   void                 ResetLastTradeEvent(void)                       { this.m_events.ResetLastTradeEvent(); }
//--- Return the (1) last trading event and (2) hedge account flag
   ENUM_TRADE_EVENT     LastTradeEvent(void)                      const { return this.m_acc_trade_event;       }
   bool                 IsHedge(void)                             const { return this.m_is_hedge;              }
//--- Create the timer account
   void                 CreateCounter(const int id,const ulong frequency,const ulong pause);
//--- Timer
   void                 OnTimer(void);
//--- Constructor/destructor
                        CEngine();
                       ~CEngine();
  };
//+------------------------------------------------------------------+
//| CEngine constructor                                              |
//+------------------------------------------------------------------+
CEngine::CEngine() : m_first_start(true),m_acc_trade_event(TRADE_EVENT_NO_EVENT)
  {
   ::ResetLastError();
   if(!::EventSetMillisecondTimer(TIMER_FREQUENCY))
      Print(DFUN,"Не удалось создать таймер. Ошибка: ","Could not create timer. Error: ",(string)::GetLastError());
   this.m_list_counters.Sort();
   this.m_list_counters.Clear();
   this.CreateCounter(COLLECTION_COUNTER_ID,COLLECTION_COUNTER_STEP,COLLECTION_PAUSE);
   this.m_is_hedge=bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING);
  }
//+------------------------------------------------------------------+
//| CEngine destructor                                               |
//+------------------------------------------------------------------+
CEngine::~CEngine()
  {
   ::EventKillTimer();
  }
//+------------------------------------------------------------------+
//| CEngine timer                                                    |
//+------------------------------------------------------------------+
void CEngine::OnTimer(void)
  {
//--- Timer of historical orders, deals, market orders and positions collections
   int index=this.CounterIndex(COLLECTION_COUNTER_ID);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      //--- If unpaused, work with the collections events
      if(counter!=NULL && counter.IsTimeDone())
        {
         this.TradeEventsControl();
        }
     }
  }
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
//| Check trading events                                             |
//+------------------------------------------------------------------+
void CEngine::TradeEventsControl(void)
  {
//--- Initialize trading event code and flags
   this.m_is_market_trade_event=false;
   this.m_is_history_trade_event=false;
//--- Update the lists
   this.m_market.Refresh();
   this.m_history.Refresh();
//--- Actions during the first launch
   if(this.IsFirstStart())
     {
      this.m_acc_trade_event=TRADE_EVENT_NO_EVENT;
      return;
     }
//--- Check the changes in the market status and account history
   this.m_is_market_trade_event=this.m_market.IsTradeEvent();
   this.m_is_history_trade_event=this.m_history.IsTradeEvent();

//--- If there is any event, send the lists, the flags and the number of new orders and deals to the event collection, and update it
   if(this.m_is_history_trade_event || this.m_is_market_trade_event)
     {
      this.m_events.Refresh(this.m_history.GetList(),this.m_market.GetList(),
                            this.m_is_history_trade_event,this.m_is_market_trade_event,
                            this.m_history.NewOrders(),this.m_market.NewPendingOrders(),
                            this.m_market.NewMarketOrders(),this.m_history.NewDeals());
      //--- Receive the last account trading event
      this.m_acc_trade_event=this.m_events.GetLastTradeEvent();
     }
  }
//+------------------------------------------------------------------+
//| Return the list of market positions                              |
//+------------------------------------------------------------------+
CArrayObj* CEngine::GetListMarketPosition(void)
  {
   CArrayObj* list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_POSITION,EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of market pending orders                         |
//+------------------------------------------------------------------+
CArrayObj* CEngine::GetListMarketPendings(void)
  {
   CArrayObj* list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_PENDING,EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of market orders                                 |
//+------------------------------------------------------------------+
CArrayObj* CEngine::GetListMarketOrders(void)
  {
   CArrayObj* list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_ORDER,EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of historical orders                             |
//+------------------------------------------------------------------+
CArrayObj* CEngine::GetListHistoryOrders(void)
  {
   CArrayObj* list=this.m_history.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_HISTORY_ORDER,EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of removed pending orders                        |
//+------------------------------------------------------------------+
CArrayObj* CEngine::GetListHistoryPendings(void)
  {
   CArrayObj* list=this.m_history.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_HISTORY_PENDING,EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of deals                                         |
//+------------------------------------------------------------------+
CArrayObj* CEngine::GetListDeals(void)
  {
   CArrayObj* list=this.m_history.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_DEAL,EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
//|  Return the list of all position orders                          |
//+------------------------------------------------------------------+
CArrayObj* CEngine::GetListAllOrdersByPosID(const ulong position_id)
  {
   CArrayObj* list=this.GetListHistoryOrders();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_POSITION_ID,position_id,EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
//| Return the last position                                         |
//+------------------------------------------------------------------+
COrder* CEngine::GetLastPosition(void)
  {
   CArrayObj* list=this.GetListMarketPosition();
   if(list==NULL) return NULL;
   list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
   COrder* order=list.At(list.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
//| Return position by ticket                                        |
//+------------------------------------------------------------------+
COrder* CEngine::GetPosition(const ulong ticket)
  {
   CArrayObj* list=this.GetListMarketPosition();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TICKET,ticket,EQUAL);
   if(list==NULL) return NULL;
   list.Sort(SORT_BY_ORDER_TICKET);
   COrder* order=list.At(list.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
//| Return the last deal                                             |
//+------------------------------------------------------------------+
COrder* CEngine::GetLastDeal(void)
  {
   CArrayObj* list=this.GetListDeals();
   if(list==NULL) return NULL;
   list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
   COrder* order=list.At(list.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
//| Return the last market pending order                             |
//+------------------------------------------------------------------+
COrder* CEngine::GetLastMarketPending(void)
  {
   CArrayObj* list=this.GetListMarketPendings();
   if(list==NULL) return NULL;
   list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
   COrder* order=list.At(list.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
//| Return the last historical pending order                         |
//+------------------------------------------------------------------+
COrder* CEngine::GetLastHistoryPending(void)
  {
   CArrayObj* list=this.GetListHistoryPendings();
   if(list==NULL) return NULL;
   list.Sort(#ifdef __MQL5__ SORT_BY_ORDER_TIME_OPEN_MSC #else SORT_BY_ORDER_TIME_CLOSE_MSC #endif);
   COrder* order=list.At(list.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
//| Return the last market order                                     |
//+------------------------------------------------------------------+
COrder* CEngine::GetLastMarketOrder(void)
  {
   CArrayObj* list=this.GetListMarketOrders();
   if(list==NULL) return NULL;
   list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
   COrder* order=list.At(list.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
//| Return the last historical market order                          |
//+------------------------------------------------------------------+
COrder* CEngine::GetLastHistoryOrder(void)
  {
   CArrayObj* list=this.GetListHistoryOrders();
   if(list==NULL) return NULL;
   list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
   COrder* order=list.At(list.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
//| Return historical order by its ticket                            |
//+------------------------------------------------------------------+
COrder* CEngine::GetHistoryOrder(const ulong ticket)
  {
   CArrayObj* list=this.GetListHistoryOrders();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TICKET,(long)ticket,EQUAL);
   if(list==NULL || list.Total()==0)
     {
      list=this.GetListHistoryPendings();
      list=CSelect::ByOrderProperty(list,ORDER_PROP_TICKET,(long)ticket,EQUAL);
      if(list==NULL) return NULL;
     }
   COrder* order=list.At(0);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
//| Return the first historical market order                         |
//| from the list of all position orders                             |
//+------------------------------------------------------------------+
COrder* CEngine::GetFirstOrderPosition(const ulong position_id)
  {
   CArrayObj* list=this.GetListAllOrdersByPosID(position_id);
   if(list==NULL) return NULL;
   list.Sort(SORT_BY_ORDER_TIME_OPEN);
   COrder* order=list.At(0);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
//| Return the last historical market order                          |
//| from the list of all position orders                             |
//+------------------------------------------------------------------+
COrder* CEngine::GetLastOrderPosition(const ulong position_id)
  {
   CArrayObj* list=this.GetListAllOrdersByPosID(position_id);
   if(list==NULL) return NULL;
   list.Sort(SORT_BY_ORDER_TIME_OPEN);
   COrder* order=list.At(list.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
```

### Testing the processes of defining, handling and receiving events

Now we are ready to work with events. It is time to prepare the EA for testing and handling event descriptions and sending them to the control
program.

In the terminal directory\\MQL5\\Experts\\TestDoEasy, create the Part05 folder and copy the TestDoEasyPart04.mq5 EA from the previous
part to it under a new name:

**TestDoEasyPart05.mq5**

Change its OnChartEvent() event handler to receive custom events:

```
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
   if(MQLInfoInteger(MQL_TESTER))
      return;
   if(id==CHARTEVENT_OBJECT_CLICK && StringFind(sparam,"BUTT_")>0)
     {
      PressButtonEvents(sparam);
     }
   if(id>=CHARTEVENT_CUSTOM)
     {
      ushort event=ushort(id-CHARTEVENT_CUSTOM);
      Print(DFUN,"id=",id,", event=",EnumToString((ENUM_TRADE_EVENT)event),", lparam=",lparam,", dparam=",DoubleToString(dparam,Digits()),", sparam=",sparam);
     }
  }
//+------------------------------------------------------------------+
```

Here, if the event ID exceeds or is equal to the custom event ID,

receive the event code passed from the library by CEvent class
descendants. When sending a custom event by the

[EventChartCustom()](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom) function specified in the **custom\_event\_id**
function parameter (the one we write our event in), the value of the CHARTEVENT\_CUSTOM constant (equal to 1000) from the [ENUM\_CHART\_EVENT](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents)
enumeration is added to the event value. Therefore, in order to get the event value back, we need to simply subtract CHARTEVENT\_CUSTOM value from the event
ID. After that, we display event data in the terminal journal.

The following data is displayed: the ID ('as is'), event description in the form of the ENUM\_TRADE\_EVENT enumeration value, lparam
value storing the order or position ticket, dparam value storing the event price and sparam value — symbol of an order or a position
participating in the event or account currency name in case the event is a balance operation.

For example:

```
2019.04.06 03:19:54.442 OnChartEvent: id=1001, event=TRADE_EVENT_PENDING_ORDER_PLASED, lparam=375419507, dparam=1.14562, sparam=EURUSD
```

Also, we need to correct the lot calculated for partial closing. It was incorrect in the previous versions of the test EAs, since the value of
unexecuted position volume (VolumeCurrent()) was used for the lot calculation. It is always equal to zero in the tester when opening a
position since the tester does not simulate partial openings. Accordingly, the minimum lot value was taken for closing since the lot
calculation function always adjusted zero to the least acceptable lot value.

Let's find the strings where lot for partial closing is calculated and replace VolumeCurrent() with Volume():

```
               //--- Calculate the closing volume and close half of the Buy position by ticket
               trade.PositionClosePartial(position.Ticket(),NormalizeLot(position.Symbol(),position.Volume()/2.0));
```

```
               //--- Calculate the closing volume and close half of the Sell position by ticket
               trade.PositionClosePartial(position.Ticket(),NormalizeLot(position.Symbol(),position.Volume()/2.0));
```

Only two places in the code — closing the half of the Buy position and closing the half of the Sell position.

Also, add button shift by X and Y axes to the EA inputs for more
convenient location of button sets on the visual tester chart (I shifted the buttons to the right to see order and position tickets in the
visualizer since they could be hidden by the buttons):

```
//--- input variables
input ulong    InpMagic       =  123;  // Magic number
input double   InpLots        =  0.1;  // Lots
input uint     InpStopLoss    =  50;   // StopLoss in points
input uint     InpTakeProfit  =  50;   // TakeProfit in points
input uint     InpDistance    =  50;   // Pending orders distance (points)
input uint     InpDistanceSL  =  50;   // StopLimit orders distance (points)
input uint     InpSlippage    =  0;    // Slippage in points
input double   InpWithdrawal  =  10;   // Withdrawal funds (in tester)
input uint     InpButtShiftX  =  40;   // Buttons X shift
input uint     InpButtShiftY  =  10;   // Buttons Y shift
//--- global variables
```

Let's slightly change the button creation function code:

```
//+------------------------------------------------------------------+
//| Create the buttons panel                                         |
//+------------------------------------------------------------------+
bool CreateButtons(const int shift_x=30,const int shift_y=0)
  {
   int h=18,w=84,offset=2;
   int cx=offset+shift_x,cy=offset+shift_y+(h+1)*(TOTAL_BUTT/2)+2*h+1;
   int x=cx,y=cy;
   int shift=0;
   for(int i=0;i<TOTAL_BUTT;i++)
     {
      x=x+(i==7 ? w+2 : 0);
      if(i==TOTAL_BUTT-3) x=cx;
      y=(cy-(i-(i>6 ? 7 : 0))*(h+1));
      if(!ButtonCreate(butt_data[i].name,x,y,(i<TOTAL_BUTT-3 ? w : w*2+2),h,butt_data[i].text,(i<4 ? clrGreen : i>6 && i<11 ? clrRed : clrBlue)))
        {
         Alert(TextByLanguage("Не удалось создать кнопку \"","Could not create button \""),butt_data[i].text);
         return false;
        }
     }
   ChartRedraw(0);
   return true;
  }
//+------------------------------------------------------------------+
```

and implement calling the function in the EA's OnInit() handler:

```
//--- create buttons
   if(!CreateButtons(InpButtShiftX,InpButtShiftY))
      return INIT_FAILED;
//--- setting trade parameters
```

**The full code of the test EA is provided below:**

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart05.mq5 |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
#include <Trade\Trade.mqh>
//--- enums
enum ENUM_BUTTONS
  {
   BUTT_BUY,
   BUTT_BUY_LIMIT,
   BUTT_BUY_STOP,
   BUTT_BUY_STOP_LIMIT,
   BUTT_CLOSE_BUY,
   BUTT_CLOSE_BUY2,
   BUTT_CLOSE_BUY_BY_SELL,
   BUTT_SELL,
   BUTT_SELL_LIMIT,
   BUTT_SELL_STOP,
   BUTT_SELL_STOP_LIMIT,
   BUTT_CLOSE_SELL,
   BUTT_CLOSE_SELL2,
   BUTT_CLOSE_SELL_BY_BUY,
   BUTT_DELETE_PENDING,
   BUTT_CLOSE_ALL,
   BUTT_PROFIT_WITHDRAWAL
  };
#define TOTAL_BUTT   (17)
//--- structures
struct SDataButt
  {
   string      name;
   string      text;
  };
//--- input variables
input ulong    InpMagic       =  123;  // Magic number
input double   InpLots        =  0.1;  // Lots
input uint     InpStopLoss    =  50;   // StopLoss in points
input uint     InpTakeProfit  =  50;   // TakeProfit in points
input uint     InpDistance    =  50;   // Pending orders distance (points)
input uint     InpDistanceSL  =  50;   // StopLimit orders distance (points)
input uint     InpSlippage    =  0;    // Slippage in points
input double   InpWithdrawal  =  10;   // Withdrawal funds (in tester)
input uint     InpButtShiftX  =  40;   // Buttons X shift
input uint     InpButtShiftY  =  10;   // Buttons Y shift
//--- global variables
CEngine        engine;
CTrade         trade;
SDataButt      butt_data[TOTAL_BUTT];
string         prefix;
double         lot;
double         withdrawal=(InpWithdrawal<0.1 ? 0.1 : InpWithdrawal);
ulong          magic_number;
uint           stoploss;
uint           takeprofit;
uint           distance_pending;
uint           distance_stoplimit;
uint           slippage;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Check account type
   if(!engine.IsHedge())
     {
      Alert(TextByLanguage("Ошибка. Счёт должен быть хеджевым","Error. Account must be hedge"));
      return INIT_FAILED;
     }
//--- set global variables
   prefix=MQLInfoString(MQL_PROGRAM_NAME)+"_";
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
//--- create buttons
   if(!CreateButtons(InpButtShiftX,InpButtShiftY))
      return INIT_FAILED;
//--- setting trade parameters
   trade.SetDeviationInPoints(slippage);
   trade.SetExpertMagicNumber(magic_number);
   trade.SetTypeFillingBySymbol(Symbol());
   trade.SetMarginMode();
   trade.LogLevel(LOG_LEVEL_NO);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- delete objects
   ObjectsDeleteAll(0,prefix);
   Comment("");
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   static ENUM_TRADE_EVENT last_event=WRONG_VALUE;
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer();
      int total=ObjectsTotal(0);
      for(int i=0;i<total;i++)
        {
         string obj_name=ObjectName(0,i);
         if(StringFind(obj_name,prefix+"BUTT_")<0)
            continue;
         PressButtonEvents(obj_name);
        }
     }
   if(engine.LastTradeEvent()!=last_event)
     {
      Comment("\nLast trade event: ",EnumToString(engine.LastTradeEvent()));
      last_event=engine.LastTradeEvent();
     }
  }
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
   if(!MQLInfoInteger(MQL_TESTER))
      engine.OnTimer();
  }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
   if(MQLInfoInteger(MQL_TESTER))
      return;
   if(id==CHARTEVENT_OBJECT_CLICK && StringFind(sparam,"BUTT_")>0)
     {
      PressButtonEvents(sparam);
     }
   if(id>=CHARTEVENT_CUSTOM)
     {
      ushort event=ushort(id-CHARTEVENT_CUSTOM);
      Print(DFUN,"id=",id,", event=",EnumToString((ENUM_TRADE_EVENT)event),", lparam=",lparam,", dparam=",DoubleToString(dparam,Digits()),", sparam=",sparam);
     }
  }
//+------------------------------------------------------------------+
//| Create the buttons panel                                         |
//+------------------------------------------------------------------+
bool CreateButtons(const int shift_x=30,const int shift_y=0)
  {
   int h=18,w=84,offset=2;
   int cx=offset+shift_x,cy=offset+shift_y+(h+1)*(TOTAL_BUTT/2)+2*h+1;
   int x=cx,y=cy;
   int shift=0;
   for(int i=0;i<TOTAL_BUTT;i++)
     {
      x=x+(i==7 ? w+2 : 0);
      if(i==TOTAL_BUTT-3) x=cx;
      y=(cy-(i-(i>6 ? 7 : 0))*(h+1));
      if(!ButtonCreate(butt_data[i].name,x,y,(i<TOTAL_BUTT-3 ? w : w*2+2),h,butt_data[i].text,(i<4 ? clrGreen : i>6 && i<11 ? clrRed : clrBlue)))
        {
         Alert(TextByLanguage("Не удалось создать кнопку \"","Could not create button \""),butt_data[i].text);
         return false;
        }
     }
   ChartRedraw(0);
   return true;
  }
//+------------------------------------------------------------------+
//| Create the button                                                |
//+------------------------------------------------------------------+
bool ButtonCreate(const string name,const int x,const int y,const int w,const int h,const string text,const color clr,const string font="Calibri",const int font_size=8)
  {
   if(ObjectFind(0,name)<0)
     {
      if(!ObjectCreate(0,name,OBJ_BUTTON,0,0,0))
        {
         Print(DFUN,TextByLanguage("не удалось создать кнопку! Код ошибки=","Could not create button! Error code="),GetLastError());
         return false;
        }
      ObjectSetInteger(0,name,OBJPROP_SELECTABLE,false);
      ObjectSetInteger(0,name,OBJPROP_HIDDEN,true);
      ObjectSetInteger(0,name,OBJPROP_XDISTANCE,x);
      ObjectSetInteger(0,name,OBJPROP_YDISTANCE,y);
      ObjectSetInteger(0,name,OBJPROP_XSIZE,w);
      ObjectSetInteger(0,name,OBJPROP_YSIZE,h);
      ObjectSetInteger(0,name,OBJPROP_CORNER,CORNER_LEFT_LOWER);
      ObjectSetInteger(0,name,OBJPROP_ANCHOR,ANCHOR_LEFT_LOWER);
      ObjectSetInteger(0,name,OBJPROP_FONTSIZE,font_size);
      ObjectSetString(0,name,OBJPROP_FONT,font);
      ObjectSetString(0,name,OBJPROP_TEXT,text);
      ObjectSetInteger(0,name,OBJPROP_COLOR,clr);
      ObjectSetString(0,name,OBJPROP_TOOLTIP,"\n");
      ObjectSetInteger(0,name,OBJPROP_BORDER_COLOR,clrGray);
      return true;
     }
   return false;
  }
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
  }
//+------------------------------------------------------------------+
//| Transform enumeration into the button text                       |
//+------------------------------------------------------------------+
string EnumToButtText(const ENUM_BUTTONS member)
  {
   string txt=StringSubstr(EnumToString(member),5);
   StringToLower(txt);
   StringReplace(txt,"buy","Buy");
   StringReplace(txt,"sell","Sell");
   StringReplace(txt,"_limit"," Limit");
   StringReplace(txt,"_stop"," Stop");
   StringReplace(txt,"close_","Close ");
   StringReplace(txt,"2"," 1/2");
   StringReplace(txt,"_by_"," by ");
   StringReplace(txt,"profit_","Profit ");
   StringReplace(txt,"delete_","Delete ");
   return txt;
  }
//+------------------------------------------------------------------+
//| Handle pressing the buttons                                      |
//+------------------------------------------------------------------+
void PressButtonEvents(const string button_name)
  {
   //--- Convert the button name into its string ID
   string button=StringSubstr(button_name,StringLen(prefix));
   //--- If the button is pressed
   if(ButtonState(button_name))
     {
      //--- If the BUTT_BUY button is pressed: Open Buy position
      if(button==EnumToString(BUTT_BUY))
        {
         //--- Get the correct StopLoss and TakeProfit prices relative to StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_BUY,0,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_BUY,0,takeprofit);
         //--- Open Buy position
         trade.Buy(NormalizeLot(Symbol(),lot),Symbol(),0,sl,tp);
        }
      //--- If the BUTT_BUY_LIMIT button is pressed: Set BuyLimit
      else if(button==EnumToString(BUTT_BUY_LIMIT))
        {
         //--- Get the correct order placement price relative to StopLevel
         double price_set=CorrectPricePending(Symbol(),ORDER_TYPE_BUY_LIMIT,distance_pending);
         //--- Get the correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_BUY_LIMIT,price_set,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_BUY_LIMIT,price_set,takeprofit);
         //--- Set BuyLimit order
         trade.BuyLimit(lot,price_set,Symbol(),sl,tp);
        }
      //--- If the BUTT_BUY_STOP button is pressed: Set BuyStop
      else if(button==EnumToString(BUTT_BUY_STOP))
        {
         //--- Get the correct order placement price relative to StopLevel
         double price_set=CorrectPricePending(Symbol(),ORDER_TYPE_BUY_STOP,distance_pending);
         //--- Get the correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_BUY_STOP,price_set,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_BUY_STOP,price_set,takeprofit);
         //--- Set BuyStop order
         trade.BuyStop(lot,price_set,Symbol(),sl,tp);
        }
      //--- If the BUTT_BUY_STOP_LIMIT button is pressed: Set BuyStopLimit
      else if(button==EnumToString(BUTT_BUY_STOP_LIMIT))
        {
         //--- Get the correct BuyStop price relative to StopLevel
         double price_set_stop=CorrectPricePending(Symbol(),ORDER_TYPE_BUY_STOP,distance_pending);
         //--- Calculate BuyLimit order price relative to BuyStop placement level considering StopLevel
         double price_set_limit=CorrectPricePending(Symbol(),ORDER_TYPE_BUY_LIMIT,distance_stoplimit,price_set_stop);
         //--- Get correct StopLoss and TakeProfit prices relative to order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_BUY_STOP,price_set_limit,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_BUY_STOP,price_set_limit,takeprofit);
         //--- Set BuyStopLimit order
         trade.OrderOpen(Symbol(),ORDER_TYPE_BUY_STOP_LIMIT,lot,price_set_limit,price_set_stop,sl,tp);
        }
      //--- If the BUTT_SELL button is pressed: Open Sell position
      else if(button==EnumToString(BUTT_SELL))
        {
         //--- Get the correct StopLoss and TakeProfit prices relative to StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_SELL,0,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_SELL,0,takeprofit);
         //--- Open Sell position
         trade.Sell(lot,Symbol(),0,sl,tp);
        }
      //--- If the BUTT_SELL_LIMIT button is pressed: Set SellLimit
      else if(button==EnumToString(BUTT_SELL_LIMIT))
        {
         //--- Get the correct order placement price relative to StopLevel
         double price_set=CorrectPricePending(Symbol(),ORDER_TYPE_SELL_LIMIT,distance_pending);
         //--- Get the correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_SELL_LIMIT,price_set,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_SELL_LIMIT,price_set,takeprofit);
         //--- Set SellLimit order
         trade.SellLimit(lot,price_set,Symbol(),sl,tp);
        }
      //--- If the BUTT_SELL_STOP button is pressed: Set SellStop
      else if(button==EnumToString(BUTT_SELL_STOP))
        {
         //--- Get the correct order placement price relative to StopLevel
         double price_set=CorrectPricePending(Symbol(),ORDER_TYPE_SELL_STOP,distance_pending);
         //--- Get the correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_SELL_STOP,price_set,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_SELL_STOP,price_set,takeprofit);
         //--- Set SellStop order
         trade.SellStop(lot,price_set,Symbol(),sl,tp);
        }
      //--- If the BUTT_SELL_STOP_LIMIT button is pressed: Set SellStopLimit
      else if(button==EnumToString(BUTT_SELL_STOP_LIMIT))
        {
         //--- Get the correct SellStop order price relative to StopLevel
         double price_set_stop=CorrectPricePending(Symbol(),ORDER_TYPE_SELL_STOP,distance_pending);
         //--- Calculate SellLimit order price relative to SellStop level considering StopLevel
         double price_set_limit=CorrectPricePending(Symbol(),ORDER_TYPE_SELL_LIMIT,distance_stoplimit,price_set_stop);
         //--- Get the correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_SELL_STOP,price_set_limit,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_SELL_STOP,price_set_limit,takeprofit);
         //--- Set SellStopLimit order
         trade.OrderOpen(Symbol(),ORDER_TYPE_SELL_STOP_LIMIT,lot,price_set_limit,price_set_stop,sl,tp);
        }
      //--- If the BUTT_CLOSE_BUY button is pressed: Close Buy with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_BUY))
        {
         //--- Get the list of all open positions
         CArrayObj* list=engine.GetListMarketPosition();
         //--- Select only Buy positions from the list
         list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Buy position with the maximum profit
         int index=CSelect::FindOrderMax(list,ORDER_PROP_PROFIT_FULL);
         if(index>WRONG_VALUE)
           {
            COrder* position=list.At(index);
            if(position!=NULL)
              {
               //--- Get the Buy position ticket and close the position by the ticket
               trade.PositionClose(position.Ticket());
              }
           }
        }
      //--- If the BUTT_CLOSE_BUY2 button is pressed: Close the half of Buy with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_BUY2))
        {
         //--- Get the list of all open positions
         CArrayObj* list=engine.GetListMarketPosition();
         //--- Select only Buy positions from the list
         list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Buy position with the maximum profit
         int index=CSelect::FindOrderMax(list,ORDER_PROP_PROFIT_FULL);
         if(index>WRONG_VALUE)
           {
            COrder* position=list.At(index);
            if(position!=NULL)
              {
               //--- Calculate the closed volume and close the half of the Buy position by the ticket
               trade.PositionClosePartial(position.Ticket(),NormalizeLot(position.Symbol(),position.Volume()/2.0));
              }
           }
        }
      //--- If the BUTT_CLOSE_BUY_BY_SELL button is pressed: Close Buy with the maximum profit by the opposite Sell with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_BUY_BY_SELL))
        {
         //--- Get the list of all open positions
         CArrayObj* list_buy=engine.GetListMarketPosition();
         //--- Select only Buy positions from the list
         list_buy=CSelect::ByOrderProperty(list_buy,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list_buy.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Buy position with the maximum profit
         int index_buy=CSelect::FindOrderMax(list_buy,ORDER_PROP_PROFIT_FULL);
         //--- Get the list of all open positions
         CArrayObj* list_sell=engine.GetListMarketPosition();
         //--- Select only Sell positions from the list
         list_sell=CSelect::ByOrderProperty(list_sell,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list_sell.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Sell position with the maximum profit
         int index_sell=CSelect::FindOrderMax(list_sell,ORDER_PROP_PROFIT_FULL);
         if(index_buy>WRONG_VALUE && index_sell>WRONG_VALUE)
           {
            //--- Select the Buy position with the maximum profit
            COrder* position_buy=list_buy.At(index_buy);
            //--- Select the Sell position with the maximum profit
            COrder* position_sell=list_sell.At(index_sell);
            if(position_buy!=NULL && position_sell!=NULL)
              {
               //--- Close the Buy position by the opposite Sell one
               trade.PositionCloseBy(position_buy.Ticket(),position_sell.Ticket());
              }
           }
        }
      //--- If the BUTT_CLOSE_SELL button is pressed: Close Sell with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_SELL))
        {
         //--- Get the list of all open positions
         CArrayObj* list=engine.GetListMarketPosition();
         //--- Select only Sell positions from the list
         list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Sell position with the maximum profit
         int index=CSelect::FindOrderMax(list,ORDER_PROP_PROFIT_FULL);
         if(index>WRONG_VALUE)
           {
            COrder* position=list.At(index);
            if(position!=NULL)
              {
               //--- Get the Sell position ticket and close the position by the ticket
               trade.PositionClose(position.Ticket());
              }
           }
        }
      //--- If the BUTT_CLOSE_SELL2 button is pressed: Close half of the Sell with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_SELL2))
        {
         //--- Get the list of all open positions
         CArrayObj* list=engine.GetListMarketPosition();
         //--- Select only Sell positions from the list
         list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Sell position with the maximum profit
         int index=CSelect::FindOrderMax(list,ORDER_PROP_PROFIT_FULL);
         if(index>WRONG_VALUE)
           {
            COrder* position=list.At(index);
            if(position!=NULL)
              {
               //--- Calculate the closed volume and close the half of the Sell position by the ticket
               trade.PositionClosePartial(position.Ticket(),NormalizeLot(position.Symbol(),position.Volume()/2.0));
              }
           }
        }
      //--- If the BUTT_CLOSE_SELL_BY_BUY button is pressed: Close Sell with the maximum profit by the opposite Buy with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_SELL_BY_BUY))
        {
         //--- Get the list of all open positions
         CArrayObj* list_sell=engine.GetListMarketPosition();
         //--- Select only Sell positions from the list
         list_sell=CSelect::ByOrderProperty(list_sell,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list_sell.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Sell position with the maximum profit
         int index_sell=CSelect::FindOrderMax(list_sell,ORDER_PROP_PROFIT_FULL);
         //--- Get the list of all open positions
         CArrayObj* list_buy=engine.GetListMarketPosition();
         //--- Select only Buy positions from the list
         list_buy=CSelect::ByOrderProperty(list_buy,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list_buy.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Buy position with the maximum profit
         int index_buy=CSelect::FindOrderMax(list_buy,ORDER_PROP_PROFIT_FULL);
         if(index_sell>WRONG_VALUE && index_buy>WRONG_VALUE)
           {
            //--- Select the Sell position with the maximum profit
            COrder* position_sell=list_sell.At(index_sell);
            //--- Select the Buy position with the maximum profit
            COrder* position_buy=list_buy.At(index_buy);
            if(position_sell!=NULL && position_buy!=NULL)
              {
               //--- Close the Sell position by the opposite Buy one
               trade.PositionCloseBy(position_sell.Ticket(),position_buy.Ticket());
              }
           }
        }
      //--- If the BUTT_CLOSE_ALL is pressed: Close all positions starting with the one with the least profit
      else if(button==EnumToString(BUTT_CLOSE_ALL))
        {
         //--- Get the list of all open positions
         CArrayObj* list=engine.GetListMarketPosition();
         if(list!=NULL)
           {
            //--- Sort the list by profit considering commission and swap
            list.Sort(SORT_BY_ORDER_PROFIT_FULL);
            int total=list.Total();
            //--- In the loop from the position with the least profit
            for(int i=0;i<total;i++)
              {
               COrder* position=list.At(i);
               if(position==NULL)
                  continue;
               //--- close each position by its ticket
               trade.PositionClose(position.Ticket());
              }
           }
        }
      //--- If the BUTT_DELETE_PENDING button is pressed: Remove the first pending order
      else if(button==EnumToString(BUTT_DELETE_PENDING))
        {
         //--- Get the list of all orders
         CArrayObj* list=engine.GetListMarketPendings();
         if(list!=NULL)
           {
            //--- Sort the list by placement time
            list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
            int total=list.Total();
            //--- In the loop from the position with the most amount of time
            for(int i=total-1;i>=0;i--)
              {
               COrder* order=list.At(i);
               if(order==NULL)
                  continue;
               //--- delete the order by its ticket
               trade.OrderDelete(order.Ticket());
              }
           }
        }
      //--- If the BUTT_PROFIT_WITHDRAWAL button is pressed: Withdraw funds from the account
      if(button==EnumToString(BUTT_PROFIT_WITHDRAWAL))
        {
         //--- If the program is launched in the tester
         if(MQLInfoInteger(MQL_TESTER))
           {
            //--- Emulate funds withdrawal
            TesterWithdrawal(withdrawal);
           }
        }
      //--- Wait for 1/10 of a second
      Sleep(100);
      //--- "Unpress" the button and redraw the chart
      ButtonState(button_name,false);
      ChartRedraw();
     }
  }
//+------------------------------------------------------------------+
```

Now we can compile the EA and launch it in the tester. When clicking the buttons, short two-line messages about occurring account events are
displayed in the tester journal.

![](https://c.mql5.com/2/35/2019-04-08_17-44-56.gif)

Entries from the EA event handler are not displayed in the journal since they work outside of the tester. If click on the EA buttons on a demo account,
three lines are displayed in the terminal journal: two lines from the method for displaying short messages of the CEvent class and another
one — from the EA's OnChartEvent() handler.

Below is a sample of displaying a message in the journal when placing
and removing a pending order:

```
- Pending order placed: 2019.04.05 23:19:55.248 -
EURUSD 0.10 Sell Limit #375419507 at price 1.14562
OnChartEvent: id=1001, event=TRADE_EVENT_PENDING_ORDER_PLASED, lparam=375419507, dparam=1.14562, sparam=EURUSD
- Pending order removed: 2019.04.05 23:19:55.248 -
EURUSD 0.10 Sell Limit #375419507 at price 1.14562
OnChartEvent: id=1002, event=TRADE_EVENT_PENDING_ORDER_REMOVED, lparam=375419507, dparam=1.14562, sparam=EURUSD
```

### What's next?

In the next article, we will start adding the functionality for working on MetaTrader 5 netting accounts.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/6211#node00)

**Previous articles within the series:**

[Part 1. Concept, data management.](https://www.mql5.com/en/articles/5654)

[Part \\
2\. Collection of historical orders and deals.](https://www.mql5.com/en/articles/5669)

[Part 3. Collection of market orders \\
and positions, arranging the search.](https://www.mql5.com/en/articles/5687)

[Part 4. Trading events. Concept.](https://www.mql5.com/en/articles/5724)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/6211](https://www.mql5.com/ru/articles/6211)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/6211.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/6211/mql5.zip "Download MQL5.zip")(73.98 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/315267)**
(13)


![Alvaro Arioni](https://c.mql5.com/avatar/2020/8/5F4C397C-E698.jpg)

**[Alvaro Arioni](https://www.mql5.com/en/users/adarioni)**
\|
29 Aug 2020 at 19:05

Hello Artyom, congratulations for the great job! Following the description in the text, it seems that [function](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions")**CHistoryCollection::OrderSearch(…)** may have a **break** missing.

The **for** loop always complete all iterations from **start-1** to **0**, whether it finds the “lost order” or not.

Perhaps, it would be more efficient to include a **break** after finding the “lost order”:

```
ulong CHistoryCollection::OrderSearch(const int start,ENUM_ORDER_TYPE &order_type)
  {
   ulong order_ticket=0;
   for(int i=start-1;i>=0;i--)
     {
      ulong ticket=::HistoryOrderGetTicket(i);
      if(ticket==0)
         continue;
      ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)::HistoryOrderGetInteger(ticket,ORDER_TYPE);
      if(this.IsPresentOrderInList(ticket,type))
         continue;
      order_ticket=ticket;
      order_type=type;
      break;
     }
   return order_ticket;
  }
```

What do you think?

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
30 Aug 2020 at 00:08

**Alvaro Arioni :**

Hello Artyom, congratulations for the great job! Following the description in the text, it seems that function **CHistoryCollection::OrderSearch(…)** may have a **break** missing.

The **for** loop always complete all iterations from **start-1** to **0**, whether it finds the “lost order” or not.

Perhaps, it would be more efficient to include a **break** after finding the “lost order”:

...

What do you think?

There may be more than one lost order

![Keith Watford](https://c.mql5.com/avatar/avatar_na2.png)

**[Keith Watford](https://www.mql5.com/en/users/forexample)**
\|
30 Aug 2020 at 02:03

**Alvaro Arioni:**

Please **_edit_** your post and

use the code button (Alt+S) when pasting code

![Alvaro Arioni](https://c.mql5.com/avatar/2020/8/5F4C397C-E698.jpg)

**[Alvaro Arioni](https://www.mql5.com/en/users/adarioni)**
\|
30 Aug 2020 at 16:36

**Artyom Trishkin:**

There may be more than one lost order

OK, but the [function](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") is still returning just one of the lost orders, that which is closest to zero.

In this case, wouldn’t be the same iterating from 0 to start-1 with a break?

```
ulong CHistoryCollection::OrderSearch(const int start,ENUM_ORDER_TYPE &order_type)
  {
   ulong order_ticket=0;
   for(int i=0; i < start ;i++)
     {
      ulong ticket=::HistoryOrderGetTicket(i);
      if(ticket==0)
         continue;
      ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)::HistoryOrderGetInteger(ticket,ORDER_TYPE);
      if(this.IsPresentOrderInList(ticket,type))
         continue;
      order_ticket=ticket;
      order_type=type;
      break;
     }
   return order_ticket;
  }
```

I know that it is just a detail, but I thought this could be a little improvement, especially in case of a too big history of orders.

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
30 Aug 2020 at 21:06

**Alvaro Arioni :**

OK, but the [function](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") is still returning just one of the lost orders, that which is closest to zero.

In this case, wouldn’t be the same iterating from 0 to start-1 with a break?

I know that it is just a detail, but I thought this could be a little improvement, especially in case of a too big history of orders.

Ok, thanks, I'll check and test everything.

It may well be that "break" will come in handy.

![Developing graphical interfaces based on .Net Framework and C# (part 2): Additional graphical elements](https://c.mql5.com/2/36/icon.png)[Developing graphical interfaces based on .Net Framework and C# (part 2): Additional graphical elements](https://www.mql5.com/en/articles/6549)

The article is a follow-up of the previous publication "Developing graphical interfaces for Expert Advisors and indicators based on .Net Framework and C#". It introduces new graphical elements for creating graphical interfaces.

![Selection and navigation utility in MQL5 and MQL4: Adding data to charts](https://c.mql5.com/2/35/Select_Symbols_Utility_MQL5__2.png)[Selection and navigation utility in MQL5 and MQL4: Adding data to charts](https://www.mql5.com/en/articles/5614)

In this article, we will continue expanding the functionality of the utility. This time, we will add the ability to display data that simplifies our trading. In particular, we are going to add High and Low prices of the previous day, round levels, High and Low prices of the year, session start time, etc.

![Applying OLAP in trading (part 1): Online analysis of multidimensional data](https://c.mql5.com/2/36/OLAP_02.png)[Applying OLAP in trading (part 1): Online analysis of multidimensional data](https://www.mql5.com/en/articles/6602)

The article describes how to create a framework for the online analysis of multidimensional data (OLAP), as well as how to implement this in MQL and to apply such analysis in the MetaTrader environment using the example of trading account history processing.

![Library for easy and quick development of MetaTrader programs (part IV): Trading events](https://c.mql5.com/2/35/MQL5-avatar-doeasy__3.png)[Library for easy and quick development of MetaTrader programs (part IV): Trading events](https://www.mql5.com/en/articles/5724)

In the previous articles, we started creating a large cross-platform library simplifying the development of programs for MetaTrader 5 and MetaTrader 4 platforms. We already have collections of historical orders and deals, market orders and positions, as well as the class for convenient selection and sorting of orders. In this part, we will continue the development of the base object and teach the Engine Library to track trading events on the account.

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/6211&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070494929509095135)

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