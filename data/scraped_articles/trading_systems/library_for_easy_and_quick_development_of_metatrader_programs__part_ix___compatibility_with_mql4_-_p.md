---
title: Library for easy and quick development of MetaTrader programs (part IX): Compatibility with MQL4 - Preparing data
url: https://www.mql5.com/en/articles/6651
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:41:22.809510
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/6651&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070484900760458928)

MetaTrader 5 / Examples


### Contents

- [MQL4 vs MQL5](https://www.mql5.com/en/articles/6651#node01)
- [Improving the library](https://www.mql5.com/en/articles/6651#node02)
- [Testing](https://www.mql5.com/en/articles/6651#node03)
- [What's next?](https://www.mql5.com/en/articles/6651#node04)


In the previous parts of the article series, we prepared the following tools for the [MetaTrader \\
5](https://www.metaquotes.net/en/metatrader5 "https://www.metaquotes.net/en/metatrader5") and MetaTrader 4 cross-platform library:

- for creating user-case functions enabling fast access from programs to any data on any orders and positions on hedging and netting
accounts,
- for tracking events occurring to orders and positions — placing, removing and activating pending orders, as well as opening, closing
and modifying positions and orders.

Now it is time to implement the library compatibility with MQL4, since we are going to develop trading classes, and the library should work
correctly both in MQL5 and MQL4.

In this article, we will start improving the library to implement its cross-platform nature.

### MQL4 vs MQL5

Copy the entire library folder to the appropriate MetaTrader 4 directory **\\MQL4\\Include\\DoEasy**. We will take test
EAs from the appropriate folders containing MQL5 EAs and save them with the \*.mq4 extension to the

**\\MQL4\\Experts\\TestDoEasy** EA directory (to the folder corresponding to the article number, which in this case is **Part09**).

Find the library directory \\MQL4\\Include\\DoEasy in the Editor's Navigator, right-click on it and select Compile.

![](https://c.mql5.com/2/36/CompileAll.png)

This will compile all library files resulting in over two thousand compilation errors:

![](https://c.mql5.com/2/36/CompileErrors.png)

If we analyze the obtained errors, we will see that their vast majority has to do with MQL5 [constants \\
and enumerations](https://www.mql5.com/en/docs/constants) MQL4 knows nothing about. This means we need to let MQL4 know about the constants used in the library. There are also the
errors of different nature, like the absence of certain functions, which means we will implement their operation logic using MQL4
functions.

Besides, the order systems of MQL4 and MQL5 are very different. We will have to implement a separate event handler for MQL4 different
from the one implemented in MQL5, since the list of historical orders in MQL4 provides much less data on orders (and no data on deals), which
means we cannot take data on orders and deals directly from the terminal lists. Here we will have to logically compare the occurring events in
the lists of active and historical market orders and define occurred events based on the comparison.

### Improving the library

In the **DoEasy** library root folder, create the new **ToMQL4.mqh** include file. Here we will describe all necessary
constants and enumerations for MQL4.

Include it to the Defines.mqh file
for MQL4 compilation at the very beginning of the Defines.mqh listing:

```
//+------------------------------------------------------------------+
//|                                                      Defines.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#ifdef __MQL4__
#include "ToMQL4.mqh"
#endif
//+------------------------------------------------------------------+
```

After that, the entire MQL4 library will be able to see what is written in the ToMQL4.mqh file during compilation.

Let's move to the very beginning of the error list in the Errors tab of the Editor's Toolbox by pressing NumPad Home or by simply scrolling up to the
very start. Double-click the very first error:

![](https://c.mql5.com/2/36/ClickTo.png)

The editor moves us to the error string in the Defines.mqh
file:

```
//+------------------------------------------------------------------+
//| List of possible trading events on the account                   |
//+------------------------------------------------------------------+
enum ENUM_TRADE_EVENT
  {
   TRADE_EVENT_NO_EVENT = 0,                                // No trading event
   TRADE_EVENT_PENDING_ORDER_PLASED,                        // Pending order placed
   TRADE_EVENT_PENDING_ORDER_REMOVED,                       // Pending order removed
//--- enumeration members matching the ENUM_DEAL_TYPE enumeration members
//--- (constant order below should not be changed, no constants should be added/deleted)
   TRADE_EVENT_ACCOUNT_CREDIT = DEAL_TYPE_CREDIT,           // Charging credit (3)
   TRADE_EVENT_ACCOUNT_CHARGE,                              // Additional charges
```

Naturally, MQL4 does not know anything about deals and their types. This should be fixed. Simply open the MQL5 Reference and search for data on [deal \\
properties](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties) using the DEAL\_TYPE\_CREDIT query:

|     |     |     |
| --- | --- | --- |
| ID | Description | Type |
| DEAL\_TICKET | Deal ticket. Unique number assigned to each deal | long |
| DEAL\_ORDER | Deal [order number](https://www.mql5.com/en/docs/trading/historyordergetticket) | long |
| DEAL\_TIME | Deal time | datetime |
| DEAL\_TIME\_MSC | The time of a deal execution in milliseconds since 01.01.1970 | long |
| DEAL\_TYPE | Deal type | [ENUM\_DEAL\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_type) |
| DEAL\_ENTRY | Deal direction - market entry, exit or reversal | [ENUM\_DEAL\_ENTRY](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_entry) |
| DEAL\_MAGIC | Deal magic number (see [ORDER\_MAGIC](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties)) | long |
| DEAL\_REASON | The reason or source for deal execution | ENUM\_DEAL\_REASON |
| DEAL\_POSITION\_ID | [The \<br>ID of the position](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_integer), which the deal opened, modified or closed. Each position has a unique ID that is assigned to all <br>deals executed on the symbol during the position lifetime. | long |

In the table, we are most interested in [ENUM\_DEAL\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_type).
Follow the link and obtain the list of all deal types:

|     |     |
| --- | --- |
| ID | Description |
| DEAL\_TYPE\_BUY | Buy |
| DEAL\_TYPE\_SELL | Sell |
| DEAL\_TYPE\_BALANCE | Balance |
| DEAL\_TYPE\_CREDIT | Credit |
| DEAL\_TYPE\_CHARGE | Additional charge |
| DEAL\_TYPE\_CORRECTION | Correction |
| DEAL\_TYPE\_BONUS | Bonus |
| DEAL\_TYPE\_COMMISSION | Additional commission |
| DEAL\_TYPE\_COMMISSION\_DAILY | Daily commission |
| DEAL\_TYPE\_COMMISSION\_MONTHLY | Monthly commission |
| DEAL\_TYPE\_COMMISSION\_AGENT\_DAILY | Daily agent commission |
| DEAL\_TYPE\_COMMISSION\_AGENT\_MONTHLY | Monthly agent commission |
| DEAL\_TYPE\_INTEREST | Interest rate |
| DEAL\_TYPE\_BUY\_CANCELED | Canceled buy deal. There <br>can be a situation when a previously executed buy deal is canceled. In this case, the type of the previously executed <br>deal (DEAL\_TYPE\_BUY) is <br>changed to DEAL\_TYPE\_BUY\_CANCELED, <br>and its profit/loss is zeroized. Previously obtained profit/loss is charged/withdrawn using a separated balance <br>operation |
| DEAL\_TYPE\_SELL\_CANCELED | Canceled sell deal. There <br>can be a situation when a previously executed sell deal is canceled. In this case, the type of the previously executed <br>deal (DEAL\_TYPE\_SELL) is <br>changed to DEAL\_TYPE\_SELL\_CANCELED, <br>and its profit/loss is zeroized. Previously obtained profit/loss is charged/withdrawn using a separated balance <br>operation |
| DEAL\_DIVIDEND | Dividend operations |
| DEAL\_DIVIDEND\_FRANKED | Franked (non-taxable) dividend operations |
| DEAL\_TAX | Tax charges |

**Add the deal types from the ENUM\_DEAL\_TYPE enumeration to the ToMQL4.mqh file:**

```
//+------------------------------------------------------------------+
//|                                                       ToMQL4.mqh |
//|              Copyright 2017, Artem A. Trishkin, Skype artmedia70 |
//|                         https://www.mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, Artem A. Trishkin, Skype artmedia70"
#property link      "https://www.mql5.com/en/users/artmedia70"
#property strict
#ifdef __MQL4__
//+------------------------------------------------------------------+
//| MQL5 deal types                                                  |
//+------------------------------------------------------------------+
enum ENUM_DEAL_TYPE
  {
   DEAL_TYPE_BUY,
   DEAL_TYPE_SELL,
   DEAL_TYPE_BALANCE,
   DEAL_TYPE_CREDIT,
   DEAL_TYPE_CHARGE,
   DEAL_TYPE_CORRECTION,
   DEAL_TYPE_BONUS,
   DEAL_TYPE_COMMISSION,
   DEAL_TYPE_COMMISSION_DAILY,
   DEAL_TYPE_COMMISSION_MONTHLY,
   DEAL_TYPE_COMMISSION_AGENT_DAILY,
   DEAL_TYPE_COMMISSION_AGENT_MONTHLY,
   DEAL_TYPE_INTEREST,
   DEAL_TYPE_BUY_CANCELED,
   DEAL_TYPE_SELL_CANCELED,
   DEAL_DIVIDEND,
   DEAL_DIVIDEND_FRANKED,
   DEAL_TAX
  };
//+------------------------------------------------------------------+
#endif
```

Save the file and compile all the library files again. There are fewer errors now:

![](https://c.mql5.com/2/36/CompileErrors2.png)

Move to the beginning of the error list again and click the first one. Now it is ENUM\_POSITION\_TYPE,
so let's add:

```
//+------------------------------------------------------------------+
//|                                                       ToMQL4.mqh |
//|              Copyright 2017, Artem A. Trishkin, Skype artmedia70 |
//|                         https://www.mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, Artem A. Trishkin, Skype artmedia70"
#property link      "https://www.mql5.com/en/users/artmedia70"
#property strict
#ifdef __MQL4__
//+------------------------------------------------------------------+
//| MQL5 deal type                                                   |
//+------------------------------------------------------------------+
enum ENUM_DEAL_TYPE
  {
   DEAL_TYPE_BUY,
   DEAL_TYPE_SELL,
   DEAL_TYPE_BALANCE,
   DEAL_TYPE_CREDIT,
   DEAL_TYPE_CHARGE,
   DEAL_TYPE_CORRECTION,
   DEAL_TYPE_BONUS,
   DEAL_TYPE_COMMISSION,
   DEAL_TYPE_COMMISSION_DAILY,
   DEAL_TYPE_COMMISSION_MONTHLY,
   DEAL_TYPE_COMMISSION_AGENT_DAILY,
   DEAL_TYPE_COMMISSION_AGENT_MONTHLY,
   DEAL_TYPE_INTEREST,
   DEAL_TYPE_BUY_CANCELED,
   DEAL_TYPE_SELL_CANCELED,
   DEAL_DIVIDEND,
   DEAL_DIVIDEND_FRANKED,
   DEAL_TAX
  };
//+------------------------------------------------------------------+
//| Open position direction                                          |
//+------------------------------------------------------------------+
enum ENUM_POSITION_TYPE
  {
   POSITION_TYPE_BUY,
   POSITION_TYPE_SELL
  };
//+------------------------------------------------------------------+
#endif
```

After compiling, we get even less errors. Move to the first error in the list, define the reason and add
the following enumeration:

```
//+------------------------------------------------------------------+
//|                                                       ToMQL4.mqh |
//|              Copyright 2017, Artem A. Trishkin, Skype artmedia70 |
//|                         https://www.mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, Artem A. Trishkin, Skype artmedia70"
#property link      "https://www.mql5.com/en/users/artmedia70"
#property strict
#ifdef __MQL4__
//+------------------------------------------------------------------+
//| MQL5 deal types                                                  |
//+------------------------------------------------------------------+
enum ENUM_DEAL_TYPE
  {
   DEAL_TYPE_BUY,
   DEAL_TYPE_SELL,
   DEAL_TYPE_BALANCE,
   DEAL_TYPE_CREDIT,
   DEAL_TYPE_CHARGE,
   DEAL_TYPE_CORRECTION,
   DEAL_TYPE_BONUS,
   DEAL_TYPE_COMMISSION,
   DEAL_TYPE_COMMISSION_DAILY,
   DEAL_TYPE_COMMISSION_MONTHLY,
   DEAL_TYPE_COMMISSION_AGENT_DAILY,
   DEAL_TYPE_COMMISSION_AGENT_MONTHLY,
   DEAL_TYPE_INTEREST,
   DEAL_TYPE_BUY_CANCELED,
   DEAL_TYPE_SELL_CANCELED,
   DEAL_DIVIDEND,
   DEAL_DIVIDEND_FRANKED,
   DEAL_TAX
  };
//+------------------------------------------------------------------+
//| Open position direction                                          |
//+------------------------------------------------------------------+
enum ENUM_POSITION_TYPE
  {
   POSITION_TYPE_BUY,
   POSITION_TYPE_SELL
  };
//+------------------------------------------------------------------+
//| Order status                                                     |
//+------------------------------------------------------------------+
enum ENUM_ORDER_STATE
  {
   ORDER_STATE_STARTED,
   ORDER_STATE_PLACED,
   ORDER_STATE_CANCELED,
   ORDER_STATE_PARTIAL,
   ORDER_STATE_FILLED,
   ORDER_STATE_REJECTED,
   ORDER_STATE_EXPIRED,
   ORDER_STATE_REQUEST_ADD,
   ORDER_STATE_REQUEST_MODIFY,
   ORDER_STATE_REQUEST_CANCEL
  };
//+------------------------------------------------------------------+
#endif
```

During the next compilation, we got the wrong order type [ORDER\_TYPE\_BUY\_STOP\_LIMIT](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type).

MQL4 already features the [ENUM\_ORDER\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type)
enumeration. We cannot add new constants to it. Therefore, add them as macro substitutions.

In MQL5, the ORDER\_TYPE\_BUY\_STOP\_LIMIT constant from the ENUM\_ORDER\_TYPE enumeration is set to 6, while in MQL4, such an order type
exists. This balance operation, like ORDER\_TYPE\_SELL\_STOP\_LIMIT in MQL5, is set to 7, while in MQL4, this order type is a credit operation.

Therefore, set the values exceeding the ORDER\_TYPE\_CLOSE\_BY
closing order constant in MQL5 for them:

ORDER\_TYPE\_CLOSE\_BY+1 and ORDER\_TYPE\_CLOSE\_BY+2
accordingly:

```
//+------------------------------------------------------------------+
//|                                                       ToMQL4.mqh |
//|              Copyright 2017, Artem A. Trishkin, Skype artmedia70 |
//|                         https://www.mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, Artem A. Trishkin, Skype artmedia70"
#property link      "https://www.mql5.com/en/users/artmedia70"
#property strict
#ifdef __MQL4__
//+------------------------------------------------------------------+
//| MQL5 deal types                                                  |
//+------------------------------------------------------------------+
enum ENUM_DEAL_TYPE
  {
   DEAL_TYPE_BUY,
   DEAL_TYPE_SELL,
   DEAL_TYPE_BALANCE,
   DEAL_TYPE_CREDIT,
   DEAL_TYPE_CHARGE,
   DEAL_TYPE_CORRECTION,
   DEAL_TYPE_BONUS,
   DEAL_TYPE_COMMISSION,
   DEAL_TYPE_COMMISSION_DAILY,
   DEAL_TYPE_COMMISSION_MONTHLY,
   DEAL_TYPE_COMMISSION_AGENT_DAILY,
   DEAL_TYPE_COMMISSION_AGENT_MONTHLY,
   DEAL_TYPE_INTEREST,
   DEAL_TYPE_BUY_CANCELED,
   DEAL_TYPE_SELL_CANCELED,
   DEAL_DIVIDEND,
   DEAL_DIVIDEND_FRANKED,
   DEAL_TAX
  };
//+------------------------------------------------------------------+
//| Open position direction                                          |
//+------------------------------------------------------------------+
enum ENUM_POSITION_TYPE
  {
   POSITION_TYPE_BUY,
   POSITION_TYPE_SELL
  };
//+------------------------------------------------------------------+
//| Order status                                                     |
//+------------------------------------------------------------------+
enum ENUM_ORDER_STATE
  {
   ORDER_STATE_STARTED,
   ORDER_STATE_PLACED,
   ORDER_STATE_CANCELED,
   ORDER_STATE_PARTIAL,
   ORDER_STATE_FILLED,
   ORDER_STATE_REJECTED,
   ORDER_STATE_EXPIRED,
   ORDER_STATE_REQUEST_ADD,
   ORDER_STATE_REQUEST_MODIFY,
   ORDER_STATE_REQUEST_CANCEL
  };
//+------------------------------------------------------------------+
//| Order types                                                      |
//+------------------------------------------------------------------+
#define ORDER_TYPE_CLOSE_BY         (8)
#define ORDER_TYPE_BUY_STOP_LIMIT   (9)
#define ORDER_TYPE_SELL_STOP_LIMIT  (10)
//+------------------------------------------------------------------+
#endif
```

Compile the entire library. After implementing macro substitutions of StopLimit order types, the error indicates the functions returning the
correct order placement price, namely the ENUM\_ORDER\_TYPE enumerations having no values of 9 and 10, since we use the

value of the order type in the switch operator having the
ENUM\_ORDER\_TYPE enumeration type:

```
//+------------------------------------------------------------------+
//| Return the correct order placement price                         |
//| relative to StopLevel                                            |
//+------------------------------------------------------------------+
double CorrectPricePending(const string symbol_name,const ENUM_ORDER_TYPE order_type,const double price_set,const double price=0,const int spread_multiplier=2)
  {
   double pt=SymbolInfoDouble(symbol_name,SYMBOL_POINT),pp=0;
   int lv=StopLevel(symbol_name,spread_multiplier), dg=(int)SymbolInfoInteger(symbol_name,SYMBOL_DIGITS);
   switch(order_type)
     {
      case ORDER_TYPE_BUY_LIMIT        :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_ASK) : price); return NormalizeDouble(fmin(pp-lv*pt,price_set),dg);
      case ORDER_TYPE_BUY_STOP         :
      case ORDER_TYPE_BUY_STOP_LIMIT   :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_ASK) : price); return NormalizeDouble(fmax(pp+lv*pt,price_set),dg);
      case ORDER_TYPE_SELL_LIMIT       :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_BID) : price); return NormalizeDouble(fmax(pp+lv*pt,price_set),dg);
      case ORDER_TYPE_SELL_STOP        :
      case ORDER_TYPE_SELL_STOP_LIMIT  :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_BID) : price); return NormalizeDouble(fmin(pp-lv*pt,price_set),dg);
      default                          :  Print(DFUN,TextByLanguage("Неправильный тип ордера: ","Invalid order type: "),EnumToString(order_type)); return 0;
     }
  }
//+------------------------------------------------------------------+
//| Return the correct order placement price                         |
//| relative to StopLevel                                            |
//+------------------------------------------------------------------+
double CorrectPricePending(const string symbol_name,const ENUM_ORDER_TYPE order_type,const int distance_set,const double price=0,const int spread_multiplier=2)
  {
   double pt=SymbolInfoDouble(symbol_name,SYMBOL_POINT),pp=0;
   int lv=StopLevel(symbol_name,spread_multiplier), dg=(int)SymbolInfoInteger(symbol_name,SYMBOL_DIGITS);
   switch(order_type)
     {
      case ORDER_TYPE_BUY_LIMIT        :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_ASK) : price); return NormalizeDouble(fmin(pp-lv*pt,pp-distance_set*pt),dg);
      case ORDER_TYPE_BUY_STOP         :
      case ORDER_TYPE_BUY_STOP_LIMIT   :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_ASK) : price); return NormalizeDouble(fmax(pp+lv*pt,pp+distance_set*pt),dg);
      case ORDER_TYPE_SELL_LIMIT       :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_BID) : price); return NormalizeDouble(fmax(pp+lv*pt,pp+distance_set*pt),dg);
      case ORDER_TYPE_SELL_STOP        :
      case ORDER_TYPE_SELL_STOP_LIMIT  :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_BID) : price); return NormalizeDouble(fmin(pp-lv*pt,pp-distance_set*pt),dg);
      default                          :  Print(DFUN,TextByLanguage("Неправильный тип ордера: ","Invalid order type: "),EnumToString(order_type)); return 0;
     }
  }
//+------------------------------------------------------------------+
```

The solution is simple — **order\_type** in **switch** is
converted to

integer type:

```
//+------------------------------------------------------------------+
//| Return the correct order placement price                         |
//| relative to StopLevel                                            |
//+------------------------------------------------------------------+
double CorrectPricePending(const string symbol_name,const ENUM_ORDER_TYPE order_type,const double price_set,const double price=0,const int spread_multiplier=2)
  {
   double pt=SymbolInfoDouble(symbol_name,SYMBOL_POINT),pp=0;
   int lv=StopLevel(symbol_name,spread_multiplier), dg=(int)SymbolInfoInteger(symbol_name,SYMBOL_DIGITS);
   switch((int)order_type)
     {
      case ORDER_TYPE_BUY_LIMIT        :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_ASK) : price); return NormalizeDouble(fmin(pp-lv*pt,price_set),dg);
      case ORDER_TYPE_BUY_STOP         :
      case ORDER_TYPE_BUY_STOP_LIMIT   :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_ASK) : price); return NormalizeDouble(fmax(pp+lv*pt,price_set),dg);
      case ORDER_TYPE_SELL_LIMIT       :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_BID) : price); return NormalizeDouble(fmax(pp+lv*pt,price_set),dg);
      case ORDER_TYPE_SELL_STOP        :
      case ORDER_TYPE_SELL_STOP_LIMIT  :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_BID) : price); return NormalizeDouble(fmin(pp-lv*pt,price_set),dg);
      default                          :  Print(DFUN,TextByLanguage("Неправильный тип ордера: ","Invalid order type: "),EnumToString(order_type)); return 0;
     }
  }
//+------------------------------------------------------------------+
//| Return the correct order placement price                         |
//| relative to StopLevel                                            |
//+------------------------------------------------------------------+
double CorrectPricePending(const string symbol_name,const ENUM_ORDER_TYPE order_type,const int distance_set,const double price=0,const int spread_multiplier=2)
  {
   double pt=SymbolInfoDouble(symbol_name,SYMBOL_POINT),pp=0;
   int lv=StopLevel(symbol_name,spread_multiplier), dg=(int)SymbolInfoInteger(symbol_name,SYMBOL_DIGITS);
   switch((int)order_type)
     {
      case ORDER_TYPE_BUY_LIMIT        :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_ASK) : price); return NormalizeDouble(fmin(pp-lv*pt,pp-distance_set*pt),dg);
      case ORDER_TYPE_BUY_STOP         :
      case ORDER_TYPE_BUY_STOP_LIMIT   :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_ASK) : price); return NormalizeDouble(fmax(pp+lv*pt,pp+distance_set*pt),dg);
      case ORDER_TYPE_SELL_LIMIT       :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_BID) : price); return NormalizeDouble(fmax(pp+lv*pt,pp+distance_set*pt),dg);
      case ORDER_TYPE_SELL_STOP        :
      case ORDER_TYPE_SELL_STOP_LIMIT  :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_BID) : price); return NormalizeDouble(fmin(pp-lv*pt,pp-distance_set*pt),dg);
      default                          :  Print(DFUN,TextByLanguage("Неправильный тип ордера: ","Invalid order type: "),EnumToString(order_type)); return 0;
     }
  }
//+------------------------------------------------------------------+
```

Let's perform compilation. Now there is an error in the Order.mqh file — MQL4 does not know the values of [ORDER\_FILLING\_RETURN](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type_filling),

[ORDER\_TIME\_GTC](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type_time),

[ORDER\_REASON\_SL, \\
\\
ORDER\_REASON\_TP and ORDER\_REASON\_EXPERT](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_reason)
constants.

```
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
      case ORDER_STATUS_MARKET_ORDER      :
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
      case ORDER_STATUS_MARKET_ORDER      :
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
      case ORDER_STATUS_MARKET_POSITION   : res=::PositionGetInteger(POSITION_REASON);          break;
      case ORDER_STATUS_MARKET_ORDER      :
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
```

Let's add macro substitutions to the end of the ToMQL4.mqh file (I will not give a full listing here to save space):

```
//+------------------------------------------------------------------+
//| Order types, execution policy, lifetime, reasons                 |
//+------------------------------------------------------------------+
#define ORDER_TYPE_CLOSE_BY         (8)
#define ORDER_TYPE_BUY_STOP_LIMIT   (9)
#define ORDER_TYPE_SELL_STOP_LIMIT  (10)
#define ORDER_FILLING_RETURN        (2)
#define ORDER_TIME_GTC              (0)
#define ORDER_REASON_EXPERT         (3)
#define ORDER_REASON_SL             (4)
#define ORDER_REASON_TP             (5)
//+------------------------------------------------------------------+
#endif
```

Another compilation leads us to the missing HistoryOrderGetTicket() MQL5 function in the HistoryCollection.mqh file of the
CHistoryCollection::OrderSearch() method. The code analysis suggests applying conditional compilation directives here. Let's
supplement the method:

```
//+------------------------------------------------------------------+
//| Return the "lost" order's type and ticket                        |
//+------------------------------------------------------------------+
ulong CHistoryCollection::OrderSearch(const int start,ENUM_ORDER_TYPE &order_type)
  {
   ulong order_ticket=0;
#ifdef __MQL5__
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
#else
   for(int i=start-1;i>=0;i--)
     {
      if(!::OrderSelect(i,SELECT_BY_POS,MODE_HISTORY))
         continue;
      ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)::OrderType();
      ulong ticket=::OrderTicket();
      if(ticket==0 || type<ORDER_TYPE_BUY_LIMIT || type>ORDER_TYPE_SELL_STOP)
         continue;
      if(this.IsPresentOrderInList(ticket,type))
         continue;
      order_ticket=ticket;
      order_type=type;
     }
#endif
   return order_ticket;
  }
//+------------------------------------------------------------------+
```

All things intended for MQL5 are framed by the #ifdef
\_\_MQL5\_\_ directive. The code is added for MQL4 after
the

#else directive up to #endif.

The next error is located in the CEvent class constructor. Supplement the code using the same conditional compilation directives:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CEvent::CEvent(const ENUM_EVENT_STATUS event_status,const int event_code,const ulong ticket) : m_event_code(event_code),m_digits(0)
  {
   this.m_long_prop[EVENT_PROP_STATUS_EVENT]       =  event_status;
   this.m_long_prop[EVENT_PROP_TICKET_ORDER_EVENT] =  (long)ticket;
   this.m_is_hedge=#ifdef __MQL4__ true #else bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING) #endif;
   this.m_digits_acc=#ifdef __MQL4__ 2 #else (int)::AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS) #endif;
   this.m_chart_id=::ChartID();
  }
//+------------------------------------------------------------------+
```

When checking the account for "hedging" type, we face the
absence of a constant error, therefore simply

return true at once
as all accounts are hedging ones in MetaTrader 4.

Also, when receiving the number of decimal places in the account currency, return
2, since MQL4 cannot obtain this value.

The next compilation leads us to the CEventsCollection::NewDealEventHedge() method — receiving an event for a MetaTrader 5 hedging
account. It works with deals absent in MQL4. Temporarily disable the method by enclosing all the method code in the framework of conditional
compilation:

Insert the directive at the beginning of the method

```
//+------------------------------------------------------------------+
//| Create a hedging account event                                   |
//+------------------------------------------------------------------+
void CEventsCollection::NewDealEventHedge(COrder* deal,CArrayObj* list_history,CArrayObj* list_market)
  {
#ifdef __MQL5__
   double ask=::SymbolInfoDouble(deal.Symbol(),SYMBOL_ASK);
   double bid=::SymbolInfoDouble(deal.Symbol(),SYMBOL_BID);
   //--- Market entry
```

and at the end of the method

```
#endif
  }
//+------------------------------------------------------------------+
```

Next, we end up with the error in the CEventsCollection::NewDealEventNetto() method — creating an event for a netting account. The solution is
the same as in the previous case — frame the entire NewDealEventNetto() method code with the conditional compilation directive.

Compile and face the DEAL\_ENTRY\_IN unknown constant error in the CEventsCollection::GetListAllDealsInByPosID() method. Add
the necessary enumeration to the ToMQL4.mqh file (we could use conditional compilation again to disable the code, but we may need
this enumeration later):

```
//+------------------------------------------------------------------+
//| MQL5 deal types                                                  |
//+------------------------------------------------------------------+
enum ENUM_DEAL_TYPE
  {
   DEAL_TYPE_BUY,
   DEAL_TYPE_SELL,
   DEAL_TYPE_BALANCE,
   DEAL_TYPE_CREDIT,
   DEAL_TYPE_CHARGE,
   DEAL_TYPE_CORRECTION,
   DEAL_TYPE_BONUS,
   DEAL_TYPE_COMMISSION,
   DEAL_TYPE_COMMISSION_DAILY,
   DEAL_TYPE_COMMISSION_MONTHLY,
   DEAL_TYPE_COMMISSION_AGENT_DAILY,
   DEAL_TYPE_COMMISSION_AGENT_MONTHLY,
   DEAL_TYPE_INTEREST,
   DEAL_TYPE_BUY_CANCELED,
   DEAL_TYPE_SELL_CANCELED,
   DEAL_DIVIDEND,
   DEAL_DIVIDEND_FRANKED,
   DEAL_TAX
  };
//+------------------------------------------------------------------+
//| Position change method                                           |
//+------------------------------------------------------------------+
enum ENUM_DEAL_ENTRY
  {
   DEAL_ENTRY_IN,
   DEAL_ENTRY_OUT,
   DEAL_ENTRY_INOUT,
   DEAL_ENTRY_OUT_BY
  };
//+------------------------------------------------------------------+
//| Open position direction                                          |
//+------------------------------------------------------------------+
enum ENUM_POSITION_TYPE
  {
   POSITION_TYPE_BUY,
   POSITION_TYPE_SELL
  };
//+------------------------------------------------------------------+
```

Next, we end up with the already familiar error of checking the account for the "hedging" type, but now it is in the event collection class
constructor.

Let's fix it:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CEventsCollection::CEventsCollection(void) : m_trade_event(TRADE_EVENT_NO_EVENT),m_trade_event_code(TRADE_EVENT_FLAG_NO_EVENT)
  {
   this.m_list_events.Clear();
   this.m_list_events.Sort(SORT_BY_EVENT_TIME_EVENT);
   this.m_list_events.Type(COLLECTION_EVENTS_ID);
   this.m_is_hedge=#ifdef __MQL4__ true #else bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING) #endif;
   this.m_chart_id=::ChartID();
   ::ZeroMemory(this.m_tick);
  }
//+------------------------------------------------------------------+
```

Next, implement the same correction to the CEngine class
constructor:

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
   this.m_is_hedge=#ifdef __MQL4__ true #else bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING) #endif;
  }
//+------------------------------------------------------------------+
```

All is set. Now the entire library is compiled without errors. But this is only the first stage. Now we need to launch it. Since we disabled a few
methods using conditional compilation, we will need to develop them to work in MetaTrader 4.

In MQL5, balance operations are deals. They can be found in the list of historical orders and deals. In MQL4, balance operations are orders of
ORDER\_TYPE\_BALANCE (6) and ORDER\_TYPE\_CREDIT (7) types. Therefore, I made a separate class of a balance operation object for MQL4 stored
in the list of historical orders and positions.

Create the new **CHistoryBalance** class in \\MQL4\\Include\\DoEasy\\Objects\ **Orders** of the **HistoryBalance.mqh** file. COrder
should be a basic class:

```
//+------------------------------------------------------------------+
//|                                               HistoryBalance.mqh |
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
//| Historical balance operation                                     |
//+------------------------------------------------------------------+
class CHistoryBalance : public COrder
  {
public:
   //--- Constructor
                     CHistoryBalance(const ulong ticket) : COrder(ORDER_STATUS_BALANCE,ticket) {}
   //--- Supported deal properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_ORDER_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_ORDER_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_ORDER_PROP_STRING property);
  };
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CHistoryBalance::SupportProperty(ENUM_ORDER_PROP_INTEGER property)
  {
   if(property==ORDER_PROP_TICKET      ||
      property==ORDER_PROP_TIME_OPEN   ||
      property==ORDER_PROP_STATUS      ||
      property==ORDER_PROP_TYPE        ||
      property==ORDER_PROP_REASON
     ) return true;
   return false;
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CHistoryBalance::SupportProperty(ENUM_ORDER_PROP_DOUBLE property)
  {
   return(property==ORDER_PROP_PROFIT ? true : false);
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| string property, otherwise return 'false'                        |
//+------------------------------------------------------------------+
bool CHistoryBalance::SupportProperty(ENUM_ORDER_PROP_STRING property)
  {
   if(property==ORDER_PROP_SYMBOL || property==ORDER_PROP_EXT_ID)
      return false;
   return true;
  }
//+------------------------------------------------------------------+
```

The class contains nothing new for us. We already reviewed all historical order classes [in \\
the second part](https://www.mql5.com/en/articles/5669#node01) of the library description.

We have two types of balance operations — balance and credit ones. Accordingly, their types have numeric values of 6 and 7. We will use a
single balance operation class for both types and clarify a certain type in the "reason" order property.

Add two missing order "reasons" to the ToMQL4.mqh file:

```
//+------------------------------------------------------------------+
//| Order types, execution policy, lifetime, reasons                 |
//+------------------------------------------------------------------+
#define ORDER_TYPE_CLOSE_BY         (8)
#define ORDER_TYPE_BUY_STOP_LIMIT   (9)
#define ORDER_TYPE_SELL_STOP_LIMIT  (10)
#define ORDER_FILLING_RETURN        (2)
#define ORDER_TIME_GTC              (0)
#define ORDER_REASON_EXPERT         (3)
#define ORDER_REASON_SL             (4)
#define ORDER_REASON_TP             (5)
#define ORDER_REASON_BALANCE        (6)
#define ORDER_REASON_CREDIT         (7)
//+------------------------------------------------------------------+
```

Since we have a new class derived from the abstract order class, we need to add the missing functionality in COrder.

In the COrder::OrderPositionID() method, replace returning the magic number
for MQL4

```
//+------------------------------------------------------------------+
//| Return position ID                                               |
//+------------------------------------------------------------------+
long COrder::OrderPositionID(void) const
  {
#ifdef __MQL4__
   return ::OrderMagicNumber();
#else
```

with returning the ticket (a kind of PositionID for MQL4
positions is to be implemented later):

```
//+------------------------------------------------------------------+
//| Return position ID                                               |
//+------------------------------------------------------------------+
long COrder::OrderPositionID(void) const
  {
#ifdef __MQL4__
   return ::OrderTicket();
#else
```

The method returning the order status in MQL4 always returns ORDER\_STATE\_FILLED from the [ENUM\_ORDER\_STATE \\
enumeration](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_state), which is not true for remote pending orders. Implement the order
status check, and if this is a remote pending order, return ORDER\_STATE\_CANCELED.

```
//+------------------------------------------------------------------+
//| Return the order status                                          |
//+------------------------------------------------------------------+
long COrder::OrderState(void) const
  {
#ifdef __MQL4__
   return(this.Status()==ORDER_STATUS_HISTORY_ORDER ? ORDER_STATE_FILLED : ORDER_STATE_CANCELED);
#else
   long res=0;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetInteger(m_ticket,ORDER_STATE); break;
      case ORDER_STATUS_MARKET_ORDER      :
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetInteger(ORDER_STATE);                 break;
      case ORDER_STATUS_MARKET_POSITION   :
      case ORDER_STATUS_DEAL              :
      default                             : res=0;                                              break;
     }
   return res;
#endif
  }
//+------------------------------------------------------------------+
```

Add the two newly added "reasons" to the method returning the
order reason for MQL4:

```
//+------------------------------------------------------------------+
//| Order reason or source                                           |
//+------------------------------------------------------------------+
long COrder::OrderReason(void) const
  {
#ifdef __MQL4__
   return
     (
      this.TypeOrder()==ORDER_TYPE_BALANCE   ?  ORDER_REASON_BALANCE :
      this.TypeOrder()==ORDER_TYPE_CREDIT    ?  ORDER_REASON_CREDIT  :
      this.OrderCloseByStopLoss()            ?  ORDER_REASON_SL      :
      this.OrderCloseByTakeProfit()          ?  ORDER_REASON_TP      :
      this.OrderMagicNumber()!=0             ?  ORDER_REASON_EXPERT  : WRONG_VALUE
     );
#else
```

In our case, the method returning unexecuted volume for MQL4 always returns an order lot, which is incorrect for positions. For remote
pending orders, we will return a lot order, while for positions, we will
return zero:

```
//+------------------------------------------------------------------+
//| Return unexecuted volume                                         |
//+------------------------------------------------------------------+
double COrder::OrderVolumeCurrent(void) const
  {
#ifdef __MQL4__
   return(this.Status()==ORDER_STATUS_HISTORY_PENDING ? ::OrderLots() : 0);
#else
```

Add descriptions of the two new "reasons" in the method returning an order reason description. For balance
and credit operations, check
the profit. If it exceeds zero, then the funds are deposited, otherwise, the funds are withdrawn:

```
//+------------------------------------------------------------------+
//| Reason description                                               |
//+------------------------------------------------------------------+
string COrder::GetReasonDescription(const long reason) const
  {
#ifdef __MQL4__
   return
     (
      this.IsCloseByStopLoss()            ?  TextByLanguage("Срабатывание StopLoss","Due to StopLoss")                  :
      this.IsCloseByTakeProfit()          ?  TextByLanguage("Срабатывание TakeProfit","Due to TakeProfit")              :
      this.Reason()==ORDER_REASON_EXPERT  ?  TextByLanguage("Выставлен из mql4-программы","Placed from mql4 program")   :
      this.Comment()=="cancelled"         ?  TextByLanguage("Отменён","Cancelled")                                      :
      this.Reason()==ORDER_REASON_BALANCE ?  (
                                              this.Profit()>0 ? TextByLanguage("Пополнение баланса","Deposit of funds on the account balance") :
                                              TextByLanguage("Снятие средств с баланса","Withdrawal from the balance")
                                             )                                                                          :
      this.Reason()==ORDER_REASON_CREDIT  ?  (
                                              this.Profit()>0 ? TextByLanguage("Начисление кредитных средств","Received credit funds") :
                                              TextByLanguage("Изъятие кредитных средств","Withdrawal of credit")
                                             )                                                                          :

      TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4")
     );
#else
```

Besides, some minor edits were made. They are too insignificant to be described here. They are mostly related to a text displayed in MQL5/MQL4
journal. All edits are available in the library files attached to the article.

**Now let's improve the historical collection class in the HistoryCollection.mqh file.**

First, include
the new class file to it:

```
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Services\Select.mqh"
#include "..\Objects\Orders\HistoryOrder.mqh"
#include "..\Objects\Orders\HistoryPending.mqh"
#include "..\Objects\Orders\HistoryDeal.mqh"
#ifdef __MQL4__
#include "..\Objects\Orders\HistoryBalance.mqh"
#endif
//+------------------------------------------------------------------+
```

Since we need the CHistoryBalance class only for the MQL4 version of the library, including the file of this class is enclosed in the conditional
compilation directives for MQL4.

Now we have a new class of balance operations. In order to develop and place it into the collection, we need to add the
check of order types for the type of a balance and credit operation and adding them to the collection in the Refresh() method of the
CHistoryCollection class for MQL4:

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
      //--- Closed positions
      if(order_type<ORDER_TYPE_BUY_LIMIT)
        {
         CHistoryOrder *order=new CHistoryOrder(::OrderTicket());
         if(order==NULL) continue;
         if(!this.m_list_all_orders.InsertSort(order))
           {
            ::Print(DFUN,TextByLanguage("Не удалось добавить ордер в список","Could not add order to the list"));
            delete order;
           }
        }
      //--- Balance/credit operations
      else if(order_type>ORDER_TYPE_SELL_STOP)
        {
         CHistoryBalance *order=new CHistoryBalance(::OrderTicket());
         if(order==NULL) continue;
         if(!this.m_list_all_orders.InsertSort(order))
           {
            ::Print(DFUN,TextByLanguage("Не удалось добавить ордер в список","Could not add order to the list"));
            delete order;
           }
        }
      else
        {
         //--- Removed pending orders
         CHistoryPending *order=new CHistoryPending(::OrderTicket());
         if(order==NULL) continue;
         if(!this.m_list_all_orders.InsertSort(order))
           {
            ::Print(DFUN,TextByLanguage("Не удалось добавить ордер в список","Could not add order to the list"));
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
```

**Let's**
**make some corrections in the historical order class:**

```
//+------------------------------------------------------------------+
//| Return 'true' if an order supports the passed                    |
//| real property, otherwise, return 'false'                         |
//+------------------------------------------------------------------+
bool CHistoryOrder::SupportProperty(ENUM_ORDER_PROP_DOUBLE property)
  {
   if(
   #ifdef __MQL5__
      property==ORDER_PROP_PROFIT                  ||
      property==ORDER_PROP_PROFIT_FULL             ||
      property==ORDER_PROP_SWAP                    ||
      property==ORDER_PROP_COMMISSION              ||
      property==ORDER_PROP_PRICE_CLOSE             ||
      (
       property==ORDER_PROP_PRICE_STOP_LIMIT       &&
       (
        this.TypeOrder()<ORDER_TYPE_BUY_STOP_LIMIT ||
        this.TypeOrder()>ORDER_TYPE_SELL_STOP_LIMIT
       )
      )
   #else
      property==ORDER_PROP_PRICE_STOP_LIMIT        &&
      this.Status()==ORDER_STATUS_HISTORY_ORDER
   #endif
     ) return false;

   return true;
  }
//+------------------------------------------------------------------+
```

Previously, a StopLimit order price in MQL5 was not passed to
the journal. Therefore, I implemented a check:

if the checked property is
a StopLimit order price, and if an order type is not a StopLimit one,
the property is not used. Otherwise, this is a StopLimit order and the property is necessary.

In MQL4, a StopLimit order price is not used for
positions.

This completes the improvement of the first stage of compatibility with MQL4.

### Testing

For testing purposes, take the TestDoEasyPart03\_1.mq5 EA from \\MQL5\\Experts\\TestDoEasy\\Part03 and save it under the name **TestDoEasyPart09.mq4** in the
folder for MQL4 EAs

**\\MQL4\\Experts\\TestDoEasy\\Part09**.

The EA is compiled without changes but if we have a look at the code, it turns out that it uses the
list of deals absent in MQL4:

```
//--- enums
enum ENUM_TYPE_ORDERS
  {
   TYPE_ORDER_MARKET,   // Market orders
   TYPE_ORDER_PENDING,  // Pending orders
   TYPE_ORDER_DEAL      // Deals
  };
//--- input parameters
```

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- update history
   history.Refresh();
//--- get the collection list within the date range
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
```

Simply replace deals with balance operations. In this case, we will use the conditional compilation directly in the EA, which is not correct for
the final product, where all actions for delimiting by language versions should be hidden from users. But in this case, we simply test the
library improvement results, so this is not a big deal.

Let's add minor changes to the EA code replacing MQL5 deals
with

MQL4 balance operations:

```
//+------------------------------------------------------------------+
//|                                           TestDoEasyPart03_1.mq4 |
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
#ifdef __MQL5__
   TYPE_ORDER_DEAL      // Deals
#else
   TYPE_ORDER_BALANCE   // Balance/Credit
#endif
  };
//--- input parameters
input ENUM_TYPE_ORDERS  InpOrderType   =  TYPE_ORDER_MARKET;   // Show type:
input datetime          InpTimeBegin   =  0;                   // Start date of required range
input datetime          InpTimeEnd     =  END_TIME;            // End date of required range
//--- global variables
CHistoryCollection history;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- update history
   history.Refresh();
//--- get the collection list within the date range
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
   #ifdef __MQL5__
      if(order.Status()==ORDER_STATUS_DEAL && InpOrderType==TYPE_ORDER_DEAL)
         order.Print();
   #else
   //--- if this is a balance/credit operation
      if(order.Status()==ORDER_STATUS_BALANCE && InpOrderType==TYPE_ORDER_BALANCE)
         order.Print();
   #endif
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

Compile and launch the EA in the terminal (the test EA from the [third \\
article](https://www.mql5.com/en/articles/5687#node01) works only in the OnInit() handler, so it will display the required historical collection list once after the launch or after
changing the list in the settings).

Before launching the EA, select the "All History" option in the context menu of the terminal's "Account History" tab, since in MetaTrader 4,
the amount of history available for applications depends on a history size selected in the tab.

**Balance/Credit is selected in the settings, and the very first balance deposit is displayed in the journal:**

![](https://c.mql5.com/2/36/Balance.gif)

Now we need to check if the search and display of closed positions is correct. Since I recently opened a MetaTrader 4 account, there was no
trading on it. I opened Sell, set StopLoss and TakeProfit and left to make some coffee. When I returned, the position was closed by the stop
loss, upon which the market started moving in the direction of the Sell position. Yeah, that's always the way! :)

But now there is one closed position for the test. **"Market orders" is selected in the settings:**

![](https://c.mql5.com/2/36/ClosedPos.gif)

Now let's check the list of removed pending orders. I set a couple of orders and removed them afterwards.

**"Pending**
**orders" is selected in the settings:**

![](https://c.mql5.com/2/36/Deleted.gif)

The list of removed pending orders is displayed as well.

### What's next?

In the next article, we will implement the ability to work with market positions and active pending orders in MQL4.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/6651#node00)

**Previous articles within the series:**

[Part 1. Concept, data management.](https://www.mql5.com/en/articles/5654)

[Part \\
2\. Collection of historical orders and deals.](https://www.mql5.com/en/articles/5669)

[Part 3. Collection of market orders \\
and positions, arranging the search.](https://www.mql5.com/en/articles/5687)

[Part 4. Trading events. Concept.](https://www.mql5.com/en/articles/5724)

[Part 5. Classes and collection of trading events. Sending events to the program.](https://www.mql5.com/en/articles/6211)

[Part \\
6\. Netting account events.](https://www.mql5.com/en/articles/6383)

[Part 7. StopLimit order activation events, preparing \\
the functionality for order and position modification events.](https://www.mql5.com/en/articles/6482)

[Part 8. Order and \\
position modification events.](https://www.mql5.com/en/articles/6595)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/6651](https://www.mql5.com/ru/articles/6651)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/6651.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/6651/mql5.zip "Download MQL5.zip")(89.98 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/6651/mql4.zip "Download MQL4.zip")(89.98 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/318971)**
(9)


![Sergey Voytsekhovsky](https://c.mql5.com/avatar/2018/7/5B51BA55-B751.jpg)

**[Sergey Voytsekhovsky](https://www.mql5.com/en/users/logic)**
\|
21 May 2019 at 12:53

**Artyom Trishkin:**

Gradually, so much functionality will be added to the library that it will be very easy to work on algorithms - the way you want. And this is exactly what it was designed for.

Now, while there is no such functionality there, you can see how the test Expert Advisor works with the [CTrade trading class](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) included in the [standard library](https://www.mql5.com/en/docs/standardlibrary) in MQL5, and write similar constructs to call the necessary trading functions. There (in the test EA) there is a call of tester trading functions for MQL4.

Thank you, I will study it.

![Nikolai Karetnikov](https://c.mql5.com/avatar/2013/4/517CE93C-6F64.jpg)

**[Nikolai Karetnikov](https://www.mql5.com/en/users/ns_k)**
\|
19 Jun 2020 at 20:58

**Sergey Voytsekhovsky:**

Good afternoon. I like your trial Expert Advisor. I want to try to use it as a kernel, which will receive signals and filters from various indicators, their combinations, or controlled manually, through button presses.

You have already seen the first of such Expert Advisors and helped me to breathe life into it on a neighbouring thread of this forum.

Can you show me how to press the buttons programmatically in this trial EA of yours?

Is there a suitable function - can you share it?

Or suggest how best to do it, please.

Good afternoon!

Sergey, I will support you because I see you are in a similar situation.

Yes, the articles are great, but they contain very little information about _how to_ use the written code. Libraries, generally speaking, are valuable for hiding the implementation and providing a clear interface for practical tasks. The help of the https://docs.mql4.com/strings/stringsubstr function doesn't contain a single word about its internals. A description of incoming parameters, the result of their processing and example(s). This is what I would like to see.

Yes, Artem, you are undoubtedly a talented programmer, but application engineers need to develop another algorithm as quickly as possible and not spend hours over hundreds of lines of other people's code in search of enlightenment. The series of articles so far is more theoretical.

This is not my first post on this topic ). In no way I want to belittle the merits of the series. On the contrary - I hope, Artem, you will take into account the requests of the forum members and the written libraries will be used in EAs as eagerly as good films are quoted.

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
19 Jun 2020 at 21:29

**Nikolai Karetnikov:**

Good afternoon!

Sergey, I support you, because I see you are in a similar situation.

Yes, the articles are great, but they contain very little information about _how to_ use the written code. Libraries, generally speaking, are valuable for hiding the implementation and providing a clear interface for practical tasks. The help of the https://docs.mql4.com/strings/stringsubstr function doesn't contain a single word about its internals. A description of incoming parameters, the result of their processing and example(s). This is what I would like to see.

Yes, Artem, you are undoubtedly a talented programmer, but application engineers need to solve practical tasks rather than spend hours over hundreds of lines of other people's code in search of enlightenment. The series of articles so far is more theoretical.

This is not my first post on this topic ). In no way I want to belittle the merits of the series. On the contrary - I hope, Artem, you will take into account the requests of the forum members and the written libraries will be used in EAs as eagerly as good films are quoted.

The goal is to guide the reader from the beginning of the library creation to its completion.

You see - the articles are more educational in nature, while having a useful practical purpose, and more than one. The design of the codes is easy to understand, without using twists and undocumented features for the sake of twists and "coolness". But there is an undeniable plus - how many beta versions of the terminal have been released, and how many people have already said that their codes stopped working, and the library - lives from build to build without forced fixes because something suddenly stopped working....

The library currently has one entry point - the CEngine class (there will be a second entry point, but much later), and the object of this class in the EA gives full access to all the features.

And further - it is not difficult to create such an object, for example: CEngine lib; and in the code type lib and put a dot (like this: lib.) - after the dot the editor will show you a window with a list of all available for use methods of the library. Most of them have meaningful names - with a little practice you can use them. All methods are described in articles. In each article there is an example of a test programme showing, however, a small part of the possibilities.

I agree - searching for the shown methods and their application in numerous articles without reference material is a difficult task.... But the cycle of articles is a cycle so that the reader goes through it together with me, and then something will be stored in his head :) And the purpose, I remind you, is educational.

There will be reference material. But at the very end - when the library will be created. And examples, of course, too.

In the meantime, you can ask practical questions. Show a part of your code and I will give you a hint. I'm here and I'm not going anywhere - it's not in my rules to abandon what I've started.

![Nikolai Karetnikov](https://c.mql5.com/avatar/2013/4/517CE93C-6F64.jpg)

**[Nikolai Karetnikov](https://www.mql5.com/en/users/ns_k)**
\|
17 Jul 2020 at 21:54

**Artyom Trishkin:**

The goal is to take the reader from the beginning of the library to its completion.

You see - the articles are more educational in nature, while having a useful practical purpose, and more than one. The design of the codes is easy to understand, without using twists and undocumented features for the sake of twists and "coolness". But there is an undeniable plus - how many beta versions of the terminal have been released, and how many people have already said that their codes stopped working, and the library - lives from build to build without forced corrections because something suddenly stopped working....

The library currently has one entry point - the CEngine class (there will be a second entry point, but much later), and the object of this class in the EA gives full access to all features.

And further - it is not difficult to create such an object, for example: CEngine lib; and in the code type lib and put a dot (like this: lib.) - after the dot the editor will show you a window with a list of all available for use methods of the library. Most of them have meaningful names - with a little practice you can use them. All methods are described in articles. Each article contains an example of a test programme, which shows only a small part of the possibilities.

I agree - searching for the shown methods and their application in numerous articles without reference material is a difficult task.... But the cycle of articles is a cycle so that the reader goes through it together with me, and then something will be stored in his head :) And the purpose, I remind you, is educational.

There will be reference material. But at the very end - when the library will be created. And examples, of course, too.

In the meantime, you can ask practical questions. Show a part of your code and I will give you a hint. I'm here and I'm not going anywhere - it's not in my rules to abandon what I've started.

I understand your intentions are the best and you probably have a lot of free time ).

I just saw your articles of the series "MakingSimple"\[ Library for easy and fast creation of programmes for MetaTrader\] and thought that after 10-15 minutes of reading I would be able to use useful code. I expected to see a classic article like [https://www.mql5.com/en/articles/272,](https://www.mql5.com/en/articles/272) where the logic is hidden and the interface is open, where the questions are answered: "why it is needed", "how to work with it" and examples. Turns out the goal is training, not RAD (rapid development).

Well, we look forward to seeing you write one of these! ))

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
17 Jul 2020 at 22:23

**Nikolai Karetnikov:**

I realise your intentions are the best and you probably have a lot of free time )

I just saw your articles of the series "MakingSimple"\[ Library for easy and fast creation of programmes for MetaTrader\], I thought that after 10-15 minutes of reading I would be able to use useful code. I expected to see a classic article like [https://www.mql5.com/en/articles/272,](https://www.mql5.com/en/articles/272) where the logic is hidden and the interface is open, where the questions are answered: "why it is needed", "how to work with it" and examples.  It turned out that the goal was training, not RAD (rapid development).

Well, we look forward to seeing you write one of these! ))

Thegoal is learning + rapid development. About rapid development it's only worth asking practical application questions if you're too lazy to read and there's no reference material with examples yet.

The title rather translates as "Make it easy". (English..., allows to translate any way you want, if there is no context).

![Developing a cross-platform Expert Advisor to set StopLoss and TakeProfit based on risk settings](https://c.mql5.com/2/36/fix_open_200.png)[Developing a cross-platform Expert Advisor to set StopLoss and TakeProfit based on risk settings](https://www.mql5.com/en/articles/6986)

In this article, we will create an Expert Advisor for automated entry lot calculation based on risk values. Also the Expert Advisor will be able to automatically place Take Profit with the select ratio to Stop Loss. That is, it can calculate Take Profit based on any selected ratio, such as 3 to 1, 4 to 1 or any other selected value.

![Developing a cross-platform grider EA (part II): Range-based grid in trend direction](https://c.mql5.com/2/36/mql5_ea_adviser_grid.png)[Developing a cross-platform grider EA (part II): Range-based grid in trend direction](https://www.mql5.com/en/articles/6954)

In this article, we will develop a grider EA for trading in a trend direction within a range. Thus, the EA is to be suited mostly for Forex and commodity markets. According to the tests, our grider showed profit since 2018. Unfortunately, this is not true for the period of 2014-2018.

![Arranging a mailing campaign by means of Google services](https://c.mql5.com/2/36/logo_Csharp.png)[Arranging a mailing campaign by means of Google services](https://www.mql5.com/en/articles/6975)

A trader may want to arrange a mailing campaign to maintain business relationships with other traders, subscribers, clients or friends. Besides, there may be a necessity to send screenshotas, logs or reports. These may not be the most frequently arising tasks but having such a feature is clearly an advantage. The article deals with using several Google services simultaneously, developing an appropriate assembly on C# and integrating it with MQL tools.

![Library for easy and quick development of MetaTrader programs (part VIII): Order and position modification events](https://c.mql5.com/2/36/MQL5-avatar-doeasy__3.png)[Library for easy and quick development of MetaTrader programs (part VIII): Order and position modification events](https://www.mql5.com/en/articles/6595)

In the previous articles, we started creating a large cross-platform library simplifying the development of programs for MetaTrader 5 and MetaTrader 4 platforms. In the seventh part, we added tracking StopLimit orders activation and prepared the functionality for tracking other events involving orders and positions. In this article, we will develop the class for tracking order and position modification events.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/6651&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070484900760458928)

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