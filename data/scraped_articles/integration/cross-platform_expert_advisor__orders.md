---
title: Cross-Platform Expert Advisor: Orders
url: https://www.mql5.com/en/articles/2590
categories: Integration, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:18:20.468971
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/2590&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071733516472888681)

MetaTrader 5 / Integration


### Table of Contents

- [Introduction](https://www.mql5.com/en/articles/2590#Introduction)
- [Conventions](https://www.mql5.com/en/articles/2590#Conventions)

  - [MetaTrader 4](https://www.mql5.com/en/articles/2590#Metatrader4)
  - [MetaTrader 5, Netting](https://www.mql5.com/en/articles/2590#Metatrader5N)
  - [MetaTrader 5, Hedging](https://www.mql5.com/en/articles/2590#Metatrader5H)

- [Trade Identifier (Ticket)](https://www.mql5.com/en/articles/2590#TradeIdentifier)
- [States](https://www.mql5.com/en/articles/2590#States)
- [Volume](https://www.mql5.com/en/articles/2590#Volume)
- [Orders Container](https://www.mql5.com/en/articles/2590#OrdersContainer)
- [Example](https://www.mql5.com/en/articles/2590#Example)
- [Extensions](https://www.mql5.com/en/articles/2590#Extensions)

- [Conclusion](https://www.mql5.com/en/articles/2590#Conclusion)


### Introduction

MetaTrader 4 and MetaTrader 5 uses
different conventions in processing trade requests. This article
discusses the possibility of using a class object that can be used to
represent the trades processed by the server, in order for a
cross-platform expert advisor to further work on them, regardless of
the version of the trading platform and mode being used.

### Conventions

There are many differences on how
MetaTrader 4 and MetaTrader 5 process trade requests from the
terminal. In saving the details of trade requests processed by the
trade servers, we need to consider three different versions/modes in
both trading platforms: (1) MetaTrader 4, (2) MetaTrader 5, netting
mode, and (3) MetaTrader 5, hedging mode.

#### MetaTrader 4

In MetaTrader 4, when an expert advisor
successfully sends an order, it receives a ticket number, which the a
numerical identifier for that order. When modifying or closing the
order, the same ticket is normally used until it leaves the market.

The situation is a bit more
complicated when partially closing an order. The operation is done by
also using the [OrderClose](https://docs.mql4.com/trading/orderclose) function, by specifying the number of lots
less than the total lot size of the order in question. When such a
trade request is sent, the order (or order ticket) is closed with by
a certain lot size amount indicated in the function call. The
remaining lot size would then remain on the market as a new order
with the same type as the partially closed order. Since the
OrderClose function only returns a Boolean variable, there is no
quick way to get the new ticket other than re-examining the list of
orders currently active on the account. Note that it is not possible
to get the ticket using the OrderClose function alone. The function
returns a Boolean variable, whereas functions like OrderSend, in MQL4,
returns a valid ticket upon a successful transaction.

#### MetaTrader 5 (Netting)

The process of trade operations in
MetaTrader 5 will look quite complicated at first glance, but is much
simpler to manage than in MetaTrader 4. This would be true at least
for the trader's (non-programmer) side.

The default mode in MetaTrader 5 is
netting mode. Under this mode, the results of orders processed by the
server are consolidated into a single position. The volume of type of
this particular position can change over time, based on the volume
and types of orders entered. On the programmer's side, it is a little
complicated. Unlike in MQL4, where there is only a concept of orders,
the programmer will have to deal with three different types of tokens
used in trading. The following table shows some comparison between
the netting mode in MQL5 and its rough equivalent in MQL4:

| Artifact | MQL5 (Netting) | MQL4 (Rough Equivalent) |
| --- | --- | --- |
| Order | Trade request (pending or market) | Trade request (pending or market) |
| Deal | Deal(s) made based on a single order (market order, or executed<br> pending order) | Single market order as reflected on trading terminal |
| Position | Trades (consolidated) | Sum of all market orders on trading terminal (order types<br> apply) |

In MQL5, orders, once executed, are
unchangeable on the client-side, whereas in MQL4, some properties can
still be changed while an order is in the market. That is, in the
former, an order is simply a trade request sent to the server. On the
latter, it can be used to represent the trade request as well as the
result of such a request. In this aspect, MQL5 made the entire
process more complex in order to make it less ambiguous, since an
apparent distinction is made between a trade request and a trade
result. An order in MQL4 can enter the market and leave it with a
different configuration, whereas in MQL5 all trades can be traced
back to the orders or trade requests that triggered them.

When one sends a trade request, there
are only two outcomes: processed or not processed. If the trade was
not processed, it means that there was no deal, as the trade server
was not able to process it for some reason (usually due to errors).
Now, if the trade is processed, in MQL5, the client and the server
have a deal. In this case, the order can be fully executed or
partially executed.

MetaTrader 4 does not have this option,
as the order is only executed entirely or not (fill or kill).

One notable disadvantage with this mode
in MetaTrader 5 is that it does not allow hedging. The type of
position on a given symbol can change. For example, if there is a 0.1
lot long position on a given symbol, entering a sell order with a
volume of 1.0 will convert the position for that symbol into a short position, with a volume of 0.9 lot.

#### MetaTrader 5 (Hedging)

The hedging mode on MetaTrader 5
resemble the conventions used in MetaTrader 4. Rather than
consolidating all the processed trades into a single position,
hedging mode allows a symbol to have more than one position. A
position is generated whenever a pending order has triggered or a
trade request by market has been processed by the trade server.

| Artifact | MQL5 (Netting) | MQL4 (Rough Equivalent) |
| --- | --- | --- |
| Order | Trade request (pending or market) | Trade request (pending or market) |
| Deals | Deal(s) made based on a single order | Market orders as reflected on trading terminal |
| Position | Trades (consolidated) based on a single trade request | Order as reflected on trading terminal |

In order to make a cross-platform
expert advisor able to accommodate these differences, a possible
solution would be to have the expert advisor store the details of the
individual trades it has placed on the market. Every time a trade has
been successfully placed, a copy of the details of the order will be
saved on a class object, the COrder class. The following code shows
the declaration of its base class:

```
class COrderBase : public CObject
  {
protected:
   bool              m_closed;
   bool              m_suspend;
   long              m_order_flags;
   int               m_magic;
   double            m_price;
   ulong             m_ticket;
   ENUM_ORDER_TYPE   m_type;
   double            m_volume;
   double            m_volume_initial;
   string            m_symbol;
public:
                     COrderBase(void);
                    ~COrderBase(void);
   //--- getters and setters
   void              IsClosed(const bool);
   bool              IsClosed(void) const;
   void              IsSuspended(const bool);
   bool              IsSuspended(void) const;
   void              Magic(const int);
   int               Magic(void) const;
   void              Price(const double);
   double            Price(void) const;
   void              OrderType(const ENUM_ORDER_TYPE);
   ENUM_ORDER_TYPE   OrderType(void) const;
   void              Symbol(const string);
   string            Symbol(void) const;
   void              Ticket(const ulong);
   ulong             Ticket(void) const;
   void              Volume(const double);
   double            Volume(void) const;
   void              VolumeInitial(const double);
   double            VolumeInitial(void) const;
   //--- output
   virtual string    OrderTypeToString(void) const;
   //--- static methods
   static bool       IsOrderTypeLong(const ENUM_ORDER_TYPE);
   static bool       IsOrderTypeShort(const ENUM_ORDER_TYPE);
  };
```

Since the EA remembers its own trades,
it can work with greater independence on the conventions used by the
trading platform it is being run on. Its disadvantage, however, lies
in the fact that the instances of this class will persist only during
the expert advisor's operation. In case the expert advisor or trading
platform would need to be restarted, all the saved data would be loss
unless there is a mean to save and load the information.

### Trade Identifier (Ticket)

Another possible hindrance in using
instances of COrder in creating cross-platform compatible expert
advisors is on how the order (or position) ticket numbers should be
stored. The differences are summarized in the following table:

| Operation | MQL4 | MQL5 (Netting) | MQL5 (Hedging) |
| --- | --- | --- | --- |
| Sending an Order | New Order Ticket | New Position Ticket (no existing positions) or Existing Position Ticket (with existing position) | New Position Ticket |
| Partial Close | New Order Ticket | Same ticket (if with remainder), otherwise N/A | Same Ticket |

When sending an order, all the three
versions have different ways to represent the trade entered. In MQL4,
when a trade request was successful, a new order will be opened. This
new order will be represented by an identifier (order ticket). In
MQL5 netting mode, each trade request to enter a trade is represented
by an order ticket. However, an order ticket may not be the best way
to represent the trade entered, but the resulting position itself.
The reason is that unlike in MQL4, the order ticket cannot be used
directly when further working on the resulting trade it has entered
on the market (but getting the ticket number may be useful when
trying to get the position that resulted from executing a certain
order). Furthermore, when there is an existing position of the same
type, the position ticket will remain the same (unlike in MQL4). On
the other hand, in MQL5 hedging mode, each new deal generates a new
position (a rough equivalent of an MQL4 order ticket). However, its
difference in MQL4 is that, a single trade request always results in
a single order, whereas in MQL5 (hedging mode), it is possible for a
single order to have more than one deals (when the filling policy is
set other than SYMBOL\_FILLING\_FOK).

There is also another issue when
partially closing a market order (MQL4) or position (MQL5). As stated
earlier, in MQL4, when a particular ticket is closed at a volume less
than its total ( [OrderLots](https://docs.mql4.com/trading/orderlots)), the ticket representing the trade is
closed, while the remaining volume will be assigned a new ticket with
the same type as the one partially closed. This is a little different
in MQL5. In netting mode, it requires a trade on the opposite
direction (buy on sell, or sell on buy), in order to close a position
(either partially or in full). In hedging mode, the process is more
similar to MQL4 (OrderClose versus CTrade's [PositionClose](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradepositionclose)), but unlike in
MQL4, partially closing a position will not trigger any change on the
identifier representing it.

One way to solve this problem is to
split the implementation on how to represent the identifier for a
particular trade on two platforms. Since the an order ticket does not
change in MetaTrader 5, we can simply assign it a typical numerical
variable. On the other hand, for the MetaTrader 4 version, we will
use an instance of [CArrayInt](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayint) to store the ticket numbers. For COrderBase (and consequently for the MQL5 COrder version), the following code for the Ticket method will be used:

```
COrderBase::Ticket(const ulong value)
  {
   m_ticket=value;
  }
```

This method will be overridden on the MQL4 version, with the following code:

```
COrder::Ticket(const ulong ticket)
  {
   m_ticket_current.InsertSort((int)ticket);
  }
```

### States

Within the internal states of orders on
the cross-platform expert advisor, there would be at least two
possible states:

- Closed

- Suspend


The two states are very similar but
there is a fundamental difference. The closed state of an order
indicates that an order is already closed and the expert advisor
would need to archive the order in its internal data. This would be
roughly equivalent to moving an order to the history in MQL4. The
suspend state, on the other hand, would only happen when the expert
advisor failed to close an order or one of the stops linked to it. In
this case, the expert advisor can attempt to close the order again
(and its stops) until its it is fully closed.

### Volume

In MQL4, the calculation of volume is
straightforward. Whenever an expert advisor sends a trade request,
the volume of the request is also included, and it would be either
denied or accepted. This is equivalent to the Fill or Kill margin
policy in MQL5, which is also the default setting for the trade
object (CTrade and CExpertTrade). Getting the common feature would
give us the FOK margin policy. And in order to make the handling of
volume consistent between MQL4 and MQL5, one approach would be to
derive the volume of the instance of COrder based on the volume of
the trade request itself. However, this would mean that for the MQL5
version, we would need to stick to the FOK policy. It is still
possible to use the other margin policies, but the results would be
slightly different (i.e. the count of COrder instances on the MQL5 version on a
given test of the same EA may be greater).

### Orders Container

If the expert advisor is to handle more than one instance of COrder, some method of organization may be needed. One of the classes that will facilitate this is the orders container, or COrders. The class extends [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) and stores instances of COrder. This would allow for easy storage and retrieval of trades entered by the expert advisor. The base template for the said class is shown below:

```
#include <Arrays\ArrayObj.mqh>
#include "OrderBase.mqh"
class CExpertAdvisor;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class COrdersBase : public CArrayObj
  {
public:
                     COrdersBase(void);
                    ~COrdersBase(void);
   virtual bool      NewOrder(const ulong,const string,const int,const ENUM_ORDER_TYPE,const double,const double);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
COrdersBase::COrdersBase(void)
  {
   if(!IsSorted())
      Sort();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
COrdersBase::~COrdersBase(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool COrdersBase::NewOrder(const ulong ticket,const string symbol,const int magic,const ENUM_ORDER_TYPE type,const double volume,const double price)
  {
   COrder *order=new COrder(ticket,symbol,type,volume,price);
   if(CheckPointer(order)==POINTER_DYNAMIC)
      if(InsertSort(GetPointer(order)))
         order.Magic(magic);
   return false;
  }
//+------------------------------------------------------------------+
#ifdef __MQL5__
#include "..\..\MQL5\Order\Orders.mqh"
#else
#include "..\..\MQL4\Order\Orders.mqh"
#endif
//+------------------------------------------------------------------+
```

Since its main function is to store instances of COrder, it has to have a means to add instances of the said class. The default method Add of CArrayObj may not always be ideal, as an instance of COrder requires to be instantiated. For this, we would have the NewOrder method, which would already create a new instance of COrder and automatically add it as an array member:

```
bool COrdersBase::NewOrder(const ulong ticket,const string symbol,const int magic,const ENUM_ORDER_TYPE type,const double volume,const double price)
  {
   COrder *order=new COrder(ticket,symbol,type,volume,price);
   if(CheckPointer(order)==POINTER_DYNAMIC)
      if(InsertSort(GetPointer(order)))
         order.Magic(magic);
   return false;
  }
```

Now that the base template is done, other methods may be added to this class. One example is an OnTick method. In this method, the container class would simply iterate over the items it has stored (COrder). Another possibility is to have the class COrder have an OnTick method as well. Then, it can then be coded such that this method on COrders is called every tick.

### Example

Our example code tries to enter a long position. After the position has been entered, the details of the trade are then stored on an instance of COrder. This is achieved by calling the NewOrder method of COrders (which would deal with the creation of an instance of COrder).

Both versions will use instances of a trade object (CExpertTradeX), an orders object (COrders), and symbol object ( [CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo)). On the OnTick handler of the expert advisors, the trade object will try to enter a long position using its Buy method. The only difference between the two versions (MQL4 and MQL5) is on how the details of the trade is to be retrieved. For the MQL5 version, the details are retrieved by using [HistoryOrderSelect](https://www.mql5.com/en/docs/trading/historyorderselect) and other related functions. The order ticket is retrieved using the [ResultOrder](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctraderesultorder) method of the trade object. The implementation for this version is shown below:

```
ulong retcode=trade.ResultRetcode();
ulong order = trade.ResultOrder();
 if(retcode==TRADE_RETCODE_DONE)
 {
  if(HistoryOrderSelect(order))
  {
   ulong ticket=HistoryOrderGetInteger(order,ORDER_TICKET);
   ulong magic=HistoryOrderGetInteger(order,ORDER_MAGIC);
   string symbol = HistoryOrderGetString(order,ORDER_SYMBOL);
   double volume = HistoryOrderGetDouble(order,ORDER_VOLUME_INITIAL);
   double price=HistoryOrderGetDouble(order,ORDER_PRICE_OPEN);
   ENUM_ORDER_TYPE order_type=(ENUM_ORDER_TYPE)HistoryOrderGetInteger(order,ORDER_TYPE);
   orders.NewOrder((int)ticket,symbol,(int)magic,order_type,volume,price);
  }
 }
```

The trade object in MQL4 has less features than in the MQL5 version. One may extend the trade object for this version, or simply iterate over all the active orders on the account in order to get the trade just entered:

```
for(int i=0;i<OrdersTotal();i++)
{
 if(!OrderSelect(i,SELECT_BY_POS))
  continue;
 if(OrderMagicNumber()==12345)
  orders.NewOrder(OrderTicket(),OrderSymbol(),OrderMagicNumber(),(ENUM_ORDER_TYPE)OrderType(),OrderLots(),OrderOpenPrice());
}
```

The full code for the main header file is shown below:

(test\_orders.mqh)

```
#include <MQLx-Orders\Base\Trade\ExpertTradeXBase.mqh>
#include <MQLx-Orders\Base\Order\OrdersBase.mqh>
CExpertTradeX trade;
COrders orders;
CSymbolInfo symbolinfo;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   if(!symbolinfo.Name(Symbol()))
     {
      Print("failed to initialize symbol");
      return INIT_FAILED;
     }
   trade.SetSymbol(GetPointer(symbolinfo));
   trade.SetExpertMagicNumber(12345);
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
   if(!symbolinfo.RefreshRates())
   {
      Print("cannot refresh symbol");
      return;
   }
   if(trade.Buy(1.0,symbolinfo.Ask(),0,0))
     {
#ifdef __MQL5__
      int retcode=trade.ResultRetCode();
      ulong order = trade.ResultOrder();
      if(retcode==TRADE_RETCODE_DONE)
        {
         if(HistoryOrderSelect(order))
           {
            ulong ticket=HistoryOrderGetInteger(order,ORDER_TICKET);;
            ulong magic=HistoryOrderGetInteger(order,ORDER_MAGIC);
            string symbol = HistoryOrderGetString(order,ORDER_SYMBOL);
            double volume = HistoryOrderGetDouble(order,ORDER_VOLUME_INITIAL);
            double price=HistoryOrderGetDouble(order,ORDER_PRICE_OPEN);
            ENUM_ORDER_TYPE order_type=order_type;
            m_orders.NewOrder((int)ticket,symbol,(int)magic,order_type,volume,price);
           }
        }
#else
      for(int i=0;i<OrdersTotal();i++)
        {
         if(!OrderSelect(i,SELECT_BY_POS))
            continue;
         if(OrderMagicNumber()==12345)
            orders.NewOrder(OrderTicket(),OrderSymbol(),OrderMagicNumber(),(ENUM_ORDER_TYPE)OrderType(),OrderLots(),OrderOpenPrice());
        }
#endif
     }
   Sleep(5000);
   ExpertRemove();
  }
//+------------------------------------------------------------------+
```

The header file contains all of the needed code. Thus, the main source files will only need to have at least the pre-processor directive to include test\_orders.mqh:

(test\_orders.mq4 and test\_orders.mq5)

```
#include "test_orders.mqh"
```

Running the expert advisor on the platforms will give the following log entries:

In MetaTrader 4, the following log file will be generated:

Expert test\_orders EURUSD,H1: loaded successfully

test\_orders EURUSD,H1: initialized

test\_orders EURUSD,H1: open #358063536 buy 1.00 EURUSD at 1.12470 ok

test\_orders EURUSD,H1: ExpertRemove function called

test\_orders EURUSD,H1: uninit reason 0

Expert test\_orders EURUSD,H1: removed

The following shows a screen shot of the platform upon executing the EA. Note that since the EA calls the [ExpertRemove](https://www.mql5.com/en/docs/common/expertremove) function, it is automatically removed from the chart as soon as it executes its code (only single execution for the OnTick handler).

![](https://c.mql5.com/2/24/im1__1.png)

In MetaTrader 5, the log file generated will be almost the same:

Experts    expert test\_orders (EURUSD,M1) loaded successfully

Trades    '3681006': instant buy 1.00 EURUSD at 1.10669 (deviation: 10)

Trades    '3681006': accepted instant buy 1.00 EURUSD at 1.10669 (deviation: 10)

Trades    '3681006': deal #75334196 buy 1.00 EURUSD at 1.10669 done (based on order #90114599)

Trades    '3681006': order #90114599 buy 1.00 / 1.00 EURUSD at 1.10669 done in 275 ms

Experts    expert test\_orders (EURUSD,M1) removed

Unlike in the MetaTrader 4, the log messages displayed above are found on the Journal tab of the Terminal window (not on the Experts tab):

![](https://c.mql5.com/2/24/im2__1.png)

The EA also prints a message on the Experts tab. However, the message is not about the execution of the trade, but only a message showing that a call to the ExpertRemove function has been made, in comparison with MetaTrader 4, which displays the messages on the Experts tab:

![](https://c.mql5.com/2/24/im3__1.png)

### Extensions

Our current implementation lacks certain features, which are often used in real-world expert advisors, such as the following:

1\. Initial stoploss and takeprofit values for trades entered

2\. Modification of SL and TP levels (e.g. breakeven, trailingstop, or any custom method)

3\. Persistence of data - there are differences on how the two platforms save trades and their stop levels. Our class object reconciles the methods of the two, but are saved only on memory. Thus, we would need a method to make the data persistent. That is, we need a method for our expert advisors to save and load the orders information in events such as terminal restarts, or when switching charts in MetaTrader with the cross-platform expert advisor still loaded on that chart.

These will be covered in future articles.

### Conclusion

In this article, we have discussed one
method by which a cross-platform expert advisor would be able to save
the details of a trade request successfully processed by the trade
server as an instance of a class object. This object instance can
then be used by the expert advisor to further work on the trade it
represents, based on a particular strategy. A basic template has been
provided for this class object, which can be further developed in
order to be useful in more sophisticated trading strategies.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2590.zip "Download all attachments in the single ZIP archive")

[Orders.zip](https://www.mql5.com/en/articles/download/2590/orders.zip "Download Orders.zip")(208.87 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Cross-Platform Expert Advisor: The CExpertAdvisor and CExpertAdvisors Classes](https://www.mql5.com/en/articles/3622)
- [Cross-Platform Expert Advisor: Custom Stops, Breakeven and Trailing](https://www.mql5.com/en/articles/3621)
- [Cross-Platform Expert Advisor: Stops](https://www.mql5.com/en/articles/3620)
- [Cross-Platform Expert Advisor: Time Filters](https://www.mql5.com/en/articles/3395)
- [Cross-Platform Expert Advisor: Money Management](https://www.mql5.com/en/articles/3280)
- [Cross-Platform Expert Advisor: Signals](https://www.mql5.com/en/articles/3261)
- [Cross-Platform Expert Advisor: Order Manager](https://www.mql5.com/en/articles/2961)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/95624)**
(6)


![mbjen](https://c.mql5.com/avatar/avatar_na2.png)

**[mbjen](https://www.mql5.com/en/users/mbjen)**
\|
15 Oct 2017 at 23:14

**mbjen:**

Enrico is it possible to check if the order (or position) was partially closed.

In MT4 I have a new ticket in case of partial close. In fact, it is not a new entry but still previous entry. So I need separate such orders. The ones which indicate a true market entry and the ones which appear after partial close (not a new entry). Is it possible using your classes?

Or it would be one COrder object until the order closed fully? I mean after partial close no new COrder object?

If so how do I get the total order profit?

Also, how do I get initial [order ticket](https://www.mql5.com/en/docs/trading/ordergetticket "MQL5 documentation: OrderGetTicket function") or initial order type?

![mbjen](https://c.mql5.com/avatar/avatar_na2.png)

**[mbjen](https://www.mql5.com/en/users/mbjen)**
\|
16 Oct 2017 at 13:08

Suppose I have sell 5.00 #1 order at 1.09246. Next partial close buy 2.00 #2 at 1.08896. And final close remaining buy 3.00 #3 at 1.09161 (due to trailing stop actually).

This is my code:

```
   COrders *orders=order_manager.OrdersHistory();
   for(int i=order_manager.OrdersHistoryTotal()-1;i>=0;i--)
     {
      COrder *order=orders.At(i);
      if(!CheckPointer(order))
         continue;
      order.OnTick();
      if(!order.IsClosed())
         continue;
      if(order.OrderType()!=0 && order.OrderType()!=1)
         continue;
```

To simplify let's call all these 3 orders as position.

IsClosed() method returns true when the position closes (at last order #3). But how do I get the position type, position first order ticket, [position open price](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_double "MQL5 documentation: Position Properties"). Position initial volume looks like can be calculated by using InitVolume() method. What about the rest?

![Enrico Lambino](https://c.mql5.com/avatar/2014/10/54465D5F-0757.jpg)

**[Enrico Lambino](https://www.mql5.com/en/users/iceron)**
\|
17 Oct 2017 at 18:33

**mbjen:**

Suppose I have sell 5.00 #1 order at 1.09246. Next partial close buy 2.00 #2 at 1.08896. And final close remaining buy 3.00 #3 at 1.09161 (due to trailing stop actually).

This is my code:

To simplify let's call all these 3 orders as position.

IsClosed() method returns true when the position closes (at last order #3). But how do I get the position type, position first order ticket, [position open price](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_double "MQL5 documentation: Position Properties"). Position initial volume looks like can be calculated by using InitVolume() method. What about the rest?

Not entirely sure at what you are trying to do, but to get the unrealized profit/loss in MQL5 on a COrder instance, one way is to get the remaining volume, and then multiply it with the difference between the current market price (bid or ask) and the COrder entry price, and then multiply it with the tick value. The trickier part is when the symbol's point is not equal to the tick size (you will need to divide the difference by the tick size). Gold used to be like this, but not anymore as far as I know. It is simply better to find a broker that offers greater liquidity. The same COrder instance is used until the end. In MQL4, you just need to select the order ticket and call the OrderProfit() function.

![mbjen](https://c.mql5.com/avatar/avatar_na2.png)

**[mbjen](https://www.mql5.com/en/users/mbjen)**
\|
30 Oct 2017 at 08:29

2017/10/30 09:04:38 Completed #169758 keltrem


![Alain Verleyen](https://c.mql5.com/avatar/2024/5/663a6cdf-e866.jpg)

**[Alain Verleyen](https://www.mql5.com/en/users/angevoyageur)**
\|
15 Dec 2017 at 02:21

When one sends a trade request, there are only two outcomes: processed or not processed. If the trade was not processed, it means that there was no deal, as the trade server was not able to process it for some reason (usually due to errors). Now, if the trade is processed, in MQL5, the client and the server have a deal. In this case, the order can be fully executed or partially executed.

That's not totally exact, you can also have a timeout. Which means you don't know if the order was processed or not. Of course in the end, an order is either processed or not, but it's important to know and process timeout on a live account.

MetaTrader 4 does not have this option, as the order is only executed entirely or not (fill or kill).

...

In MQL4, the calculation of volume is straightforward. Whenever an expert advisor sends a trade request, the volume of the request is also included, and it would be either denied or accepted.

That's not exact. MT4 can also have partially [filled orders](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#symbol_filling_mode "MQL5 documentation: Symbol properties"). Of course it should be rare on Forex which is mainly traded by MT4, but it may happen technically.

* * *

In general I don't see the usefulness of the classes your provided in this article, as you still need specific MT4/MT5 in the main code, and compiler directives. (Maybe it's addressed on further articles, I didn't read them yet).


![MQL5 vs QLUA - Why trading operations in MQL5 are up to 28 times faster?](https://c.mql5.com/2/24/speed_over_28_03.png)[MQL5 vs QLUA - Why trading operations in MQL5 are up to 28 times faster?](https://www.mql5.com/en/articles/2635)

Have you ever wondered how quickly your order is delivered to the exchange, how fast it is executed, and how much time your terminal needs in order to receive the operation result? We have prepared a comparison of trading operation execution speed, because no one has ever measured these values using applications in MQL5 and QLUA.

![MQL5 Cookbook - Trading signals of moving channels](https://c.mql5.com/2/24/ava2.png)[MQL5 Cookbook - Trading signals of moving channels](https://www.mql5.com/en/articles/1863)

The article describes the process of developing and implementing a class for sending signals based on the moving channels. Each of the signal version is followed by a trading strategy with testing results. Classes of the Standard Library are used for creating derived classes.

![LifeHack for trader: "Quiet" optimization or Plotting trade distributions](https://c.mql5.com/2/24/avaf2i.png)[LifeHack for trader: "Quiet" optimization or Plotting trade distributions](https://www.mql5.com/en/articles/2626)

Analysis of the trade history and plotting distribution charts of trading results in HTML depending on position entry time. The charts are displayed in three sections - by hours, by days of the week and by months.

![How to quickly develop and debug a trading strategy in MetaTrader 5](https://c.mql5.com/2/24/avae17.png)[How to quickly develop and debug a trading strategy in MetaTrader 5](https://www.mql5.com/en/articles/2661)

Scalping automatic systems are rightfully regarded the pinnacle of algorithmic trading, but at the same time their code is the most difficult to write. In this article we will show how to build strategies based on analysis of incoming ticks using the built-in debugging tools and visual testing. Developing rules for entry and exit often require years of manual trading. But with the help of MetaTrader 5, you can quickly test any such strategy on real history.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/2590&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071733516472888681)

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