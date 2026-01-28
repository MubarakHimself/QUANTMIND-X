---
title: Cross-Platform Expert Advisor: Order Manager
url: https://www.mql5.com/en/articles/2961
categories: Integration, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:17:50.441183
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=qkycdnikrvknrllqjwgqnopegkgxwpui&ssn=1769192268440107870&ssn_dr=0&ssn_sr=0&fv_date=1769192268&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2961&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Cross-Platform%20Expert%20Advisor%3A%20Order%20Manager%20-%20MQL5%20Articles&scr_res=1920x1080&ac=1769192268921179&fz_uniq=5071727512108608844&sv=2552)

MetaTrader 5 / Integration


### Table of Contents

- [Introduction](https://www.mql5.com/en/articles/2961#intro)
- [Objectives](https://www.mql5.com/en/articles/2961#obj)
- [Base Implementation](https://www.mql5.com/en/articles/2961#base-implement)

  - Calculation of Trade Volume
  - Calculation of Stoploss and Takeprofit
  - Closing an Order or Position
  - Validation of Settings
  - Counting of the Number of Trades
  - Archiving a COrder Instance

- [MQL4-Specific Implementation](https://www.mql5.com/en/articles/2961#mql4-implement)

  - Opening an Order or Position
  - Closing an Order or Position


- [MQL5-Specific Implementation](https://www.mql5.com/en/articles/2961#mql5-implement)

  - Opening an Order or Position
  - Closing an Order or Position

- [Creation of an Instance of COrder](https://www.mql5.com/en/articles/2961#corder-instance)
- [Modification of Order or Position](https://www.mql5.com/en/articles/2961#corder-modify)

- [Example](https://www.mql5.com/en/articles/2961#example)
- [Overview of Structure](https://www.mql5.com/en/articles/2961#overview)

- [Conclusion](https://www.mql5.com/en/articles/2961#conclusion)


### Introduction

As also discussed in earlier articles in the series, MetaTrader 4 and MetaTrader 5 has certain differences that make it difficult to simply copy MQL4 source file and compile them using the MQL5 compiler. One of the most prominent differences between the two is on how the two platforms differ in executing trade operations. This article deals with the creation of the COrderManager class. The said class, along with other helper classes, would be mainly responsible for the execution of trades within the expert advisor as well as in maintenance of the trades entered by the expert.

### Objectives

The order manager in this
article will be capable of executing the following operations:

1. Lotsize Calculation
2. Stoploss and
    Takeprofit
3. Misc Parameters
    Needed for Entry (Expiration, Comment, Magic)
4. Some Preconditions
    before sending an order
5. Management of Orders
    and Orders History

The part involving
lotsize calculation is often best delegated to an object member since
there are many ways by which the optimum lotsize for the next trade
can be calculated (depending on the trading strategy involved). The
same is true for the calculation of Stoploss and Takeprofit levels.

Miscellaneous parameters
needed to enter an order, such as expiration, order comment, and
magic number require less complexity and so are best handled by the
order manager object itself.

Certain preconditions are
required before a trade can be entered on the market by an expert
advisor. These include trading signals based on market conditions,
time constraints, as well as the maximum number of active trades at
any given time and the maximum number of trades to open in its entire
lifetime (between [OnInit and OnDeinit](https://www.mql5.com/en/docs/basis/function/events)). For the order manager, the
last two preconditions will be included as a condition before an
order is finally sent to the market. This will be implemented so that
duplicate trades will not occur during the operation of the expert
advisor. The other preconditions, on the other hand, can be delegated
to certain components external to the order manager.

### Base Implementation

Similar to other class objects discussed in this article series, a common ground between MQL4 and MQL5 must be sought, and implemented in the base class, while the parts where the implementation diverges are implemented in language-specific descendants of the base class. In developing the base class, we have to understand the following areas on the processing of trade requests:

1. How trade requests
    are sent is different
2. How trade actions
    are documented is different
3. There are certain
    features in MQL5 that have no MQL4 equivalents

There are some components that are different between MQL4 and MQL5 in this area. Consider the OrderSend function ( [mql4](https://docs.mql4.com/trading/ordersend), [mql5](https://www.mql5.com/en/docs/trading/ordersend)), as shown in the documentations for the two platforms:

(MQL4)

intOrderSend(

string   symbol,              // symbol

int      cmd,                 // operation

double   volume,              // volume

double   price,               // price

int      slippage,            // slippage

double   stoploss,            // stop loss

double   takeprofit,          // take profit

string   comment=NULL,        // comment

int      magic=0,             // magic number

datetime expiration=0,        // pending order expiration

color    arrow\_color=clrNONE// color

    );

(MQL5)

boolOrderSend(

MqlTradeRequest&  request,      // query structure

MqlTradeResult&   result        // structure of the answer

    );

The MQL4 function has a
more straightforward approach. The MQL5 function, on the other hand,
is a bit more complicated, but reduces the number of parameters to
just two, which contain the data ( [struct](https://www.mql5.com/en/docs/basis/types/classes)) for the request and the
result, respectively. This obstacle has been largely addressed on a
previous article regarding the importation of certain components of
the MQL5 Standard Library into MQL4, particularly the CExpertTrade
and CExpertTradeX classes. Thus, the order manager will simply
utilize these classes to ensure compatibility between the two
languages when issuing trade requests.

Another aspect is the way
the exit of the trade or the deletion of an order is handled in
MetaTrader 4 and MetaTrader 5. While there is not much difference in
the way pending orders are deleted, there is a huge difference in the
way market orders (MQL4) or positions (MQL5) are exited from the
market. In MQL4, exiting a market order is achieved by calling the [OrderClose](https://docs.mql4.com/trading/orderclose) function. In MetaTrader 5, the same effect is realized by
calling the [PositionClose](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradepositionclose) function or issuing a trade request with
the same volume as the current position and opposite the current
position.

In MetaTrader 5, every
trade action is documented. Whether an action is an trade entry,
modification, or exit, that action would leave a footprint, and the
data concerning these actions are accessible to the expert advisor.
This is not the case in MetaTrader 4. For example, when a ticket ID
is assigned to a pending order, the same ID is often used throughout
the lifetime of that order, even if it has hit its trigger price and
became a market order before it has left the market. In order to see
the full progression of a certain trade, one has to look at the
expert and journal log files, which can be a time-consuming task.
Furthermore, log files are meant to be read by humans, and there are
no built-in MQL4 function(s) that would make it easier for an expert
advisor to access those information.

There are certain
features in MQL5 that are simply not available in MQL4. An example of
this is the order [filling type](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_integer). MQL5 has the following volume filling
options for an order:

- ORDER\_FILLING\_FOK –
when the requested volume cannot be filled, cancel the order
- ORDER\_FILLING\_IOC –
when the requested volume cannot be filled, use the maximum volume
available and cancel the remaining volume.
- ORDER\_FILLING\_RETURN –
when the requested volume cannot be filled, use the maximum volume available. The order with
remaining volume stays on the market.

In MetaTrader 4, a trade
request is simply filled or not (cancelled), which is essentially
equivalent to ORDER\_FILLING\_FOK, while the other two options are
unavailable. However, these filling policies are only implemented
whenever the requested volume exceeds the available volume on the
market, which does not happen often especially for setups with low
risk and/or low account balance. ORDER\_FILLING\_IOC and
ORDER\_FILLING\_RETURN can be difficult, if not impossible or impractical
to implement in MQL4, primarily because expert advisors do not have
any means to determine how much volume is available for a certain
trade request (and if it were so, such information can be highly
volatile and would be frequently subject to change). Thus, to ensure
compatibility between MQL4 and MQL5, ORDER\_FILLING\_FOK will be the
only filling option used (which is also the default in MetaTrader 5).
Meanwhile, there are events where an expert calculates a lot size for
a trade request which exceeds SYMBOL\_VOLUME\_MAX, which is the maximum
allowable volume for any trade/deal set by the broker. MetaTrader 5
addresses this by automatically splitting the trade into several
deals, but this feature is not available in MetaTrader 4 (causing the
cancellation of the trade request). Thus, it is better for a cross
platform expert advsior to check this in advance (preferably after
deriving or calculating the volume to trade) prior to sending a trade
request for market entry using the order manager.

The following figure
broadly illustrates how the order manager will implement the entry of
trades:

![Diagram for Entry](https://c.mql5.com/2/25/TradeOpen.png)

As shown in the figure, the method starts by preparing the data needed for the operation. If the position is allowed to be entered and the preconditions are satisfied, the method proceeds with the entry of the order. Otherwise, the process ends. Before sending the trade requests, necessary values will need to be calculated. If the request is successful, the result is checked and a new instance of COrder is created, which is then added to the list of current orders/positions (m\_orders).

Methods that only perform pure calculation can be found within the
base class. The methods invoking functions that are different between
the two languages extends the base methods into their respective
language-specific classes. On this method, however, there is very little difference on how the operation is to be performed. Thus, the base class method is purely virtual, and implementations can be found separately between the two versions. We will find the implementation for the base class method as follows:

COrder\* COrderManagerBase::TradeOpen(conststring,ENUM\_ORDER\_TYPE)

{

returnNULL;

}

#### Calculation of Trade  Volume

As mentioned earlier,
calculation of trade volume for the next trade is best delegated to
another class object, which will be a class member of the order
manager. This approach is also used in the Experts library of the
MQL5 Standard Library. The following shows the code for the
LotSizeCalculate method, which is the method responsible for
calculating the volume of the next trade:

double COrderManagerBase::LotSizeCalculate(constdouble price,constENUM\_ORDER\_TYPE type,constdouble stoploss)

{

if(CheckPointer(m\_moneys))

return m\_moneys.Volume(m\_symbol.Name(),0,type,stoploss);

return m\_lotsize;

}

The method checks of a
pointer to an instance of CMoneys, which is just a container for the
money management objects used by the order manager (in the same way
that COrders is a container for instances of COrder). This money
management object will be discussed in a separate article. At this
point at least, it is sufficient to know that there is a separate
component that deals with the calculation of the lotsize, and that
the lotsize calculated will be valid. In the event that no money
management instance was provided to the order manager, the order
manager would simply use the default lotsize through its class member, m\_lotsize.

#### Calculation of Stop Loss  and Takeprofit

The calculation of
Stoploss and Takeprofit is achieved through the methods
StopLossCalculate and TakeProfitCalculate, respectively. The
following code snippets show how each of these methods are
implemented in the order manager:

double COrderManagerBase::StopLossCalculate(constENUM\_ORDER\_TYPE type,constdouble price)

{

if(CheckPointer(m\_main\_stop))

return m\_main\_stop.StopLossTicks(type,price);

return0;

}

double COrderManagerBase::TakeProfitCalculate(constENUM\_ORDER\_TYPE type,constdouble price)

{

if(CheckPointer(m\_main\_stop))

return m\_main\_stop.TakeProfitTicks(type,price);

return0;

}

The calculation of the
stop levels are delegated to a separate class object, which will be a
member of the order manager (will be discussed on another article).
The stop object will also have its separate implementations for
MetaTrader 4 and MetaTrader 5. However, in the event that no pointer
to a stop object is supplied to order manager, the calculated
Stoploss or Takeprofit would default to zero (no SL/TP).

#### Closing an Order or Position

The following figure
broadly illustrates how the order manager will close positions and
delete orders:

![diagram for exit](https://c.mql5.com/2/25/CloseOrder.png)

As shown in the figure,
the method first checks if the pointer to a instance of COrder is
valid. It then proceeds to getting the correct instances to the
symbol and trade objects which are needed to process the exit
request. It then either deletes or closes the order depending on its
type. After a successful closing/deletion of the order, the COrder instance is then moved from the list of active orders to the list of
historical orders for the order manager (archived). The method would also set
the flags that would mark the COrder object as closed.

#### Validation of Settings

The validation of
settings on the order manager will be achieved by calling the
Validate method of the class. The code for the said method is shown
in the following snippet:

bool COrderManagerBase::Validate(void) const

{

if(CheckPointer(m\_moneys)==POINTER\_DYNAMIC)

      {

if(!m\_moneys.Validate())

returnfalse;

      }

if(CheckPointer(m\_stops)==POINTER\_DYNAMIC)

      {

if(!m\_stops.Validate())

returnfalse;

      }

returntrue;

}

This code is similar to
the [ValidationSettings](https://www.mql5.com/en/search#!keyword=validationsettings) method which is often found in the in some
classes within the MQL5 Standard Library. It simply calls the
Validate methods of its object members, and returns false whenever
one of these objects fails validation (ultimately causing the OnInit
function of the expert to fail or return INIT\_FAILED). The method is meant to be called
during execution of the Initialization function of the expert
advisor.

#### Counting of the Number of  Trades

The trades total is the
total number of trades the order manager has entered thus far,
including those already in the history, since the expert advisor or
script, is started. The orders total refers to the
number of current trades on the account, and the orders history total refers to those found within the order manager's own tally of historical orders. Thus:

int COrderManagerBase::OrdersTotal(void) const

{

return m\_orders.Total();

}

int COrderManagerBase::OrdersHistoryTotal(void) const

{

return m\_orders\_history.Total();

}

int COrderManagerBase::TradesTotal(void) const

{

return m\_orders.Total()+m\_orders\_history.Total()+m\_history\_count;

}

In both cases, we use the
standard concept of MetaTrader 4 when counting orders (each order is
one position, whether market or pending). For example, in the MQL4
implementation, we would see the first two lines of the TradeOpen
method as follows:

int trades\_total =TradesTotal();

int orders\_total = OrdersTotal();

It is important to note
that here, OrdersTotal refers to the method native to the class, not
the OrdersTotal function that is found in both MQL4 and MQL5. If we
are to use the native function in the language instead, OrdersTotal
should be invoked with a scope resolution operator preceding the name
of the function:

int orders\_total = ::OrdersTotal();

#### Archiving a COrder Instance

Since the order manager will keep its own independent tally of trades entered the way MetaTrader 4 and MetaTrader 5 does (but compatible with both), it has to have a way to flag instances of COrder as already belonging to history. The orders are stored in m\_orders and m\_orders\_history, which are instances of COrders, depending on whether they are already history, or still active in the market. As a consequence, for both versions, the cross-platform expert would need to check if a given order or trade has already left the market.

Since the two
platforms differ in the way they document trades entered on the market, the
order manager will need to keep its own independent record of trades
entered. Upon a successful opening of an order, a COrder instance
will be created, which will be eventually added to m\_orders. As soon
as order or position it represents exits the market, the order
manager will then have to move the COrder instance to
m\_orders\_history. The ArchiveOrder method of the class, which will be
used for both versions, is shown below:

bool COrderManagerBase::ArchiveOrder(COrder \*order)

{

return m\_orders\_history.Add(order);

}

### MQL4-Specific Implementation

#### Opening an Order or Position

The following snippet of
code shows the TradeOpen method of the MQL4-specifc class descendant
of CorderManagerBase:

COrder\* COrderManager::TradeOpen(conststring symbol,ENUM\_ORDER\_TYPE type)

{

int trades\_total = TradesTotal();

int orders\_total = OrdersTotal();

    m\_symbol = m\_symbol\_man.Get(symbol);

if (!CheckPointer(m\_symbol))

returnNULL;

if(!IsPositionAllowed(type))

returnNULL;

if(m\_max\_orders>orders\_total && (m\_max\_trades>trades\_total \|\| m\_max\_trades<=0))

      {

ENUM\_ORDER\_TYPE ordertype = type;

double price=PriceCalculate(ordertype);

double sl=0,tp=0;

if(CheckPointer(m\_main\_stop)==POINTER\_DYNAMIC)

         {

          sl = m\_main\_stop.StopLossCustom()?m\_main\_stop.StopLossCustom(symbol,type,price):m\_main\_stop.StopLossCalculate(symbol,type,price);

          tp = m\_main\_stop.TakeProfitCustom()?m\_main\_stop.TakeProfitCustom(symbol,type,price):m\_main\_stop.TakeProfitCalculate(symbol,type,price);

         }

double lotsize=LotSizeCalculate(price,type,sl);

ulong ticket = SendOrder(type,lotsize,price,sl,tp);

if (ticket>0)

       {

if (OrderSelect((int)ticket,SELECT\_BY\_TICKET))

return m\_orders.NewOrder(OrderTicket(),OrderSymbol(),OrderMagicNumber(),(ENUM\_ORDER\_TYPE)::OrderType(),::OrderLots(),::OrderOpenPrice());

       }

      }

returnNULL;

}

The function accepts two
parameters, the name of symbol or instrument, and the type of order
to open. It starts by getting the values needed for processing the
request: the trades total, orders total, and the symbol object for
the order (as indicated by the first argument of the method).

Upon satisfying the 2
preconditions (max orders and max trades), the method proceeds to the
calculation of the main stop loss and take profit levels, and the
volume of the trade, using special object members. Finally, it sends
the order, and after a successful operation, creates a new instance
of COrder and stores it in the list of active orders (within the
order manager).

#### Closing an Order or Position

The following snippet of
code shows the CloseOrder method of the MQL4-specifc class descendant
of COrderManagerBase:

```
bool COrderManager::CloseOrder(COrder *order,const int index=-1)
  {
   bool closed=true;
   if(CheckPointer(order)==POINTER_DYNAMIC)
     {
      if(!CheckPointer(m_symbol) || StringCompare(m_symbol.Name(),order.Symbol())!=0)
         m_symbol=m_symbol_man.Get(order.Symbol());
      if(CheckPointer(m_symbol))
         m_trade=m_trade_man.Get(order.Symbol());
      if(order.Volume()>0)
        {
         if(order.OrderType()==ORDER_TYPE_BUY || order.OrderType()==ORDER_TYPE_SELL)
            closed=m_trade.OrderClose((ulong)order.Ticket());
         else
            closed=m_trade.OrderDelete((ulong)order.Ticket());
        }
      if(closed)
        {
         int idx = index>=0?index:FindOrderIndex(GetPointer(order));
         if(ArchiveOrder(m_orders.Detach(idx)))
           {
            order.Close();
            order.Volume(0);
           }
        }
     }
   return closed;
  }
```

As we will later see, the
MQL4 version is much simpler than the MQL5 version, primarily because
it only has one margin mode (hedging). And even if hedging is
disabled by the MetaTrader 4 broker, the process of closing an order
will remain the same: delete if pending, close if market.

The function accepts two
parameters, the order object, and its index on the list of active
orders/positions. If the pointer to the COrder object is valid, we
then get the correct instance of CExpertTradeX and [CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo) to
close the order, and then put it on the history of the trading
terminal by calling the appropriate function.

As soon as the order or
position is exited, the COrder object will need to be updated. First,
it is removed from the list of active orders, and then transferred at
the end of the list of the historical orders. Then, the object is
flagged as closed, and its volume zeroed out.

The second argument of the method of the class accepts an optional parameter (index). This is to speed up the processing of the request if the index of the COrder instance in an orders array is already known in advance (this is usually the case since orders often need to be iterated). In the event that the index is not known, then the method can be called with only a single argument and the method would call another class method, FindOrderIndex, which would be responsible for locating the position of the COrder instance in the order array.

### MQL5-Specific Implementation

#### Opening an Order or Position

The following snippet of
code shows the TradeOpen method of the MQL5-specifc class descendant
of COrderManagerBase:

COrder\* COrderManager::TradeOpen(conststring symbol,ENUM\_ORDER\_TYPE type)

{

double lotsize=0.0,price=0.0;

int trades\_total =TradesTotal();

int orders\_total = OrdersTotal();

    m\_symbol=m\_symbol\_man.Get(symbol);

if(!IsPositionAllowed(type))

returnNULL;

if(m\_max\_orders>orders\_total && (m\_max\_trades>trades\_total \|\| m\_max\_trades<=0))

      {

       price=PriceCalculate(type);

       lotsize=LotSizeCalculate(price,type,m\_main\_stop==NULL?0:m\_main\_stop.StopLossCalculate(symbol,type,price));

if (SendOrder(type,lotsize,price,0,0))

return m\_orders.NewOrder((int)m\_trade.ResultOrder(),m\_trade.RequestSymbol(),    (int)m\_trade.RequestMagic(),m\_trade.RequestType(),m\_trade.ResultVolume(),m\_trade.ResultPrice());

      }

returnNULL;

}

As we can see, the
implementation does not differ greatly from the MQL4 one. However,
one of the key difference is that in this code, we do not need to get
the values for the Stoploss and Takeprofit levels. The reason for
this is that how the stop levels behave in MetaTrader 5 is different
from that of its predecessor. A programmer who has experience in
working with MQL5 Expert Advisors will know that in this library, we
are going to use pending orders as a counterpart for broker-based
stops in MQL4.

#### Closing an Order or Position

The following snippet of
code shows the CloseOrder method of the MQL5-specifc class descendant
of COrderManagerBase:

bool COrderManager::CloseOrder(COrder \*order,constint index=-1)

{

bool closed=true;

    COrderInfo ord;

if(!CheckPointer(order))

returntrue;

if(order.Volume()<=0)

returntrue;

if(!CheckPointer(m\_symbol) \|\| StringCompare(m\_symbol.Name(),order.Symbol())!=0)

       m\_symbol=m\_symbol\_man.Get(order.Symbol());

if(CheckPointer(m\_symbol))

       m\_trade=m\_trade\_man.Get(order.Symbol());

if(ord.Select(order.Ticket()))

    {

       closed=m\_trade.OrderDelete(order.Ticket());

    }

else

      {

ResetLastError();

if(IsHedging())

       {

          closed=m\_trade.PositionClose(order.Ticket());

       }

else

         {

if(COrder::IsOrderTypeLong(order.OrderType()))

             closed=m\_trade.Sell(order.Volume(),0,0,0);

elseif(COrder::IsOrderTypeShort(order.OrderType()))

             closed=m\_trade.Buy(order.Volume(),0,0,0);

         }

      }

if(closed)

      {

if(ArchiveOrder(m\_orders.Detach(index)))

         {

          order.Close();

          order.Volume(0);

         }

      }

return closed;

}

In closing an order or a
position in MQL5, we have to consider the margin mode (netting or
hedge). But first we need to identify whether or not the item to be
closed is an MQL5 order or an MQL5 position. We can use the functions
OrderSelect and HistoryOrderSelect to accomplish this, but in order
to shorten the code needed for this method and make the process
easier, we will simply use the COrderInfo class from the MQL5
Standard Library to accomplish this.

An order in MQL5 comes as
a result of a trade request, which would often result to a deal or a
set of deals (roughly equivalent to a MetaTrader 4 market order).
However, if the request is not instant execution, then it refers to a
pending order (as opposed in MetaTrader 4, where orders can be market
or pending). Now, to differentiate the item to exit, the method first
checks if the item is a pending order using COrderInfo, and if
so, that pending order is deleted. If it fails selection by COrderInfo, then we are certain that it is a market order or a
position. For hedging mode, the position is simply closed using
PositionClose function. Otherwise, in netting mode, we simply
neutralize the position by entering an opposite position with
equivalent volume.

### Creation of an Instance of  COrder

We have seen how the
order manager opens and closes a trade. In a previous article, we
have also shown a way on how the CExpertTrade Class can be altered so
as to make it compatible with both trading platforms. Now, there is a
difference in how Stoploss and Takeprofit levels are implemented in
the two trading platforms, and this is only barely processed by the
order manager. The rest of the process is set upon the initialization
of the COrder instance, which is invoked on the NewOrder method of COrders. The following code shows the Init method of COrdersBase:

COrder\* COrdersBase::NewOrder(constulong ticket,conststring symbol,constint magic,constENUM\_ORDER\_TYPE type,constdouble volume,constdouble price)

{

    COrder \*order=new COrder(ticket,symbol,type,volume,price);

if(CheckPointer(order)==POINTER\_DYNAMIC)

if(InsertSort(GetPointer(order)))

       {

          order.Magic(magic);

          order.Init(GetPointer(this),m\_stops);

return order;

       }

returnNULL;

}

As we can see, the Init
method of the COrder class accepts a certain custom object (CStops)
as its second argument. This is a container for stop objects (like
m\_main\_stop object shown earlier). This class object will be
discussed on a separate article.

### Modification of Order or  Position

We have not shown any
code yet that would allow the order manager to modify an existing position. This
would be delegated to another stop object (CStop and CorderStop),
which will be discussed on a separate article. These objects would be
responsible for any update or modification of the stop levels of the
position, as well as coordination with the COrder object that they
belong to.

In MetaTrader 4, the
pending order entry price can be modified any number of times. This
is not the case in MetaTrader 5. This time, the MQL5 version is the
limiting component, so we will adopt the MQL5 standard. Modifying a
pending order will require the deletion of the existing pending order
and the creation of a new one with updated properties.

### Example

As an example, we will implement an expert advisor using the class objects discussed so far in this article series. After creating the expert advisor source file on MetaEditor, we will begin with the reference to the library:

```
#include "MQLx\Base\OrderManager\OrderManagerBase.mqh"
```

Note that in this statement, we are using quotation marks rather than "<" and ">". We are going to put the library on the same directory as the expert advisor source code file.

For this EA, we are going to require at least three pointers, which has to be declared globally within the program (COrderManager, CSymbolInfo, and CSymbolManager):

```
COrderManager *order_manager;
CSymbolManager *symbol_manager;
CSymbolInfo *symbol_info;
```

Under the OnInit function, we will have to initialize these three pointers, especially for the instance of CSymbolInfo, which requires a particular name of an instrument to assign to during initialization.

```
int OnInit()
  {
//---
   order_manager = new COrderManager();
   symbol_manager = new CSymbolManager();
   symbol_info = new CSymbolInfo();
   if (!symbol_info.Name(Symbol()))
   {
      Print("symbol not set");
      return (INIT_FAILED);
   }
   symbol_manager.Add(GetPointer(symbol_info));
   order_manager.Init(symbol_manager,NULL);
//---
   return(INIT_SUCCEEDED);
  }
```

Under OnDeinit, we will have to delete these three pointers so they won't leak memory (at least within the trading platform):

```
void OnDeinit(const int reason)
  {
//---
   delete symbol_info;
   delete symbol_manager;
   delete order_manager;
  }
```

Under OnTick, we have to implement the actual strategy. The expert in this example will use a simple method of detecting a new bar (by checking the count of bars on the chart). The previous bar count has to be stored in a static variable (or global variable). The same is true for the direction variable, which will be used to store the previous direction the expert previously took (or zero, if it's the first time to trade). However, since the function used to counting the bars on the chart differ between the two platforms, we will have to split the implementation in this regard:

```
static int bars = 0;
static int direction = 0;
int current_bars = 0;
#ifdef __MQL5__
   current_bars = Bars(NULL,PERIOD_CURRENT);
#else
   current_bars = Bars;
#endif
```

For the MQL4 version, we are simply using the predefined variable named [Bars](https://docs.mql4.com/predefined) (a call to the function [iBars](https://docs.mql4.com/series/ibars), can also be used). On the other hand, for the MQL5 version, we are using a call to the [Bars](https://www.mql5.com/en/docs/series/bars) function.

The next code snippet implements the actual behavior of the expert that can be observed. If a discrepancy between the previous and current bars has been detected, the expert begins by initializing the rates of the symbol (CSymbolInfo), to be used for further operations. It then checks if there is a previous trade to close. If one was found, the expert closes it, and proceeds to processing the entry of another trade, based on the previous direction. The code ends by updating the count of bars on the EA.

```
if (bars<current_bars)
   {
      symbol_info.RefreshRates();
      COrder *last = order_manager.LatestOrder();
      if (CheckPointer(last) && !last.IsClosed())
         order_manager.CloseOrder(last);
      if (direction<=0)
      {
         Print("Entering buy trade..");
         order_manager.TradeOpen(Symbol(),ORDER_TYPE_BUY,symbol_info.Ask());
         direction = 1;
      }
      else
      {
         Print("Entering sell trade..");
         order_manager.TradeOpen(Symbol(),ORDER_TYPE_SELL,symbol_info.Bid());
         direction = -1;
      }
      bars = current_bars;
   }
```

To ensure that we use the same code for the initial version and for all future versions of the expert advisor, we move the code we have made so far in a header file, and then reference it in the main source file (each for the MQL4 and MQL5 versions). Both the source files (test\_ordermanager.mq4 or test\_ordermanager.mq5, depending on the target platform) will have a single line of code referencing the main header file:

```
#include "test_ordermanager.mqh"
```

The following tables show the results of running the expert advisor on MetaTrader 4, and the netting and hedging modes of MetaTrader 5, as they appeared on their respective strategy tester reports. For the sake of brevity, the first 10 trades were only included in the article (the full reports can be found on the zip package at the end of this article).

MT4:

|     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| # | Time | Type | Order | Size | Price | S / L | T / P | Profit | Balance |
| 1 | 2017.01.02 00:00 | buy | 1 | 0.10 | 1.05102 | 0.00000 | 0.00000 |  |
| 2 | 2017.01.02 01:00 | close | 1 | 0.10 | 1.05172 | 0.00000 | 0.00000 | 7.00 | 10007.00 |
| 3 | 2017.01.02 01:00 | sell | 2 | 0.10 | 1.05172 | 0.00000 | 0.00000 |  |
| 4 | 2017.01.02 02:00 | close | 2 | 0.10 | 1.05225 | 0.00000 | 0.00000 | -5.30 | 10001.70 |
| 5 | 2017.01.02 02:00 | buy | 3 | 0.10 | 1.05225 | 0.00000 | 0.00000 |  |
| 6 | 2017.01.02 03:00 | close | 3 | 0.10 | 1.05192 | 0.00000 | 0.00000 | -3.30 | 9998.40 |
| 7 | 2017.01.02 03:00 | sell | 4 | 0.10 | 1.05192 | 0.00000 | 0.00000 |  |
| 8 | 2017.01.02 04:00 | close | 4 | 0.10 | 1.05191 | 0.00000 | 0.00000 | 0.10 | 9998.50 |
| 9 | 2017.01.02 04:00 | buy | 5 | 0.10 | 1.05191 | 0.00000 | 0.00000 |  |
| 10 | 2017.01.02 05:00 | close | 5 | 0.10 | 1.05151 | 0.00000 | 0.00000 | -4.00 | 9994.50 |
| 11 | 2017.01.02 05:00 | sell | 6 | 0.10 | 1.05151 | 0.00000 | 0.00000 |  |
| 12 | 2017.01.02 06:00 | close | 6 | 0.10 | 1.05186 | 0.00000 | 0.00000 | -3.50 | 9991.00 |
| 13 | 2017.01.02 06:00 | buy | 7 | 0.10 | 1.05186 | 0.00000 | 0.00000 |  |
| 14 | 2017.01.02 07:00 | close | 7 | 0.10 | 1.05142 | 0.00000 | 0.00000 | -4.40 | 9986.60 |
| 15 | 2017.01.02 07:00 | sell | 8 | 0.10 | 1.05142 | 0.00000 | 0.00000 |  |
| 16 | 2017.01.02 08:00 | close | 8 | 0.10 | 1.05110 | 0.00000 | 0.00000 | 3.20 | 9989.80 |
| 17 | 2017.01.02 08:00 | buy | 9 | 0.10 | 1.05110 | 0.00000 | 0.00000 |  |
| 18 | 2017.01.02 09:00 | close | 9 | 0.10 | 1.05131 | 0.00000 | 0.00000 | 2.10 | 9991.90 |
| 19 | 2017.01.02 09:00 | sell | 10 | 0.10 | 1.05131 | 0.00000 | 0.00000 |  |
| 20 | 2017.01.02 10:00 | close | 10 | 0.10 | 1.05155 | 0.00000 | 0.00000 | -2.40 | 9989.50 |

MT5 (netting):

|     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Open Time** | **Order** | **Symbol** | **Type** | **Volume** | **Price** | **S / L** | **T / P** | **Time** | **State** | **Comment** |
| 2017.01.02 00:00:00 | 2 | EURUSD | buy | 0.10 / 0.10 | 1.05140 |  |  | 2017.01.02 00:00:00 | filled |  |
| 2017.01.02 01:00:00 | 3 | EURUSD | sell | 0.10 / 0.10 | 1.05172 |  |  | 2017.01.02 01:00:00 | filled |  |
| 2017.01.02 01:00:00 | 4 | EURUSD | sell | 0.10 / 0.10 | 1.05172 |  |  | 2017.01.02 01:00:00 | filled |  |
| 2017.01.02 02:00:00 | 5 | EURUSD | buy | 0.10 / 0.10 | 1.05293 |  |  | 2017.01.02 02:00:00 | filled |  |
| 2017.01.02 02:00:00 | 6 | EURUSD | buy | 0.10 / 0.10 | 1.05293 |  |  | 2017.01.02 02:00:00 | filled |  |
| 2017.01.02 03:00:00 | 7 | EURUSD | sell | 0.10 / 0.10 | 1.05192 |  |  | 2017.01.02 03:00:00 | filled |  |
| 2017.01.02 03:00:00 | 8 | EURUSD | sell | 0.10 / 0.10 | 1.05192 |  |  | 2017.01.02 03:00:00 | filled |  |
| 2017.01.02 04:00:00 | 9 | EURUSD | buy | 0.10 / 0.10 | 1.05234 |  |  | 2017.01.02 04:00:00 | filled |  |
| 2017.01.02 04:00:00 | 10 | EURUSD | buy | 0.10 / 0.10 | 1.05234 |  |  | 2017.01.02 04:00:00 | filled |  |
| 2017.01.02 05:00:00 | 11 | EURUSD | sell | 0.10 / 0.10 | 1.05151 |  |  | 2017.01.02 05:00:00 | filled |  |
| 2017.01.02 05:00:00 | 12 | EURUSD | sell | 0.10 / 0.10 | 1.05151 |  |  | 2017.01.02 05:00:00 | filled |  |
| 2017.01.02 06:00:00 | 13 | EURUSD | buy | 0.10 / 0.10 | 1.05230 |  |  | 2017.01.02 06:00:00 | filled |  |
| 2017.01.02 06:00:00 | 14 | EURUSD | buy | 0.10 / 0.10 | 1.05230 |  |  | 2017.01.02 06:00:00 | filled |  |
| 2017.01.02 07:00:00 | 15 | EURUSD | sell | 0.10 / 0.10 | 1.05142 |  |  | 2017.01.02 07:00:00 | filled |  |
| 2017.01.02 07:00:00 | 16 | EURUSD | sell | 0.10 / 0.10 | 1.05142 |  |  | 2017.01.02 07:00:00 | filled |  |
| 2017.01.02 08:00:00 | 17 | EURUSD | buy | 0.10 / 0.10 | 1.05169 |  |  | 2017.01.02 08:00:00 | filled |  |
| 2017.01.02 08:00:00 | 18 | EURUSD | buy | 0.10 / 0.10 | 1.05169 |  |  | 2017.01.02 08:00:00 | filled |  |
| 2017.01.02 09:00:00 | 19 | EURUSD | sell | 0.10 / 0.10 | 1.05131 |  |  | 2017.01.02 09:00:00 | filled |  |
| 2017.01.02 09:00:00 | 20 | EURUSD | sell | 0.10 / 0.10 | 1.05131 |  |  | 2017.01.02 09:00:00 | filled |  |
| 2017.01.02 10:00:00 | 21 | EURUSD | buy | 0.10 / 0.10 | 1.05164 |  |  | 2017.01.02 10:00:00 | filled |  |

MT5 (hedging):

|     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Open Time** | **Order** | **Symbol** | **Type** | **Volume** | **Price** | **S / L** | **T / P** | **Time** | **State** | **Comment** |
| 2017.01.02 00:00:00 | 2 | EURUSD | buy | 0.10 / 0.10 | 1.05140 |  |  | 2017.01.02 00:00:00 | filled |  |
| 2017.01.02 01:00:00 | 3 | EURUSD | sell | 0.10 / 0.10 | 1.05172 |  |  | 2017.01.02 01:00:00 | filled |  |
| 2017.01.02 01:00:00 | 4 | EURUSD | sell | 0.10 / 0.10 | 1.05172 |  |  | 2017.01.02 01:00:00 | filled |  |
| 2017.01.02 02:00:00 | 5 | EURUSD | buy | 0.10 / 0.10 | 1.05293 |  |  | 2017.01.02 02:00:00 | filled |  |
| 2017.01.02 02:00:00 | 6 | EURUSD | buy | 0.10 / 0.10 | 1.05293 |  |  | 2017.01.02 02:00:00 | filled |  |
| 2017.01.02 03:00:00 | 7 | EURUSD | sell | 0.10 / 0.10 | 1.05192 |  |  | 2017.01.02 03:00:00 | filled |  |
| 2017.01.02 03:00:00 | 8 | EURUSD | sell | 0.10 / 0.10 | 1.05192 |  |  | 2017.01.02 03:00:00 | filled |  |
| 2017.01.02 04:00:00 | 9 | EURUSD | buy | 0.10 / 0.10 | 1.05234 |  |  | 2017.01.02 04:00:00 | filled |  |
| 2017.01.02 04:00:00 | 10 | EURUSD | buy | 0.10 / 0.10 | 1.05234 |  |  | 2017.01.02 04:00:00 | filled |  |
| 2017.01.02 05:00:00 | 11 | EURUSD | sell | 0.10 / 0.10 | 1.05151 |  |  | 2017.01.02 05:00:00 | filled |  |
| 2017.01.02 05:00:00 | 12 | EURUSD | sell | 0.10 / 0.10 | 1.05151 |  |  | 2017.01.02 05:00:00 | filled |  |
| 2017.01.02 06:00:00 | 13 | EURUSD | buy | 0.10 / 0.10 | 1.05230 |  |  | 2017.01.02 06:00:00 | filled |  |
| 2017.01.02 06:00:00 | 14 | EURUSD | buy | 0.10 / 0.10 | 1.05230 |  |  | 2017.01.02 06:00:00 | filled |  |
| 2017.01.02 07:00:00 | 15 | EURUSD | sell | 0.10 / 0.10 | 1.05142 |  |  | 2017.01.02 07:00:00 | filled |  |
| 2017.01.02 07:00:00 | 16 | EURUSD | sell | 0.10 / 0.10 | 1.05142 |  |  | 2017.01.02 07:00:00 | filled |  |
| 2017.01.02 08:00:00 | 17 | EURUSD | buy | 0.10 / 0.10 | 1.05169 |  |  | 2017.01.02 08:00:00 | filled |  |
| 2017.01.02 08:00:00 | 18 | EURUSD | buy | 0.10 / 0.10 | 1.05169 |  |  | 2017.01.02 08:00:00 | filled |  |
| 2017.01.02 09:00:00 | 19 | EURUSD | sell | 0.10 / 0.10 | 1.05131 |  |  | 2017.01.02 09:00:00 | filled |  |
| 2017.01.02 09:00:00 | 20 | EURUSD | sell | 0.10 / 0.10 | 1.05131 |  |  | 2017.01.02 09:00:00 | filled |  |
| 2017.01.02 10:00:00 | 21 | EURUSD | buy | 0.10 / 0.10 | 1.05164 |  |  | 2017.01.02 10:00:00 | filled |  |

Note that for the hedging and netting modes on MT5, the results are identical. Although the underlying implementation is the same, the difference lies in the fact that in netting mode, a position is neutralized by entering a trade of the same volume, while on hedging mode, the position is closed similar to what many traders are accustomed with in MetaTrader 4. Under the hedging mode, we could see a message such as the following:

```
PE      0       16:19:15.747    Trade   2017.01.02 01:00:00   instant sell 0.10 EURUSD at 1.05172, close #2 (1.05172 / 1.05237 / 1.05172)
GP      0       16:19:15.747    Trades  2017.01.02 01:00:00   deal #3 sell 0.10 EURUSD at 1.05172 done (based on order #3)
DS      0       16:19:15.747    Trade   2017.01.02 01:00:00   deal performed [#3 sell 0.10 EURUSD at 1.05172]
```

Note the statement that says "close #2" right on the first line. Hedging mode indicates which particular trade is to be neutralized (closed). On the other hand, under netting mode, we would only see a message such as the following:

```
PG      0       16:20:51.958    Trade   2017.01.02 01:00:00   instant sell 0.10 EURUSD at 1.05172 (1.05172 / 1.05237 / 1.05172)
MQ      0       16:20:51.958    Trades  2017.01.02 01:00:00   deal #3 sell 0.10 EURUSD at 1.05172 done (based on order #3)
KN      0       16:20:51.958    Trade   2017.01.02 01:00:00   deal performed [#3 sell 0.10 EURUSD at 1.05172]
```

Under this mode, it is less obvious if the deal performed was for the entry of a new order, or simply an act of neutralizing an existing position.

### Overview of Structure

The COrderManager class is one of the most complex class objects to be discussed in this article series. In order to give an idea on how the final order manager would look like with its object members, consider the following diagram:

[![overview of structure of order manager](https://c.mql5.com/2/25/overview__1.png)](https://c.mql5.com/2/25/overview.png)

In simple words, the order manager would contain two instances of COrders (current and historical), which would act as containers for the orders entered by the order manager (COrder). Each of these orders can accommodate stop levels (can be none, but more than one is possible), and each of these levels can have its own trailing methods (can be none, but more than one is also possible). Although most expert advisors may not find use for such a complexity, some strategies, especially those that deal with multiple support and resistance levels, will find this structure useful. These class members will be discussed on future articles.

### Conclusion

In this article, we have discussed the COrderManager class, which is responsible for the management of the trade operations of the EA. The COrderManager class was designed in such a way to be able to deal with MQL4 and MQL5 so traders and programmers coding expert advisors would be able to ensure cross-platform compatibility in the code that they write on the main source or header file.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2961.zip "Download all attachments in the single ZIP archive")

[ordermanager\_sample.zip](https://www.mql5.com/en/articles/download/2961/ordermanager_sample.zip "Download ordermanager_sample.zip")(894.99 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Cross-Platform Expert Advisor: The CExpertAdvisor and CExpertAdvisors Classes](https://www.mql5.com/en/articles/3622)
- [Cross-Platform Expert Advisor: Custom Stops, Breakeven and Trailing](https://www.mql5.com/en/articles/3621)
- [Cross-Platform Expert Advisor: Stops](https://www.mql5.com/en/articles/3620)
- [Cross-Platform Expert Advisor: Time Filters](https://www.mql5.com/en/articles/3395)
- [Cross-Platform Expert Advisor: Money Management](https://www.mql5.com/en/articles/3280)
- [Cross-Platform Expert Advisor: Signals](https://www.mql5.com/en/articles/3261)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/193287)**
(3)


![Alexandr Gavrilin](https://c.mql5.com/avatar/2025/12/694aad80-f58e.png)

**[Alexandr Gavrilin](https://www.mql5.com/en/users/dken)**
\|
3 Jun 2017 at 09:36

super, I've been waiting for a ready solution for a long time.

thanks to the author in advance.

![mbjen](https://c.mql5.com/avatar/avatar_na2.png)

**[mbjen](https://www.mql5.com/en/users/mbjen)**
\|
13 Oct 2017 at 23:18

Hello Enrico,

I haven't found any method for getting order profit. There is no one?

![Forex5xx](https://c.mql5.com/avatar/avatar_na2.png)

**[Forex5xx](https://www.mql5.com/en/users/forex5xx)**
\|
7 Jan 2022 at 19:02

**mbjen [#](https://www.mql5.com/en/forum/193287#comment_5905880):**

Hello Enrico,

I haven't found any method for getting order profit. There is no one?

An Order can't have a profit until it is opened and becomes a Position.


![Trading with Donchian Channels](https://c.mql5.com/2/26/MQL5-avatar-Donchian-002.png)[Trading with Donchian Channels](https://www.mql5.com/en/articles/3146)

In this article, we develop and tests several strategies based on the Donchian channel using various indicator filters. We also perform a comparative analysis of their operation.

![How Long Is the Trend?](https://c.mql5.com/2/27/MQL5-avatar-TrendTime-001.png)[How Long Is the Trend?](https://www.mql5.com/en/articles/3188)

The article highlights several methods for trend identification aiming to determine the trend duration relative to the flat market. In theory, the trend to flat rate is considered to be 30% to 70%. This is what we'll be checking.

![MQL5 Cookbook - Creating a ring buffer for fast calculation of indicators in a sliding window](https://c.mql5.com/2/26/Fon.png)[MQL5 Cookbook - Creating a ring buffer for fast calculation of indicators in a sliding window](https://www.mql5.com/en/articles/3047)

The ring buffer is the simplest and the most efficient way to arrange data when performing calculations in a sliding window. The article describes the algorithm and shows how it simplifies calculations in a sliding window and makes them more efficient.

![Comparative Analysis of 10 Trend Strategies](https://c.mql5.com/2/26/MQL5-avatar-sravn-analiz-001__1.png)[Comparative Analysis of 10 Trend Strategies](https://www.mql5.com/en/articles/3074)

The article provides a brief overview of ten trend following strategies, as well as their testing results and comparative analysis. Based on the obtained results, we draw a general conclusion about the appropriateness, advantages and disadvantages of trend following trading.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/2961&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071727512108608844)

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