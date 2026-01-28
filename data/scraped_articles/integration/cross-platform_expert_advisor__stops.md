---
title: Cross-Platform Expert Advisor: Stops
url: https://www.mql5.com/en/articles/3620
categories: Integration, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:17:20.573960
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/3620&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071721284406029613)

MetaTrader 5 / Examples


### Table of Contents

01. [Introduction](https://www.mql5.com/en/articles/3620#introduction)
02. [COrder](https://www.mql5.com/en/articles/3620#corder)
03. [CStop](https://www.mql5.com/en/articles/3620#cstop)
04. [CStops](https://www.mql5.com/en/articles/3620#cstops)
05. [COrderStop](https://www.mql5.com/en/articles/3620#corderstop)
06. [COrderStops](https://www.mql5.com/en/articles/3620#corderstops)
07. [Chart Representation of Stops](https://www.mql5.com/en/articles/3620#chart)
08. [Order Stop Checking](https://www.mql5.com/en/articles/3620#orderstopchecking)
09. [Examples](https://www.mql5.com/en/articles/3620#examples)
10. [Conclusion](https://www.mql5.com/en/articles/3620#conclusion)
11. [Programs Used in the Article](https://www.mql5.com/en/articles/3620#progra)
12. [Class Files Featured in the Article](https://www.mql5.com/en/articles/3620#class)


### Introduction

As discussed in a previous article, for
a cross-platform expert advsior, we have created an order manager
(COrderManager) which takes care of most of the differences between
MQL4 and MQL5 as far as the entry and exit of trades are concerned.
In both versions, the expert advisor saves the trades information in
memory by creating instances of COrder. The containers of dynamic
pointers to instances of this class object are also available as
class members of COrderManager (for both current and historical
trades).

It is possible to have the order
manager directly handle the stop levels of each trade. However, doing this
would have certain limitations:

- COrderManager would most likely have to
be extended to customize how it handles the stop levels for each
trade.
- The existing methods of COrderManager
only deal with the entry of main trades.


It is possible for COrderManager to be extended so that it can handle the stop levels for each trade on its own. However, this is not only limited to the placing of the actual levels, but also for other tasks related to the monitoring of these stops as well, such as modification and checking if the market has already hit a certain level. These functions would make COrderManager much more complex that it currently is, not mentioning the split-implementations.

The order manager only deals with the entry of main trades, while some stop levels would require the EA to perform other trading operations, such as the placing of pending orders, and closing of actual positions. While it is possible for the order manager to handle it on its own, it would be best to have it focus on the entry of main trades, and let another class object handle the stop levels.

In this article, we will discuss the
implementation where the COrderManager would exclusively handle the
entry and exit of main trades (as it currently is), and then have the
implementation of stop levels handled separately by another class
object (CStop).

### COrder

As we have learned from a previous
article (see [Cross-Platform Expert Advisor: Orders](https://www.mql5.com/en/articles/2590)), an instance of
COrder is created after a successful trade operation (entry). For
both MetaTrader 4 and MetaTrader 5, the information regarding
stoploss and takeprofit can be saved from the broker's end. However,
in the event where the stops are hidden from the broker or multiple
stop levels are involved, the information regarding most of these
stops should be saved locally. Thus, in the latter case, after a
successful entry of a position, the COrder instance should be created
first, followed by the object instances representing its stoploss and
takeprofit levels. In an earlier article, we have demonstrated how the COrder instances are added to the order manager upon creation, as shown also in the following figure:

![TradeOpen](https://c.mql5.com/2/29/TradeOpen.png)

The stop levels are added in the same way. To do this, we just need to insert the method after the new COrder instance is created, and before it is added to the list of ongoing trades (COrders). Thus, we just need to slightly modify the illustration above, as shown in the following illustration:

![TradeOpen - with order stops](https://c.mql5.com/2/29/tradeopen_-_mod.png)

The general operation of the creation of stop levels is shown in the following figure:

![General operation of creation of stop levels](https://c.mql5.com/2/29/CreateStops.png)

As shown in the previous two flowcharts above, as
soon as a trade has been successfully entered, a new COrder instance
will be created representing it. After that, instances of COrderStop
will be created for each of the stoploss and takeprofit levels
defined. If there were no CStop instance declared on the expert advisor's initialization, this particular process would be skipped. On the other hand, if instances of COrderStop were created, the pointers to these instances will then be stored under
the COrder instance created earlier. We can find this operation
within the Init method of COrder:

```
bool COrderBase::Init(COrders *orders,CStops *stops)
  {
   if(CheckPointer(orders))
      SetContainer(GetPointer(orders));
   if(CheckPointer(stops))
      CreateStops(GetPointer(stops));
   m_order_stops.SetContainer(GetPointer(this));
   return true;
  }
```

For the creation of the stop levels for
the instance of COrder, we can see that it calls its CreateStops
method, which is shown below:

```
void COrderBase::CreateStops(CStops *stops)
  {
   if(!CheckPointer(stops))
      return;
   if(stops.Total()>0)
     {
      for(int i=0;i<stops.Total();i++)
        {
         CStop *stop=stops.At(i);
         if(CheckPointer(stop)==POINTER_INVALID)
            continue;
         m_order_stops.NewOrderStop(GetPointer(this),stop);
        }
     }
  }
```

The method iterates over all the
available instances of CStop, which represents each pair of stoploss
and takeprofit levels. Each of these CStop instances are passed to an
instance of COrderStops, which is simply a container for all stop
levels for a given order. We can then construct the hirerarchy of the
class objects as follows:

![Hierarchy of stop levels](https://c.mql5.com/2/29/overview.png)

For each instance of COrder, there is
one member of type COrderStops. This COrderStops instance is a
container (an extension of CArrayObj), which contains the pointers to
instances of COrderStop. Each COrderStop instance represents a stop
level (CStop) for that particular trade only (COrder instance).

### CStop

As discussed earlier, we would like to
have a certain level of freedom of customizing how the stop levels of
each trade is handled, without having to modify the source code of
the order manager. This is largely accomplished by the class CStop.
Among the responsibilities of this class are the following:

1. define the stoploss and takeprofit
    levels
2. perform the calculations needed to
    calculate the stop levels
3. implement stoploss and takeprofit
    levels for the main trade
4. check if the stop level has been
    triggered
5. reconcile the differences between MQL4
    and MQL5 in implementing the stoploss and takeprofit levels
6. define how the stops should be handled
    over time (e.g. breakeven, trailing, etc.)

Pointers #1-#5 will be discussed in
this article, while #6 will be discussed in a separate article.

#### Types

Three types of stops will be discussed
in this article:

1. Broker-Based Stop – stops that are
    sent to the broker along with the trade request
2. Pending Order Stop – stops using
    pending orders that would either act as a partial or full hedge
    against the main position (MetaTrader 4, MetaTrader 5 hedging mode),
    or subtract from the main position (MetaTrader 5 netting mode).
3. Virtual Stop (Stealth Stop) – stops
    that are hidden from the broker and are managed locally by the expert
    advisor.

In MetaTrader 4, the broker-based stop
are the stoploss and takeprofit that traders are most familiar with.
These are the prices or stop levels that are sent to the broker along
with the main trade. The MetaTrader 5 version for hedging mode uses
the same mechanic as the MetaTrader 4 version. On the other hand, for
netting mode, a pending order stop will be used for this type of
CStop, since the stoploss and takeprofit is different (applied on the
entire position for the symbol or instrument).

A pending order stop uses pending
orders to somehow mimic a broker-based stop. As soon as the pending
orders were executed, the EA would perform a trade operation
(OrderCloseBy) to close main position using the volume of the pending
order just triggered. This is true for the MetaTrader 4 and
MetaTrader 5, hedging mode, versions. For the MetaTrader 5, netting
mode version, the exit of some or all volume from the main position
is done automatically, since there can only be one position
maintained at any given time per symbol.

#### Main Stop

The main stop is the stop level that
would trigger the exit of an entire position. Normally, this is the
broker-based stop. When either the broker-based stoploss or
takeprofit is triggered, the entire position leaves the market,
whether the EA intends to do further work on it or not. However, if
there are multiple stop levels and there are no broker-based stops,
letting the EA decide which is the main stop is probably a bad idea.
In this case, the coder has to select which CStop instance is the
main stop for the trade. This is useful especially for some functions
and features that depend on the main stop of the position, such as
money management. Since the stoploss of the main stop level leads to
the exit of the entire position, it represents the maximum risk for
entering that trade. And knowing the main stop instance, the EA would
be able to calculate the lotsize accordingly.

It is worth noting that when a stop is
assigned as a main stop, its volume would always be equal to the
initial volume of the main trade. This would work well with
broker-based and virtual stops, but not on stops that are based on
pending orders. There are two associated problems which lead to this
approach.

The first reason is that in MetaTrader
4, the pending order entry price can be modified while the pending
order is not yet triggered. In MetaTrader 5, this is not possible
since a clear footprint of trade operations has to be maintained. A
new pending order has to be issued, and if successful, the old one
should be deleted, and from then, the expert advisor should use the
new pending order as the new stop level.

The second problem is the possibility
of the old pending order to be triggered while the replacement
pending order is being sent to the broker. Since the EA does not have
control on the triggering of pending orders (the broker has), this
can lead to problems such as orphan trades or a stop level closing
more volume from the main trade than it is supposed to.

In order to avoid these problems, the
simpler approach would be to allocate the volume of pending orders at
trade creation, rather than adjusting the volume of the pending
orders dynamically throughout the lifetime of the main trade.
However, this requires that there should be no other pending order
stop if the main stop is of pending order type.

#### Volume

For the main stop level, there is no
need to assign the lotsize to deduct from the main position, for as
soon as it is triggered, the entire position should be closed.
However, for the other types of stop levels, the volume would have to
be considered, as they are usually to be intended for partial closing
of the main position only. The allocation is divided into four
different types:

1. Fixed – fixed lotsizing
2. Percent Remaining – percentage of the
    remaining lotsize of the main trade
3. Percent Total – percentage of the
    total lotsize (initial volume) of the main trade
4. Remaining – the remaining volume of
    the main trade.

Fixed lotsizing is the most simple form
of volume allocation. However, this would not work optimally if the
expert advisor is using some form of dynamic lotsize calculation for
each position i.e. money management. This is ideal only for use when
the lotsize is fixed althroughout the operation of the expert advisor.

Percent Remaining and Remaining are
best used when the main stop is virtual. Remaining would prevent the
EA from creating orphan trades from unclosed volume, or from
executing a close operation for a volume greater than the remaining
volume of the main trade. Percent Remaining, on the other hand, is
used when the expert advisor should close the trade not based on the
initial volume, but rather on what volume is currently left in the
main trade.

The percent total can be used with
broker-based, virtual, and pending-order-based stops. The
calculations using this method are based on the initial main trade
volume.

#### One-Cancels-the-Other (OCO)

CStop is always represented by a pair
of values, with each value representing either the stoploss or the
takeprofit. However, a one-sided level is possible (having takeprofit
and missing stoploss, or vice versa) by assigning zero to either
value. By default, one stop level, when triggered, closes the other.
The only exception to this is when the the CStop instance is a main
stop.

#### Base Class

The base class for CStop (CStopBase) is shown in the following code:

```
class CStopBase : public CObject
  {
protected:
   //--- stop order parameters
   bool              m_active;
   bool              m_main;
   string            m_name;
   bool              m_oco;
   double            m_stoploss;
   string            m_stoploss_name;
   ENUM_STOP_TYPE    m_stop_type;
   double            m_takeprofit;
   string            m_takeprofit_name;
   int               m_delay;
   //--- stop order trade parameters
   ENUM_VOLUME_TYPE  m_volume_type;
   double            m_volume;
   int               m_magic;
   int               m_deviation;
   string            m_comment;
   //--- stop order objects parameters
   bool              m_entry_visible;
   bool              m_stoploss_visible;
   bool              m_takeprofit_visible;
   color             m_entry_color;
   color             m_stoploss_color;
   color             m_takeprofit_color;
   ENUM_LINE_STYLE   m_entry_style;
   ENUM_LINE_STYLE   m_stoploss_style;
   ENUM_LINE_STYLE   m_takeprofit_style;
   //--- objects
   CSymbolManager   *m_symbol_man;
   CSymbolInfo      *m_symbol;
   CAccountInfo     *m_account;
   CTradeManager     m_trade_man;
   CExpertTradeX    *m_trade;
   CTrails          *m_trails;
   CEventAggregator *m_event_man;
   CStops           *m_stops;
public:
                     CStopBase(void);
                    ~CStopBase(void);
   virtual int       Type(void) const {return CLASS_TYPE_STOP;}
   //--- initialization
   virtual bool      Init(CSymbolManager*,CAccountInfo*,CEventAggregator*);
   virtual bool      InitAccount(CAccountInfo*);
   virtual bool      InitEvent(CEventAggregator*);
   virtual bool      InitSymbol(CSymbolManager*);
   virtual bool      InitTrade(void);
   virtual CStops   *GetContainer(void);
   virtual void      SetContainer(CStops*);
   virtual bool      Validate(void) const;
   //--- getters and setters
   bool              Active(void);
   void              Active(const bool);
   bool              Broker(void) const;
   void              Comment(const string);
   string            Comment(void) const;
   void              Delay(int delay);
   int               Delay(void) const;
   void              SetDeviation(const int);
   int               SetDeviation(void) const;
   void              EntryColor(const color clr);
   void              EntryStyle(const ENUM_LINE_STYLE);
   void              EntryVisible(const bool);
   bool              EntryVisible(void) const;
   void              Magic(const int);
   int               Magic(void) const;
   void              Main(const bool);
   bool              Main(void) const;
   void              Name(const string);
   string            Name(void) const;
   void              OCO(const bool oco);
   bool              OCO(void) const;
   bool              Pending(void) const;
   void              StopLoss(const double);
   double            StopLoss(void) const;
   void              StopLossColor(const color);
   bool              StopLossCustom(void);
   void              StopLossName(const string);
   string            StopLossName(void) const;
   void              StopLossVisible(const bool);
   bool              StopLossVisible(void) const;
   void              StopLossStyle(const ENUM_LINE_STYLE);
   void              StopType(const ENUM_STOP_TYPE);
   ENUM_STOP_TYPE    StopType(void) const;
   string            SymbolName(void);
   void              TakeProfit(const double);
   double            TakeProfit(void) const;
   void              TakeProfitColor(const color);
   bool              TakeProfitCustom(void);
   void              TakeProfitName(const string);
   string            TakeProfitName(void) const;
   void              TakeProfitStyle(const ENUM_LINE_STYLE);
   void              TakeProfitVisible(const bool);
   bool              TakeProfitVisible(void) const;
   bool              Virtual(void) const;
   void              Volume(double);
   double            Volume(void) const;
   void              VolumeType(const ENUM_VOLUME_TYPE);
   ENUM_VOLUME_TYPE  VolumeType(void) const;
   //--- stop order checking
   virtual bool      CheckStopLoss(COrder*,COrderStop*);
   virtual bool      CheckTakeProfit(COrder*,COrderStop*);
   virtual bool      CheckStopOrder(ENUM_STOP_MODE,COrder*,COrderStop*)=0;
   virtual bool      DeleteStopOrder(const ulong)=0;
   virtual bool      DeleteMarketStop(const ulong)=0;
   virtual bool      OrderModify(const ulong,const double);
   //--- stop order object creation
   virtual CStopLine *CreateEntryObject(const long,const string,const int,const double);
   virtual CStopLine *CreateStopLossObject(const long,const string,const int,const double);
   virtual CStopLine *CreateTakeProfitObject(const long,const string,const int,const double);
   //--- stop order price calculation
   virtual bool      Refresh(const string);
   virtual double    StopLossCalculate(const string,const ENUM_ORDER_TYPE,const double);
   virtual double    StopLossCustom(const string,const ENUM_ORDER_TYPE,const double);
   virtual double    StopLossPrice(COrder*,COrderStop*);
   virtual double    StopLossTicks(const ENUM_ORDER_TYPE,const double);
   virtual double    TakeProfitCalculate(const string,const ENUM_ORDER_TYPE,const double);
   virtual double    TakeProfitCustom(const string,const ENUM_ORDER_TYPE,const double);
   virtual double    TakeProfitPrice(COrder*,COrderStop*);
   virtual double    TakeProfitTicks(const ENUM_ORDER_TYPE,const double);
   //--- trailing
   virtual bool      Add(CTrails*);
   virtual double    CheckTrailing(const string,const ENUM_ORDER_TYPE,const double,const double,const ENUM_TRAIL_TARGET);
protected:
   //--- object creation
   virtual CStopLine *CreateObject(const long,const string,const int,const double);
   //--- stop order price calculation
   virtual double    LotSizeCalculate(COrder*,COrderStop*);
   //--- stop order entry
   virtual bool      GetClosePrice(const string,const ENUM_ORDER_TYPE,double&);
   //--- stop order exit
   virtual bool      CloseStop(COrder*,COrderStop*,const double)=0;
   //--- deinitialization
   virtual void      Deinit(void);
   virtual void      DeinitSymbol(void);
   virtual void      DeinitTrade(void);
   virtual void      DeinitTrails(void);
  };
```

For stoploss and takeprofit levels based on number of pips or points, at least four class methods need to be remembered:

1. The type of stop (broker-based, pending order, or virtual), using the method StopType

2. The type of volume calculation used, using the method VolumeType

3. The stoploss level in points (whenever needed), using the the method StopLoss

4. The takeprofit level in points (whenever needed), using the method TakeProfit


All of these are mere setters on class members, so there is no need to elaborate on these methods. Most of the rest of the methods are needed only by the class in its internal calculations. Among the most important of these protected methods are the StopLossCalculate and TakeProfitCalculate methods, whose code are shown below:

```
double CStopBase::StopLossCalculate(const string symbol,const ENUM_ORDER_TYPE type,const double price)
  {
   if(!Refresh(symbol))
      return 0;
   if(type==ORDER_TYPE_BUY || type==ORDER_TYPE_BUY_STOP || type==ORDER_TYPE_BUY_LIMIT)
      return price-m_stoploss*m_symbol.Point();
   else if(type==ORDER_TYPE_SELL || type==ORDER_TYPE_SELL_STOP || type==ORDER_TYPE_SELL_LIMIT)
      return price+m_stoploss*m_symbol.Point();
   return 0;
  }

double CStopBase::TakeProfitCalculate(const string symbol,const ENUM_ORDER_TYPE type,const double price)
  {
   if(!Refresh(symbol))
      return 0;
   if(type==ORDER_TYPE_BUY || type==ORDER_TYPE_BUY_STOP || type==ORDER_TYPE_BUY_LIMIT)
      return price+m_takeprofit*m_symbol.Point();
   else if(type==ORDER_TYPE_SELL || type==ORDER_TYPE_SELL_STOP || type==ORDER_TYPE_SELL_LIMIT)
      return price-m_takeprofit*m_symbol.Point();
   return 0;
  }
```

Both methods take three arguments, all of which are related to the main trade. The methods start by first refreshing the symbol through the Refresh method, which simply updates the symbol manager with the correct symbol to use. Once the symbol has been refreshed, it would then return the value of stoploss or takeprofit based on the point value supplied to the class instance during initialization.

### CStops

Class CStops will serve as container
for instances of CStop. An instance of this class will have to be
dynamically added to the order manager as one of its members.The following code shows CStopsBase, which is the base class for CStops:

```
class CStopsBase : public CArrayObj
  {
protected:
   bool              m_active;
   CEventAggregator *m_event_man;
   CObject          *m_container;
public:
                     CStopsBase(void);
                    ~CStopsBase(void);
   virtual int       Type(void) const {return CLASS_TYPE_STOPS;}
   //--- initialization
   virtual bool      Init(CSymbolManager*,CAccountInfo*,CEventAggregator*);
   virtual CObject  *GetContainer(void);
   virtual void      SetContainer(CObject*);
   virtual bool      Validate(void) const;
   //--- setters and getters
   virtual bool      Active(void) const;
   virtual void      Active(const bool);
   virtual CStop    *Main(void);
   //--- recovery
   virtual bool      CreateElement(const int);
  };
```

This is class is very similar to the other containers described so far in this article-series.

### COrderStop

COrderStop represents the
implementation of CStop for a particular trade. For a given position,
a CStop can create a maximum of one COrderStop. However, an arbitrary
number of COrderStop instances can share the same instance of CStop.
Thus, if an expert advisor has 3 different instances of CStop, we
would typically expect each COrder instance to have the same number
of instances of COrderStop. If the expert advisor has made 1000
trades, the number of COrderStop instances created would be 1000 \* 3
= 3000, whereas the number of CStop instances created would still be
3\.

The definition of COrderStopBase, from
which COrderStop is based, is shown in the code below:

```
class COrderStopBase : public CObject
  {
protected:
   bool              m_active;
   //--- stop parameters
   double            m_volume;
   CArrayDouble      m_stoploss;
   CArrayDouble      m_takeprofit;
   ulong             m_stoploss_ticket;
   ulong             m_takeprofit_ticket;
   bool              m_stoploss_closed;
   bool              m_takeprofit_closed;
   bool              m_closed;
   ENUM_STOP_TYPE    m_stop_type;
   string            m_stop_name;
   //--- main order object
   COrder           *m_order;
   //--- stop objects
   CStop            *m_stop;
   CStopLine        *m_objentry;
   CStopLine        *m_objsl;
   CStopLine        *m_objtp;
   COrderStops      *m_order_stops;
public:
                     COrderStopBase(void);
                    ~COrderStopBase(void);
   virtual int       Type(void) const {return CLASS_TYPE_ORDERSTOP;}
   //--- initialization
   virtual void      Init(COrder*,CStop*,COrderStops*);
   virtual COrderStops *GetContainer(void);
   virtual void      SetContainer(COrderStops*);
   virtual void      Show(bool);
   //--- getters and setters
   bool              Active(void) const;
   void              Active(bool active);
   string            EntryName(void) const;
   ulong             MainMagic(void) const;
   ulong             MainTicket(void) const;
   double            MainTicketPrice(void) const;
   ENUM_ORDER_TYPE   MainTicketType(void) const;
   COrder           *Order(void);
   void              Order(COrder*);
   CStop            *Stop(void);
   void              Stop(CStop*);
   bool              StopLoss(const double);
   double            StopLoss(void) const;
   double            StopLoss(const int);
   void              StopLossClosed(const bool);
   bool              StopLossClosed(void);
   double            StopLossLast(void) const;
   string            StopLossName(void) const;
   void              StopLossTicket(const ulong);
   ulong             StopLossTicket(void) const;
   void              StopName(const string);
   string            StopName(void) const;
   bool              TakeProfit(const double);
   double            TakeProfit(void) const;
   double            TakeProfit(const int);
   void              TakeProfitClosed(const bool);
   bool              TakeProfitClosed(void);
   double            TakeProfitLast(void) const;
   string            TakeProfitName(void) const;
   void              TakeProfitTicket(const ulong);
   ulong             TakeProfitTicket(void) const;
   void              Volume(const double);
   double            Volume(void) const;
   //--- checking
   virtual void      Check(double&)=0;
   virtual bool      Close(void);
   virtual bool      CheckTrailing(void);
   virtual bool      DeleteChartObject(const string);
   virtual bool      DeleteEntry(void);
   virtual bool      DeleteStopLines(void);
   virtual bool      DeleteStopLoss(void);
   virtual bool      DeleteTakeProfit(void);
   virtual bool      IsClosed(void);
   virtual bool      Update(void) {return true;}
   virtual void      UpdateVolume(double) {}
   //--- deinitialization
   virtual void      Deinit(void);
   //--- recovery
   virtual bool      Save(const int);
   virtual bool      Load(const int);
   virtual void      Recreate(void);
protected:
   virtual bool      IsStopLossValid(const double) const;
   virtual bool      IsTakeProfitValid(const double) const;
   virtual bool      Modify(const double,const double);
   virtual bool      ModifyStops(const double,const double);
   virtual bool      ModifyStopLoss(const double) {return true;}
   virtual bool      ModifyTakeProfit(const double){return true;}
   virtual bool      UpdateOrderStop(const double,const double){return true;}
   virtual bool      MoveStopLoss(const double);
   virtual bool      MoveTakeProfit(const double);
  };
```

COrderStop is contained by COrderStops (to be discussed in the next section), which is in turn contained by COrder. In the actual development of an EA, there is no further need to declare an instance of COrderStop. This is automatically created by COrder based on a particular instance of CStop.

### COrderStops

```
class COrderStopsBase : public CArrayObj
  {
protected:
   bool              m_active;
   CArrayInt         m_types;
   COrder           *m_order;
public:
                     COrderStopsBase(void);
                    ~COrderStopsBase(void);
   virtual int       Type(void) const {return CLASS_TYPE_ORDERSTOPS;}
   void              Active(bool);
   bool              Active(void) const;
   //--- initialization
   virtual CObject *GetContainer(void);
   virtual void      SetContainer(COrder*);
   virtual bool      NewOrderStop(COrder*,CStop*)=0;
   //--- checking
   virtual void      Check(double &volume);
   virtual bool      CheckNewTicket(COrderStop*);
   virtual bool      Close(void);
   virtual void      UpdateVolume(const double) {}
   //--- hiding and showing of stop lines
   virtual void      Show(const bool);
   //--- recovery
   virtual bool      CreateElement(const int);
   virtual bool      Save(const int);
   virtual bool      Load(const int);
  };
```

Just like CStop this looks like just a typical container. However, it contains some methods which mirror the methods of the objects whose pointers it is intended to store (COrderStop).

### Chart Representation of Stops

CStopLine is a class member of CStop,
whose primary responsibility is the graphical representation of stop
levels on the chart. It has three core functions in an expert
advisor:

1. Show the stop levels at the
    initialization of a COrder instance
2. Update the stop levels in the event
    where one or both of the stop levels were changed
3. Remove the stop lines as soon as the
    main trade has left the market

All of these functions are implemented
through the CStop class.

The definition of CStopLineBase, from
which CStopLine is based, is shown below:

```
class CStopLineBase : public CChartObjectHLine
  {
protected:
   bool              m_active;
   CStop            *m_stop;
public:
                     CStopLineBase(void);
                    ~CStopLineBase(void);
   virtual int       Type(void) const {return CLASS_TYPE_STOPLINE;}
   virtual void      SetContainer(CStop*);
   virtual CStop    *GetContainer(void);
   bool              Active(void) const;
   void              Active(const bool);
   virtual bool      ChartObjectExists(void) const;
   virtual double    GetPrice(const int);
   virtual bool      Move(const double);
   virtual bool      SetStyle(const ENUM_LINE_STYLE);
   virtual bool      SetColor(const color);
  };
```

In older versions of MetaTrader 4,
stoploss and takeprofit levels are not modifiable by dragging. These
levels had to be modified either through the order window or through
the use of expert advsiors and scripts. However, showing graphical
representations of these levels may still be useful, especially when
dealing with virtual stops.

### Order Stop Checking

After the stop levels were successfully created for a given position, the next step would be to check if the market has hit a given stop level. For broker-based stops, the process is not necessary, since the closing operation is done on the server side. However, for virtual stops and stops based on pending orders, in most cases, it would be up to the expert advisor itself to perform the closing operation.

For virtual stops, the expert advisor is entirely responsible for monitoring the movement of the market and for checking whether or not the market has hit a given stop level. The following figure shows the general operation. As soon as the market hits a stop level, the expert advisor performs the appropriate closing operation.

![OrderStop Checking (virtual)](https://c.mql5.com/2/29/OrderStopCheckVirtual.png)

For stops based on pending orders, the process is slightly more complex. A stop level being triggered would lead to a pending order being triggered i.e. becoming a position. This is done automatically on the broker side. Thus, it would be the responsibility of the expert advisor to detect if the given pending order is still pending or has already entered the market. If the pending order has already been triggered, then the expert advisor would be responsible for closing the main position with the volume of the pending order that just triggered - a trade close by operation.

[![COrderStop checking (pending order stop)](https://c.mql5.com/2/29/OrderStopCheckPending__1.png)](https://c.mql5.com/2/29/OrderStopCheckPending.png)

In all cases, as soon as a given order stop level is triggered, the stop level is marked as closed. This is to prevent a given stop level from being executed more than once.

### Examples

#### Example \#1: An EA using Heiken Ashi and  Moving Average, with a Single Stop Level (Main Stop)

In most expert advisor, one stoploss and one takeprofit is often enough. We will now extend the example in the previous article (see [Cross-Platform Expert Advisor: Time Filters](https://www.mql5.com/en/articles/3395)) to add a stoploss and takeprofit to the source code. To do this, first, we create a new instance of the container (CStops). Then create an instance of CStop, and then add its pointer to the container. The pointer to the container will then be eventually added to the order manager. The code is shown below:

```
int OnInit()
  {
//--- other code

   CStops *stops=new CStops();

   CStop *main=new CStop("main");
   main.StopType(stop_type_main);
   main.VolumeType(VOLUME_TYPE_PERCENT_TOTAL);
   main.Main(true);
   main.StopLoss(stop_loss);
   main.TakeProfit(take_profit);
   stops.Add(GetPointer(main));

   order_manager.AddStops(GetPointer(stops));
//--- other code
  }
```

Testing the broker-based stop on MetaTrader 4 shows the results on the following table. This is what traders would usually expect for trades executed by an expert advisor on the platform.

|     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| # | Time | Type | Order | Size | Price | S / L | T / P | Profit | Balance |
| 1 | 2017.01.03 10:00 | sell | 1 | 0.30 | 1.04597 | 1.05097 | 1.04097 |  |
| 2 | 2017.01.03 11:34 | t/p | 1 | 0.30 | 1.04097 | 1.05097 | 1.04097 | 150.00 | 3150.00 |
| 3 | 2017.01.05 11:00 | sell | 2 | 0.30 | 1.05149 | 1.05649 | 1.04649 |  |
| 4 | 2017.01.05 17:28 | s/l | 2 | 0.30 | 1.05649 | 1.05649 | 1.04649 | -150.00 | 3000.00 |

For stops based on pending orders, the stops are placed in advance just like in standard sl/tp, but in the form of pending orders. As soon as the pending order is triggered, the expert advisor will then perform a closeby operation, closing the main trade by the volume of the triggered pending order. However, unlike standard sl/tp, the expert advisor performs this on the client-side. This will not be executed unless the trading platform is live and the EA is running on a chart.

|     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| # | Time | Type | Order | Size | Price | S / L | T / P | Profit | Balance |
| 1 | 2017.01.03 10:00 | sell | 1 | 0.30 | 1.04597 | 0.00000 | 0.00000 |  |
| 2 | 2017.01.03 10:00 | buy stop | 2 | 0.30 | 1.05097 | 0.00000 | 0.00000 |  |
| 3 | 2017.01.03 10:00 | buy limit | 3 | 0.30 | 1.04097 | 0.00000 | 0.00000 |  |
| 4 | 2017.01.03 11:34 | buy | 3 | 0.30 | 1.04097 | 0.00000 | 0.00000 |  |
| 5 | 2017.01.03 11:34 | close by | 1 | 0.30 | 1.04097 | 0.00000 | 0.00000 | 150.00 | 3150.00 |
| 6 | 2017.01.03 11:34 | close by | 3 | 0.00 | 1.04097 | 0.00000 | 0.00000 | 0.00 | 3150.00 |
| 7 | 2017.01.03 11:34 | delete | 2 | 0.30 | 1.05097 | 0.00000 | 0.00000 |  |
| 8 | 2017.01.05 11:00 | sell | 4 | 0.30 | 1.05149 | 0.00000 | 0.00000 |  |
| 9 | 2017.01.05 11:00 | buy stop | 5 | 0.30 | 1.05649 | 0.00000 | 0.00000 |  |
| 10 | 2017.01.05 11:00 | buy limit | 6 | 0.30 | 1.04649 | 0.00000 | 0.00000 |  |
| 11 | 2017.01.05 17:28 | buy | 5 | 0.30 | 1.05649 | 0.00000 | 0.00000 |  |
| 12 | 2017.01.05 17:28 | close by | 4 | 0.30 | 1.05649 | 0.00000 | 0.00000 | -150.00 | 3000.00 |
| 13 | 2017.01.05 17:28 | close by | 5 | 0.00 | 1.05649 | 0.00000 | 0.00000 | 0.00 | 3000.00 |
| 14 | 2017.01.05 17:28 | delete | 6 | 0.30 | 1.04649 | 0.00000 | 0.00000 |  |

Virtual stops do not send anything about the stops on the main trade. The broker is only informed as soon as the EA executes a closing order. The following table shows the behavior of the EA using virtual stops. Here, as soon as the target price for a stop level is triggered, the EA sends a trade request to the server in order to close the main trade.

|     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| # | Time | Type | Order | Size | Price | S / L | T / P | Profit | Balance |
| 1 | 2017.01.03 10:00 | sell | 1 | 0.30 | 1.04597 | 0.00000 | 0.00000 |  |
| 2 | 2017.01.03 11:34 | close | 1 | 0.30 | 1.04097 | 0.00000 | 0.00000 | 150.00 | 3150.00 |
| 3 | 2017.01.05 11:00 | sell | 2 | 0.30 | 1.05149 | 0.00000 | 0.00000 |  |
| 4 | 2017.01.05 17:28 | close | 2 | 0.30 | 1.05649 | 0.00000 | 0.00000 | -150.00 | 3000.00 |

Now, let us move on to MetaTrader 5. The hedging mode in MetaTrader 5 is much closer to MetaTrader 4 in terms of end results. The following table shows the results of testing with broker-based stops. Here, we can see clearly the stoploss and takeprofit levels of main trades in their respective columns.

|  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Orders** |
| **Open Time** | **Order** | **Symbol** | **Type** | **Volume** | **Price** | **S / L** | **T / P** | **Time** | **State** | **Comment** |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | 0.30 / 0.30 | 1.04597 | 1.05097 | 1.04097 | 2017.01.03 10:00:00 | filled |  |
| 2017.01.03 11:34:38 | 3 | EURUSD | buy | 0.30 / 0.30 | 1.04097 |  |  | 2017.01.03 11:34:38 | filled | tp 1.04097 |
| 2017.01.05 11:00:00 | 4 | EURUSD | sell | 0.30 / 0.30 | 1.05149 | 1.05649 | 1.04649 | 2017.01.05 11:00:00 | filled |  |
| 2017.01.05 17:28:37 | 5 | EURUSD | buy | 0.30 / 0.30 | 1.05649 |  |  | 2017.01.05 17:28:37 | filled | sl 1.05649 |
|  |
| **Deals** |
| **Time** | **Deal** | **Symbol** | **Type** | **Direction** | **Volume** | **Price** | **Order** | **Commission** | **Swap** | **Profit** | **Balance** | **Comment** |
| 2017.01.01 00:00:00 | 1 |  | balance |  |  |  |  | 0.00 | 0.00 | 3 000.00 | 3 000.00 |  |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | in | 0.30 | 1.04597 | 2 | 0.00 | 0.00 | 0.00 | 3 000.00 |  |
| 2017.01.03 11:34:38 | 3 | EURUSD | buy | out | 0.30 | 1.04097 | 3 | 0.00 | 0.00 | 150.00 | 3 150.00 | tp 1.04097 |
| 2017.01.05 11:00:00 | 4 | EURUSD | sell | in | 0.30 | 1.05149 | 4 | 0.00 | 0.00 | 0.00 | 3 150.00 |  |
| 2017.01.05 17:28:37 | 5 | EURUSD | buy | out | 0.30 | 1.05649 | 5 | 0.00 | 0.00 | -150.00 | 3 000.00 | sl 1.05649 |
|  | **0.00** | **0.00** | **0.00** | **3 000.00** |  |
|  |

In hedging mode, pending orders work in basically the same way as MetaTrader 4, so the EA will also perform a closeby operation when a pending order has triggered.

|  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Orders** |
| **Open Time** | **Order** | **Symbol** | **Type** | **Volume** | **Price** | **S / L** | **T / P** | **Time** | **State** | **Comment** |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | 0.30 / 0.30 | 1.04597 |  |  | 2017.01.03 10:00:00 | filled |  |
| 2017.01.03 10:00:00 | 3 | EURUSD | buy stop | 0.30 / 0.00 | 1.05097 |  |  | 2017.01.03 11:34:38 | canceled |  |
| 2017.01.03 10:00:00 | 4 | EURUSD | buy limit | 0.30 / 0.30 | 1.04097 |  |  | 2017.01.03 11:34:38 | filled |  |
| 2017.01.03 11:34:38 | 5 | EURUSD | close by | 0.30 / 0.30 | 1.04097 |  |  | 2017.01.03 11:34:38 | filled | close #2 by #4 |
| 2017.01.05 11:00:00 | 6 | EURUSD | sell | 0.30 / 0.30 | 1.05149 |  |  | 2017.01.05 11:00:00 | filled |  |
| 2017.01.05 11:00:00 | 7 | EURUSD | buy stop | 0.30 / 0.30 | 1.05649 |  |  | 2017.01.05 17:28:37 | filled |  |
| 2017.01.05 11:00:00 | 8 | EURUSD | buy limit | 0.30 / 0.00 | 1.04649 |  |  | 2017.01.05 17:28:37 | canceled |  |
| 2017.01.05 17:28:37 | 9 | EURUSD | close by | 0.30 / 0.30 | 1.05649 |  |  | 2017.01.05 17:28:37 | filled | close #6 by #7 |
|  |
| **Deals** |
| **Time** | **Deal** | **Symbol** | **Type** | **Direction** | **Volume** | **Price** | **Order** | **Commission** | **Swap** | **Profit** | **Balance** | **Comment** |
| 2017.01.01 00:00:00 | 1 |  | balance |  |  |  |  | 0.00 | 0.00 | 3 000.00 | 3 000.00 |  |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | in | 0.30 | 1.04597 | 2 | 0.00 | 0.00 | 0.00 | 3 000.00 |  |
| 2017.01.03 11:34:38 | 3 | EURUSD | buy | in | 0.30 | 1.04097 | 4 | 0.00 | 0.00 | 0.00 | 3 000.00 |  |
| 2017.01.03 11:34:38 | 4 | EURUSD | buy | out by | 0.30 | 1.04097 | 5 | 0.00 | 0.00 | 150.00 | 3 150.00 | close #2 by #4 |
| 2017.01.03 11:34:38 | 5 | EURUSD | sell | out by | 0.30 | 1.04597 | 5 | 0.00 | 0.00 | 0.00 | 3 150.00 | close #2 by #4 |
| 2017.01.05 11:00:00 | 6 | EURUSD | sell | in | 0.30 | 1.05149 | 6 | 0.00 | 0.00 | 0.00 | 3 150.00 |  |
| 2017.01.05 17:28:37 | 7 | EURUSD | buy | in | 0.30 | 1.05649 | 7 | 0.00 | 0.00 | 0.00 | 3 150.00 |  |
| 2017.01.05 17:28:37 | 9 | EURUSD | sell | out by | 0.30 | 1.05149 | 9 | 0.00 | 0.00 | 0.00 | 3 150.00 | close #6 by #7 |
| 2017.01.05 17:28:37 | 8 | EURUSD | buy | out by | 0.30 | 1.05649 | 9 | 0.00 | 0.00 | -150.00 | 3 000.00 | close #6 by #7 |
|  | **0.00** | **0.00** | **0.00** | **3 000.00** |  |
|  |

Using virtual stops, we do not typically see a "close" tag the way we see in MetaTrader 4. Rather, the closing operation uses the type opposite of the main trade, but we can see on the deals history whether or not it is buying/selling in or out.

|  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Orders** |
| **Open Time** | **Order** | **Symbol** | **Type** | **Volume** | **Price** | **S / L** | **T / P** | **Time** | **State** | **Comment** |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | 0.30 / 0.30 | 1.04597 |  |  | 2017.01.03 10:00:00 | filled |  |
| 2017.01.03 11:34:38 | 3 | EURUSD | buy | 0.30 / 0.30 | 1.04097 |  |  | 2017.01.03 11:34:38 | filled |  |
| 2017.01.05 11:00:00 | 4 | EURUSD | sell | 0.30 / 0.30 | 1.05149 |  |  | 2017.01.05 11:00:00 | filled |  |
| 2017.01.05 17:28:37 | 5 | EURUSD | buy | 0.30 / 0.30 | 1.05649 |  |  | 2017.01.05 17:28:37 | filled |  |
|  |
| **Deals** |
| **Time** | **Deal** | **Symbol** | **Type** | **Direction** | **Volume** | **Price** | **Order** | **Commission** | **Swap** | **Profit** | **Balance** | **Comment** |
| 2017.01.01 00:00:00 | 1 |  | balance |  |  |  |  | 0.00 | 0.00 | 3 000.00 | 3 000.00 |  |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | in | 0.30 | 1.04597 | 2 | 0.00 | 0.00 | 0.00 | 3 000.00 |  |
| 2017.01.03 11:34:38 | 3 | EURUSD | buy | out | 0.30 | 1.04097 | 3 | 0.00 | 0.00 | 150.00 | 3 150.00 |  |
| 2017.01.05 11:00:00 | 4 | EURUSD | sell | in | 0.30 | 1.05149 | 4 | 0.00 | 0.00 | 0.00 | 3 150.00 |  |
| 2017.01.05 17:28:37 | 5 | EURUSD | buy | out | 0.30 | 1.05649 | 5 | 0.00 | 0.00 | -150.00 | 3 000.00 |  |
|  | **0.00** | **0.00** | **0.00** | **3 000.00** |  |
|  |

For netting mode, since MetaTrader 5 in this mode uses global stoploss and takeprofit (applies to the entire position), the EA has to use pending orders in order to make the trades have distinct stoploss and takeprofit levels. As discussed earlier, when in hedging mode, broker-based stops will operate as stops based on pending orders, as shown in the following table:

|  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Orders** |
| **Open Time** | **Order** | **Symbol** | **Type** | **Volume** | **Price** | **S / L** | **T / P** | **Time** | **State** | **Comment** |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | 0.30 / 0.30 | 1.04597 |  |  | 2017.01.03 10:00:00 | filled |  |
| 2017.01.03 10:00:00 | 3 | EURUSD | buy stop | 0.30 / 0.00 | 1.05097 |  |  | 2017.01.03 11:34:38 | canceled |  |
| 2017.01.03 10:00:00 | 4 | EURUSD | buy limit | 0.30 / 0.30 | 1.04097 |  |  | 2017.01.03 11:34:38 | filled |  |
| 2017.01.05 11:00:00 | 5 | EURUSD | sell | 0.30 / 0.30 | 1.05149 |  |  | 2017.01.05 11:00:00 | filled |  |
| 2017.01.05 11:00:00 | 6 | EURUSD | buy stop | 0.30 / 0.30 | 1.05649 |  |  | 2017.01.05 17:28:37 | filled |  |
| 2017.01.05 11:00:00 | 7 | EURUSD | buy limit | 0.30 / 0.00 | 1.04649 |  |  | 2017.01.05 17:28:37 | canceled |  |
|  |
| **Deals** |
| **Time** | **Deal** | **Symbol** | **Type** | **Direction** | **Volume** | **Price** | **Order** | **Commission** | **Swap** | **Profit** | **Balance** | **Comment** |
| 2017.01.01 00:00:00 | 1 |  | balance |  |  |  |  | 0.00 | 0.00 | 3 000.00 | 3 000.00 |  |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | in | 0.30 | 1.04597 | 2 | 0.00 | 0.00 | 0.00 | 3 000.00 |  |
| 2017.01.03 11:34:38 | 3 | EURUSD | buy | out | 0.30 | 1.04097 | 4 | 0.00 | 0.00 | 150.00 | 3 150.00 |  |
| 2017.01.05 11:00:00 | 4 | EURUSD | sell | in | 0.30 | 1.05149 | 5 | 0.00 | 0.00 | 0.00 | 3 150.00 |  |
| 2017.01.05 17:28:37 | 5 | EURUSD | buy | out | 0.30 | 1.05649 | 6 | 0.00 | 0.00 | -150.00 | 3 000.00 |  |
|  | **0.00** | **0.00** | **0.00** | **3 000.00** |  |
|  |

The following table shows the result when stops are based on pending orders. This would be the same as the previous table, for the reason mentioned earlier.

|  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Orders** |
| **Open Time** | **Order** | **Symbol** | **Type** | **Volume** | **Price** | **S / L** | **T / P** | **Time** | **State** | **Comment** |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | 0.30 / 0.30 | 1.04597 |  |  | 2017.01.03 10:00:00 | filled |  |
| 2017.01.03 10:00:00 | 3 | EURUSD | buy stop | 0.30 / 0.00 | 1.05097 |  |  | 2017.01.03 11:34:38 | canceled |  |
| 2017.01.03 10:00:00 | 4 | EURUSD | buy limit | 0.30 / 0.30 | 1.04097 |  |  | 2017.01.03 11:34:38 | filled |  |
| 2017.01.05 11:00:00 | 5 | EURUSD | sell | 0.30 / 0.30 | 1.05149 |  |  | 2017.01.05 11:00:00 | filled |  |
| 2017.01.05 11:00:00 | 6 | EURUSD | buy stop | 0.30 / 0.30 | 1.05649 |  |  | 2017.01.05 17:28:37 | filled |  |
| 2017.01.05 11:00:00 | 7 | EURUSD | buy limit | 0.30 / 0.00 | 1.04649 |  |  | 2017.01.05 17:28:37 | canceled |  |
|  |
| **Deals** |
| **Time** | **Deal** | **Symbol** | **Type** | **Direction** | **Volume** | **Price** | **Order** | **Commission** | **Swap** | **Profit** | **Balance** | **Comment** |
| 2017.01.01 00:00:00 | 1 |  | balance |  |  |  |  | 0.00 | 0.00 | 3 000.00 | 3 000.00 |  |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | in | 0.30 | 1.04597 | 2 | 0.00 | 0.00 | 0.00 | 3 000.00 |  |
| 2017.01.03 11:34:38 | 3 | EURUSD | buy | out | 0.30 | 1.04097 | 4 | 0.00 | 0.00 | 150.00 | 3 150.00 |  |
| 2017.01.05 11:00:00 | 4 | EURUSD | sell | in | 0.30 | 1.05149 | 5 | 0.00 | 0.00 | 0.00 | 3 150.00 |  |
| 2017.01.05 17:28:37 | 5 | EURUSD | buy | out | 0.30 | 1.05649 | 6 | 0.00 | 0.00 | -150.00 | 3 000.00 |  |
|  | **0.00** | **0.00** | **0.00** | **3 000.00** |  |
|  |

The following table shows the use of virtual stops in MetaTrader 5, netting mode. The virtual stop levels in netting mode may look the same as in hedging mode, but the internal workings are different.

|  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Orders** |
| **Open Time** | **Order** | **Symbol** | **Type** | **Volume** | **Price** | **S / L** | **T / P** | **Time** | **State** | **Comment** |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | 0.30 / 0.30 | 1.04597 |  |  | 2017.01.03 10:00:00 | filled |  |
| 2017.01.03 11:34:38 | 3 | EURUSD | buy | 0.30 / 0.30 | 1.04097 |  |  | 2017.01.03 11:34:38 | filled |  |
| 2017.01.05 11:00:00 | 4 | EURUSD | sell | 0.30 / 0.30 | 1.05149 |  |  | 2017.01.05 11:00:00 | filled |  |
| 2017.01.05 17:28:37 | 5 | EURUSD | buy | 0.30 / 0.30 | 1.05649 |  |  | 2017.01.05 17:28:37 | filled |  |
|  |
| **Deals** |
| **Time** | **Deal** | **Symbol** | **Type** | **Direction** | **Volume** | **Price** | **Order** | **Commission** | **Swap** | **Profit** | **Balance** | **Comment** |
| 2017.01.01 00:00:00 | 1 |  | balance |  |  |  |  | 0.00 | 0.00 | 3 000.00 | 3 000.00 |  |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | in | 0.30 | 1.04597 | 2 | 0.00 | 0.00 | 0.00 | 3 000.00 |  |
| 2017.01.03 11:34:38 | 3 | EURUSD | buy | out | 0.30 | 1.04097 | 3 | 0.00 | 0.00 | 150.00 | 3 150.00 |  |
| 2017.01.05 11:00:00 | 4 | EURUSD | sell | in | 0.30 | 1.05149 | 4 | 0.00 | 0.00 | 0.00 | 3 150.00 |  |
| 2017.01.05 17:28:37 | 5 | EURUSD | buy | out | 0.30 | 1.05649 | 5 | 0.00 | 0.00 | -150.00 | 3 000.00 |  |
|  | **0.00** | **0.00** | **0.00** | **3 000.00** |  |
|  |

#### Example \#2: An EA using Heiken Ashi and  Moving Average, with Three Stop Levels

More complex expert advisors often require more than one stoploss and takeprofit. We use the same method as the previous example in adding additional stop levels, namely the stop levels named "stop1" and "stop2":

```
int OnInit()
  {
//--- other code

   CStops *stops=new CStops();

   CStop *main=new CStop("main");
   main.StopType(stop_type_main);
   main.VolumeType(VOLUME_TYPE_PERCENT_TOTAL);
   main.Main(true);
   main.StopLoss(stop_loss);
   main.TakeProfit(take_profit);
   stops.Add(GetPointer(main));

   CStop *stop1=new CStop("stop1");
   stop1.StopType(stop_type1);
   stop1.VolumeType(VOLUME_TYPE_PERCENT_TOTAL);
   stop1.Volume(0.35);
   stop1.StopLoss(stop_loss1);
   stop1.TakeProfit(take_profit1);
   stops.Add(GetPointer(stop1));

   CStop *stop2=new CStop("stop2");
   stop2.StopType(stop_type2);
   stop2.VolumeType(VOLUME_TYPE_PERCENT_TOTAL);
   stop2.Volume(0.35);
   stop2.StopLoss(stop_loss2);
   stop2.TakeProfit(take_profit2);
   stops.Add(GetPointer(stop2));

   order_manager.AddStops(GetPointer(stops));

//--- other code
  }
```

The following table shows the results
of a test on MetaTrader 4:

|     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| # | Time | Type | Order | Size | Price | S / L | T / P | Profit | Balance |
| 1 | 2017.01.03 10:00 | sell | 1 | 0.30 | 1.04597 | 1.05097 | 1.04097 |  |
| 2 | 2017.01.03 10:00 | buy stop | 2 | 0.11 | 1.04847 | 0.00000 | 0.00000 |  |
| 3 | 2017.01.03 10:00 | buy limit | 3 | 0.11 | 1.04347 | 0.00000 | 0.00000 |  |
| 4 | 2017.01.03 10:21 | buy | 3 | 0.11 | 1.04347 | 0.00000 | 0.00000 |  |
| 5 | 2017.01.03 10:21 | close by | 1 | 0.11 | 1.04347 | 1.05097 | 1.04097 | 27.50 | 3027.50 |
| 6 | 2017.01.03 10:21 | sell | 4 | 0.19 | 1.04597 | 1.05097 | 1.04097 |  |
| 7 | 2017.01.03 10:21 | close by | 3 | 0.00 | 1.04347 | 0.00000 | 0.00000 | 0.00 | 3027.50 |
| 8 | 2017.01.03 10:21 | delete | 2 | 0.11 | 1.04847 | 0.00000 | 0.00000 |  |
| 9 | 2017.01.03 10:34 | close | 4 | 0.11 | 1.04247 | 1.05097 | 1.04097 | 38.50 | 3066.00 |
| 10 | 2017.01.03 10:34 | sell | 5 | 0.08 | 1.04597 | 1.05097 | 1.04097 |  |
| 11 | 2017.01.03 11:34 | t/p | 5 | 0.08 | 1.04097 | 1.05097 | 1.04097 | 40.00 | 3106.00 |
| 12 | 2017.01.05 11:00 | sell | 6 | 0.30 | 1.05149 | 1.05649 | 1.04649 |  |
| 13 | 2017.01.05 11:00 | buy stop | 7 | 0.11 | 1.05399 | 0.00000 | 0.00000 |  |
| 14 | 2017.01.05 11:00 | buy limit | 8 | 0.11 | 1.04899 | 0.00000 | 0.00000 |  |
| 15 | 2017.01.05 12:58 | buy | 8 | 0.11 | 1.04899 | 0.00000 | 0.00000 |  |
| 16 | 2017.01.05 12:58 | close by | 6 | 0.11 | 1.04899 | 1.05649 | 1.04649 | 27.50 | 3133.50 |
| 17 | 2017.01.05 12:58 | sell | 9 | 0.19 | 1.05149 | 1.05649 | 1.04649 |  |
| 18 | 2017.01.05 12:58 | close by | 8 | 0.00 | 1.04899 | 0.00000 | 0.00000 | 0.00 | 3133.50 |
| 19 | 2017.01.05 12:58 | delete | 7 | 0.11 | 1.05399 | 0.00000 | 0.00000 |  |
| 20 | 2017.01.05 16:00 | close | 9 | 0.19 | 1.05314 | 1.05649 | 1.04649 | -31.35 | 3102.15 |
| 21 | 2017.01.05 16:00 | buy | 10 | 0.30 | 1.05314 | 1.04814 | 1.05814 |  |
| 22 | 2017.01.05 16:00 | sell stop | 11 | 0.11 | 1.05064 | 0.00000 | 0.00000 |  |
| 23 | 2017.01.05 16:00 | sell limit | 12 | 0.11 | 1.05564 | 0.00000 | 0.00000 |  |
| 24 | 2017.01.05 17:09 | sell | 12 | 0.11 | 1.05564 | 0.00000 | 0.00000 |  |
| 25 | 2017.01.05 17:09 | close by | 10 | 0.11 | 1.05564 | 1.04814 | 1.05814 | 27.50 | 3129.65 |
| 26 | 2017.01.05 17:09 | buy | 13 | 0.19 | 1.05314 | 1.04814 | 1.05814 |  |
| 27 | 2017.01.05 17:09 | close by | 12 | 0.00 | 1.05564 | 0.00000 | 0.00000 | 0.00 | 3129.65 |
| 28 | 2017.01.05 17:09 | delete | 11 | 0.11 | 1.05064 | 0.00000 | 0.00000 |  |
| 29 | 2017.01.05 17:28 | close | 13 | 0.11 | 1.05664 | 1.04814 | 1.05814 | 38.50 | 3168.15 |
| 30 | 2017.01.05 17:28 | buy | 14 | 0.08 | 1.05314 | 1.04814 | 1.05814 |  |
| 31 | 2017.01.05 17:40 | t/p | 14 | 0.08 | 1.05814 | 1.04814 | 1.05814 | 40.00 | 3208.15 |

As shown on the table above, all the
three stop levels were triggered for the first trade. For the virtual
stop level, the EA performed a partial close on the main trade as
expected. For the stop level using pending orders, as soon as the
pending order was executed, the EA performed a close by operation and
deducted its volume from the main trade. Finally, for the main stop
level, which is a broker-based stop, the main trade has left the
market with its remaining volume of 0.08 lot.

The following table shows the results
of a test on MetaTrader 5, hedging mode:

|  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Orders** |
| **Open Time** | **Order** | **Symbol** | **Type** | **Volume** | **Price** | **S / L** | **T / P** | **Time** | **State** | **Comment** |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | 0.30 / 0.30 | 1.04597 | 1.05097 | 1.04097 | 2017.01.03 10:00:00 | filled |  |
| 2017.01.03 10:00:00 | 3 | EURUSD | buy stop | 0.11 / 0.00 | 1.04847 |  |  | 2017.01.03 10:21:32 | canceled |  |
| 2017.01.03 10:00:00 | 4 | EURUSD | buy limit | 0.11 / 0.11 | 1.04347 |  |  | 2017.01.03 10:21:32 | filled |  |
| 2017.01.03 10:21:32 | 5 | EURUSD | close by | 0.11 / 0.11 | 1.04347 |  |  | 2017.01.03 10:21:32 | filled | close #2 by #4 |
| 2017.01.03 10:33:40 | 6 | EURUSD | buy | 0.11 / 0.11 | 1.04247 |  |  | 2017.01.03 10:33:40 | filled |  |
| 2017.01.03 11:34:38 | 7 | EURUSD | buy | 0.08 / 0.08 | 1.04097 |  |  | 2017.01.03 11:34:38 | filled | tp 1.04097 |
| 2017.01.05 11:00:00 | 8 | EURUSD | sell | 0.30 / 0.30 | 1.05149 | 1.05649 | 1.04649 | 2017.01.05 11:00:00 | filled |  |
| 2017.01.05 11:00:00 | 9 | EURUSD | buy stop | 0.11 / 0.00 | 1.05399 |  |  | 2017.01.05 12:58:27 | canceled |  |
| 2017.01.05 11:00:00 | 10 | EURUSD | buy limit | 0.11 / 0.11 | 1.04899 |  |  | 2017.01.05 12:58:27 | filled |  |
| 2017.01.05 12:58:27 | 11 | EURUSD | close by | 0.11 / 0.11 | 1.04896 |  |  | 2017.01.05 12:58:27 | filled | close #8 by #10 |
| 2017.01.05 16:00:00 | 12 | EURUSD | buy | 0.19 / 0.19 | 1.05307 |  |  | 2017.01.05 16:00:00 | filled |  |
| 2017.01.05 16:00:00 | 13 | EURUSD | buy | 0.30 / 0.30 | 1.05307 | 1.04807 | 1.05807 | 2017.01.05 16:00:00 | filled |  |
| 2017.01.05 16:00:00 | 14 | EURUSD | sell stop | 0.11 / 0.00 | 1.05057 |  |  | 2017.01.05 17:09:40 | canceled |  |
| 2017.01.05 16:00:00 | 15 | EURUSD | sell limit | 0.11 / 0.11 | 1.05557 |  |  | 2017.01.05 17:09:40 | filled |  |
| 2017.01.05 17:09:40 | 16 | EURUSD | close by | 0.11 / 0.11 | 1.05557 |  |  | 2017.01.05 17:09:40 | filled | close #13 by #15 |
| 2017.01.05 17:28:47 | 17 | EURUSD | sell | 0.11 / 0.11 | 1.05660 |  |  | 2017.01.05 17:28:47 | filled |  |
| 2017.01.05 17:29:15 | 18 | EURUSD | sell | 0.08 / 0.08 | 1.05807 |  |  | 2017.01.05 17:29:15 | filled | tp 1.05807 |
|  |
| **Deals** |
| **Time** | **Deal** | **Symbol** | **Type** | **Direction** | **Volume** | **Price** | **Order** | **Commission** | **Swap** | **Profit** | **Balance** | **Comment** |
| 2017.01.01 00:00:00 | 1 |  | balance |  |  |  |  | 0.00 | 0.00 | 3 000.00 | 3 000.00 |  |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | in | 0.30 | 1.04597 | 2 | 0.00 | 0.00 | 0.00 | 3 000.00 |  |
| 2017.01.03 10:21:32 | 3 | EURUSD | buy | in | 0.11 | 1.04347 | 4 | 0.00 | 0.00 | 0.00 | 3 000.00 |  |
| 2017.01.03 10:21:32 | 4 | EURUSD | buy | out by | 0.11 | 1.04347 | 5 | 0.00 | 0.00 | 27.50 | 3 027.50 | close #2 by #4 |
| 2017.01.03 10:21:32 | 5 | EURUSD | sell | out by | 0.11 | 1.04597 | 5 | 0.00 | 0.00 | 0.00 | 3 027.50 | close #2 by #4 |
| 2017.01.03 10:33:40 | 6 | EURUSD | buy | out | 0.11 | 1.04247 | 6 | 0.00 | 0.00 | 38.50 | 3 066.00 |  |
| 2017.01.03 11:34:38 | 7 | EURUSD | buy | out | 0.08 | 1.04097 | 7 | 0.00 | 0.00 | 40.00 | 3 106.00 | tp 1.04097 |
| 2017.01.05 11:00:00 | 8 | EURUSD | sell | in | 0.30 | 1.05149 | 8 | 0.00 | 0.00 | 0.00 | 3 106.00 |  |
| 2017.01.05 12:58:27 | 9 | EURUSD | buy | in | 0.11 | 1.04896 | 10 | 0.00 | 0.00 | 0.00 | 3 106.00 |  |
| 2017.01.05 12:58:27 | 10 | EURUSD | buy | out by | 0.11 | 1.04896 | 11 | 0.00 | 0.00 | 27.83 | 3 133.83 | close #8 by #10 |
| 2017.01.05 12:58:27 | 11 | EURUSD | sell | out by | 0.11 | 1.05149 | 11 | 0.00 | 0.00 | 0.00 | 3 133.83 | close #8 by #10 |
| 2017.01.05 16:00:00 | 12 | EURUSD | buy | out | 0.19 | 1.05307 | 12 | 0.00 | 0.00 | -30.02 | 3 103.81 |  |
| 2017.01.05 16:00:00 | 13 | EURUSD | buy | in | 0.30 | 1.05307 | 13 | 0.00 | 0.00 | 0.00 | 3 103.81 |  |
| 2017.01.05 17:09:40 | 14 | EURUSD | sell | in | 0.11 | 1.05557 | 15 | 0.00 | 0.00 | 0.00 | 3 103.81 |  |
| 2017.01.05 17:09:40 | 16 | EURUSD | buy | out by | 0.11 | 1.05307 | 16 | 0.00 | 0.00 | 0.00 | 3 103.81 | close #13 by #15 |
| 2017.01.05 17:09:40 | 15 | EURUSD | sell | out by | 0.11 | 1.05557 | 16 | 0.00 | 0.00 | 27.50 | 3 131.31 | close #13 by #15 |
| 2017.01.05 17:28:47 | 17 | EURUSD | sell | out | 0.11 | 1.05660 | 17 | 0.00 | 0.00 | 38.83 | 3 170.14 |  |
| 2017.01.05 17:29:15 | 18 | EURUSD | sell | out | 0.08 | 1.05807 | 18 | 0.00 | 0.00 | 40.00 | 3 210.14 | tp 1.05807 |
|  | **0.00** | **0.00** | **210.14** | **3 210.14** |  |
|  |

As shown on the table above, the EA
works in somewhat a similar way in MetaTrader 5, hedging mode. The
mechanics were the same as in MetaTrader 4 for the three types of
stop levels.

The following table shows the results
of a test on MetaTrader 5, netting mode:

|  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Orders** |
| **Open Time** | **Order** | **Symbol** | **Type** | **Volume** | **Price** | **S / L** | **T / P** | **Time** | **State** | **Comment** |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | 0.30 / 0.30 | 1.04597 |  |  | 2017.01.03 10:00:00 | filled |  |
| 2017.01.03 10:00:00 | 3 | EURUSD | buy stop | 0.30 / 0.00 | 1.05097 |  |  | 2017.01.03 11:34:38 | canceled |  |
| 2017.01.03 10:00:00 | 4 | EURUSD | buy limit | 0.30 / 0.30 | 1.04097 |  |  | 2017.01.03 11:34:38 | filled |  |
| 2017.01.03 10:00:00 | 5 | EURUSD | buy stop | 0.11 / 0.00 | 1.04847 |  |  | 2017.01.03 10:21:32 | canceled |  |
| 2017.01.03 10:00:00 | 6 | EURUSD | buy limit | 0.11 / 0.11 | 1.04347 |  |  | 2017.01.03 10:21:32 | filled |  |
| 2017.01.03 10:33:40 | 7 | EURUSD | buy | 0.11 / 0.11 | 1.04247 |  |  | 2017.01.03 10:33:40 | filled |  |
| 2017.01.05 11:00:00 | 8 | EURUSD | sell | 0.30 / 0.30 | 1.05149 |  |  | 2017.01.05 11:00:00 | filled |  |
| 2017.01.05 11:00:00 | 9 | EURUSD | buy stop | 0.30 / 0.00 | 1.05649 |  |  | 2017.01.05 16:00:00 | canceled |  |
| 2017.01.05 11:00:00 | 10 | EURUSD | buy limit | 0.30 / 0.00 | 1.04649 |  |  | 2017.01.05 16:00:00 | canceled |  |
| 2017.01.05 11:00:00 | 11 | EURUSD | buy stop | 0.11 / 0.00 | 1.05399 |  |  | 2017.01.05 12:58:27 | canceled |  |
| 2017.01.05 11:00:00 | 12 | EURUSD | buy limit | 0.11 / 0.11 | 1.04899 |  |  | 2017.01.05 12:58:27 | filled |  |
| 2017.01.05 16:00:00 | 13 | EURUSD | buy | 0.19 / 0.19 | 1.05307 |  |  | 2017.01.05 16:00:00 | filled |  |
| 2017.01.05 16:00:00 | 14 | EURUSD | buy | 0.30 / 0.30 | 1.05307 |  |  | 2017.01.05 16:00:00 | filled |  |
| 2017.01.05 16:00:00 | 15 | EURUSD | sell stop | 0.30 / 0.00 | 1.04807 |  |  | 2017.01.05 17:29:15 | canceled |  |
| 2017.01.05 16:00:00 | 16 | EURUSD | sell limit | 0.30 / 0.30 | 1.05807 |  |  | 2017.01.05 17:29:15 | filled |  |
| 2017.01.05 16:00:00 | 17 | EURUSD | sell stop | 0.11 / 0.00 | 1.05057 |  |  | 2017.01.05 17:09:40 | canceled |  |
| 2017.01.05 16:00:00 | 18 | EURUSD | sell limit | 0.11 / 0.11 | 1.05557 |  |  | 2017.01.05 17:09:40 | filled |  |
| 2017.01.05 17:28:47 | 19 | EURUSD | sell | 0.11 / 0.11 | 1.05660 |  |  | 2017.01.05 17:28:47 | filled |  |
|  |
| **Deals** |
| **Time** | **Deal** | **Symbol** | **Type** | **Direction** | **Volume** | **Price** | **Order** | **Commission** | **Swap** | **Profit** | **Balance** | **Comment** |
| 2017.01.01 00:00:00 | 1 |  | balance |  |  |  |  | 0.00 | 0.00 | 3 000.00 | 3 000.00 |  |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | in | 0.30 | 1.04597 | 2 | 0.00 | 0.00 | 0.00 | 3 000.00 |  |
| 2017.01.03 11:34:38 | 5 | EURUSD | buy | in/out | 0.30 | 1.04097 | 4 | 0.00 | 0.00 | 40.00 | 3 040.00 |  |
| 2017.01.03 10:21:32 | 3 | EURUSD | buy | out | 0.11 | 1.04347 | 6 | 0.00 | 0.00 | 27.50 | 3 067.50 |  |
| 2017.01.03 10:33:40 | 4 | EURUSD | buy | out | 0.11 | 1.04247 | 7 | 0.00 | 0.00 | 38.50 | 3 106.00 |  |
| 2017.01.05 11:00:00 | 6 | EURUSD | sell | in/out | 0.30 | 1.05149 | 8 | 0.00 | -0.61 | 231.44 | 3 336.83 |  |
| 2017.01.05 12:58:27 | 7 | EURUSD | buy | in/out | 0.11 | 1.04896 | 12 | 0.00 | 0.00 | 20.24 | 3 357.07 |  |
| 2017.01.05 16:00:00 | 8 | EURUSD | buy | in | 0.19 | 1.05307 | 13 | 0.00 | 0.00 | 0.00 | 3 357.07 |  |
| 2017.01.05 16:00:00 | 9 | EURUSD | buy | in | 0.30 | 1.05307 | 14 | 0.00 | 0.00 | 0.00 | 3 357.07 |  |
| 2017.01.05 17:29:15 | 12 | EURUSD | sell | out | 0.30 | 1.05807 | 16 | 0.00 | 0.00 | 157.11 | 3 514.18 |  |
| 2017.01.05 17:09:40 | 10 | EURUSD | sell | out | 0.11 | 1.05557 | 18 | 0.00 | 0.00 | 30.11 | 3 544.29 |  |
| 2017.01.05 17:28:47 | 11 | EURUSD | sell | out | 0.11 | 1.05660 | 19 | 0.00 | 0.00 | 41.44 | 3 585.73 |  |
|  | **0.00** | **-0.61** | **586.34** | **3 585.73** |  |
|  |

Here, we encountered a problem. The final stop level closed at 0.30 lot rather than 0.08 lot as observed in the two previous tests.

As
discussed earlier, the standard stoploss and takeprofit in MetaTrader
5, netting mode, is different to those of the previous two. Thus, the
EA would convert the broker-based stop into a pending-order-based
stop instead. However, with this setting, the EA already has two stop
levels based on pending orders, and one of them is a main stop. As
also discussed earlier, this would lead to orphan trades. The final or
main stop level would always have a volume equal to the initial
volume of the main trade. And since there were other stop levels
involved, it can lead to the main stop level to close the main
position for more than its remaining volume.

In order to fix this, one way is to set
the main stop to be virtual instead, rather than being broker-based
or pending-order based. With this new setting, the EA now has 2 stops that are virtual, and one stop that is based on pending orders. The result of a test using this new setting is shown
below.

|  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Orders** |
| **Open Time** | **Order** | **Symbol** | **Type** | **Volume** | **Price** | **S / L** | **T / P** | **Time** | **State** | **Comment** |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | 0.30 / 0.30 | 1.04597 |  |  | 2017.01.03 10:00:00 | filled |  |
| 2017.01.03 10:00:00 | 3 | EURUSD | buy stop | 0.11 / 0.00 | 1.04847 |  |  | 2017.01.03 10:21:32 | canceled |  |
| 2017.01.03 10:00:00 | 4 | EURUSD | buy limit | 0.11 / 0.11 | 1.04347 |  |  | 2017.01.03 10:21:32 | filled |  |
| 2017.01.03 10:33:40 | 5 | EURUSD | buy | 0.11 / 0.11 | 1.04247 |  |  | 2017.01.03 10:33:40 | filled |  |
| 2017.01.03 11:34:38 | 6 | EURUSD | buy | 0.08 / 0.08 | 1.04097 |  |  | 2017.01.03 11:34:38 | filled |  |
| 2017.01.05 11:00:00 | 7 | EURUSD | sell | 0.30 / 0.30 | 1.05149 |  |  | 2017.01.05 11:00:00 | filled |  |
| 2017.01.05 11:00:00 | 8 | EURUSD | buy stop | 0.11 / 0.00 | 1.05399 |  |  | 2017.01.05 12:58:27 | canceled |  |
| 2017.01.05 11:00:00 | 9 | EURUSD | buy limit | 0.11 / 0.11 | 1.04899 |  |  | 2017.01.05 12:58:27 | filled |  |
| 2017.01.05 16:00:00 | 10 | EURUSD | buy | 0.19 / 0.19 | 1.05307 |  |  | 2017.01.05 16:00:00 | filled |  |
| 2017.01.05 16:00:00 | 11 | EURUSD | buy | 0.30 / 0.30 | 1.05307 |  |  | 2017.01.05 16:00:00 | filled |  |
| 2017.01.05 16:00:00 | 12 | EURUSD | sell stop | 0.11 / 0.00 | 1.05057 |  |  | 2017.01.05 17:09:40 | canceled |  |
| 2017.01.05 16:00:00 | 13 | EURUSD | sell limit | 0.11 / 0.11 | 1.05557 |  |  | 2017.01.05 17:09:40 | filled |  |
| 2017.01.05 17:28:47 | 14 | EURUSD | sell | 0.11 / 0.11 | 1.05660 |  |  | 2017.01.05 17:28:47 | filled |  |
| 2017.01.05 17:29:15 | 15 | EURUSD | sell | 0.08 / 0.08 | 1.05807 |  |  | 2017.01.05 17:29:15 | filled |  |
|  |
| **Deals** |
| **Time** | **Deal** | **Symbol** | **Type** | **Direction** | **Volume** | **Price** | **Order** | **Commission** | **Swap** | **Profit** | **Balance** | **Comment** |
| 2017.01.01 00:00:00 | 1 |  | balance |  |  |  |  | 0.00 | 0.00 | 3 000.00 | 3 000.00 |  |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | in | 0.30 | 1.04597 | 2 | 0.00 | 0.00 | 0.00 | 3 000.00 |  |
| 2017.01.03 10:21:32 | 3 | EURUSD | buy | out | 0.11 | 1.04347 | 4 | 0.00 | 0.00 | 27.50 | 3 027.50 |  |
| 2017.01.03 10:33:40 | 4 | EURUSD | buy | out | 0.11 | 1.04247 | 5 | 0.00 | 0.00 | 38.50 | 3 066.00 |  |
| 2017.01.03 11:34:38 | 5 | EURUSD | buy | out | 0.08 | 1.04097 | 6 | 0.00 | 0.00 | 40.00 | 3 106.00 |  |
| 2017.01.05 11:00:00 | 6 | EURUSD | sell | in | 0.30 | 1.05149 | 7 | 0.00 | 0.00 | 0.00 | 3 106.00 |  |
| 2017.01.05 12:58:27 | 7 | EURUSD | buy | out | 0.11 | 1.04896 | 9 | 0.00 | 0.00 | 27.83 | 3 133.83 |  |
| 2017.01.05 16:00:00 | 8 | EURUSD | buy | out | 0.19 | 1.05307 | 10 | 0.00 | 0.00 | -30.02 | 3 103.81 |  |
| 2017.01.05 16:00:00 | 9 | EURUSD | buy | in | 0.30 | 1.05307 | 11 | 0.00 | 0.00 | 0.00 | 3 103.81 |  |
| 2017.01.05 17:09:40 | 10 | EURUSD | sell | out | 0.11 | 1.05557 | 13 | 0.00 | 0.00 | 27.50 | 3 131.31 |  |
| 2017.01.05 17:28:47 | 11 | EURUSD | sell | out | 0.11 | 1.05660 | 14 | 0.00 | 0.00 | 38.83 | 3 170.14 |  |
| 2017.01.05 17:29:15 | 12 | EURUSD | sell | out | 0.08 | 1.05807 | 15 | 0.00 | 0.00 | 40.00 | 3 210.14 |  |
|  | **0.00** | **0.00** | **210.14** | **3 210.14** |  |
|  |

The total volume taken by the stop
levels are now equal to the initial volume of the main trade.

#### Example \#3: Stops with Money Management

In an earlier article (see [Cross-Platform Expert Advisor: Money Management](https://www.mql5.com/en/articles/3280)), it was discussed that some money management methods depend on a certain stop level (takeprofit is possible, but the stoploss is normally used). Money management methods such as fixed fractional, fixed risk, and fixed risk per point/pip would often require a finite stoploss level. Not setting this level suggests a very large or even infinite risk, and thus, the volume calculated would be very large as well. In order to show an expert advisor capable of using the class objects mentioned in this article with money management, we will modify the first example so that the previously commented lines of code would be included in the compilation.

First, we remove the comment tags on the lines of code for the money management methods, as shown in the code below, within the OnInit functon:

```
int OnInit()
  {
//--- other code
   order_manager=new COrderManager();
   money_manager= new CMoneys();
   CMoney *money_fixed=new CMoneyFixedLot(0.05);
   CMoney *money_ff=new CMoneyFixedFractional(5);
   CMoney *money_ratio=new CMoneyFixedRatio(0,0.1,1000);
   CMoney *money_riskperpoint=new CMoneyFixedRiskPerPoint(0.1);
   CMoney *money_risk=new CMoneyFixedRisk(100);

   money_manager.Add(money_fixed);
   money_manager.Add(money_ff);
   money_manager.Add(money_ratio);
   money_manager.Add(money_riskperpoint);
   money_manager.Add(money_risk);
   order_manager.AddMoneys(money_manager);
//--- other code
  }
```

Then, at the top of the source code, we declare a custom enumeration so that we could create an external parameter that would allow us to select the money management method to use, in the order they were presented in the source code:

```
enum ENUM_MM_TYPE
  {
   MM_FIXED=0,
   MM_FIXED_FRACTIONAL,
   MM_FIXED_RATIO,
   MM_FIXED_RISK_PER_POINT,
   MM_FIXED_RISK
  };
```

We then declare a new input parameter for this enumeration:

```
input ENUM_MM_TYPE mm_type=MM_FIXED_FRACTIONAL;
```

Finally, we have to link this parameter to the actual selection of the money management method within the OnTick function:

```
void OnTick()
  {
//---
   manage_trades();
   if(symbol_info.RefreshRates())
     {
      signals.Check();
      if(signals.CheckOpenLong())
        {
         close_last();
         if(time_filters.Evaluate(TimeCurrent()))
           {
            Print("Entering buy trade..");
            money_manager.Selected((int)mm_type); //use mm_type, cast to 'int' type
            order_manager.TradeOpen(Symbol(),ORDER_TYPE_BUY,symbol_info.Ask());
           }
        }
      else if(signals.CheckOpenShort())
        {
         close_last();
         if(time_filters.Evaluate(TimeCurrent()))
           {
            Print("Entering sell trade..");
            money_manager.Selected((int)mm_type); //use mm_type, cast to 'int' type
            order_manager.TradeOpen(Symbol(),ORDER_TYPE_SELL,symbol_info.Bid());
           }
        }
     }
  }
```

Since the money management methods deal only with pure calculation, there is no issue regarding compatibility between MQL4 and MQL5. The previous two examples have already shown how the stop levels work on the two platforms. The following tables show the results of tests on MetaTrader 5, hedging mode.

The following table shows a test result using fixed fractional money management. The coded setting is to risk 5% of the account. With a starting balance of $3000, we expect the value to be around $150 if it hits stoploss for the first trade. Since the stoploss and takeprofit have the same value (500 points), then we expect the trade to earn $150 as well if it hit takeprofit.

|  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Orders** |
| **Open Time** | **Order** | **Symbol** | **Type** | **Volume** | **Price** | **S / L** | **T / P** | **Time** | **State** | **Comment** |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | 0.30 / 0.30 | 1.04597 | 1.05097 | 1.04097 | 2017.01.03 10:00:00 | filled |  |
| 2017.01.03 11:34:38 | 3 | EURUSD | buy | 0.30 / 0.30 | 1.04097 |  |  | 2017.01.03 11:34:38 | filled | tp 1.04097 |
| 2017.01.05 11:00:00 | 4 | EURUSD | sell | 0.32 / 0.32 | 1.05149 | 1.05649 | 1.04649 | 2017.01.05 11:00:00 | filled |  |
| 2017.01.05 16:00:00 | 5 | EURUSD | buy | 0.32 / 0.32 | 1.05307 |  |  | 2017.01.05 16:00:00 | filled |  |
| 2017.01.05 16:00:00 | 6 | EURUSD | buy | 0.31 / 0.31 | 1.05307 | 1.04807 | 1.05807 | 2017.01.05 16:00:00 | filled |  |
| 2017.01.05 17:29:15 | 7 | EURUSD | sell | 0.31 / 0.31 | 1.05807 |  |  | 2017.01.05 17:29:15 | filled | tp 1.05807 |
|  |
| **Deals** |
| **Time** | **Deal** | **Symbol** | **Type** | **Direction** | **Volume** | **Price** | **Order** | **Commission** | **Swap** | **Profit** | **Balance** | **Comment** |
| 2017.01.01 00:00:00 | 1 |  | balance |  |  |  |  | 0.00 | 0.00 | 3 000.00 | 3 000.00 |  |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | in | 0.30 | 1.04597 | 2 | 0.00 | 0.00 | 0.00 | 3 000.00 |  |
| 2017.01.03 11:34:38 | 3 | EURUSD | buy | out | 0.30 | 1.04097 | 3 | 0.00 | 0.00 | 150.00 | 3 150.00 | tp 1.04097 |
| 2017.01.05 11:00:00 | 4 | EURUSD | sell | in | 0.32 | 1.05149 | 4 | 0.00 | 0.00 | 0.00 | 3 150.00 |  |
| 2017.01.05 16:00:00 | 5 | EURUSD | buy | out | 0.32 | 1.05307 | 5 | 0.00 | 0.00 | -50.56 | 3 099.44 |  |
| 2017.01.05 16:00:00 | 6 | EURUSD | buy | in | 0.31 | 1.05307 | 6 | 0.00 | 0.00 | 0.00 | 3 099.44 |  |
| 2017.01.05 17:29:15 | 7 | EURUSD | sell | out | 0.31 | 1.05807 | 7 | 0.00 | 0.00 | 155.00 | 3 254.44 | tp 1.05807 |
|  | **0.00** | **0.00** | **254.44** | **3 254.44** |  |
|  |

Note that the calculated volume value would also depend on the lot precision of the broker, determined by the minimum lot and minimum lot step parameters. We cannot expect the expert advisor to always make the precise calculations. With these constraints, the expert advisor may find it necessary to round off the value at some point. The same is true for the other methods of money management.

For fixed risk per point money management, the coded setting is to risk $0.1 per point of stoploss. With the setting of 500 points for both stoploss and takeprofit, we expect the profit/loss to be worth $50 either way.

|  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Orders** |
| **Open Time** | **Order** | **Symbol** | **Type** | **Volume** | **Price** | **S / L** | **T / P** | **Time** | **State** | **Comment** |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | 0.10 / 0.10 | 1.04597 | 1.05097 | 1.04097 | 2017.01.03 10:00:00 | filled |  |
| 2017.01.03 11:34:38 | 3 | EURUSD | buy | 0.10 / 0.10 | 1.04097 |  |  | 2017.01.03 11:34:38 | filled | tp 1.04097 |
| 2017.01.05 11:00:00 | 4 | EURUSD | sell | 0.10 / 0.10 | 1.05149 | 1.05649 | 1.04649 | 2017.01.05 11:00:00 | filled |  |
| 2017.01.05 16:00:00 | 5 | EURUSD | buy | 0.10 / 0.10 | 1.05307 |  |  | 2017.01.05 16:00:00 | filled |  |
| 2017.01.05 16:00:00 | 6 | EURUSD | buy | 0.10 / 0.10 | 1.05307 | 1.04807 | 1.05807 | 2017.01.05 16:00:00 | filled |  |
| 2017.01.05 17:29:15 | 7 | EURUSD | sell | 0.10 / 0.10 | 1.05807 |  |  | 2017.01.05 17:29:15 | filled | tp 1.05807 |
|  |
| **Deals** |
| **Time** | **Deal** | **Symbol** | **Type** | **Direction** | **Volume** | **Price** | **Order** | **Commission** | **Swap** | **Profit** | **Balance** | **Comment** |
| 2017.01.01 00:00:00 | 1 |  | balance |  |  |  |  | 0.00 | 0.00 | 3 000.00 | 3 000.00 |  |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | in | 0.10 | 1.04597 | 2 | 0.00 | 0.00 | 0.00 | 3 000.00 |  |
| 2017.01.03 11:34:38 | 3 | EURUSD | buy | out | 0.10 | 1.04097 | 3 | 0.00 | 0.00 | 50.00 | 3 050.00 | tp 1.04097 |
| 2017.01.05 11:00:00 | 4 | EURUSD | sell | in | 0.10 | 1.05149 | 4 | 0.00 | 0.00 | 0.00 | 3 050.00 |  |
| 2017.01.05 16:00:00 | 5 | EURUSD | buy | out | 0.10 | 1.05307 | 5 | 0.00 | 0.00 | -15.80 | 3 034.20 |  |
| 2017.01.05 16:00:00 | 6 | EURUSD | buy | in | 0.10 | 1.05307 | 6 | 0.00 | 0.00 | 0.00 | 3 034.20 |  |
| 2017.01.05 17:29:15 | 7 | EURUSD | sell | out | 0.10 | 1.05807 | 7 | 0.00 | 0.00 | 50.00 | 3 084.20 | tp 1.05807 |
|  | **0.00** | **0.00** | **84.20** | **3 084.20** |  |
|  |

For fixed risk money management, the coded setting is to risk $100. The EA, in turn, would have to calculate a volume that would fit to this set risk. The result is shown below.

|  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Orders** |
| **Open Time** | **Order** | **Symbol** | **Type** | **Volume** | **Price** | **S / L** | **T / P** | **Time** | **State** | **Comment** |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | 0.20 / 0.20 | 1.04597 | 1.05097 | 1.04097 | 2017.01.03 10:00:00 | filled |  |
| 2017.01.03 11:34:38 | 3 | EURUSD | buy | 0.20 / 0.20 | 1.04097 |  |  | 2017.01.03 11:34:38 | filled | tp 1.04097 |
| 2017.01.05 11:00:00 | 4 | EURUSD | sell | 0.20 / 0.20 | 1.05149 | 1.05649 | 1.04649 | 2017.01.05 11:00:00 | filled |  |
| 2017.01.05 16:00:00 | 5 | EURUSD | buy | 0.20 / 0.20 | 1.05307 |  |  | 2017.01.05 16:00:00 | filled |  |
| 2017.01.05 16:00:00 | 6 | EURUSD | buy | 0.20 / 0.20 | 1.05307 | 1.04807 | 1.05807 | 2017.01.05 16:00:00 | filled |  |
| 2017.01.05 17:29:15 | 7 | EURUSD | sell | 0.20 / 0.20 | 1.05807 |  |  | 2017.01.05 17:29:15 | filled | tp 1.05807 |
|  |
| **Deals** |
| **Time** | **Deal** | **Symbol** | **Type** | **Direction** | **Volume** | **Price** | **Order** | **Commission** | **Swap** | **Profit** | **Balance** | **Comment** |
| 2017.01.01 00:00:00 | 1 |  | balance |  |  |  |  | 0.00 | 0.00 | 3 000.00 | 3 000.00 |  |
| 2017.01.03 10:00:00 | 2 | EURUSD | sell | in | 0.20 | 1.04597 | 2 | 0.00 | 0.00 | 0.00 | 3 000.00 |  |
| 2017.01.03 11:34:38 | 3 | EURUSD | buy | out | 0.20 | 1.04097 | 3 | 0.00 | 0.00 | 100.00 | 3 100.00 | tp 1.04097 |
| 2017.01.05 11:00:00 | 4 | EURUSD | sell | in | 0.20 | 1.05149 | 4 | 0.00 | 0.00 | 0.00 | 3 100.00 |  |
| 2017.01.05 16:00:00 | 5 | EURUSD | buy | out | 0.20 | 1.05307 | 5 | 0.00 | 0.00 | -31.60 | 3 068.40 |  |
| 2017.01.05 16:00:00 | 6 | EURUSD | buy | in | 0.20 | 1.05307 | 6 | 0.00 | 0.00 | 0.00 | 3 068.40 |  |
| 2017.01.05 17:29:15 | 7 | EURUSD | sell | out | 0.20 | 1.05807 | 7 | 0.00 | 0.00 | 100.00 | 3 168.40 | tp 1.05807 |
|  | **0.00** | **0.00** | **168.40** | **3 168.40** |  |
|  |

Always remember that the order manager will always consider the main stop to be the one to use for money management. If no main stop was declared, the stoploss-based money management methods cannot be used, even if there are other stops present.

### Conclusion

In this article, we have discussed the addition of stop levels in a cross-platform expert advisor. The two trading platforms, although having many parallel features, vary significantly in the way stop levels are implemented. This article has a provided a method where these differences can be reconciled in order for an expert advisor to be compatible with both platforms.

### Programs Used In The Article

| \# | Name | Type | Description |
| --- | --- | --- | --- |
| 1. | stops\_ha\_ma1.mqh | Header File | The main header file used for the expert advisor in the first example |
| 2. | stops\_ha\_ma1.mq4 | Expert Advisor | The main source file used for the MQL4 expert advisor in the first example |
| 3. | stops\_ha\_ma1.mq5 | Expert Advisor | The main source file used for the MQL5 expert advisor in the first example |
| 4. | stops\_ha\_ma2.mqh | Header File | The main header file used for the expert advisor in the second example |
| 5. | stops\_ha\_ma2.mq4 | Expert Advisor | The main source file used for the MQL4 expert advisor in the second example |
| 6. | stops\_ha\_ma2.mq5 | Expert Advisor | The main source file used for the MQL5 expert advisor in the second example |
| 7. | stops\_ha\_ma3.mqh | Header File | The main header file used for the expert advisor in the third example |
| 8. | stops\_ha\_ma3.mq4 | Expert Advisor | The main source file used for the MQL4 expert advisor in the third example |
| 9. | stops\_ha\_ma3.mq5 | Expert Advisor | The main source file used for the MQL5 expert advisor in the third example |

### Class Files Featured In The Article

| \# | Name | Type | Description |
| --- | --- | --- | --- |
| 1. | MQLx\\Base\\Stop\\StopBase.mqh | Header File | CStop (base class) |
| --- | --- | --- | --- |
| 2. | MQLx\\MQL4\\Stop\\Stop.mqh | Header File | CStop (MQL4 version) |
| --- | --- | --- | --- |
| 3. | MQLx\\MQL5\\Stop\\Stop.mqh | Header File | CStop (MQL5 version) |
| --- | --- | --- | --- |
| 4. | MQLx\\Base\\Stop\\StopsBase.mqh | Header File | CStops (CStop container, base class) |
| --- | --- | --- | --- |
| 5. | MQLx\\MQL4\\Stop\\Stops.mqh | Header File | CStops (MQL4 version) |
| --- | --- | --- | --- |
| 6. | MQLx\\MQL5\\Stop\\Stops.mqh | Header File | CStops (MQL5 version) |
| --- | --- | --- | --- |
| 7. | MQLx\\Base\\Stop\\StopLineBase.mqh | Header File | CStopLine (graphical representation, base class) |
| --- | --- | --- | --- |
| 8. | MQLx\\MQL4\\Stop\\StopLine.mqh | Header File | CStopLine (MQL4 version) |
| --- | --- | --- | --- |
| 9. | MQLx\\MQL5\\Stop\\StopLine.mqh | Header File | CStopLine (MQL5 version) |
| --- | --- | --- | --- |
| 10. | MQLx\\Base\\Order\\OrderStopBase.mqh | Header File | COrderStop (base class) |
| --- | --- | --- | --- |
| 11. | MQLx\\MQL4\\Order\\OrderStop.mqh | Header File | COrderStop (MQL4 version) |
| --- | --- | --- | --- |
| 12. | MQLx\\MQL5\\Order\\OrderStop.mqh | Header File | COrderStop (MQL5 version) |
| --- | --- | --- | --- |
| 13\. | MQLx\\Base\\Order\\OrderStopVirtualBase.mqh | Header File | COrderStopVirtual (virtual stop level, base class) |
| --- | --- | --- | --- |
| 14. | MQLx\\Base\\Order\\OrderStopVirtual.mqh | Header File | COrderStopVirtual (MQL4 version) |
| --- | --- | --- | --- |
| 15. | MQLx\\Base\\Order\\OrderStopVirtual.mqh | Header File | COrderStopVirtual (MQL5 version) |
| --- | --- | --- | --- |
| 16. | MQLx\\Base\\Order\\OrderStopPendingBase.mqh | Header File | COrderStopPending (pending order stop level, base class) |
| --- | --- | --- | --- |
| 17. | MQLx\\Base\\Order\\OrderStopPending.mqh | Header File | COrderStopPending (MQL4 version) |
| --- | --- | --- | --- |
| 18\. | MQLx\\Base\\Order\\OrderStopPending.mqh | Header File | COrderStopPending (MQL5 version) |
| --- | --- | --- | --- |
| 19\. | MQLx\\Base\\Order\\OrderStopBroker.mqh | Header File | COrderStopBroker (broker-based stop level, base class) |
| --- | --- | --- | --- |
| 20. | MQLx\\Base\\Order\\OrderStopBroker.mqh | Header File | COrderStopBroker (MQL4 version) |
| --- | --- | --- | --- |
| 21. | MQLx\\Base\\Order\\OrderStopBroker.mqh | Header File | COrderStopBroker (MQL5 version) |
| --- | --- | --- | --- |
| 22. | MQLx\\Base\\Order\\OrderStopsBase.mqh | Header File | COrderStops (COrderStop container, base class) |
| --- | --- | --- | --- |
| 23. | MQLx\\Base\\Order\\OrderStops.mqh | Header File | COrderStops (MQL4 version) |
| --- | --- | --- | --- |
| 24. | MQLx\\Base\\Order\\OrderStops.mqh | Header File | COrderStops (MQL5 version) |
| --- | --- | --- | --- |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/3620.zip "Download all attachments in the single ZIP archive")

[tester.zip](https://www.mql5.com/en/articles/download/3620/tester.zip "Download tester.zip")(1445.59 KB)

[MQL5.zip](https://www.mql5.com/en/articles/download/3620/mql5.zip "Download MQL5.zip")(708.29 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Cross-Platform Expert Advisor: The CExpertAdvisor and CExpertAdvisors Classes](https://www.mql5.com/en/articles/3622)
- [Cross-Platform Expert Advisor: Custom Stops, Breakeven and Trailing](https://www.mql5.com/en/articles/3621)
- [Cross-Platform Expert Advisor: Time Filters](https://www.mql5.com/en/articles/3395)
- [Cross-Platform Expert Advisor: Money Management](https://www.mql5.com/en/articles/3280)
- [Cross-Platform Expert Advisor: Signals](https://www.mql5.com/en/articles/3261)
- [Cross-Platform Expert Advisor: Order Manager](https://www.mql5.com/en/articles/2961)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/215228)**
(18)


![Enrico Lambino](https://c.mql5.com/avatar/2014/10/54465D5F-0757.jpg)

**[Enrico Lambino](https://www.mql5.com/en/users/iceron)**
\|
16 Sep 2017 at 21:38

**Shephard Mukachi:**

Hi Enrico,

Thanks for the quick response. You are quite right about the forward declarations.  I was worried that it might be a deeper problem than that.  It makes sense that the object using the forwardly declared class has no knowledge of that class's members.

I use a different model for my execution engine which uses the strategy pattern.  I have a few annoying problems with it, and was just taking a look at your library when I came across those issues.

Your work is really great, you have sound design and engineering skills and thanks for being kind enough to share.

Thanks, Shep.

Hi Shep,

You're welcome and thank you for letting me know. I recall I was also a bit puzzled when I ran into that issue with forward declarations.

Glad to hear that you have found the articles to be useful. I wish you all the best in your efforts in building your own EA engine.

Regards, Enrico

![Enrico Lambino](https://c.mql5.com/avatar/2014/10/54465D5F-0757.jpg)

**[Enrico Lambino](https://www.mql5.com/en/users/iceron)**
\|
16 Sep 2017 at 21:50

**mbjen:**

It's not updating because of checking in COrderStopVirtualBase::Update

StopLoss() returns new value but sl\_line still old...

The purpose of COrderStopVirtualBase::Update is actually the opposite of what you intend. It is meant to adjust the sl/tp value when its own stop line is updated, usually from outside the EA (dragging on chart or directly altering the value on the [object properties](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property "MQL5 documentation: Object Properties") window). Use MoveStopLoss() and MoveTakeProfit() methods instead for virtual stops.

This is supposed to be for the next article, but if you are eager, you may want to take a look at the CheckTrailing() method of COrderStopBase. It modifies the order stop when eligible for trailing. The method applies to all three types:

```
bool COrderStopBase::CheckTrailing(void)
  {
   if(!CheckPointer(m_stop) || m_order.IsClosed() || m_order.IsSuspended() ||
      (m_stoploss_closed && m_takeprofit_closed))
      return false;
   double stoploss=0,takeprofit=0;
   string symbol=m_order.Symbol();
   ENUM_ORDER_TYPE type=m_order.OrderType();
   double price=m_order.Price();
   double sl = StopLoss();
   double tp = TakeProfit();
   if(!m_stoploss_closed)
      stoploss=m_stop.CheckTrailing(symbol,type,price,sl,TRAIL_TARGET_STOPLOSS);
   if(!m_takeprofit_closed)
      takeprofit=m_stop.CheckTrailing(symbol,type,price,tp,TRAIL_TARGET_TAKEPROFIT);
   if(!IsStopLossValid(stoploss))
      stoploss=0;
   if(!IsTakeProfitValid(takeprofit))
      takeprofit=0;
   return Modify(stoploss,takeprofit); //<---- this
  }
```

Alternatively, the CTrail class can also be used to alter sl/tp levels without having to retrieve an instance of an order stop (not just for trailing or breakeven).

![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
10 Oct 2017 at 15:30

How far from practice and convenience is the proposed cross-platform approach. As its own cumbersome bicycle, it's fine. But why publish it! It's easier to write everything completely your own, even for a beginner, than to study and master this monster. I don't understand.

[SB under MT5](https://www.mql5.com/en/docs/standardlibrary/tradeclasses) is an example of elegance compared to this cycle of articles. And it is also [ported to MT4](https://www.mql5.com/en/articles/3068).

![Simalb](https://c.mql5.com/avatar/2017/11/59FF5B41-9224.png)

**[Simalb](https://www.mql5.com/en/users/simalb)**
\|
6 Nov 2017 at 19:52

I really enjoyed the article, it opened my eyes. Please, write a follow up article. I am really interested in how you would approach scalping with this tool.

Thanks a million for this and all your other work.

![Viktar Dzemikhau](https://c.mql5.com/avatar/2020/11/5FC52891-F98D.jpg)

**[Viktar Dzemikhau](https://www.mql5.com/en/users/hoz)**
\|
25 Apr 2018 at 15:19

**fxsaber:**

How far from practice and convenience is the proposed cross-platform approach. As its own cumbersome bicycle, it's fine. But why publish it! It's easier to write everything completely your own, even for a beginner, than to study and master this monster. I don't understand.

[SB under MT5](https://www.mql5.com/en/docs/standardlibrary/tradeclasses) is an example of elegance compared to this cycle of articles. And it's also [ported to MT4](https://www.mql5.com/en/articles/3068).

I completely agree. Even being very familiar with OOP, this author's approach is too perverted. There are too many excesses and the code is not a code at all but a "monster"....

![Deep Neural Networks (Part I). Preparing Data](https://c.mql5.com/2/48/Deep_Neural_Networks_01.png)[Deep Neural Networks (Part I). Preparing Data](https://www.mql5.com/en/articles/3486)

This series of articles continues exploring deep neural networks (DNN), which are used in many application areas including trading. Here new dimensions of this theme will be explored along with testing of new methods and ideas using practical experiments. The first article of the series is dedicated to preparing data for DNN.

![How to conduct a qualitative analysis of trading signals and select the best of them](https://c.mql5.com/2/27/MQL5-avatar-qualityAnalysis-001.png)[How to conduct a qualitative analysis of trading signals and select the best of them](https://www.mql5.com/en/articles/3166)

The article deals with evaluating the performance of Signals Providers. We offer several additional parameters highlighting signal trading results from a slightly different angle than in traditional approaches. The concepts of the proper management and perfect deal are described. We also dwell on the optimal selection using the obtained results and compiling the portfolio of multiple signal sources.

![Custom Walk Forward optimization in MetaTrader 5](https://c.mql5.com/2/28/MQL5-avatar-WalkForward-001.png)[Custom Walk Forward optimization in MetaTrader 5](https://www.mql5.com/en/articles/3279)

The article deals with the approaches enabling accurate simulation of walk forward optimization using the built-in tester and auxiliary libraries implemented in MQL.

![Naive Bayes classifier for signals of a set of indicators](https://c.mql5.com/2/27/MQL5-avatar-naiveClass-001.png)[Naive Bayes classifier for signals of a set of indicators](https://www.mql5.com/en/articles/3264)

The article analyzes the application of the Bayes' formula for increasing the reliability of trading systems by means of using signals from multiple independent indicators. Theoretical calculations are verified with a simple universal EA, configured to work with arbitrary indicators.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=yvzstwdfmunugipdcrmusescpkshmacu&ssn=1769192238630074333&ssn_dr=0&ssn_sr=0&fv_date=1769192238&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F3620&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Cross-Platform%20Expert%20Advisor%3A%20Stops%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919223812082089&fz_uniq=5071721284406029613&sv=2552)

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