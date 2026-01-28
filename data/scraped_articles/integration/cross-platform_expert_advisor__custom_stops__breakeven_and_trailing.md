---
title: Cross-Platform Expert Advisor: Custom Stops, Breakeven and Trailing
url: https://www.mql5.com/en/articles/3621
categories: Integration, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:16:39.810270
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/3621&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071713020888952069)

MetaTrader 5 / Examples


### Table of Contents

01. [Introduction](https://www.mql5.com/en/articles/3621#introduction)
02. [Custom Stops](https://www.mql5.com/en/articles/3621#custom)
03. [Modification of Stop Level](https://www.mql5.com/en/articles/3621#modification)
04. [Breakeven](https://www.mql5.com/en/articles/3621#breakeven)
05. [Trailingstop](https://www.mql5.com/en/articles/3621#trailingstop)
06. [Trailing Takeprofit](https://www.mql5.com/en/articles/3621#trailingtake)
07. [Implementation](https://www.mql5.com/en/articles/3621#implementation)
08. [CTrails (Container)](https://www.mql5.com/en/articles/3621#ctrails)
09. [Extending CTrail](https://www.mql5.com/en/articles/3621#extending)
10. [Examples](https://www.mql5.com/en/articles/3621#examples)


### Introduction

In the previous article, Cross-Platform Expert Advisor: Stops, it has been shown how the CStop class can be used to set stoploss and takeprofit levels for a given trade in a cross-platform expert advisor. The said levels can be expressed in terms of pips or points. However, in a good number of real-world expert advisors, this is not always the case. Stoploss and takeprofit often do not need to be calculated based on their distance from the entry price. Rather, in those instances, the stoploss and/or takeprofit levels are expressed in terms of the chart price, usually as a result of some other calculation. This article will discuss on how to use the CStop class to specify stoploss and takeprofit levels in terms of chart price. The article also discusses how such stop levels can be modified through the use of the library featured in this article. Moreover, it also introduces a new class, CTrail, which closely resembles how custom stops are implemented. However, unlike custom stops, CTrail is used primarily to define the evolution of a stop level over time.

### Custom Stops

In the previous article, it has been
shown how a pip- or point-based stoploss and takeprofit can be
implemented in cross-platform expert advisors. However, this is not
always the case for all EAs. There are strategies which require the
dynamic calculation of stoploss and takeprofit levels, usually based
on data such as time-series data (OHLC) as well as outputs from
technical indicators.

Dynamically-calculated stoploss and
takeprofit levels can be setup in CStop through its two methods,
StopLossCustom and TakeProfitCustom. The following code snippet shows
the two methods:

```
bool CStopBase::StopLossCustom(void)
  {
   return m_stoploss==0;
  }

bool CStopBase::TakeProfitCustom(void)
  {
   return m_takeprofit==0;
  }
```

The coder or programmer is free to
extend these methods in order to achieve the desired stoploss or
takeprofit level for any given instance of CStop.

The two methods mentioned above are
actually overloaded methods in the class. The following are the
alternate methods, which return values of type [Boolean](https://www.mql5.com/en/docs/basis/types/integer/boolconst):

```
bool CStopBase::StopLossCustom(void)
  {
   return m_stoploss==0;
  }

bool CStopBase::TakeProfitCustom(void)
  {
   return m_takeprofit==0;
  }
```

Both functions return true if a
particular class member (either m\_stoploss or m\_takeprofit) has a
zero value. Their purpose will be discussed shortly.

CStop calculates a stop level using the
following guideline. The stop level can be a stoploss or a takeprofit
level, represented in CStop as the members m\_stoploss and
m\_takeprofit. In the steps below, assume that were are only
dealing with the stoploss level (m\_stoploss):

1. If m\_stoploss is zero, use the
    StopLossCustom in calculating the stoploss
2. If m\_stoploss is not zero, use
    m\_stoploss in calculating the actual stoploss of the trade with respect to the entry price


The same guideline is used for
calculating takeprofit using the method TakeProfitCustom and the
class member m\_takeprofit.

The actual usage of these four methods
can be found in two major areas. For main stop levels, the method
call can be found within the order manager itself (COrderManager).
For the other stops, within each order instance (COrder).

There are instances where the main stop
levels are sent to the broker along with the initial trade request,
and thus, the expert advisor would need these information the moment
it starts to process the trade request, not after the initial trade
has been successfully sent. This is true for broker-based stops in
Metatrader 4 and Metatrader 5 hedging mode. For these modes, the
information for the stoploss and takeprofit levels need to be
included in the trade request for the OrderSend ( [MQL4](https://docs.mql4.com/trading/ordersend "https://docs.mql4.com/trading/ordersend"), [MQL5](https://www.mql5.com/en/docs/trading/ordersend))
function, and these stop levels only apply on the main trade
concerned (not global).

Within the TradeOpen method of the
order manager, we can find the call to the two methods of CStop. The
following shows the code for the MQL4 version:

```
bool COrderManager::TradeOpen(const string symbol,ENUM_ORDER_TYPE type,double price,bool in_points=true)
  {
   int trades_total = TradesTotal();
   int orders_total = OrdersTotal();
   m_symbol=m_symbol_man.Get(symbol);
   if(!CheckPointer(m_symbol))
      return false;
   if(!IsPositionAllowed(type))
      return true;
   if(m_max_orders>orders_total && (m_max_trades>trades_total || m_max_trades<=0))
     {
      ENUM_ORDER_TYPE ordertype=type;
      if(in_points)
         price=PriceCalculate(type);
      double sl=0,tp=0;
      if(CheckPointer(m_main_stop)==POINTER_DYNAMIC)
        {
         sl = m_main_stop.StopLossCustom()?m_main_stop.StopLossCustom(symbol,type,price):m_main_stop.StopLossCalculate(symbol,type,price);
         tp = m_main_stop.TakeProfitCustom()?m_main_stop.TakeProfitCustom(symbol,type,price):m_main_stop.TakeProfitCalculate(symbol,type,price);
        }
      double lotsize=LotSizeCalculate(price,type,sl);
      if(CheckPointer(m_main_stop)==POINTER_DYNAMIC)
      {
         if (!m_main_stop.Broker())
         {
            sl = 0;
            tp = 0;
         }
      }
      int ticket=(int)SendOrder(type,lotsize,price,sl,tp);
      if(ticket>0)
        {
         if(OrderSelect(ticket,SELECT_BY_TICKET))
         {
            COrder *order = m_orders.NewOrder(OrderTicket(),OrderSymbol(),OrderMagicNumber(),
(ENUM_ORDER_TYPE)::OrderType(),::OrderLots(),::OrderOpenPrice());
            if (CheckPointer(order))
            {
               LatestOrder(GetPointer(order));
               return true;
            }
         }
        }
     }
   return false;
  }
```

The following shows the code for the
MQL5 version:

```
bool COrderManager::TradeOpen(const string symbol,ENUM_ORDER_TYPE type,double price,bool in_points=true)
  {
   bool ret=false;
   double lotsize=0.0;
   int trades_total =TradesTotal();
   int orders_total = OrdersTotal();
   m_symbol=m_symbol_man.Get(symbol);
   if(!IsPositionAllowed(type))
      return true;
   if(m_max_orders>orders_total && (m_max_trades>trades_total || m_max_trades<=0))
     {
      if(in_points)
         price=PriceCalculate(type);
      double sl=0.0,tp=0.0;
      if(CheckPointer(m_main_stop)==POINTER_DYNAMIC)
        {
         sl = m_main_stop.StopLossCustom()?m_main_stop.StopLossCustom(symbol,type,price):m_main_stop.StopLossCalculate(symbol,type,price);
         tp = m_main_stop.TakeProfitCustom()?m_main_stop.TakeProfitCustom(symbol,type,price):m_main_stop.TakeProfitCalculate(symbol,type,price);
        }
      lotsize=LotSizeCalculate(price,type,m_main_stop==NULL?0:m_main_stop.StopLossCalculate(symbol,type,price));
      if(CheckPointer(m_main_stop)==POINTER_DYNAMIC)
      {
         if (!m_main_stop.Broker() || !IsHedging())
         {
            sl = 0;
            tp = 0;
         }
      }
      ret=SendOrder(type,lotsize,price,sl,tp);
      if(ret)
      {
         COrder *order = m_orders.NewOrder((int)m_trade.ResultOrder(),m_trade.RequestSymbol(),(int)m_trade.RequestMagic(),
m_trade.RequestType(),m_trade.ResultVolume(),m_trade.ResultPrice());
         if (CheckPointer(order))
         {
            LatestOrder(GetPointer(order));
            return true;
         }
      }
     }
   return ret;
  }
```

Although TradeOpen is a
split-implementation, we can find a common denominator in terms of
how the two versions of the method calculate the main stop levels.
First, it calculates the stoploss and takeprofit (represented as sl
and tp, respectively) from the main stop level. This is done whether
or not the stoploss and takeprofit are needed in the actual initial
trade request, since the information may also be needed in the
calculation of lotsize (money management).

After the lotsize has been calculated,
it is important the information be reset to zero in certain cases. In
MQL4, the variables sl and tp are set to zero if the main stop is not
a broker-based stop:

```
if(CheckPointer(m_main_stop)==POINTER_DYNAMIC)
{
   if (!m_main_stop.Broker())
   {
      sl = 0;
      tp = 0;
   }
}
```

In MQL5, we are trying to avoid the use
of stoploss and takeprofit in netting mode, since those would apply
to the entire position, not just to the volume of the trade to be
entered. Thus, the variables are reset to zero if the stop is not a
broker-based stop OR the platform is in netting mode:

```
if(CheckPointer(m_main_stop)==POINTER_DYNAMIC)
{
   if (!m_main_stop.Broker() || !IsHedging())
   {
      sl = 0;
      tp = 0;
   }
}
```

As discussed in the previous article
(see Cross-Platform Expert Advisor: Stops), the expert advisor would
convert broker-based stops to pending order stops in netting mode, so
there is no longer any need to send the stoploss and takeprofit on
the trade request for the initial trade under this mode.

For trades having multiple stoploss and
takeprofit levels, these levels are processed as soon as the main
trade has been successfully processed. The trade has to be entered
first, since these stop levels would be meaningless if the main trade
has not actually entered the market. We can find the generation of
these other stop levels within the initialization of the COrder
instance, available through COrder's method CreateStops (from COrderBase, because MQL4 and MQL5 share the implementation):

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

As shown in the code above, a COrderStop instance is created for each instance of CStop found by the expert advisor. These additional stop levels, however, are only created after the main trade has successfully entered the market.

### Modification of Stop Level

CStop has its own separate instance of
CTradeManager, and thus CExpertTradeX (an extension of CExpertTrade,
from the MQL5 Standard Library). This instance is independent of the
instance found in the order manager (COrderManager), which is used
exclusively for handling the entries of main trades. The modification
of stop levels therefore is not managed by COrderManager but the very
instance of CStop itself. However, since the modification of stop
levels is to be performed on a per-order basis, the call for
modifying a stop level must originate from trade to be modified
itself i.e. within the COrder instance representing the trade.

The monitoring of the stop levels
begins with the CheckStops method of COrder, whose code is shown below:

```
void COrderBase::CheckStops(void)
  {
   m_order_stops.Check(m_volume);
  }
```

Here, it just calls the Check method of
one of its class members, which is an instance of COrderStops. As we
recall from the previous article, COrderStops is a container for
pointers to instances of COrderStop. COrderStop, on the other hand,
is a representation of CStop in an actual instance of COrder.

Now, let us examine the Check method of
COrderStops. The method is shown in the following code snippet:

```
COrderStopsBase::Check(double &volume)
  {
   if(!Active())
      return;
   for(int i=0;i<Total();i++)
     {
      COrderStop *order_stop=(COrderStop *)At(i);
      if(CheckPointer(order_stop))
        {
         if(order_stop.IsClosed())
            continue;
         order_stop.CheckTrailing();
         order_stop.Update();
         order_stop.Check(volume);
         if(!CheckNewTicket(order_stop))
            return;
        }
     }
  }
```

As we can see from the code, it
performs at least five functions on a particular instance of
COrderStop

1. It checks if a particular instance of
    COrderStop is already closed (IsClosed method)
2. It updates the stop level based on the
    trailingstop instances assigned to it, if there are any
    (CheckTrailing method)
3. It updates the stop level (Update
    method)
4. It checks if the market has hit a
    particular stop level (Check method)
5. It checks if a new ticket ID represents
    the trade

Of these tasks, the second task is the
one relating to the trailing of the stoploss or takeprofit. The first
task is only used to see if further action is needed on the stop
level (if the stop level is closed, the expert advisor need not check
it any further). The third method is used when the stop level was
modified from outside the expert advisor (e.g. dragging of stop line for virtual stops). The fourth task is used to see if
the stop level (updated or not by trailing) is hit by the market. The
fifth task is used only in Metatrader 4, since only in this platform
will the expert advisor experience a trade change its ticket ID when
partially closed. Metatrader 5 makes a simpler implementation in this case, since it maintains a clear footprint of trade progression.

The instance of COrderStop represents
the stop level of a trade as defined by CStop. Any trailing of a stop
level therefore, will ultimately result to the alteration of an
instance of this class object. The following code shows its
CheckTrailing method:

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
   return Modify(stoploss,takeprofit);
  }
```

From this code, we see that it still
follows the general guideline of not modifying a particular stop
level when that stop level has already closed. If the order stop
level is eligible, then it moves on to calling the method
CheckTrailing of the CStop instance it represents. Now, let us
examine the CheckTrailing method of CStop:

```
double CStopBase::CheckTrailing(const string symbol,const ENUM_ORDER_TYPE type,const double entry_price,const double price,const ENUM_TRAIL_TARGET mode)
  {
   if(!CheckPointer(m_trails))
      return 0;
   return m_trails.Check(symbol,type,entry_price,price,mode);
  }
```

Here, CStop calls the Check method of
one of its class members, m\_trails. m\_trails is simply a container
for pointers to trailing or trailingstop objects. The code of the method is shown
below:

```
double CTrailsBase::Check(const string symbol,const ENUM_ORDER_TYPE type,const double entry_price,const double price,const ENUM_TRAIL_TARGET mode)
  {
   if(!Active())
      return 0;
   double val=0.0,ret=0.0;
   for(int i=0;i<Total();i++)
     {
      CTrail *trail=At(i);
      if(!CheckPointer(trail))
         continue;
      if(!trail.Active())
         continue;
      int trail_target=trail.TrailTarget();
      if(mode!=trail_target)
         continue;
      val=trail.Check(symbol,type,entry_price,price,mode);
      if((type==ORDER_TYPE_BUY && trail_target==TRAIL_TARGET_STOPLOSS) || (type==ORDER_TYPE_SELL && trail_target==TRAIL_TARGET_TAKEPROFIT))
      {
         if(val>ret || ret==0.0)
            ret=val;
      }
      else if((type==ORDER_TYPE_SELL && trail_target==TRAIL_TARGET_STOPLOSS) || (type==ORDER_TYPE_BUY && trail_target==TRAIL_TARGET_TAKEPROFIT))
      {
         if(val<ret || ret==0.0)
            ret=val;
      }
     }
   return ret;
  }
```

At this point, it is enough to
understand the CTrails container iterates on its own instances of
CTrail, and returns a final value. This final value is in terms of
the chart price of the selected symbol, and thus is of type [double](https://www.mql5.com/en/docs/basis/types/double).
This is the new value of the stoploss or takeprofit after a
successful trailing. Now, let us go back to the CheckTrailing method
of COrderStop, since it is from this method where the actual call for
the modification of the stop level would take place:

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
   return Modify(stoploss,takeprofit);
  }
```

The return value of this method is of
type Boolean, which is the result of the alteration of the stop level
through COrderStop's Modify method (returns true if successful). But
before it sends the request for modification, it checks if the
stoploss and takeprofit are valid, using the methods IsStopLossValid
and IsTakeProfitValid. If the proposed value is not valid, it is reset
to zero:

```
bool COrderStopBase::IsStopLossValid(const double stoploss) const
  {
   return stoploss!=StopLoss();
  }

bool COrderStopBase::IsTakeProfitValid(const double takeprofit) const
  {
   return takeprofit!=TakeProfit();
  }
```

In the code above, for both stoploss and takeprofit, the proposed value should not be equal to the current value.

After the evaluation of stoploss and
takeprofit, the Modify method of COrderStop is called, which is shown
below:

```
bool COrderStopBase::Modify(const double stoploss,const double takeprofit)
  {
   bool stoploss_modified=false,takeprofit_modified=false;
   if(stoploss>0 && takeprofit>0)
     {
      if(ModifyStops(stoploss,takeprofit))
        {
         stoploss_modified=true;
         takeprofit_modified=true;
        }
     }
   else if(stoploss>0 && takeprofit==0)
      stoploss_modified=ModifyStopLoss(stoploss);
   else if(takeprofit>0 && stoploss==0)
      takeprofit_modified=ModifyTakeProfit(takeprofit);
   return stoploss_modified || takeprofit_modified;
  }
```

At this point, three types of
operations are performed, depending on the value of stoploss and
takeprofit. Normally, the modification can be performed in one
action, as in the case of broker-based stops, but this is not always
the case. Stops based on pending orders as well as virtual stops have
to be processed individually each for stoploss and takeprofit. The code for modifying both values can be found on the following methods:

```
bool COrderStopBase::ModifyStops(const double stoploss,const double takeprofit)
  {
   return ModifyStopLoss(stoploss) && ModifyTakeProfit(takeprofit);
  }
```

The method ModifyStops simply calls the
other two methods. The split implementation starts at this point, based on two factors: (1) type of compiler used (MQL4 or MQL5) and (2) the type of stop
(broker-based, pending, or virtual). If the stop is a broker-based stop, then it would result in a trade request modifying the main trade. If the stop is based on pending orders, then the expert advisor would have to move the entry price of the targeted pending order. If the stop is a virtual stop, then the expert advisor simply has to update its internal data concerning the stop level.

COrderStop does not have a trade object
(or a pointer to it) as one of its class members, and thus, it is not
inherently capable of altering its own stop levels. It still needs to
rely on the CStop instance it represents for the trade in order to
modify its own stop levels. Therefore, any modification of a stop level will ultimately result in calling a certain method in CStop.

### Breakeven

A break-even, or simply breakeven, is a
point where the revenue earned is equal to the cost. In trading, the
revenue is the actual profit of the trade, while the cost is the
spread and/or commission. This is the point where a trade leaves the
market with zero profit/loss. For market makers, or brokers that do
not charge commissions, the breakeven point is usually the entry price of the
trade.

In expert advisors, a trade is said to
have broken even when the expert has successfully set the trade
stoploss equal to its entry price. Upon reaching this state, the
maximum risk of the trade becomes zero. However, the trade can still
leave the market through other means, such as exit signals, usually
in profit.

In breaking-even a trade, an expert
advisor is limited by at least two restrictions:

1. Dynamics of the trade stoploss
2. The broker's minimum distance

The stoploss of a trade should be
always below the current market price for long positions, and above
the market price for short positions. Given this restriction, an
expert advisor can move the stoploss to the breakeven point only if
the trade is in the state of unrealized profit.

The broker's minimum distance
requirement still applies when modifying a trade's stoploss, not just
to the initial entry of the trade. Thus, it is often not possible to
set a trade's stoploss to the breakeven point as soon as it has
entered the market. The market has to move a significant distance in
favor of the position before the expert advisor can move the stoploss
in the direction of the breakeven point.

Given these restrictions, in designing
the breakeven feature, we need to consider at least two factors:

1. The activation price
2. The new stoploss upon activation

The activation price, or the trigger
price, is the price that the market has to reach in order for an
expert advisor to move the stoploss of a trade to the breakeven
point. This activation price has to be evaluated on a per-trade
basis, since trades are expected to have varying entry prices.

Upon activation, the expert advisor
will be compelled to move the stoploss of the trade to the breakeven
point. Usually, this is the entry price of the trade in question, but
this is not always the case. For brokers that charge commission, the
breakeven point is somewhere above the entry price for long
positions, and somewhere below the entry price for short positions.

These values are usually calculated in
reference to another value, such as the current market price and the
entry price of the trade.

The following feature shows a diagram of calculation of the breakeven in the way described above. Based on the this flowchart, the three values activation, deactivation, and new stop level are calculated beforehand. If the current price level is greater than or equal to the minimum price level required for the initial stage (setting stoploss to breakeven), the calculated new stop level calculated earlier will be used as the tentative new stop level for the trade. If not, then the output would be zero. The next step would be to check if the new stop level is within the current stop level, which should always return true if the previous condition was satisfied, and thus would return the calculated stop level as the final output.

![Breakeven](https://c.mql5.com/2/29/breakeven.png)

### Trailingstop

Trailingstop can be considered a
special case of breakeven. While breakeven is usually applied only
once, a trailingstop can be applied any number of times. Upon
activation, the trailing usually applies throughout the remaining
lifetime of the trade. At least three factors need to be considered
in designing a trailingstop:

1. The activation price
2. The new stoploss upon activation
3. The frequency of trailing

The trailingstop feature shares the
first two variables with breakeven. However, unlike in breakeven, it
is possible for the activation price in trailingstop to be a point in
the price chart where the trade is in a state of unrealized loss or “out of
the money”. It is possible for the stoploss of a trade to be
modified while that trade is losing, and so it is also possible to
trail the stoploss as well. When this happens, the effective stoploss
of the trade will still result in loss when hit by the market, as it
does not yet satisfy the prerequisites for breaking even or even going past it.

The third factor is the frequency of
trailing the stoploss, or the “step”. This determines the
distance (in points or pips) of the current stoploss level to the
next, after the initial trailing of the stoploss. This makes it quite
similar to the first variable, which is the activation price.
However, unlike the activation price, the calculated price arising
from this third variable changes with each stage of trailing.

There is a fourth factor in some
trailingstops, which is the deactivation price. Upon reaching this
point for the trade, the expert advisor would stop trailing the stoploss. This is also evaluated on a per-trade basis.

The following illustration shows the
flowchart for trailing the stoploss of a trade. This is not very different from the previous diagram (breakeven). If the price is greater than the initial stage of trailing, then it is highly likely that the stop level is already beyond the initial stage. Otherwise, it would still use the initial stage of trailing as the value of the new stop level for the current run. If the stop level is not yet beyond the activation price, then it would simply follow the rest of the procedure for calculating breakeven (deactivation price does not apply to breakeven).

![Trailingstop](https://c.mql5.com/2/29/trailing.png)

### Trailing Takeprofit

Trailing or trailingstop usually refer
to the trailing of the stoploss, but it is possible to trail the
takeprofit of a trade as well. In this case, the takeprofit is
adjusted as the price moves closer to the takeprofit level (the
opposite of trailing the stoploss). Theoretically, a takeprofit being
trailed will never be hit. But this is possible in certain
situations, such as sudden price spikes and gap price movements –
any condition where the market moves faster than the expert advisor
can possibly react to with respect to this feature.

In order for the trailing of the
takeprofit to be meaningful, some things need to be satisfied:

1. The activation price should be within
    the current takeprofit level
2. The next price to trigger the next
    stage of trailing should be within the current takeprofit level
3. Ideally, the trailing should be less
    frequent

If the activation price is beyond the
current takeprofit level, then the market will hit the takeprofit
level first before the activation price. And as soon as the
takeprofit is hit, the trade will leave the market long before any
trailing of the takeprofit level can even start. For long positions,
the activation price should be less than the takeprofit level. For
short positions, the activation price should be greater than the
takeprofit level. The same is true for the second condition, only
that it applies for further stages after the initial activation.

Ideally, the trailing should be less
frequent. This means that the distance between one takeprofit level
to the next (the step) should be wide enough. The more frequent the
trailing of the takeprofit, the lesser the chances of the market
hitting the takeprofit level, as the takeprofit level is pushed away
from the current price at each stage of trailing.

### Implementation

CTrailBase, which serves as the base
class for CTrail is shown in the code below:

```
class CTrailBase : public CObject
  {
protected:
   bool              m_active;
   ENUM_TRAIL_TARGET m_target;
   double            m_trail;
   double            m_start;
   double            m_end;
   double            m_step;
   CSymbolManager   *m_symbol_man;
   CSymbolInfo      *m_symbol;
   CEventAggregator *m_event_man;
   CObject          *m_container;
public:
                     CTrailBase(void);
                    ~CTrailBase(void);
   virtual int       Type(void) const {return CLASS_TYPE_TRAIL;}
   //--- initialization
   virtual bool      Init(CSymbolManager*,CEventAggregator*);
   virtual CObject *GetContainer(void);
   virtual void      SetContainer(CObject*);
   virtual bool      Validate(void) const;
   //--- getters and setters
   bool              Active(void) const;
   void              Active(const bool);
   double            End(void) const;
   void              End(const double);
   void              Set(const double,const double,const double,const double);
   double            Start(void) const;
   void              Start(const double);
   double            Step(void) const;
   void              Step(const double);
   double            Trail(void) const;
   void              Trail(const double);
   int               TrailTarget(void) const;
   void              TrailTarget(const ENUM_TRAIL_TARGET);
   //--- checking
   virtual double    Check(const string,const ENUM_ORDER_TYPE,const double,const double,const ENUM_TRAIL_TARGET);
protected:
   //--- price calculation
   virtual double    ActivationPrice(const ENUM_ORDER_TYPE,const double);
   virtual double    DeactivationPrice(const ENUM_ORDER_TYPE,const double);
   virtual double    Price(const ENUM_ORDER_TYPE);
   virtual bool      Refresh(const string);
  };
```

The Set method allows us configure the
settings of an instance of CTrails. It only works like the usual class constructor. If needed, one may also declare a
custom constructor that calls this method:

```
void CTrailBase::Set(const double trail,const double st,const double step=1,const double end=0)
  {
   m_trail=trail;
   m_start=st;
   m_end=end;
   m_step=step;
  }
```

CTrail relies on market data for its
calculations. Thus, it has an instance of symbol manager
(CSymbolManager) as one of its class members. Refreshing the symbol
is needed before any further calculation is performed.

```
bool CTrailBase::Refresh(const string symbol)
  {
   if(!CheckPointer(m_symbol) || StringCompare(m_symbol.Name(),symbol)!=0)
      m_symbol=m_symbol_man.Get(symbol);
   return CheckPointer(m_symbol);
  }
```

The activation price is the price that
would trigger the initial movement of the stoploss or takeprofit by
the CTrail instance. Since the start, step, and end parameters are in
terms of points, the class needs to calculate the activation price in
terms of chart price. The same method is used in calculating the
other prices:

```
double CTrailBase::ActivationPrice(const ENUM_ORDER_TYPE type,const double entry_price)
  {
   if(type==ORDER_TYPE_BUY)
      return entry_price+m_start*m_symbol.Point();
   else if(type==ORDER_TYPE_SELL)
      return entry_price-m_start*m_symbol.Point();
   return 0;
  }
```

Calculating the deactivation price also
follows the same routine, but this time using the m\_end class member.

```
double CTrailBase::DeactivationPrice(const ENUM_ORDER_TYPE type,const double entry_price)
  {
   if(type==ORDER_TYPE_BUY)
      return m_end==0?0:entry_price+m_end*m_symbol.Point();
   else if(type==ORDER_TYPE_SELL)
      return m_end==0?0:entry_price-m_end*m_symbol.Point();
   return 0;
  }
```

Using a value of zero as deactivation
price means that the feature is disabled. The trailing will apply
until the main trade leaves the market.

The Price method calculates the new
value of the stoploss or takeprofit if the conditions allow the
moving or trailing of the stoploss at the time of evaluation:

```
double CTrailBase::Price(const ENUM_ORDER_TYPE type)
  {
   if(type==ORDER_TYPE_BUY)
     {
      if(m_target==TRAIL_TARGET_STOPLOSS)
         return m_symbol.Bid()-m_trail*m_symbol.Point();
      else if(m_target==TRAIL_TARGET_TAKEPROFIT)
         return m_symbol.Ask()+m_trail*m_symbol.Point();
     }
   else if(type==ORDER_TYPE_SELL)
     {
      if(m_target==TRAIL_TARGET_STOPLOSS)
         return m_symbol.Ask()+m_trail*m_symbol.Point();
      else if(m_target==TRAIL_TARGET_TAKEPROFIT)
         return m_symbol.Bid()-m_trail*m_symbol.Point();
     }
   return 0;
  }
```

Now, let us discuss the Check method. A
particular instance of CTrail can trail either the stoploss or the
takeprofit. Therefore, we just need the capacity for CTrail to
specify whether it is aiming to alter the stoploss or the takeprofit
of the trade. This is achieved by the enumeration, ENUM\_TRAIL\_TARGET. The declaration of this enumeration can be found under MQLx\\Common\\Enum\\ENUM\_TRAIL\_TARGET.mqh. Its code is shown below:

```
enum ENUM_TRAIL_TARGET
  {
   TRAIL_TARGET_STOPLOSS,
   TRAIL_TARGET_TAKEPROFIT
  };
```

The Check method of the class is shown
in the following code. Unlike the other methods discussed so far in this class, this method is a public method. This is the method to be called when the trailingstop level needs to be checked for any update.

```
double CTrailBase::Check(const string symbol,const ENUM_ORDER_TYPE type,const double entry_price,const double price,const ENUM_TRAIL_TARGET mode)
  {
   if(!Active())
      return 0;
   if(!Refresh(symbol))
      return 0;
   if(m_start==0 || m_trail==0)
      return 0;
   double next_stop=0.0,activation=0.0,deactivation=0.0,new_price=0.0,point=m_symbol.Point();
   activation=ActivationPrice(type,entry_price);
   deactivation=DeactivationPrice(type,entry_price);
   new_price=Price(type);
   if(type==ORDER_TYPE_BUY)
     {
      if (m_target==TRAIL_TARGET_STOPLOSS)
      {
         if(m_step>0 && (activation==0.0 || price>=activation-m_trail*point) && (new_price>price+m_step*point))
            next_stop=new_price;
         else next_stop=activation-m_trail*point;
         if(next_stop<price)
            next_stop=price;
         if((deactivation==0) || (deactivation>0 && next_stop>=deactivation && next_stop>0.0))
            if(next_stop<=new_price)
               return next_stop;
      }
      else if (m_target==TRAIL_TARGET_TAKEPROFIT)
      {
         if(m_step>0 && ( activation==0.0 || price>=activation) && (new_price>price+m_step*point))
            next_stop=new_price;
         else next_stop=activation+m_trail*point;
         if(next_stop<price)
            next_stop=price;
         if((deactivation==0) || (deactivation>0 && next_stop<=deactivation && next_stop>0.0))
            if(next_stop>=new_price)
               return next_stop;
      }
     }
   if(type==ORDER_TYPE_SELL)
     {
      if (m_target==TRAIL_TARGET_STOPLOSS)
      {
         if(m_step>0 && (activation==0.0 || price<=activation+m_trail*point) && (new_price<price-m_step*point))
            next_stop=new_price;
         else next_stop=activation+m_trail*point;
         if(next_stop>price)
            next_stop=price;
         if((deactivation==0) || (deactivation>0 && next_stop<=deactivation && next_stop>0.0))
            if(next_stop>=new_price)
               return next_stop;
      }
      else if (m_target==TRAIL_TARGET_TAKEPROFIT)
      {
         if(m_step>0 && (activation==0.0 || price<=activation) && (new_price<price-m_step*point))
            next_stop=new_price;
         else next_stop=activation-m_trail*point;
         if(next_stop>price)
            next_stop=price;
         if((deactivation==0) || (deactivation>0 && next_stop<=deactivation && next_stop>0.0))
            if(next_stop<=new_price)
               return next_stop;
      }
     }
   return 0;
  }
```

From here, we can see that the
calculation is different between trailing the stoploss and the
takeprofit. For the stoploss value, we would want it so that it goes
closer to the market with each stage of trailing. For the takeprofit
value, we would want the opposite – push it away from the current
market price at a certain distance (in points or pips) with each stage of trailing.

### CTrails (Container)

CTrail would also have a container named CTrails. Its definition is shown in the following code:

```
class CTrailsBase : public CArrayObj
  {
protected:
   bool              m_active;
   CEventAggregator *m_event_man;
   CStop            *m_stop;
public:
                     CTrailsBase(void);
                    ~CTrailsBase(void);
   virtual int       Type(void) const {return CLASS_TYPE_TRAILS;}
   //--- initialization
   virtual bool      Init(CSymbolManager*,CEventAggregator*);
   virtual CStop    *GetContainer(void);
   virtual void      SetContainer(CStop*stop);
   virtual bool      Validate(void) const;
   //--- getters and setters
   bool              Active(void) const;
   void              Active(const bool activate);
   //--- checking
   virtual double    Check(const string,const ENUM_ORDER_TYPE,const double,const double,const ENUM_TRAIL_TARGET);
  };
```

The container would have to interface between CStop, and the CTrail objects it references. It would also have its own Check method:

```
double CTrailsBase::Check(const string symbol,const ENUM_ORDER_TYPE type,const double entry_price,const double price,const ENUM_TRAIL_TARGET mode)
  {
   if(!Active())
      return 0;
   double val=0.0,ret=0.0;
   for(int i=0;i<Total();i++)
     {
      CTrail *trail=At(i);
      if(!CheckPointer(trail))
         continue;
      if(!trail.Active())
         continue;
      int trail_target=trail.TrailTarget();
      if(mode!=trail_target)
         continue;
      val=trail.Check(symbol,type,entry_price,price,mode);
      if((type==ORDER_TYPE_BUY && trail_target==TRAIL_TARGET_STOPLOSS) || (type==ORDER_TYPE_SELL && trail_target==TRAIL_TARGET_TAKEPROFIT))
      {
         if(val>ret || ret==0.0)
            ret=val;
      }
      else if((type==ORDER_TYPE_SELL && trail_target==TRAIL_TARGET_STOPLOSS) || (type==ORDER_TYPE_BUY && trail_target==TRAIL_TARGET_TAKEPROFIT))
      {
         if(val<ret || ret==0.0)
            ret=val;
      }
     }
   return ret;
  }
```

CTrails, similar to other containers
used in this article series, is a descendant of CArrayObj, which is
designed to store multiple pointers of instances of CObject and its
heirs. CTrails then, can store more than one instance of CTrail. In
the case where multiple pointers to instances of CTrail are present
in CTrails, when its Check method is called, it would call the Check
method of all the CTrail instances. However, only the value that is
closest to the current market price will be returned as the final
output. This behavior can be changed by extending CTrails.

The classes CTrail and CTrails have
been reduced to pure calculation. This means that all of the methods
are coded in the base class (CTrailBase) and not on any of its
language-specific implementation. When called to check the status of the trailingstop (by CStop), it gets the
new value of the stop, and CStop modifies the stop level of a trade
accordingly.

### Extending CTrail

The CTrail class does not only apply to
trailingstop and breakeven per se. It can be used to define almost any
evolution of a stop level of a trade over time. The process is very
similar to the way custom stops are implemented as discussed in this
article. However, the change is applied by extending the methods of CTrail in order to apply changes to the stop level after the trade has entered the market.

### Examples

Example #1: Custom Stops

Our first example for this article uses custom stoploss and takeprofit for the trades that the expert advisor will enter. We will extend the third example expert advisor from the previous article (see Cross-Platform Expert Advisor: Stops) in order to create this expert advisor. The custom stop levels will be calculated based on the high and low prices of the previous candle. As an additional safeguard, we will add a condition of a minimum stop level: if the calculated stoploss or takeprofit is closer to the entry price by less than 200 points, we will set the stoploss or takeprofit to 200 points from the entry price instead. This can help ensure that our trade request will be accepted by the broker.

First, we will declare a descendant to CStop. In this example, we will call that descendant CCustomStop. Its definition is shown below:

```
class CCustomStop : public CStop
  {
   public:
                     CCustomStop(const string);
                    ~CCustomStop(void);
   virtual double    StopLossCustom(const string,const ENUM_ORDER_TYPE,const double);
   virtual double    TakeProfitCustom(const string,const ENUM_ORDER_TYPE,const double);
  };
```

Here, we are going to extend the methods StopLossCustom and TakeProfitCustom. The following code shows the method StopLossCustom:

```
double CCustomStop::StopLossCustom(const string symbol,const ENUM_ORDER_TYPE type,const double price)
  {
   double array[1];
   double val=0;
   if(type==ORDER_TYPE_BUY)
     {
      if(CopyHigh(symbol,PERIOD_CURRENT,1,1,array))
        {
         val=array[0];
        }
     }
   else if(type==ORDER_TYPE_SELL)
     {
      if(CopyLow(symbol,PERIOD_CURRENT,1,1,array))
        {
         val=array[0];
        }
     }
   if(val>0)
     {
      double distance=MathAbs(price-val)/m_symbol.Point();
      if(distance<200)
        {
         if(type==ORDER_TYPE_BUY)
            val = price+200*m_symbol.Point();
         else if(type==ORDER_TYPE_SELL)
            val=price-200*m_symbol.Point();
        }
     }
   return val;
  }
```

First, we use the functions CopyLow and CopyHigh to calculate the stoploss depending on the type of the trade (buy or sell). After getting the initial value, we then get the absolute value of its distance from the entry price in points. If the distance is less than the minimum, we set the distance to the minimum instead.

The method TakeProfitCustom also undertakes a similar process, as shown below:

```
double CCustomStop::TakeProfitCustom(const string symbol,const ENUM_ORDER_TYPE type,const double price)
  {
   double array[1];
   double val=0;
   m_symbol=m_symbol_man.Get(symbol);
   if(!CheckPointer(m_symbol))
      return 0;
   if(type==ORDER_TYPE_BUY)
     {
      if(CopyLow(symbol,PERIOD_CURRENT,1,1,array))
        {
         val=array[0];
        }
     }
   else if(type==ORDER_TYPE_SELL)
     {
      if(CopyHigh(symbol,PERIOD_CURRENT,1,1,array))
        {
         val=array[0];
        }
     }
   if(val>0)
     {
      double distance=MathAbs(price-val)/m_symbol.Point();
      if(distance<200)
        {
         if(type==ORDER_TYPE_BUY)
            val = price-200*m_symbol.Point();
         else if(type==ORDER_TYPE_SELL)
            val=price+200*m_symbol.Point();
        }
     }
   return val;
  }
```

The class members of CStop responsible for the assigned values of stoploss and takeprofit in points (m\_stoploss and m\_takeprofit, respectively) are initialized to zero by default. Thus, we only need to put comments on the following lines within the OnInit function:

```
//main.StopLoss(stop_loss);
//main.TakeProfit(take_profit);
```

If the values (in points) of the stoploss and takeprofit of a stop level is set to zero, only then will that instance of CStop use the custom calculation.

Example #2: Breakeven

For the breakeven feature, we will also modify the main header file of the third example from the previous article (same as example #1). But this time, we are not going to extend any class objects. Rather, we will just declare an instance of CTrail as well as an instance of CTrails:

```
CTrails *trails = new CTrails();
CTrail *trail = new CTrail();
trail.Set(breakeven_value,breakeven_start,0);
trails.Add(trail);
main.Add(trails);
```

The last line is essential. We would need to add an instance of CTrails to an existing instance of CStop. This way, behavior would be applied to that particular instance of CStop.

The third argument of the Set method of CTrail is the step. Its default value is 1 (1 point). Since we are only using the breakeven feature, we give it a value of 0.

The code above requires the variables breakeven\_start, which is the activation price (in points) from the entry price, and breakeven\_value, which is the new distance of the stoploss (in points) from the activation price. We will declare input parameters for these two:

```
input int breakeven_value = 200;
input int breakeven_start = 200;
```

With this setting, as soon as the market moves at least 200 points in favor of the trade, the stoploss would be moved 200 points from the activation price. 200 - 200 = 0, and so the new calculated stoploss would be the entry price of the trade itself.

Example #3: Trailingstop

Now, let us move on to implementing a trailingstop feature in an EA instead. This example is very similar to the previous example. Recall that from the previous example, we insert the following code within the OnInit function:

```
CTrails *trails = new CTrails();
CTrail *trail = new CTrail();
trail.Set(breakeven_value,breakeven_start,0);
trails.Add(trail);
main.Add(trails);
```

The process is no different when using a trailingstop instead:

```
CTrails *trails = new CTrails();
CTrail *trail = new CTrail();
trail.Set(trail_value,trail_start,trail_step);
trails.Add(trail);
main.Add(trails);
```

This time, the Set method of CTrail uses three parameters. The first two of these is identical with the previous example. The third parameter is the step, which is the frequency of trailing after activation. In the previous example, this method parameter takes a default value of 0, which means that no additional trailing will be done after the initial activation. We then declare the input parameters trail\_value, trail\_start, and trail\_step in the EA.

Example #4: Custom Trailing

In this example, let us them implement a custom trailing feature. We will extend the expert advisor used in the first example of this article. Recall that in the expert advisor from the first example, we declared a custom stop that sets the stoploss and takeprofit based on the high and low values of the previous candle. In this example, we will expand on it, but not only set the stoploss and takeprofit with entry. We would also want the stoploss of the trade to trail the market price using the high or low price of the previous candle. As a safety measure, we will also implement a minimum stoploss for each stage of trailing.

To create this expert advisor, we first extend CTrail. We will name the class descendant as CCustomTrail. Its definition is shown below:

```
class CCustomTrail : public CTrail
  {
public:
                     CCustomTrail(void);
                    ~CCustomTrail(void);
   virtual double    Check(const string,const ENUM_ORDER_TYPE,const double,const double,const ENUM_TRAIL_TARGET);
  };
```

Now, let us extend the Check method. Note that not checking for trail target is fine in this case. We are trailing the stoploss, and this is the default trail target for CTrail:

```
double CCustomTrail::Check(const string symbol,const ENUM_ORDER_TYPE type,const double entry_price,const double price,const ENUM_TRAIL_TARGET mode)
  {
   if(!Active())
      return 0;
   if(!Refresh(symbol))
      return 0;
   double array[1];
   double val=0;
   if(type==ORDER_TYPE_BUY)
     {
      if(CopyLow(symbol,PERIOD_CURRENT,1,1,array))
        {
         val=array[0];
        }
     }
   else if(type==ORDER_TYPE_SELL)
     {
      if(CopyHigh(symbol,PERIOD_CURRENT,1,1,array))
        {
         val=array[0];
        }
     }
   if(val>0)
     {
      double distance=MathAbs(price-val)/m_symbol.Point();
      if(distance<200)
        {
         if(type==ORDER_TYPE_BUY)
            val = m_symbol.Ask()-200*m_symbol.Point();
         else if(type==ORDER_TYPE_SELL)
            val=m_symbol.Bid()+200*m_symbol.Point();
        }
     }
   if((type==ORDER_TYPE_BUY && val<=price+10*m_symbol.Point()) || (type==ORDER_TYPE_SELL && val>=price-10*m_symbol.Point()))
      val = 0;
   return val;
  }
```

As we can see, the calculation is quite similar to what we see in the methods in CCustomStop. Also, at the latter portions of the code, we add check the returned value so that the proposed new stoploss will be 10 points (1 pips) beyond the previous. This is to prevent the stoploss of the trade from floating based on the value of the recent high or low. Rather than going up or down based on the value of the high/low price, we set it so that the new stop loss level would always be beyond the value it would replace (higher for long positions, lower for short positions).

### Conclusion

In this article, we have demonstrated how custom stop levels can be achieved in a cross-platform expert advisor. Rather than setting stoploss and takeprofit levels in terms of pips and points, the methods presented introduce a way where the said levels can be represented in terms of chart price values. The article has also demonstrated a way wherein the stoploss and takeprofit levels can be changed over time.

### Programs Used in the Article

| \# | Name | Type | Description |
| --- | --- | --- | --- |
| 1. | breakeven\_ha\_ma.mqh | Header File | The main header file used in the first example |
| 2. | breakeven\_ha\_ma.mq4 | Expert Advisor | The main source file used in the MQL4 version of the first example |
| 3. | breakeven\_ha\_ma.mq5 | Expert Advisor | The main source file used in the MQL5 version of the first example |
| 4. | trail\_ha\_ma.mqh | Header File | The main header file used in the second example |
| 5. | trail\_ha\_ma.mq4 | Expert Advisor | The main source file used in the MQL4 version of the second example |
| 6. | trail\_ha\_ma.mq5 | Expert Advisor | The main source file used in the MQL5 version of the second example |
| 7. | custom\_stop\_ha\_ma.mqh | Header File | The main header file used in the third example |
| 8. | custom\_stop\_ha\_ma.mq4 | Expert Advisor | The main source file used in the MQL4 version of the third example |
| 9. | custom\_stop\_ha\_ma.mq5 | Expert Advisor | The main source file used in the MQL5 version of the third example |
| 10. | custom\_trail\_ha\_ma.mqh | Header File | The main header file used in the fourth example |
| 11. | custom\_trail\_ha\_ma.mq4 | Expert Advisor | The main source file used in the MQL4 version of the fourth example |
| 12. | custom\_trail\_ha\_ma.mq5 | Expert Advisor | The main source file used in the MQL5 version of the fourth example |

### Class Files Featured in the Article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1. | MQLx\\Base\\Stop\\StopBase.mqh | Header File | CStop (base class) |
| 2. | MQLx\\MQL4\\Stop\\Stop.mqh | Header File | CStop (MQL4 version) |
| 3. | MQLx\\MQL5\\Stop\\Stop.mqh | Header File | CStop (MQL5 version) |
| 4. | MQLx\\Base\\Trail\\TrailBase.mqh | Header File | CTrail (base class) |
| 5. | MQLx\\MQL4\\Trail\\Trail.mqh | Header File | CTrail (MQL4 version) |
| 6. | MQLx\\MQL5\\Trail\\Trail.mqh | Header File | CTrail (MQL5 version) |
| 7. | MQLx\\Base\\Trail\\TrailsBase.mqh | Header File | CTrails (base class) |
| 8. | MQLx\\MQL4\\Trail\\Trails.mqh | Header File | CTrails (MQL4 version) |
| 9. | MQLx\\MQL5\\Trail\\Trails.mqh | Header File | CTrails (MQL5 version) |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/3621.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/3621/mql5.zip "Download MQL5.zip")(741.35 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Cross-Platform Expert Advisor: The CExpertAdvisor and CExpertAdvisors Classes](https://www.mql5.com/en/articles/3622)
- [Cross-Platform Expert Advisor: Stops](https://www.mql5.com/en/articles/3620)
- [Cross-Platform Expert Advisor: Time Filters](https://www.mql5.com/en/articles/3395)
- [Cross-Platform Expert Advisor: Money Management](https://www.mql5.com/en/articles/3280)
- [Cross-Platform Expert Advisor: Signals](https://www.mql5.com/en/articles/3261)
- [Cross-Platform Expert Advisor: Order Manager](https://www.mql5.com/en/articles/2961)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/216754)**
(15)


![mbjen](https://c.mql5.com/avatar/avatar_na2.png)

**[mbjen](https://www.mql5.com/en/users/mbjen)**
\|
8 Oct 2017 at 22:39

**Enrico Lambino:**

Your first code activates breakeven, not trailingstop. If there would be any further modification of SL, it would be takeprofit. But if your TP is 500 points, the trailing would not activate at all at 500 points, since the trade has already left the market at that point.

Your second code uses trailingstop, but not breakeven. Because even before the breakeven can be applied, the SL has already moved above the breakeven price.

Hi Enrico,

I didn't get your point.

This is breakeven:

```
 //--- setting breakeven
   CTrail *trail_be=new CTrail();
   //trail_be.Set(BELevel,BEOpenPriceDist,0);
   trail_be.Set(230,250,0);
   trails.Add(trail_be);
```

This is trailing:

```
//--- setting trailing stop
   CTrail *trail=new CTrail();
   //trail.Set(trail_value,trail_start,trail_step);
   trail.Set(400,500,10);
   trails.Add(trail);
```

Breakeven activates at 250 points, trailing activates at 500 point. As you see trailing is not activated earlier than breakeven.

SL was not moved above (or below) the breakeven price. Stop loss was the same initial SL below the [open price](https://www.mql5.com/en/docs/constants/indicatorconstants/prices#enum_applied_price_enum "MQL5 documentation: Price Constants"). The trailing stop moved SL first time above the open price (and breakeven price).

Trailing works fine. Breakeven don't work (trail\_be object). If I don't use the trailing stop breakeven works fine.

![mbjen](https://c.mql5.com/avatar/avatar_na2.png)

**[mbjen](https://www.mql5.com/en/users/mbjen)**
\|
8 Oct 2017 at 23:28

The problem seems to be only for [sell order](https://www.mql5.com/en/docs/constants/tradingconstants/enum_book_type "MQL5 documentation: Trade Orders in Depth Of Market").

```
double CTrailsBase::Check(const string symbol,const ENUM_ORDER_TYPE type,const double entry_price,const double price,const ENUM_TRAIL_TARGET mode)
  {
   if(!Active())
      return 0;
   double val=0.0,ret=0.0;
   MqlDateTime time_curr;
   TimeCurrent(time_curr);
   if(time_curr.day==3 && time_curr.hour==11 && time_curr.min>=30)
      Print("");
   for(int i=0;i<Total();i++)
     {
      CTrail *trail=At(i);
      if(!CheckPointer(trail))
         continue;
      if(!trail.Active())
         continue;
      int trail_target=trail.TrailTarget();
      if(mode!=trail_target)
         continue;
      val=trail.Check(symbol,type,entry_price,price,mode);
      if((type==ORDER_TYPE_BUY && trail_target==TRAIL_TARGET_STOPLOSS) || (type==ORDER_TYPE_SELL && trail_target==TRAIL_TARGET_TAKEPROFIT))
      {
         if(val>ret || ret==0.0)
            ret=val;
      }
      else if((type==ORDER_TYPE_SELL && trail_target==TRAIL_TARGET_STOPLOSS) || (type==ORDER_TYPE_BUY && trail_target==TRAIL_TARGET_TAKEPROFIT))
      {
         if(val<ret || ret==0.0)
            ret=val;
      }
     }
   return ret;
  }
```

Problem in this place

```
 else if((type==ORDER_TYPE_SELL && trail_target==TRAIL_TARGET_STOPLOSS) || (type==ORDER_TYPE_BUY && trail_target==TRAIL_TARGET_TAKEPROFIT))
      {
         if(val<ret || ret==0.0)
            ret=val;
      }
```

I guess it should be changed to:

```
 else if((type==ORDER_TYPE_SELL && trail_target==TRAIL_TARGET_STOPLOSS) || (type==ORDER_TYPE_BUY && trail_target==TRAIL_TARGET_TAKEPROFIT))
      {
         if( (val>0 && val<ret) || ret==0.0)
            ret=val;
      }
```

![Enrico Lambino](https://c.mql5.com/avatar/2014/10/54465D5F-0757.jpg)

**[Enrico Lambino](https://www.mql5.com/en/users/iceron)**
\|
9 Oct 2017 at 12:48

Thank you for explaining the issue further, and sorry that I misunderstood earlier. Regarding this change:

```
else if((type==ORDER_TYPE_SELL && trail_target==TRAIL_TARGET_STOPLOSS) || (type==ORDER_TYPE_BUY && trail_target==TRAIL_TARGET_TAKEPROFIT))
      {
         if( (val>0 && val<ret) || ret==0.0)
            ret=val;
      }
```

Normally, I just  put the breakeven CTrail as the last index so it is evaluated last, but the code above is a more permanent solution for trailing the stoploss of sell trades.

![mbjen](https://c.mql5.com/avatar/avatar_na2.png)

**[mbjen](https://www.mql5.com/en/users/mbjen)**
\|
18 Feb 2018 at 20:21

How do I bind a specific stop to a specific signal? Due to my strategy logic every entry signal has its own stop.

Same with exit signal.

![Picee](https://c.mql5.com/avatar/avatar_na2.png)

**[Picee](https://www.mql5.com/en/users/picee)**
\|
5 May 2021 at 23:58

hello i have aproblem with the expert can you help me?

```
'SetContainer' - unexpected token, probably type is missing?	SymbolManagerBase.mqh	55	21
'SetContainer' - function already defined and has different type	SymbolManagerBase.mqh	55	21
'Deinit' - unexpected token, probably type is missing?	SymbolManagerBase.mqh	62	21
'Deinit' - function already defined and has different type	SymbolManagerBase.mqh	62	21
```

![Deep Neural Networks (Part IV). Creating, training and testing a model of neural network](https://c.mql5.com/2/48/Deep_Neural_Networks_04.png)[Deep Neural Networks (Part IV). Creating, training and testing a model of neural network](https://www.mql5.com/en/articles/3473)

This article considers new capabilities of the darch package (v.0.12.0). It contains a description of training of a deep neural networks with different data types, different structure and training sequence. Training results are included.

![Deep Neural Networks (Part III). Sample selection and dimensionality reduction](https://c.mql5.com/2/48/Deep_Neural_Networks_03.png)[Deep Neural Networks (Part III). Sample selection and dimensionality reduction](https://www.mql5.com/en/articles/3526)

This article is a continuation of the series of articles about deep neural networks. Here we will consider selecting samples (removing noise), reducing the dimensionality of input data and dividing the data set into the train/val/test sets during data preparation for training the neural network.

![TradeObjects: Automation of trading based on MetaTrader graphical objects](https://c.mql5.com/2/29/MQL5_TradeObjects__1.png)[TradeObjects: Automation of trading based on MetaTrader graphical objects](https://www.mql5.com/en/articles/3442)

The article deals with a simple approach to creating an automated trading system based on the chart linear markup and offers a ready-made Expert Advisor using the standard properties of the MetaTrader 4 and 5 objects and supporting the main trading operations.

![Using cloud storage services for data exchange between terminals](https://c.mql5.com/2/28/7l8-fbt8.png)[Using cloud storage services for data exchange between terminals](https://www.mql5.com/en/articles/3331)

Cloud technologies are becoming more popular. Nowadays, we can choose between paid and free storage services. Is it possible to use them in trading? This article proposes a technology for exchanging data between terminals using cloud storage services.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/3621&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071713020888952069)

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