---
title: Cross-Platform Expert Advisor: Money Management
url: https://www.mql5.com/en/articles/3280
categories: Integration, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:26:10.085866
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/3280&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068227491719476992)

MetaTrader 5 / Examples


### Table of Contents

1. [Introduction](https://www.mql5.com/en/articles/3280#introduction)
2. [Objectives](https://www.mql5.com/en/articles/3280#objectives)
3. [Base Class](https://www.mql5.com/en/articles/3280#base)
4. [Money Management Classes and Types](https://www.mql5.com/en/articles/3280#money)

5. [Container for Money Management Objects](https://www.mql5.com/en/articles/3280#container)
6. [Example](https://www.mql5.com/en/articles/3280#example)
7. [Conclusion](https://www.mql5.com/en/articles/3280#conclusion)


### Introduction

Money management is a common feature in expert advisors. It allows
an expert advisor to dynamically determine the lot size for the next
trade to be entered. In this article, we will introduce several money
management classes that would allow us to automate the entire process
of trade volume calculation in a cross-platform expert advisor.

### Objectives

- Understand and apply the most common money management methods
used in trading
- Allow an expert advisor to dynamically select from a list of
available money management methods
- Be compatible with MQL4 and MQL5

### Base Class

All the money management classes described in this article will
have a certain base class as its parent, named CMoney, derived from
CMoneyBase. The CMoneyBase class is defined in the following class snippet:

```
class CMoneyBase : public CObject
  {
protected:
   bool              m_active;
   double            m_volume;
   double            m_balance;
   double            m_balance_inc;
   int               m_period;
   bool              m_equity;
   string            m_name;
   CSymbolManager   *m_symbol_man;
   CSymbolInfo      *m_symbol;
   CAccountInfo     *m_account;
   CEventAggregator *m_event_man;
   CObject          *m_container;
public:
                     CMoneyBase(void);
                    ~CMoneyBase(void);
   virtual int       Type(void) const {return CLASS_TYPE_MONEY;}
   //--- initialization
   virtual bool      Init(CSymbolManager*,CAccountInfo*,CEventAggregator*);
   bool              InitAccount(CAccountInfo*);
   bool              InitSymbol(CSymbolManager*);
   CObject          *GetContainer(void);
   void              SetContainer(CObject*);
   virtual bool      Validate(void);
   //--- getters and setters
   bool              Active(void) const;
   void              Active(const bool);
   void              Equity(const bool);
   bool              Equity(void) const;
   void              LastUpdate(const datetime);
   datetime          LastUpdate(void) const;
   void              Name(const string);
   string            Name(void) const;
   double            Volume(const string,const double,const ENUM_ORDER_TYPE,const double);
   void              Volume(const double);
   double            Volume(void) const;
protected:
   virtual void      OnLotSizeUpdated(void);
   virtual bool      UpdateLotSize(const string,const double,const ENUM_ORDER_TYPE,const double);
  };
```

Most of the methods of the class are either setters or getters of
the various members of the class, and are therefore,
self-explanatory. In practical applications, that three methods that
really matter are the methods UpdateLotSize, OnLotSizeUpdated, and
Volume.

The UpdateLotSize method is where the actual calculation of the
trade volume takes place. This is also the main method that is
extended from the base class, and thus, most of the differences
between the money management classes can be found within this very
method. For the base class CMoneyBase, the method can be considered
virtual, since it does nothing except return a true value:

```
bool CMoneyBase::UpdateLotSize(const string,const double,const ENUM_ORDER_TYPE,const double)
  {
   return true;
  }
```

Sometimes, after the calculation of trade volume, it is necessary
to update certain variables used for future calculations. In such
cases, the OnLotSizeUpdated method is used. This method is
automatically called within the UpdateLotSize method. The following
code shows the said method:

```
void CMoneyBase::OnLotSizeUpdated(void)
  {
   m_symbol=m_symbol_man.Get();
   double maxvol=m_symbol.LotsMax();
   double minvol=m_symbol.LotsMin();
   if(m_volume<minvol)
      m_volume=minvol;
   if(m_volume>maxvol)
      m_volume=maxvol;
  }
```

In order to get the actual trade volume calculated by the money
management object, the expert advisor does not need to call
UpdateLotSize or OnLotSizeUpdated. Rather, the Volume method of the
class should be called. This method would automatically call the
other two methods within its code:

```
double CMoneyBase::Volume(const string symbol,const double price,const ENUM_ORDER_TYPE type,const double sl=0)
  {
   if(!Active())
      return 0;
   if(UpdateLotSize(symbol,price,type,sl))
      OnLotSizeUpdated();
   return m_volume;
  }
```

### Money Management Classes and Types

#### Fixed Lot

This is the most common method of lot sizing, and the one which
most traders are familiar with. With fixed lot sizing, all trades have
constant trade size, whether the account balance or equity is
decreasing or increasing over time.

In this type of money management, we only need a fixed amount of volume. Thus, its main difference with CMoney/CMoneyBase can be found in its constructor, where we specify a fixed lot size:

```
CMoneyFixedLotBase::CMoneyFixedLotBase(double volume)
  {
   Volume(volume);
  }
```

In case we need to dynamically change the output of this money management method, we simply alter its m\_volume class member by calling the Volume method.

#### Fixed Risk (Fixed Fractional)

The risk percent or fixed fractional method of money management
allocates a certain percentage of the account balance or equity to be
risk per trade. This is implemented in the standard library as
CmoneyFixedRisk. If a trade suffers a loss, the loss amount will be
equivalent to the percentage of the account balance at the time of
entry. This loss is not just any loss, but rather the maximum loss
that the trade can incur i.e. the market hitting the stop loss value
of the trade. This method requires a non-zero stop loss in order to
work.

Calculating the risk percentage per trade is expressed in the
following formula:

Volume = (balance \* account\_percentage / ticks) / tick\_value

where:

- balance – account balance or equity
- account\_percentage – a percentage of the account to risk (range:
0.0-1.0)
- ticks – the stop loss value, expressed in ticks

- tick\_value – the value in the deposit currency per tick movement
of the symbol or instrument (based on 1.0 lot)

A tick is defined as the smallest possible movement in price for a
given instrument or currency pair. For example, EURUSD on fractional
pip pricing (5-digit broker) has a tick size of 0.00001, which is the
smallest possible movement for the currency pair. When a stop loss
value is expressed in points or pips, the result is the difference
between the trade's entry price and its stop loss price in terms of
points or pips.

For the same currency pair, the tick value of a currency for a
4-digit broker is different from that of a 5-digit broker. This is
because for a 4-digit broker, 1 tick is equivalent to 1 point (or
pip) whereas in a 5-digit broker, a pip is equivalent to 10 points.

As an example of fixed risk money management, suppose we have a
$1,000 balance on a USD account, and the risk percentage per trade is
5%. Assuming a tick value of 0.1 and a 200-point (20 pips) stop loss
on a 5-digit broker:

Volume = (1000 \* 0.05 / 200) / 0.1 = 2.5 lot

The calculated lot size increases depending on the risk percentage
and the available balance, while decreases based on the magnitude of
the stop loss and the tick value. The account balance, risk, and tick
value are mostly constant, but the stop loss being variable is not
uncommon (dynamically calculated). With this, fixed risk is usually
not suitable for strategies where there is no upper limit in the
distance between the entry price and stop loss, since this can result
to a very small lot size (and thus rejected by the broker). On the
other hand, a too small stop loss value will result to a very large
lot size, and may also cause trouble with some brokers that have a low
max lot setting. This issue has been largely resolved in MetaTrader 5,
with orders being split into several deals if the size is too large.
However, in MetaTrader 4, there is no such functionality – the
trade size has to be prepared (split into several minor trades) to
deal with the huge trade size, or simply avoid exceeding the maximum
lot size allowed.

The formula used in calculation can be found within its UpdateLotSize method:

```
bool CMoneyFixedFractionalBase::UpdateLotSize(const string symbol,const double price,const ENUM_ORDER_TYPE type,const double sl)
  {
   m_symbol=m_symbol_man.Get(symbol);
   double last_volume=m_volume;
   if(CheckPointer(m_symbol))
     {
      double balance=m_equity==false?m_account.Balance():m_account.Equity();
      double ticks=0;
      if(price==0.0)
        {
         if(type==ORDER_TYPE_BUY)
            ticks=MathAbs(m_symbol.Bid()-sl)/m_symbol.TickSize();
         else if(type==ORDER_TYPE_SELL)
            ticks=MathAbs(m_symbol.Ask()-sl)/m_symbol.TickSize();
        }
      else ticks=MathAbs(price-sl)/m_symbol.TickSize();
      m_volume=((balance*(m_risk/100))/ticks)/m_symbol.TickValue();
     }
   return last_volume-m_volume!=0;
  }
```

First, we get the value of the stop loss in ticks. After that, we use the actual formula to update the m\_volume member of the class, which is then used as the final output.

#### Fixed Ratio

Fixed ratio money management calculates the trade size in
proportion to the current balance available on the account. This can
be considered a special case of fixed lot money management except
that in this type of money management, the lot size is adjusted
automatically, rather than manually by the trader. If the account is
increasing, the lot size would also increase after every threshold. If
the lot size is decreasing, the lot size would also adjust accordingly.

Unlike fixed risk money management, fix ratio does not require a
non-zero stop loss. This makes it ideal to use when trades do not
require a stop loss, but whose exits are managed in a different
manner (closing by profit/loss in the deposit currency, etc.).

The calculation of trade size based on fixed ratio money
management is generally expressed in the following formula:

Volume = base\_volume + (balance / balance\_increase) \*
volume\_increment

where:

- base\_volume – volume to be added to the total volume, regardless
of account size
- balance – current balance on the account
- balance\_increase – balance increase on the account to trigger an increase in the lot size
- volume\_increment – volume to be added/subtracted from the total
volume when the balance changes by balance\_increase

As an example, suppose we have a base volume of 0.0 lot, and the
volume should increase by 0.1 for every $1,000 on the account. The
account is currently worth $2,500. The total volume is therefore
calculated as follows:

Volume = 0 + (2500 / 1000) \* 0.1 = 0.25 lot

This method has many variations. One of these is the method where
the lot size is updated only at certain levels (this is the one
implemented in fixed ratio money management). For example, in the
example mentioned earlier, the calculated volume was 0.25 lot, but in
some, it may remain 0.2 lot, and would only increase to 0.3 lot once
the balance reaches or exceeds $3,000.

Its UpdateLotSize method can be implemented like the following:

```
bool CMoneyFixedRatioBase::UpdateLotSize(const string symbol,const double price,const ENUM_ORDER_TYPE type,const double sl=0)
  {
   m_symbol=m_symbol_man.Get(symbol);
   double last_volume=m_volume;
   if(CheckPointer(m_symbol))
     {
      double balance=m_equity==false?m_account.Balance():m_account.Equity();
      m_volume=m_volume_base+((int)(balance/m_balance_inc))*m_volume_inc;
      m_balance=balance;
     }
   return last_volume-m_volume!=0;
  }
```

#### Fixed Risk Per Point (Fixed Margin)

Fixed Risk per Point works in such a way that each point in
stop loss is worth a certain value in the deposit currency. The
algorithm calculates the lot size based on the desired tick value of
the trader. For example, if the account is in USD, and if the fixed
risk per point is 2.0, each point in stop loss is worth $2. If the
stop loss of the trade is 200 points, the maximum risk for the entire
trade is $400 ($400 as loss if the market hits the stop loss of the
trade).

For a typical trader, using this type of money management is
easier to handle, since the risk amount of the trade is expressed in
a value that traders are most familiar with i.e. in the deposit
currency. The trader simply needs to state the desired tick value of the
asset, and the trade volume will be calculated automatically. The
tick value, or the change in profit/loss per minimum movement in
price will remain the same, but the total risk will depend on the
magnitude of the stop loss of the trade.

Using the formula used in this method of money management, its UpdateLotSize method can be implemented in the following manner:

```
bool CMoneyFixedRiskPerPointBase::UpdateLotSize(const string symbol,const double price,const ENUM_ORDER_TYPE type,const double sl=0)
  {
   m_symbol=m_symbol_man.Get(symbol);
   double last_volume=m_volume;
   if(CheckPointer(m_symbol))
     {
      double balance=m_equity==false?m_account.Balance():m_account.Equity();
      m_volume=(m_risk/m_symbol.TickValue());
     }
   return last_volume-m_volume!=0;
  }
```

#### Fixed Risk (Fixed Margin)

The fixed risk by margin is the equivalent of the [CMoneyFixedMargin](https://www.mql5.com/en/docs/standardlibrary/expertclasses/samplemmclasses/cmoneyfixedmargin) class from the MQL5 Standard Library. This is
actually a special case of fixed risk per point method of money
management. However, unlike Fixed Risk Per Point, this method
considers the entire stop loss value in the calculation of the trade
volume such that no matter the size of the stop loss is, the risk
remains the same. In the previous example, We had 200 points in
stop loss and $400 as the maximum risk. If the stop loss were reduced
to 100 points, the maximum risk for the trade in fixed risk per point
would also be halved ($200), whereas in fixed margin money
management, the maximum risk would remain constant ($400).

Given this formula, we can implement the UpdateLotSize method as follows:

```
bool CMoneyFixedRiskBase::UpdateLotSize(const string symbol,const double price,const ENUM_ORDER_TYPE type,const double sl)
  {
   m_symbol=m_symbol_man.Get(symbol);
   double last_volume=m_volume;
   if(CheckPointer(m_symbol))
     {
      double balance=m_equity==false?m_account.Balance():m_account.Equity();
      double ticks=0;
      if(price==0.0)
        {
         if(type==ORDER_TYPE_BUY)
            ticks=MathAbs(m_symbol.Bid()-sl)/m_symbol.TickSize();
         else if(type==ORDER_TYPE_SELL)
            ticks=MathAbs(m_symbol.Ask()-sl)/m_symbol.TickSize();
        }
      else ticks=MathAbs(price-sl)/m_symbol.TickSize();
      m_volume=(m_risk/m_symbol.TickValue())/ticks;
     }
   return last_volume-m_volume!=0;
  }
```

The formula used here is quite similar to fixed risk per point, except that we need to get the tick value of the stop loss, and then divide the output of the previous formula with this value.

### Container for Money Management Objects

Similar to the signal classes discussed in an earlier article, our
money management objects would also have a container. This would
allow an expert advisor to dynamically select from a list of
available money management objects loaded into the platform. Ideally, this container will act as a mediator between the money management classes and the rest of the code of the expert advisor. The base
class for this object is CMoneysBase, whose definition is shown
below:

```
class CMoneysBase : public CArrayObj
  {
protected:
   bool              m_active;
   int               m_selected;
   CEventAggregator *m_event_man;
   CObject          *m_container;
public:
                     CMoneysBase(void);
                    ~CMoneysBase(void);
   virtual int       Type(void) const {return CLASS_TYPE_MONEYS;}
   //--- initialization
   virtual bool      Init(CSymbolManager*,CAccountInfo*,CEventAggregator*);
   CObject          *GetContainer(void);
   void              SetContainer(CObject*);
   virtual bool      Validate(void) const;
   //--- setters and getters
   virtual bool      Active(void) const;
   virtual void      Active(const bool);
   virtual int       Selected(void) const;
   virtual void      Selected(const int);
   virtual bool      Selected(const string);
   //--- volume calculation
   virtual double    Volume(const string,const double,const ENUM_ORDER_TYPE,const double);
  };
```

Since this object is designed to contain multiple money management
objects, it requires at least two methods in order to make it usable
in an expert advisor:

1. Selection, or the capacity to dynamically switch between money
    management methods
2. Use the selected money management object and get its calculated
    trade volume

The selection is done in two ways: through the assignment of the
index of the money management object in the object array (CMoneysBase
extends [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj)), or though finding the object to be selected
through its name (Name method of CMoneyBase/CMoney). The following
shows the overloaded Selected method which accepts an integer
argument (or index):

```
CMoneysBase::Selected(const int value)
  {
   m_selected=value;
  }
```

The following shows the overloaded
Selected method which accepts a string argument (name of money
management object). Note that this requires a non-empty name for the money management object, which can be assigned through its Name method.

```
bool CMoneysBase::Selected(const string select)
  {
   for(int i=0;i<Total();i++)
     {
      CMoney *money=At(i);
      if(!CheckPointer(money))
         continue;
      if(StringCompare(money.Name(),select))
        {
         Selected(i);
         return true;
        }
     }
   return false;
  }
```

The third overloaded method is a method without any arguments. It
simply returns the index of the selected money management object, which is only useful when the expert advisor wants to know which money management method is currently selected:

```
int CMoneysBase::Selected(void) const
  {
   return m_selected;
  }
```

The actual volume is calculated through this object by its Volume
method. The method first gets the pointer to the selected money
management object, and then calling its own Volume method. The code
for the Volume method for CMoneysBase is shown below:

```
double CMoneysBase::Volume(const string symbol,const double price,const ENUM_ORDER_TYPE type,const double sl=0)
  {
   CMoney *money=At(m_selected);
   if(CheckPointer(money))
      return money.Volume(symbol,price,type,sl);
   return 0;
  }
```

Here, the method accesses the object from the object array and
stores it in a pointer. In order to avoid errors, one must make sure
that the actual element referred to by the index actually exists
within the object array.

### Example

As an example, we will use the last example from the [previous\\
article](https://www.mql5.com/en/articles/3261). We will modify
it in such a way that we introduce the money management classes
introduced in this article, place them in a single container, and
then add them to the order manager. Most of the additions will only
deal with the OnInit function of the EA, which is shown below:

```
int OnInit()
  {
//---
   order_manager=new COrderManager();
   money_manager = new CMoneys();
   CMoney *money_fixed= new CMoneyFixedLot(0.05);
   //CMoney *money_ff= new CMoneyFixedFractional(5);
   CMoney *money_ratio= new CMoneyFixedRatio(0,0.1,1000);
   //CMoney *money_riskperpoint= new CMoneyFixedRiskPerPoint(0.1);
   //CMoney *money_risk= new CMoneyFixedRisk(100);

   money_manager.Add(money_fixed);
   //money_manager.Add(money_ff);
   money_manager.Add(money_ratio);
   //money_manager.Add(money_riskperpoint);
   //money_manager.Add(money_risk);
   order_manager.AddMoneys(money_manager);
   //order_manager.Account(money_manager);
   symbol_manager=new CSymbolManager();
   symbol_info=new CSymbolInfo();
   if(!symbol_info.Name(Symbol()))
      Print("symbol not set");
   symbol_manager.Add(GetPointer(symbol_info));
   order_manager.Init(symbol_manager,new CAccountInfo());

   MqlParam params[1];
   params[0].type=TYPE_STRING;
#ifdef __MQL5__
   params[0].string_value="Examples\\Heiken_Ashi";
#else
   params[0].string_value="Heiken Ashi";
#endif
   SignalHA *signal_ha=new SignalHA(Symbol(),0,1,params,signal_bar);
   SignalMA *signal_ma=new SignalMA(Symbol(),(ENUM_TIMEFRAMES) Period(),maperiod,0,mamethod,maapplied,signal_bar);
   signals=new CSignals();
   signals.Add(GetPointer(signal_ha));
   signals.Add(GetPointer(signal_ma));
   signals.Init(GetPointer(symbol_manager),NULL);
//---
   return(INIT_SUCCEEDED);
  }
```

Here, we included the lines of code used for using fixed
fractional, fixed risk, and fixed risk per point money management
methods. However, since these methods require a non-zero stop loss,
and our EA at this point only enters trades with zero stops, we will
refrain from using these methods from now. For the meantime, we will only
use fixed lot and fixed ratio money management methods. In the event
that these objects return an invalid stop loss (less than zero),
however, the default lot size of the order manager will be used
(default of 0.1 lot, available in the m\_lotsize class member of
CorderManager/COrderManagerBase) .

COrderManager has its own class member which is a pointer to a money management container (CMoney). So, using COrderManager will also result to the money management header files to be included in the source. If an expert will not use COrderManager, then an [#include](https://www.mql5.com/en/docs/basis/preprosessor/include) directive to the money management classes will need to be indicated on the source code.

For the OnTick function, we modify the EA in such a way that for
long positions, the EA will use fixed lot sizing, while for short
positions, it will use fixed ratio lot sizing. This can be achieved
altering the selected money management type before the TradeOpen method of the order manager is called, through the method
Selected of CMoneys:

```
void OnTick()
  {
//---
   if(symbol_info.RefreshRates())
     {
      signals.Check();
      if(signals.CheckOpenLong())
        {
         close_last();
         //Print("Entering buy trade..");
         money_manager.Selected(0);
         order_manager.TradeOpen(Symbol(),ORDER_TYPE_BUY,symbol_info.Ask());
        }
      else if(signals.CheckOpenShort())
        {
         close_last();
         //Print("Entering sell trade..");
         money_manager.Selected(1);
         order_manager.TradeOpen(Symbol(),ORDER_TYPE_SELL,symbol_info.Bid());
        }
     }
  }
```

Since money management, in essence, is only pure calculation, we
expect the calculated lotsize to be the same in MetaTrader 4 and
MetaTrader 5. The following shows a test result of the EA in
MetaTrader 4 (first 10 trades):

|     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| # | Time | Type | Order | Size | Price | S / L | T / P | Profit | Balance |
| 1 | 2017.01.02 00:00 | sell | 1 | 1.00 | 1.05100 | 0.00000 | 0.00000 |  |
| 2 | 2017.01.03 03:00 | close | 1 | 1.00 | 1.04679 | 0.00000 | 0.00000 | 419.96 | 10419.96 |
| 3 | 2017.01.03 03:00 | buy | 2 | 0.05 | 1.04679 | 0.00000 | 0.00000 |  |
| 4 | 2017.01.03 10:00 | close | 2 | 0.05 | 1.04597 | 0.00000 | 0.00000 | -4.10 | 10415.86 |
| 5 | 2017.01.03 10:00 | sell | 3 | 1.00 | 1.04597 | 0.00000 | 0.00000 |  |
| 6 | 2017.01.03 20:00 | close | 3 | 1.00 | 1.04285 | 0.00000 | 0.00000 | 312.00 | 10727.86 |
| 7 | 2017.01.03 20:00 | buy | 4 | 0.05 | 1.04285 | 0.00000 | 0.00000 |  |
| 8 | 2017.01.03 22:00 | close | 4 | 0.05 | 1.04102 | 0.00000 | 0.00000 | -9.15 | 10718.71 |
| 9 | 2017.01.03 22:00 | sell | 5 | 1.00 | 1.04102 | 0.00000 | 0.00000 |  |
| 10 | 2017.01.04 02:00 | close | 5 | 1.00 | 1.04190 | 0.00000 | 0.00000 | -89.04 | 10629.67 |
| 11 | 2017.01.04 02:00 | buy | 6 | 0.05 | 1.04190 | 0.00000 | 0.00000 |  |
| 12 | 2017.01.04 03:00 | close | 6 | 0.05 | 1.03942 | 0.00000 | 0.00000 | -12.40 | 10617.27 |
| 13 | 2017.01.04 03:00 | sell | 7 | 1.00 | 1.03942 | 0.00000 | 0.00000 |  |
| 14 | 2017.01.04 06:00 | close | 7 | 1.00 | 1.04069 | 0.00000 | 0.00000 | -127.00 | 10490.27 |
| 15 | 2017.01.04 06:00 | buy | 8 | 0.05 | 1.04069 | 0.00000 | 0.00000 |  |
| 16 | 2017.01.05 11:00 | close | 8 | 0.05 | 1.05149 | 0.00000 | 0.00000 | 54.05 | 10544.32 |
| 17 | 2017.01.05 11:00 | sell | 9 | 1.00 | 1.05149 | 0.00000 | 0.00000 |  |
| 18 | 2017.01.05 16:00 | close | 9 | 1.00 | 1.05319 | 0.00000 | 0.00000 | -170.00 | 10374.32 |
| 19 | 2017.01.05 16:00 | buy | 10 | 0.05 | 1.05319 | 0.00000 | 0.00000 |  |
| 20 | 2017.01.06 05:00 | close | 10 | 0.05 | 1.05869 | 0.00000 | 0.00000 | 27.52 | 10401.84 |

In MetaTrader 5, we can see the following results (hedging mode, first 10 trades):

|  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Orders** |
| **Open Time** | **Order** | **Symbol** | **Type** | **Volume** | **Price** | **S / L** | **T / P** | **Time** | **State** | **Comment** |
| 2017.01.02 00:00:00 | 2 | EURUSD | sell | 1.00 / 1.00 | 1.05100 |  |  | 2017.01.02 00:00:00 | filled |  |
| 2017.01.03 03:00:00 | 3 | EURUSD | buy | 1.00 / 1.00 | 1.04669 |  |  | 2017.01.03 03:00:00 | filled |  |
| 2017.01.03 03:00:00 | 4 | EURUSD | buy | 0.05 / 0.05 | 1.04669 |  |  | 2017.01.03 03:00:00 | filled |  |
| 2017.01.03 10:00:00 | 5 | EURUSD | sell | 0.05 / 0.05 | 1.04597 |  |  | 2017.01.03 10:00:00 | filled |  |
| 2017.01.03 10:00:00 | 6 | EURUSD | sell | 1.00 / 1.00 | 1.04597 |  |  | 2017.01.03 10:00:00 | filled |  |
| 2017.01.03 20:00:00 | 7 | EURUSD | buy | 1.00 / 1.00 | 1.04273 |  |  | 2017.01.03 20:00:00 | filled |  |
| 2017.01.03 20:00:00 | 8 | EURUSD | buy | 0.05 / 0.05 | 1.04273 |  |  | 2017.01.03 20:00:00 | filled |  |
| 2017.01.03 22:00:00 | 9 | EURUSD | sell | 0.05 / 0.05 | 1.04102 |  |  | 2017.01.03 22:00:00 | filled |  |
| 2017.01.03 22:00:00 | 10 | EURUSD | sell | 1.00 / 1.00 | 1.04102 |  |  | 2017.01.03 22:00:00 | filled |  |
| 2017.01.04 02:00:00 | 11 | EURUSD | buy | 1.00 / 1.00 | 1.04180 |  |  | 2017.01.04 02:00:00 | filled |  |
| 2017.01.04 02:00:00 | 12 | EURUSD | buy | 0.05 / 0.05 | 1.04180 |  |  | 2017.01.04 02:00:00 | filled |  |
| 2017.01.04 03:00:00 | 13 | EURUSD | sell | 0.05 / 0.05 | 1.03942 |  |  | 2017.01.04 03:00:00 | filled |  |
| 2017.01.04 03:00:00 | 14 | EURUSD | sell | 1.00 / 1.00 | 1.03942 |  |  | 2017.01.04 03:00:00 | filled |  |
| 2017.01.04 06:00:00 | 15 | EURUSD | buy | 1.00 / 1.00 | 1.04058 |  |  | 2017.01.04 06:00:00 | filled |  |
| 2017.01.04 06:00:00 | 16 | EURUSD | buy | 0.05 / 0.05 | 1.04058 |  |  | 2017.01.04 06:00:00 | filled |  |
| 2017.01.05 11:00:00 | 17 | EURUSD | sell | 0.05 / 0.05 | 1.05149 |  |  | 2017.01.05 11:00:00 | filled |  |
| 2017.01.05 11:00:00 | 18 | EURUSD | sell | 1.00 / 1.00 | 1.05149 |  |  | 2017.01.05 11:00:00 | filled |  |
| 2017.01.05 16:00:00 | 19 | EURUSD | buy | 1.00 / 1.00 | 1.05307 |  |  | 2017.01.05 16:00:00 | filled |  |
| 2017.01.05 16:00:00 | 20 | EURUSD | buy | 0.05 / 0.05 | 1.05307 |  |  | 2017.01.05 16:00:00 | filled |  |
| 2017.01.06 05:00:00 | 21 | EURUSD | sell | 0.05 / 0.05 | 1.05869 |  |  | 2017.01.06 05:00:00 | filled |  |

Since the order manager already takes care of the difference
between the two platforms (and languages), the method and result of
lot size calculation would be the same, and any differences that may
arise will be up to the order manager itself.

### Conclusion

This article shows how money management can be applied in a
cross-platform expert advisor. It introduces 5 different money
management methods. It also features a custom container object for
the pointers to these objects, which is used for dynamic money
management method selection.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/3280.zip "Download all attachments in the single ZIP archive")

[mm\_ha\_ma.zip](https://www.mql5.com/en/articles/download/3280/mm_ha_ma.zip "Download mm_ha_ma.zip")(1038.69 KB)

[tester\_results.zip](https://www.mql5.com/en/articles/download/3280/tester_results.zip "Download tester_results.zip")(292.36 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Cross-Platform Expert Advisor: The CExpertAdvisor and CExpertAdvisors Classes](https://www.mql5.com/en/articles/3622)
- [Cross-Platform Expert Advisor: Custom Stops, Breakeven and Trailing](https://www.mql5.com/en/articles/3621)
- [Cross-Platform Expert Advisor: Stops](https://www.mql5.com/en/articles/3620)
- [Cross-Platform Expert Advisor: Time Filters](https://www.mql5.com/en/articles/3395)
- [Cross-Platform Expert Advisor: Signals](https://www.mql5.com/en/articles/3261)
- [Cross-Platform Expert Advisor: Order Manager](https://www.mql5.com/en/articles/2961)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/209332)**
(6)


![Shephard Mukachi](https://c.mql5.com/avatar/2017/5/5920C545-5DFD.jpg)

**[Shephard Mukachi](https://www.mql5.com/en/users/mukachi)**
\|
13 Jul 2017 at 19:48

Hi Enrico. I agree with Roberto, it is excellent work. Thanks a million.

![mbjen](https://c.mql5.com/avatar/avatar_na2.png)

**[mbjen](https://www.mql5.com/en/users/mbjen)**
\|
19 Sep 2017 at 12:24

Hi,

No simple MM method which calculates lot in % from balance or equity irregarding of SL size?

As for this MM type

```
Volume = base_volume + (balance / balance_increase) * volume_increment
```

Does it decrease the lot size when balance becomes less?

Also, would be great to have some MM types depending on previous trades results (losses or profits). Anyway, it can be easily coded basing on existing classes. Thanks.

![mbjen](https://c.mql5.com/avatar/avatar_na2.png)

**[mbjen](https://www.mql5.com/en/users/mbjen)**
\|
2 Nov 2017 at 00:14

Hello Enrico,

How to implement MM based on number of losing/profitable trades? Martingale and so on? How to calculate previous trade (COrder instance) profit?

![Simalb](https://c.mql5.com/avatar/2017/11/59FF5B41-9224.png)

**[Simalb](https://www.mql5.com/en/users/simalb)**
\|
7 Nov 2017 at 17:19

![Tafadzwa Nyamwanza](https://c.mql5.com/avatar/2018/3/5AA18E7B-85B9.jpg)

**[Tafadzwa Nyamwanza](https://www.mql5.com/en/users/tnyamwanza)**
\|
7 Mar 2018 at 13:24

**MetaQuotes Software Corp.:**

New article [Cross-Platform Expert Advisor: Money Management](https://www.mql5.com/en/articles/3280) has been published:

Author: [Enrico Lambino](https://www.mql5.com/en/users/Iceron "Iceron")

great article and explanation of the money management classes. i had stop out issues due to incorrect money management settings. article greatly improved my trading results.


![Developing custom indicators using CCanvas class](https://c.mql5.com/2/28/MQL5-avatar-CCanvasIndicator-001.png)[Developing custom indicators using CCanvas class](https://www.mql5.com/en/articles/3236)

The article deals with developing custom graphical indicators using graphical primitives of the CCanvas class.

![Forecasting market movements using the Bayesian classification and indicators based on Singular Spectrum Analysis](https://c.mql5.com/2/27/MQL5-avatar-SSAtrend-001__1.png)[Forecasting market movements using the Bayesian classification and indicators based on Singular Spectrum Analysis](https://www.mql5.com/en/articles/3172)

The article considers the ideology and methodology of building a recommendatory system for time-efficient trading by combining the capabilities of forecasting with the singular spectrum analysis (SSA) and important machine learning method on the basis of Bayes' Theorem.

![Cross-Platform Expert Advisor: Time Filters](https://c.mql5.com/2/28/Cross_Platform_Expert_Advisor__3.png)[Cross-Platform Expert Advisor: Time Filters](https://www.mql5.com/en/articles/3395)

This article discusses the implementation of various methods of time filtering a cross-platform expert advisor. The time filter classes are responsible for checking whether or not a given time falls under a certain time configuration setting.

![DiNapoli trading system](https://c.mql5.com/2/26/8ahkxppg.png)[DiNapoli trading system](https://www.mql5.com/en/articles/3061)

The article describes the Fibo levels-based trading system developed by Joe DiNapoli. The idea behind the system and the main concepts are explained, as well as a simple indicator is provided as an example for more clarity.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/3280&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068227491719476992)

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