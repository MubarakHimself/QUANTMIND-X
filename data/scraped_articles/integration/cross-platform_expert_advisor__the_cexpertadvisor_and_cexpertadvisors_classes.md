---
title: Cross-Platform Expert Advisor: The CExpertAdvisor and CExpertAdvisors Classes
url: https://www.mql5.com/en/articles/3622
categories: Integration, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:16:29.568165
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/3622&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071710727376415995)

MetaTrader 5 / Examples


### Table of Contents

01. [Introduction](https://www.mql5.com/en/articles/3622#introduction)
02. [The Expert Advisor Class](https://www.mql5.com/en/articles/3622#expert_advisor)

03. [Initialization](https://www.mql5.com/en/articles/3622#initialization)
04. [New Bar Detection](https://www.mql5.com/en/articles/3622#new_bar)
05. [OnTick Handler](https://www.mql5.com/en/articles/3622#ontick)
06. [Expert Advisors Container](https://www.mql5.com/en/articles/3622#expert_advisors)
07. [Persistence of Data](https://www.mql5.com/en/articles/3622#persistence)
08. [Examples](https://www.mql5.com/en/articles/3622#examples)
09. [Final Remarks](https://www.mql5.com/en/articles/3622#final)
10. [Conclusion](https://www.mql5.com/en/articles/3622#conclusion)

### Introduction

In earlier articles regarding this topic, the expert advisor examples featured have their components scattered all over the expert advisor main header file through the use of custom-defined functions. This article features the classes CExpertAdvisor and CExpertsAdvisors, which aim at the creating a more harmonious interaction between the various components of a cross-platform expert advisor. It also addresses some common problems usually encountered in expert advisors, such as loading and saving volatile data, and new bar detection.

### The Expert Advisor Class

The CExpertAdvisorBase class is shown
in the following code snippet. At this point, most of the differences between MQL4 and MQL5 are handled by the other class objects that were discussed in previous articles.

```
class CExpertAdvisorBase : public CObject
  {
protected:
   //--- trade parameters
   bool              m_active;
   string            m_name;
   int               m_distance;
   double            m_distance_factor_long;
   double            m_distance_factor_short;
   bool              m_on_tick_process;
   //--- signal parameters
   bool              m_every_tick;
   bool              m_one_trade_per_candle;
   datetime          m_last_trade_time;
   string            m_symbol_name;
   int               m_period;
   bool              m_position_reverse;
   //--- signal objects
   CSignals         *m_signals;
   //--- trade objects
   CAccountInfo      m_account;
   CSymbolManager    m_symbol_man;
   COrderManager     m_order_man;
   //--- trading time objects
   CTimes           *m_times;
   //--- candle
   CCandleManager    m_candle_man;
   //--- events
   CEventAggregator *m_event_man;
   //--- container
   CObject          *m_container;
public:
                     CExpertAdvisorBase(void);
                    ~CExpertAdvisorBase(void);
   virtual int       Type(void) const {return CLASS_TYPE_EXPERT;}
   //--- initialization
   bool              AddEventAggregator(CEventAggregator*);
   bool              AddMoneys(CMoneys*);
   bool              AddSignal(CSignals*);
   bool              AddStops(CStops*);
   bool              AddSymbol(const string);
   bool              AddTimes(CTimes*);
   virtual bool      Init(const string,const int,const int,const bool,const bool,const bool);
   virtual bool      InitAccount(void);
   virtual bool      InitCandleManager(void);
   virtual bool      InitEventAggregator(void);
   virtual bool      InitComponents(void);
   virtual bool      InitSignals(void);
   virtual bool      InitTimes(void);
   virtual bool      InitOrderManager(void);
   virtual bool      Validate(void) const;
   //--- container
   void              SetContainer(CObject*);
   CObject          *GetContainer(void);
   //--- activation and deactivation
   bool              Active(void) const;
   void              Active(const bool);
   //--- setters and getters
   string            Name(void) const;
   void              Name(const string);
   int               Distance(void) const;
   void              Distance(const int);
   double            DistanceFactorLong(void) const;
   void              DistanceFactorLong(const double);
   double            DistanceFactorShort(void) const;
   void              DistanceFactorShort(const double);
   string            SymbolName(void) const;
   void              SymbolName(const string);
   //--- object pointers
   CAccountInfo     *AccountInfo(void);
   CStop            *MainStop(void);
   CMoneys          *Moneys(void);
   COrders          *Orders(void);
   COrders          *OrdersHistory(void);
   CStops           *Stops(void);
   CSignals         *Signals(void);
   CTimes           *Times(void);
   //--- order manager
   string            Comment(void) const;
   void              Comment(const string);
   bool              EnableTrade(void) const;
   void              EnableTrade(bool);
   bool              EnableLong(void) const;
   void              EnableLong(bool);
   bool              EnableShort(void) const;
   void              EnableShort(bool);
   int               Expiration(void) const;
   void              Expiration(const int);
   double            LotSize(void) const;
   void              LotSize(const double);
   int               MaxOrdersHistory(void) const;
   void              MaxOrdersHistory(const int);
   int               Magic(void) const;
   void              Magic(const int);
   uint              MaxTrades(void) const;
   void              MaxTrades(const int);
   int               MaxOrders(void) const;
   void              MaxOrders(const int);
   int               OrdersTotal(void) const;
   int               OrdersHistoryTotal(void) const;
   int               TradesTotal(void) const;
   //--- signal manager
   int               Period(void) const;
   void              Period(const int);
   bool              EveryTick(void) const;
   void              EveryTick(const bool);
   bool              OneTradePerCandle(void) const;
   void              OneTradePerCandle(const bool);
   bool              PositionReverse(void) const;
   void              PositionReverse(const bool);
   //--- additional candles
   void              AddCandle(const string,const int);
   //--- new bar detection
   void              DetectNewBars(void);
   //-- events
   virtual bool      OnTick(void);
   virtual void      OnChartEvent(const int,const long&,const double&,const string&);
   virtual void      OnTimer(void);
   virtual void      OnTrade(void);
   virtual void      OnDeinit(const int,const int);
   //--- recovery
   virtual bool      Save(const int);
   virtual bool      Load(const int);

protected:
   //--- candle manager
   virtual bool      IsNewBar(const string,const int);
   //--- order manager
   virtual void      ManageOrders(void);
   virtual void      ManageOrdersHistory(void);
   virtual void      OnTradeTransaction(COrder*) {}
   virtual datetime  Time(const int);
   virtual bool      TradeOpen(const string,const ENUM_ORDER_TYPE,double,bool);
   //--- symbol manager
   virtual bool      RefreshRates(void);
   //--- deinitialization
   void              DeinitAccount(void);
   void              DeinitCandle(void);
   void              DeinitSignals(void);
   void              DeinitSymbol(void);
   void              DeinitTimes(void);
  };
```

Most of the class methods declared within this class serve as wrappers to methods of its components. The key methods in this class will be discussed in later sections.

### Initialization

During the initialization phase of the expert advisor, our primary goal is to instantiate the objects needed by the trading strategy (e.g. money management, signals, etc.) and then integrate them with the instance of CExpertAdvisor, which would also need to be created during [OnInit](https://www.mql5.com/en/docs/basis/function/events). With this goal, when any of the event functions is triggered within the expert advisor, all we
need to supply is a single line of code calling the appropriate handler or method of the CExpertAdvisor instance. This is very similar to the way the MQL5 Standard Library CExpert is used.

After the creation of an instance of CExpertAdvisor, the next method to call is its Init method. The code of the said method is shown below:

```
bool CExpertAdvisorBase::Init(string symbol,int period,int magic,bool every_tick=true,bool one_trade_per_candle=true,bool position_reverse=true)
  {
   m_symbol_name=symbol;
   CSymbolInfo *instrument;
   if((instrument=new CSymbolInfo)==NULL)
      return false;
   if(symbol==NULL) symbol=Symbol();
   if(!instrument.Name(symbol))
      return false;
   instrument.Refresh();
   m_symbol_man.Add(instrument);
   m_symbol_man.SetPrimary(m_symbol_name);
   m_period=(ENUM_TIMEFRAMES)period;
   m_every_tick=every_tick;
   m_order_man.Magic(magic);
   m_position_reverse=position_reverse;
   m_one_trade_per_candle=one_trade_per_candle;
   CCandle *candle=new CCandle();
   candle.Init(instrument,m_period);
   m_candle_man.Add(candle);
   Magic(magic);
   return false;
  }
```

Here, we create the instances of most components that are often found in trading strategies. This includes the symbol or instrument to use (which has to be translated to a type of object), and the default period or timeframe. It also contains the rule on whether or not it should operate its core tasks at every new tick or at the first tick of the each candle only, whether or not it should limit a maximum of one trade per candle only (to prevent multiple entries on the same candle), and whether it should reverse its position at opposite signal (close existing trades and re-enter based on the new signal).

At the end of the OnInit function, the instance of CExpertAdvisor would have to make a call to its InitComponents method. The following code shows the the said method of CExpertBase:

```
bool CExpertAdvisorBase::InitComponents(void)
  {
   if(!InitSignals())
     {
      Print(__FUNCTION__+": error in signal initialization");
      return false;
     }
   if(!InitTimes())
     {
      Print(__FUNCTION__+": error in time initialization");
      return false;
     }
   if(!InitOrderManager())
     {
      Print(__FUNCTION__+": error in order manager initialization");
      return false;
     }
   if(!InitCandleManager())
     {
      Print(__FUNCTION__+": error in candle manager initialization");
      return false;
     }
   if(!InitEventAggregator())
     {
      Print(__FUNCTION__+": error in event aggregator initialization");
      return false;
     }
   return true;
  }
```

In this method, the Init method of each of the components of the expert advisor instance is called. It is also through this method where the Validate methods of each component is called to see if their settings would pass validation.

### New Bar Detection

Some trading strategies require only to operate at the first tick of a new candle. There are many ways to implement this feature. One of this is the comparison of the open time and open price of the current candle to their previous states, which is the method implemented in the CCandle class. The following code shows the declaration for CCandleBase, from which CCandle is based:

```
class CCandleBase : public CObject
  {
protected:
   bool              m_new;
   bool              m_wait_for_new;
   bool              m_trade_processed;
   int               m_period;
   bool              m_active;
   MqlRates          m_last;
   CSymbolInfo      *m_symbol;
   CEventAggregator *m_event_man;
   CObject          *m_container;
public:
                     CCandleBase(void);
                    ~CCandleBase(void);
   virtual int       Type(void) const {return(CLASS_TYPE_CANDLE);}
   virtual bool      Init(CSymbolInfo*,const int);
   virtual bool      Init(CEventAggregator*);
   CObject          *GetContainer(void);
   void              SetContainer(CObject*);
   //--- setters and getters
   void              Active(bool);
   bool              Active(void) const;
   datetime          LastTime(void) const;
   double            LastOpen(void) const;
   double            LastHigh(void) const;
   double            LastLow(void) const;
   double            LastClose(void) const;
   string            SymbolName(void) const;
   int               Timeframe(void) const;
   void              WaitForNew(bool);
   bool              WaitForNew(void) const;
   //--- processing
   virtual bool      TradeProcessed(void) const;
   virtual void      TradeProcessed(bool);
   virtual void      Check(void);
   virtual void      IsNewCandle(bool);
   virtual bool      IsNewCandle(void) const;
   virtual bool      Compare(MqlRates &) const;
   //--- recovery
   virtual bool      Save(const int);
   virtual bool      Load(const int);
  };
```

The checking of the presence of a new candle on the chart is done through its Check method, which is shown below:

```
CCandleBase::Check(void)
  {
   if(!Active())
      return;
   IsNewCandle(false);
   MqlRates rates[];
   if(CopyRates(m_symbol.Name(),(ENUM_TIMEFRAMES)m_period,1,1,rates)==-1)
      return;
   if(Compare(rates[0]))
     {
      IsNewCandle(true);
      TradeProcessed(false);
      m_last=rates[0];
     }
  }
```

If checking for a new bar, the expert advisor instance should always call this method every tick. The coder is then free to extend CCxpertAdvisor so that it can perform additional tasks when a new candle appears on the chart.

As shown in the code above, the actual comparison of the open time and open price of the bar is done through the Compare method of the class, which is shown in the following code:

```
bool CCandleBase::Compare(MqlRates &rates) const
  {
   return (m_last.time!=rates.time ||
           (m_last.open/m_symbol.TickSize())!=(rates.open/m_symbol.TickSize()) ||
           (!m_wait_for_new && m_last.time==0));
  }
```

This method of checking for the existence of a new bar depends on three conditions. Satisfying at least one will guarantee a result of true, which indicates the presence of a new candle on the chart:

1. The last recorded open time is not equal to the open time of the current bar
2. The last recorded open price is not equal to the open price of the current bar
3. The last recorded open time is zero and a new bar does not have to be the first tick for that bar

The first two conditions involves the direct comparison of the rates of the current bar to the previous recorded state. The third condition only applies to the very first tick that the expert advisor will encounter. As soon as an expert advisor is loaded on a chart, it does not have yet any previous record of the rates (open time and open price), and so the last recorded open time would be zero. Some traders consider this bar as a new bar for their expert advisors, while others prefer to have the expert advisor wait for an actual new bar to appear on the chart after the initialization of the expert advisor.

Similar to other types of classes discussed previously, the class CCandle would also have its container, CCandleManager. The following code shows the declaration of CCandleManagerBase:

```
class CCandleManagerBase : public CArrayObj
  {
protected:
   bool              m_active;
   CSymbolManager   *m_symbol_man;
   CEventAggregator *m_event_man;
   CObject          *m_container;
public:
                     CCandleManagerBase(void);
                    ~CCandleManagerBase(void);
   virtual int       Type(void) const {return(CLASS_TYPE_CANDLE_MANAGER);}
   virtual bool      Init(CSymbolManager*,CEventAggregator*);
   virtual bool      Add(const string,const int);
   CObject          *GetContainer(void);
   void              SetContainer(CObject *container);
   bool              Active(void) const;
   void              Active(bool active);
   virtual void      Check(void) const;
   virtual bool      IsNewCandle(const string,const int) const;
   virtual CCandle *Get(const string,const int) const;
   virtual bool      TradeProcessed(const string,const int) const;
   virtual void      TradeProcessed(const string,const int,const bool) const;
   //--- recovery
   virtual bool      Save(const int);
   virtual bool      Load(const int);
  };
```

An instance of CCandle is created based on the name of the instrument and the timeframe. Having CCandleManager would make it easier for an expert advisor to track multiple charts for a given instrument, for example, having the capacity to check the occurrence of a new candle on EURUSD M15 and EURUSD H1 in the same expert advisor. Instances of CCandle that have the same symbol and timeframe are redundant and should be avoided. When looking for a certain instance of CCandle, one should simply call the appropriate method found on CCandleManager and specify the symbol and timeframe. CCandleManager, in turn, would look for the appropriate CCandle instance and call the intended method.

Aside from checking the occurrence of a new candle, CCandle and CCandleManager serve another purpose: checking if a trade has been entered for a given symbol and timeframe within an expert advisor. The recent trade on a symbol can be checked, but not for a timeframe. The toggle for this flag should be set or reset by the instance of the CExpertAdvisor itself, when needed. For both classes, toggle can be set using the TradeProcessed method.

For the candle manager, the TradeProcessed methods (getter and setter) only deals with finding the instance of CCandle requested and applying the appropriate value:

```
bool CCandleManagerBase::TradeProcessed(const string symbol,const int timeframe) const
  {
   CCandle *candle=Get(symbol,timeframe);
   if(CheckPointer(candle))
      return candle.TradeProcessed();
   return false;
  }
```

For CCandle, the process involves the assigning of a new value to one of its its class members, m\_trade\_processed. The following methods deal with the setting of the value of the said class member:

```
bool CCandleBase::TradeProcessed(void) const
  {
   return m_trade_processed;
  }

CCandleBase::TradeProcessed(bool value)
  {
   m_trade_processed=value;
  }
```

### OnTick Handler

The OnTick method of CExpertAdvisor is the most used function within the class. It is from this method where most of the action takes place. The core operation of this method is shown in the following diagram:

![CExpertAdvisorBase OnTick](https://c.mql5.com/2/29/expert__1.png)

The process begins by toggling the tick flag of the expert advisor. This is to ensure that double processing of a tick cannot occur. The OnTick method of CExpertAdvisor is ideally called only within the OnTick event function, but it can also be called through other means, such as OnChartEvent. In the absence of this flag, if the OnTick method of the class is called while it is still processing an earlier tick, a tick may be processed more than once, and if the tick would generate a trade, this would often result to a duplicate trade.

The refreshing of data is also necessary, as this ensures that the expert advisor has access to the most recent market data, and will not reprocess an earlier tick. If the expert advisor fails to refresh the data, it would reset the tick process flag, terminate the method, and wait for a new tick.

The next steps are the detection of new bars and checking of trade signals. The checking for this is done every tick by default. However, it is possible to extend this method so that it only checks signals when a new signal is detected (to speed up processing time, especially during backtesting and optimization).

The class also provides a member, m\_position\_reverse, which is intended to reverse position(s) opposite the current signal. The reversal performed here is only for the neutralization of the current position(s). In MetaTrader 4 and MetaTrader 5 hedging mode, this deals with the exit of the trades that are opposite the current signal (those going with the current signal will not be exited). In MetaTrader 5 netting mode, there can
only be one position at any given time, so the expert advisor will enter a new position of equal volume and opposite to that of the current position.

The trade signal is mostly check using m\_signals, but other factors such as trading on new bar only, and time filters, can also prevent the expert advisor from executing a new trade. Only when all the conditions are satisfied will the EA be able to enter a new trade.

At the end of the processing of the tick, the expert advisor will set the tick flag to false, and is then allowed to process another tick.

### Expert Advisors Container

Similar to other class objects discussed in previous articles, the CExpertAdvisor class would also have its designated container, which is CExpertAdvisors. The following code shows the declaration for its base class,
CExpertAdvisorsBase:

```
class CExpertAdvisorsBase : public CArrayObj
  {
protected:
   bool              m_active;
   int               m_uninit_reason;
   CObject          *m_container;
public:
                     CExpertAdvisorsBase(void);
                    ~CExpertAdvisorsBase(void);
   virtual int       Type(void) const {return CLASS_TYPE_EXPERTS;}
   virtual int       UninitializeReason(void) const {return m_uninit_reason;}
   //--- getters and setters
   void              SetContainer(CObject *container);
   CObject          *GetContainer(void);
   bool              Active(void) const;
   void              Active(const bool);
   int               OrdersTotal(void) const;
   int               OrdersHistoryTotal(void) const;
   int               TradesTotal(void) const;
   //--- initialization
   virtual bool      Validate(void) const;
   virtual bool      InitComponents(void) const;
   //--- events
   virtual void      OnTick(void);
   virtual void      OnChartEvent(const int,const long&,const double&,const string&);
   virtual void      OnTimer(void);
   virtual void      OnTrade(void);
   virtual void      OnDeinit(const int,const int);
   //--- recovery
   virtual bool      CreateElement(const int);
   virtual bool      Save(const int);
   virtual bool      Load(const int);
  };
```

This container primarily mirrors the public methods found in the CExpertAdvisor class. An example of this is the OnTick handler. The method simply iterates on each instance of CExpertAdvisor to call its OnTick method:

```
void CExpertAdvisorsBase::OnTick(void)
  {
   if(!Active()) return;
   for(int i=0;i<Total();i++)
     {
      CExpertAdvisor *e=At(i);
      e.OnTick();
     }
  }
```

With this container it is possible to store multiple instances of CExpertAdvisor. This is probably the only way to run multiple expert advisors on a single chart instance. Simply initialize multiple instances of CExpertAdvisor, store their pointers under a single CExpertAdvisors container, and then use the container's OnTick method to trigger the OnTick methods of each CExpertAdvisor instance. The same thing can be also done with each
instance of the CExpert class of the MQL5 Standard Library using the CArrayObj class or its heirs.

### Persistence of Data

Some data used in an instance of CExpertAdvisor only reside in computer memory. Normally, the data are often stored in the platform and the expert advisor gets the needed data from the platform itself through a function call. However, for data created dynamically while the expert advisor is running, this is not usually the case. When the OnDeinit event is triggered on an expert advisor, the expert advisor destroys all objects, and thus
loses the data.

[OnDeinit](https://www.mql5.com/en/docs/basis/function/events) can be triggered in a number of ways, such as closing the entire trading platform (MetaTrader 4 or MetaTrader 5), unloading the expert advisor from the chart, or even the act of recompiling the expert advisor source code. The full list of possible events that can trigger deinitialization can be found using the [UninitializeReason](https://www.mql5.com/en/docs/constants/namedconstants/uninit) function. When an expert advisor loses access on those data, it may behave as if it was just loaded on the chart for the first time.

Most of the volatile data in the CExpertAdvisor class can be found in one of its members, which is an instance of COrderManager. This is where instances of COrder and COrderStop (and descendants) are created as the expert advisor performs its usual routine. Since these instances are created dynamically during OnTick, they are not recreated when the expert advisor reinitializes. Therefore, the expert advisor should implement a method to save and
retrieve these volatile data. One way to implement this is to use a descendant of the [CFileBin](https://www.mql5.com/en/docs/standardlibrary/fileoperations/cfilebin) class, CExpertFile. The following code snippet shows the declaration of CExpertFileBase, its base class:

```
class CExpertFileBase : public CFileBin
  {
public:
                     CExpertFileBase(void);
                    ~CExpertFileBase(void);
   void              Handle(const int handle) { m_handle=handle; };
   uint              WriteBool(const bool value);
   bool              ReadBool(bool &value);
  };
```

Here, we are extending CFileBin to explicity declare methods to writing and reading data of [Boolean](https://www.mql5.com/en/docs/basis/types/integer/boolconst) type.

At the end of the class file, we declare an instance of the CExpertFile class. This instance will be used throughout the expert advisor if it were to save and load volatile data. Alternatively, one may simply rely on the [Save](https://www.mql5.com/en/docs/standardlibrary/cobject/cobjectsave) and [Load](https://www.mql5.com/en/docs/standardlibrary/cobject/cobjectload) methods inherited from [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject), and process the saving and loading of data in the usual way. However, this can be a very rigorous process. A great deal of effort and lines of code can be saved from using CFile (or its heirs) alone.

```
//CExpertFileBase class definition
//+------------------------------------------------------------------+
#ifdef __MQL5__
#include "..\..\MQL5\File\ExpertFile.mqh"
#else
#include "..\..\MQL4\File\ExpertFile.mqh"
#endif
//+------------------------------------------------------------------+
CExpertFile file;
//+------------------------------------------------------------------+
```

The order manager saves volatile data
through its Save method:

```
bool COrderManagerBase::Save(const int handle)
  {
   if(handle==INVALID_HANDLE)
      return false;
   file.WriteDouble(m_lotsize);
   file.WriteString(m_comment);
   file.WriteInteger(m_expiration);
   file.WriteInteger(m_history_count);
   file.WriteInteger(m_max_orders_history);
   file.WriteBool(m_trade_allowed);
   file.WriteBool(m_long_allowed);
   file.WriteBool(m_short_allowed);
   file.WriteInteger(m_max_orders);
   file.WriteInteger(m_max_trades);
   file.WriteObject(GetPointer(m_orders));
   file.WriteObject(GetPointer(m_orders_history));
   return true;
  }
```

Most of these data are of primitive types, except the last two, which are the orders and historical orders containers. For these data, the [WriteObject](https://www.mql5.com/en/docs/standardlibrary/fileoperations/cfilebin/cfilebinwriteobject) method of [CFileBin](https://www.mql5.com/en/docs/standardlibrary/fileoperations/cfilebin) is used, which simply calls the Save method of the object to be written. The following code shows the Save method of COrderBase:

```
bool COrderBase::Save(const int handle)
  {
   if(handle==INVALID_HANDLE)
      return false;
   file.WriteBool(m_initialized);
   file.WriteBool(m_closed);
   file.WriteBool(m_suspend);
   file.WriteInteger(m_magic);
   file.WriteDouble(m_price);
   file.WriteLong(m_ticket);
   file.WriteEnum(m_type);
   file.WriteDouble(m_volume);
   file.WriteDouble(m_volume_initial);
   file.WriteString(m_symbol);
   file.WriteObject(GetPointer(m_order_stops));
   return true;
  }
```

As we can see here, the process just repeats when saving objects. For primitive data types, the data is simply saved to file as usual. For complex data types, the Save method of the object is called through CFileBin's WriteObject method.

In cases where multiple instance of CExpertAdvisor is present, the container CExpertAdvisors should also have the capacity to save data:

```
bool CExpertAdvisorsBase::Save(const int handle)
  {
   if(handle!=INVALID_HANDLE)
     {
      for(int i=0;i<Total();i++)
        {
         CExpertAdvisor *e=At(i);
         if(!e.Save(handle))
            return false;
        }
     }
   return true;
  }
```

The method calls the Save method of each CExpertAdvisor instance. The single file handle means that there would only be one save file for each expert advisor file. It is possible for each CExpertAdvisor instance to have its own save file, but this would be the more complicated approach.

The more complex part is the loading of data. In saving data, the values of some class members are simply written to file. On the other hand, when loading data, the object instances would need to be recreated in ideally the same state prior to saving. The following code shows the Load method of the order manager:

```
bool COrderManagerBase::Load(const int handle)
  {
   if(handle==INVALID_HANDLE)
      return false;
   if(!file.ReadDouble(m_lotsize))
      return false;
   if(!file.ReadString(m_comment))
      return false;
   if(!file.ReadInteger(m_expiration))
      return false;
   if(!file.ReadInteger(m_history_count))
      return false;
   if(!file.ReadInteger(m_max_orders_history))
      return false;
   if(!file.ReadBool(m_trade_allowed))
      return false;
   if(!file.ReadBool(m_long_allowed))
      return false;
   if(!file.ReadBool(m_short_allowed))
      return false;
   if(!file.ReadInteger(m_max_orders))
      return false;
   if(!file.ReadInteger(m_max_trades))
      return false;
   if(!file.ReadObject(GetPointer(m_orders)))
      return false;
   if(!file.ReadObject(GetPointer(m_orders_history)))
      return false;
   for(int i=0;i<m_orders.Total();i++)
     {
      COrder *order=m_orders.At(i);
      if(!CheckPointer(order))
         continue;
      COrderStops *orderstops=order.OrderStops();
      if(!CheckPointer(orderstops))
         continue;
      for(int j=0;j<orderstops.Total();j++)
        {
         COrderStop *orderstop=orderstops.At(j);
         if(!CheckPointer(orderstop))
            continue;
         for(int k=0;k<m_stops.Total();k++)
           {
            CStop *stop=m_stops.At(k);
            if(!CheckPointer(stop))
               continue;
            orderstop.Order(order);
            if(StringCompare(orderstop.StopName(),stop.Name())==0)
              {
               orderstop.Stop(stop);
               orderstop.Recreate();
              }
           }
        }
     }
   return true;
  }
```

The code above for the COrderManager much more complicated in contrast with CExpertAdvisor's Load method. The reason is that unlike the order manager, the instances of CExpertAdvisor are created during OnInit, and so the container would simply have to call the Load method of each instance of CExpertAdvisor, rather than using the ReadObject method of CFileBin.

Class instances that were not created during OnInit, will have to be created as well when reloading the expert advisor. This is achieved by extending the [CreateElement](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjcreateelement) method of [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj). An object cannot simply create itself on its own, so it has to be created by its parent object or container, or even from the main source or header file itself. An example can be seen in the extended CreateElement method found on COrdersBase. Under this class, the container is COrders (a descendant of COrdersBase), and the object to be created is of type COrder:

```
bool COrdersBase::CreateElement(const int index)
  {
   COrder*order=new COrder();
   if(!CheckPointer(order))
      return(false);
   order.SetContainer(GetPointer(this));
   if(!Reserve(1))
      return(false);
   m_data[index]=order;
   m_sort_mode=-1;
   return CheckPointer(m_data[index]);
  }
```

Here, aside from creating the element, we also set its parent object or container, in order to differentiate if it belongs to the list of active trades (m\_orders class member of COrderManagerBase) or the history (m\_orders\_history of COrderManagerBase).

### Examples

Examples #1-#4 in this article are modified versions of the four examples found on the previous article (see Cross-Platform Expert Advisor: Custom Stops, Trailing, and Breakeven). Let us take a look at the most complex example, expert\_custom\_trail\_ha\_ma.mqh, which is a modified version of custom\_trail\_ha\_ma.mqh.

Before the OnInit function, we declared the following global object instances:

```
COrderManager *order_manager;
CSymbolManager *symbol_manager;
CSymbolInfo *symbol_info;
CSignals *signals;
CMoneys *money_manager;
CTimes *time_filters;
```

We replace this with an instance of CExpert. Some of the above can be found within CExpetAdvisor itself (e.g. COrderManager), while the rest have to be instantiated during OnInit (i.e. containers):

```
CExpertAdvisors experts;
```

At the very beginning of the method, we create an instance of CExpertAdvisor. We also call its Init method inputting the most basic settings:

```
int OnInit()
  {
//---
   CExpertAdvisor *expert=new CExpertAdvisor();
   expert.Init(Symbol(),Period(),12345,true,true,true);
//--- other code
//---
   return(INIT_SUCCEEDED);
  }
```

CSymbolInfo / CSymbolManager no longer needs to be instantiated since the instance of the CExpertAdvisor class is able to create instances of these classes on its own.

The user-defined function would also have to be removed, since our new expert advisor will no longer need these.

We removed the global declaration for the containers in our code, so they need to be declared from within OnInit. An example of this is the time filters container (CTimeFilters), as shown in the following code, found within the OnInit function:

```
CTimes *time_filters=new CTimes();
```

Pointers to containers that are previously “added” to the order manager are instead added to the instance of CExpertAdvisor. All the other containers that are not added to the order manager will have to be added to the
CExpertAdvisor instance as well. It would be the instance of COrderManager that would store the pointers. The CExpertAdvisor instance only creates wrapper methods.

After this, we add the CExpertAdvisor instance to an instance of CExpertAdvisors. We then call the InitComponents method of the CExpertAdvisors instance. This would ensure the initialization of all instances of CExpertAdvisor and their components.

```
int OnInit()
  {
//---
//--- other code
   experts.Add(GetPointer(expert));
   if(!experts.InitComponents())
      return(INIT_FAILED);
//--- other code
//---
   return(INIT_SUCCEEDED);
  }
```

Finally, we insert the code needed for
loading if the expert advisor was interrupted in its operation:

```
int OnInit()
  {
//---
//--- other code
 file.Open(savefile,FILE_READ);
   if(!experts.Load(file.Handle()))
      return(INIT_FAILED);
   file.Close();
//---
   return(INIT_SUCCEEDED);
  }
```

If the expert advisor fails to load from the file, it would return [INIT\_FAILED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode). However, in the event where no save file was supplied (and hence would generate [INVALID\_HANDLE](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants)), the expert advisor will not fail initialization, since the Load methods of CExpertAdvisors and CExpertAdvisor both
returns true upon receiving an invalid handle. There is some risk with this approach, but it is very unlikely for a save file to be opened by another program. Just make sure that each instance of expert advisor running on a chart has an exclusive save file (just like magic number).

The fifth example cannot be found on the previous article. Rather, it combines all the four expert advisors in this article into a single expert advisor. It simply uses a slightly modified version of the OnInit function of each of the expert advisors and declare it as a custom-defined function. Its return value is of type CExpertAdvisor\*. If the creation of the expert advisor failed, it would return [NULL](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants) instead of [INIT\_SUCCEEDED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode). The following code shows the updated OnInit function of the combined expert advisor header file:

```
int OnInit()
  {
//---
   CExpertAdvisor *expert1=expert_breakeven_ha_ma();
   CExpertAdvisor *expert2=expert_trail_ha_ma();
   CExpertAdvisor *expert3=expert_custom_stop_ha_ma();
   CExpertAdvisor *expert4=expert_custom_trail_ha_ma();

   if (!CheckPointer(expert1))
      return INIT_FAILED;
   if (!CheckPointer(expert2))
      return INIT_FAILED;
   if (!CheckPointer(expert3))
      return INIT_FAILED;
   if (!CheckPointer(expert4))
      return INIT_FAILED;

   experts.Add(GetPointer(expert1));
   experts.Add(GetPointer(expert2));
   experts.Add(GetPointer(expert3));
   experts.Add(GetPointer(expert4));

   if(!experts.InitComponents())
      return(INIT_FAILED);
   file.Open(savefile,FILE_READ);
   if(!experts.Load(file.Handle()))
      return(INIT_FAILED);
   file.Close();
//---
   return(INIT_SUCCEEDED);
  }
```

The expert advisor starts by instantiating each instance of CExpertAdvisor. It would then proceed to checking each of the pointers to CExpertAdvisor. If the pointer is not dynamic, then the function returns INIT\_FAILED, and the initialization fails. If each of the instances passes the checking of pointers, these pointers are then stored in an instance of CExpertAdvisors. The CExpertAdvisors instance (the container, not the expert advisor instance) would then initialize its components and load previous data if necessary.

The expert advisor uses custom-defined functions to create an instance of CExpertAdvisor. The following code shows the function used to create the 4th expert advisor instance:

```
CExpertAdvisor *expert_custom_trail_ha_ma()
{
   CExpertAdvisor *expert=new CExpertAdvisor();
   expert.Init(Symbol(),Period(),magic4,true,true,true);
   CMoneys *money_manager=new CMoneys();
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
   expert.AddMoneys(GetPointer(money_manager));

   CTimes *time_filters=new CTimes();
   if(time_range_enabled && time_range_end>0 && time_range_end>time_range_start)
     {
      CTimeRange *timerange=new CTimeRange(time_range_start,time_range_end);
      time_filters.Add(GetPointer(timerange));
     }
   if(time_days_enabled)
     {
      CTimeDays *timedays=new CTimeDays(sunday_enabled,monday_enabled,tuesday_enabled,wednesday_enabled,thursday_enabled,friday_enabled,saturday_enabled);
      time_filters.Add(GetPointer(timedays));
     }
   if(timer_enabled)
     {
      CTimer *timer=new CTimer(timer_minutes*60);
      timer.TimeStart(TimeCurrent());
      time_filters.Add(GetPointer(timer));
     }

   switch(time_intraday_set)
     {
      case INTRADAY_SET_1:
        {
         CTimeFilter *timefilter=new CTimeFilter(time_intraday_gmt,intraday1_hour_start,intraday1_hour_end,intraday1_minute_start,intraday1_minute_end);
         time_filters.Add(timefilter);
         break;
        }
      case INTRADAY_SET_2:
        {
         CTimeFilter *timefilter=new CTimeFilter(0,0,0);
         timefilter.Reverse(true);
         CTimeFilter *sub1 = new CTimeFilter(time_intraday_gmt,intraday2_hour1_start,intraday2_hour1_end,intraday2_minute1_start,intraday2_minute1_end);
         CTimeFilter *sub2 = new CTimeFilter(time_intraday_gmt,intraday2_hour2_start,intraday2_hour2_end,intraday2_minute2_start,intraday2_minute2_end);
         timefilter.AddFilter(sub1);
         timefilter.AddFilter(sub2);
         time_filters.Add(timefilter);
         break;
        }
      default: break;
     }
   expert.AddTimes(GetPointer(time_filters));

   CStops *stops=new CStops();
   CCustomStop *main=new CCustomStop("main");
   main.StopType(stop_type_main);
   main.VolumeType(VOLUME_TYPE_PERCENT_TOTAL);
   main.Main(true);
//main.StopLoss(stop_loss);
//main.TakeProfit(take_profit);
   stops.Add(GetPointer(main));

   CTrails *trails=new CTrails();
   CCustomTrail *trail=new CCustomTrail();
   trails.Add(trail);
   main.Add(trails);

   expert.AddStops(GetPointer(stops));

   MqlParam params[1];
   params[0].type=TYPE_STRING;
#ifdef __MQL5__
   params[0].string_value="Examples\\Heiken_Ashi";
#else
   params[0].string_value="Heiken Ashi";
#endif
   SignalHA *signal_ha=new SignalHA(Symbol(),0,1,params,signal_bar);
   SignalMA *signal_ma=new SignalMA(Symbol(),(ENUM_TIMEFRAMES) Period(),maperiod,0,mamethod,maapplied,signal_bar);
   CSignals *signals=new CSignals();
   signals.Add(GetPointer(signal_ha));
   signals.Add(GetPointer(signal_ma));
   expert.AddSignal(GetPointer(signals));
//---
   return expert;
}
```

As we can see, the code looks very similar to the OnInit function of the original expert advisor header file (expert\_custom\_trail\_ha\_ma.mqh). The other custom-defined functions are also organized in the same way.

### Final Notes

Before concluding this article, any reader who wishes to use this library should be made aware of these factors that contribute to the library's development:

At the time of this writing, the library featured in this article has over 10,000 lines of code (including comments). Despite this, it still remains a work in progress. More work needs to be done in order to fully utilize the
capabilities of both MQL4 and MQL5.

The author started working on this project prior to the introduction of hedging mode in MetaTrader 5. This has greatly influenced the further development of the library. As a result, the library tends to be more closer to adopting the conventions used in MetaTrader 4 than MetaTrader 5. Furthermore, the author also experienced some compatibility issues with some build updates released over the last few years, which led to some minor and
major adjustments to the code (and some delay on publishing some articles). At the time of this writing, the author experienced the build updates for both platforms to be less frequent and more stable over time. This trend is expected to further improve. Nevertheless, future build updates that may cause incompatibilities would still need to be addressed.

The library relies on data saved on memory to keep track of its own trades. This causes the expert advisors created using this library to heavily depend on saving and loading data in order to deal with possible interruptions the expert advisor may experience during its execution. Future work on this library, as well as any other library aiming at cross-platform compatibility, should be geared at achieving a stateless or near-stateless implementation, similar to the implementation of the MQL5 Standard Library.

As a final remark, the library featured in this article should not be viewed as a permanent solution. Rather, it should be used as an opportunity for a smoother transition from MetaTrader 4 to MetaTrader 5. The incompatibilities between MQL4 and MQL5 presents a huge roadblock to traders who intend to transition to the new platform. As a result, the MQL4 source code of their expert advisors need to be refactored in order to be made compatible with the MQL5 compiler. The library featured in this article is provided as a means to deploy an expert advisor to the new platform with little or no adjustments to the main expert advisor source code. This can help the trader in his decision whether to still use MetaTrader 4, or switch to MetaTrader 5. In the event that he decided to switch, very little adjustments would be necessary, and the trader can operate the usual way with his expert advisors. On the other hand, if he decides to stay on using the old platform, he is provided an option to quickly switch to the new platform once MetaTrader 4 becomes legacy software.

### Conclusion

This article has featured the CExpertAdvisor and CExpertAdvisors class objects, which are used to integrate all the components of a cross-platform expert advisor discussed in this article series. The article discusses how the two classes are instantiated and linked with the other components of a cross-platform expert advisor. It also introduces some solutions to problems usually encountered by expert advisors, such as new bar detection and the saving and loading of volatile data.

Programs Used in the Article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1. | expert\_breakeven\_ha\_ma.mqh | Header File | The main header file used in the first example |
| 2. | expert\_breakeven\_ha\_ma.mq4 | Expert Advisor | The main source file used for the MQL4 version in the first example |
| 3. | expert\_breakeven\_ha\_ma.mq5 | Expert Advisor | The main source file used for the MQL5 version in the first example |
| 4. | expert\_trail\_ha\_ma.mqh | Header File | The main header file used in the second example |
| 5. | expert\_trail\_ha\_ma.mq4 | Expert Advisor | The main source file used for the MQL4 version in the second example |
| 6. | expert\_trail\_ha\_ma.mq5 | Expert Advisor | The main source file used for the MQL5 version in the second example |
| 7. | expert\_custom\_stop\_ha\_ma.mqh | Header File | The main header file used in the third example |
| 8. | expert\_custom\_stop\_ha\_ma.mq4 | Expert Advisor | The main source file used for the MQL4 version in the third example |
| 9. | expert\_custom\_stop\_ha\_ma.mq5 | Expert Advisor | The main source file used for the MQL5 version in the third example |
| 10. | expert\_custom\_trail\_ha\_ma.mqh | Header File | The main header file used in the fourth example |
| 11. | expert\_custom\_trail\_ha\_ma.mq4 | Expert Advisor | The main source file used for the MQL4 version in the fourth example |
| 12. | expert\_custom\_trail\_ha\_ma.mq5 | Expert Advisor | The main source file used for the MQL5 version in the fourth example |
| 13. | combined.mqh | Header File | The main header file used in the fifth example |
| 14. | combined.mq4 | Expert Advisor | The main source file used for the MQL4 version in the fifth example |
| 15. | combined.mq5 | Expert Advisor | The main source file used for the MQL5 version in the fifth example |

Class Files Featured in the Article

| \# | Name | Type | Description |
| --- | --- | --- | --- |
| 1. | MQLx\\Base\\Expert\\ExperAdvisorsBase | Header File | CExpertAdvisors (CExpertAdvisor container, base class) |
| 2. | MQLx\\MQL4\\Expert\\ExperAdvisors | Header File | CExpertAdvisors (MQL4 version) |
| 3\. | MQLx\\MQL5\\Expert\\ExperAdvisors | Header File | CExpertAdvisors (MQL5 version) |
| 4\. | MQLx\\Base\\Expert\\ExperAdvisorBase | Header File | CExpertAdvisor (base class) |
| 5\. | MQLx\\MQL4\\Expert\\ExperAdvisor | Header File | CExpertAdvisor (MQL4 version) |
| 6\. | MQLx\\MQL5\\Expert\\ExperAdvisor | Header File | CExpertAdvisor (MQL5 version) |
| 7. | MQLx\\Base\\Candle\\CandleManagerBase | Header File | CCandleManager (CCandle container, base class) |
| 8. | MQLx\\MQL4\\Candle\\CandleManager | Header File | CCandleManager (MQL4 version) |
| 9. | MQLx\\MQL5\\Candle\\CandleManager | Header File | CCandleManager (MQL5 version) |
| 10\. | MQLx\\Base\\Candle\\CandleBase | Header File | CCandle (base class) |
| 11\. | MQLx\\MQL4\\Candle\\Candle | Header File | CCandle (MQL4 version) |
| 12. | MQLx\\MQL5\\Candle\\Candle | Header File | CCandle (MQL5 version) |
| 13\. | MQLx\\Base\\File\\ExpertFileBase | Header File | CExpertFile(base class) |
| 14\. | MQLx\\MQL4\\File\\ExpertFile | Header File | CExpertFile(MQL4 version) |
| 15\. | MQLx\\MQL5\\File\\ExpertFile | Header File | CExpertFile(MQL5 version) |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/3622.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/3622/mql5.zip "Download MQL5.zip")(143.3 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Cross-Platform Expert Advisor: Custom Stops, Breakeven and Trailing](https://www.mql5.com/en/articles/3621)
- [Cross-Platform Expert Advisor: Stops](https://www.mql5.com/en/articles/3620)
- [Cross-Platform Expert Advisor: Time Filters](https://www.mql5.com/en/articles/3395)
- [Cross-Platform Expert Advisor: Money Management](https://www.mql5.com/en/articles/3280)
- [Cross-Platform Expert Advisor: Signals](https://www.mql5.com/en/articles/3261)
- [Cross-Platform Expert Advisor: Order Manager](https://www.mql5.com/en/articles/2961)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/218058)**
(24)


![mbjen](https://c.mql5.com/avatar/avatar_na2.png)

**[mbjen](https://www.mql5.com/en/users/mbjen)**
\|
23 Jan 2018 at 04:09

How do I create a MM type based on previous orders results? It's so unflexible...

![mbjen](https://c.mql5.com/avatar/avatar_na2.png)

**[mbjen](https://www.mql5.com/en/users/mbjen)**
\|
11 Feb 2018 at 03:32

I managed to create a custom MM class, thanks.

How do I get the order close reason? How do I know it's closed due stop (any virtual or broker)?

How to use CEventAggregator class? Do you have any examples?

![Juer](https://c.mql5.com/avatar/avatar_na2.png)

**[Juer](https://www.mql5.com/en/users/juer)**
\|
16 Apr 2018 at 07:58

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CCandleManagerBase::Add(const string symbol,const int period)
  {
   if(CheckPointer(m_symbol_man))
     {
      CSymbolInfo *instrument=m_symbol_man.Get(symbol);
      if(CheckPointer(instrument))
        {
         instrument.Name(symbol);
         instrument.Refresh();
         CCandle *candle=new CCandle();
         candle.Init(instrument,period);
         return Add(/*instrument*/candle);
        }
     }
   return false;
  }
```

There is error. I commented the wrong value.


![Enrico Lambino](https://c.mql5.com/avatar/2014/10/54465D5F-0757.jpg)

**[Enrico Lambino](https://www.mql5.com/en/users/iceron)**
\|
16 Apr 2018 at 09:00

**Juer:**

There is error. I commented the wrong value.

I see. Thank you for this. It is now corrected. You can see the latest version here:

[https://github.com/iceron/MQLx/commits/master](https://www.mql5.com/go?link=https://github.com/iceron/MQLx/commits/master "/go?link=https://github.com/iceron/MQLx/commits/master")

![crt6789](https://c.mql5.com/avatar/avatar_na2.png)

**[crt6789](https://www.mql5.com/en/users/crt6789)**
\|
19 Jun 2023 at 08:52

May I ask the moderator, for the time of major news, volatility is intense, when a new candle can be both long and short, which for this one k line bar can only be placed once the order will conflict? A new candlestick can only be placed once order means only a total of one order or that long orders can be placed once, short orders can also be placed a single order a total of two orders?


![Auto search for divergences and convergences](https://c.mql5.com/2/29/MQL5_article_Divergention.png)[Auto search for divergences and convergences](https://www.mql5.com/en/articles/3460)

The article considers all kinds of divergence: simple, hidden, extended, triple, quadruple, convergence, as well as divergences of A, B and C classes. A universal indicator for their search and display on the chart is developed.

![Universal Expert Advisor: CUnIndicator and Use of Pending Orders (Part 9)](https://c.mql5.com/2/29/9lzld4pycep_npq2.png)[Universal Expert Advisor: CUnIndicator and Use of Pending Orders (Part 9)](https://www.mql5.com/en/articles/2653)

The article describes the work with indicators through the universal CUnIndicator class. In addition, new methods of working with pending orders are considered. Please note: from this point on, the structure of the CStrategy project has undergone substantial changes. Now all its files are located in a single directory for the convenience of users.

![Implementing a Scalping Market Depth Using the CGraphic Library](https://c.mql5.com/2/28/MQL5-avatar-cup-005.png)[Implementing a Scalping Market Depth Using the CGraphic Library](https://www.mql5.com/en/articles/3336)

In this article, we will create the basic functionality of a scalping Market Depth tool. Also, we will develop a tick chart based on the CGraphic library and integrate it with the order book. Using the described Market Depth, it will be possible to create a powerful assistant tool for short-term trading.

![Graphical Interfaces XI: Integrating the Standard Graphics Library (build 16)](https://c.mql5.com/2/29/MQL5-avatar-XI-CGraph.png)[Graphical Interfaces XI: Integrating the Standard Graphics Library (build 16)](https://www.mql5.com/en/articles/3527)

A new version of the graphics library for creating scientific charts (the CGraphic class) has been presented recently. This update of the developed library for creating graphical interfaces will introduce a version with a new control for creating charts. Now it is even easier to visualize data of different types.

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/3622&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071710727376415995)

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