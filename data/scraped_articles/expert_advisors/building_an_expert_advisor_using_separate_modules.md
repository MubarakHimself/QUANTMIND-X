---
title: Building an Expert Advisor using separate modules
url: https://www.mql5.com/en/articles/7318
categories: Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:30:05.182420
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/7318&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068312029560764394)

MetaTrader 5 / Expert Advisors


### Introduction

When developing indicators, Expert Advisors and scripts, developers often need to create various pieces of code, which are not directly
related to the trading strategy. For example, such code may concern Expert Advisor operation schedule: daily, weekly or monthly. As a
result, we will create an independent project, which can interact with the trading logic and other components using a simple
interface. With the least effort, such a schedule can be further used in custom developed Expert Advisors and indicators without
serious modifications. In this article we will try to apply a system approach to Expert Advisor creation using blocks. We will also
consider an interesting possibility, which will emerge as a result of our work. The article is intended for beginners.

### What Is the Usual Practice?

Let us try to understand how such an Expert Advisor can look like and what parts/components/modules it can contain. Where do we take such
components? The answer is simple and clear — in the process of application development, the programmer has to create various components,
which often have similar or the same functionality.

It is obvious that there is no need to implement the same feature, for example a trailing stop function, every time from scratch. In general
case, the trailing stop can perform similar functions and have similar inputs for different Expert Advisors. Thus, the programmer can
create trailing function code once and then insert it in required EAs with minimal effort. The same applies to a lot of other components,
including the trading schedule, various news filters and modules containing trading functions, etc.

Thus, we have kind of a construction set, based on which we can assemble an Expert Advisor using separate modules/blocks. Modules exchange
information with each other and with the "kernel" of the Expert Advisor, which is the "strategy" that makes decisions. Let us display
possible relationships between separate modules:

![](https://c.mql5.com/2/37/art1__1.png)

The resulting scheme is rather confusing. While it only shows the interaction of three modules and two
EA handlers: OnStart and OnTick. In a more complex Expert Advisor, internal binds will be even more complicated. Such an Expert Advisor is
hard to manage. Furthermore, if any of the modules needs to be excluded or an additional one needs to be added, this would cause considerable
difficulties. Furthermore, initial debugging and troubleshooting wouldn't be easy. One of the reasons for such difficulties is
connected with the fact, that binds are designed without a proper systematic approach. Forbid the modules to communicate with each other
and with the EA handlers, whenever it is necessary, and a certain order will appear:

- All modules are created in OnInit.
- EA logic is contained in OnTick.
- Modules exchange information only with OnTick.
- If necessary, modules are deleted in OnDeinit.

Such a simple solution can have a quick positive effect. Separate modules are easier to connect/disconnect, debug and modify. Logic in
OnTick will become more accessible for maintenance and improvement if binds are implemented in one handler instead of being added in
different places throughout the EA code:

![](https://c.mql5.com/2/37/art3.png)

This minor design change provides a clearer EA structure, which becomes intuitive. The new structure resembles the result of application of
the "Observer" pattern, though the structure itself is different from the pattern. Let's see how we can further improve the design.

### EA for Experiments

We need a simple Expert Advisor to check our ideas. We do not need a very complicated EA, because our purpose now is only to demonstrate the
features. The EA will open one sell order if the previous candlestick is bearish. The Expert Advisor will be designed using a modular
structure. The first module implements trading functions:

```
class CTradeMod {
public:
   double dBaseLot;
   double dProfit;
   double dStop;
   long   lMagic;
   void   Sell();
   void   Buy();
};

void CTradeMod::Sell()
{
  CTrade Trade;
  Trade.SetExpertMagicNumber(lMagic);
  double ask = SymbolInfoDouble(Symbol(),SYMBOL_ASK);
  Trade.Sell(dBaseLot,NULL,0,ask + dStop,ask - dProfit);
}

void CTradeMod::Buy()
{
  CTrade Trade;
  Trade.SetExpertMagicNumber(lMagic);
  double bid = SymbolInfoDouble(Symbol(),SYMBOL_BID);
  Trade.Buy(dBaseLot,NULL,0,bid - dStop,bid + dProfit);
}
```

The module is implemented as a class having open fields and methods. As for now, we do not need an implemented Buy() method in the module, but it
will be needed later. The value of separate fields should be clear; they are used for the volume, trade levels and Magic. How to use the module:
create it and call the Sell() method when an entry signal emerges.

Another module will be included in the EA:

```
class CParam {
public:
   double dStopLevel;
   double dFreezeLevel;
    CParam() {
      new_day();
    }//EA_PARAM()
    void new_day() {
      dStopLevel   = SymbolInfoInteger(Symbol(),SYMBOL_TRADE_STOPS_LEVEL) * Point();
      dFreezeLevel = SymbolInfoInteger(Symbol(),SYMBOL_TRADE_FREEZE_LEVEL) * Point();
    }//void new_day()
};
```

Let us consider this module in more detail. This is an auxiliary module which contains various parameters used by other modules and EA handlers.
You might have met a code like this:

```
...
input int MaxSpread = 100;
...
OnTick()
 {
   if(ask - bid > MaxSpread * Point() ) return;
....
 }
```

Obviously this fragment is ineffective. If we add all input (and other) parameters which need to be updated or converted, into a separate module
(here we deal with MaxSpread \* Point() ), we keep the global space clean and can efficiently control their state, as it is done with
stops\_level and freeze\_level values in the above CParam module.

Probably, a better solution would be to provide special getters rather than making the module fields open. Here the above solution is used to
simplify the code. For a real project, it is better to use a getter.

In addition, we could make an exception for the CParam module and allow access to this module not only by the OnTick() handler, but also by all
other modules and handlers.

Here is the inputs block and EA handlers:

```
input double dlot   = 0.01;
input int    profit = 50;
input int    stop   = 50;
input long   Magic  = 123456;

CParam par;
CTradeMod trade;

int OnInit()
  {
   trade.dBaseLot = dlot;
   trade.dProfit  = profit * _Point;
   trade.dStop    = stop   * _Point;
   trade.lMagic   = Magic;

   return (INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
  }

void OnTick()
  {
   int total = PositionsTotal();
   ulong ticket, tsell = 0;
   ENUM_POSITION_TYPE type;
   double l, p;
   for (int i = total - 1; i >= 0; i--) {
      if ( (ticket = PositionGetTicket(i)) != 0) {
         if ( PositionGetString(POSITION_SYMBOL) == _Symbol) {
            if (PositionGetInteger(POSITION_MAGIC) == Magic) {
               type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
               l    = PositionGetDouble(POSITION_VOLUME);
               p    = PositionGetDouble(POSITION_PRICE_OPEN);
               switch(type) {
                  case POSITION_TYPE_SELL:
                     tsell = ticket;
                     break;
               }//switch(type)
            }//if (PositionGetInteger(POSITION_MAGIC) == lmagic)
         }//if (PositionGetString(POSITION_SYMBOL) == symbol)
      }//if ( (ticket = PositionGetTicket(i)) != 0)
   }//for (int i = total - 1; i >= 0; i--)
   if (tsell == 0)
      {
        double o = iOpen(NULL,PERIOD_CURRENT,1);
        double c = iClose(NULL,PERIOD_CURRENT,1);
        if (c < o)
          {
            trade.Sell();
          }
      }
  }
```

The EA initializes modules in the OnInit() handler and then accesses them only from the OnTick() handler. In OnTick(), the EA loops through
open positions to check if the required position has already been opened. If the position has not yet been opened, the EA opens one provided
there is a signal.

Please note that the OnDeinit(const int reason) handler is currently empty. Modules are created so as they do not need to be explicitly deleted.
In addition, the CParam has not been used yet, because checks for position opening are not performed. If such checks were performed, the
CTradeMod module could need access to the CParam module and the developer would need to make an aforementioned exception and allow access to
CParam. However, this is not needed in our case.

Let us view this moment in more detail. The CTradeMod module may need data from CParam to check the stop loss and take profit levels as well as the
position volume. But the same check can be performed at the decision-making point: if the levels and the volume do not meet the requirements,
do not open a position. Thus, the check is moved to the OnTick() handler. As for our example, since trade level and volume values are specified
in input parameters, the check can be performed once, in the OnInit() handler. If the check is unsuccessful, then the initialization of the
entire EA should be ended with an error. Thus, the CTradeMod and CParam modules can act independently. This is relevant for the most of Expert
Advisors: independent modules operate via the OnTick() handler and do not know anything about each other. However, this condition cannot
be observed in some cases. We will consider them later.

### Making Improvements

The first issue to be addressed is the large position loop in the OnTick() handler. This piece of code is needed if the developer wishes to:

1. avoid further position opening if the entry signal remains active,
2. enable the EA to detect already placed orders after a pause or operation break,
3. enable the EA to collect current statistics, such as the total volume of open positions or the maximum drawdown.

This loop will be even larger in Expert Advisors that use grids and averaging techniques. Moreover, it is required if the EA uses virtual
trading levels. Thus, it is better to create a separate module based on this piece of code. In the simplest case, the module will detect
positions with the specific magic number and will notify the EA of such positions. In a more complicated case, the module could be
implemented as a kernel including simpler modules, such as logging or statistics collection modules. In this case the EA will have a
tree-like structure with the OnInit() и OnTick() handlers at the base of the tree. Here is what such a module could look like:

```
class CSnap {
public:
           void CSnap()  {
               m_lmagic = -1;
               m_symbol = Symbol();
           }
   virtual void  ~CSnap() {}
           bool   CreateSnap();
           long   m_lmagic;
           string m_symbol;
};
```

All fields are again open. Actually, the code has two fields: for the magic number and for the name of the symbol on which the EA is running. If
necessary, values for these fields can be set in the OnInit() handler. The main part of work is performed by the CreateSnap() method:

```
bool CSnap::CreateSnap() {
   int total = PositionsTotal();
   ulong ticket;
   ENUM_POSITION_TYPE type;
   double l, p;
   for (int i = total - 1; i >= 0; i--) {
      if ( (ticket = PositionGetTicket(i)) != 0) {
         if (StringLen(m_symbol) == 0 || PositionGetString(POSITION_SYMBOL) == m_symbol) {
            if (m_lmagic < 0 || PositionGetInteger(POSITION_MAGIC) == m_lmagic) {
               type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
               l    = PositionGetDouble(POSITION_VOLUME);
               p    = PositionGetDouble(POSITION_PRICE_OPEN);
               switch(type) {
                  case POSITION_TYPE_BUY:
// ???????????????????????????????????????
                     break;
                  case POSITION_TYPE_SELL:
// ???????????????????????????????????????
                     break;
               }//switch(type)
            }//if (lmagic < 0 || PositionGetInteger(POSITION_MAGIC) == lmagic)
         }//if (StringLen(symbol) == 0 || PositionGetString(POSITION_SYMBOL) == symbol)
      }//if ( (ticket = PositionGetTicket(i)) != 0)
   }//for (int i = total - 1; i >= 0; i--)
   return true;
}
```

The code is simple, but it has a few issues. How and where should the module pass the obtained information? What should be written in the
lines having question marks in the last code fragment? This may seem simple. Call CreateSnap() in the OnTick() handler: this method
performs the required tasks and saves the results in the CSnap class fields. Then the handler checks the fields and draws conclusions.

Well, this solution can be implemented in the simplest case. What would we do, if we need separate handling of each position parameters, for
example to calculate a weighted average value? In this case, a more universal approach is needed, in which data are forwarded to some next
object for further processing. For this purpose, a special pointer to such an object must be provided in CSnap:

```
CStrategy* m_strategy;
```

The method for assigning a value to this file:

```
     bool SetStrategy(CStrategy* strategy) {
              if(CheckPointer(strategy) == POINTER_INVALID) return false;
                 m_strategy = strategy;
                 return true;
              }//bool SetStrategy(CStrategy* strategy)
```

The object name CStrategy was selected because entry decisions as well as other decision can be made in this object. Therefore, the object can
define the strategy of the entire EA.

Now, the 'switch' in the CreateSnap() method will be as follows:

```
               switch(type) {
                  case POSITION_TYPE_BUY:
                     m_strategy.OnBuyFind(ticket, p, l);
                     break;
                  case POSITION_TYPE_SELL:
                     m_strategy.OnSellFind(ticket, p, l);
                     break;
               }//switch(type
```

If necessary, the code can be easily supplemented with switches for pending orders and appropriate method calls. In addition, the method
can be easily modified to enable collection of more data. CreateSnap() could be implemented as a virtual method to provide for potential
inheritance from the CSnap class. But this is not necessary in our case and we therefore will use a simpler code version.

Furthermore, a pointer to such an object (we have implemented it as the CStrategy\* pointer) can be useful not only for the current module. The potential
connection of a module with the EA operation logic can be needed for any module that performs active calculations. Therefore, let's provide
a special field and an initialization method in the base class:

```
class CModule {
   public:
      CModule() {m_strategy = NULL;}
     ~CModule() {}
     virtual bool SetStrategy(CStrategy* strategy) {
                     if(CheckPointer(strategy) == POINTER_INVALID) return false;
                     m_strategy = strategy;
                     return true;
                  }//bool SetStrategy(CStrategy* strategy)
   protected:
      CStrategy* m_strategy;
};
```

Further we will create modules inherited from CModule. In some cases, we may have excess code. But this drawback will be compensated by those modules,
in which such a pointer may really be needed. If any of the modules does not need such a pointer, simply do not call the SetStrategy(...) method
in such a module. The base module class can also be useful for placing additional fields and methods that are yet unknown to us. For example,
the following method (which is not implemented in our case) could be useful:

```
public:
   const string GetName() const {return m_sModName;}
protected:
         string m_sModName;
```

The method returns the module name which can be used for troubleshooting, debugging or in information panels.

Now let us see how the important CStrategy class can be implemented:

### Expert Advisor Strategy

As mentioned earlier, this must be an object which makes entry decisions. Such an object can also make decisions concerning exits,
modification and partial closure, among others. That is why the object is called like this. Obviously, it cannot be implemented as a module:
it is fundamentally important that the decision-making object is individual in each Expert Advisor. Otherwise, the resulting EA will be
identical to the previously developed one. Thus, we cannot develop such an object once and insert it in all EAs, like it is done with modules.
But this is not our purpose. Let's start the base class development using already known facts:

```
class CStrategy  {
public:
   virtual  void    OnBuyFind  (ulong ticket, double price, double lot) = 0;
   virtual  void    OnSellFind (ulong ticket, double price, double lot) = 0;
};// class CStrategy
```

Simple as that. We have added two methods to call in case CreateSnap() detects required positions. The CStrategy is implemented as an abstract
class, while the methods are declared virtual. This is because the object will be different for different Expert Advisor. Thus, the base
class can only be used for inheritance, while its methods will be overridden.

Now we need to add the CStrategy.mqh file to the CModule.mqh file:

```
#include "CStrategy.mqh"
```

After that the EA framework can be considered completed and we can proceed to further enhancements and improvements.

### Expert Advisor Strategy Improvements

Using the virtual methods, the CSnap object accesses the CStrategy object. But there must be other methods in the CStrategy object. The
strategy object must be able to make decisions. Thus, it should provide entry recommendations if an appropriate signal is detected, and
then it should execute such an entry. We need methods that force the CSnap object to call its CreateSnap() method, etc. Let's add some of the
methods to the CStrategy class:

```
   virtual  string  Name() const     = 0;
   virtual  void    CreateSnap()     = 0;
   virtual  bool    MayAndEnter()    = 0;
   virtual  bool    MayAndContinue() = 0;
   virtual  void    MayAndClose()    = 0;
```

This is a very conditional list, it can be changed or expanded for specific EAs. What these methods do:

- Name()                — returns the strategy name.

- CreateSnap()        — calls the same method of the CSnap object.
- MayAndEnter()      — checks if there is an entry signal and enters if there is a signal.

- MayAndContinue() — checks if there is an entry signal and enters again if there is a signal.

- MayAndClose()      — checks if there is an exit signal and closes all positions if there is such a signal.

Of course, this list is far from being complete. It lacks a very important method, the strategy initialization method. Our purpose is to
implement pointers to all modules in the CStrategy object and thus the OnTick() and other handlers will access only the CStrategy
object and will know nothing about the existence of other modules. Therefore, module pointers need to be added to the CStrategy object.
We cannot simply provide corresponding open fields and initialize them in the OnInit() handler. This will be explained later.

Instead, let's add a method for initialization:

```
virtual  bool    Initialize(CInitializeStruct* pInit) = 0;
```

with the initializing object CInitializeStruct, which will contain the required pointers. The object is described in the CStrategy.mqh as
follows:


```
class CInitializeStruct {};
```

It is an empty class, which is intended for inheritance, similar to CStrategy. We have completed the preparatory works and can proceed
to a real Expert Advisor.

### Practical Usage

Let's create a demo Expert Advisor operating according to a very simple logic: if the previous candlestick is bearish, open a sell position
with fixed Take Profit and Stop Loss levels. The next position should not be opened until the previous one is closed.

We have almost ready modules. Let's consider the class derived from CStrategy:

```
class CRealStrat1 : public CStrategy   {
public:
   static   string  m_name;
                     CRealStrat1(){};
                    ~CRealStrat1(){};
   virtual  string  Name() const {return m_name;}
   virtual  bool    Initialize(CInitializeStruct* pInit) {
                        m_pparam = ((CInit1* )pInit).m_pparam;
                        m_psnap = ((CInit1* )pInit).m_psnap;
                        m_ptrade = ((CInit1* )pInit).m_ptrade;
                        m_psnap.SetStrategy(GetPointer(this));
                        return true;
                    }//Initialize(EA_InitializeStruct* pInit)
   virtual  void    CreateSnap() {
                        m_tb = 0;
                        m_psnap.CreateSnap();
                    }
   virtual  bool    MayAndEnter();
   virtual  bool    MayAndContinue() {return false;}
   virtual  void    MayAndClose()   {}
   virtual  bool    Stop()            {return false;}
   virtual  void    OnBuyFind  (ulong ticket, double price, double lot) {}
   virtual  void    OnBuySFind (ulong ticket, double price, double lot) {}
   virtual  void    OnBuyLFind (ulong ticket, double price, double lot) {}
   virtual  void    OnSellFind (ulong ticket, double price, double lot) {tb = ticket;}
   virtual  void    OnSellSFind(ulong ticket, double price, double lot) {}
   virtual  void    OnSellLFind(ulong ticket, double price, double lot) {}
private:
   CParam*          m_pparam;
   CSnap*           m_psnap;
   CTradeMod*       m_ptrade;
   ulong            m_tb;
};
static string CRealStrat1::m_name = "Real Strategy 1";

bool CRealStrat1::MayAndEnter() {
   if (tb != 0) return false;
   double o = iOpen(NULL,PERIOD_CURRENT,1);
   double c = iClose(NULL,PERIOD_CURRENT,1);
   if (c < o) {
      m_ptrade.Sell();
      return true;
   }
   return false;
}
```

The EA code is simple and does not need explanations. Let us consider only some of the parts. The CreateSnap() method of the CRealStrat1
class resets the field containing the ticket of the already open Sell position and calls the CreateSnap() method of the CSnap module.
The CSnap module checks open positions. If a Sell position opened by this EA is found, the module calls the OnSellFind(...) method of the
CStrategy class, a pointer to which is contained in the CSnap module. As a result, the OnSellFind(...) method of the CRealStrat1
class is called. It changes the m\_tb field value once again. The MayAndEnter() method sees that a position has already been opened and
will not open a new one. Other methods of the CStrategy base class are not used in our EA, that is why their implementation is empty.

Another interesting point concerns the Initialize(...) method. This method adds in the CRealStrat1 class pointers to other modules which
may be needed to make separate decisions. The CStrategy class does not know which modules can be needed for the CRealStrat1 class and
therefore it uses an empty CInitializeStruct class. We will add the CInit1 class in the file containing the CRealStrat1 class
(although this is not necessary). CInit1 is inherited from CInitializeStruct:

```
class CInit1: public CInitializeStruct {
public:
   bool Initialize(CParam* pparam, CSnap* psnap, CTradeMod* ptrade) {
   if (CheckPointer(pparam) == POINTER_INVALID ||
       CheckPointer(psnap)  == POINTER_INVALID) return false;
      m_pparam = pparam;
      m_psnap  = psnap;
      m_ptrade = ptrade;
       return true;
   }
   CParam* m_pparam;
   CSnap*  m_psnap;
   CTradeMod* m_ptrade;
};
```

The class object can be created and initialized in the OnInit handler and it can be passed to the appropriate method in the CRealStrat1
class object. Thus, we have a complex structure consisting of separate objects. But we can work with the structure via a simple
interface in the OnTick() handler.

### OnInit() and OnTick() Handlers

Here is the OnInit() handler and a possible list of global objects:

```
CParam     par;
CSnap      snap;
CTradeMod  trade;

CStrategy* pS1;

int OnInit()
  {
   ...
   pS1 = new CRealStrat1();
   CInit1 ci;
   if (!ci.Initialize(GetPointer(par), GetPointer(snap), GetPointer(trade)) ) return (INIT_FAILED);
   pS1.Initialize(GetPointer(ci));
   return (INIT_SUCCEEDED);
  }

```

Only one object is created in the OnInit() handler: a CRealStrat1 class instance. It is initialized with a CInit1 class object. The
object is then destroyed in the OnDeinit() handler:


```
void OnDeinit(const int reason)
  {
      if (CheckPointer(pS1) != POINTER_INVALID) delete pS1;
  }
```

The resulting OnTick() handler is very simple:

```
void OnTick()
  {
      if (IsNewCandle() ){
         pS1.CreateSnap();
         pS1.MayAndEnter();

      }
  }
```

Check existing position at the opening of a new candlestick and then check if there is an entry signal. If there is a signal and the EA has not
yet performed an entry, open a new position. The handler is so simple that some additional "global" code can be easily added inside it, if
necessary. For example, you can instruct the EA not to start trading immediately after launch on a chart, but to wait for user
confirmation through a button click and so on.

Some other EA functions are not described here, but they are available in the attached zip archive.

Thus, we have designed an Expert Advisor consisting of separate bricks, the modules. But that is not all. Let's see further interesting
opportunities provided by this programming approach.

### What's next?

The first thing which can be further implemented concerns dynamic replacement of modules. A simple schedule can be replaced with a more
advanced one. In this case we need to replace the schedule as an object, rather than add properties and methods to the existing one, which
would complicate debugging and maintenance. We can provide "Debug" and "Release" versions for separate modules, create a manager to
control the modules.

But there is an even more interesting possibility. With our programming approach, we can implement dynamic EA strategy replacement,
which in our case is implemented as a replacement of the CRealStrat1 class. The resulting EA will have two kernels which implement two
strategies, e.g. trend trading and flat trading strategies. Furthermore, a third strategy can be added to trade during the Asian
session. It means we can implement multiple Expert Advisors within one EA and switch them dynamically. How to do it:

1. Develop a class containing decision logic, derived from the CStrategy class (like CRealStrat1)
2. Develop a class that initializes this new strategy, derived from CInitializeStruct (CInit1)
3. Connect the resulting file to the project.
4. Add a new variable in input parameters, which sets an active strategy at start.
5. Implement strategy switchers, i.e. a trading panel.

As an example, let's add one more strategy to our demo EA. Again, we will prepare a very simple strategy in order not to complicate the
code. The first strategy opened a Sell position. This one has a similar logic: if the previous candlestick is bullish, open a Buy
position with fixed Take Profit and Stop Loss levels. The next position should not be opened until the previous one is closed. The
second strategy code is very similar to the first one and is available in the attached zip archive. The new strategy initialization
class is also contained in the archive.

Let's consider what changes should be made to the EA file containing input parameters and handlers:

```
enum CSTRAT {
   strategy_1 = 1,
   strategy_2 = 2
};

input CSTRAT strt  = strategy_1;

CSTRAT str_curr = -1;

int OnInit()
  {
   ...
   if (!SwitchStrategy(strt) ) return (INIT_FAILED);
   ...
   return (INIT_SUCCEEDED);
  }

void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
      if (id == CHARTEVENT_OBJECT_CLICK && StringCompare(...) == 0 ) {
         SwitchStrategy((CSTRAT)EDlg.GetStratID() );
      }
  }

bool SwitchStrategy(CSTRAT sr) {
   if (str_curr == sr) return true;
   CStrategy* s = NULL;
   switch(sr) {
      case strategy_1:
         {
            CInit1 ci;
            if (!ci.Initialize(GetPointer(par), GetPointer(snap), GetPointer(trade)) ) return false;
            s = new CRealStrat1();
            s.Initialize(GetPointer(ci));
         }
         break;
      case strategy_2:
         {
            CInit2 ci;
            if (!ci.Initialize(GetPointer(par), GetPointer(snap), GetPointer(trade)) ) return false;
            s = new CRealStrat2();
            s.Initialize(GetPointer(ci));
         }
         break;
   }
   if (CheckPointer(pS1) != POINTER_INVALID) delete pS1;
   pS1 = s;
   str_curr = sr;
   return true;
}
```

The strategy switcher function SwitchStrategy(...) and the OnChartEvent(...) handler are connected with a trading panel. Its
code is not provided in the article but is available in the attached zip archive. Also, dynamic strategy management is not a
complicated task. Create a new object with the strategy, delete the previous one and write a new pointer to the variable:

```
CStrategy* pS1;
```

After that the EA will access the new strategy in OnTick() and thus the new operation logic will be used. The hierarchy of objects and the
main dependencies look like this:

![](https://c.mql5.com/2/37/art4.png)

The figure does not show the trading panel and the connections which occur as secondary during initialization. At this stage, we can
consider our task completed: the Expert Advisor is ready for operation and further improvement. With the used block approach,
critical improvements and modifications can be made in a fairly short time.



### Conclusion

We have designed an Expert Advisor using elements of standard design patterns Observer and Facade. The full description of these
(and other) patterns is provided in the book "Design Patterns. Elements of Reusable Object-Oriented Software" by Erich Gamma, Richard
Helm, Ralph Johnson and John Vlissides. I recommend reading this book.

### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | Ea&Modules.zip | Archive | A zip archive with the Expert Advisor files. |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7318](https://www.mql5.com/ru/articles/7318)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7318.zip "Download all attachments in the single ZIP archive")

[EaxModules.zip](https://www.mql5.com/en/articles/download/7318/eaxmodules.zip "Download EaxModules.zip")(7.74 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MVC design pattern and its application (Part 2): Diagram of interaction between the three components](https://www.mql5.com/en/articles/10249)
- [MVC design pattern and its possible application](https://www.mql5.com/en/articles/9168)
- [Using cryptography with external applications](https://www.mql5.com/en/articles/8093)
- [Parsing HTML with curl](https://www.mql5.com/en/articles/7144)
- [Arranging a mailing campaign by means of Google services](https://www.mql5.com/en/articles/6975)
- [A DLL for MQL5 in 10 Minutes (Part II): Creating with Visual Studio 2017](https://www.mql5.com/en/articles/5798)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/327418)**
(5)


![Aleksey Mavrin](https://c.mql5.com/avatar/avatar_na2.png)

**[Aleksey Mavrin](https://www.mql5.com/en/users/alex_all)**
\|
10 Nov 2019 at 16:31

Modularity, interchangeability, basic [design](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it's not for sure ") principles. I think for the majority of those who develop on a more or less regular basis it is obvious and the article does not bring anything new. But for newcomers who are getting acquainted with programming through MQL, it may open their eyes).


![dmc9966](https://c.mql5.com/avatar/avatar_na2.png)

**[dmc9966](https://www.mql5.com/en/users/dmc9966)**
\|
30 Nov 2019 at 18:06

Mr Novichkov,

Thank you for sharing your hard work with the community. I am teaching myself MQL and eventually want to code my own EA's, your work with modules and pattern design helps me tremendously

Thank you

Dan

![Andrei Novichkov](https://c.mql5.com/avatar/2016/11/58342001-4AC3.png)

**[Andrei Novichkov](https://www.mql5.com/en/users/andreifx60)**
\|
30 Nov 2019 at 18:23

**dmc9966 :**

Mr Novichkov,

Thank you for sharing your hard work with the community. I am teaching myself MQL and eventually want to code my own EA's, your work with modules and pattern design helps me tremendously

Thank you

Dan

Thanks. I am happy to help you )


![Manuraj Dhanda](https://c.mql5.com/avatar/avatar_na2.png)

**[Manuraj Dhanda](https://www.mql5.com/en/users/mdhanda)**
\|
3 Dec 2019 at 10:30

I really liked this approach and finally some structure for an EA.

What all need to be changed to adapt it to MQL4? I'm using MQL\_Easy library to develop common code for my EA on both platforms.

I'll appreciate your advice. Thanks.

![Andrei Novichkov](https://c.mql5.com/avatar/2016/11/58342001-4AC3.png)

**[Andrei Novichkov](https://www.mql5.com/en/users/andreifx60)**
\|
3 Dec 2019 at 19:33

**Manuraj Dhanda:**

I really liked this approach and finally some structure for an EA.

What all need to be changed to adapt it to MQL4? I'm using MQL\_Easy library to develop common code for my EA on both platforms.

I'll appreciate your advice. Thanks.

To work with MQL4 you just need to try to compile the [project](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it is not exactly") in MT4 )))) There shouldn't be a lot of mistakes. The code is pretty simple

![Library for easy and quick development of MetaTrader programs (part XX): Creating and storing program resources](https://c.mql5.com/2/37/MQL5-avatar-doeasy__2.png)[Library for easy and quick development of MetaTrader programs (part XX): Creating and storing program resources](https://www.mql5.com/en/articles/7195)

The article deals with storing data in the program's source code and creating audio and graphical files out of them. When developing an application, we often need audio and images. The MQL language features several methods of using such data.

![Library for easy and quick development of MetaTrader programs (part XIX): Class of library messages](https://c.mql5.com/2/37/MQL5-avatar-doeasy__1.png)[Library for easy and quick development of MetaTrader programs (part XIX): Class of library messages](https://www.mql5.com/en/articles/7176)

In this article, we will consider the class of displaying text messages. Currently, we have a sufficient number of different text messages. It is time to re-arrange the methods of their storage, display and translation of Russian or English messages to other languages. Besides, it would be good to introduce convenient ways of adding new languages to the library and quickly switching between them.

![Library for easy and quick development of MetaTrader programs (part XXI): Trading classes - Base cross-platform trading object](https://c.mql5.com/2/37/MQL5-avatar-doeasy__3.png)[Library for easy and quick development of MetaTrader programs (part XXI): Trading classes - Base cross-platform trading object](https://www.mql5.com/en/articles/7229)

In this article, we will start the development of the new library section - trading classes. Besides, we will consider the development of a unified base trading object for MetaTrader 5 and MetaTrader 4 platforms. When sending a request to the server, such a trading object implies that verified and correct trading request parameters are passed to it.

![Library for easy and quick development of MetaTrader programs (part XVIII): Interactivity of account and any other library objects](https://c.mql5.com/2/37/MQL5-avatar-doeasy.png)[Library for easy and quick development of MetaTrader programs (part XVIII): Interactivity of account and any other library objects](https://www.mql5.com/en/articles/7149)

The article arranges the work of an account object on a new base object of all library objects, improves the CBaseObj base object and tests setting tracked parameters, as well as receiving events for any library objects.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/7318&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068312029560764394)

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