---
title: The Prototype of a Trading Robot
url: https://www.mql5.com/en/articles/132
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:02:25.189226
---

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/132&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071527435352091055)

MetaTrader 5 / Examples


### Introduction

The life cycle of any trading system is reduced to opening and closing positions. This is beyond any doubts. But when it comes to the algorithm realization, here there are as many opinions as programmers. Everyone will be able to solve the same problem in his own way, but with the same final result.

Over the years the of programming practice several approaches to constructing experts' logic and structure have been tried. At the moment it can be argued that established a clear pattern template that is used in all codes.

This approach is not 100% universal, but it may change your method of designing expert's logic. And the case is not what capabilities of working with orders you want to use the expert. The whole point - is the **principle** of creating a trading model.

### 1\. Principles of Designing Trading Systems and Types of Event Sources

The basic approach to design of the algorithm, used by the majority, is to trace one position from its opening until closing. This is linear approach. And if you want to make changes to the code - it often leads to great complications, since a large number of conditions emerges and the code accumulates new branches of analysis.

The best solution to model a trading robot is **"to serve conditions"**. And fundamental principle – to analyze not how this condition of expert and its positions and orders arose - but what we should **do with them now**. This basic principle is fundamentally changing the management of trade and simplifies the development of code.

Consider it in more detail.

**1.1. The Principle of "Serving Conditions"**

As already mentioned, the expert does not need to know how the current state has been achieved. It must know what to do with it now according to its environment (parameter values, stored [orders properties](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties) etc.).

This principle is directly related to the fact that the expert exists from loop to loop (particularly - from the tick to tick), and it should not worry about what happened with orders at the previous tick. Therefore, you must use an [event-driven approach](https://www.mql5.com/en/docs/basis/function/events) of managing orders. I.e. on the current tick the expert saves its state, which is the starting point for decision about the next tick.

For example, you must remove all [pending orders](https://www.mql5.com/en/docs/trading/orderstotal) of expert and only then continue to analyze indicators and to place new orders. Most of the code examples that we have seen use the "while (true) {try to remove}" looping or slightly softer the "while (k < 1000) {try to remove; k++;}" looping. We will skip the variant, where one-time call of remove command without error analysis.

This method is linear, it it "hangs" the expert for indefinite amount of time.

Therefore, it will be more correct not to loop an expert, but to store the order to remove orders, so that at every new tick this order will be checked while attempting to delete pending order. In this case, an expert, while reading the state parameters, knows that in this moment it must delete orders. And it will attempt to remove them. If a trading error will occur, an expert will simply block further analysis and work before the next loop.

**1.2. The Second Main Principle of Design - is the maximal possible abstraction** from considered [position direction](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type) (Buy/Sell), currency and chart. All expert functions should be implemented in such way, that direction or symbol are analyzed in rare cases when it really can not be avoided (for example, when you consider the favorable growth of price for the open position, although there are different options of avoiding specifics). Always try to avoid such "low level" design. This will reduce the code and the process of writing functions at least twice. And will make them "trade-independent.

The implementation of this principle is to replace the explicit analysis of order types, symbol parameters and dependent calculated parameters with the macro-functions. Next article we cover this implementation in details.

**1.3. Third principle** **–** **segmentation of algorithm into logical lexemes (independent modules)**

In practice, we can say that the best approach is the separation of expert operations into individual functions. I think you will agree that it is difficult to write the whole algorithm of the expert writing in one function, and it complicates the subsequent analysis and editing. So we should not do it in MQL5, which now provides almost complete control over your environment.

Therefore, the logical lexemes (e.g. opening, trailing, closure of orders) should be implemented separately from each other with full analysis of environmental parameters and events. Through this approach, the expert becomes flexible in design. You can easily add new independent modules into it without touching existing ones, or disable existing modules without altering the main code.

These three principles make it possible to create a single prototype for all the experts, that you can easily modify and adapt to any given task.

**Event sources for the expert system are:**

**![](https://c.mql5.com/2/1/Fig0.jpg)**

**1\. Indicators.** An example - is the analysis of the indicator lines values, their intersections, combinations, etc. Also, indicators can be: current time, the data obtained from Internet, etc. In most cases, the indicator events are used to signal the opening and closing of orders. Less for their adjustment (usually Trailing Stop Loss or pending order for the indicator).

For example, the practical implementation of indicator can be called an expert, that analyzes intersection of the fast and slow MA with the further opening of the position into direction of intersection.

**2\. Existing orders, positions and their state.** For example, the current loss or profit size, the presence/absence of positions or pending orders, the profit of closed position, etc. Practical implementation of these events is much broader and more diverse, as there are more options of their relationship than for the indicator events.

The simplest example of an expert, based only on the trade event, is refilling for averaging existing position and output it into desired profit. I.e. the presence of loss on available position will be an event to place a new averaging order.

Or, for example, Trailing Stop Loss. This function checks an event, when the price moves into profit for a specified number of points from the previous Stop Loss. As a result, expert pulls thee Stop Loss after the price.

**3\. External events.** Although such event usually doesn't occur in purely expert system, but in general it should be considered for making a decision. This includes adjusting the orders, positions, processing trade errors, processing chart events (moving/creating/deleting objects, pressing buttons, etc.). In general, these are the events, that are not verifiable on history and occur only when expert works.

A striking example of such experts are trade-information systems with graphical trade control.

All variety of experts is based on combination of these three sources of events

### 2\. The CExpertAdvisor base class – expert constructor

What will be the work of trade expert? General scheme of MQL-program interactions is shown on the diagram below.

![Figure 1. General scheme of MQL-program elements interactions](https://c.mql5.com/2/1/Fig1__4.png)

Figure 1. General scheme of MQL-program elements interactions

As you can see from the scheme, first comes the entrance to the working loop (this can be a tick or a timer signal). At this stage in the first block this tick can be filtered without processing. This is done in those cases, when the expert is not needed to work on every tick, but only on new bar or if the expert is simply not allowed to work.

Then the program goes into second block - modules of working with orders and positions, and only then the event processing blocks are called from modules. Each module can inquire only its interested event.

This sequence can be called as scheme with **direct** logic, since first it determines **WHAT** the expert will do (what event processing modules are used), and only then it implements **HOW** and **WHY** it will do it (getting event signals).

Direct logic is consistent with our perception of the world and universal logic. After all, a man thinks first concrete concepts, then he summarizes them and then classifies and identifies their interrelationships.

Designing experts is no exception in this regard. First, it is declared what an expert should do (to open and close positions, to pull the protective stop), and only then it is specified, in which events and how it should do it. But in any case not vice versa: receive signal and think where and how to process it. This is the reverse logic, and it's better not to use it, since as a result you'll get cumbersome code with large number of condition branches.

Here is an example of reverse and direct logic. Take the opening/closing by the [RSI](https://www.mql5.com/en/code/47) signal.

- In _reverse logic_ the expert starts with obtaining the indicator value, and then it checks the direction of signal and what you have to do with the position: to open the Buy and to close the Sell, or vice versa - to open the Sell and to close the Buy. That is, the entry point is to obtain and analyze the signal.

- In _direct logic_ everything is opposite. Expert has two modules of opening and closing positions, and it simply checks the conditions to execute these modules. I.e., after entering the opening module, the expert receives the indicator value and checks whether it is a signal to open. Then, after entering the orders closure module, the expert checks whether it is a signal to close position. That is, there is no entry point - there are independently working modules of system state analysis (the first principle of design).


Now, if you want to complicate the expert, it will be much easier using the second variant than the first. It will be sufficient to create a new module of event processing.

And in the first variant you'll have to revise the structure of signal processing or to paste it as a separate function.

**Recommendation:** When describing the trading system, don't begin with the words like "1. Get the signal ... open order", but to immediately split into sections: "a) The condition of opening orders, b) Conditions of orders maintenance, etc." and in each to analyze the required signals.

To better understand this approach, here are the different work schemes in the contexts of four different experts.

![Figure 2. Examples of experts implementation](https://c.mql5.com/2/1/Fig2__4.png)

Figure 2. Examples of experts implementation

> a). Expert, based only on signals of some indicator. It can open and close positions when signal is changing. Example - an MA expert.
>
> b). Expert with graphical trade control.
>
> c). Expert based on indicators, but with addition of Trailing Stop Loss and operating time. Example - scalping on news with opening position into the trend by the MA indicator.
>
> d). Expert without indicators, with positions averaging. It verifies position parameters only once when opening a new bar. Example - averaging expert.

As can be seen from the schemes, any trading system is very easy to describe using direct logic

### 3\. Implementation of Expert Class

Create a class using all the above mentioned rules and requirements, which will be the basis for all future experts.

The minimum functionality that should be in the **CExpertAdvisor** class is following:

_1\. Initialization_:

> - Register indicators
> - Set the initial values of parameters
> - Adjust to required symbol and timeframe

_2\. Functions of Getting Signals_

> - Allowed working time (traded intervals)
> - Determine the signal to open/close positions or orders
> - Determine the filter (trend, time, etc.)
> - Start/Stop timer

_3\. Service Functions_

> - Calculate the open price, the SL and TP levels, the volume of order
> - Send trade requests (open, close, modify)

_4\. Trade Modules_

> - Process the signals and filters
> - Control the positions and orders
> - Work in expert functions: OnTrade(), OnTimer(), OnTester(), OnChartEvent().

_5\. Deinitialization_

> - Output messages, reports
> - Clear chart, unload indicators

All functions of the class are divided into three groups. The general scheme of nested functions and their descriptions are presented below.

![Figure 3. Scheme of nesting functions of an expert](https://c.mql5.com/2/1/Fig3__3.png)

Figure 3. Scheme of nesting functions of an expert

**_1\. Macro Functions_**

This small group of functions is the basis for working with order types, symbol parameters and price values to set orders (the opening and the stops). These macro functions provide the second principle of design - abstraction. They work in the context of the symbol, which is used by the expert.

Macro functions of converting types work with the direction of the market - buy or sell. Therefore, in order not to create your own constants, better use the existing ones - [ORDER\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type) and [ORDER\_TYPE\_SELL](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type). Here are some examples of using macro and the results of their work.

```
   //--- Type conversion macro
   long       BaseType(long dir);        // returns the base type of order for specified direction
   long       ReversType(long dir);      // returns the reverse type of order for specified direction
   long       StopType(long dir);        // returns the stop-order type for specified direction
   long       LimitType(long dir);       // returns the limit-order type for specified direction

   //--- Normalization macro
   double     BasePrice(long dir);       // returns Bid/Ask price for specified direction
   double     ReversPrice(long dir);     // returns Bid/Ask price for reverse direction

   long dir,newdir;
   dir=ORDER_TYPE_BUY;
   newdir=ReversType(dir);               // newdir=ORDER_TYPE_SELL
   newdir=StopType(dir);                 // newdir=ORDER_TYPE_BUY_STOP
   newdir=LimitType(dir);                // newdir=ORDER_TYPE_BUY_LIMIT
   newdir=BaseType(newdir);              // newdir=ORDER_TYPE_BUY

   double price;
   price=BasePrice(dir);                 // price=Ask
   price=ReversPrice(dir);               // price=Bid
```

When developing experts the macro allow you _not to specify_ processed direction and help to create more compact code.

**_2\. Service Functions_**

These functions are designed to work with orders and positions. Like the macro function they are also low level. For convenience, they can be divided into two categories: information functions and executive functions. They all perform only one kind of action, without analyzing any events. They perform orders from senior expert handlers.

_Examples of information functions:_ finding the maximum opening price of current pending orders; finding out how position has been closed - with profit or loss; getting the number and list of tickets of expert's orders, etc.

_Examples of executive functions:_ closing the specified orders; modifying Stop Loss in the specified position, etc.

This group is the largest. This is the kind of functionality, on which the whole routine work of the expert is based. The large number of examples of these functions can be found on the forum at [https://www.mql5.com/ru/forum/107476](https://www.mql5.com/ru/forum/107476 "https://www.mql5.com/ru/forum/107476"). But in addition to this the MQL5 [standard library](https://www.mql5.com/en/docs/standardlibrary) already contains classes that take upon themselves part of the work on placing orders and positions, particularly - the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class.

But any task of yours will require to create new implementations or to slightly modify existing ones.

**_3\. Event Processing Modules_**

The group of these functions is a high-level superstructure over the first two groups. As mentioned above - these are ready-to-use blocks of which your expert is constructed. In general, they are included into the event processing function of MQL-program: [OnStart()](https://www.mql5.com/en/docs/basis/function/events#onstart), [OnTick()](https://www.mql5.com/en/docs/basis/function/events#ontick), [OnTimer()](https://www.mql5.com/en/docs/basis/function/events#ontimer), [OnTrade()](https://www.mql5.com/en/docs/basis/function/events#ontrade), [OnChartEvent()](https://www.mql5.com/en/docs/basis/function/events#onchartevent). This group is not numerous, and the contents of these modules can be adjusted from task to task. But essentially nothing changes.

In the modules everything should be abstract (the second design principle) in order for the same module can be invoked both for buy and sell. This is achieved, of course, with the help of macro.

**So, proceed with implementation**

_1\. Initialization, Deinitialization_

```
class CExpertAdvisor
  {
protected:
   bool              m_bInit;       // flag of correct initialization
   ulong             m_magic;       // magic number of expert
   string              m_smb;       // symbol, on which expert works
   ENUM_TIMEFRAMES      m_tf;       // working timeframe
   CSymbolInfo      m_smbinf;       // symbol parameters
   int               m_timer;       // time for timer

public:
   double              m_pnt;       // consider 5/3 digit quotes for stops
   CTrade            m_trade;       // object to execute trade orders
   string              m_inf;       // comment string for information about expert's work
```

This is the minimum required set of parameters for the expert functions to work.

The m\_smb and m\_tf parameters are specially placed in the expert properties to easily tell the expert, what currency and what period to work on. For example, if you assign m\_smb = "USDJPY", the expert will work on that symbol, regardless on which symbol it was run. If you set tf = PERIOD\_H1, then all the signals and analysis of indicators will take place on the H1 chart.

Further there are class methods. The first three methods are initialization and deinitialization of an expert.

```
public:
   //--- Initialization
   void              CExpertAdvisor();                               // constructor
   void             ~CExpertAdvisor();                               // destructor
   virtual bool      Init(long magic,string smb,ENUM_TIMEFRAMES tf); // initialization
```

The constructor and destructor in the base class do nothing.

The Init() method makes initialization of expert parameters by the symbol, timeframe and magic number.

```
//------------------------------------------------------------------ CExpertAdvisor
void CExpertAdvisor::CExpertAdvisor()
  {
   m_bInit=false;
  }
//------------------------------------------------------------------ ~CExpertAdvisor
void CExpertAdvisor::~CExpertAdvisor()
  {
  }
//------------------------------------------------------------------ Init
bool CExpertAdvisor::Init(long magic,string smb,ENUM_TIMEFRAMES tf)
  {
   m_magic=magic; m_smb=smb; m_tf=tf;         // set initializing parameters
   m_smbinf.Name(m_smb);                      // initialize symbol
   m_pnt=m_smbinf.Point();                    // calculate multiplier for 5/3 digit quote
   if(m_smbinf.Digits()==5 || m_smbinf.Digits()==3) m_pnt*=10;
   m_trade.SetExpertMagicNumber(m_magic);     // set magic number for expert

   m_bInit=true; return(true);                // trade allowed
  }
```

_2\. Functions of Getting Signals_

These functions analyze market and indicators.

```
   bool              CheckNewBar();                          // check for new bar
   bool              CheckTime(datetime start,datetime end); // check allowed trade time
   virtual long      CheckSignal(bool bEntry);               // check signal
   virtual bool      CheckFilter(long dir);                  // check filter for direction
```

The first two functions have quite specific implementation and can be used in further children of this class.

```
//------------------------------------------------------------------ CheckNewBar
bool CExpertAdvisor::CheckNewBar()          // function of checking new bar
  {
   MqlRates rt[2];
   if(CopyRates(m_smb,m_tf,0,2,rt)!=2)      // copy bar
     { Print("CopyRates of ",m_smb," failed, no history"); return(false); }
   if(rt[1].tick_volume>1) return(false);   // check volume
   return(true);
  }
//---------------------------------------------------------------   CheckTime
bool CExpertAdvisor::CheckTime(datetime start,datetime end)
  {
   datetime dt=TimeCurrent();                          // current time
   if(start<end) if(dt>=start && dt<end) return(true); // check if we are in the range
   if(start>=end) if(dt>=start|| dt<end) return(true);
   return(false);
  }
```

The latter two always depends on those indicators, that you are using. It's simply impossible to set these functions for all cases.

The main thing - it is important to understand that the CheckSignal() and CheckFilter() signal functions can analyze absolutely any indicators and their combinations! I.e. trade modules, in which these signals will be subsequently included, are independent from sources.

This allows you to use once written expert as a template for other experts that work on a similar principle. Just change the analyzed indicators or add new filtering conditions.

_3\. Service Functions_

As already mentioned, this group of functions is the most numerous. For our practical tasks described in the article it will be enough to implement four such functions:

```
   double         CountLotByRisk(int dist,double risk,double lot); // calculate lot by size of risk
   ulong          DealOpen(long dir,double lot,int SL,int TP);     // execute deal with specified parameter
   ulong          GetDealByOrder(ulong order);                     // get deal ticket by order ticket
   double         CountProfitByDeal(ulong ticket);                 // calculate profit by deal ticket
```

```
//------------------------------------------------------------------ CountLotByRisk
double CExpertAdvisor::CountLotByRisk(int dist,double risk,double lot) // calculate lot by size of risk
  {
   if(dist==0 || risk==0) return(lot);
   m_smbinf.Refresh();
   return(NormalLot(AccountInfoDouble(ACCOUNT_BALANCE)*risk/(dist*10*m_smbinf.TickValue())));
  }
//------------------------------------------------------------------ DealOpen
ulong CExpertAdvisor::DealOpen(long dir,double lot,int SL,int TP)
  {
   double op,sl,tp,apr,StopLvl;
   // determine price parameters
   m_smbinf.RefreshRates(); m_smbinf.Refresh();
   StopLvl = m_smbinf.StopsLevel()*m_smbinf.Point(); // remember stop level
   apr     = ReversPrice(dir);
   op      = BasePrice(dir);                         // open price
   sl      = NormalSL(dir, op, apr, SL, StopLvl);    // stop loss
   tp      = NormalTP(dir, op, apr, TP, StopLvl);    // take profit

   // open position
   m_trade.PositionOpen(m_smb,(ENUM_ORDER_TYPE)dir,lot,op,sl,tp);
   ulong order = m_trade.ResultOrder();
   if(order<=0) return(0);                           // order ticket
   return(GetDealByOrder(order));                    // return deal ticket
  }
//------------------------------------------------------------------ GetDealByOrder
ulong CExpertAdvisor::GetDealByOrder(ulong order) // get deal ticket by order ticket
  {
   PositionSelect(m_smb);
   HistorySelectByPosition(PositionGetInteger(POSITION_IDENTIFIER));
   uint total=HistoryDealsTotal();
   for(uint i=0; i<total; i++)
     {
      ulong deal=HistoryDealGetTicket(i);
      if(order==HistoryDealGetInteger(deal,DEAL_ORDER))
         return(deal);                            // remember deal ticket
     }
   return(0);
  }
//------------------------------------------------------------------ CountProfit
double CExpertAdvisor::CountProfitByDeal(ulong ticket)  // position profit by deal ticket
  {
   CDealInfo deal; deal.Ticket(ticket);                 // deal ticket
   HistorySelect(deal.Time(),TimeCurrent());            // select all deals after this
   uint total  = HistoryDealsTotal();
   long pos_id = deal.PositionId();                     // get position id
   double prof = 0;
   for(uint i=0; i<total; i++)                          // find all deals with this id
     {
      ticket = HistoryDealGetTicket(i);
         if(HistoryDealGetInteger(ticket,DEAL_POSITION_ID)!=pos_id) continue;
      prof += HistoryDealGetDouble(ticket,DEAL_PROFIT); // summarize profit
     }
   return(prof);                                        // return profit
  }
```

_4\. Trade Modules_

Finally, this group of functions binds the entire process of trading, processing the signals and events, using the service functions and macro. Logical lexemes of trade operations are few, they depend on your specific objectives. However, we can distinguish the common concepts, that exist almost in all experts.

```
   virtual bool      Main();                            // main module controlling trade process
   virtual void      OpenPosition(long dir);            // module of opening position
   virtual void      CheckPosition(long dir);           // check position and open additional ones
   virtual void      ClosePosition(long dir);           // close position
   virtual void      BEPosition(long dir,int BE);       // moving Stop Loss to break-even
   virtual void      TrailingPosition(long dir,int TS); // trailing position of Stop Loss
   virtual void      OpenPending(long dir);             // module of opening pending orders
   virtual void      CheckPending(long dir);            // work with current orders and open additional ones
   virtual void      TrailingPending(long dir);         // move pending orders
   virtual void      DeletePending(long dir);           // delete pending orders
```

We will consider specific implementations of these functions in examples below.

Adding new functions won't be difficult, since we've chosen the right approach and composed the expert structure. If you'll use exactly this scheme, your designs will require minimum efforts and time, the code will be readable even after a year.

Of course, your experts are not limited to them. In the CExpertAdvisor class we've declared only the most necessary methods. You can add new handlers in [children classes](https://www.mql5.com/en/docs/basis/oop/inheritance), modify existing ones, expand your own modules, thus creating a single library. Having such a library, developing of "turnkey" experts takes from half an hour to two days.

### **4\. Examples of Using the CExpertAdvisor Class**

**4.1. Example of Work Based on the Indicator Signals**

As a first example, let's start with the simplest task - consider the MovingAverage Expert Advisor (basic example of MetaTrader 5) using the CExpertAdvisor class. Let's just complicate it a little.

Algorithm:

**a) Condition for position opening**

> - If price crosses the MA bottom-up, then open position for Buy.
> - If price crosses the MA top-down, then open position for Sell.
> - Set SL (Stop Loss), TP (TakeProfit).
> - Position lot is calculated by the Risk parameter - how many will lose from deposit when Stop Loss is triggered.

**b) Condition of position closing**

> - If price crosses the MA bottom-up, then close position for Sell.
> - If price crosses the MA top-down, then close position for Buy.

**c) Limitation**

> - Limiting the work of an expert by time from HourStart up to HourEnd daily.
> - Expert makes trade operations only on new bar.

**d) Position support**

> - Use simple trailing stop at a distance of TS.

For our expert we will need seven functions of the CExpertAdvisor class:

- Signal function - CheckSignal()
- Ticks filter - CheckNewBar()
- Time filter - CheckTime()
- Service function of opening positions - DealOpen()
- Three working modules - OpenPosition(), ClosePosition(), TrailingPosition()

THe CheckSignal() function and modules must be defined in a child class to solve specifically its task. We also need to add the initialization of indicator.

```
//+------------------------------------------------------------------+
//|                                              Moving Averages.mq5 |
//|                        Copyright 2010, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "ExpertAdvisor.mqh"

input double Risk      = 0.1; // Risk
input int    SL        = 100; // Stop Loss distance
input int    TP        = 100; // Take Profit distance
input int    TS        =  30; // Trailing Stop distance
input int    pMA       =  12; // Moving Average period
input int    HourStart =   7; // Hour of trade start
input int    HourEnd   =  20; // Hour of trade end
//---
class CMyEA : public CExpertAdvisor
  {
protected:
   double            m_risk;          // size of risk
   int               m_sl;            // Stop Loss
   int               m_tp;            // Take Profit
   int               m_ts;            // Trailing Stop
   int               m_pMA;           // MA period
   int               m_hourStart;     // Hour of trade start
   int               m_hourEnd;       // Hour of trade end
   int               m_hma;           // MA indicator
public:
   void              CMyEA();
   void             ~CMyEA();
   virtual bool      Init(string smb,ENUM_TIMEFRAMES tf); // initialization
   virtual bool      Main();                              // main function
   virtual void      OpenPosition(long dir);              // open position on signal
   virtual void      ClosePosition(long dir);             // close position on signal
   virtual long      CheckSignal(bool bEntry);            // check signal
  };
//------------------------------------------------------------------ CMyEA
void CMyEA::CMyEA() { }
//----------------------------------------------------------------- ~CMyEA
void CMyEA::~CMyEA()
  {
   IndicatorRelease(m_hma); // delete MA indicator
  }
//------------------------------------------------------------------ Init
bool CMyEA::Init(string smb,ENUM_TIMEFRAMES tf)
  {
   if(!CExpertAdvisor::Init(0,smb,tf)) return(false);  // initialize parent class

   m_risk=Risk; m_tp=TP; m_sl=SL; m_ts=TS; m_pMA=pMA;  // copy parameters
   m_hourStart=HourStart; m_hourEnd=HourEnd;

   m_hma=iMA(m_smb,m_tf,m_pMA,0,MODE_SMA,PRICE_CLOSE); // create MA indicator
   if(m_hma==INVALID_HANDLE) return(false);            // if there is an error, then exit
   m_bInit=true; return(true);                         // trade allowed
  }
//------------------------------------------------------------------ Main
bool CMyEA::Main()                            // main function
  {
   if(!CExpertAdvisor::Main()) return(false); // call function of parent class

   if(Bars(m_smb,m_tf)<=m_pMA) return(false); // if there are insufficient number of bars

   if(!CheckNewBar()) return(true);           // check new bar

   // check each direction
   long dir;
   dir=ORDER_TYPE_BUY;
   OpenPosition(dir); ClosePosition(dir); TrailingPosition(dir,m_ts);
   dir=ORDER_TYPE_SELL;
   OpenPosition(dir); ClosePosition(dir); TrailingPosition(dir,m_ts);

   return(true);
  }
//------------------------------------------------------------------ OpenPos
void CMyEA::OpenPosition(long dir)
  {
   if(PositionSelect(m_smb)) return;     // if there is an order, then exit
   if(!CheckTime(StringToTime(IntegerToString(m_hourStart)+":00"),
                 StringToTime(IntegerToString(m_hourEnd)+":00"))) return;
   if(dir!=CheckSignal(true)) return;    // if there is no signal for current direction
   double lot=CountLotByRisk(m_sl,m_risk,0);
   if(lot<=0) return;                    // if lot is not defined then exit
   DealOpen(dir,lot,m_sl,m_tp);          // open position
  }
//------------------------------------------------------------------ ClosePos
void CMyEA::ClosePosition(long dir)
  {
   if(!PositionSelect(m_smb)) return;                 // if there is no position, then exit
   if(!CheckTime(StringToTime(IntegerToString(m_hourStart)+":00"),
                 StringToTime(IntegerToString(m_hourEnd)+":00")))
     { m_trade.PositionClose(m_smb); return; }        // if it's not time for trade, then close orders
   if(dir!=PositionGetInteger(POSITION_TYPE)) return; // if position of unchecked direction
   if(dir!=CheckSignal(false)) return;                // if the close signal didn't match the current position
   m_trade.PositionClose(m_smb,1);                    // close position
  }
//------------------------------------------------------------------ CheckSignal
long CMyEA::CheckSignal(bool bEntry)
  {
   MqlRates rt[2];
   if(CopyRates(m_smb,m_tf,0,2,rt)!=2)
     { Print("CopyRates ",m_smb," history is not loaded"); return(WRONG_VALUE); }

   double ma[1];
   if(CopyBuffer(m_hma,0,0,1,ma)!=1)
     { Print("CopyBuffer MA - no data"); return(WRONG_VALUE); }

   if(rt[0].open<ma[0] && rt[0].close>ma[0])
      return(bEntry ? ORDER_TYPE_BUY:ORDER_TYPE_SELL); // condition for buy
   if(rt[0].open>ma[0] && rt[0].close<ma[0])
      return(bEntry ? ORDER_TYPE_SELL:ORDER_TYPE_BUY); // condition for sell

   return(WRONG_VALUE);                                // if there is no signal
  }

CMyEA ea; // class instance
//------------------------------------------------------------------ OnInit
int OnInit()
  {
   ea.Init(Symbol(),Period()); // initialize expert
   return(0);
  }
//------------------------------------------------------------------ OnDeinit
void OnDeinit(const int reason) { }
//------------------------------------------------------------------ OnTick
void OnTick()
  {
   ea.Main();                  // process incoming tick
  }
```

Let's parse the structure of the _Main()_ function. Conventionally it is divided into two parts.

In the first part the parent function is called. This function process the possible parameters that globally affect the work of an expert. These include checking the allowance to trade for an expert and validation of historical data.

In the second part the market events are directly processed.

The _CheckNewBar_ _() filter is tested_ \- checking a new bar. And modules for two directions of trade are called one after another.

In modules everything is organized pretty abstract (the second design principle). There is no direct address to the symbol properties. And three modules - _OpenPosition_ _()_, _ClosePosition_ _()_ and _TrailingPosition_ _()_ \- rely only on those parameters that come to them from the outside. This allows you to call these modules for verification of orders both for the Buy and for the Sell.

**4.2. Example of Using the CExpertAdvisor - Expert Without Indicators, Analyzing the Position State and Result**

To demonstrate let's take the system that trades only on the position reverse with lot increase after loss (this type of experts is usually called "Martingale")

**a) Place initial order**

-  when expert starts it opens the first position for Buy with initial lot

**b) Open subsequent positions**

- if previous position was closed in profit, then open position in the same direction with initial lot
- if previous position was closed with loss, open position in the opposite direction with a larger lot (using factor).


For our expert we will need three functions of the CExpertAdvisor class:

- open position - DealOpen()
- get profit value of closed position by deal ticket - CountProfitByDeal()
- working modules - OpenPosition(), CheckPosition()

Since the expert doesn't analyze any indicators, but only deals results, we'll use the [OnTrade()](https://www.mql5.com/en/docs/basis/function/events#ontrade) events for the optimal productivity. That is, the expert, that have once placed the first initial order for Buy, will place all subsequent orders only after the closing this position. So we will place the initial order in the [OnTick()](https://www.mql5.com/en/docs/basis/function/events#ontick) and will do all subsequent work in the [OnTrade()](https://www.mql5.com/en/docs/basis/function/events#ontrade).

The Init() function, as usual, simply initializes the parameters of the class with external parameters of the expert.

The OpenPosition() module opens the initial position and is blocked by the m\_first flag.

The CheckPosition() module controls the further reverses of position.

These modules are called from respective functions of the expert: OnTick() and OnTrade().

```
//+------------------------------------------------------------------+
//|                                                       eMarti.mq5 |
//|              Copyright Copyright 2010, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "ExpertAdvisor.mqh"
#include <Trade\DealInfo.mqh>

input double Lots    = 0.1; // Lot
input double LotKoef = 2;   // lot multiplier for loss
input int    Dist    = 60;  // distance to Stop Loss and Take Profit
//---
class CMartiEA : public CExpertAdvisor
  {
protected:
   double            m_lots;       // Lot
   double            m_lotkoef;    // lot multiplier for loss
   int               m_dist;       // distance to Stop Loss and Take Profit
   CDealInfo         m_deal;       // last deal
   bool              m_first;      // flag of opening the first position
public:
   void              CMartiEA() { }
   void             ~CMartiEA() { }
   virtual bool      Init(string smb,ENUM_TIMEFRAMES tf); // initialization
   virtual void      OpenPosition();
   virtual void      CheckPosition();
  };
//------------------------------------------------------------------ Init
bool CMartiEA::Init(string smb,ENUM_TIMEFRAMES tf)
  {
   if(!CExpertAdvisor::Init(0,smb,tf)) return(false); // initialize parent class
   m_lots=Lots; m_lotkoef=LotKoef; m_dist=Dist;       // copy parameters
   m_deal.Ticket(0); m_first=true;
   m_bInit=true; return(true);                        // trade allowed
  }
//------------------------------------------------------------------ OnTrade
void CMartiEA::OpenPosition()
  {
   if(!CExpertAdvisor::Main()) return;                       // call parent function
   if(!m_first) return;                                      // if already opened initial position
   ulong deal=DealOpen(ORDER_TYPE_BUY,m_lots,m_dist,m_dist); // open initial position
   if(deal>0) { m_deal.Ticket(deal); m_first=false; }        // if position exists
  }
//------------------------------------------------------------------ OnTrade
void CMartiEA::CheckPosition()
  {
   if(!CExpertAdvisor::Main()) return;           // call parent function
   if(m_first) return;                           // if not yet placed initial position
   if(PositionSelect(m_smb)) return;             // if position exists

   // check profit of previous position
   double lot=m_lots;                            // initial lot
   long dir=m_deal.Type();                       // previous direction
   if(CountProfitByDeal(m_deal.Ticket())<0)      // if there was loss
     {
      lot=NormalLot(m_lotkoef*m_deal.Volume());  // increase lot
      dir=ReversType(m_deal.Type());             // reverse position
     }
   ulong deal=DealOpen(dir,lot,m_dist,m_dist);   // open position
   if(deal>0) m_deal.Ticket(deal);               // remember ticket
  }

CMartiEA ea; // class instance
//------------------------------------------------------------------ OnInit
int OnInit()
  {
   ea.Init(Symbol(),Period()); // initialize expert
   return(0);
  }
//------------------------------------------------------------------ OnDeinit
void OnDeinit(const int reason) { }
//------------------------------------------------------------------ OnTick
void OnTick()
  {
   ea.OpenPosition();          // process tick - open first order
  }
//------------------------------------------------------------------ OnTrade
void OnTrade()
  {
   ea.CheckPosition();         // process trade event
  }
```

### 5\. Working With Events

In this article you have met examples of processing two events - [NewTick](https://www.mql5.com/en/docs/runtime/event_fire#newtick) and [Trade](https://www.mql5.com/en/docs/runtime/event_fire#trade) that were represented by the OnTick() and OnTrade() functions, respectively. In most cases, these two events are constantly used.

For experts, there are four functions of processing events:

- **[OnChartEvent](https://www.mql5.com/en/docs/basis/function/events#onchartevent)** processes a large group of events: when working with graphical objects, keyboard, mouse and custom events. For example, the function is used to create interactive experts or experts, built on the principle of graphical management of orders. Or just to create active controls of MQL-program parameters (using buttons and edit fields). In general, this function is used to process external event of an expert.

- **[OnTimer](https://www.mql5.com/en/docs/basis/function/events#ontimer)** is called when system timer event is processed. It is used in cases, when MQL-program require to analyze its environment on a regular basis, to calculate indicators values, when it is needed to continually refer to external sources of signals, etc. Roughly speaking, the OnTimer() function - is an alternative, even the best replacement for :

while(true) {  /\* perform analysis \*/; Sleep(1000); }.

I.e. the expert does not have to work in an endless loop on its start, but enough to move the calls of its functions from the OnTick() to OnTimer().
- **[OnBookEvent](https://www.mql5.com/en/docs/basis/function/events#onbookevent)** processes an event, that is generated when the Depth of Market changes its state. This event can be attributed to external and perform its processing in accordance with the task.
- **[OnTester](https://www.mql5.com/en/docs/basis/function/events#ontester)** is called after testing the expert on a given date range, before the OnDeinit() function for possible screening of test generations, when using genetic optimization by the Custom max parameter.

Do not forget that any events and their combinations are always advisable to use for solution of their specific task.

### Afterword

As you can see, writing an expert, while having the right scheme, does not take much time. Due to new possibilities of processing events in MQL5, we have more flexible structure of managing trading process. But all this stuff becomes a really powerful tool only if you've properly prepared your trading algorithms.

The article describes three main principles of their creation - **eventfulness, abstraction, modularity**. You will make your trade easier, if you'll base your experts on these "three pillars".

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/132](https://www.mql5.com/ru/articles/132)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/132.zip "Download all attachments in the single ZIP archive")

[emarti.mq5](https://www.mql5.com/en/articles/download/132/emarti.mq5 "Download emarti.mq5")(3.57 KB)

[emyea.mq5](https://www.mql5.com/en/articles/download/132/emyea.mq5 "Download emyea.mq5")(5.9 KB)

[expertadvisor.mqh](https://www.mql5.com/en/articles/download/132/expertadvisor.mqh "Download expertadvisor.mqh")(17.92 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Working with sockets in MQL, or How to become a signal provider](https://www.mql5.com/en/articles/2599)
- [SQL and MQL5: Working with SQLite Database](https://www.mql5.com/en/articles/862)
- [Getting Rid of Self-Made DLLs](https://www.mql5.com/en/articles/364)
- [Promote Your Development Projects Using EX5 Libraries](https://www.mql5.com/en/articles/362)
- [Using WinInet in MQL5. Part 2: POST Requests and Files](https://www.mql5.com/en/articles/276)
- [Tracing, Debugging and Structural Analysis of Source Code](https://www.mql5.com/en/articles/272)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/1862)**
(21)


![---](https://c.mql5.com/avatar/avatar_na2.png)

**[\-\-\-](https://www.mql5.com/en/users/sergeev)**
\|
16 Sep 2010 at 11:01

**echostate:**

In the function void CExpertAdvisor::TrailingPosition(long dir,int TS), there is one line:

sl=NormalSL(dir,apr,apr,TS,StopLvl);                     // calculate Stop Loss

Should we use apr for both the second and the third argument when calling NormalSL? I thought it should be:

sl=NormalSL(dir,op,apr,TS,StopLvl);

no.

the second and third argument must be apr.

because the calculation of tral is derived from the price at which the position will be closed. Bid for the Buy and Ask for Sell.  [function](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") is correct.

since the second argument should be the bid/ask price for "specified" direction (i.e., the variable op) rather than the "reverse" direction (i.e., the variable apr).

should be calculated from the "reverse" direction. In this case, apr.

![echostate](https://c.mql5.com/avatar/avatar_na2.png)

**[echostate](https://www.mql5.com/en/users/echostate)**
\|
17 Sep 2010 at 03:22

**sergeev:**

no.

the second and third argument must be apr.

because the calculation of tral is derived from the price at which the position will be closed. Bid for the Buy and Ask for Sell.  function is correct.

should be calculated from the "reverse" direction. In this case, apr.

Thanks for the quick reply! I thought I must be wrong.

Can I also ask in the function

```
double CExpertAdvisor::CountLotByRisk(int dist,double risk,double lot) // calculate lot by size of risk
  {
   if(dist==0 || risk==0) return(lot);
   m_smbinf.Refresh();
   return(NormalLot(AccountInfoDouble(ACCOUNT_BALANCE)*risk/(dist*10*m_smbinf.TickValue())));
  }
```

why we have a "10" between "dist" and "m\_smbinf.TickValue()" in the return value? I guess "dist" is the stop loss (in terms of pips), and "m\_smbinf.TickValue()" is the US dollar value per pip per lot for the [currency pair](https://www.mql5.com/en/blogs/tags/forexnews "Latest news from foreign exchange market"). So I'm not sure why we multiply another "10" in between them.

Thanks!

![NFTrader](https://c.mql5.com/avatar/2011/2/4D483C7E-2ED1.jpg)

**[NFTrader](https://www.mql5.com/en/users/nftrader)**
\|
10 Apr 2011 at 16:18

Thanks a million.

![Haitao Jiang](https://c.mql5.com/avatar/avatar_na2.png)

**[Haitao Jiang](https://www.mql5.com/en/users/jht)**
\|
10 Apr 2011 at 17:41

Very useful article. Thanks a lot!

![Simalb](https://c.mql5.com/avatar/2017/11/59FF5B41-9224.png)

**[Simalb](https://www.mql5.com/en/users/simalb)**
\|
20 Nov 2017 at 18:44

When I'll have enough money I'll [buy a robot](https://www.mql5.com/en/articles/498 "Article: How to buy and install a trading robot in the MetaTrader Market? ") .So I'll have more free time to do others works.

![20 Trade Signals in MQL5](https://c.mql5.com/2/0/20_Trading_Signals_MQL5__1.png)[20 Trade Signals in MQL5](https://www.mql5.com/en/articles/130)

This article will teach you how to receive trade signals that are necessary for a trade system to work. The examples of forming 20 trade signals are given here as separate custom functions that can be used while developing Expert Advisors. For your convenience, all the functions used in the article are combined in a single mqh include file that can be easily connected to a future Expert Advisor.

![Interview with Leonid Velichkovsky: "The Biggest Myth about Neural Networks is Super-Profitability" (ATC 2010)](https://c.mql5.com/2/0/25.png)[Interview with Leonid Velichkovsky: "The Biggest Myth about Neural Networks is Super-Profitability" (ATC 2010)](https://www.mql5.com/en/articles/525)

The hero of our interview Leonid Velichkovski (LeoV) has already participated in Automated Trading Championships. In 2008, his multicurrency neural network was like a bright flash in the sky, earning $110,000 in a certain moment, but eventually fell victim to its own aggressive money management. Two years ago, in his interview Leonid share his own trading experience and told us about the features of his Expert Advisor. On the eve of the ATC 2010, Leonid talks about the most common myths and misconceptions associated with neural networks.

![How to Quickly Create an Expert Advisor for Automated Trading Championship 2010](https://c.mql5.com/2/0/Fast_Expert_Advisor_Writing_MQL5.png)[How to Quickly Create an Expert Advisor for Automated Trading Championship 2010](https://www.mql5.com/en/articles/148)

In order to develop an expert to participate in Automated Trading Championship 2010, let's use a template of ready expert advisor. Even novice MQL5 programmer will be capable of this task, because for your strategies the basic classes, functions, templates are already developed. It's enough to write a minimal amount of code to implement your trading idea.

![How to Create Your Own Trailing Stop](https://c.mql5.com/2/0/Trailing_Stop_MQL5.png)[How to Create Your Own Trailing Stop](https://www.mql5.com/en/articles/134)

The basic rule of trader - let profit to grow, cut off losses! This article considers one of the basic techniques, allowing to follow this rule - moving the protective stop level (Stop loss level) after increasing position profit, i.e. - Trailing Stop level. You'll find the step by step procedure to create a class for trailing stop on SAR and NRTR indicators. Everyone will be able to insert this trailing stop into their experts or use it independently to control positions in their accounts.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/132&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071527435352091055)

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