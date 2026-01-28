---
title: Universal Expert Advisor: the Event Model and Trading Strategy Prototype (Part 2)
url: https://www.mql5.com/en/articles/2169
categories: Trading Systems, Integration
relevance_score: 6
scraped_at: 2026-01-23T11:50:41.623146
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/2169&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062760208805046367)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/2169#intro)
- [Event Model Based on Centralized Processing, ENUM\_MARKET\_EVENT\_TYPE](https://www.mql5.com/en/articles/2169#c1)
- [Access to Events Occurring on Other Instruments, the MarketEvent Structure](https://www.mql5.com/en/articles/2169#c2)
- [The "New Bar" Event. New Tick and Bar Detection Algorithms](https://www.mql5.com/en/articles/2169#c3)
- [The CPositionMT5 class — Basis of the Platform-Independent Algorithm](https://www.mql5.com/en/articles/2169#c4)
- [Trading Strategy Prototype — the CStrategy Class](https://www.mql5.com/en/articles/2169#c5)
- [Conclusion](https://www.mql5.com/en/articles/2169#exit)

### Introduction

The article contains further description of the universal trading engine CStrategy. In the first article [Universal Expert Advisor: Trading Modes of Strategies (Part 1)](https://www.mql5.com/en/articles/2166), we have discussed in detail trading modes and functions that allow implementing them. We have analyzed a universal Expert Advisor scheme consisting of four methods, two of which open new positions and the other two methods close them. Different combinations of method calls define a particular trading mode. For example, an Expert Advisor can be allowed only to Sell or to Buy, can manage previously opened positions or wait. Using these modes, an Expert Advisor operation can be flexibly set up depending on the trading time or day of the week.

However, the trading modes are not the only thing that an Expert Advisor may need. In the second part we will discuss the event model of the CStrategy trading mode based on the centralized event handling. The proposed event handling scheme differs from the system events in that all the events are gathered in one place. The advantages of such an implementation will be considered later.

Also, this article describes two important classes of the trading engine — **CStrategy** and **CPosition**. The first one is the core of the whole EA trading logic, it unites events and modes into a single flexible framework that the custom EA inherits directly. The second class is the basis of universal trading operations. It contains actions applied to an open position (like closing of a position or modification of its Stop Loss or Take Profit). This allows formalizing all trading actions and making them platform-independent.

Please note that all the specific features of working with the engine, which you will have to comply with, are ultimately created for your benefit. For example, the usual search through positions and access to system events are not available for the strategy. Therefore there is no need to worry about the sequence of actions and about which handler to use to handle the event. Instead, CStrategy offers a custom EA to focus on its trading logic by undertaking the implementation of these and many other operations.

Trading engines similar to the described one are designed for ordinary users allowing them to easily use the desired functionality. You do not need to analyze the details of the described algorithms. You should only understand the general principles and the functionality of CStrategy. Therefore, if you find some parts of the article difficult to understand, feel free to skip them.

### Event Model Based on Centralized Processing, ENUM\_MARKET\_EVENT\_TYPE

MetaTrader 5 provides a lot of events. These events include notifications of market price changes ( [NewTick](https://www.mql5.com/en/docs/runtime/event_fire#newtick), [BookEvent](https://www.mql5.com/en/docs/runtime/event_fire#bookevent)) and system events like [Timer](https://www.mql5.com/en/docs/runtime/event_fire#timer) or [TradeTransaction](https://www.mql5.com/en/docs/runtime/event_fire#tradetransaction). A system function of the same name with the On\* prefix is available for each event. This function is the _handler_ of this event. For example, if you want to handle the arrival of a new tick, add an appropriate set of procedures to the [OnTick](https://www.mql5.com/en/docs/basis/function/events#ontick) function:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Handling new tick arrival
   // ...
  }
```

When OnBookEvent occurs, another unit of the code that handles changes in the order book (Depth of Market) prices should be called:

```
void OnBookEvent (const string& symbol)
  {
   // Here we handle changes in the order book prices
   // ...
  }
```

With this approach, the more events are handled by our Expert Advisor, the more fragmented its logic becomes. Conditions for position opening and closing can be separated into different units. From the viewpoint of efficient programming, separation into different units can be a good solution, but this approach is undesirable in our case. On the contrary, a better solution is to gather the trading logic in one specially allocated place.

Moreover, some events are not supported in MetaTrader 5. In this case the Expert Advisor should detect the occurrence of conditions on its own. For example, MetaTrader 5 has no event that handles the opening of a new bar. Meanwhile, it is the most commonly used check performed by Expert Advisors. Therefore, the event model of the described trading engine supports not only system events, but also custom events (not to be confused with the user events on the chart), which greatly facilitates development of Expert Advisors. For example, one of such events is the creation of a new bar on the chart.

To understand the proposed event model, let us describe the events related to the price or time changes using the special enumeration **ENUM\_MARKET\_EVENT\_TYPE**:

```
//+------------------------------------------------------------------+
//| Determines the type of the market event.                         |
//+------------------------------------------------------------------+
enum ENUM_MARKET_EVENT_TYPE
  {
   MARKET_EVENT_TICK,               // Arrival of a new tick
   MARKET_EVENT_BAR_OPEN,           // Opening of a new bar
   MARKET_EVENT_TIMER,              // Triggered timer
   MARKET_EVENT_BOOK_EVENT          // Depth of Market changes (including arrival of a new tick).
  };
```

As you can see, the enumeration includes description of both system events and the event that is not supported directly in MetaTrader 5 (MARKET\_EVENT\_BAR\_OPEN — opening of a new bar of the EA's working symbol).

Suppose that our universal Expert Advisor has four methods of trading logic: InitBuy, InitSell, SupportBuy, SupportSell. We have described these methods in the [first part of the article](https://www.mql5.com/en/articles/2166) "Universal Expert Advisor". If one of the enumeration values is used as a parameter in these methods, the EA's processing logic can anytime find out the event, on the basis of which the method has been called. Let us describe a simplified scheme of the Expert Advisor:

```
//+------------------------------------------------------------------+
//|                                                     ExampExp.mq5 |
//|                                 Copyright 2015, Vasiliy Sokolov. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Vasiliy Sokolov."
#property link      "http://www.mql5.com"
#property version   "1.00"
#include <Strategy\Series.mqh>
ulong ExpertMagic=12345; // the magic number of the Expert Advisor
//+------------------------------------------------------------------+
//| Determines the type of the market event.                         |
//+------------------------------------------------------------------+
enum ENUM_MARKET_EVENT_TYPE
  {
   MARKET_EVENT_TICK,               //Arrival of a new tick of the current symbol
   MARKET_EVENT_BAR_OPEN,           // Opening of a new bar of the current instrument
   MARKET_EVENT_TIMER,              // Triggered timer
   MARKET_EVENT_BOOK_EVENT          // Depth of Market changes (including tick arrival).
  };
//+------------------------------------------------------------------+
//| Prototype of the Universal Expert Advisor.                       |
//+------------------------------------------------------------------+
class CExpert
  {
public:
   void              InitBuy(ENUM_MARKET_EVENT_TYPE &event_id);
   void              InitSell(ENUM_MARKET_EVENT_TYPE &event_id);
   void              SupportBuy(ENUM_MARKET_EVENT_TYPE &event_id);
   void              SupportSell(ENUM_MARKET_EVENT_TYPE &event_id);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CExpert::InitBuy(ENUM_MARKET_EVENT_TYPE &event_id)
  {
   printf(__FUNCTION__+" EventID: "+EnumToString(event_id));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CExpert::InitSell(ENUM_MARKET_EVENT_TYPE &event_id)
  {
   printf(__FUNCTION__+" EventID: "+EnumToString(event_id));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CExpert::SupportBuy(ENUM_MARKET_EVENT_TYPE &event_id)
  {
   printf(__FUNCTION__+" EventID: "+EnumToString(event_id));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CExpert::SupportSell(ENUM_MARKET_EVENT_TYPE &event_id)
  {
   printf(__FUNCTION__+" EventID: "+EnumToString(event_id));
  }

ENUM_MARKET_EVENT_TYPE event_type;
CExpert Expert;
datetime last_time;
CTime Time;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {
   EventSetTimer(1);
   return INIT_SUCCEEDED;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   event_type=MARKET_EVENT_TICK;
   CallExpertLogic(event_type);
   if(last_time!=Time[0])
     {
      event_type=MARKET_EVENT_BAR_OPEN;
      CallExpertLogic(event_type);
      last_time=Time[0];
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTimer()
  {
   event_type=MARKET_EVENT_TIMER;
   CallExpertLogic(event_type);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnBookEvent(const string &symbol)
  {
   event_type=MARKET_EVENT_TIMER;
   CallExpertLogic(event_type);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CallExpertLogic(ENUM_MARKET_EVENT_TYPE &event)
  {
   Expert.InitBuy(event);
   Expert.InitSell(event);
   Expert.SupportBuy(event);
   Expert.SupportSell(event);
  }
//+------------------------------------------------------------------+
```

The events that indicate market price changes call the same CallExpertMagic function, to which the current event is passed as a parameter. This function in turn calls the four methods of CExpert. When called, each of these methods prints its name and the identifier of the event that caused its call. If you run this Expert Advisor on a chart, appropriate records indicating that the event IDs have been processed will be displayed after some time:

```
2015.11.30 14:13:39.981 ExampExp2 (Si-12.15,M1) CExpert::SupportSell EventID: MARKET_EVENT_BAR_OPEN
2015.11.30 14:13:39.981 ExampExp2 (Si-12.15,M1) CExpert::SupportBuy EventID: MARKET_EVENT_BAR_OPEN
2015.11.30 14:13:39.981 ExampExp2 (Si-12.15,M1) CExpert::InitSell EventID: MARKET_EVENT_BAR_OPEN
2015.11.30 14:13:39.981 ExampExp2 (Si-12.15,M1) CExpert::InitBuy EventID: MARKET_EVENT_BAR_OPEN
2015.11.30 14:13:40.015 ExampExp2 (Si-12.15,M1) CExpert::InitSell EventID: MARKET_EVENT_TICK
2015.11.30 14:13:40.015 ExampExp2 (Si-12.15,M1) CExpert::InitBuy EventID: MARKET_EVENT_TICK
2015.11.30 14:13:39.981 ExampExp2 (Si-12.15,M1) CExpert::SupportSell EventID: MARKET_EVENT_BAR_OPEN
2015.11.30 14:13:39.981 ExampExp2 (Si-12.15,M1) CExpert::SupportBuy EventID: MARKET_EVENT_BAR_OPEN
2015.11.30 14:13:39.981 ExampExp2 (Si-12.15,M1) CExpert::InitSell EventID: MARKET_EVENT_BAR_OPEN
2015.11.30 14:13:39.981 ExampExp2 (Si-12.15,M1) CExpert::InitBuy EventID: MARKET_EVENT_BAR_OPEN
2015.11.30 14:13:39.981 ExampExp2 (Si-12.15,M1) CExpert::SupportSell EventID: MARKET_EVENT_TICK
2015.11.30 14:13:39.981 ExampExp2 (Si-12.15,M1) CExpert::SupportBuy EventID: MARKET_EVENT_TICK
2015.11.30 14:13:39.981 ExampExp2 (Si-12.15,M1) CExpert::InitSell EventID: MARKET_EVENT_TICK
2015.11.30 14:13:39.981 ExampExp2 (Si-12.15,M1) CExpert::InitBuy EventID: MARKET_EVENT_TICK
```

### Access to Events Occurring on Other Instruments, the MarketEvent Structure

When designing a trading system which analyzes multiple symbols, you need to create a mechanism that can track changes in the prices of multiple instruments. However, the standard OnTick function is only called for a new tick of the instrument the Expert Advisor is running on. On the other hand, trading system developers may use the [OnBookEvent](https://www.mql5.com/en/docs/basis/function/events#onbookevent) function that responds to changes in the order book (Depth of Market). Unlike OnTick, OnBookEvent is called for any change in the order book of the instrument, to which you subscribed using the [MarketBookAdd](https://www.mql5.com/en/docs/marketinformation/marketbookadd) function.

Changes in the order book happen very often, that is why monitoring this event is a resource-intensive procedure. As a rule, monitoring changes in the tick stream of the required symbol is enough for Expert Advisors. On the other hand, the event of the order book change also includes the arrival of a new tick. Apart from OnBookEvent, you can set up calls of [OnTimer](https://www.mql5.com/en/docs/basis/function/events#ontimer) at specified intervals and analyze price changes of multiple symbols in this function.

So in the system functions that react to NewTick, BookEvent and Timer events, you can add a call of some _intermediate module_ (let's call it EventProcessor), which would simultaneously analyze changes in prices of multiple instruments and generate an appropriate event. Each event would have a unified description in the form of a structure and would be sent by the control methods of the strategy. Having received an appropriate event as a structure, the strategy would either react to it or ignore. In this case, the system function which actually initiated the event for the final Expert Advisor would be unknown.

Indeed, if an Expert Advisor receives a notification of a new incoming tick, it does not matter whether the information is received through OnTick, OnTimer or OnBookEvent. The only thing that matters is that there is a new tick for the specified symbol. One event handler can be used for many strategies. For example, if each strategy is represented as a custom class, multiple instances of these classes can be stored in the special list of strategies. In this case, any strategy from the list will be able to receive a new event generated by EventProcessor. The following diagram shows how events are generated and sent:

![](https://c.mql5.com/2/21/3._nwqvvajs9uv9tjv_0xlfhx5__1.png)

Fig. 1. Diagram of event generation and sending

Now let us consider the actual structure that will be passed to Expert Advisors as an event. The structure is called **MarketEvent**, its definition is as follows:

```
//+------------------------------------------------------------------+
//| The structure defines the type of event, the instrument on which |
//| it occurred and the timeframe (for the BarOpen event)            |
//+------------------------------------------------------------------+
struct MarketEvent
  {
   ENUM_MARKET_EVENT_TYPE type;     // Event type.
   ENUM_TIMEFRAMES   period;        // The timeframe of the chart the event applies to (only for MARKET_EVENT_BAR_OPEN).
   string            symbol;        // The name of the symbol the event occurred on.
  };
```

After receiving an instance of the structure by reference, the method that makes a trading decision will be able to analyze it and make the right decision based on the following information:

- the type of the event
- the timeframe of the chart, on which the event has occurred
- the name of the instrument, on which the event has occurred

If analyzing the chart timeframe is useless for an event (e.g. NewTick or Timer), the _period_ field of the MarketEvent structure is always filled with the PERIOD\_CURRENT value.

### The "New Bar" Event. New Tick and Bar Detection Algorithms

In order to track new ticks and formation of new bars on multiple instruments, we will need to write appropriate module classes implementing this task. These modules are internal parts of the CStrategy class, and the user does not interact with them directly. The first module to consider is the **CTickDetector** class. The source code of the class is available below:

```
//+------------------------------------------------------------------+
//|                                              NewTickDetector.mqh |
//|                        Copyright 2015, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#include <Object.mqh>
//+------------------------------------------------------------------+
//| New tick detector                                                |
//+------------------------------------------------------------------+
class CTickDetector : public CObject
  {
private:
   string            m_symbol;         // The symbol to track new ticks for.
   MqlTick           m_last_tick;      // Last remembered tick.
public:
                     CTickDetector(void);
                     CTickDetector(string symbol);
   string            Symbol(void);
   void              Symbol(string symbol);
   bool              IsNewTick(void);
  };
//+------------------------------------------------------------------+
//| The constructor sets by default the current timeframe            |
//| and symbol.                                                      |
//+------------------------------------------------------------------+
CTickDetector::CTickDetector(void)
  {
   m_symbol=_Symbol;
  }
//+------------------------------------------------------------------+
//| Creates an object with a preset symbol and timeframe.            |
//+------------------------------------------------------------------+
CTickDetector::CTickDetector(string symbol)
  {
   m_symbol=symbol;
  }
//+------------------------------------------------------------------+
//| Sets the name of the symbol, on which formation of a new tick    |
//| should be tracked.                                               |
//+------------------------------------------------------------------+
void CTickDetector::Symbol(string symbol)
  {
   m_symbol=symbol;
  }
//+------------------------------------------------------------------+
//| Returns name of symbol on which emergence of a new tick          |
//| is tracked.                                                      |
//+------------------------------------------------------------------+
string CTickDetector::Symbol(void)
  {
   return m_symbol;
  }
//+------------------------------------------------------------------+
//| Returns true if for the given symbol and timeframe there is      |
//| a new tick.                                                      |
//+------------------------------------------------------------------+
bool CTickDetector::IsNewTick(void)
  {
   MqlTick tick;
   SymbolInfoTick(m_symbol,tick);
   if(tick.last!=m_last_tick.last ||
      tick.time!=m_last_tick.time)
     {
      m_last_tick=tick;
      return true;
     }
   return false;
  }
```

Its main used method is IsNewTick. It returns true if a new tick of the monitored symbol has been received. The financial instrument to monitor is set using the Symbol method. The CTickDetector class is derived from CObject. Therefore it can be added as an element of the CArrayObj collection. That's what we need. For example, we can create ten copies of CTickDetect, each of them will monitor its own symbol. By consistently referring to the collection of classes of the CArrayObj type, you can quickly find out the symbol, on which the new tick was formed, and then generate the corresponding event that would pass this information to the collection of Expert Advisors.

As already mentioned, in addition to new ticks, it is often necessary to determine emergence of new bars. It is best to deliver this task to the special **CBarDetector** class. It works similar to CTickDetector. The main method of CBarDetector is IsNewBar — the method returns true, if there is a new bar of the instrument, which name was previously specified using the Symbol method. Its source code is as follows:

```
//+------------------------------------------------------------------+
//|                                               NewBarDetecter.mqh |
//|                                 Copyright 2015, Vasiliy Sokolov. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Vasiliy Sokolov."
#property link      "http://www.mql5.com"
#include <Object.mqh>
//+------------------------------------------------------------------+
//| The class detects the emergence of a new bar of the specified    |
//| symbol and period.                                               |
//+------------------------------------------------------------------+
class CBarDetector : public CObject
  {
private:
   ENUM_TIMEFRAMES   m_timeframe;      // The timeframe to track the formation of a new bar
   string            m_symbol;         // The symbol to track the formation of a new bar
   datetime          m_last_time;      // The time of the last known bar
public:
                     CBarDetector(void);
                     CBarDetector(string symbol,ENUM_TIMEFRAMES timeframe);
   void              Timeframe(ENUM_TIMEFRAMES tf);
   ENUM_TIMEFRAMES   Timeframe(void);
   void              Symbol(string symbol);
   string            Symbol(void);
   bool              IsNewBar(void);
  };
//+------------------------------------------------------------------+
//| The constructor sets by default the current timeframe            |
//| and symbol.                                                      |
//+------------------------------------------------------------------+
CBarDetector::CBarDetector(void)
  {
   m_symbol=_Symbol;
   m_timeframe=Period();
  }
//+------------------------------------------------------------------+
//| Creates an object with a preset symbol and timeframe.            |
//+------------------------------------------------------------------+
CBarDetector::CBarDetector(string symbol,ENUM_TIMEFRAMES tf)
  {
   m_symbol=symbol;
   m_timeframe=tf;
  }
//+------------------------------------------------------------------+
//| Sets the timeframe on which you want to track emergence of       |
//| a new bar.                                                       |
//+------------------------------------------------------------------+
void CBarDetector::Timeframe(ENUM_TIMEFRAMES tf)
  {
   m_timeframe=tf;
  }
//+------------------------------------------------------------------+
//| Returns the timeframe on which you track the emergence of        |
//| a new bar.                                                       |
//+------------------------------------------------------------------+
ENUM_TIMEFRAMES CBarDetector::Timeframe(void)
  {
   return m_timeframe;
  }
//+------------------------------------------------------------------+
//| Sets the name of the symbol, on which formation of a new bar     |
//| should be tracked.                                               |
//+------------------------------------------------------------------+
void CBarDetector::Symbol(string symbol)
  {
   m_symbol=symbol;
  }
//+------------------------------------------------------------------+
//| Returns name of symbol on which formation of a new bar           |
//| is tracked.                                                      |
//+------------------------------------------------------------------+
string CBarDetector::Symbol(void)
  {
   return m_symbol;
  }
//+------------------------------------------------------------------+
//| Returns true if for the given symbol and timeframe there is      |
//| a new bar.                                                       |
//+------------------------------------------------------------------+
bool CBarDetector::IsNewBar(void)
  {
   datetime time[];
   if(CopyTime(m_symbol, m_timeframe, 0, 1, time) < 1)return false;
   if(time[0] == m_last_time)return false;
   return m_last_time = time[0];
  }
```

### The CPositionMT5 Class — the Basis of the Platform-Independent Algorithm

Now it is time to analyze one of the most important classes providing operation of universal Expert Advisors. This class includes methods for working with positions in MetaTrader 5. From a technical point of view, it is very simple. It is a wrapper class which acts as an intermediary between an Expert Advisors and system functions related to operations with positions in MetaTrader 5 — [PositionSelect](https://www.mql5.com/en/docs/trading/positionselect) and PositionGet. However, it implements an important function of the offered classes — _platform independence_.

By carefully analyzing all the modules described above, we can conclude that they do not use functions that are only specific to one trading platform (MetaTrader 4 or MetaTrader 5). This is true because modern versions of MQL4 and MQL5 are, in fact, one and the same programming language with different sets of functions. Specific sets of functions for the two platforms are mainly connected with the management of trading positions.

Naturally, all Expert Advisors use position management functions. However, if instead of these functions, an Expert Advisor used a single abstract interface in the form of a position class, it would be possible (at least theoretically) to develop a trading robot that could be compiled both for MetaTrader 4 and MetaTrader 5 without any changes in its source code. All we need to do is develop a few alternate classes that implement work with positions. One class will use MetaTrader 4 functions, the other one will use functions from MetaTrader 5. However, both of these classes will provide the same set of methods for the final Expert Advisor, thereby providing "platform independence".

In practice, however, the problem of creation of a platform-independent Expert Advisor is somewhat more complicated and requires a separate article. This would be quite a lengthy material, so we will not discuss it in this article, it will be considered in other parts.

The second argument in favor of the special class representing an open position of the strategy is that each position is managed individually. The trading engine goes through the list of positions and passes each of them to the Expert Advisor's logic. Thus, we achieve maximum flexibility: the Expert Advisor manages each position individually and thus it becomes possible to describe rules for strategies that manage more than one trading position. Of course, you can have only one position in MetaTrader 5. But if we port the trading engine to the MetaTrader 4 platform, this features become very important.

Here is the source code of this class. It is simple and straightforward:

```
//+------------------------------------------------------------------+
//|                                                  PositionMT5.mqh |
//|                                 Copyright 2015, Vasiliy Sokolov. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Vasiliy Sokolov."
#property link      "http://www.mql5.com"
#include <Object.mqh>
#include "Logs.mqh"
#include <Trade\Trade.mqh>
//+------------------------------------------------------------------+
//| Active position class for classical strategies                   |
//+------------------------------------------------------------------+
class CPositionMT5 : public CObject
  {
private:
   ulong             m_id;                // A uniques position identifier
   uint              m_magic;             // A unique identifier if the Expert Advisor to the position belongs to.
   ENUM_POSITION_TYPE m_direction;         // Position direction
   double            m_entry_price;       // Position entry price
   string            m_symbol;            // The symbol the position is open for
   datetime          m_time_open;         // Open time
   string            m_entry_comment;     // Incoming comment
   bool              m_is_closed;         // True if the position has been closed
   CLog*             Log;                 // Logging
   CTrade            m_trade;             // Trading module
public:
                     CPositionMT5(void);
   uint              ExpertMagic(void);
   ulong             ID(void);
   ENUM_POSITION_TYPE Direction(void);
   double            EntryPrice(void);
   string            EntryComment(void);
   double            Profit(void);
   double            Volume(void);
   string            Symbol(void);
   datetime          TimeOpen(void);
   bool              CloseAtMarket(string comment="");
   double            StopLossValue(void);
   bool              StopLossValue(double sl);
   double            TakeProfitValue(void);
   bool              TakeProfitValue(double tp);
  };
//+------------------------------------------------------------------+
//| Initialization of the basic properties of a position             |
//+------------------------------------------------------------------+
void CPositionMT5::CPositionMT5(void) : m_id(0),
                                        m_entry_price(0.0),
                                        m_symbol(""),
                                        m_time_open(0)
  {
   m_id=PositionGetInteger(POSITION_IDENTIFIER);
   m_magic=(uint)PositionGetInteger(POSITION_MAGIC);
   m_direction=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   m_entry_price=PositionGetDouble(POSITION_PRICE_OPEN);
   m_symbol=PositionGetString(POSITION_SYMBOL);
   m_time_open=(datetime)PositionGetInteger(POSITION_TIME);
   m_entry_comment=PositionGetString(POSITION_COMMENT);
   m_trade.SetExpertMagicNumber(m_magic);
  }
//+------------------------------------------------------------------+
//| Returns position direction.                                      |
//+------------------------------------------------------------------+
ENUM_POSITION_TYPE CPositionMT5::Direction(void)
  {
   return m_direction;
  }
//+------------------------------------------------------------------+
//| Returns the unique ID of the Expert Advisor                      |
//| the current position belongs to.                                 |
//+------------------------------------------------------------------+
uint CPositionMT5::ExpertMagic(void)
  {
   return m_magic;
  }
//+------------------------------------------------------------------+
//| Returns the unique position identifier.                          |
//+------------------------------------------------------------------+
ulong CPositionMT5::ID(void)
  {
   return m_id;
  }
//+------------------------------------------------------------------+
//| Returns the position entry price.                                |
//+------------------------------------------------------------------+
double CPositionMT5::EntryPrice(void)
  {
   return m_entry_price;
  }
//+------------------------------------------------------------------+
//| Returns the incoming comment of an active position.              |
//+------------------------------------------------------------------+
string CPositionMT5::EntryComment(void)
  {
   return m_entry_comment;
  }
//+------------------------------------------------------------------+
//| Returns the name of the symbol for which there is currently open |
//| position.                                                        |
//+------------------------------------------------------------------+
string CPositionMT5::Symbol(void)
  {
   return m_symbol;
  }
//+------------------------------------------------------------------+
//| Returns position open time.                                      |
//+------------------------------------------------------------------+
datetime CPositionMT5::TimeOpen(void)
  {
   return m_time_open;
  }
//+------------------------------------------------------------------+
//| Returns an absolute Stop Loss level for the current position.    |
//| If the Stop Loss level is not set, returns 0.0                   |
//+------------------------------------------------------------------+
double CPositionMT5::StopLossValue(void)
  {
   if(!PositionSelect(m_symbol))
      return 0.0;
   return PositionGetDouble(POSITION_SL);
  }
//+------------------------------------------------------------------+
//| Sets an absolute stop loss level                                 |
//+------------------------------------------------------------------+
bool CPositionMT5::StopLossValue(double sl)
  {
   if(!PositionSelect(m_symbol))
      return false;
   return m_trade.Buy(0.0, m_symbol, 0.0, sl, TakeProfitValue(), NULL);
  }
//+------------------------------------------------------------------+
//| Returns an absolute Stop Loss level for the current position.    |
//| If the Stop Loss level is not set, returns 0.0                   |
//+------------------------------------------------------------------+
double CPositionMT5::TakeProfitValue(void)
  {
   if(!PositionSelect(m_symbol))
      return 0.0;
   return PositionGetDouble(POSITION_TP);
  }
//+------------------------------------------------------------------+
//| Sets an absolute stop loss level                                 |
//+------------------------------------------------------------------+
bool CPositionMT5::TakeProfitValue(double tp)
  {
   if(!PositionSelect(m_symbol))
      return false;
   return m_trade.Buy(0.0, m_symbol, 0.0, StopLossValue(), tp, NULL);
  }
//+------------------------------------------------------------------+
//| Closes the current position by market and sets a closing         |
//| comment equal to 'comment'                                       |
//+------------------------------------------------------------------+
bool CPositionMT5::CloseAtMarket(string comment="")
  {
   if(!PositionSelect(m_symbol))
      return false;
   return m_trade.PositionClose(m_symbol);
  }
//+------------------------------------------------------------------+
//| Returns current position volume.                                 |
//+------------------------------------------------------------------+
double CPositionMT5::Volume(void)
  {
   if(!PositionSelect(m_symbol))
      return false;
   return PositionGetDouble(POSITION_VOLUME);
  }
//+------------------------------------------------------------------+
//| Returns current profit of a position in deposit currency.        |
//+------------------------------------------------------------------+
double CPositionMT5::Profit(void)
  {
   if(!PositionSelect(m_symbol))
      return false;
   return PositionGetDouble(POSITION_PROFIT);
  }
```

As you can see, the main task of the class is to return a particular property of the currently open positions. Also, the class provides several methods for managing the current position: closing, changing its Take Profit and Stop Loss.

### Trading Strategy Prototype — the CStrategy Class

We have considered a number of modules that perform common tasks. We have also analyzed the event model, the decision-making model, and the action algorithm of a typical Expert Advisor. Now we need to combine the information into a single module of Expert Advisors — the **CStrategy** class. It is to perform various tasks. Here are some of them:

- Arranging a single sequence of trading actions for all the strategies that are based on CStrategy.
- Identifying the Timer, NewTick, BookEvent, NewBar events and passing them to a custom strategy.
- Implementing easy access to symbol quotes based on the MetaTrader 4 model.
- Managing trading modes.
- Accessing information about open positions and receiving statistics on them.

CStrategy is quite s large class, so we will not publish its full source code here. Instead, we will focus on the most important algorithms of the class. The first thing we will consider is its event model. We already know about this model. Now we only need to consider a process of receiving events and delivering them to a certain strategy. CStrategy is a parent class of our future strategy. Consequently, we are familiar with its methods that notify of receipt of appropriate events:

```
//+------------------------------------------------------------------+
//| The basic class of the layer strategy.                           |
//+------------------------------------------------------------------+
class CStrategy : public CObject
  {
   //...
   void              OnTick(void);
   void              OnTimer(void);
   void              OnBookEvent(string symbol);
   virtual void      OnTradeTransaction(const MqlTradeTransaction &trans,
                                        const MqlTradeRequest &request,
                                        const MqlTradeResult &result);
   //...
  };
```

Please note that all event handlers except OnTradeTransaction _are not virtual_. This means that Tick, Timer or BookEvent cannot be handled directly at the level of the strategy. Instead, they are handled by CStrategy. The TradeTransaction event is not used in this class, therefore it is virtual.

Below are contents of OnTick, OnTimer and OnBookEvent:

```
//+------------------------------------------------------------------+
//| Called by the strategy manager upon the system event             |
//| 'new tick'.                                                      |
//+------------------------------------------------------------------+
void CStrategy::OnTick(void)
  {
   NewTickDetect();
   NewBarsDetect();
  }
//+------------------------------------------------------------------+
//| Called by the strategy manager upon the system event             |
//| 'OnTimer'.                                                       |
//+------------------------------------------------------------------+
void CStrategy::OnTimer(void)
  {
   m_event.symbol=Symbol();
   m_event.type=MARKET_EVENT_TIMER;
   m_event.period=(ENUM_TIMEFRAMES)Period();
   CallSupport(m_event);
   CallInit(m_event);
   NewTickDetect();
   NewBarsDetect();
  }
//+------------------------------------------------------------------+
//| Called by the strategy manager upon the system event             |
//| 'OnBookEvent'.                                                   |
//+------------------------------------------------------------------+
void CStrategy::OnBookEvent(string symbol)
  {
   m_event.symbol=symbol;
   m_event.type=MARKET_EVENT_BOOK_EVENT;
   m_event.period=PERIOD_CURRENT;
   CallSupport(m_event);
   CallInit(m_event);
   NewTickDetect();
   NewBarsDetect();
  }
```

As mentioned above, the NewTick system event only applies to the instrument, on which the current Expert Advisor is running. If we want NewTick to be a multiple-symbol event, we should use a special handler of new ticks and track the emergence of new bars. This task is delegated to appropriate closed methods: NewBarDetect and NewTickDetect. Below is their source code:

```
//+------------------------------------------------------------------+
//| Detects emergence of a new bar and generates an appropriate      |
//| event for the Expert Advisor.                                    |
//+------------------------------------------------------------------+
void CStrategy::NewBarsDetect(void)
  {
   if(m_bars_detecors.Total()==0)
      AddBarOpenEvent(ExpertSymbol(),Timeframe());
   for(int i=0; i<m_bars_detecors.Total(); i++)
     {
      CBarDetector *bar=m_bars_detecors.At(i);
      if(bar.IsNewBar())
        {
         m_event.period = bar.Timeframe();
         m_event.symbol = bar.Symbol();
         m_event.type=MARKET_EVENT_BAR_OPEN;
         CallSupport(m_event);
         CallInit(m_event);
        }
     }
  }
//+------------------------------------------------------------------+
//| Detects the arrival of new ticks of multi-instruments.           |
//+------------------------------------------------------------------+
void CStrategy::NewTickDetect(void)
  {
   if(m_ticks_detectors.Total()==0)
      AddTickEvent(ExpertSymbol());
   for(int i=0; i<m_ticks_detectors.Total(); i++)
     {
      CTickDetector *tick=m_ticks_detectors.At(i);
      if(tick.IsNewTick())
        {
         m_event.period=PERIOD_CURRENT;
         m_event.type=MARKET_EVENT_TICK;
         m_event.symbol=tick.Symbol();
         CallSupport(m_event);
         CallInit(m_event);
        }
     }
  }
```

The methods actually check detectors of new ticks and bars (the CTickDetector and CBarDetector classes described above). Each of these detectors is first configured for the financial instrument the Expert Advisor actually trades. If a new tick or bar appears, special CallSupport and CallInit methods are called, which then call trading methods of the specific strategy.

The general algorithm of event handlers, including NewBarsDetect and NewTickDetect, is as follows:

1. detecting the occurrence of a new event;
2. filling the MarketEvent structure with a specification of the appropriate event ID and its attributes;
3. calling the CallSupport method, which then calls the virtual SupportBuy and SupportSell with the event passed to the method;
4. similarly, calling the CallInit method, which then calls the virtual InitBuy and InitSell with the event passed to the method.

To understand how the control is passed to a certain strategy, we need to consider the CallInit and CallSupport methods. Here is the source code of CallInit:

```
//+------------------------------------------------------------------+
//| Calls position opening logic provided that the trading           |
//| state does not explicitly restrict this.                         |
//+------------------------------------------------------------------+
void CStrategy::CallInit(const MarketEvent &event)
  {
   m_trade_state=m_state.GetTradeState();
   if(m_trade_state == TRADE_STOP)return;
   if(m_trade_state == TRADE_WAIT)return;
   if(m_trade_state == TRADE_NO_NEW_ENTRY)return;
   SpyEnvironment();
   if(m_trade_state==TRADE_BUY_AND_SELL || m_trade_state==TRADE_BUY_ONLY)
      InitBuy(event);
   if(m_trade_state==TRADE_BUY_AND_SELL || m_trade_state==TRADE_SELL_ONLY)
      InitSell(event);
  }
```

The method receives the trading state from the _m\_state_ module (of the CTradeState class described in the [first article](https://www.mql5.com/en/articles/2166#c3)) and decides whether it is possible to call new position initialization methods for the given trading state. If the trading state prohibits trading, these methods are not called. Otherwise they are called with an indication of the event, which caused the call of CallInit the method.

CallSupport works similarly, but its logic is somewhat different. Here is the source code:

```
//+------------------------------------------------------------------+
//| Calls position maintenance logic provided that the trading       |
//| state isn't equal to TRADE_WAIT.                                 |
//+------------------------------------------------------------------+
void CStrategy::CallSupport(const MarketEvent &event)
  {
   m_trade_state=m_state.GetTradeState();
   if(m_trade_state == TRADE_WAIT)return;
   SpyEnvironment();
   for(int i=ActivePositions.Total()-1; i>=0; i--)
     {
      CPosition *pos=ActivePositions.At(i);
      if(pos.ExpertMagic()!=m_expert_magic)continue;
      if(pos.Direction()==POSITION_TYPE_BUY)
         SupportBuy(event,pos);
      else
         SupportSell(event,pos);
      if(m_trade_state==TRADE_STOP && pos.IsActive())
         ExitByStopRegim(pos);
     }
  }
```

In a similar manner, the method receives the current trading state of the Expert Advisor. If it allows moving on to position management, the method will start to search through all currently open position. These positions are represented by the special CPosition class. Having received access to each position, the method compares its magic number with the magic number of the current Expert Advisor. Once a match is found, appropriate position management methods are called: SupportBuy for a long position and SupportSell for a short one.

### Conclusion

We have examined all the main modules of the CStrategy class, through which it provides extensive functionality for a custom strategy. A strategy derived from this class can operate with price quotes and trade events. Also, the strategy gets a unified sequence of trading action inherited from the CStrategy class. In this case, the strategy can be developed as a separate module, which only describes trading operations and rules. This solution helps to significantly reduce the effort invested into the development of the trading algorithm.

In the next article "Universal Expert Advisor: Custom Strategies and Auxiliary Trade Classes (Part 3)", we will discuss the process of strategy development based on the algorithms described.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2169](https://www.mql5.com/ru/articles/2169)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2169.zip "Download all attachments in the single ZIP archive")

[strategyarticle.zip](https://www.mql5.com/en/articles/download/2169/strategyarticle.zip "Download strategyarticle.zip")(100.23 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing graphical interfaces based on .Net Framework and C# (part 2): Additional graphical elements](https://www.mql5.com/en/articles/6549)
- [Developing graphical interfaces for Expert Advisors and indicators based on .Net Framework and C#](https://www.mql5.com/en/articles/5563)
- [Custom Strategy Tester based on fast mathematical calculations](https://www.mql5.com/en/articles/4226)
- [R-squared as an estimation of quality of the strategy balance curve](https://www.mql5.com/en/articles/2358)
- [Universal Expert Advisor: CUnIndicator and Use of Pending Orders (Part 9)](https://www.mql5.com/en/articles/2653)
- [Implementing a Scalping Market Depth Using the CGraphic Library](https://www.mql5.com/en/articles/3336)
- [Universal Expert Advisor: Accessing Symbol Properties (Part 8)](https://www.mql5.com/en/articles/3270)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/76930)**
(41)


![Alain Verleyen](https://c.mql5.com/avatar/2024/5/663a6cdf-e866.jpg)

**[Alain Verleyen](https://www.mql5.com/en/users/angevoyageur)**
\|
6 Jan 2017 at 20:11

**Amy Liu:**

This is why I want to learn classes in MetaEditor5, they're from "government". ;)

And bugged and not well supported :-D


![Metin Kostekci](https://c.mql5.com/avatar/avatar_na2.png)

**[Metin Kostekci](https://www.mql5.com/en/users/doksandokuzprof)**
\|
11 Feb 2018 at 14:47

thanks.Very good article.


![zemo](https://c.mql5.com/avatar/2018/2/5A79AE10-E9F2.JPG)

**[zemo](https://www.mql5.com/en/users/zemo)**
\|
29 Nov 2018 at 14:26

Mr Vasily,

very nice code... and useful to me...

in the mt5 news builds (1952), we got a "message" in the compiler,

```
bool CBarDetector::IsNewBar(void)

  {

   datetime time[];

   if(CopyTime(m_symbol, m_timeframe, 0, 1, time) < 1)return false;

   if(time[0] == m_last_time)return false;

   return (m_last_time = time[0]);    //<=============HERE

  }
```

//+------------------------------------------------------------------+

///////////MESSAGE in THE METAEDITOR compiler///////////////

expression not boolean NewBarDetector.mqh 87 24

the correct should be this? please confirm

```
//+------------------------------------------------------------------+
//| Returns true if for the given symbol and timeframe there is |
//| a new bar.|
//+------------------------------------------------------------------+
bool CBarDetector::IsNewBar(void)
  {
   datetime time[];
   if(CopyTime(m_symbol, m_timeframe, 0, 1, time) < 1)return (false);
   if(time[0] == m_last_time)return (false);
   return (m_last_time == time[0]);
  }
//+------------------------------------------------------------------+
```

![Shephard Mukachi](https://c.mql5.com/avatar/2017/5/5920C545-5DFD.jpg)

**[Shephard Mukachi](https://www.mql5.com/en/users/mukachi)**
\|
13 Dec 2018 at 03:50

Hi Vasiliy,

Please pardon me for asking a question on your article this far down after you wrote it. I'm only going through the articles properly now in search of alternatives to a framework. Something has struck me as odd, very likely due to my misunderstanding.

With regards to New Tick and New Bar event handlers. You loop through the list of added ticks, then build the event structure, passing it to Init and Support event handlers, e.g. new tick event below:

```
//+------------------------------------------------------------------+
//| Detects the arrival of new ticks of multi-instruments.           |
//+------------------------------------------------------------------+
void CStrategy::NewTickDetect(void)
  {
   if(m_ticks_detectors.Total()==0)
      AddTickEvent(ExpertSymbol());
   for(int i=0; i<m_ticks_detectors.Total(); i++)
     {
      CTickDetector *tick=m_ticks_detectors.At(i);
      if(tick.IsNewTick())
        {
         m_event.period=PERIOD_CURRENT;
         m_event.type=MARKET_EVENT_TICK;
         m_event.symbol=tick.Symbol();
         CallSupport(m_event);
         CallInit(m_event);
        }
     }
  }
```

In one of your examples, e.g. moving average clip below;

```
bool CMovingAverage::IsTrackEvents(const MarketEvent &event)
  {
//--- We handle only opening of a new bar on the working symbol and timeframe
   if(event.type != MARKET_EVENT_BAR_OPEN)return false;
   if(event.period != Timeframe())return false;
   if(event.symbol != ExpertSymbol())return false;
   return true;
  }
```

This IsTrackEvents [function](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") seems to nullify the purpose of NewTickDetect function above! So the Moving Average example above should be able to trade on multiple instruments based on its ability to check multiple symbols as in the NewTickDetect, but the IsTrackEvents allows trading only for the Strategy timeframe & Symbol (symbol being key here). Does this not mean that the NewTickDetect loop is not really required, as the strategy can only trade on its symbol? In effect the new tick detect should only check if the received tick is of the strategy symbol - without looping. Which in effect is similar to having a strategy object for each symbol of interest, which the CStragyList loops over?

I sure hope that I'm making sense, and hope that you can clarify this for me.

I love your work. I have learn't a lot from your articles, so a great many thanks.

Kind regards,

Shep

![Tolga Demir](https://c.mql5.com/avatar/2018/12/5C0E9BFD-30EC.jpg)

**[Tolga Demir](https://www.mql5.com/en/users/tolgademir)**
\|
16 Dec 2018 at 22:46

**Shephard Mukachi:**

Hi Vasiliy,

Please pardon me for asking a question on your article this far down after you wrote it. I'm only going through the articles properly now in search of alternatives to a framework. Something has struck me as odd, very likely due to my misunderstanding.

With regards to New Tick and New Bar event handlers. You loop through the list of added ticks, then build the event structure, passing it to Init and Support event handlers, e.g. new tick event below:

In one of your examples, e.g. moving average clip below;

This IsTrackEvents [function](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") seems to nullify the purpose of NewTickDetect function above! So the Moving Average example above should be able to trade on multiple instruments based on its ability to check multiple symbols as in the NewTickDetect, but the IsTrackEvents allows trading only for the Strategy timeframe & Symbol (symbol being key here). Does this not mean that the NewTickDetect loop is not really required, as the strategy can only trade on its symbol? In effect the new tick detect should only check if the received tick is of the strategy symbol - without looping. Which in effect is similar to having a strategy object for each symbol of interest, which the CStragyList loops over?

I sure hope that I'm making sense, and hope that you can clarify this for me.

I love your work. I have learn't a lot from your articles, so a great many thanks.

Kind regards,

Shep

![MetaTrader 5 features hedging position accounting system](https://c.mql5.com/2/22/hedge.png)[MetaTrader 5 features hedging position accounting system](https://www.mql5.com/en/articles/2299)

In order to expand possibilities of retail Forex traders, we have added the second accounting system — hedging. Now, it is possible to have multiple positions per symbol, including oppositely directed ones. This paves the way to implementing trading strategies based on the so-called "locking" — if the price moves against a trader, they can open a position in the opposite direction.

![Area method](https://c.mql5.com/2/21/area.png)[Area method](https://www.mql5.com/en/articles/2249)

The "area method" trading system works based on unusual interpretation of the RSI oscillator readings. The indicator that visualizes the area method, and the Expert Advisor that trades using this system are detailed here. The article is also supplemented with detailed findings of testing the Expert Advisor for various symbols, time frames and values of the area.

![Graphical Interfaces II: Setting Up the Event Handlers of the Library (Chapter 3)](https://c.mql5.com/2/22/Graphic-interface-part2__2.png)[Graphical Interfaces II: Setting Up the Event Handlers of the Library (Chapter 3)](https://www.mql5.com/en/articles/2204)

The previous articles contain the implementation of the classes for creating constituent parts of the main menu. Now, it is time to take a close look at the event handlers in the principle base classes and in the classes of the created controls. We will also pay special attention to managing the state of the chart depending on the location of the mouse cursor.

![Universal Expert Advisor: Trading Modes of Strategies (Part 1)](https://c.mql5.com/2/21/gu84ttj7g7r_klt2.png)[Universal Expert Advisor: Trading Modes of Strategies (Part 1)](https://www.mql5.com/en/articles/2166)

Any Expert Advisor developer, regardless of programming skills, is daily confronted with the same trading tasks and algorithmic problems, which should be solved to organize a reliable trading process. The article describes the possibilities of the CStrategy trading engine that can undertake the solution of these tasks and provide a user with convenient mechanism for describing a custom trading idea.

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/2169&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062760208805046367)

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