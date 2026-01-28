---
title: Universal Expert Advisor: Trading Modes of Strategies (Part 1)
url: https://www.mql5.com/en/articles/2166
categories: Trading Systems, Integration
relevance_score: 3
scraped_at: 2026-01-23T19:45:35.740570
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/2166&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070537569944410029)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/2166#intro)
- [Methods for Opening New Positions and Managing Existing Ones](https://www.mql5.com/en/articles/2166#c1)
- [Trading Modes of a Strategy](https://www.mql5.com/en/articles/2166#c2)
- [CTradeState Trading Mode Switch](https://www.mql5.com/en/articles/2166#c3)
- [Conclusion](https://www.mql5.com/en/articles/2166#exit)

### Introduction

Various tasks may arise while implementing automated trading algorithms, including analysis of the market environment to interpret market entry signals, and closing of an existing position. Another possible task is control over Expert Advisor operations and proper handling of trading errors. Finally, it is a task of easy and convenient access to market data and trading positions of the Expert Advisor. All these tasks are implemented directly in the Expert Advisor source code.

On the other hand, we should separate the technical part of the trading process and the idea implemented in the custom Expert Advisors. With the object-oriented approach, we can separate these two essentially different trading tasks and entrust implementation of the trading process to a special class common to all the strategies, which is sometimes also referred to as the **_trading engine_**.

This is the first article in the series of articles describing the operation of such engine, which can be called a Universal Expert Advisor. This name unifies a set of classes that enable easy development of trading algorithms by a usual enumeration of position entry and exit conditions. You will not need to add required data and trading logics to the Expert Advisor, e.g. position search — all the required procedures are done by the trading engine.

The material for the proposed article is extensive, therefore is divided into four parts. Here are the details of these parts.

**Part 1. Trading Modes of Strategies.** They are described in this article. The first part describes the original position management concept based on _trading modes_. An Expert Advisor trading logic can be easily defined using trading modes. An Expert Advisor written in this style is easy to debug. The logic of these EAs becomes universal and alike, which also facilitates management of such strategies. The ideas expressed in this material are universal and do not require additional object-oriented programming. This means that regardless of whether you will use the set of libraries offered or not, this material can be useful for you.

**Part 2. The Event Model and Trading Strategy Prototype.** This section describes an original event model based on centralized event handling. It means that all events are "gathered" in one place of the EA trading logic that processes them. Also, events are _multi-currency_. For example, if an Expert Advisor is running on the EURUSD chart, it is possible to receive an event of a new tick of GBPUSD. This event model can be extremely useful when developing Expert Advisors that trade multiple financial instruments. In this part, we will also describe the base class of the CStrategy trading engine and the CPositionMT5 class that represents a position in MetaTrader 5.

**Part 3. Custom Strategies and Auxiliary Trade Classes.** The material covers the process of custom Expert Advisor development. From this article you will find out how to create an Expert Advisor by a simple enumeration of position entry and exit conditions. This part also describes various auxiliary algorithms that can greatly simplify access to trading information.

**Part 4. Trading in a Group and Managing a Portfolio of Strategies.** This part contains a description of special algorithms for integrating several trading logics into a single executable ex5 module. It also describes mechanisms, which can be used to generate a set of custom strategies using a XML file.

### **Methods for Opening New Positions and Managing Existing Ones**

To understand the approach offered in this article, we will first try to describe a classical trading system based on two moving averages, one of which has a short averaging period, and the second one has a long period. Thus, the moving average with a large period of averaging is slower than the moving average with a smaller period of averaging. Trading rules are simple: if the fast moving average is above the slow one, the EA is to buy. Conversely, if the fast moving average is below the slow one, the EA is to sell. The following chart displays our strategy schematically:

![](https://c.mql5.com/2/21/1._cm4rx_MA__1.png)

Fig. 1. The chart of a trading system based on two moving averages

The red line shows the fast simple moving average with a period of 50. The blue line shows the slow moving average with a period of 120. When they intersect (intersections are marked with blue dotted lines), the direction of Expert Advisor position reverses. From the point of view of non-algorithmic approach, the description is enough for any trader to understand how to trade using this strategy. However, this description is not enough for creating an Expert Advisor based on this strategy.

Let's consider trading actions that the EA would need to perform at a time when the fast MA crosses the slow one from the bottom up:

1. If the EA has an open **short** position when the MAs intersect, this position should be closed.
2. The existence of an open **long** position should be checked. If there is no long position, one should be opened. If a long position already exists, nothing should be done.

For an opposite crossover when the fast MA crosses the slow one from top to bottom, opposite actions should be performed:

1. If the EA has an open **long** position when the MAs intersect, this position should be closed.
2. The existence of an open **short** position should be checked. If there is no short position, one should be opened. If a short position already exists, nothing should be done.

We have four trading actions to describe the trading process of the strategy. Two trading actions describe the long position opening and maintaining rules. Two other actions describe the short position opening and maintaining rules. It may seem that a four-action sequence is too much for the description of such a simple trading process. In fact, long position entries coincide with the short position exits in our strategy, so would not it be easier to combine them into one trading or at least logical action? No, it would not. To prove this, let's change conditions of our initial strategy.

Now our strategy will use different sets of Moving Averages for buys and sells. For example, a long position will be opened when the fast Moving Average with a period of 50 crosses the slow one with a period of 120. And a short position will be opened when the fast Moving Average with a period of 20 crosses the slow one with a period of 70. Now buy signals will differ from sell signals — they will occur at different times, in different market situations.

The proposed rules are not thought up. Strategies often use "mirror" conditions for entry and exit: entering a long position means exiting a short one and vice versa. However, other cases are also possible, and if we want to create a universal prototype of an Expert Advisor, we need to take this into account, so we will have four rules.

Further we will consider our actions from a different angle. The below table shows the trading operation type (Buy or Sell) and the trading action type (open or close). The table cells contains a specific set of actions:

|     |     |     |
| --- | --- | --- |
|  | **Buy** | **Sell** |
| **Open** | **1\. If there are no long positions and the fast MA with a period of 50 is above the slow MA with a period of 120, a long position should be opened** | **2\. If there are no short positions and the fast MA with a period of 20 is below the slow MA with a period of 70, a short position should be opened** |
| **Close** | **3\. If the fast MA with a period of 50 is below the slow MA with a period of 120, a long position should be closed** | **4\. If the fast MA with a period of 120 is above the slow MA with a period of 70, a short position should be closed** |

Table 1. Sets of trading action

From a programming perspective, these "sets of rules" or table blocks will be usual functions or methods that are part of the future class of universal strategies. We will name these four methods as follows:

- **BuyInit** — the method opens a new long positions if it is time to open a long position as per the conditions enumerated in it;
- **SellInit** — the method opens a new short positions if it is time to open a short position as per the conditions enumerated in it;
- **SupportBuy** — the method receives a long position as a parameter. If the passed position needs to be closed, the method should perform the appropriate trading action.
- **SupportSell** — the method receives a short position as a parameter. If the passed position needs to be closed, the method should perform the appropriate trading action.

What do we have from the proposed approach? First, we have classified trading actions that the EA needs to perform for the proper execution of a trade task. All actions are divided into separate independent blocks, i.e. usual class methods. This means that we will not need to think about where in the code we should handle different parts of trading logic. The programming task is reduced to the description of the four methods.

Second, if we ever need to change the logic of the Expert Advisor, we will only need to include additional conditions to appropriate methods. Third, the proposed arrangement of the trade logic will support simple and natural **_trading modes_** for any Expert Advisor developed in this style.

### **Trading Modes of a Strategy**

Very often, trading actions of an Expert Advisor need to be limited. The simplest example is to prevent the EA from performing short or long deals. MetaTrader 4 provides a standard switch of these modes. It is located directly on a tab of the EA properties window that appears at EA launch:

![](https://c.mql5.com/2/22/2.Trading_Modes.png)

Fig. 2. Trading modes in MetaTrader 4

However, even more modes are possible. Furthermore, we may need more flexible tools for configuring these modes. For example, some EAs need a pause in trading in certain moments of time. Suppose that during the Pacific session of the Forex market, the EA should ignore new position entry signals. This approach is a classic way to restrict EA trading during low volatility periods. What is the best way to implement this mode, additionally making it optional? This can be done through the four-block arrangement of trading logic.

Sell operations can be disabled for some time by temporary disabling calls of the SellInit method, which contains rules for opening short positions. It is because all trading actions initiating sell operations will be performed inside this method. The same applies to Buy operations: long positions will not open without calls of the BuyInit methods. Thus, certain combinations of calls of these methods will correspond to appropriate Expert Advisor trading modes. Describe these methods in Table 2:

| Trading mode | Description | The methods that are called | The methods whose calls are ignored |
| --- | --- | --- | --- |
| **Buy and Sell operations** | Buy and Sell operations are allowed. No trading limitations. | BuyInit<br> SellInit<br> BuySupport<br> SellSupport |  |
| **Buy operations only** | Only buy operations are allowed. No sell operations are performed. Previously opened short positions are managed in normal mode until closed. | BuyInit<br> BuySupport<br> SellSupport | SellInit |
| **Sell operations only** | Only sell operations are allowed. No buy operations are performed. Previously opened long positions are managed in normal mode until closed. | SellInit<br> BuySupport<br> SellSupport | BuyInit |
| **Now new entries** | Performing new buy and sell trades is not allowed. Previously opened short positions are managed in normal mode until closed by exit signals. | BuySupport<br> SellSupport | BuyInit<br>SellInit |
| **Pause** | Previously opened positions are not managed. Initialization of new buy and sell trades is paused. This mode is usually used when the market is closed and trading actions cannot be performed. |  | BuyInit<br> SellInit<br> BuySupport<br> SellSupport |
| **Stop** | All previously opened positions are closed. Buy and Sell operations are not initialized. | All positions are closed using a special method | BuyInit<br> SellInit<br> BuySupport<br> SellSupport |

Table 2. Expert Advisor Trading Modes

All trading modes are given through the practical implementation in MQL using a special structure **ENUM\_TRADE\_STATE**. Here is its description:

```
//+------------------------------------------------------------------+
//| Determines the trading state of the EA.                          |
//+------------------------------------------------------------------+
enum ENUM_TRADE_STATE
{
   TRADE_BUY_AND_SELL,              // Buy and sell operations are allowed.
   TRADE_BUY_ONLY,                  // Only buy operations are allowed. Sell operations not allowed.
   TRADE_SELL_ONLY,                 // Only sell operations are allowed. Buy operations are not allowed.
   TRADE_STOP,                      // Trading is not allowed. Close all positions immediately. Do not accept new entry signals.
   TRADE_WAIT,                      // Control over opened positions is lost. New signals are ignored. Useful during news releases.
   TRADE_NO_NEW_ENTRY               // Entry signals are ignored. However, opened positions are maintained according to the trading logic.
};
```

These modes allow any Expert Advisor developed under the proposed approach to flexibility connect and disconnect trading modules, thus to switch it to one or another trading mode "on the fly".

### **CTradeState trading mode switch**

Using trading modes, the Expert Advisor will always be able to understand at what point of time to perform certain actions. However, this point of time should be determined for each Expert Advisor individually. Trading mode control is particularly required when trading the FORTS section of MICEX. FORTS trading has several specific features, the main of which is clearing performed twice a day, from 14:00 to 14:03 (intermediate clearing) and from 18:45 to 19:00 (main clearing). It is advisable not to allow Expert Advisors to perform trading operations during clearing.

Of course, if an EA only performs operations with the arrival of new ticks or formation of new bars, it will not work while the market is closed, because no new quotes will be received. But many Expert Advisors operate at specified intervals (using a timer). For such EAs, control over trading actions is essential. In addition, sometimes trades can be performed on weekends and holidays, and some Forex brokers allow trading even on weekends. However, due to low volatility of such days, as well as their low statistical significance, these days should better be skipped.

Anyway, control over trading modes is a necessary procedure for any professional algorithmic trader. This task can be entrusted to the special **CTradeState** module. This module is implemented as an MQL5 class, and its task is to return the trading mode corresponding to the current time. For example, if the current time corresponds to the clearing time, the module will return the TRADE\_WAIT state. If it is time to close all positions, the module will return TRADE\_STOP. Let's describe its operation and configuration methods in more detail. Here is the header of this class:

```
//+------------------------------------------------------------------+
//|                                                  TimeControl.mqh |
//|                                 Copyright 2015, Vasiliy Sokolov. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Vasiliy Sokolov."
#property link      "http://www.mql5.com"
#include "Strategy.mqh"
#define ALL_DAYS_OF_WEEK 7
//+------------------------------------------------------------------+
//| Module of trading states TradeState                              |
//+------------------------------------------------------------------+
class CTradeState
{
private:
   ENUM_TRADE_STATE  m_state[60*24*7];  // Mask of trading states
public:
                     CTradeState(void);
                     CTradeState(ENUM_TRADE_STATE default_state);
   ENUM_TRADE_STATE  GetTradeState(void);
   ENUM_TRADE_STATE  GetTradeState(datetime time_current);
   void              SetTradeState(datetime time_begin, datetime time_end, int day_of_week, ENUM_TRADE_STATE state);
};
```

The main task of this class is to return to the current mode of the strategy, for which it is necessary to call its **GetTradeState** method. Before the module is able to return the state, this state should be added using the **SetTradeState** method.

The module operation algorithm is similar to the "Schedule" tab of the MetaTrader 5 testing agent:

![](https://c.mql5.com/2/22/fig3__1.png)

Fig. 3. The Schedule tab in the MetaTrader 5 testing agent

This window allows you to set the days of the week during which the agent can perform tasks from the MQL5 Cloud Network. The CTradeState class works in a similar way, but allows you to set one of the five values ​​of ENUM\_TRADE\_STATE for each range.

To better understand how to use CTradeState, let us configure the module of trading states. For daily operations on the FORTS market, the author of the article uses the following configuration presented as a table:

| Timing | Mode | Description |
| --- | --- | --- |
| 10:00-10:01 | TRADE\_WAIT | Time of market opening. The moment of opening is characterized by high volatility and price spikes. Trading activities in these moments are associated with high risk, so in the first minutes after session opening it is better to refrain from trading, for that the EA needs to be set to the waiting mode. |
| 14:00 - 14:03 | TRADE\_WAIT | Time Of Intermediate Clearing. In this time interval the market does not work, so the EA needs to be set to the TRADE\_WAIT mode as well. |
| 18:45 - 18:49 | TRADE\_WAIT | Time Of Main Clearing. At this time the market is also closed, and trading is disabled. The TRADE\_WAIT mode is active. |
| 23:50 - 9:59 | TRADE\_WAIT | Market is closed, trading is disabled. The EA mode is TRADE\_WAIT. |
| Friday, from 15:00 | TRADE\_NO\_NEW\_ENTRY | Friday — the last trading day of the week. In order not to leave open positions for the weekend, they need to be closed at the last day of trading. Therefore, there is no point in opening new positions on the last trading day just to close them a few hours later. For these very reasons the NO\_NEW\_ENTRY mode is used. Every Friday, starting from 15:00, new entry signals are ignored. The existing positions can only be closed. |
| Friday, 23:40-23:50 | TRADE\_STOP | The time before the market closes. This is the time when all the positions must be closed. EA switches to the TRADE\_STOP mode at 23:40, closes its open position and switches to the waiting mode. |
| Saturday, Sunday | TRADE\_WAIT | Trading is not performed during the weekend. Due to the transfer of holidays, some Saturdays may be work days. The Exchange is working on such days. This is a very rare case, and such "work" days must be avoided due to low volatility and statistical uncertainty. Trading on these days must be disabled regardless of whether it is a working day or not. |

Table 3. Trading modes depending on the time

As can be seen from the table, the required configuring is rather a challenging task, but the CTradeState class allows you to create such a combination of modes. Below is a sample script that sets modes from the table and then requests the mode that matches particular time:

```
//+------------------------------------------------------------------+
//|                                               TestTradeState.mq5 |
//|                                 Copyright 2015, Vasiliy Sokolov. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Vasiliy Sokolov."
#property link      "http://www.mql5.com"
#property version   "1.00"
#include <Strategy\TradeState.mqh>

CTradeState TradeState(TRADE_BUY_AND_SELL);  // Set the default mode to Buy And Sell
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
   TradeState.SetTradeState(D'15:00', D'23:39', FRIDAY, TRADE_NO_NEW_ENTRY);
   TradeState.SetTradeState(D'10:00', D'10:01', ALL_DAYS_OF_WEEK, TRADE_WAIT);
   TradeState.SetTradeState(D'14:00', D'14:03', ALL_DAYS_OF_WEEK, TRADE_WAIT);
   TradeState.SetTradeState(D'18:45', D'18:59', ALL_DAYS_OF_WEEK, TRADE_WAIT);
   TradeState.SetTradeState(D'23:50', D'23:59', ALL_DAYS_OF_WEEK, TRADE_WAIT);
   TradeState.SetTradeState(D'0:00',  D'9:59',  ALL_DAYS_OF_WEEK, TRADE_WAIT);
   TradeState.SetTradeState(D'23:40', D'23:49', FRIDAY, TRADE_STOP);
   TradeState.SetTradeState(D'00:00', D'23:59', SATURDAY, TRADE_WAIT);
   TradeState.SetTradeState(D'00:00', D'23:59', SUNDAY, TRADE_WAIT);

   printf("10:00 - " + EnumToString(TradeState.GetTradeState(D'10:00')));
   printf("14:01 - " + EnumToString(TradeState.GetTradeState(D'14:01')));
   printf("18:50 - " + EnumToString(TradeState.GetTradeState(D'18:50')));
   printf("23:50 - " + EnumToString(TradeState.GetTradeState(D'23:51')));
   printf("Friday, > 15:00 - " + EnumToString(TradeState.GetTradeState(D'2015.11.27 15:00')));
   printf("Saturday - " + EnumToString(TradeState.GetTradeState(D'2015.11.28')));
   printf("Sunday - " + EnumToString(TradeState.GetTradeState(D'2015.11.29')));
   printf("Default State - " + EnumToString(TradeState.GetTradeState(D'11:40')));
}
//+------------------------------------------------------------------+
```

The script output will be like this:

```
Default State - TRADE_BUY_AND_SELL
Sunday - TRADE_WAIT
Saturday - TRADE_WAIT
Friday, > 15:00 - TRADE_NO_NEW_ENTRY
23:50 - TRADE_STOP
18:50 - TRADE_WAIT
14:01 - TRADE_WAIT
10:00 - TRADE_WAIT
```

Please note the format the trading modes are set in. They do not use date components, only the hours and minutes (D'15:00' or D'18:40'). If pass full date to the method, e.g.:

```
TradeState.SetTradeState(D'2015.11.27 15:00', D'2015.11.27 23:39', FRIDAY, TRADE_NO_NEW_ENTRY);
```

the date component will still be ignored.

The second point to note is the sequence of SetTradeState calls. The sequence matters! The CTradeState module stores the mask of trading states as the ENUM\_TRADE\_STATE array, in which the number of elements is equal to the number of minutes in a week (10,080 elements). Using the passed dates, the SetTradeState method calculates the range of elements of this array and fills them with the appropriate state. This means that the previous state is replaced by a new one. Thus, the latest update is set as the final state. The code of this method is given below:

```
//+------------------------------------------------------------------+
//| Sets the trade state TradeState                                  |
//| INPUT:                                                           |
//| time_begin  - Time, from which the trading state is              |
//|               valid.                                             |
//| time_end    - Time, until which the trading state is valid       |
//| day_of_week - Day of the week, the setting of the trading        |
//|               state applies to. Corresponds to the modifiers     |
//|               ENUM_DAY_OF_WEEK or the modifier ALL_DAYS_OF_WEEK  |
//| state       - The trade state.                                   |
//| Note: date component in time_begin and time_end is ignored.      |
//+------------------------------------------------------------------+
void CTradeState::SetTradeState(datetime time_begin,datetime time_end, int day_of_week, ENUM_TRADE_STATE state)
{
   if(time_begin > time_end)
   {
      string sb = TimeToString(time_begin, TIME_MINUTES);
      string se = TimeToString(time_end, TIME_MINUTES);
      printf("Time " + sb + " must be more time " + se);
      return;
   }
   MqlDateTime btime, etime;
   TimeToStruct(time_begin, btime);
   TimeToStruct(time_end,  etime);
   for(int day = 0; day < ALL_DAYS_OF_WEEK; day++)
   {
      if(day != day_of_week && day_of_week != ALL_DAYS_OF_WEEK)
         continue;
      int i_day = day*60*24;
      int i_begin = i_day + (btime.hour*60) + btime.min;
      int i_end = i_day + (etime.hour*60) + etime.min;
      for(int i = i_begin; i <= i_end; i++)
         m_state[i] = state;
   }
}
```

GetTradeState works easier. It calculates the index of the array element that corresponds to the requested time, and then returns the element value:

```
//+------------------------------------------------------------------+
//| Returns the previously set trading state for the passed          |
//| time.                                                            |
//+------------------------------------------------------------------+
ENUM_TRADE_STATE CTradeState::GetTradeState(datetime time_current)
{
   MqlDateTime dt;
   TimeToStruct(time_current, dt);
   int i_day = dt.day_of_week*60*24;
   int index = i_day + (dt.hour*60) + dt.min;
   return m_state[index];
}
```

The full source code of the CTradeState class is available in the TradeState.mqh file, and is included in the source code of the described trading engine. The next articles will demonstrate how this class works in the trading engine.

### Conclusion

We have described the four main trading rules, with which you can easily and quickly define the logic of almost any Expert Advisor. Each trading rule is a separate function or a class method. Various combinations of method calls determine the particular mode of the strategy. Thus, a flexible Expert Advisor management system is implemented using minimal resources.

In the next part of this series, we will discuss a centralized event model — what makes the basic methods of the trading logic understand what trade event has occurred. We will also discuss auxiliary trading algorithms, which greatly facilitate obtaining of trade information.

You can download and install the full code of the "Universal Expert Advisor" library to your computer. The source code of the library is attached to this article. Description of more classes of this library will be given in the next articles of this series.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2166](https://www.mql5.com/ru/articles/2166)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2166.zip "Download all attachments in the single ZIP archive")

[strategyarticle.zip](https://www.mql5.com/en/articles/download/2166/strategyarticle.zip "Download strategyarticle.zip")(100.09 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/76457)**
(14)


![Pierre Rougier](https://c.mql5.com/avatar/2018/4/5AE5ECFE-3BFA.jpg)

**[Pierre Rougier](https://www.mql5.com/en/users/pierre8r)**
\|
18 Apr 2018 at 16:32

Hello,

The [source code](https://www.mql5.com/go?link=https://forge.mql5.io/help/en/guide "MQL5 Algo Forge: Cloud Workspace for Algorithmic Trading Development") of the article does not compile.

The error returned is:

cannot cast 'DoubleValue' to 'ULongValue'Dictionary.mqh21014

```
     lValue=(ULongValue)dValue;
```

Thanks for your help,

Pierre8r

![Pierre Rougier](https://c.mql5.com/avatar/2018/4/5AE5ECFE-3BFA.jpg)

**[Pierre Rougier](https://www.mql5.com/en/users/pierre8r)**
\|
13 Jun 2018 at 18:47

**Pierre Rougier:**

Hello,

The source code of the article does not compile.

The error returned is:

cannot cast 'DoubleValue' to 'ULongValue'Dictionary.mqh21014

Thanks for your help,

Pierre8r

I found how to compile the program.

Change the Dictionary.mqh file of this article by the Dictionary.mqh file of this article

[https://www.mql5.com/en/articles/2653](https://www.mql5.com/en/articles/2653)

![Livermore Ch](https://c.mql5.com/avatar/avatar_na2.png)

**[Livermore Ch](https://www.mql5.com/en/users/livermore2019)**
\|
15 Dec 2019 at 11:40

**MetaQuotes Software Corp.:**

New article [Generalised Intelligent Trading System: patterns of trading strategies (Chapter 1) has been](https://www.mql5.com/en/articles/2166) published:

Author: [Vasiliy Sokolov](https://www.mql5.com/en/users/C-4 "C-4")

![Livermore Ch](https://c.mql5.com/avatar/avatar_na2.png)

**[Livermore Ch](https://www.mql5.com/en/users/livermore2019)**
\|
15 Dec 2019 at 11:41

Where are the attachments, please?


![Andrew Thompson](https://c.mql5.com/avatar/2015/9/55FE3C61-771D.jpg)

**[Andrew Thompson](https://www.mql5.com/en/users/andydoc)**
\|
21 Dec 2024 at 12:12

Surely it would make more sense to only update the effected elements in m\_state so that the order of calling SetTradeState did not matter? Also, if this were implemented as part of strategy instead of or as well as engine, then different states could be assigned for different assets (eg particular forex pairs during relevent [news releases](https://www.mql5.com/en/economic-calendar "Economic indicators and events"))

![Area method](https://c.mql5.com/2/21/area.png)[Area method](https://www.mql5.com/en/articles/2249)

The "area method" trading system works based on unusual interpretation of the RSI oscillator readings. The indicator that visualizes the area method, and the Expert Advisor that trades using this system are detailed here. The article is also supplemented with detailed findings of testing the Expert Advisor for various symbols, time frames and values of the area.

![Graphical Interfaces II: the Separation Line and Context Menu Elements (Chapter 2)](https://c.mql5.com/2/22/Graphic-interface-part2__1.png)[Graphical Interfaces II: the Separation Line and Context Menu Elements (Chapter 2)](https://www.mql5.com/en/articles/2202)

In this article we will create the separation line element. It will be possible to use it not only as an independent interface element but also as a part of many other elements. After that, we will have everything required for the development of the context menu class, which will be also considered in this article in detail. Added to that, we will introduce all necessary additions to the class, which is the base for storing pointers to all the elements of the graphical interface of the application.

![Universal Expert Advisor: the Event Model and Trading Strategy Prototype (Part 2)](https://c.mql5.com/2/21/smyf67hqftm_kaz2.png)[Universal Expert Advisor: the Event Model and Trading Strategy Prototype (Part 2)](https://www.mql5.com/en/articles/2169)

This article continues the series of publications on a universal Expert Advisor model. This part describes in detail the original event model based on centralized data processing, and considers the structure of the CStrategy base class of the engine.

![Fuzzy logic to create manual trading strategies](https://c.mql5.com/2/22/2195.png)[Fuzzy logic to create manual trading strategies](https://www.mql5.com/en/articles/2195)

This article suggests the ways of improving manual trading strategy by applying fuzzy set theory. As an example we have provided a step-by-step description of the strategy search and the selection of its parameters, followed by fuzzy logic application to blur overly formal criteria for the market entry. This way, after strategy modification we obtain flexible conditions for opening a position that has a reasonable reaction to a market situation.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=inpbjpfthoilesbebwofzrlqulmrbcuf&ssn=1769186734308523825&ssn_dr=0&ssn_sr=0&fv_date=1769186734&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2166&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Universal%20Expert%20Advisor%3A%20Trading%20Modes%20of%20Strategies%20(Part%201)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918673441031438&fz_uniq=5070537569944410029&sv=2552)

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