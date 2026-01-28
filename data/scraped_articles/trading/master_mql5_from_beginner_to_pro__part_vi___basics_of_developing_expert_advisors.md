---
title: Master MQL5 from Beginner to Pro (Part VI): Basics of Developing Expert Advisors
url: https://www.mql5.com/en/articles/15727
categories: Trading
relevance_score: 9
scraped_at: 2026-01-22T17:25:58.442285
---

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/15727&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049128618262635892)

MetaTrader 5 / Examples


### Introduction

At last, we've reached the stage of creating Expert Advisors (EAs). In a way, we've crossed the Rubicon.

To make the most of this article, you should already be comfortable with the following concepts:

- Variables (local and global),
- Functions and their parameters (both by reference and by value),
- Arrays (including a basic understanding of series arrays),
- Core operators, including logical, arithmetic, conditional (if, switch, ternary), and loop operators (primarily 'for', but familiarity with 'while' and 'do...while' is also useful).

From a programmer's perspective, Expert Advisors aren't much more complex than indicators which we discussed in the previous article of this series. Trading logic similarly involves checking for certain conditions and, if those conditions are met, performing an action (typically sending a trade order to the server). The key is understanding the structure of trade orders, knowing the functions for sending those orders, and being able to access the necessary data for trading.

**Important**! All the Expert Advisors featured in this article are intended solely to illustrate programming principles and **not for real trading or generating profit**. If you plan to use this code on live accounts, you will likely need to refine the decision-making algorithms. Otherwise, you risk incurring losses.

In fact, the code for the EAs provided here isn't suitable for real trading even if the entry logic is improved. The first example doesn't include any error handling: neither for request submission nor for the server response. This is done intentionally to simplify the code, making it easier to understand, but it limits the EA to use in quick prototyping or testing basic strategy logic. The second EA includes a bit more validation... However, even that isn't sufficient for publishing on the Market or for reliable live trading, where problems must be _handled_ as they arise (if they can be handled at all, rather than simply being reported and shutting down).

A fully functional EA that could technically be accepted for publication on the Market will be covered in the next article. That EA will include the necessary validations and a slightly more complex logic than what we'll cover here. In this article, I focus on the _fundamentals_ of trading automation. We'll build two EAs: one without indicators, and another that uses the built-in Moving Average indicator. The first will trade using pending orders, while the second will execute trades at market price.

### Expert Advisor Template

Every Expert Advisor starts with the creation of a blank template, typically using the MQL5 Wizard (Figure 1).

![MQL Wizard - first screen](https://c.mql5.com/2/155/master-first-screen__1.png)

**Figure 1.** MQL Wizard - first screen

The MQL Wizard offers two main options: to create an Expert Advisor _from a template_ (top option), or to _generate_ a more advanced, structured version. For beginner programmers, I strongly recommend choosing the _first option_, i.e. the template.

The more _advanced generated_ version is object-oriented and split across multiple files. Without additional tools or experience, it can be quite challenging to understand, and even harder to customize for your own trading logic. That’s why I suggest only using this version _after_ you've gained a solid understanding of OOP (object-oriented programming) concepts and practices. Reviewing the generated code might be educational in terms of "seeing what's possible", but remember that it's just one of many possible implementations, optimized for automated generation. By the time you're ready to fully understand the intricacies of its class structure, you'll likely prefer to write your own code templates. Naturally, editing your own code is much easier than deciphering someone else's. And chances are, your own templates will be just as good as, if not better than, what the wizard provides.

Most of the optional functions that the wizard can add (Figures 2 and 3) aren't strictly necessary but are often very helpful. For example, functions from the third wizard screen (Figure 2) allow you to handle events triggered during trading operations, such as when the server receives a signal, a position is opened, etc. ( [OnTrade](https://www.mql5.com/en/docs/event_handlers/ontrade) and [OnTradeTransaction](https://www.mql5.com/en/docs/event_handlers/ontradetransaction)), as well as timer events ( [OnTimer](https://www.mql5.com/en/docs/event_handlers/ontimer)), chart interactions like button presses or object creation ( [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent)), and order book updates ( [OnBookEvent](https://www.mql5.com/en/docs/event_handlers/onbookevent)).

![Creating an Expert Advisor - third screen of the Wizard](https://c.mql5.com/2/155/master-third-screen-source__1.png)

**Figure 2.** Creating an Expert Advisor - third screen of the wizard (additional EA functions)

There are also special functions used exclusively within the strategy tester, but not during normal operation (Figure 3). These are mainly useful for demo versions that work only in the tester and not on live accounts. Sometimes, you may need more detailed logs during testing or want to fetch data from alternative sources. _Personally_, I use these functions rarely, but they can be valuable in the right context.

![EA functions for Strategy Tester](https://c.mql5.com/2/155/master-fourth-screen-source__1.png)

**Figure 3.** Creating an Expert Advisor - fourth screen of the wizard (functions for tester-only operation)

Some of the functions from Figure 2 will be discussed in more detail in future articles, while the ones in Figure 3 are left for you to explore on your own.

When creating an EA using the wizard, the generated file always includes at least three functions (Example 1):

```
//+------------------------------------------------------------------+
//|                                                  FirstExpert.mq5 |
//|                                       Oleg Fedorov (aka certain) |
//|                                   mailto:coder.fedorov@gmail.com |
//+------------------------------------------------------------------+
#property copyright "Oleg Fedorov (aka certain)"
#property link      "mailto:coder.fedorov@gmail.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

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

  }
//+------------------------------------------------------------------+
```

**Example 1.** Minimal template of the EA created by the wizard

- OnInit - a familiar function from indicator development, used for initial setup. It runs once _at the start_ of the program.

- OnDeinit - also likely familiar, this function is called when the Expert Advisor is _stopped_. It's intended for cleanup: removing graphical objects created by the EA, closing files, freeing indicator resources, and performing other finalization tasks.

- [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) \- executes _on every tick_(similar to OnCalculate in indicators). This is where the core logic of your EA runs.

Among these, only OnTick is mandatory for an Expert Advisor. You can omit OnInit and OnDeinit if your EA is simple.

### Key Terms in MetaTrader 5 Automated Trading

When developing automated trading systems in MetaTrader 5, there are certain terms and concepts every programmer must understand. These terms are associated with the trading logic and function names. Let's explore them.

**Order** – a message sent to the server indicating your intent to buy or sell a specific instrument at a specific price.

Every change in trading - whether it's placing a market order or modifying a Stop Loss level - happens through trade orders. Orders can be either immediate execution (e.g., buy/sell at market price) or pending, meaning the trade is intended to execute in the future when price conditions are met (such as stop or limit orders).

Pending orders include Stop Loss, Take Profit, Buy/Sell Stop, Buy/Sell Limit, and Buy/Sell Stop Limit. Example functions related to order handling include [OrderSend](https://www.mql5.com/en/docs/trading/ordersend) (sends an order to the server) and [OrderGetInteger](https://www.mql5.com/en/docs/trading/ordergetinteger) (returns integer parameters of the order, such as ticket number or creation time).

**Deal** – the actual execution of an order.

A deal in MetaTrader 5 is essentially related to history. It's the point at which an order is filled. You cannot influence a deal directly, as it occurs on the server once an order is executed. However, you can retrieve historical information about deals, such as execution price and time. Example functions: [HistoryDealGetDouble](https://www.mql5.com/en/docs/trading/historydealgetdouble) (gets a double parameter of a deal, like price), [HistoryDealsTotal](https://www.mql5.com/en/docs/trading/historydealstotal) (returns the total number of deals in the history).

**Position** \- the resulting status of your portfolio after one or more trades on a _specific symbol_.

MetaTrader 5 was originally designed so that each symbol could have only one position. However, actual behavior depends on the account type. On netting accounts, all trades update a single position. On hedging accounts, each trade creates its own position (unless it's a stop order). This can result in multiple positions on the same symbol and even in opposite directions. In such cases, you may need to calculate the net long and short exposure manually. Positions can be modified using trade orders - closed fully or partially, or adjusted in terms of Stop Loss and Take Profit levels. Examples of functions that work with positions include: [PositionSelectByTicket](https://www.mql5.com/en/docs/trading/positionselectbyticket) (select a position by its ticket) and [PositionGetString](https://www.mql5.com/en/docs/trading/positiongetstring) (get string parameters like symbol name or user comment).

Every deal results from the execution of a trade order, and every position reflects the cumulative effect of one or more deals.

**Event** – any significant occurrence in the program environment.

Submitting a trade request is an event. The server accepting the request, a user clicking the chart, a chart scaling change, a new tick - all of these are also events. Some of them are handled via standard handler functions that begin with On - like OnInit (triggered by the Init event) or OnTick (for Tick events).

Other event types are processed using constant identifiers. This means that one of the global event handler functions must be triggered first. For example, OnChartEvent is called for _any_ chart event. Inside this function, you determine the exact event type by comparing the event code variable to known constants. Parameters passed to these functions help identify the event details. This article won't go into the specifics of these smaller events.

### Basic Principles of Automated Trading in MetaTrader 5

Let's take a high-level look at how trading actually works in MetaTrader 5. We'll start with an important fact.

The terminal software running on your computer and the server software that executes trades with your money are two separate programs. They communicate _exclusively_ via the network (typically the internet).

So when you click "Buy" or "Sell," the following sequence of _events_ takes place:

- Your terminal generates a special data packet, filling in a special [MqlTradeRequest](https://www.mql5.com/en/book/automation/experts/experts_mqltraderequest) structure.
- This filled structure is sent to the server using OrderSend (synchronous mode) or [OrderSendAsync](https://www.mql5.com/en/docs/trading/ordersendasync) (asynchronous mode), forming a trade _order_.
- The server receives the packet and checks whether it meets all requirements: if matching prices are available, if your balance is sufficient, and so on.
- If everything is OK, the order is placed into the order queue alongside orders from other traders, waiting to be executed.
- A confirmation message is sent back to your terminal.

- If the market reaches the requested price level, the server _executes the deal_ and records the event in its log.
- The server sends the results back to your terminal.
- The terminal receives this result in the form of an [MqlTradeResult](https://www.mql5.com/en/book/automation/experts/experts_mqltraderesult) structure, and generates corresponding events like [Trade](https://www.mql5.com/en/docs/runtime/event_fire#trade) and [TradeTransaction](https://www.mql5.com/en/docs/runtime/event_fire#tradetransaction).

- The terminal checks for errors from the server side (done by checking the 'retcode' field of the MqlTradeResult structure).

- If everything is OK, the terminal updates its internal variables, log entries, and graphical chart.
- As a result, a new _position_ (or an updated one) appears in your portfolio for the relevant instrument.


This entire process can be visualized as a simplified diagram, as shown in Figure 4:

![The trading process is distributed between the terminal and the server](https://c.mql5.com/2/155/TradingProcess__2.png)

**Figure 4**. Trade order processing diagram

### Asynchronous Data Transmission Mode

You may have noticed that during the interaction between the terminal and the server, the terminal must communicate over the network at least twice: once to send data and again to receive a response. When using the OrderSend function, this process is essentially synchronous, i.e.m, the EA waits for a response, occupying system resources such as internet bandwidth and, potentially, CPU time.

However, from the processor point of view, network operations are very slow. A trading script might typically execute in a few hundred microseconds (e.g., 200 μs = 2e-4 seconds), but network data transmission is usually measured in milliseconds (e.g., 20 ms = 2e-2 seconds) which is at least 100 times slower. Add server-side processing time, and sometimes unexpected delays due to maintenance or technical issues... In the worst case, the gap between sending a trade request and receiving a response could stretch to seconds or even minutes. If multiple EAs are waiting idly during this time, a lot of CPU resources will be wasted unproductively.

To address this inefficiency, MetaTrader provides a special _asynchronous_ trading mode. The word asynchronous means the EA can send a trade request and continue doing other tasks - sleep, run calculations, or anything else - without waiting for a reply. When the server response eventually arrives, the terminal generates a TradeTransaction event (and then a Trade event). The EA can then "wake up" and process the response to make further trading decisions. The advantages of this approach are obvious.

In both synchronous and asynchronous modes, trade errors are handled using the OnTradeTransaction function. This does not make the code more complex - some logic is simply moved from OnTick to OnTradeTransaction. And if you move this code into separate functions, then calling and transferring it will not cause any problems at all. Therefore, the choice between synchronous and asynchronous trading modes depends entirely on the your preferences and the task at hand. The data structures used in both modes remain the same.

### Starting to Trade

Let’s suppose we want to build an Expert Advisor for the FOREX market that looks for inside bars. As a reminder, an inside bar is a candlestick whose high and low are completely within the range of the previous (larger) candle. The EA will operate once per candlestick, and when it detects the inside bar pattern, it will simultaneously place two pending orders:

- A Buy Stop a few points (configurable) above the high of the larger candlestick,
- A Sell Stop the same distance below the low of that candlestick,
- Each order will have a lifetime of two bars. If it's not triggered within this period, it will be deleted,

- Stop Loss for both orders will be placed at the midpoint of the larger candlestick,
- Take Profit will be set at 7/8 of the larger candlestick’s range,
- Trade volume will be the minimum allowed lot.


This initial version of the EA will avoid additional conditions to keep the code easier to understand. We'll build a skeleton framework that can be extended later. We'll start by creating the EA template using the wizard, leaving _all checkboxes unchecked_ in the third window (since in this version we won't handle server responses). The resulting code will be similar to Example 1. To make the EA configurable and optimizable, we'll define four input parameters: distance from high/low to place pending orders (inp\_PipsToExtremum), distance to place Stop Loss and Take Profit (inp\_StopCoefficient and inp\_TakeCoefficient), and the number of bars after which untriggered orders will be deleted (inp\_BarsForOrderExpired). Additionally, we'll declare a magic number for the EA - this helps distinguish "our" orders from those placed by other EAs or manually.

```
//--- declare and initialize input parameters
input int     inp_PipsToExtremum      = 2;
input double  inp_TakeCoeffcient      = 0.875;
input double  inp_StopCoeffcient      = 0.5;
input int     inp_BarsForOrderExpired = 2;

//--- declare and initialize global variables
#define EXPERT_MAGIC 11223344
```

**Scenario 2**. Description of EA input parameters and magic number

Just a reminder: the code from Example 2 must be placed at the very top of the EA file, immediately after the #property directives.

The _rest of the code_ in this example will be placed _inside the OnTick function_. We will leave all other functions empty for now. Here is the code that needs to be placed in the OnTick body:

```
 /****************************************************************
  *    Please note: this Expert Advisor uses standard functions  *
  * to access price/time data. Therefore, it's convenient to     *
  * work with series as arrays (time and prices).                *
  ****************************************************************/
  string          symbolName  = Symbol();
  ENUM_TIMEFRAMES period      = PERIOD_CURRENT;

//--- Define a new candlestick (Operations only at the start of a new candlestick)
  static datetime timePreviousBar = 0; // Time of the previous candlestick
  datetime timeCurrentBar;             // Time of the current candlestick

  // Get the time of the current candlestick using the standard function
  timeCurrentBar = iTime(
                     symbolName, // Symbol name
                     period,     // Period
                     0           // Candlestick index (remember it's series)
                   );

  if(timeCurrentBar==timePreviousBar)
   {
    // If the time of the current and previous candlesticks match
    return;  // Exit the function and do nothing
   }
  // Otherwise the current candlestick becomes the previous one,
  //   so as not to trade on the next tick
  timePreviousBar = timeCurrentBar;

//--- Prepare data for trading
  double volume=SymbolInfoDouble(symbolName,SYMBOL_VOLUME_MIN); // Volume (lots) - get minimum allowed volume

  // Candlestick extrema
  double high[],low[]; // Declare arrays

  // Declare that arrays are series
  ArraySetAsSeries(high,true);
  ArraySetAsSeries(low,true);

  // Fill arrays with values of first two closed candlesticks
  //   (start copying with index 1
  //   as we only need closed candlesticks; use 2 values)
  CopyHigh(symbolName,period,1,2,high);
  CopyLow(symbolName,period,1,2,low);

  double lengthPreviousBar; // The range of the "long" bar
  MqlTradeRequest request;  // Request structure
  MqlTradeResult  result;   // Server response structure

  if( // If the first closed bar is inside
    high[0]<high[1]
    && low[0]>low[1]
  )
   {
    // Calculate the range
    lengthPreviousBar=high[1]-low[1];  // Timeseries have right-to-left indexing

  //--- Prepare data for a buy order
    request.action      =TRADE_ACTION_PENDING;                         // order type (pending)
    request.symbol      =symbolName;                                   // symbol name
    request.volume      =volume;                                       // volume deal
    request.type        =ORDER_TYPE_BUY_STOP;                          // order action (buy)
    request.price       =high[1] + inp_PipsToExtremum*Point();         // buy price
    // Optional parameters
    request.deviation   =5;                                            // acceptable deviation from the price
    request.magic       =EXPERT_MAGIC;                                 // EA's magic number

    request.type_time   =ORDER_TIME_SPECIFIED;                         // Parameter is required to set the lifetime
    request.expiration  =timeCurrentBar+
                         PeriodSeconds()*inp_BarsForOrderExpired;      // Order lifetime

    request.sl          =high[1]-lengthPreviousBar*inp_StopCoeffcient;  // Stop Loss
    request.tp          =high[1]+lengthPreviousBar*inp_TakeCoeffcient;  // Take Profit

  //--- Send a buy order to the server
    OrderSend(request,result); // For asynchronous mode you need to use OrderSendAsync(request,result);

  //--- Clear the request and response structures for reuse
    ZeroMemory(request);
    ZeroMemory(result);

  //--- Prepare data for a sell order. Parameers are the same as in the previous function.
    request.action      =TRADE_ACTION_PENDING;                         // order type (pending)
    request.symbol      =symbolName;                                   // symbol name
    request.volume      =volume;                                       // volume
    request.type        =ORDER_TYPE_SELL_STOP;                         // order action (sell)
    request.price       =low[1] - inp_PipsToExtremum*Point();          // sell price
    // Optional parameters
    request.deviation   =5;                                            // acceptable deviation from the price
    request.magic       =EXPERT_MAGIC;                                 // EA's magic number

    request.type_time   =ORDER_TIME_SPECIFIED;                         // Parameter is required to set the lifetime
    request.expiration  =timeCurrentBar+
                         PeriodSeconds()*inp_BarsForOrderExpired;      // Order lifetime

    request.sl          =low[1]+lengthPreviousBar*inp_StopCoeffcient;   // Stop Loss
    request.tp          =low[1]-lengthPreviousBar*inp_TakeCoeffcient;   // Take Profit

  //--- Send a sell order to the server
    OrderSend(request,result);
   }

```

**Example 3.** The OnTick function of this EA contains all the trading logic.

The standard [Point](https://www.mql5.com/ru/docs/check/point) function returns the point size for the current chart. For example, if the broker provides five-digit quotes, a point for EURUSD will be 0.00001, and for USDJPY it will be 0.001. The functions [iTime](https://www.mql5.com/en/docs/series/itime), [iHigh](https://www.mql5.com/en/docs/series/ihigh), and [iLow](https://www.mql5.com/en/docs/series/ilow) allow you to retrieve the time, the highest, and the lowest price of a specific candlestick (by its index from right to left, with 0 being the current bar). In this example, we only used iTime to check for a new bar by retrieving the current time. For obtaining the high and low values, we used the array-copying functions [CopyHigh](https://www.mql5.com/en/docs/series/copyhigh) and [CopyLow](https://www.mql5.com/en/docs/series/copylow).

The code is divided into two main sections: checking for a new bar, and trading (which starts with a preparation phase). The trading block is further divided into two nearly identical segments: one for buying and one for selling. Clearly, the structure setup and order submission logic are so similar in both cases that it would make sense to refactor them into a separate function, which would fill in the common fields and branch out only for specifics like order type and execution prices (price, tp, sl). However, in this example, code compactness and reusability were intentionally sacrificed for clarity and readability.

Each stage of the process is marked with a comment using //---, while comments within each stage are written in plain style without dashes. The trading logic consists of two main parts: populating the request structure, and sending it. When filling in the request structure, only the first five fields are mandatory.

It's important to note that if you intend to use order expiration time, as demonstrated in this example, you must fill out _both_ request.type\_time and request.expiration. If you leave the first field unset, the second one will be ignored by default.

To test how this Expert Advisor operates, you can run it on any timeframe on a _demo_ account (it works even on minute charts, although the actual performance depends on the spread for the chosen symbol). Alternatively, press <Ctrl>+<F5> in MetaEditor to launch a backtest using historical data in the Strategy Tester. The complete source code can be found in the attached file TrendPendings.mq5.

### Using Standard Indicators

The Expert Advisor in Example 3 did not use any indicators, but that won't always be the case. For strategies based on standard indicators, there are two main approaches: using built-in indicator functions or working with indicator classes. We'll cover both methods, starting with the built-in functions.

Let’s say we want to create an Expert Advisor based on a simple moving average. As a reminder, the goal of this article is _not_ to build a _profitable_ strategy, but to _demonstrate_ fundamental trading logic. Therefore, I'll keep the code as simple and readable as possible. Based on that, we'll define our trading rules as follows:

- Like in the previous EA, trades will only be considered when a new bar forms;
- To buy, the previous candlestick must close above the moving average;
- To sell, the previous candlestick must close below the moving average;
- As a filter, we'll use the slope of the moving average: if the average rises from the second-to-last bar to the most recent closed bar, we buy; if it falls, we sell;

- We exit the position when an opposite signal appears;
- A protective Stop Loss is placed at the high (for sell signals) or low (for buy signals) of the signal candlestick;
- Only one open position per symbol is allowed, even on hedging accounts; if a signal appears but a position is already open, we skip it.


Figure 5 illustrates the filtering principle used in this EA.

![Signal candlestick filtering principle](https://c.mql5.com/2/155/MATradeBW-Edited__2.png)

**Figure 5.** Candlestick filtering principle used in a moving average-based EA

In this EA, I will continue striving to keep the code as clean and understandable as possible. However, I will begin incorporating more error checks to bring the implementation closer to real-world standards.

All moving average parameters, along with the maximum allowable price deviation from the requested order price, will be added to the input parameters. Additionally, we will declare a global variable for the indicator handle (I'll explain this in a moment), and also define the EA's magic number.

```
//--- declare and initialize global variables

#define EXPERT_MAGIC 3345677

input int inp_maPeriod = 3;                                 // MA period
input int inp_maShift = 0;                                  // Shift
input ENUM_MA_METHOD inp_maMethod = MODE_SMA;               // Calculation mode
input ENUM_APPLIED_PRICE inp_maAppliedPrice = PRICE_CLOSE;  // Applied price
input int inp_deviation = 5;                                // Max price deviation from the request price in points

//--- MA indicator handle
int g_maHandle;
```

**Example 4.** EA's global variables for trading using moving averages

Before using any indicator in an EA or another indicator, we need to do three things:

1. _Initialize_ the indicator and obtain a handle to it. This is typically done using built-in indicator functions (such as [iMA](https://www.mql5.com/en/docs/indicators/ima) for the Moving Average) or [iCustom](https://www.mql5.com/en/docs/indicators/icustom) for custom, user-defined indicators. This initialization is usually carried out inside the OnInit function.

In programming, a handle is like a reference or pointer to a resource that your code can interact with. Think of it as a _ticket number_ that lets you access the indicator and request its data whenever needed.

2. Before you can use the indicator values, you need to _fetch the most recent data_. This is usually done by creating one array per indicator buffer and populating these arrays using the [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer) function.
3. _Use_ the data from filled arrays.

4. To avoid memory leaks or unnecessary resource use, it's important to _release_ the indicator handle when the program ends. This is done in the OnDeinit function using [IndicatorRelease](https://www.mql5.com/en/docs/series/indicatorrelease).


In this particular EA, the OnInit and OnDeinit functions are fairly straightforward and contain no unusual or complex logic:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
 {
//--- Before an action that could potentially cause an error, reset
//   built-in _LastError variable to default
//   (assuming there's no error yet)
  ResetLastError();

//--- The standard iMA function returns the indicator handle
  g_maHandle = iMA(
                 _Symbol,           // Symbol
                 PERIOD_CURRENT,    // Chart period
                 inp_maPeriod,      // MA period
                 inp_maShift,       // MA shift
                 inp_maMethod,      // MA calculation method
                 inp_maAppliedPrice // Applied price
               );
// inp_maAppliedPrice in general case can be
// either a price type as in this example,
// (from ENUM_APPLIED_PRICE),
// or a handle of another indicator

//--- if the handle is not created
  if(g_maHandle==INVALID_HANDLE)
   {
    //--- report failure and output error code
    PrintFormat("Failed to crate iMA indicator handle for the pair %s/%s, error code is %d",
                _Symbol,
                EnumToString(_Period),
                GetLastError() // Output error code
               );
    //--- If an error occurs, terminate the EA early
    return(INIT_FAILED);
   }
//---
  return(INIT_SUCCEEDED);
 }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
 {
//--- Release resources occupied by the indicator
  if(g_maHandle!=INVALID_HANDLE)
    IndicatorRelease(g_maHandle);
 }
```

**Example 5.** Initialization and deinitialization of indicators in an Expert Advisor

One nuance I'd like to draw your attention to is the use of the [ResetLastError](https://www.mql5.com/en/docs/common/resetlasterror) and [GetLastError](https://www.mql5.com/en/docs/check/getlasterror) function pair within the initialization function. The former resets the system variable \_LastError to a "no error" state, while the latter allows you to retrieve the code of the most recent error if one has occurred.

Other than that, everything is fairly straightforward. Initialization functions for indicators (including iMA) return either a valid indicator handle or a special constant INVALID\_HANDLE if the handle could not be obtained. This mechanism allows us to detect when something has gone wrong and handle the error accordingly - in our case, by displaying an error message. If OnInit returns INIT\_FAILED, the Expert Advisor (or indicator) will not be launched. And indeed, if we fail to obtain a valid reference to the moving average indicator, halting execution is the only correct course of action.

As for the OnTick function, we'll break it down step by step. The first part involves the declaration and initialization of variables.

```
//--- Declare and initialize variables
  MqlTradeRequest requestMakePosition;  // Request structure for opening a new position
  MqlTradeRequest requestClosePosition; // Request structure for closing an existing position
  MqlTradeResult  result;               // Structure for receiving the server's response
  MqlTradeCheckResult checkResult;      // Structure for validating the request before sending

  bool positionExists = false;      // Flag indicating if a position exists
  bool tradingNeeds = false;        // Flag indicating whether trading is allowed
  ENUM_POSITION_TYPE positionType;  // Type of currently open position
  ENUM_POSITION_TYPE tradingType;   // Desired position type (used for comparison)
  ENUM_ORDER_TYPE orderType;        // Desired order type
  double requestPrice=0;            // Entry price for the future position

  /* The MqlRates structure contains
     all candle data: open, close, high,
     and low prices, tick volume,
     real volume, spread, and time.

     In this example, I decided to demonstrate how to fill the entire structure at once,
     instead of retrieving each value separately.                              */

  MqlRates rates[];   // Array of price data used for evaluating trading conditions
  double maValues[];  // Array of MA values

// Declare data arrays as series
  ArraySetAsSeries(rates,true);
  ArraySetAsSeries(maValues,true);
```

**Example 6.** Local variables of the OnTick function

We have the same check of whether a bar has appeared:

```
//--- Check whether there's a new bar
  static datetime previousTime  = iTime(_Symbol,PERIOD_CURRENT,0);
  datetime currentTime          = iTime(_Symbol,PERIOD_CURRENT,0);
  if(previousTime==currentTime)
   {
    return;
   }
  previousTime=currentTime;
```

**Example 7.** Checking if it's a new bar

Next, we obtain all the data we need using special functions. Here we assume that there may be errors, for example, that the terminal did not have time to load the necessary data, so we process these potential errors using the branching operator.

```
//---  Prepare data for processing
// Copy the quotes of two bars, starting from the first one
  if(CopyRates(_Symbol,PERIOD_CURRENT,1,2,rates)<=0)
   {
    PrintFormat("Data error for symbol %s, error code is %d", _Symbol, GetLastError());
    return;
   }

// Copy the values of the moving average indicator buffer
  if(CopyBuffer(g_maHandle,0,1,2,maValues)<=0)
   {
    PrintFormat("Error getting indicator data, error code is %d", GetLastError());
    return;
   }
```

**Example 8.** Copying current indicator data and quotes to local arrays

And now we select an open position for the current symbol using the standard PositionSelect function. We decided that there can only be one position for a symbol, so there shouldn't be any problems, but we still think carefully about what could go wrong... At least, we need to check that the position has been opened by _our_ EA:

```
//--- Determine if there is an open position
  if(PositionSelect(_Symbol))
   {
    // Set the open position flag - for further processing
    positionExists = true;
    // Save the type of the open position
    positionType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

    // Check if the position has been opened by our EA
    requestClosePosition.magic = PositionGetInteger(POSITION_MAGIC); // I didn't create a separate variable for
                                                                     // the existing position magic number
    if(requestClosePosition.magic!= EXPERT_MAGIC)
     {
      // Some other EA started trading our symbol. Let it do so...
      return;
     } // if(requestClosePosition.magic!= EXPERT_MAGIC)
   } // if(PositionSelect(_Symbol))
```

**Example 9.** Getting current position data

Now we can check the trading conditions. In this example, I will save the results of the check in separate variables, and then use these variables when making the final decision: buy or sell. In large and complex solutions, this approach justifies itself, firstly, by its flexibility, and secondly, by the fact that the final code becomes shorter. Here I used this technique mainly for the second reason: since the final algorithm is not very clear, I try to improve the clarity in different ways.

```
//--- Check trading conditions,
  if( // Conditions for BUY
    rates[0].close>maValues[0] // If the first candlestick closed above MA
    && maValues[0]>maValues[1] // and the MA slope is upwards
  )
   {
    // Set the trade flag
    tradingNeeds = true;
    // and inform the EA about the direction (here - BUY)
    tradingType = POSITION_TYPE_BUY; // to check the direction of the open direction
    orderType = ORDER_TYPE_BUY;      // to trade in the right direction
    // calculate the deal price
    requestPrice = SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   }

  else
    if( // conditions for SELL
      rates[0].close<maValues[0]
      && maValues[0]<maValues[1]
    )
     {
      tradingNeeds = true;
      tradingType = POSITION_TYPE_SELL;
      orderType = ORDER_TYPE_SELL;
      requestPrice = SymbolInfoDouble(_Symbol,SYMBOL_BID);
     }
```

**Example 10.** Checking trading conditions

The code shown below determines whether a trade should be executed at the current moment. The decision is based on three key questions:

- Is there a trading setup? In other words, has the candlestick closed beyond the moving average? This condition is handled by the tradingNeeds variable. If the answer is “no” (tradingNeeds == false), no trade should be made.

- Is there already an open position? This is checked using the positionExists variable. If there’s no open position - go ahead and trade. If there is - move on to the next check.

- Is the existing position aligned with or opposite to the new trade signal? This is determined by comparing tradingType and positionType. If they are equal, the position is aligned with the new signal, so no new trade is opened. If they differ, the current position is in the opposite direction and must be closed before opening a new one.

This decision logic is visualized in the flowchart (Figure 6).

![Flowchart of Key Trading Logic Branches](https://c.mql5.com/2/155/MA-expert-logic__1.png)

**Figure 6.** Flowchart representing the main decision points of the trading algorithm

In MQL5, both closing and opening a position involve sending a market order. The approach is the same as the one you're already familiar with: fill in a trade request structure and send it to the server. And the difference between _these_ two structures is that when closing a position, you need to specify its ticket and accurately copy the parameters of the existing position into the request. When opening a new position there is nothing to copy, so we are more free, and there is no old position ticket, so there is no need to pass anything.

Compared to the earlier example with pending orders, the code here differs in two fields:

- The action field, which previously held the value TRADE\_ACTION\_PENDING, now contains TRADE\_ACTION\_DEAL.

- The type field, which now represents a direct market order (ORDER\_TYPE\_BUY or ORDER\_TYPE\_SELL) rather than a pending order.

To make it easier to follow the correspondence between code fragments and the flowchart in Figure 6, the example code has been color-coded in alignment with the branching logic shown in the diagram.

There are two more notable differences from Example 3. Before sending a trade request, the structure is validated using [OrderCheck](https://www.mql5.com/en/docs/trading/ordercheck). This allows the program to catch incorrectly filled fields and provides both a return code (retcode) and a textual explanation (comment). After sending the request, we check whether the server accepted it. If an error occurs, the program will report it with a relevant message.

```
// If the setup is to trade
  if(tradingNeeds)
   {
    // If there is a position
    if(positionExists)
     {
      // And it is opposite to the desired direction of trade
      if(positionType != tradingType)
       {
        //--- Close the position

        //--- Clear all participating structures, otherwise you may get an "invalid request" error
        ZeroMemory(requestClosePosition);
        ZeroMemory(checkResult);
        ZeroMemory(result);
        //--- set operation parameters
        // Get position ticket
        requestClosePosition.position = PositionGetInteger(POSITION_TICKET);
        // Closing a position is just a trade
        requestClosePosition.action = TRADE_ACTION_DEAL;
        // position type is opposite to current trading direction,
        // therefore, for the closing deal, we can use the current order type
        requestClosePosition.type = orderType;
        // Current price
        requestClosePosition.price = requestPrice;
        // Operation volume must match the current position volume
        requestClosePosition.volume = PositionGetDouble(POSITION_VOLUME);
        // Set acceptable deviation from the current price
        requestClosePosition.deviation = inp_deviation;
        // Symbol
        requestClosePosition.symbol = Symbol();
        // Position magic number
        requestClosePosition.magic = EXPERT_MAGIC;

        if(!OrderCheck(requestClosePosition,checkResult))
         {
          // If the structure is filled incorrectly, display a message
          PrintFormat("Error when checking an order to close position: %d - %s",checkResult.retcode, checkResult.comment);
         }
        else
         {
          // Send order
          if(!OrderSend(requestClosePosition,result))
           {
            // If position closing failed, report
            PrintFormat("Error closing position: %d - %s",result.retcode,result.comment);
           } // if(!OrderSend)
         } // else (!OrderCheck)
       } // if(positionType != tradingType)
      else
       {
        // Position opened in the same direction as the trade signal. Do not trade
        return;
       } // else(positionType != tradingType)
     } // if(positionExists)

    //--- Open a new position

    //--- Clear all participating structures, otherwise you may get an "invalid request" error
    ZeroMemory(result);
    ZeroMemory(checkResult);
    ZeroMemory(requestMakePosition);

    // Fill the request structure
    requestMakePosition.action = TRADE_ACTION_DEAL;
    requestMakePosition.symbol = Symbol();
    requestMakePosition.volume = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
    requestMakePosition.type = orderType;
    // While waiting for position to close, the price could have changed
    requestMakePosition.price = orderType == ORDER_TYPE_BUY ?
                                SymbolInfoDouble(_Symbol,SYMBOL_ASK) :
                                SymbolInfoDouble(_Symbol,SYMBOL_BID) ;
    requestMakePosition.sl = orderType == ORDER_TYPE_BUY ?
                             rates[0].low :
                             rates[0].high;
    requestMakePosition.deviation = inp_deviation;
    requestMakePosition.magic = EXPERT_MAGIC;

    if(!OrderCheck(requestMakePosition,checkResult))
     {
      // If the structure check fails, report a check error
      PrintFormat("Error when checking a new position order: %d - %s",checkResult.retcode, checkResult.comment);
     }
    else
     {
      if(!OrderSend(requestMakePosition,result))
       {
        // If position opening failed, report an error
        PrintFormat("Error opening position: %d - %s",result.retcode,result.comment);
       } // if(!OrderSend(requestMakePosition

      // Trading completed, reset flag just in case
      tradingNeeds = false;
     } // else (!OrderCheck(requestMakePosition))
   } // if(tradingNeeds)
```

**Example 11.** The main trading code (most of the space is taken up by populating the structure and checking for errors)

The full source code of this example is included in the attached file: MADeals.mq5.

### Using Indicator Classes from the Standard Library

The classes for standard indicators are located in the <Include\\Indicators> folder. You can include all of them at once by importing the file <Include\\Indicators\\Indicators.mqh> (note the 's' at the end of the filename), or load them by group — e.g., "Trend.mqh", "Oscillators.mqh", "Volumes.mqh", or "BillWilliams.mqh". There are also separate files to include time series access classes ("TimeSeries.mqh") and a class for working with custom indicators ("Custom.mqh").

The remaining files in that folder are helper modules and will likely be of little use to those unfamiliar with object-oriented programming. Each "functional" file in the folder generally contains several related classes. These classes are typically named following a consistent convention: the prefix C followed by the same name used in the indicator creation function. For example, the class for working with moving averages is called CiMA and can be found in "Trend.mqh".

Working with these classes is very similar to working with the native MQL5 indicator functions. The main differences include the method calls and their naming. At the first stage - creation – we call the Create method and pass in the necessary parameters for the indicator. Atthe second stage - getting the data - we use the Refresh method, usually without parameters. If needed, you can specify which timeframes to update, for example: (OBJ\_PERIOD\_D1 \| OBJ\_PERIOD\_H1). During use, we utilize the GetData method, most commonly with two parameters: the buffer number and the candle index (note that indexing follows the time series model, increasing from right to left).

In Example 12, I provide a minimal Expert Advisor that uses the CiMA class. This EA simply outputs the value of the moving average on the first closed candlestick. If you'd like to see how this class-based approach can be used in an actual trading strategy, copy the Expert Advisor from the previous section (MADeals.mq5) into a new file and replace the appropriate lines with those from Example 12.

```
#include <Indicators\Indicators.mqh>
CiMA g_ma;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
 {
//--- Create the indicator
  g_ma.Create(_Symbol,PERIOD_CURRENT,3,0,MODE_SMA,PRICE_CLOSE);

//---
  return(INIT_SUCCEEDED);
 }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
 {
//---
  Comment("");

 }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
 {
//--- Get data
  g_ma.Refresh();

//--- Use
  Comment(
    NormalizeDouble(
      g_ma.GetData(0,1),
      _Digits
    )
  );
 }
//+------------------------------------------------------------------+
```

**Example 12.** Using the CiMA class (moving average indicator)

### Conclusion

After reading this article, you should now be able to write simple Expert Advisors (EAs) for rapid prototyping of any straightforward trading strategy - whether based solely on candlestick data or incorporating standard indicators that draw their signals via indicator buffers (rather than graphical representation). I hope the topic didn't seem overly complex. But if it did, it may be worth revisiting the material from earlier articles for a clearer understanding.

In the next article, I plan to present an Expert Advisor that is technically ready for publication on the Market. This EA will include even more validation checks than the second example in this article. These checks will make the EA more robust and reliable. Its structure will also be slightly different. The OnTick function will no longer serve as the sole center of business logic. Additional functions will appear to better organize the code. Most importantly, the EA will gain the ability to handle order placement errors (such as requotes). To achieve this, we'll restructure OnTick so that each "stage" of the EA operation (e.g., placing a trade, waiting for a new bar, calculating lot size...) can be accessed directly, without having to pass through other stages. We'll also use the TradeTransaction event to track server responses. The result will be a functionally organized, easily modifiable template that you can use to build your own EAs of any complexity - still without diving deep into OOP, but fully operational and production-ready.

List of previous articles within the series:

- [Mastering MQL5 from beginner to pro (Part I): Getting started with programming](https://www.mql5.com/en/articles/13594)
- [Master MQL5 from beginner to pro (Part II): Basic data types and use of variable](https://www.mql5.com/en/articles/13749)
- [Master MQL5 from beginner to pro (Part III): Complex data types and include files](https://www.mql5.com/en/articles/14354)
- [Master MQL5 from beginner to pro (Part IV): About Arrays, Functions and Global Terminal Variables](https://www.mql5.com/en/articles/15357)
- [Master MQL5 from beginner to pro (Part V): Fundamental control flow operators](https://www.mql5.com/en/articles/15499) (this article also discusses the principles of constructing custom indicators)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/15727](https://www.mql5.com/ru/articles/15727)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15727.zip "Download all attachments in the single ZIP archive")

[MADeals.mq5](https://www.mql5.com/en/articles/download/15727/madeals.mq5 "Download MADeals.mq5")(20.77 KB)

[TrendPendings.mq5](https://www.mql5.com/en/articles/download/15727/trendpendings.mq5 "Download TrendPendings.mq5")(12.84 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Master MQL5 from beginner to pro (Part V): Fundamental control flow operators](https://www.mql5.com/en/articles/15499)
- [Master MQL5 from beginner to pro (Part IV): About Arrays, Functions and Global Terminal Variables](https://www.mql5.com/en/articles/15357)
- [Master MQL5 from Beginner to Pro (Part III): Complex Data Types and Include Files](https://www.mql5.com/en/articles/14354)
- [Master MQL5 from beginner to pro (Part II): Basic data types and use of variable](https://www.mql5.com/en/articles/13749)
- [Master MQL5 from beginner to pro (Part I): Getting started with programming](https://www.mql5.com/en/articles/13594)
- [DRAKON visual programming language — communication tool for MQL developers and customers](https://www.mql5.com/en/articles/13324)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/490783)**
(5)


![Utkir Khayrullaev](https://c.mql5.com/avatar/2023/1/63BEC725-333C.png)

**[Utkir Khayrullaev](https://www.mql5.com/en/users/utkirkhayrullae)**
\|
7 Jan 2025 at 11:06

Thank you so much for your hard work! A lot of things became clear and easy.


![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
19 Feb 2025 at 14:33

excellent clear article and a lot of things are explained - thank you very much. Especially at the end how to use indicators through classes! Cool! I will consider to test prototypes in my development of simple TS.

![Oleh Fedorov](https://c.mql5.com/avatar/2017/12/5A335A41-73FE.jpg)

**[Oleh Fedorov](https://www.mql5.com/en/users/certain)**
\|
19 Mar 2025 at 11:39

Cheers! Glad it helped.


![Khanh Nguyen](https://c.mql5.com/avatar/2023/5/64537576-48a1.jpg)

**[Khanh Nguyen](https://www.mql5.com/en/users/danielkhanhnguyen)**
\|
30 Sep 2025 at 09:58

Thank you so much!

We are looking forward to your next part.

![Kevin V](https://c.mql5.com/avatar/2025/12/69339256-BEFC.png)

**[Kevin V](https://www.mql5.com/en/users/kevin_v)**
\|
6 Dec 2025 at 02:21

Just wanted to say thank you for these tutorials. They are very helpful.


![Singular Spectrum Analysis in MQL5](https://c.mql5.com/2/155/18777-singular-spectrum-analysis-logo.png)[Singular Spectrum Analysis in MQL5](https://www.mql5.com/en/articles/18777)

This article is meant as a guide for those unfamiliar with the concept of Singular Spectrum Analysis and who wish to gain enough understanding to be able to apply the built-in tools available in MQL5.

![MQL5 Wizard Techniques you should know (Part 74):  Using Patterns of Ichimoku and the ADX-Wilder with Supervised Learning](https://c.mql5.com/2/155/18776-mql5-wizard-techniques-you-logo.png)[MQL5 Wizard Techniques you should know (Part 74): Using Patterns of Ichimoku and the ADX-Wilder with Supervised Learning](https://www.mql5.com/en/articles/18776)

We follow up on our last article, where we introduced the indicator pair of the Ichimoku and the ADX, by looking at how this duo could be improved with Supervised Learning. Ichimoku and ADX are a support/resistance plus trend complimentary pairing. Our supervised learning approach uses a neural network that engages the Deep Spectral Mixture Kernel to fine tune the forecasts of this indicator pairing. As per usual, this is done in a custom signal class file that works with the MQL5 wizard to assemble an Expert Advisor.

![Self Optimizing Expert Advisors in MQL5 (Part 8): Multiple Strategy Analysis (3) — Weighted Voting Policy](https://c.mql5.com/2/155/18770-self-optimizing-expert-advisors-logo.png)[Self Optimizing Expert Advisors in MQL5 (Part 8): Multiple Strategy Analysis (3) — Weighted Voting Policy](https://www.mql5.com/en/articles/18770)

This article explores how determining the optimal number of strategies in an ensemble can be a complex task that is easier to solve through the use of the MetaTrader 5 genetic optimizer. The MQL5 Cloud is also employed as a key resource for accelerating backtesting and optimization. All in all, our discussion here sets the stage for developing statistical models to evaluate and improve trading strategies based on our initial ensemble results.

![Developing a Replay System (Part 74): New Chart Trade (I)](https://c.mql5.com/2/101/Desenvolvendo_um_sistema_de_Replay_Parte_74___LOGO.png)[Developing a Replay System (Part 74): New Chart Trade (I)](https://www.mql5.com/en/articles/12413)

In this article, we will modify the last code shown in this series about Chart Trade. These changes are necessary to adapt the code to the current replay/simulation system model. The content presented here is intended solely for educational purposes. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/15727&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049128618262635892)

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