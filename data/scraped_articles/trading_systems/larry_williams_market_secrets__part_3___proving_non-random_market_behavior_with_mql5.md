---
title: Larry Williams Market Secrets (Part 3): Proving Non-Random Market Behavior with MQL5
url: https://www.mql5.com/en/articles/20510
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:26:16.740657
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/20510&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069437620229965078)

MetaTrader 5 / Trading systems


### Introduction

Are financial markets truly random, or do they exhibit patterns that can be measured and tested? This question sits at the heart of every trading system ever created. If price movement is completely random, then any attempt to build a strategy, an indicator, or an automated trading system is ultimately pointless. There would be no edge to discover, only chance.

In this article, we revisit those ideas from a modern, practical perspective _(for a more detailed discussion of the underlying statistical concepts, see the [previous article](https://www.mql5.com/en/articles/20512))._ Instead of accepting conclusions at face value, we will use [MQL5](https://www.mql5.com/en/docs) to recreate and extend Williams’ experiments through code. By writing a custom Expert Advisor, we will test whether certain price behaviors occur more frequently than would be expected by chance. The goal is not to predict markets with certainty, but to determine whether small, measurable biases exist.

This matters because even a slight statistical bias can form the foundation of a trading edge. If past price behavior has no relationship to future outcomes, then systematic trading cannot work. However, if markets are even partially non-random, then structure, logic, and probability become meaningful tools for traders.

### How We Test Non-Random Market Behavior

In the introduction, we established a simple but essential idea. If markets were truly random, price behavior would resemble a coin toss. Each candle would be independent of the previous one, and the probability of an up or down close would remain fixed at fifty percent, regardless of what happened before.

Larry Williams challenged this assumption by asking a practical question. What happens after the price closes up or down? If markets have no memory, then the outcome of the next candle should not change. His research showed that this is not what actually happens.

To verify this concept practically and measurably, we will take an automated approach. Instead of relying on assumptions or selective examples, we will let code scan historical price data and count how often specific conditions occur. Each test is designed to answer a particular probability question about price behavior.

The experiments we will conduct fall into three broad categories. First, we test whether there is an overall directional bias within a single candle. This answers the fundamental question of whether the price closes higher than it opens more often than would be expected by chance.

Second, we test conditional probability. Here, we ask whether the market behaves differently after one or more consecutive bullish and bearish candles. If markets were random, the probability of the next candle being bullish or bearish would remain unchanged. If the probabilities shift, then the price clearly reacts to its recent past.

Third, we test a simple form of market structure. Larry Williams described a short-term low as a three-bar pattern where the price makes a low and then fails to continue lower. If this pattern has predictive value, then the candle that follows should close higher more often than chance alone would suggest.

To cover these ideas, we automate the following eight test cases:

Open to close bias

We measure how often a candle closes bullish, meaning the close price is higher than the open price. In a truly random market, bullish and bearish closes should occur roughly fifty percent of the time each.

Bullish response after one bearish close

We test how often the next candle closes bullish after a single bearish candle. This helps us see whether a short-term pullback tends to attract buying pressure.

Bullish response after two consecutive bearish closes

Here, we check whether the probability of a bullish close increases after two bearish candles in a row. This examines whether short-term declines tend to set up rebounds.

Bullish response after three consecutive bearish closes

This test extends the same idea further by examining price behavior after a more profound short-term decline.

Bullish response after one bullish close

We test whether bullish momentum tends to continue or fade after a single bullish candle.

Bullish response after two consecutive bullish closes

This experiment tests whether strength tends to follow strength or whether price stalls after a short rally.

Bullish response after three consecutive bullish closes

This test checks whether extended short-term buying pressure leads to continuation or exhaustion.

Bullish response after a short-term market low

Based on Larry Williams’ definition, we detect a short-term low using a three-bar price pattern and measure how often the following candle closes bullish.

Each of these tests answers a simple question. Does the outcome of previous candles influence the probability of the next one? If the answer were always fifty percent, then markets would behave like a coin toss. Any consistent deviation from that level points to non-random behavior.

With these test cases clearly defined, we can now move to the technical part. In the next section, we will implement an MQL5 expert advisor that allows us to run each experiment individually on different markets and observe the results directly from historical data.

### Implementing the Experiments in MQL5

To test whether these market behaviors are truly non-random, we need more than statistics. We want to simulate real trading conditions. That means opening a position at candle open, closing it at candle close, and recording what actually happens. This approach allows us to evaluate probabilities and also answer a more practical question. Can these behaviors be turned into tradable ideas later on?

To begin, open [MetaEditor 5,](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor") create a new Expert Advisor, choose an empty template, and name the file, _larryWilliamsNonRandomMarketBehaviorTester.mq5_. Once created, paste the provided source code into the file and save it.

```
//+------------------------------------------------------------------+
//|                   larryWilliamsNonRandomMarketBehaviorTester.mq5 |
//|          Copyright 2025, MetaQuotes Ltd. Developer is Chacha Ian |
//|                          https://www.mql5.com/en/users/chachaian |
//+------------------------------------------------------------------+

#property copyright   "Copyright 2025, MetaQuotes Ltd. Developer is Chacha Ian"
#property link        "https://www.mql5.com/en/users/chachaian"
#property version     "1.00"
#property description "This Expert Advisor is designed to run one statistical experiment at a time."
#property description "The test to be executed is selected through a dropdown input when attaching the EA to the chart."

//+------------------------------------------------------------------+
//| Standard Libraries                                               |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+
//--- Create a CTrade object to handle trading operations
CTrade Trade;

//--- Bid and Ask
double   askPrice;

//+------------------------------------------------------------------+
//| User input variables                                             |
//+------------------------------------------------------------------+
input group "Information"
input ulong           magicNumber                 = 254700680002;
input ENUM_TIMEFRAMES timeframe                   = PERIOD_CURRENT;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){

   //---  Assign a unique magic number to identify trades opened by this EA
   Trade.SetExpertMagicNumber(magicNumber);

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason){

   //--- Notify why the program stopped running
   Print("Program terminated! Reason code: ", reason);

}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){

   //--- Retrieve current market prices for trade execution
   askPrice      = SymbolInfoDouble (_Symbol, SYMBOL_ASK);

}

//+------------------------------------------------------------------+
//| TradeTransaction function                                        |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
{
}

//+------------------------------------------------------------------+
```

This initial code is the foundation we will build on. It does not yet run any experiments, but it sets up everything we need to do so in a clean, controlled way. To help you follow along, we have attached the final source code: _larryWilliamsNonRandomMarketBehaviorTester.mq5_. Feel free to download the file and compare it with your version to ensure everything is set up correctly.

File header and property directives

```
//+------------------------------------------------------------------+
//|                   larryWilliamsNonRandomMarketBehaviorTester.mq5 |
//|          Copyright 2025, MetaQuotes Ltd. Developer is Chacha Ian |
//|                          https://www.mql5.com/en/users/chachaian |
//+------------------------------------------------------------------+

#property copyright   "Copyright 2025, MetaQuotes Ltd. Developer is Chacha Ian"
#property link        "https://www.mql5.com/en/users/chachaian"
#property version     "1.00"
#property description "This Expert Advisor is designed to run one statistical experiment at a time."
#property description "The test to be executed is selected through a dropdown input when attaching the EA to the chart."
```

The first section defines the file name, author details, version, and description. These properties are important because they describe the Expert Advisor's purpose when attached to a chart. In this case, the description clearly states that the EA runs one statistical experiment at a time and that the experiment is selected using an input dropdown. This aligns directly with the article's goals.

Standard library inclusion

```
//+------------------------------------------------------------------+
//| Standard Libraries                                               |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
```

Next, we include the standard trading library. This library provides access to the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class, which handles order execution in a safe, structured way. Using this class is recommended because it simplifies trade management and reduces the risk of execution errors.

User input settings

```
//+------------------------------------------------------------------+
//| User input variables                                             |
//+------------------------------------------------------------------+
input group "Information"
input ulong           magicNumber                 = 254700680002;
input ENUM_TIMEFRAMES timeframe                   = PERIOD_CURRENT;
```

The input section defines parameters that the user can control when attaching the EA to a chart. We specify a _magic number_ to uniquely identify trades opened by this EA, preventing them from interfering with other strategies. We also allow the user to select the experiment _timeframe_. This makes the EA flexible and reusable across different chart setups.

Global trading objects and variables

```
//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+
//--- Create a CTrade object to handle trading operations
CTrade Trade;

//--- Bid and Ask
double   askPrice;
```

We then create a global _CTrade_ object. This object will be used throughout the EA to open and close positions. We also declare variables for market prices. At this stage, we only retrieve the [ask price](https://en.wikipedia.org/wiki/Ask_price "https://en.wikipedia.org/wiki/Ask_price"), which is sufficient for opening buy trades during our experiments.

Initialization logic

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){

   //---  Assign a unique magic number to identify trades opened by this EA
   Trade.SetExpertMagicNumber(magicNumber);

   return(INIT_SUCCEEDED);
}
```

The [initialization](https://www.mql5.com/en/docs/event_handlers/oninit) function runs once when the EA is attached to a chart. Here, we assign _the magic number_ to the _CTrade_ object. This step is essential because it ensures that every trade opened by the EA can be tracked and managed correctly later on.

Deinitialization logic

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason){

   //--- Notify why the program stopped running
   Print("Program terminated! Reason code: ", reason);

}
```

When the EA is removed or stopped, the [deinitialization](https://www.mql5.com/en/docs/event_handlers/ondeinit) function is called. In our case, it simply prints a message indicating why the program stopped. This is useful for debugging and for understanding when and how the EA lifecycle ends.

Tick handling

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){

   //--- Retrieve current market prices for trade execution
   askPrice      = SymbolInfoDouble (_Symbol, SYMBOL_ASK);

}
```

The [tick function](https://www.mql5.com/en/docs/event_handlers/ontick) runs continuously as new market data arrives. For now, it only retrieves the current _ask price_. Later, this function will serve as the control center for our experiments. It will detect new candles, evaluate test conditions, open trades, and close them at the appropriate time.

Trade transaction handler

```
//+------------------------------------------------------------------+
//| TradeTransaction function                                        |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
{
}
```

Finally, we define the [trade transaction](https://www.mql5.com/en/docs/event_handlers/ontradetransaction) function. This function is triggered whenever a trade action occurs, such as opening or closing a position. Although it is currently empty, it will later help track trade outcomes and collect experiment results.

At this point, the Expert Advisor shows no visible activity on the chart. That is intentional. We have built a clean and reliable skeleton that follows best practices. In the next section, we will begin adding logic to detect new candles, apply our non-random test cases, and execute trades to observe real market behavior in action.

With the EA skeleton in place, the next phase is to teach our program how to recognize market behavior. At this stage, we are not trading yet. Our goal is simple and crucial. We want the EA to correctly detect each pattern that represents one of our non-random test cases.

Before an EA can test anything, it must clearly understand which experiment it is running. That is why the first step is to define a mechanism for switching between test scenarios in a clean, controlled way.

Defining the test modes

Just below the standard library inclusion, we introduce a custom enumeration that lists all the non-random experiments we want to run. Each value represents one distinct market behavior we want to study.

```
//--- CUSTOM ENUMERATIONS
//+------------------------------------------------------------------+
//| Non-random market behavior test modes                            |
//+------------------------------------------------------------------+
enum ENUM_NON_RANDOM_TEST_MODE
{
   TEST_OPEN_TO_CLOSE_BIAS,
   TEST_AFTER_ONE_DOWN_CLOSE,
   TEST_AFTER_TWO_DOWN_CLOSES,
   TEST_AFTER_THREE_DOWN_CLOSES,
   TEST_AFTER_ONE_UP_CLOSE,
   TEST_AFTER_TWO_UP_CLOSES,
   TEST_AFTER_THREE_UP_CLOSES,
   TEST_AFTER_SHORT_TERM_LOW
};
```

This enumeration allows the EA to operate in one test mode at a time. When the EA is attached to a chart, the user will select which experiment to run from a dropdown menu defined as an input parameter. This design keeps the logic simple and avoids mixing results from different tests.

At this point, we are not testing probabilities yet. We are only defining the structure. This structure will guide all pattern detection and decision-making that follows.

Detecting the bar open instance

All our experiments are evaluated once, when a new candle opens. This is critical. If we react to every tick, we introduce noise and duplicate signals. To avoid this, we detect patterns only when a new bar forms.

To achieve this, we define a small but powerful utility function that checks whether a new bar has opened on the selected timeframe.

```
//--- UTILITY FUNCTIONS
//+------------------------------------------------------------------+
//| Function to check if there's a new bar on a given chart timeframe|
//+------------------------------------------------------------------+
bool IsNewBar(string symbol, ENUM_TIMEFRAMES tf, datetime &lastTm)
{

   datetime currentTm = iTime(symbol, tf, 0);
   if(currentTm != lastTm){
      lastTm       = currentTm;
      return true;
   }
   return false;
}
```

The function compares the current bar's opening time with the last recorded bar's time. If the time has changed, a new bar has formed.

To support this logic, we also declare a global variable that stores the opening time of the most recent bar.

```
//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+

...

//--- To help track new bar open
datetime lastBarOpenTime;
```

This variable is initialized to zero inside the initialization function so that the first bar is always detected correctly.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){

   ...

   //--- Initialize global variables
   lastBarOpenTime = 0;

   return(INIT_SUCCEEDED);
}
```

From this point onward, every pattern check in the EA will happen only when a new bar is confirmed. This ensures consistency across all experiments.

A universal function for consecutive bar behavior

Most of our test cases are based on a simple question. After a certain number of bullish or bearish bars, what happens next?

Instead of writing separate logic for each case, we build a single, flexible function that handles them all. This function checks whether the last N completed candles all closed in the same direction relative to their open.

```
//+------------------------------------------------------------------+
//| Checks whether the last N completed bars all closed              |
//| either up or down relative to their open                         |
//+------------------------------------------------------------------+
bool IsConsecutiveBarCloseState(string symbol, ENUM_TIMEFRAMES tf, int barsToCheck, ENUM_BAR_CLOSE_STATE closeState)
{
   // Start from bar index 1 (last fully closed bar)
   for(int i = 1; i <= barsToCheck; i++){

      double openPrice  = iOpen (symbol, timeframe, i);
      double closePrice = iClose(symbol, timeframe, i);

      // Safety check (in case of missing data)
      if(openPrice == 0.0 || closePrice == 0.0){
         return false;
      }

      // Validate close direction
      if(closeState == BAR_CLOSE_UP && closePrice <= openPrice){
         return false;
      }

      if(closeState == BAR_CLOSE_DOWN && closePrice >= openPrice){
         return false;
      }
   }
   return true;
}
```

To make the function reusable and clear, we pass four inputs. The symbol being tested, the timeframe, how many consecutive bars to evaluate, and the direction of those bars. To clearly describe bar direction, we introduce a small enumeration for bullish and bearish closes.

```
//--- CUSTOM ENUMERATIONS
//+------------------------------------------------------------------+
//| Non-random market behavior test modes                            |
//+------------------------------------------------------------------+

...

//+------------------------------------------------------------------+
//| Direction of candle close relative to open                       |
//+------------------------------------------------------------------+
enum ENUM_BAR_CLOSE_STATE
{
   BAR_CLOSE_UP,
   BAR_CLOSE_DOWN
};
```

The function loops only through completed candles. It starts from the most recently closed bar and moves backward. For each bar, it compares the open and close prices and verifies that they match the expected direction. If any bar fails the condition, the function returns _false_ immediately. If all bars match, the function returns _true_.

This single function supports six of our eight experiments. It is the core building block of our non-random tests and will later drive real trades.

Detecting a Larry Williams short-term low

The final pattern we need to detect is more structural. Larry Williams defines a short-term low as a three-bar formation where the middle bar forms a swing low, with higher lows on both sides.

However, he also adds two important filters. The swing bar must not be an outside bar, and the most recent bar must not be an inside bar. These conditions help eliminate weak or misleading signals.

To detect this pattern, we build a dedicated function that analyzes the last three completed bars.

```
//+------------------------------------------------------------------+
//| Detects a Larry Williams short-term low on the last three bars   |
//| Bar index 2 must be a swing low with higher lows on both sides   |
//| Bar 2 must NOT be an outside bar                                 |
//| Bar 1 must NOT be an inside bar                                  |
//+------------------------------------------------------------------+
bool IsLarryWilliamsShortTermLow(string symbol, ENUM_TIMEFRAMES tf){

   //--- Price data for the three bars
   double high1 = iHigh(symbol, tf, 1);
   double low1  = iLow (symbol, tf, 1);

   double high2 = iHigh(symbol, tf, 2);
   double low2  = iLow (symbol, tf, 2);

   double high3 = iHigh(symbol, tf, 3);
   double low3  = iLow (symbol, tf, 3);

   //--- Condition 1: Bar 2 must be a swing low
   bool isSwingLow =
      (low2 < low1) &&
      (low2 < low3);

   if(!isSwingLow){
      return false;
   }

   //--- Condition 2: Bar 2 must NOT be an outside bar relative to bar 3
   bool isOutsideBar =
      (high2 > high3) &&
      (low2  < low3);

   if(isOutsideBar){
      return false;
   }

   //--- Condition 3: Bar 1 must NOT be an inside bar relative to bar 2
   bool isInsideBar =
      (high1 < high2) &&
      (low1  > low2);

   if(isInsideBar){
      return false;
   }

   //--- All conditions satisfied
   return true;
}
```

First, it checks whether the middle bar has the lowest low. Then it verifies that the middle bar is not an outside bar relative to the previous one. Finally, it confirms that the most recent bar is not an inside bar relative to the swing bar.

Only if all conditions are satisfied does the function return _true_.

Selecting the active experiment

To make testing easy, we introduce a user input that allows the trader to select which non-random test mode to run. This input uses the enumeration we defined earlier and appears as a dropdown when the EA is attached to a chart.

```
//+------------------------------------------------------------------+
//| User input variables                                             |
//+------------------------------------------------------------------+

...

input group "Trade And Risk Management"
input ENUM_NON_RANDOM_TEST_MODE nonRandomTestMode = TEST_OPEN_TO_CLOSE_BIAS;
```

This approach allows us to reuse the same EA for all experiments without changing the code each time.

Verifying pattern detection inside OnTick

With all detection logic in place, we now connect everything inside the tick function.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){

   ...

   //--- Run this block only when a new bar is detected on the selected timeframe
   if(IsNewBar(_Symbol, timeframe, lastBarOpenTime)){

      //--- Execute a buy trade when the Open-to-Close bias test mode is selected
      if(nonRandomTestMode == TEST_OPEN_TO_CLOSE_BIAS    ){
         Print("EA should open a long position");
      }

      //--- Enter a buy position to test bullish bias following a single down close
      if(nonRandomTestMode == TEST_AFTER_ONE_DOWN_CLOSE   ){
         if(IsConsecutiveBarCloseState(_Symbol, timeframe, 1, BAR_CLOSE_DOWN)){
            Print("Previous bearish bar");
         }
      }

      //--- Enter a buy position to test bullish bias following two consecutive down closes
      if(nonRandomTestMode == TEST_AFTER_TWO_DOWN_CLOSES  ){
         if(IsConsecutiveBarCloseState(_Symbol, timeframe, 2, BAR_CLOSE_DOWN)){
            Print("Two consecutive bearish bars");
         }
      }

      //--- Enter a buy position to test bullish bias following three consecutive down closes
      if(nonRandomTestMode == TEST_AFTER_THREE_DOWN_CLOSES){
         if(IsConsecutiveBarCloseState(_Symbol, timeframe, 3, BAR_CLOSE_DOWN)){
            Print("Three consecutive bearish bars");
         }
      }

      //--- Enter a buy position to test bullish bias following a single up close
      if(nonRandomTestMode == TEST_AFTER_ONE_UP_CLOSE     ){
         if(IsConsecutiveBarCloseState(_Symbol, timeframe, 1, BAR_CLOSE_UP)){
            Print("Previous bullish bar");
         }
      }

      //--- Enter a buy position to test bullish bias following two consecutive up closes
      if(nonRandomTestMode == TEST_AFTER_TWO_UP_CLOSES    ){
         if(IsConsecutiveBarCloseState(_Symbol, timeframe, 2, BAR_CLOSE_UP)){
            Print("Two consecutive bullish bars");
         }
      }

      //--- Enter a buy position to test bullish bias following three consecutive up closes
      if(nonRandomTestMode == TEST_AFTER_THREE_UP_CLOSES  ){
         if(IsConsecutiveBarCloseState(_Symbol, timeframe, 3, BAR_CLOSE_UP)){
            Print("Three consecutive bullish bars");
         }
      }

      //---  Enter a buy position when a Larry Williams-defined short-term low pattern is detected
      if(nonRandomTestMode == TEST_AFTER_SHORT_TERM_LOW   ){
         if(IsLarryWilliamsShortTermLow(_Symbol, timeframe)){
            Print("Short-term low");
         }
      }
   }
}
```

The EA first checks whether a new bar has formed. If not, nothing happens. When a new bar is detected, the EA evaluates only the selected test mode. For each case, it calls the appropriate detection function and prints a message when a pattern is found.

At this stage, no trades are placed. The printed messages act as confirmation signals. They allow us to visually verify that patterns are being detected at the correct time and under the correct conditions.

This step is critical. Before adding real trading logic, we must ensure our pattern detection is accurate. Building on incorrect logic would invalidate all subsequent results.

In the next section, we will replace these print statements with actual trade execution logic. At that point, our EA will move from observation to experimentation, allowing us to test whether these non-random behaviors can translate into real market outcomes.

At this stage, our EA already knows how to detect all the patterns required for the non-random market tests. The next step is to turn those signals into real market actions. This is where observation becomes experimentation.

For consistency with Larry Williams’ original logic, our EA will **only open long positions**. Each trade opens at the start of a new bar and closes at the end of that bar. There are **no stop-loss or take-profit** levels. The goal is not trade management, but measurement. We want to observe how the price behaves immediately after a specific condition appears.

Because of this design, the EA must reliably be able to do three things. First, it must know whether it already has an open position. Second, it must be able to close that position cleanly. Third, it must open a new position only when the selected test condition is met.

Checking for an active position

Before opening a new trade, the EA needs to confirm whether it already has an active buy position. Since multiple EAs or manual trades may exist on the same account, we use a magic number to identify trades opened by this EA uniquely.

To handle this, we define a function that scans all open positions and checks whether any have a matching magic number. If such a position exists, the function returns true. Otherwise, it returns false.

```
//+------------------------------------------------------------------+
//| To verify whether this EA currently has an active buy position.  |                                 |
//+------------------------------------------------------------------+
bool IsThereAnActiveBuyPosition(ulong magic){

   for(int i = PositionsTotal() - 1; i >= 0; i--){
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0){
         Print("Error while fetching position ticket ", _LastError);
         continue;
      }else{
         if(PositionGetInteger(POSITION_MAGIC) == magic && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY){
            return true;
         }
      }
   }

   return false;
}
```

This check allows the EA to remain disciplined. At any given time, there can only be one active position per test. This keeps the results clean and avoids overlapping trades that would distort the statistics.

Closing an existing position

Once a new bar forms, the previous bar has officially closed. That moment represents the end of our test window for the previous trade. If a position is still open, it must be closed immediately.

To achieve this, we define a function that loops through all open positions and closes those that belong to this EA based on the magic number. The function does not attempt to manage profit or loss. It simply exits the trade at market price.

```
//+------------------------------------------------------------------+
//| To close all position with a specified magic number              |
//+------------------------------------------------------------------+
void ClosePositionsByMagic(ulong magic) {

    for (int i = PositionsTotal() - 1; i >= 0; i--) {
        ulong ticket = PositionGetTicket(i);
        if (PositionSelectByTicket(ticket)) {
            if (PositionGetInteger(POSITION_MAGIC) == magic) {
                ulong positionType = PositionGetInteger(POSITION_TYPE);
                double volume = PositionGetDouble(POSITION_VOLUME);
                if (positionType == POSITION_TYPE_BUY) {
                    Trade.PositionClose(ticket);
                } else if (positionType == POSITION_TYPE_SELL) {
                    Trade.PositionClose(ticket);
                }
            }
        }
    }
}
```

This approach ensures that every test follows the same rules. One bar in, one bar out. Nothing more.

Opening a new long position

With position control in place, we define a simple function to open a market buy order. The function takes two inputs. The first is the entry price, which is the current ask price already tracked by the EA. The second is the position size.

```
//+------------------------------------------------------------------+
//| Function to open a market buy position                           |
//+------------------------------------------------------------------+
bool OpenBuy(double entryPrice, double lotSize){
   if(!Trade.Buy(NormalizeDouble(lotSize, 2), _Symbol, entryPrice)){
      Print("Error while executing a market buy order: ", GetLastError());
      Print(Trade.ResultRetcode());
      Print(Trade.ResultComment());
      return false;
   }
   return true;
}
```

If the trade fails for any reason, the function prints diagnostic information. Otherwise, it returns true to confirm that the order was successfully placed.

To make this flexible, we allow the user to define their preferred lot size through an input parameter. This keeps the EA usable across different account sizes while preserving the experiment's logic.

```
//+------------------------------------------------------------------+
//| User input variables                                             |
//+------------------------------------------------------------------+

...

input double positionSize                         = 0.01;
```

Executing the logic inside OnTick

Now everything comes together.

Each time a new bar is detected, the EA performs the following steps in order. First, it closes any existing position opened by this EA. This marks the end of the previous test cycle. A short pause is added to ensure the platform processes the closure cleanly.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){

   ...

   //--- Run this block only when a new bar is detected on the selected timeframe
   if(IsNewBar(_Symbol, timeframe, lastBarOpenTime)){

      //--- Close any existing buy positions for this EA before opening a new one
      if(IsThereAnActiveBuyPosition(magicNumber)){
         ClosePositionsByMagic(magicNumber);
         Sleep(100);
      }

      ...

   }
}
```

Next, the EA checks which test mode is currently selected. Based on that choice, it evaluates the corresponding condition. If the condition is satisfied, a new buy position is opened immediately at the start of the new bar.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){

   ...

   //--- Run this block only when a new bar is detected on the selected timeframe
   if(IsNewBar(_Symbol, timeframe, lastBarOpenTime)){

      ...

      //--- Execute a buy trade when the Open-to-Close bias test mode is selected
      if(nonRandomTestMode == TEST_OPEN_TO_CLOSE_BIAS    ){
         OpenBuy(askPrice, positionSize);
      }

      //--- Enter a buy position to test bullish bias following a single down close
      if(nonRandomTestMode == TEST_AFTER_ONE_DOWN_CLOSE   ){
         if(IsConsecutiveBarCloseState(_Symbol, timeframe, 1, BAR_CLOSE_DOWN)){
            OpenBuy(askPrice, positionSize);
         }
      }

      //--- Enter a buy position to test bullish bias following two consecutive down closes
      if(nonRandomTestMode == TEST_AFTER_TWO_DOWN_CLOSES  ){
         if(IsConsecutiveBarCloseState(_Symbol, timeframe, 2, BAR_CLOSE_DOWN)){
            OpenBuy(askPrice, positionSize);
         }
      }

      //--- Enter a buy position to test bullish bias following three consecutive down closes
      if(nonRandomTestMode == TEST_AFTER_THREE_DOWN_CLOSES){
         if(IsConsecutiveBarCloseState(_Symbol, timeframe, 3, BAR_CLOSE_DOWN)){
            OpenBuy(askPrice, positionSize);
         }
      }

      //--- Enter a buy position to test bullish bias following a single up close
      if(nonRandomTestMode == TEST_AFTER_ONE_UP_CLOSE     ){
         if(IsConsecutiveBarCloseState(_Symbol, timeframe, 1, BAR_CLOSE_UP)){
            OpenBuy(askPrice, positionSize);
         }
      }

      //--- Enter a buy position to test bullish bias following two consecutive up closes
      if(nonRandomTestMode == TEST_AFTER_TWO_UP_CLOSES    ){
         if(IsConsecutiveBarCloseState(_Symbol, timeframe, 2, BAR_CLOSE_UP)){
            OpenBuy(askPrice, positionSize);
         }
      }

      //--- Enter a buy position to test bullish bias following three consecutive up closes
      if(nonRandomTestMode == TEST_AFTER_THREE_UP_CLOSES  ){
         if(IsConsecutiveBarCloseState(_Symbol, timeframe, 3, BAR_CLOSE_UP)){
            OpenBuy(askPrice, positionSize);
         }
      }

      //---  Enter a buy position when a Larry Williams-defined short-term low pattern is detected
      if(nonRandomTestMode == TEST_AFTER_SHORT_TERM_LOW   ){
         if(IsLarryWilliamsShortTermLow(_Symbol, timeframe)){
            OpenBuy(askPrice, positionSize);
         }
      }
   }
}
```

The only change from our earlier testing phase is simple but important. We have replaced all Print statements with actual trade execution calls. The detection logic remains unchanged. This confirms that our earlier validation work was correct.

At this point, the EA is fully operational. It detects patterns, opens trades, closes them after one bar, and repeats the process consistently across all test cases.

Configuring the Chart for Clear Visual Testing

Since this Expert Advisor is designed for visual and statistical experimentation, it is essential that we can clearly see what is happening on the chart while tests are running. We want to immediately recognize bullish and bearish bars, identify trade entries and exits, and avoid unnecessary visual clutter that could distract from observation.

For this reason, we introduce a small helper function that configures the chart’s appearance as soon as the EA is attached.

```
//+------------------------------------------------------------------+
//| This function configures the chart's appearance.                 |
//+------------------------------------------------------------------+
bool ConfigureChartAppearance()
{
   if(!ChartSetInteger(0, CHART_COLOR_BACKGROUND, clrWhite)){
      Print("Error while setting chart background, ", GetLastError());
      return false;
   }

   if(!ChartSetInteger(0, CHART_SHOW_GRID, false)){
      Print("Error while setting chart grid, ", GetLastError());
      return false;
   }

   if(!ChartSetInteger(0, CHART_MODE, CHART_CANDLES)){
      Print("Error while setting chart mode, ", GetLastError());
      return false;
   }

   if(!ChartSetInteger(0, CHART_COLOR_FOREGROUND, clrBlack)){
      Print("Error while setting chart foreground, ", GetLastError());
      return false;
   }

   if(!ChartSetInteger(0, CHART_COLOR_CANDLE_BULL, clrSeaGreen)){
      Print("Error while setting bullish candles color, ", GetLastError());
      return false;
   }

   if(!ChartSetInteger(0, CHART_COLOR_CANDLE_BEAR, clrBlack)){
      Print("Error while setting bearish candles color, ", GetLastError());
      return false;
   }

   if(!ChartSetInteger(0, CHART_COLOR_CHART_UP, clrSeaGreen)){
      Print("Error while setting bearish candles color, ", GetLastError());
      return false;
   }

   if(!ChartSetInteger(0, CHART_COLOR_CHART_DOWN, clrBlack)){
      Print("Error while setting bearish candles color, ", GetLastError());
      return false;
   }

   return true;
}
```

During experimentation, visual feedback plays a key role. Even though our conclusions will ultimately be based on data and statistics, seeing how price behaves across each test condition helps validate assumptions and spot anomalies early.

The configuration applied by this function achieves three core objectives:

- Clarity – A white background with black foreground elements ensures maximum contrast.
- Directional emphasis – Bullish bars are clearly distinguished from bearish bars using modern, intuitive colors.
- Focus – Grid lines are removed to keep attention on price action and executed trades.

The chart configuration function is called inside the _OnInit_ function.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){

   ...

   //--- To configure the chart's appearance
   if(!ConfigureChartAppearance()){
      Print("Error while configuring chart appearance", GetLastError());
      return INIT_FAILED;
   }

   return(INIT_SUCCEEDED);
}
```

This ensures that the chart is adjusted immediately when the EA starts running. If, for any reason, the platform fails to apply a setting, the EA reports the issue and stops initialization.

With this addition, the development of our test EA is complete. In the next section, we will begin running these experiments across different markets. We will observe the results, compare outcomes across asset classes, and finally answer the question that started this journey. Are markets truly random, or do they leave measurable footprints behind?

### Test Execution and Market Behavior Observations

Up to this point, our focus has been on ideas and preparation. We explored Larry Williams’ argument that price behavior is not entirely random, translated those ideas into clearly defined test cases, and built an Expert Advisor capable of executing those tests consistently.

The EA is intentionally simple. It is not optimized; it has no stop loss nor take profit, and it does not manage trades dynamically. Each position opens at the start of a new bar and closes at the end of that same bar. Only one trade can be open at a time, and each test runs independently from the others.

Our goal here is not to build a profitable trading system, nor to optimize entries, exits, or risk management. Instead, we want to answer a much simpler and more fundamental question:

_Do certain price behaviors occur more often than randomness would suggest, and can those behaviors be observed consistently across different markets_ _?_

Test Environment

To ensure that the results of our experiments are consistent, comparable, and reproducible, all tests are conducted under the same fixed conditions. No parameters are adjusted from one market to another, and no optimization is performed at any stage.

All experiments are run on the Daily (D1) timeframe. Using a higher timeframe helps reduce market microstructure noise and ensures that each trade reflects a complete trading session rather than short-term price fluctuations.

The test period spans from 01.01.2025 to 30.11.2025 for every financial instrument. This identical date range allows us to compare behavior across markets without introducing time-based bias.

A fixed lot size (0.1) is used throughout all experiments. The EA opens only one trade at a time, and each trade is held for exactly one bar. There is no pyramiding, scaling, or overlapping positions. This ensures that every outcome is independent and directly attributable to the test condition being evaluated.

All tests start with the same initial balance of $10,000 USD and use 1:500 leverage. The trading rules, execution logic, and exit behavior remain unchanged across all markets and test cases. This uniform setup is critical for isolating market behavior rather than strategy tuning.

To make replication straightforward, a configuration file ( _configurations.ini_) is provided. When running the EA in the [_MetaTrader 5 Strategy Tester_](https://www.metatrader5.com/en/automated-trading/strategy-tester "https://www.metatrader5.com/en/automated-trading/strategy-tester"), this file can be loaded from the _Settings_ tab by clicking “Load” and selecting the provided **.ini** file. This applies the same testing parameters used in these experiments, allowing readers to follow along with identical conditions.

Financial Securities Under Test

To evaluate how non-random market behavior appears across different market types, we run all tests on the following financial instruments:

- XAUUSD – A Commodity (Gold)
- BTCUSD – A Cryptocurrency (Bitcoin)
- GBPUSD – A Currency pair (British Pound vs US Dollar)
- NAS100  – A Stock index (Nasdaq 100)

These instruments were selected deliberately from different, largely unrelated asset classes. Each market operates under different conditions regarding liquidity, participant behavior, trading hours, and sensitivity to news and sentiment.

By applying the same test logic to commodities, cryptocurrencies, currencies, and equity indices, we can determine whether the detected non-random behavior is specific to certain markets or appears consistently across asset classes.

This comparison is essential for separating market structure effects from instrument-specific characteristics. This lays the foundation for the test results that follow, in which each instrument is evaluated independently under identical rules.

Running Experiments and Recording Results

To maintain data integrity and prevent cross-contamination of results, each test case will be conducted independently. The findings for each experiment will be documented in its own dedicated table to ensure maximum clarity. For every experiment, we will record the following fields in each table:

- Instrument – The financial market being tested (XAUUSD, BTCUSD, GBPUSD, NAS100).
- Trade Count – The total number of trades triggered by this scenario during the test period.
- Win Rate (%) – The percentage of trades that closed profitably.
- Net Result - A flag showing whether the total outcome of trades was net positive (Profitable) or net negative (Losing).

This structured approach allows us to compare results across markets and scenarios while keeping the data clean and easy to analyze.

**Experiment 1: Open-to-Close Bias**

The Open-to-Close Bias test examines whether candles tend to close higher or lower than their opening price. This test is the most fundamental scenario and gives us insight into the basic directional tendencies of each market during the chosen timeframe.

|  | Trade Count | Win Rate (%) | Net Results |
| --- | --- | --- | --- |
| XAUUSD | 236 | 56.36 | Profitable |
| BTCUSD | 331 | 51.06 | Loosing |
| GBPUSD | 238 | 45.38 | Loosing |
| NASDAQ | 236 | 55.93 | Profitable |

**Experiment 2: Bullish Bias After One Down Close**

This test examines whether a bullish candle is more likely to follow a single bearish candle. It helps us understand if the market tends to reverse after a single down bar. The EA will check whether a bullish candle follows a bearish candle.

|  | Trade Count | Win Rate (%) | Net Results |
| --- | --- | --- | --- |
| XAUUSD | 101 | 62.38 | Profitable |
| BTCUSD | 161 | 53.42 | Profitable |
| GBPUSD | 106 | 46.23 | Loosing |
| NASDAQ | 106 | 62.26 | Profitable |

**Experiment 3: Bullish Bias After Two Consecutive Down Closes**

This test investigates whether two consecutive bearish candles increase the probability of a bullish reversal on the next candle.

|  | Trade Count | Win Rate (%) | Net Results |
| --- | --- | --- | --- |
| XAUUSD | 35 | 57.14 | Profitable |
| BTCUSD | 73 | 50.68 | Profitable |
| GBPUSD | 47 | 44.68 | Loosing |
| NASDAQ | 40 | 57.50 | Loosing |

**Experiment 4: Bullish Bias After Three Consecutive Down Closes**

This test extends the previous one to three consecutive bearish candles, examining if a stronger downtrend increases the likelihood of an upward reversal.

|  | Trade Count | Win Rate (%) | Net Results |
| --- | --- | --- | --- |
| XAUUSD | 14 | 71.43 | Profitable |
| BTCUSD | 35 | 60.00 | Profitable |
| GBPUSD | 20 | 35.00 | Loosing |
| NASDAQ | 19 | 52.63 | Profitable |

**Experiment 5: Bullish Bias After One Up Close**

This test evaluates if a bullish candle tends to be followed by another bullish candle, revealing any short-term momentum tendencies.

|  | Trade Count | Win Rate (%) | Net Results |
| --- | --- | --- | --- |
| XAUUSD | 135 | 51.85 | Profitable |
| BTCUSD | 170 | 48.82 | Loosing |
| GBPUSD | 131 | 45.04 | Loosing |
| NASDAQ | 130 | 50.77 | Loosing |

**Experiment 6: Bullish Bias After Two Consecutive Up Closes**

This test examines whether momentum strengthens after two consecutive bullish candles and whether a third is likely to follow.

|  | Trade Count | Win Rate (%) | Net Results |
| --- | --- | --- | --- |
| XAUUSD | 69 | 53.62 | Profitable |
| BTCUSD | 82 | 43.90 | Loosing |
| GBPUSD | 72 | 41.67 | Loosing |
| NASDAQ | 64 | 54.69 | Loosing |

**Experiment 7: Bullish Bias After Three Consecutive Up Closes**

This test examines whether three consecutive bullish candles increase the likelihood of a fourth, revealing stronger momentum.

|  | Trade Count | Win Rate (%) | Net Results |
| --- | --- | --- | --- |
| XAUUSD | 37 | 37.84 | Loosing |
| BTCUSD | 34 | 52.94 | Profitable |
| GBPUSD | 37 | 40.54 | Loosing |
| NASDAQ | 34 | 50.00 | Loosing |

**Experiment 8: Bullish Bias After Larry Williams Short-Term Low**

This test examines whether a bullish candle is likely to follow a short-term swing low as defined by Larry Williams. It combines price action and swing analysis for potential predictive behavior.

|  | Trade Count | Win Rate (%) | Net Results |
| --- | --- | --- | --- |
| XAUUSD | 21 | 61.90 | Profitable |
| BTCUSD | 26 | 53.85 | Profitable |
| GBPUSD | 23 | 65.22 | Loosing |
| NASDAQ | 29 | 62.07 | Profitable |

Evidence of Non-Random Market Behavior

This section brings together all eight experiments and focuses on what truly matters—the observed win rates. If markets were purely random, bullish and bearish closes would be evenly distributed around 50%, regardless of prior conditions. What we observe here is different.

Across multiple instruments and multiple test cases, price behavior changes depending on what happened before. This alone challenges the assumption that each candle is an independent event with no memory.

**1\. Open-to-Close Bias Is Not Uniform**

The open-to-close test sets a baseline. If markets were random, results across instruments should cluster tightly around fifty percent. Instead, we see divergence. _Gold_ and _Nasdaq_ show win rates above fifty-five percent, and both end profitably. _Bitcoin_ stays close to random and turns unprofitable. _GBPUSD_ drops well below fifty percent and performs poorly. This tells us something important. Even the most basic price behavior is not identical across markets: structure and participant behavior matter.

**2\. Down Closes Reveal Stronger Predictive Behavior**

The most substantial evidence against randomness appears after bearish candles. After one down close, _Gold_ and _Nasdaq_ show win rates above sixty percent. _Bitcoin_ moves above fifty percent and becomes profitable. _GBPUSD_ remains weak.

After two consecutive down closes, _Gold_ and _Bitcoin_ still maintain win rates above fifty percent. Even though _Nasdaq_ loses profitability, the win rate remains elevated.

After three consecutive down closes, the effect becomes clearer. _Gold_ reaches over seventy percent. _Bitcoin_ remains strong. _Nasdaq_ stays above fifty percent. Only _GBPUSD_ continues to fail.

This progression matters. In a random system, increasing the number of prior bearish candles should not increase bullish probability. Here, it does, especially in _Gold_ and _Bitcoin_.

This is not noise. This is conditional behavior.

**3\. Up Closes Behave Very Differently**

The same logic applied to bullish sequences yields weaker, less consistent results.

After one up close, most instruments fall below fifty percent. After two up-closes, only _Gold_ maintains a slight edge. After three up-closes, results continue to deteriorate.

This asymmetry is critical. Markets respond differently to weakness than to strength. Mean reversion appears firmer after declines than continuation after advances.

Random systems do not show asymmetry.

**4\. Short-Term Lows Confirm Structural Behavior**

The short-term low test delivers some of the most consistent results.

_Gold_, _Bitcoin_, and the _Nasdaq_ all show win rates above 60% and finish profitable. Even _GBPUSD_ has a high win rate, though it fails to convert those wins into profitability.

This confirms that simple structural patterns matter. A well-defined short-term low changes the probability of the next candle.

This directly supports Larry Williams’ argument that price does not move blindly from bar to bar.

**5\. Asset Class Differences Strengthen the Case**

Results vary by instrument, but the variation itself is meaningful.

_Gold_ consistently shows the most substantial bullish bias across multiple tests. _Bitcoin_ shows conditional behavior that strengthens after declines. _Nasdaq_ responds well to short-term weakness and structure. _GBPUSD_ remains largely inefficient under these rules.

If markets were random, asset classes would not matter. It clearly does.

**Final Insight**

These experiments do not claim certainty. They do not promise prediction. What they demonstrate is probability distortion.

Certain conditions shift the odds away from fifty percent. That alone disproves pure randomness.

Markets are not entirely predictable. However, they are not memoryless either. Price reacts to prior behavior in measurable ways. That is the foundation of edge.

This is precisely what Larry Williams proposed. Moreover, through automation, data, and controlled testing, we can now see it for ourselves.

### Conclusion

This article set out to answer a simple but important question. Do markets move in a purely random way, or does past price action influence what comes next?

Through controlled testing, automation, and real market data, we have shown that price behavior is not evenly distributed. Certain conditions consistently shift the probabilities away from 50%. That alone is enough to reject the idea of an entirely random market.

The results did not suggest certainty or perfect prediction. Instead, they revealed something more practical. Markets respond differently after weakness than after strength. Some structures repeat more often than chance would allow. Some instruments display these tendencies more clearly than others. These are not coincidences. They are measurable behaviors.

Just as important as the results is the process used to obtain them. You were taken step by step through the design and development of a testing Expert Advisor. The EA was intentionally simple, transparent, and modular. It was built to test ideas, not to optimize profits. This approach is essential when searching for genuine edges.

With this foundation, you now have a working framework. You can modify the logic, add new conditions, test different timeframes, or explore other price patterns. The same techniques can be reused to build research tools, trading systems, or strategy components, grounded in clear evidence rather than assumptions.

This conclusion supports Larry Williams’ core idea. Markets are not entirely predictable, but they are not memoryless either. When this is understood, system development becomes meaningful.

In the following parts of this series, we will build on these findings and begin constructing trading systems that aim to capture these non-random behaviors in a structured, repeatable way.

For your convenience, the following files are attached so you can follow along or reproduce the experiments exactly.

|  | File Name | Description |
| --- | --- | --- |
| 1. | larryWilliamsNonRandomMarketBehaviorTester.mq5 | The full source code of the Expert Advisor developed in this article. |
| 2. | configurations.ini | The configuration file used for all strategy tester runs. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20510.zip "Download all attachments in the single ZIP archive")

[larryWilliamsNonRandomMarketBehaviorTester.mq5](https://www.mql5.com/en/articles/download/20510/larryWilliamsNonRandomMarketBehaviorTester.mq5 "Download larryWilliamsNonRandomMarketBehaviorTester.mq5")(13.89 KB)

[configurations.ini](https://www.mql5.com/en/articles/download/20510/configurations.ini "Download configurations.ini")(1.01 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Larry Williams Market Secrets (Part 6): Measuring Volatility Breakouts Using Market Swings](https://www.mql5.com/en/articles/20862)
- [Larry Williams Market Secrets (Part 5): Automating the Volatility Breakout Strategy in MQL5](https://www.mql5.com/en/articles/20745)
- [Larry Williams Market Secrets (Part 4): Automating Short-Term Swing Highs and Lows in MQL5](https://www.mql5.com/en/articles/20716)
- [Larry Williams Market Secrets (Part 2): Automating a Market Structure Trading System](https://www.mql5.com/en/articles/20512)
- [Larry Williams Market Secrets (Part 1): Building a Swing Structure Indicator in MQL5](https://www.mql5.com/en/articles/20511)
- [Mastering Kagi Charts in MQL5 (Part 2): Implementing Automated Kagi-Based Trading](https://www.mql5.com/en/articles/20378)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/503222)**
(1)


![Hezekiah Bukola Oyetunde](https://c.mql5.com/avatar/2025/12/692df20f-eed4.jpg)

**[Hezekiah Bukola Oyetunde](https://www.mql5.com/en/users/hezetrade)**
\|
13 Jan 2026 at 14:31

Thanks for sharing the [source code](https://forge.mql5.io/help/en/guide "MQL5 Algo Forge: Cloud Workspace for Algorithmic Trading Development"). I really like how you structured the ENUM\_NON\_RANDOM\_TEST\_MODE to keep the experiments isolated. Have you considered adding a 'Spread' or 'Commission' filter to the results? It would be interesting to see if these statistical biases remain profitable once real-world trading costs are factored in.


![Building AI-Powered Trading Systems in MQL5 (Part 8): UI Polish with Animations, Timing Metrics, and Response Management Tools](https://c.mql5.com/2/189/20722-building-ai-powered-trading-logo.png)[Building AI-Powered Trading Systems in MQL5 (Part 8): UI Polish with Animations, Timing Metrics, and Response Management Tools](https://www.mql5.com/en/articles/20722)

In this article, we enhance the AI-powered trading system in MQL5 with user interface improvements, including loading animations for request preparation and thinking phases, as well as timing metrics displayed in responses for better feedback. We add response management tools like regenerate buttons to re-query the AI and export options to save the last response to a file, streamlining interaction.

![Market Simulation (Part 08): Sockets (II)](https://c.mql5.com/2/120/Simula92o_de_mercado_Parte_08__LOGO.png)[Market Simulation (Part 08): Sockets (II)](https://www.mql5.com/en/articles/12672)

How about creating something practical using sockets? In today's article, we'll start creating a mini-chat. Let's look together at how this is done - it will be very interesting. Please note that the code provided here is for educational purposes only. It should not be used for commercial purposes or in ready-made applications, as it does not provide data transfer security and the content transmitted over the socket can be accessed.

![Introduction to MQL5 (Part 33): Mastering API and WebRequest Function in MQL5 (VII)](https://c.mql5.com/2/189/20700-introduction-to-mql5-part-33-logo.png)[Introduction to MQL5 (Part 33): Mastering API and WebRequest Function in MQL5 (VII)](https://www.mql5.com/en/articles/20700)

This article demonstrates how to integrate the Google Generative AI API with MetaTrader 5 using MQL5. You will learn how to structure API requests, handle server responses, extract AI-generated content, manage rate limits, and save the results to a text file for easy access.

![Successful Restaurateur Algorithm (SRA)](https://c.mql5.com/2/124/Successful_Restaurateur_Algorithm___LOGO_2.png)[Successful Restaurateur Algorithm (SRA)](https://www.mql5.com/en/articles/17380)

Successful Restaurateur Algorithm (SRA) is an innovative optimization method inspired by restaurant business management principles. Unlike traditional approaches, SRA does not discard weak solutions, but improves them by combining with elements of successful ones. The algorithm shows competitive results and offers a fresh perspective on balancing exploration and exploitation in optimization problems.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=pmzvhtaqntrlsrlyuwdpegvsyvnzaftd&ssn=1769181975246979764&ssn_dr=0&ssn_sr=0&fv_date=1769181975&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F20510&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Larry%20Williams%20Market%20Secrets%20(Part%203)%3A%20Proving%20Non-Random%20Market%20Behavior%20with%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691819750678261&fz_uniq=5069437620229965078&sv=2552)

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