---
title: Developing a multi-currency Expert Advisor (Part 8): Load testing and handling a new bar
url: https://www.mql5.com/en/articles/14574
categories: Trading Systems, Integration
relevance_score: 9
scraped_at: 2026-01-22T17:37:10.097268
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/14574&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049255830898976788)

MetaTrader 5 / Tester


### Introduction

In the first [article](https://www.mql5.com/en/articles/14026), we have developed an EA featuring two instances of trading strategies. In the [second](https://www.mql5.com/en/articles/14107) one, we have already used nine instances, while in the last one this number jumped to 32. There were no problems with testing time. It is clear that the shorter the time of a single test pass, the better. But if the overall optimization takes about a few hours, this is still better than several days or weeks. Likewise, if we have combined several strategy instances in one EA and want to see its results, then a single pass should be completed in seconds or minutes, not hours or days.

If we perform optimization to select groups of strategy instances, then several instances already participate in all optimization passes. Then the time spent on individual passes and on the entire optimization in general increases. Therefore, we limited ourselves to selecting groups of no more than eight instances with this optimization.

Let's try to find out how the time of a single pass in the tester depends on the number of instances of trading strategies for testing periods of different durations. Let's also look at the memory consumed. Of course, we need to see how EAs behave with different numbers of trading strategy instances when launched on the terminal chart.

### Different number of instances in the tester

To conduct such an experiment, we will need to write a new EA based on one of the existing ones. Let's take the _OptGroupExpert.mq5_ EA as a basis and make the following changes to it:

- Let's remove the inputs that specify the indices of the eight parameter sets that were taken from the full array of sets loaded from the file. Let's leave the _count\__ parameter, which will now specify the number of sets to load from the full array of sets.
- Let's remove the check for uniqueness of indices that no longer exist. We will add new strategies to the array of strategies, with sets of parameters taken from the first _count\__ elements of the params set array. If there are not enough instances in this array, then we will take the new ones in the loop from the beginning of the array.
- Let's remove the _OnTesterInit()_ and _OntesterDeinit()_ functions, since we will not use this EA for optimization of anything yet.

We will receive the following code:

```
//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
input group "::: Money management"
sinput double expectedDrawdown_ = 10;  // - Maximum risk (%)
sinput double fixedBalance_ = 10000;   // - Used deposit (0 - use all) in the account currency
sinput double scale_ = 1.00;           // - Group scaling multiplier

input group "::: Selection for the group"
sinput string fileName_ = "Params_SV_EURGBP_H1.csv";  // - File with strategy parameters (*.csv)
input int     count_ = 8;              // - Number of strategies in the group (1 .. 8)

input group "::: Other parameters"
sinput ulong  magic_        = 27183;   // - Magic

...

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
// Load strategy parameter sets
   int totalParams = LoadParams(fileName_, params);

// If nothing is loaded, report an error
   if(totalParams == 0) {
      PrintFormat(__FUNCTION__" | ERROR: Can't load data from file %s.\n"
                  "Check that it exists in data folder or in common data folder.",
                  fileName_);
      return(INIT_PARAMETERS_INCORRECT);
   }

// Report an error if
   if(count_ < 1) { // number of instances is less than 1
      return INIT_PARAMETERS_INCORRECT;
   }

   ArrayResize(params, count_);

// Set parameters in the money management class
   CMoney::DepoPart(expectedDrawdown_ / 10.0);
   CMoney::FixedBalance(fixedBalance_);

// Create an EA handling virtual positions
   expert = new CVirtualAdvisor(magic_, "SimpleVolumes_BenchmarkInstances");

// Create and fill the array of all strategy instances
   CVirtualStrategy *strategies[];

   FOREACH(params, APPEND(strategies, new CSimpleVolumesStrategy(params[i % totalParams])));

// Form and add a group of strategies to the EA
   expert.Add(CVirtualStrategyGroup(strategies, scale_));

   return(INIT_SUCCEEDED);
}
```

Save the resulting code in the _BenchmarkInstancesExpert.mq5_ file of the current folder.

Let's now try to run this EA several times in the tester with different numbers of trading strategy instances and different tick simulation modes.

### Test results for different modes

Let's start with the already familiar "1 minute OHLC" tick simulation mode, which we used in all previous articles. We will double the number of instances at the next launch. Let's start with 8 instances. If the test time becomes too long, we will reduce the test period.

![](https://c.mql5.com/2/91/5578477918408.png)

Fig. 1. Results of single runs in "1 minute OHLC" mode

As you can see, when testing with up to 512 instances, we used a 6 year test period, then switched to a 1 year period, and for the last two passes we used only 3 months.

In order to be able to compare the time costs for different test periods, we will calculate a separate value: the simulation time of one strategy instance over the course of one day. To do this, divide the total time by the number of strategy instances and by the duration of the test period in days. To avoid struggling with small numbers, let's convert this time into nanoseconds by multiplying by 10^9.

In the logs, the tester reports information about the memory used during the run, indicating the total volume and the volumes spent on historical and tick data. Subtracting them from the total memory volume, we get the amount of memory that the EA itself required.

Based on the results, we can say that even the maximum number of copies (16,384) does not require a catastrophically large amount of time to run the tester. Generally, such a number of copies is quite sufficient for arranging a joint work, for example, of fifteen symbols with a hundred instances on each. So that is already a lot. At the same time, memory consumption does not increase much with the increase in the number of instances. For some reason, there is a peak in memory consumption for the EA itself at 8192 instances, but then less memory was required again.

For more accurate results, we can repeat several passes for each number of instances and calculate the average times and average memory sizes, since the results still differed on different passes with the same number of instances. But these differences were not very large, so there was little point in conducting more extensive tests. We just wanted to make sure that we would not run into limitations even with relatively small quantities of copies.

Let's now try to look at the results of running the EA in the tester in the "Every tick" simulation mode.

![](https://c.mql5.com/2/91/1438193801219.png)

Fig. 2. Results of single runs in "Every tick" mode

The time for one pass increased by about 10 times, so we reduced the size of the testing period on the same number of instances compared to the previous mode. The size of the tick memory has naturally increased, which has led to an increase in the total amount of allocated memory. But the amount of memory allocated to the EA turned out to be almost the same for all the numbers of instances used. There is some growth, but it is quite slow.

Anomalously low running times were observed for 512 and 1024 instances - almost twice as fast as for other instance sizes. The possible reason is most likely related to the order of the trading strategy instance parameter sets in the CSV data file.

The last simulation mode to be explored is "Every tick based on real ticks". We ran a few more runs for it than for the "Every tick" mode.

![](https://c.mql5.com/2/91/2860295477876.png)

Fig. 3. Results of single runs in "Every tick based on real ticks" mode

Compared to the previous mode, the time has increased by about 30%, and the memory used has increased by about 20%.

It is worth noting that one copy of the EA attached to the chart was running in the terminal during the test. It used 8192 instances. In this case, the terminal memory consumption was about 200 MB, while CPU resources consumption ranged from 0% to 4%.

Overall, the experiment showed that we have a fairly large reserve for the possible number of instances of trading strategies that will work together in one EA. Of course, this amount will largely depend on the specific content of trading strategies. The more calculations one instance needs to perform, the fewer we can afford to combine.

Let's now think about what simple steps we can take to speed up the test.

### Disabling output

In the current implementation, we display quite a lot of information during the EA's operation. When it comes to optimizing single instances, this is not a problem since the output functions are simply not executed. If we run a single pass of the EA in the tester, all messages are sent to the log. In the applied _VirtualOrder.mqh_ library, we display a message about handling events of each virtual order. When there are few virtual orders, this has little effect on testing time, but when their number starts to amount to tens of thousands, this can have a noticeable effect.

Let's try to measure it. We can disable the output of all our messages to the log by adding the following line at the beginning of the EA file:

```
#define PrintFormat StringFormat
```

Due to the relatedness of these functions, all _PrintFormat()_ calls can be replaced with _StringFormat()_ ones. They will generate a string, but will not output it to the log.

After conducting several launches, some of them showed a 5-10% reduction in time, while in others the time could even increase slightly. We might still need a similar method of _PrintFormat()_ replacement in the future.

### Migration to 1 minute OHLC

Another way to speed up the process of both single test passes and optimization is to avoid using the "Every tick" and "Every tick based on real ticks" simulation modes.

It is clear that not all trading strategies can afford this. If the strategy involves very frequent opening/closing of positions (more than once per minute), then it is impossible to abandon testing on all ticks. Even high-frequency trading does not last all the time, but only during designated time periods. But if the strategy does not require frequent opening/closing and is not so sensitive to the loss of several points due to insufficiently accurate triggering of Stop Loss and Take Profit levels, then why not take advantage of this opportunity?

The trading strategy considered as an example is one of those that allows us to get away from using the all-tick mode. However, another problem arises here. If we simply optimize the parameters of single instances in the "1 minute OHLC" mode, and then put the assembled EA to work in the terminal, then the EA will have to work in every tick mode. It will not receive a fixed 4 ticks per minute, but much more. Therefore, the _OnTick()_ function will be called more often, and the set of prices that the EA handles will be a little more diverse.

This difference may change the picture of the results shown by the EA. To check how realistic this scenario is, let's compare the trading results obtained when testing the EA with the same inputs in the "1 minute OHLC" and "Every tick based on real ticks" modes.

![](https://c.mql5.com/2/75/4445248982504.png)

Fig. 4. Comparing the results of single runs in the

"Every tick based on real ticks" (left) and "1 minute OHLC" (right)

modes

We can see that for different modes the opening, closing time and price are slightly different. At first, this is the only difference, but then there comes the moment when on the left we see a deal opening, and on the right at the same time there is no opening: look at the lines with deal #25. Thus, the results for the "1 minute OHLC" mode contain fewer trades than for the "Every tick based on real ticks" mode.

In every tick mode, the profit turned out to be slightly higher. If we look at the balance curve, there are no significant differences between them:

![](https://c.mql5.com/2/75/5814493145130.png)

![](https://c.mql5.com/2/75/3478716810599.png)

Fig. 5. Test results in "1 minute OHLC" (top) and "Every tick based on real ticks" (bottom)

Therefore, when running this EA in the terminal, we will most likely get results no worse than in the test in the "1 minute OHLC" mode. This means that a faster tick simulation mode can be used for optimization. If some calculations for a strategy can only be performed at the beginning of a new bar, then we can further speed up the work of the EA by refusing such calculations at each tick. To do this, we need a way to determine a new bar in the EA.

If the results in the every tick mode are worse than in the "1 minute OHLC" mode, we can try to prohibit the EA from performing transactions not at the beginning of the bar. In this case, we should get the closest possible results in all tick modeling modes. To achieve this, we again need a way to define a new bar in the EA.

### Defining a new bar

Let's first formulate our wishes. We would like to have one function that returns true if a new bar has occurred on a given symbol and a given timeframe. When developing an EA that implements a single instance of a trading strategy, such a function is usually implemented for one symbol and timeframe (or for one symbol and several timeframes), using variables to store the time of the last bar. It is often possible to see that the code implementing this functionality is not allocated as a separate function, but is implemented in the only place where it is needed.

This approach becomes quite inconvenient when it is necessary to perform multiple checks for the occurrence of a new bar for different instances of trading strategies. It is possible, of course, to embed this code directly into the implementation of a trading strategy instance, but we will do it differently.

Let's make the _IsNewBar(symbol, timeframe)_ public function, which should be able to report the occurrence of a new bar on the current tick by _symbol_ and _timeframe_. It is desirable that, in addition to calling this function, no additional variables or actions need to be added to the trading logic code of the strategies. Also, if a new bar has arrived on the current tick, and the function is called several times (for example, from different instances of trading strategies), then it should return 'true' on each call, not just on the first one.

Then we will need to store information about the times of the last bar for each symbol and timeframe. But by "each" we mean not all that are available in the terminal, but only those that are actually required for the operation of specific instances of trading strategies. To define the range of these necessary symbols and timeframes, we will expand the list of actions performed by the _IsNewBar(symbol, timeframe)_ function. Let it first check if there is some form of remembered time for the current bar on the given symbol and timeframe. If it does not exist, the function will create such a time storage location. If it exists, the function returns the result of checking for a new bar.

In order for our _IsNewBar()_ function to be called multiple times on a single tick, we will have to split it into two separate functions. One will check for new bars at the beginning of a tick for all symbols and timeframes of interest and save this information for the second function, which will simply find the desired result of the new bar occurrence event and return it. Let's name the first function _UpdateNewBar()_. We will make it so that it also returns a logical value showing that at least one symbol and timeframe has a new bar.

The _UpdateNewBar()_ function should be called once at the beginning of handling a new tick. For example, its call can be placed at the beginning of the _CVirtualAdvisor::Tick()_ method:

```
void CVirtualAdvisor::Tick(void) {
// Define a new bar for all required symbols and timeframes
   UpdateNewBar();

   ...
// Start handling in strategies where IsNewBar(...) can already be used
   CAdvisor::Tick();

   ...
}
```

To arrange the storage of the times of the last bars, first create the _CNewBarEvent_ static class. This means that we will not create objects of this class, but will only use its static properties and methods. This is essentially equivalent to creating the required global variables and functions in a dedicated namespace.

In this class, we will have two arrays: the array of symbol names ( _m\_symbols_) and the array of pointers to the new class objects ( _m\_symbolNewBarEvent_). The first one will contain the symbols we will use to track the new bar events. The second one is pointers to the new _CSymbolNewBarEvent_ class, which will store bar times for one symbol, but for different timeframes.

These two classes will have three methods:

- The method for registering a new monitored symbol or symbol timeframe _Register(...)_
- The method for updating new bar flags _Update()_
- The method for obtaining a new bar flag _IsNewBar(...)_

If it is necessary to register a tracking of a new bar event on a new symbol, a new class object _CSymbolNewBarEvent_ is created. Therefore, it is necessary to take care of cleaning up the memory occupied by these objects when the EA completes its work. This task is fulfilled by the _CNewBarEvent::Destroy()_ static method and the _DestroyNewBar()_ global function. We will add the function call to the EA destructor:

```
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
void CVirtualAdvisor::~CVirtualAdvisor() {
   delete m_receiver;         // Remove the recipient
   delete m_interface;        // Remove the interface
   DestroyNewBar();           // Remove the new bar tracking objects
}
```

A complete implementation of these classes might look something like this:

```
//+------------------------------------------------------------------+
//| Class for defining a new bar for a specific symbol               |
//+------------------------------------------------------------------+
class CSymbolNewBarEvent {
private:
   string            m_symbol;         // Tracked symbol
   long              m_timeFrames[];   // Array of tracked symbol timeframes
   long              m_timeLast[];     // Array of times of the last bars for timeframes
   bool              m_res[];          // Array of flags of a new bar occurrence for timeframes

   // The method for registering a new tracked timeframe for a symbol
   int               Register(ENUM_TIMEFRAMES p_timeframe) {
      APPEND(m_timeFrames, p_timeframe);  // Add it to the timeframe array
      APPEND(m_timeLast, 0);              // The last time bar for it is still unknown
      APPEND(m_res, false);               // No new bar for it yet
      Update();                           // Update new bar flags
      return ArraySize(m_timeFrames) - 1;
   }

public:
   // Constructor
                     CSymbolNewBarEvent(string p_symbol) :
                     m_symbol(p_symbol) // Set a symbol
   {}

   // Method for updating new bar flags
   bool              Update() {
      bool res = (ArraySize(m_res) == 0);
      FOREACH(m_timeFrames, {
         // Get the current bar time
         long time = iTime(m_symbol, (ENUM_TIMEFRAMES) m_timeFrames[i], 0);
         // If it does not match the saved one, it is a new bar
         m_res[i] = (time != m_timeLast[i]);
         res |= m_res[i];
         // Save the new time
         m_timeLast[i] = time;
      });
      return res;
   }

   // Method for getting the new bar flag
   bool              IsNewBar(ENUM_TIMEFRAMES p_timeframe) {
      int index;
      // Search for the required timeframe index
      FIND(m_timeFrames, p_timeframe, index);

      // If not found, then register a new timeframe
      if(index == -1) {
         PrintFormat(__FUNCTION__" | Register new event handler for %s %s", m_symbol, EnumToString(p_timeframe));
         index = Register(p_timeframe);
      }

      // Return the new bar flag for the necessary timeframe
      return m_res[index];
   }
};

//+------------------------------------------------------------------+
//| Static class for defining a new bar for all                      |
//| symbols and timeframes                                           |
//+------------------------------------------------------------------+
class CNewBarEvent {
private:
   // Array of objects to define a new bar for one symbol
   static   CSymbolNewBarEvent     *m_symbolNewBarEvent[];

   // Array of required symbols
   static   string                  m_symbols[];

   // Method to register new symbol and timeframe to track a new bar
   static   int                     Register(string p_symbol)  {
      APPEND(m_symbols, p_symbol);
      APPEND(m_symbolNewBarEvent, new CSymbolNewBarEvent(p_symbol));
      return ArraySize(m_symbols) - 1;
   }

public:
   // There is no need to create objects of this class - delete the constructor
                            CNewBarEvent() = delete;

   // Method for updating new bar flags
   static bool              Update() {
      bool res = (ArraySize(m_symbolNewBarEvent) == 0);
      FOREACH(m_symbols, res |= m_symbolNewBarEvent[i].Update());
      return res;
   }

   // Method to free memory for automatically created objects
   static void              Destroy() {
      FOREACH(m_symbols, delete m_symbolNewBarEvent[i]);
      ArrayResize(m_symbols, 0);
      ArrayResize(m_symbolNewBarEvent, 0);
   }

   // Method for getting the new bar flag
   static bool              IsNewBar(string p_symbol, ENUM_TIMEFRAMES p_timeframe) {
      int index;
      // Search for the required symbol index
      FIND(m_symbols, p_symbol, index);

      // If not found, then register a new symbol
      if(index == -1) index = Register(p_symbol);

      // Return the new bar flag for the necessary symbol and timeframe
      return m_symbolNewBarEvent[index].IsNewBar(p_timeframe);
   }
};

// Initialize static members of the CSymbolNewBarEvent class members;
CSymbolNewBarEvent* CNewBarEvent::m_symbolNewBarEvent[];
string CNewBarEvent::m_symbols[];

//+------------------------------------------------------------------+
//| Function for checking a new bar occurrence                       |
//+------------------------------------------------------------------+
bool IsNewBar(string p_symbol, ENUM_TIMEFRAMES p_timeframe) {
   return CNewBarEvent::IsNewBar(p_symbol, p_timeframe);
}

//+------------------------------------------------------------------+
//| Function for updating information about new bars                 |
//+------------------------------------------------------------------+
bool UpdateNewBar() {
   return CNewBarEvent::Update();
}

//+------------------------------------------------------------------+
//| Function for removing new bar tracking objects                   |
//+------------------------------------------------------------------+
void DestroyNewBar() {
   CNewBarEvent::Destroy();
}
//+------------------------------------------------------------------+
```

Save this code in the _NewBarEvent.mqh_ of the current folder.

Let's now see how this library can be applied in a trading strategy and EA. But first, let's make some minor adjustments to the trading strategy that are not related to handling a new bar.

### Trading strategy improvements

Unfortunately, during the writing of this article, two errors were discovered in the strategy used. They did not have a significant impact on the previous results, but since they were detected, let's fix them.

The first error resulted in the fact that when a negative value of _openDistance\__ was set in the parameters, it was reset to a small positive number equal to the spread for the current symbol. In other words, instead of opening BUY STOP and SELL\_STOP pending orders, market positions were opened. This meant that during optimization we did not see the results that could have been achieved by trading such pending orders. This means we missed out on some potentially profitable sets of parameters.

The error occurred in the following string of code in the functions for opening pending orders of the _SimpleVolumesStrategy.mqh_ file:

```
// Let's make sure that the opening distance is not less than the spread
   int distance = MathMax(m_openDistance, spread);
```

If _m\_openDistance_ turned out to be negative, then the _distance_ value of the shift in the opening price from the current one turned into a positive one. To save the same sign in _distance_ as in _m\_openDistance_, we simply need to multiply the following expression by it:

```
// Let's make sure that the opening distance is not less than the spread
   int distance = MathMax(MathAbs(m_openDistance), spread) * (m_openDistance < 0 ? -1 : 1);
```

The second error was that when calculating the average volume for the last few bars, the volume of the current bar was also included in the calculation. Although ,according to the strategy description, we should not use it to calculate the average. However, the impact of this error is probably also quite small. The longer the volume averaging period, the smaller the contribution to the average made by the last bar.

To fix this error, we only need to slightly change the function for calculating the average, excluding the very first element of the passed array:

```
//+------------------------------------------------------------------+
//| Average value of the array of numbers from the second element    |
//+------------------------------------------------------------------+
double CSimpleVolumesStrategy::ArrayAverage(const double &array[]) {
   double s = 0;
   int total = ArraySize(array) - 1;
   for(int i = 1; i <= total; i++) {
      s += array[i];
   }

   return s / MathMax(1, total);
}
```

Save the changes in the _SimpleVolumesStrategy.mqh_ file of the current folder.

### Considering a new bar in the strategy

In order for some actions in the trading strategy to be performed only when a new bar occurs, we only need to place this block of code in a conditional operator like this:

```
// If a new bar arrived on H1 for the current strategy symbol, then
if(IsNewBar(m_symbol, PERIOD_H1)) {

       // perform the necessary actions
   ...
}
```

The presence of such a code in the strategy will automatically lead to the registration of tracking the new bar event on H1 and in the _m\_symbol_ strategy symbol.

We can easily add checking for the occurrence of new bars on other additional timeframes. For example, if the strategy uses the values of some average price range (ATR or ADR), then its recalculation can be easily arranged only once a day the following way:

```
// If a new bar arrived on D1 for the current strategy symbol, then
if(IsNewBar(m_symbol, PERIOD_H1)) {
   CalcATR(); // call our ATR calculation function
}
```

In the trading strategy we are considering in this series of articles, we can completely exclude all actions outside the moment of the new bar arrival:

```
//+------------------------------------------------------------------+
//| "Tick" event handler function                                    |
//+------------------------------------------------------------------+
void CSimpleVolumesStrategy::Tick() override {
// If there is no new bar on M1,
   if(!IsNewBar(m_symbol, PERIOD_M1)) return;

// If their number is less than allowed
   if(m_ordersTotal < m_maxCountOfOrders) {
      // Get an open signal
      int signal = SignalForOpen();

      if(signal == 1 /* || m_ordersTotal < 1 */) {          // If there is a buy signal, then
         OpenBuyOrder();         // open the BUY_STOP order
      } else if(signal == -1) {  // If there is a sell signal, then
         OpenSellOrder();        // open the SELL_STOP order
      }
   }
}
```

We can also prohibit any processing in the EA's OnTick event handler at the times of those ticks that do not match the start of a new bar for any of the symbols or timeframes used. To achieve this, we can make the following changes to the _CVirtualAdvisor::Tick()_ method:

```
//+------------------------------------------------------------------+
//| OnTick event handler                                             |
//+------------------------------------------------------------------+
void CVirtualAdvisor::Tick(void) {
// Define a new bar for all required symbols and timeframes
   bool isNewBar = UpdateNewBar();

// If there is no new bar anywhere, and we only work on new bars, then exit
   if(!isNewBar && m_useOnlyNewBar) {
      return;
   }

// Receiver handles virtual positions
   m_receiver.Tick();

// Start handling in strategies
   CAdvisor::Tick();

// Adjusting market volumes
   m_receiver.Correct();

// Save status
   Save();

// Render the interface
   m_interface.Redraw();
}
```

In this code, we have added a new _m\_useOnlyNewBar_ EA property, which can be set when creating an EA object:

```
//+------------------------------------------------------------------+
//| Class of the EA handling virtual positions (orders)              |
//+------------------------------------------------------------------+
class CVirtualAdvisor : public CAdvisor {
protected:
   ...
   bool              m_useOnlyNewBar;  // Handle only new bar ticks

public:
                     CVirtualAdvisor(ulong p_magic = 1, string p_name = "",
                                     bool p_useOnlyNewBar = false); // Constructor
    ...
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CVirtualAdvisor::CVirtualAdvisor(ulong p_magic = 1,
                                 string p_name = "",
                                 bool p_useOnlyNewBar = false) :
// Initialize the receiver with a static receiver
   m_receiver(CVirtualReceiver::Instance(p_magic)),
// Initialize the interface with the static interface
   m_interface(CVirtualInterface::Instance(p_magic)),
   m_lastSaveTime(0),
   m_useOnlyNewBar(p_useOnlyNewBar) {
   m_name = StringFormat("%s-%d%s.csv",
                         (p_name != "" ? p_name : "Expert"),
                         p_magic,
                         (MQLInfoInteger(MQL_TESTER) ? ".test" : "")
                        );
};
```

We could have created a new EA class by inheriting it from _CVirtualAdvisor_ and adding a new property, as well as new bar presence verification to it. But we can leave everything as is, since with the default value for _m\_useOnlyNewBar = false_, everything works as it should without adding this functionality to the EA class.

If we have extended the EA class in this way, then inside the trading strategy class we can do without checking the new minute bar event inside the Tick() method. It is sufficient to call the _IsNewBar()_ function once in the strategy constructor with the current symbol and M1 timeframe to start tracking the event of a new bar with such a symbol and timeframe. In this case, the EA with _m\_useOnlyNewBar = true_ will simply not trigger tick handling for strategy instances unless there is a new bar on M1:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CSimpleVolumesStrategy::CSimpleVolumesStrategy(
   ...) :
// Initialization list
   ... {
   CVirtualReceiver::Get(GetPointer(this), m_orders, m_maxCountOfOrders);

// Load the indicator to get tick volumes
   m_iVolumesHandle = iVolumes(m_symbol, m_timeframe, VOLUME_TICK);

// Set the size of the tick volume receiving array and the required addressing
   ArrayResize(m_volumes, m_signalPeriod);
   ArraySetAsSeries(m_volumes, true);

// Register the event handler for a new bar on the minimum timeframe
   IsNewBar(m_symbol, PERIOD_M1);
}

//+------------------------------------------------------------------+
//| "Tick" event handler function                                    |
//+------------------------------------------------------------------+
void CSimpleVolumesStrategy::Tick() override {
// If their number is less than allowed
   if(m_ordersTotal < m_maxCountOfOrders) {
      // Get an open signal
      int signal = SignalForOpen();

      if(signal == 1 /* || m_ordersTotal < 1 */) {          // If there is a buy signal, then
         OpenBuyOrder();         // open the BUY_STOP order
      } else if(signal == -1) {  // If there is a sell signal, then
         OpenSellOrder();        // open the SELL_STOP order
      }
   }
}
```

Save the changes in the _SimpleVolumesStrategy.mqh_ file of the current folder.

### Results

The _BenchmarkInstancesExpert.mq5_ EA gets a new input _useOnlyNewBars\__, in which we set whether it should handle ticks that do not match the start of a new bar. When initializing the EA, pass the parameter value to the EA's constructor:

```
//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
...

input group "::: Other parameters"
sinput ulong  magic_          = 27183;   // - Magic
input bool    useOnlyNewBars_ = true;    // - Work only at bar opening

...

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   ...

// Create an EA handling virtual positions
   expert = new CVirtualAdvisor(magic_, "SimpleVolumes_BenchmarkInstances", useOnlyNewBars_);

   ...
}
```

Let's run a test on a small period with 256 instances of trading strategies in the "Every tick based on real ticks" mode - first with _useOnlyNewBars\_ = false_, then with _useOnlyNewBars\_ = true._

In the first case, when the EAs worked on every tick, the profit was USD 296, the run was completed in 04:15. In the second case, when the EA skipped all ticks except those at the beginning of a new bar, the profit was USD 434, the pass was completed at 00:25. So, we not only reduced the computational costs 10 times, but also received a slightly higher profit in the second case.

But we should not be too optimistic here. For other trading strategies, repeating similar results is by no means guaranteed. Each trading strategy should be examined separately.

### Conclusion

Let's take another look at the results achieved. We tested the EA's performance when a fairly large number of trading strategy instances were running simultaneously. This opens up prospects for good diversification of trading across different symbols, timeframes and trading strategies, as we will be able to combine them in one EA.

We also added new functionality to our class library - the ability to track new bar events. Although we do not really need this feature in the strategy under consideration, its presence can be very useful for implementing other trading strategies. Also, having the ability to limit the EA's operation to the start of a new bar can help reduce computing costs and achieve more similar results for testing in different tick simulation modes.

But again we have deviated a little from the intended project trajectory. Well, that can also help us achieve our ultimate goal. After having a little break, let's try to return with renewed vigor to the path of automating the EA testing. It seems that it is time to return to initializing instances of trading strategies using string constants and building a system for storing optimization results.

Thank you for your attention! See you soon!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14574](https://www.mql5.com/ru/articles/14574)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14574.zip "Download all attachments in the single ZIP archive")

[BenchmarkInstancesExpert.mq5](https://www.mql5.com/en/articles/download/14574/benchmarkinstancesexpert.mq5 "Download BenchmarkInstancesExpert.mq5")(13.23 KB)

[NewBarEvent.mqh](https://www.mql5.com/en/articles/download/14574/newbarevent.mqh "Download NewBarEvent.mqh")(11.52 KB)

[SimpleVolumesStrategy.mqh](https://www.mql5.com/en/articles/download/14574/simplevolumesstrategy.mqh "Download SimpleVolumesStrategy.mqh")(33.62 KB)

[VirtualAdvisor.mqh](https://www.mql5.com/en/articles/download/14574/virtualadvisor.mqh "Download VirtualAdvisor.mqh")(14.64 KB)

[VirtualOrder.mqh](https://www.mql5.com/en/articles/download/14574/virtualorder.mqh "Download VirtualOrder.mqh")(39.52 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Moving to MQL5 Algo Forge (Part 4): Working with Versions and Releases](https://www.mql5.com/en/articles/19623)
- [Moving to MQL5 Algo Forge (Part 3): Using External Repositories in Your Own Projects](https://www.mql5.com/en/articles/19436)
- [Moving to MQL5 Algo Forge (Part 2): Working with Multiple Repositories](https://www.mql5.com/en/articles/17698)
- [Moving to MQL5 Algo Forge (Part 1): Creating the Main Repository](https://www.mql5.com/en/articles/17646)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (II)](https://www.mql5.com/en/articles/17328)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (I)](https://www.mql5.com/en/articles/17277)
- [Developing a multi-currency Expert Advisor (Part 23): Putting in order the conveyor of automatic project optimization stages (II)](https://www.mql5.com/en/articles/16913)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/472248)**
(2)


![filippa.barbosa](https://c.mql5.com/avatar/avatar_na2.png)

**[filippa.barbosa](https://www.mql5.com/en/users/filippa.barbosa)**
\|
23 Feb 2025 at 12:52

Hello Yuriy,

thanks for the article. I have attached the csv just inquiring if I am on the right track. are those the right parameters

![Yuriy Bykov](https://c.mql5.com/avatar/avatar_na2.png)

**[Yuriy Bykov](https://www.mql5.com/en/users/antekov)**
\|
25 Feb 2025 at 06:03

HelloFilippa.

Thank you for your feedback.

Judging by the attached file, you are on the right track. This is what it should look like

![Building A Candlestick Trend Constraint Model (Part 8): Expert Advisor Development (II)](https://c.mql5.com/2/91/Building_A_Candlestick_Trend_Constraint_Model_Part_8__LOGO.png)[Building A Candlestick Trend Constraint Model (Part 8): Expert Advisor Development (II)](https://www.mql5.com/en/articles/15322)

Think about an independent Expert Advisor. Previously, we discussed an indicator-based Expert Advisor that also partnered with an independent script for drawing risk and reward geometry. Today, we will discuss the architecture of an MQL5 Expert Advisor, that integrates, all the features in one program.

![Example of Causality Network Analysis (CNA) and Vector Auto-Regression Model for Market Event Prediction](https://c.mql5.com/2/91/Vector_Auto-Regression_Model_for_Market_Event_Prediction___LOGO.png)[Example of Causality Network Analysis (CNA) and Vector Auto-Regression Model for Market Event Prediction](https://www.mql5.com/en/articles/15665)

This article presents a comprehensive guide to implementing a sophisticated trading system using Causality Network Analysis (CNA) and Vector Autoregression (VAR) in MQL5. It covers the theoretical background of these methods, provides detailed explanations of key functions in the trading algorithm, and includes example code for implementation.

![Neural Networks Made Easy (Part 83): The "Conformer" Spatio-Temporal Continuous Attention Transformer Algorithm](https://c.mql5.com/2/74/Neural_networks_are_easy_0Part_83a___LOGO.png)[Neural Networks Made Easy (Part 83): The "Conformer" Spatio-Temporal Continuous Attention Transformer Algorithm](https://www.mql5.com/en/articles/14615)

This article introduces the Conformer algorithm originally developed for the purpose of weather forecasting, which in terms of variability and capriciousness can be compared to financial markets. Conformer is a complex method. It combines the advantages of attention models and ordinary differential equations.

![Reimagining Classic Strategies (Part VII) : Forex Markets And Sovereign Debt Analysis on the USDJPY](https://c.mql5.com/2/91/Reimagining_Classic_Strategies_Part_VII___LOGO.png)[Reimagining Classic Strategies (Part VII) : Forex Markets And Sovereign Debt Analysis on the USDJPY](https://www.mql5.com/en/articles/15719)

In today's article, we will analyze the relationship between future exchange rates and government bonds. Bonds are among the most popular forms of fixed income securities and will be the focus of our discussion.Join us as we explore whether we can improve a classic strategy using AI.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=fffcgjgouyudvjiufhcprclfudhivows&ssn=1769092628846637012&ssn_dr=0&ssn_sr=0&fv_date=1769092628&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14574&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20multi-currency%20Expert%20Advisor%20(Part%208)%3A%20Load%20testing%20and%20handling%20a%20new%20bar%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909262853870408&fz_uniq=5049255830898976788&sv=2552)

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