---
title: Developing a multi-currency Expert Advisor (Part 2): Transition to virtual positions of trading strategies
url: https://www.mql5.com/en/articles/14107
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:29:24.783394
---

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/14107&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049168230746007105)

MetaTrader 5 / Trading


### Introduction

In the previous [article](https://www.mql5.com/en/articles/14026), we started developing a multi-currency EA that works simultaneously with various trading strategies. At the first stage there were only two different strategies. They represented the implementation of the same trading idea, worked on the same trading instrument (symbol) and chart period (timeframe). They differed from each other only in the numerical values of the parameters.

We also determined the optimal size of open positions based on the desired maximum drawdown level (10% of the deposit). We did this for each strategy separately. When we combined the two strategies together, we had to reduce the size of the opened positions in order to maintain the given drawdown level. For two strategies, the decrease was small. But what if we want to combine tens or hundreds of strategy instances? It may well happen that we will have to reduce the position size to a value less than the minimum size of open positions, allowed by the broker, for some strategies. In this case, these strategies simply will not be able to participate in trading. How can we make them work?

To do this, we will take away from the strategies the right to independently open positions and place pending orders. Strategies will only have to conduct virtual trading, that is, remember at what levels positions of a certain size should be opened and report what volume should be opened now upon request. We will open real market positions only after surveying all strategies and calculating the total required volume, taking into account scaling to maintain a given drawdown.

We are now only interested in testing the suitability of this approach, and not the efficiency of its implementation. Therefore, within the framework of this article, we will try to develop at least some working implementation of this approach, which later will help us build a more beautiful one from an architectural point of view since we will already have knowledge on how to avoid mistakes.

Let's try to implement this.

### Revisiting previous accomplishments

We have developed the _CAdvisor_ EA class, which stores the array of trading strategy instances (more precisely, pointers to instances). This allows creating one instance of an Expert Advisor in the main program and add several instances of strategy classes to it. Since the array stores pointers to objects of the _CStrategy_ base class, it can store pointers to objects of any descendant classes inherited from _CStrategy_. In our case, we created one descendant class _CSimpleVolumesStrategy_, whose two objects were added to this array in the EA.

Let's agree on convenient names for ease of presentation:

- The EA is our final _mq5_ file, which, after compilation, provides an executable _ex5_ file, suitable for running in the tester and the terminal.
- The EA is the _CAdvisor_ class object declared in the program. We will use only one EA instance in one program.
- The strategy is an object of a child class inherited from the _CStrategy_ base class of strategies.

Let us also recall that a pointer to an object (strategy or any other class) is information about the location in memory of a previously created object (simplified). It allows us to avoid re-creating the same object in another memory location when passing it to functions, assigning new variables or array elements.

That is why in the Expert Advisor we store pointers to strategy objects in an array, so that when filling this array, copies of strategy objects are not created. Then, when accessing the elements of the strategy array, we will access the original strategy objects.

The EA work consisted of the following stages:

- The EA in the static memory area was created.
- When the program was initialized, two strategies were created in dynamic memory and pointers to them were stored in the EA.
- When the program was running, the EA successively called on each strategy to perform the necessary trading actions by calling the _CStrategy::Tick()_ method.
- When deinitializing the program, the EA deleted strategy objects from dynamic memory.

Before we start, let’s make some minor corrections to the EA and the EA class. In the EA, we will make so that the expert is created in the dynamic memory area.

```
CAdvisor     expert;          // EA object
CAdvisor     *expert;         // Pointer to the EA object

int OnInit() {
   expert = new CAdvisor();   // Create EA object

   // The rest of the code from OnInit() ...
}
```

In the EA class, we will create a destructor - a function automatically called when the EA object is deleted from memory. The destructor receives the operations of removing strategy objects of the _CAdvisor::Deinit()_ method from the dynamic memory. We do not need this method now. Let's delete it. We will also remove the class variable that stores the number of strategies _m\_strategiesCount_. We can use _ArraySize()_ where needed.

```
class CAdvisor {
protected:
   CStrategy         *m_strategies[];  // Array of trading strategies
public:
   ~CAdvisor();                        // Destructor

   // ...
};

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
void CAdvisor::~CAdvisor() {
// Delete all strategy objects
   for(int i = 0; i < ArraySize(m_strategies); i++) {
      delete m_strategies[i];
   }
}
```

In the _OnDeinit()_ function, replace the _CAdvisor::Deinit()_ method with removing the EA object.

```
void OnDeinit(const int reason) {
   expert.Deinit();
   delete expert;
}
```

### Roadmap

If trading strategies are no longer able to open market positions themselves, then

- we need objects that will store information about virtual positions of strategies;
- we need objects that will translate information about virtual positions into real market positions.

Objects for virtual positions should be an integral part of the strategy and there should be several of them. Therefore, let's call the first new class _CVirtualOrder_ and add the array of these objects to the _CStrategy_ strategy class. _CStrategy_ also gets a property that stores an indication of changes in open virtual positions and methods for obtaining and setting its value. This property actually determines, which of two states the strategy is currently in:

- no changes - the entire open virtual volume has been released to the market;
- there are changes - the virtual volume does not correspond to the market one, therefore it is necessary to adjust the volumes of real market positions.

For now, it seems that these two states will be sufficient, so we will limit ourselves to this model.

Since now someone else will be responsible for opening real positions, the _m\_magic_ magic number property can be removed from the strategy base class. In the future, we will further clean up the most basic class of strategies, but for now we will limit ourselves to only partial cleaning.

With that said, the base strategy class will now look like this.

```
#include "VirtualOrder.mqh"

//+------------------------------------------------------------------+
//| Base class of the trading strategy                               |
//+------------------------------------------------------------------+
class CStrategy {
protected:
   string            m_symbol;         // Symbol (trading instrument)
   ENUM_TIMEFRAMES   m_timeframe;      // Chart period (timeframe)
   double            m_fixedLot;       // Size of opened positions (fixed)

   CVirtualOrder     m_orders[];       // Array of virtual positions (orders)
   int               m_ordersTotal;    // Total number of open positions and orders
   double            m_volumeTotal;    // Total volume of open positions and orders

   bool              m_isChanged;      // Sign of changes in open virtual positions
   void              CountOrders();    // Calculate the number and volumes of open positions and orders

public:
   // Constructor
   CStrategy(string p_symbol = "",
             ENUM_TIMEFRAMES p_timeframe = PERIOD_CURRENT,
             double p_fixedLot = 0.01);

   virtual void      Tick();           // Main method - handling OnTick events
   virtual double    Volume();         // Total volume of virtual positions
   virtual string    Symbol();         // Strategy symbol (only one for a single strategy so far)
   virtual bool      IsChanged();      // Are there any changes in open virtual positions?
   virtual void      ResetChanges();   // Reset the sign of changes in open virtual positions
};
```

We can already implement the _Symbol()_, _IsChanged()_ and _ResetChanges()_ methods.

```
//+------------------------------------------------------------------+
//| Strategy symbol                                                  |
//+------------------------------------------------------------------+
string CStrategy::Symbol() {
   return m_symbol;
}

//+------------------------------------------------------------------+
//| Are there any changes to open virtual positions?                 |
//+------------------------------------------------------------------+
bool CStrategy::IsChanged() {
   return m_isChanged;
}

//+------------------------------------------------------------------+
//| Reset the flag for changes in virtual positions                  |
//+------------------------------------------------------------------+
void CStrategy::ResetChanges() {
   m_isChanged = false;
}
```

We will implement the remaining methods ( _Tick()_, _Volume()_ and _CountOrders()_) either in base class descendants or in the class itself later.

The second new class, whose objects will be involved in bringing virtual positions of strategies to the market, will be called _CReceiver_. To be able to work, this object should have access to all the EA strategies in order to be able to find which symbols and what volume should be used to open real positions. One such object will be sufficient for one EA. The _CReceiver_ object should have a magic number that will be set for opened market positions.

```
#include "Strategy.mqh"

//+------------------------------------------------------------------+
//| Base class for converting open volumes into market positions     |
//+------------------------------------------------------------------+
class CReceiver {
protected:
   CStrategy         *m_strategies[];  // Array of strategies
   ulong             m_magic;          // Magic

public:
   CReceiver(ulong p_magic = 0);                // Constructor
   virtual void      Add(CStrategy *strategy);  // Adding strategy
   virtual bool      Correct();                 // Adjustment of open volumes
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CReceiver::CReceiver(ulong p_magic) : m_magic(p_magic) {
   ArrayResize(m_strategies, 0, 128);
}

//+------------------------------------------------------------------+
//| Add strategy                                                     |
//+------------------------------------------------------------------+
void CReceiver::Add(CStrategy *strategy) {
   APPEND(m_strategies, strategy);
}

//+------------------------------------------------------------------+
//| Adjust open volumes                                              |
//+------------------------------------------------------------------+
bool CReceiver::Correct() {
   return true;
}
```

This base class does not contain an implementation of a specific volume adjustment mechanism. Therefore, we will be able to make different implementations of the adjustment in different descendants of this class. The class object can serve as a stub for those strategies that, for now, will themselves open market positions. We will need this to debug the adjustment mechanism: it will be necessary to compare which positions are opened by the EA, in which the strategies themselves conduct real trading, and which positions are opened by the EA, in which the strategies conduct only virtual trading.

Therefore, we will prepare two EAs, in which the strategies from the previous article themselves conduct real trading.

In the first EA, the strategy will be in one instance. In its parameters, we will be able to specify the parameters of this single strategy instance for its optimization.

The second EA will contain several instances of trading strategies with predefined parameters obtained as a result of the first EA optimization.

### EA for optimizing strategy parameters

Last time we optimized the strategy parameters using the strategy implementation not in the form of the _CStrategy_ class object. But now we already have a ready-made class _CSimpleVolumesStrategy_, so let's create a separate program in which the EA will contain a single instance of this strategy. We will call this class a little differently to emphasize that the strategy itself will open market positions: instead of _CSimpleVolumesStrategy_, we will use _CSimpleVolumesMarketStrategy_ and save it in the _SimpleVolumesMarketStrategy.mqh_ file of the current folder.

In the EA file, we will make the strategy parameters loaded from the EA input variables and add one instance of the strategy to the EA object. We will get an EA we can optimize the strategy parameters with.

```
#include "Advisor.mqh"
#include "SimpleVolumesMarketStrategy.mqh"
#include "VolumeReceiver.mqh"

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
input string      symbol_              = "EURGBP";    // Trading instrument (symbol)
input ENUM_TIMEFRAMES
timeframe_           = PERIOD_H1;   // Chart period

input group "===  Opening signal parameters"
input int         signalPeriod_        = 130;   // Number of candles for volume averaging
input double      signalDeviation_     = 0.9;   // Relative deviation from the average to open the first order
input double      signaAddlDeviation_  = 1.4;   // Relative deviation from the average for opening the second and subsequent orders

input group "===  Pending order parameters"
input int         openDistance_        = 0;     // Distance from price to pending order
input double      stopLevel_           = 2000;  // Stop Loss (in points)
input double      takeLevel_           = 475;   // Take Profit (in points)
input int         ordersExpiration_    = 6000;  // Pending order expiration time (in minutes)

input group "===  Money management parameters"
input int         maxCountOfOrders_    = 3;     // Maximum number of simultaneously open orders
input double      fixedLot_            = 0.01;  // Single order volume

input group "===  EA parameters"
input ulong       magic_              = 27181; // Magic

CAdvisor     *expert;         // Pointer to the EA object

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   expert = new CAdvisor();
   expert.Add(new CSimpleVolumesMarketStrategy(
                 magic_, symbol_, timeframe_,
                 fixedLot_,
                 signalPeriod_, signalDeviation_, signaAddlDeviation_,
                 openDistance_, stopLevel_, takeLevel_, ordersExpiration_,
                 maxCountOfOrders_)
             );       // Add one strategy instance

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   expert.Tick();
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   delete expert;
}
```

Let's save it in the current folder of the _SimpleVolumesMarketExpertSingle.mq5_ file.

Now let’s complicate the trading strategy a little to simplify the implementation of the task. It will be easier for us to transfer to virtual trading a strategy that uses market positions rather than pending orders. The current version of the strategy only works with pending orders. Let's add parameter value analysis to the _openDistance\__ strategy. If it is above zero, then the strategy will open the BUY\_STOP and SELL\_STOP pending orders. If it is below zero, then the strategy will open the BUY\_LIMIT and SELL\_LIMIT pending orders. If it is equal to zero, then market positions will be opened.

To do this, just make changes to the code of the _CSimpleVolumesMarketStrategy::OpenBuyOrder()_ and _CSimpleVolumesMarketStrategy::OpenSellOrder()_ methods.

```
void CSimpleVolumesMarketStrategy::OpenBuyOrder() {
// Previous code in the method ...

// Order volume
   double lot = m_fixedLot;

// Set a pending order
   bool res = false;
   if(openDistance_ > 0) {
      res = trade.BuyStop(lot, ...);
   } else if(openDistance_ < 0) {
      res = trade.BuyLimit(lot, ...);
   } else {
      res = trade.Buy(lot, ...);
   }

   if(!res) {
      Print("Error opening order");
   }
}
```

Another necessary change that will need to be made to the strategy is to move the initialization code from the _Init()_ method to the strategy constructor. This is necessary because now the EA will not call the strategy initialization method, assuming that its code is located inside the strategy constructor.

Let's compile a new EA and set it to optimize on H1 using three symbols: EURGBP, GBPUSD and EURUSD.

![Fig. 1. Test results with the [EURGBP, H1, 17, 1.7, 0.5, 0, 16500, 100, 52000, 3, 0.01] parameters](https://c.mql5.com/2/79/2024-02-01_17-10-04.png)

Fig. 1. Test results with the \[EURGBP, H1, 17, 1.7, 0.5, 0, 16500, 100, 52000, 3, 0.01\] parameters

Let's select several good options for parameters from the optimization results (for example, three options for each symbol) and create a second EA, in which nine instances of the strategy with the selected parameters will be created. For each instance, we will calculate the optimal size of open positions, at which the drawdown of one strategy does not exceed 10%. The calculation method was described in the previous article.

To demonstrate changes in the EA performance, we will make it possible to set the strategies to be included. To do this, we will first place all strategy instances in the array of nine elements. Let's add the _startIndex\__ input setting the initial index in the strategy array the strategies start working from. The _totalStrategies\__ parameter determines how many sequential strategies from the array are to be launched starting from _startIndex\__. At the end of initialization, add the corresponding strategies from the array to the EA object.

```
#include "Advisor.mqh"
#include "SimpleVolumesMarketStrategy.mqh"

input int startIndex_      = 0;        // Starting index
input int totalStrategies_ = 1;        // Number of strategies
input double depoPart_     = 1.0;      // Part of the deposit for one strategy
input ulong  magic_        = 27182;    // Magic

CAdvisor     *expert;                  // EA object

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
// Check if the parameters are correct
   if(startIndex_ < 0 || startIndex_ + totalStrategies_ > 9) {
      return INIT_PARAMETERS_INCORRECT;
   }

// Create and fill the array of strategy instances
   CStrategy *strategies[9];
   strategies[0] = new CSimpleVolumesMarketStrategy(
      magic_ + 0, "EURGBP", PERIOD_H1,
      NormalizeDouble(0.01 / 0.16 * depoPart_, 2),
      13, 0.3, 1.0, 0, 10500, 465, 1000, 3);
   strategies[1] = new CSimpleVolumesMarketStrategy(
      magic_ + 1, "EURGBP", PERIOD_H1,
      NormalizeDouble(0.01 / 0.09 * depoPart_, 2),
      17, 1.7, 0.5, 0, 16500, 220, 1000, 3);
   strategies[2] = new CSimpleVolumesMarketStrategy(
      magic_ + 2, "EURGBP", PERIOD_H1,
      NormalizeDouble(0.01 / 0.16 * depoPart_, 2),
      51, 0.5, 1.1, 0, 19500, 370, 22000, 3);
   strategies[3] = new CSimpleVolumesMarketStrategy(
      magic_ + 3, "GBPUSD", PERIOD_H1,
      NormalizeDouble(0.01 / 0.25 * depoPart_, 2),
      80, 1.1, 0.2, 0, 6000, 1190, 1000, 3);
   strategies[4] = new CSimpleVolumesMarketStrategy(
      magic_ + 4, "GBPUSD", PERIOD_H1,
      NormalizeDouble(0.01 / 0.09 * depoPart_, 2),
      128, 2.0, 0.9, 0, 2000, 1170, 1000, 3);
   strategies[5] = new CSimpleVolumesMarketStrategy(
      magic_ + 5, "GBPUSD", PERIOD_H1,
      NormalizeDouble(0.01 / 0.14 * depoPart_, 2),
      13, 1.5, 0.8, 0, 2500, 1375, 1000, 3);
   strategies[6] = new CSimpleVolumesMarketStrategy(
      magic_ + 6, "EURUSD", PERIOD_H1,
      NormalizeDouble(0.01 / 0.23 * depoPart_, 2),
      24, 0.1, 0.3, 0, 7500, 2400, 24000, 3);
   strategies[7] = new CSimpleVolumesMarketStrategy(
      magic_ + 7, "EURUSD", PERIOD_H1,
      NormalizeDouble(0.01 / 0.20 * depoPart_, 2),
      18, 0.2, 0.4, 0, 19500, 1480, 6000, 3);
   strategies[8] = new CSimpleVolumesMarketStrategy(
      magic_ + 8, "EURUSD", PERIOD_H1,
      NormalizeDouble(0.01 / 0.22 * depoPart_, 2),
      128, 0.7, 0.3, 0, 3000, 170, 42000, 3);

   expert = new CAdvisor();

// Add the necessary strategies to the EA
   for(int i = startIndex_; i < startIndex_ + totalStrategies_; i++) {
      expert.Add(strategies[i]);
   }

   return(INIT_SUCCEEDED);
}

void OnTick() {
   expert.Tick();
}

void OnDeinit(const int reason) {
   delete expert;
}
```

Thanks to this, we can use optimization on the initial index of strategies in the strategy array to obtain results for each strategy instance. Let's launch it on the initial deposit of USD 100,000 and get the following results.

![](https://c.mql5.com/2/68/6222111429528.png)

Fig. 2. Results of single runs of nine strategy instances

It is clear that the drawdown is about 1% of the initial deposit, that is, approximately USD 1000, as we planned when selecting the optimal size of the positions to be opened. The average Sharpe ratio is 1.3.

Now let’s turn on all instances and select the appropriate _depoPart\__ multiplier to maintain a drawdown of USD 1000. If _depoPart\__ = 0.38, the drawdown remains within acceptable limits.

![](https://c.mql5.com/2/68/6161218252728.png)

![](https://c.mql5.com/2/68/1258859447742.png)

Fig. 3. Results of testing the simultaneous operation of nine strategy instances

Comparing the results of the work of single copies of strategies and the results of the simultaneous work of all copies, we can see that with the same drawdown, we received an increase in profit by approximately 3 times, as well as an increase in the Sharpe ratio from 1.3 to 2.84.

Now let's focus on the main task.

### Class of virtual positions (orders)

So, let's create the promised _CVirtualOrder_ class and add fields to it to store all the properties of open positions.

```
class CVirtualOrder {
private:
//--- Order (position) properties
   ulong             m_id;          // Unique ID

   string            m_symbol;      // Symbol
   double            m_lot;         // Volume
   ENUM_ORDER_TYPE   m_type;        // Type
   double            m_openPrice;   // Open price
   double            m_stopLoss;    // StopLoss level
   double            m_takeProfit;  // TakeProfit level
   string            m_comment;     // Comment

   datetime          m_openTime;    // Open time

//--- Closed order (position) properties
   double            m_closePrice;  // Close price
   datetime          m_closeTime;   // Close time
   string            m_closeReason; // Closure reason

   double            m_point;       // Point value

   bool              m_isStopLoss;  // StopLoss activation property
   bool              m_isTakeProfit;// TakeProfit activation property
};
```

Each virtual position should have a unique ID. Therefore, let's add the _s\_count_ class static variable to count the number of all position objects created in the program. When a new position object is created, this counter is incremented by 1 and this value becomes a unique position number. Set the _s\_count_ initial value equal to 0.

We will also need the _CSymbolInfo_ class object for pricing information. Let's make it a static member of the class as well.

```
class CVirtualOrder {
private:
   static int        s_count;
   static
   CSymbolInfo       s_symbolInfo;

//--- Order (position) properties ...

};

int               CVirtualOrder::s_count = 0;
CSymbolInfo       CVirtualOrder::s_symbolInfo;
```

It is worth noting that creating a virtual position object and "opening" a virtual position will be different operations. The position object can be created in advance and wait for the moment when the strategy wants to open a virtual position. At this moment, the position properties will be filled with the current values of symbol, volume, opening price, and others. When the strategy decides to close a position, the object will store the values of the closing properties: price, time and closure reason. During the next operation of opening a virtual position, we can use the same instance of the object, clearing its closing properties and filling it again with new values of symbol, volume, opening price, and others.

Let's add methods to this class. We will need public methods for opening and closing a virtual position and a constructor. The methods that check the position status (is it open and in what direction?) and its most important properties - volume and current profit - are useful as well.

```
class CVirtualOrder {
//--- Previous code...

public:
                     CVirtualOrder();  // Constructor

//--- Methods for checking the order (position) status
   bool              IsOpen();         // Is the order open?
   bool              IsMarketOrder();  // Is this a market position?
   bool              IsBuyOrder();     // Is this an open BUY position?
   bool              IsSellOrder();    // Is this an open SELL position?

//--- Methods for obtaining order (position) properties
   double            Volume();         // Volume with direction
   double            Profit();         // Current profit

//--- Methods for handling orders (positions)
   bool              Open(string symbol,
                          ENUM_ORDER_TYPE type,
                          double lot,
                          double sl = 0,
                          double tp = 0,
                          string comment = "",
                          bool inPoints = true);   // Opening an order (position)
   bool              Close();                      // Closing an order (position)
};
```

The implementation of some of these methods is very simple and short, so it can be placed right inside the class declaration, for example:

```
class CVirtualOrder : public CObject {
// ...

//--- Methods for checking the order (position) status
   bool              IsOpen() {        // Is the order open?
      return(this.m_openTime > 0 && this.m_closeTime == 0);
   };
   bool              IsMarketOrder() { // Is this a market position?
      return IsOpen() && (m_type == ORDER_TYPE_BUY || m_type == ORDER_TYPE_SELL);
   }

// ...
};
```

The constructor will assign empty (in the sense of obviously invalid) values to all properties of the virtual position with the exception of one - a unique ID - through the initialization list. As already mentioned, in the constructor, the ID will be assigned a value obtained from the value of the counter for previously created class objects. This value will remain intact throughout the entire EA operation. Before assigning, we will increment the counter of created objects.

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CVirtualOrder::CVirtualOrder() :
// Initialization list
   m_id(++s_count),  // New ID = object counter + 1
   m_symbol(""),
   m_lot(0),
   m_type(-1),
   m_openPrice(0),
   m_stopLoss(0),
   m_takeProfit(0),
   m_openTime(0),
   m_comment(""),
   m_closePrice(0),
   m_closeTime(0),
   m_closeReason(""),
   m_point(0) {
}
```

Before further implementing the _CVirtualOrder_ class methods, let's look ahead a little and think about how we will use the class. We have strategy objects that now open real market positions. Moreover, we know the maximum number of open market positions (set in the strategy parameters). We want to move to virtual positions. Then their number will also be limited. This means that we can create an array of virtual position objects in a strategy, fill it with the required number of virtual positions when initializing the strategy, and then work only with this array.

When conditions arise for opening a new position, we will take a virtual position that has not yet been opened from the array and turn it into an open one. If conditions for forced closing of positions arise, convert them into closed ones.

As long as there are open virtual positions, any strategy must process these objects in the same way at each tick: going through everything in turn, check whether the StopLoss or TakeProfit level has been reached, and, if yes, close the position. This sameness allows us to transfer the implementation of handling open virtual behavior positions to their own class, and only call the corresponding method from the strategy.

The _CVirtualOrder_ class receives the _Tick()_ method, which will check the conditions for closing a position, and if they are met, the position will be transferred to the closed state. If there is a change in the state of the position, the method will return true.

Let's also add the _Tick()_ static method, which handles several virtual position objects at once. It will accept a link to the array of such objects as a parameter. The _Tick()_ method will be called for each array object. If at least one virtual position is closed, then 'true' is returned eventually.

```
class CVirtualOrder {
private:
   //...
public:
   //...

//--- Methods for handling orders (positions)
   bool              Open(string symbol,
                          ENUM_ORDER_TYPE type,
                          double lot,
                          double sl = 0,
                          double tp = 0,
                          string comment = "",
                          bool inPoints = false
                         );      // Open order (position)

   bool              Tick();     // Handle tick for an order (position)
   bool              Close();    // Close an order (position)

   static bool       Tick(CVirtualOrder &orders[]);   // Handle a tick for the array of virtual orders
};

//...

//+------------------------------------------------------------------+
//| Handle a tick of a single virtual order (position)               |
//+------------------------------------------------------------------+
bool CVirtualOrder::Tick() {
   if(IsMarketOrder()) {  // If this is a market virtual position
      if(CheckClose()) {  // Check if SL or TP levels have been reached
         Close();         // Close when reached
         return true;     // Return the fact that there are changes in open positions
      }
   }

   return false;
}

//+------------------------------------------------------------------+
//| Handle a tick for the array of virtual orders (positions)        |
//+------------------------------------------------------------------+
bool CVirtualOrder::Tick(CVirtualOrder &orders[]) {
   bool isChanged = false;                      // We assume that there will be no changes
   for(int i = 0; i < ArraySize(orders); i++) { // For all orders (positions)
      isChanged |= orders[i].Tick();            // Check and close if necessary
   }
   return isChanged;
}
//+------------------------------------------------------------------+
```

Let's save this code in the _VirtualOrder.mqh_ file of the current folder.

### Improving the simple trading strategy class

Now we can return to the trading strategy class and make changes to it that allow us to work with virtual positions. As we have already agreed, in the base _CStrategy_ class, we already have the _m\_orders\[\]_ array for storing virtual position objects. Therefore, it is also available in the _CSimpleVolumesStrategy_ class. The strategy in question has the _m\_maxCountOfOrders_ parameter, which determines the maximum number of simultaneously open positions. Then set the size of the array of virtual positions in the constructor equal to this parameter.

Next, we just need to replace the opening of real positions in the _OpenBuyOrder()_ and _OpenSellOrder()_ methods with opening virtual ones. There is currently nothing we can replace the opening of real pending orders with, so we will just comment out these operations.

```
//+------------------------------------------------------------------+
//| Open BUY order                                                   |
//+------------------------------------------------------------------+
void CSimpleVolumesStrategy::OpenBuyOrder() {
// ...

   if(m_openDistance > 0) {
      /* // Set BUY STOP pending order
         res = trade.BuyStop(lot, ...);  */
   } else if(m_openDistance < 0) {
      /* // Set BUY LIMIT pending order
         res = trade.BuyLimit(lot, ...); */
   } else {
      // Open a virtual BUY position
      for(int i = 0; i < m_maxCountOfOrders; i++) {   // Iterate through all virtual positions
         if(!m_orders[i].IsOpen()) {                  // If we find one that is not open, then open it
            res = m_orders[i].Open(m_symbol, ORDER_TYPE_BUY, m_fixedLot,
                                   NormalizeDouble(sl, digits),
                                   NormalizeDouble(tp, digits));
            break;                                    // and exit
         }
      }
   }

   ...
}

//+------------------------------------------------------------------+
//| Open SELL order                                                  |
//+------------------------------------------------------------------+
void CSimpleVolumesStrategy::OpenSellOrder() {
// ...

   if(m_openDistance > 0) {
      /* // Set SELL STOP pending order
      res = trade.SellStop(lot, ...);          */
   } else if(m_openDistance < 0) {
      /* // Set SELL LIMIT pending order
      res = trade.SellLimit(lot, ...);         */
   } else {
      // Open a virtual SELL position
      for(int i = 0; i < m_maxCountOfOrders; i++) {   // Iterate through all virtual positions
         if(!m_orders[i].IsOpen()) {                  // If we find one that is not open, then open it
            res = m_orders[i].Open(m_symbol, ORDER_TYPE_SELL, m_fixedLot,
                                   NormalizeDouble(sl, digits),
                                   NormalizeDouble(tp, digits));
            break;                                    // and exit
         }
      }
   }

   ...
}
```

Save the changes in the _SimpleVolumesStrategy.mqh_ file of the current folder.

### Creating a class for converting open volumes into market positions

We have already created a base class for objects for converting open volumes into market positions, which so far does nothing except populate the array of used strategies. Now we need to write a derived class containing a specific implementation of placing positions on the market. Let's create the _CVolumeReceiver_ class. We will need to add quite a lot of code to it to implement the _Correct()_ method. We will break it down into several protected class methods.

```
#include "Receiver.mqh"

//+------------------------------------------------------------------+
//| Class for converting open volumes into market positions          |
//+------------------------------------------------------------------+
class CVolumeReceiver : public CReceiver {
protected:
   bool              m_isNetting;      // Is this a netting account?
   string            m_symbols[];      // Array of used symbols

   double            m_minMargin;      // Minimum margin for opening

   CPositionInfo     m_position;
   CSymbolInfo       m_symbolInfo;
   CTrade            m_trade;

   // Filling the array of open market volumes by symbols
   void              FillSymbolVolumes(double &oldVolumes[]);

   // Correction of open volumes using the array of volumes
   virtual bool      Correct(double &symbolVolumes[]);

   // Volume correction for this symbol
   bool              CorrectPosition(string symbol, double oldVolume, double diffVolume);

   // Auxiliary methods
   bool              ClearOpen(string symbol, double diffVolume);
   bool              AddBuy(string symbol, double volume);
   bool              AddSell(string symbol, double volume);

   bool              CloseBuyPartial(string symbol, double volume);
   bool              CloseSellPartial(string symbol, double volume);
   bool              CloseHedgingPartial(string symbol, double volume, ENUM_POSITION_TYPE type);
   bool              CloseFull(string symbol = "");

   bool              FreeMarginCheck(string symbol, double volume, ENUM_ORDER_TYPE type);

public:
   CVolumeReceiver(ulong p_magic, double p_minMargin = 100);   // Constructor
   virtual void      Add(CStrategy *strategy) override;        // Add strategy
   virtual bool      Correct() override;                       // Adjustment of open volumes
};
```

The general algorithm for the open volume correction method is as follows:

- For each symbol used, go through all the strategies and calculate the total open volume for each symbol used. The resulting _newVolumes_ array is passed to the next overloaded _Correct()_ method



```
//+------------------------------------------------------------------+
//| Adjustment of open volumes                                       |
//+------------------------------------------------------------------+
bool CVolumeReceiver::Correct() {
     int symbolsTotal = ArraySize(m_symbols);
     double newVolumes[];

     ArrayResize(newVolumes, symbolsTotal);
     ArrayInitialize(newVolumes, 0);

     for(int j = 0; j < symbolsTotal; j++) {  // For each used symbol
        for(int i = 0; i < ArraySize(m_strategies); i++) { // Iterate through all strategies
           if(m_strategies[i].Symbol() == m_symbols[j]) {  // If the strategy uses this symbol
              newVolumes[j] += m_strategies[i].Volume();   // Add its open volume
           }
        }
     }
     // Call correction of open volumes using the array of volumes
     return Correct(newVolumes);
}
```

- For each symbol, define how much the volume of open positions for the symbol should be changed. If necessary, call the volume correction method for this symbol



```
//+------------------------------------------------------------------+
//| Adjusting open volumes using the array of volumes                |
//+------------------------------------------------------------------+
bool CVolumeReceiver::Correct(double &newVolumes[]) {
     // ...
     bool res = true;

     // For each symbol
     for(int j = 0; j < ArraySize(m_symbols); j++) {
        // ...
        // Define how much the volume of open positions for the symbol should be changed
        double oldVolume = oldVolumes[j];
        double newVolume = newVolumes[j];

        // ...
        double diffVolume = newVolume - oldVolume;

        // If there is a need to adjust the volume for a given symbol, then do that
        if(MathAbs(diffVolume) > 0.001) {
           res = res && CorrectPosition(m_symbols[j], oldVolume, diffVolume);
        }
     }

     return res;
}
```

- For one symbol, determine what type of trading operation we need to perform (add, close and re-open), based on the values of the previous open volume and the required change, and call the corresponding auxiliary method:



```
//+------------------------------------------------------------------+
//| Adjust volume by the symbol                                      |
//+------------------------------------------------------------------+
bool CVolumeReceiver::CorrectPosition(string symbol, double oldVolume, double diffVolume) {
     bool res = false;

     // ...

     double volume = MathAbs(diffVolume);

     if(oldVolume > 0) { // Have BUY position
        if(diffVolume > 0) { // New BUY position
           res = AddBuy(symbol, volume);
        } else if(diffVolume < 0) { // New SELL position
           if(volume < oldVolume) {
              res = CloseBuyPartial(symbol, volume);
           } else {
              res = CloseFull(symbol);

              if(res && volume > oldVolume) {
                 res = AddSell(symbol, volume - oldVolume);
              }
           }
        }
     } else if(oldVolume < 0) { // Have SELL position
        if(diffVolume < 0) { // New SELL position
           res = AddSell(symbol, volume);
        } else if(diffVolume > 0) { // New BUY position
           if(volume < -oldVolume) {
              res = CloseSellPartial(symbol, volume);
           } else {
              res = CloseFull(symbol);

              if(res && volume > -oldVolume) {
                 res = AddBuy(symbol, volume + oldVolume);
              }
           }
        }
     } else { // No old position
        res = ClearOpen(symbol, diffVolume);
     }

     return res;
}
```


Save the code in the _VolumeReceiver.mqh_ file of the current folder.

### EA with one strategy and virtual positions

Create an EA that will use one instance of a trading strategy with virtual positions, based on the _SimpleVolumesMarketExpertSingle.mq5_ file. We will need to connect the necessary files, when calling the EA constructor, pass it the new _CVolumeReceiver_ class object and replace the class of the created strategy.

```
#include "Advisor.mqh"
#include "SimpleVolumesStrategy.mqh"
#include "VolumeReceiver.mqh"

// Input parameters...

CAdvisor     *expert;         // Pointer to the EA object

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   expert = new CAdvisor(new CVolumeReceiver(magic_));
   expert.Add(new CSimpleVolumesStrategy(
                         symbol_, timeframe_,
                         fixedLot_,
                         signalPeriod_, signalDeviation_, signaAddlDeviation_,
                         openDistance_, stopLevel_, takeLevel_, ordersExpiration_,
                         maxCountOfOrders_)
                     );       // Add one strategy instance

   return(INIT_SUCCEEDED);
}

void OnTick() {
   expert.Tick();
}

void OnDeinit(const int reason) {
   delete expert;
}
```

Save this code in the _SimpleVolumesExpertSingle.mq5_ file of the current folder.

### Comparison real and virtual trading

Let's launch EAs with the same strategy over a short time interval, using the same strategy parameters, but different ways of opening real positions - direct and via virtual positions. Save these results into reports and look at the list of trades made by both EAs.

![](https://c.mql5.com/2/68/6105745752202.png)

Fig. 4. Trades made by two EA (with and without virtual positions)

To reduce the width, columns with the same values in all rows, such as symbol (always EURGBP), volume (always 0.01) and others, were removed from the tables. As we can see, first positions are opened at the same price and at the same points in time in both cases. In case we have an open position SELL (2018.03.02 15:46:47 sell in) and open the BUY one (2018.03.06 13:56:04 buy in), the EA working via virtual positions simply closes the previous SELL position (2018.03.06 13:56:04 buy out). The overall result improved from this, since the first EA continued to pay swaps for open positions in different directions, while it was not the case in the second one.

EA with multiple strategies and virtual positions

Let's perform similar manipulations with the EA from the _SimpleVolumesMarketExpert.mq5_ file. We will include the necessary files. When calling the EA constructor, we will provide it with the new _CVolumeReceiver_ class object and replace the class of created strategies. Save the result to the _SimpleVolumesExpert.mq5_ file and look at the results.

![](https://c.mql5.com/2/68/1639143499462.png)

![](https://c.mql5.com/2/68/5619320200823.png)

Fig. 5. Results of the EA work with nine strategy instances and virtual positions

While comparing these results with the results of a similar EA not using virtual positions, we can note an improvement in some indicators: profit has increased slightly and drawdown has decreased, the Sharpe ratio and profit factor have increased as well.

### Conclusion

We have taken another step towards achieving our goal. By making the transition to using virtual position strategies, we have increased the ability for a large number of trading strategies to work together without interfering with each other. This will also allow us to use a lower minimum deposit for trading compared to using each instance of the strategy to trade independently. Another nice bonus will be the opportunity to work on Netting accounts.

But there are still many further steps to be taken. For example, so far only strategies have been implemented that open market positions, but not pending orders. Issues regarding money management are left for the future as well. At the moment, we trade with a fixed volume and select the optimal position size manually. Strategies that should work on several symbols at once (those that cannot be divided into simpler single-symbol strategies) are not be able to use this operation structure as well.

Stay tuned for the updates.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14107](https://www.mql5.com/ru/articles/14107)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14107.zip "Download all attachments in the single ZIP archive")

[Advisor.mqh](https://www.mql5.com/en/articles/download/14107/advisor.mqh "Download Advisor.mqh")(6.03 KB)

[Receiver.mqh](https://www.mql5.com/en/articles/download/14107/receiver.mqh "Download Receiver.mqh")(4.05 KB)

[SimpleVolumesExpert.mq5](https://www.mql5.com/en/articles/download/14107/simplevolumesexpert.mq5 "Download SimpleVolumesExpert.mq5")(8.09 KB)

[SimpleVolumesExpertSingle.mq5](https://www.mql5.com/en/articles/download/14107/simplevolumesexpertsingle.mq5 "Download SimpleVolumesExpertSingle.mq5")(6.96 KB)

[SimpleVolumesMarketExpert.mq5](https://www.mql5.com/en/articles/download/14107/simplevolumesmarketexpert.mq5 "Download SimpleVolumesMarketExpert.mq5")(8.3 KB)

[SimpleVolumesMarketExpertSingle.mq5](https://www.mql5.com/en/articles/download/14107/simplevolumesmarketexpertsingle.mq5 "Download SimpleVolumesMarketExpertSingle.mq5")(7.28 KB)

[SimpleVolumesMarketStrategy.mqh](https://www.mql5.com/en/articles/download/14107/simplevolumesmarketstrategy.mqh "Download SimpleVolumesMarketStrategy.mqh")(27.64 KB)

[SimpleVolumesStrategy.mqh](https://www.mql5.com/en/articles/download/14107/simplevolumesstrategy.mqh "Download SimpleVolumesStrategy.mqh")(27.53 KB)

[Strategy.mqh](https://www.mql5.com/en/articles/download/14107/strategy.mqh "Download Strategy.mqh")(9.32 KB)

[VirtualOrder.mqh](https://www.mql5.com/en/articles/download/14107/virtualorder.mqh "Download VirtualOrder.mqh")(24.22 KB)

[VolumeReceiver.mqh](https://www.mql5.com/en/articles/download/14107/volumereceiver.mqh "Download VolumeReceiver.mqh")(34.14 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/468025)**
(57)


![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
14 Feb 2024 at 13:21

**fxsaber [#](https://www.mql5.com/ru/forum/462052/page6#comment_52283234):**

You're right. Did it this way. This is a global string-variable into which all input variables are automatically (and created). That is, no matter what objects are created, this variable is always input.

Just in case I remind you that string inputs are cut by 63 characters by the optimiser.

![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
14 Feb 2024 at 13:31

**Stanislav Korotky [#](https://www.mql5.com/ru/forum/462052/page6#comment_52284296):**

Just in case, I remind you that string inputs are cut by 63 characters by the optimiser.

Thank you. It is not an input, so the length is not limited.

```
string inInputsAll = NULL;
```

![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
14 Feb 2024 at 18:45

[Forum on trading, automated trading systems and testing trading strategies](https://www.mql5.com/ru/forum)

[Discussion of the article "Developing a multicurrency Expert Advisor (Part 2): Moving to virtual positions trading strategies"](https://www.mql5.com/ru/forum/462052/page6#comment_52283234)

[fxsaber](https://www.mql5.com/ru/users/fxsaber), 2024.02.14 11:36 AM

You are right. Did it this way.

```
expert.Add(new CSimpleVolumesStrategy(inInputsAll));
```

This is a global string-variable into which all input variables are automatically (and created). I.e. whatever objects are not created, this variable is always fed to the input.

Attached.

![gardee005](https://c.mql5.com/avatar/avatar_na2.png)

**[gardee005](https://www.mql5.com/en/users/gardee005)**
\|
9 Nov 2024 at 13:02

Hi,

im on a steep learning curve of OOP. This follow-on to the previous article has been very helpful.

Still working through it. Thank you.

![Yuriy Bykov](https://c.mql5.com/avatar/avatar_na2.png)

**[Yuriy Bykov](https://www.mql5.com/en/users/antekov)**
\|
10 Nov 2024 at 12:46

Hi, I can't see the screenshots. Can you attach them again?


![Reimagining Classic Strategies: Crude Oil](https://c.mql5.com/2/79/Reimagining_Classic_Strategies____Crude_Oil____LOGO___5.png)[Reimagining Classic Strategies: Crude Oil](https://www.mql5.com/en/articles/14855)

In this article, we revisit a classic crude oil trading strategy with the aim of enhancing it by leveraging supervised machine learning algorithms. We will construct a least-squares model to predict future Brent crude oil prices based on the spread between Brent and WTI crude oil prices. Our goal is to identify a leading indicator of future changes in Brent prices.

![MQL5 Wizard Techniques you should know (Part 21): Testing with Economic Calendar Data](https://c.mql5.com/2/79/MQL5_Wizard_Techniques_you_should_know_Part_21____LOGO.png)[MQL5 Wizard Techniques you should know (Part 21): Testing with Economic Calendar Data](https://www.mql5.com/en/articles/14993)

Economic Calendar Data is not available for testing with Expert Advisors within Strategy Tester, by default. We look at how Databases could help in providing a work around this limitation. So, for this article we explore how SQLite databases can be used to archive Economic Calendar news such that wizard assembled Expert Advisors can use this to generate trade signals.

![News Trading Made Easy (Part 2): Risk Management](https://c.mql5.com/2/79/News_Trading_Made_Easy_Part_2_____LOGO.png)[News Trading Made Easy (Part 2): Risk Management](https://www.mql5.com/en/articles/14912)

In this article, inheritance will be introduced into our previous and new code. A new database design will be implemented to provide efficiency. Additionally, a risk management class will be created to tackle volume calculations.

![Master MQL5 from beginner to pro (Part II): Basic data types and use of variable](https://c.mql5.com/2/64/Learning_MQL5_-_from_beginner_to_pro_xPart_IIv_LOGO.png)[Master MQL5 from beginner to pro (Part II): Basic data types and use of variable](https://www.mql5.com/en/articles/13749)

This is a continuation of the series for beginners. In this article, we'll look at how to create constants and variables, write dates, colors, and other useful data. We will learn how to create enumerations like days of the week or line styles (solid, dotted, etc.). Variables and expressions are the basis of programming. They are definitely present in 99% of programs, so understanding them is critical. Therefore, if you are new to programming, this article can be very useful for you. Required programming knowledge level: very basic, within the limits of my previous article (see the link at the beginning).

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/14107&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049168230746007105)

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