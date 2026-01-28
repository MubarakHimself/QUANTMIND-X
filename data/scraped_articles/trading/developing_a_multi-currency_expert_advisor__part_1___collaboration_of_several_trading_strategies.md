---
title: Developing a multi-currency Expert Advisor (Part 1): Collaboration of several trading strategies
url: https://www.mql5.com/en/articles/14026
categories: Trading, Integration, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:29:34.591328
---

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/14026&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049170163481290316)

MetaTrader 5 / Trading


During my working activities, I had to deal with various trading strategies. As a rule, EAs implement only one trading idea. The difficulties of ensuring stable collaboration of many EAs on one terminal usually force us to select only a small number of the best ones. But it is still a pity to throw away completely workable strategies for this reason. How can we make them work together?

### Defining the problem

We need to decide what we want and what we have.

We have (or almost have):

- some different trading strategies that work on different symbols and timeframes in the form of a ready-made EA code or just a formulated set of rules for performing trading operations
- starting deposit
- maximum permissible drawdown

We want:

- collaboration of all selected strategies on one account on several symbols and timeframes
- distribution of the starting deposit between everyone equally or in accordance with the specified ratios
- automatic calculation of the volumes of opened positions to comply with the maximum allowable drawdown
- correct handling of terminal restarts
- ability to launch in MetaTrader 5 and 4

We will use an object-oriented approach, MQL5 and a standard tester in MetaTrader 5.

The task at hand is quite large, so we will solve it step by step.

At the first stage, let's take a simple trading idea. Let's make a simple EA using it. Optimize it and select the two best sets of parameters. Create an EA that will contain two copies of the original simple EA and look at its results.

### From trading idea to strategy

Let’s take the following idea as an experimental one.

Suppose that when intensive trading starts for a certain symbol, the price can change more per unit of time than when trading for a symbol is sluggish. Then if we see that trading has intensified and the price has changed in some direction, then perhaps it will change in the same direction in the near future. Let's try to make a profit from this.

Trading strategy is a set of rules for opening and closing positions based on a trading idea. It does not contain any unknown parameters. This set of rules should allow us to determine for any moment in time that the strategy is running whether any positions should be opened and, if so, which ones.

Let's try to turn the idea into a strategy. First of all, we need to somehow detect an increase in trade intensity. Without this, we will not be able to determine when to open positions. For this, we will use tick volume, that is, the number of new prices that were received by the terminal during the current candle. A larger tick volume will be considered a sign of more intense trading. But for different symbols the intensity can differ significantly. Therefore, we cannot set a single level for a tick volume, the excess of which we will consider the beginning of intensive trading. Then, to determine this level, we can start from the average tick volume of several candles. After some thought, we can provide the following description:

Place a pending order at the moment when the tick volume of the candle exceeds the average volume in the direction of the current candle. Each order will have an expiration time, after which it will be deleted. If a pending order has turned into a position, it will be closed only upon reaching the specified StopLoss and TakeProfit levels. If the tick volume exceeds the average even more, then additional orders may be placed in addition to the already opened pending order.

This is a more detailed description, but not a complete one. Therefore, we read it again and highlight all the places where something is not clear. More detailed explanations are required there.

Here are the questions that arose:

- " _Place a pending order ..._" — What pending orders should we place?
- " _... average volume ..._"  — How to calculate the candle average volume?
- " _... exceeds the average volume ..._" — How to determine an excess of the average volume?
- " _... If the tick volume exceeds the average even more ..._" — How to determine this greater excess?

- "... additional orders may be placed" — How many orders can be placed in total?

What pending orders will we place? Based on the idea, we hope that the price will continue to move in the same direction in which it went from the start of the candle. For example, if the price is currently higher than at the beginning of the candle period, then we should open a pending buy order. If we open BUY\_LIMIT, then for it to work, the price should first return (drop) a little, and then for the opened position to make a profit, the price should rise again. If we open BUY\_STOP, then to open a position the price should continue to move a little more (rise), and then rise even higher to make a profit.

It is not immediately clear, which of these options is better. Therefore, for simplicity, let's always open stop orders (BUY\_STOP and SELL\_STOP). In the future, this can be made a strategy parameter with its value determining which orders will be opened.

How to calculate the average candle volume? To calculate the average volume, we need to select candles whose volumes will be included in the average calculation. Let's take a number of consecutive last closed candles. Then if we set the number of candles, we can calculate the average tick volume.

How to determine the excess of the average volume? If we take the condition

V > V\_avr ,

where

V is a tick volume of the current candle,

V\_avr is an average tick volume,

then the fulfillment of this condition will be achieved on approximately half of the candles. Based on the idea, we should place orders only when the volume significantly exceeds the average. Otherwise, this cannot yet be considered a sign of more intense trading on this candle, unlike previous candles. For example, we can use the following equation:

V > V\_avr + D \* V\_avr,

where D is a numerical ratio. If D = 1, then the opening occurs when the current volume exceeds the average by 2 times, and if, for example, D = 2, then the opening occurs when the current volume exceeds the average 3 times.

However, this condition can be applied to open only one order, since if it is used to open the second and subsequent ones, then they will open immediately after the first one. This can be replaced simply by opening one order of a larger volume.

How to determine the greater excess? To do this, let's add one more parameter to the condition equation - the number of open orders N:

V > V\_avr + D \* V\_avr + N \* D \* V\_avr.

Then, in order for the second order to open after the first one (that is, N = 1), the following condition should be met:

V > V\_avr + 2 \* D \* V\_avr.

To open the first order (N = 0), the equation takes on the form already known to us:

V > V\_avr + D \* V\_avr.

Finally, the last correction to the opening equation. Let’s make two independent parameters D and D\_add for the first and subsequent orders instead of the same D:

V > V\_avr + D \* V\_avr + N \* D\_add \* V\_avr,

V > V\_avr \* (1 + D + N \* D\_add)

It seems that this will give us greater freedom in selecting the optimal parameters for the strategy.

If our condition uses the N value as the total number of orders and positions, then we mean that each pending order turns into a separate position, and does not increase the volume of an already open position. Therefore, for now we will have to limit the scope of application of such a strategy only to work on accounts with independent accounting of positions ("hedging").

When everything is clear, let's list the variables that can take different values, not just one single value. These will be our strategy inputs. Let's take into account that to open orders we also need to know the volume, distance from the current price, expiration time and StopLoss and TakeProfit levels. Then we get the following description:

The EA runs on a specific symbol and period (timeframe) on the Hedge account

Set the input:

- Number of candles for volume averaging (K)
- Relative deviation from the average for opening the first order (D)
- Relative deviation from the average for opening the second and subsequent orders (D\_add)
- Distance from price to pending order
- Stop Loss (in points)
- Take Profit (in points)
- Expiration time of pending orders (in minutes)
- Maximum number of simultaneously open orders (N\_max)
- Single order volume

Find the number of open orders and positions (N).

If it is less than N\_max, then:

        calculate the average tick volume for the last K closed candles, get the V\_avr value.

         If the V > V\_avr \* (1 + D + N \* D\_add) condition is met, then:

                 determine the direction of price change on the current candle: if the price has increased, then we will place a BUY\_STOP pending order, otherwise - SELL\_STOP

                 place a pending order at the distance, expiration time, and StopLoss and TakeProfit levels specified in the parameters.

### Implementing a trading strategy

Let's start writing the code. First, let's list all the parameters dividing them into groups for clarity and providing each parameter with a comment. These comments (if any) will be displayed in the parameters dialog when launching the EA and in the parameters tab in the strategy tester instead of the variable names we have chosen for them.

For now, we just set some default values. We will look for the best ones during optimization.

```
input group "===  Opening signal parameters"
input int         signalPeriod_        = 48;    // Number of candles for volume averaging
input double      signalDeviation_     = 1.0;   // Relative deviation from the average to open the first order
input double      signaAddlDeviation_  = 1.0;   // Relative deviation from the average for opening the second and subsequent orders

input group "===  Pending order parameters"
input int         openDistance_        = 200;   // Distance from price to pending order
input double      stopLevel_           = 2000;  // Stop Loss (in points)
input double      takeLevel_           = 75;    // Take Profit (in points)
input int         ordersExpiration_    = 6000;  // Pending order expiration time (in minutes)

input group "===  Money management parameters"
input int         maxCountOfOrders_    = 3;     // Maximum number of simultaneously open orders
input double      fixedLot_            = 0.01;  // Single order volume

input group "===  EA parameters"
input ulong       magicN_              = 27181; // Magic
```

Since the EA will perform trading operations, we will create a global object of the CTrade class. We will place pending orders by calling the object methods.

```
CTrade            trade;            // Object for performing trading operations
```

Keep in mind that global variables (or objects) are variables (or objects) declared**outside**of a function in the EA code. Therefore, they are available in all our EA functions. They should not be confused with global terminal variables.

To calculate the parameters for opening orders, we will need to obtain current prices and other symbol properties the EA will be launched on. To do this, create a global object of the CSymbolInfo class.

```
CSymbolInfo       symbolInfo;       // Object for obtaining data on the symbol properties
```

Also we will need to count the number of open orders and positions. To achieve this, let's create global objects of the COrderInfo and CPositionInfo classes used to get data on open orders and positions. We will store the quantity itself in two global variables - countOrders and countPositions.

```
COrderInfo        orderInfo;        // Object for receiving information about placed orders
CPositionInfo     positionInfo;     // Object for receiving information about open positions

int               countOrders;      // Number of placed pending orders
int               countPositions;   // Number of open positions
```

To calculate the average tick volume of several candles, we can use, for example, the iVolumes technical indicator. To get its values, we need a variable to store the handle of this indicator (an integer that stores the serial number of this indicator out of all others to be used in the EA). To find the average volume, we will first have to copy the values from the indicator buffer into a preliminarily prepared array. We will also make this array global.

```
int               iVolumesHandle;   // Tick volume indicator handle
double            volumes[];        // Receiver array of indicator values (volumes themselves)
```

Now we can proceed to the OnInit() EA initialization function and the OnTick() tick processing function.

During initialization, we can do the following:

- Load the indicator to obtain tick volumes and remember its handle
- Set the size of the receiving array in accordance with the number of candles to calculate the average volume and set its addressing as in timeseries
- Set Magic Number for placing orders through the trade object

This is what our initialization function will look like:

```
int OnInit() {
   // Load the indicator to get tick volumes
   iVolumesHandle = iVolumes(Symbol(), PERIOD_CURRENT, VOLUME_TICK);

   // Set the size of the tick volume receiving array and the required addressing
   ArrayResize(volumes, signalPeriod_);
   ArraySetAsSeries(volumes, true);

   // Set Magic Number for placing orders via 'trade'
   trade.SetExpertMagicNumber(magicN_);

   return(INIT_SUCCEEDED);
}
```

According to the strategy description, we should start by finding the number of open orders and positions in the tick processing function. Let's implement this as a separate UpdateCounts() function. In this function, we will go through all open positions and orders, and count only those whose Magic Number matches the one of our EA.

```
void UpdateCounts() {
// Reset position and order counters
   countPositions = 0;
   countOrders = 0;

// Loop through all positions
   for(int i = 0; i < PositionsTotal(); i++) {
      // If the position with index i is selected successfully and its Magic is ours, then we count it
      if(positionInfo.SelectByIndex(i) && positionInfo.Magic() == magicN_) {
         countPositions++;
      }
   }

// Loop through all orders
   for(int i = 0; i < OrdersTotal(); i++) {
      // If the order with index i is selected successfully and its Magic is the one we need, then we consider it
      if(orderInfo.SelectByIndex(i) && orderInfo.Magic() == magicN_) {
         countOrders++;
      }
   }
}
```

Next, make sure the number of open positions and orders does not exceed the one specified in the settings. In this case, we need to check whether the conditions for opening a new order are satisfied. Let's implement this check as a separate SignalForOpen() function. It will return one of three possible values:

- +1  — signal to open the BUY\_STOP order
-  0  — no signal
- -1  — signal to open the SELL\_STOP order

To place pending orders, we will also write two separate functions: OpenBuyOrder() and OpenSellOrder().

Now we can write a complete implementation of the OnTick() function.

```
void OnTick() {
// Count open positions and orders
   UpdateCounts();

// If their number is less than allowed
   if(countOrders + countPositions < maxCountOfOrders_) {
      // Get an open signal
      int signal = SignalForOpen();

      if(signal == 1) {          // If there is a buy signal, then
         OpenBuyOrder();         // open the BUY_STOP order
      } else if(signal == -1) {  // If there is a sell signal, then
         OpenSellOrder();        // open the SELL_STOP order
      }
   }
}
```

After this, we add the implementation of the remaining functions and the EA code is ready. Let's save it in the SimpleVolumes.mq5 file in the current folder.

```
#include <Trade\OrderInfo.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\Trade.mqh>

input group "===  Opening signal parameters"
input int         signalPeriod_        = 48;    // Number of candles for volume averaging
input double      signalDeviation_     = 1.0;   // Relative deviation from the average to open the first order
input double      signaAddlDeviation_  = 1.0;   // Relative deviation from the average for opening the second and subsequent orders

input group "===  Pending order parameters"
input int         openDistance_        = 200;   // Distance from price to pending order
input double      stopLevel_           = 2000;  // Stop Loss (in points)
input double      takeLevel_           = 75;    // Take Profit (in points)
input int         ordersExpiration_    = 6000;  // Pending order expiration time (in minutes)

input group "===  Money management parameters"
input int         maxCountOfOrders_    = 3;     // Maximum number of simultaneously open orders
input double      fixedLot_            = 0.01;  // Single order volume

input group "===  EA parameters"
input ulong       magicN_              = 27181; // Magic

CTrade            trade;            // Object for performing trading operations

COrderInfo        orderInfo;        // Object for receiving information about placed orders
CPositionInfo     positionInfo;     // Object for receiving information about open positions

int               countOrders;      // Number of placed pending orders
int               countPositions;   // Number of open positions

CSymbolInfo       symbolInfo;       // Object for obtaining data on the symbol properties

int               iVolumesHandle;   // Tick volume indicator handle
double            volumes[];        // Receiver array of indicator values (volumes themselves)

//+------------------------------------------------------------------+
//| Initialization function of the expert                            |
//+------------------------------------------------------------------+
int OnInit() {
// Load the indicator to get tick volumes
   iVolumesHandle = iVolumes(Symbol(), PERIOD_CURRENT, VOLUME_TICK);

// Set the size of the tick volume receiving array and the required addressing
   ArrayResize(volumes, signalPeriod_);
   ArraySetAsSeries(volumes, true);

// Set Magic Number for placing orders via 'trade'
   trade.SetExpertMagicNumber(magicN_);

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| "Tick" event handler function                                    |
//+------------------------------------------------------------------+
void OnTick() {
// Count open positions and orders
   UpdateCounts();

// If their number is less than allowed
   if(countOrders + countPositions < maxCountOfOrders_) {
      // Get an open signal
      int signal = SignalForOpen();

      if(signal == 1) {          // If there is a buy signal, then
         OpenBuyOrder();         // open the BUY_STOP order
      } else if(signal == -1) {  // If there is a sell signal, then
         OpenSellOrder();        // open the SELL_STOP order
      }
   }
}

//+------------------------------------------------------------------+
//| Calculate the number of open orders and positions                |
//+------------------------------------------------------------------+
void UpdateCounts() {
// Reset position and order counters
   countPositions = 0;
   countOrders = 0;

// Loop through all positions
   for(int i = 0; i < PositionsTotal(); i++) {
      // If the position with index i is selected successfully and its Magic is ours, then we count it
      if(positionInfo.SelectByIndex(i) && positionInfo.Magic() == magicN_) {
         countPositions++;
      }
   }

// Loop through all orders
   for(int i = 0; i < OrdersTotal(); i++) {
      // If the order with index i is selected successfully and its Magic is the one we need, then we consider it
      if(orderInfo.SelectByIndex(i) && orderInfo.Magic() == magicN_) {
         countOrders++;
      }
   }
}

//+------------------------------------------------------------------+
//| Open the BUY_STOP order                                          |
//+------------------------------------------------------------------+
void OpenBuyOrder() {
// Update symbol current price data
   symbolInfo.Name(Symbol());
   symbolInfo.RefreshRates();

// Retrieve the necessary symbol and price data
   double point = symbolInfo.Point();
   int digits = symbolInfo.Digits();
   double bid = symbolInfo.Bid();
   double ask = symbolInfo.Ask();
   int spread = symbolInfo.Spread();

// Let's make sure that the opening distance is not less than the spread
   int distance = MathMax(openDistance_, spread);

// Opening price
   double price = ask + distance * point;

// StopLoss and TakeProfit levels
   double sl = NormalizeDouble(price - stopLevel_ * point, digits);
   double tp = NormalizeDouble(price + (takeLevel_ + spread) * point, digits);

// Expiration time
   datetime expiration = TimeCurrent() + ordersExpiration_ * 60;

// Order volume
   double lot = fixedLot_;

// Set a pending order
   bool res = trade.BuyStop(lot,
                            NormalizeDouble(price, digits),
                            Symbol(),
                            NormalizeDouble(sl, digits),
                            NormalizeDouble(tp, digits),
                            ORDER_TIME_SPECIFIED,
                            expiration);

   if(!res) {
      Print("Error opening order");
   }
}

//+------------------------------------------------------------------+
//| Open the SELL_STOP order                                         |
//+------------------------------------------------------------------+
void OpenSellOrder() {
// Update symbol current price data
   symbolInfo.Name(Symbol());
   symbolInfo.RefreshRates();

// Retrieve the necessary symbol and price data
   double point = symbolInfo.Point();
   int digits = symbolInfo.Digits();
   double bid = symbolInfo.Bid();
   double ask = symbolInfo.Ask();
   int spread = symbolInfo.Spread();

// Let's make sure that the opening distance is not less than the spread
   int distance = MathMax(openDistance_, spread);

// Opening price
   double price = bid - distance * point;

// StopLoss and TakeProfit levels
   double sl = NormalizeDouble(price + stopLevel_ * point, digits);
   double tp = NormalizeDouble(price - (takeLevel_ + spread) * point, digits);

// Expiration time
   datetime expiration = TimeCurrent() + ordersExpiration_ * 60;

// Order volume
   double lot = fixedLot_;

// Set a pending order
   bool res = trade.SellStop(lot,
                             NormalizeDouble(price, digits),
                             Symbol(),
                             NormalizeDouble(sl, digits),
                             NormalizeDouble(tp, digits),
                             ORDER_TIME_SPECIFIED,
                             expiration);

   if(!res) {
      Print("Error opening order");
   }
}

//+------------------------------------------------------------------+
//| Signal for opening pending orders                                |
//+------------------------------------------------------------------+
int SignalForOpen() {
// By default, there is no signal
   int signal = 0;

// Copy volume values from the indicator buffer to the receiving array
   int res = CopyBuffer(iVolumesHandle, 0, 0, signalPeriod_, volumes);

// If the required amount of numbers have been copied
   if(res == signalPeriod_) {
      // Calculate their average value
      double avrVolume = ArrayAverage(volumes);

      // If the current volume exceeds the specified level, then
      if(volumes[0] > avrVolume * (1 + signalDeviation_ + (countOrders + countPositions) * signaAddlDeviation_)) {
         // if the opening price of the candle is less than the current (closing) price, then
         if(iOpen(Symbol(), PERIOD_CURRENT, 0) < iClose(Symbol(), PERIOD_CURRENT, 0)) {
            signal = 1; // buy signal
         } else {
            signal = -1; // otherwise, sell signal
         }
      }
   }

   return signal;
}

//+------------------------------------------------------------------+
//| Number array average value                                       |
//+------------------------------------------------------------------+
double ArrayAverage(const double &array[]) {
   double s = 0;
   int total = ArraySize(array);
   for(int i = 0; i < total; i++) {
      s += array[i];
   }

   return s / MathMax(1, total);
}
//+------------------------------------------------------------------+
```

Let's start optimizing the EA parameters for EURGBP H1 on MetaQuotes quotes from 2018-01-01 to 2023-01-01 with the starting deposit of USD 100 000 and the minimum lot of 0.01. Note that the same EA may show slightly different results when tested on quotes from different brokers. Sometimes, these results may differ greatly.

Let's select two nice sets of parameters with the following results:

![](https://c.mql5.com/2/78/3106789131207.png)

Fig. 1. Test results for \[130, 0.9, 1.4, 231, 3750, 50, 600, 3, 0.01\]

![](https://c.mql5.com/2/78/4932938644630.png)

Fig. 2. Test results for \[159, 1.7, 0.8, 248, 3600, 495, 39000, 3, 0.01\]

It was not by chance that the test was carried out on a large starting deposit. The reason is that if the EA opens positions having a fixed volume, then the run may end early if the drawdown becomes greater than the available funds. In this case, we will not know whether, it would have been possible to reasonably reduce the volume of open positions (or, equivalently, increase the starting deposit) in order to avoid a loss, while using the same parameters.

Let us review an example. Suppose that our starting deposit is USD 1,000. When running in the tester, we got the following results:

- The final deposit is USD 11,000 (profit 1,000%, the EA earned + USD 10,000 to the initial USD 1,000)
- The maximum absolute drawdown is USD 2,000

Obviously, we were just lucky that such a drawdown happened after the EA increased the deposit to more than USD 2,000. Therefore, the tester run completed and we were able to see these results. If such a drawdown had happened earlier (for example, we would have chosen a different start of the testing period), then we would have lost the entire deposit.

If we do runs manually, then we can change the volume in the parameters or increase the starting deposit and start the run again. But if runs are performed during the optimization, then this is not possible. In this case, a potentially good set of parameters may be rejected due to incorrectly selected money management settings. To reduce the likelihood of such outcomes, we can run optimization with an initially very large starting deposit and a minimum volume.

Getting back to the example, if the starting deposit was USD 100,000, then in case of repeating the drawdown of USD 2,000, the loss of the entire deposit would not occur and the tester would receive these results. And we could calculate that if the maximum allowable drawdown for us is 10%, then the initial deposit should be at least $20,000. In this case, the profitability will be only 50% (the EA earned + USD 10,000 to the initial USD 20,000)

Let's do similar calculations for our two selected combinations of parameters for the starting deposit of USD 10,000 and the allowable drawdown of 10% of the starting deposit.

| Parameters | Lot | Drawdown | Profit | Acceptable <br>drawdown | Acceptable <br>lot | Acceptable <br>gain |
| --- | --- | --- | --- | --- | --- | --- |
|  | L | D | P | Da | La = L \* (Da / D) | Pa =   P \\* (Da / D) |
| \[130, 0.9, 1.4, 231, <br>3750, 50, 600, 3, 0.01\] | 0.01 | 28.70 (0.04%) | 260.41 | 1000 (10%) | 0.34 | 9073 (91%) |
| \[159, 1.7, 0.8, 248,<br>3600, 495, 39000, 3, 0.01\] | 0.01 | 92.72 (0.09%) | 666.23 | 1000 (10%) | 0.10 | 7185 (72%) |

As we can see, both input options can yield approximately similar returns (~80%). The first option earns less in absolute terms, but with a smaller drawdown. Therefore, in this case, we can increase the volume of opened positions more than for the second option, which, although it earns more, allows for a larger drawdown.

So, we have found several promising input combinations. Let’s start combining them into one EA.

### Base strategy class

Let's create the CStrategy class, in which we will collect all the properties and methods inherent in all strategies. For example, any strategy will have some kind of symbol and timeframe, regardless of its relationship to indicators. We will also allocate each strategy its own Magic Number for opening positions and the size of one position. For simplicity, we will not consider the operation of the strategy with a variable position size for now. We will definitely do this later.

Among the necessary methods, only the constructor that initializes the strategy parameters, the initialization method, and the OnTick event handler can be identified. We get the following code:

```
class CStrategy : public CObject {
protected:
   ulong             m_magic;          // Magic
   string            m_symbol;         // Symbol (trading instrument)
   ENUM_TIMEFRAMES   m_timeframe;      // Chart period (timeframe)
   double            m_fixedLot;       // Size of opened positions (fixed)

public:
   // Constructor
   CStrategy(ulong p_magic,
             string p_symbol,
             ENUM_TIMEFRAMES p_timeframe,
             double p_fixedLot);

   virtual int       Init() = 0; // Strategy initialization - handling OnInit events
   virtual void      Tick() = 0; // Main method - handling OnTick events
};
```

The Init() and Tick() methods are declared purely virtual (the method header is followed by = 0). This means that we will not write the implementation of these methods in the CStrategy class. Based on this class, we will create descendant classes in which the Init() and Tick() methods should necessarily be present and contain the implementation of specific trading rules.

The class description is ready. After it, we will add the necessary implementation of the constructor. Since this is a method function that is automatically called when a strategy object is created, it is in this method that we need to ensure that the strategy parameters are initialized. The constructor will take four parameters and assign their values to the corresponding class member variables via the initialization list.

```
CStrategy::CStrategy(
   ulong p_magic,
   string p_symbol,
   ENUM_TIMEFRAMES p_timeframe,
   double p_fixedLot) :
// Initialization list
   m_magic(p_magic),
   m_symbol(p_symbol),
   m_timeframe(p_timeframe),
   m_fixedLot(p_fixedLot)
{}
```

Save this code in the Strategy.mqh file of the current folder.

### Trading strategy class

Let's transfer the logic of the original simple EA to a new descendant class CSimpleVolumesStrategy. To do this, make all input variables and global variables members of the class. We will replace the fixedLot\_ and magicN\_ variables with the m\_fixedLot and m\_magic base class members inherited from the CStrategy base class.

```
#include "Strategy.mqh"

class CSimpleVolumeStrategy : public CStrategy {
   //---  Open signal parameters
   int               signalPeriod_;       // Number of candles for volume averaging
   double            signalDeviation_;    // Relative deviation from the average to open the first order
   double            signaAddlDeviation_; // Relative deviation from the average for opening the second and subsequent orders

   //---  Pending order parameters
   int               openDistance_;       // Distance from price to pending order
   double            stopLevel_;          // Stop Loss (in points)
   double            takeLevel_;          // Take Profit (in points)
   int               ordersExpiration_;   // Pending order expiration time (in minutes)

   //---  Money management parameters
   int               maxCountOfOrders_;   // Maximum number of simultaneously open orders

   CTrade            trade;               // Object for performing trading operations

   COrderInfo        orderInfo;           // Object for receiving information about placed orders
   CPositionInfo     positionInfo;        // Object for receiving information about open positions

   int               countOrders;         // Number of placed pending orders
   int               countPositions;      // Number of open positions

   CSymbolInfo       symbolInfo;          // Object for obtaining data on the symbol properties

   int               iVolumesHandle;      // Tick volume indicator handle
   double            volumes[];           // Receiver array of indicator values (volumes themselves)
};
```

The OnInit() and OnTick() functions become the Init() and Tick() public methods, and all other functions become new private methods of the CSimpleVolumesStrategy class. Public methods can be called for strategies from an external code, for example from EA object methods. Private methods can only be called from methods of a given class. Let's add method headers to the class description.

```
class CSimpleVolumeStrategy : public CStrategy {
private:
   //---  ... previous code
   double            volumes[];           // Receiver array of indicator values (volumes themselves)

   //--- Methods
   void              UpdateCounts();      // Calculate the number of open orders and positions
   int               SignalForOpen();     // Signal for opening pending orders
   void              OpenBuyOrder();      // Open the BUY_STOP order
   void              OpenSellOrder();     // Open the SELL_STOP order
   double            ArrayAverage(
      const double &array[]);             // Average value of the number array

public:
   //--- Public methods
   virtual int       Init();              // Strategy initialization method
   virtual void      Tick();              // OnTick event handler
};
```

In those places where the implementation of these functions is located, add the "CSimpleVolumesStrategy::" prefix to their name to make it clear to the compiler that these are no longer just functions, but function methods of our class.

```
class CSimpleVolumeStrategy : public CStrategy {
   // Class description listing properties and methods...
};

int CSimpleVolumeStrategy::Init() {
// Function code ...
}

void CSimpleVolumeStrategy::Tick() {
// Function code ...
}

void CSimpleVolumeStrategy::UpdateCounts() {
// Function code ...
}

int CSimpleVolumeStrategy::SignalForOpen() {
// Function code ...
}

void CSimpleVolumeStrategy::OpenBuyOrder() {
// Function code ...
}

void CSimpleVolumeStrategy::OpenSellOrder() {
// Function code ...
}

double CSimpleVolumeStrategy::ArrayAverage(const double &array[]) {
// Function code ...
}
```

In the original simple EA, the input values were assigned when declared. When launching the compiled EA, the values from the input parameters dialog (not the ones set in the code) were assigned to them. This cannot be done in the class description, so this is where the constructor comes into play.

Let's create a constructor with the necessary list of parameters. The constructor should also be public, otherwise we will not be able to create strategy objects from an external code.

```
class CSimpleVolumeStrategy : public CStrategy {
private:
   //---  ... previous code

public:
   //--- Public methods
   CSimpleVolumeStrategy(
      ulong            p_magic,
      string           p_symbol,
      ENUM_TIMEFRAMES  p_timeframe,
      double           p_fixedLot,
      int              p_signalPeriod,
      double           p_signalDeviation,
      double           p_signaAddlDeviation,
      int              p_openDistance,
      double           p_stopLevel,
      double           p_takeLevel,
      int              p_ordersExpiration,
      int              p_maxCountOfOrders
   );                                     // Constructor

   virtual int       Init();              // Strategy initialization method
   virtual void      Tick();              // OnTick event handler
};
```

The class description is ready. All of its methods already have an implementation, except for the constructor. Let's add it. In the simplest case, the constructor of this class will only assign the values of the received parameters to the corresponding members of the class. Moreover, the first four parameters will do this by calling the base class constructor.

```
CSimpleVolumeStrategy::CSimpleVolumeStrategy(
   ulong            p_magic,
   string           p_symbol,
   ENUM_TIMEFRAMES  p_timeframe,
   double           p_fixedLot,
   int              p_signalPeriod,
   double           p_signalDeviation,
   double           p_signaAddlDeviation,
   int              p_openDistance,
   double           p_stopLevel,
   double           p_takeLevel,
   int              p_ordersExpiration,
   int              p_maxCountOfOrders) :
   // Initialization list
   CStrategy(p_magic, p_symbol, p_timeframe, p_fixedLot), // Call the base class constructor
   signalPeriod_(p_signalPeriod),
   signalDeviation_(p_signalDeviation),
   signaAddlDeviation_(p_signaAddlDeviation),
   openDistance_(p_openDistance),
   stopLevel_(p_stopLevel),
   takeLevel_(p_takeLevel),
   ordersExpiration_(p_ordersExpiration),
   maxCountOfOrders_(p_maxCountOfOrders)
{}
```

There is very little left to do. Rename fixedLot\_ and magicN\_ into m\_fixedLot and m\_magic in all the places where they are met. Replace the use of the function for getting the Symbol() current symbol with the m\_symbol base class variable and the PERIOD\_CURRENT constant with m\_timeframe. Save this code in the SimpleVolumesStrategy.mqh file of the current folder.

### EA class

Let's create the CAdvisor base class. Its objective is store the list of objects of specific trading strategies and launch their event handlers. For this class, the name CExpert would be more appropriate, but it is already used in the standard library, so we will use CAdvisor instead.

```
#include "Strategy.mqh"

class CAdvisor : public CObject {
protected:
   CStrategy         *m_strategies[];  // Array of trading strategies
   int               m_strategiesCount;// Number of strategies

public:
   virtual int       Init();           // EA initialization method
   virtual void      Tick();           // OnTick event handler
   virtual void      Deinit();         // Deinitialization method

   void              AddStrategy(CStrategy &strategy);   // Strategy adding method
};
```

In the Init() and Tick() methods, all strategies from the m\_strategies\[\] array are looped through and the corresponding event handling methods are called for them.

```
void CAdvisor::Tick(void) {
   // Call OnTick handling for all strategies
   for(int i = 0; i < m_strategiesCount; i++) {
      m_strategies[i].Tick();
   }
}
```

In the strategy adding method, this is exactly what happens.

```
void CAdvisor::AddStrategy(CStrategy &strategy) {
   // Increase the strategy number counter by 1
   m_strategiesCount = ArraySize(m_strategies) + 1;

   // Increase the size of the strategies array
   ArrayResize(m_strategies, m_strategiesCount);
   // Write a pointer to the strategy object to the last element
   m_strategies[m_strategiesCount - 1] = GetPointer(strategy);
}
```

Let's save this code in the Advisor.mqh file of the current folder. Based on this class, it will be possible to create descendants that implement any specific methods of managing multiple strategies. But for now we will limit ourselves to only this base class and will not interfere with the work of individual strategies in any way.

### Trading EA with multiple strategies

To write a trading EA, we just need to create a global EA object (of the CAdvisor class).

In the OnInit() initialization event handler, we will create strategy objects with the selected parameters and add them to the EA object. After this, we call the Init() method of the EA object so that all strategies are initialized in it.

The OnTick() and OnDeinit() event handlers simply call the corresponding methods of the EA object.

```
#include "Advisor.mqh"
#include "SimpleVolumesStartegy.mqh"

input double depoPart_  = 0.8;      // Part of the deposit for one strategy
input ulong  magic_     = 27182;    // Magic

CAdvisor     expert;                // EA object

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   expert.AddStrategy(...);
   expert.AddStrategy(...);

   int res = expert.Init();   // Initialization of all EA strategies

   return(res);
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
   expert.Deinit();
}
//+------------------------------------------------------------------+
```

Now let's look at creating strategy objects in more detail. Since each instance of the strategy opens and takes into account its own orders and positions, they should have different Magic. Magic is the first parameter of the strategy constructor. Therefore, to guarantee different Magic, we will add different numbers to the original Magic specified in the magic\_ parameter.

```
   expert.AddStrategy(new CSimpleVolumeStrategy(magic_ + 1, ...));
   expert.AddStrategy(new CSimpleVolumeStrategy(magic_ + 2, ...));
```

The second and third constructor parameters are the symbol and period. Since we performed optimization on EURGBP H1, we indicate these specific values.

```
   expert.AddStrategy(new CSimpleVolumeStrategy(
                         magic_ + 1, "EURGBP", PERIOD_H1, ...));
   expert.AddStrategy(new CSimpleVolumeStrategy(
                         magic_ + 2, "EURGBP", PERIOD_H1, ...));
```

The next important parameter is the size of the positions being opened. We have already calculated the appropriate size for two strategies (0.34 and 0.10). But this is the size for handling a drawdown of up to 10% of USD 10,000 with strategies operating separately. If two strategies work simultaneously, the drawdown of the first may be added to the drawdown of the second. In the worst case, to stay within the stated 10%, we will have to halve the size of the opened positions. But it may happen that the drawdowns of the two strategies do not coincide or even somewhat compensate each other. In this case, we can reduce the position size a bit and still not exceed 10%. Therefore, let's make the reducing multiplier an EA parameter (depoPart\_), for which we will then select the optimal value.

Remaining parameters of the strategy constructor are the sets of values that we selected after optimizing the simple EA. The final results are as follows:

```
   expert.AddStrategy(new CSimpleVolumeStrategy(
                         magic_ + 1, "EURGBP", PERIOD_H1,
                         NormalizeDouble(0.34 * depoPart_, 2),
                         130, 0.9, 1.4, 231, 3750, 50, 600, 3)
                     );
   expert.AddStrategy(new CSimpleVolumeStrategy(
                         magic_ + 2, "EURGBP", PERIOD_H1,
                         NormalizeDouble(0.10 * depoPart_, 2),
                         159, 1.7, 0.8, 248, 3600, 495, 39000, 3)
                     );
```

Save the resulting code in the SimpleVolumesExpert.mq5 file of the current folder.

### Test results

Before testing the combined EA, let's remember that the strategy with the first set of parameters should have yielded a profit of approximately 91%, and with the second set of parameters - 72% (for the starting deposit of USD 10,000 and a maximum drawdown of 10% (USD 1,000) with an optimal lot).

Let's select the optimal value of the depoPart\_ parameter according to the criterion of maintaining a given drawdown and get the following results.

![](https://c.mql5.com/2/65/411458913979.png)

Fig. 3. Combined EA operation result

The balance at the end of the test period was approximately USD 22,400, which means the profit of 124%. This is more than we got when running individual instances of this strategy. We were able to improve trading results by working only with the existing trading strategy without making any changes to it.

### Conclusion

We have only taken one small step towards achieving our goal. It has given us greater confidence that this approach can improve the quality of trading. As of now, the EA lacks many important aspects.

For example, we looked at a very simple strategy that does not control the closing of positions in any way, works without the need to accurately determine the beginning of the bar, and does not use any heavy calculations. To restore the state after restarting the terminal, you do not need to make any additional efforts except for counting open positions and orders, which the EA can do. But not every strategy will be so simple. In addition, the EA cannot work on Netting accounts and can keep opposite positions open at the same time. We have not considered working on different symbols. And so on and so forth...

These aspects should definitely be considered before real trading begins. So stay tuned for the new articles.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14026](https://www.mql5.com/ru/articles/14026)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14026.zip "Download all attachments in the single ZIP archive")

[SimpleVolumes.mq5](https://www.mql5.com/en/articles/download/14026/simplevolumes.mq5 "Download SimpleVolumes.mq5")(21.22 KB)

[Strategy.mqh](https://www.mql5.com/en/articles/download/14026/strategy.mqh "Download Strategy.mqh")(3.78 KB)

[SimpleVolumesStartegy.mqh](https://www.mql5.com/en/articles/download/14026/simplevolumesstartegy.mqh "Download SimpleVolumesStartegy.mqh")(25.84 KB)

[Advisor.mqh](https://www.mql5.com/en/articles/download/14026/advisor.mqh "Download Advisor.mqh")(6.29 KB)

[SimpleVolumesExpert.mq5](https://www.mql5.com/en/articles/download/14026/simplevolumesexpert.mq5 "Download SimpleVolumesExpert.mq5")(4.97 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/467423)**
(31)


![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
25 Jan 2024 at 15:19

**fxsaber [#](https://www.mql5.com/ru/forum/461185/page3#comment_51900256):**

It seems to me that the architectural skeleton should be extremely concise and easy to use. That's why the basic class of TC is like this.

Next, a little fleshing out of the tendons. It should be very simple.

There is something similar simple (in terms of interface) but extended (in terms of implementation) in the [book](https://www.mql5.com/ru/book/automation/tester/tester_example_ea).

```
interface TradingSignal
{
   virtual int signal(void);
};

interface TradingStrategy
{
   virtual bool trade(void);
};

...
```

```
...
AutoPtr<TradingStrategy> strategy;

int OnInit()
{
   strategy = new SimpleStrategy(
      new BandOsMaSignal(...параметры...), Magic, StopLoss, Lots);
   return INIT_SUCCEEDED;
}

void OnTick()
{
   if(strategy[] != NULL)
   {
      strategy[].trade();
   }
}
...
```

![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
26 Jan 2024 at 13:09

**Stanislav Korotky [#](https://www.mql5.com/ru/forum/461185/page3#comment_51909531):**

There is something similarly simple (in terms of interface) but extended (in terms of implementation) in the [book](https://www.mql5.com/ru/book/automation/tester/tester_example_ea).

Where can I download the source code?

![Aleksandr Slavskii](https://c.mql5.com/avatar/2017/4/58E88E5E-2732.jpg)

**[Aleksandr Slavskii](https://www.mql5.com/en/users/s22aa)**
\|
26 Jan 2024 at 13:47

**fxsaber [#](https://www.mql5.com/ru/forum/461185/page3#comment_51923281):**

Where to download the source code?

[https://www.mql5.com/ru/code/45595](https://www.mql5.com/ru/code/45595)

![Isaac Amo](https://c.mql5.com/avatar/2024/5/664e0931-d877.png)

**[Isaac Amo](https://www.mql5.com/en/users/amietron)**
\|
23 May 2024 at 19:21

Very interesting strategy!!


![gardee005](https://c.mql5.com/avatar/avatar_na2.png)

**[gardee005](https://www.mql5.com/en/users/gardee005)**
\|
28 Oct 2024 at 15:36

nice article , what i understand of it as a novice. nicely explained too. thank you.


![Spurious Regressions in Python](https://c.mql5.com/2/78/Spurious_Regressions_in_Python___LOGO__BIG-transformed.png)[Spurious Regressions in Python](https://www.mql5.com/en/articles/14199)

Spurious regressions occur when two time series exhibit a high degree of correlation purely by chance, leading to misleading results in regression analysis. In such cases, even though variables may appear to be related, the correlation is coincidental and the model may be unreliable.

![MQL5 Wizard Techniques you should know (Part 20): Symbolic Regression](https://c.mql5.com/2/78/MQL5_Wizard_Techniques_you_should_know_4Part_20x___LOGO.png)[MQL5 Wizard Techniques you should know (Part 20): Symbolic Regression](https://www.mql5.com/en/articles/14943)

Symbolic Regression is a form of regression that starts with minimal to no assumptions on what the underlying model that maps the sets of data under study would look like. Even though it can be implemented by Bayesian Methods or Neural Networks, we look at how an implementation with Genetic Algorithms can help customize an expert signal class usable in the MQL5 wizard.

![Building A Candlestick Trend Constraint Model(Part 3): Detecting changes in trends while using this system](https://c.mql5.com/2/78/Building_A_Candlestick_Trend_Constraint_Model_Part_3___LOGO.png)[Building A Candlestick Trend Constraint Model(Part 3): Detecting changes in trends while using this system](https://www.mql5.com/en/articles/14853)

This article explores how economic news releases, investor behavior, and various factors can influence market trend reversals. It includes a video explanation and proceeds by incorporating MQL5 code into our program to detect trend reversals, alert us, and take appropriate actions based on market conditions. This builds upon previous articles in the series.

![Population optimization algorithms: Binary Genetic Algorithm (BGA). Part II](https://c.mql5.com/2/65/Population_optimization_algorithms__Binary_Genetic_Algorithm_gBGAm___Part_2____LOGO.png)[Population optimization algorithms: Binary Genetic Algorithm (BGA). Part II](https://www.mql5.com/en/articles/14040)

In this article, we will look at the binary genetic algorithm (BGA), which models the natural processes that occur in the genetic material of living things in nature.

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/14026&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049170163481290316)

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