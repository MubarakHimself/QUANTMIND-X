---
title: Advanced Order Execution Algorithms in MQL5: TWAP, VWAP, and Iceberg Orders
url: https://www.mql5.com/en/articles/17934
categories: Trading Systems, Integration
relevance_score: 6
scraped_at: 2026-01-23T11:35:49.097411
---

[![](https://www.mql5.com/ff/sh/7h2yc16rtqsn2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Market analysis\\
\\
Dozens of channels, thousands of subscribers and daily updates. Learn more about trading.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=glufvbpblsoxonicqfngsyuzwfebnilr&s=103cc3ab372a16872ca1698fc86368ffe3b3eaa21b59b4006d5c6c10f48ad545&uid=&ref=https://www.mql5.com/en/articles/17934&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062578162321237219)

MetaTrader 5 / Trading systems


01. [Introduction](https://www.mql5.com/en/articles/17934#section1)
02. [Understanding Execution Algorithms](https://www.mql5.com/en/articles/17934#section2)
03. [Implementation in MQL5](https://www.mql5.com/en/articles/17934#section3)
04. [Performance Analyzer Implementation](https://www.mql5.com/en/articles/17934#section4)
05. [Comparing Algorithm Performance](https://www.mql5.com/en/articles/17934#section5)
06. [Integrating Execution Algorithms with Trading Strategies](https://www.mql5.com/en/articles/17934#section6)
07. [Integration Examples](https://www.mql5.com/en/articles/17934#section7)
08. [Integrated Strategy](https://www.mql5.com/en/articles/17934#section8)
09. [Backtest Results](https://www.mql5.com/en/articles/17934#section9)
10. [Conclusion](https://www.mql5.com/en/articles/17934#section10)

### Introduction

Imagine you’re standing at the trading floor’s edge, heart pounding as prices tick by in real time. One wrong move, one oversized order, and your edge evaporates in a flash. Welcome to the world where execution quality isn’t just a nice-to-have—it’s the secret weapon separating winners from the rest.

For decades, institutional heavyweights have quietly wielded sophisticated algorithms to slice, dice, and stealthily deploy their orders, all to dodge slippage and tame market impact. Now, thanks to the flexibility of MQL5, that same powerhouse playbook is within reach of every ambitious retail trader.

**What’s the Big Deal?**

Picture this: you spot a golden opportunity and decide to go big. You slam in a market order for your full size, only to watch the price slip away under the weight of your own trade. In seconds, your ideal entry becomes a shaky compromise. That’s the notorious drag of market impact—and it bites even in the most liquid venues.

Execution algorithms are your antidote. By breaking a large order into a sequence of smaller, strategically timed slices, they smooth out your footprint on the order book. The result? Less slippage, tighter fills, and an overall improvement in your average execution price.

**From Ivory Towers to Your Desktop**

“Sure,” you might shrug, “but I’m not moving institutional sums.” Here’s the kicker: you don’t have to. Whether you’re deploying a half-lot or a handful of mini-lots, volatility can still twist your execution. These tools help you:

- Tame Slippage: Even modest orders can wander in choppy markets.
- Sharpen Your Edge: Layered executions often land you a more favorable average price than a one-shot gamble.
- Stay Zen: Automated workflows strip away the temptation to panic-buy or panic-sell.
- Scale Seamlessly: As your account grows, your execution stays crisp—no matter how hefty your orders become.
- Fly Under the Radar: Iceberg Orders, in particular, cloak your true order size, keeping prying algos guessing.


Today’s democratized landscape means the same execution tech that once demanded multi-million-dollar budgets can now run on your personal trading station. By dropping polished MQL5 code for TWAP, VWAP, and Iceberg strategies into your platform, you’ll arm yourself with institutional firepower—without ever leaving the retail domain.

Get ready to flip the script on your execution process. The game is changing, and with these algorithms in your toolkit, you’ll be playing to win.

### Understanding Execution Algorithms

Before diving into the implementation details, it's essential to understand the theory behind each execution algorithm and why they're effective in different market scenarios.

1. Time-Weighted Average Price (TWAP): TWAP is a straightforward execution algorithm that divides a large order into equal parts and sends them at fixed time intervals over a set period. Its aim is to match the average price of the instrument over that time.


   - How it works:

     - Sends orders at regular time intervals between start and end.
     - Usually uses equal-sized orders (though you can add randomness to sizes).
     - Follows a predetermined timetable, regardless of price moves.
     - Spreads market impact evenly over time to keep slippage low.
   - When to use:

     - You need an average execution price over a specific timeframe.
     - Liquidity is steady throughout the trading period.
     - You have a fixed window to complete your order.
     - You prefer a simple, predictable approach.
2. Volume-Weighted Average Price (VWAP): VWAP improves on TWAP by weighting order sizes according to expected volume. Instead of equal chunks, it sends larger trades when volume tends to be higher.


   - How it works:

     - Allocates order size in proportion to historical volume patterns.
     - Analyzes past trading volumes to predict future volume distribution.
     - Can adapt to real-time volume changes in some implementations.
     - Executes more during high-volume periods to reduce impact.
   - When to use:

     - Your performance is measured against VWAP.
     - Volume follows a predictable daily pattern.
     - You are trading in a market where liquidity varies through the session.
     - You want to align with the market’s natural flow.
3. Iceberg Orders: Iceberg Orders focus on hiding the true size of a large order. Only a small “tip” is visible at any time; once it fills, the next portion appears.


   - How it works:

     - Displays only part of the total order.
     - Releases new visible chunks after each tip is executed.
     - You can fix or randomize the visible size to reduce detection.
     - Often placed as limit orders for better price control.
   - When to use:

     - You need to conceal your full order size.
     - The market is not very liquid and large trades can move prices.
     - You want to maintain execution at a specific price level.
     - You’re concerned about other traders detecting and front-running your order.

### Implementation in MQL5

Now that we understand the theory behind these execution algorithms, let's implement them in MQL5. We'll create a modular, object-oriented framework that allows these algorithms to be used individually or combined into a unified execution system.

**Base Class: CExecutionAlgorithm**

We'll start by defining a base class that provides common functionality for all execution algorithms:

```
//+------------------------------------------------------------------+
//| Base class for all execution algorithms                          |
//+------------------------------------------------------------------+
class CExecutionAlgorithm
{
protected:
   string            m_symbol;              // Symbol to trade
   double            m_totalVolume;         // Total volume to execute
   double            m_executedVolume;      // Volume already executed
   double            m_remainingVolume;     // Volume remaining to execute
   datetime          m_startTime;           // Start time for execution
   datetime          m_endTime;             // End time for execution
   bool              m_isActive;            // Flag indicating if the algorithm is active
   int               m_totalOrders;         // Total number of orders placed
   int               m_filledOrders;        // Number of filled orders
   double            m_avgExecutionPrice;   // Average execution price
   double            m_executionValue;      // Total value of executed orders
   int               m_slippage;            // Allowed slippage in points

public:
   // Constructor
   CExecutionAlgorithm(string symbol, double volume, datetime startTime, datetime endTime, int slippage = 3);

   // Destructor
   virtual ~CExecutionAlgorithm();

   // Common methods
   virtual bool      Initialize();
   virtual bool      Execute() = 0;
   virtual bool      Update() = 0;
   virtual bool      Terminate();

   // Utility methods
   bool              PlaceOrder(ENUM_ORDER_TYPE orderType, double volume, double price);
   bool              CancelOrder(ulong ticket);
   void              UpdateAverageExecutionPrice(double price, double volume);

   // Getters
   string            GetSymbol() const { return m_symbol; }
   double            GetTotalVolume() const { return m_totalVolume; }
   double            GetExecutedVolume() const { return m_executedVolume; }
   double            GetRemainingVolume() const { return m_remainingVolume; }
   datetime          GetStartTime() const { return m_startTime; }
   datetime          GetEndTime() const { return m_endTime; }
   bool              IsActive() const { return m_isActive; }
   int               GetTotalOrders() const { return m_totalOrders; }
   int               GetFilledOrders() const { return m_filledOrders; }
   double            GetAverageExecutionPrice() const { return m_avgExecutionPrice; }
};
```

The PlaceOrder method is particularly important as it handles the actual order execution and updates the volume tracking:

```
bool CExecutionAlgorithm::PlaceOrder(ENUM_ORDER_TYPE orderType, double volume, double price)
{
   // Prepare the trade request
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.symbol = m_symbol;
   request.volume = volume;
   request.type = orderType;
   request.price = price;
   request.deviation = m_slippage;
   request.magic = 123456; // Magic number for identification

   // Send the order
   bool success = OrderSend(request, result);

   if(!success)
   {
      Print("OrderSend error: ", GetLastError());
      return false;
   }

   // Check the result
   if(result.retcode != TRADE_RETCODE_DONE)
   {
      Print("OrderSend failed with code: ", result.retcode);
      return false;
   }

   // Update statistics
   m_totalOrders++;
   m_filledOrders++;
   m_executedVolume += volume;
   m_remainingVolume -= volume;
   UpdateAverageExecutionPrice(price, volume);

   // Store the order ticket for future reference
   ulong ticket = result.order;

   return true;
}
```

This function builds and sends a market order by zero‐initializing an MqlTradeRequest and MqlTradeResult, filling in symbol, volume, order type, price, slippage and a magic number, then calling OrderSend. If the send fails or the broker’s return code isn’t TRADE\_RETCODE\_DONE, it logs the error and returns false. On success it updates internal counters (total/fill counts, executed and remaining volume), recalculates the average price, captures the ticket ID, and returns true.

**Implementation of TWAP**

The TWAP algorithm divides the execution period into equal time intervals and places orders of equal (or randomized) size at each interval:

```
//+------------------------------------------------------------------+
//| Time-Weighted Average Price (TWAP) Algorithm                     |
//+------------------------------------------------------------------+
class CTWAP : public CExecutionAlgorithm
{
private:
   int               m_intervals;           // Number of time intervals
   int               m_currentInterval;     // Current interval
   datetime          m_nextExecutionTime;   // Next execution time
   double            m_intervalVolume;      // Volume per interval
   bool              m_useRandomization;    // Whether to randomize order sizes
   double            m_randomizationFactor; // Factor for randomization (0-1)
   ENUM_ORDER_TYPE   m_orderType;           // Order type (buy or sell)
   bool              m_firstOrderPlaced;    // Flag to track if first order has been placed
   int               m_initialDelay;        // Initial delay in seconds before first execution
   datetime          m_lastCheckTime;       // Last time order status was checked
   int               m_checkInterval;       // How often to check order status (seconds)

public:
   // Constructor
   CTWAP(string symbol, double volume, datetime startTime, datetime endTime,
         int intervals, ENUM_ORDER_TYPE orderType, bool useRandomization = false,
         double randomizationFactor = 0.2, int slippage = 3, int initialDelay = 10);

   // Implementation of virtual methods
   virtual bool      Initialize() override;
   virtual bool      Execute() override;
   virtual bool      Update() override;
   virtual bool      Terminate() override;

   // TWAP specific methods
   void              CalculateIntervalVolume();
   datetime          CalculateNextExecutionTime();
   double            GetRandomizedVolume(double baseVolume);
   bool              IsTimeToExecute();
};
```

Key Method: CalculateNextExecutionTime

This method ensures orders are properly spaced over time, with an initial delay for the first order:

```
datetime CTWAP::CalculateNextExecutionTime()
{
   // Calculate the duration of each interval
   int totalSeconds = (int)(m_endTime - m_startTime);
   int intervalSeconds = totalSeconds / m_intervals;

   // Calculate the next execution time
   datetime nextTime;

   if(m_currentInterval == 0) {
      // First interval - start at the defined start time plus initial delay
      nextTime = m_startTime + m_initialDelay;

      Print("TWAP: First execution time calculated with ", m_initialDelay,
            " seconds delay: ", TimeToString(nextTime));
   } else {
      // For subsequent intervals, ensure proper spacing from current time
      datetime currentTime = TimeCurrent();
      nextTime = currentTime + intervalSeconds;

      // Make sure we don't exceed the end time
      if(nextTime > m_endTime)
         nextTime = m_endTime;

      Print("TWAP: Next execution time calculated: ", TimeToString(nextTime),
            " (interval: ", intervalSeconds, " seconds)");
   }

   return nextTime;
}
```

This method splits the window from m\_startTime to m\_endTime into m\_intervals equal segments and returns when the next trade should fire: on the very first call it’s simply m\_startTime + m\_initialDelay, and on every subsequent call it’s TimeCurrent() + one interval’s worth of seconds (but never past m\_endTime).

Execute Method:The Execute method checks if it's time to place an order and handles the actual order placement:

```
bool CTWAP::Execute()
{
   if(!m_isActive)
      return false;

   // Check if it's time to execute the next order
   if(!IsTimeToExecute())
      return true; // Not time yet

   // Calculate the volume for this execution
   double volumeToExecute = m_useRandomization ?
                           GetRandomizedVolume(m_intervalVolume) :
                           m_intervalVolume;

   // Ensure we don't exceed the remaining volume
   if(volumeToExecute > m_remainingVolume)
      volumeToExecute = m_remainingVolume;

   // Get current market price
   double price = 0.0;
   if(m_orderType == ORDER_TYPE_BUY)
      price = SymbolInfoDouble(m_symbol, SYMBOL_ASK);
   else
      price = SymbolInfoDouble(m_symbol, SYMBOL_BID);

   Print("TWAP: Placing order for interval ", m_currentInterval,
         ", Volume: ", DoubleToString(volumeToExecute, 2),
         ", Price: ", DoubleToString(price, _Digits));

   // Place the order using OrderSend directly for more control
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.symbol = m_symbol;
   request.volume = volumeToExecute;
   request.type = m_orderType;
   request.price = price;
   request.deviation = m_slippage;
   request.magic = 123456; // Magic number for identification

   // Send the order
   bool success = OrderSend(request, result);

   if(!success)
   {
      Print("TWAP: OrderSend error: ", GetLastError());
      return false;
   }

   // Check the result
   if(result.retcode != TRADE_RETCODE_DONE)
   {
      Print("TWAP: OrderSend failed with code: ", result.retcode);
      return false;
   }

   // Update statistics
   m_totalOrders++;
   m_filledOrders++;
   m_executedVolume += volumeToExecute;
   m_remainingVolume -= volumeToExecute;

   // Update interval counter
   m_currentInterval++;
   m_firstOrderPlaced = true;

   // Calculate the time for the next execution
   if(m_currentInterval < m_intervals && m_remainingVolume > 0)
      m_nextExecutionTime = CalculateNextExecutionTime();
   else
      m_isActive = false; // All intervals completed or no volume left

   Print("TWAP: Executed ", DoubleToString(volumeToExecute, 2),
         " at price ", DoubleToString(price, _Digits),
         ". Remaining: ", DoubleToString(m_remainingVolume, 2),
         ", Next execution: ", TimeToString(m_nextExecutionTime));

   return true;
}
```

This Execute method manages one slice of your TWAP run. First it aborts if the strategy isn’t active or if it isn’t yet time to trade. When it is, it picks either a fixed or randomized slice of your remaining volume (never exceeding what’s left), then looks up the current ask (for buys) or bid (for sells). It logs the interval, volume and price, builds an MqlTradeRequest with your symbol, volume, type, price, slippage and magic number, and calls OrderSend. If the send fails or the broker returns anything other than TRADE\_RETCODE\_DONE, it prints an error and returns false.

On success it increments your order counters, adjusts executed and remaining volume, bumps the interval count, marks that the first order went out, then either schedules the next execution time or deactivates the strategy if you’ve run out of intervals or volume. Finally it logs what happened and returns true.

**Implementation of VWAP**

The VWAP algorithm is similar to TWAP but distributes order sizes based on historical volume patterns:

```
//+------------------------------------------------------------------+
//| Volume-Weighted Average Price (VWAP) Algorithm                   |
//+------------------------------------------------------------------+
class CVWAP : public CExecutionAlgorithm
{
private:
   int               m_intervals;           // Number of time intervals
   int               m_currentInterval;     // Current interval
   datetime          m_nextExecutionTime;   // Next execution time
   double            m_volumeProfile[];     // Historical volume profile
   double            m_intervalVolumes[];   // Volume per interval based on profile
   bool              m_adaptiveMode;        // Whether to adapt to real-time volume
   ENUM_ORDER_TYPE   m_orderType;           // Order type (buy or sell)
   int               m_historyDays;         // Number of days to analyze for volume profile
   bool              m_profileLoaded;       // Flag indicating if profile was loaded
   bool              m_firstOrderPlaced;    // Flag to track if first order has been placed
   int               m_initialDelay;        // Initial delay in seconds before first execution
   datetime          m_lastCheckTime;       // Last time order status was checked
   int               m_checkInterval;       // How often to check order status (seconds)

public:
   // Constructor
   CVWAP(string symbol, double volume, datetime startTime, datetime endTime,
         int intervals, ENUM_ORDER_TYPE orderType, int historyDays = 5,
         bool adaptiveMode = true, int slippage = 3, int initialDelay = 10);

   // Implementation of virtual methods
   virtual bool      Initialize() override;
   virtual bool      Execute() override;
   virtual bool      Update() override;
   virtual bool      Terminate() override;

   // VWAP specific methods
   bool              LoadVolumeProfile();
   void              CalculateIntervalVolumes();
   void              AdjustToRealTimeVolume();
   datetime          CalculateNextExecutionTime();
   double            GetCurrentVWAP();
   bool              IsTimeToExecute();
};
```

Like TWAP, VWAP also implements the CalculateNextExecutionTime method to ensure proper spacing of orders:

```
datetime CVWAP::CalculateNextExecutionTime()
{
   // Calculate the duration of each interval
   int totalSeconds = (int)(m_endTime - m_startTime);
   int intervalSeconds = totalSeconds / m_intervals;

      // Calculate the next execution time
   datetime nextTime;

   if(m_currentInterval == 0) {
      // First interval - start at the defined start time plus initial delay
      nextTime = m_startTime + m_initialDelay;

      Print("VWAP: First execution time calculated with ", m_initialDelay,
            " seconds delay: ", TimeToString(nextTime));
   } else {
      // For subsequent intervals, ensure proper spacing from current time
      datetime currentTime = TimeCurrent();
      nextTime = currentTime + intervalSeconds;

      // Make sure we don't exceed the end time
      if(nextTime > m_endTime)
         nextTime = m_endTime;

      Print("VWAP: Next execution time calculated: ", TimeToString(nextTime),
            " (interval: ", intervalSeconds, " seconds)");
   }

   return nextTime;
}
```

**Implementation of Iceberg Orders**

Iceberg Orders hide the true size of an order by exposing only a small portion to the market at any given time:

```
//+------------------------------------------------------------------+
//| Iceberg Order Implementation                                     |
//+------------------------------------------------------------------+
class CIcebergOrder : public CExecutionAlgorithm
{
private:
   double            m_visibleVolume;          // Visible portion of the order
   double            m_minVisibleVolume;       // Minimum visible volume
   double            m_maxVisibleVolume;       // Maximum visible volume
   bool              m_useRandomVisibleVolume; // Whether to randomize visible volume
   int               m_orderPlacementDelay;    // Delay between order placements (ms)
   bool              m_avoidRoundNumbers;      // Whether to avoid round numbers in price
   double            m_limitPrice;             // Limit price for the orders
   ulong             m_currentOrderTicket;     // Current active order ticket
   ENUM_ORDER_TYPE   m_orderType;              // Order type (buy or sell)
   bool              m_orderActive;            // Flag indicating if an order is currently active
   int               m_priceDeviation;         // Price deviation to avoid round numbers (in points)
   datetime          m_lastCheckTime;          // Last time order status was checked
   int               m_checkInterval;          // How often to check order status (seconds)
   int               m_maxOrderLifetime;       // Maximum lifetime for an order in seconds
   datetime          m_orderPlacementTime;     // When the current order was placed

public:
   // Constructor
   CIcebergOrder(string symbol, double volume, double limitPrice, ENUM_ORDER_TYPE orderType,
                 double visibleVolume, double minVisibleVolume = 0.0, double maxVisibleVolume = 0.0,
                 bool useRandomVisibleVolume = true, int orderPlacementDelay = 1000,
                 bool avoidRoundNumbers = true, int priceDeviation = 2, int slippage = 3);

   // Implementation of virtual methods
   virtual bool      Initialize() override;
   virtual bool      Execute() override;
   virtual bool      Update() override;
   virtual bool      Terminate() override;

   // Iceberg specific methods
   double            GetRandomVisibleVolume();
   double            AdjustPriceToAvoidRoundNumbers(double price);
   bool              CheckAndReplaceOrder();
   bool              IsOrderFilled(ulong ticket);
   bool              IsOrderPartiallyFilled(ulong ticket, double &filledVolume);
   bool              IsOrderCancelled(ulong ticket);
   bool              IsOrderExpired(ulong ticket);
   bool              IsOrderTimeout();
   ulong             GetCurrentOrderTicket() { return m_currentOrderTicket; }
   bool              IsOrderActive() { return m_orderActive; }
};
```

The Execute method places a new visible portion of the order:

```
bool CIcebergOrder::Execute()
{
   if(!m_isActive)
   {
      Print("Iceberg: Execute called but algorithm is not active");
      return false;
   }

   // If an order is already active, check its status
   if(m_orderActive)
   {
      Print("Iceberg: Execute called with active order ", m_currentOrderTicket);
      return CheckAndReplaceOrder();
   }

   // Calculate the volume for this execution
   double volumeToExecute = m_useRandomVisibleVolume ?
                           GetRandomVisibleVolume() :
                           m_visibleVolume;

   // Ensure we don't exceed the remaining volume
   if(volumeToExecute > m_remainingVolume)
      volumeToExecute = m_remainingVolume;

   Print("Iceberg: Placing order for ", DoubleToString(volumeToExecute, 2),
         " at price ", DoubleToString(m_limitPrice, _Digits));

   // Place the order using OrderSend directly for more control
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_PENDING;
   request.symbol = m_symbol;
   request.volume = volumeToExecute;
   request.type = m_orderType;
   request.price = m_limitPrice;
   request.deviation = m_slippage;
   request.magic = 123456; // Magic number for identification

   // Send the order
   bool success = OrderSend(request, result);

   if(!success)
   {
      Print("Iceberg: OrderSend error: ", GetLastError());
      return false;
   }

   // Check the result
   if(result.retcode != TRADE_RETCODE_DONE)
   {
      Print("Iceberg: OrderSend failed with code: ", result.retcode);
      return false;
   }

   // Store the order ticket
   m_currentOrderTicket = result.order;
   m_orderActive = true;
   m_orderPlacementTime = TimeCurrent();

   Print("Iceberg: Order placed successfully. Ticket: ", m_currentOrderTicket,
         ", Volume: ", DoubleToString(volumeToExecute, 2),
         ", Remaining: ", DoubleToString(m_remainingVolume, 2));

   return true;
}
```

When Execute runs, it first verifies the algorithm is active. If there’s already a live iceberg child order, it calls CheckAndReplaceOrder() to see if it needs cancellation or refill. Otherwise, it picks a visible slice (either fixed or randomized), caps it by the remaining total, and logs size and price.

It then builds a pending‐order request ( TRADE\_ACTION\_PENDING ) with symbol, volume, limit price, slippage, and magic number, and calls OrderSend. On error or non-done return codes it logs and returns false; on success it saves the new ticket, marks the order active, records placement time, logs the details, and returns true.

The Update method includes order timeout detection to ensure orders don't remain active indefinitely:

```
bool CIcebergOrder::Update()
{
   if(!m_isActive)
   {
      Print("Iceberg: Update called but algorithm is not active");
      return false;
   }

   // Check if all volume has been executed
   if(m_remainingVolume <= 0)
   {
      Print("Iceberg: All volume executed. Terminating algorithm.");
      return Terminate();
   }

   // Check if it's time to check order status
   datetime currentTime = TimeCurrent();
   if(currentTime >= m_lastCheckTime + m_checkInterval)
   {
      m_lastCheckTime = currentTime;

      // Log current market conditions
      double currentBid = SymbolInfoDouble(m_symbol, SYMBOL_BID);
      double currentAsk = SymbolInfoDouble(m_symbol, SYMBOL_ASK);

      Print("Iceberg: Market update - Bid: ", DoubleToString(currentBid, _Digits),
            ", Ask: ", DoubleToString(currentAsk, _Digits),
            ", Limit Price: ", DoubleToString(m_limitPrice, _Digits));

      // If an order is active, check its status
      if(m_orderActive)
      {
         // Check if the order has been active too long
         if(IsOrderTimeout())
         {
            Print("Iceberg: Order ", m_currentOrderTicket, " has timed out. Replacing it.");

            // Cancel the current order
            if(!CancelOrder(m_currentOrderTicket))
            {
               Print("Iceberg: Failed to cancel timed out order ", m_currentOrderTicket);
            }

            // Reset order tracking
            m_orderActive = false;
            m_currentOrderTicket = 0;

            // Place a new order after a delay
            Sleep(m_orderPlacementDelay);
            return Execute();
         }

         return CheckAndReplaceOrder();
      }
      else
      {
         // If no order is active, execute a new one
         Print("Iceberg: No active order, executing new order");
         return Execute();
      }
   }

   return true;
}
```

Update periodically polls market and order status at intervals defined by m\_checkInterval. If all volume is done, it terminates. Otherwise, once the check time arrives, it logs current bid/ask and limit price. If an order is active, it tests for timeout: if expired, it cancels, resets state, sleeps for m\_orderPlacementDelay, and re-executes to place a fresh slice; if not timed out, it defers to CheckAndReplaceOrder(). If there’s no active order, it simply calls Execute to send the next visible portion.

### Performance Analyzer Implementation

Our performance analyzer class tracks these metrics and provides methods to analyze and compare algorithm performance:

```
//+------------------------------------------------------------------+
//| Performance Analyzer for Execution Algorithms                    |
//+------------------------------------------------------------------+
class CPerformanceAnalyzer
{
private:
   string            m_symbol;              // Symbol being analyzed
   datetime          m_startTime;           // Analysis start time
   datetime          m_endTime;             // Analysis end time
   double            m_decisionPrice;       // Price at decision time
   double            m_avgExecutionPrice;   // Average execution price
   double            m_totalVolume;         // Total volume executed
   double            m_implementationShortfall; // Implementation shortfall
   double            m_marketImpact;        // Estimated market impact
   double            m_slippage;            // Average slippage
   int               m_executionTime;       // Total execution time in seconds
   double            m_priceImprovement;    // Total price improvement

public:
   // Constructor
   CPerformanceAnalyzer(string symbol, double decisionPrice);

   // Analysis methods
   void              RecordExecution(datetime time, double price, double volume);
   void              CalculateMetrics();
   void              CompareAlgorithms(CPerformanceAnalyzer &other);

   // Reporting methods
   void              PrintReport();
   void              SaveReportToFile(string filename);

   // Getters
   double            GetImplementationShortfall() const { return m_implementationShortfall; }
   double            GetMarketImpact() const { return m_marketImpact; }
   double            GetSlippage() const { return m_slippage; }
   int               GetExecutionTime() const { return m_executionTime; }
   double            GetPriceImprovement() const { return m_priceImprovement; }
};
```

The CPerformanceAnalyzer encapsulates all of your post‐trade metrics in one place. When you construct it, you give it a symbol and the decision‐time benchmark price; it stamps the current time as the start. As each child order fills, you call RecordExecution(time, price, volume), which updates running totals—cumulative volume, weighted average execution price, and timestamps. Once the strategy finishes (or periodically), you call CalculateMetrics(), which computes:

- Implementation shortfall (the P&L difference between your decision price and the actual executions),
- Average slippage versus quoted prices,
- Estimated market impact from your footprint,
- Total execution time (end minus start),
- Price improvement if any versus benchmarks.

You can even compare two runs via CompareAlgorithms(otherAnalyzer) to see which strategy fared better. Finally, PrintReport() spits the key stats to the log for quick review, and SaveReportToFile(filename) lets you persist a full report externally. Lightweight getters expose each metric for custom dashboards or further analysis.

### Comparing Algorithm Performance

Different market conditions favor different execution algorithms. Here's a general comparison i.e. rule of thumb:

1. TWAP:
   - Best for: Stable markets with consistent liquidity
   - Advantages: Simple, predictable execution pattern
   - Disadvantages: Doesn't adapt to changing market conditions
2. VWAP:
   - Best for: Markets with predictable volume patterns
   - Advantages: Aligns with natural market rhythm, often achieves better prices
   - Disadvantages: Requires historical volume data, more complex implementation
3. Iceberg Orders:
   - Best for: Less liquid markets or when price sensitivity is high
   - Advantages: Minimizes market impact, maintains price control
   - Disadvantages: Execution time can be unpredictable, risk of partial execution

### Integrating Execution Algorithms with Trading Strategies

The true power of these execution algorithms emerges when they're integrated with trading strategies. This section demonstrates how to incorporate our execution algorithms into complete trading systems.

**Execution Manager**

To simplify integration, we'll create an Execution Manager class that serves as a facade for all our execution algorithms:

```
//+------------------------------------------------------------------+
//| Execution Manager - Facade for all execution algorithms          |
//+------------------------------------------------------------------+
class CExecutionManager
{
private:
   CExecutionAlgorithm* m_algorithm;       // Current execution algorithm
   CPerformanceAnalyzer* m_analyzer;       // Performance analyzer

public:
   // Constructor
   CExecutionManager();

   // Destructor
   ~CExecutionManager();

   // Algorithm creation methods
   bool              CreateTWAP(string symbol, double volume, datetime startTime, datetime endTime,
                               int intervals, ENUM_ORDER_TYPE orderType, bool useRandomization = false,
                               double randomizationFactor = 0.2, int slippage = 3);

   bool              CreateVWAP(string symbol, double volume, datetime startTime, datetime endTime,
                               int intervals, ENUM_ORDER_TYPE orderType, int historyDays = 5,
                               bool adaptiveMode = true, int slippage = 3);

   bool              CreateIcebergOrder(string symbol, double volume, double limitPrice, ENUM_ORDER_TYPE orderType,
                                       double visibleVolume, double minVisibleVolume = 0.0, double maxVisibleVolume = 0.0,
                                       bool useRandomVisibleVolume = true, int orderPlacementDelay = 1000,
                                       bool avoidRoundNumbers = true, int priceDeviation = 2, int slippage = 3);

   // Execution methods
   bool              Initialize();
   bool              Execute();
   bool              Update();
   bool              Terminate();

   // Performance analysis
   void              EnablePerformanceAnalysis(double decisionPrice);
   void              PrintPerformanceReport();

   // Getters
   CExecutionAlgorithm* GetAlgorithm() { return m_algorithm; }
};
```

The CExecutionManager acts as a simple façade over any of your execution algorithms and ties them into a unified trading workflow. Internally it holds a pointer to the currently selected CExecutionAlgorithm (TWAP, VWAP or Iceberg) plus a CPerformanceAnalyzer to track how well your orders are doing.

You pick your strategy by calling one of the Create… methods—passing in symbol, total volume, start/end times (for TWAP/VWAP), interval counts or slice sizes, order type and any algorithm-specific knobs (randomization, history window, limit price, etc.). Once created, you drive it through the usual lifecycle:

1. Initialize() sets up any state or data you need.
2. Execute() fires off the next slice or child order.
3. Update() polls fills, market data or timeouts.
4. Terminate() cleans up once you’ve filled everything or want to stop.


If you enable performance analysis with EnablePerformanceAnalysis(), the manager will record execution prices against a decision benchmark, and you can dump a concise P/L and slippage report via PrintPerformanceReport(). You can always grab the raw algorithm object with GetAlgorithm() for custom probing or metrics.

### Integration Examples

Here are examples of how to integrate our execution algorithms with different types of trading strategies:

1. Trend-Following Strategy with TWAP Execution.


```
//+------------------------------------------------------------------+
//| Trend-Following Strategy with TWAP Execution                     |
//+------------------------------------------------------------------+
void OnTick()
{
      // Strategy parameters
      int maPeriodFast = 20;
      int maPeriodSlow = 50;
      double volume = 1.0;
      int executionIntervals = 5;

      // Calculate indicators
      double maFast = iMA(Symbol(), PERIOD_CURRENT, maPeriodFast, 0, MODE_SMA, PRICE_CLOSE, 0);
      double maSlow = iMA(Symbol(), PERIOD_CURRENT, maPeriodSlow, 0, MODE_SMA, PRICE_CLOSE, 0);

      // Check for entry conditions
      static bool inPosition = false;
      static CExecutionManager executionManager;

      if(!inPosition)
      {
         // Buy signal: Fast MA crosses above Slow MA
         if(maFast > maSlow)
         {
            // Create TWAP execution algorithm
            datetime startTime = TimeCurrent();
            datetime endTime = startTime + 3600; // 1 hour execution window

            if(executionManager.CreateTWAP(Symbol(), volume, startTime, endTime,
                                          executionIntervals, ORDER_TYPE_BUY))
            {
               executionManager.Initialize();
               inPosition = true;
               Print("Buy signal detected. Starting TWAP execution.");
            }
         }
      }
      else
      {
         // Update the execution algorithm
         if(executionManager.Update())
         {
            // Check if execution is complete
            if(!executionManager.GetAlgorithm().IsActive())
            {
               inPosition = false;
               Print("TWAP execution completed.");
            }
         }
      }
}
```

2. Mean-Reversion Strategy with VWAP Execution.


```
//+------------------------------------------------------------------+
//| Mean-Reversion Strategy with VWAP Execution                      |
//+------------------------------------------------------------------+
void OnTick()
{
      // Strategy parameters
      int rsiPeriod = 14;
      int rsiOversold = 30;
      int rsiOverbought = 70;
      double volume = 1.0;
      int executionIntervals = 5;

      // Calculate indicators
      double rsi = iRSI(Symbol(), PERIOD_CURRENT, rsiPeriod, PRICE_CLOSE, 0);

      // Check for entry conditions
      static bool inPosition = false;
      static bool isLong = false;
      static CExecutionManager executionManager;

      if(!inPosition)
      {
         // Buy signal: RSI oversold
         if(rsi < rsiOversold)
         {
            // Create VWAP execution algorithm
            datetime startTime = TimeCurrent();
            datetime endTime = startTime + 3600; // 1 hour execution window

            if(executionManager.CreateVWAP(Symbol(), volume, startTime, endTime,
                                          executionIntervals, ORDER_TYPE_BUY))
            {
               executionManager.Initialize();
               inPosition = true;
               isLong = true;
               Print("Buy signal detected. Starting VWAP execution.");
            }
         }
         // Sell signal: RSI overbought
         else if(rsi > rsiOverbought)
         {
            // Create VWAP execution algorithm
            datetime startTime = TimeCurrent();
            datetime endTime = startTime + 3600; // 1 hour execution window

            if(executionManager.CreateVWAP(Symbol(), volume, startTime, endTime,
                                          executionIntervals, ORDER_TYPE_SELL))
            {
               executionManager.Initialize();
               inPosition = true;
               isLong = false;
               Print("Sell signal detected. Starting VWAP execution.");
            }
         }
      }
      else
      {
         // Update the execution algorithm
         if(executionManager.Update())
         {
            // Check if execution is complete
            if(!executionManager.GetAlgorithm().IsActive())
            {
               inPosition = false;
               Print("VWAP execution completed.");
            }
         }

         // Check for exit conditions
         if(isLong && rsi > rsiOverbought)
         {
            executionManager.Terminate();
            inPosition = false;
            Print("Exit signal detected. Terminating VWAP execution.");
         }
         else if(!isLong && rsi < rsiOversold)
         {
            executionManager.Terminate();
            inPosition = false;
            Print("Exit signal detected. Terminating VWAP execution.");
         }
      }
}
```

3. Breakout Strategy with Iceberg Orders.



```
//+------------------------------------------------------------------+
//| Breakout Strategy with Iceberg Orders                            |
//+------------------------------------------------------------------+
void OnTick()
{
      // Strategy parameters
      int channelPeriod = 20;
      double volume = 1.0;
      double visibleVolume = 0.1;

      // Calculate indicators
      double upperChannel = iHigh(Symbol(), PERIOD_CURRENT, iHighest(Symbol(), PERIOD_CURRENT, MODE_HIGH, channelPeriod, 1));
      double lowerChannel = iLow(Symbol(), PERIOD_CURRENT, iLowest(Symbol(), PERIOD_CURRENT, MODE_LOW, channelPeriod, 1));

      double currentPrice = SymbolInfoDouble(Symbol(), SYMBOL_BID);

      // Check for entry conditions
      static bool inPosition = false;
      static CExecutionManager executionManager;

      if(!inPosition)
      {
         // Buy signal: Price breaks above upper channel
         if(currentPrice > upperChannel)
         {
            // Create Iceberg Order
            double limitPrice = upperChannel; // Place limit order at breakout level

            if(executionManager.CreateIcebergOrder(Symbol(), volume, limitPrice, ORDER_TYPE_BUY,
                                                 visibleVolume, visibleVolume * 0.8, visibleVolume * 1.2,
                                                 true, 1000, true, 2, 3))
            {
               executionManager.Initialize();
               inPosition = true;
               Print("Buy breakout detected. Starting Iceberg Order execution.");
            }
         }
         // Sell signal: Price breaks below lower channel
         else if(currentPrice < lowerChannel)
         {
            // Create Iceberg Order
            double limitPrice = lowerChannel; // Place limit order at breakout level

            if(executionManager.CreateIcebergOrder(Symbol(), volume, limitPrice, ORDER_TYPE_SELL,
                                                 visibleVolume, visibleVolume * 0.8, visibleVolume * 1.2,
                                                 true, 1000, true, 2, 3))
            {
               executionManager.Initialize();
               inPosition = true;
               Print("Sell breakout detected. Starting Iceberg Order execution.");
            }
         }
      }
      else
      {
         // Update the execution algorithm
         if(executionManager.Update())
         {
            // Check if execution is complete
            if(!executionManager.GetAlgorithm().IsActive())
            {
               inPosition = false;
               Print("Iceberg Order execution completed.");
            }
         }
      }
}
```


Across all three examples, the integration follows the same high-level pattern:

1. Maintain state

   - A boolean inPosition tracks whether you’re currently filling an order.
   - A static CExecutionManager executionManager lives across ticks to manage your chosen algorithm’s lifecycle.
2. Entry logic

   - On your entry signal (MA crossover, RSI threshold, channel break), call the appropriate creation method on executionManager (TWAP, VWAP or Iceberg), passing symbol, total volume, time window or limit price, slice parameters, and order type.
   - If creation succeeds, immediately call executionManager.Initialize(), set inPosition=true, and log your start.
3. Ongoing execution

   - While inPosition is true, every OnTick() invoke executionManager.Update().
   - Inside Update(), the manager will internally call Execute() as needed, poll fills, handle timeouts or market updates, and schedule the next slice (or cancel/replace child orders for Iceberg).
4. Completion & exit

   - After each update, check executionManager.GetAlgorithm()->IsActive(). Once it returns false (all intervals done or volume exhausted), set inPosition=false and log that execution has completed.
   - In the VWAP mean-reversion example, there’s an extra exit check: if price reverses beyond your RSI threshold mid-execution, you call executionManager.Terminate() to stop early.

Rules for each are as follows:

1. Trend-Following + TWAP

Entry: Fast MA crossing above slow MA triggers

Execution: Splits your 1 lot buy into 5 equal slices over the next hour

2. Mean-Reversion + VWAP

Entry: RSI < 30 for buys, > 70 for sells

Execution: Distributes 1 lot against historical volume over 5 slices in 1 hour

Early Exit: If the signal flips (e.g. RSI > 70 during a buy), executionManager.Terminate() aborts remaining slices

3. Breakout + Iceberg

Entry: Price breaks channel high (buy) or low (sell)

Execution: Places a pending limit at the breakout price, revealing only ~0.1 lots at a time and refilling until the full 1 lot is done

By swapping out CreateTWAP, CreateVWAP or CreateIcebergOrder, you plug any execution algorithm into your signal logic without duplicating order-management boilerplate.

### Integrated Strategy

Full Code:

```
//+------------------------------------------------------------------+
//|                                  IntegratedStrategy.mq5          |
//|                        Copyright 2025, MetaQuotes Software Corp. |
//|                                    https://www.metaquotes.net    |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Software Corp."
#property link      "https://www.metaquotes.net"
#property version   "1.00"

#include "ExecutionAlgorithm.mqh"
#include "TWAP.mqh"
#include "VWAP.mqh"
#include "IcebergOrder.mqh"
#include "PerformanceAnalyzer.mqh"
#include "ExecutionManager.mqh"

// Input parameters
input int FastMA = 20;                  // Fast moving average period
input int SlowMA = 50;                  // Slow moving average period
input double TradingVolume = 0.1;       // Trading volume
input bool UseAdaptiveExecution = true; // Use adaptive execution based on market conditions

// Global variables
CExecutionManager *g_executionManager = NULL;
int g_maHandle1 = INVALID_HANDLE;
int g_maHandle2 = INVALID_HANDLE;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Initialize execution manager
   g_executionManager = new CExecutionManager(Symbol(), 3, UseAdaptiveExecution);

   // Initialize indicators
   g_maHandle1 = iMA(Symbol(), Period(), FastMA, 0, MODE_SMA, PRICE_CLOSE);
   g_maHandle2 = iMA(Symbol(), Period(), SlowMA, 0, MODE_SMA, PRICE_CLOSE);

   if(g_maHandle1 == INVALID_HANDLE || g_maHandle2 == INVALID_HANDLE)
   {
      Print("Failed to create indicator handles");
      return INIT_FAILED;
   }

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Clean up
   if(g_executionManager != NULL)
   {
      delete g_executionManager;
      g_executionManager = NULL;
   }

   // Release indicator handles
   IndicatorRelease(g_maHandle1);
   IndicatorRelease(g_maHandle2);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Update execution algorithms
   if(g_executionManager != NULL)
      g_executionManager.UpdateAlgorithms();

   // Only process at bar open
   if(iVolume(_Symbol, PERIOD_CURRENT, 0) > 1)
      return;

   // Get indicator values
   double fastMA[2], slowMA[2];

   if(CopyBuffer(g_maHandle1, 0, 0, 2, fastMA) <= 0 ||
      CopyBuffer(g_maHandle2, 0, 0, 2, slowMA) <= 0)
   {
      Print("Failed to copy indicator buffers");
      return;
   }

   // Check for trend signals
   bool buySignal = (fastMA[0] > slowMA[0]) && (fastMA[1] <= slowMA[1]);
   bool sellSignal = (fastMA[0] < slowMA[0]) && (fastMA[1] >= slowMA[1]);

   // Execute signals using the execution manager
   if(buySignal)
   {
      Print("Buy signal detected");

      if(UseAdaptiveExecution)
      {
         // Let the execution manager select the best algorithm
         g_executionManager.ExecuteSignal(SIGNAL_TYPE_BUY, TradingVolume);
      }
      else
      {
         // Manually create a TWAP algorithm
         datetime currentTime = TimeCurrent();
         CTWAP *twap = g_executionManager.CreateTWAP(TradingVolume, currentTime,
                                                   currentTime + 3600, 6,
                                                   ORDER_TYPE_BUY, true);
      }
   }
   else if(sellSignal)
   {
      Print("Sell signal detected");

      if(UseAdaptiveExecution)
      {
         // Let the execution manager select the best algorithm
         g_executionManager.ExecuteSignal(SIGNAL_TYPE_SELL, TradingVolume);
      }
      else
      {
         // Manually create a TWAP algorithm
         datetime currentTime = TimeCurrent();
         CTWAP *twap = g_executionManager.CreateTWAP(TradingVolume, currentTime,
                                                   currentTime + 3600, 6,
                                                   ORDER_TYPE_SELL, true);
      }
   }
}
//+------------------------------------------------------------------+
```

The IntegratedStrategy.mq5 expert advisor begins by declaring its metadata (copyright, link, version) and including the headers for all of our execution-algorithm classes and the performance analyzer. It then defines four user-adjustable inputs: the fast and slow SMA periods, the total trading volume per signal, and a boolean flag to toggle “adaptive” execution (where the manager decides whether to use TWAP, VWAP or Iceberg under the hood). A global pointer to CExecutionManager and two indicator handles are also declared so they persist between ticks.

In OnInit(), we instantiate the execution manager—passing it the current symbol, a maximum of three concurrent algorithms, and our adaptive flag—and then create the two SMA indicator handles. If either handle fails, initialization aborts. OnDeinit() simply cleans up by deleting the manager and releasing the indicator handles, ensuring no memory or handle leaks when the EA is removed or the platform shuts down.

The core logic lives in OnTick(). First, we call UpdateAlgorithms() on the execution manager so that any existing child orders (TWAP slices, VWAP buckets or Iceberg legs) get processed, cancelled or refilled as needed. Then we wait for a new bar (by skipping if tick volume is still building). Once at bar-open, we pull the last two SMA values for both fast and slow periods. A crossover from below to above triggers a buy signal; the reverse triggers a sell.

If adaptive execution is enabled, we hand off the signal and volume to g\_executionManager.ExecuteSignal(), letting it pick the appropriate algorithm. Otherwise, we manually spin up a TWAP instance for a one-hour window and six slices. This pattern cleanly separates your entry logic (trend detection) from your order-management logic, letting the same facade drive multiple execution styles without duplicating boilerplate.

After incorporation Take Profit, the code changes to:

```
//+------------------------------------------------------------------+
//|                                  IntegratedStrategy.mq5          |
//|                        Copyright 2025, MetaQuotes Software Corp. |
//|                                    https://www.metaquotes.net    |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Software Corp."
#property link      "https://www.metaquotes.net"
#property version   "1.00"

#include "ExecutionAlgorithm.mqh"
#include "TWAP.mqh"
#include "VWAP.mqh"
#include "IcebergOrder.mqh"
#include "PerformanceAnalyzer.mqh"
#include "ExecutionManager.mqh"
#include <Trade\Trade.mqh>

// Input parameters
input int    FastMA              = 20;    // Fast moving average period
input int    SlowMA              = 50;    // Slow moving average period
input double TradingVolume       = 0.1;   // Trading volume
input bool   UseAdaptiveExecution = true; // Use adaptive execution based on market conditions
input double EquityTPPercent     = 10.0;  // Equity Take Profit in percent
input double EquitySLPercent     = 5.0;   // Equity Stop Loss in percent

// Global variables
CExecutionManager *g_executionManager = NULL;
int                g_maHandle1       = INVALID_HANDLE;
int                g_maHandle2       = INVALID_HANDLE;
double             g_initialEquity   = 0.0;
CTrade             trade;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
    // Record initial equity
    g_initialEquity = AccountInfoDouble(ACCOUNT_EQUITY);

    // Initialize execution manager
    g_executionManager = new CExecutionManager(Symbol(), 3, UseAdaptiveExecution);

    // Initialize indicators
    g_maHandle1 = iMA(Symbol(), Period(), FastMA, 0, MODE_SMA, PRICE_CLOSE);
    g_maHandle2 = iMA(Symbol(), Period(), SlowMA, 0, MODE_SMA, PRICE_CLOSE);

    if(g_maHandle1 == INVALID_HANDLE || g_maHandle2 == INVALID_HANDLE) {
        Print("Failed to create indicator handles");
        return INIT_FAILED;
    }

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    // Clean up
    if(g_executionManager != NULL) {
        delete g_executionManager;
        g_executionManager = NULL;
    }

    // Release indicator handles
    IndicatorRelease(g_maHandle1);
    IndicatorRelease(g_maHandle2);
}

//+------------------------------------------------------------------+
//| Check equity-based TP and SL, then reset baseline                |
//+------------------------------------------------------------------+
void CheckEquityTPandSL() {
    double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
    double tpEquity      = g_initialEquity * (1.0 + EquityTPPercent / 100.0);
    double slEquity      = g_initialEquity * (1.0 - EquitySLPercent / 100.0);

    if(currentEquity >= tpEquity) {
        Print("Equity Take Profit reached: ", currentEquity);
        CloseAllPositions();
        g_initialEquity = currentEquity;
        Print("Equity baseline reset to: ", g_initialEquity);
    } else if(currentEquity <= slEquity) {
        Print("Equity Stop Loss reached: ", currentEquity);
        CloseAllPositions();
        g_initialEquity = currentEquity;
        Print("Equity baseline reset to: ", g_initialEquity);
    }
}

//+------------------------------------------------------------------+
//| Close all open positions                                         |
//+------------------------------------------------------------------+
void CloseAllPositions() {
    CPositionInfo  m_position;
    CTrade m_trade;
    for(int i = PositionsTotal() - 1; i >= 0; i--) // loop all Open Positions
        if(m_position.SelectByIndex(i)) { // select a position
            m_trade.PositionClose(m_position.Ticket()); // then delete it --period
        }
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
    // Check and reset equity thresholds
    CheckEquityTPandSL();

    // Update execution algorithms
    if(g_executionManager != NULL)
        g_executionManager.UpdateAlgorithms();

    // Only process at bar open
    if(iVolume(_Symbol, PERIOD_CURRENT, 0) > 1)
        return;

    // Get indicator values
    double fastMA[2], slowMA[2];

    if(CopyBuffer(g_maHandle1, 0, 0, 2, fastMA) <= 0 ||
            CopyBuffer(g_maHandle2, 0, 0, 2, slowMA) <= 0) {
        Print("Failed to copy indicator buffers");
        return;
    }

    // Check for trend signals
    bool buySignal  = (fastMA[0] > slowMA[0]) && (fastMA[1] <= slowMA[1]);
    bool sellSignal = (fastMA[0] < slowMA[0]) && (fastMA[1] >= slowMA[1]);

    // Execute signals using the execution manager
    if(buySignal) {
        Print("Buy signal detected");

        if(UseAdaptiveExecution) {
            // Let the execution manager select the best algorithm
            g_executionManager.ExecuteSignal(SIGNAL_TYPE_BUY, TradingVolume);
        } else {
            datetime currentTime = TimeCurrent();
            CTWAP *twap = g_executionManager.CreateTWAP(TradingVolume, currentTime,
                          currentTime + 3600, 6,
                          ORDER_TYPE_BUY, true);
        }
    } else if(sellSignal) {
        Print("Sell signal detected");

        if(UseAdaptiveExecution) {
            // Let the execution manager select the best algorithm
            g_executionManager.ExecuteSignal(SIGNAL_TYPE_SELL, TradingVolume);
        } else {
            datetime currentTime = TimeCurrent();
            CTWAP *twap = g_executionManager.CreateTWAP(TradingVolume, currentTime,
                          currentTime + 3600, 6,
                          ORDER_TYPE_SELL, true);
        }
    }
}
//+------------------------------------------------------------------+
```

### Backtest Results

![](https://c.mql5.com/2/137/5695579647180.png)

![](https://c.mql5.com/2/137/6184783865555.png)

1\. Equity & Balance Curves

The green “Balance” stair-steps show your account’s book equity whenever the EA closes a position; the blue “Equity” line smooths in unrealized P&L between trades. We can see a clear upward trend from January through early March, with a few pullbacks—each drawdown topping out around 10–16% before your next series of wins restores the gain. That pattern suggests the system thrives in trending conditions but still suffers tolerable equity dips.

2\. Volume & Risk Utilization

At the bottom, the “Deposit Load” triangles shrink gradually over time—this is your position size as a percentage of equity. It starts near 10% of your balance and tapers as your equity grows (with fixed-volume sizing), meaning ourrisk per trade actually decreases as the account climbs. That’s why drawdowns stay proportionally similar even as your dollar equity rises.

3\. Key Profitability Metrics

- Initial deposit: $1,000

- Net profit: + $703 (a 70% return over ~2 months)

- Profit factor: 2.34 (you make $2.34 for every $1 lost)

- Expected payoff: $2.34 per trade on average

- Sharpe ratio: 5.47 (very high—strong risk-adjusted returns)


These figures tell us the strategy is not only profitable but earns a healthy buffer above its own volatility.

4\. Drawdown & Recovery

- Max balance drawdown: 156 points or 9.99%

- Max equity drawdown: 228 points or 15.89%

- Recovery factor: 3.08 (net profit ÷ max drawdown)


A recovery factor above 2 is generally considered good, so at 3.08 you’re generating over three times your worst loss in gain.

5\. Trade Distribution

- Total trades: 300 (600 deals, so every entry+exit counts as two)

- Win rate: 76% (228 winners vs. 72 losers)

- Average win: $5.39

- Average loss: – $7.31


Although your win rate and profit factor are strong, notice that your losers are on average larger than winners—something to watch if market conditions flip.

6\. Streaks & Consistency

- Max consecutive wins: 87 trades, + $302

- Max consecutive losses: 23 trades, – $156

- Average winning streak: 57 trades

- Average losing streak: 18 trades


Long winning streaks drive the upward slope, while the longest losing run still only costs about 15% of equity.

### Conclusion

Imagine you’re a solo trader in a crowded market arena—every tick matters, every fill price whispers profit or loss. By weaving TWAP, VWAP and Iceberg Orders into your toolkit, you’re no longer just reacting to price swings; you’re orchestrating them. These once-elite, institutional-grade algorithms are now at your fingertips, slicing through liquidity like a laser and turning chaotic order books into opportunities.

TWAP becomes your steady metronome, pacing your size evenly across a set interval—perfect for when the tide is calm and you simply want a smooth ride. VWAP morphs you into a savvy volume-tracker, attacking the heaviest trading beats of the day and riding the market’s own pulse. And when you need to cloak your intentions, Iceberg Orders slip your true size beneath the surface, revealing just enough to get filled without spooking the big players.

But these aren’t just standalone tricks. With our modular MQL5 framework, you plug them into any strategy—trend followers, mean-reverters, breakout hunters—with the ease of snapping on a new lens. A single ExecutionManager façade lets you swap, combine or even layer algorithms mid-trade, while the PerformanceAnalyzer keeps score like a hawk, measuring slippage, shortfall and market impact down to the last pip.

What’s next? Think of execution as a living creature that adapts. Let your TWAP learn from volatility spikes. Route your VWAP slices to the deepest pools. Teach your Iceberg to sense where predators lurk and hide deeper. And why stop there? Inject machine-learning to predict the perfect microsecond to fire, or blend order types into bespoke hybrids that match your unique edge.

The trading world never stands still—and neither should your order execution. Dive in, experiment boldly, and turn every slice, every fill, into a calculated advantage. Your edge awaits in the code.

For your convenience, here's a summary of the files included with this article:

| File Name | Description |
| --- | --- |
| ExecutionAlgorithm.mqh | Base class for all execution algorithms |
| TWAP.mqh | Time-Weighted Average Price implementation |
| VWAP.mqh | Volume-Weighted Average Price implementation |
| IcebergOrder.mqh | Iceberg Order implementation |
| PerformanceAnalyzer.mqh | Tools for analyzing execution performance |
| ExecutionManager.mqh | Facade for easy integration with trading strategies |
| IntegratedStrategy.mq5 | Example EA showing integration with a trading strategy |
| IntegratedStrategy - Take Profit.mq5 | Example EA showing integration with a trading strategy with take profit and stop loss in percentage on the account balance |

By incorporating these advanced execution algorithms into your trading toolkit, you're taking a significant step toward more professional and efficient trading. Whether you're looking to minimize the impact of larger trades, improve your average execution prices, or simply add more sophistication to your trading approach, these algorithms provide valuable solutions that can enhance your trading performance in today's competitive markets.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17934.zip "Download all attachments in the single ZIP archive")

[ExecutionAlgorithm.mqh](https://www.mql5.com/en/articles/download/17934/executionalgorithm.mqh "Download ExecutionAlgorithm.mqh")(7.86 KB)

[TWAP.mqh](https://www.mql5.com/en/articles/download/17934/twap.mqh "Download TWAP.mqh")(9.35 KB)

[VWAP.mqh](https://www.mql5.com/en/articles/download/17934/vwap.mqh "Download VWAP.mqh")(15.91 KB)

[IcebergOrder.mqh](https://www.mql5.com/en/articles/download/17934/icebergorder.mqh "Download IcebergOrder.mqh")(15.33 KB)

[PerformanceAnalyzer.mqh](https://www.mql5.com/en/articles/download/17934/performanceanalyzer.mqh "Download PerformanceAnalyzer.mqh")(15.98 KB)

[ExecutionManager.mqh](https://www.mql5.com/en/articles/download/17934/executionmanager.mqh "Download ExecutionManager.mqh")(17.76 KB)

[IntegratedStrategy.mq5](https://www.mql5.com/en/articles/download/17934/integratedstrategy.mq5 "Download IntegratedStrategy.mq5")(4.04 KB)

[IntegratedStrategy\_-\_Take\_Profit.mq5](https://www.mql5.com/en/articles/download/17934/integratedstrategy_-_take_profit.mq5 "Download IntegratedStrategy_-_Take_Profit.mq5")(5.76 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/486550)**
(6)


![i_vergo](https://c.mql5.com/avatar/avatar_na2.png)

**[i\_vergo](https://www.mql5.com/en/users/i_vergo)**
\|
16 May 2025 at 07:44

Great Article

Testing your algo.

IN file, ExecutionAlgorithm.mqh, added this line    request.type\_filling = ORDER\_FILLING\_IOC; in placing order to fix order placing issue.

Back tested it at M5, it opened only 1 trade for 2 months period, no partial order opened.

Tested it at H1, it never applied the SL, or TP and all trades closed in loss.

also while compiling it generates warnings

[possible loss of data](https://www.mql5.com/en/docs/basis/types/casting "MQL5 Documentation: Typecasting") due to type conversion from 'long' to 'double'VWAP.mqh27141

possible loss of data due to type conversion from 'long' to 'double'VWAP.mqh27222

possible loss of data due to type conversion from 'long' to 'double'VWAP.mqh44917

possible loss of data due to type conversion from 'long' to 'double'PerformanceAnalyzer.mqh22217

possible loss of data due to type conversion from 'long' to 'double'ExecutionManager.mqh41817

suggest how tyo test the algo,

Time frame. and any other recommendations.

![i_vergo](https://c.mql5.com/avatar/avatar_na2.png)

**[i\_vergo](https://www.mql5.com/en/users/i_vergo)**
\|
16 May 2025 at 07:57

to fix the warnins

also while compiling it generates warnings

[possible loss of data](https://www.mql5.com/en/docs/basis/types/casting "MQL5 Documentation: Typecasting") due to type conversion from 'long' to 'double'VWAP.mqh27141

possible loss of data due to type conversion from 'long' to 'double'VWAP.mqh27222

possible loss of data due to type conversion from 'long' to 'double'VWAP.mqh44917

possible loss of data due to type conversion from 'long' to 'double'PerformanceAnalyzer.mqh22217

possible loss of data due to type conversion from 'long' to 'double'ExecutionManager.mqh41817

i changed the code line

m\_volumeProfile\[intervalIndex\] += rates\[i\].tick\_volu

to

m\_volumeProfile\[intervalIndex\] += (double)rates\[i\].tick\_volume;

It fixed the warnings

Now need you guidence regarding my other queries, as

Time frame

And also

why all trades during backtest result in Loss

how to test this great work from you..

![Dominic Michael Frehner](https://c.mql5.com/avatar/2024/11/672504f5-a016.jpg)

**[Dominic Michael Frehner](https://www.mql5.com/en/users/cryptonist)**
\|
16 May 2025 at 07:57

**i\_vergo [#](https://www.mql5.com/en/forum/486550#comment_56714850):**

Great Article

Testing your algo.

IN file, ExecutionAlgorithm.mqh, added this line    request.type\_filling = ORDER\_FILLING\_IOC; in placing order to fix order placing issue.

Back tested it at M5, it opened only 1 trade for 2 months period, no partial order opened.

Tested it at H1, it never applied the SL, or TP and all trades closed in loss.

also while compiling it generates warnings

[possible loss of data](https://www.mql5.com/en/docs/basis/types/casting "MQL5 Documentation: Typecasting") due to type conversion from 'long' to 'double'VWAP.mqh27141

possible loss of data due to type conversion from 'long' to 'double'VWAP.mqh27222

possible loss of data due to type conversion from 'long' to 'double'VWAP.mqh44917

possible loss of data due to type conversion from 'long' to 'double'PerformanceAnalyzer.mqh22217

possible loss of data due to type conversion from 'long' to 'double'ExecutionManager.mqh41817

suggest how tyo test the algo,

Time frame. and any other recommendations.

The warnings aren't the problem but could get fixed quickly. But yes it would be great if the author could show step-by-step which settings and inputs he used for the backtest.

![CapeCoddah](https://c.mql5.com/avatar/avatar_na2.png)

**[CapeCoddah](https://www.mql5.com/en/users/capecoddah)**
\|
16 May 2025 at 12:01

I agree with Dominic as the warnings are just warnings.  I\_Virgo's results are probably because he used the wrong time frame and currency pair.  From the Back Test report, of nearly 2000 bars, it must have been either M1 or M5 as the time frame with an unknown pair.

It would be nice if MQ added the time frame and currency pair or pairs  and also separated the pair results in more details to the Back Test report so we could more closely replicate author's back [test results](https://www.mql5.com/en/docs/common/TesterStatistics "MQL5 Documentation: TesterStatistics function") as well as determining its applicability over the forex pairs..  Also, it would be extremely helpful if the EA could post text to the chart while it is running.

I also think it is a great article and plan to study it thoroughly in anticipation of adapting his techniques to other EAs

CapeCoddah

![Shashin Wijewardhane](https://c.mql5.com/avatar/2025/8/68992d37-7848.png)

**[Shashin Wijewardhane](https://www.mql5.com/en/users/shashin1024)**
\|
11 Jul 2025 at 08:40

```
//+------------------------------------------------------------------+
//| Base class for all execution algorithms                           |
//+------------------------------------------------------------------+
class CExecutionAlgorithm
{
protected:
   string            m_symbol;           // Trading symbol
   double            m_totalVolume;      // Total volume to execute
   double            m_executedVolume;   // Volume already executed
   double            m_remainingVolume;  // Volume remaining to execute
   datetime          m_startTime;        // Start time of execution
   datetime          m_endTime;          // End time of execution
   int               m_slippage;         // Allowed slippage in points
   bool              m_isActive;         // Is algorithm currently active

   // Statistics
   double            m_avgExecutionPrice; // Average execution price
   int               m_totalOrders;       // Total number of orders placed
   int               m_filledOrders;      // Number of filled orders

public:
   // Constructor
   CExecutionAlgorithm(string symbol, double volume, datetime startTime, datetime endTime, int slippage);

   // Destructor
   virtual ~CExecutionAlgorithm();

   // Virtual methods to be implemented by derived classes
   virtual bool      Initialize();
   virtual bool      Execute() = 0;
   virtual bool      Update() = 0;
   virtual bool      Terminate() = 0;

   // Common methods
   bool              IsActive() { return m_isActive; }
   double            GetExecutedVolume() { return m_executedVolume; }
   double            GetRemainingVolume() { return m_remainingVolume; }
   double            GetAverageExecutionPrice() { return m_avgExecutionPrice; }

   // Helper methods
   bool              PlaceOrder(ENUM_ORDER_TYPE orderType, double volume, double price = 0.0);
   bool              ModifyOrder(ulong ticket, double price, double sl, double tp);
   bool              CancelOrder(ulong ticket);
   void              UpdateAverageExecutionPrice(double price, double volume);

   // Helper method to get appropriate filling mode
   ENUM_ORDER_TYPE_FILLING GetFillingMode();
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CExecutionAlgorithm::CExecutionAlgorithm(string symbol, double volume,
                                       datetime startTime, datetime endTime,
                                       int slippage)
{
   m_symbol = symbol;
   m_totalVolume = volume;
   m_executedVolume = 0.0;
   m_remainingVolume = volume;
   m_startTime = startTime;
   m_endTime = endTime;
   m_slippage = slippage;
   m_isActive = false;

   m_avgExecutionPrice = 0.0;
   m_totalOrders = 0;
   m_filledOrders = 0;
}

//+------------------------------------------------------------------+
//| Destructor                                                        |
//+------------------------------------------------------------------+
CExecutionAlgorithm::~CExecutionAlgorithm()
{
   // Clean up resources if needed
}

//+------------------------------------------------------------------+
//| Initialize the algorithm                                          |
//+------------------------------------------------------------------+
bool CExecutionAlgorithm::Initialize()
{
   // Validate inputs
   if(m_symbol == "" || m_totalVolume <= 0.0)
   {
      Print("Invalid inputs for execution algorithm");
      return false;
   }

   // Check if the symbol exists
   if(!SymbolSelect(m_symbol, true))
   {
      Print("Symbol not found: ", m_symbol);
      return false;
   }

   // Reset statistics
   m_executedVolume = 0.0;
   m_remainingVolume = m_totalVolume;
   m_avgExecutionPrice = 0.0;
   m_totalOrders = 0;
   m_filledOrders = 0;

   return true;
}

//+------------------------------------------------------------------+
//| Get appropriate filling mode for the symbol                      |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING CExecutionAlgorithm::GetFillingMode()
{
   // Get symbol filling modes
   int filling_modes = (int)SymbolInfoInteger(m_symbol, SYMBOL_FILLING_MODE);

   // Check available filling modes in order of preference
   if((filling_modes & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)
      return ORDER_FILLING_FOK;
   else if((filling_modes & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC)
      return ORDER_FILLING_IOC;
   else
      return ORDER_FILLING_RETURN;
}

//+------------------------------------------------------------------+
//| Place an order                                                    |
//+------------------------------------------------------------------+
bool CExecutionAlgorithm::PlaceOrder(ENUM_ORDER_TYPE orderType, double volume, double price = 0.0)
{
   // Validate inputs
   if(volume <= 0.0)
   {
      Print("Invalid order volume");
      return false;
   }

   // Prepare the request
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);

   request.symbol = m_symbol;
   request.volume = volume;
   request.type = orderType;
   request.deviation = m_slippage;
   request.magic = 123456; // Magic number for identification

   // Set appropriate action and price based on order type
   if(orderType == ORDER_TYPE_BUY || orderType == ORDER_TYPE_SELL)
   {
      // Market order
      request.action = TRADE_ACTION_DEAL;
      request.type_filling = GetFillingMode();

      if(orderType == ORDER_TYPE_BUY)
         request.price = SymbolInfoDouble(m_symbol, SYMBOL_ASK);
      else
         request.price = SymbolInfoDouble(m_symbol, SYMBOL_BID);
   }
   else
   {
      // Pending order
      request.action = TRADE_ACTION_PENDING;
      if(price <= 0.0)
      {
         Print("Price must be specified for pending orders");
         return false;
      }
      request.price = price;
   }

   // Send the order
   if(!OrderSend(request, result))
   {
      Print("OrderSend error: ", GetLastError());
      return false;
   }

   // Check the result
   if(result.retcode != TRADE_RETCODE_DONE)
   {
      Print("OrderSend failed with code: ", result.retcode, " - ", result.comment);
      return false;
   }

   // Update statistics
   m_totalOrders++;

   // For market orders, update execution statistics immediately
   if(orderType == ORDER_TYPE_BUY || orderType == ORDER_TYPE_SELL)
   {
      m_filledOrders++;
      UpdateAverageExecutionPrice(request.price, volume);
      m_executedVolume += volume;
      m_remainingVolume -= volume;
   }

   Print("Order placed successfully. Ticket: ", result.order, " Volume: ", volume, " Price: ", request.price);

   return true;
}

//+------------------------------------------------------------------+
//| Modify an existing order                                          |
//+------------------------------------------------------------------+
bool CExecutionAlgorithm::ModifyOrder(ulong ticket, double price, double sl, double tp)
{
   // Prepare the request
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);

   request.action = TRADE_ACTION_MODIFY;
   request.order = ticket;
   request.price = price;
   request.sl = sl;
   request.tp = tp;

   // Send the modification request
   if(!OrderSend(request, result))
   {
      Print("OrderModify error: ", GetLastError());
      return false;
   }

   // Check the result
   if(result.retcode != TRADE_RETCODE_DONE)
   {
      Print("OrderModify failed with code: ", result.retcode, " - ", result.comment);
      return false;
   }

   Print("Order modified successfully. Ticket: ", ticket);

   return true;
}

//+------------------------------------------------------------------+
//| Cancel an existing order                                          |
//+------------------------------------------------------------------+
bool CExecutionAlgorithm::CancelOrder(ulong ticket)
{
   // Prepare the request
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);

   request.action = TRADE_ACTION_REMOVE;
   request.order = ticket;

   // Send the cancellation request
   if(!OrderSend(request, result))
   {
      Print("OrderCancel error: ", GetLastError());
      return false;
   }

   // Check the result
   if(result.retcode != TRADE_RETCODE_DONE)
   {
      Print("OrderCancel failed with code: ", result.retcode, " - ", result.comment);
      return false;
   }

   Print("Order cancelled successfully. Ticket: ", ticket);

   return true;
}

//+------------------------------------------------------------------+
//| Update the average execution price                                |
//+------------------------------------------------------------------+
void CExecutionAlgorithm::UpdateAverageExecutionPrice(double price, double volume)
{
   // Calculate the new average execution price
   if(m_executedVolume > 0.0)
   {
      // Weighted average of old and new prices
      m_avgExecutionPrice = (m_avgExecutionPrice * m_executedVolume + price * volume) /
                           (m_executedVolume + volume);
   }
   else
   {
      // First execution
      m_avgExecutionPrice = price;
   }
}
//+------------------------------------------------------------------+
```

![Neural Networks in Trading: Generalized 3D Referring Expression Segmentation](https://c.mql5.com/2/96/Neural_Networks_in_Trading_Data_Segmentation_Based_on_Refinement_Expressions__LOGO2.png)[Neural Networks in Trading: Generalized 3D Referring Expression Segmentation](https://www.mql5.com/en/articles/15997)

While analyzing the market situation, we divide it into separate segments, identifying key trends. However, traditional analysis methods often focus on one aspect and thus limit the proper perception. In this article, we will learn about a method that enables the selection of multiple objects to ensure a more comprehensive and multi-layered understanding of the situation.

![From Basic to Intermediate: Arrays and Strings (III)](https://c.mql5.com/2/96/Do_bhsico_ao_intermedixrio_Array_e_String_III__LOGO.png)[From Basic to Intermediate: Arrays and Strings (III)](https://www.mql5.com/en/articles/15461)

This article considers two aspects. First, how the standard library can convert binary values to other representations such as octal, decimal, and hexadecimal. Second, we will talk about how we can determine the width of our password based on the secret phrase, using the knowledge we have already acquired.

![Data Science and ML (Part 40): Using Fibonacci Retracements in Machine Learning data](https://c.mql5.com/2/143/18078-data-science-and-ml-part-40-logo.png)[Data Science and ML (Part 40): Using Fibonacci Retracements in Machine Learning data](https://www.mql5.com/en/articles/18078)

Fibonacci retracements are a popular tool in technical analysis, helping traders identify potential reversal zones. In this article, we’ll explore how these retracement levels can be transformed into target variables for machine learning models to help them understand the market better using this powerful tool.

![Developing a Replay System (Part 68): Getting the Time Right (I)](https://c.mql5.com/2/96/Desenvolvendo_um_sistema_de_Replay_Parte_68___LOGO.png)[Developing a Replay System (Part 68): Getting the Time Right (I)](https://www.mql5.com/en/articles/12309)

Today we will continue working on getting the mouse pointer to tell us how much time is left on a bar during periods of low liquidity. Although at first glance it seems simple, in reality this task is much more difficult. This involves some obstacles that we will have to overcome. Therefore, it is important that you have a good understanding of the material in this first part of this subseries in order to understand the following parts.

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/17934&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062578162321237219)

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