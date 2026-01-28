---
title: Developing a multi-currency Expert Advisor (Part 4): Pending virtual orders and saving status
url: https://www.mql5.com/en/articles/14246
categories: Trading Systems, Integration, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:37:52.836479
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/14246&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049264455193307197)

MetaTrader 5 / Trading systems


### Introduction

In the previous [article](https://www.mql5.com/en/articles/14148), we have significantly revised the code architecture to build a multi-currency EA with several parallel working strategies. Trying to achieve simplicity and clarity, we have so far only considered a certain minimum set of functionality. Even considering the limitations of our task, we have significantly altered the code from the previous [articles](https://www.mql5.com/en/articles/14107).

Now hopefully we have the groundwork that is sufficient enough to increase functionality without radical changes to the already written code. We will try to make a minimum number of edits only where it is really necessary.

As a further development, we will try to do the following:

- add the ability to open virtual pending orders (Buy Stop, Sell Stop, Buy Limit, Sell Limit), and not just virtual positions (Buy, Sell);
- add a simple way to visualize placed virtual orders and positions so that we can visually control when testing the correct implementation of the rules for opening positions/orders in the trading strategies used;
- implement saving current status data by the EA, so that when the terminal is restarted or the EA is moved to another terminal, it can continue working from the state it found itself in at the moment its work was interrupted.

Let's start with the simplest thing - handling virtual pending orders.

### Virtual pending orders

We have created the _CVirtualOrder_ class to handle virtual positions. We can create a separate similar class to handle virtual pending orders. Let's see if handling positions is really that different from handling orders.

Their set of properties largely coincides. However, pending orders have a property added that stores the expiration time. Because of this, they add a new reason for closing - upon reaching the expiration time. Accordingly, we will need to check at each tick whether the open virtual pending order has been closed for this reason, and when the Stop Loss and Take Profit levels are reached, it should not be closed.

When the price reaches the trigger level of a virtual pending order, it should turn into an open virtual position. Here it already becomes clear that when implemented in the form of different classes, we will have to do additional work to create an open virtual position and delete a triggered virtual pending order. If we implement them as one class, then we only have to change one property of the class object - its type. Therefore, we will implement exactly this option.

How else will an order differ from a position? We will also need to indicate the opening price of a virtual pending order. For a virtual position, specifying the opening price was unnecessary, since it was automatically determined from current market data. Now we will make it a required parameter of the opening function.

Let's start adding to the class. We will add:

- the _m\_expiration_ property for storing the expiration time;
- the _m\_isExpired_ logical property to indicate the pending order expiration;
- the _CheckTrigger()_ method to check whether a pending order is triggered, several methods for checking whether a given object belongs to a certain type (pending order, limit pending order, etc.);
- condition stating that 'true' is also returned in cases where it is a pending order in the desired direction, regardless of whether it is a limit order or a stop order - to the methods for checking whether a given object is in the BUY or SELL direction.
- the list of _Open()_ method parameters receives the open price and expiration time.

```
//+------------------------------------------------------------------+
//| Class of virtual orders and positions                            |
//+------------------------------------------------------------------+
class CVirtualOrder {
private:
   ...
//--- Order (position) properties
   ...
   datetime          m_expiration;     // Expiration time
   ...
   bool              m_isExpired;      // Expiration flag
   ...
//--- Private methods
   ...
   bool              CheckTrigger();   // Check for pending order trigger

public:
    ...

//--- Methods for checking the position (order) status
   ...
   bool              IsPendingOrder() {// Is it a pending order?
      return IsOpen() && (m_type == ORDER_TYPE_BUY_LIMIT
                          || m_type == ORDER_TYPE_BUY_STOP
                          || m_type == ORDER_TYPE_SELL_LIMIT
                          || m_type == ORDER_TYPE_SELL_STOP);
   }
   bool              IsBuyOrder() {    // Is it an open BUY position?
      return IsOpen() && (m_type == ORDER_TYPE_BUY
                          || m_type == ORDER_TYPE_BUY_LIMIT
                          || m_type == ORDER_TYPE_BUY_STOP);
   }
   bool              IsSellOrder() {   // Is it an open SELL position?
      return IsOpen() && (m_type == ORDER_TYPE_SELL
                          || m_type == ORDER_TYPE_SELL_LIMIT
                          || m_type == ORDER_TYPE_SELL_STOP);
   }
   bool              IsStopOrder() {   // Is it a pending STOP order?
      return IsOpen() && (m_type == ORDER_TYPE_BUY_STOP || m_type == ORDER_TYPE_SELL_STOP);
   }
   bool              IsLimitOrder() {  // is it a pending LIMIT order?
      return IsOpen() && (m_type == ORDER_TYPE_BUY_LIMIT || m_type == ORDER_TYPE_SELL_LIMIT);
   }
   ...

//--- Methods for handling positions (orders)
   bool              CVirtualOrder::Open(string symbol,
                                         ENUM_ORDER_TYPE type,
                                         double lot,
                                         double price,
                                         double sl = 0,
                                         double tp = 0,
                                         string comment = "",
                                         datetime expiration = 0,
                                         bool inPoints = false); // Opening a position (order)

   ...
};
```

In the _Open()_ method of opening a virtual position, which will now open virtual pending orders as well, add assigning the open price to the _m\_openPrice_ property. If it turns out that this is not a pending order, but a position, add assigning the current market opening price to the property:

```
bool CVirtualOrder::Open(string symbol,         // Symbol
                         ENUM_ORDER_TYPE type,  // Type (BUY or SELL)
                         double lot,            // Volume
                         double price = 0,      // Open price
                         double sl = 0,         // StopLoss level (price or points)
                         double tp = 0,         // TakeProfit level (price or points)
                         string comment = "",   // Comment
                         datetime expiration = 0,  // Expiration time
                         bool inPoints = false  // Are the SL and TP levels set in points?
                        ) {
   ...

   if(s_symbolInfo.Name(symbol)) {  // Select the desired symbol
      s_symbolInfo.RefreshRates();  // Update information about current prices

      // Initialize position properties
      m_openPrice = price;
     ...
      m_expiration = expiration;

      // The position (order) being opened is not closed by SL, TP or expiration
      ...
      m_isExpired = false;

      ...
      // Depending on the direction, set the opening price, as well as the SL and TP levels.
      // If SL and TP are specified in points, then we first calculate their price levels
      // relative to the open price
      if(IsBuyOrder()) {
         if(type == ORDER_TYPE_BUY) {
            m_openPrice = s_symbolInfo.Ask();
         }
         ...
      } else if(IsSellOrder()) {
         if(type == ORDER_TYPE_SELL) {
            m_openPrice = s_symbolInfo.Bid();
         }
         ...
      }

      ...

      return true;
   }
   return false;
}
```

In the _CheckTrigger()_ method of checking whether a virtual pending order is triggered, we get the current market price Bid or Ask depending on the direction of the order and check if it has reached the opening price on the desired side. If yes, replace the _m\_type_ property of the current object with the value corresponding to the position of the desired direction, and notify the recipient and strategy objects that a new virtual position has opened.

```
//+------------------------------------------------------------------+
//| Check whether a pending order is triggered                       |
//+------------------------------------------------------------------+
bool CVirtualOrder::CheckTrigger() {
   if(IsPendingOrder()) {
      s_symbolInfo.Name(m_symbol);     // Select the desired symbol
      s_symbolInfo.RefreshRates();     // Update information about current prices
      double price = (IsBuyOrder()) ? s_symbolInfo.Ask() : s_symbolInfo.Bid();
      int spread = s_symbolInfo.Spread();

      // If the price has reached the opening levels, turn the order into a position
      if(false
            || (m_type == ORDER_TYPE_BUY_LIMIT && price <= m_openPrice)
            || (m_type == ORDER_TYPE_BUY_STOP  && price >= m_openPrice)
        ) {
         m_type = ORDER_TYPE_BUY;
      } else if(false
                || (m_type == ORDER_TYPE_SELL_LIMIT && price >= m_openPrice)
                || (m_type == ORDER_TYPE_SELL_STOP  && price <= m_openPrice)
               ) {
         m_type = ORDER_TYPE_SELL;
      }

      // If the order turned into a position
      if(IsMarketOrder()) {
         m_openPrice = price; // Remember the open price

         // Notify the recipient and the strategy of the position opening
         m_receiver.OnOpen(GetPointer(this));
         m_strategy.OnOpen();
         return true;
      }
   }
   return false;
}
```

This method will be called when handling a new tick in the method Tick() in the event that this is truly a virtual pending order:

```
//+------------------------------------------------------------------+
//| Handle a tick of a single virtual order (position)               |
//+------------------------------------------------------------------+
void CVirtualOrder::Tick() {
   if(IsOpen()) {  // If this is an open virtual position or order
      if(CheckClose()) {  // Check if SL or TP levels have been reached
         Close();         // Close when reached
      } else if (IsPendingOrder()) {   // If this is a pending order
         CheckTrigger();  // Check if it is triggered
      }
   }
}
```

In the _Tick()_ method, call the _CheckClose()_ method, where we also need to add the code that checks the closure of a virtual pending order based on expiration time:

```
//+------------------------------------------------------------------+
//| Check the need to close by SL, TP or EX                          |
//+------------------------------------------------------------------+
bool CVirtualOrder::CheckClose() {
   if(IsMarketOrder()) {               // If this is a market virtual position,
      ...
      // Check that the price has reached SL or TP
      ...
   } else if(IsPendingOrder()) {    // If this is a pending order
      // Check if the expiration time has been reached, if one is specified
      if(m_expiration > 0 && m_expiration < TimeCurrent()) {
         m_isExpired = true;
         return true;
      }
   }
   return false;
}
```

Save the changes to the _VirtualOrder.mqh_ file in the current folder.

Now let's go back to our _CSimpleVolumesStrategy_ trading strategy class. In that strategy, we left the margin for future edits to be able to add support for handling virtual pending orders. There were such places in the _OpenBuyOrder()_ and _OpenSellOrder()_ methods. Let's add here calling the _Open()_ method with parameters leading to opening virtual pending orders. We will first calculate the opening price from the current price, stepping back from it in the desired direction by the number of points specified by the _m\_openDistance_ parameter. We present the code only for the _OpenBuyOrder()_ method. In another code, the edits will be similar.

```
//+------------------------------------------------------------------+
//| Open BUY order                                                   |
//+------------------------------------------------------------------+
void CSimpleVolumesStrategy::OpenBuyOrder() {
// Update symbol current price data
   ...
// Retrieve the necessary symbol and price data
   ...

// Let's make sure that the opening distance is not less than the spread
   int distance = MathMax(m_openDistance, spread);

// Opening price
   double price = ask + distance * point;

// StopLoss and TakeProfit levels
   ...

// Expiration time
   datetime expiration = TimeCurrent() + m_ordersExpiration * 60;

   ...
   for(int i = 0; i < m_maxCountOfOrders; i++) {   // Iterate through all virtual positions
      if(!m_orders[i].IsOpen()) {                  // If we find one that is not open, then open it
         if(m_openDistance > 0) {
            // Set SELL STOP pending order
            res = m_orders[i].Open(m_symbol, ORDER_TYPE_BUY_STOP, m_fixedLot,
                                   NormalizeDouble(price, digits),
                                   NormalizeDouble(sl, digits),
                                   NormalizeDouble(tp, digits),
                                   "", expiration);

         } else if(m_openDistance < 0) {
            // Set SELL LIMIT pending order
            res = m_orders[i].Open(m_symbol, ORDER_TYPE_BUY_LIMIT, m_fixedLot,
                                   NormalizeDouble(price, digits),
                                   NormalizeDouble(sl, digits),
                                   NormalizeDouble(tp, digits),
                                   "", expiration);

         } else {
            // Open a virtual SELL position
            res = m_orders[i].Open(m_symbol, ORDER_TYPE_BUY, m_fixedLot,
                                   0,
                                   NormalizeDouble(sl, digits),
                                   NormalizeDouble(tp, digits));

         }
         break; // and exit
      }
   }
  ...
}
```

Save the changes in the _SimpleVolumesStrategy.mqh_ file of the current folder.

This completes the changes necessary to support the operation of strategies with virtual pending orders. We have made changes to only two files and now we can compile the _SimpleVolumesExpertSingle.mq5_ EA. When setting the _openDistance\__ parameter not equal to zero, the EA should open virtual pending orders instead of virtual positions. However, we will not see the opening on the chart. We can see the appropriate messages in the log only. We will be able to see them on the chart only after they have been converted into an open virtual position, which will be brought to the market by the object of recipients of virtual trading volumes.

It would be good to somehow see placed virtual pending orders on the chart. We will return to this issue a little later, but now let’s move on to a more important issue - ensuring that the EA’s state is saved and loaded after a restart.

### Saving status

We created two EAs based on the developed classes. The first one ( _SimpleVolumesExpertSingle.mq5_) was intended to optimize the parameters of a single instance of a trading strategy, and the second ( _SimpleVolumesExpert.mq5_) already included several copies of the trading strategy with the best parameters selected using the first EA. In the future, only the second EA has the prospect of being used on real accounts, and the first one is intended to be used only in the strategy tester. Therefore, we will only need to load and save the status in the second EA or others, which will also include many instances of trading strategies.

Next, it should be clarified that we are now talking about saving and loading the state of the EA, and not the different sets of strategies that should be used in the EA. In other words, the set of strategy instances with specified parameters in the EA is fixed and is the same every time it is launched. After the first launch, the EA opens virtual and real positions, and perhaps calculates some indicators based on price data. It is this information that makes up the EA status as a whole. If we now restart the terminal, the EA should recognize open positions as its own, as well as restore all its virtual positions and the necessary calculated values. While information about open positions can be obtained from the terminal, the EA should save information about virtual positions and calculated values independently.

For the simple trading strategy we have considered, there is no need to accumulate any calculated data, so the EA status will be entirely determined only by a set of virtual positions and pending orders. However, saving only the array of all _CVirtualOrder_ class objects to the file is insufficient.

Let's imagine that we already have several EAs with different sets of trading strategies. But the total number of _CVirtualOrder_ class objects created by each EA turned out to be the same. For example, in each we used 9 instances of trading strategies requesting 3 virtual position objects. Then each EA will store information about 27 _CVirtualOrder_ class objects. In this case, we need to somehow insure ourselves against the fact that one of the EAs will upload information not about their 27 virtual positions, but about others.

To do this, we can add information about the parameters of the strategies working as part of this EA, and, possibly, information about the parameters of the EA itself, to the saved file.

Now let's consider at what points in time the state should be saved. If the strategy parameters are fixed, then they are the same at any time. At the same time, objects of virtual positions can change the values of their properties during opening and closing operations. This means that it makes sense to save the status after these operations. Saving more frequently (for example, on every tick or by a timer) seems redundant for now.

Let us proceed with the implementation. Since we have a hierarchical structure of objects like

- EA
  - strategies
    - virtual positions

delegate our part of saving to each level, rather than focusing everything on the top level, although this is possible.

At each level, add two methods to the _Save()_ and _Load()_ classes responsible for saving and loading, respectively. At the top level, these methods will open a file, and at lower levels, they will get a descriptor of an already opened file as a parameter. Thus, we will leave the question of choosing a file name for saving only at the EA level. This question will not arise at the levels of strategy and virtual positions.

### EA modification

The _CVirtualAdvisor_ class gets the _m\_name_ field for storing the EA name. Since it will not change during operation, we will assign it in the constructor. It also makes sense to immediately expand the name by adding the magic number and the optional ".test" suffix to it in case the EA is launched in visual test mode.

To implement saving only in case of change in the composition of virtual positions, add the _m\_lastSaveTime_ field, which will store the time of the last save.

```
//+------------------------------------------------------------------+
//| Class of the EA handling virtual positions (orders)              |
//+------------------------------------------------------------------+
class CVirtualAdvisor : public CAdvisor {
protected:
   ...

   string            m_name;           // EA name
   datetime          m_lastSaveTime;   // Last save time

public:
   CVirtualAdvisor(ulong p_magic = 1, string p_name = ""); // Constructor
   ...

   virtual bool      Save();           // Save status
   virtual bool      Load();           // Load status
};
```

When creating an Expert Advisor, we will assign initial values to two new properties as follows:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CVirtualAdvisor::CVirtualAdvisor(ulong p_magic = 1, string p_name = "") :
   ...
   m_lastSaveTime(0) {
   m_name = StringFormat("%s-%d%s.csv",
                         (p_name != "" ? p_name : "Expert"),
                         p_magic,
                         (MQLInfoInteger(MQL_TESTER) ? ".test" : "")
                        );
};
```

Place the logic for checking whether saving is necessary inside the _Save()_ method. We can simply add the call of this method on each tick after performing the remaining actions on it:

```
//+------------------------------------------------------------------+
//| OnTick event handler                                             |
//+------------------------------------------------------------------+
void CVirtualAdvisor::Tick(void) {
// Receiver handles virtual positions
   m_receiver.Tick();

// Start handling in strategies
   CAdvisor::Tick();

// Adjusting market volumes
   m_receiver.Correct();

// Save status
   Save();
}
```

In the save method, the first thing we need to do is check whether we need to execute it. To do this, we will have to agree in advance that we will add a new property to the recipient object, which will store the time of the last changes in open virtual positions or the time of the last correction of real open volumes. If the time of the last save became less than the time of the last correction, then changes have occurred, and we need to save them.

Also, we will not save changes if optimization or single testing is currently underway without using visual mode. If testing is currently underway in visual mode, then saving will be performed. This will allow us to check saving in the strategy tester as well.

This is what the _Save()_ might look like at the EA level: we check the need for saving, then save the current time and number of strategies. After that, we call the saving method of all strategies in a loop.

```
//+------------------------------------------------------------------+
//| Save status                                                      |
//+------------------------------------------------------------------+
bool CVirtualAdvisor::Save() {
   bool res = true;

   // Save status if:
   if(true
         // later changes appeared
         && m_lastSaveTime < CVirtualReceiver::s_lastChangeTime
         // currently, there is no optimization
         && !MQLInfoInteger(MQL_OPTIMIZATION)
         // and there is no testing at the moment or there is a visual test at the moment
         && (!MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_VISUAL_MODE))
     ) {
      int f = FileOpen(m_name, FILE_CSV | FILE_WRITE, '\t');

      if(f != INVALID_HANDLE) {  // If file is open, save
         FileWrite(f, CVirtualReceiver::s_lastChangeTime);  // Time of last changes
         FileWrite(f, ArraySize(m_strategies));             // Number of strategies

         // All strategies
         FOREACH(m_strategies, ((CVirtualStrategy*) m_strategies[i]).Save(f));

         FileClose(f);

         // Update the last save time
         m_lastSaveTime = CVirtualReceiver::s_lastChangeTime;
         PrintFormat(__FUNCTION__" | OK at %s to %s",
                     TimeToString(m_lastSaveTime, TIME_DATE | TIME_MINUTES | TIME_SECONDS), m_name);
      } else {
         PrintFormat(__FUNCTION__" | ERROR: Operation FileOpen for %s failed, LastError=%d",
                     m_name, GetLastError());
         res = false;
      }
   }
   return res;
}
```

After saving the strategies, we update the time of the last save in accordance with the time of the last changes in virtual positions. Now the saving method will not save anything to the file until the next change.

The _Load()_ status loading method should do a similar job, but instead of writing, it will read data from the file. In other words, first we read the saved time and number of strategies. Here, we can check whether the number of read strategies corresponds to the number of strategies added to the EA just in case. If not, then there is no point in reading further, this is a wrong file. If yes, then all is well, we can read on. Then we again delegate the subsequent work to objects of the next level of the hierarchy: we go through all the added strategies and call their reading method from the open file.

This might look something like this in the code:

```
//+------------------------------------------------------------------+
//| Load status                                                      |
//+------------------------------------------------------------------+
bool CVirtualAdvisor::Load() {
   bool res = true;

   // Load status if:
   if(true
         // file exists
         && FileIsExist(m_name)
         // currently, there is no optimization
         && !MQLInfoInteger(MQL_OPTIMIZATION)
         // and there is no testing at the moment or there is a visual test at the moment
         && (!MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_VISUAL_MODE))
     ) {
      int f = FileOpen(m_name, FILE_CSV | FILE_READ, '\t');

      if(f != INVALID_HANDLE) {  // If the file is open, then load
         m_lastSaveTime = FileReadDatetime(f);     // Last save time
         PrintFormat(__FUNCTION__" | LAST SAVE at %s", TimeToString(m_lastSaveTime, TIME_DATE | TIME_MINUTES | TIME_SECONDS));

         // Number of strategies
         long f_strategiesCount = StringToInteger(FileReadString(f));

         // Does the loaded number of strategies match the current one?
         res = (ArraySize(m_strategies) == f_strategiesCount);

         if(res) {
            // Load all strategies
            FOREACH(m_strategies, res &= ((CVirtualStrategy*) m_strategies[i]).Load(f));

            if(!res) {
               PrintFormat(__FUNCTION__" | ERROR loading strategies from file %s", m_name);
            }
         } else {
            PrintFormat(__FUNCTION__" | ERROR: Wrong strategies count (%d expected but %d found in file %s)",
                        ArraySize(m_strategies), f_strategiesCount, m_name);
         }
         FileClose(f);
      } else {
         PrintFormat(__FUNCTION__" | ERROR: Operation FileOpen for %s failed, LastError=%d", m_name, GetLastError());
         res = false;
      }
   }
   return res;
}
```

Save the changes made to the _VirtualAdvisor.mqh_ file in the current folder.

We should call the status loading method only once when launching the EA, but we cannot do this in the EA object constructor, since at this moment strategies have not yet been added to the EA. So, let's do it in the EA file of the _OnInit()_ function after adding all strategy instances to the EA object:

```
CVirtualAdvisor     *expert;                  // EA object

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
// Create and fill the array of strategy instances
   CStrategy *strategies[9];
   strategies[0] = ...
   ...
   strategies[8] = ...

// Create an EA handling virtual positions
   expert = new CVirtualAdvisor(magic_, "SimpleVolumes");

// Add strategies to the EA
   FOREACH(strategies, expert.Add(strategies[i]));

// Load the previous state if available
   expert.Load();

   return(INIT_SUCCEEDED);
}
```

Save the changes in the _SimpleVolumesExpert.mq5_ file of the current folder.

### Basic strategy modification

Add the _Save()_ and _Load()_ methods, as well as the method to convert the current strategy object to a string, to the base class of a trading strategy. For brevity, let's implement this method as an overloaded unary operator ~ (tilde).

```
//+------------------------------------------------------------------+
//| Class of a trading strategy with virtual positions               |
//+------------------------------------------------------------------+
class CVirtualStrategy : public CStrategy {
   ...

public:
   ...

   virtual bool      Load(const int f);   // Load status
   virtual bool      Save(const int f);   // Save status

   string operator~();                    // Convert object to string
};
```

The method of converting an object to a string will return a string with the name of the current class and the number of elements of the virtual positions array:

```
//+------------------------------------------------------------------+
//| Convert an object to a string                                    |
//+------------------------------------------------------------------+
string CVirtualStrategy::operator~() {
   return StringFormat("%s(%d)", typename(this), ArraySize(m_orders));
}
```

The descriptor of the file opened for writing is passed to the _Save()_ method. The string obtained from the object when converting to a string is written to the file first. Then, its saving method will be called in a loop for each virtual position.

```
//+------------------------------------------------------------------+
//| Save status                                                      |
//+------------------------------------------------------------------+
bool CVirtualStrategy::Save(const int f) {
   bool res = true;
   FileWrite(f, ~this); // Save parameters

   // Save virtual positions (orders) of the strategy
   FOREACH(m_orders, res &= m_orders[i].Save(f));

   return res;
}
```

The _Load()_ method will simply read the data in the same order it was written in, while making sure the parameter string in the file and in the strategy match:

```
//+------------------------------------------------------------------+
//| Load status                                                      |
//+------------------------------------------------------------------+
bool CVirtualStrategy::Load(const int f) {
   bool res = true;
   // Current parameters are equal to read parameters
   res = (~this == FileReadString(f));

   // If yes, then load the virtual positions (orders) of the strategy
   if(res) {
      FOREACH(m_orders, res &= m_orders[i].Load(f));
   }

   return res;
}
```

Save the changes implemented to the _VirtualStrategy.mqh_ file in the current folder.

### Trading strategy modification

In the _CSimpleVolumesStrategy_ class of the specific trading strategy, we need to add the same set of methods as in the base class:

```
//+------------------------------------------------------------------+
//| Trading strategy using tick volumes                               |
//+------------------------------------------------------------------+
class CSimpleVolumesStrategy : public CVirtualStrategy {
   ...

public:
   ...

   virtual bool      Load(const int f) override;   // Load status
   virtual bool      Save(const int f) override;   // Save status

   string operator~();                    // Convert object to string
};
```

In the method of converting a trading strategy into a string, we will generate the result from the name of the class and listing all critical parameters. Currently, for simplicity, we will add the values of all parameters here, but in the future it may be more convenient for us to shorten this list. This will allow us to restart the EA with slightly changed parameters without completely closing all previously open market positions. For example, if we increase the TakeProfit parameter, then we can safely leave open positions with a lower TakeProfit level.

```
//+------------------------------------------------------------------+
//| Convert an object to a string                                    |
//+------------------------------------------------------------------+
string CSimpleVolumesStrategy::operator~() {
   return StringFormat("%s(%s,%s,%.2f,%d,%.2f,%.2f,%d,%.2f,%.2f,%d,%d)",
                       // Strategy instance parameters
                       typename(this), m_symbol, EnumToString(m_timeframe), m_fixedLot,
                       m_signalPeriod, m_signalDeviation, m_signaAddlDeviation,
                       m_openDistance, m_stopLevel, m_takeLevel, m_ordersExpiration,
                       m_maxCountOfOrders
                      );
}
```

This turned out to be another place where we need to write all the strategy parameters in the code again. We will remember about it until we get to working with inputs.

The _Save()_ method turns out to be very concise, since the base class will do the main work:

```
//+------------------------------------------------------------------+
//| Save status                                                      |
//+------------------------------------------------------------------+
bool CSimpleVolumesStrategy::Save(const int f) {
   bool res = true;
   FileWrite(f, ~this);                // Save parameters
   res &= CVirtualStrategy::Save(f);   // Save strategy
   return res;
}
```

The _Load()_ method will be slightly larger, but mainly due to increased code readability:

```
//+------------------------------------------------------------------+
//| Load status                                                      |
//+------------------------------------------------------------------+
bool CSimpleVolumesStrategy::Load(const int f) {
   bool res = true;
   string currentParams = ~this;             // Current parameters
   string loadedParams = FileReadString(f);  // Read parameters

   PrintFormat(__FUNCTION__" | %s", loadedParams);

   res = (currentParams == loadedParams);

   // Load if read parameters match the current ones
   if(res) {
      res &= CVirtualStrategy::Load(f);
   }

   return res;
}
```

Save the changes in the _SimpleVolumesExpert.mqh_ file of the current folder.

### Modification of virtual positions

We need to add three methods for the _CVirtualOrder_ class of virtual positions:

```
class CVirtualOrder {
   ...

   virtual bool      Load(const int f);   // Load status
   virtual bool      Save(const int f);   // Save status

   string            operator~();         // Convert object to string
};
```

We will not use the method of converting to string when saving to a file, but it will be useful when logging the loaded object data:

```
//+------------------------------------------------------------------+
//| Convert an object to a string                                    |
//+------------------------------------------------------------------+
string CVirtualOrder::operator~() {
   if(IsOpen()) {
      return StringFormat("#%d %s %s %.2f in %s at %.5f (%.5f, %.5f). %s, %f",
                          m_id, TypeName(), m_symbol, m_lot,
                          TimeToString(m_openTime), m_openPrice,
                          m_stopLoss, m_takeProfit,
                          TimeToString(m_closeTime), m_closePrice);
   } else {
      return StringFormat("#%d --- ", m_id);
   }

}
```

Although, we might change this later by adding the method for reading object properties from a string.

We will finally have some more significant information written to the file in the _Save()_ method:

```
//+------------------------------------------------------------------+
//| Save status                                                      |
//+------------------------------------------------------------------+
bool CVirtualOrder::Save(const int f) {
   FileWrite(f, m_id, m_symbol, m_lot, m_type, m_openPrice,
             m_stopLoss, m_takeProfit,
             m_openTime, m_closePrice, m_closeTime,
             m_expiration, m_comment, m_point);
   return true;
}
```

The _Load()_ will then not only read what was written, filling the necessary properties with the read information, but also notify the associated strategy objects and the recipient about whether this virtual position (order) is open or closed:

```
//+------------------------------------------------------------------+
//| Load status                                                      |
//+------------------------------------------------------------------+
bool CVirtualOrder::Load(const int f) {
   m_id = (ulong) FileReadNumber(f);
   m_symbol = FileReadString(f);
   m_lot = FileReadNumber(f);
   m_type = (ENUM_ORDER_TYPE) FileReadNumber(f);
   m_openPrice = FileReadNumber(f);
   m_stopLoss = FileReadNumber(f);
   m_takeProfit = FileReadNumber(f);
   m_openTime = FileReadDatetime(f);
   m_closePrice = FileReadNumber(f);
   m_closeTime = FileReadDatetime(f);
   m_expiration = FileReadDatetime(f);
   m_comment = FileReadString(f);
   m_point = FileReadNumber(f);

   PrintFormat(__FUNCTION__" | %s", ~this);

// Notify the recipient and the strategy that the position (order) is open
   if(IsOpen()) {
      m_receiver.OnOpen(GetPointer(this));
      m_strategy.OnOpen();
   } else {
      m_receiver.OnClose(GetPointer(this));
      m_strategy.OnClose();
   }

   return true;
}
```

Save the obtained code in the _VirtualOrder.mqh_ file of the current folder.

### Testing the saving

Now let's test the saving and loading status data. To avoid waiting for a favorable moment to open positions, we will make temporary changes to our trading strategy forcing us to open one position or a pending order at startup, if there are no open positions/orders yet.

Since we added the display of messages about the loading progress, when we restart the EA (the easiest way for us to do this is to simply re-compile), we will see something like the following in the logs:

```
CVirtualAdvisor::Load | LAST SAVE at 2027.02.23 08:05:33

CSimpleVolumesStrategy::Load | class CSimpleVolumesStrategy(EURGBP,PERIOD_H1,0.06,13,0.30,1.00,0,10500.00,465.00,1000,3)
CVirtualOrder::Load | Order#1 EURGBP 0.06 BUY in 2027.02.23 08:02 at 0.85494 (0.75007, 0.85985). 1970.01.01 00:00, 0.000000
CVirtualReceiver::OnOpen#EURGBP | OPEN VirtualOrder #1
CVirtualOrder::Load | Order#2  ---
CVirtualOrder::Load | Order#3  ---

CSimpleVolumesStrategy::Load | class CSimpleVolumesStrategy(EURGBP,PERIOD_H1,0.11,17,1.70,0.50,210,16500.00,220.00,1000,3)
CVirtualOrder::Load | Order#4 EURGBP 0.11 BUY STOP in 2027.02.23 08:02 at 0.85704 (0.69204, 0.85937). 1970.01.01 00:00, 0.000000
CVirtualOrder::Load | Order#5  ---
CVirtualOrder::Load | Order#6  ---

CSimpleVolumesStrategy::Load | class CSimpleVolumesStrategy(EURGBP,PERIOD_H1,0.06,51,0.50,1.10,500,19500.00,370.00,22000,3)
CVirtualOrder::Load | Order#7 EURGBP 0.06 BUY STOP in 2027.02.23 08:02 at 0.85994 (0.66494, 0.86377). 1970.01.01 00:00, 0.000000
CVirtualOrder::Load | Order#8  ---
CVirtualOrder::Load | Order#9  ---

CSimpleVolumesStrategy::Load | class CSimpleVolumesStrategy(GBPUSD,PERIOD_H1,0.04,80,1.10,0.20,0,6000.00,1190.00,1000,3)
CVirtualOrder::Load | Order#10 GBPUSD 0.04 BUY in 2027.02.23 08:02 at 1.26632 (1.20638, 1.27834). 1970.01.01 00:00, 0.000000
CVirtualReceiver::OnOpen#GBPUSD | OPEN VirtualOrder #10
CVirtualOrder::Load | Order#11  ---
CVirtualOrder::Load | Order#12  ---

CSimpleVolumesStrategy::Load | class CSimpleVolumesStrategy(GBPUSD,PERIOD_H1,0.11,128,2.00,0.90,220,2000.00,1170.00,1000,3)
CVirtualOrder::Load | Order#13 GBPUSD 0.11 BUY STOP in 2027.02.23 08:02 at 1.26852 (1.24852, 1.28028). 1970.01.01 00:00, 0.000000
CVirtualOrder::Load | Order#14  ---
CVirtualOrder::Load | Order#15  ---

CSimpleVolumesStrategy::Load | class CSimpleVolumesStrategy(GBPUSD,PERIOD_H1,0.07,13,1.50,0.80,550,2500.00,1375.00,1000,3)
CVirtualOrder::Load | Order#16 GBPUSD 0.07 BUY STOP in 2027.02.23 08:02 at 1.27182 (1.24682, 1.28563). 1970.01.01 00:00, 0.000000
CVirtualOrder::Load | Order#17  ---
CVirtualOrder::Load | Order#18  ---

CSimpleVolumesStrategy::Load | class CSimpleVolumesStrategy(EURUSD,PERIOD_H1,0.04,24,0.10,0.30,330,7500.00,2400.00,24000,3)
CVirtualOrder::Load | Order#19 EURUSD 0.04 BUY STOP in 2027.02.23 08:02 at 1.08586 (1.01086, 1.10990). 1970.01.01 00:00, 0.000000
CVirtualOrder::Load | Order#20  ---
CVirtualOrder::Load | Order#21  ---
CSimpleVolumesStrategy::Load | class CSimpleVolumesStrategy(EURUSD,PERIOD_H1,0.05,18,0.20,0.40,220,19500.00,1480.00,6000,3)
CVirtualOrder::Load | Order#22 EURUSD 0.05 BUY STOP in 2027.02.23 08:02 at 1.08476 (0.88976, 1.09960). 1970.01.01 00:00, 0.000000
CVirtualOrder::Load | Order#23  ---
CVirtualOrder::Load | Order#24  ---
CSimpleVolumesStrategy::Load | class CSimpleVolumesStrategy(EURUSD,PERIOD_H1,0.05,128,0.70,0.30,550,3000.00,170.00,42000,3)
CVirtualOrder::Load | Order#25 EURUSD 0.05 BUY STOP in 2027.02.23 08:02 at 1.08806 (1.05806, 1.08980). 1970.01.01 00:00, 0.000000
CVirtualOrder::Load | Order#26  ---
CVirtualOrder::Load | Order#27  ---

CVirtualAdvisor::Save | OK at 2027.02.23 08:19:48 to SimpleVolumes-27182.csv
```

As we can see, the data about open virtual positions and pending orders is loaded successfully. In case of virtual positions, the open event handler is called, which, if necessary, opens real market positions, for instance, if they were closed manually.

In general, the EA behavior in relation to open positions during a restart is a rather complex issue. For example, if we want to change the magic number, the EA will no longer consider previously opened trades as the ones of its own. Should they be forcefully closed? If we want to replace the EA version, then we may need to completely ignore the presence of saved virtual positions and forcefully close all open positions. We should decide each time which scenario suits us best. These issues are not very pressing yet, so we will postpone them for later.

### Visualization

Now it is time to visualize virtual positions and pending orders. At first glance, it seems quite natural to implement it as an extension of the _CVirtualOrder_ virtual position class. It already has all the necessary information about the visualized object. It knows better than anyone when it needs to redraw itself. Actually, the first draft implementation was done exactly this way. But then very unpleasant questions began to emerge.

One of these questions is "What chart do we plan to perform visualization on?" The simplest answer is the current one. But it is only suitable for the case when the EA works on one symbol, and it matches the chart symbol, on which the EA is launched. As soon as there are several symbols, it becomes very inconvenient to display all virtual positions on the symbol chart. The chart turns into a mess.

In other words, the issue of choosing a chart for display required a solution, but it no longer related to the main functionality of the _CVirtualOrder_ class objects. They worked well even without our visualization.

Therefore, let's leave this class alone and look a little ahead. If we slightly expand our objectives, it would be nice to be able to selectively enable/disable the display of virtual positions grouped by strategy or type, as well as implement a more detailed data, for example visible open prices, StopLoss and TakeProfit, planned loss and profit if they are triggered, and maybe even the ability to manually modify these levels. Although the latter would be more useful if we were developing a panel for semi-automated trading, rather than an EA based on strictly algorithmic strategies. In general, even when starting a simple implementation, we can get ahead a bit, imagining the further code development. This can help us choose a direction, in which we will be less likely to return to revising already written code.

So let's create a new base class for all objects that will in one way or another be associated with visualizing something on charts:

```
//+------------------------------------------------------------------+
//| Basic class for visualizing various objects                      |
//+------------------------------------------------------------------+
class CInterface {
protected:
   static ulong      s_magic;       // EA magic number
   bool              m_isActive;    // Is the interface active?
   bool              m_isChanged;   // Does the object have any changes?
public:
   CInterface();                    // Constructor
   virtual void      Redraw() = 0;  // Draw changed objects on the chart
   virtual void      Changed() {    // Set the flag for changes
      m_isChanged = true;
   }
};

ulong CInterface::s_magic = 0;

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CInterface::CInterface() :
   m_isActive(!MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_VISUAL_MODE)),
   m_isChanged(true) {}
```

Save this code in the _Interface.mqh_ of the current folder.

Create two new classes based on this class:

- _CVirtualChartOrder —_ object displaying one virtual position or a pending order on a chart in the terminal (graphical virtual position). It will be able to draw a virtual position on the chart if changes have occurred in it, and the chart with the required instrument will automatically open if it was not opened in the terminal.
- _CVirtualInterface —_ aggregator of all graphical objects of the EA interface. For now, it only contains the array of graphical virtual positions. It will deal with creating graphical virtual position objects every time a virtual position object is created. It will also receive messages about changes in the composition of virtual positions and cause the corresponding graphical virtual positions to be redrawn. Such an aggregator will exist in a single instance (implementing the Singleton design pattern) and will be available in the _CVirtualAdvisor_ class.

The _CVirtualChartOrder_ class looks as follows:

```
//+------------------------------------------------------------------+
//| Graphic virtual position class                                   |
//+------------------------------------------------------------------+
class CVirtualChartOrder : public CInterface {
   CVirtualOrder*    m_order;          // Associated virtual position (order)
   CChart            m_chart;          // Chart object to be displayed

   // Objects on the chart to display the virtual position
   CChartObjectHLine m_openLine;       // Open price line

   long              FindChart();      // Search/open the desired chart
public:
   CVirtualChartOrder(CVirtualOrder* p_order);     // Constructor
   ~CVirtualChartOrder();                          // Destructor

   bool              operator==(const ulong id) {  // Comparison operator by Id
      return m_order.Id() == id;
   }

   void              Show();    // Show a virtual position (order)
   void              Hide();    // Hide a virtual position (order)

   virtual void      Redraw() override;   // Redraw a virtual position (order)
};
```

The _Redraw()_ method checks whether it needs to be executed and, if necessary, call methods for showing or hiding a virtual position from the chart:

```
//+------------------------------------------------------------------+
//| Redraw a virtual position (order)                                |
//+------------------------------------------------------------------+
void CVirtualChartOrder::Redraw() {
   if(m_isChanged) {
      if(m_order.IsOpen()) {
         Show();
      } else {
         Hide();
      }
      m_isChanged = false;
   }
}
```

In the _Show()_ display method, we first call the _FindChart()_ method to determine which chart the virtual position should be displayed on. In this method, we simply iterate through all open charts until we find a chart with a matching symbol. If we do not find one, then we will open a new chart. The found (or opened) chart is stored in the _m\_chart_ property.

```
//+------------------------------------------------------------------+
//| Finding a chart to display                                       |
//+------------------------------------------------------------------+
long CVirtualChartOrder::FindChart() {
   if(m_chart.ChartId() == -1 || m_chart.Symbol() != m_order.Symbol()) {
      long currChart, prevChart = ChartFirst();
      int i = 0, limit = 1000;

      currChart = prevChart;

      while(i < limit) { // we probably have no more than 1000 open charts
         if(ChartSymbol(currChart) == m_order.Symbol()) {
            return currChart;
         }
         currChart = ChartNext(prevChart); // get new chart on the basis of the previous one
         if(currChart < 0)
            break;        // end of chart list is reached
         prevChart = currChart; // memorize identifier of the current chart for ChartNext()
         i++;
      }

      // If a suitable chart is not found, then open a new one
      if(currChart == -1) {
         m_chart.Open(m_order.Symbol(), PERIOD_CURRENT);
      }
   }
   return m_chart.ChartId();
}
```

The _Show()_ method simply draws one horizontal line corresponding to the opening price. Its color and type will be determined depending on the position (order) direction and type. The _Hide()_ method will delete this line. Further enrichment of the class will occur in these methods.

Save the obtained code in the _VirtualChartOrder.mqh_ file of the current folder.

The implementation of the _CVirtualInterface_ class may be done as follows:

```
//+------------------------------------------------------------------+
//| EA GUI class                                                     |
//+------------------------------------------------------------------+
class CVirtualInterface : public CInterface {
protected:
// Static pointer to a single class instance
   static   CVirtualInterface *s_instance;

   CVirtualChartOrder *m_chartOrders[];   // Array of graphical virtual positions

//--- Private methods
   CVirtualInterface();   // Closed constructor

public:
   ~CVirtualInterface();  // Destructor

//--- Static methods
   static
   CVirtualInterface  *Instance(ulong p_magic = 0);   // Singleton - creating and getting a single instance

//--- Public methods
   void              Changed(CVirtualOrder *p_order); // Handle virtual position changes
   void              Add(CVirtualOrder *p_order);     // Add a virtual position

   virtual void      Redraw() override;   // Draw changed objects on the chart
};
```

In the static method for creating a single _Instance()_, add initialization of the _s\_magic_ static variable if a non-zero magic number was passed:

```
//+------------------------------------------------------------------+
//| Singleton - creating and getting a single instance               |
//+------------------------------------------------------------------+
CVirtualInterface* CVirtualInterface::Instance(ulong p_magic = 0) {
   if(!s_instance) {
      s_instance = new CVirtualInterface();
   }
   if(s_magic == 0 && p_magic != 0) {
      s_magic = p_magic;
   }
   return s_instance;
}
```

In the method for handling the opening/closing event of a virtual position, we will find its corresponding graphical virtual position object and mark that changes have occurred in it:

```
//+------------------------------------------------------------------+
//| Handle virtual position changes                                  |
//+------------------------------------------------------------------+
void CVirtualInterface::Changed(CVirtualOrder *p_order) {
   // Remember that this position has changes
   int i;
   FIND(m_chartOrders, p_order.Id(), i);
   if(i != -1) {
      m_chartOrders[i].Changed();
      m_isChanged = true;
   }
}
```

Finally, in the _Redraw()_ interface rendering method, call methods for drawing all graphical virtual positions in a loop:

```
//+------------------------------------------------------------------+
//| Draw changed objects on a chart                                  |
//+------------------------------------------------------------------+
void CVirtualInterface::Redraw() {
   if(m_isActive && m_isChanged) {  // If the interface is active and there are changes
      // Start redrawing graphical virtual positions
      FOREACH(m_chartOrders, m_chartOrders[i].Redraw());
      m_isChanged = false;          // Reset the changes flag
   }
}
```

Save the obtained code in the _VirtualInterface.mqh_ file of the current folder.

Now all that remains is to make the final edits to get the subsystem for displaying virtual positions on charts to work. In the _CVirtualAdvisor_ class, add the new _m\_interface_ property, which will store a single instance of the display interface object. We need to take care of initializing it in the constructor and deleting it in the destructor:

```
//+------------------------------------------------------------------+
//| Class of the EA handling virtual positions (orders)              |
//+------------------------------------------------------------------+
class CVirtualAdvisor : public CAdvisor {
protected:
   ...
   CVirtualInterface *m_interface;     // Interface object to show the status to the user

   ...
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CVirtualAdvisor::CVirtualAdvisor(ulong p_magic = 1, string p_name = "") :
   ...
// Initialize the interface with the static interface
   m_interface(CVirtualInterface::Instance(p_magic)),
   ... {
   ...
};

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
void CVirtualAdvisor::~CVirtualAdvisor() {

...
   delete m_interface;        // Remove the interface
}
```

In the _OnTick_ event handler, after all the operations, add calling the method of redrawing the interface since this is the least important part of tick handling:

```
//+------------------------------------------------------------------+
//| OnTick event handler                                             |
//+------------------------------------------------------------------+
void CVirtualAdvisor::Tick(void) {
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

In the _CVirtualReceiver_ class, also add the new _m\_interface_ property, which will store a single instance of the display interface object. We need to take care of its initialization in the constructor:

```
//+------------------------------------------------------------------+
//| Class for converting open volumes to market positions (receiver) |
//+------------------------------------------------------------------+
class CVirtualReceiver : public CReceiver {
protected:
   ...
   CVirtualInterface
   *m_interface;                          // Interface object to show the status to the user

   ...
};

//+------------------------------------------------------------------+
//| Closed constructor                                               |
//+------------------------------------------------------------------+
CVirtualReceiver::CVirtualReceiver() :
   m_interface(CVirtualInterface::Instance()),
   ... {}
```

In the method for allocating the required number of virtual positions to strategies, we will simultaneously add them to the interface:

```
//+------------------------------------------------------------------+
//| Allocate the necessary amount of virtual positions to strategy   |
//+------------------------------------------------------------------+
static void CVirtualReceiver::Get(CVirtualStrategy *strategy,   // Strategy
                                  CVirtualOrder *&orders[],     // Array of strategy positions
                                  int n                         // Required number
                                 ) {
   CVirtualReceiver *self = Instance();   // Receiver singleton
   CVirtualInterface *draw = CVirtualInterface::Instance();
   ArrayResize(orders, n);                // Expand the array of virtual positions
   FOREACH(orders,
           orders[i] = new CVirtualOrder(strategy); // Fill the array with new objects
           APPEND(self.m_orders, orders[i]);
           draw.Add(orders[i])) // Register the created virtual position
   PrintFormat(__FUNCTION__ + " | OK, Strategy orders: %d from %d total",
               ArraySize(orders),
               ArraySize(self.m_orders));
}
```

The last thing we need to do is add an interface alert in the methods for handling opening/closing virtual positions in this class:

```
void CVirtualReceiver::OnOpen(CVirtualOrder *p_order) {
   m_interface.Changed(p_order);
   ...
}

//+------------------------------------------------------------------+
//| Handle closing a virtual position                                |
//+------------------------------------------------------------------+
void CVirtualReceiver::OnClose(CVirtualOrder *p_order) {
   m_interface.Changed(p_order);
   ...
   }
}
```

Save the changes made to the _VirtualAdvisor.mqh_ and _VirtualReceiver.mqh_ files in the current folder.

If we now compile and run our EA, then we can see something like this if there are open virtual positions or pending orders:

![Fig. 1. Displaying virtual orders and positions on the chart](https://c.mql5.com/2/71/2024-02-24_10-58-50.png)

Fig. 1. Displaying virtual orders and positions on the chart

Here, the dotted lines show virtual pending orders: orange - SELL STOP, blue - BUY STOP, while solid lines display virtual positions: blue - BUY, red - SELL. For now, only opening levels are visible, but in the future it will be possible to make the display more saturated.

### Beautiful chart

In the forum discussions, there was an idea that readers of articles (at least some of them) first of all look at the end of the article, where they expect to see a beautiful chart showing the growth of funds when testing a developed EA. If there is indeed such a chart, then this becomes an additional incentive to return to the beginning of the article and read it.

While working on this article, we did not make any changes to the demo strategy used or to the set of strategy instances in the demo EA. Everything remains in the same condition as at the time of publication of the previous article. Therefore, there are no impressive charts here.

However, in parallel to the current article, I have optimized and trained other strategies to be published later. Their test results allow us to say that combining a large number of strategy instances into one EA is a promising method.

Here are two examples of test runs of the EA, which uses about 170 instances of strategies working on various symbols and timeframes. The test period: from 2023-01-01 to 2024-02-23. The period data was not used for optimization and training. In the capital management settings, parameters were set that assumed the acceptable drawdown in one case was about 10%, and in another - about 40%.

![](https://c.mql5.com/2/71/4470826082421.png)

![](https://c.mql5.com/2/71/6197723829832.png)

Fig. 2. Test results with an acceptable drawdown of 10%

![](https://c.mql5.com/2/71/2049887009351.png)

![](https://c.mql5.com/2/71/5432177210538.png)

Fig. 3. Test results with an acceptable drawdown of 40%

The presence of such results does not guarantee their repetition in the future. But we can work to make that outcome more likely. I will try not to worsen these results and not fall into the self-deception of overtraining.

### Conclusion

In this article, we have done work that takes us somewhat away from the main direction. I would like to move towards automatic optimization, which, for example, was discussed in a recently published article [Using optimization algorithms to configure EA parameters on the fly](https://www.mql5.com/en/articles/14183).

But nevertheless, the ability to save and load status is an important component of any EA that has prospects of ever being launched on a real account. No less important is the ability to work not only with market positions, but also with pending orders used in many trading strategies. At the same time, visualization of the EA work can help identify any implementation errors at the development stage.

Thank you for your attention and see you in the next part!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14246](https://www.mql5.com/ru/articles/14246)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14246.zip "Download all attachments in the single ZIP archive")

[Advisor.mqh](https://www.mql5.com/en/articles/download/14246/advisor.mqh "Download Advisor.mqh")(4.3 KB)

[Interface.mqh](https://www.mql5.com/en/articles/download/14246/interface.mqh "Download Interface.mqh")(3.21 KB)

[Macros.mqh](https://www.mql5.com/en/articles/download/14246/macros.mqh "Download Macros.mqh")(2.28 KB)

[Receiver.mqh](https://www.mql5.com/en/articles/download/14246/receiver.mqh "Download Receiver.mqh")(1.79 KB)

[SimpleVolumesExpert.mq5](https://www.mql5.com/en/articles/download/14246/simplevolumesexpert.mq5 "Download SimpleVolumesExpert.mq5")(7.77 KB)

[SimpleVolumesExpertSingle.mq5](https://www.mql5.com/en/articles/download/14246/simplevolumesexpertsingle.mq5 "Download SimpleVolumesExpertSingle.mq5")(7.32 KB)

[SimpleVolumesStrategy.mqh](https://www.mql5.com/en/articles/download/14246/simplevolumesstrategy.mqh "Download SimpleVolumesStrategy.mqh")(30.18 KB)

[Strategy.mqh](https://www.mql5.com/en/articles/download/14246/strategy.mqh "Download Strategy.mqh")(1.73 KB)

[VirtualAdvisor.mqh](https://www.mql5.com/en/articles/download/14246/virtualadvisor.mqh "Download VirtualAdvisor.mqh")(13.2 KB)

[VirtualChartOrder.mqh](https://www.mql5.com/en/articles/download/14246/virtualchartorder.mqh "Download VirtualChartOrder.mqh")(10.83 KB)

[VirtualInterface.mqh](https://www.mql5.com/en/articles/download/14246/virtualinterface.mqh "Download VirtualInterface.mqh")(8.41 KB)

[VirtualOrder.mqh](https://www.mql5.com/en/articles/download/14246/virtualorder.mqh "Download VirtualOrder.mqh")(38.33 KB)

[VirtualReceiver.mqh](https://www.mql5.com/en/articles/download/14246/virtualreceiver.mqh "Download VirtualReceiver.mqh")(17.37 KB)

[VirtualStrategy.mqh](https://www.mql5.com/en/articles/download/14246/virtualstrategy.mqh "Download VirtualStrategy.mqh")(7.4 KB)

[VirtualSymbolReceiver.mqh](https://www.mql5.com/en/articles/download/14246/virtualsymbolreceiver.mqh "Download VirtualSymbolReceiver.mqh")(33.97 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/469228)**
(5)


![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
29 Feb 2024 at 23:46

Wouldn't use such an architecture.

Imagine that MQ made virtual commerce built into the language. Then you wouldn't get the idea of embedding other entities in it, because it simply can't be done through OOP inheritance - the sources aren't available. You would create a different architecture.

Graphical objects in a virtualisation - well that's all in the same pile again.

The architecture has become so unwieldy that it makes you feel uncomfortable. Instead of assembling from simple bricks, you decided to create universal bricks.

![Yuriy Bykov](https://c.mql5.com/avatar/avatar_na2.png)

**[Yuriy Bykov](https://www.mql5.com/en/users/antekov)**
\|
1 Mar 2024 at 07:10

We're back to not really liking the implementation, but liking others (that aren't done but have been thought about) even less. Perhaps in the process of further development it will be possible to do it differently, but for now.

About the situation with inaccessible sources: in this implementation I decided just to take advantage of the fact that the sources are available and you can add something to them that was not necessary to add at once. I tried to make such additions minimal. The alternative was to create several new inheritor classes for the family of CVirtual\* classes. This approach seemed even more cumbersome to me. But it is quite possible that we will come to it when there will be more classes and it will become ugly to store them in one folder.

I needed graphical objects to control the development of [trading strategies](https://www.mql5.com/en/articles/3074 "Article: Comparative Analysis of 10 Trending Strategies "), so they were implemented. And the CVirtualOrder class was not changed at all. But I had to add four new lines of code to the CVirtualReceiver class. I chose this option among different possible ones.

If there is no need in graphical display of virtual positions, you can either not use it or return to the library variant from the previous article.

![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
1 Mar 2024 at 07:31

**Yuriy Bykov [#](https://www.mql5.com/ru/forum/463345#comment_52573809):**

We're back to not really liking the implementation, but liking others (that aren't done but have been thought about) even less. Perhaps in further development it will be possible to do things differently, but for now this is how it is.

Unfortunately, I don't have enough time to show my vision on the example of partial refactoring.

After the first launch, the Expert Advisor opens virtual and real positions, calculates, maybe, some indicators on price data. It is this information that makes up the state of the EA as a whole. If you now reload the terminal, the EA should not only recognise the open positions as its own, but also restore all its virtual positions and the necessary calculation values. If the information about open positions can be obtained from the terminal, the information about virtual positions and calculation values must be saved by the Expert Advisor itself.

Imagine that the Expert Advisor works on two trading accounts of one broker. One of the terminals is switched off. During this time, the virtual positions have changed in the running one. How to make the behaviour of the Expert Advisor coincide with the terminal that did not stop working?

How I do it. I do not save virtual positions at all. When starting the Expert Advisor, all internal TSs are launched in the virtual tester before the current TimeCurrent. Thus, we get that in the reloaded EA all the data at the current moment coincide with the version without reloading.

By "reloading" we also mean the situation when there is a pause in the work of the Expert Advisor. For example, a long OrderSend or reping. That's why we need to request price data from the moment of the last request. And run them in a virtual machine.

![Yuriy Bykov](https://c.mql5.com/avatar/avatar_na2.png)

**[Yuriy Bykov](https://www.mql5.com/en/users/antekov)**
\|
3 Mar 2024 at 12:26

**fxsaber [#](https://www.mql5.com/ru/forum/463345#comment_52573859):**

I'm unfortunately tight on time to show my vision with an example of partial refactoring.

You already pay a lot of attention to reviewing other people's code, for which I thank you very much.

Imagine that an Expert Advisor works on two trading accounts of the same broker. One terminal is down. During this time, the virtual positions have changed in the running one. How can we make the behaviour of the Expert Advisor to match the terminal that did not stop working?

I have encountered such a situation, but it was an insignificant factor affecting the trading result. Besides, there were times when the terminal that "slept through" the closing of positions closed them at a more favourable price. Therefore, ensuring full identity was not an end in itself.

And if there are different brokers, even Expert Advisors working synchronously can show slightly different results due to small differences in quotes. Although one should strive for identity.

How I do it. I don't save virtual accounts at all. When launching an Expert Advisor, all internal TS are launched in the virtual tester before the current TimeCurrent. Thus, we get that in the reloaded EA all the data at the current moment coincide with the version without reloading.

This is an interesting approach, I have not considered this option before. If I understand it correctly, when using it, we must have a fixed date of the start of virtual trading, and it must be the same for different instances of Expert Advisors in different terminals. This is not difficult. And a virtual tester must be implemented or your ready-made one must be used. This is more complicated.

![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
3 Mar 2024 at 12:31

**Yuriy Bykov [#](https://www.mql5.com/ru/forum/463345#comment_52594130):**

a virtual tester must be implemented

You almost already have it: throw the "Tester" ticks into the already implemented virtual trade.

![Population optimization algorithms: Resistance to getting stuck in local extrema (Part I)](https://c.mql5.com/2/72/Population_optimization_algorithms__Resistance_to_getting_stuck_in_local_extrema__LOGO.png)[Population optimization algorithms: Resistance to getting stuck in local extrema (Part I)](https://www.mql5.com/en/articles/14352)

This article presents a unique experiment that aims to examine the behavior of population optimization algorithms in the context of their ability to efficiently escape local minima when population diversity is low and reach global maxima. Working in this direction will provide further insight into which specific algorithms can successfully continue their search using coordinates set by the user as a starting point, and what factors influence their success.

![Building A Candlestick Trend Constraint Model (Part 5): Notification System (Part II)](https://c.mql5.com/2/82/Building_A_Candlestick_Trend_Constraint_Model_Part_5__NEXT_LOGO_2.png)[Building A Candlestick Trend Constraint Model (Part 5): Notification System (Part II)](https://www.mql5.com/en/articles/14968)

Today, we are discussing a working Telegram integration for MetaTrader 5 Indicator notifications using the power of MQL5, in partnership with Python and the Telegram Bot API. We will explain everything in detail so that no one misses any point. By the end of this project, you will have gained valuable insights to apply in your projects.

![Developing a Replay System (Part 39): Paving the Path (III)](https://c.mql5.com/2/64/Desenvolvendo_um_sistema_de_Replay_dParte_39w_Pavimentando_o_Terreno_nIIIu_LOGO.png)[Developing a Replay System (Part 39): Paving the Path (III)](https://www.mql5.com/en/articles/11599)

Before we proceed to the second stage of development, we need to revise some ideas. Do you know how to make MQL5 do what you need? Have you ever tried to go beyond what is contained in the documentation? If not, then get ready. Because we will be doing something that most people don't normally do.

![Integrate Your Own LLM into EA (Part 4): Training Your Own LLM with GPU](https://c.mql5.com/2/82/Integrate_Your_Own_LLM_into_EA_Part_4____LOGO.png)[Integrate Your Own LLM into EA (Part 4): Training Your Own LLM with GPU](https://www.mql5.com/en/articles/13498)

With the rapid development of artificial intelligence today, language models (LLMs) are an important part of artificial intelligence, so we should think about how to integrate powerful LLMs into our algorithmic trading. For most people, it is difficult to fine-tune these powerful models according to their needs, deploy them locally, and then apply them to algorithmic trading. This series of articles will take a step-by-step approach to achieve this goal.

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/14246&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049264455193307197)

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