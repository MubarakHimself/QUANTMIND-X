---
title: Developing a multi-currency Expert Advisor (Part 3): Architecture revision
url: https://www.mql5.com/en/articles/14148
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:29:13.625930
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/14148&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049166276535887415)

MetaTrader 5 / Trading


### Introduction

In the previous articles, we started developing a multi-currency EA that works simultaneously with various trading strategies. The solution provided in the [second article](https://www.mql5.com/en/articles/14107) is already significantly different from the one presented in the [first](https://www.mql5.com/en/articles/14026) one. This indicates that we are still in search of the best options.

Let's try to look at the developed system as a whole, abstracting from the small details of the implementation, in order to understand ways to improve it. To do this, let us trace the albeit short, but still noticeable evolution of the system.

### First operation mode

We have allocated an EA object (of the _CAdvisor_ class or its descendants), which is an aggregator of trading strategy objects ( _CStrategy_ class or its descendants). At the beginning of the EA operation, the following happens in the _OnInit()_ handler:

- EA object is created.
- Objects of trading strategies are created and added to the EA in its array for trading strategies.

The following happens in the _OnTick()_ event handler:

- The _CAdvisor::Tick()_ method is called for the EA object.
- This method iterates through all strategies and calls their _CStrategy::Tick()_ method.
- The strategies within _CStrategy::Tick()_ perform all necessary operations to open and close market positions.

This can be represented schematically like this:

![Fig. 1. Operation mode from the first article](https://c.mql5.com/2/80/Fig1.en.drawio.png)

Fig. 1. Operation mode from the first article

The advantage of this mode was that, it was possible to make the EA work with other instances of trading strategies via a number of relatively simple operations provided that you had the source code of an EA following a certain trading strategy.

However, the main drawback quickly emerged: when combining several strategies, we have to reduce the size of the positions opened by each instance of the strategy to one degree or another. This may lead to the complete exclusion of some or even all strategy instances from trading. The more strategy instances we include in parallel work or the smaller the initial deposit, the more likely such an outcome is, since the minimum size of open market positions is fixed.

Also, when several strategy instances were working together, a situation was encountered when opposite positions of the same size were opened. In terms of total volume, this is equivalent to the absence of open positions, but swaps continued to accrue on opposing open positions.

### Second operation mode

To eliminate the shortcomings, we decided to move all operations with market positions to a separate place, removing the ability of trading strategies to directly open market positions. This somewhat complicates the reworking of ready-made strategies, but this is not a very big loss, since it is compensated by eliminating the main drawback of the first mode.

Two new entities appear in our operation mode: virtual positions ( _CVirtualOrder_ class) and a recipient of trading volumes from strategies ( _CReceiver_ class and its descendants).

At the beginning of the EA operation, the following happens in the _OnInit()_ handler:

- A recipient object is created.
- An EA object is created and the created recipient is passed to it.
- Objects of trading strategies are created and added to the EA in its array for trading strategies.
- Each strategy creates its own array of virtual position objects with the required number of these objects.

The following happens in the _OnTick()_ event handler:

- The _CAdvisor::Tick()_ method is called for the EA object.
- This method iterates through all strategies and calls their _CStrategy::Tick()_ method.
- The strategies within _CStrategy::Tick()_ perform all necessary operations to open and close virtual positions. If some event occurs related to a change in the composition of open virtual positions, the strategy remembers that changes by setting a flag.
- If at least one strategy has set a change flag, then the recipient launches the method for adjusting open volumes of market positions. If the adjustment is successful, the change flag for all strategies is reset.

This can be represented schematically like this:

![Fig. 2. Operation mode from the second article](https://c.mql5.com/2/80/Fig2.en.drawio.png)

Fig. 2. Operation mode from the second article

In this operation mode, we will no longer be faced with the fact that some strategy instance does not in any way affect the size of open market positions. On the contrary, even an instance that opens a very small virtual volume can become that very drop that overwhelms the total volume of virtual positions from several strategy instances beyond the minimum allowable volume of a market position. The real market position will be opened in this case.

Along the way, we received other pleasant changes, including some possible savings on swaps, less load on the deposit, less observed drawdown and improved trading quality assessment (Sharpe ratio, profit factor).

While testing the second mode, I realized the following:

- Each strategy first handles already open virtual positions to determine the triggered StopLoss and TakeProfit levels. If any of the levels has been reached, then such a virtual position is closed. Therefore, this handling was immediately relocated to the _CVirtualOrder_ class method. But this solution still seems to be an insufficient generalization.
- We have expanded the composition of the base classes by adding new required entities. If we do not want to handle virtual positions, we can still use such base classes, simply passing "empty" objects to them. For example, we can create the _CReceiver_ class object, which contains only empty stub methods. But this also seems more like a temporary solution that needs to be reworked.
- We have added new methods and the property for base class tracking changes in the composition of open virtual positions to the _CStrategy_ base class, which spilled over into the use of these methods in the _CAdvisor_ base class. Again, this looks like a step towards narrowing the possibilities and imposing an overly specific implementation in the base class.
- The _CStrategy_ base class received the _Volume()_ method returning the total volume of open virtual positions, since the developed _CVolumeReceiver_ recipient class needed data on open virtual volumes of each strategy. However, by doing so, we have cut off the ability to open virtual positions for several symbols within one instance of a trading strategy. In this case, the total volume loses its meaning. This solution is suitable for testing single-symbol strategies, but nothing more.
- We used the array for storing the pointers to EA strategies in the _CReceiver_ class, so that the recipient can use them to find out the open virtual volume of the strategies. This led to duplication of the code that fills the strategy arrays in the EA and the receiver.
- Each strategy opens positions on a single symbol: when adding a recipient to the array of strategies, the strategy is asked for its symbol, and it is added to the array of used symbols. We used this feature in the _CVolumeReceiver_ class. The receiver then works only with the symbols added to its symbol array. We have already mentioned the limitation generated as a result of this behavior.

We will make the following changes based on the analysis of the listed shortcomings and discussion in the comments:

- Let's clean up the _CStrategy_ and _CAdvisor_ base classes as much as possible. Let's write the custom derived _CVirtualStrategy_ and _CVirtualAdvisor_ classes to create the development branch of EAs using virtual trading. They will now be our parent classes for specific strategies and EAs.
- It is time to expand the class of virtual positions. Each virtual position should receive a pointer to a recipient object, which will be responsible for bringing the virtual trading volume to the market, and a trading strategy object, which will make decisions about opening/closing a virtual position. This will allow for notification of interested entities about the operations of opening/closing virtual positions.
- Let's move the storage of all virtual positions into one array, instead of distributing them across several arrays belonging to strategy instances. Each strategy instance will request several elements from this array for its operation. The owner of the common array will be the recipient of trading volumes.
- There will be only one recipient in one EA. Therefore, we will implement it as Singleton. Its only instance will be available in all necessary places. We will formalize this implementation as the _CVirtualReceiver_ derived class.
- We will add the array of new entities - symbol recipients ( _CVirtualSymbolReceiver_ class) - into the recipient. Each symbol receiver will only work with the virtual positions of its symbol, which will be automatically attached to the symbol receiver when opened and unattached when closed.

Let's try to implement all this.

### Cleaning up base classes

Let's leave only the essentials in the _CStrategy_ and _CAdvisor_ base classes. In case of _CStartegy_, leave only the method for handling the OnTick event and get the following concise code:

```
//+------------------------------------------------------------------+
//| Base class of the trading strategy                               |
//+------------------------------------------------------------------+
class CStrategy {
public:
   virtual void      Tick() = 0; // Handle OnTick events
};
```

Everything else will be located in the class descendants.

In the _CAdvisor_ base class, include a small file _Macros.mqh_, which contains several useful macros for performing operations with regular arrays:

- _APPEND(A, V)_ — add V element to the end of the A array;
- _FIND(A, V, I)_ — write the A array element, equal to A, to I variable. If the element is not found, then -1 is stored in the I variable;
- _ADD(A, V)_ — add V element to the end of the A array if such an element is not already in the array;
- _FOREACH(A, D)_ — loop through the A array indices (the index will be in the i local variable) performing D actions in the body;
- _REMOVE\_AT(A, I)_ — remove an element from the A array at an I index position, shifting subsequent elements and reducing the array size;
- _REMOVE(A, V)_ — remove an element equal to V from the A array

```
// Useful macros for array operations
#ifndef __MACROS_INCLUDE__
#define APPEND(A, V)    A[ArrayResize(A, ArraySize(A) + 1) - 1] = V;
#define FIND(A, V, I)   { for(I=ArraySize(A)-1;I>=0;I--) { if(A[I]==V) break; } }
#define ADD(A, V)       { int i; FIND(A, V, i) if(i==-1) { APPEND(A, V) } }
#define FOREACH(A, D)   { for(int i=0, im=ArraySize(A);i<im;i++) {D;} }
#define REMOVE_AT(A, I) { int s=ArraySize(A);for(int i=I;i<s-1;i++) { A[i]=A[i+1]; } ArrayResize(A, s-1);}
#define REMOVE(A, V)    { int i; FIND(A, V, i) if(i>=0) REMOVE_AT(A, i) }
#define __MACROS_INCLUDE__
#endif
//+------------------------------------------------------------------+
```

These macros will be used in other files, as this allows us to make the code more compact but readable and avoid calling additional functions.

We will remove all places, where the recipient was encountered, from the _CAdvisor_ class, as well as leave only the call of the corresponding strategy handlers in the OnTick event handling method. We will receive the following code:

```
#include "Macros.mqh"
#include "Strategy.mqh"

//+------------------------------------------------------------------+
//| EA base class                                                    |
//+------------------------------------------------------------------+
class CAdvisor {
protected:
   CStrategy         *m_strategies[];  // Array of trading strategies
public:
                    ~CAdvisor();                // Destructor
   virtual void      Tick();                    // OnTick event handler
   virtual void      Add(CStrategy *strategy);  // Method for adding a strategy
};

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
void CAdvisor::~CAdvisor() {
// Delete all strategy objects
   FOREACH(m_strategies, delete m_strategies[i]);
}

//+------------------------------------------------------------------+
//| OnTick event handler                                             |
//+------------------------------------------------------------------+
void CAdvisor::Tick(void) {
// Call OnTick handling for all strategies
   FOREACH(m_strategies, m_strategies[i].Tick());
}

//+------------------------------------------------------------------+
//| Strategy adding method                                           |
//+------------------------------------------------------------------+
void CAdvisor::Add(CStrategy *strategy) {
   APPEND(m_strategies, strategy);  // Add the strategy to the end of the array
}
//+------------------------------------------------------------------+
```

These classes will remain in the _Strategy.mqh_ and _Advisor.mqh_ files in the current folder.

Now let's move the necessary code to the derived strategy and EA classes, which should work with virtual positions.

Create the _CVirtualStrategy_ class inherited from _CStrategy_. Let's add the following fields and methods to it:

- array of virtual positions (orders);
- total number of open positions and orders;
- method for counting open virtual positions and orders;
- methods for handling opening/closing a virtual position (order).

For now, the methods for handling opening/closing virtual positions will simply call the method for recalculating open virtual positions, which will update the _m\_ordersTotal_ field value. There is no need to perform any other actions yet. Probably, we will have to do that later. Therefore, for now these methods are made separate from the method of counting open virtual positions.

```
#include "Strategy.mqh"
#include "VirtualOrder.mqh"

//+------------------------------------------------------------------+
//| Class of a trading strategy with virtual positions               |
//+------------------------------------------------------------------+
class CVirtualStrategy : public CStrategy {
protected:
   CVirtualOrder     *m_orders[];   // Array of virtual positions (orders)
   int               m_ordersTotal; // Total number of open positions and orders

   virtual void      CountOrders(); // Calculate the number of open positions and orders

public:
   virtual void      OnOpen();      // Event handler for opening a virtual position (order)
   virtual void      OnClose();     // Event handler for closing a virtual position (order)
};

//+------------------------------------------------------------------+
//| Counting open virtual positions and orders                       |
//+------------------------------------------------------------------+
void CVirtualStrategy::CountOrders() {
   m_ordersTotal = 0;
   FOREACH(m_orders, if(m_orders[i].IsOpen()) { m_ordersTotal += 1; })
}

//+------------------------------------------------------------------+
//| Event handler for opening a virtual position (order)             |
//+------------------------------------------------------------------+
void CVirtualStrategy::OnOpen() {
   CountOrders();
}

//+------------------------------------------------------------------+
//| Event handler for closing a virtual position (order)             |
//+------------------------------------------------------------------+
void CVirtualStrategy::OnClose() {
   CountOrders();
}
```

Save this code in the _VirtualStrategy.mqh_ file of the current folder.

Since we removed working with the recipient from the _CAdvisor_ base class, then it should be transferred to our new _CVirtualAdvisor_ child class. In this class, we will add the _m\_receiver_ field for storing the pointer to the object of the recipient of trading volumes.

In the constructor, the field will be initialized with the pointer to the only possible recipient object, which will be created exactly when the _CVirtualReceiver::Instance()_ static method is called. The destructor will make sure the object is properly deleted.

We will also add new actions in the OnTick event handler. Before launching the handlers for this event in the strategies, we will first launch the handler for this event in the recipient. After the event is handled by the strategies, we will launch the recipient's method that adjusts the open volumes. If the recipient is now the owner of all virtual positions, then they themselves are able to determine the presence of changes. Therefore, there is no implementation of tracking changes in the trading strategy class, so we are removing it not only from the base strategy class, but completely.

```
#include "Advisor.mqh"
#include "VirtualReceiver.mqh"

//+------------------------------------------------------------------+
//| Class of the EA handling virtual positions (orders)              |
//+------------------------------------------------------------------+
class CVirtualAdvisor : public CAdvisor {
protected:
   CVirtualReceiver  *m_receiver; // Receiver object that brings positions to the market

public:
                     CVirtualAdvisor(ulong p_magic = 1); // Constructor
                    ~CVirtualAdvisor();                  // Destructor
   virtual void      Tick() override;                    // OnTick event handler

};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CVirtualAdvisor::CVirtualAdvisor(ulong p_magic = 1) :
// Initialize the receiver with a static receiver
   m_receiver(CVirtualReceiver::Instance(p_magic)) {};

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
void CVirtualAdvisor::~CVirtualAdvisor() {
   delete m_receiver;         // Remove the recipient
}

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
}
//+------------------------------------------------------------------+
```

Save this code in the _VirtualAdvisor.mqh_ file of the current folder.

### Expanding the class of virtual positions

The virtual position class receives the pointer to the _m\_receiver_ and _m\_strategy_ objects. The values for these fields will have to be passed through the constructor parameters, so we will make changes to it as well. I also added a couple of getters for the private properties of the virtual position: _Id()_ and _Symbol()_. Let's show the added code in the class description:

```
//+------------------------------------------------------------------+
//| Class of virtual orders and positions                            |
//+------------------------------------------------------------------+
class CVirtualOrder {
private:
//--- Static fields...

//--- Related recipient objects and strategies
   CVirtualReceiver  *m_receiver;
   CVirtualStrategy  *m_strategy;

//--- Order (position) properties ...

//--- Closed order (position) properties ...

//--- Private methods

public:
                     CVirtualOrder(
      CVirtualReceiver *p_receiver,
      CVirtualStrategy *p_strategy
   );                                  // Constructor

//--- Methods for checking the position (order) status ...


//--- Methods for receiving position (order) properties ...
   ulong             Id() {            // ID
      return m_id;
   }
   string            Symbol() {        // Symbol
      return m_symbol;
   }

//--- Methods for handling positions (orders) ...

};
```

In the constructor implementation, we simply added two lines to the initialization list to set the values of new fields from the constructor parameters:

```
CVirtualOrder::CVirtualOrder(CVirtualReceiver *p_receiver, CVirtualStrategy *p_strategy) :
// Initialization list
   m_id(++s_count),  // New ID = object counter + 1
   m_receiver(p_receiver),
   m_strategy(p_strategy),
   ...,
   m_point(0) {
}
```

Notification of the recipient and strategy should only occur when a virtual position is opened or closed. This only happens in the _Open()_ and _Close()_ methods, so let's add a little code to them:

```
//+------------------------------------------------------------------+
//| Open a virtual position                                          |
//+------------------------------------------------------------------+
bool CVirtualOrder::Open(...) {
   // If the position is already open, then do nothing ...

   if(s_symbolInfo.Name(symbol)) {  // Select the desired symbol
      // Update information about current prices ...

      // Initialize position properties ...

      // Depending on the direction, set the opening price, as well as the SL and TP levels ...

      // Notify the recipient and the strategy that the position (order) is open
      m_receiver.OnOpen(GetPointer(this));
      m_strategy.OnOpen();

      ...

      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Close a position                                                 |
//+------------------------------------------------------------------+
void CVirtualOrder::Close() {
   if(IsOpen()) { // If the position is open
      ...
      // Define the closure reason to be displayed in the log ...

      // Save the close price depending on the type ...

      // Notify the recipient and the strategy that the position (order) is open
      m_receiver.OnClose(GetPointer(this));
      m_strategy.OnClose();
   }
}
```

Pass the pointer to the current virtual position object to the _OnOpen()_ and _OnClose()_ recipient handlers. There has not yet been a need for this in strategy handlers, so they are implemented without a parameter.

This code remains in the current folder in the file with the same name — _VirtualOrder.mqh_.

### Implementing a new recipient

Let's start implementing the _CVirtualReceiver_ receiver class to ensure the uniqueness of an instance of a given class. To do this, we will use a standard design pattern called Singleton. We will need to:

- make the class constructor non-public;
- add a static class field that stores the pointer to the class object;
- add a static method that creates, if absent, one instance of this class or returns an already existing one.

```
//+------------------------------------------------------------------+
//| Class for converting open volumes to market positions (receiver) |
//+------------------------------------------------------------------+
class CVirtualReceiver : public CReceiver {
protected:
// Static pointer to a single class instance
   static   CVirtualReceiver *s_instance;

   ...

   CVirtualReceiver(ulong p_magic = 0);   // Private constructor

public:
//--- Static methods
   static
   CVirtualReceiver  *Instance(ulong p_magic = 0);    // Singleton - creating and getting a single instance

   ...
};

// Initializing a static pointer to a single class instance
CVirtualReceiver *CVirtualReceiver::s_instance = NULL;

//+------------------------------------------------------------------+
//| Singleton - creating and getting a single instance               |
//+------------------------------------------------------------------+
CVirtualReceiver* CVirtualReceiver::Instance(ulong p_magic = 0) {
   if(!s_instance) {
      s_instance = new CVirtualReceiver(p_magic);
   }
   return s_instance;
}
```

Next, add the _m\_orders_ array for storing all virtual positions to the class. Each strategy instance will request a certain number of virtual positions from the recipient. To do this, add the _Get()_ static method, which will create the required number of virtual position objects, adding pointers to them to the recipient array and the strategy virtual position array:

```
class CVirtualReceiver : public CReceiver {
protected:
   ...
   CVirtualOrder     *m_orders[];         // Array of virtual positions

   ...

public:
//--- Static methods
   ...
   static void       Get(CVirtualStrategy *strategy,
                         CVirtualOrder *&orders[],
                         int n); // Allocate the necessary amount of virtual positions to the strategy
   ...
};

...

//+------------------------------------------------------------------+
//| Allocate the necessary amount of virtual positions to strategy   |
//+------------------------------------------------------------------+
static void CVirtualReceiver::Get(CVirtualStrategy *strategy,   // Strategy
                                  CVirtualOrder *&orders[],     // Array of strategy positions
                                  int n                         // Required number
                                 ) {
   CVirtualReceiver *self = Instance();   // Receiver singleton
   ArrayResize(orders, n);                // Expand the array of virtual positions
   FOREACH(orders,
           orders[i] = new CVirtualOrder(self, strategy); // Fill the array with new objects
           APPEND(self.m_orders, orders[i])) // Register the created virtual position
   ...
}
```

Now it is time to add the array for pointers to symbol recipient objects (the _CVirtualSymbolReceiver_ class) into the class. This class has not yet been created, but we already understand what it should do - directly open and close market positions in accordance with virtual volumes for a single symbol. Therefore, we can say that the number of symbol recipient objects will be equal to the number of different symbols used in the EA. We will make its class a descendant of _CReceiver_, so it will have the _Correct()_ method, which does the main useful work. We will also add the necessary auxiliary methods.

Let's leave this for later and now get back to the _CVirtualReceiver_ class and add the virtual override of the _Correct()_ method to it.

```
class CVirtualReceiver : public CReceiver {
protected:
   ...
   CVirtualSymbolReceiver *m_symbolReceivers[];       // Array of recipients for individual symbols

public:
   ...
//--- Public methods
   virtual bool      Correct() override;              // Adjustment of open volumes
};
```

The implementation of the _Correct()_ method is now quite simple, since we transfer the main work to a lower level of the hierarchy. For now, it is sufficient for us to simply loop through all the symbol recipients and call their _Correct()_ method.

To reduce the number of unnecessary calls, we will add a preliminary check that trading is now generally allowed by adding the _IsTradeAllowed()_ method, which answers the question. We will also add the _m\_isChanged_ class field, which should serve as a flag of changes in open virtual positions. We will also check it before calling for an adjustment.

```
class CVirtualReceiver : public CReceiver {
   ...
   bool              m_isChanged;         // Are there any changes in open positions?
   ...
   bool              IsTradeAllowed();    // Is trading available?

public:
   ...

   virtual bool      Correct() override;  // Adjustment of open volumes
};
//+------------------------------------------------------------------+
//| Adjust open volumes                                              |
//+------------------------------------------------------------------+
bool CVirtualReceiver::Correct() {
   bool res = true;
   if(m_isChanged && IsTradeAllowed()) {
      // If there are changes, then we call the adjustment of the recipients of individual symbols
      FOREACH(m_symbolReceivers, res &= m_symbolReceivers[i].Correct());
      m_isChanged = !res;
   }
   return res;
}
```

In the _IsTradeAllowed()_ method, check the status of the terminal and trading account to determine whether it is possible to conduct real trading:

```
//+------------------------------------------------------------------+
//| Is trading available?                                            |
//+------------------------------------------------------------------+
bool CVirtualReceiver::IsTradeAllowed() {
   return (true
           && MQLInfoInteger(MQL_TRADE_ALLOWED)
           && TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)
           && AccountInfoInteger(ACCOUNT_TRADE_EXPERT)
           && AccountInfoInteger(ACCOUNT_TRADE_ALLOWED)
           && TerminalInfoInteger(TERMINAL_CONNECTED)
          );
}
```

We applied the change flag in the _Correct()_ method. The flag was reset if the volume adjustment was successful. But where should this flag be set? Obviously, this should happen if any virtual position is opened or closed. In the _CVirtualOrder_ class, we specifically added the calls for OnOpen() and OnClose() methods, not yet present in the _CVirtualReceiver_ class, to the open/close methods. We will set the changes flag in them.

In addition, we should notify the desired symbol recipient about the changes in these handlers. When opening the very first virtual position for a certain symbol, the corresponding symbol recipient does not yet exist, so we need to create and notify it. During subsequent operations of opening/closing virtual positions for a given symbol, there is already a corresponding symbol recipient, so you just need to notify it.

```
class CVirtualReceiver : public CReceiver {
   ...

public:
   ...

//--- Public methods
   void              OnOpen(CVirtualOrder *p_order);  // Handle virtual position opening
   void              OnClose(CVirtualOrder *p_order); // Handle virtual position closing
   ...
};

//+------------------------------------------------------------------+
//| Handle opening a virtual position                                |
//+------------------------------------------------------------------+
void CVirtualReceiver::OnOpen(CVirtualOrder *p_order) {
   string symbol = p_order.Symbol();      // Define position symbol
   CVirtualSymbolReceiver *symbolReceiver;
   int i;
   FIND(m_symbolReceivers, symbol, i);    // Search for the symbol recipient

   if(i == -1) {
      // If not found, then create a new recipient for the symbol
      symbolReceiver = new CVirtualSymbolReceiver(m_magic, symbol);
      // and add it to the array of symbol recipients
      APPEND(m_symbolReceivers, symbolReceiver);
   } else {
      // If found, then take it
      symbolReceiver = m_symbolReceivers[i];
   }

   symbolReceiver.Open(p_order); // Notify the symbol recipient about the new position
   m_isChanged = true;           // Remember that there are changes
}

//+------------------------------------------------------------------+
//| Handle closing a virtual position                                |
//+------------------------------------------------------------------+
void CVirtualReceiver::OnClose(CVirtualOrder *p_order) {
   string symbol = p_order.Symbol();   // Define position symbol
   int i;
   FIND(m_symbolReceivers, symbol, i); // Search for the symbol recipient

   if(i != -1) {
      m_symbolReceivers[i].Close(p_order);   // Notify the symbol recipient about closing a position
      m_isChanged = true;                    // Remember that there are changes
   }
}
```

In addition to opening/closing virtual positions based on trading strategy signals, they can be closed when StopLoss or TakeProfit levels are reached. In the _CVirtualOrder_ class, we have the _Tick()_ method specifically for this. It checks the levels and closes the virtual position if necessary. It should be called at every tick and for all virtual positions. This is exactly what the _Tick()_ method in the _CVirtualReceiver_ class will do. Let's add the class:

```
class CVirtualReceiver : public CReceiver {
   ...

public:
   ...

//--- Public methods
   void              Tick();     // Handle a tick for the array of virtual orders (positions)
   ...
};

//+------------------------------------------------------------------+
//| Handle a tick for the array of virtual orders (positions)        |
//+------------------------------------------------------------------+
void CVirtualReceiver::Tick() {
   FOREACH(m_orders, m_orders[i].Tick());
}
```

Finally, take care of correctly freeing the memory allocated for virtual position objects. Since they are all in the _m\_orders_ array, add a destructor, in which we will delete them:

```
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CVirtualReceiver::~CVirtualReceiver() {
   FOREACH(m_orders, delete m_orders[i]); // Remove virtual positions
}
```

Save the resulting code in the _VirtualReceiver.mqh_ file of the current folder.

### Implementing a symbol receiver

It remains to implement the last class _CVirtualSymbolReceiver_, so that the operation mode takes on a finished form suitable for use. We will take its main content from the _CVolumeReceiver_ class from the previous article, removing places related to determining the symbol of each virtual position and enumerating symbols while performing the adjustment.

Objects of this class will also have their own arrays of pointers to virtual position objects, but here their composition will constantly change. We will require that this array contains only open virtual positions. Then it becomes clear what to do when opening and closing a virtual position: as soon as the virtual position becomes open, we should add it to the array of the corresponding symbol recipient, and as soon as it is closed, remove it from the array.

It will also be convenient for us to have a flag of the presence of changes in the composition of open virtual positions. This will help avoid unnecessary checks on every tick.

Let's add fields for a symbol, an array of positions and a flag of changes, as well as two methods for handling opening/closing, to the class:

```
class CVirtualSymbolReceiver : public CReceiver {
   string            m_symbol;         // Symbol
   CVirtualOrder     *m_orders[];      // Array of open virtual positions
   bool              m_isChanged;      // Are there any changes in the composition of virtual positions?

   ...

public:
   ...
   void              Open(CVirtualOrder *p_order);    // Register opening a virtual position
   void              Close(CVirtualOrder *p_order);   // Register closing a virtual position
   ...
};
```

The implementation of these methods itself is trivial: we add/remove the passed virtual position from the array and set the flag for the presence of changes.

```
//+------------------------------------------------------------------+
//| Register opening a virtual position                              |
//+------------------------------------------------------------------+
void CVirtualSymbolReceiver::Open(CVirtualOrder *p_order) {
   APPEND(m_orders, p_order); // Add a position to the array
   m_isChanged = true;        // Set the changes flag
}

//+------------------------------------------------------------------+
//| Register closing a virtual position                              |
//+------------------------------------------------------------------+
void CVirtualSymbolReceiver::Close(CVirtualOrder *p_order) {
   REMOVE(m_orders, p_order); // Remove a position from the array
   m_isChanged = true;        // Set the changes flag
}
```

We also need to search for the desired symbol recipient by a symbol name. In order to use the normal linear search algorithm from the _FIND(A,V,I)_ macro, let's add an overloaded operator comparing the symbol recipient to the string that returns 'true' if the instance symbol matches the passed string:

```
class CVirtualSymbolReceiver : public CReceiver {
   ...

public:
   ...
   bool              operator==(const string symbol) {// Operator for comparing by a symbol name
      return m_symbol == symbol;
   }
   ...
};
```

Here is a complete description of the _CVirtualSymbolReceiver_ class. Find the specific implementation of all methods in the attached files.

```
class CVirtualSymbolReceiver : public CReceiver {
   string            m_symbol;         // Symbol
   CVirtualOrder     *m_orders[];      // Array of open virtual positions
   bool              m_isChanged;      // Are there any changes in the composition of virtual positions?

   bool              m_isNetting;      // Is this a netting account?

   double            m_minMargin;      // Minimum margin for opening

   CPositionInfo     m_position;       // Object for obtaining properties of market positions
   CSymbolInfo       m_symbolInfo;     // Object for getting symbol properties
   CTrade            m_trade;          // Object for performing trading operations

   double            MarketVolume();   // Volume of open market positions
   double            VirtualVolume();  // Volume of open virtual positions
   bool              IsTradeAllowed(); // Is trading by symbol available?

   // Required volume difference
   double            DiffVolume(double marketVolume, double virtualVolume);

   // Volume correction for the required difference
   bool              Correct(double oldVolume, double diffVolume);

   // Auxiliary opening methods
   bool              ClearOpen(double diffVolume);
   bool              AddBuy(double volume);
   bool              AddSell(double volume);

   // Auxiliary closing methods
   bool              CloseBuyPartial(double volume);
   bool              CloseSellPartial(double volume);
   bool              CloseHedgingPartial(double volume, ENUM_POSITION_TYPE type);
   bool              CloseFull();

   // Check margin requirements
   bool              FreeMarginCheck(double volume, ENUM_ORDER_TYPE type);

public:
                     CVirtualSymbolReceiver(ulong p_magic, string p_symbol);  // Constructor
   bool              operator==(const string symbol) {// Operator for comparing by a symbol name
      return m_symbol == symbol;
   }
   void              Open(CVirtualOrder *p_order);    // Register opening a virtual position
   void              Close(CVirtualOrder *p_order);   // Register closing a virtual position

   virtual bool      Correct() override;              // Adjustment of open volumes
};
```

Save this code in the _VirtualSymbolReceiver.mqh_ file of the current folder.

### Comparing results

The resulting operation mode can be represented as follows:

![Fig. 3. Operation mode from the current article](https://c.mql5.com/2/80/Fig3.en.drawio.png)

Fig. 3. Operation mode from the current article

Now comes the most interesting part. Let's compile the EA that uses nine instances of strategies with the same parameters as in the previous article. Let's perform test runs of a similar EA from the previous article and the one we have just compiled:

![](https://c.mql5.com/2/69/5221598330037.png)

Fig. 4. Results of the EA from the previous article

![](https://c.mql5.com/2/69/1689090023408.png)

Fig. 5. Results of the EA from the current article

In general, the results are almost the same. The balance graphs are generally indistinguishable. Small differences visible in the reports may be due to various reasons and will be analyzed further.

### Assessing further potential

In the discussion of the previous article, a logical question was asked: what are the most attractive trading results that can be obtained using the approach in question? So far, the graphs have shown a return of 20% over 5 years, which does not look particularly attractive.

For now, the answer to this question can be as follows. First, it is necessary to clearly separate the results due to the selected simple strategies and the results due to the implementation of their joint work.

The results of the first category will change when changing a simple strategy to another. It is clear that the better the results shown by individual instances of simple strategies, the better their overall result will be. The results presented here were obtained on one trading idea, and were initially determined precisely by its quality and suitability. We evaluate these results simply by the profit/drawdown ratio for the test interval.

The results of the second category are the comparative results of joint and solo work. Here the assessment is made based on other parameters: improving the linearity of the graph of the equity growth curve, reducing drawdown, and others. It is these results that seem more important, since there is hope to bring the not particularly outstanding results of the first category to an acceptable level with their help.

But for all results, it is advisable for us to first implement variable lot trading. Without this, it is more difficult to even estimate the profitability/drawdown ratio based on test results, although it is still possible.

Let's try to take a small initial deposit and select a new optimal value for the size of open positions for the maximum allowed drawdown of 50% for a period of 5 years (2018.01.01 — 2023.01.01). Below are the results of EA runs from this article with a different position size multiplier, but constant for all five years with an initial deposit of USD 1000. In the previous article, position sizes were calibrated to the deposit size of USD 10,000, so the initial _depoPart\__ value was reduced by about 10 times.

![](https://c.mql5.com/2/81/607356788956.png)

Fig. 6. Test results with different position sizes

We see that with minimal _depoPart\__ = 0.04, the EA did not open real positions, since their volume when recalculated in proportion to the balance is less than 0.01. But starting from the next multiplier value _depoPart\__ = 0.06, market positions were opened.

At the maximum _depoPart\__ = 0.4, we get the profit of approximately USD 22,800. However, the drawdown shown here is the relative drawdown encountered over the entire run. But 10% of 23,000 and 1000 are very different values. Therefore, we should definitely look at the results of a single run:

![](https://c.mql5.com/2/81/904549617495.png)

![](https://c.mql5.com/2/81/3335598245160.png)

Fig. 7. Test results at maximum _depoPart\__ = 0.4

As you can see, the drawdown of USD 1167 was actually reached, which at the time of achievement was only 9.99% of the current balance, but if the beginning of the test period was located immediately before this unpleasant moment, then we would have lost the entire deposit. Therefore, we cannot use this position size.

Let's look at the results when _depoPart\__ = 0.2

![](https://c.mql5.com/2/81/3828724725919.png)

![](https://c.mql5.com/2/81/6455754886724.png)

Fig. 8. Test results at _depoPart\__ = 0.2

Here the maximum drawdown did not exceed USD 494, that is, about 50% of the initial deposit of USD 1000. Therefore, with such a position size, even if the beginning of the period during the five years under consideration is chosen as poorly as possible, there will be no loss of the entire deposit.

With this position size, the test results for 1 year (2022) will be as follows:

![](https://c.mql5.com/2/81/2965310456086.png)

![](https://c.mql5.com/2/81/6308152357648.png)

Fig. 9. Test results for 2022 at _depoPart\__ = 0.2

So, with the maximum expected drawdown of approximately 50%, we see the profit of about 150% per year.

These results look encouraging, but there is a fly in the ointment. The results for 2023, which was not included in the parameter optimization, are noticeably worse:

![](https://c.mql5.com/2/81/1471680678158.png)

![](https://c.mql5.com/2/81/4457949280185.png)

Fig. 10. Test results for 2023 at _depoPart\__ = 0.2

Of course, we received the 40% profit in the test results at the end of the year, but there was no sustainable growth 8 out of 12 months. This problem seems to be the main one, and this series of articles will be devoted to considering different approaches to solving it.

### Conclusion

In this article, we have prepared for further development of the code by simplifying and optimizing the code from the previous part. We have addressed some previously identified deficiencies that could limit our ability to utilize various trading strategies. The test results showed that the new implementation works no worse than the previous one. The speed of the implementation remained unchanged, but it is possible that the growth will only appear with a multiple increase in the number of strategy instances.

To do this, we need to finally figure out how we will store the input parameters of strategies, how we will combine them into parameter libraries, and how we will select the best combinations from those that will be obtained as a result of optimizing single strategy instances.

We will continue working in the chosen direction in the next article.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14148](https://www.mql5.com/ru/articles/14148)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14148.zip "Download all attachments in the single ZIP archive")

[Advisor.mqh](https://www.mql5.com/en/articles/download/14148/advisor.mqh "Download Advisor.mqh")(4.3 KB)

[Macros.mqh](https://www.mql5.com/en/articles/download/14148/macros.mqh "Download Macros.mqh")(2.28 KB)

[Receiver.mqh](https://www.mql5.com/en/articles/download/14148/receiver.mqh "Download Receiver.mqh")(2.55 KB)

[SimpleVolumesExpert.mq5](https://www.mql5.com/en/articles/download/14148/simplevolumesexpert.mq5 "Download SimpleVolumesExpert.mq5")(7.4 KB)

[SimpleVolumesExpertSingle.mq5](https://www.mql5.com/en/articles/download/14148/simplevolumesexpertsingle.mq5 "Download SimpleVolumesExpertSingle.mq5")(7.29 KB)

[SimpleVolumesStrategy.mqh](https://www.mql5.com/en/articles/download/14148/simplevolumesstrategy.mqh "Download SimpleVolumesStrategy.mqh")(26.51 KB)

[Strategy.mqh](https://www.mql5.com/en/articles/download/14148/strategy.mqh "Download Strategy.mqh")(1.73 KB)

[VirtualAdvisor.mqh](https://www.mql5.com/en/articles/download/14148/virtualadvisor.mqh "Download VirtualAdvisor.mqh")(4.65 KB)

[VirtualOrder.mqh](https://www.mql5.com/en/articles/download/14148/virtualorder.mqh "Download VirtualOrder.mqh")(25.3 KB)

[VirtualReceiver.mqh](https://www.mql5.com/en/articles/download/14148/virtualreceiver.mqh "Download VirtualReceiver.mqh")(15.12 KB)

[VirtualStrategy.mqh](https://www.mql5.com/en/articles/download/14148/virtualstrategy.mqh "Download VirtualStrategy.mqh")(4.43 KB)

[VirtualSymbolReceiver.mqh](https://www.mql5.com/en/articles/download/14148/virtualsymbolreceiver.mqh "Download VirtualSymbolReceiver.mqh")(34.11 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/468569)**
(12)


![Yuriy Bykov](https://c.mql5.com/avatar/avatar_na2.png)

**[Yuriy Bykov](https://www.mql5.com/en/users/antekov)**
\|
16 Feb 2024 at 18:53

**fxsaber [#](https://www.mql5.com/ru/forum/462536#comment_52306480):**

It would be a good idea to add an on/off strategy mask to account for the volumetrician (recipient of virtual volumes).

For example, you need to switch off some TS from the portfolio for a while: it continues to trade virtually, but does not affect the real environment. Similarly with its reverse switching on.

It is not difficult to realise such a thing, but without using it, you will need clear on/off criteria for each strategy. This is a more complex task, I haven't approached it yet; probably I won't.

![Yuriy Bykov](https://c.mql5.com/avatar/avatar_na2.png)

**[Yuriy Bykov](https://www.mql5.com/en/users/antekov)**
\|
16 Feb 2024 at 18:58

**fxsaber [#](https://www.mql5.com/ru/forum/462536#comment_52306616):**

And there must be a competent incorporation of something like this somewhere.

```
CAdvisor *m_advisors[];  // Массив виртуальных портфелей
```

with its own mages.

There are no plans for that. Merging into portfolios will happen at an intermediate level between CAdvisor and CStrategy. There is a draft solution, but it is likely to change a lot during the ongoing refactoring.

![Yuriy Bykov](https://c.mql5.com/avatar/avatar_na2.png)

**[Yuriy Bykov](https://www.mql5.com/en/users/antekov)**
\|
16 Feb 2024 at 19:00

**fxsaber trading session.**
**The flag that volume synchronisation was successful can save you.**

It seems to be already there:

```
class CVirtualSymbolReceiver : public CReceiver {
  ...
   bool              m_isChanged;      // Есть ли изменения в составе виртуальных позиций
```

It is reset only when the required real volume is successfully opened for each symbol.

![Yuriy Bykov](https://c.mql5.com/avatar/avatar_na2.png)

**[Yuriy Bykov](https://www.mql5.com/en/users/antekov)**
\|
16 Feb 2024 at 19:03

**fxsaber [#](https://www.mql5.com/ru/forum/462536#comment_52307061):**

Putting a completely different entity into a virtual order is a questionable solution.

I really wanted to avoid this. I searched in every possible way how to leave CVirtualOrder unrelated to these entities. I liked what I got even less. That's why this is how it is for now.

![Cristian-bogdan Buzatu](https://c.mql5.com/avatar/avatar_na2.png)

**[Cristian-bogdan Buzatu](https://www.mql5.com/en/users/buza20)**
\|
17 Jun 2024 at 00:18

Fascinating! Can't wait for the next article on the matter. Good work!


![Neural networks made easy (Part 74): Trajectory prediction with adaptation](https://c.mql5.com/2/65/Neural_networks_are_easy_4Part_74w_Adaptive_trajectory_prediction____LOGO.png)[Neural networks made easy (Part 74): Trajectory prediction with adaptation](https://www.mql5.com/en/articles/14143)

This article introduces a fairly effective method of multi-agent trajectory forecasting, which is able to adapt to various environmental conditions.

![MQL5 Trading Toolkit (Part 1): Developing A Positions Management EX5 Library](https://c.mql5.com/2/80/MQL5_Trading_Toolkit_Part_1___LOGO.png)[MQL5 Trading Toolkit (Part 1): Developing A Positions Management EX5 Library](https://www.mql5.com/en/articles/14822)

Learn how to create a developer's toolkit for managing various position operations with MQL5. In this article, I will demonstrate how to create a library of functions (ex5) that will perform simple to advanced position management operations, including automatic handling and reporting of the different errors that arise when dealing with position management tasks with MQL5.

![Neural networks made easy (Part 75): Improving the performance of trajectory prediction models](https://c.mql5.com/2/68/Neural_Networks_Made_Easy_dPart_751_Improving_the_Performance_of_Trajectory_Prediction_Models____LOG.png)[Neural networks made easy (Part 75): Improving the performance of trajectory prediction models](https://www.mql5.com/en/articles/14187)

The models we create are becoming larger and more complex. This increases the costs of not only their training as well as operation. However, the time required to make a decision is often critical. In this regard, let us consider methods for optimizing model performance without loss of quality.

![A Step-by-Step Guide on Trading the Break of Structure (BoS) Strategy](https://c.mql5.com/2/80/A_Step-by-Step_Guide_on_Trading_the_Break_of_Structure____LOGO_.png)[A Step-by-Step Guide on Trading the Break of Structure (BoS) Strategy](https://www.mql5.com/en/articles/15017)

A comprehensive guide to developing an automated trading algorithm based on the Break of Structure (BoS) strategy. Detailed information on all aspects of creating an advisor in MQL5 and testing it in MetaTrader 5 — from analyzing price support and resistance to risk management

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/14148&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049166276535887415)

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