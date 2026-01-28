---
title: Universal Expert Advisor: Pending Orders and Hedging Support (Part 5)
url: https://www.mql5.com/en/articles/2404
categories: Integration
relevance_score: 4
scraped_at: 2026-01-23T17:50:18.111601
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/2404&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068706642565987495)

MetaTrader 5 / Examples


### Table of Contents

- [Introduction](https://www.mql5.com/en/articles/2404#intro)
- [Access to current prices through the Ask, Bid and Last methods. Overriding the Digits function](https://www.mql5.com/en/articles/2404#c1)
- [Support for hedging](https://www.mql5.com/en/articles/2404#c2)
- [Use of pending orders in earlier CStrategy versions](https://www.mql5.com/en/articles/2404#c3)
- [CPendingOrders and COrdersEnvironment for operations with pending orders](https://www.mql5.com/en/articles/2404#c4)
- [Reference to pending orders in the Expert Advisor code](https://www.mql5.com/en/articles/2404#c5)
- [The trading logic of the CImpulse Expert Advisor, which uses pending orders](https://www.mql5.com/en/articles/2404#c6)
- [The CImpulse strategy class](https://www.mql5.com/en/articles/2404#c7)
- [Analyzing the CImpulse strategy on various account types](https://www.mql5.com/en/articles/2404#c8)
- [Conclusion](https://www.mql5.com/en/articles/2404#exit)

### Introduction

The new article of the series devoted to the development of a Universal Expert Advisor provides further description of the functionality of the special trading engine, using which you can easily develop a trading robot that utilizes complex logic. The set of CStrategy classes keeps actively improving. New algorithms are being added to the engine to make trading even more efficient and simple. Further algorithm development mainly stems from the feedback of users who read this series of articles and ask questions in the article discussion topics or by sending personal messages. One of the most frequent questions was about the use of pending orders. Well, pending orders were not discussed in the previous articles, while the CStrategy engine did not provide a convenient tool for working with pending orders. This article introduces significant additions to the CStrategy trading engines. After all these changes, CStrategy now provides new tools for working with pending orders. We will discuss the details of changes later.

In addition, the new version of the MetaTrader platform 5 now supports bi-directional trading on accounts with the hedging option (see article ["MetaTrader 5 features hedging position accounting system"](https://www.mql5.com/en/articles/2299) for details). The code of CStrategy has been modified to cover these innovations and to operate correctly on new account types. Only a few changes have been made to add hedging support to the code, which indicates that the initially chosen approach is correct: changes and additions, no matter how global they are, do not hinder the operation of the trading engine. Moreover, new MetaTrader 5 tools, such as hedging support, offer many interesting opportunities, which can be implemented in the new versions of CStrategy.

In addition, CStrategy now supports intuitive and popular methods for obtaining the current Ask, Bid, and Last prices. These methods are described in the first chapter of this article.

The article contains a lot of information in order to cover all the innovations and changes both in CStrategy and MetaTrader 5. This article covers many of the topics that have not been discussed in the previous parts. I hope it will interest readers.

### **Access to current prices through the Ask, Bid and Last methods. Overriding the Digits function**

Traders often need to have access to current prices. Previous versions of CStrategy did not include special methods for such data access. Instead, it was assumed that the user will use standard functions to request current prices. For example, in order to find out the Ask price, the user needed to write the following code:

```
double ask = SymbolInfoDouble(ExpertSymbol(), SYMBOL_ASK);
int digits = SymbolInfoInteger(ExpertSymbol(), SYMBOL_DIGITS);
ask = NormalizeDouble(ask, digits);
```

Although you only need one function in order to receive the Ask price — SymbolInfoDouble — the received value should be normalized to the precision of the current instrument. So the actual receiving of the Ask prices requires more actions to be performed. The same refers to Bid and Last receiving algorithm.

In order to make the use of the strategy easier, three methods have been added to CStrategy: Ask(), Bid() and Last(). Each of them receives a corresponding price and normalizes it in accordance with the current symbol:

```
//+------------------------------------------------------------------+
//| Returns the Ask price.                                           |
//+------------------------------------------------------------------+
double CStrategy::Ask(void)
  {
   double ask = SymbolInfoDouble(ExpertSymbol(), SYMBOL_ASK);
   int digits = (int)SymbolInfoInteger(ExpertSymbol(), SYMBOL_DIGITS);
   ask = NormalizeDouble(ask, digits);
   return ask;
  }
//+------------------------------------------------------------------+
//| Returns the Bid price.                                           |
//+------------------------------------------------------------------+
double CStrategy::Bid(void)
  {
   double bid = SymbolInfoDouble(ExpertSymbol(), SYMBOL_BID);
   int digits = (int)SymbolInfoInteger(ExpertSymbol(), SYMBOL_DIGITS);
   bid = NormalizeDouble(bid, digits);
   return bid;
  }
//+------------------------------------------------------------------+
//| Returns the Last price.                                          |
//+------------------------------------------------------------------+
double CStrategy::Last(void)
  {
   double last = SymbolInfoDouble(ExpertSymbol(), SYMBOL_LAST);
   int digits = (int)SymbolInfoInteger(ExpertSymbol(), SYMBOL_DIGITS);
   last = NormalizeDouble(last, digits);
   return last;
  }
```

These methods are defined in a new version of the CStrategy class and are now available for use directly in the classes of derived strategies. We will use them in the strategy development examples.

In addition to the organization of access to Ask, Bid and Last via methods with appropriate names, CStrategy overrides the system **Digits** function. The function returns the number of decimal places for the current instrument. It may seem that such function overriding is meaningless, but it is not true. The working symbol of an Expert Advisor may differ from the symbol on which the executing module containing the strategies is running. In this case the call of the Digits system function can be misleading. It will return the number of digits of the symbol on which the EA is running, not of the working symbol. In order to avoid this confusion, the Digits function in CStartegy is overridden by the method with the same name. Any reference to this function actually calls this method. It returns the number of decimal places of the EA's working symbol. Here is the source code of this method:

```
//+------------------------------------------------------------------+
//| Returns the number of decimal places for the working             |
//| instrument                                                       |
//+------------------------------------------------------------------+
int CStrategy::Digits(void)
  {
   int digits = (int)SymbolInfoInteger(ExpertSymbol(), SYMBOL_DIGITS);
   return digits;
  }
```

You must be aware of this feature and understand the meaning of this overriding.

### Support for accounts with the hedging option

Support for accounts with the hedging option has recently been added to MetaTrader 5. On such accounts, a trader can have multiple open positions at the same time, both in the opposite (Buy and Sell) or same direction. In CStrategy, all operations with positions are processed in special handlers SupportBuy and SupportSell. Positions listed for the current strategy are transmitted one by one to these methods. It does not matter whether there is only one position or many positions. Positions can be opened on the same symbol or different symbols. The same mechanism is used to process and transmit these positions. Therefore there will be little changes in order to add hedging support. First of all, we need to change the RebuildPosition method. When the trade environment changes (new trades are executed), the method will re-arrange the list of positions. Re-arranging differs depending on the mode used. For netting accounts, the algorithm for selecting a position of a symbol will be used. For hedging account accounts, the index of a position in the list will be used.

Here is the previous version of the RebuildPosition method:

```
//+------------------------------------------------------------------+
//| Re-arranges the list of positions.                               |
//+------------------------------------------------------------------+
void CStrategy::RebuildPositions(void)
{
   ActivePositions.Clear();
   for(int i = 0; i < PositionsTotal(); i++)
   {
      string symbol = PositionGetSymbol(i);
      PositionSelect(symbol);
      CPosition* pos = new CPosition();
      ActivePositions.Add(pos);
   }
}
```

In the new version of RebuildPosition, two position selection algorithms are used, depending on the account type:

```
//+------------------------------------------------------------------+
//| Re-arranges the list of positions.                               |
//+------------------------------------------------------------------+
void CStrategy::RebuildPositions(void)
  {
   ActivePositions.Clear();
   ENUM_ACCOUNT_MARGIN_MODE mode=(ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE);
   if(mode!=ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
     {
      for(int i=0; i<PositionsTotal(); i++)
        {
         string symbol=PositionGetSymbol(i);
         PositionSelect(symbol);
         CPosition *pos=new CPosition();
         ActivePositions.Add(pos);
        }
     }
   else
     {
      for(int i=0; i<PositionsTotal(); i++)
        {
         ulong ticket=PositionGetTicket(i);
         PositionSelectByTicket(ticket);
         CPosition *pos=new CPosition();
         ActivePositions.Add(pos);
        }
     }
  }
```

Note that the same position class **CPosition** is used for both hedging and netting accounts. Once the position is selected, its properties can be accessed through the PositionGetInteger, PositionGetDouble and PositionGetString methods. It does not matter whether a hedging or a normal position is selected. Position access is the same in both cases. That is why it is possible to use the same position class CPosition on different account types.

There are no other methods that should be overridden inside the CStrategy class. CStrategy is designed so that the operation of strategies based on this engine depends on the _context_. This means that if a strategy works on accounts with the hedging option and opens multiple positions in one direction, it will manage these positions in parallel, treating each position as a separate CPosition class. If on the contrary, only one position can be open on an account, the strategy will only manage this position in the form of the same CPosition object.

In addition to changes in the RebuildPosition method, we also need to modify the internal contents of some CPosition methods. The class is located in the PositionMT5.mqh file, and it contains methods that are based on system function calls. Also, CPosition actively uses the standard trade class **CTrade**. The latest version of CTrade has been modified so as to allow using properties of hedging positions. For example, a hedging position can be closed by an opposite position, for which the new CTrade::PositionCloseBy method is called. Below are CPosition methods, the contents of which have been changed:

```
//+------------------------------------------------------------------+
//|                                                  PositionMT5.mqh |
//|                                 Copyright 2016, Vasiliy Sokolov. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2016, Vasiliy Sokolov."
#property link      "http://www.mql5.com"
#include <Object.mqh>
#include "Logs.mqh"
#include <Trade\Trade.mqh>
#include "Trailing.mqh"
//+------------------------------------------------------------------+
//| Active position class for classical strategies                   |
//+------------------------------------------------------------------+
class CPosition : public CObject
  {
   ...
  };
...
//+------------------------------------------------------------------+
//| Returns an absolute Stop Loss level for the current position.    |
//| If the Stop Loss level is not set, returns 0.0                   |
//+------------------------------------------------------------------+
double CPosition::StopLossValue(void)
{
   if(!IsActive())
      return 0.0;
   return PositionGetDouble(POSITION_SL);
}
//+------------------------------------------------------------------+
//| Sets an absolute stop loss level                                 |
//+------------------------------------------------------------------+
bool CPosition::StopLossValue(double sl)
{
   if(!IsActive())
      return false;
   return m_trade.PositionModify(m_id, sl, TakeProfitValue());
}
//+------------------------------------------------------------------+
//| Returns an absolute Stop Loss level for the current position.    |
//| If the Stop Loss level is not set, returns 0.0                   |
//+------------------------------------------------------------------+
double CPosition::TakeProfitValue(void)
{
   if(!IsActive())
      return 0.0;
   return PositionGetDouble(POSITION_TP);
}
//+------------------------------------------------------------------+
//| Sets an absolute stop loss level                                 |
//+------------------------------------------------------------------+
bool CPosition::TakeProfitValue(double tp)
  {
   if(!IsActive())
      return false;
   return m_trade.PositionModify(m_id, StopLossValue(), tp);
  }
//+------------------------------------------------------------------+
//| Closes the current position by market and sets a closing         |
//| comment equal to 'comment'                                       |
//+------------------------------------------------------------------+
bool CPosition::CloseAtMarket(string comment="")
  {
   if(!IsActive())
      return false;
   ENUM_ACCOUNT_MARGIN_MODE mode=(ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE);
   if(mode != ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
      return m_trade.PositionClose(m_symbol);
   return m_trade.PositionClose(m_id);
  }
//+------------------------------------------------------------------+
//| Returns current position volume.                                 |
//+------------------------------------------------------------------+
double CPosition::Volume(void)
  {
   if(!IsActive())
      return 0.0;
   return PositionGetDouble(POSITION_VOLUME);
  }
//+------------------------------------------------------------------+
//| Returns current profit of a position in deposit currency.        |
//+------------------------------------------------------------------+
double CPosition::Profit(void)
  {
   if(!IsActive())
      return 0.0;
   return PositionGetDouble(POSITION_PROFIT);
  }
//+------------------------------------------------------------------+
//| Returns true if the position is active.  Returns false           |
//| if otherwise.                                                    |
//+------------------------------------------------------------------+
bool CPosition::IsActive(void)
{
   ENUM_ACCOUNT_MARGIN_MODE mode=(ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE);
   if(mode!=ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
      return PositionSelect(m_symbol);
   else
      return PositionSelectByTicket(m_id);
}
//+------------------------------------------------------------------+
```

As you can see, the basis of these methods is the call of another IsActive method. The method returns true if the active position represented by the CPosition object really exists in the system. The method actually selects a position using one of the two methods depending on the account type. For classical accounts with the netting option, a position is selected using the PositionSelect function, for which you need to specify the symbol. For accounts with the hedging option, a position is selected based on its ticket using the new PositionSelectByTicket function. The selection result (true or false) is returned to the procedure that has called the method. This method sets the context for further operations with the position. There is no need to modify the trading algorithm, because all the trading functions of CPosition are based on the CTrade class. CTrade can modify, open and close both traditional and bi-directional positions.

### Use of pending orders in earlier CStrategy versions

Use of pending orders is an important part of many trading algorithms. After the release of the first version of the CStrategy trading engine, I received a lot of questions about the use of pending orders. In this and the following sections we will discuss trading though pending orders.

The CStrategy engine has originally been created without the support for pending orders. However, this does not mean that the strategy developed based on the CStrategy class cannot work with pending orders. In the first CSTrategy version, this could be done by using standard MetaTrader 5 functions, such as OrdersTotal() and OrderSelect().

For example, an Expert Advisor sets or modifies a previously set pending stop-order on each new bar so that its trigger price is 0.25% higher than the current price (to buy) or below it (to sell). The idea is that if the price makes a sudden movement (impulse) within a bar, then this order will be filled, and the Expert Advisor will enter during the strong price movement. If the movement is not strong enough, then the order will not be filled within the current bar. In this case it is necessary to move the order to a new level. The full implementation of this strategy is available below in sections describing the algorithm, which I called CImpulse. As clear from the system name, it is based on the impulse algorithm. It requires a single entry upon triggering of a pending order. As we already know, CStrategy contains special overridden methods BuyInit and SellInit for entering a position. Therefore, the algorithm of pending order use should be added to these methods. Without direct support from CStrategy, the code for Buy operations will be the following:

```
//+------------------------------------------------------------------+
//| The logic of operations with pending Buy orders.                 |
//+------------------------------------------------------------------+
void CMovingAverage::InitBuy(const MarketEvent &event)
  {
   if(!IsTrackEvents(event))return;                      // Handle only the required event!
   if(positions.open_buy > 0) return;                    // If there is at least one open position, no need to buy, as we have already bought!
   int buy_stop_total = 0;
   for(int i = OrdersTotal()-1; i >= 0; i--)
   {
      ulong ticket = OrderGetTicket(i);
      if(!OrderSelect(ticket))continue;
      ulong magic = OrderGetInteger(ORDER_MAGIC);
      if(magic != ExpertMagic())continue;
      string symbol = OrderGetString(ORDER_SYMBOL);
      if(symbol != ExpertSymbol())continue;
      ENUM_ORDER_TYPE order_type = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      if(order_type == ORDER_TYPE_BUY_STOP)
      {
         buy_stop_total++;
         Trade.OrderModify(ticket, Ask()*0.0025, 0, 0, 0);
      }
   }
   if(buy_stop_total == 0)
      Trade.BuyStop(MM.GetLotFixed(), Ask() + Ask()*0.0025, ExpertSymbol(), 0, 0, NULL);
  }
```

The IsTrackEvents method identifies that the received event corresponds to opening of a new bar on the current symbol. Then, the Expert Advisor checks the number of open positions in the buy direction. If there is at least one long position, the EA should not buy, and the logic is completed. Next the strategy checks all current pending orders. It loops through the indexes of pending orders. Each of the orders is selected by index, and then its magic and symbol are analyzed. If these two parameters correspond to the parameters of the Expert Advisor, it is considered that the order belongs to the current Expert Advisor, and the order counter is increased by one. The order is modified, its entry price is replaced with the current price + 0.25% above it. If there are no pending orders, which can be clear from the fact that the buy\_stop\_order counter is equal to zero, a new order is placed at a distance of 0.25% away from the current price.

Note that there is no position opening in InitBuy. CStrategy does not impose such a restriction on the position opening methods. So formally, any Expert Advisor logic can be added here. However, for proper event handling, the logic must be connected precisely with the opening of a position, either via pending or market orders.

As can be seen from this example, operations with pending orders are performed similarly to conventional positions. The main requirement is the same: it is necessary to divide the Buy and Sell logic by describing them in separate method BuyInit and SellInit. Only pending buy orders should be processed in BuyInit. Only pending sell orders should be processed in SellInit. The rest of the logic of operations with pending orders is similar to the classical scheme adopted in MetaTrader 4: searching through orders, selecting orders that belong to the current Expert Advisor, analyzing the trading environment and deciding to open or change an existing order.

### **CPendingOrders and COrdersEnvironment for operations with pending orders**

Standard MetaTrader 5 functions for working with pending orders provide a convenient system for a comprehensive and easy monitoring of pending orders. However, CStrategy is a set of object-oriented classes that allow creating a trading system. All actions performed in CStrategy are object-oriented, i.e they are performed by objects that in turn perform trading operations. This approach provides several advantages. Here are some of them.

- **Reduced size of custom source code.** Many actions, such as price normalization or preparation of receiver arrays for the CopyBuffer class functions are performed "behind the scenes", while the method provides then a ready result. Thus, you do not need to write additional procedures for result checking and other intermediate steps, which cannot be avoided if you work with the MetaTrader 5 system functions directly.
- **Platform-independence.** Since all the classes and methods provided are written in the formal MQL language, it is possible to obtain any property using a universal method. However, the actual implementation of this method will be different on different platforms. However this does not matter on the user level. It means that an Expert Advisor developed for one platform, in theory can be compiled to run on another platform. However, in practice there are many nuances, and we will not discuss platform-independence in this article.
- **Functionality.** A set of MQL functions provides basic functionality, by combining which you can create complex algorithms and useful functions. When these algorithms are included into a single library of classes, like CStartegy, this functionality becomes much easier to access, while the addition of new functions does not complicate its operation, because in such cases only new modules are added, which an EA developer may chose to use or not.

In order to stick to this approach, we decided to expand the functionality of CStartegy so as to provide a convenient object-oriented mechanism for operations with pending orders. This mechanism is represented by two classes: **CPendingOrder** and **COrdersEnvironment**. COrder provides a convenient object that contains all the properties of a pending order, which can be accessed through OrderGetInteger, OrderGetDouble and OrderGetString. The purpose of COrdersEnvironment will be explained later.

Suppose that the CPendingOrder object represents a pending order, which actually exists in the system. If you delete this pending order, then what should happen to the CPendingOrder object which represents the order? If the object remains after the deletion of the actual pending order, this will cause serious errors. The Expert Advisor will find the CPendingOrder and will erroneously "think" that the pending order still exists in the system. In order to avoid such errors, we must ensure the synchronization of the trading environment with the CStrategy object environment. In other words, we need to create a mechanism that would guarantee access to only those objects that really exist in the system. This is what the COrdersEnvironment class is doing. Its implementation is simple enough, and it allows the access to only those CPendingOrders objects, which represent actual pending orders.

The basis of the COrdersEnvironment class is comprised of the GetOrder and Total methods. The first order returns the CPendingOrders object which corresponds to a pending order with a particular index in the MetaTrader 5 system of pending orders. The second method returns the total number of pending orders. Now it is time to examine this class in detail. Here is the source code of the class:

```
//+------------------------------------------------------------------+
//| A class for operations with pending orders                       |
//+------------------------------------------------------------------+
class COrdersEnvironment
{
private:
   CDictionary    m_orders;         // The total number of all pending orders
public:
                  COrdersEnvironment(void);
   int            Total(void);
   CPendingOrder* GetOrder(int index);
};
//+------------------------------------------------------------------+
//| We need to know the current symbol and magic number of the EA    |
//+------------------------------------------------------------------+
COrdersEnvironment::COrdersEnvironment(void)
{
}
//+------------------------------------------------------------------+
//| Returns a pending order                                          |
//+------------------------------------------------------------------+
CPendingOrder* COrdersEnvironment::GetOrder(int index)
{
   ulong ticket = OrderGetTicket(index);
   if(ticket == 0)
      return NULL;
   if(!m_orders.ContainsKey(ticket))
      return m_orders.GetObjectByKey(ticket);
   if(OrderSelect(ticket))
      return NULL;
   CPendingOrder* order = new CPendingOrder(ticket);
   m_orders.AddObject(ticket, order);
   return order;
}
//+------------------------------------------------------------------+
//| Returns the number of pending orders                             |
//+------------------------------------------------------------------+
int COrdersEnvironment::Total(void)
{
   return OrdersTotal();
}
```

The Total method returns the number of pending orders, which currently exist in the system. The method is never wrong, because it returns the system value received from the OrdersTotal() function.

For the operation of the GetOrder method, it is necessary to specify the index of the pending order in the system. We always know the exact total number of orders through the Total method, and the index of a required order is also always known, and corresponds exactly to the actual index of a pending order in the system. Further, the GetOrder method receives an identifier of a pending order by its index. If an order has been removed for some reason, the order ID will be equal zero, and therefore the NULL constant will be returned to the Expert Advisor, which indicates that the order with the requested index cannot be found.

Each object created dynamically, requires to be explicitly removed by a special delete operator. Since GetOrder creates CPendingOrders objects dynamically using the new operator, these objects also need to be deleted. In order to eliminate the necessity to delete an object by a user after creation, we used a special technique — the object is located in a special dictionary container inside the COrdersEnvironment object. An element in the dictionary can be accessed by its unique key, which is an order ID in this case. In this case, if the order still exists, the previously created object representing this order has most likely been created and placed in the container. So this previously created object is returned by the GetOrder function. If it is the first call of the order with the specified ID, a new CPendingOrder object is created, after which it is added to the dictionary, and a reference to it is then returned to the user.

What do we have from the proposed approach? First of all, the GetOrder method ensures to only return the object that represents a really existing pending order. The system function OrderGetTicket is responsible for that. Secondly, the object will be created only if it has not yet been created. This saves additional computer resources. And finally, thirdly, the algorithm frees the user from the need to remove the resulting object. Since all objects are stored in the dictionary, they will be deleted automatically after the deinitialization of COrdersEnvironment.

### **Reference to pending orders in the Expert Advisor code**

Now it is time to rewrite the logic of operations with pending orders presented in the previous section "Trading through pending orders in earlier CStrategy versions". Here is the code with the CPendingOrder and COrdersEnvironment classes:

```
//+------------------------------------------------------------------+
//| We buy when the fast MA is above the slow one.                   |
//+------------------------------------------------------------------+
void CMovingAverage::InitBuy(const MarketEvent &event)
  {
   if(!IsTrackEvents(event))return;                      // Handle only the required event!
   if(positions.open_buy > 0) return;                    // If there is at least one open position, no need to buy, as we have already bought!
   int buy_stop_total = 0;
   for(int i = PendingOrders.Total()-1; i >= 0; i--)
     {
      CPendingOrder* Order = PendingOrders.GetOrder(i);
      if(Order == NULL || !Order.IsMain(ExpertSymbol(), ExpertMagic()))
         continue;
      if(Order.Type() == ORDER_TYPE_BUY_STOP)
       {
         buy_stop_total++;
         Order.Modify(Ask() + Ask()*0.0025);
       }
       //delete Order; No need to delete the Order object!
     }
   if(buy_stop_total == 0)
      Trade.BuyStop(MM.GetLotFixed(), Ask() + Ask()*0.0025, ExpertSymbol(), 0, 0, NULL);
  }
```

The PendingsOrders object is a COrdersEnvironment class. The search through pending orders starts the same ways as when using system functions. Then an attempt is made to obtain an order object in accordance with the pending order index equal to the _i_ variable. If the order has not been received for some reason or it belongs to another Expert Advisor, search through orders continues with a new order. In these cases the IsMain method of the CPendingorder object is used, which returns true if the order's symbol and magic number match the EA's symbol and Magic, which means that the order belongs to this Expert Advisor.

If the order type corresponds to ORDER\_TYPE\_BUY\_STOP, it means that the placed order has not triggered and its level should be modified by the formula: _current price + 0.25%_. Pending order modification is done by using the Modify method and its overridden version. It allows you to set the price and other parameters that you may want to change.

Note that after completing operations with a pending order, there is no need to delete the order object. The reference to the order should be left as it is. The PendingOrders container manages objects of the CPendingOrder type and does not require removal of the returned objects in user functions.

If there are no pending orders and the buy\_stop\_total counter is equal to zero, a new pending order is placed. A new order is placed by using the Trade module, which was described in the previous parts of the article.

In the object-oriented approach, the properties of a pending order are accessed through the appropriate methods of the CPendingOrders object. This access type reduces the code volume while preserving its reliability, since PendingOrders guarantees that only an existing order object will be received through it, i.e. the pending order really exists in the system.

### The trading logic of the CImpulse Expert Advisor, which uses pending orders

We have analyzed use of pending orders, and we can now create a full-fledged Expert Advisor that utilizes the possibilities of the trading engine. Our strategy will be based on entries during strong market movements in the direction of the movement, so it will be called **CImpulse**. At the opening of each new bar, the EA will measure a predefined distance from the current bar, expressed as a percentage. The EA will place pending BuyStop and SellStop orders at an equal distance from the current price. The distance is set as a percentage. If one of the orders trigger within one bar, it means that the price has moved quite a long distance, which indicates the market impulse. The order will be executed and will turn into an open position.

Open positions will be managed by a simple moving average. If the price returns to the Moving Average, the position will be closed. The screenshot below shows a typical entry into a long position upon triggering of a pending BuyStop order:

![](https://c.mql5.com/2/22/iImpulse.png)

Fig. 1. A long position of the CImpulse strategy.

On the above screenshot, the beginning of a bar is marked with a black triangle. At this bar beginning, two pending orders are placed at a distance of 0.1% from the bar open price. One of them — the BuyStop order — triggered. A new long position was opened. As soon as one of the bars closed below the moving average displayed as a red line, the Expert Advisor closed the position. In Fig. 1 the closure is displayed as a blue triangle.

If a pending does not trigger within one bar, it is moved to a new level calculated based on the new current price.

The described strategy has one specific feature of processing pending orders. The level of a BuyStop order can be lower than the current Moving Average. In this case, the position will be closed immediately after opening, because the current price is below the Moving Average. The same is true for short positions: the trigger price of a SellStop order may be higher than the Moving Average level. In order to avoid the immediate closure, we need to add an extra check of the Moving Average level in the BuyInit and SellInit methods. So the algorithm will only place pending BuyStop orders, if their price is above the Moving Average value. The same is true for SellStop: they will only be placed if their price is below the Moving Average.

We will also use the new feature of MetaTrader 5 — operation on hedging accounts. This feature means that both a long and a short position can be opened within a bar. However, the originally presented logic of the Expert Advisor, which implied the separation of long and short positions, makes it possible to leave the code of the trading strategy unchanged. Regardless of the number of positions, all short positions will be closed if the open price of the bar moves above the Moving Average, and long positions will be closed when the price falls below the Moving Average. In this case it does not matter if we are trading on hedging accounts or not.

First of all we need to open an account with the hedging option enabled, by checking an appropriate flag in the "Account opening" window:

![](https://c.mql5.com/2/23/open_acc__3.png)

Fig. 2. Opening an account with the hedging option

After opening an account, we can proceed to trade from it. First, we need to write the CImpulse class that implements the logic of the discussed trading algorithm.

### The CImpulse strategy class

Further, there is a listing of the CImpulse class that describes the logic of our new trading Expert Advisor. For a clearer description of operations with pending orders, this class is as simple as possible, and it does not contain procedures connected with logging, as well as special methods that parse the strategy parameters from an XML file:

```
//+------------------------------------------------------------------+
//|                                                      Impulse.mqh |
//|                                 Copyright 2016, Vasiliy Sokolov. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2016, Vasiliy Sokolov."
#property link      "http://www.mql5.com"
#include <Strategy\Strategy.mqh>
#include <Strategy\Indicators\MovingAverage.mqh>

input double StopPercent = 0.05;
//+------------------------------------------------------------------+
//| Defines the actions that need to be performed with a pending     |
//| order.                                                           |
//+------------------------------------------------------------------+
enum ENUM_ORDER_TASK
{
   ORDER_TASK_DELETE,   // Delete a pending order
   ORDER_TASK_MODIFY    // Modify a pending order
};
//+------------------------------------------------------------------+
//| The CImpulse strategy                                            |
//+------------------------------------------------------------------+
class CImpulse : public CStrategy
{
private:
   double            m_percent;        // Percent value for the level of a pending order
   bool              IsTrackEvents(const MarketEvent &event);
protected:
   virtual void      InitBuy(const MarketEvent &event);
   virtual void      InitSell(const MarketEvent &event);
   virtual void      SupportBuy(const MarketEvent &event,CPosition *pos);
   virtual void      SupportSell(const MarketEvent &event,CPosition *pos);
   virtual void      OnSymbolChanged(string new_symbol);
   virtual void      OnTimeframeChanged(ENUM_TIMEFRAMES new_tf);
public:
   double            GetPercent(void);
   void              SetPercent(double percent);
   CIndMovingAverage Moving;
};
//+------------------------------------------------------------------+
//| Working with the pending BuyStop orders for opening a long       |
//| position                                                         |
//+------------------------------------------------------------------+
void CImpulse::InitBuy(const MarketEvent &event)
{
   if(!IsTrackEvents(event))return;
   if(positions.open_buy > 0) return;
   int buy_stop_total = 0;
   ENUM_ORDER_TASK task;
   double target = Ask() + Ask()*(m_percent/100.0);
   if(target < Moving.OutValue(0))                    // The order trigger price must be above the Moving Average
      task = ORDER_TASK_DELETE;
   else
      task = ORDER_TASK_MODIFY;
   for(int i = PendingOrders.Total()-1; i >= 0; i--)
   {
      CPendingOrder* Order = PendingOrders.GetOrder(i);
      if(Order == NULL || !Order.IsMain(ExpertSymbol(), ExpertMagic()))
         continue;
      if(Order.Type() == ORDER_TYPE_BUY_STOP)
      {
         if(task == ORDER_TASK_MODIFY)
         {
            buy_stop_total++;
            Order.Modify(target);
         }
         else
            Order.Delete();
      }
   }
   if(buy_stop_total == 0 && task == ORDER_TASK_MODIFY)
      Trade.BuyStop(MM.GetLotFixed(), target, ExpertSymbol(), 0, 0, NULL);
}
//+------------------------------------------------------------------+
//| Working with the pending SellStop orders for opening a short     |
//| position                                                         |
//+------------------------------------------------------------------+
void CImpulse::InitSell(const MarketEvent &event)
{
   if(!IsTrackEvents(event))return;
   if(positions.open_sell > 0) return;
   int sell_stop_total = 0;
   ENUM_ORDER_TASK task;
   double target = Bid() - Bid()*(m_percent/100.0);
   if(target > Moving.OutValue(0))                    // The order trigger price must be below the Moving Average
      task = ORDER_TASK_DELETE;
   else
      task = ORDER_TASK_MODIFY;
   for(int i = PendingOrders.Total()-1; i >= 0; i--)
   {
      CPendingOrder* Order = PendingOrders.GetOrder(i);
      if(Order == NULL || !Order.IsMain(ExpertSymbol(), ExpertMagic()))
         continue;
      if(Order.Type() == ORDER_TYPE_SELL_STOP)
      {
         if(task == ORDER_TASK_MODIFY)
         {
            sell_stop_total++;
            Order.Modify(target);
         }
         else
            Order.Delete();
      }
   }
   if(sell_stop_total == 0 && task == ORDER_TASK_MODIFY)
      Trade.SellStop(MM.GetLotFixed(), target, ExpertSymbol(), 0, 0, NULL);
}
//+------------------------------------------------------------------+
//| Managing a long position in accordance with the Moving Average   |
//+------------------------------------------------------------------+
void CImpulse::SupportBuy(const MarketEvent &event,CPosition *pos)
{
   if(!IsTrackEvents(event))return;
   if(Bid() < Moving.OutValue(0))
      pos.CloseAtMarket();
}
//+------------------------------------------------------------------+
//| Managing a short position in accordance with the Moving Average  |
//+------------------------------------------------------------------+
void CImpulse::SupportSell(const MarketEvent &event,CPosition *pos)
{
   if(!IsTrackEvents(event))return;
   if(Ask() > Moving.OutValue(0))
      pos.CloseAtMarket();
}
//+------------------------------------------------------------------+
//| Filters incoming events. If the passed event is not              |
//| processed by the strategy, returns false; if it is processed     |
//| returns true.                                                    |
//+------------------------------------------------------------------+
bool CImpulse::IsTrackEvents(const MarketEvent &event)
  {
//We handle only opening of a new bar on the working symbol and timeframe
   if(event.type != MARKET_EVENT_BAR_OPEN)return false;
   if(event.period != Timeframe())return false;
   if(event.symbol != ExpertSymbol())return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Respond to the symbol change                                     |
//+------------------------------------------------------------------+
void CImpulse::OnSymbolChanged(string new_symbol)
  {
   Moving.Symbol(new_symbol);
  }
//+------------------------------------------------------------------+
//| Respond to the timeframe change                                  |
//+------------------------------------------------------------------+
void CImpulse::OnTimeframeChanged(ENUM_TIMEFRAMES new_tf)
  {
   Moving.Timeframe(new_tf);
  }
//+------------------------------------------------------------------+
//| Returns the percent of the breakthrough level                    |
//+------------------------------------------------------------------+
double CImpulse::GetPercent(void)
{
   return m_percent;
}
//+------------------------------------------------------------------+
//| Sets percent of the breakthrough level                           |
//+------------------------------------------------------------------+
void CImpulse::SetPercent(double percent)
{
   m_percent = percent;
}
```

The standard mq5 file of the Expert Advisor that configures and starts this strategy as an Expert Advisor is available below:

```
//+------------------------------------------------------------------+
//|                                                ImpulseExpert.mq5 |
//|                                 Copyright 2016, Vasiliy Sokolov. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2016, Vasiliy Sokolov."
#property link      "http://www.mql5.com"
#property version   "1.00"
#include <Strategy\StrategiesList.mqh>
#include <Strategy\Samples\Impulse.mqh>

CStrategyList Manager;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   CImpulse* impulse = new CImpulse();
   impulse.ExpertMagic(1218);
   impulse.Timeframe(Period());
   impulse.ExpertSymbol(Symbol());
   impulse.ExpertName("Impulse");
   impulse.Moving.MaPeriod(28);
   impulse.SetPercent(StopPercent);
   if(!Manager.AddStrategy(impulse))
      delete impulse;
//---
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   Manager.OnTick();
  }
//+------------------------------------------------------------------+
```

### Analyzing the CImpulse strategy

Please see the attached video, which demonstrates EA operation using pending orders during the strategy testing. BuyStop and SellStop orders are placed when a new bar opens. The orders are placed at a certain distance from the current price, and thus form a dynamic channel. Once the level of a pending order falls below the Moving Average line, it is completely removed. However, when the trigger price becomes higher than the Moving Average, a pending order appears again. The same rule applied to SellStop orders:

impulse - YouTube

[Photo image of Василий Соколов](https://www.youtube.com/channel/UCxbtFmFmwZRe-0q3SP9kgLw?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2404)

Василий Соколов

42 subscribers

[impulse](https://www.youtube.com/watch?v=QWpKtchob0M)

Василий Соколов

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

0:00

0:00 / 1:27

•Live

•

The below screenshot clearly shows the time when the hedge situation was formed, i.e. there was an open Buy position and closure of a short position triggered. After that, the long position continued to exist and was closed according to its logic, when the bar open price moved below the Moving Average:

![](https://c.mql5.com/2/22/7xuxi5fs9nimzwog7_fcxki2mb.png)

Fig. 3. Position management on a hedge supporting account.

The same logic of the Expert Advisor implemented for a classical account would give a similar, though slightly different situation:

![](https://c.mql5.com/2/22/uhkfo_447lep0i.png)

Fig. 4. Position management on a netting account.

A Buy trade was executed during a strong movement, and this trade closed a previously opened short position. Thus, all open positions were closed. Therefore, at the next bar the EA started to place new Buy orders, one of which was filled and turned into a new long position, which, in turn, closed just like a position on the account with the hedging option.

We can change the EA's logic and make it polymorphic, i.e. the corresponding algorithm will be executed depending on the type of account. It is obvious that only one position can exist on a netting account. To avoid the situations when opening of a new position closes the opposite one, we should add stop loss for all net positions. The Stop Loss value should be equal to the breakthrough level of the opposite orders. Thus, triggering of one of the Stop-orders would mean triggering of Stop Loss of an opposite position if such a position exists at the moment. The logic will only be enabled on netting accounts. It should be added to the BuySupport and SellSupport methods. Here is the modified source code of the updated support methods:

```
//+------------------------------------------------------------------+
//| Managing a long position in accordance with the Moving Average   |
//+------------------------------------------------------------------+
void CImpulse::SupportBuy(const MarketEvent &event,CPosition *pos)
{
   if(!IsTrackEvents(event))return;
   ENUM_ACCOUNT_MARGIN_MODE mode = (ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE);
   if(mode != ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
   {
      double target = Bid() - Bid()*(m_percent/100.0);
      if(target < Moving.OutValue(0))
         pos.StopLossValue(target);
      else
         pos.StopLossValue(0.0);
   }
   if(Bid() < Moving.OutValue(0))
      pos.CloseAtMarket();
}
//+------------------------------------------------------------------+
//| Managing a short position in accordance with the Moving Average  |
//+------------------------------------------------------------------+
void CImpulse::SupportSell(const MarketEvent &event,CPosition *pos)
{
   if(!IsTrackEvents(event))return;
   ENUM_ACCOUNT_MARGIN_MODE mode = (ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE);
   if(mode != ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
   {
      double target = Ask() + Ask()*(m_percent/100.0);
      if(target > Moving.OutValue(0))
         pos.StopLossValue(target);
      else
         pos.StopLossValue(0.0);
   }
   if(Ask() > Moving.OutValue(0))
      pos.CloseAtMarket();
}
```

The testing results of the strategy with the updated functionality differ slightly. On netting accounts, the EA will behave like a classical Expert Advisor that only works with one position:

![](https://c.mql5.com/2/22/7r8hc_bxhafzys_kbqb6d8cohlve.png)

Fig. 5. A position managed by a polymorphic Expert Advisor on a classical netting account.

The attachment to this article includes the latest version of the CImpulse strategy implementing different logics for different account types.

Note that all versions of the trading logic are correct. However, the exact operation of a trading strategy depends on the strategy itself. CStrategy only provides a unified interface for working with positions depending on their type.

### Conclusion

We have reviewed the new features of the CStrategy trading engine. The new functions include support for new account types, object operations with pending orders and an extended set of functions for working with current prices.

The new methods of CStrategy allow for a quick and easy access to current prices such as Ask, Bid, and Last. The overridden Digits method now always returns the correct number of decimal places in the symbol price.

Work with pending orders by using the special CPendingOrders and COrdersEnvironment classes simplifies the trading logic. The Expert Advisor can access a pending order through a special object of the CPendingOrder type. By changing the properties of the object, e.g. the order triggering level, the EA changes the appropriate property of the actual order corresponding to the object. The model of objects provides a high level of reliability. It is impossible to access an object, if there is no actual pending order in the system corresponding to that object. Operations with pending orders are performed in the BuyInit and SellInit methods which should be overridden in the strategy. BuyInit is only designed for working with the pending BuyStop and BuyLimit orders. SellInit is only designed for working with the pending SellStop and SellLimit orders.

Within the proposed CStrategy engine, work on accounts with hedging support practically does not differ from that of the classical accounts. Operations with positions do not depend on the position type, and are available through a special CPosition class. The only difference in operations with these types of accounts is the logic of the strategy. If the strategy works with a single position, it implements the appropriate logic for a proper management of that position. If the strategy works with multiple positions at the same time, its logic must take into account that appropriate BuySupport and SellSupport methods and can be transferred to several positions in succession. The trading engine itself does not implement any trade logic. It only provides the type of the position that corresponds to the account type.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2404](https://www.mql5.com/ru/articles/2404)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2404.zip "Download all attachments in the single ZIP archive")

[strategyarticle\_20.04.16.zip](https://www.mql5.com/en/articles/download/2404/strategyarticle_20.04.16.zip "Download strategyarticle_20.04.16.zip")(105.25 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/86905)**
(41)


![Сергей Криушин](https://c.mql5.com/avatar/2018/6/5B2BF337-FB5C.jpeg)

**[Сергей Криушин](https://www.mql5.com/en/users/chipo)**
\|
16 May 2016 at 20:59

**Vasiliy Sokolov:**

You now have a new version of the standard library and an old version of CStrategy codes. Update CStrategy to the new version.

I don't know... maybe in some time...when everything is settled...I'll try again...((((


![Miguel Angel Vico Alba](https://c.mql5.com/avatar/2025/10/68e99f33-714e.jpg)

**[Miguel Angel Vico Alba](https://www.mql5.com/en/users/mike_explosion)**
\|
6 Jul 2016 at 19:20

![](https://c.mql5.com/3/99/homer-excited__8.png)

![Roman Vasilchenko](https://c.mql5.com/avatar/avatar_na2.png)

**[Roman Vasilchenko](https://www.mql5.com/en/users/vrs42)**
\|
18 Oct 2016 at 18:45

Good afternoon, Vasily. I don't understand.... is it possible to get the latest version of your classes somewhere?

Or only the ones attached to the article?

![Alexander Lasygin](https://c.mql5.com/avatar/2013/10/526BB0DA-DE50.jpg)

**[Alexander Lasygin](https://www.mql5.com/en/users/argo)**
\|
16 Sep 2017 at 13:43

Please tell me why my compiler generates this message.![](https://c.mql5.com/3/153/ScreenShot_20170916143321.png)

![alia El-masry](https://c.mql5.com/avatar/2020/3/5E7CFC8F-5972.jpg)

**[alia El-masry](https://www.mql5.com/en/users/onward2020)**
\|
31 Mar 2020 at 09:00

thanks a lot ..

good job

![Graphical Interfaces VI: the Checkbox Control, the Edit Control and their Mixed Types (Chapter 1)](https://c.mql5.com/2/23/avad1j.png)[Graphical Interfaces VI: the Checkbox Control, the Edit Control and their Mixed Types (Chapter 1)](https://www.mql5.com/en/articles/2466)

This article is the beginning of the sixth part of the series dedicated to the development of the library for creating graphical interfaces in the MetaTrader terminals. In the first chapter, we are going to discuss the checkbox control, the edit control and their mixed types.

![How to create an indicator of non-standard charts for MetaTrader Market](https://c.mql5.com/2/23/ava3_.png)[How to create an indicator of non-standard charts for MetaTrader Market](https://www.mql5.com/en/articles/2297)

Through offline charts, programming in MQL4, and reasonable willingness, you can get a variety of chart types: "Point & Figure", "Renko", "Kagi", "Range bars", equivolume charts, etc. In this article, we will show how this can be achieved without using DLL, and therefore such "two-for-one" indicators can be published and purchased from the Market.

![Self-optimization of EA: Evolutionary and genetic algorithms](https://c.mql5.com/2/22/images__2.png)[Self-optimization of EA: Evolutionary and genetic algorithms](https://www.mql5.com/en/articles/2225)

This article covers the main principles set fourth in evolutionary algorithms, their variety and features. We will conduct an experiment with a simple Expert Advisor used as an example to show how our trading system benefits from optimization. We will consider software programs that implement genetic, evolutionary and other types of optimization, and provide examples of application when optimizing a predictor set and parameters of the trading system.

![Graphical Interfaces V: The Combobox Control (Chapter 3)](https://c.mql5.com/2/22/v-avatar__1.png)[Graphical Interfaces V: The Combobox Control (Chapter 3)](https://www.mql5.com/en/articles/2381)

In the first two chapters of the fifth part of the series, we developed classes for creating a scrollbar and a view list. In this chapter, we will speak about creating a class for the combobox control. This is also a compound control containing, among others, elements considered in the previous chapters of the fifth part.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=pmiqnsexbcyinpzuurshttqbkrtozhjt&ssn=1769179815943327508&ssn_dr=0&ssn_sr=0&fv_date=1769179815&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2404&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Universal%20Expert%20Advisor%3A%20Pending%20Orders%20and%20Hedging%20Support%20(Part%205)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917981571718417&fz_uniq=5068706642565987495&sv=2552)

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