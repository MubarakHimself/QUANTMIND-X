---
title: Processing of trade events in Expert Advisor using the OnTrade() function
url: https://www.mql5.com/en/articles/40
categories: Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:30:05.477156
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/40&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071878707842330667)

MetaTrader 5 / Examples


### Introduction

Any trader, that writes Experts on MQL, sooner or later is facing the necessity of reporting how his Expert is working. Or he may need to to implement SMS or e-mail notification about Expert's actions. In any case, we have to "catch" certain events, occurring in the market or actions made by an expert, and notify users.


In this article I want to tell you about, how you can implement processing of trade events, and offer you my implementation.


In this article we will consider processing of the following events:


- Positions


1. Open

2. Add

3. Modify (change Stop Loss and Take Profit)

4. Reverse

5. Close entire position

6. Close part of position


- Pending Orders

1. Place

2. Modify

### 1\. How does it work?

Before we start, in general terms I will describe how trade events work, and all the necessary details will be explained on the fly.


There are [predefined](https://www.mql5.com/en/docs/runtime/event_fire) and [custom](https://www.mql5.com/en/docs/eventfunctions) events in MQL5. We are interested in predefined ones, particularly in the [Trade](https://www.mql5.com/en/docs/runtime/event_fire#trade) event.


The Trade event is generated every time, when trade operation is completed. Each time after the Trade event generation the [OnTrade()](https://www.mql5.com/en/docs/basis/function/events#ontrade) function is called. Processing of orders and positions will be made exactly inside the OnTrade() function.


### 2\. Expert Template

So, let's create a new Expert Advisor. In MetaEditor click **File** -\> **New** to launch MQL5 Wizard. Select **Expert Advisor** and click **Next** . In "General properties of the Expert Advisor" dialog enter the **Name** of Expert Advisor and your own data, if necessary. I named my Expert Advisor as "TradeControl". You can take this name or choose your own, it's not important. We will not specify any parameters, as they will be created on the fly when writing an expert.


Done! Expert Advisor template is created, we have to add the OnTrade() function into it.

As a result, you should get the following code:


```
//+------------------------------------------------------------------+
//|                                              TradeControl_en.mq5 |
//|                                             Copyright KlimMalgin |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "KlimMalgin"
#property link      ""
#property version   "1.00"
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| OnTrade function                                                 |
//+------------------------------------------------------------------+
void OnTrade()
  {
//---

//---
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

  }
//+------------------------------------------------------------------+
```

### 3\. Working with positions

Let's begin with the simplest trade event - opening and closing positions. First, you should understand, what processes occur after pressing the "Sell" and "Buy" buttons.


If we place a call in the OnTrade() function:


```
Alert("The Trade event occurred");
```

Then we will see, that after the opening by market function OnTrade() and along with it our Alert were executed four times:


![Figure 1. Alerts](https://c.mql5.com/2/0/fig_1.png)

Figure 1. Alerts

Why the OnTrade() is called four times, and how we can respond to these alerts? To understand this, let's look at the documentation:


**OnTrade**

The function is called when the [Trade](https://www.mql5.com/en/docs/runtime/event_fire#trade) event occurs. This happens, when the list of [placed orders](https://www.mql5.com/en/docs/trading/orderstotal), [opened positions](https://www.mql5.com/en/docs/trading/positionstotal), [orders history](https://www.mql5.com/en/docs/trading/historyorderstotal) and [deals history](https://www.mql5.com/en/docs/trading/historydealstotal) is changed.

Here I must mention one thing:


When writing this article and communicating with developers, I found that changes in history do not lead to the OnTrade() call! The fact is that the OnTrade() function is called only, when list of placed orders and opened positions is changed! When developing trade events handler you may face the fact that executed orders and deals can appear in history with delay, and you won't be able to process them when the OnTrade() function is running.


Now let's go back to events. As we have seen - when you open by market, the Trade event occurs 4 times:


1. Creating order to open by market.

2. Deal execution.

3. Passing complete order to history.

4. Position Opening.



To track this process in terminal, pay attention to the list of orders in the "Trade" tab of MetaTrader window:

![Figure 2. List of orders in the "Trade" tab](https://c.mql5.com/2/0/fig_2.png)

Figure 2. List of orders in the "Trade" tab

Once you open a position (e.g. down), in the orders list appears an order that has the **[started](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_state)** state (Fig. 2). This changes the list of placed orders, and the Trade event is called. It is the first time when the [OnTrade()](https://www.mql5.com/en/docs/basis/function/events#ontrade) function is activated. Then a deal is executed by created order. At this stage the OnTrade() function is executed second time. As soon as deal is executed, the completed order and its executed deal will be sent to history, and the OnTrade() function is called third time. At the last stage a position is opened by executed deal and the OnTrade() function is called fourth time.

To "catch" moment of position opening, every time you call OnTrade() you have to analyze the list of orders, orders history and deals history. This is what we are going to do now!


OK, the OnTrade() function is called, and we need to know if the number of orders has changed in the "Trade" tab. To do this, we have to compare the number of orders in the list at the time of the previous OnTrade() call and now. To find out how many orders in the list are at the moment, we will use the [OrdersTotal()](https://www.mql5.com/en/docs/trading/orderstotal) function. And to know how many orders were listed on the previous call, we will have to keep the value of OrdersTotal() in each OnTrade() call. For this we will create a special variable:


```
int OrdersPrev = 0;        // Number of orders at the time of previous OnTrade() call
```

At the end of the OnTrade() function the OrdersPrev variable will be assigned with value of OrdersTotal().


You should also consider situation when you run Expert Advisor, and there are pending orders already in the list. Expert must be able to spot them, so in the [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) function the OrdersPrev variable also must be assigned with value of OrdersTotal(). The changes, that we have just made in the Expert, will look like this:


```
int OrdersPrev = 0;        // Number of orders at the time of previous OnTrade() call

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   OrdersPrev = OrdersTotal();
//---
   return(0);
  }
//+------------------------------------------------------------------+
//| OnTrade function                                                 |
//+------------------------------------------------------------------+
void OnTrade()
  {
//---

OrdersPrev = OrdersTotal();
//---
  }
```

Now that we know the number of orders for the current and previous calls - we can find out when the order appeared in the list, and when he, for whatever reasons, had disappeared. To do this, we will use the following condition:


```
if (OrdersPrev < OrdersTotal())
{
  // Order appeared
}
else if(OrdersPrev > OrdersTotal())
{
  // Order disappeared
}
```

So, it turns out that if the previous call we have less orders than now, the order appears in the list (multiple orders can't appear simultaneously), but if the opposite, i.e. now we have less orders than on a previous OnTrade() call, then order is either executed or canceled by some reason. Almost all the work with positions begins with these two conditions.

Only Stop Loss and Take Profit require a separate work with them. To the OnTrade() function I'll add the code, that works with positions. Let's consider it:


```
void OnTrade()
  {
//---
Alert("Trade event occurred");

HistorySelect(start_date,TimeCurrent());

if (OrdersPrev < OrdersTotal())
{
   OrderGetTicket(OrdersTotal()-1);// Select the last order to work with
   _GetLastError=GetLastError();
   Print("Error #",_GetLastError);ResetLastError();
   //--
   if (OrderGetInteger(ORDER_STATE) == ORDER_STATE_STARTED)
   {
      Alert(OrderGetTicket(OrdersTotal()-1),"Order has arrived for processing");
      LastOrderTicket = OrderGetTicket(OrdersTotal()-1);    // Saving the order ticket for further work
   }

}
else if(OrdersPrev > OrdersTotal())
{
   state = HistoryOrderGetInteger(LastOrderTicket,ORDER_STATE);

   // If order is not found, generate an error
   _GetLastError=GetLastError();
   if (_GetLastError != 0){Alert("Error #",_GetLastError," Order is not found!");LastOrderTicket = 0;}
   Print("Error #",_GetLastError," state: ",state);ResetLastError();

   // If order is fully executed
   if (state == ORDER_STATE_FILLED)
   {
      // Then analyze the last deal
      // --
      Alert(LastOrderTicket, "Order executed, going to deal");
      switch(HistoryDealGetInteger(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_ENTRY))
      {

         // Entering the market
         case DEAL_ENTRY_IN:
         Alert(HistoryDealGetInteger(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_ORDER),
         " order invoked deal #",HistoryDealGetTicket(HistoryDealsTotal()-1));

            switch(HistoryDealGetInteger(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_TYPE))
            {
               case 0:
               // If volumes of position and deal are equal, then position has just been opened
                  if (PositionSelect(HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL))
                  && (PositionGetDouble(POSITION_VOLUME) == HistoryDealGetDouble(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_VOLUME)))
                  {
                     Alert("Buy position has been opened on pair ",
                           HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL));
                  }
                  else
               // If volumes of position and deal are not equal, then position has been incremented
                  if (PositionSelect(HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL))
                  && (PositionGetDouble(POSITION_VOLUME) > HistoryDealGetDouble(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_VOLUME)))
                  {
                     Alert("Buy position has incremented on pair ",
                           HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL));
                  }
               break;

               case 1:
               // If volumes of position and deal are equal, then position has just been opened
                  if (PositionSelect(HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL))
                  && (PositionGetDouble(POSITION_VOLUME) == HistoryDealGetDouble(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_VOLUME)))
                  {
                     Alert("Sell position has been opened on pair ",
                           HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL));
                  }
                  else
               // If volumes of position and deal are not equal, then position has been incremented
                  if (PositionSelect(HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL))
                  && (PositionGetDouble(POSITION_VOLUME) > HistoryDealGetDouble(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_VOLUME)))
                  {
                     Alert("Sell position has incremented on pair ",
                           HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL));
                  }

               break;

               default:
                  Alert("Unprocessed code of type: ",
                        HistoryDealGetInteger(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_TYPE));
               break;
            }
         break;

         // Exiting the market
         case DEAL_ENTRY_OUT:
         Alert(HistoryDealGetInteger(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_ORDER),
         " order invoked deal #",HistoryDealGetTicket(HistoryDealsTotal()-1));

            switch(HistoryDealGetInteger(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_TYPE))
            {
               case 0:
               // If position, we tried to close, is still present, then we have closed only part of it
                  if (PositionSelect(HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL)) == true)
                  {
                     Alert("Part of Sell position has been closed on pair ",
                           HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL),
                           " with profit = ",
                           HistoryDealGetDouble(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_PROFIT));
                  }
                  else
               // If position is not found, then it is fully closed
                  if (PositionSelect(HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL)) == false)
                  {
                     Alert("Sell position has been closed on pair ",
                           HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL),
                           " with profit = ",
                           HistoryDealGetDouble(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_PROFIT));
                  }
               break;

               case 1:
               // If position, we tried to close, is still present, then we have closed only part of it
                  if (PositionSelect(HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL)) == true)
                  {
                     Alert("Part of Buy position has been closed on pair ",
                           HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL),
                           " with profit = ",
                           HistoryDealGetDouble(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_PROFIT));
                  }
                  else
               // If position is not found, then it is fully closed
                  if (PositionSelect(HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL)) == false)
                  {
                     Alert("Buy position has been closed on pair ",
                           HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL),
                           " with profit = ",
                           HistoryDealGetDouble(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_PROFIT));
                  }

               break;

               default:
                  Alert("Unprocessed code of type: ",
                        HistoryDealGetInteger(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_TYPE));
               break;
            }
         break;

         // Reverse
         case DEAL_ENTRY_INOUT:
         Alert(HistoryDealGetInteger(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_ORDER),
         " order invoked deal #",HistoryDealGetTicket(HistoryDealsTotal()-1));

            switch(HistoryDealGetInteger(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_TYPE))
            {
               case 0:
                  Alert("Sell is reversed to Buy on pair ",
                        HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL),
                        " resulting profit = ",
                        HistoryDealGetDouble(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_PROFIT));
               break;

               case 1:
                  Alert("Buy is reversed to Sell on pair ",
                        HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL),
                        " resulting profit = ",
                        HistoryDealGetDouble(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_PROFIT));
               break;

               default:
                  Alert("Unprocessed code of type: ",
                        HistoryDealGetInteger(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_TYPE));
               break;
            }
         break;

         // Indicates the state record
         case DEAL_ENTRY_STATE:
            Alert("Indicates the state record. Unprocessed code of type: ",
            HistoryDealGetInteger(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_TYPE));
         break;
      }
      // --
   }
}

OrdersPrev = OrdersTotal();

//---
  }
```

Also be sure that at the beginning of program you have declared the following variables:


```
datetime start_date = 0;   // Date, from which we begin to read history

int OrdersPrev = 0;        // Number of orders at the time of previous OnTrade() call
int PositionsPrev = 0;     // Number of positions at the time of previous OnTrade() call
ulong LastOrderTicket = 0; // Ticket of the last processed order

int _GetLastError=0;       // Error code
long state=0;              // Order state
```

Let's go back to the contents of OnTrade().

You can comment out the Alert at the beginning, but I'll leave it Next goes the [HistorySelect()](https://www.mql5.com/en/docs/trading/historyselect) function. It generates a list of deals and orders history for the specified period of time, which is defined by two parameters of the function. If this function is not called before going to deals and orders history, we won't get any information because history lists will be empty. After calling HistorySelect() conditions are evaluated, as it was written just before.

When new order comes, first we select it and check for errors:


```
OrderGetTicket(OrdersTotal()-1);// Select the last order for work
_GetLastError=GetLastError();
Print("Error #",_GetLastError);ResetLastError();
```

After selecting the order, we get the error code using the [GetLastError()](https://www.mql5.com/en/docs/check/getlasterror) function. Then using the [Print()](https://www.mql5.com/en/docs/common/print) function we print code into journal and using the [ResetLastError()](https://www.mql5.com/en/docs/common/resetlasterror) function we reset the error code to zero, so on the next [GetLastError()](https://www.mql5.com/en/docs/check/getlasterror) call for other situations we won't get the same error code.


After checking for errors, if the order has been successfully selected, check its state:


```
if (OrderGetInteger(ORDER_STATE) == ORDER_STATE_STARTED)
{
   Alert(OrderGetTicket(OrdersTotal()-1),"Order has arrived for processing");
   LastOrderTicket = OrderGetTicket(OrdersTotal()-1);    // Saving the order ticket for further work
}
```

If order has the [started](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_state) state, i.e. it is checked for correctness, but not yet accepted, then it is expected to be executed in the near future, and we simply give an [Alert()](https://www.mql5.com/en/docs/common/alert) notifying that order is being processed and save its ticket on the next OnTrade() call. Instead of Alert() you can use any other kind of notifications.


In the code above the


```
OrderGetTicket(OrdersTotal()-1)
```

will return the last order ticket from the entire list of orders.

OrdersTotal()-1 indicates that we need to get the latest order. Since the [OrdersTotal()](https://www.mql5.com/en/docs/trading/orderstotal) function returns the total number of orders (e.g. if there is 1 order in the list, then OrdersTotal() will return 1), and the order index number is counted from 0, then to obtain the index number of the last order we must subtract 1 from the total number of orders (if OrdersTotal() returns 1, then index number of this order will be equal to 0). And the [OrderGetTicket()](https://www.mql5.com/en/docs/trading/ordergetticket) function in its turn will return the order ticket, which number will be passed to it.


It was the first condition, it is usually triggered when on the first OnTrade() call . Next comes the second condition, that is met on the second OnTrade() call, when the order is executed, went down to history and position should open.


If the order is missing in the list, then it went down to history, it must be definitely there! Therefore, we appeal to the history of orders using the [HistoryOrderGetInteger()](https://www.mql5.com/en/docs/trading/historyordergetinteger) function to get the order state. And to read history data for particular order, we need its ticket. For this if the first condition the ticket of incoming order has been stored in the LastOrderTicket variable.

Thus we get the order state, indicating order ticket as the first parameter for HistoryOrderGetInteger(), and type of [needed property](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_property_integer) \- as the second. After trying to get the order state, we get the error code and write it to the journal. It is necessary in case your order, which we need to work with, has not yet managed to get into history, and we appealing to him (experience shows that this is possible and quite a lot. I wrote about this in the beginning of this article).


If an error occurs, processing stops, because there is no data to work with and none of the following conditions is met. And if the HistoryOrderGetInteger() call was successful and the order has state "Order is fully executed":


```
// If order is fully executed
if (state == ORDER_STATE_FILLED)
```

Then give another notification:


```
// Then analyze the last deal
// --
  Alert(LastOrderTicket, "Order executed, going to deal");
```

And let's go to processing the deal, that was invoked by this order. First, find out the direction of deal ( [DEAL\_ENTRY](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_entry) property). Direction is not the Buy or Sell, but the Entering the market, Exiting the market, Reverse or Indication of state record. Thus, using the DEAL\_ENTRY property we can find out whether the order has been set to open position, to close position or to reverse.


To analyze the deal and its results, let's also appeal to the history using the following construction:


```
switch(HistoryDealGetInteger(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_ENTRY))
{
  ...
}
```

It works the same as with orders:


[HistoryDealsTotal()](https://www.mql5.com/en/docs/trading/historydealstotal) returns the total number of deals. To get the number of latest deal we subtract 1 from value of HistoryDealsTotal(). The resulting number of deal is passed to the [HistoryDealGetTicket()](https://www.mql5.com/en/docs/trading/historydealgetticket) function, which in turn passes the ticket of selected deal to the [HistoryDealGetInteger()](https://www.mql5.com/en/docs/trading/historydealgetinteger) function. And the HistoryDealGetInteger() by specified ticket and property type will return the direction of deal.

Let's examine in details the direction of Entering the market. The other directions will be covered briefly, as they are processed the almost same way:


The value of expression, obtained from HistoryDealGetInteger(), is compared with values of the case blocks, until a match is found. Suppose we're entering the market, i.e. opening the Sell order. Then the first block will be executed:


```
// Entering the market
case DEAL_ENTRY_IN:
```

In the beginning of block you are notified about the creation of deal. At the same this notification ensures that everything was OK and the deal is being processed.


After notification comes another switch block, that analyzes the type of deal:


```
   switch(HistoryDealGetInteger(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_TYPE))
   {
      case 0:
      // If volumes of position and deal are equal, then position has just been opened
         if (PositionSelect(HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL))
         && (PositionGetDouble(POSITION_VOLUME) == HistoryDealGetDouble(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_VOLUME)))
         {
            Alert("Buy position has been opened on pair ",
                  HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL));
         }
         else
      // If volumes of position and deal are not equal, then position has been incremented
         if (PositionSelect(HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL))
         && (PositionGetDouble(POSITION_VOLUME) > HistoryDealGetDouble(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_VOLUME)))
         {
            Alert("Buy position has incremented on pair ",
                  HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL));
         }
      break;

      case 1:
      // If volumes of position and deal are equal, then position has just been opened
         if (PositionSelect(HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL))
         && (PositionGetDouble(POSITION_VOLUME) == HistoryDealGetDouble(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_VOLUME)))
         {
            Alert("Sell position has been opened on pair ",
                  HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL));
         }
         else
      // If volumes of position and deal are not equal, then position has been incremented
         if (PositionSelect(HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL))
         && (PositionGetDouble(POSITION_VOLUME) > HistoryDealGetDouble(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_VOLUME)))
         {
            Alert("Sell position has incremented on pair ",
                  HistoryDealGetString(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_SYMBOL));
         }

      break;

      default:
         Alert("Unprocessed code of type: ",
               HistoryDealGetInteger(HistoryDealGetTicket(HistoryDealsTotal()-1),DEAL_TYPE));
      break;
   }
```

Get information about the deal from history - the same way as earlier, except the specified property. This time you must specify the [DEAL\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_type) to know, whether Buy or Sell deal is made. I analyze only the Buy and Sell types, but besides them there are four more. But these remaining four types of deals are less common, so instead of four case blocks only one default block is used for them. It will give an Alert() with the type code.


As you've probably noticed in the code, not only opening of Buy and Sell positions are processed, but also their increment. To determine when the position has been incremented, and when it has been opened - you need to compare the volume of executed deal and position, that became the result of this deal. If the volume of position is equal to the volume of executed deal - this position has been opened, and if the volumes of position and deal are different - this position has been incremented. This applies both to the Buy positions (in the case '0' block) and to the Sell positions (in the case '1' block). The last block is the default, which handles all situations other than Buy and Sell. The entire processing consists of notification about the type code, returned by the [HistoryDealGetInteger()](https://www.mql5.com/en/docs/trading/historydealgetinteger) function.


And finally, the last concern about the work with positions. This is the processing of changes in Stop Loss and Take Profit values. To know which one of position parameters has changed, we need to compare the current and previous state of its parameters. The current values of position parameters can always be obtained using service functions, but the previous values should be saved.


For this we will write a special function, that will save position parameters to the array of structures:


```
void GetPosition(_position &Array[])
  {
   int _GetLastError=0,_PositionsTotal=PositionsTotal();

   int temp_value=(int)MathMax(_PositionsTotal,1);
   ArrayResize(Array, temp_value);

   _ExpertPositionsTotal=0;
   for(int z=_PositionsTotal-1; z>=0; z--)
     {
      if(!PositionSelect(PositionGetSymbol(z)))
        {
         _GetLastError=GetLastError();
         Print("OrderSelect() - Error #",_GetLastError);
         continue;
        }
      else
        {
            // If the position is found, then put its info to the array
            Array[z].type         = PositionGetInteger(POSITION_TYPE);
            Array[z].time         = PositionGetInteger(POSITION_TIME);
            Array[z].magic        = PositionGetInteger(POSITION_MAGIC);
            Array[z].volume       = PositionGetDouble(POSITION_VOLUME);
            Array[z].priceopen    = PositionGetDouble(POSITION_PRICE_OPEN);
            Array[z].sl           = PositionGetDouble(POSITION_SL);
            Array[z].tp           = PositionGetDouble(POSITION_TP);
            Array[z].pricecurrent = PositionGetDouble(POSITION_PRICE_CURRENT);
            Array[z].comission    = PositionGetDouble(POSITION_COMMISSION);
            Array[z].swap         = PositionGetDouble(POSITION_SWAP);
            Array[z].profit       = PositionGetDouble(POSITION_PROFIT);
            Array[z].symbol       = PositionGetString(POSITION_SYMBOL);
            Array[z].comment      = PositionGetString(POSITION_COMMENT);
        _ExpertPositionsTotal++;
        }
     }

   temp_value=(int)MathMax(_ExpertPositionsTotal,1);
   ArrayResize(Array,temp_value);
  }
```

To use this function we must add the following code into the block of global variables declaration:


```
/*
 *
 * Structure that stores information about positions
 *
 */
struct _position
{

long     type,          // Position type
         magic;         // Magic number for position
datetime time;          // Time of position opening

double   volume,        // Position volume
         priceopen,     // Position price
         sl,            // Stop Loss level for opened position
         tp,            // Take Profit level for opened position
         pricecurrent,  // Symbol current price
         comission,     // Commission
         swap,          // Accumulated swap
         profit;        // Current profit

string   symbol,        // Symbol, by which the position has been opened
         comment;       // Comment to position
};

int _ExpertPositionsTotal = 0;

_position PositionList[],     // Array that stores info about position
          PrevPositionList[];
```

The GetPosition() function prototype was found long time ago in the www.mql4.com articles, but I couldn't find it now and I can't specify the source. I'm not going to discuss the work of this function in details. The point is, that as a parameter by reference passed an array of the \_position type (structure with fields corresponding to the position fields), to which all information about the currently open positions and values of their parameters is passed.


To conveniently track changes in the position parameters let's create two array of the \_position type. These are PositionList\[\] (the current state of positions) and PrevPositionList\[\] (the previous state of positions).


To begin work with positions, we must add the next call into OnInit() and to the end of OnTrade():


```
GetPosition(PrevPositionList);
```

Also in the beginning of Ontrade() we must add the call:


```
GetPosition(PositionList);
```

Now in the PositionList\[\] and PrevPositionList\[\] arrays at our disposal will be information about positions on the current and previous OnTrade() call respectively.


Now let's consider the actual code of tracking changes in sl and tp:


```
if ((PositionsPrev == PositionsTotal()) && (OrdersPrev == OrdersTotal()))
{
   string _alerts = "";
   bool modify = false;

   for (int i=0;i<_ExpertPositionsTotal;i++)
   {
      if (PrevPositionList[i].sl != PositionList[i].sl)
      {
         _alerts += "On pair "+PositionList[i].symbol+" Stop Loss changed from "+ PrevPositionList[i].sl +" to "+ PositionList[i].sl +"\n";
         modify = true;
      }
      if (PrevPositionList[i].tp != PositionList[i].tp)
      {
         _alerts += "On pair "+PositionList[i].symbol+" Take Profit changed from "+ PrevPositionList[i].tp +" to "+ PositionList[i].tp +"\n";
         modify = true;
      }

   }
   if (modify == true)
   {
      Alert(_alerts);
      modify = false;
   }
}
```

As we see, the code is not too big, but this is only because of the considerable preparatory work. Let's delve into it.


It all starts with condition:


```
if ((PositionsPrev == PositionsTotal()) && (OrdersPrev == OrdersTotal()))
```

Here we see, that neither orders nor positions were placed or deleted. If condition is met, then most likely the parameters of some positions or orders have changed.


In the beginning of function two variables are declared:


- \_alerts - stores all notifications about changes.

- modify - allows you to display messages about the changes only if they really were.


Next in the loop we check the matching of values of Take Profits and Stop Losses on previous and current call [OnTrade()](https://www.mql5.com/en/docs/basis/function/events#ontrade) for each position. Information on all mismatches will be stored in the \_alerts variable and later it will be displayed by the [Alert()](https://www.mql5.com/en/docs/common/alert) function. By the way, processing of pending orders modification will be carried out the same way.


For now let's finish with positions and proceed to the placement of pending orders.


### 4\. Working with orders

Let's start with placement of pending orders event.


When new pending order appears, the Trade event is generated only once, but it's enough to process it! Put the code, that works with pending orders, into the body of operator:


```
if (OrdersPrev < OrdersTotal())
```

And get the following:


```
if (OrdersPrev < OrdersTotal())
{
   OrderGetTicket(OrdersTotal()-1);// Select the last order to work with
   _GetLastError=GetLastError();
   Print("Error #",_GetLastError);ResetLastError();
   //--
   if (OrderGetInteger(ORDER_STATE) == ORDER_STATE_STARTED)
   {
      Alert(OrderGetTicket(OrdersTotal()-1),"Order has arrived for processing");
      LastOrderTicket = OrderGetTicket(OrdersTotal()-1);    // Saving the order ticket for further work
   }

   state = OrderGetInteger(ORDER_STATE);
   if (state == ORDER_STATE_PLACED)
   {
      switch(OrderGetInteger(ORDER_TYPE))
      {
         case 2:
            Alert("Pending order Buy Limit #", OrderGetTicket(OrdersTotal()-1)," accepted!");
         break;

         case 3:
            Alert("Pending order Sell Limit #", OrderGetTicket(OrdersTotal()-1)," accepted!");
         break;

         case 4:
            Alert("Pending order Buy Stop #", OrderGetTicket(OrdersTotal()-1)," accepted!");
         break;

         case 5:
            Alert("Pending order Sell Stop #", OrderGetTicket(OrdersTotal()-1)," accepted!");
         break;

         case 6:
            Alert("Pending order Buy Stop Limit #", OrderGetTicket(OrdersTotal()-1)," accepted!");
         break;

         case 7:
            Alert("Pending order Sell Stop Limit  #", OrderGetTicket(OrdersTotal()-1)," accepted!");
         break;
      }
   }
}
```

Here the code, that works with pending orders, begins with the following:


```
   state = OrderGetInteger(ORDER_STATE);
   if (state == ORDER_STATE_PLACED)
   {
```

First the [order state](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_state) is checked. Order must have the ORDER\_STATE\_PLACED state, i.e. is should be accepted. And if this condition is met, then comes the [switch](https://www.mql5.com/en/docs/basis/operators/switch) operator, that prints a message depending on the order type.


Next we will work with events, that occur when orders are modified. Modification of orders is similar to modification of positions. Likewise, the structure that stores orders properties is created:


```
/*
 *
 * Structure that stores information about orders
 *
 */
struct _orders
{

datetime time_setup,       // Time of order placement
         time_expiration,  // Time of order expiration
         time_done;        // Time of order execution or cancellation

long     type,             // Order type
         state,            // Order state
         type_filling,     // Type of execution by remainder
         type_time,        // Order lifetime
         ticket;           // Order ticket

long     magic,            // Id of Expert Advisor, that placed an order
                           // (intended to ensure that each Expert
                           // must place it's own unique number)

         position_id;      // Position id, that is placed on order,
                           // when it is executed. Each executed order invokes a
                           // deal, that opens new or changes existing
                           // position. Id of that position is placed on
                           // executed order in this moment.

double volume_initial,     // Initial volume on order placement
       volume_current,     // Unfilled volume
       price_open,         // Price, specified in the order
       sl,                 // Stop Loss level
       tp,                 // Take Profit level
       price_current,      // Current price by order symbol
       price_stoplimit;    // Price of placing Limit order when StopLimit order is triggered

string symbol,             // Symbol, by which the order has been placed
       comment;            // Comment

};

int _ExpertOrdersTotal = 0;

_orders OrderList[],       // Arrays that store info about orders
        PrevOrderList[];
```

Each field of the structure corresponds to one of the order properties. After declaring the structure, the variable of int type and two array of the \_orders type are declared. The \_ExpertOrdersTotal variable will store the total number of orders, and the OrderList\[\] and PrevOrderList\[\] arrays will store information about orders in the current and previous OnTrade() call respectively.


The function itself will look as following:


```
void GetOrders(_orders &OrdersList[])
  {

   int _GetLastError=0,_OrdersTotal=OrdersTotal();

   int temp_value=(int)MathMax(_OrdersTotal,1);
   ArrayResize(OrdersList,temp_value);

   _ExpertOrdersTotal=0;
   for(int z=_OrdersTotal-1; z>=0; z--)
     {
      if(!OrderGetTicket(z))
        {
         _GetLastError=GetLastError();
         Print("GetOrders() - Error #",_GetLastError);
         continue;
        }
      else
        {
        OrdersList[z].ticket          = OrderGetTicket(z);
        OrdersList[z].time_setup      = OrderGetInteger(ORDER_TIME_SETUP);
        OrdersList[z].time_expiration = OrderGetInteger(ORDER_TIME_EXPIRATION);
        OrdersList[z].time_done       = OrderGetInteger(ORDER_TIME_DONE);
        OrdersList[z].type            = OrderGetInteger(ORDER_TYPE);

        OrdersList[z].state           = OrderGetInteger(ORDER_STATE);
        OrdersList[z].type_filling    = OrderGetInteger(ORDER_TYPE_FILLING);
        OrdersList[z].type_time       = OrderGetInteger(ORDER_TYPE_TIME);
        OrdersList[z].magic           = OrderGetInteger(ORDER_MAGIC);
        OrdersList[z].position_id     = OrderGetInteger(ORDER_POSITION_ID);

        OrdersList[z].volume_initial  = OrderGetDouble(ORDER_VOLUME_INITIAL);
        OrdersList[z].volume_current  = OrderGetDouble(ORDER_VOLUME_CURRENT);
        OrdersList[z].price_open      = OrderGetDouble(ORDER_PRICE_OPEN);
        OrdersList[z].sl              = OrderGetDouble(ORDER_SL);
        OrdersList[z].tp              = OrderGetDouble(ORDER_TP);
        OrdersList[z].price_current   = OrderGetDouble(ORDER_PRICE_CURRENT);
        OrdersList[z].price_stoplimit = OrderGetDouble(ORDER_PRICE_STOPLIMIT);

        OrdersList[z].symbol          = OrderGetString(ORDER_SYMBOL);
        OrdersList[z].comment         = OrderGetString(ORDER_COMMENT);

        _ExpertOrdersTotal++;
        }
     }

   temp_value=(int)MathMax(_ExpertOrdersTotal,1);
   ArrayResize(OrdersList,temp_value);

  }
```

Similar to the GetPosition() function, it reads the information about properties of each placed order and puts it into array, passed to it as input parameter. The function code must be placed at the end of your expert, and its calls - as follows:


```
GetOrders(PrevOrderList);
```

Placed in the OnInit() and at the end of OnTrade().


```
GetOrders(OrderList);
```

Placed at the beginning of OnTrade().


Now consider the code, that will process the modification of orders. It is a loop and it complements the code of positions modification:


```
   for (int i = 0;i<_ExpertOrdersTotal;i++)
   {
      if (PrevOrderList[i].sl != OrderList[i].sl)
      {
         _alerts += "Order "+OrderList[i].ticket+" has changed Stop Loss from "+ PrevOrderList[i].sl +" to "+ OrderList[i].sl +"\n";
         modify = true;
      }
      if (PrevOrderList[i].tp != OrderList[i].tp)
      {
         _alerts += "Order "+OrderList[i].ticket+" has changed Take Profit from "+ PrevOrderList[i].tp +" to "+ OrderList[i].tp +"\n";
         modify = true;
      }
   }
```

The loop process all orders and compares the values of Stop Losses and Take Profits on the current and previous OnTrade() calls. If there are differences, they are saved in the \_alerts variable, and when the loop is complete they will be displayed by the Alert() function.


This code is placed into the body of operator:


```
if ((PositionsPrev == PositionsTotal()) && (OrdersPrev == OrdersTotal()))
{
```

Immediately after the loop, that works with positions.


For now the work with trade events does not end. This article covers only the main principles of work with the [Trade](https://www.mql5.com/en/docs/runtime/event_fire#trade) event. In general, the possibilities offered by this method are quite large and are beyond the scope of this article.

### Conclusion

Ability to work with trade events (as part of the MQL5 language) is potentially a powerful tool, that allows not only to relatively quickly implement algorithms of orders verification and to generate trade reports, but also to reduce the cost of system resources and volume of source code, which will undoubted benefit for developers.


Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/40](https://www.mql5.com/ru/articles/40)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/40.zip "Download all attachments in the single ZIP archive")

[tradecontrol\_en.mq5](https://www.mql5.com/en/articles/download/40/tradecontrol_en.mq5 "Download tradecontrol_en.mq5")(20.25 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [OOP in MQL5 by Example: Processing Warning and Error Codes](https://www.mql5.com/en/articles/70)
- [How to call indicators in MQL5](https://www.mql5.com/en/articles/43)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/1499)**
(25)


![Tango_X](https://c.mql5.com/avatar/avatar_na2.png)

**[Tango\_X](https://www.mql5.com/en/users/tango_x)**
\|
21 Oct 2018 at 07:29

Strange...., why isn't it shown how to track [closing a position](https://www.metatrader5.com/en/terminal/help/trading/performing_deals#position_manage "Help: Opening and closing positions in MetaTrader 5 trading terminal") by stop or take?


![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)

**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**
\|
21 Oct 2018 at 07:32

**Tango\_X:**

Strange...., why is it not shown how to track [position closing](https://www.metatrader5.com/en/terminal/help/trading/performing_deals#position_manage "Help: Opening and closing positions in MetaTrader 5 trading terminal") by stop or take?

The article was written a long time ago. Since then a new feature has appeared

Starting from [build 1625](https://www.mql5.com/ru/forum/206431 "MetaTrader 5 build 1625 beta version: Custom Financial Instruments") there is a  wonderful enumeration ENUM\_DEAL\_REASON:

| ENUM\_DEAL\_REASON | Reason description |
| --- | --- |
| ... | ... |
| DEAL\_REASON\_SL | The operation was performed as a result of Stop Loss triggering. |
| DEAL\_REASON\_TP | The operation was executed as a result of Take Profit triggering |
| ... | ... |

which can be tracked in OnTradeTransaction.

Example of operation: [Stop Loss Take Profit](https://www.mql5.com/ru/code/18755)

![Tango_X](https://c.mql5.com/avatar/avatar_na2.png)

**[Tango\_X](https://www.mql5.com/en/users/tango_x)**
\|
21 Oct 2018 at 07:41

**Vladimir Karputov:**

The article was written a long time ago. Since then, a new opportunity has arisen

Starting with [build 1625](https://www.mql5.com/ru/forum/206431 "MetaTrader 5 build 1625 beta version: Custom Financial Instruments") there is a  wonderful enumeration ENUM\_DEAL\_REASON:

| ENUM\_DEAL\_REASON | Reason description |
| --- | --- |
| ... | ... |
| DEAL\_REASON\_SL | The operation was performed as a result of Stop Loss triggering. |
| DEAL\_REASON\_TP | The operation was executed as a result of Take Profit triggering |
| ... | ... |

which can be tracked in OnTradeTransaction.

Example of operation: [Stop Loss Take Profit](https://www.mql5.com/ru/code/18755)

super! thanks!!!

![Tango_X](https://c.mql5.com/avatar/avatar_na2.png)

**[Tango\_X](https://www.mql5.com/en/users/tango_x)**
\|
21 Oct 2018 at 12:17

**Vladimir Karputov:**

The article was written a long time ago. Since then, a new opportunity has arisen

Starting with [build 1625](https://www.mql5.com/ru/forum/206431 "MetaTrader 5 build 1625 beta version: Custom Financial Instruments") there is a  wonderful enumeration ENUM\_DEAL\_REASON:

| ENUM\_DEAL\_REASON | Reason description |
| --- | --- |
| ... | ... |
| DEAL\_REASON\_SL | The operation was performed as a result of Stop Loss triggering. |
| DEAL\_REASON\_TP | The operation was executed as a result of Take Profit triggering |
| ... | ... |

which can be tracked in OnTradeTransaction.

Example of operation: [Stop Loss Take Profit](https://www.mql5.com/ru/code/18755)

One more question along the way.

I use the "Comments" field in a position to store the period of opening this position, and when Stop Loss/Take Profit is triggered, the terminal writes st/tp in this field. How to forbid the terminal and broker to change the comment? Or maybe you know another way to store the period for each position?

![Serhii Tymchenko](https://c.mql5.com/avatar/2022/1/61F7CE63-9F51.jpg)

**[Serhii Tymchenko](https://www.mql5.com/en/users/sergo_mql)**
\|
2 Sep 2022 at 15:24

in mql5 I can't complete this thing. How to recognise only new orders irrespective of comments and majic #, I have at closing [buy signalling](https://www.mql5.com/en/articles/522 "Article: MetaTrader 5 Added Trading Signals - Better Than PAMM Accounts! ") that sell came in (and vice versa).


![Testing Performance of Moving Averages Calculation in MQL5](https://c.mql5.com/2/0/moving_averages_performance_MQL5__1.png)[Testing Performance of Moving Averages Calculation in MQL5](https://www.mql5.com/en/articles/106)

A number of indicators have appeared since the time of first Moving Average indicator creation. Many of them use the similar smoothing methods, but the performances of different moving averages algorithms have not been studied. In this article, we will consider possible ways of use the Moving Averages in MQL5 and compare their performance.

![The Use of ORDER_MAGIC for Trading with Different Expert Advisors on a Single Instrument](https://c.mql5.com/2/0/order_magic_MQL5__1.png)[The Use of ORDER\_MAGIC for Trading with Different Expert Advisors on a Single Instrument](https://www.mql5.com/en/articles/112)

This article considers the questions of information coding, using the magic-identification, as well as the division, assembly, and synchronization of automatic trading of different Expert Advisors. This article will be interesting to beginners, as well as to more experienced traders, because it tackles the question of virtual positions, which can be useful in the implementation of complex systems of synchronization of Expert Advisors and various strategies.

![How to Write an Indicator on the Basis of Another Indicator](https://c.mql5.com/2/0/indicator_based_on_other_MQL5__1.png)[How to Write an Indicator on the Basis of Another Indicator](https://www.mql5.com/en/articles/127)

In MQL5 you can write an indicator both from a scratch and on the basis of another already existing indicator, in-built in the client terminal or a custom one. And here you also have two ways - to improve an indicator by adding new calculations and graphical styles to it , or to use an indicator in-built in the client terminal or a custom one via the iCustom() or IndicatorCreate() functions.

![A Library for Constructing a Chart via Google Chart API](https://c.mql5.com/2/0/Google_Chart_API_MQL5__1.png)[A Library for Constructing a Chart via Google Chart API](https://www.mql5.com/en/articles/114)

The construction of various types of diagrams is an essential part of the analyses of the market situation and the testing of a trading system. Frequently, in order to construct a nice looking diagram, it is necessary to organize the data output into a file, after which it is used in applications such as MS Excel. This is not very convenient and deprives us of the ability to dynamically update the data. Google Charts API provided the means for creating charts in online modes, by sending a special request to the server. In this article we attempt to automate the process of creating such a request and obtaining a chart from the Google server.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/40&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071878707842330667)

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