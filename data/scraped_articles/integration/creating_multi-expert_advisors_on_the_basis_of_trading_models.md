---
title: Creating Multi-Expert Advisors on the basis of Trading Models
url: https://www.mql5.com/en/articles/217
categories: Integration
relevance_score: 6
scraped_at: 2026-01-23T17:27:10.046350
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=xaepphdtumxysnfspmaxhmglfohjmjnc&ssn=1769178428953798164&ssn_dr=0&ssn_sr=0&fv_date=1769178428&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F217&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20Multi-Expert%20Advisors%20on%20the%20basis%20of%20Trading%20Models%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917842847996286&fz_uniq=5068249439002359612&sv=2552)

MetaTrader 5 / Tester


### Introduction

The technical capabilities of the MetaTrader 5 terminal, and its strategy tester, determine the work and testing of multi-currency trading systems. The complexity of developing such systems for MetaTrader 4 conditioned, first of all, by the inability of simultaneous tick by tick testing of several trading tools. In addition, the limited language resources of the MQL4 language did not allow for the organization of complex data structures and for the efficient management of the data.

With the release of MQL5 the situation has changed. Henceforth, MQL5supports the [object-oriented approach](https://www.mql5.com/en/docs/basis/oop), is based on a developed mechanism of auxiliary functions, and even has a set of [Standard Library](https://www.mql5.com/en/docs/standardlibrary) base classes to facilitate the daily tasks of users - ranging from the organization of data to the work interfaces for standard system functions.

And although the technical specifications of the strategy tester and the terminal allow for the use of multi-currency EAs, they do not have built-in methods for the parallelization of the work of a single EA simultaneously on several instruments or time-frames. As before, for the work of an EA in the simplest case, you need to run it in the window of the symbol, which determines the name of the trading instrument and its time-frame. As a result, the work methodology, accepted from the time of MetaTrader 4, does not allow to take full advantage of the strategy tester and the MetaTrader 5 terminal.

The situation is complicated by the fact that only one cumulative position for each instument, equal to the total amount of deals on that instrument, is allowed, Certainly, the transition to a net position is correct and timely. Net position comes closest to the perfect representation of the trader's interest on a particular market.

However, such an organization of the deals does not make the trading process simple and easily visualized. Previously, it was sufficient enough for an EA to select its open order (for example, the order could be identified using the magic number), and implement the required action. Now, even the absence of a net position on an instrument does not mean that a particular instance of an EA on it at the moment is not on the market!

Third-party developers offer various ways to solving the problem with the net position - ranging from writing a special managers of virtual orders (see the article [A Virtual Order Manager to track orders within the position-centric MT5 environment](https://www.mql5.com/en/articles/88))  to integrating the inputs in an aggregated position, using the magic number (see [The Optimal Method for Calculation of Total Position Volume by Specified Magic Number](https://www.mql5.com/en/articles/125) or [The Use of ORDER\_MAGIC for Trading with Different Expert Advisors on a Single Instrument](https://www.mql5.com/en/articles/112)).

However, in addition to problems with the aggregated position, there is a problem of the so-called multi-currency, when the same EA is required to trade on multiple instruments. The solution of this problem can be found in the article [Creating an Expert Advisor which Trades on Different Instruments](https://www.mql5.com/en/articles/105).

All of the proposed methods work and have their own advantages. However, their fatal flaw is that each of these methods is trying to approach the issue from its own perspective, offering solutions that are, for example, well suited for simultaneous trading by several EAs on a single instrument, but are not suitable for multi-currency solutions.

This article aims to solve all of the problems with a single solution. Using this solution can solve the problem of multi-currency and even multi-system testing of the interaction between different EAs on a single instrument. This seems difficult, or even impossible to achieve, but in reality it's all much easier.

**_Just imagine_** **_your a single EA trades simultaneously on several dozens of trading strategies, on all of the available instruments, and on all of the possible time frames!_** In addition, the EA is easily tested in the tester, and for all of the strategies, included in its composition, has one or several working systems of money management.

So, here are the main tasks that we will need to solve:

1. The EA needs to trade on the basis of several trading systems at the same time. In addition, it must equally easily trade on a single, as well as on multiple trading systems;
2. All of the trading system, implemented in the EA, must not conflict with each other. Each trading system must handle only it own contribution to the total net position, and only its own orders;
3. Each system should be equally easy to trade with on a single time-frame of the instrument, as well as on all time-frames at once.
4. Each systems should be equally easy to trade with on a single trading instrument, as well as on all of the available instruments at once.

If we examine carefully the list of the tasks that we need to handle, we will arrive at a three-dimensional array. The first dimension of the array - the number of trading systems, the second dimension - the number of time-frames, on which the specific TS needs to operate, and the third - the number of trading instruments for the TS. A simple calculation shows that even such a simple EA as the MACD Sample, when working simultaneously on 8 major currency pairs, will have 152 independent solutions: 1 EA \* 8 pairs \* 19 time-frames (weekly and monthly time-frames are not included).

If the trading system will be much larger, and the trading portfolio of EAs considerably more extensive, then the number of solutions could easily be over 500, and in some cases, over 1000! It is clear that it is impossible to manually configure and then upload each combination separately. Therefore it is necessary to build a system in such a way, that it would automatically adjusts each combination, load it into the memory of the EA, and the EA would then trade, based on the rules of a specific instance of this combination.

### Terms and concepts

Here and further the notion "trading strategy" will be replaced by a more specific term _trading model_ or simply _model_. _A trading model_ is a special class, built according to specific rules, which fully describes the trading strategy: indicators, used in trade, the trade conditions of entry and exit, the methods of money management, etc. Each trading model is abstract and does not define specific parameters for its operation.

A simple example is the trading tactic, based on the crossover of two moving averages. If the fast moving average crosses the slow one upward, it opens a deal to buy, if on the contrary, downwards, it opens a deal to sell. This formulation is sufficient for writing a trading model that trades on its grounds.

However, once such a model will be described, it is necessary to determine the methods of the moving averages, with theiraveraging period, the period of the data window, and the instrument, on which this model will be trading. In general, this abstract model will contain parameters that will need to be filled once you need to create a specific _model instance_. It is obvious that under this approach an abstract model can be a parent of multiple instances of the models, which differ in their parameters.

### Complete rejection of the accounting of the net position

Many developers are trying to keep track of the aggregated position. However, we can see from the above, that neither the size of the aggregated position, nor its dynamics, are relevant to a particular instance of the model. The model can be short, while the aggregated position may not exist at all (neutral aggregated position). On the contrary, the aggregated position may be short, while the model will have a long position.

In fact, let's consider these cases in more detail. Assume that one instrument trades with three different trading tactics, each of which has its own independent system of money management. Also assume that the first of the three systems decided to sell three contracts without cover, or simply put, to make a short position, with a volume of three contracts. After the completion of the deals, the net position will consist solely from the deals of the first trading system, its volume will be minus three contract, or simply three short contract without cover. After some time, the second trading system makes the decision to buy 4 contracts of the same asset.

As a result, the net position will change, and will be consisting of 1 long contract. This time it will include the contributions of two trading systems. Further, the third trading system enters the scene and makes a short position with the same asset, with a volume of one standard contract. The net position will become neutral because -3 short\+ 4 long \- 1 short = 0\.

Does the absence of the net position mean that all of the three trading systems are not on the market? Not at all. Two of them hold four contracts without cover, which means that the cover will be made over time. On the other hand, the third system holds 4 long contract, which are yet to be sold back. Only when the full repayment of the four short contracts is complete, and a covered sale of four long contracts is made, the neutral position will mean a real lack of positions in all three systems.

We can, of course, each time reconstruct the entire sequence of actions for each of the models, and thereby determine its specific contribution in the size of the current position, but a much simpler method exists. This method is simple - it is necessary to completely abandon the accounting of the aggregated position, which can be any size and which can depend on both external (eg, manual trading), as well as internal factors (the work of other models of the EA on one instrument). Since the current aggregated position can't be depended on, then how do we account the actions of a specific model instance?

The simplest and most effective way would be to equip each instance of a model with its own table of orders, which would consider all of the orders - both, pending, and those initiated by a deal or deleted. Extensive information on orders is stored on the trading server. Knowing the ticket of the order, we can get almost any information on an order, ranging from the time of its opening and to its volume.

The only thing we need to do is to link the ticket order with a specific instance of the model. Each model instance will have to contain its individual instance of a special class - the table of orders, which would contain a list of the current orders, set out by the model instance.

### Projecting an abstract trading model

Now let's try to describe the common abstract class of the model, on which specific trading tactics will be based. Since the EA should use multiple models (or unlimited), it is obvious that this class should have a uniform interface through which an external power expert will give the signals.

For example, this interface may be the Processing() function. Simply put, each CModel class will have its Processing() function. This function will be called every tick or every minute, or upon the occurrence of a new event of Trade type.

Here is a simple example of solving this task:

```
class CModel
{
protected:
   string            m_name;
public:
   void             CModel(){m_name="Model base";}
   bool virtual      Processing(void){return(true);}
};

class cmodel_macd : public CModel
{
public:
   void              cmodel_macd(){m_name="MACD Model";}
   bool              Processing(){Print("Model name is ", m_name);return(true);}
};

class cmodel_moving : public CModel
{
public:
   void              cmodel_moving(){m_name="Moving Average";}
   bool              Processing(){Print("Model name is ", m_name);return(true);}
};

cmodel_macd     *macd;
cmodel_moving   *moving;
```

Let's figure out how this code works. The CModel base class contains one protected variable of string type called the m\_name. The "protected" keyword allows the use of this variable by the class heirs, so its descendants will already contain this variable. Further, the base class defines the Processing() virtual function. In this case the ['virtual](https://www.mql5.com/en/docs/basis/oop/virtual)' word indicates that this is a wrapper or the interface between the Expert Advisor and the specific instance of the model.

Any class, inherited from the CModel, will be guaranteed to have the Processing() interface for interaction. The implementation of code of this function is delegated to its descendants. This delegation is obvious, since the inner workings of models may differ significantly from each other, and therefore there is no common generalizations that could be located on a general level CModel.

Further is the description of two classes cmodel\_macd and cmodel\_moving. Both are generated from the CModel class, therefore both have their own instances of the Processing() function and the m\_name variable. Note that the internal implementation of the Processing() function of both models is different. In first model, it consists of the Print ("It is cmodel\_macd. Model name is ", m\_name), inthe secondofPrint("It is cmodel\_moving. Modelnameis ", m\_name). Next, two pointers are created, each of them may point to a specific instance of the model, one to the class of cmodel\_macd type, and the other to cmodel\_moving type.

In the OnInit function these pointers inherit the dynamically created classes-models, after which within the OnEvent() function  a Processing() function is called, which is contained in each class. Both pointers are announced at a global level, so even after exiting the OnInit() function, the classes created in it are not deleted, but continue to exist on a global level. Now, every five seconds, the OnTimer() function will sample both models in turn, calling in them the appropriate Processing() function.

This primitive system of sampling the models, which we have just created, lacks flexibility and scalability. What do we do if we want to work with several dozens of such models? Working with each one of them separately is inconvenient. It would be much easier to collect all of the models into a single community, for example an array, and then iterate over all of the elements of this array by calling the Processing() function, of each such element.

But the problem is that the organization of arrays requires that the data, stored in them, is of the same type. In our case, although the model cmodel\_macd and cmodel\_moving are very similar to each other, they are not identical, which automatically makes it impossible to use them in arrays.

Fortunately, the arrays are not the only way to summarize the data, there are other more flexible and scalable generalizations. One of them is the technique of linked lists. Its working scheme is simple. Each item that is included in the overall list should contain two pointers. One pointer points to the previous list item, the second - to the next one.

Also, knowing the index number of the item, you can always refer to it. When you want to add or delete an item, it is enough to rebuild its pointers, and the pointers of the neighboring items, so that they consistently refer to each other. Knowing the internal organization of such communities is not necessary, it is enough to understand their common device.

The standard installation of MetaTrader 5 includes a special auxiliary [CList](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist) class, which provides the opportunity to work with linked lists. However, the element of this list can only be an object of [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject) type, since only they have the special pointers for working with linked lists. On its own, the [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject) class is rather primitive, being simply an interface for interacting with the [CList](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist) class.

You can see this by taking a look at its implementation:

```
//+------------------------------------------------------------------+
//|                                                       Object.mqh |
//|                      Copyright © 2010, MetaQuotes Software Corp. |
//|                                       https://www.metaquotes.net/ |
//|                                              Revision 2010.02.22 |
//+------------------------------------------------------------------+
#include "StdLibErr.mqh"
//+------------------------------------------------------------------+
//| Class CObject.                                                   |
//| Purpose: Base class element storage.                             |
//+------------------------------------------------------------------+
class CObject
  {
protected:
   CObject          *m_prev;               // previous list item
   CObject          *m_next;               // next list item

public:
                     CObject();
   //--- methods of access to protected data
   CObject          *Prev()                { return(m_prev); }
   void              Prev(CObject *node)   { m_prev=node;    }
   CObject          *Next()                { return(m_next); }
   void              Next(CObject *node)   { m_next=node;    }
   //--- methods for working with files
   virtual bool      Save(int file_handle) { return(true);   }
   virtual bool      Load(int file_handle) { return(true);   }
   //--- method of identifying the object
   virtual int       Type() const          { return(0);      }

protected:
   virtual int       Compare(const CObject *node,int mode=0) const { return(0); }
  };
//+------------------------------------------------------------------+
//| Constructor CObject.                                             |
//| INPUT:  no.                                                      |
//| OUTPUT: no.                                                      |
//| REMARK: no.                                                      |
//+------------------------------------------------------------------+
void CObject::CObject()
  {
//--- initialize protected data
   m_prev=NULL;
   m_next=NULL;
  }
//+------------------------------------------------------------------+
```

As can be seen, the basis of this class are two pointers, which the typical features are implemented for.

Now the most important part. Owing to the mechanism of inheritance, it is possible to include this class into the trading model, which means that the class of the trading model can be included into a list of CList type! Let's try to do this.

And so, we will make our abstract CModel class as a descendant of the CObject class:

```
class CModel : public CObject
```

Since our classes cmodel\_moving and cmodel\_average are inherited from CModel class, they include the data and methods of CObject class, therefore, they can be included in the list of CList type. The source code, which creates the two conditional trading models, places them in the list, and sequentially samples each tick, is presented below:

```
//+------------------------------------------------------------------+
//|                                            ch01_simple_model.mq5 |
//|                            Copyright 2010, Vasily Sokolov (C-4). |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, Vasily Sokolov (C-4)."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <Arrays\List.mqh>

// Base model
class CModel:CObject
{
protected:
   string            m_name;
public:
        void              CModel(){m_name="Model base";}
        bool virtual      Processing(void){return(true);}
};

class cmodel_macd : public CModel
{
public:
   void              cmodel_macd(){m_name="MACD Model";}
   bool              Processing(){Print("Processing ", m_name, "...");return(true);}
};

class cmodel_moving : public CModel
{
public:
   void              cmodel_moving(){m_name="Moving Average";}
   bool              Processing(){Print("Processing ", m_name, "...");return(true);}
};

//Create list of models
CList *list_models;

void OnInit()
{
   int rezult;
   // Great two pointer
   cmodel_macd          *m_macd;
   cmodel_moving        *m_moving;
   list_models =        new CList();
   m_macd   =           new cmodel_macd();
   m_moving =           new cmodel_moving();
   //Check valid pointer
   if(CheckPointer(m_macd)==POINTER_DYNAMIC){
      rezult=list_models.Add(m_macd);
      if(rezult!=-1)Print("Model MACD successfully created");
      else          Print("Creation of Model MACD has failed");
   }
   //Check valid pointer
   if(CheckPointer(m_moving)==POINTER_DYNAMIC){
      rezult=list_models.Add(m_moving);
      if(rezult!=-1)Print("Model MOVING AVERAGE successfully created");
      else          Print("Creation of Model MOVING AVERAGE has failed");
   }
}

void OnTick()
{
   CModel               *current_model;
   for(int i=0;i<list_models.Total();i++){
      current_model=list_models.GetNodeAtIndex(i);
      current_model.Processing();
   }
}

void OnDeinit(const int reason)
{
   delete list_models;
}
```

Once this program is compiled and run, similar lines, indicating the normal operation of the EA, should appear in the journal "Experts".

```
2010.10.10 14:18:31     ch01_simple_model (EURUSD,D1)   Prosessing Moving Average...
2010.10.10 14:18:31     ch01_simple_model (EURUSD,D1)   Processing MACD Model...
2010.10.10 14:18:21     ch01_simple_model (EURUSD,D1)   Model MOVING AVERAGE was created successfully
2010.10.10 14:18:21     ch01_simple_model (EURUSD,D1)   Model MACD was created successfully
```

Let's analyze in detailhow this code works. So, as mentioned above, our basic trading model CModel is derived from the class [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject), which gives us the right to include the descendants of the basic model in the list of CList type:

```
rezult=list_models.Add(m_macd);
rezult=list_models.Add(m_moving);
```

The organization of data requires working with pointers. Once the pointers of specific models are created at the local level of [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) function and are entered into the global list list\_models, the need for them disappears, and they can be safely destructed, along with other variables of this function.

In general, a distinguishing feature of the proposed model is that the only global variable (in addition to the model classes themselves) is a dynamically linked list of these models. Thus, from the beginning, there is support of a high degree of encapsulation of the project.

If the creation of the model failed for some reason (for example, the values of the required parameters were listed incorrectly), then this model will not be added to the list. This will not affect the overall work of the EA, since it will handle only those models that were successfully added to the list.

The sampling of created models is made in the [OnTick()](https://www.mql5.com/en/docs/basis/function/events#ontick) function. It consists of a for loop. In this loop the number of elements is determined, after which there is a serial passage from the first element of the cycle (i = 0) to the last (i <list\_models.Total();i++):

```
CModel               *current_model;
for(int i=0;i<list_models.Total();i++){
   current_model=list_models.GetNodeAtIndex(i);
   current_model.Processing();
}
```

The pointer to the CModel base class is used as a universal adapter. This ensures that any function that is supported by this indicator will be available to the derivative models. In this case, we need the only Processing() function. Each model has its own version of Processing(), the internal implementation of which may differ from similar functions of other models. Overloading this function is not necessary, it can only exist in one form: not having any input parameters and returning the value of bool type.

Tasks that fall upon the "shoulders" of this function are extensive:

1. The function should independently determine the current market situation based on its own trading models.
2. After a decision is made to enter the market , the function should independently calculate the required amount of collateral, involved in the deal (the margin), the deal volume, the value of the maximum possible loss or the profit level.
3. The behavior of the model should be correlated with its previous actions. For example, if there is a short position, initiated by the model, then building it further in the future may be impossible. All these verifications should be carried out within the Processing() function.
4. Each of these functions should have access to the common parameters, such as the account status. Based on this data, this function should carry out its own money management, using the parameters embedded in its model. For example, if money management in one of the models is done through the means of the optimum formula f, then its value should be different for each of its models.

Obviously, the Processing() function,  due to the magnitude of the tasks laid upon it, will rely on the developed apparatus of the auxiliary classes, those that are included in the MetaTrader 5, as well as those that are designed specifically for this solution.

As can be seen, most of the work is delegated to specific instances of the model. The external level of the EA gives the control in turn to each model, and its work is completed on this. What will be done by the specific model will depend on its internal logic.

In general, the system of interaction, which we have built, can be described by the following scheme:

[https://c.mql5.com/2/2/5txbv.gif](https://c.mql5.com/2/2/5txbv.gif "https://c.mql5.com/2/2/5txbv.gif")

![](https://c.mql5.com/2/2/5txbv__1.gif)

Note that although the sorting of models, as presented in the above code, occurs inside the OnTick() function, it does not necessary have to be so. The sorting cycle can be easily placed in any other desired function, such as the OnTrade() or [OnTimer()](https://www.mql5.com/en/docs/basis/function/events#ontimer).

### The table of virtual orders - the basis of the model

Once we have combined all of the trading models into a single list, it's time to describe the process of trade. Let's go back to the CModel class  and try to supplement it with additional data and functions, which could be based upon in the trading process.

As mentioned above, the new net position paradigm defines the different rules for working with orders and deals. In MetaTrader 4, each deal is accompanied by its order, which existed on the tab "Trade" from the moment of its issue and until the termination of the order or the closing of the deal, initiated by it.

In MetaTrader 5 [the pending orders](https://www.mql5.com/en/docs/trading/orderstotal) only exist until the moment of the actual completion of the [deal](https://www.mql5.com/en/docs/trading/historydealstotal). After the deal, or the entrance of the market on it, is implemented, these orders pass to the history of orders, which is stored on the trading server. This situation creates uncertainty. Suppose the EA put out an order, which was executed. The aggregated position has changed. After some time, the EA needs to close its position.

Closing the specific order, as it could be done in MetaTrader 4, can not be done, as there is a lack of the very concept of closure of orders, we can close the [position](https://www.mql5.com/en/docs/trading/positionselect) or a part of it. The question is, what part of the position should be closed. Alternatively, we can look through all of the [historical orders](https://www.mql5.com/en/docs/trading/historyorderstotal), select the ones which have been put out by the EA, and then correlate these orders with the current market situation and, if necessary, block their counter orders. This method contains many difficulties.

For example, how can we determine that the orders were not already blocked in the past? We can take another way, assuming that the current position belongs exclusively to the current EA. This option can be used only if you intend to trade with one EA, trading on one strategy. All of these methods are not able to elegantly solve the challenges we face.

The most obvious and simplest solution would be to store all of the necessary information about the orders of the current model (ie, orders that are not blocked by opposite deals) within the model itself.

For example, if the model put out an order, its ticket is recorded in a special area of the memory of this model, for example, it can be organized with the help of an already familiar to us system of linked lists.

Knowing the ticket order, you can find almost any information about it, so all we need - is to link the ticket order with the model which put it out. Let the ticket order be stored in the **CTableOrder** special class. In addition to the ticket, it can accommodate the most essential information, for example, the volume of orders, the time of its installation, the magic number, etc.

Let's see how this class is structured:

```
#property copyright "Copyright 2010, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"

#include <Trade\_OrderInfo.mqh>
#include <Trade\_HistoryOrderInfo.mqh>
#include <Arrays\List.mqh>
class CTableOrders : CObject
{
private:
   ulong             m_magic;       // Magic number of the EA that put out the order
   ulong             m_ticket;      // Ticket of the basic order
   ulong             m_ticket_sl;    // Ticket of the simulated-Stop-Loss order, assigned with the basic order
   ulong             m_ticket_tp;    // Ticket of the simulated-Take-Profit, assigned with the basic order
   ENUM_ORDER_TYPE   m_type;         // Order type
   datetime          m_time_setup;  // Order setup time
   double            m_price;       // Order price
   double            m_sl;          // Stop Loss price
   double            m_tp;          // Take Profit price
   double            m_volume_initial;  // Order Volume
public:
                     CTableOrders();
   bool              Add(COrderInfo &order_info, double stop_loss, double take_profit);
   bool              Add(CHistoryOrderInfo &history_order_info, double stop_loss, double take_profit);
   double            StopLoss(void){return(m_sl);}
   double            TakeProfit(void){return(m_tp);}
   ulong             Magic(){return(m_magic);}
   ulong             Ticket(){return(m_ticket);}
   int               Type() const;
   datetime          TimeSetup(){return(m_time_setup);}
   double            Price(){return(m_price);}
   double            VolumeInitial(){return(m_volume_initial);}
};

CTableOrders::CTableOrders(void)
{
   m_magic=0;
   m_ticket=0;
   m_type=0;
   m_time_setup=0;
   m_price=0.0;
   m_volume_initial=0.0;
}

bool CTableOrders::Add(CHistoryOrderInfo &history_order_info, double stop_loss, double take_profit)
{
   if(HistoryOrderSelect(history_order_info.Ticket())){
      m_magic=history_order_info.Magic();
      m_ticket=history_order_info.Ticket();
      m_type=history_order_info.Type();
      m_time_setup=history_order_info.TimeSetup();
      m_volume_initial=history_order_info.VolumeInitial();
      m_price=history_order_info.PriceOpen();
      m_sl=stop_loss;
      m_tp=take_profit;
      return(true);
   }
   else return(false);
}

bool CTableOrders::Add(COrderInfo &order_info, double stop_loss, double take_profit)
{
   if(OrderSelect(order_info.Ticket())){
      m_magic=order_info.Magic();
      m_ticket=order_info.Ticket();
      m_type=order_info.Type();
      m_time_setup=order_info.TimeSetup();
      m_volume_initial=order_info.VolumeInitial();
      m_price=order_info.PriceOpen();
      m_sl=stop_loss;
      m_tp=take_profit;
      return(true);
   }
   else return(false);
}

int   CTableOrders::Type() const
{
   return((ENUM_ORDER_TYPE)m_type);
}
```

Similar to CModel class, the CTableOrders class is inherited from [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject). Just like the classes of the models, we will place instances of CTableOrders into the ListTableOrders list of [CList](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist) type.

In addition to its own ticket order (m\_tiket), the class contains information about the magic number ( [ORDER\_MAGIC](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties)) of the EA that put it out, its type, opening price, volume, and the level of the estimated overlap of orders: stoploss (m\_sl) and takeprofit (m\_tp). On the last two values, we need to speak separately. It is obvious that any deal should sooner or later be closed by an opposite deal. The opposite deal can be initiated on the basis of the current market situation or the partial close of the position at a predetermined price, at the time of its conclusion.

In MetaTrader4, such "unconditional exits from position" are special types of exits: StopLoss and TakeProfit. The distinguishing feature of MetaTrader 4 is the fact that these levels apply to a specific orders. For example, if a stop occurs in one of the active orders, it will not affect the other open orders on this instrument.

In MetaTrader 5, this is somewhat different. Although for each of the set orders, among other things, you can specify a price of the StopLoss and the TakeProfit, these levels will not act against a specific order, in which these prices were set, but in respect to the whole position on this instrument.

Suppose there is an open BUYposition for EURUSD of 1 standard lot without the levels of StopLoss and TakeProfit. Some time later, another order is put out for EURUSD to buy 0.1 lot, with the set levels of StopLoss and TakeProfit \- each at a distance of 100 points from the current price. After some time, the price reaches the level of StopLossor the level of TakeProfit. When this occurs, the entire position with a size of 1.1 lot at EURUSDwill be closed.

In other words, the StopLoss and TakeProfit can be set only in relation to the aggregated position, and not against a particular order. On this basis, it becomes impossible to use these orders in multi-system EAs. This is obvious, because if one system will put out its own StopLoss and TakeProfit, then it will apply to all other systems, the interests of which is already included in the aggregated position of the instrument!

Consequently, each of the subsystems of the trading EA should only use their own, internal StopLoss and TakeProfit for each order individually. Also, this can concept can be derived from the fact that even within the same trading system, different orders may have different levels of StopLoss and TakeProfit, and as already mentioned above, in MetaTrader 5, these outputs can not be designated to individual orders.

If we place, within the virtual orders, synthetic levels of StopLoss and TakeProfit, the EA will be able to independently block the existing orders once the price reaches or exceeds these levels. After blocking these orders, they can be safely removed from the list of active orders. The way this is done is described below.

The class CTableOrders, aside from its own data, contains a highly important Add() function. This function receives the order ticket, which needs to be recorded into the table. In addition to the order ticket, this function receives the levels of the virtual StopLoss and TakeProfit. First, the Add() function tries to allocate the order among the historical orders, which are stored on the server. If it is able to do this, it inputs the information on the ticket into the instance of the class history\_order\_info, and then begins to enter the information through it into the new TableOrders element. Further, this element is added to the list of orders. If the selection of the order could not be completed, then, perhaps, we are dealing with a pending order, so we have to try to allocate this order from the current orders via the [OrderSelect()](https://www.mql5.com/en/docs/trading/orderselect) function. In case of a successful selection of this order, the same actions are taken as for the historical order.

At the moment, before the introduction of the [structure](https://www.mql5.com/en/docs/basis/types/classes), which describes the event [Trade](https://www.mql5.com/en/docs/runtime/event_fire#trade), working with pending orders for multi-system EAs is difficult. Certainly, after the introduction of this structure, it will become possible to design EAs, based upon pending orders. Moreover, if an orders table is present, virtually any trading strategy with pending orders can be moved into performance on the market. For these reasons, all of the trading models presented in the article will have [market execution](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type) (ORDER\_TYPE\_BUY or ORDER\_TYPE\_SELL).

### CModel \- the base class  of the trading model

And so, when the table of orders is fully designed, comes the time to describe the full version of the basic model CModel:

```
class CModel : public CObject
{
protected:
   long              m_magic;
   string            m_symbol;
   ENUM_TIMEFRAMES   m_timeframe;
   string            m_model_name;
   double            m_delta;
   CTableOrders      *table;
   CList             *ListTableOrders;
   CAccountInfo      m_account_info;
   CTrade            m_trade;
   CSymbolInfo       m_symbol_info;
   COrderInfo        m_order_info;
   CHistoryOrderInfo m_history_order_info;
   CPositionInfo     m_position_info;
   CDealInfo         m_deal_info;
   t_period          m_timing;
public:
                     CModel()  { Init();   }
                     ~CModel() { Deinit(); }
   string            Name(){return(m_model_name);}
   void              Name(string name){m_model_name=name;}
   ENUM_TIMEFRAMES    Timeframe(void){return(m_timeframe);}
   string            Symbol(void){return(m_symbol);}
   void              Symbol(string set_symbol){m_symbol=set_symbol;}
   bool virtual      Init();
   void virtual      Deinit(){delete ListTableOrders;}
   bool virtual      Processing(){return (true);}
   double            GetMyPosition();
   bool              Delete(ENUM_TYPE_DELETED_ORDER);
   bool              Delete(ulong Ticket);
   void              CloseAllPosition();
   //bool virtual      Trade();
protected:
   bool              Add(COrderInfo &order_info, double stop_loss, double take_profit);
   bool              Add(CHistoryOrderInfo &history_order_info, double stop_loss, double take_profit);

   void              GetNumberOrders(n_orders &orders);
   bool              SendOrder(string symbol, ENUM_ORDER_TYPE op_type, ENUM_ORDER_MODE op_mode, ulong ticket, double lot,
                              double price, double stop_loss, double take_profit, string comment);
};
```

The data from this class contains the fundamental constants of any trading model.

This is the magic number (m\_magic), the symbol on which the model will be launched, (m\_symbol) the time-frame (m\_timeframe), and the name of the most traded model (m\_name).

In addition, the model includes the, already familiar to us, class of the orders table, (CTableOrders \* table) and the list, in which the instances of this table will be kept, one copy for each order( [CList](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist)\*ListTableOrders). Since all of the data will be created dynamically, as the need arises, the work with this data will be carried out through [pointers](https://www.mql5.com/en/docs/basis/types/object_pointers).

This is followed by the variable m\_delta. This variable should hold a special coefficient for calculating the current lot in the formulas for money managing. For example, for the fixed-fractional formula of capitalization, this variable can store a share of the account, which can be risked, for instance,for the risk of 2% of the account, this variable must be equal to 0.02. For the more aggressive methods, for example, for the optimal method, fthis variable might be larger.

What is important in this variable is that it permits the individual selection of the risk for each model, which is part of a single EA. If the capitalization formula is not used, then filling it out is not required.By default it is equal to 0.0.

Next follows the inclusion of all auxiliary trading classes, which are designed to facilitate the receipt and processing of all of the required information, ranging from [account information](https://www.mql5.com/en/docs/account) and to information about  position. It is understood that the derivatives of specific trade models need to actively use these auxiliary classes, and not the regular features of type [OrderSelect](https://www.mql5.com/en/docs/trading/orderselect) or [OrderSend](https://www.mql5.com/en/docs/trading/ordersend).

The variable m\_timing needs to be described separately. During the process of the work of the EA, it is necessary to call certain events at certain time intervals. The [OnTimer()](https://www.mql5.com/en/docs/basis/function/events#ontimer) function is not suitable for this, since the different models may exist in different time intervals.

For example, some events need to be called at each [new bar](https://www.mql5.com/en/articles/159). For the model, trading on an hourly graph, such events should be called each hour, for a model, trading on a daily graph - each new day bar. It is clear that these models have different time settings, and each must be stored, respectively, in its own model. The structure t\_period, included in the CModel class, allows you to store these settings separately, each in its model.

Here is what the structure looks like:

```
struct t_period
{
   datetime m1;
   datetime m2;
   datetime m3;
   datetime m4;
   datetime m5;
   datetime m6;
   datetime m10;
   datetime m12;
   datetime m15;
   datetime m20;
   datetime m30;
   datetime h1;
   datetime h2;
   datetime h3;
   datetime h4;
   datetime h6;
   datetime h8;
   datetime h12;
   datetime d1;
   datetime w1;
   datetime mn1;
   datetime current;
};
```

As can be seen, it includes the usual [listing of timeframes](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes). To see whether [a new bar has occurred](https://www.mql5.com/en/articles/159), you need to compare the time of the last bar, with the time that was recorded in the structure t\_period. If times do not match, then a new bar has occurred, and the time in the structure needs to be updated to the time of the current bar and return a positive result (true). If the time of the last bar and the structure are identical, this means that the new bar has not yet occurred, and a negative result needs to be returned (false).

Here is a function that works, based on the algorithm described:

```
bool timing(string symbol, ENUM_TIMEFRAMES tf, t_period &timeframes)
{
   int rez;
   MqlRates raters[1];
   rez=CopyRates(symbol, tf, 0, 1, raters);
   if(rez==0)
   {
      Print("Error timing");
      return(false);
   }
   switch(tf){
      case PERIOD_M1:
         if(raters[0].time==timeframes.m1)return(false);
         else{timeframes.m1=raters[0].time; return(true);}
      case PERIOD_M2:
         if(raters[0].time==timeframes.m2)return(false);
         else{timeframes.m2=raters[0].time; return(true);}
      case PERIOD_M3:
         if(raters[0].time==timeframes.m3)return(false);
         else{timeframes.m3=raters[0].time; return(true);}
      case PERIOD_M4:
         if(raters[0].time==timeframes.m4)return(false);
         else{timeframes.m4=raters[0].time; return(true);}
     case PERIOD_M5:
         if(raters[0].time==timeframes.m5)return(false);
         else{timeframes.m5=raters[0].time; return(true);}
     case PERIOD_M6:
         if(raters[0].time==timeframes.m6)return(false);
         else{timeframes.m6=raters[0].time; return(true);}
     case PERIOD_M10:
         if(raters[0].time==timeframes.m10)return(false);
         else{timeframes.m10=raters[0].time; return(true);}
     case PERIOD_M12:
         if(raters[0].time==timeframes.m12)return(false);
         else{timeframes.m12=raters[0].time; return(true);}
     case PERIOD_M15:
         if(raters[0].time==timeframes.m15)return(false);
         else{timeframes.m15=raters[0].time; return(true);}
     case PERIOD_M20:
         if(raters[0].time==timeframes.m20)return(false);
         else{timeframes.m20=raters[0].time; return(true);}
     case PERIOD_M30:
         if(raters[0].time==timeframes.m30)return(false);
         else{timeframes.m30=raters[0].time; return(true);}
     case PERIOD_H1:
         if(raters[0].time==timeframes.h1)return(false);
         else{timeframes.h1=raters[0].time; return(true);}
     case PERIOD_H2:
         if(raters[0].time==timeframes.h2)return(false);
         else{timeframes.h2=raters[0].time; return(true);}
     case PERIOD_H3:
         if(raters[0].time==timeframes.h3)return(false);
         else{timeframes.h3=raters[0].time; return(true);}
     case PERIOD_H4:
         if(raters[0].time==timeframes.h4)return(false);
         else{timeframes.h4=raters[0].time; return(true);}
     case PERIOD_H6:
         if(raters[0].time==timeframes.h6)return(false);
         else{timeframes.h6=raters[0].time; return(true);}
     case PERIOD_H8:
         if(raters[0].time==timeframes.h8)return(false);
         else{timeframes.h8=raters[0].time; return(true);}
     case PERIOD_H12:
         if(raters[0].time==timeframes.h12)return(false);
         else{timeframes.h12=raters[0].time; return(true);}
     case PERIOD_D1:
         if(raters[0].time==timeframes.d1)return(false);
         else{timeframes.d1=raters[0].time; return(true);}
     case PERIOD_W1:
         if(raters[0].time==timeframes.w1)return(false);
         else{timeframes.w1=raters[0].time; return(true);}
     case PERIOD_MN1:
         if(raters[0].time==timeframes.mn1)return(false);
         else{timeframes.mn1=raters[0].time; return(true);}
     case PERIOD_CURRENT:
         if(raters[0].time==timeframes.current)return(false);
         else{timeframes.current=raters[0].time; return(true);}
     default:
         return(false);
   }
}
```

At the moment there is no possibility of a sequential sorting of the structures. Such sorting may be required when you need to create multiple instances in a cycle of the same trading model, trading on different time-frames. So I had to write a special function-sorters regarding the structure t\_period.

Here is thesourcecodeof thisfunction:

```
int GetPeriodEnumerator(uchar n_period)
{
   switch(n_period)
   {
      case 0: return(PERIOD_CURRENT);
      case 1: return(PERIOD_M1);
      case 2: return(PERIOD_M2);
      case 3: return(PERIOD_M3);
      case 4: return(PERIOD_M4);
      case 5: return(PERIOD_M5);
      case 6: return(PERIOD_M6);
      case 7: return(PERIOD_M10);
      case 8: return(PERIOD_M12);
      case 9: return(PERIOD_M15);
      case 10: return(PERIOD_M20);
      case 11: return(PERIOD_M30);
      case 12: return(PERIOD_H1);
      case 13: return(PERIOD_H2);
      case 14: return(PERIOD_H3);
      case 15: return(PERIOD_H4);
      case 16: return(PERIOD_H6);
      case 17: return(PERIOD_H8);
      case 18: return(PERIOD_H12);
      case 19: return(PERIOD_D1);
      case 20: return(PERIOD_W1);
      case 21: return(PERIOD_MN1);
      default:
         Print("Enumerator period must be smallest 22");
         return(-1);
   }
}
```

All of these functions are conveniently combined into a single file in the folder \\\Include. Let's name it Time.mqh.

This is what will be included in our base class CModel:

```
…
#incude <Time.mqh>
…
```

In addition to simple functions get/setof type Name(),Timeframe() andSymbol(), class CModelcontain complex functions of typeInit(),GetMyPosition(), Delete(), CloseAllPosition() и Processing(). The designation of the last function should already be familiar to you, we will discuss in greater detail its internal structure later, but for now let's start with a description of the main functions of the base class CModel.

The CModel::Add() function dynamically creates an instance of class CTableOrders, and then fills it using the appropriate CTabeOrders::Add() function. Its principle of operation has been described above. After being filled, this item gets included into the general list of all orders of the current model (ListTableOrders.Add (t)).

The CModel::Delete() function, on the other hand, deleted the element of CTableOrderstype from the list of active orders. To do this you must specify the ticket of the orders, which must be deleted. The principles of its works is simple. The function sequentially sorts through the entire table of orders in search for the order with the right ticket. If it finds such an order, it deletes it.

The CModel::GetNumberOrders() function counts up the number of active orders. Itfillsthe specialstructuren\_orders:

```
struct n_orders
{
   int all_orders;
   int long_orders;
   int short_orders;
   int buy_sell_orders;
   int delayed_orders;
   int buy_orders;
   int sell_orders;
   int buy_stop_orders;
   int sell_stop_orders;
   int buy_limit_orders;
   int sell_limit_orders;
   int buy_stop_limit_orders;
   int sell_stop_limit_orders;
};
```

As can be seen, after it is called we can find out how many specific types of orders have been set. For example, to obtain the number of all short orders, you must read all of the values of short\_ordersof the instance n\_orders.

The CModel::SendOrder() function is the basic and only function for the factual sending of orders to the trading server. Instead of each particular model having its own algorithm for sending orders to the server, the SendOrder() function defines the general procedure for these submission. Regardless of the model, the process of putting out orders is associated with the same [checks](https://www.mql5.com/en/docs/trading/ordercheck), which are efficiently carried out in a centralized location.

Let's get ourselvesfamiliarizedwith the sourcecodeof thisfunction:

```
bool CModel::SendOrder(string symbol, ENUM_ORDER_TYPE op_type, ENUM_ORDER_MODE op_mode, ulong ticket,
                          double lot, double price, double stop_loss, double take_profit, string comment)
{
   ulong code_return=0;
   CSymbolInfo symbol_info;
   CTrade      trade;
   symbol_info.Name(symbol);
   symbol_info.RefreshRates();
   mm send_order_mm;

   double lot_current;
   double lot_send=lot;
   double lot_max=m_symbol_info.LotsMax();
   //double lot_max=5.0;
   bool rez=false;
   int floor_lot=(int)MathFloor(lot/lot_max);
   if(MathMod(lot,lot_max)==0)floor_lot=floor_lot-1;
   int itteration=(int)MathCeil(lot/lot_max);
   if(itteration>1)
      Print("The order volume exceeds the maximum allowed volume. It will be divided into ", itteration, " deals");
   for(int i=1;i<=itteration;i++)
   {
      if(i==itteration)lot_send=lot-(floor_lot*lot_max);
      else lot_send=lot_max;
      for(int i=0;i<3;i++)
      {
         //Print("Send Order: TRADE_RETCODE_DONE");
         symbol_info.RefreshRates();
         if(op_type==ORDER_TYPE_BUY)price=symbol_info.Ask();
         if(op_type==ORDER_TYPE_SELL)price=symbol_info.Bid();
         m_trade.SetDeviationInPoints(ulong(0.0003/(double)symbol_info.Point()));
         m_trade.SetExpertMagicNumber(m_magic);
         rez=m_trade.PositionOpen(m_symbol, op_type, lot_send, price, 0.0, 0.0, comment);
         // Sleeping is not to be deleted or moved! Otherwise the order will not have time to get recorded in m_history_order_info!!!
         Sleep(3000);
         if(m_trade.ResultRetcode()==TRADE_RETCODE_PLACED||
            m_trade.ResultRetcode()==TRADE_RETCODE_DONE_PARTIAL||
            m_trade.ResultRetcode()==TRADE_RETCODE_DONE)
         {
               //Print(m_trade.ResultComment());
               //rez=m_history_order_info.Ticket(m_trade.ResultOrder());
               if(op_mode==ORDER_ADD){
                  rez=Add(m_trade.ResultOrder(), stop_loss, take_profit);
               }
               if(op_mode==ORDER_DELETE){
                  rez=Delete(ticket);
               }
               code_return=m_trade.ResultRetcode();
               break;
         }
         else
         {
            Print(m_trade.ResultComment());
         }
         if(m_trade.ResultRetcode()==TRADE_RETCODE_TRADE_DISABLED||
            m_trade.ResultRetcode()==TRADE_RETCODE_MARKET_CLOSED||
            m_trade.ResultRetcode()==TRADE_RETCODE_NO_MONEY||
            m_trade.ResultRetcode()==TRADE_RETCODE_TOO_MANY_REQUESTS||
            m_trade.ResultRetcode()==TRADE_RETCODE_SERVER_DISABLES_AT||
            m_trade.ResultRetcode()==TRADE_RETCODE_CLIENT_DISABLES_AT||
            m_trade.ResultRetcode()==TRADE_RETCODE_LIMIT_ORDERS||
            m_trade.ResultRetcode()==TRADE_RETCODE_LIMIT_VOLUME)
         {
            break;
         }
      }
   }
   return(rez);
}
```

The first thing that this function does is it verifies the possibility of executing the stated volume of the trading server. This it does by using the CheckLot() function. There may be some [trading restrictions](https://www.mql5.com/en/articles/22) on the size of the position. They need to be taken into account.

Consider the following case: there is a limit on the size of a trading positions of 15 standard lots in both directions. The current position is long and equals to 3 lots. The trading model, based of its system of money  management, wants to open a long position with a volume of 18.6 lots.The CheckLot() function will return the corrected volume of the deal. In this case, it will be equal to 12 lots (since 3 lots out of 15 are already occupied by other deals). If the current open position was short, rather than long, then the function would returns 15 lots instead of 18.6. This is the maximum possible volume of positions.

After putting out 15 lots buy, the net position, in this case, will be 12 lots (3 - sell, 15 - buy). When another model overrides its initial short position of 3 lots, buy the aggregated position will become the maximum possible - 15 lots. Other signals to buy will not be processed until the model overrides some or all of its 15 lots buy. A possible volume for the requested deal has been exceed, the function returns a constant [EMPTY\_VALUE](https://www.mql5.com/en/docs/constants/namedconstants/typeconstants), such signal must be passed.

If the check for the possibility of the set volume has been successful, then calculations are made on the value of the required margin. The may not be enough funds in the account for the stated volume. For these purposes, there is a CheckMargin() function. If the margin is not enough, it will try to correct the order volume so that the current free margin allowed it to open. If the margin is not enough even to open the minimum amount, we are in a state of Margin-Call.

If there are currently no positions, and the margin is not used, it means only one thing - technical margin-call \- a state where it is impossible to open a deal. Without adding money to the account, we are unable to continue. If some margin is still in use, then we are left with nothing else but to wait until the transaction, which uses this margin, is closed. In any case, the lack of margin will return a constant [EMPTY\_VALUE](https://www.mql5.com/en/docs/constants/namedconstants/typeconstants).

A distinguishing feature of this function is the ability to divide of the current order into several independent deals. If the trading models use a system of capitalization of the account, then the required amount can easily exceed all conceivable limits (for example, the system of capitalization may require the opening of a deal with a volume of several hundred, and sometimes thousand, standard lots). Clearly is is impossible to ensure such an amount for a single deal. Typically, the trading conditions determine the size of a maximum deal of one hundred lots, but some trading servers have other restrictions, for example, on the MetaQuotes [Championship 2010](https://championship.mql5.com/2010/en/rules "https://championship.mql5.com/2010/en/rules") server [https://championship.mql5.com/2010/en/rules](https://championship.mql5.com/2010/en/rules "https://championship.mql5.com/2010/en/rules") this limitation was 5 lots. It is clear that such restrictions must be taken into account, and based on this, correctly calculate the actual volume of the deal.

First the number of orders, needed to implement the set volume, is counted. If the set amount does not exceed the maximum amount of the transaction, it requires only one pass for putting out this order. If the desired volume of the transactions exceeds the maximum possible volume, then this volume is divided into several parts. For example, you want to buy a 11.3 lots of EURUSD. The maximum size of the transaction on this instrument is 5.0 lots. Then the OrderSendfunction breaks up this volume into three orders: one order - of 5.0 lots, second order - 5.0 lots, third order - 1.3 lots.

Thus, instead of one order, there will be as many as three. Each of them will be listed in the table of orders, and will have their own independent settings, such as the virtual values of Stop Loss and Take Profit, the magic number, and other parameters. In the processing of such orders there should not be any difficulties, since the trading models are designed in such a way that they can handle any number of orders in their lists.

Indeed, all of the orders will have the same values TakeProfitand StopLoss. Each of them will be sequentially sorted by the LongClose and ShortClose functions. Once the right conditions for their closure occur, or they reach their thresholds SL and TP, they will all be closed.

Every order is sent to the server using the OrderSendfunction of the CTrade class. The most interesting detail of the work is hidden below.

The fact is that the assignment of an order may be twofold. The order to buy or sell can be sent upon the occurrence of the signal, or it may be an order for blocking the previously existing one. The OrderSendfunction must know the type of the sent order, since this is the function that actually places all of the orders in the table of orders, or removes them from the table upon the occurrence of [certain events](https://www.mql5.com/en/docs/eventfunctions).

If the order type you want to add is ADD\_ORDER. I.e. is an independent order, which needs to be placed in the table of orders, then the function adds information about this order into the table of orders. If the order is put out to override previously placed order (for example, in the occurrence of a virtual stop-loss), then it must have a type DELETE\_ORDER. After it has been put up, the function OrderSend manually removes the information about the order, with which it is linked from the list of orders. For this the function, in addition to the order type, inherits an order ticket, with which it is linked. If this is ADD\_ORDER, then the ticket can be filled with a simple zero.

### The first trading model, based on the crossover of moving averages

We discussed all of the major elements of the CModel base class. It is time to consider a specific trading class.

For these purposes, we will first create a simple trading model, based on a simple indicator MACD.

This model always will have the long position or a short one. As soon as the fast line crosses the slow line downwards, we will open a short position, while a long position, if there is one, will be closed. In the case of the upward crossover, we will open a long position, while a short position, if there is one, will be closed. In this model, we do not use protective stops and profit levels.

```
#include <Models\Model.mqh>
#include <mm.mqh>
//+----------------------------------------------------------------------+
//| This model uses MACD indicator.                                      |
//| Buy when it crosses the zero line downward                           |
//| Sell when it crosses the zero line upward                            |
//+----------------------------------------------------------------------+
struct cmodel_macd_param
{
   string            symbol;
   ENUM_TIMEFRAMES   timeframe;
   int               fast_ema;
   int               slow_ema;
   int               signal_ema;
};

class cmodel_macd : public CModel
{
private:
   int               m_slow_ema;
   int               m_fast_ema;
   int               m_signal_ema;
   int               m_handle_macd;
   double            m_macd_buff_main[];
   double            m_macd_current;
   double            m_macd_previous;
public:
                     cmodel_macd();
   bool              Init();
   bool              Init(cmodel_macd_param &m_param);
   bool              Init(string symbol, ENUM_TIMEFRAMES timeframes, int slow_ma, int fast_ma, int smothed_ma);
   bool              Processing();
protected:
   bool              InitIndicators();
   bool              CheckParam(cmodel_macd_param &m_param);
   bool              LongOpened();
   bool              ShortOpened();
   bool              LongClosed();
   bool              ShortClosed();
};

cmodel_macd::cmodel_macd()
{
   m_handle_macd=INVALID_HANDLE;
   ArraySetAsSeries(m_macd_buff_main,true);
   m_macd_current=0.0;
   m_macd_previous=0.0;
}
//this default loader
bool cmodel_macd::Init()
{
   m_magic      = 148394;
   m_model_name =  "MACD MODEL";
   m_symbol     = _Symbol;
   m_timeframe  = _Period;
   m_slow_ema   = 26;
   m_fast_ema   = 12;
   m_signal_ema = 9;
   m_delta      = 50;
   if(!InitIndicators())return(false);
   return(true);
}

bool cmodel_macd::Init(cmodel_macd_param &m_param)
{
   m_magic      = 148394;
   m_model_name = "MACD MODEL";
   m_symbol     = m_param.symbol;
   m_timeframe  = (ENUM_TIMEFRAMES)m_param.timeframe;
   m_fast_ema   = m_param.fast_ema;
   m_slow_ema   = m_param.slow_ema;
   m_signal_ema = m_param.signal_ema;
   if(!CheckParam(m_param))return(false);
   if(!InitIndicators())return(false);
   return(true);
}

bool cmodel_macd::CheckParam(cmodel_macd_param &m_param)
{
   if(!SymbolInfoInteger(m_symbol, SYMBOL_SELECT))
   {
      Print("Symbol ", m_symbol, " selection has failed. Check symbol name");
      return(false);
   }
   if(m_fast_ema == 0)
   {
      Print("Fast EMA must be greater than 0");
      return(false);
   }
   if(m_slow_ema == 0)
   {
      Print("Slow EMA must be greater than 0");
      return(false);
   }
   if(m_signal_ema == 0)
   {
      Print("Signal EMA must be greater than 0");
      return(false);
   }
   return(true);
}

bool cmodel_macd::InitIndicators()
{
   if(m_handle_macd==INVALID_HANDLE)
   {
      Print("Load indicators...");
      if((m_handle_macd=iMACD(m_symbol,m_timeframe,m_fast_ema,m_slow_ema,m_signal_ema,PRICE_CLOSE))==INVALID_HANDLE)
      {
         printf("Error creating MACD indicator");
         return(false);
      }
   }
   return(true);
}

bool cmodel_macd::Processing()
{
   //if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_DISABLED)return(false);
   //if(m_account_info.TradeAllowed()==false)return(false);
   //if(m_account_info.TradeExpert()==false)return(false);

   m_symbol_info.Name(m_symbol);
   m_symbol_info.RefreshRates();
   CopyBuffer(this.m_handle_macd,0,1,2,m_macd_buff_main);
   m_macd_current=m_macd_buff_main[0];
   m_macd_previous=m_macd_buff_main[1];
   GetNumberOrders(m_orders);
   if(m_orders.buy_orders>0)   LongClosed();
   else                        LongOpened();
   if(m_orders.sell_orders!=0) ShortClosed();
   else                        ShortOpened();
   return(true);
}

bool cmodel_macd::LongOpened(void)
{
   if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_DISABLED)return(false);
   if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_SHORTONLY)return(false);
   if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_CLOSEONLY)return(false);

   bool rezult, ticket_bool;
   double lot=0.1;
   mm open_mm;
   m_symbol_info.Name(m_symbol);
   m_symbol_info.RefreshRates();
   CopyBuffer(this.m_handle_macd,0,1,2,m_macd_buff_main);

   m_macd_current=m_macd_buff_main[0];
   m_macd_previous=m_macd_buff_main[1];
   GetNumberOrders(m_orders);

   //Print("LongOpened");
   if(m_macd_current>0&&m_macd_previous<=0&&m_orders.buy_orders==0)
   {
      //lot=open_mm.optimal_f(m_symbol, ORDER_TYPE_BUY, m_symbol_info.Ask(), 0.0, m_delta);
      lot=open_mm.jons_fp(m_symbol, ORDER_TYPE_BUY, m_symbol_info.Ask(), 0.1, 10000, m_delta);
      rezult=SendOrder(m_symbol, ORDER_TYPE_BUY, ORDER_ADD, 0, lot, m_symbol_info.Ask(), 0, 0, "MACD Buy");
      return(rezult);
   }
   return(false);
}

bool cmodel_macd::ShortOpened(void)
{
   if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_DISABLED)return(false);
   if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_LONGONLY)return(false);
   if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_CLOSEONLY)return(false);

   bool rezult, ticket_bool;
   double lot=0.1;
   mm open_mm;

   m_symbol_info.Name(m_symbol);
   m_symbol_info.RefreshRates();
   CopyBuffer(this.m_handle_macd,0,1,2,m_macd_buff_main);

   m_macd_current=m_macd_buff_main[0];
   m_macd_previous=m_macd_buff_main[1];
   GetNumberOrders(m_orders);

   if(m_macd_current<=0&&m_macd_previous>=0&&m_orders.sell_orders==0)
   {
      //lot=open_mm.optimal_f(m_symbol, ORDER_TYPE_SELL, m_symbol_info.Bid(), 0.0, m_delta);
      lot=open_mm.jons_fp(m_symbol, ORDER_TYPE_SELL, m_symbol_info.Bid(), 0.1, 10000, m_delta);
      rezult=SendOrder(m_symbol, ORDER_TYPE_SELL, ORDER_ADD, 0, lot, m_symbol_info.Bid(), 0, 0, "MACD Sell");
      return(rezult);
   }
   return(false);
}

bool cmodel_macd::LongClosed(void)
{
   if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_DISABLED)return(false);
   CTableOrders *t;
   int total_elements;
   int rez=false;
   total_elements=ListTableOrders.Total();
   if(total_elements==0)return(false);
   for(int i=total_elements-1;i>=0;i--)
   {
      if(CheckPointer(ListTableOrders)==POINTER_INVALID)continue;
      t=ListTableOrders.GetNodeAtIndex(i);
      if(CheckPointer(t)==POINTER_INVALID)continue;
      if(t.Type()!=ORDER_TYPE_BUY)continue;
      m_symbol_info.Refresh();
      m_symbol_info.RefreshRates();
      CopyBuffer(this.m_handle_macd,0,1,2,m_macd_buff_main);
      if(m_symbol_info.Bid()<=t.StopLoss()&&t.StopLoss()!=0.0)
      {

         rez=SendOrder(m_symbol, ORDER_TYPE_SELL, ORDER_DELETE, t.Ticket(), t.VolumeInitial(),
                       m_symbol_info.Bid(), 0.0, 0.0, "MACD: buy close buy stop-loss");
      }
      if(m_macd_current<0&&m_macd_previous>=0)
      {
         //Print("Long position closed by Order Send");
         rez=SendOrder(m_symbol, ORDER_TYPE_SELL, ORDER_DELETE, t.Ticket(), t.VolumeInitial(),
                       m_symbol_info.Bid(), 0.0, 0.0, "MACD: buy close by signal");
      }
   }
   return(rez);
}

bool cmodel_macd::ShortClosed(void)
{
   if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_DISABLED)return(false);
   CTableOrders *t;
   int total_elements;
   int rez=false;
   total_elements=ListTableOrders.Total();
   if(total_elements==0)return(false);
   for(int i=total_elements-1;i>=0;i--)
   {
      if(CheckPointer(ListTableOrders)==POINTER_INVALID)continue;
      t=ListTableOrders.GetNodeAtIndex(i);
      if(CheckPointer(t)==POINTER_INVALID)continue;
      if(t.Type()!=ORDER_TYPE_SELL)continue;
      m_symbol_info.Refresh();
      m_symbol_info.RefreshRates();
      CopyBuffer(this.m_handle_macd,0,1,2,m_macd_buff_main);
      if(m_symbol_info.Ask()>=t.StopLoss()&&t.StopLoss()!=0.0)
      {
         rez=SendOrder(m_symbol, ORDER_TYPE_BUY, ORDER_DELETE, t.Ticket(), t.VolumeInitial(),
                                 m_symbol_info.Ask(), 0.0, 0.0, "MACD: sell close buy stop-loss");
      }
      if(m_macd_current>0&&m_macd_previous<=0)
      {
         rez=SendOrder(m_symbol, ORDER_TYPE_BUY, ORDER_DELETE, t.Ticket(), t.VolumeInitial(),
                                 m_symbol_info.Ask(), 0.0, 0.0, "MACD: sell close by signal");
      }
   }
   return(rez);
}
```

The CModel base class does not impose any restrictions on the internal content of its descendants. The only thing that it makes mandatory, is the use of the Processing() interface function. All of the problems of the internal organization of this function are delegated to a specific class of models. There is no any universal algorithm that could be placed inside a function Processing(), therefore, there are no reason to impose on the descendants its method of how a particular model should be arranged. However, the internal structure of virtually any model can be standardized. Such standardization will greatly facilitate the understanding of an externaland even your code, and will make the model more "formulaic."

Each model must have its own initializer. The initializer of the model is responsible for loading the correct parameters, which are necessary for its operation, for example, in order for our model to work, we need to select the MACD indicator values, obtain a handle of the corresponding buffer, and in addition, of course,determine  the trading instrument and time-frame for the model. All of this must be done by the initializer.

The initializers of models - are simple overloaded methods of their classes. These methods have a common name Init. The fact is that MQL5 does not support the overloaded constructors, so there is no way to create the initializer of the model in its constructor, because the overload will be needed for input parameters. Although no one restricts us from indicating in the model's constructor the its basic parameters.

Every model should have three initializers. First - the default initializer. It must configure and load the model by default, without requesting the parameters. This can be very convenient for testing in the "as is" mode. For example, for our model, the default initializer as an instrument tooland time-frame of the model, will select the current graph and the current time-frame.

The settings of the MACDindicator will also be standard: fastEMA = 12, slowEMA = 26, signalMA = 9; If the model is required to be configured in a certain way, such initializer will no longer be suitable. We will need initializers with parameters. It is desirable (but not necessarily) to make two types. The first will receive its parameters as a classical function: Init (type param1, type param2, ..., type paramN). The second will find out the parameters of the model using a special structure, which saves these parameters. This option is sometimes more preferable because sometimes the number of parameters can be large, in which case it would be convenient to pass them through the structures.

Each model has the structure of its parameters. This structure can have any name, but it is preferable to call it by the pattern modelname\_param. Configuring the model - is a very important step in using the opportunities of multi-time-frame/multi-system/multi-currency trading. It is at this stage that it is determined, how and on what instrument this model will be trading on.

Our trading model has only four trading functions. The function for opening a long position: LongOpen, the function for opening a short position: ShortOpen, the function for closing a long position: LongClosed, and the function for closing a short position: ShortClosed. The work of the functions LongOpenи ShortOpenis trivial. Both receive the indicator value of the MACD previous bar, which is compared with the value of two bars before that. To avoid the "redrawing", the current (zero) bar will not be used.

If there is a downward crossover, then the ShortOpencalculates function using the functions, included in the header file mm.mqh, after which the necessary lot sends its command to the OrderSendfunction. The LongClose at this moment, on the contrary, closes all of the long positions in the model. This occurs because the function sequentially sorts all of the current open orders in the orders table of the model. If there is a long order found, then the function closes it with a counter-order. The same exact thing, although in the opposite direction, is done by the ShortClose() function. The work of these functions can be found in the listing provided above.

Let us analyze in more detail how the current lot is calculated in the trading model.

As mentioned above, for these purposes we use special functions for the capitalization of the account. In addition to the formulas of capitalizations, these functions include the verification of the calculation of the lot, based on the level of the used margin and by the restriction of the size of the trading positions.There may be some trading restrictions on the size of the position. They need to be taken into account.

Consider the following case: there is a limit on the size of a trading positions of 15 standard lots in both directions. The current position is long and equals to 3 lots. The trading model, based on its system of capital management, wants to open a long position with a volume of 18.6 lots. The function CheckLot() will return the corrected amount of the order volume. In this case, it will be equal to 12 lots (since 3 lots out of 15 are already occupied by other deals). If the current open position was short, and not long, then the function would have returned 15 lots instead of 18.6. This is the maximum possible volume of the positions.

After putting up 15 lots to buy, the net position, in this case, will be 12 lots (3 - sell, 15 - buy). When another model overrides its initial short position of 3 lots, buy the aggregated position will become the maximum possible - 15 lots. Other signals to buy will not be processed until themodel overrides some or all of its 15 lots buy. If the available volume for the requested transactionis exhausted, then the function will return a [EMPTY\_VALUE](https://www.mql5.com/en/docs/constants/namedconstants/typeconstants) constant. This signal must be passed.

If the check for the possibility of the set volume has been successful, then calculations are made on the value of the required margin. Account funds may be insufficient for the set volume. For these purposes, there is a function CheckMargin(). If the margin is not enough, it will try to correct the stated amount of the transaction so that the current free margin allowed it to open. If the margin is not enough even to open the minimum volume, then we are in a state of Margin-Call.

If there are currently no positions, and the margin is not used, it means only one thing - technical margin-call \- a state where it is impossible to open a deal. Without adding money to the account, we are unable to continue. If some margin is still used, then we have no choice but to wait  until the closure of the position that uses this margin. In any case, the lack of margin will return a constant EMPTY\_VALUE.

The functions for controlling the lot size and the margins are usually not used directly. They are called by special functions for managing capital. These functions implement the formulas for the capitalization of accounts. The file mm.mqh includes only two basic functions of capital management, one is calculated on the basis of a fixed fraction of the account, and the other, based on the method proposed by Ryan Jones, known as the method of fixed proportions.

The purpose of the first method is to define a fixed part of the account, which can be risked. For example, if the allowed risk is 2% of the account, and the account is equal to 10,000 dollars, then the maximum risk amount is $ 200. In order to calculate what lot should be used for a 200 dollar stop, you need to know exactly what the maximumdistance can be reached by the price against the opened position. Therefore, to calculate the lot through this formula, we need to accurately determine the Stop Loss and the price level, at which the deal will be made.

The method proposed by Ryan Jones is different from the previous one. Its essence is that capitalization is done by the function, defined by a particular case of a quadratic equation.

Here is its solution:

**_x=((1.0+MathSqrt(1+4.0\*d))/2)\*Step_**;

where: _x_ \- the lower boundary of the transition to the next level _d_ = (Profit / delta) \* 2.0 _Step_\- a step of the delta, such as 0.1 lots.

The lower the size of the delta, the more aggressively the function tries to increase the number of positions.For more details on how this function is constructed, you can refer to the book by Ryan Jones: [The Trading Game: Playing by the Numbers to Make Millions](https://www.mql5.com/go?link=https://www.amazon.com/Trading-Game-Playing-Numbers-Millions/dp/0471316989 "http://www.amazon.com/Trading-Game-Playing-Numbers-Millions/dp/0471316989").

If the functions of capital management are not planned to be used, it is necessary to directly call the functions of controlling the lot and margin.

So we have reviewed all of the elements of our basic EA. It is time to reap the fruits of our work.

To begin with, let's create four models. Let one model trade by EURUSD default parameters, the second one will also trade on EURUSD, but on a 15-minute time-frame. The third model will be launched on the graph GBPUSDwith default parameters. The fourth - on USDCHF on a two-hour graph, with the following parameters: SlowEMA= 6 FastEMA = 12 SignalEMA = 9\. The period for testing - H1, the testing mode - all of the ticks, the time from 01.01.2010 to 01.09.2010.

But before we run this model in four different modes, first we'll try to test it for each instrument and time-frame separately.

Here is the table on which the main indicators of testing is made:

| System | Number of Deals | Profit, $ |
| --- | --- | --- |
| MACD(9,12,26)H1 EURUSD | 123 | 1092 |
| MACD (9,12,26) EURUSD M15 | 598 | -696 |
| MACD(9,6,12) USDCHF H2 | 153 | -1150 |
| MACD(9,12,26) GBPUSD H1 | 139 | -282 |
| **All systems** | **1013** | **-1032** |

The table shows that the total number of deals for all models should be 1013, and the total profit should be $ -1032.

Consequently, these are the same values that we should obtain if we test these systems at the same time. The results should not differ, although some minor deviations still occur.

So here's the final test:

![](https://c.mql5.com/2/2/EURUSD-H1_M15_GBPUSD-H1_USDCHF_H29imgh.png)

As can be seen, there is only one less deals, and the profits differs only by $ 10, which corresponds to only 10 points of difference with a lot of 0.1. It should be noted that the results of combined testing, in the case of using the system of money management will be radically different from theamount of test results of each model individually. This is because the dynamics of the balance influence each of the systems, so the calculated values of the lots will vary.

Despite the fact that on its own, the results do not present an interest, we have created a complicated, but very flexible and manageable structure of the EA. Let us briefly again consider its structure.

For this we will turn to the scheme shown below:

![Class reference](https://c.mql5.com/2/2/Class_reference.png)

The scheme shows the basic structure of the model's instance.

Once a class instance of the trading model is created, we call the overloaded Init() function. It initializes the necessary parameters, prepares the data, loads the handles of [Indicators](https://www.mql5.com/en/docs/indicators), if they are being used. All this happens during the initialization of the EA, ie inside the [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) function. Note that the data includes instances of the base classes, designed to facilitate trade. It is assumed that the trading models need to actively use these classes instead of the standard functions of MQL5. After the model class is created successfully and [initialized](https://www.mql5.com/en/articles/28), it falls into the list of models [CList](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist). Further communication with it is carried out through a universal adapter [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject).

After the occurrence of the [OnTrade()](https://www.mql5.com/en/docs/basis/function/events#ontrade) or [OnTick()](https://www.mql5.com/en/docs/basis/function/events#ontrade) events, there is a sequential sorting of all instances of the models in the list. Communication with them is carried out by calling the Processing() function. Further, it calls the trading functions of its own model (the blue functions group). Their list and names are not strictly defined, but it is convenient to use the standard names such as LongOpened(), ShortClosed(), etc. These functions, based on their embedded logic, chose the time to complete the deal, and then send a specially formed request for the opening or closing of the deal of the SendOrder() function.

The latter makes the necessary checks, and then outputs the orders to the market. The trading functions rely on the auxiliary functions of the model (green group), which, in turn, actively use the basic auxiliary classes (purple group). All of the auxiliary classes are represented as instances of classes in the data section (pink group). The interaction between the groups is shown by dark-blue arrows.

### The trading model, based on the indicator of Bollinger Bands

Now that the general structure of data and methods becomes clear, we will create another trading model, based on the trend indicators of the Bollinger Bands. As the basis for this trading model, we used a simple EA [An Expert Advisor, based on Bollinger Bands](https://www.mql5.com/en/code/166) by Andrei Kornishkin. Bollinger Bands - are levels, equal to a certain size of standard deviations from the simple moving average. More details on how this indicator is constructed can be found in the Help section for technical analysis, attached to the MetaTrader 5 terminal.

The essence of the trading idea is simple: the price has the property of returning, ie if the price reached a certain level, then most likely, it will turn to the opposite direction. This thesis is proven by a test on a normal distribution of any real trading instrument: the normal distribution curve will be slightly elongated. The Bollinger bands determine the most probable culminations of price levels. Once they are reached (upper or lower Bollinger bands), the price is likely to turn in the opposite direction.

We slightly simplify the trading tactic, and will not use the auxiliary indicator - the double exponential moving average (Double Exponential Moving Average, or [DEMA](https://www.mql5.com/en/docs/indicators/idema)). But we will use strict protective stops - virtual Stop Loss. They will make the trading process more stable, and at the same time help us understand an example, in which each trading model uses its own independent level of protective stops.

For the level of protective stops, we use the current price plus or minus the indicator value volatility [ATR](https://www.mql5.com/en/docs/indicators/iatr). For example, if the current value of ATR is equal to 68 points and there is a signal to sell at a price of 1.25720, then the virtual Stop Loss for this deal will be equal to 1.25720 + 0.0068 = 1.26400. Similarly, but in the opposite direction, it is done for buying: 1.25720 - 0.0068 = 1.25040.

The source code of this model is provided below:

```
#include <Models\Model.mqh>
#include <mm.mqh>
//+----------------------------------------------------------------------+
//| This model use Bollinger bands.
//| Buy when price is lower than lower band
//| Sell when price is higher than upper band
//+----------------------------------------------------------------------+
struct cmodel_bollinger_param
{
   string            symbol;
   ENUM_TIMEFRAMES   timeframe;
   int               period_bollinger;
   double            deviation;
   int               shift_bands;
   int               period_ATR;
   double            k_ATR;
   double            delta;
};

class cmodel_bollinger : public CModel
{
private:
   int               m_bollinger_period;
   double            m_deviation;
   int               m_bands_shift;
   int               m_ATR_period;
   double            m_k_ATR;
   //------------Indicators Data:-------------
   int               m_bollinger_handle;
   int               m_ATR_handle;
   double            m_bollinger_buff_main[];
   double            m_ATR_buff_main[];
   //-----------------------------------------
   MqlRates          m_raters[];
   double            m_current_price;
public:
                     cmodel_bollinger();
   bool              Init();
   bool              Init(cmodel_bollinger_param &m_param);
   bool              Init(ulong magic, string name, string symbol, ENUM_TIMEFRAMES TimeFrame, double delta,
                          uint bollinger_period, double deviation, int bands_shift, uint ATR_period, double k_ATR);
   bool              Processing();
protected:
   bool              InitIndicators();
   bool              CheckParam(cmodel_bollinger_param &m_param);
   bool              LongOpened();
   bool              ShortOpened();
   bool              LongClosed();
   bool              ShortClosed();
   bool              CloseByStopSignal();
};

cmodel_bollinger::cmodel_bollinger()
{
   m_bollinger_handle   = INVALID_HANDLE;
   m_ATR_handle         = INVALID_HANDLE;
   ArraySetAsSeries(m_bollinger_buff_main,true);
   ArraySetAsSeries(m_ATR_buff_main,true);
   ArraySetAsSeries(m_raters, true);
   m_current_price=0.0;
}
//this default loader
bool cmodel_bollinger::Init()
{
   m_magic              = 322311;
   m_model_name         =  "Bollinger Bands Model";
   m_symbol             = _Symbol;
   m_timeframe          = _Period;
   m_bollinger_period   = 20;
   m_deviation          = 2.0;
   m_bands_shift        = 0;
   m_ATR_period         = 20;
   m_k_ATR              = 2.0;
   m_delta              = 0;
   if(!InitIndicators())return(false);
   return(true);
}

bool cmodel_bollinger::Init(cmodel_bollinger_param &m_param)
{
   m_magic              = 322311;
   m_model_name         = "Bollinger Model";
   m_symbol             = m_param.symbol;
   m_timeframe          = (ENUM_TIMEFRAMES)m_param.timeframe;
   m_bollinger_period   = m_param.period_bollinger;
   m_deviation          = m_param.deviation;
   m_bands_shift        = m_param.shift_bands;
   m_ATR_period        = m_param.period_ATR;
   m_k_ATR              = m_param.k_ATR;
   m_delta              = m_param.delta;
   //if(!CheckParam(m_param))return(false);
   if(!InitIndicators())return(false);
   return(true);
}

bool cmodel_bollinger::Init(ulong magic, string name, string symbol, ENUM_TIMEFRAMES timeframe, double delta,
                           uint bollinger_period, double deviation, int bands_shift, uint ATR_period, double k_ATR)
{
   m_magic           = magic;
   m_model_name      = name;
   m_symbol          = symbol;
   m_timeframe       = timeframe;
   m_delta           = delta;
   m_bollinger_period= bollinger_period;
   m_deviation       = deviation;
   m_bands_shift     = bands_shift;
   m_ATR_period      = ATR_period;
   m_k_ATR           = k_ATR;
   if(!InitIndicators())return(false);
   return(true);
}

/*bool cmodel_bollinger::CheckParam(cmodel_bollinger_param &m_param)
{
   if(!SymbolInfoInteger(m_symbol, SYMBOL_SELECT)){
      Print("Symbol ", m_symbol, " select failed. Check valid name symbol");
      return(false);
   }
   if(m_ma == 0){
      Print("Fast EMA must be bigest 0. Set MA = 12 (default)");
      m_ma=12;
   }
   return(true);
}*/

bool cmodel_bollinger::InitIndicators()
{
   m_bollinger_handle=iBands(m_symbol,m_timeframe,m_bollinger_period,m_bands_shift,m_deviation,PRICE_CLOSE);
   if(m_bollinger_handle==INVALID_HANDLE){
      Print("Error in creation of Bollinger indicator. Restart the Expert Advisor.");
      return(false);
   }
   m_ATR_handle=iATR(m_symbol,m_timeframe,m_ATR_period);
   if(m_ATR_handle==INVALID_HANDLE){
      Print("Error in creation of ATR indicator. Restart the Expert Advisor.");
      return(false);
   }
   return(true);
}

bool cmodel_bollinger::Processing()
{
   //if(timing(m_symbol,m_timeframe, m_timing)==false)return(false);

   //if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_DISABLED)return(false);
   //if(m_account_info.TradeAllowed()==false)return(false);
   //if(m_account_info.TradeExpert()==false)return(false);

   //m_symbol_info.Name(m_symbol);
   //m_symbol_info.RefreshRates();
   //Copy last data of moving average

   GetNumberOrders(m_orders);

   if(m_orders.buy_orders>0)   LongClosed();
   else                        LongOpened();
   if(m_orders.sell_orders!=0) ShortClosed();
   else                        ShortOpened();
   if(m_orders.all_orders!=0)CloseByStopSignal();
   return(true);
}

bool cmodel_bollinger::LongOpened(void)
{
   //if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_DISABLED)return(false);
   //if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_SHORTONLY)return(false);
   //if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_CLOSEONLY)return(false);
   //Print("Model Bollinger: ", m_orders.buy_orders);
   bool rezult, time_buy=true;
   double lot=0.1;
   double sl=0.0;
   double tp=0.0;
   mm open_mm;
   m_symbol_info.Name(m_symbol);
   m_symbol_info.RefreshRates();
   //lot=open_mm.optimal_f(m_symbol,OP_BUY,m_symbol_info.Ask(),sl,delta);
   CopyBuffer(m_bollinger_handle,2,0,3,m_bollinger_buff_main);
   CopyBuffer(m_ATR_handle,0,0,3,m_ATR_buff_main);
   CopyRates(m_symbol,m_timeframe,0,3,m_raters);
   if(m_raters[1].close>m_bollinger_buff_main[1]&&m_raters[1].open<m_bollinger_buff_main[1])
   {
      sl=NormalizeDouble(m_symbol_info.Ask()-m_ATR_buff_main[0]*m_k_ATR,_Digits);
      SendOrder(m_symbol,ORDER_TYPE_BUY,ORDER_ADD,0,lot,m_symbol_info.Ask(),sl,tp,"Add buy");
   }
   return(false);
}

bool cmodel_bollinger::ShortOpened(void)
{
   //if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_DISABLED)return(false);
   //if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_LONGONLY)return(false);
   //if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_CLOSEONLY)return(false);

   bool rezult, time_sell=true;
   double lot=0.1;
   double sl=0.0;
   double tp;
   mm open_mm;

   m_symbol_info.Name(m_symbol);
   m_symbol_info.RefreshRates();
   CopyBuffer(m_bollinger_handle,1,0,3,m_bollinger_buff_main);
   CopyBuffer(m_ATR_handle,0,0,3,m_ATR_buff_main);
   CopyRates(m_symbol,m_timeframe,0,3,m_raters);
   if(m_raters[1].close<m_bollinger_buff_main[1]&&m_raters[1].open>m_bollinger_buff_main[1])
   {
      sl=NormalizeDouble(m_symbol_info.Bid()+m_ATR_buff_main[0]*m_k_ATR,_Digits);
      SendOrder(m_symbol,ORDER_TYPE_SELL,ORDER_ADD,0,lot,m_symbol_info.Ask(),sl,tp,"Add buy");
   }
   return(false);
}

bool cmodel_bollinger::LongClosed(void)
{
   if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_DISABLED)return(false);
   CTableOrders *t;
   int total_elements;
   int rez=false;
   total_elements=ListTableOrders.Total();
   if(total_elements==0)return(false);
   m_symbol_info.Name(m_symbol);
   m_symbol_info.RefreshRates();
   CopyBuffer(m_bollinger_handle,1,0,3,m_bollinger_buff_main);
   CopyBuffer(m_ATR_handle,0,0,3,m_ATR_buff_main);
   CopyRates(m_symbol,m_timeframe,0,3,m_raters);
   if(m_raters[1].close<m_bollinger_buff_main[1]&&m_raters[1].open>m_bollinger_buff_main[1])
   {
      for(int i=total_elements-1;i>=0;i--)
      {
         if(CheckPointer(ListTableOrders)==POINTER_INVALID)continue;
         t=ListTableOrders.GetNodeAtIndex(i);
         if(CheckPointer(t)==POINTER_INVALID)continue;
         if(t.Type()!=ORDER_TYPE_BUY)continue;
         m_symbol_info.Refresh();
         m_symbol_info.RefreshRates();
         rez=SendOrder(m_symbol, ORDER_TYPE_SELL, ORDER_DELETE, t.Ticket(), t.VolumeInitial(),
                       m_symbol_info.Bid(), 0.0, 0.0, "BUY: close by signal");
      }
   }
   return(rez);
}

bool cmodel_bollinger::ShortClosed(void)
{
   if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_DISABLED)return(false);
   CTableOrders *t;
   int total_elements;
   int rez=false;
   total_elements=ListTableOrders.Total();
   if(total_elements==0)return(false);
   CopyBuffer(m_bollinger_handle,2,0,3,m_bollinger_buff_main);
   CopyBuffer(m_ATR_handle,0,0,3,m_ATR_buff_main);
   CopyRates(m_symbol,m_timeframe,0,3,m_raters);
   if(m_raters[1].close>m_bollinger_buff_main[1]&&m_raters[1].open<m_bollinger_buff_main[1])
   {
      for(int i=total_elements-1;i>=0;i--)
      {
         if(CheckPointer(ListTableOrders)==POINTER_INVALID)continue;
         t=ListTableOrders.GetNodeAtIndex(i);
         if(CheckPointer(t)==POINTER_INVALID)continue;
         if(t.Type()!=ORDER_TYPE_SELL)continue;
         m_symbol_info.Refresh();
         m_symbol_info.RefreshRates();
         rez=SendOrder(m_symbol, ORDER_TYPE_BUY, ORDER_DELETE, t.Ticket(), t.VolumeInitial(),
                       m_symbol_info.Ask(), 0.0, 0.0, "SELL: close by signal");
      }
   }
   return(rez);
}

bool cmodel_bollinger::CloseByStopSignal(void)
{
   if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_DISABLED)return(false);
   CTableOrders *t;
   int total_elements;
   bool rez=false;
   total_elements=ListTableOrders.Total();
   if(total_elements==0)return(false);
   for(int i=total_elements-1;i>=0;i--)
   {
      if(CheckPointer(ListTableOrders)==POINTER_INVALID)continue;
      t=ListTableOrders.GetNodeAtIndex(i);
      if(CheckPointer(t)==POINTER_INVALID)continue;
      if(t.Type()!=ORDER_TYPE_SELL&&t.Type()!=ORDER_TYPE_BUY)continue;
      m_symbol_info.Refresh();
      m_symbol_info.RefreshRates();
      CopyRates(m_symbol,m_timeframe,0,3,m_raters);
      if(m_symbol_info.Bid()<=t.StopLoss()&&t.Type()==ORDER_TYPE_BUY)
      {
         rez=SendOrder(m_symbol, ORDER_TYPE_SELL, ORDER_DELETE, t.Ticket(), t.VolumeInitial(),
                       m_symbol_info.Bid(), 0.0, 0.0, "BUY: close by stop");
         continue;
      }
      if(m_symbol_info.Ask()>=t.StopLoss()&&t.Type()==ORDER_TYPE_SELL)
      {
         rez=SendOrder(m_symbol, ORDER_TYPE_BUY, ORDER_DELETE, t.Ticket(), t.VolumeInitial(),
                       m_symbol_info.Ask(), 0.0, 0.0, "SELL: close by stop");
         continue;
      }
   }
   return(rez);
}
```

As can be seen, the code of the trading model is very similar to the source code of the previous trading tactics. The main variation here is the emergence of virtual stop orders and the function cmodel\_bollinger:: CloseByStopSignal(), serving these protective stops.

In fact, in case of the use of protective stops, their values must simply be pass to the function SendOrder(). And this function will input these levels into the table of orders. When the price crosses or touches these levels, the function CloseByStopSignal() will close the deal with a counter order, and remove it from the list of active orders.

### The combination of trading models, instruments, and time-frames into a single entity

Now that we have two trading models, it is time to test them simultaneously. Before we make a layout of the models, it would be useful to determine their most effective parameters. To do this, we have to make the optimization of each model individually.

Optimization will demonstrate on which market and time-frame the model is most effective. For each model, we will be selected only two of the best time-frames and three best instruments. As a result, we get twelve independent solutions of (2 models \* 3 instruments \* 2 time-frames), which will be tested all together. Of course, the selected optimization method suffers from the so-called "adjustment" of the results, but for our purposes this is not important.

The graphs below demonstrate the best results of the sample:

1.1 MACD
EURUSD M30

![MACD EURUSD M30](https://c.mql5.com/2/2/1d1_EURUSD_m30_macd.png)

1.2 . MACD
EURUSD H3

![MACD EURUSD H3](https://c.mql5.com/2/2/1f2_EURUSD_h3_macd.png)

1.3 MACDAUDUSDH4

![MACD AUDUSD H4](https://c.mql5.com/2/2/103_AUDUSD_h4_macd.png)

1.4 . MACDAUDUSDH1

![MACD AUDUSD H1](https://c.mql5.com/2/2/1j4_AUDUSD_h1_macd.png)

1.5 MACDGBPUSDH12

![MACD GBPUSD H12](https://c.mql5.com/2/2/185_GBPUSD_h12_macd.png)

1.6 MACDGBPUSDH6

![MACD GBPUSD H6](https://c.mql5.com/2/2/1t6_GBPUSD_h6_macd.png)

2.1 BollingerGBPUSDM15

![Bollinger GBPUSD M15](https://c.mql5.com/2/2/2n1_GBPUSD_m15_boll.png)

2.2 Bollinger GBPUSD H1

![Bollinger GBPUSD H1](https://c.mql5.com/2/2/2d2_GBPUSD_h1_boll.png)

2.3 Bollinger EURUSD M30

![Bollinger EURUSD M30](https://c.mql5.com/2/2/203_EURUSD_m30_boll.png)

2.4  Bollinger EURUSD H4

![Bollinger EURUSD H4](https://c.mql5.com/2/2/2o4_EURUSD_h4_boll.png)

2.5 Bollinger USDCAD M15

![Bollinger USDCAD M15](https://c.mql5.com/2/2/2i5_USDCAD_m15_boll.png)

2.6 Bollinger USDCAD H2

![Bollinger USDCAD H2](https://c.mql5.com/2/2/2y6_USDCAD_h2_boll.png)

Now that the optimal results are known, we are left with only a bit more to do - to gather the results into a single entity.

Below is the source code of the function-loader, which creates the twelve trading models, shown above, after which the EA consistently begins to trade on them:

```
bool macd_default=true;
bool macd_best=false;
bool bollinger_default=false;
bool bollinger_best=false;

void InitModels()
{
   list_model = new CList;             // Initialized pointer of the model list
   cmodel_macd *model_macd;            // Create the pointer to a model MACD
   cmodel_bollinger *model_bollinger;  // Create the pointer to a model Bollinger

//----------------------------------------MACD DEFAULT----------------------------------------
   if(macd_default==true&&macd_best==false)
    {
      model_macd = new cmodel_macd; // Initialize the pointer by the model MACD
      // Loading of the parameters was completed successfully
      if(model_macd.Init(129475, "Model macd M15", _Symbol, _Period, 0.0, Fast_MA,Slow_MA,Signal_MA))
      {

         Print("Print(Model ", model_macd.Name(), " with period = ", model_macd.Period(),
              " on symbol ", model_macd.Symbol(), " successfully created");
         list_model.Add(model_macd);// Загружаем модель в список моделей
      }
      else
      {
                                 // The loading of parameters was completed successfully
         Print("Print(Model ", model_macd.Name(), " with period = ", model_macd.Period(),
         " on symbol ", model_macd.Symbol(), " creation has failed");
      }
   }
//-------------------------------------------------------------------------------------------
//----------------------------------------MACD BEST------------------------------------------
   if(macd_best==true&&macd_default==false)
   {
      // 1.1 EURUSD H30; FMA=20; SMA=24;
      model_macd = new cmodel_macd; // Initialize the pointer to the model MACD
      if(model_macd.Init(129475, "Model macd H30", "EURUSD", PERIOD_M30, 0.0, 20,24,9))
      {
         Print("Print(Model ", model_macd.Name(), " with period = ", model_macd.Period(),
               " on symbol ", model_macd.Symbol(), " created successfully");
         list_model.Add(model_macd);// load the model into the list of models
      }
      else
      {// Loading parameters was completed unsuccessfully
         Print("Print(Model ", model_macd.Name(), " with period = ", model_macd.Period(),
         " on symbol ", model_macd.Symbol(), " creation has failed");
      }
      // 1.2 EURUSD H3; FMA=8; SMA=12;
      model_macd = new cmodel_macd; // Initialize the pointer by the model MACD
      if(model_macd.Init(129475, "Model macd H3", "EURUSD", PERIOD_H3, 0.0, 8,12,9))
      {
         Print("Print(Model ", model_macd.Name(), " with period = ", model_macd.Period(),
              " on symbol ", model_macd.Symbol(), " successfully created");
         list_model.Add(model_macd);// Load the model into the list of models
      }
      else
       {// Loading of parameters was unsuccessful
         Print("Print(Model ", model_macd.Name(), " with period = ", model_macd.Period(),
         " on symbol ", model_macd.Symbol(), " creation has failed");
      }
      // 1.3 AUDUSD H1; FMA=10; SMA=18;
      model_macd = new cmodel_macd; // Initialize the pointer by the model MACD
      if(model_macd.Init(129475, "Model macd M15", "AUDUSD", PERIOD_H1, 0.0, 10,18,9))
      {
         Print("Print(Model ", model_macd.Name(), " with period = ", model_macd.Period(),
              " on symbol ", model_macd.Symbol(), " successfully created");
         list_model.Add(model_macd);// Load the model into the list of models
      }
      else
      {// The loading of parameters was unsuccessful
         Print("Print(Model ", model_macd.Name(), " with period = ", model_macd.Period(),
               " on symbol ", model_macd.Symbol(), " creation has failed");
      }
      // 1.4 AUDUSD H4; FMA=14; SMA=15;
      model_macd = new cmodel_macd; // Initialize the pointer by the model MACD
      if(model_macd.Init(129475, "Model macd H4", "AUDUSD", PERIOD_H4, 0.0, 14,15,9))
      {
         Print("Print(Model ", model_macd.Name(), " with period = ", model_macd.Period(),
         " on symbol ", model_macd.Symbol(), " successfully created");
         list_model.Add(model_macd);// Load the model into the list of models
      }
      else{// Loading of parameters was unsuccessful
         Print("Print(Model ", model_macd.Name(), " with period = ", model_macd.Period(),
              " on symbol ", model_macd.Symbol(), " creation has failed");
      }
      // 1.5 GBPUSD H6; FMA=20; SMA=33;
      model_macd = new cmodel_macd; // Initialize the pointer by the model MACD
      if(model_macd.Init(129475, "Model macd H6", "GBPUSD", PERIOD_H6, 0.0, 20,33,9))
      {
         Print("Print(Model ", model_macd.Name(), " with period = ", model_macd.Period(),
              " on symbol ", model_macd.Symbol(), " successfully created");
         list_model.Add(model_macd);// Load the model into the list of models
      }
      else
      {// Loading of parameters was  unsuccessful
         Print("Print(Model ", model_macd.Name(), " with period = ", model_macd.Period(),
               " on symbol ", model_macd.Symbol(), " creation has failed");
      }
      // 1.6 GBPUSD H12; FMA=12; SMA=30;
      model_macd = new cmodel_macd; // Initialize the pointer by the model MACD
      if(model_macd.Init(129475, "Model macd H6", "GBPUSD", PERIOD_H12, 0.0, 12,30,9))
      {
         Print("Print(Model ", model_macd.Name(), " with period = ", model_macd.Period(),
              " on symbol ", model_macd.Symbol(), " successfully created");
         list_model.Add(model_macd);// Load the model into the list of models
      }
      else
      {// Loading of parameters was unsuccessful
         Print("Print(Model ", model_macd.Name(), " with period = ", model_macd.Period(),
              " on symbol ", model_macd.Symbol(), " creation has failed");
      }
   }
//----------------------------------------------------------------------------------------------
//-------------------------------------BOLLINGER DEFAULT----------------------------------------
   if(bollinger_default==true&&bollinger_best==false)
   {
      model_bollinger = new cmodel_bollinger;
      if(model_bollinger.Init(1829374,"Bollinger",_Symbol,PERIOD_CURRENT,0,
                             period_bollinger,dev_bollinger,0,14,k_ATR))
      {
         Print("Model ", model_bollinger.Name(), " successfully created");
         list_model.Add(model_bollinger);
      }
   }
//----------------------------------------------------------------------------------------------
//--------------------------------------BOLLLINGER BEST-----------------------------------------
   if(bollinger_best==true&&bollinger_default==false)
   {
      //2.1 Symbol: EURUSD M30; period: 15; deviation: 2,75; k_ATR=2,75;
      model_bollinger = new cmodel_bollinger;
      if(model_bollinger.Init(1829374,"Bollinger","EURUSD",PERIOD_M30,0,15,2.75,0,14,2.75))
      {
         Print("Model ", model_bollinger.Name(), "Period: ", model_bollinger.Period(),
              ". Symbol: ", model_bollinger.Symbol(), " successfully created");
         list_model.Add(model_bollinger);
      }
      //2.2 Symbol: EURUSD H4; period: 30; deviation: 2.0; k_ATR=2.25;
      model_bollinger = new cmodel_bollinger;
      if(model_bollinger.Init(1829374,"Bollinger","EURUSD",PERIOD_H4,0,30,2.00,0,14,2.25))
      {
         Print("Model ", model_bollinger.Name(), "Period: ", model_bollinger.Period(),
         ". Symbol: ", model_bollinger.Symbol(), " successfully created");
         list_model.Add(model_bollinger);
      }
      //2.3 Symbol: GBPUSD M15; period: 18; deviation: 2.25; k_ATR=3.0;
      model_bollinger = new cmodel_bollinger;
      if(model_bollinger.Init(1829374,"Bollinger","GBPUSD",PERIOD_M15,0,18,2.25,0,14,3.00))
      {
         Print("Model ", model_bollinger.Name(), "Period: ", model_bollinger.Period(),
         ". Symbol: ", model_bollinger.Symbol(), " successfully created");
         list_model.Add(model_bollinger);
      }
      //2.4 Symbol: GBPUSD H1; period: 27; deviation: 2.25; k_ATR=3.75;
      model_bollinger = new cmodel_bollinger;
      if(model_bollinger.Init(1829374,"Bollinger","GBPUSD",PERIOD_H1,0,27,2.25,0,14,3.75))
      {
         Print("Model ", model_bollinger.Name(), "Period: ", model_bollinger.Period(),
         ". Symbol: ", model_bollinger.Symbol(), " successfully created");
         list_model.Add(model_bollinger);
      }
      //2.5 Symbol: USDCAD M15; period: 18; deviation: 2.5; k_ATR=2.00;
      model_bollinger = new cmodel_bollinger;
      if(model_bollinger.Init(1829374,"Bollinger","USDCAD",PERIOD_M15,0,18,2.50,0,14,2.00))
      {
         Print("Model ", model_bollinger.Name(), "Period: ", model_bollinger.Period(),
         ". Symbol: ", model_bollinger.Symbol(), " successfully created");
         list_model.Add(model_bollinger);
      }
      //2.6 Symbol: USDCAD M15; period: 21; deviation: 2.5; k_ATR=3.25;
      model_bollinger = new cmodel_bollinger;
      if(model_bollinger.Init(1829374,"Bollinger","USDCAD",PERIOD_H2,0,21,2.50,0,14,3.25))
      {
         Print("Model ", model_bollinger.Name(), "Period: ", model_bollinger.Period(),
         ". Symbol: ", model_bollinger.Symbol(), " successfully created");
         list_model.Add(model_bollinger);
      }
   }
//----------------------------------------------------------------------------------------------
}
```

Now let's test all twelve models simultaneously:

![](https://c.mql5.com/2/2/picture1__2.png)

### Capitalization of the results

The resulting graph is impressive. However, what is important is not the result, but the fact that all models are trading simultaneously, all using their individual protective stops, and all independent of each other. Now let's try to capitalize the resulting graph. For this we will use the standard functions of capitalization: fixed-proportional method and the method proposed by Ryan Jones.

The so-called _optimal f_ \- a special case of the fixed-proportional method. The essence of this method is that each deal is given a limit of losses, equal to a percentage of the account. Conservative capitalization strategies usually apply a 2% loss limit. I.e. at $10,000, the position size is calculated so that the loss, after the Stop Loss has been passed, does not exceed $ 200. There is, however, a certain function of the growth of the final balance from the increase in the risk. This function has a bell-like form. I.e. at first, with the increase of the risk, there is an increase of the total profit. However, there is a threshold of risk for each deal, after which the total profit balance begins to decrease. This threshold is the so-called optimal f.

This article is not devoted to the issue of capital management, all that we need to know to use the fixed-proportional method - is the level of protective stops, and the percentage of the account that we can risk. The Ryan Jones formula is constructed differently. For its work, using fixed protective stops is not required. Since the first of the proposed models (MACD model), is rather primitive and has no protective stops, we will use this method for the capitalization of this model. For the model, based on Bollinger bands, we will use the fixed-proportional method of proportions.

In order to begin using the formula of capitalization, we need to fill the variable m\_delta, included in the basic model.

When using the formula of fixed proportions, it must be equal to the percentage of risk for each deal. When using the method of Ryan Jones, it is equal to the so-called delta increment, i.e. to the amount of money that need to be earned to get to a higher level of position volume.

Below is the graph of capitalization:

![](https://c.mql5.com/2/2/picture2__2.png)

As can be seen, all models have their own formulas of capitalization (the fixed-proportional method or the method of Ryan Jones).

In the provided example, we used the same values of maximum risk and delta for all models. However, for each of them, we can select individual parameters for capitalization. The fine-tuning of each of the models is not included in the range of issues covered by this article.

### Working with pending orders

The presented trading models do not use the so-called pending orders. These are orders that are executed only when certain terms and conditions are present.

In reality, any trading strategy, using pending orders may be adjusted to the use of orders in the market. Rather, the pending orders are needed for a more reliable system operation, for when the break-down of the robot, or the equipment on which it works, the pending orders would still execute protective stops, or vice versa, would enter the market based on the previously determined prices.

The proposed trading model allows you to work with pending trade orders, although the process of control of their presentation, in this case, is much more complicated. To work with these orders, we use the overloaded method Add (COrderInfo & order\_info, double stop\_loss, double take\_profit) of class CTableOrders. In this case, the variable m\_type of this class will contain the appropriate type of the pending order, e.g. [ORDER\_TYPE\_BUY\_STOP](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type) or [ORDER\_TYPE\_SELL\_LIMIT](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type).

Later, when the pending order will be put out, you need to "catch" the moment when it was triggered to work, or its term of relevance will expire. This is not that simple, since it is not enough to simply control the event Trade, but it is also required to know what triggered it.

The language of MQL5 is evolving, and currently the issue of the inclusion of a special service structure into it, which would explain the event [Trade](https://www.mql5.com/en/docs/runtime/event_fire#trade), is being solved. But for now, we have to view the list of orders in the history. If an order, with the same ticket as the pending order in the table of orders, is found in the history, then, there has occurred some events which needs to be reflected in the table.

The code of the basic model includes a special method CModel:: ReplaceDelayedOrders. This method works on the following algorithm. First, all of the active orders in the table of orders are checked. The ticket of of these orders is compared with the ticket of the orders in the history ( [HistoryOrderGetTicket()](https://www.mql5.com/en/docs/trading/historyordergetticket)). If the ticket of the order in the history coincides with the pending order in the table of orders, but [the order status](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties) is executed ([ORDER\_STATE\_PARTIAL](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_state) or [ORDER\_STATE\_FILLED](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_state)), then the status of the pending orders in the table of orders is also changed for executed.

Further, if this order is not linked with any pending orders, imitating the work of Stop Loss and Take Profit, such order are put out, and their tickets are entered into the appropriate table values (TicketSL, TicketTP). And the pending orders, imitating the protective stop and profit levels, are put out at the price, specified in advance with the help of variables m\_sl and m\_tp. I.e. these prices should be known at the moment of calling the method ReplaceDelayedOrders.

It is noteworthy that the method is adapted to handle the situation when an order is put out to the market, on which the required pending orders are of the type Stop Loss and Take Profit.

In general, the work of the proposed method is not trivial, and requires some understanding for its use:

```
bool CModel::ReplaceDelayedOrders(void)
{
   if(m_symbol_info.TradeMode()==SYMBOL_TRADE_MODE_DISABLED)return(false);
   CTableOrders *t;
   int total_elements;
   int history_orders=HistoryOrdersTotal();
   ulong ticket;
   bool rez=false;
   long request;
   total_elements=ListTableOrders.Total();
   int try=0;
   if(total_elements==0)return(false);
   // View every order in the table
   for(int i=total_elements-1;i>=0;i--)
   {
      if(CheckPointer(ListTableOrders)==POINTER_INVALID)continue;
      t=ListTableOrders.GetNodeAtIndex(i);
      if(CheckPointer(t)==POINTER_INVALID)continue;
      switch(t.Type())
      {
         case ORDER_TYPE_BUY:
         case ORDER_TYPE_SELL:
            for(int b=0;i<history_orders;b++)
            {
               ticket=HistoryOrderGetTicket(b);
               // If the ticket of the historical order is equal to one of the tickets
               // Stop Loss or Take Profit, then the order was blocked, and it needs to be
               // deleted from the table of orders
               if(ticket==t.TicketSL()||ticket==t.TicketTP())
               {
                  ListTableOrders.DeleteCurrent();
               }
            }
            // If the orders, imitating the Stop Loss and Take Profit are not found in the history,
            // then perhaps they are not yet set. Therefore, they need to be inputted,
            // using the process for pending orders below:
            // the cycle  keeps going, the exit 'break' does not exist!!!
         case ORDER_TYPE_BUY_LIMIT:
         case ORDER_TYPE_BUY_STOP:
         case ORDER_TYPE_BUY_STOP_LIMIT:
         case ORDER_TYPE_SELL_LIMIT:
         case ORDER_TYPE_SELL_STOP:
         case ORDER_TYPE_SELL_STOP_LIMIT:
            for(int b=0;i<history_orders;b++)
            {
               ticket=HistoryOrderGetTicket(b);
               // If the ticket of the historical order is equal to the ticket of the pending order
               // then the pending order has worked and needs to be put out
               // the pending orders, imitating the work of Stop Loss and Take Profit.
               // It is also necessary to change the pending status of the order in the table
               // of orders for the executed (ORDER_TYPE_BUY или ORDER_TYPE_SELL)
               m_order_info.InfoInteger(ORDER_STATE,request);
               if(t.Ticket()==ticket&&
                  (request==ORDER_STATE_PARTIAL||request==ORDER_STATE_FILLED))
                  {
                  // Change the status order in the table of orders:
                  m_order_info.InfoInteger(ORDER_TYPE,request);
                  if(t.Type()!=request)t.Type(request);
                  //------------------------------------------------------------------
                  // Put out the pending orders, imitating Stop Loss an Take Profit:
                  // The level of pending orders, imitating Stop Loss and Take Profit
                  // should be determined earlier. It is also necessary to make sure that
                  // the current order is not already linked with the pending order, imitating Stop Loss
                  // and Take Profit:
                  if(t.StopLoss()!=0.0&&t.TicketSL()==0)
                    {
                     // Try to put out the pending order:
                     switch(t.Type())
                     {
                        case ORDER_TYPE_BUY:
                           // Make three attempts to put out the pending order
                           for(try=0;try<3;try++)
                           {
                              m_trade.SellStop(t.VolumeInitial(),t.StopLoss(),m_symbol,0.0,0.0,0,0,"take-profit for buy");
                              if(m_trade.ResultRetcode()==TRADE_RETCODE_PLACED||m_trade.ResultRetcode()==TRADE_RETCODE_DONE)
                              {
                                 t.TicketTP(m_trade.ResultDeal());
                                 break;
                              }
                           }
                        case ORDER_TYPE_SELL:
                           // Make three attempts to put up a pending order
                           for(try=0;try<3;try++)
                           {
                              m_trade.BuyStop(t.VolumeInitial(),t.StopLoss(),m_symbol,0.0,0.0,0,0,"take-profit for buy");
                              if(m_trade.ResultRetcode()==TRADE_RETCODE_PLACED||m_trade.ResultRetcode()==TRADE_RETCODE_DONE)
                              {
                                 t.TicketTP(m_trade.ResultDeal());
                                 break;
                              }
                           }
                     }
                  }
                  if(t.TakeProfit()!=0.0&&t.TicketTP()==0){
                     // Attempt to put out the pending order, imitating Take Profit:
                     switch(t.Type())
                     {
                        case ORDER_TYPE_BUY:
                           // Make three attempts to put out the pending order
                           for(try=0;try<3;try++)
                           {
                              m_trade.SellLimit(t.VolumeInitial(),t.StopLoss(),m_symbol,0.0,0.0,0,0,"take-profit for buy");
                              if(m_trade.ResultRetcode()==TRADE_RETCODE_PLACED||m_trade.ResultRetcode()==TRADE_RETCODE_DONE)
                              {
                                 t.TicketTP(m_trade.ResultDeal());
                                 break;
                              }
                           }
                           break;
                        case ORDER_TYPE_SELL:
                           // Make three attempts to put out the pending order
                           for(try=0;try<3;try++)
                           {
                              m_trade.BuyLimit(t.VolumeInitial(),t.StopLoss(),m_symbol,0.0,0.0,0,0,"take-profit for buy");
                              if(m_trade.ResultRetcode()==TRADE_RETCODE_PLACED||m_trade.ResultRetcode()==TRADE_RETCODE_DONE)
                              {
                                 t.TicketTP(m_trade.ResultDeal());
                                 break;
                              }
                           }
                     }
                  }
               }
            }
            break;

      }
   }
   return(true);
}
```

Using this method, you can effortlessly create a model based on pending orders.

### Conclusion

Unfortunately, it is impossible to consider in a single article all of the nuances, challenges, and benefits of the proposed approach. We have not considered the serialization of data - a method, allowing you to store, record, and retrieve all of the necessary information on the current status of models from the data files. Beyond our consideration were remained the trading models, based on the trade of synthetic spreads. These are very interesting topics, and,certainly, they have their own effective solutions for the proposed concepts.

The main thing that we managed to do - is to develop a fully dynamic and manageable data structure. The concept of linked lists effectively manages them, makes trading tactics independent and individually customizable. Another very important advantage of this approach is that it is completely universal.

For example, on its basis, it is enough to create two trading EAs and place them on the same instrument. Both of them will automatically work only with their own orders, and only with their own system of capital management. Thus, there is a support of a downward compatibility. All that can be done simultaneously in one EA, can be distributed amongst several robots. This property is extremely important when working in net positions.

The model described is not just a theory. It includes an advanced apparatus of auxiliary functions, the functions for managing capital and checking marginal requirements. The system of sending orders is resistant to the so-called requotes and slippages - the effects of which are often seen in real trading.

The trading engine determines the maximum size of the position and the maximum volume of deals. A special algorithm separates trading requests into several independent deals, each of which is processed separately. Moreover, [the presented model](https://championship.mql5.com/2010/en/users/C-4 "https://championship.mql5.com/2010/en/users/C-4") has proven itself well in [Automated Trading Championship of 2010](https://championship.mql5.com/2010/ru "https://championship.mql5.com/2010/ru") \- all deals are carried out accurately and in accordance with the trading plan, the trading models, presented on the Championship, are managed with a varying degrees of risks, and on different systems of money management.

The presented approach is almost a complete solution for the participation in the Championships, as well as for the parallel operation on several instruments and time-frames, on several trading tactics. The only difficulty in getting acquainted with this approach lies in its complexity.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/217](https://www.mql5.com/ru/articles/217)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/217.zip "Download all attachments in the single ZIP archive")

[files-en.zip](https://www.mql5.com/en/articles/download/217/files-en.zip "Download files-en.zip")(18.6 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/2982)**
(64)


![kzh125](https://c.mql5.com/avatar/avatar_na2.png)

**[kzh125](https://www.mql5.com/en/users/kzh125)**
\|
25 Jan 2021 at 09:21

You shouldn't delete node when iterating the list.

for example:

```
class Test : public CObject {
  public:
    int i_;
    Test(int i) {
        i_ = i;
    };
    int get_i() {
        return i_;
    };
};
```

```
    CList *list = new CList();
    for (int i = 0; i < 10; i++) {
        Test *t = new Test(i);
        list.Add(t);
    }

    for (int i = 0; i < list.Total(); i++) {
        Test *t = list.GetNodeAtIndex(i);
        if (i == 5) {
            list.DeleteCurrent();
        }
        if (CheckPointer(t) == POINTER_INVALID) {
            continue;
        }
        Print(t.get_i());
    }
```

After deleting node at index 5, you iterate index 6, but the next element is still index 5.

It's a better idea using GetFirstNode / GetNextNode

```
    for (Test *t = list.GetFirstNode(); t != NULL;) {
        t_current = t;
        if (t.get_i() == 5) {
            list.DeleteCurrent();
            t = list.GetCurrentNode();
            if (t == t_current) {break;}
            continue;
        }
        Print(t.get_i());
        t = list.GetNextNode();
    }
```

Thanks for your contribution!

![kzh125](https://c.mql5.com/avatar/avatar_na2.png)

**[kzh125](https://www.mql5.com/en/users/kzh125)**
\|
26 Jan 2021 at 05:33

In the CTableOrders class, I think you should use order\_info.OrderType() instead of order\_info.Type()

![zhurs](https://c.mql5.com/avatar/2023/10/65416d86-61d8.png)

**[zhurs](https://www.mql5.com/en/users/zhurs)**
\|
3 Jul 2022 at 14:35

Thank you!!! Very interesting article. I will try to take it as a basis for developing my own [multicurrency Expert Advisor](https://www.mql5.com/en/articles/648 "Article: MQL5 Recipes - Multicurrency Expert Advisor: an example of a simple, accurate and fast scheme "). I tried to write it myself - it works, but I see a problem in the architecture of the programme and I can't understand how to fix it myself. As a beginner and inexperienced, such things are hard for me.


![Ruben Osvaldo Rodriguez](https://c.mql5.com/avatar/2022/7/62CF1C83-4097.jpg)

**[Ruben Osvaldo Rodriguez](https://www.mql5.com/en/users/rubenosvaldorodriguez)**
\|
25 Jul 2022 at 06:59

Dear EA author and articles:

I want to congratulate you as I found your design excellent. :)

I have a problem loading the code and trying to test it because the line:

result=list\_models.Add(m\_macd); //FROM FILE model\_simple.mq5

gives an [error message](https://www.mql5.com/en/docs/constants/errorswarnings/errorscompile "MQL5 Documentation: Compilation errors ") (

```
conversion is not accessible because of inheritance access
```

)

when trying to compile.

And I couldn't solve this error despite browsing the forums and trying a couple of alternatives.

Could you help me if you are so kind?

Thank you very much.

![isanchez96](https://c.mql5.com/avatar/avatar_na2.png)

**[isanchez96](https://www.mql5.com/en/users/isanchez96)**
\|
8 Dec 2024 at 11:17

Hi, could you update this post a bit, I find it interesting but there are errors, I have been correcting some but others are more complicated. For example, in [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "MetaTrader 5 Help: MACD Indicator") you open many sell positions and don't close the others. Still the concept is very good, although it would be nice if you could update it and use it in live trades. Thank you. Best regards.

![Moving Mini-Max: a New Indicator for Technical Analysis and Its Implementation in MQL5](https://c.mql5.com/2/0/MQL5_Mini-Max_Indicator.png)[Moving Mini-Max: a New Indicator for Technical Analysis and Its Implementation in MQL5](https://www.mql5.com/en/articles/238)

In the following article I am describing a process of implementing Moving Mini-Max indicator based on a paper by Z.G.Silagadze 'Moving Mini-max: a new indicator for technical analysis'. The idea of the indicator is based on simulation of quantum tunneling phenomena, proposed by G. Gamov in the theory of alpha decay.

![Create Your Own Expert Advisor in MQL5 Wizard](https://c.mql5.com/2/0/masterMQL5__2.png)[Create Your Own Expert Advisor in MQL5 Wizard](https://www.mql5.com/en/articles/240)

The knowledge of programming languages is no longer a prerequisite for creating trading robots. Earlier lack of programming skills was an impassable obstacle to the implementation of one's own trading strategies, but with the emergence of the MQL5 Wizard, the situation radically changed. Novice traders can stop worrying because of the lack of programming experience - with the new Wizard, which allows you to generate Expert Advisor code, it is not necessary.

![How to Copy Trading from MetaTrader 5 to MetaTrader 4](https://c.mql5.com/2/0/MetaTrader5_MetaTrader4_MQL5.png)[How to Copy Trading from MetaTrader 5 to MetaTrader 4](https://www.mql5.com/en/articles/189)

Is it possible to trade on a real MetaTrader 5 account today? How to organize such trading? The article contains the theory of these questions and the working codes used for copying trades from the MetaTrader 5 terminal to MetaTrader 4. The article will be useful both for the developers of Expert Advisors and for practicing traders.

![MQL5 Wizard: How to Create a Module of Trading Signals](https://c.mql5.com/2/0/MQL5_CExpertSignal.png)[MQL5 Wizard: How to Create a Module of Trading Signals](https://www.mql5.com/en/articles/226)

The article discusses how to write your own class of trading signals with the implementation of signals on the crossing of the price and the moving average, and how to include it to the generator of trading strategies of the MQL5 Wizard, as well as describes the structure and format of the description of the generated class for the MQL5 Wizard.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=prgrtoxeqrcooxkjeeimxgomrdkuhkzx&ssn=1769178428953798164&ssn_dr=0&ssn_sr=0&fv_date=1769178428&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F217&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20Multi-Expert%20Advisors%20on%20the%20basis%20of%20Trading%20Models%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917842847998665&fz_uniq=5068249439002359612&sv=2552)

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