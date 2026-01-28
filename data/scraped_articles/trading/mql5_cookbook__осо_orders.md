---
title: MQL5 Cookbook: ОСО Orders
url: https://www.mql5.com/en/articles/1582
categories: Trading
relevance_score: 3
scraped_at: 2026-01-23T18:19:18.431779
---

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/1582&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069341516041749364)

MetaTrader 5 / Examples


### Introduction

This article focuses on dealing with such type of order pair as OCO. This mechanism is implemented in some trading terminals competing with MetaTrader 5. I pursue two aims through the example of creation of an EA with a panel for OCO orders processing. On the one hand, I wish to describe features of the [Standard Library](https://www.mql5.com/en/docs/standardlibrary), on the other hand I would like to extend a trader's tool set.

### 1\. Essence of OCO Orders

OCO orders (one-cancels-the-other order) represent a pair of two pending orders.

They are connected by mutual cancellation function: if the first one triggers, the second one should be removed, and vice versa.

![Fig. 1 Pair of OCO orders](https://c.mql5.com/2/17/Fig1_OCO_Orders.png)

Fig. 1 Pair of OCO orders

Fig.1 shows a simple order interdependence scheme. It reflects an essential definition: a pair exists for so long as both orders exist. In terms of logic any \[one\] order of the pair is an essential but not sufficient condition for the pair existence.

Some sources say that the pair must have one limit order and one stop order, moreover orders must have one direction (either buy or sell). To my mind such restriction cannot aid in creation of flexible trading strategies. I suggest that various OCO orders shall be analyzed in the pair, and most importantly we shall try to program this pair.

### 2\. Programming Order Pair

In my thinking, ООP toolset is suited for programming tasks connected with control over OCO orders in the best possible way.

Following sections are devoted to new data types which will serve our purpose. CiOcoObject class comes first.

**2.1. CiOcoObject Class**

So, we need to come up with some software object responsible for control over two interconnected orders.

Traditionally, let's create a new object on the basis of [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject) abstract class.

This new class can look as follows:

```
//+------------------------------------------------------------------+
//| Class CiOcoObject                                                |
//| Purpose: a class for OCO orders                                  |
//+------------------------------------------------------------------+
class CiOcoObject : public CObject
  {
   //--- === Data members === ---
private:
   //--- tickets of pair
   ulong             m_order_tickets[2];
   //--- initialization flag
   bool              m_is_init;
   //--- id
   uint              m_id;

   //--- === Methods === ---
public:
   //--- constructor/destructor
   void              CiOcoObject(void){m_is_init=false;};
   void             ~CiOcoObject(void){};
   //--- copy constructor
   void              CiOcoObject(const CiOcoObject &_src_oco);
   //--- assignment operator
   void              operator=(const CiOcoObject &_src_oco);

   //--- initialization/deinitialization
   bool              Init(const SOrderProperties &_orders[],const uint _bunch_cnt=1);
   bool              Deinit(void);
   //--- get id
   uint              Id(void) const {return m_id;};

private:
   //--- types of orders
   ENUM_ORDER_TYPE   BaseOrderType(const ENUM_ORDER_TYPE _ord_type);
   ENUM_BASE_PENDING_TYPE PendingType(const ENUM_PENDING_ORDER_TYPE _pend_type);
   //--- set id
   void              Id(const uint _id){m_id=_id;};
  };
```

Each pair of OCO orders will have its own identifier. Its value is set by means of the generator of random numbers (object of CRandom class).

Methods of pair initialization and deinitialization are of concern in the context of interface. The first one creates (initializes) the pair, and the second one removes (deinitializes) it.

CiOcoObject::Init() method accepts array of structures of **SOrderProperties** type as argument. This type of structure represents properties of the order in the pair, i.e. OCO order.

**2.2 SOrderProperties Structure**

Let us consider fields of the aforementioned structure.

```
//+------------------------------------------------------------------+
//| Order properties structure                                       |
//+------------------------------------------------------------------+
struct SOrderProperties
  {
   double                  volume;           // order volume
   string                  symbol;           // symbol
   ENUM_PENDING_ORDER_TYPE order_type;       // order type
   uint                    price_offset;     // offset for execution price, points
   uint                    limit_offset;     // offset for limit price, points
   uint                    sl;               // stop loss, points
   uint                    tp;               // take profit, points
   ENUM_ORDER_TYPE_TIME    type_time;        // expiration type
   datetime                expiration;       // expiration
   string                  comment;          // comment
  }
```

So, to make the initialization method work we should previously fill the structures array consisting of two elements. In simple words, we need to explain the program which orders it will be placing.

Enumeration of **ENUM\_PENDING\_ORDER\_TYPE** type is used in the structure:

```
//+------------------------------------------------------------------+
//| Pending order type                                               |
//+------------------------------------------------------------------+
enum ENUM_PENDING_ORDER_TYPE
  {
   PENDING_ORDER_TYPE_BUY_LIMIT=2,       // Buy Limit
   PENDING_ORDER_TYPE_SELL_LIMIT=3,      // Sell Limit
   PENDING_ORDER_TYPE_BUY_STOP=4,        // Buy Stop
   PENDING_ORDER_TYPE_SELL_STOP=5,       // Sell Stop
   PENDING_ORDER_TYPE_BUY_STOP_LIMIT=6,  // Buy Stop Limit
   PENDING_ORDER_TYPE_SELL_STOP_LIMIT=7, // Sell Stop Limit
  };
```

Generally speaking, it looks the same as the [ENUM \_ORDER\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type) standard enumeration, but it allows selecting only pending orders, or, more truly, types of such orders.

It protects from errors when selecting the corresponding order type in the Input parameters (Fig.2).

![Fig. 2. The "Type" field with a drop-down list of available order types](https://c.mql5.com/2/17/en_fig2.png)

Fig. 2. The "Type" field with a drop-down list of available order types

If however we use [ENUM \_ORDER\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type) standard enumeration, then we could set a type of a market order (ORDER\_TYPE\_BUY or ORDER\_TYPE\_SELL), which is not required as we are dealing with pending orders only.

**2.3. Initialization of** **Pair**

As noted above, CiOcoObject::Init() method is engaged in order pair initialization.

In fact it places the order pair itself and records success or failure of new pair emergence. I should say that this is an active method, as it performs trading operations by itself. We can also create passive method as well. It will simply connect into a pair already active pending orders which have been placed independently.

I will not provide a code of the entire method. But I would like to note that it is important to calculate all prices (opening, stop, profit, limit), so the CTrade::OrderOpen() trade class method can perform a trade order. For that end we should consider two things: order direction (buy or sell) and position of an order execution price relative to a current price (above or under).

This method calls a couple of private methods: BaseOrderType() and PendingType(). The first one defines order direction, the second one determines pending order type.

If the order is placed, its ticket is recorded in the **m\_order\_tickets\[\]** array.

I used a simple Init\_OCO.mq5 script to test this method.

```
#property script_show_inputs
//---
#include "CiOcoObject.mqh"
//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
sinput string Info_order1="+===--Order 1--====+";   // +===--Order 1--====+
input ENUM_PENDING_ORDER_TYPE InpOrder1Type=PENDING_ORDER_TYPE_SELL_LIMIT; // Type
input double InpOrder1Volume=0.02;                  // Volume
input uint InpOrder1PriceOffset=125;                // Offset for execution price, points
input uint InpOrder1LimitOffset=50;                 // Offset for limit price, points
input uint InpOrder1SL=250;                         // Stop loss, points
input uint InpOrder1TP=455;                         // Profit, points
input string InpOrder1Comment="OCO Order 1";        // Comment
//---
sinput string Info_order2="+===--Order 2--====+";   // +===--Order 2--====+
input ENUM_PENDING_ORDER_TYPE InpOrder2Type=PENDING_ORDER_TYPE_SELL_STOP; // Type
input double InpOrder2Volume=0.04;                  // Volume
input uint InpOrder2PriceOffset=125;                // Offset for execution price, points
input uint InpOrder2LimitOffset=50;                 // Offset for limit price, points
input uint InpOrder2SL=275;                         // Stop loss, points
input uint InpOrder2TP=300;                         // Profit, points
input string InpOrder2Comment="OCO Order 2";        // Comment

//--- globals
CiOcoObject myOco;
SOrderProperties gOrdersProps[2];
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- property of the 1st order
   gOrdersProps[0].order_type=InpOrder1Type;
   gOrdersProps[0].volume=InpOrder1Volume;
   gOrdersProps[0].price_offset=InpOrder1PriceOffset;
   gOrdersProps[0].limit_offset=InpOrder1LimitOffset;
   gOrdersProps[0].sl=InpOrder1SL;
   gOrdersProps[0].tp=InpOrder1TP;
   gOrdersProps[0].comment=InpOrder1Comment;

//--- property of the 2nd order
   gOrdersProps[1].order_type=InpOrder2Type;
   gOrdersProps[1].volume=InpOrder2Volume;
   gOrdersProps[1].price_offset=InpOrder2PriceOffset;
   gOrdersProps[1].limit_offset=InpOrder2LimitOffset;
   gOrdersProps[1].sl=InpOrder2SL;
   gOrdersProps[1].tp=InpOrder2TP;
   gOrdersProps[1].comment=InpOrder2Comment;

//--- initialization of pair
   if(myOco.Init(gOrdersProps))
      PrintFormat("Id of new OCO pair: %I32u",myOco.Id());
   else
      Print("Error when placing OCO pair!");
  }
```

Here you can set various properties of future orders of the pair. MetaTrader 5 has six different types of pending orders.

With this context, there may be 15 variants ([combinations](https://www.mql5.com/en/code/1197)) of pairs (provided that there are different orders in the pair).

C(k,N) = C(2,6) = 15

All variants have been tested with the aid of the script. I'll give an example for **Buy Stop** \- **Buy Stop Limit** pair.

Types of orders should be specified in script parameters (Fig.3).

![Fig. 3. Pair of "Buy Stop" order with "Buy Stop Limit" order](https://c.mql5.com/2/17/en_fig3.png)

Fig. 3. Pair of "Buy Stop" order with "Buy Stop Limit" order

The following information will appear in the register "Experts":

```
QO      0       17:17:41.020    Init_OCO (GBPUSD.e,M15) Code of request result: 10009
JD      0       17:17:41.036    Init_OCO (GBPUSD.e,M15) New order ticket: 24190813
QL      0       17:17:41.286    Init_OCO (GBPUSD.e,M15) Code of request result: 10009
JH      0       17:17:41.286    Init_OCO (GBPUSD.e,M15) New order ticket: 24190814
MM      0       17:17:41.379    Init_OCO (GBPUSD.e,M15) Id of new OCO pair: 3782950319
```

But we cannot work with OCO orders to the utmost with the aid of the script without resorting to looping.

**2.4. Deinitialization of Pair**

This method is responsible for control over the order pair. The pair will "die" when any order leaves the list of active orders.

I suppose that this method should be placed in [OnTrade()](https://www.mql5.com/en/docs/basis/function/events#ontrade) or [OnTradeTransaction()](https://www.mql5.com/en/docs/basis/function/events#ontradetransaction) handlers of the EA's code. In such a manner, the EA will be able to process activation of any pair order without delay.

```
//+------------------------------------------------------------------+
//| Deinitialization of pair                                         |
//+------------------------------------------------------------------+
bool CiOcoObject::Deinit(void)
  {
//--- if pair is initialized
   if(this.m_is_init)
     {
      //--- check your orders
      for(int ord_idx=0;ord_idx<ArraySize(this.m_order_tickets);ord_idx++)
        {
         //--- current pair order
         ulong curr_ord_ticket=this.m_order_tickets[ord_idx];
         //--- another pair order
         int other_ord_idx=!ord_idx;
         ulong other_ord_ticket=this.m_order_tickets[other_ord_idx];

         //---
         COrderInfo order_obj;

         //--- if there is no current order
         if(!order_obj.Select(curr_ord_ticket))
           {
            PrintFormat("Order #%d is not found in active orders list.",curr_ord_ticket);
            //--- attempt to delete another order
            if(order_obj.Select(other_ord_ticket))
              {
               CTrade trade_obj;
               //---
               if(trade_obj.OrderDelete(other_ord_ticket))
                  return true;
              }
           }
        }
     }
//---
   return false;
  }
```

I'd like to mention one detail. Pair initialization flag is checked in the body of the class method. Attempt to check orders will not be made if flag is cleared. This approach prevents deleting one active order when another one has not been placed yet.

Let's add functionality to the script where a couple of orders have been placed. For this purpose, we will create Control\_OCO\_EA.mq5 test EA.

Generally speaking the EA will differ from the script only by the [Trade()](https://www.mql5.com/en/docs/runtime/event_fire#trade) event handling block in its code:

```
//+------------------------------------------------------------------+
//| Trade function                                                   |
//+------------------------------------------------------------------+
void OnTrade()
  {
//--- OCO pair deinitialization
   if(myOco.Deinit())
     {
      Print("No more order pair!");
      //--- clear  pair
      CiOcoObject new_oco;
      myOco=new_oco;
     }
  }
```

The video shows work of both programs in MetaTrader 5 terminal.

MQL5 Cookbook: ОСО Orders - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1582)

MQL5.community

1.91K subscribers

[MQL5 Cookbook: ОСО Orders](https://www.youtube.com/watch?v=jFFPgMtuayE)

MQL5.community

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

0:00 / 2:28

•Live

•

However both test programs have weaknesses.

The first program (script) can only actively create the pair but then it looses control over it.

The second program (Expert Advisor) though controls the pair, but it can't repeatedly create other pairs after creation of the first one. To make OCO order program (script) full-featured, we need to expand its toolset with the opportunity to place orders. We will do that in the next section.

### 3\. Controlling EA

Let us create OCO Order Management Panel on the chart for placing and setting parameters of pair orders.

It will be a part of the controlling EA (Fig.4). The source code is located in Panel\_OCO\_EA.mq5.

![Fig. 4. Panel for creation OCO orders: initial state](https://c.mql5.com/2/17/en_fig4.png)

Fig. 4. Panel for creation OCO orders: initial state

We should select a type of a future order and fill out the fields to place the pair of OCO orders.

Then the label on the only button on the panel will be changed (text property, Fig.5).

![Fig. 5. Panel for creation OCO orders: new pair](https://c.mql5.com/2/17/en_fig5.png)

Fig. 5. Panel for creation OCO orders: new pair

The following classes of the [Standard Library](https://www.mql5.com/en/docs/standardlibrary) were used to construct our Panel:

- CAppDialog is the main application dialog;
- CPanel is a rectangle label;
- CLabel is a text label;
- CComboBox is a field with a drop-down list;
- CEdit is an input field;
- CButton is a button.

Of course, parent class methods were called automatically.

Now we get down to the code. It has to be said that the part of the [Standard Library](https://www.mql5.com/en/docs/standardlibrary) which has been dedicated for creation of indication panels and dialogs is quite large.

For instance, if you want to catch a drop-down list closing event, you will have to delve deep into the stack of calls (Fig. 6).

![Fig. 6. Stack of Calls](https://c.mql5.com/2/17/en_fig6.png)

Fig. 6. Stack of Calls

A developer sets macros and a notation in %MQL5\\Include\\Controls\\Defines.mqh file for specific events.

I have created **ON\_OCO** custom event to create the OCO pair.

```
#define ON_OCO (101) // OCO pair creation event
```

Parameters of future orders are filled and the pair is generated in [OnChartEvent()](https://www.mql5.com/en/docs/basis/function/events#onchartevent) handler body.

```
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//--- handling all chart events by main dialog
   myDialog.ChartEvent(id,lparam,dparam,sparam);

//--- drop-down list handling
   if(id==CHARTEVENT_CUSTOM+ON_CHANGE)
     {
      //--- if it is Panel list
      if(!StringCompare(StringSubstr(sparam,0,7),"myCombo"))
        {
         static ENUM_PENDING_ORDER_TYPE prev_vals[2];
         //--- list index
         int combo_idx=(int)StringToInteger(StringSubstr(sparam,7,1))-1;

         ENUM_PENDING_ORDER_TYPE curr_val=(ENUM_PENDING_ORDER_TYPE)(myCombos[combo_idx].Value()+2);
         //--- remember order type change
         if(prev_vals[combo_idx]!=curr_val)
           {
            prev_vals[combo_idx]=curr_val;
            gOrdersProps[combo_idx].order_type=curr_val;
           }
        }
     }

//--- handling input fields
   else if(id==CHARTEVENT_OBJECT_ENDEDIT)
     {
      //--- if it is Panel's input field
      if(!StringCompare(StringSubstr(sparam,0,6),"myEdit"))
        {
         //--- find object
         for(int idx=0;idx<ArraySize(myEdits);idx++)
           {
            string curr_edit_obj_name=myEdits[idx].Name();
            long curr_edit_obj_id=myEdits[idx].Id();
            //--- if names coincide
            if(!StringCompare(sparam,curr_edit_obj_name))
              {
               //--- get current value of field
               double value=StringToDouble(myEdits[idx].Text());
               //--- define gOrdersProps[] array index
               int order_num=(idx<gEditsHalfLen)?0:1;
               //--- define gOrdersProps structure field number
               int jdx=idx;
               if(order_num)
                  jdx=idx-gEditsHalfLen;
               //--- fill up gOrdersProps structure field
               switch(jdx)
                 {
                  case 0: // volume
                    {
                     gOrdersProps[order_num].volume=value;
                     break;
                    }
                  case 1: // execution
                    {
                     gOrdersProps[order_num].price_offset=(uint)value;
                     break;
                    }
                  case 2: // limit
                    {
                     gOrdersProps[order_num].limit_offset=(uint)value;
                     break;
                    }
                  case 3: // stop
                    {
                     gOrdersProps[order_num].sl=(uint)value;
                     break;
                    }
                  case 4: // profit
                    {
                     gOrdersProps[order_num].tp=(uint)value;
                     break;
                    }
                 }
              }
           }
         //--- OCO pair creation flag
         bool is_to_fire_oco=true;
         //--- check structure filling
         for(int idx=0;idx<ArraySize(gOrdersProps);idx++)
           {
            //---  if order type is set
            if(gOrdersProps[idx].order_type!=WRONG_VALUE)
               //---  if volume is set
               if(gOrdersProps[idx].volume!=WRONG_VALUE)
                  //---  if offset for execution price is set
                  if(gOrdersProps[idx].price_offset!=(uint)WRONG_VALUE)
                     //---  if offset for limit price is set
                     if(gOrdersProps[idx].limit_offset!=(uint)WRONG_VALUE)
                        //---  if stop loss is set
                        if(gOrdersProps[idx].sl!=(uint)WRONG_VALUE)
                           //---  if take profit is set
                           if(gOrdersProps[idx].tp!=(uint)WRONG_VALUE)
                              continue;

            //--- clear OCO pair creation flag
            is_to_fire_oco=false;
            break;
           }
         //--- create OCO pair?
         if(is_to_fire_oco)
           {
            //--- complete comment fields
            for(int ord_idx=0;ord_idx<ArraySize(gOrdersProps);ord_idx++)
               gOrdersProps[ord_idx].comment=StringFormat("OCO Order %d",ord_idx+1);
            //--- change button properties
            myButton.Text("New pair");
            myButton.Color(clrDarkBlue);
            myButton.ColorBackground(clrLightBlue);
            //--- respond to user actions
            myButton.Enable();
           }
        }
     }
//--- handling click on button
   else if(id==CHARTEVENT_OBJECT_CLICK)
     {
      //--- if it is OCO pair creation button
      if(!StringCompare(StringSubstr(sparam,0,6),"myFire"))
         //--- if to respond to user actions
         if(myButton.IsEnabled())
           {
            //--- generate OCO pair creation event
            EventChartCustom(0,ON_OCO,0,0.0,"OCO_fire");
            Print("Command to create new bunch has been received.");
           }
     }
//--- handling new pair initialization command
   else if(id==CHARTEVENT_CUSTOM+ON_OCO)
     {
      //--- OCO pair initialization
      if(gOco.Init(gOrdersProps,gOcoList.Total()+1))
        {
         PrintFormat("Id of new OCO pair: %I32u",gOco.Id());
         //--- copy pair
         CiOcoObject *ptr_new_oco=new CiOcoObject(gOco);
         if(CheckPointer(ptr_new_oco)==POINTER_DYNAMIC)
           {
            //--- add to list
            int node_idx=gOcoList.Add(ptr_new_oco);
            if(node_idx>-1)
               PrintFormat("Total number of bunch: %d",gOcoList.Total());
            else
               PrintFormat("Error when adding OCO pair %I32u to list!",gOco.Id());
           }
        }
      else
         Print("OCO-orders placing error!");

      //--- clear properties
      Reset();
     }
  }
```

Handler code isn't small. I would like to lay emphasis on several blocks.

First handling of all chart events is given to the main dialog.

Next are blocks of various events handling:

- Changing drop-down lists for defining an order type;
- Editing input fields for filling up properties of orders;
- Click on button for **ON\_OCO** event generation;
- **ON\_OCO** event response: order pair creation.

The EA does not verify the correctness of filling the panel's fields. That is why we have to check values by ourselves, otherwise the EA will show OCO orders placing error.

Necessity to remove the pair and close the remaining order is checked in [OnTrade()](https://www.mql5.com/en/docs/basis/function/events#ontrade) handler body.

### Conclusion

I tried to demonstrate the riches of the [Standard Library](https://www.mql5.com/en/docs/standardlibrary) classes which can be used for fulfillment of some specific tasks.

Particularly, we were dealing with a problem of OCO orders handling. I hope that the code of the EA with Panel for OCO orders handling will be a starting point for creation of more complicated order pairs.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1582](https://www.mql5.com/ru/articles/1582)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1582.zip "Download all attachments in the single ZIP archive")

[ciocoobject.mqh](https://www.mql5.com/en/articles/download/1582/ciocoobject.mqh "Download ciocoobject.mqh")(27.71 KB)

[crandom.mqh](https://www.mql5.com/en/articles/download/1582/crandom.mqh "Download crandom.mqh")(5.19 KB)

[init\_oco.mq5](https://www.mql5.com/en/articles/download/1582/init_oco.mq5 "Download init_oco.mq5")(6.42 KB)

[control\_oco\_ea.mq5](https://www.mql5.com/en/articles/download/1582/control_oco_ea.mq5 "Download control_oco_ea.mq5")(7.88 KB)

[panel\_oco\_ea.mq5](https://www.mql5.com/en/articles/download/1582/panel_oco_ea.mq5 "Download panel_oco_ea.mq5")(30.25 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Cookbook — Macroeconomic events database](https://www.mql5.com/en/articles/11977)
- [MQL5 Cookbook — Services](https://www.mql5.com/en/articles/11826)
- [MQL5 Cookbook – Economic Calendar](https://www.mql5.com/en/articles/9874)
- [MQL5 Cookbook: Trading strategy stress testing using custom symbols](https://www.mql5.com/en/articles/7166)
- [MQL5 Cookbook: Getting properties of an open hedge position](https://www.mql5.com/en/articles/4830)
- [MQL5 Cookbook - Pivot trading signals](https://www.mql5.com/en/articles/2853)
- [MQL5 Cookbook - Trading signals of moving channels](https://www.mql5.com/en/articles/1863)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/40751)**
(13)


![vijanda](https://c.mql5.com/avatar/avatar_na2.png)

**[vijanda](https://www.mql5.com/en/users/vijanda)**
\|
7 Feb 2019 at 21:27

I just downloaded all the zip files but i need help with the instraction on  how to make them work or install

![Denis Kirichenko](https://c.mql5.com/avatar/2019/5/5CEDB8D2-7CB7.jpg)

**[Denis Kirichenko](https://www.mql5.com/en/users/denkir)**
\|
7 Feb 2019 at 22:37

**vijanda:**

_I just downloaded all the zip files but i need help with the instraction on  how to make them work or install_

You have to create a folder where all the relevant files will reside. After creation just copy the files into the folder.  For example:

![oco_files](https://c.mql5.com/3/267/oco_files.png)

After compilation you will find the expert file in the MT5 Navigator.

![oco_files_mt5](https://c.mql5.com/3/267/oco_files_mt5.png)

Much time has elapsed since the article publication. But the code runs fine. Build 1981.

![vijanda](https://c.mql5.com/avatar/avatar_na2.png)

**[vijanda](https://www.mql5.com/en/users/vijanda)**
\|
8 Feb 2019 at 15:06

I followed all the instructions but the OCO Folder is not showing on my MT5 Navigator I even tried refreshing it

![Cristian Mateo Duque Ocampo](https://c.mql5.com/avatar/avatar_na2.png)

**[Cristian Mateo Duque Ocampo](https://www.mql5.com/en/users/ekud87)**
\|
18 Sep 2025 at 20:58

I got it to work correctly. The code compiles without any errors or warnings.

Steps to compile it:

01. Download the source code from the article.
02. Extract all contents to a new folder (in my case, I called it "OCO EA")
03. Open MT5
04. Open the MT5 data folder
05. Open the "Experts" folder
06. In the "Experts" folder, paste the "OCO EA" folder
07. Open the MQL5 IDE (you can open it directly from MT5 by pressing F4 or from the Tools -> MetaQuotes Language Editor menu)
08. Once the IDE is open, open each of the files in the "OCO EA" folder, which should be contained in the "Experts" folder. You must compile each file.
09. Open MT5, and the new "OCO EA" folder should appear in the "Experts" section in the "Navigator."
10. Drag the "panel\_oco\_ea" file onto a chart.
11. Enjoy!

\-\-\----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

I got it to work correctly. The code compiles without any errors or warnings.

Steps to compile it:

01. Download the source code contained in the article.
02. Extract all the content to a new folder (in my case call the folder "OCO EA").
03. Open MT5
04. Open the MT5 data folder
05. Open the "Experts" folder
06. In the "Experts" folder paste the folder "OCO EA".
07. Open the MQL5 IDE (you can open it directly from MT5 by pressing F4 or in the menu tools -> MetaQuotes Language Editor)

08. Once the IDE is open open each of the files in the "OCO EA" folder which must be contained in the "Experts" folder, you must compile each file.
09. Open MT5 and in the "Navigator" the new "OCO EA" folder should appear in  the "Experts" section.
10. Drag the file "panel\_oco\_ea" to a chart.
11. Enjoy.

screenshot of the compilation:

[![](https://c.mql5.com/3/474/3203939629267__1.png)](https://c.mql5.com/3/474/3203939629267.png "https://c.mql5.com/3/474/3203939629267.png")

if you do it right you should be able to see something like this in your MT5:

[![](https://c.mql5.com/3/474/3134257103030__1.png)](https://c.mql5.com/3/474/3134257103030.png "https://c.mql5.com/3/474/3134257103030.png")

![Cristian Mateo Duque Ocampo](https://c.mql5.com/avatar/avatar_na2.png)

**[Cristian Mateo Duque Ocampo](https://www.mql5.com/en/users/ekud87)**
\|
18 Sep 2025 at 21:01

**vijanda [#](https://www.mql5.com/es/forum/64251/page2#comment_57165398) :**

I followed all the instructions, but the OCO folder is not showing in my MT5 Navigator even tried to update it.

Check the step by step.


![Principles of Exchange Pricing through the Example of Moscow Exchange's Derivatives Market](https://c.mql5.com/2/12/MOEX.png)[Principles of Exchange Pricing through the Example of Moscow Exchange's Derivatives Market](https://www.mql5.com/en/articles/1284)

This article describes the theory of exchange pricing and clearing specifics of Moscow Exchange's Derivatives Market. This is a comprehensive article for beginners who want to get their first exchange experience on derivatives trading, as well as for experienced forex traders who are considering trading on a centralized exchange platform.

![Building an Interactive Application to Display RSS Feeds in MetaTrader 5](https://c.mql5.com/2/17/RSS_Feed_MetaTrader5__1.png)[Building an Interactive Application to Display RSS Feeds in MetaTrader 5](https://www.mql5.com/en/articles/1589)

In this article we look at the possibility of creating an application for the display of RSS feeds. The article will show how aspects of the Standard Library can be used to create interactive programs for MetaTrader 5.

![Bi-Directional Trading and Hedging of Positions in MetaTrader 5 Using the HedgeTerminal Panel, Part 1](https://c.mql5.com/2/17/HedgeTerminalaArticle200x200_2.png)[Bi-Directional Trading and Hedging of Positions in MetaTrader 5 Using the HedgeTerminal Panel, Part 1](https://www.mql5.com/en/articles/1297)

This article describes a new approach to hedging of positions and draws the line in the debates between users of MetaTrader 4 and MetaTrader 5 about this matter. The algorithms making such hedging reliable are described in layman's terms and illustrated with simple charts and diagrams. This article is dedicated to the new panel HedgeTerminal, which is essentially a fully featured trading terminal within MetaTrader 5. Using HedgeTerminal and the virtualization of the trade it offers, positions can be managed in the way similar to MetaTrader 4.

![Trader's Statistical Cookbook: Hypotheses](https://c.mql5.com/2/12/Trader_Statistics_Recipes_MetaTrader5_Alglib_MQL5__1.png)[Trader's Statistical Cookbook: Hypotheses](https://www.mql5.com/en/articles/1240)

This article considers hypothesis - one of the basic ideas of mathematical statistics. Various hypotheses are examined and verified through examples using methods of mathematical statistics. The actual data is generalized using nonparametric methods. The Statistica package and the ported ALGLIB MQL5 numerical analysis library are used for processing data.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=dtuapslgoiwchjvwloeharnzyijdsgcz&ssn=1769181556662257430&ssn_dr=0&ssn_sr=0&fv_date=1769181556&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1582&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Cookbook%3A%20%D0%9E%D0%A1%D0%9E%20Orders%20-%20MQL5%20Articles&scr_res=1920x1080&ac=1769181556884670&fz_uniq=5069341516041749364&sv=2552)

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